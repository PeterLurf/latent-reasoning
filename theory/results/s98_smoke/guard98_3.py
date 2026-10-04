"""Note 98-3 guard test: overfit a fixed batch of 8 problems for 50 steps (Adam lr 1e-3, fp32, CPU), training every
test-time chunk with true starts, then greedy-decode those 8 problems with the k = 4 rollout. The final slot's step
text AND the answer must decode exactly. Run for the fixed boxes/mult98.py and the pre-fix version (two forks at
the final slot from the same state), for genm and mask, 4x4 and 5x5.
    python theory/results/s98_smoke/guard98_3.py <old_mult98.py>
"""
import importlib.util
import os
import sys

os.environ["BOXES_NB"] = "5"
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from boxes import s5  # noqa: E402,F401


def mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def run(M, arm, task, steps=150, nb=8, dev="cpu"):
    torch.manual_seed(0); D = M.load(f"D:/Latent/bench/data/mult_{task}_test.txt", limit=nb); n = D["n"]
    m = M.MultGen(s5.load_gpt2(dev), arm).to(dev); opt = torch.optim.Adam(m.parameters(), lr=3e-4); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    pre = T(np.concatenate([M.DIGIT_IDS[D["A"]], np.full((nb, 1), M.TOK["*"])], 1))
    for _ in range(steps):
        opt.zero_grad(); tot = 0
        for c0, c1 in M.chunks_of(n):
            st = T(D["steps"][c0 - 1]) if c0 > 0 else None
            loss, _, _, _ = m.chunk(pre, st, T(D["B"][:, c0:c1]), step_tgt=[T(D["steps"][j]) for j in range(c0, c1)],
                                    ans_tgt=T(D["ans"]) if c1 == n else None)
            loss.backward(); tot += loss.item()
        opt.step()
    m.eval(); per, ok, _ = M.gen_rollout(m, D, dev, None, nb)
    return per[:, -1].mean(), ok.mean(), tot


if __name__ == "__main__":
    torch.set_num_threads(4); DEV = sys.argv[2] if len(sys.argv) > 2 else "cpu"   # fp32 on any device
    new, old = mod("D:/Latent/boxes/mult98.py", "m98_new"), mod(sys.argv[1], "m98_old")
    for label, M in (("FIXED", new), ("OLD", old)):
        for arm in ("genm", "mask"):
            for task in ("4x4", "5x5"):
                last, ans, loss = run(M, arm, task, dev=DEV)
                verdict = "PASS" if last == 1 and ans == 1 else "FAIL"
                print(f"{label:5s} {arm:4s} {task}: final-slot step text exact {last:.3f}, answer exact {ans:.3f}, final train loss {loss:.3f} -> {verdict}", flush=True)
