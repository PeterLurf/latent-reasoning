"""THEORY proposal #4 decider NORM-CLAMP (descriptive): s175_auxk8_s1, the same 256 items as s175_memnorm.py (n 3..8, rng 17501), M_0 = 0, t = 1..16.
Clamped arm: for every t > 8, after each write the additive part S[:, :h_add] is rescaled per item to that item's own norm at t = 8 (direction kept);
the delta part untouched. Reports pooled accuracy per t (clamped / unclamped, both computed here) and accuracy by n at t = 16.
    python scratch/s175_normclamp.py [--device cpu]   -> boxes/runs_fsq/s175_auxk8_s1/eval_normclamp.json"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from boxes.halt173 import load_model  # noqa: E402
from boxes.nested170 import test_set, NUM_IDS  # noqa: E402


@torch.no_grad()
def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3"))); t0 = time.time()
    run = "boxes/runs_fsq/s175_auxk8_s1"; kind, m, a = load_model(run, dev); assert kind == "nested"
    pool = [it for it in test_set(a.get("per_n", 1000)) if 3 <= it["n"] <= 8]; its = [pool[i] for i in np.random.default_rng(17501).permutation(len(pool))[:256]]
    tgt = np.array([it["t"] for it in its]); nn = np.array([it["n"] for it in its]); ha = m.mem.h_add; ok = {"clamped": {}, "unclamped": {}}; add_n = {}
    for i in range(0, len(its), 64):
        ch = its[i:i + 64]; x, msk = m.think_inputs(ch, dev)
        for arm in ("unclamped", "clamped"):
            S = m.mem.zero_state(len(ch), dev); n8 = None
            for t in range(1, 17):
                S = m.write(S, x, msk)
                if t == 8:
                    n8 = S[:, :ha].float().flatten(1).norm(dim=1)
                if arm == "clamped" and t > 8:
                    cur = S[:, :ha].float().flatten(1).norm(dim=1).clamp_min(1e-12); f = (n8 / cur).to(S.dtype)[:, None, None, None]
                    S = torch.cat([S[:, :ha] * f, S[:, ha:]], 1)
                pr = m.dec_logits(S, ch, dev)[:, NUM_IDS].argmax(-1).cpu().numpy(); ok[arm].setdefault(t, []).append(pr == tgt[i:i + len(ch)])
                if arm == "clamped":
                    add_n.setdefault(t, []).append(S[:, :ha].float().flatten(1).norm(dim=1).cpu().numpy())
    res = {"by_t": {str(t): {"acc_clamped": float(np.concatenate(ok["clamped"][t]).mean()), "acc_unclamped": float(np.concatenate(ok["unclamped"][t]).mean()),
                             "add_norm_median_clamped": float(np.median(np.concatenate(add_n[t])))} for t in range(1, 17)},
           "acc_by_n_t16": {arm: {str(n): float(np.concatenate(ok[arm][16])[nn == n].mean()) for n in range(3, 9)} for arm in ("clamped", "unclamped")},
           "run": run, "n_items": len(its), "items": "as eval_memnorm.json (n 3..8, rng 17501 permutation[:256])", "device": dev, "sec": round(time.time() - t0),
           "clamp": "t > 8: after each write, S[:, :h_add] rescaled per item to its own norm at t = 8 (direction kept); delta part untouched (Theory proposal #4 decider)"}
    json.dump(res, open(os.path.join(run, "eval_normclamp.json"), "w", newline="\n"), indent=1)
    for t, v in res["by_t"].items():
        print(t, round(v["acc_clamped"], 3), round(v["acc_unclamped"], 3), flush=True)
    print("by n t16", res["acc_by_n_t16"], flush=True)


if __name__ == "__main__":
    main()
