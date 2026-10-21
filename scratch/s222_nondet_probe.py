"""§222 engine-equivalence follow-up (10-21): which op makes t2 probe training non-bitwise? Builds the untrained model and a small pool, trains one
lin arm at t = 0, 1, 2 for 3 steps each under torch.use_deterministic_algorithms(True, warn_only=True) (warnings name non-deterministic kernels), and runs each
arm twice from the same state to report whether its params agree bitwise. Scratch diagnostic only; nothing under gate_s222 or boxes is written.
    python -B scratch/s222_nondet_probe.py --device cuda:1"""
import sys
import warnings

sys.path.insert(0, "D:/Latent/scratch")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import s222_cert as C  # noqa: E402

E = C.E


def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:1"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True); warnings.simplefilter("always")
    m, _ = E.untrained(dev, 16); held, pool = C.build_pool(2200, 32); dirs0 = C.plant_dirs_seeded(m, pool, [0, 1, 2], dev, 16, 22750); dirs = {t: C.scaled(dirs0[t], 3.0) for t in dirs0}
    for t in (0, 1, 2):
        out = []
        for rep in (0, 1):
            x = C.Arm(m, "lin", t, dev); rb = np.random.default_rng(22701)
            with warnings.catch_warnings(record=True) as W:
                warnings.simplefilter("always")
                for _ in range(3):
                    ix = rb.integers(0, len(pool), 64); sub = [pool[j] for j in ix]; b, S = E.stored(m, sub, t, dev, 16)
                    y, args = x.draw_train(64); x.pr.opt.zero_grad(set_to_none=True)
                    for q0 in range(0, 64, 16):
                        sA, sB = x.scores(m, sub[q0:q0 + 16], tuple(v[q0:q0 + 16] for v in b), [s_[q0:q0 + 16] for s_ in S], C.sl(args, q0, q0 + 16), dirs[t])
                        (F.cross_entropy(torch.stack([sA, sB], 1), y[q0:q0 + 16]) * 0.25).backward()
                    x.pr.opt.step()
            out.append({k: v.detach().cpu().clone() for k, v in x.pr.P.items()})
            if rep == 0:
                msgs = sorted({str(w.message).split(".")[0][:160] for w in W})
                print("t", t, "non-deterministic-kernel warnings:", msgs if msgs else "none", flush=True)
        same = all(torch.equal(out[0][k], out[1][k]) for k in out[0])
        print("t", t, "two runs from the same state bitwise equal:", same, flush=True)


if __name__ == "__main__":
    main()
