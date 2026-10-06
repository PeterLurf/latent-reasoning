"""Plain-path smoke (Audit 2026-10-06, after the 5387d85e FSQ120.__init__ break): CPU, seconds. Constructs FSQ120 (s5, ff, one held-out
family) and Table132 with DEFAULT settings (no new flags), checks the attributes every code path reads, runs one training step (chain
loss -> backward -> AdamW step) and one eval forward (run_test / rollout) on tiny shapes. Exit 1 on any failure.
Run before any commit touching boxes/fsq120.py, boxes/table132.py or boxes/core.py, and before every deploy pack (vdeploy_toy.sh).
    python scratch/smoke_plain.py"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(3)
ATTRS = ("p_hard", "no_round", "noise", "soft_beta", "carrier_ln", "tsd_on", "std", "raw_h", "anneal", "lam", "skip", "std2", "var_on", "det_train")


def main():
    t0 = time.time(); from boxes import s5; from boxes import fsq120 as F; from boxes import table132 as T
    gpt = lambda: s5.load_gpt2("cpu")  # noqa: E731
    ok = True
    for fam_name in ("s5", "ff", "cyc"):
        fam = F.Family(fam_name); m = F.FSQ120(gpt(), fam); miss = [a for a in ATTRS if not hasattr(m, a)]
        if miss:
            print("FAIL FSQ120(%s) missing attributes: %s" % (fam_name, miss)); ok = False; continue
        opt = F.make_opt(m, 1e-4); m.train(); m.p_hard = 1.0; s0, x, lab = F.make_batch(fam, np.random.default_rng(0), 2, 3)
        loss, _, _, _ = m.chain(m.start(2, "cpu", s0 if fam_name == "s5" else None), torch.tensor(x), lab)
        opt.zero_grad(); loss.backward(); opt.step(); m.eval()
        X = np.random.default_rng(1).integers(0, fam.n_items, (2, 4))
        pred, codes = F.run_test(m, fam, np.zeros(2, np.int64) if fam_name == "s5" else None, X, "cpu", None, 2)
        print("ok FSQ120(%s): train loss %.3f, eval pred %s" % (fam_name, float(loss), pred.shape), flush=True)
    sp = T.Splits(4, 2); tf = T.TableFam(4, 2); m = T.Table132(gpt(), tf); miss = [a for a in ATTRS if not hasattr(m, a)]
    if miss:
        print("FAIL Table132 missing attributes: %s" % miss); ok = False
    else:
        opt = F.make_opt(m, 1e-4); m.train(); m.p_hard = 1.0; c, x, lab = T.problems(np.random.default_rng(0), sp.draw_train(np.random.default_rng(0), 2), 3, 2)
        m.ctx = torch.tensor(c); loss, _, _, _ = m.chain(m.start(2, "cpu"), torch.tensor(x), lab); opt.zero_grad(); loss.backward(); opt.step(); m.eval()
        pred = T.rollout(m, c, x, "cpu", None, 2); print("ok Table132: train loss %.3f, rollout %s" % (float(loss), pred.shape), flush=True)
    print("SMOKE_PLAIN", "PASS" if ok else "FAIL", "(%.0f s)" % (time.time() - t0), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
