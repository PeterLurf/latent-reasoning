"""§140 post-hoc accumulator probe (Theory 2026-10-07, descriptive, $0): where does the dropped DIGPAIR 4x4 model lose the accumulator?
On boxes/runs_fsq/<run>/model.pt (multpair), its own hard chains over the 1,000 test items (all n^2 + 1 steps), the TRUE accumulator after
each step (multpair.accumulators: carry-in + the column's products so far), probed linearly from the carried code_t and from h_t (before
W_down) with ridge (lambda by 5-fold CV on the fit half; fit on half the problems, scored on the other half):
  by_position: steps grouped by their pair's position within its column (1st / 2nd / 3rd / 4th pair; the flush separately) - one fit per group;
  by_magnitude: one fit over all pair steps, its held-out R^2 and RMSE within true-accumulator bins < 100, 100-200, > 200.
    python scratch/probe140_acc.py boxes/runs_fsq/s140_pair_4x4_s1 [device]   -> <run>/eval_probe_acc.json"""
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch"))
os.environ.setdefault("BOXES_NB", "5")
try:
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
import boxes.fsq120 as F  # noqa: E402
import boxes.multcol as MC  # noqa: E402
import boxes.multpair as MP  # noqa: E402
from boxes import s5  # noqa: E402
from probe137_percol import ridge_cv_r2  # noqa: E402


def main():
    d = sys.argv[1].rstrip("/"); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; a = json.load(open(os.path.join(d, "args.json")))
    F.D_Q, F.LEV = int(a.get("d_q", 8)), int(a.get("levels", 8)); F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF); memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    m = MP.MultPair(s5.load_gpt2(dev), MP.PairFam()).to(dev); m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
    A, B = MC.load_bench(os.path.join(ROOT, "bench", "data", "mult_%s_test.txt" % a["task"]))
    if len(sys.argv) > 3:                                                   # smoke only
        A, B = A[:int(sys.argv[3])], B[:int(sys.argv[3])]
    N = len(A); h = N // 2; acc, _, _, pairs, col, emit = MP.accumulators(A, B); _, Q, H = MP.rollout(m, A, B, dev, amp, 250, record=True)
    pos = np.zeros(len(pairs), int)                                         # 1-based position of each pair within its column; 0 = the flush
    for t in range(len(pairs)):
        pos[t] = 0 if pairs[t][0] < 0 else (1 if t == 0 or col[t] != col[t - 1] else pos[t - 1] + 1)
    res = {"run": os.path.basename(d), "n": N, "fit": h, "steps": int(len(pairs)), "by_position": {}, "by_magnitude": {}}
    for p in sorted(set(pos.tolist())):
        ts = np.where(pos == p)[0]; key = "flush" if p == 0 else str(p); y = acc[:, ts]; r = {"n_steps": int(len(ts))}
        for src, X in (("code", Q), ("h", H)):
            Xs = X[:, ts]; Xf, Xt = Xs[:h].reshape(h * len(ts), -1), Xs[h:].reshape((N - h) * len(ts), -1)
            r["r2_%s" % src], r["lam_%s" % src] = ridge_cv_r2(Xf, y[:h].reshape(-1).astype(float), Xt, y[h:].reshape(-1).astype(float))
        res["by_position"][key] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    tp = np.where(pos > 0)[0]; yf, yt = acc[:h, tp].reshape(-1).astype(float), acc[h:, tp].reshape(-1).astype(float)
    for src, X in (("code", Q), ("h", H)):
        Xf, Xt = X[:h, tp].reshape(len(yf), -1), X[h:, tp].reshape(len(yt), -1)
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; _, lam = ridge_cv_r2(Xf, yf, Xt, yt)
        Z, Zt = np.c_[(Xf - mu) / sd, np.ones(len(Xf))], np.c_[(Xt - mu) / sd, np.ones(len(Xt))]
        w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ yf); pr = Zt @ w
        for lo, hi, key in ((0, 100, "lt100"), (100, 200, "100_200"), (200, 10 ** 9, "gt200")):
            mk = (yt >= lo) & (yt < hi)
            if mk.sum() > 1:
                res["by_magnitude"].setdefault(key, {})["r2_%s" % src] = round(float(1 - ((yt[mk] - pr[mk]) ** 2).sum() / max(((yt[mk] - yt[mk].mean()) ** 2).sum(), 1e-12)), 4)
                res["by_magnitude"][key]["rmse_%s" % src] = round(float(np.sqrt(((yt[mk] - pr[mk]) ** 2).mean())), 3); res["by_magnitude"][key]["n"] = int(mk.sum())
    res.update({"eval_dtype": str(amp), "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"})
    json.dump(res, open(os.path.join(d, "eval_probe_acc.json"), "w", newline=chr(10)), indent=1); print("PROBE_ACC", json.dumps(res["by_position"]), json.dumps(res["by_magnitude"]), flush=True)


if __name__ == "__main__":
    main()
