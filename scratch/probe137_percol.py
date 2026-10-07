"""§137 post-hoc per-column probe (Theory 2026-10-07, descriptive, $0): where is the carry lost in the dropped 4x4 proof model?
On boxes/runs_fsq/<run>/model.pt (multcol), the model's own hard chains over the 1,000 test items (rollout, eval mode); per column
k = 1..8, from BOTH h_k (the segment's last hidden, before W_down) and code_k (the carried FSQ code):
  carry_out  (the true carry out of column k)            ridge R^2
  carry_in   (the true carry into column k; 0 at k = 1)  ridge R^2
  colsum     (context-only sum_{i+j=k} a_i b_j, no carry) ridge R^2
  digit      (the product digit p_k; 10 classes)         multinomial-logistic accuracy (+ the majority-class rate)
Fit on the first half of the problems, scored on the second (ridge as multcol.ridge_r2; logistic: full-batch Adam on standardised features).
h high / code low -> the W_down / FSQ bottleneck loses it; h low -> the segment does not compute it.
    python scratch/probe137_percol.py boxes/runs_fsq/s137_col_4x4_s1 [device]  -> <run>/eval_probe_percol.json"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
try:
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
import boxes.fsq120 as F  # noqa: E402
import boxes.multcol as MC  # noqa: E402
from boxes import s5  # noqa: E402


def logit_acc(Xf, yf, Xs, ys, dev, steps=500, lr=0.05, wd=1e-4):
    torch.manual_seed(0); Xf = torch.tensor(Xf, dtype=torch.float32, device=dev); Xs = torch.tensor(Xs, dtype=torch.float32, device=dev)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Xf, Xs = (Xf - mu) / sd, (Xs - mu) / sd; y = torch.tensor(yf, device=dev)
    W = torch.zeros(Xf.shape[1], 10, device=dev, requires_grad=True); b = torch.zeros(10, device=dev, requires_grad=True); opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); Fn.cross_entropy(Xf @ W + b, y).backward(); opt.step()
    with torch.no_grad():
        return float(((Xs @ W + b).argmax(-1).cpu().numpy() == ys).mean())


def main():
    d = sys.argv[1].rstrip("/"); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; a = json.load(open(os.path.join(d, "args.json")))
    F.D_Q, F.LEV = int(a.get("d_q", 8)), int(a.get("levels", 8)); F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF); memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    m = MC.MultCol(s5.load_gpt2(dev), MC.MultFam()).to(dev); m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
    A, B = MC.load_bench(os.path.join(ROOT, "bench", "data", "mult_%s_test.txt" % a["task"])); P, C, _ = MC.columns(A, B); L = P.shape[1]; N = len(A); h = N // 2
    colsum = np.zeros((N, L), np.int64)
    for k in range(L):
        for i in range(max(0, k - B.shape[1] + 1), min(A.shape[1], k + 1)):
            colsum[:, k] += A[:, i] * B[:, k - i]
    cin = np.c_[np.zeros(N, np.int64), C[:, :-1]]
    pred, Q, H = MC.rollout(m, A, B, dev, amp, 250, record=True); res = {"run": os.path.basename(d), "n": N, "fit": h, "test_acc_own_chain": float((pred == P).all(1).mean()), "by_column": {}}
    for k in range(L):
        r = {}
        for src, X in (("h", H[:, k]), ("code", Q[:, k])):
            for name, y in (("carry_out", C[:, k]), ("carry_in", cin[:, k]), ("colsum", colsum[:, k])):
                r["%s_r2_%s" % (name, src)] = MC.ridge_r2(X[:h], y[:h].astype(float), X[h:], y[h:].astype(float)) if len(np.unique(y)) > 1 else None
            r["digit_acc_%s" % src] = logit_acc(X[:h], P[:h, k], X[h:], P[h:, k], dev)
        r["digit_majority_rate"] = float(np.bincount(P[h:, k], minlength=10).max() / (N - h)); r["model_digit_acc"] = float((pred[:, k] == P[:, k]).mean())
        r["n_carry_values"] = int(len(np.unique(C[:, k]))); res["by_column"][str(k + 1)] = {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}
        print(k + 1, res["by_column"][str(k + 1)], flush=True)
    res.update({"eval_dtype": str(amp), "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"})
    json.dump(res, open(os.path.join(d, "eval_probe_percol.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
