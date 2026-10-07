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


def ridge_cv_r2(Xf, yf, Xs, ys, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), folds=5):
    """Ridge with lambda (x n, standardised features) chosen by 5-fold CV on the fit half - for h (768 features vs ~500 fit rows the fixed
    1e-3 penalty overfits: negative held-out R^2). Returns (R^2 on the held-out half, the chosen lambda)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Z = (Xf - mu) / sd; Zs = (Xs - mu) / sd; idx = np.arange(len(Z)) % folds
    def fit(Za, ya, lam):
        Za1 = np.c_[Za, np.ones(len(Za))]; return np.linalg.solve(Za1.T @ Za1 + lam * len(Za) * np.eye(Za1.shape[1]), Za1.T @ ya)
    def r2(w, Zb, yb):
        r = yb - np.c_[Zb, np.ones(len(Zb))] @ w; return 1 - (r ** 2).sum() / max(((yb - yb.mean()) ** 2).sum(), 1e-12)
    best = max(grid, key=lambda lam: np.mean([r2(fit(Z[idx != f], yf[idx != f], lam), Z[idx == f], yf[idx == f]) for f in range(folds)]))
    return float(r2(fit(Z, yf, best), Zs, ys)), best


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
    A, B = MC.load_bench(os.path.join(ROOT, "bench", "data", "mult_%s_test.txt" % a["task"]))
    if len(sys.argv) > 3:                                                   # smoke only: the first N test items
        A, B = A[:int(sys.argv[3])], B[:int(sys.argv[3])]
    P, C, _ = MC.columns(A, B); L = P.shape[1]; N = len(A); h = N // 2
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
                if len(np.unique(y)) > 1:                                   # v2 (2026-10-07): lambda by 5-fold CV on the fit half
                    r["%s_r2cv_%s" % (name, src)], r["%s_lamcv_%s" % (name, src)] = ridge_cv_r2(X[:h], y[:h].astype(float), X[h:], y[h:].astype(float))
            r["digit_acc_%s" % src] = logit_acc(X[:h], P[:h, k], X[h:], P[h:, k], dev)
        r["digit_majority_rate"] = float(np.bincount(P[h:, k], minlength=10).max() / (N - h)); r["model_digit_acc"] = float((pred[:, k] == P[:, k]).mean())
        r["n_carry_values"] = int(len(np.unique(C[:, k]))); res["by_column"][str(k + 1)] = {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}
        print(k + 1, res["by_column"][str(k + 1)], flush=True)
    # §141 gate (Theory 9e45b92b): NONLINEAR code -> carry lookup: each distinct code_k (the rounded FSQ tuple) -> its majority true carry_k on
    # the fit half; accuracy on the held-out half (codes unseen in the fit half count wrong). Per column + pooled; pooled < 0.9 opens cell r1s16.
    def lookup_acc(Qf, yf, Qs, ys):
        tab = {}
        for c_, y_ in zip(map(bytes, np.round(Qf).astype(np.int8)), yf):
            tab.setdefault(c_, []).append(int(y_))
        maj = {c_: np.bincount(v).argmax() for c_, v in tab.items()}
        hits = [maj.get(c_, None) == int(y_) for c_, y_ in zip(map(bytes, np.round(Qs).astype(np.int8)), ys)]
        unseen = sum(c_ not in maj for c_ in map(bytes, np.round(Qs).astype(np.int8)))
        return float(np.mean(hits)), int(unseen)
    pooled_hit = []
    for k in range(L):
        acc_k, uns = lookup_acc(Q[:h, k], C[:h, k], Q[h:, k], C[h:, k]); res["by_column"][str(k + 1)]["carry_code_nonlinear_acc"] = round(acc_k, 4)
        res["by_column"][str(k + 1)]["carry_code_nonlinear_unseen"] = uns; pooled_hit.append(acc_k * (N - h))
    res["carry_code_nonlinear_acc"] = float(sum(pooled_hit) / ((N - h) * L)); res["r1s16_gate_open"] = bool(res["carry_code_nonlinear_acc"] < 0.9)
    print("carry_code_nonlinear_acc pooled", round(res["carry_code_nonlinear_acc"], 4), "gate open (< 0.9):", res["r1s16_gate_open"], flush=True)
    res.update({"eval_dtype": str(amp), "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"})
    json.dump(res, open(os.path.join(d, os.environ.get("PERCOL_OUT", "eval_probe_percol.json")), "w", newline="\n"), indent=1)   # v2 (CV ridge): PERCOL_OUT


if __name__ == "__main__":
    main()
