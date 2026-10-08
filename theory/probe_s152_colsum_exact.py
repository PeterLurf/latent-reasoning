"""THEORY note 152-5 (descriptive; Audit): is the carry-save column sum stored EXACTLY? Rounded exact-match accuracy of the held-out column-sum readout, per column.
Same states, targets and halves as multrow.s152_eval's colsum_readout: the additive heads' state after segments 1-4 (flattened), carry-save column sums k = 1..7
(no carries), fit half = the first 500 test items, score half = the rest; the same dual ridge on standardised features with lambda by 3-fold CV (multrow.kridge_r2),
refit here to return predictions. exact = (round(pred) == y).
  final:  after segment 4 (all of a column's products written), per column
  live:   pooled over the (segment, column) cells that already hold >= 1 product
CPU only (torch threads <= 3; run at Idle priority). Writes theory/results/colsum_exact_s152.json.
    python theory/probe_s152_colsum_exact.py s152_a_4x4_s1 s152_ctrl_4x4_s1
"""
import os
os.environ["BOXES_NB"] = "5"                                                # boxes.core must load with BOXES_NB=5
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.multrow as MR  # noqa: E402
from boxes.multcol import load_bench  # noqa: E402

LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)                                         # multrow.kridge_r2's grid (the registered probe)
if "--lams_ext" in sys.argv:                                                 # robustness (descriptive): the registered grid's minimum was chosen for every column
    LAMS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3); sys.argv.remove("--lams_ext"); OUT_SUFFIX = "_lamsext"
else:
    OUT_SUFFIX = ""
FIT_TRAIN = 0                                                                # --fit_train N: fit on N train items (final state only), score on the 1,000 test items
if "--fit_train" in sys.argv:
    i_ = sys.argv.index("--fit_train"); FIT_TRAIN = int(sys.argv[i_ + 1]); del sys.argv[i_:i_ + 2]
    LAMS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5); OUT_SUFFIX = "_fit%d" % FIT_TRAIN


def kridge_fit_pred(Xf, yf, Xs):
    """multrow.kridge_r2's estimator (dual ridge on standardised features, lambda x n by 3-fold CV on the fit half), returning (predictions on Xs, lambda)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Zf = ((Xf - mu) / sd).astype(np.float32); Zs = ((Xs - mu) / sd).astype(np.float32); d = Zf.shape[1]

    def fit_pred(Za, ya, Zb, lam):
        K = (Za @ Za.T).astype(np.float64) / d; ym = ya.mean(); al = np.linalg.solve(K + lam * len(Za) * np.eye(len(Za)), ya - ym)
        return (Zb @ Za.T).astype(np.float64) / d @ al + ym

    def r2(y_, p_):
        return float(1 - ((y_ - p_) ** 2).sum() / max(((y_ - y_.mean()) ** 2).sum(), 1e-12))
    fold = np.arange(len(Zf)) % 3
    best = max(LAMS, key=lambda l_: np.mean([r2(yf[fold == k], fit_pred(Zf[fold != k], yf[fold != k], Zf[fold == k], l_)) for k in range(3)]))
    return fit_pred(Zf, yf, Zs, best), best


def gram_ridge_all(Xtr, Xte, Y, lams):
    """Fast variant of kridge_fit_pred for many targets: standardise on train, one Gram matrix G = Z Z^T / d over [train; test], 3-fold CV on train per
    target column, then the refit. Returns (pred (n_te, k), chosen lambda per column)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = ((np.concatenate([Xtr, Xte]) - mu) / sd).astype(np.float32); d = Z.shape[1]; n = len(Xtr)
    G = (Z @ Z.T).astype(np.float64) / d; del Z
    Gtt, Gst = G[:n, :n], G[n:, :n]; fold = np.arange(n) % 3; preds, chosen = [], []
    for c in range(Y.shape[1]):
        y = Y[:, c]
        def cv(l_):
            sc = []
            for k in range(3):
                tr, va = fold != k, fold == k; ym = y[tr].mean()
                al = np.linalg.solve(Gtt[np.ix_(tr, tr)] + l_ * tr.sum() * np.eye(tr.sum()), y[tr] - ym)
                p_ = Gtt[np.ix_(va, tr)] @ al + ym; sc.append(1 - ((y[va] - p_) ** 2).sum() / max(((y[va] - y[va].mean()) ** 2).sum(), 1e-12))
            return np.mean(sc)
        best = max(lams, key=cv); ym = y.mean(); al = np.linalg.solve(Gtt + best * n * np.eye(n), y - ym)
        preds.append(Gst @ al + ym); chosen.append(best)
    return np.stack(preds, 1), chosen


def main():
    torch.set_num_threads(3); t0 = time.time()
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"))[:2]
    N = len(teA); h = N // 2; res = {"def": __doc__.split("\n")[1], "n_fit": h, "n_score": N - h}
    if FIT_TRAIN:
        from boxes.multcol import load_bench_head
        trA, trB = load_bench_head(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"), FIT_TRAIN)
        res["fit"] = "%d train items (final state), scored on the %d test items" % (len(trA), N)
    for d in sys.argv[1:]:
        run = os.path.join(ROOT, "boxes", "runs_fsq", d); a = json.load(open(os.path.join(run, "args.json")))
        F = MR.F                                                            # as multrow.main: fsq120's code width / levels from the run's args (unused by DeltaMem,
        if int(a["d_q"]) != F.D_Q or int(a["levels"]) != F.LEV:             # but present in the state dict)
            F.D_Q, F.LEV = int(a["d_q"]), int(a["levels"]); F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
        m = MR.MultRowDelta(s5.load_gpt2("cpu"), MR.RowFam(), int(a.get("mem_layer", 6)))
        m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu")); m.eval(); m.p_hard = 1.0
        with torch.no_grad():
            _, kept, _, _ = MR.rollout_drow(m, teA, teB, "cpu", None, 100, keep=(1, 2, 3, 4), decode=False)
        if FIT_TRAIN:                                                       # final state only: fit on train items, score on test items
            def final_states(A_, B_):
                with torch.no_grad():
                    _, kp, _, _ = MR.rollout_drow(m, A_, B_, "cpu", None, 100, keep=(4,), decode=False)
                return kp[4][:, :DM.H_ADD].reshape(len(A_), -1)
            def colsums(A_, B_):
                c_ = np.zeros((len(A_), 7))
                for j_ in range(4):
                    for i_ in range(4):
                        c_[:, i_ + j_] += A_[:, i_] * B_[:, j_]
                return c_
            Xtr, Xte = final_states(trA, trB), final_states(teA, teB); Ctr, Cte = colsums(trA, trB), colsums(teA, teB)
            out = {"final_exact_by_column": {}, "r2_by_column": {}, "max_abs_resid_final": {}, "lambda": {}}
            P_, lams_ = gram_ridge_all(Xtr, Xte, Ctr, LAMS)
            for k in range(7):
                pred, lam = P_[:, k], lams_[k]
                out["final_exact_by_column"][str(k + 1)] = float((np.rint(pred) == Cte[:, k]).mean())
                out["r2_by_column"][str(k + 1)] = float(1 - ((Cte[:, k] - pred) ** 2).sum() / max(((Cte[:, k] - Cte[:, k].mean()) ** 2).sum(), 1e-12))
                out["max_abs_resid_final"][str(k + 1)] = float(np.abs(pred - Cte[:, k]).max()); out["lambda"][str(k + 1)] = lam
            res[d] = out; print(d, "FIT-TRAIN final exact", out["final_exact_by_column"], "| R2", out["r2_by_column"], "| lam", out["lambda"], round(time.time() - t0), flush=True)
            del Xtr, Xte, m; continue
        cs = np.zeros((N, 4, 7), np.float64)                                # as s152_eval: carry-save column sums after segment j
        for j in range(4):
            for i in range(4):
                cs[:, j:, i + j] += (teA[:, i] * teB[:, j])[:, None]
        XA = np.stack([kept[j + 1][:, :DM.H_ADD].reshape(N, -1) for j in range(4)], 1); del kept
        out = {"final_exact_by_column": {}, "live_exact_by_column": {}, "r2_by_column": {}, "max_abs_resid_final": {}, "lambda": {}}
        for k in range(7):
            y = cs[:, :, k]
            if y.std() == 0:
                continue
            pred, lam = kridge_fit_pred(XA[:h].reshape(h * 4, -1), y[:h].reshape(-1), XA[h:].reshape((N - h) * 4, -1))
            P = pred.reshape(N - h, 4); Y = y[h:]
            live = np.array([[j >= k - 3 and k <= j + 3 for j in range(4)]] * (N - h))  # (segment j 0-based) holds >= 1 product of column k (0-based)
            out["final_exact_by_column"][str(k + 1)] = float((np.rint(P[:, 3]) == Y[:, 3]).mean())
            out["live_exact_by_column"][str(k + 1)] = float((np.rint(P[live]) == Y[live]).mean())
            out["r2_by_column"][str(k + 1)] = float(1 - ((Y - P) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12))
            out["max_abs_resid_final"][str(k + 1)] = float(np.abs(P[:, 3] - Y[:, 3]).max()); out["lambda"][str(k + 1)] = lam
        res[d] = out
        print(d, "final exact", out["final_exact_by_column"], "| live", out["live_exact_by_column"], "| R2", out["r2_by_column"], round(time.time() - t0), flush=True)
        del XA, m
    res["sec"] = round(time.time() - t0)
    res["lams"] = list(LAMS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "colsum_exact_s152%s.json" % OUT_SUFFIX), "w", newline=chr(10)), indent=1)


if __name__ == "__main__":
    main()
