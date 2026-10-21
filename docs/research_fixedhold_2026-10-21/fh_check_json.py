"""FIXED-HOLD $0 check A (draft FH_draft.md §1): read-only analysis of the six committed eval_khorizon.json files (Note 216-15).
No model, no GPU. Writes fh_check_json.out.json next to this script (LF). Run: python -B docs/research_fixedhold_2026-10-21/fh_check_json.py
Fields read: items[arm][n][K'] (argmax answer index into NUM_IDS, 0..10), items.y[n], ks.
Readings (descriptive):
  A1 class histogram of intact answers per K' (pooled n 1-8) + top-2 share; the same for wrong answers only.
  A2 attractor set S = the 2 most frequent intact-wrong classes at the largest K'; acc on items with y not in S ("non-attractor acc") vs all.
  A3 per-item path: share monotone (once wrong, wrong at every later K'); recovery = P(correct at K'b | wrong at K'a), a < b.
  A4 wrong-answer persistence: P(answer at 64 == answer at 48 | wrong at 48).
  A5 intact-wrong answer == bank_ablate_slva answer at the same K' (state route), and == the attractor set.
  A6 idle-count vs absolute: logistic fit correct ~ 1 + K' + n on non-attractor items (Newton, fp64); idle-count predicts b_n = -b_K, absolute b_n = 0.
  A7 cross-model item agreement (s1 lineage): share of items wrong in both D3-T s1 and FT-CTRL at K' 48 vs the independence product.
"""
import json
import os

import numpy as np

RUNS = ["s213_d3t_s1", "s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_d3t_s2", "s216_r1_s2"]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "boxes", "runs_fsq")


def load(run):
    d = json.load(open(os.path.join(ROOT, run, "eval_khorizon.json")))
    ks = [str(k) for k in d["ks"]]; ns = sorted(d["items"]["y"], key=int)
    y = np.concatenate([np.array(d["items"]["y"][n]) for n in ns]); nn = np.concatenate([np.full(len(d["items"]["y"][n]), int(n)) for n in ns])
    A = {arm: {k: np.concatenate([np.array(d["items"][arm][n][k]) for n in ns]) for k in ks} for arm in d["items"] if arm not in ("y", "idx")}
    return ks, y, nn, A


def logit_fit(X, z, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p); H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(w)); g = X.T @ (z - p)
        w = w + np.linalg.solve(H, g)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return w, se


out = {}
for run in RUNS:
    ks, y, nn, A = load(run); I = A["intact"]; r = {}; kl = ks[-1]
    r["A1_hist_intact"] = {k: np.bincount(I[k], minlength=11).tolist() for k in ks}
    r["A1_hist_y"] = np.bincount(y, minlength=11).tolist()
    r["A1_hist_wrong"] = {k: np.bincount(I[k][I[k] != y], minlength=11).tolist() for k in ks}
    hw = np.bincount(I[kl][I[kl] != y], minlength=11); S = [int(c) for c in np.argsort(-hw)[:2]]
    r["A2_attractor_S"] = S; r["A2_top2_share_of_wrong"] = {k: float(np.sort(np.bincount(I[k][I[k] != y], minlength=11))[-2:].sum() / max(1, (I[k] != y).sum())) for k in ks}
    na = ~np.isin(y, S)
    r["A2_acc_all"] = {k: float((I[k] == y).mean()) for k in ks}
    r["A2_acc_nonattr"] = {k: float((I[k] == y)[na].mean()) for k in ks}
    r["A2_acc_attr_y"] = {k: float((I[k] == y)[~na].mean()) for k in ks}
    r["A2_share_items_y_in_S"] = float((~na).mean())
    C = np.stack([I[k] == y for k in ks], 1)
    mono = np.all([~C[:, i] <= ~C[:, j] for i in range(len(ks)) for j in range(i + 1, len(ks))], 0)
    r["A3_monotone_share"] = float(mono.mean())
    r["A3_recovery"] = {f"{ks[i]}->{ks[j]}": [float(C[~C[:, i], j].mean()) if (~C[:, i]).any() else None, int((~C[:, i]).sum())] for i in range(len(ks)) for j in range(i + 1, len(ks))}
    if len(ks) >= 2:
        w48 = I[ks[-2]] != y
        r["A4_wrong_persist_same_answer"] = [float((I[kl] == I[ks[-2]])[w48].mean()), int(w48.sum())]
    for arm in ("bank_ablate_slva", "self_ablate"):
        if arm in A:
            r[f"A5_wrong_eq_{arm}"] = {k: float((I[k] == A[arm][k])[I[k] != y].mean()) if (I[k] != y).any() else None for k in ks}
            r[f"A5_{arm}_in_S"] = {k: float(np.isin(A[arm][k], S).mean()) for k in ks}
    r["A5_wrong_in_S"] = {k: float(np.isin(I[k][I[k] != y], S).mean()) if (I[k] != y).any() else None for k in ks}
    # A6: logistic on non-attractor items, K' in {32, 48, 64} (K' 24 is ~1.0)
    rows = [(int(k), n_, c) for k in ks if int(k) >= 32 for n_, c in zip(nn[na], (I[k] == y)[na])]
    X = np.array([[1.0, (k - 48) / 16.0, (n_ - 4.5) / 3.5] for k, n_, _ in rows]); z = np.array([c for _, _, c in rows], float)
    w, se = logit_fit(X, z)
    bK = w[1] / 16.0; bn = w[2] / 3.5
    r["A6_logit_per_pass"] = {"b_K": float(bK), "b_n": float(bn), "se_K": float(se[1] / 16), "se_n": float(se[2] / 3.5), "ratio_minus_bn_over_bK": float(-bn / bK)}
    r["A6_acc_nonattr_by_n"] = {k: {str(n_): float((I[k] == y)[na & (nn == n_)].mean()) for n_ in range(1, 9)} for k in ks}
    out[run] = r
    print(run, "S", S, "acc_all", {k: round(v, 3) for k, v in r["A2_acc_all"].items()}, "acc_nonattr", {k: round(v, 3) for k, v in r["A2_acc_nonattr"].items()},
          "top2/wrong", {k: round(v, 2) for k, v in r["A2_top2_share_of_wrong"].items()}, "mono", round(r["A3_monotone_share"], 3),
          "persist", r.get("A4_wrong_persist_same_answer"), "A6 -bn/bK %.2f (bK %.4f se %.4f, bn %.4f se %.4f)" % (r["A6_logit_per_pass"]["ratio_minus_bn_over_bK"], bK, se[1] / 16, bn, se[2] / 3.5), flush=True)

# A7: cross-model agreement (same test items, same order; s1 lineage)
ks, y, nn, A1 = load("s213_d3t_s1"); _, y2, _, A2 = load("s215_ftctrl_s1"); assert (y == y2).all()
for k in ("48", "64"):
    w1 = A1["intact"][k] != y; w2 = A2["intact"][k] != y
    out.setdefault("A7_cross", {})[k] = {"p_both_wrong": float((w1 & w2).mean()), "indep": float(w1.mean() * w2.mean()), "same_wrong_answer_given_both": float((A1["intact"][k] == A2["intact"][k])[w1 & w2].mean())}
print("A7", out["A7_cross"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fh_check_json.out.json"), "w", newline="\n"), indent=1)
