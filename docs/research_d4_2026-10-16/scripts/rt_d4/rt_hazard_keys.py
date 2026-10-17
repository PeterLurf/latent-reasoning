"""RT-3: what is the collapse hazard keyed on? On-schedule events (alive[t-1] == n - t + 1 >= 3), n 10..16, t 2..12 (draft's conditioning).
Table h(t, n). Weighted logistic fits on cells (k >= 30): M_t (t only), M_n (n only), M_r (r = alive[t-1] only), M_tn (t + n), M_tT (t + T_slots),
M_tnT (t + n + T). T = the item's slot count (len(syms) + 2) from the replicated test set; at fixed (t, n), T varies with brackets.
Deviance comparisons (lower = better). Also h(t, n) split by T tercile at fixed n (size effect at fixed n and t)."""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from testset import test_set
OUT = sys.argv[1]; TE = test_set(); res = {}
def fit(X, k, c):
    X = np.column_stack([np.ones(len(X)), X]); beta = np.zeros(X.shape[1]); Y = c / k
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta)); g = X.T @ (k * Y - k * p); H = (X * (k * p * (1 - p))[:, None]).T @ X
        beta = beta + np.linalg.solve(H + 1e-8 * np.eye(len(beta)), g)
    p = np.clip(1 / (1 + np.exp(-X @ beta)), 1e-9, 1 - 1e-9)
    dev = -2 * float((c * np.log(p) + (k - c) * np.log(1 - p)).sum() - (c * np.log(np.clip(Y, 1e-9, 1)) + (k - c) * np.log(np.clip(1 - Y, 1e-9, 1))).sum())
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {"beta": [round(float(b), 4) for b in beta], "se": [round(float(s), 4) for s in se], "deviance": round(dev, 1), "cells": len(k)}
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    rows = json.load(open(run + "/eval_vbhalt.json"))["rows"]
    ev = []  # (t, n, T, r, collapse)
    for r in rows:
        n, al = r["n"], r["alive"]
        if not (10 <= n <= 16):
            continue
        T = TE[r["item"]]["T"]
        for t in range(2, 13):
            if al[t - 1] == n - t + 1 and al[t - 1] >= 3:
                ev.append((t, n, T, al[t - 1], int(al[t - 1] - al[t] >= 2)))
    ev = np.array(ev)
    tab = {}
    for t in range(2, 13):
        tab[str(t)] = {}
        for n in range(10, 17):
            m = (ev[:, 0] == t) & (ev[:, 1] == n)
            if m.sum() >= 30:
                tab[str(t)][str(n)] = [round(float(ev[m, 4].mean()), 3), int(m.sum())]
    # cell aggregation for fits: cells by (t, n, T-bin)
    Tb = np.digitize(ev[:, 2], [25, 29, 33, 37, 41, 45, 49])
    keys = {}
    for (t, n, T, rr, c), tb in zip(ev, Tb):
        kk = keys.setdefault((t, n, tb), [0, 0, 0.0]); kk[0] += 1; kk[1] += c; kk[2] += T
    cells = [(t, n, tb, v[0], v[1], v[2] / v[0]) for (t, n, tb), v in keys.items() if v[0] >= 30]
    C = np.array(cells, float); t_, n_, Tm, k, c = C[:, 0], C[:, 1], C[:, 5], C[:, 3], C[:, 4]; r_ = n_ - t_ + 1
    fits = {"t": fit(t_[:, None], k, c), "n": fit(n_[:, None], k, c), "r": fit(r_[:, None], k, c), "t+n": fit(np.column_stack([t_, n_]), k, c),
            "t+T": fit(np.column_stack([t_, Tm]), k, c), "t+n+T": fit(np.column_stack([t_, n_, Tm]), k, c), "n+T": fit(np.column_stack([n_, Tm]), k, c)}
    # size effect at fixed (t, n): T tercile split
    tsplit = {}
    for t in (6, 7, 8):
        for n in (10, 11, 12):
            m = (ev[:, 0] == t) & (ev[:, 1] == n)
            if m.sum() < 90:
                continue
            q1, q2 = np.quantile(ev[m, 2], [1 / 3, 2 / 3]); lo = m & (ev[:, 2] <= q1); hi = m & (ev[:, 2] > q2)
            tsplit["t%d_n%d" % (t, n)] = {"h_lowT": round(float(ev[lo, 4].mean()), 3), "h_highT": round(float(ev[hi, 4].mean()), 3), "T_lo<=": float(q1), "T_hi>": float(q2), "k": [int(lo.sum()), int(hi.sum())]}
    res[name] = {"h_t_n": tab, "fits": fits, "T_split_fixed_t_n": tsplit}
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, v in res.items():
    print("==", name)
    for t, row in v["h_t_n"].items():
        print("  t", t, " ".join("n%s:%.3f(%d)" % (n, x[0], x[1]) for n, x in row.items()))
    for f, x in v["fits"].items():
        print("  fit", f, x)
    for kk, x in v["T_split_fixed_t_n"].items():
        print("  Tsplit", kk, x)
