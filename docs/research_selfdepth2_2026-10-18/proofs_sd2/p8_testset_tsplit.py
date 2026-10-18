"""PROOFS_SD2 check 8 ($0, CPU, 1 thread; read-only on the repo, no bytecode written).
(a) test_set(1000, 24)[:20000] == test_set(1000, 20) and [:16000] == test_set(1000) (syms, t, n, depth), recomputed here; T = len(syms) + 2 maxima by n.
(b) tsplit: FT-SELF eval_selfdepth items_n2 + T (n 13..16): accuracy by T band (<= 47 / >= 49), two-proportion z per n, within-n OLS slope of correct on T,
    and a stratified (n fixed effects) permutation test of the T slope; T alignment checked against test_set(1000) order."""
import os, sys, json, math
sys.dont_write_bytecode = True; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "D:/Latent")
import numpy as np
import psutil
try:
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
import boxes.nested170 as N
HERE = os.path.dirname(os.path.abspath(__file__))
A = N.test_set(1000, 24); B = N.test_set(1000, 20); C = N.test_set(1000)
f = lambda it: (tuple(it["syms"]), it["t"], it["n"], it["depth"])  # noqa: E731
eq20 = len(B) == 20000 and all(f(a) == f(b) for a, b in zip(A[:20000], B)); eq16 = len(C) == 16000 and all(f(a) == f(c) for a, c in zip(A[:16000], C))
Tn = {n: [len(it["syms"]) + 2 for it in A if it["n"] == n] for n in range(1, 25)}
Tmax = {n: max(v) for n, v in Tn.items()}; Tmean = {n: float(np.mean(v)) for n, v in Tn.items()}
e = json.load(open("D:/Latent/boxes/runs_fsq/s215_ftself_s1/eval_selfdepth.json"))
align = {n: e["T"][str(n)] == [len(it["syms"]) + 2 for it in C if it["n"] == n] for n in range(9, 17)}
ts = {}; rng = np.random.default_rng(0)
Xall, Yall, Gall = [], [], []
for n in range(13, 17):
    y = np.array(e["items_n2"][str(n)], dtype=float); T = np.array(e["T"][str(n)], dtype=float)
    lo, hi = T <= 47, T >= 49
    p1, n1, p2, n2 = y[lo].mean(), lo.sum(), y[hi].mean(), hi.sum(); pp = y[lo | hi].mean()
    z = (p2 - p1) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) if 0 < pp < 1 else None
    Tc = T - T.mean(); slope = float((Tc * (y - y.mean())).sum() / (Tc ** 2).sum())
    ts[n] = {"acc": float(y.mean()), "le47": [round(float(p1), 4), int(n1)], "ge49": [round(float(p2), 4), int(n2)], "z_ge49_minus_le47": z, "ols_slope_per_T": slope,
             "slope_per_10T": 10 * slope, "T_range": [int(T.min()), int(T.max())]}
    Xall.append(Tc); Yall.append(y - y.mean())
Xc = np.concatenate(Xall); Yc = np.concatenate(Yall); s_obs = float((Xc * Yc).sum() / (Xc ** 2).sum())
cnt = 0; NP = 2000
for _ in range(NP):                                                         # permute T within each n (stratified), recompute the pooled within-n slope
    xs = np.concatenate([rng.permutation(x) for x in Xall]); s_ = (xs * Yc).sum() / (xs ** 2).sum()
    cnt += abs(s_) >= abs(s_obs)
out = {"eq_n1_20_vs_test_set_20": eq20, "eq_n1_16_vs_test_set_16": eq16, "T_max_by_n": Tmax, "T_mean_by_n": {n: round(v, 2) for n, v in Tmean.items()},
       "selfdepth_T_aligned_with_test_set": align, "tsplit": ts, "pooled_within_n_slope_n13_16": s_obs, "perm_p_two_sided": (cnt + 1) / (NP + 1)}
json.dump(out, open(os.path.join(HERE, "p8_testset_tsplit.json"), "w", newline="\n"), indent=1, default=float)
print("eq20", eq20, "eq16", eq16); print("Tmax", Tmax); print("align", align)
for n, v in ts.items():
    print(n, v)
print("pooled slope", s_obs, "perm p", out["perm_p_two_sided"])
