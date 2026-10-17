"""RT-6: decode-free (R-free) size test on D3-T itself: eval_vb 'probe' per-item correctness (a fresh coda trained on n 3-8 reading V_t,
t in {8, 12}) split by slot count T within n. A t-keyed limit predicts little T dependence at fixed n; a size-keyed limit (N2-like) predicts accuracy
falling with T at fixed n. Logistic fit acc ~ n + T over n 9-12 (items with probe value 0/1). N1 (t-keyed by vbhalt) and N2 (T-keyed) are the controls."""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from testset import test_set
TE = test_set(); OUT = sys.argv[1]; res = {}
def fit(X, y):
    X = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b)); H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(b)); b = b + np.linalg.solve(H, X.T @ (y - p))
    return [round(float(v), 4) for v in b], [round(float(v), 4) for v in np.sqrt(np.diag(np.linalg.inv(H)))]
for name in ("s211_vb_carry_s1", "s211_vb_pa_s1", "s213_d3t_s1", "s213_d3f_s1"):
    e = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_vb.json" % name)); pr = e["probe"]; R = {}
    assert e["test_meta"]["n"] == [it["n"] for it in TE]
    for t in ("8", "12"):
        if t not in pr:
            continue
        it_ = np.array(pr[t]["items"], float); R[t] = {"acc_by_n": {n: pr[t]["acc_by_n"][str(n)] for n in (8, 9, 10, 11, 12)}}
        sp = {}
        for n in (9, 10, 11):
            idx = [j for j, x in enumerate(TE) if x["n"] == n]; T = np.array([TE[j]["T"] for j in idx]); a = it_[idx]
            q1, q2 = np.quantile(T, [1 / 3, 2 / 3]); sp[str(n)] = {"acc_lowT": round(float(a[T <= q1].mean()), 3), "acc_highT": round(float(a[T > q2].mean()), 3),
                                                                  "k": [int((T <= q1).sum()), int((T > q2).sum())], "T_cut": [float(q1), float(q2)]}
        idx = [j for j, x in enumerate(TE) if 9 <= x["n"] <= 12]
        X = np.array([[TE[j]["n"], TE[j]["T"] - 2 * TE[j]["n"]] for j in idx], float)   # T - 2n = 3 + 2 * bracket pairs: size beyond n
        R[t]["Tsplit"] = sp; R[t]["fit_acc~n+(T-2n)"] = fit(X, it_[idx])
    res[name] = R
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, R in res.items():
    print("==", name)
    for t, v in R.items():
        print("  probe t", t, "acc_by_n", v["acc_by_n"], "| fit [b0, b_n, b_extraT] / se", v["fit_acc~n+(T-2n)"])
        for n, x in v["Tsplit"].items():
            print("     n", n, x)
