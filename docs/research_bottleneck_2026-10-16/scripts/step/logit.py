"""Joint logistic model of the prefix-clean first-error hazard (187-3 rows, n 9-12, t <= 8): err ~ t + hh + age_max + held + alive + bdepth + prov.
Standardised coefficients (per SD), plain IRLS with a small ridge. CPU, descriptive."""
import json, sys
import numpy as np
F = json.load(open(sys.argv[1])); ns = F["n"]; feats = F["feats"]
idx = [i for i, n in enumerate(ns) if 9 <= n <= 12]
NAMES = ["t", "hh", "age_max", "held", "alive", "bdepth", "prov"]
def irls(X, y, lam=1e-3, it=50):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(it):
        z = X @ w; p = 1 / (1 + np.exp(-z)); W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1]); g = X.T @ (y - p) - lam * w
        w = w + np.linalg.solve(H, g)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return w, se
for m in sys.argv[2:]:
    rows = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_s187_deepaux_items_n9_12.json" % m))["items"]
    X, y = [], []
    for k, i in enumerate(idx):
        hb = rows[k]["handle_by_t"]; f = feats[i]; n = ns[i]
        for t in range(1, min(n, 8) + 1):
            if str(t) not in hb: break
            s = f[t - 1]; e = 1 - hb[str(t)]
            X.append([t, s["hh"], s["age_max"], s["held"], s["alive"], s["bdepth"], s["prov"]]); y.append(e)
            if e: break
    X = np.array(X, float); y = np.array(y, float); mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd
    w, se = irls(Z, y)
    print("%-20s N %5d err %.3f | " % (m, len(y), y.mean()) + " ".join("%s %+.2f(%.2f)" % (nm, w[j + 1], se[j + 1]) for j, nm in enumerate(NAMES)))
