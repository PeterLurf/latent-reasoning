"""$0 CLOCK-KEY (descriptive, eval_vbhalt rows): is the beyond-range collapse hazard keyed on the pass index t (c-like: state age / computed passes)
or on the content progress rho = reductions already done (decoded)?  rho_{t-1} = n - alive[t-1].  For items at pass t with deviation dev = alive[t-1] - (n - t + 1)
in {-1 (ahead: rho = t), 0 (on schedule: rho = t - 1), +1 (behind: rho = t - 2)}: hazard h = P(alive[t-1] - alive[t] >= 2).
c-keyed predicts h(t | dev) ~ h_on(t) for every dev; rho-keyed predicts h(t | dev) ~ h_on(t + dev*(-1)) i.e. h_on(rho + 1)."""
import json, sys
import numpy as np
OUT = sys.argv[1]; res = {}
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    rows = [r for r in json.load(open(run + "/eval_vbhalt.json"))["rows"] if 10 <= r["n"] <= 16]
    cnt = {}  # (t, dev) -> [items, collapses, alive_at_t-1 sum]
    byrho = {}  # (rho, dev) -> [items, collapses]
    for r in rows:
        n, al = r["n"], r["alive"]
        for t in range(2, 13):
            if al[t - 1] < 3:            # need >= 3 alive so that a >= 2 drop is possible and not merely 'finishing'
                continue
            dev = al[t - 1] - (n - t + 1)
            if dev not in (-1, 0, 1):
                continue
            col = int(al[t - 1] - al[t] >= 2)
            c_ = cnt.setdefault((t, dev), [0, 0]); c_[0] += 1; c_[1] += col
            rho = n - al[t - 1]
            b_ = byrho.setdefault((rho, dev), [0, 0]); b_[0] += 1; b_[1] += col
    tab = {}
    for t in range(2, 13):
        tab[str(t)] = {str(d): ({"items": cnt[(t, d)][0], "hazard": cnt[(t, d)][1] / cnt[(t, d)][0]} if (t, d) in cnt and cnt[(t, d)][0] >= 30 else None) for d in (-1, 0, 1)}
    # fit: hazard ~ logistic(a + b*t + g*dev). c-keyed: g ~ 0 ; rho-keyed (rho = t-1-dev): g ~ -b
    X = []; Y = []; W = []
    for (t, d), (k, c) in cnt.items():
        if k >= 30:
            X.append([1.0, t, d]); Y.append(c / k); W.append(k)
    X = np.array(X); Y = np.clip(np.array(Y), 1e-3, 1 - 1e-3); W = np.array(W, float)
    beta = np.zeros(3)
    for _ in range(200):                                     # weighted logistic IRLS on aggregated rates
        p = 1 / (1 + np.exp(-X @ beta)); g = X.T @ (W * (Y - p)); H = (X * (W * p * (1 - p))[:, None]).T @ X
        beta = beta + np.linalg.solve(H + 1e-9 * np.eye(3), g)
    cov = np.linalg.inv(H)
    res[name] = {"by_t_dev": tab, "logit_fit": {"intercept": beta[0], "b_t": beta[1], "g_dev": beta[2], "se": list(np.sqrt(np.diag(cov))),
                 "ratio_g_over_minus_b": float(beta[2] / -beta[1]), "reading": "c-keyed iff ratio ~ 0; rho-keyed iff ratio ~ 1"}}
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, v in res.items():
    print("==", name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v["logit_fit"].items() if k != "reading"})
    for t, row in v["by_t_dev"].items():
        print("  t", t, " ".join("dev%+d: %s" % (int(d), ("%.3f (%d)" % (x["hazard"], x["items"]) if x else "-")) for d, x in row.items()))
