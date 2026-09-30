"""PROTOCOL7 general rules for T9/T10: model-bootstrap 95 % CI of AUC(P7) (5000 resamples of runs) and one-sided
permutation p (10 000 label shuffles), seed 0, on the scored rows of results/p7s_t9.json (primary: above the floor).
    python scratch/p7s_t9_stats.py"""
import json

import numpy as np

d = json.load(open("results/p7s_t9.json"))
rows = d["rows"]
FLOOR = 0.03


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def stats(sel):
    x = np.array([rows[r]["P7"] for r in sel]); y = np.array([rows[r]["extrap"] for r in sel])
    a = auc(x, y)
    if a is None:
        return {"n": len(sel), "auc": None}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(5000):
        i = rng.integers(0, len(sel), len(sel))
        b = auc(x[i], y[i])
        if b is not None:
            boots.append(b)
    perm = [auc(x, rng.permutation(y)) for _ in range(10000)]
    p = (sum(v >= a for v in perm) + 1) / (len(perm) + 1)
    return {"n": len(sel), "n_extrap": int(y.sum()), "auc": a, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "perm_p_one_sided": p, "holm_bound_5x": min(1.0, 5 * p)}


prim = [r for r in rows if rows[r]["in_peak"] >= FLOOR]
out = {"T9_pooled": stats([r for r in prim if not r.startswith("t10")]),
       "T10": stats([r for r in prim if r.startswith("t10")])}
for fam in ("t9l3", "t9l4", "t9w512"):
    out[fam] = stats([r for r in prim if r.startswith(fam + "_")])
json.dump(out, open("results/p7s_t9_stats.json", "w"), indent=1)
for k, v in out.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
