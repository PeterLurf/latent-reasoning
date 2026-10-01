"""Score PROTOCOL8 amendment 3 (secondary SC, docs/PROTOCOL8.md). Written before any NA evaluation ran.
Inputs: prosqa/res_p8c_na/<arm>_s<seed>.json (eval_na_p8.py; latent arms coco/at/rt) and
prosqa/res_p8c_na/cot_s<seed>.json (eval_p8.py --mode cot on p8_na_test; original = prosqa/res_p8c/cot_s<seed>/cot_test_e30.json).
Per latent seed: D710 = acc at K' = 10, depth 7-10; D11 = acc at K' = 16, depth 11; on p8_test (orig) and p8_na_test (NA).
Cue reliance = orig - NA. Secondary comparisons at vs coco and rt vs coco on D710_NA / D11_NA with the P8-1 rule
(exact one-sided MW p <= 0.05 and mean difference >= 0.05 -> supported; mean difference <= 0 -> not supported).
Reference: the proximity heuristic's accuracy per file and stratum (theory/cue_origin.py rule).
    python prosqa/score_p8_na.py
"""
import glob
import itertools
import json
import os
import sys
from collections import deque

import numpy as np

R = "prosqa/res_p8c_na"


def acc(rows, f, K, lo, hi):
    s = [r for r in rows if r["file"] == f and lo <= r["depth"] <= hi]
    return float(np.mean([r[f"m{K}"] > 0 for r in s])) if s else None


def mw_p(x, y):
    """Exact one-sided Mann-Whitney P(U >= observed) for x > y (permutation over group labels)."""
    allv = list(x) + list(y); n = len(x)
    obs = sum((a > b) + 0.5 * (a == b) for a in x for b in y)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        xs = [allv[i] for i in idx]; ys = [allv[i] for i in range(len(allv)) if i not in idx]
        u = sum((a > b) + 0.5 * (a == b) for a in xs for b in ys)
        cnt += u >= obs - 1e-9; tot += 1
    return cnt / tot


def heuristic(path, lo, hi):
    """Proximity heuristic accuracy: the option farther from every person statement (ties 0.5)."""
    out = []
    for x in json.load(open(path)):
        d = len(x["steps"])
        if not lo <= d <= hi:
            continue
        sym = x["idx_to_symbol"]; adj = {}
        for u, v in x["edges"]:
            adj.setdefault(u, []).append(v)
        people = [i for i, s in enumerate(sym) if s[:1].isupper()]
        dist = {}
        q = deque()
        for p in people:
            dist[p] = 0; q.append(p)
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        dt, dn = dist.get(x["target"], 99), dist.get(x["neg_target"], 99)
        out.append(1.0 if dn < dt else 0.0 if dn > dt else 0.5)
    return float(np.mean(out))


def main():
    res = {"latent": {}, "cot": {}}
    for arm in ("coco", "at", "rt"):
        for f in sorted(glob.glob(f"{R}/{arm}_s*.json")):
            rows = json.load(open(f))["rows"]; s = os.path.basename(f)[:-5]
            v = {"D710_orig": acc(rows, "p8_test", 10, 7, 10), "D710_NA": acc(rows, "p8_na_test", 10, 7, 10),
                 "D11_orig": acc(rows, "p8_test", 16, 11, 11), "D11_NA": acc(rows, "p8_na_test", 16, 11, 11)}
            v["cue_D710"] = v["D710_orig"] - v["D710_NA"]; v["cue_D11"] = v["D11_orig"] - v["D11_NA"]
            res["latent"][s] = v
    for f in sorted(glob.glob(f"{R}/cot_s*.json")):
        s = os.path.basename(f)[:-5]
        res["cot"][s] = {"NA_file": f, "orig_file": f"prosqa/res_p8c/{s}/cot_test_e30.json"}
    get = lambda arm, k: [v[k] for n, v in res["latent"].items() if n.startswith(arm + "_s")]
    comp = {}
    for arm in ("at", "rt"):
        for k in ("D710_NA", "D11_NA"):
            x, y = get(arm, k), get("coco", k)
            if len(x) >= 3 and len(y) >= 3:
                d = float(np.mean(x) - np.mean(y)); p = mw_p(x, y)
                comp[f"{arm}_vs_coco_{k}"] = {"mean_diff": d, "p": p, "decision": "supported" if p <= 0.05 and d >= 0.05
                                              else "not supported" if d <= 0 else "inconclusive"}
    res["secondary_comparisons"] = comp
    res["arm_means"] = {arm: {k: float(np.mean(get(arm, k))) for k in ("D710_orig", "D710_NA", "D11_orig", "D11_NA", "cue_D710", "cue_D11")}
                        for arm in ("coco", "at", "rt") if get(arm, "D710_NA")}
    res["heuristic"] = {f: {"D710": heuristic(f"prosqa/p8data/{f}.json", 7, 10), "D11": heuristic(f"prosqa/p8data/{f}.json", 11, 11)}
                        for f in ("p8_test", "p8_na_test")}
    json.dump(res, open("results/p8_na.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("arm_means", "secondary_comparisons", "heuristic")}, indent=1))


if __name__ == "__main__":
    main()
