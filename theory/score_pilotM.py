"""Score THEORY §38 (Pilot M: PM1–PM4 + descriptives) as registered.

Deep measure = mean bal-acc over d13–18 of one loop at K' = 24 on §37's probe (theory/results/exp18_pilotM.json, from
theory/exp18_deep_reanchor.py --r5 on the pm_* runs). Competence = eval_arch's gate. Round-1 students (pl_t8_s*) are
the baseline: their d8 from eval.json, delayed from eval.json, deep measure from theory/results/exp18_deep_reanchor.json.
    python theory/score_pilotM.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pilotFG import ev, pair_rule  # noqa: E402

H = [102, 109, 111, 112, 113, 115, 118, 119, 125, 126, 130, 131, 132, 135, 140, 145, 146, 147, 157, 158]
T8F = H[0::2]
GT = H[1::2]


def main():
    X = json.load(open("theory/results/exp18_pilotM.json"))
    R1 = json.load(open("theory/results/exp18_deep_reanchor.json"))
    deep = lambda o, k="single24", lo=12, hi=18: float(np.mean(o[k][lo:hi]))
    E = lambda r: ev(r)
    comp = lambda r: E(r) is not None and E(r)["gate"]
    out = {"arms": {}}
    for arm, seeds in (("t8", H), ("self", H), ("t8f", T8F), ("gt", GT)):
        rs = [f"pm_{arm}_s{s}" for s in seeds]
        got = [r for r in rs if E(r) is not None and r in X]
        out["arms"][arm] = {"registered": len(rs), "evaluated": len(got), "competent": sum(comp(r) for r in got),
                            "deep_ge_0.60": sum(deep(X[r]) >= 0.60 for r in got),
                            "median_deep": float(np.median([deep(X[r]) for r in got])) if got else None,
                            "median_d19_24": float(np.median([deep(X[r], lo=18, hi=24) for r in got])) if got else None,
                            "median_d25_30_K32": float(np.median([deep(X[r], "r5_single32", 24, 30) for r in got if "r5_single32" in X[r]])) if got else None,
                            "median_d8": float(np.median([E(r)["d8"] for r in got])) if got else None,
                            "missing": [r for r in rs if r not in got]}
    pairs = [(deep(X[f"pm_t8_s{s}"]), deep(X[f"pm_self_s{s}"])) for s in H
             if f"pm_t8_s{s}" in X and f"pm_self_s{s}" in X and comp(f"pm_t8_s{s}") and comp(f"pm_self_s{s}")]
    out["PM1"] = pair_rule(pairs, lambda x, y: x > y)
    n_ok = sum(deep(X[f"pm_t8_s{s}"]) >= 0.60 for s in H if f"pm_t8_s{s}" in X)
    base = sum(deep(R1[f"pl_t8_s{s}"]) >= 0.60 for s in H)
    out["PM2"] = {"t8_deep_ge_0.60": n_ok, "of": len(H), "round1_students": base, "met": n_ok >= 10}
    d8 = [E(f"pm_t8_s{s}")["d8"] for s in H if E(f"pm_t8_s{s}")]
    out["PM3"] = {"median_d8_t8": float(np.median(d8)) if d8 else None, "threshold": 0.608,
                  "met": bool(d8) and float(np.median(d8)) >= 0.608}
    drops = [E(f"pl_t8_s{s}")["delayed"] - E(f"pm_t8_s{s}")["delayed"] for s in H if E(f"pm_t8_s{s}") and E(f"pl_t8_s{s}")]
    out["PM4"] = {"n": len(drops), "median_delayed_drop": float(np.median(drops)) if drops else None,
                  "met": bool(drops) and float(np.median(drops)) <= 0.02}
    fp = [(deep(X[f"pm_t8f_s{s}"]), deep(X[f"pm_t8_s{s}"])) for s in T8F
          if f"pm_t8f_s{s}" in X and f"pm_t8_s{s}" in X and comp(f"pm_t8f_s{s}") and comp(f"pm_t8_s{s}")]
    out["desc_T8F_vs_T8"] = pair_rule(fp, lambda x, y: x > y)
    json.dump(out, open("theory/results/score_pilotM.json", "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
