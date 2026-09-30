"""Score THEORY §41 amendment 41-2 (GT18: does reach keep growing with training depth?) as registered.
Primary: median over available pn_gt18 students of one-loop bal-acc at d19-24 (single24, K' = 24, §37's probe) >= 0.60.
Reference: the same models' GT12 (pl_gt) and GT9 (pn_gt9). Descriptive: r5 d25-30 at K' = 32.
    python theory/score_s41b.py
"""
import json

import numpy as np

H = [102, 111, 113, 118, 125, 130]


def main():
    X = json.load(open("theory/results/exp18_pilotN18.json"))
    R = json.load(open("theory/results/exp18_deep_reanchor.json"))
    N = json.load(open("theory/results/exp18_pilotN.json"))
    m = lambda o, k, lo, hi: float(np.mean(o[k][lo - 1:hi]))
    have = [s for s in H if f"pn_gt18_s{s}" in X]
    med = lambda v: float(np.median(v)) if v else None
    out = {"n": len(have), "missing": [s for s in H if s not in have],
           "gt18_median_d19_24": med([m(X[f"pn_gt18_s{s}"], "single24", 19, 24) for s in have]),
           "gt18_median_d13_18": med([m(X[f"pn_gt18_s{s}"], "single24", 13, 18) for s in have]),
           "gt18_median_r5_d25_30_K32": med([m(X[f"pn_gt18_s{s}"], "r5_single32", 25, 30) for s in have if "r5_single32" in X[f"pn_gt18_s{s}"]]),
           "gt12_median_d19_24": med([m(R[f"pl_gt_s{s}"], "single24", 19, 24) for s in have]),
           "gt9_median_d19_24": med([m(N[f"pn_gt9_s{s}"], "single24", 19, 24) for s in have if f"pn_gt9_s{s}" in N]),
           "gt18_profile": [float(x) for x in np.median([X[f"pn_gt18_s{s}"]["single24"] for s in have], 0)] if have else None}
    out["met"] = out["gt18_median_d19_24"] is not None and out["gt18_median_d19_24"] >= 0.60
    json.dump(out, open("theory/results/score_s41b.json", "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
