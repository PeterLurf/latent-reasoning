"""Score THEORY §41 (content key vs moved count) as registered, from theory/results/exp18_pilotN.json (one loop at
K' = 24 on §37's probe) for the pn_gt9 students; pl_gt (GT12) of the same models from exp18_deep_reanchor.json as the
descriptive reference.
    python theory/score_s41.py
"""
import json

import numpy as np

H = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]


def main():
    X = json.load(open("theory/results/exp18_pilotN.json"))
    R = json.load(open("theory/results/exp18_deep_reanchor.json"))
    m = lambda o, lo, hi: float(np.mean(o["single24"][lo - 1:hi]))
    have = [s for s in H if f"pn_gt9_s{s}" in X]
    d10_12 = [m(X[f"pn_gt9_s{s}"], 10, 12) for s in have]
    d13_18 = [m(X[f"pn_gt9_s{s}"], 13, 18) for s in have]
    med = lambda v: float(np.median(v)) if v else None
    out = {"n": len(have), "median_d10_12": med(d10_12), "median_d13_18": med(d13_18),
           "profile_median_d1_24": [float(x) for x in np.median([X[f"pn_gt9_s{s}"]["single24"] for s in have], 0)] if have else None,
           "ref_gt12_median_d13_18": med([m(R[f"pl_gt_s{s}"], 13, 18) for s in have])}
    if out["median_d10_12"] is not None and out["median_d10_12"] < 0.60:
        out["verdict"] = "moved count"
    elif out["median_d13_18"] is not None and out["median_d13_18"] >= 0.60:
        out["verdict"] = "content key"
    else:
        out["verdict"] = "partial"
    json.dump(out, open("theory/results/score_s41.json", "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
