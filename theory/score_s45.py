"""Score THEORY §45 (Pilot CLR-1: PC1-PC4) as registered, from theory/results/exp20_clr_eval.json and eval.json.
    python theory/score_s45.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pilotFG import ev, pair_rule  # noqa: E402

H = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
H_B = [109, 112, 115, 119, 126, 131, 135, 145, 147, 158]    # amendment CLR-1b
if len(sys.argv) > 1 and sys.argv[1] == "pooled":
    H = H + H_B
elif len(sys.argv) > 1 and sys.argv[1] == "b":
    H = H_B


def main():
    X = json.load(open("theory/results/exp20_clr_eval.json"))
    comp = lambda r: ev(r) is not None and ev(r)["gate"]
    far = lambda r: float(np.mean(X[r]["two_seg"][7:10]))
    pairs = [(far(f"pclr3_s{s}"), far(f"pclr1_s{s}")) for s in H
             if f"pclr3_s{s}" in X and f"pclr1_s{s}" in X and comp(f"pclr3_s{s}") and comp(f"pclr1_s{s}")]
    c3 = [f"pclr3_s{s}" for s in H if f"pclr3_s{s}" in X]
    ch = [float(np.mean(X[r]["chain4"][12:18])) for r in c3]
    iia = [X[r]["edit"]["IIA"] for r in c3 if X[r]["edit"]["IIA"] is not None]
    drops = []
    for s in H:
        r = f"pclr3_s{s}"
        f0 = f"results/competence_p30_s{s}.json"
        if ev(r) and os.path.exists(f0):
            c0 = json.load(open(f0)); c0 = c0[0] if isinstance(c0, list) else c0
            drops.append(c0["budgets"][str(c0["config"]["K"])]["delayed"]["bal_acc"][0] - ev(r)["delayed"])
    out = {"n_clr3": len(c3), "n_clr1": sum(f"pclr1_s{s}" in X for s in H),
           "competent": {"clr3": sum(comp(f"pclr3_s{s}") for s in H), "clr1": sum(comp(f"pclr1_s{s}") for s in H)},
           "PC1": pair_rule(pairs, lambda x, y: x > y),
           "PC2": {"chain4_d13_18_ge_0.60": sum(x >= 0.60 for x in ch), "of": len(ch), "median": float(np.median(ch)) if ch else None,
                   "met": sum(x >= 0.60 for x in ch) >= len(ch) / 2},
           "PC3": {"median_IIA": float(np.median(iia)) if iia else None, "met": bool(iia) and float(np.median(iia)) >= 0.30},
           "PC4": {"median_delayed_drop": float(np.median(drops)) if drops else None, "met": bool(drops) and float(np.median(drops)) <= 0.02},
           "desc_clr1": {"median_two_seg_far": float(np.median([far(f"pclr1_s{s}") for s in H if f"pclr1_s{s}" in X])) if X else None,
                         "median_chain4_d13_18": float(np.median([np.mean(X[f"pclr1_s{s}"]["chain4"][12:18]) for s in H if f"pclr1_s{s}" in X])) if X else None,
                         "median_IIA": float(np.median([X[f"pclr1_s{s}"]["edit"]["IIA"] for s in H if f"pclr1_s{s}" in X and X[f"pclr1_s{s}"]["edit"]["IIA"] is not None])) if X else None}}
    sh = [X[r]["anchor_maxshare"] for r in c3 if "anchor_maxshare" in X[r]]
    dr = [float(np.mean(X[r]["two_seg"][7:10]) - np.mean(X[r]["two_seg_rand"][7:10])) for r in c3 if "two_seg_rand" in X[r]]
    used = sum(x < 0.8 for x in sh) >= 0.6 * len(sh) and bool(dr) and float(np.median(dr)) >= 0.02
    out["CLR1a"] = {"maxshare_lt_0.8": sum(x < 0.8 for x in sh), "of": len(sh), "median_maxshare": float(np.median(sh)) if sh else None,
                    "median_two_seg_minus_rand": float(np.median(dr)) if dr else None, "set_used": used}
    if out["PC1"]["verdict"] == "supported" and not used:
        out["PC1"]["reading"] = "gain without set use"
    tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
    json.dump(out, open(f"theory/results/score_s45{tag}.json", "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
