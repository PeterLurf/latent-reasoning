"""Score THEORY §53 (BD1-BD3) from theory/results/exp26_bidirectional.json. Scored by Experiments (handover 2026-09-30).
Per model: mean over distance indices (6-11 = d7-12, 0-5 = d1-6); then the median over the 59 models.
BD1: bidir_bal d7-12 >= 0.70. BD2: bidir_decoy_fp d7-12 <= 0.30. BD3: bidir_bal d1-6 >= 0.85.
    python theory/score_exp26.py
"""
import json

import numpy as np

R = json.load(open("theory/results/exp26_bidirectional.json"))
f = lambda k, lo, hi: [float(np.mean(v[k][lo:hi])) for v in R.values()]
med = lambda xs: float(np.median(xs))
out = {"n": len(R),
       "BD1_bidir_bal_d7_12": med(f("bidir_bal", 6, 12)), "BD2_decoy_fp_d7_12": med(f("bidir_decoy_fp", 6, 12)),
       "BD3_bidir_bal_d1_6": med(f("bidir_bal", 0, 6)),
       "fwd_bal_d7_12": med(f("fwd_bal", 6, 12)), "fwd_bal_d1_6": med(f("fwd_bal", 0, 6)),
       "re2top3_d7_12": med(f("re2top3", 6, 12)),
       "bidir_reach_d7_12": med(f("bidir_reach", 6, 12)),
       "fp_by_distance_median": [float(np.median([v["bidir_decoy_fp"][i] for v in R.values()])) for i in range(12)],
       "models_BD1_and_BD2": sum(1 for v in R.values() if np.mean(v["bidir_bal"][6:12]) >= 0.70 and np.mean(v["bidir_decoy_fp"][6:12]) <= 0.30)}
out["BD1"] = "met" if out["BD1_bidir_bal_d7_12"] >= 0.70 else "not met"
out["BD2"] = "met" if out["BD2_decoy_fp_d7_12"] <= 0.30 else "not met"
out["BD3"] = "met" if out["BD3_bidir_bal_d1_6"] >= 0.85 else "not met"
json.dump(out, open("theory/results/score_exp26.json", "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
