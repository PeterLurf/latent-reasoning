"""Score THEORY §26 RA1–RA4 from theory/results/exp12_reanchor.json (59 P30 ∪ L models) and exp5_block.json.
    python theory/score_reanchor.py
"""
import json

import numpy as np

R = json.load(open("theory/results/exp12_reanchor.json"))
B = json.load(open("theory/results/exp5_block.json"))
names = sorted(n for n in R if n in B)
ok = lambda v: v[6] >= 0.60                                    # distance 7 (index 6: distances 1..12)
out = {"n": len(names)}
for key in ("single", "re2", "re3", "re2top3"):
    out[key] = {"7hop_success": sum(ok(R[n][key]) for n in names),
                "mean_far8_10": float(np.mean([np.mean(R[n][key][7:10]) for n in names])),
                "mean_near1_6": float(np.mean([np.mean(R[n][key][:6]) for n in names]))}
out["block_7hop_success"] = sum(B[n]["bL2@K"]["far"][0] >= 0.60 for n in names)
resist = [n for n in names if B[n]["base"]["far"][0] < 0.60 and B[n]["bL2@K"]["far"][0] < 0.60]
out["RA1"] = {"re2top3": out["re2top3"]["7hop_success"], "bar": 41, "met": out["re2top3"]["7hop_success"] >= 41}
share = sum(ok(R[n]["re2top3"]) for n in resist) / len(resist)
out["RA2"] = {"block_resistant": len(resist), "rescued_by_re2top3": sum(ok(R[n]["re2top3"]) for n in resist),
              "share": share, "met": share >= 0.40, "rescued": [n for n in resist if ok(R[n]["re2top3"])]}
d = out["re3"]["mean_far8_10"] - out["single"]["mean_far8_10"]
out["RA3"] = {"re3_minus_single_far8_10": d, "met": d >= 0.05}
dn = out["re2top3"]["mean_near1_6"] - out["single"]["mean_near1_6"]
dn3 = out["re3"]["mean_near1_6"] - out["single"]["mean_near1_6"]
out["RA4"] = {"near_change_re2top3": dn, "near_change_re3": dn3, "met": dn >= -0.02 and dn3 >= -0.02}
json.dump(out, open("theory/results/score_reanchor.json", "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
