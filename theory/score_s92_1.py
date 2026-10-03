"""THEORY note 92-1 scoring (rules fixed in 92-1 / 86-3, written before reading): C6 = k = 4 seeds 1-5 reach cumulative
>= 0.99 at 512 in >= 4/5; R1's k = 1 part = cumulative >= 0.99 at 512 in >= 2/3 and the exhaustive 14,400-input
certificate (certified in every passing seed -> "certified at every length").
    python theory/score_s92_1.py
"""
import json
import os

out = {"k4": {}, "k1": {}}
for s in range(1, 6):
    f = "boxes/runs_s5/l92_k4_s%d/eval.json" % s
    if os.path.exists(f):
        r = json.load(open(f)); out["k4"][s] = {"cum512": r["cumulative"]["512"], "ext2048": r["extension_2048"]["cumulative"]["2048"]}
for s in range(1, 4):
    f = "boxes/runs_s5/l92_k1_s%d/eval.json" % s
    if os.path.exists(f):
        r = json.load(open(f)); c = r["certificate"]
        out["k1"][s] = {"cum512": r["cumulative"]["512"], "ext2048": r["extension_2048"]["cumulative"]["2048"],
                        "cert_inputs": c["inputs"], "cert_errors": c["errors"], "certified": c["certified"],
                        "flops": r["compute_512_per_problem"]["flops"]}
k4_pass = sum(v["cum512"] >= 0.99 for v in out["k4"].values())
k1_pass = sum(v["cum512"] >= 0.99 for v in out["k1"].values())
out["C6_met"] = len(out["k4"]) == 5 and k4_pass >= 4
out["R1_k1_met"] = k1_pass >= 2
out["k1_certified_all_passing"] = all(v["certified"] for v in out["k1"].values() if v["cum512"] >= 0.99) and k1_pass > 0
print("k4", out["k4"], "-> C6", out["C6_met"], "(%d/%d)" % (k4_pass, len(out["k4"])))
print("k1", out["k1"], "-> R1 k1", out["R1_k1_met"], "certified", out["k1_certified_all_passing"])
json.dump(out, open("theory/results/score_s92_1.json", "w", newline="\n"), indent=1, default=float)
