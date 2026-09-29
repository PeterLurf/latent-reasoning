import json
import sys

import numpy as np

split = sys.argv[1] if len(sys.argv) > 1 else "discovery"
for s, run in [("s0", "runs/r3_pc_gated_gpt2"), ("s1", "runs/main_s1"), ("s2", "runs/main_s2")]:
    P = json.load(open(f"{run}/followup/{split}_pair/summary.json"))["results"]
    for W, R in P.items():
        p = R["pair"]
        print(s, "W" + W, "n", p["n"],
              "both: f1 %+.3f f2 %+.3f | only1: f1 %+.3f f2 %+.3f | only2: f1 %+.3f f2 %+.3f | add1 %+.3f add2 %+.3f" % (
                  p["both"]["follow1"][0], p["both"]["follow2"][0], p["only1"]["follow1"][0], p["only1"]["follow2"][0],
                  p["only2"]["follow1"][0], p["only2"]["follow2"][0], p["additivity_branch1"][0], p["additivity_branch2"][0]))
    G = json.load(open(f"{run}/followup/{split}_gainfit/summary.json"))["results"]
    print(s, "mean |diff| by W:", {W: round(float(np.mean(R["gainfit"]["diff_norm"])), 2) for W, R in G.items()})
