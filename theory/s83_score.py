"""THEORY §83 scoring (rules fixed in §83, notes 83-1 / 83-3): self-gated restarts on retrained WC0_4 solvers.
Included seeds: fixed k = 4 reaches cumulative 1.000 at 512 (§71's own check; 83-1). U1: adaptive cumulative >= 0.99 at
512 in every included seed (>= 2 included) and fixed k = 16 median < 0.5. U2: pooled detection, first wrong states with
confidence < tau >= 0.99. U3 (83-1): adaptive cum@512 >= fixed k = 4 - 0.005 and mean accepted chunk > 4 in every included
seed. Primary files theory/results/s83/wc0k4r_s<seed>.json; *_ext.json (2,048 extension, sweep) descriptive.
    python theory/s83_score.py [--dir theory/results/s83] [--out theory/results/score_s83.json]
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="theory/results/s83"); ap.add_argument("--out", default="theory/results/score_s83.json")
    a = ap.parse_args(); runs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "wc0k4r_s*.json"))):
        if f.endswith("_ext.json"):
            continue
        runs[os.path.basename(f)[:-5]] = json.load(open(f))
    inc = {k: r for k, r in runs.items() if r["fixed_k4"]["cum"]["512"] >= 1.0}
    out = {"seeds": {}, "included": sorted(inc)}
    for k, r in runs.items():
        d = r["adaptive"]["detection"]
        out["seeds"][k] = {"included": k in inc, "tau": r["tau"], "adaptive_cum512": r["adaptive"]["cum"]["512"],
                           "fixed_k4_cum512": r["fixed_k4"]["cum"]["512"], "fixed_k16_cum512": r["fixed_k16"]["cum"]["512"],
                           "mean_accepted": r["adaptive"]["mean_accepted"], "fallbacks": d.get("fallbacks"),
                           "miss_rate": (d["wrong"] - d["wrong_flagged"]) / d["wrong"] if d["wrong"] else None,
                           "false_restart_rate": d["right_flagged"] / d["right"] if d["right"] else None,
                           "in_range_acc": r["calibration"]["in_range_acc"]}
        print(k, out["seeds"][k])
    ok = len(inc) >= 2
    W = sum(r["adaptive"]["detection"]["wrong"] for r in inc.values()); WF = sum(r["adaptive"]["detection"]["wrong_flagged"] for r in inc.values())
    out["U1"] = ok and all(r["adaptive"]["cum"]["512"] >= 0.99 for r in inc.values()) and \
        float(np.median([r["fixed_k16"]["cum"]["512"] for r in inc.values()])) < 0.5
    out["U2"] = ok and W > 0 and WF / W >= 0.99
    out["U2_pooled"] = {"wrong_first_states": W, "flagged": WF}
    out["U3"] = ok and all(r["adaptive"]["cum"]["512"] >= r["fixed_k4"]["cum"]["512"] - 0.005 and r["adaptive"]["mean_accepted"] > 4
                           for r in inc.values())
    out["informative"] = ok
    print("included", sorted(inc), "| U1", out["U1"], "| U2", out["U2"], out["U2_pooled"], "| U3", out["U3"])
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
