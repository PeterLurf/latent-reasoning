"""THEORY §107 scoring (rules fixed in §107; written before any §107 implementation or run; Theory is the only scorer).
Runs: boxes/runs_mult/a107_{lat2,lat4,pause4}_4x4_s{1,2,3}/eval.json (§99 keys: per_slot_decode_acc (test), test_acc,
removal.{zero,mean}.answer_acc, steps_to_threshold, ...). Control m = 1: §99's a99_markov_4x4_s*; CTRL: a99_ctrl_4x4_s*.
CARRY-3: per_slot_decode_acc[2] (op 3) >= 0.95 in >= 2/3 seeds.
NO-BYPASS: in those seeds zero- and mean-removal answer accuracy <= §98 NC-M 4x4 median + 0.05.
    python theory/score_s107.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "boxes", "runs_mult")


def load(prefix):
    out = {}
    for s in (1, 2, 3):
        f = os.path.join(RUNS, "%s_4x4_s%d" % (prefix, s), "eval.json")
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def main():
    nc = [json.load(open(f))["acc_test"] for f in sorted(glob.glob(os.path.join(RUNS, "m98_nc_4x4_s*", "eval.json")))]
    floor = float(np.median(nc)) + 0.05 if nc else None
    res = {"nc_floor_plus_0.05": floor, "arms": {}}
    for arm, prefix in (("m=1 (§99 MARKOV)", "a99_markov"), ("LAT-2", "a107_lat2"), ("LAT-4", "a107_lat4"), ("PAUSE-4", "a107_pause4")):
        rs = load(prefix)
        op3 = {s: r["per_slot_decode_acc"][2] for s, r in rs.items()}
        carry = [s for s, v in op3.items() if v >= 0.95]
        nob = floor is not None and bool(carry) and all(rs[s]["removal"]["zero"]["answer_acc"] <= floor and rs[s]["removal"]["mean"]["answer_acc"] <= floor for s in carry)
        A = {"n": len(rs), "op3": op3, "carry_seeds": carry, "CARRY_3": len(carry) >= 2, "NO_BYPASS": nob,
             "test_acc": {s: r.get("test_acc") for s, r in rs.items()},
             "steps": {s: r.get("steps_to_threshold") for s, r in rs.items()},
             "removal": {s: {m: r["removal"][m]["answer_acc"] for m in ("zero", "mean")} for s, r in rs.items()}}
        res["arms"][arm] = A
        print("%-17s n %d CARRY-3 %s NO-BYPASS %s | op3 %s | test %s | steps %s" % (arm, A["n"], A["CARRY_3"], nob,
              {s: round(v, 3) for s, v in op3.items()}, A["test_acc"], A["steps"]))
    L = res["arms"]; lat = L["LAT-2"]["CARRY_3"] or L["LAT-4"]["CARRY_3"]; pause = L["PAUSE-4"]["CARRY_3"]
    res["reading"] = ("both: compute suffices" if lat and pause else "carrier width" if lat else "compute per op" if pause else "neither")
    print("reading:", res["reading"], "| floor", floor)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s107.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
