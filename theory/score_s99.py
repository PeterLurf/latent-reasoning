"""THEORY §99 scoring (rules fixed in §99; written before any §99 implementation or run; Theory is the only scorer).
Runs: boxes/runs_mult/a99_{ctrl,kcurr,std,textfirst,textfirst_mask,tfslot,combo}_4x4_s{1,2,3}/eval.json (note 99-1).
steps_to_threshold must count only eligible logs (note 99-1: final configuration; k = 4; gold share 0).
steps_to_threshold: first 1k-step valid-split log with every slot decode >= 0.95 and answer acc >= 0.95; null = censored
(counted as 24,000).
ACCELERATES (per non-CTRL arm): median steps-to-threshold <= 0.5 x CTRL's median (if CTRL's median is censored, <= 12,000)
AND, in the arm's threshold-reaching seeds, zero- and mean-removal answer accuracy <= NC floor + 0.05, with the NC floor =
§98's NC-M 4x4 median (boxes/runs_mult/m98_nc_4x4_s*/eval.json acc_test).
    python theory/score_s99.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "boxes", "runs_mult")
CAP = 24000
ARMS = ("ctrl", "kcurr", "std", "textfirst", "textfirst_mask", "tfslot", "combo", "markov")


def load(arm):
    out = {}
    for s in (1, 2, 3):
        f = os.path.join(RUNS, "a99_%s_4x4_s%d" % (arm, s), "eval.json")
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def main():
    nc = [json.load(open(f))["acc_test"] for f in sorted(glob.glob(os.path.join(RUNS, "m98_nc_4x4_s*", "eval.json")))]
    floor = float(np.median(nc)) + 0.05 if nc else None
    A = {a: load(a) for a in ARMS}
    stt = {a: {s: (r["steps_to_threshold"] if r.get("steps_to_threshold") is not None else CAP) for s, r in rs.items()} for a, rs in A.items()}
    med = {a: (float(np.median(list(v.values()))) if v else None) for a, v in stt.items()}
    ctrl = med["ctrl"]
    bar = None if ctrl is None else (12000.0 if ctrl >= CAP else 0.5 * ctrl)
    res = {"nc_floor_plus_0.05": floor, "nc_m_4x4": nc, "ctrl_median_steps": ctrl, "bar_steps": bar, "arms": {}}
    for a in ARMS:
        rs = A[a]
        reached = [s for s, r in rs.items() if r.get("steps_to_threshold") is not None]
        nobypass = floor is not None and all(rs[s]["removal"]["zero"]["answer_acc"] <= floor and rs[s]["removal"]["mean"]["answer_acc"] <= floor
                                             for s in reached)
        row = {"n": len(rs), "steps": stt[a], "median_steps": med[a], "reached": reached,
               "test_acc": {s: r.get("test_acc") for s, r in rs.items()},
               "removal": {s: {m: r["removal"][m]["answer_acc"] for m in ("zero", "mean")} for s, r in rs.items() if "removal" in r},
               "train_flops_to_stop": {s: r.get("training", {}).get("flops") for s, r in rs.items()},
               "no_bypass": nobypass}
        if a != "ctrl":
            row["accelerates"] = (bar is not None and med[a] is not None and med[a] <= bar and len(rs) == 3 and nobypass)
        res["arms"][a] = row
        print("%-14s n %d median steps %s | steps %s | test %s | removal %s | no-bypass %s | accelerates %s" % (
            a, row["n"], row["median_steps"], row["steps"], row["test_acc"], row["removal"], nobypass, row.get("accelerates")))
    km, mm = med.get("kcurr"), med.get("markov")
    res["H_starve_markov_earlier_than_kcurr"] = (km is not None and mm is not None and mm < km)
    print("CTRL median", ctrl, "| bar", bar, "| NC floor+0.05", floor, "| H-starve (markov < kcurr)", res["H_starve_markov_earlier_than_kcurr"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s99.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
