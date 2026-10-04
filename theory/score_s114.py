"""THEORY §114 scoring (rules fixed in §114; written before any §114 implementation or run; Theory is the only scorer).
Runs: boxes/runs_mult/a114_{loopr,loop1}_4x4_s{1,2,3}/eval.json (accel99 keys + donor + early_escape + acc_by_r).
PAUSE-4 comparator: boxes/runs_mult/a107_pause4_4x4_s*/eval.json.
CARRY-3: per_slot_decode_acc[2] >= 0.95 in >= 2/3.  C1-donor: in those seeds, every in-chunk donor slot donor >= 0.90 and
recipient < 0.10, and zero/mean removal answer acc <= §98 NC-M 4x4 median + 0.05.  DEPTH: LOOP-R meets CARRY-3 + C1-donor,
LOOP-1 does not, PAUSE-4 did not.  No-read seeds (early_escape false) count as fails; also reported among escaped seeds.
    python theory/score_s114.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_mult")


def load(prefix):
    out = {}
    for s in (1, 2, 3):
        f = os.path.join(R, "%s_4x4_s%d" % (prefix, s), "eval.json")
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def main():
    nc = [json.load(open(f))["acc_test"] for f in sorted(glob.glob(os.path.join(R, "m98_nc_4x4_s*", "eval.json")))]
    floor = float(np.median(nc)) + 0.05 if nc else None
    res = {"floor": floor, "arms": {}}
    for arm, prefix in (("LOOP-R", "a114_loopr"), ("LOOP-1", "a114_loop1"), ("PAUSE-4 (§107)", "a107_pause4")):
        rs = load(prefix)
        op3 = {s: r["per_slot_decode_acc"][2] for s, r in rs.items()}
        carry = [s for s, v in op3.items() if v >= 0.95]
        def c1(r):
            d = r.get("donor") or {}
            return (bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values())
                    and floor is not None and r["removal"]["zero"]["answer_acc"] <= floor and r["removal"]["mean"]["answer_acc"] <= floor)
        ok = [s for s in carry if c1(rs[s])]
        esc = [s for s, r in rs.items() if r.get("early_escape")]
        A = {"n": len(rs), "op3": op3, "carry_seeds": carry, "CARRY_3": len(carry) >= 2, "c1_seeds": ok, "C1_donor": len(ok) >= 2,
             "escaped_seeds": esc, "noread_seeds": [s for s in rs if s not in esc],
             "among_escaped_carry": [s for s in carry if s in esc], "test_acc": {s: r.get("test_acc") for s, r in rs.items()},
             "acc_by_r": {s: r.get("acc_by_r") for s, r in rs.items()}}
        res["arms"][arm] = A
        print("%-15s n %d CARRY-3 %s C1 %s | op3 %s | escaped %s | test %s" % (arm, A["n"], A["CARRY_3"], A["C1_donor"],
              {s: round(v, 3) for s, v in op3.items()}, esc, A["test_acc"]))
    L = res["arms"]
    res["DEPTH"] = (L["LOOP-R"]["CARRY_3"] and L["LOOP-R"]["C1_donor"] and not (L["LOOP-1"]["CARRY_3"] and L["LOOP-1"]["C1_donor"])
                    and not L["PAUSE-4 (§107)"]["CARRY_3"])
    print("DEPTH", res["DEPTH"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s114.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
