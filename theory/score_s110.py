"""THEORY §110 scoring (rules fixed in §110; written before any §110 implementation or run; Theory is the only scorer).
Runs: boxes/runs_s5/s110_s{1..5}/eval.json (independent seeds) and, if run, s110_student_s{1..5}. Keys as §103 plus
removal.{zero,mean}.acc_after_first_ablated_slot and donor {"slot<j>": {donor_consistent, recipient_consistent}} (§104
protocol, in-chunk slots), short_fraction, rounds.
SELF2: cumulative >= 0.99 at 512 in >= 4/5.  C1: in SELF2 seeds, mean-removal <= 0.06 and donor USE >= 0.90 with
recipient < 0.10 at every in-chunk slot, in >= 4/5 seeds overall.  C6: cumulative >= 0.99 at 2,048 in >= 4/5.
STUDENT (conditional): the same rules on s110_student_s*.
§110b (corrected; identity-start shorts, random textual starts for length >= 3): the same rules on s110b_s* / s110b_student_s*.
    python theory/score_s110.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score(prefix):
    runs = {}
    for s in range(1, 6):
        f = os.path.join(ROOT, "boxes", "runs_s5", "%s_s%d" % (prefix, s), "eval.json")
        if os.path.exists(f):
            r = json.load(open(f)); assert r.get("short_fraction", 0) <= 0.05 + 1e-9, f
            runs[s] = r
    cum = {s: r["cumulative"]["512"] for s, r in runs.items()}
    self2 = [s for s, v in cum.items() if v >= 0.99]
    def c1ok(r):
        d = r.get("donor") or {}
        return (r["removal"]["mean"]["acc_after_first_ablated_slot"] <= 0.06 and bool(d) and
                all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values()))
    c1 = [s for s in self2 if c1ok(runs[s])]
    ext = {s: (r.get("extension_2048", {}).get("cumulative", {}).get("2048") or 0) for s, r in runs.items()}
    A = {"n": len(runs), "cum512": cum, "median512": float(np.median(list(cum.values()))) if cum else None,
         "range512": [min(cum.values()), max(cum.values())] if cum else None, "ext2048": ext,
         "SELF2": len(self2) >= 4, "self2_seeds": self2, "C1": len(c1) >= 4, "c1_seeds": c1,
         "C6": sum(v >= 0.99 for v in ext.values()) >= 4,
         "mean_removal": {s: r["removal"]["mean"]["acc_after_first_ablated_slot"] for s, r in runs.items()},
         "rounds": {s: r.get("rounds") for s, r in runs.items()}}
    A["student_trigger"] = (not A["SELF2"]) and sum(v >= 0.90 for v in cum.values()) >= 3
    return A


def main():
    res = {"main": score("s110"), "student": score("s110_student"), "s110b": score("s110b"), "s110b_student": score("s110b_student")}
    for k, A in res.items():
        print(k, {x: A[x] for x in ("n", "median512", "range512", "SELF2", "self2_seeds", "C1", "c1_seeds", "C6", "student_trigger")})
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s110.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
