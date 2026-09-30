"""Score THEORY §39 (Pilot T12: PT1-PT4) as registered; pairs with pd_at_s900-907.
    python theory/score_pilotT12.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pilotFG import ev, pair_rule  # noqa: E402

SEEDS = range(900, 908)


def main():
    SD = json.load(open("theory/results/step_decode.json"))
    P = json.load(open("theory/results/le_pointer.json"))
    M = json.load(open("theory/results/pilotDE_meas.json"))
    ptr = lambda r: P[r]["ptr"] if r in P else (M[r]["ptr"] if r in M and "ptr" in M[r] else None)
    comp = lambda r: ev(r) is not None and ev(r)["gate"]
    T, A = {s: f"pt12_s{s}" for s in SEEDS}, {s: f"pd_at_s{s}" for s in SEEDS}
    both = [s for s in SEEDS if comp(T[s]) and comp(A[s])]
    out = {"competent": {"T12": sum(comp(T[s]) for s in SEEDS), "AT": sum(comp(A[s]) for s in SEEDS)},
           "runs": {T[s]: ({"gate": ev(T[s])["gate"], "extrap": ev(T[s])["extrap"], "d7": json.load(open(f"theory/runs/{T[s]}/eval.json"))["d7_K12"],
                            "d8": ev(T[s])["d8"], "delayed": ev(T[s])["delayed"], "step_acc": SD.get(T[s], {}).get("step_acc"),
                            "ptr": ptr(T[s])} if ev(T[s]) else "stage-1 guard exhausted (run, not competent)") for s in SEEDS}}
    out["PT1"] = pair_rule([(SD[T[s]]["step_acc"], SD[A[s]]["step_acc"]) for s in both if T[s] in SD and A[s] in SD], lambda x, y: x < y)
    pt = [max(ptr(T[s])) >= 8 for s in SEEDS if comp(T[s]) and ptr(T[s])]
    pa = [max(ptr(A[s])) >= 8 for s in SEEDS if comp(T[s]) and comp(A[s]) and ptr(A[s])]
    sT = float(np.mean(pt)) if pt else None
    sA = float(np.mean(pa)) if pa else None
    out["PT2"] = {"T12_pass8": [int(sum(pt)), len(pt)], "AT_same_seeds_pass8": [int(sum(pa)), len(pa)],
                  "met": sT is not None and sA is not None and sT >= 0.50 and sT > sA}
    out["PT3"] = pair_rule([(ev(T[s])["d8"], ev(A[s])["d8"]) for s in both], lambda x, y: x > y)
    out["PT4"] = {"T12": out["competent"]["T12"], "AT": out["competent"]["AT"],
                  "met": out["competent"]["T12"] >= out["competent"]["AT"] - 2}
    json.dump(out, open("theory/results/score_pilotT12.json", "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
