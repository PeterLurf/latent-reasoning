"""THEORY §124 scoring (rules fixed in §124; written before any §124 run; Theory is the only scorer).
Runs (seeds 26-29, launched together): boxes/runs_s5/s124_base_s{26..29}, s124_copy_s{26..29}, s124_expose_s{26..29}
(eval.json keys as §112: cumulative, k1_never_labelled_acc, per_round_split [{round, never_labelled_acc}], donor).
Acquired: final k1_never_labelled_acc >= 0.99. Acquisition round: first per_round_split entry with never_labelled_acc > 0.5.
MECH-112: COPY 4/4 AND BASE <= 2/4 AND TEXT-EXPOSE <= 2/4.  NOT-112-4: COPY acquires in no more seeds than BASE and not
earlier (median acquisition round).  Otherwise descriptive.  FOOTHOLD (as §112) per arm.
    python theory/score_s124.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_s5")
ORDER = ["R0", "L3", "L4", "L5", "L6", "L7", "L8", "L10", "L12", "L16", "L32", "final"]
SEEDS = range(26, 30)


def rank(name):
    if name is None:
        return 999
    if name in ORDER:
        return ORDER.index(name)
    try:
        return 50 + int(str(name).lstrip("L"))
    except ValueError:
        return 999


def acq_round(r):
    for x in r.get("per_round_split", []):
        if (x.get("never_labelled_acc") or 0) > 0.5:
            return x.get("round")
    return None


def arm(pre):
    runs = {}
    for s in SEEDS:
        p = os.path.join(R, "%s_s%d" % (pre, s), "eval.json")
        if os.path.exists(p):
            runs[s] = json.load(open(p))
    acq = [s for s, r in runs.items() if (r.get("k1_never_labelled_acc") or 0) >= 0.99]
    rounds = {s: acq_round(r) for s, r in runs.items()}
    foot = [s for s, r in runs.items() if r["cumulative"]["512"] >= 0.99 and (r.get("k1_never_labelled_acc") or 0) >= 0.99]
    med = sorted(rank(v) for v in rounds.values())[len(rounds) // 2] if rounds else None
    return {"n": len(runs), "acquired": acq, "acq_rounds": rounds, "median_round_rank": med, "foothold_seeds": foot,
            "FOOTHOLD": len(foot) >= 2, "cum512": {s: r["cumulative"]["512"] for s, r in runs.items()},
            "copy_acc": {s: r.get("copy_acc") for s, r in runs.items()}}


def main():
    A = {k: arm("s124_%s" % k) for k in ("base", "copy", "expose")}
    full = all(A[k]["n"] == 4 for k in A)
    mech = full and len(A["copy"]["acquired"]) == 4 and len(A["base"]["acquired"]) <= 2 and len(A["expose"]["acquired"]) <= 2
    not_ = full and len(A["copy"]["acquired"]) <= len(A["base"]["acquired"]) and (A["copy"]["median_round_rank"] or 999) >= (A["base"]["median_round_rank"] or 999)
    res = {"arms": A, "MECH_112": mech, "NOT_112_4": not_, "descriptive": full and not mech and not not_}
    for k, v in A.items():
        print("%-6s n %d acquired %s rounds %s FOOTHOLD %s" % (k, v["n"], v["acquired"], v["acq_rounds"], v["FOOTHOLD"]))
    print("MECH-112", mech, "| NOT-112-4", not_, "| descriptive", res["descriptive"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s124.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
