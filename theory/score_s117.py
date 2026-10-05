"""THEORY §117 scoring (rules fixed in §117; written before any §117 run; Theory is the only scorer).
Runs: boxes/runs_s5/s117_mk_s{51..55}/eval.json and s117_st_s{61..65}/eval.json. Keys as §112 (cumulative,
extension_2048, k1_never_labelled_acc, donor (single patch), ...) plus switch_step (ST).
ALL (per seed): cumulative >= 0.99 at 512 AND single-slot donor SERIAL (every in-chunk slot donor >= 0.90, recipient < 0.10)
AND k1_never_labelled_acc >= 0.99.
INTEGRATED: ALL in >= 4/5 (extended arm).  EXIST: ALL in >= 1 seed.  C6: ALL + >= 0.99 at 2,048 in >= 4/5.
Extension: ALL in >= 2/3 of the first three seeds.  Sanity: cumulative >= 0.99 with SERIAL failing -> LEAK flag.
    python theory/score_s117.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"MK": ("s117_mk", list(range(51, 56))), "ST": ("s117_st", list(range(61, 66)))}


def serial(r):
    d = r.get("donor") or {}
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values())


def ext(r):
    return ((r.get("extension_2048") or {}).get("cumulative") or {}).get("2048") or 0


def main():
    res = {"arms": {}}
    for arm, (pre, seeds) in ARMS.items():
        runs = {}
        for s in seeds:
            p = os.path.join(ROOT, "boxes", "runs_s5", "%s_s%d" % (pre, s), "eval.json")
            if os.path.exists(p):
                runs[s] = json.load(open(p))
        acc = {s: r["cumulative"]["512"] >= 0.99 for s, r in runs.items()}
        allp = [s for s, r in runs.items() if acc[s] and serial(r) and (r.get("k1_never_labelled_acc") or 0) >= 0.99]
        first3 = [s for s in seeds[:3] if s in allp]
        A = {"n": len(runs), "ALL_seeds": allp, "EXIST": len(allp) >= 1, "INTEGRATED": len(allp) >= 4,
             "C6_2048": len([s for s in allp if ext(runs[s]) >= 0.99]) >= 4,
             "extend": len(first3) >= 2 if all(s in runs for s in seeds[:3]) else None,
             "LEAK": [s for s, r in runs.items() if acc[s] and not serial(r)],
             "cum512": {s: r["cumulative"]["512"] for s, r in runs.items()}, "ext2048": {s: ext(r) for s, r in runs.items()},
             "k1_never": {s: r.get("k1_never_labelled_acc") for s, r in runs.items()},
             "switch_step": {s: r.get("switch_step") for s, r in runs.items()}}
        res["arms"][arm] = A
        print("%-2s n %d ALL %s EXIST %s INTEGRATED %s C6 %s extend %s LEAK %s | cum512 %s | k1 never %s | switch %s" % (
            arm, A["n"], allp, A["EXIST"], A["INTEGRATED"], A["C6_2048"], A["extend"], A["LEAK"], A["cum512"], A["k1_never"], A["switch_step"]))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s117.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
