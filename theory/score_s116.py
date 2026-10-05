"""THEORY §116 scoring (rules fixed in §116; written before any §116 run; Theory is the only scorer).
Runs: boxes/runs_s5/s116_opdef_s{31..35}/eval.json and s116_none_s{41..45}/eval.json. Keys as §112
(cumulative, extension_2048, k1_never_labelled_acc, k1_certified, coverage_drawn, donor, removal) plus
foothold_yield / acc_by_length per round (descriptive).
FOOTHOLD-OPDEF: >= 2/5 seeds with cumulative >= 0.99 at 512 AND k1_never_labelled_acc >= 0.99.
STRICT-C4 (NONE): >= 2/5 seeds with cumulative >= 0.99 at 512.
C1 tag: single-patch donor SERIAL (every in-chunk slot donor >= 0.90, recipient < 0.10). C6: seed rate; >= 0.99 at 2,048.
    python theory/score_s116.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"OPDEF": ("s116_opdef", range(31, 36)), "NONE": ("s116_none", range(41, 46))}


def serial(r):
    d = r.get("donor") or {}
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values())


def main():
    res = {"arms": {}}
    for arm, (pre, seeds) in ARMS.items():
        runs = {}
        for s in seeds:
            p = os.path.join(ROOT, "boxes", "runs_s5", "%s_s%d" % (pre, s), "eval.json")
            if os.path.exists(p):
                runs[s] = json.load(open(p))
        if arm == "OPDEF":
            ok = [s for s, r in runs.items() if r["cumulative"]["512"] >= 0.99 and (r.get("k1_never_labelled_acc") or 0) >= 0.99]
        else:
            ok = [s for s, r in runs.items() if r["cumulative"]["512"] >= 0.99]
        A = {"n": len(runs), "pass_seeds": ok, "PASS": len(ok) >= 2,
             "cum512": {s: r["cumulative"]["512"] for s, r in runs.items()},
             "ext2048": {s: (r.get("extension_2048") or {}).get("cumulative", {}).get("2048") for s, r in runs.items()},
             "k1_never": {s: r.get("k1_never_labelled_acc") for s, r in runs.items()},
             "certified": {s: r.get("k1_certified") for s, r in runs.items()},
             "coverage_drawn": {s: r.get("coverage_drawn") for s, r in runs.items()},
             "serial_seeds": [s for s in ok if serial(runs[s])],
             "c6_2048_seeds": [s for s in ok if ((runs[s].get("extension_2048") or {}).get("cumulative", {}).get("2048") or 0) >= 0.99]}
        res["arms"][arm] = A
        print("%-5s n %d PASS %s seeds %s | serial %s | 2048 %s | cum512 %s | k1 never %s | cert %s | cov %s" % (
            arm, A["n"], A["PASS"], ok, A["serial_seeds"], A["c6_2048_seeds"], A["cum512"], A["k1_never"], A["certified"], A["coverage_drawn"]))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s116.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
