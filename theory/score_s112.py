"""THEORY §112 scoring (rules fixed in §112; written before any §112 run; Theory is the only scorer).
Runs: boxes/runs_s5/s112_f{050,025,010}_s{21..25}/eval.json. Keys as §110 plus k1_never_labelled_acc,
k1_labelled_acc, k1_certified (all 14,400), coverage_realised, k1_never_labelled_in_pool / _not_in_pool, per_round_split.
FOOTHOLD-f: >= 2/5 seeds with cumulative >= 0.99 at 512 AND k1_never_labelled_acc >= 0.99.
Primary reading: the smallest f meeting FOOTHOLD-f.
    python theory/score_s112.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSES = {"0.5": "f050", "0.25": "f025", "0.1": "f010"}


def main():
    res = {"doses": {}}
    for f, tag in DOSES.items():
        runs = {}
        for s in range(21, 26):
            p = os.path.join(ROOT, "boxes", "runs_s5", "s112_%s_s%d" % (tag, s), "eval.json")
            if os.path.exists(p):
                runs[s] = json.load(open(p))
        ok = [s for s, r in runs.items() if r["cumulative"]["512"] >= 0.99 and (r.get("k1_never_labelled_acc") or 0) >= 0.99]
        D = {"n": len(runs), "pass_seeds": ok, "FOOTHOLD": len(ok) >= 2,
             "cum512": {s: r["cumulative"]["512"] for s, r in runs.items()},
             "k1_never": {s: r.get("k1_never_labelled_acc") for s, r in runs.items()},
             "certified": {s: r.get("k1_certified") for s, r in runs.items()},
             "coverage": {s: r.get("coverage_realised") for s, r in runs.items()}}
        res["doses"][f] = D
        print("f %-4s n %d FOOTHOLD %s seeds %s | cum512 %s | k1 never %s | cert %s" % (f, D["n"], D["FOOTHOLD"], ok, D["cum512"], D["k1_never"], D["certified"]))
    passed = [float(f) for f in DOSES if res["doses"][f]["FOOTHOLD"]]
    res["smallest_f"] = min(passed) if passed else None
    print("smallest f meeting FOOTHOLD:", res["smallest_f"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s112.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
