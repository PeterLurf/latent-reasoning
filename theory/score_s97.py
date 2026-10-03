"""THEORY §97 scoring (rules fixed in §97 and note 97-1; written before any §97 implementation or run; Theory is the only
scorer).
Runs: boxes/runs_s5/g97_{gen4,gen1,hyb4}_s{1..5}/eval.json.
G4 (the only primary rule, per arm at k = 4: GEN-4 and HYB-4): cumulative >= 0.99 at 512 in >= 2/3 of seeds 1-3 AND
    zero-removal accuracy after the first ablated slot <= 0.10 in those seeds.
C3 (as §86 R3): median FLOPs per 512-problem <= 0.6 x the retrained §71 WC0_4 (wc0k4r_s*).
C6 (if an arm passes G4): >= 4/5 of seeds 1-5 at cumulative >= 0.99 at 512.
GEN-1 is descriptive (certificate reported). The attribution table of note 97-1 uses §92 (l92_k4_s1-3, rule as G4).
Descriptive: 2,048 extension, mean removal, per-slot decode accuracy, checkpoint validity, wte drift, edit IIA.
    python theory/score_s97.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "boxes", "runs_s5")


def load(prefix):
    out = {}
    for s in range(1, 6):
        f = os.path.join(RUNS, "%s_s%d" % (prefix, s), "eval.json")
        if os.path.exists(f):
            r = json.load(open(f))
            rem = r.get("removal", {})
            out[s] = {"cum512": r["cumulative"]["512"], "ext2048": r.get("extension_2048", {}).get("cumulative", {}).get("2048"),
                      "zero_after": rem.get("zero", {}).get("acc_after_first_ablated_slot"),
                      "mean_after": rem.get("mean", {}).get("acc_after_first_ablated_slot"),
                      "per_slot": r.get("chunk_check", {}).get("per_slot_acc"), "flops": r["compute_512_per_problem"]["flops"],
                      "certificate": r.get("certificate"), "validity": r.get("checkpoint_validity"),
                      "wte_drift": r.get("wte_drift"), "edit_iia": r.get("edit_iia")}
    return out


def g4(runs):
    first3 = {s: v for s, v in runs.items() if s <= 3}
    passing = [s for s, v in first3.items() if v["cum512"] >= 0.99 and v["zero_after"] is not None and v["zero_after"] <= 0.10]
    return {"n_seeds_1_3": len(first3), "pass_seeds": passing, "G4": len(passing) >= 2}


def main():
    wc = [json.load(open(f))["compute_512_per_problem"]["flops"] for f in sorted(glob.glob(os.path.join(RUNS, "wc0k4r_s*", "eval.json")))]
    wc_med = float(np.median(wc)) if wc else None
    res = {"wc0k4r_flops_median": wc_med, "arms": {}}
    for arm, prefix in (("GEN-4", "g97_gen4"), ("HYB-4", "g97_hyb4"), ("GEN-1", "g97_gen1"), ("§92 k4", "l92_k4")):
        runs = load(prefix); a = {"runs": runs}
        if arm != "GEN-1":
            a.update(g4(runs))
            fl = [v["flops"] for v in runs.values()]
            a["flops_ratio_vs_wc0k4r"] = float(np.median(fl) / wc_med) if fl and wc_med else None
            a["C3"] = a["flops_ratio_vs_wc0k4r"] is not None and a["flops_ratio_vs_wc0k4r"] <= 0.6
            if len(runs) == 5:
                a["C6"] = sum(v["cum512"] >= 0.99 for v in runs.values()) >= 4
        res["arms"][arm] = a
        print("%-7s n %d %s | cum512 %s | zero_after %s | ext2048 %s | flops ratio %s | cert %s | validity %s" % (
            arm, len(runs), {k: a[k] for k in ("pass_seeds", "G4", "C3", "C6") if k in a},
            {s: round(v["cum512"], 4) for s, v in runs.items()},
            {s: v["zero_after"] for s, v in runs.items()}, {s: v["ext2048"] for s, v in runs.items()},
            a.get("flops_ratio_vs_wc0k4r"), {s: v["certificate"] for s, v in runs.items()}, {s: v["validity"] for s, v in runs.items()}))
    A = res["arms"]
    key = (A["§92 k4"].get("G4"), A["HYB-4"].get("G4"), A["GEN-4"].get("G4"))
    table = {(True, True, True): "no hand-built state is needed",
             (True, True, False): "the text checkpoint (snap) is the problem",
             (True, False, False): "the text target (head) is the problem; the snap untested",
             (True, False, True): "reported as found; no reading fixed in 97-1"}
    res["attribution"] = {"key_92_hyb4_gen4": key, "reading": table.get(key, "outside 97-1's table")}
    print("attribution", res["attribution"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s97.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
