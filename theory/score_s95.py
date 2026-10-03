"""THEORY §95 scoring (rules fixed in §95 and note 95-1; written before Theory read any §95 value; Theory is the only scorer).
Runs: boxes/runs_s5/l95_f{005,025,05}_s{1,2,3} (latent86 --k 4 --sup last --foot f, per-batch draw, as §93's --var).
A dose f is SUFFICIENT iff >= 2/3 seeds have cumulative >= 0.99 at 512 AND zero-removal acc after the first ablated slot
<= 0.10. "Little enough for C4" (fixed): f <= 0.05, i.e. the f = 0.05 dose sufficient. Reference f = 0: §93's l93_s4fix (0/3).
Descriptive: 2,048 extension, mean removal, per-slot accuracy (zero-shot on unlabelled slots 1-3), probe / IIA per slot, FLOPs.
    python theory/score_s95.py
"""
import json
import os

DOSES = {0.0: "l93_s4fix", 0.05: "l95_f005", 0.25: "l95_f025", 0.5: "l95_f05"}


def arm(prefix, f):
    rows = {}
    for s in (1, 2, 3):
        p = "boxes/runs_s5/%s_s%d/eval.json" % (prefix, s)
        if os.path.exists(p):
            r = json.load(open(p))
            if f > 0:
                assert abs(float(r.get("foot", -1)) - f) < 1e-9, (p, r.get("foot"))
            rows[s] = {"cum512": r["cumulative"]["512"], "ext2048": r["extension_2048"]["cumulative"]["2048"],
                       "zero_after": r["removal"]["zero"]["acc_after_first_ablated_slot"],
                       "mean_after": r["removal"]["mean"]["acc_after_first_ablated_slot"],
                       "per_slot": r["chunk_check"]["per_slot_acc"], "probe": r.get("probe_acc_per_slot"),
                       "iia": r.get("iia_per_slot"), "flops": r["compute_512_per_problem"]["flops"]}
    passing = [s for s, v in rows.items() if v["cum512"] >= 0.99 and v["zero_after"] <= 0.10]
    return {"f": f, "labels_per_op": 1.0 / (f + 4 * (1 - f)), "n": len(rows), "pass_seeds": passing,
            "sufficient": len(passing) >= 2, "runs": rows}


def main():
    res = {str(f): arm(p, f) for f, p in DOSES.items()}
    res["little_enough_for_C4"] = res["0.05"]["sufficient"]
    res["min_sufficient_f"] = min([f for f in DOSES if res[str(f)]["sufficient"]], default=None)
    for f in DOSES:
        a = res[str(f)]
        print("f %-4s lab/op %.3f  n %d  pass %s  sufficient %s | cum512 %s | ext2048 %s | zero_after %s | per_slot %s" % (
            f, a["labels_per_op"], a["n"], a["pass_seeds"], a["sufficient"],
            {s: round(v["cum512"], 4) for s, v in a["runs"].items()}, {s: round(v["ext2048"], 4) for s, v in a["runs"].items()},
            {s: round(v["zero_after"], 3) for s, v in a["runs"].items()},
            {s: [round(x, 3) for x in v["per_slot"]] for s, v in a["runs"].items()}))
    print("little enough for C4 (f = 0.05 sufficient):", res["little_enough_for_C4"], "| min sufficient f:", res["min_sufficient_f"])
    json.dump(res, open("theory/results/score_s95.json", "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
