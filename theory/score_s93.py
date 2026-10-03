"""THEORY §93 scoring (rules fixed in §93; written before Theory read any §93 value).
P-S4: SPARSE-4-fix (l93_s4fix_s1-3) cumulative >= 0.99 at 512 in >= 2/3 seeds AND zero-removal acc after the first
      ablated slot <= 0.10 (in those seeds). SPARSE-4-var reported under the same rule.
P-S8: SPARSE-8-fix (l93_s8fix) the same rule. SPARSE-8-var and WARM-SPARSE-8 reported under it.
Controls: DENSE-4 = §92 (l92_k4_s1-3); DENSE-8 = l93_d8 (if DENSE-8 fails, SPARSE-8 is uninformative).
Descriptive: per-slot accuracy (shared head; zero-shot for unsupervised slots), probe and IIA per slot, removal modes,
2,048 extension, FLOPs.
    python theory/score_s93.py
"""
import json
import os

import numpy as np

ARMS = {"DENSE-4 (§92)": "l92_k4", "SPARSE-4-fix": "l93_s4fix", "SPARSE-4-var": "l93_s4var", "DENSE-8": "l93_d8",
        "SPARSE-8-fix": "l93_s8fix", "SPARSE-8-var": "l93_s8var", "WARM-SPARSE-8": "l93_warm8"}


def arm(prefix):
    rows = {}
    for s in (1, 2, 3):
        f = "boxes/runs_s5/%s_s%d/eval.json" % (prefix, s)
        if os.path.exists(f):
            r = json.load(open(f))
            rows[s] = {"cum512": r["cumulative"]["512"], "ext2048": r["extension_2048"]["cumulative"]["2048"],
                       "zero_after": r["removal"]["zero"]["acc_after_first_ablated_slot"],
                       "mean_after": r["removal"]["mean"]["acc_after_first_ablated_slot"],
                       "per_slot": r["chunk_check"]["per_slot_acc"], "probe": r.get("probe_acc_per_slot"),
                       "iia": r.get("iia_per_slot"), "flops": r["compute_512_per_problem"]["flops"]}
    passing = [s for s, v in rows.items() if v["cum512"] >= 0.99 and v["zero_after"] <= 0.10]
    return {"n": len(rows), "pass_seeds": passing, "rule_met": len(passing) >= 2, "runs": rows}


def main():
    res = {name: arm(p) for name, p in ARMS.items()}
    res["P-S4"] = res["SPARSE-4-fix"]["rule_met"]
    res["P-S8"] = res["SPARSE-8-fix"]["rule_met"]
    res["DENSE-8_ok"] = res["DENSE-8"]["rule_met"]
    for name in ARMS:
        a = res[name]
        print("%-15s n %d  pass %s  rule %s | cum512 %s | zero_after %s | per_slot %s" % (
            name, a["n"], a["pass_seeds"], a["rule_met"], {s: round(v["cum512"], 4) for s, v in a["runs"].items()},
            {s: round(v["zero_after"], 3) for s, v in a["runs"].items()},
            {s: [round(x, 3) for x in v["per_slot"]] for s, v in a["runs"].items()}))
    print("P-S4", res["P-S4"], "| P-S8", res["P-S8"], "| DENSE-8 ok", res["DENSE-8_ok"])
    json.dump(res, open("theory/results/score_s93.json", "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
