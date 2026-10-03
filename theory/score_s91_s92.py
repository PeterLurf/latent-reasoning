"""THEORY §91 / §92 scoring (rules fixed in §91, §92 and §86 / 86-2 / 86-4; written after Audit's raw read, before Theory read
any value; disclosed).
§91: per arm (BASE / LONG / WD-SP / CURR, seeds 4-8), reliable iff acc_b_unseen_words >= 0.90 in >= 4/5 seeds.
§92: gate = cumulative accuracy >= 0.95 at 512 in >= 2/3 seeds (l92_k4_s1..3). If passed, §86's predictions on these runs:
  R1 (k = 4 part; §92 has no k = 1 runs): cumulative >= 0.99 at 512 in >= 2/3.
  R2: zero ablation of the intermediate slots -> accuracy after the first ablated slot <= 0.10 (mean / isolate reported).
  R3: inference FLOPs per 512-problem <= 0.6 x the retrained §71 WC0_4's (wc0k4r_s*), medians.
    python theory/score_s91_s92.py
"""
import glob
import json
import os

import numpy as np


def s91():
    out = {}
    for arm in ("base", "long", "wdsp", "curr"):
        rows = []
        for s in range(4, 9):
            f = "c1/runs/p91_%s_s%d/eval.json" % (arm, s)
            if os.path.exists(f):
                r = json.load(open(f))
                rows.append({"seed": s, "b": r["acc_b_unseen_words"], "a": r["acc_a_heldout_pairs"], "train": r["acc_train"],
                             "coverage_full": r["coverage_full"], "per_iter_b": r["per_iteration_acc_b"],
                             "b_split": r.get("b_split_by_transition_seen")})
        k = sum(x["b"] >= 0.90 for x in rows)
        out[arm] = {"n": len(rows), "n_ge_0.90": k, "reliable": len(rows) == 5 and k >= 4,
                    "b": [round(x["b"], 3) for x in rows], "train": [round(x["train"], 3) for x in rows], "runs": rows}
        print("§91 %-5s b>=0.90 in %d/%d  reliable=%s  b %s  train %s" % (arm, k, len(rows), out[arm]["reliable"], out[arm]["b"], out[arm]["train"]))
    return out


def s92():
    runs = {}
    for s in (1, 2, 3):
        f = "boxes/runs_s5/l92_k4_s%d/eval.json" % s
        if os.path.exists(f):
            runs[s] = json.load(open(f))
    wc = [json.load(open(f)) for f in sorted(glob.glob("boxes/runs_s5/wc0k4r_s*/eval.json"))]
    wc_flops = [w["compute_512_per_problem"]["flops"] for w in wc if "compute_512_per_problem" in w]
    cum = {s: r["cumulative"]["512"] for s, r in runs.items()}
    gate = sum(v >= 0.95 for v in cum.values()) >= 2
    out = {"cum512": cum, "gate": gate, "ext2048": {s: r["extension_2048"]["cumulative"]["2048"] for s, r in runs.items()},
           "per_slot": {s: r["chunk_check"]["per_slot_acc"] for s, r in runs.items()},
           "removal_after_first": {s: {m: r["removal"][m]["acc_after_first_ablated_slot"] for m in ("zero", "mean", "isolate")}
                                   for s, r in runs.items()},
           "flops": {s: r["compute_512_per_problem"]["flops"] for s, r in runs.items()}, "wc0k4r_flops": wc_flops}
    if gate:
        out["R1_k4"] = sum(v >= 0.99 for v in cum.values()) >= 2
        out["R2"] = all(v["zero"] <= 0.10 for v in out["removal_after_first"].values())
        ratio = float(np.median(list(out["flops"].values())) / np.median(wc_flops)) if wc_flops else None
        out["R3_ratio"] = ratio; out["R3"] = ratio is not None and ratio <= 0.6
    print("§92 cum512", cum, "gate", gate, "| ext2048", out["ext2048"], "| removal", out["removal_after_first"],
          "| R1_k4", out.get("R1_k4"), "R2", out.get("R2"), "R3 ratio", out.get("R3_ratio"), out.get("R3"))
    return out


if __name__ == "__main__":
    res = {"s91": s91(), "s92": s92()}
    json.dump(res, open("theory/results/score_s91_s92.json", "w", newline="\n"), indent=1, default=float)
