"""Score THEORY §90 (+ notes 90-1 / 90-2) from c1/runs/p90_<arm>_s<seed>/eval.json. Written after the 12 runs
finished and before any eval.json was read (committed first).
V1: S3 k3-25 median (b) < 0.50.
V2 (restated by note 90-2 over the full-coverage arms): S4 k2-400 median (b) >= 0.90; k2-100 reported; S4 k2-25
(coverage 0.77-0.80) descriptive, its (b) split by all-seen vs >= 1-unseen transitions.
    python c1/score_90.py
"""
import json
import os

import numpy as np

ARMS = ("S3k3_n25", "S4k2_n25", "S4k2_n100", "S4k2_n400")


def main():
    R = {a: [json.load(open(f"c1/runs/p90_{a}_s{s}/eval.json")) for s in (1, 2, 3) if os.path.exists(f"c1/runs/p90_{a}_s{s}/eval.json")]
         for a in ARMS}
    out = {"n_runs": {a: len(v) for a, v in R.items()}, "arms": {}}
    for a, rs in R.items():
        if not rs:
            continue
        out["arms"][a] = {"median_b": float(np.median([r["acc_b_unseen_words"] for r in rs])),
                          "per_seed_b": [r["acc_b_unseen_words"] for r in rs],
                          "median_a": float(np.median([r["acc_a_heldout_pairs"] for r in rs if r["acc_a_heldout_pairs"] is not None])),
                          "median_train": float(np.median([r["acc_train"] for r in rs])),
                          "per_iteration_b_median": np.median([r["per_iteration_acc_b"] for r in rs], 0).tolist(),
                          "coverage_min": [min(r["coverage_per_iteration"]) for r in rs],
                          "b_split": [r["b_split_by_transition_seen"] for r in rs]}
    A = out["arms"]
    out["V1"] = {"S3k3_n25_median_b": A["S3k3_n25"]["median_b"], "met": A["S3k3_n25"]["median_b"] < 0.50}
    out["V2"] = {"S4k2_n400_median_b": A["S4k2_n400"]["median_b"], "S4k2_n100_median_b": A["S4k2_n100"]["median_b"],
                 "met": A["S4k2_n400"]["median_b"] >= 0.90, "note": "restated by 90-2; the 25-word arm is descriptive"}
    if any(v < 3 for v in out["n_runs"].values()):
        out["status"] = "incomplete"
    json.dump(out, open("theory/results/score_s90.json", "w", newline="\n"), indent=1)
    for a, v in A.items():
        print(a, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "b_split"})
    print("V1", out["V1"]); print("V2", out["V2"])


if __name__ == "__main__":
    main()
