"""Score THEORY §88 (30857ca; notes 88-1/88-2) from c1/runs/p88_<arm>_k<k>_s<seed>/eval.json, arms align / end,
k in {2, 3}, seeds 1-3. Written after the runs finished but before any eval.json was read (committed first).
G1: LOOP-align median (over seeds) accuracy on (b) unseen words >= 0.90 at k = 2 AND at k = 3.
G2: LOOP-end median (b) accuracy < 0.50 (at each k; reported per k and met if both).
Kill (Literature): LOOP-align median (b) < 0.50 (at either k).
Descriptive: training-word and (a) held-out-pair accuracy, per-iteration (b) accuracy, per seed; §80 P0 reference.
    python c1/score_88.py
"""
import json
import os

import numpy as np


def main():
    R = {}
    for arm in ("align", "end"):
        for k in (2, 3):
            for s in (1, 2, 3):
                f = f"c1/runs/p88_{arm}_k{k}_s{s}/eval.json"
                if os.path.exists(f):
                    R[(arm, k, s)] = json.load(open(f))
    med = lambda arm, k, key: float(np.median([R[(arm, k, s)][key] for s in (1, 2, 3) if (arm, k, s) in R]))  # noqa: E731
    out = {"n_runs": len(R), "table": {}}
    for arm in ("align", "end"):
        for k in (2, 3):
            out["table"][f"{arm}_k{k}"] = {
                "median_b": med(arm, k, "acc_b_unseen_words"), "median_a": med(arm, k, "acc_a_heldout_pairs"),
                "median_train": med(arm, k, "acc_train"),
                "per_seed_b": [R[(arm, k, s)]["acc_b_unseen_words"] for s in (1, 2, 3) if (arm, k, s) in R],
                "per_iteration_b_median": np.median([R[(arm, k, s)]["per_iteration_acc_b"] for s in (1, 2, 3) if (arm, k, s) in R], 0).tolist()}
    t = out["table"]
    out["G1"] = {"met": t["align_k2"]["median_b"] >= 0.90 and t["align_k3"]["median_b"] >= 0.90,
                 "align_b": [t["align_k2"]["median_b"], t["align_k3"]["median_b"]]}
    out["G2"] = {"met": t["end_k2"]["median_b"] < 0.50 and t["end_k3"]["median_b"] < 0.50,
                 "end_b": [t["end_k2"]["median_b"], t["end_k3"]["median_b"]]}
    out["kill"] = t["align_k2"]["median_b"] < 0.50 or t["align_k3"]["median_b"] < 0.50
    out["P0_reference_b"] = {k: json.load(open(f"c1/runs/p80_p0_k{k}/eval.json"))["acc_b_unseen_words"] for k in (2, 3)
                             if os.path.exists(f"c1/runs/p80_p0_k{k}/eval.json")}
    if len(R) < 12:
        out["status"] = "incomplete"
    json.dump(out, open("theory/results/score_s88.json", "w", newline="\n"), indent=1)
    for kk, v in t.items():
        print(kk, {a: (round(b, 3) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in v.items()})
    for kk in ("G1", "G2", "kill", "P0_reference_b"):
        print(kk, out[kk])


if __name__ == "__main__":
    main()
