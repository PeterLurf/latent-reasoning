"""THEORY §106 scoring (rules fixed in §106; written before any §106 implementation or run; Theory is the only scorer).
Runs: boxes/runs_mult/s106_{gens,gens_mask,wcs,tcs}_s{1,2,3}/eval.json with keys acc_by_b {"4","8","16","32"} (exact
product), acc_a5_b8, removal.{zero,mean}.acc_b8 (latent arms), compute.flops_by_b {...}, per_step_decode_acc.
C2-S (per arm): acc_by_b["16"] >= 0.95 AND acc_by_b["32"] >= 0.90 in >= 2/3 seeds.
C1-S: GEN-S-MASK meets C2-S AND, in its C2-S seeds, zero- and mean-removal acc at b = 8 <= 0.05.
Attribution (descriptive): WC-S meets C2-S -> C2 belongs to format + cut; latents credited with C1 (C1-S) and C3 (GEN-S vs
WC-S FLOPs at b = 16 / 32 where both are accurate).
    python theory/score_s106.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"GEN-S": "gens", "GEN-S-MASK": "gens_mask", "WC-S": "wcs", "TC-S": "tcs", "GEN-S-MASK-k2": "gens_mask_k2", "WC-S-k2": "wcs_k2"}


def main():
    res = {"arms": {}}
    for name, tag in ARMS.items():
        runs = {}
        for s in (1, 2, 3):
            f = os.path.join(ROOT, "boxes", "runs_mult", "s106_%s_s%d" % (tag, s), "eval.json")
            if os.path.exists(f):
                runs[s] = json.load(open(f))
        c2 = [s for s, r in runs.items() if r["acc_by_b"]["16"] >= 0.95 and r["acc_by_b"]["32"] >= 0.90]
        A = {"n": len(runs), "acc_by_b": {s: r["acc_by_b"] for s, r in runs.items()}, "c2_seeds": c2, "C2_S": len(c2) >= 2,
             "removal_b8": {s: r.get("removal") for s, r in runs.items()},
             "flops_by_b": {s: r.get("compute", {}).get("flops_by_b") for s, r in runs.items()}}
        res["arms"][name] = A
        print("%-11s n %d C2-S %s seeds %s | acc_by_b %s" % (name, len(runs), A["C2_S"], c2, A["acc_by_b"]))
    for mk, wk in (("GEN-S-MASK-k2", "WC-S-k2"),):
        mm = res["arms"][mk]
        res["C1_S_" + mk] = mm["C2_S"] and all(mm["removal_b8"][s]["zero"]["acc_b8"] <= 0.05 and mm["removal_b8"][s]["mean"]["acc_b8"] <= 0.05 for s in mm["c2_seeds"])
        res["attribution_format_plus_cut_" + wk] = res["arms"][wk]["C2_S"]
    m = res["arms"]["GEN-S-MASK"]
    res["C1_S"] = m["C2_S"] and all(m["removal_b8"][s]["zero"]["acc_b8"] <= 0.05 and m["removal_b8"][s]["mean"]["acc_b8"] <= 0.05 for s in m["c2_seeds"])
    res["attribution_format_plus_cut"] = res["arms"]["WC-S"]["C2_S"]
    print("C1-S", res["C1_S"], "| WC-S extrapolates (format + cut)", res["attribution_format_plus_cut"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s106.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
