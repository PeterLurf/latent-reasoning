"""THEORY §89 scoring (rules fixed in §89 and notes 89-1 to 89-8; written before Theory read any §89 value).
F1 (per FADE arm, best of 3: fade_std, fade_std_dec, fade_soft): pure-latent accuracy at N 5-8, median over seeds >= 0.90.
F2 (each arm meeting F1): mean-ablation accuracy at N 5-8 <= NC median + 0.05, where NC = §66's nc_s1-3 (note 89-3), N 5-8
    accuracy over the same bench items (bench/results/boxes__boxes_nc_s*.json, items with depth 5-8).
F3 (each arm meeting F1): median over seeds of (FADE - COCONUT-stage) at N 5-8, paired by seed, >= 0.20.
Descriptive (89-6): per question type ("all" / "where") with removal; N 1-8, 9-16; raw / soft / gold; coco_stage with and
without the fp32 seeds s1-s2 (89-8).
    python theory/score_s89.py
"""
import glob
import json
import os

import numpy as np

ARMS = ["fade_std", "fade_std_dec", "fade_soft", "fade_raw", "fade_gold", "coco_stage"]


def load(arm):
    out = {}
    for f in sorted(glob.glob("boxes/runs/f89_%s_s*/eval.json" % arm)):
        s = int(f.replace("\\", "/").split("_s")[-1].split("/")[0]); out[s] = json.load(open(f))
    return out


def nc_floor():
    vals = []
    for s in (1, 2, 3):
        d = json.load(open("bench/results/boxes__boxes_nc_s%d.json" % s))
        it = [x for x in d["items"] if 5 <= x["depth"] <= 8]
        vals.append(float(np.mean([x["correct"] for x in it])))
    return float(np.median(vals)), vals


def main():
    runs = {a: load(a) for a in ARMS}
    floor, nc_vals = nc_floor()
    res = {"nc_floor_N5_8": floor, "nc_seeds": nc_vals, "arms": {}}
    for a, rs in runs.items():
        m = lambda mode, key: {s: r["modes"][mode][key] for s, r in rs.items()}           # noqa: E731
        res["arms"][a] = {"n": len(rs), "none_N5_8": m("none", "acc_N5_8"), "mean_N5_8": m("mean", "acc_N5_8"),
                          "zero_N5_8": m("zero", "acc_N5_8"), "none_N9_16": m("none", "acc_N9_16"),
                          "none_all": m("none", "acc_N5_8_all"), "none_where": m("none", "acc_N5_8_where"),
                          "mean_all": m("mean", "acc_N5_8_all"), "zero_all": m("zero", "acc_N5_8_all"),
                          "dtype": {s: r.get("eval_dtype") for s, r in rs.items()},
                          "nearest_token_cos": {s: r.get("nearest_token_cos_mean") for s, r in rs.items()}}
    coco = res["arms"]["coco_stage"]["none_N5_8"]
    verdict = {}
    for a in ("fade_std", "fade_std_dec", "fade_soft"):
        A = res["arms"][a]; acc = A["none_N5_8"]
        if not acc:
            continue
        med = float(np.median(list(acc.values()))); f1 = med >= 0.90
        v = {"median_N5_8": med, "n": len(acc), "F1": f1}
        if f1:
            v["F2_mean_ablation_median"] = float(np.median(list(A["mean_N5_8"].values())))
            v["F2"] = v["F2_mean_ablation_median"] <= floor + 0.05
            pairs = [acc[s] - coco[s] for s in acc if s in coco]
            v["F3_paired_median"] = float(np.median(pairs)) if pairs else None; v["F3"] = bool(pairs) and v["F3_paired_median"] >= 0.20
        verdict[a] = v
        print("%-13s n %d  N5-8 median %.3f  F1 %s  %s" % (a, len(acc), med, f1, {k: v[k] for k in v if k.startswith("F2") or k.startswith("F3")}))
    res["verdict"] = verdict
    res["F1_arms_met"] = [a for a, v in verdict.items() if v["F1"]]
    c_all = list(coco.values()); c_bf16 = [coco[s] for s in coco if s not in (1, 2)]
    res["coco_median_all_seeds"] = float(np.median(c_all)) if c_all else None
    res["coco_median_bf16_only"] = float(np.median(c_bf16)) if c_bf16 else None
    print("NC floor N5-8 %.3f (seeds %s) | coco median all %s, bf16-only %s | F1 met by %s" % (
        floor, [round(x, 3) for x in nc_vals], res["coco_median_all_seeds"], res["coco_median_bf16_only"], res["F1_arms_met"]))
    for a in ARMS:
        A = res["arms"][a]
        print("  %-13s none %s | mean %s | all %s | where %s" % (a, {s: round(x, 3) for s, x in A["none_N5_8"].items()},
              {s: round(x, 3) for s, x in A["mean_N5_8"].items()}, {s: round(x, 3) for s, x in A["none_all"].items()},
              {s: round(x, 3) for s, x in A["none_where"].items()}))
    json.dump(res, open("theory/results/score_s89_theory.json", "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
