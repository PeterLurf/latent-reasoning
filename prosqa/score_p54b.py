"""Score THEORY §54b (F1 on GPT-2, revision; PROTOCOL12 line). Inputs prosqa/runs_ail/p54b_<arm>_s<seed>/eval.json.
AG0: in-distribution (depth <= 6) soft-test minus hardened accuracy <= 0.03 (median over seeds; per run and per depth band reported).
AG1': hardened in-distribution accuracy >= 0.95 in >= 2/3 seeds.  AG2: hardened accuracy at depths 11-14 >= 0.85 in >= 2/3.
AG3: edit IIA (negative option turns on AND target label unchanged) >= 0.8 (median).  AG4: Spearman(fixed point, depth) >= 0.8.
AI-soft primary; AI-stoch scored on the same criteria; AI-hint = ceiling. 54b-1 cue control: plain - cue accuracy (> 0.05 = uses the cue).
    python prosqa/score_p54b.py
"""
import json
import os

import numpy as np


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else float("nan")


out = {}
for arm in ("soft", "stoch", "hint2"):
    per = {}
    for s in (1, 2, 3):
        f = f"prosqa/runs_ail/p54b_{arm}_s{s}/eval.json"
        if not os.path.exists(f):
            continue
        R = json.load(open(f))["rows"]
        band = lambda lo, hi, k: [r[k] for r in R if lo <= r["depth"] <= hi and k in r]
        m = lambda xs: float(np.mean(xs)) if xs else None
        v = {"hard_d3_6": m(band(3, 6, "acc")), "hard_d7_10": m(band(7, 10, "acc")), "hard_d11_14": m(band(11, 14, "acc")),
             "soft_d3_6": m(band(3, 6, "acc_soft")), "soft_d7_10": m(band(7, 10, "acc_soft")), "soft_d11_14": m(band(11, 14, "acc_soft")),
             "cue_d3_6": m(band(3, 6, "acc_cue")), "plain_on_cue_rows_d3_6": m([r["acc"] for r in R if 3 <= r["depth"] <= 6 and "acc_cue" in r]),
             "edit_IIA": m([float(r["edit_neg_on"] and r["edit_tgt_same"]) for r in R if "edit_neg_on" in r]),
             "spearman_fp_depth": spearman([r["fixed_point"] for r in R], [r["depth"] for r in R]),
             "fixed_point_median": float(np.median([r["fixed_point"] for r in R])),
             "label_prec": m([r["label_prec"] for r in R]), "label_recall": m([r["label_recall"] for r in R])}
        v["gap_d3_6"] = v["soft_d3_6"] - v["hard_d3_6"] if v["soft_d3_6"] is not None else None
        v["cue_drop_d3_6"] = v["plain_on_cue_rows_d3_6"] - v["cue_d3_6"] if v["cue_d3_6"] is not None else None
        log = [json.loads(l) for l in open(f"prosqa/runs_ail/p54b_{arm}_s{s}/log.jsonl")]
        last = [r for r in log if "binarity" in r]
        v["train_end"] = {k: last[-1][k] for k in ("acc_hard", "acc_soft", "binarity")} if last else None
        per[f"s{s}"] = v
    g = lambda k: [v[k] for v in per.values() if v[k] is not None]
    res = {"per_seed": per,
           "AG0_median_gap": float(np.median(g("gap_d3_6"))) if g("gap_d3_6") else None,
           "AG1p_seeds_ge_0.95": sum(x >= 0.95 for x in g("hard_d3_6")),
           "AG2_seeds_ge_0.85": sum(x >= 0.85 for x in g("hard_d11_14")),
           "AG3_median_IIA": float(np.median(g("edit_IIA"))), "AG4_median_spearman": float(np.nanmedian(g("spearman_fp_depth"))),
           "cue_drop_median": float(np.median(g("cue_drop_d3_6")))}
    res["AG0"] = None if res["AG0_median_gap"] is None else res["AG0_median_gap"] <= 0.03
    res["AG1p"] = res["AG1p_seeds_ge_0.95"] >= 2; res["AG2"] = res["AG2_seeds_ge_0.85"] >= 2
    res["AG3"] = res["AG3_median_IIA"] >= 0.8; res["AG4"] = res["AG4_median_spearman"] >= 0.8
    out[arm] = res
json.dump(out, open("results/p54b.json", "w", newline="\n"), indent=1)
for arm, r in out.items():
    print(f"== {arm}: AG0 {r['AG0']} (gap {r['AG0_median_gap']}), AG1' {r['AG1p']} ({r['AG1p_seeds_ge_0.95']}/3), AG2 {r['AG2']}, AG3 {r['AG3']} (IIA {r['AG3_median_IIA']:.2f}), AG4 {r['AG4']} (rho {r['AG4_median_spearman']:.2f}), cue drop {r['cue_drop_median']:.3f}")
    for s, v in r["per_seed"].items():
        print("  ", s, " ".join(f"{k} {v[k]:.3f}" for k in ("hard_d3_6", "soft_d3_6", "hard_d7_10", "hard_d11_14", "edit_IIA", "label_prec", "label_recall") if v[k] is not None), "| fp", v["fixed_point_median"], "| train_end", v["train_end"])
