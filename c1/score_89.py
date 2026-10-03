"""Score THEORY §89 (fade to latent; notes 89-1/-3/-5) from boxes/runs/f89_<arm>_s<seed>/eval.json (pure latent, the §66
bench items, seed 6600). Committed before any F1-F3 panel result beyond the s1 placement pilots was read.
* F1: FADE-std, FADE-std-dec or FADE-soft, median (over seeds) accuracy at N 5-8 >= 0.90.
* F2 (C1): for each arm meeting F1, median mean-ablated accuracy at N 5-8 <= NC median + 0.05 (NC = §66's nc_s1-3 on
  the same items, note 89-3: 0.13 / 0.14 / 0.15). Zero-ablation also reported.
* F3: (FADE arm - COCONUT-stage) at N 5-8 >= 0.20, paired by seed (median of per-seed differences), applied only to
  arms meeting F1 (note 89-2; gate added 2026-10-03 after note 89-9, the difference is still reported descriptively).
* Descriptive: raw / soft / gold; nearest-token cosine; aux decode (std_dec); coco_stage with and without the fp32 seeds
  (s1, s2 on the RTX 2070; the others bf16, from eval_dtype) when they differ.
Seeds: 5 for fade_std, fade_std_dec, coco_stage; 3 for raw / soft / gold. Incomplete arms are marked.
    python c1/score_89.py
"""
import json
import os

import numpy as np

ARMS = {"fade_std": 5, "fade_std_dec": 5, "coco_stage": 5, "fade_raw": 3, "fade_soft": 3, "fade_gold": 3}


def nc_tc():
    out = {}
    for arm in ("nc", "tc"):
        v = []
        for s in (1, 2, 3):
            r = json.load(open(f"bench/results/boxes__boxes_{arm}_s{s}.json"))
            v.append(float(np.mean([x["correct"] for x in r["items"] if 5 <= int(x["id"].split(":")[1]) <= 8])))
        out[arm] = {"per_seed": v, "median": float(np.median(v))}
    return out


def main():
    R = {a: {s: json.load(open(f"boxes/runs/f89_{a}_s{s}/eval.json")) for s in range(1, n + 1)
             if os.path.exists(f"boxes/runs/f89_{a}_s{s}/eval.json")} for a, n in ARMS.items()}
    base = nc_tc(); floor = base["nc"]["median"] + 0.05
    acc = lambda e, mode="none": e["modes"][mode]["acc_N5_8"]          # noqa: E731
    out = {"n_runs": {a: len(v) for a, v in R.items()}, "nc_tc_s66": base, "F2_floor": floor, "arms": {}}
    for a, rs in R.items():
        if not rs:
            continue
        t = {"median_N5_8": float(np.median([acc(e) for e in rs.values()])), "per_seed": {s: acc(e) for s, e in rs.items()},
             "median_N9_16": float(np.median([e["modes"]["none"]["acc_N9_16"] for e in rs.values()])),
             "median_zero": float(np.median([acc(e, "zero") for e in rs.values()])),
             "median_mean": float(np.median([acc(e, "mean") for e in rs.values()])),
             "nearest_token_cos": float(np.median([e["nearest_token_cos_mean"] for e in rs.values()])),
             "dtype": {s: e.get("eval_dtype") for s, e in rs.items()}, "complete": len(rs) == ARMS[a]}
        if a == "fade_std_dec":
            t["aux_decode_alpha1"] = float(np.median([e["modes"]["none"]["aux_decode_acc_alpha1"] for e in rs.values()]))
        out["arms"][a] = t
    A = out["arms"]; F1 = {}
    for a in ("fade_std", "fade_std_dec", "fade_soft"):
        if a in A:
            F1[a] = {"median_N5_8": A[a]["median_N5_8"], "met": A[a]["median_N5_8"] >= 0.90, "complete": A[a]["complete"]}
    out["F1"] = {"arms": F1, "met": any(v["met"] for v in F1.values())}
    out["F2"] = {a: {"median_mean_ablated": A[a]["median_mean"], "floor": floor, "met": A[a]["median_mean"] <= floor}
                 for a, v in F1.items() if v["met"]}
    co = R.get("coco_stage", {}); F3 = {}
    for a in ("fade_std", "fade_std_dec", "fade_soft"):
        d = [acc(R[a][s]) - acc(co[s]) for s in R.get(a, {}) if s in co]
        if d:
            F3[a] = {"median_paired_diff": float(np.median(d)), "per_seed": d,
                     "met": bool(F1.get(a, {}).get("met")) and float(np.median(d)) >= 0.20,
                     "applies": bool(F1.get(a, {}).get("met"))}       # note 89-2 / 89-9: F3 only for arms meeting F1
    out["F3"] = F3
    if co:
        bf = [acc(e) for e in co.values() if e.get("eval_dtype") == "torch.bfloat16"]
        out["coco_by_dtype"] = {"all_median": float(np.median([acc(e) for e in co.values()])),
                                "bf16_only_median": float(np.median(bf)) if bf else None, "n_bf16": len(bf)}
    json.dump(out, open("theory/results/score_s89.json", "w", newline="\n"), indent=1)
    for a, t in A.items():
        print(a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items() if k not in ("dtype", "per_seed")})
    for k in ("F1", "F2", "F3", "coco_by_dtype"):
        print(k, out.get(k))


if __name__ == "__main__":
    main()
