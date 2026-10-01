"""Score THEORY §64 (A5 / S3) and §64b (A5 with curriculum, a5c) by Experiments, from theory/runs/st/<run>/eval_st.json.
in-range = mean full-state accuracy (single, K' = max(8, N)) at N 1-8; gated if >= 0.90.
ST1: median K'=0 accuracy at N 5-8 <= 0.10 (A5; S3 reported; over gated runs).
ST2: mean trunc agreement > mean true accuracy (single) at N 9-16 in >= 70 % of gated runs, AND median true accuracy at
     N 13-16 <= half the median in-range accuracy.
ST3: median restart accuracy at N 9-16 >= 0.75, AND restart - single >= 0.20 (N 9-16) in >= 70 % of gated runs.
ST4: median edit_IIA >= 0.90.  Note 64-2: S3's ST2/ST3 count as loop evidence only if S3's median K'=0 at N 5-8 <= 0.50.
§64b: the same on a5c; fewer than 3 gated runs -> no mechanism verdict.
    python theory/score_s64.py
"""
import json
import os

import numpy as np

def mean_n(d, lo, hi):
    return float(np.mean([d[str(n)] for n in range(lo, hi + 1) if str(n) in d]))

res = {}
for grp in ("a5", "s3", "a5c"):
    runs = {}
    for s in range(1, 11):
        f = f"theory/runs/st/{grp}_s{s}/eval_st.json"
        if not os.path.exists(f):
            continue
        e = json.load(open(f))
        r = {"in_range": mean_n(e["single"], 1, 8), "k0_5_8": mean_n(e["single_K0"], 5, 8),
             "true_9_16": mean_n(e["single"], 9, 16), "trunc_9_16": mean_n(e["trunc"], 9, 16),
             "true_13_16": mean_n(e["single"], 13, 16), "restart_9_16": mean_n(e["restart"], 9, 16),
             "restart_17_24": mean_n(e["restart"], 17, 24) if "17" in e["restart"] else None,
             "k24_9_16": mean_n(e["single_K24"], 9, 16), "edit_IIA": float(e["edit_IIA"]), "chance": float(e["chance"]),
             "pointer_rank": e.get("pointer_rank")}
        r["gated"] = r["in_range"] >= 0.90
        runs[f"s{s}"] = r
    g = [r for r in runs.values() if r["gated"]]
    med = lambda k, rs: float(np.median([r[k] for r in rs])) if rs else None
    out = {"n": len(runs), "gated": len(g), "runs": runs,
           "median_in_range_all": med("in_range", list(runs.values())), "median_k0_5_8_gated": med("k0_5_8", g)}
    if g:
        out["ST1_value"] = med("k0_5_8", g); out["ST1"] = out["ST1_value"] <= 0.10
        frac2 = np.mean([r["trunc_9_16"] > r["true_9_16"] for r in g])
        out["ST2"] = {"share_trunc>true": float(frac2), "median_true_13_16": med("true_13_16", g), "median_in_range": med("in_range", g)}
        out["ST2"]["met"] = frac2 >= 0.70 and out["ST2"]["median_true_13_16"] <= 0.5 * out["ST2"]["median_in_range"]
        frac3 = np.mean([r["restart_9_16"] - r["true_9_16"] >= 0.20 for r in g])
        out["ST3"] = {"median_restart_9_16": med("restart_9_16", g), "share_restart>=single+0.2": float(frac3)}
        out["ST3"]["met"] = out["ST3"]["median_restart_9_16"] >= 0.75 and frac3 >= 0.70
        out["ST4"] = {"median_edit_IIA": med("edit_IIA", g)}; out["ST4"]["met"] = out["ST4"]["median_edit_IIA"] >= 0.90
    res[grp] = out
if res["s3"].get("gated"):
    res["s3"]["note_64_2_loop_evidence_admissible"] = res["s3"]["median_k0_5_8_gated"] <= 0.50
res["a5c"]["mechanism_verdict_possible"] = res["a5c"]["gated"] >= 3
json.dump(res, open("theory/results/score_s64.json", "w", newline="\n"), indent=1,
          default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
for grp, o in res.items():
    print("==", grp, "n", o["n"], "gated", o["gated"], "median in-range (all)", round(o["median_in_range_all"], 3))
    for k in ("ST1_value", "ST1", "ST2", "ST3", "ST4", "note_64_2_loop_evidence_admissible", "mechanism_verdict_possible"):
        if k in o:
            print("  ", k, o[k])
