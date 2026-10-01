"""Score THEORY §61 (F1-S) from prosqa/runs_ail/f1s_s{1,2,3}/eval.json (hardened labels, P = 16).
S1: accuracy at d1-6 >= 0.95 in >= 2/3 seeds.  S2: d11-14 >= 0.85 in >= 2/3.  S3: edit IIA >= 0.8 (median).
S4: Spearman(fixed point, depth) >= 0.8 (median).  S0: soft - hardened accuracy <= 0.03 per run and depth band.
Descriptive: accuracy at P = d, cue-insertion drop, label precision / recall.  Reference: §54b AI-soft (0/3).
    python prosqa/score_f1s.py
"""
import json

import numpy as np

BANDS = ((1, 2), (3, 6), (1, 6), (7, 10), (11, 14))


def sp(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


per = {}
for s in (1, 2, 3):
    R = json.load(open(f"prosqa/runs_ail/f1s_s{s}/eval.json"))["rows"]
    m = lambda k, lo, hi: (lambda v: float(np.mean(v)) if v else None)([float(r[k]) for r in R if lo <= r["depth"] <= hi and r.get(k) is not None])
    v = {f"acc_d{lo}_{hi}": m("acc", lo, hi) for lo, hi in BANDS}
    v.update({f"gap_d{lo}_{hi}": m("acc_soft", lo, hi) - m("acc", lo, hi) for lo, hi in BANDS})
    v.update({f"accPd_d{lo}_{hi}": m("acc_at_P_eq_d", lo, hi) for lo, hi in BANDS})
    v["edit_IIA"] = float(np.mean([r["edit_neg_on"] and r["edit_tgt_same"] for r in R if "edit_neg_on" in r]))
    v["spearman_fp_depth"] = sp([r["fixed_point"] for r in R], [r["depth"] for r in R])
    cue = [r for r in R if "acc_cue" in r]
    v["cue_drop"] = float(np.mean([r["acc"] for r in cue]) - np.mean([r["acc_cue"] for r in cue]))
    v["label_prec"] = float(np.mean([r["label_prec"] for r in R])); v["label_recall"] = float(np.mean([r["label_recall"] for r in R]))
    v["n"] = {f"d{lo}_{hi}": sum(lo <= r["depth"] <= hi for r in R) for lo, hi in BANDS}
    per[f"s{s}"] = v
g = lambda k: [v[k] for v in per.values()]
out = {"per_seed": per,
       "S1": sum(x >= 0.95 for x in g("acc_d1_6")) >= 2, "S2": sum(x >= 0.85 for x in g("acc_d11_14")) >= 2,
       "S3": float(np.median(g("edit_IIA"))) >= 0.8, "S4": float(np.median(g("spearman_fp_depth"))) >= 0.8,
       "S0": all(v[f"gap_d{lo}_{hi}"] <= 0.03 for v in per.values() for lo, hi in BANDS)}   # one-sided, as AG0
json.dump(out, open("results/f1s.json", "w", newline="\n"), indent=1)
print({k: out[k] for k in ("S0", "S1", "S2", "S3", "S4")})
for s, v in per.items():
    print(s, " ".join(f"{k} {v[k]:.3f}" for k in v if k.startswith("acc_") or k in ("edit_IIA", "spearman_fp_depth", "cue_drop", "label_prec", "label_recall")))
    print("   gaps", {k: round(v[k], 3) for k in v if k.startswith("gap_")}, "| P=d", {k: (None if v[k] is None else round(v[k], 3)) for k in v if k.startswith("accPd_")})
