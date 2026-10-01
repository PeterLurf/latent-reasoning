"""Score THEORY §63 (F4) by Experiments. Inputs theory/runs/f4/<model>_<S|N>_t{1,2}/eval_f4.json and eval_f4_swap.json (in t1).
Per model: own-translator metrics averaged over t1/t2 at i = 2; swap metrics = mean of AB and BA at i = 2. Medians over the
10 models of each family. F4-1..F4-4 on P30 arm S; F4-5 compares median RT(S) - RT(N) between P30 and GT students.
    python theory/score_f4.py
"""
import json

import numpy as np

SEEDS = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
R = "theory/runs/f4"
def own(name, arm, k):
    return float(np.mean([json.load(open(f"{R}/{name}_{arm}_t{t}/eval_f4.json"))["i2"][k] for t in (1, 2)]))
def swap(name, arm, k):
    s = json.load(open(f"{R}/{name}_{arm}_t1/eval_f4_swap.json")); return float(np.mean([s[d]["i2"][k] for d in ("AB", "BA")]))
fam = {"p30": [f"p30_s{s}" for s in SEEDS], "plgt": [f"plgt_s{s}" for s in SEEDS]}
K = ["RT", "IIA", "flip_collateral", "err_repro", "err_repro_input", "prec", "rec", "RT_mean", "RT_mismatch", "flip_on"]
out = {}
for f, names in fam.items():
    for arm in ("S", "N"):
        per = {n: {k: own(n, arm, k) for k in K} for n in names}
        for n in names:
            per[n]["RT_swap"] = swap(n, arm, "RT"); per[n]["IIA_swap"] = swap(n, arm, "IIA")
        out[f"{f}_{arm}"] = {"per_model": per, "median": {k: float(np.median([per[n][k] for n in names])) for k in K + ["RT_swap", "IIA_swap"]}}
m = out["p30_S"]["median"]
d_p30 = float(np.median([out["p30_S"]["per_model"][n]["RT"] - out["p30_N"]["per_model"][n]["RT"] for n in fam["p30"]]))
d_gt = float(np.median([out["plgt_S"]["per_model"][n]["RT"] - out["plgt_N"]["per_model"][n]["RT"] for n in fam["plgt"]]))
out["verdicts"] = {
    "F4-1": m["RT"] >= 0.95,
    "F4-2": m["IIA"] >= 0.80 and m["flip_collateral"] <= 0.05,
    "F4-3": m["err_repro"] >= 0.80 and m["err_repro"] >= m["err_repro_input"] + 0.15,
    "F4-4": m["RT_swap"] >= m["RT"] - 0.05 and m["IIA_swap"] >= 0.60,
    "F4-5": d_p30 >= d_gt + 0.02, "RT_S_minus_N_p30": d_p30, "RT_S_minus_N_plgt": d_gt}
json.dump(out, open("theory/results/score_f4.json", "w", newline="\n"), indent=1)
for k in ("p30_S", "p30_N", "plgt_S", "plgt_N"):
    print(k, {kk: round(v, 3) for kk, v in out[k]["median"].items()})
print(out["verdicts"])
