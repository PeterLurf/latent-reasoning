"""Score THEORY §62 (noise-calibration test) by Experiments. Inputs theory/results/exp28_s62.json (f11: G_by_age['10'],
informative), exp18_s62.json (one loop at K' = 24: single24, d13-18 = indices 12..17), exp30_s62.json (step_d6_d7).
Pairs by parent seed. NC-1..NC-5 as registered (THEORY §62, 4f087eb).
    python theory/score_s62.py
"""
import json

import numpy as np

S = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
I = json.load(open("theory/results/exp28_s62.json")); D = json.load(open("theory/results/exp18_s62.json"))
C = json.load(open("theory/results/exp30_s62.json"))
def g10(r):
    x = I[r]["f11"]; return x["G_by_age"].get("10", x["G_by_age"].get(10)) if x.get("informative") in (True, "True") else None
d1318 = lambda r: float(np.mean(D[r]["single24"][12:18]))
arms = {a: {s: {"G10": g10(f"pn_{a}_s{s}"), "d13_18": d1318(f"pn_{a}_s{s}"), "step": float(C[f"pn_{a}_s{s}"]["step_d6_d7"])}
            for s in S} for a in ("ns", "ns45", "nb", "nu")}
med = lambda a, k: float(np.median([v[k] for v in arms[a].values() if v[k] is not None]))
pairs = lambda a, b, k: sum(1 for s in S if arms[a][s][k] is not None and arms[b][s][k] is not None and arms[a][s][k] > arms[b][s][k])
npair = lambda a, b, k: sum(1 for s in S if arms[a][s][k] is not None and arms[b][s][k] is not None)
diffs = [arms["nu"][s]["d13_18"] - arms["ns"][s]["d13_18"] for s in S]
allx = [(v["G10"], v["d13_18"]) for a in arms for v in arms[a].values() if v["G10"] is not None]
rx = np.argsort(np.argsort([x for x, _ in allx])); ry = np.argsort(np.argsort([y for _, y in allx]))
rho = float(np.corrcoef(rx, ry)[0, 1])
out = {"arms": arms,
       "NC-1": {"pairs_NS>NU": pairs("ns", "nu", "G10"), "n_pairs": npair("ns", "nu", "G10"), "med_NS": med("ns", "G10"), "med_NU": med("nu", "G10")},
       "NC-2": {"pairs_NU>NS": pairs("nu", "ns", "d13_18"), "median_diff": float(np.median(diffs))},
       "NC-3": {"med_NB_G10": med("nb", "G10"), "med_NB_d13_18": med("nb", "d13_18"), "med_NU_d13_18": med("nu", "d13_18")},
       "NC-4": {"spearman_G10_d13_18": rho, "n": len(allx)},
       "NC-5": {"med_step_NS": med("ns", "step"), "med_step_NU": med("nu", "step")},
       "descriptive_NS45": {"med_G10": med("ns45", "G10"), "med_d13_18": med("ns45", "d13_18"), "med_step": med("ns45", "step")}}
o = out
o["NC-1"]["met"] = o["NC-1"]["pairs_NS>NU"] >= 8 and o["NC-1"]["med_NS"] >= 0.5 and o["NC-1"]["med_NU"] <= 0.25
o["NC-2"]["met"] = o["NC-2"]["pairs_NU>NS"] >= 8 and o["NC-2"]["median_diff"] >= 0.10
o["NC-3"]["met"] = o["NC-3"]["med_NB_G10"] >= 0.5 and o["NC-3"]["med_NB_d13_18"] <= o["NC-3"]["med_NU_d13_18"] - 0.10
o["NC-4"]["met"] = rho <= -0.5
o["NC-5"]["met"] = o["NC-5"]["med_step_NS"] >= 0.10 and o["NC-5"]["med_step_NU"] <= 0.03
refs = {}
for fam in ("pl_gt", "pl_t8"):
    refs[fam] = float(np.median([float(C[f"theory/runs/{fam}_s{s}" if f"theory/runs/{fam}_s{s}" in C else f"{fam}_s{s}"]["step_d6_d7"]) for s in S]))
o["NC-5"]["reference_med_step"] = refs
json.dump(out, open("theory/results/score_s62.json", "w", newline="\n"), indent=1)
for k in ("NC-1", "NC-2", "NC-3", "NC-4", "NC-5", "descriptive_NS45"):
    print(k, out[k])
