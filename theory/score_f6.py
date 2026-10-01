"""Score THEORY §58 (F6) by Experiments. Inputs theory/runs/f6{a,y,m}_s<seed>/eval.json (eval_arch.py) and
theory/results/exp28_f6.json (f = 5). Counts among gated runs; medians over gated runs.
F6-1: extrap_K12 (aged) >= 6/10 and >= young + 3 and >= mismatch + 3.  F6-2: aged median dG(f5) <= 0.10, young >= 0.25.
F6-3: |median delayed bal-acc aged - young| <= 0.02.  Falsified if aged <= young + 1 on extrap_K12.
    python theory/score_f6.py
"""
import json

import numpy as np

SEEDS = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
I = json.load(open("theory/results/exp28_f6.json"))
D = json.load(open("theory/results/exp18_f6.json"))
res = {}
for arm, t in (("aged", "a"), ("young", "y"), ("mismatch", "m")):
    rows = {}
    for s in SEEDS:
        r = f"f6{t}_s{s}"; e = json.load(open(f"theory/runs/{r}/eval.json"))
        x = I.get(r, {}).get("f5", {})
        rows[r] = {"gate": bool(e["gate"]), "extrap": bool(e["gate"] and e["extrap_K12"]), "delayed": e["delayed"],
                   "d7_K12": e["d7_K12"], "far8_10": e["far8_10_K12"], "dG": x.get("dG"), "informative": x.get("informative"),
                   "d13_18_single24": float(np.mean(D[r]["single24"][12:18])) if r in D and "single24" in D[r] else None}
    g = [v for v in rows.values() if v["gate"]]
    med = lambda k, vs: float(np.median([v[k] for v in vs if v[k] is not None])) if [v for v in vs if v[k] is not None] else None
    res[arm] = {"runs": rows, "gated": len(g), "extrap": sum(v["extrap"] for v in rows.values()),
                "median_dG_f5_gated": med("dG", [v for v in g if v["informative"] in (True, "True")]),
                "median_delayed_gated": med("delayed", g), "median_d7_gated": med("d7_K12", g)}
a, y, m = res["aged"], res["young"], res["mismatch"]
out = {"arms": res,
       "F6-1": a["extrap"] >= 6 and a["extrap"] >= y["extrap"] + 3 and a["extrap"] >= m["extrap"] + 3,
       "F6-2": (a["median_dG_f5_gated"] is not None and a["median_dG_f5_gated"] <= 0.10) and (y["median_dG_f5_gated"] is not None and y["median_dG_f5_gated"] >= 0.25),
       "F6-3": abs(a["median_delayed_gated"] - y["median_delayed_gated"]) <= 0.02,
       "falsified": a["extrap"] <= y["extrap"] + 1}
json.dump(out, open("theory/results/score_f6.json", "w", newline="\n"), indent=1)
for k in ("aged", "young", "mismatch"):
    r = res[k]
    print(k, "gated", r["gated"], "extrap", r["extrap"], "median dG f5", r["median_dG_f5_gated"], "delayed", round(r["median_delayed_gated"], 4), "d7", round(r["median_d7_gated"], 3))
print({k: out[k] for k in ("F6-1", "F6-2", "F6-3", "falsified")})
