"""Score THEORY §55 (exp27 fall-back kill, KT1/KT2) and §56 (exp28 interchange, NF1-NF3); by Experiments (handover).
    python theory/score_exp27_28.py
"""
import json

import numpy as np

out = {}
K = json.load(open("theory/results/exp27_fallback_kill.json"))
ins = {m: v for m, v in K.items() if v.get("inside") in (True, "True", "true")}
med = lambda xs: float(np.median(xs)) if xs else None
dk = [v["kill"][0] - v["base"][0] for v in ins.values()]
dp = [v["pos"][0] - v["base"][0] for v in ins.values()]
db = [v["kill"][1] - v["base"][1] for v in ins.values()]
kt2 = [float(np.mean(np.array(v["probe_kill"][6:12]) - np.array(v["probe_base"][6:12]))) for v in ins.values()]
mk, mp, mb = med(dk), med(dp), med(db)
if mp is None or mp <= 0:
    kt1 = "uninformative"
elif mk <= 0.25 * mp and abs(mb) <= 0.01:
    kt1 = "KT1 met (fall-back not needed: kill costs <= 1/4 of a position-matched control)"
elif mk >= 0.5 * mp:
    kt1 = "killed (the fall-back carries the load)"
else:
    kt1 = "partial"
out["s55"] = {"models_inside": sorted(ins), "n": len(ins), "dBCE_kill": mk, "dBCE_pos": mp, "dbal_kill": mb,
              "KT1": kt1, "KT2_probe_kill_minus_base_6_11": med(kt2), "KT2_met": (med(kt2) is not None and med(kt2) <= 0.02)}
I = json.load(open("theory/results/exp28_interchange.json"))
fam = {"p30": ("p30_s", 5), "pl_gt": ("pl_gt", 11), "pl_t8": ("pl_t8", 11), "pclr1": ("pclr1", 5)}
res = {}
for name, (pre, f) in fam.items():
    ms = [v[f"f{f}"] for m, v in I.items() if m.startswith(pre)]
    inf = [x for x in ms if x.get("informative") in (True, "True", "true")]
    g = lambda x: x["G_by_age"].get(str(f - 1), x["G_by_age"].get(f - 1))
    res[name] = {"f": f, "models": len(ms), "informative": len(inf), "G_f-1": med([g(x) for x in inf]),
                 "dG": med([x["dG"] for x in inf]), "AR_closed": med([x["AR_closed"] for x in inf]),
                 "G_root": med([x["G_root"] for x in inf])}
gt = res["pl_gt"]
nf1 = ("normal form" if gt["G_f-1"] <= 0.25 and gt["dG"] <= 0.10 else "age-causal" if gt["dG"] >= 0.25 else "partial")
out["s56"] = {"families": res, "NF1_pl_gt": nf1, "NF2_p30_dG>=0.25": res["p30"]["dG"] >= 0.25,
              "NF3_p30_AR_closed>=0.5": res["p30"]["AR_closed"] >= 0.5}
json.dump(out, open("theory/results/score_exp27_28.json", "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
