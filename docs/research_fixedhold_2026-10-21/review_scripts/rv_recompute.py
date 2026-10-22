"""REVIEW_FH $0 recomputation from committed files only (fh_check_cpu.<run>.json, eval_khorizon.json, eval_vb.json). No model. Writes rv_recompute.out.json (LF).
R1 null H(64) mean / SD over the six models (acc.intact['64'][n], n 1-8). R2 per-item link of drift speed and margin slope (A1 'mdot ~ omega'):
omega_i = mean over t 24..64 of acos(1 - step_ang_v), slope_i = OLS margin slope t 16..48 (items with margin_16 > 0); pooled Pearson per model and the
lineage ratio slope / omega. R3 FT/self gains at K' 64 vs their base. R4 DEPTH noise: acc_n2 n 9-11 of every 48k/FT model."""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, ".."); RUNS = os.path.join(HERE, "..", "..", "..", "boxes", "runs_fsq")
M = ["s213_d3t_s1", "s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_d3t_s2", "s216_r1_s2"]
out = {}
h64 = {r: float(np.mean([json.load(open(os.path.join(RUNS, r, "eval_khorizon.json")))["acc"]["intact"]["64"][str(n)] for n in range(1, 9)])) for r in M}
a = np.array(list(h64.values())); out["R1"] = {"h64": h64, "mean": float(a.mean()), "sd_ddof1": float(a.std(ddof=1)), "sd_ddof0": float(a.std())}
out["R2"] = {}
for r in M:
    d = json.load(open(os.path.join(D, "fh_check_cpu.%s.json" % r))); om, sl, v = [], [], []
    for n in ("1", "4", "8"):
        o = d[n]
        for i in range(len(o["y"])):
            if o["margin"][16][i] <= 0:
                continue
            om.append(np.mean([math.acos(max(-1, min(1, 1 - o["step_ang_v"][t][i]))) for t in range(24, 65)]))
            v.append(np.mean([o["step_ang_v"][t][i] for t in range(24, 65)]))
            sl.append(np.polyfit(np.arange(16, 49), [o["margin"][t][i] for t in range(16, 49)], 1)[0])
    om, sl = np.array(om), np.array(sl)
    out["R2"][r] = {"omega_mean": float(om.mean()), "slope_mean": float(sl.mean()), "slope_over_omega": float(sl.mean() / om.mean()),
                    "pearson_item_omega_slope": float(np.corrcoef(om, sl)[0, 1]), "n_items": int(len(om))}
base = {"s215_ftctrl_s1": "s213_d3t_s1", "s215_ftself_s1": "s213_d3t_s1", "s216_r2_s1": "s213_d3t_s1", "s216_r1_s2": "s216_d3t_s2"}
g = {r: h64[r] - h64[b] for r, b in base.items()}; ga = np.array(list(g.values()))
out["R3"] = {"gains_k64": g, "mean": float(ga.mean()), "sd_ddof1": float(ga.std(ddof=1))}
out["R4"] = {r: float(np.mean([json.load(open(os.path.join(RUNS, r, "eval_vb.json")))["acc_n2"][str(n)] for n in (9, 10, 11)])) for r in M}
for k, v in out.items():
    print(k, json.dumps(v)[:900])
json.dump(out, open(os.path.join(HERE, "rv_recompute.out.json"), "w", newline="\n"), indent=1)
