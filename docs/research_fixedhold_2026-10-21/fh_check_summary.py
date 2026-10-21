"""FIXED-HOLD $0 check C: summarise fh_check_cpu.<run>.json (check B) per model and relate it to eval_khorizon.json acc.intact (Note 216-15).
Per model, pooled over n {1, 4, 8} x 16 items: v = mean step_ang_v over t 24..64 (1 - cos(V_t, V_{t-1}), value slot); coh = coherence of the drift,
theta(drift_v at 64 vs n + 2) / (sum of per-step angles over t n + 3..64) (1 = every step in one direction, ~1/sqrt(L) = random walk);
mslope = OLS slope of margin_t over t 16..48 (items correct at t 16); m12 = margin at t 12; t_cross_pred = 12 + m12 / -mslope (linear margin model);
t_cross_obs = first t >= 12 with margin < 0 (64 + 1 if never); rms64 = rms_v at 64; clock64 = clock_mean_pred at 64; recok24 = rec_ok_v at 24.
Spearman rank correlation across the six models of v and mslope with the K' 64 intact mean (n 1-8). Writes fh_check_summary.out.json (LF).
Run: python -B docs/research_fixedhold_2026-10-21/fh_check_summary.py"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "..", "boxes", "runs_fsq")
MODELS = ["s213_d3t_s1", "s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_d3t_s2", "s216_r1_s2"]


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


out = {}
for r in MODELS:
    d = json.load(open(os.path.join(HERE, "fh_check_cpu.%s.json" % r))); kh = json.load(open(os.path.join(RUNS, r, "eval_khorizon.json")))
    v, coh, ms, m12, tcp, tco, rms, clk, rok = [], [], [], [], [], [], [], [], []
    for n in ("1", "4", "8"):
        o = d[n]; ni = int(n); I = len(o["y"])
        for i in range(I):
            v.append(np.mean([o["step_ang_v"][t][i] for t in range(24, 65)]))
            steps = [math.acos(max(-1.0, min(1.0, 1 - o["step_ang_v"][t][i]))) for t in range(ni + 3, 65)]
            coh.append(math.acos(max(-1.0, min(1.0, 1 - o["drift_ang_v"][64][i]))) / max(1e-12, sum(steps)))
            mg = [o["margin"][t][i] for t in range(16, 49)]
            if o["margin"][16][i] > 0:
                s = np.polyfit(np.arange(16, 49), mg, 1)[0]; ms.append(s); m12.append(o["margin"][12][i])
                tcp.append(12 + o["margin"][12][i] / max(1e-9, -s))
                tco.append(next((t for t in range(12, 65) if o["margin"][t][i] < 0), 65))
            rms.append(o["rms_v"][64][i]); rok.append(o["rec_ok_v"][24][i])
        clk.append(o["clock_mean_pred"]["64"])
    acc64 = float(np.mean([kh["acc"]["intact"]["64"][str(n)] for n in range(1, 9)]))
    out[r] = {"v_step_24_64": float(np.mean(v)), "coherence": float(np.median(coh)), "margin_slope_16_48": float(np.mean(ms)), "margin_t12": float(np.mean(m12)),
              "t_cross_pred_median": float(np.median(tcp)), "t_cross_obs_median": float(np.median(tco)), "corr_tcross_pred_obs": float(np.corrcoef(tcp, tco)[0, 1]),
              "rms_v_64": float(np.mean(rms)), "clock64_mean_over_n": float(np.mean(clk)), "rec_ok_v_24": float(np.mean(rok)), "acc_intact_k64": acc64}
    print(r, {k: round(x, 5) for k, x in out[r].items()}, flush=True)
a = np.array([out[r]["acc_intact_k64"] for r in MODELS])
for k in ("v_step_24_64", "margin_slope_16_48", "coherence", "margin_t12"):
    x = np.array([out[r][k] for r in MODELS]); out.setdefault("spearman_vs_acc64", {})[k] = float(np.corrcoef(rank(x), rank(a))[0, 1])
print("spearman vs acc64", out["spearman_vs_acc64"])
json.dump(out, open(os.path.join(HERE, "fh_check_summary.out.json"), "w", newline="\n"), indent=1)
