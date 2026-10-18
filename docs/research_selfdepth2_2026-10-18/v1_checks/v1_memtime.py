"""SD2_v1 check 4 ($0, CPU, 1 thread, idle priority; read-only on the repo, no bytecode). Memory, time, labelling and cost under D10 (pool 2 : 1 by window,
74,000 records per n at n 13..16 and 37,000 at n 9..12) and D4 (auto-micro self loop {1, 2, 4, 8}).
(a) Training memory, PROOFS' model GiB(K, T, m) = 2 + 0.93 + [0.58 K T / 31 + 0.0258 (T / 31)^2 sum_{t=1..K} (t + 1)] / m [ASSUMED form; validated once on
    round 1: 14.47 projected vs 13.91 measured]. Peak shape = the LONGEST n-16 record (G-S9's micro-batch 0; PROOFS #27): T max over 74,000 / 0.957 n-16
    draws (the drawn set bounds the kept set), 5 generator seeds.
(b) Time: PROOFS' compute-bound refit (two measured points: plain K 12 / T 33 and round-1 forced self K 16 / G-TIME chunk max-T 41.5) predicted at the
    D10 G-TIME chunk shape (pool-order chunks of 16, K 20 forced); launch-bound t_self = 0.896 s x (20 x micro) / (16 x 2). Real / forced from the
    per-micro-batch E[Kb] (v1_support_d10.json), calibrated on round 1's measured 0.897.
(c) G-S10's registered projection form (0.5 forced self + 0.5 forced plain, 12k steps, + 0.4 h) at $0.17-0.26/h; micro 8 contingency.
(d) Labelling time on the 3070 at FT-SELF's measured chain sec/item (eval_chainlabel_fresh.json; D3-T's for the s2 base labeller as a proxy).
(e) Family cost, admissions, guards."""
import json
import math
import os
import sys

sys.dont_write_bytecode = True; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:  # noqa: BLE001
    pass
import boxes.nested170 as N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"
P5 = json.load(open(os.path.join(os.path.dirname(HERE), "proofs_sd2", "p5_mem_time.json")))
SUP = json.load(open(os.path.join(HERE, "v1_support_d10.json")))


def gib(K, T, m):
    S = sum(t + 1 for t in range(1, K + 1)); return 2.0 + 0.93 + (0.58 * K * T / 31 + 0.0258 * (T / 31) ** 2 * S) / m


T_of = lambda rg, n: len(N.render(N.generate(rg, n)[0])) + 2  # noqa: E731
out = {}
fr1 = json.load(open(R + "s215_ftself_s1/eval_chainlabel_fresh.json"))["by_n"]; fr0 = json.load(open(R + "s213_d3t_s1/eval_chainlabel_fresh.json"))["by_n"]
ndraw16 = int(math.ceil(74000 / fr1["16"]["keep"]))
mx = {}
for seed in (12345, 1, 2, 3, 4):
    rg = np.random.default_rng(seed); T16 = sorted((T_of(rg, 16) for _ in range(ndraw16)), reverse=True)
    mx[seed] = {"max": T16[0], "16th": T16[15], "64th": T16[63]}
out["mem"] = {"n16_draws": ndraw16, "T16_by_seed": mx, "gib_K20": {str(T): {str(m): round(gib(20, T, m), 2) for m in (1, 2, 4, 8)} for T in (59, 61, 63, 65)},
              "round1_validation": P5["round1"]}
# (b) time
rg = np.random.default_rng(7)
gen = {n: [T_of(rg, n) for _ in range(4000)] for n in range(9, 17)}
wts = {**{n: 1 / 12 for n in range(9, 13)}, **{n: 1 / 6 for n in range(13, 17)}}
ns = np.array(list(wts)); ps = np.array([wts[n] for n in ns])


def chunk_maxT(b, reps=4000):
    vals = []
    for _ in range(reps):
        nn = rg.choice(ns, b, p=ps); vals.append(max(gen[int(n)][int(rg.integers(0, 4000))] for n in nn))
    return float(np.mean(vals))


cm16 = chunk_maxT(16); cm8 = chunk_maxT(8)
tp, ts1 = P5["time"]["t_plain_forced"], P5["time"]["t_self1_forced"]; T1c = P5["time"]["gtime_chunk_maxT_mean_r1_32"]
S12, S16, S20 = (sum(t + 1 for t in range(1, K + 1)) for K in (12, 16, 20))
det = (12 * 33) * (T1c ** 2 * S16) - (33 ** 2 * S12) * (16 * T1c)
a_ = (tp * T1c ** 2 * S16 - 33 ** 2 * S12 * ts1) / det; b_ = ((12 * 33) * ts1 - (16 * T1c) * tp) / det
t_comp = a_ * 20 * cm16 + b_ * cm16 ** 2 * S20; t_comp_5050 = a_ * 20 * P5["time"]["gtime_chunk_maxT_mean_r2_16"] + b_ * P5["time"]["gtime_chunk_maxT_mean_r2_16"] ** 2 * S20
t_launch = {m: ts1 * (20 * m) / (16 * 2) for m in (4, 8)}
kb_r1 = SUP["Kb_round1"]["32"]["E_Kb"] / 16; kb_r2 = SUP["Kb_D10"]["16"]["E_Kb"] / 20; rof = P5["time"]["real_over_forced"] * kb_r2 / kb_r1


def proj(ts):
    fh = (0.5 * ts + 0.5 * tp) * 12000 / 3600 + 0.4
    rh = (5992 * ts * rof + (12000 - 5992) * P5["time"]["plain_real"]) / 3600 + 0.4
    return {"t_self_forced": ts, "forced_h": fh, "real_h": rh, "cost_forced": [fh * 0.17, fh * 0.26], "cost_real": [rh * 0.17, rh * 0.26]}


out["time"] = {"gtime_chunk_maxT_D10_16": cm16, "gtime_chunk_maxT_D10_8": cm8, "gtime_chunk_maxT_5050_16": P5["time"]["gtime_chunk_maxT_mean_r2_16"],
               "refit_a_b": [a_, b_], "compute_bound_D10": proj(t_comp), "compute_bound_5050_PROOFS": proj(t_comp_5050), "launch_bound_micro4": proj(t_launch[4]),
               "launch_bound_micro8_contingency": proj(t_launch[8]), "real_over_forced_D10": rof, "EKb_ratio_r1_r2": [kb_r1, kb_r2]}
# (d) labelling (3070, fp32, bs 32)
spi1 = {n: fr1[str(n)]["sec_per_item"] for n in range(9, 17)}; keep1 = {n: fr1[str(n)]["keep"] for n in range(9, 17)}
spi0 = {n: fr0[str(n)]["sec_per_item"] for n in range(9, 13)}; keep0 = {n: fr0[str(n)]["keep"] for n in range(9, 13)}
CNT = {**{n: 37000 for n in range(9, 13)}, **{n: 74000 for n in range(13, 17)}}
pool_s = sum(CNT[n] / keep1[n] * spi1[n] for n in CNT); gate_s = sum(2000 * spi1[n] for n in range(9, 17))
r1s2_s = sum(110000 / keep0[n] * spi0[n] for n in range(9, 13)); g1s2_s = sum(2000 * spi0[n] for n in range(9, 13))
out["labelling"] = {"s1_r2_pool_h": pool_s / 3600, "s1_gate2_gate1p_h": gate_s / 3600, "s2_r1_pool_h_D3T_proxy": r1s2_s / 3600, "s2_gate1_h": g1s2_s / 3600,
                    "s2_r2_pool_h_FTSELF_proxy": pool_s / 3600, "draws_per_n_s1_r2": {n: int(math.ceil(CNT[n] / keep1[n])) for n in CNT}}
# (e) costs
lo_r2 = min(out["time"]["compute_bound_D10"]["cost_forced"][0], out["time"]["launch_bound_micro4"]["cost_forced"][0])
hi_r2 = max(out["time"]["compute_bound_D10"]["cost_forced"][1], out["time"]["launch_bound_micro4"]["cost_forced"][1])
cells = {"r2_s1": (lo_r2, hi_r2, 1.45, 1.70), "d3t_s2": (1.11, 1.70, 1.75, 2.00), "r1_s2": (0.45, 0.68, 0.95, 1.05), "r2_s2": (lo_r2, hi_r2, 1.45, 1.70)}
out["cost"] = {"cells": cells, "family_lo": sum(v[0] for v in cells.values()), "family_hi": sum(v[1] for v in cells.values()) + 0.05,
               "admission_sum": sum(v[2] for v in cells.values()), "guards_sum": sum(v[3] for v in cells.values()),
               "headroom_upper_vs_admission_r2": 1.45 / hi_r2 - 1}
json.dump(out, open(os.path.join(HERE, "v1_memtime.json"), "w", newline="\n"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
