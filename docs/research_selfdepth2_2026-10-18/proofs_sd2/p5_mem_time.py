"""PROOFS_SD2 check 5 ($0, CPU, 1 thread, idle priority; read-only on the repo, no bytecode).
(a) Training-memory model GiB(K, T, m) = 2 + 0.93 + [0.58 K T / 31 + 0.0258 (T / 31)^2 sum_{t=1..K} (t + 1)] / m (PROOFS §11; [ASSUMED] form).
    Which T? vb_loss pads each micro-batch (batch_x per chunk) to its own max T; G-S9 sorts the 64 longest n_top records descending, so micro-batch 0
    holds the longest record: the peak shape is T = max T of the pool's n_top records, not the 64th longest.
    Round-1 check: T of the round-1 pool's n-12 records (pool_r1.jsonl, read-only): max and 64th longest; the measured 13.91 GiB (micro 2, 2 steps).
    Round 2: T distribution of n-16 generator draws (57k, 5 seeds): max, 16th, 64th longest; GiB at those T, micro 1 / 2 / 4.
(b) Time model (draft §3.6) recomputed; chunk max-T of the forced G-TIME self steps (pool order, chunks of 32 (r1, micro 2) / 16 (r2, micro 4)).
(c) Eval slopes from scripts/evalmem_cpu.jsonl (B 4 -> 8 and 8 -> 16), projections, the 3090 check arithmetic.
(d) §8 cost arithmetic: per-cell, family totals, guards, admissions."""
import os, sys, json, math
sys.dont_write_bytecode = True; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "D:/Latent")
import numpy as np
import psutil
try:
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
import boxes.nested170 as N
HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"
SC = os.path.join(os.path.dirname(HERE), "scripts")


def gib(K, T, m):
    S = sum(t + 1 for t in range(1, K + 1)); return 2.0 + 0.93 + (0.58 * K * T / 31 + 0.0258 * (T / 31) ** 2 * S) / m


out = {}
# (a) round-1 pool n-12 T
T12 = []; Tall1 = {}
with open(R + "s215_pool/pool_r1.jsonl") as fh:
    for line in fh:
        r = json.loads(line); T_ = len(r["syms"]) + 2; Tall1.setdefault(r["n"], []).append(T_)
        if r["n"] == 12:
            T12.append(T_)
T12s = sorted(T12, reverse=True)
mem1 = json.load(open(R + "s215_ftself_s1_mem/mem_check.json"))
out["round1"] = {"n12_records": len(T12), "T_max": T12s[0], "T_16th": T12s[15], "T_32nd": T12s[31], "T_64th": T12s[63],
                 "trained_T_max_by_n": {n: max(v) for n, v in sorted(Tall1.items())},
                 "gib_K16_m2_at_Tmax": gib(16, T12s[0], 2), "gib_K16_m2_at_T47": gib(16, 47, 2), "gib_K16_m2_at_T64th": gib(16, T12s[63], 2),
                 "measured_peak_micro2_2steps": mem1["peak_gib_train_step"]}
# round-2: n-16 generator draws
r2 = {}
for seed in (12345, 1, 2, 3, 4):
    rg = np.random.default_rng(seed); T16 = sorted((len(N.render(N.generate(rg, 16)[0])) + 2 for _ in range(57000)), reverse=True)
    r2[seed] = {"max": T16[0], "16th": T16[15], "64th": T16[63]}
out["round2_T16_57k"] = r2
out["round2_gib_K20"] = {str(T): {str(m): round(gib(20, T, m), 2) for m in (1, 2, 4)} for T in (59, 61, 63, 65)}
# (b) time model
gt = json.load(open(R + "s215_ftself_s1_gtime/timing.json")); ts1, tp = gt["sec_per_step_self"], gt["sec_per_step_plain"]
S12, S16, S20 = (sum(t + 1 for t in range(1, K + 1)) for K in (12, 16, 20))
A_ = np.array([[12 * 33, 33 ** 2 * S12], [16 * 47, 47 ** 2 * S16]]); a_, b_ = np.linalg.solve(A_, np.array([tp, ts1]))
logS = [json.loads(l) for l in open(R + "s215_ftself_s1/log.jsonl") if '"loss"' in l]; logC = [json.loads(l) for l in open(R + "s215_ftctrl_s1/log.jsonl") if '"loss"' in l]
realS, realC = logS[-1]["sec"], logC[-1]["sec"]; nself1 = logS[-1]["n_self_steps"]; plain_real = realC / 12000; self_real = (realS - (12000 - nself1) * plain_real) / nself1
tm = {}
for Td in (59, 61, 55):
    tc = a_ * 20 * Td + b_ * Td ** 2 * S20; tl = ts1 * (20 * 4) / (16 * 2)
    for nm, t_ in (("compute", tc), ("launch", tl)):
        fh_ = (0.5 * t_ + 0.5 * tp) * 12000 / 3600 + 0.4; rh_ = (nself1 * t_ * self_real / ts1 + (12000 - nself1) * plain_real) / 3600 + 0.4
        tm["%s_T%d" % (nm, Td)] = {"t_self": t_, "forced_h": fh_, "real_h": rh_, "cost_forced": [fh_ * 0.17, fh_ * 0.26], "cost_real": [rh_ * 0.17, rh_ * 0.26]}
# chunk max-T of forced G-TIME self steps: round 1 pool order (n 9-12 uniform), chunks of 32; round 2 (n 9-16 uniform), chunks of 16
rg = np.random.default_rng(7)
pool1 = [t for n in (9, 10, 11, 12) for t in Tall1[n]]
c1 = [max(rg.choice(pool1, 32)) for _ in range(4000)]
gen = {n: [len(N.render(N.generate(rg, n)[0])) + 2 for _ in range(4000)] for n in range(9, 17)}
pool2 = [t for n in range(9, 17) for t in gen[n]]
c2 = [max(rg.choice(pool2, 16)) for _ in range(4000)]
out["time"] = {"a_b": [a_, b_], "t_self1_forced": ts1, "t_plain_forced": tp, "self_real1": self_real, "plain_real": plain_real, "real_over_forced": self_real / ts1,
               "n_self_steps_r1": nself1, "models": tm, "gtime_chunk_maxT_mean_r1_32": float(np.mean(c1)), "gtime_chunk_maxT_mean_r2_16": float(np.mean(c2))}
# (c) eval slopes
em = [json.loads(l) for l in open(os.path.join(SC, "evalmem_cpu.jsonl"))]
d = {(e["n"], e["K"], e["B"]): e["delta_peak_gib"] for e in em if e["how"] == "longest"}
sl = {}
for (n, K) in ((16, 24), (24, 28)):
    s48 = (d[(n, K, 8)] - d[(n, K, 4)]) / 4; s816 = (d[(n, K, 16)] - d[(n, K, 8)]) / 8; inter = d[(n, K, 16)] - 16 * s816
    sl["n%d_k%d" % (n, K)] = {"slope_4_8": s48, "slope_8_16": s816, "rel_diff": (s816 - s48) / s816, "intercept_8_16_fit": inter,
                              "proj_bs": {bs: round(bs * s816 + 0.47, 2) for bs in (8, 16, 32)}, "proj_bs_with_fit_intercept": {bs: round(bs * s816 + inter + 0.47, 2) for bs in (8, 16, 32)}}
out["eval_mem"] = sl
out["eval_mem"]["check_3090"] = {"pred_64x_slope": 64 * sl["n16_k24"]["slope_8_16"], "measured_minus": mem1["peak_gib_eval_n16_k24"] - 0.47 - 0.93,
                                 "note": "3090 mem_check eval = bf16 autocast, ONE coda (K' 24) [vbank.py l.853-857]; CPU = fp32, codas at every K' 0..K"}
# (d) costs
cells = {"r2_s1": (0.62, 1.27, 1.35, 1.60), "d3t_s2": (1.12, 1.70, 1.75, 2.00), "r1_s2": (0.45, 0.68, 0.95, 1.05), "r2_s2": (0.62, 1.27, 1.35, 1.60)}
lo = sum(v[0] for v in cells.values()); hi = sum(v[1] for v in cells.values())
out["cost"] = {"family_lo": lo, "family_hi": hi, "family_hi_plus_egress": hi + 0.05, "guards_sum": sum(v[3] for v in cells.values()), "admission_sum": sum(v[2] for v in cells.values()),
               "option_b_s2_only": [1.12 + 0.45 + 0.62, 1.70 + 0.68 + 1.27], "option_a_s2_only": [0.45 + 0.62, 0.68 + 1.27],
               "alt_a_total": [0.62 + 0.45 + 0.62, 1.27 + 0.68 + 1.27], "r1_forced_h": (0.5 * ts1 + 0.5 * tp) * 12000 / 3600 + 0.4}
gt0 = json.load(open(R + "s213_d3t_s1_gtime/timing.json")) if os.path.exists(R + "s213_d3t_s1_gtime/timing.json") else None
log0 = [json.loads(l) for l in open(R + "s213_d3t_s1/log.jsonl") if '"loss"' in l]
out["cost"]["d3t_s1"] = {"gtime_sec_per_step": gt0 and gt0.get("sec_per_step"), "forced_h_48k": gt0 and gt0["sec_per_step"] * 48000 / 3600, "real_h": log0[-1]["sec"] / 3600, "last_step": log0[-1]["step"]}
json.dump(out, open(os.path.join(HERE, "p5_mem_time.json"), "w", newline="\n"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
