"""LANG v1 $0: the A2 self-pool arithmetic (RT F3-1 blocker) from vbank.py's coin-per-step semantics, and the labelling compute plan
(RT F3-5 / F6-4).
vbank.py l.644 / l.755-766 [read]: rng_s = default_rng(seed + 21800); each step draws ONE coin rng_s.random() < p_self; a self step takes
the next bs (=64) pool records (no reuse; exhaustion raises) and then draws one U = rng_s.integers(0, self_kextra + 1) per record.
The new LANG trainer must keep exactly this rng_s order (G-L21), so the self-step count of a seed is known before launch.
    python -B v1_pool.py   -> v1_pool.json (reads v1_sym.json for T, |S|, L-RR chunks at D 9..12)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def replay(seed, steps=12000, p_self=0.5, bs=64, kextra=4):
    r = np.random.default_rng(seed + 21800); n = 0
    for _ in range(steps):
        if bool(r.random() < p_self):
            n += 1
            for _ in range(bs):
                r.integers(0, kextra + 1)
    return n


out = {}
n1 = replay(1); out["seed1_self_steps"] = n1; out["seed1_demand"] = n1 * 64
assert n1 == 5992 and n1 * 64 == 383488, "replay must reproduce the measured §216 s1 cursor (5,992 self steps, 383,488 records)"
others = [replay(s) for s in range(2, 22)]
out["seeds2_21_self_steps"] = {"min": min(others), "max": max(others), "mean": round(sum(others) / len(others), 1)}
sd = math.sqrt(12000 * 0.25)
KEPT_PER_D = 106000; NDEPTH = 4; kept = KEPT_PER_D * NDEPTH
out["kept_per_D"] = KEPT_PER_D; out["kept_total"] = kept
out["margin_vs_seed1"] = {"records": kept - n1 * 64, "ratio": round(kept / (n1 * 64), 4), "per_D_consumed_seed1": n1 * 64 // NDEPTH,
                          "per_D_spare_seed1": KEPT_PER_D - n1 * 64 // NDEPTH}
out["exhaustion_z_any_seed"] = round((kept / 64 - 6000) / sd, 2)                       # self steps needed to exhaust vs Binomial(12000, 1/2)
out["draft_pool_240k_exhausts_at_step"] = int(240000 / 64 / 0.5)
out["draw_cap_per_D"] = math.ceil(KEPT_PER_D / 0.2)

# ---------- labelling compute (forward only) ----------
sym = json.load(open(os.path.join(HERE, "v1_sym.json")))["bands"]
d = 768


def flops_prelude(T):
    return 6 * T * (24 * d * d + 2 * T * d)


def flops_core(S, T, passes):
    return sum(4 * S * (24 * d * d + 4 * (T + (t - 1) * S + S) * d) for t in range(1, passes + 1))


lab = {}
for D in ("9", "10", "11", "12"):
    T = sym[D]["T_mean"]; S = sym[D]["S_mean"]; ch = sym[D]["LRR_chunks_mean"]
    rr_step = flops_prelude(T) + flops_core(S, T, 2)                  # one L-RR step: re-rendered prelude + K = 2
    roll = flops_prelude(T) + flops_core(S, T, 16)                    # L-ROLL: one rollout to K' 16
    lab[D] = {"T": T, "S": S, "chunks": ch, "LRR_GFLOP_per_drawn": [round(rr_step * ch * f / 1e9, 1) for f in (1.0, 1.3)],
              "LROLL_GFLOP_per_drawn": round(roll / 1e9, 1)}
out["per_D"] = lab
mean_rr = [sum(lab[D]["LRR_GFLOP_per_drawn"][i] for D in lab) / 4 for i in (0, 1)]
mean_roll = sum(lab[D]["LROLL_GFLOP_per_drawn"] for D in lab) / 4
RATES = {"local_fp32_8TF": 8e12, "local_bf16_14TF": 14e12, "3090_bf16_20TF": 20e12}   # sustained [ASSUMED]; RT used 8 (local fp32) / 20 (3090)
plan = {}
for y in (0.2, 0.3, 0.5):
    drawn = kept / y
    row = {"drawn_total": int(drawn)}
    for name, rate in RATES.items():
        row["LRR_h_" + name] = [round(drawn * g * 1e9 / rate / 3600, 1) for g in mean_rr]
        row["LROLL_h_" + name] = round(drawn * mean_roll * 1e9 / rate / 3600, 1)
    row["LRR_3090_usd_at_0.21"] = [round(h * 0.21, 2) for h in row["LRR_h_3090_bf16_20TF"]]
    row["LRR_two_local_cards_wallclock_h_bf16"] = [round(h / 2, 1) for h in row["LRR_h_local_bf16_14TF"]]
    plan[str(y)] = row
gate = 2000 * 4 * 2                                                    # GATE-1-L: 2,000 per D, both chains
out["gate1_GPU_h_local_bf16"] = round(gate / 2 * (mean_rr[1] + mean_roll) * 1e9 / 14e12 / 3600, 2)
out["labelling_plan"] = plan
out["assumptions"] = "FLOP model: GPT-2 small fwd 24d^2 + attention 4*keys*d per token-layer; prelude causal (keys T/2); L-RR steps per drawn item = chunks x {1.0, 1.3} (rejected chains run to their stop rule); sustained rates ASSUMED"
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "v1_pool.json"), "w", newline="\n"), indent=1)
