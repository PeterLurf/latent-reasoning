"""SD2 draft numbers ($0, CPU, 1-2 threads; read-only on the repo). Inputs: boxes/runs_fsq/{s213_d3t_s1, s215_ftself_s1}/eval_vb.json (acc_n2, acc_k),
s215_ftself_s1/eval_chainlabel_fresh.json, s213_d3t_s1/eval_chainlabel_fresh.json, s215_ftself_s1_mem / _gtime, s215_ftctrl_s1/log.jsonl, s215_ftself_s1/log.jsonl,
scripts/evalmem_cpu.jsonl. Output: sd2_numbers.json beside this file.
Sections: (1) crossings n50 / n90 and margins; (2) hypothesis panels for round 2 (n 13..24, K' 0..28) and their readings (M2, M_far, n50, STOP-LOC-2,
K'-BEYOND-2); (3) per-n sampling SEs; (4) GATE-2 exact OC; (5) label noise / kappa thresholds; (6) pool sizing; (7) memory (train, eval); (8) time and cost;
(9) OC arithmetic. Every hypothesis is a CONJECTURE; numbers are exact arithmetic given it."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"
ev0 = json.load(open(R + "s213_d3t_s1/eval_vb.json")); ev1 = json.load(open(R + "s215_ftself_s1/eval_vb.json"))
FLOOR = 0.11                                                               # chance + bias level (base n 13-16: 0.141/0.127/0.135/0.115) [ASSUMED]
KS1 = list(range(17)) + [24]
out = {}


def acc_n2(ev):
    return {int(n): v for n, v in ev["acc_n2"].items()}


def acc_k(ev):
    return {int(k): {int(n): v for n, v in d.items()} for k, d in ev["acc_k"].items()}


a0, a1 = acc_n2(ev0), acc_n2(ev1); K0, K1 = acc_k(ev0), acc_k(ev1)


def crossing(a, level, lo, hi):
    for n in range(lo, hi):
        if a[n] >= level > a[n + 1]:
            return n + (a[n] - level) / (a[n] - a[n + 1])
    return None


# (1) crossings
cr = {}
for tag, a, B in (("base_D3T", a0, 8), ("round1_FTSELF", a1, 12)):
    c50, c90 = crossing(a, 0.5, B, 16), crossing(a, 0.9, B, 16)
    cr[tag] = {"B": B, "n50": c50, "m50": c50 - B, "m50_over_B": (c50 - B) / B, "n90": c90, "m90": c90 - B, "m90_over_B": (c90 - B) / B}
out["crossings"] = cr

# (2) hypothesis panels. a1 extension beyond n 16 (round 1 not evaluated there): excess over FLOOR decays with the trend of the last ratios [ASSUMED]
ex = {n: a1[n] - FLOOR for n in (14, 15, 16)}; r1, r2 = ex[15] / ex[14], ex[16] / ex[15]; r3 = r2 * (r2 / r1)
A1x = dict(a1); A1x[17] = FLOOR + ex[16] * r3; A1x[18] = FLOOR + (A1x[17] - FLOOR) * r3 * (r2 / r1)
for n in range(19, 29):
    A1x[n] = FLOOR
out["a1_extension_assumed"] = {n: round(A1x[n], 4) for n in range(16, 21)}


def interp_n(a, x):
    lo = int(math.floor(x)); f = x - lo
    return a[lo] if f == 0 else (1 - f) * a[lo] + f * a[lo + 1]


def K1x(k, n):
    """round-1 acc at K' k for item depth n, extended: n > 16 -> the extension (flat in k once k >= n - 1, else floor); k > 16 -> K' 16..24 linear."""
    if n > 16:
        return A1x[n] if k >= n - 1 else FLOOR
    if k <= 16:
        return K1[k][n]
    return K1[16][n] + (K1[24][n] - K1[16][n]) * min(1.0, (k - 16) / 8)


def K1xf(k, n):                                                             # bilinear in (k, n) for fractional arguments
    if k < 0:
        return 0.0
    k0, n0 = int(math.floor(k)), int(math.floor(n)); fk, fn = k - k0, n - n0
    g = lambda kk, nn: K1x(kk, max(1, nn))  # noqa: E731
    return ((1 - fk) * (1 - fn) * g(k0, n0) + fk * (1 - fn) * g(k0 + 1, n0) + (1 - fk) * fn * g(k0, n0 + 1) + fk * fn * g(k0 + 1, n0 + 1))


NS2, KS2 = list(range(13, 25)), list(range(0, 29))


def panel(kind):
    """-> (a2 {n: acc at K' n + 2}, A2 {k: {n: acc}}) for n 13..24, K' 0..28."""
    A2 = {k: {} for k in KS2}; a2 = {}
    for n in NS2:
        for k in KS2:
            if kind == "H_shift":                                           # translate round 1 by +4 in n and K'
                v = K1x(k - 4, n - 4) if k >= 4 else FLOOR
            elif kind == "H_prop":                                          # round 1's picture scaled about the boundary by 12 / 16 (d / B invariant)
                v = K1xf(12 + 0.75 * (k - 16), 12 + 0.75 * (n - 16))
            elif kind == "SHRINK_base_margin":                              # margin back to the base's: D3-T translated by +8
                v = (K0[k - 8][n - 8] if (k - 8) in K0 and n - 8 <= 16 else (K0[16][n - 8] if k - 8 > 16 and n - 8 <= 16 else FLOOR)) if k >= 8 else FLOOR
            A2[k][n] = max(v, 0.0)
        a2[n] = A2[n + 2][n] if n + 2 <= 28 else A2[28][n]
    return a2, A2


def readings(a2, A2):
    M2 = sum(a2[n] for n in range(17, 21)) / 4; Mfar = sum(a2[n] for n in range(21, 25)) / 4
    c50 = crossing({**{n: 1.0 for n in range(1, 13)}, **a2}, 0.5, 13, 24); c90 = crossing({**{n: 1.0 for n in range(1, 13)}, **a2}, 0.9, 13, 24)
    G = {}
    for k in range(13, 24):
        ns = list(range(k + 1, min(k + 4, 24) + 1)); G[k] = sum(a2[n] - A2[k][n] for n in ns) / len(ns)
    stop = next((k for k in range(13, 24) if G[k] <= 0.02 + 1e-9), 24)
    KB = sum(a2[n] - A2[18][n] for n in (19, 20)) / 2
    return {"a2": {n: round(a2[n], 4) for n in NS2}, "M2_n17_20": M2, "M_far_n21_24": Mfar, "n50": c50, "m50": None if c50 is None else c50 - 16,
            "n90": c90, "STOP_G": {k: round(v, 4) for k, v in G.items()}, "STOP": stop, "KB2_n19_20_vs_k18": KB}


pan = {}
for kind in ("H_shift", "H_prop", "SHRINK_base_margin"):
    a2, A2 = panel(kind); pan[kind] = readings(a2, A2)
fullp = {n: v for n, v in zip(NS2, [1.0, 1.0, 1.0, 1.0, 0.97, 0.95, 0.93, 0.90, 0.88, 0.85, 0.82, 0.80])}
pan["H_full_floor"] = {"a2": fullp, "M2_n17_20": sum(fullp[n] for n in range(17, 21)) / 4, "M_far_n21_24": sum(fullp[n] for n in range(21, 25)) / 4,
                       "STOP": "24+ (no stop by construction)", "note": "H_full stated as a floor: >= these values; flat in K' past n"}
out["panels"] = pan

# (3) sampling SEs at 1,000 items per n
se = {k: {n: math.sqrt(max(v, 1e-9) * (1 - v) / 1000) for n, v in p["a2"].items()} for k, p in pan.items() if k != "H_full_floor"}
out["se_per_n"] = {k: {n: round(v, 4) for n, v in d.items()} for k, d in se.items()}
out["se_M2"] = {k: math.sqrt(sum(se[k][n] ** 2 for n in range(17, 21))) / 4 for k in se}
out["M2_gap_in_SE"] = {k: (0.80 - pan[k]["M2_n17_20"]) / out["se_M2"][k] for k in se}


# (4) GATE-2 exact OC (rule as GATE-1: pi_pool >= 0.995, pi_n >= 0.99 every n; integer cutoffs on kept-wrong counts), kept counts = 2,000 x FT-SELF keep
def binom_pmf(N, p):
    lg = [math.lgamma(N + 1) - math.lgamma(k + 1) - math.lgamma(N - k + 1) for k in range(N + 1)]
    return np.array([math.exp(lg[k] + (k * math.log(p) if p > 0 else (0 if k == 0 else -1e300)) + ((N - k) * math.log(1 - p) if p < 1 else 0)) for k in range(N + 1)])


fr1 = json.load(open(R + "s215_ftself_s1/eval_chainlabel_fresh.json"))["by_n"]; fr0 = json.load(open(R + "s213_d3t_s1/eval_chainlabel_fresh.json"))["by_n"]
Nn = [int(round(2000 * fr1[str(n)]["keep"])) for n in range(13, 17)]; c_n = [int(math.floor(0.01 * x)) for x in Nn]; c_pool = int(math.floor(0.005 * sum(Nn)))
goc = {}
for pi in (0.9995, 0.999, 0.997, 0.995, 0.993, 0.99):
    q = 1 - pi; dist = np.array([1.0])
    for N_, c_ in zip(Nn, c_n):
        pm = binom_pmf(N_, q)[:c_ + 1]; dist = np.convolve(dist, pm)
    goc[str(pi)] = float(dist[:c_pool + 1].sum())
out["gate2_oc"] = {"N_kept_n13_16": Nn, "c_n": c_n, "c_pool": c_pool, "P_pass_by_pi": goc}


# (5) label noise at W2 = n 13..16 (FT-SELF chain, fresh, 1,000 per n); kappa thresholds (PROOFS S2(b) forms) [PROVED UNDER A1', A2, A3']
def cp_upper(k, n, a=0.05):
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2; cdf = sum(math.comb(n, j) * mid ** j * (1 - mid) ** (n - j) for j in range(k + 1))
        lo, hi = (mid, hi) if cdf > a else (lo, mid)
    return hi


lab = {}
for tag, fr in (("FTSELF", fr1), ("D3T", fr0)):
    kw = {n: int(round(fr[str(n)]["n_kept"] * (1 - fr[str(n)]["prec_kept"]))) for n in range(13, 17)}
    nk = {n: fr[str(n)]["n_kept"] for n in range(13, 17)}; ns_ = {n: int(round(1000 * (1 - fr[str(n)]["precision_strict"]))) for n in range(13, 17)}
    K_ = sum(kw.values()); NK = sum(nk.values()); NS_ = sum(ns_.values())
    P16 = fr["16"]["precision_strict"]; phi = K_ / NS_ if NS_ else 0.0; w = K_ / NK; w_up = cp_upper(K_, NK); phi_up = cp_upper(K_, NS_)
    ebar16 = (1 - P16) / 16
    lab[tag] = {"kept_wrong_n13_16": kw, "kept": nk, "nonstrict": ns_, "w_pool": w, "w_pool_cp95_upper": w_up, "pi_pool": 1 - w, "pi_pool_cp95_lower": 1 - w_up,
                "phi": phi, "phi_cp95_upper": phi_up, "P16": P16, "ebar16_lower": ebar16,
                "kappa_avalanche_phi_form_point": P16 / (phi * 16) if phi > 0 else None, "kappa_avalanche_phi_form_phi_up": P16 / (phi_up * 16),
                "kappa_doubling_w_form_point": ebar16 / w if w > 0 else None, "kappa_doubling_w_form_w_up": ebar16 / w_up}
out["label_noise_W2"] = lab
# two W-4 rounds (PROOFS S2(b)): prod (1 + kappa phi n_q / P_q), kappa 1, phi 0.010 (round-1 base CP-upper) and FT-SELF's phi at n 16
P16f = lab["FTSELF"]["P16"]   # (fix: P16 after the loop was D3-T's)
out["two_round_factor_kappa1"] = {"phi_0.010": (1 + 0.010 * 12 / 0.976) * (1 + 0.010 * 16 / P16f), "phi_FTSELF_point": (1 + 0.0049 * 12 / 0.976) * (1 + lab["FTSELF"]["phi"] * 16 / P16f)}

# (6) pool sizing: self items consumed = 64 x Bin(12000, p_self 0.5)
mean_c, sd_c = 64 * 6000, 64 * math.sqrt(12000 * 0.25); POOL = 8 * 56000
out["pool"] = {"records": POOL, "per_n": 56000, "expected_consumed": mean_c, "sd": sd_c, "z_exhaust": (POOL - mean_c) / sd_c,
               "round1_cursor_measured": 383488, "per_n_expected": mean_c / 8, "round1_per_n_expected": mean_c / 4}

# (7) memory. train: PROOFS §11 model with T_ref 31 + AdamW 0.93 (GiB, bs 64) [ASSUMED form; validated on round 1 below]
sys.path.insert(0, "D:/Latent")
import boxes.nested170 as N  # noqa: E402
rg = np.random.default_rng(12345); T16 = sorted((len(N.render(N.generate(rg, 16)[0])) + 2 for _ in range(57000)), reverse=True)


def gib(K, T, micro):
    S = sum(t + 1 for t in range(1, K + 1)); act = 0.58 * K * T / 31 + 0.0258 * (T / 31) ** 2 * S
    return 2.0 + 0.93 + act / micro


mem1 = json.load(open(R + "s215_ftself_s1_mem/mem_check.json"))
out["memory_train"] = {"deep_T_64_longest_n16_of_57k": {"max": T16[0], "64th": T16[63]}, "round1_check": {"model_K16_T47_micro2": gib(16, 47, 2),
                       "measured_mem_check_micro2_2steps": mem1["peak_gib_train_step"], "model_K16_T47_micro1": gib(16, 47, 1), "measured_micro1": "OOM (24 GB)"},
                       "round2_K20": {str(T): {str(mc): round(gib(20, T, mc), 2) for mc in (1, 2, 4)} for T in (59, 61, 63, T16[0])}}
# eval: CPU fp32 slopes (GiB per item) from evalmem_cpu.jsonl (B 8 -> 16 difference; heap reuse makes small-B deltas biased low)
em = [json.loads(l) for l in open(os.path.join(HERE, "evalmem_cpu.jsonl"))]
def slope(n, K):  # noqa: E302
    d = {e["B"]: e["delta_peak_gib"] for e in em if e["n"] == n and e["K"] == K and e["how"] == "longest"}
    return (d[16] - d[8]) / 8 if 16 in d and 8 in d else None
s16, s24 = slope(16, 24), slope(24, 28)
pk3090 = mem1["peak_gib_eval_n16_k24"]
out["memory_eval"] = {"slope_gib_per_item_n16_k24": s16, "slope_gib_per_item_n24_k28": s24,
                      "check_3090_bf16_bs64_n16_k24": {"predicted_rollout": 64 * s16, "measured_minus_weights_adamw": pk3090 - 0.47 - 0.93},
                      "3070_fp32_n24_k28": {str(bs): round(bs * s24 + 0.47, 2) for bs in (8, 16, 32)},
                      "3070_fp32_n16_k24_bs32_round1": round(32 * s16 + 0.47, 2)}

# (8) time and cost. measured: forced self K16 T47 micro2 0.896 s, forced plain 0.439 s (G-TIME); real FT-SELF 7,260 s, FT-CTRL 4,877 s for 12k
gt = json.load(open(R + "s215_ftself_s1_gtime/timing.json")); t_self1, t_plain = gt["sec_per_step_self"], gt["sec_per_step_plain"]
logS = [json.loads(l) for l in open(R + "s215_ftself_s1/log.jsonl") if '"loss"' in l]; logC = [json.loads(l) for l in open(R + "s215_ftctrl_s1/log.jsonl") if '"loss"' in l]
real_S, real_C = logS[-1]["sec"], logC[-1]["sec"]; plain_real = real_C / 12000; self_real1 = (real_S - 6008 * plain_real) / 5992
# two bounding time models for the round-2 forced deep self step (K 20, T_deep, micro 4)
a_, b_ = np.linalg.solve(np.array([[12 * 33, 33 ** 2 * sum(t + 1 for t in range(1, 13))], [16 * 47, 47 ** 2 * sum(t + 1 for t in range(1, 17))]]), np.array([t_plain, t_self1]))
Td = T16[63]
t_compute = a_ * 20 * Td + b_ * Td ** 2 * sum(t + 1 for t in range(1, 21)); t_launch = t_self1 * (20 * 4) / (16 * 2)
ratio_real = self_real1 / t_self1
proj = {}
for nm, ts in (("compute_bound", t_compute), ("launch_bound", t_launch)):
    forced = 0.5 * ts + 0.5 * t_plain; real = 6000 * ts * ratio_real + 6000 * plain_real
    proj[nm] = {"t_self_forced": ts, "forced_proj_h": forced * 12000 / 3600 + 0.4, "real_h": real / 3600 + 0.4,
                "cost_forced_lo_hi": [(forced * 12000 / 3600 + 0.4) * 0.17, (forced * 12000 / 3600 + 0.4) * 0.26],
                "cost_real_lo_hi": [(real / 3600 + 0.4) * 0.17, (real / 3600 + 0.4) * 0.26]}
out["time"] = {"t_self1_forced": t_self1, "t_plain_forced": t_plain, "self_real1": self_real1, "plain_real": plain_real, "real_over_forced_self": ratio_real,
               "round1_real_h": real_S / 3600, "ctrl_real_h": real_C / 3600, "round2": proj, "time_model_a_b": [a_, b_]}

# (9) OC arithmetic (priors ASSUMED; pre-stated round-1 shift: GROW -> P(H_grow) x 1.5, cap 0.8, the rest renormalised in round-1 proportions)
pr1 = {"H_shift": 0.65, "H_grow": 0.25, "H_full": 0.03, "SHRINK": 0.07}
g2 = min(0.8, pr1["H_grow"] * 1.5); rest = 1 - g2; base_rest = 1 - pr1["H_grow"]
pr2 = {"H_grow": g2, **{k: pr1[k] / base_rest * rest for k in ("H_shift", "H_full", "SHRINK")}}
P_DBS = {"H_shift": 0.05, "H_grow": 0.60, "H_full": 0.95, "SHRINK": 0.0}; P_MG = {"H_shift": 0.10, "H_grow": 0.75, "H_full": 1.0, "SHRINK": 0.0}
P_KB = {"H_shift": 0.05, "H_grow": 0.45, "H_full": 0.90, "SHRINK": 0.0}
P_ABS, kept = 0.75, 0.9 * 0.9 * 0.70
sd2_self = P_ABS * kept
dbs = sum(pr2[h] * P_DBS[h] for h in pr2); mg = sum(pr2[h] * P_MG[h] for h in pr2); kb = sum(pr2[h] * P_KB[h] for h in pr2)
c2 = sd2_self * dbs; c2_both_indep = sum(pr2[h] * (sd2_self * P_DBS[h]) ** 2 for h in pr2)
c2_both_corr = sum(pr2[h] * sd2_self * (0.85 * P_DBS[h]) * P_DBS[h] for h in pr2)  # the 2nd seed's ABSORB / kept given the 1st's: 0.85 (correlated)
out["oc"] = {"priors_round2_given_ABSORB": pr2, "P_ABSORB2": P_ABS, "kept_factor": kept, "P_SD2_SELF": sd2_self, "P_DBS2_given_ABSORB": dbs, "P_MG2": mg, "P_KB2": kb,
             "P_SD2_C2_one_seed": c2, "P_SD2_C2_given": {h: sd2_self * P_DBS[h] for h in pr2},
             "P_SD2_C2_two_seeds_independent": c2_both_indep, "P_SD2_C2_two_seeds_correlated": c2_both_corr,
             "P_SD2_C2_MEM": c2 * 0.25, "P_DBS_alone_given": P_DBS, "P_MG_given": P_MG, "P_KB_given": P_KB}

# labelling time (FT-SELF chain sec_per_item at 1,000 per n, 3070, fp32, bs 32; round-1 pool rate check: 3,398 s for 451k labelled)
spi = {n: fr1[str(n)]["sec_per_item"] for n in range(9, 17)}; keep1 = {n: fr1[str(n)]["keep"] for n in range(9, 17)}
lab_s = sum(56000 / keep1[n] * spi[n] for n in range(9, 17)); gate_s = sum(2000 * spi[n] for n in range(9, 17))
out["labelling"] = {"sec_per_item": spi, "pool_sec": lab_s, "gate_sec": gate_s, "round1_pool_rate_s_per_item": 3398 / (110592 * 2 + 114688 * 2)}
json.dump(out, open(os.path.join(HERE, "sd2_numbers.json"), "w", newline="\n"), indent=1, default=float)
print(json.dumps({"crossings": cr, "M2": {k: v.get("M2_n17_20") for k, v in pan.items()}, "Mfar": {k: v.get("M_far_n21_24") for k, v in pan.items()},
                  "STOP": {k: v.get("STOP") for k, v in pan.items()}, "KB2": {k: v.get("KB2_n19_20_vs_k18") for k, v in pan.items()},
                  "n50": {k: v.get("n50") for k, v in pan.items()}}, indent=1, default=float))
print("gate2", out["gate2_oc"]); print("label", json.dumps(lab, indent=1, default=float)); print("mem", json.dumps(out["memory_train"], default=float))
print("evalmem", out["memory_eval"]); print("time", json.dumps(out["time"], indent=1, default=float)); print("oc", json.dumps(out["oc"], indent=1, default=float))
print("labelling", out["labelling"], "pool", out["pool"], "two_round", out["two_round_factor_kappa1"], "ext", out["a1_extension_assumed"])
