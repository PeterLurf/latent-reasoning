"""PROOFS_SD2 check 6 + 7 ($0, CPU; independent rewrite).
(6) Exposure: seed-1 replay -> the self-step count is round 1's exactly (rng_s = default_rng(seed + 21800) consumed by 1 coin per step and 64 U per
    self step in both rounds, self_kextra 4, bs 64; vbank.py draw_step l.744-766), so 5,992 self steps / 383,488 items, not 6,000 +- 55.
    Per-n count from a random permutation of 8 x 56,000 (hypergeometric SD). Round-1 absorption point from s215_ftself_s1/log.jsonl (self_answer_ce
    per 100-step window): first-below and sustained-below thresholds, items per n at those steps.
(7) OC recompute (priors ASSUMED, as sd2_oc.py) + sensitivities: P(ABSORB-2) after the exposure correction; K'24-STABLE dependence on ABSORB-2;
    the 2/2 lineage figure; time-model refit at the G-TIME chunk shapes (p5_mem_time.json)."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"
L = [json.loads(l) for l in open(R + "s215_ftself_s1/log.jsonl")]
S = [(r["step"], r["self_answer_ce"], r["pool_cursor"]) for r in L if r.get("self_answer_ce") is not None]
ab = {}
for thr in (0.05, 0.02, 0.01):
    first = next(s for s in S if s[1] < thr); i = next(i for i in range(len(S)) if all(c < thr for _, c, _ in S[i:]))
    ab[str(thr)] = {"first_step": first[0], "first_items_per_n": first[2] / 4, "sustained_from_step": S[i][0], "sustained_items_per_n": S[i][2] / 4}
ab["step1000"] = [x for x in S if x[0] == 1000][0]; ab["step3000"] = [x for x in S if x[0] == 3000][0]
ab["windows_ge_0.03_after_3000"] = [(s, round(c, 4)) for s, c, _ in S if s > 3000 and c >= 0.03]
dl = json.load(open(R + "s215_ftself_s1/draws_log.json")); nself = sum(1 for d in dl if d["coin"])
Nc = 64 * nself; M = 8 * 56000
sd_n = math.sqrt(Nc * (1 / 8) * (7 / 8) * (M - Nc) / (M - 1)); sd_w = math.sqrt(Nc * 0.25 * (M - Nc) / (M - 1))
expo = {"round1_self_steps": nself, "round2_self_steps_seed1_replay": nself, "items": Nc, "per_n_mean": Nc / 8, "per_n_sd": sd_n, "window_mean": Nc / 2, "window_sd": sd_w,
        "pool_margin_records": M - Nc, "prior_model_6000_pm": math.sqrt(12000 * 0.25), "absorption": ab,
        "round2_per_n_over_round1_sustained_0.02": (Nc / 8) / ab["0.02"]["sustained_items_per_n"], "round2_per_n_over_round1_sustained_0.01": (Nc / 8) / ab["0.01"]["sustained_items_per_n"],
        "round2_per_n_over_round1_first_0.01": (Nc / 8) / ab["0.01"]["first_items_per_n"]}
# (7) OC
pr1 = {"H_shift": 0.65, "H_grow": 0.25, "H_full": 0.03, "SHRINK": 0.07}
PD = {"H_shift": 0.05, "H_grow": 0.60, "H_full": 0.95, "SHRINK": 0.0}; PM = {"H_shift": 0.10, "H_grow": 0.75, "H_full": 1.0, "SHRINK": 0.0}; PK = {"H_shift": 0.05, "H_grow": 0.45, "H_full": 0.90, "SHRINK": 0.0}


def pri(g):
    return {"H_grow": g, **{k: pr1[k] / (1 - pr1["H_grow"]) * (1 - g) for k in ("H_shift", "H_full", "SHRINK")}}


def oc(g=0.375, pabs=0.75, k24=0.70, lf=0.9, vb=0.9, ret=0.95, keepmem=0.25, corr=0.85, s2=0.8 * 0.85):
    p = pri(g); kept = lf * vb * k24 * ret; s = pabs * kept; dbs = sum(p[h] * PD[h] for h in p)
    both_i = sum(p[h] * (s * PD[h]) ** 2 for h in p); both_c = sum(p[h] * s * (corr * PD[h]) * PD[h] for h in p)
    return {"priors": p, "kept": kept, "SD2_SELF": s, "DBS_given_ABSORB": dbs, "DBS": pabs * dbs, "MG": pabs * sum(p[h] * PM[h] for h in p), "KB": pabs * sum(p[h] * PK[h] for h in p),
            "SD2_C2": s * dbs, "C2_given": {h: s * PD[h] for h in p}, "C2_MEM": s * dbs * keepmem, "both_indep": both_i, "both_corr": both_c,
            "both_x_s2": [both_i * s2, both_c * s2]}
base = oc()
sens = {"pabs_0.60": oc(pabs=0.60), "pabs_0.50": oc(pabs=0.50), "k24_0.55": oc(k24=0.55), "k24_0.58_binomial_at_0.990": oc(k24=0.583),
        "pabs_0.60_k24_0.55": oc(pabs=0.60, k24=0.55), "g_0.25": oc(g=0.25), "g_0.50": oc(g=0.50)}
# K'24-STABLE vs ABSORB-2 dependence: P(K24 | ABSORB) = rho-shifted; if K24 and ABSORB are negatively associated, SD2_SELF = P(A) P(K24 | A) ...
dep = {}
for k24_given_A in (0.70, 0.60, 0.50):
    dep[str(k24_given_A)] = oc(k24=k24_given_A)["SD2_C2"]
# time-model refit at the G-TIME chunk shapes
p5 = json.load(open(os.path.join(HERE, "p5_mem_time.json"))); tp, ts1 = p5["time"]["t_plain_forced"], p5["time"]["t_self1_forced"]
S12, S16, S20 = (sum(t + 1 for t in range(1, K + 1)) for K in (12, 16, 20))
T1c, T2c = p5["time"]["gtime_chunk_maxT_mean_r1_32"], p5["time"]["gtime_chunk_maxT_mean_r2_16"]
det = (12 * 33) * (T1c ** 2 * S16) - (33 ** 2 * S12) * (16 * T1c)
a_ = (tp * T1c ** 2 * S16 - 33 ** 2 * S12 * ts1) / det; b_ = ((12 * 33) * ts1 - (16 * T1c) * tp) / det
tc = a_ * 20 * T2c + b_ * T2c ** 2 * S20; fh = (0.5 * tc + 0.5 * tp) * 12000 / 3600 + 0.4
refit = {"T_r1_self_point": T1c, "T_r2_pred": T2c, "a": a_, "b": b_, "t_self_compute": tc, "forced_h": fh, "cost": [fh * 0.17, fh * 0.26]}
out = {"exposure": expo, "oc_base": base, "oc_sens": sens, "k24_dependence_SD2_C2": dep, "time_refit_gtime_shapes": refit}
json.dump(out, open(os.path.join(HERE, "p6_oc_exposure.json"), "w", newline="\n"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
