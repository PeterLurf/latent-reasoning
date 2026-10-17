"""SELFDEPTH v1 $0 CPU checks (1 thread; reads explicit JSON paths only; writes only to argv[1]).
 (1) STOP-LOC (fixed, unfiltered G; k 9..15) on base D3-T eval_vb and on the exact H_shift translation (n+4, K'+4).
 (2) K'-BEYOND (v1 def) and RISE-PAST-K on the base analogue and under exact H_shift; in-range saturation at K' = n.
 (3) Mechanism partition (exhaustive) on H_shift / H_grow / H_full / SHRINK panels.
 (4) GATE-1 pi^E term: exact binomial OC; joint with the value terms needs gate_oc_exact (not recomputed).
 (5) Pool sizing under v1 (no stage A; uniform n 9-12; one interleaved pool, cursor 64 per self step): exhaustion and per-n use.
 (6) OC arithmetic: draft priors, RT-shaded priors, shaded + P(ABSORB) 0.45 sensitivity."""
import json, math, os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ev = json.load(open("D:/Latent/boxes/runs_fsq/s213_d3t_s1/eval_vb.json"))
K, A2 = ev["acc_k"], ev["acc_n2"]
out = {}


def acc_base(k, n):                      # base acc at K' k (k n+2 -> acc_n2), n 1..16
    if k == n + 2:
        return A2[str(n)]
    return K[str(k)][str(n)]


def acc_shift(k, n):                     # exact H_shift: FT(K' k, n) = base(K' k-4, n-4)
    return acc_base(k - 4, n - 4) if (n + 2 != k) else A2[str(n - 4)]


def G(acc, k, B_top=16):
    ns = [n for n in range(k + 1, min(k + 4, B_top) + 1)]
    return sum(acc(n + 2, n) - acc(k, n) for n in ns) / len(ns), ns


def stop_loc(acc, ks):
    gs = {}
    for k in ks:
        g, ns = G(acc, k); gs[k] = round(g, 4)
    stop = next((k for k in ks if gs[k] <= 0.02 + 1e-9), None)
    return {"G": gs, "STOP": stop if stop is not None else ">=16"}


out["stop_loc_base_k_7_15"] = stop_loc(acc_base, range(7, 16))
out["stop_loc_Hshift_k_9_15"] = stop_loc(acc_shift, range(9, 16))
# (2) K'-BEYOND v1 = mean_{n 15,16} acc(K' n+2) - acc(K' 14); base analogue (B 8): mean_{n 11,12} acc_n2 - acc(K' 10)
kb_base = sum(A2[str(n)] - K["10"][str(n)] for n in (11, 12)) / 2
kb_shift = sum(acc_shift(n + 2, n) - acc_shift(14, n) for n in (15, 16)) / 2
rise_base = sum(A2[str(n)] - K["12"][str(n)] for n in (11, 12)) / 2          # base analogue of acc(K' n+2) - acc(K' kmax)
sat = {str(n): {"K'=n-1": K[str(n - 1)][str(n)], "K'=n": K[str(n)][str(n)], "K'=n+2": A2[str(n)], "K'=24": K["24"][str(n)]} for n in range(3, 10)}
out["kbeyond"] = {"base_analogue_n11_12_vs_K10": round(kb_base, 4), "Hshift_n15_16_vs_K14": round(kb_shift, 4),
                  "rise_past_kmax_base_analogue_n11_12_vs_K12": round(rise_base, 4), "in_range_saturation": sat}
# H_full panel (CONJECTURE): acc(K' n+2) 0.85 / 0.80 at n 15 / 16; acc(K' 14) at n 15 / 16 = base one-/two-short in-range levels (n 8 at K' 7 / K' 6)
one_short, two_short = K["7"]["8"], K["6"]["8"]
out["kbeyond"]["Hfull_panel"] = {"acc_n2_15_16": [0.85, 0.80], "acc_k14_15_16": [one_short, two_short],
                                 "KB": round(((0.85 - one_short) + (0.80 - two_short)) / 2, 4)}
# H_grow panel (draft S4 curve 0.95 / 0.82 / 0.60 / 0.38): stop ~ B+3 = 15 -> acc(K' 14) at n 15 / 16 ~ 1/2-short levels of a shifted base x grow
out["kbeyond"]["Hgrow_panel_bounds"] = {"if_acc_k14 = one/two-short levels": round(((0.60 - one_short) + (0.38 - two_short)) / 2, 4),
                                       "if_acc_k14 = 0.5 * acc_n2": round(((0.60 - 0.30) + (0.38 - 0.19)) / 2, 4)}


# (3) mechanism partition
def mech(absorb, M, stop):
    if not absorb:
        return "UNLABELLED (not ABSORB)"
    if M >= 0.80 - 1e-9:
        return "FULL"
    if M >= 0.63 - 1e-9:
        return "GROW"
    if M < 0.40:
        return "SHRINK"
    s = 16 if stop == ">=16" else stop
    return "SHIFT" if s >= 13 else "MIXED"


Msh = sum(acc_shift(n + 2, n) for n in range(13, 17)) / 4
out["mechanism_panels"] = {"Hshift": [round(Msh, 4), out["stop_loc_Hshift_k_9_15"]["STOP"], mech(True, Msh, out["stop_loc_Hshift_k_9_15"]["STOP"])],
                           "Hgrow_0.69": mech(True, 0.69, 15), "Hfull_0.85": mech(True, 0.85, ">=16"), "shrink_0.30": mech(True, 0.30, 12),
                           "mixed_0.50_stop12": mech(True, 0.50, 12), "noabsorb": mech(False, 0.48, 14)}
Ms = [acc_shift(n + 2, n) for n in range(13, 17)]
out["Hshift_curve_13_16"] = [round(v, 4) for v in Ms]
se = math.sqrt(sum(v * (1 - v) / 1000 for v in Ms)) / 4
out["Hshift_mean_SE"] = round(se, 4)


# (4) pi^E term OC: pooled kept N ~ 7,900 (KEEP x CV at D3-T fresh keep n 9-12), allow c = floor(0.01 N) state-wrong
def binom_cdf(c, N, p):
    lp = [math.lgamma(N + 1) - math.lgamma(k + 1) - math.lgamma(N - k + 1) + k * math.log(p) + (N - k) * math.log1p(-p) for k in range(c + 1)]
    m = max(lp); return math.exp(m) * sum(math.exp(v - m) for v in lp)


Nk = int(2000 * (1.000 + 0.996 + 0.988 + 0.976)); c = int(math.floor(0.01 * Nk))
out["piE_OC"] = {"N_pooled_kept": Nk, "c_allowed": c, **{"P(pass|piE=%s)" % q: round(binom_cdf(c, Nk, 1 - q), 4) for q in (0.9997, 0.997, 0.993, 0.99, 0.988, 0.985)}}
# keep term: keep_n >= 0.70 at every n with N = 2,000 per n; at D3-T keep >= 0.976 -> P(fail) ~ 0
out["keep_OC_note"] = "keep_n (KEEP and CV) >= 0.70 per n at N 2,000: at true 0.976 the binomial lower tail at 0.70 is < 1e-100; at 0.70 ~ 0.5"

# (5) pool sizing, v1: self steps S ~ Bin(12000, 0.5); cursor 64 / self step through one interleaved pool of 4 x 110k (fixed permutation)
r = np.random.default_rng(215); sims = 20000
S = r.binomial(12000, 0.5, sims); used = 64 * S
per_n_sd = math.sqrt(384000 * 0.25 * 0.75)
out["pool_v1"] = {"pool_total": 440000, "per_n": 110000, "self_steps_mean": float(S.mean()), "self_steps_max_sim": int(S.max()),
                  "items_used_mean": float(used.mean()), "items_used_max_sim": int(used.max()), "P(exhaust)": float((used > 440000).mean()),
                  "z_exhaust": round((440000 / 64 - 6000) / math.sqrt(3000), 1),
                  "per_n_used_mean": 96000, "per_n_used_sd_approx": round(math.sqrt(per_n_sd ** 2 + (64 * math.sqrt(3000) / 4) ** 2)),
                  "note": "per-n use is the permutation's prefix count; the pool holds 110k per n, so no n can run out before the global cursor"}

# (6) OC arithmetic
kept = 0.9 * 0.9 * 0.85


def oc(pA, h, dbs_grow, mg_shift=0.07, dbs_shift=0.01, dbs_full=0.9, mg_grow=0.85):
    hs, hg, hf = h
    dbs = pA * (hs * dbs_shift + hg * dbs_grow + hf * dbs_full); mg = pA * (hs * mg_shift + hg * mg_grow + hf * 1.0)
    return {"P_ABSORB": pA, "P_SD_SELF": round(pA * kept, 4), "P_MG": round(mg, 4), "P_DBS": round(dbs, 4), "P_SD_C2": round(dbs * kept, 4),
            "P_SD_C2_MEM": round(dbs * kept * 0.7, 4), "P_SD_C2|Hshift": round(dbs_shift * kept, 4), "P_SD_C2|Hfull": round(dbs_full * kept, 4)}


out["OC"] = {"draft": oc(0.65, (0.55, 0.30, 0.10), 0.25), "RT_shaded": oc(0.65, (0.65, 0.25, 0.03), 0.15),
             "shaded_ABSORB_0.45": oc(0.45, (0.65, 0.25, 0.03), 0.15), "kept_factor": kept}
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
