"""PROOFS_SD2 check 3 + 4 ($0, CPU; exact binomial sums in log space, independent of sd2_numbers).
(3) Label noise at W_2 = n 13..16 from eval_chainlabel_fresh (FT-SELF, D3-T): kept-wrong counts, pi, Clopper-Pearson one- and two-sided 95 % bounds,
    phi-hat = kept-wrong / non-strict, kappa thresholds in both PROOFS §3b forms: phi form P/(phi n) and the direct-w form ebar/w >= (1 - P)/(n w);
    the pooled round-2 pool (n 9..16) w; two-round factor (kappa 1) with per-round phi.
(4) GATE-2 / GATE-1' exact OC (value terms: per-n c_n = floor(0.01 N_n), pooled c = floor(0.005 N)): DP over per-n kept-wrong counts; the pi^E pooled
    term (<= floor(0.01 N) state-wrong); sensitivity to kept counts below the KEEP-only projection (CV < KEEP); LABEL-AUDIT-2 per-window OC; RETAIN."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"


def logpmf(k, n, p):
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def cdf(c, n, p):
    return sum(math.exp(logpmf(k, n, p)) for k in range(0, c + 1))


def cp_upper(k, n, a):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(k, n, mid) > a else (lo, mid)
    return hi


fr = {"FTSELF": json.load(open(R + "s215_ftself_s1/eval_chainlabel_fresh.json"))["by_n"], "D3T": json.load(open(R + "s213_d3t_s1/eval_chainlabel_fresh.json"))["by_n"]}
noise = {}
for tag, b in fr.items():
    kw = {n: round(b[str(n)]["n_kept"] * (1 - b[str(n)]["prec_kept"])) for n in range(9, 21)}
    kept = {n: b[str(n)]["n_kept"] for n in range(9, 21)}; nonstrict = {n: round(b[str(n)]["items"] * (1 - b[str(n)]["precision_strict"])) for n in range(9, 21)}
    W2 = range(13, 17); K = sum(kw[n] for n in W2); NK = sum(kept[n] for n in W2); NS = sum(nonstrict[n] for n in W2)
    P16 = b["16"]["precision_strict"]
    w = K / NK; w1s = cp_upper(K, NK, 0.05); w2s = cp_upper(K, NK, 0.025); phi = K / NS; phi_up = cp_upper(K, NS, 0.05)
    Kp = sum(kw[n] for n in range(9, 17)); NKp = sum(kept[n] for n in range(9, 17)); wp = Kp / NKp; wp_up = cp_upper(Kp, NKp, 0.05)
    eb16 = (1 - P16) / 16
    noise[tag] = {"kept_wrong": kw, "kept": kept, "nonstrict": nonstrict, "W2": {"kept_wrong": K, "kept": NK, "pi": 1 - w, "pi_lo95_one_sided": 1 - w1s,
                  "pi_lo95_two_sided": 1 - w2s, "phi": phi, "phi_up95": phi_up, "nonstrict": NS},
                  "kappa_phi_form": {"point": P16 / (phi * 16) if phi else None, "phi_up": P16 / (phi_up * 16)},
                  "kappa_w_form_W2": {"point": eb16 / w if w else None, "w_up": eb16 / w1s},
                  "pool_n9_16": {"kept_wrong": Kp, "kept": NKp, "w": wp, "w_up95": wp_up, "kappa_w_form_point": eb16 / wp if wp else None, "kappa_w_form_up": eb16 / wp_up},
                  "P16": P16, "ebar16_lower": eb16, "bias_class_mass_bound_point": 2 * w, "bias_class_mass_bound_up": 2 * w1s,
                  "kept_wrong_n17_20": sum(kw[n] for n in range(17, 21)), "kept_n17_20": sum(kept[n] for n in range(17, 21))}
# round-1 phi (PROOFS: D3-T 6 / 1,224 = 0.0049, CP up 0.00965); two-round factor kappa 1
phi1, phi1_up = 6 / 1224, cp_upper(6, 1224, 0.05)
ph2 = noise["FTSELF"]["W2"]["phi"]; ph2u = noise["FTSELF"]["W2"]["phi_up95"]; P12, P16 = 0.976, noise["FTSELF"]["P16"]
f = lambda p1, p2: (1 + p1 * 12 / P12) * (1 + p2 * 16 / P16)  # noqa: E731
two = {"phi_0.010_both": f(0.010, 0.010), "point_0.0049_then_FTSELF": f(phi1, ph2), "up_0.00965_then_FTSELF_up": f(phi1_up, ph2u), "phi1_up": phi1_up,
       "w_form_point": (1 + 12 * (2 / 5000) / (1 - P12)) * (1 + 16 * noise["FTSELF"]["W2"]["kept_wrong"] / noise["FTSELF"]["W2"]["kept"] / (1 - P16)),
       "note": "w-form round 1 uses LABEL-AUDIT's measured 2/5,000 kept-wrong share and 1 - P(12) of D3-T; round 2 FT-SELF's W2 share and 1 - P(16)"}


# (4) GATE OC
def gate_oc(Ns, pi):
    q = 1 - pi; cn = [int(math.floor(0.01 * N + 1e-9)) for N in Ns]; cp = int(math.floor(0.005 * sum(Ns) + 1e-9))
    dist = {0: 1.0}
    for N, c in zip(Ns, cn):
        pm = [math.exp(logpmf(k, N, q)) for k in range(c + 1)]; nd = {}
        for s, ps in dist.items():
            for k, pk in enumerate(pm):
                if s + k <= cp:
                    nd[s + k] = nd.get(s + k, 0.0) + ps * pk
        dist = nd
    return sum(dist.values()), cn, cp


keepF = {n: fr["FTSELF"][str(n)]["keep"] for n in range(9, 17)}
N2 = [int(round(2000 * keepF[n])) for n in range(13, 17)]; N1p = [int(round(2000 * keepF[n])) for n in range(9, 13)]
oc = {}
for name, Ns in (("GATE2", N2), ("GATE1p", N1p), ("GATE2_kept_1900_each", [1900] * 4), ("GATE2_kept_1850_each", [1850] * 4), ("GATE2_cv_0.98xKEEP", [int(0.98 * x) for x in N2])):
    o = {}
    for pi in (0.9995, 0.999, 0.997, 0.996, 0.995, 0.994, 0.993, 0.99):
        p, cn, cp = gate_oc(Ns, pi); o[str(pi)] = p
    oc[name] = {"N": Ns, "c_n": cn, "c_pool": cp, "P_pass": o}
NK2 = sum(N2); cE = int(math.floor(0.01 * NK2 + 1e-9))
piE = {str(p): cdf(cE, NK2, 1 - p) for p in (0.995, 0.993, 0.99, 0.989, 0.988, 0.985)}
keep_term = {"P_keep_ge_0.70_at_keep_0.957": min(1.0, sum(math.exp(logpmf(k, 2000, 0.957)) for k in range(1400, 2001))),
             "P_keep_ge_0.70_at_keep_0.72": sum(math.exp(logpmf(k, 2000, 0.72)) for k in range(1400, 2001))}
# LABEL-AUDIT-2: 5,000 consumed records, pooled <= 50 wrong (pi >= 0.99); per window ~2,500 records (pool order 50/50), <= 25 wrong
la = {"pooled_5000": {str(p): cdf(50, 5000, 1 - p) for p in (0.9995, 0.995, 0.99, 0.985)}, "per_window_2500": {str(p): cdf(25, 2500, 1 - p) for p in (0.9995, 0.995, 0.99, 0.985)},
      "pooled_P_wrong_le_50_at_0.985": cdf(50, 5000, 0.015)}
ret = {"one_n_p_err_0.001": cdf(10, 1000, 0.001), "all4_p_err_0.001": cdf(10, 1000, 0.001) ** 4, "one_n_p_err_0.02": cdf(10, 1000, 0.02),
       "K24_n1_true_0.990": cdf(10, 1000, 0.01), "K24_n1_true_0.995": cdf(10, 1000, 0.005), "K24_n1_true_0.985": cdf(10, 1000, 0.015)}
out = {"noise": noise, "two_round_kappa1": two, "gate_oc": oc, "piE_term_N%d_c%d" % (NK2, cE): piE, "keep_term": keep_term, "label_audit2": la, "retain_k24": ret}
json.dump(out, open(os.path.join(HERE, "p3_noise_gate.json"), "w", newline="\n"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
