"""PROOFS_SELFDEPTH check 1 ($0, CPU, pure Python): the EXACT joint OC of GATE-q's value terms
   pass  <=>  K_pool = sum_n K_n <= floor(0.005 N_pool)  AND  K_n <= floor(0.01 N_n) for every n,
with K_n ~ Bin(N_n, 1 - pi) independent, N_n = round(2000 * keep_n) (conditioning on the kept counts, as gate_oc2.py).
gate_oc2.py reports P_pool * P_all_n ("approx indep"); both events are decreasing in (K_n), so by Harris-FKG the product is a LOWER bound
and min(P_pool, P_all_n) an upper bound. Here: the exact value by convolution DP, plus integer-exact thresholds (Fraction, no float floor).
Also: Clopper-Pearson bounds for phi and for the measured post-filter precision per round window (213-35 / 213-38 raw counts)."""
import json, math, sys
from fractions import Fraction as Fr


def pmf(N, q):
    if q <= 0:
        return [1.0] + [0.0] * N
    lq, l1 = math.log(q), math.log1p(-q)
    return [math.exp(math.lgamma(N + 1) - math.lgamma(i + 1) - math.lgamma(N - i + 1) + i * lq + (N - i) * l1) for i in range(N + 1)]


def floor_frac(num, den, N):            # floor(num/den * N) exactly
    return (num * N) // den


def gate_exact(Nn, pi, pool=(5, 1000), per=(1, 100)):
    q = 1.0 - pi; Np = sum(Nn); cp = floor_frac(*pool, Np); cn = [floor_frac(*per, N_) for N_ in Nn]
    dist = [1.0]                                                   # distribution of the running sum, truncated at cp
    for N_, c in zip(Nn, cn):
        p = pmf(N_, q)[:c + 1]; new = [0.0] * (min(len(dist) - 1 + c, cp) + 1)
        for s, ps in enumerate(dist):
            if ps == 0.0:
                continue
            for k, pk in enumerate(p):
                if s + k > cp:
                    break
                new[s + k] += ps * pk
        dist = new
    exact = sum(dist)
    p_pool = sum(pmf(Np, q)[:cp + 1]); p_n = 1.0
    for N_, c in zip(Nn, cn):
        p_n *= sum(pmf(N_, q)[:c + 1])
    return {"P_exact": exact, "P_pool": p_pool, "P_all_n": p_n, "fkg_lower": p_pool * p_n, "upper_min": min(p_pool, p_n), "c_pool": cp, "c_n": cn}


def beta_ppf_upper(k, n, conf):
    """one-sided Clopper-Pearson upper bound on a binomial rate: the p with P(Bin(n, p) <= k) = 1 - conf (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(pmf(n, mid)[:k + 1]) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


out = {"gate": {}, "measured": {}}
keep = {"R1 (n 9-12)": [1.000, 0.996, 0.988, 0.976], "R2 (n 13-16)": [0.972, 0.952, 0.943, 0.917], "R3 (n 17-20)": [0.892, 0.839, 0.809, 0.781]}
for name, ks in keep.items():
    Nn = [int(round(2000 * k)) for k in ks]; r = {}
    for pi in (0.9995, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993, 0.99):
        g = gate_exact(Nn, pi); r[str(pi)] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in g.items()}
    out["gate"][name] = {"N_kept_per_n": Nn, "oc": r}
    print(name, Nn, "c_pool", r["0.997"]["c_pool"], "c_n", r["0.997"]["c_n"])
    for pi in ("0.999", "0.997", "0.995", "0.993", "0.99"):
        print("   pi %s exact %.4f  (pool %.4f, all-n %.4f, FKG lower %.4f)" % (pi, r[pi]["P_exact"], r[pi]["P_pool"], r[pi]["P_all_n"], r[pi]["fkg_lower"]))
# rounding check: gate_oc2 used int(2000*k) and int(0.005*Np) in floats
for name, ks in keep.items():
    a = [int(2000 * k) for k in ks]; b = [int(round(2000 * k)) for k in ks]
    print("N_n float-trunc vs round", name, a, b, "pool thr float", int(0.005 * sum(a)), "exact", (5 * sum(a)) // 1000)

# measured post-filter counts (raw files: 213-35/36 eval_chainlabel.json, 213-38 eval_chainlabel_fresh.json; kept = steps_minus_n '0')
D3T = {"test": {9: (1000, 0), 10: (996, 1), 11: (986, 0), 12: (977, 0), 13: (970, 0), 14: (958, 0), 15: (932, 0), 16: (898, 1)},
       "fresh": {9: (1000, 0), 10: (996, 0), 11: (988, 0), 12: (976, 0), 13: (972, 0), 14: (952, 0), 15: (943, 3), 16: (917, 0),
                 17: (892, 0), 18: (839, 0), 19: (809, 1), 20: (781, 0)}}
strict_t = {9: 1.000, 10: 0.995, 11: 0.986, 12: 0.977, 13: 0.970, 14: 0.958, 15: 0.932, 16: 0.897}
strict_f = {9: 1.0, 10: 0.996, 11: 0.988, 12: 0.976, 13: 0.972, 14: 0.952, 15: 0.94, 16: 0.917, 17: 0.892, 18: 0.839, 19: 0.808, 20: 0.781}
nonstrict = {"test": {n: round(1000 * (1 - s)) for n, s in strict_t.items()}, "fresh": {n: round(1000 * (1 - s)) for n, s in strict_f.items()}}
for w, ns in (("R1 9-12", range(9, 13)), ("R2 13-16", range(13, 17)), ("R3 17-20", range(17, 21))):
    kept = sum(D3T[s][n][0] for s in D3T for n in ns if n in D3T[s]); wr = sum(D3T[s][n][1] for s in D3T for n in ns if n in D3T[s])
    ub = beta_ppf_upper(wr, kept, 0.95)
    out["measured"][w] = {"kept": kept, "kept_wrong": wr, "pi_hat": 1 - wr / kept, "pi_lower95": 1 - ub}
    print("window", w, out["measured"][w])
per_n_min = min((1 - D3T[s][n][1] / D3T[s][n][0], s, n) for s in D3T for n in D3T[s])
print("min per-n post-filter precision", per_n_min)
ns_all = sum(nonstrict["test"].values()) + sum(nonstrict["fresh"].values()); kw_all = sum(v[1] for s in D3T for v in D3T[s].values())
phi = kw_all / ns_all; phi_u95 = beta_ppf_upper(kw_all, ns_all, 0.95); phi_u975 = beta_ppf_upper(kw_all, ns_all, 0.975)
print("phi pooled", kw_all, "/", ns_all, "= %.5f; CP one-sided 95%% upper %.5f; 97.5%% %.5f" % (phi, phi_u95, phi_u975))
# phi by stratum and the n 15 fresh cluster
strata = [(s, n, nonstrict[s][n], D3T[s][n][1]) for s in D3T for n in D3T[s]]
p15 = 1 - sum(pmf(60, phi)[:3]); n15_phi_u = beta_ppf_upper(3, 128, 0.95)
# multiplicity: P(some stratum has >= 3 kept-wrong | homogeneous phi)
p_none = 1.0
for s, n, ns_, kw in strata:
    p_none *= sum(pmf(ns_, phi)[:3])
print("P(>=3 kept-wrong in fresh n15 | phi pooled, 60 non-strict) = %.4f; any of 20 strata >= 3: %.4f; phi_15 (test+fresh 3/128) 95%% upper %.4f" % (p15, 1 - p_none, n15_phi_u))
# rate per kept item: homogeneous Poisson check on kept-wrong per kept
kept_all = sum(v[0] for s in D3T for v in D3T[s].values()); lam = kw_all / kept_all
p15k = 1 - sum(pmf(943, lam)[:3])
p_none_k = 1.0
for s in D3T:
    for n, (k_, w_) in D3T[s].items():
        p_none_k *= sum(pmf(k_, lam)[:3])
print("kept-wrong per kept %.2e (%d/%d); P(>=3 at fresh n15) %.4f; any stratum %.4f" % (lam, kw_all, kept_all, p15k, 1 - p_none_k))
out["phi"] = {"kept_wrong": kw_all, "non_strict": ns_all, "phi": phi, "cp_upper95": phi_u95, "cp_upper975": phi_u975,
              "n15_fresh_cluster_p": p15, "any_stratum_ge3_p": 1 - p_none, "phi_n15_upper95": n15_phi_u,
              "rate_per_kept": lam, "kept_total": kept_all, "n15_cluster_p_perkept": p15k, "any_stratum_ge3_p_perkept": 1 - p_none_k,
              "strata": [{"set": s, "n": n, "non_strict": a, "kept_wrong": b} for s, n, a, b in strata]}
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
