"""GATE rule v2 OC (exact binomial): pooled post-filter value precision over the round's window >= 0.995 (k_pool <= floor(0.005 N_pool))
AND per-n precision >= 0.99 (k_n <= floor(0.01 N_n)). N_n = kept gate items per n (2,000 drawn x keep)."""
import math, json, sys
def cdf(k, n, p):
    if p <= 0: return 1.0
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1)))
keep = {"R1 (n 9-12)": [1.000, 0.996, 0.988, 0.976], "R2 (n 13-16)": [0.972, 0.952, 0.943, 0.917], "R3 (n 17-20)": [0.892, 0.839, 0.809, 0.781]}
out = {}
for name, ks in keep.items():
    Nn = [int(2000 * k) for k in ks]; Np = sum(Nn); r = {}
    for pi in (0.9995, 0.999, 0.997, 0.995, 0.993, 0.99, 0.98):
        q = 1 - pi
        p_pool = cdf(int(0.005 * Np), Np, q); p_n = 1.0
        for N_ in Nn:
            p_n *= cdf(int(0.01 * N_), N_, q)
        r[str(pi)] = {"P_pool": round(p_pool, 4), "P_all_n": round(p_n, 4), "P_gate_approx_indep": round(p_pool * p_n, 4)}
    out[name] = {"N_kept_per_n": Nn, "N_pool": Np, "oc": r}; print(name, Nn, json.dumps(r))
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
