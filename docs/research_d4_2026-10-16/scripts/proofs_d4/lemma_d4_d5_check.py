"""PROOFS_D4 check 5: Lemma D4 (drift recursion bounds) and Lemma D5 (TBPTT bias), numerics in fp64.
D4: delta_t = A_t delta_{t-1} + eps_t, ||A_t|| <= rho, ||eps_t|| <= eps (adversarial and random draws); bound rho^t ||d0|| + eps (1 - rho^t)/(1 - rho).
    (a) the uniform-in-K bound is max(||d0||, eps/(1-rho)), = eps/(1-rho) only if ||d0|| <= eps/(1-rho).
    (b) rho = 1: worst case ||d0|| + t eps; RMS sqrt(t D) sigma for zero-mean independent eps and orthogonal A_t (<= for ||A_t|| <= 1).
    (c) anchor model x_t <= (1 - alpha_t) x_{t-1} + alpha_t delta_a + eps, alpha_t >= alpha0: sup <= max(x_0, delta_a + eps/alpha0); tight.
D5: J_{t,s} for losses t = 1..K, parameter uses s = 1..t (lag l = t - s); ||J_{t,s}|| <= G rho_b^l.
    sliding-window TBPTT-k drops l >= k; chunked TBPTT-k (detach every k passes) drops pairs in different chunks (incl. lag 1 at a boundary).
    Exact worst-case bias sums vs D4's K G rho^k / (1 - rho) and G (K - k)(K - k + 1)/2."""
import json, sys, math
import numpy as np
rng = np.random.default_rng(0); out = {}
# ---- D4
def run_rec(rho, eps, d0n, K, D=16, adversarial=True):
    d = np.zeros(D); d[0] = d0n; xs = [np.linalg.norm(d)]
    for t in range(1, K + 1):
        Q, _ = np.linalg.qr(rng.standard_normal((D, D))); A = rho * Q
        d = A @ d
        e = d / (np.linalg.norm(d) + 1e-300) * eps if adversarial else (lambda z: z / np.linalg.norm(z) * eps)(rng.standard_normal(D))
        d = d + e; xs.append(np.linalg.norm(d))
    return xs
viol = 0; worst_ratio = 0.0
for rho in (0.5, 0.9, 0.99, 1.0):
    for eps in (0.01, 0.1):
        for d0n in (0.0, 0.5, 5.0):
            for adv in (True, False):
                xs = run_rec(rho, eps, d0n, 24, adversarial=adv)
                for t, x in enumerate(xs):
                    bnd = rho ** t * d0n + (eps * t if rho == 1.0 else eps * (1 - rho ** t) / (1 - rho))
                    viol += x > bnd * (1 + 1e-9) + 1e-12; worst_ratio = max(worst_ratio, x / bnd if bnd > 0 else 0)
out["D4_bound_violations"] = int(viol); out["D4_bound_max_ratio_achieved"] = worst_ratio   # adversarial alignment attains the bound (ratio 1)
# (a) counterexample to 'bounded by eps/(1-rho)' when d0 is large
xs = run_rec(0.9, 0.01, 5.0, 12); out["D4a_counterexample"] = {"rho": 0.9, "eps": 0.01, "d0": 5.0, "eps_over_1mrho": 0.1, "delta_at_t1_to_t3": xs[1:4]}
# (b) RMS check, rho = 1, orthogonal A, iid N(0, sigma^2 I)
D, sig, K, M = 32, 0.01, 24, 4000; acc = np.zeros(K + 1)
for _ in range(M):
    d = np.zeros(D)
    for t in range(1, K + 1):
        Q, _ = np.linalg.qr(rng.standard_normal((D, D))); d = Q @ d + sig * rng.standard_normal(D); acc[t] += d @ d
rms = np.sqrt(acc / M); out["D4b_rms_over_sqrt_tD_sigma"] = [float(rms[t] / (math.sqrt(t * D) * sig)) for t in (1, 6, 12, 24)]
# (c) anchor: worst case x_t = (1 - a0) x + a0 da + eps from x0 = 0 and from x0 above the bound
for a0, da, eps in ((0.2, 0.05, 0.01), (0.05, 0.0, 0.01)):
    for x0 in (0.0, 2.0):
        x = x0; mx = x
        for t in range(2000):
            x = (1 - a0) * x + a0 * da + eps; mx = max(mx, x)
        out.setdefault("D4c", []).append({"alpha0": a0, "delta_a": da, "eps": eps, "x0": x0, "sup": mx, "limit": x, "claimed_sup": da + eps / a0})
# ---- D5
def bias_sum(K, k, rho, kind):
    s = 0.0; npairs = 0
    for t in range(1, K + 1):
        for ss in range(1, t + 1):
            l = t - ss
            drop = (l >= k) if kind == "sliding" else ((t - 1) // k != (ss - 1) // k)
            if drop:
                s += rho ** l; npairs += 1
    return s, npairs
def full_sum(K, rho):
    return sum(rho ** (t - ss) for t in range(1, K + 1) for ss in range(1, t + 1))
rows = []
for K, k in ((20, 12), (24, 12), (16, 12)):
    for rho in (0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
        bs_, ns_ = bias_sum(K, k, rho, "sliding"); bc_, nc_ = bias_sum(K, k, rho, "chunked"); fs = full_sum(K, rho)
        rows.append({"K": K, "k": k, "rho_b": rho, "sliding_bias_bound/G": bs_, "sliding_pairs": ns_, "chunked_bias_bound/G": bc_, "chunked_pairs": nc_,
                     "full_bound/G": fs, "D4_formula KG rho^k/(1-rho) /G": (K * rho ** k / (1 - rho)) if rho < 1 else None,
                     "D4_formula_rho1 (K-k)(K-k+1)/2": (K - k) * (K - k + 1) / 2 if rho == 1.0 else None,
                     "sliding_rel_to_full_bound": bs_ / fs, "chunked_rel_to_full_bound": bc_ / fs, "rho^k": rho ** k})
out["D5"] = rows
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
print("D4 violations", out["D4_bound_violations"], "max ratio", round(worst_ratio, 6)); print("D4a", out["D4a_counterexample"]); print("D4b", out["D4b_rms_over_sqrt_tD_sigma"])
for r in out["D4c"]:
    print("D4c", r)
for r in rows:
    print("D5 K%d k%d rho %.2f | sliding %.3f (%d pairs) chunked %.3f (%d pairs) full %.3f | D4 formula %s | rel sliding %.4f chunked %.4f rho^k %.4f" % (
        r["K"], r["k"], r["rho_b"], r["sliding_bias_bound/G"], r["sliding_pairs"], r["chunked_bias_bound/G"], r["chunked_pairs"], r["full_bound/G"],
        r["D4_formula KG rho^k/(1-rho) /G"] if r["rho_b"] < 1 else r["D4_formula_rho1 (K-k)(K-k+1)/2"], r["sliding_rel_to_full_bound"], r["chunked_rel_to_full_bound"], r["rho^k"]))
