# Operating characteristics of binary seed-success designs (exact binomial; no GPU).
import math, sys
sys.stdout.reconfigure(encoding="utf-8")
from math import comb
def pmf(k, n, p): return comb(n, k) * p**k * (1 - p)**(n - k)
def tail(k, n, p): return sum(pmf(j, n, p) for j in range(k, n + 1))   # P(X >= k)
def cdf(k, n, p): return sum(pmf(j, n, p) for j in range(0, k + 1))     # P(X <= k)

print("== single-stage exact: smallest k with P(X>=k | p0) <= alpha; power at p1 ==")
for p0 in (0.6, 0.667):
    for a in (0.05, 0.10):
        rows = []
        for n in range(3, 21):
            ks = [k for k in range(n + 1) if tail(k, n, p0) <= a]
            if not ks: continue
            k = ks[0]
            rows.append((n, k, round(tail(k, n, p0), 4), round(tail(k, n, 0.9), 3), round(tail(k, n, 0.95), 3)))
        print("p0=%.3f alpha=%.2f" % (p0, a))
        for r in rows[:12]: print("   n=%d reject if >=%d: size %.4f, power(.9) %.3f, power(.95) %.3f" % r)

print("\n== Clopper-Pearson one-sided 95% lower bound after n/n: 0.05**(1/n) ==")
for n in (3, 5, 6, 8, 10, 14, 20, 29):
    print("  %d/%d -> LB %.3f" % (n, n, 0.05 ** (1 / n)))

print("\n== Simon two-stage (futility stop after stage 1): stop if X1 <= r1 of n1; else reject H0 if X > r of n ==")
def simon(p0, p1, a, b, nmax=25):
    best_opt, best_mm = None, None
    for n in range(2, nmax + 1):
        for n1 in range(1, n):
            for r1 in range(-1, n1):
                pet0 = cdf(r1, n1, p0); pet1 = cdf(r1, n1, p1)
                for r in range(max(r1, 0), n):
                    def rej(p):
                        s = 0.0
                        for x1 in range(r1 + 1, n1 + 1):
                            need = r + 1 - x1
                            s += pmf(x1, n1, p) * (tail(need, n - n1, p) if need > 0 else 1.0)
                        return s
                    al = rej(p0)
                    if al > a: continue
                    pw = rej(p1)
                    if pw < 1 - b: break
                    en0 = n1 + (1 - pet0) * (n - n1)
                    rec = (round(en0, 3), n1, r1, n, r, round(al, 4), round(pw, 3), round(pet0, 3), round(n1 + (1 - pet1) * (n - n1), 2))
                    if best_opt is None or rec[0] < best_opt[0]: best_opt = rec
                    if best_mm is None or (n, rec[0]) < (best_mm[3], best_mm[0]): best_mm = rec
                    break
    return best_opt, best_mm
for (p0, p1, a, b) in ((0.6, 0.9, 0.05, 0.2), (0.667, 0.9, 0.05, 0.2), (0.6, 0.9, 0.10, 0.2), (0.6, 0.95, 0.05, 0.2)):
    o, m = simon(p0, p1, a, b)
    print("p0 %.3f p1 %.2f alpha %.2f power %.1f" % (p0, p1, a, 1 - b))
    for lab, rec in (("optimal", o), ("minimax", m)):
        print("   %s: n1=%d r1=%d n=%d r=%d | size %.4f power %.3f | P(stop early|p0) %.3f | E[N|p0] %.2f E[N|p1] %.2f" % (lab, rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[0], rec[8]))

print("\n== Bayesian reading after s/n (uniform Beta(1,1) prior; Jeffreys in brackets) ==")
def beta_tail(x, a, b, N=200000):
    # P(p >= x) for Beta(a,b) by midpoint integration
    h = (1 - x) / N; lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return sum(math.exp((a - 1) * math.log(x + (i + .5) * h) + (b - 1) * math.log(1 - x - (i + .5) * h) - lbeta) for i in range(N)) * h
for s, n in ((3, 3), (5, 6), (6, 6), (8, 8), (12, 14)):
    print("  %d/%d: P(p>0.6) %.3f [%.3f], P(p>=0.9) %.3f [%.3f]" % (s, n, beta_tail(.6, s + 1, n - s + 1), beta_tail(.6, s + .5, n - s + .5), beta_tail(.9, s + 1, n - s + 1), beta_tail(.9, s + .5, n - s + .5)))

print("\n== two-arm concurrent control, one-sided Fisher exact (alpha .05), power at 0.6 vs 0.9 ==")
def fisher_p(x1, n1, x2, n2):   # P(X2 >= x2 | margins), X2 = arm 2 successes
    t = x1 + x2; N = n1 + n2
    return sum(comb(n2, j) * comb(n1, t - j) for j in range(x2, min(n2, t) + 1) if 0 <= t - j <= n1) / comb(N, t)
for n in (6, 10, 15, 20, 25, 30):
    pw = sum(pmf(x1, n, .6) * pmf(x2, n, .9) for x1 in range(n + 1) for x2 in range(n + 1) if fisher_p(x1, n, x2, n) <= .05)
    print("  n=%d per arm: power %.3f (%d runs)" % (n, pw, 2 * n))

print("\n== single-seed 'rescue' of a known-failing seed: P(rescue | no effect) = p*(1-rho) ==")
for rho in (0, .25, .5, .75):
    print("  rho %.2f: FPR %.2f (p 0.6)" % (rho, .6 * (1 - rho)))
