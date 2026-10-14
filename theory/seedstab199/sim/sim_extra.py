import math, numpy as np
from math import erf, sqrt
from statistics import NormalDist
nd = NormalDist(); Phi = nd.cdf; Pinv = nd.inv_cdf
rng = np.random.default_rng(7)
def run_pitch(b_fn, a_fn, sigma, T, dt, n=20000):
    m = np.zeros(n); t = 0.0
    for k in range(int(T / dt)):
        m += (b_fn(t) + a_fn(t) * m - m**3) * dt + sigma * sqrt(dt) * rng.standard_normal(n); t += dt
    return m
print("S5' STEER: P = Phi( (m_eff + b/a) / (sigma sqrt(1/(2(k-a)) + 1/(2a))) ), m_eff = (b + k m*)/(k - a)")
a, s, b, toff = 1.0, 0.1, -0.01, 10.0
for (k, ms) in [(2.0, 0.1), (2.0, 0.2), (5.0, 0.1), (5.0, 0.3)]:
    m = run_pitch(lambda t: b + (k * ms if t < toff else 0.0), lambda t: a - (k if t < toff else 0.0), s, toff + 15, 0.01)
    me = (b + k * ms) / (k - a); f = Phi((me + b / a) / (s * sqrt(1 / (2 * (k - a)) + 1 / (2 * a))))
    print("   k %.1f m* %.2f  MC %.3f  formula %.3f" % (k, ms, (m > 0).mean(), f))
print("Entry-time spread: SD of ln Z for Z ~ N(z0, 1) | Z > 0")
for z0 in [0.0, 0.25, 0.43]:
    Z = z0 + rng.standard_normal(2_000_000); Z = Z[Z > 0]; print("   z0 %.2f  SD(ln Z) %.2f   P(ln Z < E ln Z - 3) = %.4f" % (z0, np.log(Z).std(), (np.log(Z) < np.log(Z).mean() - 3).mean()))
ent = np.array([16.2, 17.5, 18.3, 18.3, 19.2, 21.0, 23.5, 25.1, 26.6, 41.7])
print("entry (k steps) of the 10 successes: mean %.1f SD %.1f; without 41.7k: mean %.1f SD %.1f" % (ent.mean(), ent.std(ddof=1), ent[:-1].mean(), ent[:-1].std(ddof=1)))
print("s3 conditional per-step success -> probit and log-probit")
s = [0.97, 0.92, 0.90, 0.86, 0.73, 0.72]; pr = [Pinv(x) for x in s]; t = np.arange(3, 9)
print("   probit", np.round(pr, 2), " log probit", np.round(np.log(pr), 2))
sl = np.polyfit(t, np.log(pr), 1)[0]; print("   slope of log-probit per pass %.3f" % sl)
print("z needed: base P -> target P; factor in z; batch/window factor (z ~ sqrt) ")
for p0 in [0.6, 2/3]:
    for p1 in [0.9, 0.95, 0.99]:
        f = Pinv(p1) / Pinv(p0); print("   P %.2f -> %.2f : z x %.1f ; sqrt-levers (batch, window) x %.0f ; Grokfast lambda for that (z x sqrt(1+lam)): %.0f" % (p0, p1, f, f * f, f * f - 1))
print("P(all k seeds succeed)")
for p in [0.6, 2/3, 0.9, 0.95]:
    print("   p %.2f : 2/2 %.2f  3/3 %.2f  6/6 %.2f" % (p, p**2, p**3, p**6))
print("Restart cost E = [c + (p ts + (1-p) fp)(C - c)] / (p ts), C = 1 (48k), c = 8k/48k")
c = 8 / 48
for p in [0.6, 2/3]:
    for ts, fp in [(1, 1), (1, 0), (0.9, 0.1), (0.8, 0.2)]:
        E = (c + (p * ts + (1 - p) * fp) * (1 - c)) / (p * ts); print("   p %.2f TPR %.1f FPR %.1f  E[cost] = %.2f C   P(success per accepted run) = %.2f" % (p, ts, fp, E, p * ts / (p * ts + (1 - p) * fp)))
for p in [0.9, 0.95]:
    print("   method with p %.2f, no detector: %.2f C" % (p, 1 / p))
# within-family 8k ranks
F = {"den": 1.785, "rms3": 1.840, "nofade": 1.858, "hold": 1.987}; S = {"win1": 1.848, "rms1": 1.907, "rms2": 2.027, "rn1": 2.027, "win2": 2.052, "rn2": 2.122}
import itertools
vals = sorted([(v, "F") for v in F.values()] + [(v, "S") for v in S.values()]); ranksF = [i + 1 for i, (v, l) in enumerate(vals) if l == "F"]
obs = sum(ranksF); cnt = sum(1 for c_ in itertools.combinations(range(1, 11), 4) if sum(c_) <= obs)
print("answer CE @8k, 10 non-reweighted models: failure ranks", ranksF, "rank-sum", obs, "one-sided exact p = %d/210 = %.3f" % (cnt, cnt / 210))
FA = {"den": 0.327, "rms3": 0.349, "nofade": 0.355}; SA = {"win1": 0.361, "rms1": 0.379, "rms2": 0.416, "rn1": 0.423, "win2": 0.416, "rn2": 0.435}
print("aux CE @8k (hold excluded, its aux targets differ): all 3 failures below all 6 successes; exact p = 1/84 = %.3f" % (1 / 84))
