"""MC checks of the selection formulas in MATH_selection.md (numpy only; seeded; ~1-2 min on 3 threads)."""
import math, numpy as np
from math import erf, sqrt, pi, exp, log
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
rng = np.random.default_rng(20261013)
N = 20000

def run_pitch(b_fn, a_fn, sigma, T, dt, c=1.0, m0=0.0, n=N):
    m = np.full(n, m0); t = 0.0
    for k in range(int(T / dt)):
        m += (b_fn(t) + a_fn(t) * m - c * m**3) * dt + sigma * sqrt(dt) * rng.standard_normal(n); t += dt
    return m

def se(p, n=N): return sqrt(p * (1 - p) / n)
print("S1a  constant a: P = Phi(b sqrt(2/a)/sigma)")
for b in [-0.02, 0.0, 0.01, 0.02, 0.05]:
    a, s = 1.0, 0.1
    m = run_pitch(lambda t: b, lambda t: a, s, 20, 0.01); p = (m > 0).mean()
    print("   b %+.3f  MC %.3f +- %.3f   formula %.3f" % (b, p, se(p), Phi(b * sqrt(2 / a) / s)))
print("S1b  neutral window T_w then a: P = Phi(b (T_w + 1/a) / (sigma sqrt(T_w + 1/(2a))))")
for (b, Tw) in [(0.003, 10), (0.003, 40), (0.0015, 40), (-0.002, 10)]:
    a, s = 1.0, 0.03
    m = run_pitch(lambda t: b, lambda t: (0.0 if t < Tw else a), s, Tw + 20, 0.01); p = (m > 0).mean()
    print("   b %+.4f T_w %3d  MC %.3f +- %.3f   formula %.3f   (drifted-BM limit Phi(b sqrt(T_w)/sigma) = %.3f)" % (b, Tw, p, se(p), Phi(b * (Tw + 1/a) / (s * sqrt(Tw + 1/(2*a)))), Phi(b * sqrt(Tw) / s)))
print("S1c  swept a(t) = kappa (t - t0): P = Phi(sqrt(2) (pi/kappa)^(1/4) b / sigma)")
for (kap, b) in [(0.1, 0.005), (0.025, 0.005), (0.4, 0.005)]:
    s = 0.05; t0 = math.sqrt(2 * 12 / kap)
    m = run_pitch(lambda t: b, lambda t: kap * (t - t0), s, t0 + 3.0 / math.sqrt(kap) + 15, 0.01); p = (m > 0).mean()
    print("   kappa %.3f  MC %.3f +- %.3f   formula %.3f" % (kap, p, se(p), Phi(sqrt(2) * (pi / kap) ** 0.25 * b / s)))
print("S1d  race / decaying bias: b(t) = b_P - beta_C c(t), c = 1 - exp(-r_C (t + T_h));")
print("     P = Phi( sqrt(2a)/sigma * [ (b_P - beta_C)/a + beta_C exp(-r_C T_h)/(a + r_C) ] )")
for (bP, bC, rC, Th) in [(0.01, 0.03, 0.5, 0.0), (0.01, 0.03, 0.5, 1.0), (0.01, 0.03, 0.2, 0.0), (0.01, 0.03, 2.0, 0.0), (0.02, 0.03, 0.5, 2.0)]:
    a, s = 1.0, 0.05
    m = run_pitch(lambda t: bP - bC * (1 - math.exp(-rC * (t + Th))), lambda t: a, s, 25, 0.01); p = (m > 0).mean()
    mu = (bP - bC) / a + bC * math.exp(-rC * Th) / (a + rC); sz = s / sqrt(2 * a)
    print("   b_P %.3f beta_C %.3f r_C %.2f T_h %.1f  MC %.3f +- %.3f  formula %.3f" % (bP, bC, rC, Th, p, se(p), Phi(mu / sz)))

print("S2  'hare' effect: per-seed C speed r_C,i = r_C exp(0.5 xi_i); C progress at t_obs predicts FAILURE")
a, s, bP, bC, rbar, Th, tobs = 1.0, 0.05, 0.01, 0.03, 0.5, 0.5, 3.0
rCi = rbar * np.exp(0.5 * rng.standard_normal(N)); m = np.zeros(N); t = 0.0; dt = 0.01; cobs = None
for k in range(int(25 / dt)):
    c = 1 - np.exp(-rCi * (t + Th)); m += (bP - bC * c + a * m - m**3) * dt + s * sqrt(dt) * rng.standard_normal(N); t += dt
    if cobs is None and t >= tobs:
        cobs = c.copy(); mobs = m.copy()
succ = m > 0; q = np.quantile(cobs, [1/3, 2/3])
for lo, hi, name in [(-1, q[0], "C slowest third"), (q[0], q[1], "middle third"), (q[1], 2, "C fastest third (ahead early)")]:
    sel = (cobs > lo) & (cobs <= hi); print("   %-30s P(success) %.3f" % (name, succ[sel].mean()))
loss_obs = 1.0 - 0.6 * cobs                                        # toy early loss: C's visible progress (P is invisible before commit)
r = np.corrcoef(loss_obs, succ.astype(float))[0, 1]; print("   corr(early loss, success) = %+.3f  (positive: lower early loss -> failure)" % r)
print("   sign of m at t_obs predicts outcome: accuracy %.3f" % ((np.sign(mobs) == np.sign(m)).mean()))

print("S3  log-amplitude with multiplicative noise (stochastic collapse): dy = (a - zeta^2/2) dt + zeta dW, success = hit h within tau")
def bl(mu, z, h, tau):   # Bachelier-Levy first passage of mu t + z W to level h > 0 by tau
    return Phi((mu * tau - h) / (z * sqrt(tau))) + math.exp(2 * mu * h / z**2) * Phi((-mu * tau - h) / (z * sqrt(tau)))
a, h, tau, z1 = 0.5, 4.0, 10.0, 1.6
for B in [1, 2, 3, 4, 8, 16]:
    zeta = z1 / sqrt(B); mu = a - zeta**2 / 2; dt = 0.01; y = np.zeros(N); hit = np.zeros(N, bool)
    for k in range(int(tau / dt)):
        y += mu * dt + zeta * sqrt(dt) * rng.standard_normal(N); hit |= y >= h
    print("   B %2d  zeta^2/2 = %.3f vs a = %.2f  MC %.3f +- %.3f  Bachelier-Levy %.3f   (collapse: mu<0 iff B < z1^2/(2a) = %.2f)" % (B, zeta**2/2, a, hit.mean(), se(hit.mean()), bl(mu, zeta, h, tau), z1**2/(2*a)))

print("S4  flip probability between two runs whose decision variables share correlation rho: P_flip = 2 [Phi(z) - Phi2(z,z;rho)]; z = 0 -> arccos(rho)/pi")
for (zz, rho) in [(0.0, 0.0), (0.0, 0.5), (0.0, 0.9), (0.25, 0.0), (0.25, 0.5), (0.25, 0.9)]:
    X = rng.standard_normal(4 * N); Y1 = rng.standard_normal(4 * N); Y2 = rng.standard_normal(4 * N)
    Z1 = zz + sqrt(rho) * X + sqrt(1 - rho) * Y1; Z2 = zz + sqrt(rho) * X + sqrt(1 - rho) * Y2
    pf = (np.sign(Z1) != np.sign(Z2)).mean()
    # Phi2 by numerical integration over X
    xs = np.linspace(-8, 8, 4001); w = np.exp(-xs**2 / 2) / sqrt(2 * pi) * (xs[1] - xs[0])
    if rho < 1:
        pz = np.array([Phi((zz + sqrt(rho) * x) / sqrt(1 - rho)) for x in xs]); both_pos = float((w * pz**2).sum())
    pform = 2 * (Phi(zz) - both_pos)
    print("   z %.2f rho %.1f  MC %.3f  formula %.3f  (arccos(rho)/pi = %.3f; 2p(1-p) = %.3f)" % (zz, rho, pf, pform, math.acos(rho) / pi, 2 * Phi(zz) * (1 - Phi(zz))))

print("S5  STEER control: m driven by -k (m - m*) during [0, t_off], then free: P = Phi(m* sqrt(2(k-a))/sigma) for k > a")
a, s, b = 1.0, 0.1, -0.01
for (k, ms) in [(2.0, 0.1), (2.0, 0.2), (5.0, 0.1)]:
    toff = 10.0
    m = run_pitch(lambda t: b + (k * ms if t < toff else 0.0), lambda t: a - (k if t < toff else 0.0), s, toff + 15, 0.01)
    p = (m > 0).mean(); mstar_eff = (b + k * ms) / (k - a)
    print("   k %.1f m* %.2f  MC %.3f  formula Phi(m_eff sqrt(2(k-a))/sigma) = %.3f   (m_eff = (b + k m*)/(k - a) = %.3f; no control: %.3f)" % (k, ms, p, Phi(mstar_eff * sqrt(2 * (k - a)) / s), mstar_eff, Phi(b * sqrt(2 / a) / s)))
