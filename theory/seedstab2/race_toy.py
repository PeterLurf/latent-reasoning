"""SEEDSTAB2 check R: a cartoon of Part I sec 5.4's threshold kernel with an ABSORBABLE drive (the full-trunk pass-token aux: the reader-side
skill c of the contractive class C soaks up the residual on the passes within reader depth) and a NON-ABSORBABLE drive (SVR: a shallow pass-free
value readout that only state-carried progress can satisfy, except a 'precompute' share p_pre of the shallow targets that C could game).
Per seed (Euler-Maruyama, N paths):
  c(t)   = 1 - exp(-r_C t), r_C lognormal across seeds (C's speed varies; Part II S2)
  cov(e) = e^2 / (e^2 + 1)        (progress coverage; e = closed-loop progress gain, e_needed = 1)
  D_aux  = (1 - cov) [ (1 - c) s_early + s_late ]
  D_svr  = (1 - cov) [ s_deep + s_shal (1 - p_pre c) ]
  de     = eta(t) [ 2 e (D_aux + w D_svr) + w g_add (1 - cov) (s_deep + s_shal (1 - p_pre c)) ] dt + sig e dW   (multiplicative noise)
  eta(t) = cosine schedule over [t_on, T]; the SVR term is on only for t >= t_on_svr (the loop-closing step)
Success = cov(e(T)) >= 0.5. ln e0 ~ N(mu0, s0^2). Calibrated so that w = 0 gives P ~ 0.6-0.67 (the observed base rate).
Illustrative only (CHECK of the kernel's logic, not of the network). Output: race_toy.json beside this file."""
import json, os
import numpy as np

rng = np.random.default_rng(199002)
N, T, dt = 20000, 48.0, 0.02                                               # time in k-steps
t_on, t_on_svr = 2.0, 2.0
s_early, s_late = 0.55, 0.06                                               # aux residual shares: within C's reader depth / beyond it
s_shal, s_deep = 0.57 * 0.6, 0.43 * 0.6                                    # SVR shares (t >= 2 heights from heights.json: h<=2 57%, h>=3 43%), scaled
sig, mu0, s0 = 0.25, np.log(0.02), 1.0


def run(w, p_pre, g_add, seed=0, w_abs=0.0):
    r = np.random.default_rng(seed); e = np.exp(mu0 + s0 * r.standard_normal(N)); rC = np.exp(np.log(0.5) + 0.5 * r.standard_normal(N))
    for k in range(int(T / dt)):
        t = k * dt
        if t < t_on:
            continue
        eta = 0.5 * (1 + np.cos(np.pi * t / T)); c = 1 - np.exp(-rC * t); cov = e * e / (e * e + 1)
        on = 1.0 if t >= t_on_svr else 0.0
        Dsv = (1 - cov) * (s_deep + s_shal * (1 - p_pre * c)) * on
        Dax = (1 - cov) * ((1 - c) * s_early + s_late)
        e = e + eta * (2 * e * ((1 + w_abs) * Dax + w * Dsv) + w * g_add * Dsv) * dt + sig * e * np.sqrt(dt) * r.standard_normal(N)
        e = np.abs(e)
    return float(np.mean(e * e / (e * e + 1) >= 0.5))


from statistics import NormalDist
lo, hi = 0.0, 0.6
for _ in range(16):                                                         # calibrate s_late (the base recipe's non-absorbable aux share): P(w = 0) = 0.62
    s_late = 0.5 * (lo + hi); p0 = run(0.0, 0.0, 0.0)
    lo, hi = (lo, s_late) if p0 > 0.62 else (s_late, hi)
print("calibrated s_late = %.4f, P(w=0) = %.3f" % (s_late, p0), flush=True)
zf = lambda p: NormalDist().inv_cdf(min(max(p, 1e-4), 1 - 1e-4))  # noqa: E731
out = {"params": {"N": N, "T_k": T, "s_early": s_early, "s_late": s_late, "s_shal": s_shal, "s_deep": s_deep, "sig": sig, "mu0": float(mu0), "s_late_calibrated": s_late, "s0": s0}, "grid": []}
for g_add in (0.0,):
    for p_pre in (0.0, 0.5, 1.0):
        for w in (0.0, 0.25, 0.5, 1.0, 2.0):
            if w == 0.0 and (p_pre > 0 or g_add > 0):
                continue
            p = run(w, p_pre, g_add); out["grid"].append({"w": w, "p_pre": p_pre, "g_add": g_add, "P_success": p, "z": zf(p)})
            print("g_add %.2f p_pre %.1f w %.2f -> P %.3f z %.2f" % (g_add, p_pre, w, p, zf(p)), flush=True)
for wa in (1.0, 2.0):                                                     # control: the same weight on an ABSORBABLE drive (more pass-token aux)
    p = run(0.0, 0.0, 0.0, w_abs=wa); out["grid"].append({"w_abs": wa, "P_success": p, "z": zf(p)}); print("absorbable control w_abs %.1f -> P %.3f z %.2f" % (wa, p, zf(p)), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "race_toy.json"), "w", newline="\n"), indent=1)
