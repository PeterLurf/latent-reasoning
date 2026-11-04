# MT JOINT-2 OC block (DATA-SIM; numpy only; python -I). Seed effect: clipped normal per family shared across that family's rows (rho_f),
# item noise binomial. Bars = the registered single-task bars (score_fp RETAIN set / score_n1 loop rows).
import numpy as np
R = np.random.default_rng(27460)
NSIM = 20000
def clip(x): return np.clip(x, 0.0, 1.0)
def seedpass_wp(mu_in, mu6, mu7, sig, rho=0.5, size=NSIM):
    z0 = R.standard_normal(size)
    def eff(s): return s * (np.sqrt(rho) * z0 + np.sqrt(1 - rho) * R.standard_normal(size))
    ok = np.ones(size, bool)
    # in-range: TEST2 d1-4 >= 0.98 (n 1000 each, d4 binding-most), TEST-PERM d4 >= 0.90 (true mu_in - 0.02), LATE-PREMISE >= 0.85 (true mu_in-0.01)
    for bar, off in ((0.98, 0.0), (0.90, -0.02), (0.85, -0.01)):
        p = clip(mu_in + off + eff(0.011)); ok &= R.binomial(1000, p) / 1000 >= bar
    # NOT-DECODER / SWAP4 / LP5 / C6 lumped at 0.98 per seed when retained (DATA: of 2/2, bun 2/2)
    ok &= R.random(size) < (0.98 if mu_in > 0.98 else 0.5)
    for mu, bar in ((mu6, 0.971 - 0.05), (mu7, 0.909 - 0.05)):
        p = clip(mu + eff(sig)); ok &= R.binomial(500, p) / 500 >= bar
    return ok
def seedpass_so(mu8, sig, qmech, rho=0.5, size=NSIM):
    z0 = R.standard_normal(size)
    def eff(s): return s * (np.sqrt(rho) * z0 + np.sqrt(1 - rho) * R.standard_normal(size))
    ok = np.ones(size, bool)
    for m in range(1, 5):  # L1 n5 m1-4 >= 0.95, 500 items, true 0.98
        p = clip(0.98 + eff(0.011)); ok &= R.binomial(500, p) / 500 >= 0.95
    p8 = clip(mu8 + eff(sig)); p6 = clip(mu8 + 0.05 + eff(sig) * 0 + (p8 - mu8))
    ok &= R.binomial(1060, p8) / 1060 >= 0.80
    ok &= R.binomial(299, clip(p6)) / 299 >= 0.85
    ok &= R.random(size) < qmech
    return ok, p8
def arm(ok, k=2):  # >= k of 3 seeds
    o = ok.reshape(-1, 3).sum(1); return (o >= k).mean()
print("== per-seed and arm (>= 2/3) P(pass) ==")
print("-- WP RETAIN block (in-range + C2-RETAIN vs of_s1 - 0.05), sigma_seed at depth")
for sig in (0.09, 0.03):
    for name, mu_in, mu6, mu7 in (("retained (of-like)", 0.993, 0.965, 0.900), ("depth -0.05", 0.993, 0.915, 0.850),
                                   ("depth -0.15", 0.993, 0.815, 0.750), ("in-range collapse 0.95", 0.95, 0.90, 0.80)):
        ok = seedpass_wp(mu_in, mu6, mu7, sig, size=3 * NSIM)
        print("  sig %.2f %-24s seed %.3f  arm %.3f" % (sig, name, ok.mean(), arm(ok)))
print("-- SO N1 loop block (L1, C2-1, mechanism lumped q)")
for sig in (0.09, 0.03):
    for mu8 in (0.30, 0.70, 0.80, 0.90, 0.95):
        for q in (0.5, 0.8):
            ok, _ = seedpass_so(mu8, sig, q, size=3 * NSIM)
            print("  sig %.2f mu8 %.2f q %.1f  seed %.3f  arm %.3f" % (sig, mu8, q, ok.mean(), arm(ok)))
print("-- R-SO arm row: mean over 3 joint seeds minus mean over 3 LOOP-SI seeds of acc(m 8 MOVED) >= -0.10 (n 1060 per cell)")
for sig in (0.09, 0.05, 0.03):
    for delta in (0.0, -0.05, -0.10, -0.20, -0.30):
        mu = 0.85
        js = clip(mu + delta + sig * R.standard_normal((NSIM, 3))); ss = clip(mu + sig * R.standard_normal((NSIM, 3)))
        jo = R.binomial(1060, js) / 1060; so = R.binomial(1060, ss) / 1060
        d = jo.mean(1) - so.mean(1)
        print("  sig %.2f true delta %+.2f  P(pass) %.3f   P(NEAR band |d+0.10| < %.3f) %.3f" % (sig, delta, (d >= -0.10).mean(), sig * np.sqrt(2 / 3), (np.abs(d + 0.10) < sig * np.sqrt(2 / 3)).mean()))
print("-- JOINT-2-PASS arm (>= 2/3 seeds pass WP block AND SO block), WP/SO independent")
for sig in (0.09, 0.03):
    for nm, wp, so_ in (("no interference", (0.993, 0.965, 0.900), (0.90, 0.8)), ("no interference, weak mech", (0.993, 0.965, 0.900), (0.90, 0.5)),
                         ("SO depth -0.10", (0.993, 0.965, 0.900), (0.80, 0.8)), ("WP depth -0.15", (0.993, 0.815, 0.750), (0.90, 0.8)),
                         ("SO no extrapolation", (0.993, 0.965, 0.900), (0.30, 0.8))):
        w = seedpass_wp(*wp, sig, size=3 * NSIM); s, _ = seedpass_so(so_[0], sig, so_[1], size=3 * NSIM)
        print("  sig %.2f %-28s seed %.3f arm %.3f" % (sig, nm, (w & s).mean(), arm(w & s)))
print()
print("== REVISED (registered form): per-seed JOINT-C1 block = in-range + mechanism rows only; depth retention on arm seed-means ==")
def seed_c1(wp_in, so_q, wp_mech=0.98, size=NSIM):
    z = R.standard_normal(size); ok = np.ones(size, bool)
    for bar, off in ((0.98, 0.0), (0.90, -0.02), (0.85, -0.01)):
        ok &= R.binomial(1000, clip(wp_in + off + 0.011 * z)) / 1000 >= bar
    ok &= R.random(size) < wp_mech
    for m in range(4):
        ok &= R.binomial(500, clip(0.98 + 0.011 * R.standard_normal(size))) / 500 >= 0.95
    ok &= R.random(size) < so_q
    return ok
for nm, wi, q, wm in (("retained, SO mech in-range 0.9", 0.993, 0.9, 0.98), ("retained, SO mech 0.7", 0.993, 0.7, 0.98),
                      ("WP in-range 0.975 (slight loss)", 0.975, 0.9, 0.98), ("SO mechanism absent (q 0.0)", 0.993, 0.0, 0.98),
                      ("WP mechanism broken (0.2)", 0.993, 0.9, 0.2)):
    ok = seed_c1(wi, q, wm, size=3 * NSIM); print("  JOINT-C1 %-34s seed %.3f arm(>=2/3) %.3f" % (nm, ok.mean(), arm(ok)))
print("-- R-WP-DEPTH arm row: mean over 3 joint seeds of acc d6 >= 0.921 AND d7 >= 0.859 (of_s1 - 0.05; 500 items per d)")
for sig in (0.09, 0.05, 0.03):
    for dl in (0.0, -0.05, -0.10, -0.15):
        z = R.standard_normal((NSIM, 3))
        a6 = R.binomial(500, clip(0.965 + dl + sig * z)) / 500; a7 = R.binomial(500, clip(0.900 + dl + sig * z)) / 500
        ok = (a6.mean(1) >= 0.921) & (a7.mean(1) >= 0.859)
        print("  sig %.2f true depth shift %+.2f  P(pass) %.3f" % (sig, dl, ok.mean()))
print()
print("-- stage-1 WP GO rows alone (TEST2 d1-4 >= 0.98 [d4 binding], TEST-PERM d4 >= 0.90, LATE-PREMISE >= 0.85; per seed, seed sd 0.011)")
for wi in (0.993, 0.985, 0.975):
    ok = seed_c1(wi, 1.0, 1.0); print("  WP in-range true %.3f  P(pass) %.3f" % (wi, ok.mean()))
