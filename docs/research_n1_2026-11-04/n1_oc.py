"""N1 v1 (THEORY S271) DATA-SIM: shortcut floors, K'-curve, M4 stratum, swap-pair law, OC tables.

Pure simulation of the task's own law (S3.1 of N1_v1.md): no model, no repo imports, no stored outputs read.
Writes nothing; prints tables. Deterministic: every section owns a seeded RNG. Run: python -I n1_oc.py
Needs numpy. Single process.

Pinned constants (G-D MED-2 fix, 2026-11-04):
  * Floor fit (bag-MAP, last-2-MAP): for EACH cell (n, m), a fresh random.Random(FIT_SEED = 27110), N_FIT = 200,000
    draws from that cell's item law (S0 uniform, m swaps uniform over unordered pairs); the MOVED subset of those
    draws is the fit set. The build's floors(items) uses the same N and seed through so_words' own generator.
  * Prediction rule: MAP label per key, ties -> smallest normalised index; a key never seen in the fit set
    falls back to the non-initial-uniform prediction (credit 1/(n-1)), never scored as wrong.
  * DATA-SIM eval sample (not the build's test items): per cell a fresh random.Random(EVAL_SEED = 27111),
    N_EVAL = 100,000 draws, MOVED subset. At build, f is evaluated on each cell's own test items.
  * OC numpy generator: np.random.default_rng(OC_SEED = 27112).
  Seeds 27110-27112 were grepped (word match) in boxes/*.py theory/*.py scratch/*.py on 2026-11-04: no hits.
"""
import collections
import itertools
import math
import random

import numpy as np

FIT_SEED, EVAL_SEED, OC_SEED = 27110, 27111, 27112
N_FIT, N_EVAL = 200_000, 100_000
NS = 200_000  # OC simulation replicates
CELLS = [(n, m) for n in (3, 4, 5) for m in (4, 6, 8, 10, 12)]


def draw(R, n, m):
    s0 = R.randrange(n)
    sw = []
    for _ in range(m):
        a = R.randrange(n)
        b = R.randrange(n - 1)
        b += b >= a
        sw.append((a, b))
    S = [s0]
    for a, b in sw:
        x = S[-1]
        S.append(b if x == a else a if x == b else x)
    return s0, sw, S


def run(x, sw):
    for a, b in sw:
        x = b if x == a else a if x == b else x
    return x


def moved(S):
    k = sum(S[i] != S[i - 1] for i in range(1, len(S)))
    return (S[-1] != S[0] and k >= 2), k


def norm(s0, sw):
    """Relabel names by first appearance (S_0 -> 0). Returns the map and the normalised swap tuple."""
    mp = {s0: 0}
    for a, b in sw:
        for x in (a, b):
            if x not in mp:
                mp[x] = len(mp)
    return mp, tuple(tuple(sorted((mp[a], mp[b]))) for a, b in sw)


def map_table(counters):
    out = {}
    for key, c in counters.items():
        mx = max(c.values())
        out[key] = min(k for k, v in c.items() if v == mx)
    return out


def floor_cell(n, m):
    Rf = random.Random(FIT_SEED)
    bag = collections.defaultdict(collections.Counter)
    l2 = collections.defaultdict(collections.Counter)
    nfit = 0
    for _ in range(N_FIT):
        s0, sw, S = draw(Rf, n, m)
        if not moved(S)[0]:
            continue
        nfit += 1
        mp, key = norm(s0, sw)
        ans = mp[S[-1]]
        bag[tuple(sorted(key))][ans] += 1
        l2[key[-2:]][ans] += 1
    bagm, l2m = map_table(bag), map_table(l2)
    Re = random.Random(EVAL_SEED)
    acc = collections.Counter()
    T = mvc = 0
    u = 1.0 / (n - 1)
    for _ in range(N_EVAL):
        s0, sw, S = draw(Re, n, m)
        if not moved(S)[0]:
            continue
        T += 1
        mp, key = norm(s0, sw)
        ans = mp[S[-1]]
        fin = S[-1]
        acc["non_initial_unif"] += u
        acc["S4_trunc"] += S[min(4, m)] == fin
        acc["last2_only"] += run(s0, sw[-2:]) == fin
        p = [b if a == s0 else a for a, b in sw if s0 in (a, b)]
        acc["lastS0partner"] += p[-1] == fin
        c = collections.Counter(x for ab in sw for x in ab)
        c.pop(s0, None)
        best = max(c.values())
        cands = [x for x in c if c[x] == best]
        acc["mostfreq"] += (fin in cands) / len(cands)
        bk = tuple(sorted(key))
        if bk in bagm:
            acc["bag_MAP"] += bagm[bk] == ans
            acc["(bag_MAP unseen=wrong)"] += bagm[bk] == ans
        else:
            acc["bag_MAP"] += u
            acc["(bag unseen share)"] += 1
        k2 = key[-2:]
        if k2 in l2m:
            acc["last2_MAP"] += l2m[k2] == ans
        else:
            acc["last2_MAP"] += u
            acc["(last2 unseen share)"] += 1
        acc["(S_m-1)"] += S[m - 1] == fin
    r = {k: v / T for k, v in acc.items()}
    if m <= 4:  # S_4 truncation IS the answer when m <= 4: tracking, not a shortcut
        r["(S4_trunc = answer, m<=4)"] = r.pop("S4_trunc")
    cand = {k: v for k, v in r.items() if not k.startswith("(")}
    arg = max(cand, key=lambda k: (cand[k], k))
    return T / N_EVAL, cand[arg], arg, nfit, len(bag), r


def floors():
    print("== A. MOVED share and shortcut floors f(n, m) on MOVED (fit: N_FIT %d draws/cell, seed %d, MOVED subset; "
          "eval: N_EVAL %d draws/cell, seed %d; unseen key -> non-initial uniform 1/(n-1)) ==" % (N_FIT, FIT_SEED, N_EVAL, EVAL_SEED))
    print("   S_m-1 and bag_MAP-with-unseen-scored-wrong (the v1 rule) are diagnostics in parentheses, not in f.")
    out = {}
    for n, m in CELLS:
        mvs, f, arg, nfit, nkeys, r = floor_cell(n, m)
        out[(n, m)] = (mvs, f)
        print(f"n{n} m{m:2d} MOVED {mvs:.3f} f {f:.3f} ({arg}) fitMOVED {nfit} bagkeys {nkeys} | "
              + " ".join(f"{k} {v:.3f}" for k, v in sorted(r.items())))
    return out


def kcurve_m4():
    R = random.Random(20261104)
    print("== B. perfect loop at K' < m: P(S_K' = S_8 | MOVED), n5 m8 (200,000 draws) ==")
    c = collections.Counter()
    T = 0
    for _ in range(200_000):
        s0, sw, S = draw(R, 5, 8)
        if not moved(S)[0]:
            continue
        T += 1
        for k in range(9):
            c[k] += S[k] == S[8]
    print(" ".join(f"K'{k} {c[k] / T:.3f}" for k in range(9)))
    print("== D. M4 stratum at n5 m4 (exact enumeration of 5 x 10^4 equally likely (S_0, swap) configurations) ==")
    pairs = list(itertools.combinations(range(5), 2))
    tot = st = stmv = 0
    for s0 in range(5):
        for sw in itertools.product(pairs, repeat=4):
            S = [s0]
            for a, b in sw:
                x = S[-1]
                S.append(b if x == a else a if x == b else x)
            tot += 1
            if S[2] != S[4]:
                st += 1
                stmv += moved(S)[0]
    p, pm = st / tot, stmv / tot
    sd = math.sqrt(500 * p * (1 - p))
    print(f"P(S2 != S4) = {st}/{tot} = {p:.4f} (closed form 1 - [1/5 + (4/5)(1/2)^2] = 0.6): in 500 items n ~ Bin(500, 0.6), "
          f"mean {500 * p:.1f}, SD {sd:.1f}, 95% range {500 * p - 1.96 * sd:.0f}-{500 * p + 1.96 * sd:.0f}")
    print(f"P(S2 != S4 and MOVED) = {pm:.4f} -> mean {500 * pm:.1f} of 500 (if the stratum were restricted to MOVED items)")
    return p, pm


def swaps():
    R = random.Random(20261105)
    print("== C. F2 swap law: donor = same s0, redrawn swaps same m; keep iff cf not in {own, dfin, donor S_t} (200,000 tries) ==")
    for n, m, t in ((5, 8, 4), (5, 4, 2), (5, 6, 3), (3, 4, 2), (5, 12, 6)):
        kept = 0
        for _ in range(200_000):
            s0, sw, S = draw(R, n, m)
            D = [s0]
            for _ in range(m):
                a = R.randrange(n)
                b = R.randrange(n - 1)
                b += b >= a
                x = D[-1]
                D.append(b if x == a else a if x == b else x)
            cf = run(D[t], sw[t:])
            kept += cf not in (S[-1], D[-1], D[t])
        print(f"n{n} m{m} t{t}: kept share {kept / 200_000:.3f} -> tries for 1000 ~ {1000 / (kept / 200_000):.0f}")


def wil_lb(k, n, z=1.96):
    n = np.maximum(n, 1)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shift(p0, e, scale):
    """Apply a seed-level effect e to a true rate p0: 'clip' = clip(p0 + e, 0, 1); 'logit' = expit(logit(p0) + e)."""
    return np.clip(p0 + e, 0, 1) if scale == "clip" else expit(logit(p0) + e)


def oc(fl, p_m4, pmv_m4):
    rng = np.random.default_rng(OC_SEED)
    sh58, f5 = fl[(5, 8)]
    sh56 = fl[(5, 6)][0]
    sh38, f3 = fl[(3, 8)]
    print("== E. OC tables (binomial items; seed effects stated per table; test sets fixed, so n_moved and the build "
          "floor estimate f_hat are ONE draw per replicate, shared by the 3 seeds) ==")
    print(f"MOVED shares n5 m8 {sh58:.3f} (x1500 = {1500 * sh58:.0f}), n5 m6 {sh56:.3f} (x500 = {500 * sh56:.0f}), "
          f"n3 m8 {sh38:.3f} (x1500 = {1500 * sh38:.0f}); f(5,8) {f5:.3f}, f(3,8) {f3:.3f}")

    # C2-1 (unchanged rule; f not involved)
    print("E1. C2-1 (per seed m6 >= 0.85 & m8 >= 0.80; >= 2/3 of 3 seeds); sigma_seed 0.090 clipped, shared across m; m6 = m8 + 0.05:")
    for mu in (0.30, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        n8 = rng.binomial(1500, sh58, (NS, 1))
        n6 = rng.binomial(500, sh56, (NS, 1))
        z = rng.normal(0, 0.09, (NS, 3))
        o8 = rng.binomial(n8, np.clip(mu + z, 0, 1)) / n8
        o6 = rng.binomial(n6, np.clip(mu + 0.05 + z, 0, 1)) / n6
        ps = (o8 >= 0.80) & (o6 >= 0.85)
        print(f"  true m8 {mu:.2f}: P(seed) {ps.mean():.3f}  P(>=2/3) {(ps.sum(1) >= 2).mean():.3f}  P(3/3) {(ps.sum(1) == 3).mean():.3f}")

    def kappa_draw(f, share, kap, sig, scale, shape):
        nm = rng.binomial(1500, share, (shape[0], 1))
        fh = rng.binomial(nm, f) / nm  # build floor on the cell's own MOVED items (shared by seeds)
        p = shift(f + kap * (1 - f), rng.normal(0, sig, shape) if sig > 0 else np.zeros(shape), scale)
        c = rng.binomial(nm, p)
        return c, nm, (c / nm - fh) / (1 - fh), fh

    # X2
    print("E2. X2 (DIRECT arm, 3/3 seeds: K3 - K5 >= 0.15 & K3 >= 0.15 & Wilson LB(acc3) > f_hat3); sigma_seed clipped; "
          "rho = corr of n3 / n5 seed effects:")
    for sig in (0.09, 0.03):
        for rho in (0.0, 0.5):
            for k3, k5, lab in ((0.0, 0.0, "null: both at floor"), (0.10, 0.10, "null: equal kappa 0.10"),
                                (0.25, 0.0, "effect 0.25"), (0.40, 0.05, "effect 0.35"), (0.60, 0.10, "effect 0.50")):
                z1 = rng.normal(0, sig, (NS, 3))
                z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.normal(0, sig, (NS, 3))
                n3 = rng.binomial(1500, sh38, (NS, 1))
                n5 = rng.binomial(1500, sh58, (NS, 1))
                fh3 = rng.binomial(n3, f3) / n3
                fh5 = rng.binomial(n5, f5) / n5
                c3 = rng.binomial(n3, np.clip(f3 + k3 * (1 - f3) + z1, 0, 1))
                c5 = rng.binomial(n5, np.clip(f5 + k5 * (1 - f5) + z2, 0, 1))
                K3 = (c3 / n3 - fh3) / (1 - fh3)
                K5 = (c5 / n5 - fh5) / (1 - fh5)
                ps = (K3 - K5 >= 0.15) & (K3 >= 0.15) & (wil_lb(c3, n3) > fh3)
                print(f"  sig {sig} rho {rho} {lab:24s}: P(seed) {ps.mean():.3f}  P(>=2/3) {(ps.sum(1) >= 2).mean():.3f}  "
                      f"P(3/3) {(ps.sum(1) == 3).mean():.3f}")

    # X1 on DIRECT-SI
    print("E3. X1 binds on DIRECT-SI (one pass, trained on the per-swap SI questions; read at its swap-m question): "
          "kappa(n5 m8 MOVED) <= 0.25 in 3/3 seeds; 1500 items/cell; true acc = f + kappa (1 - f); "
          "f_hat = build floor on the cell's own MOVED items:")
    for sig, scale, lab in ((0.09, "clip", "sigma 0.090 clipped (primary)"), (0.05, "clip", "sigma 0.050 clipped"),
                            (0.36, "logit", "sigma 0.36 logit (= 0.09 at p 0.5)"), (0.0, "clip", "no seed effect")):
        row = []
        for kap in (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.45):
            _, _, K, _ = kappa_draw(f5, sh58, kap, sig, scale, (NS, 3))
            row.append(f"k{kap:.2f} {((K <= 0.25).sum(1) == 3).mean():.3f}")
        print(f"  {lab:36s}: P(X1 pass) " + "  ".join(row))
    for kap, lab in ((0.0, "DIRECT-SI at its shortcut floor (kappa 0)"), (0.35, "DIRECT-SI partially extrapolating (kappa 0.35)")):
        _, _, K, _ = kappa_draw(f5, sh58, kap, 0.09, "clip", (NS, 3))
        print(f"  {lab}: P(X1 pass) {((K <= 0.25).sum(1) == 3).mean():.3f}  (per seed P(kappa <= 0.25) {(K <= 0.25).mean():.3f})")

    # M2 / M3
    print("E4. M2/M3 SWAP-MID (1000 pairs; follow >= 0.50 and >= MEAN floor + 0.30, MEAN floor 0.15 measured); sigma 0.090 clipped:")
    for fu in (0.20, 0.40, 0.45, 0.55, 0.65, 0.80):
        o = rng.binomial(1000, np.clip(fu + rng.normal(0, 0.09, (NS, 3)), 0, 1)) / 1000
        fl_ = rng.binomial(1000, 0.15, (NS, 3)) / 1000
        ps = (o >= 0.50) & (o >= fl_ + 0.30)
        print(f"  true follow {fu:.2f}: P(seed) {ps.mean():.3f}  P(>=2/3) {(ps.sum(1) >= 2).mean():.3f}")

    # M5
    R = random.Random(20261106)
    pk = collections.Counter()
    for _ in range(100_000):
        pk[moved(draw(R, 5, 8)[2])[1]] += 1
    hi = sum(v for k, v in pk.items() if k >= 5) / 100_000
    lo = sum(v for k, v in pk.items() if k <= 2) / 100_000
    nh, nl = int(1500 * hi), int(1500 * lo)
    for mu in (0.85, 0.95):
        a = rng.binomial(nh, mu, NS) / nh
        b = rng.binomial(nl, mu, NS) / nl
        print(f"E5. M5 n_hi {nh} n_lo {nl}: equal true {mu}: false fail per seed {((a - b) < -0.10).mean():.4f}")

    go_gate(rng, p_m4, pmv_m4)


def go_gate(rng, p_m4, pmv_m4):
    print("E6. S1 GO gate OC (pilot = seed 1 only). GO = all of:")
    print("    L1p: NUM n5 m1-4 acc >= 0.90 each m (500 items each);")
    print("    CWS: passes 2-4 at n5 m4, CWS_t = #(WS_t & WS_t-1) / #WS_t-1 >= 0.90 each, conditional binomials on the realised")
    print("         #WS_t-1 (500 items at pass 1); per item: correct-after-correct w.p. c, correct-after-wrong w.p. rec;")
    print("    M3: SWAP-MID (5,4,t2) follow >= 0.50 and >= MEAN + 0.30 (1000 pairs; MEAN ~ Bin(1000, 0.15)/1000);")
    print("    M4: K'2 on the S2 != S4 stratum of the n5 m4 cell, n_strat ~ Bin(500, %.3f): acc vs S2 >= 0.80 AND acc vs S4 <= 0.05;" % p_m4)
    print("        p(S4) = (1 - p(S2)) q (q = share of K'2 errors that land on S4; 1/4 = uniform over the other 4 names).")
    print("    Seed effect (sigma_seed, 'clip' = added on the probability scale then clipped; 'logit' = added to the logit):")
    print("    shared part u ~ N(0, sigma sqrt(rho)) on every row + row-specific v_r ~ N(0, sigma sqrt(1 - rho)) per row")
    print("    (rows: L1p [shared by m1-4], CWS [shared by passes], M3, M4[p(S2)]); items binomial, independent per row.")

    def sim(acc=0.97, c=0.96, rec=0.25, fol=0.75, p2=0.90, q=0.25, sig=0.09, rho=0.5, scale="clip", pstrat=p_m4, detail=False):
        u = rng.normal(0, sig * math.sqrt(rho), NS) if sig > 0 else np.zeros(NS)
        v = (rng.normal(0, sig * math.sqrt(1 - rho), (4, NS)) if sig > 0 else np.zeros((4, NS)))
        pa = shift(acc, u + v[0], scale)
        L1 = np.all(rng.binomial(500, pa[:, None].repeat(4, 1)) / 500 >= 0.90, 1)
        pc = shift(c, u + v[1], scale)
        wcur = rng.binomial(500, pc)  # pass 1 (from the known S_0)
        cws_ok = np.ones(NS, bool)
        for _t in (2, 3, 4):
            stay = rng.binomial(wcur, pc)
            rcv = rng.binomial(500 - wcur, rec)
            cws_ok &= (wcur > 0) & (stay >= 0.90 * np.maximum(wcur, 1))
            wcur = stay + rcv
        pf = shift(fol, u + v[2], scale)
        fo = rng.binomial(1000, pf) / 1000
        mean_fl = rng.binomial(1000, 0.15, NS) / 1000
        M3 = (fo >= 0.50) & (fo >= mean_fl + 0.30)
        p2s = shift(p2, u + v[3], scale)
        ns = np.maximum(rng.binomial(500, pstrat, NS), 1)
        # multinomial over {S2, S4, other} on the stratum
        k2 = rng.binomial(ns, p2s)
        k4 = rng.binomial(ns - k2, np.clip(q, 0, 1))
        M4 = (k2 / ns >= 0.80) & (k4 / ns <= 0.05)
        go = L1 & cws_ok & M3 & M4
        if detail:
            return go.mean(), L1.mean(), cws_ok.mean(), M3.mean(), M4.mean(), (k2 / ns >= 0.80).mean(), (k4 / ns <= 0.05).mean()
        return go.mean()

    g = sim(detail=True)
    print("  mechanism present (acc 0.97, CWS c 0.96 rec 0.25, follow 0.75, K'2->S2 0.90, q 0.25), sigma 0.090 clip, rho 0.5 [PRIMARY]:")
    print(f"     P(GO) {g[0]:.3f} | marginal pass: L1p {g[1]:.3f} CWS {g[2]:.3f} M3 {g[3]:.3f} M4 {g[4]:.3f} "
          f"(M4 S2-half {g[5]:.3f}, S4-half {g[6]:.3f})")
    print("  sensitivity (mechanism present unless stated):")
    for lab, kw in (("sigma 0.011 clip, rho 0.5 (in-range seed spread, Note 269-1 d 4; Theory 271-2)", dict(sig=0.011)),
                    ("rho 1 (one shared seed effect)", dict(rho=1.0)),
                    ("rho 0 (row effects independent)", dict(rho=0.0)),
                    ("sigma 0.050 clip, rho 1 (v1 noise level)", dict(sig=0.05, rho=1.0)),
                    ("sigma 0.36 logit, rho 0.5", dict(sig=0.36, scale="logit")),
                    ("no seed effect (items only)", dict(sig=0.0)),
                    ("q 0.5 (K'2 errors biased to S4)", dict(q=0.5)),
                    ("CWS c 0.98", dict(c=0.98)),
                    ("CWS c 0.94", dict(c=0.94)),
                    ("acc 0.99, c 0.98, follow 0.85, K'2->S2 0.95", dict(acc=0.99, c=0.98, fol=0.85, p2=0.95)),
                    ("M4 stratum restricted to MOVED (n ~ Bin(500, %.3f))" % pmv_m4, dict(pstrat=pmv_m4)),
                    ("weak mechanism (0.93, c 0.90, follow 0.50, K'2->S2 0.70)", dict(acc=0.93, c=0.90, fol=0.50, p2=0.70))):
        print(f"     {lab:58s}: P(GO) {sim(**kw):.3f}")

    # memory unused: the decoder's input ([EOT] + text + " The answer is:") carries no pass count, so if the decoder
    # ignores the memory its K'2 prediction equals its K'8 prediction item by item (greedy argmax, deterministic eval).
    print("  memory unused (decoder output invariant to the carried state; K'2 preds == K'8 preds item by item):")
    for acc in (0.90, 0.97, 1.00):
        C = rng.binomial(500, acc, NS)  # m4 cell correct at K'8 (= vs S4)
        ns = rng.binomial(500, p_m4, NS)
        cs = rng.hypergeometric(ns, 500 - ns, C)  # correct items that fall in the stratum
        L1m4 = C / 500 >= 0.90
        M4s4 = cs / np.maximum(ns, 1) <= 0.05
        print(f"     acc(m4, K'8) {acc:.2f}: P(L1p m4 pass and M4 S4-half pass) {(L1m4 & M4s4).mean():.3f}  "
              f"(analytic: L1p m4 >= 0.90 forces acc vs S4 on the stratum >= (450 - (500 - n))/n ~ 0.83 at n 300)")


if __name__ == "__main__":
    fl = floors()
    p_m4, pmv_m4 = kcurve_m4()
    swaps()
    oc(fl, p_m4, pmv_m4)
