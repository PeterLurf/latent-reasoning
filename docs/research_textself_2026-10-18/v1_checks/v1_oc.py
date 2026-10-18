"""TS_v1 check B (T7, T4, T6, m1, m2, m11, T5): the registered OC of TEXT-SELF v1, exactly (1-D Simpson quadrature of bivariate normals; exact
binomial / Poisson sums; no Monte Carlo, no RNG).
Inputs (read only): PROOFS_TS pc_oc.json (panel deltas, the measured lineage spread s_hat 0.1666 with 3 df), the draft's ts_panels.json (panel M / M2).
Noise model (draft / PROOFS / RT): Delta_1 ~ N(m1, s^2), Delta_2 ~ N(m2, 2 s^2), corr 0.5. Central s = s_hat; sensitivity 0.20 and 0.35.
Outputs: four-way comparison OC per case (with the 'marginal' |Delta_2| <= 1 split), prior-weighted headlines, protocol-bias rows, the v1 structural
tree (TEXT-NOT-ABSORBED round 1 and round 2, TEXT-LABEL-STALL, censored), ROUND-1-ONLY label OC, the gain branch (T6 BASE-PARITY fail), gate OCs
on the latent's pi terms (m1), GATE-T2 keep at n 16 under RT's reach projections, the 2070 CARD-GATE OC, per-cell readings. python -B v1_oc.py"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "v1_oc.json")
R6 = os.path.dirname(HERE)
PC = json.load(open(os.path.join(R6, "proofs_ts", "pc_oc.json")))
PAN = json.load(open(os.path.join(R6, "scripts", "ts_panels.json")))["bases"]["s170_cot_s1"]["panels"]
S_HAT = PC["noise"]["s_hat_rms_3df"]
D = {k: tuple(v) for k, v in PC["panel_deltas"].items()}
PRI = {"N-SHIFT": 0.25, "POS-ADD": 0.20, "N-PROP": 0.15, "POS-PROP": 0.20, "BETTER": 0.10, "WORSE": 0.10}
D0_BF16 = PC["text_base"]["s170_cot_s1"]["n50"] - PC["latent_s1_n50"]["0"]
INF = math.inf
BAND = 0.5; MARG = 1.0


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def rect(m1, s1, m2, s2, rho, a, b, c, d, M=4000):
    """P(a <= X1 <= b, c <= X2 <= d), bivariate normal, Simpson over X1 (PROOFS_TS pc_oc.rect)."""
    lo = max(a, m1 - 9 * s1); hi = min(b, m1 + 9 * s1)
    if hi <= lo:
        return 0.0
    h = (hi - lo) / M; sc = s2 * math.sqrt(1 - rho ** 2); tot = 0.0
    for i in range(M + 1):
        x = lo + i * h; w = 1 if i in (0, M) else (4 if i % 2 else 2)
        mc = m2 + rho * s2 / s1 * (x - m1); pdf = math.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * math.sqrt(2 * math.pi))
        up = 1.0 if d == INF else Phi((d - mc) / sc); dn = 0.0 if c == -INF else Phi((c - mc) / sc)
        tot += w * pdf * (up - dn)
    return tot * h / 3


def four_way(m1, m2, s1, s2, rho):
    e = 1e-10
    par = rect(m1, s1, m2, s2, rho, -BAND, BAND, -BAND, BAND)
    ta = rect(m1, s1, m2, s2, rho, -BAND, INF, BAND + e, INF)
    la = rect(m1, s1, m2, s2, rho, -INF, BAND, -INF, -BAND - e)
    ta_m = rect(m1, s1, m2, s2, rho, -BAND, INF, BAND + e, MARG)          # TEXT-AHEAD with Delta_2 in (0.5, 1.0]: worded 'marginal' (T6)
    la_m = rect(m1, s1, m2, s2, rho, -INF, BAND, -MARG, -BAND - e)
    return {"PARITY": par, "TEXT-AHEAD": ta, "LATENT-AHEAD": la, "CROSSED": 1 - par - ta - la, "TEXT-AHEAD marginal": ta_m, "LATENT-AHEAD marginal": la_m}


def levels(m1, m2, s, bias=0.0):
    return four_way(m1 + bias, m2 + bias, s, s * math.sqrt(2), 0.5)


def gain(m1, m2, s, rho0):
    """T6 gain branch: Delta^g_r = Delta_r - Delta_0, with corr(e_0, e_r) = rho0 for r = 1, 2 and corr(e_1, e_2) = 0.5 (all pairwise 0.5 when rho0 = 0.5)."""
    v1 = s * s + s * s - 2 * rho0 * s * s
    v2 = 2 * s * s + s * s - 2 * rho0 * s * (s * math.sqrt(2))
    c12 = 0.5 * s * (s * math.sqrt(2)) - rho0 * s * s - rho0 * s * (s * math.sqrt(2)) + s * s
    s1, s2 = math.sqrt(v1), math.sqrt(v2)
    return four_way(m1, m2, s1, s2, c12 / (s1 * s2)), {"sd1": s1, "sd2": s2, "rho": c12 / (s1 * s2)}


def head(tab):
    return {lab: sum(PRI[k] * tab[k][lab] for k in PRI) for lab in ("PARITY", "TEXT-AHEAD", "LATENT-AHEAD", "CROSSED", "TEXT-AHEAD marginal", "LATENT-AHEAD marginal")}


def binom_pmf_list(n, p):
    out = [0.0] * (n + 1)
    if p <= 0:
        out[0] = 1.0; return out
    lq, lp = math.log1p(-p), math.log(p)
    for k in range(n + 1):
        out[k] = math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
    return out


def gate_oc(kept, w, c_frac_n=0.01, c_frac_pool=0.005, maxw=200):
    """P(pi_n >= 0.99 at every n and pi_pool >= 0.995) on the latent's label-pi terms (m1), exact: per-n binomial wrong counts, pooled by convolution.
    Count rules as SD2_v1 (point estimates): wrong_n <= floor(0.01 kept_n), wrong_pool <= floor(0.005 kept_pool)."""
    dist = [1.0]
    for k in kept:
        cn = int(math.floor(c_frac_n * k + 1e-9)); pm = binom_pmf_list(k, w)[: min(k, maxw) + 1]
        pm = [p if j <= cn else 0.0 for j, p in enumerate(pm)]
        new = [0.0] * (len(dist) + len(pm) - 1)
        for i, a in enumerate(dist):
            if a == 0:
                continue
            for j, b in enumerate(pm):
                if b:
                    new[i + j] += a * b
        dist = new
    cp = int(math.floor(c_frac_pool * sum(kept) + 1e-9))
    return sum(dist[: cp + 1]), {"c_n": [int(math.floor(c_frac_n * k + 1e-9)) for k in kept], "c_pool": cp, "kept_pool": sum(kept)}


def p_binom_ge(n, p, k):
    pm = binom_pmf_list(n, p); return sum(pm[k:])


def skellam_abs_le(lam, m, kmax=80):
    """P(|X - Y| <= m), X, Y iid Poisson(lam)."""
    po = [math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else (1.0 if k == 0 else 0.0) for k in range(kmax)]
    return sum(po[x] * po[y] for x in range(kmax) for y in range(kmax) if abs(x - y) <= m)


def main():
    res = {"s_hat": S_HAT, "s_hat_ci95": PC["noise"]["s_95ci_chi2_3df"], "measured_diffs": PC["noise"]["diffs"], "panel_deltas": D, "priors": PRI,
           "D0_bf16": D0_BF16, "band": BAND, "marginal_bound": MARG}
    cases = dict(D); cases.update({"NO-DIFF": (0.0, 0.0), "-1/-1": (-1.0, -1.0), "+1/+1": (1.0, 1.0), "-0.5/-1": (-0.5, -1.0), "+0.5/+0.5": (0.5, 0.5),
                                   "-0.5/-0.5": (-0.5, -0.5), "proportional offset (D0 x 1.5 / x 2)": (D0_BF16 * 1.5, D0_BF16 * 2.0)})
    # (1) comparison OC, levels
    lev = {}
    for s in (S_HAT, 0.20, 0.35):
        tab = {k: levels(m[0], m[1], s) for k, m in cases.items()}
        lev["s%.3f" % s] = {"by_case": tab, "headline_given_comparable": head(tab)}
        print("levels s %.3f" % s, "NO-DIFF", {k: round(v, 4) for k, v in tab["NO-DIFF"].items()}, "| -1/-1", {k: round(v, 4) for k, v in tab["-1/-1"].items()})
        print("   headline", {k: round(v, 4) for k, v in head(tab).items()})
    res["levels"] = lev
    # (2) protocol bias rows at s_hat (RT M4 / M5)
    bias = {}
    for b in (0.15, 0.30, -0.15, -0.30):
        bias["%+.2f" % b] = {k: levels(cases[k][0], cases[k][1], S_HAT, b) for k in ("NO-DIFF", "N-PROP", "POS-PROP", "-1/-1")}
    res["bias_s_hat"] = bias
    # (3) gain branch (T6): only if |Delta_0(fp32)| > 0.25; cases are gain differences (panels minus the bf16 offset)
    gb = {}
    gcases = {"NO-DIFF": (0.0, 0.0), "-1/-1": (-1.0, -1.0), "+1/+1": (1.0, 1.0)}
    gcases.update({k: (v[0] - D0_BF16, v[1] - D0_BF16) for k, v in D.items()})
    for s in (S_HAT, 0.35):
        for rho0 in (0.0, 0.5):
            tab = {}; geo = None
            for k, m in gcases.items():
                tab[k], geo = gain(m[0], m[1], s, rho0)
            gb["s%.3f_rho0_%.1f" % (s, rho0)] = {"noise": geo, "by_case": tab, "headline_given_comparable": head(tab)}
            print("gain s %.3f rho0 %.1f" % (s, rho0), "NO-DIFF PARITY %.4f" % tab["NO-DIFF"]["PARITY"], "-1/-1 LA %.4f" % tab["-1/-1"]["LATENT-AHEAD"],
                  "head", {k: round(v, 3) for k, v in head(tab).items()})
    res["gain_branch"] = gb
    # (4) ROUND-1-ONLY labels (TEXT-LABEL-STALL and TEXT-NOT-ABSORBED-2 branches): PARITY-1 iff |Delta_1| <= 0.5, else by sign
    r1 = {}
    for s in (S_HAT, 0.35):
        row = {}
        for k, m in list(cases.items()):
            p_par = Phi((BAND - m[0]) / s) - Phi((-BAND - m[0]) / s); p_ta = 1 - Phi((BAND - m[0]) / s); p_la = Phi((-BAND - m[0]) / s)
            row[k] = {"PARITY-1": p_par, "TEXT-AHEAD-1": p_ta, "LATENT-AHEAD-1": p_la}
        r1["s%.3f" % s] = {"by_case": row, "headline": {lab: sum(PRI[k] * row[k][lab] for k in PRI) for lab in ("PARITY-1", "TEXT-AHEAD-1", "LATENT-AHEAD-1")}}
    res["round1_only"] = r1
    # (5) structural tree (v1): factors [ASSUMED]
    P = {"T1_UNCENS": 0.95, "LAB1": 0.99, "ABSORB": 0.90, "GATE2": 0.93, "T2_UNCENS": 0.95, "LAB2": 0.99, "ABSORB2": 0.85, "RETAIN": 0.95}
    t1 = P["T1_UNCENS"] * P["LAB1"]; r2c = t1 * P["ABSORB"]; t2run = r2c * P["GATE2"]; t2ok = t2run * P["T2_UNCENS"] * P["LAB2"]
    comp = t2ok * P["ABSORB2"]
    tree = {"factors": P, "T1 censored or GATE-T1 / LABEL-AUDIT-T fail": 1 - t1, "TEXT-NOT-ABSORBED (round 1)": t1 * (1 - P["ABSORB"]),
            "TEXT-LABEL-STALL": r2c * (1 - P["GATE2"]), "T2 censored": t2run * (1 - P["T2_UNCENS"] * P["LAB2"]),
            "TEXT-NOT-ABSORBED-2 (T2 uncensored, not absorbed)": t2ok * (1 - P["ABSORB2"]), "both rounds comparable": comp,
            "draft-style both rounds readable (T-ABSORB-2 not required)": t2ok, "T-SELF": r2c, "T-SELF-2": t2ok * P["ABSORB2"] * P["RETAIN"]}
    tree["sum_check"] = sum(v for k, v in tree.items() if k in ("T1 censored or GATE-T1 / LABEL-AUDIT-T fail", "TEXT-NOT-ABSORBED (round 1)", "TEXT-LABEL-STALL",
                                                                   "T2 censored", "TEXT-NOT-ABSORBED-2 (T2 uncensored, not absorbed)", "both rounds comparable"))
    for s in (S_HAT, 0.35):
        h = lev["s%.3f" % s]["headline_given_comparable"]; h1 = r1["s%.3f" % s]["headline"]; ro = tree["TEXT-LABEL-STALL"] + tree["TEXT-NOT-ABSORBED-2 (T2 uncensored, not absorbed)"]
        tree["unconditional_s%.3f" % s] = {"PARITY": comp * h["PARITY"], "TEXT-AHEAD": comp * h["TEXT-AHEAD"], "LATENT-AHEAD": comp * h["LATENT-AHEAD"],
                                           "CROSSED": comp * h["CROSSED"], "ROUND-1-ONLY (stall or not-absorbed-2) total": ro,
                                           "  of which PARITY-1": ro * h1["PARITY-1"], "  of which TEXT-AHEAD-1": ro * h1["TEXT-AHEAD-1"],
                                           "  of which LATENT-AHEAD-1": ro * h1["LATENT-AHEAD-1"], "TEXT-NOT-ABSORBED (round 1)": tree["TEXT-NOT-ABSORBED (round 1)"],
                                           "censored / not run": tree["T1 censored or GATE-T1 / LABEL-AUDIT-T fail"] + tree["T2 censored"]}
    # T4 is applied to round 1 only (Theory's decision as written): an uncensored T2 that fails T-ABSORB-2 is still read four-way, with the
    # pre-stated qualifier on the same line. Its OC: Delta_1 from the panel (T1 absorbed), Delta_2 at an assumed -3 (sensitivity -2) [ASSUMED].
    na2 = {}
    for s in (S_HAT, 0.35):
        for m2na in (-3.0, -2.0):
            tab = {k: levels(D[k][0], m2na, s) for k in PRI}
            na2["s%.3f_m2_%.0f" % (s, m2na)] = {"by_panel": tab, "headline": head(tab)}
    res["not_absorbed2_branch"] = na2
    readable = t2ok
    for s in (S_HAT, 0.35):
        hp = lev["s%.3f" % s]["headline_given_comparable"]; hn = na2["s%.3f_m2_-3" % s]["headline"]
        mix = {lab: (comp * hp[lab] + (readable - comp) * hn[lab]) / readable for lab in hp}
        tree["given_both_readable_mixed_s%.3f" % s] = mix
        tree["unconditional_mixed_s%.3f" % s] = {lab: readable * v for lab, v in mix.items()}
        h1 = r1["s%.3f" % s]["headline"]; st = tree["TEXT-LABEL-STALL"]
        tree["REGISTERED_unconditional_s%.3f" % s] = {
            "four-way (both rounds readable %.4f)" % readable: {lab: readable * v for lab, v in mix.items()},
            "ROUND-1-ONLY = TEXT-LABEL-STALL %.4f" % st: {"PARITY-1": st * h1["PARITY-1"], "TEXT-AHEAD-1": st * h1["TEXT-AHEAD-1"], "LATENT-AHEAD-1": st * h1["LATENT-AHEAD-1"]},
            "TEXT-NOT-ABSORBED (round 1)": tree["TEXT-NOT-ABSORBED (round 1)"],
            "TEXT-CENSORED / not run": tree["T1 censored or GATE-T1 / LABEL-AUDIT-T fail"] + tree["T2 censored"],
            "of which T-ABSORB-2 failed inside the four-way (qualifier on the line)": readable - comp}
        tree["REGISTERED_given_both_readable_s%.3f" % s] = mix
    res["structural"] = tree
    print("structural", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in tree.items() if not k.startswith("unconditional")}))
    for s in (S_HAT, 0.35):
        print("uncond s %.3f" % s, {k: round(v, 4) for k, v in tree["unconditional_s%.3f" % s].items()})
        print("  given both readable (mixed, T-ABSORB-2 not required)", {k: round(v, 4) for k, v in tree["given_both_readable_mixed_s%.3f" % s].items()})
        print("  unconditional mixed", {k: round(v, 4) for k, v in tree["unconditional_mixed_s%.3f" % s].items()})
        print("  not-absorbed-2 branch head (m2 -3)", {k: round(v, 4) for k, v in na2["s%.3f_m2_-3" % s]["headline"].items()})
    # (6) named-outcome OC (m11): P(reading | no effect), P(reading | effect)
    res["named_outcomes"] = {
        "TEXT-NOT-ABSORBED": {"no effect (text FT changes nothing: base min a(9..12) 0.123)": t1 * 1.0, "effect (working trainer)": t1 * (1 - P["ABSORB"]),
                              "note": "by construction under no effect; it never reads as LATENT-AHEAD (T4)"},
        "TEXT-LABEL-STALL": {"zero copy-reach transfer (T1 = base curve, keep 0.645 at n 16)": None, "additive / ratio transfer": None, "half transfer": None},
        "TEXT-NOT-ABSORBED-2": {"no round-2 effect (T1's n 16 a)": "~1 x P(T2 uncensored)", "effect": t2ok * (1 - P["ABSORB2"]) / t2ok}}
    # GATE-T2 keep_n >= 0.70 at n 16 (2,000 items) under RT's reach projections for T1 (rt_reach.json T1_keep_projection) and the base
    proj = {"T1 additive": 0.978, "T1 ratio": 0.984, "T1 half transfer": 0.903, "zero transfer (base pooled 0.645)": 0.645, "base fresh 0.66": 0.66}
    res["gate_t2_keep16"] = {k: p_binom_ge(2000, p, 1400) for k, p in proj.items()}
    res["named_outcomes"]["TEXT-LABEL-STALL"] = {k: 1 - v for k, v in res["gate_t2_keep16"].items()}
    print("GATE-T2 keep16 pass", {k: round(v, 6) for k, v in res["gate_t2_keep16"].items()})
    # (7) gate OCs on the latent's pi terms (m1): GATE-T1 at the base's precondition keep (draft + fresh pooled, 400 per n)
    keep_t1 = {9: 1.0, 10: 0.9975, 11: 0.985, 12: 0.985}; kept1 = [int(round(2000 * keep_t1[n])) for n in (9, 10, 11, 12)]
    keep_t2 = {13: 0.997, 14: 0.995, 15: 0.989, 16: 0.978}; kept2 = [int(round(2000 * keep_t2[n])) for n in (13, 14, 15, 16)]
    g = {}
    for nm, kept in (("GATE-T1 (base, precondition keep)", kept1), ("GATE-T2 (T1, RT additive projection)", kept2), ("GATE-T1' (T1, keep 1.0)", [2000] * 4)):
        row = {}
        for w in (0.0006, 0.003, 0.005, 0.007, 0.010):
            p, meta = gate_oc(kept, w); row["w %.4f" % w] = p
        g[nm] = {"kept": kept, "rules": meta, "P(pass) by per-kept-chain wrong-label rate w": row}
        print(nm, kept, meta, {k: round(v, 4) for k, v in row.items()})
    res["gate_oc"] = g
    # LABEL-AUDIT-T point term (5,000 records, pi >= 0.99 <=> wrong <= 50); the CI-overlap term is ignored here (it only adds failures)
    res["label_audit_point_term"] = {"pi %.4f" % pi: sum(binom_pmf_list(5000, 1 - pi)[:51]) for pi in (0.985, 0.988, 0.99, 0.992, 0.995, 0.9994)}
    print("LABEL-AUDIT point term", {k: round(v, 4) for k, v in res["label_audit_point_term"].items()})
    res["base_reroot_purity"] = {"kept_wrong": 1, "kept": 1587, "pi": 1 - 1 / 1587, "src": "draft 788 + PROOFS fresh 799 kept at n 9-12"}
    # (8) CARD-GATE (T5): the 2070 as backup; per n |Delta a| <= 0.005 (net flips <= 5 of 1,000) at every n 1..24 and |Delta n50| <= 0.02.
    # Model: at each frontier n (base: n 9..24, 16 n) every item's correctness flips with prob q, direction random (Skellam, lam = 1000 q / 2).
    cg = {}
    a10, a11 = 0.707, 0.234; slope = a10 - a11
    for q in (0.0005, 0.001, 0.002, 0.005, 0.01):
        lam = 1000 * q / 2; pn = skellam_abs_le(lam, 5); p_all = pn ** 16
        # n50 term: n50 = 10 + (a10 - 0.5) / (a10 - a11); flips at n 10 / 11 (net k10, k11 per 1,000); P(|dn50| <= 0.02) by exact enumeration
        po = [math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) for k in range(60)]
        sk = {}
        for x in range(60):
            for y in range(60):
                sk[x - y] = sk.get(x - y, 0) + po[x] * po[y]
        n50_0 = 10 + (a10 - 0.5) / slope; ok = 0.0; joint = 0.0
        for k10, p10 in sk.items():
            for k11, p11 in sk.items():
                b10, b11 = a10 + k10 / 1000, a11 + k11 / 1000
                if b10 <= b11:
                    continue
                n50 = 10 + (b10 - 0.5) / (b10 - b11); pr = p10 * p11
                if abs(n50 - n50_0) <= 0.02:
                    ok += pr
                    if abs(k10) <= 5 and abs(k11) <= 5:
                        joint += pr
        cg["q %.4f" % q] = {"per_n_pass": pn, "all_16_frontier_n": p_all, "n50_term": ok, "n50_and_n10_n11_per_n": joint,
                            "approx_total (per-n terms x n50 given n10/n11 pass)": p_all * (joint / (pn * pn))}
    # a systematic card effect: a(n) shifted by 0.01 at one frontier n (10 net flips) -> that n fails with prob P(|10 + noise| <= 5)
    for q in (0.001, 0.005):
        lam = 1000 * q / 2; po = [math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) for k in range(60)]
        p_shift = sum(po[x] * po[y] for x in range(60) for y in range(60) if abs(10 + x - y) <= 5)
        cg["systematic +0.01 at one n, q %.3f: P(that n passes)" % q] = p_shift
    res["card_gate"] = cg
    print("CARD-GATE", json.dumps({k: (round(v["approx_total (per-n terms x n50 given n10/n11 pass)"], 4) if isinstance(v, dict) else round(v, 6)) for k, v in cg.items()}))
    # (9) per-cell readings (the draft's ts_oc arithmetic, re-run on the v1 structural factors)
    MEAN = {k: (PAN[k]["1"]["M"], PAN[k]["2"]["M"]) for k in ("N-SHIFT", "POS-ADD", "N-PROP", "POS-PROP")}
    MEAN["BETTER"] = (0.90, 0.95); MEAN["WORSE"] = (0.30, 0.30)
    cell = {}
    for sd in (0.06, 0.09, 0.12):
        row = {k: {"T-DBS": 1 - Phi((0.80 - MEAN[k][0]) / sd), "T-DBS-2": 1 - Phi((0.80 - MEAN[k][1]) / sd)} for k in PRI}
        cell["sdM %.2f" % sd] = {"by_panel": row, "prior_weighted": {r: sum(PRI[k] * row[k][r] for k in PRI) for r in ("T-DBS", "T-DBS-2")}}
    hc = cell["sdM 0.09"]["prior_weighted"]
    res["per_cell"] = {"by_sd": cell, "T-SELF": r2c, "T-DBS (sd 0.09)": r2c * hc["T-DBS"], "T-SELF-2": tree["T-SELF-2"],
                       "T-DBS-2 and T-SELF-2 (sd 0.09)": tree["T-SELF-2"] * hc["T-DBS-2"]}
    print("per-cell", {k: (round(v, 4) if isinstance(v, float) else "") for k, v in res["per_cell"].items()})
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
