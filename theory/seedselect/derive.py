"""seedselect derivation numerics (THEORY_SEEDSELECT_DRAFT_2026-10-14 sections 2, 4, 5). Theory and label counts only:
no run's early-window values are read. Inputs are constants from docs (THEORY_SEEDSTAB Part II) and outcome-label counts
from theory/seedselect/labels.json. Output: theory/seedselect/derive.json.
    python theory/seedselect/derive.py
"""
import json
import math
import os

import numpy as np

try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20261014)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def Phi_inv(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def biv_upper(c1, c2, rho, n=400):
    """P(X > c1, Y > c2) for a standard bivariate normal with correlation rho (Gauss-Legendre over the conditional)."""
    if abs(rho) < 1e-12:
        return (1 - Phi(c1)) * (1 - Phi(c2))
    xs, ws = np.polynomial.legendre.leggauss(n)
    lo, hi = c1, c1 + 12.0
    x = 0.5 * (hi - lo) * xs + 0.5 * (hi + lo); w = 0.5 * (hi - lo) * ws
    dens = np.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
    s = math.sqrt(max(1e-15, 1 - rho ** 2))
    cond = np.array([1 - Phi((c2 - rho * xi) / s) for xi in x])
    return float(np.sum(w * dens * cond))


# ---------- 2. decision-variance fraction R(t) in the swept-gain reduced model (Part II C3 + constant a after the ramp)
def R_of_t(t, a_max=1 / 3300.0, t0=2000.0, t_ramp=6000.0, sigma_profile="const"):
    """Fraction of Var(Z) accumulated in [t0, t]: integral of exp(-2A(s)) over [t0, t] / over [t0, inf).
    a(s) = a_max (s - t0)/(t_ramp - t0) on the ramp (gain proportional to gamma), a_max after; a = 0 (and m frozen) for s < t0.
    sigma_profile "gamma": sigma(s) proportional to gamma(s) on the ramp (the read-dependent noise also scales with gamma)."""
    def A(s):
        if s <= t0:
            return 0.0
        if s <= t_ramp:
            return a_max * (s - t0) ** 2 / (2 * (t_ramp - t0))
        return a_max * (t_ramp - t0) / 2 + a_max * (s - t_ramp)

    def sig2(s):
        if sigma_profile == "gamma" and s < t_ramp:
            return ((s - t0) / (t_ramp - t0)) ** 2
        return 1.0
    grid = np.linspace(t0, t_ramp + 40 / a_max, 200001)
    f = np.array([math.exp(-2 * A(s)) * sig2(s) for s in grid])
    cum = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(grid))])
    tot = cum[-1]
    return float(np.interp(t, grid, cum) / tot) if t > t0 else 0.0


# ---------- AUC ceiling of an ideal observer of the partial decision variable
def auc_ceiling(R, z, n=400000):
    """X = Z_t (variance fraction R of Var Z = 1, mean fraction R of mu = z), Z_inf = X + W. Y = 1{Z_inf > 0}.
    Returns AUC of X for Y (the best any feature computed by time t can do, if it measured Z_t exactly)."""
    if R <= 0:
        return 0.5
    X = RNG.normal(R * z, math.sqrt(R), n); W = RNG.normal((1 - R) * z, math.sqrt(max(1e-12, 1 - R)), n)
    Y = (X + W) > 0
    xs, xf = X[Y], X[~Y]
    if len(xs) == 0 or len(xf) == 0:
        return None
    allx = np.concatenate([xs, xf]); ranks = allx.argsort().argsort() + 1
    rs = ranks[:len(xs)].sum()
    return float((rs - len(xs) * (len(xs) + 1) / 2) / (len(xs) * len(xf)))


def flip_prob(z, rho):
    """Part II 3.2: two runs sharing a fraction rho of the decision variance flip with 2[Phi(z) - Phi2(z, z; rho)]."""
    return 2 * ((1 - Phi(-z)) - biv_upper(-z, -z, rho))


def tetrachoric(pa, pb, pdisc):
    """rho such that two threshold-normal outcomes with marginals pa, pb disagree with probability pdisc."""
    ca, cb = Phi_inv(1 - pa), Phi_inv(1 - pb)
    lo, hi = -0.999, 0.999
    for _ in range(80):
        mid = (lo + hi) / 2
        p11 = biv_upper(ca, cb, mid); d = pa + pb - 2 * p11
        if d > pdisc:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- 5. cost
def seq_cost(p, tpr, fpr, kappa):
    """Part II 6 (PROVED): E[cost]/C = [kappa + (p tpr + (1-p) fpr)(1 - kappa)] / (p tpr)."""
    return (kappa + (p * tpr + (1 - p) * fpr) * (1 - kappa)) / (p * tpr)


def binormal_best(p, auc, kappa):
    """Equal-variance binormal detector with separation d' = sqrt(2) Phi^-1(AUC); threshold chosen to minimise seq_cost."""
    if auc <= 0.5:
        return {"cost": 1 / p, "tpr": 1.0, "fpr": 1.0}
    d = math.sqrt(2) * Phi_inv(auc); best = None
    for c in np.linspace(-4, 6, 2001):
        tpr = 1 - Phi(c - d); fpr = 1 - Phi(c)
        if tpr < 1e-6:
            continue
        e = seq_cost(p, tpr, fpr, kappa)
        if best is None or e < best["cost"]:
            best = {"cost": e, "tpr": tpr, "fpr": fpr, "thr": float(c)}
    if 1 / p < best["cost"]:
        best = {"cost": 1 / p, "tpr": 1.0, "fpr": 1.0, "thr": None}
    return best


def parallel_success(p, auc, N, n=400000):
    """N seeds to t1, continue the top-scored one (binormal scores, d' from AUC). P(the continued run succeeds)."""
    d = math.sqrt(2) * Phi_inv(auc) if auc > 0.5 else 0.0
    Y = RNG.random((n, N)) < p
    S = RNG.normal(0, 1, (n, N)) + d * Y
    pick = S.argmax(1)
    return float(Y[np.arange(n), pick].mean())


def breakeven_auc_parallel(p, N, kappa):
    lo, hi = 0.5, 0.999
    cost_none = 1 / p
    f = lambda A: (N * kappa + 1 - kappa) / parallel_success(p, A, N, 200000)  # noqa: E731
    if f(hi) > cost_none:
        return None
    for _ in range(18):
        mid = (lo + hi) / 2
        if f(mid) > cost_none:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- 4. power: stratified concordance with within-stratum permutation null
def pooled_C(strata_scores):
    num = den = 0.0
    for s_succ, s_fail in strata_scores:
        if len(s_succ) == 0 or len(s_fail) == 0:
            continue
        diff = s_succ[:, None] - s_fail[None, :]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum(); den += diff.size
    return num / den if den else None


def perm_p(scores, labels_by_stratum, C_obs, n_perm, two_sided):
    cnt = 0
    for _ in range(n_perm):
        st = []
        for x, y in zip(scores, labels_by_stratum):
            yp = RNG.permutation(y); st.append((x[yp], x[~yp]))
        c = pooled_C(st)
        if (abs(c - 0.5) >= abs(C_obs - 0.5) - 1e-12) if two_sided else (c >= C_obs - 1e-12):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def power(strata, auc, two_sided, n_sim=300, n_perm=400, alpha=0.05):
    """strata: list of (n_success, n_fail). Binormal scores with AUC auc. Returns P(perm p < alpha)."""
    d = math.sqrt(2) * Phi_inv(auc) if auc > 0.5 else 0.0
    hits = 0
    for _ in range(n_sim):
        scores, labs = [], []
        for ns, nf in strata:
            y = np.array([True] * ns + [False] * nf); x = RNG.normal(0, 1, ns + nf) + d * y
            scores.append(x); labs.append(y)
        C = pooled_C([(x[y], x[~y]) for x, y in zip(scores, labs)])
        if perm_p(scores, labs, C, n_perm, two_sided) < alpha:
            hits += 1
    return hits / n_sim


def null_sd(strata):
    """SD of the pooled concordance under H0 (pair-weighted Mann-Whitney variances)."""
    num = 0.0; den = 0
    for ns, nf in strata:
        if ns and nf:
            num += (ns * nf) ** 2 * (ns + nf + 1) / (12.0 * ns * nf); den += ns * nf
    return math.sqrt(num) / den if den else None


def exact_min_p(strata):
    """Smallest attainable one-sided p of the pooled concordance (all strata perfectly ordered): prod 1/C(n, k)."""
    p = 1.0
    for ns, nf in strata:
        p /= math.comb(ns + nf, nf)
    return p


def main():
    out = {}
    # 2. R(t) and AUC ceilings (nested; Part II calibration 1/a = 3.3k, ramp 2k-6k; z = 0.43 for RMS p = 2/3, 0.25 pooled)
    ts = [2000, 4000, 6000, 7000, 8000, 10000, 12000]
    out["R_t_const_sigma"] = {t: round(R_of_t(t), 4) for t in ts}
    out["R_t_sigma_prop_gamma"] = {t: round(R_of_t(t, sigma_profile="gamma"), 4) for t in ts}
    out["R_t_a_half"] = {t: round(R_of_t(t, a_max=1 / 6600.0), 4) for t in ts}
    out["R_t_a_double"] = {t: round(R_of_t(t, a_max=1 / 1650.0), 4) for t in ts}
    out["auc_ceiling_z0.43"] = {R: round(auc_ceiling(R, 0.43), 3) for R in [0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99]}
    out["auc_ceiling_z0.25"] = {R: round(auc_ceiling(R, 0.25), 3) for R in [0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95]}
    # hare channel (Part II MC S2: P(success) by tercile of early C progress, slowest / middle / fastest)
    s = [0.511, 0.442, 0.393]; P = sum(s) / 3; Q = 1 - P
    num = sum((s[g] / 3) * ((1 - s[h]) / 3) for g in range(3) for h in range(3) if g < h) + 0.5 * sum((s[g] / 3) * ((1 - s[g]) / 3) for g in range(3))
    out["hare_tercile_auc"] = round(num / (P * Q), 3)
    # 3. cross-card flips (labels only)
    lab = json.load(open(os.path.join(OUT, "labels.json")))["labels"]
    rx = {}
    for k, v in lab.items():
        if k.startswith("replay_x/"):
            rx.setdefault(v["seed"], {})[v["gpu"]] = v["flag2k"]
    cc = {}
    for a, b in [("2070", "3070"), ("2070", "4060"), ("3070", "4060")]:
        pr = [(rx[s_][a], rx[s_][b]) for s_ in rx if a in rx[s_] and b in rx[s_]]
        pa = sum(x for x, _ in pr) / len(pr); pb = sum(y for _, y in pr) / len(pr); pd = sum(x != y for x, y in pr) / len(pr)
        rho = tetrachoric(pa, pb, pd)
        # bootstrap CI over seeds
        bs = []
        for _ in range(200):
            idx = RNG.integers(0, len(pr), len(pr)); q = [pr[i] for i in idx]
            qa = sum(x for x, _ in q) / len(q); qb = sum(y for _, y in q) / len(q); qd = sum(x != y for x, y in q) / len(q)
            if 0 < qa < 1 and 0 < qb < 1:
                bs.append(tetrachoric(qa, qb, qd))
        cc["%s-%s" % (a, b)] = {"n": len(pr), "p_a": round(pa, 3), "p_b": round(pb, 3), "discordant": round(pd, 3),
                                "independent_expectation": round(pa * (1 - pb) + pb * (1 - pa), 3), "tetrachoric_rho": round(rho, 3),
                                "rho_boot95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    out["cross_card_flag2k"] = cc
    out["auc_ceiling_seed_only"] = {k: round(auc_ceiling(max(0.0, v["tetrachoric_rho"]), Phi_inv(1 - (v["p_a"] + v["p_b"]) / 2)), 3) for k, v in cc.items()}
    # resume flip probability (nested): perturbation at t1 decorrelates the remaining variance after k* steps
    out["resume_flip"] = {}
    for t1 in (6000, 7000, 8000):
        for kstar in (350, 1000, 3000):
            rho = R_of_t(t1 + kstar)
            out["resume_flip"]["t1=%d,k*=%d" % (t1, kstar)] = {"rho": round(rho, 3), "p_flip_z0.43": round(flip_prob(0.43, rho), 4)}
    out["flip_independent_z0.43"] = round(flip_prob(0.43, 0.0), 3)
    # 5. cost tables
    cost = {}
    for fam, C in (("nested_48k", 48000), ("gsm8k_24k", 24000)):
        for t1 in (6000, 7000, 8000):
            kappa = t1 / C
            for p in (0.6, 2 / 3):
                row = {"kappa": round(kappa, 4), "no_detector": round(1 / p, 3), "perfect": round(seq_cost(p, 1, 0, kappa), 3)}
                for A in (0.55, 0.6, 0.7, 0.8, 0.9, 0.95):
                    b = binormal_best(p, A, kappa); row["auc%.2f" % A] = {"cost": round(b["cost"], 3), "tpr": round(b["tpr"], 3), "fpr": round(b["fpr"], 3)}
                for tp, fp in ((0.9, 0.1), (0.8, 0.2), (0.9, 0.5), (0.95, 0.3)):
                    row["tpr%.2f_fpr%.2f" % (tp, fp)] = round(seq_cost(p, tp, fp, kappa), 3)
                for N in (2, 3):
                    ps = {A: parallel_success(p, A, N, 200000) for A in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0 - 1e-9)}
                    row["parallel_N%d" % N] = {"cost_per_attempt": round(N * kappa + 1 - kappa, 3),
                                               "P_success": {("%.2f" % A): round(v, 3) for A, v in ps.items()},
                                               "cost_per_success": {("%.2f" % A): round((N * kappa + 1 - kappa) / v, 3) for A, v in ps.items()},
                                               "breakeven_auc": breakeven_auc_parallel(p, N, kappa)}
                cost["%s,t1=%d,p=%.3f" % (fam, t1, p)] = row
    # sequential break-even: minimal AUC for which the best binormal threshold beats no detector by >= 5 %
    for fam, C in (("nested_48k", 48000), ("gsm8k_24k", 24000)):
        for p in (0.6, 2 / 3):
            kappa = 8000 / C
            for target in (0.0, 0.05, 0.10):
                A_needed = None
                for A in np.linspace(0.51, 0.999, 490):
                    if binormal_best(p, A, kappa)["cost"] <= (1 / p) * (1 - target):
                        A_needed = round(float(A), 3); break
                cost["breakeven_seq_%s,t1=8000,p=%.3f,gain>=%.2f" % (fam, p, target)] = A_needed
    out["cost"] = cost
    # P(k/k) for registered multi-seed claims
    out["P_k_of_k"] = {"p=%.3f" % p: {k: round(p ** k, 3) for k in (2, 3, 4)} for p in (0.6, 2 / 3, 0.8, 0.9, 0.95)}
    # 4. power for the readings' strata (counts from section 1; worst case after dedupe where stated)
    strata = {
        "R1_escapefail_proto (3070: 5S/10F, 4060: 7S/6F)": [(5, 10), (7, 6)],
        "R1b_true_outcome_24k (3090 3S/1F, a4000 1S/2F, 2080ti 1S/3F)": [(3, 1), (1, 2), (1, 3)],
        "R2_flag2k_all (replay130 4060 26/14, 3070 8/9; replay_x 2070 36/24, 3070 27/13, 4060 20/20; flag4proto 3070 10/9, 4060 9/1)": [(26, 14), (8, 9), (36, 24), (27, 13), (20, 20), (10, 9), (9, 1)],
        "R3_nested_RMS_binary (2S/1F)": [(2, 1)],
        "prospective_4_new_RMS_seeds_expected (pooled with RMS s1-s3: about 5S/2F)": [(5, 2)],
        "prospective_12_new_RMS_seeds (pooled: about 10S/5F)": [(10, 5)],
    }
    pw = {}
    for name, st in strata.items():
        row = {"pairs": sum(a * b for a, b in st), "null_sd": round(null_sd(st), 4), "min_exact_p_one_sided": exact_min_p(st),
               "C_crit_two_sided_normal": round(0.5 + 1.96 * null_sd(st), 3), "C_80pct_power_normal": round(0.5 + (1.96 + 0.84) * null_sd(st), 3)}
        if row["pairs"] >= 10:
            row["power_two_sided"] = {A: power(st, A, True, n_sim=200, n_perm=300) for A in (0.6, 0.7, 0.8)}
        else:
            row["power_one_sided"] = {A: power(st, A, False, n_sim=300, n_perm=200) for A in (0.7, 0.8, 0.9)}
        pw[name] = row
    # nested entry-time ordinal: RMS (s3 vs s1, s2; s1/s2 tied) + 4 two-seed strata; exact enumeration
    from itertools import product
    dist = {}
    for rms in (0, 1, 2):
        for others in product((0, 1), repeat=4):
            k = rms + sum(others); dist[k] = dist.get(k, 0) + (1 / 3) * (1 / 16)
    pw["R3_nested_entry_ordinal (6 pairs)"] = {"P(K>=k) one-sided": {k: round(sum(v for kk, v in dist.items() if kk >= k), 4) for k in range(7)},
                                               "P(K=6 or K=0) two-sided": round(dist[6] + dist[0], 4)}
    out["power"] = pw
    json.dump(out, open(os.path.join(OUT, "derive.json"), "w", newline="\n"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
