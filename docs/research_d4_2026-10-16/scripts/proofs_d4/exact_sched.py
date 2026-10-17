"""PROOFS_D4 check 1: EXACT (enumeration, no Monte Carlo) pass-schedule statistics of D4 section 2.2 (BASE = D3-T, PAUSE-A, PAUSE-B).
Item law: n ~ pi(n) (uniform 1..8, or the realised trainer law under section 170's exclusion, see n_law.py); with prob 1 - p_aug a plain item
(K = min(n + U{0..4}, 12), seq R^n I^(K-n)); with prob p_aug: P ~ U{1..Pmax}, D ~ U{0..floor(P/ddiv)}, S = P - D, the S scattered pauses and the n
reductions in a uniformly random order (variant 'free': trailing pauses after the last R allowed, as d4_schedule_stats.py's shuffle does;
variant 'lastR': the last of the n + S mid positions is forced to be R, i.e. completion pass = n + P as D4 section 2.1 states), then idle to K;
K = min(n + P + U{0..4} + 1[p_tail] U{0..tail_max}, KCAP).
Per pass t = 1..K (every pass has an R2 term in vb_loss): rho = reductions before t, F = input finished (rho == n), M = marker, r = n - rho,
O = output-target finished (rho + 1[R] == n).
Weights: 'pass' = each pass weight 1 (D4's convention); 'loss' = vb_loss's R2 weighting, 1 / K_i per pass (mean over t per item, then over items).
Outputs: P(F = 0 | t >= t*), P(O = 0 | t >= t*), I(t; rho), I(t; F), I(t; M), max t + r on computing passes, test-diagonal coverage per 1M items,
P(trailing pause | augmented), exact Kb law (batch max of K over bs 64, per-batch or per-item augmentation)."""
import json, math, sys
from math import comb
from collections import defaultdict

def hyper(L, n, j1, k):
    """P(k R's among the first j1 of L positions holding n R's)."""
    S = L - n
    if k < 0 or k > n or j1 - k < 0 or j1 - k > S:
        return 0.0
    return comb(n, k) * comb(S, j1 - k) / comb(L, j1)

def mid_law(n, S, variant):
    """-> list over mid positions j = 1..n+S of dict {(rho_before, typ): prob}."""
    out = []
    if variant == "lastR" and S > 0:
        L1 = n - 1 + S                                                       # first L1 positions: n-1 R and S P uniformly; position L1+1 = R
        for j in range(1, L1 + 1):
            d = {}
            for k in range(0, n):
                pk = hyper(L1, n - 1, j - 1, k)
                if pk == 0:
                    continue
                pR = (n - 1 - k) / (L1 - j + 1)
                if pR > 0:
                    d[(k, "R")] = d.get((k, "R"), 0) + pk * pR
                if pR < 1:
                    d[(k, "P")] = d.get((k, "P"), 0) + pk * (1 - pR)
            out.append(d)
        out.append({(n - 1, "R"): 1.0})
        return out
    L = n + S
    for j in range(1, L + 1):
        d = {}
        for k in range(0, n + 1):
            pk = hyper(L, n, j - 1, k)
            if pk == 0:
                continue
            pR = (n - k) / (L - j + 1)
            if pR > 0:
                d[(k, "R")] = d.get((k, "R"), 0) + pk * pR
            if pR < 1:
                d[(k, "P")] = d.get((k, "P"), 0) + pk * (1 - pR)
        out.append(d)
    return out

def item_cases(spec, pi):
    """yield (prob, n, K, aug, per-pass list of {(rho, typ): prob}) - exact."""
    for n in range(1, 9):
        pn = pi[n]
        if pn == 0:
            continue
        pl = 1.0 - spec.get("p_aug", 0.0)
        if pl > 0:
            for e in range(5):
                K = min(n + e, 12)
                yield pn * pl / 5, n, K, False, [{(t - 1, "R"): 1.0} for t in range(1, n + 1)] + [{(n, "I"): 1.0}] * (K - n)
        if spec.get("p_aug", 0) == 0:
            continue
        Pm = spec["P_max"]; dd = spec.get("D_div", 1)
        for P in range(1, Pm + 1):
            for D in range(0, P // dd + 1):
                pPD = 1 / Pm / (P // dd + 1); S = P - D; ml = mid_law(n, S, spec["variant"])
                passes = [{(0, "P"): 1.0}] * D + ml
                assert len(passes) == n + P
                pt = spec.get("p_tail", 0.0); tm = spec.get("tail_max", 0)
                ext = defaultdict(float)
                for e in range(5):
                    ext[e] += (1 - pt) / 5
                    if pt > 0:
                        for u in range(tm + 1):
                            ext[e + u] += pt / 5 / (tm + 1)
                Kl = defaultdict(float)
                for x, px in ext.items():
                    Kl[min(n + P + x, spec["KCAP"])] += px
                for K, pK in Kl.items():
                    assert K >= n + P, ("censored", n, P, K)
                    yield pn * spec["p_aug"] * pPD * pK, n, K, True, passes + [{(n, "I"): 1.0}] * (K - n - P)

def mi(j):
    tot = sum(j.values()); pa = defaultdict(float); pb = defaultdict(float)
    for (a, b), v in j.items():
        pa[a] += v; pb[b] += v
    return sum(v / tot * math.log2(v * tot / (pa[a] * pb[b])) for (a, b), v in j.items() if v > 0)

def stats(spec, pi, bs=64):
    W = {"pass": lambda K: 1.0, "loss": lambda K: 1.0 / K}; res = {}
    Klaw = {False: defaultdict(float), True: defaultdict(float)}; trail = [0.0, 0.0]
    acc = {w: {"jF": defaultdict(float), "jM": defaultdict(float), "jR": defaultdict(float), "unfI": defaultdict(float), "unfO": defaultdict(float), "all": defaultdict(float)} for w in W}
    cov = defaultdict(float); trmax = 0; pause_after_fin = 0.0; tr_hist = defaultdict(float)
    for p, n, K, aug, passes in item_cases(spec, pi):
        Klaw[aug][K] += p
        if aug:
            trail[1] += p
            last = passes[n - 1 + (len(passes) - n) * 0] if False else None
        for t, d in enumerate(passes, 1):
            for (rho, typ), q in d.items():
                pq = p * q; F = int(rho >= n); M = int(typ == "P"); O = int(rho + (typ == "R") >= n); r = n - rho
                if typ == "R":
                    cov[(t, r)] += pq; trmax = max(trmax, t + r); tr_hist[t + r] += pq
                if typ == "P" and F:
                    pause_after_fin += pq
                for w, f in W.items():
                    a = acc[w]; ww = pq * f(K)
                    a["jF"][(t, F)] += ww; a["jM"][(t, M)] += ww; a["jR"][(t, rho)] += ww; a["all"][t] += ww
                    if not F:
                        a["unfI"][t] += ww
                    if not O:
                        a["unfO"][t] += ww
    for w in W:
        a = acc[w]; T = max(a["all"])
        ge = lambda dct, ts: sum(v for t, v in dct.items() if t >= ts)  # noqa: E731
        res[w] = {"I_t_rho": mi(a["jR"]), "I_t_F": mi(a["jF"]), "I_t_M": mi(a["jM"]) if any(k[1] for k in a["jM"]) else 0.0,
                  "P_unfinished_input_given_t_ge": {ts: ge(a["unfI"], ts) / ge(a["all"], ts) for ts in range(6, T + 1)},
                  "P_unfinished_output_target_given_t_ge": {ts: ge(a["unfO"], ts) / ge(a["all"], ts) for ts in range(6, T + 1)},
                  "P_unfinished_input_at_t": {t: a["unfI"][t] / a["all"][t] for t in range(1, T + 1)}}
    # exact Kb law: per-item augmentation: F(k)^bs with F the mixed per-item CDF; per-batch: p_aug F_aug^bs + (1-p_aug) F_plain^bs
    def cdf(law, k):
        tot = sum(law.values()); return sum(v for K, v in law.items() if K <= k) / tot if tot else 1.0
    Kmax = max(list(Klaw[False]) + list(Klaw[True]))
    pa = spec.get("p_aug", 0.0)
    mix = defaultdict(float)
    for aug in (False, True):
        for K, v in Klaw[aug].items():
            mix[K] += v
    def Emax(F):
        return sum(1 - F(k) for k in range(0, Kmax))       # E[max] = sum_{k>=0} P(max > k)
    Kb_item = Emax(lambda k: cdf(mix, k) ** bs)
    Kb_batch = Emax(lambda k: (1 - pa) * cdf(Klaw[False], k) ** bs + pa * (cdf(Klaw[True], k) ** bs if Klaw[True] else 1.0))
    Kb_aug_only = Emax(lambda k: cdf(Klaw[True], k) ** bs) if Klaw[True] else None
    Kb_plain_only = Emax(lambda k: cdf(Klaw[False], k) ** bs)
    meanK = sum(K * v for K, v in mix.items()) / sum(mix.values())
    tot_aug = sum(Klaw[True].values())
    res["max_t_plus_r_computing"] = trmax
    res["t_plus_r_hist_per_item"] = {k: v for k, v in sorted(tr_hist.items())}
    res["coverage_per_1M"] = {n: {t: cov.get((t, n - t + 1), 0.0) * 1e6 for t in range(1, n + 1)} for n in range(9, 17)}
    res["pause_with_finished_input_per_item"] = pause_after_fin
    res["mean_K"] = meanK; res["Kb_mean_per_item_aug"] = Kb_item; res["Kb_mean_per_batch_aug"] = Kb_batch
    res["Kb_mean_aug_batches"] = Kb_aug_only; res["Kb_mean_plain_batches"] = Kb_plain_only
    res["P_K_capped_aug"] = None
    return res

def trailing_prob(spec, pi):
    """P(an augmented item has >= 1 scattered pause after its last reduction) under the 'free' shuffle: 1 - n/(n+S) ... exact = S/(n+S)."""
    tot = 0.0; num = 0.0
    Pm = spec["P_max"]; dd = spec.get("D_div", 1)
    for n in range(1, 9):
        for P in range(1, Pm + 1):
            for D in range(0, P // dd + 1):
                w = pi[n] / Pm / (P // dd + 1); S = P - D; tot += w; num += w * (S / (n + S))
    return num / tot

if __name__ == "__main__":
    out_path = sys.argv[1]; pi_kind = sys.argv[2] if len(sys.argv) > 2 else "uniform"
    if pi_kind == "uniform":
        pi = {n: 1 / 8 for n in range(1, 9)}
    else:
        pi = {int(k): v for k, v in json.load(open(pi_kind))["pi_accepted"].items()}
    SPECS = {"BASE": {"variant": "free"},
             "A_free": {"p_aug": 0.5, "P_max": 8, "KCAP": 20, "variant": "free"},
             "A_lastR": {"p_aug": 0.5, "P_max": 8, "KCAP": 20, "variant": "lastR"},
             "B_free": {"p_aug": 0.5, "P_max": 12, "KCAP": 24, "p_tail": 0.25, "tail_max": 8, "D_div": 3, "variant": "free"},
             "B_lastR": {"p_aug": 0.5, "P_max": 12, "KCAP": 24, "p_tail": 0.25, "tail_max": 8, "D_div": 3, "variant": "lastR"}}
    res = {}
    for k, s in SPECS.items():
        res[k] = stats(s, pi)
        if k != "BASE":
            res[k]["P_trailing_pause_given_aug_free"] = trailing_prob(s, pi)
    json.dump(res, open(out_path, "w", newline="\n"), indent=1, default=str)
    for k, v in res.items():
        print("==", k, "max t+r", v["max_t_plus_r_computing"], "meanK %.4f Kb item %.3f batch %.3f augB %s plainB %.3f" % (v["mean_K"], v["Kb_mean_per_item_aug"], v["Kb_mean_per_batch_aug"], v["Kb_mean_aug_batches"], v["Kb_mean_plain_batches"]),
              "trail", v.get("P_trailing_pause_given_aug_free"), "pause-with-finished-input/item %.4f" % v["pause_with_finished_input_per_item"])
        for w in ("pass", "loss"):
            x = v[w]
            print("  [%s] I(t;rho) %.4f I(t;F) %.4f I(t;M) %.4f" % (w, x["I_t_rho"], x["I_t_F"], x["I_t_M"]))
            print("   P(F=0|t>=t*):", " ".join("%d:%.4f" % (t, p) for t, p in x["P_unfinished_input_given_t_ge"].items() if t in (8, 9, 10, 12, 14, 15, 16, 17, 18, 20, 21, 22)))
            print("   P(O=0|t>=t*):", " ".join("%d:%.4f" % (t, p) for t, p in x["P_unfinished_output_target_given_t_ge"].items() if t in (8, 9, 10, 12, 14, 15, 16, 17, 18, 20, 21, 22)))
        print("  cov n9..12 t>=4:", {n: {t: round(c) for t, c in row.items() if t >= 4} for n, row in v["coverage_per_1M"].items() if n <= 12})
