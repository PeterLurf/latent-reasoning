"""SD2_v1 check 1 ($0, CPU, exact rationals; numpy-free). Round-2 law under Theory's D10:
  p_self 1/2; self n: window n 13..16 = 2/3 of self items (1/6 per n), window n 9..12 = 1/3 (1/12 per n); K = min(n + U{0..4}, 20).
  Plain n ~ the trainer's exclusion-weighted law on 1..8 (h_1 .8852, h_2 .0284, h_3 .0002; item_space.json, as PROOFS), K = min(n + U{0..4}, 12).
  This is PROOFS_SD2's p2_support_s1.py construction (an R2 pass t <= K is 'R' iff t <= n) with the self law replaced; the 50/50 law is re-run as a
  regression check (must reproduce p2_support_s1.json: P(R | t >= 13) = 5/11).
(1) S1 table P(R | t >= t*), loss-weighted variant, deadline cost at t_d (share of an item's R2 term), t* 9..17.
(2) G-S3 target at t* 13 (exact) and its delta-method SE at N self items; the seed-1 replay makes the realised value precomputable.
(3) Exposure under the seed-1 replay (PROOFS #39: 5,992 self steps, 383,488 items): per-n and per-window expectations for the D10 pool composition,
    hypergeometric SDs for a random-permutation order; ratios to round 1's absorption points (PROOFS #40: 46,416 / 74,176 items per n).
(4) Pool sizing: per-n counts, spare records, z(exhaust) for a non-replay seed (s2: Binomial(12000, 1/2) self steps).
(5) LABEL-AUDIT-2 exact OC (pooled 5,000; per window at the expected D10 split); RETAIN; K'24 binomial floor.
(6) Per-micro-batch rollout length Kb (vb_loss: Kb = max K over the micro-batch) under D10 for chunk sizes 64 / 32 / 16 / 8, and round 1's analogue."""
import json
import math
import os
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
h = {1: Fr(8852, 10000), 2: Fr(284, 10000), 3: Fr(2, 10000)}
w = {n: 1 - h.get(n, Fr(0)) for n in range(1, 9)}; Z = sum(w.values()); PLAIN = {n: w[n] / Z for n in w}
LAW_D10 = {**{n: Fr(1, 12) for n in range(9, 13)}, **{n: Fr(1, 6) for n in range(13, 17)}}
LAW_5050 = {n: Fr(1, 8) for n in range(9, 17)}
assert sum(LAW_D10.values()) == 1 and sum(LAW_5050.values()) == 1


def items(selfl, kmax_self=20):
    it = []
    for n, p in PLAIN.items():
        for u in range(5):
            it.append((Fr(1, 2) * p * Fr(1, 5), n, min(n + u, 12)))
    for n, p in selfl.items():
        for u in range(5):
            it.append((Fr(1, 2) * p * Fr(1, 5), n, min(n + u, kmax_self)))
    assert sum(q for q, _, _ in it) == 1
    return it


def s1_table(selfl):
    it = items(selfl); res = {}
    for ts in range(9, 18):
        num = sum(q * max(0, min(n, K) - ts + 1) for q, n, K in it); den = sum(q * max(0, K - ts + 1) for q, n, K in it)
        numw = sum(q * Fr(max(0, min(n, K) - ts + 1), K) for q, n, K in it); denw = sum(q * Fr(max(0, K - ts + 1), K) for q, n, K in it)
        res[ts] = {"P": float(num / den) if den else None, "P_exact": str(num / den) if den else None, "P_lossw": float(numw / denw) if denw else None,
                   "deadline_cost_lossw": float(numw)}
    return res


def gs3(selfl):
    pr = [(selfl[n] * Fr(1, 5), n, u) for n in selfl for u in range(5)]
    X = lambda n, u: max(0, n - 12); Y = lambda n, u: max(0, min(n + u, 20) - 12)  # noqa: E731
    EX = sum(p * X(n, u) for p, n, u in pr); EY = sum(p * Y(n, u) for p, n, u in pr); R = EX / EY
    V = sum(p * (X(n, u) - R * Y(n, u)) ** 2 for p, n, u in pr)
    return {"target_exact": str(R), "target": float(R), "se_by_N": {N: math.sqrt(float(V) / N) / float(EY) for N in (20000, 383488)}}


def kb(selfl, kmax=20):
    Kd = {}
    for n, p in selfl.items():
        for u in range(5):
            k = min(n + u, kmax); Kd[k] = Kd.get(k, 0) + p * Fr(1, 5)
    ks = sorted(Kd); cdf = {}; c = Fr(0)
    for k in ks:
        c += Kd[k]; cdf[k] = c
    out = {}
    for b in (64, 32, 16, 8):
        e = 0.0; prev = 0.0
        for k in ks:
            pk = float(cdf[k]) ** b; e += k * (pk - prev); prev = pk
        out[b] = {"P_Kb_max": 1 - float(cdf[ks[-2]]) ** b, "E_Kb": e, "Kmax": ks[-1]}
    return out


out = {"law_D10": {n: str(v) for n, v in LAW_D10.items()}}
out["S1_D10"] = s1_table(LAW_D10); out["S1_5050_regression"] = s1_table(LAW_5050)
out["GS3_D10"] = gs3(LAW_D10); out["GS3_5050_regression"] = gs3(LAW_5050)
assert out["GS3_5050_regression"]["target_exact"] == "5/11", out["GS3_5050_regression"]["target_exact"]
out["Kb_D10"] = kb(LAW_D10); out["Kb_5050"] = kb(LAW_5050); out["Kb_round1"] = kb({n: Fr(1, 4) for n in range(9, 13)}, kmax=16)

# (3) exposure under the seed-1 replay
NSELF = 5992; ITEMS = 64 * NSELF                                           # PROOFS #39 (round-1 draws_log; rng_s replay)
CNT = {**{n: 37000 for n in range(9, 13)}, **{n: 74000 for n in range(13, 17)}}   # pool per-n counts (D10: 2 : 1 by window)
M = sum(CNT.values())


def hyp_sd(m, K, N):
    return math.sqrt(m * (K / N) * (1 - K / N) * (N - m) / (N - 1))


exp_ = {n: {"mean": ITEMS * CNT[n] / M, "sd": hyp_sd(ITEMS, CNT[n], M)} for n in CNT}
win = {"n13_16": {"mean": ITEMS * sum(CNT[n] for n in range(13, 17)) / M, "sd": hyp_sd(ITEMS, sum(CNT[n] for n in range(13, 17)), M)},
       "n9_12": {"mean": ITEMS * sum(CNT[n] for n in range(9, 13)) / M, "sd": hyp_sd(ITEMS, sum(CNT[n] for n in range(9, 13)), M)}}
R1_SUST_002, R1_SUST_001, R1_FIRST_001, R1_TOTAL = 46416, 74176, 22864, 95872   # PROOFS #40 (items per n); round 1 total 383,488 / 4
out["exposure"] = {"self_steps": NSELF, "items": ITEMS, "pool_counts": CNT, "pool_total": M, "spare_records": M - ITEMS, "per_n": exp_, "window": win,
                   "share_new_window_sd": win["n13_16"]["sd"] / ITEMS,
                   "new_per_n_over_round1_sustained_ce_0.02": exp_[16]["mean"] / R1_SUST_002, "new_per_n_over_round1_sustained_ce_0.01": exp_[16]["mean"] / R1_SUST_001,
                   "new_per_n_over_round1_first_ce_0.01": exp_[16]["mean"] / R1_FIRST_001,
                   "old_per_n": exp_[9]["mean"], "old_per_n_over_round1_total": exp_[9]["mean"] / R1_TOTAL,
                   "at_5050_per_n": ITEMS / 8, "at_5050_over_sustained_0.02": ITEMS / 8 / R1_SUST_002, "at_5050_over_sustained_0.01": ITEMS / 8 / R1_SUST_001}
# (4) z(exhaust) for a non-replayed seed: self steps ~ Bin(12000, 1/2)
sd_steps = math.sqrt(12000 * 0.25)
out["pool_z"] = {"z_exhaust_s2_r2_pool_444k": (M / 64 - 6000) / sd_steps, "z_exhaust_s2_r1_pool_440k": (440000 / 64 - 6000) / sd_steps,
                 "s2_items_mean_sd": [64 * 6000, 64 * sd_steps], "s2_new_window_per_n_mean": 64 * 6000 * Fr(2, 3) / 4}
out["pool_z"]["s2_new_window_per_n_mean"] = float(out["pool_z"]["s2_new_window_per_n_mean"])


# (5) exact binomial OCs
def logpmf(k, n, p):
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def cdf(c, n, p):
    return sum(math.exp(logpmf(k, n, p)) for k in range(0, c + 1))


Nw_new = round(5000 * 2 / 3); Nw_old = 5000 - Nw_new
la = {"pooled_5000_le50": {str(p): cdf(50, 5000, 1 - p) for p in (0.9995, 0.995, 0.99, 0.985)},
      "window_new_%d_le%d" % (Nw_new, Nw_new // 100): {str(p): cdf(Nw_new // 100, Nw_new, 1 - p) for p in (0.9995, 0.995, 0.99, 0.985)},
      "window_old_%d_le%d" % (Nw_old, Nw_old // 100): {str(p): cdf(Nw_old // 100, Nw_old, 1 - p) for p in (0.9995, 0.995, 0.99, 0.985)},
      "window_split_sd": math.sqrt(5000 * (2 / 3) * (1 / 3))}
ret = {"one_n_err_0.001_le10": cdf(10, 1000, 0.001), "all4_err_0.001": cdf(10, 1000, 0.001) ** 4, "one_n_err_0.02_le10": cdf(10, 1000, 0.02),
       "K24_n1_true_err_0.010": cdf(10, 1000, 0.010), "K24_n1_true_err_0.005": cdf(10, 1000, 0.005), "K24_n1_true_err_0.015": cdf(10, 1000, 0.015)}
out["label_audit2_oc"] = la; out["retain_k24_binomial"] = ret
json.dump(out, open(os.path.join(HERE, "v1_support_d10.json"), "w", newline="\n"), indent=1, default=str)
print("S1 D10", {t: (round(v["P"], 4) if v["P"] is not None else None, round(v["deadline_cost_lossw"], 4)) for t, v in out["S1_D10"].items()})
print("S1 50/50", {t: (round(v["P"], 4) if v["P"] is not None else None, round(v["deadline_cost_lossw"], 4)) for t, v in out["S1_5050_regression"].items()})
print("GS3", out["GS3_D10"], "regression", out["GS3_5050_regression"]["target_exact"])
print("Kb D10", out["Kb_D10"]); print("Kb 5050", out["Kb_5050"]); print("Kb r1", out["Kb_round1"])
print("exposure", json.dumps(out["exposure"], default=str)); print("z", out["pool_z"]); print("LA", la); print("RET", ret)
