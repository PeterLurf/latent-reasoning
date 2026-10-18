"""PROOFS_SD2 check 2 ($0, CPU, exact rationals; independent rewrite, not a copy of support_check).
Round-2 law: p_self 1/2; self n ~ U{9..16}, K = min(n + U{0..4}, 20); plain n ~ trainer's exclusion-weighted law on 1..8 (h_1 .8852, h_2 .0284, h_3 .0002),
K = min(n + U{0..4}, 12). An R2 pass t (1 <= t <= K) is 'R' (real reduction) iff t <= n.
P(R | t >= t*) = E[#{t in [t*, min(n,K)]}] / E[#{t in [t*, K]}]; loss-weighted variant weights each pass by 1/K (vb_loss averages R2 over t per item).
Deadline cost at t_d = E[#{R passes t >= t_d} / K] (share of an item's R2 term), over the whole mixture.
Also: G-S3 sampling SE of the empirical ratio at t* 13 for N self items (delta method), and the seed-1 replay consequence (exact self-step count)."""
import json, math, os, sys
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
h = {1: Fr(8852, 10000), 2: Fr(284, 10000), 3: Fr(2, 10000)}
w = {n: 1 - h.get(n, Fr(0)) for n in range(1, 9)}; Z = sum(w.values()); plain = {n: w[n] / Z for n in w}
selfl = {n: Fr(1, 8) for n in range(9, 17)}
items = []                                                     # (prob, n, K)
for n, p in plain.items():
    for u in range(5):
        items.append((Fr(1, 2) * p * Fr(1, 5), n, min(n + u, 12)))
for n, p in selfl.items():
    for u in range(5):
        items.append((Fr(1, 2) * p * Fr(1, 5), n, min(n + u, 20)))
assert sum(q for q, _, _ in items) == 1
res = {}
for ts in range(9, 18):
    num = sum(q * max(0, min(n, K) - ts + 1) for q, n, K in items); den = sum(q * max(0, K - ts + 1) for q, n, K in items)
    numw = sum(q * Fr(max(0, min(n, K) - ts + 1), K) for q, n, K in items); denw = sum(q * Fr(max(0, K - ts + 1), K) for q, n, K in items)
    res[ts] = {"P": float(num / den) if den else None, "P_exact": str(num / den) if den else None, "P_lossw": float(numw / denw) if denw else None,
               "deadline_cost_lossw": float(numw), "E_R_passes": float(num)}
# G-S3: SE of the empirical ratio sum X / sum Y at t* 13 over N self items (plain items contribute 0 to both at t >= 13)
X = lambda n, u: max(0, n - 12); Y = lambda n, u: max(0, min(n + u, 20) - 12)  # noqa: E731
pr = [(Fr(1, 40), n, u) for n in range(9, 17) for u in range(5)]
EX = sum(p * X(n, u) for p, n, u in pr); EY = sum(p * Y(n, u) for p, n, u in pr); R = EX / EY
V = sum(p * (X(n, u) - R * Y(n, u)) ** 2 for p, n, u in pr)
se = {N: math.sqrt(float(V) / N) / float(EY) for N in (20000, 192000, 383488)}
# per-micro-batch rollout length (vb_loss: Kb = max K over the micro-batch): P(Kb = 20) and E[Kb] for chunk sizes 64 / 32 / 16
Kd = {}
for p, n, u in pr:
    Kd[min(n + u, 20)] = Kd.get(min(n + u, 20), 0) + p
ks = sorted(Kd); cdf = {}; c = Fr(0)
for k in ks:
    c += Kd[k]; cdf[k] = c
kb = {}
for b in (64, 32, 16):
    e = 0.0; prev = 0.0
    for k in ks:
        pk = float(cdf[k]) ** b; e += k * (pk - prev); prev = pk
    kb[b] = {"P_Kb_20": 1 - float(cdf[19]) ** b, "E_Kb": e}
# round 1 analogue (self n U{9..12}, kmax 16, micro 2 -> 32 per chunk)
Kd1 = {}
for n in range(9, 13):
    for u in range(5):
        Kd1[n + u] = Kd1.get(n + u, 0) + Fr(1, 20)
c = Fr(0); cdf1 = {}
for k in sorted(Kd1):
    c += Kd1[k]; cdf1[k] = c
kb1 = {}
for b in (64, 32):
    e = 0.0; prev = 0.0
    for k in sorted(Kd1):
        pk = float(cdf1[k]) ** b; e += k * (pk - prev); prev = pk
    kb1[b] = {"P_Kb_16": 1 - float(cdf1[15]) ** b, "E_Kb": e}
out = {"table": res, "GS3_target_t13": float(R), "GS3_target_exact": str(R), "GS3_se_by_N": se,
       "GS3_P_outside_pm0.01_N20000": 2 * (1 - 0.5 * (1 + math.erf(0.01 / se[20000] / math.sqrt(2)))),
       "kb_round2_by_chunk": kb, "kb_round1_by_chunk": kb1}
json.dump(out, open(os.path.join(HERE, "p2_support_s1.json"), "w", newline="\n"), indent=1)
print(json.dumps({t: (round(v["P"], 4) if v["P"] is not None else None, round(v["deadline_cost_lossw"], 4)) for t, v in res.items()}))
print("G-S3", out["GS3_target_exact"], round(out["GS3_target_t13"], 5), {k: round(v, 5) for k, v in se.items()}, out["GS3_P_outside_pm0.01_N20000"])
print("Kb r2", kb, "Kb r1", kb1)
