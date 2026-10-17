"""PROOFS_SELFDEPTH check 2 ($0, CPU, exact rationals): an independent recomputation of Prop S1 against vbank.vb_loss's actual windows
(R1: t in [n_i, K_i]; R2: t in 1..K_i; per-item K_i = min(n_i + U{0..kextra}, kmax) from draw(); the rollout runs to Kb = max K_i but passes
t > K_i carry no loss), with
  (a) the draft's law (plain n ~ U{1..8}),
  (b) the trainer's ACTUAL plain law: train_items redraws n on every excluded draw, so P(n) ~ (1 - h_n), h = item_space.json excl_hit_rate
      (n 1: 0.8852, n 2: 0.0284, n 3: 0.0002),
  (c) stage A of the self curriculum (self n in {9, 10}), which G-S3's '0.500 +- 0.01' must not pool with stage B,
  (d) the loss-weighted share (vb_loss averages R2 over t per item: weight 1/K_i per R2 pass),
plus the fresh-once pool budget under the self CurGate (stage A n {9,10} -> stage B n {9..12}, cap 4k) and E[Kb] for the time model."""
import json, math, sys
from fractions import Fraction as Fr

H = {1: Fr(8852, 10000), 2: Fr(284, 10000), 3: Fr(2, 10000)}


def plain_law(kind):
    if kind == "uniform":
        return {n: Fr(1, 8) for n in range(1, 9)}
    w = {n: 1 - H.get(n, 0) for n in range(1, 9)}; z = sum(w.values())
    return {n: v / z for n, v in w.items()}


def mixture(p_self, self_law, plain, kextra=4, kmax_plain=12, kmax_self=16):
    law = []
    for n, pn in plain.items():
        for u in range(kextra + 1):
            law.append(((1 - p_self) * pn / (kextra + 1), n, min(n + u, kmax_plain)))
    for n, pn in self_law.items():
        for u in range(kextra + 1):
            law.append((p_self * pn / (kextra + 1), n, min(n + u, kmax_self)))
    res = {}
    for ts in range(9, 18):
        tot = sum(w * max(0, K - ts + 1) for w, n, K in law); unf = sum(w * max(0, min(n, K) - ts + 1) for w, n, K in law)
        totw = sum(w * Fr(max(0, K - ts + 1), K) for w, n, K in law); unfw = sum(w * Fr(max(0, min(n, K) - ts + 1), K) for w, n, K in law)
        res[ts] = {"P": float(unf / tot) if tot else None, "E_unf": float(unf), "P_lossw": float(unfw / totw) if totw else None, "E_unf_lossw": float(unfw)}
    return res


U = lambda ns: {n: Fr(1, len(ns)) for n in ns}  # noqa: E731
cases = {"R1 stage B, draft law (uniform plain)": (Fr(1, 2), U(range(9, 13)), "uniform", 16),
         "R1 stage B, trainer law (exclusion-weighted plain)": (Fr(1, 2), U(range(9, 13)), "excl", 16),
         "R1 stage A (self n 9-10), trainer law": (Fr(1, 2), U(range(9, 11)), "excl", 16),
         "R2 (self 9-16, kmax 20), trainer law": (Fr(1, 2), U(range(9, 17)), "excl", 20)}
out = {"plain_law_excl": {n: float(v) for n, v in plain_law("excl").items()}, "support": {}}
for name, (p, sl, pk, km) in cases.items():
    r = mixture(p, sl, plain_law(pk), kmax_self=km); out["support"][name] = r
    print(name)
    print("   t*      " + " ".join("%7d" % t for t in r))
    for key in ("P", "E_unf", "P_lossw", "E_unf_lossw"):
        print("   %-8s " % key + " ".join("%7.4f" % v[key] if v[key] is not None else "    n/a" for v in r.values()))

# E[Kb] (batch max of per-item K, bs 64) for the launch-bound time model
def e_kb(law_nk, bs=64):
    ks = sorted({K for _, _, K in law_nk}); cdf = {}
    for k in ks:
        cdf[k] = sum(w for w, n, K in law_nk if K <= k)
    e = 0.0; prev = 0.0
    for k in ks:
        pk = float(cdf[k]) ** bs; e += k * (pk - prev); prev = pk
    return e
def items(law, kextra=4, kmax=12):
    return [(pn / (kextra + 1), n, min(n + u, kmax)) for n, pn in law.items() for u in range(kextra + 1)]
out["E_Kb"] = {"plain_uniform": e_kb(items(plain_law("uniform"))), "plain_excl": e_kb(items(plain_law("excl"))),
               "self_stageB": e_kb(items(U(range(9, 13)), kmax=16)), "self_stageA": e_kb(items(U(range(9, 11)), kmax=16))}
print("E[Kb] (bs 64):", {k: round(v, 3) for k, v in out["E_Kb"].items()})

# fresh-once pool budget, round 1: 12k steps, coin p_self 1/2, bs 64; stage A (n 9,10) lasts s_A GLOBAL steps (CurGate counts the step index
# it is given; cap 4000), then stage B uniform 9..12. Self items drawn at n 9 (and 10): 64 * [S_A / 2 + S_B / 4], S_A ~ Bin(s_A, 1/2), S_B ~ Bin(12000 - s_A, 1/2).
def need_n9(sA, z=0.0):
    mean = 64 * (sA * 0.5 / 2 + (12000 - sA) * 0.5 / 4)
    var = 64 ** 2 * (sA * 0.25 / 4 + (12000 - sA) * 0.25 / 16)
    return mean + z * math.sqrt(var)
bud = {}
for sA in (2000, 3000, 4000):
    bud[sA] = {"mean": need_n9(sA), "q999": need_n9(sA, 3.09)}
# if the self gate counted SELF steps instead (cap 4000 self steps ~ 8000 global)
bud["self-step cap 4000"] = {"mean": 64 * (4000 / 2 + 2000 / 4)}
out["pool_n9_need"] = bud
print("items needed at n 9 (= n 10) vs 120,000 kept:", json.dumps(bud))
# n 11/12 need at the cap
print("items needed at n 11 (= n 12) at the cap:", 64 * (12000 - 4000) * 0.5 / 4)
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1, default=str)
