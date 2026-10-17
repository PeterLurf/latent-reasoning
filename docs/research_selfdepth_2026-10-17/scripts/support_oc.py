"""SELF-DEPTH-CHAIN check 3 ($0, CPU): exact pass-level support under the round mixture (Prop S1), the per-round precision-gate OC
(binomial), the H_shift per-n prediction, and the cost / memory projection (D4 v1 §3(v) model, T-scaled; ASSUMED)."""
import json, sys, math
from fractions import Fraction as Fr
import math as _m
class binom:
    @staticmethod
    def cdf(k, n, p):
        """exact binomial CDF via log-space terms (scipy absent)."""
        tot = 0.0
        for i in range(0, k + 1):
            tot += _m.exp(_m.lgamma(n + 1) - _m.lgamma(i + 1) - _m.lgamma(n - i + 1) + (i * _m.log(p) if p > 0 else (0 if i == 0 else -1e300)) + (n - i) * _m.log1p(-p))
        return min(1.0, tot)

def mixture(p_self, self_ns, kmax, kextra=4, plain_ns=range(1, 9)):
    """item law: plain n ~ U{1..8} (prob 1 - p_self), self n ~ U(self_ns) (prob p_self); K = min(n + U{0..kextra}, kmax).
    Training passes t = 1..K; pass t is unfinished (an R pass) iff t <= n. Returns exact P(unfinished | t >= t*) (pass-weighted) and
    E[# unfinished passes with t >= t_d] per item (the cost of a deadline at t_d; D4's 'weak pressure')."""
    law = []
    for n in plain_ns:
        for u in range(kextra + 1):
            law.append((Fr(1 - p_self) / len(plain_ns) / (kextra + 1), n, min(n + u, kmax)))
    for n in self_ns:
        for u in range(kextra + 1):
            law.append((Fr(p_self) / len(self_ns) / (kextra + 1), n, min(n + u, kmax)))
    res = {}
    for ts in range(9, 21):
        tot = sum(w * max(0, K - ts + 1) for w, n, K in law); unf = sum(w * max(0, min(n, K) - ts + 1) for w, n, K in law)
        res[ts] = {"P_unfinished_given_t_ge": float(unf / tot) if tot else None, "unfinished_passes_per_item_t_ge": float(unf),
                   "max_rho_plus_r": max(n for w, n, K in law)}
    return res

out = {}
designs = {"D3-T base (no self)": (0.0, [9], 12), "R1 window 9-12 (p 0.5)": (0.5, [9, 10, 11, 12], 16),
           "+1 round q=4 (n 9..12 cumulative, newest 50%)": None, "R2 window 13-16 (p 0.5; self 9-16)": (0.5, list(range(9, 17)), 20)}
for name, d in designs.items():
    if d is None:
        continue
    out[name] = mixture(*d)
print("P(unfinished | t >= t*) and unfinished passes per item at t >= t_d")
for name, r in out.items():
    print(name)
    print("  t*   " + " ".join("%6d" % t for t in r))
    print("  P    " + " ".join("%6.3f" % v["P_unfinished_given_t_ge"] if v["P_unfinished_given_t_ge"] is not None else "   n/a" for v in r.values()))
    print("  E#   " + " ".join("%6.3f" % v["unfinished_passes_per_item_t_ge"] for v in r.values()))

# per-round precision gate OC: kept ~ Nk items per n; pass iff wrong <= floor(0.005 * Nk)
gate = {}
for Nk in (1900, 1950, 2000):
    kmax_w = int(0.005 * Nk)
    gate[Nk] = {"k_max_wrong": kmax_w, **{"P_pass_true_pi_%g" % pi: float(binom.cdf(kmax_w, Nk, 1 - pi)) for pi in (0.9999, 0.999, 0.997, 0.995, 0.993, 0.99, 0.98)}}
print("gate OC", json.dumps(gate, indent=0))
# 4-n window: the gate must pass at every n
for pi in (0.999, 0.995, 0.99):
    p1 = binom.cdf(int(0.005 * 1950), 1950, 1 - pi); print("window of 4 n, true pi %.3f: P(all pass) %.4f" % (pi, p1 ** 4))

# H_shift prediction: acc(B + k) = D3-T acc(8 + k), k = 1..4
base = {9: 0.923, 10: 0.603, 11: 0.266, 12: 0.127}; base_k8 = {9: 0.329, 10: 0.133, 11: 0.127, 12: 0.118}
pred = {"H_shift_acc_B_plus_k": [base[8 + k] for k in range(1, 5)], "H_shift_mean": sum(base.values()) / 4,
        "H_shift_think_gain_beyond_B": sum(base[n] - base_k8[n] for n in base) / 4}
print("H_shift", pred)

# cost / memory projection (D4 v1 §3(v) split for D3-T at T ~ 33: GiB = 2 + 0.58 Kb + 0.0258 sum_{t=1..Kb}(t + 1); T-scaled: linear x T/33,
# bank (quadratic) x (T/33)^2 [ASSUMED]); s/step launch-bound ~ Kb (0.407 s at E[Kb] 11.84), T factor 1.0 (launch-bound) .. T/33 (compute-bound) [ASSUMED]
def mem(Kb, T):
    return 2 + 0.58 * Kb * T / 33 + 0.0258 * (T / 33) ** 2 * sum(t + 1 for t in range(1, Kb + 1))
def sstep(Kb, T, micro):
    lo = 0.407 * Kb / 11.84 * micro; hi = lo * T / 33; return lo, hi
cost = {}
for name, Kb, T in (("D3-T plain (check)", 12, 33), ("R1 self batch", 16, 47), ("R1 self, kextra 2", 14, 47), ("R2 self batch", 20, 61), ("R2 self, kextra 2", 18, 61)):
    g = mem(Kb, T); micro = 1 if g <= 21.5 else 2 if (g - 2) / 2 + 2 <= 21.5 else 4
    lo, hi = sstep(Kb, T, micro); cost[name] = {"Kb": Kb, "T": T, "peak_gib_micro1": round(g, 1), "micro": micro, "s_step_lo": round(lo, 3), "s_step_hi": round(hi, 3)}
    print(name, cost[name])
def ft(name_self, steps=12000, p=0.5, over_h=0.4):
    c = cost[name_self]; lo = (p * c["s_step_lo"] + (1 - p) * 0.407) * steps / 3600 + over_h; hi = (p * c["s_step_hi"] + (1 - p) * 0.407) * steps / 3600 + over_h
    return {"hours": (round(lo, 2), round(hi, 2)), "usd_0.18_0.26": (round(lo * 0.18, 2), round(hi * 0.26, 2))}
proj = {"FT-SELF R1 (12k)": ft("R1 self batch"), "FT-SELF R1 kextra2": ft("R1 self, kextra 2"), "FT-CTRL (12k)": {"hours": (round(12000 * 0.407 / 3600 + 0.3, 2),) * 2,
        "usd_0.18_0.26": (round((12000 * 0.407 / 3600 + 0.3) * 0.18, 2), round((12000 * 0.407 / 3600 + 0.3) * 0.26, 2))},
        "FT-SELF R2 (12k)": ft("R2 self batch", over_h=1.0), "FT-SELF R2 kextra2": ft("R2 self, kextra 2", over_h=1.0)}
for k, v in proj.items():
    print(k, v)
json.dump({"support": {k: {str(t): v for t, v in r.items()} for k, r in out.items()}, "gate_oc": gate, "pred": pred, "cost": cost, "proj": proj}, open(sys.argv[1], "w", newline="\n"), indent=1)
