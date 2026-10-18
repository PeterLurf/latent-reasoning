"""SD2_v1 check 5 ($0, arithmetic; read-only). Label-noise numbers that move with D10's 2 : 1 window mix (PROOFS p3_noise_gate's forms, not imported).
Inputs: s215_ftself_s1/eval_chainlabel_fresh.json (FT-SELF chain, 1,000 fresh items per n) [MEASURED].
  w_W = kept-wrong share among kept in a window; pool share under D10 w_pool = (2/3) w_W2 + (1/3) w_W1.
  Direct-w kappa threshold at n 16 (PROOFS S2(b)): (1 - P(16)) / (16 w); point and an upper-w variant (each window's one-sided 95 % CP upper, mixed).
  Expected kept-wrong among LABEL-AUDIT-2's 5,000; cumulative pattern-class mass bound over the lineage 2(1 - pi_1) + 2(1 - pi_2) (PROOFS A5)."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"
b = json.load(open(R + "s215_ftself_s1/eval_chainlabel_fresh.json"))["by_n"]


def cdf(c, n, p):
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)) for k in range(c + 1))


def cp_up(k, n, a=0.05):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if cdf(k, n, mid) > a else (lo, mid)
    return hi


kw = {n: round(b[str(n)]["n_kept"] * (1 - b[str(n)]["prec_kept"])) for n in range(9, 17)}; kept = {n: b[str(n)]["n_kept"] for n in range(9, 17)}
W2 = (sum(kw[n] for n in range(13, 17)), sum(kept[n] for n in range(13, 17))); W1 = (sum(kw[n] for n in range(9, 13)), sum(kept[n] for n in range(9, 13)))
w2, w1 = W2[0] / W2[1], W1[0] / W1[1]; w2u, w1u = cp_up(*W2), cp_up(*W1)
P16 = b["16"]["precision_strict"]; e16 = (1 - P16) / 16
wD = 2 / 3 * w2 + 1 / 3 * w1; wDu = 2 / 3 * w2u + 1 / 3 * w1u; w50 = (W2[0] + W1[0]) / (W2[1] + W1[1])
out = {"kept_wrong_by_n": kw, "W2": W2, "W1": W1, "w_W2": w2, "w_W1": w1, "w_W2_up95": w2u, "w_W1_up95": w1u, "w_pool_D10": wD, "w_pool_D10_up_mixed": wDu,
       "w_pool_5050": w50, "kappa_direct_w_W2": [e16 / w2, e16 / w2u], "kappa_direct_w_pool_D10": [e16 / wD, e16 / wDu], "kappa_direct_w_pool_5050_point": e16 / w50,
       "expected_kept_wrong_in_5000_D10": 5000 * wD, "pi_pool_D10_point": 1 - wD,
       "class_mass_cumulative_point": 2 * (2 / 5000) + 2 * w2, "class_mass_round2_point": 2 * w2, "class_mass_round2_up": 2 * w2u}
json.dump(out, open(os.path.join(HERE, "v1_noise_d10.json"), "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
