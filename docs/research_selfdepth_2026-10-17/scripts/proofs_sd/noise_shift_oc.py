"""PROOFS_SELFDEPTH check 3 ($0, CPU; reads s213_d3t_s1/eval_vb.json read-only):
 (1) Prop S2: the post-filter bound, the kept-wrong share w (corrected: among KEPT items), the avalanche threshold on kappa (with / without 1/P),
     the two-round growth factor (one factor per ROUND with that round's max n, vs the draft's sum over every n 9..16), the per-transition units;
 (2) Prop S4 / H_shift: the translated curve, THINK-GAIN-BEYOND, the sampling SE of the 4-n mean, and STOP-LOC (draft §5 definition) on the base
     and under exact H_shift (base acc_k translated by +4);
 (3) the §6 OC arithmetic from its stated priors."""
import json, math, sys
d = json.load(open("D:/Latent/boxes/runs_fsq/s213_d3t_s1/eval_vb.json"))
A2 = {int(n): v for n, v in d["acc_n2"].items()}; AK = {int(k): {int(n): v for n, v in r.items()} for k, r in d["acc_k"].items()}
out = {}
# ---------------- (1) S2
P_fresh = {9: 1.0, 10: 0.996, 11: 0.988, 12: 0.976, 13: 0.972, 14: 0.952, 15: 0.94, 16: 0.917, 17: 0.892, 18: 0.839, 19: 0.808, 20: 0.781}
phi_u = 0.00965; phi_m = 6 / 1224
s2 = {"pi_lower_bound": {}, "kappa_threshold": {}}
for n, P in P_fresh.items():
    w_exact = phi_u * (1 - P) / (P + phi_u * (1 - P)); s2["pi_lower_bound"][n] = {"P": P, "w_kept_wrong_share": w_exact, "w_draft(phi(1-P))": phi_u * (1 - P),
                                                                                  "pi_lb": 1 - w_exact}
for n in (12, 16, 20):
    P = P_fresh[n]
    s2["kappa_threshold"][n] = {"draft 1/(phi n)": 1 / (0.010 * n), "corrected P/(phi n), phi 0.010": P / (0.010 * n), "corrected, phi CP95 0.00965": P / (phi_u * n),
                                "per-transition kappa' = kappa*n, corrected": P / phi_u}
# two rounds of W 4: factor prod_q (1 + kappa phi n_q / P_q), n_1 = 12, n_2 = 16 (kappa = 1)
for phi, tag in ((0.010, "phi 0.010"), (phi_m, "phi 0.0049")):
    tight = (1 + phi * 12 / P_fresh[12]) * (1 + phi * 16 / P_fresh[16]); draft = math.exp(phi * sum(range(9, 17)))
    s2["two_round_factor_" + tag] = {"per-round (n 12, 16), with 1/P": tight, "exp(phi*28)": math.exp(phi * 28), "draft exp(phi*100)": draft,
                                      "S1-C four +1 rounds n 9..12 then R2": math.exp(phi * (9 + 10 + 11 + 12))}
out["S2"] = s2
print(json.dumps(s2["kappa_threshold"], indent=1)); print({k: v for k, v in s2.items() if k.startswith("two")})
print("pi lower bounds (phi CP95):", {n: round(v["pi_lb"], 5) for n, v in s2["pi_lower_bound"].items()})

# ---------------- (2) H_shift
base = [A2[n] for n in (9, 10, 11, 12)]; k8 = [AK[8][n] for n in (9, 10, 11, 12)]
se = math.sqrt(sum(p * (1 - p) / 1000 for p in base)) / 4
tgb = sum(a - b for a, b in zip(base, k8)) / 4
out["H_shift"] = {"curve_n13_16": base, "mean": sum(base) / 4, "acc_k8_base": k8, "think_gain_beyond": tgb, "se_mean_4n": se,
                  "z_to_0.63": (0.63 - sum(base) / 4) / se, "z_to_0.80": (0.80 - sum(base) / 4) / se}
print("H_shift", out["H_shift"])

def stop_loc(acc_n2, acc_k, krange, nmax, thr=0.02, read=0.18, need=2):
    G = {}
    for k in krange:
        ns = [n for n in range(k + 1, min(k + 4, nmax) + 1) if acc_n2.get(n) is not None]
        rd = [n for n in ns if acc_n2[n] >= read]
        G[k] = {"window": ns, "readable": rd, "G_readable": (sum(acc_n2[n] - acc_k[k][n] for n in rd) / len(rd)) if rd else None,
                "G_all": sum(acc_n2[n] - acc_k[k][n] for n in ns) / len(ns), "readable_ok": len(rd) >= need}
    stop = next((k for k in krange if G[k]["readable_ok"] and G[k]["G_readable"] <= thr), None)
    return stop, G
# base: krange 10..14 over n <= 16 (as the table's 'base D3-T' column)
stop_b, G_b = stop_loc(A2, AK, range(10, 15), 16)
# the base's own analogue of round 1's range (k 6..10 = B-2..B+2 with B 8), to see where its stop reads
stop_b6, G_b6 = stop_loc(A2, AK, range(6, 11), 12)
# exact H_shift for round 1 (B 12): acc'(n) = base acc(n - 4) for n 13..16, acc'(K' k)(n) = base acc(K' k - 4)(n - 4); n 9..12 in support:
# ABSORB-level acc 1.0 at every K' >= n, 0.10 (chance) at K' < n (a stylised in-support panel; it only enters G(k) for k <= 11)
A2s = {n: (1.0 if n <= 12 else A2[n - 4]) for n in range(9, 17)}
AKs = {k: {n: ((1.0 if k >= n else 0.10) if n <= 12 else AK[k - 4][n - 4]) for n in range(9, 17)} for k in range(10, 15)}
stop_s, G_s = stop_loc(A2s, AKs, range(10, 15), 16)
out["STOP_LOC"] = {"base_k10_14": {"stop": stop_b, "G": G_b}, "base_k6_10_nle12": {"stop": stop_b6, "G": G_b6}, "H_shift_round1": {"stop": stop_s, "G": G_s}}
print("STOP-LOC base (k 10..14):", stop_b, {k: (v["readable"], v["G_readable"], round(v["G_all"], 3)) for k, v in G_b.items()})
print("STOP-LOC base analogue (k 6..10, n<=12):", stop_b6, {k: (v["readable"], v["G_readable"], round(v["G_all"], 3)) for k, v in G_b6.items()})
print("STOP-LOC exact H_shift round 1:", stop_s, {k: (v["readable"], None if v["G_readable"] is None else round(v["G_readable"], 3), round(v["G_all"], 3)) for k, v in G_s.items()})
gain10 = sum(A2[n] - AK[10][n] for n in (11, 12, 13, 14)) / 4
out["ACC_GAIN_AFTER_10_base"] = gain10; print("ACC-GAIN-AFTER-10 base", round(gain10, 4))

# ---------------- (3) OC arithmetic (draft §6 priors)
pA = 0.65; H = {"shift": 0.55, "grow": 0.30, "full": 0.10, "shrink": 0.05}
mg = {"shift": 0.07, "grow": 0.85, "full": 1.0, "shrink": 0.0}; dbs = {"shift": 0.01, "grow": 0.25, "full": 0.9, "shrink": 0.0}
sm = {"shift": 0.95, "grow": 0.95, "full": 1.0, "shrink": 0.95}
kept = 0.9 * 0.9 * 0.85; ks = 0.7
oc = {"kept": kept, "SD_SELF": pA * kept, "MARGIN_GROWS": pA * sum(H[h] * mg[h] for h in H), "DBS": pA * sum(H[h] * dbs[h] for h in H),
      "STOP_MOVED": pA * sum(H[h] * sm[h] for h in H) + (1 - pA) * 0.3}
oc["SD_C2"] = oc["DBS"] * kept; oc["SD_C2_MEM"] = oc["SD_C2"] * ks
oc["SD_C2_given"] = {h: dbs[h] * kept for h in H}; oc["SD_C2_MEM_given"] = {h: dbs[h] * kept * ks for h in H}
out["OC"] = oc; print("OC", json.dumps(oc, indent=0))
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1, default=str)
