"""SD2_v1 check 3 ($0, arithmetic only). OC under Theory's D7 (headline = the red team's shaded priors, recomputed after D9 / D10), with:
  - lineage noise in P(pass | no effect) (RT M3; sigma 0.06-0.12 on M2, v1_panels.json lineage_noise_M2);
  - K'24-STABLE modelled as dose-coupled to ABSORB-2 (PROOFS #42): P(K'24 | ABSORB-2) [ASSUMED], bounded over [0.45, 0.65];
  - FT-DIVERGED in the OC (RT m5) without double counting: its a_24 < 0.97 clause implies not K'24-STABLE, so only the a(n) < 0.99 clause is a factor
    given K'24-STABLE;
  - D10's dose (new window 1.38x round 1's sustained CE < 0.02 point, 0.86x the < 0.01 point; v1_support_d10.json) -> P(ABSORB-2), and the halved
    old-window rehearsal -> RETAIN;
  - LIT-C2-2 (D9) as its own row, not inside SD2-C2;
  - the pre-stated x1.5 GROW shift as a sensitivity row only; PROOFS' re-shade (0.070) and RT's shaded OC (0.0636) reproduced exactly as cross-checks.
Every prior is [ASSUMED]; the arithmetic is [DERIVED]."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HS = ("H_shift", "H_grow", "H_full", "SHRINK")


def oc(P_ABS, kept, pri, P_DBS, P_MG, P_KB, P_LIT, keepmem, s2):
    k = 1.0
    for v in kept.values():
        k *= v
    s = P_ABS * k
    o = {"kept_given_ABSORB": k, "P_SD2_SELF": s, "P_DBS2": P_ABS * sum(pri[h] * P_DBS[h] for h in HS), "P_MG2": P_ABS * sum(pri[h] * P_MG[h] for h in HS),
         "P_KB2_ge_0.25": P_ABS * sum(pri[h] * P_KB[h] for h in HS), "P_SD2_C2": s * sum(pri[h] * P_DBS[h] for h in HS),
         "P_SD2_C2_given": {h: s * P_DBS[h] for h in HS}, "P_LIT_C2_2_with_SD2_SELF": s * sum(pri[h] * P_LIT[h] for h in HS)}
    o["P_SD2_C2_MEM"] = o["P_SD2_C2"] * keepmem
    o["P_SD2_C2_and_LIT"] = s * sum(pri[h] * min(P_DBS[h], P_LIT[h]) for h in HS)          # LIT passes essentially only under H_full (FULL implies DBS)
    if s2 is not None:
        reach = s2["competent"] * s2["gate1"] * s2["r1_ok"] * s2["gate2"]
        o["P_x2"] = sum(pri[h] * (s * P_DBS[h]) * (reach * min(1.0, s * P_DBS[h] * s2["corr"])) for h in HS)
        o["P_x2_given_s2_competent"] = o["P_x2"] / s2["competent"]; o["P_s2_reaches_R2"] = reach
        o["P_x2_given_Hshift_both"] = (s * P_DBS["H_shift"]) * reach * min(1.0, s * P_DBS["H_shift"] * s2["corr"])
    return o


RT_PRI = {"H_shift": 0.60, "H_grow": 0.30, "H_full": 0.02, "SHRINK": 0.08}
pr1 = {"H_shift": 0.65, "H_grow": 0.25, "H_full": 0.03, "SHRINK": 0.07}
SHIFT15 = {"H_grow": 0.375, **{h: pr1[h] / 0.75 * 0.625 for h in ("H_shift", "H_full", "SHRINK")}}   # v1 §9's pre-stated x1.5 (cap 0.8)
DBS = {"H_shift": 0.12, "H_grow": 0.55, "H_full": 0.90, "SHRINK": 0.0}; MG = {"H_shift": 0.18, "H_grow": 0.70, "H_full": 1.0, "SHRINK": 0.0}
KB = {"H_shift": 0.10, "H_grow": 0.40, "H_full": 0.85, "SHRINK": 0.0}; LIT = {"H_shift": 0.01, "H_grow": 0.05, "H_full": 0.85, "SHRINK": 0.0}
S2 = {"competent": 0.75, "gate1": 0.95, "r1_ok": 0.70, "gate2": 0.98, "corr": 1.2}
P_ABS = 0.65
KEPT = {"LENGTH_FREE": 0.90, "VB_C1": 0.90, "RETAIN": 0.93, "K24_given_ABSORB": 0.55, "notDIVERGED_given_K24": 0.97}
out = {"assumed": {"priors_given_ABSORB": RT_PRI, "P_DBS": DBS, "P_MG": MG, "P_KB": KB, "P_LIT": LIT, "P_ABSORB2": P_ABS, "kept": KEPT, "KEEPS_SELF_STAB_2": 0.20, "s2": S2}}
out["headline"] = oc(P_ABS, KEPT, RT_PRI, DBS, MG, KB, LIT, 0.20, S2)
# joint ABSORB-2 and K'24 (the dropped independence): conditional model vs independence at the same marginal (P(K'24 | not ABSORB-2) 0.65 [ASSUMED])
pk_marg = P_ABS * 0.55 + (1 - P_ABS) * 0.65
out["k24_absorb_joint"] = {"conditional_model": P_ABS * 0.55, "independence_same_marginal": P_ABS * pk_marg, "marginal_P_K24": pk_marg,
                           "frechet_bounds_given_marginals": [max(0.0, P_ABS + pk_marg - 1), min(P_ABS, pk_marg)],
                           "bound_over_conditional_0.45_0.65": [P_ABS * 0.45, P_ABS * 0.65],
                           "P_FT_DIVERGED_given_ABSORB_approx": 1 - (0.55 * 0.97 + 0.45 * 0.8 * 0.97)}
sens = {}
sens["x1.5_GROW_shift_priors"] = oc(P_ABS, KEPT, SHIFT15, DBS, MG, KB, LIT, 0.20, S2)
for pa in (0.60, 0.70):
    sens["P_ABSORB2_%.2f" % pa] = oc(pa, KEPT, RT_PRI, DBS, MG, KB, LIT, 0.20, S2)
for k24 in (0.45, 0.65):
    sens["K24_given_ABSORB_%.2f" % k24] = oc(P_ABS, {**KEPT, "K24_given_ABSORB": k24}, RT_PRI, DBS, MG, KB, LIT, 0.20, S2)
for ret in (0.95, 0.90):
    sens["RETAIN_%.2f" % ret] = oc(P_ABS, {**KEPT, "RETAIN": ret}, RT_PRI, DBS, MG, KB, LIT, 0.20, S2)
for nm, (ds, dg) in (("sigma_0.06", (0.084, 0.60)), ("sigma_0.12", (0.245, 0.50))):
    sens[nm] = oc(P_ABS, KEPT, RT_PRI, {**DBS, "H_shift": ds, "H_grow": dg}, MG, KB, LIT, 0.20, S2)
sens["range_lo"] = oc(0.55, {**KEPT, "K24_given_ABSORB": 0.45, "RETAIN": 0.90}, RT_PRI, {**DBS, "H_shift": 0.18, "H_grow": 0.45}, MG, KB, LIT, 0.20,
                      {**S2, "competent": 0.70, "r1_ok": 0.60})
sens["range_hi"] = oc(0.70, {**KEPT, "K24_given_ABSORB": 0.65, "RETAIN": 0.95}, SHIFT15, {**DBS, "H_shift": 0.08, "H_grow": 0.60}, MG, KB, LIT, 0.20,
                      {**S2, "competent": 0.80, "r1_ok": 0.80})
out["sensitivity"] = sens
# cross-checks (exact reproductions)
rt = oc(0.60, {"LF": 0.9, "VB": 0.9, "K24": 0.60, "RET": 0.95, "notDIV": 0.90}, RT_PRI, DBS, MG, KB, LIT, 0.20,
        {"competent": 0.75, "gate1": 1.0, "r1_ok": 0.70, "gate2": 1.0, "corr": 1.2})
proofs = oc(0.60, {"LF": 0.9, "VB": 0.9, "K24": 0.55, "RET": 0.95}, SHIFT15, {"H_shift": 0.05, "H_grow": 0.60, "H_full": 0.95, "SHRINK": 0.0},
            {"H_shift": 0.10, "H_grow": 0.75, "H_full": 1.0, "SHRINK": 0.0}, {"H_shift": 0.05, "H_grow": 0.45, "H_full": 0.90, "SHRINK": 0.0}, LIT, 0.25, None)
assert abs(rt["P_SD2_C2"] - 0.06357609) < 1e-7 and abs(rt["P_x2"] - 0.0045265579) < 1e-8, (rt["P_SD2_C2"], rt["P_x2"])
assert abs(proofs["P_SD2_C2"] - 0.0700437375) < 1e-9, proofs["P_SD2_C2"]
out["crosscheck_RT_shaded"] = rt; out["crosscheck_PROOFS_reshade"] = proofs
# P(pass | no effect) with lineage noise, per reading (sigma 0.06 / 0.09 / 0.12; v1_panels.json)
pn = json.load(open(os.path.join(HERE, "v1_panels.json")))["lineage_noise_M2"]
s = out["headline"]["P_SD2_SELF"]
out["no_effect_rows"] = {sg: {"DBS2_given_Hshift": pn[sg]["H_shift"]["P_ge_0.80"], "MG2_given_Hshift": pn[sg]["H_shift"]["P_ge_0.785"],
                              "SD2_C2_given_Hshift": s * pn[sg]["H_shift"]["P_ge_0.80"], "DBS2_given_Hprop": pn[sg]["H_prop"]["P_ge_0.80"],
                              "DBS2_given_Hadd": pn[sg]["H_add_constant_growth"]["P_ge_0.80"], "SD2_C2_given_Hprop": s * pn[sg]["H_prop"]["P_ge_0.80"],
                              "MG_mislabel_Hshift": pn[sg]["MG_mislabel_Hshift"], "MG_mislabel_Hprop": pn[sg]["MG_mislabel_Hprop"]} for sg in ("0.06", "0.09", "0.12")}
json.dump(out, open(os.path.join(HERE, "v1_oc.json"), "w", newline="\n"), indent=1)


def short(o):
    return {k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()}) for k, v in o.items()}


print("HEADLINE", short(out["headline"])); print("joint", out["k24_absorb_joint"])
for k, v in sens.items():
    print(k, {kk: round(v[kk], 4) for kk in ("P_SD2_SELF", "P_SD2_C2", "P_x2", "P_LIT_C2_2_with_SD2_SELF")})
print("RT", short(rt)); print("PROOFS", short(proofs)); print("no-effect", json.dumps(out["no_effect_rows"], indent=0))
