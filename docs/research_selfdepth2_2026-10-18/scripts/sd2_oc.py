"""SD2 OC arithmetic (priors [ASSUMED]; exact arithmetic). Round-1 shaded priors given ABSORB (SELFDEPTH_v1 §6) shifted by the pre-stated §9 rule
(GROW read -> P(H_grow) x 1.5, cap 0.8; the rest renormalised in round-1 proportions). Kept factor = LENGTH-FREE x VB-C1 x K'24-STABLE x RETAIN."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
pr1 = {"H_shift": 0.65, "H_grow": 0.25, "H_full": 0.03, "SHRINK": 0.07}
P_DBS = {"H_shift": 0.05, "H_grow": 0.60, "H_full": 0.95, "SHRINK": 0.0}; P_MG = {"H_shift": 0.10, "H_grow": 0.75, "H_full": 1.0, "SHRINK": 0.0}
P_KB = {"H_shift": 0.05, "H_grow": 0.45, "H_full": 0.90, "SHRINK": 0.0}
P_ABS, LF, VB, K24, RET, KEEPMEM, CORR = 0.75, 0.9, 0.9, 0.70, 0.95, 0.25, 0.85
def priors(g):
    rest = 1 - g; br = 1 - pr1["H_grow"]
    return {"H_grow": g, **{k: pr1[k] / br * rest for k in ("H_shift", "H_full", "SHRINK")}}
def oc(g):
    p = priors(g); kept = LF * VB * K24 * RET; s = P_ABS * kept
    dbs = sum(p[h] * P_DBS[h] for h in p); mg = sum(p[h] * P_MG[h] for h in p); kb = sum(p[h] * P_KB[h] for h in p)
    return {"priors": p, "kept": kept, "P_SD2_SELF": s, "P_DBS_given_ABSORB": dbs, "P_MG": P_ABS * mg, "P_KB": P_ABS * kb, "P_DBS": P_ABS * dbs,
            "P_SD2_C2": s * dbs, "P_SD2_C2_MEM": s * dbs * KEEPMEM, "P_SD2_C2_given": {h: s * P_DBS[h] for h in p},
            "P_C2_both_lineages_indep": sum(p[h] * (s * P_DBS[h]) ** 2 for h in p),
            "P_C2_both_lineages_corr": sum(p[h] * s * (CORR * P_DBS[h]) * P_DBS[h] for h in p)}
out = {"pre_stated_shift": oc(min(0.8, 0.25 * 1.5)), "sens_no_shift": oc(0.25), "sens_Hgrow_0.5": oc(0.5),
       "assumed": {"P_DBS": P_DBS, "P_MG": P_MG, "P_KB": P_KB, "P_ABSORB2": P_ABS, "LENGTH_FREE": LF, "VB_C1": VB, "K24": K24, "RETAIN": RET, "KEEPS_SELF_STAB2": KEEPMEM,
                   "second_lineage_correlation_factor": CORR}}
json.dump(out, open(os.path.join(HERE, "sd2_oc.json"), "w", newline="\n"), indent=1)
for k, v in out.items():
    if k != "assumed":
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else {a: round(b, 4) for a, b in vv.items()}) for kk, vv in v.items()})
