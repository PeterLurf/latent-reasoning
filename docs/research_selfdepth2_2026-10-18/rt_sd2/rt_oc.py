"""RT-SD2 check 2 ($0, arithmetic only): the draft's OC (sd2_oc.py) reproduced, and a red-team shaded OC. Every prior is [ASSUMED].
Shading reasons (REDTEAM_SD2.md): P(ABSORB-2) 0.60 (new-window dose 48k/n vs round 1's 1k-block absorption at ~56k/n, rt_curves / log check);
K'24-STABLE 0.60 (0.990 at the edge, self-specific dip); + P(not FT-DIVERGED) 0.90 (a_24(1) < 0.97 censors every reading); LENGTH-FREE 0.90, VB-C1 0.90,
RETAIN 0.95; priors given ABSORB-2: H_shift 0.60 / H_grow 0.30 / H_full 0.02 / SHRINK 0.08 (growth seen once, H_prop's fit is level-specific, best-of-family base);
P(DBS-2 | H): 0.12 / 0.55 / 0.90 / 0 (lineage-noise sigma 0.06-0.09 on M2: rt_curves lineage_noise_OC). Second lineage: P(s2 base competent) 0.75,
P(R1-s2 meets the round-2 condition) 0.70, correlation factor 1.2 on the conditional DBS pass (same hypothesis, shared recipe)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def oc(P_ABS, kept_parts, pri, P_DBS, P_MG, P_KB, keepmem, s2=None):
    kept = 1.0
    for v in kept_parts.values():
        kept *= v
    s = P_ABS * kept; dbs = sum(pri[h] * P_DBS[h] for h in pri)
    o = {"kept": kept, "P_SD2_SELF": s, "P_DBS2_given_ABSORB": dbs, "P_DBS2": P_ABS * dbs, "P_MG2": P_ABS * sum(pri[h] * P_MG[h] for h in pri),
         "P_KB2": P_ABS * sum(pri[h] * P_KB[h] for h in pri), "P_SD2_C2": s * dbs, "P_SD2_C2_MEM": s * dbs * keepmem,
         "P_SD2_C2_given": {h: s * P_DBS[h] for h in pri}}
    if s2:
        comp, r1ok, corr = s2
        o["P_SD2_C2_x2"] = sum(pri[h] * (s * P_DBS[h]) * (comp * r1ok * min(1.0, s * P_DBS[h] * corr)) for h in pri)
    return o


pr1 = {"H_shift": 0.65, "H_grow": 0.25, "H_full": 0.03, "SHRINK": 0.07}
g2 = 0.375; draft_pri = {"H_grow": g2, **{k: pr1[k] / 0.75 * (1 - g2) for k in ("H_shift", "H_full", "SHRINK")}}
draft = oc(0.75, {"LF": 0.9, "VB": 0.9, "K24": 0.70, "RET": 0.95}, draft_pri, {"H_shift": 0.05, "H_grow": 0.60, "H_full": 0.95, "SHRINK": 0.0},
           {"H_shift": 0.10, "H_grow": 0.75, "H_full": 1.0, "SHRINK": 0.0}, {"H_shift": 0.05, "H_grow": 0.45, "H_full": 0.90, "SHRINK": 0.0}, 0.25, (0.8, 0.85, 1.0 / 0.539 * 0.85))
rt_pri = {"H_shift": 0.60, "H_grow": 0.30, "H_full": 0.02, "SHRINK": 0.08}
rt_kept = {"LF": 0.9, "VB": 0.9, "K24": 0.60, "RET": 0.95, "not_DIVERGED": 0.90}
rt_dbs = {"H_shift": 0.12, "H_grow": 0.55, "H_full": 0.90, "SHRINK": 0.0}
rt_mg = {"H_shift": 0.18, "H_grow": 0.70, "H_full": 1.0, "SHRINK": 0.0}
rt_kb = {"H_shift": 0.10, "H_grow": 0.40, "H_full": 0.85, "SHRINK": 0.0}
rt = oc(0.60, rt_kept, rt_pri, rt_dbs, rt_mg, rt_kb, 0.20, (0.75, 0.70, 1.2))
rt_preshift = oc(0.60, rt_kept, draft_pri, rt_dbs, rt_mg, rt_kb, 0.20, (0.75, 0.70, 1.2))   # RT shading but the pre-stated x1.5 prior shift kept
rt_hi = oc(0.70, {**rt_kept, "K24": 0.70}, draft_pri, {**rt_dbs, "H_shift": 0.08, "H_grow": 0.60}, rt_mg, rt_kb, 0.20, (0.8, 0.8, 1.2))
rt_lo = oc(0.50, {**rt_kept, "K24": 0.50}, rt_pri, {**rt_dbs, "H_shift": 0.18, "H_grow": 0.45}, rt_mg, rt_kb, 0.20, (0.7, 0.6, 1.2))
out = {"draft_reproduced": draft, "draft_priors": draft_pri, "rt_shaded": rt, "rt_shaded_with_prestated_shift": rt_preshift, "rt_range_hi": rt_hi, "rt_range_lo": rt_lo,
       "assumed": {"rt_priors": rt_pri, "rt_kept": rt_kept, "rt_P_DBS": rt_dbs, "rt_P_MG": rt_mg, "rt_P_KB": rt_kb}}
json.dump(out, open(os.path.join(HERE, "rt_oc.json"), "w", newline="\n"), indent=1)
for k in ("draft_reproduced", "rt_shaded", "rt_shaded_with_prestated_shift", "rt_range_hi", "rt_range_lo"):
    v = out[k]; print(k, {kk: (round(vv, 4) if isinstance(vv, float) else {a: round(b, 4) for a, b in vv.items()}) for kk, vv in v.items()})
