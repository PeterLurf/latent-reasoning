"""RT-SD2 check 1 ($0, CPU, numpy only; read-only on D:/Latent). Inputs: boxes/runs_fsq/*/eval_vb.json (acc_n2, acc_k) for the VB cells; FT-CTRL's
eval_vb only (NOT its eval_hist24 / eval_d3hist). Output rt_curves.json beside this file.
 (1) margins at several accuracy levels (0.9 / 0.7 / 0.5 / 0.3) and margin/B for D3-T (B 8) and FT-SELF (B 12): is H_prop's 0.288 = 0.288 level-specific?
 (2) H_prop back-cast: predict FT-SELF's n 13-16 from D3-T by a1(12 + d) = a0(8 + (8/12) d); compare M_pred with the measured 0.717.
 (3) the other B-8 VB cells' n50 (recipe spread at fixed B; a floor on lineage spread).
 (4) M2 under H_add (pure translation of round 1 by 4 + 1.15 = the same n50 as H_prop) vs H_prop's 0.852.
 (5) K' saturation on FT-SELF (a_n(n) and a_{n-1}(n) vs a(n), n 13-16): what K'-BEYOND measures.
 (6) FT-CTRL STOP-MOVED (score_s213.acc_gain_after10 form, from eval_vb only) -> branch P1 / P2 is already determined.
 (7) lineage-noise OC: P(M2 >= bar | H) for a Gaussian lineage SD sigma on M2 (sigma 0.03 / 0.06 / 0.09 / 0.12)."""
import json
import math
import os

R = "D:/Latent/boxes/runs_fsq/"; HERE = os.path.dirname(os.path.abspath(__file__)); out = {}


def ev(d):
    e = json.load(open(R + d + "/eval_vb.json"))
    return {int(n): v for n, v in e["acc_n2"].items()}, {int(k): {int(n): v for n, v in dd.items()} for k, dd in e["acc_k"].items()}


def crossing(a, level, lo, hi):
    for n in range(lo, hi):
        if a[n] >= level > a[n + 1]:
            return n + (a[n] - level) / (a[n] - a[n + 1])
    return None


a0, K0 = ev("s213_d3t_s1"); a1, K1 = ev("s215_ftself_s1"); ac, Kc = ev("s215_ftctrl_s1")
lv = {}
for L in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
    c0, c1 = crossing(a0, L, 8, 16), crossing(a1, L, 12, 16)
    lv[str(L)] = {"base_n": c0, "base_m_over_B": None if c0 is None else (c0 - 8) / 8, "r1_n": c1, "r1_m_over_B": None if c1 is None else (c1 - 12) / 12,
                  "ratio_r1_over_base": None if (c0 is None or c1 is None) else ((c1 - 12) / 12) / ((c0 - 8) / 8)}
out["margins_by_level"] = lv


def ip(a, x):
    lo = int(math.floor(x)); f = x - lo
    return a[lo] if f == 0 else (1 - f) * a[lo] + f * a[lo + 1]


pred = {12 + d: ip(a0, 8 + 8 / 12 * d) for d in range(1, 5)}
out["Hprop_backcast_round1"] = {"pred_a1_n13_16": pred, "measured": {n: a1[n] for n in range(13, 17)}, "M_pred": sum(pred.values()) / 4,
                                "M_measured": sum(a1[n] for n in range(13, 17)) / 4,
                                "slope_at_50pct": {"base": a0[10] - a0[11], "r1": a1[15] - a1[16], "Hprop_expected_r1": (a0[10] - a0[11]) * 8 / 12}}
others = {}
for d in ("s211_vb_pa_s1", "s211_vb_carry_s1", "s211_vb_pamct_s1", "s213_d3f_s1", "s213_d3t_s1", "s215_ftctrl_s1"):
    try:
        a, _ = ev(d); c = crossing(a, 0.5, 8, 16)
        others[d] = {"a9_12": [round(a[n], 3) for n in range(9, 13)], "n50": c, "M_9_12": sum(a[n] for n in range(9, 13)) / 4}
    except Exception as e:  # noqa: BLE001
        others[d] = str(e)
out["B8_cells_n50"] = others
ext = {17: 0.180, 18: 0.124, 19: 0.110, 20: 0.110}                       # the draft's assumed a1 extension
A1 = {**a1, **ext}
Hadd = {16 + d: ip(A1, 12 + d - 1.1532) for d in range(1, 5)}
Hprop = {16 + d: ip(A1, 12 + 0.75 * d) for d in range(1, 5)}
Hshift = {16 + d: A1[12 + d] for d in range(1, 5)}
out["M2_panels"] = {"H_shift": sum(Hshift.values()) / 4, "H_prop": sum(Hprop.values()) / 4, "H_add_translate_5.15": sum(Hadd.values()) / 4,
                    "H_add_curve": Hadd, "H_prop_curve": Hprop, "H_shift_curve": Hshift,
                    "note": "H_add = round 1's curve translated by +5.15 (same n50 20.61 as H_prop, steepness kept); H_prop stretches it by 16/12"}
sat = {n: {"a": a1[n], "a_k_n": K1[n][n] if n in K1 else None, "a_k_n_minus_1": K1[n - 1][n], "a_k_n_minus_2": K1[n - 2][n]} for n in range(13, 17)}
out["Kprime_saturation_FTSELF"] = sat
out["Kprime_beyond_round1_decomposed"] = {"KB": sum(a1[n] - K1[14][n] for n in (15, 16)) / 2,
                                          "gain_passes_le_16": sum(K1[16][n] - K1[14][n] for n in (15, 16)) / 2,
                                          "gain_passes_gt_16 (RISE-PAST-K)": sum(a1[n] - K1[16][n] for n in (15, 16)) / 2}
g = {n: ac[n] - Kc[10][n] for n in range(11, 15)}
out["CTRL_STOP_MOVED"] = {"gain_n2_minus_k10": g, "mean_gain": sum(g.values()) / 4, "STOP_MOVED": sum(g.values()) / 4 >= 0.10,
                          "note": "acc_gain_after10 reads eval_vb only; eval_vb of FT-CTRL is not blinded -> PAIR branch P1/P2 is decidable now"}
gs = {n: a1[n] - K1[10][n] for n in range(11, 15)}
out["SELF_STOP_MOVED_check"] = sum(gs.values()) / 4


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


Ms = {"H_shift": out["M2_panels"]["H_shift"], "H_prop": out["M2_panels"]["H_prop"], "H_add": out["M2_panels"]["H_add_translate_5.15"], "SHRINK": 0.4798}
noc = {}
for s in (0.03, 0.06, 0.09, 0.12):
    noc[str(s)] = {h: {"P_M2_ge_0.80": 1 - Phi((0.80 - m) / s), "P_M2_ge_0.785": 1 - Phi((0.785 - m) / s), "P_M2_lt_0.60": Phi((0.60 - m) / s)} for h, m in Ms.items()}
    noc[str(s)]["LR_DBS_pass_Hprop_vs_Hshift"] = noc[str(s)]["H_prop"]["P_M2_ge_0.80"] / max(1e-12, noc[str(s)]["H_shift"]["P_M2_ge_0.80"])
out["lineage_noise_OC"] = noc
json.dump(out, open(os.path.join(HERE, "rt_curves.json"), "w", newline="\n"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
