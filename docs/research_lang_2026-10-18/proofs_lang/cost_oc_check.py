"""PROOFS_LANG item 6: re-derive cost_model.py's numbers from its stated formulas (my code), re-run them at the token
counts measured in sv_*.json (2.5-token names), sum the cost ranges, and recompute the OC headline arithmetic."""
import json
def item(Tq, S, K, Ldec, T0=None, nwin=5):
    T0 = Tq if T0 is None else T0
    tl = 6 * Tq + 4 * S * K + 2 * (S + Ldec) * nwin
    att = 48 * sum(S * (T0 + (t - 1) * S + S) for t in range(1, K + 1)) + 72 * Tq * Tq / 2
    return tl, att
ref = item(34, 34, 12, 0)[0]
def row(Tq, S, K, Ldec=12, usd_h=0.22, steps=24000):
    tl, att = item(Tq, S, K, Ldec); r = tl / ref; s = max(0.42, 0.42 * (0.4 + 0.6 * r))
    return {"work_ratio": round(r, 3), "s_step": round(s, 3), "attn_M": round(att / 1e6, 1), "GB_fp32_bs64": round(att * 4 * 64 / 1e9, 1), "usd": round(steps * s / 3600 * usd_h, 2)}
out = {"draft_rows": {"sym_D8": row(450, 48, 12), "sym_D12_self": row(600, 64, 16, steps=12000), "sym_D16_eval": row(760, 82, 20), "mention_D8": row(450, 100, 12)},
       "measured_T": {"sym_D8_T533_S50": row(533, 50, 12), "sym_D12_self_T690_S64": row(690, 64, 16, steps=12000), "sym_D16_eval_T830_S84": row(830, 84, 20)},
       "fullT_entries_M": round(48 * 450 ** 2 * 12 * 13 / 2 / 1e6, 1), "fullT_GB_bs64": round(48 * 450 ** 2 * 78 * 4 * 64 / 1e9, 1)}
# B1 launch-bound scaling: batch max K under K_i ~ U{m_bin..16} at bs 64
pmax16 = 1 - (1 - 1 / 14) ** 64
out["B1"] = {"P_batch_has_K16": round(pmax16, 4), "launch_bound_s_step_16_over_12": round(0.42 * 16 / 12, 3),
             "usd_120k_at_0.45": round(120000 * 0.45 / 3600 * 0.22, 2), "usd_120k_at_0.56": round(120000 * 0.56 / 3600 * 0.22, 2), "usd_120k_at_0.9": round(120000 * 0.9 / 3600 * 0.22, 2)}
A1, A2, B1 = (0.8, 1.2, 2.5), (0.5, 0.75, 1.5), (3.3, 4.0, 6.6)
lo, c, hi = (round(A1[i] + A2[i] + B1[i], 2) for i in range(3))
out["required_total"] = {"sum_of_lows": lo, "central": c, "sum_of_highs": hi, "draft_low_is": round(A1[1] + A2[1] + B1[0], 2), "with_10pct": [round(lo * 1.1, 2), round(hi * 1.1, 2)],
                         "with_B1_low_0.56": round(A1[0] + A2[0] + 0.56 * 120000 / 3600 * 0.22, 2)}
out["guards_sum"] = round(2.8 + 1.7 + 1.2 + 1.4 + 9, 2)
out["worst_case_spend_incl_optionals_10pct"] = round((2.5 + 1.5 + 6.6 + 1.1 + 1.4) * 1.1, 2)
out["A2CTRL_usd_at_0.22"] = round(12000 * 0.82 / 3600 * 0.22, 2); out["A2_usd"] = round(12000 * (0.5 * 1.15 + 0.5 * 0.82) / 3600 * 0.22, 2); out["ATXT_usd"] = round(12000 * 1.35 / 3600 * 0.22, 2)
out["D3T_anchor_usd_48k_0.20"] = round(48000 * 0.42 / 3600 * 0.20, 2)
# OC
pA, pG, pSD = 0.2, 0.08, 0.05
out["OC"] = {"VB_C1_L_components_product": round(0.8 * 0.7 * 0.85 * 0.9, 4), "VB_C1_GSM_components_product": round(0.85 * 0.6 * 0.9 * 0.8, 4),
             "P_VBC1L_from_0.35x0.5": round(0.35 * 0.5, 4), "plus_false_pos_bound": round(0.35 * 0.5 + 0.65 * 0.02, 4),
             "P_ge1_indep_C1_only": round(1 - (1 - pA) * (1 - pG), 4), "P_ge1_indep_incl_SD": round(1 - (1 - pA) * (1 - pG) * (1 - pSD), 4), "P_ge1_lower_bound_max": max(pA, pG),
             "SD_C2_chain_low": round(0.4 * 0.85 * 0.8 * 0.65 * 0.4 * 0.7 * 0.268, 4), "SD_C2_chain_high": round(0.4 * 0.9 * 0.9 * 0.65 * 0.5 * 0.8 * 0.5, 4),
             "ABSORB_unconditional_from_chain": round(0.4 * 0.85 * 0.8 * 0.65, 4),
             "BEATS_fp": None, "A0_2_max_0.5+3.2SE": round(0.5 + 3.2 * (0.25 / 1000) ** 0.5, 4)}
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))  # noqa: E731
out["OC"]["BEATS_fp"] = round(1 - Phi(0.03 / 0.014), 4); out["OC"]["BEATS_power_d0.06"] = round(1 - Phi((0.03 - 0.06) / 0.014), 4)
print(json.dumps(out, indent=1))
json.dump(out, open("cost_oc_check.json", "w", newline="\n"), indent=1)
