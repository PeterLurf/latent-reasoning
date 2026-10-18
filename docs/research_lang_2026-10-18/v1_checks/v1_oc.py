"""LANG v1 $0: the OC. (1) Re-runs RT's rt_lang/oc_shade.py mixture logic (copied) and ASSERTS its oc_shade.json exactly; (2) applies
Theory's L1-L4 effects with every added factor stated [ASSUMED]; (3) sampling-level P(pass | no effect) / P(pass | effect) per reading at the
registered n; (4) clean-pair projections from the measured exposure counts (v1_sym_lead.json, v1_gsm_exposure.json).
    python -B v1_oc.py   -> v1_oc.json
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__)); RT = os.path.join(os.path.dirname(HERE), "rt_lang", "oc_shade.json")
Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))  # noqa: E731


def mix_vbc1(p_entry, p_deep_entry, mech, pp):
    p_deep = p_entry * p_deep_entry; serial = [m for m in mech if m.startswith("serial")]
    tot = p_deep * sum(mech[m] * pp[m] for m in mech); fal = p_deep * sum(mech[m] * pp[m] for m in mech if m not in serial)
    nos = sum(mech[m] * pp[m] for m in mech if m not in serial) / sum(mech[m] for m in mech if m not in serial)
    eff = sum(mech[m] * pp[m] for m in serial) / sum(mech[m] for m in serial)
    return {"P(pass)": round(tot, 3), "P(false positive)": round(fal, 3), "P(pass | no serial use)": round(nos, 3),
            "P(pass | serial use)": round(eff, 3), "share of positives that are false": round(fal / tot, 2)}, tot


out = {}
# ---------- (1) RT reproduction (identical inputs to rt_lang/oc_shade.py) ----------
mech = {"serial_qcond": 0.35, "serial_allpersons": 0.15, "precompute_carry_or_backchase": 0.30, "recompute_v0": 0.10, "mixed_onepass_D5_6": 0.10}
pp_fixed = {"serial_qcond": 0.50, "serial_allpersons": 0.45, "precompute_carry_or_backchase": 0.02, "recompute_v0": 0.02, "mixed_onepass_D5_6": 0.10}
pp_cur = {"serial_qcond": 0.55, "serial_allpersons": 0.02, "precompute_carry_or_backchase": 0.19, "recompute_v0": 0.02, "mixed_onepass_D5_6": 0.10}
rt = json.load(open(RT))
r_fixed, a_fixed = mix_vbc1(0.55, 0.60, mech, pp_fixed); r_cur, _ = mix_vbc1(0.55, 0.60, mech, pp_cur)
assert r_fixed == rt["VB-C1-L_fixed"] and r_cur == rt["VB-C1-L_current"], "RT reproduction failed"
gm = {"serial_step": 0.08, "precompute_carry": 0.45, "recompute_v0": 0.30, "step_not_learned": 0.17}
gp = {"serial_step": 0.35, "precompute_carry": 0.01, "recompute_v0": 0.01, "step_not_learned": 0.0}
g_tot = 0.75 * sum(gm[m] * gp[m] for m in gm); g_fal = 0.75 * sum(gm[m] * gp[m] for m in gm if m != "serial_step")
g_nos = sum(gm[m] * gp[m] for m in gm if m != "serial_step") / sum(gm[m] for m in gm if m != "serial_step")
assert round(g_tot, 3) == rt["VB-C1-GSM_fixed"]["P(pass)"] and round(g_nos, 3) == rt["VB-C1-GSM_fixed"]["P(pass | no serial use)"]
se_b = math.sqrt(0.014 ** 2 + 2 * 0.015 ** 2); p_beats_null = 0.5 * math.erfc(0.03 / se_b / math.sqrt(2))
assert round(p_beats_null, 3) == rt["BEATS-CODI20"]["with seed variance"]
sd_c2 = round(rt["VB-C1-L_fixed"]["P(pass)"] * 0.5 * 0.4 * 0.4, 3); assert sd_c2 == rt["SD-C2-L"]["P(pass), fixed pool + fixes"]
out["rt_reproduced"] = {"VB-C1-L_fixed": r_fixed, "VB-C1-GSM_fixed_ungated": {"P(pass)": round(g_tot, 3), "P(pass | no serial use)": round(g_nos, 3)},
                        "BEATS_null_with_seed_var": round(p_beats_null, 3), "SD-C2-L": sd_c2, "asserted_equal_to": "rt_lang/oc_shade.json"}

# ---------- (2) L1-L4 effects, each factor stated ----------
adj = {}
# (a) D-13: B1 launches only if VB-C1-L reads; P(VB-C1-GSM | launched) = 0.10 (RT's own assumption in oc_shade.py)
p_b1 = a_fixed; p_gsm_given = 0.10; p_gsm = p_b1 * p_gsm_given
adj["a_B1_gate"] = {"P(B1 launched)": round(p_b1, 3), "P(VB-C1-GSM | launched)": p_gsm_given, "P(VB-C1-GSM) unconditional": round(p_gsm, 4),
                    "P(VB-C1-GSM | launched, no serial use)": round(g_nos, 3)}
# (b) N8: with the gate, P(>=1) = P(VB-C1-L) + P(SD-C2-L and not VB-C1-L on A1); A2 can run without VB-C1-L (GO = DEEP and NO-CLOCK)
p_a2 = 0.15                                                          # RT F3-4: P(A2 runs) ~ 0.15
p_a2_not_vbc1 = max(0.0, p_a2 - a_fixed * 0.5)                       # ~ half of VB-C1-L lineages also clear the yield gate [ASSUMED]
p_sdc2_not = p_a2_not_vbc1 * 0.01                                    # SD-C2-L needs VB-C1-L re-read on A2 [ASSUMED 0.01]
p_any = a_fixed + p_sdc2_not
adj["b_P_any"] = {"P(>=1 decisive C1)": round(p_any, 3), "E[number of decisive positives]": round(a_fixed + p_gsm + sd_c2, 3),
                  "RT's 0.095 is": "a + 0.1 a = the expected count when B1 is gated (P(any) counts VB-C1-GSM inside VB-C1-L)"}
# (c) DIRECT-SYM: C1 band = {D in 5..8: DIRECT acc <= 0.65 for all D' >= D}; < 2 depths -> VB-C1-L not met
p_band_lt2 = 0.05                                                    # P(D*_fresh >= 7) [ASSUMED]
a_c = a_fixed * (1 - p_band_lt2)
adj["c_DIRECT_SYM_band"] = {"P(C1 band < 2 depths)": p_band_lt2, "P(VB-C1-L)": round(a_c, 3)}
# (d) N1 ALL-ANCHORS: a third of the RT's ALL-PERSONS mass is anchor-tagged and reads own under PREMISE-SWAP (fails SERIAL-STEP-L)
mech_d = dict(mech); mech_d.pop("serial_allpersons"); mech_d["serial_allpersons_nametag"] = 0.10; mech_d["serial_allanchors"] = 0.05
pp_d = dict(pp_fixed); pp_d.pop("serial_allpersons"); pp_d["serial_allpersons_nametag"] = 0.45; pp_d["serial_allanchors"] = 0.02
r_d, a_d = mix_vbc1(0.55, 0.60, mech_d, pp_d)
adj["d_ALL_ANCHORS"] = r_d
# (e) A-2 CONSUMPTION-FRONT (CF-L) added to VB-C1-L's conjunction; P(CF-L | class) from v1_cf.json's calibration (registered rule):
#     serial (incl. 50 % one-hop look-ahead, readout flips <= 0.10): 1.0 in the MC -> 0.9 [ASSUMED: allows flips > 0.10 / mixed schedules];
#     precompute-carry, recompute, clocks, doubling, partial one-pass: 0.0 in 150 reps -> 0.02 [ASSUMED upper]; mixed one-pass at D 5-6: 0.2 [ASSUMED]
cfp = {"serial_qcond": 0.9, "serial_allpersons_nametag": 0.9, "serial_allanchors": 0.9, "precompute_carry_or_backchase": 0.02, "recompute_v0": 0.02, "mixed_onepass_D5_6": 0.2}
pp_e = {m: pp_d[m] * cfp[m] for m in pp_d}
r_cf, a_cf = mix_vbc1(0.55, 0.60, mech_d, pp_e)
adj["e_CONSUMPTION_FRONT"] = {**r_cf, "P(CF-L | class)": cfp}
# headline after (a)-(e)
a_head = a_cf * (1 - p_band_lt2); r_d = r_cf
head = {"P(VB-C1-L)": round(a_head, 3), "P(VB-C1-L | no serial use)": r_d["P(pass | no serial use)"], "P(VB-C1-L | serial use)": r_d["P(pass | serial use)"],
        "share of positives that are false": r_d["share of positives that are false"],
        "P(SD-C2-L)": round(a_head * 0.5 * 0.4 * 0.4, 4), "P(SD-C2-L | H_shift)": 0.02,
        "P(B1 launched)": round(a_head, 3), "P(VB-C1-GSM | launched)": p_gsm_given, "P(VB-C1-GSM) unconditional": round(a_head * p_gsm_given, 4),
        "P(VB-C1-GSM | launched, no serial use)": round(g_nos, 3), "P(VB-C1-GSM) if B1 ran ungated (RT fixed)": round(g_tot, 3),
        "P(BEATS-CODI20 | launched, equal accuracy) item-only / with seed variance": [0.016, round(p_beats_null, 3)],
        "P(>=1 decisive C1)": round(a_head + p_sdc2_not, 3)}
out["L1_L4_adjustments"] = adj; out["headline_v1"] = head
# (f) sensitivity: decoys make backward precompute harder (backchase mass 0.30 -> 0.25, to serial_qcond) but entry harder (0.55 -> 0.50)
mech_e = dict(mech_d); mech_e["precompute_carry_or_backchase"] = 0.25; mech_e["serial_qcond"] = 0.40
r_e, a_e = mix_vbc1(0.50, 0.60, mech_e, {m: pp_d[m] * cfp[m] for m in pp_d})
out["sensitivity_decoys"] = {**r_e, "P(VB-C1-L) incl. band factor": round(a_e * (1 - p_band_lt2), 3)}
out["sensitivity_rt_exact"] = {"P(VB-C1-L)": r_fixed["P(pass)"], "P(>=1)": round(r_fixed["P(pass)"] + p_sdc2_not, 3)}
out["sensitivity_without_CF"] = {"P(VB-C1-L)": round(a_d * (1 - p_band_lt2), 3), "P(pass | no serial use)": adj["d_ALL_ANCHORS"]["P(pass | no serial use)"]}

# ---------- (3) sampling-level OC per reading ----------
def p_ge(bar, mu, se):
    return 1 - Phi((bar - mu) / se)


rows = {}
n_c1 = 4000; se_g = math.sqrt(2 * 0.25 / n_c1)
rows["PROBE-GAIN-L >= 0.35 (n ~ 1,000 per C1-band depth)"] = {"no_effect": "gain 0 +- %.3f -> %.1e" % (se_g, p_ge(0.35, 0.0, se_g)), "effect": "~0.8 (RT; nested gain 0.87 at chance 1/11, here max 0.5)"}
rows["PROBE-VALID (PLANT lin + XOR, planted acc >= 0.90)"] = {"no_effect": "a probe that cannot decode a planted 1-bit code fails -> PROBE-INVALID (censor)", "effect": "~0.95 [ASSUMED]"}
n1 = n2 = 8800                                                       # mirror slots per class per (t, k), pooled over D 6..8 (v1_sym: ~2.95 per item x 3,000)
se_auc = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
rows["NO-PRECOMPUTE-L: every (t in 1,2; k >= 3) AUROC <= 0.65 (lin and MLP)"] = {"AUROC_SE": round(se_auc, 4), "P(pass | no precompute, AUROC 1/2 exact)": round(Phi((0.65 - 0.5) / se_auc), 6),
    "P(pass | decodable precompute, AUROC >= 0.9)": "%.1e" % (1 - Phi((0.9 - 0.65) / se_auc))}
for arm in ("PREMISE-SWAP x (ii') follow >= 0.80 pooled, >= 0.60 per D", "SLOT-LOCAL (i') follow >= 0.80 pooled, >= 0.60 per D"):
    rows[arm] = {"no_effect (random disruption, follow 0.5, >= 1,000 clean pairs)": "%.1e" % p_ge(0.80, 0.5, math.sqrt(0.25 / 1000)),
                 "no_effect (recompute: follow ~0)": "~0", "effect (serial, name-tagged; nested follow_i 0.996-1.000)": "~0.85-0.9 [ASSUMED]"}
rows["NO-CLOCK-L STALL >= 0.9"] = {"no_effect (count-keyed, N3 0.499)": "<= 0.05", "effect": "~0.9"}
rows["SERIAL-ONSET-L slope >= 0.6 and ON(8)-ON(4) >= 3"] = {"no_effect (flat ON)": "<= 0.05", "effect": "~0.7"}
se_beats_item = 0.014; se_beats_seed = se_b
rows["BEATS-CODI20 (delta >= 0.03 and exact McNemar p <= 0.05)"] = {"P(pass | equal) item-only": round(1 - Phi(0.03 / se_beats_item), 3), "P(pass | equal) with sigma_seed 0.015 per model": round(p_beats_null, 3),
    "P(pass | true delta +0.06) item-only / with seed var": [round(1 - Phi((0.03 - 0.06) / se_beats_item), 3), round(1 - Phi((0.03 - 0.06) / se_beats_seed), 3)],
    "McNemar adds nothing at SE 0.014": "delta 0.03 -> z = %.2f -> one-sided p %.3f < 0.05" % (0.03 / se_beats_item, 1 - Phi(0.03 / se_beats_item))}
rows["A0-1 anti-stratum acc <= 0.40 (n 78)"] = {"P(<= 0.40 | no reliance, true acc 0.95)": "%.1e" % Phi((0.40 - 0.95) / math.sqrt(0.95 * 0.05 / 78))}
out["oc_rows"] = rows

# ---------- (4) clean-pair projections from measured exposure ----------
sym = json.load(open(os.path.join(HERE, "v1_sym_lead.json")))["bands"]; gsm = json.load(open(os.path.join(HERE, "v1_gsm_exposure.json")))
proj = {"ProsQA_exposed_per_D_(test 1,000 per D; (i') and (ii'))": {D: sym[D]["expo"]["exposed"] for D in sym}}
proj["ProsQA_clean_per_D_at_acc"] = {a: round(1000 * a * a) for a in (0.90, 0.95, 0.99)}
ex = gsm["m_ge_3"]["items_exposed"]; cand = ex * 21
proj["GSM_exposed_base_items_m_ge_3"] = ex; proj["GSM_candidates_with_20_Rx_variants"] = cand
proj["GSM_clean_pairs"] = {"acc %.2f rho %.1f" % (a, r): round(cand * (a * a + r * a * (1 - a))) for a in (0.15, 0.27, 0.40) for r in (0.0, 0.3, 0.5)}
proj["GSM_clean_pairs_without_Rx"] = {"acc 0.27 rho 0.3": round(ex * (0.27 ** 2 + 0.3 * 0.27 * 0.73))}
out["exposure_projection"] = proj
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "v1_oc.json"), "w", newline="\n"), indent=1)
