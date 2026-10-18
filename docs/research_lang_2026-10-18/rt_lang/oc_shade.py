"""REDTEAM_LANG $0: shaded OC for LANG-VB's decisive readings, as a mechanism mixture (every number [ASSUMED], stated so Theory can
re-shade). 'current' = the v0 design (PERSON-SWAP arm ii only; PROBE without a positive control; no FUTURE-DECODE on ProsQA; no PROBE-VALID
on GSM8K); 'fixed' = with F1-1 (PLANT-certified probes, FD-L decisive, slot-local arm co-primary), F1-2 (PREMISE-SWAP) applied.
    python -B oc_shade.py
"""
import json

out = {}
# ---------------- A1 (ProsQA-SYM) ----------------
p_entry, p_deep_entry = 0.55, 0.60                       # draft 0.6 / 0.67: language binding, all slots on the token "us" (F2-4), T ~ 530-690
p_deep = p_entry * p_deep_entry
mech = {"serial_qcond": 0.35, "serial_allpersons": 0.15, "precompute_carry_or_backchase": 0.30, "recompute_v0": 0.10, "mixed_onepass_D5_6": 0.10}
p_pass = {"current": {"serial_qcond": 0.55, "serial_allpersons": 0.02, "precompute_carry_or_backchase": 0.19, "recompute_v0": 0.02, "mixed_onepass_D5_6": 0.10},
          "fixed": {"serial_qcond": 0.50, "serial_allpersons": 0.45, "precompute_carry_or_backchase": 0.02, "recompute_v0": 0.02, "mixed_onepass_D5_6": 0.10}}
serial = ("serial_qcond", "serial_allpersons")
for v, pp in p_pass.items():
    tot = p_deep * sum(mech[m] * pp[m] for m in mech); fal = p_deep * sum(mech[m] * pp[m] for m in mech if m not in serial)
    nos = sum(mech[m] * pp[m] for m in mech if m not in serial) / sum(mech[m] for m in mech if m not in serial)
    eff = sum(mech[m] * pp[m] for m in serial) / sum(mech[m] for m in serial)
    out["VB-C1-L_" + v] = {"P(pass)": round(tot, 3), "P(false positive)": round(fal, 3), "P(pass | no serial use)": round(nos, 3),
                           "P(pass | serial use)": round(eff, 3), "share of positives that are false": round(fal / tot, 2)}
# ---------------- A2 / SD-C2-L ----------------
p_vbc1 = out["VB-C1-L_fixed"]["P(pass)"]; p_yield = 0.5; p_sdself_a2 = 0.4; p_beyond = 0.4
out["SD-C2-L"] = {"P(pass), fixed pool + fixes": round(p_vbc1 * p_yield * p_sdself_a2 * p_beyond, 3), "P(pass), pool as drafted (60k/D)": 0.0,
                  "P(pass | H_shift) as drafted (no MARGIN-GROWS / CTRL term)": "about P(A1 DEPTH-L >= 0.8 | SD-SELF) ~ 0.25",
                  "P(pass | H_shift) with MARGIN-GROWS-L + A2-CTRL in the conjunction": 0.02}
# ---------------- B1 (GSM8K) ----------------
p_b_trains = 0.75; gm = {"serial_step": 0.08, "precompute_carry": 0.45, "recompute_v0": 0.30, "step_not_learned": 0.17}
gp = {"current": {"serial_step": 0.40, "precompute_carry": 0.10, "recompute_v0": 0.01, "step_not_learned": 0.0},
      "fixed": {"serial_step": 0.35, "precompute_carry": 0.01, "recompute_v0": 0.01, "step_not_learned": 0.0}}
for v, pp in gp.items():
    tot = p_b_trains * sum(gm[m] * pp[m] for m in gm); fal = p_b_trains * sum(gm[m] * pp[m] for m in gm if m != "serial_step")
    nos = sum(gm[m] * pp[m] for m in gm if m != "serial_step") / sum(gm[m] for m in gm if m != "serial_step")
    out["VB-C1-GSM_" + v] = {"P(pass)": round(tot, 3), "P(false positive)": round(fal, 3), "P(pass | no serial use)": round(nos, 3),
                             "share of positives that are false": round(fal / tot, 2)}
# accuracy label vs CODI-20: paired item SE 0.014 (draft) plus one-seed variance of each model (sigma_seed ~ 0.015 each) [ASSUMED]
import math
se = math.sqrt(0.014 ** 2 + 2 * 0.015 ** 2); z = 0.03 / se; pz = 0.5 * math.erfc(z / math.sqrt(2))
out["BEATS-CODI20"] = {"P(pass | equal true accuracy), item SE only (draft)": 0.016, "with seed variance": round(pz, 3), "total SE": round(se, 3)}
# ---------------- design headline ----------------
for v in ("current", "fixed"):
    a = out["VB-C1-L_" + v]["P(pass)"]; g = out["VB-C1-GSM_" + v]["P(pass)"]
    out["P(>=1 decisive C1 positive)_" + v + "_B_ungated"] = round(1 - (1 - a) * (1 - g), 3)
a = out["VB-C1-L_fixed"]["P(pass)"]
out["P(>=1 decisive C1 positive)_fixed_B_gated_on_VB-C1-L"] = round(a + a * 0.10, 3)   # P(VB-C1-GSM | VB-C1-L on SYM) ~ 0.10 [ASSUMED]
for k, v in out.items():
    print(k, v)
json.dump(out, open("oc_shade.json", "w", newline="\n"), indent=1)
