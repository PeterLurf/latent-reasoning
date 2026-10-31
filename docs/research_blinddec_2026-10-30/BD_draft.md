# BLIND-DEC draft: does removing the decoder's last-step shortcut make answer-only loop formation work? (C4) (2026-10-30)

**Status:** draft for red team. Theory's choice. GATE-CURRIC was parked (Note 251-8); this targets the mechanism found there.

## 1. Why
- **Note 251-8:**
  - The decoder reads the full problem text plus S_K, so it takes the last step itself: d 2 at K′ 1 ≈ 1.00 in every model.
  - Under answer-only training the d 2 answer is reachable via 1 pass + decoder, so pass 2 is never made to write. After the onset break, AEX0 re-learned this shortcut and never extended.
  - §244 / §245 (answer-only from askname, d ≤ 2, full decoder) formed 1 of 4, labelled DEC-STEP2, i.e. "chaining untested at d 2".
- **Change:** a **blind decoder** that sees only the final question, not the problem text:
  - `[EOT] + "How many <u> does <S_d> have?" + " The answer is:"`, reading S_K as now.
  - The op of every step lives only in the problem text, which only the think passes see. So the answer at depth d can only come from a state that already holds v_d, and pass d must write.
- **Question:** without any per-pass targets, does the d ≤ 2 loop then form reliably from askname? (C4 scope: askname init, d 1 foothold, plan given; no DEC-STEP2 caveat.)
- **Precedent in the project:** §171 / §179 blind end decoders on synthetic arithmetic. There the failure was carrying 3+ columns to one readout; here the answer is a single value mod 11.

## 2. Arms (P-2 recipe: askname init s233_askname_s1, d ~ U{1, 2}, 12k, K 8, RTX 3090, micro pinned)
| Arm | Runs | Flags vs P-2 (aw_plaw2_s1 args) | Role |
|---|---|---|---|
| BLD-AO | bld_ao_s1 / s2 | `--dec_blind --aux_fade 0,0` | the C4 test |
| BLD-AUX | bld_aux_s1 / s2 | `--dec_blind` | positive control: is blind decoding learnable at all with aux? |

- **Comparators:**
  - §244 / §245 ao_s1–s4 (full decoder, answer-only; formation 1/4);
  - P-2 aw_plaw2_s1 (full decoder + aux).
- **Law:** P-2's law (TOPO, as §244 / §245), so formation rates compare directly.
- **Build:** `--dec_blind` in tok_aw sets it["dec"] = [EOT] + final-question tokens + SFX. Default off = bitwise current. Eval reads the flag from args.json, so every eval part uses the blind decoder.

## 3. Readings (TEST2 of P-2's law; per seed; binding)
**score_ao's gates on d 2,** with blind adaptations:
- **ACC2:** d 2 ≥ 0.90.
- **CHAIN2:** d 2 at K′ 1 ≤ 0.20. This is new: with a blind decoder, K′ 1 should fail; if it does not, the "blindness" leaks.
- **LP5:** mean state at pass 1, d 2 drop ≥ 0.30.
- **SWAP-MID d 2 t 1:** follow_U ≥ 0.50 and ≥ floor + 0.30.
- **CUE-ONLY guard:** think passes on the cue only, blind decoder; acc ≤ majority + 0.05.
- **STABLE.**

**Labels:**
- FORMED (all gates);
- D1-ONLY (d 1 ≥ 0.9, d 2 fails);
- NOT-FORMED;
- VOID (guard);
- UNSTABLE.

**Per arm:** FORMED k/2.

**Readings:**
- **C4 reading (scoped)** if BLD-AO forms 2/2 and BLD-AUX forms ≥ 1/2: "AUX-FREE CHAIN at d 2 (askname init, plan given)".
- **BLD-AUX 0/2:** the blind decoder itself is not learnable in this harness. No C4 reading (INCONCLUSIVE).

**Descriptive:**
- WS per pass;
- the learning curve vs ao_s*;
- d 3–4 extrapolation;
- d 1 retention from step 0 (the askname foothold under a blind decoder starts at chance? step-0 ref).

## 4. OC (per seed)
- BLD-AO: P(FORMED | the shortcut was the blocker) ≈ 0.5; P(FORMED | answer-only formation is hard regardless) ≈ 0.15 (the §244 / §245 base rate 1/4, minus the DEC-STEP2 route).
- BLD-AUX: P(FORMED) ≈ 0.8.

## 5. Cost
- 4 × 12k at d ≤ 2 (P-2 realised ~$0.6?) ≈ $2.4; guard $3.5. Credit $25.7.
- **G-D:**
  - default-off bitwise;
  - blind dec tokens contain no problem-text token (assert per item);
  - eval / train tokenisation identical under the flag;
  - gradflow (answer CE reaches pass 1 and pass 2 write params);
  - step-0 refs (askname under the blind decoder, d 1 / d 2).
