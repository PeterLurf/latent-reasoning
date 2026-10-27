# REVIEW_DF: red team of DF_draft.md (3ff7ac55), 2026-10-27

Opus subagent, read-only. Simulation: 20k DF programs per d 1–6 at q = 1/8, plus real Stream / pair draws.
Verdict: **GO-with-fixes.** Findings 1 and 2 block launch. All eight are adopted in DF_v1.md.

1. **BLOCKER: pass-4 OP-ALL+ is VOID by construction.** add_s_j0 at pass 4 on d 4 × 1,000 has n ≈ 42 (P(n < 50) ≈ 0.87), and both seeds would VOID together. Pass 3, d 3–4 pooled, is fine (every class ≥ 135; mul_s 171).
   - Fix: a separate d 4 × 3,000 class set, with n ≥ 50 as a build gate.
2. **BLOCKER: SWAP-MID d 3 t 2 never contains a product.** cf mode 'a' takes the donor's states 0..t (inject_ow.py:117), so a step-3 mul_s gives ca = donor final, and the keep rule (inject_aw_eval.py:368) drops it. 0 of 1,000 kept pairs have a product.
   - Fix: a product-enriched step-3 swap; keep iff own ≠ dfin and cb ∉ {own, dfin} (1,439 / 2,000); score follow_dfin vs follow_cb vs floor.
3. **FIX: the init already scores ~0.92 (d 3) / 0.84 (d 4) on DF.** ACC, CHAIN, LP5-4 and SWAP t1 pass with no training; only mul_s is informative.
   - Fix: aim the binding gates at the product subsets; add step-0 refs for both inits and CD; pre-state what step-0 mul_s ≥ 0.8 means.
4. **FIX: the derivation and labels.** add_s j ≥ 1 already combines two computed states, so H-READ2 = "a superposed read gives a sum but not a product". At pass 3 the partner is always S_1, so split pass 4 by j.
   - Fix: define READ2 bands and the pass they refer to; add a SWAP-FAIL label.
5. **FIX: P(mul_s | t) is 0 / 0.087 / 0.104 / 0.107 at t ≤ 2 / 3 / 4 / ≥ 5, independent of d.** I(op; t, d) = 0.040 bits on t ≥ 2, 0.0008 on t ≥ 3.
   - Fix: restate, or use per-t q (0.179 / 0.151 / 0.147).
   - No value leak; the text names the partner (nested_words.py:117).
6. **FIX: build sites hardcode 0/1:** inject_aw.py:135, ow_words.py:144, inject_ow.py:129, inject_aw_eval.py:202 (silently falls back), and score_rc's regen and CLASSES.
   - Fix: gate every stream; put a law tag in the set shas.
7. **FIX: references, censoring, OC.** The DIRECT refs on DF are needed at bs 100 and bs 50. Add a censoring section. P(VOID) ≈ 0.9 with finding 1 unfixed.
8. **NOTE:** the two inits are one lineage (aw_plaw2_s1), so this is 2 data orders, not 2 seeds. Fade and calibration as §237. Budget $15.8 ≤ $30.
