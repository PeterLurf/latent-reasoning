# REVIEW_RC: red team of RC_draft.md (4035dcde), 2026-10-27

Opus subagent, read-only. Simulations: 20k REG programs per d (1..6), plus the pair laws.
Verdict: **GO-with-fixes.** Fixes 1, 4, 5, 6 and 7 are required before launch. All are adopted in RC_v1.md.

1. **FIX: [A2] "I(op; t, d) = 0 exact" is false.**
   - d 1 is forced mul_v (ow_words.py:56).
   - t 1 never draws add_s (:57-59).
   - mul is redrawn when S_{t-1} = 0 (:62).
   - Over all (t, d), I = 0.049 bits. For t ≥ 2 and d ≥ 2 it is 0.0003 bits, and that part is value-driven, not positional.
   - Fix: restate the claim with its scope and log the (t, d) op table.
2. **NOTE: REG is close to Markov.**
   - Only 7.5% (d 3) and 16.7% (d 4) of programs read a computed earlier state (add_s with j ≥ 1). add_s at t 2 always reads S_0.
   - Every REG program is affine (nested_words.py:218).
   - Fix: disclose this ("no cross-state product") and split add_s in OP-ALL by j.
3. **NOTE: the pair laws work under REG.**
   - d 3 t 1: 1000/1405; d 4 t 2: 1000/1474; 0 length mismatches.
   - d 3 t 2 is no longer empty (1000/1584), so add SWAP-MID at d 3, t 2 (a causal test of op reading at the last step).
   - No law in build_sets assumes mul_s.
4. **FIX: build.**
   - OW.test_set and pair_candidates (ow_words.py:152, 156) don't take the kwarg, but build_sets passes **OWKW (inject_ow.py:148).
   - State the law for each stream: calibration (:282, where 32 of 64 items are d 3–4), memcheck (:208), swap_pairs_d (eval :351).
   - Assert mul_s_p ∈ {0, 1}, and mul_s_p3 = 1 whenever mul_s_p < 1.
   - Gates:
     - flags-off bitwise, including the set functions;
     - zero mul_s in every REG set;
     - 0 sig overlap between training and every eval set;
     - new set shas in args.json;
     - new unused seeds;
     - the scorer regenerating the sets.
5. **FIX: the gates.**
   - The CUE-ONLY bar used s228_pdir2c_s1 on the old TEST2, so it needs a REG eval.
   - P(VOID) is missing from the OC (§235 went VOID).
   - The "both seeds" rule doesn't say what happens when a seed is VOID.
   - The labels have no full coverage and no precedence.
   - The OP-ALL bar of 0.50 equals H-CAP's ceiling, so use 0.80 with a partial band.
   - An op with n < 50 should give VOID, not be dropped.
6. **FIX: the OC.**
   - Pass 4 at d 4 failed in all three runs (WS 0.051 / 0.025 / 0.265), including the mul_s items.
   - Under REG, pass 4 is still trained only at d 4 (25% of steps).
   - Restore the "pass 4 lags" world: P(PASS) ≈ 0.2.
7. **FIX: references on the same sets.**
   - Evaluate aw_plaw4_s1, aw_plaw4_s2 and d4d_plaw4_s1 on the REG sets (ws, acc, cue_only, swap_kd) as the H-POS baseline.
   - REG models on P-4 TEST2: d 1–2 are comparable; d ≥ 3 is descriptive only (out of distribution).
