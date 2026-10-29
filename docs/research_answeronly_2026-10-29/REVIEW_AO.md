# REVIEW_AO: red team of AO_draft.md (c6ba5b25), 2026-10-29

Opus subagent, read-only (scratchpad stable_chk.py).
Verdict: **GO-with-fixes** (1 and 3–6 before freezing). All adopted in AO_v1.md.

1. **BLOCKER:** `--aux_step 0` trips asserts (inject_aw.py:207, inject_ow.py:211). Use `--aux_step 1 --aux_fade 0,0`: lam_at is 0 at every step ≥ 1, the args diff is {aux_fade}, and nothing else reads aux_step.
2. **FIX:** the gradient path is misdescribed. There is no detach. dec_logits read S_K, which holds pass 1's write directly (inject_ow.py:85).
3. **FIX:** only LP2 depends on aux (score_st.py:83), so drop it. Restore MEAN-STATE (binding at P-2) and report CUE-WRONG. score_st requires lp2, so a new score_ao.py is needed.
4. **FIX (scope):**
   - The askname init was trained on per-entity values, which C4 excludes.
   - d = 1 items are v_1 labels (the §110b-1 qualifier).
   - d = 2 shows DEC-STEP2, not chaining.
   - Label it "AUX-FREE (askname init, d1 foothold, plan given; DEC-STEP2)".
5. **FIX (STABLE):**
   - The loss can't be non-finite (bad steps are skipped), so read train_summary.n_nonfinite.
   - Use deduplicated 100-step rows, add an absolute floor (P-2 ends at 6e-5), and define rises in the last 500 steps.
   - Precedence VOID > UNSTABLE > NOT-FORMED > ACC-ONLY.
6. **FIX (censoring):** P-2 formed late (d 2 train acc 0.19 at 6k → 0.89 at 7k). Pre-register a STILL-RISING single extension. Add the OC block.
7. **NOTE:** data are matched across seeds (fixed stream). The module changed since P-2, so use --args_ref, a 3090, and a bitwise 100-step replay against step 100 loss 3.3319097.
8. **NOTE:** the §227 teacher's weights are gone, and at 1.000 its intermediates equal the aux targets, so "distillation" is set in advance by P-2.
9. **NOTE:** cost ≈ $1.4 is plausible.
