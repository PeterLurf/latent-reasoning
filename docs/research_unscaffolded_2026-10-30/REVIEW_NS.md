# REVIEW_NS: red team of NS_draft.md (Opus subagent, 2026-10-30). Verdict: GO-with-fixes (3 blockers). Theory: $0 stage first (Note 251-15)

1. **BLOCKER (scope):** the plan is still recoverable from the text.
   - topo_shuffle (nested_words.py:247) puts op sentences in chain order in 1.000 of 1,000 REG items at each of d 2-4. A counter is enough, so the "entity pointer" claim overstates.
   - The aux target "<S_t> has <v_t> <u>." supervises step t's entity name in training (inject_ow.py:70; inject_aw.py:285).
   - **Fixes:**
     - scope the claim as "no plan input at test; entity order supervised by the aux in training; text in chain order";
     - add TEST-PERM (statement order permuted against the deps; `_finish` / `OW.solve` handle any order);
     - run rcb on TEST-PERM ($0);
     - an NS-PERM arm trained on permuted order is the real north-star test.
2. **Well-posed:** S_1 is the only op sentence with all inputs stated, so the next step is resolvable. 0.076 (d 3) / 0.177 (d 4) of items have an add_s partner ambiguity; report this stratum.
3. **BLOCKER (eval threading):**
   - cue_only / cue_wrong / c7 rh_input / read_mask / transfer cue_only build *named* cues via ST.cue_str (inject_aw_eval.py:160, 175, 740, 994, 948).
   - CUE-ONLY is vacuous under a generic cue (use NOT-DECODER); drop cue_wrong.
   - SWAP pairs share skeleton and position, so they test value carry, not position.
   - Put the PLAN-FREE assert inside passes().
4. **Warm start:** a bigger input shift than the REG-law onset break. Pre-state a $0 2k local pilot.
   - NS-COLD is mislabelled (aw_plaw2 is a named-cue lineage).
   - There is no aux for t > d (the draft was wrong); report K′ = d and 8.
5. **BLOCKER (censoring):**
   - add NOT-FORMED (collapse at d 1-2) and a STILL-RISING rule;
   - an OC block is required;
   - the realistic hypotheses are counter vs pointer, so P(pass | plan needed) = 0.03 understates.
   - Build:
     - CUE_LEN = 7 is asserted, but " What is the next value?" is 6 tokens;
     - the --cue choices and asserts at inject_aw.py:188 / 215 / 238 / 68;
     - dedupe the identical lower caches.
6. **$0 reads first:**
   - (a) rcb step-0 under the generic cue (WS per pass);
   - (b) a probe on rcb S_t for pass index and spine identity;
   - (c) rcb on TEST-PERM.
