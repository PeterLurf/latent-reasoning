# REVIEW_D4: red team of D4_draft.md (74a67eeb), 2026-10-26

Opus red-team subagent, read-only. One CPU simulation of the generator: 20k programs per cell, 2 threads.
**Verdict: GO-with-fixes.** Fixes 1, 2 and 5 are required before launch. All eight are adopted in D4_v1.md.

1. **FIX: CHAIN can't separate H-POS from H-CAP at pass 3.**
   - P-4's d4 pass 3 was 0.493, right at the 0.50 bar.
   - Under H-CAP, d4 pass 3 ≈ 0.464 + 0.536 × 0.09 ≈ 0.51, which already clears it.
   - Fix: add a binding gate, d4 pass-3 WS on non-mul_s step-3 items ≥ 0.80. Get the ops by joining sigs with regenerated TEST2. Recompute the OC.
   - → v1: OP-READ.
2. **FIX: where the skip coin is drawn.**
   - If it's drawn inside the restart loop (ow_words.py:48), rejected mul_s attempts redraw it, and the realised d3 share falls to 0.41.
   - Fix: draw it once per program, before the loop, only when p < 1. Log the realised shares.
3. **FIX: it isn't a single change at pass 4.**
   - At d4, step-4 mul_s falls 0.536 → 0.28. At d3, step-3 state reads fall 1.00 → 0.57.
   - TEST2 d4 step 4 is 53% mul_s, so a D4-PASS4 outcome would be confounded.
   - Fix: skip only at d = 3. Pooled P(mul_s | pass 3) still drops 0.73 → 0.48.
   - → v1: `mul_s_p3`.
4. **FIX: d3 values and lengths shift.**
   - Skipped items have S_1 or S_2 < 2 about 8% of the time each (the ow_words.py:53 rejection no longer applies).
   - The _v ops add a given sentence.
   - Fix: disclose both.
5. **FIX: "eval unchanged" has to be enforced.**
   - Calibration (inject_aw.py:280, sets mem.norms), memcheck (:206) and the probe / swap streams (inject_aw_eval.py:202, 299, 351) all take **OWKW.
   - Fix: put the kwarg on the training stream (inject_aw.py:150) only, never in OWKW. Gate on the set shas.
6. **FIX: the failure labels have holes.**
   - No label covers non-mul_s pass 3 between 0.55 and 0.80, or d3 between 0.90 and 0.95.
   - D4-PASS4 doesn't say which items it means.
   - The labels have no precedence.
   - The d3 ≥ 0.95 prediction doesn't separate the hypotheses (TEST2 d3 is all mul_s).
7. **NOTE: the diagnosis holds.**
   - The generator facts check out: d4 step 3 is mul_s 0.464; each regular op is ≈ 0.075.
   - There is no pass embedding (inject_aw.py:45–54).
   - Audit's oracle counts against H-CAP: pass 3 is 0.844 and pass 4 is 1.000 with values given in the text. With the real memory, pass 3 copies v_{t−1} on 0.49 of items.
8. **NOTE: build details.**
   - score_aw4.py:20 hardcodes the run names; it needs a runs option.
   - Do the by-op split in the scorer, not by changing the eval code Audit already checked (G-D).
   - In C1, the shortcut's d3 training accuracy is ≈ 0.5 + 0.5 × 0.09.
