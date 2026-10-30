# REVIEW_FD: red team of FD_draft.md (162b027d), 2026-10-30

Opus subagent, read-only (scratchpad fd_check.py; templates appended to FRAMES in memory only).
Verdict: **GO-with-fixes** (blockers 1–2 first, both $0). All adopted in FD_v1.md.

1. **BLOCKER: the held-out set was partly a reworded copy.** By slot-masked trigram Jaccard:
   - 12 is frames 2 / 3 with "come to" (0.42);
   - 14 swaps the verb in trained or frame-6 syntax (0.33–0.50);
   - 15 is frame 7, which already transfers;
   - only 13 was new.
   - Fix: new constructions, plus a novelty gate of ≤ 0.20 vs frames 0–11 and 6–7. v1 rewrote 12 / 14 / 15; the max is 0.16.
2. **BLOCKER: no step-0 baseline.** Frame 7 was 1.000 untrained.
   - Fix: a $0 step-0 eval per frame; GEN binds only on frames failing at step 0 (≥ 2), else re-author.
3. **FIX (wording):**
   - Frame 9 add_c / sub_c imperatives read as changing A → "With {c} more / fewer {u}, {A} would have as many as {B}."
   - sub_c 14, add_v 15 and sub_v 8 reworded.
   - Argument orders all correct; no template leaks the answer.
   - The mod-11 wrap makes comparative verbs literally false (as in existing frames), so report the wrap share.
   - sub_v 10's "exceeds" is frame 7's verb → reworded in v1.
4. **NOTE (implementation):** appending to FRAMES is safe. Frame draws use only the frames tuple; 0 / 1,800 solve mismatches on frames 0–7.
   - Gates G1–G4 (set shas byte-identical, selfcheck, new set shas in EXCL, nested_words sha pre-declared, no in-flight importer).
   - inject_aw.py:223 hard-codes FRAME_TRAIN, so it needs a --train_frames arg.
5. **NOTE (design):**
   - DIV vs CTRL is paired: 3,000 / 3,000 identical programs, only the frames differ. Both replay §240's items (state this).
   - Add a TRAIN-NEW test set for MEMO, a 2-seed OC with censoring, and HOLD-LONG3 retention.
   - Fix the LP0 / penalty schedule.
