# REVIEW_OF: red team of OF_draft.md (4da7f002), 2026-10-28

Opus subagent, read-only. $0 CPU checks on lh_fh_s1 / s2, first 100 TEST2-DF items per d (scratchpad of_diag.py / of_s1.log / of_k0.log).
Verdict: **NO-GO.** OFFSET is withdrawn (Note 241-0); the $0 length / step-count split (§241) runs first.

1. **BLOCKER: the diagnosis is contradicted.**
   - Pass 5 does compute: WS at pass 5 is 0.83 / 0.60 / 0.32 at d 5 / 6 / 7 (s1), and d 6 goes 0.024 at K′ 4 → 0.408 at K′ 5.
   - At the same pass, step and number of stored values, accuracy falls with d, which points to the text (length / position).
   - s2's pass 4, a trained index, is only 0.79 / 0.69 / 0.71 at d 5 / 6 / 7 vs 0.99 at d 4.
   - Training texts are ≤ 94 tokens at d 4; d 6 averages 98 and d 10–12 151–178. Cues c ≥ 5 are never trained (inject_st.py:59).
2. **BLOCKER: an S_0-cued pass is not idle.** It prints v_1 (0.97–1.00). A k-prefix of S_0 passes breaks d ≥ 3 (d 4 0.19 / 0.39 / 0.17 at k 1 / 2 / 4). There is no pass embedding, so the offset only shows through S.
3. **BLOCKER: not data-only.**
   - The aux code ties the aux index to the pass index (inject_aw.py:267–272; inject_ow.py:93–96).
   - The aux prompt shows the pass number (nested170.py:233).
   - hold_step uses one d per chunk (inject_aw.py:128–130), so per-item k breaks H and Ssv indexing.
4. **FIX:** the step-0 values come from the §240 inits: d 6 0.415 / 0.213, d 8 0.092 / 0.101.
5. **FIX:** C2-RISE needs run_long with d 5..12 and K′ 8..16 (acc_kprime stops at KMAX 14).
6. **FIX:** hold_start 0 on a penalty-trained init; LP0 cap source for s2; shared lineage; re-time steps.
7. **FIX:** the OC at 2 seeds, censoring, and a clear combination rule.
