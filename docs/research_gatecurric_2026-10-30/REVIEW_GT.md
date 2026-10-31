# REVIEW_GT: red team of GT_draft.md (Opus subagent, 2026-10-30). Verdict: GO-with-fixes, leaning NO-GO on value. Theory: PARKED (Note 251-8)

1. **BLOCKER:** STABLE (score_ao.py:44-54) would flag a successful seed as UNSTABLE. When s moves 3 → 4, 90% of batches go to d 4 at chance, so the loss jumps from ~0.01 to ~2.2 and stays high for 1-2k steps. Fix if revived: a STABLE variant that skips 2k steps after each stage change.
2. **BLOCKER:** censoring text. At s = 3 rehearsal draws only from U{1, 2}, so d 4 is never trained unless s reaches 4. The draft wrongly said "seen as rehearsal".
3. **BLOCKER (substance):** Note 251-7's mechanism is only partly right.
   - The decoder shortcut is correct.
   - The write breaks at onset in every arm, not by slow erosion (aex0_s2 aux_vacc d 2 pass 2 0.13 at 1k; aex0_s1 loses pass 1; aex_s1 dips with aux; gn 7-9 in bin 1 in all four).
   - AEX0 already put 50% of steps on d >= 3, clipped at grad norm 5-9 with CE flat, and learned no d 3. Gating to 90% d 3 straight after the onset shock would amplify interference.
   - Adopted as Note 251-8.
4. **Minor:** not a true single change. The two-draw law changes rng consumption and data order. New dev sigs would change EXCL; use ED7 (inject_ow.py:157) instead, checked against TEST2.
5. **Gate logic:**
   - Backtrack-to-smallest-d can thrash and lands in the maximum-shortcut regime. Use one stage at a time, never below 3, with hysteresis (advance at 0.9, backtrack at 0.8) and a cap of 3 backtracks.
   - The gate cannot see pass-2 erosion, because d 2 stays high via the shortcut.
6. **Minor:** "d 3 at K' 1 <= 0.2" is vacuous (true in every model). Use ws by_d["3"]["2"] >= 0.9, acc_kprime d 3 "2" >= 0.9, and the time order in the log.
7. **OC:** the AEX0 evidence already predicts FAIL, so a FAIL carries little information. Add a thrash label and a 2k futility check.
8. **Cheaper first step:** a $0 local 2-3k pilot from aex0_s2 at 90% d 3, logging ws d 3 pass 2. Park GATE if the write does not recover.
9. **Minor:** pin --micro 2; gate eval at bs <= 32 under autocast; ±0.05 tolerance on the step-0 values on ED7.
