# REVIEW_LH: red team of LH_draft.md (9181002f), 2026-10-28

Opus subagent, read-only. A step-0 CPU rollout of df_plaw4_s1 (32 items per d).
Verdict: **NO-GO as written; GO-with-fixes once #1–#3 are fixed.** All 11 findings are adopted in LH_v1.md.

1. **BLOCKER (dose):** at step 0, r_anchor = 3–129 (VB ≈ 0.024). So λ 1.6 gives an L_FP of about 5–200 against an answer CE of ≈ 2e-5, and with clipping at 1.0 every update would be pure penalty.
   - Fix: a dose rule or a normalised r; log the ratios. (Adopted as the increment form, which has VB's scale.)
2. **BLOCKER (mechanism):** S accumulates (deltamem.py:105), and the additive heads hold ~97–99% of ‖S‖. The anchor penalty demands a large negative write; VB's V was recomputed, not accumulated.
   - Fix: penalise the increment ‖S_T − sg S_{T−1}‖² / ‖sg S_{d+2}‖² (step 0 ≈ 0.03–0.11).
   - The S_{d+2} anchor is sound: a frozen S_{d+2} gives acc 1.000 for d 1–4.
3. **FIX (LP0-W):** LP0 reads the total-loss grad norm, so FH would fail by construction. Fix: log the plain-loss grad norm separately and gate on it.
4. **FIX (gaming):** the relative ratio can be cut by inflating ‖S_{d+2}‖. Fix: a ±25% norm guard (VB G3).
   - g ≈ 1, so the penalty is exposed from the start.
   - The RD CE is inert at T 16–40.
5. **FIX (fade):** fade restarts over 500 steps (inject_aw.py:300). Fix: start the penalty after step 500. γ is 1 throughout.
6. **FIX (ceiling):** there is no K′ 128 baseline, and s2 already passes §239. Fix: step-0 eval_long to K′ 128 on both inits, d 1 included; credit deltas only.
7. **FIX (LEVER):** one CTRL and a 0.03 margin are below the spread between data orders. Fix: K′ 128, margin ≥ 0.10, or add CTRL s2.
   - HOLD 0.02 is coherent (paired SE ≈ 0.003–0.005).
   - K′ 16 / 32 sit inside T_far, so only 64 / 128 should bind.
8. **FIX (labels):** add precedence, the overall rule, the C1-floor mapping, a clear LEVER, and LEVER / LP0 in the OC.
9. **FIX (Audit retention):** add SWAP-MID and LP5 on DF and REG (the REG refs need swap_kd / lp5h), and name any regression > 0.05 in the scorer.
10. **FIX (implementation):** far items = the first 16 per batch, T_far from a separate rng, a flags-off bitwise gate, one card / eval_bs.
11. **NOTE (cost):** an 8k run ≈ 1.55 h on a 3090, and far passes add +24–50%. Measure step time on the host first.
