# LANG-HOLD v1 (2026-10-28)

**Status:**
- v1 = LH_draft.md (9181002f) rewritten after the red team (REVIEW_LH.md: NO-GO as written, GO with fixes 1–3; all 11 findings adopted). Where they differ, this file wins.
- Audit approved the C6 step in principle, with a retention row (§232 lesson) and an init-calibrated LP0.
- Theory's choice under the user's standing rules.

## 1. Why the draft's transfer was wrong (fixes 1, 2)
- The language loop's state accumulates: S_a += φ(k) vᵀ (deltamem.py:105). The additive heads hold ~97–99% of ‖S‖.
- VB's state was recomputed every pass (FH_v1.md:6-13). So VB's anchor penalty ‖S_T − sg S_{n+2}‖² would here demand one large negative write.
- At step 0, r_anchor = 3–129 (VB ≈ 0.024), and with clipping at 1.0 every update would be pure penalty.
- **The correct analogue penalises the hold increment.** With the read depending only on S (the decoder reads only S), a frozen S_{d+2} gives acc = 1.000 at every K′ for d 1–4 (red-team rollout; eval_acc acc_kprime at d+2). That is this domain's C8-style oracle.
- **[A1]** If every hold write ΔS_t = S_t − S_{t−1}, t > d + 2, is ≈ 0, then S_K′ ≈ S_{d+2} for all K′, and acc(K′) ≈ acc(d + 2) = 1.000 [derived from the oracle].
- **[A2]** The measured drift is ‖S_K′‖ / ‖S_8‖ ≈ K′ / 8 (Note 239-2). That is a near-constant increment per hold pass, so driving the increment to 0 removes the drift's source.

## 2. Lever
- **HOLD-INC:** L_inc = λ · mean over hold passes t ∈ {d + 3, …, 8} ∪ {T_far} of ‖S_t − sg S_{t−1}‖² / ‖sg S_{d+2}‖².
  - The in-graph hold passes are already in the training rollout. The far pass gives states beyond the trained range.
  - Step-0 scale of the increment ratio: 0.03–0.11 (red team), VB's L_FP scale, so **λ = 1.6 fixed** (§230).
  - The ratios are logged per bin (230-1 (1) practice).
- **Far pass:** the first 16 items of each batch. T_far ~ U{16..40} from a separate rng, one draw per step. No-grad prefix to T_far − 1; the grad pass at T_far contributes to L_inc only (the RD CE is inert at T ≤ 40: acc ≥ 0.998 there, fix 4).
- **Penalty starts at step 500**, after the fade ramp (fade restarts 0 → 1 over 500 steps; fix 5).
- **Norm guard (fix 4):** ‖S_8‖ and ‖S_{d+2}‖ (batch means) must stay within ±25% of the init's values every 1k steps (VB's G3), or the run is labelled GAMED. This blocks cutting the relative ratio by inflating S_{d+2}.
- **Arms (8k fine-tunes, §238 recipe otherwise, one RTX 3090 each):**
  - lh_fh_s1 / lh_fh_s2 (HOLD-INC), from df_plaw4_s1 / s2;
  - lh_ctrl_s1 / lh_ctrl_s2 (same FT, no penalty, no far pass; fix 7), from the same inits.
  - A flags-off bitwise gate keeps CTRL paired.

## 3. Readings (theory/score_lh.py, fixtures before output)
- **Evals:** eval_long_k128.json at K′ 8 / 16 / 32 / 64 / 128, d 1–5, on TEST2-DF, one card / eval_bs per model (239-1 (2)). Run on all four arms and on both inits (step 0, fix 6).
  - DF parts (acc, ws, opclass, swap_prod, swap_kd, lp5h, cue_only) on the four arms.
  - REG refs <run>_reg (acc, ws, swap_kd, lp5h) on the two FH arms.
- **Binding, per FH seed:**
  - **HOLD-LONG3:** acc(K′) ≥ acc(8) − 0.02 at K′ 64 and 128, d 1–4. K′ 16 / 32 fall inside T_far and are descriptive.
  - **NOLOSS:** the §238 gates on DF at their thresholds (PROD-ACC, CHAIN, OP-ALL+, PROD-SWAP, CUE-ONLY), plus the C1 floor (acc at K′ 0 / 1 ≤ CD-AN + 0.05, d 3–4).
  - **RETENTION (Audit, fix 9):** OP-ALL, PROD-SWAP, SWAP-MID d 3 t 1 and LP5 (lp5h d 3–4) on DF vs the init's values, and OP-ALL, SWAP-MID and LP5 on REG vs §237's rcb values. Any drop > 0.05 is a NOLOSS failure, and the gate is named.
  - **LP0-W (fix 3):** gated on the plain-loss grad norm (gn_plain_mean100, logged separately). Cap = max(2 × the init's windows 3–10, 5.0); every window ≤ cap.
  - **NORM GUARD** as above.
- **LEVER (fix 7):** for s1 (the drifting order), min over d of acc(128) for FH minus the same for CTRL s1 ≥ 0.10. For s2: FH ≥ CTRL s2 − 0.02 at K′ 128 (no harm; ceiling).
- **Labels per FH seed, first match:**
  1. GAMED;
  2. LP0-FAIL;
  3. NOLOSS-FAIL (gate named);
  4. HOLD-FAIL;
  5. PASS.
- **Overall:**
  - PASS = both FH seeds PASS ∧ LEVER;
  - MORE-TRAINING-SUFFICES = CTRL s1 also passes HOLD-LONG3 (then LEVER can't hold and the lever is not credited);
  - otherwise SPLIT (one seed PASS) or FAIL.
- **Descriptive:**
  - ‖S_K′‖ / ‖S_8‖ (prediction ≤ 1.5 at 128) and cos(S_128, S_8);
  - the increment ratio per bin;
  - d 5 across K′;
  - step-0 deltas;
  - CTRL arms' full table.

## 4. OC
- Per seed P(HOLD-LONG3 | the increment lever works) ≈ 0.8; P(NOLOSS incl. retention | works) ≈ 0.75 (§232 erosion precedent); P(LEVER | works) ≈ 0.8.
- P(PASS | doesn't work) ≈ 0.03 (it needs HOLD at 128 in both FH seeds while CTRL s1 fails).
- Prior "works" ≈ 0.5, so overall P(PASS) ≈ 0.5 × 0.8² × 0.75² × 0.8 ≈ 0.14. Including MORE-TRAINING-SUFFICES as an informative outcome: P(informative) ≈ 0.6.

## 5. Censoring
As §238 / §239: a crash means resume (dedupe); missing parts mean pending, FAIL at --final. A missing step-0 K′ 128 ref means pending.

## 6. Cost (fix 11)
- A plain 8k run ≈ 1.55 h on a 3090; the far branch adds +25–50% (step time is measured on the host before admitting).
- Four runs ≈ 4 × 2 h × $0.24/h ≈ $1.9. Admit ≤ $2.6, guard $6.
- Running: realised $10.56 + $6 = $16.6 ≤ $30.
- Step-0 K′ 128 evals: $0, local (~15 min per model).
