# LANG-HOLD draft: transfer §230's anchored far fine-tune to the language loop (C6) (2026-10-28)

**Status:** draft for red team + Literature. Theory's choice under the user's standing rules; C6 is next after C1.

## 1. Why (Note 239-2)
- The §238 models hold through K′ 32 but drift by K′ 64 in one data order (s1 d 2 0.941; §237's rcb_s1 d 2 0.817).
- ‖S_K′‖ / ‖S_8‖ grows ~linearly (2 / 4 / 8), and cos(S_64, S_8) falls to 0.69–0.95. Each hold pass adds to the additive DeltaMem state, so the hold is not a fixed point.
- In the synthetic VB line the same failure (216-15) was fixed by an anchored far pass. §224 FH plus §230 FH-FT at fixed λ 1.6 held to K′ 128 with no depth cost (s2 motivating, s1 confirming).
- This design transfers that recipe. It is an existing lever applied to a new domain, not a fix invented for this result.

## 2. Arms (fine-tunes from §238's models; one RTX 3090 each)
- **FH-FT:** per training step, a fraction ρ = 0.25 of items get one extra far pass at T_far ~ U{16..40}. The prefix runs to T_far − 1 with no grad; the grad pass is at T_far. The loss adds:
  - (i) answer CE read at T_far (RD);
  - (ii) λ · g · ‖S_{T_far} − sg S_{d+2}‖² / ‖sg S_{d+2}‖², with λ = 1.6 fixed (§230) and g = 1 if the item is answered correctly at K′ = d + 2.
  - The anchor S_{d+2} is the state two passes after the item's last step (VB used V_{n+2}).
  - Everything else is §238's recipe (DF law, d ~ U{1..4}, γ 1, think_k 8). 8k steps, lr as §238's FT, fresh optimizer.
  - Runs lh_fh_s1 (init df_plaw4_s1) and lh_fh_s2 (init df_plaw4_s2).
- **FT-CTRL:** the same 8k fine-tune with no far pass and no penalty. Run lh_ctrl_s1 (init df_plaw4_s1).
  - It separates "more training" from the lever. s1 is the drifting order, so it is the informative control.
- **Retention evals:** each run is re-evaluated on the §238 DF parts (acc, ws, opclass, swap_prod, cue_only) and on §237's REG sets.

## 3. Derivation (sketch)
- [A] On hold passes t > d, the write adds ΔS_t ≈ f(state, cue S_d) with no fixed point (measured: ‖S‖ linear in t).
- The penalty drives S_{T_far} → S_{d+2}, i.e. Σ_{t=d+3..T_far} ΔS_t → 0. The model can satisfy it either by writing ~0 on hold passes (a gate learned from "same cue as the last pass") or by cancelling writes.
- Either way S becomes a fixed point, and a read that worked at K′ = d + 2 keeps working.
- [Measured in VB] §230: H(128) 0.99 with no depth cost at λ 1.6.
- [Risk] The penalty could also suppress writes on in-range passes. g and the anchor at d + 2 restrict it to the hold, and NOLOSS gates check it.

## 4. Readings (binding, per FH seed; PASS = both)
- **HOLD-LONG2:** acc(K′) ≥ acc(8) − 0.02 at K′ 16 / 32 / 64 / 128, d 2–4, TEST2-DF (§239's eval_long, extended to 128).
- **NOLOSS (§238 gates on DF, same thresholds):** PROD-ACC ≥ 0.80, CHAIN, OP-ALL+ (every class ≥ 0.80), PROD-SWAP (follow_U ≥ 0.50 and ≥ floor + 0.30), CUE-ONLY.
- **C1 floor:** acc at K′ 0 / 1 ≤ CD-AN + 0.05 at d 3–4.
- **LP0-W (warm start, Audit 238-4):** cap = max(2 × max(init run's windows 3–10), 5.0); every FT window ≤ cap.
- **LEVER (vs FT-CTRL s1):** FH s1's min over d of acc(64) − acc(8) exceeds FT-CTRL's by ≥ 0.03, or FT-CTRL also passes HOLD-LONG2 (label "MORE-TRAINING SUFFICES").
- **Descriptive:** ‖S_K′‖ / ‖S_8‖ (prediction: ≤ 1.5 at 64), cos(S_64, S_8); d 5 at K′ 8..128; REG retention.
- **Labels:** PASS / NOLOSS-FAIL / HOLD-FAIL / MORE-TRAINING-SUFFICES / LP0-FAIL.

## 5. OC
- Per seed P(HOLD-LONG2 | recipe transfers) ≈ 0.8; P(NOLOSS | transfers) ≈ 0.85.
- P(PASS | the lever doesn't apply, e.g. the DeltaMem write ignores the anchor) ≈ 0.05. Prior "transfers" ≈ 0.55.
- Overall P(PASS, both seeds) ≈ 0.55 × 0.46 ≈ 0.25.

## 6. Censoring
As §238 / §239: a crash means resume (dedupe); missing parts mean pending, FAIL at --final.

## 7. Cost
- Three 8k fine-tunes with a ~25% far-pass overhead ≈ 3 × $0.45 = $1.4, plus local evals. Admit ≤ $1.8, guard $4.5.
- Running: realised $10.56 + $4.5 = $15.1 ≤ $30.
