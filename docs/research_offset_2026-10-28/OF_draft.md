# OFFSET draft: C2, extra passes compute past the trained step count (2026-10-28)

**Status:** draft for red team + Literature. Theory's choice under the user's standing rules (north star C4 = C2; Audit's lean is C2 first, since the hold is now stable, §240).

## 1. Why
- C2 (docs/NORTH_STAR.md): "accuracy rises with latent steps past the trained budget (K′ > K), in every competent seed".
- On the §240 models (trained d ≤ 4, think_k 8):
  - d 5 is solved at K′ = 4 (the decoder takes the last step, 237-4) and stays flat to K′ 128 (0.86 / 0.69);
  - d 6 is 0.35 / 0.37 (§237); d ≥ 7 is at chance.
  - So passes ≥ 5 never compute: in training, pass 5+ is always a hold pass (cue = S_d), so pass index predicts "hold".
- InjectAW has no pass embedding (REVIEW_D4 7). The index can only come from the memory's content / write count. This is the same position-prior family as 234-7 / 234-9, now on the step count.
- **H-IDX:** passes ≥ 5 hold because of a learned step-count prior. **H-MEM:** the memory can't keep more than ~5 states separable for the read.

## 2. Change (one, data only): offset augmentation
- Each training item gets k ~ U{0..4} leading idle passes. Pass t ≤ k is cued with S_0 ("How many <u> does <S_0> have?"; S_0 is given in the text), so it writes S_0's value again.
- Step j then happens at pass k + j, and holds follow.
- k + d ≤ 8 always holds (d ≤ 4, k ≤ 4), so think_k stays 8. Steps now occur at every pass index 1..8.
- **Aux targets** shift with the offset (value of S_0 at idle passes, v_j at pass k + j, v_d on holds). The hold-increment penalty (§240, λ 1.6) stays on, with holds counted from k + d + 3.
- **Runs (8k fine-tunes, one RTX 3090 each):**
  - of_s1 / of_s2 from lh_fh_s1 / lh_fh_s2;
  - of_ctrl_s1 (the same FT with k ≡ 0).
- Eval sets are unchanged: TEST2-DF with no offset (k = 0).

## 3. Readings (binding, per seed; PASS = both)
- **EXT-6:** acc(d 6, K′ 8) ≥ 0.70 and ≥ step-0 + 0.25. d 6 needs pass 5 to compute.
- **EXT-8:** acc(d 8, K′ 8) ≥ 0.50. d 8 needs 7 computing passes, inside the trained K 8.
- **C2-RISE:** for d ∈ {10, 11, 12}, which need 9–11 computing passes (past the trained K = 8): acc(d, K′ 16) − acc(d, K′ 8) ≥ 0.20 at some d, with acc(K′ 16) ≥ 0.40 there. Accuracy rises with latent steps past the trained budget.
- **NOLOSS:** §240's binding set on d 1–4 (HOLD-LONG3 at K′ 64 / 128; the §238 gates; retention vs lh_fh_*, drop > 0.05 named).
- **LP0** per the warm-start checklist: skip steps ≤ 1,000 and calibrate on the CTRL's windows.
- **CTRL:** if of_ctrl_s1 also passes EXT-6, the label is MORE-TRAINING (no lever credit).
- **Labels:** PASS (EXT-6, EXT-8, C2-RISE, NOLOSS) / EXT-ONLY (EXT-6 / 8 pass, C2-RISE fails) / IDX-FAIL (EXT-6 fails, suggesting H-MEM) / NOLOSS-FAIL / LP0-FAIL / MORE-TRAINING.
- **Descriptive:** acc(d, K′) for d 5–12 at K′ 4..16; WS per pass at d 6–8; ‖S‖ ratio.

## 4. OC (sketch)
- P(EXT-6 | H-IDX) ≈ 0.7; P(EXT-6 | H-MEM) ≈ 0.1. Prior 0.5.
- P(C2-RISE | EXT-8) ≈ 0.4 (past-budget passes are untrained at any index > 8).
- Overall P(PASS) ≈ 0.12; P(EXT-ONLY or better) ≈ 0.35.

## 5. Cost
- Three 8k fine-tunes ≈ $2.5 (§240 FH arms ~$0.7–0.9 each); guard $6.
- Running: realised ~$13.25 + $6 ≈ $19.3 ≤ $30.
