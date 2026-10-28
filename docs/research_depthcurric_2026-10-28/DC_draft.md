# DEPTH-CURRIC draft: does training on more depths make the step function extrapolate? (C2) (2026-10-28)

**Status:** draft for red team + Literature. Theory's choice under the user's standing rules (C2 line; pre-named after §241 STEPS / §242 NOT-NORM).

## 1. Why ($0 evidence)
- §241: d 6 fails at any text length (STEPS).
- §242: the norm is not the step counter (s2 clean).
- Audit's locus probes (4e6b4a94): at d 6, pass 5 stores and reads v_4 (≥ 0.94) but doesn't compute v_5 (u 0.48 / 0.22).
  - (a) There is a general compute decline past the trained depth; regular ops are 0.54–0.72 at pass 5.
  - (b) Back-references to an old state (src_j = 2) are at chance.
- So the step function trained on d ≤ 4 doesn't transfer to step 5+, and the extrapolation seen so far is +1 depth (the decoder's last step, 237-4).
- **Question:** does training on d ≤ 6 give a step function that extrapolates further (d 7–10), i.e. is the +1 limit a property of narrow training (H-NARROW) or of the architecture (H-ARCH)?

## 2. Arms (fine-tunes from §240's models; one RTX 3090 each)
- **DC:** dc_s1 / dc_s2 from lh_fh_s1 / lh_fh_s2. DF law, d ~ U{1..6}, think_k 10 (so d 6 still has hold passes), §240's hold-increment penalty (λ 1.6 from step 0, already formed), 8k steps, far pass T ~ U{16..40}.
- **CTRL:** dc_ctrl_s1 from lh_fh_s1, the same FT with d ~ U{1..4} and think_k 10. This separates "more training / think_k" from depth range.
- **Args diff vs §240's FH:** {d_max 6, think_k 10} (+ hold_start 0, init, name). It is two changes, disclosed. think_k is needed so d 6 has holds; the CTRL carries think_k 10 too, so the contrast with CTRL isolates the depth range.

## 3. Readings (per DC seed; binding)
- **EXTRAP:** acc(d 7, K′ 16) ≥ 0.60 and acc(d 8, K′ 16) ≥ 0.40, each ≥ CTRL s1 + 0.25 and ≥ the step-0 init + 0.25.
- **C2-RISE (past the trained budget):** for d ∈ {11, 12} (which need ≥ 10 computing passes, beyond think_k 10), acc(K′ 16) − acc(K′ 10) ≥ 0.10 with acc(K′ 16) ≥ 0.30, at some d.
- **ADDR (descriptive):** pass-t WS on back-referencing ops (src_j ≤ t − 3) at d 7–8.
- **NOLOSS:** HOLD-LONG3 at d 1–4 (K′ 64 / 128); §238's gates at d 3–4; retention vs lh_fh_* (drop > 0.05 named).
- **LP0 (warm-start checklist):** skip steps ≤ 1,000 after a re-fade, calibrate on CTRL s1's windows.
- **Norm guard:** one-sided vs CTRL (> 1.25).
- **Labels:**
  - PASS (EXTRAP + C2-RISE + NOLOSS + LP0);
  - EXTRAP-ONLY;
  - IN-RANGE-ONLY (d 5–6 ≥ 0.9 but EXTRAP fails, so the limit moves with training: H-ARCH-like);
  - NOLOSS-FAIL / LP0-FAIL / GAMED.
- Overall: both DC seeds needed for PASS; else SPLIT / FAIL.

## 4. OC (sketch)
- P(IN-RANGE | any) ≈ 0.8; P(EXTRAP | H-NARROW) ≈ 0.5, | H-ARCH ≈ 0.05; prior H-NARROW 0.4.
- P(C2-RISE | EXTRAP) ≈ 0.3.
- Overall P(PASS) ≈ 0.06; P(EXTRAP-ONLY or better) ≈ 0.2. The informative split is IN-RANGE-ONLY vs EXTRAP.

## 5. Cost
- Three 8k fine-tunes at think_k 10 (≈ 1.25× §240's step time) ≈ $3; guard $7.
- Running: realised ~$13.25 + $7 ≈ $20.3 ≤ $30.
