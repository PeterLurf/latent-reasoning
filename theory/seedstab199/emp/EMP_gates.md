# EMP gates: delta-head contraction and memory channels, converged vs failed (2026-10-13, Theory subagent, CPU fp32)

Models (final model.pt): converged = s185_rms_s1, s185_rms_s2, s190_rmsnopass_s1, s184_reweight_s2; failed = s185_rms_s3, s185_den_s1, s182_hold_s1,
s188_rwwin_s1. pilot196_s3_small_4060 has NO model.pt (only logs / eval json), so it is not covered. Scope extension: alpha_pass at n 9..12 for the
13 models with path_indep (EMP_alpha_pass.json).

Items: 160 test items (32 per n, n = 4..8, `boxes.nested170.generate`, numpy rng 19613) for (a)-(d); 48 extra (24 per n = 10, 12, rng 19614) for the
by-n gate table; 256 items (64 per n = 9..12, rng 19615) for alpha_pass. Clean trajectory K = 12 passes from M_0 = 0.
Checks: (i) gates were captured by wrapping `deltamem.write_seq` (the original computes the state; the wrapper recomputes k, v, alpha, beta from the same
u); replaying the delta recurrence with the captured values reproduces write_seq's S_delta to max relative error 2.9e-7 (every pass, every model).
(ii) Aux exact match uses the teacher-forced criterion (every target token incl. EOT is the full-vocabulary argmax), which equals nested170's
greedy `aux_greedy` exact match by induction; checked against `aux_greedy` on 24 items at passes 1/3/5/7 for s185_rms_s1 and s185_rms_s3: 96/96
agreement each (including s3's 0.79 / 0.42 passes).
Definitions: P_t = prod over real tokens of alpha (I - beta k k^T) (64 x 64, right-multiplied in token order, float64); mean log sv = log|det P_t| / 64
(analytic: sum_tokens [log alpha + log|1 - beta| / 64]); retention = ||S_{t-1} P_t||_F / ||S_{t-1}||_F (how much of the ACTUAL previous delta state
survives one pass open-loop); alpha_eff^L = exp(mean over items of sum_tokens log alpha).

Full tables: EMP_gates_tables.md (all per-pass / per-head numbers), EMP_alpha_pass_tables.md; numbers: EMP_gates.json, EMP_alpha_pass.json,
per-model emp_<run>.json / emp_alpha_pass[_v2]_<run>.json (incl. per-item correctness).

## Bottom line
- (1) The failure regime is more contractive on average, but it is not a clean "contractive delta-head regime": the isotropic decay alpha does not
  separate at all (rho ~ 0), one failed model (s188_rwwin_s1) is as non-contractive as the converged ones, and the gap is absent at passes 1..2. What
  separates is pass-dependence of the erase gate: converged models lower beta pass by pass (0.95 -> 0.40..0.60) and switch to a near-hold after the
  last reduction; failed models keep beta ~ 0.85..0.9 at every pass (s188 partly, 1.02 -> 0.73).
- (2) Delta ablation kills every model at every pass (decoders read the delta state). Additive ablation is what kills the FAILED models' late passes
  (think-read ablation: late-pass accuracy <= 0.08 in all 4 failed vs 0.22..0.88 converged; aux-read-only: failed lose 22-51 %, converged <= 8 %).
  The failed models route the step-to-step computation through the additive integrator; s185_rms_s1 / s2 run it almost entirely in the delta heads.

## Verdicts

**(1) Are failed models' delta passes more contractive?  Partly, and not in the way the hypothesis says.**
- Group means, passes 1..8, n 4..8: failed are more contractive (mean log sv -0.73 vs -0.57; retention 0.64 vs 0.73; |P|_F^2/64 0.63 vs 0.72), and every
  failed model sits below every converged one on mean log sv and retention, but with no margin: s188_rwwin_s1 is -0.610 / 0.710 vs the most
  contractive converged models at -0.607 (s190_rmsnopass_s1) / 0.714 (s185_rms_s1). s188 is, in effect, as non-contractive as a converged model.
- The difference is NOT present at the first passes: pass-1 mean log sv converged -0.71..-0.86 vs failed -0.71..-0.81; pass-2 retention 0.52..0.60 vs
  0.47..0.55 (overlapping). It opens with pass index because the converged models LOWER beta pass by pass (beta mean pass 1 -> 8: 0.95 -> 0.44,
  0.95 -> 0.40, 0.94 -> 0.60, 1.08 -> 0.59), so their delta map relaxes to a near-hold (retention 0.81..0.86 by pass 8, #sv < 0.5 falls from ~10 to ~2),
  with an extra jump at the completion pass t = n + 1 (s185_rms_s2 retention n = 4: 0.74 -> 0.87 at t = 5; n = 6: -> 0.86 at t = 7). Failed models keep
  beta ~ constant (s3 0.80 -> 0.89, den 0.94 -> 0.86, hold 0.92 -> 0.83; only s188 falls, 1.02 -> 0.73) and their retention rises less and plateaus
  at 0.61..0.70 by pass 8 (s188 0.79, close to the converged 0.81..0.86); no completion jump (<= +0.03). Since every pass sees the identical
  input, a pass-dependent gate can only come from the memory read: converged models' write gates track the computation's progress; failed models'
  gates are nearly pass-invariant (the same near-full overwrite along the keys, beta ~ 0.85..0.9, applied every pass).
- Inside the computation window the groups overlap: n = 8 items, passes 1..8: retention converged 0.667..0.705 vs failed 0.612..0.684; mean log sv
  -0.686..-0.792 vs -0.769..-0.972; alpha_eff^L 0.756..0.787 vs 0.692..0.839 (s188 the HIGHEST of all 8).
- The isotropic decay alpha does NOT separate. Over the 13 path_indep models at n 9..12, alpha_pass has rank correlation -0.09..+0.08 with
  delta_cos_final (and -0.17..-0.02 with final loss); s188 (failed) has the second-highest alpha_pass (0.83..0.84, after s190_rmsnopass_s2
  0.87..0.88) and s187_nopass_s1 (converged) the lowest (0.65..0.70). Mean log sv (dominated by the beta ~ 1 erasures, not alpha) does correlate:
  rho -0.47..-0.63 with delta_cos_final, -0.59..-0.74 with loss (retention: -0.30..-0.49 / -0.37..-0.45); at n = 12 only, the four failed models are the four most contractive (-1.08..-1.30 vs converged >= -1.05); at n 9..11 they overlap
  (s187_nopass_s1 -0.98..-1.04 sits among them).
- Not every contraction statistic agrees: the count of strong erasures (#sv < 0.5) does NOT separate (pass-mean: converged 7.5..8.7, failed 2.9..12.6;
  s188 and hold have the fewest of all 8), and at n >= 8 the converged mean is higher (10.8 vs 8.7 at n = 8).
- Within the RMS seed triple (same recipe), s3 is clearly the most contractive at every n (n = 8 retention 0.567 vs 0.678 / 0.673; alpha_eff^L 0.69 vs
  0.785 / 0.783; one head with alpha_eff^L 0.25 vs 0.40 / 0.39; beta in [0.8, 1.2] on 35 % vs 27 % / 25 % of tokens, pass-mean).

**(2) Which channel carries the computation?  The delta channel is indispensable in all 8 models; the additive channel is what separates them.**
- Any delta ablation (all passes, think reads only, or the aux read only) kills late passes in every model (late = passes 5..8: <= 0.06 everywhere).
  The aux decoder and the answer decoder read the current expression / answer from the delta state in all models (delta dec-only answer ablation:
  0.09..0.21 = chance-ish).
- Additive ablation separates the groups. Late-pass (5..8) aux exact match, pooled:

| run | grp | clean | add, all | add, think reads only | add, aux read only | delta, think reads only |
| --- | --- | --- | --- | --- | --- | --- |
| s185_rms_s1 | conv | 1.000 | 0.875 | 0.878 | 0.984 | 0.013 |
| s185_rms_s2 | conv | 1.000 | 0.791 | 0.759 | 1.000 | 0.025 |
| s190_rmsnopass_s1 | conv | 1.000 | 0.244 | 0.216 | 0.963 | 0.000 |
| s184_reweight_s2 | conv | 1.000 | 0.309 | 0.309 | 0.925 | 0.062 |
| s185_rms_s3 | FAIL | 0.659 | 0.066 | 0.019 | 0.512 | 0.000 |
| s185_den_s1 | FAIL | 0.753 | 0.147 | 0.084 | 0.416 | 0.000 |
| s182_hold_s1 | FAIL | 0.684 | 0.041 | 0.006 | 0.372 | 0.003 |
| s188_rwwin_s1 | FAIL | 0.722 | 0.075 | 0.009 | 0.353 | 0.000 |

  In failed models, removing the additive read from the think passes destroys the late passes (<= 0.08) and removing it from the aux read alone costs
  22-51 % of the clean accuracy; converged models keep 0.22..0.88 without the additive think reads and lose <= 8 % without the additive aux read.
  So failed models lean on the additive INTEGRATOR (no decay; relative increment per pass ~ 1/t: 0.83 -> 0.10 for s3) for both the cross-pass
  computation and the readout; converged models carry it in the delta heads (fully in s185_rms_s1 / s2, partly in s190 / s184, which also need the
  additive think reads late).
- First cross-pass step (aux at pass 2): delta think-read ablation leaves s3 0.86 and den 0.87 (insensitive!) vs converged 0.24..0.38; additive
  think-read ablation leaves s3 0.14 and den 0.63 vs converged 0.87..0.97. s3 / den route even the first step through the additive channel;
  hold / rwwin behave like converged models at pass 2 (delta 0.32 / 0.18, add 0.82 / 0.89), but without the additive think reads they too collapse
  by pass 4..5 (0.06 / 0.24 at pass 4, 0.02 / 0.02 at pass 5).

**(d) Increments.** Additive relative increments fall ~1/t in every model (pass 2 -> 12: 0.8..1.0 -> 0.10..0.12; failed slightly smaller). Delta
increments fall from 0.7..1.0 to ~0.06 by pass 12 in every model (cos(S_{t-1}, S_t) 0.997..0.998 at pass 12): every model's delta state approaches a
fixed point; failed models are not distinct here except smaller early increments (pass 2: 0.72..0.77 vs 0.83..1.03).

## Key tables

Pass-mean (1..8) open-loop delta map, items n 4..8:

| run | grp | mean log sv | top sv | #sv<0.5 | P_F^2/64 | retention | beta mean | beta pass 1 -> 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s185_rms_s1 | conv | -0.549 | 0.852 | 8.30 | 0.721 | 0.714 | 0.663 | 0.95 -> 0.44 |
| s185_rms_s2 | conv | -0.524 | 0.858 | 8.65 | 0.729 | 0.721 | 0.625 | 0.95 -> 0.40 |
| s190_rmsnopass_s1 | conv | -0.607 | 0.876 | 7.79 | 0.759 | 0.749 | 0.769 | 0.94 -> 0.60 |
| s184_reweight_s2 | conv | -0.595 | 0.835 | 7.46 | 0.686 | 0.724 | 0.775 | 1.08 -> 0.59 |
| s185_rms_s3 | FAIL | -0.833 | 0.762 | 12.63 | 0.586 | 0.575 | 0.886 | 0.80 -> 0.89 |
| s185_den_s1 | FAIL | -0.765 | 0.800 | 10.63 | 0.632 | 0.635 | 0.913 | 0.94 -> 0.86 |
| s182_hold_s1 | FAIL | -0.707 | 0.803 | 6.16 | 0.609 | 0.644 | 0.891 | 0.92 -> 0.83 |
| s188_rwwin_s1 | FAIL | -0.610 | 0.865 | 2.87 | 0.699 | 0.710 | 0.863 | 1.02 -> 0.73 |

By n (Audit request; mean over passes 1..8; n 10 / 12 from 24 items each):

| quantity | group | n=4 | n=6 | n=8 | n=10 | n=12 |
| --- | --- | --- | --- | --- | --- | --- |
| alpha_eff^L | conv mean (range) | 0.880 (0.868-0.895) | 0.827 | 0.778 (0.756-0.787) | 0.756 | 0.729 (0.668-0.755) |
| alpha_eff^L | fail mean (range) | 0.800 (0.716-0.884) | 0.780 | 0.763 (0.692-0.839) | 0.759 | 0.738 (0.677-0.816) |
| mean log sv | conv | -0.410 | -0.566 | -0.742 | -0.919 | -1.077 |
| mean log sv | fail | -0.581 | -0.733 | -0.877 | -1.043 | -1.203 |
| top sv | conv / fail | 0.898 / 0.827 | 0.856 / 0.807 | 0.817 / 0.789 | 0.800 / 0.781 | 0.780 / 0.763 |
| #sv < 0.5 | conv / fail | 4.99 / 7.36 | 8.00 / 8.10 | 10.82 / 8.65 | 12.37 / 8.77 | 13.53 / 9.24 |
| retention | conv / fail | 0.777 / 0.658 | 0.728 / 0.641 | 0.681 / 0.620 | 0.657 / 0.610 | 0.633 / 0.590 |
| token-mean alpha | conv / fail | 0.9917 / 0.9856 | 0.9913 / 0.9885 | 0.9910 / 0.9900 | 0.9920 / 0.9919 | 0.9923 / 0.9924 |

Scaling with n: alpha_eff^L falls only mildly with n (converged 0.88 -> 0.73 from n 4 to 12, failed 0.80 -> 0.74) although L grows 14.5 -> 38 tokens:
token-mean alpha RISES with n, i.e. the decay is concentrated on a near-fixed set of tokens per item, not a per-token constant. Mean log sv falls ~linearly
with n in both groups (more rank-1 erasures per pass); the per-n gap between the group means is 0.12..0.17 at every n.

## Scope extension: alpha_pass and P at n 9..12, 13 path_indep models (64 items per n; passes 1..n; EMP_alpha_pass.json [model][n])

alpha_pass = exp(mean over items, the 6 delta heads and passes 1..n of sum_tokens log alpha); mean log sv over passes 1..n; retention over passes 2..n.
Sorted by delta_cos_final (eval.json s182.path_indep). FAIL = final-1k loss >= 0.56.

| run | grp | loss | delta_cos | alpha_pass n9 / n10 / n11 / n12 | mean log sv n9 / n12 | retention n9 / n12 |
| --- | --- | --- | --- | --- | --- | --- |
| s184_reweight_s2 | conv | 0.003 | 0.436 | 0.743 / 0.743 / 0.736 / 0.737 | -0.832 / -0.930 | 0.670 / 0.686 |
| s182_window_s2 | conv | 0.001 | 0.470 | 0.753 / 0.765 / 0.764 / 0.777 | -0.726 / -0.733 | 0.665 / 0.713 |
| s187_nopass_s1 | conv | 0.014 | 0.472 | 0.654 / 0.671 / 0.677 / 0.701 | -0.977 / -1.054 | 0.622 / 0.664 |
| s187_nopass_s2 | conv | 0.005 | 0.474 | 0.713 / 0.727 / 0.726 / 0.739 | -0.809 / -0.856 | 0.631 / 0.670 |
| s190_rmsnopass_s1 | conv | 0.002 | 0.480 | 0.783 / 0.790 / 0.782 / 0.794 | -0.852 / -0.997 | 0.708 / 0.721 |
| s190_rmsnopass_s2 | conv | 0.228 | 0.481 | 0.872 / 0.872 / 0.871 / 0.880 | -0.696 / -0.861 | 0.745 / 0.762 |
| s182_window_s1 | conv | 0.239 | 0.529 | 0.811 / 0.816 / 0.821 / 0.830 | -0.776 / -0.874 | 0.700 / 0.733 |
| s185_rms_s2 | conv | 0.003 | 0.588 | 0.778 / 0.779 / 0.778 / 0.780 | -0.732 / -0.821 | 0.677 / 0.681 |
| s185_rms_s1 | conv | 0.006 | 0.625 | 0.778 / 0.773 / 0.771 / 0.769 | -0.776 / -0.905 | 0.679 / 0.671 |
| s188_rwwin_s1 | FAIL | 0.901 | 0.763 | 0.836 / 0.835 / 0.834 / 0.832 | -0.859 / -1.079 | 0.691 / 0.707 |
| s182_hold_s1 | FAIL | 1.060 | 0.898 | 0.762 / 0.759 / 0.755 / 0.753 | -0.943 / -1.183 | 0.610 / 0.611 |
| s185_den_s1 | FAIL | 0.642 | 0.930 | 0.741 / 0.725 / 0.719 / 0.707 | -1.013 / -1.265 | 0.611 / 0.585 |
| s185_rms_s3 | FAIL | 1.031 | 0.978 | 0.685 / 0.670 / 0.658 / 0.659 | -1.051 / -1.301 | 0.565 / 0.547 |

Descriptive Spearman rho over the 13 models (vs delta_cos_final / vs final loss), n = 9 .. 12 (Theory owns the pre-registered test):

- alpha_pass: n9 +0.08 / -0.02, n10 -0.07 / -0.14, n11 -0.07 / -0.14, n12 -0.09 / -0.17
- mean log sv: n9 -0.47 / -0.59, n10 -0.57 / -0.65, n11 -0.58 / -0.70, n12 -0.63 / -0.74
- retention: n9 -0.30 / -0.37, n10 -0.38 / -0.42, n11 -0.41 / -0.40, n12 -0.49 / -0.45
- P_F^2/64: n9 -0.22 / -0.41, n10 -0.32 / -0.47, n11 -0.31 / -0.45, n12 -0.43 / -0.53

Reading: alpha_pass carries no rank information about path independence or loss (|rho| <= 0.17); the median log singular value of P equals
log alpha_pass to ~0.001 (the bulk of directions is touched only by alpha), so the contraction signal lives in the beta erasures, not in alpha.
Retention puts s3 / den / hold at the bottom (0.55..0.61) but s188 (0.69..0.71) inside the converged range (0.62..0.76); s187_nopass_s1 / s2
(converged, path-dependent) are the next most contractive (0.62..0.67).

## Caveats / what this does not show
- Final models only (as the path_indep table): the gate behaviour of a model that never learned the task is a consequence as much as a cause.
  The converged models' pass-dependent beta is itself a learned function of the memory read; the failed models lack it. Nothing here says whether
  constant gates are what blocks basin entry or what a stuck model ends up with.
- Open-loop P measures the delta map with the pass's own gates held fixed; the closed loop (the read changes the next pass's k / v / beta) is not in P.
  Open-loop retention compounds to <= 0.86^16 ~ 0.09 (converged, late passes) or ~0.6^16 ~ 3e-4 (failed) over the 16 path_indep passes, so the
  converged models' path dependence (delta cos 0.44..0.63) cannot come from the open-loop delta map: it must come from the closed loop and / or the
  additive integrator (never forgets) feeding the delta writes.
- mean log sv is dominated by tokens with beta ~ 1 (|1 - beta| -> 0 gives log -> -inf); retention / |P|_F^2 / top sv are more robust and give the same
  group ordering but smaller gaps. The 4-vs-4 group comparison has no statistical power beyond "every failed below every converged" (p = 1/70 one-sided
  for a pre-specified direction, but s188's margin is 0.003).
- Aux counts at late passes are small (pass 8: 32 items); items n 4..8 only, so pass-pooled numbers mix computation and post-completion passes.
- CPU fp32 vs the GPU bf16 trunk used in training/eval: clean aux accuracies agree in trend with eval.json's aux_acc_by_pass (s3 here 0.99 / 1.00 /
  0.99 / 0.92 / 0.84 / 0.67 / 0.44 / 0.38 on n 4..8; eval.json 0.99 / 1.00 / 0.96 / 0.87 / 0.75 / 0.59 / 0.41 / 0.32 on n 1..8).
