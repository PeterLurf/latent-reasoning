# PROTOCOL5 results (compiled 2026-09-25 ~19:05)

Sources: `results/protocol5_confirm.json` (`python -m latent.protocol5_confirm score`), `results/strict_clock.json`,
`results/p5_endact.json`, `results/extrap_predictors.json`, `results/stall_rule.json`. Predictions and definitions:
`docs/PROTOCOL5.md` (addenda A–C, all written before the data they score). Chronology: `docs/RESEARCH_LOG.md`.

## Verdicts

| Prediction | Evidence | Reading |
|---|---|---|
| **R1** lr 5e-4 removes late-stage stalls (more gate passes than base, ≥ 6/8) | gate: lr 5e-4 **7/8**, base **5/8**; one-sided Fisher p = 0.28. Excluding the two fp32 runs: 5/6 vs 5/8 | **holds** (not significant at n = 8) |
| **P26** end activity ≥ 0.25 ↔ extrapolation (prospective, 12 competent pilot runs) | < 0.25: 1 of 6 extrapolate; ≥ 0.25: 5 of 6; AUC 0.83 | **confirmed** |
| **P23′** step-locked blocking in single-step readers (22 PROTOCOL4 models) | 7/9 pass (other 13 models: 1/13) | **mixed** (0.78, bar 0.80) |
| **P24′** same clock across W in single-step readers | 7/9 pass (other 13 models: 2/13) | **mixed** (0.78) |
| P23′ / P24′ on competent pilot runs (fully prospective set) | 3/4 and 3/4 | mixed (n = 4) |

## Reliability pilot (addendum A)

| Run | Precision | Final stage | Delayed | Matched-5 | Matched-6 | Gate | d7 @K′12 | Extrapolates | End activity |
|---|---|---|---|---|---|---|---|---|---|
| rel_base_s20 | bf16 | full | 0.883 | 0.668 | 0.615 | ✔ | 0.514 | ✘ | 0.001 |
| rel_base_s21 | bf16 | **2** | 0.697 | 0.542 | 0.548 | ✘ | 0.499 | ✘ | 0.609 |
| rel_base_s22 | bf16 | **2** | 0.706 | 0.546 | 0.548 | ✘ | 0.496 | ✘ | 0.119 |
| rel_base_s23 | bf16 | **4** | 0.875 | 0.698 | 0.529 | ✘ | 0.712 | ✔ | 1.000 |
| rel_base_s24 | bf16 | full | 0.971 | 0.854 | 0.743 | ✔ | 0.588 | ✘ | 0.166 |
| rel_base_s25 | bf16 | full | 0.938 | 0.749 | 0.678 | ✔ | 0.713 | ✔ | 0.029 |
| rel_base_s26 | bf16 | full | 0.909 | 0.668 | 0.611 | ✔ | 0.519 | ✘ | 0.009 |
| rel_base_s27 | bf16 | full | 0.959 | 0.904 | 0.696 | ✔ | 0.777 | ✔ | 0.621 |
| rel_lr5_s20 | bf16 | full | 0.919 | 0.732 | 0.627 | ✔ | 0.627 | ✔ | 1.000 |
| rel_lr5_s21 | fp32 | full | 0.906 | 0.684 | 0.588 | ✔ | 0.583 | ✘ | 0.930 |
| rel_lr5_s22 | bf16 | full | 0.919 | 0.745 | 0.683 | ✔ | 0.700 | ✔ | 0.629 |
| rel_lr5_s23 | fp32 | full | 0.900 | 0.643 | 0.586 | ✔ | 0.603 | ✔ | 0.829 |
| rel_lr5_s24 | bf16 | full | 0.961 | 0.787 | 0.696 | ✔ | 0.614 | ✔ | 1.000 |
| rel_lr5_s25 | bf16 | **4** | 0.875 | 0.758 | 0.540 | ✘ | 0.500 | ✘ | 0.018 |
| rel_lr5_s26 | bf16 | full | 0.884 | 0.621 | 0.555 | ✔ | 0.504 | ✘ | 0.013 |
| rel_lr5_s27 | bf16 | full | 0.899 | 0.637 | 0.590 | ✔ | 0.503 | ✘ | 0.123 |

* Every gate failure is a late-stage stall: each run that reached full depth also passed the accuracy clauses.
* **No stage-1 relaunch in any of the 16 runs** (the stage-1 pilot had 3/6 base-lr stalls at seeds 10–15, bf16).
* Accuracy cost of the lower LR, among full-depth runs: delayed 0.913 vs 0.932, matched-5 0.693 vs 0.769, matched-6 0.618
  vs 0.669 (lr 5e-4 vs base).
* Extrapolation among competent runs: lr 5e-4 4/7, base 2/5. End activity and gate outcome are recorded for failures
  too, but P26 is scored on competent runs only. rel_base_s23 (stage 4, gate failure) extrapolates (0.712) — like
  nostep_s0, which also ended at stage 4.

## P26 detail (addendum B)

* Low end activity (< 0.25): rel_base_s20, s24, s25, s26, rel_lr5_s26, s27 → one extrapolates (rel_base_s25, 0.713).
* High (≥ 0.25): rel_base_s27, rel_lr5_s20, s21, s22, s23, s24 → five extrapolate; rel_lr5_s21 misses narrowly (0.583,
  lower bound 0.566).
* Pooled with the 22 exploration models (not a pre-registered test): low 2 of 15 extrapolate, high 12 of 19.

## Strict clock (addendum C)

* Single-step readers (discovery qualification): s0, s1, s2, s3, v2_deep_s0, budget_k8_s0, nostep_s4, randk_s0,
  budget_k10_s1. Failures: budget_k8_s0 (P23 off-diagonal, P24 clock shift at W = 8), v2_deep_s0 (P23 control clause
  only, −0.0101 vs −0.01), main_s2 (P24 clock shift).
* Pilot single-step readers among competent runs: rel_base_s20, rel_base_s25, rel_base_s27, rel_lr5_s24. Failures:
  rel_lr5_s24 (P23: diagonal drop not ≤ −0.05 at every depth), rel_base_s25 (P24 clock shift).
* All sets together: P23′ 10/13, P24′ 10/13 (0.77).

## Step 4a: late-stage restart rule

A dwell limit T = 14 000 steps in stages 2–4 kills no attempt that later reached full depth and cuts expected steps per
full-depth run from 52 200 to 47 800 (~8%); shorter limits cost more than they save. Available as `--late T` in
`scratch/stage1_guard.py`; not a fix.

## Cost

RunPod 2026-09-25 total $4.40 (list-billing, may lag), of which ~$2.68 was PROTOCOL4; the pilot used a 4090 ($0.34/h,
3 runs), a secure A4000 ($0.25/h, 2 runs) and a community A4000 ($0.17/h, 2 runs). Every pod deleted after
checksum-verified fetch. Two secure pods billed above the $0.34/h cap were deleted within a minute, unused.
