# PROTOCOL4 results (compiled 2026-09-25)

Full write-up with figures: `report/protocol4_report.html` (also `.pdf`, `.md`).

Sources: `results/protocol4_confirm.json` (`python -m latent.protocol4_confirm`) and
`results/protocol4_variants.json` (`PYTHONPATH=. python scratch/score_p4_variants.py`, PROTOCOL3 readings
applied to the addendum-C variant seeds). Gate = addendum J: delayed bal-acc ≥ 0.85 and depth-matched
5-hop ≥ 0.60. "Extrapolates" = distance-7 bal-acc ≥ 0.60 with bootstrap lower bound > 0.55.

## Verdicts

| Prediction | Runs scored | Verdict |
|---|---|---|
| **P19** no step embedding → extrapolates at K′=12 | 5 seeds; competent: nostep_s2 (d7 0.523), nostep_s4 (0.514); gate failures: s1, s3, s5 | **Not replicated** (no competent run extrapolates); contingency exhausted |
| **P20** K=8: E ≥ 4 and no extrapolation at K′=16 | budget_k8_s1: E 5, d7 0.701; budget_k8_s2 fails gate (stage 3, 0.823) | **Fails** (the one competent run extrapolates) |
| **P21** K=8 + no step emb (decisive cell) | s0, s1 competent but E 3 (s1 extrapolates, d7 0.708); s2, s3, s4 fail the gate | **Unresolved**: 5 seeds, 0 scorable; stopping rule reached |
| **P22** randk (K ∈ 4..8) → extrapolates at K′=16 | competent: randk_s0, randk_s2 (d7 0.487 / 0.569); randk_s1 ends at stage 4 → gate failure (d7 0.548) | **Fails** (none extrapolates) |
| **Addendum G** K=10: E ≥ 4 and no extrapolation | k10_s1 E 4, d7 0.495 ✔; k10_s2 E 1 ✘; k10_s3 E 2, d7 0.657 ✘ | 1 of 3 pass |
| **Addendum I/J** K=4 | budget_k4_s3 fails gate (matched-5 0.589) | Unscored |
| **P13** pre-LN feedback: β toward 1, competition persists | v3_rawfb_s1 competent; β 0.728 (inside 0.62–0.87); competition persists | **Fails** (β not outside the range) |
| **P12 / P17** 4-layer | v2_deep_s1 fails gate (0.823, stage 3) | Unscored. For reference: Gq W4 0.344 > ref min 0.282, but depth-3 Gq 0.002; frozen clock d7 0.483 |
| **Addendum H** wide thought, seed 2 | v1_wide_s2 fails gate (0.711, stuck at stage 2) | Unscored (targeted W8/W2 would be 0.68) |

## Per-run scorecard

| Run | Final stage | Attempts | Delayed | Matched-5 | Gate | E | d7 @K′12 | d7 @K′16 | Where |
|---|---|---|---|---|---|---|---|---|---|
| nostep_s1 | 3 | 1 | 0.801 | 0.578 | ✘ | 2 | 0.502 | – | local 2070 fp32 |
| nostep_k8_s1 | full | 2 | 0.913 | 0.741 | ✔ | 3 | 0.709 | 0.708 | local 3070 |
| nostep_k8_s2 | 2 | 1 | 0.707 | 0.531 | ✘ | – | 0.496 | 0.498 | local 3070 |
| budget_k8_s2 | 3 | 1 | 0.823 | 0.543 | ✘ | 2 | 0.513 | 0.513 | local 4060 |
| nostep_s4 | full | 2 | 0.903 | 0.698 | ✔ | 4 | 0.514 | 0.510 | local 2070 fp32 |
| nostep_k8_s3 | 3 | 1 | 0.810 | 0.585 | ✘ | 2 | 0.501 | 0.501 | local 3070 |
| nostep_k8_s4 | full | 1 | 0.903 | 0.564 | ✘ | – | 0.497 | 0.497 | local 4060 |
| nostep_s5 | 4 | 2 | 0.854 | 0.528 | ✘ | – | 0.499 | – | local 2070 fp32 |
| nostep_s2 | full | 1 | 0.928 | 0.741 | ✔ | 3 | 0.523 | 0.528 | A4000 fp32 |
| nostep_s3 | 4 | 2 | 0.849 | 0.611 | ✘ | 2 | 0.512 | 0.518 | A4500 fp32 |
| budget_k8_s1 | full | 1 | 0.956 | 0.786 | ✔ | 5 | 0.699 | 0.701 | local 4060 |
| nostep_k8_s0 | full | 1 | 0.905 | 0.701 | ✔ | 3 | 0.519 | 0.530 | local 3070 |
| randk_s0 | full | 1 | 0.879 | 0.639 | ✔ | 4 | 0.488 | 0.487 | A4000 |
| randk_s1 | **4** | 1 | 0.876 | 0.619 | ✘* | 2 | 0.538 | 0.548 | A4500 |
| randk_s2 | full | 1 | 0.897 | 0.677 | ✔ | 3 | 0.563 | 0.569 | RTX 4000 Ada |
| budget_k4_s3 | 4 | 1 | 0.852 | 0.589 | ✘ | 1 | 0.511 | 0.509 | A4500 |
| budget_k10_s1 | full | 1 | 0.909 | 0.786 | ✔ | 4 | 0.495 | 0.495 | RTX 4000 Ada |
| budget_k10_s2 | full | 1 | 0.916 | 0.623 | ✔ | 1 | 0.503 | 0.509 | A4500 |
| budget_k10_s3 | full | 1 | 0.890 | 0.699 | ✔ | 2 | 0.621 | 0.657 | RTX 4000 Ada |
| v1_wide_s2 | 2 | 2 | 0.711 | 0.524 | ✘ | – | 0.501 | 0.501 | A4000 |
| v2_deep_s1 | 3 | 1 | 0.823 | 0.546 | ✘ | 5 | 0.491 | 0.491 | A4000 |
| v3_rawfb_s1 | full | 2 | 0.940 | 0.790 | ✔ | 3 | 0.668 | 0.667 | A4000 |

\* randk_s1 ended at curriculum stage 4, not full depth. Addendum D says "runs that end at 30k below full
depth are gate failures"; the first scorer passed it on the accuracy gate alone. Fixed 2026-09-25 (PROTOCOL5 plan
step 1): `latent/protocol4_confirm.py` now applies the full-depth clause, randk_s1 is a gate failure, and P22's
verdict is unchanged (s0 and s2 still fail). PROTOCOL4 totals: 11 of 22 runs competent, 4 of 11 extrapolate.

## Addendum E (evaluation-only confirmations; confirm split, 22 J-competent models)

| Prediction | Pass | Reading | Notes |
|---|---|---|---|
| **P23** blocking is step-locked | 9/22 | **fails** | diagonal drop holds in 18/22; off-diagonal limit fails in 13 |
| **P24** width changes amplitude, not clock | 9/22 | **fails** | per-edge W8/W2 0.13–0.32 in 22/22; clock shifts with W in 12 |
| **P25** budget split per frontier edge (primary, excluding s0–s3) | 14/18 | **mixed** (0.78 < 0.80) | s0–s3 replication on fresh graphs 4/4 |

Addendum-J re-score of earlier models (sensitivity only): only main_s5 changes (not competent under J).

## Open

* **All PROTOCOL4 training is done** (P19 and P21 contingencies exhausted at 5 seeds each).
* Stall pilots: base recipe 0/12 stalls; no-step recipe base 3/6, wu1000 1/6, lr5e-4 0/6, lr5e-4+wu1000 0/6.
* Cross-cutting pattern (exploratory, not a frozen prediction): the competent runs with d7 ≥ 0.60 at K′ = 12
  (budget_k8_s1, nostep_k8_s1, budget_k10_s3, v3_rawfb_s1) have E = 5, 3, 2, 3,
  and several with E ≥ 4 do not (randk_s0, budget_k10_s1). So far E does not separate extrapolators from
  non-extrapolators in either direction.
