# PROTOCOL8 Phase 0 — results

Rules in `docs/PROTOCOL8_PHASE0.md` (3a9c05e). This file is filled in as each run is scored.

## P8-0a no-CoT: one-pass capacity (scored 2026-09-29 22:16 EDT)

Vast A4000 53405514. Data md5 matched the registration (`DATA_MD5` line) and `MASK_TEST PASS`. Training exited 0
(checkpoint_30). Results and checkpoint fetched with sha256 checks (`prosqa/res_p8_nocot/`,
`prosqa/ckpt/p8_nocot_checkpoint_30`, git-ignored). The host was destroyed and the destroy verified at 22:15.

Candidate accuracy a₁(m) on `p8_test` (chance 0.5):

| m | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 419 | 419 | 419 | 419 | 419 | 419 | 419 | 417 | 376 | 245 | 86 |
| epoch 10 | 0.568 | 0.544 | 0.532 | 0.484 | 0.482 | 0.480 | 0.475 | 0.535 | 0.511 | 0.494 | 0.430 |
| epoch 20 | 0.613 | 0.558 | 0.525 | 0.504 | 0.511 | 0.463 | 0.494 | 0.487 | 0.521 | 0.494 | 0.465 |
| **epoch 30** | **0.628** | **0.561** | **0.544** | **0.549** | **0.480** | **0.487** | **0.473** | 0.494 | 0.497 | 0.486 | 0.500 |

- **One-pass capacity c = −1:** a₁(0) = 0.628 < 0.90.
- **One-pass-hard strata H = {0, 1, 2, 3, 4, 5, 6}:** every trained stratum has a₁ < 0.75.
- **Saturation (descriptive):** a₁ is flat from epoch 10 to epoch 30 (overall 0.509 → 0.517 → 0.521).
- **Control (descriptive):** at epoch 30, m = 4 gives 0.575 and m = 8 gives 0.527.
- **By answer-path length (epoch 30):** 0.653 at depth 3, 0.597 at 4, then 0.46–0.55 from depth 5 on.
- **Instrument check:** the evaluator's format matches Coconut's no-CoT training format (question + "\n", then
  "### answer"; no latent markers when `no_cot`). The in-run exact-match validation agrees (43–48 / 100 over epochs
  15–18).

**Reading** (revised by amendment 1). What this measures is **no-CoT learnability** (answer labels only), not one-pass
capacity: Coconut's curriculum can teach one-pass solutions, as in addendum Q. One-pass capacity inside the Coconut
model (c_model, F1b) comes from P8-0b's K′ = 0 vs K′ = 10 sweep.
- A GPT-2 trained to answer in one pass is below the 0.75 bar at every trained depth. It is above chance at m ≤ 3
  (0.63 / 0.56 / 0.54 / 0.55, n = 419 each, SE ≈ 0.024) and near chance from m = 4 on (0.47–0.49). Even m = 0 (the
  chains absent) reaches only 0.63, versus ~0.77 for native ProsQA in the Coconut paper, which trains on native
  problems only. (Wording corrected 2026-09-29 after the audit; the first commit subject said "at chance".)
- Rule (i) of the registration ("H empty → deepen the format") does not apply. Whether the training range is
  above one-pass capacity for a curriculum-trained model is decided by F1b.
- Whether Coconut learns H (F1) is decided by P8-0b, due ~05:30–06:00 Wed.

## P8-0b Coconut: feasibility (scored 2026-09-30 09:33 EDT, from `prosqa/res_p8_coconut/coco_test_e48.json`, sha-checked)

Vast 3090 53405925: training exited 0 (checkpoint_48); ~28 min per epoch in the late stages. The other evaluation
files (e44, ctl, CoT reference) are still running on the host.

| Stratum (trained m = 0–6, pooled, n = 2933) | Accuracy |
|---|---|
| aᴄ at K′ = 10 (the trained budget) | **0.656** |
| a₁ (no-CoT, P8-0a) | 0.532 |
| aᴄ at K′ = 0 | 0.507 |

- **F1 not met:** 0.656 < 0.75. The margin over a₁ is 0.124.
- **F1b not met:** 0.656 < 0.75. The margin over K′ = 0 is 0.149, against a bar of 0.15.
- **c_model = −1:** K′ = 0 is at chance at every m. The answer mask leaves no usable one-pass route, so the answer
  depends on the thoughts.
- **By m:** at K′ = 10, 0.82 / 0.76 / 0.73 / 0.66 / 0.59 / 0.50 / 0.52 for m = 0..6. Gains saturate by K′ ≈ 3; K′ > 10
  adds nothing.
- **By answer depth, K′ = 10:** 0.80 at depths 3–4, 0.71 at 6, 0.55 at 8, and chance from 10 on.
- **Beyond training:** m = 7–10 are at chance for every K′.
- **Rule outcome (iii):** Coconut does not learn the deep problems in this format. PROTOCOL8's primary outcome is
  whether anytime or random-tight training makes the deep trained depths learnable (in distribution). Extrapolation
  is secondary.

## Deviations from the registration (for the write-up's deviation table)

| Plan | Change | When, relative to data | Reason | Impact |
|---|---|---|---|---|
| F1 (Coconut vs a₁ on H) decides latent use, with H from P8-0a | Amendment 1 (ecbec6f, 22:17:01) adds F1b: latent use inside the Coconut model (aᴄ(K′=10) − aᴄ(K′=0) on H), with c_model; PROTOCOL8's latent-use strata move from H to above c_model | 1 min after P8-0a was scored; before any P8-0b test outcome | Theory's caveat: no-CoT at chance measures learnability, not one-pass capacity (addendum Q: 0.976 at K′ = 0) | Changes which strata PROTOCOL8 scores latent use on. F1 is still reported. The change is informed by the P8-0a outcome, so it's listed here |
