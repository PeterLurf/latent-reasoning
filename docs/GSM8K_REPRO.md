# GSM8K track, step 1: reproduce public GPT-2 latent reasoners in bench (registered 2026-10-01 18:25 EDT, before any run)

**Why.** Before any new method is tested on GSM8K, the harness must reproduce the published GPT-2 latent reasoners on the
same items, with the no-reasoning baselines printed next to them (Theory's shortcut check `theory/gsm8k_shortcuts.py`,
Audit's request; user-approved via Theory's GSM8K track).

**Items.** `bench` task `gsm8k`: the openai `test.jsonl`, 1,319 problems. The published models were evaluated on
GSM8k-Aug's test file, which has the same 1,319 problems in the same order with identical final answers
(`docs/LIT_BENCH_VERIFY_2026-10-01.md`). Answers are compared after `norm_number`.

**Models (public checkpoints, unchanged).**

| tag | checkpoint | inference recipe |
|---|---|---|
| codi | `zen-E/CODI-gpt2` | CODI GPT-2 test recipe, as in `theory/codi_probe.py`: LoRA merged, question + `<bot>`, 6 latents, `<eot>`, greedy, answer after "The answer is:" |
| simcot_codi | `internlm/SIM_COT-GPT2-CODI` | the same recipe |
| simcot_coconut | `internlm/SIM_COT-GPT2-Coconut` (`checkpoint_28`) | Coconut repo's GSM8K eval: question + "\n" + `<\|start-latent\|>` + 6 × `<\|latent\|>` + `<\|end-latent\|>`, greedy, answer after "###" |

The 6 latents are the trained budget for all three. If a recipe detail cannot be matched (e.g. a checkpoint's special
token ids), that is recorded here before scoring, not tuned on the test accuracy.

**Reference values.**
* CODI GPT-2: 43.7 % (2502.21074; verified, LIT_BENCH_VERIFY). Our §13 probe with the same code: 42.5 %.
* SIM-CoT on CODI / on Coconut, GPT-2: the values in 2509.20317's GPT-2 table. **Literature verifies them from the paper
  and commits them before this step is scored.**

**Criterion (fixed now).** A model is **reproduced** if its bench accuracy is within ±3.0 points of the published value
(the 95 % binomial interval at n = 1,319 and p ≈ 0.4 is ±2.7 points). Otherwise "not reproduced", reported with the
gap. No recipe change after seeing accuracy.

**Reported per model.** Accuracy with 95 % CI, accuracy by equation steps (bench depth), compute (latent passes,
generated tokens), and the four GSM8K baselines on the same items (last number 0.023, constant 5 0.030, answer in
question 0.064, one-op oracle 0.155).

**Run.** Local RTX 3070 / 2070, `python -m bench.run --task gsm8k --model <adapter spec>`; outputs
`bench/results/gsm8k__<tag>.json`.

## Amendment 1 (2026-10-01 18:27 EDT; before any SIM-CoT-Coconut inference and before any full-run result was read; Literature 39aa8b2)

* **Reference values** (2509.20317 Table 1, GSM8k-Aug, verified by Literature):
  - SIM-CoT + Coconut 44.8; SIM-CoT + CODI 42.6.
  - Their own re-runs: Coconut 36.6, CODI 42.0, SFT-CoT 42.7.
  - The CODI criterion keeps the original paper's 43.7, as registered; 42.0 is reported next to it.
* **SIM-CoT-Coconut recipe correction.** The repo config (`args/gsm_coconut.yaml`, `gsm_simcot*.yaml`) uses
  c_thought 2 and max_latent_stage 5 with pad_latent_to_max, i.e. **10 continuous thoughts**, not 6. The Coconut
  inference uses 10.
* **Checkpoint.** The Hugging Face upload is `checkpoint_28`, the only one public; the repo's eval config loads
  `checkpoint_30`. We evaluate `checkpoint_28` and report the mismatch as a caveat to its reproduction verdict.
* The CODI runs (6 latents, as registered) are unaffected; they started at 18:27:24 (log creation time).

## Result (2026-10-01 18:33 EDT; bench `gsm8k`, 1,319 items, local GPUs)

| model | bench accuracy [95 % CI] | registered target | gap | verdict (±3.0) |
|---|---|---|---|---|
| CODI GPT-2 (6 latents) | **42.5** [39.8, 45.1] | 43.7 (original; SIM-CoT's re-run 42.0) | −1.2 | **reproduced** |
| SIM-CoT + CODI GPT-2 (6 latents) | **39.4** [36.8, 42.0] | 42.6 | −3.2 | **not reproduced** (just outside) |
| SIM-CoT + Coconut GPT-2 (10 thoughts, HF checkpoint_28) | **44.8** [42.1, 47.5] | 44.8 | 0.0 | **reproduced** (caveat: the repo evaluates checkpoint_30) |

No-reasoning baselines on the same items: last number 2.3, constant 5 3.0, answer in question 6.4, one-op oracle 15.5.
Every model is far above them.

**By equation steps** (n per bucket: 1: 65, 2: 357, 3: 364, 4: 290, 5: 138, 6: 57):

| model | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| CODI | 0.40 | 0.66 | 0.49 | 0.31 | 0.14 | 0.02 |
| SIM-CoT + CODI | 0.45 | 0.61 | 0.46 | 0.26 | 0.17 | 0.04 |
| SIM-CoT + Coconut | 0.40 | 0.62 | 0.50 | 0.40 | 0.22 | 0.07 |

**Reading.**
* The harness reproduces CODI and SIM-CoT-Coconut. SIM-CoT-CODI lands 3.2 points low, the same as our §13 probe
  (0.393). This points to the checkpoint or recipe, not the harness.
* All three fall steeply with equation steps (≤ 0.07 at 6 steps). This is the depth horizon THEORY §13 described.
* The CODI numbers match §13 to the third decimal (same inference code), so the bench port is faithful.
* Compute: CODI 6 latent passes + ~6 generated tokens; Coconut 10 passes + ~2 tokens ("### N").

## Amendment 2 (2026-10-01 18:34 EDT; descriptive, before running; Audit's suggestion): do the GSM8K latents do the work?

Same checkpoints, items and recipes. Only the number of latent steps changes:
* SIM-CoT + Coconut at K = 0, 1, 2 thoughts (vs 10). The start and end markers are kept; K = 0 is the markers only.
* CODI and SIM-CoT + CODI at 0 and 1 latent steps (vs 6). §13 already has CODI L0 0.221, L2 0.234, L6 0.425 on the
  same code; this re-runs them inside bench.

**Reading (fixed now; descriptive, no pass/fail).**
* If accuracy at K ≤ 1 stays within 3 points of the full budget, the latents add nothing measurable (as in ProsQA
  Coconut).
* If it drops by ≥ 10 points, the latent steps carry computation.
* Reported per equation-step bucket as well.

## Amendment 2 result (2026-10-01 19:18 EDT; descriptive)

| model | latent steps | accuracy | by equation steps 1 / 2 / 3 / 4 / 5 / 6 |
|---|---|---|---|
| CODI | 0 | 22.1 | 0.26 / 0.32 / 0.24 / 0.19 / 0.06 / 0.05 |
| CODI | 1 | 24.9 | 0.29 / 0.38 / 0.25 / 0.21 / 0.10 / 0.02 |
| CODI | **6** | **42.5** | 0.40 / 0.66 / 0.49 / 0.31 / 0.14 / 0.02 |
| SIM-CoT + CODI | 0 | 32.0 | 0.34 / 0.51 / 0.32 / 0.24 / 0.17 / 0.02 |
| SIM-CoT + CODI | 1 | 28.9 | 0.29 / 0.48 / 0.30 / 0.20 / 0.16 / 0.00 |
| SIM-CoT + CODI | **6** | **39.4** | 0.45 / 0.60 / 0.46 / 0.26 / 0.17 / 0.04 |
| SIM-CoT + Coconut | 0 | 5.1 | 0.06 / 0.11 / 0.04 / 0.02 / 0.03 / 0.02 |
| SIM-CoT + Coconut | 1 | 6.2 | 0.12 / 0.12 / 0.05 / 0.03 / 0.04 / 0.02 |
| SIM-CoT + Coconut | 2 | 13.8 | 0.17 / 0.31 / 0.10 / 0.04 / 0.04 / 0.00 |
| SIM-CoT + Coconut | **10** | **44.8** | 0.40 / 0.62 / 0.50 / 0.40 / 0.22 / 0.07 |

**Reading (per the fixed rule).**
* CODI drops 20 points without latents, and SIM-CoT-Coconut 40 points. The latent steps carry computation, unlike
  ProsQA Coconut.
* SIM-CoT-CODI drops 7.4 points, between the two thresholds; its no-latent accuracy (32.0) is already high.
* **Caveat.** K = 0 / 1 are inputs these models never saw in training (CODI and Coconut always train with the full
  budget at the last stage). Some of the drop may be distribution shift rather than lost computation, especially
  Coconut's 5.1, which is below the one-op oracle.
* The gain from latents is largest at 2–4 equation steps. At 6 steps every configuration is ≤ 0.07: the latents add
  depth up to ~4 steps and no further.

## Amendment 3 (2026-10-01 20:46 EDT; descriptive, before running): thought-budget curves (C2-style)

Same checkpoints, items and recipes. Budgets beyond and between the trained ones:
* SIM-CoT + Coconut: K = 4, 6, 8, 12, 16 (trained 10).
* CODI and SIM-CoT + CODI: L = 2, 4, 8, 12 (trained 6). §13 has the CODI L-curve from the same code; this re-runs it in
  bench with per-step buckets.

Reported: accuracy and per-step accuracy vs budget; no pass/fail. The question is whether accuracy keeps rising past the
trained budget (a C2 signature) or peaks at it (§13's horizon).
