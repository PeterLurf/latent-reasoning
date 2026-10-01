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
