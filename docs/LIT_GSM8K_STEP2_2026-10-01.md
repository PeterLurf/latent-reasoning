# Prior check for GSM8K step 2 (own GPT-2-small arms on GSM8K-Aug) (2026-10-01)

Literature session, for Theory; Audit copied. IDs verified (arXiv API). Configs read from the official repos.

## 1. Reference numbers and training details (GPT-2 small, GSM8K-Aug; test = the 1,319 GSM8K test items, verified in LIT_BENCH_VERIFY)

| System | Acc. | Training details (source) |
|---|---|---|
| **Text CoT (Coconut repo)** | 42.9 ± 0.2 (2412.06769 T1) | `args/gsm_cot.yaml`: full fine-tuning from openai-community/gpt2, lr 1e-4, wd 0.01, batch 32 per process, 25 epochs, best checkpoint on `gsm_valid` (save_only_improve), fp32 |
| **Coconut** | 34.1 ± 1.5 (T1) | `args/gsm_coconut.yaml`: starts from the CoT checkpoint; **c_thought 2, max_latent_stage 3** (≤ 6 continuous thoughts), 3 epochs per stage, optimizer reset each stage, 25 epochs total, lr 1e-4; the eval config uses checkpoint 15 |
| Coconut (SIM-CoT re-run) | 36.6 | SIM-CoT configs: c_thought 2, **max_latent_stage 5** (10 thoughts). A different setting from the official one |
| **Pause tokens** | 16.4 ± 1.8 (Coconut T1) | the Coconut repo's pause baseline |
| **iCoT (KD)** | 30.0 (as cited in Coconut T1); 0.20 in 2405.14838's table | Deng 2311.01460 |
| **Stepwise internalisation (ICoT-SI)** | 0.30 (2405.14838 T3) | Δ = 1 CoT token removed per epoch, removal smoothing λ = 4, lr 5e-5, effective batch 32, AdamW reset on removal, up to 200 epochs or 24 h |
| **CODI** | 43.7 (2502.21074) | **LoRA** rank 128 / α 32 on GPT-2, lr 3e-3, 40 epochs, effective batch 128, bf16, **6 continuous thoughts**, ~36 h on one A100 |
| SIM-CoT (+Coconut / +CODI) | 44.8 / 42.6 (2509.20317 T1) | see LIT_BENCH_VERIFY addendum |
| **No CoT** | 16.5 ± 0.5 (Coconut T1) | — |

**Data:** GSM8K-Aug (Deng et al.) has **385,620** training examples with equation-only CoT (average 20.3 CoT tokens, per
CODI). A "faithful reproduction" means the official repo configs above: GPT-2 small, the same data, the best checkpoint
on validation, the same test file. Expect ±1–2 points across seeds (Coconut reports ±1.5).

## 2. History-cut or state-summarising CoT on GSM8K
- **Markovian Transformers** 2404.18988: on **GSM8K**, all information to the answer passes through a bounded CoT, with
  the question removed when answering (Llama scale; 19.6 → 57.1). The closest "text state only" result on GSM8K.
- **LightThinker** 2502.15589: compresses thought steps into gist tokens and discards the text; 4 datasets, 2 models.
  Check whether GSM8K is among them before citing a number.
- PENCIL 2503.14337 (reduction rules; evaluated on SAT, QBF and Einstein puzzles, not GSM8K); InftyThink 2503.06692 and
  Delethink 2510.06557 (competition maths, not GSM8K); MEM1 (multi-hop QA and agents).
- **Not found: a "results-only written state (question plus values computed so far), equation history erased" arm on
  GSM8K at GPT-2 scale.** New as scoped. Nearest: Markovian Transformers (text bottleneck) and LightThinker (compressed
  history).

## 3. Pitfalls
- **Label noise in the GSM8K test:** Vendrow et al. 2502.03461 (GSM8K-Platinum). Of 219 flagged questions, 110 were
  removed as ambiguous and 10 relabelled; over half of frontier-model "errors" were label problems. At ~40% accuracy
  this barely matters, but report GSM8K-Platinum as a sensitivity check.
- **Contamination:** none for GPT-2's pretraining (2019, before GSM8K's 2021 release). GSM8K-Aug's training set comes
  from GSM8K *train* questions, and its test file equals the original test (verified).
- **Format:** GSM8K-Aug answers come after "####" in `<<expr=val>>` equation format. Normalise commas, trailing ".0",
  negatives and decimals (bench `norm_number`).
- **Equation CoT vs natural language:** GSM8K-Aug's CoT is equations only, while the original GSM8K has
  natural-language CoT. Text-CoT numbers (42.9) refer to equation CoT.
- **Compute accounting:** CODI uses LoRA, Coconut full fine-tuning. Matched-FLOPs comparisons must count trainable
  parameters and training epochs, not only inference thoughts.

## Addendum (2026-10-01, for amendment 68-1): is there a public GPT-2-small GSM8K-Aug CoT-SFT checkpoint?
**No usable one found** (Hugging Face API, 2026-10-01).
- **internlm** (SIM-CoT) releases only `SIM_COT-GPT2-Coconut` (MIT; the files include **`checkpoint_28`**, so the HF upload
  is checkpoint 28 even though the repo's eval YAML names checkpoint_30), `SIM_COT-GPT2-CODI` and LLaMA variants. No
  CoT-SFT baseline.
- **zen-E** (CODI): `CODI-gpt2` and `CODI-llama3.2-1b-Instruct` only. No CoT-SFT teacher.
- **Coconut** (facebookresearch): no checkpoints released (repo README). bmarti44 releases a *ProsQA* CoT baseline only.
- **yuntian-deng** (iCoT): GPT-2 explicit/implicit CoT checkpoints for *multiplication* only; GSM8K only for Mistral-7B
  (implicit).
- **Community** (`fiveflow/gpt2-gsm8k`, `thekarthikeyansekar/gpt2-gsm8k`, `princeLee1984/GPT2-COCONUT-gsm8k`, …): no
  licence declared, no reported accuracy, and the training format is undocumented (most likely original GSM8K
  natural-language CoT, not GSM8K-Aug equations). Not usable as a faithful Coconut stage-0.
- **Recommendation:** train TC from scratch with the official `args/gsm_cot.yaml` (GPT-2, GSM8K-Aug 385k, lr 1e-4,
  batch 32, 25 epochs, best on `gsm_valid`; reference 42.9 ± 0.2).
