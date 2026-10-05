# Prior and method check for §119: unbottlenecked latent-CoT baselines (Coconut, CODI, SIM-CoT) on our S5 protocol (2026-10-04)

Literature session, for Theory; Audit copied. Recipe facts from the papers (arXiv HTML) and the official configs
(facebookresearch/coconut `args/*.yaml`; InternLM/SIM-CoT `Coconut/args/gsm_simcot.yaml`), read 2026-10-04.

## (1) Faithful recipes
- **Coconut 2412.06769.**
  - The last hidden state is fed back as the next input embedding. The question (here: chunk start text + ops) stays
    visible, and thoughts attend to the whole prefix.
  - Curriculum: stage s replaces the first s text steps with s·c thoughts; the CE loss is on the remaining text and the
    answer only. Optimiser reset each stage. `pad_latent_to_max: True`.
  - Official configs:
    - ProsQA / ProntoQA: **c = 1, 5 epochs per stage, max_latent_stage 6, 50 epochs**, from GPT-2 base;
    - GSM8K: **c = 2, 3 epochs per stage, max_latent_stage 3, 25 epochs**, from a CoT checkpoint.
    - lr 1e-4, wd 0.01, batch 32.
  - For S5 (one step = one op): c = 1, max_latent_stage 4 per chunk (ProsQA-style) is the faithful choice.
    Budget-match the epochs.
- **CODI 2502.21074.**
  - Teacher (text CoT) and student (latents) **share one model**.
  - Loss α·L_student + β·L_KD + γ·L_teacher, with **α = β = γ = 1 for GPT-2**.
  - L_KD is **L1 with stop-gradient on the teacher**, on the hidden state of **the token immediately preceding the
    answer (":" in "The answer is:")**, **across all layers**, each layer normalised by the teacher's batch std.
  - **6 continuous thoughts.** Latents pass through a **2-layer MLP + LayerNorm** projection. The question stays in
    context.
  - **LoRA r = 128, α = 32, bf16** (GPT-2 lr 3e-3, 40 epochs; LIT_GSM8K_STEP2).
  - For S5, add an answer prompt with a fixed pre-answer token to distil (e.g. " state:"), and set the number of
    thoughts to 4 per chunk (one per op), or 6 as published (report which).
- **SIM-CoT 2509.20317.**
  - Coconut plus an **auxiliary decoder p_φ "architecturally identical to the LLM", with separate parameters φ**. It
    **takes only the k-th latent z_k as conditioning (no question or context)** and autoregressively generates step k's
    text. Loss: λ_step·L_step + λ_lm·L_ans. Removed at inference.
  - One step corresponds to the latents of c = 2 by default.
  - The released GPT-2 config **initialises from a trained Coconut checkpoint** (`load_model_path …/gsm-coconut/
    checkpoint_24`), with c = 2, 3 epochs per stage, max_latent_stage 5, 35 epochs, mode
    `coconutgpt_same_word_embedding`, full training.
  - The λ values are not in the released yaml; take them from the code defaults and record them.

## (2) Is the chunked-restart host fair, or is the native form needed too?
- **Both, and reviewers will ask for both.**
- **Native form** (one sequence: start + all ops + latents + answer, no restarts) is the methods' own setting.
  - But 512 ops × c thoughts exceeds GPT-2's 1,024-token context, so native is testable only at **in-range lengths**
    (e.g. train ≤ 16 ops, test to the context limit).
  - Report the native length curve; that's where their failure mode shows.
- **Chunked host** (§97 harness) gives them our scaffold's advantage, which is the fair apples-to-apples test of the
  *carrier*, and the only way to reach 512 / 2,048.
- **The donor protocol applies in both.** With full attention to visible ops, bypass is the expected outcome (Aswal
  2606.12689; SCIT 2608.27265). That is the point of the controlled-C1 claim.

## (3) Our GEN / MASK vs SIM-CoT
**Close, with five differences:**

| | SIM-CoT | §97 GEN | §105 MASK |
|---|---|---|---|
| decoder | separate copy (φ ≠ θ) | same model, in-context fork | same model, fork sees only the slot's KV |
| decoder sees | the latent only | full context (so bypass, §104) | slot only (≈ SIM-CoT's isolation) |
| latents per step | c = 2 | 1 | 1 |
| init / curriculum | from a Coconut checkpoint, stage curriculum | from base; K-CURR on k | same |
| restarts / snap | none | text snap every k + history cut | same |

- **So MASK ≈ "SIM-CoT with a tied, in-context decoder, one latent per op, plus snaps".**
- GEN is unlike SIM-CoT in the decisive respect: its decoder sees the context.
- Keep arm (c) as native SIM-CoT (a separate decoder copy) so the difference is testable.

## (4) Newer methods a reviewer may expect
- **Thinking States 2602.08332**, the most comparable on *state tracking with length generalisation* (Qwen 0.5B,
  parity / vars). If budget allows, this is the fourth arm.
- Also cite, as related (no arm needed):
  - KaVa 2510.02312 (KV-trajectory distillation, CODI-like);
  - CoLaR 2505.16552;
  - PCCoT 2506.18582;
  - System-1.5 2505.18962;
  - LIFT 2609.38149 (teacher-forced state feedback);
  - SELR 2608.13570 (self-decoding);
  - MUX 2607.18264.

## Fairness pitfalls
- **Training budget and trainable parameters:** CODI uses LoRA r = 128; the others full fine-tuning. Report trainable
  parameters and updates. Optionally run CODI in full fine-tuning as well.
- **Removal is descriptive (as registered); keep it.** Zero-ablation is off-distribution (2309.16042), so donor is
  primary.
- **Per-arm hyperparameter search:** give each baseline a small, equal lr sweep on a training-split criterion. That
  avoids the "baselines under-tuned" objection.
