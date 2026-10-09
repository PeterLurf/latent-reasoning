# Is pretraining what blocks identical-input think passes? Evidence for the FROM-SCRATCH INJECT diagnostic (2026-10-09)

Literature session, for Audit (the lead) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Details:** from the HTML of 2604.06374, 2511.07384 and 2608.11233 (2026-10-09).

## 1. 2604.06374 (Rizvi-Martel, Rabusseau, Mosbach): Tables 2a / 2b confirmed (as in LIT_DEEP_PROSQA)
- **2a, fine-tuned pretrained Coconut on ProsQA, a counterfactual no-latent evaluation:**
  - GPT-2 124M: 99.0 → 99.0;
  - SmolLM2-135M: 93.3 → 92.3; SmolLM2-360M: 98.7 → 98.0; SmolLM2-1.7B: 100.0 → 100.0.
  - **The maximum drop is 1.0%.** The answer comes in one pass and is copied through the latents (a shortcut).
- **2b, trained from scratch (8 heads, 768-d), w/ / w/o latents:** 2L 94.5 / 13.8; 4L 96.2 / 16.0; 8L 95.2 / 53.0; 12L 91.2 / 34.1.
  **The latents are used.**
- **Their explanation:**
  - (i) pretraining biases late layers to **commit to a token**;
  - (ii) **capacity** decides which solution is favoured (more layers → more single-pass solving: the w/o-latent accuracy rises with depth,
    53.0 at 8L).
- **The direct pretrained-vs-scratch evidence that pretrained models ignore the extra passes.** It's on graph reachability (ProsQA) with Coconut-style
  latents.

## 2. McLeish et al. 2511.07384, "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence" (McLeish, Li, Kirchenbauer, …)
- **Setup:**
  - converts TinyLlama-1.1B, OLMo-2-1B and Llama-3.2-1B into prelude / recurrent / coda models, dropping layers (e.g. a (4, 8, 4) layer selection for
    TinyLlama);
  - Huginn's Poisson-log-normal recurrence sampling, with a **curriculum on its mean** (train recurrence k = the distribution mean);
  - truncated backprop as in Huginn (the last 8);
  - a **"healing" period** on general data before task (math) data.
- **The pretrained-vs-scratch result points the *other* way here:**
  - **initialising from Llama-3.2-1B "strongly outperforms" random initialisation per training FLOP**, in loss and benchmarks, over about 120B FineWeb-Edu
    tokens. The curves would intersect only around 950B tokens;
  - the converted models beat their base models on GSM8K / MATH.
- **Reconciling with 2604.06374:**
  - McLeish trains recurrence with **random depth from the start, at LM-pretraining scale (about 10¹¹ tokens)**, so the recurrent block *must* be
    useful at every sampled depth.
  - 2604.06374 fine-tunes on a small task where one pass already suffices, which is the shortcut setting.
- **So pretraining doesn't block recurrence per se.** It blocks it when the task admits a single-pass solution and the training signal never
  forces the passes (fixed K, small data). That matches Goyal's pause result (LIT_SCHEDULE_FADE): pause-*pre*training is needed.

## 3. 2608.11233, "Retrofitting Recurrent Depth into a Pretrained Language Model … at Two Parameter Budgets"
- Qwen2.5-0.5B-Instruct is split into a prelude, a weight-tied recurrent block and a coda, with an **identity-preserving one-loop path** and a
  re-entry bridge on later loops.
- **The two budgets:** a 6M-parameter adapter over frozen base weights vs a 180M full block. They're similar overall (83.8% vs 84.0%), with the adapter
  ahead to depth 11 and behind past it.
- **The key ingredient:** intermediate-step supervision, so each loop computes one task step. The mechanism persists after outcome-only annealing.
- **Extrapolation:** to about 1.5× the supervised depth, holding 70% through depth 18.
- **vs a same-size scratchpad model:** that model matched within its horizon but collapsed beyond it (84% vs 72% overall; 53% vs 2.5% past depth 10;
  7.6× faster).
- **So a pretrained model *can* be made to use its loops**, with per-loop step supervision (cf. LIT_STEP_AUX), not with outcome-only training.

## 4. Anything closer?
- **Also from scratch, latents used and generalising:** ProsQA-Ext 2609.35643 (latent variants beat CoT / pause on 7–12 hops); Kohli 2604.07822 (looped,
  from scratch, depth extrapolation).
- **Pretrained, latents unused:** Aswal 2606.12689 (BFS-like patterns appear in controls; graded use); our own §104 / §119 / §171 (GPT-2).
- **Not found:** a single study that trains the *same* looped / think-pass architecture on arithmetic or graph tasks from both a pretrained and a
  random initialisation, with an extra-pass ablation in both. 2604.06374 comes closest (ProsQA, Coconut-style, fine-tune vs scratch).
- **The FROM-SCRATCH INJECT diagnostic would fill exactly this gap for arithmetic / ProsQA-Ext with a memory carrier.**

## Predictions and flags for the diagnostic
- **Prediction from 2604.06374:** the from-scratch model uses its think passes (THINK-NECESSARY passes); pretrained INJECT doesn't (as §171).
  - Capacity matters too: a small from-scratch model (2–4 layers) uses passes most. A 12-layer one may again solve in one pass when the task allows it.
  - So **match depth to the task's single-pass boundary**: use a shallow from-scratch model, or problems beyond L*.
- **The alternative reading from McLeish / 2608.11233:** pretrained models do use loops when the training **forces** them, through random depth from
  the start, per-loop step supervision, or a healing phase.
  - A pretrained arm with RAND-K (LIT_RAND_K) or STEP-AUX would show whether "pretraining blocks" or "the training signal doesn't force" is the cause.
  - **Run it beside the scratch arm**, so a scratch success isn't over-read as "pretraining is the problem".
