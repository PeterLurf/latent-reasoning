# Facts for the PCCoT-PROBE lead: PCCoT (2506.18582) on GPT-2 GSM8K-Aug, and a §149-style C1 battery (2026-10-09)

Literature session, for Audit (the lead) and Theory.
- **Sources:** the paper's HTML (Table 1, Sec. 4, App. C); the GitHub API (whyNLP/PCCoT); the HF API, model card, config and args for
  whynlp/pccot-gpt2 (2026-10-09).

## 1. The 49.48 number
- **Table 1, GPT-2 Small, GSM8K-Aug test:**
  - **PCCoT 49.48 ± 0.31** (GSM8K-Aug-NL 49.23 ± 0.80);
  - Continuous CoT 48.24 ± 1.61; pause tokens 39.27 ± 0.46; implicit CoT 37.78 ± 0.31;
  - **CoT 44.1** (GSM8K-Aug-NL 34.8); CODI 43.7 (NL 35.3); Coconut 34.1 ± 1.5; iCoT 30.
- **Seeds:** "averaged over 3 random runs with standard deviations", for implicit CoT, pause, continuous CoT and PCCoT.
- **The CoT 44.1 is not their run:** "The CoT baseline is taken from CODI" (CODI's reported number), as are the CODI / Coconut / iCoT
  rows. Our own CoT references are 42.9 (Coconut paper), 41.6 (Dilgren) and 41.8 on our bench.
- **Llama-3.2-1B-Instruct:** PCCoT 53.35, *below* CoT 61.6 and CODI 55.6. The GPT-2 advantage over CoT doesn't hold at 1B.

## 2. Training setup
- **CODI-style self-distillation:** a shared model trains a teacher task (standard CoT CE on gold reasoning plus answer) and a student
  task (latent).
  - An L1 loss matches teacher and student prediction distributions on the last token of the answer prompt (":" in "The answer is:").
  - **CODI's MLP on latent tokens is dropped** "for fair comparison". Config loss_alpha / beta / gamma = 1.0.
- **Latents:** **c = 24 latent thought tokens, T = 3 extra Jacobi iterations** (config num_iterations 3).
  - All latents are updated in parallel per iteration. T ≥ c reduces to continuous CoT (sequential); T = 0 ≈ pause tokens.
  - Ablation: accuracy improves up to about T = 3, then fluctuates (unstable beyond T = 6).
  - Baselines: pause = 24 tokens; continuous CoT = 12 latents.
- **Optimisation (GPT-2):**
  - **LoRA** r 128, α 32 (dropout 0.1; c_attn / c_fc / c_proj), batch 128;
  - AdamW **lr 3 × 10⁻³**, wd 0.01; **40 epochs**;
  - greedy decoding.
  - Llama-1B: lr 8 × 10⁻⁴, 10 epochs.
- **Tokens:** bot 50257, eot 50258, latent 50259; answer prompt "The answer is:".

## 3. Released checkpoints: yes
- **HF:** **whynlp/pccot-gpt2** (GSM8K-Aug), **whynlp/pccot-gpt2-nl** (GSM8K-Aug-NL), **whynlp/pccot-llama1b**.
  - Files: a LoRA adapter (adapter_model.safetensors) + config.json (num_iterations 3) + pccot_args.json (24 latents) + tokenizer.
  - Last modified 2026-03-29.
  - **Loading needs the repo's `models` module** (AutoPeftModel + PCCoT args + COTDataProcessor; card quick-start).
- **Code:** **github.com/whyNLP/PCCoT** (MIT; EMNLP 2025).
- **The HF card has no licence tag and states no accuracy.** Use the GitHub MIT licence, and reproduce the accuracy on our bench
  before probing (as GSM8K_REPRO did for CODI / SIM-CoT).

## 4. Causal or faithfulness analysis of PCCoT's latents
- **None found.**
  - PCCoT isn't mentioned in 2607.06648, 2606.12689, 2604.06374, 2609.35643 or 2607.09698 (full-text check, 2026-10-09).
  - A web search found none.
  - The paper itself reports accuracy, efficiency and T / c ablations only.
- **So a C1 battery on PCCoT would be the first.**

## Flags for the probe design
1. **Jacobi parallelism changes what "step alignment" means.** All 24 latents are updated together T = 3 times. There's no
   latent ↔ equation-step mapping by construction (like CODI, unlike SIM-CoT + Coconut).
   - So per-step donor SERIAL is probe-defined and descriptive. **Ablation, K / T-removal (T = 0 ≈ pause tokens, a natural control) and
     ρ across Jacobi iterations are the rule-able tests.**
2. **ρ is natural here:** the Jacobi iterations form a fixed-point map on the latent block. Measure the contraction across T = 1 … 6+.
   - Their own instability beyond T = 6 predicts **ρ ≥ 1 at larger T**, i.e. expansion (cf. 2607.09698's Coconut-expanding /
     CODI-attractor split).
3. **Reproduce first:** 49.48 is a 3-run mean. The single released checkpoint may differ. Run the bench (1,319 items, GSM8K_REPRO
   harness; the CoT reference is our 41.8) before any probe.
4. **Shortcut / bypass check:** with the question visible to all latents (as for CODI), run the no-latent (c = 0) and pause-equivalent
   (T = 0) evaluations on the trained model (out of distribution, as GSM8K_REPRO's caveat).

## Addendum (2026-10-09): the bench row, and the same-items check
- **PCCoT's dataset whynlp/gsm8k-aug** (HF datasets-server): train 385,620 / validation 500 / **test 1,319**. The first test items match
  ours (Janet's ducks, robe), in the equation-step format. So same_items = true for our GSM8K bench (as for CODI / Coconut).
- **Added to bench/literature.json:** "PCCoT (parallel continuous CoT, Jacobi; c = 24 latents, T = 3), GPT-2 Small", acc 0.4948,
  source 2506.18582 Table 1 (3-run mean ± 0.31; CoT row copied from CODI). Verified by this note.
