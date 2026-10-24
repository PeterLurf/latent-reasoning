# LARGE-BASE draft (LB_draft.md, 9c38f364): ID checks, binding-with-scale priors, N1–N3 (2026-10-24)

Written 2026-10-24 by the Literature session, for Theory and Audit.
- **No subagents.** Abs pages checked myself; the Qwen2.5 table read in the arXiv HTML.

## (a) IDs
All verified (title / authors):
- Saxe 1312.6120;
- Feng & Steinhardt 2310.17191;
- Kim & Schuster 2305.02363;
- Prakash 2402.14811;
- Qwen2.5 report 2412.15115;
- Saunshi et al. 2502.17416 ("Reasoning with Latent Thoughts: On the Power of Looped Transformers");
- Biran 2406.12775 ("Hopping Too Late").

**Qwen2.5-0.5B base GSM8K = 41.6 (4-shot), Table 5** of 2412.15115 (Qwen2-0.5B 36.4; Qwen2.5-1.5B 68.5). It replaces the draft's [ASSUMED].

## (b) Binding / entity tracking with scale, and transfer to fine-tuned latent loops [empirics]
- **Feng & Steinhardt:** a binding-ID mechanism "in every sufficiently large model from the Pythia and LLaMA families". The size threshold is not stated in the abstract; check the paper before relying on 0.5B.
- **Kim & Schuster:** among Flan-T5, GPT-3 and GPT-3.5, **only GPT-3.5 models (pretrained on lots of code) track entity states**. Capability depends on pretraining data as well as size.
- **Prakash:** fine-tuning (Llama-7B → Vicuna / Goat / FLoat) **enhances the existing entity-tracking circuit** rather than building a new one.
- **Transfer of a base's binding features into a fine-tuned latent loop:** not found. Prakash is the nearest support for "fine-tuning reuses base circuits", but for one-pass models only.

## (c) Novelty
| # | verdict | basis |
|---|---|---|
| N1 (the §184 DeltaMem latent loop ported to Qwen2.5-0.5B, C1-tested on word programs vs exposure-matched one-pass and text-CoT) | **NARROW** | Latent / looped methods on modern pretrained LMs exist: CODI on Llama-3.2-1B; Huginn 2502.05171 / Ouro from scratch; **retrofits onto pretrained bases, including Shapiro 2608.11233 on Qwen2.5-0.5B** and McLeish 2511.07384; Saunshi 2502.17416 (looped ≈ deeper on synthetic reasoning). New only as the combination: a delta-rule memory loop, causal serial C1 tests, matched baselines, this task. |
| N2 (a frozen-base linear binding probe, pre-registered as a predictor of whether a latent write can carry step 1, derived from deep-linear escape at a zero-initialised read) | **OK** (not found) | Saxe 1312.6120 supplies the escape-time theory [theorem, deep linear nets]. Binding probes come from Feng / Kim / Prakash. A probe used as a pre-registered trainability predictor for a latent write was not found in these. Tag the derivation's linearisation as an assumption. |
| N3 (the same latent recipe fails at GPT-2 small, re-tested at a larger base with relative-update lr matching) | **NARROW** | Latent-CoT scale comparisons exist on fixed benchmarks: CODI GPT-2 vs Llama-1B; SIM-CoT GPT-2 → 8B; Cui 2602.22441 GPT-2 vs Llama-1B; LOTUS 2606.31779 (latent methods trail CoT above ~1B, with the gap growing). On a fixed synthetic word-program task with lr transfer by relative-update matching: not found. |
