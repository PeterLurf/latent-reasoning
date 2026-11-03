# SubLAT's ProsQA-OOD splits (for the Lane D1 port), wave-2 ID spot-check, new LITERATURE.md entries (2026-11-03)

Written 2026-11-03 by the Literature session, for Theory.
- All IDs resolved through the export.arxiv.org API.
- SubLAT and Qwen3 were read in their arXiv HTML.

## (3) SubLAT: Zeng & Zhang, "Latent Path Exploration via Potential-Guided Subtrajectory Balance", 2610.40283 (v1, 2026-10-29)
**Splits.**
- Trained on ProsQA (17,886 train / 300 val / 500 test).
- Evaluated on the official test set plus four structural-OOD splits: "The 7-hop and 8-hop splits increase reasoning depth, while DAG and Dense vary the graph structure" (§4, App. A.1, Table 4).
- **Not given in the HTML:**
  - the exact construction of DAG / Dense (edge density, branching, distractor rules);
  - the split sizes;
  - any generator code or link (no GitHub / HF link found);
  - a data licence. The paper carries the arXiv perpetual non-exclusive licence only.
- **For the port:** treat the splits' definitions as unavailable until the authors release code. Either contact the authors or regenerate with Coconut's ProsQA generator (facebookresearch/coconut; check its licence before reuse) using our own pre-stated hop / density parameters, cited as "in the spirit of SubLAT's 7-hop / 8-hop / DAG / Dense splits".

**Protocol.**
- Mean ± sample sd over **3 training seeds**, each averaged over 5 inference runs.
- † = a public checkpoint evaluated over 5 inference runs.
- COCONUT and SubLAT are initialised from the same CoT-SFT checkpoint.
- Best-validation checkpoint for test and OOD.

**GPT-2 Small, Table 2** (ProsQA test / 7-hop / 8-hop / DAG / Dense / OOD average):

| method | ProsQA test | 7-hop | 8-hop | DAG | Dense | OOD avg |
|---|---|---|---|---|---|---|
| CoT-SFT | 71.93 ± 3.63 | 11.57 ± 4.01 | 10.50 ± 4.40 | 36.40 ± 7.98 | 22.00 ± 8.91 | 20.12 |
| COCONUT | 98.01 ± 0.12 | 73.28 ± 5.58 | 72.46 ± 0.71 | 58.68 ± 1.60 | 65.65 ± 2.68 | 67.52 |
| CODI † | 82.48 ± 1.03 | 69.94 ± 0.73 | 70.07 ± 0.69 | 59.29 ± 0.89 | 65.77 ± 0.88 | 66.27 |
| CoLaR-5 | 52.35 ± 2.82 | 29.11 ± 10.63 | 28.47 ± 10.69 | 49.41 ± 4.57 | 27.71 ± 10.99 | 33.68 |
| SubLAT | 88.93 ± 0.72 | 77.03 ± 1.06 | 77.20 ± 0.55 | 50.70 ± 1.58 | 72.45 ± 1.63 | 69.34 |

- LLaMA-3.2-1B and DeepSeek-R1-Distill-Qwen-1.5B rows are also in Table 2. Notably, LLaMA CoT-SFT gets 95.11 in-domain but 10.60 / 10.72 at 7 / 8 hops.
- Caveat for interpretation: ProsQA is shortcut-prone (Cui 2602.22441, LIT_S218_CLOSE). Pair any port with a zero-latent / noise arm.

## (2) Wave-2 spot-check: all resolve and match their stated roles
- **2606.29983** Kuo et al.: loop count ↔ length spurious correlation; stochastic stopping.
- **2604.06374** Rizvi-Martel et al., "The Illusion of Superposition?": pretrained models' latents largely unused, shortcut-like (LIT_PRETRAINED_VS_SCRATCH_2026-10-09).
- **2602.00449** Liang & Pan: CODI on sequential polynomial iteration; deeper horizons compress or lose the latent rollout.
- **2608.11233** Shapiro: recurrent-depth retrofit of **Qwen2.5-0.5B-Instruct** (Prelude / weight-tied Recurrent Block / Coda).
- **2505.09388** Qwen3 report: **Qwen3-0.6B-Base GSM8K = 59.59** (4-shot CoT, Table 8; Qwen2.5-0.5B 41.62 in the same table). Confirmed.
- **2007.01919** Correia, Niculae, Aziz & Martins: exact marginalisation of discrete latents via sparsity.
- **2203.16634** Haviv, Ram, Press et al.: transformer LMs without positional encodings still learn positional information (via the causal mask).

## (1) Added to LITERATURE.md
SubLAT 2610.40283; Selective Latent Thinking (SLT, Xie et al.) 2605.25745; Pagadala 2611.02333 (selection rules decide adaptive halting vs trajectory readouts on Ouro-1.4B); Frey et al. 2603.08391 (adaptive loops and memory in transformers, "think harder or know more?").
