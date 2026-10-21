# SCALE-PROBE (public ≥1B latent checkpoints) and CHUNK-SNAP prior art (Theory's asks, 2026-10-20)

Written 2026-10-20 ~22:50 EDT by the Literature session, for Theory and Audit.
- **Method:** two read-only subagents.
- **Checks I re-ran myself:**
  - every HF repo below against huggingface.co/api/models/<id>: public, not gated, licence tag as listed;
  - every arXiv ID against its abs page.
- **Not checked by me:** the GSM8K numbers are the papers' own figures, as the subagents read them [empirics]. File sizes and repo script paths also come from the subagents; check them before building.

## 1. SCALE-PROBE: loadable latent-reasoning checkpoints at 1B or more

| model (paper) | HF id (verified) | licence | base / data | latent-loop code | reported GSM8K |
|---|---|---|---|---|---|
| **CODI** (2502.21074) | `zen-E/CODI-llama3.2-1b-Instruct` | MIT | Llama-3.2-1B-Instruct (gated base; `alpindale` mirror) / GSM8k-Aug | `zhenyi4/codi` scripts/test_llama1b.sh: 6 latents, projection 2048 + LN, LoRA r128/α32, bf16; `--inf_latent_iterations` = K | 55.6 |
| SIM-CoT + CODI (2509.20317) | `internlm/SIM_COT-LLaMA3-CODI-1B` (+3B, 8B) | MIT | same | InternLM/SIM-CoT CODI/test.py; empty config.json, so it needs the repo loader | 56.1 |
| Dilgren family (2604.04902) | `connordilgren/llama32-1b-gsm8k-{coconut,codi}` (+ ProsQA / ProntoQA, cot, no-cot) | MIT | same / GSM8k-Aug, ProsQA, ProntoQA | connordilgren/are-lrms-easily-interpretable | Coconut 35.7, CODI 56.0 |
| PCCoT (2506.18582) | `whynlp/pccot-llama1b` (LoRA) | none stated | same | whyNLP/PCCoT, 24 latents, 3 Jacobi iterations | 53.35 ± 0.18 |
| CoLaR (2505.16552) | `ModalityDance/latent-tts-colar` | MIT | same / GSM8K | ModalityDance/LatentTTS | 26.8 |
| **MUX** (2607.18264) | `MisakiTaro0414/MUX` (mux-llama1b-gsm8k.bin) | Apache-2.0 | same / GSM8k-Aug | MisakiTaro0414/mux scripts/eval_llama1b.sh | 56.7 ± 0.5 |
| Latent-GRPO (2604.27998) | `DJCheng/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10` (+ SFT) | none stated | same; RL on extra generated questions | SGLang; latents = top-10 vocabulary-embedding mix | 66.29 (SFT 47.55) |
| CODI answer-only control | `bcywinski/codi_llama1b-answer_only` | none stated | same | CODI | 41.60 latent vs 36.72 direct (blog) |
| LOTUS (2606.31779) | `yingfanbot/gsm-lotus-llama3b` (3B only) | MIT | Llama-3.2-3B-Instruct | yingfan-bot/lotus | 70.05 |
| Huginn (2502.05171) | `tomg-group-umd/huginn-0125` (3.5B) | Apache-2.0 | from scratch, 800B tokens | bundled; `num_steps` = depth | 34.8 / 42.1 (GSM8K-CoT) |
| Ouro (2510.25741) | `ByteDance/Ouro-1.4B` (+2.6B) | Apache-2.0 | from scratch | bundled; `total_ut_steps` | 78.92 at 4 steps |

**Not released:**
- KaVa 2510.02312, PMPS 2609.09928 and Cui 2602.22441.
- No paper-backed ≥1B Coconut; the third-party repos have no card or are gated.

**Published causal tests at ≥1B** [empirics]:
- **Cui 2602.22441** (own Llama-1B models), Gaussian noise on the latents, GSM8K:
  - CODI 55.57 → 27.45; SIM-CoT 56.03 → 10.08; CoT 61.74 → 0.03.
  - ProsQA barely moves.
- **Dilgren 2604.04902** (the released checkpoints):
  - Mean-latent replacement: GSM8K CODI −25.0, Coconut −5.8; ProsQA and ProntoQA ≈ 0.
  - Early stopping: the answer first matches after 30.9 % (CODI) or 19.5 % (Coconut) of the latents.
- **Jin 2607.06648** (Llama-1B ProsQA, across training checkpoints): CODI's output changes 0.39 % under mean replacement of all latents, 1.40 % under norm-noise and 36.91 % under zero replacement.
- **Li 2602.08783** (official CODI-1B): zeroing one latent step flips about 0.1–0.2 of GSM8K answers, peaking at the middle steps.
- **Cywiński et al.** (LessWrong, 2025-12-22; blog, not peer-reviewed):
  - On the CODI-1B checkpoint, averaging latents 3 / 5 over prompts with **different** intermediate values hurts; same-value averages don't.
  - Accuracy peaks at K 5 of 6.
  - This is the nearest to a donor swap.
- **Gap:** no published full question-swap (latents from a different question) or donor-follow rate on a ≥1B model. Our 210-4 battery (donor follow 0.04 on GPT-2 CODI-20) would be **new at 1B**.

**Recommendation:** start with `zen-E/CODI-llama3.2-1b-Instruct`.
- It is official and MIT-licensed, has a K flag, and four papers report on it.
- In bf16 it should fit an 8 GB local card, so it costs $0.
- Test on GSM8K, not ProsQA: Cui and Dilgren both find ProsQA latents unnecessary.
- Add `connordilgren/llama32-1b-gsm8k-coconut` as a method contrast and `bcywinski/codi_llama1b-answer_only` as a same-weights direct baseline.
- The MUX 1B checkpoint makes a matched local-loss contrast possible at $0.

## 2. CHUNK-SNAP prior art (k latent passes → decode to reduced-problem text → re-encode fresh → repeat)

**Verdicts:**
- **(1) Accuracy vs k: PARTLY done.** Every k-like sweep keeps the latent context. None decodes to text and drops the latent history.
- **(2) Depth extrapolation from re-rooting: DONE text-side, NEW latent-side.**

| ID | what was varied / found | tag |
|---|---|---|
| Thinking States 2602.08332 (Amos, Caciularu, Geva, Globerson et al.) | thoughts every c tokens, added to later inputs (not re-rooted). GSM accuracy peaks at medium c and drops for c > 8 (Fig. 3b, figure only). Parity at N 10: 98.4 vs CoT 12.4 | empirics |
| CoLaR 2505.16552 | compression c on Llama-1B: c 2 → 40.1 % (length 12.7), c 5 → 26.8 % (5.6); explicit CoT 49.4 % | empirics |
| Token Assorted 2502.03275 | one latent per r text tokens, Llama-3B: CoT 25.2, r 2 27.1, r 16 28.1 (mean of 7 math sets) | empirics |
| Coconut 2412.06769 | thoughts per stage c: c 2 gives 34.1 ± 1.5; c 3 dips with more variance; ProsQA k 0–6 latents before language (figure); no curriculum 14.4 | empirics |
| SpiralThinker 2511.08983 | latents per step N 1–10 make little difference, best N 5 | empirics |
| **He 2512.02677** | looped locate-and-replace that **rewrites the expression text** each pass; trained at depth ≤ 5, tested to 12: Boolean algebra at depth 12, 66.7 vs 51.8 end-to-end; a whole-state re-feed with no locate step reaches 0 by depth 8 | empirics |
| Neural Rewriting System 2402.17407 | Selector / Solver / Combiner; trained at nesting depth 1–3, tested at 4–6; most errors are Selector (locate / copy) errors | empirics |
| PENCIL 2503.14337 | reduction erases thoughts. Theorem: Turing-complete with optimal space, assuming exact steps. Each size trained in-distribution, so **not** extrapolation | theorem + empirics |
| Iterative decoding 2110.04169; Recursion of Thought 2306.06891 | text-side iterate / recurse on sub-contexts; length generalisation on PCFG / Cartesian product (2110.04169) | empirics |

**Design lessons for CHUNK-SNAP** [empirics]:
- **Errors compound mainly in locate and copy, not in compute.** He's re-feed ablation hits 0 by depth 8, and Selector errors dominate in the Neural Rewriting System. So score decode fidelity on the **copied remainder** separately from the **substituted value**, and report how error grows with the number of snaps.
- **Expect an interior optimum in k.** Too-large chunks hurt: Thinking States beyond c 8, Coconut at c 3, CoLaR at c 5.
- PENCIL's trace accuracy is 83 % at n 10 while its answers are 100 %, so not every decode error is fatal. Pre-state whether the re-root uses the decoded text exactly as decoded or after canonicalisation.
- He 2512.02677 is the natural text-side comparator. It is already in LIT_DEPTH_RECURRENT_CHEN_HE_2026-10-11.
