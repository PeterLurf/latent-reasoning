# Task families and architecture precedents for latent-loop tests (2026-09-30)

Literature session. Answers Audit (a: benchmarks, b: restart/count-free architectures) and Experiments (1: edits to
pretrained LMs, 2: long-chain tasks); Theory and Audit copied. IDs verified via Semantic Scholar or arXiv. Licences are
from the GitHub or Hugging Face APIs; "?" means unverified. Result numbers are from abstracts or model cards unless noted.

## A. Long-chain tasks with checkable intermediate states

| Task | Depth control | Licence | Known latent / looped results; latent vs text CoT? |
|---|---|---|---|
| **ProsQA** (Coconut repo) | 3–6 hops; generator extendable | MIT | Coconut 97.0 vs CoT 83.0 (bmarti44 card). **Latent vs CoT: yes.** 7/8-hop OOD: Coconut 66/68, pause 75 |
| **ProntoQA** 2210.01240 / PrOntoQA-OOD 2305.15269 | 1–5+ hops, generator | Apache-2.0 | Coconut evaluated (near ceiling). Latent vs CoT: yes (Coconut) |
| **GSM8K** 2110.14168 (+ GSM8K-Aug) | 2–8 equation steps | MIT | Coconut 34.1 (Zou 2602.01148); CODI ≈ CoT at GPT-2; SIM-CoT +2.1 over CoT (2509.20317). **Latent vs CoT: yes** |
| **iGSM** (Physics of LMs 2.1) 2407.20311 | Operation count controlled | ? | Hidden-reasoning probes; no latent-CoT arm |
| **Multi-digit multiplication** (Deng 2311.01460, 2405.14838) | 4×4 to 9×9 digits | ? (repo has no licence) | Implicit CoT (stepwise internalisation) reaches 9×9 without text CoT. **Latent vs CoT: yes** |
| **CLUTRR** 1908.06177 | Kinship 2–10 hops (train 2–3, test to 10) | NOASSERTION / "unknown" | No latent-CoT result found |
| **StepGame** 2204.08292 | Spatial, k = 1–10 | MIT (GitHub, HF) | No latent-CoT result found |
| **ProofWriter** 2012.13048 / RuleTaker 2002.05867 | Proof depth 0–5 | ? | No latent-CoT result found |
| **CLRS-Text** 2406.04229 | 30 algorithms, size-scalable traces (every step checkable) | Apache-2.0 | No latent result found. **The best fit for step-level C7 checks** |
| **Learning to Execute** 1410.4615; CRUXEval 2401.03065 | Program length and nesting (L2E); CRUXEval short | ? / MIT | No latent-CoT arm |
| **Mazes**: maze-dataset 2309.10498; easy-to-hard 2108.06011 | Maze size and path length | LGPL-3.0 / MIT | Recurrent nets extrapolate small → large mazes (Bansal 2202.05826). Looped only, no CoT arm |
| **Sudoku-Extreme** (HRM / TRM repos) | Difficulty | Apache-2.0 / MIT | HRM 2506.21734, TRM 2510.04871 (recursion with halting) |
| **ZebraLogic** 2502.01100 | Grid size (search space) | Apache-2.0 | LLM CoT scaling limits; no latent arm |
| **Graph in NL**: NLGraph 2305.10037, GraphQA 2310.04560, Saparov generator 2412.04703 | Graph size and diameter | ? | Search generator used for looped / path-merging analyses |
| **Shapiro's retrofit task** 2608.11233 | Rule depth, to 18+ | — | **Recurrent vs scratchpad (text CoT):** matched within the horizon; beyond depth 10, 53% vs 2.5% |

**Picks.**
- **C3 vs text CoT:** GSM8K, multiplication, ProsQA/ProntoQA already have latent-vs-CoT baselines.
- **Clean depth sweeps beyond one pass:** CLUTRR, StepGame, extended ProsQA, CLRS-Text (checkable steps).
- CLUTRR's licence is unclear; check it before redistributing.

## B. Architecture precedents

**Edits to pretrained LMs (Experiments Q1).**

| Paper | Modification | Extrapolation? |
|---|---|---|
| **Shapiro 2608.11233** (preregistered) | Qwen2.5-0.5B split into prelude / weight-tied recurrent block / coda; identity one-loop path; re-entry bridge | **Yes: ~1.5× supervised depth** (70% through depth 18); scratchpad collapses beyond its horizon. Learned depth selection "remains open" |
| McLeish 2511.07384 | Retrofit recurrence with a curriculum of recurrences | Maths gain at matched compute; no extrapolation claim |
| LoopUS 2605.11011 | Encoder / looped block / decoder; selective gate against drift; random deep supervision; **confidence-head early exit** | Reasoning gains; extrapolation not claimed |
| Relaxed Recursive Transformers 2410.20672 | Layer-tying plus layer-wise LoRA | Efficiency only |
| Think-at-Hard 2511.08577 | LoRA latent iterations on hard tokens; attention across iterations | "Latent overthinking" otherwise |
| LOTUS 2606.31779 | Looped padded transformer | Matches CoT at 3B |
| Coconut / CODI / SIM-CoT; pause 2310.02226; Quiet-STaR 2403.09629 | Continuous or filler thoughts on pretrained GPT-2 / LLaMA | Horizon at the trained budget (THEORY §13; 2605.26733) |
| RMT 2207.06881 / 2304.11062 | Memory tokens carried across segments (pretrained) | Context length, not reasoning depth |
| Markovian Transformers 2404.18988; Delethink 2510.06557; MEM1 2506.15841 | Text-bottleneck / restart training (architecture-agnostic) | Delethink and MEM1 extend beyond the training horizon |

**Built-in restart / checkpoint modules and count-free recurrence (Audit b).**
- **Restart built in:**
  - HRM 2506.21734: the low-level module is reset each high-level cycle.
  - TRM 2510.04871: deep supervision continues from a detached carried state.
  - Transformer-XL 1901.02860, Block-Recurrent 2203.07852: segment-level state.
  - CTM 2505.05522: re-application from the current position.
- **Count-free recurrence:**
  - DEQ 1909.01377 (solve to a fixed point, no step index); path independence 2211.09961.
  - Recurrent Relational Networks 1711.08028 (loss at every step).
  - Recall / input injection 2202.05826; energy descent IRED 2406.11179.
  - Kavuncu 2609.19521: current-state-only BUT extrapolates, cache-keeping CoTFormer fails; gated LSTM-UT.
- **Learned write:** NTM 1410.5401 (write heads to external memory); MEM1 (RL-written state).
- **Counter-examples with explicit clocks:** LoopFormer 2602.11451; timestep encoding 2410.01405.

## C. Theory's additional questions

**Fit for GPT-2 small on 8 GB.** Every task in A is small-vocabulary text. ProsQA and GSM8K are already trained on GPT-2
here or in the literature (CODI and Coconut GPT-2; bmarti44 used batch 128). On 8 GB, use gradient accumulation plus
bf16. The best depth-extrapolation splits: CLUTRR (train 2–3, test to 10), StepGame (test k = 6–10 beyond training),
PrOntoQA-OOD, extended ProsQA, CLRS-Text (size-scaled). The natural discrete state to write: entities (CLUTRR, ProsQA),
positions (StepGame, mazes), registers or values (arithmetic, CLRS-Text, program traces).

**Segment-isolating attention** (later tokens see only written or compressed state, not earlier latents):
- **Gist tokens** 2304.08467: an attention mask forces later tokens to attend *only* to gist tokens. The cleanest
  mask-level precedent.
- LightThinker 2502.15589 (mask to compressed tokens); AutoCompressors 2305.14788 (summary vectors across segments).
- Compressive Transformer 1911.05507; RMT 2207.06881; Transformer-XL 1901.02860 (segments see a carried memory).
- Markovian Transformers 2404.18988 and Delethink 2510.06557 (text-only carry-over).
- Kavuncu 2609.19521: clearing CoTFormer's cache restores extrapolation.
- **Written entity checkpoints between latent segments: none found** (our CLR).

**Learned write heads over entity tokens, self-distilled from attention.**
- Pointing via attention: Pointer Networks 1506.03134; the pointer-generator copy mechanism 1704.04368 (supervised).
- Attention as a distillation target: MiniLM 2002.10957 (teacher → student attention).
- Pointer Graph Networks 2006.06380 (supervised pointers).
- **No head found that writes entity tokens trained by self-distillation from the model's own attention.** New as
  scoped (Pilot L / CLR-1).

**Novelty note.** Shapiro 2608.11233 is now the closest *pretrained-model* precedent for PROTOCOL8b / CLR-on-GPT-2. It has
a retrofitted latent loop, preregistered, beats text scratchpad beyond its horizon, and extrapolates ~1.5×. It has no
restart or checkpoint, no label-free training (it uses intermediate-step supervision) and no C7 test. Cite and
distinguish.
