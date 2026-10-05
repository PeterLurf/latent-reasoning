# Prior work for task-agnostic latent reasoning: no snaps, or automatic snaps without hand-coded designs (2026-10-04)

Literature session, for Theory, on the user's directive via Audit (GSM8K paused; no task-specific state templates or
fixed schedules). IDs verified via the arXiv API, Semantic Scholar or abstract pages (2026-10-02..04; LIT_INDEX).
**Multi** = shown on more than one task family. "Per-task format?" means hand-built state or trace templates. Facts
not read in the full paper are hedged.

## (a) Stable long latent recurrence without discrete checkpoints
| Work | Mechanism | Multi? | Per-task format? | Transfer to GPT-2 |
|---|---|---|---|---|
| Huginn 2502.05171 | recurrent depth with input injection, random-r training, convergence-based exit | ✓ (maths, code, general) | no | loop a middle block per step with input injection; random r |
| Ouro 2510.25741 | looped LM at scale | ✓ | no | ⚠ degrades beyond its trained depth |
| TRM 2510.04871 | tiny recursive refinement with deep supervision | ✓ (Sudoku, Maze, ARC) | grid formats | deep supervision per iteration |
| **Attractor Models 2605.12466** | fixed-point refinement, implicit differentiation | ✓ (LM, Sudoku, Maze) | no | **per-step fixed-point refinement of the carried latent**: the no-snap "restoring map" |
| DEQ 1909.01377; Implicit LMs 2502.07827 | equilibrium layers; fixed-point iteration gives state tracking | ✓ | no | an equilibrium carry |
| 2609.26487 | finished recursions are contractive along the motion direction (anisotropic) | Sudoku-type | — | diagnostic for drift and halting |
| Howe 2609.18966; DeltaProduct 2502.10297 | norm-preserving input-driven transitions; additive path parasitic | groups, parity | no | an orthogonal carry channel; drop the additive input |
| nGPT 2410.01131 | all states on the hypersphere | ✓ (LM) | no | a norm constraint on carried latents |
| Professor forcing 1610.09038; SimSiam 2011.10566 / BYOL 2006.07733 | match free-running to teacher-forced dynamics; stop-grad self-distillation needs asymmetry | ✓ | no | §118 ANCHOR (text-path stop-grad target) |
| 2510.14095 | error-correction noise on latent states | one task | factored symbols (task-specific) | add noise late |
| LIFT 2609.38149; Next-Latent Prediction 2511.05963 | teacher-forced state targets; self-supervised next-latent consistency | ✓ (LM, reasoning, planning) | no | **task-agnostic state supervision** for a carried latent |
| Kuo 2606.29983; Kohli 2604.07822 | randomised stopping stabilises looped extrapolation; overthinking degrades it | ✓ (algorithmic) | no | random-length training |

## (b) Learned, task-agnostic discretisation of latents
| Work | Mechanism | Multi? | Per-task format? | Transfer |
|---|---|---|---|---|
| VQ-VAE 1711.00937 (+ collapse fixes 2005.08520); **FSQ 2309.15505** | learned codebook / fixed scalar grid | ✓ | no | **FSQ snap: no codebook collapse, no task template** |
| **SR-RNN 1901.08817** | the state transitions between learnable centroids | ✓ (formal languages, sentiment) | no | the closest "learned snap" for recurrence (our SELF-SNAP) |
| QBN 1811.12530 | post-hoc quantised bottleneck | ✓ (RL policies) | no | — |
| Token Assorted 2502.03275 | VQ codes replace early CoT steps | ✓ (maths, logic) | no | — |
| **Abstract-CoT 2604.22709** | reserved code vocabulary; bottleneck ↔ self-distil, then RL | ✓ (maths, instruction-following, multi-hop) | no | a learned discrete vocabulary between text and latent |
| CIRF 2605.28292; DLR 2606.29712; Latent-SFT 2510.15522; Planning tokens 2310.05707 | functional or clustered discrete thought units | ✓ | no (learned from traces) | — |
| Discrete KV bottleneck 2207.11240 | quantised key-value memory | ✓ | no | — |

## (c) Learned self-summarisation and context resets
| Work | Who decides *what* / *when* | Multi? | Per-task format? |
|---|---|---|---|
| **Delethink / Markovian Thinker 2510.06557** | what: the model, via RL (the carry-over text); **when: a fixed chunk size** | maths (several models) | no template; fixed schedule |
| **InftyThink 2503.06692** | what: summaries learned from reconstructed data (OpenR1-Math → 333K); when: per its iterative format (likely segment-length based) | maths, GPQA | a general summary format |
| **PENCIL 2503.14337** | **when (CALL/SEP/RETURN) learned from traces**; what: the reduction result | SAT, QBF, Einstein | ⚠ traces from per-task solvers |
| Xu 2604.25166 | PENCIL scaffolding on random MicroPy programs | OOD to arithmetic / SAT | a general program format |
| **LightThinker 2502.15589** | compression into gist tokens, timing learned through data construction | ✓ (4 datasets, 2 models) | no |
| MEM1 2506.15841 | an internal state block updated each turn, via RL | ✓ (multi-objective QA, web shopping) | a per-turn schedule |
| ReSum 2509.13313 | periodic summarisation for web agents (ReSum-GRPO) | agent tasks | a fixed trigger |
| Markovian Transformers 2404.18988 | the answer from the model's own CoT, via RL | ✓ (arithmetic, GSM8K, …) | no |
| AutoCompressors 2305.14788; Gist 2304.08467 | learned summary vectors / gist tokens for context | ✓ (LM, ICL) | no; segment-based |
| Thinking States 2602.08332 | text thoughts per chunk, re-embedded | parity, vars | fixed chunking |

## (d) Learned halting / checkpoint timing across tasks
- **Learned or adaptive halting:**
  - ACT 1603.08983 (multi); PonderNet 2107.05407; Universal Transformers 1807.03819;
  - **MoR 2507.10524** (token-level learned recursion depth, LM);
  - CoTFormer 2310.10845;
  - Huginn's convergence exit;
  - **CALM 2207.07061** (calibrated early exit; summarisation, QA, MT: multi).
- **Confidence-based:** ThinkRouter 2602.11683 (multi), DEER 2504.15895. ⚠ Our §83: confidence fails out of range.
- **Learned switching policies:** SWITCH 2606.13106 (learned ⟨swi⟩ boundary tokens + GRPO); TARPO 2606.05859
  (router); Tyler 2606.16360 (RL invocation with a budget-aware reward); PLaT 2601.21358 (answer-aware stopping).
- **Signals that need no task template:** state-change residual / completion (2609.26487), reconstruction or
  round-trip discrepancy (ReLAT 2606.06252; our §115), and validity checks.

## What fits the directive (task-agnostic, same code across task families)
1. **No snaps:** a carried latent with a **task-agnostic stabiliser**:
   - per-step fixed-point refinement (Attractor / DEQ);
   - a norm or hypersphere constraint (nGPT);
   - an orthogonal input-driven carry with no additive path (Howe);
   - **self-supervised next-latent / teacher-forced state consistency** (LIFT, NextLat) instead of a task state
     template.

   All are multi-task in the literature except Howe.
2. **Automatic snaps without templates:** snap to a **learned discrete vocabulary** (FSQ or a VQ codebook; Abstract-CoT
   style) at **learned or signal-triggered** times (MoR/ACT-style router, or a residual / round-trip trigger), with no
   per-task state text.
   - Closest multi-task precedents: Abstract-CoT (codes), LightThinker (learned compression timing), Delethink
     (learned content, fixed timing).
3. **Avoid:** fixed chunk schedules (Delethink, Thinking States, our k), task-specific trace generators (PENCIL),
   confidence triggers (§83).
4. **Gap no one fills:** a learned *when* and a learned *what* together, task-agnostic, in latent space, with a
   C1-style causal test. No prior work found.
