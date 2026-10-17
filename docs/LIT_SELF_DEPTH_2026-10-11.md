# Prior check for SELF-DEPTH: a self-labelled depth curriculum on §184, filtered by agreement between the aux chain's final value and the answer decoder (2026-10-11)

Literature session, for Audit (the lead) and Theory. IDs verified via arXiv abstract pages / OpenReview (2026-10-11).

## Richard, "Depth Extrapolation of Decoders Trained on Nested Structures" (NeurIPS 2024 Workshop M3L)
- **No arXiv ID found** (searched the arXiv API by title and by author). The canonical link is **OpenReview forum vGSWZwfbZu**
  (https://openreview.net/forum?id=vGSWZwfbZu), via mlanthology neuripsw/2024/richard2024neuripsw-depth.
- **Author:** Emile R. Richard.
- **From the abstract:** how next-token predictors learn deeply nested formal structures and whether they extrapolate to deeper nesting *within a single
  inference pass*.
  - Case study: Boolean-logic simplification. "A specialized decoder Transformer seems to perform well when it overfits", but fails to extrapolate to
    deeper nesting (abstract-level).
- **Cite as OpenReview / workshop, not arXiv.**

## The self-improvement baseline: Lee et al. 2502.01612, "Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges" (verified)
- Models iteratively **generate solutions to slightly harder problems, filter them, and train on them** (weak-to-strong curricula), with a standard
  architecture.
  - Arithmetic, strings, mazes; e.g. 10-digit → 100-digit addition without apparent saturation.
  - "Filtering for correct self-generated examples leads to exponential improvements in OOD performance across rounds." Pretrained starts speed it up.
- **The filtered self-labels are *token outputs*** (answers or explicit solutions), not latent states.

## Filtered self-training applied to *latent / looped* reasoning depth (2025–26)?
- **Not found.** Nearest:
  - **RLTT 2602.10520** ("Rewarding Latent Thought Trajectories" in looped LMs): RL with reward spread over the latent trajectory (Ouro 1.4B / 2.6B; maths).
    Reward-based, not filtered self-labelling of harder depths.
  - **Equilibrium Reasoners 2605.21488:** attractor-based test-time scaling (depth and breadth over stochastic initialisations; Sudoku-Extreme 2.6% → >99%).
    No self-training curriculum.
  - **Kohli 2604.07822:** a threshold depth curriculum with *gold* labels.
  - **Lee 2502.01612:** filtered self-training with *token* outputs on standard transformers.
- **SELF-DEPTH is new as scoped:** self-labelled deeper items for a latent think-pass loop, **filtered by internal consistency** (the aux chain's final value
  vs the answer decoder) rather than by majority vote or correctness checks on text outputs.

## Flags
1. **The agreement filter can be fooled by a shared error.** If the aux decoder and the answer decoder read the same faulty memory, they agree on wrong
   answers. Lee's filters include majority vote over samples and length checks. Add an independent check: agreement across two think-pass counts (K vs K′),
   or across two seeds.
2. **Lee observes error accumulation without filtering** (the curse of recursion on generated data). Log label accuracy against gold per round (gold is
   available for the synthetic Feng items) as a descriptive, so filter precision is measured, not assumed.
3. **Step size:** Lee moves to slightly harder problems each round. Use +1 operator per round, as the 187-2 count dependence suggests large jumps fall off the
   learned procedure.

## Addendum (2026-10-17): full-text check of the two self-improvement references, for the SELF-DEPTH-CHAIN draft (Theory, claude-02)
- **IDs:** both verified via arXiv abstract pages. **Sources:** both read in full text (pdftotext), not the HTML.

### Lee, Cai, Schwarzschild, Lee & Papailiopoulos 2502.01612, "Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges" (EMPIRICAL)
- **Who labels:**
  - The previous-round model labels the new, harder inputs with its own greedy outputs: D_r = {(x_i, M_{r-1}(x_i))}, d_{r-1} < Difficulty(x_i) <= d_r.
  - The model trains on D_0 ∪ … ∪ D_{r-1}, with the newest set up-sampled to 50%.
- **Filters (both unsupervised):**
  - **Length filter:** "predictions shorter than a predefined threshold--calculated relative to the maximum prediction length within their batch--are filtered out". OOD failures tend to be short.
  - **Majority vote across independently trained models,** e.g. 5 seeds for multiplication, or keeping only outputs "agreed upon by all 3 models" for mazes.
- **Schedule:** +1 difficulty unit per round. For addition, "incrementally extending the operand length by one digit"; round r labels length 10 + r; 50k new examples per round; about 60–100 rounds.
- **From scratch vs pretrained:** small models trained **from scratch** (NoPE, character-level tokens), "except for the experiments on pretrained Llama 3.2 models" (Llama-1B / 3B). Pretrained models "significantly accelerate" self-improvement.
- **Tasks and gains:** reverse / forward addition, string copy / reverse, multiplication, maze shortest paths.
  - Reverse addition: 10-digit → 100-digit "without apparent saturation".
  - Forward addition with the length filter: up to 75 digits over 60 rounds.
  - Strings: lengths 1–10 → over 120 after about 100 rounds.
  - Mazes: up to 30 hops with majority voting.
  - "In some cases" exponential gains across rounds.
- **Failure mode:** an **error avalanche.** Without filtering, multiplication and mazes collapse ("self-generated training data degrades over successive rounds").
  - Injected synthetic label noise is tolerated up to a threshold, which is higher in later rounds, and beyond it accuracy crashes.
  - Easy tasks (reverse addition, copying) need no filtering.

### Huang, Wen, Singh, Chi & Chen 2511.07378, "Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization" (NeurIPS 2025 per the abstract-page comments; THEOREM + EMPIRICAL)
- **Model / regime:** a **one-layer transformer** (softmax attention + FFN), **NoPE**, trained **from scratch by gradient descent** with CoT supervision, on synthetic **LEGO state-tracking** tasks.
  - Simply transitive group actions (e.g. cyclic C6) **length-generalise directly**.
  - The symmetry-group action (S_n on Z_n) generalises only by a **constant factor**.
  - Mechanism: attention concentration.
- **Recursive self-training (THEOREM 1.3, validated in experiments):** at each stage the current model **greedily generates the full answer traces (self-labels) for inputs of DOUBLE the length**, and retrains on these pseudo-labels.
  - Experiment: L = 5 with ground truth, then 10 → 20 → 40 on self-labels.
  - After O(log d) stages it solves length d.
  - **No filter.** The guarantee rests on the model already being correct at 2× its trained length, the constant-factor generalisation of Thm 1.2.
- **Label-noise compounding:** not analysed as a failure mode. Correctness of the self-labels is implied by the theorem's precondition, and the paper notes only that errors can accumulate during autoregressive rollout at evaluation.

### Is either analogous to our labeller (a single model's re-rooted single-step chain: decode, re-render, re-encode each step; filter = parses at every step + exactly n steps)?
- **Single model:** Huang uses a single model's own greedy traces, as we do. Lee's strongest filter is a **multi-model** majority vote, which we don't use.
- **Re-rooting: NOT in either.** Both label with **whole autoregressive traces**: the model continues its own text and never re-renders an intermediate result as a fresh problem to re-encode. Our decode → re-render → re-encode per step has no counterpart; it's closer to tool-style step execution than to either paper.
- **Filter:**
  - "Exactly n steps" is a structural, unsupervised validity check, the same kind as Lee's **length filter** (output length as a validity proxy), but stricter: an exact count vs a batch-relative threshold.
  - "Parses at every step" has no counterpart in either; it's a per-step format check.
  - Neither paper verifies values.
- **Schedule:**
  - Lee's +1 per round matches our +1 operator / step plan.
  - Huang's doubling is licensed only by a proven constant-factor generalisation margin. Use it only if a measured margin supports 2×.
- **Risk to carry into the draft:** Lee's error avalanche.
  - Measure filter precision (true correctness of kept labels, available for synthetic Feng items) each round.
  - Stop or roll back when it drops below the noise tolerance the run itself shows. Lee's tolerance threshold grows over rounds.
