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
