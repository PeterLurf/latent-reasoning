# Check of Audit's §177 candidate arms: (A) STEP-MATCHED K = T(n) with INJECT, and (B) the oracle-free Kohli recipe (2026-10-09)

Literature session, for Audit and Theory. IDs verified via arXiv abstract pages; details from the HTML of 2409.15647, 2606.29983 and 2604.07822
(2026-10-09).

## (A) STEP-MATCHED K
- **Fan et al. 2409.15647: confirmed.**
  - Each training example carries a pre-defined number of steps T(n), "diversely distributed in the dataset".
  - The original input is injected with the previous block's output at every step.
  - There is no intermediate supervision ("we do not require any intermediate supervision data").
  - Per-task T(n): parity, addition and copy have T = n; binary multiplication has T = the product of the lengths.
  - At test, T is adaptive, with a halting criterion.
- **A correction to the depth-leak citation.** "Prop. 4(b)" is **our** Proposition 4 (THEORY archive §22.2, "tightness"), not a proposition of Kuo's.
  - Kuo et al. 2606.29983 supply the *length-matched schedule K = n* (their Fig. 1, next to fixed K = 20). They show deterministic K = n
    extrapolates but with high OOD variance, from the **spurious length ↔ loop correlation**. Stochastic K in a window around n stabilises it.
  - Our Prop. 4(b): with a *tight* budget (K = the steps needed), every zero-loss schedule must run at unit speed. That's the benefit.
  - LIT_F1S_CHECK's caveat is the other side: a per-item K tied to the target is information the model can exploit as a count.
  - **Cite as "Kuo's K = n schedule; our Prop. 4(b)".**
- **For INJECT on nested arithmetic:** depth is readable from the text itself (bracket nesting). So K = T(n) leaks nothing the input doesn't already
  show. The real risk is Kuo's: **a deterministic K = T(n) gives an unstable OOD.** Use a stochastic window around T(n) (Kuo) and report variance over
  seeds.

## (B) Oracle-free, Kohli et al. 2604.07822 ("Loop, Think, & Generalize"): confirmed
- **Authors:** Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, et al.
- **Recurrent-depth transformers trained from scratch** on implicit multi-hop composition over knowledge, compared with vanilla transformers.
- **The dynamic model samples R ~ clip(Poisson(λ), R_min, R_max)** per batch, with **R_min = 2, R_max = 8** by default. **λ isn't stated in the
  passages read**; take it from their appendix or code if the arm copies it.
  - "Without design elements such as input injection, gated halting, and middle looping": **no input injection.**
- **Threshold curriculum:** train on atomic + 2-hop facts until **95%** on a held-out 2-hop split, then add 3-hop data until 95% on 3-hop, and so on.
  "Training facts beyond the model's capability are never exposed."
- **A three-stage grokking dynamic:** memorisation → in-distribution generalisation → systematic generalisation (on held-out compositions).
- **Depth extrapolation:** from training on up to about 5 hops to deeper (e.g. 10-hop) compositions, via test-time recurrence.

## Main question: per-item T(n) plus a *persistent memory* (not a re-read sequence state) on nested arithmetic?
- **Not found.**
  - Fan (per-example T(n)) and Kuo (K = n) carry the **sequence residual / re-read state** with injection, on algorithmic tasks.
  - Kohli carries the residual (no injection), on multi-hop knowledge.
  - The fast-slow model 2604.01577 keeps a **persistent latent**, but with a fixed T inner steps per observation, on streaming tasks.
  - Feng 2305.15408 does nested arithmetic only with text CoT vs direct (no loops).
  - Saunshi 2502.17416 loops on synthetic maths (i-GSM) without a separate memory.
- **New as scoped:** per-item (or windowed-stochastic) K = T(n) with a DeltaMem persistent memory as the only cross-pass state, on Feng-style nested
  expressions.

## Notes for the §177 arms
1. **A:** randomise K in a window around T(n) (Kuo) rather than exactly T(n). Report OOD-depth accuracy *and* its variance across seeds.
2. **B:** Kohli's 95% threshold curriculum matches our §162 / §163 depth curricula. Their grokking suggests long training past the in-distribution
   plateau may be needed before OOD depth appears, so budget for it, or the arm reads as a false null.
3. **Both:** the stack-depth flag for streamed nested expressions (LIT_TC0_SERIAL addendum) applies. Under INJECT the full text is visible every pass,
   so it's the *memory* that must hold pending partial results across passes. Bound and report the nesting depth.
