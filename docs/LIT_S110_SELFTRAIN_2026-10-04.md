# Prior check for §110: filtered self-training by length, independent pools, a self-consistency precision gate (2026-10-04)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-04); co-training
classics are non-arXiv.

## 1. Is chain-vs-restart self-consistency a known label-free precision proxy?
- **Yes, as multi-view agreement, with a caveat.**
  - Co-training (Blum & Mitchell 1998) and the agreement-based PAC bounds (Dasgupta, Littman & McAllester 2001):
    disagreement between two views bounds error **only if the views are conditionally independent**.
  - Jiang et al. 2106.13799: disagreement between two independently trained runs predicts test error.
  - LLM-side: self-consistency (Wang 2203.11171, majority over sampled chains) used as a pseudo-label filter in
    self-improvement (Huang 2210.11610).
  - Our M-2 agreement filter.
- **Caveat:** chain and restart decode share weights, so their errors are correlated. Agreement then *over*-estimates
  precision, the same failure §103's shared-pool vote showed.
  - The gate is a necessary condition, not a precision estimate. Keep the logged truth-scored precision alongside it
    (diagnostic only), and report the gap between agreement and precision per round.

## 2. Priors for a final relabelling round (iterated pseudo-labelling)
- Noisy student 1911.04252 (iterate teacher → student with noise).
- STaR 2203.14465: **each outer iteration restarts fine-tuning from the original model**, to limit drift.
- ReST 2308.08998: a grow / improve loop with filtered self-generated data.
- Lee 2502.01612: rounds plus vote and length filters.
- **Confirmation bias:** Arazo et al. 1908.02983. Pseudo-labelling reinforces its own errors; the mitigations are mixup,
  a minimum labelled fraction, and soft labels.
- Suggestion: train the final relabelled round **from the round-0 checkpoint (STaR-style)**, not on top of the
  accumulated model, so stale label errors aren't compounded.

## 3. Risks
1. **A stalled gate.** If shared-weight paths disagree for benign reasons (format, rare positions), or agree wrongly
   (correlated errors), the rate may never cross 0.995. Then the gate either stalls coverage or passes noise.
   - Cap the number of intermediate-length insertions (e.g. ≤ 2 per round).
   - Report gate trajectories.
   - Make "gate never passes" a pre-registered outcome, not a silent stop.
2. **Selection bias in the kept set.** Keeping only final-correct, consistent problems can over-sample easy op
   sequences (Lee's length filter guards the length dimension only). Report the op/transition distribution of kept vs
   all problems per round.
3. **Independent pools raise seed variance. That's the honest C6 test.** Pre-register the pass as ≥ 4/5 seeds, and
   report the median and range. The shared-pool vote (§103) was lower-variance only because errors were shared.
4. **C1:** the masked fork removes s2-style fork bypass at decode, but slot recomputation from visible ops remains
   possible (LIT_S105_106). The donor protocol must show USE at j−1 → j. **C4** stays qualified by ≤ 5% short
   problems (round 0).
