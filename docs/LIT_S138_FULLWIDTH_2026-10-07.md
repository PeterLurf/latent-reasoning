# Prior check for §138 FULL-WIDTH: a bounded learned 768-d carrier (FSQ / NOISE / CONT / GROW) vs §126 RAW-H (2026-10-07)

Literature session, for Theory; Audit copied. Brief. Plan: docs/THEORY_FULLWIDTH_PLAN_2026-10-07.md. IDs verified via the arXiv API
(2026-10-07) or earlier notes (LIT_INDEX).

## 1. Is "a bounded learned projection at residual width" vs "the raw hidden" a known stability lever?
- **Yes, as an engineering choice. It has not been isolated as a causal stability variable.**
  - Coconut 2412.06769 feeds the raw last hidden state.
  - CODI 2502.21074 adds an MLP + LayerNorm projection (a small gain in its ablation).
  - SoftCoT 2502.12134 adds a learned projection module.
  - HRPO 2505.18454 normalises / gates the hidden state, because raw hidden states give off-distribution inputs.
  - Soft Thinking 2505.15778 stays inside the convex hull of token embeddings (bounded by construction).
  - nGPT 2410.01131 puts all representations on the hypersphere (norm control as a training-stability lever).
  - Ours: §73 standardisation.
- **None of these compares raw vs bounded at a matched width on long-chain drift.** FULL-CONT vs RAW-H does: both are 768-d and
  continuous, and the differences are only the learned bounded code and COPY (plan item 1 + 3).
  - **Two changes, so a pass can't be attributed to bounding alone.** If budget allows, a FULL-CONT without COPY (with restarts)
    separates them.
  - Otherwise state that the contrast is "bounded code + COPY".

## 2. Any precedent for FSQ at d = 768?
- **None found.** Implicit-codebook quantisers stay low-dimensional:
  - FSQ 2309.15505 uses a few dimensions (about 3–10);
  - LFQ (MAGVIT-v2 2310.05737) and BSQ 2406.07548 are binary over tens of dimensions.
  - Per-coordinate rounding at 768 is untested, so **FULL-FSQ is new.**
- **What changes at 768:**
  - "Codebook" statistics (utilisation, perplexity) are meaningless (8^768). **Report codes per state and the number of dimensions
    actually used** (std above a floor), as the plan's post-mortem lists.
  - Exact closure / Moore minimisation is still well-defined on the codes *visited*. It fails if a state spreads over many codes
    (a few coordinates flicker across rounding boundaries). **Codes per state is the decisive diagnostic.**

## 3. Pitfalls at full width
1. **The rounded-arm ρ clause will switch at 768. Fix the reading now.**
   - `contraction()` (fsq120.py:1031) perturbs every coordinate by 0.1·std_d.
   - Exact recovery requires *all* 768 coordinates to round back. Even if the per-coordinate stability matches FSQ-64, the
     `exact_recovery_share` will fall roughly geometrically with d.
   - So 127-12's "ρ₁ when rounding recovers exactly" will rarely apply, and FULL-FSQ will be read on ρ_ff / `rho_nonzero`.
   - Pre-register that, so a low recovery share isn't read as instability.
2. **At 768, random-direction ρ is a bulk average.**
   - A Gaussian δ puts about 1/√d of its norm on any given state-separating direction, so the median norm ratio is dominated by
     idle directions.
   - It can read contractive while the few directions that separate states expand (the anisotropy in 2609.26487).
   - **Add a descriptive ρ along centroid-difference directions** (δ ∝ z̄_s − z̄_s′ for nearest state pairs), or a few
     power-iteration steps for the top Jacobian gain (spectral-norm regularisation 1705.10941). Report it beside the registered ρ,
     not as a gate.
   - The same applies to RAW-H's 0.85 at 512: its drift may live in a few directions.
3. **Entry.**
   - RAW-H entered 4/4 at 768, so width alone doesn't block entry.
   - The 64-d flags (seeds 1 / 101) were without COPY, and COPY fixed d 8.
   - Risk specific to FSQ: at init the per-coordinate spread must span more than one level, or most coordinates round to a
     constant code. Log the initial code diversity (the share of coordinates not at the modal level) at step 0 and 2k.
   - "No restarts" is right for the pass rule, but log the 2k flag per seed.
4. **The hinge's cost** (if a hinge arm is added; the plan has none):
   - one extra segment forward per hand-off on the hinged steps (the trunk dominates, so roughly +50–100% of latent-path compute
     when it is applied every step; apply it on a random subset of hand-offs to cap the cost);
   - ε as a fraction of the measured nearest-centroid distance in 768-d (LIT_CAND5_6), not absolute;
   - random-δ hinges at 768 have the same bulk-average weakness as pitfall 2, so draw δ partly along centroid differences.
5. **Precision.** 768 × 768 W_down / W_up under autocast. Run `contraction()` in fp32 (as 123-x) and keep the tanh pre-activations in
   fp32 near the rounding boundaries, so that fp16 jitter doesn't flip codes.
6. **The benefit test (rule ii) is the real question.** S5 can't show gain. Name the capacity task and its budget at registration
   (S5×4 or multiplier-major). Otherwise "stable but no gain" is the expected outcome.

## 4. Novelty
- **New as scoped:**
  - a residual-width, bounded, learned carrier (rounded, dithered or continuous) in a pretrained LM;
  - scored by donor SERIAL, ρ and exactness at 2,048 against a raw-hidden control at the same width.
- The pieces are known: Coconut's raw hidden, CODI / SoftCoT projections, FSQ, nGPT's norm control.
- FSQ at 768 has no precedent.
