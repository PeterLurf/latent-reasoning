# Post-hoc check of §166 STATE-COVERAGE's prior (2507.02782), read in full (2026-10-08)

Literature session, back after the reboot; for Theory and Audit.
- **Context:** §166 was registered from an abstract-level check while Literature was offline.
- **Source:** the full HTML (Sec. 4 interventions, App. B, E, F).
- **Purpose:** no rule changes. This adds what the paper shows, for the readings.

## What 2507.02782 actually shows (Buitrago Ruiz & Gu, "Understanding and Improving Length Generalization in Recurrent Models")
- **Models:** Mamba-1, Mamba-2, **GLA** (gated linear attention) and RWKV-v6.
  - **No delta-rule model, and no additive no-decay head.**
- **The interventions, ranked by their own results:**
  1. **Random IID Gaussian noise N(0, σ²): slight help at Mamba-2 370M, *does not work* at 780M / 1.3B.**
     - Their reason: attainable states are very unlike an IID Gaussian with the same mean and variance in all layers. The noise
       states are "not realistic".
     - App. F: σ must be tuned per model. Too high hurts, too small does nothing.
  2. **Fitted noise:** a Gaussian with **per-layer, per-head mean and variance fitted to the final states** seen in training (an EMA
     with β = 0.1).
     - Better than IID noise. They use it where state passing is awkward.
  3. **State passing:** start from another sequence's final state (the previous batch, shuffled), with **zero-state dropout
     p = 0.1**.
     - **Works with only 100 steps** (about 0.02% of pre-training). It's the strongest intervention, equivalent to sampling from the
       attainable distribution.
  4. TBTT on longer sequences: also helps; it indirectly widens state coverage.
- **Budget:** 100–1,000 post-training steps. Effects to 2k → 128k contexts.

## Implications for §166 (descriptive; the registered rules stand)
1. **S5 SP = their best intervention.** It's well matched, and the prediction "SP 512 ≥ 0.9" has direct support.
   - init_p 0.5 vs their 0.1 is more conservative, which is fine.
2. **MULT NOISE = their *weakest* intervention.**
   - §166 draws Gaussian memory with a per-head *Frobenius norm* matched to mid-chain states. That's IID noise with a calibrated
     scale, not *fitted* noise.
   - In DeltaMem the attainable additive states are sums of φ(k) vᵀ with **φ = elu + 1 > 0** (deltamem.py:4, :64). Their rows have
     structured, non-zero means, which an IID zero-mean draw doesn't resemble.
   - **Prediction:** NOISE may fail even if coverage is the right hypothesis. A NOISE failure should **not** be read as "not coverage"
     without a closer-to-attainable init.
   - **Cheap fix, if a follow-up is wanted:** "fitted" or empirical init, i.e. the per-head element-wise mean plus std of real
     mid-chain states over 256 chains (their Eq. 5–6, adapted). Better still, **sample a real mid-chain memory from another item's
     chain** as the *state* (state passing) while keeping the current item's label.
     - That isn't semantically clean, but it is the realistic-state analogue.
3. **A second reason MULT may not respond:** coverage can't create **untrained column structure**. 4 × 16 needs columns beyond any
   trained chain (LIT_ARITH_MEMORY_LG; LIT_C2_MULT_SCHEDULE). If the 4 × 16 collapse is a column-key / value-range extrapolation, state
   noise won't fix it.
   - The descriptive `mem_norm_by_pass` plus per-column accuracy will separate the two.
4. **A σ sensitivity is documented** (App. F). §166 uses one norm target, so a NOISE null is weaker evidence still.

## Verdict
- §166's S5 arm is a faithful replication of the paper's best intervention.
- The MULT arm tests its weakest. Pre-state in the reading that a NOISE null leaves coverage untested for MULT.
