# Prior check for ADD-NORM: adding linear attention's normaliser φ(q)ᵀz to DeltaMem's additive read (scale-invariance across think passes) (2026-10-10)

Literature session, for Audit (the lead) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **TransNormer:** details from its PDF (2026-10-10).

## 1. Katharopoulos et al. 2006.16236 (linear attention): the normalised form (confirmed)
- **The read:** V'_i = φ(q_i)ᵀ S_i / φ(q_i)ᵀ z_i, with S_i = Σ_{j≤i} φ(k_j) v_jᵀ (the memory) and z_i = Σ_{j≤i} φ(k_j) (the normaliser state), and
  **φ(x) = elu(x) + 1**.
- **The recurrent form:** S_i = S_{i−1} + φ(k_i) v_iᵀ, z_i = z_{i−1} + φ(k_i).
- **The denominator makes the read a weighted *average* of stored values**, so it doesn't grow with the number of writes. That's the property ADD-NORM wants.
- **DeltaMem's additive heads (deltamem.py:4, :90)** read φ(q)ᵀS scaled by exp(s_add), **without the z normaliser**. So the read grows roughly linearly
  with the number of writes (passes / segments).

## 2. Unnormalised state growth, and fixes
- **The opposite warning, "The Devil in Linear Transformer" (TransNormer, 2210.10340):**
  - **the scaling (the φ(q)ᵀz denominator) is "the devil in unbounded gradients"**, causing unstable convergence;
  - their fix, **NormAttention, *removes* the denominator and appends a normalisation (LayerNorm / RMSNorm) after the attention output**;
  - they also find attention dilution over long sequences and use diagonal (local) attention in early layers.
  - **This bears directly on ADD-NORM:** re-adding the denominator may bring back the gradient instability TransNormer documents. An **output RMSNorm on
    the read** is the established alternative.
- **Decay-based fixes for state growth:**
  - RetNet 2307.08621 (fixed exponential decay per head, with GroupNorm on head outputs);
  - GLA (data-dependent gates);
  - Gated DeltaNet 2412.06464;
  - AGCLR 2606.07720 (the forget gate matters most).
  - These bound the state by forgetting, which conflicts with exact accumulation (LIT_FIVE_LEVERS #4).
- **The length / coverage view:** 2507.02782 attributes recurrent models' length failure to unexplored (out-of-range) states. Growing state norms are one
  form of that.
- **cosFormer 2202.08791:** non-negative kernels plus cosine re-weighting. It keeps the normalised form, for quality and locality, not for length.
- **Not found:** a study showing specifically that an *additive, no-decay fast-weight read* without normalisation hurts *think-pass* extrapolation, and
  that the φᵀz denominator fixes it. **ADD-NORM's specific test is new.**
- **The mechanism is sound,** though: an unnormalised integrator's read scales with the write count, so a decoder calibrated at K passes sees
  out-of-range magnitudes at K′ > K.

## 3. STARS 2605.26733: does it target additive accumulators?
- **No.** STARS ("Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models", ICML 2026):
  - targets **depth-recurrent looped LMs**, whose accuracy peaks at some depth then collapses;
  - regularises the **Jacobian spectral radius of the loop map** (with random loop sampling) so latent states approach **asymptotically stable fixed
    points**;
  - shows reliable test-time scaling on arithmetic tasks, and less degradation on maths.
- **It acts on the loop's state-update map,** not on a fast-weight accumulator's read.
- **Applied to DeltaMem's additive heads,** a spectral-radius penalty would push the integrator (eigenvalue 1 by design) toward decay, which is again the
  exact-sum conflict.
- It's relevant to our think-pass stability (C6) as an alternative to HOLD / path independence (a Jacobian penalty vs targets). Not relevant to ADD-NORM
  directly.

## Recommendations
1. **Two arms if budget allows:**
   - (a) Katharopoulos's φ(q)ᵀz denominator (the lead as proposed);
   - (b) **an RMSNorm on the additive read output** (TransNormer's NormAttention), which gives scale-invariance without the denominator's gradient issue.
   - Pre-register gradient-norm logging on the additive path.
2. **Keep exact sums representable.**
   - The denominator turns the read into a mean, so a column *sum* must then be recovered by the decoder (the count is implicit).
   - Check that the carry-save accumulation still reads exactly. It may need z exposed as a separate feature.
3. **STARS** is a candidate C6 lever for the think-pass loop (spectral-radius regularisation with random K), separate from ADD-NORM. Note it as a possible
   follow-up beside HOLD.
