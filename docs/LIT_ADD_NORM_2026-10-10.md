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

## Addendum (2026-10-10): Theory's four points for §185
1. **Confirmed (section 1):** V' = φ(q)ᵀS / φ(q)ᵀz, z = Σφ(k), φ = elu + 1 (Katharopoulos 2006.16236).
2. **Yes, later work dropped the denominator:**
   - **TransNormer 2210.10340:** the scaling causes unbounded gradients and unstable convergence. NormAttention removes it and normalises the
     *output* (LayerNorm / RMSNorm).
   - **RetNet 2307.08621:** decay plus GroupNorm on head outputs.
   - **Schlag et al. 2102.11174 (fast-weight programmers):** keep a normalisation but apply it to the **keys** ("sum normalisation": φ(k) scaled to sum
     to 1 before writing). That avoids the query-dependent denominator, and they pair it with the delta rule.
   - **Three non-denominator options, then:** output norm, key sum-normalisation, decay.
3. **A looped / recurrent-depth model normalising a fast-weight read for loop-count invariance: not found.**
   - Looped LMs normalise the *residual state* (Huginn's RMSNorm sandwich and embedding scale).
   - **MELT 2605.07721** shares one gated KV cache across loops (a gate, not a normaliser).
   - Schlag's sum normalisation targets sequence length, not loop count.
   - **§185's loop-count invariance use is new.**
4. **STARS: arXiv 2605.26733,** "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models".
   - The abstract-page comments say **ICML 2026**.
   - It's Jacobian spectral-radius regularisation with random loop sampling, so states approach asymptotically stable fixed points.
   - It doesn't target additive accumulators, because it constrains the loop map's spectrum. On an integrator (eigenvalue 1 by design) it would force
     decay, so it can't keep exact sums.

## Addendum 2 (2026-10-10, Audit's 22:45 lead): "Stuffed Mamba" 2410.07145 as the RNN-length analogue of NORM-CAP
- **The paper:** Yingfa Chen, Xinrong Zhang, Shengding Hu, et al. (Tsinghua), "Stuffed Mamba: Oversized States Lead to the Inability to Forget",
  **COLM 2025**. Verified via the arXiv abstract page and HTML.
- **The phenomenon, a direct analogue of ours:**
  - Mamba-2 **fails to learn to forget** when trained on contexts too short for its state size. Some heads keep the first token's cumulative retention
    above 0.997 at 8K.
  - **The state's mean and variance "exhibit a clear explosion when t is greater than the training length"** (their Fig. 5).
  - Beyond the training length, interference from un-forgotten tokens degrades output.
  - The minimum training length needed to learn forgetting scales linearly with state size.
- **Their training-free mitigations (both are state-update interventions, Sec. 3.2.2 and after):**
  1. **RRI (Reduced Memory Retention and Insertion):** scale the decay α_t by **0.9999** and the insertion B_t by **0.75**, chosen on 32K validation loss.
  2. **Sliding window by state difference:** h_t^(r) = h_t − α_{t−r+1:t} h_{t−r}, i.e. keep only the last r tokens' contribution, without
     reprocessing.
  - **Neither is an inference-time norm cap or state normalisation.** RRI is the closest: it shrinks what is written, and so how large the state can grow.
- **How to cite NORM-CAP:**
  - "a think-pass (loop-count) analogue of the known state-explosion beyond training length in recurrent LMs (Stuffed Mamba 2410.07145; cf. the
    unexplored-states account, 2507.02782)";
  - mitigated here by **an inference-time cap on the additive state's norm, derived from training statistics**.
  - **The phenomenon is known. The cap as the mitigation, and its application to latent think passes over a fast-weight memory, are not found in the
    literature.** Don't present the state-growth finding itself as new.
- **Mechanistic difference worth one line:** their states grow from *un-forgotten tokens* (sequence length). Ours grow from *repeated additive writes
  across think passes* (loop count) with no decay at all. RRI's insertion scaling is the nearest analogue to ADD-NORM's divisor / RMS arms.
