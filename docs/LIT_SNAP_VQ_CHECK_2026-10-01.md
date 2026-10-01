# Pre-registration check: §67 latent snapping (inference-time codebook projection) and the GPT-2 VQ latent loop (2026-10-01)

Literature session, for Theory; Audit copied. IDs verified (arXiv API).

## 1. Projecting or snapping recurrent states onto a learned set
- **State-Regularized RNNs**, Wang & Niepert 1901.08817: transitions are forced among a *finite set of learnable
  states* (centroids). This improves memorisation and length behaviour on balanced parentheses, and allows automata
  extraction. **The closest training-time precedent** for codebook snapping.
- **Quantized Bottleneck Insertion**, Koul et al. 1811.12530: quantised memory inserted into a *trained* RNN policy,
  giving finite-state representations with little loss. **The closest post-hoc precedent** for §67's codebook fitted on
  a trained model.
- **State reification**, Lamb et al. 1905.11382: at test time, project hidden states toward the training distribution.
  **The closest inference-time precedent** for "snap to the in-range manifold".
- State-Denoised RNNs, Mozer 1805.08394 (attractor clean-up); Hopfield 2008.02217; automata extraction by state
  quantisation, Weiss et al. 1711.09576; Discrete NAR 2402.11628.
- **New if scoped as:** inference-time snapping of a *latent reasoning loop's* thoughts to a per-step codebook, used to
  separate drift from count as the cause of length failure. The age-matched snap control is what makes it a diagnostic.
  **Medium novelty:** the mechanism is known; the diagnostic use is new.
- **Caution:** a codebook fitted *per step* is itself step-indexed. Snapping step k > K with which codebook? Use a pooled
  (step-agnostic) codebook as the primary, or snapping may reimpose the count you are testing.

## 2. VQ / discrete latent thoughts in LMs
- Token Assorted 2502.03275 (VQ-VAE latent tokens mixed with text); VQ-VAE 1711.00937; Soft-to-Hard VQ 1704.00648.
- **Discrete Latent Reasoning (DLR)** 2606.29712 (June 2026): converts continuous latent states into discrete,
  interpretable ones, arguing continuous latents are unstable because they lack step anchors. The closest recent LM
  precedent.
- DiscoLoop 2607.00341 (a discrete channel in a loop); Soft Tokens, Hard Truths 2509.19170 (soft-train, hard-test).
- **New if scoped as:** a quantised Coconut-style loop in GPT-2 on a must-use task, trained from final answers with a
  length curriculum against a no-latent twin.

## 3. Length or step curricula for continuous thoughts without CoT supervision
- **Abstract Token Curriculum** 2609.19717: continuous intermediate representations from a difficulty curriculum, *without*
  scratchpad supervision. **The closest precedent.**
- Zhang 2607.20594 (operator-first: n = 2 stage, then doubling); Fan 2409.15647 (variable steps, final-answer loss);
  Coconut's own curriculum uses CoT (not label-free); Latent-SFT 2510.15522.
- K = 1, 2, … curricula follow Fan and Zhang. Cite ATC as the continuous-thought, label-free precedent.
