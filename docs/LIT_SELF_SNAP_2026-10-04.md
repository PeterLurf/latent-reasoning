# Prior check for §101 SELF-SNAP: a codebook learned online from the model's own latents, used as a soft snap (2026-10-04)

Literature session, for Theory; Audit copied. Brief. IDs verified (arXiv API / abstract pages / earlier notes).

## 1. Closest priors (learned codebooks or attractors that pull a recurrent state back)
- **State-Regularized RNNs, 1901.08817 (ICML'19): the closest.** The RNN state transitions between a finite set of
  *learnable* centroids, via a stochastic assignment between cell applications. This makes automaton extraction easy
  and improves behaviour on tasks needing long memory. A learned codebook used as a recurrent snap, from scratch.
- **QBN 1811.12530:** quantised bottlenecks inserted post hoc into trained recurrent policies give finite-state
  representations with little loss.
- **VQ-VAE 1711.00937** (EMA codebooks, commitment loss). **Token Assorted 2502.03275** (VQ codes for thoughts, but
  not as a drift correction).
- Modern Hopfield networks 2008.02217 (attention as a pattern-retrieval update: a continuous clean-up). Attractor
  Models 2605.12466 (fixed-point refinement).
- Our own §67 (LIT_SNAP_VQ_CHECK): a codebook projection at inference; hard codes don't decay with length.
- **Not found:** an *online-learned* codebook snap in a pretrained LM's latent reasoning chain, for drift correction.
  New as scoped; closest is SR-RNN.

## 2. Pitfalls
1. **Codebook collapse** (few codes used). Standard remedies:
   - k-means initialisation and periodic re-initialisation of dead codes (Łańcucki et al. 2005.08520);
   - EMA updates;
   - or avoid learned centroids altogether with **FSQ 2309.15505** (a fixed scalar grid, no collapse). FSQ is a cheap
     alternative arm.
   - Report code usage and perplexity.
2. **Coverage of drift.** Centroids fitted on in-range latents only cover the manifold the model visits. When a latent
   has drifted far, its nearest centroid is arbitrary, so the snap *locks in* an error rather than correcting it.
   - That is inherent to any snap. Measure it: the per-length purity of snapped states and the drift distance at
     failure.
3. **Stale-centroid pull.** A commitment loss pulls u towards centroids learned early, which can freeze the latents.
   Use a small commitment weight, start the codebook after the hand-off works, and keep EMA decay moderate.
4. **Errors reinforcing themselves.** Centroids are fitted to the model's own latents, *including wrong ones*. A
   systematic error can get its own centroid and then be stabilised. With M = 240 > |G| this is possible.
   - Check **centroid purity**: the mutual information between centroid and true state, a certificate-style check on
     the learned codebook.
5. **Scope.** Your latents are shaped by per-op text targets (the side decoder), so they are state-separable by
   construction. The codebook is "self-discovered" only in the clustering. It is not C4.

## 3. Does clustering an LM's own latents recover task states?
- **Yes, when the states are linearly or locally separable:**
  - automaton extraction from RNNs by partitioning the hidden space (Weiss 1711.09576, with clustering /
    abstraction);
  - automata extracted from transformers by clustering latent representations (Zhang et al. 2406.05564);
  - neuron sets identifying FSA states at ≈ 100% in CoT transformers (Zhang et al. 2502.20129);
  - Othello world-state probes (2210.13382; linear in 2309.00941).
- **For you:** run k-means on frozen §97 slot latents *first* (CPU, $0) and report purity against the true S5 states.
  - If purity is ≈ 1, SELF-SNAP should match the given codebook.
  - If it is low, the online codebook will snap to mixed clusters, and the arm will likely fail.
  - This is a cheap premise check before the run.
