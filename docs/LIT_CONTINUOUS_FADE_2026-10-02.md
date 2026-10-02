# Prior check: "train CoT normally and slowly fade the token path / output matrix into latents" (user idea, 2026-10-02)

Literature session, for the user via Audit; Theory copied. All arXiv IDs verified via the arXiv API or Semantic Scholar
on 2026-10-02. Checked before any design or registration.

**The idea.** It is Coconut's goal (thoughts end fully latent) reached by a *continuous* curriculum on the interface,
instead of stage-wise replacement. On thought positions, the fed-back input moves from the token embedding to a latent
(the raw hidden state, or a soft mixture softmax(Wh/T)·E), and/or the unembedding and the thought-token loss decay to 0.

## (a) Closest prior
- **HRPO 2505.18454 (NeurIPS'25), the closest.** On think positions the input is a gated blend:
  - e = a ⊙ ê_token + √(1 − a²) ⊙ (i ⊙ h̃), with h̃ = Eᵀ softmax(Head(h)/τ) / ‖·‖, i.e. the user's soft-mixture path,
    normalised;
  - **a is initialised → 1 (token embeddings only)**, and a learnable gate shifts it towards the latent path during
    training.
  - Trained by **RL, not supervised CoT**.
  - Reported: injecting raw hidden states directly "degrades generation and yields nonsensical rollouts with zero
    reward". HRPO trains stably, and its hybrid trajectories "remain interpretable".
  - So a continuous token→latent fade exists, as a *learned* gate under RL, using a normalised vocab projection.
- **Soft-token training and inference:**
  - Soft Tokens, Hard Truths 2509.19170: RL with soft tokens (mixture·E plus input noise). Best: train soft, infer hard.
  - Soft Thinking 2505.15778 and Mixture of Inputs 2505.14827: training-free mixture inputs.
  - CoT2 2505.23648: supervise outputs to match the empirical token distribution of several traces.
  - SofT-GRPO 2511.06411 and Latent-SFT 2510.15522: Gumbel-softmax keeps latents inside the embedding span.
  - MUX 2607.18264: each latent is a lossless position-weighted superposition of a span of CoT tokens, which "prevents
    shortcut behaviours caused by latent collapse".
- **Staged / curriculum transitions:** Coconut 2412.06769 (stages, optimiser reset at each stage per the official
  config); ICoT-SI 2405.14838 (tokens removed per epoch, with *removal smoothing*); SWITCH 2606.13106
  (visible-to-latent curriculum); LT-Tuning 2602.10229 (fuses contextual hidden states with vocab-space predictions,
  which mitigates the "feature collapse and instability due to distribution mismatch when recurrently reusing hidden
  states"; 3-stage curriculum); CoLaR 2505.16552 (a compression-factor curriculum).
- **Keeping thoughts readable while latent:** SIM-CoT 2509.20317 (per-latent auxiliary decoder); SELR 2608.13570
  (the same model decodes its own latents back to steps); LatentGuard 2608.03838 (an isolated audit decoder).
- **Unembedding / LM-head decay as the curriculum:** **not found.** Coconut simply drops the LM head and the loss on
  thought positions at each stage.

## (b) Has anyone faded the token path continuously, and with what result?
- **Yes, in one form:** HRPO's learned gate, starting at token embeddings, under RL. It is stable and readable while the
  gate stays partly on tokens. Raw hidden-state injection failed outright in their comparison. LT-Tuning's fusion is a
  fixed mix, not a fade.
- **Not found:**
  - a *supervised-CoT* fade with a fixed schedule α: 1 → 0 from token embedding to the raw (or normalised) hidden state
    on GPT-2;
  - any head-to-head of a continuous fade against Coconut's stages on stage-jump collapse;
  - readability measured at α = 0.
- **So:** "new as scoped (supervised, scheduled, with a Coconut-stage comparison and C1/C7 tests). Not new as a concept
  (HRPO)."

## (c) Failure modes to design around
1. **Decaying W itself destroys the soft path.**
   - As W → 0 (or T → ∞), softmax(Wh/T) → uniform. The fed-back input becomes the *mean embedding*, a constant
     carrying no information.
   - As T → 0 it becomes the argmax token (hard CoT).
   - A soft mixture lives inside the convex hull of E, so it can never become a free latent.
   - **Fade the interpolation weight α between the token path and a hidden-state path, not the unembedding.**
2. **Scale and anisotropy mismatch (our §72b/§73).**
   - Raw GPT-2 last-layer states are nearly parallel across positions (our cosine 0.99995; Ethayarajh 1909.00512), have
     a much larger norm than token embeddings, and carry massive-activation dimensions (Sun 2402.17762).
   - In α·e + (1 − α)·h, h dominates even at small 1 − α, so a "slow" linear schedule acts like a step: the stage jump
     returns.
   - Normalise h first: per-coordinate standardisation (§73), HRPO's normalised vocab projection, or LayerNorm matched
     to embedding statistics. Then schedule on the *normalised* contributions.
3. **Bypass once the thought loss fades** (Aswal 2606.12689; our ProsQA one-pass conduit; Prop. 8).
   - Removing the thought-token loss removes the per-step targets, our rule 1 (LIT_LATENT_FROM_DISCRETE).
   - The predicted outcome is latents that are unused on problems within one-pass capacity, or collapse on longer
     ones (SIM-CoT's collapse as latents grow).
   - **Keep a step-decoding loss as α → 0** (SIM-CoT, SELR, MUX's lossless superposition) rather than fading it.
4. **Greedy single-thread behaviour in soft mixtures** (Wu 2508.03440). The mixture is dominated by the top token, so
   it behaves like hard CoT with no latent benefit. Add Gumbel or input noise (Soft Tokens Hard Truths; SofT-GRPO) if
   the soft path is used.
5. **Exposure bias and cost.** Once the hidden path carries weight, training must feed the model's *own* states
   sequentially (Coconut-style multi-pass); teacher-forced parallel training no longer matches inference. PCCoT
   2506.18582 (Jacobi iteration) is the known way to parallelise.
6. **Range.** One thought per CoT token keeps the thought count equal to the trace length, so the range principle
   still applies past the trained length. Fading changes the interface, not the length problem.
7. **Readability at α = 0** holds only with a retained decoding loss; HRPO's readability claim is for the hybrid
   regime.

## Suggested minimal test (if Theory registers it)
- **Arms, from the same D&W CoT start:**
  - Coconut stages;
  - continuous α fade on raw h;
  - α fade on standardised h;
  - α fade on standardised h + retained step-decode loss.
- **Data:** ProsQA-NA (shortcut-free) at depths above one-pass capacity, and GSM8K-Aug.
- **Measures:**
  - accuracy;
  - removal test (C1);
  - step-decode accuracy at α = 0 (C7);
  - loss spikes at schedule points (the "stage-jump" claim);
  - the h/e norm ratio along the schedule.
