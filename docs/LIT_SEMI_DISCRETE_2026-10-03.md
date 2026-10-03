# Semi-discrete reasoning: representations between free latents and text tokens (2026-10-03)

Literature session, at the user's request via Audit; Theory copied. All arXiv IDs verified via the arXiv API or
Semantic Scholar (2026-10-02/03). This consolidates earlier notes (LIT_CONTINUOUS_FADE, LIT_LATENT_FROM_DISCRETE,
LIT_S92_NOVELTY) and adds 2025–26 finds (marked new). "Removal" = whether the paper tests that the thoughts are used;
"n.r." = not reported in what I read (abstract or summary).

## Spectrum (text → free latent)

| Band | Work | Representation | Training | Strongest result; removal / readability | Testable idea for us |
|---|---|---|---|---|---|
| **1. New discrete vocabulary (codes)** | Planning tokens 2310.05707 | one learned token per step (clustered step types) | SFT | gains on math word problems; readable as step type; n.r. | — |
| | Token Assorted 2502.03275 | VQ-VAE codes replace the early CoT steps | SFT on mixed sequences | shorter traces at equal accuracy; n.r. | **M1-U:** a learned VQ codebook (M ≥ 120) in place of §92's structured code (the C4 direction) |
| | CIRF 2605.28292 | one functional token per reasoning unit, optional results | SFT | good accuracy–latency trade-off; tokens "interpretable"; n.r. | — |
| | Abstract-CoT 2604.22709 | reserved vocabulary; constrained decoding | masked-bottleneck SFT ↔ self-distillation loop, then RL | up to 11.6× fewer tokens at matched accuracy; power-law code use; n.r. | constrained decoding for §97 checkpoints |
| | DLR 2606.29712 | discrete latent vocabulary from clustering rendered-CoT features | pretraining alignment, SFT, RL | up to 20× compression; "interpretable structure"; n.r. | — |
| **2. Sparse or neighbour mixtures** | N-GRPO 2606.10768 (new) | anchor token mixed with its nearest semantic neighbours | GRPO exploration | gains over GRPO on math; stays on the local manifold | **partial snap:** project a latent onto its top-k nearest codes/tokens each op, between §92's hard snap and §86's free latent |
| **3. Full soft mixtures** (convex hull of E) | Soft Thinking 2505.15778; Mixture of Inputs 2505.14827 | Σ p_i e_i (MoI: Bayesian blend with the sampled token) | training-free | small gains; Wu 2508.03440: **single-threaded** (top token dominates) | — |
| | CoT2 2505.23648 | mixtures matched to empirical token distributions over traces | supervised + RL | parallel tracking bounded by embedding dim | — |
| | Soft Tokens Hard Truths 2509.19170; MoT-G in RLVR 2509.21482 (new); SofT-GRPO 2511.06411 | soft tokens + noise or Gumbel | RL | matches hard CoT at pass@1, better at pass@32; best: train soft, infer hard | — |
| | **Mixture collapse 2609.02049** (new; analysis) | — | — | softmax readout + feedback **amplify** one component or **contract** mixtures; pretrained rollouts amplify | explains why §89 FADE-soft should fail; predict it, don't tune it |
| | Illusion of Superposition 2604.06374 (analysis) | — | — | superposition used only when trained from scratch; collapses or is unused in fine-tuned LMs | — |
| **4. Constrained / projected latents** | Latent-SFT 2510.15522 | hidden states constrained to the vocab span (Latent-Vocab), Gumbel | SFT with induction-supervision masking | beats explicit SFT, 2.7–5.5× shorter | — |
| | MUX 2607.18264 (new) | lossless position-weighted superposition of a span of CoT tokens | distillation + KL to the superposed target | beats latent baselines; probing shows faithful content; "prevents latent collapse" | **multiplexed op-state targets:** a slot's target = a superposition of the op's state text (lossless, decodable) instead of a hard snap |
| | C-MTP 2607.16972 (new) | latent = the average of the compressed CoT embeddings | direct supervision | matches slower indirect methods; all methods −65% on long traces | — |
| | Soft Latent Thinking 2608.31069 (new) | LM head replaced by a projector; rollout in embedding space | training | best pass@32 among soft methods; n.r. | — |
| **5. Hybrid gated inputs** | HRPO 2505.18454 | a·token_emb + √(1−a²)·projected h, a learned gate starting at tokens | RL | stable; hybrid traces readable; raw h injection fails | the §89 residual-token-path arm, with an h-removal C1 test |
| | LT-Tuning 2602.10229 | fuses the contextual hidden state with a predicted vocab embedding | 3-stage curriculum | mitigates feature collapse | — |
| **6. Interleaved / routed text + latent** | SWITCH 2606.13106 | ⟨swi⟩…⟨/swi⟩ latent blocks between text | visible → latent curriculum + Switch-GRPO | boundary anchors enable probing and causal interventions | — |
| | ThinkRouter 2602.11683 (new) | routes to **discrete tokens when confidence is low**, latent otherwise | inference-time | beats CoT, random routing, latent baselines | **confidence-routed snap:** in §97, snap to text when slot confidence drops, else stay latent (adaptive k; §83's C5 in latent form) |
| | TARPO 2606.05859 | a per-token router between latent and discrete | RL | beats explicit and latent RL baselines | — |
| | Thinking States 2602.08332; PLaT 2601.21358; SpiralThinker 2511.08983 | text thoughts re-embedded as state / latent planner + text decoder / interleaved latent iterations | supervised (+ curricula) | Thinking States length-generalises on parity/vars; PLaT gives higher pass@k | — |
| **7. Discrete re-embedding in recurrence** | 2510.14095 | argmax-decoded factored symbols, re-embedded each iteration | per-iteration supervision + error-correction noise | 4× OOD on mod-23 DAGs | **error-correction noise on snaps** (corrupt the decoded state with small probability during training) for §92/§97 robustness |
| **8. Concept-level units** | Large Concept Models 2412.08821 | sentence embeddings (SONAR) as units | regression / diffusion | feasibility at scale | — |
| | SuperThoughts 2606.13862 (new) | 2 CoT tokens compressed per latent, MTP decode, confidence fallback | SFT | 20–30% shorter at −1–2 points | — |

**Gradient through discrete snaps (a design point; no reasoning paper found that does it):** straight-through or
Gumbel-softmax estimators across the snap (Gumbel-softmax 1611.01144). §92/§97 cut gradients at chunk boundaries, so
chunk-end-only labels (§93) can't train earlier chunks' hand-offs through the snap. **ST-snap** is the natural test.

## Which designs fit our recipe (latent steps + discrete checkpoints + side decoder), and what would be new
1. **Best fit: band 7 + 6 + a side decoder.** That is 2510.14095's discrete re-embedding, plus SELR-style
   self-decoding, plus ThinkRouter-style confidence routing. **New: a confidence-routed, certified snap** (adaptive k)
   in a pretrained LM on S5.
2. **Training fixes to borrow:** error-correction noise on snaps (2510.14095), straight-through gradients through
   snaps (for §93's sparse labels), and MUX-style lossless superposed targets as an alternative to hard per-op labels.
3. **Avoid as carriers:** full soft mixtures (bands 3–4 without a snap). 2609.02049 and 2604.06374 predict collapse or
   non-use in fine-tuned pretrained LMs, matching §86 and §89's α → 1 collapse.
4. **The gap across all eight bands:** none reports a certificate, few report removal tests, and length extrapolation
   is rare (Thinking States, 2510.14095). Our certified S5 plus removal protocol is the differentiator.
