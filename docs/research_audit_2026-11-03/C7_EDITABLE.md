# C7 editable latents: survey and proposals (Audit research subagent, 2026-11-03)

Status: per §247 (Note 247-2) the per-pass value is faithfully *readable* (WS ~1.0 at pass t's ':') and memory swaps are causal (SWAP4; §167 S5 DAS), but a step sentence appended to pass t's text is ignored by the write (edit follow = floor; the round trip is vacuous). C7 = NORTH_STAR round trip + edit through the model's own re-encoding. Backlog section 8 already lists R1-R10 (Patchscope readout, DAS on bun, codebook / state-encoder edit, cycle consistency, text-checkpoint restart with reset, token-write, value dropout, FSQ bottleneck, faithfulness trajectory). The proposals below avoid those; overlaps are named.

## 1. Survey (2024-2026; every id fetched from arxiv.org/abs)

**Steering continuous thoughts mostly fails.**
- 2512.21711 (Zhang et al., Dec 2025 / v2 Oct 2026): perturbation and swap steering of Coconut latents; minimal sensitivity, latents act as placeholders, gains track shortcuts.
- 2609.21662 (Huang & Qi, Sep 2026): the "latent-to-language transition gap" - steering latent CoT moves hidden states as far as steering text CoT but barely changes the output (46% vs 3% steering success, Llama-3.1-8B). Same shape as §247: a channel that reads but does not write.
- 2603.16475 (Somov et al., Mar 2026): even *textual* intermediate structures are often not causal mediators; models fail to update after the structure is edited unless the last step is delegated to a deterministic reader. Lesson: a C7 pass needs passes t+1.. to depend on the edited slot, i.e. to read memory rather than re-read the text (§247 shows they recompute from text when text is available).

**Faithfulness evaluations.**
- 2607.06648 (Jin, Yang, Wang, Jul 2026): counterfactual input edits + noise ablation of latent steps along training checkpoints; accuracy and responsiveness diverge (R9's basis).
- 2602.08783 (Li et al., Feb 2026): latent steps as SCM variables; single-step do-interventions (replace one step state, keep the rest); effects are routed, non-local. The template for "edit one thought, read downstream".
- 2602.00449 (Liang & Pan, Jan 2026): CODI (2502.21074) faithful step-by-step only at 2-3 hops, shortcut at depth. 2604.04902 (Dilgren & Wiegreffe, COLM 2026): where latent tokens are necessary, gold traces decode 65-93%.
- 2606.01243 (Chang et al., May 2026): thought vectors encode steps faithfully; early vectors are causal hubs - the positive counterpart of 2512.21711. 2604.06374 (Rizvi-Martel et al., Apr 2026): superposition appears only when trained from scratch; fine-tuned models collapse it.

**Decode-then-re-encode / vocabulary-grounded latents (the architectural route to editability).**
- 2505.15778 (Soft Thinking, NeurIPS 2025): the thought is a probability-weighted mixture of token embeddings; readable by construction.
- 2509.20317 (SIM-CoT, Sep 2025): training-only auxiliary decoder aligned to each latent (our aux decoder; readability, not editability).
- 2505.22202 (Hwang et al., 2025): sentence-embedding latents with a "discretized" mode (decode to text, re-encode) beside the continuous mode - round trip as an inference mode.
- 2605.07315 (LaTER, May 2026): final-layer state mapped into input-embedding space by the pseudo-inverse of the output projection.
- 2606.29712 (DLR, Jun 2026): a discrete latent vocabulary added to the vocab and output head; interpretable trajectories - the published form of "the latent is a token".

**Activation patching as editing.**
- 2401.06102 (Patchscopes; R1). 2303.02536 (DAS; §167 / R2).
- 2604.23351 (Mehrafarin et al., EMNLP Findings 2026): patch single CoT hidden states into a direct run on GSM8K; one token's state can carry the answer.
- 2510.13908 (Yugeswardeenoo et al., Oct 2025): intermediate arithmetic results readable in the residual stream; "partial embedding swap" changes behaviour - embedding-row edits as a steering primitive.

Reading: decodable-but-unsteerable latents are the norm. Edits propagate only where the thought is tied to a token embedding, or where the downstream reader has no other source of the value.

## 2. Proposals

Bars reuse §247: ROUND-TRIP answer agreement >= 0.95 pooled / >= 0.90 per cell; EDIT follow (answer = program recomputed from x') >= 0.50 and >= floor + 0.30 pooled, >= 0.40 per cell; cells d 3 t 1/2, d 4 t 2/3, n >= 900. Models: bun_s1 / s2 (§259); lh_fh_s1 / s2 for comparability with §247.

### P1. TEXT-SUBSTITUTE (eval-only, $0, existing models) - the non-vacuous §247
- **Mechanism.** §247 *appended* "S_t has x u." after the op sentence, so pass t's write could still compute v_t from the op sentence (and dch training teaches the model to ignore given-before-op sentences). Here pass t's input *replaces* S_t's op sentence with the given sentence "S_t has x u." (the d=1 format the model is trained to read). Only pass t's write changes; passes t+1..K' and the decoder see the original text, so x' reaches the answer only through memory. This is 2602.08783's single-step do-intervention using the model's own text encoder; unlike R5 it does not reset S or re-run later cues on edited text (no depth confound).
- **Edit** = x' != x_hat, uniform. **Round trip** = x = x_hat.
- **Controls.** (i) op sentence replaced by a neutral valueless sentence: accuracy must fall to <= floor + 0.1 at that cell, else later passes recompute v_t from text and the test cannot bite; (ii) empty substitution reproduces intact bitwise; (iii) oracle x = gold.
- **Pass.** C7-PASS = ROUND-TRIP and EDIT bars with (i) satisfied; else ROUND-TRIP-ONLY / REJECTED / VOID as §247.
- **Risk.** The write under a given sentence may differ from the write under an op sentence (provenance, not value); a failure then reads "the write encodes provenance", itself a C7 finding. Reuses theory/score_rh.py; ~2 h build.

### P2. VOCAB-DIRECTION EDIT (eval-only, $0) - is the readable direction the used direction?
- **Mechanism.** The ':' readout is `colon_logits(h)[:, NUM_IDS]` through the tied LM head, so the readable direction for value x is wte[x]. Edit the write input h_t -> h_t + alpha (wte[x'] - wte[x_hat]), alpha = smallest value flipping the readout to x' (per-item line search in [0, 8], +1 margin); passes t+1.. and the decoder unchanged. Compare at matched flip: the wte direction, the ridge probe (§154 SHIFT, which failed), and §167's DAS interchange (positive reference).
- **Edit** = the flipped readout; **round trip** = alpha 0 (trivially intact).
- **Pass.** EDIT follow >= 0.50 for the wte direction with the readout flipped on >= 0.95 of items; report follow vs alpha. If DAS follows and wte does not, C7 fails for 2609.21662's reason: the readable direction is not the used one, and only a learned encoder (R3) can re-encode text.
- **Risk.** Large alpha goes off-manifold: report ||h'||/||h||, cap 1.5x, censor beyond. Prior: 2604.23351, 2510.13908, 2303.02536.

### P3. EMBED-WRITE (minimal architecture change; ~$1-1.5 on Vast, 8k fine-tune from dch, 2 seeds)
- **Mechanism.** Replace the write source. Today `m.write(S, x, mask, h_low)` writes block 11's state at ':'. New: v_t = W_w LN(wte[x_t]), x_t the value token *decoded* at ':' (argmax over NUM_IDS at inference; teacher-forced gold in training, with a straight-through estimator from the ':' logits so the trunk still learns the value). W_w a new D x D projection; keys unchanged. This is DLR / Soft Thinking's "the thought is a vocabulary item" restricted to the 11 number tokens, and LaTER's embedding-space write, applied to the DeltaMem value only. Everything memory holds about step t is then, by construction, the token the aux decoder prints.
- **Edit** = substitute x' at pass t's write (a literal text edit); **round trip** = write from the decoded token instead of gold at every pass.
- **Non-circularity (vs the declined TEXT-WRITE).** Training never sees edited pairs; the test asks only whether passes t+1.. and the decoder *use* the memory value, which the §247 failure mode (recompute from text) could still defeat. Mandatory controls: masking S_t's op sentence for passes t+1.. keeps accuracy >= 0.95 if memory is used; a neutral write (P1 (i)) drops it.
- **Pass.** Stage gate TEST2 d1-4 >= 0.95 (within 0.03 of dch) and the K'128 hold (C6 kept); gate on pass-t WS, not accuracy alone (§253 lesson). C7-PASS = ROUND-TRIP >= 0.95 and EDIT follow >= 0.80 (higher bar: the channel is designed) plus both controls; re-read NOT-DECODER and SWAP4 so C1 is not silently moved.
- **Risks.** (a) A 1-of-11 code: multi-value state needs a wider vocabulary - fine for the mod-11 chain, not GSM8K; (b) straight-through gradients may not form pass-2 writes; (c) a value-only write drops any implicit subject the state carried (keys unchanged, so the cue still routes). Args diff vs dch: write source only.

### P4. SOFT-WRITE (variant of P3, same cost; 3rd / 4th cell of P3's queue if P3 trains)
- **Mechanism.** v_t = W_w LN(sum_x p_t(x) wte[x]), p_t = softmax over NUM_IDS at ':' (temperature 1, no straight-through): Soft Thinking's mixture, fully differentiable. Readout = p_t; **round trip** = one-hot(argmax) at every pass (tests that superposition is not load-bearing, cf. 2604.06374); **edit** = one-hot(x').
- **Pass / controls** as P3. Extra reading: mean entropy of p_t by pass and d.
- **Risk.** The mixture weights can carry a second value; then ROUND-TRIP fails while EDIT passes - report both, label SUPERPOSED. NORTH_STAR's "translation" admits a value, not a distribution.

## 3. Order and spend
P1 and P2 first ($0, one afternoon on the 4060, existing checkpoints); they decide whether an architecture change is needed and which encoder is causal. P3 only if P1 fails the edit bar (else C7 is met at $0); P4 rides P3's queue. Worst case ~$3 under the $30 standing approval; none displaces C1 > C6 > C4 = C2 work beyond P3's stage gate.
