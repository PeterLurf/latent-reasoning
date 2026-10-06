# Prior-art synthesis for a stable latent method: what each line of work predicts on our evidence (2026-10-06)

Literature session, for Theory's synthesis (user directive via Audit); Audit copied. Every arXiv ID was verified via the
arXiv API or abstract pages (2026-09-29 → 10-06; see LIT_INDEX).

**Our evidence, from NORTH_STAR as of today:**
- **The 8-d FSQ carrier is exact once entered.** §128 makes it task-agnostic on 5 regular families (2 restarts in 20
  slots), but entry is a lottery on S5.
- **RAW-H (768-d) enters 4/4 but works 0/4**: drift (0.85 at 512) or later collapse (§126).
- **ANNEAL** (wide, then narrow) collapses at λ = 1.
- **Escape without learning** (§127 / §129): the loss leaves the plateau, the code even carries state (R²_state up to
  0.96), but chains fail. Plain controls do it too, and STD2 / LR10 break learners.
- **§101:** a weak attractor (β = 0.01) suffices on HYB-4.
- **§123:** dither changes nothing.
- **§130:** the early variance-dip marker fails out of sample.

## 1. Learned attractors and error-correcting recurrent states
- **Priors:**
  - modern Hopfield clean-up, 2008.02217;
  - Attractor Models, 2605.12466 (fixed-point refinement with implicit differentiation);
  - DEQ, 1909.01377;
  - 2510.14095 (discrete re-embedding plus error-correction corruption, 4× OOD);
  - SR-RNN, 1901.08817 (learned centroids);
  - QBN, 1811.12530;
  - "When Recursive Models Finish Computing", 2609.26487 (finished recursions are anisotropically stable).
- **Prediction on our data:** exactness needs a *restoring* map at the reachable states (our Th3; §101). An attractor
  doesn't need to be strong, only present at every hand-off. This matches §101 (β 0.01 enough) and FSQ's exactness.
  - 2510.14095 predicts that error-correction training (corrupt the *discrete* state, require the correct next state)
    gives the restoring map **directly**. That differs from §123's additive dither under final-answer loss only, which
    gave no restoring gradient. **§123's null doesn't refute it.**

## 2. VQ / FSQ training stability and codebook collapse
- **Priors:**
  - VQ-VAE, 1711.00937;
  - FSQ, 2309.15505 (no dead codes);
  - dead-code re-initialisation, 2005.08520; Jukebox random restarts, 2005.00341;
  - the rotation trick, 2410.06424 (better gradients than straight-through);
  - SimVQ, 2411.02038 (reparameterise the codebook through one linear layer to prevent representation collapse);
  - soft-to-hard VQ, 1704.00648.
- **Prediction:**
  - FSQ already rules out *codebook* collapse. Our entry failure is **encoder collapse** (codes near-constant across
    inputs, the zb_std dip), which these fixes don't address.
  - ST gradients through FSQ are weak early; the rotation trick and soft-to-hard annealing target that. But ANNEAL's
    failure warns that *replacing* a working carrier collapses. Anneal **in addition to**, never **instead of**.

## 3. Contractive, Lipschitz and orthogonal constraints: the isometry vs contraction tension
- **Priors:**
  - Lipschitz RNNs, 2006.12070;
  - Stable Recurrent Models, 1805.10369 (stability ⇒ forgetting);
  - uRNN 1511.06464, expRNN 1901.08428;
  - Howe, 2609.18966 (norm-preserving Householder transitions with no additive path give exact S5);
  - DeltaProduct, 2502.10297;
  - Jacobian regularisation, 1908.02729;
  - CAE, 1104.4153;
  - nGPT, 2410.01131.
- **Prediction:**
  - A group carry is an isometry on states, so global contraction merges states (JAC risk, LIT_S133).
  - RAW-H's drift (ρ_ff 0.5–0.8) is the *opposite* failure: neither restoring nor isometric.
  - The theory says the right target is **isometric along the state manifold and contractive transverse to it**,
    which is exactly what discrete attractors (FSQ, the soft snap) provide implicitly.
  - Pure orthogonal channels (Howe) solve groups but not resets (FF / MOD need contraction), so they aren't
    task-agnostic alone.

## 4. Posterior-collapse fixes for the decoder-not-reading basin
- **Priors:**
  - Bowman, 1511.06349 (word dropout, KL annealing);
  - free bits, 1606.04934;
  - δ-VAE, 1901.03416;
  - aggressive encoder, 1901.05534;
  - skip-to-decoder, 1807.04863;
  - CPC, 1807.03748; Z-Forcing, 1711.05411;
  - active units, 1509.00519.
- **Prediction, and a correction to the collapse framing:**
  - §129 shows the body *does* read a fresh head (R²_state 0.41–0.96), and §127's escapes carry state-ish codes, yet
    chains fail. So beyond entry, the residual failure is **not** classic posterior collapse.
  - The literature closest to that symptom is **shortcut solutions at short training lengths**: Liu 2210.10749; our
    §121 / §132 input-carriage trap, where the code holds op history at L ≤ 4 and the last segment recomputes.
  - **Testable now ($0):** in escape-without-learning seeds, probe the code for *op history* vs *state* at L = 2, 4, 8.
    The shortcut predicts op-history decodability that fails as L grows.
  - Skip-to-decoder (SKIP 2/4 vs 1/4) is the only collapse fix with a positive sign in our data.

## 5. Denoising and consistency training of recurrent states
- **Priors:**
  - denoising autoencoders; error-correction corruption, 2510.14095;
  - professor forcing, 1610.09038; scheduled sampling, 1506.03099; DAgger, 1011.0686 (train on own states);
  - consistency models, 2303.01469 (enforce self-consistency along a trajectory);
  - BYOL / SimSiam, 2006.07733 / 2011.10566;
  - LIFT, 2609.38149; Next-Latent Prediction, 2511.05963.
- **Prediction:** a **consistency objective across perturbations of the carried state** gives an explicit restoring
  gradient, label-free: the next code from (perturbed z, item) must equal the next code from (clean z, item),
  stop-grad on the clean target. §123 lacked exactly this target.
  - Combine with **own-rollout training** (DAgger) at lengths beyond the curriculum, so the attractor is shaped on the
    states the model actually visits.

## 6. Latent-CoT and recurrent-depth stability
- **Priors:**
  - Coconut, 2412.06769;
  - SIM-CoT, 2509.20317 (latent collapse without per-step targets);
  - Huginn / Geiping, 2502.05171 (random recurrence depth, input injection);
  - Ouro, 2510.25741 (degrades beyond the trained depth);
  - Kuo, 2606.29983 (randomised stopping stabilises extrapolation across seeds);
  - Kohli, 2604.07822 (overthinking);
  - mixture collapse, 2609.02049;
  - illusion of superposition, 2604.06374;
  - reasoning by superposition, 2505.12514.
- **Prediction:**
  - Continuous latent chains drift unless something restores them, consistent with RAW-H and §119's continuous
    methods (0/9).
  - **Randomised rollout length** (Kuo) is the published lever for seed reliability of extrapolation. It is untested
    in our harness (the curriculum is deterministic).
  - **Superposed or large states:** fine-tuned LMs don't use superposition (2604.06374). Capacity must come from
    *width of the discrete code* (§133's d ∈ {16, 64}), not from continuous superposition.

## The synthesis: one method to propose (for Theory to shape)
**"Wide carrier with a weak learned discrete attractor, trained by state-consistency":**
1. **Carry a wide continuous state** (RAW-H style; it enters reliably). Add, **in parallel and never replacing it**, a
   d = 16–64 FSQ code computed from the carried state, distilled from step 0 (QBN 1811.12530).
2. **Weak soft snap at every hand-off:** s = (1 − β)·h + β·W_up(FSQ(h)), with β ≈ 0.01–0.1. That is §101's finding
   generalised, a Hopfield-style clean-up (2008.02217) towards a learned grid.
3. **Label-free state-consistency loss:** perturb the carried s, and require the next FSQ code to match the clean
   path's (stop-grad). This is error correction (2510.14095) and consistency training (2303.01469). It shapes the
   restoring map directly, unlike §123's dither.
4. **Own-rollout and randomised-length training** (DAgger 1011.0686; Kuo 2606.29983) beyond the curriculum length.
5. **Skip-to-decoder for entry** (1807.04863; the only lever positive in §127), with a small non-zero scale
   (LayerScale 2103.17239).

**What it predicts:**
- entry like RAW-H (wide path);
- exactness like FSQ (the attractor);
- capacity from d;
- and it removes the input-carriage shortcut only if the consistency target is the *state* code. Check with the
  op-history probe.

**Kill line:** if the snap with β ≤ 0.1 plus consistency doesn't beat RAW-H's drift (ρ_ff) in ≥ 3/4 seeds, the weak
attractor isn't enough in a wide carrier, and the hard 8-d FSQ (§128) stays the method.

## Addendum (2026-10-06): Theory's candidates (COPY-anchored entry; round-trip attractor; with SKIP)

**(a) Decode → re-encode as a train-and-test attractor for a recurrent LM latent: the closest work.**
- **Hard versions:**
  - 2510.14095 (argmax-decode factored symbols, re-embed every iteration, from scratch);
  - our §97 / HYB (text snap every k with a history cut);
  - Thinking States 2602.08332 (text thoughts re-embedded as state);
  - our §101 (a soft pull to a *structured* code: the soft version with a fixed codebook).
- **Related consistency mechanisms:**
  - self-conditioning in diffusion, *Analog Bits* 2208.04202 (feed the model's own previous estimate back in);
  - cycle consistency, CycleGAN 1703.10593;
  - back-translation, 1511.06709;
  - ReLAT 2606.06252 (a question → latent → question cycle);
  - SELR 2608.13570 (self-decoding);
  - Soft Thinking 2505.15778 (a mixture over token embeddings, which isn't an attractor and collapses per 2609.02049).
- **Not found:** a *soft, every-hand-off* pull of a wide continuous carrier toward enc(dec(z)) of its own decoded
  state text. **New as scoped.**

**(b) Traps for the round-trip attractor.**
1. **The decoder must read the carried code only** (the MASK / §111-reader isolation). If dec sees the visible ops, its
   decode is a *recomputed* state, and the attractor injects recomputation into the chain, the §104 bypass smuggled
   in. Keep the fork's mask.
2. **Lock-in at init:** before dec is accurate, enc(dec(z)) is near-constant text. The pull then drives every z to one
   point, the ANNEAL / posterior-collapse failure again.
   - Gate it label-free: β = 0 until the decode-*validity* rate (well-formed state text) and the code's input variance
     (active units, 1509.00519) pass thresholds, or run COPY warm-up first. Then ramp β per hand-off, stochastically.
3. **Self-reinforcing wrong decodes:** the attractor stabilises *whatever* is decoded. A wrong decode becomes a stable
   wrong state, and only the final-answer loss can correct it (confirmation bias, 1908.02983). Fine as long as β stays
   small (§101: 0.01 suffices), so the carrier isn't overwritten.
4. **Gradient through argmax:** not needed. Treat sg(enc(dec(z))) as a target, as in BYOL / SimSiam (asymmetry =
   text path vs latent path). Cost: an extra decode-plus-encode per hand-off (≈ 2× forward). Report FLOPs.
5. **Task-agnosticism:** the attractor needs a decodable *state text*. For FINAL-only families without one, use the
   **"answer if queried now"** in the task's native output vocabulary.
   - That is the full state for S5 / CYC / MOD / RDFA (answer ≅ state).
   - **Not for FF**, where the answer is the last written bit, not the full state. State the restriction.

**(c) Copy / autoencoding auxiliaries that fix entry without step labels.**
- Sequence-autoencoder pretraining, Dai & Le 1511.01432; word dropout and annealing, Bowman 1511.06349; aggressive
  encoder, 1901.05534; Z-Forcing 1711.05411; the NTM copy task 1410.5401.
- **Interface warm-up analogue: LP-FT**, Kumar 2202.10054 (train the interface or head first, then fine-tune jointly, so
  random head gradients don't distort features).
- Our §124: copy rows give uniform early acquisition, 4/4 at L5.
- **Prediction:** COPY plus a gated round-trip attractor is the coherent pair. COPY trains enc / dec to be accurate at
  entry, exactly what the attractor needs before β > 0. SKIP adds the decoder gradient path. Order the arms as COPY →
  COPY + RT → COPY + RT + SKIP so each addition's contribution is identifiable.
