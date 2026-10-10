# Prior check for DEC-DROP: masking a fraction q of the decoder's problem-prompt tokens during training only, to force memory use (conditional on §179 SHORTCUT) (2026-10-10)

Literature session, for Audit (the lead) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Details:** Bowman from the PDF (2026-10-10).

## 1. Bowman et al. 1511.06349: word dropout (verified)
- **The mechanism:** a variant of word dropout (Iyyer et al.). During training, decoder *input* tokens are replaced by UNK with probability 1 − k
  (keep rate k ∈ [0, 1]).
  - **At test, all words are supplied** (except in the k = 0 "inputless decoder" setting). That's the same train-masked / test-full asymmetry
    DEC-DROP proposes.
- **The effect:**
  - "Training a VAE … **without both word dropout and cost annealing** reliably results in models equivalent to the baseline RNNLM, with **zero KL
    divergence**" (the latent unused).
  - Fig. 3 (Books Corpus): **lowering the keep rate raises the KL term** (more information through the latent) at a cost in reconstruction cross-entropy.
  - The default for later models: "a fixed word dropout rate of 75%".
  - KL-cost annealing (the KL weight ramped from 0 on a sigmoid schedule) is the companion fix.
- **Mapping:** q (DEC-DROP) = 1 − k. Bowman's working point, about 75% dropout, is **aggressive**. Small q (≤ 0.25) may not be enough to beat the
  shortcut. Sweep q.

## 2. He et al. 1901.05534: aggressive inference-network training (the alternative)
- **Collapse comes from *inference lag*:** early in training the inference network fails to track the moving true posterior, so the model learns to
  ignore the latent.
- **The fix:** "depending on the model's current mutual information between latent and observation, aggressively optimise the inference network before
  each model update" (many encoder steps per decoder step early on, then normal training). No new components.
- **Our analogue:** extra updates (or a higher lr) on the *memory-write* path (W_k / W_v / gates) early, with the end decoder slowed, until the memory
  carries information (e.g. a probe R² or MI above a threshold). It's a training-dynamics fix, with no change to inputs.
- **Other decoder-weakening fixes from that literature:** a dilated-CNN decoder with a limited receptive field (Yang et al. 1702.08139), and free bits /
  IAF (Kingma et al. 1606.04934). Restricting the decoder's direct view is the common principle.

## 3. Closer priors in latent-reasoning / think-token models
- **Markovian Transformers 2404.18988:** the answer is predicted from the CoT with **the question removed**, so the CoT is forced to carry the
  information. That's full masking, at train *and* test, so it's the "blind decoder" end of the spectrum, on text CoT.
- **SIM-CoT 2509.20317:** the aux decoder sees only the latent plus a step prompt, never the question (LIT_STEP_AUX).
- **Our own:** §98 GEN-M-MASK (the fork sees only the slot), §149 BYPASS.
- **Not found:** *partial, stochastic* masking of a latent-reasoning decoder's view of the question during training, with the full view at test.
  **DEC-DROP is new in that form.** Bowman's word dropout is the methodological precedent.

## Flags
1. **Train / test mismatch is the point, but measure it.**
   - At test the prompt is visible again, so the decoder may lean back on it.
   - Score C1 with the prompt visible (the deployed condition) *and* blind.
   - Report the gap: a memory used only when blind is weaker evidence.
   - Bowman reports test with full inputs and still a non-zero KL, so the latent stays informative.
2. **Sweep q** (e.g. 0.25 / 0.5 / 0.75) given Bowman's 75% working point.
3. **Masking whole operands vs random tokens:**
   - random-token masking can be undone by redundancy (other digits, structure);
   - masking whole numbers (or all digits of one operand) more reliably forces the memory.
   - Pre-register which.
4. **Pair with a ramp** (Bowman's KL annealing analogue): start at q high and lower it, or the reverse. Choose so the memory path forms before the decoder
   learns the shortcut (He's lag argument suggests forcing early).
