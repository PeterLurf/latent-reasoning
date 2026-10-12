# Prior check for note 187-6: RMS-normalising the additive read makes early steps size-free (2026-10-12)

Literature session, for Theory; Audit copied. IDs verified via arXiv abstract pages or earlier notes (LIT_INDEX).

**The finding:**
- With an unnormalised additive read (S += Σ φ(k)vᵀ over all tokens and all passes), per-step accuracy on larger-than-trained problems decays earlier as
  size grows (size-keyed).
- **With a per-head RMSNorm on the read** (TransNormer's NormAttention), steps t 3–6 are about 0.95–0.99 at n 9–12, and failure starts at a fixed pass
  (t ≈ 7) whatever the size.
- A Katharopoulos denominator (DEN) didn't help (0.188 vs 0.174).

## 1. Do unnormalised reads leak length or count through magnitude? Does normalising help length generalisation?
- **Magnitude carrying position or count is documented:**
  - **Csordás et al. 2408.10920:** small gated RNNs encode token *position by order of magnitude* ("onion" representations). Recurrent states demonstrably use
    magnitude as a position / count code.
  - **Haviv et al. 2203.16634** (EMNLP Findings 2022): causal transformers *without positional encodings* still learn absolute position. The conjectured
    mechanism: causal attention lets a token infer how many predecessors it attends to, so **the aggregation statistics themselves leak count**.
  - **State-magnitude drift past the training length:** Stuffed Mamba 2410.07145 (state mean / variance explode beyond the training length); SANE 2608.22354
    (delta-rule state norm explosion); 2507.02782 (unexplored state statistics).
- **Normalisation or decay helping length or stability:**
  - TransNormer 2210.10340 (NormAttention: denominator removed, output norm; better LM / LRA, stabler gradients);
  - RetNet 2307.08621 (decay + GroupNorm on head outputs);
  - GLA 2312.06635 (gates + output norm; 2K → 20K perplexity);
  - SANE (a tanh soft cap; reasoning survives 24,000× the training length).
  - **For looped models:** Fixed-Point Reasoners 2606.18206 (pre-norm + residual scaling fix depth signal propagation); Huginn's sandwich norm; Chen
    2603.21676 (LayerScale + an identity-biased gate).
- **None of these frames the gain as removing a size or count cue.** They frame it as stability, numerical overflow, or out-of-range states.

## 2. Normalisation removing size-keyed computation specifically?
- **Not found. Note 187-6's reading is new.**
- **The literature supports the mechanism, though:** magnitude is a known position / count channel (Csordás onion codes; Haviv's predecessor-count
  inference). A constant read norm removes it. Combined with Zhou 2310.16028 (length-specific programs don't extrapolate) and Kuo 2606.29983 (size ↔
  loop-count correlation), "normalising removes a size cue, so a size-free procedure is learned for early steps" is a coherent, novel claim.

## Why DEN (Katharopoulos) differs from RMSNorm: a candidate explanation
- **The denominator makes the read a weighted *mean* over all written items.** A mean of t terms keeps a roughly constant scale, but its **dispersion shrinks
  with the number of terms (about 1/√t)**, and its content mixes in proportion to the counts. **That is the same count-leak channel Haviv describes for
  causal-attention averaging.** So size information remains available through how "averaged" the read is.
- **A per-head RMSNorm fixes the read's norm exactly per pass**, removing the magnitude channel. Count can still leak through the *direction*, but not through
  scale.
- **Prediction (testable on existing runs):** the DEN read's per-head dispersion or entropy should still correlate with n and t (a size cue present), while
  the RMS read's norm is flat by construction. A probe for n from the read (linear regression on the read vector) should score higher under DEN than RMS.
  That's a cheap descriptive test of the mechanism.

## For the failure at a fixed t ≈ 7
- It's consistent with Liang 2609.33144: a *progress cue* that degrades past the trained depth, and limited self-correction. With the size cue gone, what's left
  is a pass-count limit, i.e. step-position keying (the aux decodes step t at pass t).
- **The next lever:** state-relative step targets, or progressive passes (LIT_S188 recommendations). §187 AUX-NOPASS tests exactly this.
