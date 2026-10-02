# Prior check for §73: per-coordinate standardisation of frozen GPT-2 features before the bilinear head (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch); the specific
claims were read from the papers.

## 1. Anisotropy of LM representations
- **Ethayarajh 1909.00512:** GPT-2's contextual representations are highly anisotropic. Random words' average cosine
  approaches 1 in the upper layers. **Your 0.993 across codes is that phenomenon.**
- **Timkey & van Schijndel 2109.04404 (EMNLP'21):** in BERT and GPT-2, 1–3 "rogue dimensions" dominate cosine and
  Euclidean geometry while mattering little to the model's behaviour. The proposed fix is **standardisation** (per
  dimension, z-scoring). The closest prior to your fix.
- **Outlier and massive dimensions:** Kovaleva 2105.06990 (BERT Busters); Puccetti 2205.11380 (driven by token
  frequency); Dettmers 2208.07339 (LLM.int8 outlier features); Sun 2402.17762 (massive activations, first token in
  GPT-2; see LIT_FROZEN_POS0_LR).
- **Post-processing fixes:** All-but-the-Top (Mu 1702.01417: remove the mean and top PCs); BERT-flow (Li 2011.05864);
  whitening (Su 2103.15316; WhiteningBERT 2104.01767). Gao 1907.12009 (representation degeneration).
- **Evidence that anisotropy hurts heads trained on frozen features:** mostly from similarity and STS tasks, where
  whitening or flow gives large gains. For linear classifiers the effect is optimisation conditioning (next section).

## 2. Is "standardise frozen features before the head" a known trick?
- **Yes, standard in linear probing:** MAE (He et al. 2111.06377) puts "an extra BatchNorm layer without affine
  transformation (affine=False)" on frozen encoder features before the linear probe. They note it "does not break the
  linear property, and it can be absorbed into the linear classifier after training". The justification is the usual
  practice of normalising inputs to classical classifiers (SVMs). Your no-affine per-coordinate standardisation is the
  same operation computed over the table.
- **Interpretation for a multiplicative head:**
  - Standardisation is a per-coordinate affine map of each input. A bilinear head with linear and bias terms can absorb
    it exactly, so it **changes conditioning, not expressivity or information.** Without those terms, the centring
    part is not absorbable, so check which form §73's head has.
  - So "pretrained fails, standardised certifies" says the information was present but badly conditioned: rogue or
    sink dimensions dominate. It does not say pretraining lacks it.
  - Random-init GPT-2 has no rogue dimensions (cosine 0.13), so it needs no fix. Consistent.
- **Ablations that would localise the cause:**
  1. Remove only the top-k outlier coordinates or top PCs (all-but-the-top) without rescaling.
  2. Centre only vs scale only.
  3. Per-vector LN vs per-coordinate standardisation. LN alone doesn't remove a shared offset direction, which fits
     §72's failure.

## 3. Leakage across rows
- **No label leakage:** the statistics are computed from inputs only (features of each (code, op) call), never from
  targets.
- **No test leakage in the usual sense:** the domain is the finite table, enumerated identically at train and test.
  This is transductive normalisation over a known closed input set. Precedent: BN statistics, and MAE computes BN
  statistics on the training set.
- **The real concern is that the certified object must be a fixed per-call function.**
  - If the statistics are recomputed each update from the current (trainable) codes, the head is BN in training mode.
    Each call's output then depends on every other row.
  - Certify **with the statistics frozen as constants** (absorbed into the head, like BN eval mode) after training,
    and re-check the certificate under the frozen statistics.
  - Pre-register that the statistics are frozen before the certificate, OOD rollouts, and any C7 edit or intervention.
    An edited or off-table code must not re-normalise the table.
- **Numerical caveat:** coordinates with near-zero variance across rows get amplified by 1/σ. Report the σ
  distribution and ε, and check that certification is not riding on amplified noise. Sweep ε, or drop coordinates with
  σ < τ.
- **Fairness:** apply the same standardisation to the random-init arm and the trainable arm, so the comparison isolates
  pretraining rather than preprocessing.

**Bottom line.**
- A known, defensible fix (Timkey 2109.04404; MAE's affine-free BN probe), with no novelty claim needed.
- No leakage, provided the statistics are label-free and frozen before certification.
- Describe it as conditioning frozen-LM features, consistent with GPT-2's rogue and sink dimensions.
