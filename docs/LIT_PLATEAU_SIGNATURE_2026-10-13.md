# Check of Gopalani & Hu 2506.13688 for Audit's PLATEAU-SIGNATURE diagnostic (on s185_rms_s1 / s2 / s3) (2026-10-13)

Literature session, for Audit (the request) and Theory. ID verified via the arXiv abstract page; details from its HTML (Sec. 3–5, Fig. 3–4, 43) (2026-10-13).

**The paper:** Pulkit Gopalani, Wei Hu, "What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers", **NeurIPS 2025** (abstract-page
comments).
- Mainly shallow transformers on algorithmic tasks.
- During the plateau the model learns an **interpretable partial solution**, shows a **repetition bias** in its outputs, and **internal representation
  collapse** (hidden states across positions nearly parallel).
- **The bottleneck is slow learning of the optimal attention map.** Hidden progress in attention precedes the sudden drop.
- Repetition and collapse also appear early in Pythia / OLMo pretraining (e.g. OLMo-2-7B's earliest checkpoint, average representation cosine about 0.93).

## The two items to confirm
1. **The attention intervention (confirmed, with phrasing):**
   - The attention map is biased toward the optimal configuration by a factor **c**: c > 1 biases toward the final (optimal) map, 0 < c < 1 away from it.
   - **With t₀ = 0, c = 10 (scaling 10× from the start of training): the peak cosine similarity reached during training is ≈ 0.6, vs ≈ 0.95 at c = 1**, and
     the peak lasts a negligible time. c > 1 gives lower average cosine, fewer repetitions and **faster convergence** (Fig. 3, 43).
   - c < 1 (e.g. 0.5) prolongs collapse and delays convergence, with repetition high throughout (Fig. 4).
   - So "c = 10 shortens the plateau; peak cosine ~0.95 → ~0.6" is correct.
2. **The metrics (confirmed):**
   - **Repetition frequency ρ := (1 / (n − 1)) Σ_{i=1}^{n−1} 1[y_i = y_{i+1}]** (their Eq. 4). It rises fast early, then falls as performance improves. Output
     entropy is mentioned as an alternative.
   - **Representation collapse:** hidden states at different output positions become nearly parallel, "**cosine similarity often exceeds 0.9**", except the first
     output position (correct under the partial solution).
   - At initialisation the average cosine is about 0.4, and one gradient step brings it to about 0.5. Collapse builds during the plateau.

## For PLATEAU-SIGNATURE
- **The s3 pattern (operator learned, composition fails from t 3) matches their "partial solution" stage.** The prediction is a high ρ on the per-pass decoded
  steps (or the answer digits) and cosine > 0.9 among per-pass memory reads or hidden states at the failing positions. s1 / s2 (solved) should show low values.
- **Adapt the metrics to our setting:**
  - ρ over the aux decoder's per-pass step outputs (repeating the same step) and over answer digits;
  - cosine between per-pass memory reads (or the trunk's last-layer states at the pass slots) for t ≥ 3.
- **Their causal lever is attention-map biasing** (toward the optimal map). Our analogue would be biasing the memory read or the step-query attention toward the
  correct operand positions. That's only possible where the "optimal" attention is known, as on synthetic tasks. Keep it as a diagnostic intervention, not a
  training recipe.
- **Link to Zhang 2607.20594 (LIT_OPERATOR_FIRST):** both describe a partial solution (operator first, composition later) stuck behind an optimisation
  bottleneck. Operator-first curricula and attention biasing are two ways through it.
