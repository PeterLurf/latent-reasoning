# ID checks for the D3 length-seam leads (randomised / coupled positions) and 2024–26 offset-position work in looped / recurrent-depth models (2026-10-16)

Literature session, for Audit (the request) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Sources:** Fan et al. and Abacus checked in full text; the rest from abstracts.
- **Evidence tag:** all EMPIRICAL unless noted.

## The three leads
| ID | verified | claims checked |
|---|---|---|
| **2305.16843** | Ruoss, Delétang, Genewein, Grau-Moya, Csordás, Bennani et al., "Randomized Positional Encodings Boost Length Generalization of Transformers". **Verified.** | **+12.0% average test accuracy over 15 algorithmic tasks** (6,000 models): confirmed. Mechanism: simulate a longer position range and sample an ordered subset per sequence. **The venue "ACL 2023" is not on the abs page** (no comments field), so it's unconfirmed there. |
| **2409.15647** | Fan, Du, Ramchandran & Lee, "Looped Transformers for Length Generalization". **ICLR 2025, confirmed** (abs comments). | **NoPE confirmed** (full text): "we use NoPE … to avoid the impact from different positional embeddings", from scratch. **Input injection confirmed:** "the input embeddings are added to the output embeddings of the previous step". **Also needed for the summary:** training uses **step-dependent supervision with the ground-truth number of steps** (the authors list this as a limitation), and halting is adaptive (confidence-based or predefined). Tasks are n-RASP-L. |
| **2502.16792** | Golowich, Jelassi, Brandfonbrener, Kakade & Malach, "The Role of Sparsity for Length Generalization in Transformers". **Verified.** | **Predictive Position Coupling confirmed** (trains the model to predict the position IDs used by position coupling). **The theory:** length generalisation holds when each token depends on a small fixed number of earlier tokens (k-sparse planted correlations). That's a THEOREM in an idealised transformer model; it justifies position coupling. The venue isn't on the abs page. |

## 2024–26 offset / randomised positions in looped or recurrent-depth models
- **Abacus embeddings, McLeish et al. 2405.17399 (2024).** **The direct precedent for offset positions in a looped model** (full text).
  - Per-digit positions start at a **random offset from U[1, k]** (k = 100 by default), "inspired by Randomized Embeddings [Ruoss]".
  - The offset embeddings combine with **input injection and looped / recurrent layers** to give "near-perfect generalization" on large additions.
  - From scratch.
  - **In-house caution:** our §177 tried an Abacus-style offset and it **erased pass identity** (memory: check-borrowed-technique-preconditions).
- **Kuo, Chayti & Reizinger 2606.29983 (2026)**, "Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping".
  - Looped length generalisation is "brittle, yielding high OOD variance, even across well-performing in-distribution solutions".
  - They trace this to a **spurious correlation between sequence length and loop count**; randomising the loop count in training "sharply reduces OOD variance". It analyses RL-Halting as a learned stochastic schedule.
  - **Relevant to a length seam:** the variable that has to be decorrelated may be the **loop count**, not the positions. This is already in our notes as the "size ↔ loop-count correlation (Kuo)".
- **Random Float Sampling, Shimizu, Taniguchi & Matsuo 2602.14050 (2026):** continuous, randomly sampled position indices, pluggable into sinusoidal / RoPE / ALiBi. Not looped.
- **Ito, Puri & Campbell 2608.31067 (2026):** a provably correct tiny Universal Transformer whose **depth-tracking positional encoding** gives perfect depth / length generalisation on fully parenthesised Boolean circuits. A THEOREM in a constructed parameterisation; a hand-designed PE.
- **Position Coupling, Cho et al. 2405.20671 (2024):** same position IDs for digits of equal significance, with a randomised start ID in training. From scratch, not looped.
- **Not found:** a randomised or offset position scheme applied to a **pretrained** absolute-PE model (GPT-2 wpe) **with weight-tied loops**.
  - Our POS_OFFSET / POS_KEEP0 eval-only probes (note 187-5) are the in-house version.
  - Most of the above train from scratch with NoPE / RoPE / learned PEs, so the transfer precondition differs: GPT-2's wpe rows beyond the training range are pretrained but unseen in fine-tuning.

## Notes for the summaries sent to Theory
- Ruoss: add "venue not on the arXiv page".
- Fan et al.: add "with ground-truth step-count supervision and adaptive halting; from scratch".
- Golowich et al.: "sparse dependency theory (idealised model) → Predictive Position Coupling".
- Add Kuo 2606.29983 as the 2026 looped-specific result: OOD variance comes from the length ↔ loop-count correlation and is fixed by loop-count randomisation.
- Add Abacus 2405.17399 as the 2024 offset-in-looped precedent, flagged against §177.
