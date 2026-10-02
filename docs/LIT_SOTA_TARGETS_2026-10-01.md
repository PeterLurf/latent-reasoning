# Comparator numbers for "beat SOTA on smaller tasks" (2026-10-01)

Literature session, for Theory; Audit copied. Numbers are from the papers' full text, read via their HTML. Protocol
differences matter more than the numbers: the "SOTA to beat" depends on which protocol you adopt.

## 1. Group word problems (S3 / S4 / A5 / S5 prefix products)

**Two incompatible protocols in the literature:**
- **Depth-vs-length (Merrill et al. 2404.08819, *Illusion of State*):** token tagging (label = prefix product at every
  position). Models are trained *at* each length n and report the **minimum depth reaching 90% full-sequence test
  accuracy**. This is *not* length generalisation. Result: RNNs and IDS4 solve A5, A4×Z5 and Z60 with one layer at any
  length; Transformers, S4 and Mamba need depth growing with n.
- **Length extrapolation (Grazzi et al. 2411.12537 → DeltaProduct 2502.10297):** **train length 128, test up to 512,
  accuracy over all positions.**
  - **DeltaProduct:** with one layer, n_h = 2 Householder factors extrapolates robustly on S3, S4 and A5; **S5 needs
    n_h = 4**. With n_h = 1, S3 needs 3 layers, S4 6 and A5 3, and **S5 is unsolved even at 10 layers**.
  - Grazzi: DeltaNet with eigenvalues in [−1, 1] fully learns S5 only when restricted to swap permutations (1 layer).
    Parity 1.000 vs 0.017; modular arithmetic 0.971 vs 0.314 (train 3–40, test 40–256).
- **Looped transformers (Zhang 2607.20594):** curriculum training n ∈ {4, 8, 16, 32}, test n ∈ {64, 128}, exact match.
  - **S5:** a standard curriculum deadlocks; operator-first gives **10/10 seeds, exact match ≥ 0.92 at 2× training
    length**.
  - **A5:** seed lottery from scratch (8/22); operator-first gives 100% of attempted seeds.
  - S4 near-perfect at n = 64 with T = n.
- **PaTH attention 2505.16381:** A5 solved above 90% with fewer layers (~1–2) than RoPE and other baselines;
  flip-flop error 0% (in-distribution) / 0.0001% (OOD) vs RoPE 6.9% / 40.3%.
- Implicit LMs 2502.07827: "superior state tracking on regular languages". Their S5/A5 numbers were not extracted here.
- **SOTA to beat on S5/A5 under length extrapolation:** DeltaProduct (n_h = 4, 1 layer, 128 → 512, all positions) and
  Zhang's looped operator-first (≥ 0.92 exact match at 2×).
- **Certificates of arbitrary-length correctness on S5/A5:** none found. Automaton extraction exists for small regular
  languages (Weiss 1711.09576; SR-RNN 1901.08817). DeltaProduct and Grazzi give mechanistic or theoretical arguments,
  not extracted certificates. **A 120-state (S5) or 60-state (A5) transition audit would be new.**

## 2. Multi-digit multiplication, train small, test larger
| Method | Train | Test | Accuracy | Model |
|---|---|---|---|---|
| **Self-Improving Transformers** 2502.01612 | ≤ 5×5 | up to 9×9 after 31 rounds (10×10 after 19 accelerated / 41 standard rounds) | "near-perfect" | 14M (6L, 6H, d384), NoPE; majority vote and length filter essential |
| **Position coupling + scratchpad**, Cho et al. 2410.15787 | 10×10 | 20×15 | **median exact match ≥ 78.55%** (~2× each operand) | 6L, 8H, d1024 (~63M) |
| Abacus 2405.17399 | ≤ 15-digit operands | — | near-perfect in-distribution; little OOD reported for multiplication | looped, 4 layers × 4 recurrences |
| Jelassi 2306.15400 | relative PE: addition 5 → 15 digits; **multiplication fails** without priming | | | |

**SOTA to beat** depends on the format:
- with scratchpad: Cho et al. (10×10 → 20×15 at ≥ 78.55%);
- self-labelling: Lee et al. (5×5 → 9×9/10×10, near-perfect).

No latent-reasoning method reports multiplication *length* generalisation. iCoT and ICoT-SI train and test per size.

## 3. Chomsky-hierarchy suite (Delétang et al. 2207.02098)
- **Protocol** (as used by MLP-LDRU and Ruoss 2305.16843): train lengths 1–40, test 41–500, accuracy averaged over test
  lengths. 15 tasks, 20,910 models.
- **Baseline pattern:** LSTMs solve regular and counter tasks; Transformers and RNNs fail non-regular ones; only
  stack/tape-augmented models generalise on CF/CS tasks.
- **Current SOTA on regular tasks: MLP-LDRU 2605.26035**, with **100% OOD on 18/21 regular tasks** and near-perfect on
  the other 3. It beats RNN, LSTM, Gated DeltaNet, Transformers (ALiBi, randomised RoPE, NoPE), BBT-GRC and RegularGPT.
  Little headroom is left on regular tasks.

**Recommendation:** the most room for a *new* result is S5/A5 length extrapolation with a certificate (no audit
exists), or multiplication length generalisation *without scratchpad or self-labels*. The regular-language suite is
largely saturated by MLP-LDRU.
