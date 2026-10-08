# Prior check for PERIODIC READ: a scalar accumulator in DeltaMem's additive heads, read through cos / sin at periods 10^k (2026-10-08)

Literature session, for Audit (the lead) and Theory. IDs verified via arXiv abstract pages; FoNE details from its full HTML
(2026-10-08).

## 1. FoNE 2502.09741 (Zhou, Fu, Soltanolkotabi, Jia, Sharan): verified, with one correction
- **Method:** each number becomes one token embedding [φ(x, T_i)] with φ = (cos 2πx/T_i, sin 2πx/T_i) and **T_i = 10^i**. Two
  dimensions per digit, so x mod 10^i is exactly recoverable for each i. These are *fixed* periods.
  - Their ablation over base periods ([2, 5, 10] / [10] / [5] / [7]) shows that periods which don't respect base 10 (5 or 7 alone)
    fail on decimal tasks.
- **"100% on add / sub / mul": correct** ("the only method that yields 100% accuracy on over 100,000 test examples for addition,
  subtraction and multiplication").
- **"64× less data for 99% on 6-digit addition": correct**, vs subword and digit-wise embeddings, with 3× and 6× fewer tokens per
  number.
- **"10 → 50-digit length generalisation": needs qualifying.**
  - That's **FoNE combined with Abacus** (Sec. 4, Fig. 4a: trained ≤ 10 digits, tested to 50; FoNE improves over Abacus across
    seeds), not FoNE alone.
  - Separately, App. F does **60-digit addition in one forward pass** with FoNE.

## 2. 2605.01148, "Arithmetic in the Wild" (Feucht, Haklay, Bhalla, Wurgaft, …): what it shows
- In Llama-3.1-8B, cyclic concepts (months, weekdays, …) have circular *representations*. But the model **doesn't compute modular
  addition in the concept's period**.
  - It **adds in base 10** (six + August → 14), then maps back to the concept (14 → February).
  - The sum uses **task-agnostic Fourier features with periods 2, 5 and 10**, not 12.
- **28 MLP neurons at layer 18** (about 0.2%), in disjoint clusters by period, are reused across all tasks.
- **For us:** a pretrained LM's own arithmetic substrate is **base-10 periodic (T = 2, 5, 10)**. So a periodic read at 10^k matches
  the geometry GPT-2-family models already use. The Llama result implies, but doesn't test, that this holds for GPT-2.

## 3. 2604.20817, "Convergent Evolution" (Fu, Zhou, Belkin, Sharan; COLM 2026)
- Transformers, **linear RNNs**, LSTMs and word embeddings all learn number features with Fourier spikes at **T = 2, 5, 10**. But only
  some make them *geometrically separable* (linearly classifiable mod T).
- **Fourier sparsity is necessary, not sufficient,** for separability.
- Separability comes from co-occurrence signals in text, or from **multi-token** (not single-token) addition training.
- **For us:**
  - Periodic structure appears in linear RNNs too, so DeltaMem's state can in principle hold it.
  - A *learned* periodic read may still not separate digits unless trained on the right signal. A **fixed** featuriser (FoNE-style)
    sidesteps that.

## 4. Precedent: a periodic read on a recurrent or linear-memory state, or a scalar accumulator + Fourier featuriser in a latent carrier
- **Not found.** The adjacent work:
  - **Scalar number encodings:** xVal 2310.02989 (a number as a scalar times one embedding: a magnitude carrier; it loses precision
    at many digits, which FoNE was built to fix); NALU 1808.00508 / NAU 2001.05016 (scalar arithmetic units; exactness and
    extrapolation are fragile).
  - **Phase-carrying recurrent states:** LRU 2303.06349 (complex unit-modulus eigenvalues: the state is literally a rotation); Grazzi
    2411.12537 (negative / complex eigenvalues for modular counting); the clock representation (2301.05217).
  - **Why a featuriser works:** Fourier features let networks learn high-frequency functions of a low-dimensional input (Tancik
    2006.10739). Here they turn mod / floor on a scalar into something linear-readable.
- **New as scoped:** a scalar accumulator in a linear memory with a fixed or learnable FoNE-style periodic read, as a latent carrier.

## Pitfalls (pre-register)
1. **Floating-point precision is the binding constraint.**
   - A single scalar holding a 4×4 product (up to about 10⁸) has an fp32 mantissa of 2²⁴ ≈ 1.68 × 10⁷. **The units digit is not
     representable above about 1.7 × 10⁷**, and bf16 (8-bit mantissa) fails beyond about 3 digits.
   - Keep the accumulator and the cos / sin in fp32 outside autocast. **Prefer per-column (carry-save) scalars** (column sums < about
     400) over one big scalar, or use fp64 for the store.
   - FoNE itself chunks numbers for 60-digit work, for the same reason.
2. **Writes must be integer-exact.** The read of the units digit tolerates absolute error < 0.5, so the additive head has to write
   a_i·b_j (× 10^{i+j}) almost exactly, from a linear projection.
   - Regression errors accumulate across writes (the NALU / NAU fragility).
   - Consider rounding the scalar write (STE round, the FSQ principle), and log the write residual |w − round(w)|.
3. **Learnable periods are unstable at large x:** ∂cos(2πx/T)/∂T = (2πx/T²)·sin(…) is large and oscillatory. **Freeze T at 10^k**
   (FoNE's choice; base 10 is what the LM uses, per 2605.01148) or give it a tiny lr.
   - Learnable periods add risk without a clear gain, because the ablations show non-base-10 periods fail on decimal tasks.
4. **One scalar per head isn't a fix for §144's single-vector failure if precision binds** (pitfall 1). The carry-save layout keeps
   each scalar small and the read exact.
5. **Diagnostics:** decode each digit from the phase on held-out items, the write residual, and the fp32-vs-fp64 agreement of the
   read at the largest products.
