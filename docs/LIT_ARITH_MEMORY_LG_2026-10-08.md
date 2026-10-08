# Bai et al.'s running-sum loss in detail; GDN-2; log-depth recurrent units; the discrete-reasoning barriers survey; length-generalising arithmetic in recurrent / memory carriers (2026-10-08)

Literature session, at the user's request via Audit; Theory copied.
- **IDs:** verified via arXiv abstract pages (2026-10-08) or earlier notes.
- **Details:** Bai et al.'s are read from the full HTML (Sections 3–5, Eq. 1–8, App. C).

## 1. Bai et al. 2510.00184, "Why Can't Transformers Learn Multiplication?"
- **Models:**
  - **a 2-layer, 4-head GPT-style model trained from scratch.** It's the smallest architecture where ICoT works and standard
    fine-tuning fails.
  - The scaling check uses a **12-layer, 8-head** model from scratch; it doesn't use pretrained GPT-2, to avoid confounds.
- **Task:** **4×4 multiplication**, the ICoT data format, digits least-significant first. The ICoT reference model removes CoT tokens
  from the left each epoch.
- **Target:** ĉ_k ≜ s_k + r_{k−1}, with s_k = Σ_{i+j=k} a_i b_j (the column sum of partial products) and r_k = ⌊ĉ_k / 10⌋
  (Eq. 1–2).
  - So c_k = ĉ_k mod 10, and the carry follows from ĉ_k.
  - **It's the column accumulator (column sum plus carry-in), computed from the operands, not from CoT text.**
  - It is exactly our column-major accumulator (§137) and §140's flush value.
- **The loss** (Eq. 6–8):
  - a **linear regression (MSE) probe per attention head of layer 2**: z_i^h = w_hᵀ Att²_h(·);
  - applied at the timestep where output digit c_k is emitted, for k = 0..7;
  - averaged over heads and digits;
  - L = L_LM + λ L_aux.
  - (Their *diagnostic* probes read the layer-2 hidden state after attention, before the MLP.)
- **Results:**
  - standard fine-tuning **< 1% exact** (digit-level about 80%), and the **12-layer model is the same (about 1%)**;
  - **with the auxiliary loss, the 2-layer model reaches 99%**;
  - ICoT: about 100%.
  - Under standard fine-tuning, digits c₀, c₁ and c₇ are learned first and the **middle digits never get the right gradient** (a local
    optimum without the long-range dependencies). The aux-loss model learns the middle digits.
  - The aux model develops a head that attends to all {a_{i≤k}, b_{i≤k}} at once (a parallelogram pattern, like ICoT's).
- **For us:**
  - Their middle-column failure is our §137 / §140 middle-column wall (pretrained GPT-2, latent carrier).
  - Their fix is a **direct regression target on the column accumulator inside the pass**, from the operands, so not C4-strict but
    task-derived, with no CoT.
  - §147's DeltaMem broke the same wall structurally (carry-save keys) without that target.

## 2. Gated DeltaNet-2 (2605.22791)
- The delta rule ties erase (key side) and write (value side) to **one scalar gate**. GDN-2 decouples them into a **channel-wise erase
  gate b_t and a channel-wise write gate w_t**. It reduces to KDA or Gated DeltaNet as special cases.
- 1.3B on 100B FineWeb-Edu: best overall among Mamba-2, GDN, KDA and Mamba-3; strongest on RULER multi-key retrieval.
- **For us:** this is the general form of DeltaMem's split into 6 additive (write without erase) + 6 gated-delta (erase-then-write)
  heads.
  - One GDN-2 head can *learn* per channel whether to accumulate or replace, instead of our fixed split.
  - **A candidate drop-in** if the fixed split turns out to be the bottleneck. No arithmetic results.

## 3. Log-Depth Recurrent Units (2605.26035)
- MLP-LDRU: an associativity-biased operator computed by parallel reduction (log depth).
- **21 regular-language tasks:** 100% OOD (longer lengths) on 18, ≥ 99.9% on 3. That includes Delétang-style **Modular Arithmetic
  (100%)**. It beats recurrent (incl. Gated DeltaNet) and attention baselines. Competitive on ListOps.
- **For us:**
  - Regular-language length generalisation is the property our §128 families test (MOD = their Modular Arithmetic). It shows an
    *associative* operator bias gives it.
  - **LSB-first addition is regular** (a finite-state transducer), so it is in scope in principle. **Multiplication is not regular**
    (an unbounded column sum), so it is out of scope.

## 4. The "Barriers to Discrete Reasoning with Transformers" survey (2602.11175, EACL 2026)
- It unifies circuit-complexity (depth), approximation (exactness: discontinuities like ⌊·/10⌋ and mod) and communication-complexity
  (inter-token bandwidth) limits on arithmetic and algorithmic tasks.
- **For us:** it gives the citation frame for 140-3.
  - The carry is a discontinuous function (an exactness barrier).
  - A one-pass multi-digit sum is a depth barrier.
  - The single-vector hand-off is a bandwidth barrier.

## 5. Length-generalising arithmetic in recurrent / memory-augmented carriers
- **Positive, with a spatial recurrent state and local (relative) operations:**
  - **Neural GPU 1511.08228:** a convolutional recurrent grid state; binary addition and multiplication generalise to much longer
    inputs.
  - **Price et al. 1611.00736:** extends it, and shows the limits: many seeds fail, and highly structured inputs (long carry chains)
    break it.
  - Grid LSTM 1507.01526 (addition with a grid-recurrent state); Learning to Execute 1410.4615 (LSTM, limited).
- **Positive, regular-language scope only:** LDRU 2605.26035; Grazzi 2411.12537; DeltaProduct 2502.10297. Modular arithmetic and
  counting, not multi-digit carries.
- **Positive only with external memory:** tool-augmented SSMs 2510.14826 (1,000-digit addition via tool access).
- **Position coupling 2405.20671 / Abacus 2405.17399:** length generalisation from aligning digits by significance, **attention-only**.
  There's no recurrent or memory version.

## Key question: does column-position-keyed memory or position coupling give length generalisation in a recurrent / memory carrier?
- **No direct evidence found.** Position coupling and Abacus are shown only for attention.
- **The only recurrent positive evidence (Neural GPU, Grid LSTM) uses a spatially laid-out state with *relative / local* updates**,
  not absolute column keys.
- **Implication for DeltaMem:**
  - **absolute column keys learned for columns 0..7 give no key for column 8+** (the same untrained-key issue as §153's step token);
  - length generalisation needs **relative keys**.
- **Two literature-grounded routes:**
  - (a) **Rotary / relative key generation:** key_{k+1} = R · key_k with a fixed rotation (the RoPE 2104.09864 principle). Column
    keys are then defined for any k, and the read for column k+1 is the same operation as for k.
  - (b) **A shift-register memory** (the Neural GPU principle, expressible as a permutation in DeltaProduct-style transitions): shift
    the accumulator one column per output digit, so the active column is always at a fixed slot. That's position coupling realised
    as state dynamics.
- **Both are untested for memory carriers. New if tried.**
  - The cheap first test is 4×n → 4×(n+k) along the multiplier with the accumulator range bounded (LIT_S140's note: bounded for 4×n
    only).
