# Prior check for the C2 lead on DeltaMem-R multiplication (4×4 → 4×8): mixed 4×n training and Abacus-style random column offsets (2026-10-08)

Literature session, for Audit (the lead) and Theory. IDs verified via arXiv abstract pages; details from the full HTML of 2405.17399,
2410.15787 and 2407.03310 (2026-10-08).

**The failure being fixed:** the exact 4×4 DeltaMem-R gets 0 on 4×8, with column 5 at exactly 0. That's the signature of a learned
4×4 pair-count schedule, and of absolute column keys never trained past column 4 (LIT_ARITH_MEMORY_LG).

## 1. Do Abacus offsets / position coupling apply to a recurrent memory carrier with short per-pass segments?
- **What they are:**
  - **Abacus 2405.17399:** each digit gets a positional embedding for its significance. **During training the start index is drawn
    from U[1, k]**, so embeddings up to k are trained on short numbers.
  - **Position coupling 2405.20671:** tokens of equal significance share a position ID; 1–30-digit addition → 200 digits (6.67×);
    also N×2 multiplication.
  - **Multi-level coupling 2410.15787:** extends it with task-specific scratchpads.
- **All are attention-side:** the IDs tell attention which positions to align. Abacus additionally helps *looped* (recurrent-depth)
  transformers, but over depth, not over segments.
- **Precedent for a recurrent / memory carrier across short segments: not found.**
- **The mapping to our carrier is direct, though.** A per-pass column-index token is the key for the memory write / read. Random
  offsets mean that column IDs up to k are trained as keys.
  - That removes the *untrained-key* failure (column 5 never seen), **provided the test columns are ≤ k**.
  - It does not by itself make the per-column operation relative. The model still has to learn that the column-c+1 write / read
    relates to the column-c one.
  - **The rotary-key or shift-register options** (LIT_ARITH_MEMORY_LG) are the length-free alternatives. Offsets are the cheaper,
    bounded one.

## 2. Does training-shape diversity alone improve multiplication length generalisation?
- **Necessary, not sufficient:**
  - Every successful method trains on mixed lengths (Cho 2410.15787 samples operand lengths from [1 : n]; Abacus trains on all lengths
    ≤ 20).
  - But **mixed lengths with standard positions fail out of distribution**. Cho: "without the scratchpad, all three PE methods
    generalize well in-distribution but struggle with OOD queries".
  - Jelassi 2306.15400: relative PE gives addition length generalisation (5 → 15 digits) but **fails for multiplication**.
- **The closest numbers for "one operand grows" (our 4×n):**
  - **Jelassi 2306.15400, train-set priming:** trained on 5×3 plus **10–50 long examples** generalises to **35×3**. The priming size
    scales with log(train size).
    - Note that the target length is *seen*, in a few examples, so it isn't held-out extrapolation.
  - **Turing Programs 2407.03310:** n×1 and **n×3 multiplication, 50 → 100 digits (2×) at 97%**, with a tape-copy CoT. Text
    scratchpad, attention.
  - **Position coupling 2405.20671:** N×2 multiplication.
  - **Cho 2410.15787:** about **2–3× on *both* operand lengths** (trained 1–10 digits), with a scratchpad plus tri-level coupling.
    The first result of its kind for both operands.
  - **Abacus:** near-perfect *in-distribution* multiplication up to 15 digits with looped transformers. Its extrapolation for
    multiplication is much weaker than for addition (100-digit addition from 20).
- **For us:**
  - Arm (a), 4×n with n ∈ {2..5}, tests column 5 in distribution only up to n = 5. **4×8 still needs columns 6–11, which are unseen.**
    Expect (a) alone to fail at 4×8, unless the model learns a relative per-column operation.
  - A **priming** arm (a few 4×8 items) is the Jelassi precedent, but **it makes 4×8 non-held-out**. If used, test C2 at 4×12 / 4×16.

## 3. Does randomising offsets hurt in-distribution accuracy at small budgets?
- **Abacus states the trade-off itself:** a larger k "enables generalization by training a larger range of embeddings *for a given
  computational budget*". Each ID is seen less often, so in-distribution learning is slower at a fixed budget.
- **Randomised positional encodings, Ruoss 2305.16843:** +12.0% average out-of-distribution test accuracy over 6,000 models and 15
  tasks. The in-distribution cost is small in their study.
- **Robustness warning, Zhou 2402.09371:** addition length generalisation (2.5×) is **fragile across seeds and data order**, unlike
  in-distribution accuracy.
- **Recommendations:**
  - (i) Pick k just above the largest test column (for example 12 for 4×8), not much larger.
  - (ii) Run ≥ 3 seeds before reading C2.
  - (iii) Report 4×4 in-distribution accuracy beside 4×8 at an equal budget, so any in-distribution cost is visible.

## Verdict
- New in the carrier setting (offset column keys for a recurrent memory). The ingredients are attention-side precedents.
- **Predictions:**
  - (a) alone fixes column 5 but not columns 6–11;
  - (b), with k ≥ 12, can reach 4×8 if the operation becomes column-relative;
  - a priming arm is the strongest known alternative, at the cost of the held-out claim.
