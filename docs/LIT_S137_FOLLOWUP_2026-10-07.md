# Prior check for the §137 2×2 follow-up: latent repeats per column (r 1 / 4) × carrier (FSQ-8 / SOFT-16) (2026-10-07)

Literature session, for Theory; Audit copied. Brief. IDs as in LIT_INDEX.

## 1. Precedents for repeated latent steps on the same input
- **Recurrent depth with input injection:** Huginn 2502.05171 (the same injected input every recurrence, random r);
  Universal Transformers 1807.03819 / ACT 1603.08983 / PonderNet 2107.05407; Ouro 2510.25741.
- **Looped transformers for algorithmic tasks:** Fan 2409.15647; Saunshi 2502.17416 (looping ≈ implicit CoT).
- **Thoughts without new input:** Coconut 2412.06769.
- **Pause / filler positions:** 2310.02226, 2505.21024 (our §107 PAUSE-4 didn't help, but pause positions aren't
  carried).
- **The closest is Huginn** (iterate on the same input through a carried state). **New as scoped:** repeats carried
  through a *discrete* (FSQ) code in a pretrained LM's latent chain.

## 2. Pitfalls
1. **What the repeats should compute.** A column sum is Σ a_i·b_{k−i} (≤ 4 products) + carry.
   - The natural use of 4 repeats is **one product per repeat**. But with identical inputs, a repeat can't know *which*
     pair to add unless the code carries a progress counter.
   - A **repeat-index token** gives the step its pair directly. **r4 + index ≈ §140 DIGPAIR with the whole column
     visible.** That makes the cross with §140 informative: does seeing only the pair (§140) or the whole column help?
2. **A repeat index is a clock, but a harmless one here.** Our Prop. 9 / §35 concern is a clock that leaves its trained
   range. With r fixed at 4 in training and test it never does.
   - Only if you want extrapolation in r: randomise r in training instead (Huginn; Kuo 2606.29983).
   - **Without an index, expect a fixed point after one repeat** (the same map on the same input), so repeats 2–4 add
     nothing. Log the code change per repeat.
3. **Quantisation per repeat:** each repeat must move the code by at least one FSQ level to register progress. Small
   increments are erased by rounding. SOFT-16 (β 0.1) doesn't have that problem, which is another reason the cross is
   informative.

## 3. Is a wider code a plausible fix?
- **Capacity isn't binding:** the column state (sum < 400 plus carry) is about 9 bits, and FSQ-8 at 8 levels holds 24.
- **A low linear R² for the carry in the code (0.24 vs 0.69 in h) isn't proof of absence.** The carry is a nonlinear
  (⌊·/10⌋) function, and the code is discrete. **Fit a nonlinear probe** (a per-value classifier or MLP) on the code
  before concluding "compression loss at W_down".
- The arithmetic literature points to **in-pass computation / learning dynamics** as the binding constraint, not
  storage: 2510.00184 (long-range dependencies not learned; a running-sum auxiliary fixes it); our §107.
- **Prediction:** r4 (serial compute) helps more than SOFT-16. SOFT-16 helps only if the nonlinear probe shows the
  carry missing from the FSQ-8 code.

## 4. Novelty
New as scoped (2 × 2: discrete-carried repeats × width). The pieces are known (recurrent depth, pause, looped
transformers).
