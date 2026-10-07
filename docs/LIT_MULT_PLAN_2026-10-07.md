# Prior check for THEORY_MULT_PLAN_2026-10-07: the task-agnostic FSQ carrier on 4×4 / 5×5 multiplication, final-answer-only (2026-10-07)

Literature session, for Theory; Audit copied. Brief. IDs verified earlier (LIT_S98_MULT, LIT_S121_CARRY, LIT_INDEX). A
fresh search (2026-10-07) found no new latent multiplication results.

## 1. Latent / implicit results on 4×4 and 5×5 (GPT-2, bench items)
- **ICoT-SI 2405.14838:** released logs 1.0 / 0.996 (the paper table says 5×5 = 0.95). No CoT tokens at inference;
  trained per size, 4×4 → 11×11 (0.74, partial).
- **ICoT-KD 2311.01460:** GPT-2 small 0.97 / 0.10; medium 0.96 / 0.96.
- **2510.00184:** a running-sum regression auxiliary on a 2-layer, 4-head model from scratch takes 4×4 from < 1% to
  99%.
- **Coconut / CODI / PCCoT / CoLaR / SIM-CoT on 4×4 / 5×5: none found** (again).
- **Ours:**
  - §98 GEN-M k = 4 fails (0.16 / 0.005);
  - §100 k = 2 is 1.000 / 1.000 / 0.979 on 4×4 (not C1) and fails 5×5;
  - §107 pause doesn't help;
  - §121 / §122: the carrier holds the b-prefix, not S_j;
  - §106 written bounded-carry streaming: 1.000 to b = 32.

## 2. Novelty
- **New as scoped:** final-answer-only acquisition of multiplication in a discrete latent carrier.
  - ICoT-SI ends with no CoT but *starts* from full CoT and removes it, i.e. step-supervised.
  - 2510.00184 supervises the running sum.
- **Carried-running-sum precedents:**
  - 2510.00184 (supervised);
  - **Neural GPU 1511.08228** (binary multiplication with a bounded local state, length-generalising);
  - our §106 (a written bounded carry).

## 3. Pitfalls
1. **The b-prefix shortcut is *stronger* under FINAL-only** (§121's diagnosis). With a visible in every segment and no
   per-step targets, the cheapest code holds the multiplier digits seen so far, and the final decode recomputes
   a × (b-prefix) in one pass.
   - 4×2 fits in one pass (§121), so at multiplier length ≤ 2 the shortcut is exact and the curriculum's early stages
     reward it.
   - Probe S_j vs b-prefix on the code (§121's probes), and treat "4×2 exact, 4×3 fails" as the shortcut signature.
2. **Op ordering: the strongest design point.**
   - **Multiplier-major** (one op per b_j, state = S_j): the state grows with j (26.6 bits at 4×4, 33.2 at 5×5) and
     leaves its range at longer multipliers (the range principle). Per-op compute is 4×1 product + multi-digit add in
     one pass (§107's compute question).
   - **Output-column-major** (op k = output digit k: add every a_i·b_j with i + j = k, plus the carry, then emit digit
     k): the carried state is **only the carry**, about log₁₀(n·81) + 1 digits (≤ 3), *bounded for any length*.
     Per-op compute is ≤ n single-digit products + an add (smaller than a 4×1 product).
     - This is schoolbook convolution + carry: Cho et al. 2410.15787's position coupling aligns digits of equal
       significance; Neural GPU.
     - **It fits FSQ-8 easily, and is the only ordering whose state stays in range for 4×8 length generalisation.**
   - Digit-pair ordering (one op per a_i·b_j) is in between: tiny per-op compute, n² ops, state = the running sum
     (grows).
3. **Emitted digits and FINAL-only.** Column-major emits answer digit k at step k. Supervising each emitted digit uses
   only final-answer content (no intermediate state labels), but it is *temporally split* supervision. State that, as
   "answer digits supervised when they become final", which is still C4-compatible.
   - Pure FINAL (decode the whole product from the last code) would force the code to hold all digits, i.e. capacity
     grows again.
4. **Capacity numbers:** 26.6 / 33.2 bits for multiplier-major exceed FSQ-8 at 8 levels (24 bits). They need the §133
   / §136 width results. Column-major needs only ≈ 10 bits.
5. **Length generalisation (4×8):** multiplier-major S_j outgrows the trained range. Column-major with a bounded carry
   is the design that can extrapolate (§106 written analogue: 1.000 to b = 32).
6. **Bar:** ICoT-SI (1.0 / 0.996 at zero CoT tokens) is the accuracy bar. Report FLOPs (ICoT-SI is cheaper) and C1
   (donor) as the distinguishing claim.

**Recommendation:** make **output-column-major with a bounded carry** the primary format (task-agnostic carrier, a
format choice as in §106). Keep multiplier-major as the capacity stress arm after the width gates.
