# Length generalisation of multiplication without a scratchpad: what's published (framing for §161 MIX and the SHAPE-MIX-2D lead) (2026-10-08)

Literature session, for Audit and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Settings:** read from the full HTML of 2405.17399, 2405.20671, 2409.15647 and 2306.15400 (2026-10-08).

**§161 MIX s1:** DeltaMem-R; trained 4×n for n ≤ 5; no scratchpad, CoT or index tokens. **4×6 / 4×7 / 4×8 = 0.994 / 0.976 / 0.964.**

## 1. Published length generalisation for multiplication, by whether intermediate text is used
| Paper | Scratchpad / CoT? | Shape | Train → test | Result |
|---|---|---|---|---|
| **Position coupling, Cho et al. 2405.20671** | **no** (direct product) | **N × 2** (second operand fixed at 2 digits) | N 1–40 → **90** | length generalisation to 90 digits (about 2.25×). Prior N×M work had near-perfect in-distribution but poor length generalisation |
| **Fan et al. 2409.15647** (looped, adaptive steps) | **no** (no intermediate supervision) | **binary**, first number ≤ 2 digits | test length 16 | NTP-Loop improves length generalisation on multiplication; the loop converges after solving |
| **Jelassi et al. 2306.15400** (train-set priming) | **no** (direct) | 5 × 3 | 5 → 35 digits | **only with priming:** 50 of 5,000 examples from the target length. Without it, extrapolation is "limited to short lengths (7 digits)" |
| **Abacus, McLeish et al. 2405.17399** (looped, 4 layers × 4 recurrences) | **no** (direct) | both operands ≤ 15 | in-distribution only | **near-perfect in-distribution to 15 × 15** (better than Shen 2023 on the hardest training cases). **The text makes no OOD claim for multiplication**; OOD cells are only in the Fig. 6 heat map. Abacus's 20 → 100-digit result is *addition* |
| Cho et al. 2410.15787 | **yes** (task scratchpad + tri-level coupling) | both operands vary | 1–10 → about 2–3× | first length generalisation on *both* operands |
| Turing Programs 2407.03310 | **yes** (tape-copy CoT) | n × 1, n × 3 | 50 → 100 | 97% |
| ICoT-SI 2405.14838 and the implicit / latent CoT line | no CoT at inference | n × n, per size | — | **no length-generalisation result** (a separate model per size; LIT_S98) |

## 2. Answers
1. **Without a scratchpad, published length generalisation is over one operand only, with the other fixed and small:** position
   coupling N × 2 (40 → 90); Fan's looped binary × ≤ 2-digit; Jelassi's 5 × 3 → 35 × 3 with priming (the target length seen in 1% of
   the data).
   - **None over both operands without a scratchpad.** Abacus shows in-distribution only; Cho needs a scratchpad.
   - **None in a pretrained LM, and none through a latent recurrent memory.**
2. **Abacus multiplication:** looped, direct output, no scratchpad. Trained and evaluated in-distribution up to 15-digit operands
   (near-perfect). No OOD multiplication numbers are claimed in the text.
3. **Framing for §161, if it replicates (≥ 3 seeds; LIT_C2_MULT_SCHEDULE's fragility warning):**
   - "Direct-answer length generalisation over the multiplier with a **4-digit fixed operand** (n ≤ 5 → 8, 1.6×, 0.964), in a
     **pretrained GPT-2 whose only cross-pass channel is a latent recurrent memory**. No scratchpad, CoT, priming or position tokens."
   - Published direct-answer results fix the other operand at ≤ 2 digits (position coupling N × 2; Fan, binary) or need priming
     (Jelassi, 5 × 3).
   - Fair caveats: the extrapolation ratio is smaller than position coupling's 2.25×. Training mixed n ∈ {2..5}, a shape curriculum
     comparable to the others' variable-length training, not a single length.

## 3. For SHAPE-MIX-2D (m × n in 2..5 → up to 8 × 8)
- **It would be new:** two-operand length generalisation without a scratchpad has no published precedent.
- **The flag: value-range extrapolation, not just length.**
  - Column sums grow with min(m, n): at most 81 × 5 + carry ≈ 440 in training (5 × 5), versus 81 × 8 + carry ≈ 700 at 8 × 8.
  - So 2D growth asks the memory and readout for **accumulator values never seen in training**. That isn't true for 4 × n, where
    the column sums stay ≤ 81 × 4 (LIT_S140's point).
  - Report accuracy by maximum column sum. Expect the first failures where the column sum exceeds the trained maximum, not at a
    particular length.
- Also report m × n with n ≤ 5 < m separately from both > 5. The first isolates multiplicand growth from value-range growth.
