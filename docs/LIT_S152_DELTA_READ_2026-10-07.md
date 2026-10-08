# Prior check for §152 DELTA-READ: DeltaMem as the sole carrier, a per-step S_j reader, ICoT-style removal, vs TC-M / ICoT-SI on compute (2026-10-07)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-07) or earlier notes (LIT_INDEX).

## (1) Prior work
- **Linear-attention / linear-RNN memory as the carrier for multi-digit arithmetic: not found.** That gap is restated from
  LIT_ARCH_SWEEP / LIT_BIG_BETS.
  - The state-tracking results (Grazzi 2411.12537, DeltaProduct 2502.10297) cover parity, modular counting and permutations, not
    carries.
  - 2510.14826 needs external tool memory for long addition.
- **An additive linear-attention state as an accumulator of sufficient statistics (Σ v kᵀ)** is the standard reading of linear
  attention: fast weights 2102.11174; in-context regression by GD, von Oswald 2212.07677. A carry-save column sum is that
  mechanism applied to digit products. **New in application.**
- **Reading a recurrent memory without writing:**
  - read-only queries in RMT / ARMT 2407.04841;
  - LM2 2502.06049 (cross-attention reads);
  - the auxiliary decoding of a running sum, 2510.00184 (a linear probe on h), is the closest *auxiliary-target* precedent;
  - SIM-CoT's per-latent decoder is the closest *text-reader* precedent.
- **New as scoped:** carry-save DeltaMem as the sole cross-segment channel, with a per-step text reader, earliest-first removal, and
  donor SERIAL on the memory.

## (2) Removal pitfalls when the carrier is a recurrent memory
1. **The input doesn't change at removal.**
   - ICoT-SI removes *tokens*, which shifts positions and inputs; its smoothing and optimizer reset target that shift.
   - Here only the loss changes, so the main ICoT-SI instability source is absent.
   - The **AdamW reset may do more harm than good** to the memory's gate / gain parameters, whose moments are settled. Make the reset
     an explicit choice, or reset only the reader / decoder groups, and log the loss jump at each removal.
2. **Format drift after removal.** With S_j targets gone, the memory can reorganise (for example, into b-prefix storage, (4)) while
   the final answer holds.
   - Keep decoding the removed S_j unweighted (retention), as LIT_MULT_READ_REMOVAL recommended. A retention drop with stable final
     accuracy = the format moved.
3. **No-decay additive heads grow in norm with the number of segments.** That's harmless at 4 segments. At 5×5 or longer chains the
   readout sees larger sums than in training (C2 drift). Log the per-segment memory norm, and check at 5×5 before claiming length.
4. **Order:** S_1 → S_2 → S_3 is earliest-first, as ICoT-SI and Coconut. Good.

## (3) The compute framing against ICoT-SI
- **Against TC-M it's fair:** about 9e9 vs 1.55e10 FLOPs (about 0.58×) and about 12 vs 57 sequential passes (about 0.21×), at equal
  accuracy.
- **Against ICoT-SI it's not a C3 win:** ICoT-SI is equally accurate (0.998), with about 0.55× our FLOPs (4.93e9) and about the same
  number of passes (13 vs about 12).
- ICoT-SI is *also* latent (no CoT at inference), so "latent beats text" is already shown by it. **Our distinguishable claims are
  C1 and the carrier, not C3.**
- **Suggested wording:** "DELTA-READ matches text CoT's accuracy at about 0.6× its FLOPs and about 0.2× its sequential passes, with
  serial causal use of an explicit memory (SERIAL). ICoT-SI is cheaper still at 4×4 and equally accurate, but its computation has no
  causally testable carrier."
- Also report **training compute** (48k steps × batch vs ICoT-SI's up to 200 epochs on 808k items). That may be a genuine advantage;
  state it separately.
- A real C3 claim needs a setting where ICoT-SI degrades: 5×5 at matched budget, or larger. ICoT-SI 5×5 is 0.996 per the repo log
  and 0.95 per the paper.

## (4) The b-prefix shortcut (L1) with a matrix memory
- **Capacity allows it:** 12 heads of d_k × d_v easily store b_1..b_j at position keys (and a, though a is visible every segment).
- **The shortcut then needs a one-pass 4×4 multiplication at the final decode.** That is GPT-2 small's no-CoT regime (0.29 at
  ICoT-SI's budget; ours 0.003). So it is unlikely to *reach* PASS, but it can coexist with partial carry-save storage, especially
  after removal.
- **Make the SERIAL test discriminate the hypotheses:** use donors that differ from the recipient in **both a and the b-prefix**.
  After a swap at segment 2 the predictions differ:
  - (i) carry-save S_2 carriage: S_2^donor + a^rec × (b_3 b_4)^rec at their place values;
  - (ii) b-prefix storage with the visible a: a^rec × (b_1 b_2)^donor (b_3 b_4)^rec;
  - (iii) recipient recomputation: the recipient's own product.
  - With equal a, (i) and (ii) coincide, so require a^donor ≠ a^rec.
  - Report the share of each, as §104's donor-consistent / recipient-consistent split.
- **A second probe:** decode b_1..b_j from the memory with a held-out linear / MLP probe. High b-prefix decodability *after* removal
  is the L1 signature.

## Verdict
- **New as scoped.**
- **Three additions:**
  1. three-way donor predictions with a^donor ≠ a^rec;
  2. a deliberate choice on the AdamW reset at removal (input unchanged);
  3. C3 stated against TC-M, with ICoT-SI reported as cheaper; the claim vs ICoT-SI is C1, not compute.
