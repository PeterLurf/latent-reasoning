# Synthesis spec v0 (Theory, 2026-10-16): VB-2, fixing the three bottlenecks on VERSION-BANK

**Sources** (read in full; all in docs/research_bottleneck_2026-10-16/):
- S_step_capability.md (inventor S): COMMUTE + OPX (+ SELF-SNAP); the diagnosis that failed seeds recompute.
- D_depth_invariance.md (inventor D): SS-VB, i.e. compaction + restamp + V0-OFF + age floor + STALL-AUG + CANON; RESTART-SPLIT.
- H_halting.md (inventor H): CONTENT-HALT + STALL-AUG.
- LIT2.md (prior art).

**Base design:** docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md and THEORY.md §211 with notes 211-1..15 (§211 lands about 05:00; nothing is read yet).

**User requests:**
- 02:44 EDT: "fix bottle necks": (1) per-pass step capability, (2) beyond-training depth (C2), (3) learned stopping (C5).
- Standing (22:07, 10-15): pass-global attention memory, not RNN memory.

## Shared diagnosis
- **Failed seeds recompute from the re-shown input.** Step success depends on subtree height, not on the pass (S's HH-FIT; Audit has not reproduced it yet; scripts committed). Failed seeds write no usable value after pass 2 (205-4, Audit-reproduced). So the per-pass step is not the limit in converged seeds; recompute wins the early race in failed ones.
- **VB carries about seven size or clock cues** (D §1.1):
  - absolute wpe;
  - the PA gather's wpe queries;
  - the VOID count;
  - the bank normaliser;
  - V_0 every pass;
  - K ≤ 12, which makes "t ≥ 9 ⇒ hold" zero-loss;
  - copy-age drift.
  V_1's fixed-origin age is a clock (H §1).
- **Feng's step is a radius-2 local rule plus one prefix-OR in compacted coordinates** (D Lemma 0, 0 failures in 852,056 steps). Compaction makes the state after t steps a fresh size-(n−t) problem (D Lemma 1).
- **Full cross-iteration history hurts depth extrapolation** (LIT2: CoTFormer 2609.19521; Bohde 2403.04929). This is in tension with VB's full bank.

## VB-2 (proposed main cell): VB + the following, each tied to a bottleneck
1. **(Depth) Compaction M1 + restamp M2** (D §2): VOID slots are dropped between passes (teacher keep mask in training, R decode at eval), and positions are re-indexed to 0..m−1.
2. **(Depth / step) V0-OFF** (D M3): V_0 is masked for t ≥ 2, which structurally closes recompute-from-input after pass 1. **Age floor μ ≥ 2** on versions older than the newest, plus a relative-position bias ω on the newest.
3. **(Step) COMMUTE** (S §2.1): the R2 targets become the correct one-step reduction of the model's own decoded version (skeleton fallback), and R1-C makes the coda read dec V_t's value. **OPX** (S §2.2): operand-interchange edits in R's row space in every version since slot creation; the step must follow them (ρ_X 0.5).
4. **(Halting / clock) STALL-AUG** (H §2.4 = D M4): p_s 0.1–0.3, S ∈ {1, 2}. Targets follow content, τ and K are content-defined, and there is no K cap at 12.
5. **(Halting) CONTENT-HALT** (H §2.1): a zero-parameter readout DONE(dec V_t), with the G-PROG and STUCK-STOP guards.
6. **Base input:** CARRY (D's choice: the PA gather's wpe queries are a position cue, and the anneal cost N2 a bump, 211-11). The PA-relative gather is the constraint-faithful follow-up.
   - **OPEN:** the choice should follow §211's N1 / N2 outcome and the user's constraint.

## Ablation and optional cells (one seed, 48k, 3090)
- **B1** = VB-2 (1–6).
- **B2** = VB-2 + CANON (D M5): the compacted version is pinned to the prelude's fresh encoding of the remaining expression. Alternatively, S's SELF-SNAP (a row-space canonical code). Choose one.
- **B3** = VB-2 without COMMUTE / OPX: attributes the step fixes.

## Readings
- **B-C1** = VB-C1's untrained parts (DEEP, PROBE-GAIN, SERIAL-ONSET, VALUE-SWAP SERIAL-STEP) ∧ STEP-COMPOSES (S) ∧ C5-VB (H: HALT-EXACT, NOT-PREMATURE, STALL-SHIFT*, DEPTH-TIMING) ∧ SS-C2 (D: DEPTH-GENERAL, SELF-SIMILAR, STEP-FLAT).
- \* STALL-AUG trains stalls, so STALL-SHIFT / NO-CLOCK become trained-for. D says PF and RS replace them, and H proposed held-out stall forms (triple stall, REWIND). **OPEN:** which held-out clock test is decisive.

## Stage 0 ($0; pre-stated in 211-13 / 211-15; running needs the user's OK)
- CONTENT-HALT eval, step diagnostics D2 and depth diagnostics RS / PF / HIST-TRUNC on the §211 cells.
- D1 / RS / PF on INJECT models.
- They decide which bottleneck binds in VB, and so which of B1–B3 to propose.

## Open questions for proof / red team
1. Are compaction + V0-OFF + the age floor compatible with the user's pass-global constraint? The bank stays readable, but its use is recency-weighted.
2. Do COMMUTE + OPX + compaction + STALL-AUG interact badly (target re-indexing under stalls and compaction; OPX edits on compacted versions)?
3. Which held-out clock test is decisive once stalls are trained?
4. CANON vs SELF-SNAP for the canonical-state assumption (D's A3 / S's S2).
5. Optimisation risk from stacking five changes (N3's non-recovery, 211-12). Should B1 be smaller?
6. Cost and gates.
