# UNSCAFFOLDED draft: does the loop chain when the per-pass cue no longer names the step's entity? (C1 without "plan given") (2026-10-30)

**Status:** draft for red team. Theory's choice among Audit's three premise-changing candidates (10-30).
- C1 is the top priority, and "plan given" is the qualifier on every language C1 claim, so this is the cheapest test that removes a premise.
- Audit's REAL-DATA candidate (GSM8K-Aug AW) is next, with its own $0 stage.

## 1. Why
- **The scaffold:** every pass today reads "How many <u> does <S_t> have? The answer is:". The cue names the entity whose value pass t must compute, so the plan (which entity, in which order) is given (summary; §237 C1-TOLD).
- **Change:** one generic cue for every pass, "What is the next value? The answer is:" (identical tokens at every pass; the ':' write as AW).
  - Each pass must then work out from S_{t−1} and the text which entity comes next and compute it. The memory has to carry both a value and a position in the chain.
- **Hold:** passes after d have no next step. Their aux target is the final value, as now; hold_inc as §240 is optional, not used here.

## 2. Arms (warm start; REG law; d ~ U{1..4}; K 8; 16k; RTX 3090; micro pinned)
| Arm | Runs | Init | Change |
|---|---|---|---|
| NS | ns_s1 / ns_s2 | rcb_plaw4_s1 / s2 | `--cue generic` (the generic pass cue) |

- The decoder input is unchanged: the full text + the final question, which names only the target entity.
- The aux targets (per-pass value, t ≤ d) are unchanged. This is a C1 test, not C4.
- **Comparators:**
  - rcb_plaw4_s1 / s2 (scaffolded; C1 PASS);
  - CD-AN rcb_pcd4_s1 (one pass);
  - **NS-COLD** (optional, if the warm arm passes): the same from aw_plaw2_s1, to rule out an inherited plan.

## 3. Readings (REG TEST2; per seed; binding; §251 d 4-centred gates)
- ACC d 1–4 ≥ 0.90.
- NOT-DECODER: d 4 at K′ 2 ≤ 0.20.
- SWAP4: SWAP-MID d 4 t 2 ≥ 0.50 and ≥ floor + 0.30.
- LP5-D4 ≥ 0.30.
- CUE-ONLY guard; STABLE.
- **PLAN-FREE assert:** the think-cue token ids are identical across passes for every item, checked in the build and the eval.
- **Labels:**
  - C1-UNSCAFFOLDED (all gates, both seeds);
  - SPLIT;
  - SCAFFOLD-NEEDED (ACC fails at d ≥ 3);
  - VOID / UNSTABLE.
- **Descriptive:**
  - an entity probe: decode step t's entity name (spine index) from S_t;
  - WS per pass;
  - d 5–8;
  - the learning curve.

## 4. OC (sketch)
- P(C1-UNSCAFFOLDED per seed | the loop can track chain position in S) ≈ 0.4; P(… | the plan must be given) ≈ 0.03.
- **Risks:**
  - The REG chain order is recoverable from the text (each step's source is the previous entity), so a pointer may be learnable.
  - Warm init: the trained trunk expects names in the cue, so it may first collapse (onset break, Note 251-8). The aux targets should re-form it, as in AEX.

## 5. Cost
- 2 × 16k (rcb realised ~$0.8) ≈ $1.7; guard $2.5. Credit $25.7.
- **Build:** a `--cue generic` option in tok_st (think cues), threaded through every tok_aw call site (REVIEW_BD #4 lists them), plus the PLAN-FREE assert.
- **G-D:**
  - default-off bitwise;
  - a step-0 eval of rcb under the generic cue (expected collapse: the step-0 ref);
  - gradflow;
  - args diff.
