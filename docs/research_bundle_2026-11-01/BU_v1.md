# BUNDLE v1 (§259): one model meeting the C1, C2, C6 and C5 gates (scoped), conditional on L2 (2026-11-01)

**Status:** v1 after REVIEW_BU.md (GO-with-fixes, all adopted). Supersedes BU_draft.md. **Conditional:** it launches only if Note 257-2's L2 reads DRIFT in at least one dis seed, and the penalty arm runs only on the drifting seed(s).
- If both seeds HOLD, the bundle is eval-only, at $0 extra: dis_s* with L1's head, scored with the gates below.
- The scorer (theory/score_bu.py) is committed before launch, after L2 lands.

## 1. Why
- **The pieces:** C2 in dis_s* (§256), C6 via §240's hold-increment penalty (lh lineage), and C5 via a post-hoc halt head (§248; out-of-range test L1 pending).
- **Mechanism risk (REVIEW_BU B3):**
  - At hold passes the input equals pass d's (cue S_d), while at test on d 6, pass 5 sees a new cue S_5.
  - "Write ~0 at pass ≥ 5" fits every penalised training pass, and would suppress the untrained-pass computation that gives dis its C2.
  - §240's d 5 evidence is mixed: FH − CTRL +0.46 in s1, −0.12 in s2, from inits that computed pass 5 at only ~0.55. So C2 is a retention gate here.

## 2. Arms (RTX 3090; micro 2; 8k; hold_start 500 as §240)
| Arm | Runs | Init | Change |
|---|---|---|---|
| BUN | bun_s<k> (drifting seeds) | dis_s<k> | + --hold_inc 1.6 + --hold_log, keeping --distract_p 1 |
| BUN-CTRL | bunc_s<k> | dis_s<k> | the same 8k FT with --hold_log and no penalty (LP0 calibration, a one-sided norm guard, C2 / C6 attribution) |

- **Args diff vs dis:** {init_model + sha, steps 8000, hold_inc 1.6, hold_log}, plus the disclosed fresh AdamW, LR warmup, fade re-ramp and repeated item stream (Note 257-1). args_diff.json is shipped.
- **C5:** a post-hoc halt head fitted on the frozen BUN model (§248 procedure, fit d ≤ 4, test d 1-7, the L1 build). Not joint training; no claim of equivalence.

## 3. Readings (per BUN seed; binding)
- **C1 RETAIN** (score_ds RETAIN path): TEST2 d 1-4 ≥ 0.98; NOT-DECODER; SWAP4; LP5-D4; CUE-ONLY.
- **C2 retention vs dis (same seed):** d 6 / d 7 ≥ dis − 0.05; WS d 6 passes 5-6 ≥ 0.90. Descriptive: d 8-10 at K' 8 / 12 / 16.
- **C6:** score_hx.hold rule (d 1-6, K' 64 / 128 ≥ K' 8 − 0.02), the ||S|| ratio K' 128 / 8, and SEED-VAR |s1 − s2| ≤ 0.05 when both seeds run.
  - **Norm guard** (one-sided vs BUN-CTRL, score_lh.norm_guard): GAMED if ||S_8|| or ||S_{d+2}|| exceeds 1.25 × CTRL in any bin.
  - **LP0-W:** windows with step > 1000, cap = max(2 × CTRL post-1k max, 10).
- **C5:** score_hx's C5-EXTRAP rule (IN-RANGE d 1-4 + EXTRAP d 5-7, with censoring).
- **STABLE.**

**Labels:**
- **BUNDLE-PASS (scoped).** Scope: plan given; chain-order text; REG frames 0-5; one rcb lineage; per-pass value aux (no C4); C2 is extrapolation within K = 8 passes, and the K' > K gain is not tested (the penalty makes K' > 8 a fixed point by design).
- Otherwise the failing criteria are named, or FAIL.

## 4. OC
- P(C6 | penalty) ≈ 0.6 (§240's 2/2 was at d 1-4; d 5-6 hold was never trained under the penalty).
- P(C2 retained) ≈ 0.6 (B3).
- P(C5-EXTRAP) from L1.
- **Censoring:** missing parts are pending; FAIL at final; VOID-RERUN once.

## 5. Cost
- ~$1.0 per BUN arm + ~$0.7 per BUN-CTRL (distract texts, K' 128 d 1-7 run_long, halt-head fit). Up to 2 + 2 cells ≈ $3.4; guard $4.
- mem_check with the penalty + distract on. Host sha asserts.
