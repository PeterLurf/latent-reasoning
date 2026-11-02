# BUNDLE draft (§259): one model meeting the C1, C2, C6 and C5 gates (2026-11-01)

**Status:** draft for red team. Bundling is at Theory's discretion (user via Audit).

## 1. Why
- **The pieces exist in separate models:**
  - C2: dis_s1 / s2 (§256): d 6 0.98 / 0.96; untrained passes compute.
  - C6: hold to K' 128 with the hold-increment penalty (§240, lh_fh lineage, without distract).
  - C5: a post-hoc halt head (§248, in range); its out-of-range test on dis_s* is pending (Note 257-2 L1).
- The rcb / dis lineage never had the hold penalty. §239 showed ||S|| growing ~linearly, and L2 (Note 257-2) measures dis_s*'s hold.
- **Question:** can one model pass every gate at once?

## 2. Arm (RTX 3090; micro 2; 8k)
| Arm | Runs | Init | Change |
|---|---|---|---|
| BUN | bun_s1 / bun_s2 | dis_s1 / dis_s2 | + --hold_inc 1.6 (§240's penalty: L_inc = 1.6 · mean ‖S_t − sg S_{t−1}‖² / ‖sg S_{d+2}‖² from hold_start 0), keeping --distract_p 1 |

- C5 is a post-hoc halt head fitted on the frozen BUN model (§248 procedure, fit on d ≤ 4), so not joint training.
- Joint training of the head with detached features would be equivalent, and gradient into the trunk would change the C2 / C1 model. Disclosed.
- The 8k FT restarts warmup / cosine (disclosed). The comparator dis_s* gives step-0 values for every gate.

## 3. Readings (per seed; binding; all existing gates)
- **C1 RETAIN:** TEST2 d 1-4 ≥ 0.98; NOT-DECODER; SWAP4; LP5-D4; CUE-ONLY.
- **C2:** d 6 ≥ 0.90, d 7 ≥ 0.75; WS d 6 passes 1-6 ≥ 0.90.
- **C6:** HOLD-LONG3 at K' 64 / 128 ≥ K' 8 − 0.02 for d 1-6, with the ||S|| ratio (K' 128 / 8) reported. LP0-W per the checklist (CTRL = dis's own windows).
- **C5:** score_hx's EXTRAP rule at d 5-7 plus IN-RANGE at d 1-4 (with censoring).
- **STABLE.**
- **Labels:** BUNDLE-PASS (all) / named partial (which criterion fails) / FAIL. Per arm: BUNDLE k/2.

## 4. OC (sketch)
- P(C6 | hold penalty added) ≈ 0.8 (§240 2/2); P(C2 kept) ≈ 0.8; P(C5-EXTRAP) depends on L1.
- Both seeds BUNDLE-PASS ≈ 0.3.

## 5. Cost
- 2 × 8k (~$0.6-0.8 each) + evals (incl. K' 128 run_long, halt-head fit) ≈ $1.8; guard $2.5.
- Build: none new (hold_inc + distract_p exist; halthead / run_long parts exist); host sha asserts.
- Order: after §257 and L1 / L2 land, so their numbers set the expectations; or in parallel if Audit prefers.
