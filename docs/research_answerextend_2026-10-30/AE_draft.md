# ANSWER-EXTEND draft: can the loop extend from d ≤ 2 to d 3–4 from final answers only? (C4-partial) (2026-10-30)

**Status:** draft for red team. Theory's choice; top of docs/ROADMAP_OPEN_CONDITIONS_2026-10-30.md.

## 1. Why
- §244 / §245: without per-pass targets the d 2 loop rarely forms from the askname init (1 of 4).
- §237: from the P-2 seed (aw_plaw2_s1; d ≤ 2 with per-pass aux), REG training with aux at all depths reached d 3–4 at 1.000 within ~3k steps.
- **Question:** once the loop exists at d ≤ 2, do the new depths (passes 3–4) still need hand-built per-pass targets, or do final answers suffice?
  - If answers suffice, hand-built targets are only a seeding cost, which is the scale-relevant form of C4.
  - C4-partial: the seed itself used simulator targets.

## 2. Arms (one change vs §237)
- **AEX:** aex_s1 / aex_s2 run §237's rcb_plaw4_s{1,2} recipe exactly (init aw_plaw2_s1, REG law, d ~ U{1..4}, 16k, the same seeds) with a new flag **--aux_d_max 2**.
  - Per-pass aux CE applies only to items with d ≤ 2; d 3–4 items get the answer CE only.
  - Default off = bitwise current.
- **CTRL:** the existing rcb_plaw4_s1 / s2 (aux on all depths), the same recipe, init and seeds. §237 PASS (d 3–4 1.000).
- Args diff = {aux_d_max 2}.
- Disclosed: the d ≤ 2 items still carry per-pass targets, so passes 1–2 keep being shaped and passes 3+ only through the answer.

## 3. Readings (REG TEST2 sets; per seed; binding)
- **ACC:** d 3 ≥ 0.60 and d 4 ≥ 0.50, each ≥ CD-AN (rcb_pcd4_s1) + 0.25.
- **CAUSAL:** SWAP-MID d 3 at t 1 and t 2, follow_U ≥ 0.50 and ≥ floor + 0.30; LP5-4 (mean mid-state) pooled drop ≥ 0.30.
- **CUE-ONLY guard;** STABLE (the §244 rule; LP0 descriptive).
- WS at passes 3–4 is **descriptive** (the ':' readout at passes ≥ 3 is never trained in AEX, so a failing WS need not mean a missing chain).
- **Labels:** EXTEND (ACC + CAUSAL + guard + STABLE) / ACC-ONLY / NOT-EXTEND / VOID / UNSTABLE. Overall both seeds.
- **SURVIVAL vs CTRL (descriptive):** d 3–4 acc, swap and LP5 ratios. The learning curve (steps to d 3 ≥ 0.9) vs CTRL's ~3k.

## 4. OC (sketch)
- The read-into-compute route already exists from the seed (pass 2 at 1.000), and the decoder takes the last step. d 3 needs pass 2's output re-used at pass 3 with a new op, trained only through the answer.
- P(EXTEND per seed) ≈ 0.35; both ≈ 0.2.
- Under "per-pass targets needed": P ≈ 0.03.

## 5. Cost
- Two 16k runs (rcb realised ~$0.8 each) ≈ $1.6; guard $3. Vast credit $30.56 → ~$29.
- Build: the --aux_d_max flag, a flags-off bitwise gate vs rcb_plaw4_s1's first 100 steps, and gradflow (zero aux grad from d ≥ 3 items).
