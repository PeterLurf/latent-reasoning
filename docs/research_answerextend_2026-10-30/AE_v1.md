# ANSWER-EXTEND v1 (§251): does depth beyond the seed need hand-built per-pass targets? (C4-partial) (2026-10-30)

**Status:** v1 after red team (REVIEW_AE.md, GO-with-fixes, all adopted) and Audit's G-D points. Scorer theory/score_ae.py (fixtures 7/7). Supersedes AE_draft.md.

## 1. Question
- §237: from the P-2 seed (aw_plaw2_s1: d ≤ 2, per-pass aux), REG training with aux at **all** depths reached d 3–4 at 1.000.
- Do passes 3–4 need hand-built per-pass targets, or do final answers suffice once the loop exists at d ≤ 2?
- C4-partial at best: the seed itself used simulator targets.

## 2. Arms (init aw_plaw2_s1, §237's rcb_plaw4 recipe: REG law, d ~ U{1..4}, 16k, the same seeds and data orders; RTX 3090 only)
| Arm | Runs | Change vs §237 | What it tests |
|---|---|---|---|
| AEX | aex_s1 / aex_s2 | `--aux_d_max 2` (new flag; aux CE only on d ≤ 2 batches) | passes 3+ from answers only, while passes 1–2 keep their targets |
| AEX0 | aex0_s1 / aex0_s2 | `--aux_fade 0,0` (existing flag; λ_aux = 0 from step 1, inject_aw.py:247) | per-pass targets **only in the seed** |
| CTRL | rcb_plaw4_s1 / s2 (existing) | none (aux on all depths) | §237 PASS; re-evaluated under the current eval module → rcb_plaw4_s{1,2}_re (eval sha drift 21ae80 vs bd515f) |

- Args diff: AEX = {aux_d_max 2}; AEX0 = {aux_fade "0,0"}. 16k is final, with no extension (§244's re-warm lesson).
- **Build (AEX only):** batches hold one d per step, so a d 3–4 step under `--aux_d_max 2` has no aux item. As written this crashes (`torch.stack([])`, inject_aw.py:283). Fix: aux := 0 (a zero tensor in the graph) when no item qualifies.
- What remains shaped: in AEX the decoder CE trains every pass through S_K (no detach; hold_inc 0; the fade restarts as in CTRL). Per-pass targets exist only for passes t ≤ d (inject_aw.py:271), so there is no hold target to remove.

## 3. G-D gates (before any launch)
1. `--aux_d_max` default off: bitwise equal to §237's module (det wrap) over the first 100 steps of rcb_plaw4_s1's args.
2. Flag on: bitwise equal to CTRL until the first d ≥ 3 step. A smoke run passes through a d 3 and a d 4 step without error.
3. Gradflow: in AEX, aux gradient zero on d ≥ 3 steps and normal on d ≤ 2 steps. At d 4, a nonzero answer-CE gradient on the write and read params.
4. Step-0 refs (init aw_plaw2_s1 under the current eval module, REG): d 3 / d 4 = 0.028 / 0.096 (Audit's pre-stated values; the eval re-run confirms them).

## 4. Readings (REG TEST2; per seed; binding; d 4-centred)
**Why d 4-centred:** d 3 can pass via passes 1–2 plus the decoder's last step. CTRL d 3 acc at K′ 2 is 1.000, while d 4 at K′ 2 is 0.008 / 0.012. Only d 4 requires a third pass trained by the answer alone.
- **ACC4:** d 4 acc (K′ 8) ≥ 0.50 and ≥ CD-AN (rcb_pcd4_s1, 0.090) + 0.25.
- **NOT-DECODER:** d 4 acc at K′ 2 ≤ 0.20.
- **SWAP4:** SWAP-MID d 4 (t 2; by_d.4.swap_mid["4"]): follow_U ≥ 0.50 and ≥ floor + 0.30.
- **LP5-D4:** lp5h d 4: intact − mean-at-t (K′ 4) ≥ 0.30.
- **CUE-ONLY guard (d 3–4):** answer_acc ≤ max(majority, min over the DIRECT bs-variants) + 0.05.
- **STABLE:** the score_ao rule.
- **Labels (first match):**
  - VOID (guard fails);
  - UNSTABLE;
  - EXTEND (all four d 4 gates);
  - EXTEND-D3-ONLY (d 3 acc ≥ 0.60 and SWAP-MID d 3 t 2 passes, d 4 gates fail);
  - NOT-EXTEND.
- **Per arm overall:** EXTEND (both seeds) / SPLIT / FAIL.
- **Descriptive:**
  - WS at passes 3–4 (the ':' readout is shaped by the decoder CE equally in both arms, so it does not discriminate);
  - the learning curve from the log's by-d answer acc vs CTRL (d 3 ≥ 0.9 at 2k, d 4 ≥ 0.9 at 4k);
  - SURVIVAL ratios vs CTRL_re;
  - CD-AN d 3 / d 4 = 0.085 / 0.090.

## 5. OC (per seed, per arm; two data orders, same init, not independent)
| | P(EXTEND \| answers suffice) | P(EXTEND \| per-pass targets needed) |
|---|---|---|
| AEX | 0.45 | 0.04 |
| AEX0 | 0.30 | 0.02 |

- **Rationale:**
  - The read-into-compute route exists from the seed: pass 2 at 1.000. The answer gradient reaches pass 3 through S_K.
  - §244 / §245: forming a loop from answers alone succeeded in 1 of 4, and that was without a seed.
  - AEX0 is lower because passes 1–2 can also drift without their targets.
- **Both seeds EXTEND:** AEX ≈ 0.25 under "answers suffice" and ≈ 0.003 under "targets needed".
- **d 3 alone:** P(EXTEND-D3-ONLY or better) ≈ 0.7 even under "targets needed". This is why d 3 is descriptive.
- **Censoring:** a run that does not finish 16k, or whose eval crashes, is VOID-RERUN once on a fresh host. It is never scored on partial logs.

## 6. Cost
- 4 × 16k runs at ~$1.0–1.2 each (rcb realised ~$0.8) plus 2 CTRL re-evals (local or ~$0.1): ≈ $4.5 guard; expected ≈ $4.0.
- Vast credit $30.56 → ~$26.
- Above the ~$3 step guide because AEX0 was added (the red team's honest test of "seed-only"). Within the $30 cap.
