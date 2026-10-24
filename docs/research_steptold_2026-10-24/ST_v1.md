# STEP-TOLD v1: a step-cued latent loop on the TOPO word programs, GPT-2 small (2026-10-24)

**Status:** v1 = ST_draft.md (ec88980f), changed as listed here; where they differ, this file wins. Red team: REVIEW_ST.md (GO with fixes 1–7; 8–11 also adopted). Literature: LIT_STEPTOLD_2026-10-24.md.
- N1 OK, narrow: "a latent-loop analogue of decomposition prompting" (least-to-most 2205.10625, self-ask 2210.03350, successive prompting 2212.04092, planning tokens 2310.05707).
- N2 narrow, reworded to "not cued" (Biran 2406.12775, Yang 2402.16837).
- N3 narrow (mean ablation standard; swaps per SCIT / §221).

## 1. Claim (fixes 2, 10)
- **"C1, scaffolded (plan given by the generator)"**, i.e. C1-TOLD. It adds nothing to C4. Any text comparison gives TEXT the same plan.
- D2's "gradients now agree" argument is withdrawn: §228's pass 1 was already asked for v_1 through the teacher-forced aux. ST adds only an input-side pointer.
- P(LP2) is revised to ≈ 0.25 [ASSUMED].

## 2. $0 local gates, before any spend (fixes 1, 7, 8)
- **P1-ASK+NAME** (replaces P1-ASK; one local run of about 30–40 min on the 4060): the §228 P-DIR2 recipe, with every item's think input cued "How many <u> does <X> have?" and target v_X. X is drawn uniformly from four classes:
  - **(i)** S_1 (one hop);
  - **(ii)** S_0 (given);
  - **(iii)** a distractor quantity (given);
  - **(iv)** S_2 with S_1's defining sentence replaced by "S_1 has v_1 u." (one hop from a given).
- **Pass:** test accuracy ≥ 0.80 on each one-hop class (i, iv) and ≥ 0.90 on each given class (ii, iii), with d = 1 ≥ 0.80. A fail stops ST at $0.
- **CD-local ($0 pre-gate):** CUED-DIRECT at d ≤ 2 locally (8k). If CUED-DIRECT d = 2 ≥ 0.80, the plan alone solves d = 2 one-pass → ST stops (a finding, not a fail of the arm).
- **The zero-shot check (st_told_zs) is relabelled DESCRIPTIVE.**
  - Its predictions were written before the run but not committed (a procedural slip, disclosed).
  - ASK0 (the trained trunk cannot report a named given value) is a finding about §228's trunks, not a failed pipeline control: D1 / ORIG reproduce the closed runs.
- The amended rules and the wrapper sha are committed before Experiments launches.

## 3. P-2 cloud pilot (one RTX 3090, d ≤ 2) (fixes 3, 4, 5, 9)
- **Steps:** 12k, with a still-rising rule (if LP3's d = 2 accuracy rises over the last 4 bins, one extension to 16k).
- **Gates:**
  - LP0 per arm, self-calibrated (REVIEW_LB #12);
  - LP1;
  - **LP2 on d = 2 items only** (key aux_vacc_by_d_pass);
  - **LP3 binding vs CUED-DIRECT**: d = 2 ≥ 0.70 and ≥ CD + 0.30;
  - **LP5 binding at K′ = d**: pass-1 state replaced by its mean, cue kept, must drop ≥ 0.30;
  - **pooled MEAN-STATE**: acc_MEAN ≤ CD + 0.10.
- **CUE-ONLY guard:** read on d ≥ 2 only, against max(majority, plain DIRECT) + 0.05. It trips → VOID.
- **G2b:** a batch-of-one fp64 cache check against cross-item cache reuse.
- **CUE-WRONG reading (labels the claim):** the wrong target name is given at pass t.
  - Accuracy drop ≥ 0.20 → the cue is used, claim C1-TOLD.
  - Otherwise plain C1 (the loop works without reading the cue).
- **Gates added:** G9 (decoded-input fixtures), G10 (log-key consistency), a CUED-DIRECT args diff.

## 4. P-4 and Stage 1 (fix 6)
- P-4 (d ≤ 4, 16k) as the draft, with LP4.
- **Stage 1 runs on TWO RTX 3090 hosts:** LAT-ST on one; CUED-DIRECT + plain DIRECT on the other. Wall ≈ 29–37 h at the same $.
- **Host-speed probe:** redeploy if the projection is > 40 h. Off-host full-state checkpoints + a tested resume.
- Stage 1 adds a binding mean-state ablation at t* (cue kept) and a TOLD-RECOMPUTE label. R3 (a state swap with the cue kept) reads serial use.

## 5. OC and cost (reproduced by REVIEW_ST)
- P-2 passes: P = 0 under no effect / recompute / cue-solves-one-pass / pass-1-only; 1.0 under strong serial use; 0.88 marginal.
- P(C1-TOLD) ≈ 0.04.
- **Cost:** guards P-2 $1.16 / P-4 $2.82 / Stage 1 $14.78 → family $18.76, **cap $19** (Audit). Expected spend ≈ $2.0. Realised at registration $0.
