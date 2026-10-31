# LOOKUP-REBUILD v1 (§253): a C1 loop that finds its steps by name (2026-10-31)

**Status:** v1 after REVIEW_LB.md (GO-with-fixes, all adopted). Supersedes LB_draft.md. Scorer: theory/score_lb.py (fixtures 9/9), committed before any output.

## 1. Why
- The C1 loops are position-bound (Note 251-16): TEST-PERM d 2–4 0.15–0.20; pass 1 0.10.
- The cause is the init's data (Note 252-4).
- The retrained init s252_askperm_s1 has one-hop name lookup (Note 252-5): PERM 0.988 on REG no-wrap items.
- **Question:** does a loop built from it, along §237's path with permuted training text (perm_p 0.5), chain by name?
- **Scope:** plan given; any trained order. Every test order is also a trained order.

## 2. Stages (RTX 3090; micro pinned at the reference values; --keep_final_ckpt; binding evals = the cloud in-run eval only; the local box only for CPU G-D smokes)
**Step-0 gate ($0):** eval_test_perm on the askperm init under P2P's args at step 0 (dir lb_step0).
- Required: by_d['2'].ws['1'] ≥ 0.95 on all 1,000 TEST-PERM d 2 items, wrap included.
- Fail → STOP (STEP0-FAIL).

**Stage 1, P2P (d ≤ 2, aux on, 12k):**
- p2p_s1 / p2p_s2: aw_plaw2_s1's args (TOPO law) with `--init_trunk s252_askperm_s1/model.pt` (+ sha), `--perm_p 0.5`, eval_parts + test_perm. Seeds 1 / 2.
- p2p_old_s1 (attribution, descriptive): the same with the askname init (s233), i.e. perm training without the lookup init. A single-factor contrast on the Stage-1 TEST-PERM reads.
- **Gate per seed:**
  - NOT-FORMED: eval_acc by_d['2'].acc < 0.90;
  - POSITION-REEMERGED: eval_test_perm by_d['2'].acc_kprime['8'] < 0.80 or ws['1'] < 0.80;
  - else SEED-OK.
  - No SEED-OK in either seed → NO-SEED, STOP.

**Stage 2, LBR (REG d ≤ 4, aux on, 16k final):**
- lbr_s1 from p2p_s1 and lbr_s2 from p2p_s2: rcb_plaw4_s{1,2}'s args with `--init_model p2p_s{k}` (sha asserted), `--perm_p 0.5`, eval_parts + test_perm.
  - init_trunk is set to askperm for provenance; the strict full load from init_model overrides it.
- If only one P2P seed is SEED-OK, both LBR seeds start from it (seeds 1 / 2), as §237 did.

**Args diff vs the existing path:** init (askperm, + sha), perm_p, eval_parts.

## 3. Readings (Stage 2, per seed; binding)
**TEST-PERM (eval_test_perm.json):**
- by_d[d].acc_kprime['8'] ≥ 0.90, d 2–4;
- NOT-DECODER: by_d['4'].acc_kprime['2'] ≤ 0.20 (eval build adds K′ 2 at d 4);
- LP5P: mean_at_2_d4.intact['8'] − mean_at_t['8'] ≥ 0.30.

**RETAIN (TEST2):**
- eval_acc d 1–4 ≥ 0.98;
- SWAP4: eval_swap_kd by_d['4'].swap_mid['4'] follow_U ≥ 0.50 and ≥ floor + 0.30;
- LP5-D4: eval_lp5h d 4 ≥ 0.30.

**Guards:** CUE-ONLY d 3–4 (the score_ae rule); STABLE (the score_ao rule, as §251 / NS_v1).

**Labels (first match):**
- VOID / UNSTABLE;
- NOT-FORMED (TEST2 d 2 < 0.90);
- DECODER (NOT-DECODER fails);
- **LOOKUP-C1** (TEST-PERM + RETAIN);
- LOOKUP-NOCAUSAL (TEST-PERM passes, RETAIN fails);
- PARTIAL (TEST-PERM d 2 ≥ 0.90);
- ORDER-ONLY.
- Per arm: PASS (both LOOKUP-C1) / SPLIT / FAIL.

**Descriptive:**
- pass-1 WS on TEST-PERM;
- a cue_wrong on TEST-PERM (does the answer follow the name?);
- a probe of S_t for spine identity and pass index;
- the add_s-ambiguity stratum;
- TEST-PERM by number of inversions;
- d 5–8 on TEST2 only (test_perm stops at d 4);
- the attribution contrast p2p_old_s1 vs p2p_s1.

## 4. OC
- Step-0 gate: P ≈ 0.9 (REG no-wrap 0.988 already).
- Stage 1: P(SEED-OK per seed | lookup survives) ≈ 0.8; P(… | a position route re-emerges) ≈ 0.1.
- Stage 2: P(LOOKUP-C1 per seed | lookup survives) ≈ 0.5; P(… | position re-emerges) ≈ 0.05.
- Both seeds LOOKUP-C1: ≈ 0.25 vs ≈ 0.003.
- **Censoring:**
  - VOID-RERUN once on a crash or an unfinished run;
  - Stage 2 only after the Stage-1 gate;
  - no extensions; 16k final.

## 5. Cost
- Stage 1: 3 × 12k at ~$0.68 (aw_plaw2_s1 realised) ≈ $2.0. Stage 2: 2 × 16k at ~$0.8 ≈ $1.6.
- Total ≈ $3.6–4.5; guard $5. Credit $25.7.
- **Build (eval only):** K′ 2 at d 4 in part_test_perm. --perm_p already works on both laws (REVIEW_LB #1).
- **G-D:**
  - CPU smokes of perm on TOPO (solve + disjointness);
  - the step-0 gate;
  - args diff;
  - K′ 2 present in eval_test_perm.
