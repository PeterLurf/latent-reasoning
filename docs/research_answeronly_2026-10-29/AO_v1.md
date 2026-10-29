# ANSWER-ONLY v1: does the d 2 latent loop form without per-pass value targets? (2026-10-29)

**Status:** v1 = AO_draft.md (c6ba5b25) with the red team's findings (REVIEW_AO.md: GO-with-fixes) and Audit's three points. Where they differ, this file wins. Theory's choice under the user's standing rules.

## 1. Scope (Audit (1); red team 4) — this is NOT a C4 test in the strict sense
- The init s233_askname_s1 was trained on QUESTION(u, X) → v_X over intermediate entities, i.e. the simulator's per-step values, which NORTH_STAR excludes.
- d = 1 items (half the training mix) are short-problem labels of v_1 (the §110b-1 qualifier).
- The step cue gives the plan.
- At d = 2 the decoder takes step 2, so the readings show pass-1 causal use, not chaining (DEC-STEP2).
- **Claim label on a pass: "AUX-FREE (askname init, d 1 foothold, plan given; DEC-STEP2)".** It answers one narrower question: is the per-pass aux loss needed for the loop to form, given an askname-pretrained trunk?

## 2. Derivation, and the history prior (Audit (2))
- §171 / §176 / §177: final-answer-only loops with identical passes never trained.
- **What differs here:**
  - (i) each pass's input is the askname question form the init already answers (G-AW0: v_1 at ':' ≥ 0.80 at step 0);
  - (ii) the write comes from that ':' state, so pass 1 can write v_1 with no new computation;
  - (iii) the gradient reaches pass 1's write directly, because dec_logits read S_K, which holds pass 1's write (inject_ow.py:85; no detach in deltamem; red team 2), and also through the reads at later passes.
- So AO only has to learn to read S at the decoder. That is the same need as in P-2, minus the per-pass shaping.
- **Prior:** the history argues against; (i)–(iii) argue for. P(AUX-FREE per seed) ≈ 0.3.

## 3. Arms
- **AO:** ao_s1 / ao_s2 run the §234 P-2 recipe with `--aux_step 1 --aux_fade 0,0` (red team 1: aux_step 0 trips the asserts at inject_aw.py:207 / inject_ow.py:211).
  - lam_at is then exactly 0 at every step ≥ 1; the aux forward runs at weight 0, so cost matches.
  - Args diff vs AUX = {aux_fade} (+ seed, name).
- **AUX comparator:** aw_plaw2_s1 (existing) and aux_s2 = aw_plaw2_s1's args with seed 2 (single-change gate; Audit (3)).
- **All new runs:** `--args_ref boxes/runs_fsq/aw_plaw2_s1/args.json`, on an RTX 3090.
  - The module has changed since P-2 (§235–§243 flags, default off), so a gate replays aw_plaw2_s1's first 100 steps bitwise (log step 100 loss 3.3319097) before launch.
- **CD-AN:** aw_pcd2_s1; DIRECT bar s228_pdir2c_s1. Data are matched across arms (fixed OW.SEED_TRAIN stream).

## 4. Readings (theory/score_ao.py; per AO seed; binding)
- **LP1:** the last-bin d 1 train acc ≥ 0.90.
- **LP3:** d 2 acc ≥ 0.70 and ≥ CD-AN + 0.30.
- **LP5:** the mean pass-1 state (cue kept) drops d 2 acc by ≥ 0.30.
- **MEAN-STATE:** pooled mean-state acc ≤ CD-AN + 0.10 (restored; binding in P-2).
- **STATE-USE:** SWAP-K follow_U ≥ 0.50 and ≥ floor + 0.30, n_pairs ≥ 1,000.
- **CUE-ONLY guard** → VOID.
- LP2 is dropped: it reads the aux-trained head (score_st.py:83).
- **STABLE (red team 5):**
  - train_summary.n_nonfinite ≤ 10 (non-finite steps are skipped);
  - on the deduplicated 100-step loss rows after step 1,000, no row with loss > max(3 m, 0.05) (m = the median of the previous 10 rows) that fails to return to ≤ max(m, 0.05) within the next 5 rows;
  - a rise in the last 500 steps counts only if the final row is still > max(3 m, 0.05).
- **STILL-RISING (red team 6; ST_v1 rule):** if LP3 fails and the slope of the last 4 d 2 train-bin accs is > 0.02 per bin, the seed gets one pre-registered +6k extension (resume from 12k; ~$0.2). It is re-scored by the same rules at 18k, which is final.
- **Labels, first match:**
  1. VOID;
  2. UNSTABLE;
  3. STILL-RISING (pending the extension);
  4. NOT-FORMED (LP3 fails);
  5. ACC-ONLY (LP3 passes, LP5 / MEAN-STATE / STATE-USE fails);
  6. AUX-FREE (all pass; LP1 must also pass, else NOT-FORMED).
  - Plus CUE-WRONG: the C1-TOLD vs plain label.
- **Overall:** AUX-FREE (both seeds) / SPLIT (one) / FAIL.
- **SURVIVAL (descriptive):** AO / AUX same-seed ratios of d 2 acc, LP5 drop and SWAP-K follow; LP0 and the pass-1 ':' argmax (WS) descriptive.

## 5. OC (2 seeds)
- P(AUX-FREE per seed | the aux is unnecessary given askname) ≈ 0.7; | the aux is necessary ≈ 0.02. Prior 0.4, so per seed ≈ 0.29.
- Both seeds (ρ ≈ 0.5, one trunk) ≈ 0.18. P(ACC-ONLY in some seed) ≈ 0.15.
- Under no effect, P(AUX-FREE both) < 0.01.

## 6. Pre-named next (red team 8)
- The §227 text-CoT teacher's weights are gone. At 1.000 accuracy its intermediates equal the aux targets, so "distillation" from it would be P-2 again; it is not named.
- On **AUX-FREE:** the next C4 step removes the askname foothold, i.e. AO from the base GPT-2 init (or from a text-CoT-trained init), the stricter test.
- On **FAIL:** the per-pass aux is necessary in this setup. The C4 line then pauses for a new design, with no pre-named lever.

## 7. Cost
- Three arms (ao_s1, ao_s2, aux_s2) at ~$0.45 ≈ $1.4, plus an optional extension of $0.2 each. Guard $3.5.
- Running: realised $16.65 + $3.5 = $20.15 ≤ $30.
