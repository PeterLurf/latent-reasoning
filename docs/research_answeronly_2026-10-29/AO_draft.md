# ANSWER-ONLY draft: does the latent chain form without hand-built per-pass targets? (C4) (2026-10-29)

**Status:** draft for red team + Literature. Theory's choice under the user's standing rules (north star C4 = C2; C2's "more passes past the budget" failed in §239 / §240 / §243). Audit agrees C4 is next, with its two points adopted.

## 1. Why
- C4 (docs/NORTH_STAR.md): reach C1 from a pretrained LM trained on answers / rewards or text-CoT outputs, with **no hand-built intermediate states**.
- Every passing latent model so far (§234 P-2 onward) was trained with per-pass value targets from the task simulator (`--aux_step 1.0`: CE on v_t at pass t). That is exactly the hand-built supervision C4 excludes.
- The plan cue ("How many <u> does <S_t> have?") names entities, not values. It stays as the "plan given" scaffold, and every claim says so.

## 2. Arms (one change: per-pass aux off)
- **AO:** ao_s1 / ao_s2 run the §234 P-2 recipe exactly (askname init, answer-position write, γ 1, d ≤ 2, 12k steps, bs 64, lr 1e-4) with **`--aux_step 0`**: answer CE at the decoder only.
- **AUX (comparator, Audit (1)):** the same recipe with aux on, at matched steps and data.
  - Seed 1 = the existing aw_plaw2_s1 (P-2; LP3 1.000, LP5 drop 0.882, SWAP-K 1.000).
  - Seed 2 = a new aux_s2 (aw_plaw2_s2).
- **CD-AN** (one pass, no latent loop): the existing aw_pcd2_s1.
- **Args diff** AO vs AUX = {aux_step 0} (+ seed, name).

## 3. Readings (binding, per AO seed; P-2's gates)
- **LP3:** d 2 acc ≥ 0.70 and ≥ CD-AN + 0.30.
- **LP5 (pass-1 causal use):** the mean pass-1 state (cue kept) drops d 2 acc by ≥ 0.30.
- **STATE-USE:** SWAP-K on d 2 pairs: follow_U ≥ 0.50 and ≥ floor + 0.30.
- **CUE-ONLY guard.**
- **STABLE (replaces LP0 as binding, Audit (2)):** no non-finite loss, and no window where the loss rises above 3× its previous-1k median without recovering below that median within 500 steps. LP0 is reported descriptively.
- **SURVIVAL (descriptive, Audit (1)):** AO / AUX same-seed ratios for d 2 acc, LP5 drop and SWAP-K follow, plus the pass-1 probe of v_1 in S_1 (Audit's locus protocol).
- **Labels:**
  - C4-FORMED (LP3 + LP5 + STATE-USE + guard + STABLE);
  - ACC-ONLY (LP3 passes, LP5 or STATE-USE fails: answered without a causal chain);
  - NOT-FORMED (LP3 fails);
  - UNSTABLE;
  - VOID (CUE-ONLY).
- **Overall:** both seeds needed for C4-FORMED; SPLIT; FAIL.

## 4. OC (sketch)
- With no per-pass target, the gradient to pass 1's write comes only through the decoder's read at pass 2+. Coconut-style results suggest curricula help (to check with Literature).
- P(LP3 | AO) ≈ 0.4.
- P(LP5 and STATE-USE | LP3) ≈ 0.6.
- P(C4-FORMED per seed) ≈ 0.24; both ≈ 0.1.
- P(ACC-ONLY) ≈ 0.15.

## 5. Pre-named next
On NOT-FORMED or ACC-ONLY, the next step is distilled per-pass targets taken from a text-CoT model's generated intermediates (C4-admissible "from a text-CoT model's outputs"), not from the simulator.

## 6. Cost
- P-2 realised $0.68 for LAT + CD, so ~$0.45 per LAT arm. Three arms (ao_s1, ao_s2, aux_s2) ≈ $1.4; guard $3.5.
- Running: realised $16.65 + $3.5 = $20.15 ≤ $30.
