# DAG-FREE v1 (2026-10-27)

**Status:** v1 = DF_draft.md (3ff7ac55) with red-team findings 1–8 (REVIEW_DF.md); where they differ, this file wins. Theory's choice under the user's standing rules; Audit approves.
- Replicate wording (finding 8): both inits descend from aw_plaw2_s1, so this is "one lineage, 2 data orders", not 2 independent seeds.

## 1. Law (findings 5, 6)
- **mul_s_p = 2 ("DF")**; the values 0 / 1 are unchanged, bitwise.
- Per step t ≥ 3 the generator draws mul_s with probability q_t, else a regular OPS_REG op as REG. Partner j ~ U over valid partners in 1..t−2; if none is valid, fall back to a regular op.
  - **q_t = 0.179 (t 3), 0.151 (t 4), 0.147 (t ≥ 5).** These are the red team's per-t values for a realised P(mul_s | t) ≈ 0.125 after fallbacks.
  - Steps 1–2 are REG (no partner can exist).
  - Realised P(mul_s | t, d) is logged from 50k draws; the build gates it at 0.125 ± 0.01 for t 3–6.
- **Position → op information:** 0 at t ≤ 2 by construction. The build reports I(op; t, d) on t ≥ 3 from the actual stream (expected < 0.001 bits). The claim is scoped to t ≥ 3.
- **Text:** a mul_s step names its partner (nested_words.py:117). Without S_j the product is at chance (~1/9). No value leak (the product is never 0 mod 11).
- **Build sites that hardcode 0/1 and must take 2:**
  - argparse choices (inject_aw.py:135);
  - `int(mul_s_p)` (ow_words.py:144);
  - the build_sets assert (inject_ow.py:129);
  - `reg_kw` (inject_aw_eval.py:202): it must raise on unknown values, not fall back;
  - score_df regenerates the sets.
  - Every stream (training, calibration, memcheck, all eval sets) uses DF with new seeds, and a law tag goes into each set sha.

## 2. Sets (findings 1, 2)
- **TEST2-DF:** d 1..12 × 1,000, new seed SEED_TEST2_DF.
- **OPCLASS-DF (new):** d 4 × 3,000, new seed SEED_OPCLASS_DF, for pass-4 class sizes. The eval part `opclass` writes eval_opclass.json {items: {sig, pred_by_pass, v}}, in the ws item format.
- **PROD-SWAP (new):** d 3 programs from a stream conditioned on step 3 = mul_s (new seed). The recipient's memory after pass 2 is replaced by the donor's, then pass 3 runs on the recipient's cue.
  - Keep a pair iff own ≠ dfin and cb ∉ {own, dfin} (cb = donor S_2 × recipient S_1).
  - Cap 1,000 kept pairs. The red team's simulation kept 1,439 of 2,000 tried.
  - Floor = MEAN state at pass 2.
  - The part `swap_prod` writes eval_swap_prod.json {n_pairs, tried, follow_dfin, follow_cb, floor, stay_own, follow_U = follow_dfin + follow_cb}.
  - Reading: follow_dfin means both factors came from memory; follow_cb means S_2 came from memory and the partner S_1 was recomputed from the text.
- **Build gate (finding 1):** every op class has n ≥ 50 at pass 3 (TEST2-DF d 3–4 pooled) and at pass 4 (OPCLASS-DF; mul_s split by j = 1 / 2). This is checked on the built sets before launch, not left as an outcome.
- 0 sig overlap between training and every eval set; set shas in args.json.

## 3. Training
- LAT seed 1 from rcb_plaw4_s1, seed 2 from rcb_plaw4_s2; CD-AN from rcb_pcd4_s1.
- DF law, d ~ U{1..4}, 16k, γ = 1. Calibration skipped and fade restart as §237 (237-2 (a)).
- Args diff vs rcb_plaw4_s1 = {mul_s_p 2, init_model} + seed + name.

## 4. Derivation (finding 4)
- §237 showed a pass reading two computed states and adding them (add_s j ≥ 1: OP-ALL 1.000).
- **H-EXT:** the same read supports a product (mul_s ≥ 0.8 at passes 3 and 4).
- **H-READ2:** a superposed read can deliver a sum but not a product (mul_s ≤ 0.3 while the other classes stay ≈ 1.0).
- At pass 3 the partner is always S_1 (j = 1). The model could recompute S_1 from the text or use a fixed "read S_{t−2}" rule. Pass 4 split by j (1 / 2) and PROD-SWAP's follow_dfin / follow_cb split tell these apart.

## 5. Readings (finding 3: the init already solves the non-product items, so the binding gates are aimed at products)
- **Per LAT seed, binding:**
  - LP0 (seed + CD-AN);
  - **PROD-ACC:** final accuracy on TEST2-DF d 3–4 items with ≥ 1 mul_s ≥ 0.80, and ≥ CD-AN + 0.25 on the same items;
  - **CHAIN:** WS ≥ 0.50 at every pass 2..d, d 3–4 (all items);
  - **OP-ALL+:** every op class ≥ 0.80 at pass 3 (TEST2-DF d 3–4) and at pass 4 (OPCLASS-DF), with mul_s at pass 4 counted separately for j = 1 and j = 2;
  - **PROD-SWAP:** follow_U ≥ 0.50 and ≥ floor + 0.30;
  - **CUE-ONLY guard** on d 3–4. The bar is max(majority, min over s228_pdir2c_s1_df at bs 100 / bs 50) + 0.05.
- **Step 0:** both LAT inits and rcb_pcd4_s1 are evaluated on all DF parts (eval-only *_df refs). Pre-stated: if a step-0 init already has mul_s ≥ 0.80 at pass 3 and 4 and passes PROD-SWAP, that seed's PASS is labelled ZERO-SHOT (product chaining present without DF training; not credited to §238 training).
- **Seed labels, first match:**
  1. VOID (guard, sig mismatch, or a class n < 50 despite the gate);
  2. PASS (ZERO-SHOT sub-label if applicable);
  3. READ2 (mul_s < 0.80 at pass 3 or 4 while every other class ≥ 0.80; sub-band "partial" if min mul_s ≥ 0.55; the failing pass is named);
  4. CAP (a non-mul_s class < 0.80);
  5. SWAP-FAIL (OP-ALL+ and PROD-ACC pass, PROD-SWAP fails);
  6. MERGED (pass t prints v_{t+1} ≥ 0.50);
  7. PASS4-LAG;
  8. GATE-OTHER.
- **Overall:** PASS (both) / SPLIT / FAIL / VOID (both VOID).
- **Descriptive:**
  - full-set ACC d 2–4 and d 5–6;
  - SWAP-MID d 3 t 1 / t 2 and d 4 t 2 (all items, as before);
  - PARTNER-MEM vs PARTNER-TEXT from follow_dfin vs follow_cb;
  - mul_s by j;
  - step-0 deltas on the product subsets.

## 6. OC (finding 7)
- P(VOID) per seed ≈ 0.05 (the class-size gate moves the n risk to the build).
- Per seed P(PASS | H-EXT) ≈ 0.65; P(PASS | H-READ2) ≈ 0.02. Prior H-EXT 0.5.
- Per seed P(PASS) ≈ 0.31; overall P(PASS) ≈ 0.15, P(≥ SPLIT) ≈ 0.45.

## 7. Censoring
As §236: a crash means resume (dedupe); missing parts mean pending, FAIL at --final; a missing reference means pending.

## 8. Cost
- Three runs ≈ $2.5 plus local refs (six eval-only DF refs: two LAT inits, CD init, DIRECT at bs 100 and bs 50).
- Admit ≤ $2.6, guard $7. Running: $8.76 + $7 = $15.8 ≤ $30.
