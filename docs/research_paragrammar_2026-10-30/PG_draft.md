# PARA-GRAMMAR draft: does decorrelating surface words from ops turn template matching into composition? (task-agnostic steps) (2026-10-30)

**Status:** draft for red team + Literature. Theory's choice; task-agnostic steps are first on the user's GSM8K path.

## 1. Why
- **Note 251-6:** the cued step picks its op from the nearest trained template. Unseen phrasings fail all-or-nothing with systematic substitutions:
  - "exceeds A's by c" computes mul, cf. "multiplied by c";
  - "the u of A less those of C" computes add, cf. "... plus those of C";
  - "For each of C's u, B has as many u as A" computes add, cf. "as many u as A and C together".
- **§250:** adding 4 hand frames (6 → 10) taught only those frames (MEMO). With ~1 template per op per frame, every content word stays near-perfectly predictive of the op ("by" → mul, "as many … as" → add).
- **Hypothesis:** a training distribution where each surface cue appears with several ops removes the shortcut. Examples:
  - "by" appears in add ("exceeds by c"), sub ("falls short by c") and mul ("multiplied by c");
  - "as many as" appears in add_v, mul_v ("as many as A for each of C's") and mul_c ("c times as many as").
  - Only the composition of the words then determines the op.
- **Prediction:** held-out combinations of trained factors generalise. Whether brand-new constructions (frames 6, 7, 12–15) improve is the open part.

## 2. Grammar (to be written in v1; checked like §250's G1–G4)
- **Factors per op:**
  - verb: has / holds / owns / keeps / carries / ends up with / possesses;
  - construction: comparative-than / possessive-count / relative-number / conditional / inverted-roles / by-difference / for-each;
  - argument order;
  - optional filler.
  - Each (op, construction) is rendered with the op's own operator word drawn from a pool shared across ops.
- **Splits:**
  - GT-TRAIN: about 70% of (verb × construction × op) cells.
  - GT-COMBO: held-out cells whose verb and construction each appear in training with other ops / verbs (systematicity).
  - GT-LEX: 2 held-out verbs.
  - Existing held-out constructions: frames 6, 7 and 12–15 (Note 251-6 / §250 sets).
- **Shortcut check:** for every content word in GT-TRAIN, the max over ops of P(op | word) ≤ 0.6, where frames 0–5 sit near 1.0.
- **Solve check:** simulator answers on every rendering; 0 mismatches.

## 3. Stages
- **Stage 0 ($0, local):** lh_fh_s1 / s2 per-cell d 1 / d 2 accuracy on GT-TRAIN / GT-COMBO / GT-LEX. FAILING cells are those with d 2 < 0.80. Stage 1 runs only if ≥ 30% of GT-COMBO cells are FAILING in both models; else VACUOUS.
- **Stage 1:** 8k FT of lh_fh_s1 / s2 (as §250), comparing GRAM (train frames 0–5 + GT-TRAIN) with §250's CTRL / DIV recipe (fd_ctrl / fd_div exist; MEMO).
  - Same items; single-change diff; hold penalty from step 0; LP0 as checklist.

## 4. Readings (binding, per seed)
- **COMBO:** on FAILING GT-COMBO cells, GRAM d 2–4 ≥ 0.80 and ≥ CTRL + 0.20, plus WS ≥ 0.50 at passes 2..d.
- **NEW-CONSTR:** frames 6 / 7 / 12–15 at d 2–4, GRAM ≥ DIV (fd_div) + 0.20 on ≥ 2 FAILING frames. Binding second.
- **NOLOSS** as §250.
- **Labels:**
  - COMPOSE (COMBO + NEW-CONSTR);
  - SYSTEMATIC (COMBO only);
  - MEMO (GT-TRAIN ≥ 0.9, COMBO fails);
  - NO-GAIN;
  - NOLOSS-FAIL.
- **Descriptive:** the substitution table of 251-6 re-read on GRAM (does "exceeds … by c" still compute mul?).

## 5. OC (sketch)
- P(SYSTEMATIC or better | decorrelation works) ≈ 0.6; P(COMPOSE | works) ≈ 0.25.
- P(SYSTEMATIC | nearest-template matching persists) ≈ 0.1.

## 6. Cost
- Stage 0 $0. Stage 1: 2 × 8k (§250 realised $2.82 for 4 runs) ≈ $1.5; guard $2.5.
- Build: the grammar module (appended to FRAMES like §250, existing sets sha-identical), the stage-0 eval (eval_frames generalised to cells).
