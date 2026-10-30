# FRAME-DIV v1: does phrasing diversity let the latent pass generalise to unseen phrasings? (2026-10-30)

**Status:** v1 = FD_draft.md (162b027d) with the red team's findings (REVIEW_FD.md: GO-with-fixes; blockers 1–2 fixed) and Audit's points. Where they differ, this file wins.
- This file was added minutes after the §250 registration commit (a5d32eaf), whose generating script failed. The THEORY.md §250 text is complete and identical in substance. No build or output exists yet.
- Literature: diversity thresholds are known in general (2306.15063, 2311.04420; Wiedemer 2307.05596: support must cover the components); template sensitivity in LLMs (GSM-Symbolic 2410.05229); paraphrase transfer comes from lexical exposure (Kim, Linzen & Smolensky 2212.10769). None of this is reported for latent reasoning.
- NARROW wording: "template held-out transfer of the latent loop's per-pass step computation vs training frame count (6 → 10)".

## 1. Templates (appended to nested_words.FRAMES at indices 8–15; FRAME_ALL = range(8) unchanged)
**DIV-training frames 8–11** (red-team wording fixes applied):

| op | 8 | 9 | 10 | 11 |
|---|---|---|---|---|
| add_c | {B} ends up with {c} {u} more than {A}. | With {c} more {u}, {A} would have as many as {B}. | {B} possesses {A}'s number of {u} increased by {c}. | There are {c} more {u} with {B} than with {A}. |
| sub_c | {B} ends up with {c} {u} fewer than {A}. | With {c} fewer {u}, {A} would have as many as {B}. | {B} possesses {A}'s number of {u} decreased by {c}. | There are {c} fewer {u} with {B} than with {A}. |
| mul_c | {B} ends up with {c} times {A}'s {u}. | Scale {A}'s {u} by {c} and you have {B}'s {u}. | {B} possesses {A}'s number of {u} multiplied by {c}. | There are {c} times as many {u} with {B} as with {A}. |
| add_v | {B} ends up with {A}'s {u} and {C}'s {u} put together. | Pool the {u} of {A} and {C} and you have as many as {B}. | {B} possesses the combined number of {u} of {A} and {C}. | There are as many {u} with {B} as with {A} and {C} jointly. |
| sub_v | {B} ends up with {A}'s {u} once {C}'s {u} are subtracted from them. | Subtract {C}'s {u} from {A}'s and you have as many as {B}. | {B} possesses the number of {u} left when {C}'s are taken from {A}'s. | There are as many {u} with {B} as {A} has in excess of {C}. |
| mul_v | {B} ends up with {A}'s {u} times {C}'s {u}. | Multiply the {u} of {A} and of {C} and you have as many as {B}. | {B} possesses the product of the {u} counts of {A} and {C}. | There are as many {u} with {B} as {A}'s {u} multiplied by {C}'s. |

**Held-out frames 12–15** (binding). 12, 14 and 15 are new constructions (pseudo-cleft "What … has is"; passive "is reached by"; "In {u}, … stands"); 13 is the red team's accepted novel frame.

| op | 12 | 13 | 14 | 15 |
|---|---|---|---|---|
| add_c | What {B} has is {c} {u} past {A}'s tally. | {A} trails {B} by {c} {u}. | {B}'s count is reached by adding {c} {u} to {A}'s. | In {u}, {B} stands {c} above {A}. |
| sub_c | What {B} has is {c} {u} short of {A}'s tally. | {B} trails {A} by {c} {u}. | {B}'s count is reached by taking {c} {u} off {A}'s. | In {u}, {B} stands {c} below {A}. |
| mul_c | What {B} has is {A}'s tally of {u}, {c} times over. | {B}'s pile of {u} is {c}-fold {A}'s. | {B}'s count is reached by counting {A}'s {u} {c} times. | In {u}, {B} stands at {c} times {A}. |
| add_v | What {B} has is {A}'s tally of {u} joined with {C}'s. | {A} and {C} between them match {B}'s {u}. | {B}'s count is reached by counting {A}'s {u} and then {C}'s. | In {u}, {B} stands at the total of {A} with {C}. |
| sub_v | What {B} has is {A}'s tally of {u} once {C}'s is removed. | {A} outnumbers {C} in {u} by exactly {B}'s count. | {B}'s count is reached by taking {C}'s {u} off {A}'s. | In {u}, {B} stands at {A} reduced by {C}. |
| mul_v | What {B} has is {A}'s tally of {u}, once for each of {C}'s {u}. | {A}'s {u}, repeated as often as {C} has {u}, give {B}'s {u}. | {B}'s count is reached by counting {A}'s {u} once per {u} of {C}. | In {u}, {B} stands at {A} scaled by {C}. |

- add_s / mul_s reuse add_v / mul_v (the alias).
- **Novelty gate (pre-checked, fd_v1_check.py):** max slot-masked trigram Jaccard of every held-out template vs the same op's frames 0–11 is **0.16** (gate ≤ 0.20).
- **Solve check:** 0 / 5,400 mismatches of OW.solve on extended-frame items (frames 8–11, 12–15, 0–7; laws 1 / 2).
- **Mod-11 caveat:** "trails / outnumbers / above / below / short of / past / in excess of" are literally false when the subtraction wraps, as in existing frames. The wrap share and the no-wrap subset are reported (descriptive).

## 2. Build gates (red team 4)
- **G1:** every existing set sha is byte-identical before / after: build_sets(0 / 1 / 2) incl. transfer 195f6756, test2, pairs2, floors2, s226, s227; NW / OW test_set and pair sets.
- **G2:** selfcheck on frames 0–15.
- **G3:** the new frame-group test sets carry shas, and their sigs go into EXCL.
- **G4:** the nested_words sha change is pre-declared in ARGS-DIFF, and the edit waits until no running job or in-flight deploy imports nested_words.
- **New arg:** inject_aw --train_frames (default = FRAME_TRAIN, bitwise), since inject_aw.py:223 hard-codes FRAME_TRAIN.

## 3. Stage 0 ($0, local): step-0 baseline (red team 2) — decides whether Stage 1 runs
- lh_fh_s1 / s2 at step 0 on per-frame test sets (1,000 items per d 1–4 for each of frames 8–15; DF law). acc at K′ 8 and WS.
- **FAILING frame** = a held-out frame (12–15) with step-0 d 2 acc < 0.80 in that model.
- **Rule:** Stage 1 runs only if ≥ 2 held-out frames are FAILING in both models; GEN then binds on those frames. Otherwise VACUOUS: the held-out frames are already handled, like frame 7, and the held-out set is re-authored before any spend.
- Also the lexical check (Literature): content words in frames 6 / 7 not present in frames 0–5, beside each frame's d 1 acc (descriptive).

## 4. Stage 1 (≈ $3.2; guard $5): fine-tunes of lh_fh_s1 / s2, 8k, §240 recipe
- **Arms:**
  - DIV: --train_frames 0–5, 8–11;
  - CTRL: frames 0–5.
  - Same seed / items: the 22930 stream gives identical programs (3,000 / 3,000), and only the frames differ. Both replay §240's items (disclosed). Single-change args diff {train_frames}.
- The hold penalty runs from step 0 (no re-ramp). LP0 per the warm-start checklist (skip ≤ 1,000 after the fade re-ramp; cap = max(2 × CTRL windows after 1k, 10)).
- **Binding, per seed (PASS = both):**
  - **GEN:** on each FAILING held-out frame, DIV d 2–4 acc ≥ 0.80 and ≥ CTRL + 0.20, and WS ≥ 0.50 at every pass 2..d (d 3–4), on ≥ 2 FAILING frames.
  - **NOLOSS:** TEST2-DF acc ≥ 0.98 at d 1–4; §238 gates; HOLD-LONG3 at K′ 64 / 128; retention drops > 0.05 named.
  - **Labels:** GEN-PASS / MEMO (TRAIN-NEW frames 8–11 test sets ≥ 0.90 but GEN fails) / NO-GAIN / NOLOSS-FAIL / LP0-FAIL.
- **Descriptive:** frames 6 / 7 for both arms; per-frame tables; the wrap / no-wrap split.
- **OC (2 seeds):** P(GEN | H-DIV) ≈ 0.6 per seed; P(GEN | H-MEMO) ≈ 0.05. Prior 0.4, so P(PASS, both) ≈ 0.15.
- **Censoring:** as §240. The Stage-1 scorer is written before any Stage-1 output.

## 5. Cost
Stage 0 $0. Stage 1 ≈ $3.2, guard $5 (a new design family; credit ~$34 → ~$29).
