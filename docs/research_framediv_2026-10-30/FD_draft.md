# FRAME-DIV draft: does phrasing diversity let the latent pass generalise to unseen phrasings? (2026-10-30)

**Status:** draft for red team + Literature. Theory's choice under the user's standing rules, on Audit's suggestion after §249 (and not a fix loop: the binding test is on NEW held-out frames, not frame 6).

## 1. Why
- §249: the §240 loop's pass is frame-specific. On unseen frames it fully handles frame 7 (d 1 1.000) and fails frame 6 (d 1 0.064), while the one-pass CD-AN parses both (0.971). Phrasing diversity is the next binding limit on the task-agnostic (GSM8K-path) roadmap.
- **Question:** does training the pass on more phrasings make it generalise to phrasings it has never seen (H-DIV)? Or does it only learn the added ones (H-MEMO)?

## 2. New frame family (FRAME_EXT, indices 8–15; FRAME_ALL = range(8) untouched)
Templates (B is the step's result; A is the previous quantity; C is the second operand or partner state; c is a constant):

| op | 8 | 9 | 10 | 11 |
|---|---|---|---|---|
| add_c | {B} ends up with {c} {u} more than {A}. | Give {A} {c} more {u} and you have as many as {B}. | {B} possesses {A}'s number of {u} increased by {c}. | There are {c} more {u} with {B} than with {A}. |
| sub_c | {B} ends up with {c} {u} fewer than {A}. | Remove {c} {u} from {A} and you have as many as {B}. | {B} possesses {A}'s number of {u} decreased by {c}. | There are {c} fewer {u} with {B} than with {A}. |
| mul_c | {B} ends up with {c} times {A}'s {u}. | Scale {A}'s {u} by {c} and you have {B}'s {u}. | {B} possesses {A}'s number of {u} multiplied by {c}. | There are {c} times as many {u} with {B} as with {A}. |
| add_v | {B} ends up with {A}'s {u} and {C}'s {u} put together. | Pool the {u} of {A} and {C} and you have as many as {B}. | {B} possesses the combined number of {u} of {A} and {C}. | There are as many {u} with {B} as with {A} and {C} jointly. |
| sub_v | {B} ends up with {A}'s {u} after {C}'s {u} are subtracted. | Subtract {C}'s {u} from {A}'s and you have as many as {B}. | {B} possesses the number of {u} by which {A} exceeds {C}. | There are as many {u} with {B} as {A} has in excess of {C}. |
| mul_v | {B} ends up with {A}'s {u} times {C}'s {u}. | Multiply the {u} of {A} and of {C} and you have as many as {B}. | {B} possesses the product of the {u} counts of {A} and {C}. | There are as many {u} with {B} as {A}'s {u} multiplied by {C}'s. |

| op | 12 | 13 | 14 | 15 |
|---|---|---|---|---|
| add_c | {B}'s {u} come to {A}'s {u} plus {c}. | {A} trails {B} by {c} {u}. | {B} keeps {c} {u} on top of what {A} keeps. | The {u} belonging to {B} outnumber {A}'s by {c}. |
| sub_c | {B}'s {u} come to {A}'s {u} minus {c}. | {B} trails {A} by {c} {u}. | {B} keeps {c} {u} under what {A} keeps. | The {u} belonging to {A} outnumber {B}'s by {c}. |
| mul_c | {B}'s {u} come to {c} times {A}'s {u}. | {B}'s pile of {u} is {c}-fold {A}'s. | {B} keeps {c} times the {u} that {A} keeps. | The {u} belonging to {B} are {c} times as numerous as {A}'s. |
| add_v | {B}'s {u} come to {A}'s {u} plus {C}'s {u}. | {A} and {C} between them match {B}'s {u}. | {B} keeps as many {u} as {A} and {C} keep in total. | The {u} belonging to {B} sum those of {A} and {C}. |
| sub_v | {B}'s {u} come to {A}'s {u} minus {C}'s {u}. | {A} outnumbers {C} in {u} by exactly {B}'s count. | {B} keeps the {u} that {A} keeps beyond {C}'s. | The {u} belonging to {B} equal {A}'s with {C}'s taken away. |
| mul_v | {B}'s {u} come to {A}'s {u} times {C}'s {u}. | {A}'s {u}, repeated as often as {C} has {u}, give {B}'s {u}. | {B} keeps {A}'s {u} times {C}'s {u}. | The {u} belonging to {B} equal {A}'s times {C}'s. |

- add_s / mul_s reuse the add_v / mul_v templates (as FRAMES does).
- **Build gates:** every existing set sha is unchanged (FRAME_ALL untouched; new indices are reached only when a stream's frames include ≥ 8); each template renders with every op's slots; a fixture renders 10 items per template for a human read.

## 3. Arms (8k fine-tunes of lh_fh_s1 / s2; §240 recipe incl. the hold penalty)
- **DIV:** training frames 0–5 + {8, 9, 10, 11}.
- **CTRL:** training frames 0–5 only (the same 8k), separating "more training" from diversity.
- **Held out:** NEW frames {12, 13, 14, 15}, plus the old unseen frames 6–7.

## 4. Readings (TEST sets built per frame group, d 1–4, 1,000 items; plus TEST2-DF retention)
- **GEN (binding):** on the held-out NEW frames, DIV acc at d 2–4 ≥ 0.80 and ≥ CTRL + 0.20; WS ≥ 0.50 at every pass 2..d at d 3–4.
- **OLD-6 (descriptive):** frame 6 at d 1–4, DIV vs CTRL.
- **NOLOSS:** TEST2-DF (frames 0–5) acc and §238 gates retained; retention drop > 0.05 named.
- **LP0** per the warm-start checklist; norm guard vs CTRL.
- **Labels:** GEN-PASS / MEMO (DIV learns the trained new frames, check d ≥ 0.9, but not the held-out ones) / NO-GAIN / NOLOSS-FAIL.

## 5. OC (sketch)
- P(GEN | H-DIV) ≈ 0.6; P(GEN | H-MEMO) ≈ 0.05. Prior 0.4, so ≈ 0.27 per seed.

## 6. Cost
- Four 8k fine-tunes ≈ $0.8 each ≈ $3.2; guard $5.
- That is a new design family (Audit: the per-design $30 line). Vast credit ~$34 → ~$29.
