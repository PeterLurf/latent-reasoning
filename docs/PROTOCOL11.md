# PROTOCOL11 — bidirectional latent search in Coconut GPT-2 on ProsQA (zero-shot, inference only)

Registered 2026-09-30 21:05 EDT, before any PROTOCOL11 output on the test models exists. Design: Theory, THEORY §52 F2 ("first non-toy
test"; the toy version is §53, running). Implementation and registration: Experiments (role split of 2026-09-30).
North-star tags: **C2** (reach without new training), **C7** (the meeting node is a checkable certificate), **C5** (meeting
as a natural halt; not tested here).

## Procedure (`prosqa/bidir_p11.py`, K′ = 10 per episode)
* **Models.** The 6 PROTOCOL8b step-1 models (primary). The Phase 0 model is descriptive only: it was used for the smoke test.
* **Data.** `p8_test.json`, answer depths 3–14, the first 100 per depth (all at depth 14); the 100 calibration problems excluded.
* **Forward episode.** The problem as given.
* **Backward episode, per answer option O.** Every statement reversed ("Every A is a B." → "Every B is a A.";
  "Name is a B." → "Every B is a Name."), plus "<Z> is a O." for a fresh person Z and the question
  "Is <Z> a <root> or <another person>?".
* **Frontier of an episode.** The objects of the m statements with the most sweep-head attention, summed over the
  thoughts. The sweep head is calibrated as in `reanchor_p8.py` on forward problems. m = 10 is primary; m = 5 and 15 are descriptive.
* **Bidirectional answer.** If the forward frontier meets the backward frontier of exactly one option, that option;
  otherwise the forward answer.
* **Compute.** 3 episodes against 1 for forward-only; §52's criterion compares with forward-only.

## Predictions and decision rules (fixed now; m = 10)
Gain(model) = acc(bidir) − acc(forward), at d11–14.
* **P11-1 (primary, §52's success criterion).** Supported if the mean gain over the 6 models is ≥ 0.10 and the gain is
  > 0 in ≥ 5/6 models. Not supported if the mean gain is ≤ 0. Otherwise inconclusive.
* **P11-2 (precision, the analogue of §53 BD2).** At d11–14, the median over models of the share of problems where the
  negative option's backward frontier meets the forward frontier is ≤ 0.30.
* **P11-3 (no in-range loss).** At d3–6, the median gain is ≥ −0.02.
* **Descriptive.** m = 5 and 15; accuracy by depth; the target's hit rate.

## Disclosure
* **Smoke test.** Phase 0 model, 24 problems (2 per depth), run before this registration, outside the primary set. The
  negative option's frontier often meets the forward frontier at m = 10 (in 23 of 24 problems), less so at m = 5. The
  primary m was set in the code before the smoke and is not changed.
* **Prior from PROTOCOL10 / the hop sweep.** These models answer at chance at d ≥ 11 and misread written statements, so
  the expected outcome is null. This is the cheap test §52 asked for.
