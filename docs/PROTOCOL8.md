# PROTOCOL8 — can training rules make a pretrained model's latents reach deeper? (confirmatory)

Written 2026-09-30 09:35 EDT, committed before any PROTOCOL8 run started. Append-only; changes go in dated amendments.

**North-star conditions:** C1 (the thoughts do the work), C2 (more thinking helps beyond training), C3 (competitive
with text CoT at matched compute), C4 (only partly: Coconut's curriculum uses step labels). See `docs/NORTH_STAR.md`.

## What is already known (provenance)

- **Phase 0** (`docs/PROTOCOL8_PHASE0_RESULTS.md`; registered 3a9c05e, amendment 1 ecbec6f):
  - A GPT-2 trained without thoughts learns this deep-ProsQA format only weakly (a₁ 0.53 on m 0–6).
  - Standard Coconut with the addendum-G answer mask needs its thoughts: K′ = 0 is at chance. But it learns only
    shallow depths: at K′ = 10, 0.80 at answer depth 3–4, 0.55 at 8, chance from 10. Gains saturate by K′ ≈ 3.
  - F1 and F1b were not met: registered outcome (iii).
  - So the primary question is whether anytime or random-tight training makes the deep **trained** depths learnable.
- **Theory account** (THEORY §§17, 25, 35, 40; Pilots D, E, F, K; PROTOCOL9): slack teaches a fractional-speed brake
  and a fall-back keyed to an iteration count carried in the thoughts.
  - Anytime removes the in-range brake (and made toy models more learnable), but not the fall-back past the trained
    depth.
  - Random-tight is only "consistent with" anytime on the toy (4 competent runs, 2 still braked).
  - Jitter breaks learnability: not used.
- **Literature** (`docs/LIT_REQUESTS_2026-09-29.md`):
  - A depth-masked per-step answer target has no prior; the nearest are CLRS hints, RELAY, SIM-CoT and PonderNet.
  - The random-budget arm must be independent of the example, with minimum 1 (Bansal 2202.05826, Huginn).
- **Design choices informed by the pilots above**, all made before any PROTOCOL8 data: the depth-keyed contrasts, the
  answer depth 11 stratum, jitter off, C = 0.

## Design

- **Data.** The Phase 0 files, unchanged: `prosqa/p8data/` (md5 078ca274fe train, 92c1dc9ed2 valid, ff5baa26e4 test,
  c0ee4b3ade ctl). Training depth ≤ 10; test m 0–10.
- **Arms** (4 seeds each, seeds 1–4; one Vast host per run; `prosqa/pod_p8c.sh`, `prosqa/p8c_<arm>_s<seed>.yaml` from
  `make_p8c_configs.py`):

  | Arm | Recipe | Evaluation |
  |---|---|---|
  | `coco` | Phase 0 recipe: official Coconut, addendum-G answer mask, one latent stage per hop to 10, 4 epochs/stage, 48 epochs | long_eval2, K′ ∈ {0,1,2,3,4,6,8,10,12,14,16}, test + ctl |
  | `at` | `coco` + anytime supervision, `anytime: 0`: after latent j, answer loss for examples with depth ≤ j (`prosqa/patch_p8.py`) | as `coco` |
  | `rt` | `coco` + random-tight: at every latent stage s ≥ 1, K ~ U{1..min(s, 10)} per example, independent of it (THEORY §22.7) | as `coco` |
  | `cot` | Text chain of thought (`cot: True`), 30 epochs | `eval_p8.py --mode cot`, 150 per m, with generated-token counts |

- **Code.** `prosqa/patch_p8.py` (sha256 9f439a753008009b…) is verified by `prosqa/test_p8.py`, which every pod runs
  before training and aborts on failure:
  - the anytime branch at j = K reproduces the ordinary logits to 3e-7;
  - with no supervised pairs the loss is identical;
  - random-tight draws per call.

  Pod script `prosqa/pod_p8c.sh` (e1c80050002f93bd…); jitter is off.
- **Cost.** About 18 h per latent run on a 3090-class card, ≈ $45–55 in total. Credit was $86.99 at 09:35.
- **Cut-off.** Runs not finished by Sat 2026-10-03 00:00 count as missing. An arm with fewer than 3 seeds is
  unresolved.

## Outcomes (scorer `prosqa/score_p8.py`, sha256 8a151ebe1e451475…, written before any data)

Seed is the unit. "Exact MW" means the exact one-sided Mann–Whitney test on the 4 vs 4 seed values.

- **Measures:**
  - D710 = accuracy at K′ = 10 on test problems with answer depth 7–10 (deep but inside the trained range).
  - D11 = accuracy at answer depth 11 (first depth past training) at K′ = 16.
  - D12p = accuracy at depths ≥ 12, K′ = 16.
  - LU = acc(m 0–6, K′ = 10) − acc(m 0–6, K′ = 0).
- **P8-1 (primary; C1 in depth, and learnability):** at vs coco, and separately rt vs coco, on D710.
  - **Supported** if exact MW p ≤ 0.05 and mean difference ≥ 0.05.
  - **Not supported** if the mean difference ≤ 0; otherwise inconclusive.
- **P8-2 (secondary; C2):** the same rule on D11. D12p is reported.
- **P8-3 (latent use, per latent arm):** LU ≥ 0.15 in at least 3 of 4 seeds.
- **P8-4 (C3):** the best latent arm is "competitive with text CoT" if its mean D710 ≥ CoT's mean D710 − 0.03.
  Reported with compute per answer: the latent arm's 10 latent steps + answer tokens vs CoT's mean generated tokens +
  answer tokens.
- **P8-5 (mechanistic signature):** K′sat = the smallest K′ whose depth 7–10 accuracy is within 0.02 of K′ = 16.
  "Signature met" if anytime has K′sat ≥ 6 in at least 3 seeds and coco has K′sat ≤ 4 in at least 3 seeds.
- **Learnability:** competent = acc(m 0–6, K′ = 10) ≥ 0.75; counts per arm.

## Predictions (as given by the theory session before the registration commit, 09:35:36)

- **Anytime:** at K′ = 10, anytime > coco on mean depth 7–10 accuracy, with the gain growing with depth; depths 3–4
  unchanged (both at ceiling).
- **Random-tight:** the same direction, with a gain ≤ anytime's and more seed-to-seed spread (lower confidence).
- **Signature:** coco saturates at K′ ≈ 3; under anytime the saturation moves to K′ ≥ 6 for depths ≥ 7.
- **Extrapolation:** no gain beyond d = D_train + c1 (D_train = 10, c1 = 1 from the toy; Phase 0's K′ = 0 at chance
  gives no measurable c1). A gain at d = 11 is possible; none is expected at d ≥ 12.
- **Falsifier:** if anytime does not lift depths 7–10 and the saturation stays at K′ ≈ 3, the limit is per-thought
  capacity rather than a slack-induced brake, and the account does not apply in distribution.

## Follow-up (registered separately)

**PROTOCOL8b:** re-anchoring plus distillation on the `coco` checkpoints (THEORY §32 recipe). Anchor decode by
attention (primary) and by text (secondary; the first C7 round-trip / edit test, with an input-only control).

## Amendment 1 (2026-09-30 09:38 EDT, clarification only, before any PROTOCOL8 run started)

* P8-1 and P8-2 use the **one-sided** exact Mann–Whitney test, as already stated under Outcomes. It matches the
  directional prediction.
* With 4 vs 4 seeds, p = 1/70 = 0.014 at complete separation and 2/70 = 0.029 with one inversion. Two inversions give
  4/70 = 0.057, which fails. So "supported" requires at most one inverted seed pair, plus a mean gain ≥ 0.05.
* No rule changes.

## Amendment 2 (2026-09-30 09:50 EDT, instrument failure in the CoT arm; no PROTOCOL8 data existed)

* **What happened.** The first CoT runs (hosts 53533314/19/22/26) crashed at the first training step. `patch_p8.py`
  made `run.py` pass `sample_idx` to every model, but the CoT arm's plain GPT-2 rejects that keyword. The test gate
  exercised only the Coconut path.
* **Fix.** `sample_idx` is passed only when `configs.coconut`. `test_p8.py` check 6 now asserts the guard and that
  a plain GPT-2 forward accepts the batch. `pod_p8c.sh` now clears `/tmp/coco_t` before the test copy: a stale copy
  from the first attempt had made the gate test old code.
* **Handling.** On each CoT host, Coconut was re-cloned and re-patched and the CoT seed restarted from scratch; the
  failed logs were kept as `prosqa.out.void_attempt*`. All four pass `P8_TEST` and train.
* **Latent arms.** Their `run.py` (Coconut) behaves identically before and after the fix, so they were not
  restarted.

## Amendment 3 (2026-10-01 00:27 EDT, before any PROTOCOL8 result was read): ProsQA shortcut control (pre-specified secondary)

**Why.** THEORY §57 (6c9920e), confirmed independently by Audit, found a shortcut in ProsQA's construction. The
negative option always hangs off another person, never off the root. A person-agnostic proximity heuristic ("the option
farther from every person statement") therefore scores 0.91 on ProsQA and 0.90 on p8_test without any reasoning from
the root. A high D710 could come from this cue. The primary rules P8-1 to P8-5 are **unchanged**.

**Secondary SC (pre-specified now; the user approved, relayed by Audit).**
* **Test set.** `prosqa/p8data/p8_na_test.json` (THEORY §59, c012588; md5 5bb6dda2fa; 4,053 problems). It equalises both
  options' distance to the nearest person statement and their in-degree with answer-preserving statements. The
  proximity heuristic scores 0.500 on it (Audit's own parser); every problem keeps target reachable and negative
  unreachable from the root.
* **Latent arms (coco, at, rt).**
  - Measures: D710_NA = candidate accuracy at K′ = 10, answer depth 7–10; D11_NA = at K′ = 16, answer depth 11.
  - Instrument: the local evaluator `prosqa/gpt2c.py` rollout with the addendum-G answer mask, on the drained
    checkpoints (local cards).
  - The same evaluator also re-scores the original `p8_test.json`, over the same depths and K′, so that both numbers
    come from one instrument.
  - **Cue reliance** = original − NA, per arm and seed.
  - Reported per seed, with arm means and the same exact-MW comparisons as P8-1/P8-2 on the NA measures, labelled secondary.
* **CoT arm.** `eval_p8.py --mode cot --n_per_m 150` on p8_na_test, locally. Original = the pod's cot_test_e30
  (same script, different GPU).
* **Reference line.** The proximity heuristic's accuracy on each file and stratum (D710, D11), from
  `theory/cue_origin.py`'s rule.
* **Reading (fixed now).** An arm whose D710 advantage over coco (P8-1) disappears on NA (difference ≤ 0) gets the
  label "advantage cue-dependent" next to its P8-1 verdict.

Scorer: `prosqa/score_p8_na.py`, written and committed before any NA evaluation runs. The same caveat applies
retroactively to Phase 0, PROTOCOL8b and PROTOCOL10/11 (notes added there).

**Note (2026-10-01 01:02 EDT; interrupted-evaluation re-run, no rule change).** at_s4's host (Vast 53539933) went offline during the
post-training evaluation. Its container rebooted; training had completed, with checkpoint_48 written 2026-10-01 03:32 UTC.
`res/` was empty, so no evaluation output existed. The unchanged pod script was re-run on the same host from
checkpoint_48 (evaluation only, since training is skipped when the final checkpoint exists). The interrupted log is kept
as `prosqa.out.interrupted_eval`. Audit saw two partial budget-sweep lines from it; neither is used.
