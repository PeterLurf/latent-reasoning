# PROTOCOL9 — results

Registration: `docs/PROTOCOL9.md`. Scorer: `scratch/score_p9.py`. Results: `results/p9.json` (primary),
`results/p9_pooled64.json` (amendment-6 pooled replication, after the extension).

| Test | Result | Verdict |
|---|---|---|
| **AP1** anytime beats control at d8 (pairs, pooled over 4 families) | 5 of 10 evaluable pairs (share 0.50, sign p 0.62) | **not supported** |
| **AP2** extrapolation success (PROTOCOL5 rule, d7-based) | anytime-only 10, control-only 3 (sign p 0.046); among pairs where both are competent 1 vs 1 | **supported**, via learnability |
| **AP3** learnability cost | competent anytime 21 / 32 vs control 12 / 32 | **no cost**; anytime is more often competent |
| Pooled 64-pair replication (amendment 6, pre-specified) | AP1 15/26 (0.58, p 0.28); AP2 18 vs 10 (p 0.09); AP3 44 vs 34 competent | **AP1 inconclusive, AP2 inconclusive**: the primary AP2 does not replicate |

## Primary, 32 pairs (scored 2026-09-30 07:45 EDT)

| Family | Pairs | Competent anytime / control | Extrapolate anytime / control | AP1 evaluable (anytime better) |
|---|---|---|---|---|
| 3 layers | 8 | 4 / 2 | 4 / 2 | 2 (0) |
| 4 layers | 8 | 7 / 6 | 4 / 4 | 5 (2) |
| 512 wide | 8 | 5 / 1 | 5 / 0 | 1 (1) |
| fork | 8 | 5 / 3 | 2 / 2 | 2 (2) |

- **Means over competent runs:** d7 is 0.649 (anytime) vs 0.641 (control); d8 is 0.550 vs 0.576; d9 is 0.516 vs 0.544.
  The ceiling rule (amendment 1) did not trigger in any family, so d8 was the measure throughout.
- **Reading:**
  - Anytime makes the task learnable more often, and more models pass the d7 extrapolation rule.
  - It does not push accuracy deeper: d8 does not improve pair by pair.
  - This matches THEORY Pilot D (gain at d7, d8–10 barely move) and §35 (a learned fall-back keyed to an iteration
    count).
- **Where AP2 comes from: learnability, not reach.** Split by competence (Audit's point, verified here):
  - Among competent runs the extrapolating shares are similar: anytime 15/21 (0.71), control 8/12 (0.67).
  - In the 10 pairs where both runs are competent, anytime-only extrapolators number 1 and control-only 1.
  - So AP2 means "anytime makes more seeds both learn the task and pass the d7 rule". Together with AP1's null, that is
    not "anytime extends the reach of a trained model on these architectures". It is the same pattern as THEORY
    Pilot D.

## Pooled 64-pair replication (amendment 6; scored 2026-09-30 13:31, `scratch/score_p9.py --ext --final`, `results/p9_pooled64.json`)

| Family (16 pairs each) | Competent anytime / control | Extrapolate anytime / control | AP1 evaluable (anytime better) |
|---|---|---|---|
| 3 layers | 10 / 7 | 6 / 7 | 6 (0) |
| 4 layers | 14 / 11 | 8 / 6 | 9 (5) |
| 512 wide | 12 / 6 | 12 / 3 | 6 (6) |
| fork | 8 / 10 | 3 / 5 | 5 (4) |

- **Pooled:** AP1 15/26 (share 0.58, sign p 0.28), inconclusive. AP2 18 vs 10 (p 0.09), inconclusive. AP3: 44 vs 34
  competent, no cost.
- **The extension's 32 pairs alone (descriptive):** AP2 8 vs 7, AP1 10/16, competent 23 vs 22. The primary AP2
  effect (10 vs 3) does not recur in the new seeds.
- **Reading:**
  - With 64 pairs the evidence that anytime raises extrapolation success across architectures is weak (p 0.09).
  - It is concentrated in the 512-wide family (12 vs 3). Fork graphs go the other way (3 vs 5), and 3 layers is even.
  - The mean d7 gain is small (0.651 vs 0.621), and d8 is unchanged.
  - The primary AP2 "supported" should be read as fragile: it does not replicate in the pre-specified pooled
    analysis.

## Mechanism check by the theory session (THEORY §40, measurement only, 33 competent models)

- Anytime removes the in-range brake: median p[K−1] is 5 vs 2 for controls.
- The fall-back remains: only 1/16 anytime runs have a pointer past depth 8.
- The step count is decodable from the thoughts in 25/25 runs.
- Same cause as on the base toy: a fall-back keyed to an iteration count.

## Deviation (scorer fix, disclosed)

The first scoring run left 4 pairs "pending": `l3_s501`, `l3_s504`, `w512_s547` and `fork_s566`, each with a
stage-1 failure on a Vast host. The scorer's no-model test required the run dir locally, but the drain scripts do not
fetch model-less dirs. The fix identifies such runs from the queue log's "no model for <r>" line, which implements the
registered rule ("a run without a model is not competent"). It was made after the first output had been seen. All
three decisions (AP1–AP3) are identical before and after the fix; only the pair count moved from 28 to 32.
