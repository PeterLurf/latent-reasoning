# PROTOCOL9 — results

Registration: `docs/PROTOCOL9.md`. Scorer: `scratch/score_p9.py`. Results: `results/p9.json` (primary),
`results/p9_pooled64.json` (amendment-6 pooled replication, after the extension).

| Test | Result | Verdict |
|---|---|---|
| **AP1** anytime beats control at d8 (pairs, pooled over 4 families) | 5 of 10 evaluable pairs (share 0.50, sign p 0.62) | **not supported** |
| **AP2** extrapolation success (PROTOCOL5 rule, d7-based) | anytime-only 10, control-only 3 (sign p 0.046) | **supported** |
| **AP3** learnability cost | competent anytime 21 / 32 vs control 12 / 32 | **no cost**; anytime is more often competent |
| Pooled 64-pair replication (amendment 6) | pending (extension runs until ~11:00) | — |

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
- **Caveat:** AP2 counts successes regardless of competence. Part of the anytime advantage is the higher competence
  rate (AP3), not better extrapolation given competence.

## Deviation (scorer fix, disclosed)

The first scoring run left 4 pairs "pending": `l3_s501`, `l3_s504`, `w512_s547` and `fork_s566`, each with a
stage-1 failure on a Vast host. The scorer's no-model test required the run dir locally, but the drain scripts do not
fetch model-less dirs. The fix identifies such runs from the queue log's "no model for <r>" line, which implements the
registered rule ("a run without a model is not competent"). It was made after the first output had been seen. All
three decisions (AP1–AP3) are identical before and after the fix; only the pair count moved from 28 to 32.
