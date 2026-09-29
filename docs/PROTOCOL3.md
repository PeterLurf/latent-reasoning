# Protocol 3 — architecture ablations of the proposed mechanism

Written 2026-09-24 **before** any of these models was trained. Each variant changes one element
that the Part II account depends on. All variants use the seed-0 recipe (GPT-2 init, PARENT +
CHILDREN scaffold annealed to 0 by 15k, gated depth curriculum, 30k steps, batch 512, K = 6)
and are analysed with the frozen Part II pipeline on fresh graphs (seed 271828, same exclusion
list), 300 graphs per width, W ∈ {2, 4, 8}, at the variant's own frozen site k* (Part I rule).

| Variant | Change | Tests |
|---|---|---|
| V1 wide | d = 512 (d_id = 64) | capacity / superposition interference |
| V2 deep | 4 layers | capacity of the prefix / query-time pathway |
| V3 raw feedback | recurrent input = pre-final-LayerNorm residual | LayerNorm renormalisation of the thought |

Reference: seed-0 main model (s0) and the three-seed range where available.

## Predictions

* **P11 (V1, normalisation vs capacity).** If the width effect is set by a normalised relative
  code rather than by superposition capacity, V1 shows the same qualitative pattern: targeted
  effect (full, remove, dose 1) at W = 8 is ≤ 0.5 × its W = 2 value, and the per-branch attention
  excess still falls at least as fast as 1/√W. A capacity account predicts a markedly weaker
  decline (W8/W2 > 0.5) and lower collateral than s0.
* **P12 (V2, query pathway).** With 4 layers the prefix can compute more hops, so the
  latent/query boundary moves outward: at W = 4 the query-time graph share (Gq/clean, all nodes)
  is higher than the three-seed recurrent range's minimum, and 3-hop nodes are mostly
  query-resolved (Gq share at distance 3 > 0.5).
* **P13 (V3, renormalisation).** Removing the final LayerNorm from the recurrent input changes the
  amplitude law of the thought: the fitted exponent β of |h_don − h_rec| vs W differs from the
  three-seed recurrent range [0.62, 0.87] (predicted direction: closer to 1, since per-branch
  shares ~1/W are no longer renormalised). Relative competition (P6-style reachable-boost
  lowering others) persists, because it arises in the softmax, not the final LayerNorm.
* **Competence gate.** A variant that fails the 85% delayed balanced-accuracy bar is reported as a
  training failure; mechanism predictions are then not scored.
* **Stage-1 restart rule** (written before any Part III model other than V3 was evaluated).
  Some runs stall at curriculum stage 1 (delayed loss plateau ≈ 0.565, upfront loss at ln 2; seen
  so far only under bf16: main_s3 on the 4060, V1 attempts 2–3 on the 3070). A run that has not
  passed the stage-1 gate by step 4 000 (successful runs pass at 1 500–2 000) is declared a
  training failure, kept on disk, and relaunched with the identical command (training is
  nondeterministic, so a relaunch is a new draw). At most three relaunches per model; every
  attempt is listed in the report with its outcome. This conditions on competence, as the gate
  already does; no mechanism measurement of a failed attempt is taken.
* **Scoring note for P11's attention clause** (written while V1 was still training, never evaluated):
  "per-branch attention excess" = the Part I figure's quantity — layer-1 attention from latent
  positions 1–3 onto a reachable branch's edges minus a decoy branch's (latent/mechanism.py,
  discovery graphs); "falls at least as fast as 1/√W" = excess(W = 8) / excess(W = 2) ≤ √(2/8) = 0.5.
  Reference seeds: 0.20, 0.20, 0.22. P11 passes only if both clauses hold.

## Addendum B (written after discovery of the amplification test, before its fresh run)

* **P14 (within-state relative code).** Rescaling a reachable branch's own pattern component of the
  saved thought at k*, h' = h + (λ−1)UUᵀh, on fresh graphs at W = 4 and 8, in every recurrent seed:
  erasing (λ = 0) lowers that branch's P(reachable) on nodes ≥ 3 hops (CI < 0); doubling (λ = 2)
  raises it (CI > 0); a random rank-6 subspace moves reachable branches by < 1/3 of the λ = 0 own
  effect. Competition inside the thought (doubling one branch lowers the other reachable branches,
  CI < 0) is predicted in ≥ 2 of 3 seeds at W = 4 (discovery: s0 and s2 yes, s1 no).

## Addendum C (written while runs/main_s3 was still training, before any evaluation of it)

* **Held-out-seed replication.** runs/main_s3 (seed 3, same recipe) was started after P1–P10 and
  P14 were frozen. It is evaluated with the identical frozen Part II pipeline (its own k* by the
  Part I rule; fresh graphs). Prediction: each of P1, P2, P3, P5, P7, P9, P14 (the predictions that
  held for every model so far) also holds for seed 3; P4, P6, P8, P10 (mixed so far) are reported
  without a directional expectation. Competence gate as above.
* Note (added 2026-09-24, after the crash): the first main_s3 run died at step 19 200 in a host crash
  (bugcheck 0x1A) and was restarted from step 0 with the identical command; the crashed run was
  never evaluated beyond its own training-time monitoring.
* Note (added later the same night): the first restart was moved to the RTX 4060, where training runs in bf16. It stalled
  at curriculum stage 1 (delayed loss flat at ≈ 0.57 from step 2 000 to 8 400; the fp32 original
  had advanced at step 1 500). Kept as runs/main_s3_bf16_stuck and reported as a training
  failure of that device/precision. main_s3 was restarted a second time on its original device
  (RTX 2070, fp32) with the identical command, as the faithful continuation of the addendum-C run.
  dev_s4 (addendum D) had reached step ~2 000, unevaluated except one step-1000 development probe;
  it is restarted from step 0 (runs/dev_s4_moved kept) on the RTX 2070 (fp32) once main_s3 and its
  evaluation battery have finished there (scratch/chain.py); V1 attempt 3 had meanwhile stalled at
  stage 1 and took the 3070 back.

* Note (added after the second host crash, 07:18): main_s3 and V1 had completed training; V2 was
  lost at step 23 000 (checkpoint truncated by the crash) and restarted from step 0 with the
  identical command. Runs launched from here on keep a full training state and resume after a
  crash (`--resume`), which changes nothing but the (already nondeterministic) trajectory.

## Addendum D (written 2026-09-24, before runs/dev_s4 started training)

* **Second held-out seed + developmental checkpoints.** runs/dev_s4 = seed 4, identical recipe,
  plus `--save_every 1000` (weights also saved every 1000 steps and at each curriculum advance;
  saving consumes no RNG and does not change training). Final checkpoint: same prediction and
  scoring as addendum C (P1, P2, P3, P5, P7, P9, P14 expected to hold; P4, P6, P8, P10 reported
  without direction). Competence gate as above.
* **Developmental analysis (exploratory, discovery graphs only).** For each saved checkpoint:
  depth-matched competence and the cache-route decomposition (T, TL, Gq shares by distance) at
  W ∈ {2, 4, 8} with k = 1. Descriptive question: does the query-time route for shallow nodes
  appear before the latent route for deep nodes, and does the width-dependent boundary exist
  from the first stage at which deep nodes are solved? No threshold is pre-specified.

## Addendum E (written 2026-09-24, after the exploratory length test on s0, s1, s2, V3, pause_s0,
## k0_s2 and before any length or budget test of main_s3, dev_s4, V1 or V2)

* **Tests.** (i) `latent/budget_sweep.py` (discovery graphs; depth-matched accuracy at K' = 0..10,
  step embedding of the last trained step reused beyond K). (ii) `latent/length_gen.py` (all-chain
  graphs; a decoy chain re-hung from the leaf of a reachable chain puts nodes at distance 7..12,
  matched against a decoy chain re-hung from a decoy leaf; graph-level bootstrap).
* **Definition.** A model *extrapolates* if balanced accuracy at distance 7 with K' = 12 is ≥ 0.60
  with bootstrap lower bound > 0.55. So far: s2 yes (0.71); s0, s1, V3, pause_s0, k0_s2 no.
* **P15 (link between in-range and out-of-range test-time gains).** A model extrapolates if and
  only if its depth-6 depth-matched accuracy on ordinary graphs rises by ≥ 0.05 from K' = 6 to
  K' = 8 in the budget sweep. Scored on every newly evaluated recurrent model (main_s3, dev_s4,
  V1, V2); passes if the two classifications agree for all of them.
* The fraction of seeds that extrapolate is reported without a directional expectation.

## Addendum F (written 2026-09-24, before any trace, length or budget test of main_s3, dev_s4, V1, V2)

* **Observation (exploratory, s0, s1, s2, V3; latent/upfront_trace.py --delayed).** In the delayed
  format, layer-1 attention of latent step k concentrates on reachable-branch edges of one
  within-branch depth (decoy edges stay at baseline), and that depth advances with k: s1 one depth
  per step, finishing depth 5 at step 5; s0 the same, one step later; s2 one depth per two steps
  (depth 3 at step 6); V3 slower still. Hypothesis: a search that finishes the trained depth
  inside the trained budget is followed by a learned stopping state (the s0 rewind result), while
  a search still running at step K stays open-ended.
* **Definition.** Frontier depth at step k = the depth j (0–5) with the largest per-edge layer-1
  attention on reachable-branch edges, provided it is ≥ 1.5 × the mean per-edge attention on decoy
  edges at step k (W = 4, 400 discovery graphs); otherwise undefined. E = the largest frontier depth
  reached at any step k ≤ 6. So far: s0 5, s1 5, s2 3, V3 2.
* **P16 (necessary condition).** No newly evaluated recurrent model (main_s3, dev_s4, V1, V2) with
  E ≥ 4 extrapolates (addendum E definition). Models with E ≤ 3 are reported without expectation.

## Addendum G (written 2026-09-24, before either model below was trained)

* **Training-budget manipulation (exploratory, one run each).** The addendum-F hypothesis says the
  stop is learned because the search finishes inside the budget. Two runs of the seed-0 recipe
  change only K: runs/budget_k4_s0 (K = 4) and runs/budget_k10_s0 (K = 10); stage-1 restart rule
  as above; competence gate as above (at their own K).
* **Expectations.** With K = 10 a one-depth-per-step frontier finishes the trained depth by step 5–6,
  leaving four or more steps after it: the model should *not* extrapolate (distance-7 accuracy at
  K' = 16 below 0.60), and its frontier should reach depth ≥ 4 by step 6. With K = 4 no frontier of
  one depth per step can finish depth 5 inside the budget; if the model still solves deep nodes it
  must do so either with an open-ended loop (predicted to extrapolate: distance-7 accuracy at
  K' = 12 ≥ 0.60) or with query-time graph reading (then a query-time share at W = 4 above the
  recurrent reference range). Which of the two happens is not predicted; that neither happens
  (competent, no extrapolation, no query-time increase) would count against the hypothesis.

## Addendum H (written after P11 was scored and failed, before runs/v1_wide_s1 was trained)

* **Second wide-thought seed.** P11 failed on V1 (targeted W8/W2 = 0.58, outside the four d = 256
  seeds' 0.18–0.37), which could be the architecture or the single run. runs/v1_wide_s1 = V1
  recipe with seed 1, same battery. Reading fixed in advance: if its targeted W8/W2 is again > 0.5
  (and it passes the competence gate), the weaker width decline is attributed to the wider thought;
  if it falls inside 0.18–0.37, to V1's run. Anything in between is reported as unresolved.

## Addendum I (written after seed 4 failed the competence gate and budget_k4_s0 was scored, before either run below was trained)

* **Third held-out seed.** Seed 4 failed training, so the held-out replication rests on seed 3
  alone. runs/main_s5 = seed 5, main recipe (RTX 4060, bf16, stage-1 rule), scored exactly as
  addendum C: P1, P2, P3, P5, P7, P9, P14 expected to hold; the rest reported without direction.
* **Second K = 4 seed.** budget_k4_s1 = the K = 4 recipe with seed 1 (RTX 2070, fp32). Expectation
  from addendum G, now directional after the first K = 4 run: if competent, it extrapolates
  (distance-7 accuracy at K' = 12 ≥ 0.60, lower bound > 0.55) or its query-time share at W = 4
  exceeds the reference range; neither would count against the budget hypothesis.

## Addendum J (written after addendum I was scored, before either run below was trained)

* **Gate for these runs.** Overall delayed balanced accuracy ≥ 0.85 *and* depth-matched accuracy at
  5 hops ≥ 0.60. The second clause was chosen after seeing that seed 5 (0.584) passed the overall
  gate without using its latent loop; every model that uses the loop scored ≥ 0.60 (V3 0.603,
  K = 4 0.62, s0 0.67, …). It applies to runs from this addendum on, not retroactively.
* **budget_k4_s2** (K = 4, seed 2, RTX 4060 bf16). Expectation as addendum I: if it passes the gate,
  it extrapolates (distance 7 at K' = 12 ≥ 0.60, lower bound > 0.55) or its query-time share at W = 4
  exceeds the reference range.
* **budget_k8_s0** (K = 8, seed 0, RTX 4060 bf16, after the K = 4 run). The long-budget side of the
  addendum-G hypothesis, with a budget that should train more easily than K = 10: if it passes the
  gate, its layer-1 sweep reaches depth ≥ 4 within six steps (E ≥ 4) and it does *not* extrapolate
  (distance 7 at K' = 16 < 0.60).
* **P17 (the stop is keyed to the step embedding).** Exploratory result on s0, s1, s3 and V2
  (latent/clock_swap.py, clock_freeze.py): holding the step embedding at step 2 for every later step
  (step_emb[min(k, 2)]) leaves the depth read at each step unchanged but removes the stop, so these
  non-extrapolating models follow chains past six hops (s3 0.87 and V2 0.88 at distance 7). Prediction,
  written before either model was tested this way: for **V1** (already trained, never clock-tested)
  and for **budget_k8_s0** (if it passes the gate and does not extrapolate normally), the frozen-clock
  run (step embedding held at 2, K' = 12) reaches distance-7 balanced accuracy ≥ 0.60 with bootstrap
  lower bound > 0.55. Scored on each model separately.
* Stage-1 restart rule and all other rules as above. Device note: with the RTX 3070 lost, torch now
  numbers the cards cuda:0 = RTX 4060, cuda:1 = RTX 2070.

## Addendum K (written after the clock-swap/freeze/trace results, before the run below was trained)

* **No step embedding.** If the halt is learned from the step index, a model that has no step index
  cannot learn it. runs/nostep_s0 = the main recipe (K = 6, seed 0) with `--no_step_emb` (latent
  steps receive no step embedding; everything else identical), RTX 2070 fp32, stage-1 rule.
* **P18.** If it passes the addendum-J gate (overall ≥ 0.85 and depth-matched 5-hop ≥ 0.60), it
  extrapolates: distance-7 balanced accuracy at K' = 12 ≥ 0.60 with bootstrap lower bound > 0.55.
  (A content-based stop cannot fire on the long chains, whose frontier never runs out.) If it fails
  the gate, P18 is unscored and the failure itself is reported.

(Clock times were removed from these notes on 2026-09-24 ~06:00 because several had been estimated
rather than read from the clock; the order of the notes and of the runs is what the file vouches for,
and file modification times plus the research log give the actual sequence.)
