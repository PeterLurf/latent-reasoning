# Protocol 5 — training reliability, what predicts extrapolation, and the clock under a strict criterion

Written 2026-09-25 after the Part IV report; plan in `docs/RESEARCH_TIMELINE.md` §6 (steps 1–6). PROTOCOL4 training is
closed. Definitions carried over unchanged unless stated: gate = PROTOCOL4 addendum J (overall delayed balanced
accuracy ≥ 0.85 at the run's own K **and** depth-matched 5-hop ≥ 0.60, and — addendum D — the curriculum reached full
depth by 30k); "extrapolates at K′" = distance-7 delayed balanced accuracy ≥ 0.60 with bootstrap lower bound > 0.55.

## Addendum A (written 2026-09-25 ~15:10, before any run below started) — reliability pilot (plan step 4)

* **Question.** Most PROTOCOL4 gate failures were late-stage (curriculum stages 2–4) stalls, which the stage-1 restart
  rule does not cover. The stage-1 pilot found lr 5e-4 removed stage-1 stalls in the no-step recipe (0/12 vs 3/6).
  Does it also remove late-stage stalls, at full length?
* **Runs.** No-step-embedding recipe (`--no_step_emb`, K = 6; otherwise the seed-0 base recipe: batch 512, 30k steps,
  warmup 300, gated curriculum, PARENT+CHILDREN scaffold annealed to 0 by 15k, GPT-2 init), bf16, stage-1 guard as in
  PROTOCOL4. Two arms with paired seeds 20–27 (never used before):
  * `rel_base_s20..s27`: lr 1e-3 (the recipe as scored in P19).
  * `rel_lr5_s20..s27`: `--lr 5e-4`, nothing else changed.
  One run per RunPod pod (community, ≤ $0.34/h, hosts pass the `cuInit` check). Every run gets the addendum-D battery
  (competence, competence_matched, length_gen, delayed trace, budget_sweep) on its pod.
* **Outcome per run.** Reached full curriculum depth by 30k (yes/no); gate (yes/no); step of each curriculum advance;
  attempts (stage-1 relaunches).
* **Prediction R1.** The lr 5e-4 arm has more gate passes than the base arm, and at least 6 of 8 of its runs pass.
  Reported with a one-sided Fisher exact test (arm × gate pass). R1 **holds** if both clauses hold, **fails** if the lr
  5e-4 arm has no more passes than base, otherwise **partial**.
* **Secondary (reported, no prediction).** Among runs that reach full depth, delayed accuracy and matched 5/6-hop by
  arm (a lower LR might trade reliability for accuracy). Stage-1 stall counts by arm (replicates the stage-1 pilot).
* **Use of the runs elsewhere.** The runs are also the fresh test set for the extrapolation predictor of plan step 5.
  That predictor is frozen in addendum B **before** any evaluation output of these runs is fetched or read. Until
  then only training logs (step, stage, loss, accuracy at the training budget) are inspected.

## Addendum B (written 2026-09-25 ~15:45, before any evaluation output of the pilot runs was fetched or read) — extrapolation predictor (plan steps 2, 5)

* **Exploratory basis (plan step 2; `scratch/extrap_predictors.py`, results/extrap_predictors.json).** Over the 22
  J-competent recurrent models of PROTOCOL4 addendum E, 19 features measured inside the trained regime were screened
  against "extrapolates at K′ = 12 or 16". Only one in-budget feature reached AUC ≥ 0.80: **end activity** = layer-1
  per-edge excess attention (reachable − decoy) at the model's last trained latent step K, taking the most-attended
  depth 1–5 at that step, divided by the peak of the same quantity over steps 1..K and depths 1–5 (delayed trace, W = 4, confirm graphs). AUC 0.81 (bootstrap CI
  0.57–0.97; permutation p = 0.017, Holm-corrected over 15 in-budget features 0.26), and 0.84 / 0.82 with the W = 2 /
  W = 8 traces. Pattern: of 9 models with end activity < 0.25, one extrapolates (nostep_k8_s1); of 13 with ≥ 0.25,
  seven do. A sweep still running at step K looks close to necessary and far from sufficient. E, clock rate, slack,
  accuracy and the manipulated factors (K, step embedding) did not separate the groups (AUC 0.50–0.66).
* **P26 (prospective, scored on the competent runs of addendum A).** End activity is measured exactly as above
  (`confirm_sweep.trace`, W = 4, 600 confirm graphs, generator seed 2026 + 4, per-edge excess at depths 1–5, steps
  1..K). Threshold fixed now: **0.25**.
  * (a) *Necessity:* at most one competent run with end activity < 0.25 extrapolates at K′ = 12.
  * (b) *Direction:* the extrapolation rate among runs with end activity ≥ 0.25 is higher than among runs < 0.25.
  * (c) If both groups have ≥ 2 runs and both outcomes occur: AUC of end activity for extrapolation ≥ 0.70.
  * Reading: **confirmed** if (a), (b) and (c, where computable) hold; **fails** if (a) fails or (b) is reversed;
    otherwise **unresolved** (e.g. all runs on one side of the threshold, or no run extrapolates).
* The pilot runs have K = 6, so "extrapolates" uses K′ = 12 only (PROTOCOL4 P19 convention).

## Addendum C (written 2026-09-25 ~15:45, before any code or run below) — the clock under a strict criterion (plan step 3)

* **Why.** PROTOCOL4 P23/P24 failed (9/22) because the qualifying criterion (argmax ≥ 1.5× decoy) admitted models
  that read a depth over several latent steps. This addendum re-tests the step-locked clock only on models that read
  each depth in one step, with the criterion fixed in advance and measured on graphs never used for scoring.
* **Disclosure.** The threshold below was chosen after looking at the addendum-E confirm traces (already scored):
  there, a 0.35 concentration rule admits 9 of 22 models, 8 of which passed P23. The new scoring therefore uses fresh
  graphs only.
* **Qualification (discovery split, new generator seeds 5050 + W).** Delayed layer-1 trace at W = 4, 600 discovery
  graphs, steps 1..max(8, K). k*(x) = argmax step of per-edge reachable attention on depth-x edges (x = 1, 2, 3). A
  model is a **single-step reader** if k*(1) < k*(2) < k*(3), and for each x the per-edge excess (reachable − decoy)
  at k*(x) is > 0 and every step with |k − k*(x)| ≥ 2 has excess ≤ 0.35 × the excess at k*(x).
* **Scoring (confirm split, new generator seeds 3030 + W for traces at W = 2, 4, 8; 3037 for blocking).** For each
  single-step reader, P23′ and P24′ use the PROTOCOL4 addendum-E rules unchanged (P23: Δ ≤ −0.05 at k*(x), |Δ| ≤
  0.02 at |k − k*(x)| ≥ 2, other branches ≥ −0.01; P24: same k* at W = 2, 4, 8, per-edge W8/W2 in [0.10, 0.45],
  total W8/W2 in [0.6, 1.6]); k* is re-derived from the fresh W = 4 confirm trace, as in addendum E.
* **Readings.** Confirmed if ≥ 80% of single-step readers pass, fails if < 50%, otherwise mixed; reported with the
  non-qualifying models' results for comparison. Models: the 22 of addendum E. Pilot runs of addendum A that pass the
  gate are added as a second, fully prospective set, reported separately.
* **Hardware amendment to addendum A (written 2026-09-25 ~15:20, before the RTX 2070 claimed any run; no prediction
  changes).** Cloud stock was nearly empty, so the local RTX 2070 (cuda:2) also takes runs from the shared queue. It
  has no fast bf16 and trains in fp32 (`--amp fp32`, as the P19 no-step runs did on that card). Each run's precision
  is in its `config.json`; R1 is reported both pooled and with fp32 runs excluded.

## Results

Scored 2026-09-25 ~19:05: `docs/PROTOCOL5_RESULTS.md`. R1 holds (7/8 vs 5/8, p = 0.28); P26 confirmed (1/6 vs 5/6,
AUC 0.83); P23′ and P24′ mixed (7/9 each; pilot set 3/4 each).
