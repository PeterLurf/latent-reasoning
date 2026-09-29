# Protocol 4 — runs needed to support the Part III claims

Written 2026-09-24, after the Part III scorecard (`results/protocol3_confirm.json`) and before any run
below was started. Run list and rationale: `docs/TRAINING_TODO.md`. Every run uses the seed-0 main recipe
(`runs/nostep_s0/config.json`: GPT-2 init, PARENT + CHILDREN scaffold annealed to 0 by 15k, gated
curriculum, 30k steps, batch 512, warmup 300, lr 1e-3) with only the flags named for it changed.

Rules carried over unchanged: stage-1 restart rule (PROTOCOL3; at most three relaunches, every attempt
kept on disk and reported), the addendum-J gate (overall delayed balanced accuracy ≥ 0.85 **and**
depth-matched 5-hop ≥ 0.60), and "extrapolates" = distance-7 delayed balanced accuracy ≥ 0.60 with
bootstrap lower bound > 0.55 at the stated K′. A run that fails the gate leaves its predictions
unscored and is reported as a training failure.

Device note: the RTX 3070 is visible again. torch now numbers the cards cuda:0 = RTX 4060,
cuda:1 = RTX 3070, cuda:2 = RTX 2070 (bf16 on the first two, fp32 on the 2070).

## Addendum A (Phase 1: step embedding and the stop; written before any of these runs started)

* **P19 (replication of P18).** nostep_s1, nostep_s2, nostep_s3 (`--no_step_emb`, K = 6, seeds 1–3,
  RTX 2070 fp32 like nostep_s0). Each run that passes the gate extrapolates at K′ = 12.
  Scored per run; the headline claim ("without a step index the stop is not learned") is called
  replicated if every competent run passes, and weakened if any competent run fails.
* **P20 (replication of the K = 8 cell).** budget_k8_s1, budget_k8_s2 (K = 8, seeds 1–2, RTX 4060
  bf16 like budget_k8_s0). Each run that passes the gate has a delayed layer-1 sweep that reaches
  depth ≥ 4 within six steps (E ≥ 4, same trace and reading as addendum J) and does **not**
  extrapolate at K′ = 16 (distance 7 < 0.60). Secondary (P17 extended): if such a run does not
  extrapolate, its frozen-clock run (step embedding held at step 2, K′ = 12) does (≥ 0.60, lower
  bound > 0.55).
* **P21 (the decisive cell).** nostep_k8_s0, nostep_k8_s1, nostep_k8_s2 (K = 8 **and**
  `--no_step_emb`, seeds 0–2, RTX 3070 bf16). Manipulation check first: the run's sweep must reach
  E ≥ 4 within the budget, otherwise it does not test the question and P21 is unscored for it.
  The halt account predicts that each competent run with E ≥ 4 extrapolates at K′ = 16.
  A "finishing inside the budget is what prevents extrapolation" account (Zhang et al. 2607.20594)
  predicts that none does. Reading fixed now: ≥ 2 of the scorable runs extrapolate → supports the
  halt account; ≥ 2 do not → supports the finish-early account; anything else (including fewer than
  two scorable runs) → unresolved, and more seeds are needed.
* **randk** (K sampled per batch) is not covered here; its code does not exist yet. It gets its own
  addendum before it is trained.

## Addendum B (written 2026-09-24 23:40, before any cloud run started; hardware only, no prediction changes)

* **Cloud hardware.** Some runs of this protocol and of TRAINING_TODO Phases 0-3 may train on a rented
  RTX 4090 (RunPod community cloud, Linux, torch 2.14.0+cu130 as locally), several runs at once on one
  card. Recipe, seeds, gate, restart rule and scoring are unchanged. Each run's `config.json` records
  its device and the new `amp` field; the research log lists which runs trained where.
* **Precision is held to the addendum-A statement.** Runs addendum A places on the RTX 2070 (fp32)
  use `--amp fp32` when they move to the 4090. That flag also disables TF32, which the 2070 does not
  have. Runs addendum A places on the RTX 4060 / 3070 (bf16) run bf16 (`--amp auto`) on the 4090.
  `--amp auto` (the default) is exactly the previous behaviour, so local runs are unaffected.
* **Restart rule on Linux.** `scratch/stage1_guard.py` and `scratch/chain.py` now use POSIX process
  calls off Windows. The rule itself (kill at step >= 4000 with no stage-1 advance, archive, relaunch,
  at most three relaunches) is unchanged.
* **randk** still has no frozen prediction. The `--K_random lo:hi` flag now exists in `latent/train.py`,
  but randk runs stay unstarted until their own addendum is written.

## Addendum C (written 2026-09-25 ~00:00, before any run below started; which runs train where, no new predictions)

* **Benchmark (not scored).** On a RunPod RTX A4000 (Linux, torch 2.14.0+cu126, driver 550 host) one
  K = 6 run takes 91 ms/step (RTX 4060 locally: 133). Two runs on one card share it with no gain in total
  throughput (10.7 vs 11.0 steps/s), so each cloud run gets its own pod. Hosts with driver 580.178 failed
  CUDA initialisation (`cuInit` 999) inside the container on three pods and were deleted unused.
  Cloud pods therefore use hosts that pass a `cuInit` check, with torch 2.14.0+cu126. The local cards
  use cu130; that is the only software difference.
* **Local runs (unchanged from addendum A):** nostep_s1 (RTX 2070), budget_k8_s1 and budget_k8_s2 (RTX 4060),
  nostep_k8_s0, nostep_k8_s1 and nostep_k8_s2 (RTX 3070). The RTX 2070 chain stops after nostep_s1.
* **Cloud runs, one pod each; stage-1 rule and addendum-J gate as before.** Predictions are the
  existing ones, restated so each run is scored as already written:
  * nostep_s2 and nostep_s3: P19 (addendum A), run with `--amp fp32` (addendum B).
  * budget_k4_s3 (K = 4, seed 3): addendum I/J. If it passes the gate, it extrapolates (distance 7 at
    K' = 12 >= 0.60, lower bound > 0.55) or its query-time share at W = 4 exceeds the reference range.
  * budget_k10_s1, s2, s3 (K = 10): addendum G. If a run passes the gate, it does not extrapolate
    (distance 7 at K' = 16 < 0.60) and its frontier reaches depth >= 4 by step 6.
  * v3_rawfb_s1 (`--feedback pre_ln`, seed 1): P13. beta differs from [0.62, 0.87] in the direction
    of 1, and competition persists.
  * v2_deep_s1 (`--layers 4`, seed 1): P12 (Gq share at W = 4 above the recurrent range's minimum;
    distance-3 Gq share > 0.5), plus P17 if it does not extrapolate normally (frozen-clock distance 7
    at K' = 12 >= 0.60, lower bound > 0.55).
  * v1_wide_s2 (`--d 512`, seed 2): addendum H reading. Targeted W8/W2 > 0.5 → the wider thought;
    inside 0.18–0.37 → V1's run; in between → unresolved.
* **Not started:** randk (no frozen prediction yet), fork_* (generator code missing), the Phase 0 stall pilot.

## Addendum D (written 2026-09-25 ~00:45, before any run of addenda A-C was evaluated)

Written by the assistant running the experiments overnight, with the user's go-ahead to proceed without per-step approval.

* **Scoring battery for every new run** (same tools as PROTOCOL3; results/<experiment>_<run>.json,
  run key "runs/<run>"): `competence --n 4096`, `competence_matched`, `length_gen`,
  `upfront_trace --delayed --n 400`, `budget_sweep`, and `clock_freeze` wherever P17 applies. Cloud runs
  are evaluated on their own pod right after training. The scorer is `latent/protocol4_confirm.py`.
* **Definitions carried over unchanged.** Gate = addendum J (overall delayed balanced accuracy >= 0.85
  at the run's own K and depth-matched 5-hop >= 0.60). E = addendum F (largest layer-1 frontier depth at
  any latent step k <= 6). "Extrapolates at K'" = distance-7 balanced accuracy at K' >= 0.60 with
  bootstrap lower bound > 0.55. "Does not extrapolate" (P20, G) = distance-7 accuracy at K' = 16 < 0.60,
  as J_k8 was scored.
* **P21 contingency.** If fewer than two of nostep_k8_s0..s2 are scorable (gate passed and E >= 4),
  train nostep_k8_s3 and then nostep_k8_s4 (same recipe, stopping at 5 seeds or once two are scorable
  in each direction). The reading from addendum A is applied to all scorable runs. If both ">= 2
  extrapolate" and ">= 2 do not" hold, P21 is unresolved.
* **P19 contingency.** If fewer than three of nostep_s1..s3 pass the gate, train nostep_s4, then
  nostep_s5 if still short (fp32). Every competent run is scored. The headline reading is as in addendum A.
* **Late-stage curriculum stalls.** The restart rule stays stage-1 only, and runs that end at 30k below
  full depth are gate failures. Replications must stay comparable to the runs already scored. The
  scorecard lists each run's final curriculum stage.
* **P22 (randk).** randk_s0, s1, s2 = base recipe with `--K_random 4:8` (K drawn uniformly per batch,
  step embedding present; gate evaluated at K = 8), bf16. If halting is keyed to a fixed final step,
  sampling K removes the key. Prediction: each run that passes the gate extrapolates at K' = 16.
  E is reported. P22 is supported if every scorable run extrapolates, and fails if any does not.
* **Evaluation-only confirmations** (fresh-graph re-runs of the Part III delayed blocking and width
  traces, the frontier-size-versus-width test) get their own addendum before they run.

## Addendum E (written 2026-09-25 ~13:40, before any code or run below; evaluation only, no training)

Written by the assistant; the user asked for these confirmations to run locally. **Disclosure:** the
frontier-size-versus-width test was run exploratorily earlier today on r3_pc_gated_gpt2, main_s1, main_s2,
main_s3, dev_s4 and main_s5 (discovery graphs; results/frontier_amp*.json) before this addendum existed,
against the note at the end of addendum D. Those four sweeping models (s0–s3) are therefore reported as
replications only. The primary test of P25 uses the other models.

* **Models.** Every recurrent (coconut) model that passes the addendum-J gate: r3_pc_gated_gpt2 (s0), main_s1,
  main_s2, main_s3, v1_wide_s0, v2_deep_s0, v3_rawfb_s0, budget_k4_s0, budget_k4_s2, budget_k8_s0, nostep_s0,
  nostep_s2, nostep_s4, budget_k8_s1, nostep_k8_s0, nostep_k8_s1, randk_s0, randk_s2, budget_k10_s1,
  budget_k10_s2, budget_k10_s3, v3_rawfb_s1. (randk_s1 is excluded: its curriculum ended at stage 4.)
* **Fresh graphs.** Every test samples the held-out **confirm** split (the family buckets that no exploratory
  analysis has used), with generator seed 2026 + a per-test offset. Fork graphs (P25) have no family split and
  use seed 4052.
* **Clock map (definition, not a prediction).** Delayed-format layer-1 trace at W = 4, 600 confirm graphs,
  Kp = max(8, K). For depth x = 1, 2, 3: k*(x) = the step with the largest mean per-edge attention on reachable
  depth-x edges. The model **sweeps** if k*(1) < k*(2) < k*(3) and at each k*(x) that attention is ≥ 1.5× the
  per-edge attention on decoy depth-x edges. Only sweeping models are scored on P23–P25; the rest are reported.
* **P23 (blocking is step-locked).** Blocking layer-1 attention onto one reachable branch's depth-x edges at step
  k (the latent/delayed_block.py design; W = 4, 400 confirm graphs, branch with within-branch depth ≥ 4). Per
  sweeping model, for x = 1, 2, 3: ΔP(reachable) of that branch's deeper nodes ≤ −0.05 at k = k*(x), and
  |Δ| ≤ 0.02 at every step with |k − k*(x)| ≥ 2. The other reachable branches' mean Δ ≥ −0.01 in every cell.
  A model passes if all of these hold.
* **P24 (width changes amplitude, not the clock).** Traces at W = 2, 4, 8 (600 confirm graphs each). Per
  sweeping model: (a) k*(x) is the same at all three widths for x = 1, 2, 3; (b) per-edge excess attention
  (reachable minus decoy) on depth-1 edges at k*(1), W8 / W2, lies in [0.10, 0.45]; (c) total excess on the
  depth-1 frontier (per-edge excess × mean number of reachable depth-1 edges), W8 / W2, lies in [0.6, 1.6].
  A model passes if (a)–(c) all hold.
* **P25 (the budget is split per frontier edge, not per branch).** Fork-mode graphs (`--shapes fork`
  generator, p_fork 0.5, W ∈ {2, 4, 8}, 8000 graphs). Manipulation check first: the model's delayed balanced
  accuracy on fork graphs is ≥ 0.80, else P25 is unscored for it. Per model: the fork/chain ratio of per-edge
  excess attention at (k*(x), depth x), among fork graphs with 2W edges at depth x, averaged over the three
  widths, is ≤ 0.70 for x = 2 and for x = 3. A per-branch budget predicts ≈ 1.0 and a per-edge budget ≈ 0.5.
* **Readings.** For each of P23–P25: **confirmed** if ≥ 80% of the scorable models pass, **fails** if < 50%
  pass, otherwise **mixed**. P25's primary reading uses the scorable models other than s0–s3; s0–s3 are
  reported separately as a replication on fresh graphs. The non-sweeping models' traces are reported without
  a reading.
