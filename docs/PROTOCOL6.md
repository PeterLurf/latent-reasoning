# PROTOCOL6 — why some sweeps keep running (Part VI)

Follows `docs/RESEARCH_TIMELINE.md` §7. Part V showed that *end activity* (EA; PROTOCOL5 addendum B: layer-1 per-edge
excess attention at the last trained latent step K, relative to its peak over steps 1..K, depths 1–5, W = 4 delayed
trace, 600 confirm graphs, generator seed 2026 + 4) predicts extrapolation (P26 confirmed). EA itself is a seed-level
outcome. This protocol asks what sets it, and re-tests P26 and the strict clock on a second recipe.

Definitions carried over unchanged: **competent** = PROTOCOL4 gate with the full-depth clause (delayed ≥ 0.85,
matched 5-hop ≥ 0.60, final stage full); **extrapolates** = distance-7 balanced accuracy ≥ 0.60 with bootstrap lower
bound > 0.55 at K′ = 12 (K = 6 runs; `latent.length_gen`); EA high = EA ≥ 0.25.

## Addendum A (written 2026-09-25 ~19:55, before any run or code below) — does further training close the sweep?

* **Exploratory basis (`scratch/ea_vs_lrint.py`, read before writing).** Across the 38 models with EA measured, EA is
  unrelated to the learning-rate mass spent at full depth (Spearman +0.15) or to the learning rate on reaching full
  depth (+0.08); two stage-4 stalls extrapolate with high EA (nostep_s0, rel_base_s23). A "closure" account (training
  at full depth gradually ends the sweep inside the budget) is therefore not favoured, but cross-sectional data cannot
  exclude it, because when full depth is reached is itself seed-dependent.
* **Intervention.** Each of the 12 competent PROTOCOL5 pilot runs is copied to `runs/<run>_ext` and continued from
  its final weights with `--steps 45000`, all other arguments unchanged, same precision as the original run. The
  cosine schedule therefore restarts at 0.25 × lr at step 30 001 and anneals to 0 at step 45 000: 15 000 more steps
  of full-depth training. Checkpoints every 2 500 steps. The final models get the standard battery
  (`scratch/eval_battery.sh`) and EA (as above).
* **Amendment (written ~20:10, before any continuation started).** Seven of the 17 runs to be continued trained on
  pods whose `state.pt` was never fetched, and five of them are low-EA runs. To keep the treatment uniform, *every*
  continuation starts from `model.pt` with a fresh AdamW state (zero moments, step count 0) and a fresh data stream
  (generator seed + 100 003), with the curriculum state rebuilt from the log (`scratch/p6_setup.py`). The
  intervention is therefore "15 000 further steps from the final weights with a restarted optimizer".
* **A1 (primary; my expectation is "stable").** EA class after continuation vs before (PROTOCOL5 values):
  * **stable** if the class is unchanged in ≥ 10 of 12 runs;
  * **closure** if ≥ 4 of the 6 high-EA runs end below 0.25 and ≤ 1 of the 6 low-EA runs ends at or above it;
  * **reopening** if the mirror image holds;
  * otherwise **unstable** (transitions reported).
* **A2 (P26 applied to the continued models).** Among continued models that are competent, P26 (a)–(c) of PROTOCOL5
  addendum B at threshold 0.25. These models are not independent of the pilot set; A2 asks whether the EA ↔
  extrapolation link survives a change of training length. In addition, if ≥ 3 runs change EA class, extrapolation
  should change in the same direction in ≥ 2/3 of them (**tracks** / **does not track**); fewer than 3 changes →
  not scored.
* **A3 (exploratory, no prediction).** The five stalled runs whose final stage is below full — rel_base_s21 (stage 2),
  rel_base_s22 (2), rel_base_s23 (4), rel_lr5_s25 (4), nostep_s0 (4, PROTOCOL3) — are continued the same way. Do
  they reach full depth, and if so does a high EA (rel_base_s21 0.61, rel_base_s23 1.00, nostep_s0 0.89) survive it?
  This is the direct test of "training without full depth favours an open sweep" (findings.md, Open).

## Addendum B (written 2026-09-25 ~19:55, before any run below) — development runs on a second recipe

* **Runs.** 12 new runs `dev6_s30..41`: the step-embedding recipe (PROTOCOL4 main recipe: K = 6, step embedding on)
  at lr 5e-4 (the §7 recipe for new questions), `--save_every 1000`. RTX 4060 / 3070 in bf16; the RTX 2070 in fp32
  (as in PROTOCOL5; precision recorded in each `config.json`, results reported with and without fp32 runs).
* **B1 (exploratory).** EA and distance-7 accuracy at K′ = 12 at every checkpoint (same graphs for every
  checkpoint; EA as above, length generalisation with 512 graphs). Questions: is the final EA class already set
  1 000 steps after full depth is reached? Does it drift monotonically? Does d7 move with EA inside a run?
* **P26b (prospective; the Part V caveat "one recipe").** On the competent `dev6` runs, P26 (a)–(c) exactly as in
  PROTOCOL5 addendum B, threshold 0.25, same readings (confirmed / fails / unresolved).
* **P23″ / P24″ (strict clock, second prospective set).** Single-step readers among competent `dev6` runs, qualified on
  discovery graphs and scored on fresh confirm graphs exactly as PROTOCOL5 addendum C (`latent.protocol5_confirm
  strict`). Readings over the pooled prospective sets (PROTOCOL5 pilot 4 + this set): **confirmed** if ≥ 80% pass,
  **fails** if < 50%, otherwise **mixed**; each set also reported separately. The 22 exploration models are not
  pooled (their threshold was tuned on them).
* **Descriptive.** Gate pass rate and failure stages of the step-embedding recipe at lr 5e-4.

* **Extension B′ (written 2026-09-26 ~04:45, after the 12 addendum-B runs were measured).** Only one competent dev6
  run has EA < 0.25, so P26b's AUC clause cannot be computed. The idle RTX 2070 trains three more runs of the same
  recipe (`dev6_s42..44`, fp32). Because they were added after the first 12 outcomes were seen, P26b is scored on the
  pre-registered 12 runs as written; the 15-run result is reported separately as an extension, never in place of it.
* **Extension B″ (written 2026-09-26 ~23:40, before any run below started).** The RTX 2070 is idle overnight (addenda
  F and H need bf16, which it lacks). It trains `dev6_s45..s50` in order (same recipe and flags as `dev6_s42..44`: step
  embedding, K = 6, lr 5e-4, fp32, `--save_every 1000`) until the addendum-H runs finish (~Sun 09:00); runs not started
  by then are dropped, and a run still training at that point is finished. The dev6 outcomes seen so far motivate it
  (only 1–2 low-EA competent runs), so it is an **extension, not a confirmation**: P26b stays scored on its 12 runs.
  Reported: the frozen P26 rule (PROTOCOL5 addendum B) on all competent dev6 runs s30–s50 pooled, and separately on the
  fp32 dev6 runs alone (s42–s50 plus any fp32 among s30–s41); plus B1-style trajectories (class at full depth + 1 000
  steps vs final) for the new runs.

## Addendum C (written 2026-09-25 ~19:55) — transfer to Coconut GPT-2 on ProsQA (preparation)

* Build a Coconut GPT-2 ProsQA pipeline (the public Coconut code and data) on a rented 24 GB card within the
  approved rate, then define before any trained model is inspected: (i) a step-specific blocking test (attention from
  latent step k to the graph statements at hop k), and (ii) the EA analogue. The predictions are written as a
  separate addendum once the pipeline runs and before the trained model is evaluated.

## Addendum D (written 2026-09-26 ~00:40, before any Coconut model was trained) — ProsQA predictions

* **Model.** Official Coconut recipe for ProsQA (`prosqa/prosqa_1gpu.yaml`: GPT-2, 6 stages of 5 epochs, 1 thought
  per step, padded to 6, 50 epochs, lr 1e-4, effective batch 128, fp32, seed 0),
  final checkpoint (epoch 50). Analysis code frozen as `prosqa/analyze.py` + `prosqa/gpt2c.py` (tested on untrained
  GPT-2; definitions in the module docstring). Decoy edges: statements whose subject is unreachable from the queried
  person, depth-matched by BFS from the other person.
* **Splits.** Everything chosen from data (layer, clock) is fixed on `prosqa_valid.json` (300 problems; corrected
  from "500" on ~21:15, nothing evaluated); every test is scored on `prosqa_test.json` (500).
* **T0 competence.** Candidate-scoring accuracy (log p of the correct vs the other option's answer sentence) at
  K = 6 on test ≥ 0.90; otherwise T1–T2 are reported but not scored (the paper reports 97%).
* **Frontier layer L\*.** argmax over the 12 layers of `frontier_score` (mean over thought steps of the largest
  per-hop excess, hops 1–5) on valid. Path clock k\*(h) = argmax step of attention on the answer path's hop-h
  statement at L\*, on valid, h = 1, 2, 3 (every problem has ≥ 3 path steps).
* **T1 sweep.** On test at L\*: k\*(1) < k\*(2) < k\*(3) re-derived on test equals the valid clock, and excess(k\*(h),
  h) > 0 for h = 1..3. Reading: **sweep present** if both, otherwise **absent**.
* **T2 step-locked use (P23 analogue).** On test, for each h = 1, 2, 3 with the valid clock k\*(h):
  (a) mean Δmargin for blocking the path's hop-h statement at step k\*(h) < 0, with the 95% bootstrap interval
  (examples resampled, 1 000 draws) below 0; (b) every step with |k − k\*(h)| ≥ 2 has |mean Δ| ≤ 0.25 × |Δ at
  k\*(h)|; (c) the same-hop decoy control at k\*(h) has |mean Δ| ≤ 0.25 × |Δ at k\*(h)|.
  Readings: **step-locked** if (a)–(c) hold for all three hops; **thoughts not needed** if blocking the statement at
  all six thought steps (`delta_all`) moves the margin by less than 0.1 nats at every hop (the answer re-reads the
  graph itself); otherwise **not step-locked**. No directional expectation is stated: the answer tokens can attend to
  the statements directly, which the synthetic model's query could not do for deep nodes.
* **T3 (descriptive).** End activity at L\* (and per layer) for the final model and at the stage-boundary checkpoints
  of epochs 30–50; candidate accuracy at K′ = 0–12 thoughts by path length. There is one model and no path longer than
  6 hops, so no P26 test is possible here.

* **Amendment (~20:45, training epoch 1, nothing evaluated).** Batch 32 × gradient accumulation 4 ran out of memory
  on the 24 GB RTX 4090 (community pod, $0.34/h); the run uses 16 × 8 (same effective batch 128), with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. At 16 × 8 one long batch ran out of memory in epoch 3
  (~21:15); the run was restarted from scratch at **8 × 16** (still effective batch 128; ~12 GB used). Only the
  validation accuracies of epochs 1–2 of the aborted run were seen (0.69, 0.72).
* **Speed patch (~23:45, stage 1, nothing evaluated but validation accuracy).** The official `coconut.py` rebuilds the
  input embeddings position by position in Python after each latent pass; in stage 1 the GPU sat at ~18% (13.5 s per
  optimizer step), which would have stretched the run to ~30–40 h. `prosqa/patch_coconut.py` replaces that rebuild with
  one out-of-place `index_put`; `prosqa/test_patch.py` checked it against the unpatched code on real batches from the
  epoch-5 checkpoint at stages 1, 3 and 6: identical losses, relative gradient difference ≤ 4.2e-7, 6–10× faster.
  Training was stopped in epoch 7, checkpoints after epoch 5 deleted, and the run resumed from `checkpoint_5` (the
  stage-1 boundary; the recipe resets the optimizer at every stage) with the patch.

## Addendum E (written 2026-09-26 ~00:55, before the end activity or extrapolation of any dev6 run was read except dev6_s39) — when is the open sweep decided? (branching)

* **Why.** Addendum A found EA stable under 15 000 further steps (A1 "stable"), so EA looks like a property of the
  solution a run settles into. Branching localises when that happens: if runs restarted from an early checkpoint
  with different data and a fresh optimizer all end in the original's EA class, the class was already set by that
  checkpoint; if they diverge, later training decides it. Disclosure: dev6_s39's end activity (0.37) and
  non-extrapolation were seen when scoring addendum A.
* **Runs branched.** Among the competent dev6 runs (addendum B), in order of seed: the first two with final EA ≥ 0.25
  and the first two with EA < 0.25 (if fewer exist, all of that class). If one class has none, the four lowest-seed
  competent runs are used and E1 is reported without the class balance.
* **Branch points.** The checkpoints saved at the curriculum advances to stage 2 and stage 4 and at the advance to full
  depth (train.py saves one at every advance).
* **Branches.** Two per (run, point): weights from the checkpoint, fresh AdamW state, fresh data stream (seed +
  200 003 + 7j), curriculum state from the log up to that step, then the original schedule to step 30 000 (same lr,
  precision and recipe; the gated curriculum runs as usual, so a branch may stall). `scratch/p6_branch.py`.
* **E1 (primary).** For each branch point, agreement = fraction of branches that reach full depth and end in the same
  EA class as their original. Readings per point: **set** if ≥ 7/8 of the competent branches agree, **open** if
  ≤ 5/8, else **partial**. The earliest "set" point is the answer to "when". My expectation: set by full-depth entry;
  no expectation for stages 2 and 4.
* **E2 (descriptive).** Extrapolation of the branches vs their original; gate outcomes (a branch that stalls is
  reported, not counted in E1).

## Addendum F (written 2026-09-26 ~10:20, before any run below) — does training precision change extrapolation?

* **Exploratory basis (read before writing).** Pooling every competent model with a recorded or inferred precision
  (41 models; the precision of some exploration models is inferred from the card, marked "?" in the working table):
  with EA ≥ 0.25, bf16-trained models extrapolate 14 of 19, fp32-trained 3 of 8; with EA < 0.25, 2 of 14 and 0 of 2.
  The fp32 models trained mostly on the RTX 2070 (no bf16), so precision and card are confounded, and recipes differ.
  If real, reduced-precision rounding noise would be a training-level cause of the extrapolating solution beyond EA.
* **Design.** Recipe of the PROTOCOL5 lr 5e-4 arm (no step embedding, K = 6, lr 5e-4, base recipe otherwise). Seeds
  50–57, each trained twice **on the same card** (RTX 4060 for seeds 50–53, RTX 3070 for 54–57): `--amp bf16` and
  `--amp fp32` (fp32 also disables TF32, as on the 2070). Same init and data stream within a pair. A third arm trains
  seeds 50–53 in fp32 on the RTX 2070 (card vs precision). Standard guard and battery; EA as in PROTOCOL5 addendum B.
* **P27 (primary).** Pairs in which both same-card arms are competent: count bf16-only extrapolators (b) and
  fp32-only extrapolators (f). **Supported** if b − f ≥ 3 and f ≤ 1; **not supported** if f ≥ b; otherwise
  **inconclusive**. Exact one-sided sign test on the discordant pairs reported.
* **Amendment (~16:00, no effect on training).** The eight runs not yet started also save a checkpoint every 1 000
  steps (`--save_every 1000`; saving draws no random numbers), for a follow-up on attention sharpness at full-depth
  entry (docs/LITERATURE_NOTES.md, bounded index-matching logit).
* **Amendment (~19:35, before either seed-53 run started).** The seed-53 pair moved from the RTX 4060 to the RTX
  3070 (both arms still on one card) because the 3070 finished its queue early.
* **Secondary (descriptive).** EA by arm (does precision move EA, or extrapolation given EA?); P26 within each arm;
  fp32-4060/3070 vs fp32-2070 for seeds 50–53; gate rates by arm.

## Addendum G (written 2026-09-26 ~16:25, before the model below was trained) — bypass-blocked Coconut on ProsQA

* **Why.** Addendum D: the official Coconut GPT-2 does not need its thoughts on ProsQA (accuracy 0.976 with no
  thoughts, 0.974 with six; blocking statements at the thoughts moves the margin ≤ 0.09 of 12.2 nats; no advancing
  sweep), replicating Aswal et al. (arXiv 2606.12689). Exploratory checks on that model (read before writing): with the
  statements hidden from the answer positions accuracy is 0.755 (K′ 0) / 0.73 (K′ 6) — the question sentence carries
  the rest; with everything before the thoughts hidden from `<end-latent>` and the answer, 0.52 / 0.54 (chance). The
  transfer question can only be asked of a model whose answer must go through its thoughts, as in our delayed format.
* **Model G.** The official recipe exactly as in addendum D (8 × 16, speed patch, 50 epochs, seed 0), plus the
  answer mask (`prosqa/patch_mask.py`): the `<end-latent>` position and every position from the answer marker "###"
  on cannot attend to any token before `<start-latent>`; thoughts and the written steps of early stages see everything.
  The mask is applied through the fused SDPA kernel (and, for the check, the eager path). It is verified on the
  pod before training (`prosqa/test_mask.py`: zero attention mass crosses it in training forward and generation;
  SDPA and eager give the same loss and the same generation; the mask changes the stage-6 loss).
* **Amendment (~20:10, before training).** Cloud: RunPod had no stock; the run uses a Vast.ai RTX 3090
  (on-demand, $0.20/h, driver 550 / CUDA 12.4, reliability 99.96%), same recipe. Masking moved from eager attention
  to the SDPA wrapper because eager attention stores every attention matrix, which risks running out of memory on
  24 GB at batch 8; the test above checks the two paths agree.
* **Replication (written ~20:15, before any addendum-G result was read).** Seeds 1 and 2 are trained
  the same way on two more Vast.ai RTX 3090s (≤ $0.34/h). G0–G3 are scored per seed; the addendum-G reading of each
  test is the one shared by at least 2 of the 3 seeds (otherwise "seed-dependent"). The analyses use the same mask
  (`analyze.py --mask_answer`) and are otherwise identical to addendum D (layer and clock from valid; tests on test).
* **G0 competence.** Test accuracy at K = 6 ≥ 0.90; below that, G2–G3 are reported but not scored.
* **G1 thoughts needed.** Accuracy at K′ = 0 ≤ 0.65 and (K = 6) − (K′ = 0) ≥ 0.20: **needed**; otherwise **not
  needed** (then G2–G3 are not scored). My expectation: needed.
* **G2 sweep.** T1 of addendum D unchanged: frontier layer L\* from valid; clock k\*(1) < k\*(2) < k\*(3) on valid,
  equal on test, excess > 0 at the clock. **Present / absent.**
* **G3 step-locked use.** T2 of addendum D unchanged ((a)–(c), thresholds 0.25, 1 000 bootstrap draws).
  **Step-locked / not step-locked** ("thoughts not needed" is superseded by G1).
* No directional expectation for G2–G3: the thoughts can see the question, so a goal-directed (backward) search is
  as possible as a breadth-first one. G4 (descriptive): end activity per layer; accuracy at K′ = 0–12 by path length.
* **Cost.** One community RTX 4090 at ≤ $0.34/h, ~17 h (≈ $6); the pod is deleted after a checksum-verified fetch.

## Addendum H (written 2026-09-26 ~20:37, before any run below) — does fp32 training stall the curriculum?

* **Exploratory basis (addendum F data, read before writing).** Same-card pairs so far (seeds 50–52, 54–56): bf16
  reached full curriculum depth while its fp32 twin did not in 3 pairs (50, 51, 55), the reverse in none; both
  reached it in 2 (54, 56), neither in 1 (52). On the 2070, fp32 runs reached full depth 0 of 4. The earlier
  "fp32 high-EA runs rarely extrapolate" pattern may be a training-reliability effect rather than an extrapolation one.
* **Design.** Same recipe as addendum F (no step embedding, K = 6, lr 5e-4), new seeds 70–77, each trained in bf16 and
  in fp32 on the same card (RTX 4060: 70–73; RTX 3070: 74–77), checkpoints every 1 000 steps.
* **P28.** Count pairs where only the bf16 arm reaches full depth (b) and where only the fp32 arm does (f).
  **Supported** if b − f ≥ 2 and f ≤ 1; **not supported** if f ≥ b; otherwise **inconclusive**. Exact one-sided sign
  test reported; also pooled with the addendum-F same-card pairs as a descriptive (not pre-registered) summary.
* **Secondary.** Steps spent per stage by arm; gate pass rates; EA and extrapolation of competent runs.

## Results

Pending.

## Addendum I (written 2026-09-27 ~06:51, before any run below finished; p29_s80 started right after) — does the frontier cross the horizon?

* **Exploratory basis (read before writing; `scratch/far_sweep.py`, `results/far_sweep.json`).** End activity is
  close to necessary but not sufficient for extrapolation. On the long-chain graphs of `latent/length_gen.py`, the
  layer-1 frontier excess on the **distance-7 edge** (attached chain hung from the leaf of a depth-6 chain, minus the
  matched unreachable chain), maximised over steps 1..12 and divided by the largest in-horizon excess, is
  **P7**. Over all 72 competent K = 6 models measured so far: AUC for extrapolation 0.96 (EA 0.82); within high-EA
  models 0.98; P7 ≥ 0.2 → 28/29 extrapolate, P7 < 0.2 → 8/43; Spearman with d7 0.84 (EA 0.51). Reading: an open sweep
  that fails to extrapolate usually stops at the trained horizon (propagation failure) rather than crossing it while
  the answer fails to use it (readout failure). Caveats: the 0.2 threshold was chosen on these data, and P7 was
  measured on the same graphs as d7 (seed 99). On fresh graphs (seed 2033) P7 moved by ≤ 0.03 on 8 models.
* **Test set (prospective).** Every competent K = 6 run whose eval battery had not finished at the time of writing:
  the addendum-H pair `stall_{bf16,fp32}_s73`, extension-B″ runs `dev6_s48..s50` (those that get trained), and a new
  batch `p29_s80..s87` (no step embedding, K = 6, lr 5e-4, bf16, `--save_every 1000`; the addendum F/H recipe, whose
  competent models extrapolate about half the time): seeds 80–83 on the RTX 3070 now, 84–87 on the RTX 4060 after
  its addendum-H queue. Competence and extrapolation as in PROTOCOL5 (gate; d7 ≥ 0.60 with lower bound > 0.55 at
  K′ = 12).
* **Measurement.** P7 from `scratch/far_sweep.py --seed 2033` (fresh long-chain graphs, n = 1024, widths 2/4/6), code
  frozen as of this addendum. EA as before.
* **P29.** On the competent test runs: (a) P7 ≥ 0.2 classifies extrapolation correctly in ≥ 80 % of runs; (b) AUC of
  P7 ≥ 0.85. **Supported** if (a) and (b); **not supported** if the AUC < 0.70 or accuracy < 65 %; otherwise
  **inconclusive**. Needs ≥ 2 extrapolating and ≥ 2 non-extrapolating competent runs; else **unresolved**, and more
  seeds of the same batch are added (84+ numbering continues) until it resolves or the batch reaches 16 runs.
* **Secondary.** AUC of EA on the same runs (does P7 add to EA?); how many high-EA non-extrapolators have P7 < 0.1
  (propagation failure) vs ≥ 0.2 (readout failure); `far_peak_step` (does the frontier keep one hop per step beyond K?).
* **Amendment (2026-09-27 ~11:27, applying the stopping rule).** P29 was unresolved at 10:25 (one extrapolator), so
  seeds 88–95 were queued on the RTX 3070 as the addendum allows. At 11:26 the test set first had ≥ 2 extrapolating and
  ≥ 2 non-extrapolating competent runs, so it resolved: seeds not yet started (90–95) were dropped; seeds already
  started (88, 89) and the original batch (80–87) stay in the test set. P29 is scored once s86, s87 and s89 finish.

## Addendum J (written 2026-09-27 ~11:45, before any model below was trained) — delayed-query ProsQA

* **Why (exploratory basis, read before writing).** Addendum G seed 1: with the answer blind to the prefix, the
  thoughts become necessary (K′ = 0: 0.498; K = 6: 0.972) but do no step-by-step work — K′ = 1, whose single thought
  is the final hidden state of the prefix pass, already gives 0.938 (5-hop paths 0.913); no hop-by-hop sweep, no
  step-locked use. The candidate question precedes the thoughts, so GPT-2 answers it inside the prefix pass and the
  thought only carries the answer. The toy model's **delayed** format rules this out by putting the query after the
  thoughts. Addendum J does the same on ProsQA.
* **Design.** `prosqa/make_delayed.py`: the prefix is the statements plus "Start: <root>."; the candidate question
  ("Is <root> a <c1> or <c2>?") is placed right after `<end-latent>` (`prosqa/patch_delayed.py`; never a training
  target). Mask: `<end-latent>`, the query tokens with the newline after them, and the answer may not attend to
  anything before `<start-latent>` (written steps of early curriculum stages stay open, as in addendum G). The
  thoughts see the graph and the start entity but not the candidates, so to answer they must carry which concepts are
  reachable from the root. Otherwise the addendum-G recipe (official Coconut GPT-2, 50 epochs, effective batch 128,
  fp32, same seeds' data order). Pre-training check `prosqa/test_delayed.py` must pass (query placement, labels,
  exact blocked rows, eager = SDPA, no mass crosses). Analyses as addendum G on the delayed test set, plus K′ = 0–3.
* **Seeds and cards.** Seeds 0 and 1 (`prosqa_j.yaml`, `prosqa_j_s1.yaml`); RunPod first (≤ $0.34/h, CUDA 12.4–12.8),
  Vast.ai otherwise. Seed 2 (`prosqa_j_s2.yaml`) runs only if seeds 0 and 1 disagree on J0, J1, J1b or J2; the reading
  of each test is the one shared by both seeds (or by 2 of 3).
* **J0 competence.** Test accuracy at K = 6 ≥ 0.85 (the toy model's delayed gate). Below: "not learned in this
  recipe"; J1–J3 reported, not scored.
* **J1 thoughts needed.** K′ = 0 ≤ 0.65 and (K = 6) − (K′ = 0) ≥ 0.20.
* **J1b work in the thoughts.** (K = 6) − (K′ = 1) ≥ 0.10: **multi-step**; otherwise **prefix-computed** (as G).
* **J1c depth dependence.** At K′ = 2, accuracy on path length ≥ 5 is lower than on path length 3 by ≥ 0.10:
  **depth-dependent** (more hops need more thoughts, the signature of a sweep); otherwise **not**.
* **J2 hop-by-hop sweep** = T1/G2 rule; **J3 step-locked use** = T2/G3 rule (unchanged code, `prosqa/score.py`).
* **Readings.** J0 + J1 + J1b + J2 → the frontier sweep transfers to GPT-2 when the query is delayed. J0 + J1 + J1b
  without J2 → multi-step latent computation of another form. J1b "prefix-computed" → even with a delayed query GPT-2
  packs the reachable set into the prefix pass (superposition in one step), which is a finding in itself.

## Addendum K (written 2026-09-27 ~13:23, after P29 was scored and before any run below started) — P7 as a ranking

* **Why.** P29 failed as registered (accuracy 0.50, AUC 0.73). Read after scoring: the fixed 0.2 threshold was too high
  (extrapolators at P7 0.08 and 0.11; K = 8/10 models likewise), and the ratio is noise when the in-horizon excess is
  near zero (false positives at 0.0003 and 0.005, below every model of the 72-model basis). Every competent run with a
  measurable in-horizon frontier was ranked correctly. This addendum tests that revised, threshold-free claim on new
  runs; it does not rescue P29, which stays "not supported".
* **Measurement.** Unchanged frozen code (`scratch/far_sweep_frozen_I.py --seed 2033`). **Floor:** a run whose
  in-horizon excess (`in_peak`) is < 0.03 (about the 5th percentile of the basis) has no measurable frontier on these
  graphs; it is excluded from the primary test and reported separately.
* **Test set.** New runs only: `p30_s100..s111` on the RTX 4060 and `p30_s112..s123` on the RTX 3070 (the addendum-I
  recipe: no step embedding, K = 6, lr 5e-4, bf16, `--save_every 1000`), and `dev6_s51..s56` on the RTX 2070 (the dev6
  recipe, fp32). All queued runs are trained (no optional stopping); runs not started by Mon 2026-09-28 09:00 are
  dropped. Competence and extrapolation as in PROTOCOL5.
* **P30.** Among competent test runs with `in_peak` ≥ 0.03: AUC of P7 for extrapolation. **Supported** if AUC ≥ 0.80
  and AUC(P7) > AUC(EA) on the same runs; **not supported** if AUC < 0.65; otherwise **inconclusive**. Needs ≥ 3
  extrapolating and ≥ 3 non-extrapolating runs, else **unresolved** (no further seeds).
* **Secondary.** Runs below the floor: how many extrapolate (predicted: none). P7 ≥ 0.2 accuracy (for comparison with
  P29). Per-recipe AUCs.
* **Amendment (2026-09-27 ~14:56, before any P30 result was looked at (two runs have finished, unread); the user added Vast.ai credit).** The test set
  grows by `p30_s124..s159` (36 runs, same recipe and flags as `p30_s100..123`) trained on three Vast.ai 24 GB bf16-capable cards (RTX A5000 / 3090 class)
  (12 per instance, in seed order; RunPod had no bf16-capable card ≤ $0.34/h). All 36 are trained (same no-stopping
  rule; runs not started by Mon 09:00 dropped). Rule, floor, thresholds and measurement unchanged; P7 and EA are
  measured locally on the fetched models. Per-card AUCs reported as a secondary.
* **Amendment (~15:05, before any Vast run started).** The third Vast host (RTX 3090, driver 580) never finished
  loading and was destroyed unused; its seeds 148–159 are split over the two RTX A5000 hosts (18 runs each, same
  order within a host). Nothing else changes.
* **Amendment (~17:10, no Vast result read).** Host B (RTX A5000, RO) ran at a 135 MHz SM clock (~20× slow) and had
  not finished `p30_s136`; it was destroyed and replaced by another RTX A5000 (AU, checked 2 100 MHz, ~30 bf16 TFLOPS),
  which runs the same host-B queue from `p30_s136`. Nothing else changes.

## Addendum L (written 2026-09-27 ~22:22, before any run below) — spare thoughts: does the frontier cross the training depth?

* **Why (exploratory basis).** Eight competent K = 8/10 models (Part IV) gave P7 AUC ≈ 0.90, and their
  non-extrapolators stop at distance 6 although they have thoughts to spare (P7 0.00–0.14): the horizon looked like the
  training graph depth (6), not K. Five of those models are K = 8, on an older recipe (lr 1e-3).
* **Design.** The addendum-K recipe with 8 thoughts: `--no_step_emb --K 8 --lr 5e-4 --save_every 1000`, bf16 on the
  RTX 4060/3070 (`k8_s200..s211`), fp32 on the RTX 2070 (`k8_s212..s215`), each queued behind that card's P30 queue.
  All queued runs are trained; runs not started by Mon 2026-09-28 12:00 are dropped. Competence and extrapolation as in
  PROTOCOL5 (gate at K = 8; d7 at K′ = 12). P7 from the frozen `far_sweep_frozen_I.py --seed 2033`, floor in_peak ≥ 0.03.
* **L1.** Among competent runs above the floor, AUC of P7 for extrapolation ≥ 0.80: **supported**; < 0.65: **not**;
  else **inconclusive** (needs ≥ 2 extrapolating and ≥ 2 non-extrapolating runs, else **unresolved**).
* **L2 (horizon at training depth).** Among competent non-extrapolators above the floor, ≥ 80 % have P7 < 0.10:
  **supported** (the frontier stops at depth 6 despite 2 spare thoughts); ≤ 50 %: **not supported**; else
  **inconclusive** (needs ≥ 3 such runs).
* **Amendment (2026-09-28 ~01:45, before any addendum-L result was read; one run had finished, unopened).** To keep
  the cards busy until the next experiment is ready, seven seeds are appended to the same queues: `k8_s216..s219`
  (RTX 3070, bf16), `k8_s220..s221` (RTX 4060, bf16), `k8_s222` (RTX 2070, fp32). Same recipe, rules and 12:00 cut-off.

## Addendum M (written 2026-09-28 ~02:03, before any run below and before any addendum-L result was read) — move the training depth

* **Why.** Failing sweeps stop at distance 6, where both the number of thoughts (K = 6) and the training depth end.
  Addendum L gives spare thoughts at training depth 6. Addendum M does the converse: K = 6 but training graphs capped
  one level shallower (within-branch depth ≤ 4, distance ≤ 5), so every model has one thought to spare. If the horizon is
  the **training depth**, failing models now stop at distance 5; if it is the **step budget**, they cross to 6.
* **Design.** New flag `--depth_cap 4` in `latent/train.py` (the gated curriculum runs as usual; its final "full" stage
  keeps the cap; checked in a 400-step smoke run: max_bdepth stays 4). Recipe as addendum K: `--no_step_emb --K 6 --lr
  5e-4 --save_every 1000`, bf16 on the RTX 3070/4060, fp32 on the RTX 2070: `m_s300..s307` (3070), `m_s308..s313`
  (4060), `m_s314..s316` (2070), each queued behind that card's addendum-L queue. All queued runs are trained; runs not
  started by Mon 2026-09-28 20:00 are dropped.
* **Competence.** The standard gate (delayed ≥ 0.85, matched 5-hop ≥ 0.60, final stage "full" = cap reached).
  **Extrapolation by one level:** matched 6-hop bal-acc ≥ 0.60 (`competence_matched`, distance 6 is beyond training).
* **Horizon measure (from the frozen `far_sweep_frozen_I.py --seed 2033` output, no new code).** near[k][j] = layer-1
  excess on chain A's edge at distance j + 1. `in5` = max over steps 1–6 and distances 1–5; `e6` = max over steps 1–12 of
  the distance-6 edge; **P6 = e6 / in5**. Floor: `in5` ≥ 0.03.
* **M1 (horizon at training depth).** Among competent runs above the floor, the share with P6 < 0.10 (frontier stops at
  distance 5 with a thought to spare). **Supported** if ≥ 30 % (for comparison: 53 % of competent depth-6 K = 6 models
  stop at 6, P7 < 0.1, in the 72-model basis); **not supported** if ≤ 10 %; else **inconclusive** (needs ≥ 5 runs).
* **M2.** AUC of P6 for extrapolation by one level ≥ 0.80 **supported**; < 0.65 **not**; else inconclusive (needs ≥ 3
  of each outcome).
* **Secondary.** d7 at K′ = 12; `far_peak_step`; comparison with addendum L (L2 and M1 together separate step budget from
  training depth).

## Addendum N (written 2026-09-28 ~03:57, before any model below was trained) — ProsQA deeper than one pass

* **Why (exploratory basis, read before writing).** Addendum J seed 0: with the candidate question after the thoughts
  and the answer blind to the statements, GPT-2 answers with **no** thoughts at 0.85 and with one at 0.90 (= K = 6);
  no sweep, not step-locked. The `<start-latent>` position (whose final state is thought 1) sees the statements and the
  start entity, and GPT-2's 12 layers compute the reachable set in that single pass. ProsQA's 3–6 hops fit inside one
  12-layer pass, so the thoughts have nothing serial to do. Our toy model, which does sweep, has 2 layers.
* **Design.** Addendum J exactly (data, mask, recipe, 50 epochs, analyses) but GPT-2 truncated to its **first 4
  pretrained blocks** (`n_layer: 4`, `prosqa/patch_layers.py`; `prosqa_n4.yaml`, `prosqa_n4_s1.yaml`). K′ analysis
  adds 4 and 5 thoughts. Seeds 0 and 1 on Vast.ai (RunPod first if a card ≤ $0.34/h exists); seed 2 only if they
  disagree on N0, N1, N1b or N2.
* **N0 competence** test accuracy at K = 6 ≥ 0.85; below: "not learned by 4 layers", N1–N3 reported, not scored.
* **N1 thoughts needed** K′ = 0 ≤ 0.65 and (K = 6) − (K′ = 0) ≥ 0.20.
* **N1b work in the thoughts** (K = 6) − (K′ = 1) ≥ 0.10 → **multi-step**; else **prefix-computed**.
* **N1c depth dependence** at K′ = 2, accuracy on path length ≥ 5 below path length 3 by ≥ 0.10 → **depth-dependent**.
* **N2 sweep** = T1 rule; **N3 step-locked** = T2 rule (unchanged code; layer chosen on validation as before).
* **Readings.** N0 + N1b (+ N1c) → a shallower GPT-2 is forced into multi-step latent computation (the depth of one pass
  was what made thoughts unnecessary in J); with N2/N3 the sweep itself transfers. N1b "prefix-computed" even at 4
  layers → ProsQA is shallow for GPT-2 in a different sense (e.g. parallel hop composition within a layer).

## Addendum O (written 2026-09-28 ~15:40, before any run below) — does a longer scaffold make the frontier cross?

* **Exploratory basis (read before writing; P30 and L data).** Competent runs that reached full curriculum depth before
  step 12 500 — while the PARENT/CHILDREN lookup scaffold was still on (weight 1 until step 10 000, annealed to 0 by
  15 000) — extrapolated 6/7 (P30) and 2/3 (L); those reaching it later 15/35 and 2/14. Full-depth step alone ranks
  extrapolation with AUC ≈ 0.66. This may only mean that fast-learning runs are also good runs; an intervention decides.
  (Earlier parts: without the scaffold the toy model does not learn the task at all; addendum N: a 4-layer GPT-2 with
  Coconut's recipe, which has no such scaffold, does not learn delayed ProsQA.)
* **Design (paired).** Re-train P30 seeds on the same card with only the scaffold extended: `--aux_steps 20000
  --aux_anneal 5000` (weight 1 until step 20 000, 0 by 25 000; 30 000 steps as before). Everything else identical
  (addendum-K recipe, same seed → same init and data stream). Arm O: `o_s100..s111` (RTX 4060), `o_s112..s123`
  (RTX 3070), paired with `p30_s100..s123`; `o_dev6_s51..s56` (RTX 2070, dev6 recipe, fp32), paired with `dev6_s51..s56`.
  Each card starts after its addendum-M queue. All queued runs are trained; runs not started by Tue 2026-09-29 12:00 are
  dropped (their pairs too).
* **O1 (primary, no-step recipe, 24 pairs).** A run "succeeds" if it is competent and extrapolates (PROTOCOL5 rules).
  Base arm: 6/24 (seeds 105, 114, 116, 117, 122, 123). Count discordant pairs: long-scaffold-only successes (l) and
  base-only successes (b). **Supported** if l − b ≥ 4 and the one-sided exact sign test p ≤ 0.05; **not supported** if
  l ≤ b; otherwise **inconclusive**.
* **Secondary.** Full-depth step by arm (does the longer scaffold move full depth before its end?); competence rate by
  arm; P7 by arm (frozen code, seed 2033; does the frontier cross more often?); the dev6 pairs (6, descriptive).
* **Amendment (2026-09-28 ~16:42, before any addendum-O run finished; user: "keep running experiments").** More pairs
  on Vast.ai: seeds 124–159 (no-step recipe, bf16), each trained **twice on the same instance** — base scaffold
  (`ob_s<seed>`, the addendum-K recipe) and long scaffold (`o_s<seed>`, `--aux_steps 20000 --aux_anneal 5000`) — so both
  arms of a pair share the card. Four RTX 3090 instances, 9 seeds each, base then long per seed. O1 is scored on **all**
  pairs (24 local + 36 Vast; same rule: l − b ≥ 4 and one-sided sign test p ≤ 0.05 → supported; l ≤ b → not
  supported), and separately per set as a secondary. Runs not finished by Tue 2026-09-29 12:00 drop with their pair.
* **Amendment (~16:50, before any Vast O run started).** No RTX 3090 on a proven driver was rentable; the extension runs
  on three bf16 cards instead (2 × RTX A4000, 1 × RTX 4070) and is cut to seeds **124–147** (24 pairs, 16 runs per
  instance, base then long per seed, both arms on the same instance) so that it finishes before the Tuesday-noon
  cut-off. O1 is scored on all 48 pairs (24 local + 24 Vast).

## Addendum P (written 2026-09-28 ~21:18, before any extended problem was evaluated; only a 20-problem m = 0 smoke test was run) — ProsQA deeper than training

* **Why.** In the toy model a failing sweep stops at the training depth (L2, M1), and extrapolators use extra thoughts
  to go deeper. ProsQA training paths are 3–6 hops (train: 7140 / 7645 / 2541 / 560). Does Coconut GPT-2 show a
  training-depth wall, and do extra thoughts help on deeper problems? Evaluation only; no training.
* **Data (`prosqa/make_long.py`, seed 0; control seed 1).** Each of the 500 test problems with m ∈ {0, 2, 4, 6}: two
  fresh linear chains of m new concepts (training vocabulary, unused in that problem) hang from the correct and the wrong
  option; the question asks about the chain ends; path length = original + m (m = 6: 9–12 hops, all beyond training).
  **Control** (`--control`): the same 2m statements hang from two random off-path concepts; question and path unchanged.
  Delayed versions via `make_delayed.py` for the J models. Files `prosqa/data/prosqa{,_d}_{long,ctl}_test.json`
  (md5 of prosqa_long_test.json 4a581f7e…).
* **Models.** G s0–s2 (question first) on the plain sets, J s0–s2 (delayed query) on the delayed sets, all at epoch 50
  with the answer mask (`prosqa/long_eval.py`). N models are excluded (not competent). J s2 enters once fetched.
* **Measures (K′ = 6, the trained budget, unless stated).** a_m = candidate accuracy at extension m; c_m = the same on
  the control set.
* **P1 (training-depth wall).** Per model: **wall** if a_6 ≤ 0.60 and c_6 ≥ 0.80; **generalises** if a_6 ≥ 0.80;
  **length failure** if a_6 ≤ 0.60 and c_6 < 0.80; else intermediate. Across the six models: **supported** if ≥ 4 walls;
  **not supported** if ≥ 4 generalise; else **mixed**.
* **P2 (extra thoughts help with depth).** Per model, gain = a_6(K′ = 12) − a_6(K′ = 6) ≥ 0.05. **Supported** if ≥ 4
  of 6 models; **not supported** if ≤ 1; else **inconclusive**.
* **Secondary (descriptive).** Accuracy by path length 3–12 and by m at K′ = 0, 1, 6, 12; the drop from length 6 to 7
  vs 5 to 6 (sharp wall vs gradual decline); G vs J.
* **Expectation (not scored).** Since one thought already gives nearly full accuracy on ProsQA and more do not help,
  I expect gradual decline rather than a wall, and no gain from K′ = 12 — i.e. P1 and P2 not supported, which would mark
  the toy's training-depth horizon as specific to the step-by-step sweep.

## Addendum Q (written 2026-09-28 ~22:02, before any model below was trained; addendum-P K′ ≤ 6 accuracies on the long set had been seen for g_s0 and j_s1, the control set for none) — does GPT-2's depth horizon follow the training depth?

* **Why.** In the toy model the horizon of a failing sweep is the training depth (L2, M1). Addendum-G models (trained on
  3–6 hop problems) are as accurate on 5–6 hops as on 3–4 (K = 6 gap 3–4 minus 5–6: −0.033, −0.004, −0.027; J:
  −0.028, −0.030, −0.001). Train the addendum-G recipe on 3–4 hop problems only: if GPT-2's horizon follows the
  training depth, 5–6 hop problems (in the test set, never trained) fail.
* **Design.** `prosqa/pod_prosqa_q.sh` = `pod_prosqa_g.sh` with the training set filtered on the pod to paths ≤ 4
  (14785 of 17886 problems; validation unchanged), seeds 0 and 1 (`prosqa_q.yaml`, `prosqa_q_s1.yaml`), answer mask,
  50 epochs. Confound noted: 83 % of the training data. Analyses: trace and K′ ∈ {0, 1, 2, 6, 12} on the test set,
  and `long_eval.py` on the addendum-P long and control sets (regenerated on the pod; md5 printed and checked).
  Cards: RunPod first (≤ $0.34/h), Vast otherwise.
* **Q0 competence.** K = 6 accuracy on 3–4 hop test problems ≥ 0.85. Below: not scored.
* **Q1 (horizon at training depth).** gap = acc(3–4 hop) − acc(5–6 hop) on the original test set at K = 6 (n = 419 /
  81). Per seed: **wall** if gap ≥ 0.15; **no wall** if gap ≤ 0.05. **Supported** if both seeds show a wall; **not
  supported** if both show none; else **mixed**.
* **Q2 (thoughts rescue depth).** acc(5–6 hop, K′ = 12) − acc(5–6 hop, K = 6) ≥ 0.05 in both seeds: supported;
  ≤ 0.02 in both: not supported; else inconclusive.
* **Secondary.** Long set: a_m by m and by path length (5–6 hop problems made by extending 3–4 hop ones, n = 419,
  a second estimate of the gap); comparison with the addendum-P curves of the G models.

## Addendum R (written 2026-09-28 ~22:02, before any run below) — dose-response of the toy horizon, and more depth-cap-4 runs

* **Why.** L2 and M1 place the horizon at the training depth for depth 6 and 5. A third point (cap 3: distance ≤ 4
  trained, two spare thoughts) tests the dose-response; more cap-4 runs tighten M1 and give M2 a chance to resolve.
* **Design.** Addendum-M recipe (`--no_step_emb --K 6 --lr 5e-4 --amp bf16 --save_every 1000`), `--depth_cap 3`
  (`r3_s400..s411`) and `--depth_cap 4` (`m_s317..s328`), interleaved on the RTX 3070 (`scratch/p6_r_3070.txt`) and
  RTX 4060 (`scratch/p6_r_4060.txt`), each chained after that card's addendum-O queue. Runs not started by Wed
  2026-09-30 06:00 are dropped. (Cap-3 smoke run: max_bdepth ≤ 3 through stage 4.)
* **Competence (cap 3).** Final stage "full" and matched 4-hop bal-acc ≥ 0.60 (the deepest trained distance; the
  delayed-format gate is dropped because its test graphs include untrained distances). Extrapolation by one level:
  matched 5-hop ≥ 0.60. Horizon: in4 = max over steps 1–6 and distances 1–4 of the frozen far_sweep near array; e5 = max
  over steps 1–12 of the distance-5 edge; **P5 = e5 / in4**; floor in4 ≥ 0.03.
* **R1 (horizon at training depth, cap 3).** Share of competent runs above the floor with P5 < 0.10: **supported** if
  ≥ 30 %; **not supported** if ≤ 10 %; else inconclusive (needs ≥ 5 runs).
* **R2 (dose-response across caps).** Share of competent runs whose frontier stops one level past the training depth
  is ≥ 30 % at every cap (cap 3: R1; cap 4: M1 pooled with the new runs; depth 6: L2 basis) → the horizon follows the
  training depth at all three depths: **supported**; any cap ≤ 10 %: **not supported**.
* **R3 (M2 pooled).** Cap-4 runs of addenda M and R pooled: AUC of P6 for extrapolation by one level ≥ 0.80 supported,
  < 0.65 not, else inconclusive (needs ≥ 3 of each outcome, else unresolved). Also for cap 3 with P5 (exploratory).
* **Note (2026-09-29 ~07:10, addendum O cut-off applied early; no dev6 outcome read).** On the RTX 2070 (fp32, shared
  with addendum-P evaluation) each dev6 run takes ~4 h: o_dev6_s53 ends ~10:00 and o_dev6_s54 then runs past noon, so
  o_dev6_s55 and o_dev6_s56 cannot start before the Tue 12:00 cut-off and are dropped now (marked claimed in
  `scratch/p6_o_2070.txt.claimed`), letting the card move on to PROTOCOL7 T10. The dev6 pairs are descriptive only.
