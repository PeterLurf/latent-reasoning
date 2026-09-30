# PROTOCOL 7 — confirmation suite for the horizon account (pre-registration)

Written 2026-09-28 ~23:30 EDT, before any data below were generated or any analysis below was run. Append-only:
deviations go in dated amendments at the end. Frozen code named here is copied to `scratch/*_frozen_P7S.py` (or
`latent/frontier_causal.py` as committed) before the first confirmatory run.

## What is being tested

The claims to be made as strong as possible (`report/findings.md` findings 8–10, PROTOCOL6 addenda K–R):

* **H1 (horizon crossing).** Among models with an open sweep, whether the layer-1 frontier crosses the trained depth
  (P7) decides extrapolation. Evidence so far is correlational (P30 AUC 0.993, L1 0.90).
* **H2 (training-depth horizon).** A failing frontier stops at the training depth, not the step budget (L2, M1; R
  running). Evidence: attention readouts only.
* **H3 (set point).** The outcome is set around the arrival at full depth (E1, A1).
* **H4 (transfer).** Coconut GPT-2 on ProsQA computes in one pass; whether it has a training-depth wall is being
  tested (addenda P, Q).

Known weak points (from reviewers' likely objections and `docs/LITERATURE.md`): no causal test that the frontier
*produces* beyond-horizon answers (T1, T2); an in-distribution predictor has prior art (Zhang et al. 2026,
arXiv 2607.20594), so P7 must be compared head-to-head with in-distribution instruments (T4) and shown to add to
in-distribution competence (T5); forking paths in P7's definition (floor chosen after P29; T6) and code errors (T7);
attention patterns can look like a sweep without being used (Aswal et al. 2026; T1, T8); one architecture (T9) and one
graph family (T10); the toy→ProsQA gap (T11, T12).

## General rules

* Every test states its population, measure, decision rule and what each outcome would mean, before its data exist.
* "Competent" and "extrapolates" as in PROTOCOL6 (gate delayed ≥ 0.85, matched 5-hop ≥ 0.60, final stage full;
  extrapolation d7 ≥ 0.60 with CI lower bound > 0.55 at K′ = 12) unless a test states otherwise.
* Probe graphs for the new tests use **fresh seeds** (4242 unless stated) never used for P7, d7 or model selection.
* All AUCs are reported with graph-bootstrap or model-bootstrap 95 % CIs; primary p-values (permutation, 10 000
  shuffles) are Holm-corrected across the five primary tests (T1, T2, T3, T4, T9).
* Outcomes of retrospective tests on existing models (T3, T5, T6, T7, T8) are new measurements never computed before;
  they are labelled "retrospective" wherever reported.

## Tier 1 — evaluation only (existing models; runs in GPU gaps)

### T1 Frontier necessity (causal) — primary

* **Population.** Competent extrapolators of the P30 and L test sets (model files present locally).
* **Intervention** (`latent/frontier_causal.py`). Rollout to K′ = 12 on 1024 fresh long-chain probe graphs (seed 4242,
  widths 2/4/6; chain A at distance 1–6, chain B at 7–12, look-alike decoys D and C). In **layer 1 only, latent
  positions only**, the attention logits onto the six B-chain edge tokens get −1e4 at every step (**block-B**). Control
  **block-D**: the same on the six D-chain (unreachable, in-range look-alike) edges. The query/answer positions are
  never touched. Latent steps follow `LatentReasoner.latent_step` exactly (step embedding only if the model uses one).
* **Measures.** TPR on B1 (distance 7: share of B1 queries answered "reachable"); TPR on A1–A6 (in-range).
* **Per model.** Necessity shown if TPR_B1 falls by ≥ 0.20 under block-B, changes by ≤ 0.05 under block-D, and mean
  TPR on A1–A6 changes by ≤ 0.05 under block-B.
* **Decision.** Supported if ≥ 80 % of the population shows necessity; not supported if ≤ 50 %; else mixed.
* **Meaning.** Supported → beyond-horizon answers are produced through the thought's layer-1 reading of the
  beyond-horizon edges (not by query-time reading of the edge list). Not supported → the answer can bypass the
  frontier (then P7 is a correlate, not the path).

### T2 Frontier rescue (causal sufficiency) — primary

* **Population.** Competent non-extrapolators of the P30 and L test sets with P7 < 0.10 (frozen values).
* **Intervention.** Same probe graphs and rollout. At latent step 6 + j (j = 1…6), in layer 1, each head's attention on
  B-chain edge j (the edge into the node at distance 6 + j) is raised to that head's **natural frontier mass**, and the
  rest of the row is rescaled to sum to 1 (**inject-B**). Natural mass = the same model's mean attention of that head
  at step 6 on A-edge 6 (its last in-horizon frontier edge) on the same graphs, unintervened. If a head already
  attends more, it is left unchanged. Control **inject-C**: the same onto the C chain (unreachable).
* **Measures.** d7 = ½(TPR_B1 + TNR_C1); TPR_Bj and TNR_Cj for j = 1…6.
* **Per model.** Rescued if d7 under inject-B ≥ 0.60 and exceeds base d7 by ≥ 0.10.
* **Decision.** **Supported (propagation account)** if ≥ 60 % of the population is rescued; **not supported
  (the readout is also limited at the horizon)** if ≤ 20 %; else mixed. Secondary: under inject-C, TNR_C1 falls by
  ≥ 0.10 in ≥ 60 % of models (the readout trusts whatever the frontier holds). Dose (0.5×, 2× natural mass) reported.
* **Meaning.** Supported → the failing models' only defect is that the frontier stops; the readout beyond the
  horizon works when the frontier is supplied. Not supported → the training-depth horizon is also in the readout.

### T3 In-training prediction (set point + predictor) — primary

* **Population.** Locally trained competent runs with step checkpoints: P30 (local cards), L (k8), O (both arms, once
  finished), R cap-4 runs are excluded (different horizon).
* **Measure.** P7 (frozen `far_sweep_frozen_I.py --seed 2033`, floor in_peak ≥ 0.03) at the saved checkpoint nearest
  to (full-depth arrival + 2000 steps); full-depth arrival = first logged step whose `max_bdepth` is None/full.
* **Decision.** AUC of early P7 for *final* extrapolation ≥ 0.80 → supported; < 0.65 → not; else inconclusive.
  Secondary: the same at arrival + 0 and + 5000; agreement of the P7 class (≥ 0.10) with the final model.
* **Meaning.** Supported → the horizon crossing is set (and visible) shortly after full depth, joining H3 and H1 and
  giving an in-training diagnostic (which Kuo et al. 2026 lack).

### T4 Predictor head-to-head, prospective — primary

* **Population.** Competent runs of addendum O (both arms) and of T9/T10 (below): none of these has been measured.
* **Predictors.** P7; end activity (EA, PROTOCOL5); two in-distribution instruments in the spirit of Zhang et al.
  2026, computed on in-distribution graphs only (confirm split, depth ≤ 6, no probe chains): **completion time**
  (first step at which the distance-6 frontier excess reaches half its maximum) and **completion at K** (distance-6
  excess at step K / its maximum); in-distribution accuracy (delayed, matched 5- and 6-hop); full-depth arrival step.
* **Decision.** AUC(P7) − max AUC(in-distribution predictors) with model-bootstrap 95 % CI. **Supported (a
  horizon-crossing probe is needed)** if the difference ≥ 0.10 and its CI excludes 0; **not supported
  (in-distribution instruments suffice)** if the best in-distribution AUC ≥ 0.90 and within 0.05 of P7; else
  inconclusive.

### T5 Incremental validity (retrospective, existing P30 ∪ L competent runs)

* Cross-validated AUC (10-fold × 20 repeats, L2 logistic regression) of in-distribution covariates alone (delayed,
  matched 5, matched 6, full-depth step, EA) vs the same plus P7. **Supported** if ΔAUC ≥ 0.10 and the covariate-only
  AUC ≤ 0.80; **not** if ΔAUC < 0.03.

### T6 Multiverse and permutation (retrospective, existing P30 ∪ L competent runs)

* Specifications: probe seed {2033, 7, 11} × probe widths {2/4/6, 2 only, 6 only} × layer {1, 2} × floor {0.02, 0.03,
  0.05} (54). **Supported** if AUC ≥ 0.85 in ≥ 80 % of specifications; permutation p for the frozen specification.

### T7 Independent re-implementation (code audit)

* `scratch/p7_indep.py`, written from the definition without reusing `far_sweep` or `frontier_amp` (edges located by
  matching (src, dst) ids; attention recomputed from cached q/k). **Pass** if Spearman(P7_indep, P7_frozen) ≥ 0.99
  and max |difference| ≤ 0.02 on all P30 ∪ L competent runs. A fail stops all P7 claims until resolved.

### T8 Negative controls

* **Pause-token models** (mode "pause", existing): P7 ≤ 0.10 and no extrapolation expected in all (descriptive).
* **Placebo P7**: the same ratio with the distance-7 edge replaced by C1 minus a random unreachable edge. AUC must lie
  in [0.35, 0.65] on P30 ∪ L competent runs; otherwise the P7 construction carries a confound.

## Tier 2 — new training (local cards after addendum R; cloud if credit allows)

### T9 Architecture generality — primary

* **Families** (addendum-K recipe otherwise: `--no_step_emb --K 6 --lr 5e-4 --amp bf16 --save_every 1000`):
  **L3** `--layers 3` (d 256), **L4** `--layers 4` (d 256), **W512** `--d 512` (2 layers). Seeds 500–515 (L3),
  520–535 (L4), 540–555 (W512). Exploratory basis: Part III `v2_deep` / `v1_wide` models (not in the test).
* **Layer for P7** chosen per model *without outcome data*: the layer with the largest in-horizon excess (in_peak)
  among all layers (near arrays only). `scratch/far_sweep_layers.py` (P7 per layer; frozen before first use).
* **Decision.** Per family with ≥ 3 extrapolating and ≥ 3 non-extrapolating competent runs: AUC(P7) ≥ 0.80 supported,
  < 0.65 not. Pooled over families: supported if AUC ≥ 0.85. H2 in each family: share of non-extrapolators with
  P7 < 0.10 ≥ 60 %. A family with too few competent runs is reported as "not learned by this recipe".

### T10 Graph-family generality

* `--shapes fork --p_fork 0.5` (branches fork inside; frontier holds several nodes per depth), addendum-K recipe,
  seeds 560–575. Same measures and rules as T9 (P7 probes use chains, which fork-trained models also see).

### T11 From-scratch Coconut on ProsQA (toy mechanism in a second format)

* Coconut ProsQA recipe with a 2-layer, 768-wide GPT-2 architecture **trained from scratch** (Rizvi-Martel et al. 2026
  found superposition only from scratch), official question-first format. **Pilot** (1 seed, exploratory): continue
  only if test accuracy ≥ 0.85 at K = 6 and the hop clock increases (J2 rule). **Confirmatory**: 10 seeds.
* **Measures.** Depth extrapolation on the addendum-P long set: accuracy on path lengths 7–8 at K′ = 12
  (extrapolates if ≥ 0.75); ProsQA P7-analog = frontier excess (reachable − decoy statement attention, layer chosen by
  in-horizon peak) at hop 7 / max over hops 1–6, on long-set problems with m = 2.
* **Decision.** AUC(P7-analog) ≥ 0.80 supported (needs ≥ 3 of each outcome), < 0.65 not; H2 analogue: share of
  non-extrapolators with ratio < 0.10 ≥ 60 %.

### T12 Pretrained GPT-2 depth mechanics (after P and Q are scored)

* **Q-deep**: the addendum-G recipe trained on original + m = 2 extended training problems (3–8 hops), 2 seeds.
  If Q shows a wall at the training depth, Q-deep predicts the wall moves to ≥ 9 hops (gap acc(3–8) − acc(9–12) on
  the long set ≥ 0.15 and acc(7–8) ≥ 0.80). Written in full as a dated amendment before launch, only if Q1 is supported.

## Compute and order

| Test | Compute | Where | When |
|---|---|---|---|
| T1, T2, T7, T8 | < 1 GPU-h | local, between runs | tonight / Tue |
| T3, T6 | ~4–6 GPU-h | local 2070 after addendum P | Tue |
| T4, T5 | analysis + T4 measures on O/T9/T10 runs | local | as runs finish |
| T9 (48 runs), T10 (16 runs) | ~100 GPU-h | local after R (Tue night →), cloud if credit | Wed–Fri |
| T11 (1 + 10 seeds) | ~5 GPU-h per seed on a 3090 | cloud (~$11) | after a credit top-up |
| T12 Q-deep (2 seeds) | ~20 GPU-h per seed | cloud (~$8) | only if Q1 supported |

Cut-offs: T9/T10 runs not started by Mon 2026-10-05 00:00 are dropped; T11 seeds not started by the same time are
dropped. A test whose population ends up below its minimum is reported as unresolved, never re-specified.

## Amendment 1 (2026-09-28 ~23:38, before any T1/T2 measurement on a test-population model)

* Only equivalence and manipulation checks have been run, on two exploration models outside every test population
  (`runs/main_s1` with step embedding, `runs/nostep_s0` without): the intervention-free rollout of
  `latent/frontier_causal.py` reproduces `LatentReasoner.latent_step` + `query` exactly (256/256 graphs × 24 queries
  identical), and an injection sets the requested per-head mass exactly (`scratch/test_frontier_causal.py`).
* **T2 natural mass** is the per-head *peak over steps 1–6* of the mean attention on A's distance-6 edge (the text
  above says "at step 6"): 13 of 22 Part V models read a depth over several steps, so step 6 need not be their step
  for that edge. Everything else in T1/T2 is unchanged.
* Frozen code: `latent/frontier_causal.py` as in git commit f77d6ff (local repository; pushed to the private GitHub
  repository once access is set up).
* Test populations, from `results/p30.json` and `results/p_l.json` (frozen P7, extrapolation labels): T1 = competent
  extrapolators; T2 = competent non-extrapolators with P7 < 0.10. Output `results/p7s_causal.json`; scorer
  `scratch/score_p7s_t12.py` (written before the run).

## Amendment 2 (2026-09-28 ~23:42, before any checkpoint was measured for T3)

* T3 population = P30 ∪ L competent runs above the floor that have step checkpoints locally (Vast-trained runs were
  fetched without them). Checkpoint = the saved one nearest to (full-depth arrival + offset), ties to the earlier.
  Full-depth arrival = the logged curriculum advance to "full" (`curriculum_advance_to: null`).
* An early checkpoint whose in_peak is below the floor (0.03) gets P7 = 0 (no measurable frontier to cross) and is
  counted and reported. Code: `scratch/p7s_t3.py` (frozen `far_sweep_frozen_I.measure`, loader pointed at the
  checkpoint).

## Amendment 3 (2026-09-28 ~23:48, after T1, T2 and T3 were scored and T1/T2 explored; before any addendum-O run was measured by any PROTOCOL7 code)

* **Why.** T1 and T2 were "not supported" as pre-registered, and exploration on 5 + 5 of their models
  (`scratch/p7s_explore.py`, `results/p7s_explore.json`) showed why: (i) distance-7 answers come from a one-hop read
  at query time (blocking the query's read of the A6 → B1 edge takes TPR_B1 from ~0.5 to ~0), so the latent frontier
  must be tested on distances ≥ 8, where blocking B in the latent attention does cut answers in extrapolators;
  (ii) T2's "natural mass" (anchored on A's distance-6 edge, which the latent sweep barely reads: ~0.02) was ≈ uniform
  attention; the sharp part of the sweep (depths 2–5) reaches ~0.2–0.3 in one or two heads, and a strong injection
  rescues distance 8–12 answers while the same injection on the decoy chain creates false positives. These are
  exploratory; the tests below are confirmatory on models no PROTOCOL7 code has touched.
* **Population.** Competent, above-floor runs of addendum O (both arms: `o_*`, `ob_*`; local and Vast), measured by
  `latent/frontier_causal2.py` (fresh probe seed 4243, K′ = 12) and the frozen `far_sweep_frozen_I.py --seed 2033`.
  Scorer `scratch/score_p7s_amend3.py` (written now). The same tests are applied to T9/T10 runs as secondary.
* **Positive control.** Blocking A in the first block's latent attention lowers mean TPR on A4–A6 by ≥ 0.20 in
  ≥ 80 % of runs; otherwise T1b–T2b are uninterpretable.
* **T1b (necessity at distances 8–10).** Extrapolators whose base mean TPR(B2–B4) exceeds mean FP(C2–C4) by ≥ 0.10:
  blocking B in every block's latent attention lowers mean TPR(B2–B4) by ≥ 0.15 while blocking D changes it by
  ≤ 0.05. Supported if ≥ 70 % of eligible runs; not if ≤ 40 %.
* **T1c (distance 7 is a query-time hop).** Runs with base TPR_B1 ≥ 0.35: blocking the query's read of A6 → B1
  lowers TPR_B1 by ≥ 0.30 in ≥ 80 % → supported; ≤ 50 % → not.
* **T2b (rescue at natural scale).** Non-extrapolators with P7 < 0.10: head h's attention on B-edge j at step 6 + j is
  raised to its natural mass m_h = max over depths 2–5 and steps 1–6 of its mean attention on A's edge into that depth.
  Rescued if mean TPR(B2–B6) rises by ≥ 0.20 and the same injection on C raises mean FP(C2–C6) by ≥ 0.10.
  Supported (the horizon is a propagation limit; the readout uses a supplied frontier) if ≥ 60 %; not if ≤ 20 %.
* **T13 (P8 predicts extrapolation beyond the query-time hop).** extrap8 = d8 ≥ 0.60 with CI lower bound > 0.55 at
  K′ = 12 (length_gen); P8 = max over steps of the distance-8 far excess / in_peak (frozen far_sweep output). AUC(P8)
  ≥ 0.80 supported, < 0.65 not (needs ≥ 3 of each); AUC(P7) for extrap8 reported.
* **Reporting.** T1 and T2 stay "not supported" as pre-registered; T1b–T2b are new tests, not re-scorings.

## Amendment 4 (2026-09-28 ~23:55, before any T9/T10 run started)

* Cards and precision: the 4-layer and width-512 families do not fit in fp32 next to the 2070's other job (smoke test
  OOM), and the 2070 has no bf16. So **T9 (L3, L4, W512) runs on the RTX 3070 and 4060 in bf16**
  (`scratch/p7s_t9_{3070,4060}.txt`, families interleaved, seeds split 8/8 per family per card) and **T10 (fork) on the
  RTX 2070 in fp32** (`scratch/p7s_t10_2070.txt`; precision had no effect in P27/P28). Each queue is chained after that
  card's current queue (R on 3070/4060, O on 2070). Smoke runs (60 steps) passed for L3 and fork.
* Per-layer P7: `scratch/far_sweep_layers.py` (its layer-1 P7 equals the frozen far_sweep exactly on nostep_s0:
  0.90114); layer chosen per model by largest in_peak.
* If credit allows, T9 seeds may be moved to Vast.ai A4000/4070 cards (bf16) by a dated amendment before they start.

## Amendment 5 (2026-09-29 ~00:01, before any addendum-O run was measured)

* T4's in-distribution instruments were defined on the distance-6 frontier, which the exploration after T1/T2 showed
  the latent sweep barely reads (the last in-range hop is a query-time read). To avoid a strawman, the T4 set of
  in-distribution predictors is widened: completion time and completion-at-K at depth 5 **and** 6, sweep reach
  (deepest depth whose peak excess ≥ 25 % of the in-range maximum), EA, delayed / matched 5 / matched 6 accuracy and
  full-depth step; each scored in its better direction, and P7 must beat the **best** of them (maximum recomputed in
  each bootstrap resample, so the comparison is conservative for P7). Rules otherwise as T4. Code `scratch/p7s_t4.py`.

## Amendment 6 (2026-09-29 ~00:49, before the T11 pilot was launched)

* **T11 pilot (exploratory).** `prosqa/prosqa_s11p.yaml`: the Coconut ProsQA recipe (50 epochs, effective batch 128,
  fp32, standard question-first format, no answer mask) with `from_scratch: True, n_layer: 2` (GPT-2 architecture,
  768 wide, 12 heads, random init via `prosqa/patch_scratch.py`) and **lr 3e-4** (Coconut's 1e-4 is a fine-tuning
  rate; exploratory choice). Pod script `prosqa/pod_prosqa_s.sh`: trace and K′ ∈ {0, 1, 2, 6, 12} on the test set,
  `long_eval2.py --no_mask` on the addendum-P long and control sets, and `p7_prosqa.py --m 4 --Kp 12` (per-statement
  thought attention on extension-4 problems; the correct option's chain vs the wrong option's look-alike chain).
  Card: RunPod RTX A4500 (≤ $0.34/h) if available, else Vast.
* **Continuation rule** (as T11): confirmatory seeds only if the pilot reaches ≥ 0.85 at K = 6 and its hop clock
  increases. The ProsQA P7 analogue's layer rule and floor are fixed in a dated amendment after the pilot and before
  any confirmatory seed; the confirmatory seeds' outcomes decide T11, the pilot's do not.

## Amendment 7 (2026-09-29 ~01:18, before any T9 run started)

* To use the Vast RTX 4070 (vast3, $0.144/h) that finishes its addendum-O queue around 03:00, the first six seeds of the
  RTX 4060 T9 queue (`t9l3_s508`, `t9l4_s528`, `t9w512_s548`, `t9l3_s509`, `t9l4_s529`, `t9w512_s549`) run there
  instead (bf16, same recipe; `scratch/p7s_t9_vast3.txt`, chained after `queue_o_vast3` on that instance). They are
  marked claimed in `scratch/p7s_t9_4060.txt.claimed` so the local worker skips them. Fetched with
  `scratch/vfetch_toy.sh <host> <port> t9`. No other change to T9.

## Amendment 8 (2026-09-29 15:39 EDT, before any of the runs below started; placement only)

* The user added Vast credit to run faster in parallel. The 42 T9 runs not yet started (all 24 of the RTX 3070 queue,
  and the 18 unclaimed runs of the RTX 4060 queue) move to four Vast hosts, A–D. Same recipe and bf16; queues
  `scratch/p7s_t9_vA.txt`..`vD.txt` (10–11 runs each, families interleaved), run by `scratch/queue_worker2.sh`.
  They are marked claimed in the local `.claimed` files, so the local T9 workers skip them. Fetched with
  `scratch/vfetch_toy.sh <host> <port> t9`.
* The freed local cards join T10 after addendum R finishes: the RTX 3070 (`cuda:1`) and later the RTX 4060 (`cuda:0`)
  run further workers on the same shared queue `scratch/p7s_t10_2070.txt`. Each line already carries `--amp fp32`, so
  the registered precision is unchanged; only the card differs.
* No change to any rule, seed, measure or cut-off. Mixed hardware within a family already applied under amendment 7.

## Amendment 9 (2026-09-29 23:49 EDT, instrument fix; no T10 length_gen outcome existed)

* `latent/length_gen.py` (the long-chain probe behind the PROTOCOL5 extrapolation rule used by T9/T10) crashed on
  every fork-graph run. The probe asks `sample_graphs` for chain branches (`p_chain = 1`), but a fork run sets the
  global graph shape to "fork", which overrides that, so the probe's structural assertion failed. Every
  `t10fork_*` evaluation logged `FAIL results/length_gen_…`, so no T10 extrapolation value exists.
* **Fix.** The probe graphs are built under the default shape, and the run's shape setting is restored afterwards.
  This matches the probe's own definition ("every branch a 6-node chain"). For default-shape runs the graphs are
  byte-identical to before (checked: src/dst/start equal, seed 99, n 1024), so T9 and all earlier results are
  unaffected.
* **Handling.** `length_gen` is re-run with the fixed code for every finished T10 run before T10 is scored, and later
  T10 runs use the fixed code. Nothing else changes.
