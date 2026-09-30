# Readable vs usable alternatives in latent reasoning

Execution of the research proposal *"Readable versus usable alternatives in latent
reasoning"*: can several graph-search alternatives held in one continuous-thought state
be manipulated separately, and how does that selectivity change with branching width?

* `docs/NORTH_STAR.md` — **the goal: a latent reasoner meeting seven conditions (the thoughts do the work, scale past training, match text CoT, need no step labels, decide when to think and write, stay stable, and are faithfully readable in human language).**
* `docs/CURRENT_TASKS.md` — **live handoff: what is running now, where, how to check/fetch/stop it, what awaits scoring.**
* `DO_NOT_MOVE_WHILE_RUNNING.md` — **read before moving anything: paths that live experiments depend on (until ~2026-09-29 12:00).**
* `report/findings.md` — consolidated write-up: what stands after Parts I–VI, what was revised, what is open.
* `docs/README.md` — index of every doc: pre-registrations and scorecards by part, literature, historical plans.
* `report/README.md` — index of the report pages and how they are built.
* `docs/RESEARCH_LOG.md` — chronological decisions, negative pilots, deviations, disclosures.
* `docs/RESEARCH_TIMELINE.md` — how the research question moved across Parts I–VI, a claim ledger, next steps.
* `docs/PROTOCOL.md` … `docs/PROTOCOL6.md` — pre-registrations (append-only), one per part; scorecards
  `docs/PROTOCOL4_RESULTS.md`, `PROTOCOL5_RESULTS.md`, `PROTOCOL6_RESULTS.md`.
* `docs/LITERATURE_NOTES.md` (short positioning) and `docs/LITERATURE.md` (annotated bibliography).
* `results/` — evaluation and confirmation outputs; `prosqa/` — the Coconut GPT-2 / ProsQA transfer experiments.

## Results

**Part I.** Recurrent seeds reach 90–93% balanced accuracy on held-out families. Single-branch
donor edits on the shared latent state move the edited branch more than the others and weaken
with width. Probe read-out directions (filters) are causally inert; the activation patterns for
the same node ids carry the effect (readable ≠ usable). Branches compete for a shared budget.
(Part I's single-seed control conclusions were later found to be seed luck; see Part II.)

**Part II** (fresh confirmation). Dose does not rescue high-width edits; per-branch amplitude falls
as W^−β (β ≈ 0.6–0.9); at higher width 2–3-hop nodes move to query-time graph reading; edits have
recipient-graph semantics; recurrence buys reliability on the deepest queries (depth-matched).

**Part III.** In the delayed format every recurrent model runs a clocked breadth-first sweep in
layer 1 (one depth per one or two latent steps; blocking a depth's edges at its step, and only
then, cuts the branch's deeper nodes). Width halves per-edge frontier attention per doubling while
the clock stays fixed. Sweeps that finish inside the budget stop; a seed whose sweep is still
running, and a model trained with only four steps, follow chains beyond any trained length when
given more steps. With the target given first, the raw-feedback variant walks backward from the
target (width-independent). Wider and deeper variants kept more in the latent pathway, contrary to
their pre-registered predictions (P11, P12 failed). Two host crashes during the night are
documented in the research log; an integrity re-run found the stored results intact.

**Part IV** (PROTOCOL4, pre-registered addenda A–E). Neither account of the failure to extrapolate
survived: removing the step embedding (P19, P21) or randomising the budget (P22) did not produce
extrapolation, and a sweep that finishes inside the budget did not prevent it (P20, addendum G);
4 of 11 competent runs extrapolate with no consistent relation to the manipulated factors or to sweep
depth. The sweep's attention budget is split per frontier edge, not per branch (P25; the Part III width
law is the frontier-size law), but the single-step clock holds in only ~9 of 22 competent models (P23,
P24 fail). 11 of 22 runs failed the competence gate (counting the full-depth clause), mostly by late-curriculum
stalls; stage-1 stalls come with removing the step embedding and vanish at lr 5e-4 (stall pilots).

**Part V** (PROTOCOL5, addenda A–C). Whether a model extrapolates is visible inside its training budget: *end
activity* — layer-1 frontier attention at the last trained step relative to its peak — separated extrapolators among
the 22 existing models (AUC 0.81) and, frozen at 0.25, predicted 12 new models (1 of 6 below vs 5 of 6 above, AUC
0.83; P26 confirmed). The step-locked clock holds in models that read each depth in one step (qualified on discovery
graphs: P23′/P24′ 7/9 on fresh graphs, 3/4 on new models; 1–2/13 in the rest). A late-stage restart rule saves ≤ 8%;
halving the learning rate gave 7/8 vs 5/8 gate passes in a 16-run pilot (R1 holds, p = 0.28), with somewhat lower
deep-query accuracy.

**Part VI** (PROTOCOL6, addenda A–O; in progress on 2026-09-28). Whether a sweep stays open is decided late in the
curriculum and is mostly fixed at full depth (continuation 11/12 stable; branches agree 6/6 from full-depth entry);
training precision changes neither extrapolation nor curriculum progress (P27, P28 not supported). Among open
sweeps, extrapolation is decided by whether the layer-1 frontier crosses the trained horizon at test time (P7): a fixed
cut-off failed prospectively (P29), but as a floored ranking P7 predicted 42 new competent runs across five cards with
AUC 0.993 (end activity 0.673; P30 supported); with two spare thoughts failing sweeps still stop at the training depth
(L1, L2 supported). On ProsQA the official Coconut GPT-2 ignores its thoughts; with the bypass blocked or the query
delayed they carry an answer computed in one forward pass (no hop-by-hop sweep, 2–3 seeds each), and a 4-layer GPT-2
does not learn the delayed task. Running: training depth 5 (M), longer lookup scaffold (O), ProsQA J seed 2.

Full reports: `report/latent_branch_report.html` (Parts I–III, built by `report/build_report.py`) and
`report/protocol4_report.html` / `.pdf` / `.md` (Part IV, built by `report/build_protocol4_report.py`);
Part IV scorecard in `docs/PROTOCOL4_RESULTS.md`; Part V scorecard in `docs/PROTOCOL5_RESULTS.md`; Part VI scorecard
in `docs/PROTOCOL6_RESULTS.md` (no report page yet); consolidated write-up in `report/findings.md`.

## Code (`latent/`)

| File | Role |
|---|---|
| `graphs.py` | Fixed-budget branching graphs (10 branches × 6 nodes, width W attached to start, rest to a shape-matched decoy hub), exact BFS solver, family hashing and splits, single-token counterfactual edits |
| `data.py` | Balanced reachability queries; auxiliary PARENT/CHILDREN lookup targets (training scaffold) |
| `model.py` | 2-layer decoder with Coconut-style continuous-thought recurrence, pause-token variant, forked queries on a shared saved prefix, thought-patching hook |
| `graphstep.py` | CUDA-graph captured training step |
| `train.py` | Training (gated or fixed depth curriculum, annealed lookup scaffold) |
| `tests.py` | Instrument validation (generator vs BFS, counterfactual locality, cache/fork equivalences, identity patch) |
| `competence.py` | Stage A: accuracy by latent budget, depth, width, with graph-level bootstrap |
| `probes.py` | Linear id-indexed reachability probes, depth-matched decoy AUC |
| `interventions.py` | Donor replacement h' = h + UUᵀ(h_don − h), effect matrices, controls, upfront-format effects |
| `study.py` | Discovery (probes, site selection, PCA/pattern subspaces) and confirmation (frozen protocol) |
| `mechanism.py` | Exploratory, discovery-split: donor-difference energy per subspace, attention budget per branch |
| `aggregate.py` | Merges competence + confirmation across seeds into `results/summary.json` |

### Part II / III additions

| File | Role |
|---|---|
| `flex.py` | Flexible forward pass: thought edits at any step, KV overrides by position and reader, attention key bias |
| `followup.py`, `followup_analysis.py`, `followup_run.py` | Part II batteries (dose, swap, continuation, cache routes, attention, gain, controllability, amplification) |
| `followup_confirm.py`, `protocol3_confirm.py` | Scorers for `docs/PROTOCOL2.md` and `docs/PROTOCOL3.md` (held-out seeds scored separately) |
| `competence_matched.py` | Depth-matched balanced accuracy (reachable vs same-depth decoy nodes) |
| `budget_sweep.py`, `length_gen.py`, `length_gen_upfront.py` | Test-time budget sweeps and chains longer than any in training |
| `upfront_trace.py`, `upfront_block.py`, `delayed_block.py` | Attention traces of the latent sweep / backward walk and step-specific edge blocking |
| `rewind.py`, `rewind_grid.py`, `rewind_multi.py`, `phase_shift.py` | Feeding earlier thoughts back into later steps (search-state tests) |
| `development.py` | Route shares, competence and sweep schedule across training checkpoints |

### Part IV additions

| File | Role |
|---|---|
| `graphs.py` `--shapes fork` | Fork-mode graphs: each graph all chains (frontier = W) or all two-armed forks (frontier = 2W past the split) |
| `protocol4_confirm.py` | Scorer for `docs/PROTOCOL4.md` addenda A–D (P19–P22, G, I/J) |
| `frontier_amp.py` | Per-graph layer-1 frontier attention vs width and frontier size (exploratory) |
| `confirm_sweep.py` | Addendum E: fresh-graph blocking, width traces and fork test (P23–P25), with scorer |
| `train.py --K_random lo:hi`, `--amp`, `--shapes` | Budget sampled per batch (randk); precision control for cross-card runs; graph shape mode |

Part IV scripts in `scratch/`: `score_p4_variants.py`, `rescore_gateJ.py`, `frontier_fork_ratio.py`,
`stall_pilot.py`, `local_resume.sh` / `local_progress.sh` (local queues), and the RunPod helpers
`pod_setup.sh`, `pod_after_deploy.sh`, `pod_reap.sh`, `pod_fetch.sh`, `pod_status.sh`.

### Part V additions

| File | Role |
|---|---|
| `protocol5_confirm.py` | PROTOCOL5: strict-clock measurements (addendum C), end activity (P26), scorer for R1 / P26 / P23′ / P24′ |
| `confirm_sweep.py` `split=` | Traces and blocking on any split (discovery qualification, fresh confirm seeds) |
| `protocol4_confirm.py` | Now applies addendum D's full-depth clause to the gate (`gate_accuracy` keeps the old rule) |

Part V scripts in `scratch/`: `extrap_predictors.py` (step-2 feature screen), `stall_rule.py` (late-stage restart
rule replay), `queue_worker.sh` / `queue_worker2.sh` (shared local job queue, `rel_queue.txt`); `stage1_guard.py`
gained an opt-in `--late T` rule.

### Part VI additions

| File | Role |
|---|---|
| `protocol6.py` | PROTOCOL6: end activity / strict clock / checkpoint trajectories (`measure`, `traj`) and scorer for addenda A, B, E, F, H |
| `train.py --save_every`, `--amp fp32`, `--depth_cap`, `--aux_steps` | Checkpoint every N steps; fp32 without TF32 (same-card precision pairs); cap training graph depth (addendum M); lookup-scaffold length (addendum O) |
| `prosqa/` (outside `latent/`) | Official Coconut GPT-2 on ProsQA: speed patch, answer mask, delayed-query and layer-truncation patches with pre-training tests, pod scripts, frozen analyses (`analyze.py`, `gpt2c.py`) and scorer (`score.py`) |

Part VI scripts in `scratch/`: `queue_worker2.sh` + `stage1_guard.py` + `eval_battery.sh` (local queues),
`chain_after.sh` / `p6_after2.sh` (chaining and measurement loops), `far_sweep_frozen_I.py` (P7, frozen at addendum I),
`score_p29.py`, `score_p30.py`, `score_l.py` (scorers), `sharpness.py` (exploratory).

`train.py` keeps a full resumable training state (`--resume`) and writes checkpoints atomically;
`scratch/stage1_guard.py` applies the PROTOCOL3 stage-1 restart rule and `scratch/chain.py` queues jobs.

`scratch/` holds the exploratory one-off scripts referenced in the research log
(`explore_subspace.py`, `explore_pattern.py`, `attn_share.py`, `pipeline.py`, ...).

## Reproduce

```
python -m latent.tests
python -m latent.train --mode coconut --K 6 --bs 512 --steps 30000 --warmup 300 --aux_steps 10000 --aux_anneal 5000 --curriculum gated --init gpt2 --children_weight 1 --seed 0 --device cuda:0 --out runs/r3_pc_gated_gpt2
python -m latent.train --mode pause   --K 6 ... --curriculum 1:1000,2:5000,3:5000,4:5000 --out runs/pause_s0
python -m latent.train --mode coconut --K 0 ... --curriculum 1:1000,2:5000,3:5000,4:5000 --out runs/k0_s0
python -m latent.competence runs/<run> ... --out results/competence.json
python -m latent.study discover runs/<run>
python -m latent.study confirm  runs/<run> --protocol docs/protocol_frozen.json
```

Seed-0 main run lives in `runs/r3_pc_gated_gpt2` (its name predates the seed-numbered runs
`runs/main_s1`, `runs/main_s2`). Hardware used: RTX 4060, RTX 3070 (bf16 autocast),
RTX 2070 (fp32), 8 GB each; ~2 GB peak per run at batch 512. Part IV also used RunPod community
RTX A4000 / A4500 / RTX 4000 Ada pods, Part V an RTX 4090 and two RTX A4000 pods (torch 2.14.0+cu126; see
`runs/<run>.host.txt`); Part VI RunPod and Vast.ai RTX 3090 / RTX A5000 / RTX A4000 / RTX 4070 instances
(ProsQA and extra toy-model seeds; `scratch/pods.txt`).
