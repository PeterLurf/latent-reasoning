# Part IV — PROTOCOL4: the stop, the budget, and the sweep's attention budget

Compiled 2026-09-25 from `results/` (build: `report/build_protocol4_report.py`). Pre-registration: `docs/PROTOCOL4.md` addenda A–E; chronology and every deviation: `docs/RESEARCH_LOG.md`; running scorecard: `docs/PROTOCOL4_RESULTS.md`. Parts I–III: `report/latent_branch_report.html`.

## 1. Summary

* **Neither leading account of the failure to extrapolate survived.** Taking the step index away (P19, P21) or randomising the budget (P22) did not make models extrapolate, and finishing the sweep inside the budget (P20, addendum G) did not stop them. 4 of 11 competent runs extrapolate at K′ = 16 (budget_k8_s1, nostep_k8_s1, budget_k10_s3, v3_rawfb_s1), with no consistent relation to the manipulated factors or to sweep depth E.
* **Mechanism, refined:** the sweep's layer-1 attention budget is split among frontier *edges*, not branches. At fixed width, doubling the frontier halves per-edge attention (P25: 14/18 primary models, 4/4 exploratory models on fresh graphs). The Part III width law is the frontier-size law seen through W.
* **Mechanism, weakened:** the single-step clock (P23, P24) holds in only about 9 of 22 competent models; the rest read a depth over several latent steps, so 'a clocked sweep in every seed' is one strategy among several.
* **Reliability is the bottleneck:** 11 of 22 PROTOCOL4 runs failed the competence gate, almost all by stalling at curriculum stages 2–4, which the stage-1 restart rule does not cover. P21 ended unresolved at its five-seed limit for this reason.
* **Operations:** 12 runs on 8 RunPod community pods (A4000 / A4500 / RTX 4000 Ada, $0.17–0.20/h), total $2.68; everything else on three local cards; every pod deleted after checksum-verified fetch.

## 2. Questions and design

Part III found that recurrent models run a clocked breadth-first sweep in layer 1 and that sweeps which finish inside the latent budget stop, while one seed and a K = 4 model followed chains beyond any trained length. PROTOCOL4 asked what keys the stop:

* **Halt account:** the stop is keyed to the step embedding, so removing it (P19: K = 6; P21: K = 8) or sampling K per batch (P22: K ∈ 4..8) should let competent models extrapolate.
* **Finish-early account** (Zhang et al. 2607.20594): a sweep that reaches full depth (E ≥ 4) inside the budget does not extrapolate (P20: K = 8; addendum G: K = 10).
* **Variant replications:** P13 (pre-LN feedback), P12/P17 (four layers), addendum H (wider thought).
* **Mechanism confirmations (addendum E):** P23 blocking is step-locked; P24 width changes amplitude, not the clock; P25 the attention budget is split per frontier edge.

Base recipe for every run: Coconut-style recurrence, 2 layers, d = 256, batch 512, 30k steps, gated depth curriculum, annealed PARENT/CHILDREN scaffold, GPT-2 init. Gate (addendum J): overall delayed balanced accuracy ≥ 0.85 and depth-matched 5-hop ≥ 0.60. 'Extrapolates' = distance-7 accuracy ≥ 0.60 with bootstrap lower bound > 0.55. E = deepest layer-1 frontier depth reached at any step k ≤ 6.

| Addendum | Written | Content |
|---|---|---|
| A | before any run | P19–P21 |
| B | 2026-09-24 23:40 | cloud hardware; precision held to the original cards |
| C | 2026-09-25 ~00:00 | which runs train where; predictions restated |
| D | ~00:45, before any evaluation | scoring battery, contingencies, P22 (randk) |
| E | ~13:40, before its code existed | P23–P25; discloses the earlier exploratory frontier test |

## 3. Training outcomes

| Run | Final stage | Attempts | Delayed | Matched-5 | Gate | E | d7 @K′12 | d7 @K′16 | Hardware | Precision |
|---|---|---|---|---|---|---|---|---|---|---|
| nostep_s1 | 3 | 1 | 0.801 | 0.578 | ✘ | 2 | 0.502 | 0.508 | RTX 2070 (local) | fp32 |
| nostep_s2 | full | 1 | 0.928 | 0.741 | ✔ | 3 | 0.523 | 0.528 | RTX A4000 (RunPod) | fp32 |
| nostep_s3 | 4 | 2 | 0.849 | 0.611 | ✘ | 2 | 0.512 | 0.518 | RTX A4500 (RunPod) | fp32 |
| nostep_s4 | full | 2 | 0.903 | 0.698 | ✔ | 4 | 0.514 | 0.510 | RTX 2070 (local) | fp32 |
| nostep_s5 | 4 | 2 | 0.854 | 0.528 | ✘ | – | 0.499 | 0.498 | RTX 2070 (local) | fp32 |
| budget_k8_s1 | full | 1 | 0.956 | 0.786 | ✔ | 5 | 0.699 | 0.701 | RTX 4060 (local) | bf16 |
| budget_k8_s2 | 3 | 1 | 0.823 | 0.543 | ✘ | 2 | 0.513 | 0.513 | RTX 4060 (local) | bf16 |
| nostep_k8_s0 | full | 1 | 0.905 | 0.701 | ✔ | 3 | 0.519 | 0.530 | RTX 3070 (local) | bf16 |
| nostep_k8_s1 | full | 2 | 0.913 | 0.741 | ✔ | 3 | 0.709 | 0.708 | RTX 3070 (local) | bf16 |
| nostep_k8_s2 | 2 | 1 | 0.707 | 0.531 | ✘ | – | 0.496 | 0.498 | RTX 3070 (local) | bf16 |
| nostep_k8_s3 | 3 | 1 | 0.810 | 0.585 | ✘ | 2 | 0.501 | 0.501 | RTX 3070 (local) | bf16 |
| nostep_k8_s4 | full | 1 | 0.903 | 0.564 | ✘ | – | 0.497 | 0.497 | RTX 4060 (local) | bf16 |
| randk_s0 | full | 1 | 0.879 | 0.639 | ✔ | 4 | 0.488 | 0.487 | RTX A4000 (RunPod) | bf16 |
| randk_s1 | 4 | 1 | 0.876 | 0.619 | ✘ | 2 | 0.538 | 0.548 | RTX A4500 (RunPod) | bf16 |
| randk_s2 | full | 1 | 0.897 | 0.677 | ✔ | 3 | 0.563 | 0.569 | RTX 4000 Ada Generation (RunPod) | bf16 |
| budget_k4_s3 | 4 | 1 | 0.852 | 0.589 | ✘ | 1 | 0.511 | 0.509 | RTX A4500 (RunPod) | bf16 |
| budget_k10_s1 | full | 1 | 0.909 | 0.786 | ✔ | 4 | 0.495 | 0.495 | RTX 4000 Ada Generation (RunPod) | bf16 |
| budget_k10_s2 | full | 1 | 0.916 | 0.623 | ✔ | 1 | 0.503 | 0.509 | RTX A4500 (RunPod) | bf16 |
| budget_k10_s3 | full | 1 | 0.890 | 0.699 | ✔ | 2 | 0.621 | 0.657 | RTX 4000 Ada Generation (RunPod) | bf16 |
| v1_wide_s2 | 2 | 2 | 0.711 | 0.524 | ✘ | – | 0.501 | 0.501 | RTX A4000 (RunPod) | bf16 |
| v2_deep_s1 | 3 | 1 | 0.823 | 0.546 | ✘ | 5 | 0.491 | 0.491 | RTX A4000 (RunPod) | bf16 |
| v3_rawfb_s1 | full | 2 | 0.940 | 0.790 | ✔ | 3 | 0.668 | 0.667 | RTX A4000 (RunPod) | bf16 |

_Attempts > 1 = stage-1 restarts under the guard. Final stage below 'full' at 30k counts as a gate failure (addendum D)._

Runs that ended below full curriculum depth: nostep_s1, nostep_s3, nostep_s5, budget_k8_s2, nostep_k8_s2, nostep_k8_s3, randk_s1, budget_k4_s3, v1_wide_s2, v2_deep_s1. The stage-1 restart rule cannot catch these; each cost a full 30k-step run.

## 4. Verdicts

| Prediction | Evidence | Reading |
|---|---|---|
| P19 no step embedding → extrapolates at K′=12 | competent: nostep_s2, nostep_s4 | **not replicated** |
| P20 K=8: E ≥ 4 and no extrapolation at K′=16 | budget_k8_s1: gate ✔, E 5, d7 0.701; budget_k8_s2: gate ✘, E 2, d7 0.513 | **fails** |
| P21 K=8 + no step embedding (decisive cell) | 5 seeds, 0 scorable | **unresolved** (stopping rule reached) |
| P22 randk → extrapolates at K′=16 | randk_s0: d7 0.487; randk_s1: d7 0.548; randk_s2: d7 0.569 | **fails** |
| Addendum G budget_k10_s1 | gate ✔, E 4, d7 0.495 | pass |
| Addendum G budget_k10_s2 | gate ✔, E 1, d7 0.509 | fail |
| Addendum G budget_k10_s3 | gate ✔, E 2, d7 0.657 | fail |
| Addendum I/J budget_k4_s3 | gate ✘ | unscored |
| P13 pre-LN feedback: β toward 1, competition persists | β 0.728 (range 0.62–0.87), competition ✔ | **fails** |
| P12 / P17 four layers | gate ✘ (0.823) | unscored |
| Addendum H wider thought, seed 2 | gate ✘ (0.711) | unscored |
| P23 blocking is step-locked | 9/22 models pass | **fails** |
| P24 width changes amplitude, not clock | 9/22 models pass | **fails** |
| P25 budget per frontier edge (primary) | 14/18 models pass | **mixed** |
| P25 replication, s0–s3 | 4/4 models pass | **confirmed** |

## 5. Extrapolation is not predicted by the manipulated factors or by E

_[Figure: Competent PROTOCOL4 runs. The finish-early account predicts no point above the line inside the shaded band; the halt account predicts every no-step and randk run above the line. Neither pattern appears. — see protocol4_report.html]_

Extrapolating runs differ in step embedding (present / absent), budget (K = 6, 8, 10) and architecture; their same-recipe siblings mostly do not extrapolate. Among competent runs, 3 of 8 with K ≥ 8 extrapolate versus 1 of 3 with K = 6 — far too few runs to read, but a cheap pre-registration target. The working reading is that extrapolation is a seed-level outcome; comparing extrapolating and non-extrapolating siblings (nostep_k8_s0 vs s1; budget_k10_s3 vs s1/s2) is the natural next analysis.

## 6. The sweep's attention budget is split per frontier edge

**Exploratory (before addendum E; disclosed there).** On default graphs, frontier size already varies at fixed W. A per-depth regression of log excess per-edge attention on log W and log n (frontier edges) gave b_n ≈ −0.8 and b_W ≈ −0.2 to −0.5 at depths 1–2 in the four sweeping reference models (`latent/frontier_amp.py`). A first pooled fit on raw attention had pointed the other way (b_W −1.21 on main_s1); the (W, n) table showed a clear 1/n pattern and the fit was replaced. A controlled test on fork-mode graphs (`--shapes fork`: frontier exactly W in chain graphs, 2W past the split in fork graphs) gave fork/chain ratios of 0.49–0.68 in s0–s3, with competence kept on fork graphs (0.87–0.92).

| Model | d1 W2 | d1 W4 | d1 W8 | d2 W2 | d2 W4 | d2 W8 | d3 W2 | d3 W4 | d3 W8 |
|---|---|---|---|---|---|---|---|---|---|
| r3_pc_gated_gpt2 | 0.53 | 0.59 | 0.68 | 0.50 | 0.49 | 0.50 | 0.49 | 0.49 | 0.49 |
| main_s1 | 0.57 | 0.63 | 0.64 | 0.50 | 0.52 | 0.53 | 0.52 | 0.52 | 0.53 |
| main_s2 | 0.57 | 0.60 | 0.61 | 0.52 | 0.54 | 0.59 | 0.54 | 0.57 | 0.59 |
| main_s3 | 0.57 | 0.58 | 0.59 | 0.56 | 0.54 | 0.58 | 0.58 | 0.56 | 0.58 |

_Exploratory fork/chain per-edge excess ratios (results/frontier_amp_fork.json). Per-edge budget → 0.5; per-branch → 1.0._

**Confirmation (addendum E, held-out confirm families, fresh seeds):**

_[Figure: P25: mean fork/chain ratio at depths 2–3 per model. The pass rule (≤ 0.70) applies to each depth separately, so budget_k8_s1 (0.74 at depth 2) and nostep_k8_s1 (0.85 at depth 3) fail although their means sit near the line. — see protocol4_report.html]_

## 7. The step-locked clock holds in a subset of models

| Model | Clock k*(1..3) | P23 diag | P23 off | P23 | P24 same clock | edge W8/W2 | total W8/W2 | P24 | P25 d2 | P25 d3 | fork acc | P25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_s1 | [1, 2, 3] | ✔ | ✔ | ✔ | ✔ | 0.18 | 0.70 | ✔ | 0.52 | 0.53 | 0.89 | ✔ |
| r3_pc_gated_gpt2 | [2, 3, 4] | ✔ | ✔ | ✔ | ✔ | 0.24 | 0.94 | ✔ | 0.50 | 0.49 | 0.87 | ✔ |
| main_s2 | [1, 4, 5] | ✔ | ✔ | ✔ | ✘ | 0.22 | 0.86 | ✘ | 0.60 | 0.57 | 0.91 | ✔ |
| main_s3 | [1, 2, 3] | ✔ | ✔ | ✔ | ✔ | 0.24 | 0.95 | ✔ | 0.56 | 0.58 | 0.92 | ✔ |
| v1_wide_s0 | [1, 3, 6] | ✔ | ✘ | ✘ | ✘ | 0.25 | 0.98 | ✘ | 0.58 | 0.55 | 0.90 | ✔ |
| v2_deep_s0 | [2, 3, 4] | ✔ | ✔ | ✔ | ✔ | 0.25 | 0.99 | ✔ | 0.54 | 0.55 | 0.96 | ✔ |
| v3_rawfb_s0 | [1, 5, 6] | ✘ | ✘ | ✘ | ✘ | 0.17 | 0.66 | ✘ | 0.48 | 0.41 | 0.85 | ✔ |
| budget_k4_s0 | [1, 2, 4] | ✔ | ✘ | ✘ | ✘ | 0.27 | 1.04 | ✘ | 0.56 | 0.90 | 0.91 | ✘ |
| budget_k4_s2 | [3, 4, 5] | ✔ | ✘ | ✘ | ✔ | 0.32 | 1.25 | ✔ | 0.53 | 0.54 | 0.89 | ✔ |
| budget_k8_s0 | [2, 3, 5] | ✔ | ✘ | ✘ | ✘ | 0.26 | 1.02 | ✘ | 0.62 | 0.66 | 0.91 | ✔ |
| nostep_s0 | [1, 5, 6] | ✘ | ✘ | ✘ | ✔ | 0.17 | 0.65 | ✔ | 0.55 | 0.47 | 0.84 | ✔ |
| nostep_s2 | [2, 4, 5] | ✔ | ✘ | ✘ | ✘ | 0.23 | 0.89 | ✘ | 0.53 | 0.54 | 0.91 | ✔ |
| nostep_s4 | [1, 2, 3] | ✔ | ✔ | ✔ | ✔ | 0.15 | 0.60 | ✘ | 0.52 | 1.92 | 0.89 | ✘ |
| budget_k8_s1 | [1, 5, 6] | ✘ | ✘ | ✘ | ✘ | 0.13 | 0.52 | ✘ | 0.74 | 0.55 | 0.93 | ✘ |
| nostep_k8_s0 | [1, 3, 8] | ✔ | ✘ | ✘ | ✘ | 0.16 | 0.62 | ✘ | 0.55 | 0.53 | 0.88 | ✔ |
| nostep_k8_s1 | [1, 2, 3] | ✔ | ✘ | ✘ | ✘ | 0.17 | 0.68 | ✘ | 0.55 | 0.85 | 0.90 | ✘ |
| randk_s0 | [1, 2, 3] | ✔ | ✔ | ✔ | ✔ | 0.24 | 0.96 | ✔ | 0.50 | 0.51 | 0.84 | ✔ |
| randk_s2 | [1, 5, 6] | ✘ | ✘ | ✘ | ✔ | 0.18 | 0.69 | ✔ | 0.48 | 0.36 | 0.87 | ✔ |
| budget_k10_s1 | [1, 2, 3] | ✔ | ✔ | ✔ | ✔ | 0.21 | 0.82 | ✔ | 0.52 | 0.52 | 0.89 | ✔ |
| budget_k10_s2 | [2, 7, 10] | ✔ | ✘ | ✘ | ✘ | 0.21 | 0.84 | ✘ | 0.49 | 0.56 | 0.89 | ✔ |
| budget_k10_s3 | [1, 2, 3] | ✔ | ✘ | ✘ | ✘ | 0.20 | 0.79 | ✘ | 0.47 | 0.46 | 0.87 | ✔ |
| v3_rawfb_s1 | [3, 4, 6] | ✔ | ✔ | ✔ | ✘ | 0.21 | 0.83 | ✘ | 0.53 | 0.53 | 0.91 | ✔ |

_results/confirm_sweep_verdicts.json. P23 diag: Δ ≤ −0.05 at k*(x); off: |Δ| ≤ 0.02 at |k − k*| ≥ 2. P24 bands: edge 0.10–0.45, total 0.6–1.6._

The width amplitude part of P24 holds in all 22 models (per-edge W8/W2 0.13–0.32). What fails is the clock: in 13 models blocking two or more steps away from the argmax still moves the outcome (up to 0.24), and in 12 the argmax step shifts with W. Post hoc (exploratory): models split into regular clocks (one step per depth, the same at every width: s0, s1, s3, v2_deep_s0, randk_s0, budget_k10_s1, nostep_s4) and irregular ones (clocks like [1,5,6], [1,3,8], [2,7,10]) that read a depth over several steps. The frozen qualifying criterion (argmax ≥ 1.5× decoy) admitted all 22 models, so single-step predictions were scored on multi-step readers. The readings stand as scored; a retest needs a stricter criterion fixed in a new addendum.

## 8. Training reliability

_[Figure: Stall pilots (TRAINING_TODO Phase 0). Each run stops at its stage-1 advance or at step 4000 with the real 30k-step schedule. — see protocol4_report.html]_

Stage-1 stalls come with removing the step embedding: 3/6 at the base learning rate, 1/6 with warmup 1000, 0/6 with lr 5e-4 (0/12 with either lower-lr config; one-sided Fisher p ≈ 0.025 against base), and 0/12 on the recipe with a step embedding. The pilot does not address the late-stage stalls behind most gate failures (§3). Per TRAINING_TODO, a changed recipe is for runs that answer new questions, not replications.

## 9. Addendum-J re-score of earlier models (sensitivity)

| Tag | Run | Delayed | Matched-5 | Old gate | J gate |
|---|---|---|---|---|---|
| s0 | runs/r3_pc_gated_gpt2 | 0.900 | 0.666 | ✔ | ✔ |
| s1 | runs/main_s1 | 0.918 | 0.670 | ✔ | ✔ |
| s2 | runs/main_s2 | 0.926 | 0.785 | ✔ | ✔ |
| s3 | runs/main_s3 | 0.925 | 0.836 | ✔ | ✔ |
| s4 | runs/dev_s4 | 0.819 | 0.538 | ✘ | ✘ |
| s5 | runs/main_s5 | 0.891 | 0.584 | ✔ | ✘ |
| v1 | runs/v1_wide_s0 | 0.924 | 0.713 | ✔ | ✔ |
| v2 | runs/v2_deep_s0 | 0.965 | 0.905 | ✔ | ✔ |
| v3 | runs/v3_rawfb_s0 | 0.884 | 0.603 | ✔ | ✔ |
| v1b | runs/v1_wide_s1 | 0.718 | 0.507 | ✘ | ✘ |
| k4 | runs/budget_k4_s0 | 0.926 | 0.620 | ✔ | ✔ |
| k10 | runs/budget_k10_s0 | 0.836 | 0.547 | ✘ | ✘ |
| k4b | runs/budget_k4_s1 | 0.691 | 0.565 | ✘ | ✘ |
| k4c | runs/budget_k4_s2 | 0.916 | 0.601 | ✔ | ✔ |
| k8 | runs/budget_k8_s0 | 0.928 | 0.719 | ✔ | ✔ |
| nostep | runs/nostep_s0 | 0.871 | 0.641 | ✔ | ✔ |

_Only main_s5 changes. Under J it is not competent, so the addendum-I held-out replication would be unscorable on seed 5 instead of failed; the held-out replication then rests on seed 3 alone (seed 4 failed the old gate too). No other PROTOCOL3 verdict used s5 as a competent model._

## 10. Deviations and disclosures

* The frontier-size-versus-width test ran on discovery graphs before its addendum; addendum E discloses it and treats s0–s3 as replications only.
* randk_s1 ended at curriculum stage 4. The first version of `latent/protocol4_confirm.py` passed it on accuracy alone; the scorer now applies the addendum-D rule (below full depth = gate failure), which removes randk_s1 from the competent set and leaves P22 unchanged (PROTOCOL5 plan step 1, 2026-09-25).
* The first fork design (shape drawn per branch) tied frontier size more tightly to W than default graphs; it was replaced by a per-graph shape before any use.
* Local runs died at 00:42 with the assistant session that launched them; resumed from checkpoints (exact resume, logs truncated to the checkpoint). Duplicate evaluation queues ran on three pods for ~15 min; results were unaffected (outputs are skip-if-valid).
* Cloud runs used torch 2.14.0+cu126 on driver-550/580.119 hosts; hosts with driver 580.178 failed cuInit and were discarded unused (addendum C).

## 11. Next steps, by value

* Fix late-stage stalls before training more seeds: a stage 2–4 restart rule, or the lr 5e-4 recipe for new-question runs.
* Treat extrapolation as seed-level: compare extrapolating and non-extrapolating siblings mechanistically.
* Re-test P23/P24 under a new addendum with a strict single-step-reader criterion fixed before blocking.
* Phase 4 (Coconut GPT-2 on ProsQA) — the only item needing a rented 24 GB card; only after the reliability fix.

## 12. Reproduce

* Score: `python -m latent.protocol4_confirm`, `PYTHONPATH=. python scratch/score_p4_variants.py`, `python -m latent.confirm_sweep run && python -m latent.confirm_sweep score`, `PYTHONPATH=. python scratch/rescore_gateJ.py`.
* Frontier: `python -m latent.frontier_amp <runs> --out results/frontier_amp.json`, `PYTHONPATH=. python scratch/frontier_fork_ratio.py`.
* Pilots: `python scratch/stall_pilot.py <device>` and `... <device> - nostep 10,11,12,13,14,15 <configs> --K 6 --no_step_emb`.
* Training queues: `scratch/local_resume.sh` (local, launch detached), `scratch/pod_setup.sh` / `pod_after_deploy.sh` / `pod_reap.sh` / `pod_fetch.sh` (RunPod over the SSH proxy).
* This report: `python report/build_protocol4_report.py`.
