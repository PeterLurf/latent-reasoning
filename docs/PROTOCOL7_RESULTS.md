# PROTOCOL 7 — results

Pre-registration: `docs/PROTOCOL7.md` (tests T1–T13, amendments 1–7). Every verdict below is the pre-registered one;
exploratory results are labelled.

| Test | Population | Result | Verdict |
|---|---|---|---|
| **T1** frontier necessity at distance 7 (block B in layer-1 latent attention) | 25 competent extrapolators (P30, L) | TPR on B1 drops ≥ 0.20 in 0/25 (it rises slightly on average) | **not supported** |
| **T2** frontier rescue at "natural mass" (anchored on the distance-6 edge) | 33 non-extrapolators, P7 < 0.10 | rescued 0/33; the injected mass (~0.010–0.017 per head) is ≈ uniform attention | **not supported** (manipulation ≈ null; see below) |
| **T3** P7 shortly after full-depth arrival predicts final extrapolation | 34 locally trained competent runs (12 extrapolate) | AUC 0.93 at arrival + 2000 steps (0.91 at arrival, 0.95 at + 5000); P7 class agrees with the final model in 91 % | **supported** |
| **T5** P7 adds to in-distribution covariates (retrospective) | 59 competent P30 + L runs (25 extrapolate) | cross-validated AUC: covariates (delayed, matched 5/6, full-depth step, EA) 0.68 → + P7 0.92 (Δ 0.24); single AUCs: matched 5 0.73, matched 6 0.73, EA 0.66, delayed 0.62, full-depth step 0.34 (earlier = better), P7 0.98 | **supported** |
| **T6** multiverse (54 specifications) | same 59 runs | AUC ≥ 0.85 in 43/54 = 79.6 % (bar 80 %); layer 1 (P7's layer): all 27 specifications 0.95–0.99; layer 2: 0.79–1.0 (median 0.875); 3 layer-2 specifications had no run above the floor and count as misses (43/51 = 84 % without them); frozen specification AUC 0.976, permutation p < 0.0001 | **not supported** (by 0.4 %; the misses are all layer 2) |
| **T7** independent re-implementation | same 59 runs | Spearman 1.000, max |Δ| 2.4e-6 | **pass** |
| **T8a** pause-token models | 3 pause models | no extrapolation (d7 0.50); in_peak 0.0003–0.0013, 25–100× below the floor, so P7 is undefined for them (raw ratios 0.4–1.2 are noise) | as expected (floor excludes) |
| **T8b** placebo P7 (C1 − random unreachable edge) | same 59 runs | AUC 0.75 (band 0.35–0.65) | **fail** — see below |
| positive control (amendment 3): blocking chain A cuts A4–A6 by ≥ 0.20 | 44 competent addendum-O runs (new) | 44/44 | ok |
| **T1b** frontier necessity at distances 8–10 (block B in every block's latent attention) | 12 eligible O extrapolators | 11/12 | **supported** |
| **T1c** distance 7 is a query-time hop (block the query's read of A6 → B1) | 18 O runs with TPR_B1 ≥ 0.35 | 18/18 | **supported** |
| **T2b** rescue at natural scale (per-head mass of the model's own sharp frontier) | 24 O non-extrapolators, P7 < 0.10 | 20/24 rescued (B2–B6 up ≥ 0.20 and decoy injection raises false positives ≥ 0.10) | **supported** |
| **T13** P8 predicts extrapolation to distance 8 | 44 O runs, 8 extrapolate at d8 | AUC 0.98 (P7: 0.97) | **supported** |
| **T4** prospective head-to-head vs in-distribution predictors | 44 O runs, 20 extrapolate | P7 0.99; best of 10 in-distribution instruments 0.82 (completion time at depth 6); EA 0.53; accuracies 0.54–0.65; difference 0.17, 95 % CI [0.05, 0.26] | **supported** |
| **T9** P7 generality across architecture (3 layers, 4 layers, 512 wide) | 44 of 48 runs trained (4 stage-1 failures), 24 competent, 21 above the floor | pooled AUC 0.89, 95 % CI [0.71, 1.00], permutation p 0.0017 (Holm ≤ 0.0085; 14/21 extrapolate); 512 wide AUC 1.00, 4 layers 0.71, 3 layers unresolved (all 5 extrapolate) | **supported** (pooled); per family supported / inconclusive / unresolved |
| **T10** P7 on fork graphs | 16 runs, 10 competent above the floor | AUC 1.00, CI [1.00, 1.00], permutation p 0.004 (5/10 extrapolate); non-extrapolators with P7 < 0.10: 5/5 | **supported** |

Sources: `results/p7s_t5.json`, `results/p7s_t6.json`, `results/p7_indep.json`, `results/p7s_t8.json`; `results/p7s_causal.json`, `results/p7s_t12.json` (T1/T2; code `latent/frontier_causal.py`, git f77d6ff);
`results/p7s_t3.json`, `results/p7s_t3_score.json` (T3; `scratch/p7s_t3.py`).

## T9 / T10 (scored 2026-09-30 07:43, `scratch/score_p7s_t9.py`, `results/p7s_t9.json`)

| Family | Finished / competent / above floor | Extrapolate | AUC(P7) | H2 (P7 < 0.10 among non-extrapolators) | Verdict |
|---|---|---|---|---|---|
| 3 layers | 15 / 7 / 5 | 5 | — | 0/0 | **unresolved** (no non-extrapolator) |
| 4 layers | 16 / 11 / 10 | 6 | 0.71 | 4/4 | **inconclusive** (0.65–0.80); H2 supported |
| 512 wide | 13 / 6 / 6 | 3 | 1.00 | 3/3 | **supported** |
| fork graphs (T10) | 16 / 10 / 10 | 5 | 1.00 | 5/5 | **supported** |
| T9 pooled | 21 above floor | 14 | 0.89 (bar 0.85) | — | **supported** |

- **Layer choice (by in_peak, fixed in advance):** mostly layer 1 (26/31). Layer-1 P7 gives AUC 0.67 for 4 layers, 0.89
  for 512 wide and 1.00 for fork.
- **Fork caveat:** the extrapolation probe and P7's probe graphs are chain-shaped, so for fork-trained models they are
  outside the training shape distribution (PROTOCOL7 amendment 9).
- **Placement:** amendments 7–8 put runs on Vast and local cards with mixed hardware; 4 T9 runs were stage-1 failures
  (t9l3_s501, t9w512_s547, t9w512_s549, t9w512_s552).
- **General-rule statistics** (`scratch/p7s_t9_stats.py`, `results/p7s_t9_stats.json`; model bootstrap of 5000, 10 000
  permutations, seed 0; added after Audit noted the gap):
  - T9 pooled: CI [0.71, 1.00], p 0.0017. Holm across T1–T4 and T9 is bounded by 5p = 0.0085, so T9 survives whatever
    the other four p-values are. T1–T4 permutation p-values were never computed; that is a reporting gap, not needed
    for this bound.
  - T10: CI [1.00, 1.00], p 0.004.
  - 4 layers: AUC 0.71, CI [0.32, 1.00], p 0.17.
  - 512 wide: AUC 1.00, CI [1.00, 1.00], but p 0.052 with n = 6. "Supported" follows the AUC ≥ 0.80 rule; on its own
    it is weak evidence.
- **Not yet applied:** the amendment-3 causal tests and T4, registered as secondary for T9/T10.

## T1 / T2 — what went wrong, and what exploration showed (exploratory, 5 + 5 models)

`scratch/p7s_explore.py` → `results/p7s_explore.json`, on the first five models of each T1/T2 population:

* **The intervention works (positive control).** Blocking chain A in the first block's latent attention cuts the
  in-range deep answers (e.g. p30_s114: A4–A6 0.90 / 0.74 / 0.65 → 0.35 / 0.12 / 0.15). The in-range latent frontier
  is causally necessary for deep in-range nodes.
* **Distance 7 is answered at query time, not by the latent frontier.** Blocking only the query tokens' attention to
  the single edge A6 → B1 drops TPR on B1 from ~0.4–0.55 to 0.00–0.06 in every model, while blocking the latent
  attention to the B chain leaves B1 untouched. The query reads its parent edge and checks whether the parent is in
  the thought — one hop beyond the latent frontier. The distance-6 edge is likewise barely read by the latent sweep
  (peak head attention ~0.02 vs ~0.2–0.3 for depths 2–5), so the last in-range hop is also a query-time read.
* **Distances 8+ need the latent frontier.** In extrapolators, blocking B in the latent attention of both blocks cuts
  B2–B5 (p30_s114: 0.50 / 0.49 / 0.48 / 0.29 → 0.22 / 0.16 / 0.13 / 0.14; p30_s105 and s117 similar; s116, whose
  base B2+ was low, unchanged).
* **The failing models' readout uses a frontier if one is supplied.** Injecting the B chain at a fixed per-head mass
  of 0.3–0.5 raises B2–B6 from ~0.1–0.2 to 0.6–1.0 in 4 of 5 non-extrapolators (p30_s109: 0.09 → 0.86–0.88 at 0.3,
  ~1.0 at 0.5); the same injection on the unreachable C chain makes C nodes "reachable" just as often. The readout
  trusts the frontier's content; nothing in it stops at the training depth. That mass is ~1.5× the natural peak of
  the model's own frontier heads and applied to all 8 heads, so it is stronger than natural (tested at natural scale in
  T2b).
* **Consequence for the pre-registered design.** T1's outcome node (distance 7) was the one node the latent frontier
  does not produce; T2's reference edge (distance 6) was the one in-range edge the sweep does not read. Amendment 3
  (T1b, T1c, T2b, T13) re-specifies both on models not yet measured (addendum O).
* **Consequence for the account.** The horizon is a *propagation* limit: the latent sweep stops advancing, the query
  adds one hop, and the readout would use a deeper frontier. d7 ("extrapolation") partly measures that final
  query-time hop; T13 tests whether P8 predicts extrapolation beyond it.

## T8b — what the placebo failure means

The placebo's numerator (attention on the unreachable decoy's distance-7 edge minus a random unreachable edge) is
itself somewhat higher in extrapolators (AUC 0.79), but tiny: 0.0010 vs 0.0005, against 0.0455 vs 0.0024 for P7's
numerator (reachable minus decoy distance-7 edge; AUC 0.98 alone); the two correlate 0.44. So extrapolators attend a
little more to *any* edge continuing a depth-6 chain end, reachable or not — a weak structural component ~2 % the
size of the reachability signal. P7 subtracts exactly this decoy term, so the component works against P7, not for it.
Pre-registered verdict stands: fail (the construction is not confound-free); practical reading: P7's signal is
reachability-specific to within ~2 %.

## T11 pilot — not learned (2026-09-29 ~07:00; `prosqa/res_s11p/`, checkpoint `prosqa/ckpt/s11p_checkpoint_50`)

2-layer, 768-wide GPT-2 architecture trained from scratch with the Coconut ProsQA recipe (lr 3e-4, 50 epochs, GPT-2 BPE
tokens, no answer mask; Vast 53306310, destroyed after a checksum-verified fetch). Final-stage validation 0.46–0.50;
test accuracy 0.536 / 0.550 / 0.522 / 0.514 / 0.542 at K′ = 0 / 1 / 2 / 6 / 12 — chance at every budget, no
increasing hop clock (layer clocks [2,3,1,4,3,3] and [1,1,1,1,1,2]). By the pre-registered continuation rule T11
stops: no confirmatory seeds with this recipe. Difference from Rizvi-Martel et al. 2026 (96 % for 2L-768d from
scratch): they used a 40-token symbolic vocabulary; here the model must also learn GPT-2 BPE embeddings for the
invented concept names in 50 epochs. The remaining pilot analyses (long sets, P7 extractor) were stopped as moot.

## Amendment-3 tests and T4 (scored 2026-09-29 ~10:20 on the addendum-O runs, none measured before)

`results/p7s_amend3.json`, `results/p7s_causal2.json`, `results/far_sweep_o_s2033.json`, `results/p7s_t4.json`.
The exploratory account from T1/T2 replicates on 44 new models: the latent frontier is causally necessary for answers
beyond one hop past the horizon (T1b) and sufficient to produce them when supplied at the model's own natural scale
(T2b) — the readout of failing models uses a frontier they never build; distance 7 is a query-time hop (T1c); the
frontier's reach at distance 8 predicts extrapolation there (T13); and the horizon-crossing probe beats every
in-distribution instrument prospectively (T4). Together: **the training-depth horizon is a propagation limit of the
latent sweep, visible and causal in the frontier.**

## Independent out-of-sample check by the theory session (THEORY.md §21, commit 3ff40d8; 2026-09-29)

Pre-registered before any O run was measured by that session: 17 competent, locally checkpointed addendum-O runs
(8 extrapolate). **Early P7 by the T3 rule** (frozen far_sweep copy, checkpoint nearest full-depth arrival + 2000,
P7 = 0 below the floor) reaches **AUC 0.92**, a second out-of-sample confirmation of T3 on new runs. The theory
session's stage-3 retrieval-share marker reaches 0.51 (0.78 in-sample): difference −0.40, 95 % CI [−0.71, −0.11].
Its own rule (PO1) was unresolved (2 runs above threshold; ≥ 5 needed), and it downgraded the marker to in-sample
only. Exploratory: the long scaffold does not change the stage-3 retrieval share systematically (0.16 vs 0.22, lower in
13 of 23 pairs), and most O failures fail by local routes (fading or restarting fronts), consistent with the O1 null
and the slack account (§20's anytime test is pending).

## Bookkeeping (2026-09-29 ~13:00, after the audit)

* T9 seed `t9w512_s549` (moved to vast3 by amendment 7) is not missing: it failed stage 1 three times ("relaunch
  budget spent", guard log `logs/vast_174.0.242.177_1743/guard_t9w512_s549.log`). Like every stage-1 failure, it counts
  as a finished, non-competent run; it is not re-queued. vast3 was destroyed at ~09:08 after checksum-verified fetches
  of its O runs (~07:08) and T9 runs (~09:08). All Vast create/destroy events are now in `scratch/pods.txt`.
* The theory session corrected its THEORY.md timestamps (commit 973fb4f). Commit times for citation: §20 Pilot E +
  D-1 at 11:49:57, first E run at 11:55:24; E-1 and §21 at 11:50:43; §21 scored at 11:54:27. The O outcomes were
  public before §21, so §21 is out of sample for its marker only; the early-P7 comparison (PO3) there is a re-analysis
  of O runs whose labels were already known.

