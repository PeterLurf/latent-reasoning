# PROTOCOL 7 — results

Pre-registration: `docs/PROTOCOL7.md` (tests T1–T13, amendments 1–5). Every verdict below is the pre-registered one;
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

Sources: `results/p7s_t5.json`, `results/p7s_t6.json`, `results/p7_indep.json`, `results/p7s_t8.json`; `results/p7s_causal.json`, `results/p7s_t12.json` (T1/T2; code `latent/frontier_causal.py`, git f77d6ff);
`results/p7s_t3.json`, `results/p7s_t3_score.json` (T3; `scratch/p7s_t3.py`).

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
