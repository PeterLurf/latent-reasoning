# PROTOCOL 7 — results

Pre-registration: `docs/PROTOCOL7.md` (tests T1–T13, amendments 1–3). Every verdict below is the pre-registered one;
exploratory results are labelled.

| Test | Population | Result | Verdict |
|---|---|---|---|
| **T1** frontier necessity at distance 7 (block B in layer-1 latent attention) | 25 competent extrapolators (P30, L) | TPR on B1 drops ≥ 0.20 in 0/25 (it rises slightly on average) | **not supported** |
| **T2** frontier rescue at "natural mass" (anchored on the distance-6 edge) | 33 non-extrapolators, P7 < 0.10 | rescued 0/33; the injected mass (~0.010–0.017 per head) is ≈ uniform attention | **not supported** (manipulation ≈ null; see below) |
| **T3** P7 shortly after full-depth arrival predicts final extrapolation | 34 locally trained competent runs (12 extrapolate) | AUC 0.93 at arrival + 2000 steps (0.91 at arrival, 0.95 at + 5000); P7 class agrees with the final model in 91 % | **supported** |

Sources: `results/p7s_causal.json`, `results/p7s_t12.json` (T1/T2; code `latent/frontier_causal.py`, git f77d6ff);
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
