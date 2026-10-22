# MATCHED-BUDGET: C3 at matched compute, text-CoT vs latent self-training lineages (draft v0, 2026-10-22; $0, eval-only on existing files)

**Origin.** Audit's north-star lead (C3 = "at matched compute, equal or better accuracy than written CoT"). §219 (Note 219-13) closed CROSSED on depth (n50 text 20.16 vs latent 20.54), with a descriptive TOKEN-COST row. This draft turns that row into a registered C3 reading.

## 0. Already measured (the "unmeasured" rule; checked 10-22)
- **Latent, s216_r2_s1:**
  - eval_deep.json: per-item answers at every K′ 0..28, n 13–24, 1,000 items per n (test seed 17100);
  - eval_vb.json: acc_k at K′ 0..16 and 24, n 1–16.
- **Text, s219_t2_s1:** eval_text.json per item: P (prompt tokens), ntok (generated tokens), answer, y, overflow; n 1–24, the same test_set(1000, 24) items.
- **Disclosure: the direction is partly known.** 219-13's descriptive TOKEN-COST (n 17–20) showed:
  - sequential steps per correct answer, text vs latent: 438–1,115 vs 19–33;
  - token-positions per correct answer: 491–1,227 vs 1,012–2,072, so latent uses ≈ 1.7–2.1× MORE.
  - This draft pre-states readings on both axes knowing those numbers. The registered value is the full budget curve and its labels, not discovery of the direction.
- **Nothing new is computed:** every reading below is a function of the stored per-item files. There is no new GPU eval.

## 1. Two compute axes (both registered; neither is privileged without the other)
- **(S) Sequential steps** (latency / depth of computation):
  - latent: K′ passes;
  - text: generated tokens (each one decode forward).
  - The prompt prefill counts as 1 step in each arm (one parallel forward).
- **(F) Token-position compute** (a FLOPs proxy):
  - latent: K′ × T_item (each pass re-processes all T_item = len(syms) + 2 slots; no cache);
  - text: P + ntok (prefill once plus one cached position per generated token).
  - For GPT-2 small at T ≤ 1,024 the per-position MLP + projection FLOPs dominate attention. Attention-inclusive FLOPs are reported descriptively with the exact formula for both arms (VB's extra bank / SLVA attention included).
- **Truncation (censoring rule):** an item is correct at budget B only if it is correct and its own cost is ≤ B.
  - Latent: K′ ≤ B_S (or K′ · T ≤ B_F), at the smallest such K′ whose coda answer is correct and stays correct at every larger read K′ ≤ the cap. A late-flipping answer counts only from the K′ where it becomes and stays right.
  - Text: free-generation answer correct and the item's cost ≤ B.
  - Unparseable, non-terminating or CTX-overflow text = wrong at every B.
  - The latent cap is K′ 28, so for B_S > 28 latent accuracy stays at its K′ 28 value (never extrapolated).

## 2. Readings (n 13–24, each n separately and pooled)
- **Curves:** acc_A(n, B) on grids B_S ∈ {1..28, 32, 48, 64, 128, 256, 512, 1024} and B_F ∈ a log grid 2^5..2^16.
- **B\*_A(n, q):** the smallest B with acc ≥ q, for q ∈ {0.5, 0.9} ("> grid" if never).
- **Labels, per axis X ∈ {S, F}:**
  - **C3-MATCHED-X:** latent acc ≥ text acc − 0.02 at every B on the grid, at every n 13–20 where either arm reaches 0.5 by the largest B.
  - **TEXT-AHEAD-X:** text acc ≥ latent acc + 0.05 at some B for ≥ 3 of the n 13–20.
  - **MIXED-X:** otherwise. The crossing budget is reported per n.
  - **Prediction (from §0's known rows):** C3-MATCHED-S, and TEXT-AHEAD-F or MIXED-F.
- **Registered C3 statement:** worded per axis, e.g. "latent matches or beats text-CoT at matched sequential steps; at matched token-position compute text is ahead by ≈ 2× at n 17–20". There is no single "C3 met" claim unless both axes read MATCHED.
- **Scope:** one lineage each, GPT-2 small, Arithmetic(n, 11), round-2 models, a proxy compute measure (F). Never "compute-matched" without the axis named.

## 3. OC
- Per-n accuracies come from 1,000 items. At p ≈ 0.5 the SE is ≈ 0.016, so a paired difference has SE ≤ 0.022. The 0.02 / 0.05 margins are about 1 and 2.3 SE.
- Over 8 n × ~40 grid points, multiplicity makes C3-MATCHED (a dominance claim over all points) conservative. Under exact equality of the arms, P(C3-MATCHED) is small (one point a > 0.02 deficit breaks it). TEXT-AHEAD needs ≥ 3 n at +0.05.
- Under a true 20× step advantage (§0), C3-MATCHED-S passes with P ≈ 1. Under a true 2× compute disadvantage, TEXT-AHEAD-F passes with P ≈ 1.
- The readings are near-deterministic given the known direction. The design's value is the registered budget curves and B\*, and the honest two-axis statement.

## 4. Cost and checks
- $0 (CPU, ≤ 2 threads).
- Scorer theory/score_mb.py, with fixtures:
  - synthetic files where latent dominates / text dominates / they cross;
  - the "stay correct" rule;
  - text truncation.
- Field paths grepped against the writers (vbank_eval part_deep, cot eval_text).
- Known-answer check: at B_S ≥ max(ntok), text acc equals eval_text acc_by_n, and latent at K′ = n + 2 equals eval_deep acc_n2.

## 5. Novelty (for Literature)
- N1: "a registered matched-budget comparison (sequential steps and token-position compute, both reported) between matched text-CoT and latent-recurrent self-training lineages".
- Known priors to check: Coconut / CODI token-efficiency claims; Ouro 2510.25741; "compute-optimal" latent-vs-CoT analyses.
