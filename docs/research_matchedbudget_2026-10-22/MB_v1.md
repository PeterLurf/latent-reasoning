# MATCHED-BUDGET: C3 at matched compute, text-CoT vs latent self-training lineages (v1, 2026-10-22; $0, eval-only on existing files)

**Status:** v1 = draft v0 (42b17829) + the 2256a04e proxy correction + Audit's assumption requirement + Literature f55c7e5c. Approved in principle by Audit ($0).

**Origin.** Audit's north-star lead (C3 = "at matched compute, equal or better accuracy than written CoT"). §219 (Note 219-13) closed CROSSED on depth (n50 text 20.16 vs latent 20.54), with a descriptive TOKEN-COST row. This draft turns that row into a registered C3 reading.

## 0. Already measured (the "unmeasured" rule; checked 10-22)
- **Latent, s216_r2_s1:**
  - eval_deep.json: per-item answers at every K′ 0..28, n 13–24, 1,000 items per n (test seed 17100);
  - eval_vb.json: acc_k at K′ 0..16 and 24, n 1–16.
- **Text, s219_t2_s1:** eval_text.json per item: P (prompt tokens), ntok (generated tokens), answer, y, overflow; n 1–24, the same test_set(1000, 24) items.
- **Disclosure: the direction is partly known.** 219-13's descriptive TOKEN-COST (n 17–20) showed:
  - sequential steps per correct answer, text vs latent: 438–1,115 vs 19–33;
  - "token-positions" per correct answer: 491–1,227 vs 1,012–2,072.
- **Correction (10-22):** that position count treated every latent pass as a full 12-block forward.
  - In VB only the 4 core blocks (6–9) loop per pass (boxes/vbank.py:117, CORE = (6, 7, 8, 9)). The prelude (0–5) runs once, and the coda (10–11) once at readout.
  - Counting in block-positions, a rough check at n 17 gives latent T·(8 + 4K′) ≈ 53 × 84 ≈ 4.5k vs text 12 × 491 ≈ 5.9k per correct answer, so latent ≈ 0.76× text, not 2× more.
  - This rough number (one n, mean T) was seen before registration and is disclosed.
- **Nothing new is computed:** every reading below is a function of the stored per-item files. There is no new GPU eval.

## 1. Two compute axes (both registered; neither is privileged without the other)
- **(S) Sequential steps** (latency / depth of computation):
  - latent: K′ passes;
  - text: generated tokens (each one decode forward).
  - The prompt prefill counts as 1 step in each arm (one parallel forward).
- **(F) Block-position compute** (a FLOPs proxy; every GPT-2 block has the same per-position dense cost):
  - latent: T_item · (6 + 4·K′ + 2) (prelude once, 4 core blocks per pass, coda once), T_item = len(syms) + 2;
  - text: 12 · (P + ntok) (prefill plus one cached position per generated token).
  - **Assumption, stated (Audit):** no latent cache is possible. Every pass rewrites every slot's state, so recomputing all T slots in the core blocks IS the computation (as in any depth-recurrent model). This is a property of the method, not an implementation shortcut. Text reuses its KV cache because earlier tokens never change.
  - VB's extras (bank attention over up to K′ versions per slot, SLVA, gather) are not in the dense proxy. They are reported in an exact attention-inclusive FLOPs row for both arms, per n, descriptively. If they change any (F) label, that is stated.
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
  - **Prediction:** C3-MATCHED-S (P ≈ 0.95); (F) open, with a rough lean to latent ≤ text (the n 17 check), so C3-MATCHED-F has P ≈ 0.5.
- **Registered C3 statement:** worded per axis, e.g. "latent matches or beats text-CoT at matched sequential steps; at matched token-position compute text is ahead by ≈ 2× at n 17–20". There is no single "C3 met" claim unless both axes read MATCHED.
- **Scope:** one lineage each, GPT-2 small, Arithmetic(n, 11), round-2 models, a proxy compute measure (F). Never "compute-matched" without the axis named.

## 3. OC
- Per-n accuracies come from 1,000 items. At p ≈ 0.5 the SE is ≈ 0.016, so a paired difference has SE ≤ 0.022. The 0.02 / 0.05 margins are about 1 and 2.3 SE.
- Over 8 n × ~40 grid points, multiplicity makes C3-MATCHED (a dominance claim over all points) conservative. Under exact equality of the arms, P(C3-MATCHED) is small (one point a > 0.02 deficit breaks it). TEXT-AHEAD needs ≥ 3 n at +0.05.
- Under a true 20× step advantage (§0), C3-MATCHED-S passes with P ≈ 1. (F) is closer: about 0.76× at n 17 by the rough check, so its label carries real uncertainty, and B\* is the informative reading.
- The readings are near-deterministic given the known direction. The design's value is the registered budget curves and B\*, and the honest two-axis statement.

## 4. Cost and checks
- $0 (CPU, ≤ 2 threads).
- Scorer theory/score_mb.py, with fixtures:
  - synthetic files where latent dominates / text dominates / they cross;
  - the "stay correct" rule;
  - text truncation.
- Field paths grepped against the writers (vbank_eval part_deep, cot eval_text).
- Known-answer check: at B_S ≥ max(ntok), text acc equals eval_text acc_by_n, and latent at K′ = n + 2 equals eval_deep acc_n2.

## 5. Novelty (Literature f55c7e5c)
- N1 **OK**, scoped to GPT-2, the synthetic task and our lineages.
- No compute-matched latent-vs-CoT comparison was found. Prior efficiency claims count generated tokens (Coconut 2412.06769, CODI 2502.21074) or latency.
- The nearest is LOTUS 2606.31779 (latency-framed, not budget-matched), cited as such.
- Ouro 2510.25741 and iso-depth 2604.21106 scale recurrence against parameters or depth, not against CoT.
