# PROTOCOL8 — results

Scored 2026-10-01 07:34 EDT with the registered scorer `prosqa/score_p8.py` (sha256 8a151ebe…, unchanged), run through
`prosqa/run_score_p8.py`. The wrapper only adds a JSON converter for numpy scalars: the registered file crashed on
serialising a numpy bool; no computation changed. Output: `results/p8.json`. All 16 runs completed:
* 4 seeds × coco / at / rt / cot, drained with verified fetch; no Vast host left.
* at_s4's evaluation was re-run after a container reboot (note in PROTOCOL8.md).

**Primary and secondary predictions.**

| prediction | verdict | numbers (per seed) |
|---|---|---|
| P8-1 at vs coco (D710) | **not supported** | at 0.496/0.503/0.500/0.490 vs coco 0.583/0.559/0.566/0.550; mean diff −0.067, MW p 1.0 |
| P8-1 rt vs coco (D710) | **not supported** | rt 0.484/0.506/0.524/0.503; mean diff −0.060, MW p 1.0 |
| P8-2 (D11) | inconclusive (at +0.008, rt +0.001) | every latent arm 0.47–0.51 |
| P8-3 LU ≥ 0.15 in ≥ 3/4 | not met (coco 2/4; at, rt 0/4) | coco LU 0.180/0.147/0.173/0.143; at 0.07–0.08; rt 0.09–0.12 |
| learnability (competent = acc m0–6 K′=10 ≥ 0.75) | coco 0/4, at 0/4, rt 0/4; cot 3/4 | |
| P8-4 C3 | **not competitive** | best latent arm coco D710 0.564 vs CoT 0.762; CoT ≈ 84 generated tokens vs 10 latent steps |
| P8-5 signature | not met | K′sat: coco 2, at 0, rt 0 in every seed |

**Reading.**
* No latent arm learns deep ProsQA. D710, D11 and D12p are at or near chance in every latent seed.
* Anytime supervision and random-tight budgets make things worse, not better: lower in-range accuracy (SHAL 0.66–0.72 vs
  coco 0.75–0.78) and lower latent use (LU).
* Text CoT reaches 0.71–0.81 at every depth, including past the trained depth (D11 0.73–0.81, D12p 0.77–0.81).
* **Shortcut reference line (Audit).** Every arm is below the proximity heuristic's 0.896 at D710, CoT included
  (0.762). None of these accuracies alone shows deep reasoning from the root, and CoT's C3 advantage (P8-4) could partly
  be cue use. The amendment-3 re-score on p8_na_test (heuristic 0.500) decides it.
* **Shortcut caveat.** coco's small D710 edge (0.56) may partly be the ProsQA proximity cue (THEORY §57). The
  pre-registered secondary SC (amendment 3: p8_na_test) is evaluated next, locally.


## Amendment 3: shortcut-free re-score on p8_na_test (secondary; scored 2026-10-01 16:32 EDT)

Scorer `prosqa/score_p8_na.py` (47a4335, committed before any NA eval) → `results/p8_na.json`. 16/16 files (4 seeds ×
coco/at/rt/cot). Proximity heuristic: 0.896 (D710) / 0.902 (D11) on p8_test, **0.500 / 0.500 on p8_na_test**.

| arm | D710 orig → NA | D11 orig → NA | cue reliance D710 (per seed) |
|---|---|---|---|
| coco | 0.564 → **0.508** | 0.496 → 0.508 | +0.059 / +0.061 / +0.069 / +0.038 |
| at | 0.497 → 0.486 | 0.504 → 0.520 | 0.000 / +0.014 / +0.008 / +0.024 |
| rt | 0.504 → 0.501 | 0.497 → 0.502 | −0.011 / +0.001 / +0.014 / +0.008 |
| cot (descriptive; scorer lists files only) | 0.762 → **0.767** | 0.778 → 0.784 | −0.007 / +0.008 / −0.023 / 0.000 |

Secondary MW (on NA): at vs coco D710 −0.022 p 1.0 not supported; D11 +0.012 p 0.17 inconclusive. rt vs coco D710
−0.007 p 0.90, D11 −0.006 p 0.76, both not supported. Label "advantage cue-dependent": not applicable (neither at nor rt
had a P8-1 advantage).

**Reading.**
* coco's small D710 edge over chance was the proximity cue: every seed loses 0.04–0.07 on NA and lands at chance (0.497–0.524).
* With the cue removed, **every latent arm is at chance at every depth**.
* Text CoT does not use the cue (cue reliance −0.005). P8-4 therefore stands, and more strongly, on cue-free data:
  CoT 0.767 vs the best latent arm 0.508 at D710.
* The CoT column is descriptive, because the registered scorer has no CoT comparison; it uses the same depth bands,
  from eval_p8's by_path_len.
