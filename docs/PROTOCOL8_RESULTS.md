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
* **Shortcut caveat.** coco's small D710 edge (0.56) may partly be the ProsQA proximity cue (THEORY §57). The
  pre-registered secondary SC (amendment 3: p8_na_test) is evaluated next, locally.
