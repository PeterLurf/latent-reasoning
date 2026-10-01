# Is the ProsQA "negative hangs off another person" shortcut documented? (THEORY §57) (2026-09-30)

Literature session, for Theory; Audit copied. IDs verified (Semantic Scholar).

**Verdict: not documented as a specific artifact.** I found no paper that names this structural cue or reports a
heuristic baseline on ProsQA (Theory's proximity heuristic scores 0.91). Prior work documents only *that* ProsQA admits
shortcuts:

| Paper | What it says about ProsQA shortcuts | Specific artifact or heuristic baseline? |
|---|---|---|
| Cui et al. 2602.22441 | Most methods keep accuracy with zero latent steps; answers attend to question tokens; ProsQA's "simpler logical structures" allow "surface-level cues" | No |
| Rizvi-Martel et al. 2604.06374 | Fine-tuned Coconut "obtains the correct answer in a single forward pass, then copies it" | No |
| Aswal et al. 2606.12689 | Removing thoughts barely changes Coconut ProsQA accuracy | No |
| Zhang et al. 2512.21711 | Shortcut experiments on MMLU/HotpotQA with injected cues; not ProsQA | No |
| Dilgren & Wiegreffe 2604.04902 | Appendix K gives a heuristic for **PrOntoQA** (not ProsQA) that solves it without search; ProsQA is described as having more distractors | For PrOntoQA only. I couldn't read the appendix's full text; read it before citing details |
| bmarti44 checkpoints (HF model card) | Pause-curriculum ≈ Coconut; curriculum drives gains | No |

**Closest general precedent:** Jiang & Bansal 1906.07132 (multi-hop QA solvable through single-hop reasoning shortcuts;
adversarial distractors that remove them). Also Clever-Hans shortcuts on path-star graphs, Bachmann & Nagarajan
2403.06963.

**Recommendation:**
- Report it as a new finding: a person-agnostic proximity heuristic reaching 0.91 on the original ProsQA train/test, and
  a cue learned by Coconut GPT-2 (dd5ce92).
- Pair it with a **balanced re-generation** (the negative option equally often attached to a person), as in Jiang &
  Bansal's adversarial distractor construction.
- NORTH_STAR benchmark line (suggested): *ProsQA's negative option always hangs off another person; a proximity
  heuristic reaches 0.91 and Coconut GPT-2 uses that cue (THEORY §57); not documented in prior work (see
  LIT_PROSQA_SHORTCUT_2026-09-30).*
