# Literature audit — how close is prior work to our findings? (2026-09-28 ~22:50)

Scope: ~25 targeted web searches plus abstract-level reads (full-text only for "The Illusion of Superposition?").
Not exhaustive — a full review should read the closest papers (★) in full before any novelty claim.

**Correction (2026-09-28 ~23:35).** This audit was first written without consulting the project's own systematic
review, `docs/LITERATURE.md` (2026-09-25/27, arXiv IDs verified). That review lists three key papers missed here, and
one of them changes the C2 verdict:
* **Zhang, Hu, Peng, Xie 2026, *When Does Recurrence Become an Algorithm?*, arXiv 2607.20594** — in-distribution
  "head instruments" (convergence time, frontier completion) predict seed-level extrapolation of recurrent models
  (AUROC up to 1.00, 3 × 16 seeds, retrospective). A direct precedent for C2: **C2 is downgraded from medium–high to
  medium.** What remains ours: prospective, pre-registered tests in a continuous-thought model; the finding that the
  in-distribution instrument we pre-registered (end activity, P26) is weak on new runs (AUC 0.67 in P30) while the
  horizon-crossing probe P7 is not (0.99) — i.e. in our setting in-distribution instruments are *not* enough.
* **Aswal et al. 2026, *Observable Patterns Are Not Explanations*, arXiv 2606.12689** — pause-as-thought reproduces
  the BFS signature; removing thoughts barely changes Coconut ProsQA accuracy. Bears on C2 (a sweep-like attention
  pattern is not by itself evidence of use) and C5.
* **Lam-Muir 2026, arXiv 2608.03263 / 2608.16085** — pre-registered, difficulty-clocked arrival; the closest
  methodological sibling.
The table below is kept as first written; read C2 with this correction.

## Verdict per claim

| # | Our claim (evidence) | Closest prior work | Overlap | What is still ours | Novelty |
|---|---|---|---|---|---|
| C1 | Identical recipe → seeds split into generalising vs depth-limited solutions at equal in-distribution accuracy (P30: 21/42) | McCoy+ 2020 (BERT, HANS 0–66 % across 100 seeds); Zhou+ 2024 (addition length generalisation, large seed variance); Qin+ 2024 (bimodal seeds, rule competition); Juneja+ 2023 (basins = strategies); Zhong+ 2023 (clock vs pizza); ★Kuo+ 2026 (looped transformers: "high OOD variance even across well-performing ID solutions") | **High** — the phenomenon is well known | Shown for continuous-thought (Coconut-style) latent reasoning, where it was not reported | Low as a phenomenon |
| C2 | An internal readout (frontier reach, P7) predicts which seeds extrapolate, prospectively and pre-registered (AUC 0.993; replication 0.90) | Juneja+ 2023 (loss-surface clusters identify strategy); Qin+ 2024 (stable vs unstable seeds); "Are LRMs Easily Interpretable?" 2026 (decodability predicts *per-instance* correctness) | Medium — others separate strategies, not by a mechanistic readout that predicts per-model OOD before testing | Mechanistic, per-model, prospective prediction of length generalisation from one internal quantity, with the weaker alternative (end activity) beaten | **Medium–high** (strongest candidate) |
| C3 | Failing sweeps stop at the *training depth*, not the step budget; the horizon moves with training depth (L2 12/13; M1 7/10; R pending) | ★Kuo+ 2026 (spurious correlation between length and loop count); ★"Loop, Think & Generalize" 2026 (vanilla transformers fail depth extrapolation; recurrence fixes it); ★2510.19753 (training graphs within model capacity → algorithm, beyond → heuristic); Zhou+ 2023 RASP-L | Medium — "fails beyond training range" is generic; tying computation to training statistics is known for looped models | Mechanistic localisation in latent reasoning: the frontier itself halts at the trained depth with thoughts left unused; dose-response across depths | **Medium** |
| C4 | Outcome decided late in the curriculum (arrival at full depth), rarely changes after; precision irrelevant (P27, P28 not supported) | Achille+ 2019 (critical periods); ★Ali 2026 "Critical Windows of Complexity Control" (reason-vs-memorise fate set in a sharp training window); Qin+ 2024 (data composition) | Medium–high | The specific window (full-depth arrival under the gated curriculum) and the precision null | Low–medium |
| C5 | GPT-2 Coconut on ProsQA does not reason step by step: answer computed in the prefix pass, one thought ≈ six, no sweep, not step-locked — also with an answer mask and with a delayed query (G, J) | ★"The Illusion of Superposition?" 2026 (fine-tuned GPT-2 Coconut on ProsQA: removing latent tokens ≤ 1 % drop; target entity dominant from step 0; from-scratch 2-layer models do use superposition); "Are LRMs Easily Interpretable?" 2026; "Do Latent Tokens Think?" 2025; Cui+ 2026 (pervasive shortcuts, no faithful structured search) | **High** — essentially the same conclusion | Answer-masked and delayed-query controls (thoughts made necessary, still one-pass); hop-by-hop blocking | Low (confirmation) |
| C6 | GPT-2 falls to chance ≥ 4 hops beyond its ProsQA training depth (P, pending control); does the wall follow training depth? (Q, pending) | "Loop, Think & Generalize" 2026 (vanilla transformers fail deeper compositions); 2510.19753 | Low–medium — no Coconut/ProsQA depth-extrapolation test found | First depth-extrapolation test of Coconut on ProsQA; train-depth manipulation in a pretrained model | **Medium** if confirmed |
| C7 | A training change (longer lookup scaffold / earlier full depth) raises the extrapolating share (O, pending) | Kuo+ 2026 (stochastic loop counts cut OOD variance); Awasthi+ (auxiliary tasks); Zhou+ 2024 (format, position encodings); Qin+ 2024 (data) | Medium — "training choices tame seed variance" exists for other architectures | In continuous-thought reasoning, tied to the mechanism of C2/C3 | Medium if positive |

Background we build on (cite, not competing): Coconut (Hao+ 2024); Reasoning by Superposition (Zhu+ 2025, 2-layer
continuous-thought BFS construction); Emergence of Superposition (Zhu+ ICLR 2026, training dynamics, bounded
index-matching logit); Transformers Struggle to Learn to Search (Saparov+ ICLR 2025, path-merging, fails on larger
graphs); Bansal+ 2022 (recurrent extrapolation, overthinking).

## What this means for a paper

* Do **not** sell the seed split (C1) or "Coconut GPT-2 doesn't really reason" (C5) as discoveries — both are
  established. Use them as context and confirmation.
* The defensible contribution is the **mechanistic account of seed-level algorithm selection in latent reasoning**:
  one internal quantity predicts, before testing, which runs generalise (C2), and the failure mode is a horizon pinned
  to the training depth rather than the compute budget (C3), with a dose-response (R) and possibly a pretrained-model
  counterpart (C6/Q) and a training fix (C7/O).
* Must-cite and must-distinguish: Kuo+ 2026 (closest on seed instability + length/compute coupling, but looped
  transformers and no mechanistic predictor), Illusion of Superposition 2026 (closest on ProsQA), Zhu+ 2025/2026
  (the mechanism we measure), McCoy+ 2020 / Zhou+ 2024 / Qin+ 2024 (seed variance), Ali 2026 (critical windows),
  2510.19753 (training data within capacity decides algorithm vs heuristic).
* Risks: a 2026 paper we missed could already report a per-model mechanistic predictor of length generalisation;
  read the ★ papers in full, and search again before submission.

## Sources

* Coconut — https://arxiv.org/abs/2412.06769
* Reasoning by Superposition — https://arxiv.org/abs/2505.12514
* Emergence of Superposition (ICLR 2026) — https://arxiv.org/abs/2509.23365
* The Illusion of Superposition? — https://arxiv.org/abs/2604.06374
* Are Latent Reasoning Models Easily Interpretable? — https://arxiv.org/abs/2604.04902
* Do Latent Tokens Think? — https://arxiv.org/abs/2512.21711
* How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision? — https://arxiv.org/abs/2602.22441
* Interpreting Latent CoT Reasoning as Dynamical Systems — https://arxiv.org/abs/2607.09698
* Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping — https://arxiv.org/abs/2606.29983
* Stability and Generalization in Looped Transformers — https://arxiv.org/abs/2604.15259
* Loop, Think, & Generalize — https://arxiv.org/abs/2604.07822
* Unlocking OOD Generalization via Recursive Latent Space Reasoning — https://arxiv.org/abs/2510.14095
* Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data — https://arxiv.org/abs/2510.19753
* Transformers Struggle to Learn to Search — https://arxiv.org/abs/2412.04703
* Transformers Can Achieve Length Generalization But Not Robustly — https://arxiv.org/abs/2402.09371
* Sometimes I am a Tree — https://arxiv.org/abs/2412.04619
* Linear Connectivity Reveals Generalization Strategies — https://arxiv.org/abs/2205.12411
* BERTs of a feather do not generalize together — https://arxiv.org/abs/1911.02969
* The Clock and the Pizza — https://arxiv.org/abs/2306.17844
* Grokked Transformers are Implicit Reasoners — https://arxiv.org/abs/2405.15071
* Critical Windows of Complexity Control — https://arxiv.org/abs/2605.04396
* End-to-end Algorithm Synthesis with Recurrent Networks — https://arxiv.org/abs/2202.05826
* What Algorithms can Transformers Learn? (RASP-L) — https://arxiv.org/abs/2310.16028
* Learn Your Own Thoughts: Abstract Token Curriculum — https://arxiv.org/abs/2609.19717
