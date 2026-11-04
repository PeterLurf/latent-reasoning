# LIT: wave-4 08 C4 onset-drift (SG-WRITE / ANC / LPFT) prior art + wave-4 late-doc ID audit (2026-11-04 11:19 EST)

Requested in docs/design_wave4_2026-11-03/08_c4_onset_drift_pilots.md §16 / U5 ("Literature to verify ids and titles before registration"; "Narrow novelty; Literature to spot-check"). Unprompted part: ID audit of the wave-4 docs that changed after my 190-ID audit (e573e123).

## 1. Answer (one line each)

- **U5 IDs: both correct.** 1606.09282 = Li & Hoiem, *Learning without Forgetting*. 1802.01483 = Li, Grandvalet & Davoine, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks* (this is the L2-SP paper; cite it under that title, not as "L2-SP"). Checked via the arXiv API.
- **SG-WRITE novelty: NARROW. Do not claim "not found".** Detaching earlier recurrent passes is standard: Huginn 2502.05171 (truncated BPTT, last 8), HRM 2506.21734 / TRM 2510.04871 (earlier segments under no_grad), ATC 2609.19717 (detaches the first k-b passes) and Looped Flows 2609.11801 (sg between every step). **Looped Flows App. B (full text) states the doc's M2 mechanism as a conjecture:** "The stop-gradient prevents the later loss from directly teaching the earlier step what to retain for exploitation." That paper does not test the conjecture.
- **What is left to 08:** (a) a *per-item* mask (detach only where d_i > t) rather than truncation by pass index; (b) used as a **diagnostic** to localise drift of an already-present seeded code (P-direct vs P-shared), not as a training efficiency or a recipe; (c) no local loss on the detached pass. Looped Flows and TRM/HRM give every step its own loss; ATC and Huginn keep answer CE only. 08's pass 1 has no loss at all, so its code is held only by the init and the d 1 steps (08's M4). Point (c) is why M4's erosion branch exists and is not covered by any of these priors.
- **ANC: KNOWN FORM.** It is LwF 1606.09282's KL to the old model's outputs, restricted to the 11 number tokens at one position. The same form is the KL-to-reference-policy penalty in RLHF (Ouyang 2203.02155). PAR 2610.27628 is the nearest latent-reasoning use, as 08 says. Write it as "LwF-style KL to the step-0 model on the pass-1 readout", with no novelty claim.
- **LPFT: KNOWN.** Kumar 2202.10054, plus gradual unfreezing (ULMFiT 1801.06146, already in our bib). Our variant differs only in what is frozen (the trunk, with the memory module live). No claim.

## 2. Suggested §16 wording for 08

> SG-WRITE detaches the pass-t write source for items not yet answered at pass t (a per-item variant of the truncated backprop of 2502.05171 / 2506.21734 / 2510.04871 / 2609.19717). It is used as a localisation test of answer-gradient drift. Looped Flows 2609.11801 (App. B) conjectures the same mechanism (later losses teaching earlier steps what to retain) without testing it. Unlike those works, the detached pass carries no local loss. ANC is an LwF-style KL (1606.09282) to the step-0 model on the pass-1 number readout. LPFT follows 2202.10054.

## 3. Evidence tags

| Source | Claim used | Type | Key assumptions |
|---|---|---|---|
| 2609.11801 App. B | sg blocks later losses from shaping earlier retention, which hinders shortcuts | **conjecture** (empirics: more steps help, ruling out the pure repeat-first shortcut) | every step has a local denoising loss; ARC / Sudoku, from scratch |
| 2609.19717 ATC | detach the first k-b passes; window 5 advance-only 51.0 vs backtracking 97.4 | empirics (search summary; the 97.4 / 47.8 numbers in our ATC notes come from the full text, LIT_STEPTOLD) | 2-layer GPT-2 from scratch, answer CE only |
| 2506.21734 / 2510.04871 | deep supervision with detached segments | empirics (HRM gives a 1-step gradient approximation motivated by DEQ; no proof for the reasoning task) | segment-level loss |
| 1606.09282 LwF | KL to the old outputs preserves old-task behaviour | empirics | old head outputs are informative on the new data |
| 2202.10054 LP-FT | fine-tuning distorts features; LP first helps OOD | **theorem** (two-layer linear networks) + empirics | linear model, ID / OOD subspace split |

## 4. Wave-4 docs changed after e573e123: ID audit

19 .md files under docs/design_wave4_2026-11-03/ changed after e573e123. They hold 109 distinct IDs. 92 were already in our bib / LIT notes. The **17 new IDs all resolve, and each is cited for what its title says**: 1511.03643 (distillation + privileged info), 1606.09282 (LwF), 1703.01780 (Mean teachers), 1805.01500 (Noisin), 2006.07710 (Pitfalls of simplicity bias), 2109.04617 (task groupings / gradient affinity, Fifty), 2109.05925 (adversarial MWP), 2110.14056 (algorithmic reasoning transfer), 2211.12588 (PoT), 2301.10884 (Break It Down), 2302.04761 (Toolformer), 2304.01373 (Pythia), 2304.14767 (Dissecting Recall: attention knockout), 2305.13245 (GQA), 2402.06457 (V-STaR), 2410.05464 (progressive distillation implicit curriculum), 2605.22870 (Readout Shortcut). Also checked: 1802.01483, 2202.10054, 2203.02155, 2506.21734, 2510.04871, 2609.11801, 2606.06479. **No wrong IDs.**

## 5. Search log

One own-bib grep (LITERATURE / LIT_* / EXPERIMENT_BACKLOG) for truncated BPTT, Transformer-XL, gradient starvation, CODI, LwF, LP-FT and the IDs above. Two standard web searches. Looped Flows App. B and §3 were read from the arXiv HTML full text. New lead, no action: SMT 2606.06479 (Kumar & Isola), which trains RNN memory transitions on one-step labels and corrects memory drift with a DAgger phase. It is relevant only if a write-target route is ever considered, and that would not be C4.
