# Leads for THEORY_DIRECTIONS_2026-10-04: (A) fast hand-off convergence, (B) learning without step labels (C4), (C) fully latent chains without snaps (2026-10-04)

Literature session, for Theory's memo (the user's request via Audit); Audit copied. IDs verified via the arXiv API,
Semantic Scholar or arXiv abstract pages (2026-10-02..04). ⚑ marks a flag (refuted, or our own negative result).

## Theory's candidates → closest prior
| # | Candidate | Closest prior (verified) | Flag |
|---|---|---|---|
| A1 | carry the state in depth (looped middle block + input injection) | Huginn 2502.05171; TRM 2510.04871; Universal Transformers 1807.03819; **T²MLR 2607.15178** (middle-layer state fed into the next token, retrofitted into a pretrained 1.7B model); 2510.14095 (looped + discrete re-embedding) | our §88/§91 compose with per-iteration targets; seed lottery addressed by operator-first (Zhang 2607.20594) |
| A2 | round-trip consistency (slot ≈ re-read of its own checkpoint) | ReLAT 2606.06252 (question → latent → question, test-time); SELR 2608.13570 (self-decoding); **2606.20075** (generative reconstruction beats geometric imitation) | ⚑ our F4 translator never trained (§63). Use as an auxiliary loss on an already-working chain, not as the primary signal |
| A3 | compact digit-structured state target (S_j only) | **2510.00184** (a running-sum auxiliary loss unlocks multiplication); MUX 2607.18264 (lossless superposed targets) | §86: target structure matters (a 120-way index head stuck; 5 × 5 learned) |
| B1 | self-training by length | **Lee et al. 2502.01612** (self-improving transformers extend length by rounds of self-labelled, filtered data); **2511.07378** (proves recursive self-training extends the solvable length); STaR 2203.14465; our Pilot L/M | ⚑ §62: label noise installs counts. It needs filtering (majority vote / length filter as in 2502.01612; our M-2 agreement filter) |
| B2 | restart-consistency reward (a checkpoint is good if restarting reproduces the answer) | Markovian Transformers 2404.18988 (informativeness: the answer predicted from the CoT alone); Delethink 2510.06557 (RL under chunked carry-over); MEM1 2506.15841 | ⚑ §50: a reward-learned halt collapsed to a constant 2 segments. Reward-only signals need a warm start (§69) |
| B3 | exact marginalisation over snap codes in the LM | TRICE 2312.02179; Latent Thought Models 2502.01567 | ⚑ §67/§75/§76/§82: needs short footholds (≥ ~3–10% short problems, §85); §87: a warm start enters the basin |
| C1 | soft snap annealed away / kept small-β attractor | modern Hopfield clean-up (cited in PROPOSALS F3b); **Attractor Models 2605.12466** (fixed-point refinement with implicit differentiation; 770M beats a 1.3B transformer; Sudoku-Extreme 91.4%) | ⚑ annealing to β = 0 removes the carrier, the same posterior-collapse pattern as §89 at α = 1. **Keep a small β, don't anneal to 0** |
| C2 | denoising / contractive training | 2510.14095 (error-correction noise); DEQ 1909.01377; **"When Recursive Models Finish Computing" 2609.26487** (completed states are contractive along the motion direction but keep expanding directions: anisotropic stability) | ⚑ §23: jitter from step 0 broke learning. Add noise late |
| C3 | norm / manifold constraint | **nGPT 2410.01131** (all vectors on the hypersphere; 4–20× faster training); our §73 standardisation; HRPO's normalised projection 2505.18454 | — |
| C4 | orthogonal / Householder state channel | DeltaProduct 2502.10297; **Howe 2609.18966** (without the additive path, exact S5 at 16×; with it, a parasitic attractor); NC-SSM 2610.00329 (Lie-group SSM, 100% A5); uRNN 1511.06464; expRNN 1901.08428 | the strongest *no-snap* evidence: norm-preserving input-driven transitions don't drift. ⚑ an additive input path must be excluded (Howe) |
| C5 | long self-generated rollouts | professor forcing 1610.09038; scheduled sampling 1506.03099; **LIFT 2609.38149** (feeds back its own predicted state; trained teacher-forced) | — |
| — | mixture-collapse explanations | 2609.02049 (feedback amplifies or contracts mixtures; needs context-dependent correction); 2604.06374 (superposition unused in fine-tuned LMs); Wu 2508.03440 (single-threaded) | — |

## Strong candidates missing from the list
- **(A) Parallel teacher-forced state training.**
  - LIFT 2609.38149 pairs each input token with a precomputed target state and trains next-state prediction fully in
    parallel. SST V2 2605.00206 uses two-pass parallel training of a nonlinear recurrence. PCCoT 2506.18582 uses
    Jacobi-parallel latents. C-MTP 2607.16972 uses direct latent targets.
  - This is the scalable version of TF-SLOT: no sequential backprop, so later slots learn in parallel (the staircase
    fix).
- **(C) Fixed-point / implicit recurrence in LMs.**
  - Implicit LMs 2502.07827 (fixed-point iteration gives RNN-like state tracking in a parallelisable model);
    Fixed-point RNNs 2503.10799; Attractor Models 2605.12466 (above).
  - An equilibrium per op is a no-snap way to get a restoring map (our Th3 "digital pointer").
- **(C) Stability across seeds and lengths:** Kuo 2606.29983 (randomised stopping stabilises looped extrapolation
  across 18–32 seeds); Kohli 2604.07822 (overthinking degrades deep recurrence); fast–slow latent recurrence 2604.01577
  (persistent latent state, train short → test long).
- **(B) Discovering algorithms from outputs:**
  - grokking 2201.02177;
  - Wang 2405.15071 (the inferred/atomic ratio, not data size, drives implicit composition);
  - Abbe leap complexity 2302.11055 and Wies 2204.02892 (why step-free learning stalls);
  - BoLT 2503.18866 (EM bootstrapping of latent thoughts) and LaTRO 2411.04282 (variational self-reward);
  - spectral initialisation then EM (0811.4413 and the §85 lead).
  - Lee 2606.07254 is the only long-only, final-state success (S3×S3, with a hand-built projection).

## Priority picks (one per problem)
- **A:** parallel teacher-forced state training (LIFT-style), i.e. TF-SLOT done at scale.
- **B:** filtered self-training by length (2502.01612 + our M-2 filter).
- **C:** a norm-preserving input-driven state channel with no additive path (Howe/DeltaProduct), or a per-op
  fixed-point refinement (Attractor/Implicit LMs). Avoid annealing soft snaps to zero.
