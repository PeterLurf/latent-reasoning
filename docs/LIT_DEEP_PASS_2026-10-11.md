# Deep research pass (Audit, user-requested): five IDs verified, plus randomised / offset positions for pretrained absolute-position models (2026-10-11)

Literature session, for Audit and Theory. All IDs verified via arXiv abstract pages (2026-10-11).

| ID | Paper (date) | Verified claim | Use for us |
|---|---|---|---|
| **2305.16843** | Ruoss, Delétang, Genewein, … "Randomized Positional Encodings Boost Length Generalization of Transformers" (2023-05-26) | length failure is linked to positional encodings going OOD at longer lengths (even relative ones). The fix **simulates positions of longer sequences and randomly picks an ordered subset**. **6,000 models, 15 algorithmic tasks, +12.0% average** test accuracy on unseen lengths | already cited (LIT_C2_MULT_SCHEDULE). Confirmed as remembered |
| **2311.14737** | Shen, Bubeck, Eldan, … "Positional Description Matters for Transformers Arithmetic" (2023-11-22) | naive reliance on positions breaks larger numbers. Fixes modify the PE or the task representation. A 100M model on 300k samples: **direct (no scratchpad) multiplication strong to 15 digits, essentially perfect to 12**; addition extrapolates 10 → 12 digits | arithmetic format / position design; a direct-multiplication reference |
| **2606.18206** *(new)* | Movahedi, Milovanović, Feigin, … Orvieto, "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers" (2026-06-16) | deep looping causes a **signal-propagation problem** (as in deep nets) when halting is postponed. Fixed with **pre-norm layers + residual scaling**. **FPRM uses fixed-point convergence as an end-to-end halting mechanism**, so compute adapts to difficulty. Sudoku, Maze, state tracking, ARC-AGI | supports convergence halting (§173) and the LayerScale-style residual scaling (Chen 2603.21676). A published fixed-point halter to cite beside RecurTrace |
| **2608.01593** *(new)* | Zhao, Zhang, Xu, … "Latent Thought Credit: Multi-Answer Credit Assignment for Latent Reasoning" (2026-08-03) | answer-only rewards mix thought quality with answer-sampling noise. **LTC samples several latent thoughts, fixes the context after each, and averages rewards over several answers per thought** (thought-level advantages), plus advantage-weighted thought matching; GRPO-style | for a strict-C4 RL stage on latents (LIT_CAND5_6): multi-answer averaging is the variance fix for latent credit |
| **2609.33144** | Liang, Jin, Pan, "Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers" (2026-09-27) | MR-Loop (matched recurrence) vs DR-Loop (decoupled), under final-answer supervision. MR-Loop updates a state at a fixed readout and advances through a transferable *progress cue*; DR-Loop propagates states as a computational frontier. **Both degrade at depth**: readout / progress-cue degradation (MR) and declining propagation reliability (DR). **"Limited self-correction allows local errors to persist and compound."** States encode content *and* its computational status | **a fair mechanism citation for COUNT-OOD:** step errors past the trained count persist and compound, because the loop has limited self-correction. Their "progress cue" is the analogue of our pass-index / size keying (187-2) |

## Randomised or offset positions during fine-tuning of a *pretrained absolute-position* model (GPT-2-style)
- **Direct precedent, randomised indices in fine-tuning a pretrained GPT-2: not found.** The randomised-position work trains from scratch (Ruoss 2305.16843;
  Abacus random offsets 2405.17399; Kuo-style stochasticity).
- **The relevant 2025–26 theory and evidence:**
  - **2607.18759, "Relative Positions Generalize, Absolute Positions Memorize":** an optimisation (implicit-bias) account.
    - Rotary encodings make attention depend only on relative offset, so a rule learned at short lengths transfers exactly.
    - **Learned absolute encodings leave out-of-range positions unconstrained, and the trained head pins to a fixed absolute position inside the training
      range.**
    - Their framing implies (consistent with Ruoss) that randomising indices so out-of-range ones get trained restores length generalisation.
    - **The direct theory for why GPT-2's learned absolute positions limit our length / depth C2.**
  - **2407.17963 (a unified account of arithmetic generalisation):** translation invariance in addition aligns with *relative* PE. Experiments span GPT-family
    models; the search snippet indicates pretrained and random-init GPT-2-small with and without PE (not confirmed beyond the abstract).
  - **2510.04130, "On the Limitations and Capabilities of Position Embeddings for Length Generalization":** PEs structure computations rather than add
    capability. It proposes Scale Hints and a learned PE framework.
- **For us:**
  - A **random start offset for GPT-2's absolute positions during fine-tuning** (Abacus / Ruoss style) is a cheap untested lever. It trains position rows
    beyond the training range.
  - **Caveat:** GPT-2's learned rows up to 1,024 are already trained by pretraining, so the binding issue may be the *task-specific* attention pinned to
    absolute positions (2607.18759), not untrained rows. An offset in fine-tuning directly targets that pinning.
