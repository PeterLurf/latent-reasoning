# Task-specific scaffolds in SOTA latent reasoners vs task-agnostic methods (2026-10-07)

Literature session, at the user's question ("do these latent models have task-specific scaffolds or are they agnostic?").
IDs as in LIT_INDEX. Configs from the official repos where noted (LIT_S119_BASELINES, LIT_S149_SOTA_PROBE_FACTS).

**Three kinds of scaffold:**
- **S** = supervision scaffold: it needs CoT text or step segmentation.
- **L** = layout scaffold: a task-specific I/O format or step boundaries, or a per-task latent count.
- **P** = trained per task: one model per task, not one model for many.

| Method | Recipe the same across tasks? | Scaffolds | Notes |
|---|---|---|---|
| Soft Thinking 2505.15778 | **yes** (training-free) | none | inference-only, on any LM |
| Huginn 2502.05171 / Ouro 2510.25741 | **yes** | none (LM pretraining) | the only fully general ones: one model, many tasks |
| Soft Tokens Hard Truths 2509.19170 | yes | outcome RL; none beyond the task reward | maths only in the paper |
| ICoT-SI 2405.14838 | yes (a generic token-removal schedule) | S (CoT text), P | a separate model per multiplication size / GSM8K |
| CODI 2502.21074 | mostly | S (CoT for the teacher), L (a fixed 6 latents, "The answer is:" alignment), P | GSM8K-trained models are tested OOD on other maths sets without retraining |
| **Coconut 2412.06769** | **no**, tuned per task | **S + L + P**: CoT steps define the stages | **c and the stage count chosen per task:** GSM8K c = 2, 3 stages; ProsQA / ProntoQA c = 1, 6 stages |
| SIM-CoT 2509.20317 | per task | **S + L**: step-delimited CoT; an aux decoder per latent ↔ step k (the "Step k" prompt) | inherits Coconut's / CODI's per-task settings |
| LOTUS 2606.31779 | per domain | S (gold step text per latent) | — |
| MUX 2607.18264 | yes | S (spans of CoT text) | generic multiplexing of any CoT |
| Abstract-CoT 2604.22709 | **yes**, multi-domain (maths, instructions, multi-hop) | S (verbal CoT for warm-up) | one recipe; a reserved vocabulary |
| Retrofit recurrence 2608.11233 | per task family | **S + L**: intermediate-step supervision, one task step per loop, then outcome annealing | the step ↔ loop alignment is designed per task |
| HRM 2506.21734 / TRM 2510.04871 | the architecture is generic | **L + P**: a separate small model per puzzle (Sudoku, Maze, ARC), task-specific grid I/O and augmentation | trained from scratch on about 1k examples per task |
| Attractor Models 2605.12466 | generic architecture | task supervision; P for puzzles | — |

## Reading
- **Fully task-agnostic in practice:** only pretraining-scale recurrent-depth LMs (Huginn, Ouro) and training-free / RL soft-token
  methods. These have **no causal (C1) evidence**, and their latents are least structured.
- **Every latent-CoT method that shows step-structured latents relies on a CoT supervision scaffold** (S). Coconut and SIM-CoT
  also tune the latent layout per task (L). That's the "task-specific scaffold".
- **Puzzle reasoners (HRM / TRM) are agnostic in architecture but not in practice.** One model per task, with a hand-set format.

## Where we stand (honestly)
- **Our carrier also has a layout scaffold:** one Markov segment per input item (or per digit pair / column), plus the item format.
  - The recipe (FSQ carrier, FINAL loss, length curriculum, COPY, restart-on-plateau) is identical across the 5 §128 families, with no
    CoT (no S).
  - The *segmentation* is still chosen per task (L), and models are trained per task (P).
- **Multiplication needed per-task layouts** (multiplier-major, column-major, digit-pair, column flags). That's exactly the
  scaffolding the user's task-agnostic directive targets.
- **Our distinctive position:** no S on finite-state families (strict C4). Others drop L or P only by adding S, or by pretraining at
  scale.
- **The open step:** an item-agnostic segmentation, e.g. one segment per input token, or a fixed-width chunk with no task-specific
  boundary. B1 / B4 streamed-digit designs move toward it.
