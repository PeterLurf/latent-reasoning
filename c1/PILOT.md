# THEORY §67 pilot log (seed 99, not a panel seed; every configuration tried, with its outcome)

Started 2026-10-01 18:4x EDT. Common to all: 2000 updates, bs 512, AdamW (wd 0), fp32, curriculum lengths
{1}, {1,2}, {1,2,4}, {1,2,4,8} in 4 equal stages; hard-execution eval at N 8–128 (2,000 per depth).
Audit's conditions (18:5x):
* log every configuration;
* freeze ONE init rule before the panel and apply it to every GPT-2 arm (d1, reset, FP); TAB keeps its default;
* at most 4 further attempts after configuration 6, then stop and report "GPT-2 cell fails rule acquisition".

| # | arm | configuration | outcome |
|---|---|---|---|
| 0 | d1 | lr 1e-4 (all params), default heads, codes = shared mean + 0.02 noise | eval OOM (10,000-row null batch); voided `pil_d1.void_evaloom`; training plateaued at loss 0.786 (L = 1), acc ≈ 0.18–0.30 |
| 1 | tab | lr 1e-2 | **learns**: certified (0 transition errors), 1.000 at N 8–128 by update 1,500 |
| 2 | d1 | lr 1e-4, lr_new 1e-3 | loss 0.786 at L = 1 → 1.49; acc 0.15–0.18 (chance) |
| 3 | d1 | lr 1e-4, lr_new 1e-2 | loss → ln 6; acc chance |
| 4 | d1 | zero-initialised heads | loss exactly ln 6 throughout (symmetric saddle: identical codes + zero heads) |
| 5 | d1 | small heads (initial logit std ≈ 0.5) + distinct codes (N(0, 0.14²)) | acc 0.32–0.34 at every N 8–128 (no decay), transition errors 26 → 15; not certified |
| 6 | d1 | small heads only | acc 0.16–0.18 by update 1,500 (chance) |

**Diagnosis.**
* GPT-2's last hidden state has norm ~200, with one outlier dimension (|h| ≈ 194, the "massive activation"). Op-to-op
  differences are ~10.
* Default heads therefore start at logit std ~5 and saturate: configuration 2's trained cell maps 4 of 6 ops from code 0
  to one code with p = 1.0.
* With nearly identical codes, the decoder cannot tell codes apart, a symmetric saddle (configuration 4).
* Configuration 5's flat 1/3 at every depth is consistent with a stable automaton that tracks only the sign of the
  permutation (S3 → Z2): it then picks among 3.

**Attempts 7–10 (the bounded set after Audit's conditions):**

| # | arm | configuration | outcome |
|---|---|---|---|
| 7 | d1 | configuration 5, 6,000 updates | 0.48–0.51 at every N (flat); not certified |
| 8 | d1 | LayerNorm on the cell output before both heads + distinct codes, default heads, 2,000 updates | 0.50–0.52 at every N 8–128 (flat); transition errors 28 → 8 during training; not certified |
| 9 | d1 | configuration 8 with 8,000 updates | plateau 0.48–0.52 from update ~2,000 on; final 0.48–0.52 at every N; decoded codes [1, 1, 2, 5, 0, 4] (no bijection), 17/36 transitions wrong |
| 10 | d1 | configuration 8 + lr_new 1e-3, 4,000 updates | 0.24 at N 8 → 0.15 at N 128; 15/36 wrong |

## Pilot verdict (2026-10-01 19:32 EDT): GPT-2 cell fails rule acquisition (the pre-stated stop rule)

* **Attempts.** 11 configurations were tried on pilot seed 99 (the voided OOM run 0 included): 1 TAB, 10 d1.
* **No d1 configuration learns S3.**
  - The best (8, 9) settle on a stable partial automaton: ~0.50 at every depth up to 128, with no decay. So the hard
    codes are not drifting.
  - But 3 of 6 codes duplicate a decoded state, and 13–17 of the 36 transitions are wrong.
  - More updates (9) and a higher lr for the new parameters (10) do not help.
* **The exact-marginalisation objective is not the problem.** TAB, under the same loss and curriculum, is certified
  with 1.000 at every depth after 1,500 updates.
* **Diagnosis.** The failure is in getting a pretrained GPT-2 cell to produce six well-separated, op-dependent code
  transitions:
  - GPT-2's outlier hidden dimension (norm ~200) saturates default heads;
  - identical-start codes give a symmetric saddle;
  - with both fixed (LayerNorm head input, distinct codes), the cell still falls into a partial automaton that the
    marginal likelihood does not escape.
  This matches the external agent's "likeliest failure": good-ish marginal likelihood, bad hard rollouts.
* **The panel (D1/TAB/RESET/FP × 5 seeds) is not run.** The stop rule applies. Theory decides the next step (e.g. M = 12
  codes for spare capacity, the external agent's next item, or a frozen-GPT-2 cell with a trainable adapter).

## Amendment 67-1 run record (2026-10-01 19:50 EDT; before any of these runs; descriptive)

* **Seeds and budget.** 99, 1, 2; 2,000 updates, bs 512, the same marginalisation and curriculum.
* **Cells and settings** (fixed now):
  - **mlp:** lr 1e-3.
  - **tf_small:** lr 1e-3 (2 layers, d 256, 4 heads, pre-norm, LayerNorm before heads).
  - **gpt2_rand:** lr 1e-4 with the best pilot init (configuration 8: LayerNorm head input + distinct codes); GPT-2's own
    init, weights seeded by the run seed.
* **Placement.** Local: gpt2_rand on the 4060, tf_small on the 3070, mlp on the 2070, 3 seeds sequentially each.
* **Reported.** Certified (36-transition audit) and hard-execution accuracy at N 8–128 per run.
* CPU plumbing smokes (100 updates) are voided as `_smoke_*.void`.

## Amendment 67-1 result (2026-10-01 20:04 EDT; descriptive)

| cell | seed 99 | seed 1 | seed 2 | certified |
|---|---|---|---|---|
| mlp (lr 1e-3) | 0.32–0.34 flat; 22/36 wrong | 0.31–0.34 flat; 21 | 0.16–0.18 (chance); 24 | 0/3 |
| tf_small (lr 1e-3) | 0.15–0.19 (chance); 17 | 0.30 at N 8 → 0.17; 20 | 0.25 → 0.17; 20 | 0/3 |
| gpt2_rand (lr 1e-4, LN + distinct codes) | 0.30 → 0.17; 14 | **0.49–0.52 flat**; 18 | 0.32–0.34 flat; 20 | 0/3 |
| TAB (pilot reference, lr 1e-2, seed 99) | 1.000 at every N; 0 wrong | | | 1/1 |

**Reading.**
* No cell acquires S3: 0/9, against the pilot's pretrained d1 at 0/10 configurations.
* Random-init GPT-2 does not learn where pretrained GPT-2 does not. Neither do the small cells, so neither "pretraining
  geometry" nor "scale/depth" alone is the obstacle.
* Every neural cell that learns anything lands on the same stable partial automata as pretrained d1: flat 1/3 (sign
  only) or 1/2 at every depth to 128.
* So the obstacle is common to neural transition cells under this objective and these settings: the local optima of the
  marginal likelihood (HMM-style symmetric/partial solutions).
* **Confound, stated.** The only learner, TAB, used lr 1e-2 and is a direct table parametrisation; the cells used 1e-3 /
  1e-4. A lr-matched MLP would separate "parametrisation" from "learning rate". Theory decides; it is not run here.
* By 67-1's readings, both GPT-2 variants are recorded as acquisition failures.

## Diagnostic 2×2 on lr (2026-10-01 20:04 EDT; Audit; logged before running; descriptive)
* TAB at lr 1e-3, and MLP at lr 1e-2, seeds 99 / 1 / 2. TAB at lr 1e-2 on seeds 1 / 2 completes the reference.
  Everything else as 67-1.
* **Reading.**
  - TAB stalls at 1e-3: optimisation (lr / symmetric optima) is the obstacle.
  - TAB learns at 1e-3 while the MLP fails at 1e-2: the shared-function parametrisation is the obstacle.

### Diagnostic result (2026-10-01 20:06 EDT)

| run | seed 99 | seed 1 | seed 2 |
|---|---|---|---|
| TAB lr 1e-2 | **certified, 1.000** (pilot) | 0.49–0.51 flat; 14/36 wrong | 0.48–0.51 flat; 13 |
| TAB lr 1e-3 | 0.85 at N 8 → 0.51 at 128; **1/36 wrong** | 0.38 → 0.17; 11 | 0.32–0.35 flat; 14 |
| MLP lr 1e-2 | chance; 16 | 0.32–0.34 flat; 22 | chance; 24 |

**Reading.** The pilot's TAB success was a lucky draw: TAB is certified in 1/6 runs across both lrs. Even a free
6×6×6 table under this objective and curriculum mostly lands on the same stable partial automata (flat 1/3 or 1/2) as
every neural cell.
* The obstacle is the optimisation of the exact marginal likelihood: HMM-style local optima, with symmetric or merged
  states.
* It is neither GPT-2 nor the shared-function parametrisation. d1's failure is therefore not evidence against a GPT-2
  cell as such.
* Any C1 Design-1 follow-up must first make the objective find the group table reliably in TAB, e.g. multiple random
  restarts selected on training likelihood, an annealed temperature on T, or a longer curriculum. This is a design
  question for Theory, not a pilot tweak.

## Amendment 67-2 result (2026-10-01 22:33 EDT): GPT-2 cell, M = 12, LN heads + distinct codes, 8,000 updates, seeds 1–10

**3/10 certified** (s1, s2, s8: product automaton, 0 bad pairs, 1.000 at N 8–128). The other 7 sit on flat partial
automata (0.24–0.53 at every N; 5–15 bad pairs).
Under 67-2's rule this GPT-2 M = 12 cell does not reach 9/10. The rule was met by the bilinear toy cell (Theory), and
§67b (the GPT-2 cell + bilinear readout + τ annealing, M = 24) is the registered revision.
