# No snaps, or snaps without hand design: task-agnostic proposals (2026-10-05)

A memo for the user (directive ~23:55 via Audit: "figure out how to not have snaps, or at least auto snaps without
hard-coded design for each problem"). It contains no registrations. Priors: Literature b284f980
(`docs/LIT_TASK_AGNOSTIC_2026-10-04.md`) and 46fe5ca0 (`docs/LIT_TASK_AGNOSTIC_PROPOSALS_2026-10-04.md`).

## What has to go, and what we know
* **Two hand-built pieces carry every current success:**
  - (i) a task-specific **state template** that snaps write to (S5 permutation text, the carry format);
  - (ii) a hand-set **schedule** (k = 4).
  - §115's trigger and §118's ANCHOR still use (i) through the text path, so they are comparison points, not
    endpoints.
* **Drift is direction, not norm** (115-1). Without snaps, the free-running latent keeps its norm, but its cosine to the
  snapped latent is about 0.01 by step 64.
  - The reader (§111) slows the drift a lot: per-hand-off accuracy without snaps is 0.94 against 0.2, and the
    disagreement trigger snaps 6.4× less often.
* **The carried latent is not state-clustered, even in range** (`theory/autosnap_premise.py`, descriptive).
  - On §111 READ s6, a per-state centroid codebook fitted on in-range latents classifies held-out in-range latents at
    3–5 % (chance 0.8 %). k-means purity is 0.21, yet the reader decodes the state at about 1.0.
  - The state is carried in an entangled, history-dependent code. **A post-hoc snap cannot work** (as 101-2); the
    discretisation must shape the code during training.

## One harness for every proposal (the directive's hard requirement)
* **The same code and hyperparameters** on four families:
  - S5 (group, our protocol);
  - flip-flop 2306.00946 (reset and non-invertible, the canonical non-group automaton);
  - a Delétang 2207.02098 regular task (cycle navigation; train lengths 1–40, test 41–500, many published baselines);
  - 4×4 multiplication (Audit). It is included even though its step-3 wall may be compute, not snapping (diagnosis
    running).
* **One Markov segment per input item** (the dataset's own item delimiter, the same rule everywhere): [EOT, carried
  state, item tokens] → carried state. There are no text restarts and no schedule.
* **Supervision.**
  - **Primary: final answers only** (Literature (e)4).
  - A second arm adds a reader of each dataset's native step trace, read only and never fed back: "task-agnostic
    architecture, per-task supervision".
* **A pass needs ≥ 2 families, including ≥ 1 non-group family.** S5 alone does not count.

## Proposals (ranked)

### 1. P-A: a learned discrete snap at every step (FSQ digital recurrence)
* **Mechanism.** The carried latent is standardised, projected 768 → d (6–10), and each coordinate is rounded to L levels
  (FSQ 2309.15505, straight-through), then projected back to 768.
  - It is a snap at every step to a learned, task-agnostic code: no text, no schedule.
  - Soft → hard rounding is annealed early, but never annealed away from a working carrier.
* **Why it should generalise in length.** Under Markov segments the next code is a deterministic function of (code,
  item).
  - If every step is exact on the reachable codes, the chain is a learned finite-state machine: exact at any length,
    and **certifiable by enumerating the reachable table** (Literature (a)).
  - Moore minimisation gives codes per state against the minimal automaton (a C7 measure).
* **Sizing.** L^d well above the state count. Multiplication's bounded carry is about 10⁵ states, so for example d = 8,
  L = 8 gives 1.7·10⁷ codes. That is the same for every task.
* **Prior.** SR-RNN 1901.08817 and QBN 1811.12530 are the closest (from scratch, or post hoc). Latent-CoT VQ work
  quantises text spans, never the carried state per hand-off. **New as scoped.**
* **Risks.** The straight-through bias plus early quantisation noise can recreate the no-read basin (§111): the
  trace-reader arm has the reader from step 0. Code usage and perplexity are reported.

### 2. P-B: a continuous stabiliser (the comparator)
* Denoising noise on carried latents plus a norm / hypersphere constraint (nGPT 2410.01131), or a self-supervised
  next-latent consistency (NextLat 2511.05963, LIFT 2609.38149). No quantisation.
* Literature expects P-A > P-B: there is no pretrained-LM precedent for about 0.9999 per hand-off without
  discretisation, and stability trades against memory (1805.10369). It is run because it is cheap and settles whether
  discretisation is needed.

### 3. P-E: a learned *what* plus a learned *when* (after P-A)
* P-A's snap applied only when a latent-only router fires (MoR 2507.10524 / ACT-style), trained with a snap-count
  penalty.
* It saves compute that §115's trigger cannot (Audit: the text path runs at every hand-off).
* This is the gap no prior work fills (b284f980): learned content and learned timing together, in latent space, with
  a C1 test.

### 4. P-C: learned-notation text checkpoints (later)
* Free m-token writes at a learned gate with a history cut (ST-Gumbel ≫ RL for discrete messages, 1705.11192). The model
  invents its notation (InftyThink, Delethink and LightThinker are the nearest).
* C7-friendly, but emergent notations are rarely interpretable (1706.08502). It is expensive.

## Rules sketch (fixed at registration)
* **NOSNAP-GEN:** no-snap cumulative ≥ 0.99 at the family's long test length (S5 512; flip-flop and cycle navigation at
  the Delétang 500; multiplication at the bench test) in ≥ 2/3 seeds per family, in ≥ 2 families including ≥ 1
  non-group, with the same code and hyperparameters.
* **C1:** single-slot donor SERIAL at item segments in passing seeds.
* **CERT (P-A):** the enumerated reachable table is exact.
* Reported: codes per state, code usage, and drift.

## Suggested next
* **P-A pilot:** S5 + flip-flop, final-only and trace-reader arms, 2 seeds each, 8 runs, about $10.
* P-B on the same 8-cell grid, about $10.
* Extend to cycle navigation and multiplication on a pass. P-E follows a P-A pass.
* Credit is about $113 with $60–80 committed, so the pilot is posted to Audit before any launch.
