# PROTOCOL8 Phase 0 — deep ProsQA: one-pass capacity and Coconut feasibility

Written 2026-09-29 15:09 EDT, committed before any Phase 0 run started; no Phase 0 data had been looked at beyond
generation counts and a parser check. Append-only; later changes go in dated amendments at the end.

## Why

PROTOCOL8 (plan in `docs/NEXT_STEPS.md`, end) asks whether a pretrained model can be trained to do multi-step latent
work, and whether the toy-model findings then transfer to it. The toy findings are the wall at training depth, the
schedule end, and the anytime and random-budget fixes. Addenda P and Q show that Coconut GPT-2 on native ProsQA
answers in one pass, so there is no step schedule to test. THEORY Prop. 8 ("slack explains bypass") predicts latent
use only beyond one-pass capacity. Biran et al. 2406.12775 (via the Literature session) likewise says to choose depths
where one-pass accuracy falls.

Phase 0 is the part of PROTOCOL8 that no pending result can change. The user approved it at 2026-09-29 ~14:50 ("go
ahead with the experiments that don't require the results of currently running ones"). It measures:
1. the one-pass capacity on the exact PROTOCOL8 format;
2. whether standard (addendum-G, bypass-blocked) Coconut learns problems above that capacity at all.

The confirmatory arms (Coconut / anytime / random tight budgets) are pre-registered separately after THEORY Pilots D/E
(~2026-09-30 01:00–05:00) and F (~08:00–10:00) report.

**Status: exploratory feasibility.** Its outcomes set PROTOCOL8's parameters. No confirmatory claim rests on it. Its
seed (0) is not reused by PROTOCOL8's confirmatory runs.

## Data (`prosqa/make_deep.py --src data --out p8data`, seed 0)

- **Construction.** Short ProsQA problems (original answer path 3–4 hops) get a chain of m new concepts hung off each
  option (`make_long.extend`, as in addendum P). The answer path becomes the original path plus the m chain
  statements, and only the base of each option's chain decides the answer. Depth = original + m.
- **Files (md5, first 10 hex):**

  | File | Contents | md5 |
  |---|---|---|
  | `p8_train.json` | 12 000 problems, m ~ U{0..6} (depth 3–10) | 078ca274fe |
  | `p8_valid.json` | 100 problems (Coconut's per-epoch validation) | 92c1dc9ed2 |
  | `p8_test.json` | every short test problem × m = 0..10 | ff5baa26e4 |
  | `p8_ctl.json` | m = 4 and 8, with the chains hung off non-option concepts: same length, original depth | c0ee4b3ade |

- **Test sizes per m (test problems):**

  | m | 0–6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|
  | n | 419 each | 417 | 376 | 245 | 86 |

  m = 9–10 are thinner because unused names run out, and they skew toward smaller graphs, so they are descriptive only.
  m = 7–8 are the beyond-training strata.

## Runs (one cloud GPU each; official Coconut code, speed patch, addendum-G answer mask; `prosqa/pod_p8.sh`)

- **P8-0a no-CoT** (`p8_nocot.yaml`): GPT-2 trained to answer directly, 30 epochs. It is evaluated by candidate
  accuracy (`eval_p8.py --mode nocot`) at epochs 10, 20 and 30 on `p8_test`, and at epoch 30 on `p8_ctl`.
- **P8-0b Coconut** (`p8_coconut.yaml`): addendum-G recipe (answer tokens cannot see the question), one latent stage per
  hop up to 10, 4 epochs per stage, 48 epochs.
  - `long_eval2.py` (mask on) at epochs 48 and 44 on `p8_test`, with K′ ∈ {0,1,2,3,4,6,8,10,12,14,16}, and at epoch 48 on
    `p8_ctl`.
  - The end-of-stage-0 checkpoint (epoch 4, language CoT) goes through `eval_p8.py --mode cot` (greedy generation,
    150 per m) as a CoT reference.

## Definitions and decision rules (fixed now)

- **Accuracy measure.** a₁(m) = P8-0a epoch-30 candidate accuracy at extension m. aᴄ(m, K′) = P8-0b epoch-48
  accuracy at budget K′. The trained budget is K = 10.
- **One-pass capacity c** = the largest m ≤ 6 such that a₁(m′) ≥ 0.90 for every m′ ≤ m (c = −1 if a₁(0) < 0.90).
- **One-pass-hard strata H** = {m ≤ 6 : a₁(m) < 0.75} (in-distribution, deeper than one pass).
- **F1 (feasibility: Coconut does multi-step latent work above one-pass capacity).** H is non-empty, and on H pooled:
  - aᴄ(·, 10) ≥ 0.75;
  - aᴄ(·, 10) − a₁ ≥ 0.15 (latents beat one pass on the same problems).
- **What each outcome sets for PROTOCOL8:**
  - **(i) H empty:** one pass solves every trained depth. PROTOCOL8 needs a deeper format (m beyond 6, a larger name
    pool), and that is reported as the Phase 0 result.
  - **(ii) H non-empty and F1 met:** PROTOCOL8 uses this format and training range (m ≤ 6). The primary beyond-training
    strata are m = 7–8.
  - **(iii) H non-empty and F1 not met:** Coconut does not learn deep problems here. PROTOCOL8's primary outcome becomes
    whether anytime / random-budget training makes H learnable (in-distribution), with extrapolation secondary.
- **Descriptive (no decision attached):**
  - the K′ sweep of aᴄ by m, including monotonicity in K′ above c (Theory's suggested PROTOCOL8 outcome 1) and budget
    dependence (aᴄ(m, K′) for K′ below depth vs at K = 10);
  - beyond training: aᴄ(7–10, K′ = 10..16) vs a₁(7–10);
  - the control file, which separates length from depth;
  - the CoT reference by m;
  - a₁ at epochs 10/20/30, to check that no-CoT training has saturated.
- **Instrument checks, reported:**
  - the pod's `DATA_MD5` matches the table above;
  - `MASK_TEST PASS`;
  - the P8-0b final stage is reached (checkpoint_48 exists).

  A failure of any of these makes that run void, not an outcome.

## Cost and schedule

Two cloud GPUs (RunPod ≤ $0.34/h first, else Vast), about 3 h (no-CoT) and 9–12 h (Coconut), ~$3–5 total. The results
should arrive by ~2026-09-30 08:00, before the PROTOCOL8 confirmatory registration.
