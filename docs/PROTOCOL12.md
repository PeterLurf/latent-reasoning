# PROTOCOL12 — annotated-input reasoning in GPT-2 small (THEORY §54, F1): registered outcome "not learnable as specified"

Registered 2026-09-30 22:23 EDT. Theory source: THEORY §54 (bf82897). Implementation: `prosqa/ail_gpt2.py` (original GPT-2 small,
sha256 7c5d3f4b…). Pre-registration tuning under Audit's rules (training data only; ≤ 6 settings; all logged):
`docs/PROTOCOL12_TUNING.md`.

**Outcome.** None of the 6 settings learned within its smoke budget (≤ 594 steps × 4 problems).
* **Settings tried.** Head and label-embedding scales, bias −4 / −2, the hint arm, and a soft warm-up with a depth curriculum.
* **The hint arm.** It also fails, so the failure is not credit assignment from final answers alone.
* **The diagnosis.** A one-batch check shows the cause is the specified hard forward labels with a sigmoid
  straight-through estimator: the loss is piecewise constant and no label crosses its threshold. With soft forward
  labels the same model fits the per-pass targets exactly.
* **Consequence.** The arms AI-mono / AI-free / AI-hint and predictions AG1–AG4 are not run as specified. A redesign
  (e.g. soft labels in training, hardened at test, with a readability check) needs Theory's design and a new
  registration before any test-set run. No test-set result exists; the 12 problems touched by the setting-#0 plumbing
  check are excluded from any later evaluation.
* **Success bar for a redesign.** 2608.11233 is cited with Audit's correction (intermediate-state supervision, no
  matched compute).

## THEORY §54b panel result (2026-10-01 06:26 EDT; `python prosqa/score_p54b.py` → `results/p54b.json`; 9 runs, settings frozen in PROTOCOL12_TUNING.md)

| arm | hardened acc d3–6 / d7–10 / d11–14 (per seed) | AG0 gap | AG1′ | AG2 | AG3 edit IIA | AG4 ρ | cue drop |
|---|---|---|---|---|---|---|---|
| **AI-soft** (primary) | 0.54/0.54/0.50 · 0.50/0.48/0.45 · 0.50/0.50/0.48 | 0.008 | **0/3** | no | 0.36 | 0.19 | 0.003 |
| AI-stoch | 0.39/0.40/0.35 · 0.14/0.12/0.14 · 0.45/0.43/0.42 | −0.005 | 0/3 | no | 1.00 (all labels on) | 0.12 | 0.008 |
| AI-hint (ceiling; per-pass targets) | **1.000 / 1.000 / 1.000 in all 3 seeds** | 0.000 | 3/3 | yes | 1.00 | 1.00 | 0.000 |

* **AI-soft: AG1′ not met (0/3).** Every depth is at chance; the soft labels sit at ≈ 0.5 (binarity 0.04). AG0 is met only
  trivially (both soft and hardened accuracy at chance).
* **AI-stoch: AG1′ not met.** The labels flood: every entity is marked (recall 1.000, precision 0.549), the answer falls to the
  logit tie-break, and accuracy is below chance. Its edit IIA of 1.0 is an artefact of flooding.
* **AI-hint (upper bound, uses step labels, not C4): perfect.** In every seed:
  - hardened accuracy 1.000 at every depth band, including d11–14, far past the trained depth of 6, with P = 14 passes;
  - label precision and recall 1.000;
  - edit IIA 1.000, and fixed-point vs depth ρ = 1.00;
  - the cue control drops accuracy by 0.000.
* **Reading (§54b's fixed readings): "AG1′ failed in both arms".** Final-answer training of an entity-label channel does not
  converge on GPT-2 / ProsQA within this budget, so F1's C4 claim stands on the toy only. But the hint ceiling shows the
  architecture itself runs an exact, readable (C7), halting (C5), depth-extrapolating (C2), cue-free search in GPT-2 once
  the per-pass label rule is learned. The obstacle is credit assignment from final answers, not representation.

## THEORY §61 (F1-S) result (2026-10-01 07:26 EDT; `python prosqa/score_f1s.py` → `results/f1s.json`; 3 seeds, final answers only)

Labels: 1,486 steps × 8 = 11,888 question presentations, matching §54b AI-soft's 11,892. Test: p8_na_test d3–14 (first 100
per depth) plus 200 held-out shallow questions, P = 16, hardened labels.

| seed | acc d1–2 / d3–6 / d7–10 / d11–14 | edit IIA | ρ(fixed point, depth) | label prec / recall | cue drop |
|---|---|---|---|---|---|
| s1 | 0.915 / 0.477 / 0.477 / 0.491 | 0.006 | −0.25 | 0.54 / 0.77 | −0.002 |
| s2 | 0.900 / 0.472 / 0.512 / 0.488 | 0.127 | −0.25 | 0.57 / 0.78 | −0.006 |
| **s3** | **1.000 / 1.000 / 1.000 / 1.000** | **1.000** | 0.75 | **1.00 / 1.00** | 0.000 |

* **Verdicts.** S1 not met (1/3 ≥ 0.95 at d1–6); S2 not met (1/3); S3 not met (median IIA 0.13); S4 not met (median
  ρ −0.25; s3 0.75). S0 met (soft − hardened ≤ 0.03 in every run and band).
* **Reading.** Seed 3, trained with final answers only (no step labels, C4), learned the exact per-entity reachability
  rule: perfect at every depth through 14 (2.3× the deepest training question), labels exactly the reachable set,
  edits behave as BFS predicts, cue-free. Seeds 1–2 learned the shallow questions but not the iterated rule, and their
  labels flood with extra passes. Learnability from final answers is real but seed-dependent (1/3), so the registered
  2/3 criteria fail. Per §61 this was the settling test: no further F1 variants.

## THEORY §61b (F1-S replication) — run record (started 2026-10-01 13:44 EDT; registration da85e97)

* **Code.** prosqa/ail_f1s.py at 248866f, byte-identical to the original §61 runs; same data (f1s_train.json md5
  a8e09288a3) and init (GPT-2 sha256 7c5d3f4b…). Wrapper: `scratch/f1sb_vast_job.sh` writes `host.txt` per run (GPU
  name, driver, torch/CUDA versions, bf16 support), then train → built-in eval → bench prosqa_deep_na --per_depth 150.
* **Placement.**
  - **(a) Seed-3 reruns f1s_r3a/b/c:** Vast offer 31458818, RTX 3090, driver 570.211.01 (the only 570.x 3090 offer;
    the original f1s_s3 ran on offer 31458823, RTX 3090, 570.211.01). **3 concurrent runs on one GPU.** The original
    f1s_s3 trained alone on its GPU, so concurrency (kernel timing, order of atomic operations) is part of the
    nondeterminism arm (a) samples.
  - **(b) Fresh seeds:** s4–s11 on 4 Vast hosts (drivers 550–570), 2 concurrent runs per GPU; s12–s13 on the local
    RTX 3070, 2 concurrent.
* **If all 3 seed-3 reruns fail** (Audit): one solo seed-3 rerun on a 3090 separates "concurrency" from "seed 3 was
  luck". Descriptive only, not a new arm.
* **Hosts (13:44).**
  - R 53732203: RTX 3090, 570.211.01, at 151.237.25.16, the same IP as the original f1s_s3 host, so most likely the same machine.
  - F0 53732210: A4000, 560.35.05.
  - F1 53732219: 3080 Ti, 560.35.03.
  - F2 53732227: 3080 Ti, 550.144.03.
  - F3 53732236: A5000, 560.35.03.
  - Local: RTX 3070, driver 591.86, torch 2.14.0+cu130.
