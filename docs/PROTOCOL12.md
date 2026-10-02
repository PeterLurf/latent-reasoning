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


* **Re-placement of s12/s13 (18:11 EDT; user-approved).**
  - The two local runs shared the 8 GB RTX 3070, spilled into shared memory and stalled: s12 had no log after step 300
    (13:50); s13 ran ~26 s/step.
  - They were stopped and kept as `prosqa/runs_ail/f1s_s1{2,3}.void_3070oom`.
  - Both were rerun with the same frozen recipe on Vast 53761097 (RTX A4000, driver 595.71.05; 2 concurrent; host.txt
    per run), from 18:02.
  - These are fresh seeds, so the class/driver is irrelevant to R2.

## THEORY §66 (boxes, GPT-2 state tracking) panel result (2026-10-01 17:23 EDT; `python theory/score_s66.py` → `theory/results/score_s66.json`)

50/50 runs (10 arms × 5 seeds, 8000 steps × bs 32), bench boxes test (seed 6600, N 1–32 × {all, where} × 50). The scorer
(47e6804, ab0d67f) was committed before any result was read. All arms are complete (5/5 seeds). The CL K′ curve is
pending (local GPU); it is descriptive only.

| arm | acc N 1–8 | 9–16 | 17–32 | GFLOPs / seq passes (17–32) |
|---|---|---|---|---|
| NC | 0.221 | 0.138 | 0.147 | 20.1 / 2.5 |
| TC | 1.000 | 0.180 | 0.139 | 51.8 / 126 |
| TC2 | 1.000 | 0.168 | 0.137 | 36.5 / 66 |
| CL (K′ = N) | 0.177 | 0.135 | 0.140 | 25.1 / 28 |
| **WC k=4** | 1.000 | **1.000** | **1.000** | 43.4 / 57 |
| WC k=2 | 1.000 | 1.000 | 1.000 | 64.1 / 87 |
| WC k=8 | 0.509 | 0.164 | 0.144 | 33.1 / 42 |
| **WC0 k=4** | 1.000 | **1.000** | **1.000** | **34.8** / 26 |
| WC0 k=2 | 1.000 | 1.000 | 1.000 | 51.4 / 50 |
| WC0 k=8 | 0.760 | 0.260 | 0.165 | 26.5 / 14 |

No-reasoning baselines at 17–32: identity 0.151, last op 0.146, majority 0.136, chance 0.146 (pooled).

| verdict | result |
|---|---|
| Gate (NC median, N 5–8, "all") | **passed**: 0.045 (chance 1/24 = 0.042) |
| P1 CL 17–32 ≤ ½ in-range | **not met, uninformative**: CL never learned in range (0.177 ≈ identity 0.176; every seed 0.17–0.24) |
| P2 WC4 ≥ 0.95 at 17–32 | **met**: 1.000 (every seed; "all" 1.000, "where" 1.000) |
| P3 WC4 vs TC, paired, ≤ TC FLOPs | **met**: +0.858 [0.848, 0.867], 43.4 vs 51.8 GFLOPs ("all" +0.962, "where" +0.754) |
| P4 WC4 edit IIA ≥ 0.90 | **met**: 1.000 |
| P5 WC8 − WC0_8 ≥ 0.05 | **not met**: median −0.026 (per seed −0.02 / −0.57 / −0.42 / −0.03 / +0.19); item-paired −0.169 [−0.180, −0.157] |

**Descriptive.**
* **WC0 = WC wherever the solver works.** WC − WC0 at 17–32 is 0.000 for k = 2 and 0.000 for k = 4, in every seed.
  - WC0_4 also meets P3's rule (+0.859, at 34.8 GFLOPs, cheaper than WC4) and has edit IIA 1.000.
  - At k = 8, neither solver learns k′ ≤ 8 swaps in one go. The latents hurt (wc8 in-range 0.32–0.92 vs wc0_8
    0.47–0.97).
* **TC fails length generalisation.** Text CoT trained on N ≤ 8 writes correct states up to 8 ops, then falls to chance
  (0.18 at 9–16, 0.14 at 17–32), even though it is forced to write N states.
  - Two TC seeds are only 0.62 in range.
  - TC2 (state every 2 ops) is the same: partial under P3's rule vs TC (−0.001).
* **Validity.** The final arrangement is a valid permutation in ≥ 0.98 of items for every arm.

**Reading.**
* P2 + P3 + P4 are met: GPT-2 tracks S4 state to 32 ops from short-problem training, more cheaply than text CoT.
* **But the win comes from the scaffold, not from latent computation.** WC0 (no continuous thoughts) does exactly as
  well and is cheaper. The latents do no work at k ≤ 4 (C1 not shown, as Literature expected) and harm at k = 8 (P5).
* What transfers is §64's written-state restart with the history cut. Each chunk's context stays inside the trained
  length, while TC's single growing context leaves it at N > 8.
* The C3 comparison is therefore "chunked text state with a history cut" vs "one long CoT". This is close to
  Delethink / MEM1 (text carry-over with a history cut), not a latent-reasoning C3 result.
* In-range accuracy overlaps the training space (amendment 66-1); every verdict above uses N > 8.

**CL K′ curve (added 18:10 EDT; PILOT.md K′ set, local RTX 3070, bf16).** Accuracy is flat at the baseline for
every budget: K′ = ⌈N/2⌉ / N / 2N gives 0.169 / 0.177 / 0.176 at N 1–8 and 0.142 / 0.140 / 0.139 at N 17–32
(23.0 / 25.1 / 29.4 GFLOPs at 17–32). Extra thoughts buy nothing, consistent with CL never having learned.


## THEORY §65 (restart-distillation on S3) result (2026-10-01 19:18 EDT; `python theory/score_s65.py` → `theory/results/score_s65.json`)

27/27 runs (RD / GT / SELF × the 9 gated §64 parents). The scorer (888a5cf) was committed before any §65 result was read.
* **Placement.** 21 runs on Vast A4000s at 2 concurrent (3 concurrent OOM'd; voided `*.void_oom`); 6 local.
* **Restart.** s3rd_s4 was restarted alone on the 3070 after a VRAM-spill stall on the 4060 (user-approved; partial kept
  as `s3rd_s4.void_4060spill`).

| verdict | result |
|---|---|
| D1 RD single loop at N 9–16 ≥ 0.85 and ≥ SELF + 0.50 in 9/9 | **not met**: median 0.658; 3/9 pairs |
| D2 RD ≥ GT − 0.05 | **met**: RD 0.658 vs GT 0.657 |
| D3 RD valid share at N 12 ≥ 0.95 | **not met**: 0.78 (SELF 0.31) |
| D4 RD in range ≥ 0.97 | **met**: 0.998 |

**Single-loop accuracy by N** (medians):

| arm | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17–24 |
|---|---|---|---|---|---|---|---|---|---|
| RD | 0.97 | 0.93 | 0.87 | 0.77 | 0.62 | 0.47 | 0.37 | 0.25 | 0.06 |
| GT | 0.98 | 0.93 | 0.88 | 0.76 | 0.63 | 0.47 | 0.37 | 0.24 | 0.06 |
| SELF | 0.78 | 0.27 | 0.06 | 0.02 | 0.02–0.03 | | | | 0.01 |

The teacher's mean error at N 9–16 is 0.2–5.7 % per parent; restarts stay at 0.997.

**Reading.**
* The label-free restart teacher works exactly like true labels (D2: RD = GT at every N). Near-clean self-labels are
  not the bottleneck.
* What fails is the single loop itself. Even trained on true final answers at N 9–16, its accuracy decays smoothly with
  N (0.97 at 9 → 0.25 at 16) and is near zero past the distilled range (0.06 at 17–24). Its state validity also decays
  (0.78 at N 12, 0.35 at 16).
* So distillation extends the loop by ~3–4 ops at ≥ 0.85, not to the trained 16 (D1 failed). The loop has no restoring
  dynamics, and its error compounds with length whatever the label source.
* Restarts from written states (§64, §66) remain the only length-robust mechanism here.


## THEORY §61b (F1-S replication) result (2026-10-01 19:38 EDT; `python prosqa/score_f1sb.py` → `results/f1sb.json`)

13/13 runs (3 seed-3 reruns, seeds 4–13). The scorer (c3ee7f1, b0ea23b) was committed before any §61b eval.json was
read. Its failure-mode labels are descriptive: their thresholds were set after the s6–s11 bench outcomes were visible
(disclosed in the scorer).

| prediction | result |
|---|---|
| R1: ≥ 2/3 seed-3 reruns succeed (same class: RTX 3090, driver 570.211.01, as the original s3) | **not met: 0/3** |
| R2: 2–5 of 10 fresh seeds succeed | **not met: 0/10** ("optimistic draw") |

* **Every run floods.** Hardened accuracy at P = 16 is 0.78–0.96 at d1–2, and 0.40–0.56 (chance) at d3–14.
  Label precision is 0.53–0.63 and edit IIA ≤ 0.39.
* 10/13 are exact at their trained budget P = d at d1–6, as §61's s1 was (flag *).
* Bench prosqa_deep_na: 0.488–0.510 for every run.
* The seed-3 reruns ran on the original host's IP and the same driver/torch/CUDA. With 3 concurrent runs on one GPU, all
  three flooded.

**Reading.**
* §61's 1/3 success was an optimistic draw: the frozen F1-S recipe succeeds in 0/13 replications.
* Seed 3's success did not reproduce on the same hardware class, so it was not a property of the seed's init and data
  order alone. Either non-deterministic numerics decide it, or concurrent execution changes it.
* The registered follow-up (one solo seed-3 rerun on a 3090, descriptive) separates these and is placed next.
* **Registered follow-up, solo seed-3 rerun (descriptive; 20:26 EDT).** `f1s_r3d`, alone on an RTX 3090, driver
  570.211.01 (Vast 53771343), floods like the others: 0.93 at d1–2, chance at d3–14, label precision 0.57, edit IIA
  0.19, bench 0.497. Concurrency did not cause the reruns' failure. §61's seed-3 success does not reproduce under the same
  seed, code, data, GPU class and driver, so it was a draw from non-deterministic numerics, not a property of the seed.
