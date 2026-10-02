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


## THEORY §67b (GPT-2 cell, M 24, τ 4→1, bilinear readout) panel result (2026-10-02 01:53 EDT; `python c1/score_67b.py` → `theory/results/score_s67b.json`)

**Panel.** Seeds 1–5 × D1b / TAB / RESET / FP_ex / FP_cmp. FP_cmp is the corrected budget (67b-impl-1); the original is
descriptive. Settings were frozen after the clean seed-99 pilot (c1/PILOT_67b.md). The scorer (94ac692 → fc4a28f) was
committed before any panel result.

**Clarifications, both made on pilot seed 99 before any panel read.**
* C1-4's irrelevant substitution uses codes visited in donor rollouts (Audit).
* The compute-matched budget uses 20M positions (67b-impl-1).

**Evaluation.** Some evaluations OOM'd on shared local cards after training finished. They were re-run from model.pt
(`c1/reeval.py`).

| prediction | result |
|---|---|
| C1-1: D1b ≥ 0.99 at N 8–128 in ≥ 4/5 seeds | **not met: 1/5** (s1 1.000 everywhere; s2 0.33–0.36, s3 0.50, s4 0.31–0.34, s5 0.16–0.30) |
| C1-2: certificate on passing seeds | met (s1: 0 bad pairs) |
| C1-3: nulls and RESET ≤ 0.1867 (upper bound) | nulls met (worst 0.180, s1). **RESET not met as written**: accuracy 0.147–0.182 in every seed (chance 0.167), but with n_test = 2,000 the Wilson upper bound reaches 0.18–0.20 > 0.1867. A sample-size artefact: RESET is at chance |
| C1-4: mechanism (s1) | met: donor 1.000; reachable irrelevant change 0.000 (n = 335; unrestricted 0.248, descriptive); same-semantic agreement 1.000 (171 distinct-code donors) |
| C1-5: FP ≥ 20 points behind at N ≥ 16 (both matchings) | **not met**: median gap 0.16–0.18. D1b's median is near chance, and FP is at chance past N = 8 |
| TAB (descriptive) | **5/5 certified**, 1.000 at every N |

* **Counter check** (s1): H(code | state) 0.21 bits (8 visited codes for 6 states); MI with length mod 2/3/4 ≤ 0.0004
  bits, so no counter.
* **Training seconds** (median, shared GPUs, indicative): D1b 1,679; RESET 1,590; FP_ex 1,906; FP_cmp 2,755; TAB 46.

**Reading.**
* The τ-annealed bilinear recipe that certifies the toy shared cell 10/10 certifies the GPT-2 cell in only 1/5 seeds.
  The other four settle on partial automata: flat 1/3 and 1/2, as in §67.
* Per §67b's rule, the C1 line records **shared-cell acquisition with a pretrained LM as the obstacle**. The certified
  seed shows the GPT-2 cell can hold an exact, counter-free S3 machine (certificate + mechanism), but not reliably.
* §72 (running) asks whether freezing GPT-2 and standardising its features changes this.


## THEORY §72 (where does the GPT-2 cell lose acquisition?) result (2026-10-02 01:54 EDT; `python c1/score_72.py` → `theory/results/score_s72.json`)

20 runs: PF / RF / PF-std / PT × seeds 2001–2005, on Vast 3090s. The scorer (committed before any eval.json was opened)
applies §72's rule. TOY is Theory's CPU arm.

| arm | certified | verdict (§72 rule) | hard accuracy |
|---|---|---|---|
| PF (frozen pretrained) | 0/5 | fails | 0.15–0.19 (chance) |
| RF (frozen random) | 0/5 | fails | chance |
| PF-std (frozen, table-standardised) | 0/5 | fails | chance |
| PT (fine-tuned, new params at 1e-2) | 0/5 | fails | chance |

**But every run collapsed the same way, so the diagnostic does not separate the three causes.**
* In the first curriculum stage (length 1 only) the loss falls (PF 0.78, PT 0.93, PF-std 0.000). When longer sequences
  enter (update ≥ ~2,000–4,000), the loss jumps to ln 6 = 1.79 and stays there.
* The trained cells route every (code, op) to one code: PF s2001 has 1 reachable code, decoded [4, 4, …], with constant
  output. So the per-seed accuracies are identical across arms (the test frequency of one state).
* This is a collapse to a uniform marginal, consistent with dead ReLU units in the bilinear readout
  W·ReLU((U·h_c) ⊙ (V·h_o)) at the new-parameter lr of 1e-2. §72 fixed that lr for every arm. D1b's lr 1e-4 gave 1/5
  with partial automata, not this collapse.
* The model files were not fetched (model.pt excluded), so dead units are not verified directly.
* **Distinctness is still informative:**
  - PF: head inputs stay near-parallel (cos 0.99 → 0.96–0.99).
  - RF: distinct (0.10 → 0.15–0.28).
  - PF-std: decorrelated by construction (−0.04).
  None of the three learns. So feature conditioning alone, at this lr, did not rescue acquisition.
* **Measure flaw (mine).** The PT per-layer between/within-code variance was taken at the code position. Under causal
  attention, that position cannot see the op, so within-code variance is ~0 and the ratio is meaningless (~10^15).
  The op or last position should have been used. Not reported as a result.

**Reading.** The registered rule says "all four fail". The evidence says one optimisation collapse common to all arms at
lr 1e-2. Whether pretrained geometry, fine-tuning or the interface is the obstacle remains open. This is reported with no
follow-up variant, per §72; the decision is Theory's.


## THEORY §72b (the §72 diagnostic at lr 1e-4) panel result (2026-10-02 04:30 EDT; `python c1/score_72.py p72b` → `theory/results/score_s72b.json`)

Seeds 2001–2005 × PF / RF / PF-std / PT, all trained parameters at lr 1e-4. The pilot (seed 990) is recorded separately
as pilot. Placement: 3 local cards (rolling, one job per card) and Vast for seed 2001. The seed-2002 host failed the
GPT-2 sha / image pull; its arms ran locally.

| arm | certified | verdict (§72 rule) | accuracy | ReLU units ever active (of 256) | head-input cosine (normed, end) |
|---|---|---|---|---|---|
| PF (frozen pretrained) | 0/5 | **fails** | chance (0.15–0.19) | 147–165 | 0.985–0.991 |
| RF (frozen random-init) | 5/5 | **succeeds** | 1.000 at N 8–128 | 256 | 0.08–0.10 |
| PF-std (frozen pretrained, table-standardised) | 5/5 | **succeeds** | 1.000 at N 8–128 | 256 | −0.04 (raw 0.99) |
| PT (fine-tuned) | 1/5 | **fails** | s2004 1.000; others 0.18–0.51 | 256 | 0.21–0.72 |

PT per-layer between/within-code variance at the op position: the certified seed rises to ~1.75 from layer 5 on; the
failed seeds plateau at ~0.45.

**Reading** (§72's rules).
* "PF fails, RF succeeds": pretrained geometry is the obstacle.
* PF-std succeeds, so the code information is present in frozen pretrained GPT-2, but its anisotropy (head inputs
  nearly parallel across codes, cosine 0.99) ruins the readout's conditioning. Per-coordinate standardisation over the
  table cures it. About 40 % of PF's ReLU units never fire.
* The trigger for §73 (PF-std ≥ 4/5) is met; §73 is running.


## THEORY §71 (S5 at 4× training length, GPT-2 written-state chunked loop, DeltaProduct's protocol) result (2026-10-02 07:15 EDT; `python boxes/score_s71.py` → `theory/results/score_s71.json`)

**Setup.** One scorer on the same 2,000 test sequences: the first 2,000 of S5=512 from DeltaProduct's generator (seed
7102), with labels asserted at every position. Settings: boxes/S5_PILOT.md (amendment 71-1). Runs: WC0₁/₂/₄ × 5, tagger
× 3, P3 WC0₄/WC₄ × 3.
* **Evaluation reruns.** Three local runs OOM'd in evaluation on shared cards; their evaluation was re-run from model.pt
  (`boxes.s5 eval`, eval_bs 100, recorded). The k = 1/2 certificates and the k = 4 chunk check ran at the rollout's own
  dtype and batch.

| prediction | result |
|---|---|
| **S1** WC0₁ cumulative ≥ 0.99 at t = 512 and k = 1 certificate in ≥ 4/5 | **met**: 1.000 at t = 128/256/512 in 5/5; certified 5/5 (0 errors on all 14,400 (state, op) inputs) |
| **S2** GPT-2 tagger cumulative at t = 256 ≤ 0.50 | **met**: 0.0 at t = 128, 256 and 512 in 3/3 (final-state 0.0075) |
| **S3** vs DeltaProduct | **comparator not reproduced on our setup** (amendment 71-2; boxes/S5_PILOT.md). Only the published figure is cited (n_h = 4, 1 layer: robust to 512, Fig. 5). No exceeds / matches / below wording |
| **S4** P3 WC0₄ final-only ≥ 0.95 final-state at 512 | **not met**: 0.009 (chance 1/120). The final-only solver did not learn 4-op S5 composition in 8,000 steps (loss plateau 0.3–0.8 per target token) |
| **C1** WC₄ − WC0₄ ≥ 0.05 | not met: median +0.0005 (item-paired +0.002 [−0.001, 0.006]); both at chance |

**Descriptive.**
* WC0₂ and WC0₄ (all-position) are also 1.000 at 512 in 5/5. The k = 4 per-chunk accuracy is 1.000 on 100,000 fresh
  chunks, so the predicted p^128 = 1.000.
* **Extended test (500 sequences of length 2,048):** certified WC0₁ (s2) stays 1.000 at t = 512 / 1,024 / 2,048, 16×
  its training length.
* **DeltaProduct and GRU.**
  - The DeltaProduct arms (dp, dpvar) were not reproduced (0.0 in both environments) and are excluded.
  - The GRU (word-problem train_gru, descriptive) was still training at the overnight stop and is not scored here.

**Wording (per §71).** "The first transformer / pretrained-LM result on S5 at 4× training length (512 from 128),
certified at every length" — **scaffolded, different interface**:
* sequential decoding of a written discrete state, with the history cut between chunks;
* 124M parameters vs about 10⁵ for the recurrent comparators;
* inference at 512, k = 1: 3,584 sequential passes and 2.4 TFLOPs per problem.

## THEORY §73 (a stable latent in GPT-2: the standardised-readout cell, 10-seed confirmation) result (2026-10-02 07:15 EDT; `python c1/score_73.py` → `theory/results/score_s73.json`)

The scorer (f128ae3) was committed before any §73 result. Placement: local rolling worker (PF-std, RESET at
n_test = 10,000, PF-ctr); PT-std on Vast (one host failed to start and was re-placed). Statistics were frozen after
training; σ and the dead-unit check are in each eval.json.

| prediction | result |
|---|---|
| **L1** PF-std certified in ≥ 9/10 | **met: 10/10** (1.000 at N 8–128 in every seed) |
| **L2** T0/T1/T2 nulls and RESET ≤ 0.1867 (Wilson upper bound, n = 10,000) | **met**: nulls worst 0.180; RESET within the bound in 10/10 at n = 10,000 (§67b's sample-size artefact resolved) |
| **L3** mechanism in every certified seed | **met**: donor 1.000; same-semantic 1.000; irrelevant-change 0.000 where tested (6 seeds), vacuous in 4 (one visited code per state) |

**Descriptive.**
* PT-std (fine-tuned + standardised readout) is certified **10/10**. PF-ctr (centring only) is certified **5/5**.
* **Counter check:** H(code | state) 0–0.43 bits; MI with length mod 2/3/4 ≤ 0.0015 bits. No counter.
* C1-5 vs the existing full-prompt fpcmp2: +0.82 to +0.84 at N 16–128.
* All 256 ReLU units are active in every seed. σ_min for the code features is ~6e-4 to 1e-3.

**Reading** (§73's wording, within its limits): "A latent finite-state machine hosted by a frozen pretrained GPT-2 is
acquired from final answers alone in 10/10 seeds and certified at every length (S3)."
* The ingredient is standardising GPT-2's anisotropic features before the multiplicative readout. Centring alone already
  suffices.
* Not a claim that pretraining or GPT-2's computation is necessary: RF and DENSE also learn (§72b; §74 tests language
  use).

## THEORY §74 (paraphrase transfer: does the latent machine use GPT-2's language knowledge?) result (2026-10-02 07:46 EDT; `python c1/score_74.py` → `theory/results/score_s74.json`)

The scorer was committed before any §74 eval.json was read. All 20 runs (4 arms × seeds 2021–2025) ran on the local
cards. A descriptive rerun of PF-std (`c1/runs/p74diag_pfstd_s*`, with bad-pair listing) reproduced every seed exactly.

| prediction | result |
|---|---|
| **P1** PF-std certified on the training phrasing in ≥ 4/5 | **not met: 0/5**, so §74 is uninformative by its own rule |
| **P2** median agreement ≥ 0.90 on F2/F3/F4 (F3 needs neg ≤ 0.50) | not met: F2 0.405, F3 0.375, F4 0.458; neg 0.708 |
| **P3** family certificate ≥ 3/5 | not met (no family qualifies) |

**Descriptive.**
* PF-std hard accuracy on the training phrasing: seeds 2021–2023 reach 1.000 at N 8–128 but fail the certificate;
  2024 gets 0.56→0.35 and 2025 0.68→0.66.
* **Bad pairs** (code, state; first length reached; decoded vs true):
  - 2021–2023 have 3–4 each. **All are first reached at length 3, and all are parity flips**: a transposition state
    (1, 2, 5) decodes as an even one (0, 3, 4). The errors sit on codes that hard accuracy at N ≥ 8 does not visit.
  - 2024 has 28 and 2025 has 9, with no parity pattern. A few codes absorb many states, a merge failure.
* Training-phrasing certificates: RF-std 0/5, DENSE-bag 1/5, pool 0/5.
* Median agreement (frozen σ), F1/F2/F3/F4/pos/neg:
  - RF-std 1.00/0.33/0.94/0.33/1.00/1.00;
  - DENSE-bag 1.00/1.00/1.00/0.89/1.00/1.00;
  - pool 0.46/0.61/0.44/0.42/0.97/0.75.
  - The σ-floor changes PF-std's F4 only (0.458 → 0.417).
* " one" vs " 1": wte cosine 0.498; final-hidden cosine 0.979.

**Reading.** Unlike §73's single-token ops (10/10), the multi-token text-op cell at M 24 / 8,000 updates does not
certify. Paraphrase transfer is therefore untested. In 3/5 seeds the failure is a small parity confusion at length-3
codes, not a broken machine. Theory has the bad-pair list.

## THEORY §75 (stable latent scaled to S4) pilot gate (2026-10-02 09:11 EDT; seed 990 only, Vast 3090)

**Gate failed**: neither arm certifies or reaches ≥ 0.95 at N 128, so the panel and A5 were not started. The PF-std
seeds 2031–2035 run provisionally on the local cards are void as panel evidence (Audit's conditions); they are labelled
descriptive only.
* PF-std: acc N 8–128 0.761/0.653/0.543/0.512/0.486, not certified. Code partition: identity block plus 12 pairs.
* PT-std: 0.867/0.777/0.619/0.424/0.233, not certified. Partition: identity block plus 6 blocks of 4.
* In both arms the blocks are neither left nor right cosets. 26 codes are used, and there is no bijection on states.

Evals: `c1/runs/p75_{pfstd,ptstd}_s990/eval.json`. The §73 recipe works at S3 scale only, as tested.

## THEORY §76(B) S4 (amortising free-table teachers into the stable GPT-2 cell) result (2026-10-02 10:07 EDT; `python c1/score_76b.py S4` → `theory/results/score_s76b_S4.json`)

The scorer (5a3c259) and the 76-3 extension were committed before any student result. Placement: local cards (feeder
`scratch/feed76.sh`). Teachers: §76(A) tables made locally (notes 76-1 and 76-5).

| arm | teachers certified | PF-std certified | PT-std certified | verdict |
|---|---|---|---|---|
| **(B), ge3 teachers (primary)** | 0/10 | 0/10 | 0/10 | **fails** (every seed a teacher failure) |
| **B-std (76-2), std teachers, 10 % rows held out** | 10/10 | 0/10 | 0/10 | **fails** |
| 76-3 full arm, std teachers, all rows (descriptive) | 10/10 | 10/10 | 10/10 | descriptive: hosted exactly |

**Descriptive.**
* Every B-std student fits all training and decoder rows in 50–125 updates. The certificate with the held-out rows
  patched to the teacher's entries holds 20/20, so the only defects are the held-out rows.
* Held-out-row accuracy (student next code = teacher's), chance ≈ 1/|R| ≈ 0.03:
  - PF-std median 0.04 (0.012–0.134);
  - PT-std median 0.17 (0.089–0.273).
* So the students copy the rows they were given and do not compose the missing ones. Own-table accuracy at N 128 is
  about 0.04 (chance) for both.
* Full arm: 1.000 at N 8–128, 0 bad pairs, no codes outside R, in 20/20 runs.
* Fix disclosed: the 100 % stop check was made exact (2c0fa96). The two s2052 runs that overran the stop were reset and
  rerun before scoring.

**Reading.** A frozen or fine-tuned pretrained GPT-2 can host an exactly certified S4 automaton, acquired by a tabular
learner from final answers (system level; 76-3 wording: "a pretrained LM can host a certified automaton acquired by a
tabular learner"). It does not generalise the automaton to unseen transitions. The registered (B) and B-std rules are
not met.

## THEORY §76(B) A5 (amortisation into the stable GPT-2 cell, 60-element group) result (2026-10-02 12:19 EDT; `python c1/score_76b.py A5` → `theory/results/score_s76b_A5.json`)

Same scorer and rules as S4. A path fix was made before this verdict: the per-group run prefix (A5 runs are
`p76a5_*`; the first A5 scoring call had read the S4 runs). The verdict logic is unchanged. Students use a row
minibatch (bs 512, full-table statistics refreshed every 10 updates) for 8 GB memory, disclosed.

| arm | teachers certified | PF-std certified | PT-std certified | verdict |
|---|---|---|---|---|
| **(B), ge3 teachers (primary)** | 0/10 | 0/10 | 0/10 | **fails** |
| **B-std, std teachers, 10 % rows held out** | 10/10 | 0/10 | 0/10 | **fails** |
| 76-3 full arm (descriptive) | 10/10 | 10/10 | 10/10 | descriptive: hosted exactly (1.000 at N 128) |

**Descriptive.** Held-out-row accuracy is at chance (about 1/|R|, |R| ≈ 60): PF-std median about 0.005, PT-std about
0.008. Every student copies its training rows exactly, and the held-out-patched certificate holds. As with S4, the
students host but do not compose.

## THEORY §80 premise P0 (text: does GPT-2 compose unseen k-op words in its forward pass?) result (2026-10-02 13:11 EDT; `c1/runs/p80_p0_k{2,3}/eval.json`)

Split committed before training (seed 80 + k, 2ac5ffc); 4,000 steps per Theory's change before any result (f6f6052).
Base GPT-2, lr 1e-4, bs 64, fp32, greedy exact match; local RTX 2070.

| k | training pairs | (b) unseen words | (a) held-out (state, word) pairs of seen words | at step 1,500: train / (b) / (a) |
|---|---|---|---|---|
| 2 | 1.000 (135) | **0.030** (66) | 0.800 (15) | 1.000 / 0.000 / 0.667 |
| 3 | 1.000 (815) | **0.118** (390) | 1.000 (91) | 0.540 / 0.049 / 0.330 |

**Rule (§80): (b) < 0.95 at the final checkpoint, so §80 stops; the cell is not run.**
* GPT-2 fits every training word and generalises to new start states of seen words ((a): 0.80 / 1.00).
* It does not compose unseen k-op words. (b) is below chance (1/6) at both k, i.e. memorisation of words.
* Run note: a first k = 2 attempt at 1,500 steps was aborted before any evaluation and its directory deleted; its log
  was overwritten by the restart. Nothing of it survives, and no result was read (reported to Theory).

## THEORY §78 (GSM8K step-count split: text CoT vs written state, GPT-2 small from base) result (2026-10-02 17:50 EDT; `python gsmlen/score78.py` → `theory/results/score_s78.json`)

The scorer (4fcd147) was committed before any prediction. 9 runs (3 arms × seeds 1–3; 6 epochs, bs 16 × 2, bf16,
checkpoint by short-valid) on 6 Vast RTX 3090s, about $5 (`gsmlen/PILOT78.md`). Bins are on the GSM8K test.

| arm (median of 3 seeds) | in-distribution n 1–2 (422) | mid n 3–4 (654) | **long n ≥ 5 (225)** | n ≥ 6 (87) | aug_long (secondary) |
|---|---|---|---|---|---|
| TC (text CoT) | 0.462 | 0.061 | 0.022 | 0.011 | 0.022 |
| WS (written state, §68) | 0.441 | 0.058 | 0.018 | 0.011 | 0.021 |
| WS-pad (padded to ≤ 8) | 0.441 | 0.049 | 0.018 | 0.023 | 0.020 |

* **L1 (descriptive):** TC long is 0.022 against in-distribution 0.462, a ratio of 0.05. Short-trained GPT-2 collapses
  past its step count.
* **L2 not met:** WS-pad − TC on long is −0.006, CI [−0.021, +0.009], paired by item and pooled over seeds.
* **L3 met:** WS-pad in-distribution 0.441 ≥ TC 0.462 − 0.03.
* **L4:** WS-pad − WS on long is −0.003, CI [−0.015, +0.009]; padding makes no difference.
* **C3:** no win. Median long-bin FLOPs: TC 2.2e10, WS 5.8e10, WS-pad 6.0e10 (states re-encoded each call).
* **Descriptive.**
  - By depth, all arms fall from about 0.33 at depth 3 to about 0.04 at depth 4–5.
  - By final values-list length:
    - WS ends with 2 values in 3,688 of 3,957 problems, i.e. it stops at its trained length (n ≤ 2).
    - WS-pad spreads out, and in 276 problems hits the 12-call cap without stopping.
  - The stop decision, not the state's range, looks like the failure point. This is what §84 tests (oracle stop).

**Reading (§78):** "L2 fails: on GSM8K the bottleneck is not context length or range" at GPT-2 small. The restart
method does not extend reasoning past the trained step count. Written state, padded or not, does no better than text
CoT on long problems.
