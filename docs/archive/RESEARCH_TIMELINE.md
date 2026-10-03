# Research timeline — how the question moved

Written 2026-09-25 after the Part IV report; Part V (PROTOCOL5, plan steps 1–6) added the same evening. Part VI (PROTOCOL6, §4c, ledger rows, §8) added 2026-09-28 ~19:00, while addenda J (seed 2), M and O were still running. Sources: `docs/RESEARCH_LOG.md` (chronology, exact numbers),
`docs/PROTOCOL*.md` (predictions as frozen), `report/latent_branch_report.html` (Parts I–III),
`report/protocol4_report.html` (Part IV). This file adds no new results; it records *why* each part asked
what it asked, which claims survived, and what to do next.

---

## 0. The original question (proposal)

> Can several graph-search alternatives held in one continuous-thought state be manipulated separately,
> and how does that selectivity change with branching width W?

Setting fixed on day 1 and kept throughout: 10 branches × 6 nodes, W branches on the start hub, the rest on a
shape-matched decoy hub; 2-layer decoder, d = 256, Coconut-style recurrence, K = 6 latent steps; family-disjoint
train / discovery / confirm splits; predictions frozen before confirmation.

---

## 1. Part I — 2026-09-23: "Are the alternatives separately usable?"

**Question as asked:** does a single-branch donor edit on the shared thought move that branch and not the others,
and does selectivity fall with W?

| When | Event | Effect on the question |
|---|---|---|
| Morning | Literature audit; 2606.12689 (Observable Patterns Are Not Explanations) found | Pause-token control becomes mandatory, not optional |
| | Pilots 1–2 stuck at ln 2 (two-match composition saddle) | PARENT, then CHILDREN lookup scaffold + gated depth curriculum. **Claims restricted to scaffolded models** |
| | H-norm pre-registered (softmax sharing, each branch ~1/W) | Gives Part I a mechanistic prediction, not just a width curve |
| | Seeds 0–2 + pause and K = 0 controls, frozen protocol | Answers below |

**Answer:**
* Competence 0.90–0.93. Targeted effect falls monotonically with W (s0: 0.26 → 0.03 from W 1 → 8).
* Signed collateral confirms competition (removing a branch makes the others *more* reachable).
* Probe read-out directions (filters) are causally inert; activation patterns carry the effect → **readable ≠ usable**.
* Strict selectivity bar met only by s2 at W = 2.
* Pause tokens match recurrence on overall accuracy; recurrence helps only at 5–6 hops.

**What it left open:** *why* edits weaken with width — less information per branch, or a different route to the answer?

---

## 2. Part II — 2026-09-24 (day): "Why do edits weaken with width?"

**Question shift:** from *whether* branches are separable to *what mechanism* sets the width law (handoff plan
`docs/NEXT_STEPS.md`: calibration, cache routes, controllability, sparse coding).

* Preflight reproduced all 862 Part I numbers exactly; new instrument `flex.py` (edits at any step, KV overrides).
* PROTOCOL2, P1–P10 on fresh confirm graphs:
  * **Route shift (P1):** with W, 2–3-hop nodes move from the latent pathway to query-time graph reading.
  * **No dose rescue (P2)**, per-branch amplitude ∝ W^−β, β ≈ 0.6–0.9 (P10), composition holds (P9).
  * P8 (geometry adds nothing) falsified for targeted effects in two seeds.
* Post hoc: the width effect is compositional (the latent/query boundary moves inward), not weaker edits per node.
* **First reversal:** replicating the controls on seeds 1–2 showed Part I's control conclusions were seed luck.
  Revised claim: recurrence buys cross-seed reliability and the deepest (6-hop) queries.

**Exploratory results that redirected the project:**
* s2 follows chains **longer than any in training** when given more latent steps; s0, s1 do not.
* Rewinding s0's thought to an early step *unlocks* search past its stop.
* Attention traces: every recurrent seed runs a **clocked breadth-first sweep** in layer 1; width changes its
  amplitude (fixed budget split ~1/W), not its clock.

**What it left open:** the sweep is the mechanism — but is it architecture-general, and what makes it stop?

---

## 3. Part III — 2026-09-24 (night): "Is the sweep general, and what stops it?"

**Question shift:** from the width law to the **sweep itself**: architecture variants (V1 wide, V2 deep, V3 raw
feedback), budget variants (K = 4, 8, 10), held-out seeds, and the halt.

Operational events (all in the log): host crash 03:45 (bad RAM; a flipped bit caught inside `bootstrap_ci`),
second crash 07:18, RTX 3070 faults. Integrity re-run: 7 034 stored numbers agree within 1e-4. Training became
crash-safe (atomic checkpoints, `--resume`). Stage-1 stalls in ~half of attempts.

**Answers (PROTOCOL3):**
* P11 (wider thought steepens nothing) fails; P12 (4 layers → more query-time) fails *in the opposite direction*.
* P15 (in-range gain signals an open loop) fails; P16 (sweep finishing inside the budget → no extrapolation) passes.
* K = 4 models extrapolate (2/2 competent). Seed 5 passes the accuracy gate while using no latent loop → stricter
  gate (addendum J: + depth-matched 5-hop ≥ 0.60).
* **Headline:** the stop appeared to be **keyed to the step embedding**. Freezing the clock let s3 and V2 walk to
  11–12 hops; a model trained without a step embedding extrapolated (P18, one run).
* Complication: K = 8 fades hop by hop instead of halting (P17 fails) — two kinds of limit.

**What it left open:** is the step-keyed halt the cause of non-extrapolation, or does finishing early matter
(Zhang et al. 2607.20594)?

---

## 4. Part IV — 2026-09-24 22:52 → 2026-09-25 14:30: "What keys the stop?"

**Question shift:** a direct test between two accounts, with the attention budget and clock re-tested on fresh graphs.

* 22 runs (12 on RunPod, $2.68 total; 10 locally). Local chains died with the launching session at 00:42 and
  resumed from checkpoints at 08:56.
* **Halt account fails:** no-step-embedding K = 6 not replicated (P19, 0/2 competent extrapolate); random K fails (P22).
* **Finish-early account fails:** P20 fails (the one competent K = 8 run extrapolates); addendum G passes 1 of 3.
* **P21 (decisive cell) unresolved:** 5 seeds, none scorable.
* **Extrapolation looks seed-level:** 4 of 11 competent runs (12 before the Part V scorer fix), unrelated to the
  manipulated factors or to sweep depth E.
* **Mechanism refined:** attention is split per **frontier edge**, not per branch (P25 mixed 14/18; replication 4/4).
  The Part II width law is the frontier-size law seen through W.
* **Mechanism weakened:** a clean one-depth-per-step clock holds in only ~9 of 22 models (P23, P24 fail).
* **Reliability is the bottleneck:** 11 of 22 runs failed the gate (10 before the scorer fix), mostly late-stage (2–4)
  stalls.

---

## 4b. Part V — 2026-09-25 15:00 → 19:05: "Can extrapolation be predicted, and can training be made reliable?"

**Question shift:** from *what stops the sweep* (no manipulated factor did) to *which trained models keep sweeping*,
tested prospectively, plus the clock and the training bottleneck. Plan: §6 below; protocol `docs/PROTOCOL5.md`;
scorecard `docs/PROTOCOL5_RESULTS.md`.

* **Step 1:** the Part IV scorer now applies addendum D's full-depth clause (randk_s1 becomes a gate failure; no
  verdict changes). Reading logs also showed nostep_s0 (P18) never reached full depth — disclosed.
* **Step 2 (exploratory, 22 models):** of 15 in-budget features only **end activity** (frontier attention at the last
  trained step, relative to its peak) separates extrapolators, AUC 0.81. Frozen as P26 with threshold 0.25.
* **Step 3:** a single-step-reader criterion fixed on discovery graphs admits 9 of 22 models; on fresh graphs
  P23′ 7/9 and P24′ 7/9 (mixed) vs 1/13 and 2/13 for the rest. The step-locked clock is a property of that subset.
* **Step 4:** a late-stage restart rule saves ≤ 8% (not a fix). 16-run pilot (no-step recipe, 8 paired seeds):
  lr 5e-4 passes the gate 7/8 vs 5/8 at lr 1e-3 (R1 holds, p = 0.28), at some cost in 5–6-hop accuracy. No stage-1
  stall in any run. Local cards plus three RunPod pods (~$1.7); one pod's container restarted mid-run and was resumed.
* **Step 5:** **P26 confirmed** on the 12 competent pilot runs: end activity < 0.25 → 1 of 6 extrapolate, ≥ 0.25 →
  5 of 6 (AUC 0.83).
* **Step 6:** `report/findings.md` — what stands, what was revised, what is open.

---

## 4c. Part VI — 2026-09-25 19:55 → (running): "What keeps a sweep open, what decides extrapolation, and does it transfer?"

Pre-registrations: `docs/PROTOCOL6.md` addenda A–O; scorecard `docs/PROTOCOL6_RESULTS.md`.

* **Is the open sweep a stable property, and when is it set?** (A, B, E) End activity survived 15 000 more steps with a
  restarted optimizer (A1 stable, 11/12); branching from checkpoints located the decision between stage-4 entry and
  full-depth entry (E1: 2/4 → 5/6 → 6/6). Continuing stalled runs to full depth closed no open sweep (A3), so the Part V
  hunch "less full-depth training keeps sweeps open" fell. P26 held on a second recipe, thinly (P26b).
* **Is precision the hidden cause?** (F, H) An exploratory bf16/fp32 gap motivated same-card pairs; neither
  extrapolation (P27) nor curriculum progress (P28) differed. The gap was a card/recipe confound.
* **What decides extrapolation among open sweeps?** (I, K, L) An exploratory measure, P7 (does the frontier cross the
  trained horizon at test time?), ranked 72 models far better than end activity. Its first prospective test (P29) failed
  as a fixed cut-off with no floor on its denominator; re-registered as a floored ranking it held on 42 new competent
  runs across five cards (P30, AUC 0.993). With two spare thoughts (L) failing sweeps still stop at distance 6: the
  horizon is the training depth, not K. M (training depth 5) is the converse test; O asks whether a longer lookup
  scaffold makes the frontier cross (exploratory basis: runs reaching full depth while the scaffold was on extrapolated
  far more often).
* **Transfer to Coconut GPT-2 on ProsQA** (C, D, G, J, N) — the step deferred since Part III. The official model
  ignores its thoughts (D). Blocking the bypass makes them necessary but they only carry a one-pass answer (G, 3 seeds);
  moving the question after the thoughts (J) and truncating GPT-2 to 4 layers (N) did not produce a sweep either — N
  did not learn the task. ProsQA fits in one 12-layer pass.

## 5. Claim ledger

| Claim | First made | Status now |
|---|---|---|
| Single-branch edits are selective, and selectivity falls with W | I | **Stands** (3/3 seeds, re-confirmed in II) |
| Probe filters are inert; activation patterns are causal (readable ≠ usable) | I | **Stands** (P3, 6–57× at matched norm) |
| Branches compete for a shared budget | I | **Stands**, refined: budget per frontier edge (IV) |
| Budget divides ~1/W among branches | I (H-norm) | **Revised**: 1/frontier-edges; W is a proxy (IV) |
| Pause ≈ recurrence to 4 hops; no-latent fails > 2 hops | I | **Overturned** (seed luck, II) |
| Width moves 2–3-hop nodes to query-time reading | II | **Stands** (P1); not in 4-layer V2 (III) |
| Every recurrent model runs a clocked breadth-first sweep | III | **Weakened**: sweep in all, step-locked clock only in single-step readers (P23′/P24′ 10/13 there vs 1–2/13 elsewhere; IV, V) |
| The stop is keyed to the step embedding | III | **Not supported at training level** (P19, P22; IV). Clock-freeze rescue on s3/V2 still stands as an intervention result |
| Sweeps that finish inside the budget don't extrapolate | III (P16) | **Fails** at K = 8, 10 (IV) |
| Extrapolation is a property of the recipe | implicit, III | **Replaced**: seed-level outcome (IV), predictable in-budget from end activity (P26 confirmed, V) |
| A restart rule can fix late-stage stalls | IV next steps | **No** (≤ 8%); lr 5e-4 helps (7/8 vs 5/8, V) |
| End activity predicts extrapolation | V (P26) | **Weakened**: near-necessary, weak on new runs (AUC 0.53, 0.67); P7 ranks far better (VI) |
| The open/closed outcome is set late, at full depth | VI (E1) | **Stands** (A1 11/12, E1 6/6), with slow drift in ~1 run in 8 |
| Precision (bf16 vs fp32) affects extrapolation or stalls | VI exploratory | **Not supported** (P27, P28) |
| Whether the frontier crosses the horizon decides extrapolation | VI exploratory | **Stands as a ranking** (P30 AUC 0.993, L1 0.90); failed as a fixed cut-off (P29) |
| Failing sweeps stop at the training depth, not the step budget | VI (L) | **Supported** (L2 12/13); converse test M running |
| The sweep transfers to Coconut GPT-2 on ProsQA | VI | **No**: thoughts unused (D) or a one-pass conduit (G, J); 4 layers don't learn it (N) |

The arc: **separable?** (partly, less with W) → **why less with W?** (shared budget, route shift) →
**what is the mechanism?** (a clocked sweep) → **what stops it?** (not the step index, not finishing early) →
**which models keep sweeping?** (those whose sweep is still active at the last trained step — visible before any
extrapolation test). → (VI) **what decides it?** (set late in the curriculum; among open sweeps, whether the frontier
crosses the trained horizon, which stops at the training depth) → **does it transfer?** (not to Coconut GPT-2 on ProsQA,
which solves the problem in one pass).

---

## 6. Plan steps 1–6 (proposed after Part IV; all carried out in Part V — outcomes in §4b)

Ordered by information per GPU-hour. Freeze each prediction before it runs, as before; Part IV training is closed,
so new predictions go in a new `docs/PROTOCOL5.md`.

### Step 1 — Housekeeping (minutes, no compute)
* `docs/PROTOCOL4_RESULTS.md` "Open" list of d7 ≥ 0.60 runs omits nostep_k8_s1 (0.709).
* Make `latent/protocol4_confirm.py` treat "final stage < full" as a gate failure (addendum-D wording; randk_s1).

### Step 2 — Explain extrapolation from models already trained (evaluation only, ~hours)
The largest open question is answerable without training: 22 J-competent recurrent models exist (the addendum-E
set), 8 of which extrapolate (s2, budget_k4_s0/s2, nostep_s0, budget_k8_s1, nostep_k8_s1, budget_k10_s3, v3_rawfb_s1).
* Measure per model, with existing tools (`clock_trace.py`, `delayed_block.py`, `confirm_sweep.py`,
  `budget_sweep.py`): halt sharpness (frontier-attention collapse after the last trained depth), per-hop
  attenuation, clock regularity, E, latent-pathway share, depth-matched 5–6-hop accuracy.
* Compare same-recipe siblings first: nostep_k8_s0 vs s1; budget_k10_s3 vs s1/s2; budget_k8_s1 vs s0.
* **Deliverable:** one candidate predictor of extrapolation, stated with a threshold, frozen in PROTOCOL5.
* **Stopping rule:** if no measure separates the groups (e.g. AUC < 0.8 over the 22 models), report
  extrapolation as seed-level and do not train more seeds for it.

### Step 3 — Re-test the clock under a strict criterion (evaluation only, ~minutes)
* Freeze a single-step-reader criterion (e.g. off-diagonal |Δ| ≤ 0.02 on discovery graphs) *before* blocking on
  confirm graphs, then re-score P23/P24 on the models that qualify. Settles whether "step-locked clock" is a
  claim about a subset or should be dropped.

### Step 4 — Fix late-stage stalls before any new training (~1 day of mostly local compute)
* Fit time-to-advance per curriculum stage from the existing `log.jsonl` files and set a stage 2–4 restart rule
  from that distribution (no training needed to set it).
* Then a small pilot: lr 5e-4 vs base, 6 seeds each, full 30k steps, no-step recipe (the one that stalls). About
  12 runs × ~1.5 h; on cloud ≈ $4 at $0.17–0.20/h, one run per pod.
* Use the fixed recipe only for runs answering *new* questions (as TRAINING_TODO already says).

### Step 5 — Prospective test of the Step 2 predictor (only if Step 2 finds one)
* At a ~1/3 extrapolation rate, ~12 competent runs are needed to test a predictor; with the fixed recipe and
  ~1.3 attempts per run that is ~16 runs, ≈ $5–6 on RunPod. Predict each run's outcome from its trace
  before measuring extrapolation.

### Step 6 — Consolidate a write-up of what stands
* Parts I–IV now support a coherent core: readable ≠ usable; per-frontier-edge attention budget and the
  resulting width law; the route shift; recurrence buying deep-query reliability. Write these up with the claim
  ledger above, including the reversals — they are part of the result.

### Deferred — Phase 4 (Coconut GPT-2 on ProsQA)
* The only step that tests transfer beyond the toy model and the only one needing a rented 24 GB card.
  Worth doing, but after Steps 2–4: without a reliable recipe and a clear extrapolation account, a new setting
  multiplies the open questions rather than answering one.

### Not recommended
* More replication seeds for P19–P22 on the current recipe: the Part IV contingencies are exhausted, and at a
  ~45% gate-failure rate each added seed buys little.

---

## 7. Next steps after Part V

*All four items below were taken up in Part VI (§4c): why seeds keep sweeping (A, E, F, H, I, K, L), the clock (P23″/P24″
mixed / confirmed at the bar), the lr 5e-4 recipe (used throughout), transfer (D, G, J, N). Current next steps: §8.*

* **Why some seeds keep sweeping.** lr 5e-4 runs had high end activity more often (5 of 7 competent vs 1 of 5), and
  two stage-4 stalls (nostep_s0, rel_base_s23) extrapolate strongly. Candidate pre-registered test: learning rate and
  full-depth training time as causes of end activity, with end activity measured at checkpoints (`development.py`).
* **Make the clock claim precise.** P23′/P24′ were 0.78 against a 0.80 bar; with 13 single-step readers overall, one
  more prospective set would settle "confirmed" vs "mixed".
* **Recipe for new-question runs:** lr 5e-4 (fewer late stalls, somewhat lower deep-query accuracy); keep lr 1e-3 for
  replications of earlier cells.
* **Transfer (Coconut GPT-2 on ProsQA)** is now the main untested step: do step-specific blocking and end activity
  carry over? Needs a rented 24 GB card.

## 8. Next steps after Part VI (2026-09-28; running work in `docs/CURRENT_TASKS.md`)

* Score M (training depth 5: does the horizon move with it?), O (does a longer scaffold make the frontier cross?) and
  J seed 2 (J1) when they finish.
* If O is supported, the account has a training-level cause for the extrapolating solution; if not, look at what
  happens between stage 4 and full depth (E1's decision window) in the checkpointed P30 runs.
* ProsQA: a pretrained model will only need serial latent steps on problems deeper than one forward pass (longer
  chains), or with the toy model's recipe (scaffold, gated curriculum) — N shows removing layers is not enough.
* Write-up: `report/findings.md` findings 8–10 hold the Part VI results; a Part VI report page is not built yet.
