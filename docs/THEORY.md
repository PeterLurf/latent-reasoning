# Why some latent reasoners stop at the training depth — mechanism, theory, architectures

Theory stream, started 2026-09-29 ~01:10 EDT (separate Claude session; experiments and project flow are run by the
other session — see `docs/CURRENT_TASKS.md`). Everything here is **exploratory** (measured on existing models after their
labels were known) unless a section says it was pre-registered. Code: `theory/*.py`; outputs: `theory/results/`. All
measurements run on the CPU on existing `runs/` models; nothing here touches the queues or the pre-registrations.

**Self-contained write-up of the general theory:** `docs/THEORY_FRAMEWORK.md`.

**Contents:** Executive summary · Timestamp correction · §0 question · §1 pointer · §2 retrieval (Mechanism 1) ·
§3 theory · §4 attenuation (Mechanism 2) · §5 architectures · §6 Pilot A pre-registration · §9 Pilot B · §10 developmental
race · §11 Pilot C · §12 training-free repair · §13 GPT-2 latent reasoners · §14 transfer curves (+ failed out-of-sample
rule) · §15 Pilot A scored · §16 Pilot B interim · §17 Pilot D · §18 formal sketch · §19 Pilot B scored · §20 Pilot E ·
§21 O marker test · **§22 four conditions C1–C4, Props. 4–8, new methods T6/T7 and architecture A6** · §23 Pilot F · §24 Pilot G (A6) · **§25 Prop. 9: slack creates the clock** · §26 CTM-inspired T8/A8 · §27 Pilot H (fine-tune repair) · §28 Pilot I (periodic clock) · §29 Pilot J (chunk-reset attention) · §30 zero-cost follow-ups · §7 log · §8 toy simulation. (Sections were appended in the order written, so numbers are not in
page order.)

## Timestamp correction (2026-09-29 12:59 EDT, prompted by the Audit session; machine clock via `date`, Vast logs in UTC = EDT + 4)

Times written in this document before this note were my own estimates, not read from the clock; many run 30–110 min
ahead of the commits, one runs behind. The original text is left unchanged. Authoritative times are the git commit
times below (local EDT) and the first-run times from the host queue logs.

| item (as written) | stated time | commit that contains it | first run of that pilot started | first result |
|---|---|---|---|---|
| §6 Pilot A pre-registration | ~02:05 | 7f43b08 **01:57:04** | **01:53:19** (host 1), 01:54–01:56 hosts 2–4 | 02:27:40 |
| §6 amendment A-1 | ~05:50 | 84fa0c1 05:57:30 | (re-runs pa_mb2 after 05:57) | — |
| §9 Pilot B pre-registration | ~06:15 | 84fa0c1 **05:57:30** | 06:28:59 | 07:22:05 |
| §11 Pilot C pre-registration | ~06:35 | 5d0e1bc **06:00:32** | 10:03:24 | 10:32:30 |
| §14 out-of-sample rule check | ~07:40 | 044c1b9 **09:10:10** | — | — |
| §15 Pilot A scored | ~09:15 | 457cf81 09:15:21 | — | — |
| §16 Pilot B interim | ~09:40 | 9f4ea6b **09:16:37** | — | — |
| §17 Pilot D pre-registration | ~09:50 | f05c4b1 **09:17:22** | not started before 11:55 | — |
| Executive summary | ~10:30 | 3a81019 **09:30:18** | — | — |
| §19 Pilot B scored | ~11:45 | a0d3648 11:43:59 | — | — |
| §20 Pilot E + amendment D-1 | ~12:40 | 3ded4f5 **11:49:57** | 11:55:24 (pe8_at_s1006, host 4) | 12:57:31 |
| amendment E-1, §21 pre-registration | ~13:00 | b895d96 **11:50:43** | — | — |
| §21 PO3 added | ~13:15 | f12c4ca **11:51:36** | — | — |
| §21 scorer written | ~13:25 | a898bcf **11:52:27** | — | — |
| §21 scored | ~13:50 | 3ff40d8 **11:54:27** | — | — |
| scope note (addendum Q) | ~14:10 | eb2e2f6 **12:25:40** | — | — |

* **Order of registration and data, corrected:** Pilots B, C, D, E and amendments D-1, E-1 were committed before any of
  their runs started. **Pilot A's text was written before the hosts were deployed, but its commit (01:57:04) is 4 min
  after the first Pilot A run started (01:53:19)**; no Pilot A run produced any output before 02:27:40, so no outcome
  could have informed it — but the commit record alone does not prove the order.
* **§14's claim "frozen before any Pilot model was analysed" is inaccurate.** Before the rule was written (~07:00),
  Pilot A pointer trajectories (analyse_pilotA.py) and the Coconut-arm retrieval shares (analyse_pilotA_shares.py) had
  been computed (~06:10); no Pilot A transfer curve had. The retrieval threshold (0.5) predates Pilot A (§2); the
  transfer-curve criterion was fixed after three P30 curves. The out-of-sample test failed (8/15), so nothing was inflated.
* **§21 is out of sample for the marker only.** Addendum O's outcomes had been scored and published by the experiments
  session (~10:05, results/p_o.json; PROTOCOL6_RESULTS) and its aggregate (O1 13 vs 13; P7 AUC 0.99) was in a message I
  had read before §21 was written. I had not read any per-run O label and had not measured any O run; only the marker
  was unmeasured. It is not a blind prospective test.
* From here on every time in this document is taken from `date` or the commit.

## Executive summary (updated 2026-09-29 14:27 EDT)

**Question.** Why do some Coconut-style latent reasoners follow chains past their training depth and others stop there?

**Answer so far (two mechanisms, one cause):**
1. **The thought is a pointer** that must move one hop per step (§1). In failing seeds it breaks at the first untrained
   step (K + 1), not at a fixed depth.
2. **Mechanism 1 — step-indexed schedules.** Most failing seeds write the pointer by *retrieving* an earlier thought
   with a step-dependent lag (§2). Blocking that retrieval only past the trained budget (provably no in-distribution
   effect) rescues 17/23 such seeds; 7-hop success 24 → 41 of 59 (46 with a mild gain boost, §12). **Causal.**
3. **Mechanism 2 — graded vs restoring gain.** Even local pointers fade ~×0.8 per hop unless the local map is restoring
   (clamps to a stable fixed point; §4, §14, toy §8). Among retrieval failures, the block rescues 9/12 whose hidden
   local map is restoring vs 1/11 otherwise. **Causal/dissociation.**
4. **Cause: slack.** When the latent budget exceeds the hops needed, any schedule that visits each depth once fits the
   loss, so the continuation past K is unidentified (§18 Prop. 1). The circuit is chosen in curriculum stage 3 (§10);
   retrieval ≥ 0.25 at the end of stage 3 → 0/12 extrapolate in-sample, but the marker **failed out of sample** on addendum O (AUC 0.51 vs early P7 0.92, §21). The K dose-response form of the account **failed** in Pilot C
   (K5 3/4 = K7 3/4 extrapolate; PC1 not supported, §11 result), though retrieval share does rise with K.
5. **Slack creates the clock (§25, Prop. 9).** With slack, the front reaches only the demanded depth by step K (p[K−1] ≈ 3
   at K = 5, 6, 7 and 8), i.e. it runs at a fractional speed; a pointer-state transition can only realise that with a
   phase variable — a clock (history count, step embedding, or a decaying amplitude) — whose continuation past K is
   unidentified. Enforcing tightness removes the need for any clock (links C1 and C2 below). Pre-registered tests LE1–LE4
   and LE-R (out of sample on addendum R) are pending.

**General theory (§22): four conditions for extrapolation in the number of latent steps** — C1 identification (tight
contracts during circuit selection; Prop. 4 — random, matched and per-step budgets are one mechanism, refined in §22.7
after Kuo et al.), C2 homogeneity (no usable clock), C3 stability (a restoring map; Prop. 5), C4 idempotence at
completion (Prop. 6). No Coconut-family method targets C3 or C4. Prop. 8: latent tokens are bypassed whenever the
problem fits in one forward pass (slack = K), so extrapolation in latent steps can only be tested above one-pass capacity
(the basis of the experiments session's PROTOCOL8). New methods: **T6 amplitude jitter** (C3 through the loss),
**T7 selection-window supervision**, **A6 monotone latent set** (C3 + C4 by construction) — Pilots F and G (§23, §24).

**What failed or did not hold (reported, not buried):**
* Markov architecture alone (no latent→latent attention): no gain, harder to train (Pilot A: PA1 unresolved, PA2 not
  supported, PA3 fails; §15). Failures reappear as fade-triggered restarts.
* The one-step two-factor rule: 18/23 vs 7/36 in-sample but **8/15 out of sample** (§14). P7 remains the reliable
  predictor.
* The toy's inference-time repair does **not** transfer to public GPT-2 latent reasoners (CODI, SIM-CoT): they peak
  at their trained 6 latents, rely on earlier latents, and masking them past the budget slightly hurts (§13).

**Most promising fixes (small n, pre-registered tests running):**
* **Anytime (minimum-progress) supervision** — the answer after every latent step must know everything within k + 1
  hops: produces a one-hop-per-step front with no dwell (e.g. `1 1 2 3 4 5 6 7 8 9 10 11`) in 3/4 competent runs;
  2/3 extrapolate so far (§16, Prop. 2). Confirmatory Pilot D (12 + 12 seeds) queued.
* **Unity contract** (budget matched to curriculum depth): 2/2 competent Markov runs extrapolate, incl. the pilot's best
  (0.77 at 8–10 hops; §15).
* A complete fix likely needs **both** no slack and a restoring map (§18 corollary; Pilot B arms MR, ATR pending).
* **Scope (added ~14:10 after the experiments session's addendum Q):** the slack account applies to models whose thoughts
  carry a *step schedule* (the toy). GPT-2 Coconut on ProsQA computes in one pass: trained only on 3–4 hops it is as
  accurate on 5–6 hops (0.947 vs 0.951; 0.928 vs 0.938) and 12 thoughts add nothing (Q1, Q2 not supported;
  `docs/PROTOCOL6_RESULTS.md`) — no training-depth wall, because there is no schedule to run out. Its limit is
  chain-following past the answer options (addendum P), a one-pass capacity limit. Our CODI / SIM-CoT GSM8K probe (§13)
  fits the same picture: extra latents neither help nor expose a hidden stepwise operator.

**Running (as of 14:27):** Pilots D/E (anytime vs control, 28 same-seed pairs; ~01:00–05:00 EDT 2026-09-30), then F
(jitter, jitter + anytime, selection window; 32 runs) and G (set memory; 16 runs) on 4 Vast GPUs (~$0.60/h), until
~12:00–14:00 EDT 2026-09-30; two G feasibility runs on the experiments session's local GPUs. Pilots A, B and C are scored. Proposals for the field: `docs/PROPOSALS.md` (§8 = second round).

## 0. The question

In the toy setting (2-layer Coconut-style reasoner, graph reachability, delayed query, K = 6 or 8 latent steps, training
graphs of depth ≤ 6), about half of competent seeds follow chains longer than any in training when given more latent
steps, and half do not. PROTOCOL6/7 established *that* the failing frontier stops near the training depth (L2, M1) and
that whether the frontier crosses it (P7) predicts extrapolation (P30 AUC 0.993). This stream asks **why** — what in the
network makes the sweep stop — and then what architecture or training would remove the cause.

Population used throughout: the 59 competent P30 ∪ L models (25 extrapolate, 34 do not), labels from
`results/p30.json` and `results/p_l.json`; probe graphs = the long-chain graphs of `latent/length_gen.py` (chain A at
distance 1–6 continued by chain B at 7–12; look-alike unreachable chains D and C), fresh seed 4247, 512 graphs.

## 1. What the latent state is: a pointer that moves along the chain

Layer-1 keys of edge tokens depend only on the edge's own (src, dst) embedding, so the latent query's logit on an edge
u→v measures how strongly the thought fed into that step "points at" u. Membership of the node at distance j in thought
h_{k−1} = logit(edge out of it) − logit(matched decoy edge), for the model's sweep head (`theory/exp2_membership.py`).

* In **every** model the thought is a sharp **pointer**: at each step it points almost exclusively at one depth of the
  chain (logit excess 4–10 on that depth, ~0 elsewhere). Two to three steps per depth early on (dwell), then faster.
* **Extrapolators**: the pointer keeps advancing one depth per step past K (e.g. p30_s105: depths 1 1 2 2 3 4 | 5 6 7 8 9
  10), with a slowly shrinking amplitude (logit ~7 → ~2 over 12 steps; §4).
* **Non-extrapolators** fall into four patterns (`theory/analyse_pointer.py`, `theory/results/pointer_taxonomy.json`):

| Pattern after step K | Non-extrapolators (34) | Extrapolators (25) |
|---|---|---|
| **reverts**: the untrained last thought h_K points to a *shallower* depth, then a 2-cycle (h_k = h_{k−2}) | **18** | **0** |
| advances (but restarts from depth 1 periodically, advances too slowly, or fades out) | 10 | 21 |
| stalls (pointer stays at one depth) | 4 | 4 |
| dies (pointer amplitude → 0) | 2 | 0 |

In failing K = 6 **and** K = 8 models the break happens at step K + 1, the first untrained step — not at a fixed depth.
The "training-depth horizon" of L2/M1 is where the front happens to be when the trained steps run out.

## 2. Mechanism 1 — the pointer is *retrieved* from earlier thoughts on a step-bound schedule

Each thought h_k = ln_f(x_k) and the final residual x_k is an exact sum of components: the input thought (residual
carry), each attention layer's reads split by key source (edge tokens, START, each earlier latent position by lag, self),
and the two MLPs. With both LayerNorms linearised at the actual input, each component's contribution to the next step's
pointer is computed exactly (`theory/exp3_attrib.py`, `theory/exp4_shares.py`).

* **Extrapolators compute the pointer locally.** In p30_s105 and p30_s155 the new pointer is written almost entirely by
  the MLPs of the current step (mlp2 ≈ 80–90 %), from the children that the same step's layer-1 attention just read. That
  transition — "read the edges out of the nodes I point at, point at their children" — does not depend on the step
  index, so it keeps working past K.
* **Most non-extrapolators retrieve the pointer.** In p30_s115, layer-2 attention to an *earlier latent position*
  writes the pointer, with a lag that changes from step to step: h_4 copies lag 2, h_5 lag 1, and **h_6 (the untrained
  last thought) copies lag 4 — the depth-2 result of step 2**. That is the reversion. The sweep is a learned delay line
  addressed over latent positions; its schedule is only defined on the steps training used.
* Across the 59 models (share of the pointer written by attention to earlier latent positions):

| | Extrapolators | Non-extrapolators |
|---|---|---|
| retrieval share of the untrained last thought h_K | 0.08 (mean) | 0.56 |
| trained steps where retrieval writes > 50 % of the pointer | 0.12 per model (2 of 25 models have any) | 1.12 per model (23 of 34) |

  AUC for extrapolation: retrieval share at h_K 0.82 (as a negative predictor), local share at h_K 0.83; P7 0.976 on the
  same models. (P7 reads the outcome on the far chain; these read the mechanism inside the training budget.)

### Causal test: block retrieval from step K on (`theory/exp5_block.py`, `theory/analyse_block.py`)

From step K onward, layer-2 attention of the latent position may not see earlier latent positions (it still sees the
edges, START and itself). Answers at the trained budget K′ = K are **mathematically unchanged** (the layer-2 keys and
values of position K come from its layer-1 output, which the block does not touch); only the untrained last thought and
later steps change. Rollout to K′ = 12, queries read the full cache.

| Group | n | mean accuracy at distance 8–10: base → blocked | improved ≥ 0.05 | ≥ 0.60 after |
|---|---|---|---|---|
| retrieval-type non-extrapolators (retrieval share ≥ 0.5 on a trained step or at h_K) | 23 | 0.501 → **0.585** | **17** | 10 |
| local-type non-extrapolators | 11 | 0.509 → 0.531 | 2 | 1 |
| extrapolators | 25 | 0.573 → 0.609 | 8 | 14 (8 before) |

* 7-hop accuracy ≥ 0.60: **24 → 41 of 59 models**; no extrapolator got worse.
* In rescued models the pointer advances one depth per step to distance 10 (e.g. p30_s147: 4 4 4 4 4 4 4 → 4 5 6 7 8 9
  10; 8–10-hop 0.50 → 0.69). The generalising local transition **already exists** in these networks; at untrained steps
  the retrieval schedule overrides it.
* The dissociation is the evidence: the block helps the models whose pointer is retrieved and not the ones whose pointer
  is computed locally (those fail for other reasons, §4).
* Earlier, weaker version (`theory/exp1_window.py`): hiding all but the last K − 1 latent positions from steps > K turns
  some 2-cycles into slow advances (e.g. p30_s146 7-hop 0.59 → 0.67) — the growing latent context feeds the schedule.

## 3. The theory so far — a schedule-bound recurrence

1. A Coconut-style latent step can produce the next thought in two ways: **locally** (a content-addressed update of the
   current pointer; time-homogeneous) or by **retrieval** (attention to earlier latent positions; the choice of lag can
   depend on how many latent positions exist — an implicit clock, since causal attention reveals the count).
2. The curriculum trains shallow graphs with more steps than hops (stage 1: depth ≤ 2 with K = 6), so the pointer must
   **dwell**. Dwelling by retrieval (copy an earlier pointer) is an easy solution; it ties the dynamics to the step index.
3. The loss constrains a thought only as the input of a later step or through the keys/values that queries read. The
   **output of the last trained step is never used** (no step consumes it) — it is "dangling" — and nothing beyond K is
   constrained. A step-indexed schedule has no defined continuation, and at step K it retrieves a stale pointer.
4. So the front stops wherever it is at step K. Training makes that the training depth (the budget is spent reaching
   it), which is why L2/M1 see a "training-depth" horizon: horizon = depth reached when the schedule runs out.
5. Models whose pointer is always computed locally have a time-homogeneous transition and continue past K; their reach is
   then limited by **attenuation** (§4). P7 is the downstream symptom of both.
6. Which solution a seed finds is set during the late curriculum (PROTOCOL6 E1, PROTOCOL7 T3), consistent with a
   competition between the two routes while dwell is still needed.

Relation to prior work: Bansal et al. 2022 (progressive loss "prevents the model from learning behaviors that are specific
to iteration number"); Kuo et al. 2026 (spurious length–loop correlation; stochastic stopping); Anil et al. 2022 (path
independence predicts upward generalisation); Rodionov & Prokhorenkova 2025 (hard attention + discrete state give
provable size generalisation); the Markovian Thinker (2510.06557; bounded carried state). None of these identifies
*retrieval from earlier latent positions* as the step-bound component in continuous-thought transformers, or rescues a
trained model by blocking it.

## 4. Mechanism 2 — attenuation (in progress)

Even the local transition loses amplitude: extrapolators' 8–10-hop accuracy averages only 0.57.
(`theory/exp6_gain.py`, `theory/analyse_gain.py`.)

* **The loss is per hop, not per step.** While the pointer dwells at one depth its logit holds (p30_s155: 6.25, 6.89,
  6.75 at depth 1; 7.23, 6.71 at depth 2); each advance multiplies it by ~0.8 (6.7 → 5.3 → 4.1 → 3.4 → 3.1 → 2.6 → 2.3).
* **Not a dilution of the frontier read.** The sweep head's attention mass on the frontier edge stays ~0.2–0.3 while the
  pointer shrinks; the gain is lost in turning the children just read into the next pointer (the local write). Training
  needs only ~4 advances above threshold, so nothing pushes the loop gain to 1 — and a gain ≥ 1 would amplify noise on the
  shape-matched decoys.
* **Width sets the starting amplitude.** The pointer is weaker the more branches share the frontier (W = 2 ≫ 4 > 6, the
  per-edge budget law), and it does not recover when the frontier shrinks to one node past distance 6.
* **Death = sink capture.** In p30_s126 the pointer drops below the competition at step 6; the sweep head's attention
  then drains to earlier latent positions, START and itself (0.8 on latent positions by step 8) and the front is gone.
  Blocking layer-1 latent attention does not save it (the mass goes to START/self).
* After the retrieval block, reach follows the post-K front gain (Spearman 0.77 over 59 models; gain ≥ 0.9: mean 8–10-hop
  0.61, below: 0.53) — a consistency check, not a prediction (both come from the far chain). An in-distribution gain
  estimate (2–3 advances inside training) is too noisy to predict anything (AUC 0.56).

**Two-factor account.** Extrapolation needs (i) a *local* (content-addressed, step-independent) transition — otherwise
the front stops when the retrieval schedule runs out (Mechanism 1) — and (ii) a loop gain near 1 on that transition —
otherwise the front fades within a few hops of the training depth (Mechanism 2).

## 5. Architectures and training changes suggested by the mechanism (to be tested)

* **A1 Markovian latent step** — the latent position attends to the prefix (edges, START) and itself only, never to
  earlier latent positions; queries still read every position. Removes the retrieval route and the implicit clock by
  construction; the thought must carry the state. (Inference-time version already rescues 17/23 retrieval failures.)
* **A2 No dangling thought** — progressive / random-budget training (Bansal et al. 2022) with queries after every budget,
  so every thought is both a readout state and the input of a further step; or an auxiliary loss on step K + 1.
* **A3 Pointer restoration** — a gain-restoring nonlinearity on the node-id part of the thought (e.g. sharpen the
  pointer's logits over node ids and re-embed), so amplitude does not decay with steps (targets Mechanism 2).
* **A4 Non-competitive edge reading** — sigmoid (or stick-breaking) attention for the frontier read, so per-edge mass does
  not depend on how many other keys compete (per-edge budget law, softmax dispersion; Veličković et al. 2024).
* **A5 Budget-matched curriculum** — K grows with the curriculum depth so the pointer never needs to dwell.

## 6. Pilot A — architecture and schedule variants (pre-registered 2026-09-29 ~02:05 EDT, before any run started)

**Question.** If the retrieval route (Mechanism 1) causes the horizon, a latent step that *cannot* read earlier latent
positions should extrapolate far more often, with the recipe otherwise unchanged.

* **Recipe** (all arms): the P30 recipe — `--no_step_emb --K 6 --lr 5e-4 --amp bf16`, bs 512, 30 000 steps, gated
  curriculum, lookup scaffold 10 000 + 5 000 anneal, children weight 1, gpt2 init, stage-1 guard (3 relaunches).
  Trainer `theory/train_arch.py` (= `latent/train.py` + options), evaluation `theory/eval_arch.py` (the project's
  competence, matched-depth and length_gen code with the loader swapped; same seeds and definitions as P30).
* **Arms and seeds.** C = coconut (control), seeds 700–707; M = markov (both layers; latent positions never attend to
  earlier latent positions, queries read everything), seeds 710–717; secondary: M2 = markov_l2 (layer 2 only) 720–723;
  MB = markov + budget-matched curriculum (K = min(6, stage depth + 1) during training) 730–733; OV = coconut + overrun 2
  (training budget drawn from 6..8 each step) 740–743. Four Vast.ai hosts, every host runs every arm (card balanced).
* **Definitions** (as PROTOCOL6): competent = gate (delayed ≥ 0.85, matched 5-hop ≥ 0.60, final stage full);
  extrapolates = d7 ≥ 0.60 with CI lower bound > 0.55 at K′ = 12; far8–10 = mean balanced accuracy at distances 8–10,
  K′ = 12.
* **PA1 (primary).** Among competent runs, the extrapolating share is higher in M than in C. Supported if share(M) ≥ 0.70
  and share(M) − share(C) ≥ 0.25; not supported if share(M) ≤ share(C); otherwise inconclusive. (Basis: C-type models
  extrapolate in 21/42 of P30; blocking retrieval at inference rescued 17/23 retrieval-type failures.) Needs ≥ 4
  competent runs per arm, else unresolved.
* **PA2.** Mean far8–10 among competent runs is higher in M than in C by ≥ 0.05.
* **PA3 (learnability).** M reaches the gate in ≥ (share in C − 0.25). If not, the Markov step is "harder to train with
  this recipe" and PA1 is read on the competent runs only, with that caveat.
* **PA4 (mechanism).** Competent non-extrapolators in C have a retrieval-type pointer (retrieval share ≥ 0.5 on a trained
  step or at h_K) in ≥ 50 %; in M every failure is of the local kinds (fade / stall / restart) by construction, and the
  M failures are mostly fade (pointer dies within 4 hops past K).
* Secondary arms M2, MB, OV: descriptive only (4 seeds each).
* **Amendment A-1 (2026-09-29 ~05:50 EDT).** Arm MB had an implementation error: with `--budget_match` the curriculum
  gate still evaluated at K = 6 while training used K = stage depth + 1, so no MB run could pass stage 1 (all four
  launches of pa_mb_s731 and pa_mb_s733 stopped at step 4600 with identical logs; no MB model was evaluated). Fixed in
  `theory/train_arch.py` (the gate evaluates at the stage's budget); MB is re-run as `pa_mb2_s730..733` with the same
  seeds and rules. Interim Pilot A results for arms C, M, M2 had been seen when this was fixed; nothing else changes.


## 9. Pilot A interim reading and Pilot B (pre-registered 2026-09-29 ~06:15 EDT, before any Pilot B run started)

**Interim Pilot A (seen before this section was written; not the scored verdict).** Competent-and-extrapolating: C 3/6,
M 1/3 (s714 pending), M2 1/2. Pointer trajectories (`theory/analyse_pilotA.py`, `theory/results/pilotA_ptr.json`):
Markov failures cannot retrieve, yet they still break at K + 1 — s713 **restarts the sweep** every 4 steps
(1 2 3 | 1 2 3 4 | 1 2 3 4), s712 stalls and fades, and the non-gated M/M2 runs show 1 2 3 | 1 2 3 | 1 1 1 cycles.
**Reading:** the Markov restriction removes one *implementation* of a step-indexed schedule (retrieval), not the reason
for having one. The reason is **slack**: whenever the latent budget exceeds the hops the model must compute (and the
query adds 1–2 hops), any schedule that visits each depth once fits the loss — dwell by retrieval (Coconut), restart
cycles (Markov), clocks — and none continues past K. Developmental data agree (exp7, 24 models with checkpoints):
retrieval appears during **curriculum stage 3** (e.g. p30_s115 retrieval share 0.02 → 0.32 at step 8 000) and persists;
local extrapolators stay ≤ ~0.15 throughout.

**Pilot B tests the slack account directly with anytime (minimum-progress) supervision** (`--anytime 1`): after every
latent step k the delayed and upfront answers are supervised for reachable nodes within distance k + 1 and for all
unreachable nodes; reachable nodes farther than k + 1 are masked (reaching them early is never penalised); step K is the
ordinary loss. Dwell and restarts then cost loss; a time-homogeneous one-hop-per-step front satisfies every step without
a counter. (Relatives: SIM-CoT's per-thought supervision, TRM/DIS deep supervision; none is minimum-progress with
masked look-ahead, and none is motivated by a slack mechanism.)

* Recipe, evaluation and definitions exactly as Pilot A (§6). Arms, 4 seeds each, one of each arm per host:
  **AT** coconut + anytime 1 (seeds 750–753); **MAT** markov + anytime 1 (760–763); **MR** markov + gain-restoring
  pointer (`markov_restore`, A2; 770–773); **ATR** coconut + restore + anytime 1 (`restore --anytime 1`; 780–783).
* **PB1 (primary).** Among competent AT runs the extrapolating share is ≥ 0.75 and exceeds Pilot A's C share by ≥ 0.25.
  Not supported if it is ≤ C's share. Needs ≥ 3 competent AT runs, else unresolved. Pooled AT ∪ MAT reported too.
* **PB2 (mechanism).** In competent AT/MAT models the pointer does not dwell (depth increases at every step from step 2 to
  step K on the long-chain probe, allowing one repeat) and does not revert at K + 1 (0 reverts).
* **PB3 (learnability).** AT reaches the gate in ≥ 2 of 4 runs.
* **PB4.** Mean far8–10 of competent MR runs exceeds that of competent M runs (Pilot A) by ≥ 0.05. ATR descriptive.

## 10. When the circuit is chosen — developmental data (exp7, exploratory)

`theory/exp7_dev.py` → `theory/results/exp7_dev.json`: pointer and retrieval share at every saved checkpoint of 24 local
models (13 retrieval-type failures, 11 local extrapolators).

* **End of stage 2: every model has a static pointer** — it stays on depth 1 for all K steps (e.g. p30_s115, p30_s109,
  k8_s207, p30_s114 all read `1 1 1 1 1 1`); stage-2 graphs (distance ≤ 3) are solved by query-time hops.
* **Stage 3 is where the first latent advance is acquired, and the two circuits compete.** Failures acquire it as a
  lagged copy: p30_s115 at step 8 000 goes to `1 1 1 2 2 2 2 2` with retrieval 0.6–0.9 on steps 4–7 (0.0 at 7 500);
  p30_s109, p30_s113, k8_s207, k8_s221 the same within stage 3. Extrapolators acquire it locally: p30_s114 at 4 000
  `1 1 2 2 2 3 3 4` with retrieval 0.0; p30_s105 advances locally from stage 3–4 on.
* Whichever circuit wins in stage 3 persists to the end (retrieval shares stay 0.3–1.0 on the same steps through full
  depth); later stages only extend it. This places the commitment window (PROTOCOL6 E1: 2/4 agreement from stage-2
  entry, 5/6 from stage 4, 6/6 at full depth) in stage 3, when the first latent advance is learned.
* **Unselected check (all 34 locally checkpointed competent P30/L models, 12 extrapolate;
  `theory/results/exp7_stage_features.json`).** Mean retrieval share of the trained transitions at the *last checkpoint
  of stage 3* predicts final extrapolation: AUC 0.78 (0.79 at the first stage-4 checkpoint, 0.75 at full-depth arrival,
  0.67 at the end). With share ≥ 0.25 at the end of stage 3, **0 of 12** extrapolate; below, 12 of 22. A near-certain
  negative marker available before stage 4 (P7 at full-depth arrival, PROTOCOL7 T3, is the better positive marker:
  AUC 0.93, but later). Exploratory; a prospective test needs new checkpointed runs (Pilot A–C runs have checkpoints).
* **Markov failures are fade-triggered resets, not clocks.** In pa_m_s713 the pointer restarts at depth 1 exactly when
  its amplitude has decayed to ~2.4 (7.3 5.8 3.6 2.4 | 2.7 …): a weak pointer lets the step fall back to the start's
  edges. That is Mechanism 2 plus a default attractor — a time-homogeneous failure — so gain restoration (A2, arm MR)
  is the matching fix for Markov models, and anytime supervision (arms AT/MAT) the matching fix for the retrieval race.

## 11. Pilot C — slack dose-response (pre-registered 2026-09-29 ~06:35 EDT, before any Pilot C run started)

**Retrospective basis** (`theory/results/k_slack_rows.json`; all existing 2-layer d256 default-shape coconut runs with
evaluations; confounded by batch and time): competent runs that extrapolate — K = 6 without step embedding 49/105
(47 %), K = 6 with step embedding 20/33 (61 %), K = 8 without 5/19 (26 %), K = 10 1/3. The slack account predicts this
decline: more latent steps than hops → more room for step-indexed solutions.

* **Design.** Pilot A recipe and evaluation, only K changes: **K5** seeds 790–797, **K7** seeds 800–807 (8 each; two of each
  per host, same four hosts); the K = 6 point is Pilot A's coconut arm (seeds 700–707, same hosts and recipe).
* **PC1 (primary).** Extrapolating share among competent runs decreases with K: share(K5) > share(K6) > share(K7), and
  share(K5) − share(K7) ≥ 0.30. Not supported if share(K5) ≤ share(K7). Needs ≥ 3 competent runs per arm, else the
  arm is unresolved ("K too tight to learn" if K5 fails the gate).
* **PC2 (mechanism).** Mean retrieval share of the trained transitions (exp4 split) increases with K, and the number of
  steps the pointer spends before reaching depth 4 on the long-chain probe increases with K.

**§11 result (scored 2026-09-29 14:25 EDT; `theory/score_pilotC.py` → `theory/results/score_pilotC.json`).** All 16
Pilot C runs finished; three (pc_k5_s797, pc_k7_s800, pc_k7_s804) exhausted the stage-1 guard (4 attempts each) and
count as not competent (§15 rule).

| arm | competent / run | extrapolate (of competent) | retrieval share (mean, competent) | p[K−1] (competent) |
|---|---|---|---|---|
| K5 | 4 / 8 | 3 / 4 | 0.06 | 2 3 3 3 |
| K6 (Pilot A C) | 6 / 8 | 3 / 6 | 0.14 | 2 2 3 3 3 3 |
| K7 | 4 / 8 | 3 / 4 | 0.26 | 0 2 2 3 |

* **PC1: not supported** — share(K5) 0.75 = share(K7) 0.75 (rule: not supported if share(K5) ≤ share(K7)). With 4
  competent runs per arm the shares are imprecise, but there is no sign of the predicted decline.
* **PC2: half met** — the retrieval share of the trained transitions rises with K (0.06 → 0.14 → 0.26) as predicted; the
  steps to reach depth 4 do not (6.75, 9.17, 8.0).
* **Reading.** More slack buys more retrieval, but in these runs retrieval did not decide the outcome (pc_k7_s801 and
  pc_k7_s806 extrapolate with retrieval shares 0.51 and 0.43), consistent with the out-of-sample failure of the retrieval
  marker (§21). *Post hoc (not a test):* under Prop. 9 (§25) every K here has slack — the in-range reach is ≈ 3 at every K
  (p[K−1] medians 3, 3, 2), i.e. fractional speeds 3/5, 3/6, 3/7 that all need a brake — so the discriminating contrast
  is slack vs none (anytime, budget matching; Pilots D/E, F), not K5 vs K7. The slack account's dose-response form (PC1)
  is withdrawn.

## 12. Training-free repair at inference (exp8, exploratory)

`theory/exp8_gainboost.py` → `theory/results/exp8_gainboost.json`, the 59 P30 ∪ L models, long-chain probe, K′ = 12.
From step K + 1 on: layer-2 retrieval blocked (as exp5) **and** each input thought amplified around the per-step batch
mean, h ← c + α(h − c). Trained answers (K′ = K) are unchanged by construction.

| | no repair | block only (α = 1) | block + α 1.3 | block + α 1.6 | block + α 2.0 |
|---|---|---|---|---|---|
| models with 7-hop ≥ 0.60 | 24 | 41 | **46** | 47 | 44 |
| models with 8–10-hop ≥ 0.60 | 8 | 25 | **27** | 22 | 15 |
| unreachable-chain false-positive rate (8–10) | — | 0.14–0.19 | same | same | same |

* The boost helps extrapolators (8–10-hop 0.609 → 0.625 at α 1.3) and does nothing for local-type failures (0.531 →
  0.539): scaling the whole thought is not the restoration Mechanism 2 needs — the fade is inside the local map (the
  learned restoring module, Pilot B arm MR, is the real test).
* Practical reading: for an already-trained Coconut-style model, masking latent→latent attention past the trained budget
  plus a mild gain (α ≈ 1.3) roughly doubles the number of seeds that follow chains past the trained depth (24 → 46 of 59
  at 7 hops), with no retraining and no in-distribution cost.

## 13. Broadening: public GPT-2 latent reasoners on GSM8K

`theory/codi_probe.py` (self-contained inference: GPT-2 small with the checkpoint's LoRA merged, the latent
projection, CODI's GPT-2 test recipe; run on Vast host 4, `theory/results/codi_probe*_vast*.json` there). Models:
`zen-E/CODI-gpt2` and `internlm/SIM_COT-GPT2-CODI`, both trained with 6 latents on GSM8K-Aug. GSM8K test, 1 319 problems.

| latents at inference | 0 | 2 | 4 | **6 (trained)** | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| CODI GPT-2 | 0.221 | 0.234 | 0.368 | **0.425** (published 0.437) | 0.418 | 0.409 | 0.401 |
| SIM-CoT on CODI GPT-2 | 0.320 | 0.276 | 0.356 | **0.393** (published ≈ 0.44) | 0.391 | 0.394 | 0.397 |

* **The trained budget is a horizon.** Neither model gains from more latents, overall or on longer problems (4-step:
  CODI 0.31 → 0.24 at 12 latents; SIM-CoT 0.26 → 0.28); CODI degrades past the budget, SIM-CoT stays flat (its
  step-level supervision buys stability, as the SIM-CoT paper claims, but not extrapolation).
* **Latents read earlier latents, increasingly.** Attention mass of latent j on earlier latents (layer 0 / 5 / 11):
  SIM-CoT latent 2 0.05 / 0.07 / 0.08 → latent 6 0.32 / 0.19 / 0.26; CODI latent 2 0.06 / 0.02 / 0.01 → latent 6
  0.28 / 0.10 / 0.08. Making latents Markov within the budget (latents ≥ 2 cannot see earlier latents) costs 7.5 points
  (SIM-CoT 0.393 → 0.318) and 12.5 points (CODI 0.425 → 0.300): the history channel is used.
* **The toy's training-free repair does not transfer.** Masking earlier latents only past the trained budget (latents
  7–12) slightly lowers accuracy (SIM-CoT 10 latents 0.394 → 0.383, 12: 0.397 → 0.379; CODI 10: 0.409 → 0.393,
  12: 0.401 → 0.380). In the toy, the block exposed a hidden local "advance" operator; these GPT-2 reasoners have none
  to expose — consistent with the literature that their latents mostly relay a prefix computation (Aswal et al.;
  Illusion of Superposition; our ProsQA addenda). For LLM-scale latent reasoners the remedy has to be in training
  (anytime / unity-contract supervision, Pilot B/C), not at inference.
* Caveats: GSM8K step counts are not clean hop counts; problems with > 6 steps are few (n ≈ 20–40 per bucket).
* **Scope, with addendum Q (experiments session, 2026-09-29):** GPT-2 Coconut trained only on 3–4-hop ProsQA generalises
  to 5–6 hops with no loss and gains nothing from 12 thoughts. GPT-2-scale latent reasoners here compute in one pass;
  their "latent steps" do not implement a step schedule, so neither the schedule failure (Mechanism 1) nor the slack fix
  applies directly. The slack account is a theory of *multi-step latent computation*; making a pretrained model do such
  computation (problems deeper than one pass, T12-style) is the precondition for it to matter at scale.

**§13 addendum (2026-09-29 14:28 EDT; exploratory re-reading of the saved `by_steps` field of the same runs, no new
inference).** Accuracy by the number of gold CoT steps (s) and latents (L), GSM8K test (n = 1319; s = count of «…»
calculations in the gold CoT):

| CODI | s1 (65) | s2 (357) | s3 (364) | s4 (290) | s5 (138) | s6 (57) |
|---|---|---|---|---|---|---|
| L0 | 0.26 | 0.32 | 0.24 | 0.19 | 0.06 | 0.05 |
| L2 | 0.38 | 0.50 | **0.15** | **0.11** | 0.07 | 0.00 |
| L6 (trained) | 0.40 | 0.66 | 0.49 | 0.31 | 0.14 | 0.02 |
| L12 | 0.37 | 0.64 | 0.46 | **0.24** | 0.15 | 0.07 |

(SIM-CoT-CODI: same shape, smaller effects; s3 L0 0.32 → L2 0.27 → L6 0.46 → L12 0.47.)
* **Latents help most at 2–3 steps and not at all at ≥ 6** (L6 − L0: +0.14, +0.34, +0.25, +0.12, +0.08, −0.03): a
  reasoning-depth horizon inside the trained budget, and 12 latents do not move it (s4 falls 0.31 → 0.24 for CODI).
* **Truncated latent chains are worse than none** (CODI s3/s4: L2 0.15/0.11 < L0 0.24/0.19): the thought after 2 steps is
  not partial progress in a form the readout can use — the latent dynamics are keyed to the trained budget, not
  answer-ready at every step. This is the LLM-scale signature of a non-anytime contract (C1): an anytime-trained latent
  reasoner should have accuracy **non-decreasing in L** for every s. Proposed as an outcome for PROTOCOL8 (monotonicity
  in L per difficulty stratum), alongside test-time gains at L > K for problems above one-pass capacity (Prop. 8).

## 14. Mechanism 2 measured directly: transfer curves of the local map (exp9, exploratory)

`theory/exp9_transfer.py` → `theory/results/exp9_transfer.json`, `exp9_summary.json`. For each of the 59 models: roll out
to step K − 1, scale the input thought's deviation from the batch mean by s ∈ {0.25 … 2}, run step K with layer-2
retrieval blocked (the local map alone), and read the output pointer through the sweep head of the next step.

* **Two shapes.** *Restoring* (saturating): weak pointers collapse, strong ones are clamped to a fixed level — p30_s105
  inputs 1.4 → 6.4 give outputs 0.3, 1.4, 3.6, 3.8, 3.7, 3.6, 3.2 (a stable non-zero fixed point near 3.7).
  *Linear* (graded): output ∝ input with gain < 1 — p30_s126, gain ≈ 0.7 at every scale, no fixed point → the front fades.
  Some retrieval-type models have an almost absent local path (p30_s115, gain 0.15) or one that points back to depth 1
  (k8_s214).
* Classification (fixed before the tables below were computed, after inspecting three curves): *restoring* = the local
  map advances the pointer, its output at the operating point is ≥ 0.6 of the input, and the log-log slope around s = 1
  is < 0.6 (clamped).
* **Restoring maps are common in extrapolators** (19/25) and less so in failures (15/34).
* **The retrieval block rescues exactly the models whose hidden local map is restoring:** among retrieval-type failures,
  8–10-hop ≥ 0.60 after the block in 9/12 with a restoring local map vs 1/11 without.
* **Two-factor rule** (untrained last thought computed locally — retrieval share at h_K < 0.5 — *and* restoring local
  map): **18/23 extrapolate vs 7/36 otherwise**; with retrieval at h_K only 1/21 extrapolate whatever the map.

| local map | h_K computed | extrapolate |
|---|---|---|
| restoring | locally | **18 / 23** |
| restoring | by retrieval | 1 / 11 |
| not restoring | locally | 6 / 15 |
| not restoring | by retrieval | **0 / 10** |

This is the two-factor account in one table: Mechanism 1 (retrieval schedule at the untrained step) blocks
extrapolation almost completely; given a local step, Mechanism 2 (restoring vs graded map) decides most of the rest.
Formal anchor: saturated recurrences act as discrete automata (Merrill et al. 2020; Weiss et al. 2018). Exploratory;
the classification thresholds should be frozen and tested on new runs (Pilots A–C provide them).

**Out-of-sample check — the rule does not replicate (2026-09-29 ~07:40; `theory/exp10_rule_oos.py`,
`theory/results/exp10_rule_oos.json`).** The rule above, frozen before any Pilot model was analysed, applied to the 15
competent Pilot A runs (C, M, M2, MB2, OV): **8/15 correct** (predicted-extrapolate 11, of which 6 do; predicted-fail 4,
of which 2 do) — chance level. Errors: (i) *missed extrapolators* (pa_c_s704, pa_m2_s720): the transition probed at step
K happens to be a dwell step (depth 3 → 3), so a single-transition probe reads "not advancing" although the map advances
over two steps; (ii) *false alarms* (pa_c_s702, s706, s707, pa_m2_s721, pa_m_s713): step K looks local and restoring but
the sweep restarts or falls back later (restart cycles, retrieval on other steps; pa_c_s707's retrieval at h_K is 0.41,
just under the 0.5 cut, but 0.83 on a trained step). **Reading:** the in-sample 18/23 vs 7/36 split was partly fitted to
the 59 models; the qualitative mechanisms stand on the causal results (§2 block, §10 race), but a diagnostic read at one
step does not predict new seeds. P7 (reads the far chain directly) remains the reliable predictor; a mechanistic
predictor would have to integrate the pointer dynamics over several steps past K (dwell-aware), which is what P7 does
implicitly.

## 15. Pilot A — scored by the pre-registered rules (2026-09-29 ~09:15 EDT)

Sources: `theory/runs/pa_*/eval.json`, `theory/pilot_table.py`. Stage-1 failures after the relaunch budget count as
not competent (pa_m_s715, pa_mb2_s731). The buggy MB arm (pa_mb_*) is excluded (amendment A-1).

| arm | competent / run | extrapolate (of competent) | far8–10, competent |
|---|---|---|---|
| C coconut | 6 / 8 | 3 / 6 | 0.50 0.57 0.53 0.55 0.50 0.51 (mean 0.526) |
| M markov | 3 / 8 | 1 / 3 | 0.50 0.51 0.70 (mean 0.570) |
| M2 markov_l2 | 3 / 4 | 2 / 3 | 0.55 0.50 0.67 |
| MB2 markov + budget-matched | 2 / 4 | **2 / 2** | **0.59 0.77** |
| OV coconut + overrun 2 | 2 / 4 | 1 / 2 | 0.52 0.52 |

* **PA1 (primary): unresolved** — only 3 competent M runs (rule needs ≥ 4). Descriptively M 1/3 vs C 3/6: no sign of the
  predicted jump.
* **PA2: not supported** — mean far8–10 M 0.570 vs C 0.526 (difference 0.044 < 0.05).
* **PA3: fails** — M reaches the gate in 3/8 vs C 6/8 (bar: ≥ 0.50): the Markov step is harder to train with this recipe.
* **PA4:** C's competent non-extrapolators are retrieval-type in 2/3 (pa_c_s702, s707; s706 is a local restart cycle) —
  meets the ≥ 50 % bar. M failures: pa_m_s712 fades, pa_m_s713 restarts after fading — local failure modes as predicted.
* Secondary (descriptive, n tiny): the unity contract (MB2) gave the two strongest extrapolators of the pilot (8–10-hop
  0.59 and 0.77) but only 2 of 4 runs became competent; overrun (OV) did not help (1/2, far 0.52).
* **Reading:** removing the retrieval channel by architecture does not raise the extrapolating share and costs
  learnability; the evidence points from the *channel* to the *contract* (slack), which Pilots B (anytime) and C (K
  dose-response) test directly.

## 16. Pilot B interim — anytime supervision produces the homogeneous front (2026-09-29 ~09:40, not yet scored)

`theory/results/pb2_ptr.json`: pointer depth of h_0..h_11 on the long-chain probe (sweep head), competent runs.

| run | pointer h_0 … h_11 | non-advancing steps h_2..h_K | revert at K + 1 | extrapolates |
|---|---|---|---|---|
| pb_at_s753 (anytime) | 1 1 2 3 4 5 6 7 8 9 10 11 | 0 | no | yes (8–10: 0.67) |
| pb_mat_s762 (Markov + anytime) | 1 1 2 3 4 5 6 7 8 9 10 11 | 0 | no | yes (0.60) |
| pb_at_s751 (anytime) | 1 1 2 3 4 5 6 7 0 0 0 0 | 0 | no | yes at 7 hops (fades after) |
| pb_at_s752 (anytime) | 1 1 2 3 4 3 2 2 3 2 3 3 | 2 | no (falls back inside the budget) | no |
| controls pa_c_s702 … s707 | e.g. 1 1 1 2 2 3 3 3 …; 0 1 2 3 1 2 3 … | 1–3 each | 2 / 6 | 3 / 6 |
| pa_mb2_s733 (unity contract) | 1 1 1 2 3 4 5 6 7 8 9 10 | 1 | no | yes (0.77) |

* Anytime supervision (minimum progress, masked look-ahead) yields a one-hop-per-step front from step 1 with no dwell
  in 3 of 4 competent runs, and none reverts at K + 1 — the dynamics the slack account says a no-slack contract selects.
  The remaining failures are the other mechanism (s751 fades after distance 7) or an in-budget fall-back (s752).
* PB2 as pre-registered ("no dwell from step 2 to K, one repeat allowed; 0 reverts at K + 1"): no-dwell 3/4, reverts
  0/4 — final scoring after all Pilot B runs.

## 17. Pilot D — anytime supervision vs control, 12 + 12 seeds (pre-registered 2026-09-29 ~09:50 EDT, before any
## Pilot D run started; basis: Pilot A/B interim above)

* **Design.** Recipe and evaluation exactly as Pilots A–C. **AT** coconut + `--anytime 1`, seeds 900–911; **C**
  coconut control, seeds 920–931. Three of each arm per host on the same four Vast hosts, AT and C interleaved; queued
  after Pilot C. Definitions as §6 (gate; extrapolates = d7 ≥ 0.60 with CI lower bound > 0.55 at K′ = 12; far8–10).
* **PD1 (primary).** Among competent runs, extrapolating share(AT) − share(C) ≥ 0.20 and share(AT) ≥ 0.60 → supported;
  share(AT) ≤ share(C) → not supported; otherwise inconclusive. One-sided Fisher exact p reported. Needs ≥ 6 competent
  runs per arm, else unresolved.
* **PD2.** Mean far8–10 among competent runs: AT − C ≥ 0.05.
* **PD3 (mechanism).** Share of competent runs whose pointer advances at every step from h_2 to h_K on the long-chain
  probe (one repeat allowed): AT ≥ 0.60 and C ≤ 0.30.
* **PD4 (learnability).** Gate rate(AT) ≥ gate rate(C) − 0.25.
* Pilot A's C arm (seeds 700–707) and Pilot B's AT arm (750–753) are **not** pooled into Pilot D; they may be reported
  alongside, labelled as pilot data.

## 18. Formal sketch of the slack account (theory; to be made rigorous)

**Setting.** Delayed-query reachability; training graphs have depth ≤ D; the latent budget is K; after the latents, a
query can resolve up to q further hops from what the latent positions hold (here q ≈ 1–2: T1c, §2). A latent
*schedule* is the sequence of pointer depths π(1), …, π(K) (the depth each thought points at; sets in general). The
query reads every latent position, so the delayed loss is zero (in the idealised, noise-free model) iff the schedule
**covers** {1, …, D − q}: every such depth is pointed at by some step. Let the slack be σ = K − (D − q).

**Proposition 1 (slack ⇒ the continuation is not identified).** If σ > 0, the zero-loss schedules include ones with dwell
(π(k+1) = π(k)), restarts (π(k+1) < π(k)) and any continuation after coverage. For any prescribed behaviour after step
K — advance, stall, revert, restart — there is a zero-loss schedule that produces it, because steps after coverage are
unconstrained and the output of step K is never consumed. Hence the training loss does not determine whether the front
continues past K; the inductive bias does. *Sketch:* coverage needs D − q distinct depths in K > D − q steps; the σ free
steps can be spent anywhere, and h_K's successor is unconstrained.
*Realisation:* a schedule with dwell or restarts must know *when* to advance. With access to the history (Coconut) this
phase can be read from the number or content of earlier latent positions (retrieval with count-dependent lags, §2);
without it (Markov) it must live in the state (phase or amplitude — fade-triggered resets, §9). Both are step-indexed:
their continuation past K is whatever the phase variable does out of range.

**Proposition 2 (anytime supervision ⇒ minimum speed).** With anytime supervision (after step k, reachable nodes within
distance k + c are supervised; deeper ones masked), a zero-loss schedule must satisfy max_{j≤k} π(j) ≥ k + c − q for every
k until coverage: the front must advance at least one depth per step from the first step. Dwell and restarts before
coverage cost loss; a time-homogeneous advance operator (π(k+1) = π(k) + 1, no phase variable) satisfies every step, and
any step-indexed alternative needs extra machinery with no loss benefit. Under a simplicity bias this selects the
homogeneous operator — observed in Pilot B (§16: `1 1 2 3 4 5 6 7 8 9 10 11`, 0 dwell steps in 3/4 competent runs).
Steps after coverage (k > D − q) remain unconstrained, so Proposition 2 removes the *reason* for a schedule, not every
possibility of one.

**Proposition 3 (reach of a homogeneous advance depends on its gain map).** Let the pointer amplitude evolve as
a_{d+1} = φ(a_d) under the advance operator and let θ be the matching threshold (the competition of the other keys).
(i) If φ has a stable fixed point a* > θ whose basin contains a_{D−q} (a *restoring* map), the front continues
indefinitely (reach unbounded, up to noise). (ii) If φ(a) ≤ g·a with g < 1 (graded, *linear* map), reach ≤ (D − q) +
log(a_{D−q}/θ)/log(1/g). Neither the delayed nor the anytime loss constrains φ beyond the amplitudes realised at depths
≤ D, so neither forces (i) — consistent with pb_at_s751 (clean front to depth 7, then fades), with the transfer curves
(§14: restoring vs linear maps coexist among competent models) and with the toy simulation (§8).

**Corollary (what a complete fix needs).** Extrapolation requires removing slack (Prop. 2: anytime or unity
contracts) *and* making the advance restoring (Prop. 3(i): an architectural restoring nonlinearity, or training signal on
amplitudes beyond depth D — e.g. chains longer than the latent budget with anytime supervision, where the only way to
keep up is a sustained front). Pilots B (arms ATR, MR) and D test parts of this; the combined recipe is the next
experiment.

## 19. Pilot B — scored where complete (2026-09-29 ~11:45 EDT; host 1 still owes pb_mat_s760, pb_mr_s770, pb_atr_s780)

| arm | competent / run | extrapolate (of competent) | far8–10, competent |
|---|---|---|---|
| **AT** coconut + anytime 1 | **4 / 4** | **3 / 4** | 0.56 0.52 0.51 0.67 |
| MAT markov + anytime 1 | 1 / 3 | 1 / 1 | 0.60 |
| MR markov + restoring pointer | **0 / 3** | — | — |
| ATR coconut + restoring pointer + anytime | 1 / 3 | 1 / 1 | 0.53 |

* **PB1 (primary): supported (n = 4).** AT extrapolating share 0.75 ≥ 0.75 and ≥ Pilot A C's 0.50 + 0.25. Small n —
  Pilot D (12 + 12, same hosts) is the real test.
* **PB3 (learnability): met** — AT reaches the gate 4/4 (C in Pilot A: 6/8). Anytime supervision slows stage 2 in some
  seeds but does not prevent full depth within 30 000 steps.
* **PB2 (mechanism), competent AT/MAT so far:** no dwell (one repeat allowed) in 4/5 (s750 `1 2 3 3 4 5 6 5 …`, s751,
  s753, s762; not s752); revert at K + 1 in 1/5 (s750, 6 → 5, then fades) — the "0 reverts" half is **not met**.
* **PB4: unresolved** — no competent MR run. The restoring module as implemented **hurts learnability** badly (MR 0/3,
  ATR 1/3 vs AT 4/4): the per-id sigmoid write-back interferes with the curriculum (runs stall at stages 3–4). A
  restoring nonlinearity needs a gentler design (e.g. zero-initialised gate, applied only after full depth, or a
  soft clamp on the pointer norm) before it can be tested.
* Pilot C interim (not scored): K5 1/3 competent (does not extrapolate), K7 1/1 competent (extrapolates, 0.63) — early and
  opposite in direction to PC1 so far; K5 is hard to learn with this recipe.

**§19 final (2026-09-29 14:37 EDT; host 1's last three runs fetched).** pb_mat_s760 stopped at stage 4 (not competent);
pb_mr_s770 is competent and extrapolates (d7 0.772, far8–10 0.709); pb_atr_s780 is competent and extrapolates (d7 0.647,
far8–10 0.510).

| arm | competent / run | extrapolate (of competent) | far8–10, competent |
|---|---|---|---|
| AT | 4 / 4 | 3 / 4 | 0.56 0.52 0.51 0.67 |
| MAT | 1 / 4 | 1 / 1 | 0.60 |
| MR | 1 / 4 | 1 / 1 | 0.71 |
| ATR | 2 / 4 | 2 / 2 | 0.51 0.53 |

* PB1 **supported (n = 4)**, PB3 **met** — unchanged. Pooled AT ∪ MAT: 4 of 5 competent runs extrapolate.
* PB2 unchanged: no dwell in 4/5 competent AT/MAT runs, but a revert at K + 1 in 1/5 → the "0 reverts" half **not met**.
* PB4 **met on a single run** (MR 0.709 vs Pilot A M mean 0.570; rule ≥ +0.05) — with n = 1 it is not evidence; the
  restore module's main effect is on learnability (MR 1/4, ATR 2/4 vs AT 4/4), which motivated the zero-initialised gate
  in A6 (§24).

## 20. Pilot E — depth wall or schedule end? (pre-registered 2026-09-29 ~12:40 EDT, before any Pilot D or E run started;
## design agreed with the experiments session)

**Question.** Failing fronts stop at the training depth (PROTOCOL6 L2, M1). Is that a **depth wall** (DW: the frontier
cannot cross the trained depth whatever the schedule) or a **schedule end** (SE: the front stops because a
slack-induced step-indexed schedule ends; without slack it continues)? Anytime (minimum-progress) supervision removes
the slack-induced schedule (§16, §18 Prop. 2), so the two accounts make opposite predictions for anytime-trained models.

* **Amendment D-1 to Pilot D (§17), made before any D run started:** the control arm uses the **same seeds** as the anytime
  arm (C seeds 900–911 instead of 920–931), each pair on the same host. PD1–PD4 unchanged; a paired sign test is added.
* **Arms** (same recipe and evaluation as Pilots A–D; each pair = anytime run + control run, same seed, same host):
  **D** K = 6, training depth 6: seeds 900–911 (12 pairs, = Pilot D); **E8** K = 8, training depth 6
  (`--K 8`, anytime `--K 8 --anytime 1`): seeds 1000–1007 (8 pairs); **E3** K = 6, training capped at depth 3, i.e.
  distance ≤ 4 (`--depth_cap 3`, anytime `--depth_cap 3 --anytime 1`): seeds 1010–1017 (8 pairs).
* **Competence.** D, E8: the gate of §6. E3: final stage full and matched 4-hop bal-acc ≥ 0.60 (the R rule for cap 3,
  `scratch/score_r.py`).
* **Beyond-depth latent accuracy** (the query itself adds one hop — T1c — so the first latent-dependent distance is
  trained distance + 2): D, E8: **d8** = balanced accuracy at distance 8 at K′ = 12 (length_gen); E3: **d6** = balanced
  accuracy at distance 6 on the long-chain probe (chain A vs its look-alike D) at K′ = 12. A run **crosses** if this
  accuracy ≥ 0.60.
* **PE1 (primary; pooled D ∪ E8 ∪ E3; pairs with both runs competent).** SE supported if the anytime run's beyond-depth
  accuracy exceeds its control's in ≥ 70 % of evaluable pairs with one-sided sign-test p < 0.05, **and** ≥ 50 % of
  competent anytime runs in E8 ∪ E3 cross. DW supported if ≤ 20 % of competent anytime runs in E8 ∪ E3 cross. Otherwise
  inconclusive. Needs ≥ 8 evaluable pairs, else unresolved.
* **PE2 (slack).** The anytime − control difference in crossing share is larger in E8 (K = 8) than in D (K = 6).
  Descriptive (reported with both shares).
* **PE3 (pointer).** On the long-chain probe (theory/rollout.py, sweep head, K′ = 12), the pointer passes trained
  depth + 2 (D, E8: distance 8; E3: distance 6) in ≥ 60 % of competent anytime runs and ≤ 30 % of competent controls.
* **PE4 (causal).** Competent controls that do not cross: masking layer-2 latent→latent attention from step K on raises
  beyond-depth accuracy by ≥ 0.05 in ≥ 50 %. Competent anytime runs: mean absolute change ≤ 0.05 (they need no block).
* **Supplementary** (reported, not scored): P7/P8 per layer from the experiments session's frozen tools, copied
  unmodified to `theory/frozen/` (far_sweep_layers.py, far_sweep_frozen_I.py, frontier_causal2.py; natural-mass inject
  and block).
* **Order and hosts.** Pairs interleaved across the four theory hosts after Pilot C, E8 and E3 pairs placed early; host 1
  (slower) gets fewer pairs. Expected completion ~01:00 EDT 2026-09-30, cost ≈ $7 at $0.60/h.

* **Amendment E-1 (2026-09-29 ~13:00 EDT, before any Pilot D or E run started; prompted by the experiments session).**
  PE1's pair comparison uses the **continuous** beyond-depth latent accuracy (d8 for D/E8, d6 for E3): a pair counts for
  anytime if its anytime run's accuracy is strictly higher, for control if strictly lower; exactly equal pairs are ties and
  are excluded from both the 70 % share and the one-sided sign test. "Evaluable pairs" = pairs with both runs competent
  and not tied; PE1 needs ≥ 8 of them, else unresolved.

* **Amendment E-2 (2026-09-29 14:30 EDT; operationalisation only, written with the scoring code before any Pilot D run
  finished; of Pilot E, only pe8_at_s1006 / pe8_c_s1006's gate and d7 lines have been seen, a pair that is not
  evaluable because its control is not competent).** (i) PE3: "the pointer passes distance X" = the maximum of the
  ptr_traj pointer depths over thoughts h_0..h_11 (probe(512), K′ = 12) is ≥ X. (ii) PE4: the length_gen evaluator cannot
  apply the block, so base and blocked beyond-depth accuracy are both taken from `rollout.query_eval` on probe(512) at
  K′ = 12 — distance 8 = 0.5·(TPR_B + TNR_C) at far index 8 for D/E8, distance 6 = 0.5·(TPR_A + TNR_D) at near index 6
  for E3 (the same instrument as §20's E3 d6) — and the change is blocked − base. (iii) PD3's "advances at every step from
  h_2 to h_K, one repeat allowed" = p[i] = p[i−1] + 1 for every i in 2..K except at most one i with p[i] = p[i−1], and no
  i with p[i] < p[i−1]. Code: `theory/score_pilotDE.py`.

## 21. Out-of-sample test of the stage-3 retrieval marker on addendum O (pre-registered 2026-09-29 ~13:00 EDT)

Addendum O is scored by the experiments session (results/p_o.json); I have seen only its aggregate (O1 null, 13 vs 13 of
48 pairs), not any per-run label, and have not measured any O run.

* **Population.** The 27 locally trained O runs with step checkpoints (o_s100–o_s123, o_dev6_s51–s54 as present) that are
  competent by the experiments session's O scoring; labels (extrapolates or not) read from results/p_o.json **after**
  the measurement below is written to theory/results/.
* **Marker (frozen from §10, unchanged).** Mean retrieval share of the trained transitions (exp4 split, sweep head from
  exp7) at the **last checkpoint of curriculum stage 3** (`theory/exp7_dev.py` code path). Rule: share ≥ 0.25 → predicts
  *does not extrapolate*.
* **PO1.** Among competent O runs with share ≥ 0.25, the extrapolating share is ≤ 0.15 (in-sample: 0/12). Supported if
  so with ≥ 5 such runs; not supported if > 0.35; else inconclusive; unresolved with < 5 such runs.
* **PO2.** AUC of (−share) for extrapolation ≥ 0.70 (in-sample 0.78). Reported with the AUC of P7 on the same runs.
* **PO3 (added 2026-09-29 ~13:15, before any measurement; suggested by the experiments session).** Head-to-head with
  PROTOCOL7 T3's in-training diagnostic on the same competent O runs: early P7 = frozen `far_sweep_frozen_I.measure`
  (seed 2033; copied unmodified to theory/frozen/) at the checkpoint nearest full-depth arrival + 2000 steps, set to 0 if
  in_peak < 0.03 (T3 rule). Report AUC(−stage-3 share), AUC(early P7) and the model-bootstrap 95 % CI of their
  difference; descriptive (no decision rule). The stage-3 marker is the earlier signal (stage 3 precedes full depth).
* **Exploratory, labelled as such:** does the longer scaffold (O) change the retrieval share at the end of stage 3 relative
  to the paired P30 run of the same seed (same card)? — a mechanistic reading of O1's null.

**§21 result (scored 2026-09-29 ~13:50; `theory/score_o_marker.py` → `theory/results/score_o_marker.json`).** 17 competent
O runs with local checkpoints, 8 extrapolate.
* **PO1: unresolved** — only 2 runs have stage-3 retrieval share ≥ 0.25 (both fail; rule needs ≥ 5).
* **PO2: not met** — AUC(−share) **0.51** (in-sample 0.78).
* **PO3:** early P7 (T3 rule) AUC **0.92** vs stage-3 marker 0.51; difference −0.40, 95 % CI [−0.71, −0.11]. The experiments
  session's early P7 is the better in-training diagnostic; the stage-3 marker does not generalise.
* **Exploratory (scaffold):** mean stage-3 retrieval share 0.16 in O vs 0.22 in the same-seed P30 runs (lower in 13 of 23
  pairs) — no systematic change, consistent with O1's null. In O most failures have *low* retrieval: they fail by the
  local routes (fade, restart), so the retrieval schedule is one failure route whose prevalence depends on the recipe,
  not a general marker. The causal block result (§2) and the mechanism stand; the marker claims of §10 are downgraded
  to "in-sample only".

## 22. Second-generation theory: four conditions for extrapolation in the number of latent steps (2026-09-29, written 13:55–14:09 EDT; committed f1f21d8 14:09:05)

§18 explained our failures. This section generalises the account into conditions that any latent reasoner (Coconut
family, looped transformers, recursive models) has to meet, states them as propositions, and derives three new methods
(T6, T7, A6) and a scope result for pretrained models (Prop. 8). Prior art to position against (Literature session,
`docs/ADVISOR_2026-09-29.md`): 2609.33144 *Beyond the Training Horizon* (looped transformers: a decaying frontier and a
"fresh–aged" usability direction; progress tracked by a *content* cue, no count clock, no fix); D'Amour et al.
2011.03395 (underspecification — Prop. 1 is an instance); Zhang et al. 2607.20594 (budget law; T = n training); RELAY
2502.08482 (iteration-wise supervision); Kuo et al. 2606.29983 (random loop counts); Rodionov & Prokhorenkova 2402.11628
(discrete NAR) and Altabaa et al. 2510.14095 (anchored latents) — restoring states that train, both with state
supervision; Haviv et al. 2203.16634 (causal attention infers position from the number of predecessors — the implicit
clock of Mechanism 1).

### 22.1 The four conditions

Write a latent reasoner as h_{k+1} = F(h_k, H_{<k}, x), read out after a budget T. It extrapolates in the number of
steps (and hence in reasoning depth when depth needs steps) only if:

* **C1 Identification.** The training contracts pin the transition down on the states it visits: no slack during the
  stage in which the circuit is chosen (Props. 1, 2, 4).
* **C2 Homogeneity.** The transition has no usable clock — no step embedding, no count-indexed access to the history —
  or training makes the clock irrelevant (Mechanism 1; Haviv et al. for why causal attention always offers a clock).
* **C3 Stability.** On the coordinates that carry the computation, the per-step map has a stable non-zero fixed point
  (restoring), so errors and fading do not compound (Props. 3, 5; Mechanism 2; the "fresh–aged" direction of 2609.33144).
* **C4 Idempotence at completion.** Once the computation is finished, further steps leave the answer-relevant state
  invariant (overthinking; the untrained last thought of §3 is a violation; Prop. 6).

Where published methods stand (from their designs, not from new measurements; "?" = not determined by the design):

| method | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| Coconut, CODI | slack (fixed K ≥ steps needed) | no (history + position) | no | no (last thought dangling) |
| SIM-CoT, PMPS (per-thought decoders) | partly (every thought supervised) | no | ? (state supervised in-range only) | no |
| RELAY, CLRS hints, TRM deep supervision | yes (per-iteration targets) | yes for weight-shared loops | ? | TRM: partly |
| Kuo et al. random stopping, DR-Loop Poisson depth | yes **iff** the budget support reaches the steps needed (Prop. 4) | yes | ? | partly |
| attractor / DEQ / path-independent models | — | yes | yes (fixed point by construction) | yes |
| Discrete NAR (hard attention + discrete states) | via hints | yes | yes (discrete) | yes |
| ours: anytime supervision (T5) | yes (Prop. 2) | only if Markov/clock-free | no (Prop. 3) | no |
| ours: retrieval block + α boost (inference) | — | yes past K | partly (α) | no |
| **new: T6 amplitude jitter** | — | — | **yes (Prop. 5)** | — |
| **new: A6 monotone latent set** | — | yes | **yes** | **yes (Prop. 6)** |

Reading: no Coconut-family method addresses C3 or C4, and the ones that meet C1 do so only up to the training depth. The
combinations predicted to extrapolate are C1 + C3 (anytime + jitter; tested in Pilot F, §23) and architectures that
meet C3 + C4 by construction (A6).

### 22.2 Proposition 4 — tightness (resolves the random-budget tension)

*Setting as §18:* depth-n problems at curriculum stage s, query resolves q hops, so a latent schedule must cover depths
1..n − q. A training *contract* is a budget distribution μ_s(T | n); T is not observable by the latent steps (the query
comes after them). Call μ_s **tight** if its support contains T = n − q for every depth n trained at stage s.

**Prop. 4.** (a) If μ_s is not tight — every budget has slack T ≥ n − q + 1 — then the delayed schedule π(k) = max(0,
k − 1) (one dwell step, then unit speed) has zero loss under every budget in the support, so random budgets alone do not
exclude dwell/restart schedules and their step-indexed implementations (§18 Prop. 1). (b) If μ_s is tight, every
zero-loss schedule has π(k) ≥ k for all k ≤ n − q, i.e. unit speed from the first step: the zero-loss set equals the
anytime zero-loss set of Prop. 2 on 1..n − q. *Proof sketch:* (a) the delayed schedule covers 1..n − q by step n − q + 1
≤ T. (b) T is unobservable, so the schedule must succeed when T = n − q, which forces full speed. ∎

**Corollary.** Random budgets (Kuo et al., DR-Loop), budget matching (T2, Zhang et al., MR-Loop) and anytime supervision
(T5, RELAY) are one mechanism: each puts mass on tight contracts. They help iff that mass is present **at the stage where
the circuit is chosen** (for us, curriculum stage 3, §10), because Prop. 1's freedom is exercised there.
*Retrospective consistency (exploratory, n tiny, not a test):* P22 randk (K ∈ 4..8, step embeddings on) had slack 2–3 at
stage 3 (depth 3, n − q = 1–2) → 0/2 competent extrapolate; OV (budgets 6–8) never tight → 1/2; MB2 (tight at every
stage) → 2/2 with the pilot's two best far8–10; anytime (tight at every step) → PB1 3/4. *Prediction for others:* random
loop counts help in looped models only when their minimum budget is ≤ the steps needed at the selection stage; a
distribution with min(T) above that should not help. *New test (T7 below and §23 arm ATW/ATL):* tightness only inside
the selection window should give most of the benefit; tightness only after it should give little (a critical period for
algorithm selection; Achille et al. 1711.08856).

### 22.3 Proposition 5 — contraction training forces a restoring map (method T6: amplitude jitter)

Let a ≥ 0 be the amplitude of the computation-carrying component (pointer logit excess, §4), b the amplitude of leakage
on a non-reachable decoy, and φ the per-step map; readout: reachable iff a > θ.

**Prop. 5.** Train with *amplitude jitter*: the thought fed into each step is rescaled by s ∈ [1/r, r] (r > 1),
adversarially or at random with full support, and the answer must stay correct at every trained depth ≤ D. (a) No
linear map φ(a) = g·a (the graded regime of §14) can satisfy this beyond depth n_lin = ln(a₀/b₀) / (2 ln r), where
a₀/b₀ is the initial signal-to-leak ratio: correctness under all-down jitter needs (g/r)^n a₀ > θ, under all-up jitter on
a decoy (g·r)^n b₀ < θ, and both hold only while r^{2n} < a₀/b₀. (b) Any map that satisfies it for D > n_lin must be
expanding below some threshold and contracting around a stable fixed point a* > θ whose basin contains [a*/r, r·a*] —
a bistable (restoring) map. (c) A restoring map with such a basin keeps the front alive for any number of steps under
bounded per-step noise η < (a* − θ)(1 − L), where L < 1 is its Lipschitz constant on the basin (invariant interval), so
reach is unbounded (Prop. 3(i)). *Proof sketch:* (a) as stated; (b) the image of the jitter interval must stay above θ
and below the decoy bound after every step, so φ maps [a*/r, r·a*] into a subinterval: by the intermediate value theorem
there is a fixed point there, and the average slope on the interval is < 1; since φ(0) ≈ 0 (no pointer stays no pointer)
and φ(a*/r) > a*/r, φ exceeds the diagonal between 0 and a*/r only after an expanding stretch (the threshold); (c)
standard invariant-set argument. ∎

**Why it is new.** A2 (restore module, Pilot B) tried to impose (b) architecturally and mostly broke learnability (Pilot B arm
MR: 1 of 4 runs competent); Rodionov and Altabaa impose discreteness with state supervision. T6 imposes C3 through the
*loss* without state targets or new parameters: the transfer-curve perturbation of exp9 (our diagnostic) becomes the
augmentation. It is also the principled version of the training-free α ≈ 1.3 boost (§12), and a trained counterpart of
2609.33144's "fresh–aged" restoration. Cost: one elementwise rescale per latent step during training. Applies unchanged
to CODI/SIM-CoT/Coconut on LLMs (jitter the fed-back hidden state about its batch mean).

*Minimal model (`theory/sim_jitter.py`, not evidence):* a 200-step smoke run of a 1-D learned map without an amplitude
ceiling became explosive (gain 2.1 at a = 1). A 1-D model has no fixed-norm thought (post-LN) and therefore no reason for
the sub-unity regime seen in trained transformers, so it cannot test Prop. 5; the test is the transformer pilot of §23.

### 22.4 Proposition 6 — a monotone latent set is idempotent and cannot fade (architecture A6)

Many reasoning problems are least fixed points of monotone operators: reachability and ProsQA entailment (R ← R ∪
succ(R)), Horn-clause/Datalog inference, dataflow analysis, constraint propagation. Let the latent state carry a
membership vector m ∈ [0, 1]^V over entities, updated by **m_{k+1} = max(m_k, σ(β(u(m_k, x) − τ)))** with u monotone
in m (e.g. non-competitive sigmoid attention from the set to edge/rule tokens, A3), and let the next thought receive
m_{k+1}·E (the entity embeddings) additively.

**Prop. 6.** (a) m_k is non-decreasing in k: a found entity never fades (C3 on membership coordinates, without any gain
condition). (b) If u is exact on sets, m_k is the k-step closure and m_{k+1} = m_k after convergence: any number of
extra steps leaves m invariant (C4; no overthinking, no dangling last state). (c) Errors are one-sided and absorbing:
with per-step false-positive probability ≤ ε per entity, P(some false positive by step k) ≤ k·V·ε — linear in k,
against the geometric decay (1 − h)^k of 2609.33144's hazard model or g^k of a graded pointer. *Proof:* (a), (b) from max
and monotonicity (Knaster–Tarski on the finite lattice [0,1]^V with inflationary updates); (c) union bound. ∎

**Design A6 ("latent set memory") for LLM latent reasoning.** A persistent set vector over the entity (or vocabulary)
space next to the continuous thought: each latent step writes sigmoid scores (non-competitive, so many members do not
dilute each other — Th4), the set updates by max, and both the next thought and the answer read it. Relatives:
max-aggregation in neural execution of graph algorithms (Veličković et al. 1910.10593) and NBFNet; algorithmic
alignment (Xu et al. 1905.13211); the KV cache is already a monotone store for the *readout* — A6 makes the *transition*
read the accumulated set rather than a fading pointer. Limits: only inflationary reasoning; retraction or negation needs
stratified sets (as in stratified Datalog); false positives are permanent, so the threshold must be calibrated (T6
jitter on the set scores is the natural training signal). Toy test deferred until Pilot F reports (GPU budget).

### 22.5 Proposition 7 — method T7: tightness only where it matters

From Prop. 4's corollary and §10 (the circuit is chosen at stage 3): **anytime supervision applied only during the
selection window** (stages ≤ 3) should give most of the benefit of full-length anytime supervision, and anytime applied
only after it (stages ≥ 4) little. This is cheaper than full anytime training (its intermediate readouts can be
dropped once the circuit is set) and it is a falsifiable critical-period claim about algorithm selection in latent
reasoners. Tested by §23 arms ATW/ATL.

### 22.6 Proposition 8 — slack explains bypass: scope for pretrained models

Let c be the number of hops a model resolves in one forward pass on the task (one-pass capacity; it grows with depth —
Merrill & Sabharwal 2503.03961, Ye et al. 2510.19753 — and our addendum Q shows c ≥ 6 for GPT-2 on ProsQA-style graphs).
For an n-hop problem and K latent steps each worth c_s hops, the latent slack is σ = K − ⌈max(0, n − c)/c_s⌉. If n ≤ c,
σ = K: every latent step is unidentified (Prop. 1 with no constraint at all), and bypassing the latents is a zero-loss
solution that is shorter than any use of them. **Prediction P8a:** the causal effect of the latent tokens on the answer
is ≈ 0 for n ≤ c and grows once n > c — a single account of the reports that latent tokens are bypassed or shortcut
(Aswal et al. 2606.12689; Cui et al. 2602.22441; Rizvi-Martel et al. 2604.06374; our addenda D/G/J/N/Q). **P8b:** test-time
gains from more latents than trained (K′ > K) appear only when n > c *and* the contract was tight (C1). Design
consequence (PROTOCOL8): measure c first (one-pass accuracy vs n on the in-context format; addendum P's curve), then
train at depths above c. At n ≤ c no latent method can show extrapolation in steps, so ProsQA at its native depth is the
wrong benchmark for it.

### 22.7 Addendum (2026-09-29, committed 2e9fa17 14:15:32 EDT): literature check of §22 (Literature session; `docs/ADVISOR_2026-09-29.md` addendum)

* **Prop. 4 needs a second condition.** Kuo et al. 2606.29983 train with K = clip(n + Δ, 1, 20), Δ ~ U{−5..5} (the support
  includes tight and under-budget K), and it helps; but their **length-matched K = n was the worst** out of distribution
  (22.1 % vs 34.2 % for fixed K = 20 and 45.0 % for RL halting). A deterministic tight budget that is a function of
  something the model can observe (length) makes T observable, so the premise of Prop. 4(b) fails, and it makes the
  loop count a usable clock (C2). **Prop. 4, refined:** random or matched budgets help iff (a) the support reaches the
  tight budget at the selection stage **and** (b) T is not a deterministic function of an observable of the example.
  Our MB2 (K = stage depth + 1, constant within a stage while example depths vary) satisfies (b); a per-example
  T = depth + 1 would not, and is predicted to do worse than MB2 or anytime. DR-Loop (R ~ clip(Poisson(4), 2, 8),
  independent of n) and Huginn (r mostly below the mean) satisfy both, which is consistent.
* **Refinement of (b) (21:51 EDT, after the Literature session's full reads, `docs/LIT_REQUESTS_2026-09-29.md`).** Fan et al.
  2409.15647 use a deterministic, length-dependent budget T(n) and still extrapolate, because the *test* budget also
  follows T(n). Kuo et al.'s schedule is n-centred, K = clip(n + Δ, 1, 20) with Δ ~ U{−5..5}, which includes budgets
  below the steps needed; their worst arm, K = n, was tested at n = 20–60 against a training clip at 20, so part of that
  failure may be the clipped test budget. So (b) is not a general law. The correct form: if the budget is a function of
  an observable, the schedule may key on it, and the continuation is identified only where the test budget follows the
  same function. For extrapolation to *arbitrary* test budgets (our K′ > K), the budget must be independent of the
  example. "Random loop counts help" should be cited as "n-centred budgets that sometimes undershoot help", which is
  the tightness claim (a). No published scheme manipulates slack directly (Bansal 2202.05826, Huginn and DR-Loop all
  use fixed minimums of 1–2), so F1a vs P22 is the discriminating test.
* **Novelty, stated honestly.** T6 (jitter) — *medium-low*: training a recurrence to denoise its own state is known
  (Mozer et al. 1805.08394 State-Denoised RNNs, the closest in purpose; Lim et al. 2102.04877 Noisy RNNs, with theory;
  Lamb et al. 1905.11382 state reification; Huginn's random initial state). New here: multiplicative amplitude jitter of
  a Coconut-style fed-back thought with no auxiliary loss or target, and Prop. 5's statement that it excludes graded
  maps. A6 — the principle (max aggregation for reachability) is standard (Veličković et al. 1910.10593; Bonet
  2609.15932, set-aggregation recurrent GNNs ≡ reachability logic); new as a latent-CoT state (no Coconut-style model
  with a max-updated set thought found; Gu 2609.13747 cumulative superposition is the continuous counterpart). T7 —
  *low-medium*: Ali 2605.04396 shows weight decay in one 25 % window matches full weight decay OOD; ours is supervision,
  not regularisation, in a mechanistically located window. Prop. 8 — *medium*: bypass is well documented (Rizvi-Martel,
  Cui 2602.22441, Aswal; Think-at-Hard 2511.08577: most tokens right after one pass, extra iterations overthink; Pfau
  2404.15758: filler tokens help only when more computation is needed); a formal slack/capacity statement with a
  falsifiable depth threshold was not found.
* **Independent support for C2 (history hurts extrapolation):** Kavuncu 2609.19521 (cellular automata, train to depth 6,
  test to 12): a current-state-only recurrent transformer reaches 98.1 % at depth 12, CoTFormer with an expanding cache
  of earlier states 50.3 %; clearing the cache restores extrapolation. There the cache *interferes* rather than being
  retrieved from — a different route to the same C2 violation as our retrieval schedules.

## 23. Pilot F — amplitude jitter (T6), the complete fix (C1 + C3) and the selection window (T7) (pre-registered 2026-09-29, committed f1f21d8 14:09:05 EDT, before any Pilot F run started and before any Pilot D run finished)

**Question.** Does contraction training (§22.3, Prop. 5) make the latent map restoring and extend reach, alone and on top
of anytime supervision; and does tightness matter only in the selection window (§22.5)?

* **Arms** (P30 recipe of §6 via `theory/queue_arch.sh`; seeds 900–907, 8 runs per arm; paired by seed with Pilot D):
  **J** `--jitter 2` (s ~ LogUniform[1/2, 2], p = 0.5 per thought, training only); **JAT** `--anytime 1 --jitter 2`;
  **ATW** `--anytime 1 --anytime_stages 1:3` (anytime only in curriculum stages 1–3, the selection window);
  **ATL** `--anytime 1 --anytime_stages 4:5` (anytime only from stage 4 to full depth). Comparison runs: Pilot D's
  **C** (pd_c_s900–907) and **AT** (pd_at_s900–907), same seeds and code path. Cards may differ within a pair (recorded).
* **Code.** `theory/arch.py:add_jitter` and `theory/train_arch.py --jitter/--jitter_p/--anytime_stages` (this commit).
  Unit checks (2026-09-29 ~14:00): jitter active only in training and the state dict unchanged; windowed anytime with
  the window closed equals the plain loss exactly; with the window open equals unwindowed anytime exactly; CUDA-graph
  replay draws fresh jitter each step (checked on host 4). Options off ⇒ the Pilot D/E computation is unchanged.
* **Competence and outcomes** as §6/§20: gate of §6; beyond-depth latent accuracy **d8** = balanced accuracy at distance
  8 at K′ = 12 (length_gen in eval.json); extrapolates = extrap_K12; far8–10 reported.
* **Restoring index** (C3). exp9's transfer curve (`theory/exp9_transfer.py:measure`, unchanged; sweep head from
  `theory/exp2_membership.py`, unchanged; new driver only): **z = log₂(a_out(2)/a_out(1))** — 1 for a linear map, ≤ 0 for a
  clamped (restoring) one. Computed for J, JAT, C, AT (competent runs).
* **Pair rule** (all PF comparisons): a pair counts if both runs are competent and their values differ; supported if the
  predicted direction holds in ≥ 75 % of counted pairs; not supported if ≤ 50 %; otherwise inconclusive; unresolved with
  fewer than 5 counted pairs. One-sided sign-test p reported (8 pairs is a pilot; no claim rests on p alone).
* **PF1 (primary, T6 → C3).** z(J) < z(C).
* **PF2 (primary, T6 → reach).** d8(J) > d8(C).
* **PF3 (complete fix, C1 + C3).** d8(JAT) > d8(AT); descriptive bar: ≥ 75 % of competent JAT runs extrapolate.
* **PF4 (selection window, T7).** d8(ATW) > d8(ATL); descriptive: median d8(ATW) within 0.05 of median d8(AT). If Pilot D
  shows no anytime advantage (PD1 not supported), PF4 is reported as moot.
* **PF5 (learnability guard).** Competent counts: J vs C and JAT vs AT differ by ≤ 2 of 8 (the restore module's failure
  mode, §19, must not recur).
* **Order and hosts.** Appended to the theory hosts' queues after Pilot D/E: J and JAT for all seeds first, then ATW and
  ATL; hosts 2–4 (host 1 is slow and is released after its D/E pairs). 32 runs, ≈ 24 GPU-hours, ≈ $4; expected
  completion ~08:00–10:00 EDT 2026-09-30.
* **Amendment F-1 (2026-09-29 15:28 EDT, before any Pilot F run started; budget).** Arms **ATW and ATL are withdrawn**
  (16 runs) to keep the shared Vast credit within the user's "don't overspend" instruction and make room for Pilot J.
  PF4 (the selection-window test, T7) is therefore not run; Pilot H's fine-tuning arms (§27) address the related
  critical-period question on trained models. PF1–PF3 and PF5 are unchanged.
* **Amendment F-2 (2026-09-29 15:38 EDT, before any Pilot F run started).** The user added Vast credit ($145.83 at 15:37),
  so **ATW and ATL are restored** exactly as registered, and PF4 is back in force (still moot if PD1 is not supported).
  Pilot F now runs on additional hosts in parallel (placement below does not change any prediction).

## 24. Pilot G — monotone latent set (A6) (pre-registered 2026-09-29, committed 2e9fa17 14:15:32 EDT, before any Pilot G run started)

**Question.** Does a monotone, max-updated set memory (§22.4, Prop. 6; `theory/arch.py:SetMemReasoner`, `--arch setmem`)
make the latent computation extrapolate, alone and with anytime supervision (C1 + C3 + C4)?

* **Architecture.** After each latent step the thought is scored against the id table, m_k = max(m_{k−1},
  σ(β(r(h_k)·E_idᵀ − τ))), and γ·W(m_k·E_id) is added to the next latent input; γ is initialised to 0 (so at
  initialisation the model equals Coconut exactly — checked, max difference 0.0), β = 8, τ = 0.5 learned. Unit checks
  (~14:13): m non-decreasing over 12 steps; anytime path runs; CUDA-graph training runs and γ moves (host 4 smoke run).
* **Arms** (P30 recipe; seeds 900–907; paired by seed with Pilot D): **G** `--arch setmem`; **GAT** `--arch setmem
  --anytime 1`. Comparison: Pilot D's C and AT. 16 runs.
* **Outcomes, pair rule and competence** exactly as §23 (d8 at K′ = 12; ≥ 75 % of counted pairs; ≥ 5 counted pairs).
* **Manipulation check (decides interpretability).** Final |γ|, and d8 with the set ablated at evaluation (γ → 0). The set
  counts as *used* in a run if ablation lowers d8 by ≥ 0.05 or lowers delayed in-distribution bal-acc by ≥ 0.05. If it is
  used in fewer than half of the competent G ∪ GAT runs, PG1–PG3 are reported as "set unused" (no verdict on A6).
* **PG1.** d8(G) > d8(C). **PG2.** d8(GAT) > d8(AT). **PG3 (C4, no overthinking).** Far-chain bal-acc at K′ = 16 minus at
  K′ = 12 (length_gen, distances 8–10 pooled) is ≥ −0.01 in ≥ 75 % of competent G ∪ GAT runs and higher (paired) than in
  C/AT. **PG4 (learnability).** Competent counts G vs C and GAT vs AT differ by ≤ 2 of 8.
* **Amendment PG5 (2026-09-29 14:33 EDT, before any Pilot G run or feasibility run finished) — a monotone set removes
  the need for a brake.** Under slack, Prop. 9 says a pointer-state front must brake with a clock; with a monotone set,
  running at full speed and then idling at the fixed point needs no clock (the set simply stops growing). Prediction: in
  competent G runs whose set is used (manipulation check), the set depth at step K on the long-chain probe
  (`theory/set_traj.py`: deepest d with m_K(chain node) − m_K(look-alike) ≥ 0.5 at every d' ≤ d) is ≥ K − 2 in ≥ 75 % of
  runs (≥ 4 such runs, else unresolved) — against p[K−1] ≈ 3 for Coconut pointers (§25).
* **Amendment G-1 (2026-09-29 15:46 EDT, before any Pilot G run started; after the two feasibility runs).** fz_g_s950
  (setmem) ended at curriculum stage 4 (not competent; delayed 0.875, m5 0.592) and fz_gat_s951 (setmem + anytime) was
  still in stage 3 at step 27 200 — neither reached full depth, against 71 % for the P30 recipe (41 of 58 P30 runs reach
  full depth, median step 18 000; n = 2, so P ≈ 0.08 if learnability were unchanged — weak evidence; numbers corrected
  from "~75 %, P ≈ 0.06" at 15:55 after counting). The set gate moved off 0 (≈ −0.08), so the set is used and apparently interferes with
  the curriculum, as the restore module did. G and GAT stay as registered (PG4 measures learnability). Added arms, on
  the Literature session's suggestion to introduce restoration after the curriculum: **GL** `--arch setmem
  --set_from_stage 5` (the set write is off — the model is exactly Coconut — until full depth, then on) and **GLAT** (the
  same + `--anytime 1`), seeds 900–907. PG1–PG5 apply to GL/GLAT with G → GL and GAT → GLAT; PG4's guard for GL/GLAT is the
  same (competent counts within 2 of 8 of C/AT).
* **Order and hosts.** Host 1 (after its D/E pairs, ~02:00): GAT then G for seeds 900–901; hosts 2–4 after Pilot F:
  the remaining 12, GAT first. Expected completion ~12:00–14:00 EDT 2026-09-30; ≈ $3.

## 25. Proposition 9 — slack creates the clock (least-effort brake); pre-registered pointer predictions for Pilots C, D, E (2026-09-29, committed 7b6875c 14:18:52 EDT, before any Pilot C/D/E pointer trajectory was computed)

**Observation (retrospective, 59 P30/L models, `theory/results/pointer_taxonomy.json`).** The pointer depth of the last
fed-back thought, p[K−1] (long-chain probe, top sweep head, `analyse_pointer.py`), is the same for K = 6 and K = 8:
median 3 in both (K6: 29/42 models at 3; K8: 10/17). The in-range front therefore moves at speed ≈ 3/6 and 3/8, by
dwelling (mean in-range dwell steps 2.6 at K6, 3.5 at K8) — equally in extrapolators and failures. The two differ only
**after** K: extrapolators switch to unit speed (p30_s155 `1 1 1 2 2 3 | 4 5 6 7 8 9`), failures cycle (`2 3 2 3`) or
restart. This is Zhang et al.'s budget law (v ≈ n/T, "the demanded minimum is met, never exceeded") measured inside the
latent thought.

**Proposition 9 (fractional speed needs a clock).** Suppose training at budget K demands latent depth d* at the deepest
problems and SGD meets the demand exactly (least effort: the front reaches d* at step K). If d*/K is not an integer, the
front must advance on some steps and dwell on others. A transition whose state is only the pointer (plus the fixed
problem) maps equal inputs to equal outputs, so it cannot dwell at step j and advance at step j′ from the same pointer:
it needs a **phase variable** — a clock — read from the step embedding, the number of earlier thoughts (Haviv et al.),
or a phase coordinate carried in the state. Hence slack (C1 violated) *forces* a clock (C2 violated): the "brake" that
slows a speed-1 operator to d*/K is step-indexed, and past K it either disengages (extrapolation) or keeps cycling
(failure). Tight contracts demand speed 1, which a clock-free operator realises. *Consequences:* (i) this is why removing
the history channel (Pilot A, Markov) did not help: the clock moved into the state (pa_m_s713 restarts); (ii) C1 and C2
are not independent — enforcing C1 removes the need for C2 violations; (iii) the budget law's fractional speeds are the
signature of a clock, so Zhang et al.'s v = n/T regime predicts step-indexed mechanisms whenever T > n.

**Pre-registered predictions** (measurement frozen: `theory/analyse_pilotA.py:ptr_traj`, unchanged, on `probe(512)` from
`theory/rollout.py`; p[K−1] = pointer depth of the last fed-back thought h_{K−1}; competent runs only; a new driver
calls it for pc_*, pd_*, pe8_*, pe3_* runs):
* **LE1 (budget law in the thought).** Control (non-anytime) arms at different K reach the same depth by K: the arm
  medians of p[K−1] for Pilot C K5 and K7, Pilot D control (K6) and E8 control (K8) span at most 1 (max − min ≤ 1).
  *Not supported* if the K8 or K7 median exceeds the K5 median by ≥ 2 (a homogeneous speed would give p[K−1] ≈ K − 1).
* **LE2 (anytime removes the brake).** Anytime arms (Pilot D AT, E8 AT): median p[K−1] ≥ K − 2, and p[K−1] higher than
  the same-seed control's in ≥ 75 % of pairs with both competent (≥ 5 such pairs, else unresolved).
* **LE3 (the demand sets the depth).** E3 controls (training capped at depth 3): median p[K−1] ≤ 2, below Pilot D
  control's median.
* Reported with each: in-range dwell counts, and post-K speed (p[K+3] − p[K−1]) split by extrapolation.

* **Amendment LE-R (2026-09-29 14:23 EDT, suggested by the experiments session; before any addendum-R outcome was read
  and before any R model was measured).** Out-of-sample test of Prop. 9 on the experiments session's addendum R (K = 6,
  `--depth_cap 3`: r3_s400–411; `--depth_cap 4`: m_s317–328; different recipe owner, local cards). Same frozen
  measurement (ptr_traj on probe(512), model loaded from `runs/<run>/model.pt` only); competence and labels taken from
  the experiments session's R scoring **after** it is posted. Predictions: **LE3-R** median p[K−1] ≤ 2 in competent
  cap-3 runs and ≤ 3 in competent cap-4 runs; **LE3-R dose** median(cap 3) < median(cap 4) ≤ median(uncapped P30,
  in-sample = 3). Unresolved for an arm with < 4 competent runs. The already-scored m_s301–316 (cap 4) are in-sample
  context only.

* **Amendment LE4 (2026-09-29 14:25 EDT, before any Pilot D run finished) — attenuation as a clock.** In a Markov or
  weakly history-dependent transition, a decaying pointer amplitude is itself a monotone counter, so under slack a
  sub-unity gain can serve as the brake's clock (the fade-triggered restarts of pa_m_s713 are this). Prediction: without
  slack there is no use for the clock, so anytime-trained models have more restoring local maps than their controls —
  **LE4:** transfer-curve saturation index z = log₂(a_out(2)/a_out(1)) (exp9, as §23 PF1) is lower in the anytime run
  than in its same-seed control in ≥ 75 % of Pilot D pairs with both runs competent (≥ 5 such pairs, else unresolved;
  ≤ 50 % = not supported).

**Exploratory addendum to §25 (2026-09-29 14:22 EDT; retrospective, P30/L data only, computed after LE1–LE3 were
committed; no Pilot C/D/E data used).**
* *Demand met stage by stage* (`theory/results/exp7_dev.json`, 34 checkpointed models, last checkpoint of each curriculum
  stage, pointer counted only if its logit ≥ 1): median p[K−1] = 0 at the end of stage 2, **2** at stage 3, **3** at stage
  4, 3.5 at full depth — one more depth per stage, as the demand grows by one latent hop per stage. The front's in-range
  reach tracks the curriculum's demand rather than the budget.
* *The brake is (partly) retrieval* (`exp4_shares.json` × `pointer_taxonomy.json`, in-range steps): retrieval share on
  dwell steps 0.22 in extrapolators vs 0.43 in failures; on advance steps 0.04 vs 0.14. Dwelling is where the history is
  copied; failures brake by copying twice as much, and past K their copies continue periodically (e.g. `2 3 2 3`, with
  retrieval share ≈ 1.0 on every dwell). Dwell *positions* do not separate the groups (dwell at h_1–h_5 in 86/57/52/43/19 %
  of extrapolators vs 86/67/24/52/29 % of failures, K6). Reading: the brake exists in both groups (Prop. 9); what differs
  is how it is built — a copy keyed to the history count (fails past K) or a local mechanism that releases past K.
* *A textbook clock (14:35; `theory/le_pointer.py` on the budget-matched Markov runs, in-sample):* pa_mb2_s730 (Markov, no
  history channel) runs `1 1 2 2 3 3 | 4 4 5 5 6 6` — a period-2 phase bit carried in the state, realising speed ½, and
  it keeps running past K at the same speed (extrapolates, far8–10 0.59). pa_mb2_s733 runs `1 1 1 2 3 4 | 5 6 7 8 9 10`
  (a short warm-up, then unit speed; far8–10 0.77). A clock need not stop at K; what fails is a clock whose continuation
  past K stops or cycles (the count-indexed retrieval of §2), as Prop. 1 allows.

## 26. Continuous Thought Machines as a source for C2 — re-anchored episodes and stationary history (2026-09-29 14:48 EDT, prompted by the user; re-anchoring predictions registered before the 59-model run)

**What CTMs do** (Darlow et al., arXiv 2505.05522, read in full today): internal ticks decoupled from the input; each neuron
has private weights applied to a sliding window of its last M pre-activations (neuron-level models); the representation
is *synchronisation* — S_ij^t = Σ_τ e^(−r_ij(t−τ)) z_i^τ z_j^τ / √(Σ_τ e^(−r_ij(t−τ))), learnable r_ij ≥ 0 — which forms
the attention queries and the outputs; no positional embeddings; the loss averages the loss at the lowest-loss tick
and at the most-certain tick; on mazes, a model trained to output 100 steps on 39×39 mazes solves longer paths and
99×99 mazes **by re-applying its policy** from the current position.

**What carries over to C2 (no usable clock) — and what does not.**
1. *Re-application from the current state* (the maze result) is the cleanest route: a bounded-horizon latent loop that is
   restarted from a decoded state never sees a step count outside its trained range, and each restart begins from a
   crisp token embedding (amplitude restored, C3). → **Method T8, re-anchored latent episodes** (below; training-free
   test now).
2. *Stationary history.* A sliding window of M steps, or history weighted by e^(−r(t−τ)) with r > 0, is time-translation
   invariant: after ~M or ~1/r steps it carries no information about the absolute tick, unlike causal attention over the
   whole latent history, whose normalisation counts predecessors (Haviv et al.). Note the CTM normaliser √Σe^(−r(t−τ)) is
   a √t clock when r = 0, which CTMs allow; a clock-free variant needs r ≥ r_min > 0. → **Architecture A8, stationary
   history reads**: latent→latent access only through a window of M earlier thoughts or a decayed summary with r ≥ r_min,
   so the history channel (which Pilot A showed helps learnability) stays but cannot index steps beyond M ≈ 1/r_min.
   *Prediction from Prop. 9:* under slack A8 alone will not fix extrapolation — the clock will be rebuilt in the state as in
   Pilot A — so A8 must be paired with a tight contract (C1). Its value is C2 without Markov's learnability cost.
3. *The tick-selection loss* (min-loss tick + max-certainty tick) lets the model answer when ready. By Prop. 4 it does not
   remove slack (being right at *some* tick leaves the speed free); combined with minimum-progress masking (T5) it
   would, and max-certainty halting is a content-triggered stop (C4, A5) rather than a count.
4. *Neuron-level temporal filters* are a per-coordinate nonlinearity over time — a possible substrate for restoring
   (bistable) dynamics (C3); not pursued yet.

**Method T8 — re-anchored latent episodes (training-free).** Run the trained K-step loop; decode where its frontier is
from the model itself (the target of the edge token that the sweep head attends to most at step K); start a new K-step
episode from that node on the same input; combine query logits by max over episodes. Variant re2top3 branches the
second episode from the 3 most-attended edges' targets. Code `theory/exp12_reanchor.py`; exploratory debug on one model
outside the test set (pa_c_s706: distance 7–10 bal-acc 0.50 → 0.65/0.65/0.61/0.54 with re2top3) was run before this entry.

**Predictions for the 59 P30 ∪ L models** (long-chain probe(512); "7-hop success" = distance-7 bal-acc ≥ 0.60 as in exp5:
single episode at K′ = 12 → 24/59, layer-2 block → 41/59; 16 failures are not rescued by the block):
* **RA1.** re2top3 gives 7-hop success in ≥ 41 of 59 (at least the block's count).
* **RA2 (differential).** Among the 16 block-resistant failures, re2top3 reaches 7-hop success in ≥ 40 % (block: 0 %) —
  re-anchoring resets both the clock (Mechanism 1) and the amplitude (Mechanism 2); the block only removes the first.
* **RA3.** Mean far8–10 bal-acc over the 59 models: re3 − single ≥ 0.05.
* **RA4 (cost).** Mean near-distance (1–6) bal-acc drops by ≤ 0.02 (max-combination can add false positives).

**§26 result (scored 2026-09-29 15:03 EDT; `theory/score_reanchor.py` → `theory/results/score_reanchor.json`).**
*Definition (added 20:34 EDT at the Audit session's request):* "7-hop success" in this section is distance-7 balanced
accuracy ≥ 0.60 on the long-chain probe (as in exp5), **without** the confidence-bound clause (lower bound > 0.55) of
the confirmatory "extrapolates" criterion, so it is looser than "extrapolates" in Pilots A–J.

| inference | 7-hop success (of 59) | failures rescued (of 34) | mean far8–10 | mean near1–6 |
|---|---|---|---|---|
| single episode, K′ = 12 | 24 | 1 | 0.533 | 0.861 |
| layer-2 block from K (exp5) | 41 | — | — | — |
| **re2** (2 episodes) | **55** | 31 | 0.612 | 0.864 |
| re3 (3 episodes) | 53 | 31 | 0.624 | 0.851 |
| **re2top3** (branching 2nd episode) | **55** | 31 | **0.667** | 0.854 |

* **RA1 met** (55 ≥ 41). **RA2 met:** 14 of the 16 block-resistant failures are rescued (87.5 %; block 0 %) — including
  all 8 local-type P30 failures (s102, s109, s115, s119, s126, s135, s145, s158) and 6 of the K = 8 failures.
  **RA3 met** (+0.091). **RA4 met** (near −0.007 / −0.010).
* **Reading.** Failing models are competent *within* K steps from any start node deep in the graph; what breaks is running
  beyond K. Restarting the loop keeps every step count inside the trained range and restarts the pointer from a crisp
  node embedding, and that alone turns 31 of 34 failures into 7-hop successes [corrected 20:34 EDT from "33 of 34", a
  transcription error found by the Audit session; the table and JSON always said 31 — 24 + 31 = 55]. This is the strongest evidence yet for the
  clock account (Props. 1, 9): the local operator generalises to new positions; the step-indexed continuation does not.
  It also shows the fading mechanism is bounded by the episode length, since the local-type failures are rescued too.
* **Caveats.** Inference-time only, with extra compute (2–4 episodes) and a discrete decode (the anchor), i.e. a hybrid
  latent/explicit procedure — the latent counterpart of CTM's maze re-application and of the Markovian Thinker's chunked
  reasoning (2510.06557). The sweep head used to decode the anchor was identified on in-distribution depths. The
  delayed query still reads each episode separately (max-combined); balanced accuracy at 8–10 hops stays ≈ 0.62–0.67,
  far from perfect — anchors land off the chain in ~50–70 % of graphs (other branches of the frontier), which the
  branching variant partly fixes.
* **Controls (requested by the experiments session; registered 2026-09-29 15:17 EDT, before the 59-model control run;
  one debug run on pa_c_s706, outside the set, was seen first: model 0.60/0.61/0.58/0.53, random node 0.49/0.48/0.49/0.51,
  random same-depth node 0.60/0.61/0.58/0.53, oracle chain node 0.87/0.85/0.79/0.61 at distances 7–10).**
  `theory/exp12b_controls.py`, two episodes, episode 2 anchored at: the model's decode; a random node of the graph; a
  random reachable node at the model anchor's BFS depth; the chain node at that depth (oracle). The probe's frontier
  legitimately holds W nodes per depth (W = 2, 4, 6), so the model's anchor lies on the probed chain in only ~1/W of
  graphs; what the decode can be credited with is the *depth*.
  **RC1 (not "any fresh episode"):** random-node anchors give 7-hop success ≤ 29/59 and mean far8–10 within ±0.02 of the
  single episode. **RC2:** random same-depth anchors match the model's anchors (7-hop success within ±5) — the gain is
  carried by restarting at the frontier's depth. **RC3:** oracle anchors beat the model's by ≥ 0.10 in mean far8–10 — the
  choice among frontier nodes is the remaining bottleneck (branching over the whole frontier would close it).
  Extrapolating models are reported separately (descriptive; our account predicts gains there too at 8–10 hops,
  because they also fade — Mechanism 2 — not "no gain").
* **Controls result (scored 2026-09-29 15:26 EDT; `theory/results/score_reanchor_controls.json`).**

  | second-episode anchor | 7-hop success (of 59) | failures (of 34) | mean far8–10 | mean near1–6 |
  |---|---|---|---|---|
  | none (single episode, K′ = 12) | 24 | 1 | 0.533 | — |
  | model's decode | 55 | 31 | 0.612 | 0.864 |
  | random node | **0** | 0 | 0.503 | 0.826 |
  | random reachable node, same depth | 56 | 32 | 0.619 | 0.868 |
  | oracle: probed chain's node at that depth | **59** | **34** | **0.885** | 0.937 |

  **RC1 not met as written** — the band was ±0.02 around the single episode, and random-node anchors fall *below* it
  (far8–10 −0.030, 7-hop 0/59, near −0.035): a random fresh episode hurts rather than helps, so the conclusion RC1 was
  meant to protect ("the gain is not from any fresh episode") holds more strongly than predicted. **RC2 met** (56 vs 55):
  the gain comes from restarting at the frontier's depth; the model's choice among the W frontier nodes is at chance
  (its anchor lies on the probed chain in 30 % of graphs ≈ mean 1/W). **RC3 met** (+0.273): restarted at the right
  frontier node, *failing* models reach 0.872 at 8–10 hops and all 34 pass 7 hops — each K-step episode is fully
  competent from any start; only the continuation past K was broken. Extrapolators gain too (single 0.573 → oracle 0.903
  at 8–10 hops), as expected from their fading (Mechanism 2).
* *Exploratory (after the controls; 15:44; `theory/exp12c_branch.py`):* branching the second episode over the top-M attended
  edges — M = 1: 55/59, far8–10 0.612; **M = 3: 55/59, 0.667**; M = 6: 46/59, 0.620 (near 0.806): beyond the frontier's
  width the extra anchors are off-frontier nodes that add false positives. A calibrated frontier decode (threshold, or a
  trained set readout as in A6) is the way to approach the oracle's 0.885.
* **Implication for the field (C2).** Latent reasoners do not need to extrapolate in the number of steps if they are
  re-anchored: train a bounded-horizon latent loop, decode its state to a discrete checkpoint every K steps, and
  restart. Training-time counterpart: continue training from the model's own step-K states with the counter reset
  (sample-pool training, Growing Neural Cellular Automata, suggested by the Literature session) — proposed as Pilot J.

## 27. Pilot H — repairing trained failures by fine-tuning: anytime (C1) vs jitter + anytime (C1 + C3) vs plain (pre-registered 2026-09-29 14:49 EDT, before any Pilot H run started)

**Question.** Is the failure of a trained model to extrapolate fixed by removing slack *after* training (the slack/clock
account, Props. 1, 2, 9), or is the circuit chosen at stage 3 locked in (a critical period)? And does the answer differ
by failure mechanism, as Props. 3 and 5 predict?

* **Models.** The 20 competent non-extrapolating P30 models with K = 6 (s102, s109, s111, s112, s113, s115, s118, s119,
  s125, s126, s130, s131, s132, s135, s140, s145, s146, s147, s157, s158). By exp5, 12 are *retrieval-type* (the layer-2
  block rescues 7-hop success: s111, s112, s113, s118, s125, s130, s131, s132, s140, s146, s147, s157) and 8 are
  *local-type* (not rescued: s102, s109, s115, s119, s126, s135, s145, s158).
* **Fine-tuning** (`theory/train_arch.py --init_from runs/p30_sX/model.pt`, same seed, fresh AdamW): 6000 steps, lr 1e-4
  (warm-up 100, cosine), no aux loss, bs 512, bf16, gated curriculum (the trained model advances every 500 steps, so
  ~4000 steps at full depth). Arms: **FT-C** plain loss (all 20); **FT-AT** `--anytime 1` (all 20); **FT-JAT**
  `--anytime 1 --jitter 2` (the 8 local-type models).
* **Outcomes** from `theory/eval_arch.py` (gate; d7 at K′ = 12; extrapolates = extrap_K12; far8–10) and the pointer
  (le_pointer.py: p[K−1]).
* **PH1 (primary: slack removal repairs).** FT-AT extrapolates in ≥ 40 % of its competent runs and FT-C in ≤ 15 %; and
  d7(FT-AT) > d7(FT-C) in ≥ 75 % of models with both competent. *Critical-period reading* if FT-AT ≤ FT-C + 0.10 in
  extrapolating share.
* **PH2 (the brake goes).** Median p[K−1] ≥ K − 2 = 4 for competent FT-AT runs; ≤ 3 for FT-C runs.
* **PH3 (mechanism split).** FT-AT's extrapolating share among retrieval-type models exceeds that among local-type
  models by ≥ 0.25 (anytime removes the schedule; a fading map also needs C3).
* **PH4 (C3 adds to C1).** Among local-type models, d7(FT-JAT) > d7(FT-AT) in ≥ 6 of 8 (both competent; ties excluded;
  unresolved if < 5 counted).
* **Guard.** Each arm keeps the gate in ≥ 75 % of its runs; otherwise that arm's verdicts are flagged.
* **Order and hosts.** 48 runs inserted after Pilot D/E and before Pilot F on all four theory hosts (~15 min per run on
  hosts 2–4); Pilots F and G shift by ~3 h (to ~15:00–17:00 EDT 2026-09-30).

## 28. Pilot I — a clock that never runs out (pre-registered 2026-09-29 15:05 EDT, before any Pilot I run started; idea #2 of the Literature session's adjacent-field list, Clockwork RNN 1402.3511 / Phased LSTM 1610.09513)

**Logic (Prop. 9).** Under slack the model wants a clock to brake with; what fails is a clock whose continuation past K is
unidentified (a count that leaves its trained range). A periodic clock — step embedding ((k − 1) mod p) + 1 — has an
identified continuation: past K it repeats exactly the cycles seen in training, so a brake keyed to it keeps braking at
the same rate and the front keeps moving at its in-range fractional speed instead of stopping or cycling (in-sample
existence proof: pa_mb2_s730's period-2 phase bit, `1 1 2 2 3 3 | 4 4 5 5 6 6`). Counterpoint: Xu & Sato 2410.01405 argue
timestep encodings raise the expressivity of looped transformers; this pilot is about extrapolation, not expressivity.

* **Arms** (P30 recipe; seeds 900–907; paired with Pilot D's control pd_c_s*, which has no step embedding): **PER2**
  `--periodic_step 2`, **PER3** `--periodic_step 3` (`theory/arch.py:add_periodic_step`; evaluation and the rollout
  harness follow the period — harness vs model max |diff| 0.0 at K′ = 6 and 12, checked 2026-09-29).
* **PI1 (mechanism).** Among competent runs, the pointer (le_pointer.py, h_0..h_11) keeps advancing after K — max over
  h_K..h_11 ≥ p[K−1] + 2 and no backstep after K — in ≥ 60 % of competent PER2 ∪ PER3 runs, and that share exceeds the
  share among the same seeds' competent controls by ≥ 0.30.
* **PI2 (outcome).** Pair rule of §23 on far8–10 at K′ = 16 (length_gen, mean of distances 8–10): PER > C in ≥ 75 % of
  counted pairs (pooled PER2 and PER3 pairs; ≥ 5 counted).
* **PI3 (the speed is kept).** In competent PER runs, the post-K speed (p[11] − p[K−1]) / (11 − (K − 1)) is within ±0.25 of
  the in-range speed p[K−1] / K in ≥ 60 % of runs.
* **Order.** 16 runs appended after Pilot G on the four theory hosts (~2 h at the end of the queue).
* **Amendment I-1 (2026-09-29 15:28 EDT, before any Pilot I run started; budget).** Arm **PER3 is withdrawn**; Pilot I is
  PER2 only (8 runs). PI1–PI3 are evaluated on PER2.
* **Amendment I-2 (2026-09-29 15:38 EDT, before any Pilot I run started).** Credit added: **PER3 is restored**; PI1–PI3 are
  evaluated on PER2 ∪ PER3 as first registered.

## 29. Pilot J — chunk-reset latent attention (A8; CTM sliding window) (pre-registered 2026-09-29 15:19 EDT, before any Pilot J run started)

**Design (`--arch chunk3`, `theory/arch.py:ChunkReasoner`).** The latent budget is cut into chunks of L = 3 steps; latent
step k attends to the prefix and only to the earlier latent positions of its own chunk; at each chunk boundary the latent
history is cut and the within-chunk count restarts. The thought carries state across chunks; the queries read every
latent position. This is §26's A8 made concrete: a finite, periodic history window (a CTM-style sliding window, and the
attention-level form of Pilot I's periodic clock). It is also the training-time counterpart of re-anchoring (§26):
every chunk at test time is the same computation as a trained chunk, started from the previous chunk's final thought,
so continuing past K means running more copies of a trained computation rather than extrapolating a count. (Checked
2026-09-29: harness = model at K′ = 6 and 12; identical to Coconut up to step 3.)

* **Arms** (P30 recipe; seeds 900–907; paired with Pilot D): **CH3** `--arch chunk3`; **CH3AT** `--arch chunk3 --anytime 1`.
* **PJ1 (continuation).** The pointer passes distance 8 by h_11 (le_pointer.py; max p ≥ 8) in ≥ 50 % of competent CH3 ∪ CH3AT
  runs and in a larger share than the same seeds' competent C ∪ AT runs.
* **PJ2 (outcome).** §23 pair rule on d8 (K′ = 12): CH3 > C and CH3AT > AT (pooled; ≥ 5 counted pairs).
* **PJ3 (learnability).** Competent counts CH3 vs C and CH3AT vs AT differ by ≤ 2 of 8 (unlike the full Markov step,
  Pilot A PA3).
* **Where.** Not queued on Vast yet (credit): offered to the experiments session's local GPUs after addendum R, or appended
  after Pilot I if the user approves the extra ≈ $2.
* **Placement (2026-09-29 15:28 EDT).** The user left arms and placement to the agents' discretion within budget (relayed by
  the Audit session); agreed with the experiments session: Pilot J runs on the theory Vast hosts, capped at $5. Queue
  order on hosts 2–4: D/E → H → F (J, JAT) → G → I (PER2) → J; host 1 keeps only its D/E and H runs and is destroyed after
  them (its G/I/J lines moved to hosts 2–4). Projected theory Vast spend from 15:30 to the end of all queues ≈ $15–16.

## 30. Zero-cost follow-ups from PROPOSALS §9: renormalisation (F3a), the randk re-analysis (F1a's $0 test), and branch selection (pre-registered 2026-09-29 21:48 EDT, before any of these measurements on the named models)

* **Dropped before running: F6a (learned frontier decoder) and F6b (soft restart).** On one debug model outside the test set
  (pa_c_s706), a linear decoder h_K → node ids put the probed chain's node at the anchor depth first in 32 % of graphs,
  with a mean rank of 22 — the same as the attention anchor (≈ 1/W). The sweep head's attention at step K has ≥ 50 % of
  its maximum on only 1.29 edges on average (W = 4.1), and 98 % of them lie at the anchor's depth. *In this one debug
  model and this one head*, the pointer seems to follow about one branch of the frontier. That conflicts with Part IV
  P25 (14/18 models: layer-1 frontier attention splits per frontier edge and halves when the frontier doubles), and the
  Audit session saw p30_s124 alternate between branches across steps in head-mean attention. Possible reconciliations:
  different heads cover different branches, branch switching across steps, or width dependence. Dropping F6a/F6b *as
  designed* stands either way; the mechanism claim waits for B1.
* **F3a (amplitude renormalisation at inference; `theory/exp13_decode_restart.py`, renorm conditions only).** Past K, each
  fed-back thought's deviation from the batch mean is rescaled to its example's mean in-range norm (h_1..h_{K−1}), with
  and without the layer-2 block. Baselines from exp8 (block + constant α): α = 1.0 → 41/59, far8–10 0.585; α = 1.3 → 46/59,
  0.594. **RN1:** renorm + block gives 7-hop success ≥ 46/59. **RN2:** renorm + block mean far8–10 > 0.594. (7-hop success =
  distance-7 bal-acc ≥ 0.60 on the probe, as §26.)
* **R-randk (Prop. 4 on the existing P22 runs, measurement only; competent runs randk_s0 and randk_s2).** With budgets
  K ∈ 4..8 and T unobservable, zero loss requires the demanded depth by step 4 (the minimum budget), and nothing
  after it. Prop. 4 predicts a faster in-range front than fixed K = 6: **RR1** pointer depth of the last fed-back thought at
  budget 4, p[3], ≥ 3 in both runs (fixed-K6 P30 models have p[3] ≈ 2). **RR2** (least effort after the tight point) p[7]
  ≤ p[3] + 1 in both runs — randk had step embeddings, so a clock was available to brake after step 4.
* **B1 (branch selection; descriptive).** Per model: the mean number of edges with sweep-head attention ≥ 0.5 × max at step
  K, and the share of those at the anchor's BFS depth.
* **Amendment B1-a (21:48 EDT, before B1 was measured; at the Audit session's request).** B1 is reported per head (all 8
  layer-1 heads) and for the head-mean, split by width W = 2 / 4 / 6, at every step 1..K, as (i) the number of frontier
  edges (edges out of nodes at the pointer's BFS depth) with attention ≥ 0.5 × the maximum over frontier edges, and (ii)
  the effective number exp(entropy) of the attention restricted to frontier edges. It is compared explicitly with P25's
  per-edge split: P25 predicts an effective number ≈ W at every step; single-branch following predicts ≈ 1.

## 7. Log

* 2026-09-29 01:10–01:45 — exp1 (window), exp2 (pointer), exp3/4 (component attribution), exp5 (retrieval block) on the
  59 P30 ∪ L competent models; this document.
* 2026-09-29 ~01:45–02:05 — attenuation analysis (exp6); literature pass; trainer/evaluator for variants; Pilot A
  pre-registered.
* 2026-09-29 13:55–14:09 — §22 (four conditions C1–C4; Props. 4–8; methods T6 jitter, T7 selection-window
  anytime; architecture A6 monotone latent set) and §23 (Pilot F pre-registration); jitter/window options added to
  theory/train_arch.py and uploaded to the hosts (options off ⇒ unchanged computation).
* 2026-09-29 ~14:11–14:16 — theory/anytime_loss.py for PROTOCOL8; §22.7 (literature check: Prop. 4 refined with
  condition (b), novelty levels); A6 implemented (`--arch setmem`); §24 Pilot G pre-registered.
* 2026-09-29 ~14:15–14:19 — §25 Prop. 9 (fractional budget-law speed needs a clock; slack forces C2 violations) and
  pointer predictions LE1–LE3 for Pilots C/D/E, registered before their pointer trajectories were computed.
* 2026-09-29 14:25 — Pilot C scored (§11 result): PC1 not supported, PC2 half met; LE-R amendment (§25).

## 8. Toy simulation of the loop-gain account (Th3) — `theory/sim_gain.py` (exploratory)

A pointer amplitude moves one hop per step along a chain; every node also receives fresh noise each step, which the same
map propagates (decoy leakage). Readout: amplitude > θ. Parameters fitted (grid search, balanced error) on depths ≤ D
only; reach = deepest depth with balanced accuracy ≥ 0.9 on 40-hop chains.

| per-hop noise sd | analog a ← g·a + noise: reach at D = 3 / 5 / 8 (fitted g) | restoring a ← σ(β(g·a − τ)) + noise: reach |
|---|---|---|
| 0.08 | 10 / 16 / 21 (g 0.96–1.10) | 6 at D = 3; **40** for D ≥ 4 |
| 0.15 | **7 / 8 / 10** (g 1.10–1.16) | **40 / 40** (D = 3, 5) |
| 0.25 | 3 / 3 / 3 (noise too high to fit) | 3 / 4 |

Reading: with moderate noise, a graded update fitted on depth ≤ D generalises only a few hops past D (reach ≈ D + 2–4,
growing with D) — a horizon tied to the training depth that a larger budget cannot remove; a restoring (bistable)
update has no horizon once it is fitted at all. With very little noise the graded update also reaches far, so the
*strength* of the horizon depends on the noise level. In the real models the horizon at exactly the training depth is
mainly Mechanism 1 (the schedule ends when the trained steps end); Mechanism 2 limits how far a local front gets
beyond it (THEORY §4), as this toy predicts. Supports proposal A2 (gain restoration).
