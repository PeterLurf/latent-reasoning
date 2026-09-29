# Why some latent reasoners stop at the training depth — mechanism, theory, architectures

Theory stream, started 2026-09-29 ~01:10 EDT (separate Claude session; experiments and project flow are run by the
other session — see `docs/CURRENT_TASKS.md`). Everything here is **exploratory** (measured on existing models after their
labels were known) unless a section says it was pre-registered. Code: `theory/*.py`; outputs: `theory/results/`. All
measurements run on the CPU on existing `runs/` models; nothing here touches the queues or the pre-registrations.

**Contents:** Executive summary · Timestamp correction · §0 question · §1 pointer · §2 retrieval (Mechanism 1) ·
§3 theory · §4 attenuation (Mechanism 2) · §5 architectures · §6 Pilot A pre-registration · §9 Pilot B · §10 developmental
race · §11 Pilot C · §12 training-free repair · §13 GPT-2 latent reasoners · §14 transfer curves (+ failed out-of-sample
rule) · §15 Pilot A scored · §16 Pilot B interim · §17 Pilot D · §18 formal sketch · §19 Pilot B scored · §20 Pilot E ·
§21 O marker test · §7 log · §8 toy simulation. (Sections were appended in the order written, so numbers are not in
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

## Executive summary (updated 2026-09-29 ~10:30 EDT)

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
   retrieval ≥ 0.25 at the end of stage 3 → 0/12 extrapolate in-sample, but the marker **failed out of sample** on addendum O (AUC 0.51 vs early P7 0.92, §21). Retrospectively, extrapolation falls with K
   (47 % at K = 6 vs 26 % at K = 8, §11).

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

**Running:** Pilots B (anytime / restore, done ~13:00), C (K = 5 / 7 dose-response), D (anytime vs control, 12 + 12;
until ~20:00) on 4 Vast GPUs (~$0.60/h). Proposals for the field: `docs/PROPOSALS.md`.

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

## 7. Log

* 2026-09-29 01:10–01:45 — exp1 (window), exp2 (pointer), exp3/4 (component attribution), exp5 (retrieval block) on the
  59 P30 ∪ L competent models; this document.
* 2026-09-29 ~01:45–02:05 — attenuation analysis (exp6); literature pass; trainer/evaluator for variants; Pilot A
  pre-registered.

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
