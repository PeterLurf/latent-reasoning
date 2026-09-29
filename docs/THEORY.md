# Why some latent reasoners stop at the training depth — mechanism, theory, architectures

Theory stream, started 2026-09-29 ~01:10 EDT (separate Claude session; experiments and project flow are run by the
other session — see `docs/CURRENT_TASKS.md`). Everything here is **exploratory** (measured on existing models after their
labels were known) unless a section says it was pre-registered. Code: `theory/*.py`; outputs: `theory/results/`. All
measurements run on the CPU on existing `runs/` models; nothing here touches the queues or the pre-registrations.

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
