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
§21 O marker test · **§22 four conditions C1–C4, Props. 4–8, new methods T6/T7 and architecture A6** · §23 Pilot F · §24 Pilot G (A6) · **§25 Prop. 9: slack creates the clock** · §26 CTM-inspired T8/A8 · §27 Pilot H (fine-tune repair) · §28 Pilot I (periodic clock) · §29 Pilot J (chunk-reset attention) · §30 zero-cost follow-ups · §31 Pilot K (label-free tight budgets) · §32 Pilot L (re-anchoring distillation) · **§33 north star: checkpointed latent reasoner + C7 tests** · §7 log · §8 toy simulation. (Sections were appended in the order written, so numbers are not in
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

**Frontier (§52, 2026-09-30):** five untested directions:
* F1: a discrete per-token annotation state, count-free and readable by construction;
* F2: bidirectional meet-in-the-middle latent search;
* F3: compositional self-synthesis of deeper training data;
* F4: a behavioural round-trip translator for silent thoughts;
* F5: verify-then-extend search.

Novelty is with Literature, and Experiments runs the tests.

**External input (§52 addendum, 2026-09-30 22:22):** registered two cheap tests on existing models: §55 (is the fall-back
reward-neutral?) and §56 (are aged latents interchangeable?). Added F6 (age-transplant training) and F7 (path doubling).
§51 halt line closed: TH1 0/10, TH2 1/10.
**F1 on GPT-2:** PROTOCOL12 not learnable as specified (hard-label ST gives no gradient); one revision
registered as §54b (relaxed labels in training, hardened at test, hardening gap AG0).
**ProsQA cue, confirmed:** without the cue, every GPT-2 latent arm is at chance on deep ProsQA (Coconut 0.508), while text CoT holds 0.767.
**ProsQA cue (§57):** the generator always hangs the negative option off another person. A person-agnostic proximity
heuristic scores 0.91, which explains the Coconut cue.
**F2 toy (§53):** BD1 met (0.71 vs 0.50 at d7–12), but precision fails (FP 0.42). Parked.
**Kill test (§55):** KT1 partial at the boundary (the fall-back costs about ¼ of one advance's loss and 0.4 points of
accuracy); identity buys no reach. D3 is parked.
**Interchange (§56):** GT-distilled students' latents are age-invariant (G 0.09); T8 students' and parents' are not.
Path independence tracks one-loop reach (Spearman −0.87, post hoc).
**GSM8K step 2 (§68):** our own text-CoT, Coconut and written-state arms on GSM8K-Aug (3 seeds), with C1 controls on Coconut's latents.
**C1 frontier (§67):** the exact-marginal objective has partial-automaton local optima. The table is certified 1/6 at 6 codes but 10/10
at 12 codes. Every neural cell fails (GPT-2, random GPT-2, small transformer, MLP). Hard codes do not decay with length.
**§66:** in GPT-2 state tracking, the written-state chunked loop is 1.000 at N 17–32 vs text CoT 0.14 (fewer FLOPs). But the no-latent
version is identical, so the scaffold (short contexts plus written state), not the latents, does the work. C1 is not shown. See the range principle.
**§65:** restart labels equal true labels (a clean C4 teacher), but even true labels give the S3 single loop only 0.66 at N 9–16
and 0.06 beyond. The limit is the continuous state, not the labels.
**Beyond graphs (§64):** on S3, written-state restarts fix the single loop's length failure (0.995 at N 9–16). The failure is
not a count-keyed stop: the state loses validity. A5 never learned (0/10, also with the curriculum).
**F4 (§63):** not met on every measure. The hard straight-through translator never trained (flat losses), so it is uninformative on C7
for silent steps (§34's 0.36 stands). F3 was dropped at its premise check.
**§62:** depth structure does not install the age code: any label noise keeps latents age-dependent (uniform 0.67 vs clean 0.09).
The confidence step at the seam grows with structure (0.008 → 0.243). A one-sided miss bias removes reach entirely. Self-labels must be near-clean.
**F1 on GPT-2 (§54b):** not learnable from deep final answers (soft and stoch 0/3). With per-pass targets it is exact
to d14 (2.3×), readable, halting and cue-free. F1-S (§61) did not meet its predictions (1/3 seeds). One seed learned an exact, readable BFS to d14 from final answers alone; the other two
flood at long budgets. The F1 line is closed.
**§60b result:** Gate A not passed (0/4). Order consistency lets 2/4 seeds learn partially (best 0.88/0.84 on
instances/orders), but the tables are not algebraically valid and fade with depth. The boundary-memory line stops at
Phase A.
**§60b:** the development pilot collapsed to the identity map (0.874), caused by Theory's note 60-3 labels. One
revision: extensivity by construction, non-trivial labels, a size curriculum. No further revision.
**Boundary memory (§60):** the user's proposal. Phase A is registered (learned Boolean-table merger, with vs without
order consistency, 4 seeds, Gate A ≥ 0.95 on held-out instances, motifs and orders), and the Phase B gate is fixed.
**F6 (§58):** falsified. Aged transplants make aged latents interchangeable (dG −0.04) but give no reach. The age-reset
control does best. Path independence is not the cause of single-loop reach.

**Overnight results (2026-09-30 02:15 EDT; one line each, details in the § Result sections):**
* Anytime supervision (D): 8/10 extrapolate vs 3/9, Fisher p 0.055. The gain is at d7 only; the pointer falls back after
  K (§17).
* Depth wall (E): anytime runs stop at the trained depth whatever K (0/12 cross). This is a learned fall-back, not a
  capacity limit (§20).
* Fine-tuning repair (H): 1/18. The brake is kept, as a critical period predicts (§27).
* Jitter (F): alone it breaks learning (1/8); with anytime, no gain (§23).
* Set memory (G): unused, so A6 is untested (§24).
* Random-tight (K): consistent with anytime, on 4 competent runs (§31).
* Periodic clock (I): PI1 0.57, narrowly not met (§28).
* Chunk reset (J): PJ2 supported (6/8); best from-scratch d8 (§29).
* Stop key (Prop. 10, §35): the fall-back is keyed to counts. Skip → step key. History and step clock carry part of it.
* Prop. 10 mechanism across architectures (§40): brake removed, fall-back kept (1/16 pass 8), count carried in 25/25.
* PROTOCOL9 (Experiments, across architectures): the anytime gain at d7 is weak and family-dependent. It is strong in
  512-wide models only: the pooled 64 pairs give 18 vs 10 (p 0.09), and w512 alone has discordant pairs 9 vs 0. There is no d8 gain
  (`docs/PROTOCOL9_RESULTS.md`).
* **Re-anchoring distillation (L): 19/20 failures fixed without labels (0/20 plain), §32.** GT-distilled loops
  extrapolate to 2× their training depth (17/20, §37).
* **Architecture review (§47, updated by §48):** the best current design is a single-anchor checkpointed reasoner,
  trained label-free. Six chained segments reach d19–24 at 0.65, without compounding. Beams add nothing. Unproven: a
  learned write/halt policy, and transfer to GPT-2.
* One restart covers exactly its trained horizon: 0.91–0.95 up to 6 hops from the anchor, chance from 7 (§46).
* A learned write head (0.85 agreement) and a closure halt that tracks depth (3/4/5 segments for 12/18/24 hops, 8/10
  models) give label-free C5 v1–v2 (§49).
* A halt learned from final-answer reward beats closure and fixed halts (9/10), but stops at 2 segments regardless of
  depth. The bottleneck is that max-combined answers accumulate false positives (§50).
* **Checkpointed restarts (CLR-1, §45): editing the written checkpoint changes answers as predicted (IIA 0.84 for sets,
  1.00 single).** Chains reach d13–18 label-free (single anchor 0.67). Sets are used but don't beat one anchor.
* Depth of training sets reach (§41): trained to 9 → fades by ~15; to 12 → ~22; to 18 → flat to 23 on §37's probe (0.88 at 25–30 on an easier width-1 probe).
* Second label-free rung (M): the teacher beats self-labels (10/12, p 0.019). **With the agreement filter (M-2), 12/20
  reach 13–18 hops in one loop, and the old rung holds** (T8F > T8 9/11, p 0.033; §38).
* T12 step-adversarial thoughts (§39): breaks learning (2/8 competent). Removing the count is not the fix: extrapolators
  keep it (§37).
* **Silent thoughts are causally editable (T-C7c, §34):** IIA 0.36; answers follow the edited pointer (+0.83 / −0.59).
* Counter clamp (§36): rescues 9/34 vs control 0/34, but costly. LE-R: supported out of sample (§25).

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

**Running (as of 14:27 on 2026-09-29; superseded by the overnight results above):** Pilots D/E (anytime vs control, 28 same-seed pairs; ~01:00–05:00 EDT 2026-09-30), then F
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

**Result (2026-09-30, committed 440d6ae 00:12:55 EDT; all 24 runs; `theory/score_pilotDE.py` → `theory/results/score_pilotDE.json`,
measurements in `theory/results/pilotDE_meas.json`). PD1 supported; PD2 and PD3 not met; PD4 met.**
* Runs: AT 12 (pd_at_s905 ran and exhausted its stage-1 guard, so it counts as run and not competent, §15 rule); C 12.
* **PD1 supported.** Competent AT 10/12, of which 8 extrapolate (0.80). Competent C 9/12, of which 3 extrapolate (0.33).
  Difference 0.47 ≥ 0.20, and share(AT) ≥ 0.60. One-sided Fisher p = 0.055.
* **PD2 not met.** Mean far8–10: AT 0.543, C 0.526 (difference 0.017 < 0.05).
* **PD3 not met.** Pointer advances at every step: AT 4/10 (0.40 < 0.60), C 0/9.
  - Exact rule used, per amendment E-2 (iii): p[i] = p[i−1] + 1 for every i in 2..K (transitions h_1→h_2 through
    h_{K−1}→h_K), except at most one i with p[i] = p[i−1], and no i with p[i] < p[i−1].
  - Audit's more literal reading of §17 starts at h_2 (transitions h_2→h_3 onward). Under it pd_c_s902 passes, giving
    C 1/9. PD3 fails under either reading.
* **PD4 met.** Gate rate AT 0.83, C 0.75.
* **Paired sign test (amendment D-1; the metric was not named there, so both are reported).** Pairs with both runs
  competent: 8 pairs.
  - d7, the metric that defines "extrapolates": AT > C in 6/8, one-sided p = 0.145.
  - far8–10: AT > C in 7/8, one-sided p = 0.035.
  - Added 2026-09-30 after Audit's recomputation.
* **Reading.**
  - Anytime supervision removes the in-range brake. The pointer runs `1 1 2 3 4 5` through h_5 in 6/10 competent AT
    runs and in 0/9 controls (controls: the `1 1 1 2 2 3` brake of §25).
  - But the continuation past K dies within one to three steps. In 7/10 AT runs the pointer falls back to 2–3 after K
    (e.g. s900 `1 1 2 3 4 5 6 2 2 2 2 2`). Only s903 and s910 reach 8–9.
  - So the gain is concentrated one hop past training (d7, which defines "extrapolates"). Distances 8–10 barely move.
  - This is the split §22 predicts: anytime supplies identification (C1: unit speed, no clock) but not stability (C3: a
    restoring map). Pilot F's JAT arm (jitter + anytime) is the registered test of whether C3 adds the missing distance.

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

## 20. Pilot E — depth wall or schedule end? (pre-registered 2026-09-29, committed 3ded4f5 11:49:57 EDT [first written "~12:40", an estimate; see the timestamp table], before any Pilot D or E run started;
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

* **Amendment E-1 (2026-09-29, committed b895d96 11:50:43 EDT [first written "~13:00", an estimate], before any Pilot D or E run started; prompted by the experiments session).**
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

**Result (2026-09-30, all 56 runs; scored after the last pair was fetched at 00:23; `theory/score_pilotDE.py` →
`theory/results/score_pilotDE.json`; committed with this text). PE1: DW supported by the registered rule. PE3 and PE4 not
met; PE2 reversed.**
* **Competent runs.** D: AT 10/12, C 9/12. E8: AT 7/8, C 7/8. E3 (R rule): AT 5/8, C 5/8.
  - Guard-exhausted runs count as run and not competent: pd_at_s905, pe3_at_s1014, and the E3 runs that stopped at
    stage 4.
* **PE1 — DW supported.** 16 evaluable pairs (D 8, E8 6, E3 2); anytime higher in 10 (0.63 < 0.70; one-sided sign
  p = 0.23).
  - By arm: D 7/8, E8 3/6, E3 0/2.
  - 0 of the 12 competent anytime runs in E8 ∪ E3 cross (share 0.00 ≤ 0.20).
* **PE2 — reversed.** Crossing share, anytime vs control: D 0.30 vs 0.11 (+0.19); E8 0.00 vs 0.43 (−0.43); E3 0.00 vs
  0.00. Slack did not make anytime's advantage larger in E8. Anytime lost there.
* **PE3 — not met.** The pointer passes trained depth + 2 in 2/22 competent anytime runs (0.09) and 1/21 competent
  controls (0.05).
* **PE4 — not met.** Blocking layer-2 latent→latent attention from step K raises beyond-depth accuracy by ≥ 0.05 in 6/17
  non-crossing competent controls (0.35 < 0.50). Mean |change| in anytime runs is 0.045 (≤ 0.05).
* **Reading.**
  - By the registered rule this is a depth wall: anytime runs do not cross the trained depth, whatever K.
  - But the wall is not a limit on what the network can represent. In E8, 3/7 slack controls cross d8 (0.67–0.74), and
    their pointers run past depth 6 (s1003 reaches 8).
  - What stops the anytime runs is a learned fall-back at the trained depth. In E8 it comes *inside* the budget: the
    pointer drops at h_5–h_8 with steps to spare (§35).
  - So the registered dichotomy (DW vs SE) was too coarse. Anytime removes the slack-induced brake (in-range unit speed,
    §17), but it aligns step count and hop count with the trained depth. Their continuation is then unidentified in a
    new way.
  - §35 (Prop. 10) states this, and its P10e test (skip/delay interventions) decides whether the fall-back is keyed to
    the step (latent history length) or to the hop count.

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
*Measurement note (22:16 EDT, after PROTOCOL8 Phase 0).* A no-CoT model trained from answer labels alone can sit at chance
(Phase 0: 0.63 at m = 0, ≈ 0.5 deeper) even where the same architecture solves the problems in one pass after a CoT or
latent curriculum (addendum Q: Coconut GPT-2 at K′ = 0 scored 0.976 on native ProsQA). Such a baseline measures
*learnability without intermediate supervision*, not one-pass *capacity*. So c should be measured on the latent-trained
model itself: c is the largest depth where accuracy at K′ = 0 matches accuracy at K′ = K. The latent benefit per stratum
(accuracy(K) − accuracy(0)) is the direct evidence of latent use (PROPOSALS §9, F4b).

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

**Result (2026-09-30, committed 440d6ae 00:12:55 EDT; all 32 runs; `theory/score_pilotFG.py` → `theory/results/score_pilotFG.json`; z from
`theory/results/transfer_pilots.json`).**

| Arm | Competent | Extrapolate (of competent) | Median d8 |
|---|---|---|---|
| J | **1/8** | 0 | 0.508 |
| JAT | 6/8 | 5 | 0.568 |
| ATW | 7/8 | 5 | 0.570 |
| ATL | 7/8 | 5 | 0.517 |
| C (pd_c_s900–907) | 5/8 | 2 | 0.503 |
| AT (pd_at_s900–907) | 6/8 | 6 | 0.584 |

pd_at_s905 exhausted its stage-1 guard, and so did pf_atw_s905. Both count as run and not competent.

* **PF5 not met (learnability).** J 1/8 vs C 5/8. Jitter alone breaks the gated curriculum, the failure mode PF5 guarded
  against. JAT 6/8 vs AT 6/8 is fine.
* **PF1, PF2 unresolved** (0 counted pairs, because J has only one competent run).
* **PF3 unresolved** (4 counted pairs; JAT > AT in 2). Descriptive bar met: 5/6 competent JAT runs extrapolate (0.83).
  Median d8 JAT 0.568 vs AT 0.584.
* **PF4 not supported.** ATW > ATL in 2 of 5 counted pairs. Descriptive: median d8(ATW) − median d8(AT) = −0.014 (within
  0.05). PD1 is supported, so PF4 is in force.
* **Manipulation check (descriptive, not registered).** Jitter does make the map restoring when it is learnable.
  - Median z over competent runs: JAT −0.14 vs AT +0.05. z ≤ 0 in 6/6 competent JAT runs vs 2/6 competent AT runs.
  - Controls are restoring too (C −0.14, 4/5 ≤ 0). So anytime *loosens* the map, and jitter on top brings it back to
    the control's level.
* **Reading.**
  - Contraction training reaches its mechanism: a restoring map (C3).
  - But on top of anytime it does not add reach: d8 is no better, and far8–10 is lower (0.534 vs 0.554).
  - Alone, it prevents learning.
  - So the post-K collapse of the AT pointer (§17 result) is not fixed by making the map restoring. What restoring
    buys is a stable fixed point, and the AT runs' post-K pointer already sits at one (it falls back to 2–3 and stays).
    The missing ingredient is continued *advance*, not stability. That points to the clock/identification side, i.e.
    Pilots I, J and K, and to re-anchoring (§26), which advances by restarting rather than by extrapolating the loop.
  - Selection window (T7): anytime only in stages 1–3 (ATW) is as good as full anytime (median d8 within 0.014), but it
    does not beat anytime from stage 4 on (ATL) pairwise. The window effect is not established.

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

**Result (2026-09-30; all 32 runs; `theory/score_pilotG.py` → `theory/results/score_pilotG.json`; ablations in
`theory/results/setmem_ablation.json`, set depths in `set_traj.json`; committed with this text). The set was unused,
so there is no verdict on A6. PG4 met; PG5 unresolved.**
* **Manipulation check fails.** Ablating the set matters (≥ 0.05) in 3/11 competent G ∪ GAT runs and 0/13 GL ∪ GLAT
  runs. Final gates stay near 0 (|γ| ≤ 0.11). So PG1–PG3 read "set unused" by the registered rule.
* **PG4 met.** Competent: G 7 vs C 5, GAT 4 vs AT 6, GL 5 vs C 5, GLAT 8 vs AT 6.
* **PG5 unresolved.** 3 runs use the set (G family), and 0 in the GL family.
* **Reading.**
  - A set write gated from 0 is not taken up by gradient descent at this scale. The pointer solution forms first (§10)
    and the set path never becomes useful enough to open.
  - A6 is untested, not refuted. A test would need the set to carry the answer by construction, e.g. queries that read
    only the set. That is the checkpointed design's set-valued checkpoint (§33.1), where the set is the only channel
    across a restart.

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

*Formal anchor (21:52 EDT, from the Literature session, `docs/LIT_REQUESTS_2026-09-29.md` Part C).* At finite precision
the state set is finite, so a homogeneous map s_{k+1} = f(s_k) is eventually periodic (pigeonhole). The average
advance per step is then m/q for some period q, and a non-integer speed needs q ≥ 2 phases, i.e. a mod-q counter —
a clock (standard for finite automata: ultimately periodic outputs). With exact reals, an irrational speed needs a
rotation on a phase variable (circle maps; the rotation number is p/q iff there is a period-q orbit — Katok & Hasselblatt
1995 ch. 11). Floor(nα) advance sequences for irrational α are Sturmian (Morse & Hedlund 1940), so no finite automaton
produces them. **Added prediction (mode locking):** learned fractional speeds should lock to small rationals (½, ⅓, ⅔,
…) rather than irrational-looking averages; the in-sample speeds 3/6 = ½ and 3/8 are consistent.

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
  - **LE-R result (2026-09-30, after the experiments session posted R's competence labels in 26c2ed1;
    `theory/results/le_pointer.json`). LE3-R and its dose ordering are supported.**
    - Competent cap-3 runs (8): p[K−1] = 3, 1, 2, 2, 2, 0, 2, 3, median 2 (≤ 2).
    - Competent cap-4 runs (8): median 3 (≤ 3).
    - Dose ordering: 2 < 3 ≤ 3.
    - Provenance: 6 of the 16 were measured blind before the labels. The rest were measured with the same frozen
      script after the labels were posted.

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

**Result (2026-09-30 00:01 EDT; all 48 runs; `theory/score_pilotHIJKL.py H` → `theory/results/score_pilotH.json`;
pointers from `theory/results/le_pointer.json`).** **Critical-period reading; PH1–PH4 not met.**
* Competent: FT-C 20/20, FT-AT 18/20, FT-JAT 5/8 (guard: **FT-JAT flagged**, gate rate 0.63 < 0.75).
* **PH1 not met.** Extrapolating share among competent runs: FT-AT 1/18 (0.06; s112, a retrieval-type model, d7 0.636),
  FT-C 0/20. The critical-period clause holds (0.06 ≤ 0.00 + 0.10). d7(FT-AT) > d7(FT-C) in 12 of 18 models with both
  competent (0.67 < 0.75; one-sided sign p = 0.12); medians d7 0.545 vs 0.527, d8 0.511 vs 0.505.
* **PH2 not met.** Median p[K−1] = 3 in both FT-AT (n 18) and FT-C (n 20): the brake did not go. Against the initial
  models' own trajectories (measured 2026-09-30 after 00:01, committed in 8f80824 00:02:46, with the same `le_pointer.py`, added to le_pointer.json as
  p30_s*), the whole trajectory h_0..h_11 is unchanged, digit for digit, in 13/20 FT-C models and 6/20 FT-AT models
  (s109, s118, s125, s132, s145, s158, all the `1 1 1 2 2 3 2 3 2 3 2 3` period-2 oscillation of §25 or its period-3
  relative). The in-range part h_0..h_5 is unchanged in 18/20 FT-C and 9/20 FT-AT models. Anytime fine-tuning moves the
  pointer somewhat, but not past the brake.
* **PH3 not met.** FT-AT extrapolating share: retrieval-type 1/12, local-type 0/6.
* **PH4 not supported** (and its arm is flagged): d7(FT-JAT) > d7(FT-AT) in 3 of 5 counted local-type models.
* **Reading.** Removing slack after training does not repair a trained failure at this fine-tuning dose (6000 steps,
  lr 1e-4, ~4000 steps at full depth). The step-counting circuit chosen at curriculum stage 3 (§10, §25) is kept, in
  most models down to its exact pointer schedule. That is what a critical period predicts. **Caveat:** the dose was fixed in
  advance and is gentler than training (lr 1e-4 vs 5e-4). "Locked in" is established only for this dose, not for a
  larger learning rate or a re-initialised step-sensitive component. Consequence for the fixes: the conditions of §22
  must hold **at circuit selection**, as Prop. 4(a) states (random-tight or anytime budgets from the first latent
  stage). Post-hoc anytime fine-tuning is not a repair route; Pilot L (§32) tests a different one, distillation of the
  re-anchored teacher on deep graphs.

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

**Result (2026-09-30; all 16 runs; `theory/score_pilotHIJKL.py I` → `theory/results/score_pilotI.json`). PI1 narrowly
not met; PI2 inconclusive; PI3 not met.**
* **Competent:** PER2 4/8 and PER3 3/8 (pi_per3_s905 exhausted its guard), vs C 5/8.
* **PI1.** The pointer keeps advancing after K in 4/7 competent PER runs (0.57 < 0.60), vs 0/4 same-seed controls.
* **PI2.** far8–10 at K′ = 16: PER > C in 3/5 counted pairs.
* **PI3.** The in-range speed is kept after K in 1/7.
* **Descriptive.** Pointer passes 8: PER 3/7, controls 0/5. Median d8: PER2 0.622, C 0.503.
* **Reading.** A periodic clock removes part of the stop key (§35) in some runs, at a cost in learnability. It does not
  keep the in-range speed.

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

**Result (2026-09-30; all 16 runs; `theory/score_pilotHIJKL.py J` → `theory/results/score_pilotJ.json`). PJ2 supported;
PJ3 met; PJ1 not met.**
* **Competent:** CH3 6/8, CH3AT 4/8 (pj_ch3at_s905 exhausted its guard). C 5/8, AT 6/8.
* **PJ2 supported.** Pooled pair rule on d8: 6/8 (one-sided sign p = 0.14). Median d8: CH3 0.604 vs C 0.503; CH3AT 0.644
  vs AT 0.584. That is the highest median d8 of any from-scratch toy arm.
* **PJ1 not met.** The pointer passes 8 in 4/10 competent J runs (0.40 < 0.50), vs 1/11 same-seed C ∪ AT runs.
  - CH3AT: 2/4 run at unit speed to 11. The other 2 fall back into a period-3 cycle (s901 `1 1 2 3 4 5 6 1 2 3 1 2`).
  - AT (same seeds): 1/4.
* **PJ3 met.**
* **Reading.** Cutting the latent history every 3 steps removes the stop key in half the anytime runs. The fall-back in
  the rest is locked to the chunk period. This refutes P10b′ (§35), which predicted no gain over AT.

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

**§30 results (scored 2026-09-29 21:58 EDT; `theory/exp14_branch_renorm.py` → `theory/results/exp14_branch_renorm.json`;
`le_pointer.py` for randk).**
* **F3a renormalisation — RN1 not met, RN2 not met.** Renorm + block reaches 7-hop success in 45/59 (bar 46) with far8–10
  0.590 (bar > 0.594); renorm alone 26/59, 0.543 (single episode 24/59). Matching each model's in-range amplitude is
  no better than a constant α = 1.3 (46/59, 0.594). Amplitude restoration at inference is exhausted at ≈ 46/59; the
  remaining gap to re-anchoring (55/59) is not an amplitude problem. F3a is dropped.
* **R-randk — RR1 not supported (1 of 2), RR2 1 of 2.** randk_s0 runs `1 2 3 4 | 0 0 …`: unit speed to depth 4 by the
  minimum budget, then the pointer stops — exactly Prop. 4's zero-loss prediction. randk_s2 runs `1 1 1 1 2 3 4 3 3 …`:
  it dwells three steps and reaches depth 3 only at h_5, so it would fail at budget 4 by the idealised account (its
  loss at K = 4 was not examined). randk_s1 (not competent) cycles. The P22 re-analysis neither confirms nor refutes the
  tightness account; F1a's own arm (K ∈ 1..6, no step embedding) remains the test.
* **B1 branch selection (descriptive; amendment B1-a).** Effective number of attended frontier edges (exp entropy over the
  W frontier edges), averaged over steps 1..K and all 59 models:

  | W | sweep head | head-mean | range over the 8 heads | at step K: sweep / head-mean |
  |---|---|---|---|---|
  | 2 | 1.37 | 1.56 | 1.57–1.64 | 1.38 / 1.60 |
  | 4 | 1.95 | 2.59 | 2.61–2.80 | 1.65 / 2.25 |
  | 6 | 2.33 | 3.49 | 3.56–3.85 | 1.76 / 2.78 |

  Neither pure superposition (≈ W) nor single-branch following (≈ 1): attention covers part of the frontier with uneven
  weights, and the chain-selecting sweep head is the most concentrated. Relation to P25: the effective number grows with
  W (so per-edge mass falls as the frontier widens, as P25 found), but sub-linearly — 1.4 → 2.3 for W 2 → 6 in the
  sweep head. The earlier "about one branch" from the debug model is withdrawn as a general claim: it held for that
  model's sweep head at step K. What it implies for re-anchoring stands: the most-attended edge lands on the probed chain
  in ≈ 1/W of graphs, because attention mass is spread unevenly over several branches rather than concentrated on the
  queried one. Literature context (`docs/LIT_REQUESTS_2026-09-29.md` Part C): partial, uneven superposition matches Gu
  2609.13747 (uneven weights) and Coconut's "parallel exploration then commitment", not Zhu's balanced superposition.
* **Disclosures for B1 (added 21:59 EDT at the Audit session's request).**
  - *Mid-run code fix.* In the first pass, the head-mean line computed `A[i, :, fr].mean(0)`. NumPy advanced indexing
    returns that as (frontier edges × heads), so the "head-mean" was an entropy over the 8 heads rather than over the
    frontier edges. Head-mean values seen in the run log before the fix (e.g. 3.27 / 3.45 / 3.77 for p30_s111 at
    W = 2 / 4 / 6) exposed the bug, since 3.27 > W = 2 is impossible. The per-head values, including the sweep head,
    were computed correctly throughout and were seen during the run. The fix (8b0670e, 21:52:55; a literal-newline
    slip repaired in 62dc024) changed code only, not the registered rule. All B1 numbers above come from the
    `--branch_only` recompute after the fix. The renormalisation (RN) numbers were not affected.
  - *Aggregation.* `branch[k][W]` is keyed by latent step k = 1..K (K = 6 or 8) and width W. Each holds 9 entries
    [count ≥ 0.5 × max, effective number]: the 8 layer-1 head slots, then the head-mean. For each model the sweep head
    is its exp2 top head. Table values are unweighted means over all (model, step) pairs, pooled across the 59 models,
    so K = 8 models contribute 8 steps each. Pairs where the sweep head had no frontier (None) are skipped. The per-head
    range is the min and max over the 8 head *slots* of that pooled mean; slots are not functionally aligned across
    models. The "at step K" column uses k = K only. Other reasonable aggregations (per-model first, then models)
    shift the numbers by ≤ ~0.2, and the pattern holds under all of them (the Audit session's re-check).

## 31. Pilot K — label-free tight budgets (PROPOSALS §9 F1a) (pre-registered 2026-09-29 22:22 EDT, before any Pilot K run started)

**Claim tested (Prop. 4 with (a) and refined (b), §22.7).** Tightness, not intermediate labels, is what identifies unit
speed: random budgets whose support reaches the tight budget at the circuit-selection stage, drawn independently of
the example, should do what anytime supervision does, with final-answer labels only.

* **How this differs from what failed.** P22 randk used K ∈ 4..8 *with* step embeddings, so at curriculum stage 3 (1–2
  latent hops needed) every budget had slack ≥ 2 and a clock was available. Arm OV used budgets 6–8, never tight. Kuo
  et al.'s schedule is n-centred, K = clip(n + Δ, 1, 20) with Δ ~ U{−5..5}; it sometimes undershoots, but it is tied to
  the example's length. **RT** here: K ~ U{1..6} per batch (`--K_random 1:6`), independent of the example, no step
  embedding (P30 recipe), final-answer loss at the sampled budget; evaluation and the curriculum gate at K = 6. The
  support contains the tight budget at every stage.
* **Arm.** RT, seeds 900–907, paired by seed with Pilot D's control (pd_c) and anytime (pd_at) runs. 8 runs.
* **PK1 (primary; tightness without labels).** §23 pair rule on d8 (K′ = 12): RT > C (≥ 75 % of ≥ 5 counted pairs).
* **PK2 (RT ≈ anytime).** |median d8(RT) − median d8(AT)| ≤ 0.05, and the RT extrapolating share among competent runs
  is ≥ AT's − 0.15.
* **PK3 (no brake; Prop. 9).** Median p[K−1] of competent RT runs ≥ K − 2 = 4 (anytime-like), against ≈ 3 for fixed-K
  controls (§25).
* **PK4 (learnability).** Competent RT runs ≥ competent C runs − 2 (of 8).
* **Falsified** if PK1 is not supported while Pilot D's anytime-vs-control comparison is. Then tight budgets without labels
  are not enough, and the minimum-progress labels are doing the work.

**Result (2026-09-30; all 8 runs; `theory/score_pilotHIJKL.py K` → `theory/results/score_pilotK.json`; pointers in
`theory/results/le_pointer.json`). PK1 unresolved; PK2, PK3 and PK4 met; not falsified.**
* **Competent:** RT 4/8, C 5/8, AT 6/8 (seeds 900–907). All 4 competent RT runs extrapolate.
* **PK1 unresolved.** 3 counted pairs, RT > C in 2. The falsification clause does not apply: its referent PD1 is
  supported, but PK1 is unresolved rather than "not supported". For reference, same seeds, AT > C in 4/4 pairs.
* **PK2 met.** Median d8: RT − AT = −0.013. Extrapolating share: 1.0 in both.
* **PK3 met at the threshold; 2/4 runs brake.** Median p[K−1] is 4.0, and the values are bimodal (2, 6, 6, 2).
  s900 and s905 have p[K−1] = 2, the fixed-K brake.
* **PK4 met.** 4 vs 5 competent.
* **Reading.**
  - With final-answer labels only, random tight budgets reach the anytime outcome: d7 median 0.62, d8 0.57 vs AT 0.58.
    They also share its fall-back after K (s901 `1 2 3 4 5 6 7 0 …`, s904 `1 2 3 4 5 6 2 2 …`), as P10a predicted (§35).
  - This is consistent with, not proof of, the claim that minimum-progress labels are not needed for the in-range fix
    (C4): only 4 runs are competent, PK1 is unresolved and PK3 is split. Neither method fixes continuation past the
    trained depth.
  - Learnability is a real cost: 4/8 competent.

## 32. Pilot L — distilling re-anchored inference into one latent loop (PROPOSALS §9 F5a) (pre-registered 2026-09-29 22:57 EDT, before any Pilot L run started)

**Idea.** Re-anchored inference (§26) is right where the single uninterrupted loop is not. Use it as a teacher for
deeper problems and fine-tune the model to reproduce its answers in *one* episode with K′ = 12. This is
amplify-then-distil (Christiano et al. 1810.08575) with a clock-resetting procedure as the amplifier; related:
self-improving transformers 2502.01612, Deng et al. 2405.14838, 2407.06023, STaR / Quiet-STaR, progressive
distillation 2202.00512 (precedents from the Literature session).

* **Code.** `theory/distill_reanchor.py` (smoke-tested on CPU with all three teachers; its outputs load in eval_arch).
  Fine-tuning from each model, fresh AdamW, lr 1e-4 (100-step warm-up, cosine), 3000 steps, bs 256. Each step has a
  normal batch (training graphs, full depth, K = 6, delayed + upfront BCE) and a deep batch (long-chain graphs from the
  length_gen generator with fresh random seeds; all A/B/C/D chain nodes queried; one student episode at K′ = 12).
* **Arms** (the 20 P30 K6 failures of Pilot H, each model's own seed and sweep head from exp12):
  - **L-T8:** deep targets from a frozen copy of the initial model run with re-anchoring (re2top3: episode 1 from START,
    episode 2 from each of the 3 most-attended edges' targets at step K; max-combined logits > 0).
  - **L-self:** deep targets from the frozen copy run as one episode at K′ = 12 (plain self-training on deep graphs; the
    control that isolates the re-anchoring teacher).
  - **L-GT:** the true labels (upper bound).
  - Lower bound: Pilot H's FT-C (plain fine-tune on normal graphs).
* **Outcomes.** theory/eval_arch.py: d7 and d8 at K′ = 12 (length_gen seed 99; graphs different from the training ones),
  extrapolates (extrap_K12), delayed in-distribution bal-acc.
* **PL1 (primary: the re-anchoring teacher matters).** §23 pair rule on d8: L-T8 > L-self (≥ 75 % of ≥ 5 counted pairs).
* **PL2 (label-free extension works).** L-T8 extrapolates in ≥ 50 % of the 20 models, and in ≥ 0.30 more of them than
  Pilot H's FT-C.
* **PL3 (upper bound, descriptive).** median d8: L-GT ≥ L-T8.
* **PL4 (no in-distribution cost).** Median drop in delayed bal-acc from the initial model ≤ 0.02 in L-T8.
* **Caveat, stated now.** The deep graphs come from the same generator as the evaluation, so this is label-free
  improvement on harder problems, not zero-shot extrapolation. The pure test of the teacher is L-T8 vs L-self: same
  graphs, different label source.
* **Where.** 60 runs (≈ 15 min each) on ~12 new Vast hosts, labelled theory-L*, destroyed as each queue ends.

**Result (2026-09-30; all 60 runs; `theory/score_pilotHIJKL.py L` → `theory/results/score_pilotL.json`; committed with
this text). PL1–PL4 all met.**
* **Competent:** T8 20/20, self 20/20, GT 19/20.
* **PL1 supported.** d8(L-T8) > d8(L-self) in 20/20 pairs (one-sided sign p = 1e-6).
* **PL2 met.** L-T8 extrapolates in 19/20 models, Pilot H's FT-C in 0/20.
* **PL3.** Median d8: L-GT 0.667 ≥ L-T8 0.638. L-self 0.505, the same as FT-C's 0.505.
* **PL4 met.** Median drop in delayed bal-acc: 0.003.
* **Descriptive.**
  - The student recovers most of its teacher's gain. Median d8: teacher (re2top3, probe graphs) 0.698, student 0.638,
    one uninterrupted episode 0.503.
  - Student medians at K′ = 12: d7 0.651, d8 0.638, d9 0.627, d10 0.542. At K′ = 16, d10 rises to 0.591 and d11 to
    0.551: more steps than distilled help.
* **Reading.**
  - The same deep graphs with the model's own single-episode labels teach nothing (self 0/20). With labels from its own
    re-anchored inference they fix 19/20 failures. So the teacher, not the data, does the work.
  - This is label-free (C4) improvement in depth, obtained by distilling checkpointed inference into one loop. As
    registered, it is not zero-shot extrapolation: the deep graphs come from the evaluation generator.
  - It also fixes what fine-tuning with anytime could not (Pilot H: 1/18). The critical period binds loss reweighting,
    not a teacher that supplies deeper targets.
  - Natural next step: iterate. Re-anchor the student, whose episodes now reach about 12 hops, to reach about 24, then
    distil again. That is an amplify-and-distil ladder in depth, with no labels at any rung.

## 33. Toward the north star: the checkpointed latent reasoner (CLR), C7 tests, and what each pilot tests (2026-09-29 23:32 EDT; design, exploratory — nothing here is registered)

Goal (`docs/NORTH_STAR.md`, set by the user): one model meeting C1–C7 at once. This section turns §26's re-anchoring
result into an architecture, defines the causal readability tests for the toy, and maps the running pilots onto the
conditions. The Literature session has been asked for prior work on each part; tests are registered only after it
answers.

### 33.1 The architecture (toy version)

A problem is solved in **segments**. Each segment is:
1. **Input.** The prefix (edges, question) and the current **checkpoint**: a *set* of node-id tokens F_j (type
   CHECKPOINT, embedded with the shared id table like any node token). F_0 = {start}.
2. **Silent work.** Up to L latent steps (L ≤ the trained budget). They attend to the prefix, all checkpoint tokens so far
   and this segment's own latent positions only — the latent history is cut at every checkpoint (Pilot J's chunk reset).
3. **Write.** A checkpoint head reads the last thought and scores every node id through the id table,
   s_v = ⟨W h, E_id[v]⟩. The next checkpoint is F_{j+1} = {v : σ(s_v) > ½}, capped at W_max tokens.
4. **Restart.** The next segment starts from a fixed START-type embedding plus the checkpoint tokens, **not** from the
   previous thought, so no residue crosses a checkpoint except through the written text.
5. **Answer.** The queries read every checkpoint and the current segment. A halting head decides when to stop writing
   and answer.

What each part is for:

| part | condition | mechanism (THEORY section) |
|---|---|---|
| segments of ≤ L steps | **C2** | the step count never leaves the trained range; deeper problems use more segments, each an exact copy of a trained computation (§25 Prop. 9, §26, §29) |
| restart from crisp checkpoint embeddings | **C6** | amplitude restored at every checkpoint (§26: re-anchoring rescues the fading failures the history block could not) |
| checkpoint = set of node ids, restart only from it | **C7** (checkpoints) | readable by construction; and because the restart uses only the written text, the round-trip and hidden-channel tests below pass by construction if and only if the isolation holds |
| set-valued checkpoint | open question 2 | keeps the parallelism: the model's own single anchor is on the right branch in only ≈ 1/W of graphs (§26 controls); attention covers ≈ 1.4–2.3 of W branches (§30 B1); a set can carry all of them |
| learned write/halt heads | **C5** | a PonderNet-style halting head per latent step (end the segment) and per segment (answer now); a natural stopping signal is closure — the checkpoint stops changing (§22.4 Prop. 6, C4 of the framework) |
| label-free training of the checkpoint | **C4** | the checkpoint is discrete, so options are: (i) straight-through Gumbel-sigmoid over node ids; (ii) REINFORCE on the final answer; (iii) self-distillation from the model's own attention-decoded frontier (T8, as in Pilot L). Only (iii) has evidence so far, and only for single anchors |
| silent stretches | open question 1 (C7 for latents) | tested separately (33.2, T-C7c) |

### 33.2 The C7 tests on the toy

Node ids are the toy's "human language": a checkpoint "frontier = {37, 42}" is a sentence.
* **T-C7a — edit test (checkpoints).** Take the model's checkpoint and edit it: drop a node, add a decoy node at the
  same depth, or swap in a node on another branch. Re-run from the edited checkpoint.
  - *Prediction:* every query's answer changes exactly as BFS from the edited frontier predicts, and nothing else
    changes.
  - *Metric:* interchange-intervention accuracy (IIA; DAS 2303.02536) — the share of edits for which every query's
    output equals BFS on the edited frontier. Sensitivity (did the output change at all) is reported separately, as is
    a per-node-id claim-flip audit (Literature's recommendation after 2607.20379).
  - *Input-only control:* can the checkpoint be predicted from the graph and the step count alone, without the
    thought? (2509.13316 found verbalisers matching their own performance from the input.) For the toy the true
    frontier is input-computable, so the checkpoint counts as the *model's* only if it tracks the model's errors: on
    graphs where the model is wrong, the checkpoint should be wrong in the same way.
  - *First version, runnable now on the 59 existing models with no training:* re-anchored inference (§26) with the
    episode-2 anchor edited (oracle chain node, decoy node, other-branch node).
* **T-C7b — round trip and hidden channel.**
  - *Round trip:* decode the checkpoint to text, re-encode the text as checkpoint tokens, continue; behaviour must be
    unchanged.
  - *Hidden channel:* keep the checkpoint text and replace everything before it — the previous segment's latent states —
    with those of a *different* input; behaviour must be unchanged.
  - In the CLR these pass by construction if the restart truly isolates. The tests certify the isolation in the
    implementation. In a model without isolation (plain Coconut) the hidden-channel test is the measurement: how much
    behaviour moves when only the unwritten state changes.
* **T-C7c — silent-stretch edit (latent thoughts).** At latent step k, edit the thought's pointer along the model's own
  node-embedding directions: h′ = h + α(E_node[u] − E_node[v]), with v the node it points at. Continue.
  - *Prediction if the silent thought is faithful to its decoded pointer:* the next steps follow u's successors (sweep-head
    target and query answers).
  - Part I found probe directions causally inert; this uses the model's own input embedding instead, and Finding 1
    reported every branch causally editable at the edge level. A clean positive here would be the first C7 evidence for
    silent thoughts.
  - Exploratory, CPU, existing models.

*Addendum to 33.2 (23:41 EDT; Literature, `docs/LIT_NORTHSTAR_2026-09-29.md` §5, commit 49ff87e).*
* **Scope of the round trip.** In a standard transformer, restarting from the *full* emitted text is identical by
  construction, because the KV cache is a function of the text. The test is informative only when the restart drops
  text or carries non-text state; the CLR's latent loop is the second case, so T-C7b is well posed.
* **Prediction from Delethink (2510.06557).** It clears the KV cache every chunk and keeps a short text carry-over. It
  breaks exactly when the live state exceeds the carry-over (CrossWordBench 14×14, but not 7×7). Our analogue:
  restarted-vs-uninterrupted agreement should fail when the frontier holds more branches than the checkpoint can carry
  (W > W_max).
* **Closest trained precedent.** Markovian Transformers (Viteri et al. 2404.18988): all question → answer information
  must pass through a bounded CoT, trained with RL and no step labels. They show larger log-prob drops under corruption
  and CoTs that transfer across models. This makes "checkpoint sufficiency" a C4-compatible training objective for the
  CLR (option (iv) beside (i)–(iii) in 33.1).
* **Metrics.** Per-instance answer agreement and the KL divergence between restarted and uninterrupted answer
  distributions, alongside IIA for the edit test. The literature reports only aggregate accuracy for restarts.

### 33.3 What each pilot tests (for Audit's status table)

| pilot / study | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| D/E anytime vs control (§17, §20) | | ● extrapolation past K | | | | ● seed variance (pairs) | |
| F jitter (§23) | | ● | | | | ● restoring map | |
| G set memory (§24) | | ● | | | ○ closure = natural halt | ● no fade / overthinking | ○ set readable by construction |
| H fine-tune repair (§27) | | ● repairability | | | | | |
| I periodic clock (§28) | | ● | | | | | |
| J chunk-reset attention (§29) | | ● | | | | | ○ isolation needed for checkpoints |
| K label-free tight budgets (§31) | | ● | | ● | | | |
| L re-anchoring distillation (§32) | | ● | | ● self-labels | | | |
| §26 re-anchoring (done) | | ● (inference) | | | ○ hand-set K | ● | ○ checkpoints = node ids |
| PROTOCOL8 (Experiments) | ● GPT-2, F1b | ● | | ● final answers + curriculum | | | |
| PROTOCOL9 (Experiments) | | ● generality of anytime | | | | | |

● = the pilot's primary claim bears on the condition; ○ = relevant but not tested as such. **Gaps:** C3 (nothing
compares to text CoT at matched compute), C5 (no learned halting or writing yet), C7 for silent thoughts (T-C7c is the
first test).

### 33.4 Proposed order (after tonight's pilots are scored)
1. **T-C7a and T-C7c on existing models** (CPU, $0): do re-anchored checkpoints behave faithfully under edits, and are silent
   thoughts editable along the model's own embeddings?
2. **Set-valued checkpoints at inference** (CPU): restart one episode from a *set* of start tokens. This needs the model
   to accept several START tokens, which it never saw — test whether it generalises, and if not, a short fine-tune.
3. **A CLR toy pilot** (GPU, ≈ $2–4): segments of L = 3 (Pilot J's chunk reset plus isolation), checkpoint head trained
   by self-distillation from the attention-decoded frontier, fixed K first, then learned halting.
4. **Hand the design to Experiments** for PROTOCOL8's format once the toy shows C2 + C6 + C7a together.

## 34. T-C7c — are silent thoughts faithfully editable? Edit and round-trip tests on the pointer (pre-registered 2026-09-30 00:10 EDT, before any measurement on the 59 test models)

**Question (NORTH_STAR open question 1; C7 for latent steps).** The readable content of a thought is its pointer: the
node whose out-edge the sweep head reads next (§1). Part I found probe directions causally inert. Is the pointer
different? If the thought is re-written to say "I am at node u", does the model then do what BFS from u predicts, and
nothing else?

* **Code.** `theory/exp15_c7_edit.py` (the docstring defines every measure). CPU, no training of the model.
  - *Translation* (thought → text): v_i = the source of the sweep head's top edge at step i + 1, per graph.
  - *Re-encoding* (text → thought): a linear map U from the model's own id table, fitted per model by ridge least squares on
    separate probe graphs (seed 1): h_i ≈ μ_i + U id_emb[v_i]. A second encoder, **emb**, uses the model's node-token
    embedding (the §33 T-C7c wording), scaled by the model's own coefficient.
  - *Edit:* h_i + Enc(u) − Enc(v_i), where u is v_i's look-alike on the unreachable chain (A/B ↔ D/C, same index j).
  - *Control:* a random direction of the same norm.
  - *Round trip:* full substitution h_i → μ_i + U id_emb[v_i]; the residual is discarded.
  - *Test graphs:* the long-chain probe (512, seed 4247). Eligible graphs are those whose unedited pointer is on the
    reachable chain at index j ≤ 8.
* **Models.** The 59 P30 ∪ L models of §26 (exp12 set, each with its §26 sweep head). Primary: edit at thought i = 2 with
  the lin encoder at dose ×2. Secondary: doses ×1 and ×4, i = 3, and the emb encoder.
* **Disclosure.** The measures were shaped on one model outside the test set, the fine-tuned ph_c_s102.
  - The nodes at j and j + 1 are decided by the prefix pass, so answer changes are scored at j + 2 and j + 3.
  - The ridge fit shrinks U (r² 0.23), so ×2 is primary.
  - Its values at i = 2, ×2: read 0.74, follow 0.76, ins +0.88, del −0.55, collateral 0.02, IIA 0.47. Random control: all
    ≤ 0.06. Round-trip agreement 0.98, read_v 0.93. emb edit read 0.00.
* **Predictions** (medians over the 59 models, primary condition):
  - **E1 (read and follow).** M0 ≥ 0.50 and M1 ≥ 0.40. The random control has M0 ≤ 0.05 and M1 ≤ 0.05 in ≥ 90 % of
    models.
  - **E2 (answers follow the edit past the one-pass horizon).** ins ≥ +0.40 and del ≤ −0.30. The random control has
    |ins| ≤ 0.10 and |del| ≤ 0.10.
  - **E3 (locality).** Collateral ≤ 0.05, and within 0.03 of the random control's.
  - **E4 (interchange-intervention accuracy).** IIA ≥ 0.30. Also reported: the share of models with IIA ≥ 0.30.
  - **E5 (round trip).** Answer agreement ≥ 0.95 and read_v ≥ 0.85. The pointer plus the step mean then carries what the
    probe's answers depend on, although it explains only about a fifth of the thought's variance.
  - **E6 (descriptive).** The emb edit has M0 ≤ 0.05: thoughts do not write the pointer in the node-token code. §26's
    restart works because START is an input token, not because thoughts share its code.
* **Readings.**
  - E1–E4 hold: first causal C7 evidence for silent latent steps, on a toy and at the granularity of a single node id.
  - E1 holds but E2 fails: the pointer steers attention but the answers are computed elsewhere (Part I's pattern).
  - E1 fails: the pointer is readable but not linearly writable.
  - E5 fails: the thought carries answer-relevant content beyond the pointer, a hidden channel. That is expected where
    attention covers several branches (§30 B1: 1.4–2.3).
* **Scope.**
  - The translation's vocabulary is node ids, not language.
  - The encoder is fitted on the model's own thoughts, not trained independently of it.
  - Only single-node claims are edited; set-valued claims come with the checkpointed design (§33).
  - The input-only control of NORTH_STAR (c) concerns decoders. It applies to T-C7a and is not part of this edit test.
* **Moderators (descriptive).** IIA by extrapolating vs non-extrapolating models (Mann–Whitney), and by K.

**Result (2026-09-30 01:06 EDT; all 59 models; `theory/score_c7_clamp.py 34` → `theory/results/score_s34.json`; raw
`theory/results/exp15_c7_edit.json`). E2, E3, E4 and E6 met; E1 and E5 narrowly not met.**
* **E1 not met.** The edit is read as written: median M0 = 0.90. It is followed exactly one step later less often:
  M1 = 0.35 < 0.40.
  - Random control: M0 and M1 ≤ 0.05 in 59/59 models.
  - M1 counts only an exact advance at step i + 2. Braking models often repeat a step, so M1 is 0.27 in extrapolating
    models and 0.59 in non-extrapolating ones. The answer measures below are not affected by this.
* **E2 met.** Past the one-pass horizon, answers follow the edited pointer: ins +0.83, del −0.59. Random control:
  |ins| 0.01, |del| 0.08.
* **E3 met.** Collateral flips 0.026, vs 0.015 for the random control.
* **E4 met.** Median IIA 0.36; 32/59 models have IIA ≥ 0.30. At i = 3 the median is 0.48. Dose ×4: 0.45. Dose ×1: 0.09.
* **E5 not met, by 0.003.** The round trip, i.e. full substitution μ_i + U id_emb[v_i], agrees with the unedited run on
  0.947 of answers (read_v 1.00). The pointer plus the step mean carries almost everything the probe's answers depend
  on. The residual 5 % is a small hidden channel.
* **E6 met.** The node-token embedding is not the thought's code: emb edit M0 = 0.006.
* **Moderator.** IIA is 0.30 in extrapolating models (n 25) and 0.39 in non-extrapolating ones (n 34). Mann–Whitney,
  normal approximation, two-sided p = 0.15.
* **Reading.**
  - This is the first causal C7 evidence for silent latent steps on the toy, at single-node granularity. Re-writing
    the thought's pointer, through a linear map from the model's own id table, changes the model's later reads and its
    answers as BFS from the edited node predicts, and leaves the rest alone.
  - Unlike Part I's probe directions, the pointer is readable, writable and used.
  - Limits: IIA is a median 0.36, not near 1. The encoder is fitted on the model's own thoughts. Only single-node claims
    were edited.

## 35. Proposition 10 — confounded stop keys: why anytime runs fall back at the trained depth, and a test of which count they use (theory, and pre-registration of P10e; written 2026-09-30 00:22 EDT, after Pilots D, F and H were scored and Pilot E's pointers measured, before exp16 was run on any test model)

**Observation (Pilot E, E8 arm: K = 8, training depth 6; `theory/results/pilotDE_meas.json`).**
* **Anytime runs fall back at the trained depth, not at K.**
  - Seven competent anytime runs. Six advance at unit speed, and one (s1005) stalls at 2.
  - The in-range maxima are 5, 5, 7, 4, 5 and 6. p = 5 means the pointer reads the edge from depth 5 into depth 6, the
    last edge of a training graph.
  - All six fall back *inside* the trained budget, at h_5–h_8. Examples: s1000 `1 1 2 3 4 5 | 2 3 4 1 2 0`, s1006
    `1 1 2 3 4 5 | 2 2 3 3 2 2`.
  - In 3 of 6 the drop comes right after the last trained edge (p = 5), with two to three steps of budget unused. None
    advances through h_8.
* **Slack controls that crawl pass the trained depth.**
  - s1003 runs `1 2 3 3 1 2 3 4 5 6 7 8`, s1005 runs `1 1 1 1 2 2 2 3 4 5 6 7`, and s1002 also crosses. These are 3/7
    competent controls, with d8 0.67–0.74.
  - Pilot D (K = 6) shows the same fall-back in its anytime runs, right after K (§17 result), where the trained depth and
    K coincide.

**Proposition 10 (confounded stop keys).**
* **Claim.** In training, the true stop condition ("no unvisited out-edge") coincides with every monotone quantity a
  latent loop can carry:
  - the step index. With no step embedding (the P30 recipe) it is still available as the length of the latent history
    that attention sees;
  - the hop count;
  - the amplitude (Mechanism 2).

  A stop or fall-back keyed to any of them costs nothing in training, and only the content key generalises. This
  extends §22's identification condition C1 from "the step count must not be usable" to every count whose training
  range is bounded.
* **Anytime supervision aligns the counts.** It forces hop count = step index up to the trained depth D_max and leaves
  nothing to read afterwards. So a fall-back at D_max is free, and behaviour alone cannot tell which count keys it.
  - This explains E8: the fall-back sits at step ≈ hop ≈ D_max, below K.
  - Slack controls de-align the counts (fractional speed, Prop. 9), which is why some of them pass D_max.
* **Pilot F rules out amplitude as the whole key.** The restoring JAT runs (z ≤ 0 in 6/6) still stop (median d8 0.568).
* **Re-anchoring removes every count at once.** An episode restarted from a node token has step index 1, hop count 0,
  an empty latent history and fresh amplitude, and relative to its anchor no count leaves its trained range. This is
  the principled reason re-anchoring rescued 31/34 models (§26) where blocking, renormalisation (§30) and jitter
  (Pilot F) did not, and the reason for segmenting in the checkpointed design (§33).

**P10e — which count keys the fall-back? (registered here, before `theory/exp16_stop_key.py` is run on any test model)**
* **Interventions** at thought h_2, with the model's own sweep head, on the long-chain probe (512, seed 4247):
  - **Skip.** h_2 + 2(Enc(reach[p+1]) − Enc(reach[p−1])), with §34's lin encoder: the pointer jumps two hops ahead.
  - **Delay.** h_2 is fed again into step 4, so the pointer re-reads one edge and falls one hop behind.
* **Pointer.** ptr_traj's aggregate rule with a fixed head.
* **Measures.** The peak depth (running maximum before the first drop by more than one) and the step at which it is
  reached. See the script's docstring for the exact definitions.
* **Models.** All 17 competent anytime runs with a pointer in `pilotDE_meas.json`: pd_at (10, K = 6) and pe8_at (7,
  K = 8). A model is eligible if its base pointer at h_2 is 1..8.
* **Rule** (medians over eligible models of skip − base; delay reported alongside):
  - **Step key:** peak-step change within ±1 and peak-depth change ≥ +1.
  - **Hop key:** peak-depth change within ±1 and peak-step change ≤ −1.
  - Otherwise: mixed or unresolved.
* **Disclosure.** Two smoke models outside the test set (pb_at_s750 and pb_at_s751, Pilot B anytime runs, same recipe:
  no step embedding, K = 6) gave:

  | Model | Base (depth@step) | Skip | Delay |
  |---|---|---|---|
  | pb_at_s751 | 8@9 | 10@9 | 7@9 |
  | pb_at_s750 | 7@8 | 8@6 | 6@8 |

  - Step-keyed in s751, mostly step-keyed in s750.
  - A per-graph pointer was tried first on s751 and dropped as noise-dominated.
  - My hop-count version of Prop. 10, drafted before the smoke run, is therefore disfavoured.
  - **Expectation: step key.**
* **What each outcome means.**
  - **Step key.** The key is the latent history length, a count that any loop with growing attention history carries,
    and a step embedding is not needed for it. Isolating the history removes it: Pilot J's chunk reset (so PJ1, pointer
    passes 8, should be met in CH3AT), or restarts (§26). Rebalancing losses does not remove it (Pilot H: the brake
    survives fine-tuning).
  - **Hop key.** The count lives in the thought itself. Chunk reset would not remove it; only restarting from a written
    node token does.
  - Either way, restarts remove the key.

**Further predictions (exploratory; stated before Pilots J, K and L are scored).**
* **P10a (Pilot K, RT).** Draws K ≤ 6 never penalise a fall-back after step 6. Expect RT to extrapolate at d7 like AT,
  while d8 stays flat.
* **P10b (Pilot J, CH3AT).** If the key is the step, meaning the history length: CH3AT's pointer passes 8 in more runs
  than AT's (PJ1). If the key is the hop count: it does not.
* **P10c (Pilot L).** Students trained on depth-12 graphs at K′ = 12 move D_max to 12. Expect L-T8 and L-GT to pass 8 on
  the probe: a label-free gain, not extrapolation (§32's caveat).

**One line for a pretrained LM (PROTOCOL8).**
* Anytime or tight-budget training on depth-n problems should give gains up to m = n + 1 (the query hop) and none at
  m ≥ n + 2, for any K′. The loop learns a fall-back keyed to a count that never exceeds its trained range.
* Deeper answers need every count bounded at inference: a chunked latent history, or segments restarted from written
  checkpoints (§33). Alternatively, train at the test depth.

**P10e result (2026-09-30 00:25 EDT; `theory/exp16_stop_key.py` → `theory/results/exp16_stop_key.json`, log alongside; all 17
models eligible).**
* **Verdict by the registered rule: step key.**
  - Skip minus base: median peak-depth change +2, median peak-step change 0.
  - 13/17 models show the step-key signature individually.
  - After skipping ahead, the pointer passes depth 8 in 8/17 models, i.e. well past the trained depth. Examples:
    pd_at_s900 `1 1 4 5 6 7 8 2 …`, pe8_at_s1002 `1 1 4 5 6 7 8 9 10 11 1 0`.
  - So the fall-back is not a hop limit at the trained depth. A thought moved ahead in depth keeps going until its step
    count runs out.
* **Delay, reported alongside: the count is carried in the thought, not in the history length.**
  - Median change: peak depth 0, peak step +1.
  - 10/17 models fall back one step *later* at the *same* depth. 4/17 fall back at the same step, one hop shallower.
  - Feeding h_2 again leaves the history length unchanged, so a history-length count would predict "same step, one hop
    shallower", the 4/17 pattern. The majority instead follows the re-fed thought: its count is one behind.
  - Together with skip, which changed the node identity but not the fall-back step, this says the fall-back is keyed to
    an **iteration count carried in the thought**, separable from the node identity that the lin encoder edits.
  - This reading is post hoc. P10e's rule used skip only, and the delay interpretation was not stated before the data.
* **Revised predictions, stated now, before any Pilot J or Pilot I result has been looked at.** Their runs are still on
  the theory hosts.
  - **P10b′ (Pilot J, CH3AT).** The thought crosses chunk boundaries with its count, so chunk reset does not remove the
    key. The CH3AT pointer falls back near the trained depth like AT, and PJ1's "passes 8" share is not larger than AT's.
    This replaces P10b's conditional.
  - **P10f (Pilot I, PER2/PER3).** A periodic step embedding does not touch a count inside the thought. PI1 is not met.
* **Outcome of P10b′ and P10f (2026-09-30, after Pilots I and J were scored).**
  - P10b′ is not supported: CH3AT passes 8 in 2/4 runs vs AT's 1/4.
  - P10f holds only by the letter: PI1 is 0.57 < 0.60, but PER passes 8 in 3/7 runs vs 0/5 controls.
  - So both the history length and the step clock carry part of the key. The thought is not the only carrier, which
    corrects the post-hoc delay reading above.
* **Across architectures (PROTOCOL9, experiments session, 2026-09-30; updated after the pooled 64-pair replication).**
  The anytime gain at d7 is weak and family-dependent:
  - pooled AP2 18 vs 10 (p 0.09) and AP1 15/26, both inconclusive;
  - discordant pairs: w512 9 vs 0; l4 5/3; l3 3/4; fork 1/3.

  The absence of a d8 gain holds throughout, which is the part Prop. 10 predicts (`docs/PROTOCOL9_RESULTS.md`).
* **Consequence.** The count can be read, and possibly written, without restarting. The per-step mean of the thoughts
  (the lin encoder's μ_i) is common to all graphs and so carries no node identity, which makes it the natural candidate
  for the count. §36 registers an inference-time test of that ("counter clamp").

## 36. Counter clamp — an inference-time fix that holds the thoughts' iteration count in range (pre-registered 2026-09-30 00:29 EDT, before `theory/exp17_counter_clamp.py` is run on any test model)

**Idea (from §35).** The anytime fall-back is keyed to an iteration count carried in the thought, and that count is
separable from node identity. So hold the count in range without restarting.
* Over many graphs the node identities average out, so the batch mean of the thoughts at step i, m_i, carries the
  count.
* **freeze3:** from thought h_4 on, feed h_i − m_i + m_3. Every later step then sees the count of step 3 while keeping
  its own node identity.
* This is the inference-time, in-thought counterpart of Pilot I's periodic clock, and a restart-free counterpart of
  re-anchoring (§26).
* If it works, a latent loop can run past its trained budget with no written checkpoint. This bears on C2, and on
  whether checkpoints are needed for C2 at all or only for C7.

**Code.** `theory/exp17_counter_clamp.py`, CPU, no training.
* m_i is the batch mean over the probe's 512 graphs of the thoughts produced in the same rollout. It is transductive:
  a deployable version would use means fitted on training graphs, with the in-range targets fitted ahead of time.
* Variants: freeze2, freeze3, freeze4, and cycle (2, 3, 4, 2, …).
* Control rand3: the same per-step shift norm as freeze3, in a fixed random direction.
* Outcomes on the long-chain probe at K′ = 12: bal-acc by distance, 7-hop success (bal-acc(7) ≥ 0.60, §26's
  criterion), d8, and the pointer's peak depth (exp16's rule).

**Models.**
* **Group A:** the 17 competent anytime runs of Pilots D and E8 (pd_at, pe8_at).
* **Group B:** the 59 P30 ∪ L models of §26 (exp12 set). 34 of them do not extrapolate, and re-anchoring rescues 31 of
  those 34 (§26).

**Primary: freeze3.** The other variants are secondary.

**Disclosure: four smoke models outside both groups** (d7/d8, peak depth):

| Model | Base | freeze3 | rand3 | Notes |
|---|---|---|---|---|
| pb_at_s751 (anytime) | 0.68/0.56, 8 | 0.69/0.74, 11 | 0.61/0.52, 7 | the pointer runs to 11 under every clamp variant |
| pb_at_s750 (anytime) | 0.63/0.60, 7 | 0.67/0.61, 5 | — | little change |
| pa_c_s703 (fixed-K control) | 0.67/0.61, 6 | 0.61/0.62, 9 | — | the pointer runs `1 1 2 1 2 3 4 5 6 7 8 9`; freeze2 and cycle hurt |
| pa_c_s701 | pointer stuck at 1 | unchanged | unchanged | degenerate |

freeze3 was chosen as primary because it was the only variant that did not hurt any smoke model.

**Predictions.**
* **CC1 (group A, primary).** freeze3 raises median d8 by ≥ 0.05 over base, and median peak depth by ≥ 2. rand3 changes
  median d8 by ≤ 0.02.
* **CC2 (group B, the 34 non-extrapolating models).** freeze3 gives 7-hop success in ≥ 8/34; rand3 in ≤ 3/34.
  Re-anchoring's 31/34 is the reference.
* **CC3 (in-range cost).** In both groups, the median drop in mean bal-acc over distances 1..K is ≤ 0.02 under freeze3.
  The clamp starts inside the budget, at h_4.
* **Descriptive.** The same numbers for the 25 extrapolating models of group B and for every variant. The per-model
  pointer under freeze3 is compared with base.

**Readings.**
* **CC1 and CC2 hold.** The count, not the content, limits continuation, and it can be held in range without a
  restart. That would be the first restart-free C2 fix on the toy. The next step is a trained version: a counter-free
  thought, e.g. a loss that makes m_i constant across steps.
* **CC1 holds, CC2 fails.** Fixed-K models brake on the step schedule (Prop. 9), not only on the count. Clamping the
  count cannot restore a speed their loop never had in range.
* **Both fail.** The batch mean is not the count, or the count is not linearly separable. Restarts (§26, §33) remain
  the only working route.

**Result (2026-09-30 01:06 EDT; 76 models; `theory/score_c7_clamp.py 36` → `theory/results/score_s36.json`; raw
`theory/results/exp17_counter_clamp.json`). CC2 met; CC1 and CC3 not met.**
* **CC1 not met (group A, anytime, n 17).** Two readings of the rule:
  - The scorer's rule, the median of per-model differences: d8 +0.028, peak depth +1; rand3 d8 −0.011.
  - Read literally as a difference of medians (Audit): d8 +0.049, peak +2.
  - CC1 fails either way on d8. Under the literal reading it misses by 0.001.
* **CC2 met (group B, 34 non-extrapolating).** 7-hop success: freeze3 9/34, base 1/34, rand3 0/34. freeze2 gives
  12/34. Re-anchoring gives 31/34.
* **CC3 not met.** Median in-range cost: 0.023 (A) and 0.045 (B).
* **Descriptive.** Any shift of this size from h_4 hurts models that already extrapolate. 7-hop success goes from 23/25
  to 10/25 under freeze3 and to 7/25 under rand3, so the damage is non-specific (Audit).
* **Reading.**
  - The per-step mean carries part of what stops the loop. Clamping it rescues some failures, which a random shift of
    equal size never does.
  - The count-specific evidence is only this rescue (9/34 vs 0/34). The in-range cost and the damage to working models
    are non-specific effects of shifting the thought.
  - Restarts remain far ahead (31/34), which supports the checkpointed design (§33) over clamping.
  - A trained, count-free thought (PROPOSALS §10, T12) is the remaining restart-free route.

## 37. Do Pilot L's students extrapolate past their own training depth? (exploratory predictions registered 2026-09-30 01:16 EDT, before `theory/exp18_deep_reanchor.py` is run on the remaining 57 Pilot L models)

**Setup.**
* Probe: a 24-hop chain vs a look-alike decoy (`theory/deep_chain.py`, R = 4, widths 2–3, 512 graphs, seed 7).
* Arms: one loop at K′ = 24 (single24), K′ = 12 (single12), and re-anchored with 12-step episodes (re12top3).
* Outcome: mean bal-acc over distances 13–18. The students were trained on chains of ≤ 12 hops at K′ = 12.

**Already seen, disclosed.** Mean bal-acc at d13–18:

| Model | single24 | re12top3 |
|---|---|---|
| pl_gt_s111 | 0.89 | 0.55 |
| pl_t8_s111 | 0.50 | 0.50 |
| pl_t8_s126 | 0.51 | 0.65 |
| p30_s111 | 0.50 | 0.49 |

**Why a GT student might extrapolate (Prop. 10).**
* The GT student was trained with tight budgets at two depths: 6 in the normal batch, 12 in the deep batch. So a stop
  keyed to count 6 is penalised.
* The T8 student's labels stop where its teacher stops (d10 ≈ 0.54). That teaches a stop at about 10.

**Predictions** (counts over 20 models per arm; a model "extrapolates" if its mean bal-acc at d13–18 is ≥ 0.60):
* **X1.** GT students, single24: ≥ 10/20.
* **X2.** T8 students, single24: ≤ 5/20. T8 students, re12top3: ≥ 5/20 (the next rung of a label-free ladder).
* **X3.** Self students, single24: 0/20.

**Result (2026-09-30 01:36 EDT; all 60 students; raw `theory/results/exp18_deep_reanchor.json`, log alongside). X1, X2 and X3 all
hold.**
* **X1.** GT students extrapolate in one loop in 17/20 (median d13–18 = 0.74). Median bal-acc stays ≥ 0.59 out to d23:
  `0.81 … 0.80 (d10) 0.77 (d13) 0.73 (d16) 0.69 (d19) 0.62 (d22) 0.52 (d24)`.
* **X2.** T8 students in one loop: 2/20 (median 0.50; their profile falls to chance at d13). Re-anchored with 12-step
  episodes: 14/20 (median 0.63).
* **X3.** Self students: 0/20 in one loop, and 0/20 re-anchored.
* **Reading.**
  - **Training at a second, deeper tight budget makes one loop extrapolate to about twice its training depth.** The GT
    students were trained on 12-hop chains at K′ = 12 plus 6-hop graphs at K = 6. They run to 24 hops at K′ = 24.
  - This is the first single-loop C2 result past training depth in this project. It matches Prop. 10: when the loop
    must continue past its first natural count (6), no count stays a free stop key, and a content-keyed continuation is
    what is learned.
  - Caveat: X1 used true labels on deeper chains, so it is not C4.
  - **The label-free route works one rung at a time.** T8 students do not extrapolate in one loop, because they inherit
    their teacher's stop at about 10. But they re-anchor to 13–18 in 14/20. So an iterated ladder (distil, re-anchor,
    distil again) can extend depth without labels at every rung. Whether a second distillation also yields one-loop
    extrapolation like GT's is the natural pilot (§38, to be registered).
* **Descriptive, post hoc** (2026-09-30, `theory/step_decode.py` → `theory/results/step_decode.json`): a fresh probe
  reads the step index from the students' thoughts at median accuracy 0.94 (GT), 0.98 (T8) and 0.98 (self). Chance is
  0.17. The 17 GT students that extrapolate are at 0.94.
  - So a one-loop extrapolator still carries the count. What changed is that its stop is not keyed to the count.
  - This refines Prop. 10: a decodable count is harmless unless the stop uses it.
  - For T12 (§39) it means PT1, "count removed", may hold without being necessary. PT2 is the test that matters.

## 38. Pilot M — a second rung of the label-free ladder (pre-registered 2026-09-30 01:40 EDT, before any Pilot M run started)

**Idea.** Round 2 of Pilot L (§32).
* The teacher is each round-1 T8 student, re-anchored with 12-step episodes. §37 found it reaches d13–18 in 14/20.
* It is distilled into one 24-step loop.
* No labels are used at either rung.
* Prior work (Literature, `docs/LIT_PILOT_M_2026-09-30.md`, e1e93f1):
  - the closest is expert iteration (1705.08439);
  - no iterated label-free ladder on latent or looped models, and no restart-based amplifier, was found;
  - the known failure mode is the "error avalanche" (2502.01612).

**Code.** `theory/distill_reanchor.py` with `--deep_R 4 --deep_widths 2,3 --student_Kp 24 --teacher_ep 12`.
* The round-1 defaults reproduce Pilot L's trainer bit for bit (checked: max |Δw| = 0).
* Everything else is as §32: 3000 steps, lr 1e-4, bs 256, a normal batch every step, fresh AdamW.

**Arms.** All start from the round-1 student pl_t8_sX, with its §37 sweep head.
* **M-T8:** the round-2 re-anchoring teacher. 20 models.
* **M-self:** the student's own one-episode labels at K′ = 24. 20 models.
* **M-T8F:** T8 with Literature's agreement filter; loss only where top-1 and top-3 re-anchoring agree. 10 models: the
  even-indexed ones in sorted seed order.
* **M-GT:** true labels, as the ceiling. The other 10 models.
* 60 runs.

**Outcomes.**
* **Deep measure:** mean bal-acc over d13–18, one loop at K′ = 24, on §37's probe (seed 7, never used in training).
  Measured by `theory/exp18_deep_reanchor.py` after fetch.
* Plus eval_arch's gate, d8 at K′ = 12, and delayed bal-acc.

**Predictions.**
* **PM1 (primary: the teacher matters).** §23 pair rule on the deep measure: M-T8 > M-self.
* **PM2 (label-free one-loop reach to 18).** M-T8 is ≥ 0.60 in ≥ 10/20. The round-1 students were 2/20.
* **PM3 (no avalanche at the old rung).** Median d8 of M-T8 ≥ 0.608, i.e. the round-1 median 0.638 − 0.03.
* **PM4 (no in-distribution cost).** Median delayed drop vs the round-1 student ≤ 0.02.
* **Descriptive.**
  - T8F vs T8 on the same 10 models.
  - The GT ceiling.
  - d19–24 at K′ = 24.
  - d25–30 on an R = 5, width-1 probe at K′ = 32.

**Where and cost.** 20 new Vast hosts (theory-M01..M20), 3 runs each, destroyed as each queue ends. About 2 h wall
time and about $7. That is over the earlier theory plan, so the user is told.

**Result (2026-09-30; all 60 runs; `theory/score_pilotM.py` → `theory/results/score_pilotM.json`, deep measure
`theory/results/exp18_pilotM.json`; committed with this text). PM1 supported and PM4 met; PM2 narrowly and PM3 not
met.**
* **PM1 supported.** Deep measure: M-T8 > M-self in 10/12 counted pairs (one-sided p = 0.019).
* **PM2 not met.** M-T8 reaches d13–18 in one loop in 9/20 (≥ 10 needed). The round-1 students did so in 2/20.
* **PM3 not met.** Median d8 of M-T8 is 0.595 < 0.608: the old rung eroded. Competence fell as well: M-T8 13/20, vs
  M-self 17/20.
* **PM4 met.** Median delayed drop 0.011.
* **Descriptive.**

  | Arm | Deep median | ≥ 0.60 | d8 | Competent |
  |---|---|---|---|---|
  | M-T8F (filtered) | 0.62 | 6/10 | 0.640 | 8/10 |
  | M-T8 | 0.56 | 9/20 | 0.595 | 13/20 |
  | M-GT | 0.72 | 6/10 | — | 7/10 |

  - T8F > T8 in 5/7 counted pairs (inconclusive).
  - M-GT at d25–30 with K′ = 32: 0.98, past its 24-hop training depth.
* **Reading.**
  - The ladder climbs a second rung without labels: the teacher beats self-labels, and one-loop reach goes from 2/20 to
    9/20.
  - But unfiltered labels erode the rung below. This is the error avalanche of 2502.01612.
  - Literature's agreement filter keeps the old rung (d8 0.640) and climbs further (6/10), on 10 models only.
  - PM3's failure, together with lower competence (13/20 vs 17/20), is the early sign of self-training error build-up
    that 2502.01612 warns about. The filtered arm (8/10 competent, d8 0.640) is the descriptive hint that filtering
    contains it.
  - **The agreement filter is the default for any third rung.** It should also mix in the previous rung's teacher
    data, to hold the lower depths.

**Amendment M-2 (2026-09-30 04:54 EDT): does the filter contain the avalanche? Registered before any M-2 run.** Pilot M's hint
(T8F vs T8) rests on 10 models.
* **Runs.** M-T8F on the other 10 models (the odd-indexed ones, which had M-GT), exactly as §38. The only difference
  is the hardware: the local RTX 2070 in fp32, where M ran in bf16 on Vast. 10 runs.
* **Predictions** over all 20 models, with M-T8F paired with M-T8 on the same model:
  - **M2a:** §23 pair rule on the deep measure, T8F > T8.
  - **M2b (no avalanche):** median d8 of T8F ≥ 0.608 (PM3's bar).
  - **M2c:** T8F reaches ≥ 0.60 on the deep measure in ≥ 10/20 (PM2's bar).
  - **M2d:** competent T8F ≥ competent T8 (13/20).

**M-2 result (2026-09-30; the 10 new runs finished 08:51; `theory/score_pilotM2.py` →
`theory/results/score_pilotM2.json`). M2a supported; M2b and M2c met; M2d narrowly not met.**
* **M2a.** T8F > T8 on the deep measure in 9/11 counted pairs (one-sided p = 0.033).
* **M2b.** Median d8 of T8F over 20 models is 0.624 (≥ 0.608): the old rung holds.
* **M2c.** T8F reaches d13–18 in one loop in 12/20 (PM2's bar was 10).
* **M2d.** Competent T8F 12/20 vs T8 13/20.
* **Reading.**
  - The agreement filter contains the error avalanche. With it, the label-free ladder climbs a second rung without
    losing the first. After two rungs, 12/20 models answer at 13–18 hops in one loop, trained only on their own
    re-anchored inference.
  - Caveats:
    - The half of the models run in this amendment used fp32 on a different card. The fp32 half reaches 6/10, the same
      as the bf16 half's 6/10 (Audit), so precision does not drive the result.
    - Competence is not better (12 vs 13).
    - The deep graphs still come from the evaluation generator (§32's caveat).


## 39. Pilot T12 — step-adversarial thoughts (PROPOSALS §10) (pre-registered 2026-09-30 03:22 EDT, before any T12 run started)

**Idea (§35, Prop. 10).** The anytime fall-back is keyed to counts the loop carries. T12 removes the count from the
thought during training: a probe learns to read the step index k from every thought h_k, and the model is trained to
make that probe's output uniform.
* It is the domain-confusion objective, with weight λ = 0.5.
* The probe trains on detached thoughts. The model's term goes through a frozen copy of the probe.
* Frontier expansion needs no count, so the task stays solvable.

**Code.** `theory/arch.py:add_step_adversary`, `theory/train_arch.py --step_adv`.
* The probe is not a registered submodule, so the state dict is unchanged.
* With the flag off, training is bit-identical to the previous commit (checked on CPU: max |Δw| = 0).
* Disclosure: plain gradient reversal was tried first in a CPU smoke run and dropped. It drove the probe above chance
  (CE 2.5 > ln 7), i.e. the probe became confidently wrong rather than the count being removed.
* With the confusion form, the smoke model's probe CE sits at ln 6 and post-hoc decode accuracy is at chance. That
  model had not yet learned the task.
* The CUDA-graph path could not be tested locally (GPUs busy). The first GPU run's log must show finite
  loss_step_probe before any result is read.

**Arm.** AT + T12: the P30 recipe (theory/queue_arch.sh BASE) plus `--anytime 1 --step_adv 0.5`, seeds 900–907, paired
with Pilot D's pd_at_s900–907. 8 runs.

**Measures.**
* eval_arch: gate, d7, d8, extrap.
* Pointer: le_pointer.py.
* Post-hoc step decodability: `theory/step_decode.py`. A fresh 2-layer probe on the frozen thoughts, held-out
  accuracy; chance is 1/K.

**Predictions.**
* **PT1 (manipulation).** Step-decode accuracy is lower in T12 than in the same-seed AT run in ≥ 75 % of pairs with
  both runs competent (≥ 5 counted, else unresolved).
* **PT2 (primary, C2).** The pointer passes 8 (max p ≥ 8) in ≥ 50 % of competent T12 runs, and in a larger share than
  the same seeds' competent AT runs.
* **PT3.** §23 pair rule on d8: T12 > AT.
* **PT4 (learnability).** Competent T12 runs ≥ competent AT runs − 2 (AT: 6/8).
* **Reading.**
  - If PT1 holds and PT2 fails: removing the thought's count is not enough, because history and clock also carry it
    (§35's I/J outcome).
  - If PT1 fails: the adversary did not remove the count.

**Where.** The experiments session runs it on 1–2 Vast hosts (≈ $1) from a frozen copy of this commit, with its own
worker.

**Amendment T12-1 (2026-09-30 03:49 EDT, before any T12 run produced a model). A graph-capture fix only; the design and
predictions are unchanged.**
* All first attempts on the experiments session's hosts crashed at CUDA-graph capture, at step 0, so they are not
  outcomes. Cause: the adversary indexed log-probabilities with a Python list, which copies an index tensor host →
  device during capture.
* Fix: the same computation as a contiguous slice (`theory/arch.py`).
* GPU check on a rented A4000 (theory-T12smoke, destroyed after). Graph capture works. Over 2000 steps:
  - with a negligible adversary weight (1e-6), the probe learns the step index: CE 1.37 → 0.16;
  - with the registered λ = 0.5, it stays at chance (1.78 ≈ ln 6).
* Disclosed observation: the task loss at step 2000 is higher with the adversary (≈ 1.5 vs ≈ 0.8). λ is kept as
  registered, and PT4 measures the learnability cost.

**Baseline, measured before any T12 run finished** (2026-09-30; `theory/results/step_decode.json`). A fresh probe reads
the step index from the thoughts with accuracy 0.87–1.00 in pd_at_s900–907 (7 runs with models) and 0.77–1.00 in
pd_c_s900–907; chance is 0.17. This is direct evidence that the thought carries a count (Prop. 10), and it sets PT1's
bar.

**Result (2026-09-30; 8 runs from the experiments session, frozen at 6a9784c; `theory/score_pilotT12.py` →
`theory/results/score_pilotT12.json`). PT4 fails; PT1 and PT3 unresolved; PT2 not met.**
* **Competent: 2/8** (s900, s905). Three runs exhausted their stage-1 guard (s901, s906, s907). Three finished below
  the gate (s902, s903, s904). AT is 6/8, so **PT4 fails**: the adversary at λ = 0.5 breaks learnability.
* **PT1 and PT3 unresolved:** 1 counted pair each.
* **PT2 not met:** the pointer passes 8 in 0/2 competent runs.
* **Descriptive.**
  - s900 kept both a pointer and the count: `1 2 3 4 5 6 | 0 …`, step decode 0.86. It falls back after K like AT: d7
    0.63, d8 0.51.
  - s905 had the count largely removed (step decode 0.30). It has no readable pointer and does not extrapolate (d7
    0.54).
  - The four non-s900 models show no pointer by ptr_traj's rule at all.
* **Reading.**
  - Removing the count from the thought is neither sufficient (s905) nor, per §37, necessary: the GT students keep it
    at 0.94 and still extrapolate.
  - The count is harmless unless the stop is keyed to it. The fixes that worked change what the stop is keyed to:
    training at a second depth (§37), chunked history (§29), or restarts (§26, §32). They do not delete the count.
  - T12 in this form is not pursued further (PROPOSALS §10 updated).

## 40. Prop. 10 across architectures: pointers and counts in the PROTOCOL9 runs (exploratory predictions registered 2026-09-30 07:47 EDT, before any PROTOCOL9 model was measured by this session)

**Models.** The competent runs in `results/p9.json` (experiments session; scored 07:45): anytime p9at_* and their
paired controls t9l3/t9l4/t9w512/t10fork.

**Measures** (frozen tools):
* the pointer: `theory/le_pointer.py`, on probe(512);
* step decodability: `theory/step_decode.py`.
The fork family is probed on chain graphs, i.e. off its training shape, and is reported separately.

**Predictions** (non-fork families):
* **X9a (the in-range brake goes).** Median p[K−1] of competent anytime runs ≥ K − 2, and above the competent
  controls' median. This is the LE2 pattern.
* **X9b (the fall-back remains).** The pointer passes 8 in ≤ 30 % of competent anytime runs. This is the d7-not-d8
  pattern.
* **X9c (the count is carried).** Step-decode accuracy ≥ 0.70 in ≥ 80 % of competent runs of both arms.

**Result (2026-09-30; 33 competent models; `theory/results/le_pointer.json`, `theory/results/step_decode.json`, logs
`*_p9.log`). X9a, X9b and X9c all hold.**
* **X9a.** Median p[K−1]: anytime 5 (n 16) vs controls 2 (n 9).
* **X9b.** The pointer passes 8 in 1/16 anytime runs.
* **X9c.** Step decode ≥ 0.70 in 25/25 runs (medians 0.96 and 0.99).
* **Fork family (descriptive).** Anytime 5 vs 4; passes 8 in 2/5 and 1/3; count decoded in 8/8.
* **Reading.** The mechanism of §17 and §35 holds on 3- and 4-layer models and on 512-wide models. Anytime removes the
  in-range brake, the fall-back after the trained depth remains, and the thoughts carry a count.
* Measurement note: the first attempt skipped every model silently, because CRLF in a path list left a trailing 
 on
  each path. It was re-run with clean paths.

## 41. Why a second training depth gives length generalisation: content key or moved count? (pre-registered 2026-09-30 09:04 EDT, before any run)

**Question (§37 X1).** Students distilled with true labels on 12-hop chains at K′ = 12 extrapolate in one loop to
about 24 hops. Two accounts:
* **Content key.** Training past the old stop teaches continuation keyed to content (there is an edge to read). Reach
  is then unbounded and only fades.
* **Moved count.** The stop key moves to the new training depth.

**Test.** Supervise the deep chains only up to distance 9, with a tight 9-step loop.
* Flags: `--teacher gt --student_Kp 9 --deep_sup_max 9`. Look-alikes are masked at the same indices, and there is a
  normal batch as before.
* Everything else is as §32. The flag-off path is bit-identical to Pilot L's (checked).
* A stop keyed to count 9 is then free.

**Models.** The 10 even-indexed P30 failures: 102, 111, 113, 118, 125, 130, 132, 140, 146, 157. Local RTX 2070, fp32.

**Measure.** exp18: one loop at K′ = 24 on §37's probe, bal-acc by distance.

**Rule** (medians over the 10):
* **Moved count** if median d10–12 < 0.60.
* **Content key** if median d13–18 ≥ 0.60.
* Otherwise partial.

**Reference, descriptive.** The same models' GT12 students (pl_gt), paired.

**Result (2026-09-30; 10 runs finished 10:49; `theory/score_s41.py` → `theory/results/score_s41.json`, raw
`theory/results/exp18_pilotN.json`). Partial by the registered rule.**
* Median d10–12 = 0.67 (not < 0.60), so the count did not simply move to 9.
* Median d13–18 = 0.53 (< 0.60), so continuation is not unbounded.
* Median profile: d9 0.71, d10 0.69, d11 0.67, d12 0.64, d13 0.60, d14 0.55, d15 0.52, then chance.
* Reference: the same models' GT12 students keep 0.70 at d13–18 and 0.59 at d23.
* **Reading (hypothesis, not tested).** Reach past the supervised depth is graded and grows with it: training to 9 →
  about 13–14 hops; training to 12 → about 22. This looks like a per-hop fade whose rate depends on the depth trained
  (Mechanism 2), not a stop key. Deeper training makes the transition more restoring, so the pointer survives more
  hops. It would explain why a second depth helps even though the count stays decodable (§37). A test: transfer
  curves (exp9's z) for GT9 vs GT12 students.
* **Registered test of the reading (2026-09-30 10:52 EDT, before measuring).**
  - z (`theory/transfer_pairs.py`) is lower for the GT12 student than for the same model's GT9 student in ≥ 75 % of
    the 10 pairs (§23 pair rule).
  - Descriptive: z vs the one-loop reach, i.e. the last distance with bal-acc ≥ 0.60.
  - **Result: inconclusive.** z(GT12) < z(GT9) in 7/10 pairs (0.70 < 0.75; one-sided sign p = 0.17). Median z is
    −0.02 vs 0.05. Across all 20 students, z vs reach gives Spearman −0.36: weakly in the predicted direction. The fade
    reading stays a hypothesis. Raw data: `theory/results/transfer_pilots.json`, log `transfer_s41.log`.

**Amendment 41-2 (2026-09-30 11:02 EDT, before any run): does reach keep growing with training depth?**
* **Runs.** GT distillation on 18-hop chains (`--deep_R 3 --deep_widths 2,3,4 --student_Kp 18`) for 6 of §41's models
  (102, 111, 113, 118, 125, 130), otherwise as §32. Local 2070, fp32, in the gap before PROTOCOL8b.
* **Prediction.** Reach grows with training depth (9 → ≈ 13–14, 12 → ≈ 22). So the GT18 median one-loop bal-acc at
  d19–24 (K′ = 24, §37's probe) is ≥ 0.60. The same models' GT12 students give the reference.
* **Descriptive.** d25–30 at K′ = 32 on the 30-hop width-1 probe (exp18 --r5).
* **Result (2026-09-30; both workers ended 12:06; `theory/score_s41b.py` → `theory/results/score_s41b.json`, raw
  `theory/results/exp18_pilotN18.json`). Met.**
  - All 6 ran, none cut: 5 on the 2070 (fp32) and s130 on the 3070 (bf16), which was added once it became idle.
  - GT18 median one-loop bal-acc at d19–24 is 0.686 (≥ 0.60). The same models give 0.627 as GT12 and 0.511 as GT9.
  - The median profile is flat at ≈ 0.72 from d7 to d23, then 0.56 at d24, which is the budget edge at K′ = 24.
  - d25–30 at K′ = 32 on the 30-hop probe: 0.88. That probe is width-1 and easier than §37's, so this number is not
    comparable with 0.686 (Audit).
  - **Reading.** The fade shrinks as the trained depth grows. Depth 9 fades out by about 15; depth 12 fades
    gradually; depth 18 shows no fade out to 30 hops. A loop trained on long enough chains, with tight budgets,
    continues at a constant accuracy past its training depth. That is single-loop C2 with final-answer labels on
    deeper data (C4 holds, but the deep data are from the evaluation generator).

## 42. Predictions sent for PROTOCOL8's revised primary (in-distribution deep depths) (2026-09-30 09:34 EDT, before any PROTOCOL8 confirmatory data)

Context: Phase 0 outcome (iii), from the experiments session. GPT-2 Coconut on deep ProsQA learns about 5–6 hops in its
latents. Accuracy saturates by K′ ≈ 3, and depths ≥ 10 are at chance in distribution.

**Predictions** (slack/brake account, §§17, 25, 35, 40):
* At K′ = 10, anytime > Coconut in mean accuracy over answer depths 7–10, and the gain grows with depth. Depths 3–4 are
  unchanged.
* Random-tight: the same direction, with a gain ≤ anytime's and more seed variance.
* Signature: for depths ≥ 7, the K′ at which accuracy saturates moves from ≈ 3 (Coconut) to ≥ 6 (anytime).
* Falsified if anytime does not lift depths 7–10 and saturation stays at K′ ≈ 3. The limit would then be per-thought
  capacity, not a brake.

**Outcome (2026-10-01; PROTOCOL8 scored by the experiments session, `docs/PROTOCOL8_RESULTS.md`, `results/p8.json`).
Falsified, more strongly than the stated falsifier.**
* **Depths 7–10.** Anytime is *below* Coconut (0.50 vs 0.56; mean difference −0.067), and random-tight likewise
  (−0.060).
* **Saturation.** K′ moves to 0, not ≥ 6: at and rt answer without using their latents (latent use LU 0.07–0.12 vs
  Coconut's 0.14–0.18). D11–12 are at chance for every latent arm, and no latent seed is competent (0/12).
* **Reading.** The toy's slack/brake account does not transfer to GPT-2 Coconut on ProsQA.
  - The per-thought-capacity alternative named in the falsifier is not shown either: the anytime arms stop using their
    thoughts altogether.
  - Coconut's 0.56 at D7–10 may partly rest on ProsQA's proximity cue (§57). The p8_na secondary will tell.
  - **It did** (PROTOCOL8 amendment 3, 10cc74c). On cue-free p8_na_test, Coconut's D7–10 falls 0.564 → 0.508 (chance);
    anytime is 0.486 and random-tight 0.501. Text CoT is unaffected (0.762 → 0.767). Coconut's only edge on deep ProsQA
    was the cue.
* **Reference for C3.** Text CoT reaches 0.71–0.81 at every depth, including D11–12, with ~84 tokens: the bar any
  latent scheme must meet.

## 43. Where is the deep frontier read in 3- and 4-layer models? (exploratory, registered 2026-09-30 10:58 EDT, before measuring; prompted by the experiments session's T9 secondary)

**Background.** Blocking chain A at the in_peak-chosen layer fails the positive control in several T9 models
(PROTOCOL7_RESULTS).

**Measure.**
* `theory/rollout.py` gained an opt-in `record_layers` option. It is off by default, and the default output was checked
  bit-identical.
* Script: `theory/layer_sweep.py`.
* Per layer l and head h, sweep strength S[l, h] = the sum over steps 1..K and chain edges of the positive part of the
  mean attention excess on the reachable vs the look-alike chain. This is ptr_traj's head-selection statistic, applied
  per layer.
* Per model: the argmax layer, and the top layer's share of Σ_l max_h S[l, h].

**Models.** All 24 T9 models of `results/p7s_causal2_t9.json`. The positive control is recomputed as the mean drop of
A4–A6 ≥ 0.20 at the model's chosen layer: 8 fail.

**Predictions.**
* **L1.** In failing models, the argmax layer differs from the chosen layer, or the top-layer share is < 0.5, in ≥ 60 %
  of models.
* **L2.** In passing models the same holds in ≤ 30 %.

**Result (2026-09-30, CPU; `theory/results/layer_sweep.json`). L1 not met (4/8 = 0.50 < 0.60); L2 met (3/16 = 0.19).**
* **Denominators (Audit).** My 24 is every T9 model in `results/p7s_causal2_t9.json`, including 3 below the P7 floor.
  Within the experiments session's primary 21, the same rule gives 6/21 failing, so the two counts agree.
* **4 failing models:** the sweep is strongest in layer 1, or spread (top share < 0.5), while in_peak chose layer 2 or
  3. These are t9l3_s510 (weak sweep everywhere, max S 0.06), t9l4_s523, t9l4_s526 and t9w512_s548. Blocking at layer
  1 might pass the control there.
* **The other 4** (t9l3_s502, t9l3_s512, t9l4_s522, t9l4_s534): the sweep is concentrated in the blocked layer, yet
  blocking it does not cut A4–A6.
* **Reading.** The positive-control failures are half a layer-choice problem and half redundancy. In the latter, the
  in-range deep nodes are also answered by another route, e.g. a query-time read (T1c holds in 13/13). The T1b/T2b
  nulls in T9 are therefore uninterpretable where the control fails, as the experiments session concluded.

## 44. Set-valued checkpoints zero-shot: can one episode restart from several written anchors? (pre-registered 2026-09-30 12:13 EDT, before measuring the 59 models; NORTH_STAR open question 2)

**Test.** `theory/exp19_set_restart.py`, CPU, the 59 P30 ∪ L models of §26.
* After episode 1, take the top-3 attended anchors (§26).
* **top1:** restart one episode from a1.
* **top3max:** three separate episodes, one per anchor (§26's re2top3).
* **set3:** ONE episode whose prefix ends in three START tokens, (a3, a2, a1).
* **set3r (control):** the same, with a2 and a3 replaced by random off-chain nodes.
* All conditions are max-combined with episode 1. Measure: far8–10 bal-acc on the long-chain probe.

**Disclosure.** Smoke model outside the set, pb_at_s751: top1 0.614, top3max 0.680, set3 0.650, set3r 0.613.

**Predictions** (medians over the 59 models, of per-model differences):
* **S1 (a written set carries information zero-shot).** set3 − top1 ≥ +0.02, and set3r − top1 ≤ +0.005.
* **S2 (one set-episode is almost as good as separate episodes).** top3max − set3 ≤ 0.03.

**Readings.**
* **S1 and S2 hold.** A checkpoint can be a set of node ids, written as START tokens, with no retraining. §33's
  set-valued checkpoint then keeps the parallelism with one episode per segment.
* **S1 holds, S2 fails.** Sets help, but training is needed to use them fully.
* **S1 fails.** Sets need training.

**Result (2026-09-30; 59 models; `theory/results/exp19_set_restart.json`). S1 and S2 not met: sets need training.**
* Medians of per-model differences in far8–10: set3 − top1 = −0.006; set3r − top1 = −0.011; top3max − set3 = +0.064.
* 7-hop success is 55/59 for top1, top3max and set3, and 52/59 for set3r.
* **Reading.** The P30 models' loop starts from the last START and ignores the others. A correct set helps no more
  than a single anchor, and a random one hurts slightly. The smoke model (Pilot B anytime) was not representative.
* For the checkpointed reasoner (§33), set-valued restarts must be trained. The pilot's first stage is therefore to
  teach restarts from sets of START tokens. Separate episodes per anchor (top3max) remain the working inference-time
  version.

## 45. Pilot CLR-1 — training set-valued checkpoints (the checkpointed latent reasoner's first stage) (pre-registered 2026-09-30 12:27 EDT, before any run)

**Why.** §44: trained models ignore extra START tokens zero-shot. Literature (`docs/LIT_SET_RESTART_2026-09-30.md`,
6fcb828) found no precedent for a latent loop restarted from a self-chosen set of node-id tokens with the history cut,
and flagged collapse onto one start token as the known failure.

**Training.** `theory/train_clr.py`.
* Fine-tune each model (fresh AdamW, lr 1e-4, 3000 steps, bs 256) on the normal batch plus a deep batch of 12-hop
  chains.
* The student's deep answer is two segments. Episode 1 runs K steps from START. The student then writes a checkpoint:
  the W targets its own sweep head attends most at step K, chosen without gradient. Episode 2 runs K steps from those
  W START tokens. Answers are max-combined.
* Targets come from the frozen initial model's re-anchored inference (§26 re2top3; no labels).
* Random anchor order during training, a Literature mitigation.

**Arms.** CLR3 (W = 3) and CLR1 (W = 1, same teacher and data). Models: the 10 even-indexed P30 failures (as §41), 20
runs. Both arms of a model run on the same local card, 5 models on the 3070 (bf16) and 5 on the 2070 (fp32).

**Measures** (`theory/exp20_clr_eval.py`; evaluation writes the top-1 anchor last):
* two_seg far8–10 on probe(512);
* chain S = 2..4 segments on §37's 24-hop probe;
* checkpoint edit (C7): replace the written anchors that lie on the reachable chain by their look-alikes. IIA = the
  look-alike's successors at +2, +3 are answered reachable and the true ones are not.
* eval_arch for in-distribution delayed bal-acc.

**Predictions.**
* **PC1 (a written set is used after training).** §23 pair rule on two_seg far8–10: CLR3 > CLR1.
* **PC2 (C2 through checkpoints).** chain4 mean d13–18 ≥ 0.60 in ≥ 5/10 CLR3 models.
* **PC3 (C7 on written checkpoints).** Median edit IIA ≥ 0.30 in CLR3.
* **PC4 (no in-distribution cost).** Median delayed drop vs the initial model ≤ 0.02 in CLR3.

**Disclosure.** The smoke students, trained 3 steps on CPU from p30_s111, gave edit IIA 0.30 (W3) and 0.48 (W1). They
are essentially the untrained model.

**Amendment CLR-1a (2026-09-30 12:31 EDT, before any §45 result was read or evaluated; requested by Audit). A manipulation check:
is the set actually used?** Both measures are computed in `theory/exp20_clr_eval.py` at evaluation time, for CLR3
models only.
* **anchor_maxshare.** At the first latent step after the restart, the attention mass on each of the W START tokens,
  summed over all layers and heads. Per graph, the largest anchor's share of the W-token total, averaged over graphs.
  1/W means even use; 1 means collapse.
* **two_seg_rand.** two_seg with the written set's non-seed anchors replaced by random off-chain nodes (the §44 set3r
  control).
* **Rule.** PC1 is read as "set used" only if both hold:
  - anchor_maxshare < 0.8 in ≥ 6/10 CLR3 models;
  - median (two_seg − two_seg_rand) far8–10 ≥ +0.02.

  Otherwise a PC1 pass is reported as "gain without set use".
* **Smoke disclosure.** The 3-step CPU student gave share 0.51, two_seg 0.542, two_seg_rand 0.544. Spread attention
  with no use is exactly §44's pattern, which is why both measures are required.

**Result (2026-09-30; 20 runs on the local 3070/2070/4060; the queues ended 15:15; evaluation 15:17 →
`theory/results/exp20_clr_eval.json`; `theory/score_s45.py` → `theory/results/score_s45.json`). PC3 and PC4 met; the
set is used (CLR-1a); PC1 inconclusive; PC2 narrowly not met.**
* **All competent:** CLR3 10/10, CLR1 10/10.
* **CLR-1a (manipulation check) passes: the set is used.** Anchor max-share < 0.8 in 10/10 models (median 0.46).
  Replacing the non-seed anchors with random nodes costs far8–10 +0.039 (median).
* **PC1 inconclusive.** two_seg: CLR3 > CLR1 in 6/10 pairs (p = 0.38). Median far8–10: 0.613 vs 0.595. One segment
  alone gives 0.50.
* **PC2 not met.** chain4 d13–18 ≥ 0.60 in 4/10 CLR3 models (median 0.596).
  - CLR1 chains better: median 0.671.
  - Medians by segment count (CLR3 / CLR1): chain2 0.50 / 0.50; chain3 0.53 / 0.55; chain4 0.60 / 0.67.
  - d19–24 stays at ≈ 0.51 for both arms.
* **PC3 met: the written checkpoint is the causal channel (C7).** Swapping the on-chain anchors for their look-alikes
  changes the answers as BFS from the edited checkpoint predicts.
  - Median IIA: CLR3 0.84 (ins +0.86, del −0.89); CLR1 1.00 (ins +1.00, del −1.00).
* **PC4 met.** Median delayed drop 0.0004.
* **Reading.**
  - Training makes restarts from written node-id checkpoints fully faithful: an edit to the written text changes the
    answers as predicted, near IIA 1 for single anchors. That is C7 at the checkpoint level, far above the silent
    thoughts' 0.36 (§34).
  - Chained segments extend reach to d13–18 without labels (C2 through checkpoints; CLR1 median 0.67). Each segment
    adds about 6 hops.
  - **A 3-node set is used, but at this scale it costs reach rather than adding it** (Audit): chain4 median 0.596 for
    the set vs 0.671 for a single anchor; PC1 is inconclusive. A set of 3 anchors does not beat a single anchor. The parallelism gain of sets is
    not realised at this width and training dose; §30's B1 measured the attention's branch coverage at only ≈ 1.4–2.3.
  - Next for the design: single-anchor checkpoints are the working C2 + C7 unit. Sets need either more training or a
    per-anchor coverage target (Literature's mitigation).

**Amendment CLR-1b (2026-09-30 15:24 EDT, before any run): the other 10 failure models.** PC1 was inconclusive and PC2 missed by one
model at n = 10.
* **Runs.** The same design on the 10 odd-indexed P30 failures (109, 112, 115, 119, 126, 131, 135, 145, 147, 158), both
  arms per model on one local card. 20 runs. Placement: 4 models on the 3070, 6 on the 4060. The 2070 was busy at launch, so its 3 models moved
  to the 4060.
* **Scoring.**
  - **Primary: the new 10 alone, as the clean replication**, with the §45 thresholds (PC2 ≥ 5/10; CLR-1a ≥ 6/10).
  - Secondary: the pooled 20 by the same rules (PC2 ≥ 10/20; CLR-1a ≥ 12/20). CLR-1b was registered after the first 10
    were scored, so a pooled verdict is partly determined by results already seen (Audit).

**CLR-1b result (2026-09-30; 20 runs on the 3070/4060, finished 18:49; `theory/score_s45.py b` and `pooled` →
`theory/results/score_s45_b.json`, `_pooled.json`).**

| Test | New 10 (primary) | Pooled 20 (secondary) |
|---|---|---|
| PC1 | not supported, 5/10 pairs | inconclusive, 11/20 |
| PC2 | met, 6/10 (median 0.606) | met, 10/20 |
| PC3 | met, IIA 0.83 (CLR1 1.00) | met, 0.84 |
| PC4 | met, 0.0005 | met |
| CLR-1a | set used: 10/10 < 0.8; rand cost +0.058 | set used |

* The descriptive pattern replicates: CLR1 chains further (chain4 d13–18 median 0.675).
* **Reading (replicated).** After label-free training, written checkpoints are the causal channel, and chains of them
  reach d13–18. A 3-node set is used but adds no reach over a single anchor. Within the scope set by Literature, the C7
  claim is about the trained written channel, not the silent steps.


## 46. Where does a written anchor help most? Toy hop sweep (pre-registered 2026-09-30 17:46 EDT, before measuring; counterpart of PROTOCOL8b's k-sweep)

**Test.** `theory/exp21_hop_sweep.py`, CPU, the 59 P30 ∪ L models.
* Episode 1, plus one restart from an oracle anchor: the reachable chain's node at distance k (k = 1..10).
* Answers max-combined. Measure: bal-acc at distances d = 7..12, indexed by hops remaining r = d − k.

**Predictions** (medians over models, pooling (d, k) cells with the same r):
* **H1 (a one-episode horizon).** Accuracy depends on r, not on d: ≥ 0.80 for r ≤ 4, and ≤ 0.60 for r ≥ 7. One K = 6
  episode plus the query hop covers about 6–7 hops.
* **H2 (anchors near the target help most).** Accuracy is non-increasing in r for r = 1..7, allowing ties within 0.02.

For PROTOCOL8b this would say: write the checkpoint within one episode's reach of the target, i.e. far from the root.

**Result (2026-09-30; 59 models; `theory/results/exp21_hop_sweep.json`). H1 and H2 hold, with a sharp edge.**
* Median accuracy by hops remaining r: r = 1–5: 0.94–0.95; r = 6: 0.91; r = 7: 0.51; r ≥ 8: 0.50.
  - Aggregation: one median over all (model, d, k) cells with the same r, pooled.
  - Audit's per-model mean over cells, then the median over models, gives r 1–6 = 0.95/0.93/0.94/0.94/0.94/0.91,
    r 7 = 0.51, r ≥ 8 = 0.50.
  - The verdicts are the same under both.
* At fixed r it is flat across d = 7..12. Without a restart, d7–12 is 0.50.
* **Reading.**
  - One restarted episode covers exactly its trained 6 hops, at any absolute depth.
  - A checkpoint helps only if it lies within one episode's horizon of the target.
  - So the checkpoint must be written at least every K hops, and a checkpoint more than K hops behind the frontier is
    useless.

## 47. Architecture review: the checkpointed latent reasoner after §§34–46 (2026-09-30 18:07 EDT; requested by the user via Audit; prior art `docs/LIT_CLR_REVIEW_2026-09-30.md`, 57a92dd)

**What the evidence says, in one paragraph.**
* A single latent loop stops where a count it carries says to stop. The count lives in the thought, the history and
  the clock (§35, P10e; I and J).
* Loss-side fixes (anytime, random-tight) remove the in-range brake. They add about one hop, and not reliably across
  architectures (D, K, PROTOCOL9 pooled).
* Removing the count breaks learning (T12). Clamping it helps a little (§36, 9/34).
* What works is to **bound every count by construction**: restart from a written node token (§26, 31/34), or train
  on deeper chains so that the count stops being a free stop key (§41: trained to 18 → flat to 23).
* Both work without step labels:
  - the restarts are the model's own (§26);
  - distilling them fixes 19/20 (L);
  - a filtered second rung reaches d13–18 in one loop in 12/20 (M-2).
* After training, a written checkpoint is the causal channel: edit IIA 1.00 for single anchors, 0.84 for sets (§45).
* A 3-node set inside one episode is used, but reaches less far than a single anchor (0.596 vs 0.671, §45).
* GPT-2 Coconut ignores untrained checkpoints (PROTOCOL8b pilot), as Lanham 2307.13702 would predict. Use has to be
  trained.

### (1) The design that best fits C2 + C4 + C7 now

**A single-anchor checkpointed reasoner, with multi-start beams, trained label-free by self-distillation.**
* **Segments.** Each segment runs L ≤ K latent steps from written START tokens, with the latent history cut. Every
  count therefore stays inside its trained range (C2).
* **Checkpoints.** A segment writes node ids chosen by its own attention (C4, no step labels). The next segment
  restarts from them. After training, the written checkpoint is the causal channel (C7 at checkpoint level, IIA ≈ 1).
* **Parallelism.** Keep the top-W anchors as W separate single-anchor episodes and combine their answers, instead of
  W tokens in one episode. This follows Multiverse 2506.09991 and SoftCoT++ 2505.11484, and costs W× compute.
  - **Update after §48:** beams did not help, since they add false positives, and a single anchor with more segments
    reaches d19–24 (S6 0.65). Recommendation: single anchor, enough segments.
* **Teacher.** The model's own re-anchored inference, with the agreement filter (M-2). The label-free ladder (L → M-2)
  lengthens each segment's reach, so fewer checkpoints are needed.

**Still unproven:**
* Chains past about 18 hops. chain4 d19–24 is ≈ 0.51 in both §45 arms, i.e. errors compound across segments.
* A learned write/halt policy (C5). Anchors come from attention decoding, not a trained head.
* Stability over many segments (C6).
* Transfer to a pretrained LM. That is PROTOCOL8b step 1, which asks whether training makes GPT-2 use checkpoints.
* C3 (vs text CoT). Untested.
* Faithfulness of the silent steps between checkpoints: 0.36 (§34).
* **Novelty boundary (Literature).** Isolation makes the checkpoint the only channel by construction. The claim is
  "training makes the model use the written channel", not "latent reasoning is faithful".

### (2) What changes in §33's design

* **Single vs set.**
  - Default to single-anchor episodes run as a beam of W.
  - Sets-in-one-episode stay a research option. They would need a per-anchor coverage target and a shared position
    id (2405.20671).
* **When to write.**
  - After each segment of L ≤ K steps, i.e. before any count leaves its trained range.
  - §46 makes this sharp: a restarted episode answers 0.91–0.95 up to 6 hops from its anchor and is at chance from 7,
    at any depth. A checkpoint is useful only within one horizon of the frontier, so write every ≤ K hops.
* **How to learn the write policy (C5).** Literature's order:
  1. **v1: a write head distilled from the model's own attention-decoded anchors.** Label-free (C4), and it replaces
     the probe with a trained head.
  2. **v2: halting from closure.** Stop when a new segment's checkpoint adds no node, which is a free label (§22 Prop.
     6). Then a TRM-style correctness head.
  3. **v3: RL on the final answer only**, for write and halt (MEM1 2506.15841, SWITCH 2606.13106).
* **Stability (C6).**
  - Restarts reset amplitude and counts, so drift inside a segment is bounded.
  - The risk moves to anchor errors compounding across segments. Beams and the agreement filter are the controls.
  - Stability is measured as accuracy vs number of segments at fixed hops remaining.

### (3) The cheapest decisive next tests (CPU or local GPU; no Vast)

1. **Beam chaining on existing models (CPU, $0).**
   - Set-up: on the §45 CLR1 students, chain 4–6 segments, keeping the top-W anchors as W separate episodes at each
     segment. Probes: the 24-hop one, and the 30-hop width-1 one for depth only.
   - Decides whether beams fix compounding past 18 hops (C2) and whether beams beat sets (design choice). It also
     gives C6 as accuracy vs segment count.
2. **A learned write head plus a closure halt (local GPU, about 2 h).**
   - Train a small head on the last thought to output node-id scores, distilled from the model's own top-W attended
     anchors. Halting is labelled by closure.
   - Test: the head's anchors chain as well as the attention decode, and the halt fires at closure. This is C5 v1–v2,
     C4-compatible.
3. **External, already registered:** PROTOCOL8b step 1 (experiments session). Does training make GPT-2 use written
   checkpoints? If not, the design does not transfer as is.

**Addendum (2026-09-30 18:28 EDT): architecture edits and other domains are now allowed (user, via Audit).**
* **Architecture change to test first: segment-isolating attention.** Later segments' latent positions see only the
  text and the written checkpoints, never earlier latent states.
  - It is a mask, not new parameters. It bounds every count (§35) and makes the written text the only channel, so C7
    is well posed.
  - For GPT-2 it is sent to the experiments session as a concrete design ("segmented Coconut with written entity
    checkpoints"): single anchor, a write head, closure halt, trained in stages A (checkpoint use) → B (write head by
    self-distillation) → C (re-anchor distillation).
  - The first test there is inference only, on the 8b step-1 checkpoints: chained isolated segments at depths 11–14,
    with a C7 swap.
* **Domain.** The prior is CLUTRR or StepGame next: entity checkpoints transfer directly and depth is controllable. The
  C3 comparison is a text-CoT baseline. Prior art has been requested from Literature, and a choice will be registered
  after its answer.


## 48. Multi-start beams across checkpointed segments (§47 test 1; pre-registered 2026-09-30 18:07 EDT, before measuring)

**Test.** `theory/exp22_beam_chain.py`, CPU.
* Models: the 10 §45 CLR1 students (single-anchor restarts, trained label-free).
* Probe: §37's 24-hop probe.
* Chains of S = 2..6 segments of K steps, each from written single anchors.
* Beam B = 3: each segment runs one episode per anchor, and the next anchors are the 3 distinct most-attended targets
  over all episodes. B = 1 reproduces §45's chain. Answers are max-combined.

**Predictions.**
* **BM1 (beams fix compounding).** Median over models of S4 d19–24: B3 ≥ 0.60, and B3 − B1 ≥ +0.05. §45 B1 was ≈ 0.52.
* **BM2 (no loss nearer in).** Median S4 d13–18: B3 ≥ B1 − 0.02.
* **Descriptive (C6).** Accuracy vs S at fixed distance bands, for both B.

**Result (2026-09-30; 10 models; `theory/results/exp22_beam_chain.json`). BM1 and BM2 not met: beams do not help. But
single-anchor chains keep extending with more segments.**
* S4 d19–24: B1 0.517, B3 0.510 (median per-model difference −0.008).
* S4 d13–18: B1 0.671, B3 0.631. B3 is worse, presumably from the extra false positives of max-combining more episodes.
* Medians by segment count (B1; bands d7–12 / d13–18 / d19–24):

  | Segments | d7–12 | d13–18 | d19–24 |
  |---|---|---|---|
  | S2 | 0.60 | 0.50 | 0.50 |
  | S3 | 0.68 | 0.55 | 0.50 |
  | S4 | 0.67 | 0.67 | 0.52 |
  | S5 | 0.67 | 0.67 | 0.62 |
  | S6 | 0.66 | 0.66 | 0.65 |

* **Reading (C2, C6).**
  - Errors do not compound: each band holds about 0.66–0.67 once reached, over 6 segments.
  - **But near depths erode (C6; Audit).** Median d2–5 is 0.946 at S2, 0.913 at S4 and 0.889 at S6 (worst model: 0.71
    → 0.60). Each extra segment's max-combined answers add false positives near the root. Halting (§49) or a filter on
    new positives would bound this.
  - Reach grows by about 4–5 hops per segment, less than K = 6, and the anchor lands about a hop behind the frontier.
  - §45's "d19–24 ≈ 0.51" was too few segments, not a limit.
  - Multi-start beams add false positives rather than reach. §47's beam recommendation is withdrawn: a single anchor
    with more segments is the working design.

## 49. A learned write head and a closure halt (C5 v1–v2; §47 test 2) (pre-registered 2026-09-30 18:17 EDT, before any run)

**Test.** `theory/exp23_write_halt.py`, CPU, the 10 frozen §45 CLR1 students.
* **Write head.** A linear map from the segment's last thought through the model's id table, trained to predict the
  model's own attention-decoded anchor. This is self-distillation: no labels (C4).
* **Chains.** Up to 6 segments on the 24-hop probe, with head-written anchors vs attention-decoded ones.
* **Closure halt.** On probes with 12-, 18- and 24-hop chains: halt at the first segment after which no answer newly
  turns reachable. That is a free label (§22 Prop. 6; Literature's first step).

**Predictions.**
* **WH1 (the state can write its own checkpoint).** Median held-out agreement with the attention anchor ≥ 0.70, and
  median head-chain S6 d19–24 ≥ 0.60 (attention: 0.65, §48).
* **WH2 (halting adapts to depth, C5).** The median halting segment increases strictly with chain length (12 < 18 <
  24 hops) in ≥ 7/10 models.

**Scope.** C5 here is the halting and writing rule, not an RL-learned policy (v3). The write head reads the state, so
it is a probe that is used causally. It is not a claim about the silent steps (§34).

**Result (2026-09-30; 10 models, CPU; `theory/results/exp23_write_halt.json`). WH1 and WH2 met.**
* **WH1.** The head agrees with the attention anchor on held-out graphs in a median 0.85 of cases (0.65–0.97). Chains
  written by the head reach S6 d19–24 = 0.639 (attention-written 0.649).
* **WH2.** The closure halt fires after median 3, 4 and 5 segments for 12-, 18- and 24-hop chains. The halt is
  strictly increasing with chain length in 8/10 models. One model halts after segment 1 at every length.
* **Reading (C5 v1–v2, label-free).**
  - A segment can write its own checkpoint from its final thought, and a free closure signal stops the chain at a
    depth that tracks the problem: about one segment per 4–5 hops, as in §48.
  - Not yet learned end to end: v3 (RL on the final answer only, MEM1-style) is the next step for C5.

## 50. A halt learned from final-answer reward (C5 v3, toy) (pre-registered 2026-09-30 19:24 EDT, before measuring the 10 test models)

**Design.** `theory/exp24_learned_halt.py`, CPU.
* The model and the write head (§49) are frozen, so the segment trajectory does not depend on the halt. Segments 1–8
  are cached per graph: the last thought, and the per-graph balanced accuracy of the max-combined answers.
* A stop head reads only the thought, with no step index: π_k = σ(w·h_k + b).
* It is trained to maximise the exact expected final-answer reward E[acc(stop) − 0.01·stop] over the cached training
  graphs (12/18/24-hop chains, seeds 21–23). This is "learned from final-answer reward", not RL (Audit's wording).

**Evaluation.**
* Held-out graphs, disjoint from training: §49's probes, seed 9, 12/18/24 hops. Greedy stop at the first π_k > 0.5.
* Baselines on the same graphs:
  - §49's closure rule;
  - fixed S = 6;
  - the best fixed S chosen on the training cache.
* The accuracy–segments trade-off is reported for each.

**Disclosure.** A smoke model outside the set (pclr1_s109, 18-hop chains) gave accuracy by segment 0.62, 0.66, 0.69,
0.70, 0.67, 0.65, 0.64, 0.63. Running longer costs accuracy, so there is an optimum to learn.

**Predictions.**
* **LH1 (better use of compute).** Held-out reward of the learned halt ≥ the closure rule's and ≥ the best fixed S's,
  pooled over the three lengths, in ≥ 7/10 models.
* **LH2 (tracks depth, as WH2).** The learned halt's median segment count increases strictly with chain length (12 < 18
  < 24) in ≥ 7/10 models.

**Result (2026-09-30; 10 models; `theory/results/exp24_learned_halt.json`). LH1 met (9/10); LH2 not met (0/10).**
* **LH1.** Pooled held-out reward is highest for the learned halt in 9/10 models. The failure is s113 (0.637 vs best
  fixed 0.643). s111 ties the best fixed S at 0.645, which counts under ≥. Example, s102: learned 0.719, closure
  0.685, best fixed (S = 3) 0.701, fixed 6 0.645. Accuracy at the learned halt, by chain length 12/18/24: median
  ≈ 0.78 / 0.74 / 0.69.
* **LH2.** The learned halt stops after a median of 2 segments at every chain length, in every model (s111: 1).
* **Reading.**
  - With max-combined answers, each extra segment adds false positives faster than it adds coverage (§48's near-depth
    erosion). So the reward-optimal stop is short and constant. It beats the closure rule, which tracks depth but pays
    for the false positives.
  - The bottleneck for C5 tracking depth, and for C6, is the aggregator, not the halting policy.
  - Fix to test next: an answer rule that does not accumulate false positives. Either answer from the last or halting
    segment (as PROTOCOL10 does), or accept a node only if it is reached from a verified chain of checkpoints. Then
    re-learn the halt.

## 51. Fixing the aggregator: single-target answers from the halting segment, with a halt learned from final-answer reward (pre-registered 2026-09-30 20:34 EDT, before measuring the 10 test models)

**Provenance note.** A first commit (639fd62) added the code but not this text: the f-string that wrote it failed. A
run started from that commit was stopped within a minute, before it wrote any result (no results file existed). The
runs scored below start after this text is committed. The aborted run's partial log was deleted, which was a mistake. Its content is not
recoverable: the task output holds only "[killed]". Since the script prints only after a whole model, which takes
minutes, no outcome line could have been printed. Note in `theory/results/exp25_target_halt_aborted.void`.

**Design.** `theory/exp25_target_halt.py`, CPU, cached trajectories; the toy analogue of PROTOCOL10.
* **Target.** One per graph: the reachable node at distance d ~ U{1..L}, or its look-alike (50/50).
* **Answer rules at a stop k.**
  - *last*: segment k's answer alone. This does not accumulate.
  - *max*: the max over segments 1..k, §50's rule.
* **Stop head.** Reads the thought and the current target logit. It is learned from final-answer reward, exact
  expected E[correct − 0.01·stop], on training graphs (seeds 31–33). Evaluation is held out (seeds 41–43, 12/18/24
  hops).

**Disclosure.** Smoke model outside the set, pclr1_s109, 64 graphs. Correctness by k:
* last rule: 0.69 → 0.52, falling, because later segments restart past shallow targets;
* max rule: 0.69 → 0.80.

**Predictions.**
* **TH1 (depth tracking under the last rule).** For reachable targets, the median stop segment is non-decreasing across
  distance bins 1–6, 7–12, 13–18, 19–24, and strictly larger at 19–24 than at 1–6, in ≥ 7/10 models.
* **TH2 (no accuracy cost).** Held-out accuracy (last rule + learned halt) ≥ (max rule + learned halt) − 0.02 in
  ≥ 7/10 models.

**Result (2026-09-30; scored by the experiments session, 073337d 22:00; `theory/results/score_exp25.json`). TH1 not met
(0/10); TH2 not met (1/10).**
* **TH1.** Under the last rule, the learned halt stops at segment 1 in every distance bin of all 10 models.
* **TH2.** Last trails max by 0.05–0.12 in 9/10. s111 passes only because both rules stop at segment 1 there.
* **Reading.** With single-target answers, later segments restart past shallow targets, so the reward-optimal halt under
  the last rule is "stop at once". Final-answer reward does not teach a depth-tracking halt here.
* As planned, the halt/aggregator line ends here: no GPT-2 port. Halting is pursued by construction instead (F1's fixed
  point, §54).

## 52. Frontier: new directions for latent reasoning (2026-09-30 20:39 EDT; requested by the user via Audit; nothing here is registered, and Literature checks novelty first)

**What the last two days taught, as constraints on any new design.**
* A loop stops on a count it carries (§35).
* A single vector per step holds only ≈ 1.4–2.3 branches (§30), and sets packed into one state compete (§45).
* Restarting from written tokens resets every count, and a trained written channel becomes causal (§26, §45).
* Max-combined answers accumulate errors (§50).
* Reach grows with the depth trained (§41).
* GPT-2 ignores untrained checkpoints (PROTOCOL8b).

The directions below change the state, the inference scheme, the training signal or how readability is obtained.
Each is meant to hit several north-star conditions at once.

### F1. Annotated-input reasoning: the state lives on the problem's own tokens
* **Idea.** Drop "one latent vector per step". Give every input token a small discrete label from a few states (e.g.
  unseen / frontier / reached / dead), carried between passes.
  - Each pass is the same transformer applied to [tokens + current labels]. Hidden activations are discarded; only the
    labels persist.
  - The answer reads the labels. Labels are trained from final answers only (straight-through Gumbel), with random pass
    budgets.
* **Why several conditions at once.**
  - The state is a set over input entities, so parallel branches are free (no superposition limit).
  - It is readable by construction: each label is a proposition about an input entity (C7). An edit flips one label.
  - It is count-free: there is no step index or growing history, and the state size is fixed (C2).
  - It is discrete, so nothing fades (C6).
  - It halts at a fixed point, when labels stop changing (C5).
  - Test-time compute is the number of passes.
* **Falsifier.**
  - Trained on depth ≤ 6, it fails beyond about 6 passes on longer chains;
  - or the labels become a covert counter (label patterns encode depth);
  - or flipping a label doesn't change the answers as BFS predicts.
* **Cheapest decisive test (toy, one local GPU, about 1 h).**
  - Set-up: a 2-layer transformer over edge and node tokens with 4-state labels per node token, trained on depth ≤ 6.
  - Measure: chains of 12–24 at 12–24 passes, plus a label-flip edit test.
  - For an LM, the labels would be an extra learned embedding added to the LM's own input tokens.
* **First non-toy test: CLUTRR with GPT-2 small.** Each person token carries a relation-to-anchor label; one pass
  composes one hop. Train k = 2–4, test k ≤ 10. Success would be:
  - accuracy at k = 8–10 at least that of text CoT;
  - reach ≥ 2× the trained depth, beyond the ≈ 1.5× of 2608.11233;
  - label-flip IIA ≥ 0.8.

### F2. Bidirectional latent reasoning (meet in the middle)
* **Idea.** Run two latent searches: forward from the question's root, and backward from a candidate answer over
  reversed relations. Each writes its frontier. The answer is "yes" when the two written frontiers intersect.
* **Why several conditions at once.**
  - Each search needs only half the depth, so reach doubles at the same trained horizon (C2), with no new training if
    the model is relation-symmetric.
  - The meeting node is a checkable certificate (C7) and a natural halt (C5).
  - The two searches run in parallel.
* **Falsifier.** Backward search can't be learned from final answers, or the intersection gives false positives, or
  there's no reach gain over forward search alone at equal compute.
* **Cheapest decisive test (CPU, $0, existing toy models, zero-shot).**
  - The same model computes reachability *to* a target by reversing edge directions and starting from the target.
  - So F = forward set within 6 hops, B = backward set within 6 hops, and "reachable within 12" ⇔ F ∩ B ≠ ∅.
  - Measure bal-acc at d7–12 vs one episode (chance) and vs restarts (§46).
* **First non-toy test: GPT-2 Coconut on ProsQA (the PROTOCOL8 checkpoints).** Forward from the root; backward from each
  answer option over reversed "is-a" statements; answer by intersection. Success would be:
  - accuracy at answer depth 11–14 at least 0.10 above forward-only at equal compute;
  - i.e. reach ≥ 2× a single search's horizon, above the 1.5× bar.

### F3. Compositional self-synthesis: deeper training problems from verified shallow ones
* **Idea.** The model makes its own deeper training data by composing problems it has solved and verified: root→x
  and x→y become root→y.
  - The composite label comes *by construction* from the verified sub-answers, not from the model's guess on the deep
    problem.
  - Iterating doubles the trained depth each round.
  - In real domains: relation chains in CLUTRR, operation chains in arithmetic, lemma chains in proofs.
* **Why several conditions at once.**
  - Reach grows with trained depth (§41), so this is C2 from data. No external labels are used (C4).
  - It avoids the error avalanche by construction: errors enter only through verification of shallow sub-answers,
    which is the reliable part.
* **Falsifier.** The model learns the seam (composed problems are off-distribution), or sub-answer verification fails,
  or reach doesn't grow faster than the restart ladder (L/M).
* **Cheapest decisive test (toy, local GPU, about 2 h).**
  - Set-up: compose pairs of 6-hop problems into 12-hop training chains, labelled from the model's own verified 6-hop
    answers.
  - Compare against the re-anchoring teacher (L-T8) and true labels (L-GT) on d13–24.
* **First non-toy test: multi-step arithmetic or CLUTRR with GPT-2 small.** Compose verified short operation or relation
  chains into longer ones. Success would be:
  - the trained depth doubles per round with no external labels;
  - extrapolation beyond 1.5× trained length, at least matching a CoT-supervised baseline.

### F4. A behavioural round-trip translator for silent thoughts (C7 by optimisation)
* **Idea.** Learn a decoder from thought to a short text over a constrained vocabulary (the input's entities plus a
  few relation words), and an encoder from text back to thought. Train them through the *frozen* reasoner, on the
  north star's own C7 tests:
  - Round-trip loss: substitute encode(decode(h)) for h, and the answers must not change.
  - Edit loss: change an entity in the text, and the answers must change as BFS on the edited text predicts.
  - A paraphrase-invariance term detects steganography.
* **Why several conditions at once.** It is the first method aimed directly at C7 for the *silent* steps (§34 reached
  only 0.36 with a fitted linear map). It does not touch the reasoner (C1–C6 keep their status), and it applies to any
  latent model, including Coconut GPT-2.
* **Falsifier.**
  - Round-trip agreement plateaus well below about 0.95, or edit IIA stays near 0.36. Either means a hidden channel the
    vocabulary can't express, which is itself a measurement.
  - Or paraphrases break behaviour, which means steganography.
* **Cheapest decisive test (toy, CPU or local GPU, under 1 h).**
  - Set-up: decoder = multi-label over node ids (a written frontier set); encoder = a small MLP. Train through the
    frozen CLR1 or P30 models with both losses.
  - Measure: held-out round-trip agreement and edit IIA vs §34.
* **First non-toy test: the frozen Phase 0 Coconut GPT-2 on ProsQA**, with entity names as the vocabulary. Success would
  be round-trip agreement ≥ 0.95 and edit IIA ≥ 0.8 on silent thoughts. Published decoders (NLA) report only
  reconstruction and steering, not behavioural substitution.

### F5. Verify-then-extend: latent proof checking as test-time compute
* **Idea.** Treat each written checkpoint as a claim. Spend test-time compute on short latent *verification*
  episodes ("from c_j, is c_{j+1} reachable?") and extend only verified checkpoints; on failure, branch to the next
  candidate.
  - Compute scales with the verification budget.
  - Errors are rejected at the seam instead of accumulating (§50).
* **Falsifier.** Verification episodes catch wrong checkpoints no better than chance, or the cost exceeds the gain.
* **Cheapest test (CPU, existing CLR models).** Filter each chain step's anchor by a verification episode, and compare
  chain accuracy and false positives with §48.
* **First non-toy test: ProsQA or GSM-style problems.** Success would be an accuracy-vs-test-time-compute curve above
  text-CoT self-consistency at matched FLOPs (C3).

**Ranking by expected north-star coverage** (requested by Audit):

| Rank | Direction | Conditions |
|---|---|---|
| 1 | F1 | C1, C2, C4, C5, C6, C7 |
| 2 | F2 | C2, C5, C7 |
| 3 | F3 | C2, C4 (C6) |
| 4 | F4 | C7 for silent steps |
| 5 | F5 | C2, C6 and a C3 route |

**Run order for the experiments session, once Literature's checks land.**
1. F2 first: zero-shot on CPU, and decisive for doubling reach.
2. F1: one toy GPU-hour, highest coverage.
3. F4: CPU or local GPU.
4. F3: local GPU.
5. F5: last.

### Addendum (2026-09-30 22:22 EDT): the external input, evaluated as pioneer input

Source: `docs/EXTERNAL_IDEAS_2026-09-30.md` (87949a7). Literature's check: `docs/LIT_EXTERNAL_CHECK_2026-09-30.md`
(8ce27e8).

**Verdicts.**
* **Unifying hypothesis (entry-state normal form): adopted as a test (§56).**
  - In our terms, a state is in normal form when its future depends only on the residual problem. This is a causal
    state (Shalizi & Crutchfield 2001) or a bisimulation class (2006.10742).
  - Restarts (§26) impose it by construction: a written node token has no age.
  - The open question is whether the *latent* states of repaired single loops have it, e.g. Pilot L's students.
  - Closest empirical precedent: path independence in equilibrium models (Anil 2211.09961). Our delta is a
    growing-history latent loop and a causal test of restart-induced canonicalisation.
  - Scored as IIA, with an age-only reset as the control.
* **Direction 3 (commit gates): its kill test is adopted (§55). The direction itself is predicted to give C5/C6, not
  C2.**
  - Prop. 10 applies to gates too. On graphs of depth ≤ D_max, "freeze at D_max" costs as little as "fall back at
    D_max".
  - A penalty on reward-neutral mutations therefore turns the fall-back into a fixed point: a halt, with stable
    long-budget answers. It cannot make the loop continue past D_max, because no training residual ever needs that.
  - Prediction for any D3 training test: reach unchanged, long-budget stability improved.
* **Direction 1 (boundary-response elimination): the most novel; the same family as F1.**
  - Both put the state on the problem's own structure and update it with a fixed-size local rule. Both are count-free
    by construction: there is no learned stop, because the contraction tree or the fixed point decides.
  - D1 adds what F1 lacks: composition of regions, i.e. log depth.
  - Caveat: response tables over distances (min-plus) grow with region size and bring a count back. Over Boolean
    reachability they are canonical bits.
  - Its minimal version on our substrate is F7 below. A scaffolded mechanism test, as Audit says.
* **Direction 2 (conserved continuations): later.**
  - Call/return is established (PENCIL 2503.14337, Recursion of Thought 2306.06891, Recursive NPI 1704.06611). Only
    exactly-once returns are new.
  - Here a continuation is a CLR checkpoint with a return slot. Exactly-once consumption bounds fan-out, which may
    bear on the false-positive build-up of §48/§50.
* **Direction 4 (predictive state): already F1's semantics.** F1's labels are predictions of named tests ("is X
  reachable?"), so F1 is a predictive-state representation whose tests are the queries. The Boolean-register domain is
  a narrow C7 substrate; not prioritised.
* **Direction 5 (Krylov): a control only.**
* **Benchmark correction (2608.11233).**
  - It reports ~1.44–1.50× extrapolation, but with dense intermediate labels first and no matched compute.
  - So it is a pretrained-LM baseline, not evidence for C3 and C4 together. Literature confirmed this, and NORTH_STAR
    already scopes it.

**New directions derived from the input** (Literature checks novelty before either is registered).
* **F6. Age-transplant training: path independence from shallow data (C2 + C4 in one loop, with no deep labels and
  no restarts).**
  - **Idea.**
    - Train on graphs of depth ≤ D as usual.
    - Sometimes feed a training episode, at step 2, an *aged* latent: one taken at age a ≤ D from another episode
      whose frontier is the same node id (another graph, ids matched).
    - Train only on the recipient's final answers.
    - The recipient must then continue r ≤ D hops after receiving an age-a state, with a + r up to 2D. Any stop keyed
      to a count at D now costs loss, while every graph stays within depth D.
  - **Why.** By Prop. 10, a stop keyed to a count is free whenever training residuals never outlast that count.
    - Deeper data breaks this (§41), but needs deep labels.
    - Restarts sidestep it (§26), but need written checkpoints.
    - F6 breaks it with shallow data and keeps a single loop.
  - **Falsifier.** Reach stays at D. Or the model learns a seam detector, treating transplanted states specially, so
    natural long runs gain nothing.
  - **Cheapest test.**
    - Set-up: the P30 recipe with transplants, 10 seeds, vs P30 controls, at the same budget as a P30 run.
    - Measure: one-loop bal-acc at d7–12 (K′ = 12) and d13–24 (§37's probe).
  - **Sequencing.** It runs after §56.
    - If §56 shows the students' latents are interchangeable, F6 trains that property directly.
    - If age stays causal in them, F6 is the only single-loop route aimed at age itself.
* **F7. Path doubling on pair labels (D1 ∩ F1): log-depth latent reasoning.**
  - **Idea.** Labels sit on node pairs: l(i, j) = "j reachable from i within 2^p hops". A learned pass over pair
    tokens composes l′(i, j) = l(i, j) ∨ max_k [l(i, k) ∧ l(k, j)].
  - Reach doubles per pass, so P passes cover 2^P hops: test-time compute logarithmic in depth (a C3 route).
  - **Prior to check:** pointer jumping, transitive closure by squaring, neural algorithmic reasoning.
  - A toy test comes after F1's language-model result.

**Order.** §55 and §56 first: CPU, existing models, handed to Experiments. Then F6 (local GPU), then F7. D2 waits.

### Addendum (2026-10-01 10:10 EDT): F3's premise fails on this toy — not registered

* **Plan.** F3 (compositional self-synthesis) was to label deep chains by composing the frozen model's own in-range
  answers at the generator's junction (A5): label = 1 − (1 − p₁)(1 − p_J·p₂). Code: `distill_reanchor.py --teacher
  comp|compv`.
* **Prior** (Literature `docs/LIT_F3_CHECK_2026-10-01.md`, f987aea; medium-low novelty). Rajaraman 2606.27721 is the
  theory; MuSiQue 2108.00573 composes single-hop questions.
* **Literature's caution.** Composed error ≈ 2ε − ε².
* **Premise check before registration.** `theory/exp29_teacher_errors.py --kind comp|compv` →
  `theory/results/exp29_comp_errors.json`, `exp29_compv_errors.json`; 20 parents, the training distribution.
  - **COMP.** Error at d1–6 0.162, at d7–12 0.379. Seam +0.228 (range +0.13 to +0.30), the same as T8's +0.239. It
    misses 65 % of reachable deep nodes.
  - **COMP-V.** The confidence filter keeps 77 % of labels at d1–6 and 48 % at d7–12, with kept error 0.088 vs 0.311.
    The seam remains.
* **Why.** The generator glues at A5, at depth 6, which is the parents' weakest in-range depth. Composed labels
  inherit the junction's error.
* **Decision.** Seam-free labels would need shorter glued segments, i.e. a new generator, which would be a design
  patch. F3 is not registered on this toy.
* **The finding, as a constraint on label-free depth extension.** A composed label is only as good as the model's
  accuracy at the junction depth. The junction must sit well inside the reliable range, not at its edge.

 — learning F1's local rule without step labels

§54b showed that GPT-2's label channel extrapolates perfectly once its one-hop rule is learned. Deep final answers alone
(ProsQA has no question shallower than 3 hops) do not teach that rule.
* **F1-S. A shallow-question curriculum.**
  - **Idea.** Generate questions of every depth 1–6 over the same graphs, with matched passes P = d, and train on final
    answers only.
  - A depth-1 question's answer exercises exactly one pass, so the rule is learned where credit assignment is trivial.
    Deeper questions then teach its composition.
  - This mirrors the toy (§54), whose make_batch asks queries at every depth.
  - **C4 question for Audit.** No intermediate state of any training problem is labelled, but the training
    distribution gains shallow problems from the same generator. That is the same disclosure class as Pilot L's deep
    graphs.
  - **Falsifier.** Hardened accuracy ≥ 0.95 at d1–6 but no extrapolation to d11–14 (composition not learned), or no
    convergence even at d1.
  - **Cheapest test.** One arm, 3 seeds, the frozen §54b code paths plus a question generator. Use p8_na graphs (§59),
    so the cue is unavailable.
  - **Literature to check first:** easy-to-hard and length curricula for looped or iterative models, and "train on one
    step, compose at test" without step labels.

### Addendum (2026-10-01 15:00 EDT): design draft for GPT-2 state tracking — the written-state chunked loop (not registered; for Literature's and Audit's comments)

**Why this rather than per-variable value slots.**
* §64's transferable result is the written-state restart: 0.995 at N 9–16 on S3, with the written state read
  faithfully (IIA 0.969).
* F1-style public-label passes have twice failed on credit assignment from final answers: F1-S trained in 1/3 seeds;
  F4's straight-through translator never trained.
* A state written as text is public, readable and overwritten at every chunk by construction, with no new discrete
  bottleneck to optimise.

**Task.** Permutation tracking in text: n = 4 boxes holding items A–D.
* Operations are transpositions, "swap boxes i and j" (generators of S4, non-abelian), in random order.
* Question: "Which box holds item X?" (4-way), or the full arrangement.
* Generator-built, so there is no answer shortcut; the identity predictor is registered as the no-reasoning baseline.
* Train N ≤ 8 operations; test N = 1–32.

**Arms** (all from original GPT-2 small; the same training problems and token budget accounted for):
* **NC (no CoT).** Question → answer directly. Final answers only.
* **TC (text CoT, the C3 bar).** The arrangement written after every operation, then the answer. Trained with step
  labels: the standard CoT reference, not C4.
* **CL (Coconut single loop).** K′ = N continuous thoughts, then the answer. Final answers only.
* **WC (written-state chunked loop; the proposed method).**
  - The model reads [current arrangement as text + the next k ≤ 4 operations], runs k continuous thoughts, then writes
    the new arrangement as text.
  - The next chunk restarts from [original question + new arrangement + the next k operations], with the latent
    history cut.
  - Training uses final answers of short problems only: "start arrangement + k ≤ 4 operations → end arrangement", with
    random start arrangements. That is C4, since each chunk's target is a final answer, never a thought.
  - At test, chunks are chained, so the state is overwritten as text every k operations.

**Measures.**
* Accuracy vs N (1–32).
* Compute per problem: forward FLOPs and generated or latent positions; matched-compute curves vs TC (C3).
* Readability of WC's written states, which is exact by construction (text). Edit IIA: replace a written arrangement
  and check that the answer follows the true composition from it (as §64 ST4).
* For CL: §64's state-validity probe, via a linear readout of the arrangement from the thoughts.

**Predictions to register** (draft):
* CL fails past the trained length (§64's analogue).
* WC holds ≥ 0.95 to N = 32.
* WC matches TC's accuracy at ≈ 1/k of TC's written tokens (the C3 claim).
* Edit IIA ≥ 0.9.

**Open points for comments.**
* Which group: S4 transpositions (solvable; a GPT-2 shortcut is possible), or S5 / A5 (non-solvable; the deadlock risk
  of §64)?
* What k should be. GPT-2 may compose 2–4 transpositions per forward pass without latent steps (an NC-style shortcut);
  WC with k latent steps should be compared with WC₀ (chunks of k with no latent thoughts).
* Matched-compute accounting rules for TC vs WC.
* How it relates to scratchpad chunking and PENCIL-style reduction (Literature).

### Addendum (2026-10-01 18:30 EDT): post-hoc latent snapping fails its premise — not registered

* **Plan.** Snap the S3 loop's continuous thoughts to a k-means codebook built from in-range thoughts, to separate drift
  from range (the draft of a "§67" never registered). Code: `theory/exp32_snap.py`.
* **Premise check before registration** (in-range N only, so no outcome was measured; s3_s7 outside the set and gated
  s3_s1, in-range only). Snapping destroys in-range accuracy.

  | s3_s1, accuracy at N = 8 | Value |
  |---|---|
  | No snapping | 0.99 |
  | Pooled 128 codes, every step | 0.27 |
  | Pooled 256 codes, every step | 0.30 |
  | Pooled 128 codes, every 4th step | 0.90 |
  | Per-step 32 codes (age cycle) | 0.20 |

  Purity against the current arrangement is 0.20–0.30, and all 6 states appear as cluster majorities.
* **Reading.** The toy's thoughts are not a finite-state code. They carry continuous content beyond the 6 states,
  e.g. the content pointer to random operation positions, which no post-hoc codebook preserves. Drift-vs-range cannot
  be separated this way.
* **Consequence.** Discreteness must be built in during training, with a credit-assignment method that works. That is
  §67 (Design 1).

## 53. F2 — bidirectional latent reasoning, zero-shot on toy models (pre-registered 2026-09-30 20:45 EDT, after Literature's check 3ba2507 and before any measurement on the test models)

**Design.** `theory/exp26_bidirectional.py`, CPU, inference only, no training.
* **Forward.** One K-step episode from the root, on the graph as given. F = the graph nodes the model answers
  reachable.
* **Backward.** One K-step episode on the same graph with every edge reversed, START = the target t. B = the graph
  nodes it answers reachable, i.e. t's ancestors.
* **Answer.** "t reachable" iff F ∩ B ≠ ∅. Both are written node-id sets.
* **Probe.** The long-chain probe(512): targets are the chain nodes at distance 1..12 and their look-alikes.
* **Models.** The 59 P30 ∪ L models of §26.
* **References.** Forward-only (F alone) and §26's re-anchoring re2top3.

**Required control (Literature).** Precision by depth: the false-positive rate of the intersection on look-alike
targets, for each distance.

**Disclosure.** Two smoke models outside the set:
* ph_c_s102: bidirectional d1–6 0.90, d7–12 0.78; forward-only d7–12 0.50; look-alike FP rate d7–12 0.25.
* pb_at_s751: no gain, look-alike FP rate 0.98. Its sets are flooded with positives, which is exactly the risk.

**Predictions** (medians over the 59 models):
* **BD1 (reach doubles zero-shot).** Bidirectional bal-acc at d7–12 ≥ 0.70 (forward-only ≈ 0.50).
* **BD2 (precision).** Look-alike FP rate at d7–12 ≤ 0.30.
* **BD3 (no in-range loss).** Bidirectional bal-acc at d1–6 ≥ 0.85.

**Descriptive.** Bidirectional vs re2top3 at d7–12, and the FP rate by distance.

**Runs and scoring.** Handed to the experiments session:
* Command: `python theory/exp26_bidirectional.py` → `theory/results/exp26_bidirectional.json`.
* Score: medians of `bidir_bal` over indices 6–11 and 0–5, and of `bidir_decoy_fp` over indices 6–11.

**Result (2026-09-30; scored by the experiments session, 474f6bd; `theory/results/score_exp26.json`, 59 models). BD1 met;
BD2 and BD3 not met.**
* **BD1.** Bidirectional bal-acc at d7–12 is 0.712, vs 0.501 for forward-only and 0.625 for re2top3.
* **BD2.** The look-alike FP rate at d7–12 is 0.417 (by distance: 0.75 at d1, 0.25 at d4, 0.49 at d12).
* **BD3.** d1–6 is 0.803, vs 0.853 forward.
* 16/59 models meet BD1 and BD2 together.
* **Reading.** The zero-shot meet-in-the-middle does extend reach: the intersection finds 0.90 of reachable targets at
  d7–12. But the backward episodes over-mark, so it pays in precision. In GPT-2 there was no signal (PROTOCOL11, FP
  0.92).
* F2 stays a mechanism for models whose sets are precise. It is parked unless a trained backward direction is wanted.

## 54. F1 — annotated-input reasoning: toy feasibility, and the registered language-model test (2026-09-30 21:22 EDT; registered after Literature's check 3ba2507 and before any language-model run)

**Toy feasibility (exploratory, not registered; disclosed).** `theory/ail.py`; one seed, 800 CPU steps, batch 32;
`theory/results/ail_smoke/`.
* The state is a binary monotone "reached" label per node id, the only thing carried between passes. Each pass is a
  2-layer transformer over edge tokens with the endpoint labels added. It is trained from final answers only, with
  P = 6 and depth ≤ 6.
* In distribution: 1.00. 12-hop probe at P = 12: 1.00 at every distance. 24-hop probe at P = 24 and P = 32: 1.00.
* The fixed point comes at exactly 12 and 24 passes, so halting tracks depth.
* Label-flip edit (switch on a look-alike node at pass 2): IIA 1.00.
* **Reading.** By construction, the labels make the BFS rule a one-edge local rule, and the toy hands the model the
  graph structure as edge tokens. This shows the architecture can be learned from final answers and gives count-free,
  readable, depth-tracking reasoning. It is *not* decisive: it resembles discrete neural algorithmic reasoning on
  given graphs (Literature: 2306.13411, DiscoLoop 2607.00341). The open question is whether the same holds when the
  structure must be read from text.
* Two prototype bugs were found and fixed during smoke testing: monotone labels flooded at a random init, and the loss
  clamp killed the straight-through gradient.

**Registered test: GPT-2 small on ProsQA text** (spec for the experiments session to implement and run).
* **Architecture.** GPT-2 small reads the problem text.
  - Every token that belongs to an entity mention (person or concept) gets a label embedding added, for the entity's
    current label ∈ {0, 1}.
  - One pass = one GPT-2 forward over [tokens + label embeddings]. A linear mark head on every token gives a logit.
    Per entity, the max over its mentions; then the new label = old OR (logit > 0).
  - Hard labels forward, sigmoid straight-through backward, with an affine-squashed loss (not a clamp).
  - Initial labels: the root person = 1. Answer: whichever option's label is on (both or neither: compare logits).
    Nothing else carries between passes.
* **Training.** Final answers only (C4), on problems of answer depth ≤ 6, with P = 6 passes.
* **Arms (3 seeds each).**
  - AI-mono (primary).
  - AI-free: labels may switch off. This is the counter-risk ablation.
  - AI-hint: per-pass targets = entities within p hops. This is the upper bound Literature asked for.
* **Test.** Answer depths 3–14 (PROTOCOL8's test sets), with P = the depth needed (up to 14), and P = 2× trained.
* **Predictions (AI-mono).**
  - **AG1 (learnable).** In-distribution accuracy (depth ≤ 6) ≥ 0.95 in ≥ 2/3 seeds.
  - **AG2 (C2).** Accuracy at depths 11–14 ≥ 0.85 in ≥ 2/3 seeds. That is reach ≥ 2× the trained depth, against the
    ≈ 1.5× of 2608.11233 and Coconut's chance.
  - **AG3 (C7).** Label-flip edit IIA ≥ 0.8. Switch on a concept on the negative option's side at pass k; the negative
    option must become reached, with nothing else changing.
  - **AG4 (C5).** The pass at which labels stop changing correlates with answer depth (Spearman ≥ 0.8).
* **Descriptive.**
  - AI-free vs AI-mono extrapolation (counter check).
  - AI-hint as the ceiling.
  - Text CoT at matched compute (C3).
* **Falsified if** AG1 holds but AG2 fails, i.e. the labels become a covert counter or the model learns a depth-bounded
  rule. If AG1 fails, final-answer training of the label channel does not converge in text (Literature's risk).

### 54b. F1 on GPT-2, the one revision: relaxed labels in training, hard labels at test (pre-registered 2026-09-30 22:24 EDT, after PROTOCOL12's registered outcome 1dcac15 and before any revised run)

**Why §54 failed** (PROTOCOL12, `docs/PROTOCOL12.md`, `docs/PROTOCOL12_TUNING.md`).
* Hard forward labels make the loss piecewise constant. The straight-through gradient moves the parameters, but no
  label crosses its threshold at GPT-2's step sizes, so the loss never changes. At head lr 1e-2 the labels flood and
  then collapse.
* The toy escaped this because it trains from scratch at lr 1e-3 with 16 queries per graph at every depth, so crossings
  happen.
* With soft forward labels, the same GPT-2 model fits per-pass targets exactly.
* So this is an optimisation-design failure, not evidence against F1 (Audit). This is the one revision; whatever it
  gives is F1's language-model result.

**Changes from §54.** Everything else is unchanged: architecture, data, depths, P, AG2–AG4, edit test.
* **Training forward, two estimator arms.**
  - **AI-soft (primary).** l′ = max(l, σ(z/τ)), with τ annealed linearly from 1 to 0.3 over the first two thirds of
    training, then held. A binarisation penalty λ·mean[l(1 − l)] over the entity labels is ramped from 0 to λ = 0.1
    over the last third.
  - **AI-stoch.** The forward is binary even in training: l′ = max(l, b) with b ~ Bernoulli(σ(z)), and the gradient
    goes straight through σ (stochastic binary units). The expected loss is smooth in z, and the training-time state
    is binary, so no analog channel can form.
* **Curriculum, both arms.** Answer depth 3 only for the first sixth of training, widening linearly to 3–6 by one
  third.
* **Test, both arms.** Hardened labels: the hard OR at threshold 0.5, as specified in §54. Soft-test accuracy is
  reported alongside.
* **Arms and seeds.** AI-soft, AI-stoch and AI-hint (the ceiling, soft forward with per-pass targets), 3 seeds each:
  9 runs. AI-free (the counter ablation) is dropped for budget; AG2 on hardened labels covers the counter risk (a
  covert counter fails AG2).
* **Tuning cap (Audit's rules).** At most 3 settings or 30 min on training data, all logged in
  `docs/PROTOCOL12_TUNING.md`. Only τ's end value and λ may be tuned. Then the settings are frozen with the code hash
  for all arms.

**Predictions** (AI-soft primary; AI-stoch scored on the same criteria and reported):
* **AG0 (hardening gap; new).** In-distribution, soft-test accuracy − hardened-test accuracy ≤ 0.03 (median over
  seeds). A larger gap means the relaxed labels carried non-binary information: a covert analog channel, which is
  F1's readability failure mode.
* **AG1′ (learnable).** Hardened in-distribution accuracy (depth ≤ 6) ≥ 0.95 in ≥ 2/3 seeds.
* **AG2–AG4.** As in §54, on hardened labels: depths 11–14 ≥ 0.85; edit IIA ≥ 0.8; fixed-point pass vs depth Spearman
  ≥ 0.8.
* **Estimator comparison (descriptive).** AI-stoch vs AI-soft on AG0 and AG2. My expectation: AI-stoch has the
  smaller gap but learns more slowly.
* **Descriptive.** End-of-training binarity: the share of label values within 0.1 of 0 or 1. Text CoT at matched
  compute (C3). AI-hint as the ceiling.

**Readings.**
* **AG1′ and AG0 met.** F1 is learnable in text with labels that stay readable when hardened. Then AG2–AG4 decide C2,
  C7 and C5.
* **AG1′ met, AG0 failed.** The soft channel does the work, and the readable state is not the computing state.
* **AG1′ failed in both arms.** Final-answer training of an entity-label channel does not converge on GPT-2 / ProsQA
  within this budget. F1 stands on the toy only.

**Priors for the estimators (Literature to confirm).** Straight-through and stochastic binary units (Bengio et al.
1308.3432); Gumbel-softmax / Concrete (1611.01144, 1611.00712).

**Run (experiments session).** `prosqa/ail_gpt2.py` with `--arm soft|stoch|hint`. 9 runs on the 3 local cards, about
6 h at ~0.65 s/step, or Vast within budget. Score AG0–AG4 on PROTOCOL8's test sets, excluding the 12 plumbing-check
problems.

**Amendment 54b-1 (2026-09-30 22:28 EDT, before any revised run; Literature `docs/LIT_F6F7_CHECK_2026-09-30.md`, c1ab04c,
and Audit).**
* **Hardening gap per run.** Report the soft-vs-hard gap for every run and every depth band, not only AG0's pooled
  median. This follows the DARTS discretisation-gap precedent (1806.09055, 1909.09656). τ is annealed rather than
  hardened abruptly (Soft-to-Hard VQ 1704.00648); the schedule above already does this.
* **Cue control (from §57; descriptive).** Hardened accuracy on test problems with one added statement "<another
  person> is a <a parent of the target>".
  - The answer is unchanged, so a reasoner from the root is unaffected, while ProsQA's proximity cue flips toward the
    negative.
  - Report the drop vs plain per arm. A drop > 0.05 means the label channel uses the cue.

**§54b result (2026-10-01; scored by the experiments session, 814b2ff, `results/p54b.json`, `docs/PROTOCOL12.md`; 9 runs,
settings frozen at τ_end 0.3, λ 0.1). "AG1′ failed in both arms." The ceiling arm is perfect.**

| Arm | Hardened accuracy d3–6 / d7–10 / d11–14 | AG1′ | AG3 IIA | AG4 ρ | Cue drop |
|---|---|---|---|---|---|
| AI-soft (primary) | 0.50–0.54 / 0.48–0.54 / 0.45–0.50 | 0/3 | 0.36 | 0.19 | 0.003 |
| AI-stoch | 0.14–0.45 at every band | 0/3 | 1.00 (flooding) | 0.12 | 0.008 |
| AI-hint (per-pass targets, not C4) | **1.000 / 1.000 / 1.000, all 3 seeds** | 3/3 | 1.00 | 1.00 | 0.000 |

* **AI-soft.**
  - Within 100 steps every soft label sits at σ ≈ 0.5: loss 2 ln 2, binarity 0.04. That is the symmetric uninformative
    fixed point, reached before τ or λ can act.
  - AG0 is met only trivially, since soft and hardened accuracy are both at chance.
* **AI-stoch.** Every entity gets marked (recall 1.0, precision 0.55), so answers fall to the tie-break.
* **AI-hint.** Trained at depth ≤ 6, tested with P = 14:
  - exact at every band, including d11–14 (2.3× the trained depth);
  - label precision and recall 1.000; edit IIA 1.000;
  - fixed point vs depth ρ = 1.00;
  - no cue use.
* **Reading (fixed in §54b).**
  - Final-answer training of the entity-label channel did not converge, so F1's C4 claim rests on the toy only.
  - The representation question is answered positively. GPT-2 with an entity-label channel runs an exact, readable
    (C7), halting (C5), depth-extrapolating (C2), cue-free search once the one-hop label rule is learned.
  - The obstacle is learning the rule from deep final answers (credit assignment through P discrete passes), not
    representing or iterating it.
  - Where F1 goes next is a §52 frontier item (addendum 2026-10-01 below), not a further §54 revision.

## 55. Kill test: is the anytime fall-back reward-neutral? (external direction 3; pre-registered 2026-09-30 22:22 EDT, before measuring any of the 17 test models)

**Question.**
* Prop. 10 (§35) says the count-keyed fall-back costs nothing in training.
* Direction 3, commit gates (§52 addendum), needs exactly that. A penalty can only target mutations whose terminal
  utility is neutral.
* Priors (Literature): COMA 1705.08926, counterfactual credit 2011.09464, Skip RNN 1708.06834.

**Design.** `theory/exp27_fallback_kill.py`, CPU, no training.
* **Fall-back point.** The model's base peak step t* on the long-chain probe (P10e, `theory/results/exp16_stop_key.json`).
* **Kill.** From step t*+2 on, the thought fed into each step is h_{t*} again: identity after the peak. The fallen-back
  thoughts never enter.
* **Positive control.** From step t*+1 on, the input is h_{t*−1}. This removes one real advance.
* **Data.** In-distribution graphs: split "discovery", 2048 graphs, seed 2027, with make_batch's 16 balanced queries per
  graph. Answers are read at the trained K, which is the training readout. Outcomes: terminal BCE and bal-acc.
* **Models.** The 17 anytime runs of P10e.
  - The kill acts inside the trained budget only where t*+2 ≤ K. That gives 6 models: pe8_at s1000, s1001, s1003,
    s1005 and s1006, and pd_at s908. s1005's pointer stalls at 2.
  - In the other 11, the fall-back starts at or after K, so training never executed it. Those are reward-neutral by
    construction and reported as "outside".

**Disclosure.** Smoke models outside the test set, with an artificial t* = 4: pb_at_s751 and pb_at_s750 (K = 6, so the
kill covers step 6 only).

| Model | ΔBCE, kill | ΔBCE, positive control | Bal-acc |
|---|---|---|---|
| pb_at_s751 | +0.0001 | +0.033 | unchanged |
| pb_at_s750 | +0.0013 | +0.009 | unchanged |

**Predictions** (medians over the inside models):
* **KT1 (reward-neutral, Prop. 10).** ΔBCE_kill ≤ 0.25 × ΔBCE_pos and |Δbal_kill| ≤ 0.01.
* **Killed** (direction 3 has no lever) if ΔBCE_kill ≥ 0.5 × ΔBCE_pos. Between these: partial.
* **Uninformative** if the median ΔBCE_pos ≤ 0.
* **KT2 (descriptive; the addendum's gate argument).** On the probe at K′ = 12, the kill does not raise d7–12 bal-acc
  (median change ≤ +0.02). Identity freezes the state; it cannot advance it.

**Readings.**
* **KT1 met.** The fall-back is a free mutation, so a gate penalty has a target. Per the addendum, though, the penalty
  turns the fall-back into a fixed point (C5/C6), not into reach (C2).
* **Killed.** The post-peak updates carry answer value, so a gate penalty would cost in-distribution accuracy.

**Run and score (experiments session).**
* Run: `python theory/exp27_fallback_kill.py` → `theory/results/exp27_fallback_kill.json`. CPU, a few minutes.
* Score, over models with `"inside": true`: medians of kill[0] − base[0], pos[0] − base[0] and kill[1] − base[1].
* KT2: probe_kill − probe_base at indices 6–11.

**Result (2026-10-01; scored by the experiments session, 298c9f7, `theory/results/score_exp27_28.json`; Audit
reproduced). KT1 partial, at the boundary; KT2 met.**
* **KT1, 6 inside models.** Median ΔBCE_kill is 0.00529 vs 0.25 × ΔBCE_pos = 0.00529 (above it by 4·10⁻⁶).
  - The median of per-model kill/pos ratios is 0.30 (0.21–0.60), so "partial" holds under either aggregation.
  - |Δbal_kill| = 0.004 ≤ 0.01. "Killed" (≥ 0.5) is not reached.
* **KT2.** Probe d7–12, kill − base: median −0.004. Identity cannot advance.
* **Reading.**
  - The fall-back is nearly free in accuracy (−0.4 points), but in loss the post-peak updates do about a quarter to a
    third of one real advance's work.
  - A commit-gate penalty (direction 3) therefore has a target with a small in-range price, and by KT2 it buys no reach.
  - D3 is parked as a C2 route. F1's fixed point already gives the C5/C6 benefit by construction.

## 56. Are aged latent states interchangeable? The entry-state normal form, tested by interchange (external unifying hypothesis; pre-registered 2026-09-30 22:22 EDT, before measuring any of the 80 test models)

**Question.**
* Do restarts and distillation work by moving progress into an entry-state normal form? That is a state whose future
  depends only on the residual problem.
* Formal anchors (Literature): causal states (Shalizi & Crutchfield 2001), bisimulation (2006.10742) and path
  independence (Anil 2211.09961).
* Restarts impose the normal form by construction.
* Do the *latent* states of repaired single loops have it too? The competing account: deeper or re-anchored targets
  only move or slow the count (§41's fade reading), and age stays causal.

**Design.** `theory/exp28_interchange.py`, CPU, no training.
* **Probe and queries.** §37's 24-hop probe (seed 7, 512 graphs), with transport point f, i.e. frontier c_f. Every
  condition continues 12 steps from c_f on the same graph. All are scored on the same 24 residual queries per graph:
  the chain nodes at f+1..f+12 and their look-alikes.
* **CT (reference).** A fresh episode from the written node c_{f−1}, 13 steps. Its own h_1 holds c_f at age 1.
* **RT_a (the interchange intervention).** The thought fed into CT's step 2 is replaced by h_a from an episode started
  at c_{f−a}, with a ∈ {2, 3, ⌈f/2⌉, f−1}. That donor has the same frontier c_f at age a. Its route stays on the chain,
  so the residual problem matches exactly.
* **RT_root (secondary).** The root episode's h_f: age f. That route has also visited the root's other branches.
* **AR (age-only reset, the control).** The oldest chain donor shifted along the per-step mean:
  h − μ_{f−1} + μ_1, with μ from a separate probe (seed 8).
* **RND (content control).** The oldest chain donor, taken from another graph.
* **Scores.**
  - IIA_a is the agreement of RT_a's answers with CT's. The high-level model "the future depends only on the residual
    problem" predicts IIA = 1.
  - Gap G_a = (1 − IIA_a) / (1 − IIA_RND): an age swap's effect as a share of a content swap's.
  - Age effect: dG = G_{f−1} − G_2.
  - A model counts only if 1 − IIA_RND ≥ 0.05.
* **Models and primary f** (just below each family's trained depth). Both f values are computed for every model; the
  non-primary one is descriptive.

  | Family | Models | Primary f |
  |---|---|---|
  | P30 parents of Pilot L | `runs/p30_s{H}`, 20 | 5 |
  | Pilot L GT students | pl_gt, 20 | 11 |
  | Pilot L T8 students | pl_t8, 20 | 11 |
  | CLR1 | pclr1, 20 | 5 |

**Disclosure.** Smoke models outside the test set. Code paths were checked; the donor ages were extended after the first
smoke showed that the root route confounds age with the extra branches.

| Model | f | G by age (2 / 3 / ⌈f/2⌉ / f−1) | dG | G_root | AR closes |
|---|---|---|---|---|---|
| p30_s100 | 5 | 0.37 / 0.78 / – / 0.96 | +0.59 | 0.92 | 0.47 |
| pn_gt18_s102 (GT18 student) | 5 | 0.14 / 0.18 / – / 0.23 | +0.09 | 0.60 | −0.08 |
| pn_gt18_s102 | 11 | 0.15 / 0.22 / 0.28 / 0.48 | +0.33 | 0.77 | 0.17 |
| p30_s100 | 11 | 0.38 / 0.83 / 0.93 / 1.02 | +0.64 | 1.03 | 0.38 |
| pclr3_s102 | 5 | 0.71 / 0.41 / – / 0.47 | −0.24 | 0.60 | 0.01 |
| pclr3_s102 | 11 | 0.39 / 0.21 / 0.78 / 0.82 | +0.43 | 0.71 | 0.61 |

At f = 5, ⌈f/2⌉ = 3, so that column is empty.

**Predictions** (medians over informative models):
* **NF1 (primary: does H-NF hold for distilled students?)** pl_gt at f = 11:
  - normal form if G_10 ≤ 0.25 and dG ≤ 0.10;
  - age-causal (the competing account) if dG ≥ 0.25;
  - otherwise partial.
  - My expectation, from the GT18 smoke and §41: age-causal at f = 11, with interchangeability only well inside the
    trained range.
* **NF2 (parents).** p30 at f = 5: dG ≥ 0.25. Age is causal, as P10e and §36 predict.
* **NF3 (age-only reset).** In the parents, AR closes ≥ 0.5 of the oldest donor's gap. Expectation: not met (smoke
  0.47; §36 rescued only 9/34), i.e. age is not a single mean direction.
* **Descriptive.**
  - CLR1: H-NF puts CLR's normal form in the written channel, so its latents need not be canonical.
  - T8 students: X2 says they extrapolate only when re-anchored.
  - G_root vs G_{f−1}: does the route's extra history matter beyond its age?
  - T0: the root episode continued.

**Readings.**
* **NF1 normal form.** Distillation installs path independence in the latent; F6 would train it directly.
* **NF1 age-causal.** Repaired students extrapolate *despite* a causal age, i.e. a slower count. The normal form then
  lives only in written tokens. Canonical transport (restarts, F1's labels) is the mechanism, and F6 is the only
  single-loop route aimed at age itself.

**Run and score (experiments session).**
* Run: `python theory/exp28_interchange.py`: 80 models, f = 5 and 11, about 40 s per model on 3 CPU threads →
  `theory/results/exp28_interchange.json`.
* Score: per family at its primary f, medians over informative models of `G_by_age[f−1]`, `dG` and `AR_closed`.

**Result (2026-10-01; scored by the experiments session, 298c9f7, `theory/results/score_exp27_28.json`; Audit
reproduced G, dG and AR_closed from the stored IIA). NF1 normal form, against my stated expectation; NF2 and NF3 not
met.**

| Family (f) | Informative | G_{f−1} | G_2 | dG | AR closes | Residual bal CT / RT_old / T0 |
|---|---|---|---|---|---|---|
| pl_gt, GT students (11) | 20/20 | **0.09** (≤ 0.25 in 14) | 0.05 | **+0.01** | −0.25 | 0.98 / 0.95 / 0.70 |
| p30 parents (5) | 20/20 | 0.47 | 0.36 | +0.11 | 0.17 | 0.74 / 0.68 / 0.52 |
| pl_t8, T8 students (11), descriptive | 20/20 | 0.84 | 0.25 | +0.47 | 0.54 | 0.87 / 0.62 / 0.51 |
| pclr1 (5), descriptive | 19/20 | 0.50 | 0.39 | +0.16 | 0.11 | 0.73 / 0.67 / 0.52 |

* **NF1 (GT students, f = 11).** Normal form: G_10 ≤ 0.25 and dG ≤ 0.10. An age-10 latent works like a fresh one.
* **NF1's limit.** Root-route donors still differ (G_root 0.51): the GT latent is age-invariant, but not invariant to a
  history that visited other branches.
* **NF2 not met.** Parents at the registered f = 5: dG +0.11. Their latents are already partly bound to context at a
  young age (G_2 0.36).
* **NF2 at f = 11 (descriptive, as Audit asks).** dG +0.52. Caveat: parents do not reliably reach c_11, so old donors
  there also lose content.
* **NF3 not met.** The age-only reset closes 0.17 of the gap: age is not one mean direction.
* **Caveat on AR closes.** It is undefined where the old donor's gap is ≈ 0, because it divides by 1 − IIA_old. That
  happens for 6/20 pl_gt (median −0.07 over the other 14) and 1/20 pl_t8 (0.57 without it). It is uninformative for
  pl_gt, which has no gap to close. p30's 0.17 and pclr1's 0.11 are unaffected.
* **The contrast that matters (Audit's point (a), corrected on one fact).**
  - The normal form appears with true-label distillation (pl_gt) and not with label-free re-anchored distillation
    (pl_t8, dG +0.47, like p30 and pclr1).
  - pl_t8 students do **not** extrapolate in one loop: one-loop d13–18 median 0.50, as in X2 (2/20). They reach depth
    only when re-anchored (14/20, §37), i.e. through canonical transport by written tokens.
  - So the normal-form account holds for the GT students' latents. The label-free route reaches depth only by restarts,
    where the normal form is the written token.
* **Exploratory, post hoc (not registered).** Across the 40 students, G_10 at f = 11 vs one-loop bal-acc at d13–18
  (§37's `single24`): Spearman −0.87 (permutation p < 2·10⁻⁴). Within GT −0.49 (p 0.03); within T8 −0.77
  (p < 2·10⁻⁴).
  - Latent path independence tracks single-loop reach, model by model.
* **Consequence.**
  - This supports H-NF as the mechanism of single-loop extrapolation, and raises the prior for F6 (§58, running), which
    trains age invariance from shallow data without true deep labels.
  - The open question becomes why label-free targets (T8) do not install it.
* **Process (Audit).** `exp28_interchange.py` stores IIA but not the per-query answers. From now on it also saves them
  (`--answers_dir`), so IIA can be recomputed.

## 57. Origin of the ProsQA adjacency cue (2026-09-30 22:28 EDT; data statistics, exploratory; asked by the experiments session after its cue test)

**Question.** The cue test (`docs/PROTOCOL8B_STEP1.md`, dd5ce92) showed that all 6 Coconut GPT-2 step-1 models treat an
option adjacent to an entity named in a "<root> is a X" statement as the wrong one. Where in the training distribution
does that come from?

**Analysis.** `theory/cue_origin.py` → `theory/results/cue_origin.json`. ProsQA problems name 2–4 persons, and only one
is the root. For each problem, take the shortest distance from *any* person to the target (dt) and to the negative
option (dn).

**Result.** The generator hangs the negative option off another person.

| Data | n | neg reachable from another person | dn < dt | tie | neg adjacent | target adjacent | P(neg \| one adjacent) | proximity heuristic |
|---|---|---|---|---|---|---|---|---|
| prosqa_train | 17,886 | 1.00 | 0.85 | 0.12 | 0.51 | 0.03 | 0.97 | 0.91 |
| prosqa_test | 500 | 1.00 | 0.84 | 0.14 | 0.51 | 0.03 | 0.98 | 0.91 |
| p8_train | 12,000 | 1.00 | 0.83 | 0.15 | 0.10 | 0.01 | 0.96 | 0.90 |
| p8_test | 4,057 | 1.00 | 0.82 | 0.16 | 0.08 | 0.01 | 0.96 | 0.90 |

* The negative is never reachable from the root, and always reachable from another person.
* "Adjacent" means one rule below a person-named concept.
* Proximity heuristic: answer = the option farther from every person statement, ties broken by coin. It is
  person-agnostic and needs no reasoning from the root. Its expected accuracy is 0.90–0.91, including 0.86–0.96 at each
  original depth 3–6.

**Reading.**
* The models learned a true regularity of ProsQA's generator: "the option close to a concept named in a person
  statement is the distractor". It is valid in 97 % of the problems where exactly one option is adjacent, and it ignores
  *which* person is named.
* That explains both cue-test signs. PT puts the target one rule below a person statement, so the target looks like
  the distractor. PN does the same to the negative, so the models answer "target" even when both options are derivable.
* **Consequence.** On ProsQA, accuracy up to ~0.9 is available without reasoning from the root. Claims that rest on
  ProsQA accuracy need a control that keeps the answer fixed while moving the cue.
* **Proposed standard control.**
  - Add "<another person> is a <a parent of the target>". The answer is unchanged; a reasoner is unaffected, while the
    cue flips toward the negative.
  - Alternatively, report the anti-cue subset (dn ≥ dt, ≈ 16–18 % of test problems).
  - Added to §54b as amendment 54b-1.
* **Prior art (Literature, `docs/LIT_PROSQA_SHORTCUT_2026-09-30.md`, 028cef6).** This specific artefact is not documented.
* **Confirmed on trained models (PROTOCOL8 amendment 3, 10cc74c).** Removing the cue (p8_na_test) drops every latent
  arm to chance at D7–10 (Coconut 0.564 → 0.508), while text CoT holds (0.767).
  - Prior work shows only that ProsQA admits shortcuts: 2602.22441, 2604.06374, 2606.12689.
  - General precedent: multi-hop shortcuts and adversarial distractors (Jiang & Bansal 1906.07132).
  - A balanced re-generation of ProsQA would pair the finding with a fix.

## 58. F6 — age-transplant training: can shallow data teach a single loop to ignore its own age? (pre-registered 2026-09-30 22:36 EDT, before any F6 run; Literature `docs/LIT_F6F7_CHECK_2026-09-30.md`, c1ab04c)

**Idea (§52 addendum).**
* Prop. 10: a stop keyed to a count is free whenever no training residual outlasts the count.
* F6 makes training residuals outlast the age without building any graph deeper than the training depth.
  - A recipient episode receives, at step 2, a donor thought of age a. Its frontier is matched by node id.
  - The recipient must then continue up to 4 more hops, so its thought's age runs up to a + 5 ≤ 9 > 6.
  - Every supervised distance stays ≤ 5 from the recipient's START, so no deep labels are used (C4).
* **Prior (Literature).** Interchange intervention training (IIT, 2112.00826): F6 is IIT with variable = frontier and
  swap = an aged latent. Anil 2211.09961 uses only content-agnostic initial states.
* **New as scoped.** A content-matched aged transplant aimed at count-keyed stops.

**Implementation.** `theory/train_arch.py --transplant 1 --transplant_mode M --transplant_ages 2:4 --transplant_bs 128`
(`transplant_batch`, `transplant_step`).
* **Pairing.** Donor and recipient come from separate full-depth training graphs.
  - The donor starts at the head of a chain branch and runs a steps, giving frontier w.
  - In the recipient graph, the first node of a chain branch is renamed to w (a swap of two ids). The recipient starts
    at that branch's head.
* **Transplant.** The recipient's step-2 input is the donor thought.
* **Loss.** BCE on 8 recipient queries: the chain's nodes 2–5, reachable from START, plus 4 nodes of other branches.
* **Gradients.** They flow into the donor too, so the model may learn age-free thoughts or learn to ignore age.
* **Schedule.** The transplant runs as one extra eager optimizer step per training step, once the gated curriculum is
  at full depth. Same optimizer, `zero_grad(set_to_none=False)` so the CUDA graph's buffers persist.
* **Checks.**
  - CPU unit check: labels match BFS in 51/51 per mode; the donor frontier equals the recipient's node 1 in 51/51
    (aged, young) and 0/51 (mismatch).
  - CPU loop smoke: 200 steps run, and `loss_transplant` is logged.
  - On an existing model, outside any arm: aged donors already raise the recipient's loss (0.27 vs 0.03 young, pclr3_s102).

**Arms.** All use the P30 recipe exactly (`runs/p30_s102/config.json`: 30k steps, bs 512, gated curriculum, auxiliary
lookup, init gpt2, no step embedding, lr 5e-4). Seeds: the 10 even-indexed P30 failures (102, 111, 113, 118, 125, 130,
132, 140, 146, 157). 30 runs.

| Arm | Donor | Content | Role |
|---|---|---|---|
| F6-aged | a ~ U{2..4} | matched | primary |
| F6-young | a = 1 | matched | control: the transplant mechanics without age; the training counterpart of §56's age-only reset |
| F6-mismatch | a ~ U{2..4} | not matched | control: same-age, content-mismatched (Literature, Audit) |

**Required before the full launch.** A 300-step GPU smoke with `--curriculum ""`, so transplants are active at once,
on one card. Check that the CUDA-graph step and the eager transplant step coexist and that `loss_transplant` falls.
This is a plumbing check, not a result.

**Measures** (experiments session):
* `theory/eval_arch.py`: gate, delayed bal-acc, extrap_K12 (7-hop success: d7 at K′ = 12 ≥ 0.60 with CI lower bound
  > 0.55), far8_10_K12.
* §37's probe: one loop at K′ = 24, `exp18_deep_reanchor.py --runs`.
* §56's interchange: `exp28_interchange.py --runs ... --f 5`.

**Predictions** (counts among gated runs; medians over gated runs):
* **F6-1 (primary, C2 from shallow data).** extrap_K12 in F6-aged ≥ 6/10, and ≥ 3 more runs than F6-young and than
  F6-mismatch.
* **F6-2 (mechanism).** In F6-aged, §56's dG at f = 5 has median ≤ 0.10; in F6-young, ≥ 0.25. Aged training makes aged
  thoughts interchangeable.
* **F6-3 (no in-range cost).** Median delayed bal-acc of F6-aged within 0.02 of F6-young.
* **Falsified if** F6-aged ≤ F6-young + 1 on extrap_K12. A seam detector would show as F6-2 met with F6-1 failed: the
  recipients learn to handle transplants while natural long runs stay unchanged.
* **My expectation.** F6-1 met with a modest margin, with gains concentrated at d7–9. F6 does not touch the
  history-length part of the stop key (P10b′), which grows with K′ at test.

**Readings.**
* **F6-1 met.** Single-loop C2 from shallow labels, by training age invariance directly. This is the first fix that
  needs neither deep data (§41) nor written restarts (§26).
* **F6-1 failed, F6-2 met.** Age invariance holds on transplants but is not what limits natural runs. The
  history-length key or the step clock remains, so restarts or chunked histories stay necessary.
* **Both failed.** The thought-carried age cannot be trained out this way.

**Cost.** About 1–1.5 h per run on an A4000-class GPU, roughly +30–50 % over P30 from the eager transplant step: ≈ 40
GPU-h. Placement by the experiments session: local cards after §54b, or Vast within budget, about $5–8.

**Result (2026-10-01; scored by the experiments session, 93feb6c, `theory/results/score_f6.json`; Audit reproduced; all
30 runs gated). Falsified by its own rule; F6-2 not met; F6-3 met.**

| Arm | extrap_K12 | d7 (K′ = 12) median | dG at f = 5, median (n informative) | One-loop d13–18 median |
|---|---|---|---|---|
| F6-aged | 6/10 | 0.61 | **−0.04** (8) | 0.51 |
| F6-young | 6/10 | **0.72** | +0.14 (10) | 0.52 |
| F6-mismatch | 5/10 | 0.57 | +0.06 (10) | 0.50 |

* **F6-1 not met.** Aged ≤ young + 1, so the registered falsifier holds.
* **F6-2 not met.** Aged dG −0.04 meets its half; young's +0.14 is not ≥ 0.25. F6-3 met: delayed 0.919 vs 0.921.
* **Reading (with Audit's).**
  - Aged transplants did train age invariance (dG −0.04). It bought no reach: one-loop d13–18 is at chance in every
    arm, and the young (age-reset) control has the best d7.
  - So any gain from the fine-tune comes from the transplant mechanics or the extra training, not from exposure to aged
    states.
* **Consequence for §56.** Training age invariance directly does not produce single-loop reach. The across-student
  association (ρ −0.87) is therefore not causal in the direction "age-free latent → reach". In GT students, path
  independence looks like a co-product of what deep supervision teaches, not its mechanism. H-NF is weakened as the
  account of extrapolation. §62 tests one candidate cause of the co-variation: calibration from depth-structured labels.
* **Design limitation.** There is no plain (non-transplant) eager arm. Extrapolation of 5–6/10 among seeds that failed
  as P30 runs therefore cannot be read in absolute terms: they are selected failures re-drawn with new numerics, and
  P30's base rate is 25/59. The registered aged-vs-young contrast is unaffected.

## 59. ProsQA-NA: the §57 cue neutralised without changing any answer (2026-09-30 22:42 EDT; data, exploratory)

**Construction.** `theory/prosqa_na.py SRC DST`. Outputs go to `prosqa/p8data/` (gitignored; regenerate with the script,
seed 0).
* **Distance equalisation.** The two options' distance from the nearest person statement is equalised at
  L = min(dt, dn, 2). This adds "<another person> is a <option or a parent of it>."
* **In-degree equalisation.** The options' in-degrees are equalised with "Every <C> is a <option>.", where C is a
  concept the root cannot reach.
* Statements go at random positions. The answer, the root's reachable set and the target's depth are asserted
  unchanged per problem.

**Residual-cue check.** Expected accuracy of shallow heuristics: the option farther from every person / with more
parents / more mentions / mentioned first / mentioned last / whose parents have higher in-degree.

| Data | n | proximity | parents | mentions | first | last | parents' in-degree |
|---|---|---|---|---|---|---|---|
| prosqa_test (original) | 500 | 0.91 | 0.66 | 0.66 | 0.59 | 0.55 | 0.52 |
| prosqa_na_test | 496 | 0.50 | 0.50 | 0.50 | 0.50 | 0.51 | **0.62** |
| p8_na_test | 4,053 | 0.50 | 0.50 | 0.50 | 0.51 | 0.51 | 0.50 |
| p8_na_train | 11,985 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

* **p8_na is clean on every check.** Few problems were dropped: 4 from test and 15 from train.
* **prosqa_na keeps a second-order cue (0.59–0.62).** It comes from the added parents' in-degree. A fully
  exchangeable re-generation (Literature: "balanced re-generation") would remove it. I am not iterating further on data
  patches.

**Uses.**
1. **Cue reliance of existing models.** Accuracy on p8_na_test vs p8_test for the same model (Phase 0, PROTOCOL8b
   step-1, PROTOCOL10/11). A model that reasons from the root loses nothing; a cue model falls toward its
   reasoning-only accuracy.
2. **Training data** for any future ProsQA-based latent-reasoning claim (C2/C3/C7), so that ~0.9 is not available
   without reasoning.

## 60. Boundary memory, Phase A — can a learned worker compose Boolean boundary-response tables? With the Phase B gate fixed in advance (pre-registered 2026-10-01 00:24 EDT, after Literature's check 503df27 and before any implementation or run)

**Source.** `docs/PROPOSAL_BOUNDARY_MEMORY_2026-10-01.md` (81a8453): the user's "try this out", relayed by Audit. It is
the full version of external direction 1 (§52 addendum). This section registers its Phase A (local algebra) and fixes
its Phase B gate now. The experiments session implements and runs it. Its own gates and falsifiers are kept; this is one
proposal, not a fix loop.

**Prior art** (`docs/LIT_BOUNDARY_MEMORY_CHECK_2026-10-01.md`). New only as a combination.
* Learned merge: Divide and Conquer Networks 1611.02401.
* Associativity bias: Learnable Commutative Monoids 2212.08541 and MLP-LDRU 2605.26035. LDRU's associativity-biased
  operators give OOD length generalisation, which supports a directional prediction here.
* Exact port algebra: Kron reduction / Schur complement 1102.2950.
* Learned composition over subdomains: Mosaic Flows 2104.10873 and LatentDDM 2609.03069.
* Also NBFNet 2106.06935 and neural coarsening 2102.01350.

**How the Boolean-response design handles the count (my min-plus caveat, §52 addendum).**
* **Min-plus tables bring a count back.** A min-plus (shortest-distance) table stores path lengths that grow with
  region size, so the interface itself carries a hop count — exactly the free stop key of Prop. 10.
* **Boolean tables cannot, when exact.** An exact Boolean table is a reachability relation over ≤ 4 ports. Its range is
  finite and independent of region size, so a 1-edge region and a 20-edge region can have the identical table.
* **Where age can still live: the learned approximation.**
  - Sub-threshold values (e.g. 250/255 vs 247/255 fading with depth): a confidence clock, i.e. Mechanism 2.
  - Which regions exist: workspace structure.
* **How each channel is handled.**
  - Committed codes are quantised to uint8 after every commit.
  - The merger and answer head get no age, size, depth or creation-order feature.
  - Phase A measures the channel: the age-decodability audit and the binarised-commit check below.
  - Phase B tests its causal effect directly (B4, history equivalence).
  - A learned deadline can only enter with the Phase C controller, which is out of scope here.

**Phase A design** (the experiments session implements; spec fixed here).
* **Task data** (generator with a reference BFS; the model never sees BFS).
  - Small directed acyclic regions of 2–8 edges, from 8 motif families: path, out-tree, in-tree, diamond, chain with
    skip edges, random sparse DAG, ladder (two parallel paths with rungs) and zigzag (alternating edge directions).
  - **Held-out motifs:** ladder and zigzag, never trained.
  - Each region declares pinned terminals P (|P| ∈ {2, 3, 4}). Only regions with a legal schedule are kept: every
    intermediate region's boundary (vertices with an edge outside it, plus pinned ones) must have ≤ 4 ports. Report the
    rejection rate per family.
* **Terminal labels only.**
  - Each task is one region and one label: an entry T_R[a, j] of the *final* table (seed pattern a ≠ 0 over P, output
    port j), balanced 50/50.
  - |P| = 2 tasks with a = {root} and j = query are the ordinary reachability answers. |P| = 3–4 tasks are the
    independent boundary probes. Their counts are reported separately.
  - No intermediate table, frontier or path label is used.
* **Schedules.**
  - Training: uniform random legal schedules with no merge whose two children both have ≥ 3 edges.
  - **Held-out merge orders:** schedules with at least one merge of two composites, each with ≥ 3 edges (regions of
    6–8 edges). This separates a learned composition *operator* from a learned one-edge-at-a-time *sweep*.
* **Model.**
  - **Leaf encoder.** Maps one edge u → v, with its two ports at randomly assigned local slots, to a 16 × 4 table
    (learned; true leaf tables are never inserted).
  - **Merger.** A shared 2-layer d = 256 transformer over ≈ 64 tokens: 32 child row tokens (seed bits, 4 response codes,
    local vertex-id embeddings of the 4 slots, child role), up to 16 parent row queries, and wiring tokens (which child
    slots are the same local vertex, which survive as parent ports).
  - Local vertex ids are randomly permuted per call. There is no global id, age, size, depth or order feature, and
    activations are fresh per call.
  - **Output.** Sigmoid, padding masked to 0, then Q8 = round(255·p)/255 with a straight-through gradient (disclosed).
* **Arms (4 seeds each, the same seeds in both).**
  - **BM-oc (primary).** Terminal loss + λ_order × masked MSE between the final tables of two different legal training
    schedules of the same region, aligned by port. Both branches count as training compute.
  - **BM-no.** λ_order = 0.
  - **Exact composer (diagnostic ceiling only).** An exact Boolean fixed-point composer (Kron reduction over the Boolean
    semiring) on exact leaf tables. It must score 1.000 on every split; anything less is a data or bookkeeping bug.
* **Fixed before the seed panel (development pilot on a separate development set, logged in the run directory).**
  - λ_order ∈ {0.1, 1}, chosen by development accuracy on training motifs and orders only.
  - Steps and batch from a throughput pilot, identical for both arms.
  - Failed or unfinished seeds are not dropped and get no extra training.

**Measures** (held-out evaluation sets; true tables are used for evaluation only):
1. **Probe accuracy.**
   - (a) Held-out instances: training motifs, training-type orders, 2–8 edges.
   - (b) Held-out motifs.
   - (c) Held-out merge orders.
2. **Semantic validity** of thresholded final tables:
   - entry accuracy vs the true tables;
   - the share of tables satisfying all four reachability-closure laws: extensivity f(a) ≥ a; monotonicity;
     idempotence f(f(a)) = f(a); union preservation f(a ∨ a′) = f(a) ∨ f(a′).
3. **Residual associativity error** (Literature, Audit), on held-out regions under 2 held-out orders: the mean
   |T₁ − T₂| over aligned valid entries (code/255), and the thresholded disagreement rate.
4. **Age-channel audit (descriptive).**
   - Group held-out regions by identical true table.
   - Cross-validated decodability of merge-tree depth and edge count from the committed codes, vs a majority baseline.
   - Mean margin |T − 0.5| by merge depth: does confidence fade?
5. **Binarised-commit check (descriptive).** Accuracy when every commit is thresholded to {0, 255}. A drop > 0.02 means
   the model relies on sub-threshold values.
6. **Size preview (descriptive).** Regions of 9–16 edges, i.e. more merges than trained.

**Gate A (advance to Phase B).** BM-oc in **every one of the 4 seeds**:
* **A1.** Probe accuracy ≥ 0.95 on held-out instances (1a).
* **A2.** ≥ 0.95 on held-out motifs (1b) and ≥ 0.95 on held-out merge orders (1c).
* **A3.** Entry accuracy ≥ 0.95, and ≥ 95 % of tables satisfy all four laws.

**Directional prediction (LDRU precedent; reported, not gating).**
* **PA-dir.** BM-oc beats BM-no on held-out merge orders (1c) and has lower residual associativity error (3), in ≥ 3 of
  4 seed pairs.
* Expected magnitude: BM-no fails 1c (a sweep rather than an operator) more often than it fails 1a.

**Falsifiers and decisions** (the proposal's §15, fixed now):
* **Only the exact composer passes 1a.** Learned composition or credit assignment is inadequate. Stop: no Phase B, no
  GPT-2.
* **A1 passes, A2 orders fail in both arms.** The worker learned a sweep, not an operator. Stop. Phase B's arbitrary
  schedules need composite merges.
* **A1 and A2 pass, A3 fails.** Accurate answers without a valid reachability algebra. Stop, and report it as
  "accurate but not semantic".
* **Gate A passed by BM-oc.** Run Phase B as fixed below, with BM-no carried as the comparison arm whatever its Phase A
  result.
* **Considered, not adopted.**
  - *Overlap ablation* (Mosaic Flows). Regions here already share boundary vertices, and Boolean composition over
    shared ports is exact without further overlap. Mosaic Flows' need arose from continuous boundary data.
  - *DCN's complexity regulariser.* Not added.
  - Neither is a pre-declared rescue.

**Result of the registered §60 panel (2026-10-01; scored by the experiments session, 13bcbe7,
`bm/runs/score_panel_a.json`; note 60-3 code, λ 1, 10k steps × bs 128). Gate A not passed: 0/4 seeds in both arms.**
* All 8 runs equal each split's exact identity-map baseline: instances 0.877, motifs 0.911, orders 0.891, size 0.895.
* Entry accuracy is 0.839. Laws score 0.99–1.00 only because the identity satisfies them. PA-dir 1/4.
* **Reading.** The development pilot's collapse to the identity map holds for every seed and both arms: nothing beyond
  the trivial entries was learned.
* The one revision, §60b, was fixed before this panel, and its result (Gate A 0/4) ends the line at Phase A.

**Phase B gate, fixed now** (the proposal's §11–12; its own registration amendment will fix only implementation
quantities after a throughput pilot: steps, batch, deep-set size).
* **Task.**
  - Fixed 64-node branch-and-decoy graphs, restricted to legal executions with ≤ 4-port interfaces.
  - Edge count, node vocabulary, input length and degree/decoy distributions matched across depth conditions.
  - A legal schedule derived from topology without reachability labels or edge directions. It is a disclosed scaffold.
  - If the graphs cannot fit the cap or cannot be learned under H, that is reported as a limitation. No path-only
    substitute.
* **Training.**
  - H = 16 atomic merges per trajectory.
  - Terminal answer loss through the always-available answer head over the region collection, plus order consistency
    and boundary probes.
  - Arms BM-oc and BM-no, the same 4 seeds.
* **Evaluation.** The same frozen balanced deep set (including 19–24-hop queries where feasible) at K = 16, 24, 32, 48.
  Merge calls, maximum merge ancestry and answer-head cost are counted separately.
* **Gates** (BM-oc; all must hold to advance to Phase C):
  - **B0.** ≥ 95 % held-out accuracy at training depths, in every seed.
  - **B1 (C2).** Paired accuracy gain K16 → K32 on the deep set ≥ +10 points in every seed. Each seed's one-sided
    98.75 % cluster-bootstrap lower bound (graphs as clusters) must be > 0 (Bonferroni over 4 seeds).
  - **B2 (no reversal).** No adjacent-budget loss > 2 points across 16 → 24 → 32 → 48. Deep-negative false positives
    are reported.
  - **B3 (C1).** Replaying the same schedule with compatible sham tables costs ≥ 20 points of deep accuracy (several
    sham types reported).
  - **B4 (mechanism: history equivalence).**
    - Take regions with the identical true boundary response but different construction histories (structure, merge
      depth or merge order). They are matched by held-out truth tables, evaluation only.
    - Inserted into the same held-out continuation, the paired downstream accuracy difference must have its 90 % CI
      inside ±2 points.
    - This is the boundary-memory counterpart of §56's interchange test.
* **Readings** (the proposal's §15).
  - Flat beyond H: C2 rejected in this configuration.
  - Budget gains but B4 fails: more computation helps, but history still matters, so the age-invariance claim is
    rejected.
  - B3 fails: the answer head solves without merger content, so C1 fails.

**Cost and placement.** Phase A is a 2-layer d = 256 model with ≈ 64 tokens per call. The proposal targets ≤ 8 GB;
wall time is unmeasured. The throughput pilot sets the hours-scale allowance. 8 runs (2 arms × 4 seeds), on local cards
or Vast at the experiments session's discretion. CPU suffices for the data generator, the exact composer and unit tests.

**Amendment 60-1 (2026-10-01 00:30 EDT, before any implementation or run; Literature's addendum to the check, 9069f67
00:30:29).**
* **Added priors.**
  - Transformers Learn Shortcuts to Automata 2210.10749 (must-cite). The Boolean 16 × 4 tables form a transition-summary
    semigroup, and learned log-depth compositions there were brittle out of distribution.
  - Learned interface conditions in domain decomposition (Taghibakhshi 2205.09833, the closest learned interface
    operator).
  - Learned AMG transfer operators (2003.05744, 1902.10248).
  - Scans: 1709.04057, Mamba 2312.00752, 2411.12537, 2607.20594.
  - None does Boolean reachability or latent reasoning.
* **Numeric held-out-order thresholds** (Liu et al. predict in-distribution fit with unseen-order failure).
  - **Phase A.** Unchanged: A2 requires ≥ 0.95 on the composite+composite held-out orders in every seed. The
    residual associativity error is reported per seed.
  - **Phase B, new gate B5.** On the training-depth evaluation set, accuracy under a held-out schedule family ≥ 0.95 in
    every seed. The family: legal schedules from the same topology rule, but with merges of two composites (each ≥ 3
    edges) that training never contains.
  - Advancement to Phase C now needs B0–B5.

**Amendment 60-2 (2026-10-01 00:31 EDT, before any implementation or run; Audit).** B5 also requires ≥ 95 % agreement of
thresholded answers between the training-family and held-out-family schedules on the same instances, in every seed.
This mirrors Phase A's held-out-order gate.

**Implementation note 60-3 (2026-10-01 00:56 EDT; Theory's review of the experiments session's build in `bm/`, before
the seed panel; no panel data exists).**
* **Accepted.**
  - Leaf ports follow the boundary rule.
  - Wiring tokens carry in-A/in-B/in-P flags.
  - The age audit is a ridge-CV decoder within truth-matched groups (n ≥ 20).
* **Changed before the panel.**
  - **(i) The root's slot is permuted per call like every other vertex.** A pinned root slot would be a role channel,
    which the proposal (§6) excludes. The host still binds root and query for readout.
  - **(ii) Training labels are sampled over all valid entries, including the trivial ones (j ∈ a), then balanced 50/50
    by label.** Otherwise A3's extensivity law would be tested only on entries never seen in training.
* The λ_order / steps development pilot is re-run on the fixed code, at the experiments session's discretion, before
  settings are frozen.

### 60b. Boundary memory, Phase A — the one revision: extensivity by construction and a size curriculum (pre-registered 2026-10-01 02:26 EDT, after the development pilot 0b9f9a6 and before the seed panel and the 2–3-edge diagnostic's result)

**What the development pilot showed** (`bm/runs/DEV_LOG.md`, development set only).
* BM-oc at λ 1 and λ 0.1 gave identical metrics: probe 0.874, entry 0.839, laws 1.000. Training accuracy was flat at
  0.88 from step 2k to 10k.
* 0.874 is exactly the identity map f(a) = a. Trivial entries (j ∈ a) are 37 % of the labels and are always 1. The
  identity predicts every non-trivial entry 0, and 80 % of those are negatives: 0.37 + 0.63 × 0.80 = 0.874.
* The identity map also satisfies all four laws, so the laws score was uninformative.
* An exact operator pushed through the same batched runner scores 1.000, so this is not a bug.
* **Cause, owned by Theory.** Note 60-3's fix (ii) put the trivial entries into the training labels. That made the
  identity map an 87 %-accuracy minimum. As in §54b, this is an optimisation-design failure, not evidence about the
  hypothesis. This is the one revision; whatever it gives is Phase A's result.

**Changes from §60.** Everything else is unchanged: data, motifs, held-out splits, arms, seeds, measures, Gate A,
falsifiers, the Phase B gate and amendments 60-1/60-2.
* **(1) Extensivity by construction.** Each committed table is T[a, j] = max(a_j, Q8(σ(z_aj))), i.e. seeded ports
  always respond (zero-length paths, proposal §4).
  - This is a fixed elementwise OR with the input seed, the counterpart of F1's monotone OR. It is not a reachability
    solver.
  - It does enforce one audited property, so extensivity leaves A3. A3 now uses the other three laws: monotonicity,
    idempotence, union preservation.
  - Disclosed deviation from proposal §4 ("no projection to enforce those properties").
* **(2) Labels.** Training and probe labels are drawn over non-trivial entries only (j ∉ a), balanced 50/50. Trivial
  entries are constant by construction.
  - The identity map now scores 0.50, so its basin earns nothing.
  - A1/A2 thresholds (≥ 0.95) now apply to non-trivial entries, which is stricter than before.
* **(3) Size curriculum (terminal labels only; label budget unchanged).**
  - Training regions have 2–3 edges for the first 20 % of steps, then the maximum size widens linearly to 8 edges by
    50 %, then stays at 2–8.
  - Order-consistency pairs start once regions of ≥ 3 edges appear.
* **Pilot.** The λ_order ∈ {0.1, 1} / steps / batch development pilot is re-run on the revised code under §60's rules
  (development set, training motifs and orders only). Settings are frozen before the panel.
* **Added report.** Non-trivial-entry accuracy beside the identity baseline (0.50) for every split and seed.
* **Disclosure.** A 2–3-edge diagnostic on the old code (λ 0, 2k steps) is running. This revision was fixed before its
  result, and the result will be reported without changing anything here.

**Decision rule, fixed now.** If BM-oc fails Gate A under §60b, §60's falsifiers apply as written ("only the exact
composer passes", "sweep not operator", "accurate but not semantic") and the boundary-memory line stops at Phase A.
There is no further revision.

**Amendment 60b-1 (2026-10-01 02:26 EDT, before any panel; Audit).**
* **Order.**
  - First, the §60 panel as registered (with note 60-3): 2 arms × 4 seeds, settings from the development pilot that has
    run. It gives the registered verdict; the development pilot cannot stand in for it.
  - Then the §60b panel.
* **The revision's two suspected causes, both named before any panel data.**
  - (a) The identity map as a strong local optimum while 37 % of labels are trivial positives. Change: (1) + (2), one
    coupled change. Building extensivity in makes trivial entries constant, so they leave the labels.
  - (b) Terminal-only credit assignment through multi-merge trees. Change: (3), the size curriculum.
  - Both are in this single revision so that there is no second one.
* **Reference line.** Both panels report the identity-map baseline per split, computed exactly from that split's labels
  (§60: ≈ 0.87; §60b: 0.50 on non-trivial entries). A pass is judged against it as well as against Gate A's 0.95.
* **The 2–3-edge diagnostic, reported as promised (old code, `bm/runs/diag_k3.void`).** The same plateau: training
  accuracy 0.880–0.887 at steps 500–2000. With trivial labels in the mix, even single merges are not learned. This
  changes nothing in §60b, which was fixed before it.

**§60b result (2026-10-01; scored by the experiments session, 5901972, `bm/runs/score_panel_b.json`). Gate A not
passed: 0/4 seeds, failing A1, A2 and A3 in every seed. PA-dir not met (1/4).** Run alongside the still-running §60
panel, on a freed card; their order does not affect either verdict.

| Arm, seed | Instances | Motifs | Orders | Size 9–16 |
|---|---|---|---|---|
| Identity / chance baseline | 0.51 | 0.51 | 0.50 | 0.53 |
| BM-oc s1 / s2 | 0.63 / 0.63 | 0.56 / 0.55 | 0.61 / 0.61 | 0.60 / 0.61 |
| BM-oc s3 | **0.88** | **0.86** | **0.84** | 0.77 |
| BM-oc s4 | 0.79 | 0.73 | 0.71 | 0.67 |
| BM-no, all 4 seeds | 0.61–0.63 | 0.54–0.55 | 0.58–0.62 | 0.59–0.61 |

Non-trivial-entry probe accuracy.
* **A3 fails everywhere.** Monotonicity, idempotence and union preservation hold together in only 19–41 % of tables
  (extensivity by construction).
* **Order consistency.**
  - It is what lets learning start. 2/4 BM-oc seeds leave the ~0.6 plateau, and no BM-no seed does. Held-out orders
    are learned about as well as instances (s3: 0.84 vs 0.88), so the partial learner is an operator, not a sweep.
  - The learners have *higher* associativity error (s3/s4 0.086/0.072 vs 0.026–0.041). The stuck seeds' near-constant
    tables agree trivially, so that comparison is confounded, and PA-dir counts it against them.
* **The predicted age channel appears in table values.**
  - In both learners, mean confidence |T − 0.5| fades with merge depth: 0.48 → 0.38 (s3) and 0.48 → 0.35 (s4).
  - Thresholding every commit costs 0.05 (s3: 0.884 → 0.832), so the learners rely on sub-threshold values.
  - Merge depth is still not decodable within truth-matched groups (0.29 vs majority 0.32).
  - So the confidence clock (Mechanism 2) that the §60 count analysis warned of shows up as soon as composition is
    learned at all.
* **Decision (fixed in §60b).**
  - The falsifier "only the exact composer passes" applies: terminal-only training does not learn the Boolean
    composition algebra in this configuration.
  - **The boundary-memory line stops at Phase A.** No Phase B, no GPT-2, no further revision.
  - The registered §60 panel's verdict will be added when it ends.
* **What it leaves for the frontier.** Order consistency is the ingredient that makes composition learnable at all,
  in line with LDRU. The learned tables fade with depth, i.e. the count returns as fading confidence. Neither is
  pursued further here.

## 61. F1-S — learning F1's one-hop label rule from shallow final answers on separate graphs (pre-registered 2026-10-01 06:30 EDT, after Literature's check 07f68a9 and Audit's C4 conditions, before any run)

**Why.**
* §54b: GPT-2's entity-label channel runs an exact, readable, halting, cue-free BFS to d14 when per-pass targets teach
  the one-hop rule.
* From ProsQA's deep final answers alone (depth ≥ 3) it learns nothing.
* F1-S adds final-answer questions of depth 1–2, so the rule can be learned where one pass is the whole problem.

**Prior art** (`docs/LIT_F1S_CHECK_2026-10-01.md`).
* Low novelty as a method. Fan 2409.15647 uses the same recipe for looped transformers: loss only at T(n), mixed
  lengths.
* Ye 2510.19753 predicts §54b: within-capacity data teaches the algorithm, beyond-capacity data teaches heuristics.
* Also Bansal 2202.05826, Kohli 2604.07822, Saparov 2412.04703 (needs a balanced depth mix) and 2405.15071.
* New only in scope: a discrete per-entity label channel in a pretrained LM, learned from mixed-depth answers and
  iterated deeper. Nearest: EntNet 1612.03969 (continuous), 2305.02363, DiscoLoop.

**C4 conditions (Audit, verbatim).**
1. "No graph instance contributes both a shallow and a deep training question", checked in code.
2. "Deep test problems must run at a fixed P (or the model's own halt), never at P = true depth." Test P is registered
   below.
3. "Disclose the number of extra final-answer labels and match total final-answer supervision against the original
   arm. Report the depth mix."

With these it counts as final answers only, from a disclosed generator-shaped distribution: the same class as Pilot L's
deep graphs.

**Design.** The experiments session implements it on the frozen §54b code (`prosqa/ail_gpt2.py` 02a03ac, AI-soft
settings: τ 1 → 0.3 over two thirds, λ ramped 0 → 0.1 over the last third, lab_std 0.144, bias −4, bs 4 × accum 2,
lr 5e-5 / 1e-3), plus a question generator.
* **Instances.** p8_na_train problems of original depth ≤ 6 (§59, cue-neutralised).
  - A fixed seeded split sends one third to *shallow* use and two thirds to *deep* use (their original question).
  - Each instance contributes exactly one training question. Condition 1 is checked by an assertion on instance ids.
* **Shallow questions.**
  - On a shallow instance, ask "Is <root> a <T> or <N>?" with T at distance d ∈ {1, 2} from the root (half each) and N
    the instance's unreachable negative. Option order is random.
  - Then re-run the §59 NA equalisation for (T, N), so the proximity and in-degree cues stay at chance. The answer is
    asserted by BFS.
* **Label count (condition 3).** Questions × epochs equal §54b AI-soft's (its depth ≤ 6 problem count × 2 epochs).
  - With a third of instances turned shallow, the mix is ≈ 1/6 d1, 1/6 d2 and 2/3 d3–6, in their original
    proportions.
  - It is reported exactly from the generated set. There are no extra labels: the same count is redistributed.
* **Passes.** Batches are bucketed by depth d.
  - Training uses P = d + U{0, 1, 2}, so even training does not hand over the exact depth.
  - Curriculum: d = 1 only for the first sixth of steps, then widening linearly to d = 1–6 by one third.
* **Test (condition 2).**
  - p8_na_test deep problems (depths 3–14) and held-out shallow questions (d = 1–2, built the same way on
    p8_na_test instances).
  - **All are run at a fixed P = 16**, with hardened labels (OR at 0.5).
  - Also reported: the model's own fixed-point pass (its halt), and, as a descriptive reference only, accuracy at
    P = d.
* **Seeds.** 1–3. **Reference arm:** §54b AI-soft (same code and label count, depths 3–6 only), 0/3.

**Predictions** (hardened labels, P = 16):
* **S1 (learnable).** Accuracy ≥ 0.95 at d1–6 in ≥ 2/3 seeds.
* **S2 (C2).** Accuracy ≥ 0.85 at d11–14 in ≥ 2/3 seeds: reach ≥ 1.8× the deepest training question, with no depth
  hint.
* **S3 (C7).** Label-flip edit IIA ≥ 0.8, as AG3.
* **S4 (C5).** Fixed-point pass vs depth, Spearman ≥ 0.8.
* **S0.** Soft-vs-hard gap ≤ 0.03, per run and per depth band (54b-1).
* **Descriptive.** The other-person cue-insertion drop; label precision and recall; accuracy at P = d vs P = 16.
* **My expectation.** S1 and S2 met, by Ye's within-capacity account and the hint ceiling. The main risk is §54b's
  σ ≈ 0.5 collapse recurring before the depth-1 signal acts.

**Readings.**
* **S1 and S2 met.** F1 reaches C2, C5 and C7 in GPT-2 from final answers (C4, with the disclosed depth mix). That is the
  first joint C2 + C4 + C5 + C7 result on a pretrained LM in this project. C3 (matched-compute text CoT) comes next.
* **S1 met, S2 failed.** The rule is learned but not iterated beyond the training depths: heuristics, not the algorithm
  (Ye).
* **S1 failed.** Even one-pass final answers do not teach the label rule at GPT-2 scale. F1's C4 claim stays on the toy,
  and this line closes there. No further revision.

**Result (2026-10-01; scored by the experiments session, daf3960, `results/f1s.json`, `docs/PROTOCOL12.md` §61; 1,486
steps × 8 = 11,888 presentations vs §54b's 11,892). S1–S4 not met (1/3 seeds each); S0 met. By the registered reading
the F1 line closes. One seed is an existence proof.**

| Seed | d1–2 | d3–6 | d7–10 | d11–14 | Label precision / recall | Edit IIA | ρ (fixed point, depth) | At P = d, d1–6 (descriptive) |
|---|---|---|---|---|---|---|---|---|
| s1 | 0.92 | 0.48 | 0.48 | 0.49 | 0.54 / 0.77 | 0.01 | −0.25 | **1.000** |
| s2 | 0.90 | 0.47 | 0.51 | 0.49 | 0.57 / 0.78 | 0.13 | −0.25 | 0.82 |
| s3 | **1.000** | **1.000** | **1.000** | **1.000** | 1.000 / 1.000 | 1.000 | 0.75 | 1.000 |

Hardened labels, fixed P = 16.
* **s3: an existence proof.** It learned, from final answers only (C4, with the disclosed depth mix and disjoint
  instances), an exact, readable (IIA 1.0), cue-free entity-label BFS in GPT-2 that extrapolates to d14 (2.3× the
  deepest training question) at a fixed budget.
* **s1 and s2: how they fail.** Both learned an imprecise rule whose false positives spread slowly.
  - Run at its trained budget (P = d), s1 is exact at d1–6 (1.000). At the registered P = 16 the labels flood
    (precision 0.54) and accuracy falls to chance.
  - Training with P = d + U{0,1,2} never ran long enough to penalise that slow flooding. This is Prop. 10's pattern
    in a new form: an error harmless within the trained budget, i.e. a rule that is only correct up to a count.
* **Lesson (recorded, not pursued).** A monotone public state reaches a true fixed point only if its local rule is
  exact. Training budgets must therefore include runs far longer than needed (P ≫ d), or slow flooding stays free.
* **Verdict.** S1 1/3, S2 1/3, S3 median 0.13, S4 median −0.25 (s3 itself 0.75 < 0.8).
  - Not met as registered: C4 learnability exists but is seed-dependent.
  - No further F1 variants, as committed.

### 61b. F1-S replication: does seed 3 reproduce, and how often does the frozen recipe succeed? (pre-registered 2026-10-01 13:41 EDT, before any replication run; user directive via Audit; Literature's note 08cbb19)

**Why.** §61: 1 of 3 seeds (s3) learned an exact, readable entity-label BFS in GPT-2 from final answers. Is that
success seed-intrinsic, or GPU-nondeterminism luck, and what is the recipe's success rate?

**Recipe: frozen and unchanged.** `prosqa/ail_f1s.py` at 248866f, the same `p8data/f1s_train.json`, the original GPT-2
init, 1,486 steps. Any fix for flooding seeds would be a separate registration.
* **(a) Exact reruns.** Seed 3, three times (`f1s_r3a`, `f1s_r3b`, `f1s_r3c`), on the original host's class: an RTX 3090, driver 570.x
  (s3 ran on 570.211.01, CUDA 12.4 image, PyTorch 2.5.1, bf16 autocast). Each host's GPU name and precision are recorded from the queue wrapper.
* **(b) Fresh seeds.** Seeds 4–13 (`f1s_s4` … `f1s_s13`).
* **Evaluations per run.**
  - The built-in `ail_f1s.py` eval (§61's measures).
  - `python -m bench.run --task prosqa_deep_na --model "f1s:prosqa/runs_ail/<run>?P=16" --per_depth 150` for the
    leaderboard.

**Success (fixed now, from s3's profile).** All three must hold on §61's evaluation:
* hardened accuracy ≥ 0.99 in every band (d1–2, d3–6, d7–10, d11–14) at P = 16;
* label precision and recall ≥ 0.99;
* edit IIA ≥ 0.90.

**Predictions.**
* **R1 (exact reruns).** Same GPU class: ≥ 2/3 seed-3 reruns succeed (seed-intrinsic). Different class: no prediction,
  since the trajectory can diverge.
  - Context: s2 also ran on a 3090 (driver 560.35.03) and flooded; s1 ran on a 3080 Ti. The hardware class alone did
    not decide §61.
  - Init and data order are pinned (`torch.manual_seed` and `np.random.default_rng` with the seed, fixed GPT-2 init),
    so arm (a) isolates GPU nondeterminism.
  - Literature's counter-expectation: one-bit changes produce diverse models (Summers & Dinneen 2103.04514).
    Nondeterminism alone gives variance comparable to seed variance (Zhuang 2106.11872).
  - Seed-bimodal success is common: Zhou 2402.09371, Zhang 2607.20594, Kuo 2606.29983, McCoy 1911.02969.
* **R2 (success rate).** About 0.3: 2–5 of the 10 fresh seeds succeed. Fewer than 2 would make §61's 1/3 an optimistic
  draw; more than 5 would make the recipe more reliable than §61 suggested.
* **Descriptive.**
  - The failure mode of each non-success: flooding at P = 16 with exact answers at P = d (as s1), or no learning.
  - The bench prosqa_deep_na accuracies.

**Readings.**
* **R1 met.** Success is a property of the seed: data order and init, not hardware noise.
* **R1 not met on the same class.** Success depends on non-deterministic numerics, i.e. a lottery even at a fixed seed.
* **R2** gives the recipe's reliability. A recipe-level fix (e.g. training budgets with P ≫ d against slow flooding,
  §61's lesson) would be its own registration.

**Run (experiments session).** One parallel Vast wave of 13 runs, ≈ $2.

**Note 61b-1 (launch record, 13:44 EDT; experiments session).**
* `ail_f1s.py` is byte-identical to 248866f.
* The three seed-3 reruns share the one available RTX 3090 with driver 570.211.01 (offer 31458818), running in
  parallel. The original s3 ran alone. Co-location can change the order of non-deterministic GPU reductions but not the
  arithmetic, so it is disclosed, not treated as a deviation from the class.
* Fresh seeds 4–11 run on four Vast hosts (drivers 550–570), and 12–13 on the local RTX 3070.
* host.txt is recorded per run.

**Result (2026-10-01; scored by the experiments session, 2fd44c9, `results/f1sb.json`; solo follow-up bf6b48a). R1 not met
(0/3), R2 not met (0/10). All 13 runs flood. The solo seed-3 rerun floods too.**
* **R1.** The 3 seed-3 reruns on the original class (RTX 3090, driver 570.211.01) are all flooding: label precision
  0.57–0.63, edit IIA ≤ 0.014.
  - The registered descriptive follow-up, f1s_r3d alone on the same class, also floods (chance at d3–14, precision
    0.57, IIA 0.19).
  - So s3's success was not a property of the seed and not caused by running three jobs on one GPU. It was a
    non-deterministic numerics draw, as Literature expected (Summers & Dinneen).
* **R2.** Fresh seeds 4–13: 0/10 succeed (precision 0.53–0.58).
* **Overall.** F1-S succeeds 1 time in 15. §61's 1/3 was an optimistic draw.
* **Mechanism (Audit's read of the logs).**
  - Identical-seed runs differ from step 50.
  - All runs look alike until the curriculum widens from depth 4 to 5. There s3 locked into the exact rule (loss at the
    ε floor), and every other run fell into the flooding plateau (loss ≈ 1.15) and stayed there.
  - Two attractors, chosen at the widening step.
  - Held-out shallow-question accuracy at P = 16 separated s3 (1.00) from the rest (0.78–0.95). Best-of-n selection by
    final-answer validation would be the honest way to report such a recipe, with n and its compute stated.
* **Reading.** The same two-attractor lottery reappears in §67 with a 36-parameter table, where over-complete codes
  remove it.

## 62. Does depth-structured label noise install a count? Why label-free distillation gives age-dependent latents (pre-registered 2026-10-01 06:44 EDT, after Literature's check 2e3d16b and before any run)

**Question (from §56).**
* Students distilled from true labels (pl_gt) are path-independent (G_10 0.09) and extrapolate in one loop.
* Students distilled from the label-free re-anchoring teacher (pl_t8) are age-causal (G_10 0.84) and do not.
* Across the 40 students, G tracks one-loop reach (ρ −0.87, post hoc).
* Why does the label source decide whether a latent carries a usable age?

**Premise check (exploratory; `theory/exp29_teacher_errors.py` → `theory/results/exp29_teacher_errors.json`, 1ea2ec5).**
* Setting: each Pilot L parent's T8 teacher with its own sweep head, on 512 training-distribution chains (widths 2,4,6).
* Error at d1–6: 0.11–0.30 (median ≈ 0.15). At d7–12: 0.32–0.46 (median ≈ 0.38).
* The seam contrast is positive in 20/20 parents (+0.15 to +0.31).
* Beyond the seam the errors are mostly misses: false-negative rate ≈ 0.40–0.55 on reachable nodes, false positives
  ≈ 0.2–0.3 on look-alikes.
* Within the T8 family, per-parent seam contrast does not predict the student's G (ρ +0.11): all 20 teachers share
  the structure.
* Two earlier passes of this check are superseded and kept as void files (void rule; added after Audit's note):
  - `theory/results/exp29_pass1.void`: head 3 for every parent and widths 2–3;
  - `theory/results/exp29_pass2.void`: head 3 for every parent and widths 2,4,6.
  - Head 3 was a bug: Pilot L used each parent's own sweep head. Both passes also showed a positive seam in 20/20
    parents.

**Hypothesis: calibration pressure** (Literature, `docs/LIT_NOISE_CLOCK_CHECK_2026-10-01.md`).
* With instance-dependent flip rates below ½, the Bayes-optimal *classifier* is unchanged (Menon, van Rooyen &
  Natarajan 1605.00751). A count therefore does not pay for decisions.
* It pays for *calibration*. Under cross-entropy, a flip rate η gives an optimal confidence of 1 − η: about 0.855 at
  d ≤ 6 vs 0.625 at d ≥ 7 for the teacher's profile. Matching that requires knowing the depth, so an age code is
  learned.
* Uniform noise lowers confidence uniformly, so no depth is needed and continuing the sweep stays optimal.
* The age code then degrades reach as a side effect. Once the code exists, a loop can key behaviour to it (Prop. 10).
* So depth-structured supervision, not the absence of labels, would cause the T8 students' age-dependence.
* **Prior.** Instance-dependent noise (2012.05458, 2006.07836, 2010.02347); distillation passes on teacher errors
  (2106.05945, 2205.16004); Lee 2502.01612 filters exactly such seam errors. Noise structure → counter: not found, so
  new as scoped. Nearest: Kuo's loop–length correlation, Prop. 4(b).

**Design.** `theory/distill_reanchor.py --teacher gt` with label flips on the deep chains (`--flip_all`, `--flip_far`,
1ea2ec5; a seeded stream per step). Otherwise exactly Pilot L's round-1 recipe: 3000 steps, bs 256, lr 1e-4, widths
2,4,6, K′ = 12, init from the parent.
* Parents: the 10 even-indexed P30 failures (102, 111, 113, 118, 125, 130, 132, 140, 146, 157), each as `--seed <s>`.
* Arms (40 runs):

  | Arm | Flips on deep-chain labels | Mean flip rate | Role |
  |---|---|---|---|
  | NS (structured noise) | 0.145 at d1–6, 0.375 at d7–12, symmetric | 0.26 | the teacher's median profile |
  | NS45 (secondary) | 0.145 at d1–6, 0.45 at d7–12 | 0.30 | the teacher's worst beyond-seam rate (Literature) |
  | NB (structured bias) | false negatives only: reachable d7–12 labels set to 0 with probability 0.47 (`--fn_far`) | 0.12 | the teacher's dominant error, a "stop at the seam" signal |
  | NU (uniform noise) | 0.26 at every depth, symmetric | 0.26 | same total as NS, no structure |

* References: the same parents' pl_gt (clean labels) and pl_t8 (teacher labels) students.
* Measures:
  - `theory/exp28_interchange.py --f 11` (G_10, dG, saving answers);
  - `theory/exp18_deep_reanchor.py` (one loop at K′ = 24: d7–12, d13–18);
  - `theory/eval_arch.py` (in-distribution sanity).
  - Per-depth confidence (NC-5), from the students' K′ = 12 logits on 512 training-distribution chains.

**Predictions** (paired by parent, per-pair sign counts):
* **NC-1 (structure installs age).** G_10 higher in NS than in NU in ≥ 8/10 pairs. Median G_10: NS ≥ 0.5 and NU ≤ 0.25.
* **NC-2 (reach follows).** One-loop d13–18 higher in NU than in NS in ≥ 8/10 pairs, by a median ≥ +0.10.
* **NC-3 (a systematic stop suffices).** NB is age-causal like NS: median G_10 ≥ 0.5, and one-loop d13–18 ≤ NU's median
  − 0.10.
* **NC-4 (prospective replication of §56's association).** Across all 40 students, G_10 vs one-loop d13–18 Spearman
  ≤ −0.5.
* **NC-5 (the signature of calibration pressure).** The mean predicted probability on reachable deep-chain nodes
  (training distribution, K′ = 12) steps down between d6 and d7 by ≥ 0.10 in the NS median, and by ≤ 0.03 in NU.
  Reported per depth for every arm, including NS45 and the references.
* **My expectation.** NC-1, NC-2 and NC-3 met.

**Readings.**
* **NC-1 and NC-2 met.** The age code comes from depth-structured supervision **via calibration**, not via a change in
  the decision rule, which is noise-invariant here. NC-5 should then show the confidence step. A label-free (C4) route
  to single-loop reach needs self-labels whose error does not rise with depth: drop or mask uncertain deep self-labels
  rather than train on them (as Lee 2502.01612 filters), or mix teacher horizons.
* **NC-3 met, NC-1 not.** A systematic one-sided signal installs the count, not symmetric noise. Its rate (0.47) also
  stays below ½, so this too would be read via calibration of the reachable class beyond the seam.
* **Neither met.** Label structure is not the cause. The T8 students' age code comes from elsewhere, e.g. the teacher's
  input distribution or the re-anchored episodes' internal states.

**Disclosure 62-1 (2026-10-01 06:45 EDT, before any §62 arm runs).** The NC-5 script
`theory/exp30_conf_by_depth.py` was smoke-tested on one reference pair (not a test arm). The d6 → d7 confidence step on
reachable nodes is +0.009 for pl_gt_s102 and +0.094 for pl_t8_s102, already the predicted direction.

**Run and score (experiments session).** 40 short GPU fine-tunes, the same cost class as one Pilot L arm. Score with
the measures above.

**Result (2026-10-01; scored by the experiments session, 5e0e8ce, `theory/results/score_s62.json`; all 40 informative
at f = 11). NC-4 met; NC-1, NC-2, NC-3 and NC-5 not met. The registered hypothesis (depth structure installs the age
code) is not supported. Calibration does show, graded by structure.**

| Arm (10 parents) | G_10 median (≤ 0.25) | One-loop d13–18 median (≥ 0.60) | d6 → d7 confidence step, median |
|---|---|---|---|
| NU, uniform 0.26 | 0.67 (3) | **0.60** (5) | 0.008 |
| NS, teacher profile | 0.88 (3) | 0.50 (2) | 0.059 |
| NS45 | 0.93 (0) | 0.50 (0) | 0.132 |
| NB, misses beyond the seam | 0.86 (0) | 0.50 (0) | **0.243** |
| References: pl_gt (clean) / pl_t8 | 0.09 / 0.84 | 0.74 / 0.50 | 0.014 / 0.039 |

* **NC-1 not met.** NS > NU in 7/10 pairs (8 needed), and NU's 0.67 is far above 0.25. Even uniform noise leaves the
  latents age-dependent.
* **NC-2 not met.** NU > NS on d13–18 in 10/10 pairs, but the median difference is +0.048 < 0.10.
* **NC-3 not met, by 0.002.** NB's d13–18 is 0.501 vs the 0.499 bar.
* **NC-4 met.** Across the 40 students, G_10 vs d13–18 Spearman −0.66: §56's association, replicated prospectively.
* **NC-5 not met for NS** (0.059 < 0.10; NU 0.008).
* **Reading.**
  - **The calibration mechanism is real, for confidence.** The confidence step at the seam grows with the strength of
    depth-structured bias: NU 0.008 < NS 0.059 < NS45 0.132 < NB 0.243.
  - **The age code does not need depth structure.** Any label noise keeps the students' latents age-dependent
    (G 0.67–0.93 vs 0.09 with clean labels).
  - These students start from age-causal P30 parents (G ≈ 0.9 at f = 11). Most likely, noise prevents deep
    supervision from overwriting the parents' code, rather than installing a new one. That is a hypothesis, not tested.
  - **Reach.** The depth-structured arms lose single-loop reach in median (d13–18 ≈ 0.50). With one cut for every arm
    (d13–18 ≥ 0.60), the runs that keep reach are:
    - NB 0/10 (its best, s146, is 0.57);
    - NS45 0/10;
    - NS 2/10 (s118 0.73, s146 0.81);
    - NU 5/10 (median 0.60).
    - Wording corrected after Audit's reproduction.
* **Together with §58** (training age invariance gives no reach): clean deep supervision delivers reach and age
  invariance together, and label noise removes both. Age invariance is a marker of how fully deep supervision rewrote
  the loop, not the mechanism.
* **Constraint for label-free (C4) depth extension.** Self-labels must be close to clean. A one-sided miss bias beyond
  the teacher's horizon, the realistic failure of self-labels, is the worst case: it removes reach in every parent.
  Masking uncertain deep self-labels beats labelling them negative.

## 63. F4 — a behavioural round-trip translator: can silent toy thoughts be made readable as node sets? (pre-registered 2026-10-01 10:21 EDT, after Literature's checks 3ba2507 and bb74280, before any run)

**Question (C7 for silent steps).**
* §34 read the pointer with a fitted linear map: round trip 0.947, edit IIA 0.36.
* F4 learns a decoder (thought → a hard set of node ids: the text) and an encoder (text → thought) **through the frozen
  reasoner's behaviour only**. The reasoner is never trained, and the set's meaning is never supervised.
* Is the silent state expressible as a set of input entities that behaves as written when edited?

**Prior** (`docs/LIT_F4_DELTA_2026-10-01.md`).
* Nothing since 3ba2507 trains a decoder and encoder through a frozen model with a behavioural substitution loss.
* Closest set-valued precedent with causal edits: Concept Bottleneck Models 2007.04612. Also IIT 2112.00826, NLA,
  2607.20379 (whose three audits are adopted below) and 2608.17843.

**Code.** `theory/f4_translator.py` (D: MLP → V logits over the graph's present ids, hard forward with an ST gradient;
E: MLP on [set, step one-hot] + a linear term).
* **Losses.**
  - Round trip: BCE of R's answers with thought i replaced by E(D(h_i), i), against R's unpatched answers.
  - Edit (IIT-style): add a random node u unreachable from START to the decoded set; the targets are R's answers OR
    BFS-reachable-from-u.
* **Training.** Long chains (K′ = 12) and training graphs (K = 6), half each; i ~ U{1..5}; 3000 steps, bs 128,
  Adam 1e-3.
* **Arms (Literature's age-channel caution).** S: the encoder sees the step index. N: `--no_step`, set only.
  2 seeds each.
* **Canonicalisation (2310.18512).** The text is an unordered multi-hot set over a fixed id vocabulary, so there is no
  order or no-op-token channel by construction.

**Models.** The 10 even-indexed P30 failures (`runs/p30_s{102,…,157}`) and the same parents' GT students
(`theory/runs/pl_gt_s…`): 20 models × 2 arms × 2 seeds = 80 translator trainings.

**Evaluation (held out).** Long-chain probe, 512 graphs, seed 4247 (training uses other seeds), K′ = 12, i = 2
(primary) and 3. **Aggregation (Audit): each metric per model over its 512 × 24 answers, then medians over models per
family and arm.**
* **RT.** Agreement of the hard-text round trip with R's unpatched answers.
* **IIA (held-out edit type).** §34's test on the text.
  - u = the look-alike of the deepest reachable-chain node j in the decoded set.
  - ins: both look-alikes at j+2 and j+3 answered reachable. keep: all reachable-chain answers unchanged.
  - IIA = mean(ins & keep) over graphs with such a j ≤ 8.
  - Training never uses look-alike-of-frontier edits.
* **Input-only control (Audit, Literature).** The encoder is fed the input-derived text: the true set reached within i
  hops, by BFS on the input, with the same step index. Reported: RT_input, and the error-reproduction rates below.
* **Error reproduction (faithful to the model, not the task).** On (graph, query) pairs where R's unpatched answer is
  wrong, the share where the round trip reproduces R's answer: err_repro (own text) vs err_repro_input (input-only
  text).
* **Evaluator swap (2607.20379).** Decode with seed-1's D and encode with seed-2's E, and vice versa. Report RT_swap
  and IIA_swap (mean of the two directions).
* **Claim-flip audit (2607.20379).** Add a random unreachable node: flip_on is the share of its BFS descendants among
  the probe queries that turn on; flip_collateral is the share of other answers that change.
* **Grounded-vs-true cross.** Precision and recall of the decoded set against the true set reached within i hops. A
  set that is behaviourally effective but untrue would be co-adapted content.
* **Controls.** RT_mean (h_i ← the mean thought at step i) and RT_mismatch (decoding another graph's thought).

**Predictions** (P30 models, arm S, i = 2, medians over the 10 models; the C7 claim needs F4-1 to F4-4 together):
* **F4-1 (round trip).** RT ≥ 0.95.
* **F4-2 (edits behave as written).** IIA ≥ 0.80 (§34: 0.36), with flip_collateral ≤ 0.05.
* **F4-3 (faithful to the model).** err_repro ≥ 0.80 and ≥ err_repro_input + 0.15.
* **F4-4 (no private code).** RT_swap ≥ RT − 0.05 and IIA_swap ≥ 0.60.
* **F4-5 (age channel; cross-check of §56).** The step-index gap RT(S) − RT(N) is larger for P30 than for GT students
  by ≥ 0.02 (medians). §56 found P30 latents age-causal and GT latents age-invariant.
* **Descriptive.** All metrics for GT students and for arm N; i = 3; precision and recall; RT_mean and RT_mismatch.
* **My expectation.** F4-1 and F4-3 met; F4-2 partial (0.5–0.8), because the probe's edit type is held out; F4-4 met
  if F4-2 is.
* **Disclosure.** A CPU pipeline smoke ran on runs/p30_s100, outside the set, for 300 steps (untrained: RT 0.84, IIA
  0.0). It checks plumbing only.

**Readings.**
* **F4-1 to F4-4 met.** Silent latent states of a toy reasoner are faithfully readable as sets of input entities, by
  optimisation, with no change to the reasoner: C7 for silent steps.
* **F4-1 met, F4-2 failed.** The text reproduces behaviour but does not control it. A hidden channel lies outside the
  set vocabulary; its size is the measurement.
* **F4-3 failed.** The text describes the task, not the model.
* **F4-4 failed.** The translators co-adapted a private code.

**Run (experiments session).** One wave of parallel Vast hosts.
* 80 runs:
  `python theory/f4_translator.py --model <model> --seed {1,2} [--no_step] --device cuda:0 --out theory/runs/f4/<model>_<S|N>_t<seed>`.
* Then 40 swaps: `python theory/f4_translator.py --swap <…_t1> <…_t2> --device cuda:0`.
* A 100-step GPU smoke first, to time it.

**Implementation note 63-1 (2026-10-01 10:28 EDT, before any run; the experiments session's GPU smoke crashed at step
1).**
* **Cause.** When the decoder marks every node unreachable from START, which happens early in training, the edit has
  no candidate node.
* **Fix.**
  - In training, such rows are masked out of the edit loss.
  - In evaluation, such graphs are excluded from the claim-flip audit (`flip_eligible` reports the share).
* Unit-checked on a forced empty case, with a CPU smoke on runs/p30_s100 (outside the set). No metric definition
  changes.

**Result (2026-10-01; scored by the experiments session, `theory/results/score_f4.json`, 80 runs plus 40 swaps; P30, arm S,
i = 2, medians). F4-1 to F4-5 all not met. The translator did not train, so F4 as specified is uninformative about
whether silent thoughts are set-expressible.**

| Measure | Value | Bar |
|---|---|---|
| RT | 0.916 | ≥ 0.95. Controls: RT_mismatch 0.905, RT_input 0.864, RT_mean 0.850 (Audit's recomputation) |
| Edit IIA | 0.013, collateral 0.081 | ≥ 0.80 (§34's linear pointer map: 0.36) |
| err_repro | 0.887 vs input-only 0.812 | +0.075 < +0.15 |
| Evaluator swap | RT 0.875, IIA 0.008 | — |
| F4-5 step-index gap | median of per-model differences: −0.001 (P30) vs −0.006 (GT); difference of medians: −0.002 vs −0.010 | P30 larger by ≥ 0.02 under either aggregation |
| GT students, arm S | RT 0.833, IIA 0.002 | descriptive |

* **Why.**
  - The training losses are flat from step 100 to 3000 (e.g. p30_s102: l_rt 0.20 → 0.18, l_edit 0.30 → 0.32).
  - The hard set stays at ≈ 28 of ≈ 61 present nodes, with precision 0.13 and recall 0.43 against the reached set.
  - The hard-forward, straight-through decoder never left its initial regime: the same optimisation failure as §54's
    hard labels (PROTOCOL12).
  - The round trip looks high only because these reasoners tolerate a replaced step-2 thought: decoding another graph's
    thought keeps 0.905 of answers.
  - The text carries little that is specific to the model (Audit): RT is barely above the mismatch, input-only and
    mean-thought controls, and edit IIA is 0.013. It mostly re-encodes what the input already fixes.
* **Reading.**
  - Nothing is learned here about whether silent thoughts are set-expressible. §34's hand-built pointer map remains the
    best C7 result for silent steps (IIA 0.36).
  - Recurring lesson, now twice (§54, §63): hard straight-through discrete bottlenecks did not optimise in this project.
    Any discrete-text method needs a relaxed training path (as in §54b's soft-label hint arm, which worked).
  - F4 is closed as a frontier test. No revision.

## 64. Beyond graph search: permutation tracking in the latent loop — do the count-keyed stop and the written-state restart reappear? (pre-registered 2026-10-01 11:39 EDT, after Literature's check b6e57a7, before any run; user directive via Audit)

**Why.**
* Every result so far, toy and GPT-2, is on reachability, which is monotone closure.
* Permutation tracking (group word problems) overwrites its state at every step, and the non-solvable case has no
  shallow shortcut. Are this project's mechanisms properties of latent reasoning, or of graph search?
* This test asks whether two toy findings reappear: the count-keyed stop beyond the trained length (Prop. 10, P10e),
  and written-state restarts fixing it (§26).

**Prior** (`docs/LIT_PERM_TRACKING_CHECK_2026-10-01.md`).
* Looped and recurrent state tracking: Liu 2210.10749, 2404.08819, 2411.12537, Implicit LMs 2502.07827, Zhang
  2607.20594, Giannou 2301.13196.
* Randomised positions: Ruoss 2305.16843.
* Closest competitor: Supervised Thinking States 2602.08332, whose thoughts are supervised.
* New as scoped: a final-answer-only Coconut loop on group word problems, and a written-state restart with the history
  cut.

**Design.** `theory/st_model.py`.
* **Task.** n items in an initial arrangement (identity with probability ½, else random) and N group elements applied
  in order. Query: each item's final slot (n-way).
* **Input.** Init tokens plus operation tokens at sorted random positions in [0, 64), so every position embedding is
  trained.
* **Loop.** The same toy architecture (d 256, 2 layers, 8 heads, no step embedding). K latent steps (Coconut feedback),
  then n query tokens.
* **Training.** Final answers only, N ~ U{1..8}, K = 8, 20k steps, bs 256, AdamW 5e-4.
* **Groups, scored separately (Audit).**
  - **A5 (primary).** The even permutations of 5 items: non-solvable, 60 elements. Literature notes that solvable groups
    allow shallow shortcuts that bypass the loop.
  - **S3 (secondary).** Solvable; its K′ = 0 accuracy shows any bypass.
  - S5 is not run: looped models deadlock on it without an operator-first curriculum (Zhang). A5 is the registered
    non-solvable fallback.
* **Seeds.** 10 per group: 20 runs.

**Measures** (per run; held-out N = 1..24, 1024 instances each).
* single: full-state accuracy at K′ = max(8, N), plus K′ = 24 and K′ = 0.
* trunc: for N > 8, agreement of the answers with the arrangement after the first 8 operations. This is the count-keyed
  stop's signature.
* restart: written-state restarts. Run 8 operations, decode the most probable arrangement, write it as the init block,
  continue on the next 8.
* edit_IIA: for N = 8, the init block replaced by a random arrangement; the answers must equal the true composition
  from the edited state. C7 for the written state, scored against the truth (Literature).
* pointer: per latent step, the rank of the operation token most attended by layer 1, on N = 16.
* Baselines: chance 1/n! and the identity (no-update) predictor, per N.

**Learnability gate (Audit).** A run is *gated* if its full-state accuracy at N ≤ 8 is ≥ 0.90. Only gated runs enter
ST2–ST4. An A5 learnability failure is a possible outcome (Zhang reports seed-bimodal A5) and is not evidence against the
mechanisms.

**Predictions** (each group separately; over gated runs):
* **ST1 (the loop does the work; C1).** A5: median K′ = 0 accuracy at N = 5–8 ≤ 0.10 (no shortcut). For S3 it is
  reported, since a shallow shortcut there would make S3 uninformative about the loop.
* **ST2 (count-keyed stop; the P10e analogue).** At N = 9–16 with K′ = N, the answers agree more with the 8-operation
  truncation than with the true final arrangement in ≥ 70 % of gated runs. Median true accuracy at N = 13–16 is ≤ half
  the in-range accuracy.
* **ST3 (restarts fix it; the §26 analogue).** Written-state restarts reach median accuracy ≥ 0.75 at N = 9–16, and beat
  the single loop by ≥ 0.20 in ≥ 70 % of gated runs.
* **ST4 (the written state is read faithfully).** Median edit_IIA ≥ 0.90.
* **Descriptive.** The pointer-rank profile (does it stop advancing at rank 8?), K′ = 24 vs K′ = N, the restart curve
  to N = 24, and the number of gated runs per group.
* **My expectation.** ST1–ST4 met for A5, if enough seeds pass the gate.

**Readings.**
* **ST2 and ST3 met.** The count-keyed stop and its restart fix are properties of latent loops, not of graph search. The
  next registration is the GPT-2 overwritable public state (value slots), with text CoT at matched compute (C3), then
  one-loop distillation and the §62 label check on this task.
* **ST2 failed.** The loop continues past its trained length on state tracking. The count-keyed stop would then be
  specific to the frontier sweep of reachability, and the mechanism story must be scoped to graph search.
* **Too few gated runs.** No mechanism verdict. A5 with Zhang's curriculum becomes the follow-up.

**Run (experiments session).** One wave, 20 runs: `python theory/st_model.py --group {A5|S3} --seed {1..10} --device
cuda:0 --out theory/runs/st/{a5|s3}_s<seed>`. Evaluation is written as `eval_st.json`. Run a short GPU smoke first for
timing.

**Result (2026-10-01; scored by the experiments session, 00071a6, `theory/results/score_s64.json`; Audit reproduced from
the 30 eval_st.json files).**
* **A5: no mechanism verdict.** 0/10 runs pass the gate (median in-range 0.008). With §64b's curriculum, also 0/10:
  best in-range 0.26 and 0.07.
  - The curriculum's length-2 stage did not escape the ln 5 plateau in 20k steps either (e.g. a5c_s1: loss 1.6094
    throughout).
  - By §64b's fixed rule the A5 line ends here: no further revision. The non-graph evidence rests on S3 (note 64-2).
* **S3 (9/10 gated, median in-range 0.997).** "On a solvable group".
  - **ST1 met.** The K′ = 0 accuracy at N 5–8 is 0.020, so the loop does the work and note 64-2 admits ST2/ST3.
  - **ST2 not met.** Truncation agreement exceeds true accuracy in 0/9 runs, though true accuracy at N 13–16
    collapses (0.032, below half of in-range).
  - **ST3 met.** Written-state restarts reach median 0.995 at N 9–16 and 0.976 at N 17–24, and beat the single loop by
    ≥ 0.20 in 9/9.
  - **ST4 met.** Median edit_IIA 0.969: the written state is read as the true composition from it.
* **What the single loop does past its trained length** (descriptive, post hoc, at Audit's request:
  `theory/exp31_st_wrong_answers.py` → `theory/results/exp31_st_wrong_answers.json`, plus a validity check).
  - It extends 1–2 operations (median single at N = 9: 0.77; N = 10: 0.26), then fails.
  - Extra latent steps make it worse: K′ = 24 at N 9–16 gives 0.049 vs K′ = N. There is no stable fixed point; the
    loop keeps rewriting the state.
  - At N = 12, its answers agree with no prefix or suffix product of the operations (all ≈ 0.04–0.09, under the 1/6
    chance), because most answers stop being valid arrangements.
    - The share of answers that are permutations falls from 1.00 at N = 6 to medians 0.82 (N 9), 0.47 (N 10), 0.31
      (N 12) and 0.14 (N 16) (`theory/exp31b_st_validity.py`, output `theory/results/exp31b_st_validity.txt`).
    - Among valid answers at N = 12, accuracy is near chance (median 0.08).
  - The read pointer advances past rank 8 in most runs and falls back in some.
* **Reading.**
  - The transferable result is that **written-state restarts fix the single loop's length failure outside graph search
    too** (0.995 at N 9–16), with a faithfully read written state.
  - The failure itself differs from the frontier sweep's. It is not a count-keyed stop at the trained state (ST2), but
    a loss of the state's validity once the loop runs past its trained length: the latent leaves the set of valid
    arrangements.
  - Prop. 10's specific form (a stop keyed to a count) is therefore scoped to the graph toy. The general statement that
    survives: a latent loop is reliable only within the step counts it was trained on, and restarting from a written
    state resets that range.

**Disclosure 64-1 (11:39 EDT).**
* A CPU plumbing smoke of an earlier code version (S3, seed 1, 1500 steps, bs 128) started at 11:25, before this
  registration. It was stopped at 11:39 before any evaluation; it had logged at most training-loss lines.
* Its partial log is kept as `theory/results/void/st_smoke_s3_seed1_cpu.void`.
* The registered S3 seed 1 run (20k steps, bs 256) is unaffected. A separate 20-step A5 smoke checked the evaluation
  code path only.

**Note 64-2 (2026-10-01 12:26 EDT, before §64 is scored; Audit's live-log observation that A5 sits at loss ln 5).**
* **Not a plumbing bug.** A CPU check outside the registered runs (scratch script; A5, bs 128, 400 steps) shows:
  - N = 1 problems learn to full-state accuracy 0.97;
  - with N ≤ 2 the loss leaves ln 5 by step 200 (accuracy 0.31 at step 400).
  The registered A5 runs sit on an optimisation plateau, Zhang's deadlock pattern, not on a broken target.
* **What S3 alone can show (fixed before scoring).** S3's ST2 and ST3 count as evidence about the loop's mechanism only
  if its median K′ = 0 accuracy at N = 5–8 is ≤ 0.50, i.e. the loop is needed. Otherwise S3 is reported descriptively
  (Liu's shortcut caveat). An S3-only verdict is stated as "on a solvable group".
* **A5 follow-up.** The single registered revision is §64b, an operator-first length curriculum (registered next). It
  is the follow-up §64's readings named in advance. There is no revision after it.

### 64b. Permutation tracking, the single A5 revision: Zhang et al.'s operator-first curriculum (pre-registered 2026-10-01 12:29 EDT, before §64 is scored; Literature's addendum 68a0a3d)

**Why.** §64's A5 runs sit at loss ln 5 (Audit's live-log note).
* Note 64-2 showed this is an optimisation plateau, not a bug: A5 N = 1 problems learn to 0.97 in 400 CPU steps.
* Zhang 2607.20594 (App. D) reports that joint training deadlocks on S5, while a sequenced curriculum learns in 10/10
  runs.
* Its schedule starts with "an extended n = 2 stage that masters the composition table before any chaining", then
  doubles the length.
* §64's readings named this as the follow-up. It is the single revision; there is none after it.

**Change from §64 (A5 only).** `theory/st_model.py --curriculum` (schedule commit 9ae43d2).
* Stage 1, the first 32 % of steps: length-2 problems only, N = 2.
* Stage 2, 32–48 %: N ~ U{1..4}.
* Stage 3, the rest: §64's distribution, N ~ U{1..8}.
* This is Zhang's operator-first schedule adapted to our maximum training length of 8. Zhang: length 2 for ≈ 32 % of
  steps, then doubling stages.
* Everything else is unchanged: 20k steps, bs 256, K = 8, the model, measures, the gate (≥ 0.90 at N ≤ 8), and ST1–ST4
  with their thresholds and readings.
* Seeds 1–10: 10 runs.

**Decision rule, fixed now.**
* If fewer than 3 seeds pass the gate, A5 gives no mechanism verdict, and the non-graph evidence rests on S3 (note
  64-2's rule).
* Otherwise ST1–ST4 are scored on the gated A5 runs as in §64.

## 65. Restart-distillation on permutation tracking: can a near-clean, label-free teacher give one loop its reach? (pre-registered 2026-10-01 14:55 EDT, after Literature's note 62fc986 and before any run)

**Why.**
* §64: on S3, written-state restarts are a near-clean, label-free procedure (0.943–0.998 at N 9–16 in the 9 gated
  models). The single loop collapses past N = 8 and its state loses validity.
* §62: label-free distillation needs near-clean labels. §32 (Pilot L): distilling restarts extended one loop on graphs,
  but with a seam-noisy teacher.
* So this tests whether a near-clean restart teacher gives a single loop the reach that true labels give, outside
  graph search.

**Prior** (`docs/LIT_PERM_TRACKING_CHECK_2026-10-01.md` addendum).
* Closest: context distillation (Snell, Klein & Zhong 2209.15189): generate with a scratchpad, then fine-tune to answer
  without it.
* Also implicit CoT 2311.01460, stepwise internalisation 2405.14838, System 2 → 1 2407.06023, progressive distillation
  2202.00512, consistency models 2303.01469, Lee 2502.01612, and Pilot L.
* New as scoped: label-free distillation of a written-state restart procedure into one latent loop, for state
  tracking.

**Design.** `theory/st_distill.py`.
* **Student.** Each of the 9 gated §64 S3 models: s3_s1, s2, s3, s4, s5, s6, s8, s9, s10. s3_s7 is ungated and was used
  only for the code smoke.
* **Teacher.** A frozen copy of the student.
* **Each step.**
  - A normal batch: N ≤ 8, K = 8, true final answers (the original task).
  - A deep batch: one N ~ U{9..16}; the student answers in one loop at K′ = N against the teacher's targets.
* 3000 steps, bs 256, AdamW 1e-4.
* **Arms (27 runs).**
  - **RD (primary, label-free).** The frozen copy's restart chain, every 8 operations.
  - **GT (ceiling).** True final arrangements.
  - **SELF (control).** The frozen copy's own single loop at K′ = N, which is near chance there.
* **Measures.**
  - §64's eval (`eval_st.json`): single loop at K′ = max(8, N) for N 1–24, K′ = 24, restarts, and edit IIA.
  - `valid.json`: the share of single-loop answers that are valid arrangements at N 12, 16 and 20.
  - The teacher's per-N error, i.e. 1 − the parent's restart accuracy at each N 9–16 in its §64 `eval_st.json`
    (Literature's §62 check). It is seam-free if it does not jump at N = 9 or 16.

**Predictions** (medians over the 9 parents; paired counts):
* **D1 (label-free single-loop reach).** RD single-loop accuracy at N 9–16 ≥ 0.85, and ≥ SELF + 0.50 in 9/9 pairs.
* **D2 (near-clean ≈ clean).** At N 9–16, RD ≥ GT − 0.05, as §62 predicts when label error is small.
* **D3 (valid state).** RD's valid share at N 12 ≥ 0.95 (§64's single loops: 0.31).
* **D4 (in-range kept).** RD in-range accuracy (N ≤ 8) ≥ 0.97.
* **Descriptive.** N 17–24 (beyond the distilled length; §41 predicts partial reach), K′ = 24, and SELF and GT
  profiles.
* **My expectation.** D1–D4 met.

**Readings.**
* **D1 and D2 met.** A label-free procedure, restarts from written states, gives a single latent loop its reach on a
  non-graph task, and near-clean self-labels act like true ones. This is a C2 + C4 single-loop result outside graph
  search.
* **D1 met, D2 failed.** The gap between near-clean and clean labels matters here.
* **D1 failed.** Distillation does not transfer restart reach into the loop on state tracking.

**Run (experiments session).** `python theory/st_distill.py --init theory/runs/st/s3_s<k> --teacher {rd|gt|self}
--device cuda:0 --out theory/runs/st/s3<rd|gt|self>_s<k>`: 27 runs, local 2070 and other free cards.

**Result (2026-10-01; scored by the experiments session, c6c09d0, `theory/results/score_s65.json`; Audit reproduced).
D1 not met, D2 met, D3 not met, D4 met.**

| Arm (median of 9) | N ≤ 8 | N 9–16 | N 17–24 | Valid share at N 12 |
|---|---|---|---|---|
| RD (restart labels) | 0.998 | 0.658 | 0.060 | 0.78 |
| GT (true labels) | 0.998 | 0.657 | 0.060 | 0.78 |
| SELF | 0.997 | 0.162 | 0.013 | 0.31 |

* **D1 not met.** RD is 0.658 < 0.85, and RD ≥ SELF + 0.50 in only 3/9 pairs.
* **D2 met.** RD equals GT, parent by parent (e.g. s1 0.89 / 0.89, s5 0.27 / 0.27). That is expected: the restart
  labels are 99.5 % correct, and the runs share seed and data order.
* **D3 not met.** The valid share at N 12 is 0.78 (§64's single loops: 0.31).
* **D4 met.** In-range accuracy 0.998.
* **Reading (with Audit's).**
  - **The limit is not label quality.** True labels on N 9–16 give the same 0.66, and nothing past the distilled length
    (0.06 at 17–24).
  - On S3, deep supervision extends the single loop only partly within the supervised range, and not beyond it. On the
    graph toy, GT supervision on 12 hops reached about 22, and GT18 stayed flat to 30 (§41).
  - This is the range principle again. The S3 loop holds an exact discrete state in a continuous vector, which decays
    with every step past the trained range (the valid share falls). The graph frontier sweep has a content-keyed
    continuation that does not decay.
  - So on state tracking, neither better labels nor deeper supervision gives a single continuous loop its reach. This
    is the case for §67's discrete carried state.
* **For C4.** The label-free restart teacher is as good as true labels (D2), so restart-distillation is a clean C4
  method. The ceiling is the student's architecture.

## 66. GPT-2 state tracking with a written-state chunked loop (WC): a scaffolded test of C2, C3 and C4 beyond graph search (pre-registered 2026-10-01 15:02 EDT, after Literature's check 27a4a05 and Audit's review; before any implementation or run)

**Why.** §64: on S3 the written-state restart fixes the single loop's length failure (0.995 at N 9–16), with a
faithfully read written state. This ports it to a pretrained LM and asks whether it beats text CoT at matched compute
(C3). Design draft: §52 addendum (186d5f5); every point of Audit's review is adopted.

**Prior** (`docs/LIT_PERM_TRACKING_CHECK_2026-10-01.md` addendum).
* Nearest:
  - Thinking States 2602.08332: supervised text thoughts, no history cut.
  - Delethink 2510.06557 and MEM1 2506.15841: text carry-over with a history cut, but explicit CoT and RL.
  - Markovian Transformers 2404.18988, PENCIL, InftyThink, RMT.
  - None combines latents within a chunk, a text state between chunks, a history cut, and training on short-problem
    final answers. New as scoped.
* No compute-matched CoT-vs-latent comparison on state tracking was found, so a matched C3 result here would be new.
* **Caution.** LMs learn an associative scan for permutation tracking (Li, Guo & Andreas 2503.02854), and S4 is
  solvable (Liu). WC₀ may handle small k, so WC vs WC₀ is reported by k to measure C1.
* Boxes-task priors: 2305.02363, 2402.14811.

**Task (generator-built; added to `bench/` with a generator loader and a generate-kind scorer, so every arm is scored by
one harness).**
* 4 boxes holding A–D, in a random start arrangement. Operations "swap boxes i and j" (transpositions; S4), in random
  order.
* Question: the full final arrangement, or "which box holds X" (4-way).
* Train N ≤ 8 operations. Test N = 1–32: in range 1–8, extrapolation 9–16 and 17–32.

**Arms (from original GPT-2 small; ≥ 5 seeds each; the same task data budget).**
* **NC (no CoT).** Question → answer. Final answers only, N ≤ 8.
* **TC (text CoT, the C3 bar).** The arrangement written after every operation, then the answer. Step-labelled; the
  standard reference, not C4.
* **CL (Coconut single loop).** K′ = N continuous thoughts, then the answer. Final answers only, N ≤ 8.
* **WC (the proposed method; scaffolded).**
  - **Training** (credit assignment, Audit's (1)): a short-problem solver trained only on final answers. Input: start
    arrangement + k′ ≤ k operations; k′ continuous thoughts; output: the end arrangement as text. Random start
    arrangements; k = 4.
  - **Inference:** chunk the long problem every k operations. Decode each chunk's final arrangement (greedy) and feed
    it as the next chunk's start, with the latent history cut. The answer is read from the last arrangement.
  - The written state is just the model's own final answer to a sub-problem: no intermediate labels exist.
  - **Fixed chunk boundaries are a hand-built scaffold** (as in §46). A learned chunk or halt decision is a later C5
    step.
* **WC₀ (control).** WC with no continuous thoughts. It tests whether GPT-2 composes k swaps per forward pass anyway.
* **k sweep (C1; Literature).** WC and WC₀ at k ∈ {2, 4, 8}. The k = 8 solver is trained on k′ ≤ 8, i.e. the full
  training length per chunk. The latents do work (C1) where WC > WC₀ at some k, paired by seed and item.
* **A5 (exploratory, separate verdicts; not gating).** WC and CL with "rotate boxes i, j, k" 3-cycles on 5 boxes, with
  Zhang's operator-first curriculum, 3 seeds each, if budget allows.

**Baselines (no reasoning).** The identity (start arrangement), last-op-only, and the majority arrangement.

**Gate (bypass check; note 64-2's form).** The main results count as evidence about reasoning only if NC's median
accuracy at N = 5–8 is ≤ 0.50 (full arrangement). Otherwise they are descriptive.

**Compute (three axes per problem; FLOPs primary).**
* FLOPs: tokens processed including latent positions, with KV-cache use stated.
* Sequential forward passes.
* Written tokens.
* Each arm gives an accuracy-vs-FLOPs curve: TC with its natural length, and with the state written every 2 operations
  as a cheaper variant; WC with k ∈ {2, 4}; CL with K′.

**Predictions (S4; medians over seeds; in range and extrapolation reported separately).**
* **P1 (single loop fails past training; the §64 analogue).** CL at N 17–32 ≤ ½ of its in-range accuracy.
* **P2 (WC holds; C2 + C4, scaffolded).** WC (k = 4) ≥ 0.95 at N 17–32.
* **P3 (C3; Audit's rule, fixed now).** At N 17–32, WC accuracy ≥ TC accuracy, paired by item, with the 95 % CI of the
  difference above −0.02, at ≤ TC's median FLOPs. "Partial" if within 0.05; otherwise not met.
* **P4 (C7, scoped to the written state; latents within a chunk are not claimed).** Edit IIA: replace a written
  arrangement mid-chain; the answer must follow the true composition from it. ≥ 0.90.
* **P5 (C1: the latents do work).** At k = 8, WC beats WC₀ at N 17–32 by ≥ 0.05 (median paired difference). At k ≤ 4,
  Literature expects WC₀ ≈ WC (a shallow scan suffices); reported, not predicted.
* **Descriptive.** WC₀ under P3's rule (chunked text without latents is itself a C3 candidate); the TC length curve;
  seed variance; the A5 exploratory arm.

**Readings.**
* **P2 and P3 met.** A pretrained LM tracks state far past its trained length from final answers alone (scaffolded),
  more cheaply than text CoT: a C2 + C3 + C4 result beyond graph search.
* **P2 met, P3 not.** Restarts transfer to the LM, but text CoT is as cheap or cheaper here.
* **P2 failed.** The short-problem solver errs too often for chaining (errors compound over N/k chunks).

**Implementation and run (experiments session).** GPT-2 code reuse: prosqa/gpt2c.py (Coconut rollout). One parallel
Vast wave after a smoke and a throughput pilot (the pilot may fix only steps and batch, logged).

**Amendment 66-1 (2026-10-01 15:12 EDT, before any run; Audit's review of the bench task).**
* **Test seed.** The bench boxes test set uses generator seed 6600, disjoint from every training generator seed (the
  implementation asserts this).
* **In-range overlap.** In-range N (≤ 8) necessarily overlaps the training problem space: 4 boxes give few distinct
  short problems. In-range accuracy is therefore not a held-out generalisation measure. All C2 and C3 verdicts (P1–P3,
  P5) use extrapolation N > 8, which is clean.
* **Chance per question type.** Full arrangement 1/24; "which box holds X" 1/4. Reported per type next to the
  baselines.
* **Scoring.** Generated answers are normalised before exact match (the letters, or the box digit, extracted), so
  formatting such as "A,C,B,D" or extra spaces cannot change a score. Implemented in the bench scorer, with a unit test.

**Result (2026-10-01; scored by the experiments session, dc8f219, `theory/results/score_s66.json`; 10 arms × 5 seeds).
Gate passed; P2, P3 and P4 met; P5 not met; P1 uninformative. The no-latent control does exactly as well.**

| Arm (median) | N 1–8 | N 9–16 | N 17–32 | GFLOPs at N 17–32 |
|---|---|---|---|---|
| NC (direct) | 0.22 | 0.14 | 0.15 | 20 |
| TC (text CoT, every op) | **1.00** | 0.18 | 0.14 | 52 |
| CL (Coconut single loop) | 0.18 | 0.14 | 0.14 | 25 |
| WC4 (4 latents per chunk) | 1.00 | 1.00 | **1.00** | 43 |
| WC0_4 (no latents) | 1.00 | 1.00 | **1.00** | 35 |
| WC8 / WC0_8 | — | — | both fail | — |
| Baselines (chance, identity, last op, majority) | ≈ 0.14–0.27 | ≈ 0.15 | ≈ 0.15 | — |

* **Gate.** NC's median full-arrangement accuracy at N 5–8 is 0.045, so the gate is passed.
* **P2 met.** WC4 is 1.000 at N 17–32. Every written chunk state is a valid arrangement (validity 1.0).
* **P3 met.** WC4 − TC is +0.858 (95 % CI 0.848–0.867, 1,600 items) at 43 vs 52 GFLOPs. Caveat: TC is perfect in range
  but at chance past N = 8. Text CoT in one growing GPT-2 context does not length-generalise. WC wins because every
  context stays within the trained length, not because it is a cheaper way to do the same computation in range.
* **P4 met.** Written-state edit IIA is 1.0.
* **P5 not met.**
  - At k = 8, WC − WC0 has median −0.026 (per seed −0.02, −0.57, −0.42, −0.03, +0.19): the latents hurt or do nothing.
  - At k ≤ 4, WC and WC0 are identical (Δ 0.000 in every seed).
  - **C1 is not shown.** WC0 alone meets P3's rule (35 vs 52 GFLOPs).
* **P1 uninformative.** CL never learned, even in range (0.18 ≈ baselines). Final-answer training of GPT-2 continuous
  thoughts without Coconut's CoT-replacement curriculum has not produced latent computation anywhere in this project
  (PROTOCOL8: latents unused; here CL ≈ chance).
* **Reading.** What transferred to a pretrained LM is the written-state restart with a history cut, i.e. short
  contexts. GPT-2's ordinary forward pass composes up to 4 swaps per chunk; the continuous thoughts add nothing.

### Synthesis (2026-10-01): the range principle behind §26, §64 and §66

* **Every failure past the trained length** in this project is a quantity that grows with the problem and was bounded in
  training, then leaves its trained range:
  - the iteration count or step clock (graph toy, Prop. 10, §35);
  - the latent history length (P10b′);
  - the latent state's validity, i.e. the loop running past its trained steps (§64);
  - the context length and positions of text CoT (§66).
* **Every fix that worked** keeps all such quantities in range by restarting from a written state with the history cut:
  re-anchoring (§26), CLR (§45), S3 restarts (§64), WC/WC0 (§66). The written state is the only thing carried.
* **Deep supervision** (§41, GT students) extends the range instead: training on longer problems.
* **What did not work** tried to make a growing quantity harmless without bounding it: clamps (§36), adversarial count
  removal (T12), age-transplant training (§58), counter-free label channels learned from final answers (F1-S 1/3, F4).
* **Consequence for the north star.**
  - C2 is reached, scaffolded, by chunking into in-range pieces with written states. C7 holds for those states by
    construction.
  - The open problems are now C5 (learning where to cut instead of fixed k) and C1 (latent computation earning its
    place). In GPT-2, latents have not yet been trainable from final answers.

## 67. C1 Design 1: a six-code categorical latent with GPT-2 as the transition cell, trained by exact marginalisation (pre-registered 2026-10-01 18:30 EDT, after Literature's check c2c6957 and Audit's triage; before any implementation or run)

**Source.** `docs/EXTERNAL_IDEAS_C1_2026-10-01.md` (an external agent at the user's request; Audit's triage at the top).
C1 is the main frontier, by user decision.

**Why this design.** It attacks both diagnosed failure modes.
* **Credit assignment.** §54, F1-S's two attractors and F4's flat losses all came from straight-through or relaxed
  discrete training. Here the latent is trained by *exact* forward marginalisation, with no straight-through.
* **Drift.** §64, and the snapping premise above. Here the carried state is one of 6 codes, so it cannot drift.

**Prior** (`docs/LIT_EXTERNAL_C1_CHECK_2026-10-01.md`).
* Input–Output HMMs (Bengio & Frasconi, NeurIPS 1994): neural input-conditioned transitions over discrete states,
  trained by forward likelihood. This is the same recipe; GPT-2 is the transition net.
* Also State-Regularized RNNs 1901.08817, differentiable FSMs (2022) and neural HMMs 1609.09007.
* New: GPT-2 as the transition cell, streaming, hard execution at test, the full 36-transition audit, and the causal
  controls.

**Task.** S3 composition. Each operation is a uniform random element of S3; the answer is the final permutation, a
6-way label. With uniform independent operations, a stream processor without memory is at chance (1/6).

**Model.**
* **Cell.** Original GPT-2 small. Input: [code embedding (one of M = 6 learned) + operation symbol + fixed marker].
  Positions restart identically at every call. A 6-way head gives the next code.
* **Initial code.** Fixed.
* **Terminal decoder call.** [code + terminal marker] → 6-way answer.
* **Carried state.** Only the code embedding crosses calls. Logits, activations and KV cache are discarded.

**Training (final answers only).**
* For each update, GPT-2 is evaluated on all 36 (code, operation) pairs, giving T_a(i, j), and on the 6 codes for the
  decoder.
* Then α_t = α_{t−1} T_{a_t}, and the loss is −log Σ_i α_N(i) D(i, y).
* Curriculum: complete examples of lengths 1, 2, 4, 8, on a fixed schedule.

**Inference.** Real GPT-2 calls; the argmax code is kept after each operation. There is no probability feedback and no
table executor.

**Arms (5 seeds each).** A throughput pilot fixes updates, batch, precision and optimizer, logged; at most 3 GPU-h per
arm per seed on a 3090-class card; timeouts count as failures.
* **D1, the GPT-2 cell.**
* **TAB, a tiny control.** A learned 6×6×6 table (or a small MLP) under the same exact marginalisation.
* **RESET.** An identical scaffold trained with the carried code reset every round.
* **FP, full-prompt GPT-2, no latent.** The same operation strings in one context, final answer, example-matched and
  compute-matched.

**Tests.**
* Depths N = 8, 16, 32, 64, 128.
* 2,000 examples per depth for the full panel, or 10,000 if the pilot allows (frozen before training). The nulls always
  use 10,000.
* **Nulls on the trained D1.** T = 0, 1, 2 updates (latent update calls counted literally); first-op-only,
  last-op-only and two-op execution. Each upper 95 % bound ≤ chance + 0.02 = 0.1867.
* **Mechanism.**
  - Donor-code substitution at an interior step: the answer must be the donor's decoded state composed with the
    recipient's remaining suffix, ≥ 99 %.
  - Irrelevant substitution (a code with the same decoded state): ≤ 1 % of answers change.
  - Same-semantic donors from different depths: ≥ 99 % continuation agreement.
* **Audit.**
  - Decode the 6 hard codes (decoder argmax); require a bijection to S3.
  - Check all 36 hard transitions against group composition.
  - Zero errors plus correct initialisation and readout certify correctness at any length, by induction. This is
    conditional on stable argmaxes; margins are reported.

**Predictions.**
* **C1-1 (main pass).** D1 reaches ≥ 99 % hard-execution accuracy at every depth 8–128 in ≥ 4/5 seeds.
* **C1-2 (certificate).** The 36-transition audit is error-free in every seed that passes C1-1.
* **C1-3 (nulls).** All nulls and RESET are ≤ 0.1867 (upper bound) at the deep test depths.
* **C1-4 (mechanism).** The donor, irrelevant and same-semantic criteria are met.
* **C1-5 (the stronger claim).** FP trails D1 by ≥ 20 points at each extrapolation depth N ≥ 16, under both matchings.
  Otherwise only "constrained latent-execution success" is claimed.
* **Descriptive.** TAB vs D1. My expectation is that TAB also succeeds, so the result shows compositional latent
  execution, not a benefit from pretraining.
* **Likeliest failure (external).** A collapsed or stochastic solution with good marginal likelihood but bad greedy
  rollouts.

**Readings.**
* **C1-1 to C1-4 met.** A pretrained LM's latent state does the work (C1), is trained from final answers only (C4),
  runs at any length (C2, certified), and its codes are readable by the audit (C7 for the latent). Scope: streaming
  interface, small state space.
* **C1-5 met as well.** The latent machine beats the full-prompt LM.
* **C1-1 failed with a good likelihood.** Hardening is the bottleneck, not credit assignment.

**Order (Audit and external).** M = 12 codes (does spare capacity re-encode a counter?), then A5 (60 states, where the
toys deadlocked), then Design 2 (Boolean-tree registers; Ito 2608.31067 is nearest).

**Run (experiments session).** The pilot, then one wave: D1, TAB, RESET and FP × 5 seeds.

**Result: the pilot stop rule applies, so the panel is not run (2026-10-01; `c1/PILOT.md`; 11 configurations on pilot seed
99, all logged).**
* **The objective works.** TAB, under the same exact marginalisation and curriculum, is certified: 0 transition errors
  and 1.000 at every N 8–128 by update 1,500.
* **The GPT-2 cell fails rule acquisition.**
  - Default heads saturate on GPT-2's massive outlier activation (hidden norm ~200; cf. 2402.17762).
  - Zero heads with identical codes sit at a symmetric saddle (exactly ln 6).
  - With both fixed (a LayerNorm on the head input, distinct codes), the cell settles on a stable partial automaton:
    0.48–0.52 at every N, no bijection (decoded codes [1, 1, 2, 5, 0, 4]), 13–17 of 36 transitions wrong.
  - More updates or a higher lr for the new parameters do not help.
  - This is the external agent's "likeliest failure".
* **Positive mechanistic finding.** The partial automaton's accuracy is flat from N = 8 to 128. Hard discrete codes do
  not decay with length, unlike every continuous loop in §64 and §65. Discreteness removes drift; what remains is
  acquiring the correct transitions.
* **Reading.** C1-1 is not met. The bottleneck is the pretrained-GPT-2 parameterisation of the transition rule, not
  exact credit assignment (TAB) and not drift.

**Correction (2026-10-01 20:22 EDT; Audit, the experiments session's lr 2×2 diagnostic, and Theory's independent
`theory/c1_diag.py`).** The result above overstated the control.
* **TAB is certified in only 1 of 6 c1 runs** (pil_tab, seed 99, lr 1e-2). Seeds 1–2 at lr 1e-2 sit at 0.50 with
  13–14/36 wrong; lr 1e-3 runs are partial. The MLP is 0/6.
* So the exact-marginal objective with the 1-2-4-8 curriculum does *not* reliably find the S3 table. Its partial
  automata are HMM-style local optima, and GPT-2 is not singled out.
* This is the same two-attractor lottery as F1-S (§61b), now on a 36-parameter table.
* **Independent diagnostic** (`theory/c1_diag.py`, CPU, a minimal re-implementation of the objective;
  certificate = a product-automaton search over reachable (code, true state) pairs):

  | Cell, 8,000 updates, lr 1e-2 | M = 6 codes | M = 12 codes |
  |---|---|---|
  | Free table | 1/5 certified | **10/10 certified**, 1.00 at N 8–128, odd and even (`c1_diag_tab12.json`) |
  | MLP (one-hot code and operation → 256 → M) | 0/5 | 0/5 (`c1_diag.json`; grid in `c1_diag_mlp.json`) |

  - **Over-complete codes remove the local optima for a free table**: the classic over-parameterised-HMM remedy, and
    certified, so not a counter.
  - A shared-parameter cell (MLP) still fails, so a shared parameterisation adds a second difficulty.
* **Rule (Audit).** Only a recipe that certifies TAB in ≥ 9/10 seeds goes to a neural cell. M = 12 meets it for TAB.
  Whether any recipe certifies the MLP is the next question.
* **Disclosure: two certificate definitions (20:25 EDT; Audit).**
  - `c1_diag_b.json` and `c1_diag_c.json` were produced under a first definition: a bijection of all reachable codes
    plus 36 transitions checked against each code's decoder label. That definition wrongly checked the start code's
    arbitrary label, so it reports M = 12 runs with accuracy 1.0 as "not certified".
  - `c1_diag.json`, `c1_diag_tab12.json` and `c1_diag_mlp.json` use the fixed product-automaton certificate.
  - Accuracy to N = 128 at odd and even N corroborates independently: 23/23 M = 12 table runs score 1.000 (Audit).
* **Counter check (Audit's caution; `c1_diag.counter_check`, `c1_diag_counter.json`, 5 seeds).**
  - In 3/5 M = 12 tables, H(code | true state) = 0: one code per S3 state, a pure automaton.
  - In 2/5 there is small redundancy (0.04 and 0.32 bits), with mutual information between code and length mod 2/3/4,
    given the state, ≤ 0.015 bits.
  - So the spare capacity does not encode a length counter. The certificate already guarantees correctness at every
    length.
* **Shared-cell diagnostics so far (`c1_diag_mlp.json`, `c1_diag_bil.json`; 8,000 updates, lr 1e-2).**
  - MLP: 1/40 certified across M 12/24, τ-annealing 1/4 and lr 1e-2/3e-3.
  - Second-order (bilinear) cell, (U code) ⊙ (V op), the classic automaton-learning architecture (Giles et al. 1992):
    0/10 at M = 6 but **7/10 at M = 12**.
  - Multiplicative state–input interaction plus over-complete codes is most of what a shared cell needs.

**Amendment 67-2 (2026-10-01 20:53 EDT; user decision via Audit: a time-box, no fix loop).**
* **What remains.** The running bilinear grid (M = 24, τ0 1/4, 10 seeds) and exactly ONE more cell variant: the GPT-2
  cell (d1) at **M = 12**, with LayerNorm heads and distinct codes (the best pilot init), 10 seeds, 8,000 updates, on
  the local GPUs, certified by the product-automaton check.
* **Decision rule, fixed now.**
  - If any shared cell certifies in ≥ 9/10 seeds, §67b is registered as the single revision, with the full §67 panel.
  - Otherwise the line closes with this finding: an over-complete free table learns the discrete latent rule from final
    answers in 10/10 seeds, while shared networks do not reach 9/10 (MLP 1/40, bilinear 7/10 at best, GPT-2 0/11
    at M = 6 plus this M = 12 result, tf_small and gpt2_rand 0/6). Shared-cell acquisition is the C1 obstacle.
* **67-2's final variant: the GPT-2 cell at M = 12 (`c1/PILOT.md`; recorded 22:34 EDT).** 3/10 seeds certified (s1, s2,
  s8: 1.000 at N 8–128, 0 bad pairs).
  - The other 7 sit on flat partial automata (0.24–0.53, 5–15 bad pairs).
  - Below 9/10, so on its own it would not have triggered §67b. But the pretrained GPT-2 cell can acquire the rule:
    0/11 at M = 6 vs 3/10 at M = 12.
* **The bilinear grid (`c1_diag_bil24.json`, finished 21:06 EDT).** The second-order cell at M = 24 certifies in **10/10**
  seeds with τ annealed 4 → 1 over the first half, and 8/10 without annealing.
  - The 67-2 rule is met: a shared cell certifies in ≥ 9/10.
  - The recipe that makes a shared cell acquire the rule has three parts: multiplicative state–input interaction,
    over-complete codes, and temperature annealing. §67b ports it to the GPT-2 cell.

### 67b. C1 Design 1, the single revision: the GPT-2 cell with the recipe that certifies a shared cell (pre-registered 2026-10-01 21:06 EDT, before any 67b run; amendment 67-2's rule triggered)

**Why.** Under §67's objective, a shared second-order cell with 24 codes and τ annealing learns the S3 rule in 10/10
seeds, while every additive shared cell, GPT-2 included, falls into partial automata. This ports that recipe to the
GPT-2 transition cell.

**Changes from §67** (everything else unchanged: task, exact marginalisation, curriculum, hard execution, tests, nulls,
mechanism criteria, predictions C1-1 to C1-5, and readings).
* **M = 24 codes.** The certificate is the product-automaton check: decoded states cover S3, and every reachable
  (code, true state) pair after ≥ 1 operation decodes correctly. The 36-transition bijection audit applies only at
  M = 6.
* **Temperature.** T = softmax(logits / τ), with τ annealed linearly 4 → 1 over the first half of updates. Inference uses
  hard argmax.
* **Second-order readout.** Next-code logits = W · ((U · LN(h_code)) ⊙ (V · LN(h_op))). Here h_code and h_op are GPT-2's
  final hidden states at the code-embedding position and the operation-symbol position of the same call. U and V map to
  256, W maps to M. LayerNorm handles the outlier activation (§67 pilot).
* Distinct code initialisation, as in the best pilot.
* **Arms.** D1b (the GPT-2 cell with this recipe), TAB (M = 24, same τ schedule), RESET and FP, 5 seeds each. The panel
  follows a pilot on seed 99 (which may fix updates, batch and lr only, logged).
* **Added descriptive measure.** Counter check: H(code | state) and MI(code; length mod 2/3/4 | state) on the certified
  runs (`c1_diag.counter_check`'s definition).

**No further revision.** If D1b fails C1-1, §67's readings apply and the C1 line records shared-cell acquisition with
a pretrained LM as the obstacle, with the 10/10 bilinear toy cell as the existence proof that a shared cell can acquire
the rule.

**Amendment 67b-1 (2026-10-01 21:06 EDT, before running; Audit).** The recipe was chosen as the best cell of a grid on
seeds 0–9, so the existence claim must rest on seeds not used for selection.
* **Check, pre-stated.** The bilinear toy cell (M = 24, τ 4 → 1, lr 1e-2, 8,000 updates, `theory/c1_diag.py`) on 10
  fresh seeds, 100–109. It must certify in ≥ 9/10.
* If it does not, the "shared cell can acquire the rule" claim is withdrawn to "in 10/10 selection seeds only".
* **Result (21:29 EDT; `theory/results/c1_diag_bil24_heldout.json`). 8/10 certified: not met.**
  - The two failures (s103, s108) are near-correct automata: 0.97 and 0.93 at N 8, decaying to 0.58 and 0.51 at N 128,
    i.e. a few transitions wrong.
  - Per the rule, the claim is withdrawn to: the recipe certifies 10/10 on the selection seeds and 8/10 on held-out
    seeds (18/20 overall).
  - A shared cell acquires the rule in most seeds, but not with the ≥ 9/10 reliability of an over-complete free table.
  - §67b proceeds as registered, since this check gated the claim, not the registration. Its reading uses the
    withdrawn wording.

**Amendment 67b-2 (2026-10-01 21:22 EDT, before any D1b run; the experiments session's implementation query).**
* §67b's readout formula omitted the ReLU that the certified toy cell has (`theory/c1_diag.py`:
  out(ReLU((U c) ⊙ (V a)))).
* To port the recipe that was actually validated, the D1b readout is **W · ReLU((U · LN(h_code)) ⊙ (V · LN(h_op)))**.

**Amendment 67-1 (2026-10-01 19:34 EDT, before any of these runs; descriptive, no C1 claim).** Where does acquisition
break?
* Under §67's objective and curriculum, three further cells: an MLP (one-hot code and operation → 256 → 6), a small
  transformer from scratch (2 layers, d = 256, the d1 input format), and the GPT-2 architecture with random weights
  (124M, d1 format, the best pilot init).
* Pilot seed 99 plus seeds 1–2, at most 2,000 updates each; the TAB result above as the reference.
* Each cell is reported as certified or not (the 36-transition audit) with its hard-execution accuracy.
* **Reading.**
  - Random-init GPT-2 learns where pretrained GPT-2 does not: pretraining geometry (the outlier activation, entangled
    features) is the obstacle.
  - Only small cells learn: scale or depth is the obstacle.
  - Both GPT-2 variants are recorded as acquisition failures, and the line moves on.

**Pointer 67b-3 (2026-10-01 23:40 EDT; at Audit's request).** The experiments session's implementation amendment
67b-impl-1 (`c1/PILOT_67b.md`, d2a2fae) was recorded before any panel read. It makes three corrections.
* The d1 budget was 120 positions, which is correct only for M = 6; it is now 20·M.
* C1-5's compute-matched gap is scored on the re-run fpcmp2 (s1–5); the original fp_cmp is descriptive.
* The logged d1 gpt2_tokens at M ≠ 6 are corrected by ×M/6.

## 68. GSM8K step 2: our own GPT-2 arms on GSM8K-Aug — text CoT, Coconut and a written-state chunked arm (pre-registered 2026-10-01 20:55 EDT, after Literature's check 40b2ac9 and before any run; user decision via Audit)

**Why.**
* The first real task beyond toy and ProsQA. It has no cheap shortcut: heuristics ≤ 0.03, and a one-operation oracle
  reaches 0.155 (`theory/gsm8k_shortcuts.py`).
* Public latent checkpoints do use their latents on GSM8K (GSM8K amendment 2) but were trained with step supervision.
* This step reproduces the references with our own runs and adds the written-state arm that §66 found effective on
  state tracking.
* **None of these arms is C4.** All use GSM8K-Aug's equation steps. A final-answer-only arm is a later step.

**Prior** (`docs/LIT_GSM8K_STEP2_2026-10-01.md`).
* Text CoT 42.9 (official gsm_cot.yaml: full fine-tune, lr 1e-4, bs 32, 25 epochs, best on gsm_valid).
* Coconut 34.1 ± 1.5 (from the CoT checkpoint; c_thought 2, max_latent_stage 3, 3 epochs per stage, optimizer reset).
* CODI 43.7 (LoRA, 40 epochs), iCoT 30.0, Pause 16.4, No-CoT 16.5.
* GSM8K-Aug: 385,620 training examples.
* History cut on GSM8K: Markovian Transformers 2404.18988 (Llama scale) and LightThinker 2502.15589. A
  values-so-far written state with the equation history erased is not found at GPT-2 scale, so it is new as scoped.

**Arms** (GPT-2 small, original weights; seeds 1–3; official configs where they exist).
* **TC (text CoT).** gsm_cot.yaml as published: 25 epochs, checkpoint chosen on gsm_valid. Each seed's checkpoint also
  initialises its Coconut run.
* **CO (Coconut).** The official GSM schedule from that seed's TC checkpoint: c_thought 2, max_latent_stage 3 (≤ 6
  thoughts), 3 epochs per stage, optimizer reset.
* **WS (written-state chunked; the §66 WC0 analogue).**
  - Each call sees [question + "values so far: v1, v2, …"] and outputs either the next equation "<<expr=value>>" or
    "### answer".
  - Training: one example per gold step (teacher forcing; the same step supervision as TC). Tokens are matched to TC
    within ±10 %, with the epoch count logged.
  - Inference: generate an equation, parse its value, append it to the state and cut the history, until "###" or 10
    calls.
  - The written state is readable by construction (C7 for the state).

**Measures** (bench gsm8k harness, 1,319 test items, the normaliser for "####", commas and ".0").
* Accuracy per arm and seed.
* Shortcut baselines (≤ 0.03; one-op oracle 0.155).
* GSM8K-Platinum (2502.03461) as a sensitivity check.
* Accuracy by number of gold steps.
* **Compute.** Training FLOPs per arm (Literature: matched accounting must cover training) and inference FLOPs per
  problem (tokens processed, KV cache stated), forward passes and written tokens.
* **C1 controls for CO (the tightened definition).** Inference with K = 0, 1 and 2 latent thoughts against the trained
  K, plus a donor-latent swap: replace CO's thoughts with another problem's (answer agreement with the donor vs the
  recipient).

**Predictions** (medians over 3 seeds):
* **G1 (reproduction).** TC ≥ 0.40 (reference 0.429); CO ≥ 0.31 (reference 0.341).
* **G2 (CO's latents carry computation).** CO at K = 0 is ≥ 10 points below CO at the trained K. Reported with K = 1, 2
  and the donor swap; the K = 0 distribution-shift caveat of amendment 2 is stated.
* **G3 (the written state suffices).** WS ≥ TC − 0.03.
* **G4 (compute, descriptive; no C3 claim expected).** WS's inference FLOPs relative to TC's. WS re-encodes the
  question at each call, so it may cost more on GSM8K's short chains.
* **Readings.**
  - G1 met: our pipeline reproduces the public references, and later C4 arms can be compared within it.
  - G3 met: a compact written state of intermediate values loses nothing on a real task.
  - G2 met with the donor swap: the latents carry problem-specific computation (C1 on a real task, with step
    supervision).

**Budget and placement.** A throughput pilot (fixing only batch and precision, logged) sizes the wave. Estimate 40–80
GPU-h, ≈ $10–25 on Vast 3090/A4000-class hosts plus the local cards, behind §67-2.

**Amendment 68-1 (2026-10-01 21:28 EDT, before any §68 training; user decision via Audit: "Lean: reuse public
checkpoints").**
* **Why.** Measured cost: official code, 3090, fp32. CoT 1.73 h per epoch, Coconut stage 3 12.7 h per epoch: ≈ $110–170
  for 3 seeds, against the registered $10–25. My sizing was wrong by about ten times.
* **(1) TC is replaced by a public GPT-2 GSM8K CoT-SFT checkpoint**, e.g. the SIM-CoT or CODI release's CoT-SFT baseline
  (Literature: CoT-SFT 42.7).
  - It is identified first, and benched on bench gsm8k as the TC reference.
  - If no usable public checkpoint exists, the user decides (via Audit) before any from-scratch TC training.
* **(2) CO and WS, 1 seed each, are trained from that checkpoint** on a multi-GPU Vast host, with bf16 if the timing
  job supports it.
  - CO: the official Coconut GSM schedule to the official epoch-15 checkpoint.
  - WS: fine-tuned on the written-state format with checkpoint selection on gsm_valid. Its training GPU-hours are capped
    at CO's, and its training tokens are logged.
* **Budget: $15–30.** Training FLOPs are reported for both arms.
* **Predictions, re-scoped** (1 seed, so the comparisons are single-run and their intervals are item-level bootstrap):
  - G1: the public TC checkpoint scores ≥ 0.40 on our bench; CO ≥ 0.31.
  - G2: CO's K = 0 / 1 / 2 and the donor swap are descriptive (amendment 2 already answered it on public checkpoints).
  - G3: WS ≥ TC − 0.03.
  - G4: compute, unchanged.

**Status (21:37 EDT): §68 is PAUSED by user decision, not failed.**
* The user's words (via Audit): "find a working method that beats sota on smaller tasks then run gsm8k".
* No TC, WS or CO training was run.
* Standing results: GSM8K step 1 (public-checkpoint reproduction and latent ablations, GSM8K amendment 2) and the
  shortcut check.
* Before the pause:
  - Literature found no public CoT-SFT checkpoint in its search.
  - The experiments session found Dilgren & Wiegreffe's matched GPT-2 GSM8K-Aug CoT, Coconut and no-CoT set
    (2604.04902, MIT) and is benching it.
  - That set would make a resumed §68 cheap: train only WS.
  - **Literature's correction (8e7dfd4).** Their Table 1 (GPT-2, GSM8K-Aug): CoT 41.6, Coconut 33.1, No-CoT 16.8,
    CODI 42.2. Official recipe: CoT 25 epochs; Coconut from CoT checkpoint_6, 47 epochs. gpt2-gsm8k-cot is a faithful
    TC start and gpt2-gsm8k-coconut a matched CO reference.
  - They are benched now as descriptive leaderboard rows, with the K = 0/1/2 ablation on their Coconut (Audit).

## 69. RL-style training for discrete-latent acquisition from outcome-only reward (pre-registered 2026-10-01 23:11 EDT, after Literature's check cea50a1 and before any scored run; user request via Audit)

**Why.**
* Exact marginalisation gives the exact likelihood gradient, yet it falls into partial-automaton optima.
  - The S3 table at M = 6 certifies 1/5; the MLP 0/x; the bilinear cell at M = 6 0/10, and at M = 24 with τ annealing
    8/10 on held-out seeds (67b-1).
  - At S4 and A5 (premise check, `c1_group_*.json`) it certifies 0/3.
* Discrete-program learning from final answers has the same pathology: spurious programs. There, sampling-based
  updates with meritocratic weighting escape where both MML and REINFORCE fail (Guu 1704.07926, RandoMer; NSM 1611.00020;
  MAPO 1807.02322).

**Prior** (`docs/LIT_RL_ACQUISITION_2026-10-01.md`).
* As above.
* HMM local optima and their escapes: annealing (Ueda & Nakano; Smith & Eisner), Gibbs sampling, hard-EM (Spitkovsky),
  split–merge (Petrov).
* A global, non-neural alternative: spectral learning of weighted automata (Balle et al. 2014).

**Design.** `theory/c1_rl.py` (CPU). The machine, cells, curricula and certificate are as in `c1_group.py`.
* **Methods**, all trained on final answers only:
  - marg (baseline);
  - hem: hard-EM / expert iteration, K sampled code paths;
  - grpo: REINFORCE with a group baseline and an annealed entropy bonus;
  - rmer: RandoMer-style uniform weights over the correct sampled paths;
  - bon: best-of-4 restarts of marg, selected on training reward. One "seed" is the whole 4-restart procedure.
* **Budget matching (Audit).**
  - Each method runs a fixed number of updates, and its CPU seconds are recorded.
  - Seeds-to-certificate is compared at matched CPU time: a method using more than 1.5× marg's median CPU seconds is
    rescaled. Its update count is cut to match before the scored runs, and this is fixed on the pilot.
  - bon is reported with n = 4 and its 4× cost stated.
* **Hyperparameters, fixed on pilot seeds 990–994 only** (disjoint from the scored seeds).
  - Choices: K = 8, β0 = 0.05, lr 1e-2, and τ0 ∈ {1, 4}, picked per method on the pilot.
  - Updates: 8,000 for S3 and 16,000 for S4/A5.
  - Curriculum: std for S3, zhang for S4/A5.
* **Scored seeds.** 200–209 for S3; 200–202, then 203–212, for S4/A5.

**Stages, with the pass rule fixed now.** The certificate is tracked, not the reward (spurious automata).
* **Stage 1 (S3).** Cells MLP (M = 6) and tab (M = 6), where marg fails. A method passes if it certifies ≥ 9/10 scored
  seeds for at least one of these cells at matched budget, against marg on the same seeds.
* **Stage 2 (S4, A5).** For every method that passes Stage 1: bilinear cell, M = 2|G|, 3 seeds. ≥ 2/3 certified, then
  a 10-seed confirmation at ≥ 9/10.
* **Descriptive.** The running marg-with-Zhang-curriculum retry on S4/A5 is the marg reference. A spectral
  weighted-automaton baseline is optional, if time allows.

**Readings.**
* **A method passes both stages.** Discrete-latent acquisition from outcome-only reward scales to non-solvable groups.
  Target 1 is back on (beat SOTA on A5/S5 acquisition, with a certificate), and the method goes to the GPT-2 cell.
* **Stage 1 only.** RL-style updates fix small-state acquisition; scaling is still open.
* **Neither.** Outcome-only acquisition of the latent automaton is not fixed by these methods; Target 1 is dropped.

**Amendment 69-1 (2026-10-01 23:18 EDT, before any pilot or scored run).**
* **Why-paragraph numbers corrected.** These are baselines; the design is unchanged.
  - Table at M = 6: 1/6 (the corrected §67 figure), not 1/5.
  - MLP: the "0/x" placeholder should read:
    - 0/8 at M = 6 (`c1_diag.json`, `c1_diag_b.json`; pre-fix certificate, which can only add failures);
    - 1/40 at M = 12/24 over τ0 × lr (`c1_diag_mlp.json`).
  - The bilinear 8/10 held-out figure was at M = 24 with τ0 = 4.
  - Since registration, the marg-with-Zhang retry on S4 has given 0/3 (`c1_group_S4_zhang.json`); A5 is still running.
* **Placement.** One Vast host with many CPU cores (≤ $0.20/h), CPU only.
  - One torch thread per process, with concurrency ≤ the host's cores. `c1_rl.py` gained `--threads` and records
    process CPU seconds (`proc_sec`).
  - Budget matching uses `proc_sec`.
  - The dead duplicate `run()` was removed (no behaviour change).
* **Calibration (pilot seed 990, τ0 = 1, 8,000 updates).** One run per method and cell on the host.
  - For each method m and cell: if proc(m) > 1.5 × proc(marg), then U_m = round(8,000 × proc(marg) / proc(m)); otherwise
    8,000.
  - hyb splits U_m in half (marg, then hem).
  - bon is 4 × 8,000 marg updates (4× cost, stated with the result).
  - The U_m values are written to `theory/results/c1_rl/calib.json` before the pilot.
* **Pilot selection.**
  - For each (method, cell), marg included, choose τ0 ∈ {1, 4} by certified pilot seeds out of 5.
  - Ties go to the higher mean hard accuracy at N 32, then to τ0 = 1.
* **Stage-1 discriminating condition (tightening).** A pass on a cell counts only if marg, at its own chosen τ0, certifies
  ≤ 6/10 on the same scored seeds and cell. Otherwise the cell shows no acquisition failure to fix, and it is
  reported descriptively.
* **Descriptive extra.** Each sampled method is also run at the full 8,000 updates on the scored seeds (unmatched; not
  a pass).
* **Stage 2 details.**
  - Bilinear cell, M = 2|G|, τ0 = 4 for all methods (the 67b recipe for this cell), Zhang curriculum, 16,000 updates.
  - Matched counts come from a fresh seed-990 calibration on S4 (A5 uses the same ratios).
  - Seeds 200–202, with marg on the same seeds and host as the reference.
  - The std curriculum is run alongside as descriptive only; the registered pass uses Zhang.

**Amendment 69-2 (2026-10-01 23:25 EDT, before any Stage-2 run or result; Audit and Literature): a spectral weighted-automaton baseline, descriptive.**
* **Learner.** Spectral learning of a vector-output weighted automaton (Balle et al. 2014), `theory/c1_spectral.py`.
  - Data: n final-answer problems from the same training distribution. For each problem the update index is drawn
    uniformly, the length follows the run's curriculum, ops are uniform, and the label is the final element.
  - Hankel blocks over prefixes P = Σ (the |G| length-1 strings), suffix ε and outputs Y = G:
    - H[p, y] = empirical P(y | p);
    - H_a[p, y] = empirical P(y | p a);
    - strings never seen contribute 0.
  - Rank-|G| truncated SVD, H ≈ U S Vᵀ. Then A_a = (H V)⁺ H_a V and decoder B = (H V)⁺ H.
  - Execution: the state after the first op x₁ is row x₁ of H V; apply A_{x₂} … A_{x_N}; predict argmax(state · B).
  - The estimator therefore uses only problems of length ≤ 2.
* **Certificate (the analogue of the product-automaton one).**
  - BFS from the |G| (start state, true element) pairs under every A_a. States are deduplicated after rounding to
    1e-6 of their norm.
  - Certified iff the reachable set closes within 20·|G| distinct states and every reached pair decodes correctly.
  - Hard accuracy is also reported at N 8, 9, 32, 128.
* **Grid.**
  - Groups S3, S4, A5; curricula std and zhang (each run's schedule: 8,000 updates for S3, 16,000 for S4/A5).
  - n ∈ {10³, 10⁴, 10⁵, 10⁶}; data seeds 200–202.
  - For comparison, the neural runs see 256 problems per update: 2.0 M (S3) or 4.1 M (S4/A5).
* **Reading (descriptive, no pass rule).**
  - **Spectral certifies S4/A5 at n ≤ the neural data budget.** Learnability from final answers is not in question. A
    neural "acquisition" win is then about acquiring the automaton in a gradient-trained, GPT-2-compatible cell, and
    is framed that way.
  - **Spectral fails, or needs n beyond that budget.** This supports the neural route.
  - The baseline exploits short, exact (noise-free) problems. The neural objective sees the same data.

**69-2 result (2026-10-01 23:29 EDT; descriptive; `theory/results/c1_spectral.json`, log alongside).**
* Spectral WFA learning recovers every group exactly from final answers, once every length-2 pair has been seen. The
  first n (of 10³–10⁶) at which all 3 seeds certify:
  - S3: 10³ under both curricula;
  - S4: 10⁵ (std) and 10⁵ (zhang; 2/3 already at 10⁴);
  - A5: 10⁶ (std; at 10⁵ it reaches 3,598–3,599 of the 3,600 pairs, 0.93–0.96 at N 128, not certified) and 10⁵ (zhang).
* Every certified fit closes on exactly |G| states with N-128 accuracy 1.000. That is ≤ 1/4 of the neural runs' data
  (2.0 M / 4.1 M problems).
* **Framing, fixed now.** Learnability from final answers is not in question for these groups. Any §69 neural success
  is about acquiring the automaton in a gradient-trained cell that can sit inside GPT-2. The spectral learner has no
  such cell, and it needs complete short-string coverage, i.e. exact supervision of the whole Cayley table at length 2.

**§69 Stage-1 result (2026-10-01 23:38 EDT; scored by Theory with `theory/c1_rl_score.py` under the 69-1 rules; raw
`theory/results/c1_rl/`, summary `theory/results/score_s69_stage1.json`).**
* **No method passes. Reading "Neither": Target 1 is dropped, and Stage 2 is not run.**
* Scored seeds 200–209, at matched CPU (median process seconds 16.6–17.8 on the table cell, 20.8–23.2 on the MLP):

  | cell (M = 6) | marg | hem | grpo | rmer | hyb | bon (n = 4, 3.2× CPU) |
  |---|---|---|---|---|---|---|
  | table | 0/10 | 0/10 | 1/10 | 1/10 | 1/10 | 5/10 |
  | MLP | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

* Unmatched runs at the full 8,000 updates (descriptive): table 0–2/10, MLP 0/10.
* The pilot was consistent: on the table, 0/5 for marg and ≤ 1/5 for each sampled method.
* **Interpretation.** Sampling-based outcome-reward updates (hard-EM, group-baseline REINFORCE with entropy, RandoMer
  weighting) do not escape the partial-automaton optima. They are no better than exact marginalisation, even unmatched.
  - Only restarts help: bon reaches 5/10, as the §67 lottery predicts (per-restart success ≈ 16%).
  - Together with 69-2, the obstacle is not data or identifiability: the spectral learner recovers A5 from ≤ 10⁶
    problems. It is the optimisation landscape of a gradient-trained cell.
  - What worked before is extra capacity: over-complete codes, the free table at M = 12 (10/10), and the bilinear cell
    at M = 24 with τ annealing (10/10 selection, 8/10 held out). It did not scale to S4/A5 (0/3, std and Zhang).

**Note 69-3 (2026-10-01 23:39 EDT; labels only, definitions and result unchanged; Audit and the external reply
`docs/EXTERNAL_IDEAS_RL_2026-10-01.md`).** Read the §69 method names as follows.
* `hem` is on-policy success reinforcement, not Viterbi hard-EM. Successful on-policy paths are posterior samples, so
  its expected update is the marginal-likelihood gradient up to batch weighting.
* `rmer` is occurrence-weighted success reinforcement, not Guu's RandoMer: duplicate successful paths are not
  deduplicated, so frequent paths still dominate.
* `grpo` is REINFORCE with a self-inclusive group mean, which scales the gradient by 1 − 1/K. It has no clipping.
* With this reading, the Stage-1 null says that sampling-based estimates of essentially the likelihood or reward
  gradient do not escape the optima. It is not a test of deduplicated meritocratic search or of objectives that differ
  in kind.
* The exact expected-reward control Audit asked for is moot for §69, since no method won. It moves to the next design.

## 70. Discrete-latent acquisition, one time-boxed toy round: (A) operator-conditioned exact likelihood and (B) tempered posterior into Viterbi (pre-registered 2026-10-01 23:46 EDT, after Literature's check f9f6bce and before any scored run; proposals A and B of `docs/EXTERNAL_IDEAS_RL_2026-10-01.md`, triaged by Audit)

**Why.** §69 ruled out sampling-based estimates of the likelihood or reward gradient, and 69-2 ruled out data. The two
untested explanations each get one shot.
* **A: parametrisation.** How the op enters the transition (additive vs multiplicative vs a per-op matrix) decides
  acquisition, not exploration.
* **B: optimisation path.** Exact posterior continuation (η-sharpening, then Viterbi) escapes the partial-automaton
  optima that plain likelihood falls into.

**Prior** (f9f6bce).
* A: second-order RNNs represent every DFA, while single-layer first-order ones cannot (Goudreau et al. 1994; Giles
  et al. 1992); per-input matrices (ISAN 1611.09434); mRNN; multiplicative integration (1606.06630). No study compares
  these cells under exact likelihood from terminal labels with a full automaton certificate.
* B: Unified EM (Samdani et al. 2012), deterministic annealing (Rose 1998; Ueda & Nakano 1998), Lateen EM (Spitkovsky et
  al. 2011). Known risk: hardening locks in a wrong partition.
* J vs log J: Misra et al. 2018 (1809.01299), Guu et al. 2017, RAML (1609.00150).

**Design.** `theory/c1_acq.py` (CPU), shared by both parts.
* Settings: S3, std curriculum, 8,000 optimizer updates, batch 512, Adam lr 1e-2, τ 4 → 1 linearly by update 4,000.
* Evaluation: product-automaton certificate and hard accuracy at N 8–128. Also length-stratified soft NLL and the
  transition-row entropy every 500 updates (Literature: report where entropy collapses).
* **A (M = 24, exact marginal likelihood).** Cells:
  - `add` (MLP on [code; op]) — the matched control;
  - `mul` (multiplicative);
  - `opb`, L(c, a, j) = e_out(j)ᵀ B_a e_in(c) + b_j, rank 12.
  - Their transition-parameter counts are matched: 1,509 / 1,480 / 1,464.
  - Initial transition-logit std is rescaled to 0.1 in all three.
  - Reference (unmatched, 4,632 parameters): `opb` at full rank 24.
* **B (historical MLP cell, hid 256, M = 12).** Methods:
  - `temper`: outer steps of two updates on one batch.
    - First 1/8: marginal likelihood.
    - Then exact posterior counts from the detached model under q ∝ w^η (unnormalised powered potentials): η 0.5 → 2
      over [1/8, 1/2), η = 4 over [1/2, 3/4), Viterbi over the last 1/4.
    - Loss 0.25 (−log J) + 0.75 · (cross-entropy to the fixed counts).
  - `eta1`: η = 1 throughout, no Viterbi.
  - `mml2`: marginal likelihood, each batch used for two updates. This is the paired control (same init and the same
    example stream as `temper` and `eta1`).
  - `mml`: one update per batch.
  - `exr`: exact expected reward, max mean J.
  - The posterior counts were checked against brute-force enumeration (edge marginals, log Z, Viterbi path).
* **Seeds.** Development 1300–1302, confirmation 1400–1409; both checked unused. Smoke runs used seed 9999 only.
* **Placement.** One Vast CPU host (≤ $0.20/h), one thread per process. Every arm runs on all 13 seeds.

**Rules (fixed now).**
* **A candidate.** Chosen on the development seeds from {`mul`, `opb`} by certified count, then mean accuracy at N 128,
  then `opb`.
* **A passes** if the candidate certifies ≥ 9/10 on seeds 1400–1409 and at least 3 more than `add` on the same seeds.
  Report paired outcomes and Wilson 95 % intervals.
* **B passes** if `temper` certifies ≥ 9/10 on 1400–1409 and at least 3 more than `mml2`, paired. Report `eta1`,
  `mml` and `exr` alongside.
* **Close rule.** If neither passes, the toy-acquisition line closes: no proposal C, no further variants.

**Readings.**
* **A passes.** Parametrisation, not exploration, was the obstacle. The GPT-2 S3 cell gets an op-indexed or
  multiplicative head arm, in a separate registration (S3 only; Audit's point 5).
* **B passes (and A does not).** Posterior continuation escapes the optima. Next, replicate on the bilinear cell at
  M = 24, then the GPT-2 S3 cell (separately registered).
* **Both pass.** Both readings hold; the GPT-2 arm uses A's head first, since it is cheaper.
* **Neither passes.** Close the line, as above.

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
