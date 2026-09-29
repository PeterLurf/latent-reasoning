# Research log â€” readable vs usable alternatives in latent reasoning

Entries are chronological. Predictions are recorded before the corresponding run.
Sampling unit for all inferential statistics: independently generated graph
(each has its own structural family with probability > 0.99; families are
split-disjoint across train / discovery / confirm).

---

## 2026-09-23 â€” Week-1 items

### Literature audit (novelty check against the proposal's gap)

Checked current versions of the four cited papers plus a search for newer work.

| Paper | What it establishes | Does it already do branch-level shared-state edits or a width sweep? |
|---|---|---|
| Coconut (2412.06769) | Continuous-thought recurrence; BFS-like interpretation on ProsQA | No |
| Emergence of Superposition (2509.23365) | Training dynamics of a 2-layer continuous-thought transformer on reachability; "index-matching logit" | Theory/dynamics; no branch-specific causal edits |
| Dynamics within latent CoT (2602.08783) | Step-level zero-overwrite do-interventions on Coconut/CODI (GSM8K, CSQA, StrategyQA); probe-vs-answer readout gap | No graph tasks; holistic step ablation, no donor patching of components |
| Illusion of Superposition? (2604.06374) | Superposition signs only when trained from scratch; shortcuts in fine-tuned; depth/width ablations; entity-level probing | No branch interventions; branching not varied |
| **Observable Patterns Are Not Explanations (2606.12689)** â€” *not in the proposal table* | Coconut/CODI on ProsQA vs matched controls incl. **Pause-as-Thought**; BFS signature also appears in PaT; per-position causal tracing; gradient-subspace ablation with rank-matched random control | No per-branch/per-node edits, no branch selectivity, no width manipulation |

**Update to plan:** the gap survives, but 2606.12689 makes the pause-token control
mandatory rather than optional ("BFS signature" alone is known not to be
recurrence-specific), and the proposal table should cite it.

### Task / generator (implemented `latent/graphs.py`)

* 10 branches Ã— 6 nodes (random rooted trees, chain-biased), start hub `s`, decoy hub `s'`.
  Width W âˆˆ {1..8} = number of branches attached to `s`; the rest hang off `s'`.
  Every graph has 62 nodes and 60 edge tokens regardless of W, and â‰¥2 shape-matched
  decoy branches (an earlier NB=8 design had no negatives at W=8 â€” caught by tests).
* Families = multiset of (attached?, canonical tree shape) â†’ hashed into 20 buckets:
  train 14 / discovery 3 / confirm 3. Renamed or re-ordered copies of a family can
  never cross splits. 20 000 draws â†’ 19 990 distinct families.
* Counterfactual partner = one edge token changes source (s â†’ s', s â†’ graft into a
  decoy branch, or s' â†’ s). Verified by BFS that exactly the edited branch changes.

### Instrument validation (`python -m latent.tests`, all pass)

generator == independent BFS on 3000 graphs; split disjointness; 50/50 query balance;
KV-cache incremental == full causal recompute; forked queries == separate sequences;
identity patch bit-exact; patching thought k leaves thoughts â‰¤ k untouched;
batch-composition invariance; shared-prefix training path == separate runs.

### Pilots 1â€“2: training does not leave chance (negative, diagnosed)

* Plain end-to-end training (binary reachability only, untied src/dst/node embeddings,
  GPT-2 init) stayed at exactly ln 2 for 3â€“5k steps Ã— 512 graphs in every variant
  (coconut, pause, K=0, lr 1e-3/3e-4, depth curriculum, shared "slot" embeddings,
  PyTorch-default init).
* Diagnosis: the model can memorise a fixed batch (gradients flow). A *single*
  identity match ("does v have outgoing edges?") is learned in ~400 steps. The
  *composition* of two matches needed even for 1-hop reachability ("find v's
  incoming edge" âˆ˜ "is its source s?") never left the class prior in 1500 steps
  (K=0 and K=2 alike): a product-of-two-unlearned-circuits saddle.

### Decision: auxiliary PARENT lookup as a training scaffold (documented deviation)

* Added query type PARENT(v) â†’ id of v's parent (tied id readout), forked against the
  **prefix cache only** (edges + START; no latent positions), weight 1 for 3000 steps
  then linearly annealed to 0 by step 6000.
* It supervises an input-lookup primitive, not reachability, not the frontier, and
  never touches latent thoughts. Comparable in spirit to Coconut's stage-0 language CoT.
* **Claim restriction:** results concern models trained with this scaffold. It plausibly
  *favours* the "input reconstruction" alternative (query-time backward lookup), which
  the intervention design must therefore be able to detect (depth-resolved effects).
* Also adopted: shared identity table in role slots ("slots" embedding), PyTorch-default
  init, CUDA-graph captured training step (~2Ã— faster; numerics unchanged).

### Round 2 (PARENT scaffold only): depth-1 learned, depth â‰¥ 2 not

* PARENT lookup learned in < 500 steps. Depth-1 reachability then learned
  (100% by ~3k steps) â€” necessarily *at query time* (probe AUC on thoughts â‰ˆ 0.50,
  donor patches have zero effect). Depth â‰¥ 2 stayed at the prior through 6k steps.
* Interpretation: depth 2 in two layers requires START to gather the children of s
  and the query to check membership of its parent â€” again two unlearned matchings.

### Decision: add CHILDREN(v) lookup + accuracy-gated depth curriculum

* CHILDREN(v): multi-label over ids of v's children, prefix-only fork, v drawn from
  branch nodes and s' â€” **never s** (so reachability/frontier of s is never a target).
* Gated curriculum: start at within-branch depth â‰¤ 1 (star branches); advance when
  held-out delayed balanced accuracy at the current stage â‰¥ 0.9 (checked every 500
  steps); full depth after stage 4. Aux weight 1 for 10k steps, annealed to 0 by 15k.
* Bug found and fixed: at stage 1 all branches are stars, so only 8 families exist and
  the family-split rejection sampler could loop forever (and would restrict training to
  the widths whose family hashed into train). Stage-1 curriculum graphs now skip split
  filtering; all evaluation is at full depth where such graphs have probability ~1e-9.
* Baseline controls (pause, K=0) must replay the main run's realised curriculum
  schedule rather than gate on their own accuracy (else they see different data).

### Pre-registered predictions for confirmation (written before any confirm-split run)

Smoke test on an intermediate seed-0 checkpoint (discovery graphs only) showed full-vector
donor patches with collateral â‰ˆ targeted, and targeted falling from 0.16 (W=2) to 0.01
(W=8). Hypothesis **H-norm**: branches share the thought through softmax-normalised
attention averaging, so each branch's share scales ~1/W. Predictions:
1. Targeted effect of a single-branch edit decreases with W.
2. Removing branch i *raises* the other branches' P(reach) (signed collateral < 0),
   because their normalised share grows; inserting lowers it (signed collateral > 0).
3. Probe decodability degrades more slowly with W than causal selectivity
   (readable-but-not-separately-usable at high W).
Alternative **H-damage**: collateral is unsigned general damage â†’ signed collateral â‰ˆ 0
with |collateral| > 0, and the identity / same-status controls would also show effects.

### Disclosure: pipeline smoke tests touched confirm-split graphs

Two pipeline smoke tests (intermediate seed-0 checkpoints at ~8k and ~16k steps, n=60
graphs/width, W âˆˆ {2,8}) drew *confirm-split* graphs before `split` was made a protocol
parameter. Seen: full-vector patch at k=1 had targeted â‰ˆ collateral; probe-subspace
patches â‰ˆ 0; identity control exactly 0. No protocol choice below was changed because of
these numbers (all conditions were already specified in PROTOCOL.md). Final confirmation
uses final checkpoints and a fresh sampling seed; smoke tests since then use discovery graphs.

### Seed 0 (runs/r3_pc_gated_gpt2) â€” competence, discovery, frozen protocol, confirmation

**Competence (confirm split, n=4096):** delayed bal-acc 0.900 at K'=6; 0.74 at K'=0 (inference
truncation). Distance 1..6 accuracy at K'=6: 1.00/1.00/0.93/0.70/0.59/0.57. Oddity: distance-1
queries are ~0% at every truncated budget and 100% only at K'=6 â€” the model reads depth-1
membership from the last latent position. Truncation is therefore not a clean ablation.

**Discovery (discovery split only).** Reach-probe matched AUC 0.67â€“0.74 pooled, but within-graph
per-branch AUC only 0.55â€“0.63 (controls 0.49â€“0.52). Full-vector donor patches small (k=1:
targeted 0.06, collateral 0.03). Probe-filter subspace: exactly 0 effect at every step.
Exploration: the probe's 6 readout directions hold 0.3% of the donor-difference energy â€”
*less* than a random 6-dim subspace (2.3%). Donor differences are low-rank and shared across
graphs (top-20 PCs 77â€“98%). A PCA-20 subspace reproduces the full patch; the branch-specific
**Haufe pattern** subspace (Cov(h)Â·W_probe, 6 ids) reproduces ~88% of it. Filters â‰  patterns.
These discoveries added the `pattern` and `pca` conditions *before* freezing
`docs/protocol_frozen.json` (k*=1 by the pre-set rule).

**Confirmation (frozen protocol, confirm split, 300 graphs/width, fresh seed).**
* Targeted (full, remove, k*=1): W1 0.261 â†’ W2 0.085 â†’ W4 0.059 â†’ W8 0.026 (monotone â†“).
  Pattern: 0.212 â†’ 0.052 â†’ 0.052 â†’ 0.025. Probe filter: 0.000 at every W. Random rank-6/20:
  â‰¤ 0.009. Norm-matched isotropic: targeted â‰¤ 0.011 but collateral 0.013â€“0.046 (generic damage).
* Depth-resolved (k*=1): distance 1â€“2 unaffected (Â±0.001); distance 3â€“6 at W=2:
  0.06/0.19/0.32/0.26; W=8: 0.03/0.06/0.07/0.07. Step k=0 at W=2: targeted 0.405, distance-3
  0.70, collateral 0.077 â€” but k=0 is nearly inert at Wâ‰¥4.
* **H-norm prediction 2 confirmed:** signed collateral < 0 under removal (W2 âˆ’0.060, W4 âˆ’0.016,
  W8 âˆ’0.006) and > 0 under insertion (+0.096, +0.018, +0.009): removing one branch makes the
  others *more* reachable. Identity control 0 exactly; same-status control 0.002â€“0.013.
* Downstream trace: after a full k=1 edit the edited branch's probe margin collapses at k=2â€“3
  (W2: 0.38 â†’ 0.03) but partly recovers by k=4â€“6 (0.22â€“0.29): re-derivation from the intact cache.
* Upfront format: full edit of the target's branch moves the answer 0.14â€“0.18 at every W with
  collateral â‰ˆ 0.001; the delayed-format pattern subspace is inert there (different directions
  when the target is known).
* Provisional "selective" bar (â‰¥ 0.10 targeted, â‰¤ 0.03 collateral, pooled over the branch) is
  **not met** for any delayed-format condition; it is exceeded on deep nodes only.

**Exploratory mechanism check (discovery split):** layer-0 attention from LAT_1..3 onto each
reachable branch's edges, excess over a decoy branch: W1 0.25, W2 0.12, W4 0.04, W8 0.016 â€”
a roughly fixed attention budget (0.25 â†’ 0.13 total) divided among W branches.

### Seeds 1 and 2 (runs/main_s1, runs/main_s2) â€” same frozen pipeline

* Competence (confirm split): delayed 0.918 / 0.926, upfront 0.873 / 0.896.
* Frozen site rule gave k* = 0 for both (seed 0: k* = 1). Cross-seed figures therefore use
  each run's own k*; the step profile (k = 0..4) is reported for all seeds.
* Targeted (full, remove) W2 â†’ W8: s1 0.300 â†’ 0.059; s2 0.324 â†’ 0.114 (s0 0.085 â†’ 0.026).
* Probe filters â‰¤ 0.001 everywhere; branch-pattern energy share 0.53 / 0.49 (s0 0.35) vs
  filters 0.003 / 0.002 vs random-6 0.022.
* Selectivity bar (â‰¥ 0.10 targeted, â‰¤ 0.03 collateral) met only by **s2 at W = 2**
  (full 0.324/0.022; pattern 0.265/0.019; insert 0.288/0.017). s0, s1: not met at any W.
* Signed collateral: sign pattern (remove < 0, insert > 0) in s0 and s2 at every W; s1 at
  W 2â€“3 and 7â€“8, â‰ˆ 0 at W 4â€“6.
* Depth: distance-1 never affected, distance-2 â‰¤ 0.05; effects on distance â‰¥ 3 (W2 up to 0.60â€“0.66).
* Downstream recovery: s0 partial, s2 partial, s1 none (edited branch stays suppressed).
* Upfront format: target-branch edits move the answer 0.64â€“0.74 (s1), 0.38â€“0.51 (s2) with
  collateral â‰¤ 0.03 except s2 W2 (0.053); delayed-format pattern subspace â‰¤ 0.027 there.
* Seed variation in effect size (~4Ã—) is large relative to within-checkpoint CIs; the
  qualitative pattern is what replicates.

### Controls (single seed each, same data/scaffold/realised curriculum schedule as seed 0)

* **No latent (K=0):** delayed 0.701, upfront 0.668; distance 1/2 = 1.00/0.94, distance 3â€“6 =
  0.10â€“0.12 (almost always "unreachable"). Extra positions are required beyond two hops.
* **Pause tokens (K=6, no feedback):** delayed 0.912, upfront 0.815. Distance 1â€“6 =
  1.00/1.00/0.99/0.77/0.29/0.27 vs recurrent seeds 1.00/1.00/0.92â€“0.97/0.70â€“0.80/0.57â€“0.70/0.38â€“0.60.
  â†’ Overall accuracy does **not** separate recurrence from pause positions; the recurrent
  advantage is confined to 5â€“6-hop queries and the upfront format. Consistent with the
  2606.12689 caution that pause-as-thought reproduces much of what is attributed to recurrence.
  Thought-level interventions are not applicable to the pause model (no thought is fed back).
* Operational note: two CUDA-graph jobs on one GPU cost ~2Ã— combined throughput under WDDM;
  the pause run was suspended (NtSuspendProcess) while seed 1 finished, then resumed.

### Final assessment against the proposal's decision criteria

| Criterion | Result |
|---|---|
| Competence â‰¥ 85% (delayed, confirm split) | Met, all seeds (0.90â€“0.93) |
| Latent computation contributes | Met vs K=0; vs pause only at 5â€“6 hops / upfront |
| Minimum positive result: â‰¥ 2 alternatives in one shared state with distinct predicted effects, replicated | Met qualitatively in 3/3 seeds (targeted > collateral at W â‰¥ 3; depth-specific effects; controls â‰ˆ 0) |
| Provisional selectivity bar (â‰¥ 0.10 targeted, â‰¤ 0.03 collateral) | Met in 1/3 seeds, W = 2 only |
| Bounded test of width | Done: targeted and selectivity decline monotonically W 2â†’8; decodability declines more slowly |
| Valuable alternative result | Readable â‰  usable: probe filters inert, activation patterns causal; shared attention budget âˆ 1/W |

### Round 3 early observations

* PARENT+CHILDREN, GPT-2 init: star stage passed at 1k steps; at 4k steps distance-3
  accuracy 0.54 on full-depth held-out graphs (stage 2). PARENT-only run: depth 1 only.

---

## 2026-09-24 â€” Part II: why branch edits weaken with width (handoff plan docs/NEXT_STEPS.md)

### Preflight
* Reproduced all 862 published seed-0 confirmation point estimates exactly (max |diff| = 0.0).
* Split audit: zero family overlap train/discovery/confirm; generator sha256 prefix recorded.
* Donor audit: W=1 removal/insertion involves an untrained W=0 state â†’ W=1 now reported only as a
  distribution-shift test. Swap insertion at W=8 creates W=9 donors (flagged).
* New instrument `latent/flex.py` (thought edits at any steps, KV overrides by position and reader,
  attention key bias); validated: equals model.run, full-route override == donor run bit-exact.

### Protocol 2
* Hypotheses/predictions written in docs/PROTOCOL2.md before discovery; after discovery, frozen
  numerical choices + discovery-derived predictions P1â€“P8 in docs/protocol2_frozen.json, then
  addendum A (P9 composition, P10 gain model) frozen before their fresh runs.
* Fresh confirmation graphs: confirm buckets, seed 271828, excluding every family drawn by any
  earlier confirm-split sampler (replayed in followup_run.used_confirm_families).
* Bug fixes during discovery (no confirmation data involved): pause models lacked probe assets;
  driver ignored --k for pause models.

### Fresh-confirmation outcomes (results/followup_confirm.json)
* P1 route shift â€” pass all (latent share â†“, query-graph share â†‘ with W).
* P2 no dose rescue â€” pass all. P3 pattern â‰« filter at matched norm (6â€“57Ã—) â€” pass all.
* P4 swap â€” partial: signed collateral always cancels; unsigned only reduced at W=2 in some seeds.
* P5 semantics â€” pass at every frozen site; fails only at seed 0's extra common site k=0 (edits too weak).
* P6 competition â€” pass s0, s2; s1: uniform all-edge boost also lifts reachable branches (+0.02).
* P7 no gain restoration â€” pass all. P8 geometry adds nothing â€” FALSIFIED for targeted effects in
  s0 (+0.07 RÂ²) and s2 (+0.11); holds for collateral.
* P9 composition â€” pass all. P10 gain 1/âˆšW â€” pass s1, s2; s0 prefers 1/W (Î² = 0.87).
* P1b pause query-time share â‰¥ 0.85 â€” pass pause_s0; FAIL pause_s1 (START cache carries ~50%).

### Post-hoc (not pre-registered) depth breakdown of the cache factorial, same fresh graphs
* 1-hop nodes: always query-time; â‰¥4-hop nodes: always latent pathway (TL â‰ˆ 0.72â€“1.0) at every W.
* The latent/query boundary moves inward with W (2â€“3-hop nodes switch to query-time resolution).
* The model's own clean counterfactual on deep nodes collapses with W (s0, 4 hops: 0.79 â†’ 0.19).
* Per deep node, a thought-only edit reproduces a similar share of the clean effect at every W
  (5â€“6 hops: s0 0.45â†’0.77, s1 â‰ˆ0.95, s2 â‰ˆ0.65). â†’ The width effect on branch-averaged edits is
  compositional (collapsing deep-node sensitivity + inward boundary), not declining edit efficacy.

### Control replication (seed-matched curriculum schedules)
* Pause s1/s2 and no-latent s1/s2 trained. Competence is strongly seed-dependent: pause_s1 never
  passes 2 hops; no-latent s1 reaches 0.91 overall (3 hops 0.97) with no latent steps at all.
* Per-distance *recall* can be inflated by a "reachable" bias (pause_s1: 0.98 recall at 6 hops,
  0.50 depth-matched). New depth-matched balanced accuracy (latent/competence_matched.py) used instead.
* Part I's single-seed control conclusions (pause â‰ˆ recurrent to 4 hops; no-latent fails > 2 hops)
  were partly seed luck. Final depth-matched result (all 9 models): 6 hops — recurrent 0.59–0.66 vs all six controls
  0.50–0.56; 5 hops — no-latent seed 2 reaches 0.67 (= weakest recurrent seed); 4 hops — no-latent
  s1/s2 0.86/0.88 (> recurrent 0.75–0.83). Recurrence buys cross-seed reliability and the deepest queries.


### Exploratory (discovery graphs): route decomposition in the upfront format (seeds 0, 1; seed 2 interrupted)
* When the target is given before the latent steps, the ANSWER token does not read the graph at
  all: query-time graph share Gq ≈ 0.00 at W = 2, 4, 8 and every target distance. The latent pathway
  (thought + START/TARGET/early-latent cache) carries 100% of the target branch's effect at every
  width. Seed 1: the thought alone (T) carries 100%; seed 0 (k* = 1): T carries the 1-hop targets,
  the cache route the rest. → The route shift is specific to the delayed-query diagnostic.

### Part III (PROTOCOL3) status
* V2 (4 layers) OOM'd on the fp32 RTX 2070; rerun on the bf16 RTX 4060 with the identical recipe.
* V1 (d=512) stalled at step 6400 under GPU memory oversubscription (captured graph + eager gate
  eval + a concurrent analysis job); restarted with evaluation chunks of 128 graphs (recipe unchanged).
* V3 (raw pre-LN feedback) trained: delayed 0.884, upfront 0.999 (discovery-split eval at 30k).
* Seed 3 (runs/main_s3) training on the 2070 for the held-out-seed replication (addendum C).
