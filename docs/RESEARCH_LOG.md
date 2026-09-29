# Research log — readable vs usable alternatives in latent reasoning

Entries are chronological. Predictions are recorded before the corresponding run.
Sampling unit for all inferential statistics: independently generated graph
(each has its own structural family with probability > 0.99; families are
split-disjoint across train / discovery / confirm).

---

## 2026-09-23 — Week-1 items

### Literature audit (novelty check against the proposal's gap)

Checked current versions of the four cited papers plus a search for newer work.

| Paper | What it establishes | Does it already do branch-level shared-state edits or a width sweep? |
|---|---|---|
| Coconut (2412.06769) | Continuous-thought recurrence; BFS-like interpretation on ProsQA | No |
| Emergence of Superposition (2509.23365) | Training dynamics of a 2-layer continuous-thought transformer on reachability; "index-matching logit" | Theory/dynamics; no branch-specific causal edits |
| Dynamics within latent CoT (2602.08783) | Step-level zero-overwrite do-interventions on Coconut/CODI (GSM8K, CSQA, StrategyQA); probe-vs-answer readout gap | No graph tasks; holistic step ablation, no donor patching of components |
| Illusion of Superposition? (2604.06374) | Superposition signs only when trained from scratch; shortcuts in fine-tuned; depth/width ablations; entity-level probing | No branch interventions; branching not varied |
| **Observable Patterns Are Not Explanations (2606.12689)** — *not in the proposal table* | Coconut/CODI on ProsQA vs matched controls incl. **Pause-as-Thought**; BFS signature also appears in PaT; per-position causal tracing; gradient-subspace ablation with rank-matched random control | No per-branch/per-node edits, no branch selectivity, no width manipulation |

**Update to plan:** the gap survives, but 2606.12689 makes the pause-token control
mandatory rather than optional ("BFS signature" alone is known not to be
recurrence-specific), and the proposal table should cite it.

### Task / generator (implemented `latent/graphs.py`)

* 10 branches × 6 nodes (random rooted trees, chain-biased), start hub `s`, decoy hub `s'`.
  Width W ∈ {1..8} = number of branches attached to `s`; the rest hang off `s'`.
  Every graph has 62 nodes and 60 edge tokens regardless of W, and ≥2 shape-matched
  decoy branches (an earlier NB=8 design had no negatives at W=8 — caught by tests).
* Families = multiset of (attached?, canonical tree shape) → hashed into 20 buckets:
  train 14 / discovery 3 / confirm 3. Renamed or re-ordered copies of a family can
  never cross splits. 20 000 draws → 19 990 distinct families.
* Counterfactual partner = one edge token changes source (s → s', s → graft into a
  decoy branch, or s' → s). Verified by BFS that exactly the edited branch changes.

### Instrument validation (`python -m latent.tests`, all pass)

generator == independent BFS on 3000 graphs; split disjointness; 50/50 query balance;
KV-cache incremental == full causal recompute; forked queries == separate sequences;
identity patch bit-exact; patching thought k leaves thoughts ≤ k untouched;
batch-composition invariance; shared-prefix training path == separate runs.

### Pilots 1–2: training does not leave chance (negative, diagnosed)

* Plain end-to-end training (binary reachability only, untied src/dst/node embeddings,
  GPT-2 init) stayed at exactly ln 2 for 3–5k steps × 512 graphs in every variant
  (coconut, pause, K=0, lr 1e-3/3e-4, depth curriculum, shared "slot" embeddings,
  PyTorch-default init).
* Diagnosis: the model can memorise a fixed batch (gradients flow). A *single*
  identity match ("does v have outgoing edges?") is learned in ~400 steps. The
  *composition* of two matches needed even for 1-hop reachability ("find v's
  incoming edge" ∘ "is its source s?") never left the class prior in 1500 steps
  (K=0 and K=2 alike): a product-of-two-unlearned-circuits saddle.

### Decision: auxiliary PARENT lookup as a training scaffold (documented deviation)

* Added query type PARENT(v) → id of v's parent (tied id readout), forked against the
  **prefix cache only** (edges + START; no latent positions), weight 1 for 3000 steps
  then linearly annealed to 0 by step 6000.
* It supervises an input-lookup primitive, not reachability, not the frontier, and
  never touches latent thoughts. Comparable in spirit to Coconut's stage-0 language CoT.
* **Claim restriction:** results concern models trained with this scaffold. It plausibly
  *favours* the "input reconstruction" alternative (query-time backward lookup), which
  the intervention design must therefore be able to detect (depth-resolved effects).
* Also adopted: shared identity table in role slots ("slots" embedding), PyTorch-default
  init, CUDA-graph captured training step (~2× faster; numerics unchanged).

### Round 2 (PARENT scaffold only): depth-1 learned, depth ≥ 2 not

* PARENT lookup learned in < 500 steps. Depth-1 reachability then learned
  (100% by ~3k steps) — necessarily *at query time* (probe AUC on thoughts ≈ 0.50,
  donor patches have zero effect). Depth ≥ 2 stayed at the prior through 6k steps.
* Interpretation: depth 2 in two layers requires START to gather the children of s
  and the query to check membership of its parent — again two unlearned matchings.

### Decision: add CHILDREN(v) lookup + accuracy-gated depth curriculum

* CHILDREN(v): multi-label over ids of v's children, prefix-only fork, v drawn from
  branch nodes and s' — **never s** (so reachability/frontier of s is never a target).
* Gated curriculum: start at within-branch depth ≤ 1 (star branches); advance when
  held-out delayed balanced accuracy at the current stage ≥ 0.9 (checked every 500
  steps); full depth after stage 4. Aux weight 1 for 10k steps, annealed to 0 by 15k.
* Bug found and fixed: at stage 1 all branches are stars, so only 8 families exist and
  the family-split rejection sampler could loop forever (and would restrict training to
  the widths whose family hashed into train). Stage-1 curriculum graphs now skip split
  filtering; all evaluation is at full depth where such graphs have probability ~1e-9.
* Baseline controls (pause, K=0) must replay the main run's realised curriculum
  schedule rather than gate on their own accuracy (else they see different data).

### Pre-registered predictions for confirmation (written before any confirm-split run)

Smoke test on an intermediate seed-0 checkpoint (discovery graphs only) showed full-vector
donor patches with collateral ≈ targeted, and targeted falling from 0.16 (W=2) to 0.01
(W=8). Hypothesis **H-norm**: branches share the thought through softmax-normalised
attention averaging, so each branch's share scales ~1/W. Predictions:
1. Targeted effect of a single-branch edit decreases with W.
2. Removing branch i *raises* the other branches' P(reach) (signed collateral < 0),
   because their normalised share grows; inserting lowers it (signed collateral > 0).
3. Probe decodability degrades more slowly with W than causal selectivity
   (readable-but-not-separately-usable at high W).
Alternative **H-damage**: collateral is unsigned general damage → signed collateral ≈ 0
with |collateral| > 0, and the identity / same-status controls would also show effects.

### Disclosure: pipeline smoke tests touched confirm-split graphs

Two pipeline smoke tests (intermediate seed-0 checkpoints at ~8k and ~16k steps, n=60
graphs/width, W ∈ {2,8}) drew *confirm-split* graphs before `split` was made a protocol
parameter. Seen: full-vector patch at k=1 had targeted ≈ collateral; probe-subspace
patches ≈ 0; identity control exactly 0. No protocol choice below was changed because of
these numbers (all conditions were already specified in PROTOCOL.md). Final confirmation
uses final checkpoints and a fresh sampling seed; smoke tests since then use discovery graphs.

### Seed 0 (runs/r3_pc_gated_gpt2) — competence, discovery, frozen protocol, confirmation

**Competence (confirm split, n=4096):** delayed bal-acc 0.900 at K'=6; 0.74 at K'=0 (inference
truncation). Distance 1..6 accuracy at K'=6: 1.00/1.00/0.93/0.70/0.59/0.57. Oddity: distance-1
queries are ~0% at every truncated budget and 100% only at K'=6 — the model reads depth-1
membership from the last latent position. Truncation is therefore not a clean ablation.

**Discovery (discovery split only).** Reach-probe matched AUC 0.67–0.74 pooled, but within-graph
per-branch AUC only 0.55–0.63 (controls 0.49–0.52). Full-vector donor patches small (k=1:
targeted 0.06, collateral 0.03). Probe-filter subspace: exactly 0 effect at every step.
Exploration: the probe's 6 readout directions hold 0.3% of the donor-difference energy —
*less* than a random 6-dim subspace (2.3%). Donor differences are low-rank and shared across
graphs (top-20 PCs 77–98%). A PCA-20 subspace reproduces the full patch; the branch-specific
**Haufe pattern** subspace (Cov(h)·W_probe, 6 ids) reproduces ~88% of it. Filters ≠ patterns.
These discoveries added the `pattern` and `pca` conditions *before* freezing
`docs/protocol_frozen.json` (k*=1 by the pre-set rule).

**Confirmation (frozen protocol, confirm split, 300 graphs/width, fresh seed).**
* Targeted (full, remove, k*=1): W1 0.261 → W2 0.085 → W4 0.059 → W8 0.026 (monotone ↓).
  Pattern: 0.212 → 0.052 → 0.052 → 0.025. Probe filter: 0.000 at every W. Random rank-6/20:
  ≤ 0.009. Norm-matched isotropic: targeted ≤ 0.011 but collateral 0.013–0.046 (generic damage).
* Depth-resolved (k*=1): distance 1–2 unaffected (±0.001); distance 3–6 at W=2:
  0.06/0.19/0.32/0.26; W=8: 0.03/0.06/0.07/0.07. Step k=0 at W=2: targeted 0.405, distance-3
  0.70, collateral 0.077 — but k=0 is nearly inert at W≥4.
* **H-norm prediction 2 confirmed:** signed collateral < 0 under removal (W2 −0.060, W4 −0.016,
  W8 −0.006) and > 0 under insertion (+0.096, +0.018, +0.009): removing one branch makes the
  others *more* reachable. Identity control 0 exactly; same-status control 0.002–0.013.
* Downstream trace: after a full k=1 edit the edited branch's probe margin collapses at k=2–3
  (W2: 0.38 → 0.03) but partly recovers by k=4–6 (0.22–0.29): re-derivation from the intact cache.
* Upfront format: full edit of the target's branch moves the answer 0.14–0.18 at every W with
  collateral ≈ 0.001; the delayed-format pattern subspace is inert there (different directions
  when the target is known).
* Provisional "selective" bar (≥ 0.10 targeted, ≤ 0.03 collateral, pooled over the branch) is
  **not met** for any delayed-format condition; it is exceeded on deep nodes only.

**Exploratory mechanism check (discovery split):** layer-0 attention from LAT_1..3 onto each
reachable branch's edges, excess over a decoy branch: W1 0.25, W2 0.12, W4 0.04, W8 0.016 —
a roughly fixed attention budget (0.25 → 0.13 total) divided among W branches.

### Seeds 1 and 2 (runs/main_s1, runs/main_s2) — same frozen pipeline

* Competence (confirm split): delayed 0.918 / 0.926, upfront 0.873 / 0.896.
* Frozen site rule gave k* = 0 for both (seed 0: k* = 1). Cross-seed figures therefore use
  each run's own k*; the step profile (k = 0..4) is reported for all seeds.
* Targeted (full, remove) W2 → W8: s1 0.300 → 0.059; s2 0.324 → 0.114 (s0 0.085 → 0.026).
* Probe filters ≤ 0.001 everywhere; branch-pattern energy share 0.53 / 0.49 (s0 0.35) vs
  filters 0.003 / 0.002 vs random-6 0.022.
* Selectivity bar (≥ 0.10 targeted, ≤ 0.03 collateral) met only by **s2 at W = 2**
  (full 0.324/0.022; pattern 0.265/0.019; insert 0.288/0.017). s0, s1: not met at any W.
* Signed collateral: sign pattern (remove < 0, insert > 0) in s0 and s2 at every W; s1 at
  W 2–3 and 7–8, ≈ 0 at W 4–6.
* Depth: distance-1 never affected, distance-2 ≤ 0.05; effects on distance ≥ 3 (W2 up to 0.60–0.66).
* Downstream recovery: s0 partial, s2 partial, s1 none (edited branch stays suppressed).
* Upfront format: target-branch edits move the answer 0.64–0.74 (s1), 0.38–0.51 (s2) with
  collateral ≤ 0.03 except s2 W2 (0.053); delayed-format pattern subspace ≤ 0.027 there.
* Seed variation in effect size (~4×) is large relative to within-checkpoint CIs; the
  qualitative pattern is what replicates.

### Controls (single seed each, same data/scaffold/realised curriculum schedule as seed 0)

* **No latent (K=0):** delayed 0.701, upfront 0.668; distance 1/2 = 1.00/0.94, distance 3–6 =
  0.10–0.12 (almost always "unreachable"). Extra positions are required beyond two hops.
* **Pause tokens (K=6, no feedback):** delayed 0.912, upfront 0.815. Distance 1–6 =
  1.00/1.00/0.99/0.77/0.29/0.27 vs recurrent seeds 1.00/1.00/0.92–0.97/0.70–0.80/0.57–0.70/0.38–0.60.
  → Overall accuracy does **not** separate recurrence from pause positions; the recurrent
  advantage is confined to 5–6-hop queries and the upfront format. Consistent with the
  2606.12689 caution that pause-as-thought reproduces much of what is attributed to recurrence.
  Thought-level interventions are not applicable to the pause model (no thought is fed back).
* Operational note: two CUDA-graph jobs on one GPU cost ~2× combined throughput under WDDM;
  the pause run was suspended (NtSuspendProcess) while seed 1 finished, then resumed.

### Final assessment against the proposal's decision criteria

| Criterion | Result |
|---|---|
| Competence ≥ 85% (delayed, confirm split) | Met, all seeds (0.90–0.93) |
| Latent computation contributes | Met vs K=0; vs pause only at 5–6 hops / upfront |
| Minimum positive result: ≥ 2 alternatives in one shared state with distinct predicted effects, replicated | Met qualitatively in 3/3 seeds (targeted > collateral at W ≥ 3; depth-specific effects; controls ≈ 0) |
| Provisional selectivity bar (≥ 0.10 targeted, ≤ 0.03 collateral) | Met in 1/3 seeds, W = 2 only |
| Bounded test of width | Done: targeted and selectivity decline monotonically W 2→8; decodability declines more slowly |
| Valuable alternative result | Readable ≠ usable: probe filters inert, activation patterns causal; shared attention budget âˆ 1/W |

### Round 3 early observations

* PARENT+CHILDREN, GPT-2 init: star stage passed at 1k steps; at 4k steps distance-3
  accuracy 0.54 on full-depth held-out graphs (stage 2). PARENT-only run: depth 1 only.

---

## 2026-09-24 — Part II: why branch edits weaken with width (handoff plan docs/NEXT_STEPS.md)

### Preflight
* Reproduced all 862 published seed-0 confirmation point estimates exactly (max |diff| = 0.0).
* Split audit: zero family overlap train/discovery/confirm; generator sha256 prefix recorded.
* Donor audit: W=1 removal/insertion involves an untrained W=0 state → W=1 now reported only as a
  distribution-shift test. Swap insertion at W=8 creates W=9 donors (flagged).
* New instrument `latent/flex.py` (thought edits at any steps, KV overrides by position and reader,
  attention key bias); validated: equals model.run, full-route override == donor run bit-exact.

### Protocol 2
* Hypotheses/predictions written in docs/PROTOCOL2.md before discovery; after discovery, frozen
  numerical choices + discovery-derived predictions P1–P8 in docs/protocol2_frozen.json, then
  addendum A (P9 composition, P10 gain model) frozen before their fresh runs.
* Fresh confirmation graphs: confirm buckets, seed 271828, excluding every family drawn by any
  earlier confirm-split sampler (replayed in followup_run.used_confirm_families).
* Bug fixes during discovery (no confirmation data involved): pause models lacked probe assets;
  driver ignored --k for pause models.

### Fresh-confirmation outcomes (results/followup_confirm.json)
* P1 route shift — pass all (latent share ↓, query-graph share ↑ with W).
* P2 no dose rescue — pass all. P3 pattern ≫ filter at matched norm (6–57×) — pass all.
* P4 swap — partial: signed collateral always cancels; unsigned only reduced at W=2 in some seeds.
* P5 semantics — pass at every frozen site; fails only at seed 0's extra common site k=0 (edits too weak).
* P6 competition — pass s0, s2; s1: uniform all-edge boost also lifts reachable branches (+0.02).
* P7 no gain restoration — pass all. P8 geometry adds nothing — FALSIFIED for targeted effects in
  s0 (+0.07 R²) and s2 (+0.11); holds for collateral.
* P9 composition — pass all. P10 gain 1/√W — pass s1, s2; s0 prefers 1/W (β = 0.87).
* P1b pause query-time share ≥ 0.85 — pass pause_s0; FAIL pause_s1 (START cache carries ~50%).

### Post-hoc (not pre-registered) depth breakdown of the cache factorial, same fresh graphs
* 1-hop nodes: always query-time; ≥4-hop nodes: always latent pathway (TL ≈ 0.72–1.0) at every W.
* The latent/query boundary moves inward with W (2–3-hop nodes switch to query-time resolution).
* The model's own clean counterfactual on deep nodes collapses with W (s0, 4 hops: 0.79 → 0.19).
* Per deep node, a thought-only edit reproduces a similar share of the clean effect at every W
  (5–6 hops: s0 0.45→0.77, s1 ≈0.95, s2 ≈0.65). → The width effect on branch-averaged edits is
  compositional (collapsing deep-node sensitivity + inward boundary), not declining edit efficacy.

### Control replication (seed-matched curriculum schedules)
* Pause s1/s2 and no-latent s1/s2 trained. Competence is strongly seed-dependent: pause_s1 never
  passes 2 hops; no-latent s1 reaches 0.91 overall (3 hops 0.97) with no latent steps at all.
* Per-distance *recall* can be inflated by a "reachable" bias (pause_s1: 0.98 recall at 6 hops,
  0.50 depth-matched). New depth-matched balanced accuracy (latent/competence_matched.py) used instead.
* Part I's single-seed control conclusions (pause ≈ recurrent to 4 hops; no-latent fails > 2 hops)
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

## 2026-09-24 (night) — hardware incident and restarts

* 03:45 the machine rebooted from a bugcheck 0x1A (MEMORY_MANAGEMENT); after reboot Windows logged
  "removed bad memory regions from this PC". All three trainings (main_s3 at step 19 200, V1 at 4 000,
  V2 at 1 600) died with it; no optimizer state was saved, so each restarts from step 0 with the same
  recipe and seed. Crashed folders are kept as runs/*_crashed.
* 03:41 (before the crash) the V3 battery failed inside `bootstrap_ci`: `rng.integers(0, 300, ...)`
  returned 4 294 967 307 = 2^32 + 11, i.e. a single flipped bit in host RAM. The draw is seeded, so
  the rerun reproduces the correct resample. This is direct evidence of silent host-memory corruption
  on this night; see the integrity re-run below.
* The earlier V1 "stall" coincides with 940 `nvlddmkm` "Graphics FECS Exception" events on the RTX 3070
  (PCI bus 6 = torch cuda:1) between 02:38 and 03:31, so it was a GPU fault, not (only) memory
  oversubscription. After the reboot the 3070 passed a 90 s bit-exact matmul check (fp32 + bf16) and a
  6.7 GB VRAM pattern test with no new driver events; it is back in use with driver-event monitoring.
* V2 in fp32 needs > 7 GB and cannot run on the 2070; it trains in bf16 on the 4060 (V1 on the 3070,
  main_s3 on the 4060, V3 battery on the 2070). Precision already differed across the reference seeds
  (s0 3070 bf16, s1 4060 bf16, s2 2070 fp32).
* Research log had cp1252 mojibake from an earlier patch script (37 lines, arrows/≥/≈); repaired
  run-by-run (scratch/check_mojibake.py), no content change.
* Co-locating V2 with main_s3 on the 4060 oversubscribed its 8 GB (7.9 GB used) and stalled *both*
  runs (main_s3 lost ~6 min, V2 never reached step 1). V2 now waits for main_s3 to exit
  (scratch/after.py). Rule from here on: one captured-graph training per GPU.
* **Integrity re-run.** main_s1's fresh-confirmation cache factorial and dose experiments were
  recomputed from scratch on a different GPU (2070, fp32) after the crash and compared with the
  stored confirmatory summary: all 7 034 deterministic numbers agree within 1e-4 (max |diff| of
  those < 1e-4). The only differences (601 numbers) are in the random rank-6 null-control subspace,
  whose basis is redrawn from a device RNG; its effects are ≈ 0.001 either way. No evidence that
  the stored Part II numbers were corrupted.

### Second host crash (07:18) and crash-safe training
* 07:18 the host rebooted again, bugcheck 0x133 (DPC_WATCHDOG_VIOLATION), with no GPU-driver
  exceptions beforehand; nothing ran again until 10:35. main_s3 and V1 had both finished 30 000
  steps (checkpoints intact). V2 was at step 23 000 (still only at curriculum stage 3, reached at
  step 21 000; delayed 0.74) and its model.pt was being written at the moment of the crash
  (truncated file) → runs/v2_deep_s0_crashed2; V2 restarted from step 0 with the same command.
* latent/train.py now writes model.pt and checkpoints atomically (temp file + rename) and keeps a
  full training state (model, AdamW moments, data/torch RNG states, curriculum state) in
  <run>/state.pt at every evaluation; `--resume` continues from it, trimming log lines beyond
  the saved step into log_discarded.jsonl. Verified on a 400 → 600-step test (loss continuous).
  The evaluation pipeline re-runs any output file that does not parse.

### Part III scoring, first batch (results/protocol3_confirm.json, results/followup_confirm.json)
* **Seed 3 (addendum C)** — competent (delayed 0.925; 6-hop depth-matched 0.856, far above
  s0–s2's 0.59–0.66). Expected-to-hold predictions: P1, P2, P3, P5, P9, P14 hold; **P7 fails**:
  restoring total gain *lowers* the targeted effect (W4 0.434 → 0.399, W8 0.186 → 0.169; tolerance
  |Δ| < 0.03) — not a rescue, but outside the frozen tolerance, which is strict for a seed with
  edits this large. Mixed ones: P4, P6, P8 hold; P10 fails (β = 0.29, far below 0.62–0.87).
  Competition inside the thought at W = 4: yes (now 3 of 4 seeds). Route shift is weak (latent
  share 0.83 → 0.72 from W = 2 to 8) and edits are large (targeted 0.56 / 0.43 / 0.19).
* **P11 (V1, d = 512) fails.** Attention clause holds (W8/W2 = 0.24), but targeted W8/W2 = 0.58 >
  0.5: a weaker width decline than all four d = 256 seeds (0.18–0.37) — the direction the capacity
  account predicted. Collateral is not lower than s0's at W = 8 (0.039 vs 0.023), so that half of
  the capacity prediction is not met either. V1 also keeps most of the effect in the latent pathway
  (0.83 / 0.79 / 0.74), like s2 and s3. One run per variant: architecture vs seed cannot be separated.
* **P15 fails** on V1: depth-6 gain K' 6→8 = +0.0502 (just over the 0.05 bar) but no extrapolation
  (distance 7 at K' = 12 below 0.60). s3 agrees (gain +0.001, no extrapolation). As flagged after
  V3, the in-range gain is not a reliable signature of an open-ended loop.
* **P16** holds so far: s3 E = 5, no extrapolation; V1 E = 3 (unconstrained), no extrapolation.
* Exploratory, new models (results/delayed_block_new.json, results/rewind_grid_s3.json):
  s3's sweep is one depth per step and finishes depth 4 at step 4; blocking a depth's edges at its
  step removes 72–81% of the branch's downstream P(reachable) (−0.81, −0.74, −0.72, −0.75), nothing
  from step 5 on — its deep nodes rest almost entirely on the clocked sweep. V1's schedule is a
  two-pass sweep: depth 1 read at steps 1, 2 and 4, depth 2 at 3 and 5, depth 3 at 6 (effects
  −0.17 to −0.23). Rewinding s3 gives a weak unlock (far 0.51 → 0.56) with s0's exact signature:
  every top condition feeds the step-3 thought (j − 1 − r = 3). Unlock by rewind: s0 strong, s3
  weak, s1 none; s2 never stops; V3 none.
* **budget_k10_s0 (addendum G) fails the competence gate**: delayed 0.836 at K' = 10 (upfront 0.715);
  depth-matched 0.99 / 0.95 / 0.81 / 0.67 / 0.55 / 0.53 for 1–6 hops. It reached curriculum stage 3
  at step 7 000 and never advanced (training depth capped at 3 hops within a branch). The longer
  unrolled recurrence makes this recipe train worse within 30k steps. Its addendum-G expectation is
  not scored; its sweep is described as exploratory only. Its layer-1 trace reads depth 1 at
  step 1, depth 2 at step 2, then falls back to depth 1 for every later step (never past depth 2).
* **budget_k4_s0 (addendum G): competent and extrapolates.** Delayed 0.926 at K' = 4, upfront 0.999;
  depth-matched 1.00 / 0.98 / 0.94 / 0.87 / 0.62 / 0.59. Long-chain test: chance at its trained 4
  steps (distance 7: 0.50), then 0.67 at K' = 6 and 0.68 [0.66, 0.70] at K' = 12; distance 8 0.63,
  distance 9 0.58. Query-time share at W = 4 is 0.20 (inside the reference range) — it took the
  open-ended-loop route, not the query-time route. Frontier E = 2. As the addendum-G hypothesis
  allows: a budget too short for the sweep to finish produced an extendable loop. (The contrast
  case, K = 10, failed training, so only one side of the manipulation is tested.) Its in-range
  gain K' 6→8 is only +0.011 — one more case against P15's signature.
* **P12 (V2, 4 layers) fails in the opposite direction.** V2 is the most competent model of the
  project (delayed 0.965, upfront 1.00; depth-matched 0.91 at 4, 5 and 6 hops) and keeps branches in
  the latent pathway at every width (TL 0.83 / 0.83 / 0.78; query-time share at W = 4 = 0.17, below
  the reference minimum 0.28; at 3 hops ≈ 0.00). Extra layers did not move nodes to query-time
  reading. Targeted W8/W2 = 0.55 (like V1's 0.58), β = 0.54, attention W8/W2 = 0.24. E = 4, no
  extrapolation (consistent with P16).
* **Seed 4 (addendum D) fails the competence gate** (delayed 0.819): curriculum stage 2 from step
  2 000 to 17 500, never full depth (depth-matched 0.55 / 0.54 / 0.53 at 4–6 hops). Not scored.
* **P16 final: passes** on the three competent new models (s3 E = 5, V2 E = 4: neither extrapolates;
  V1 E = 3). **P15 final: fails** (V1). Also consistent with the addendum-F hypothesis, outside P16's
  scope: K = 4 (E = 2) extrapolates.
* **Development (seed 4 checkpoints; results/development_dev_s4.json).** At step 1 000 (star stage)
  branch effects are almost entirely query-time (Gq 0.84 / 0.95 / 0.98 at W = 2 / 4 / 8); the
  latent share grows as 2-hop nodes are learned (TL 0.60 / 0.55 / 0.45 by step 3 000) — the
  width-dependent split is present from the first checkpoint. The layer-1 sweep never advances:
  from step 4 000 on it reads depth 1 at every step (no clock develops), deep nodes stay at chance,
  and the late gain at 3 hops (0.58 → 0.86 after step 17 500) comes through query-time reading
  (Gq at W = 4 0.40 → 0.70–0.82; W = 8 latent share 0.50 → 0.23). A run in which the clocked sweep
  does not form falls back on query-time reading, which reaches three hops at most.
* **Stage-1 stall rate.** Tonight about 7 of 15 gated-curriculum attempts stalled at stage 1
  (main_s3 on bf16; V1 attempts 2–3; V2's first post-crash attempt; main_s5, V1 seed 1 and the
  second K = 4 run on their first attempts), including one fp32 run (budget_k4_s1 on the 2070) — so
  the protocol note "seen only under bf16" no longer holds. Every gated run before tonight had
  passed (6 of 6). Checked: the evaluation-chunk change cannot affect training, because batches
  draw only from the training numpy generator (latent/data.py) and evaluation uses its own
  generator. The recipe's stage-1 outcome is simply bimodal, with a failure rate near one half.
  Later: main_s5 needed three attempts, budget_k4_s1 two, v1_wide_s1 four (the last allowed).
* **Seed 5 (addendum I): passes the gate, fails the replication, and is latent-inert.** Delayed
  0.891, upfront 0.888, but depth-matched 0.54 at six hops. Every discovery site has a targeted
  effect < 1e-4, so the Part I rule picked k* = 5 from noise; at that site every thought edit has
  exactly zero effect (dose, continuation, amplification all 0.000). Route shares: latent pathway
  0.16 / 0.12 / 0.09, query-time 0.83 / 0.89 / 0.90 at W = 2 / 4 / 8; no layer-1 sweep at any step.
  Expected-to-hold predictions: P1 and P7 hold (P7 trivially, 0 vs 0); P2, P3, P5, P9, P14 fail
  because there is nothing in the thought to edit. Verdict as scored: the addendum-I replication
  fails on seed 5. Lesson: the overall-accuracy gate (≥ 0.85) admits models that do not use the
  latent loop at all (the no-latent controls reached 0.91–0.93); a depth-matched gate would not.
* **budget_k4_s1 (addendum I) fails the competence gate** (delayed 0.691, upfront 0.699; stuck at
  curriculum stage 2 from step 1 500). Not scored; the K = 4 result rests on one run.
* **v1_wide_s1 (addendum H) fails the competence gate** after four attempts (three stage-1 stalls;
  the fourth reached delayed 0.718, 6-hop 0.51). Addendum H is unresolved: whether V1's weaker
  width decline comes from the wider thought or from V1's run remains open.
* **The sweep's content is in the thought; its stop is keyed to the step embedding**
  (latent/clock_swap.py, clock_freeze.py; exploratory on s0, s1, s2, s3, V2). Changing only the step
  embedding fed to each latent step: the depth read at step k is unchanged under frozen or delayed
  clocks (the thought carries the frontier). Advancing the clock by 1–2 steps truncates the sweep
  (s3: depth-6 accuracy 0.84 → 0.52). Holding the clock at step 2 removes the stop: far accuracy on
  long chains, K' = 20 — s3 0.87 / 0.87 / 0.82 / 0.71 / 0.62 at distances 7–11; V2 0.88 / 0.88 /
  0.88 / 0.86 / 0.80 / 0.70 at 7–12 (normal: chance beyond 6); s1 ≈ 0.62 and s0 ≈ 0.60–0.66 flat to
  distance 12; s2 (never stops) unaffected. The propagation step generalises far past the trained
  depth; the halt is learned from the step index, because training always ended by step 6.
  This supersedes the "halting state in the thought" reading of the rewind results (which restored
  early thought content but kept the late step embeddings, and unlocked less).
* **The halt stops the sweep itself** (latent/clock_trace.py; long-chain graphs; layer-1 attention on
  each chain edge per step). Normal clock: s3's frontier attention collapses at steps 5–6 (0.074 →
  0.024 → 0.001) and V2's at step 6 (0.113 → 0.007); neither ever reads the re-hung chain's edges.
  Clock held at step 2: the sweep walks the chain one hop per step — s3 reads distances 7…12 at
  steps 6…11 (attention 0.099, 0.107, 0.087, 0.054, 0.033, 0.021), V2 at steps 7…12 (0.188, 0.208,
  0.194, 0.166, 0.121, 0.072) — and stops only at the chain's end.
* **budget_k4_s2 (addendum J, J1): passes the stricter gate and extrapolates.** Delayed 0.916,
  upfront 0.9995, depth-matched 5-hop 0.601. Chance at its trained 4 steps; then ≈ one hop per step:
  distance 7 0.60 (K' = 6), 0.75 (K' = 8), 0.71 [0.69, 0.73] (K' = 12); at K' = 12 distance 12 is
  still 0.63. Query-time share at W = 4 0.21. Two of three K = 4 seeds are competent and both
  extrapolate (the third failed training). Its frontier E (steps ≤ 6) is 4, but steps 5–6 lie
  beyond its 4-step training budget, so this does not contradict the addendum-F idea, which concerns
  sweeps that finish *inside* the budget (P16 was scoped to K = 6 models).
* **budget_k8_s0 (addendum J, J2): passes; P17 on K = 8: fails.** K = 8 passes the stricter gate
  (delayed 0.928, upfront 1.00, 5-hop 0.719); E = 4; no extrapolation (distance 7 at K' = 16 =
  0.559) — the long-budget side of the budget hypothesis holds. Frozen clock at 2: distance 7 =
  0.578 [0.555, 0.600] < 0.60, so P17 fails on both prospective models (V1, K = 8). Post hoc, no
  freeze point helps K = 8 (best, freeze at 4: 0.56), and its clock trace shows why: its sweep does
  not halt sharply but fades hop by hop (frontier attention 0.050 → 0.037 → 0.015 → 0.011 at steps
  5–8, still reaching distance 7 but too weakly to matter), with or without the frozen clock.
  → Two kinds of limit: a sharp step-keyed halt (s3, V2; freezing the clock removes it) and per-hop
  attenuation (K = 8 and, apparently, V1; the clock cannot fix it). P17 predicted the first kind
  for every non-extrapolating model; the prospective tests found the second.
* **P18 (addendum K): passes.** runs/nostep_s0 (main recipe without a step embedding) passes the
  stricter gate (delayed 0.871, upfront 0.897, depth-matched 5-hop 0.641) and extrapolates:
  distance 7 at K' = 12 = 0.631 [0.611, 0.651]; at K' = 16 distances 7–12 are 0.67 / 0.67 / 0.65 /
  0.63 / 0.62 / 0.60. Reach grows about one hop per extra step (distance 7 by K' = 8, 9 by 10, 11 by
  12, 12 by 16). With no step index it learned no halt. Training-level support for the clock account
  (one run).
* **P17 on V1 (prospective): fails.** Frozen at step 2, K' = 12: distance 7 = 0.571 [0.553, 0.588]
  (normal 0.53), below the 0.60 bar, and 3–5-hop accuracy drops (0.94 → 0.85 at 3 hops). The
  step-keyed stop is strong in s3 and V2, moderate in s0 and s1, weak in V1 (whose sweep is the
  irregular two-pass one). budget_k8_s0 is the remaining P17 test.
* **RTX 3070 lost (≈15:20).** During v1_wide_s1's evaluation battery, nvidia-smi reported the card
  at PCI bus 6 as "GPU is lost. Reboot the system to recover this GPU"; the hung job was stopped.
  No reported result depends on that battery (the model had already failed the gate). The same
  card threw the FECS exceptions on the first night; it may also be behind the 0x133 crash.
  It needs a reboot (left to the user) and should be tested before further use.

### Upfront-format route decomposition — seed 2 completed (discovery, exploratory)
* s2 replicates s0/s1: Gq ≈ 0.00 at W = 2, 4, 8 and every target distance; TL = 1.00. Seed 2's
  thought alone carries 60–77% overall (1–2-hop targets 100%, ≥3-hop ≈ 0.3–0.6, rest in the
  START/TARGET/early-latent cache). 3/3 seeds: the upfront ANSWER token never reads the graph.

### Exploratory: inference-budget sweep (results/budget_sweep.json; discovery graphs)
* Depth-matched accuracy vs K' separates seeds. s2 is a staircase — depth 3 resolves at K'≈2,
  depth 4 at K'≈4, depth 5 at K'≈6 — and keeps improving beyond the trained budget (depth 6:
  0.59 at K' = 6 → 0.71 at 7 → 0.77 at 8, identical with the untrained step rows or the reused
  step-6 row). s1 has resolved almost everything by K' = 1 (saturates). s0 is odd: 1-hop nodes are
  at chance for every K' < 6 and perfect at K' = 6 (its answer read-out is tied to the final step).
* Pause models plateau by K' ≈ 2–4; no-latent models lose 0.01–0.03 with untrained latent slots.

### Exploratory: length generalisation (results/length_gen.json; all-chain graphs)
* s2 follows chains past any trained depth: distance 7 bal-acc 0.50 (K' = 6) → 0.60 (8) → 0.70 (10)
  → 0.71 (12); distance 8: 0.62, distance 9: 0.56 at K' = 12; ≈ one hop per two latent steps.
* s0, s1, V3, pause_s0, k0_s2: distance ≥ 7 stays at 0.48–0.54 for every K' ≤ 16.
* Reading: one of three seeds learned an iterative frontier-expansion loop that can be run for
  longer at test time; the others learned a fixed-depth solution. s2 is also the seed with the
  largest latent-pathway share at W = 8 (TL 0.63 vs 0.11 / 0.19). Scored prospectively on the
  new models as P15 (PROTOCOL3 addendum E).

### Exploratory: budget sweep resolved by width (results/budget_sweep_wd.json)
* s2's search *speed* does not depend on W: depth 5 resolves at K' ≈ 6 and depth 6 at K' ≈ 8 for
  W = 2, 4 and 8 alike. Its *ceiling* does: with K' = 8–10, depths 3/4/5/6 reach
  0.99/0.98/0.97/0.95 at W = 2, 0.96/0.86/0.80/0.78 at W = 4 and 0.82/0.72/0.63/0.59 at W = 8.
  Extra steps do not raise the W = 8 ceiling → the high-width deficit is per-hop fidelity of a
  shared state (consistent with the W^−β per-branch amplitude), not running out of steps.
* V3 has the same staircase at W = 2 (depth 6: 0.47 at K' = 6 → 0.80 at K' = 8) but none at W = 8.
  Its overall depth-6 gain K' 6→8 is +0.044 — just under P15's 0.05 bar, and it does not
  extrapolate, so it is (barely) consistent with the P15 rule. V3's sweep was run after addendum E
  was written and is not part of P15's scoring; noted because it shows the threshold is not robust.

### Exploratory: rewinding the thought (latent/rewind.py, rewind_grid.py, rewind_multi.py; long-chain graphs)
* Intervention: in a K' = 12 (or 16/20) run, the thought fed into step j is replaced by the run's own
  thought from step j − 1 − r (or s = source index); the KV cache of all earlier positions is kept.
* **s2** (extrapolating seed): rewinds only cost reach (distance 6: 0.78 → 0.66–0.70 for j = 5,
  r = 2–4) and never recover it fully — the thought carries the search state, and an old thought
  conflicts with the newer cache. Periodic rewinds cost more.
* **s0** (non-extrapolating seed): a single rewind *unlocks* search beyond six hops. Far accuracy
  (distances 7–9) rises from 0.51 to 0.62–0.63 whenever the injected thought is from steps 1–3 and
  the injection comes at step ≥ 6; thoughts from steps ≥ 4 do nothing (0.51–0.52), the same step's
  thought from a *different graph* does nothing (0.50–0.51), a zero thought barely helps (≤ 0.55),
  and near accuracy (3–6 hops) is unchanged. With K' = 20, distances 7–12 all reach ≈ 0.60.
  Width breakdown (K' = 20, rewind at 7): W = 2 bal-acc 0.49 → 0.66 (TPR 0.20 → 0.60, TNR ≈ 0.75),
  W = 4 ≈ 0.49 → 0.55, W = 6 ≈ 0.51 → 0.53; graphs whose whole far chain is called reachable
  3% → 26% (propagation is close to all-or-nothing per graph).
* Reading: s0's propagation operator generalises beyond its trained depth, but its thought enters a
  terminal ("done") phase around step 4 that stops the search — consistent with the budget-sweep
  oddity that its 1-hop answers only appear at K' = 6. Restoring an early-phase, graph-specific
  thought while keeping the propagated cache resumes the search. s2 never enters such a phase.
  One more instance of readable ≠ usable: the thought mixes search content with program phase.
* Linear-phase test (latent/phase_shift.py, results/phase_shift_s0.json): shifting the fed thought by
  α(μ_s − μ_{j−1}) (population means over separate calibration graphs; s = 1–3, j = 6–8,
  α = 0.5–1.5) gives far accuracy ≤ 0.523 vs 0.497 baseline (norm-matched random: 0.50), against
  0.62 for the own early thought. The stopping state is therefore *not* a graph-independent
  linear "phase" direction; the unlock needs the same graph's early thought. Wording in the
  report changed from "phase signal" to "a state the thought enters".
* Same grid on s1, s2 and V3 (results/rewind_grid_{s1,s2,v3}.json): no unlock anywhere. s1 best
  far 0.517 vs 0.494 baseline; s2's best conditions only match its baseline (0.61, K' = 12);
  V3 ≤ 0.53 vs 0.52. The phase lock + unlock is specific to seed 0 (1 of 4 recurrent models tested).

### Exploratory: upfront-format length test and attention traces
* Upfront length test (latent/length_gen_upfront.py): V3 walks ≈ one hop per step and extrapolates
  (distance 6: 0.96 at K' = 8; distance 7: 0.80 at K' = 12, 0.85 at K' = 16; distance 8: 0.69 at 16),
  although it does not extrapolate in the delayed format. s2 extrapolates in both formats (upfront
  distances 7–11 ≈ 0.59–0.62 at K' = 12–16). s0 and s1 stop at 4 hops in the upfront format.
  V3's upfront accuracy is 1.00 at every width 1–8 (delayed: 1.00 → 0.81).
* Upfront attention trace (latent/upfront_trace.py): V3's layer-1 attention climbs the target's
  ancestry one edge per latent step — edge into the target's parent at step 1, …, the hub edge at
  step 6 — with off-path edges at the same depth at ≈ 0.003; identical walk for decoy targets, which
  end at a hub edge from s' and are answered unreachable (0.00 vs 0.99). **Backward pointer-chasing**:
  a single-path search, which explains width-independence, Gq ≈ 0 and the selectivity of upfront
  edits. s2's upfront trace is instead a *forward*, frontier-wide sweep (off-path edges at the same
  depth get nearly the path's attention), one depth per step from step 2. s0/s1: no step structure.
* Delayed attention trace (--delayed): every model runs a forward frontier in layer 1 (reachable
  edges of one depth, decoys at baseline), differing in speed — s1 one depth per step (depth 5 at
  step 5, then flat), s0 the same one step later, s2 one depth per two steps (depth 3 at step 6),
  V3 slower. Hypothesis: a search that finishes the trained depth inside the budget is followed by a
  learned stop (s0 rewind); a slower one stays open-ended (s2). V3 (slow, no delayed extrapolation)
  shows slowness is not sufficient. Pre-registered as P16 (addendum F) for the new models.
* **Causal check of V3's upfront walk** (latent/upfront_block.py; W = 4, 400 reachable distance-6
  targets, baseline P(reachable) 0.997): blocking layer-1 attention onto one edge at one latent step
  (logit bias −1e4). Path edge on the trace's diagonal: step 1 × edge into the target's parent
  −0.16, step 3 × depth-4 edge −0.48, step 4 × depth-3 −0.22, step 5 × depth-2 −0.16, step 6 × hub
  edge −0.16 / depth-2 −0.22. Every off-diagonal (step, edge) pair and every same-depth edge of
  another reachable branch: |Δ| ≤ 0.004. The walk is causally step-locked; effects are partial
  (redundancy, e.g. adjacent steps or layer 2), but strictly local in step and edge.
* **Causal check of the delayed forward sweep** (latent/delayed_block.py; W = 4, 400 graphs, one
  reachable branch A with depth ≥ 4): blocking layer-1 attention onto A's edges into depth-x nodes
  at step k only, outcome = ΔP(reachable) of A's nodes deeper than x.
  s1: drop only on k = x (x = 1..4: −0.11, −0.11, −0.13, −0.15); every other cell |Δ| ≤ 0.005;
  nothing is read after step 4. s0: k = x + 1 (−0.11, −0.13, −0.13, −0.13; k = 1 × x = 1 −0.02);
  nothing from step 6. s2: two consecutive steps per depth (x = 1: k 1–2 −0.34/−0.29; x = 2: k 3–4
  −0.31/−0.23; x = 3: k 5–6 −0.24/−0.17; x = 4: k 7–8 −0.20/−0.07). Hub edges (x = 0) never matter
  (read in the prefix). Other reachable branches rise by ≤ +0.014 (competition, as in Part II).
  → In the delayed format the continuous thought runs a clocked breadth-first sweep in every seed;
  seeds differ in its clock rate, and s2's larger effects match its larger latent-pathway share.
* **Width changes the sweep's amplitude, not its clock** (delayed traces at W = 2, 4, 8): the
  step → depth schedule is identical at every width, while per-edge frontier attention halves per
  doubling of W (s1, depth-1 edges at step 1: 0.072 / 0.034 / 0.017; s2: 0.071 / 0.037 / 0.020), so
  total attention on the frontier stays ≈ 0.14 per step — a fixed budget split among the branches.
  This is the mechanism behind "fidelity, not time" (budget sweep) and Part II's W^−β amplitude law.
* Pause-token controls (results/delayed_trace_pause.json): no advancing sweep. Layer-1 attention is
  content-free by construction (a pause position's layer-1 query sees only the pause and step
  embeddings); in layer 2, pause_s0/s1 attend to the reachable hub edges at every step alike
  (0.045 vs 0.019 decoy) and pause_s2 to nothing. The clocked sweep needs the recurrent thought.

## 2026-09-24 (late) — PROTOCOL4 Phase 1 launched; cloud capacity added

* 22:52: three local job chains started (`scratch/chain.py`): RTX 2070 (cuda:2) nostep_s1 -> s2 -> s3;
  RTX 4060 (cuda:0) budget_k8_s1 -> s2; RTX 3070 (cuda:1) nostep_k8_s0 -> s1 -> s2. PROTOCOL4 addendum A
  was frozen before the launch.
* 23:26 status: nostep_k8_s0 at step 17.6k (stage 4, delayed bal-acc 0.87); budget_k8_s1 at 9.6k
  (curriculum complete at 9.5k, 0.88); nostep_s1 at 9.2k, still at stage 2 since step 2000 (0.74;
  depth 3 rising 0.40 -> 0.54). nostep_s0 spent 8.5k steps at stage 2, so this is not yet unusual.
  The guard only watches stage 1.
* Measured training speed (ms/step, from log.jsonl): K=6 ~133 on the 4060 and ~211-222 fp32 on the
  2070; K=8 ~149 on the 4060 and ~114 on the 3070; 4-layer ~251 on the 4060.
* Code: `train.py --amp {auto,bf16,fp32}` (auto = previous behaviour; fp32 also disables TF32), and
  Linux-portable `stage1_guard.py` / `chain.py`. Hardware rules are in PROTOCOL4 addendum B.
* 23:35: rented one RunPod RTX 4090 (community, $0.34/h, pod mgc3ntjlns5bfc) for a concurrency
  benchmark (1/2/4/6 simultaneous runs) before any scored run is placed there.

* 23:40–00:10: cloud setup. Three pods on hosts with driver 580.178.04 (two RTX 4090 requests that
  landed on the same GPU, UUID GPU-d4ca7f47…, and one RTX 3090) failed `cuInit` with 999 inside the
  container. The container libcuda matched the kernel driver, and the image's own torch saw no GPU.
  All three were deleted unused, as was one RTX 3090 pod whose container never started. A pod the
  capacity listing priced at $0.34/h was billed at $0.72/h (Secure Cloud) and was deleted after
  about a minute.
* Concurrency benchmark (RTX A4000, driver 550, torch 2.14.0+cu126): one K = 6 run 91 ms/step; two
  runs 186 ms/step each (10.7 vs 11.0 steps/s total), so one run per pod. `python -m latent.tests`
  passes on the pod. The Linux stage-1 guard was checked with a dummy job and then fired for real
  on nostep_s3 (attempt 1 stalled at stage 1, archived at step 4000, relaunched).
* Cloud runs (PROTOCOL4 addendum C; one pod each, follow-on runs start when the first run's guard exits):
  A4000 $0.17/h: nostep_s2 (fp32), then v1_wide_s2. A4500 $0.19/h: nostep_s3 (fp32).
  A4500 $0.19/h: budget_k4_s3, then budget_k10_s2. RTX 4000 Ada $0.20/h: budget_k10_s1, then
  budget_k10_s3. A4000 $0.17/h: v3_rawfb_s1, then v2_deep_s1. Each run's GPU, driver and torch are
  in runs/<name>.host.txt. Results come back with scratch/pod_fetch.sh (checksum-verified; state.pt
  is left out).
* The local RTX 2070 chain (chain_p4_2070) was stopped after launching nostep_s1, so nostep_s2/s3 are
  not trained twice. nostep_s1 continues under its guard.

## 2026-09-25 (~05:00 UTC) — randk launched; post-training evaluation queued on every pod

* P22 randk runs (PROTOCOL4 addendum D) started, one pod each, all hosts passed the `cuInit` check:
  randk_s0 (A4000 $0.17/h, CUDA-13.0 host that works), randk_s1 (A4500 $0.19/h), randk_s2 (RTX 4000 Ada
  $0.20/h). Args: base recipe + `--seed N --K_random 4:8 --amp bf16`.
* Status at launch of the queue: budget_k4_s3 and budget_k10_s1 done (30k); nostep_s2 22.0k (0.922),
  nostep_s3 23.8k (stage 3, 0.821, one stage-1 relaunch), budget_k10_s2 21.4k (0.91), budget_k10_s3 3.2k,
  v3_rawfb_s1 29.6k (0.939, one stage-1 relaunch); v1_wide_s2 and v2_deep_s1 queued behind them.
* `scratch/pod_after.sh` now runs on every pod (deployed with the new `scratch/pod_after_deploy.sh`), in
  the order given: nostep_s2, v1_wide_s2:variant | nostep_s3 | budget_k4_s3, budget_k10_s2 |
  budget_k10_s1, budget_k10_s3 | v3_rawfb_s1:variant, v2_deep_s1:variantclock | randk_sN (plain battery).
  `logs/EVAL_ALL_DONE` on a pod marks it ready to fetch (scratch/pod_fetch.sh) and delete.
* ~06:30 scored (latent/protocol4_confirm.py, results fetched checksum-verified; pods deleted after fetch):
  P22 randk: all three pass the gate (delayed 0.879 / 0.876 / 0.897; randk_s1's final curriculum stage is 4,
  not "full"), E = 4 / 2 / 3, and none extrapolates at K' = 16 (d7 0.487 / 0.548 / 0.569, CIs below 0.60)
  → P22 fails. Addendum G: budget_k10_s1 passes (E 4, d7 0.495); s2 (E 1) and s3 (E 2, d7 0.657) fail.
  budget_k4_s3 fails the gate (matched-5 0.589). nostep_s3 fails the gate (delayed 0.849, two attempts).
  Still training: v1_wide_s2 (5.6k) and v2_deep_s1 (17.2k), then their Part II batteries.
* ~07:30 all cloud runs fetched (checksum-verified) and every pod deleted; RunPod list-pods is empty.
  v3_rawfb_s1 and v2_deep_s1 came back with the Part II battery (+ clock_freeze for v2_deep_s1), not yet scored
  against P13 / P12 / P17. nostep_s2 passes the gate (0.928) and does not extrapolate at K' = 12 (d7 0.523) →
  P19 not replicated (addendum-A reading: a competent run failed the prediction); nostep_s1 still pending locally.
  v1_wide_s2 ended at 30k still at curriculum stage 2 (delayed 0.711, one stage-1 relaunch) → gate failure,
  addendum H unscored for it.
* 08:56 local runs resumed (they had died at 00:42 with the previous assistant session, which was their parent
  process; now launched detached via Start-Process, `scratch/local_resume.sh`): 4060 budget_k8_s2 from 10k → eval
  + clock; 3070 nostep_k8_s1 from 23k → nostep_k8_s2 → both evals. nostep_s1 had finished (30k, stage 3) and was
  evaluated on the 2070: gate fails (0.801, matched-5 0.578). P19: 1 of 3 competent → contingency (nostep_s4, s5).
  Compiled scorecard: docs/PROTOCOL4_RESULTS.md.
* Stall pilot (TRAINING_TODO Phase 0) queued on the 4060 after budget_k8_s2: `scratch/stall_pilot.py`, 4 configs × seeds
  0–2, --steps 30000 schedule kept, each run stopped at its stage-1 advance or at step 4000 → results/stall_pilot.json.
* Fork generator written (`latent/graphs.py` `--shapes fork`, tests in `latent/tests.py`). First design (fork drawn per
  branch, p = 0.5) made frontier track W *more* tightly than the default graphs (SD at fixed W 0.75 vs 1.46 at
  distance 2), so the shape is drawn per graph: all chains (frontier W) or all forks with one split depth (frontier
  2W past the split). Default-mode sampling is untouched (same RNG stream).

## 2026-09-25 — Phase 3 "check first": frontier edge count, not W, sets per-edge attention (existing models, no training)

* Models with a clocked delayed sweep: r3_pc_gated_gpt2 (s0), main_s1, main_s2, main_s3. dev_s4 and main_s5 read
  every depth at step 1 (no sweep; ~50% of graphs above the decoy floor) and are uninformative here.
* Default graphs (`latent/frontier_amp.py`, results/frontier_amp.json; 3000 discovery graphs, W 1–8; a = per-edge
  layer-1 excess attention over the same graph's decoy edges at the step that reads depth j):
  log a = c + b_W log W + b_n log n. Depths 1–2: b_n −0.55 to −0.87, b_W −0.19 to −0.51 (s0 depth 2: b_W +1.08).
  Depths 3–4: W dominates (b_W −0.56 to −1.42, b_n −0.39 to +0.33). The (W, n) table at depths 1–2 shows ~1/n at
  fixed W plus a 20–25% drop from W = 2 to 4 at fixed n. (A first pooled fit on raw attention gave b_W −1.21 on
  main_s1; the ~0.008 floor and weak deep cells drove it. Replaced.)
* Controlled test (`scratch/frontier_fork_ratio.py`, results/frontier_amp_fork.json): the same models on fork-mode
  graphs, where at fixed W the depth-j frontier is exactly W (chain graphs) or 2W (fork graphs). Per-edge excess,
  fork / chain: depth 1 0.53–0.68, depth 2 0.49–0.59, depth 3 0.49–0.59, every model and W ∈ {2, 4, 8}
  (budget per edge → 0.5; per branch → 1.0). Depth 4 is not testable (no template puts two edges per branch there).
  The models stay competent on fork graphs (delayed bal-acc 0.869 / 0.889 / 0.910 / 0.924 vs 0.902 / 0.914 /
  0.924 / 0.929 on default graphs).
  → The sweep's attention budget is split among frontier edges, not branches; the W^−β amplitude law is the
  frontier-size law seen through W. The depth-3–4 W effect in the default-graph regression is a confound of how
  n and W co-vary there. At depth 1 the total rises somewhat with n (ratio > 0.5).
* Consequence for TRAINING_TODO Phase 3: fork_s0–s2 are not needed for this claim (the generator is still there).
* ~10:25 local results. budget_k8_s2 ends at stage 3 → gate failure (0.823, matched-5 0.543); P20 rests on
  budget_k8_s1, which fails it (E 5, d7 0.701). nostep_k8_s1 competent (0.913) and extrapolates at K′ = 16 (d7 0.708,
  CI 0.690–0.725) but E = 3, so not scorable for P21; nostep_k8_s2 ends at stage 2 → gate failure. P21: 0 scorable →
  contingency: nostep_k8_s3 (3070) and nostep_k8_s4 (4060) started in parallel (both are needed for any reading).
* Stall pilot (results/stall_pilot.json): 0 of 12 stalls. Every run cleared stage 1 by step 1000 except base_s0
  (3000), in all four configs. The base recipe with a step embedding does not stall. The stage-1 failures on
  record are nostep_s3, nostep_s4, nostep_k8_s1, v1_wide_s1/s2, v2_deep_s0, v3_rawfb_s1, main_s5 and budget_k4_s1,
  mostly no-step-embedding or architecture variants, so the pilot does not test the configs that stall.
* ~12:10 nostep_s4 (2nd attempt) competent (0.903, matched-5 0.698, E 4), no extrapolation at K′ = 12 (d7 0.514)
  → P19: 2 competent runs (s2, s4), neither extrapolates; nostep_s5 training (one stage-1 relaunch).
  nostep_k8_s3 fails the gate (stage 3, 0.810); nostep_k8_s4 reaches full depth but fails matched-5 (0.564).
  P21: five seeds, none scorable → the addendum-D stopping rule is reached; **P21 unresolved**.
* 12:12 no-step stall pilot (`scratch/stall_pilot.py ... nostep`): `--K 6 --no_step_emb`, bf16 (4060 / 3070),
  seeds 10–15 × 4 configs → results/stall_pilot_nostep.json. The P19 stalls (s3, s4, s5) were fp32 runs.
* 13:11 nostep_s5 (2nd attempt) ends at stage 4, fails the gate (0.854, matched-5 0.528). **P19 final**: 5 seeds,
  2 competent (s2, s4), neither extrapolates at K′ = 12 → not replicated; the addendum-D contingency is exhausted.
* 13:17 no-step stall pilot done (results/stall_pilot_nostep.json; K 6, --no_step_emb, bf16, seeds 10–15, stopped
  at stage-1 advance or step 4000). Stalls: base 3/6, wu1000 1/6, lr5e-4 0/6, lr5e-4+wu1000 0/6. Advance steps:
  base 2000–3000 when it advances; wu1000 1000; lr5e-4 1000–3000; lr5e-4+wu1000 1000–2000. The lower LR removes
  stalls (0/12 vs 3/6 at base, one-sided Fisher p ≈ 0.025); warmup 1000 alone mostly helps. The base-recipe
  pilot (with step embedding) had 0/12. Per TRAINING_TODO, a changed recipe is only for runs answering new
  questions (Phase 3–4), not replications.
* 13:40 PROTOCOL4 addendum E written (evaluation-only confirmations P23–P25; discloses the exploratory frontier test). sha256(PROTOCOL4.md) at freeze: 50b52ed16605fdc9…

## 2026-09-25 (afternoon) — addendum-J re-score of earlier models; addendum E confirmations (P23–P25)

* **Addendum-J gate re-score (sensitivity only; J is not retroactive)** (`scratch/rescore_gateJ.py`,
  results/rescore_gateJ.json): of the 16 PROTOCOL2/3 models with competence files, only main_s5 changes status
  (delayed 0.891, matched-5 0.584 → not competent). Under J the addendum-I held-out replication would be
  unscorable on seed 5 instead of failed, leaving seed 3 alone (seed 4 failed the old gate too). No other PROTOCOL3
  verdict used s5 as a competent model. Official verdicts unchanged.
* **Addendum E run** (`latent/confirm_sweep.py`; confirm split, seeds as frozen; results/confirm_sweep*.json;
  ~7 s per model on the 4060). All 22 J-competent recurrent models meet the frozen "sweeps" criterion.
  **P23 fails** (9/22 pass): the on-diagonal drop holds in 18/22, but the off-diagonal limit (|Δ| ≤ 0.02 at
  |k − k*| ≥ 2) fails in 13. **P24 fails** (9/22): per-edge W8/W2 lies in 0.13–0.32 in all 22 (band 0.10–0.45)
  and the total ratio is in band in 20/22, but the argmax clock changes with W in 12 models.
  **P25 mixed** (primary 14/18 = 0.78, bar 0.80; fork/chain ratios 0.36–0.66 in passing models); the s0–s3
  replication on fresh graphs is 4/4 (0.49–0.60). The P25 failures are budget_k4_s0 (depth 3, 0.90), nostep_s4
  (depth 3, 1.92), budget_k8_s1 (depth 2, 0.74) and nostep_k8_s1 (depth 3, 0.85).
* Exploratory reading of the failures (after scoring, not a verdict): the models split into regular clocks
  (one step per depth, the same clock at W 2/4/8, off-diagonal |Δ| ≤ 0.014: s0, s1, s3, v2_deep_s0, randk_s0,
  budget_k10_s1, nostep_s4) and irregular ones (clocks like
  [1,5,6], [1,3,8], [2,7,10]; off-step blocking effects 0.07–0.24; argmax shifting with W), where a depth is read
  over several steps. The frozen 1.5×-decoy criterion let every model count as "sweeping", so single-step-locked
  predictions were scored on models that read a depth over several steps. The width amplitude law (P24b) holds
  in all 22 models; the step-locked clock (P23, P24a) holds only in the regular subset.
* 14:30 Part IV report built: report/protocol4_report.{html,pdf,md} (report/build_protocol4_report.py, all numbers read from results/); README updated with Part IV results and files.

## 2026-09-25 (late afternoon) — PROTOCOL5: plan steps 1–4 (docs/RESEARCH_TIMELINE.md §6)

* **Step 1, housekeeping.** `latent/protocol4_confirm.py` now applies addendum D's full-depth clause (field
  `gate_accuracy` keeps the accuracy-only gate). Only randk_s1 changes (stage 4 → gate failure); P22 still fails
  (competent randk_s0, s2). PROTOCOL4 totals become 11 of 22 competent, 4 of 11 extrapolate. Part IV report rebuilt
  (`report/build_protocol4_report.py`; one hard-coded count, 3 of 9 → 3 of 8 K ≥ 8 runs, and the disclosure bullet
  updated). docs/PROTOCOL4_RESULTS.md: randk_s1 marked as a gate failure; nostep_k8_s1 added to the "d7 ≥ 0.60 at
  K′ = 12" list it was missing from.
* **Disclosure found while reading logs:** runs/nostep_s0 (P18, PROTOCOL3 addendum K) ended at curriculum stage 4
  (advance to 4 at 21 500, never to full depth). PROTOCOL3 had no full-depth clause, so its P18 verdict stands as
  scored, but under the PROTOCOL4 addendum-D rule it would be a gate failure. It is one of the 8 extrapolating models in
  the step-2 screen and one of the 22 addendum-E models; both analyses are reported as run.
* **PROTOCOL5 addendum A** frozen (reliability pilot: no-step K = 6 recipe, lr 1e-3 vs 5e-4, paired seeds 20–27, 16
  runs, prediction R1). Launched ~15:05: local RTX 4060 / 3070 via `scratch/queue_worker.sh` (shared queue
  `scratch/rel_queue.txt`, detached), RunPod community RTX 4090 ($0.34/h: rel_base_s20, then s27 base and lr5) and
  secure RTX A4000 ($0.25/h: rel_base_s26, rel_lr5_s26). Cloud stock was nearly empty; a secure RTX 3090 billed at
  $0.50/h and a secure RTX 4090 at $0.74/h were deleted within a minute, unused.
* **Step 2, extrapolation predictor (exploratory; scratch/extrap_predictors.py → results/extrap_predictors.json).**
  22 addendum-E models, 8 extrapolate at K′ = 12 or 16. Of 15 in-budget features only *end activity* (layer-1 frontier
  attention at the last trained step K relative to its peak, W = 4 confirm trace) reaches AUC ≥ 0.80: 0.81 [0.57,
  0.97], permutation p = 0.017, Holm 0.26; 0.84 / 0.82 at W = 2 / 8. Below 0.25: 1 of 9 extrapolate (nostep_k8_s1);
  at or above: 7 of 13. E, clock rate, slack, clock regularity, accuracy, curriculum timing, K and step embedding: AUC
  0.50–0.66. Post-budget activity (steps K+1..8, K < 8 models only) gives 0.95, but it is close to restating the
  outcome. Frozen as P26 (addendum B) before any pilot evaluation was read.
* **Step 3, strict clock (addendum C; `latent/protocol5_confirm.py strict`, results/strict_clock.json; RTX 2070
  fp32; discovery seeds 5050+W for qualification, fresh confirm seeds 3030+W / 3037 for scoring).** 9 of 22 models
  qualify as single-step readers (s0, s1, s2, s3, v2_deep_s0, budget_k8_s0, nostep_s4, randk_s0, budget_k10_s1).
  **P23′ 7/9 and P24′ 7/9 → mixed** (0.78, bar 0.80). Failures: budget_k8_s0 (off-diagonal −0.033/−0.034 and its clock
  shifts at W = 8), v2_deep_s0 (only the other-branch control clause, min −0.0101 against −0.01), main_s2 (P24: clock
  [1,3,5] / [1,4,5] / [2,4,6] across W). The 13 non-qualifying models: P23 1/13, P24 2/13. The discovery clock
  equalled the fresh confirm clock in 21 of 22 models.
* **Step 4a, late-stage restart rule (scratch/stall_rule.py → results/stall_rule.json).** 53 gated attempts, 23 reach
  full depth. Successful dwell per stage: stage 2 median 4 250 (max 15 500), stage 3 6 000 (16 000), stage 4 5 000
  (14 000). Replaying "relaunch after T steps in one stage" with relaunches drawn from the same trajectories: expected
  steps per full-depth run 52 200 with no rule, 47 800 at T = 14 000 (kills no eventual success), 55 600 at T = 8 000.
  A late restart rule saves at most ~8%; slow successes and stalls overlap too much. Added to the guard as an opt-in
  `--late T` (tested with a fake trainer), not used by the pilot.
* **Pilot operations (~16:50).** Secure A4000 pod (rel_base_s26, rel_lr5_s26) finished; results fetched
  checksum-verified and the pod deleted. A third pod (community RTX A4000, $0.17/h) took the seed-25 pair. Its container
  restarted at ~20:43 UTC (uptime reset, every process killed, files intact) while rel_base_s25 was at step 13 000;
  noticed ~10 min later (user report: GPU idle), resumed exactly from `state.pt` (step 13 000; `--resume`), chain and
  eval queue re-deployed. No data lost; ~12 min of pod time idle.
* **~18:20–19:00 pilot complete.** All pods fetched (checksum-verified) and deleted; RunPod list-pods empty. End activity
  (P26) and strict-clock measurements run on the 4060 after the frozen addenda, on all 16 pilot runs.
  `latent/protocol5_confirm.py` score: pilot strict-clock set restricted to gate-passing runs, as addendum C says
  (fixed before scoring). Scorecard: docs/PROTOCOL5_RESULTS.md.
* **R1 holds, weakly.** Gate: lr 5e-4 7/8, base 5/8 (Fisher one-sided p = 0.28; 5/6 vs 5/8 without the two fp32
  runs). Every failure is a late-stage stall (base: s21, s22 at stage 2, s23 at stage 4; lr 5e-4: s25 at stage 4);
  every full-depth run passed the accuracy clauses. No stage-1 relaunch in 16 runs. The lower LR costs accuracy among
  full-depth runs (matched-5 0.693 vs 0.769, matched-6 0.618 vs 0.669).
* **P26 confirmed prospectively.** 12 competent runs: end activity < 0.25 → 1 of 6 extrapolate (rel_base_s25);
  ≥ 0.25 → 5 of 6 (rel_lr5_s21 misses at 0.583, lower bound 0.566); AUC 0.83. Pooled with the exploration models
  (descriptive): 2 of 15 vs 12 of 19. A sweep still attending to the frontier at the last trained step is close to
  necessary for extrapolation, and in this recipe usually sufficient.
* **Strict clock on new models:** 4 competent pilot runs qualify; P23′ 3/4, P24′ 3/4. All sets: 10/13 each.
* rel_base_s23 stalled at stage 4 (a gate failure) yet extrapolates strongly (d7 0.712, end activity 1.00), like
  nostep_s0 — two runs that never trained at full depth are among the stronger extrapolators. Not a prediction; noted
  for the next protocol.

## 2026-09-25 evening — PROTOCOL6 (Part VI: why some sweeps keep running; transfer preparation)

* **Exploratory before writing the protocol (scratch/ea_vs_lrint.py).** Over the 38 models with end activity (EA)
  measured, EA is unrelated to the learning-rate mass spent at full curriculum depth (Spearman +0.15) or to the LR
  when full depth was reached (+0.08). The "closure" account (training at full depth ends the sweep inside the
  budget) is not favoured cross-sectionally; runs reaching full depth late (rel_base_s20, randk_s0) even have low EA.
* **docs/PROTOCOL6.md written (~19:55)** before any run: addendum A (continue the 12 competent pilot runs for 15 000
  steps; A1 EA class stable / closure / reopening; A2 P26 on continued models and whether extrapolation tracks EA
  changes; A3 the five stalled runs continued, exploratory), addendum B (12 new development runs, step-embedding
  recipe at lr 5e-4 with a checkpoint every 1 000 steps; B1 trajectories; P26b and P23″/P24″ prospective), addendum C
  (Coconut GPT-2 on ProsQA, preparation).
* **Amendment before any continuation ran (~20:10).** Seven of the 17 runs to continue (five of them low-EA) trained
  on pods whose state.pt was never fetched; every continuation therefore starts from model.pt with a fresh AdamW state
  and a fresh data stream (scratch/p6_setup.py). A first launch crashed at once (empty optimizer dicts raise KeyError
  in the resume copy); fixed with explicit zero moments, results and logs of the failed start deleted, relaunched
  ~19:58 on all three local GPUs (scratch/p6_queue.txt for the 4060/3070 in bf16, scratch/p6_queue_2070.txt for the
  2070 in fp32). A check along the way: saved state.pt files do hold optimizer moments (34 of 36 parameters; two get
  no gradient), so earlier exact resumes were exact.
* **dev_s4 checkpoint trajectory (the only earlier run with checkpoints; stalled at stage 3).** EA ~0.1–0.2 with noise
  during stages 1–2, then settles at ~0.03 from step 20 000. Stalling below full depth is therefore not sufficient for
  an open sweep.
* **ProsQA transfer preparation.** Official Coconut code cloned (D:\coconut). prosqa/prosqa_1gpu.yaml = official
  recipe on one GPU (32 × grad-accum 4 = effective 128, fp32, 50 epochs, stages of 5 epochs, 6 thoughts);
  prosqa/pod_prosqa.sh trains on a pod and keeps stage-boundary checkpoints. prosqa/gpt2c.py re-implements the GPT-2
  forward and Coconut rollout in plain torch (verified on the public GPT-2 weights) so attention can be read and
  masked; prosqa/analyze.py measures the per-hop frontier trace, end activity per layer, step × hop blocking of
  answer-path statements (with same-hop decoy controls) and accuracy at K′ = 0–12, parsing all 800 valid/test
  problems. Tested end to end on untrained GPT-2. Cloud: a secure 4090 ($0.74/h) and a secure A6000 ($0.53/h) billed
  above the cap and were deleted at once; a community 4090 at $0.34 had driver 580.173.02 (cuInit 999) and was deleted;
  no other stock at or under the cap. Retrying.
* **ProsQA pod fixes (21:15–00:00).** Batch 16 × 8 ran out of memory in epoch 3 → restarted from scratch at 8 × 16.
  In stage 1 the official coconut.py ran at 13.5 s per optimizer step with the GPU at ~18% (per-position Python
  rebuild of the input embeddings after each latent pass). Speed patch (prosqa/patch_coconut.py, one out-of-place
  index_put) checked against the unpatched code on real batches from checkpoint_5 at stages 1/3/6: identical losses,
  gradient difference ≤ 4.2e-7, 6–10× faster. Resumed from checkpoint_5 (stage boundary) at 2.8 s per step, GPU 96%.
  Two test-script hangs on the way: datasets.map forks 32 workers, which deadlocked after CUDA started or before
  the model was built; the test now follows run.py's order.
* **~00:00 addendum A scored (latent.protocol6 score; competent continuations complete).**
  * **A1 stable:** EA class unchanged in 11 of 12 (high → low 0/6, low → high 1/6: rel_base_s24 0.17 → 0.32). No
    closure: 15 000 further full-depth steps with a restarted optimizer leave the end activity where it was.
  * **A2:** P26 on the continued models: low 1 of 5 extrapolate (rel_base_s25 again), high 4 of 7; AUC 0.71 →
    "confirmed" by the frozen rule, but weaker than before continuation. Tracking not scored (1 class change; it did
    not bring extrapolation: rel_base_s24 d7 0.596). Extrapolation itself eroded a little: 5 of the 6 extrapolators
    kept it; rel_lr5_s24 lost it (d7 0.614 → 0.561) while its EA stayed high (1.00 → 0.76).
  * **A3 (stalled runs):** rel_base_s21/s22 stayed at stage 2. rel_base_s23 reached full depth at step 38 500 and
    kept both its open sweep (EA 1.00 → 1.00) and its extrapolation (d7 0.712 → 0.711). rel_lr5_s25 reached full depth
    at 43 000, EA stayed low (0.018), no extrapolation. nostep_s0 reached full depth at 43 000: EA 0.89 → 0.63 (still
    high) but extrapolation was lost (d7 at K′ 12 0.631 → 0.543; K′ 16 0.671 → 0.559). Reaching full depth closed
    neither open sweep, so "never training at full depth keeps the sweep open" is not supported.
* **~04:40 addendum B scored (12 dev6 runs: step-embedding recipe, lr 5e-4).**
  * Gate 7/12; failures: stalls at stage 3 (s31) and 4 (s34, s36, s41), and s37 reached full depth but missed the
    accuracy clauses. No stage-1 relaunch. More late stalls than the no-step recipe at the same LR (1/8).
  * **P26b "confirmed" by the frozen rule but thin:** EA ≥ 0.25: 5 of 6 competent runs extrapolate; EA < 0.25: 0 of 1
    (dev6_s35). AUC not computable (one low run). bf16 only: 4/4 vs 0/1. Most runs of this recipe end with an open
    sweep (6 of 7 competent).
  * **Strict clock:** one single-step reader among the competent dev6 runs (dev6_s38: P23 fail, P24 pass). Pooled
    prospective sets: P23″ 3/5 (mixed), P24″ 4/5 (confirmed, at the bar). The step-embedding recipe rarely produces
    single-step readers, so this set could not settle the clock claim.
  * **B1 trajectories:** the final EA class is already set 1 000 steps after full depth in 7 of 8 competent-or-full
    runs. The exception, dev6_s35, reached full depth with an open sweep (0.57) that closed during the rest of the
    schedule (0.32 → 0.15): closure does happen inside normal training, though 15 000 extra steps after the schedule
    (addendum A) never caused it.
* **04:39 addendum E started** (scratch/p6_night.sh): selection by the frozen rule gave dev6_s30, dev6_s32 (open) and
  dev6_s35 (closed; the only low-EA competent run) → 18 branches on the 4060/3070. **Extension B′** (dev6_s42..44,
  fp32) on the otherwise idle 2070, disclosed in PROTOCOL6 and scored separately.
* **ProsQA pod:** stage 3 done at epoch 20, validation 0.94 (0.85 → 0.94 across the stage).
* **~10:20 addendum F (precision) pre-registered and started.** Extension B′ (dev6_s42–44, fp32 on the 2070) all
  ended with open sweeps (EA 0.59, 0.46, 0.52) and none extrapolates. Tabulating every competent model: with EA ≥ 0.25,
  bf16-trained models extrapolate 14/19, fp32-trained 3/8 (confounded with the 2070 card and recipe). Paired test:
  seeds 50–57 trained in bf16 and fp32 on the same card (4060: 50–53, 3070: 54–57), plus an fp32 arm on the 2070 for
  seeds 50–53. scratch/p6_prec_chain.sh starts each card when its addendum-E branch work is done.
* **~10:40 addendum E scored (18 branches; latent.protocol6 score, "E1").** Agreement with the original's EA class
  among competent branches: from stage-2 entry 2/4 (**open**; both dev6_s30 branches stalled at stage 4 and do not
  count), from stage-4 entry 5/6 (**partial**), from full-depth entry 6/6 (**set**). The open/closed outcome is fixed by
  the time the run reaches full depth and not before stage 4. dev6_s35 is the telling case: its EA at full-depth
  entry was still 0.57, yet both branches restarted from that checkpoint (new data order, fresh optimizer) closed to
  0.15/0.14 like the original. Earlier branches of dev6_s35 mostly ended open (0.35, 0.30, 0.81; one 0.16). The
  closure was already determined by the weights at full-depth entry, before EA showed it. Extrapolation followed the
  original in 17/18 branches (dev6_s32_brst2_1 lost it with EA 0.42).
* **~16:15 ProsQA transfer (addendum D) scored (prosqa/score.py → prosqa/res/protocol6_prosqa.json).** Coconut GPT-2,
  official recipe (8 × 16, speed patch), final validation 0.973 (0.99 at epochs 48–49); test accuracy by candidate
  scoring 0.974 (T0 competent).
  * **T1 sweep absent.** Frontier layer L* = 7 (valid). Attention reads the hop-1 and hop-2 statements only; the
    pattern at thoughts 2–6 is identical (a fixed point, not an advancing frontier); hop ≥ 3 statements get no excess.
    Path clock valid [1, 1, 2], test [1, 1, 1].
  * **T2 "thoughts not needed".** Blocking a path statement at all six thought steps moves the answer margin by at
    most 0.09 nats (base margin 12.2 nats); only step-1 blocks have any effect. Accuracy with 0 thoughts 0.976 vs 0.974
    with 6 (K′ 0–12 all 0.974–0.976): the answer re-reads the question. This replicates Aswal et al. (arXiv
    2606.12689: 97.8% vs 98.0% without thoughts) with an independent implementation.
  * T3: EA at L* 0.52–0.77 across epochs 30–50, but with no advancing sweep the measure has no meaning here.
  * Pod deleted after checksum-verified fetch of the results and checkpoint_50 (807 MB, sha256 match; this community
    pod exposed direct SSH, so scp worked). RunPod list-pods empty.
* **~16:25 addendum G pre-registered** (bypass-blocked Coconut on ProsQA; prosqa/patch_mask.py, test_mask.py,
  pod_prosqa_g.sh; analyze.py --mask_answer). Exploratory on the official model first: statements hidden from the
  answer → 0.755 / 0.73 (0 / 6 thoughts); everything before the thoughts hidden → 0.52 / 0.54 (chance). No community
  card ≤ $0.34/h in stock at 16:30–17:00; retrying.
* **Attention sharpness at full-depth entry (scratch/sharpness.py → results/sharpness.json; exploratory, literature
  follow-up).** 10 dev6 runs with checkpoints: the one competent closing run (dev6_s35) is not sharper at full-depth
  entry than open runs (top-edge share 0.21 vs 0.11–0.24; frontier ratio 6.6 vs 3.4–13.6). Uninformative with one
  closer; the no-step precision runs now saving checkpoints will add closing runs.
* The user gave full control over direction and pod spending (report only when RunPod credit runs low).
* **~19:45 addendum G started on Vast.ai** (RunPod still out of stock; user added a Vast.ai account, $25 credit).
  Instance 52844008: RTX 3090, $0.211/h, driver 550 / CUDA 12.4, reliability 99.96%, direct SSH. Mask moved to an SDPA
  wrapper (eager attention would store every attention matrix). test_mask.py: zero attention mass crosses the mask at
  stages 0/3/6 and in generation; eager and SDPA losses agree to 2e-6, generations identical; the mask changes the
  stage-6 loss (14.01 vs 6.08 without). A first test run failed on a test-script bug (Coconut.eval() returns None),
  fixed before training. Training at ~4.1 s per optimizer step in stage 0 (4090: 3.55).
* **~20:15 addendum G replication launched** (pre-registered: seeds 1, 2 on two more Vast.ai 3090s, 52846875 at
  $0.258/h and 52846882 at $0.281/h; reading = the one shared by ≥ 2 of 3 seeds). Note: the official validation
  *generates* the written steps after `<end-latent>`, which the mask hides from the question, so validation accuracy
  is meaningless before the final stage (epoch-1 validation 0.0); training is unaffected (written steps see the
  question in training) and the final stage has no written steps.
* **Independent replication of addendum D on a public checkpoint** (bmarti44/coconut-curriculum-checkpoints,
  `coconut/checkpoint_49`, GPT-2 Coconut on ProsQA, reported 97.0%; exploratory, test set n = 300,
  prosqa/res/ext_bmarti44_coconut_e49.json). Accuracy with 0 thoughts 0.977, with 6 thoughts 0.973; statements
  hidden from the answer 0.82 / 0.87; everything before the thoughts hidden 0.54 / 0.59. Their Coconut also does not
  need its thoughts; the thoughts carry at most ~5 points of answer information. Meta released no checkpoints; that
  repository also holds matched pause-token controls (their paper: pause 96.6% vs Coconut 97.0%).
* **~20:37 addendum H pre-registered and started** (fp32 vs bf16 curriculum stalls, same-card pairs seeds 70–77; scratch/p6_stall_chain.sh). The 4060 had gone idle after addendum F.

### 2026-09-26 21:35 — P27/P28 scorers written (before the last F run and before any H run finished)

* `latent/protocol6.py score` now also writes keys `F` (P27: both-competent same-card pairs, bf16-only vs fp32-only
  extrapolators, exact one-sided sign test, arm summaries, P26 per arm, 2070 arm) and `H` (P28: pairs where only one
  arm reaches full depth; steps per stage; descriptive pool with the F pairs). Rules copied from addenda F and H.
* A run counts as finished when its eval battery output exists (`model.pt` also appears mid-run with `--save_every`).
* Interim (7 of 8 F pairs finished): bf16 reached full depth 6/8 vs fp32 3/8; competent 5 vs 2; extrapolating 1 vs 1.
  Only pair 54 has both arms competent (fp32-only extrapolator). Full-depth discordance so far b = 3, f = 1.
* Addendum G seed 0 passed its first stage checkpoint (epoch 5) on Vast; validation 0/300 is expected before epoch 30.

### 2026-09-26 ~22:05 — Addendum F scored: P27 not supported

* `prec_fp32_s53` reached full depth at step 28 000 and finished at 21:47 (gate ✓, d7 0.601 [0.583, 0.619], EA 0.86;
  measured by hand: no measurement loop covered `p6_prec_3070b.txt`). Its bf16 twin: gate ✓, d7 0.564, EA 0.54.
* P27: both arms competent in pairs 53 and 54; fp32-only extrapolator in both → b = 0, f = 2 → **not supported**.
  Extrapolation is rare in this recipe for both precisions (competent: bf16 1/5, fp32 2/3). The exploratory
  precision pattern was the card/recipe confound.
* Full depth: bf16 6/8, fp32 4/8 same card, fp32 2070 0/4; discordant bf16-only 3, fp32-only 1 (seed 57). Same seed,
  same precision, different card (4060 vs 2070) also gives different curriculum outcomes.
* Details in `docs/PROTOCOL6_RESULTS.md` (Addendum F). The 3070 moved on to addendum H (seeds 74–77) at 21:48; the
  4060 finished `stall_bf16_s70` at 21:46.
* Fixed `scratch/p6_wait3.sh` printing the wrong event line (it now prints only the new lines).

### 2026-09-26 ~23:45 — Sharpness null; post-full-depth drift; extension B″ started

* `sharp_fp32_s62` stalled at stage 3 (EA 0.89, d7 0.50); `sharp_fp32_s60/61` reached full depth, neither extrapolates.
* Sharpness at full-depth entry (18 runs): no separation of extrapolators (top1 0.156–0.243) from the rest
  (0.108–0.356). Exploratory null.
* Trajectories: 15/17 runs keep their EA class from full depth + 1 000 to the end; `dev6_s35` slowly closes and
  `prec_bf16_s53` slowly reopens (0.06 → 0.54) without extrapolating. Within-run EA–d7 Spearman: 3 positive, 3 negative
  (runs moving ≥ 0.2). EA is a between-run marker, not a within-run dial. Written up in PROTOCOL6_RESULTS.md.
* Pre-registered extension B″ (PROTOCOL6.md, ~23:40) and started `dev6_s45..s50` on the idle 2070 (queue
  `scratch/p6_bext2_2070.txt`, measurement loop `p6_after2.sh cuda:2`).
* ProsQA G seed 0: checkpoint 12 by ~23:40, training loss ~0.05 (healthy); the mask only binds in the last stage.

### 2026-09-27 ~06:50 — Why open sweeps fail to extrapolate: the frontier stops at the horizon (exploratory) → addendum I

* 3070 finished addendum H (06:48). H so far: 7/8 pairs, b = 1 (s72), f = 0; the F stall effect is weaker on new seeds.
* New probe `scratch/far_sweep.py`: layer-1 frontier excess on the distance-7 edge of the length_gen long-chain graphs,
  relative to the in-horizon peak (P7). 72 competent K = 6 models: AUC for extrapolation 0.96 vs EA 0.82; within
  high-EA 0.98; P7 ≥ 0.2 → 28/29 extrapolate, < 0.2 → 8/43; Spearman with d7 0.84. Non-extrapolating open sweeps
  mostly have P7 ≈ 0: the frontier never crosses the trained horizon (propagation failure, not readout failure).
  Two extrapolators with P7 ≈ 0.03 (rel_base_s25, rel_lr5_s23) must use another route.
* P7 on fresh graphs (seed 2033) moved ≤ 0.03 on 8 models. Pre-registered addendum I (P29) with a new batch
  `p29_s80..87` (3070 now; 4060 after H) plus the unfinished s73 pair and dev6_s48..50 as the prospective test set.

### 2026-09-27 ~10:30 — Addendum H scored: P28 not supported; P29 unresolved → more seeds

* H complete: full depth 6/8 in both arms; bf16-only s72, fp32-only s73 → b = 1, f = 1 → **not supported**. The F
  stall imbalance was chance. Extrapolation in both-competent H pairs: 1 vs 1. Precision matters for neither.
* P29 (addendum I), test runs finished so far: competent `stall_fp32_s73` (P7 0.22, not extrap), `dev6_s48` (0.34, not),
  `p29_s80` (0.01, not), `p29_s84` (0.51, extrap); `stall_bf16_s73`, `p29_s81..83` stalled at stage 4; `dev6_s49`
  full depth but failed the gate. Only 1 extrapolator → **unresolved** → as pre-registered, more seeds of the same
  batch: `p29_s88..95` on the 3070 (`scratch/p6_p29b_3070.txt`), s85–87 still on the 4060. Scorer:
  `scratch/score_p29.py` (runs the frozen `far_sweep_frozen_I.py`, seed 2033, and EA measurement).
* Weakness noticed (not changing the frozen rule): P7 is a ratio, unstable when the in-horizon excess is ≈ 0 —
  `stall_fp32_s73` (in_peak 0.0003) and `dev6_s48` (0.005) get P7 0.22 / 0.34 from noise. The 72-model basis had few
  such models. Report the in_peak alongside every P7.
* `dev6_s50` started at 09:25, after addendum H finished (08:49): by the B″ rule it is excluded from the B″ pool; it
  stays in the P29 test set (addendum I lists dev6_s48..50 "that get trained").

### 2026-09-27 ~11:30 — P29 resolves, interim "not supported"; extra seeds stopped

* New finished test runs: `dev6_s50` competent (P7 0.08, d7 0.609 → extrapolates), `p29_s85` stage 4, `p29_s88`
  competent (P7 0.02, not). Test set now 2 extrapolating / 4 not → resolved. Interim: accuracy 0.50, AUC P7 0.75
  (EA 0.50) → **not supported** if it holds after s86, s87, s89. Seeds 90–95 dropped (amendment in PROTOCOL6.md).
* The misses: `stall_fp32_s73` (in-horizon excess 0.0003) and `dev6_s48` (0.005) get P7 ≥ 0.2 from a near-zero
  denominator — both below or at the minimum of the 72-model basis (0.0049; 5th percentile 0.035). `dev6_s50`
  extrapolates with a small distance-7 excess (0.009 vs in-horizon 0.11). Exploratory, not a rescue: with a floor of
  in_peak ≥ 0.03, 3 of 4 would be classified correctly. Lesson for the write-up: a ratio measure needs a floor
  pre-registered with it.
* ProsQA G seed 1 trained (validation 0.957); analyses running. The 2070 is idle (B″ queue finished 11:23).

### 2026-09-27 ~11:45 — Addendum G seed 1: thoughts needed, but as a conduit; no sweep

* Seed 1 finished (TRAIN_EXIT 0, ANALYSIS_DONE 11:29); fetched with checksums (res + checkpoint_50), Vast instance
  52846875 destroyed. G0 competent (0.972), G1 **needed** (K′0 0.498), G2 absent, G3 not step-locked.
* Exploratory: K′ = 1 already gives 0.938 (0.913 on 5-hop paths). The prefix pass solves ProsQA; the thought carries
  the answer past the mask. Test data fetched locally to `prosqa/data/prosqa_test.json` (sha matches the pod copy).
* Candidate next experiment (not yet pre-registered): delayed-query ProsQA — statements + start entity before the
  thoughts, candidate question after `<end-latent>`, answer positions blind to the statements. Decide after seeds 0/2.

### 2026-09-27 ~12:05 — Addendum J pre-registered and launched (delayed-query ProsQA)

* Built `prosqa/make_delayed.py` (prefix = statements + "Start: <root>."; query field), `prosqa/patch_delayed.py`
  (query after `<end-latent>`, never a target; mask extended to the query and its newline), `prosqa/test_delayed.py`
  (layout, labels, exact blocked rows, eager = SDPA), `prosqa/pod_prosqa_j.sh`, configs `prosqa_j*.yaml`; `analyze.py`
  parses delayed samples (query tokens prepended to both candidates; verified identical spans/hops/paths on 500 test
  problems) and takes `--budgets`. Patches apply cleanly to Coconut commit 27273cb (same as the pods).
* PROTOCOL6 addendum J written ~11:45, before training. RunPod: no card ≤ $0.34/h (CUDA 12.4–12.8). Vast: 52981944
  (seed 0, $0.211, driver 550) and 52981949 (seed 1, $0.281, driver 565), cuInit 0 on both, launched ~12:00.
* Vast credit $13.34 at 11:50; ~$3 expected to remain after J.

### 2026-09-27 ~12:15 — P7 on K = 8/10 models (exploratory)

* 8 competent K ≠ 6 models (budget_k8/k10, nostep_k8, randk): extrapolators P7 0.10/0.58/0.14, others 0.00–0.14;
  AUC ≈ 0.90, threshold 0.2 too high for them. Non-extrapolators with spare thoughts still stop at distance 6 → the
  horizon is the training graph depth, not K. `p29_s89` finished 12:07 (3070 idle; 2070 idle; 4060 on s86/s87).

### 2026-09-27 ~13:15 — Addendum G seed 2: replicates seed 1 → G reading settled (2 of 3)

* Seed 2 fetched (sha ok), Vast 52846882 destroyed 13:10. G0 0.940, G1 needed (K′0 0.516), G2 absent, G3 not
  step-locked; K′ = 1 0.932. Same as seed 1 on every test → addendum-G reading fixed. Seed 0 (epoch ~49) runs to the end
  as the third seed.

### 2026-09-27 ~13:25 — P29 not supported; addendum K (P30) pre-registered and launched

* P29 final (8 competent test runs, 3 extrapolate): accuracy 0.50, AUC 0.73 (EA 0.53) → **not supported**. Misses:
  two near-zero denominators (in_peak 0.0003, 0.005) and two extrapolators just under 0.2 (0.08, 0.11).
* Local GPUs were idle between their last P29/B″ runs and 13:23 (2070 from 11:23, 3070 from 12:07, 4060 from 13:18).
* Addendum K (~13:23): threshold-free P30 (AUC ≥ 0.80 among competent runs with in_peak ≥ 0.03, and > EA) on 30 new
  runs: `p30_s100..123` (4060/3070, bf16) and `dev6_s51..56` (2070, fp32). All queued runs are trained. Launched 13:23.

### 2026-09-27 ~14:35 — Addendum G complete: 3/3 seeds agree

* Seed 0 fetched (sha ok), Vast 52844008 destroyed 14:30. G0 0.972, G1 needed (K′0 0.494), G2 absent, G3 not
  step-locked, K′ = 1 0.932. All three seeds agree on every test. Only the two addendum-J instances remain on Vast.
- 2026-09-27 ~15:10: Vast A5000s 53014299, 53014302 run P30 extension seeds 124–159 (addendum K amendments 14:56/15:05)

### 2026-09-27 ~22:25 — J enters the last stage holding accuracy; addendum L (K = 8) chained

* Addendum J validation at stage 6 (all steps latent): seed 0 0.84–0.87 (epochs 31–34), seed 1 0.86–0.90 (29–32).
* P30: 7+10+4 local and 12+9 Vast runs finished. To keep the local cards busy after P30, addendum L (K = 8, 16 runs)
  was pre-registered (~22:22) and chained behind each P30 queue (`scratch/chain_after.sh`).

### 2026-09-28 ~02:05 — Addendum M (depth cap) pre-registered and chained; L extended

* User: keep designing experiments through the night without waiting (saved to memory).
* Addendum L extended by 7 seeds (amendment ~01:45, no L result read). Vast host A: 18 P30 runs fetched (sha ok),
  destroyed 01:55; host B still running.
* `latent/train.py --depth_cap` (default off; smoke-tested). Addendum M: K = 6, training depth capped at distance 5,
  17 runs chained behind the L queues. Tests whether failing sweeps stop at the training depth (5) or the step budget.

### 2026-09-28 ~03:40 — P30 supported: P7 ranks extrapolators prospectively (AUC 0.993)

* Vast host B: 16 runs fetched (sha ok; `scratch/vfetch_toy.sh` fixed to skip model-less runs), p30_s136/s159 were
  stage-1 failures; host destroyed 03:25. No P30 hosts remain.
* P30 (42 competent new runs, 21 extrapolate): AUC P7 0.993 vs EA 0.673; per card 0.96–1.00; per recipe 0.995 / 1.00;
  Spearman P7–d7 0.87. P7 ≥ 0.2 → 15/15 extrapolate; P7 < 0.1 → 3/24, all marginal (d7 ≤ 0.62). **Supported.**
- 2026-09-28 ~03:50: J seed 0 scored — competent 0.89; K′=0 0.85 (not needed), K′=1 0.898 (prefix-computed), no sweep, not step-locked; channel = <start-latent> position. Vast 52981944 destroyed.
- 2026-09-28 ~04:10: addendum N (4-layer GPT-2, delayed query) pre-registered 03:57; Vast 53139194 (s0), 53139198 (s1) launched (cuInit 0, 1950 MHz).
- 2026-09-28 ~04:45: J seed 1 scored — J1 needed (K′0 0.514) vs seed 0 not needed; J1b prefix-computed, no sweep, not step-locked (agree). Seed 2 launched on Vast 53145506 (pre-registered tiebreak).
- 2026-09-28 ~12:05: N seed 1 scored — test 0.552, N0 not competent (4-layer GPT-2 does not learn delayed ProsQA); Vast 53139198 destroyed. L noon cut-off dropped nothing (k8_s221 started 11:51).
- 2026-09-28 ~13:20: addendum L scored — L1 supported (AUC P7 0.90 vs EA 0.59; 17 competent, 4 extrap), L2 supported (12/13 non-extrapolators stop at distance 6 with 2 spare thoughts).
- 2026-09-28 ~15:40: N seed 0 scored — test 0.544, not competent; both N seeds fail N0. Vast 53139194 destroyed; only J seed 2 remains.
- 2026-09-28 ~15:45: addendum O (longer lookup scaffold, paired re-training of P30 seeds) pre-registered (basis: early full depth under scaffold → 8/10 extrapolate vs 17/49); chained after the M queues.
- 2026-09-28 ~16:50: O extension on Vast (seeds 124–147, both arms per seed on the same instance; 2×A4000 + 4070, all cuInit 0, 57–66 bf16 TFLOPS); amendments ~16:42 and ~16:50.
- 2026-09-28 ~20:10: addendum M scored — M1 supported (7/10 competent capped runs stop at distance 5 with a thought to spare; with L2: horizon = training depth, not step budget); M2 unresolved (2 extrapolators). 4060 chained into O at 20:00.
- 2026-09-28 ~21:55: J seed 2 scored (tiebreak) — K′0 0.478 → J1 needed (2 of 3); prefix-computed, no sweep, not step-locked like seeds 0/1. Vast 53145506 destroyed (sha-verified fetch). Addendum J closed. Addendum P (ProsQA deeper than training) pre-registered ~21:18, evaluation running on 3070 + 4060.
- 2026-09-28 ~22:25: gaps in the story addressed — addendum Q (GPT-2 trained on 3–4 hop ProsQA: does its depth horizon follow training depth, like the toy's?) on two Vast 3090s (53289329, 53289703; RunPod had nothing ≤ $0.34/h, two over-priced secure pods deleted within a minute); addendum R (toy cap-3 dose-response + 12 more cap-4 runs) chained locally after O. Scorers `prosqa/score_q.py`, `scratch/score_r.py` written before any data (score_r checked on M: reproduces 7/10).
- 2026-09-29 ~10:00: addendum P scored — P1 supported (5/6 GPT-2 models fall to chance with ≥ 4 added hops while controls hold), P2 not supported (12 thoughts never help). Depth-matched secondary: accuracy tracks the number of added hops, not path length (5–6-hop extended ≈ 7–8-hop extended), so it is not a training-depth horizon. 2070 memory thrash fixed (g_s2 control moved to 3070); vast1–3 and T11 pilot destroyed after fetch; T11 pilot not learned.
