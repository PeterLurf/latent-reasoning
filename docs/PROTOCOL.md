# Protocol — branch-level donor replacement on a shared latent state

Status: **draft written before discovery analysis.** Fields marked ⟨discovery⟩ are
filled from discovery-split data only and then frozen in `docs/protocol_frozen.json`
before any confirmation-split graph is generated.

## 1. System

* Model: 2-layer pre-LN decoder, d=256, 8 heads, trained from scratch
  (`latent/model.py`). Continuous-thought recurrence (Coconut-style): the final-LN
  output at recurrent position k is the input embedding of position k+1 (+ step
  embedding). K = 6 recurrent positions.
* Controls trained with the *same data, scaffold, and curriculum schedule*:
  `pause` (recurrent positions receive a learned constant: extra positions, no
  hidden-state feedback) and `K=0` (no recurrent positions).
* Training scaffold (documented deviation, see research log): auxiliary PARENT(v) and
  CHILDREN(v) lookups on the *prefix-only* cache, never for the start node, annealed
  to zero; depth curriculum gated on held-out accuracy. No loss on latent thoughts.

## 2. Task and splits

* 10 branches × 6-node random trees; W ∈ {1..8} attached to start `s`, rest to decoy
  `s'`. 62 nodes, 60 edge tokens, shuffled edge order, random ids from 80.
* Families (multiset of attached-flag × tree shape) hashed to 20 buckets:
  train 0–13 / discovery 14–16 / confirm 17–19.
* Formats: **delayed** (graph + start → K latents → any number of forked queries)
  and **upfront** (graph + start + target → K latents → answer).

## 3. Interface and intervention

* Intervention site: thought h_k, the vector that becomes the input of recurrent
  position k+1. KV entries at positions ≤ k are **kept**; positions k+1..K and all
  queries are **recomputed**. Nothing else (no earlier cache entry) is modified.
* Donor for branch i of recipient g: `move_branch(g, i)`: the one edge token
  (s → root_i) becomes (s' → root_i). Same positions, ids, and token count.
* Replacement: h' = h_rec + U Uᵀ (h_don − h_rec).
  * `full`: U = I.
  * `probe`: U = orthonormal basis of the reach-probe read-out directions (W/σ) of the
    6 node ids of branch i, probe fitted on discovery graphs at step k.
  * `random`: U = random orthonormal basis, rank 6 (matched to `probe`).
  * `normrand`: isotropic random direction with the norm of the `probe` projection.
* Directions: `remove` (recipient has branch i attached; donor detaches it) and
  `insert` (recipient = detached version; donor reattaches).
* Alternative donor: `graft` (root_i re-parented to a random node of a decoy branch).
* Negative controls: `identity` (donor = recipient; must be exactly 0),
  `same_status` (a decoy branch re-grafted onto another decoy branch; one token
  changes, no reachability changes).
* Effect matrix: from the *same saved recipient prefix*, each of the W reachable
  branches is edited in turn; every present node is queried as an independent fork.

## 4. Estimands (probability scale)

For recipient g, edited branch i, queried node v: Δ(v) = P_base(reach v) − P_patched(reach v).

* **Targeted effect** (primary): mean over nodes v of branch i of sign·Δ(v),
  sign = +1 for `remove`, −1 for `insert` → movement toward the graph-defined
  counterfactual answer.
* **Collateral (attached)**: mean |Δ(v)| over nodes of the other reachable branches.
* **Collateral (decoy)**: mean |Δ(v)| over decoy-branch nodes and s'.
* Depth-resolved targeted effect by within-branch depth j (distance j+1 from s).
* Decodability: reach-probe AUC, reachable-branch nodes vs shape-matched decoy nodes,
  averaged within within-branch-depth strata; per-branch AUC vs same-graph decoys.
* Upfront format: Δ on the answer when the target's own branch is edited (diagonal)
  vs other branches (off-diagonal).

Sampling unit: graph (independent family). 95% percentile bootstrap over graphs,
2000 resamples. Seeds reported separately from within-checkpoint uncertainty.

## 5. Discovery → frozen choices

* Site k*: argmax over k ∈ {0..K−1} of (targeted − collateral_attached) for `full`,
  `remove`, W=4, 300 discovery graphs. ⟨discovery⟩
* Probe family, rank (6 = nodes per branch), donor rules, controls: fixed a priori above.
* Confirmation: n graphs per width ⟨discovery: set from pilot variance⟩, widths 1..8
  (primary contrast 2 vs 4 vs 8).

## 6. Decision criteria (provisional project thresholds, from the proposal)

* Competence: ≥ 85% balanced accuracy on the delayed format (confirm split).
* Latent use: accuracy with K'=K exceeds (i) K'=0 at inference and (ii) the separately
  trained K=0 and pause controls, on deep queries (distance ≥ 3).
* **Selective usable alternative at width W**: targeted ≥ 0.10 with CI lower bound > 0,
  and mean collateral_attached ≤ 0.03 (CI upper bound reported).
* Interference: slope of (targeted − collateral) against W, and the full
  decodability-vs-selectivity curves; no single "capacity N" claim.
* A negligible effect is claimed only if the CI upper bound < 0.03.
