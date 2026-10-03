# Literature notes (2026-09-26) — where this project sits

> **Historical (2026-09-26).** Kept in place because other docs cite it by path. Current literature: `docs/LIT_INDEX.md` (start with LIT_FINDINGS_COMPILED, LIT_LATENT_FROM_DISCRETE, LIT_SEMI_DISCRETE).

## Closest prior work
* **Zhu, Hao, Hu, Jiao, Russell, Tian — Reasoning by Superposition (NeurIPS 2025, arXiv 2505.12514).** Proof that a
  2-layer transformer with D continuous thoughts solves graph reachability (parallel BFS as a superposition state);
  trained 2-layer model (d 768, ProsQA subset of 3–4 hops, node tokens) shows Layer-1 copy heads, Layer-2 frontier
  expansion. Their Table 1 (Layer-2 attention per edge group): Frontier 2.12 → 1.00 → 0.67 → 0.61 over steps 1–4,
  Optimal edges favoured (curriculum on the optimal path). No length-generalisation test, no causal interventions.
* **Zhu et al. — Emergence of Superposition: training dynamics (ICLR 2026, arXiv 2509.23365).** Under Coconut
  training the index-matching attention logit stays bounded (μ(t) → μ* < ∞), which keeps several traces at comparable
  weight (superposition); too large a logit would over-commit to one trace. Trained on 2 thoughts, the pattern
  "generalizes to c = 3 and c = 4" — "once superposition emerges in earlier stages, later stages can quickly reuse it
  to expand the frontier further."
* **Gozeten et al. — Continuous CoT enables parallel exploration (CoT2, ICLR 2026, arXiv 2505.23648).** Parallelism
  limited by embedding dimension.
* **Saparov et al. — Transformers Struggle to Learn to Search (ICLR 2025, arXiv 2412.04703).** Without CoT,
  transformers learn exponential path-merging over all vertices; learning gets harder with graph size.

## The sceptical line
* **Aswal, Palmeira Ferraz, Zhou, Peyrard — Observable Patterns Are Not Explanations (arXiv 2606.12689).** Coconut and
  CODI vs matched controls. On ProsQA ("graph-hopping") the BFS-like frontier signature is reproduced by a
  pause-as-thought control with the same curriculum and inverted by a curriculum-perturbed Coconut. Removing all
  thoughts at test time leaves Coconut's accuracy at 97.8% (vs 98.0%); causal tracing puts the thought positions at
  zero recoverability while prompt positions recover the output. Thought use is graded, not binary; "latent thoughts
  should be treated as hidden computation, not hidden explanation."
* **Backour — The Dynamics of Continuous Mixture Collapse (arXiv 2609.02049).** Softmax readout + autoregressive
  feedback either amplifies (one component wins) or contracts mixtures; transition near a derived threshold.

## Extrapolation with more steps
* **Fan et al. — Looped Transformers for Length Generalization (NeurIPS 2024, arXiv 2409.15647).** Adaptive loop count
  at inference length-generalises on RASP-L tasks.
* **Kuo, Chayti, Reizinger, Brendel, Jaggi — Stabilizing Extrapolation in Looped Transformers via Learned Stochastic
  Stopping (arXiv 2606.29983).** Instability traced to a spurious length–loop correlation; randomised loop counts
  sharply reduce OOD variance.
* **Kohli, Parthasarathy, Sun, Yao — Loop, Think, & Generalize (COLM 2026, arXiv 2604.07822).** Recurrent-depth
  transformers extrapolate 5-hop → 10-hop by adding recurrence; grokking-like phases (memorisation → in-distribution →
  systematic generalisation); "overthinking" (too many iterations) degrades predictions.

## Relation to this project
* Confirms independently: parallel BFS frontier in a trained 2-layer continuous-thought model; frontier attention
  decays over steps (their Table 1 is the raw quantity our end activity normalises).
* Not found elsewhere: per-branch causal edits in a shared thought and their decline with width; signed branch
  competition; the per-frontier-edge attention budget law (P25); width moving mid-depth nodes to query-time reading;
  step-locked causal blocking of the sweep; the commitment window (branching: outcome set by full-depth entry, not at
  stage 2); training precision as a candidate factor (addendum F, running).
* Corrected 2026-09-27 — has precedents:
  * **End activity (P26).** Zhang et al. 2607.20594 already show that in-distribution "head instruments" (convergence
    time τ(n,i), frontier completion) predict seed-level extrapolation (AUROC up to 1.00; three 16-seed populations;
    retrospective). Ours is prospective (threshold frozen before 12 new models) and in continuous-thought graph search,
    so present it as a prospective replication in a new model class. Also recheck that P20's "finish-early account"
    matches their claim: their abstract says extra test-time loops rescue late positions.
  * **Probe filter vs pattern.** Marks & Tegmark 2310.06824 found mass-mean directions more causal than
    logistic-regression directions at similar accuracy; Im & Li 2502.02716 find mean-of-differences the optimal steering
    direction. Ours is the stronger form (filters fully inert, 0.3% of donor energy), new for latent reasoning.
* Nearest new neighbours (Sept 2026): Gu et al. 2609.13747 (capacity of reasoning by superposition; trained models,
  number of alternatives varied, no per-alternative interventions); Ding et al. 2608.27265 (SCIT: value-cache suffixes
  carry CODI's latent computation — the protocol for our cache-recovery test); Lam-Muir 2608.03263 (pre-registered
  recurrent reasoner with difficulty-clocked arrival). Full list and novelty table: `docs/LITERATURE.md`.
* Tension: Kuo et al. find randomised loop counts stabilise extrapolation; our randomised latent budget (randk) did
  not make extrapolation reliable (P22 failed). Differences: their tasks tie loops to input length; ours tie steps to
  graph depth, and our randk runs mostly stalled or had closed sweeps.
* Prediction for our ProsQA transfer (from Aswal et al.): the answer can re-read the question, so T2 may well read
  "thoughts not needed" (K′ = 0 accuracy near K = 6). Our delayed format was built to prevent exactly this bypass.
