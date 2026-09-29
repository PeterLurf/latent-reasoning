# Protocol 2 — why branch edits weaken with width

Follow-up to `docs/PROTOCOL.md`, executing the plan in `docs/NEXT_STEPS.md` (the user's
handoff document, 24 Sep 2026). Hypotheses and predictions below were written **before**
any fresh-confirmation graph was generated. Numerical choices marked ⟨discovery⟩ are frozen in
`docs/protocol2_frozen.json` from discovery-split runs only.

## Data and units

* Models: the three recurrent seeds (s0 = `runs/r3_pc_gated_gpt2`, s1, s2) and matched pause
  seeds (s0 now; s1, s2 when trained) plus no-latent seeds for competence.
* Development: discovery-split graphs, sampling seed 31337.
* **Fresh confirmation:** confirm-split graphs, sampling seed 271828, excluding every family
  drawn by any earlier confirm-split sampler (`followup_run.used_confirm_families`).
* Widths 2, 4, 8 (W = 1 removal / insertion involves an untrained W = 0 state and is reported
  only as a distribution-shift test). Insertion-only at W = 8 creates W = 9 donors: flagged.
* Sampling unit: graph. Per-graph means over edits and nodes; percentile bootstrap over graphs
  within a seed; seeds shown separately (3 seeds is limited evidence about training runs).
* Every probability effect is also reported as a logit effect.

## Interfaces (positions: edges 0–59, START 60, LAT_j 60+j)

* **T (thought):** replace h_k before it feeds LAT_{k+1}; cache 0..60+k retained.
* **L (latent cache):** overwrite cached K/V of START..60+k with the donor's.
* **G (graph cache):** overwrite cached K/V of edge tokens 0..59 with the donor's
  (this changes the effective input problem; interpreted only with timing/pathway evidence).
* **clamp:** donor thought fed at every step k..K−1.
* Pause models have no T interface; they are tested through L and G only.

## Hypotheses

* **H-dil (dilution / gain).** Each branch's contribution to the state scales like g(W) with g
  decreasing; edits weaken because the per-branch signal is smaller, not because branches
  overlap. Rescaling or restoring gain should restore effects.
* **H-count (active-count normalisation).** Part of the collateral arises because an edit changes
  the number of active branches (a shared normaliser). Width-preserving swaps remove it.
* **H-int (interference / response alignment).** Branch response directions overlap, more so at
  high W; selective control costs more and collateral per unit target rises.
* **H-rec (recovery).** Thought edits are undone by recomputation from retained information
  (latent cache and/or graph tokens).
* **H-sem (branch-state semantics).** The edit transfers a branch-state component that downstream
  computation continues *with the recipient graph*.

## Pre-registered predictions

| Test | H-dil | H-count | H-int | H-rec | H-sem |
|---|---|---|---|---|---|
| Exp 1 dose (targeted vs dose, W=8) | rises with dose to ≥ W=2 level | – | saturates below | saturates below | – |
| Exp 1 recovery ratio (targeted / clean) | ~constant across W | – | falls with W | ≪ 1 at all W | – |
| Exp 2 swap vs removal collateral | – | swap < removal, signed ≈ 0 | swap ≈ removal | – | – |
| Continuation: ΔC_rec − ΔC_don | – | – | – | – | > 0 |
| Cache: T vs TL / TG, margin trace | – | – | – | T dips then returns; blocking the route stops return, TL or TG ≫ T | – |
| Attention boost of branch j | others' P ↓ | others' P ↓ | – | – | – |
| Gain restoration at W = 4, 8 | targeted ↑ toward W = 2 | – | little change | – | – |
| Controllability geometry | – | – | ρ = ‖p‖/‖a‖ ↓ with W; improves held-out prediction beyond W, norm, confidence | – | – |

Already seen in the discovery smoke test (seed 0, W = 4, 40 graphs), recorded here so it is not
mistaken for a confirmation: full-vector targeted effect saturates (0.056 at dose 0.5, 0.060 at
1.0, 0.057 at 1.5) against a clean counterfactual gap of 0.73.

## Primary outcomes

1. Exp 1: targeted and collateral (mean/sum/worst/signed) by dose and width; recovery ratio.
2. Exp 2: collateral (excluding A and B) for swap vs removal vs insertion at matched dose.
3. Continuation: follow index ΔP(C_rec) − ΔP(C_don) for full and pattern edits (> 0 supports
   H-sem), against the wrong-branch donor.
4. Cache: fraction of the clean counterfactual effect recovered by T, L, G, TL, TG, clamp;
   per-step branch margin; T×L interaction on probability *and* logit scales.
5. Attention: ΔP on other reachable branches when one branch's edges are boosted.
6. Gain restoration: targeted (restored − plain) at W = 4, 8.
7. Controllability: held-out R² for predicting per-edit collateral and targeted effect —
   baseline predictors (W, donor norm, clean confidence) vs + geometry (gain, ρ, linear prediction).

## Decision rules (from NEXT_STEPS.md)

* Dilution is pursued if modest rescaling largely restores selectivity at high width.
* Count explanation gains support if swap collateral < removal collateral after dose control.
* Recovery requires a route edit that prevents the return and preserves the intended final effect,
  with unrelated-route (L_wrong) and identity controls ≈ 0.
* Semantic validity requires the recipient continuation to dominate donor endpoints at > 1 branch
  depth (j = 2, 3) in held-out families.
* Geometry is kept only if it improves held-out prediction beyond the simple predictors.
