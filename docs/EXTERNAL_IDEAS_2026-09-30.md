# External theory input, 2026-09-30: "Replace aging thoughts with reusable interfaces"

Provided by the user from a parallel research agent that had no access to this repo; that agent was briefed only with a
prose summary of our results. Its proposals are speculative and unvalidated, and it says so itself. Each still needs
Literature's novelty check and a THEORY registration before anything runs. Audit's triage is at the end.

## Five directions (summary)

1. **Boundary-response elimination.** Represent a partly solved region by its response table over ≤ 4 boundary ports
   (≤ 64 numbers). A learned merger composes adjacent regions; elimination-order consistency is enforced. The aim is
   that summaries built through different histories become interchangeable, so the "age" of a computation stops
   mattering.
   - Targets: C1, C2, C4, C6.
   - Test: 64-node path skeleton with a fixed balanced contraction tree, K = 55/59/61/62 merges; preregistered gains, ≥ 97 %
     agreement across elimination orders, and history-matched transplant ≤ 2 points.
   - Falsifier: operationally equivalent summaries built through different histories still trigger different deadlines.
   - Nearest prior: NBFNet, neural graph coarsening.
2. **Conserved continuations.** The persistent state is a set of suspended calls (local question, inputs, return
   slot). A worker learns refine/call/return/write; each return channel must be consumed exactly once.
   - Targets: C1, C2, C4, C6, with a route to C5 and C7.
   - Test: H = 32 worker calls, evaluated at 32/64/128, against an unconstrained stack; return-transplant audit.
   - Nearest prior: Recursive NPI, AlphaNPI.
3. **Commit gates that must earn their causal usefulness.** A dictionary of proposed state changes with stochastic commit
   gates; changes whose paired-suffix terminal utility is neutral are penalised.
   - Targets: C2, C4, C6.
   - **Cheap kill test on existing models:** replace the diagnosed fall-back mutation by identity on shallow examples and
     measure whether terminal loss changes. If the fall-back is reward-neutral, training could penalise it.
   - Nearest prior: COMA, PonderNet.
4. **Predictions of named experiments as the whole state.** A public, controlled-language state of future-test
   probabilities; no private memory.
   - Targets: C1, C4, C7.
   - Test: 4 Boolean registers with NOT/SWAP/CNOT; 15 parity probabilities; train lengths 1–6, test 7–24; donor-state
     substitution and edits.
   - Nearest prior: PSRs, PSRNNs, Predictive-State Decoders. Not C2 (longer programs ≠ more thinking).
5. **Learned operator plus a stationary Krylov solver.** A front end emits a sparse linear system, and GMRES iterations
   are the latent computation. A lower-priority control: does removing the free nonlinear transition remove the deadline?
   - Nearest prior: Neural Krylov Iteration, DEQ.

## Unifying hypothesis and a new prediction

Our positive results (restarts, depth-bounded segments, distillation) may work by transporting progress into an
**entry-state normal form**, a state that keeps the residual problem but drops how it was reached.
- **Prediction:** after repair, states reached through different numbers and arrangements of segments are interchangeable
  for future behaviour once their residual semantics match, even if the step count is still decodable.
- **Competing account:** re-anchored teachers just supply better targets.
- **Separating test:** hold targets, input, compute and residual problem fixed, then compare canonical transport, raw
  latent transplant and an age-only reset.

## Benchmark correction

2608.11233 (the Qwen-0.5B looped retrofit) reports ~1.44–1.50× extrapolation. It uses intermediate-state supervision
before outcome-only annealing, and it disclaims matched inference compute and training lineage. It is a pretrained
baseline, not evidence that C3 and C4 are jointly solved.

## Audit triage (2026-09-30)

Cheapest decisive items first, all on existing models or CPU:
- **(a) The kill test from direction 3:** is the fall-back mutation reward-neutral?
- **(b) The interchangeability test from the unifying hypothesis:** on existing segmented (§45/§48) and distilled (Pilot L)
  toy models.

Larger new substrates come next, after Literature's novelty checks: direction 1 (most novel, but scaffolded; a mechanism
test, not a benchmark win), then 2. Direction 4 is a narrow C7 substrate. Direction 5 is a control.
