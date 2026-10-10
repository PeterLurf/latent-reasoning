# Prior check for STEP-AUX-HOLD: K sampled from n+1 .. n+4, with passes t > n trained to *hold* the final value (fixed-point extra passes) (2026-10-10)

Literature session, for Theory; Audit copied. IDs verified via arXiv abstract pages (2026-10-10).

## 1. Supervising steps past the solution to be stable or idempotent
- **Deep supervision with the final answer at every refinement step:** TRM 2510.04871 / HRM 2506.21734.
  - Each outer step is supervised with the *same final answer*, with the carried state detached between steps. Once correct, further steps are
    trained to stay correct: an implicit "hold".
  - HRM adds ACT-style Q-learned halting.
  - **The closest *training-target* precedent to HOLD.**
- **Architectural hold after halting:** Universal Transformers 1807.03819 with ACT copy a halted position's state to later steps (a hard hold,
  not a learned one). ACT 1603.08983 / PonderNet 2107.05407 weight outputs by halting probability. They don't add a stability loss, and **both
  collapse to one loop in RecurTrace 2609.03379's controlled test.**
- **Fixed-point training:**
  - DEQ 1909.01377 (solve for the equilibrium, with implicit differentiation);
  - Attractor Models 2605.12466 (fixed-point refinement on LMs / puzzles);
  - implicit / fixed-point RNNs 2502.07827, 2503.10799.
  - These make "extra iterations = staying at the fixed point" true by construction.
- **Path independence, Anil et al. 2211.09961 (verified):**
  - equilibrium / depth-recurrent models generalise "upwards" (harder instances with more iterations), and that **correlates strongly with path
    independence** (converging to the same steady state regardless of initialisation);
  - interventions **promoting path independence improve generalisation** to harder instances, and penalising it degrades it;
  - per-example, path independence on OOD samples correlates with accuracy.
  - **That's the theory-level support for HOLD:** a hold target pushes towards a convergent, initialisation-insensitive endpoint.
- **How the loop papers handle extra iterations:**
  - **Fan 2409.15647:** loops are trained with T(n) and *not* explicitly trained to converge, but **"the looped model learns to converge for a certain
    number of steps after solving"** on Addition, Copy, Multiplication and Unique Set (not on Parity or Binary Sum). A confidence-based stopping
    criterion picks the step at test.
  - **Kohli 2604.07822:** R ~ clip(Poisson(λ), 2, 8). No explicit hold.
  - **Kuo 2606.29983:** stochastic loop counts "stabilise predictions across inference-time loop counts". It does that by randomising, not by a hold target.

## 2. Is overthinking past trained depth documented as fixed by such targets?
- **Bansal et al. 2202.05826:** names "overthinking" (behaviour degenerates with many iterations). It's fixed by **recall** (keeping the input) plus
  **progressive training** (random iteration counts, with the loss after continuing from an intermediate state), "pushing the model to learn
  behaviours that can be repeated indefinitely".
  - That isn't a hold target, but it's the closest *documented fix* for overthinking.
- **Anil 2211.09961:** promoting path independence (convergence regardless of initialisation) improves upward generalisation.
- **Think-at-Hard 2511.08577:** "latent overthinking" (correct after pass 1, revised into errors), fixed by *not iterating* on easy tokens. A
  selective-iteration fix.
- **Guo 2609.16665:** a damped ¼ step fixes overshoot.
- **Not found:** an explicit **"hold the final value" aux target on passes past the solution** in a latent-reasoning model. The closest are TRM / HRM's
  final-answer deep supervision (a hold on the *answer*, not an aux *state* target) and DEQ fixed points.
- **HOLD as an aux *step-decoder* target is new as scoped.**

## 3. Per-step-supervised loops reaching length generalisation beyond the trained step count
- **The recurrence retrofit 2608.11233:** step-per-loop supervision, then outcome annealing. About **1.5× the supervised depth** (70% through depth 18);
  the scratchpad baseline collapses beyond its horizon.
- **CLRS 2205.15659 (hint supervision):** per-step algorithm "hints" on a recurrent GNN processor, tested OOD on larger graphs (16 → 64 nodes). Hints help
  some algorithms and not others.
  - 2211.00692 analyses OOD failures (input distribution shift, uninformative validation).
- **NPI 1511.06279:** execution-trace (per-step) supervision of a recurrent core with program memory. Length generalisation shown for sorting. Pointer-
  and program-based, not latent.
- **Text-step methods with length generalisation on arithmetic:** Turing Programs 2407.03310 (2×); Cho 2410.15787 scratchpad (2–3×).
- **Not found:** a per-step *latent* aux on nested or multi-digit arithmetic that extrapolates beyond the trained step count.
- **For §175's null at n 9–12:** step binding learns the training schedule (LIT_STEP_AUX's flag). HOLD addresses only the *extra-pass* instability, not the
  *new-depth* steps 9–12, which have no step targets in training.

## Notes for the draft
1. **HOLD needs a definition for the aux target at t > n.** "The final value" in Feng's CoT format means the last step's expression *is* the value.
   Make it the identical string as step n, so the decoder learns idempotence, not a new format.
2. **Pair HOLD with Kuo-style K randomisation** (already in the design: n+1 .. n+4) and **Bansal-style progressive passes** (start some chains from a
   mid-chain memory), so "repeatable from any state" is trained, not only "stop after n".
3. **Measure path independence (Anil):** run test chains from two different initial memories (zero vs a noisy or passed state, as §166) and check whether
   they converge to the same final memory and answer. Anil found that this predicts OOD success.
4. **Expected outcome:** HOLD should flatten the accuracy-vs-extra-K curve (fix overthinking). On its own it is *not* expected to give n 9–12. Pre-state
   both.
