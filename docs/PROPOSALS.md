# Improving latent reasoning — proposals from the Latent project's mechanism results and the 2025–26 field

Theory stream, 2026-09-29 (overnight). Companion to `docs/THEORY.md` (the evidence) and `docs/LITERATURE.md` /
`docs/LITERATURE_NOTES.md` (the project's systematic review). Every proposal says what it fixes, why we think so, and
how it is tested; the ones already running are marked **[running]**.

---

## 0. One-paragraph summary

Continuous-thought reasoners (Coconut and its successors) generalise beyond their trained depth only in some seeds.
In our 2-layer model we found *why*, causally, in 59 trained models: (1) the thought is a **pointer** that must move one
hop per step, and most failing seeds move it by **retrieving earlier thoughts from the KV cache on a step-indexed
schedule**, a schedule that is undefined past the trained budget (and whose last step is never trained, because the final
thought is never consumed); (2) even seeds that compute the pointer locally lose ~20 % of its amplitude per hop — a
**loop gain below 1** that training never needs to fix. Blocking retrieval after the trained budget (zero effect
in-distribution) raises 7-hop success from 24 to 41 of 59 models. These two mechanisms map onto concrete changes that
the current state of the art does not make: a **Markovian latent step** (thoughts may not read earlier thoughts), **no
dangling thoughts** (every latent output is trained as an input), **gain-restoring pointer updates**, **tight
compute contracts**, and **in-distribution-null causal tests** as a standard evaluation.

| # | Proposal | Fixes | Evidence so far | Test |
|---|---|---|---|---|
| A1 | Markovian latent step | step-bound retrieval schedules | inference-time block rescues 17/23 retrieval failures | Pilot A **[running]** |
| A2 | Gain-restoring (digital) pointer | per-hop attenuation, sink capture | reach tracks post-K gain (ρ 0.77) | Pilot B |
| A3 | Non-competitive frontier read | width dilution, drain-to-sink | per-edge budget law (Parts II–IV) | Pilot B/C |
| A4 | Two-stream state (pointer + accumulator) | forces retrieval to rebuild the visited set | exp3: retrieval rebuilds pointers from old reads | Pilot C |
| A5 | Convergence-based halting | overthinking, wasted steps | pointer "dies"/cycles after the graph ends | Pilot C |
| T1 | No dangling thought (overrun / progressive / deep supervision) | untrained final transition | 18/34 failures revert exactly at h_K | Pilot A arm OV **[running]** |
| T2 | Unity compute contract (budget-matched curriculum) | dwell → schedule | Zhang et al. budget law; dwell in all our models | Pilot A arm MB **[running]** |
| T3 | Latent-KV dropout / position scrambling | implicit step counting | window test (exp1) partly unsticks cycles | Pilot C |
| M1–M5 | Evaluation methodology (beyond-budget tests, attribution, null-in-distribution interventions, seed triage, pre-registration) | overclaiming from signatures | this project | — |
| Th1–Th4 | Theory: support-bound recurrence, budget-law schedules, loop-gain reach, width-limited amplitude | — | — | derivations + toy sims |

### Update 2026-09-29 ~06:50 EDT — what the night changed (details in `docs/THEORY.md` §9–13)

1. **Blocking retrieval architecturally is not enough.** Interim Pilot A: Markov models extrapolate no more often than
   Coconut (1/3 vs 3/6 competent). Without retrieval, failing Markov models **restart the sweep** when their pointer has
   faded (pa_m_s713: `1 2 3 | 1 2 3 4 | 1 2 3 4`). Retrieval was one implementation of a step-indexed schedule; the
   deeper cause is **slack** — more latent steps than required hops lets any schedule that visits each depth once fit the
   loss. Proposal A1 is demoted from "the fix" to "one ingredient"; the unity contract (T2) and a new proposal T5 are
   promoted.
2. **New T5 — anytime (minimum-progress) supervision [running: Pilot B].** Supervise the answer after every latent step
   for nodes within k + 1 hops and for all unreachable nodes, masking deeper nodes (never penalise early arrival). This
   removes slack without imposing a counter, and gives a mechanistic reason why per-step supervision (SIM-CoT, PMPS,
   TRM/DIS deep supervision) stabilises latent reasoning with more latent steps. Early signal for the same idea via the
   unity contract: the first fixed Markov + budget-matched run extrapolates to 0.77 at 8–10 hops (one seed).
3. **The circuit is chosen in curriculum stage 3**, when the first latent advance is needed: all models have a static
   pointer at the end of stage 2; failures acquire the advance as a lagged copy, extrapolators as a local operator.
   Retrieval share ≥ 0.25 at the end of stage 3: 0/12 extrapolate (all 34 checkpointed models) — a cheap early
   negative marker for seed triage (T4).
4. **Mechanism 2 has a direct signature: the local map's transfer curve.** Scaling the pointer in the input thought
   and reading the output pointer: extrapolators have a **restoring** map (weak pointers collapse, strong ones clamp to a
   stable non-zero fixed point, e.g. p30_s105 output ≈ 3.7 for any input ≥ 3.6); fading failures have a **linear**
   sub-unity map (p30_s126, gain ≈ 0.7, no fixed point). This is exactly the analog vs digital distinction of Th3 and the
   toy simulation, now measured in trained networks — the design target for A2 is "a stable non-zero fixed point".
5. **Training-free repair for trained models:** mask latent→latent attention past the trained budget and amplify the
   thought mildly (α ≈ 1.3): 7-hop success 24 → **46 of 59** seeds, 8–10-hop 8 → 27, with zero in-distribution cost.
6. **Broadening to LLM-scale latent reasoners (in progress):** SIM-CoT-on-CODI GPT-2 reproduced at 0.393 on GSM8K
   (published ≈ 0.44); its six latents add +7 points overall but nothing on 4–7-step problems (0.26 vs 0.24 at 4 steps).

---

## 1. Where the field is (Sept 2026)

**Sequential continuous thoughts.** Coconut (Hao et al. 2024) feeds the last hidden state back as the next input
embedding; each thought attends to the whole prefix *and all earlier thoughts*. Successors add supervision to fight
collapse: CODI (self-distillation, 2502.21074), **SIM-CoT** (ICLR 2026; per-thought auxiliary decoder, +2.1 over explicit
CoT on GPT-2, stable at 8–16 latent tokens where others collapse), **PMPS** (2609.09928; prototype-mediated process
supervision, +2.5 over SIM-CoT, >3.9 over CoT-SFT at <50 % of tokens), supervised Thinking States (2602.08332; better
length generalisation than CoT on state tracking).

**Parallel / looped latents.** PCCoT (EMNLP 2025; Jacobi refinement of a fixed latent block), KaVa (KV-cache
distillation), **LOTUS** (2606.31779; looped padded transformer, K latent blocks refined in parallel for R iterations,
first latent method to match explicit CoT at 3B). Depth recurrence at scale: Huginn (2502.05171), **Ouro/LoopLM**
(2510.25741; 1.4–2.6B looped models matching 12B), HRM/**TRM** (2510.04871; 7M-parameter recursion with deep
supervision, 45 % ARC-AGI-1), attractor models (2605.12466; fixed-point refinement, 91 % Sudoku-Extreme with 27M).

**RL on latents.** HRPO, LEPO (2604.17892), Latent-GRPO (2604.27998), TARPO (2606.05859), SWITCH (2606.13106) — mostly
about injecting stochasticity for exploration and switching between latent and explicit modes.

**Analyses that bear on generalisation.** Latent tokens are often bypassed (Aswal et al. 2606.12689; Illusion of
Superposition 2604.06374; our ProsQA addenda D/G/J/N). Recurrent models pick different algorithms by seed; which seeds
extrapolate is predicted by in-distribution frontier completion (Zhang et al. 2607.20594), whose **budget law** — frontier
speed v ≈ n_train/T_train, "the demanded minimum is met, never exceeded" — matches the slow, dwelling pointers we see.
Randomised loop counts stabilise extrapolation in looped transformers (Kuo et al. 2606.29983); progressive loss prevents
"behaviours specific to iteration number" (Bansal et al. 2022); path independence predicts upward generalisation (Anil
et al. 2022); hard attention + discrete states give provable size generalisation (Rodionov & Prokhorenkova, ICML 2025);
latent CoT trades execution precision against exploration via decisional certainty (2602.01148); overthinking degrades
deep recurrence (Kohli et al. 2604.07822).

**What is missing.** (a) Almost all methods are evaluated in-distribution; extrapolation past the trained number of
latent steps or reasoning depth is rarely measured, and never per seed. (b) Every Coconut-family method lets thoughts
attend to all earlier thoughts; nobody has asked what that channel is used for. (c) The final latent of training is
never trained as an input — a "dangling" state — in Coconut, CODI, SIM-CoT and PMPS alike. (d) No method controls the
amplitude (gain) of the latent state across steps; LayerNorm normalises the whole vector, not the part that carries the
computation (Buehlmaier 2607.10681 treats LayerNorm as gain control for stability, not for signal restoration).
(e) Mechanistic claims rest mostly on correlational signatures, which Aswal et al. show can be reproduced by controls.

## 2. What our results add (details: `docs/THEORY.md`)

* **The thought is a pointer** over graph nodes (sharp, one depth at a time), advancing one hop per step once the
  curriculum's slack is used up (§1).
* **Mechanism 1 — retrieval schedules.** In most failing seeds, layer 2 writes the pointer by copying from an *earlier
  latent position*, with a lag that depends on the step index (implicit clock via the growing cache). Past the trained
  budget the schedule copies a stale pointer (h_K = h_{K−2}, period-2 cycles). Extrapolators compute the pointer
  locally (MLPs of the current step). Retrieval share at the untrained last thought: 0.56 in failures vs 0.08 in
  extrapolators. **Causal:** blocking latent→latent attention in layer 2 from step K on (no effect on any trained
  answer) rescues 17/23 retrieval-type failures and 7-hop success goes 24 → 41 of 59; it does not help local-type
  failures (2/11), as predicted.
* **Mechanism 2 — sub-unity loop gain.** Each advance multiplies the pointer by ~0.8 (dwelling does not); frontier
  attention mass is not the bottleneck; when the pointer falls below the competition the sweep head drains into START,
  self and earlier thoughts and the front dies. Reach after the block tracks the post-K gain (ρ = 0.77).
* **Why training produces both.** Dwell is required early (6 steps for 1–2 hops in curriculum stage 1); retrieval is
  the cheapest way to dwell, and it ties the dynamics to the step count. Nothing constrains the last thought or any step
  past K. Nothing pushes the loop gain to 1, and noise on shape-matched decoys pushes it below.

## 3. Architectural proposals

### A1 Markovian latent step **[running: Pilot A, 8 + 8 seeds, plus 4-seed variants]**
The latent position attends to the problem (prefix) and itself, never to earlier latent positions; the answer/readout
positions still read every latent position's keys/values. All state must travel in the thought, so the transition is
the same function at every step and cannot index a schedule by the count of earlier thoughts.
* *Why:* removes Mechanism 1 by construction; the inference-time version already rescues most retrieval failures.
* *Relatives:* the Markovian Thinker (2510.06557; bounded carried state for explicit CoT), input-injected looped
  transformers (Huginn, TRM, LOTUS re-read the input each loop), path independence (Anil et al.). None applies the
  restriction to Coconut-style sequential thoughts or motivates it mechanistically.
* *Cost:* none at inference (the KV of earlier thoughts is still kept for the readout; the latent query is cheaper).
* *Variants:* (i) **lag-1 only** (a thought may read its immediate predecessor — an RNN-like carry); (ii) **gated
  retrieval** with an L1 penalty on the gate (learn whether history is needed); (iii) **layer-selective** (Markov in
  the upper layers only; our retrieval lived in layer 2 — arm M2 of Pilot A).
* *Prediction (PA1):* extrapolating share among competent runs ≥ 0.70 vs ~0.50 for Coconut.
* *For LLM-scale latent CoT:* test Markovian thoughts in CODI/SIM-CoT on problems needing more latent steps than
  trained (e.g. ProsQA extended to 7–12 hops, our addendum-P data), and check whether GPT-2 thoughts retrieve earlier
  thoughts (the component attribution of §5 M2 applies unchanged).

### A2 Gain-restoring ("digital") pointer — Pilot B
After each step, pass the entity-identity part of the thought through a restoring nonlinearity: score every entity id
(e.g. against the shared id-embedding table), apply a steep sigmoid with a learned threshold (set membership) or a
temperature-controlled softmax (single pointer), and re-embed. Amplitudes are restored to a fixed level each hop, as
digital logic restores signal levels; small noise (decoy leakage) is squashed below threshold rather than propagated.
* *Why:* targets Mechanism 2 — reach ≈ log(a₀/θ)/log(1/g) diverges as g → 1 with noise still suppressed (Th3).
* *Relatives:* discrete neural algorithmic reasoning (hard attention + finite states → provable size generalisation),
  CoT2 / soft-token thoughts (thoughts as mixtures of token embeddings), modern Hopfield cleanup.
* *Risk:* a steep threshold kills superposition over many frontier nodes; use a per-id sigmoid (non-competitive), not a
  softmax, when the frontier holds several nodes.

### A3 Non-competitive frontier read — Pilot B/C
Replace the softmax in the frontier-reading head(s) by sigmoid attention (or stick-breaking / scalable softmax with
log-n scaling) so the mass on one matching edge does not fall when more edges match or when latent tokens accumulate.
* *Why:* the per-frontier-edge budget law (Findings 1, 3) makes the starting amplitude ∝ 1/width; the death mode is the
  sweep head's attention draining into START/self/earlier thoughts once the pointer weakens.
* *Relatives:* Veličković et al. 2410.01104 (softmax is not enough for sharp size generalisation), Tan et al. ICLR 2025
  (stick-breaking), Nakanishi 2501.19399 (SSMax).

### A4 Two-stream latent state (pointer + accumulator) — Pilot C
Carry two vectors per step: a **pointer** (current frontier; Markov update from the edges it points at) and an
**accumulator** (visited set; additive update, read by the answer). Our failing models used retrieval partly to
reconstitute old reads; a dedicated accumulator removes that need, and the pointer can be restored (A2) without
destroying what was found. TRM's (z, y) split and HRM's two timescales are the closest relatives; neither ties the split
to a pointer/visited-set semantics.

### A5 Convergence-based halting — Pilot C
Stop when the pointer has no successors (its restored amplitude falls below threshold) or when the accumulator stops
changing, instead of a fixed K. Zhang et al.'s T⋆ = ⌈n/v̂⌉ rule and PonderNet/ACT are the relatives; with A1–A2 the
stopping signal is directly readable from the state.

## 4. Training proposals

* **T1 No dangling thought [running: Pilot A arm OV].** Train with latent budgets drawn from K..K+m (overrun) or with a
  progressive loss (start from a random intermediate state, Bansal et al. 2022), or supervise every latent output (deep
  supervision as in TRM / Deep Improvement Supervision 2511.16886). Then the final trained thought is also an input,
  and "one more step" is in-distribution. For Coconut/SIM-CoT/PMPS: append 1–2 extra latent steps with the same answer
  loss during training.
* **T2 Unity compute contract [running: Pilot A arm MB].** Match the latent budget to the reasoning depth at every
  curriculum stage (here K = depth + 1), so the frontier must move one hop per step and never needs to dwell (Zhang et
  al.'s budget law predicts v = n/T). In LLM latent CoT this argues against padding with idle latents and for budgets
  proportional to the number of reasoning steps.
* **T3 Latent-KV dropout / position scrambling.** Randomly hide earlier latent positions from the latent step during
  training (not from the readout), or insert random numbers of no-op latents, so no schedule keyed to the count of
  earlier thoughts can fit. A soft version of A1 that keeps the architecture unchanged.
* **T4 Seed triage by mechanism.** The outcome is set at full-depth arrival (PROTOCOL7 T3, AUC 0.93); measure
  retrieval share / P7 there and continue only the runs that will generalise. For expensive latent-reasoning training
  runs, a cheap mid-training diagnostic of *which algorithm* was learned is worth more than in-distribution accuracy
  (which does not separate them: T5).

## 5. Methodological proposals (evaluation and interpretability)

* **M1 Always test beyond the trained latent budget and beyond the training depth, per seed.** Report the distribution
  over ≥ 8 seeds, not the best run; seed-level algorithm selection is the norm (C1 of the literature audit).
* **M2 Exact component attribution of the latent state.** Decompose each thought into residual carry, attention by key
  source (problem tokens, each earlier latent by lag, self) and MLPs, with LayerNorms linearised at the actual input
  (`theory/exp3_attrib.py`). "Local vs retrieved" is a two-number summary that separated our extrapolators from
  failures (retrieval share 0.08 vs 0.56) and applies unchanged to GPT-2-scale latent reasoners.
* **M3 In-distribution-null interventions.** Choose interventions that provably leave every trained output unchanged
  (here: masking latent→latent attention only from the step whose output is never read in training) and measure their
  effect out of distribution. They isolate what the model does *beyond* its training support without confounding it
  with damage to what it was trained to do — a clean answer to "observable patterns are not explanations".
* **M4 Dangling-state audit.** For any latent/looped method, list which latent outputs are never consumed by the loss;
  those are the states whose behaviour extrapolation depends on and training never shaped.
* **M5 Pre-register predictions and test them on new seeds** (as PROTOCOL5–7 do); our own early predictors (end activity)
  looked strong retrospectively and weakened prospectively.

## 6. Theoretical proposals

* **Th1 Support-bound recurrence.** For a latent step h_k = F(h_{k−1}, H_{<k}, x) trained on k ≤ K, the loss identifies F
  only on the visited states and on history lengths < K; F's dependence on |H| (the count of earlier thoughts) is
  unconstrained beyond K, and the final output h_K is unconstrained altogether. Extrapolation therefore needs an
  inductive bias that makes F independent of the history length — Markovian steps (A1) are the minimal such bias;
  time-shift invariance of the latent dynamics is the property to formalise (a discrete analogue of path independence).
  Prediction: any architecture whose step reads a growing history can learn count-indexed schedules, and will do so
  whenever the compute contract has slack (Th2).
* **Th2 Budget law ⇒ schedules.** If the contract allows T steps for n hops, SGD finds v ≈ n/T (Zhang et al.); v < 1
  requires dwell; dwell is realisable either by holding state (Markov) or by lagged copies (retrieval). Lagged copies
  with count-dependent lags are cheaper to learn from a growing KV cache, so slack contracts select retrieval schedules.
  Testable: dwell time in stage 1 predicts retrieval share at full depth; unity contracts (T2) remove both.
* **Th3 Loop-gain reach.** With pointer amplitude a_d = a₀ g^d and a matching threshold θ set by the competition
  (log-sum-exp of the other keys' logits), reach = d_train + log(a_{d_train}/θ)/log(1/g). SGD sets g as low as the
  training depth allows because g ≥ 1 amplifies decoy noise; hence horizon ∝ training depth for graded (analog)
  updates, and no horizon for restoring (digital) updates (A2). This connects the bounded index-matching logit of Zhu
  et al. (ICLR 2026) and the certainty trade-off of 2602.01148 to a quantitative reach prediction.
  *Formal anchor:* saturated networks — squashing units driven to their extreme values — behave as discrete automata
  (Merrill et al. 2020, *A Formal Hierarchy of RNN Architectures*, arXiv 2004.08500; Weiss, Goldberg & Yahav 2018,
  arXiv 1805.04908, on which recurrences implement counters/automata in practice); a discrete state machine runs for any
  length, a graded (unsaturated) one accumulates error. Our transfer curves (update item 4) show extrapolating seeds'
  latent maps operating in the saturated regime (clamped output above threshold) and fading seeds' maps in the linear
  regime. Design target: a stable non-zero fixed point of the pointer map.
  *Toy check (`theory/sim_gain.py`, THEORY §8):* with moderate noise a graded update fitted on depth ≤ D reaches only
  D + 2–4 (7 / 8 / 10 for D = 3 / 5 / 8), a restoring update 40 of 40; with little noise the graded update also reaches
  far — so the horizon's strength depends on noise, and the exact training-depth horizon in our models is mainly
  Mechanism 1, with Mechanism 2 limiting reach beyond it.
* **Th4 Width-limited amplitude.** Under softmax reading the starting amplitude a₀ ∝ 1/(frontier size) (the per-edge
  budget law), so reach falls with width — a prediction for A3 (non-competitive reads remove it).

## 7. Order of work tonight and next

1. Pilot A (A1, T1, T2) on 4 Vast GPUs — running; score by the pre-registered rules in `docs/THEORY.md` §6.
2. Pilot B: A2 + A3 variants (implemented in `theory/arch.py` next), 4 seeds each, same recipe.
3. Toy simulations for Th2/Th3 (numpy): budget law and dwell, loop-gain reach vs training depth.
4. Transfer check: component attribution on the GPT-2 ProsQA checkpoints (G/J) — do GPT-2 thoughts retrieve from
   earlier thoughts?
