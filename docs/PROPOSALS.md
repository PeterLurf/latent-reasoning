# Improving latent reasoning — proposals from the Latent project's mechanism results and the 2025–26 field

Theory stream, 2026-09-29 (overnight). Companion to `docs/THEORY.md` (the evidence) and `docs/LITERATURE.md` /
`docs/LITERATURE_NOTES.md` (the project's systematic review). Every proposal says what it fixes, why we think so, and
how it is tested; the ones already running are marked **[running]**.

> **Timestamp correction (2026-09-29 12:59 EDT, machine clock).** Times in this file were my estimates, not read from the
> clock. Authoritative commit times: "Update ~06:50" → f54609d 07:00:56; "Th1/Th2 revised 07:30", "Th5 (07:30)",
> "Status and next steps (~07:30)" → 044c1b9 09:10:10; scope note → eb2e2f6 12:25:40. See `docs/THEORY.md`, "Timestamp
> correction", for the pre-registration timeline.

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

**Second round (2026-09-29 ~14:10): §8** — four conditions for extrapolation in latent steps (C1 identification, C2
homogeneity, C3 stability, C4 idempotence), the tightness account of random/matched/per-step budgets, and new methods
T6 (amplitude jitter), T7 (selection-window supervision) and architecture A6 (monotone latent set); Pilot F tests T6/T7.

### Update 2026-09-29 ~06:50 EDT — what the night changed (details in `docs/THEORY.md` §9–13)

1. **Blocking retrieval architecturally is not enough.** Interim Pilot A: Markov models extrapolate no more often than
   Coconut (1/3 vs 3/6 competent). Without retrieval, failing Markov models **restart the sweep** when their pointer has
   faded (pa_m_s713: `1 2 3 | 1 2 3 4 | 1 2 3 4`). Retrieval was one implementation of a step-indexed schedule; the
   deeper cause is **slack** — more latent steps than required hops lets any schedule that visits each depth once fit the
   loss. Proposal A1 is demoted from "the fix" to "one ingredient"; the unity contract (T2) and a new proposal T5 are
   promoted.
2. **New T5 — anytime (minimum-progress) supervision [running: Pilot B].** (Closest prior: RELAY 2502.08482,
   iteration-wise supervision for looped-transformer length generalisation; also PonderNet, CLRS hints, SIM-CoT, PMPS.
   Ours is the minimum-progress form — deeper nodes masked, early arrival never penalised — and its mechanism.) Supervise the answer after every latent step
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
6. **LLM-scale latent reasoners have the budget horizon but not the hidden fix.** Public GPT-2 CODI and SIM-CoT-CODI
   (6 trained latents, GSM8K): accuracy peaks at 6 latents (CODI 0.425) and never improves with more (CODI degrades to
   0.401 at 12; SIM-CoT flat); their latents attend to earlier latents increasingly (up to ~30 % of mass in layer 0) and
   making them Markov costs 7.5–12.5 points; masking earlier latents past the budget slightly *hurts* (−1 to −2 points) —
   unlike the toy, there is no local advance operator underneath. Implication: for LLM latent reasoners, extrapolation
   in the number of latent steps must be trained in (T2/T5), and "more latents at test time" is not a free lunch.
   Scope (addendum Q, experiments session): GPT-2 Coconut trained on 3–4-hop ProsQA is as accurate on 5–6 hops and
   gains nothing from 12 thoughts — it solves ProsQA in one pass, so there is no step schedule and no training-depth wall.
   The slack mechanism and its fixes concern models whose latent steps do sequential work; getting a pretrained model
   into that regime (problems deeper than one forward pass) comes first.

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
et al. 2022); hard attention + discrete states give provable size generalisation (Rodionov & Prokhorenkova, ICML 2025, 2402.11628);
latent CoT trades execution precision against exploration via decisional certainty (2602.01148); overthinking degrades
deep recurrence (Kohli et al. 2604.07822).

**Closest new work found overnight (read abstracts; full texts before any claim).** *Beyond the Training Horizon:
Mechanisms and Limits of Length Generalization in Looped Transformers* (arXiv 2609.33144, Sept 2026): two looped
configurations learn a "transferable progress cue" (one) or an "advancing computational frontier" (the other), and fail
past the training depth by degradation of the readout state and progress cues / declining reliability of state
propagation, with little self-correction — the looped-transformer analogue of our two mechanisms; no fix proposed, no
Coconut-style sequential thoughts, no history-retrieval mechanism, no in-distribution-null interventions. *Fast-slow
latent recurrence* (arXiv 2604.01577): a persistent latent state updated by weight-shared steps generalises train-short /
test-long, with state-dependent transitions and feature-wise nonlinear mixing as the essential ingredients — consistent
with our Markov + restoring-map results. Also relevant: neural-algorithmic-reasoning "hints" (per-step state
supervision; Bevilacqua et al. 2302.10258 Hint-ReLIC; Rodionov et al. 2306.13411 on reasoning *without* hints) — our
anytime supervision (T5) is weaker than hints: it supervises only the readout's lower bound per step, never the state.

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
  supervision as in TRM; Asadulaev et al. 2511.16886, *Latent Reasoning in TRMs is Secretly a Policy Improvement
  Operator* — cited earlier here under an older title, corrected 2026-09-29 at the Literature session's request). Then the final trained thought is also an input,
  and "one more step" is in-distribution. For Coconut/SIM-CoT/PMPS: append 1–2 extra latent steps with the same answer
  loss during training.
* **T2 Unity compute contract [running: Pilot A arm MB]** (not new as a training scheme: T = n is Fan et al.
  2409.15647, Zhang et al. 2607.20594 and MR-Loop in 2609.33144; ours is the reason it helps, Prop. 4). Match the latent budget to the reasoning depth at every
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

* **Th1 Support-bound recurrence (revised 07:30; an instance of underspecification, D'Amour et al. 2011.03395).** For a latent step h_k = F(h_{k−1}, H_{<k}, x) trained on k ≤ K,
  the loss identifies F only on the states visited in training; its extension past K (and the untrained output h_K) is
  chosen by inductive bias, not by data. Two ways to make the extension step-indexed: through the history (retrieval
  with count-dependent lags — the Coconut failures) or through the state itself (phase/fade-triggered resets — the
  Markov failures, pa_m_s713). Removing the history (A1) therefore does not by itself give time-homogeneity; what is
  needed is that every zero-loss F be an "advance" operator on the reachable states, which is a property of the
  **contract** (Th2), not of the attention pattern. Formal target: characterise the set of zero-loss transitions under
  a contract with slack σ = K − (hops needed) and show it contains step-indexed members iff σ > 0 (or the per-step loss
  leaves intermediate states unconstrained).
* **Th2 Slack selects schedules (revised 07:30).** If the contract allows T steps for n hops, SGD finds speed v ≈ n/T
  (Zhang et al.'s budget law); slack is spent on dwell, restarts or copies, all step-indexed. Observed: every model has a
  static pointer at the end of stage 2 (queries do the hops), and the first latent advance is acquired in stage 3 either
  as a local operator or as a lagged copy (THEORY §10); retrieval share ≥ 0.25 at the end of stage 3 → 0/12 extrapolate.
  Retrospective dose-response: extrapolating share 47 % at K = 6 vs 26 % at K = 8 (no step embedding). Tests running:
  Pilot C (K = 5 / 6 / 7), Pilot A arm MB (unity contract), Pilot B (anytime supervision removes slack step by step).
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
* **Th5 Two-factor account (07:30; THEORY §14).** A seed extrapolates iff (i) its untrained last step is computed
  locally (no retrieval schedule) and (ii) its local map is restoring (stable non-zero fixed point). On the 59 P30/L
  models: 18/23 extrapolate when both hold, 7/36 otherwise, 1/21 when h_K is retrieved. An out-of-sample check on the
  Pilot A models (rule frozen before they were analysed) **failed: 8/15 correct** — a single-step probe misses dwell steps and later restarts. The two mechanisms stand on causal evidence; as a *predictor* the one-step rule does not generalise, and P7 remains the reliable in-budget diagnostic.

## 7. Status and next steps (2026-09-29 ~07:30 EDT)

1. Pilot A (A1, T1, T2): 2 runs left (host 1, ~08:45); interim above. Score by `docs/THEORY.md` §6.
2. Pilot B (T5 anytime; A2 restore; combinations) and Pilot C (K dose-response) queued behind A on the same 4 hosts;
   expected ~11:00–15:00 EDT. The first anytime run learns stage 2 slowly (11 000 steps); PB3 (learnability) may bind.
3. Done tonight: toy gain simulation (Th3), developmental race (Th2), transfer curves + two-factor rule (Th5),
   training-free repair (24 → 46/59), GPT-2 CODI / SIM-CoT probes (horizon at the trained budget; no hidden fix).
4. Next: out-of-sample rule check (running); if Pilot B/C support the slack account, a confirmatory pre-registered
   study of anytime supervision and unity contracts (8+ seeds per arm, both formats), then the same training change on
   CODI/SIM-CoT GPT-2 with more latents than training steps (does the horizon move?).

## 8. Second-generation proposals (2026-09-29, after the theory of `docs/THEORY.md` §22)

A self-contained write-up with proof sketches: `docs/THEORY_FRAMEWORK.md`. The first round (§3–§6) fixed the failures we found. This round states what *any* latent reasoner needs in order to
extrapolate in the number of latent steps, and proposes methods for the conditions nobody addresses. Full statements
and proofs sketches: THEORY §22; test: THEORY §23 (Pilot F, pre-registered, running after Pilots D/E).

**Framework: four conditions.** C1 *identification* (no slack while the circuit is chosen), C2 *homogeneity* (no usable
step clock), C3 *stability* (a restoring map with a stable non-zero fixed point on the computation-carrying
coordinates), C4 *idempotence at completion* (extra steps leave the answer-relevant state invariant). Coconut, CODI,
SIM-CoT and PMPS meet none of C2–C4; RELAY-style or deep supervision and random loop counts meet C1 only up to the
training depth; attractor / DEQ models and discrete NAR meet C3–C4 by construction. No latent-CoT method targets C3 or C4.

| # | Proposal | Condition | Why it should work | Test / status |
|---|---|---|---|---|
| Th6 | **Tightness** (Prop. 4): random budgets, budget matching and anytime supervision are one mechanism — mass on budgets equal to the steps needed; they help iff that mass is present at the circuit-selection stage | C1 | T unobservable to the latent steps ⇒ a tight budget forces unit speed; non-tight random budgets leave dwell schedules at zero loss | retrospective consistent (randk, OV fail; MB2, anytime succeed); explains why Kuo et al.'s random loop counts helped and our randk did not |
| **T6** | **Amplitude jitter**: during training rescale each fed-back thought about the batch mean by s ~ LogU[1/r, r] | C3 | Prop. 5: under jitter no linear (graded) map can fit beyond depth ln(a₀/b₀)/(2 ln r); every fitting map is bistable/restoring; restoring ⇒ unbounded reach | Pilot F arms J, JAT (8 seeds each, paired with Pilot D) **[queued]** |
| **T7** | **Selection-window anytime**: anytime supervision only in the curriculum stages where the algorithm is chosen | C1 | Prop. 4 + §10 (circuit fixed at stage 3): tightness matters only there — a critical period for algorithm selection | Pilot F arms ATW vs ATL **[queued]** |
| **A6** | **Monotone latent set** ("latent set memory"): a membership vector over entities updated by max of non-competitive sigmoid scores, fed back into the thought | C3 + C4 | Prop. 6: members never fade, extra steps are no-ops after convergence, errors grow linearly (k·V·ε) not geometrically; fits every least-fixed-point task (reachability, ProsQA, Horn/Datalog inference) | design; toy test after Pilot F |
| **T8** | **Re-anchored latent episodes** (inference, training-free; after Continuous Thought Machines' maze re-application): run K latent steps, decode the frontier from the model's own attention, restart a fresh K-step episode there, max-combine | C2 + C3 | the step count never leaves the trained range; each restart begins from a crisp node embedding | **7-hop success 24 → 55 of 59; 14/16 block-resistant (fading) failures rescued** (THEORY §26, pre-registered RA1–RA4 all met); controls running |
| A8 | **Stationary history** — chunk-reset latent attention (a sliding window that restarts every L steps; CTM-style) | C2 | a finite periodic window cannot index steps beyond L; every test-time chunk is a copy of a trained chunk | Pilot J designed (§29) |
| T9 | **Periodic step clock** e((k−1) mod p) (Clockwork RNN / Phased LSTM; Literature session) | C2 | Prop. 9: under slack the model wants a clock; a periodic one has an identified continuation | Pilot I queued (§28) |
| T10 | **Repair by fine-tuning** a trained failure with anytime (± jitter) supervision | C1 (+ C3) | tests "slack removal repairs" vs a critical period; retrieval- vs fade-type split | Pilot H queued (§27) |
| Th7 | **Slack explains bypass** (Prop. 8): with one-pass capacity c ≥ hops needed, all K latents are unidentified and bypassing them is the shortest zero-loss solution | scope | predicts latent effect ≈ 0 for n ≤ c, > 0 for n > c; test-time latent scaling only for n > c *and* tight contracts | PROTOCOL8 design: measure c first, train above it (pending the user's go/no-go) |

**LLM-scale specifications (for a PROTOCOL8 second phase; GPT-2 Coconut / CODI).**
* *T6 jitter:* rescale each fed-back thought about the batch mean, h ← c + s(h − c), s ~ LogU[1/r, r], r = 1.5–2,
  p = 0.5, training only. Centring on the batch mean leaves GPT-2's massive-activation dimensions (nearly constant
  across examples) in c, so they are not scaled. Outcome: accuracy at L > K on problems above one-pass capacity.
* *A6 set memory:* a membership vector over the entity tokens of the question (first sub-word embedding of each entity
  name, E_ent), scores s = W_r h · E_entᵀ, m ← max(m, σ(β(s − τ))), next thought input += γ · W_w(m · E_ent), γ
  initialised to 0. ProsQA answers are membership questions, so the set is also directly readable.
* *T5 anytime at LLM scale:* `theory/anytime_loss.py:anytime_ce` — after each latent step, teacher-forced answer loss for
  examples whose required depth ≤ k + C (C = hops the readout adds; 0 if the answer follows the last thought directly).
* *Evaluation for all three:* accuracy by difficulty stratum × number of latents L = 0..K′ (non-decreasing in L is the
  anytime signature; GPT-2 CODI violates it — truncated chains score below no latents, THEORY §13 addendum).

**What would change the field if Pilot F holds.** (i) A one-line training change (T6) that turns graded latent maps into
restoring ones without state supervision or new parameters — directly applicable to CODI/SIM-CoT/Coconut at LLM scale.
(ii) A unified reading of random-depth, matched-depth and per-step supervision as the same identifiability fix, with a
falsifiable condition (tight mass at selection) that tells practitioners which variant will work. (iii) A benchmark rule:
extrapolation in latent steps can only be measured on problems deeper than one forward pass (Prop. 8), which ProsQA at
its native depth is not.

## 9. Candidate fixes for the six open problems of latent test-time scaling (2026-09-29 21:40 EDT; exploratory, not a pre-registration)

Requested by the user via the Audit session, for these six problems:
1. slack / underspecification;
2. step counting;
3. drift / fading;
4. shortcuts on real tasks;
5. supervision cost;
6. no re-anchoring in latent loops.

For every fix below: the mechanism it targets, the closest prior work and how it differs, a cheap first test, the
outcome that would falsify it, and the cost. Toy GPU costs assume ~0.9 GPU-h per 30k-step run at ~$0.16/h, so an
8-seed arm is ≈ $1.2. **Covered** marks fixes a running pilot already tests; nothing here launches without its own
dated pre-registration in THEORY.md. **★** marks fixes that address several problems at once.

**Already covered by running work (not duplicated):**

| pilot / study | problems | what it tests |
|---|---|---|
| D/E anytime | 1 | tight per-step supervision |
| F jitter (J, JAT, ATW/ATL) | 3 (+1) | restoring dynamics; supervision only inside the selection window |
| G set memory (G/GAT/GL/GLAT) | 3, 6 | monotone state |
| H fine-tune repair | 1 (+3) | late removal of slack |
| I periodic clock | 2 | a clock that cannot run out |
| J chunk-reset attention | 2, 6 | stationary history |
| §26 T8 re-anchoring (done) | 2, 3, 6 at inference | 24 → 55/59 |
| PROTOCOL8 (experiments session) | 4 | train above one-pass capacity c, latent-bottleneck mask |

### Problem 1 — slack / underspecification (the seed lottery)
* **★ F1a — label-free tight budgets** (problems 1 and 5). Train with a budget K ~ U{1..K} per batch, drawn independently
  of the example, and supervise only the final answer.
  - *Mechanism:* Prop. 4 says the support must reach the tight budget at the circuit-selection stage and must not be a
    function of anything the model can observe. It then identifies unit speed with no intermediate labels (impossible
    targets at small K act as noise, and the best response is to be as fast as possible).
  - *Prior work:* Kuo et al. 2606.29983 use an n-centred budget, K = clip(n + Δ, 1, 20) with Δ ~ U{−5..5}, which
    sometimes undershoots the steps needed. Bansal 2202.05826, Huginn and DR-Loop use fixed minimums of 1–2. No
    published scheme manipulates slack directly. Our P22 randk (K ∈ 4..8, step embeddings on) was never tight at stage
    3; this version is.
  - *First test:* toy, `--K_random 1:6`, 8 seeds paired with Pilot D (the flag already exists). ≈ $1.2.
  - *Falsified if* the pair rule on d8 vs control is not supported while anytime (D) is.
* **★ F1b — clock-adversarial training** (problems 1 and 2). A small probe predicts the step index k from each thought;
  gradient reversal removes step information the transition could key a schedule on.
  - *Prior work:* domain-adversarial training (Ganin et al. 2016); new here as a way to remove a latent clock.
  - *First test:* toy, 8 seeds, ≈ $1.2 plus ~1 h of coding.
  - *Falsified if* probe accuracy on k falls to chance but extrapolation and p[K−1] are unchanged.
  - *Risk:* depth legitimately correlates with k, so the probe should be conditioned on the pointer depth, i.e. remove
    only the step information that is *not* explained by progress.
* **F1c — seed triage by early P7.** This is not a mechanism fix: it cuts the lottery's cost using the experiments
  session's instrument (AUC 0.92 in-training). Train N seeds and continue only those with high early P7. Cost: nothing
  new. Already validated (PROTOCOL7 T3, §21 PO3).

### Problem 2 — step counting (implicit clock)
* **F2a — latent-history dropout.** During training, each earlier latent position is hidden from the latent step with
  p = 0.5; the queries still read everything.
  - *Mechanism:* a predecessor count read from the history (Haviv et al. 2203.16634) becomes unreliable, so schedules
    keyed to it do not fit.
  - *Prior work:* randomised positional encodings for length generalisation (Ruoss et al. 2305.16843) randomise
    positions, not the attendable set; our T3 proposal was never run.
  - *First test:* toy, 8 seeds, ≈ $1.2 plus ~30 min of coding.
  - *Falsified if* retrieval share drops but extrapolation does not rise (Prop. 9 predicts the clock then moves into the
    state, as in Pilot A; if so, F2a only helps together with F1a / anytime).
* F1b (above) also targets this problem.
* **Covered:** I (periodic clock), J (chunk reset), T8 (inference).
* *Not proposed:* sigmoid (unnormalised) latent attention. Its summed weights still grow with the number of
  predecessors, so it keeps a count clock.

### Problem 3 — drift / fading
* **F3a — amplitude renormalisation at inference** (training-free). At every step past K, rescale the thought's deviation
  from the batch mean to its average norm over steps 1..K.
  - *Mechanism:* a gain-1 constraint that holds exactly on the trained range.
  - *Prior work:* our constant α ≈ 1.3 boost (§12) and adaptive temperature (Veličković et al. 2410.01104); this matches
    the in-range amplitude per model instead of using a fixed factor.
  - *First test:* the 59 P30 ∪ L models on CPU, ~1 h, $0.
  - *Falsified if* 7-hop success is no higher than with α = 1.3 (46/59, with the block).
* **F3b — Hopfield clean-up, introduced late.** h ← h + λ·E_id·softmax(β E_idᵀ W h), with λ zero-initialised and switched
  on only at full curriculum depth (as in GL).
  - *Prior work:* modern Hopfield networks (Ramsauer et al. 2008.02217). Our A2 restore module wrote sigmoid memberships
    back at full strength from step 0 and broke learnability; this version cleans up towards stored patterns and cannot
    act before the circuit is chosen.
  - *First test:* toy, 8 seeds, ≈ $1.2.
  - *Falsified if* the transfer-curve index z is not lower than in controls, or d8 does not rise.
* **Covered:** F (jitter: C3 through the loss), G/GL (monotone set), T8.

### Problem 4 — shortcuts on real tasks
* **Covered by PROTOCOL8** (experiments session): measure one-pass capacity c on the exact format, train above it, and use
  the latent-bottleneck (G) mask so the answer cannot bypass the latents.
* **F4a — capacity-matched looping.** Run the latent steps through only the first L′ < L layers of the pretrained model,
  so that one pass resolves fewer hops and native benchmarks exceed c.
  - *Prior work:* depth-recurrent LMs (Huginn 2502.05171, Ouro 2510.25741) loop a middle block of a model trained that
    way; here the loop is shortened deliberately to create iteration pressure on a fixed benchmark.
  - *First test:* measure c(L′) for GPT-2 Coconut at L′ = 4, 6, 8 by inference only on the Phase 0 checkpoint (after
    Phase 0 reports).
  - *Falsified if* accuracy collapses rather than shifting work into the latent steps.
  - *Cost:* ≈ $1 of inference, plus a training run only if c(L′) is clean.
* **F4b — report latent benefit per difficulty stratum.** Accuracy with K latents minus accuracy with 0, as a function of
  depth relative to c (Prop. 8). This is a methodology fix: a benchmark on which latents cannot matter should not be
  used to claim latent reasoning. Cost: $0 (a re-analysis on any sweep).

### Problem 5 — supervision cost (no intermediate targets in natural data)
* **★ F1a** (label-free tight budgets) is the main candidate.
* **★ F5a — re-anchoring distillation** (problems 2, 3, 5 and 6). Use training-free re-anchored inference (T8), which
  succeeds on 55/59 models at 7 hops, as a teacher: generate pseudo-labels for queries beyond the training depth, then
  fine-tune the model to produce them in *one* uninterrupted episode with K′ > K.
  - *Prior work:* iterated amplification and distillation (Christiano et al. 1810.08575) is the closest precedent;
    also self-improving transformers (Lee et al. 2502.01612), Deng et al. 2405.14838 (internalising CoT step by step),
    2407.06023, and STaR (Zelikman et al. 2022) for text CoT. New here: the teacher is a clock-resetting inference procedure, so the
    labels are exactly those the untrained continuation gets wrong.
  - *First test:* the 20 P30 failures from Pilot H. Pseudo-labels on long-chain graphs come from exp12 on CPU (~1 h);
    fine-tuning takes 6000 steps each, ≈ 5 GPU-h ≈ $0.8. The comparison is Pilot H's FT-C.
  - *Falsified if* single-episode d7/d8 after distillation is not above FT-C's, or near-distance accuracy falls.
  - *Engineering:* a data path for deeper graphs with pseudo-labels (~2 h).
* **F5b — label-free progress consistency.** Add a loss that makes the readout's evidence for "reachable" non-decreasing
  over latent steps (a hinge on p_k − p_{k+1}), with final-step supervision only.
  - *Prior work:* PonderNet and deep supervision use labelled targets; this uses monotonicity only.
  - *First test:* toy, 8 seeds, ≈ $1.2.
  - *Falsified if* no dwell reduction (p[K−1]) and no d8 gain.
  - *Caveat:* monotone evidence does not by itself forbid dwelling (constant evidence satisfies it). Expect it to
    remove only reverts and restarts, so it is weaker than F1a.

### Problem 6 — no re-anchoring in latent loops
*Update 21:48 EDT (reworded 21:48):* F6a and F6b as designed were dropped before running. In one debug model, neither a linear
decoder of h_K nor the sweep head's attention recovers the probed chain's node better than 1/W; whether this means
single-branch following (it conflicts with Part IV P25's per-edge split) is being measured on all 59 models (THEORY
§30 B1).
* **Covered:** T8 at inference (done), J (training-time chunking), G (monotone set).
  Relatives of T8 among chunked or erasing reasoners: HRM 2506.21734, InftyThink 2503.06692, PENCIL 2503.14337, the
  Markovian Thinker 2510.06557.
* **★ F6a — learned frontier decoder plus set restart** (problems 3 and 6). Train a small head (thought → multi-label over
  node ids, supervised by the true frontier) on *frozen* trained models, then restart each episode from the decoded set.
  - *Mechanism:* the model's own attention picks a frontier node at chance among the W at the right depth (30 % on the
    probed chain), so the gap to the oracle anchor (0.612 vs 0.885 at 8–10 hops) is a decode problem.
  - *Prior work:* Patchscopes (2401.06102) decodes entities from hidden states without training; CLRS hints supervise the
    whole state. Here only a readout is trained and the model is untouched.
  - *First test:* CPU, 59 frozen models; the head trains in minutes per model. $0.
  - *Falsified if* decoded-anchor re-anchoring is not above the model-attention anchors (far 8–10 0.667 with top-3).
  - *At LLM scale:* this is the entity readout the experiments session plans for PROTOCOL8's inference arm.
* **★ F6b — soft re-anchoring** (problems 2, 3 and 6). Every M steps, replace the thought by its projection onto the
  node-embedding simplex (softmax over ids, re-embedded) and cut the latent history. This is T8 made differentiable and
  trainable, i.e. F3b plus J.
  - *First test:* inference-only on the 59 models first (CPU, ~1 h), to check that a projected thought is a usable input
    at all; a toy training arm only if it is.
  - *Falsified if* the inference variant is below T8 (hard re-anchoring).

### Suggested order if the pilots leave room (all small)
1. **$0 CPU tests first:** F3a (renormalisation), F6a (decoded anchors), F6b (soft re-anchoring, inference only).
2. **One cheap GPU arm with the best cost-to-information ratio:** F1a (label-free tight budgets). It tests Prop. 4's
   central claim and, if it works, removes the need for intermediate labels (problems 1 and 5).
3. **Then F5a** (re-anchoring distillation), which alone targets four problems.

Each step gets a dated pre-registration first. Total ≈ $3–4 of GPU, within the reserve plan.
