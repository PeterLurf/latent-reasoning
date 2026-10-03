# Ideas to borrow from adjacent fields — reasoning, recurrence, neuroscience (2026-09-29)

From the Literature session, at the user's request ("like using clocks from CTMs"). CTMs are already worked through in
`THEORY.md` §26 (T8 re-anchoring, A8 stationary history, tick-selection loss), so they are not repeated here. arXiv IDs
were checked against the arXiv API. The non-arXiv sources (Kay 2020, Reynolds & Heeger 2009, Ma et al. 2014,
Mordvintsev 2020) were checked by search. Each idea gives: the source, what to take, which problem it targets (the
theory stream's C1 identification, C2 no clock, C3 restoring, C4 idempotence, or an experiments-stream problem), and
the cheapest test. **Cost:** A = re-analysis of existing models, B = a training arm on the current recipe, C = a new
architecture or paradigm.

## Summary, ranked by value per cost

| # | Idea | From | Targets | Cost |
|---|---|---|---|---|
| 1 | **Sample-pool training**: continue rollouts from the model's own late latent states | Neural cellular automata | C4, C2, dangling thought | B |
| 2 | **A clock that never runs out**: give a *periodic* phase (mod p), never an absolute count | Clockwork RNN, Phased LSTM, your §25 phase-bit runs | C2 | B |
| 3 | **Divisive-normalisation fit of the width law** | Reynolds & Heeger normalisation model | Finding 3, Th4 | A |
| 4 | **Slots vs continuous resource**: which model explains the branch-edit data | Working-memory research (Ma, Husain, Bays) | Findings 1, 3 | A |
| 5 | **Copy gate + geometric attention** | Neural Data Router | C3 (fade), A3 (width) | B |
| 6 | **Hopfield clean-up step** on the node-id table | Modern Hopfield networks | C3 | B |
| 7 | **Time-multiplexing alternatives** instead of superposition | Hippocampal theta cycling (Kay et al.); oscillator binding (AKOrN, CAE) | width limit | A (detect), C (build) |
| 8 | **Structure-coupled clock**: step ids tied to depth, not count | Position coupling, Abacus embeddings | C1/C2 | B |
| 9 | **Self-improvement curriculum**: train on own filtered answers one hop deeper | Self-improving transformers | horizon, PROTOCOL8 | B |
| 10 | **Reasoning as energy descent** (steps until converged) | IREM, IRED, energy-based transformers | C2, C4 | C |
| 11 | **Homogeneous iteration + progress-based termination** | Iterative homogeneous GNNs | C3, C4 | B |
| 12 | **Patchscopes** to read thoughts in pretrained models | LM interpretability | PROTOCOL8 | A |

---

## 1. Sample-pool training (neural cellular automata)

**Source.** Mordvintsev et al. 2020, *Growing Neural Cellular Automata*, Distill (distill.pub/2020/growing-ca). Models
trained only to *grow* a pattern in N steps "don't know how to persist" it: some explode, some decay. Training from a
**pool of the model's own previous end states**, so that later steps are trained as continuations, gives persistent
attractors, and adding **damage** to pooled states gives regeneration.
**Take.** Keep a pool of (graph, latent history truncated to the last few thoughts, position counter reset) from earlier
batches' step-K states. Start some training rollouts from a pooled state and continue with the ordinary (or anytime)
loss.
- Every thought, including h_K, becomes an input to further trained steps: no dangling state (T1, C4).
- The absolute step index no longer lines up with frontier depth, so a count clock stops fitting the loss (C2). That is
  what OV (overrun) did not do: OV always restarts from depth 0 at step 0.
- "Damage" is T6's jitter.

**Test.** One arm on the P30 recipe (+ anytime as a second arm). Prediction: pointer p[K−1] ≥ K − 2, and no revert at
K + 1.
**Risk.** Pooled states from early training are wrong. The NCA fix is to replace the worst-loss pool sample each batch
with a fresh seed.

## 2. A clock that never runs out (periodic phase instead of absolute count)

**Source.** Koutník et al. 2014, *A Clockwork RNN*, 1402.3511 (modules updated at different fixed periods); Neil et al.
2016, *Phased LSTM*, 1610.09513 (an oscillating time gate). Plus your own observation (THEORY §25): a run with a
**period-2 phase bit** in its state (`1 1 2 2 3 3 | 4 4 5 5 6 6`) extrapolates. A clock is harmless if it is
*periodic*, because it never leaves its trained range.
**Take.** Replace the step embedding by a **periodic** embedding e(k mod p) with p ∈ {2, 3} (or sin/cos of 2πk/p). The
model gets a phase to dwell and advance with, but no absolute count to run out. This targets Prop. 9 ("slack creates
the clock") head-on: if slack makes the model *want* a clock, provide a stationary one.
**Test.** Arm "periodic step embedding, p = 2" vs no step embedding vs absolute step embedding, on the P30 recipe,
under slack (K = 6 and K = 8). Prediction: extrapolating share(periodic) > (none) > (absolute), with the difference
largest at K = 8.
**Note.** This is the cheapest clock idea and speaks directly to the user's CTM question. The CTM normaliser with r = 0
is an absolute (√t) clock, while a periodic clock is stationary.

## 3. Fit the width law with a divisive-normalisation model (neuroscience)

**Source.** Reynolds & Heeger 2009, *The Normalization Model of Attention*, Neuron 61:168–185; Carandini & Heeger 2012,
*Normalization as a canonical neural computation*, Nat Rev Neurosci 13:51–62. Response = drive × attention gain /
(σ + pooled drive).
**Take.** The per-frontier-edge budget (Finding 3) is a normalisation law. With amplitude a(n) = c / (σ + n) (n =
frontier edges), the log-log slope is β(n) = n / (σ + n). It **rises with n**, from ~0 when σ ≫ n to 1 when n ≫ σ. That
would explain β ≈ 0.6–0.9 without a free exponent, why seeds differ (different σ), and why the P10 "β = 0.5" prediction
was mixed.
**Test (existing data).** Fit a single σ per model to the fork/chain and W = 2/4/8 amplitudes; compare held-out error
with the power-law fit. Prediction: curvature in log-log (steeper at larger n); σ correlates across the models' reach.
Pure re-analysis.

## 4. Slots vs continuous resource (working-memory research)

**Source.** Ma, Husain & Bays 2014, *Changing concepts of working memory*, Nat Neurosci 17:347–356: the slot-vs-resource
debate. Slot models hold k items perfectly and drop the rest; resource models spread a continuous resource, with
precision falling as a power law in set size. They separate on the *distribution* of per-item precision, not its mean.
**Take.** The same model comparison for branches in a thought. **Slots** predict bimodal per-branch edit effects at
W > k (held branches full-size, unheld ~0), with k constant across W. **Resource** predicts a unimodal effect shrinking
smoothly with W (your W^−β). A **hybrid** (resource with a cap) is also standard.
**Test (existing data).** Per-branch targeted effects from Parts I–II at W = 2, 4, 8: fit the slot, resource and hybrid
likelihoods per seed. Pure re-analysis; turns Finding 1 into a model comparison with a large literature behind it.

## 5. Copy gate + geometric attention (Neural Data Router)

**Source.** Csordás, Irie & Schmidhuber 2021, *The Neural Data Router*, 2110.07732: two changes to transformers, a
**copy gate** (a column keeps its value unless the gate opens) and **geometric attention** (stick-breaking: attend to the
*nearest* match, no softmax competition over all keys). Together they give 100% length generalisation on compositional
table lookup and generalisation to deeper computations.
**Take.**
- (a) Gate the thought update, h_k = g ⊙ f(h_{k−1}, ·) + (1 − g) ⊙ h_{k−1}. The pointer's default is to *persist*, so
  the fade must be actively written (C3), and a stalled step keeps rather than loses amplitude.
- (b) Geometric attention in the sweep head. Per-edge mass stops falling with the number of matching edges, which is
  A3 and the causal test of Finding 3 the advisor note proposed.

**Test.** Two arms on the P30 recipe. For (b), also rerun the fork/chain ratio: prediction ≈ 1.0 instead of 0.5.

## 6. Hopfield clean-up of the pointer

**Source.** Ramsauer et al. 2020, *Hopfield Networks is All You Need*, 2008.02217: attention over stored patterns is a
Hopfield update. At high β it converges to a single stored pattern; at intermediate β it converges to *metastable states
averaging a subset*, i.e. a restoring map that can still hold a set.
**Take.** After each latent step, one retrieval against the node-id embedding table,
h ← h + λ · E softmax(β Eᵀh), with β learned or annealed. This is a parameter-light restoring map (C3) that keeps
superpositions of a few nodes as metastable states. It is a gentler alternative to the MR arm's sigmoid write-back,
which broke learnability. Zero-initialise λ.
**Test.** One arm (+ anytime). Read the transfer curve (§14): prediction is "restoring" in most competent runs.

## 7. Time-multiplexing alternatives instead of superposing them

**Source.** Kay et al. 2020, *Constant Sub-second Cycling between Representations of Possible Futures in the
Hippocampus*, Cell 180:552 (doi 10.1016/j.cell.2020.01.014): rat hippocampus represents two possible future paths in
**alternation, one per ~125 ms theta cycle**. Also binding by phase in ML: Miyato et al. 2024, *Artificial Kuramoto
Oscillatory Neurons* (AKOrN), 2410.13821; Löwe et al. 2022, *Complex-Valued Autoencoders*, 2204.02075.
**Take.** Your width limit exists because alternatives *share* a step (per-edge budget). Brains also cycle through
alternatives, trading time for crosstalk.
- **Detect (A):** in existing models, especially upfront-format and wide (W = 8) cases, does the sweep head's
  attention alternate between branches across latent steps rather than splitting within a step? A per-step branch
  attention matrix answers it.
- **Build (C):** complex- or phase-valued thought coordinates, so branches bind by phase and do not compete for
  amplitude.

## 8. A clock coupled to the task's structure, not the step count

**Source.** Cho et al. 2024, *Position Coupling*, 2405.20671, and McLeish et al. 2024, *Transformers Can Do Arithmetic
with the Right Embeddings* (Abacus), 2405.17399. Length generalisation comes from giving related tokens the *same*
position id (ids aligned with the algorithm), not from removing position information. Contrast: Xu & Sato 2024,
*Looped Transformers … Enhancement via Timestep Encoding*, 2410.01405, argue timestep encodings *increase* looped
transformers' expressivity. That is a tension with C2 to address explicitly.
**Take.** Instead of "no clock" (C2), a **depth-coupled** clock: the step id is the frontier depth the thought is
supposed to hold. During training it comes from the anytime targets; at test time it is incremented, and is always in
range if made periodic (idea 2) or relative (difference to the previous thought's depth).
**Test.** One arm; compare with idea 2.

## 9. Self-improvement curriculum past the training depth

**Source.** Lee et al. 2025, *Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges*,
2502.01612: models iteratively train on their own filtered answers at slightly harder instances (10- to 100-digit
addition, mazes) "without apparent saturation"; filtering by self-consistency matters.
**Take.**
- (a) Toy: after training, label depth-7 chains with the model's own answers at K′ = K + 2 (keep only answers stable
  across K′ = K + 2 … K + 4), fine-tune, and repeat. Does the horizon walk outwards? Do non-extrapolators poison
  themselves (labels at chance)? The latter is a clean test of whether failures are *fixable by data* or *locked*
  (Pilot H asks the same with a different lever).
- (b) PROTOCOL8: the practical route to getting a pretrained model onto problems deeper than one pass, without gold
  labels.

## 10. Reasoning as energy descent

**Source.** Du et al. 2022, *Learning Iterative Reasoning through Energy Minimization* (IREM), 2206.15448; Du et al. 2024,
*Learning Iterative Reasoning through Energy Diffusion* (IRED), 2406.11179 (adapts step count to difficulty; solves
**pathfinding in larger graphs** than trained); Gladstone et al. 2025, *Energy-Based Transformers*, 2507.02092.
**Take.** Make the latent step a descent step on a learned energy E(graph, h): h ← h − η∇ₕE. The transition has no step
index by construction (C2); fixed points are minima, so it is restoring (C3); "done" is convergence (C4). It is the
principled version of A5 halting.
**Cost.** A new paradigm, best as a side project. IRED's larger-graph pathfinding is the evidence it extrapolates on
exactly this kind of task.

## 11. Homogeneous iteration + progress-based termination

**Source.** Tang et al. 2020, *Towards Scale-Invariant Graph-related Problem Solving by Iterative Homogeneous GNNs*,
2010.13547: the iteration count is learned from computation progress (as graph algorithms scale with graph size), and
*homogeneous* (bias-free) transforms make outputs scale-equivariant, giving generalisation across graph sizes and
diameters.
**Take.**
- (a) Cheap ablation: remove the biases (or use a positively homogeneous activation) in the pointer path, and check
  whether the transfer-curve gain becomes amplitude-independent. That isolates *where* the sub-unity gain comes from.
- (b) Their progress-based termination is a content-triggered stop (C4), the same family as CTM max-certainty.

## 12. Patchscopes for reading thoughts in pretrained models

**Source.** Ghandeharioun et al. 2024, *Patchscopes*, 2401.06102: decode a hidden state by patching it into a target
prompt and letting the model describe it.
**Take.** In PROTOCOL8 (GPT-2 or larger), the toy's sweep-head read-out of the pointer does not exist. Patch each
continuous thought into a prompt like "The node reached so far is" to read the pointer and compare with gold depth.
This gives the ProsQA experiments a pointer trajectory like the toy's.

---

**Also relevant, not proposed as ideas:** Yan et al. 2020, *Neural Execution Engines*, 2006.08084 (learned attention
*masks* as state plus binary representations → strong size generalisation; supports A6/A2); Gandhi et al. 2024, *Stream
of Search*, 2404.03683 (training on search traces with backtracking); Ye et al. 2024, *Beyond Autoregression: Discrete
Diffusion for Complex Reasoning and Planning*, 2410.14157 (parallel iterative refinement with no step order).

**Suggested order.** 3 and 4 (free, on existing data), then 2 and 1 (one arm each, and they answer Prop. 9 and the
dangling-state question), then 5b (the causal width test), then 6. Ideas 7–12 are longer-horizon.
