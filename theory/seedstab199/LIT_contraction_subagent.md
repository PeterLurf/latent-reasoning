# Contraction vs computation, delta-rule memories, looped-LM failures, attractor codes, basin selection: prior art for candidates (i)-(iv) (2026-10-13 21:3x EDT)

Literature subagent for Theory. **Extends** docs/LIT_SEED_STABILITY_2026-10-13.md (saddle escape, continuation, syre, stochastic collapse, restarts) and
docs/LIT_SEED_STABILITY_CANDIDATES_2026-10-13.md (Gradient Flossing as the closest prior for (ii), the Anil reconciliation, the orthogonal-RNN line, contracting
RNNs 2004.05290 / R2DN 2504.01250, STARS 2605.26733, code facts on alpha^T and beta = 1). Items those notes cover are only pointed to here, not repeated.

- **IDs:** every arXiv ID below was fetched from its arxiv.org/abs page (citation_title meta) on 2026-10-13, 21:15-21:32 EDT. Titles are quoted exactly from
  that meta. Non-arXiv items were checked by web search (journal, volume, pages); where an arXiv version could not be found I say so.
- **Claims:** from the abstracts unless tagged **[ft]** = I extracted the full text myself (pdftotext) and read the passage, or **[fs]** = full text read
  through a WebFetch summary (quotes not independently re-checked).
- Raw abstracts: scratchpad/seedstab/lit/meta_*.txt. Extracted PDFs: lit/pdf/.

## Short answer

1. **The two-regime picture has theory behind both horns; nobody has tied the horns to seed outcome.**
   - *Contractive horn = fading memory:* a contractive recurrence is well approximated by a truncated / feed-forward one, for inference and for training
     (Miller & Hardt 1805.10369). Contracting model classes can represent exactly the fading-memory systems (REN 2104.05942; Boyd & Chua 1985).
     This explains a plateau whose per-step accuracy decays with the pass index (s3).
   - *Neutral horn = no error correction:* **Chung, Choi, Kim 2605.07755** prove that an affine recurrence that preserves its state representations must
     act as the identity on the state-separating subspace (their Theorem 1, **[fs]**). It therefore cannot correct errors there. They also show that the
     tracking horizon is set by the within-class spread R(t) over the between-class separation M(t).
     This explains why the converged, path-dependent seeds still fail at depth (n 9-12) and after extra passes. It is our "integrator never forgets its own
     errors" in theorem form.
   - Their remedy is **state-dependent** contraction toward codewords. In their test, pointwise nonlinearities support it and **whole-vector
     normalisations do not**. That bears on our per-head RMS-normalised read; it's a lead, not tested.
2. **The closest prior to (i) is new and strong: Howe 2609.18966 (July 2026, pre-registered, 202 runs).**
   - Deleting the additive input injection b_t from a Householder (DeltaProduct-class) linear RNN turns length-generalisation collapse into the exact
     automaton (median accuracy 1.00 at 16x the training length).
   - From a verified-exact init with W_b = 0, Adam **grows** the additive path and pulls the model off the solution. It is "a norm-inflation channel that
     cross-entropy feeds" **[ft]**.
   - Our additive heads (S_a += phi(k)v^T, no decay) and the delta heads' +beta v k^T term are such injections, applied across passes.
   - Howe's setting is from-scratch training over sequence time, with seeds counted per cell. It is not a looped pretrained LM and not about basin-entry
     probability.
3. **(ii) is partly pre-empted on both of its proposed novelty axes, but not in combination.**
   - *Time-dependent target:* Wöhrer & Kuehn 2602.09613 already regularise FTLEs **only in an early window of the forward dynamics**.
   - *Seed reliability:* Engelken 2312.17306 reports that pre-training flossing raises the **success rate**, and that flossing **continuously** drives
     accuracy "close to chance" **[fs]**. That is direct evidence that a uniform lambda ≈ 0 target is wrong.
   - *Observation of the target shape:* Krishna et al. 2609.26487 observe the post-completion signature (states contract along the trajectory direction,
     with expanding directions retained) in Tiny Recursive Models.
   - No paper sets a target that is **item-dependent and indexed by completion**, on a **task-defined subspace**.
4. **(iii) and (iv) are governed by classical noisy-bifurcation theory, and it cuts against (iii) used alone.**
   - A slow passage through a loss of stability is **noise-controlled even when the noise is extremely small** (Lythe adap-org/9707008; Berglund & Gentz
     math/0008208). That fits outcomes being flipped by GPU class or kernel order.
   - The branch chosen is set by the **imperfection (bias) relative to noise**. The probability rises with the bias and falls with sweep rate (Kondepudi,
     Moss & McClintock 1986, as summarised by Juel, Darbyshire & Mullin 1997 **[ft]**).
   - A *perfectly symmetric* designed init point is the **worst case** for seed reliability (P = 1/2) unless it is deliberately unfolded. The formula is
     derived in section E below.
5. **Looped-LM failure reports from 2025-2026 are init / normalisation / gate traps fixed by hand.**
   - The cases: Geiping's "learned early to ignore the incoming state"; Shapiro's self-locked zero gate; Sapunov's ACT bias trap (most runs fail; bias -3
     fixes it); Ouro's 8 → 4 recurrent steps.
   - None measures contraction vs path dependence as the separating statistic, or gives a basin-entry probability.

## A. Contraction vs computation: memory and capacity bounds

| ID | Title (exact) | Claim | Relevance / pre-emption |
|---|---|---|---|
| 1805.10369 | Stable Recurrent Models (Miller, Hardt) | Stable (contractive) RNNs are well approximated by feed-forward networks for inference and for gradient-descent training. | The formal reason a contractive pass map has a bounded effective depth: the s3 plateau. Pre-empts none of (i)-(iv). |
| 2104.05942 | Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness | All RENs are contracting and can approximate all fading-memory and contracting systems. | Contraction-by-design = fading memory. This is the regime to forbid on progress directions. |
| (no arXiv) | Boyd & Chua, "Fading Memory and the Problem of Approximating Nonlinear Operators with Volterra Series", IEEE TCAS 32(11):1150-1171, 1985 | Fading-memory operators are approximable by finite-dimensional linear dynamics plus a nonlinear readout. | The classical bound on what a contractive map can compute. |
| (no arXiv) | Jaeger, "The 'echo state' approach to analysing and training recurrent neural networks", GMD Report 148, 2001 (erratum later) | Echo-state property = asymptotic forgetting of the initial state. | Our "path-independent" models have the echo-state property across passes. 1806.00797 and 2010.12047 (Grigoryeva / Gonon & Ortega): ESNs with fading memory are universal *for fading-memory* filters. |
| cond-mat/0402452 | Short-Term Memory in Orthogonal Neural Networks (White, Lee, Sompolinsky; PRL 92, 148102) | Memory capacity of linear networks with orthogonal connectivity scales with system size. | Information-capacity argument for (i): orthogonal transport keeps O(N) recoverable history. Contraction keeps O(1/(1-rho)). |
| (PNAS 2008, PMC2596211) | Ganguli, Huh, Sompolinsky, "Memory traces in dynamical systems" | Fisher memory: normal networks have total capacity exactly 1; at most N, reached only by non-normal feedforward structure. Saturating nonlinearities limit it to sqrt(N). | Bounds for any "memory resolution" claim in (i). |
| 2605.07755 | Rethinking State Tracking in Recurrent Models Through Error Control Dynamics (Chung, Choi, Kim) | Affine recurrent nets cannot correct errors along state-separating subspaces once they preserve the states. The failure horizon is where the distinguishability ratio crosses the decoder's threshold. | **Central for (i).** Theorem: preservation ⇒ A restricted to U is I ⇒ no correction **[fs]**. Remedy: state-dependent contraction around codewords **[fs]**. Partly pre-empts (i)'s "error-correcting contraction" as a principle. No training dynamics or seed analysis **[fs]**. |
| Wang & Slotine, Biol. Cybern. 92:38-53 (2005), DOI 10.1007/s00422-004-0527-x (no arXiv found; thesis MIT DSpace 1721.1/30343) | "On partial contraction analysis for coupled nonlinear oscillators" | Partial contraction: convergence to a manifold or behaviour rather than to a point. | The control-theory form of (i)'s split: contract transverse to the code manifold, stay neutral along it. Discrete-time version: nlin/0609064 "Examples of Synchronization in Discrete Chaotic Systems" (Botero & Slotine). |
| 1902.09689 / 2006.12070 | AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks / Lipschitz Recurrent Neural Networks | Antisymmetric (isometric) recurrence plus small damping; Lipschitz RNN splits the recurrence into symmetric and skew parts. | An isometric plus contraction split already exists, but as a *uniform* damping γI. That is not a subspace split around a code. Partly pre-empts the architecture of (i), not its purpose. |
| (Neural Comput. 16:1413, 2004; DOI 10.1162/089976604323057443) / 1611.01232 | Bertschinger & Natschläger, "Real-Time Computation at the Edge of Chaos in Recurrent Neural Networks" / Deep Information Propagation | Complex computation only near the critical boundary; trainability depth diverges at criticality. | The edge-of-chaos rationale for lambda ≈ 0 before completion. 2002.00025 (Can, Krishnamurthy, Schwab): gating creates slow modes. |
| 2607.10681 | LayerNorm as Implicit Gain Control in Looped Transformers (Buehlmaier) | LayerNorm makes the loop Jacobian non-normal and contractive at fixed points *even where the operator norm exceeds 1*. A minority of inits never converge. GD routes memory through the nonlinear recurrence, not the constrained carry. | **Measurement warning for (ii):** use spectral or Lyapunov quantities, not operator norms. Single-author CPU-scale study. |
| 2609.26487 | When Recursive Models Finish Computing (Krishna et al.) | In TRMs, latent motion drops after the first exact solution. Completed states contract along the trajectory direction while the local Jacobian keeps strongly expanding directions ("trajectory-conditioned anisotropic stability"). | Empirical support for (ii)'s *post*-completion target being directional (a subspace), not global. Observational only: no regulariser. |

## A'. Lyapunov / Jacobian targets restricted in time or subspace (focus a)

| ID | Title (exact) | What is restricted | Pre-empts (ii)? |
|---|---|---|---|
| 2312.17306 | Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians | Top-k exponents from QR → 0. k is task-dependent by count (e.g. k = 75 of N = 80 for delayed copy d = 40). Schedules tested: pre-flossing only, episodic during training, continuous. Continuous flossing "detrimental": accuracy falls "close to chance". Results over 5-20 network realisations **[fs]**. | Subspace = the *leading Oseledets subspace*, chosen by the dynamics, not by the task. The time dependence is over *training epochs*, not passes. The abstract's "enhances … success rate" **is** a seed-reliability claim. Strongest pre-emption of (ii) as a seed-reliability method. |
| 2602.09613 | Tracking Finite-Time Lyapunov Exponents to Robustify Neural ODEs (Wöhrer, Kuehn) | FTLE regularisation "in the early stage of the input dynamics" only: exponents far from zero are suppressed early. | **Pre-empts the windowed-in-forward-time form** of (ii)(b). Different goal (adversarial robustness), different setting (continuous depth), and the window is not item-dependent or tied to completion. |
| 2506.12810 | Lyapunov Learning at the Onset of Chaos (Benati et al.) | lambda_max kept "evolving around zero over time" during online learning. | Uniform target; background only. |
| 1806.09919 | Tangent-Space Regularization for Neural-Network Models of Dynamical Systems (Carlson et al.) | Model Jacobian regularised *along trajectories*. | Early trajectory-restricted Jacobian prior; not phase-dependent. |
| 2605.26733 (covered) / 2106.14342 (covered) / 2604.12946 | … / … / Parcae: Scaling Laws For Stable Looped Language Models | STARS: spectral radius < 1 throughout. DEQ: Jacobian penalty at the equilibrium. Parcae: injection made contractive by design ("discretization of a negative diagonal parameterization") to stop residual explosion and loss spikes. | All aim for contraction throughout, the opposite of (ii)'s pre-completion target. |

**Verdict for focus (a):** a target that is **lambda_U ≈ 0 for passes t ≤ n(x) and lambda_U < 0 for t > n(x)**, on the step-decodable subspace U, used to
make basin entry reliable, was not found. Each ingredient exists separately (time window: 2602.09613; seed success: 2312.17306; the post-completion
direction: 2609.26487).

## B. Linear-attention / delta-rule memories: training dynamics and seed reliability (focus b)

| ID | Title (exact) | Claim | Relevance |
|---|---|---|---|
| 2609.18966 | The Automaton Underneath: The Additive Input Pathway Is a Parasitic Attractor for State Tracking in Householder Linear RNN (Howe, Jul 2026) | With b_t the models fit length 32 and collapse OOD; without it, the exact automaton (5 seeds per cell). From an exact init with W_b = 0, Adam grows W_b (probe 0 → 0.77 on S5) and position-512 accuracy falls to 0.23. Zeroing W_b at inference restores exact generalisation in 18/20 seeds at the law-minimal width, but only 1-2/5 above it **[ft]**. | **Closest prior to (i)** and to "a parametrisation that removes the bad basin". The mechanism (norm inflation through the only non-isometric channel) maps onto our additive heads, whose norm grows every pass. Pre-empts (i)'s isometric half *as a learnability fix in LRNNs over sequence time*. Not looped, not pretrained, no basin-probability theory. |
| 2603.01959 | The Expressive Limits of Diagonal SSMs for State-Tracking (Shakerinava et al.) | Multi-layer models "often fail to learn" non-Abelian tracking that they can express. Per Howe **[ft]**, the gap survives near-solution init: "not expressivity but optimization". | Names the expressivity-learnability gap. |
| 2507.02782 | Understanding and Improving Length Generalization in Recurrent Models (Buitrago Ruiz, Gu) | "Unexplored states hypothesis". Initialising the state with Gaussian noise **or with the final state of a different input sequence** fixes length generalisation (2k → 128k) in about 500 steps. | Their intervention is exactly our derangement-donor start. As a training intervention it rewards *forgetting the start*, the property that marks our failed seeds. See the tension note below. |
| 2405.21064 | Recurrent neural networks: vanishing and exploding gradients are not the end of the story (Zucchet, Orvieto) | As memory lengthens, parameter changes cause ever-larger output changes ("curse of memory"), even without exploding gradients. Element-wise recurrence plus careful parametrisation mitigates this. | Training-dynamics cost of eigenvalues near 1: the price (i) pays on U. Predicts a higher effective sharpness on U, so (i) may need a smaller LR on the memory branch. |
| 2303.06349 | Resurrecting Recurrent Neural Networks for Long Sequences (LRU) | Stable exponential parametrisation, init near the unit circle, normalisation. | Parametrisation and init recipe for near-isometric linear recurrences. |
| 2412.06464 / 2406.06484 / 2503.14456 / 2102.11174 | Gated Delta Networks: Improving Mamba2 with Delta Rule / Parallelizing Linear Transformers with the Delta Rule over Sequence Length / RWKV-7 "Goose" with Expressive Dynamic State Evolution / Linear Transformers Are Secretly Fast Weight Programmers | Architecture and expressivity papers for our delta heads. | None reports seed reliability or basin statistics. |
| 2602.14814 / 2505.21749 | Learning State-Tracking from Code Using Linear RNNs / Revisiting Bi-Linear State Transitions in Recurrent Neural Networks | Linear RNNs can do worse than nonlinear ones under partial observability. Pure bilinear (no affine term) updates are the natural inductive bias for state tracking; per Howe **[ft]**, the bilinear ones length-generalise where affine variants fail. | Independent support for "input acting only through transitions" (the isometric part of (i)). |

**Gap:** I found no paper on the *training dynamics or seed reliability* of gated-delta / additive linear memories as a cross-pass memory in a looped or
pretrained model. Howe is the only causal seed-level study, and it is in LRNNs over sequence time.

**Inference (mine) from the fact sheet's equations; check against boxes/deltamem.py:**
- In the open-loop phase (γ = 0, steps ≤ 2000) every pass writes the same k, v, β, because the input and h6 are identical every pass.
- So the pass map is affine and time-invariant:
  - additive heads: S_a,t = S_a,0 + t·W (a pass counter along one fixed direction, Geiping's "slider");
  - delta heads: S_d,t = A S_d,t-1 + B, with A = α^T Π(I − β k kᵀ) fixed.
- No serial computation can be stored in memory before γ > 0. All serial structure must form during the γ ramp (2000-6000), which is a slow sweep of a
  control parameter: see E.

## C. Looped / recurrent-depth models: reported failures, fixes, state init (focus b)

| ID | Title (exact) | Failure → fix | State init / BPTT / dynamics analysis |
|---|---|---|---|
| 2502.05171 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Geiping et al.) | Bad run 1: token-correlation collapse, i.e. hidden states → identical. Bad run 2: "learned early to ignore the incoming state s". Fix: sandwich norm, peak LR 4e-4 → 4e-5, learned concat adapter **[fs]**. | s0 ~ N(0, σ²) (truncated, σ² = 2/5), "promotes … path independence". Backprop through the last k = 8 iterations. Trajectory types: fixed points, orbits, "sliders" **[fs]**. No contraction metric tied to failure. |
| 2510.25741 | Scaling Latent Reasoning via Looped Language Models (Ouro) | 8 recurrent steps gave loss spikes and gradient oscillations, so reduced to 4. Batch 4M → 8M tokens. KL β 0.1 → 0.05. Smaller LR than Transformers **[fs]**. | Entropy-regularised exit distribution. |
| 2608.11233 | Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets (Shapiro) | **"Self-locked gate": a zero mixing gate plus a zero-branch init gave gate, weight and bias gradients "all exactly zero"**. Forcing the gate to 1 recovered the gradients. Fix: an identity-biased live gate. "Liveness must be asserted, not assumed" **[ft]**. Also a loop-closure defect (no prelude re-injection). | Pretrained-LM retrofit (Qwen2.5-0.5B), so closest to our setting. Seed variance in "consolidation speed" is noted **[ft]**. Our W_o = 0 / gate-weight-0 init is the same family: check that it has no factor with exactly zero gradient. |
| 2604.21999 | Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning (Sapunov) | An ACT router-init trap makes **most runs fail** (zero and Graves-positive biases both). Bias −3 fixes it. Seed variance ±0.7 pp (ACT) vs ±9.3 pp (fixed depth). | An init-trap seed lottery fixed by moving the init off the trap: precedent for (iv)-as-tilt. |
| 2511.07384 | Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence (McLeish et al.) | A curriculum of recurrences preserves performance in pretrained → recurrent conversion. | Matches our pilot (curriculum n ≤ 4, K 4 rescued s3). Known lever. |
| 2202.05826 | End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking (Bansal et al.) | "Overthinking": behaviour degenerates with many iterations. Fix: recall (input re-injection) plus progressive training, which removes iteration-number-specific behaviour. | s1's K' 16 corruption is overthinking. Progressive training starts from an intermediate state reached without gradient. |
| 2510.04871 | Less is More: Recursive Reasoning with Tiny Networks (TRM) | HRM "overfit[s] quickly and then diverge[s]". EMA of weights prevents collapse. HRM's fixed point "is unlikely to be reached" when the 1-step IFT gradient is applied **[fs]**. | DEQ-style gradients assume a contraction that isn't there. |
| 2607.13491 / 2601.09588 / 2604.11791 / 2607.14427 / 2609.16665 | DeepLoop: Depth Scaling for Looped Transformers / Energy-Entropy Regularization: The True Power of Minimal Looped Transformers / A Mechanistic Analysis of Looped Reasoning Language Models / Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers / Right Direction, Wrong Step: Geometric Analysis of Finite-Step Failure in Looped Transformers | Residual scaling for tied depth. Single-head looped models "frequently fail" (Tsallis / Hamiltonian fix). Per-layer cyclic fixed points. Per-token fixed points. Lost progress from a step-scale mismatch (a quarter step helps in 72-83% of failures). | Background. 2609.16665's damping result is a cheap test for our post-completion corruption. |
| 2409.15647 / 2502.17416 / 2311.12424 / 2410.20672 / 2106.04537 | Looped Transformers for Length Generalization / Reasoning with Latent Thoughts: On the Power of Looped Transformers / Looped Transformers are Better at Learning Learning Algorithms / Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA / Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks | Expressivity and generalisation, with adaptive steps. | No seed or contraction analysis that I saw in the abstracts. |

**Tension to state in any proposal:**
- Anil (covered), Geiping (random s0) and Ruiz & Gu (donor or noise state) all *train for* forgetting the initial state, and report better
  generalisation from it.
- In our 15 models, forgetting marks failure.
- My reconciliation, extending the pre-check: what those papers forget is an *arbitrary* init, while progress is recomputed from the re-injected input.
  Our failed seeds forget *their own accumulated progress*.
- A method must therefore separate "forget the start" (allowed, even desirable: Ruiz & Gu) from "forget progress on U before completion" (forbidden).
- **Prediction:** training with donor starts **without** a protected U will push seeds toward the contractive basin (delta_cos ↑). With a protected U it
  should not.

## D. Attractor / error-correcting discrete codes in continuous states as a training target (focus c)

| ID | Title (exact) | Claim | Pre-empts (i)'s "error-correcting contraction around a discrete code"? |
|---|---|---|---|
| 1805.08394 | State-Denoised Recurrent Neural Networks (Mozer, Kazakov, Lindsey) | Attractor dynamics clean up the hidden state at every step, trained with an **auxiliary denoising loss** to recover past states from noisy versions. They beat the same attractor without that loss. | **Yes as a training-target mechanism** (attractor clean-up plus aux loss). Not isometric elsewhere; not about seeds. |
| 1901.08817 | State-Regularized Recurrent Neural Networks (Wang, Niepert) | Stochastic transitions between a **finite set of learnable centroids**. Easier automaton extraction; better long-term memorisation. | Yes, a discrete-code snap. No isometric transport, no seeds. |
| 1811.12530 / 2006.05252 / 2008.02217 / 2309.15505 | Learning Finite State Representations of Recurrent Policy Networks / A bio-inspired bistable recurrent cell allows for long-lasting memory / Hopfield Networks is All You Need / Finite Scalar Quantization: VQ-VAE Made Simple | Quantised bottlenecks; bistable per-unit memory; one-step attractor retrieval; FSQ. | Building blocks for the code-snap. FSQ is already in-house. |
| 2601.15336 (named in the pre-check) | Learning Discrete Successor Transitions in Continuous Attractor Networks: Emergence, Limits, and Topological Constraints | Short evaluation windows yield "impulse-driven associative solutions". Genuine attractor transitions emerge **only when stability is enforced over extended free-run periods**; shortcuts are the default. | Supports adding a free-run (post-completion) stability term, i.e. (ii)'s post-completion half. |
| 2408.00109 / 2112.03978 | Back to the Continuous Attractor / Attractor and integrator networks in the brain | Continuous attractors are structurally unstable, but a "persistent manifold" survives with similar finite-time behaviour. Robustness-capacity trade-off. | Warns that an exactly isometric (continuous-attractor-like) U is structurally fragile under training perturbations. The discrete-code contraction is what makes it robust. |
| 2605.07755 (see A) | … | Remedy = state-dependent contraction toward codewords. Pointwise nonlinearities work, whole-vector normalisation does not **[fs]**. | Theory for why the snap must be state-dependent (non-affine). |
| 2601.05240 | Robust Reasoning as a Symmetry-Protected Topological Phase (Sung) | Holonomic (non-Abelian gauge) network keeps fidelity 100x beyond training. | Same idea as Howe's −b (input only via group-valued transport). Single author, strong claims: cite with care. |

**Verdict for focus (c):** an error-correcting contraction around a discrete code is **known as a training target** (SDRNN, SR-RNN), and its
necessity is **known as theory** (Chung). Combining it with an *isometric* transport on the complementary subspace, as a *seed-reliability* fix, was not
found.

## E. Noise-driven basin selection: theory (focus d) and symmetric points (D-list items)

| Source | Title (exact) | Result used |
|---|---|---|
| math/0701569 | Exit asymptotics for small diffusion about an unstable equilibrium (Bakhtin) | As noise → 0, exit concentrates along the top unstable direction, with exit time ~ λ⁻¹ ln(1/ε) (refines Kifer 1981). Follow-ups: 1708.00558 "Scaling limit for escapes from unstable equilibria in the vanishing noise limit: nontrivial Jordan block case"; 1911.06437 "Atypical exit events near a repelling equilibrium" (atypical exits have polynomially small probability). |
| math/0008208 | Pathwise description of dynamic pitchfork bifurcations with additive noise (Berglund, Gentz) | Sweep speed ε, noise σ: paths stay within σ/ε^{1/4} of the unstable branch until √ε after the bifurcation, then leave during [√ε, c√(ε\|log σ\|)]. |
| adap-org/9707008 | A noise-controlled dynamic bifurcation (Lythe) | If the passage is slow enough, **the dynamics are controlled by additive noise "even if they are extremely small"**. Gives the exit-value distribution. Multiplicative noise does not produce this. Also adap-org/9707007 "Noise and dynamic transitions". |
| Physica D 21:296-306 (1986), no arXiv | Kondepudi, Moss, McClintock, "Observation of symmetry breaking, state selection and sensitivity in a noisy electronic system" | Branch selectivity from a small additive bias during a sweep. Via Juel et al. 1997 **[ft]**: the probability of the favoured state "increases with the magnitude of the disconnection parameter and decreases when the sweeping rate is raised". |
| Proc. R. Soc. Lond. A 453:2627-2647 (1997), no arXiv | Juel, Darbyshire, Mullin, "The effect of noise on pitchfork and Hopf bifurcations" | "In the case of the pitchfork we find that natural imperfections override the effects of the noise" **[ft]**. |
| 1101.2899 | A mathematical framework for critical transitions: bifurcations, fast-slow systems and stochastic dynamics (Kuehn) | Fast-slow plus noise framework: critical slowing down as an early-warning sign. |
| PRE 52:4225-4243 (1995), no arXiv found | Saad & Solla, "On-line learning in soft committee machines" | A symmetric (unspecialised) fixed point traps the dynamics on a plateau. Symmetry breaking is then driven by the initial asymmetry. ML instance of (iv). Also cond-mat/0212006 "On-Line Learning Theory of Soft Committee Machines with Correlated Hidden Units - Steepest Gradient Descent and Natural Gradient Descent -" (correlation lengthens the symmetric phase; natural gradient removes the plateau); 1906.08632 (Goldt et al.); 2512.16556 (Afanah, Rosenow). |
| 2306.04251 (covered) / 2402.07193 | … / Parameter Symmetry and Noise Equilibrium of Stochastic Gradient Descent (Ziyin et al.) | Noise makes identical-neuron (permutation) invariant sets attractive. Noise drives a "Noether flow" to init-independent noise equilibria. | **Warning for (iv):** two channels calibrated to equal norms sit near an identical-unit invariant set, which SGD noise can make *attractive* rather than unstable. |
| 2002.03495 / 2006.15191 / 1805.08522 / 1511.04210 | A Diffusion Theory For Deep Learning Dynamics: Stochastic Gradient Descent Exponentially Favors Flat Minima / Is SGD a Bayesian sampler? Well, almost / Deep learning generalizes because the parameter-function map is biased towards simple functions / On the Quality of the Initial Basin in Overspecified Neural Networks | Kramers-type selection: P_SGD(f) ≈ P_B(f), i.e. basin volume. Simplicity bias of the parameter-function map. Probability of a good initial basin vs width. | These are basin-selection *probability* results, but for minima / functions, not for a schedule-driven bifurcation in a recurrent branch. |
| 2011.09468 / 2309.07311 / 2006.07710 / 2210.10749 / 2303.11873 / 2002.06305 | Gradient Starvation: A Learning Proclivity in Neural Networks / Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs / The Pitfalls of Simplicity Bias in Neural Networks / Transformers Learn Shortcuts to Automata / A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks / Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping | Early-learned features starve later ones. **"briefly suppressing SAS improves model quality"** (2309.07311): a competing simple strategy is suppressed early. Shallow shortcuts to automata. Competing circuits. **Data order alone** gives seed variance comparable to init in pretrained fine-tuning. | s3 being *ahead* at 8k and then plateauing is the shortcut-first signature. 2309.07311 is the closest "suppress the shortcut early" intervention. 2002.06305 confirms data-order-only lotteries in pretrained LMs. |

**Selection-probability formula (derived here from the standard linear SDE; not quoted from a paper; consistent with Kondepudi's qualitative result):**
- *Dynamic pitchfork, linearised.* Take dx = (r·t·x + h) dt + σ dW, crossing the critical point at t = 0 with sweep rate r, bias h and additive noise σ.
  - Then e^{−r t²/2} x(t) → Z ~ N( h·√(2π/r), σ²·√(π/r) ).
  - So **P(favoured branch) = Φ( √2·π^{1/4} · h / (σ · r^{1/4}) )**.
  - With h = 0 (a symmetric or separatrix crossing), P = 1/2 *whatever r and σ are*. Slowing the sweep only amplifies an existing bias, and weakly
    (∝ r^{−1/4}: a 16x slower ramp halves the bias needed).
  - P ≥ 0.98 needs h ≳ 1.06·σ·r^{1/4}.
- *Static unstable point.* Take dx = (λx + h) dt + σ dW from x0. Then **P = Φ( (x0 + h/λ)·√(2λ)/σ )**.
  - A deterministic init offset by a few σ/√(2λ) to the good side of the separatrix makes selection near-certain.
  - A calibrated symmetric init (x0 = 0, h = 0) is the worst case.
- *Assumptions:* additive Gaussian noise near criticality, and the branch fixed by sign(Z) before saturation. SGD noise is anisotropic and multiplicative,
  and Lythe says multiplicative noise behaves differently. Stochastic collapse can turn the symmetric set from repelling into attracting.
- *Mapping (a conjecture):*
  - x = an order parameter for "progress subspace neutral vs contracting";
  - r = the rate at which the γ ramp (2000 → 6000) and the fade move the pass map through criticality;
  - σ = data-order noise projected on x;
  - h = the deterministic drift toward the serial branch at criticality.
- *Falsifiable consequences:*
  1. A slower γ ramp alone leaves P near its current value.
  2. Seed-to-seed outcome is set by bit-level noise (as observed for GPU class) iff the crossing is slow and h ≈ 0. Lythe's regime predicts exactly this.
  3. A designed tilt h > ~σ·r^{1/4} on x makes P → 1, and P is monotone in the tilt.
  4. A larger batch (smaller σ) raises P only if h > 0.

## Novelty verdicts

- **(i) Isometric (Householder β = 2, α = 1) update on a content subspace U plus a separate error-correcting contraction: PARTLY KNOWN.**
  - Isometric transport with input entering only through transitions fixes learnability and length generalisation in Householder LRNNs: Howe 2609.18966.
    Bilinear support: 2505.21749.
  - An error-correcting contraction must be state-dependent: Chung 2605.07755.
  - Attractor clean-up as an aux-trained target: 1805.08394, 1901.08817.
  - Contraction toward a manifold: Wang & Slotine 2005.
  - A uniform isometric plus damping split: 1902.09689.
  - **New:** the U / U⊥ split as a cross-pass memory in a looped pretrained LM, justified and evaluated as a *seed-reliability* method.
  - Howe's result also implies that β = 2 by itself is not enough: the +β v kᵀ and additive-head injections onto U must also be removed or confined to U⊥.
- **(ii) Pass-map Jacobian / Lyapunov target, no contraction before completion and contraction after: PARTLY KNOWN.**
  - λ → 0 to raise success rate, with continuous application harmful: 2312.17306.
  - A forward-time-windowed FTLE regulariser: 2602.09613.
  - The post-completion directional-contraction signature, observed: 2609.26487.
  - **New:** a completion-indexed, item-dependent target (t ≤ n(x) vs t > n(x)) on a task-defined subspace U, aimed at basin entry.
- **(iii) Homotopy in a contraction parameter: KNOWN as a lever.**
  - Continuation: Hazan 1503.03712 (in the earlier note). Teacher-forcing interpolation: 2306.04406. Episodic flossing: 2312.17306. Budget annealing:
    2607.20594 (earlier note).
  - Noisy-bifurcation theory (Lythe adap-org/9707008; Berglund & Gentz math/0008208; Kondepudi 1986) predicts **it can't fix a lottery whose crossing is
    symmetric**: slow passage is noise-controlled. It helps only together with a bias.
- **(iv) Symmetry breaking at a designed symmetric init (two channels with equal read norms): PARTLY KNOWN, and risky as stated.**
  - Symmetric points cause plateaus and noise-decided selection: Saad & Solla 1995; cond-mat/0212006.
  - Noise can make identical-unit sets attractive: 2306.04251, 2402.07193.
  - An imperfection overrides the noise: Juel 1997; Kondepudi 1986.
  - An init-trap lottery fixed by moving the bias off the trap: 2604.21999.
  - **New:** deliberately *unfolding* the designed symmetric point with a measured tilt h, sized against measured σ and r with the formula above, so that
    basin entry becomes a designed event in a looped LM memory.

**Single most defensible novel claim:**
- *Claim:* in a looped LM whose cross-pass memory acts affinely on the progress subspace U, the seed lottery is a choice between Chung's two horns:
  neutral on U (path-dependent, no error correction) or contracting on U (fading memory, per-pass decay). A parametrisation that is exactly isometric on U
  (no additive injection onto U), with a state-dependent contraction only on U⊥ around a discrete code, **removes the contracting horn by construction**.
  Then P(success) no longer depends on SGD or data-order noise.
- *Why it's defensible:*
  - It is checkable: λ_U = 0 by construction, and s3-type delta_cos ≈ 0.98 becomes impossible.
  - Its nearest neighbours (Howe 2609.18966, Chung 2605.07755) are from-scratch sequence-time LRNN results with no basin-probability or looped-LM claim.
- *Caveats:*
  - Isometry on U also keeps noise on U. All denoising falls on the U⊥ snap (Vorontsov 1702.00071, and 2408.00109 on fragility).
  - The curse of memory (2405.21064) predicts higher sensitivity to the learning rate on U.

## Cheap checks the literature suggests (no new runs needed for the first two)
1. Is s3's contraction on U (the aux-decodable directions) or on U⊥? Use relative Lyapunov growth restricted to U from saved checkpoints. Use spectral,
   not operator-norm, quantities (2607.10681).
2. Make an "h ≈ 0" estimate. Find the γ-ramp step at which the s1 / s3 trajectories diverge on a chosen order parameter (e.g. delta_cos or the U-restricted
   step ratio). Compare the divergence with the across-seed spread at that step. Lythe's regime predicts the divergence begins inside the noise band.
3. Howe-style ablation at inference: on the converged seeds, zero the additive-head contribution to U at inference and check depth at n 9-12. Howe saw
   that zeroing the parasitic path *restores* generalisation.
