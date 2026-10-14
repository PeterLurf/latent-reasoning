# Early novelty pre-check of Theory's three seed-stability candidates (2026-10-13)

Literature session, for Theory (its direction message, 2026-10-13) and Audit.
- **Not the formal verdict.** That comes when the concrete proposal arrives. This is the closest prior work for each candidate.
- **IDs:** verified via arXiv abstract pages. Claims are from abstracts.
- **Code facts:** from `boxes/deltamem.py` (l.5, l.67–69).
- **Background:** LIT_SEED_STABILITY_2026-10-13.

## Code facts that bear on all three
- **Our delta heads already use Grazzi's range:**
  - β = 2·sigmoid(·) ∈ (0, 2) (l.5). That's the negative-eigenvalue parametrisation of 2411.12537.
  - **At init, β = 1 exactly** (gate weight 0, β-bias 0; l.69). A projection, I − kkᵀ, which fully erases the written key direction.
  - α = sigmoid(5) ≈ 0.993 per **token**.
- **So one pass is a product over its T written tokens:** α^T · Π_t (I − β_t k_t k_tᵀ).
  - α^T alone, before the projections: T = 30 gives ≈ 0.81 per pass (≈ 0.12 over 10 passes); a ProsQA-length T ≈ 300 gives ≈ 0.12 per pass.
  - T is the number of tokens that actually write: check the write masks per harness.
  - The delta half therefore starts contractive by construction. The additive half (S += φ(k)vᵀ, no decay) has eigenvalue 1 and only accumulates.
  - Theory's "contractive pass map in failed seeds" should be checked per head type. A delta-only contraction would point at the init (α, β).

## (i) An isometric delta update on a content subspace (β = 2 Householder / orthogonal transport) plus an error-correcting contraction around a discrete code
- **Closest prior:**
  - **Grazzi et al. 2411.12537:** β ∈ (0, 2) gives negative eigenvalues, so parity and state tracking become possible.
  - **DeltaProduct 2502.10297:** products of Householders.
  - Both claims are about **expressivity**, not trainability or seed reliability. That matches Theory's reading.
  - **Orthogonal / unitary RNNs for trainability:** uRNN 1511.06464; Householder parametrisation 1612.00188; expRNN 1901.08428; nnRNN 1905.12080 (unit-norm spectrum with non-normal transients for expressivity).
  - **Vorontsov et al. 1702.00071:** *hard* orthogonality can **hurt** convergence and *soft* constraints do better. Relevant to a hard β = 2 limit.
  - **Contracting-by-design RNNs:** Revay et al. 2004.05290, R2DN 2504.01250, Kozachkov 2106.08928. These give *global* contraction guarantees, the opposite regime.
- **What would be new:** splitting the state into an **isometric content subspace** and a **contracting error-correction around a discrete code**, framed as a fix for *seed reliability* in a pretrained looped LM. I found no paper with that split.
  - Continuous-attractor or discrete-transition networks (e.g. 2601.15336) and our own FSQ codes are the neighbours.
- **Risk:** isometric maps keep noise as well as signal (no forgetting). The error-correction half has to do all the denoising, and Vorontsov's hard-constraint result warns about optimisation.

## (ii) A pass-map Jacobian-spectrum / Lyapunov target on the aux-decodable subspace, with no contraction before completion
- **Closest prior, which must be cited:** **Engelken 2312.17306, "Gradient Flossing".**
  - It **pushes the Lyapunov exponents of the forward dynamics toward zero during training**, by backpropagating through differentiable QR, to fix gradient instability. That's very close to (ii).
- **Also:**
  - Vogt et al. 2006.14123 (the Lyapunov spectrum and trainability);
  - Chen & Pennington 1806.05394 (dynamical isometry in gated RNNs; gating extends the trainable timescale);
  - Bai et al. 2106.14342 (Jacobian regularisation at the DEQ equilibrium);
  - STARS 2605.26733 (spectral control in looped LMs, aiming **for** stable fixed points);
  - Hess et al. 2306.04406 / Eisenmann 2310.17561 (generalised teacher forcing; loss jumps at bifurcations).
- **What would be new:**
  - (a) restricting the target to the **aux-decodable subspace** (step-relevant directions);
  - (b) a **time-dependent** target: λ ≈ 0 until completion, then contraction, i.e. an isometric phase followed by a convergent one.
  - Gradient flossing is uniform, for the whole state, at all times. STARS wants convergence throughout.
- **The tension with Anil et al. 2211.09961 has to be addressed.**
  - **Path independence** (converging to the same fixed point from any start) **correlates with *better* upward generalisation** in equilibrium models.
  - Theory's finding is that "forgetting the start state" marks *failure*.
  - **The way to reconcile them:** Anil's models re-inject the input every iteration, and "forgetting the initial latent" means forgetting an arbitrary init. Our failure is forgetting the *accumulated progress* before completion, and INJECT re-injects the input too.
  - **So the proposal must state that the contraction it forbids is pre-completion contraction of progress directions,** not path independence after completion. Otherwise a reviewer will cite Anil against it.

## (iii) Breaking the designed symmetric init (g_add equal-read-norm calibration; β = 1)
- **Closest prior:**
  - symmetry-induced collapse and the **syre** fix (Ziyin 2309.16932, 2408.15495);
  - stochastic collapse onto invariant sets (Chen / Kunin 2306.04251);
  - permutation saddles (Brea 1907.02911; Simsek 2105.12221);
  - init-scale memory (2605.29152, CIFAR ResNets).
- **Not new as a principle** (symmetry → absorbing low-capacity states is established). **New in the specific:**
  - a zero-initialised gated memory branch in a pretrained LM where *the calibration itself* creates the symmetry (equal read norms) and β = 1 creates the projection;
  - with a stated mechanism linking that init to basin-entry *probability*.
  - Any syre-like fix should be compared with plain syre as the baseline.

## Summary for the formal check
- **(ii) has the closest prior: Gradient Flossing 2312.17306.**
  - Novelty needs the subspace restriction and/or the time-dependent (pre- vs post-completion) target.
  - It also needs an explicit answer to Anil's path-independence result.
- **(i) is the least anticipated as a combination.**
- **(iii) is new only in its specific mechanism.**
- **Before any of these: check whether the failed seeds' contraction sits in the delta heads** (α^T per pass at init: 0.81 at T = 30, 0.12 at T = 300; β = 1 projection) or in the additive heads. That's cheap to log from existing checkpoints.
