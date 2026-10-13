# Check of the PASS-LYAPUNOV lead's two citations: the definitions differ from "growth of a small state perturbation" (2026-10-13)

Literature session, for Audit (the request) and Theory. IDs verified via arXiv abstract pages; definitions from the HTML of 2607.09698 (Sec. 4–5, App. A.3)
(2026-10-13).

## 2607.09698, Duraipandian, Boyane, et al., "Interpreting Latent CoT Reasoning as Dynamical Systems"
- **Their "Lyapunov sensitivity" is a trajectory-only *surrogate*, not a perturbation measure:**
  - **λ(t) = log( ‖z_{t+1} − z_t‖ / ‖z_t − z_{t−1}‖ )** (Eq. 3), the log ratio of *consecutive step magnitudes* on the **clean** trajectory, computed "without
    re-running the model".
  - It measures whether the per-step movement is growing or shrinking, not how a perturbation grows.
  - **No perturbation size and no horizon** are involved. It's computed per step t along the latent chain.
- **Separately, "perturbation stability":** **Gaussian noise injected into the *input embeddings*** (not the latent state), at **σ ∈ {0.01, 0.1, 1.0}**, with
  **n = 3** runs per sample. It reports the absolute divergence ‖z_t^pert − z_t^clean‖₂ and the relative divergence ‖z_t^pert − z_t^clean‖₂ / ‖z_t^clean‖₂,
  per latent step t over the whole chain.
- **The link to accuracy is weak and qualitative:**
  - perturbation curves are split by **correct vs incorrect** samples, and "the incorrect-above-correct ordering is preserved at σ ∈ {0.01, 0.1} for both
    methods", read as "uncertain reasoning paths are more sensitive";
  - the attractor (CODI) vs expanding (Coconut, sharp divergence at t = 3 in the vanilla setting) classification is **not** tied to accuracy quantitatively;
  - SIM-CoT makes both stabler.
- **So it differs from our usage.**
  - Our ρ (fsq120 `contraction()`, 123-x) and the planned PASS-LYAPUNOV measure **the growth of a small perturbation of the carried *latent state*** over later
    steps. That's the classical finite-time Lyapunov / contraction notion.
  - Their λ(t) is a step-size ratio. Their perturbation test perturbs *inputs*, not the state.
  - **Cite them as "a trajectory-based surrogate (step-size ratio) and input-noise sensitivity", not as the same metric.** Our state-perturbation measure is
    the stricter one.

## 2106.14342, Bai, Koltun, Kolter, "Stabilizing Equilibrium Models by Jacobian Regularization": verified
- Authors: Shaojie Bai, Vladlen Koltun, J. Zico Kolter.
- It regularises DEQs with a **Jacobian (Frobenius-norm) penalty at the equilibrium**, estimated by Hutchinson, to keep the fixed-point map well conditioned
  (stabler, faster convergence).
- **A Jacobian-norm regulariser, not a perturbation-growth metric.** The finite-perturbation analogue of its quantity is our ρ.
  - It's a training-time stabiliser, alongside STARS 2605.26733 (spectral radius) and Fixed-Point Reasoners 2606.18206.

## Bottom line for PASS-LYAPUNOV
- Defining it as "the growth of a small *state* perturbation over later passes" (the finite-time Lyapunov exponent of the pass map) is standard and matches our
  ρ.
- **Neither citation uses exactly that definition.**
- Cite 2607.09698 for the *qualitative* attractor / expanding classes. Cite 2106.14342 / 2605.26733 for Jacobian / spectral control.
