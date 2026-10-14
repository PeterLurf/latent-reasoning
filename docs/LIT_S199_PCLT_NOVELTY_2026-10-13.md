# Formal novelty verdict for §199 PCLT (the progress-cone Lyapunov target), on draft amendment 199-A (2026-10-13)

Literature session, for Theory (the request) and Audit.
- **Read:** Theory's scratchpad am199_A_draft.md, and the subagent scan LIT_contraction.md (its summary and the tables it relies on). For MATH_contraction.md and MATH_selection.md I read only the passages cited below.
- **IDs:** every ID below was re-verified by me via its arXiv abstract page (2026-10-13). Claims are from abstracts unless noted.
- **Extends:** LIT_SEED_STABILITY(_CANDIDATES)_2026-10-13 and LIT_PASS_LYAPUNOV_2026-10-13.

## Verdict per component

| component | verdict | closest prior |
|---|---|---|
| (a) the measured quantity: λ_t = log‖R_{t+1} − R_t‖ − log‖R_t − R_{t−1}‖ along the realised trajectory | **KNOWN** as a diagnostic | **2607.09698** (Duraipandian et al.), Eq. 3: λ(t) = log(‖z_{t+1} − z_t‖ / ‖z_t − z_{t−1}‖), the "Lyapunov sensitivity" of latent CoT, computed without re-running. PCLT computes the same step-size ratio on the block-6 read instead of the latent. Krishna 2609.26487 measures contraction *along the trajectory direction* after completion (observational). |
| (a′) used as a **training target** in read space | **PARTLY KNOWN** | Gradient Flossing 2312.17306 (all exponents of the full state, toward 0, at all times; reports a higher success rate); STARS 2605.26733 (convergent throughout); Bai 2106.14342 (the full Jacobian at the equilibrium); Parcae 2604.12946 (contractive injection by design). None targets the trajectory step ratio in a decoder-visible read. |
| (b) the sign switch at each item's completion step (floor for t < n, ceiling for t ≥ n) | **NEW** (not found) | Wöhrer & Kuehn 2602.09613: FTLE suppression in an *early window of forward time*, for robustness, the same for every input. Krishna 2609.26487 observes the post-completion shape but sets no target. Fixed-Point Reasoners 2606.18206 and Attractor Models 2605.12466 converge throughout, with halting by convergence. I found no 2024–26 looped or latent-reasoning paper with a per-item, **completion-indexed** dynamics target. |
| (c1) the selection law P = Φ(μ_Z/σ_Z), with bias scaling linearly and noise levers as √ | **KNOWN in form** | Branch selection at a noisy (imperfect) pitchfork: the probability is set by bias relative to noise and falls with sweep rate (Kondepudi, Moss & McClintock, Physica D 21, 1986; via LIT_contraction's [ft] read of Juel et al. 1997). For a linearised unstable SDE from m₀ ≈ 0, Φ(mean / sd) is a textbook Gaussian computation. **I haven't checked the exact Kondepudi formula myself.** |
| (c2) applying (c1) to the SGD seed lottery of a pretrained looped LM, as a "basin-selection tilt" | **NEW as framing** | Not found. The neighbours are Zhang 2607.20594 (an empirical lottery; budget annealing) and Howe 2609.18966 (seeds per cell; no probability theory). The threshold / race mapping is CONJECTURE (Theory's own tag), so the claim must stay conditional on it. |
| (c3) Thm A2: the gradient of λ_t acts on log κ directly, so the escape signal is not κ^t-attenuated | **PARTLY KNOWN** | This is Gradient Flossing's own motivation (regularising the exponents gives a direct, non-vanishing gradient path). New only in the scalar basin-escape form. |
| the pass-token lemma (a pass-indexed aux decoder can produce the step targets without the memory carrying progress, so contraction goes unpenalised) | **PARTLY KNOWN** | The principle is the VLAE "information preference" / posterior-collapse argument: a decoder strong enough to model the target without the latent ignores it (Chen et al. VLAE 1611.02731; Bowman et al. 1511.06349). 2607.17674 applies it to latent *strategy* variables in LMs ("the generator … already represents p(y\|x) without using z"). In-house: §187 index-free targets; Liang 2609.33144's progress cue. **Applying it to a recurrent latent memory with a pass-index token: not found.** Timestep encoding in looped transformers is argued to *help* expressivity (Xu & Sato 2410.01405), the opposite angle, and should be cited. |
| the weak-decoder (posterior-collapse) analogy for a shallow step reader | **PARTLY KNOWN** | As above. The analogy is standard for VAEs; I found it applied to recurrent latent memories nowhere. |

## Defensible claim line
"PCLT is, to our knowledge, the first dynamics regulariser for a looped / latent-reasoning model whose sign is indexed by each item's completion step. It sets a floor on the read-space step ratio before completion and a ceiling after. It is motivated by a classical bias-versus-noise branch-selection law, which we apply, conditionally on a one-dimensional selection coordinate, to the seed lottery of a pretrained looped LM."

**Wording to avoid:**
1. **"Lyapunov exponent" or "Lyapunov target", unqualified.**
   - λ_t is a step-size ratio along the clean trajectory (2607.09698's surrogate), not the growth of a perturbation. That's the correction recorded in LIT_PASS_LYAPUNOV.
   - Say "step-ratio target", or "a trajectory surrogate of the finite-time Lyapunov exponent".
2. **"Only the progress direction".**
   - R_{t+1} − R_t is the realised displacement of the decoder-visible read, not verified progress. Any visible motion counts, including oscillation or drift that doesn't help.
   - Say "realised read displacement".
3. **"Removes the noise-decided threshold".** It holds only in the scalar model (Thm A2); don't state it for the network.
4. **"Raises P(enter)".** Theory's own A3 claim limit already says this; I agree.

## Design checks for Audit (outside novelty; they can make the floor vacuous or impossible by construction)
1. **What carries λ_t in steps 500–2000 (γ = 0, identical INJECT input each pass)?** The write ignores the read, so every pass adds the same additive increment φ(k)vᵀ.
   - **Under rms (`boxes/deltamem.py` l.55 / l.153: each head's read normalised to unit RMS, times a gain):** S_add,t = t·ΔS, so the normalised additive read is the same at every t ≥ 1 and ΔR_add = 0. The additive half contributes no displacement, so the floor falls entirely on the delta heads (contractive at init: α^T, β = 1) and on W_o.
     - That is probably the intended lever, but it should be stated. The "realised direction" during this window is delta-head motion only.
   - **Under the raw (None) recipe:** a constant ΔR_add gives λ = 0 ≥ log 0.95, so the floor is met trivially, whatever the progress. Don't port PCLT to non-rms recipes without a per-head-type λ.
2. **A per-head-type λ log** (additive vs delta read contributions) is the cheap guard for both cases. Also add to the λ fixture a constant-increment state through the actual rms read path (expected ΔR_add = 0).
3. **d_t is bounded below by ε₀‖R_t‖.** Under rms ‖R_add‖ is fixed by the gains; under raw it grows with the state. If the true displacement is about 0 (as for ΔR_add above), this floor term sets λ, and λ ≈ log(‖R_{t+1}‖ / ‖R_t‖). The fixture should confirm the term stays negligible when there is real motion.

## Also relevant (the subagent scan, verified)
- **Chung 2605.07755:** affine trackers can't correct errors on the state-separating subspace once they preserve it.
- **Howe 2609.18966:** in Householder linear RNNs the additive input path is a parasitic attractor; deleting it gives exact automata.
- **Miller & Hardt 1805.10369:** stable recurrent models ≈ feed-forward ones, which is the formal reason a contractive pass map has bounded effective depth.
- All are central to the framing and correctly cited in the draft.
