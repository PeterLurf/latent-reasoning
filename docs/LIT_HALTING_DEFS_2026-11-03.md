# Halting / settling definitions in Fixed-Point Reasoners (2606.18206) and Viakhirev (2608.18222), against §264's rel_inc / τ rule (2026-11-03)

Written 2026-11-03 by the Literature session, for Theory (§264 SCORECARD-CHECK). Both papers read in their arXiv HTML.

## Fixed-Point Reasoners, Movahedi et al. 2606.18206 (FPRM)
- **Residual:** r_i = ‖z_i − f_θ(z_i; x)‖_∞ / (‖f_θ(z_i; x)‖_∞ + ε). A **relative** fixed-point residual in the **max-norm**: the gap between the state and its own next update, normalised by the update's size.
- **Halting:** iterate until r_i < τ. Experiments use **τ = 0.1** (e.g. the adaptivity figures halt a group when its **mean** residual < 0.1).
- **Damping and patience:** z ← η·f(z) + (1 − η)·z. If r* = min_{j ≤ i} r_j has not improved for P iterations and r > τ, then η ← γη. Training also stops a supervision step when r < τ or η becomes too small.
- **Related-work definition:** stop when ‖z_{i+1} − z_i‖ ≤ ε (absolute successive-iterate distance, DEQ style).

## Viakhirev et al. 2608.18222
- **Regime ("settle"):** from the **top finite-time Lyapunov exponent** λ_max (top-8 spectrum via Benettin QR) of the trained operator:
  - **settle:** λ_max < −0.05;
  - **marginal:** |λ_max| ≤ 0.05;
  - **drift:** λ_max > +0.05.
- **Settle ratio:** measures the per-step displacement δ_t = ‖z_{t+1} − z_t‖, an absolute norm.
- **Solve steps:** "the first iteration at which the decoded answer stops changing".
- **Depth-safety (Proposition 1(a)):** if the remaining path R_t = Σ_{u ≥ t} δ_u is below the decoder margin μ(z_t), the decoded answer is frozen for all s ≥ t.
  - Corollary 1: with geometric decay δ_u ≤ Cγ^u, the freeze time is t* = ⌈log(C / ((1 − γ)μ)) / log(1/γ)⌉.
  - (b): a finite window of unchanged answers certifies nothing if R_t = ∞ ("guess-freezing").
- **Objective:** a penalty on a_T·‖f(z_{T−1}, x) − z_{T−1}‖ at the supervised horizon.

## Against §264
§264 uses rel_inc_t = ‖S_t − S_{t−1}‖_F / ‖S_{t−1}‖_F; c = median over compute passes, h = median over hold passes; CONTRAST iff c ≥ 3h; τ = √(c·h).
- **Closest to FPRM's r_i**, which is also a relative step residual. The differences:
  - Frobenius (all tensors) vs ∞-norm;
  - the previous state vs f(z) in the denominator;
  - a **per-model data-derived τ** (the geometric mean of compute and hold levels) vs FPRM's fixed **τ = 0.1**.
  - Suggested citation: "a relative-increment settle criterion in the spirit of FPRM's residual halting (2606.18206, τ = 0.1 fixed), with τ set per model as √(c·h)".
- **Viakhirev's regime labels are operator-level** (λ_max), not per-pass increments. §264's c / h contrast is a trajectory-level proxy, not a Lyapunov estimate. Don't call it "settle" in Viakhirev's sense without a λ_max reading.
- **Viakhirev's Prop. 1(b) warns that small increments over a finite window do not certify a held answer.** That supports §264 pairing rel_inc with an accuracy condition (COMPLETION: acc at K′ 24 ≥ STD − 0.02).
