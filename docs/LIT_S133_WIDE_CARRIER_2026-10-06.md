# Prior check for §133: keeping a wide carrier exact (CONT / FSQ / SOFT / NOISE / PIN / JAC at d = 64) and an S5×4 capacity test (2026-10-06)

Literature session, for Theory, on the user's question via Audit; Audit copied. Brief. IDs verified via the arXiv API
(2026-10-06). **1802.08227 is not a contraction paper** (it is a quantum linear-systems primer), so don't cite it.

## (a) Priors
- **Soft or annealed quantisation:** soft-to-hard VQ, Agustsson et al. 1704.00648; FSQ 2309.15505; the dither proxy
  (Ballé 1611.01704; universal quantisation 2006.09952). Internal: §101's partial snap (β 0.01 sufficed on HYB-4).
- **Contraction and stability of recurrent maps:** Lipschitz RNNs 2006.12070; Stable Recurrent Models 1805.10369;
  contractive penalties (Rifai 1104.4153; Jacobian regularisation, Hoffman et al. 1908.02729); exploding/vanishing
  analysis 1712.05577.
- **Norm constraints:** nGPT 2410.01131 (hypersphere); LayerNorm pins.
- **Verdict:** new as scoped. A width sweep of continuous, hard, soft-snapped, dithered, norm-pinned and contraction-
  penalised carriers in a pretrained LM's latent chain, judged on exactness at 2,048 plus a certificate-style drift
  measure (ρ), isn't found.

## (b) Traps
1. **SOFT-64 at β = 0.1.**
   - The snap moves z 10% of the way to its grid point at each hand-off. The network re-injects its own error each
     step, so whether SOFT acts like hard FSQ depends on the learned map.
   - **Report the grid share** (the distribution of the distance to the nearest grid point at long lengths). Also
     report ρ₁ vs CONT-64: SOFT is informative only if it sits *between* CONT and FSQ on that distribution.
2. **JAC-64: a global contraction penalty conflicts with exact state carriage.**
   - S5's true carry is an *isometry* among the 120 states. A map contractive in *all* directions (‖J‖ ≤ 0.5) merges
     states, and "stability ⇒ forgetting" (Stable Recurrent Models 1805.10369).
   - Exactness needs **anisotropic** stability: contraction off the state manifold, isometry along it (2609.26487
     measured exactly this in finished recursions).
   - A random-projection penalty on ‖J‖ can't separate these, so expect JAC to cost exactness.
   - **If you run it, penalise only along noise directions** (CAE-style, at data points with small perturbations), or
     keep it optional and report it as a contraction-vs-memory test.
3. **CERT at d = 64.** Enumerating reachable codes was feasible at d = 8 (537k in §120). At d = 64 the reachable set may
   be far larger and non-canonical, so certification by enumeration may be infeasible.
   - Pre-register the fallback: "exact at 2,048 + ρ₁" as the pass, with CERT / Moore minimisation only where the
     reachable set closes.
4. **PIN-64:** LayerNorm fixes norm drift (RAW-H's) but not *directional* drift. Expect partial help. Report the
   angular drift too.
5. **S5×4 learnability.**
   - With FINAL supervision and one queried register, a register's transitions get gradient only when that register
     is queried, i.e. 1/4 of the data plus distractor items.
   - **Add a register curriculum (1 → 2 → 4)**, or query all registers at the end (still final-only supervision,
     4 targets).
   - **S5×2 (14,400 states) is a needed in-capacity control.**
6. **Is S5×4 a fair capacity probe?**
   - Required: log₂(120⁴) ≈ 27.6 bits. FSQ-8 at 8 levels: 8 × 3 = 24 bits. So FSQ-8 *cannot* represent it, and
     S5×2 (13.8 bits) fits. Good.
   - But an FSQ-8 failure on S5×4 could be learnability, not capacity. **Add FSQ-8 with 16 levels** (32 bits ≥ 27.6)
     as the capacity-matched narrow control. If it passes, capacity was the limit; if it fails too, difficulty was.

## (c) Width
- A single d = 64 confounds "wide" with "very wide". **d ∈ {16, 64} for at least FSQ and SOFT** gives a dose (FSQ-16 at
  8 levels = 48 bits, ample for S5×4).
- If budget forces one width, take d = 16. It is the smaller step from the working d = 8 and already beats S5×4's
  capacity need.
