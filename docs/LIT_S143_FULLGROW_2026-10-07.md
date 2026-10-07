# Prior check for §143 FULL-GROW: FSQ-8 → 768 (8 FSQ + 760 dithered continuous) after THRESHOLD (2026-10-07)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-07) or earlier notes (LIT_S136_GROW).

## (a) Is there a precedent for about 100× widening in one step?
- **None found.** Growth in practice is about 2× per step, staged:
  - Net2Net 1511.05641; bert2BERT 2110.07143; LiGO 2303.00980;
  - MSG 2305.02869 (masked structural growth, staged with schedules);
  - FLM-101B 2309.03852 (16B → 51B → 101B);
  - ProGAN 1710.10196 (doubling, with fade-in).
- **Staged growth is the known safe practice, but the ratio itself isn't what breaks.** With W_up's new columns at 0 the function is
  preserved at any ratio. What scales with the ratio is the **training dynamics after growth** (b).
- **The cheap middle ground:** 8 → 64 is already validated (§136 4/4). One further 12× step (64 → 768) is closer to precedent than
  96×.

## (b) Known failure with zero-init outgoing weights and many new units
- **Zero-init outgoing weights are fine in themselves:** ReZero 2003.04887 and Fixup zero-init whole residual branches; GradMax
  2201.05125.
  - At growth, the new W_down rows get **exactly zero gradient** (it flows through W_up = 0). Only the W_up columns move first, so
    the dynamics of the step are set by W_up's learning rate.
- **The real risk is the learning rate's scale with fan-in (μP, 2203.03466).**
  - Under Adam each weight moves about lr per step, whatever the size of its gradient.
  - W_up's fan-in goes from 8 to 768. The change in its output per step grows roughly in proportion to the fan-in: up to about 96×
    larger at the same lr, because Adam's updates across the new columns are correlated.
  - μP prescribes an Adam LR ∝ 1/fan-in for such matrices.
  - **Recommendation:** set the new W_up columns' lr to (8/768) × the base lr (or apply μP scaling to the whole W_up). Keep the
    300-step warm-up and the 500-step fade-in on top.
  - Without this, expect a loss spike or a shift in h shortly after the fade-in completes, not at growth itself. A warm-up delays
    the effective-step problem but doesn't remove it.
- **Fresh Adam moments:** the warm-up handles the first steps' bias-corrected large updates (as LIT_S136 said). Keep the old
  parameters' state as is.

## (c) Do wide continuous latents stay stable after growth?
- **No literature specific to grown latent carriers.** Our own evidence (§133 NOISE-64; §136 GROW-NOISE 4/4) is the strongest
  available.
- **The structural risk is the 760 continuous dimensions, which have no rounding correction** (only training dither). Drift there
  enters through W_up, exactly RAW-H's failure. The first 8 FSQ dimensions can't correct it.
- **Dither scale:** U(±0.25) on 760 dimensions has total norm about 0.25/√3 × √760 ≈ 4, comparable to the 8-d code's own norm.
  - That's harmless while W_up_new ≈ 0. It is the main training-time noise once W_up_new grows.
  - It may push W_up to keep the new dimensions small. On S5 (8-d already suffices) that's the expected null by design (LIT_S136).
- **Log:**
  - ‖W_up_new‖ and the new path's share of the W_up output;
  - ρ split into the FSQ-8 block and the continuous block;
  - R² of state on the new block (as §136).
  - Apply LIT_S138's ρ cautions: at 768 the random-δ ρ is a bulk average; add a centroid-direction ρ, descriptive only.
- **Benefit:** S5 can't show gain. Growth is useful only if the grown carrier then passes where 8 / 64 fails on a capacity task
  (§138 rule ii).

## Verdict
- **New as scoped:** function-preserving growth of an LM's latent carrier to residual width, hybrid FSQ + dithered continuous,
  scored by ρ, SERIAL and exactness at 2,048.
- **Single-step 96× growth is unprecedented.**
- **Two recommendations:**
  1. μP-style lr scaling for the new W_up columns (the main pitfall);
  2. optionally, a staged 8 → 64 → 768 arm to separate the ratio from the dynamics.

## Addendum (2026-10-07): the combined design (COPY+GROW-NOISE 8 → 64, then 64 → 768 at ≥ 6k)
1. **lr scale: keep 64/768 on the 704 new columns only. The old 64 columns stay at the base lr.**
   - The reference fan-in is 64, where the base lr was validated (cnoise 4/4).
   - The new block's correlated output step is then 704 × (64/768) ≈ 59 lr-units, against the old block's 64. The total is about 2×
     the validated step, which is acceptable. 64/704 differs by 9%, which is immaterial.
   - μP on the whole W_up would change the old columns' lr at switch 2 and break the paired comparison with s136_cnoise. Don't.
   - W_down's fan-in (h, 768) is unchanged, so it gets no scaling.
   - Note: the new W_down rows get gradient only through W_up_new, so at 64/768 recruitment is slow. That's expected, not a failure.
2. **The "carried norm ≥ 2× pre-switch" criterion fires by construction.** Adding 704 tanh-bounded dims at default-init W_down raises
   ‖z_b‖ at switch 2 with no expansion at all.
   - Measure the norm on the old 64-block, or on the embedded code e = W_up z fed to the trunk (function-preserved at the switch).
     Don't use the full z_b.
3. **rho_k3 is deflated by dims the model ignores.**
   - `contraction()` perturbs all coordinates by 0.1·std_d. While W_up_new ≈ 0, the 704-block perturbation enlarges D_0 but doesn't
     propagate, so ρ_1 (and the geomean) reads small.
   - The registered rule stays as is. Add, descriptively: ρ_k with the perturbation restricted to the old 64-block, and ρ_k measured in
     e-space. The "AVOIDS" reading should rest on these and on the Jacobian top, not on rho_k3 alone.
   - The same artefact applied in part to the 56-dim block at 64-d.
4. **Pairing:** "identical up to switch 2" holds only on the same GPU class and driver as each s136_cnoise seed. Check that the loss
   records are byte-identical up to the switch before treating it as paired. Otherwise, report the pre-switch reference measured on
   the new run itself.
5. **Use fp32** for `contraction()` and the Jacobian power iteration (as for §138).
6. **A pass is stability-only.** S5 can't show gain (rule ii), so state "stable at 768", not "benefit".
