# Prior check for §104: step-specific blocking and per-slot interchange on §92/§97 checkpoints (C1) (2026-10-04)

Literature session, for Theory; Audit copied. Brief. IDs verified (arXiv abstract pages; earlier notes).

## Citations
- **Interchange intervention / IIA:**
  - Geiger et al., *Causal Abstractions of Neural Networks* 2106.02997 (interchange interventions verify that aligned
    representations have the causal model's properties).
  - DAS 2303.02536 (learned alignments scored by IIA; the NORTH_STAR C7 standard).
- **Causal tracing:** Meng et al. ROME 2202.05262 (corrupt-and-restore localisation).
- **Methodology and pitfalls:**
  - Zhang & Nanda 2309.16042: patching conclusions depend on the corruption method and metric; prefer in-distribution
    corruptions over Gaussian noise.
  - Heimersheim & Nanda 2404.15255 (a tutorial on interpreting patching).
  - Vaidyanathan et al. 2606.27510 (multiple mediators: interaction effects inflate or hide single-site effects).
- **Per-step interventions in latent CoT:**
  - Li et al. 2602.08783 (step-level do-interventions on Coconut/CODI; influence jumps rather than flowing step by
    step);
  - SCIT 2608.27265 (suffix-cache interchange in CODI-GPT2; computation travels in the value cache, not hidden states);
  - Zhang et al. 2512.21711 (causal and adversarial tests of Coconut's latent tokens);
  - Aswal 2606.12689;
  - ours: §34 (pointer edit IIA 0.36), §56 (aged-latent interchange), §45 (checkpoint edit IIA).

## Pitfalls
1. **Zero and mean ablation are off-distribution** (2309.16042).
   - Zero is far off GPT-2's manifold. Mean can land in a valid-looking but uninformative region.
   - Make **resample/interchange (donor) ablation** the primary blocking measure. Report zero and mean as secondary.
2. **Donor mismatch.**
   - Take donors from the **same slot index j** and the same chunk position (latents may encode position or "age";
     our §56).
   - Sample donor states to cover the state distribution, not just a few chunks.
   - For each donor, report whether its state is reachable at that slot.
3. **IIA inflated or masked by recomputation from visible ops** (text-fork models). After an interchange at slot j,
   score both outcomes:
   - **(i) donor-consistent:** the donor's state at j composed with the recipient's remaining ops (latent use);
   - **(ii) recipient-consistent:** the recipient's own original state (recomputation/bypass).
   - Their split locates where recomputation happens. A high (ii) is the bypass signature.
4. **Multiple mediators.**
   - The state is also present in op-token hidden states (§92's R2 note), so patching the slot alone can underestimate
     its role, and the downstream op positions may re-supply it.
   - Add a joint patch: slot j plus the op-token residuals between j and j + 1. Report single vs joint (2606.27510).
5. **Snap boundaries bound the effects.** An interchange can only propagate to the chunk's end (the snap resets). Score
   within-chunk slots and the post-snap chunk separately.
6. **Chance levels:** S5 element 1/120; per-box on the 5 × 5 head 1/5. Report both.
