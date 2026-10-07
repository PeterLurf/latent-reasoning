# Prior check for DISTILL-2: distilling a CoT teacher's *probe-decoded* state into the FSQ carrier (2026-10-07)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-07) or earlier notes.

## 1. Precedents
- **Soft-target distillation:** Hinton et al. 1503.02531.
- **Intermediate "hint" distillation:** FitNets 1412.6550 (the student's intermediate layer regressed onto the teacher's
  hint layer).
- **Intermediate concept targets:** Concept Bottleneck Models 2007.04612 (the latent must predict human-defined concepts
  before the output).
- **Latent CoT:** ICoT-KD 2311.01460 (raw teacher hidden states), KaVa 2510.02312 (KV trajectories), CODI 2502.21074.
  Our §111 READ (same structure, gold S_j targets) and §135 (raw-hidden distillation, which failed).
- **Not found:** distilling a teacher's *probe-decoded per-digit state distribution* into a discrete latent carrier.
  **New as scoped.** The closest combination is FitNets hints + concept bottleneck targets.

## 2. KL to the probe vs CE to the teacher's decoded text
- **With an accurate teacher they nearly coincide.** TC-M is about 1.0 on 4×4, so its generated S_j text is the true
  S_j in distribution, and the probe's distribution is near one-hot. Soft targets carry "dark knowledge" (Hinton) only
  where the teacher is uncertain.
- **The probe can only add error:** a linear probe below 100% injects probe noise that the teacher's own text doesn't
  have.
  - **Measure probe accuracy against the teacher's decoded S_j first.** If it is < 0.99 per digit, use CE on the
    teacher's decoded text instead. It is cleaner and equally C4-compatible (the teacher's own CoT).
  - The probe is worth its complexity only where the teacher's text is unavailable or uncertain (e.g. beyond its
    training size).
- **Implication for interpretation:** with a perfect teacher, B ≈ "READ with teacher-generated S_j targets". A pass
  shows that a CoT teacher's state can replace gold labels (revised C4). It doesn't show that hidden-state distillation
  adds anything over text. Word it so.

## 3. Fading the auxiliary weight to 0 by 4k
- **Less risky than §126 ANNEAL.** Fading an *auxiliary target* is like ICoT-SI / Coconut removing supervision while the
  final-answer loss continues. ANNEAL removed the *carrier path* itself.
- **Two pitfalls:**
  - (a) If 4k comes **before the curriculum reaches the full multiplier length**, the longest S_j hand-offs never get the
    auxiliary. Fade only after L = 4 has been trained for a while, or tie the fade to the curriculum stage.
  - (b) After the fade, the code can drift off the S_j format if the final loss is met by the b-prefix shortcut
    (§121). **Log the readout head's S_j accuracy after the fade.** A decay means the state wasn't retained.
  - A constant-weight arm (as §111 READ) is the natural control if budget allows.

## Also
- **Design A is multiplier-major** (the state grows to 26.6 bits; out of range beyond 4×4). That's fine as the
  comparator for B, but no length generalisation is expected (LIT_MULT_PLAN recommended column-major for that).
- **SOFT-16 capacity:** confirm that its grid holds ≥ 26.6 bits (16 dims × log₂ levels).
