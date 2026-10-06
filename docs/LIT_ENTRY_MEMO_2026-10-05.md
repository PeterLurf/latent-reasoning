# Prior check for THEORY_ENTRY_2026-10-05 (VICReg-style entry regulariser, §125-v2, restart baseline, C4 framing) (2026-10-05)

Literature session, for Theory; Audit copied. Brief. IDs verified (arXiv API / HTML 2026-10-05; LIT_INDEX). It folds
in the remedy map of LIT_COLLAPSE_REMEDIES (86df4b3a).

## (a) Variance/covariance regularisation against a decoder-ignores-code basin: priors and novelty
- **VICReg 2105.04906** (variance hinge + covariance penalty against collapse) and **Barlow Twins 2103.03230**
  (redundancy reduction).
- **Closest application to latent dynamics: PLDM, 2502.14819** (a JEPA latent-dynamics model for planning). It uses a
  VICReg-based objective and **applies the variance term across the time dimension "to encourage features to capture
  information that changes, as opposed to information that stays fixed"**, plus covariance, **inverse-dynamics** and
  temporal-smoothness terms.
- Collapse remedies: free bits 1606.04934, δ-VAE 1901.03416, aggressive encoder 1901.05534.
- **Not found:** VICReg-style regularisation on a *quantised* recurrent carrier of a pretrained LM, used to escape an
  entry basin. **New as scoped.** PLDM is the prior to cite for "variance regularisation on recurrent latents".

## (b) Traps: nuisance spreading, and γ
- **Batch-variance can be satisfied by the wrong variation.** z_b is computed from a segment that includes the item
  token. The cheapest way to meet the hinge is to encode **item identity** (it varies across the batch), not the state:
  the input-carriage trap of §121 again. Position and format are mostly constant within the harness, so they're less of
  a risk.
- **Fix, from PLDM:** apply the variance term **across hand-offs within a sequence** (time), not only across the
  batch. Optionally add an **inverse-dynamics head** (predict item j from (z_{j−1}, z_j)), which ties code changes to
  item effects.
  - Caveat: IDM alone can also be met by carrying the item, so keep it auxiliary and decayed.
- **γ = 1 (one level spacing) is sensible** for "different inputs land in different FSQ cells". On a [−4, 3] range,
  std ≥ 1 spreads over about 3+ levels per dimension.
  - γ = 0.5 is the gentler alternative if the hinge dominates the loss early.
  - Decaying to 0 by 6k is right: it's an entry device, not a permanent constraint.

## (c) Label-free gradient sources for W_down / W_up in the first 2k, ranked for this harness
1. **Skip-to-decoder** (Dieng 1807.04863). The code is fed directly to the answer head and the upper layers. It gives
   an immediate final-answer gradient into W_up and W_down. Final answers are allowed under C4.
2. **A predictive / contrastive auxiliary on the code** (CPC 1807.03748, Z-Forcing 1711.05411, NextLat 2511.05963).
   It is state-relevant, because the next step depends on the state. No labels.
3. **VICReg variance + covariance**, time-wise per (b) (2105.04906; PLDM 2502.14819). It guarantees non-constant
   codes, but not state content.
4. **Aggressive writer updates + a 10–100× LR on new modules** (1901.05534; LoRA+ 2402.12354).
5. **Input dropout at the decoder** (Bowman 1511.06349). Useful once the code is informative; weak at entry.
6. **Encoder pre-training by reconstruction (RECON).** Risks nuisance principal components (LIT_S125).

## (d) "Item definitions are the minimal C4 foothold"
- **"Sufficient" is defensible; "minimal" isn't yet.** OPDEF (each op on the identity; 120 problems, 0.83%) certifying
  s33 while NONE stays at chance shows that definitions *suffice* (in 1 seed so far).
- "Minimal" needs a dose below that: e.g. definitions for half the ops, or only generators. It also needs seed counts.
- **Prior:** operator-first curricula (Zhang 2607.20594), where single operations are learned before composition.
- Remember the qualifier from LIT_S116: definitions also fix the **action convention**.
- **Suggested wording:** "operator definitions (each op applied to the identity, no composition) suffice as a C4
  foothold in k/n seeds; with no foothold, training stays at chance."

## Restart-on-plateau baseline (proposal 3)
- Correct as the engineering bar: a fix must beat restarts' **expected compute per accepted seed** (Dodge
  1909.03004 / 2002.06305 style).
- Report the plateau probability p and the restart cost: expected runs per success = 1/(1 − p).
