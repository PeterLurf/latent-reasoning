# Prior check for §196: Grokfast on a non-converging seed (2026-10-13)

Literature session, for Theory; Audit copied.
- **IDs:** verified via arXiv abstract pages.
- **Grokfast details:** from its HTML and the official repo ironjr/grokfast (MIT) (2026-10-13).

## 1. Grokfast, Lee et al. 2405.20233 ("Accelerated Grokking by Amplifying Slow Gradients"): confirmed
- **The idea:** treat each parameter's gradient sequence as a signal and amplify its slow (low-frequency, "generalisation-inducing") component.
  ĝ(t) = g(t) + h(t) ∗ g(t). Up to more than 50× faster grokking, on images, language and graphs.
- **The EMA variant (`gradfilter_ema`), defaults confirmed:**
  - **α = 0.98, λ = 2.0**;
  - per parameter: ema ← α·ema + (1 − α)·g, then **g ← g + λ·ema**;
  - the EMA is initialised from the first gradient.
- The MA variant (`gradfilter_ma`): window 100, λ 5.0, filter "mean", warm-up until the queue fills.
- **Placement (README):** "**between `loss.backward()` and `optimizer.step()`**". It modifies `p.grad` in place, so the optimiser (any first-order one) sees the
  filtered gradient.
- The paper reports a **synergistic effect with weight decay**. The repo's example runs pass `--weight_decay 0.01`.

## 2. Interaction with AdamW and global-norm clipping (our stack: clip_grad_norm_(1.0) → AdamW, lr 1e-4)
- **Order matters. Recommended: backward → Grokfast filter → clip_grad_norm_ → AdamW.step.**
  - If clipping comes *first*, the EMA accumulates clipped gradients, and the filtered gradient (up to about (1 + λ)× larger) then **reaches AdamW unclipped**:
    a systematic norm jump of up to 3× at λ = 2.
  - Filtering first, then clipping, keeps the global norm bound and preserves the *relative* boost of the slow component (clipping rescales all components
    uniformly).
- **With clipping active on 100% of steps (pre-clip median 5–7):** the global norm is pinned, so Grokfast changes only the **direction** (the share of the slow
  component), not the step size. That's arguably what you want, but it means the effect is bounded by how far the slow component is from the fast one.
- **Adam's normalisation:**
  - AdamW divides per coordinate by √v. A *uniform* rescaling cancels out, but Grokfast's amplification is **non-uniform** (it reweights coordinates whose
    gradients are persistent).
  - Adam's own first moment (β₁ = 0.9) is already a fast low-pass. Grokfast's α = 0.98 adds a slower one (an effective window of about 50 steps), so the effect
    survives, but it's partly redundant with momentum.
  - The paper frames Grokfast as a wrapper for any first-order optimiser, and reports it with weight decay. **No λ / α retuning specific to Adam is reported.**
  - **Practical:** keep the defaults (0.98 / 2.0) for the first test. If AdamW's v is dominated by the filtered gradient (that is, the update shrinks), try
    λ = 5 with the MA variant, or α = 0.99.
- **Monitor:** the slow-component share, ‖λ·ema‖ / ‖g‖, and the AdamW update norm per step, so "it did nothing" can be told apart from "it was cancelled".

## 3. Seed-dependent delayed generalisation, and fixes
- **Grokking basics:**
  - Power et al. 2201.02177 (generalisation long after overfitting; **weight decay is the most effective intervention**);
  - Omnigrok, Liu et al. 2210.01117 (**weight-norm / initialisation-scale** account: large init → delayed or no grokking; controlling the norm induces or
    removes it).
- **Adaptive optimisers:** **the Slingshot Mechanism, Thilak et al. 2206.04817.** Adam-family optimisers show cyclic last-layer weight-norm spikes, and
  grokking tends to occur at these slingshots. That's an optimiser-dynamics, not purely data, account.
- **Numerical failure modes:** **"Grokking at the Edge of Numerical Stability", 2501.04697.**
  - Without regularisation, models can **never** grok because of **Softmax Collapse** (floating-point errors once logits grow), driven by "naive loss
    minimisation" scaling up logits;
  - fixes: **StableMax** and **⊥Grad** (an orthogonal-gradient optimiser).
  - **Relevant to a seed that "never grokks".**
- **Efficiency account:** Varma et al. 2309.02390 (grokking as switching from a memorising circuit to a more efficient generalising one; weight decay drives it).
- **Ours:** the restart-on-plateau procedure (§126, the no-read basin) is the established in-house fix for non-entering seeds. Kohli 2604.07822's three-stage grok
  is the depth analogue.

## Suggestions for §196
1. **Filter before clipping** (as above). Log the slow-component share and the update norm.
2. **Pair with a weight-decay check:** Grokfast's reported gains come with weight decay. Confirm the run's AdamW weight decay is non-zero (Power: the strongest
   single lever).
3. **On a seed that never generalises,** also log the **max logit / softmax saturation** of the answer head (2501.04697's Softmax Collapse) and the weight norm
   (Omnigrok). Those are cheap diagnostics that separate "slow" from "stuck for numerical reasons".
