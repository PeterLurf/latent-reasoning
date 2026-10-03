# Prior check for §91 (making loop-aligned composition reliable across seeds), plus leads on topic (3) (2026-10-03)

Literature session, for Theory; Audit copied. arXiv IDs verified (arXiv API / Semantic Scholar).

**Context.** In §90 (raw, unscored), S4 k = 2 at 400 words with full transition coverage composes unseen 2-op words in
1 of 3 seeds (0.11 / 0.98 / 0.17): a seed lottery.

## 1. Arm (c), single-op words first: ⚑ the direct precedent exists
- **Zhang et al. 2607.20594.** In weight-tied looped transformers, an **operator-first curriculum** (learn the single
  operation, then compose) turns from-scratch seed lotteries into reliability:
  - A5: 8/22 from scratch → 100% of attempted seeds;
  - S5: deadlock → 10/10, exact match ≥ 0.92 at 2×.
- That is §91(c) in another architecture.
- **Wording:** "operator-first curriculum (Zhang 2607.20594) applied to a looped pretrained GPT-2 with per-iteration
  state targets". New only in that setting.
- Related: RELAY 2502.08482 (iteration-aligned supervision), Fan 2409.15647 (length curricula for looped models),
  Abbe 2301.13105 (degree curriculum).

## 2. Arm (b), grokking-style weight decay
- **Priors:**
  - Power 2201.02177: weight decay was the most effective intervention for generalisation on algorithmic tables.
  - Nanda 2301.05217: the generalising circuit forms gradually, and weight decay drives cleanup.
  - Omnigrok 2210.01117: generalisation is controlled by weight norm and initialisation scale.
  - Liu 2205.10343 (effective theory; a "Goldilocks" zone of hyperparameters).
  - Varma 2309.02390: past a critical dataset size the generalising circuit is more parameter-efficient, so weight
    decay selects it.
  - Kumar 2310.06110: grokking as a lazy → rich transition.
  - Lyu 2311.18817: large init plus small weight decay provably induces grokking.
  - Grokfast 2405.20233: amplifying slow gradient components accelerates grokking.
- **Flag: wd 1.0 on a pretrained backbone.**
  - With decoupled AdamW the per-step shrink is lr·wd. At lr 1e-4 and wd 1.0 the weights decay by a factor
    exp(−1e-4·N): ≈ 0.45 after 8k steps, ≈ 0.04 after 32k (if decay applies to all parameters).
  - That erases GPT-2's pretraining, which confounds "grokking" with "training a randomly scaled net".
  - Either apply wd only to the new or looped parameters, or use **L2-SP** (Li et al. 1802.01483: decay towards the
    pretrained weights, not towards 0). Report the weight-norm trajectory either way (Omnigrok).
- **Arm (a), 4× steps, is the right control.** Late generalisation long after memorisation is the grokking pattern
  (Power), so (b) must beat (a), not just the baseline.

## 3. Reliability across seeds (topic 3, leads)
- **Curriculum is the strongest documented lever:** Zhang's operator-first results above; Kuo et al. 2606.29983
  (randomised stopping stabilises looped extrapolation across 18–32 seeds).
- **Diversity and repetition:**
  - Zhou 2311.04420: less repetition per example reduces memorisation.
  - Raventós 2306.15063: a task-diversity threshold.
  - For §91, add an arm with the *same* 400 words but more distinct start states or fresh samples per epoch (less
    repetition), if the budget allows.
- **Selection rather than reliability:** best-of-n by validation loss, reported in Dodge 1909.03004 style
  (LIT_STABLE_LATENT (b)). It is legitimate if stated, but it isn't a reliability fix.
- **Early diagnostics:** our P7 / seed triage (PROTOCOL7) predicted the outcome in training. A cheap analogue here:
  per-iteration-2 accuracy on held-out *training-start* variations at 25% of training.

## Suggested reporting for §91
- Report per arm:
  - seeds certified out of 5, with the composition accuracy distribution;
  - weight-norm trajectories;
  - the step of first generalisation.
- Report (b) against (a), and (c) against the baseline.
