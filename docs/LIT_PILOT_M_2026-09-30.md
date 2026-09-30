# Prior work for Pilot M (THEORY §38): label-free amplify-and-distil ladder in depth (2026-09-30)

Literature session, for Theory; Audit copied. IDs verified (arXiv API or Semantic Scholar). "(full text)" = read from
the paper itself.

**1. Iterated self-distillation that extends depth or length without labels.**
- Self-Improving Transformers 2502.01612 (full text): self-labels at n+1 from round n, with length filtering and majority
  voting; standard transformers, no looped variant.
- Expert iteration, Anthony et al. 1705.08439: a search amplifier distilled into a network, iterated. The closest
  "procedure as amplifier" ladder.
- IDA 1810.08575; Distilling System 2 into System 1 2407.06023 (self-consistency filter, label-free).
- Progressive distillation 2202.00512: an iterated distillation ladder of a sampling procedure, in the opposite direction
  (fewer steps).
- Looped and latent models: LoopFormer 2602.11451 (shortcut-consistency self-distillation *within* one training run,
  not iterated rounds). Kohli 2604.07822 (trained on 20 hops, answers 30 by adding recurrence; no distillation).
- **No iterated label-free ladder on latent or looped models found.**

**2. Restarts or checkpoints as the amplifier.** None found. Test-time re-application exists (CTM mazes, THEORY §26), and
Delethink 2510.06557 trains chunked restarts with RL reward, not self-distillation. Restart-as-amplifier appears new;
scope the claim to latent reasoners.

**3. Known failure modes.**
- **Error avalanche** (Lee et al. 2502.01612, full text): "errors in n-digit training data propagate to n+1-digit
  examples … eventually causing a cascading collapse". It happens without filtering on harder tasks; length filtering
  and majority voting prevent it.
- **Model collapse / MAD** from training on one's own outputs: Shumailov 2305.17493, Alemohammad 2307.01850 (tails and
  diversity lost; fresh real data prevents it).
- Implication for Pilot M: frontier labels at 13–24 hops are the noisiest.
  - Add a label-free agreement filter, e.g. keep only nodes on which two re-anchoring variants (top-1 and top-3
    anchors) agree.
  - Keep some round-0 data in the mix.
  - Monitor on a small held-out labelled probe, not used for training, to catch an avalanche.

**4. Why two training depths give one-loop length generalisation (X1).**
- Kuo 2606.29983: randomised loop depths turn the gradient into an average over depths, suppressing depth-specific
  components, and supervise "compatible answers across a local window of depths" (trajectory consistency).
- Bansal 2202.05826: varied iteration counts "discourage the network from internally counting iterations".
- Zhang 2607.20594: each budget contract sets the frontier speed (v ≈ n/T); warm-starting across contracts transfers the
  algorithm and re-prices its speed, so training two contracts plausibly forces a speed-invariant mechanism.
- Fan 2409.15647: supervising variable step counts gives length generalisation.
- In our terms, two depths = two contracts, so only a time-homogeneous transition fits both (Prop. 4 / C2).
