# Prior check for §129: re-draw only the bottleneck head after a plateau, keeping GPT-2's body (2026-10-06)

Literature session, for Theory; Audit copied. Brief. IDs verified via abstract pages (2026-10-06); 1910.08475 as
earlier.

## (a) Priors
- **Partial re-initialisation to escape basins or restore plasticity:**
  - Nikishin et al., *The Primacy Bias in Deep RL*, 2205.07802 (periodic resets of the last layers);
  - Sokar et al., *ReDo*, 2302.12902 (recycle dormant neurons);
  - Ash & Adams, *shrink-and-perturb*, 1910.08475 (warm-starting hurts; shrink plus noise recovers);
  - Zhou et al., *Fortuitous Forgetting*, 2202.00155 (forget-and-relearn, re-initialising a subset of weights).
- **Head re-initialisation in transfer:** RIFLE 2007.03349 (periodically re-initialise the FC head during fine-tuning,
  so stronger gradients reach the lower layers).
- **Collapse-specific:** Łańcucki 2005.08520 and Jukebox 2005.00341 re-initialise dead VQ codes.
- **Not found:** a re-draw of an *injected latent bottleneck* (writer + reader) on a frozen-schedule pretrained LM,
  used as a diagnostic that splits head vs body as the location of a no-read basin. **New as scoped.**

## (b) Keep or reset the body's Adam moments?
- **"Keep" biases towards BODY slightly.** Adam's second moments for body parameters reflect the no-read regime. The
  input embedding and attention rows that would carry the code have had tiny gradients, so their v is small and their
  effective step is large. That *favours* re-learning, not BODY.
- The first moments (m) carry stale directions, but they decay within about 1/(1 − β1) = 10 steps.
- **Verdict:** "keep" is a fair single test. The stale-moment bias is small and short-lived.
- **The bigger body-state question is the weights, not the optimiser.** Primacy-bias work finds that body weights
  trained in a degenerate regime can lose plasticity.
  - If you can afford a second arm, make it **shrink-and-perturb the body** (1910.08475: θ ← λθ + σε, e.g. λ = 0.9,
    small σ) plus the head re-draw, not a moment reset.
  - That separates "the head is the basin" from "the body has lost plasticity" when HEAD fails.

## (c) Restarting the schedule clock
- **Restarting is right.** A head re-draw at the end of a cosine schedule would get almost no LR, and the soft phase
  plus L_max 2 is the regime where entry happens.
- **The confound is real but mild:** the body gets 2k extra steps, plus the warm-up / LR spike (a "warm restart",
  SGDR-like). That LR spike alone can kick the run out of a plateau.
- **Add the cheap control: CLOCK-ONLY.** Restart the schedule clock *without* re-drawing the head.
  - If CLOCK-ONLY also escapes, the effect is the LR restart, not the head.
  - It costs the same as one arm, on the same 4 seeds.
- **Readings with that control:**
  - HEAD = re-draw escapes and CLOCK-ONLY doesn't;
  - LR = both escape;
  - BODY = neither escapes.
