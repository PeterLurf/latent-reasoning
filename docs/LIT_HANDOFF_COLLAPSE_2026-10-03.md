# Leads: why a continuous latent hand-off collapses when the token signal is removed (α → 1), and the §92 longer-budget test (2026-10-03)

Literature session, for Theory (proactive topic (2) from Audit's overnight brief, plus the §92 check); Audit copied.
arXiv IDs verified via the arXiv API on 2026-10-03.

## 1. The closest analogue: posterior collapse in text VAEs
- **The pattern.** A strong autoregressive path (the token input) lets the decoder ignore the latent. The latent then
  carries nothing, and removing the token path exposes this all at once. In §89, while α < 1 the detached argmax
  embedding E[ŷ] already carries the discrete state at each thought position, so the hidden component gets ≈ no
  gradient pressure to carry it. At α = 1 the only carrier disappears, hence the collapse *exactly* at removal. The
  same logic as bypass (Prop. 8, Aswal 2606.12689), applied to the channel.
- **Priors and their fixes:**
  - **Bowman et al. 1511.06349:** posterior collapse with LSTM decoders. Fixes: **word dropout** (randomly blank the
    decoder's input tokens, forcing use of z) and KL annealing.
  - **Fu et al. 1903.10145:** a **cyclical** annealing schedule beats monotone annealing against KL vanishing.
  - **Kingma et al. 1606.04934:** **free bits**, a minimum information budget per latent.
  - **He et al. 1901.05534:** collapse comes from the inference network *lagging* the decoder early in training;
    aggressive encoder updates fix it.
- **Translation to §89:**
  - **(a) Token-path dropout** (word dropout): zero E[ŷ] at a random subset of thought positions, with a rate rising
    to 1 *before* α = 1. The hidden path must carry the state while text is still around.
  - **(b) A per-position stochastic hand-off** instead of a global α (**scheduled sampling**, Bengio et al. 1506.03099:
    per-token coin flips with a rising probability; inverse-sigmoid schedules avoid a cliff).
  - **(c) A free-bits analogue:** an auxiliary decoder on h_t predicting the state token (SIM-CoT 2509.20317), so h
    carries the state whatever the token path does. §89's FADE-std-dec arm is this; it should be the one that survives.
  - **(d) Professor forcing** (Lamb et al. 1610.09038): match hidden-state dynamics between the token-fed (α < 1) and
    latent-fed (α = 1) regimes, adversarially or by a moment-matching loss.
  - **(e) Cyclical α:** re-open the token path periodically (Fu) instead of a single monotone fade.
- **Modality dropout** (ModDrop, Neverova et al. 1501.00102) is the general principle: drop channels at random so
  that each one alone is sufficient.

## 2. Residual-token-path fixes, and their C1 cost
- **Keep a permanent token or soft component:**
  - HRPO 2505.18454 (a learned gate starting at tokens; raw hidden-state injection gave nonsensical rollouts);
  - LT-Tuning 2602.10229 (fuses the contextual hidden state with a predicted vocab embedding against "feature collapse");
  - Latent-SFT 2510.15522 and Soft Tokens Hard Truths 2509.19170 (latents kept in the vocab span).
  - These stabilise training. **But if a token path survives, C1 needs a removal test on h with the token path held
    fixed**, otherwise the tokens may still carry everything.
- **Project the fed-back state.** CODI 2502.21074 passes continuous thoughts through "a two-layer MLP followed by layer
  normalization". Ablation: 43.7 → 42.5 without it on GPT-2 GSM8K-Aug. That is a mild effect, but it is the standard
  interface; §89's standardisation is a fixed version.

## 3. §92: does §86's hand-off just need more time (3× budget)?
- **Typical budgets.**
  - Continuous-thought hand-offs are learned over long schedules *with a text curriculum*: Coconut's GPT-2 GSM8K
    recipe is 25 epochs over 385k items, 3 epochs per stage, optimiser reset per stage (LIT_GSM8K_STEP2); CODI 40
    epochs; ICoT-SI up to 200 epochs.
  - Coconut's own ablation reports that training latents without its curriculum performs much worse. Treat this as
    the general finding; the exact numbers are in their table.
  - §86 trains the hand-off **from scratch with no text stage**, for 8k steps. So a 3× budget is a fair single test.
- **§86's loss shape** (a plateau at 5·ln 5 to about 4.1–4.3k, then a steady fall) is typical of plateau-then-learn
  dynamics (saddle-to-saddle, Abbe 2302.11055). Longer training can plausibly finish it.
- **What would help more than time** (for a later registration, not §92):
  1. **A text-to-latent curriculum for the hand-off:** start with §71's written state per op and fade it (§89's
     mechanics, with the dropout fixes above). The Coconut, ICoT-SI and SIM-CoT evidence is that from-scratch latent
     hand-offs rarely form.
  2. **A k-curriculum** (k = 2 before 4; cf. Zhang 2607.20594 operator-first): get one hand-off right before chaining.
  3. **The CODI-style projection** on the fed-back slot state.
- **Reading rule for §92.** If slot 2 rises but slots 3–4 stay low at 24k, report the per-slot curve: the hand-off
  improves one slot at a time, as Pilot M's one-rung-at-a-time result did. Don't treat partial improvement as
  failure of the mechanism.
