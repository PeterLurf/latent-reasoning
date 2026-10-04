# Prior check for §111: reliability of §108's Markov hand-off (C6): 2× budget and label-free best-of-n (2026-10-04)

Literature session, for Theory; Audit copied. Brief. IDs verified (arXiv API 2026-10-04; earlier notes).

## 1. Seed lotteries in algorithmic tasks: is budget the standard fix?
- **Budget helps when the lottery is *timing*** (late generalisation). Grokking: Power 2201.02177; Omnigrok
  2210.01117; Nanda 2301.05217. Our §91 found budget fixed a composition lottery.
- **It doesn't when the lottery is a *basin*:**
  - Zhang 2607.20594's A5 lottery (8/22) was fixed by an **operator-first curriculum**, not by time.
  - Our §87: short-label warm starts gave basin entry where annealing didn't.
  - Kuo 2606.29983: randomised stopping stabilises looped extrapolation across 18–32 seeds.
- **For §111:**
  - s2 "stuck at chance" looks like a basin failure. Budget may not move it.
  - s3 (serial, but 1.5e-3 per hand-off) looks like a precision/timing issue. Budget plausibly helps.
  - Consider adding a **teacher-forced segment-input warm start** (LIT_S99_ACCEL / LIT_S108) as the basin-entry arm.
- Lottery ticket (Frankle 1803.03635) is about subnetworks, not seed selection. Don't cite it here.

## 2. Is best-of-n on training-only criteria honest practice?
- **Yes, if pre-registered and reported as a budget curve:** Dodge 1909.03004 (expected best validation performance as
  a function of n, with compute stated); Henderson 1709.06560, Agarwal 2108.13264 (report all runs; interval
  estimates). See also LIT_STABLE_LATENT (b).
- **Caveat specific to §111.** Final training loss or in-range accuracy may not separate s1 from s3. Both can be ≈ 1.0
  in range, while s3's 1.5e-3 per-hand-off error only shows up at 512.
  - Use a **training-only criterion sensitive to per-step error**: per-hand-off accuracy on long rollouts built from
    the *training distribution* (chains at the trained length, many of them). That is the in-training analogue of our
    P7 predictor.
  - Report how well each criterion ranks the test-passing seeds (AUC).

## 3. Driving per-step error from ~1e-3 to ~1e-4 (other than error-correcting snaps)
- **Train on the model's own rollouts (the compounding-error fix).**
  - DAgger, Ross, Gordon & Bagnell 1011.0686: with expert-only training, sequential error compounds about O(T²ε).
    Training on the states the learner itself visits gives about O(Tε). See also scheduled sampling 1506.03099.
  - Here: train segment j on the model's *own* slot j−1 latents from longer rollouts (more hand-offs per training
    sequence), so imperfect inputs are in distribution.
- **Focus the loss on rare errors:** focal loss 1708.02002 (down-weights easy examples). Or mine the hard (state, op)
  hand-offs where per-step errors concentrate. Report the per-transition error table.
- **Restoring / contractive dynamics:** a Jacobian or contraction penalty on the carry map. Attractor Models
  2605.12466 and DEQ 1909.01377 (fixed-point refinement). Our Th3: a restoring map gives unbounded reach.
- **Finishing optimisation:** longer low-LR annealing at the end. Errors of 1e-3 are often an unconverged margin.
- **Numerics:** at 1e-4 per-step error, bf16 rounding of the carried latent can matter. Run the carry in fp32 at test
  as a cheap check. Our P27 found no precision effect in the toy, but at much higher error rates.
- **Inference-side:** majority over a few seeds' chains (an ensemble). It lowers error if the seeds' errors are
  independent, but it is a different system, so report it separately.
