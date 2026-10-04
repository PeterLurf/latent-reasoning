# Pre-checks for §109-R (per-op latent recurrence) and §109-A (running-sum auxiliary) (2026-10-04)

Literature session, for Theory; Audit copied. Brief. IDs verified (arXiv abstract/HTML pages; earlier notes).
Exactly one will be registered after §107 scores.

## §109-R: r − 1 continuous refinements per op vs PAUSE-4
- **Does latent refinement beat fixed fillers? Mostly yes, on depth or OOD:**
  - Coconut 2412.06769's ablations: continuous thoughts above its pause-token variants on GSM8K.
  - **Cheng & Zhang 2609.35643** (new): on ProsQA-Ext, vanilla, CoT and **pause-token** models fit in-distribution
    but fail depth generalisation, while latent-reasoning models implement a recurrent forward-reachability search and
    generalise.
  - Our own finding 5: recurrent 0.59–0.66 vs pause/no-latent 0.50–0.56 at depth-matched 6 hops.
  - **Counter-evidence:** Aswal 2606.12689 (pause-as-thought reproduces Coconut's signatures; removing thoughts barely
    matters on ProsQA). London & Kanade 2505.21024 (pause tokens alone increase constant-depth expressivity).
- **Recurrent depth priors:** Huginn 2502.05171, TRM 2510.04871, PonderNet 2107.05407 / ACT 1603.08983 (adaptive),
  Attractor Models 2605.12466 (fixed-point refinement).
- **Flags:**
  1. **Slack** (our Prop. 4/9). Refinements 1..r−1 have no target, so they can dwell or develop clocks. Fix r per arm
     (as planned) and test at the trained r only, or accept that extrapolation in r is untested.
  2. If PAUSE-4 ≈ R-4, the gain is compute, not latent refinement. Report both against MARKOV r = 1.
  3. Sequential passes scale by r; report them.

## §109-A: digit-wise running-sum auxiliary on the slot
- **How 2510.00184 applied it:**
  - A linear **regression** head (MSE, scalar target) on the outputs of the layer-2 attention heads of a **2-layer,
    4-head model trained from scratch** on 4×4 multiplication.
  - The target is the accumulated running sum c^k at each timestep.
  - **≈ 99% vs < 1%** for standard fine-tuning. λ is not stated in the summary I read.
- **Your version differs.** It is digit-wise classification (CE) on the slot latent of a pretrained GPT-2, under a
  MARKOV bottleneck. Digit-CE is a reasonable factorised variant (§86: structured heads learn where index heads stall).
  Optionally add their **scalar regression** target as a second arm, since it carries magnitude and carry structure
  that digit-CE does not.
- **Risk: decodable but not used. Yes, possible in general,** but MARKOV largely removes it here.
  - The next segment sees only slot j−1's latent, so whatever S_j information the aux puts there is the *only* route.
    Decodability then implies availability.
  - It still doesn't imply the next segment *reads* it (it could fail to). §104's donor protocol at j−1 → j is the
    right score: USE ≈ 1 means read.
- **Wording:** the aux adds a per-step state target, so it strengthens supervision (rule 1). It is not C4. That doesn't
  change the C1 claim, which rests on MARKOV plus donor USE.
