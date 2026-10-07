# Full-width latent carrier (768-d): plan, registered only if a 64-d arm passes (2026-10-07; the user's directive via Audit, 01:10)

The user: "if throughout the night 64 wide latents pass, try full width latents. Revisit methods if fail to improve of course."

## Trigger
* A §133 64-d arm (FSQ-64, NOISE-64 or CONT-64) reads stage-1 PASS (≥ 3/4 slots exact at 2,048, SERIAL, ρ ≤ 0.28).
* Status at drafting: all three have passed slot 1; slots 2–4 are running.
* Then this plan is registered as §137, after Literature's check, with the scorer committed before any run and Audit's review and budget.

## What is different from §126 RAW-H (768-d continuous: entered 4/4, drifted; 0.85 at 512, 0.73 at 2,048)
1. **A learned, bounded code**, z_b = tanh(W_down(LN(h)) + shift) · HALF − OFF at width 768, then W_up. RAW-H carried the raw, unbounded last
   hidden state.
2. **The correction that held at 64-d:** FSQ rounding, or dither training, whichever 64-d mechanism passed.
3. **§134's copy anchor for entry.** Every 64-d arm flagged seeds 1 / 101; COPY made S5 4/4 without restarts at d 8.
4. **Optionally, growth from a settled 64-d code** (§136's function-preserving operation), if §136 shows growth is safe.

## Arms (S5, §128's recipe, COPY rows from step 0, no restarts, RTX 3090, seeds 1–4; one change each)
* **FULL-FSQ:** d 768, FSQ with 8 levels. Runs only if FSQ-64 passes.
* **FULL-NOISE:** d 768 continuous with dither U(±0.25) after 2k. Runs only if NOISE-64 passes.
* **FULL-CONT:** d 768 continuous with no correction. Runs only if CONT-64 passes. It is the direct contrast with RAW-H: only items 1 and 3
  differ.
* **GROW-768:** COPY + growth 64 → 768 at §136's THRESHOLD trigger. Runs only if §136 stage 1 shows growth is safe.
* **Proof seed first** (seed 1 per arm). An arm continues to seeds 2–4 only if seed 1 is at least 0.95 at 2,048 (as §133-2).

## Rules (fixed at registration)
* **PASS (stage 1, S5):** at least 3/4 seeds exact at 2,048, SERIAL and contractive (127-12's clause for rounded arms, 133-1's first-three form
  for continuous ones), with no restarts.
* **"Fails to improve → revisit" (the user's rule, fixed now):**
  - **(i) Stability.** If no full-width arm passes stage 1, full width is not adopted.
    - Theory writes a post-mortem from ρ, prefix 1 / 2, codes per state and dimension use.
    - It proposes at most one revised method, as a fresh registration after Literature and Audit. No automatic variants.
  - **(ii) Benefit.** On S5 an 8-d code already suffices, so S5 cannot show gain. A full-width arm that passes is compared with the passing
    64-d arm on a capacity task: S5×4 (§133 stage 2) or multiplier-major multiplication (THEORY_MULT_PLAN).
    - **Improves** = it passes where 64-d fails, or it reaches a higher exact rate at an equal budget.
    - Otherwise 64-d stays the operating width (cheaper), and full width is reported as "stable but no gain". That is also a revisit point.

## Cost
* The GPT-2 trunk dominates the compute; the 768 × 768 projections are negligible. So about $0.4–0.5 per run, as for §133.
* Proof seeds: up to 4 arms × 1 run, about $2. The full set: up to 16 runs, about $6–8.
* This is staged against the credit floor; Audit approves each stage.
