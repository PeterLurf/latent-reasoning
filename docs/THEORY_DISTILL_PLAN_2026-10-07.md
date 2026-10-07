# Distillation plan (2026-10-07; for the user, after §135's failure; nothing registered until the user says go)

The user's idea: natural-language CoT is a poor training artifact for a latent reasoner, so distil a CoT model's internals instead.

## What §135 showed (notes 135-9 / 135-10)
* Both arms failed on S5 (0/4 each) and did **worse than no distillation**:
  - TSD: InfoNCE to the CoT teacher's hidden state;
  - BASE: the same, but to pretrained GPT-2's hidden state on the same text.
* **Why:**
  - the weight-1 contrastive loss never vanishes (it has a floor);
  - its target, a hidden state that also carries position and history, pulls that history into the latent code.
  - The code then stops being a Markov state: 1,957–6,870 codes per state, against 14–29 in exact runs. Errors follow (wrong from step 1 under
    TSD, compounding under BASE).
* It helped at first: at 2k BASE was entering as fast as the copy anchor. The harm came from keeping the full pull on after entry.

## Revised design (DISTILL-2), on the task-agnostic fsq120 carrier
1. **A state-only, achievable target.** Distil only what the teacher knows about the *state*, not its whole hidden vector.
   - Concretely: the teacher's state-probe readout at each step boundary, as a distribution over states or digits.
   - Or: a low-rank (≤ code width) projection onto the probe-defined state subspace.
   - The loss can reach about 0 once the student's code holds the state, so its gradient switches itself off, as the copy anchor's does.
2. **A warm-start only:** weight 1 at step 0, faded linearly to 0 by the end of the rounding ramp (4k). Inference never uses the teacher, and after
   4k the run is final-answer-only.
3. **On COPY's base** (§134), which already fixes entry without labels. DISTILL-2 is then tested for what it adds beyond COPY.
4. **On the task where latent methods hit a wall:** multiplier-major multiplication (THEORY_MULT_PLAN), not S5. Per-step supervision already
   solves S5, so S5 cannot show the user's claim.
   - The teachers are §98's kept text-CoT models (m98_tc_4x4 / 5x5), at $0. Their CoT writes exactly the student's carried state, the running
     sum S_j.

## Arms and rules (draft)
* **A:** MULT-W + COPY (final-answer-only), the comparator, from the multiplication plan.
* **B:** A + DISTILL-2 (state-only teacher target, warm-start fade).
* 4×4 first, seeds 1–4; 5×5 after.
* **Readings:**
  - B passes where A fails, or B's exact rate is higher: **the teacher's internals help where final-answer training alone does not** (the user's
    claim, on the right task);
  - B ≈ A: no gain over COPY;
  - B < A: harm, as in §135.
* **Scope (stated):** per-step state supervision by proxy (the teacher read the CoT), so not C4. It is a C1 / C6 / C2 and multiplication tool.

## Order, cost and ETA
* It runs **after** the multiplier-major final-answer-only arm reads (it needs a wide carrier, gate G1), because A is its comparator.
* About $3 (4 seeds × 4×4, about $0.3–0.5 each), plus the teacher-probe fit ($0, local).
* ETA: about a day after multiplier-major starts (2026-10-08 / 09), if the user says go.
