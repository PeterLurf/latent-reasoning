# Prior check for the multiplier-major READ design with ICoT-style removal of reader targets (2026-10-07)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-07) or earlier notes (LIT_INDEX).

**The design:** S_j = a × (b_1..b_j) is carried in a NOISE-64 code, and the shared decoder reads S_j as text from code_j at every
hand-off.
- Arm A keeps the reader on.
- Arm B removes reader targets latest-first, down to the final answer only by about 18k.

## 1. ICoT-SI's removal order, and the analogue for a latent chain
- **ICoT-SI 2405.14838 removes from the start of the CoT:**
  - a fixed number of tokens per epoch, earliest first, so the remaining explicit CoT is always the later part;
  - **removal smoothing** (a few extra tokens removed at random, with an exponentially decaying offset);
  - an **optimizer reset** at each removal step.
  - They report removal without these as unstable.
- **Coconut 2412.06769 is also earliest-first:** stage k replaces the first k steps by latent thoughts.
- **No precedent found for latest-first.**
- **The latent-chain analogue of ICoT-SI is earliest-first** (drop S_1, then S_2, …). It also fits this design better:
  - S_1 = a × b_1 is the small, easy number. The late S_j are the large ones (up to 8 digits).
  - Note 140-3 found values ≥ 100 unrepresented even in h, which is the stated reason for the reader.
  - Latest-first removes the reader first exactly where it was motivated. Earliest-first keeps it longest there.
- **Both orders start with a one-step gap.** Earliest-first: an unsupervised prefix checked by S_{k+1}. Latest-first: an unsupervised
  tail checked only by the final answer.
- **Recommendation:** make B earliest-first, or run both orders if budget allows. If latest-first stays, state why, since it departs
  from the precedent.

## 2. What's different from §98 GEN-M (0.163 at 4×4)?
- **The closest prior is 2510.00184.**
  - Standard fine-tuning fails at multiplication because it doesn't learn the long-range dependencies.
  - An **auxiliary loss that predicts the running sum** from the hidden state (via a linear probe) makes it learnable.
  - The design's reader is that idea, with the target decoded as text from a Markov code.
- **GEN-M's fork could see a, the op token, the start text and earlier slots.** It could produce P_j ( S_j ) partly from context,
  without S_j being in the slot. In the new design the reader sees only code_j. That's GEN-M-MASK's constraint plus a Markov
  hand-off, so S_j must be carried.
  - This matches the §111 READ evidence on S5: the reader fixed the no-read basin, 5/5 serial at 512.
- **Two §98 issues the design doesn't change:**
  - **(a) Budget.** §98 trained 24k steps, about 1.9 epochs, against ICoT-SI's 200 epochs. 2510.00184 also trains much longer than
    that. If A stays low, budget is the first suspect. Pre-state the budget reading as §98 did ("not acquired within budget").
  - **(b) Hand-off count.** Multiplier-major 4×4 has only 4 hand-offs, so the "k = 4 wall" was about snaps / history cuts, which a
    Markov chain replaces. Expect the wall to move to 5×5 / length.
- **Number format is already favourable:** digits are least-significant first and space-separated, so one GPT-2 token per digit, in
  carry order (Lee 2307.03381; Shen 2311.14737). The target width is fixed per step (4 + j digits, zero-padded), so the decoder
  needs no length decision.

## 3. Pitfalls
1. **CE normalisation:**
   - Use the mean over each target's digit tokens, then the mean over hand-offs, with a fixed weight relative to the final-answer CE.
   - Summed CE lets the long late S_j (8 digits vs 5) dominate and scales the reader against the final loss by the digit count.
   - Log reader accuracy per digit position and per step: high digits after long carries are where it will fail (note 140-3's
     pattern).
2. **Removal collapse:**
   - Removing an auxiliary target is milder than §126 ANNEAL (which removed the carrier path; see LIT_DISTILL2), but §89 needed 3×
     budget to fade.
   - Import ICoT-SI's two stabilisers: smoothing of the removal boundary, and an optimizer reset (or a short LR re-warm-up) at each
     removal.
   - **Don't start removal before the multiplier curriculum reaches n = 4.** Otherwise the full-length S_j never get the reader.
   - Prefer a **gated** schedule: remove the next target only once final-answer exact has stayed at or above a threshold for N
     steps. A fixed schedule to 18k is a removal-step budget, not ICoT-SI's 200 epochs.
3. **Retention after removal:** keep decoding the removed S_j (unweighted) and log their accuracy. A decay means the code moved to a
   shortcut. In multiplier-major, a is visible in every segment, so carrying the b-prefix (§121's input carriage) and recomputing
   at the end is the likely one.
   Report SERIAL before and after removal.
4. **C4 wording:** A uses the teacher's CoT text (revised C4). B ends at final answer only, but it is *trained through* CoT, so it
   is not strict C4. State it.

## 4. Novelty
- **New as scoped:** a per-step text reader on a Markov, code-only continuous carrier in a pretrained LM, with ICoT-style removal
  of the reader targets, scored by donor SERIAL.
- **Closest:** 2510.00184 (a running-sum auxiliary), ICoT-SI (removal), §111 READ (the reader).
