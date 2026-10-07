# Prior check for (A) RUNNING-SUM (carrying a growing 4-digit number) and (B) MULTI-SLOT (K = 4 carried vectors) (2026-10-07)

Literature session, for Theory; Audit copied. Prompted by the user's "we gotta figure out how to fix the multiplication" (via Audit, 15:35).
Brief. IDs verified via the arXiv API (2026-10-07) or earlier notes (LIT_INDEX).

## (A) RUNNING-SUM: s = (s0 + Σ items) mod 10⁴, with 2-digit items and a 4-digit final answer
**Priors.**
- **How LMs represent numbers:**
  - digit-wise, base 10: 2410.11781 (Levy & Geva; per-digit circular structure, errors are digit-wise);
  - Fourier features with several periods (2, 5, 10, …), with addition computed by rotation: 2406.03445;
  - a "helix" representation: 2502.00873;
  - a bag of heuristics rather than an algorithm: 2410.21272;
  - embeddings capture magnitude only for small values, and extrapolation fails: 1909.07940.
- **Modular addition in one vector:** a clock / Fourier representation, where adding is a rotation (Nanda 2301.05217).
- **Multi-digit addition with carries:**
  - carry cascades are the hard case: Quirke & Barez 2310.13121;
  - digit significance has to be explicit: Abacus embeddings 2405.17399, Lee 2307.03381 (reversed order);
  - a running-sum auxiliary unlocks multiplication: 2510.00184.
- **Recurrent counting:** finite-precision LSTMs can count, GRUs can't (1805.04908); NALU 1808.00508 (magnitudes in single units
  don't extrapolate).
- **Not found:** a pretrained LM's *discrete latent carrier* holding a growing multi-digit number. **New as scoped**, as a diagnostic.

**What the literature predicts, and the pitfalls.**
1. **Capacity is not the issue.** 10⁴ states is 13.3 bits, against FSQ-8's 24. The question is representation and learning
   (consistent with 140-3: not even h held values ≥ 100).
2. **Two natural codes, and they differ in difficulty:**
   - **Digit-wise** (4 digits): readable by the digit decoder, but each hand-off must resolve a carry chain (up to 4 digits) in one
     pass.
   - **Fourier / clock:** adding is carry-free, but the decoder must convert phases to digits.
   - GPT-2 natively uses the mixed form (2406.03445). Log which one the code uses: R² of each digit vs cos / sin(2πs/T) for
     T ∈ {10, 100, 1000, 10⁴}.
3. **Add a magnitude ladder: mod 10² / 10³ / 10⁴, same harness.** It locates where carriage breaks (the question 140-3 raised), and
   is cheaper and more informative than a single modulus. mod 10² is close to §128's MOD and CYC, which closed.
4. **Coverage:** 10⁴ states × 100 items, so states are seen sparsely. Passing means a structured adder, not a table, which is what
   we want. CERT / Moore minimisation over up to 10⁴ states × 100 inputs is about 10⁶ transitions: feasible, but budget the
   certificate time.
5. **The start state as text** is the first segment's job. Check start-decode accuracy separately, so that a start-encoding failure
   isn't read as a carriage failure.
6. **Shortcuts:** none of the item-visibility kind (each item is seen once). But for short L, carrying a few items' digits and
   summing at the end is possible within FSQ-8's 24 bits. Read exactness at long L (≥ 32), as with the §128 families.

## (B) MULTI-SLOT: K = 4 carried vectors per hand-off
**Priors.**
- **Recurrent memory with several vectors:**
  - Memory Transformer 2006.11527 (memory tokens);
  - RMT 2207.06881 (memory tokens written at a segment's end, read by the next);
  - ARMT 2407.04841;
  - Block-Recurrent Transformer 2203.07852 (a recurrent state of many vectors).
- **Latent CoT with several vectors:** Coconut 2412.06769 (c thoughts per step); PCCoT 2506.18582 (parallel continuous thoughts,
  Jacobi-updated); pause tokens 2310.02226 (compute without carried state; our §107 PAUSE-4 didn't help).
- **Slot specialisation:**
  - Slot Attention 2006.15055: symmetric slots specialise by competing (a softmax over slots), by *object*, not by role.
  - RIMs 1909.10893: K modules specialise by sparse competitive activation.
  - **Digit-position specialisation of carried slots: not found.**
  - The nearest evidence that a significance-indexed layout matters is Abacus 2405.17399 (an explicit per-digit significance
    embedding), in token space.
- **New as scoped:** K quantised carried slots in a pretrained LM's Markov latent chain, scored for per-slot specialisation and
  donor SERIAL.

**Pitfalls.**
1. **K slots add compute as well as capacity and layout.**
   - K write positions are K extra positions of computation in the segment (like pause tokens).
   - **Control:** K = 4 write positions but only slot 1 carried (the others discarded), which matches compute.
   - Without it, a win can't be attributed to the multi-vector *state*.
2. **Symmetry and redundancy:**
   - Distinct GPT-2 positions already break exact symmetry. But with nothing pushing slots to split, expect **redundant copies**
     (each slot holds the whole state) or **one active slot** (the others ignored).
   - Per-slot W_down isn't needed to break symmetry. **Shared W_down + a learned slot-ID embedding** (added at the write and the
     read positions) is the RMT / Abacus-style choice, and it scales in K.
   - Per-slot W_down is affordable at K = 4. If you use it, keep the slot ID as well, so the next segment knows which slot is which.
3. **Diagnose specialisation directly:**
   - a per-slot × per-digit (or per-accumulator-range) probe matrix;
   - **per-slot donor swaps** (swap slot k only), so C1 is scored per slot;
   - the per-slot ablation drop.
   - "Digit-like layout" is shown only if the matrix is near block-diagonal and single-slot swaps move the matching digits.
4. **Decoder reading K slots:** feed all K as consecutive input positions with slot IDs. Attention can route digit d to slot k. Make
   sure the decoder's answer fork sees *only* the K slots (GEN-M-MASK's lesson), or it can bypass them.
5. **Per-slot quantisation:** K × FSQ-8 rounds independently. NOISE-64 per slot (K × 64 continuous) loses rounding's correction on
   all of it. If a hybrid is tried, keep at least one slot rounded (cf. §143's split).
6. **The motivation (S_j has linear R² 0.174 from one hidden state) is about TC-M's h, not about what a trained carrier can hold.**
   Note 140-3 showed the *learning* failure in h too. K slots help only if a multi-vector layout makes the carry computation
   easier to learn, which is plausible (Abacus) but untested.
   - Run (A) first. If 1 × FSQ-8 carries mod 10⁴ exactly, the §140 failure is about the in-pass computation (the column product +
     add), not carriage, and K slots are less likely to be the fix.

## Order
(A) is cheap and decides how to read (B): carriage vs in-pass computation. Recommend running (A) with the ladder before (B), or
alongside it if GPUs are idle.
