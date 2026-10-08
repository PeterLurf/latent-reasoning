# Where C1 can be forced: a program note for the user's decision (Theory, 2026-10-08 08:34 EDT; after note 158-2, Literature 6b771087, Audit)

**Nothing here is registered.** Audit relays it to the user; registrations follow the user's choice.

## The finding (note 158-2; `docs/LIT_TC0_SERIAL_2026-10-08.md`)
* **A latent memory that can hold the inputs admits a non-serial solution:** buffer the inputs and compute at read time.
  - Gradient descent takes such parallel shortcuts when they exist (Liu 2210.10749).
  - ICoT-SI's own multiplier caches and retrieves partial products this way (Bai 2510.00184).
* **Serial use is forced only when one of two conditions holds** (Literature's refinements):
  1. **Depth:** the task needs more depth than one constant-depth read over the buffer can compute. This holds for S5 and other NC¹-complete tasks at long
     length. It does not hold for fixed-size multiplication: integer multiplication is in uniform TC⁰ (Chandra–Stockmeyer–Vishkin 1984; Hesse–Allender–Barrington 2002),
     and ICoT-SI's GPT-2 small does 4×4–9×9 in one pass at 0.95–1.00.
  2. **Capacity:** the input exceeds what the memory can buffer, as with long chains and lengths beyond training.
* **This explains the multiplication line.**
  - Every exact design (§155 R / R9 / SHIFT-R) is an operand buffer with read-time compute (155-4), and none is SERIAL.
  - The §158 preview buffers raw digits.
  - On fixed-size 4×4, "exact but not serial" is the expected optimum, not a design bug. More C1 fixes there would be a problem → fix loop.
* **Caveat:** buffering is admitted, not guaranteed (membership ≠ learnability). A per-step reader can tip the optimiser toward a serial state (§111).

## Proposed rule
**C1 is claimed only on tasks or lengths where a measured single-pass baseline fails** at that size, depth or length. A direct-answer model trained on the same
data is the baseline. Then K′ = 0 / 1 far below K (C1's operational test) is meaningful, because the latent steps are necessary and not just used.

## What would change (the user's call)
1. **Multiplication (≤ 9×9) becomes the C3 / C6 / C2 testbed.**
   - §159 (5×5), §160 (rows) and §161 (SHAPE-MIX) continue. §158 reads as registered, with 158-2's prediction.
   - No further C1 designs on fixed-size multiplication.
2. **The C1 + C3 headline moves to a task that exceeds one pass, with a text-CoT baseline.** Theory's ranking:
   - **(a) Deep arithmetic-expression evaluation mod p** (Feng 2305.15408's format).
     - Why it fits:
       - "Serial maths": NC¹-complete, with a published CoT vs direct gap at small scale.
       - The depth scales past one pass.
       - It needs computed sub-expression values held across passes, which is what the write-after-read memory does exactly (§155).
     - Design sketch: the expression in context, K latent passes through DeltaMem-R, the answer at the end, with the direct (K = 1) and text-CoT baselines on
       the same items.
     - C1 test: K′-removal, plus edits of a stored sub-expression value (propagation).
     - A new family adapter. A first proof pair costs about $5–10.
   - **(b) Shortcut-free ProsQA (ProsQA-NA) at a depth beyond one pass** (planning; the 2026-10-07 plan). At the standard depth an in-pass search is possible
     (that plan's own caveat), so the generator's depth must be raised and the direct baseline measured.
   - **(c) S5 head-to-head.** It is already SERIAL on the fsq120 carrier (§133 / §134 / §143), but scaffolded and not maths.
3. **Long multiplication by capacity** (4×n, n well beyond training): the C2 route, after §161 reads.

**Theory's recommendation:** 1 + 2(a), with 3 riding on §161.

## Addendum (2026-10-08 08:41 EDT): the cheapest direct test of the finding
* **The missing cell** (Literature: no published contrast of TC⁰ against NC¹ tasks on one memory).
  - Narrow carriers learn S5 serially (§133 / §134 / §143) and fail to learn multiplication (§137 / §140).
  - DeltaMem-R learns 4×4 multiplication as a buffer (§155).
  - **DeltaMem-R on S5** would complete the 2 × 2.
* **Prediction:**
  - state-like and SERIAL at lengths beyond both the buffer and one pass's depth;
  - buffer-like (op history decodable, swap uninformative) at short lengths, where log-depth shortcuts exist (Liu 2210.10749).
* **Placement:** an S5 adapter on multpair's DeltaMem (Experiments), short against long training lengths, about $2–3. It is independent of the choice under 2.
