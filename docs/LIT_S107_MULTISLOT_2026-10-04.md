# Prior check for §107: m latent slots per op (capacity vs compute) on 4×4 multiplication (2026-10-04)

Literature session, for Theory; Audit copied. Brief. arXiv IDs as verified earlier (LIT_INDEX).

## Priors
- **Several continuous thoughts per step:** Coconut 2412.06769 (c thoughts per replaced step; GSM8K uses c = 2, and its
  ablation favoured more); CoLaR 2505.16552 (the compression factor trades latents for accuracy); SuperThoughts
  2606.13862 (2 tokens per latent).
- **Capacity of one latent is not the obvious limit:**
  - MUX 2607.18264: one latent losslessly encodes a *span* of CoT tokens (a position-weighted superposition).
  - CoT2 2505.23648: the number of parallel traces is bounded by the embedding dimension.
  - A 768-dimensional state holds an ≈ 8-digit running sum (~27 bits) easily.
  - Superposition caveats: 2505.12514 (it works when trained from scratch), 2604.06374 (unused in fine-tuned LMs).
- **Extra filler positions add *compute*:** pause tokens 2310.02226; London & Kanade 2505.21024 (pause tokens strictly
  increase constant-depth expressivity).
- **Multiplication-specific:** 2510.00184. Models stall at a local optimum lacking long-range dependencies, and a
  running-sum auxiliary target (a linear regression probe) unlocks learning.

## The reframing that matters for the design
- Slot 3 must compute S_3 = S_2 + a·b_3·10², a 4-digit × 1-digit product plus a carried multi-digit addition, in
  one pass from a latent S_2. S5's 120-way state needed only a table lookup per op.
- So the blocker may be **compute per op**, not **capacity**. m latents per op supply both.
- **Add a separating control: m *pause* positions** (non-latent fillers, the same count, no carried content).
  - If pause tokens unblock slot 3 as well as m latents do, the gain is compute, not carriage.
  - Also consider a **running-sum regression probe loss** on the slot (2510.00184) as a cheap alternative arm.

## Pitfalls
1. **The extra latents go unused or become copies.** Only the last latent is read by the fork, so latents 1..m−1 have
   no target: slack.
   - Our Prop. 4/9: slack invites dwell or clock schedules. SIM-CoT 2509.20317: latents collapse as their number grows
     without per-latent supervision.
   - Measure the pairwise cosine between the m latents and their per-latent removal effect.
   - Consider a weak per-latent target (e.g. the partial product P_j on latent 1, S_j on latent m) to break the
     symmetry.
2. **Position and length.** MARKOV segments grow by m positions. Keep contiguous re-indexing (LIT_S99_MARKOV) and the
   same construction at train and test.
3. **Compute accounting.** m latents per op multiply sequential passes by m. Report it against TC-M and ICoT-SI.
4. **Pass wording.** Op-3 ≥ 0.95 with removal at the floor shows carriage. With the pause control it can also say
   *why*: compute vs capacity.
