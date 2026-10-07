# Research program: big architectural bets after the multiplication post-mortem (2026-10-07 16:49 EDT; Theory, from Audit's sweep and the user's directive)

The user (16:40, via Audit): *"do a wide research sweep… come up with major new architecture changes we can add to the pretrained transformer and learning
paradigms. Don't get stuck in the minuscule improvements loop. Take the lessons we learned and apply them in new techniques."* Prior check: Literature 87cd2af5 (`docs/LIT_ARCH_SWEEP*`) and 75f51298 (`docs/LIT_BIG_BETS_2026-10-07.md`): all three architectures exist in some form (Infini-attention / ARMT; Feedback Transformer / TransformerFAM / T²MLR / CoTFormer; recurrent-depth retrofits), and **their use here is new**: arithmetic and state tracking, the new channel as the only cross-step route (C1 by design), scored by SERIAL and exactness. A removal ↔ loop coupled schedule is new.

## Lessons every new design must respect
* **L1 Shortcut.** If a step's inputs stay visible, the latent carries the inputs (the multiplier prefix) and recomputes. Readers and anchors do not stop this
  (114-4, §121 / §122). **First tests use tasks where carriage is forced:** digit-pair, the running sum.
* **L2 Numbers.** With carriage forced, a single-vector hand-off does not represent a growing accumulator, not even in the top hidden state (140-3). Text CoT never
  holds a number in one vector (139-2: 0.174). Numbers live across positions.
* **L3 Finite state works.** A bounded Markov carrier is exact to 2,048, SERIAL, contractive and certified on S5 and four other families (§128, §133).
* **L4 Width / installation.** Full width from scratch fails at entry (§138, §126). Growth from a narrow entry is safe, and the new dimensions are used (§136).
  Install any new module function-preservingly.
* **L5 Entry.** Per-step targets, anchors and restarts drive entry. Plateaus and escape after the 2k flag are the failure modes.
* **L6 Bypass.** Natural (non-Markov) latents are bypassed (§104), so C1 must come by design: the new channel is the only route between steps.

## The bets, ranked (north-star order C1 > C6 > C4 = C2 > C3 > C5 > C7)
1. **B1 DELTA-STATE: a linear-recurrent associative memory as the only cross-step channel** (Literature's top rank, with the additive path).
   - One gated DeltaNet layer (delta rule; β ∈ (0, 2), so eigenvalues can go negative) is inserted mid-stack in GPT-2 as a residual branch, with a zero-initialised output
     (L4).
   - Each segment is a fresh pass. Attention sees only the segment, and the DeltaNet's matrix state is carried between segments: C1 by design (L6).
   - **Why it targets L2 and L3:** a running sum is a linear update, so per-column sums can sit as key-indexed values and accumulate (carry-save) with no squashing.
     Erase-then-write with negative eigenvalues tracks finite state.
   - **Literature's flag:** a delta write *replaces* a key's value, so sums need an **additive, no-decay path** (6 additive + 6 delta heads). Carry resolution moves to
     the readout.
   - **First test, §147** (about $0.4): §140's digit-pair 4×4 (L1-free; FSQ-8 got 0.002), streamed digits, PASS = exact + SERIAL (no ρ term: an integrator by design),
     IMPROVES on columns 3–7.
2. **B3 RECURRENT DEPTH: think in depth, not across segments.**
   - A prelude / weight-tied loop / coda retrofit of GPT-2 (blocks 4–7 looped r times), with input injection and damped updates. r grows from 1 (function-preserving,
     L4), with deep supervision across loops.
   - **It targets in-pass compute:** §137's columns failed in-pass. That serves C3 (implicit, competitive), C2 (r beyond training) and C5 (stop at convergence).
   - §114's looped depth failed only under L1's shortcut.
   - **First test, §148** (about $0.8): direct-answer 4×4 (no CoT, no latent chain), LOOP-4 against PLAIN at the same budget; C2 at test r = 8 / 16.
   - ICoT-SI reaches 1.00 with no loops, so B3's distinctive claims are C2 and C5 (Literature).
   - **Second stage, if it improves:** ICoT-SI's stepwise internalisation into depth, with loops added as CoT tokens are removed.
   - **§148 read no improvement (2026-10-07 18:26 EDT):** LOOP-4 = PLAIN = 0.002, with the loops unused. **B3 stops.** The direct baseline fails at the same middle columns as
     every latent design, so the shared wall is learning multi-product column sums at this budget.
3. **B2 MID-STACK HAND-OFF: carry a middle layer, not the top.**
   - **$0 first test, §146** (registered; running): is the accumulator readable mid-stack where the top loses it?
   - If MID-STACK SIGNAL, a mid-layer carrier is registered on digit-pair, and B2 moves up, since it is the cheapest. If NO SIGNAL, it stays third.
   - **§146 read NO SIGNAL (2026-10-07 16:52 EDT):** layers 1–12 are flat at 0.10–0.22 for accumulators 100–200. **B2 is parked**, and §147 keeps ℓ_m = 6.

4. **B4 VOCAB-SLOT, a working memory of token-shaped slots (the user's 18:20 idea; §150):** 4 slots on GPT-2's own vocabulary (Gumbel straight-through, soft → hard),
   as the only cross-step channel. FULL-vocabulary and DIGIT (abacus) arms are run on digit-pair 4×4.
   - New as scoped (Literature c583bb23).
   - Literature's prediction: emergent token states help arithmetic only when supervised. The DIGIT arm and the C7 token-semantics test make it direct.
   - Ranked beside B1: the first tests run in parallel.

## Paradigms applied across the bets
* **Function-preserving installation** (zero-init outputs, fade-in, μP-scaled learning rates; §136 / §143) for every new module.
* **Shortcut-free task design first (L1).** Multiplier-major only with SERIAL and the b-prefix rate.
* **Deep supervision** across loops or steps (TRM).
* **Strict C4 later:** RL on discrete latents (synthesis candidate 5; ATF), only after a supervised carrier works.

## Deprioritised (the user's "minuscule improvements loop")
* Further variants of the bounded single-vector carrier for arithmetic, after §142 / §144 / §145 read. Width, rounding, readers, repeats, anchors and growth have each been
  tried (§107 / §114 / §121 / §122 / §133–§145).

## Order and budget
* §146 ($0, now).
* §147 and §148 proofs (about $1.2) after Audit's review, in parallel with §142–§145.
