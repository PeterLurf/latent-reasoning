# Prompt for an external agent: making latent thoughts do the work (C1), 2026-10-01

Given to a parallel research agent with no access to this repo. Its reply goes to a dated EXTERNAL_IDEAS file and needs
Literature's check and a THEORY registration before anything runs.

---

**Context.** We study *latent reasoning*: a language model that reasons in hidden vectors fed back into itself (as in
Coconut, arXiv 2412.06769) instead of in written chain-of-thought. Our goal is one pretrained LM (GPT-2 small, 124M)
whose latent thoughts **demonstrably do the work**, i.e. the following tests pass:
- removing the thoughts (zero latent steps) drops accuracy to chance on problems that exceed one forward pass;
- the same architecture without latents (same scaffold, no thoughts) does clearly worse;
- it is trained from final answers only (no step-by-step labels; curricula of shorter problems are allowed);
- it holds on shortcut-free test sets and beyond the training depth.

**What we found (all pre-registered, seeds ≥ 3–10, independently re-checked).**
1. *Coconut does not use its thoughts.* Our reproduction scores 97.4% on ProsQA with 6 thoughts and 97.6% with 0. ProsQA
   has a generator shortcut (the wrong option hangs off another person): a no-reasoning rule scores 91%. On a
   shortcut-free version Coconut drops to 74%. Coconut variants trained on 7–10-hop problems are at chance (50–53%)
   without the shortcut; text CoT holds at 75–77%.
2. *Looped models stop at their training depth.* In small looped transformers on graph reachability, the loop keys a
   fall-back to an internal step count. Restarting from a written-down state fixes 31/34 models. Distilling those
   restarts back into one loop works only when the self-labels are near-clean; noisy labels keep the counter.
3. *Continuous states drift.* On permutation tracking (S3), a latent loop is perfect up to its trained 8 operations,
   then its outputs stop being valid permutations (valid share 100% → 14% at 16 ops), and more latent steps make it
   worse. Restarts from a written state: 99.5%. A5 never trained, even with a published curriculum.
4. *In GPT-2 the scaffold, not the latents, does the work.* Permutation tracking (S4) in chunks of 4 ops with the state
   written as text and the history cut is 100% at 17–32 ops (text CoT trained to 8 ops collapses to chance). But the
   same chunks with zero latent thoughts score identically.
5. *A discrete public state can learn exact reasoning, unreliably.* A per-entity on/off label channel in GPT-2, trained
   from final answers, learned exact 14-hop graph search in 1 of 14 runs; the others lock into an over-marking rule.
   GPU numerical noise decides which attractor wins.
6. *Failed:* learned halting, count clamps, training on transplanted aged states, bidirectional search, a learned
   boundary-response memory, a behavioural thought-to-text translator, straight-through hard labels.

**Our diagnosis.** (a) *No pressure:* at the trained depth the task is solvable in the forward pass or by a cue, so
gradient descent never needs the latent channel. (b) *No error correction:* continuous states accumulate error past
their trained length, while written discrete states snap back to valid values. Every past-length failure we saw is a
quantity that grows with the problem but was bounded in training; every fix kept it in range by restarting from a
written state.

**What we want from you.** Propose 3–5 concrete designs where latent computation *must* do the work *and* stays
valid, trainable from final answers on GPT-2 small within a few GPU-hours per run. For each give:
1. the mechanism, and how it addresses both (a) and (b);
2. the cheapest decisive experiment: generator-built task, the zero-latent and same-architecture-without-latents
   controls, and a pass/fail threshold fixed in advance;
3. the nearest prior work and how yours differs;
4. its most likely failure mode.

Please critique our own candidate: latent states snapped to a learned codebook every few steps (quantised latents), on
tasks verified beyond one forward pass before training. Avoid designs that amount to writing text chain-of-thought or
that need step labels. Be explicit about uncertainty.
