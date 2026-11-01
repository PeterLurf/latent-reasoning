# Language latent loop: findings summary (§234–§250), 2026-10-30

Scope for every line: one model lineage (GPT-2 askname init → P-2 → REG → DF → LH), templated word programs (mod 11), the step cue gives the plan. Details live in docs/THEORY.md Notes 234-x … 250-x.

## The model
- GPT-2 small. Each latent pass reads a per-step cue ("How many <u> does <S_t> have? The answer is:"). The ':' state is written into a DeltaMem recurrent memory, and a decoder pass reads the memory to answer.
- Training: per-pass value targets (aux CE), plus a hold-increment penalty (§240).

## North-star scorecard
| Criterion | Status | Key evidence |
|---|---|---|
| C1, the thoughts do the work | **Met, scoped** (plan given; text in chain order; training phrasings. Note 251-16: with the op sentences permuted, d 2-4 fall to ~0.15-0.20, even pass 1. §253 / §254: a name-lookup init gives pass-1 lookup, but the chain write forms only under consistent order, and consistent order re-teaches position (Note 254-3)) | §237 / §238: d ≤ 4 at 1.000 vs a one-pass comparator at chance. The intermediate is read at every pass, every op is read, and mid-chain swaps are followed (0.98–1.00), incl. position-independent cross-state products. 2 data orders |
| C6, stable | **Met, scoped** | §240: hold to K′ 128 (16× trained) at 1.000, d 1–4, with the hold-increment penalty; ‖S‖ ratio 1.1 vs 15 untreated; C1 gates retained |
| C5, decides how long | **Partial** | §248: a post-hoc learned halt head (self-consistency labels, frozen model) halts at ~d passes with accuracy kept (≥ 0.993). The softmax alone is confidently wrong (§247). Fails out of range |
| C3, beats text CoT at matched compute | **Split** | Note 246-0: ~9× fewer serial layers, but 1.5–3× more FLOPs (each pass recomputes the prompt). A cached-prompt, 2-token-cue redesign would only reach parity with value-only text CoT (Note 251-12), and masking a trained loop keeps 1–2 hops only (251-13) |
| C2, more thinking helps past training | **Not met** | §239 / §240 / §243: passes past the trained budget add nothing. Wider-depth training extrapolates 2–3 depths further (§243, descriptive) |
| C4, no hand-built intermediate states | **Partial** | §244 / §245: without per-pass targets the d 2 loop formed in 1 of 4 seeds. §251 / §255: with per-pass targets on d <= 2 only, d 3-4 are learned from answers in 2/4 data orders (full causal bar 1/4; s3 / s4 never enter), so it is rare; dropping all targets after the seed fails (0/2) |
| C7, faithfully readable | **Not met** | §247: the printed values are faithful readouts, but text fed back through the write path is ignored (edits change nothing) |

## What limits it (diagnosed)
1. **Position priors in the data.** A fixed final op let passes take shortcuts (§234–§235). Fixed by iid op laws (§236–§238).
2. **Steps past the trained count are never computed.** The cause is compute, not storage or text length (§241–§243).
3. **Phrasing:** the cued step is frame-specific (§249). Adding frames teaches only those frames (§250, MEMO).
   - Loop training did not cause it: the askname init is already narrow in the cued question format (Note 250-4).
   - The same init trained in the direct format (CD-AN) parses an unseen frame (0.971), so the narrowness is in the cued-pass format.
   - Note 251-6: the cued step picks its op from the nearest trained template; unseen phrasings fail all-or-nothing with systematic op substitutions ("exceeds ... by c" computes mul).
4. **Answer-only training (C4):** the seeded pass codes drift at onset on items whose answer is not the current step (probe u1|v1 0.11-0.25). Only per-pass targets hold them (Notes 251-8, 251-9, 251-11); depth beyond the seed can come from answers if the seeded passes stay anchored (§251 AEX).
5. **Architecture cost:** the per-pass prompt recompute makes FLOPs uncompetitive (Note 246-0).

## Most promising next directions (each a new staged design)
- **KV-cached latent positions** (C3 FLOPs), re-using REVIEW_KV's fixes.
- ~~A pass format that inherits the direct path's phrasing generality~~ (Note 250-6: the direct model does not compute v_1 at d 2 even in-frame (probe 0.19), so it is not a usable source; parked). Phrasing generality of the step remains open.
- **Learned halting jointly with the loop** (beyond the post-hoc head), and externalisation (C5).
- **A C4 route:** RL / self-training from answer rewards, once a model can already write steps.

## Spend
§234–§250 realised ≈ $22.9 (cloud); Vast credit $30.56.
