# Language latent loop: findings summary (§234–§250), 2026-10-30

Scope for every line: one model lineage (GPT-2 askname init → P-2 → REG → DF → LH), templated word programs (mod 11), the step cue gives the plan. Details live in docs/THEORY.md Notes 234-x … 250-x.

## The model
- GPT-2 small. Each latent pass reads a per-step cue ("How many <u> does <S_t> have? The answer is:"). The ':' state is written into a DeltaMem recurrent memory, and a decoder pass reads the memory to answer.
- Training: per-pass value targets (aux CE), plus a hold-increment penalty (§240).

## North-star scorecard
| Criterion | Status | Key evidence |
|---|---|---|
| C1, the thoughts do the work | **Met, scoped** (plan given; training phrasings) | §237 / §238: d ≤ 4 at 1.000 vs a one-pass comparator at chance. The intermediate is read at every pass, every op is read, and mid-chain swaps are followed (0.98–1.00), incl. position-independent cross-state products. 2 data orders |
| C6, stable | **Met, scoped** | §240: hold to K′ 128 (16× trained) at 1.000, d 1–4, with the hold-increment penalty; ‖S‖ ratio 1.1 vs 15 untreated; C1 gates retained |
| C5, decides how long | **Partial** | §248: a post-hoc learned halt head (self-consistency labels, frozen model) halts at ~d passes with accuracy kept (≥ 0.993). The softmax alone is confidently wrong (§247). Fails out of range |
| C3, beats text CoT at matched compute | **Split** | Note 246-0: ~9× fewer serial layers, but 1.5–3× more FLOPs (each pass recomputes the prompt). Fix = KV-cached latent positions (deferred, §246-1) |
| C2, more thinking helps past training | **Not met** | §239 / §240 / §243: passes past the trained budget add nothing. Wider-depth training extrapolates 2–3 depths further (§243, descriptive) |
| C4, no hand-built intermediate states | **Not met** | §244 / §245: without per-pass targets the d 2 loop formed in 1 of 4 seeds. No C4-admissible teacher exists in this setup |
| C7, faithfully readable | **Not met** | §247: the printed values are faithful readouts, but text fed back through the write path is ignored (edits change nothing) |

## What limits it (diagnosed)
1. **Position priors in the data.** A fixed final op let passes take shortcuts (§234–§235). Fixed by iid op laws (§236–§238).
2. **Steps past the trained count are never computed.** The cause is compute, not storage or text length (§241–§243).
3. **Phrasing:** the cued step is frame-specific (§249). Adding frames teaches only those frames (§250, MEMO).
   - Loop training did not cause it: the askname init is already narrow in the cued question format (Note 250-4).
   - The same init trained in the direct format (CD-AN) parses an unseen frame (0.971), so the narrowness is in the cued-pass format.
4. **Architecture cost:** the per-pass prompt recompute makes FLOPs uncompetitive (Note 246-0).

## Most promising next directions (each a new staged design)
- **KV-cached latent positions** (C3 FLOPs), re-using REVIEW_KV's fixes.
- ~~A pass format that inherits the direct path's phrasing generality~~ (Note 250-6: the direct model does not compute v_1 at d 2 even in-frame (probe 0.19), so it is not a usable source; parked). Phrasing generality of the step remains open.
- **Learned halting jointly with the loop** (beyond the post-hoc head), and externalisation (C5).
- **A C4 route:** RL / self-training from answer rewards, once a model can already write steps.

## Spend
§234–§250 realised ≈ $22.9 (cloud); Vast credit $30.56.
