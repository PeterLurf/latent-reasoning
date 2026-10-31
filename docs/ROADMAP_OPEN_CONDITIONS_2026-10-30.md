# Roadmap: the open north-star conditions of the language loop (2026-10-30)

**Inputs:** docs/LANGUAGE_LINE_SUMMARY_2026-10-30.md; Notes 234-x … 250-6. Budget ≈ $25 of the remaining Vast credit ($30.56); steps ≤ ~$3. North-star priority C1 > C6 > C4 = C2 > C3 > C5 > C7. C1 and C6 are met (scoped); C5 is partial.

| # | Condition | Cheapest decisive next experiment | Cost | Prior | Unblocks | Score (priority × prior / cost) |
|---|---|---|---|---|---|---|
| 1 | **C4 (partial)** | **ANSWER-EXTEND:** seed the loop at d ≤ 2 with per-pass targets (P-2 exists), then extend to d 3–4 from **final answers only**, vs a matched aux-on control. Asks whether depth beyond the seed needs hand-built states. | ~$3.2 (4 × 8k FT) | 0.35 | The scale story: supervision for new depths from answers; the precondition for an RL-style C4 | high |
| 2 | C3 (FLOPs) | **KV-LAT pilot** with REVIEW_KV's fixes (ln_f'd latent with a learned gain, K ~ U{d..8}, recomputed interventions), d ≤ 2. | ~$2.5 + a large build | 0.35 | FLOP parity with text-CoT; an architecture for scale | medium (build cost, thin novelty) |
| 3 | C2 | Variable-K training (K = d + U{0..4}) so passes after the last step are trained to *compute when the cue is new*, then test d 9–12 past the budget. | ~$3 | 0.10 | "More thinking helps" | low (three failures: §239 / §240 / §243) |
| 4 | Phrasing | $0 first: the op-dependence of the frame-6 failure (pass 1 0.52 on d 2 items vs d 1 mul_v 0.064). A design only if a single op / construction explains it. | $0 → ~$2 | 0.2 | Task-agnostic steps | low until the $0 read |
| 5 | C7 | Non-circular text-write: train the write path to accept text-encoded values on edit types A, test on held-out edit types B. | ~$2 | 0.25 | Faithful readability | low (lowest priority; circularity risk) |

**Update 10-30 (Note 251-3):** #1 done. AEX SPLIT; AEX0 FAIL. Answer-only depth +2 works only with the d <= 2 step targets kept as a replay anchor. New C4 candidate (Literature): a competence-gated depth curriculum (ATC 2609.19717) for the answer-only arm. Check its preconditions before drafting.

**Update 10-30 (Notes 251-6..251-9):** #4 phrasing read DIFFUSE (nearest-template op substitution). GATE-CURRIC parked; BLIND-DEC withdrawn. The answer-only C4 line pauses: the seeded writes break at onset; only per-pass targets keep them.

**Order:** 1 (ANSWER-EXTEND) now. Then the $0 phrasing read (4), which is free. Then KV-LAT (2) if credit allows (≥ $8 left after 1). C2 and C7 wait for a new idea.

**Why ANSWER-EXTEND first:**
- C4 shares the top open priority with C2, and its prior is higher.
- It reuses existing seeds, the build is small (`--aux_d_max`: aux CE only on items with d ≤ k), and it asks the practical question for scale: if per-pass targets are needed only to seed a skill, deeper skills can come from answers.
- Scope caveat: the seed itself uses hand-built targets, so it is C4-partial at best.
