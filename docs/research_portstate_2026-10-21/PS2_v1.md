# PORT-STATE v1: a portable re-rooted latent state on a templated word task (2026-10-21)

**Status:** v1. It needs Literature's novelty check (OK / narrow / withdraw per claim; the user's 10-21 rule) and then Audit (cap ≤ $30). Draft: PS2_draft.md (80113e90). Review: REVIEW_PS2.md (421566fb; Opus; 3 BLOCKER / 8 MAJOR, all addressed below).

## 0. Question, premise, honesty
- **Premise (Note 221-2, floors cleared):** CODI-GPT2's GSM8K latents carry **intermediate values**, while the program and remaining operands are re-read from the question. A donor's latents therefore yield the counterfactual (0.33–0.54 vs floor ≤ 0.04), not the donor's answer (§221: 0.03).
- **Question:** can a CODI-style latent carry the **remaining program and its operands** (the re-rooted problem R_t), so that a mid-chain swap makes the continuation finish the **donor's** problem while the recipient's conflicting question stays visible? Tested on a templated word task whose depth exceeds one forward pass.
- **Novelty, pending Literature:**
  - M1 is SIM-CoT's (2509.20317) step supervision with a **re-rooted-state** target, read **question-blind** from the swapped object.
  - §175 ran a question-blind re-rooted-expression decoder on synthetic data.
  - §211 / §213 showed portable states on synthetic data (VALUE-SWAP 1.000).
  - §218's CONSUMPTION-FRONT is the mirror-image input switch.
  - **The delta:** the CODI family on a word task; a trained answer bottleneck; a mid-chain graph-matched donor swap with the recipient's question visible; a decode-free BUDGET-FRONT.
- **A null is the likeliest outcome** (prior ≈ 0.10, §6). Results are worded as "templated word task, one seed", never GSM8K, never C4.

## 1. Task: NESTED-WORDS v1 (REVIEW B3, M6)
- **Programs:** a straight-line program of n steps over named quantities, with a forced **dependency spine** (each step reads the previous spine quantity), so the DAG depth equals d = n.
  - At least one step per program **multiplies two variables**, which breaks the solvable-group / affine-chain shortcut (§170 literature flag).
  - Values are **bounded natural integers** (results in 0–99, regenerated if out of range); multiplicands are nonzero.
- **Language:**
  - **8 paraphrase frames per operation** (2 held out at test).
  - Sampled names; 0–2 numeric **distractor** sentences; statements in shuffled order where the DAG allows.
  - The question is the last sentence.
- **R_t:** the problem after the first t spine steps are computed, re-rendered with computed quantities as constants. The frame, names and order are kept, and parallel branches are reduced by a fixed tie-break (leftmost first). R_d = the answer sentence.
- **Splits:** train d 1–8 (CurGate curriculum); test d 1–8 plus extrapolation d 9–12 (descriptive). Test and pair sets have a sha. Exclusion by rendered match.
- **G-W0 (before any latent cell):**
  - (i) a heuristic shortcut panel (any single-feature rule ≤ 0.6 at every d, as §218 G-L0);
  - (ii) **matched-budget DIRECT cell** (§3 cells) gives one-pass accuracy by d. **n\*** := the largest d with DIRECT > 0.50.
  - The **C1 band** = {d ≥ n\*+1 : DIRECT ≤ 0.50}, and it must contain at least 3 depths in 3–8. Otherwise the task is redesigned before any latent cell.
  - n\* is fixed from DIRECT before any latent output (as 170-3).

## 2. Model and objective (the method change)
- **Base:** GPT-2 small, the CODI-style continuous latent loop (boxes/codi_port.py). Pinned: z_1 = prj(h at `<bot>`), z_{j+1} = prj(h_last at latent j), **z_j ≙ R_j** (latent j's target is the problem after j steps). K = d + U{0..2}, capped at 10. Batches are bucketed by d (one n_latent per batch). The question is **visible to every latent step**.
- **(M1) STATE-RECODE:**
  - A **separate decoder head of ≤ 2 transformer layers** reads **z_j only** (one vector, no KV from the question). It emits R_j by teacher-forced CE.
  - Loss = per-token mean over R_j, with **value tokens scored separately** (reported).
  - **λ = 1** on the per-token mean. This is stated as a choice (the §175 / SIM-CoT value), not derived.
- **(M2) ANSWER BOTTLENECK:** the answer phase is masked by an **explicit attention mask** from question content **and `<bot>`** (REVIEW M2). It reads the latent rows plus answer-prompt tokens only.
- **No** question-once, interchange training or dynamics targets.
- **Loss:** CE(answer) + λ · mean_j CE(D(z_j) → R_j).

## 3. Cells (one seed each, RTX 3090, 24k steps, bs 64; REVIEW M7)
| cell | M1 | M2 | role |
|---|---|---|---|
| **DIRECT** | — | — | the one-pass baseline (no latents) for n\* and the C1 band |
| **PS** | ✓ | ✓ | the test cell |
| **M1-only** | ✓ | — | attribution (the question still visible at the answer) |
| **M2-only** | — | ✓ | attribution (§208-like bottleneck on a deep task) |

Same seed, stream, curriculum and budget across cells. Single-change args diffs against PS are checked (frozen refs).

## 4. Derivation [tags]
- **D1 (which swaps are informative) [CORRECTED, REVIEW M1].**
  - Under M2 with `<bot>` masked, the answer reads latent rows only. A **full CACHE-SWAP** (all latent rows) therefore follows the donor by construction. It is a sanity check, not evidence.
  - A **full STATE-SWAP** (z_1..z_K fed as inputs, the recipient's forward recomputing every latent row with the recipient's question visible) is **not** trivial: it equals PREFIX-SWAP(K). It is reported as a descriptive arm.
  - The registered swap is **PREFIX-SWAP(t)** at the **state granularity**: the donor's z_1..z_t are fed as inputs, and steps t+1..K run in the recipient's context with its question visible.
- **D2 (decodable vs used) [PASS].** M1 makes R_j decodable from z_j. It does not force the continuation to use z_j over the visible question, because in training they always agree. That is exactly what PORT-FOLLOW measures, and why a null is informative (the §203 lesson).
- **D3 (what counts as computation in the latents).** "Portable computed state" holds iff all three parts hold:
  - (a) R_j's **value tokens** are decodable from z_j (DECODE-VAL ≥ 0.80 at the C1 band, t ≥ 2) **while COPY-CONTROL stays at floor** (the same decoder architecture trained on a **question-only** encoding, i.e. z from a K0 pass, must read R_j's value tokens at ≤ floor + 0.05 for t ≥ 2; Audit's C7 input-only rule);
  - (b) the state is **used**: PORT-FOLLOW **and** BUDGET-FRONT;
  - (c) it is **not precomputed**: ANSWER-ONSET.
- **D4 (BUDGET-FRONT, decode-free; REVIEW B1) [DERIVED; checked with stubs].**
  - For a swap at t, let b\* = the smallest remaining step budget (K′ − t, with steps truncated after t + b) at which donor-final first appears.
  - **A serial state** computes the donor's remaining d − t steps, so b\* ≈ d − t falls with t: slope −1 within d.
  - **Relay** (z_t holds the donor's question text, and later steps recompute everything) gives b\* ≈ d, flat in t.
  - **Precompute** (z_t already holds the answer) gives b\* ≈ 0, flat.
  - Registered: within each d in the band, Spearman(t, b\*) ≤ −0.6 and an OLS slope ≤ −0.4 over clean pairs.
  - The F3 known-answer stubs must give serial −1 and relay / precompute 0 before any model reading.
- **D5 (Goodhart routes, each with an exposed guard; REVIEW §5):**
  - copy-question decoding → COPY-CONTROL;
  - relay → BUDGET-FRONT;
  - precompute → ANSWER-ONSET plus BUDGET-FRONT;
  - the `<bot>` / marker leak → MARKER-SWAP (swap BOS and `<bot>` only; follow ≤ floor + 0.05);
  - D memorising R_j from priors → a z-shuffle control (D on a random item's z → chance on value tokens);
  - C1 removal by K0, which is vacuous under M2 → read through MEAN-Z and truncation instead.
- **D6 (precompute risk) [MEASURED analogue].** §170 DIRECT: n 3 0.635 / 0.750, n 4 0.430 / 0.533, mean n 4–8 0.23 / 0.29 vs chance 0.09. So one pass is far from chance at d 3–4. The band from G-W0 (d ≥ n\*+1) handles this, and chunked work-ahead is tolerated by BUDGET-FRONT's slope rule.

## 5. Readings (PS cell; same on M1-only / M2-only for attribution)
- **Pairs (REVIEW B2):**
  - graph-matched skeletons (identical spine and op sequence; different constants and names), exact token length, **same names** for the primary;
  - kept only if **own, donor-final and counterfactual are all distinct** (the share is reported);
  - **N ≥ 1,000 exposed pairs** over the C1 band; pairs with different names are descriptive.
- **PORT-FOLLOW (primary):** PREFIX-SWAP(t = ⌈d/2⌉) donor-final follow ≥ permutation floor + 0.30, pooled over the C1 band, with:
  - the **leak control**: a LEAK-SEG analogue (the donor's last 6 question-content rows) ≤ 0.25;
  - **MARKER-SWAP** ≤ floor + 0.05;
  - SELF-SWAP bitwise.
- **Co-primary guards:** **BUDGET-FRONT** (D4) **and ANSWER-ONSET** (the final answer decodable from z_t at ≤ floor + 0.05 for t ≤ d − 2, by the D head on the answer token).
- **PORT-STATE** = DECODE-VAL ∧ COPY-CONTROL ∧ PORT-FOLLOW ∧ BUDGET-FRONT ∧ ANSWER-ONSET.
- **Labels, in precedence:**
  - UNREAD;
  - M1-NOT-MET (R_j not decodable);
  - **PORT-STATE**;
  - **PORT-RELAY** (follow, flat front, b\* ≈ d);
  - **PORT-PRECOMPUTE** (follow, ANSWER-ONSET fails);
  - **CARRIER** (counterfactual follow ≥ floor + 0.30, as 221-2);
  - **CONTEXT-BOUND**;
  - MIXED.
- **Descriptive:**
  - full STATE / CACHE swaps;
  - INT-FOLLOW;
  - QMASK / QDROP;
  - QUESTION-SWAP(t) (donors differing in the remaining program);
  - STATE-ONSET;
  - STALL;
  - LOGIT-LENS;
  - the K-sweep;
  - d 9–12;
  - LEAD(k) (R_{j+k} decodable from z_j);
  - a C7 round-trip (decode R_j → re-encode as question → continue).
- **Attribution:** M1-only and M2-only read the same labels. PORT-STATE in PS but not in either ablation reads "needs both".

## 6. OC (REVIEW §7; review_scripts/oc.py)
- **Floor and pairs:** floor ≈ 0.09 (permutation), design effect 1.3, N 1,000.
- **PORT-FOLLOW alone:**
  - P(pass | no portability) ≈ 0 by sampling. Relay and precompute are structural false positives, which is why the conjunction includes BUDGET-FRONT and ANSWER-ONSET.
  - P(pass | true follow = floor + 0.32 / 0.35) ≈ 0.88 / 0.998.
- **Guards:** P(BUDGET-FRONT | relay or precompute) ≈ 0 (the slope bar against a flat front, checked by stubs). P(BUDGET-FRONT | serial) ≈ 0.8 [ASSUMED].
- **Priors [ASSUMED, shaded]:** P(PORT-FOLLOW) ≈ 0.12. **P(PORT-STATE) ≈ 0.08–0.10.** A null is the likeliest outcome, most likely CARRIER or CONTEXT-BOUND, which is informative for the language-C1 route.

## 7. Censoring (rule b)
- Dead, diverged, contaminated or ARGS-DIFF → every reading fails.
- G-W0 failing (band < 3 depths) → no latent cells.
- DECODE-VAL failing → M1-NOT-MET (not a pass).
- < 1,000 exposed pairs → PORT-FOLLOW UNREADABLE.
- Never-reached b\* → b\* = the maximum budget + 1, counted against the slope.
- A missing file reads pending (215-7).

## 8. Gates ($0 unless stated) and cost
- **F1:** generator + G-W0 shortcut panel (CPU).
- **F2:** a timing and memory gate for the four cells at the deepest shape (3070 / 4060).
- **F3:** known-answer stubs for BUDGET-FRONT / ANSWER-ONSET / MARKER-SWAP / COPY-CONTROL (CPU).
- Audit's diff review of the codi_port changes (bucketing, explicit masks, the D head); a plain-path smoke; per-term gradflow; G-D.
- **Cost:** ≈ 0.8–2 s/step on a 3090 at bs 64, so ≈ $1.5–4 per cell; **4 cells ≈ $6–16.**
- **Family cap $20** (Audit approves ≤ $30); per-cell guards; off-host full-state checkpoints with a tested resume; one 3090 class.
- **DIRECT runs first:** it fixes n\* and the band, then PS / M1-only / M2-only.

## Addendum A (2026-10-21; Audit's four requests, before approval; v1_checks/oc_stubs.py + .json, rng 22801)
1. **BUDGET-FRONT rule corrected by the known-answer stub (v1.1).** As first written (per-d Spearman ≤ −0.6 at every d), the stub failed even a perfect serial model, because at shallow d there are only 2 values of t, plus chunking noise and naive tie-ranking. **Registered rule v1.1:**
   - pooled over the C1 band with **depth fixed effects** (t and b\* centred within d);
   - **tie-aware (average-rank) Spearman ≤ −0.6** and **within-d OLS slope ≤ −0.4**, over ≥ 1,000 clean pairs.
2. **OC per co-primary and joint** (stub model: b\* = (d − t) + geometric chunking noise, p 0.3; band d 4–7; 2,000 reps):

   | reading | under the effect | under the alternatives |
   |---|---|---|
   | BUDGET-FRONT | serial **1.00** | relay **0.00**; precompute **0.00**; half serial / half relay 0.00 |
   | ANSWER-ONSET | not precomputed **0.95** (5 %% false-fail from binomial noise over the t cells) | precompute **0.00** |
   | PORT-FOLLOW (floor 0.09, N 1,000, design effect 1.3) | true 0.42 / 0.45 / 0.60 → 0.96 / **1.00** / 1.00 | true = floor → **0.00**; true 0.30 → 0.00; true 0.39 → 0.50 |
   | **PORT-STATE (joint)** | portable serial, follow 0.45: **0.95** | no portability: **0.00**; relay or precompute even with follow 0.60: **0.00** |

   The prior stays at P(PORT-STATE) ≈ 0.08–0.10 [ASSUMED]. The stub shows that the guards remove the structural false positives.
3. **Single-change args diffs.**
   - PS vs M1-only must differ only in the answer-bottleneck flag(s).
   - PS vs M2-only must differ only in the STATE-RECODE flag(s) and λ.
   - DIRECT must differ only in the latent-loop flags (n_latent 0, no D head, no bottleneck).
   - The diffs are produced from each cell's `--args_only` output after the build, committed with frozen-ref shas before any launch. Any extra key → ARGS-DIFF (rule b).
4. **Gradient-flow plan** (gradflow rule; read from the decorators down):
   - Before any spend, one CPU / GPU step per cell runs with every loss term computed separately. Each term must have **nonzero grad** on the parameters it should train:
     - the answer CE → backbone + prj;
     - the R_j CE → the D head **and** prj / backbone through z_j (z_j must not be detached);
     - the bottleneck mask must give **zero** grad from answer CE into the question-content and `<bot>` positions' inputs, verified by a gradient probe on those embeddings.
   - No `@torch.no_grad` / `.detach()` on z_j in the train path (grep plus an ast check of decorators).
   - The plain-path smoke (construct the model + 1 step without the new flags) must reproduce the base CODI port.
   - The check outputs gate JSONs, which Audit reads before the G-D.
