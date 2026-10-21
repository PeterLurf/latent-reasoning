# CHUNK-SNAP: k latent passes per discrete checkpoint on the VERSION-BANK models (draft, 2026-10-20)

**Status:** DRAFT, not registered. $0, eval-only, local. Theory (research step after §215–219). For proof check (PROOFS_CS.md) and
red team (REDTEAM_CS.md), then v1, then Audit (under the user's $30 rule, cap $0).

## 0. Why this, and why now

**Where the line stands** (state synthesis, 10-20):
- C1 on the synthetic VERSION-BANK line is met in range and at the new depths (§213 VB-C1; §217 CAUSAL-PRESERVED). On language it is unmet.
- Literal C2 is unmet everywhere on nested depth:
  - LIT-C2-2 0.018: passes past K 20 do not help at n 22–24.
  - STOPS-AT-10 on the bases (213-26).
  - STOP-LOC moves with support, 15 → 21 (§216).
  - n50 / support is 1.28–1.31 across five points and two seeds (216-9).
- The model's **single step generalises far past its latent depth.** The re-rooted one-pass chain (k = 1) labels n 20 at ≥ 0.997 post-filter precision on D3-T (support 8), where its latent rollout scores ≈ 0.12 (213-35 / 213-38). R2-s1's chain keeps 0.89 at n 24 with prec_kept ≥ 0.9956 (§216 LABELLER-KEPT-2).
- **The two extremes are measured:** k = 1 (one latent pass per written checkpoint) and k = ∞ (all latent). **Nothing in between has been measured.**

**The open mechanism question** decides what the latent line should do next:
- **H_pass (a pass-count deadline):** the latent rollout's first k passes are accurate on inputs of *any* size, as long as k is within a model-specific latent horizon k*. The horizon grows with support, and the loss is in the late passes (the edge collapse).
  - Then depth is bought by restarting: k latent passes, a written checkpoint, k more. Accuracy rises with the number of chunks without bound.
  - Under H_pass, a learned trigger for *when* to write (C5) is the next design.
- **H_size (a size-keyed step):** per-pass accuracy falls with the input's operator count m (COUNT-OOD, 187-3, in DeltaMem), so each chunk degrades with m whatever k is.
  - Then restarts help only through the re-rooted input shrinking, the best k is 1, and latent depth needs a size-invariant read.

**North-star mapping.** This is an untested direction, not a fix for §216's FT-DIVERGED / MEM-TRADED.
- **C2 (hybrid):** accuracy versus number of chunks at n far beyond every trained depth.
- **C5 (externalisation):** fixed k here, learned later.
- **C3:** written checkpoints n / k versus text CoT's n steps, and text CoT is context-infeasible at n ≥ 26 on GPT-2 (TS_v1 CTX-BOUND).
- **C7 (descriptive):** round trip at every checkpoint by construction, plus an EDIT arm.
- **C1** within chunks is already established (§217). It is not re-claimed here.
- The first test of §201 Prop 8 (latent → text checkpoint → restart) on a model whose write works. DeltaMem's write failed from n 11, so §201 T4 read SNAP-HURTS.

## 1. Models, card, items

**Models** (existing, no training; RTX 4060, fp32, tf32 off, eval_bs 16; one card for every comparison in this section):

| model | lineage | support (gold / self) | role |
|---|---|---|---|
| s213_d3t_s1 (D3-T s1) | s1 | 8 | primary |
| s215_ftself_s1 (FT-SELF) | s1 | 12 | primary |
| s216_r2_s1 (R2-s1) | s1 | 16 | **primary (CHUNK-C2)** |
| s216_d3t_s2 (D3-T s2) | s2 | 8 | second-seed replication of the base row |
| s216_r1_s2 (R1-s2) | s2 | 12 | descriptive only (§216 FT-DIVERGED cell; the model is evaluable) |

**4060 gate (per model, before any reading):**
- CHUNK(1), defined in §2, on §170 test n 9–12 must reproduce the committed 3070 CHAIN-LABEL `precision_strict` within ±0.01 per n.
- For R2-s1, FT-SELF and R1-s2, whose committed chain rows are on fresh items (eval_chainlabel_fresh / eval_selfdepth), the gate uses those committed fresh rows at n 13–16 on the same fresh items.
- Failure ⇒ UNREADABLE on this card (rule b).

**Items** (gold values are used for scoring only and never reach the model):
- **IN:** §170 test n 1–16, the first 500 per n in test order (N.test_set(1000, 16), seed 17100). n 1–8 feed only the first-chunk table.
- **DEEP:** eval_deep's items n 17–24, the first 500 per n (N.test_set(1000, n_max=24)). This is the same item set §216's eval_deep read.
- **FAR (new):** nested170.generate(rng, n) with rng seed 22080, 500 per n at **n ∈ {25, 28, 32, 36, 40}**. Excluded by exact rendered match, redrawn on collision, collision rate reported:
  - the §170 test set n 1–16;
  - test_set(1000, n_max=24) n 17–24;
  - **the sealed 216-5 set** (test_set(1000, n_max=30) n 25–30, seed 17100);
  - the s215 / s216 pools' recorded item signatures where they are still on disk (else the generator-collision bound is stated).
  - Rendered length at n 40 ≈ 119–129 tokens (vs ≈ 81 at n 24; the GPT-2 wpe holds 1,024).
- **The sealed n 25–30 set is never read.** FAR items come from a different rng and exclude it.

## 2. Procedure CHUNK(k)

For an item with expression cur (operator count ops(cur), visible from the text):
1. k_eff = min(k, ops(cur)).
2. Roll out the frozen model for k_eff passes on batch_x(render(cur)), batched by rendered length as long_step / part_chainlabel do (no padding effects).
3. Decode s = VB.rerender(argmax R(V_{k_eff})), with VOIDs dropped, as part_chainlabel does.
4. If s does not parse into one valid expression (_parse_tree), or ops(parse(s)) ≠ ops(cur) − k_eff, the chain FAILs. The second check is a format check that uses only the text.
5. cur := parse(s). Stop when cur is a leaf; the label is its value. Cap: ⌈n / k⌉ + 4 chunks, else FAIL.

**Recorded per item:**
- label, number of chunks, per-chunk exactness against the true trajectory: chunk j exact iff s == render(reduce_k(true_j, k_eff)), where true_j is the true state the chain should be at;
- chain_exact (every chunk exact);
- written checkpoint tokens Σ_j len(s_j);
- latent passes Σ k_eff.

**k grid:** k ∈ {1, 2, 4, 8, 12, 16, 20} for every model. Every model is LENGTH-FREE to K′ 24, so k up to 20 is in its stable range. k = ∞ is the model's committed acc at K′ = n + 2 (IN / DEEP) plus a fresh K′ = n + 2 run on FAR, as the latent-only reference.

**First-chunk table e(m, k):** the chunk-exact rate of chunk 1 on items with ops = m, for m ∈ {1..24, 25, 28, 32, 36, 40} and every k. These are the first k reductions of an ordinary rollout on a size-m input. They come from the same runs, since chunk 1 of an item of size m *is* that measurement.

**EDIT arm** (descriptive, C7 edit test; R2-s1 and D3-T s1, k = 4, DEEP n 17–20, 500 items per n):
- At the first checkpoint, if chunk 1 is exact and the expression has ≥ 2 leaves, replace one uniformly chosen leaf value v by v′ ≠ v (rng 22081).
- Continue CHUNK(4) from the edited text.
- **EDIT-FOLLOW** = the label equals the value of the edited expression; **EDIT-OWN** = the label equals the original value.

## 3. Derivation [tags: PROVED / DERIVED / ASSUMED / MEASURED]

**D1 (Markov restart) [PROVED for the procedure].** After a checkpoint the model's next chunk is a function of render(cur) only: the bank is dropped and rollout starts from V_0 of the new input. So conditional on the chain being exact up to chunk j, chunk j + 1's exactness has the law of a first chunk on the true reduced item reduce_k(item, j·k).

**D2 (product prediction) [DERIVED; the assumption is A1].**
- **A1 (sub-problem exchangeability) [ASSUMED]:** the true reduced item at a checkpoint is distributed, for the model's purposes, like a fresh item with the same ops count. Its first-chunk exactness is then e(m, k) with m = its ops count.
- Then P(chain exact | n, k) = ∏_{j ≥ 0} e(n − j·k, min(k, n − j·k)).
- A1 is not exact. Re-rooted items are not generator draws: their tree shapes are the generator's trees with innermost reductions applied.
- **MARKOV-PRODUCT** checks A1 empirically: |measured chain_exact − product| ≤ 0.02 at DEEP n.
- The k = 1 analogue held within ±0.005 through n 15 on D3-T (213-35, MEASURED).

**D3 (what H_pass and H_size predict) [DERIVED from the definitions].**
- **H_pass(k*):** e(m, k) ≥ 1 − δ for all m (to 40) whenever k ≤ k*, with δ ≤ 0.01.
  - By D2, chain_exact(n, k) ≥ (1 − δ)^⌈n/k⌉. For k = 8 at n 40 that is ≥ 0.95 at δ = 0.01.
  - So **accuracy is flat in n**, and the best k is the largest k ≤ k*: fewer checkpoints, fewer decode errors.
- **H_size:** e(m, k) = ∏_{i=0}^{k−1} s(m − i), with per-pass survival s(m) decreasing in m past support.
  - Then chain_exact(n, k) = ∏_{m=1}^{n} s(m) for every k, the same product over the same visited sizes, **independent of k up to decode error**. Decode error adds a factor per checkpoint, so k = 1 is worst only through decode error.
  - Correction to the intuition in §0: under pure H_size, k does not matter to first order. The registered discriminator is the dependence of e(m, k) on m at fixed k (SIZE-FREE-CHUNK), not which k wins.
- **H_mixed (pass-and-size):** survival s(m, i) depends on both the size and the pass index i within a chunk. The k* / m surface e(m, k) is reported in full.

**D4 (latent share) [arithmetic].** At chunk length k, (k_eff − 1) / k_eff of each chunk's reductions are silent. Written checkpoints ≈ ⌈n / k⌉ versus text CoT's n step lines. Token counts are measured, never assumed.

**D5 (evidence for the priors) [MEASURED].**
- R2-s1 full-trajectory acc: n 17–20 = 0.998 / 0.979 / 0.862 / 0.659, n 21–24 = 0.364 / 0.206 / 0.142 / 0.121. STOP-LOC 21, NOT-SIZE-KEYED-2 (T ≤ 61 vs > 61: 0.352 vs 0.367 at n 21).
- k = 1 chain on R2-s1 fresh items: keep n 17–24 = 0.993 … 0.890 with prec_kept ≥ 0.9956.
- On D3-T, the per-step exact rate on size m (long_step) and the 213-35 product predict precision 0.913 at n 16. The single step stays near 0.99 per step to m ≈ 16 on a support-8 model.
- So per-pass survival at k = 1 is size-robust to ~2× support. Whether a k-pass rollout is size-robust is exactly what is unmeasured.

## 4. Readings (per model; then across models)

| reading | definition | bar |
|---|---|---|
| **CHUNK-EXTRAPOLATES(model, k)** | strict chain precision (label = y, chunks = ⌈n / k⌉, no FAIL) | **≥ 0.90 at every DEEP n 17–24 and ≥ 0.80 at every FAR n** |
| **CHUNK-C2 (primary)** | CHUNK-EXTRAPOLATES(R2-s1, k) for k = 8 **or** k = 4 (two pre-stated k; ≥ 75 % of reductions silent) | — |
| CHUNK-C2-BASE (secondary) | CHUNK-EXTRAPOLATES(D3-T s1, 4) | — |
| **SIZE-FREE-CHUNK(model, k)** | min over m ∈ {25..40} of e(m, k) ≥ mean over m ∈ {4..support} of e(m, k) − 0.02 | mechanism: H_pass at k |
| **k\*(model)** (descriptive) | the largest k with SIZE-FREE-CHUNK and e(m, k) ≥ 0.98 at every m ≤ 40 | k* versus support across 8 / 12 / 16 (H_pass predicts growth) |
| MARKOV-PRODUCT | \|chain_exact − D2 product\| ≤ 0.02 at DEEP n | the restart property |
| **CHUNK-BEATS-LATENT** | CHUNK(k) precision − latent acc at K′ n + 2, per n | descriptive, the margin at n 21–24 |
| EDIT-FOLLOW / OWN | §2 EDIT arm | descriptive (C7 edit; ≥ 0.9 follow would be the first behavioural edit row on VB) |
| TOKEN-COST | written checkpoint tokens per item vs text CoT's measured tokens (s170_cot_s1, §219 base re-eval where present) | descriptive, never "compute-matched" |
| s2 replication | CHUNK-EXTRAPOLATES(D3-T s2, 4) and SIZE-FREE-CHUNK on D3-T s2 | descriptive |

**Wording.**
- CHUNK-C2: *"with a discrete re-rooted checkpoint every k latent passes, R2-s1 solves nested depth n 17–40 at ≥ 0.90 / 0.80, past every trained depth (16) and past its latent-only boundary (n50 20.5). Hybrid latent–discrete computation with (k−1)/k of reductions silent; one model, fixed k; not latent-only C2, not C5 (k is not learned)."*
- SIZE-FREE-CHUNK true: *"the latent limit is a pass-horizon, not the input size"*. False: *"per-pass accuracy is size-keyed"*.
- Never "the latent model extrapolates" without "with checkpoints".

## 5. OC (500 items per n; priors ASSUMED, to be shaded by the red team)

- **Sampling.** At p = 0.95 per n, SE ≈ 0.010 on 500 items. The bar 0.90 is 5 SE below. At FAR p = 0.90, SE 0.013, so the bar 0.80 is 7.4 SE below. The per-n "every" conjunction over 13 n:
  - under a true flat 0.95 / 0.90, P(miss any) ≤ 13 × P(Z < −5) ≈ 0;
  - under true 0.92 at every DEEP n, P(pass DEEP part) ≈ 0.98^8 ≈ 0.85.
- **Priors:** P(H_pass at k = 8 for R2-s1) ≈ 0.40; P(H_pass at k = 4) ≈ 0.55. H_size ≈ 0.35 and H_mixed the rest.
- **P(CHUNK-C2 | no effect)**, where no effect = H_size with per-pass survival as in D5:
  - under H_size the k = 1 chain is the best case, and its keep is 0.89 at n 24, so precision ≈ 0.89 < 0.90 at n 24;
  - FAR n 25–40 then compounds further, roughly 0.89 × ∏ s(25..40);
  - **P(pass | H_size) ≈ 0.03** (only if s(m) stays ≥ 0.997 past n 24, which contradicts H_size's definition).
- **P(CHUNK-C2 | effect = H_pass at k ≤ 8) ≈ 0.75.** The residual risk is FAR rendered lengths ≈ 1.6× n-24's (positional OOD) and decode errors over ⌈40/8⌉ = 5 checkpoints.
- **P(CHUNK-C2) ≈ 0.55 × 0.75 + 0.45 × 0.03 ≈ 0.43.**
- **SIZE-FREE-CHUNK** at k = 4 on R2-s1: P(true | H_pass) ≈ 0.95, P(true | H_size) ≈ 0.02. That is a decisive mechanism reading either way, given D1.
- **A null is informative:** H_size or H_mixed with a measured e(m, k) surface tells the next design which axis to fix (size-invariant read vs pass horizon).

## 6. Censoring (rule b) and pending

- **UNREADABLE (fails every reading of the model):** a model failing its 4060 gate; item-set sha mismatch; card / dtype / eval_bs mismatch.
- FAIL chains (unparseable, wrong ops count, cap) count wrong. Never-finishing chains count wrong.
- **CTX:** any item whose input exceeds the model's slot capacity counts wrong, reported. None is expected below 1,024 positions.
- An n with < 450 completed items (crash) reads pending; at --final it fails.
- First-chunk cells with < 100 items (m > 24 only from FAR n, m < 9 from IN n 1–8) are excluded from SIZE-FREE-CHUNK and listed.
- A missing file reads pending (the 215-7 rule).

## 7. Gates ($0)

- G-C1: CHUNK(1) reproduces part_chainlabel's per-item labels **bitwise** on the 3070 for D3-T n 9–12 (same code path except k). Then the 4060 gate (§1).
- G-C2: a known-answer fixture. A stub model whose R-decode is the true reduce_k gives chain precision 1.000 and e = 1.000 at every k, n.
- G-C3: smoke at full scale. One full FAR n 40 batch at k 20, peak memory recorded, before the queue.
- G-C4: score_s220 (or the § number assigned) written and fixture-checked before any output: pending rule, every reading, censoring.
- G-C5: an Audit diff review of the part (boxes/vbank_diag.py part_chunksnap or a new module) and the item builder (exclusion, seal check).

## 8. Cost

- **$0.** Local RTX 4060 (the 3070 for G-C1).
- Estimate per model: IN + DEEP + FAR ≈ 13,000 chains × 7 k values. Chain cost ∝ ⌈n/k⌉ rollouts of ≤ k passes, the dominant term at k = 1. That is roughly 2–5 h per model at bs 16 (CHAIN-LABEL-FRESH timing to be measured in G-C3).
- **Order:** R2-s1 → D3-T s1 → FT-SELF → D3-T s2 → R1-s2.
- The k = 1 rows partly duplicate committed chains, kept for same-card comparability.

## 9. Open questions for the red team / Audit

1. Is k ∈ {4, 8} the right primary pair, or should CHUNK-C2 be "any k ≥ 4" with a multiplicity correction?
2. Is the FAR exclusion of the sealed 216-5 set enough, or should FAR stop at n 24 + {28, 32, 36, 40} (skip n 25)?
3. Does the EDIT arm need an input-only control (Audit C7 rule c)? The checkpoint is produced by the model's own latents, but a verbalizer that is right from the input alone is not excluded by EDIT-FOLLOW.
4. H_mixed: is the SIZE-FREE-CHUNK bar (−0.02 vs in-support mean) the right discriminator, or should it be a slope test of e(m, k) on m with a CI?
