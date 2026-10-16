# D3: a length-tolerant pass input for VERSION-BANK (DRAFT, 2026-10-16)

**Status.** This is an Opus inventor + derivation subagent's draft for Theory. It is not a registration: the draft costs $0, and any spend needs the user's OK via Audit. The repo was read only.
- **Sources:** notes 212-4…212-8, as relayed by the coordinator.
- **My $0 CPU scripts** (3 threads, idle priority, outputs only) are in `scratchpad/research2/d3_scripts/`: `d3_split.py`, `d3_ready.py`, `d3_wpe_drift.py`, `d3_wpe_struct.py`, `d3_gather_probe.py`, `d3_slot_errors.py`, each with its JSON.
- **No intervention test was run.** Every number of mine is descriptive.
- **Tags:** [PROVED], [PROVED UNDER X], [DERIVED], [MEASURED], [ASSUMED], [CONJECTURE].

## 0. Summary

1. **The cliff is specific to PURE-ANNEAL (PA) training** [MEASURED, 212-6 to 212-8 plus mine].
   - N2 and N3 collapse at T 29–31. CARRY (N1) does not: it is ≥ 0.966 to m 12, and longer T does not hurt it.
   - GATHER-SPLICE reads NOT-GATHER in both PA cells.
   - In N2, the reduction itself is exact at every length: changed slots are 1.000 even at T ≥ 39. **What fails is copying late slots.** The first wrong slot has median 25–27 at every T ≥ 29.
2. **Mechanism [MEASURED signature + DERIVED incentive; causal link CONJECTURE].** PA's content-free queries W_q·wpe[j] give the position table a direct gradient path, and the gather needs adjacent slots to be distinguishable.
   - PA training sharpened the trained rows of the shared wpe table about 7× more than CARRY did. Adjacent-row 1 − cos is 0.014–0.058 (N2), 0.005–0.008 (N1) and 0.001 (pretrained).
   - Rows ≥ 33 never moved beyond weight decay (2.37–2.38% in all three cells, exactly AdamW's decay factor). That leaves a seam at the trained edge, in the input of every block.
   - N2's in-range gather reads its own slot plus right neighbours. At j ≥ 33 it puts 0.94 of its mass far away and 0.00 on its own slot.
   - GATHER-SPLICE's J0 = 29 lies right of the median first error (26), so its null is consistent with this account. It does not exclude a gather contribution.
3. **D3 (primary): SLVA + FROZEN-WPE.**
   - **SLOT-LOCAL VERSION ATTENTION (SLVA):** the pass input of slot j is attention over *every stored version of slot j*. It is pass-global and append-only, with no cross-slot mixing.
   - **FROZEN-WPE:** the position table stays exactly pretrained.
   - The core and coda keep reading the whole bank (all slots, all versions).
   - At pass 1, SLVA *is* CARRY [PROVED]. So the fresh long step is computed by the architecture that N1 showed to be length-tolerant.
4. **Second limit:** N1's trajectory drift (RS HISTORY-KEYED, +0.83). A pre-stated $0 selector (COPY-REFRESH on N1) chooses between two fixes:
   - **COPY-ANCHOR:** unchanged slots are stored as their V_0 vector;
   - G1's compaction + restamp (the layout fix).
5. **Dropped, with reasons:** position randomisation, banding, structural / relative position biases and log-n scaling. Each is either refuted by the data or not needed for the measured failure. They stay as triggered secondary levers.
6. **Plan:**
   - Stage 0 ($0);
   - FT-SLVA from N1's model.pt ($0.3–0.6);
   - one 48k D3 cell ($0.9–1.9).
   - Proposed family cap: $4.

---

## 1. Diagnosis

### 1.1 Hypotheses and evidence

| hypothesis | prediction | evidence | verdict |
|---|---|---|---|
| **H-COUNT** (operator count, or the (a, t) support) | flat in T at fixed m; falls with m at fixed T | N2 at T 27: m 7–10 = 1.00 / 1.00 / 1.00 / 0.99. N3 has a small count effect (T 27, m 10: 0.72). N1 decays gracefully in m (logit −1.32 per operator at fixed T, m ≥ 10; T +0.15 per token; `d3_ready.py`) | **not the PA cliff**; a secondary, N1-type decay at m 14–16 |
| **H-LEN-ALL** (the whole network is position-OOD past the trained edge; PROOFS2 row 4) | every cell has the cliff | N1 is 1.00 at m 10 for T 27–37, and 0.80 at m 16, T 55 | **refuted for the recipe**: derived absolute positions do not by themselves break the step |
| **H-DILUTION** (softmax dispersion) | a smooth fall in T, no knee; the global step logic degrades too | N2 changed slots are 1.000 at T ≥ 39. At m 9 the fall 0.96 → 0.06 between T 29 and 33 needs a ≥ 20× error increase, while Lemma 2's first-order bound allows ≤ 1.14× (keys +14%) | **refuted for the cliff** [DERIVED] |
| **H-GATHER-ADDR** (only the gather's absolute queries) | GATHER-SPLICE or mod31 lifts the cliff | GS +0.04 (N2), −0.17 (N3); mod31 ±0 | **not sufficient**, although the gather mis-addresses at j ≥ 33 (§1.2 c) |
| **H-PA-BLOCKS** (PA training made the shared computation position-specific within the trained rows) | PA cells only; copy failures at late slots; spreading left from the edge | all of §1.2 | **adopted** [CONJECTURE on the causal link; signature MEASURED] |
| **Offset sensitivity** | — | any POS_OFFSET (8 or 16, with or without the kept sink) gives ≈ 0 at m 1 in all three cells | the trained function is tied to the absolute layout from EOT. **Randomised positions (Ruoss / Abacus) fail their precondition** (as in 187-5 and §177) |

### 1.2 New $0 measurements (descriptive; nothing intervened)

**(a) Training length law** [MEASURED, `d3_split.py`, 200k draws, n ~ U{1..8}].
- T max is 33. P(T ≥ 28) = 2.1%, P(T ≥ 30) = 0.26%, P(T ≥ 32) = 0.0095%.
- Slot j exists only when T > j. So slots 28 / 29–30 / 31–32 / ≥ 33 appear in 2.1% / 0.26% / 0.01% / 0% of items.
- **Where training thins out, at slots 28–32, is where the PA cliff sits.**

**(b) The position table** [MEASURED, `d3_wpe_drift.py` / `d3_wpe_struct.py`].

| | relative drift of rows 1–28 | rows ≥ 33 | adjacent-row cos, rows 20–32 |
|---|---|---|---|
| pretrained | — | — | 0.999 |
| N1 CARRY | 0.07–0.16 | 0.0238 | 0.993–0.995 |
| N2 PA | 0.12–0.43 | 0.0238 | 0.94–0.97 |
| N3 PA + MCT | 0.12–0.31 | 0.0238 | 0.95–0.975 |

- 1 − ∏(1 − η_tλ) = 0.0237 for this schedule (computed from fsq120.lr_at): untouched rows are changed only by AdamW's decoupled decay (Prop 2).
- N2's drift is position-specific, not a common shift: the mean direction explains 13% and the top PCs 33 / 23 / 14%. A "drift-extrapolation" repair is therefore ill-defined.
- The N2 gather queries W_q·wpe[j] have adjacent cosine 0.80–0.84 in range and 0.999 at j ≥ 33. **Every query beyond the edge is the same vector.**

**(c) N2's pass-1 gather** [MEASURED, `d3_gather_probe.py`, 1,800 fresh items, m 6–14, head-mean].

| query slot j | self | +1 | +2 | far (\|d\| > 4) |
|---|---|---|---|---|
| ≤ 20 | 0.35–0.82 | 0.15–0.32 | 0.04–0.17 | ≤ 0.03 |
| 25–32 | 0.05–0.28 | — | — | 0.19–0.48 |
| ≥ 33 | ≤ 0.002 | — | — | 0.94–0.95 |

- **At passes 2–4** (in range, t = 4) the version mass by age is 0.58 / 0.16 / 0.05, with 0.21 on V_0. About 0.44 of the mass is on the same slot.
- **A translation-invariant symbolic fit** (class × class + clipped offset + sink and " =" anchors) leaves KL 0.31 nats (held-out 0.31), with in-range overlap 0.64–0.84.
  - So N2's gather is *not* a class + offset rule.
  - A relative-query gather would therefore not be a faithful drop-in for N2.

**(d) Where fresh steps fail** [MEASURED, `d3_slot_errors.py`; 150 items per m, m 6–14].
- **N2:**
  - Changed slots (the reduction) are 1.000 in every T bucket.
  - Copy slots are 1.000 at T ≤ 27, 0.957 at T 31, 0.798 at T 33–37 and 0.652 at T ≥ 39.
  - Slots 20–28 are 1.000 in T ≤ 27 items but 0.70–0.76 in T ≥ 33 items.
  - The median first wrong slot is 25 / 27 / 26 / 26 at T 29 / 31 / 33–37 / ≥ 39.
  - Errors lie right of the handle (offset ≥ 6: 0.56–0.73). Within ±5 of the handle, accuracy is ≥ 0.99.
- **N1:**
  - Changed slots are 1.000 and copy slots ≥ 0.998.
  - Step exact is 0.964 at T 33–37 and 0.942 at T ≥ 39.
  - The few errors sit late (median first wrong slot 31–33). Per-slot copy error is 0.2–0.3% at j 29–32 and 0.5–0.7% at j ≥ 33.

**Reading [DERIVED from (a)–(d)].**
- In the PA cells the step's *logic* is length-free.
- The *copy channel* depends on position-specific features of the right neighbours. Those features exist only for well-trained rows (j ≲ 24), and corruption spreads about 3 slots left of the untrained region.
- GS replaced only slots j ≥ 29. It left slots 24–28 reading right neighbours, and it fed the PA core a carry input it never saw. So GS's null cannot separate "blocks" from "gather". The fix below removes both.

### 1.3 Why PA training destroys length tolerance (the (b) question)

**Proposition 3** (§3) gives the incentive: a content-free gather can tell neighbours apart only through the table, so its gradient sharpens the rows it is trained on.
- The prelude reads the same table. So the blocks are trained on a sharpened code at rows ≤ 32 and meet the pretrained, blurred code at rows ≥ 33: a seam in every block's input.
- The gather's neighbour mixing then makes each slot's copy depend on its neighbours' (seam-side) features.
- CARRY has neither pressure. Its table moved about 7× less, and its copies stay slot-local.

**What is measured and what is conjecture:**
- Measured: the 7× sharpening, the seam (exact decay-only rows), the mis-addressing and the error localisation.
- Conjecture: that the seam plus neighbour mixing *cause* the cliff.
- **No clean $0 test of the seam exists.** The drift is not extrapolable, and an all-pretrained table breaks the in-range function. The FT in §6 is the test.

### 1.4 The second limit: N1's trajectory drift

- N1's RS reads HISTORY-KEYED (+0.829, control ≈ 0, note 212-7): its fresh step works, but the intact continuation fails.
- **RS cannot separate the two candidate causes:**
  - **representation drift** (copy slots re-processed by 4 blocks every pass);
  - **layout** (the intact state keeps raw VOID gaps; the restart is compact).
- So the fix is chosen by a pre-stated $0 selector, COPY-REFRESH (§6, Stage 0).

### 1.5 Decision rule: split outcome → D3 form (stated as applied, plus the pending rules)

| outcome | D3 form |
|---|---|
| count-keyed in the base (flat in T at fixed m; falls with m at fixed in-range T) | CORE-BAND + EXIST-SCALE; TRUNC-LONG as fallback |
| length-keyed in **all** cells, CARRY included | structural / relative positions in the prelude + core (STRUCT-REL, banding) |
| **length-keyed in PA cells only, CARRY tolerant** (observed) | remove what PA adds → next row |
| … with GATHER-SPLICE = GATHER-BOTTLENECK | relative-query gather alone |
| … with **NOT-GATHER** (observed ×2) | **SLVA + FROZEN-WPE** (blocks trained in the CARRY regime) |
| offset readable and POSITION-KEYED | offset / position randomisation in training |
| offset UNREADABLE (observed) | randomisation excluded (precondition failed) |

**Pending rules (pre-stated now):**
- **COPY-REFRESH on N1** decides the second-limit fix (§6).
- **FT-SLVA** decides whether the 48k cell runs (§6).
- **Secondary triggers (§2.4).**

---

## 2. Mechanism

### 2.1 SLOT-LOCAL VERSION ATTENTION (replaces the PA gather)

At pass t, for slot j and head h (12 heads × 64-d slices of N(V)):

U_t[j]_h = Σ_{s<t} α^h_{j,s} · N(V_s)[j]_h

α^h_{j,s} = softmax_s( ⟨W^h_q LN_q(N(V_{t−1})[j]), W^h_k LN_k(V_s[j])⟩/8 + β^h_{0}·1[s = 0] + γ^h·(t−1−s) − c(step)·1[s ≠ t−1] )

- **Queries are content queries from the slot's newest version; keys come from version s *at the same slot*.** There is no position query and no wpe input.
- **Initialisation:**
  - W_q, W_k and LN come from block 6 (as N2's gather);
  - γ uses the ALiBi slopes on all 12 heads (as N2's γ_q);
  - β⁰ = 0.
- **c(step)** is the gentle anneal (1e4 to 1k, then 20 → 0 over 1k–14k; 211-11 / VB-2G §2 item 5).
- **N** is N2's per-slot RMS reset (pin 5).
- **Unchanged from N2:** the prelude (blocks 0–5, V_0); the core (6–9) and coda (10–11) with their bank reads over [all versions × all slots | current, causal], the β / γ biases and the per-head calibration; R; R1 / R2 / R4. **The pass-global bank is untouched.**
- **Cost:** a softmax over ≤ t entries per slot, cheaper than PA's t·T.

### 2.2 FROZEN-WPE

- wpe.requires_grad = False from step 0 (a checksum is asserted at every checkpoint).
- Every position row is then the pretrained one, at every T.
- The EOT-anchored absolute layout that the offset test shows to be load-bearing stays exactly as GPT-2 has it.
- Nothing is shifted, randomised or restamped at pass 1.

### 2.3 Second-limit add-on (chosen by COPY-REFRESH)

- **COPY-ANCHOR (if the selector reads DRIFT).** The stored entry is Ṽ_t[j] = V_0[j] when ĉ_t(j) = ĉ_0(j), and V_t[j] otherwise.
  - ĉ_0(j) is the input token's class; ĉ_t(j) = argmax R(V_t[j]), the model's own decode in training and in eval (no teacher, so no exposure gap).
  - Ṽ_t feeds the bank K/V and SLVA. L_rec still reads the computed V_t.
  - It costs nothing extra: R is already evaluated per version for L_rec.
- **LAYOUT (if the selector reads LAYOUT):** G1's registered compaction-as-key-masks plus restamp at storage (VB-2G §2 items 1–3, with their PROOFS2 fixes).
- **BOTH** in the middle band.

### 2.4 What D3 deliberately does not include

Each lever is listed with the trigger that would add it later.
- **Position randomisation / offsets.** Precondition failed: any offset kills the trained function in all three cells (212-6, 187-5). *Never.*
- **CORE-BAND** (an annealed band bias −c_b·1[|i−j| > 4], sink exempt, on 8 / 12 core heads) and **EXIST-SCALE** (κ(T) = max(1, ln T / ln 31) on the global heads, the identity on every trained length, so an eval switch).
  - *Trigger:* D3 reads LENGTH-FREE but fails STEP-GENERAL with errors at the handle, or with late-slot copy error > 0.2% (Thm 4 c).
- **STRUCT-REL / Ito-style bracket-depth bias.** Lemma 0 needs no depth feature for leftmost-ready reduction with minimal brackets: radius 4 plus one prefix-OR suffice [PROVED, PROOFS2 row 1]. *Trigger:* none from current data.
- **Relative-query cross-slot gather (REL-GATHER).**
  - *Trigger:* D3 reads LENGTH-FREE ∧ HISTORY-IGNORED, and the user wants cross-slot gather use.
  - Its own check is LENGTH-FREE, since cross-slot mixing is what §1.3 implicates.
- **TRUNC-LONG** (training on the first n − 8 reductions of n 9–16 items).
  - It is not needed: N1 shows that length tolerance needs no long inputs in training.
  - If ever used, the C2 claim must read: "serial depth beyond the trained chain length, with input sizes up to 16 seen for their first n − 8 reductions". It is then not a length-extrapolation claim.

### 2.5 Constraint check (pass-global attention memory)

- Every version stays stored and readable. The pass input is an attention read over **all** of a slot's versions, and every core or coda layer attends over all slots of all versions.
- Nothing is overwritten or decayed by rule.
- CARRY is the degenerate attention pattern (one-hot on the newest). It is reported as such (CARRY-LIKENESS, HISTORY-USE), never imposed.
- COPY-ANCHOR stores the V_0 vector in new versions. That is append-only.

---

## 3. Mathematics

**Theorem 1 (architecture of SLVA) [PROVED].**
- (a) At t = 1 the bank holds only V_0, so α = 1 and U_1 = N(V_0) for every c and every parameter value. **A fresh single step in a SLVA model is computed by exactly the CARRY architecture**, so LONG-STEP measures only what training did to the shared weights.
- (b) For c ≥ 1e4 in fp32 / fp64, α is one-hot on s = t−1, so the whole rollout equals CARRY. With the bank masked, K = 1 is GPT-2: **G1 holds bitwise**, by the gate_carry_bitwise argument.
- (c) **Slot-locality.** If V_s[j] = V′_s[j] for all s < t, then U_t[j] = U′_t[j]. The pass input adds no cross-slot dependence, so it adds no dependence on T or layout. The same function acts at every slot and every length.
- (d) **Boundedness:** ‖U_t[j]‖ ≤ max_s ‖N(V_s)[j]‖, independent of t.
- (e) **Dilution only in t.** If a version set S* has logit ≥ λ and the other versions ≤ λ − Δ, then U deviates from the S*-restricted read by ≤ 2Bε/(1+ε), with ε = (t − |S*|)e^{−Δ} (D Lemma 2). This bound does not depend on T.
- *Proof:* the definitions. (e) is D Lemma 2 with |F| ≤ t − 1. ∎

**Proposition 2 (seams in a fine-tuned table) [PROVED].**
- Under AdamW, a row that receives zero gradient on every step has m = v = 0, so its Adam update is 0. Only the decoupled decay acts: p ← p·∏_t(1 − η_tλ).
- With this schedule that factor is 0.9763 (computed from lr_at), which matches the measured drift of 0.0237–0.0238 at rows ≥ 33 in all three cells.
- Hence any fine-tune that moves the trained rows leaves the rows ≥ T_max in the pretrained family. The table has a seam whose size is the in-range drift.
- **Under FROZEN-WPE there is no seam:** every row is pretrained. ∎

**Proposition 3 (a content-free gather sharpens its rows) [DERIVED].**
- Let q_j = W_q p_j + b and k_i = W_k LN(V_0[i]) + b_k, where V_0[i] carries a position component φ(p_i).
- A slot-relative pattern (self or next) needs the logit gap g_j = q_j·(k_{j+δ} − k_j) to be ≥ Δ for every trained j.
- Write ‖p_{j+1} − p_j‖ = ρ‖p_j‖, with ρ ≈ 0.045 for the pretrained rows (cos 0.999). Then |g_j| ≤ ‖q_j‖·‖W_k‖·L_φ·ρ‖p‖ (L_φ is the Lipschitz constant of the position component through LN).
- Raising g_j needs either a larger ρ (sharpening the rows, which only trained j receive) or larger W gains on the difference subspace. Both are position-specific adaptations confined to trained rows.
- CARRY's loss contains no term that needs q_j ≠ q_{j′}.
- *Measured consistent:* 1 − cos 0.014–0.058 (PA) vs 0.005–0.008 (CARRY). This is an incentive argument, not an SGD theorem.

**Theorem 4 (conditional length-invariance of the fresh step; dilution horizon).**
- **Assumptions:**
  - **A5′** (factored locality, PROOFS2 row 9): the trained pass's next class at slot j is a function of radius-2 operator features, the stage-2 key (own class, flags at offsets −2..2, b_j) and the OP table.
  - **COV′:** measured 1.000 of states at n 9–16.
  - **P-FREE:** the map does not depend on p_j beyond the stage keys, i.e. it is translation-invariant in the copy and step channels.
- **(a) [PROVED UNDER A5′, COV′, P-FREE, hard global aggregation].** The fresh step is exact for every m. Proof: D Theorem 2 at t = 1, where A1 / A3′ / A4 are vacuous because V_0 is the fresh encoding and the bank is {V_0}.
- **(b) [PROVED UNDER first-order composition, PROOFS2 row 10].** With q softmax global heads at margin Δ, exactness holds to T* = γe^Δ/(2BqΛ_R).
- **(c) [DERIVED, independence approximation].** If P-FREE fails only by a per-slot copy error ε_ℓ at the n_ℓ slots beyond the edge, then LONG-STEP ≈ (1 − ε_ℓ)^{n_ℓ}.
  - At T 55 (n_ℓ ≈ 22), STEP-GENERAL ≥ 0.95 needs **ε_ℓ ≤ 0.23%**.
  - N1's measured 0.2–0.7% predicts ≈ 0.86–0.95. N1's measured m 15–16 (0.73 / 0.62) is worse than that, so N1 has an additional count-type error at m ≥ 15.
- **What D3 guarantees:** D3 removes the two measured violations of P-FREE that PA introduced, the cross-slot position-addressed input [Thm 1 c] and the table seam [Prop 2].
  - **P-FREE itself is [ASSUMED]** for trained GPT-2 blocks. Its support is N1 [MEASURED].
  - It is checked by LENGTH-FREE and by the slot-error localisation.

**Lemma 5 (COPY-ANCHOR) [PROVED UNDER exact decodes].**
- Every slot whose decoded class equals its input class has Ṽ_t[j] = V_0[j] at every t, so its stored and read representation is pass-invariant: **zero copy-slot drift.** Drift can enter only through value slots, during their lifetime (≤ height-many passes), and through stale context.
- **Stale context:** V_0[j] encodes the *original* prefix. Whether that hurts is exactly what COPY-REFRESH measures.
- With R errors, a mis-decoded slot is anchored or un-anchored wrongly. That counts as a step error (censoring).

---

## 4. Goodhart routes

| route | status | read by |
|---|---|---|
| CARRY collapse (SLVA one-hot on the newest; RNN in function) | legal (the Markov optimum), but against the user's intent. Reported, not imposed | CARRY-LIKENESS; HISTORY-USE (BANK-ABLATE) |
| Position specificity re-learned inside the blocks (the c_attn gains sharpen derived positions; Prop 3's second route) | **open**: no position query exists, so there is no direct incentive [DERIVED] | LENGTH-FREE; first-wrong-slot vs the edge; adjacent-slot cos of h_6 position codes (POS-SHARP, descriptive) |
| Cross-slot position-addressed copying through the core's bank reads | open; it is the same mechanism class as PA's, at lower pressure | as above |
| VOID-slot scratch / counters (consumed slots are not anchored) | open under ANCHOR alone; closed if LAYOUT's key masks are added | STALL / NO-CLOCK, PF k-axis, CLOCK-PROBE on VOID slots |
| Anchor gaming: keeping a copy slot "unchanged" to get V_0 | the decode is R's, trained by L_rec on E_t, so a mis-decode costs L_rec | R accuracy by kind; anchor-error rate |
| Recompute from V_0 via the anchor | V_0 is already globally readable (VB). The anchor gives only *unchanged* symbols, with no new information | VALUE-SWAP own share, RECOMPUTE-V0 |
| Pass-1 precompute, coda solving, sabotage, timed release | as in VB (bounded by L1 / CAP-aff; no negative likelihood) | PROBE(1) / (2), ONESHOT, VALUE-SWAP |
| Frozen-table compensation (prelude layers re-encode positions sharply) | open | POS-SHARP; LENGTH-FREE |

---

## 5. Decisive readings (comparator-free, per cell), OC, censoring

All readings are local fp32 at eval_bs 32, on the training card's class, the §170 test set and the steppos grid (2,000 per m, seed 21600 + m).

| reading | definition |
|---|---|
| **(a) LENGTH-FREE** (limit i) | min over m 9–12 of the fresh exact ≥ 0.95, **and NO-KNEE**: for m 9–11, pooled exact at T ≥ 33 ≥ pooled exact at T ≤ 29 − 0.05 (cells ≥ 100 items) |
| (b) STEP-GENERAL (the original bar, a stretch) | fresh exact ≥ 0.95 at every m 12–16. Reported beside it: late-slot copy error ε_ℓ (Thm 4 c) |
| (c) SELF-SIMILAR (limit ii) | RS, n 9–12 pooled over readable cells, \|RESTART − INTACT\| ≤ 0.05, control within 0.03 |
| (d) DEPTH-GENERAL | mean over n 9–12 at K′ n + 2 ≥ 0.80 |
| (e) C1 kept | VB-C1 = DEEP ∧ PROBE-GAIN ∧ SERIAL-ONSET ∧ VALUE-SWAP SERIAL-STEP ∧ NO-CLOCK (§211 definitions) |

- **D3-C2 = (a) ∧ (d) ∧ (e).** (b) and (c) are reported as labelled parts.
- **Reported beside (constraint):** HISTORY-USE (USED / IGNORED / HURTS, 212-1) and the version-mass profile by age.

**OC (one seed per cell).**

| reading | P(pass \| no effect) | P(pass \| effect) |
|---|---|---|
| LENGTH-FREE | ≤ 0.02 (N2 / N3: m 10, T 33 at 0.01 / 0.00) | ≈ 0.8 (N1 passes: 1.0 / 0.998 / 0.991 / 0.966; no knee) |
| DEPTH-GENERAL | ≈ 0.05 (0 of 17+ nested models; best 0.688) | ≈ 0.6 (both limits fixed) |
| SELF-SIMILAR | ≈ 0.05 (N1 +0.83) | ≈ 0.6 |

- Per cell [ASSUMED]: P(entry) 0.65; P(DEEP) 0.55; P(LENGTH-FREE | DEEP) 0.75; P(DG | DEEP) 0.3; P(VB-C1) 0.35; **P(D3-C2) ≈ 0.12–0.15.**
- A positive is strong; a null is weak. A rate claim needs ≥ 3 seeds and the user's OK.

**Censoring (rule b).**
- A dead run (log < 48k, or < the FT length) or NO-ENTRY on true_answer_ce fails every reading.
- Unparseable decodes, keep / anchor errors and never-halting count as failures.
- Steppos (m, T) cells with < 100 items are excluded from NO-KNEE (as 212-2). NO-KNEE is UNREADABLE (not met) if any m 9–11 lacks a readable cell on either side of the edge.
- RS cells with < 100 items are UNREADABLE (not met).
- Event-only values are reported beside.

---

## 6. Experiments (nested first; RTX 3090 at $0.18–0.26/h; VB measured 0.28–0.48 s/step)

### Stage 0 ($0, local, eval-only; the interventions need Audit's OK)

1. **COPY-REFRESH on N1 (the selector).**
   - Implemented with the existing `rollout(edit=…)` hook, so vbank.py is unchanged: at every pass t, the stored entry and the carry for slots with argmax R(V_t[j]) = class(x_j) are set to V_0[j].
   - Reads: DEPTH n 9–12 at K′ n + 2, and RS cells, against intact.
   - **Control:** n ≤ 8 within −0.02 of intact, else UNREADABLE (then BOTH).
   - g = (refresh − intact) / (restart − intact) on N1's readable RS cells:
     - **DRIFT** if g ≥ 0.5;
     - **LAYOUT** if g < 0.2;
     - **BOTH** otherwise.
2. **Descriptive repeats** of `d3_slot_errors.py` and `d3_gather_probe.py` on N3 and on the G1 inputs. This is optional.
3. **Implementation gates** for a new `boxes/slva.py` (subclassing VBankN, with no edits to vbank.py or the plain paths):
   - G-D1: SLVA at c = 1e4 equals the CARRY rollout in fp64 (≤ 1e-9) at K 12; and at t = 1 it equals CARRY for every c (Thm 1 a).
   - G-D2: the wpe checksum is frozen, and FT runs start from the loaded table.
   - G-D3: the anchor identity in fp64 under exact decodes.
   - Per-term gradflow, with decorators checked by ast.
   - POS_OFFSET = 0 asserted.
   - A plain-path smoke, mem_check, G-TIME on a forced final-stage batch, and the G1b drift smoke.
   - Audit's diff review.

### Stage 1a (cheap; the premise test) — FT-SLVA from N1's model.pt

- **Why N1:** N1 is exactly SLVA at c = 1e4 (Thm 1 b), so step 0 of the FT is N1 bitwise (gate), and N1 is length-tolerant.
  - N2 is unsuitable as a start: its rows ≥ 33 cannot be re-sharpened, and its core rejects carry inputs (GS).
- **Setup:** 12k steps; LR 3e-5 constant; bs 64; n ≤ 8; R1 + R2; c 20 → 0 over 0.5k–6k; FROZEN-WPE (N1's table frozen).
- **Variant FT-SLVA-A** adds COPY-ANCHOR, only if the selector reads DRIFT or BOTH.
- **Cost:** (12k × 0.28–0.48 s + 0.5 h) × $0.18–0.26/h = **$0.26–0.55 each**.
- **Readings:**
  - **LENGTH-KEPT:** LENGTH-FREE, and fresh exact ≥ N1's value − 0.05 at every m 9–16;
  - HISTORY-USE; VB-C1 parts; RS (−A).
- **Rules (pre-stated):**
  - **LENGTH-KEPT ∧ VB-C1** → propose Stage 1b.
  - **¬LENGTH-KEPT** → SLVA's premise is falsified (version attention by itself re-creates length keying). Stop, and draft REL-GATHER / CORE-BAND with their own derivations.
  - LENGTH-KEPT ∧ HISTORY-IGNORED → Stage 1b still proposed. History use is reported, and it may need a full run.

### Stage 1b (one cell) — D3 = SLVA + FROZEN-WPE (+ the selector's add-on), from pretrained GPT-2

- **Setup:** RTX 3090, seed 1, 48k × bs 64, n 1..8, §170 test set, one GPU class, an off-host checkpoint mirror, true_answer_ce logged.
- **Recipe:** N2's otherwise (R1, reweighted R2, R4 curriculum), with the gentle anneal.
- **Cost:** (48k × 0.28–0.48 s + 1 h) × $0.18–0.26/h = **$0.85–1.95** (SLVA is cheaper than PA; G-TIME decides).
- **Proposed caps:** FT ≤ $0.7 each; cell ≤ $2.5; **family ≤ $4**. The realised $ replaces the projections, and nothing is dropped silently.

**Predictions [CONJECTURE].**

| | prediction |
|---|---|
| FT-SLVA | LENGTH-KEPT P 0.75; HISTORY-USED P 0.4; VB-C1 kept P 0.7 |
| D3 | as in the OC block. If LENGTH-FREE: fresh exact at m 13–16 ≈ N1's (0.94 → 0.62), so STEP-GENERAL P 0.2 without CORE-BAND |

---

## 7. Top risks and falsifiers

1. **The causal link of §1.3 is wrong.** Training under version attention could re-create length keying through the core's cross-slot bank reads.
   - *Falsifier:* FT-SLVA or D3 reads ¬LENGTH-FREE, with first wrong slots near the edge.
2. **SLVA learns CARRY (HISTORY-IGNORED).** The memory is then attention in form only.
   - *Not a falsifier of D3's C2 claim*, but it is reported to the user. REL-GATHER is the follow-up.
3. **Drift persists** (anchor stale context, or value-slot drift).
   - *Falsifier for the second-limit add-on:* SELF-SIMILAR fails while LENGTH-FREE holds.
4. **N1-type count decay at m 14–16.** STEP-GENERAL fails with LENGTH-FREE met.
   - That triggers CORE-BAND / EXIST-SCALE (§2.4). It is not a D3 failure.
5. **Optimisation.** FROZEN-WPE plus a new gather could delay entry.
   - *Falsifier at this budget:* NO-ENTRY while N1 / N2 entered at 11k / 16k.
6. **Seed lottery** (§199). One seed: a null is weak.

**Premise falsifier.** FT-SLVA keeps N1's table and the CARRY start but fails LENGTH-KEPT. "PA-specific mechanisms caused the cliff" is then wrong. The next step is structural or relative positions in the core (§1.5, row 2).

**Note for Audit.**
- GATHER-SPLICE's control (T ≤ 27) is a no-op at J0 = 29: no such slot exists in those items.
- Its J0 also lies right of N2's median first error (26).
- So NOT-GATHER does not exclude a gather contribution. D3 does not depend on that distinction.
