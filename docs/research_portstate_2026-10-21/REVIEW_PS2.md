# REVIEW_PS2: proof check and red team of PORT-STATE DRAFT v0 (2026-10-21)

Reviewer: Opus subagent (combined proof check and red team). Read-only; no GPU work. Two CPU scripts are in `review_scripts/`: `dag.py` (generator depth, pair exposure) and `oc.py` (OC of the primary bar).
Sources read: PS2_draft.md; THEORY Note 221-2, §221 result, §220, §218 (and LANG_v1 §2.7), §208 (and its result), §204 / §205 / §206 / §209 / §210 results, §175 (and its result), §170 / Note 170-3, §213, Note 213-35; boxes/codi_port.py, boxes/nested170.py (generate / render / cot_steps / aux_steps), boxes/vbank_diag.py (reduce_k / part_chainlabel); GCS_v1.md battery; NORTH_STAR.md C1 / C4 / C7 and Audit's C7 rules; the timing in boxes/runs_fsq/gsm_codi_{4,20}ep_s1/log.jsonl.

## Summary table

| # | Item | Grade | One-line fix |
|---|---|---|---|
| B1 | PORT-FOLLOW can pass with no portable computed state, via **precompute** (z_t already holds the donor's answer, or is several steps ahead) or **relay** (z_t holds the donor's question text and the continuation recomputes from it). STALL / ONSET are descriptive only. | **BLOCKER** | Add a decode-free co-primary, **BUDGET-FRONT**: the remaining budget b\* at which donor-final first appears must fall with t, within n (slope ≤ −0.4 and Spearman ≤ −0.6). Add **ANSWER-ONSET**: the final answer is at floor in z_t for t ≤ n − 2, read with a capacity-certified probe (§3). |
| B2 | Pair exposure is not specified. Donor-final, counterfactual and own must be distinct, and "counterfactual" is only defined for graph-matched pairs. In random-program pairs, donor-final = cf on **14.5 %** and all three are distinct on only **62 %** (dag.py). | **BLOCKER** | Use graph-matched skeletons, exact token length and the same names (primary), keep only pairs with all three outcomes distinct, and report the exposed share. NAME-DISJOINT pairs are descriptive (§2). |
| B3 | Depth "by construction" is false. (a) If each step reads a uniformly random earlier quantity, DAG depth is **3.5 at n 8**, with only 3.9 relevant steps. (b) A pure chain of constant-operand affine steps is a solvable-group word problem (TC⁰; §170's Literature flag 2). (c) G-W0 is set from a *short local* DIRECT run. Under-training underestimates one-pass ability, which biases the gate toward passing. | **BLOCKER** | Force a dependency spine and include var·var multiplication steps. Stratify by DAG depth d, not step count n. Fix n\* from a **matched-budget DIRECT cell** before any latent output (the 170-3 procedure) and set the band as in §4. |
| M1 | D1: "full CACHE / STATE-SWAP follows trivially" is true only for CACHE-SWAP, and only if the `<bot>` row is masked or swapped too. Full STATE-SWAP recomputes the latent KV rows in the recipient's context, so it is non-trivial (= PREFIX-SWAP(K)). | MAJOR | Correct D1. Full STATE-SWAP becomes an informative descriptive arm; full CACHE-SWAP is the sanity check. |
| M2 | M2 leak: under §221's QMASK, `<bot>` (a question-dependent row at all 12 layers) and the positions stay visible. The proposed leak test ("question KV removed → identical") is vacuous under a trained mask. | MAJOR | The answer phase sees latent rows, `<eot>` and the answer prompt only. Add a **MARKER-SWAP** control (swap only the BOS / `<bot>` rows → follow ≤ floor + 0.05). Use exact-length pairs. |
| M3 | D is underspecified ("z_t, or the latent rows' KV"). The KV rows contain copied question content. A shared 12-layer trunk can compute one or two steps itself (the §204 "decoder computes" route). R_t's CE is dominated by copied program tokens. | MAJOR | D reads **z_t only**, in a separate head of ≤ 2 layers. Add a **COPY-CONTROL**, which is Audit's C7 input-only rule: D trained on a question-only encoding must read R_t value tokens at floor for t ≥ 2. Score VALUE-EXACT(R_t) separately. Gate the reading on DECODABLE. |
| M4 | The prior P(follow ≥ 0.30) ≈ 0.3 is too high: §221 0.03; §175's SERIAL-NEST hybrid ≤ 0.25 even on synthetic data; §208 PASS1; §218 0.063. Training never puts the state and the question in conflict (D2's own argument). | MAJOR | Shade to about **0.10** (0.05–0.15). State that a null is likeliest and that the value is mechanism information (§7). |
| M5 | Novelty is undisclosed. M1 on a CODI / Coconut loop is SIM-CoT (2509.20317) with a state target. §175 already ran a question-blind aux decoder emitting the re-rooted expression. §211 / §213 gave portable VB states on synthetic data (VALUE-SWAP 1.000). §218's CONSUMPTION-FRONT is the mirror-image swap. | MAJOR | State the precise delta (§1), cite these, and run a Literature check before v1 (grep own bib first). |
| M6 | Is it "language"? Mod-11 semantics in words ("twice as many" wraps around) are incoherent, and with one template per op it is a token cipher of symbolic arithmetic. | MAJOR | Add paraphrase frames with held-out frames at test, numeric distractors and out-of-order statements. Use bounded natural integers or an explicit clock framing (§6). Word it as templated. |
| M7 | Cells and attribution. M2-only (§208 on a deep task) is missing. Under M2, a K0 removal is vacuous (the answer is constant by construction). There is no arm testing "the state wins" from the question side. | MAJOR | Cells: DIRECT, PS (M1 + M2), M1-only, M2-only. Read C1 removal through MEAN-Z / truncation, not K0. Add descriptive **QUESTION-SWAP(t)** (the §218 CF mirror). |
| M8 | Indexing and code. z_1 = prj(h_`<bot>`) is computed in the question pass, and the draft's R_t ↔ z_t mapping is unpinned. codi_port's `n_latent` is a per-batch scalar. The latent and decoder phases pass **no attention mask** (CODI's fix_attn_mask False), so M2 and per-item K need new code. | MAJOR | Pin z_j ≙ R_j. Define PREFIX-SWAP(t) exactly. Bucket batches by n. Implement QMASK with an explicit past mask. Run G-D, the plain-path smoke, per-term gradflow and G-TIME. |
| m1 | "λ is fixed by the derivation (§3)", but §3 has no λ derivation. | MINOR (formal FAIL) | Use λ = 1 on the per-token mean (the §175 / SIM-CoT value), stated as a choice. |
| m2 | The cost guess "≈ 2 s/step at bs 64?" is off. The port measured **2.11 s/step at effective bs 128** (§209) and 1.97 (§210), *with* the teacher passes, on GSM8K lengths. | MINOR | Use G-TIME. The projection is $1.5–4 per cell, 4 cells ≤ $16, so the ≤ $30 cap holds (§8). |
| m3 | Multiplying by a value of 0 mod 11 destroys downstream information (the state collapses). | MINOR | Keep multiplicands nonzero, or use naturals. |
| m4 | Censoring is not stated. | MINOR | A dead or diverged cell, an ARGS-DIFF or a failed gate fails every reading. A never-reached crossover or never-follow counts as a failure (PORT-FOLLOW). Report event-only values beside. One-seed wording. |
| m5 | C7 is untapped. D's decoded R_t is a language readout. | MINOR | Descriptive round trip: re-encode D's R_t as the question and check the answer matches. Apply Audit's rules (a)–(d). |
| m6 | "CoT-derived R_t on real language": GSM8K CoT does not contain re-rooted questions. | MINOR | Keep "never C4". A GSM8K stage would need §218-B1-style DECOMP programs (Audit ruled those not C4-legal as a whole). |

**Formal items (D1–D5 and §1–§2 claims).**

| Claim | Verdict |
|---|---|
| D1 full swap is trivial under M2 | **CORRECTED** (CACHE only, and only with `<bot>` handled; STATE-SWAP is not trivial) |
| D1 donor-final ⇒ steps t+1..K compute from the state | **CORRECTED** (also precompute or relay; needs B1's guards) |
| D1 cf ⇒ carriers; own ⇒ state ignored / recompute | PASS (given B2's exposure) |
| D2 M1 gives decodable, not used; nothing in the loss prefers state over question | PASS |
| D2 prediction P ≈ 0.3 | **CORRECTED** → about 0.10 |
| D3 one-pass near chance at n ≥ 3 | **FAIL** (§170 DIRECT n 3 = 0.635 / 0.750, n 4 = 0.430 / 0.533; near chance only from n ≈ 7) |
| D3 R_1's CE stops precompute | **CORRECTED** (M1 is compatible with precompute plus state; only capacity limits precompute) |
| D3 serial readings detect precompute | **CORRECTED** (they must be conjunctive and add a lead probe) |
| D4(i) z-shuffle control | PASS |
| D4(iii) QMASK leak test | **FAIL** (vacuous) |
| D4(iv) STATE-ONSET | PASS (descriptive; add LEAD(k)) |
| §1 R_t unique given a reduction order | PASS (state the tie-break for parallel branches) |
| §2 λ from the derivation | **FAIL** |
| §4 floor + 0.30 bar OC | PASS (computed, §7) |
| D5 cost | **CORRECTED** |

Totals: 7 PASS / 6 CORRECTED / 3 FAIL; 3 BLOCKER, 8 MAJOR, 6 MINOR.

**Recommendation: REDESIGN (v1), then go.** The question is worth a cell. Can a CODI-style latent hold a *portable* re-rooted state while the conflicting question stays visible? It is the right follow-up to 221-2, and it is cheap (≤ $16 projected). As drafted, though, the primary can be passed by precompute or relay, its pairs do not separate follow from counterfactual, and the generator's depth is not guaranteed. All three blockers have concrete $0 fixes.

---

## 1. Novelty and overlap

**Precedents the draft must cite:**
- **SIM-CoT (2509.20317):** a step-level auxiliary decoder that sees only the latent plus a "Step k" prompt, on Coconut / CODI GPT-2 GSM8K. M1 on a CODI loop is SIM-CoT with a different target.
- **§175 STEP-AUX (in-house):** a question-blind aux decoder (shared trunk) emitting e_{t+1}, the expression after t reductions (= R_t on symbolic nested), from M_t. Result: one reduction per pass and aux handle accuracy 0.997 → 0.47 (t 1 → 8). But SERIAL-NEST hybrid max 0.254 and not DEEP: a decodable per-step state was **not** portable even with the input re-shown.
- **§211 / §213 READY-RECODE on VERSION-BANK:** per-pass recode of the re-rooted expression; VALUE-SWAP (ii) 1.000 on synthetic. A portable state is already shown on synthetic data in a VB architecture (210-5).
- **§218 LANG-VB:**
  - It uses VB / SLVA on ProsQA-SYM with simulator REACH targets (not text), with a question-free coda.
  - Its decisive arms are PREMISE-SWAP × bank splice (a name-exchanged donor only) and **CONSUMPTION-FRONT** (input switched to a donor from pass t, own state kept). CONSUMPTION-FRONT is the exact mirror of PS2's PREFIX-SWAP (donor state, own question).
  - Stage B (GSM8K DECOMP) is gated on VB-C1-L.
- **§208:** M2 plus question-once on GSM8K gave PASS1-PRECOMPUTE.
- **§204 / §206:** per-pass targets met by pass-1 or decoder computation.
- **§205:** an interchange loss, which the draft correctly excludes.

**Precise delta.** PS2 is the first in-house design that:
1. trains a **CODI-architecture** loop (KV-cached continuous tokens; the question visible to every latent step; the GSM8K-capable family, so comparable with §221) with
2. a **text re-rooted-state target R_t** (program plus substituted values) read by a question-blind decoder from **exactly the object that is swapped** (z_t), plus
3. a trained answer bottleneck, and
4. reads **portability despite access**: a mid-chain swap from an arbitrary graph-matched donor, with the recipient's conflicting remaining program visible.

Neither §218's swaps nor §175's hybrid test (4) with arbitrary-program donors in a CODI loop. So it is **not a duplicate**, but it is incremental: M1 is SIM-CoT / §175 and the readings are §221 / §218's. Word it as "SIM-CoT-style state recode in a CODI loop, tested for portability", not as a new method.

**Sequencing.** §218 A1 has no output yet (§222 header). A §218 positive would narrow PS2's language-C1 delta to "architecture generality (CODI)"; a §218 null leaves PS2 as the CODI-family route. Either way PS2 runs independently, and its outcome table should name both cases.

## 2. D1 and the right portability test (B1, B2, M1, M2, M8)

**D1 corrected.**
- In codi_port, z_1 = prj(h at `<bot>`) and z_{j+1} = prj(h at latent j). Each latent position's K/V rows are computed from z_j **and attention to the question**.
- Under M2 (the answer phase sees only latent rows and markers):
  - **full CACHE-SWAP** (latent rows at all layers) makes the answer a function of donor rows, so follow is trivial, *provided* the `<bot>` row is masked from the answer phase or swapped too;
  - **full STATE-SWAP** (z_1..z_K) recomputes every latent row in the recipient's context, so it is **not** trivial. It equals PREFIX-SWAP(K) and is informative.
- §221's QMASK keeps BOS, `<bot>` and positions, so the `<bot>` leak is real unless M2 is defined more strictly.

**Is PREFIX-SWAP(t) with the question visible the right test?** Yes. Use the **state granularity**: donor z_1..z_t fed as inputs at positions 1..t, and the recipient computes z_{t+1}.. itself. The cache granularity hands over K/V rows that can hold copied donor question text, which is a relay. Pin:
- z_j ≙ R_j (j ≥ 1);
- PREFIX-SWAP(t) passes the donor's state R_t at position t;
- a LAST-ONLY variant (only z_t swapped; z_1..z_{t−1} the recipient's) as a descriptive Markov-state test.

**What makes donor-final follow pass trivially (B1):**
1. **Precompute:** z_t already encodes the donor's final answer (or R_{t+k} for large k), so the continuation just carries it. M1 does not forbid this; R_t can be co-encoded.
2. **Relay:** z_t encodes the donor's *original question*, and steps t+1..K recompute all n steps from it. This needs only a copy plus a model that does ≥ 2 steps per latent.
3. **Exposure failure (B2):** donor and recipient share the remaining program, or y_donor = y_cf by coincidence. Then follow cannot be told from carriers.

**Guards (B1), both decode-free or certified.**
- **BUDGET-FRONT (co-primary, decode-free, after §218's CONSUMPTION-FRONT):**
  - For each exposed pair and t ∈ {2..n−2}, run the continuation with remaining budget b ∈ {0..K−t} (K′ = t + b, then `<eot>`).
  - b\*(t) = the smallest b from which output = donor-final at every larger b.
  - Predictions:
    - serial state (s steps per latent): b\* ≈ (n − t)/s, which falls with t;
    - relay / recompute: b\* ≈ n/s, flat in t within n;
    - precompute: b\* ≈ 0, flat.
  - **Pass:** within-n OLS slope of b\* on t ≤ −0.4 ∧ Spearman(b\*, t) ≤ −0.6, pooled over the C1 band, on pairs that follow at full budget.
  - This deliberately admits chunked serial work (s = 2). That is still serial across latents.
- **ANSWER-ONSET:** with the DECODABLE-certified probe family (M3), the donor's final answer is read from z_t at ≤ floor + 0.05 for t ≤ n − 2. LEAD(k): a same-capacity D_k trained to emit R_{t+k} from z_t, reported for k = 1, 2. LEAD(2) ≈ LEAD(0) means z is ahead.
- **Censoring:** pairs that never follow count as PORT-FOLLOW failures. BUDGET-FRONT is computed on following pairs only, which is a stated selection. If fewer than 100 following pairs exist, the result is UNREADABLE.

**Pair construction (B2).**
- Graph-matched skeletons: same DAG shape and step order. This defines cf = the donor's computed values at the first t nodes, plus the recipient's ops and constants for steps t+1..n.
- Exact GPT-2 token length; same entity names (primary: the question offers a tempting re-read).
- Ops and constants differ both before and after t.
- Keep only pairs with y_own, y_donor and y_cf pairwise distinct (62 % of random pairs; dag.py). Report the exposed share per (n, t).
- NAME-DISJOINT pairs (the donor's names absent from the recipient's question) are descriptive: the easy-follow case.

**The guard's own exposure (control-must-be-exposed).** BUDGET-FRONT needs b to span below n − t, which in-distribution training (K ≥ n) never visits. That is intended: the front is where the answer stops being computable. Add a known-answer check at F3: a stub serial model (one hand-coded step per latent) gives slope −1, and a relay stub gives slope 0.

## 3. D2 / D3: what counts as computation in the latents (M3)

**Can a question-blind R_t decoder be satisfied without z carrying the computed state?**
- **From priors:** no. The names and values are item-specific. The z-shuffle control (D on a same-n random item's z; match to the *true* R_t) is exposed and suffices. PASS.
- **From latent KV rows:** yes. Those rows attend to the question and contain copied question content. **D must read z_t only.**
- **From a question copy in z_t:**
  - The **program tokens** of R_t (names, verbs, the remaining steps) are a substring of the question with constants substituted. They *are* decodable from a copy. That is carriage, not computation.
  - The **value tokens** of R_t (the computed quantities) need t steps of arithmetic. If D has the capacity, it can do them itself (the §204 failure: "the decoder computes"). A shared 12-layer trunk (as §175) can do one or two steps per decode, since DIRECT n 1–2 is ≈ 0.98.
  - So D must be a **separate head of ≤ 2 layers**, and its capacity must be certified on the failure locus. **COPY-CONTROL:** train the same D architecture to emit R_t from a question-only encoding (e.g. z_1 of the trained model with latent steps 2..t skipped, or the K0 / DIRECT `<bot>` state). Require VALUE-EXACT(R_t) ≤ floor + 0.10 for t ≥ 2. This is also Audit's C7 rule (c) (input-only control).

**Definition to register.** The latents carry **computation** at step t iff:
1. **DECODABLE:** VALUE-EXACT(R_t) from z_t ≥ 0.8 at t = ⌈n/2⌉ in the C1 band, while COPY-CONTROL stays at floor;
2. **USED and PORTABLE:** PORT-FOLLOW ≥ floor + 0.30 with BUDGET-FRONT. The continuation performs about the remaining steps only;
3. **NOT PRECOMPUTED:** ANSWER-ONSET.

Carrying the program text in z is necessary for portability but earns no credit alone. If (1) fails, a PORT-FOLLOW null reads **M1-NOT-MET**, not decodable-but-unused.

**Further specifics.**
- Per-token CE weighting: copied tokens dominate R_t's CE (§175 reported handle and copy separately for this reason). Report VALUE-EXACT and COPY-EXACT separately.
- The value-token CE may be up-weighted, as a choice stated up front, not tuned on readings.

## 4. Precompute and G-W0 (B3)

**Will one pass fail at depth up to 8?** At d ≥ 6, almost surely. §170 DIRECT (Feng nested, mod 11, 48k × 64, GPT-2 small) reads:

| n | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| s1 | 0.635 | 0.430 | 0.254 | 0.181 | 0.144 | 0.154 |
| s2 | 0.750 | 0.533 | 0.349 | 0.232 | 0.153 | 0.164 |

- The n 4–8 means are 0.23 / 0.29 (chance 0.091). At d 3–4 one pass is far from chance, and the latent loop gives K passes that can each do about two steps. So "one step per latent" is not forced, and precompute-ahead is likely at small d (BUDGET-FRONT's chunked case).
- In the §213 nested work, the VB cells' latent accuracy falls fast past training depth (D3-T latent n 12 = 0.127), while the one-step chain stays exact (213-35). One-step competence is easy; depth in one pass is not.

**But the draft's task is not §170's.**
- With uniformly random references, DAG depth is about ln n (dag.py: 2.56 / 3.08 / 3.49 at n 4 / 6 / 8; P(depth = n) = 0.000 at n 8), and about half the steps are irrelevant.
- Forcing a chain makes depth = n, but constant-operand affine steps compose in AGL(1, 11), a solvable group: §170's "flat chain loses seriality" flag.

**Generator fix.**
- A forced spine: step i reads q_{i−1}.
- With p ≥ 0.5, step i also reads a random earlier quantity with a **var·var** op (+, −, ×). Multiplicands are nonzero.
- d = spine length is the stratifier. Every step is relevant (assert it).

**G-W0 band (register).**
- DIRECT = plain GPT-2, [question] → answer, at matched exposure (24k × 64, the same data), on the cloud card class.
- n\*_d = the smallest d with DIRECT < 0.90, fixed in a dated note before any latent output (as 170-3).
- C1 band = {d ≥ n\* + 1 with DIRECT(d) ≤ 0.50}. It must contain ≥ 3 depths within 1..8, else deepen the training range before launch.
- Surface-shortcut panel (as G-L0): heuristic predictors (last-constant, op-count parity, modal answer) ≤ chance + 0.05 per d.
- **F2 (the short local run) is for generator debugging only.** It never sets the band: an under-trained one-pass model underestimates n\*, which biases the gate toward passing.
- The PS cell's own K0 is not a one-pass baseline under M2 (see M7).

## 5. Goodhart routes for M1 + M2 (control-must-be-exposed)

| Route | Draft guard | Exposed? | Fix |
|---|---|---|---|
| D emits R_t from priors | z-shuffle | yes | keep; same-n donor, match to the true R_t |
| D computes R_t's values from a copied question | none | **no** | ≤ 2-layer D on z_t only plus COPY-CONTROL (§3) |
| R_t program carried, values not | none | no | VALUE-EXACT separately; the DECODABLE gate |
| z holds the answer early (precompute) | STATE-ONSET (descriptive) | partly | ANSWER-ONSET and BUDGET-FRONT conjunctive |
| z relays the donor question; the continuation recomputes | none | **no** | BUDGET-FRONT (relay gives a flat front) |
| Answer leaks via `<bot>` / BOS / positions | "QKV removed → identical" | **no** (vacuous) | mask `<bot>`; MARKER-SWAP; exact length |
| Decodable but unused | the primary | yes, if pairs are exposed | B2's pair filter; QUESTION-SWAP(t) as the mirror arm |
| Pass-index clock instead of state | none | — | STALL (repeat z_t at t+1; the answer is unchanged only for a state) and LAST-ONLY |
| C1 removal via K0 | implied | **no** (constant answer under M2 by construction) | MEAN-Z and truncation K′ < d (accuracy → floor) |
| A one-sided dynamics target | none in v0 | — | PASS (C-d honoured) |

**QUESTION-SWAP(t) exposure.** It switches only the recipient's question to the donor's from latent t+1, keeping the recipient's state. The donor must differ in the **remaining** program (steps > t); a donor differing only in consumed steps is vacuous.

## 6. Is NESTED-WORDS language?

**The skeptic's case.** With one frame per op and mod-11 wrap-around ("Cara has twice as many as Bob" = 3 when Bob has 7), this is a bijective re-spelling of symbolic arithmetic. GPT-2's language knowledge is unused (arguably harmful: magnitudes contradict the text). Nothing transfers to GSM8K, and §213's synthetic VB already showed a portable state. On the "language" axis it is a modest step past ProsQA-SYM.

**Minimal realism that keeps exact answers.**
1. **Natural integers** in a bounded range (rejection-sample every intermediate into 0..60, single GPT-2 tokens). Otherwise use an explicit modular framing ("hours on an 11-hour clock") used consistently. The floor is then the permutation floor of answers (Σp² measured).
2. **≥ 8 paraphrase frames per op**, drawn from GSM8K phrasing ("more than", "gives away", "twice as many", "splits equally" only if exact). **2 frames per op held out** at test, as descriptive PARAPHRASE-OOD.
3. **Numeric distractor sentences** (GSM-IC / NoOp style): quantities not on the asked spine, in 0–2 sentences per item, with names from the same pool.
4. **Out-of-order statements** where dependencies allow, so that R_t's order (topological, with ties broken by sentence order) is not the sentence order.
5. Wording: "a templated arithmetic word task (paraphrase-varied; one seed)". Never GSM8K, never C4.

**Is it a step toward GSM8K?** Only through the CODI family. A positive would justify the same recipe on GSM8K-Aug with DECOMP-derived R_t (§218 B1's programs), subject to Audit's C4 ruling.

## 7. OC

**Floor.** Pairs are filtered to distinct outcomes, so donor-final appears under no portability only among "other" outputs. The permutation floor is ≈ 0.05–0.10; 0.09 is assumed below.

**P(pass) at bar floor + 0.30** (oc.py; design effect 1.3 for donor reuse ≤ 3; 200k simulations):

| true follow − floor | 0 | 0.20 | 0.25 | 0.28 | 0.30 | 0.32 | 0.35 | 0.40 |
|---|---|---|---|---|---|---|---|---|
| N 300 | 0.000 | 0.0006 | 0.058 | 0.27 | 0.51 | 0.74 | 0.94 | 0.999 |
| N 500 | 0.000 | 0.000 | 0.021 | 0.22 | 0.51 | 0.79 | 0.98 | 1.000 |
| N 1000 | 0.000 | 0.000 | 0.002 | 0.13 | 0.51 | 0.88 | 0.998 | 1.000 |

- **P(pass | no portability)** (own, carriers, or partial follow ≤ floor + 0.20) ≈ 0 by sampling.
- The structural false positives (precompute, relay) are **not** sampling events. Without B1's guards they could carry most of the pass mass. With BUDGET-FRONT they need a falling front, which neither produces.
- **P(pass | portable)**, with true follow ~ U[floor + 0.30, floor + 0.80], is 0.98–0.99 at N ≥ 500.
- **Prior (shaded):** none or carriers 0.80, partial 0.08, portable 0.12, so **P(PORT-FOLLOW) ≈ 0.12**. P(PORT-FOLLOW ∧ BUDGET-FRONT ∧ ANSWER-ONSET) ≈ 0.08–0.10.
- A synthetic test set makes N ≥ 1,000 exposed pairs cheap. Register N ≥ 1,000 (200 per depth in the band) and t = ⌈d/2⌉ for the bar. BUDGET-FRONT is pooled over t ∈ [2, d − 2] (open question 1: both).

## 8. Cost

**Measured timing.**
- §209: 25,279 s / 11,965 steps = **2.11 s/step** at effective bs 128 (micro 64 × 2) on a 3090, with three passes per step (student loop, no-grad teacher, grad teacher) at GSM8K-Aug lengths. That cost $1.51, about $0.215/h.
- §210: 117,726 s / 59,822 = **1.97 s/step**, about $0.26/h.

**PS2.**
- PS2 drops both teacher passes but adds up to 10 sequential latent steps (launch-bound) and D's teacher-forced decode of Σ_t |R_t| tokens. With a ≤ 2-layer D that is roughly 1/6 of the trunk cost per token.
- Projection: about 0.8–2 s/step at bs 64, so 24k steps take 5–13 h, **$1.5–4 per cell**.
- Cells: DIRECT, PS, M1-only, M2-only gives **$6–16**. Evals and F1–F3 are local at $0. The cap ≤ $30 holds.
- Run G-TIME on the real shapes (d 8, K 10, the longest R_t) and G-MEM at the deepest shape before admission.

## 9. Top v1 changes (in order)

1. **B1:** co-primary BUDGET-FRONT (decode-free) and ANSWER-ONSET, with known-answer stubs (serial slope −1, relay 0) at F3.
2. **B2:** graph-matched, exact-length, same-name, outcome-distinct pairs; N ≥ 1,000; exposed share reported.
3. **B3:** spine plus var·var generator stratified by d; G-W0 from a matched-budget DIRECT cell, n\* fixed before latent outputs; band d ≥ n\* + 1 with DIRECT ≤ 0.50 and ≥ 3 depths.
4. **M3:** D = a ≤ 2-layer head on z_t only; COPY-CONTROL; VALUE-EXACT gate (DECODABLE), else M1-NOT-MET.
5. **M2 / M1:** stricter M2 (mask `<bot>`); MARKER-SWAP; correct D1 (STATE-SWAP is not trivial).
6. **M7:** cells DIRECT / PS / M1-only / M2-only; C1 removal by MEAN-Z and truncation; QUESTION-SWAP(t) descriptive.
7. **M6:** naturals or a clock framing, paraphrase frames with held-out frames, distractors, out-of-order statements.
8. **M4 / M5:** prior about 0.10, with the null likeliest; cite SIM-CoT, §175, §211 / §213 and §218; Literature check.
9. **M8 and minors:** pin indices; n-bucketed K; explicit QMASK mask; λ = 1 per-token mean, stated; censoring; G-TIME; one-seed wording.

**Answers to draft §6:**
1. Both: the bar at t = ⌈d/2⌉, the front pooled over t.
2. Keep the question visible. Question-once would make follow near-trivial, which is §208's design.
3. λ = 1 per-token mean, stated as a choice, plus gradflow per term.
4. CODI, for §221 comparability and the GSM8K bridge.
5. Run both M1-only and M2-only; each ablation attributes the other change.
