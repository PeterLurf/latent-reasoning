# REDTEAM_LANG: adversarial review of LANG_draft.md v0 (LANG-VB)

- **Reviewer.** Opus red-team subagent for the Theory session. Started 2026-10-18 12:51 EDT (`date`).
- **Scope.** $0: CPU only (≤ 2 threads, low priority, `python -B`); the repo is read-only.
- **Target.** D:/Latent/docs/research_lang_2026-10-18/LANG_draft.md (v0, 12:38 EDT 10-18).
- **Scripts.** research7/rt_lang/ (sym_attack.py, sym_decoy.py, mem_cost.py, oc_shade.py).
- **Severity.** BLOCKER = registration must not proceed without the fix. MAJOR = a reading, an OC row or a cost line is wrong in a way that changes a decision. MINOR = wording, or a gate detail.

(Written incrementally. All sections were complete at 13:15 EDT 10-18.)

## 0. Summary (completed 13:15 EDT 10-18)

**Verdict.** Do not register v0. Three BLOCKERS, each with a cheap ($0, local) fix. Several MAJORs change the OC, the bands and the cost lines, but not the design's shape. The $25 family cap is realistic (about $8.5–16); the per-cell admissions and the drafted OC headline are not.

**BLOCKERS**
1. **F1-1: precompute-and-carry passes VB-C1-L and VB-C1-GSM unless an uncertified probe finds it.**
   - The decisive swaps use arm (ii) (the whole V_t), which carries a hidden answer, so they read follow. STALL and the revealed onset also pass.
   - The detectors that do separate it are §4.4's slot-only arm and FUTURE-DECODE. Neither is in VB-C1-L. GSM8K's FD as registered never computes FD(k, k), and GSM8K's probe has no PROBE-VALID.
   - *Fix:* PLANT positive controls for every PROBE(t ≤ 2); FUTURE-DECODE-L (per-slot dist = t + k, k ≥ 3, AUROC ≤ 0.65) made decisive; a slot-local arm co-primary; PROBE-VALID-GSM and an MLP FD probe.
2. **F1-2: §4.6(c) [PROVED] is false.**
   - An ALL-PERSONS serial model, one hop per pass, carries person-tagged reach sets. Those sets are question-independent, so they are identical in a PERSON-SWAP donor and recipient. The result is own ≈ 1, and RECOMPUTE-L mislabels it.
   - PROOFS_LANG found a second hole: arm (ii) also allows a history read.
   - *Fix:* PREMISE-SWAP (R ↔ P swapped in the premises) × arm (ii′) (splice V_1..V_t), re-proved.
3. **F3-1: the A2 pool is 240k kept against a demand of 384k** (self batches are whole bs-64 steps, vbank.py line 755). Exhaustion means CONTAMINATED: A2 fails with certainty. *Fix:* ≥ 106k kept per D, a draw cap ≥ 530k per D, and an A0-9 assert.

**MAJORs**
- **F2-1** SYM is a pointer chase backward: an any-parent chase is exact (1.000) at every D. Fix: decoys (prototype brings it to 0.61–0.66) and gate on H-ROOT.
- **F2-2** The C1 bands rest on an unmeasured one-pass boundary. Fix: a DIRECT-SYM cell.
- **F2-4** Every CON38 slot sits on the token "us".
- **F2-6 / F3-3** L-RR's re-roots are out of distribution in 65–73% of D 9–12 items.
- **F3-2** Reject-on-any-false-mark: needs FP ≤ 0.002, biases kept items. Fix: SOUND-CLOSURE.
- **F3-5 / F6-4** 15–37 GPU-h of labelling is unplanned.
- **F1-5 / F4-2** The trained-coda K′1 interim GO passes under §208 precompute (≈ 0.5).
- **F4-1** DECOMP's calculator-only sub-steps put "C4-legal" at risk.
- **F4-4** BEATS with seed variance: 0.12, not 0.02.
- **F4-5** Gate B1 on VB-C1-L.
- **F5-1** SD-C2-L passes under H_shift if A1 already extrapolates. Fix: MARGIN-GROWS-L and a required A2-CTRL.
- **F5-2** The CurGate / NO-ENTRY CE is about 0.17 at chance under a per-token mean, below the 0.35 threshold.
- **F6-1** to **F6-3**: T is 1.2–1.5× the assumption, A2 does not fit bs 64, and both A cells likely fail their own G-TIME admission.

**Shaded OC headline** (assumed mixtures in rt_lang/oc_shade.py).
- As drafted: P(VB-C1-L) ≈ 0.09, not 0.2. P(VB-C1-L | no serial use) ≈ 0.14, not ≤ 0.02. A quarter of ProsQA positives and about 60% of GSM8K positives would be false. P(VB-C1-GSM) ≈ 0.06. P(SD-C2-L) = 0 (pool).
- With the fixes: P(VB-C1-L) ≈ 0.09 (false share 0.07), P(VB-C1-GSM) ≈ 0.03, P(SD-C2-L) < 0.01, and **P(any decisive C1) ≈ 0.10** (draft 0.25).


## 1. Item 1: is the latent arm forced to use the bank serially? (§207 recompute, §208 precompute)

**Short answer.** No: the draft says so itself (§4.2, §4.4). Serial use is only detected, so everything rests on the detectors, and two of them are weaker than the OC assumes. Under the architecture:
- V_0 = all T rows of H_6 sit in every core layer's bank at every pass. The question's tokens are in the last sentence, so in the causal prelude they see every premise.
- The question's option tokens can therefore run a backward chase in the prelude (F2-1), and every core pass can read the result. Precompute through V_0's question rows is open. PROBE(1) sees it only if the core copies it into the slots.

**F1-1 (BLOCKER) Precompute-and-carry passes VB-C1-L (and VB-C1-GSM) unless the probe happens to find it.**
- *Counter-model PaR-C.*
  - Pass 1 computes the answer (or every REACH layer) into a subspace that R cannot read, and every later version carries it forward. SLVA is RNN-IN-FUNCTION, so the newest version is the pass input.
  - The readable IN bits are revealed one hop per pass, keyed on content (the current readable frontier).
  - R1 and R2 are met exactly. G1′ in §6.3 states this route is zero-loss.
- *What each decisive component does under PaR-C:*
  - SERIAL-ONSET-L passes, because PROBE(t) is initialised from the trained coda, which reads the revealed bits. Its onset therefore tracks the reveal.
  - **SERIAL-STEP-L passes.** follow_ii splices the donor's *whole* V_t, which carries the donor's hidden answer, so follow ≈ 1.
  - NO-CLOCK-L passes: a content-keyed reveal recovers from a stall.
  - DEEP passes.
  - Only PROBE-GAIN-L stands in the way. That holds only if a 2k-step probe, initialised to read the revealed bits and trained on 20k items, also finds a hidden code.
- *The gaps in the draft:*
  - §4.4 names the two readings that do separate PaR-C, the slot-only splice (i) and FUTURE-DECODE. Neither is in VB-C1-L. FUTURE-DECODE does not exist for ProsQA, and follow_i is not decisive in either conjunction.
  - GSM8K's probe has **no PROBE-VALID at all** (§5.2). NOT-ONE-PASS-CHAIN uses a *linear exact-digit* probe, which misses any nonlinear hidden code.
- *Shaded* [rt_lang/oc_shade.py]: P(VB-C1-L | no serial use) ≈ 0.14, not ≤ 0.02. P(VB-C1-GSM | no serial use) ≈ 0.05, not ≤ 0.01. About 26% of ProsQA positives and 60% of GSM8K positives would be false.
- *Minimal fix (all $0, local):*
  - **(a) PLANT positive control for every PROBE(t), t ≤ 2.**
    - Add ±ε·u to the option slots of V_t: a random unit direction, ε = the std of V_t's 50th principal component, signed by the true option. Use a second, XOR-coded plant on two directions as well.
    - Require planted PROBE ≥ 0.90. Otherwise PROBE-INVALID, which censors.
  - **(b) FUTURE-DECODE-L on ProsQA, made decisive.**
    - A fresh per-slot linear probe and a 1-hidden-layer MLP probe: V_t[c] → 1[dist(R, c) = t + k] among the slots with dist > t + 1. Train on 20k fresh items (the labels are dense: about 30 slots per item).
    - **NO-PRECOMPUTE-L** = AUROC ≤ 0.65 at k ≥ 3, t ∈ {1, 2} (D 6–8).
    - k = 2 is descriptive: it measures hidden over-reach, G6.
  - **(c) A slot-local splice arm co-primary** (follow ≥ 0.8 ProsQA / ≥ 0.6 GSM8K).
    - On ProsQA, define it as the slots within one hop of either world's frontier at t, never the options and never slots ≥ t + 2 from both persons.
    - That transfers a legitimately serial state, including a one-hop look-ahead, while leaving PaR-C's hidden answer behind.
  - **(d) GSM8K:** PROBE-VALID-GSM = PROBE(14) ≥ acc_fixed14 − 0.03 on the same items. Plus the PLANT control on the slot-t value digits, and an MLP FD probe beside the linear one.
  - **(e)** Train probes to a plateau (≥ 10k steps on ≥ 200k fresh SYM items; fresh data is free), with the learning curve stored.

**F1-2 (BLOCKER) §4.6(c) [PROVED] is false as stated. PERSON-SWAP cannot tell serial use from recompute.**
- *Counter-model ALL-PERSONS (serial, one hop per pass).*
  - Each slot carries S_t(c) = {persons p : dist(p, c) ≤ t}, propagated from version t−1 through the edges, plus the readable IN_t(c) = 1[R_question ∈ S_t(c)], with R read from V_0's question rows every pass.
  - R1 and R2 are met exactly, no over-reach is used, and the bank is used serially.
  - This is natural here: V_0's slot rows are question-independent (causal prelude, question last), so person-tagged features exist before any pass.
- *Under PERSON-SWAP:*
  - Donor and recipient share every premise, so S_t is identical in both. The splice changes only the IN bits, which the next pass recomputes from S_t and the recipient's question.
  - Result: own ≈ 1, follow ≈ 0. SERIAL-STEP-L fails, and **RECOMPUTE-L (own_ii > 0.3) mislabels a serial model as recomputing**.
  - Arm (i) gives the same result, since the S-sets are equal.
- §4.6(c)'s proof silently assumes that the stored state encodes R_t(asked person) only.
- *Fix: PREMISE-SWAP.* The donor is the recipient with the names R ↔ P exchanged in the **premises** (the question is unchanged), so the donor's answer is o_P.
  - Under the question-conditional serial rule: the donor's IN set is W_P, so the result is follow.
  - Under ALL-PERSONS: the donor's S_t tags W_P with R, so the recipient continues W_P, also follow. The recipient's V_0 re-seeds W_R at depth 1 and needs D more passes, so for t ≥ 3 o_R is not reached by K′ = D + 2.
  - Under recompute from V_0: own.
  - So PREMISE-SWAP identifies both serial variants. Keep PERSON-SWAP as a descriptive mechanism probe: own ≈ 1 under PERSON-SWAP together with follow under PREMISE-SWAP *is* the ALL-PERSONS signature.
  - Re-prove §4.6 for PREMISE-SWAP; the [PROVED] tag stays off until then.
- *Combine it with PROOFS_LANG's independent correction of §4.6(c).* Arm (ii) leaves the recipient's V_1..V_{t−1} in the bank, so "own" can also be a history read. Make the decisive arm **PREMISE-SWAP × (ii′)**: splice the donor's V_1..V_t.
  - Both serial variants then read follow, and recompute-from-V_0 reads own.
  - Precompute-carry also reads follow under (ii′), so F1-1's PLANT, FD-L and slot-local arm are still needed.

**F1-3 (MAJOR) CAP-aff-L (c₀ 1–3) is unsupported for fresh data and backward chasing.** See F2-1 and F2-2. Its effect is on P(effect), not on false positives: the likeliest SYM outcome moves from "serial" to "partial one-pass at D 5–6, PROBE-GAIN pooled < 0.35". Fix: DIRECT-SYM, and C1 bands set from the measured D*_fresh.

**F1-4 (MINOR) PROBE(0) is exactly ½ on ProsQA by construction.** V_0's slot rows cannot see the question (causal prelude, question last), and the coda is question-free. So max(PROBE(0), PROBE(1)) = PROBE(1). State it, so nobody reads PROBE(0) ≈ 0.5 as evidence.

**F1-5 (MAJOR) Two "precompute detectors" are trained-coda readings at untrained K′.**
- The coda's R1 window starts at τ_i = m_bin (GSM8K), so it never trains for t < m_bin. acc_k[1] on m ≥ 3 items is therefore an out-of-distribution read, low whatever V_1 holds.
- (i) PASS1-PRECOMPUTE-GSM's second clause (acc_k14 − acc_k1 < 0.03) almost never fires.
- (ii) **B0-5's interim GO rule** (acc_k14 − acc_k1 ≥ 0.03 ∧ valid acc ≥ 0.17) passes under §208-type precompute with probability about 0.5: the trained coda reads nothing at K′ 1, and precompute models reached 0.14–0.19 at this exposure. So the "$1 cap on a precompute failure" does not hold.
- Fix: GO iff a fresh PROBE(1) gain on Aug-valid is ≥ 0.05 (and FD-MLP(1, k ≥ 3) < 0.8·FD(k, k)). Run it locally on the mirrored 24k checkpoint while the host continues, and stop the host on fail. Drop the trained-coda clause from PASS1-PRECOMPUTE-GSM.

**F1-6 (keep) D-4 causal current-version attention** limits within-pass pointer doubling: a slot can read only earlier slots' current-layer values. That is the main structural brake on one-pass precompute inside the core. Keep it. Theory could consider slot-diagonal current-version attention (no cross-slot flow within a pass) as a stricter variant.


## 2. Item 2: SYM shortcuts [MEASURED rt_lang/sym_attack.py -> rt_lang/sym_attack.json; 1,000 fresh draws per D, seed 777; the draft's sym_proto.make imported read-only]

The person-agnostic proof (§3.1) is correct: any heuristic that ignores which person is asked has expectation 1/2. So the six §59 cues are vacuous on SYM by construction, and the LEVEL-PANEL gate tests only the implementation. The open class is root-aware heuristics. Measured (direction-free accuracies):

| D | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H-ROOT (follow any single parent back to a person; first-in-text or random parent) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mean chase length (first parent) | 3.3 | 4.5 | 5.7 | 6.8 | 9.1 | 10.2 | 11.5 | 12.6 | 13.7 | 15.9 | 18.5 |
| option ancestors with exactly 1 parent | 0.93 | 0.87 | 0.84 | 0.83 | 0.81 | 0.80 | 0.80 | 0.79 | 0.79 | 0.78 | 0.78 |
| backward ancestor set / forward reach set (full) | 3.9 / 9.0 | 5.7 / 12.2 | 7.6 / 15.1 | 9.3 / 18.3 | 12.9 / 24.2 | 14.7 / 27.2 | 16.9 / 30.3 | 18.5 / 33.0 | 20.2 / 36.1 | 23.8 / 41.8 | 27.7 / 48.0 |
| UBIDIR(3) (undirected 3-balls from R and from each option meet) | 0.68 | 0.75 | **0.81** | **0.83** | 0.59 | 0.57 | 0.56 | 0.56 | 0.55 | 0.53 | 0.53 |
| UDIST (the option nearer to R by undirected distance; a full but direction-blind search) | 0.93 | 0.88 | 0.81 | 0.73 | 0.67 | 0.65 | 0.64 | 0.62 | 0.61 | 0.57 | 0.56 |
| items whose largest BFS layer has > 4 concepts (L-RR then writes > 4 "R is a f." statements) | 0.34 | 0.43 | 0.45 | 0.55 | 0.61 | 0.65 | 0.66 | 0.70 | 0.73 | 0.77 | 0.84 |

**F2-1 (MAJOR) Backward, SYM is a pointer chase.**
- Every ancestor of o_R lies in W_R ∪ {R}: cross edges enter only leaves, and a leaf is nobody's parent. So following ANY one parent from each option ends at its world's person, and H-ROOT is exact at every D.
- 78–93% of option ancestors have exactly one parent, and the backward ancestor set is about half the forward reach. "SYM branches at every depth" is true forward only. Backward, SYM has the 168-6 "serial lookups" structure again, and a soft (superposed) backward chase never leaks across worlds.
- *Failure scenario.* The question's option tokens are in the last sentence, so in the causal prelude they see every premise. They chase back about one hop per 1–2 layers, and the core's option slots continue for 4 more layers. Pass 1 then holds root(o1) and root(o2) for D up to about 5–8. PROBE-GAIN-L fails at D 5–6 (a null, not a false positive). If the answer comes from the chase, the "graph search" wording is wrong.
- *Fix (cheap; generator only):* **ancestral decoys**. Per world, add 1–2 decoy persons, each with a 1–3 node chain into a random path node a_k (k ≥ 2).
  - dist(R, o_R) = D and o_P ∉ Reach(R) are unchanged, and the law stays σ-symmetric.
  - H-ROOT drops toward chance, and the backward problem becomes a set-valued search (prototype in §2b).
  - Report H-ROOT in the LEVEL-PANEL and gate on it.

**F2-2 (MAJOR) The C1 bands rest on an unmeasured one-pass boundary.**
- Directed BIDIR(r) is exact once 2r ≥ D, and UBIDIR(3) is already 0.81–0.83 at D 5–6. "C1 bands D ≥ 5" assumes one pass does at most about 2 hops from each end (CAP-aff-L, c₀ 1–3).
- The cited evidence for a low c₀ is weak:
  - DIRECT's D* = 3 was measured on fixed p8_na data, where it memorised;
  - §191's K′ 0 / 1 are trained-decoder readings at untrained K′ (the trained-readings rule).
  - With fresh per-draw items, nothing bounds c₀. NORTH_STAR C1 needs "problems that exceed one forward pass", which needs a measured boundary, as §168 had.
- *Fix:* a **DIRECT-SYM** cell: plain GPT-2, K = 1, the same fresh stream and exposure. About $0.4–0.8 on a 3090, or local if it is short. It gives a measured D*_fresh.
  - C1 bands = D > D*_fresh, fixed before any A1 output.
  - If D*_fresh ≥ 6, raise A1's gold range to D 3–12 rather than keep D 5–8.

**F2-3 (MINOR) Chance is not 1/2 for search-like readings.** A direction-blind search scores 0.61–0.65 at D 9–12 and 0.56 at D 13–16. The ABSORB / DEPTH-BEYOND bars (0.95 / 0.80) sit far above that, so nothing passes on it. But wherever the wording says "chance" for deep readings, it should say "the UDIST floor".

**F2-4 (MAJOR) Every concept slot sits on the same BPE token.**
- All 38 CON38 names end in the token "us" [MEASURED with tiktoken gpt2: 3 tokens ×30, 4 ×6, 2 ×2; last token "us" in 38 / 38]. The prototype's NONCE names also end in "…us".
- The draft puts each concept slot at the last BPE token of the concept's first mention. Slot identity is then carried only by context from the preceding 1–3 tokens, and every name match (finding the parent's slot) compares contextualised copies of the same token.
- This is not a shortcut. It is a binding risk for NO-ENTRY and for the coda's name copy.
- *Fix:* build NONCE-300 with 300 distinct final tokens (or put the slot on the first token, or mean-pool the span), and smoke L-P binding on that set in A0-4.

**F2-5 (MINOR) Person tokens.** Two ENT17 names are 2 tokens (" F" + "ae", " W" + "ren"). PERSON-SWAP R → P then shifts the question rows by one position. *Fix:* restrict R / P to single-token names, or assert equal lengths in the pair maker.

**F2-6 (MAJOR) L-RR inputs are out of distribution.**
- In 55–61% of D 6–8 items, and 65–73% of D 9–12 items, some BFS layer has more than 4 concepts. The re-rooted input then carries more than 4 "R is a f." statements, while training draws r ~ U{1..4}.
- Expect L-RR's yield to fall with D for this reason, not because of depth.
- *Fix:* train A1 with r ~ U{1..8} (or U{1..max layer}), or re-root in chunks of ≤ 4. Record the frontier size in GATE-1-L.


### 2b. F2-1 fix prototype: ancestral decoys [MEASURED rt_lang/sym_decoy.py -> sym_decoy_n12.json (600 per D), sym_decoy_n24L.json (1,000 per D)]

| variant | D | invariant fails | max §59 cue (all levels) | BIDIR(2) | H-ROOT first / random / max-outdegree parent | full backward search (soft) | words mean |
|---|---|---|---|---|---|---|---|
| SYM (draft) | 5–16 | 0 | 0.518–0.552 | 0.50 | 1.000 / 1.000 / — | 1.000 | 189–579 |
| + 1–2 decoy chains per world | 5–16 | 0 | 0.545–0.553 | 0.50 | 0.75–0.79 / 0.75–0.78 / **0.92–0.94** | 1.000 | 232–622 |
| + 2–4 decoy chains per world, chain nodes with Binomial(2, ½) leaves | 3–16 | 0 | 0.514–0.542 | 0.50 (1.00 at D 3) | 0.64–0.65 / 0.63–0.66 / 0.61–0.62 | 1.000 | 261–722 |

**Reading.**
- Plain decoy chains are easy to avoid: a "prefer the parent with more children" chase still scores 0.92–0.94. The decoys need the path nodes' child law.
- With that law and 2–4 decoys per world, every single-parent chase falls to 0.61–0.66, while the set-valued backward search stays exact. The task is then a branching search in both directions.
- To get H-ROOT under the 0.6 gate, decoys must be denser still, or attach preferentially near the options. Text grows by 25–50% (D 16 then exceeds T 1,000 without the per-world cap).
- **Recommendation:** add H-ROOT (first / random / max-outdegree) to LEVEL-PANEL's gated set. Tune the decoy law in A0-2 until H-ROOT < 0.6 at every band, inside the T ≤ 1,000 cap. Exchangeability is preserved because the decoy law is identical in both worlds.
- **Alternative if T is binding:** keep SYM, and word the result as "serial latent reachability (forward REACH-supervised), with the backward chase not excluded". This is weaker, and F2-2's DIRECT-SYM boundary then matters more.

## 3. Item 3: SOUND verifier, noise, selection, the labeller's reach

**Soundness holds, and is robust.**
- Lemma S-L needs only P1–P3 checked against the item's own statements. Any mark that passes has a statement chain from R, so M_t ⊆ R_t, whatever produced the marks.
- So a labeller bug (for example L-RR mapping marks back by slot index after a re-render shifts first mentions) cannot create a wrong label. It can only lower the yield.
- With SYM's invariant o_P ∉ Reach(R), π_label = 1 is a theorem, and SOUND-ONLY R2 targets equal gold at the same (slot, pass). I found no counterexample that makes SOUND accept a wrong *target*.
- SOUND does accept *incomplete* states, but those are never supervised (no OUT terms, no unmarked-slot terms).
- One precondition: SOUND must parse the **rendered text the model saw**, not the generator's graph (the draft says "parsed from its text"; keep it). A render bug then cannot open a gap.

"Zero noise" holds for target values. It does not mean zero bias:

**F3-1 (BLOCKER) The A2 pool is too small by construction.**
- vbank.py (line 755) draws the self coin per *step*, so a self step is a whole bs-64 batch. A2's demand is 12,000 × 0.5 × 64 ≈ **384,000** fresh-once self items.
- The draft builds 60,000 kept per D × 4 = **240,000**. The pool runs out at about step 7,500. §6.2 then makes A2 **CONTAMINATED: every A2 reading fails**, with certainty.
- §215 used 110,000 per n for exactly this law.
- *Fix:* ≥ 106k kept per D (422k total, a 10% margin), a draw cap ≥ 106k / 0.2 = 530k per D, and A0-9 asserting kept ≥ 1.1 × p_self × steps × bs before launch.

**F3-2 (MAJOR) Reject-on-any-false-mark couples yield to the false-positive rate, and biases the kept items toward small graphs.**
- KEEP needs every new mark, on every slot and every pass, to be sound. At D 12 a chain makes about 500–700 OUT-slot decisions (60–73 concepts × 12–14 passes, minus the reached ones; sym_proto sizes).
- So yield ≈ P(reach) × (1 − f)^600. Yield ≥ 0.2 at P(reach) 0.6 needs a per-decision FP rate f ≤ 0.002 at depths A1 never trained on.
- *Failure scenarios:*
  - A0-15's yield gate fails, and gets read as "no depth". It is really FP compounding.
  - Or the gate passes on the smallest worlds only: kept items skew to low concept counts, short T and few cross edges.
  - Then ABSORB on the full test distribution, and the D 13–16 extrapolation, measure that shift as well as depth.
- *Fix: SOUND-CLOSURE.*
  - Keep the item if the **maximal sound sub-chain** reaches an option: M′_1 = M_1 ∩ direct(R); M′_t = M′_{t−1} ∪ {c ∈ M_t : an in-edge from M′_{t−1}, or direct}.
  - Supervise IN on M′ only. Lemma S-L applies to M′ unchanged (π = 1, zero noise), and isolated false marks no longer veto the item.
  - Report kept-vs-drawn distributions of T, concept count, cross-edge count and frontier width by D, and FALSE-IN on GATE, before A2.

**F3-3 (MAJOR) L-RR is out of distribution** (F2-6). In 65–73% of D 9–12 items the re-rooted input has more than 4 direct statements, against U{1..4} in training. Fix: widen r in A1's law, or re-root in chunks.

**F3-4 (MAJOR) The labeller's reach is thin, and the OC ignores it.**
- The draft's own central DEPTH-L (A1, D 9–12) is 0.55, near chance and inside the UDIST floor (0.61–0.65).
- L-ROLL needs passes 13–16, beyond A1's trained K max of 12. L-RR needs in-distribution re-roots (F3-3).
- I put P(yield ≥ 0.2 at every D 9–12 | A1 DEEP ∧ NO-CLOCK) ≈ 0.5 after the fixes, about 0.3 before. So P(A2 runs) ≈ 0.15, not "on A0-16" as a near-formality.
- This is a pre-stated C2-negative outcome, which is correct. But Theory should know it is the modal A2 outcome.

**F3-5 (MAJOR) Labelling time is not in the plan.**
- About 422k kept at yield 0.2–0.5 means 0.85–2.1M drawn items × about 8 L-RR steps of a prelude over T ≈ 650.
- That is about **15–37 local GPU-hours in fp32** [rt_lang/mem_cost.py; ASSUMED 8 TFLOP/s and 8 steps per item], plus GATE-1-L and L-ROLL comparisons.
- *Fix:* build the pool on the 3090 host in bf16 (about 4–9 h, +$1–2 against the cap), or schedule the local cards explicitly. The local-short rule makes a ≥ 1-day local job a scheduling decision, not a default.

**F3-6 (MINOR) Selection rule.** "Higher kept yield at D 11–12" picks the labeller with fewer false marks, which is fine. Store both chains' per-item KEEP decisions so Audit can recompute the choice (the store-items rule).


## 4. Item 4: GSM8K Stage B

**Contamination: clean.** [MEASURED, an inline check over bench/data/gsm8k_aug: 384,620 train questions against 1,319 test questions.] There are 0 exact question matches after normalisation, and 7 / 1,319 test items share any 13-gram with any train question (common phrasing). Class-A donors and R_x variants are made from test items for eval only. No issue.

**F4-1 (MAJOR) DECOMP's C4 status is not "C4-legal as distillation" without an Audit ruling.**
- About 30% of train equations are multi-operator (draft §1.3). DECOMP turns them into binary sub-steps whose values are **computed by a calculator and never written in the CoT**. Example: "<<20+40+600+1200=1860>>" yields targets 60 and 660.
- Every operand pointer is a value-matched annotation: the provenance by value, with masked ambiguities.
- NORTH_STAR C4 excludes "hand-built intermediate states". The sub-step values are exactly that, and the pointers are arguably that.
- *Failure scenario:* VB-C1-GSM reads, the wording says "C4-legal", and Audit later withdraws it, the 171-2 pattern.
- *Fix:* Audit rules before registration. The clean option: value-supervise only the CoT-stated equation results (mask the calculator-only sub-step values), and word pointers as "parsed from the CoT". Otherwise word B1 as "CoT-structured, calculator-binarised supervision (C4 not claimed)".

**F4-2 (MAJOR) The interim GO rule does not catch the §208 failure** (F1-5).
- Its K′1 term is a trained-coda read at an untrained K′.
- Its 0.17 bar sits within the one-pass band: §208 0.138, CODI4 0.189, CODI 0-latent 0.171, all at about this exposure.
- So P(GO | PASS1-PRECOMPUTE) ≈ 0.5, and the "$1 cap" is about $4–6 in expectation under the likeliest outcome.
- *Fix:* GO on a fresh-probe gain (PROBE(1) on Aug-valid ≥ 0.05 below acc_fixed14) plus FD-MLP. Read locally from the off-host 24k checkpoint while the host continues; stop the host on fail.

**F4-3 (MAJOR) The GSM8K detectors are underpowered or unreadable as specified.**
- (i) No PROBE-VALID for the GSM8K probe. A 2k-step, 20k-item probe generating multi-digit numbers can fall well below the trained coda, which inflates PROBE-GAIN-GSM.
- (ii) NOT-ONE-PASS-CHAIN reads FD(k, k), but the registered `future {t: {k}}` has t ∈ {0, 1, 2, 4} with k ≥ t + 2, so **t = k is never computed** (found independently by PROOFS_LANG). As written, the reading cannot be computed.
- (iii) FD is a linear exact-match probe on concatenated rows, so it misses nonlinear codes.
- (iv) The decisive swap arm is (ii) (whole V_t), which reads follow under precompute-carry.
- *Fix:* as F1-1 (d), plus FD(k, k) added to the field contract, and slot-t-only follow (i) made co-primary.

**F4-4 (MAJOR, wording + OC) CODI-20 is a fair accuracy comparator only as "matched examples".**
- *Matched:* 7.66M examples, the same 382,859 items, the same 1,319 test items with paired per-item outputs, an RTX 3090 for both (note 210-1 moved CODI-20 off the A100), and the eval re-scored on B1's card (B0-2).
- *Not matched:*
  - LoRA r 128 at lr 3e-3 vs full fine-tuning at 1e-4;
  - bs 128 / 59.8k steps vs bs 64 / 120k;
  - compute per example: B1 runs up to 16 core passes plus about 7 coda reads, CODI a teacher CoT plus 6 latents;
  - R_ν name relabelling on in B1 only;
  - dense per-step supervision in B1 only.
- D-7's "one variable vs CODI-20" is wrong; it is a system comparison.
- BEATS-CODI20's P(pass | no effect) ignores seed variance. With σ_seed ≈ 0.015 per model it is about **0.12, not 0.02** [rt_lang/oc_shade.py]. A one-seed BEATS is therefore weak evidence and must be worded "one seed each, recipe-level comparison".

**F4-5 (MAJOR) Stage B's expected value is low for its share of the spend.**
- Six GSM8K latent recipes so far all recompute (§207) or precompute (§208). The draft's own modal B1 outcome is PASS1-PRECOMPUTE (0.45).
- Shaded P(VB-C1-GSM) is about 0.03 after the detector fixes (0.06 before, 60% of it false).
- B1 is the largest line: $4–6.6, about half of the realistic family spend (§6).
- *Fix (D-13):* gate B1 on A1 reading VB-C1-L. The bounded-core mechanism is the same; if it does not give serial use on binary graph search with dense REACH targets, GSM8K is very unlikely to. This follows the user's "small tasks before GSM8K" rule (memory, 10-10). The saving is about $4–6 with probability about 0.9.

**F4-6 (MINOR)**
- B1-24k vs CODI4: at 24k of a 120k cosine the LR is about 0.9 of peak, while CODI4 had annealed (the 210 caveat). Descriptive only.
- Train items with m_bin > 12 (beyond 12 slots) need a stated rule (mask the steps beyond 12, or keep R1 only); test is 100% ≤ 12.
- K′ 14 is below K_i for every item with m_bin ≥ 15 (none on test). Fine, but assert it.

**Cost realism (B1).**
- The launch floor scales with max K per batch (16 vs D3-T's 12): about 0.55–0.7 s/step.
- 120k steps is then about 18–23 h ≈ $3.9–4.9 at $0.21/h, inside the draft's $3.3–6.6.
- That is a day-long run: the off-host mirror plus a tested resume are mandatory (A0-11 analogue), and the drafted gates cover both.


## 5. Item 5: OC shading, readings that pass under no effect, censoring holes, wording

### 5.1 Shaded OC [rt_lang/oc_shade.py -> oc_shade.json; every input ASSUMED and listed in the script so Theory can re-shade]

**Mixture used for A1.**
- P(entry) 0.55 and P(DEEP | entry) 0.60, so P(DEEP) ≈ 0.33. The draft has 0.6 / 0.67. I shade down for binding when every slot sits on the token "us" (F2-4), and for T ≈ 530–690 rather than 450 (F6-1).
- Given DEEP, the mechanisms are:
  - question-conditional serial 0.35;
  - ALL-PERSONS serial 0.15;
  - precompute-carry or backward chase 0.30;
  - recompute from V_0 0.10;
  - one-pass at D 5–6 with serial use deeper 0.10.

**GSM8K mixture.** P(trains) 0.75, then: serial step 0.08, precompute-carry 0.45, recompute 0.30, step not learned 0.17.

| reading | draft P(pass \| no effect) | shaded, v0 design | shaded, with F1-1 / F1-2 fixes | draft P(pass) | shaded P(pass) v0 → fixed | share of positives that are false, v0 → fixed |
|---|---|---|---|---|---|---|
| VB-C1-L | ≤ 0.02 | **0.14** | 0.04 | ≈ 0.2 | **0.087 → 0.086** | **0.26 → 0.07** |
| P(VB-C1-L \| serial use) | ≈ 0.5 | 0.39 (ALL-PERSONS fails PERSON-SWAP) | 0.49 | — | — | — |
| SD-C2-L | ≈ 0.02 \| H_shift | ≈ 0.25 \| H_shift when A1's DEPTH-L ≥ 0.8 (F5-1) | 0.02 with MARGIN-GROWS-L + A2-CTRL in the conjunction | ≈ 0.05 | **0 as drafted (F3-1 pool)**; ≈ 0.007 fixed | — |
| VB-C1-GSM | ≤ 0.01 | **0.05** | 0.008 | ≈ 0.08 | **0.06 → 0.027** | **0.60 → 0.21** |
| BEATS-CODI20 (Δ ≥ +0.03) | ≈ 0.02 | **0.12** (with one-seed σ 0.015 per model; total SE 0.025) | same; wording only | 0.04 | ≈ 0.04 | — |
| P(≥ 1 decisive C1 positive) | — | 0.14 (B ungated) | 0.11 ungated; **0.095 with B1 gated on VB-C1-L** | ≈ 0.25 | — | — |

**Headline.**
- The draft's 0.25 is about 2× too high, and as drafted about a quarter of its ProsQA positives, and most of its GSM8K positives, would be false.
- After the fixes: P(VB-C1-L) ≈ 0.09 (false share ≈ 0.07), P(VB-C1-GSM) ≈ 0.03, P(SD-C2-L) < 0.01, and P(any decisive C1) ≈ 0.10.
- A positive is strong evidence; a null is weak. The likeliest outcome is now ProsQA "partial one-pass / LOSES-ONE-PASS / NO-ENTRY", not the GSM8K §208 failure.

### 5.2 Readings that pass under no effect

**F5-1 (MAJOR) SD-C2-L passes under H_shift whenever A1 already extrapolates.**
- Under H_shift (A2 = A1 translated by +4), A2's accuracy at D 13–16 equals A1's at D 9–12. The draft's 0.02 assumes A1's DEPTH-L ≈ 0.55.
- But a REACH hop is the same rule at every depth, and SLVA is LENGTH-FREE. So P(A1 DEPTH-L ≥ 0.8 | VB-C1-L) is plausibly about 0.25, and then SD-C2-L passes without any self-teaching effect.
- *Fix:*
  - SD-C2-L := SD-SELF-L ∧ DEPTH-BEYOND-SELF-L ∧ **MARGIN-GROWS-L** (beat the scorer's +4-translated A1 curve by ≥ 0.10 pooled over D 13–16) ∧ **beat A2-CTRL** at D 13–16 by ≥ 0.10.
  - A2-CTRL becomes required (D-10).
  - The same applies to ABSORB-L's "no effect" row.

**F5-2 (MAJOR) CurGate's and NO-ENTRY's "answer CE ≤ 0.35" passes at chance if the CE is the decoder string's per-token mean.**
- "### o_R." scores ' name' + '.' as 3–5 tokens. A model that copies both options perfectly but tosses a coin between them has a per-token mean CE of **0.173** (range 0.139–0.231 over CON38 pairs) [MEASURED with tiktoken]. Only 1% of option pairs share a first token.
- So the CurGate advances 4 → 6 → 8 at chance, and NO-ENTRY can never fire on a model that has learned only to copy.
- The draft's "binary CE; chance ln 2" states the intent but gives no definition. vbank's answer_ce is a single-token CE.
- *Fix:* gate on the **two-candidate CE**: −log softmax over the two candidates' sequence log-probabilities, chance ln 2. Add an init fixture asserting ≈ 0.69, and log it as `plain_cand_ce`.

**F5-3 (MINOR) LIT-C2-L is about 0 under complete success.**
- At D 15–16, K′ 16 ≥ D already reaches the option under one hop per pass, so acc_k[18] − acc_k[16] ≈ 0 unless STOP-LAG > 0.
- Literal C2 (needing more than the trained 16 passes) needs D ≥ 17, which does not fit 1,024 positions at SYM's size law.
- *Fix:* say so in the registration. A LIT-C2-L fail is UNINFORMATIVE here, not a C2 negative.

**F5-4 (MINOR)** P(SERIAL-ONSET-L | no effect) ≤ 0.05 is fine. But the probe grid {…, 4, 6, 8} maps ON(5) → 6 and ON(7) → 8. Add t = 5 and t = 7 to the probe set; it is cheap, and the slope then reads hops, not grid steps.

### 5.3 Censoring holes

- **(a)** A2's pool exhaustion is certain as drafted (F3-1).
- **(b)** PROBE-INVALID tests only the final state (PROBE(12)). Add the PLANT control (F1-1a); a failed plant = PROBE-INVALID = not met.
- **(c)** A G-TIME admission failure (likely as drafted, F6-3). Pre-state that no launch = that cell's readings are not met (never reached = failure), and that it goes to Audit / the user. It must not be read as "dead run" or as "pending".
- **(d)** An empty C1 band (if D*_fresh ≥ 8, F2-2) = VB-C1-L not met. Pre-state it now.
- **(e)** A0-1's CUE-RELIANT compares only the tie and pro strata, and those strata differ in base-graph structure.
  - The informative stratum is **anti** (n 78): acc(anti) ≤ 0.40 would be a direct cue-use signature (P under no reliance at true acc 0.95 ≈ 0).
  - Add it, plus the NN-rendered accuracy against p8_na at the same depth, as the cleaner contrast. All descriptive.
- **(f)** The UNREADABLE thresholds (≥ 100 clean pairs) are fine for ProsQA. For GSM8K class A (with clean pairs needing both answers right at about 0.27 accuracy), project the pair count from B0-3's smoke before launch, so SERIAL-STEP-GSM is not unreadable by construction.

### 5.4 Wording risks

- **"shortcut-exchangeable by construction".** Only person-agnostic cues are excluded by construction. The root-aware backward chase is exact (H-ROOT 1.000). Say "person-agnostic cues excluded by construction; H-ROOT gated" (after F2-1), and never "no shortcut".
- **"graph search"** → "serial latent reachability", unless decoys plus FD-L / PLANT exclude the backward chase and precompute.
- **C1 on a two-option task.** C1's "drops to chance" means 0.5 here, and the deep-band floor of a direction-blind search is 0.56–0.65 (F2-3). Quote PROBE(1) against both.
  - "C1 on problems that exceed one forward pass" needs the measured D*_fresh (F2-2). Without it, VB-C1-L is "serial use within D 5–8", not C1.
- **VB-C1-GSM "C4-legal as distillation"** needs Audit's ruling on DECOMP's sub-step values (F4-1).
- **RECOMPUTE-L / RECOMPUTES-FROM-INPUT-L** must not be worded "recomputes" on PERSON-SWAP's own share (F1-2), nor on arm (ii) (PROOFS_LANG's history read).
- **"A null is informative: PASS1-PRECOMPUTE"** holds only with the probe-based interim (F1-5). As drafted, the B1 null would mostly be "not decided".


## 6. Item 6: cost, caps, memory (T² K)

[MEASURED tokens with tiktoken on sym_proto draws; the rest DERIVED in rt_lang/mem_cost.py (→ mem_cost.json) plus an inline re-run at 2.5-token names. ASSUMED: bf16 autocast (vbank.py line 630), SDPA in the prelude (no T² save), a saved bank mask + probs at 2–4 B per entry, 38·d bytes of saved activations per token-layer, 20 TFLOP/s sustained on a 3090, the efficiency scaled to D3-T's measured 0.42 s/step, and $0.21/h.]

**F6-1 (MAJOR) T is about 1.2–1.5× the draft's assumption.**
- The draft assumes 1.6 tokens per word, giving 450 at D 8 and 740 at D 16. The measured figures are 2.3 tokens per word with the prototype's names, and about 1.8 with 2.5-token names (PROOFS_LANG agrees).
- D 8 is 533–688 mean, p95 868. Uncapped, D 12 is 790–1,033 mean, so the T ≤ 1,000 invariant fails at D 12 without the cap. D 16 is 1,040–1,361.
- The 2D + 6 cap must apply from D 9, and D 13–16 fits 1,024 positions only with short names plus the cap.
- F2-1's decoys add 25–50%, which forces a trade: either a thinner distractor law, or a D range ending at 14.
- Fix: G-L0 measures T on the real NONCE-300 before any band is fixed, and the D range of the untouched set is set from it.

**F6-2 (MAJOR) bs 64 does not fit a 24 GB 3090 for A2, nor for A1's worst batches.**

| shape | saved memory at bs 64 (GB) | est. s/step |
|---|---|---|
| A1 mean (D 8, T 533–688, \|S\| 50, K 12) | 14.4–20.5 | 1.2–1.4 |
| A1 forced worst (T 868–1,024, \|S\| 75) | 25.6–36.6 | 2.0–2.2 |
| A2 self mean (D 12, T 800–1,000, \|S\| 64, K 16) | 26–39 | 2.0–2.2 |
| A2 forced worst (T 1,000, \|S\| 80, K 16) | 36–49 | 2.6 |
| B1 GSM8K (T 77–194, \|S\| 12, K 16) | 3.4–5.8 | launch floor, 0.55–0.7 |

- Micro-batching (×2) or `--ckpt_passes` is needed for A2 always, and for A1's long-T batches. §214-2 hit the same wall on nested (micro 1 OOM at K 20).
- Fix: pin micro / ckpt_passes per batch type in the args, and mem_check the largest shapes (A0-8 already does) **before** fixing the cost.

**F6-3 (MAJOR) Both A cells are likely to fail their own G-TIME admission as drafted.**
- Forced-worst projections: A1 ≈ **$2.8–3.1** against the ≤ $2.5 admission, and A2 ≈ **$1.8** (plus micro overhead) against ≤ $1.5.
- The likely outcome is a no-launch and a round-trip to the user, as in §214-2 / 214-3.
- Fix: pre-state the admission against a projection from the realised T / K law (the 214-3 form), or raise the per-cell admissions now: A1 ≤ $3.5, A2 ≤ $2.5. The family cap is unaffected.

**F6-4 (MAJOR) The pool labelling (15–37 local fp32 GPU-hours) and the corrected pool size (F3-1) are not budgeted.** Fix: run it on the host in bf16 (+$1–2), or schedule it on the local cards.

**Family total, realistic.**

| item | cost |
|---|---|
| A1 | $1.3–2.8 |
| A2 | $1.1–2.0 |
| A2-CTRL (now required) | $0.6–1.0 |
| DIRECT-SYM (new, required) | $0.4–0.8 |
| B1 (if gated in) | $3.9–6.6 |
| host labelling (optional) | $1–2 |
| bad hosts | +10% |
| **total** | **≈ $8.5–16**, under the $25 cap |

- The cap is realistic. What is unrealistic is the per-cell admissions and the drafted central (~$6).
- Stop rules (GUARD_STOP per cell and per family; B1's interim) are adequate once F1-5's probe-based GO replaces the trained-coda GO.


## 7. Recommendations on the draft's §10 decisions

| # | draft recommendation | red-team recommendation |
|---|---|---|
| D-1 | SYM | **SYM, amended.** Add ancestral decoys with the path-node child law, tuned so that H-ROOT < 0.6, and add H-ROOT to the gated panel (F2-1). Build NONCE-300 with distinct final BPE tokens (F2-4). Use single-token person names (F2-5). Apply the 2D + 6 cap from D 9 and set the band range from G-L0's measured T (F6-1). |
| D-1b | A0-1 to Audit | **Yes**, descriptive. Add acc(anti) ≤ 0.40 (n 78) and the NN-rendered contrast, which are cleaner than tie vs pro (§5.3e). Run it on §191's original eval card class, dtype and bs. |
| D-2 | decoder through blocks 0–9 alone | Agree. G-L1b reads the drift. Assert decoder positions T + L_dec ≤ 1,024. |
| D-3 | concept slots | Agree, but the slot token must identify the concept (F2-4). Mention slots only if the A0-4 binding smoke fails. |
| D-4 | causal current-version attention | **Agree, strongly** (F1-6): it is the main brake on within-pass pointer doubling. Consider slot-diagonal as a stricter variant. |
| D-5 | choose L-ROLL / L-RR on GATE | Agree. Also fix L-RR's OOD re-roots (A1 law r ~ U{1..8}, or chunked re-roots; F2-6), replace reject-on-any-false-mark with SOUND-CLOSURE (F3-2), and store both chains' per-item KEEP decisions. |
| D-6 | K_i ~ U{m_bin..16}, headline K′ 14 | Agree on the headline. **Remove every trained-coda K′ < m_bin reading from decisions** (F1-5). |
| D-7 | R_ν on, R_x off | Acceptable. Strike "one variable vs CODI-20": B1 differs in recipe, compute and supervision (F4-4). |
| D-8 | defer B2 | Agree. |
| D-9 | 120k matched, with an interim look | Agree on 120k. **Replace B0-5's GO with a fresh-probe rule** (PROBE(1) gain ≥ 0.05 on Aug-valid plus FD-MLP), read from the off-host 24k checkpoint while the host continues (F1-5 / F4-2). |
| D-10 | A2-CTRL yes; A-TXT optional | **A2-CTRL required** (SD-C2-L attribution, F5-1). **Add DIRECT-SYM as required** (the one-pass boundary, F2-2). A-TXT optional. |
| D-11 | binary bars | Keep 0.35 / 0.80 / band onset. Apply the 0.80 to **PREMISE-SWAP × (ii′)** plus the slot-local arm (F1-1 / F1-2). Add t = 5 and 7 to the probe grid (F5-4). |
| D-12 | REACH supervised, not C4 | Agree. Also send **DECOMP's C4 status** to Audit before registration (F4-1). |
| D-13 | B1 after A1 is ¬dead ∧ ¬NO-ENTRY | **Stricter: launch B1 only if A1 reads VB-C1-L** (with the F1 fixes). Otherwise defer B1 to the user. This saves about $4–6 at P ≈ 0.9, and follows the "small tasks before GSM8K" rule (F4-5). |
| D-14 | PROOFS + REDTEAM before registration | **Yes**, plus three more items before registration. Re-prove §4.6 for PREMISE-SWAP × (ii′). Fixture-check the scorer on two counter-models: a PaR-C fixture (hidden carry, which must fail VB-C1-L via FD-L / PLANT) and an ALL-PERSONS fixture (which must pass under PREMISE-SWAP). Assert the pool arithmetic (F3-1). |

**Borrowed-technique preconditions that fail as transplanted** (the project rule):
- D3-T's CurGate threshold assumed a single-token answer CE (F5-2).
- §215's pool size assumed 110k per n (F3-1).
- §215's re-rooted chain assumed in-distribution re-roots (F2-6).
- nested PROBE-GAIN's power assumed chance 1/11 and a 0.5 bar. On a binary task with a hidden-code route, it needs a positive control (F1-1).

