# PROOFS_LANG: proof-and-arithmetic check of LANG-VB draft v0

Checker: Opus proof subagent for Theory, started 2026-10-18 12:48 EDT (`date`). $0, CPU (OMP 1-2 threads, `python -B`),
repo read-only. Target: D:/Latent/docs/research_lang_2026-10-18/LANG_draft.md (+ scripts/). My scripts:
research7/proofs_lang/*.py, JSON beside them. Verdicts: PASS / FAIL / CORRECTED (right value + minimal fix text).

(Written incrementally; sections appear as each check finishes. All sections complete 13:10 EDT 10-18.)

## 0. Summary (70 tagged / numeric claims checked): 46 PASS, 21 CORRECTED, 3 FAIL (fixed in text)
- **Anchor cue (priority 1) reproduces independently, number for number** (anchor proximity 0.500 / 0.674 / 0.896-0.926
  test, 0.897-0.945 train; option level 0.500; strata 2,630 / 511 / 78). Corrections: p8_na also fails at depth 3
  (option-level parents' in-degree 0.63-0.65); NN's anchor residual is ≈ 0.63 (3-step bases) / ≈ 0.58 (4-step bases),
  not "0.61-0.67 in every band"; **§191 A beats the best person-agnostic cue lookup at every depth 3-10** (cue ceiling
  0.58-0.90 vs 0.955-0.998), so the cue cannot be NAME-MEM's mechanism (A0-1 measures a contribution only).
- **SYM exchangeability: PASS in a stronger form** (the target is independent of the person-masked item). **FAIL:** the
  directed BIDIR is ½ / 1 by construction, and an **undirected** bidirectional search leaks 0.60-0.63 at D 8-12 (also with
  the spec's distractor-only kernel). Fix: cross edges only between the persons' extra direct-statement subtrees
  (measured BIDIR-U = 0.500 exactly; invariants 0 failures) and BIDIR-U in the gate.
- **Lemma S-L: PASS** (SOUND never over-reaches or mislabels; it accepts incomplete / late chains). **FAIL:** the
  zero-noise corollary contradicts §2.1's HOLD-after-τ (concepts beyond D in 6-22 % of items; kept counterexample).
  Fix: unheld REACH, or the direct-subtree kernel (REACH saturates at D).
- **FAIL:** §4.6(c) — "own" also arises from reading older versions (history), so RECOMPUTE-L mislabels; add a BANK arm.
  Detectors are one-sided (probe decodability); FD(k, k) is missing from the field contract.
- GSM8K statistics: all reproduce; "C4-legal" needs a qualifier (≈ 13.5 % of step targets are DECOMP sub-steps).
- OC: P(SD-C2-L) ≈ 0.02 by the draft's own chain (not 0.05); P(>= 1) 0.20-0.26. Cost: required $5.4-10.6 (not
  5.3-10.6); tokens ≈ 1.8 / word (D 8 ≈ 533). No decision changes.

## 1. p8_na anchor-level cue (§1.2) — independent re-implementation

**Method** (proofs_lang/anchor_check.py, nn_check.py; JSON ac_*.json, nn_*.json). Graph re-parsed from the premise
*text* (cross-checked equal to x["edges"] on every item); §57 proximity = min BFS distance over all persons, target =
the farther option, tie 0.5; the five other §59 heuristics in their stated directions plus direction-free. Anchors found
two independent ways (edge-list position after make_long / neutralise; text walk-back from c_1 / d_1): they agree on
3,638 / 3,638 test and 10,317 / 10,317 train ext >= 1 items, 0 ambiguous. No heur_panel.py code used.

| check | draft | mine | verdict |
|---|---|---|---|
| §59 validation, prosqa_test (prox / parents / mentions / first / last / pindeg) | 0.911 / 0.662 / 0.662 / 0.590 / 0.554 / 0.515 | 0.911 / 0.662 / 0.662 / 0.590 / 0.554 / 0.515 (parents, mentions, first, last read in the "target = fewer / later / earlier" direction: §59's table is effectively direction-free) | PASS |
| §59 validation, p8_na_test option level | 0.50 everywhere | 0.500 / 0.500 / 0.500 / 0.507 / 0.510 / 0.500 pooled | PASS |
| option-level proximity, every depth 3-14 | 0.500 | 0.500 at every depth (test and train) | PASS |
| **anchor-level proximity**, depth 4..14 (test) | 0.500 / 0.674 / 0.896 ×4 / 0.897 / 0.902 / 0.905 / 0.904 / 0.926 | identical to 3 dp | **PASS** |
| anchor mentions / parents (test) | 0.572 ... 0.735 | identical | PASS |
| anchor proximity, train depth >= 6 | 0.897-0.945 | 0.897 / 0.908 / 0.904 / 0.902 / 0.945 (depth 6..10) | PASS |
| A0-1 strata (m >= 2, depth >= 5) | pro 2,630 / tie 511 / anti 78 | 2,630 / 511 / 78 (train: 7,204 / 1,246 / 214) | PASS |
| mechanism (m = 1 anchors neutralised, m >= 2 untouched) [DERIVED] | — | re-derived from neutralise(): for m = 1 the L = 2 candidates are the option's non-person parents = the anchors, so "P is a <anchor>" puts both anchors at person-distance 1; for m >= 2 the parent is c_{m-1} / d_{m-1} | PASS |
| "(i) p8_na fails a level-wise panel at every depth >= 5" | depth >= 5 | also fails at **depth 3, option level**: parents' in-degree 0.628 test / 0.648 train (ext-0 items keep prosqa_na's 0.62 residual; §59's pooled 0.50 hides it); depth 4 passes (max 0.572 test / 0.564 train) | **CORRECTED** — fix: "fails at depth 3 (option-level parents' in-degree 0.63-0.65) and at every depth >= 5 (anchor proximity 0.674 at 5, >= 0.896 at >= 6); depth 4 alone passes" |
| §3.1 / D-1 "anchor proximity 0.86-0.93" | 0.86-0.93 | those are nn_regen.py --mode p8na regeneration values (one m per depth: 0.861 / 0.929), not p8_na_test (0.674 at 5; 0.896-0.926 at >= 6) | CORRECTED (cite the file's values) |
| NN anchor parents' in-degree "0.606-0.668 in every band" | 0.606-0.668 | same construction, my panel: seed 0 0.606-0.657, seed 1 **0.588-0.641** (bands 7, 8 at 0.591 / 0.588 pass the gate); n ≈ 200 per band, SE ≈ 0.03. Train-base regeneration (seed 2, n ≈ 7k per band): **0.634 on 3-step bases (D 5, 9), 0.575 / 0.581 on 4-step bases (D 6, 10)** | **CORRECTED** — fix: "NN keeps §59's in-degree residual at the anchors: ≈ 0.63 on 3-step bases, ≈ 0.58 on 4-step bases (n ≈ 7k per band); it fails the 0.6 gate on 3-step-base bands and is marginal (0.58) on 4-step-base bands." D-1 (use SYM) is unaffected |

**Does the anchor cue explain §191 A? (consequence (ii)) — CORRECTED.** §191 A's acc_by_depth (eval.json) is
0.955 / 0.978 / 0.988 / 0.998 / 0.990 / 0.993 / 0.979 / 0.969 at depth 3..10. At depth 3-4 there is no anchor cue
(depth 3 = options only, best cue 0.628; depth 4 = ext 0 / ext 1, anchor cue neutralised, best 0.57), and at depth >= 6
the anchor-proximity ceiling is 0.896-0.905, below A's 0.97-0.998. The combined person-agnostic cue ceiling is in
§1c below. Fix text for (ii): "§191 A's accuracy exceeds the anchor-cue ceiling at every depth 3-10 (no cue >= 0.63 at
depth 3-4; proximity ceiling <= 0.905 at 6-12), so the cue cannot be its whole mechanism; it may still contribute at
depth >= 6, which A0-1's pro / tie / anti split measures." A0-1 stays useful; the framing "not distinguishable without
it" overstates.

**Verdict on priority 1: the anchor-cue finding reproduces independently, number for number.**

### 1c. Combined person-agnostic cue ceiling vs §191 A (proofs_lang/cue_ceiling.py → cue_ceiling.json)
Bayes-optimal lookup over the sign vector of all 12 option- + anchor-level §59 cues plus the visible chain length m,
fitted on p8_na_train, scored on p8_na_test (in-sample test lookup in brackets = an over-fitted upper bound):

| depth | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| cue ceiling (train-fit) | 0.628 | 0.584 | 0.761 | 0.901 | 0.895 | 0.889 | 0.900 | 0.720* |
| (in-sample, over-fit) | (0.631) | (0.642) | (0.833) | (0.948) | (0.948) | (0.957) | (0.957) | (0.956) |
| §191 A acc_by_depth | 0.955 | 0.978 | 0.988 | 0.998 | 0.990 | 0.993 | 0.979 | 0.969 |

(*depth 10 mixes an m absent from train.) §191 A beats the out-of-sample cue ceiling by 0.07-0.40 at every depth 3-10
and the over-fit bound at 3-8. **So "chain-chase + anchor cue" cannot account for §191's NAME-MEM 0.950;** the draft's
(ii) and D-1b should say the cue may contribute at depth >= 6 (A0-1 measures how much), not that §191 is
"not distinguishable" from it. This changes no registered reading; it lowers A0-1's stakes.

## 2. ProsQA-SYM exchangeability (§3.1) and the prototype panel

### 2a. The theorem, stated precisely — PASS (in a stronger form), with two implementation conditions
Let ω = (W_1, W_2, C_12, C_21, extras, person names, concept names, statement order π, option coin b); the question asks
W_1's person. Let Z = (premise statement sequence, displayed option pair (o1, o2)) = the item **minus the question's
person token**, and Y = 1[o1 is the target]. Define σ: W_1 <-> W_2, C_12 <-> C_21, swap the two person names, keep each
concept's name on its node, keep π as an order on statement identities, flip b. Then Z(σω) = Z(ω) and Y(σω) = 1 - Y(ω).
If P∘σ = P, then P(Z ∈ A, Y = 1) = P(Z ∈ A, Y = 0) for every A, i.e. **Y is independent of Z and P(Y = 1 | Z) = ½
a.s.** Hence every predictor h(Z, U) (U independent noise) has accuracy exactly ½ — not only the six §59 heuristics but
any person-agnostic function, however deep, including ones that use person names in the premises (only *which* person
is asked is excluded). Path-level cells (a_k^R vs a_k^P) are covered by the same σ. [PROVED]
- **"Person-agnostic" must be defined as "a function of Z"** (premises + displayed options), not "ignores identity": the
  draft's wording is fine but should state this.
- **σ is measure-preserving iff every component law is symmetric.** The prototype satisfies W_1, W_2 i.i.d. (same
  world() call), the same cross-edge kernel in both directions, rng.sample names, a uniform shuffle and a fair coin. Two
  implementation conditions the spec must add: (i) the per-world 2D + 6 cap must thin each world by the same rule
  independently (thinning the union in a fixed world order breaks σ); (ii) the §3.3 contamination redraw uses a
  signature on "generator-internal ids", which are assigned W_R first, so the rejection event is not σ-invariant —
  compute the signature on a role-symmetric canonical form (e.g. sort the two worlds' canonical strings) or reject on
  both orientations. (The T <= 1000 rejection is σ-invariant up to the person-name token length; use 1-token ENT17 names.)

### 2b. Root-aware heuristics are NOT covered, and the draft's BIDIR does not measure them — CORRECTED
- **Directed BIDIR(r) is uninformative by construction.** o_P ∉ Reach(R) means no directed R-ball ever meets o_P's
  backward ball, and dist(R, o_R) = D means o_R is met iff 2r >= D. So BIDIR(r) ≡ 1 for 2r >= D and ≡ ½ for 2r < D on
  *every* item, whatever the cross edges do (my fresh draws: exactly 0.5 / 1.0 everywhere). "BIDIR(2) is exact only at
  D <= 4, as predicted: cross edges ... add no shortcut" is true but BIDIR cannot show it, and "its accuracy at 2r < D
  measures how much a shallow root-aware search gets from SYM's cross edges" is false (it is ½ identically).
- **The undirected version leaks.** uBIDIR(r) (undirected r-balls of R and of each option intersect; guess on tie), on the
  prototype kernel (cross-edge source = ANY node of the other world, incl. path nodes and the option; target = a leaf),
  n = 1,000 per D (sv_any.json): at the largest r with 2r < D it reads **0.4585 (D 6, r 2), 0.402 (D 8, r 3), 0.3935
  (D 10, r 4), 0.3725 (D 12, r 5)**, i.e. direction-free **0.54 / 0.60 / 0.61 / 0.63** — "the option inside R's
  undirected ball is the foil". Mechanism: a cross edge from a deep foil-world node into a shallow root-world leaf puts
  o_P at undirected distance d_leaf + 1 + (D - depth) from R, while o_R stays ≈ D away (ud(o_P) <= D in 31-47 % of items).
  The draft's spec kernel ("between distractor nodes only") still leaks: 0.56 / 0.60 / 0.62 at D 8 / 10 / 12 (sv_distractor.json).
- **A kernel that removes it: cross edges between the persons' extra direct-statement subtrees** (depth <= 2 in both
  worlds; those nodes join their world only through the person). Then every undirected R -> o_P path passes the foil
  person P, so ud(R, o_P) >= D + 3 > D >= ud(R, o_R), and the r-balls of R and of the options are disjoint for 2r < D.
  Measured (sv_direct.json, D 6 / 8 / 10 / 12 / 16, n = 1,000 each): **uBIDIR = 0.500 exactly for every 2r < D**,
  person-agnostic panel max 0.528-0.543, 0 invariant failures, σ-symmetry kept. It also makes REACH saturate at D (§3 below).
- **Fix text (§3.1 "Not covered" and the C1-band line):** "Root-aware partial search is reported as undirected BIDIR-U(r)
  (directed BIDIR is ½ / 1 by construction). SYM's cross edges join the persons' extra direct-statement subtrees only, so
  BIDIR-U(r) = ½ for 2r < D [DERIVED; MEASURED proofs_lang/sym_variant.py]. C1 bands D >= 5 exceed 2-hop-per-side
  search." Add BIDIR-U to the LEVEL-PANEL gate (a band with BIDIR-U(⌊(D-1)/2⌋) >= 0.6 direction-free is not used).

### 2c. Prototype panel and invariants on fresh draws — PASS (with two minor corrections)
proofs_lang/sym_check.py, seed 7, n = 1,000 per D (sym_fresh_s7.json); generator = sym_proto.make, panel = mine.

| D | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| invariant failures (dist(R, a_k) = k ∀k, dist(P, a_k) = k ∀k, o_P ∉ Reach(R), o_R ∉ Reach(P), extras reach no option, all mentioned) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| max person-agnostic cell (7 heuristics × D levels, dir-free) | 0.527 | 0.533 | 0.525 | 0.530 | 0.529 | 0.534 | 0.532 | 0.538 | 0.533 |
| items with a concept at dist > D from R | 0.218 | 0.168 | 0.166 | 0.140 | 0.096 | 0.086 | 0.089 | 0.071 | 0.059 |
| GPT-2 tokens, prototype names (mean / max of 200) | 272 / 433 | 356 / 511 | 432 / 675 | 529 / 761 | 681 / 1039 | 850 / 1403 | 1004 / 1316 | 1180 / 1628 | 1366 / 1938 |

- Unique answer, exact depth (no shortcut to any path node), symmetric invariants: **PASS** (0 / 9,000 here, 0 / 17,000
  over all my variant runs).
- "max ... 0.518-0.552" and "consistent with E = ½": PASS (mine 0.525-0.538). "Every band passes the 0.6 gate with a
  margin of >= 0.05": CORRECTED — the draft's own max 0.552 leaves 0.048; say ">= 0.048".
- **Size law / tokens — CORRECTED.** Re-scored to concept names of 2.5 GPT-2 tokens (NONCE-300's 2-3; sv_*.json T_tpn2.5,
  uncapped): mean (max) **399 (695) at D 6, 533 (774) at D 8, 659 (993) at D 10, 789 (1102) at D 12, 1042 (1368) at
  D 16**, i.e. ≈ 1.8 tokens per word, not 1.6. Fix: "about 530 tokens at D 8 (uncapped); D 12 needs the cap too (max
  1102 > 1000); at D 16 the 2D + 6 cap (≈ ×0.8 concepts) gives ≈ 830 mean." §2.3 / cost_model's Tq 450 at D 8 → 533 (see §6).
- **Concept-count bounds — CORRECTED (minor).** With the per-world cap 2D + 6 and up to two extra persons with 2-4-node
  trees, |S| <= 2(2D + 6) + 8 = 4D + 20 = **84 at D 16** (the draft says <= 76 in §2.3 and <= 82 in §4.2; cost_model uses 82).

## 3. SOUND verifier, Lemma S-L, and "SOUND-ONLY R2 ⇒ zero supervised noise"

**Lemma S-L — PASS.** Let R_t = {c : dist(R, c) <= t}, M_0 = ∅. P1 gives M_1 ⊆ direct(R) = R_1. If M_{t-1} ⊆ R_{t-1},
each c ∈ M_t \ M_{t-1} has a parent a ∈ M_{t-1} ⊆ R_{t-1} (so dist(c) <= t) or is direct (dist 1); old marks are in
R_{t-1} ⊆ R_t. So M_t ⊆ R_t for all t. The proof uses only "every new mark has a parent among the previous marks"; it
holds for any directed graph — multiple shortest paths, cycles (the prototype has 2-cycles between leaves via cross edges)
and distractor edges change nothing. With o_P ∉ Reach(R): the first option in M_s is o_R (π_label = 1, φ = 0), and
o_R ∈ R_s gives s >= D. Fuzz (proofs_lang/sound_check.py, 42,000 random chains on SYM items, half with arbitrary unsound
marks injected; 20,905 kept): 0 wrong labels, 0 with s < D, 0 subset violations.

**Is a SOUND-kept chain the gold reach set? — No (answer to priority 3).** SOUND checks soundness only, so it accepts
any *sub*-chain M_t ⊊ R_t. Counterexample (cex1, a real D = 5 SYM item): the gold-path-only chain M_t = {a_1..a_t} is
KEPT (s = 5, label o_R) while R_2 has 5 concepts and M_2 has 2 (3 omitted distractors). SOUND can never accept an
**over-reach** state (a concept at dist > t marked at t), nor a wrong label, under SYM's invariant. It can accept wrong
states only by omission and by late marking (dist < first-mark pass). The draft says this (coverage distortion, G7); PASS
as a statement, but the question "can SOUND accept a wrong state" is answered "yes, by omission / lateness, never by
over-reach" and that sentence should be in §3.1.

**Corollary "every self target equals the gold target ⇒ supervised label noise exactly 0" — CORRECTED (convention bug).**
It holds iff the gold ProsQA target is *unheld* REACH (E_t = R_t for all t). The draft's loss (§2.1) says "R2 ... HOLD
after τ_i" with τ_i = D, i.e. gold E_t = R_D for t > D, and §4.3(a) relies on that ("last changed pass <= 8"). On SYM the
two differ: concepts at dist D + 1 exist in **5.9-21.8 %** of prototype items (cross edges out of depth-D nodes; table
§2c). Counterexample (cex2, real D = 3 item): chain M_t = R_t \ {o_R} for t = 1..4, M_5 = R_4 is KEPT (s = 5, label o_R)
and marks a dist-4 concept IN at pass 4; HOLD-gold has it OUT at pass 4 (unheld gold: IN). So with HOLD the self R2
targets are *not* gold-equal whenever s > D and a dist-(D+1..s) concept is marked. Two minimal fixes (either suffices):
- (a) **Define ProsQA gold as unheld REACH** (E_t = R_t for every t <= K_i; drop "HOLD after τ" for ProsQA) and fix §4.3(a)
  to "last changed pass <= D + 1"; or
- (b) **(recommended) adopt §2b's direct-subtree cross-edge kernel**: then R_t = R_D for all t >= D whenever D >= 4
  (proofs_lang/direct_maxdist.json: share of items with a concept beyond D = 0.000 at D 4-16; 0.052 at D 3), so HOLD is
  exact, §4.3(a) holds as written and the Corollary holds as written. (At D 3 restrict cross targets to direct-subtree
  leaves.) Under (b) the corollary's "noise exactly 0" is PROVED.
- Also state that the comparison is against the gold of the *same* item at the *same* pass with the pool record's
  marks; the HOLD window for self items is [s, K_i] and must use the same convention.

**Other SOUND details.**
- P1 is implied by P2 with M_0 = ∅ (only the "R is a c" clause can fire at t = 1); harmless.
- L-RR re-renders the item ("R is a f" for the frontier). SOUND must parse the **original** item's statements, never the
  re-rendered text (else frontier concepts become "direct" and P2's second clause is weakened). Add to the spec.
- π_prog = P / (P + φ(1 - P)) (§4.5 GSM8K) — PASS (Bayes, assuming every correct program passes ARITH / PROV / CONTIG /
  ANSWER-MATCH).

## 4. One-pass precompute (§4): what is proved, what is assumed, what the detectors can see

| claim | tag | verdict |
|---|---|---|
| L1-L: "any function of V_t is computed by a layered network of depth <= P + d·t" | PROVED | **CORRECTED (minor).** The edge enumeration omits SLVA's gather: U_t = Σ_s α_{j,s} N(V_s)[j] is one softmax-attention sublayer (query from N(V_{t-1}), keys W_k LN_k(V_s)), no MLP, per pass (boxes/slva.py gather). Fix: "V_t is computed at depth <= P + d·t blocks plus t SLVA attention sublayers; the answer after K′ passes at <= P + d·K′ + 2 blocks (+K′ sublayers)". Also "any function of V_t" should read "V_t". Immaterial to the capacity argument. |
| L(0) = 0: reads of V_0 add no depth; decoder string question-free; slot subset only removes edges | PROVED | PASS |
| finite functions ⇒ no unconditional pass lower bound (§4.2) | PROVED | PASS ("<= 82 concepts" → 84, §2c) |
| CAP-aff-L (c ≈ 1, c₀ ≈ 1-3) | ASSUMED | Correctly tagged. **The cited evidence does not measure one-pass capacity**: §168 DIRECT memorised a fixed 12k set (chance even at depth 3 = a training failure); §191 A's K′ 0 / 1 were never trained to answer at K′ <= 1. Fix: say so; the in-model PROBE(1) is the only measurement. |
| over-reach penalised in decodable form (OUT / VOID targets) | DERIVED | PASS |
| precompute-and-reveal is zero-loss given capacity | PROVED, given capacity | PASS (the loss reads only R_P / R_G outputs and the coda; any mechanism with the same decodable outputs has the same loss) |
| recompute from V_0 <= c₀ levels per pass | PROVED UNDER CAP-aff-L | PASS (restates CAP-aff-L for m = 1) |
| §4.6 (a) V_0 premise / slot rows bitwise equal in donor and recipient | PROVED | PASS (causal prelude, question last, slots at first mentions, all in the premises) |
| §4.6 (b) hybrid answer o_P under one hop per pass | PROVED | PASS |
| §4.6 (c) "at t >= 3 an 'own' answer requires > 1 hop per pass from V_0 or a non-serial route" | PROVED | **CORRECTED.** Arm (ii) replaces only V_t; the recipient's V_1..V_{t-1} stay in the bank and in SLVA's source. A pass map that reads V_{t-1} (R_{t-1}(R)) needs D - t + 1 <= D + 2 - t passes: "own" without over-reach and fully serial (history use). So own ⇒ over-reach ∨ **history read of V_{<t}** ∨ non-serial. RECOMPUTE-L (own_ii > 0.3) and RECOMPUTES-FROM-INPUT-L mislabel history reading. Fix: add arm (ii′) splicing donor V_1..V_t (the §213 BANK arm); call (ii)-own with (ii′)-follow HISTORY-READ, and define RECOMPUTE-L on (ii′)'s own share. |
| §4.6 (d) follow + own = 1; random disruption ≈ 0.5; bar 0.8 | DERIVED | PASS (P(follow >= 0.8 \| 0.5, n >= 100) < 1e-9) |
| VALUE-SWAP (iv) uninformative on ProsQA | DERIVED | PASS |
| §4.3 (a) "last changed pass <= 8", (b) clock and reach-saturation rules both match all targets | PROVED | **CORRECTED** — true under HOLD; under unheld REACH the last change is <= D + 1 and "expand until an option is IN" no longer matches (it misses dist-(D+1) changes). Same convention bug as §3; fix (b) of §3 makes both hold. H_shift irreducibility itself: PASS. |

**Does the bounded core + per-pass target constrain capacity? (priority 4).** Precisely:
1. L1-L bounds *depth* from the question to V_t at 6 + 4t blocks. At t = 1 that is 10 of GPT-2's 12 blocks, at full
   width (|S| slot rows + T bank rows). So nothing in the architecture stops pass 1 from computing what a 10-block GPT-2
   computes from the question; the only barrier to one-pass precompute is CAP-aff-L [ASSUMED], i.e. that such a
   network does not solve D >= 5 search / m >= 3 GSM8K. Proved: depth bound. Assumed: that depth binds.
2. R2 pins only the *decodable* (linear R_P / R_G) state to one hop / one op per pass. Hidden content is unconstrained,
   so reveal and hidden over-reach (pointer doubling) are loss-neutral. Proved.
3. On ProsQA a clock-free reveal must either re-do the hop on the decodable state (then it *is* one hop per pass) or
   key on stored hidden distances ("reveal dist = max revealed + 1"), which is content-keyed and loss-neutral.

**What each pre-stated detector can and cannot distinguish.**
- **PROBE(t)** (fresh 2-block coda + bank, 2k steps): one-sided. High PROBE(1) ⇒ answer present after pass 1
  (precompute). Low PROBE(1) does not exclude a hidden code the probe cannot decode in 2k steps. PROBE-VALID (PROBE(12)
  >= ACC - 0.05) validates reading the *final* state only. SERIAL-ONSET-L inherits the same one-sidedness.
- **FUTURE-DECODE(t, k)** (GSM8K, *linear*): one-sided. FD(1, k) ≈ FD(k, k) ⇒ q_k linearly present early; low FD does not
  exclude a nonlinear precompute that the 4-layer core can still extract. **Field-contract gap (CORRECTED):**
  NOT-ONE-PASS-CHAIN reads FD(k, k), but the registered `future {t: {k}}` has t ∈ {0, 1, 2, 4} and k >= t + 2, so t = k
  never exists. Fix: add t = k entries (or t ∈ {0..12}, k >= t), or define FD(k, k) from stepcap. Also, FD(k, k) < 0.3
  makes NOT-ONE-PASS-CHAIN pass vacuously; make it UNREADABLE instead.
- **VALUE-SWAP (i), slot t only** (GSM8K): follow ⇒ later steps read slot t's content; own ⇒ later values come from
  elsewhere (precomputed in other slots, or recomputed). It cannot separate "reads the decodable q_t" from "reads hidden
  content carried in slot t's row", since the whole row is spliced; a reveal that stores q_{>t} in slot t reads follow.
- **VALUE-SWAP (ii), whole V_t**: follow under serial use *and* under precompute stored in the newest version; own under
  recompute or history reading (see (c)); ProsQA's binary answer makes the two exhaustive.
- **STALL**: detects pass-count clocks only. Content-keyed serial computation, content-keyed reveal and global
  precompute all recover, so STALL cannot separate them (the draft says so for reveal: PASS).
- **Undetectable combination.** A pass-1 precompute in a code that neither the 2-block probe nor the linear FD probe
  decodes, released content-keyed one step per pass and stored outside slot t, passes every C1 component on GSM8K; on
  ProsQA the analogous route (hidden distances in the newest version) passes PROBE-GAIN only if PROBE(1) fails to decode
  it. The OC's "P(VB-C1 | no serial use) <= 0.02 / 0.01" therefore assumes probe-decodability of any precompute. Fix:
  state that assumption beside the OC rows and give the probes a capacity check (e.g. PROBE(1) / FD trained on a known
  one-pass model — CODI-20's latents — must decode its answer / values).

## 5. GSM8K DECOMP statistics (§1.3, §3.2) — independent parser (proofs_lang/gsm_check.py, gsm_wrong.py; JSON beside)

| claim | draft | mine | verdict |
|---|---|---|---|
| test m (gold "<<" count) 0..8 | 18 / 65 / 357 / 364 / 290 / 138 / 57 / 21 / 9; m >= 3 879, >= 4 515, >= 6 87 | identical | PASS |
| train m (first 60k) 1..7+ | 16 / 38 / 27 / 12 / 4.5 / 1.5 / 0.9 % | 16.4 / 37.5 / 27.0 / 12.3 / 4.5 / 1.5 / 0.87 % (all 384,622: 16.3 / 37.2 / 27.0 / 12.5 / 4.6 / 1.5 / 0.8 %) | PASS |
| single-op-only items | 61 % test / 70 % train | 61.4 % / 70.0 % (ast BinOp count) | PASS |
| values: p99 digits, max, non-integer, negative | 6, 7 (test); 2.0 % / 10 %; < 0.3 % | 6, 7; 2.04 % / 10.1 %; 0.09 % / 0.23 % | PASS |
| question tokens p50 / p90 / max; literals per question | 53 / 86 / 182; 3.4 | 53 / 86 / 182 (all 1,319); 3.43 | PASS |
| DECOMP parse share | 98.6 % train (60k) / 93.9 % test | 98.48 % / 93.71 % (1,236 vs 1,238 items; my stricter unary-minus rule) | PASS (within 0.2 pt) |
| test m_bin: <= 12 covers 100 %; cum 0.259 / 0.496 / 0.846 at <= 2 / 3 / 5 | — | 1.000; 0.2597 / 0.4968 / 0.8455 | PASS |
| mean m_bin 3.80 (4,706 / 1,238) "vs mean m 3.25" | — | 3.800 (4,697 / 1,236); mean m on the **same parsed subset is 3.30** (3.25 is over all 1,319) | CORRECTED (minor): ratio 1.15, not 1.17 |
| "the 0.31 % 'wrong' equations" | 0.31 % | 0.308 % of 992,057 parsed train equations (|exact − stated| > half a unit of the stated last place); 0 on test | PASS |
| CODI-20 "m <= 2: 0.621 (n 422)" | n 422 | eval.json by_steps excludes m = 0: 422 = m 1-2 (the 18 m = 0 items are in neither band) | CORRECTED (label "m 1-2") |
| B0-1 "all 384,622 train lines" | 384,622 | 384,622 items (2 malformed) | PASS |

**Gaps the spec should close (not draft errors in numbers).**
- **m_bin > 12**: 171 train items (0.045 % of parsed; max 24) need more than Q = 12 slots; 0 test items. Spec: mask steps k > 12 (R1 kept).
- **Literals > 14**: 18 train items (max 22) exceed the lit_1..lit_14 pointer vocabulary; mask those pointer terms.
- **Pointer provenance** (train, all operand terms): unique literal 48.9 %, earlier result 26.0 %, internal sub-step 6.7 %,
  ambiguous (both) 2.6 %, non-unique literal 2.9 %, CONST / unmatched 12.9 % (number words count here: "three" is not a
  digit literal). So ≈ 5.5 % of pointer terms are masked as ambiguous and 12.9 % are CONST, whose *value* no pointer
  carries (B2's ARITH-VALID cannot check those steps).
- Values >= 10^7: 151 train values; not exact at 2 dp: 8,578 (0.86 %): masked per §2.3 (consistent).

**"Per-pass targets from GSM8K-Aug's equations, C4-legal as distillation" — CORRECTED (qualify).** NORTH_STAR C4 admits
"a text-CoT model's outputs (distillation)" and excludes "hand-built intermediate states (no task simulator's per-step
states)". GSM8K-Aug's equations are the teacher's CoT, so equation-final values are distillation targets. But DECOMP's
binary split makes **≈ 13.5 % of B1's per-pass value targets internal sub-step values that the teacher never wrote**
(train: 1,135,784 binary steps vs ≈ 982,500 equations in parsed items; test 4,697 vs ≈ 4,073), computed by DECOMP's
evaluator, and every ptrA / ptrB target is a value-matching annotation. Fix wording: "CoT-derived distillation targets:
equation results as written by the teacher; sub-step values of multi-op equations (≈ 13.5 % of step targets, 30 % of
train items) and operand pointers derived from the teacher's equations by DECOMP (no information beyond the CoT string)
— C4-qualified, not plain CoT distillation." Strict variant if wanted: mask internal sub-step value terms.

## 6. OC arithmetic and cost model (§7, §9; proofs_lang/cost_oc_check.py → cost_oc_check.json)

| claim | draft | recomputed | verdict |
|---|---|---|---|
| VB-C1-L component product (P \| effect) | 0.43 | 0.8 × 0.7 × 0.85 × 0.9 = 0.428 | PASS |
| VB-C1-GSM P(pass \| effect) ≈ 0.4 | 0.4 | 0.85 × 0.6 × 0.9 × 0.8 = 0.367 (+ positive correlation) | PASS |
| P(VB-C1-L) ≈ 0.2 = 0.35 × 0.5 | 0.2 | 0.175; + false-positive mass 0.65 × 0.02 → <= 0.188 | PASS (rounding up; say ≈ 0.18) |
| P(VB-C1-GSM) ≈ 0.08 | 0.08 | not derivable from stated parts; implies P(serial \| B1) ≈ 0.2, consistent with ONE-PASS-CHAIN 0.45 and entry 0.75 | PASS (as conjecture) |
| P(SD-C2-L) ≈ 0.05 | 0.05 | the draft's own chain: A2 needs DEEP (0.4) ∧ NO-CLOCK (≈ 0.85-0.9) ∧ yield gate (≈ 0.8-0.9), then ABSORB \| DEEP 0.65, VB-C1-L re-read on A2 (≈ 0.4-0.5), K′24-STABLE (≈ 0.7-0.8), DEPTH-BEYOND \| SD-SELF (H-mixture 0.27-0.5) → **0.013-0.042, central ≈ 0.02** | **CORRECTED** — fix: "P(SD-C2-L) ≈ 0.02 (0.01-0.04)". Also "ABSORB 0.5" (A2 row) is inconsistent with "0.65 given A1 DEEP": A2 runs only after DEEP, so unconditional ABSORB ≈ 0.18 and ABSORB \| A2 runs ≈ 0.65 |
| **P(>= 1 decisive C1 positive) ≈ 0.25** | 0.25 | independence: 1 − 0.8 × 0.92 = **0.264** (0.301 if SD-C2-L counted, but SD-C2-L needs VB-C1-L on A2, so it adds ≈ 0); positive ProsQA / GSM8K correlation (same architecture, same harness) lowers it toward max(0.2, 0.08) = 0.20 | **CORRECTED** — fix: "≈ 0.22-0.26 (0.26 is the independence upper bound; the two cells share the architecture, so expect the low end)". With P(VB-C1-L) ≈ 0.18: 0.20-0.25 |
| BEATS-CODI20 P(pass \| Δ = 0), power at Δ = +0.06 (paired SE 0.014) | 0.02 / 0.98 | 0.016 / 0.984 | PASS |
| SE claims (follow 0.5 ± 0.016; PROBE-GAIN-GSM SE 0.02; H_shift sampling SE 0.008) | — | sqrt(.25/1000) = 0.0158; ≈ 0.02; sqrt(.21/4000) = 0.0072 | PASS |
| A0-2 "max over ~100 cells ≈ 0.55 = 0.5 + 3.2 SE" | 0.55 | 0.5506; 3.2 SE ≈ the 85th percentile of the max of 100 independent \|Z\| (median 2.7) — conservative | PASS |
| speed law s/step = max(0.42, 0.42(0.4 + 0.6 r)); A1 0.82 s, $1.20 | — | 0.817 s, $1.20 (code); cost_model.py's docstring states a different law ("0.42 × r × 0.6") | PASS (fix the docstring) |
| A1 at measured tokens (T 533, S 50 at D 8, 2.5-token names) | T 450 | 0.888 s/step, **$1.30**, attention 34.9M entries (8.9 GB fp32 bs 64, not 7.3) | CORRECTED (minor, +8 %) |
| A2 ≈ $0.75; A2-CTRL $0.55; A-TXT $1.0 | — | $0.72; **$0.60** at the $0.22/h used elsewhere ($0.55 is at $0.20/h); $0.99 | PASS / minor ($/h mixed) |
| B1 0.45-0.9 s/step, $3.3-6.6 | — | under the stated launch-bound law the step scales with the batch's max K: K_i ~ U{m_bin..16} puts K = 16 in 99.1 % of bs-64 batches vs D3-T's 12 → >= 0.56 s/step → **$4.1-6.6** | CORRECTED (low end) |
| **Required total $5.3-10.6, central ≈ $6** | 5.3-10.6 | sum of the stated ranges: **$4.6-10.6** (5.3 = A centrals + B1 low, mixed); with B1 >= $4.1: **$5.4-10.6**; +10 % bad hosts: $5.9-11.7; central $5.95 | **CORRECTED** — fix: "$5.4-10.6 (central ≈ $6; ≈ $6-11.7 with the 10 % host margin)" |
| guards sum $16.1; worst-case margin > $10 under $25 | — | 16.1; worst case incl. optionals +10 % = $14.4 → margin $10.6 | PASS |
| full-T memory 760M entries / 190 GB; concept slots 28.4M / 7.3 GB | — | 758.2M / 194 GB; 28.4M / 7.3 GB (at the draft's T; 34.9M / 8.9 GB at measured T) | PASS |
| mention slots 70.7M vs 28.4M, 1.17 vs 0.82 s | — | identical | PASS |

**Independence flags.** (1) P(>= 1) treats ProsQA and GSM8K serial use as independent; they share one architecture and
harness, so the union is lower. (2) "P(VB-C1-L \| no serial use) <= 0.02" multiplies component false-positive rates as if
independent across failure modes; a single undetected route (§4: an undecodable precompute released content-keyed) passes
PROBE-GAIN, ONSET, STEP and NO-CLOCK together, so the bound is conditional on probe-decodability, not a product.
Neither changes a decision (all cells fit the cap with > $10 margin).

## 7. Tally and file index
- **FAIL (3, all with fix text above):** §3.1 BIDIR "measures shallow root-aware leakage" (it is ½ / 1 identically);
  §3.1 / §4.5 Corollary "self targets = gold, noise 0" as written with §2.1's HOLD-after-τ; §4.6(c) "own needs over-reach".
- **CORRECTED (21):** p8_na depth-3 failure; "0.86-0.93" quote; NN residual by base depth; §191 cue consistency;
  undirected leakage / C1-band rationale; 0.05 margin; token size law; |S| bound 84; §4.3 convention; L1-L SLVA sublayer;
  CAP-aff-L evidence; detector one-sidedness; FD(k, k) contract gap; m_bin / m ratio; CODI "m <= 2" label; C4 wording;
  P(SD-C2-L); P(>= 1); A1 cost at measured T; B1 low end; required total.
- **PASS (46):** everything else in §§1-6 tables, incl. Lemma S-L, exchangeability (strong form), PERSON-SWAP (a)(b)(d),
  reveal zero-loss, every GSM8K count, every SE / OC product, guards and cap margin, memory arithmetic, CON38 3-token count
  (30 / 38; ENT17: 15 one-token, 2 two-token names — use one-token names for PERSON-SWAP and σ-symmetric T caps).
- Not checked: G8 "[PROVED, VB §5.6]" (imported), SELFDEPTH S2(b) noise recursion (imported), arXiv IDs.

Scripts (research7/proofs_lang/): anchor_check.py (ac_prosqa_test / ac_p8na_test / ac_p8na_train.json), nn_check.py
(nn_seed0 / nn_seed1 / nn_train_seed2.json), cue_ceiling.py (cue_ceiling.json), sym_check.py (sym_fresh_s7.json),
sym_variant.py (sv_any / sv_distractor / sv_direct.json), direct_maxdist.py (direct_maxdist.json), sound_check.py
(sound_check.json), gsm_check.py (gsm_check.json), gsm_wrong.py (gsm_wrong.json), cost_oc_check.py (cost_oc_check.json).
