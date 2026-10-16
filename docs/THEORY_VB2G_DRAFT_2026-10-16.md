# VB-2G: fixing the three bottlenecks inside a pass-global attention memory (DRAFT, 2026-10-16 03:50 EDT)

**Status.** This is a draft from the second deep-research session, not a registration. Registration and any spend need the user's OK, relayed by Audit.
- **Request.** The user's "lets do another heavy reseach session … try and extend/modify/invent new architecture" (typed in Audit's session; the clock read 02:44 EDT), with the chosen focus "fix bottle necks": (1) per-pass step capability, (2) beyond-training depth (C2), (3) learned stopping (C5).
- **Constraint.** The standing user rule (22:07 EDT, 10-15): a pass-global attention memory, not RNN memory.
- **Process (all Opus).** Prior art LIT2; three inventors S (step), D (depth) and H (halting); a synthesis spec SYNTH2; a red team REDTEAM2; a proof pass PROOFS2. All are in docs/research_bottleneck_2026-10-16/, with scripts in scripts/. This draft adopts the red team's design (VB-2G) and smallest cell set (§B), and every proof fix (PROOFS2 verdict rows 4–9, 17, 28–31).
- **Tags:** [PROVED], [PROVED UNDER X], [DERIVED], [MEASURED], [ASSUMED], [CONJECTURE].

---

## 1. What the evidence says binds (and what it does not)

1. **Halting has no generalisation problem of its own** [PROVED, H Thm 3 / PROOFS2 row 25]. A zero-parameter content predicate read through VB's linear recode head fires exactly at t = n whenever the recode's skeleton is exact. Every depth-timing failure is the step rule's or R's.
2. **Size and clock cues in VB** [MEASURED, D §1.1; H §1]. The model can read:
   - absolute positions, including position information derived inside h_6: after removing wte and wpe, position is still linearly decodable from h_6 at R² 0.9986 and does not extrapolate past slot 33 [MEASURED, PROOFS2 row 4];
   - the PA gather's wpe queries;
   - the VOID count;
   - the bank normaliser, which saturates by t ≈ 3–4 [PROOFS2 row 31];
   - V_0 at constant bias every pass;
   - V_1's fixed-origin age;
   - the zero-loss "t ≥ 9 ⇒ hold" shortcut under n ≤ 8, K ≤ 12 training;
   - null-space progress counters.
3. **Failed vs converged seeds** [MEASURED, Audit-reproduced]:
   - §205 s3 writes recoverable step values only at passes 1–2 (205-4).
   - Item-level fits (beyond training, n 9–12, 3 failed vs 5 converged; S's logit.py): subtree height adds first-error hazard **only** in failed seeds (+0.71 to +1.10 per SD vs +0.07 to +0.27). Pass, length and held load dominate in every model (REDTEAM2 F3: +420 to +1,850 held-out nats).
   - **The main beyond-training failure is therefore pass- or size-keyed (bottleneck 2), not the step.**
   - HH-FIT is not evidence (205-5). The "recompute" account stays a conjecture until the in-range item-level test.
4. **Feng's step is local in compacted coordinates** [PROVED, D Lemma 0 / PROOFS2 row 1].
   - Readiness is a radius-2 rule plus one prefix-OR: 0 mismatches in 1,024,297 operators.
   - The in-place write needs radius 4, and that radius is tight.
   - After t exact steps, the compacted state is a valid size-(n−t) problem (D Lemma 1). It lies *in Alg. 1's support*, but not in its distribution: taller, with more brackets (PROOFS2 row 2).
5. **History and depth.**
   - Full cross-iteration KV history hurt depth extrapolation in two papers: CoTFormer 2609.19521 (50.1% vs 99.4% for recent-1) and Bohde 2403.04929.
   - It also helps in distribution (TaH).
   - That contradiction is open (LIT2), and the data must decide it, not the architecture.

## 2. The design: VB-2G (VERSION-BANK, self-similar, history-swapped)

**Base:** §211's N2 (PURE-ANNEAL) under 211-16's base rule. If N2 is NO-ENTRY / LOSES-ONE-PASS while N1 is DEEP, the base question goes to the user.

The memory stays an append-only, fully readable attention bank. V_0 stays readable, there is **no age floor and no V0-OFF**, and recency is *learned*. Changes against N2:

1. **Compaction as key masks (D M1, red-team form).**
   - Slots decoded VOID (teacher keep mask in training, R's argmax at eval) are masked as keys in every later read, on the fixed slot grid.
   - This is mathematically the same as physical compaction (an fp64 gate), and it avoids the fixed-length code paths that REDTEAM2 F8 found (vbank.py:127, 206–207, 257–259, 220/238, 179–183).
2. **Restamp at storage (PROOFS2 row 6).**
   - The **stored** entry Ṽ_t carries the re-indexed stamps of the compacted slot order: Ṽ_t[j] = V_t[j] + a ⊙ (wpe[ĵ] − wpe[j]), with ĵ the compacted index and **a frozen at 1**, so the model cannot undo it.
   - Bank K/V and the PA gather's keys are computed from Ṽ_t, and gather queries use wpe[ĵ].
   - **Limit stated:** the direct embedding component is restamped, but derived position features are not (PROOFS2 rows 4–5). D's Theorem 1 therefore needs only that the restamp is history-free; it does not need A2.
3. **Normaliser N (PROOFS2 row 7):** r(j) = rms(Φ(current content)[j]), the prelude's encoding scale for the slot's current symbol. This is the identity at pass 1, so G1 holds, and it is consistent with Theorem 1.
4. **HISTORY-SWAP (red team §C; replaces V0-OFF, the floor and STALL-AUG).**
   - With probability p_h = 0.25 per item, at passes t ≥ 2, every bank entry older than the newest (V_0 included) is replaced by another micro-batch item's stored entries, with a foreign history length uniform in {0..K_max}.
   - There is no extra forward pass; masks handle lengths.
5. **A gentler anneal end (211-11 ANNEAL-COSTLY):** c = 1e4 until 1k, then 20 → 0 linearly over 1k–14k (the 211-3 schedule with its slope halved). At equal logits, off-self mass reaches 0.5 at about 9.5k.
6. **The logged and gated answer CE is the true-answer CE (REDTEAM2 finding 1).** The R4 gate, NO-ENTRY, entry dating and every watch read a no-grad CE against y_i, never a relabelled target.
7. **CONTENT-HALT as readout (H §2.1).** Zero parameters, no gradient, with the G-PROG and STUCK-STOP guards.

**What VB-2G does not include,** and the trigger that would add each:
- **COMMUTE:** only if the in-VB step trigger fires (211-16 item 3). It becomes the bank-repair variant if REPAIR ≥ 0.2 (211-16 item 2).
- **OPX:** only if OPERAND-FOLLOW < 0.5, as a logit-swap edit (REDTEAM2 F10: MCT edits leave the original value as runner-up at about 1.05 nats, so they are detectable).
- **Self-CANON:** only if a cell reads HISTORY-KEYED with CLOCK-PROBE ≥ chance + 0.2. Its target is Φ(dec Ṽ_t), the prelude's encoding of the model's *own* decode, ramped. This avoids PROOFS2's CANON-vs-COMMUTE conflict.
- **STALL-AUG:** only as a fine-tune (FT), H's form (per item, a copy, cap 12 + S), if the best cell reads CLOCK-HOLD / EDGE-COLLAPSE.
- **V0-OFF, an age floor, recent-1:** never without the user's explicit OK, and only if 211-16's eval-only arms show EVAL-HISTORY-GAIN.

## 3. Mathematics (claims the draft relies on)

1. **Lemma 0 / Lemma 1** [PROVED] as in §1.4.
2. **Tail lift** (D Thm 1) [PROVED UNDER A1 exact compaction, A3′ (each stored Ṽ_t equals the fresh encoding's stored entry), A4 history-free].
   - After n − m exact steps the continuation is bitwise the fresh size-m run, for m ≥ 2.
   - The quantitative form holds while drift stays below R's margin (PROOFS2 row 8).
   - **In VB-2G, A3′ is not enforced** (there is no CANON), so it is read as SELF-SIMILAR (RS). A4 is pushed by HISTORY-SWAP and read as HIST-TRUNC.
3. **Full lift** (D Thm 2) [PROVED UNDER factored A5′].
   - A monolithic radius-4 rule covers only 0.93 → 0.71 of states at n 9 → 16.
   - Factored keys (readiness 248 / write 169) cover 1.000 at n 9–16 [MEASURED, PROOFS2 row 9].
   - With softmax globals, exactness holds only to m* = γe^Δ/(2BqΛ_R): **each e-fold of length costs one nat of margin** [PROVED UNDER first-order composition].
   - The global features must be existential (a prefix-OR), never counts (PROOFS2 row 3).
4. **HISTORY-SWAP H1** [PROVED, as H Prop 8(ii)]. On swapped items the bank index is independent of the newest content's progress. So "t ≥ 9 ⇒ hold" and the count rule pay L_rec > 0 on a positive-probability set. The (index, remaining) support covers index + remaining ≤ K_max + 8, with no duplicate signature to detect.
5. **HISTORY-SWAP H2** [DERIVED]. On swapped items, I(target; old entries | newest) = 0. History reads there are zero-mean noise, and the recompute reward of free-running targets is cut by (1 − p_h).
   - **Not proved:** that a swap detector (literal consistency across versions) cannot restore history use on unswapped items. This is a named Goodhart route, G-SWAP-DETECT, measured by HIST-TRUNC.
6. **HISTORY-SWAP H3** [DERIVED]. It pushes the gather toward the newest version's self-position, addressing 211-12's diffuse gather (off-self 0.90).
7. **Halting** [PROVED]: H Prop 1 (no early-halt incentive), Prop 2 (identification), Thm 3 (skeleton exactness suffices) and Prop 4 (clock-leakage bounds; the 4c threshold is 7.6 nats at T 55).
8. **Clock inventory** (PROOFS2 row 31). It remains after VB-2G: null-space progress counters, plus derived positions.
   - The **decisive comparator-free clock test is PF's k-axis**, plus the FOREIGN pair (211-16 item 5), with PF × HIST-TRUNC separating counters from bank-age clocks.
   - HISTORY-SWAP trains no stall, so 211-13's **STALL-SHIFT stays untrained and decisive** here. REWIND is descriptive.

## 4. Goodhart routes (status)

| route | status in VB-2G | read by |
|---|---|---|
| pass-1 precompute | bounded on deep items (L1 + CAP-aff) | PROBE(1) / (2), ONESHOT |
| recompute from V_0 / older versions | V_0 readable by design. HISTORY-SWAP makes old entries uninformative on 25% of items (H2) | VALUE-SWAP own, RECOMPUTE-V0 (with an "all older masked" arm, REDTEAM2 F9) |
| bank-index clock / "t ≥ 9 ⇒ hold" | broken on a positive-probability set (H1) | FOREIGN pair, PF k-axis |
| null-space progress counter | **open** (no CANON) | PF, CLOCK-PROBE; self-CANON if triggered |
| swap detection (G-SWAP-DETECT) | possible | HIST-TRUNC Δ on unswapped-like natural items |
| derived position (h_6) | **open** in GPT-2 (PROOFS2 row 4) | POS-PROBE, MARGIN-SCALE |
| keep-mask errors at eval | the mask is teacher-set in training (exposure) | **KEEP-ACC** by n (pre-stated bar 0.995 at n 13–16, D §7) |
| self-referential answer CE | closed (item 6) | — |
| copying literals | not a bypass | — |
| timed release / sabotage | no negative-likelihood term | — |

## 5. Decisive readings (per cell, comparator-free)

**VB2G-C1** combines four groups:
- **(a) C1, untrained parts of VB-C1:** DEEP ∧ PROBE-GAIN ∧ SERIAL-ONSET ∧ VALUE-SWAP SERIAL-STEP.
- **(b) C2:** DEPTH-GENERAL (n 9–12 at K′ n + 2 ≥ 0.80) ∧ SELF-SIMILAR (RS |RESTART − INTACT| ≤ 0.05 at n 9–16) ∧ STEP-FLAT.
- **(c) Clock:** PF CONTENT-KEYED ∧ ¬CLOCK-FOREIGN ∧ NO-CLOCK (STALL).
- **(d) C5:** C5-VB (HALT-EXACT ∧ NOT-PREMATURE ∧ STALL-SHIFT ∧ DEPTH-TIMING).

The definitions are in 211-1 / 211-13 / 211-15 / 211-16. Each part is also reported separately, so a cell can be C2-positive without C5.

**Per-fix attribution inside the cell** (REDTEAM2 §B; one seed cannot attribute between cells):

| fix | reading |
|---|---|
| compaction | KEEP-ACC, RS, MARGIN-SCALE |
| restamp | POS-PROBE |
| HISTORY-SWAP | HIST-TRUNC Δ, FOREIGN, PF |
| anneal | 211-10-style bins against §211 N2 |
| halt | the C5-VB parts |

**OC (one seed):**
- P(DEPTH-GENERAL | size-keyed) ≈ 0.05: 0 of 14+ nested models reached 0.80; the best is 0.688.
- P(VB2G-C1(b) | no invariance) ≤ 0.02.
- P((b) | invariance learned) ≈ 0.6.
- Per cell, P((b)) ≈ 0.15–0.25 [ASSUMED].
- A positive is strong; a null is weak.

**Censoring:**
- A dead run or NO-ENTRY (on the true-answer CE) fails everything.
- KEEP-ACC < 0.995 at n 13–16 is reported as a cap.
- RS / PF / FOREIGN cells with < 100 items are UNREADABLE (not met).
- Never-halting counts as failure.

## 6. Plan and cost

**Stage 0 ($0, eval-only; pre-stated in 211-13 / 211-15 / 211-16 before any §211 output; running needs the user's OK).**
- On §211 N1–N3: CONTENT-HALT, the STEP trigger, REPAIR, RS / PF / HIST-TRUNC, the eval-only history arms and the FOREIGN pair.
- On INJECT models: S's D1, and RS / PF.
- Stage 0 decides which Stage-1 cells are proposed.

**Stage 1** (RTX 3090, seed 1, 48k × bs 64, n 1..8 train, §170 test, one GPU class, off-host checkpoint mirror):

| cell | design | runs if | $ |
|---|---|---|---|
| **G1 (main)** | VB-2G (§2 items 1–7) on the 211-16 base | Stage 0 does not read all-HISTORY-FREE (D's go rule), or §211 fails DEPTH-GENERAL (expected) | 1.6–2.6 |
| **S** | G1 + COMMUTE (bank-repair variant if REPAIR ≥ 0.2); logit-swap OPX only if OPERAND-FOLLOW < 0.5 | the 211-16 STEP trigger fires on a §211 cell | 1.6–2.6 |
| **FT** | H's STALL-AUG fine-tune, AUG vs CTRL, 8k each, on the best DEEP §211 cell | that cell reads CLOCK-HOLD / EDGE-COLLAPSE / STALL-COUNT | about 0.6 |
| **G1+C** (follow-up) | G1 + ramped self-CANON | G1 reads HISTORY-KEYED with CLOCK-PROBE ≥ chance + 0.2 | 1.6–2.6 |

- **Family cap proposed: $6.** G1 alone is ≤ $3; G1 + S + FT is $3.8–5.8.
- **G-TIME** runs on a forced final-stage batch.
- Per-cell guard $3. Nothing is dropped silently.

**Stage 0 gates for G1** (REDTEAM2 §D, PROOFS2):
- fp64 equivalence of mask compaction and physical compaction;
- G1 identity at K = 1 with the new N;
- G-C3 with restamp at storage;
- no monkeypatching of `bias_row`;
- deltamem.POS_OFFSET asserted 0 (deltamem.py:21);
- per-term gradflow checked by ast (decorators: vbank.py mct :264 and calibrate :284, and the vbank_eval parts, are no_grad);
- HISTORY-SWAP information-zero check (old entries carry no target information on swapped items);
- KEEP-ACC harness;
- G1b drift smoke;
- mem_check.

**HISTORY-SWAP needs its own derivation note before registration** (math-backing rule). §3 items 4–6 are its draft.

## 7. Predictions [CONJECTURE, before any §211 output]
- **G1:** P(entry) 0.6; P(DEEP) 0.45; P(DEPTH-GENERAL | DEEP) 0.4; P(SELF-SIMILAR | DEEP) 0.5; P(C5-VB | DEEP) 0.3.
- **§211 Stage 0:**
  - N1 or N2 reads HISTORY-KEYED (P 0.5);
  - CLOCK-HOLD over EDGE-COLLAPSE at n 9–12 (P 0.45 vs 0.25);
  - the STEP trigger fires on no DEEP cell (P 0.6).

## 8. Top risks and falsifiers
1. **Derived position in h_6** (PROOFS2 row 4) keeps the step size-keyed despite the restamp. *Falsifier:* POS-PROBE extrapolation fails, and STEP-FLAT fails at n 13–16 with HISTORY-FREE.
2. **Swap detection.** *Falsifier for H2:* HIST-TRUNC Δ ≥ 0.10 on natural items in G1.
3. **Optimisation:** the anneal plus HISTORY-SWAP noise. *Falsifier:* G1 NO-ENTRY while §211 N2 entered.
4. **Keep errors cap depth** (KEEP-ACC < 0.995).
5. **The constraint as experienced by the user.** VB-2G keeps the bank fully readable and recency is learned, but a model that learns recent-1 behaves RNN-like in function. That is reported (CARRY-LIKENESS, HIST-TRUNC), never imposed.

**Premise falsifier:** Stage 0 reads HISTORY-FREE ∧ HEAD-LIMITED on every model. Then self-similarity is not the bottleneck, and the locality / structural-position cell (D's D3, Ito-style 2608.31067) comes first.

---

## Addendum 1 (2026-10-16, Audit's review of 6bde99cb): pass-global *use*, not just access

**The tension (Audit).** HISTORY-SWAP keeps every version addressable, but it *trains* the model to distrust older passes: on swapped items they carry no information (H2). That pushes behaviour toward recency. The user's intent is pass-global *use*.

**Change of default: HISTORY-INSERT replaces HISTORY-SWAP in G1.**
- With probability p_h per item, at passes t ≥ 2, insert o ~ U{1..K_max} stored versions of another micro-batch item at the *old end* of the bank (ages shifted by o).
- **All of the item's own versions are kept.** No extra forward pass is needed; masks handle lengths.
- **H1′ [PROVED, same argument as H1 / H Prop 8(ii)].** On inserted items the bank index and the size of the bank are shifted by o, independently of the item's progress. So "t ≥ 9 ⇒ hold", the count rule and bank-size clocks pay L_rec > 0 on a positive-probability set. This holds for any p_h > 0, and the penalty scales with p_h.
- **What it does not do [DERIVED].** It does not make the item's own history uninformative: H2 does not apply. Recompute or reuse from older *own* versions stays possible and rewarded where it helps. In-distribution history use, LIT2's TaH side, is untouched. Foreign entries are content-unrelated, so learning to ignore them is legitimate, not a bypass.
- **New risk:** dilution from extra bank entries (Lemma 2), measured by MARGIN-SCALE.
- **Consequence for readings:** FOREIGN-INSERT becomes trained-for, so it is descriptive in G1. **PF's k-axis plus FOREIGN-REPLACE stay decisive** for clocks, together with STALL, which is still untrained.

**HISTORY-SWAP is kept as a triggered alternative (G1-S).** It runs only if G1 reads HISTORY-KEYED *and* HISTORY-HURTS: older versions measurably hurt depth (eval-time HIST-TRUNC gain ≥ 0.10 at n 9–12 with n ≤ 8 ≥ −0.03). That is LIT2's CoTFormer pattern. Its recency push is then the evidence-backed remedy, and it is reported to the user as such.

**The choice p_h = 0.25** is a stated design choice, not an optimum.
- Any p_h > 0 makes the clock rules non-zero-loss (H1′).
- A larger p_h strengthens the anti-clock pressure but adds more dilution (INSERT) or more recency pressure (SWAP).
- At 0.25, 75% of items train on a clean bank.
- No derivation picks 0.25 over, say, 0.1.

**(a) New reported reading beside VB2G-C1: HISTORY-USE.** At test, with no insert or swap, compare intact accuracy against recent-1 + V_0 (HIST-TRUNC):
- **HISTORY-USED:** HIST-TRUNC drops n ≤ 8 by ≥ 0.03, or n 9–16 by ≥ 0.05. The model reads its older passes.
- **HISTORY-IGNORED:** |Δ| ≤ 0.02 at both. The model behaves recent-1 in function, and that is reported to the user as recency-in-practice.
- **HISTORY-HURTS** as in 211-16.

The VB2G-C1 row states which of these holds.
