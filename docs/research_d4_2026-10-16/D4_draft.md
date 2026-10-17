# D4 (DRAFT, $0): computed PAUSE passes, a fix for the remaining depth limit of VERSION-BANK / SLVA (2026-10-16 23:27 EDT)

**Status.** This is an Opus subagent draft for the Theory session. It is not a registration and it spent $0. No training, GPU or repo edits were made.
- **What was read:** NORTH_STAR, D3_v1 (frozen), §211 / §212 / §213 with notes 212-13…19 and 213-1…23 plus the D3-T result, the VB-2G draft, H_halting / D_depth / PROOFS2 / REDTEAM2, and §187 / §190 (the DeltaMem counter history). Code: boxes/vbank.py, slva.py, vbank_diag.py and vbank_halt.py.
- **Local analyses** (CPU, ≤ 3 threads, existing JSON only) are in this folder:
  - `s0_collapse.py`, `s0_finishpass.py`, `s0_clockkey.py` and `d4_edge_collapse.py` (fixture-checked) on N1 / N2's `eval_vbhalt.json` rows;
  - `d4_schedule_stats.py`, a simulation of the training schedules.
  - The outputs are the `.json` files of the same names.
- **Not read:** no D3-T vbhalt / stepbyt / hist24 output exists yet. D3-F's `eval_vb.json` has landed, but this draft did not open it, so the branch rule below is pre-stated blind to it.
- **Tags:** [PROVED], [PROVED UNDER X], [DERIVED], [MEASURED], [ASSUMED], [CONJECTURE], [EMPIRICS] (literature).

---

## 1. Diagnosis

### 1.1 What is established

| finding | evidence | tag |
|---|---|---|
| D3-T's limit is in the trajectory, not in the step or the budget | RS restart 1.000 in every cell, intact 0.58 (D +0.418); acc at K′ 10 → 24 is flat at n 9–12 (0.926 → 0.913, 0.609 → 0.590) | MEASURED (§213) |
| **Beyond range, D3-T finishes early** | correct at K′ = n − 1 ÷ correct at K′ = n + 2: n 8 **0.18** (in range: chance plus the coda) → n 9 0.36, **n 10 0.89**, n 11 1.03. N1: 0.15 / 0.27 / 0.96 / 1.13; N2: 0.16 / 0.20 / 0.65 / 1.01 | MEASURED (eval_vb acc_k); the reading is DERIVED |
| D3-T's decode leaves the schedule mid-trajectory while answers stay right | n 9: exact decode at t 7 is only 0.501, yet acc is 0.923 (continuation from exact t 7 is 0.996) | MEASURED (eval_rs) |
| **N1 and N2 read EDGE-COLLAPSE**, a pass-indexed deadline (new reading, this draft). Hazard h(t) = P(an on-schedule pass removes ≥ 2 operators) | **N1** (n 10–14, on-schedule events with ≥ 3 operators alive): h = 0.07 / 0.21 / 0.50 / 0.84 / 0.99 at t 5 / 6 / 7 / 8 / 9, the same at every n ≥ 10. The decoded state reaches 0 operators at **median pass 9 for n 10, 11, 12 and 13** (10 at n 14). **N2:** median t_fin = 9 at every n 9–16 | MEASURED (vbhalt rows; `d4_edge_collapse_refs.json`, `s0_finishpass.json`) |
| **N1's STEP-BY-T "GRADUAL-DRIFT" is mostly over-reduction** | Over-reduction share of in-trajectory step failures, ORS = h(t) / (1 − step exactness): 0.51 / 0.63 / 0.67 / 0.87 at t 5–8; **ORS over t 6–8 = 0.69**. So 213-18's label rule would have sent N1 to the wrong lever | MEASURED; DERIVED (the two terms are conditioned differently: alive count vs strict decode) |
| Content-independent drift is ruled out | In range, copy slots are exact at t ≤ 8 (D3-T stepcap copy_slot_acc 0.99998; acc 1.000 at n ≤ 8), while n ≥ 10 items fail at those same t. The failure needs (t × state), not t alone (Cor. D4c) | DERIVED from MEASURED |
| N1's clock is not the bank index | FOREIGN-INSERT o = 6 (bank index + 6, carry fresh): n 6 / 7 / 8 = 0.983 / 0.974 / 0.951 | MEASURED (eval_foreign) |
| Masking history does not remove D3-T's failure | HIST-TRUNC: n 9 / 10 / 11 = 0.761 / 0.587 / 0.197, against intact 0.923 / 0.603 / 0.266 | MEASURED |
| N1's hazard follows pass index × remaining work, not progress ρ | Logit fit h ~ t + dev: b_t = 1.23 (SE 0.013), g_dev = **+0.53** (SE 0.035). Items one operator behind collapse *more* (catch-up), where a ρ-keyed clock predicts g = −b_t | MEASURED, descriptive. Caveat: decode noise selects which items read "behind" (`s0_clockkey.json`) |
| Under D3-T's schedule, any deadline rule costs nothing | P(input unfinished at a pass with t ≥ 9) = 0; on computing passes, t + r ≤ 9 | PROVED (counting; `d4_schedule_stats` BASE) |
| Precedent in another harness | §190 (DeltaMem, nested arithmetic): removing the pass token and the magnitude counter removed the fixed-t edge. n 10 then held ≥ 0.95 through t 10. The residual failure was "done + remaining > 8" at n 11–12 (DEPTH 0.659) | MEASURED (§190 s1) |

### 1.2 Thesis [CONJECTURE]

- **What fails.** D3-T's depth limit is the EDGE-COLLAPSE deadline, which is Prop 8(i)'s "t ≥ 9 ⇒ finished" shortcut put into practice:
  - in training, every pass with t ≥ 9 holds a finished state;
  - so beyond range, the late passes compress the remaining work so as to "finish" by about pass 9–10.
- **What carries it.** The deadline is carried by the **lineage age** of the carried state, a_t (core applications so far), not by the bank index. That is N1's evidence. For D3-T it is checked by S0-2 / S0-3.
- **Why the obvious levers miss.**
  - LONG-K adds only finished passes, so it strengthens the deadline's zero-loss support (Prop D1).
  - Copy-stalls (H / D forms) leave the age unchanged.
  - Only **computed non-reducing passes** decouple age from progress.

### 1.3 Branch rule (pre-stated; inputs are D3-T's queued `eval_vbhalt.json` + `eval_stepbyt.json`, read with `d4_edge_collapse.py`)

- **The inputs:**
  - EC = EDGE-COLLAPSE (§5 defines it);
  - ORD = over-reduction-dominant, ORS over t 6–8 ≥ 0.5 (pooled n 10–16, ≥ 200 stepbyt events per t);
  - L = 213-18's label (PASS-COUNT-KEYED / GRADUAL-DRIFT / MIXED).

| condition (first match wins) | branch |
|---|---|
| EC ∨ ORD ∨ L = PASS-COUNT-KEYED | **A: PAUSE-A** (delayed-start-heavy, P ≤ 8, cap 20) |
| L = GRADUAL-DRIFT ∧ ¬EC ∧ ¬ORD (wrong-content failures dominate) | **B: PAUSE-B** (scattered-heavy, P ≤ 12, cap 24, long tail, TBPTT) |
| otherwise (L = MIXED, unreadable inputs) | **B** (B ⊃ A) |

- **The base.**
  - D3-F becomes the base only if both hold:
    - its mean acc_n2 over n 9–12 is ≥ D3-T's + 0.05;
    - it is LENGTH-FREE ∧ VB-C1 ∧ acc_k24 ≥ 0.99 at n 1–8.
  - The rule is then applied to D3-F's labels, and the FT keeps the table frozen (`requires_grad False` before the optimizer; G-D3 checksum). Nothing else in D4 depends on the table.
  - Otherwise the base is D3-T.
- **Predictions for D3-T [CONJECTURE]:**
  - EC P 0.6, ORD P 0.75;
  - L = GRADUAL-DRIFT P 0.6, MIXED 0.25, PASS-COUNT-KEYED 0.15;
  - so Branch A with P ≈ 0.8.

---

## 2. Mechanism

### 2.1 Computed PAUSE pass (common to A and B)

- **The pass.** At a pause pass t of item i:
  - the core runs normally on the SLVA pass input plus a train-only marker, U_t ← U_t + π at every real slot. π ∈ ℝ⁷⁶⁸ is a learned parameter `pause_vec`, initialised to 0, and is added through the `input_edit` location (after the SLVA read, before block 6; verified in `vbank.rollout`);
  - its output V_t is **stored in the bank and carried** (it is SLVA's newest version), as at any pass.
- **What it changes, against the levers already considered:**

  | lever | carry and stored entry at the stall pass | moves bank index b | moves age a | moves progress ρ |
  |---|---|---|---|---|
  | copy-stall (H / D) | V_{t−1} | yes | no | no |
  | **computed pause** | **the computed output** | **yes** | **yes** | no |
  | LONG-K | adds finished passes only | — | — | — |

- **R2 target:** E_{min(ρ_i(t), n_i)}, where ρ_i(t) is the number of non-pause passes ≤ t.
  - At a pause pass E_t = E_{t−1}, so vb_loss's existing HOLD branch applies (the unchanged-slot mean only, value classes weighted 2).
  - No new loss term is added. The change is the per-item target index map.
- **R1:** t ∈ [τ_i, K_i], with τ_i = n_i + P_i (the completion pass).
- **Idle passes** (t > τ_i): no pauses after completion, so they are D3-T's: R1 holds the answer and R2 recodes the finished E_n. This is also B's long tail. Lemma D3 explains why those targets alone give SLVA's history weights almost no gradient.
- **At test:** never any marker. The registered readings run plain passes. The marker appears only in the eval arms of §5 (PAUSE-INV, START-D).
- **What it does not do:**
  - it adds **no reduction steps**: n ≤ 8, and at most 8 work steps per trajectory;
  - it adds no position or pass input, no rule overwrite, and no new readers;
  - the bank stays pass-global and append-only (constraint kept).

### 2.2 Schedules

| | PAUSE-A (Branch A) | PAUSE-B (Branch B) |
|---|---|---|
| augmented batches | each step's batch independently with p 0.5 (rng seed + 21300); plain batches are exactly D3-T's (K = min(n + U{0..4}, 12)) | same |
| pauses per item P | U{1..8}, independent of n | U{1..12} |
| delayed start D | U{0..P}, all at passes 1..D | U{0..⌊P/3⌋} (scattered-heavy) |
| other pauses | the remaining P − D uniformly interleaved with the n reductions | same |
| budget | K = min(n + P + U{0..4}, **20**); completion n + P ≤ 16, never censored | K = min(n + P + U{0..4} + 1[0.25]·U{0..8}, **24**) (the long idle tail) |
| backprop | full BPTT if the forced worst-case mem_check is ≤ 21 GiB at micro 1, else TBPTT k = 12 | TBPTT k = 12 whenever Kb > 20 |
| decoupled up to | computing passes to t 16 | computing passes to t 20 |

- **Why A is delayed-start-heavy.** A start of D on an n-item puts the remaining-work-vs-pass profile on the tail of a size-(n + D) trajectory from pass D + 1. That is exactly the (t, r) diagonal of an n ≥ 9 test item.
- **Why B is scattered-heavy.** B ages partly reduced states and copy slots, which is where wrong-content drift would live (Lemma D3).

### 2.3 Everything else is D3-T's (ARGS-DIFF)

**New flags:**
- `--pause {none,A,B}`, `--pause_p 0.5`, `--pause_max 8|12`, `--pause_ddiv 1|3`, `--kcap 20|24`, `--tail_p 0|0.25`, `--tail_max 0|8`, `--tbptt 0|12`;
- FT only: `--init_from <base model.pt>` (strict load except the single new key `pause_vec`, with no `calibrate`, so β / β⁰ / γ are bitwise the base's), `--lr_const 3e-5 --warmup 500`, `--steps 12000`, `--cur 8`, `--c_sched 0,1` (c = 0 at every step, gate G-P4).

**Unchanged:** mode slva, table, bs 64, seed 1, kmax 12 (plain batches), kextra 4, mu, rec_w 1.0, R1 plus reweighted R2, cur_gate, wd 0.01, eval contract.
- **Verified in code:** `vb_loss` runs every batch to Kb = max K_i (so cost follows the batch max), and the HOLD branch is taken when no slot changes.
- **Absent in code (new trainer code needed):** `--init_from`, `--lr_const` and a train-time `input_edit`.

---

## 3. Mathematics

**Prop D0 (support identity: no free lunch) [PROVED].**
- For every item with n ≤ 8 under **any** pass schedule (stalls, pauses, idle passes), every pass input has ρ + r = n ≤ 8.
- **Hence (a):** a progress-keyed deadline g(ρ) ("ρ ≥ 8 ⇒ finished") is zero-loss under every schedule.
- **Hence (b):** for any fixed test pause pattern Π, *including none*, the training items whose pause history equals Π's prefix are unfinished at t only if t ≤ 8 + |Π_{<t}|. So a history-gated deadline is zero-loss.
- **So D4 is an inductive-bias bet, stated as such:**
  - if the late-pass step depends on (local content, age a) and not on ρ or on the pause history, then covering (a, r) suffices (coverage is trivial);
  - test states with ρ ≥ 9 are never in support. That is §190's residual "content limit".

**Prop D1 (which clocks each schedule decouples) [PROVED by counting; numbers COMPUTED, 200k simulated items].**
- **Definitions:** b = bank index; a = lineage age of the newest version; ρ = progress.
- **Under D3-T,** on every pass with unfinished input, b − 1 = a = ρ. So "κ ≥ 9 ⇒ finished" is zero-loss for κ ∈ {b − 1, a, ρ}.
- **Copy-stall:** b − 1 = ρ + S and a = ρ. It decouples b only.
- **LONG-K:** it only adds passes with finished input, so P(unfinished | t ≥ 9) stays 0. **It adds zero-loss support for the deadline** [DERIVED: LONG-K alone can entrench EDGE-COLLAPSE].
- **Computed pause:** b − 1 = a = ρ + P_t. It decouples b and a.
- **P(unfinished input | t ≥ t\*) by t\*:**

  | t\* | 9 | 12 | 14 | 16 | 17 | 20 | 21 |
  |---|---|---|---|---|---|---|---|
  | D3-T | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | PAUSE-A | 0.42 | 0.31 | 0.19 | 0.05 | **0** | 0 | 0 |
  | PAUSE-B | 0.41 | 0.32 | 0.24 | 0.16 | 0.22 | 0.03 | **0** |

  - So every deadline at t\* ≤ 16 (A) or ≤ 20 (B) pays loss, by the H Prop 8(ii) argument.
  - The cap moves the zero-loss deadline to 17 (A) or 21 (B). Only C2 at n ≥ 15 can feel that, and it is read descriptively.
- **The maximum t + r on computing passes** is 9 (D3-T), 17 (A) and 21 (B).

**Prop D2 (pass index vs progress; no clock is given) [PROVED for the independence; COMPUTED for the information values].**
- **Information between pass index and progress:**

  | | D3-T | PAUSE-A | PAUSE-B |
  |---|---|---|---|
  | I(t; ρ) | 2.36 bits | 0.96 | 0.87 |
  | I(t; finished) | 0.281 | 0.179 | 0.213 |

  The PAUSE values are pass-weighted and pooled with the plain batches.
- **The marker carries almost no pass-index information:** I(t; M) = 0.036 bits (A) and 0.018 (B).
- **P ⟂ n** by construction, so the pause count leaks no size.
- **Coverage of the test diagonals** (computing pass t with r = n − t + 1 alive):
  - for n 9–12 at t ≥ 5 it rises from 0 to between 6.7k (n 12, t 5, r 8) and 57k (n 9, t 9, r 1) per 1M training items under A;
  - (t ≤ 4, r ≥ 9) stays 0 by D0. Those passes are already safe (h ≤ 0.02 in N1, and the fresh step is exact to m 16).

**Lemma D3 (why LONG-K gives SLVA no history gradient here) [PROVED (softmax identity) + DERIVED (quadratic model)].**
- **Identity:** ∂L/∂ℓ^h_{j,s} = α^h_{j,s} ⟨∂L/∂U_t[j]_h, N(V_s)[j]_h − U_t[j]_h⟩.
- **At an idle pass** whose post-completion versions are drift-free, the bracket is 0 for s ≥ n. For s < n it is (old − finished) content, and descent lowers those logits (anti-history).
- **With drift δ_s,** using ∇L ≈ H δ̄ near the optimum: ∂L/∂ℓ_s ≈ α_s δ̄ᵀH(δ_s − δ̄) = **O(‖H‖ δ²)**. The gradient points toward the least-drifted versions and vanishes quadratically as drift → 0.
- **D3-T:** acc_k24 is 1.000 at n 1–8, and training loss is 0.0000 at 48k. So the LONG-K history gradient is ≈ 0.
- **Consequence:** anti-drift reads, if needed at all, must be trained on **aged unfinished** states (pauses, all slots held), where δ and ‖H‖ are both larger.

**Lemma D4 (drift against the budget K) [PROVED UNDER a linearised copy-slot recursion δ_t = A_t δ_{t−1} + ε_t, with ‖A_t‖ ≤ ρ and ‖ε_t‖ ≤ ε].**
- **The bound:** ‖δ_t‖ ≤ ρ^t‖δ_0‖ + ε(1 − ρ^t)/(1 − ρ).
- **(a) Contraction (ρ < 1):** drift is bounded uniformly in K by ε/(1 − ρ). If that is below R's margin m, no drift failure occurs at any K.
- **(b) Near-carry (ρ = 1):** ≤ tε in the worst case, and √t·σ√D for i.i.d. zero-mean ε.
- **(c) An SLVA anchor weight α ≥ α₀** on a version with drift ≤ δ_a gives sup‖δ‖ ≤ δ_a + ε/α₀.
- **(d) [DERIVED from MEASURED]** Under (a)–(c) with ε independent of the state, failures would hit in-range items at the same t. They do not (stepcap copy 0.99998; acc 1.000 at t ≤ 8, n ≤ 8). So ε, if it matters, depends on the state (size or remaining work), which is a support effect. **The PASS-COUNT vs GRADUAL-DRIFT split of 213-18 is therefore a split in the *form* of one support failure** (over-reduction vs wrong content), and both are answered by coverage.

**Lemma D5 (TBPTT bias) [PROVED UNDER ‖∏ ∂V_u/∂V_{u−1}‖ ≤ ρ_b^{t−s} and per-pass gradient bound G].**
- **Setup:** g = Σ_t Σ_{s≤t} J_{t,s}. TBPTT-k drops lags ≥ k.
- **The bound:** ‖g − g_k‖ ≤ Σ_t Σ_{s≤t−k} Gρ_b^{t−s} ≤ K·G·ρ_b^k/(1 − ρ_b). Relative to the same upper bound on ‖g‖, that is ≤ ρ_b^k.
- **For near-identity copy dynamics** (ρ_b → 1) there is no geometric decay. The bias is then ≤ G·(K − k)(K − k + 1)/2: all the lost credit is long-horizon, preventive credit (drift never built).
- **Why it is still acceptable here [DERIVED]:** R2 sits at every pass, so the local corrective credit (a late pass learning to ignore age, or to read a cleaner version) survives truncation. Only "don't build the age signal" credit is lost, and either one removes the deadline.
- **At A's worst case** (Kb 20, k 12), only lags 12–19 are cut.
- **Variance** is not binding at K ≤ 24 with clipping at 1.0 [ASSUMED].
- **Literature [EMPIRICS, LIT_RAND_K]:** Huginn 2502.05171 uses k = 8 at r̄ = 32. 2604.21106 finds that truncation weakens loop training (φ 0.46 → 0.38), so full BPTT is preferred where it fits.

**(v) Cost and memory [MEASURED base; projected].**

| | D3-T (measured) | PAUSE-A | PAUSE-B |
|---|---|---|---|
| batch max Kb (mean) | 11.8 | 15.2 (per-batch p; per-item 17.9) | 17.9 |
| final-stage s/step (3090) | 0.407 (log 16k–48k); G-TIME 0.462 | ≈ 0.55 (plain 0.41, augmented ≈ 0.70) | ≈ 0.66 (augmented ≈ 0.9) |
| peak train GiB (bs 64, micro 1, bf16) | 11.28 | ≈ 19.6 at Kb 20 full BPTT [ASSUMED split: 2 GiB fixed + 0.58 GiB per pass + bank scores ∝ Σ(t+1)]; ≈ 13 with TBPTT 12 | ≈ 24.4 full (does not fit), so ≈ 16 with TBPTT 12 |
| 12k FT | — | 1.8 h + 0.4 h overhead → **$0.40–0.62** | 2.2 + 0.4 h → $0.47–0.68 |
| 48k cell | 5.2 h, $1.12 realised | ≈ 0.50 s mean × 48k + 0.5 h → **$1.30–1.87** | ≈ 0.65 s → $1.65–2.39 |

- The rates are $0.18–0.26/h.
- The FT-CTRL arm (no pauses) costs 12k × 0.407 s + 0.4 h → **$0.32–0.46**.

---

## 4. Goodhart routes

| route | how it would work | status | read by |
|---|---|---|---|
| **MARKER-GATED clock** (per pass) | count unmarked passes only, or "pause = identity", so age = ρ again | open (D0b) | **START-D shift:** median t_fin in bank index moves ≥ +3 with acc within ±0.05 → GATED |
| **History-gated deadline** | deadline active iff no pause seen yet; test histories are pause-free | open, zero-loss (D0b) | **PAUSE-UNLOCKS:** a test-time start pause (D 1 / 4) *raises* n 9–12 acc ≥ +0.10 |
| Content limit (ρ ≥ 8 with ≥ 3 left) | not a route; the support limit | irreducible here (D0a) | DEADLINE-GONE ∧ ¬DG with first errors at ρ ≥ 7 → CONTENT-LIMITED |
| Timed release (pre-compute during pauses, then multi-reduce) | hidden work while R2 holds | harmless at test (no pauses); could skew the pass function | in-range plain acc_n2 1.000, in-range GPROG = 0 |
| Clock at the new cap | t ≥ 17 (A) / 21 (B) ⇒ finished is zero-loss | open beyond the cap | C2 n 13–16 at K′ 24: t_fin pinned at 17 / 21 (descriptive) |
| Marker as a reset | π learned as "age := 0" | open | START-D shift (−D in computed-pass t_fin) |
| Copy-stall becomes trained-for | — | **closed:** computed pauses ≠ copy-stalls, so the registered STALL / NO-CLOCK stays untrained and decisive | VB-C1 |
| Pass-1 precompute, recompute from V_0, coda reading V_0 | — | as in §211 / §213 (coda V_0-free, pin 4) | PROBE, VALUE-SWAP, ONESHOT |
| Pause count leaks size | — | closed: P ⟂ n | — |

---

## 5. Readings (comparator-free, per cell)

**Conditions.** Local fp32, eval_bs 32, one local card for every D4 cell, §170 test set. The §213 contract files (eval_vb, steppos, lfk, rs, d3hist) plus eval_vbhalt, eval_stepbyt, eval_hist24 and a new **eval_pause.json**.

| reading | definition | role |
|---|---|---|
| **DEPTH-GENERAL** (d) | mean acc_n2 over n 9–12 at K′ n + 2 ≥ 0.80 (§213) | **primary** |
| **DEADLINE-GONE** | `d4_edge_collapse.py`, n 10–14 on-schedule events with ≥ 3 alive. Readable iff ≥ 300 events at t 7–9. Met iff h(7–9) ≤ 0.10 and median t_fin(n 12) − median t_fin(n 10) ≥ 2 | mechanism |
| EDGE-COLLAPSE | readable as above. h(7–9) ≥ 0.30, h(8) ≥ 2·h(5), and median t_fin over n 10 / 11 / 12 within 1 pass. N1 / N2: met (h 0.63 / 0.61; t_fin 9 / 9 / 9). The fixture passes (a content-rule trajectory reads not-met) | branch input; reported |
| LENGTH-FREE kept | NO-KNEE m 9 and m 10 ∧ LFK (§213) | kept |
| VB-C1 kept | §211 definitions (STALL untrained) | kept |
| K′24-STABLE kept | acc_k24 ≥ 0.99 at every n 1–8 | kept |
| PAUSE-LEARNED (gate) | eval_pause.pause_inv: n 1–8 with D 4 start pauses + 2 scattered (rng 21410), K′ n + 8 → pooled acc ≥ 0.99 | if not met, the mechanism readings are UNREADABLE |
| START-D (descriptive, pre-stated labels used only if ¬DEADLINE-GONE) | n 9–12 with D ∈ {1, 4} start pauses, K′ n + 2 + D: acc and alive trajectories | see the labels below |
| HISTORY-USED (f), CONTENT-STABILISES (213-21) | §213 / 213-21 | reported |
| C2 tail | acc at K′ 24 for n 13–16; acc_k rising past K′ n + 2 | descriptive |

**START-D labels:**
- **GATED:** t_fin shifts by ≥ +3 in bank index;
- **UNGATED:** t_fin shifts by ≤ +1 and acc falls by ≥ 0.05;
- **PAUSE-UNLOCKS:** acc rises by ≥ 0.10.

**Claims.**
- **D4-C2** = DEPTH-GENERAL ∧ LENGTH-FREE ∧ VB-C1 ∧ K′24-STABLE.
- **Mechanism:** DEADLINE-GONE ∧ PAUSE-LEARNED.
- **FT pair, descriptive:** AUG − CTRL on DG, h(7–9) and t_fin. One seed each, so this is not a rate.

**OC (one cell; rule a).**

| reading | P(pass \| no effect) | P(pass \| effect) |
|---|---|---|
| DEADLINE-GONE | ≤ 0.05 [ASSUMED: N1 / N2 h = 0.61–0.63; D3-T predicted EC] | ≈ 0.8 [ASSUMED] |
| DEPTH-GENERAL | ≈ 0.05 (0 of 18 VB-family / nested models; D3-T 0.480) | ≈ 0.35–0.45 *given the deadline is removed* [ASSUMED: the §190 analogue reached 0.659] |
| LENGTH-FREE kept | — | 0.85 [ASSUMED] |

- **Per FT-PAUSE cell:** P(deadline removed) ≈ 0.45, so **P(DG) ≈ 0.2** and **P(D4-C2) ≈ 0.15**.
- **Per 48k cell:** P(entry) 0.7 and P(DG | entry) ≈ 0.2.
- **Pre-stated OC shifts from Stage 0:**
  - AGE-CLOCK (S0-3) raises P(deadline removed) to 0.6;
  - PROGRESS-ONLY lowers it to 0.2, and the 48k cell is then not proposed;
  - D3-T ¬EC ∧ ¬ORD lowers it to 0.3.
- A positive is strong; a null is weak.

**Predicted per-n under DEADLINE-GONE [CONJECTURE]:** n 9 ≥ 0.97, n 10 ≥ 0.85, n 11 0.55–0.85, n 12 0.3–0.7.

**Censoring (rule b).**
- **Fails every reading:**
  - a dead run (log < steps);
  - **FT-DIVERGED** (final in-range acc_n2 < 0.99 at any n 1–8, or acc_k24 < 0.97);
  - NO-ENTRY (48k cell, answer_ce).
- **UNREADABLE = not met:**
  - DEADLINE-GONE or EC with < 300 events (never-reached events count as failures);
  - PAUSE-LEARNED < 0.95, which makes the mechanism readings UNREADABLE.
- **Counted as failures:** never-halting, stuck and unparseable decodes.
- **Basin entry:** the 48k entry step is reported against D3-T's 11k, and a gap > 5k is CONFOUNDED (descriptive comparisons only).

---

## 6. Experiments

### Stage 0 ($0, local, eval-only; needs Audit's OK; designed here, not run)

- **S0-1 DEADLINE-READ.** `d4_edge_collapse.py out.json boxes/runs_fsq/s213_d3t_s1 [s213_d3f_s1]` on the queued vbhalt + stepbyt outputs. It runs on CPU in seconds and feeds the branch rule (§1.3). The references are already computed (N1 / N2: EC; N1 ORS 0.69).
- **S0-2 CARRIER on D3-T** (3070, after the §213 chain, about 20 min). New part `prestall`:
  - s ∈ {4, 8} copies of V_0 stored before pass 1, using `rollout(prefix=[V0]*s)`; the carry stays V_0;
  - real passes n + 2, n 1–12;
  - acc, plus alive decodes giving t_fin in computed passes;
  - plus the existing `part_foreign`.
  - **Labels:**
    - BANK-BORNE iff the pooled n 6–8 drop at s 8 is ≥ 0.15, or the median Δt_fin (computed passes, n 10–12) is ≤ −5;
    - STATE-BORNE iff the drop is ≤ 0.05 and |Δt_fin| ≤ 1;
    - otherwise MIXED.
  - Descriptive. Computed pauses cover both carriers.
- **S0-3 AGE-REVERT on D3-T,** the premise test for computed pauses (one-directional, an untrained edit).
  - **The arm:** at passes 1..D, D ∈ {2, 4}, compute V_t, then revert the slots whose decode changed back to S_{t−1}[j] (`rollout(edit=...)`, part_hold's inverse). This advances the age of copy slots with content held. Normal passes follow, K′ = D + n + 2.
  - **Control:** n 1–4 at D 2 must stay ≥ 0.97, else UNREADABLE (the revert itself is off-distribution).
  - **AGE-CLOCK iff** the control holds, the pooled n 7–8 drop at D 4 is ≥ 0.20, **and** S0-2's prestall at s 4 drops n 7–8 by ≤ 0.05.
  - **PROGRESS-ONLY iff** neither arm drops by ≥ 0.10.
  - These labels feed the OC shifts above.
- **S0-4 schedule numerics:** done (`d4_schedule_stats.json`). The realised trainer schedules must reproduce them (G-P3).
- **S0-5 gates** for the new trainer code (Audit's diff review):

  | gate | what it checks |
  |---|---|
  | G-P0 | arch `SLVA`; the only new key is `pause_vec` |
  | G-P1 | π = 0 at step 0 ⇒ pause and plain passes are bitwise the base model's |
  | G-P2 | target-map unit test: E_{ρ_t}, HOLD weighting at pauses, R1 window [n + P, K] |
  | G-P3 | schedule statistics against S0-4: I(t; M) ≤ 0.05 bits, \|corr(P, n)\| ≤ 0.02, P(unfinished \| t ≥ 9) ≥ 0.40 |
  | G-P4 | c = 0 at every FT step |
  | G-P5 | the plain path (no `--pause`) bitwise against pre-edit code at a fixed seed (pa / carry / slva) |
  | G-P6 | mem_check at the forced worst case (Kb = cap, longest items, P = D = P_max) → full BPTT or TBPTT 12; the TBPTT gate: zero gradient across the cut, nonzero local gradient |
  | G-P7 | G-TIME on forced augmented batches |
  | G-P8 | ARGS-DIFF against the base args.json (only the §2.3 flags) |
  | G-P9 | the eval harness never applies π outside eval_pause |
  | G-P10 | gradflow: ∂L/∂π ≠ 0 on augmented batches, and SLVA / core / R nonzero, with decorators checked by ast |
  | G-P11 | an off-host checkpoint mirror every 2k steps plus a tested resume |

### Stage 1 (needs the user's OK)

| option | what | cap | answers |
|---|---|---|---|
| **1 (recommended)** | **1a:** FT pair from the base model.pt (FT-PAUSE-A or -B by §1.3, and FT-CTRL = the same FT without pauses), 12k each, RTX 3090, seed 1, one card class. **1b (conditional):** one 48k from-scratch cell = D3-T args + the branch's pause flags | 1a **$1.5** (projection $0.72–1.08 for A); 1b **$2.5** (A) / $2.8 (B); **family $4.0 (A) / $4.5 (B)** | 1a: is the deadline a removable support effect on the known-good weights? 1b: does the recipe give it from scratch? |
| 2 | 48k cell only | $2.5 | the recipe, with no attribution and the seed-lottery risk |
| 3 | FT-PAUSE only | $0.8 | the mechanism without CTRL (Adam re-init noise uncontrolled) |

**Why an FT first.**
- **The base already has every property but depth:** LENGTH-FREE, STEP-GENERAL, VB-C1, K′24-stable, entered at 11k. An FT isolates the one question (support coverage) from basin entry and the seed lottery.
- **FT-CTRL controls the FT itself.** Adam renormalises the base's near-zero gradients, so even a no-pause FT may drift.
- **It costs half as much and returns in about 2.5 h.**

**1b runs iff** FT-PAUSE reads either outcome below:
- DEADLINE-GONE: confirm the recipe from scratch;
- ¬DEADLINE-GONE ∧ UNGATED ∧ PAUSE-LEARNED: entrenchment, so test from scratch.

**1b does not run** under GATED (the marker route won), FT-DIVERGED or PROGRESS-ONLY.

**Pre-stated follow-ups:** CONTENT-LIMITED (DEADLINE-GONE ∧ ¬DG) → VB-2G G1's compaction + restamp (held), whose tail-lift theorem targets the ρ-visible content, layered on D4. GATED → no marker-based design; the deadline then needs an architectural counter-free carry (outside D4).

---

## 7. Top risks and falsifiers

1. **The marker route** (GATED or history-gated; D0b proves both are zero-loss), P ≈ 0.3. *Falsifier of the mechanism:* START-D GATED or PAUSE-UNLOCKS with ¬DEADLINE-GONE.
2. **The content limit** (D0a), P ≈ 0.5 given DEADLINE-GONE. The §190 precedent: counters removed, n 11–12 still failed. *Read:* CONTENT-LIMITED, then G1.
3. **D3-T is not EDGE-COLLAPSE** (P ≈ 0.25). *Premise falsifier for A:* S0-1 reads ¬EC ∧ ORS < 0.3 ∧ L ≠ PASS-COUNT-KEYED, which routes to B at lower odds. If S0-3 also reads PROGRESS-ONLY, D4 is not proposed.
4. **Entrenchment in the FT** (UNGATED): handled by 1b.
5. **In-range damage** (FT-DIVERGED, LFK regression): censored. Mitigations: lr 3e-5, warmup 500, and 50% plain batches.
6. **Memory or time overrun at Kb 20–24:** G-P6 → TBPTT 12, with Lemma D5's bound and the measured cost replacing the projection.
7. **The cap clock** (t ≥ 17 / 21) limits C2 at n ≥ 15 only (descriptive).

**Unverified in code or data (flagged).**
- `vbank_halt` on SLVA (assumed from `s213_post_halt.sh`).
- `lr_at` warmup semantics.
- The memory split (an extrapolation from 11.28 GiB).
- ORS mixes two conditionings: alive count vs strict decode.
- The CLOCK-KEY sign is subject to decode-noise selection.
- D3-T's EC / ORD are predictions until S0-1.
