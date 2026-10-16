# D. Depth invariance for VERSION-BANK: a self-similar, clock-free step rule (bottleneck 2, C2)

**Status.**
- Opus inventor subagent for Theory, 2026-10-16 03:11 EDT; LIT2 leads folded in at 03:40.
- A draft, not a registration. The repo was read only.
- Scripts and outputs ($0, CPU) are in `scratchpad/research2/D_scripts/`: `local_rule_cov.py`, `struct_cov.py`, `raw_span.py`, `raw_span_item.py`, with their JSON.
- **Tags:** [PROVED], [PROVED UNDER X], [DERIVED], [MEASURED] (my scripts or cited notes), [ASSUMED], [CONJECTURE].

## 0. Summary

1. **VB carries at least seven size or clock cues.** The structural one: in VB's raw slot layout the step's own neighbourhood grows with n. At ≥ 1 step of an item, the radius-2 neighbourhood of the leftmost-ready handle spans more raw slots than any training step does (33). That holds for 0.2 / 21 / 94 / 100 / 100% of items at n = 9 / 10 / 12 / 14 / 16 [MEASURED].
2. **SS-VB (SELF-SIMILAR VB)** is VB's CARRY base plus:
   - compaction (VOID slots dropped between passes) with re-indexed positions;
   - V_0 masked after pass 1;
   - a steep age floor on bank reads;
   - STALL augmentation (pass count decoupled from progress);
   - CANON in the main cell: each compacted version is pinned to the prelude's encoding of the remaining expression.
3. **Math.**
   - Feng's step is a radius-2 local rule plus one prefix-OR in compacted coordinates [PROVED; 0 / 852,056 steps].
   - Compaction makes the state after t steps the *same symbol sequence* as a fresh size-(n − t) problem [PROVED].
   - The requested lift ("exact at n ≤ 8 ⇒ exact everywhere") is **false** under compaction plus content-addressing alone (counterexample).
   - Corrected forms: a **tail lift** (Theorem 1) and a **full lift** under locality plus coverage (Theorem 2). For softmax heads the full lift reaches only n* ∝ e^Δ, unless the logits are log-n scaled.
4. **$0 diagnostic first: RESTART-SPLIT.** Re-root a decoded mid-chain state as a fresh problem and compare it with the intact continuation. Only genuine task inputs are used, so its control cannot break as 187-5's did. Two companions: PREFIX-FACTORIAL (clock vs position) and HIST-TRUNC (does bank history hurt?).
5. **First spend:** D1 (full SS-VB) plus D2, whose identity is fixed by a pre-stated rule on N1's HIST-TRUNC. About $0.9–1.9 per cell on a 3090.

---

## 1. Diagnosis

### 1.1 Size cues in VB

T = 2n + 3 + 2b slots (b bracket pairs). Training has T ≤ ~33 (n ≤ 8); the test goes to T ≤ 55 (n 16; 211-7).

| cue | where | reveals | train → test (n 16) |
|---|---|---|---|
| **C-pos**: absolute positions | the prelude adds wpe[0:T] (`gpt_lower`, deltamem l.266); every version inherits it | the " =" index ≈ 2n+2+2b. Learned APE "pins the trained head to a fixed absolute position inside the training range" (2607.18759 Prop 2) | p ≤ 33 → 55 |
| **C-gq**: gather queries W_q·wpe[j] (N2 / N3) | `VBankN.gather` | the absolute slot index | rows j > 33 never trained |
| **C-void**: VOID layout | E_t semantics | VOID count ≈ 2t + 2·(brackets removed): a **content clock** | t ≤ 8 → 16 |
| **C-gap**: operand distance | same | VOID run between the handle's operator and its far operand: mean 2.5 → 5.0, p99 18 → 40, max 26 → 52 [MEASURED] | — |
| **C-bank**: normaliser and history | `bias_row`: 6 / 12 heads at γ = 0, 4 more with slope ≤ 0.025 (pin 2) | Z_bank ∝ t·T, a pass clock and a length counter (PROOFS §5 (a)); dispersion Θ(1/n) (2410.01104); normaliser count leakage (2202.12172) | t ≤ 12 → 18 |
| **C-v0**: V_0 at constant β⁰ every pass | `bias_row` | the original size n | n ≤ 8 → 16 |
| **C-K**: pass budget | K_i = min(n + U{0..4}, 12) | every training pass t ≥ 9 is a HOLD pass. The reduction support is {t + r ≤ 8} (t done, r remaining). n ≥ 11 at K′ = n + 2 needs passes 13+ | passes ≤ 12 → 18 |
| **C-drift**: latent copy-age | 4 core blocks per pass on copied slots; R2 pins 20 linear classes only | "reductions done" in free directions: §190's "done + remaining > 8" limit | — |

**History is itself a known depth-extrapolation failure** [EMPIRICS, LIT2].
- Full cross-iteration KV caches pull trajectories back: CoTFormer 50.1% at depth 20 against 99.4% for a recent-1 cache, with recent-2..8 near chance (2609.19521).
- CLRS gains from dropping history (2403.04929).
- VB's bank, read at every core layer, is this pattern.

**Coverage** [MEASURED, `struct_cov.py`]. Numbers are abstracted to N (a∘b is a 484-entry table, covered at n = 1). Train is 60k items at n 1..8, every step; test is 300 per n, every step. Each cell gives the share of test windows of radius R (all slots / handle-centred) seen in training.

| layout, R, extra cue | n 9 | n 12 | n 16 |
|---|---|---|---|
| compacted, R 2 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| compacted, R 4 | 0.996 / 0.999 | 0.987 / 0.998 | 0.981 / 0.997 |
| compacted, R 6 | 0.883 / 0.946 | 0.803 / 0.898 | 0.738 / 0.852 |
| compacted, R 2, + t | 0.901 / 0.748 | 0.732 / 0.546 | 0.586 / 0.409 |
| compacted, R 2, + version length | 0.985 / 0.992 | 0.660 / 0.787 | 0.387 / 0.587 |
| raw VB slots, R 2, + version length | 0.953 / 0.944 | 0.024 / 0.014 | 0.000 / 0.000 |

A raw radius-2 window is "covered" but usually misses the operands.
- The raw span holding the handle's radius-2 compacted neighbourhood: mean 9.7 / 13.2 / 14.6, max 33 / 45 / 59 (train / n 12 / n 16).
- It exceeds the training maximum at ≥ 1 step in 94% of n 12 items and 100% at n ≥ 14 [MEASURED]. The cause: in the last reductions the neighbours are EOT and " =", so span = T.

**Reading [DERIVED].** A step that is local in compacted coordinates (R ≤ 4) and blind to t and length has training coverage of essentially every test state to n 16. Any dependence on t, length or raw distance puts 40–100% of test states outside the support. This is the quantitative form, for VB, of COUNT-OOD (187-3 / 187-6).

### 1.2 The $0 diagnostic

**Why 187-5 broke, and the design rule here.**
- 187-5 perturbed inputs into a region no training item occupies (position offsets), so even easy items left the support.
- **Here every arm feeds only genuine task inputs,** and arms compare two in-support computations.

**RESTART-SPLIT (RS; primary).** For a test item of size n, cut at t (remaining r = n − t):
1. Decode the state: argmax R(V_t) with VOIDs dropped (VB), or `aux_greedy` at pass t (INJECT §175 family). **Condition on an exact decode** (= cot_steps[t]).
2. **INTACT-CONT(n, r)** = P(correct at K′ = n + 2 | exact at t).
3. **RESTART(n, r)** = P(correct | the decoded expression is run as a fresh problem, K′ = r + 2). By Lemma 1 that input is a valid size-r item, so RESTART is the model's own in-distribution accuracy.
4. **LONG-STEP(m):** re-root after every pass. This gives per-step exactness on fresh versions with m operators and no history.

- **Cells:** n 9..16 × r ∈ {2, 4, 6, 8}.
- **Control:** n ≤ 8, every r. Both arms are in support there, so they must agree. A disagreement would mean "the state carries more than its decode", which is itself a finding, not a broken control.
- Unparseable decodes are reported as a rate, never silently dropped.

**Readings (descriptive, fixed now):**
- **HISTORY-KEYED:** pooled n 11–16, r ∈ {4, 6}: RESTART − INTACT-CONT ≥ 0.20, with the control |Δ| ≤ 0.03 and ≥ 100 conditioned items per cell.
- **HISTORY-FREE:** |Δ| ≤ 0.05.
- **HEAD-LIMITED:** LONG-STEP(12..16) < 0.95 while LONG-STEP(≤ 8) ≥ 0.99.

**Prior art.** Liang 2609.33144 is the causal precedent. Transplanting a short-prompt looped state into a long-prompt run takes same-hop accuracy from 5.5% to 83.6%. RS is the converse transplant, done in-support.

**PREFIX-FACTORIAL (PF).** It separates clock from position at a matched remainder.
- **Construction.** Take a remainder Q (r ∈ {2, 3}) whose first token is a number v, and replace v by a subtree P of value v with k operators (render() brackets it). The state at step k is then exactly Q [MEASURED: 20,000 / 20,000, k 1..12].
- **Two axes of natural items:**
  - k at fixed span s: the clock, at identical remainder positions and VOID count;
  - s at fixed k: position / length.
- The grid fills: at span 29, k = 9 / 10 / 11 have 463 / 452 / 107 of 20k draws.
- **Statistic:** S(k, s) = P(correct | exact at k), with fresh Q as the reference.
- **Labels:**
  - CLOCK-KEYED: at fixed s, S − S(0) ≤ −0.15 once k + r > 8, with |Δ| ≤ 0.03 for k + r ≤ 8 (the in-support control);
  - POSITION-KEYED: a slope in s ≤ −0.01 per slot at fixed k;
  - CONTENT-KEYED: both flat within 0.05.

**HIST-TRUNC (VB models only).**
- **Arm:** core and coda bank restricted to {V_0, V_{t−1}}, i.e. ages 1..t−2 masked, at K′ = n + 2.
- **Not the existing BANK-ABLATE.** `vbank_eval.part_acc` sets bank_off = True, which masks *every* bank entry (`bias_row` off) and leaves the gather untouched. So a new 5-line arm is needed.
- **HISTORY-HURTS:** n 9–12 gain ≥ 0.10 while n ≤ 8 changes by ≥ −0.03. The n ≤ 8 term guards against reading "history is used in-distribution" as harm.

**CLOCK-SLICE ($0, existing JSON).** `eval_vb.json` stepcap.by_t_a, and the 187-3 rows for INJECT.

**Plan.**
- RS / PF on §190 s1 / s2 and §193 s1 / s2 now; RS / PF / HIST-TRUNC on N1–N3 at landing.
- Eval-only, local, short, memory-checked, never as a co-tenant.

**Predictions [CONJECTURE].**
- §190 s1: HISTORY-KEYED (Δ ≈ 0.4 at n 12, r 4) and CLOCK-KEYED.
- §193: HISTORY-KEYED (token).
- N1: HISTORY-KEYED, partly HEAD-LIMITED, and HISTORY-HURTS (P 0.5).
- **Go rule:** if every model reads HISTORY-FREE + HEAD-LIMITED, the locality cell D3 (§6) runs instead of D1.

---

## 2. Mechanism: SS-VB

**Base.** VB CARRY (N1) exactly (prelude 0–5, core 6–9, coda 10–11, R1, reweighted R2, R4 curriculum, 48k × bs 64). Only the items below change.

**Notation.** V̂_s is compacted version s (m_s slots); o_s(j) is slot j's original index; c_t is the number of reductions done after pass t.

**(M1) Compaction.**
- After pass t writes V_t (m_{t−1} slots), compute κ_t[j] = 1[E_{c_t}[o_{t−1}(j)] ≠ VOID] in training, or 1[argmax R(V_t[j]) ≠ VOID] at evaluation.
- Then V̂_t = V_t[κ_t] and o_t = o_{t−1}[κ_t]. The bank stores V̂_t and its per-layer K/V, with gradient.
- A keep error at evaluation is a step error.

**(M2) Re-indexed stamps.**
- The pass input is U_{t+1}[j] = N(V̂_t)[j] + a ⊙ (wpe[j] − wpe[j + δ_t(j)]).
  - δ_t(j) ∈ {0, 2, 4} is the slot's shift at this compaction (0 left of the handle).
  - a ∈ ℝ^768 is initialised to 1.
- This exactly shifts the residual's direct embedding component, so a compacted state carries the stamps 0..m_t − 1 of a fresh problem.
- N(V)[j] = g ⊙ V[j]·ρ_{cls(j)}/rms(V[j]), with ρ_cls the init-time mean rms of V_0 slots per class (EOT, " =", number, op, bracket). It replaces pin 5's history-dependent rms(V_0[o(j)]).

**(M3) Bank logits (core layer ℓ, head h, query j, pass t).**
- Current version (causal): GPT-2's q·k/8.
- Latest version V̂_{t−1} (bidirectional): q·k/8 + β_{ℓh} + ω_{ℓh}[clip(i − j, ±8)], with ω initialised to 0.
- Older versions: q·k/8 + β_{ℓh} − μ_{ℓh}(t − 1 − s), with **μ_{ℓh} = 2 + softplus(θ_{ℓh})** and no positional term.
  - The floor of 2 (e^{−2} per version) follows LIT2's history evidence: recent-1 ≫ recent-2..8.
- **V_0 is the latest entry at t = 1, and masked for t ≥ 2.**
- The coda uses the same rule over V̂_1..V̂_{t−1}. β is recalibrated (pin 3).
- **At K = 1 nothing changes** (no compaction before pass 1, δ = 0, ω = 0), **so G1 function preservation holds** [PROVED: pass 1 = VB CARRY's pass 1].

**(M4) STALL-AUG and content-defined windows.**
- In training, for t ≥ 2 with p_s = 0.1, pass t is computed and gets its R2 loss, but the stored entry is V̂_t := V̂_{t−1} and c_t = c_{t−1}.
- τ_i = the first t with c_t = n_i (the state is [EOT, v, " ="]); K_i = τ_i + U{0..4}, with no cap at 12.
- R2 targets are re-indexed: slot j of V_t is decoded against E_{c_{t−1}+1}[o_{t−1}(j)].
- This removes the Kuo 2606.29983 size ↔ loop correlation and Fan 2409.15647's need for the ground-truth step count at the source.

**(M5, D1) CANON.** L_can = (1/m_t) Σ_j ‖N̄(V̂_t[j]) − sg N̄(Φ_t[j])‖²/D, for t = 1..K_i.
- Φ_t = prelude(wte[frame(cot_steps[c_t])] + wpe[0:m_t]) is the current prelude's no-grad encoding of the **remaining** problem as a fresh input.
- N̄ is per-slot RMS normalisation, and λ_can = 1. An EMA prelude is the fallback if the targets oscillate.
- It pins the 748 directions R2 leaves free, which is where C-drift lives.

**Total loss:** L = L_ans + L_rec + λ_can·L_can.

**Held back, for D3 (if HEAD-LIMITED):**
- **(M6a) LOC:** sink + |i − j| ≤ 4 bands on 10 of 12 core heads, 2 global heads.
- **(M6b) STRUCT-PE**, Ito 2608.31067's construction in VB form. A relative bracket-depth bias ω_d[clip(d_i − d_j, ±4)], with d computed as the prefix bracket sum of R(V̂_{t−1})'s decode.
  - Ito: bracket-depth positions plus a hardmax deepest-span mask, trained at depth 1–2, exact to depth 30. A Fan-style looped transformer is at chance by depth 3, and RoPE / FIRE collapse.
- **(M6c) NDR-style** geometric (nearest-match, unnormalised sigmoid) attention plus a copy gate (2110.07732: nested mod-10, depth 5 → 7–8 at 0.98 vs 0.47).
- **(M6d) log-n logit scaling** on the global heads (2202.12172's fix; SSMax 2501.19399; see Lemma 2′).
- Prelude position jitter (Ruoss 2305.16843) needs its own precondition gate: §177's Abacus offset erased pass identity.
- **PURE-ANNEAL** is not used: its wpe[j] queries are cue C-gq, and its anneal cost N2 a 2k-step bump (211-11). A relative gather (ω on the latest version, content on older ones) is the constraint-faithful follow-up.

**Constraint check.**
- Every version stays in an append-only bank, readable at every core layer and the coda by content. That is pass-global in reach: an age-a version needs about a·μ nats of content advantage to be retrieved.
- V0-OFF loses no information, since V̂_1 holds the whole remainder.
- A training-time latest-only read (W = 1) keeps the bank architecturally but restricts its use. It is offered only as D2's conditional variant (§6). That is the user's call via Audit.

**Plug-in.** A new `boxes/ssvb.py` subclasses VBankN (gather_keep, restamp, the new `bias_row`, stall draws, CANON targets) with a `vb_loss` wrapper. There are no edits to vbank.py or the plain paths, and the vbank_eval parts are reused with compaction applied.

---

## 3. Mathematics

**Setting.** A state z is [EOT, expression, " ="], compacted. S* is Feng's step: reduce the leftmost node with two leaf children, then re-render. The model's pass is F(current, bank) followed by compaction C.

**Lemma 0 (local rule) [PROVED; VERIFIED 0 / 852,056 steps on 100k items, n ≤ 16].**
- *Statement:* operator token i roots a ready node iff:
  - z[i ± 1] are numbers;
  - L = z[i−2] ∈ {EOT, "("} or prec L < prec z[i];
  - R = z[i+2] ∈ {" =", ")"} or prec R ≤ prec z[i].

  The handle is the smallest such i (a prefix-OR). The rewrite is "x op y" → v, or "( x op y )" → v when L = "(" and R = ")".
- *Proof.*
  - (⇐) If the left child were internal, x would end its unbracketed right spine, whose operator z[i−2] has prec ≥ prec(op) by the bracket rule. That contradicts L. The right side is symmetric: an unbracketed right child needs strictly higher precedence.
  - (⇒) The neighbour tokens are "(", EOT / " =", or the operator of the first ancestor reached through an unbracketed spine. The spine inequalities give the two conditions.
  - Ready nodes are disjoint, and `_leftmost_ready` is in-order. ∎

**Lemma 1 (self-similarity) [PROVED].** After t exact steps the compacted state is frame(render(τ_t)), with τ_t the reduced tree.
- τ_t lies in Alg. 1's support at size n − t: Alg. 1 reaches every shape and consistent labelling, because PAIRS lists all preimages.
- cot_steps(τ_t) = cot_steps(x)[t:].
- *Proof:* the E_t re-render gate (0 diffs in 100k), plus determinism of cot_steps. ∎
- Everything that distinguishes "step t of x" from "step 0 of τ_t" is therefore history: the bank, hidden features, and stamps.

**Counterexample to the requested lift [PROVED].**
- Let F be an exact step, and let F′ = F except that F′ outputs garbage whenever m > 33 slots.
- m is computable with content-only softmax: the sink share is e^{ℓ_s}/(e^{ℓ_s} + m).
- F′ is compacted, content-addressed and exact on every state at n ≤ 8, and it fails at n ≥ 16. ∎

**Assumptions.**
- **A1 exact compaction.**
- **A2 RP:** logits are functions of (h_i, h_j, clip(i − j), age) only.
- **A3 CS:** on exact chains, V̂_t = Φ(z_t), the prelude's fresh encoding.
- **A4 HF:** older versions contribute ε_B = 0.
- **A5 LOC_R + COV_R:**
  - next-slot vectors depend only on radius-R windows plus q global bits computed by hard aggregation;
  - every reachable structural window, with its bits, at any n occurs at n ≤ 8.

**Theorem 1 (tail lift) [PROVED UNDER A1–A4].**
- *Claim:* if the model is exact on every problem of size ≤ m (steps, answer, H HOLD passes), then every size-n problem whose first n − m steps are exact is exact thereafter, at every K′ ∈ [n, n + H].
- *Proof.* After n − m steps:
  - Lemma 1 and A1 give the symbol sequence of τ_{n−m};
  - A3 gives vectors Φ(τ_{n−m}), which is pass 1's input for that fresh problem;
  - M2 / A2 give the same stamps;
  - A4 removes the older bank.
  - No pass-indexed input remains, so the continuation is bitwise the fresh run. ∎
- *Quantitative form* [PROVED UNDER per-pass Lipschitz Λ, readout margin γ]. With ‖V̂_t − Φ(z_t)‖ ≤ ε_C and a per-pass bank perturbation ≤ ε_B:
  - δ_j ≤ Λ^j ε_C + ε_B(Λ^j − 1)/(Λ − 1);
  - if the core contracts toward canonical states (Λ < 1, CANON's fixed point), δ ≤ ε_C + ε_B/(1 − Λ) **uniformly in j**. That is §118's contraction lesson (ρ ≤ 0.28 ⇒ no-snap exact).

**Lemma 2 (dilution) [PROVED].**
- *Statement:* let keys split into a relevant set N and a far set F, with ‖v‖ ≤ B, Z_N ≥ e^λ and ℓ ≤ λ − Δ on F. The output then lies within 2Bε/(1 + ε) of the N-restricted softmax, where ε = |F|e^{−Δ}.
- *Proof:* α_F ≤ ε/(1 + ε) and o − o_N = α_F(ō_F − o_N). ∎
- With finite features Δ cannot grow without bound, so dispersion is unavoidable (2410.01104 Thm 2.2), and any fixed softmax model has a finite size horizon.

**Lemma 2′ (log-n scaling) [PROVED UNDER Lemma 2's margin].** With logits scaled by κ(m) = log m / log m₀ (SSMax-style), ε(m) = m^{1 − Δ/log m₀}, non-increasing in m once Δ ≥ log m₀. The price is that m enters the computation by design, monotone and in a fixed form.

**Lemma 3 (margin transfer) [PROVED UNDER A2 + A3 with a local Φ + COV].**
- Far-key logits are drawn from a finite trained set of (window, window, bucket, age) types, so Δ_test = Δ_train.
- Under learned APE, heads pin to trained absolute positions (2607.18759 Prop 2), and no transfer follows. 187-5's entanglement is the symptom.

**Lemma 4 (age floor) [PROVED].**
- *Statement:* with old-entry logits ≤ λ_c, slope ≥ μ, and versions of ≤ T_0 slots, adding or removing every version older than A moves a head by ≤ 2B·T_0·e^{λ_c − λ − μA}/(1 − e^{−μ}).
- *Values:* at μ = 2 and A = 3, that is ≈ 0.0029·T_0e^{λ_c − λ}. At pin 2's slopes 0 / 0.0039 / 0.0098 the bound is vacuous, which is cue C-bank [DERIVED].

**Theorem 2 (full lift) [PROVED UNDER A1–A5, hard global aggregation].**
- *Claim:* every step of every reachable state at every n is exact.
- *Proof:* by induction on t.
  - The state is exact, so by A3 its vectors are Φ(z_t).
  - By A5 the next vectors are a fixed function of a covered window and the global bits.
  - By Lemma 0 the true step is such a function (radius 2 plus one prefix-OR), so exactness on the training windows gives exactness on every covered window.
  - A1 then gives z_{t+1} exact. ∎
- With handle-window COV measured at ≥ 0.997 (R ≤ 4, n ≤ 16), the lift is total up to that tail.
- This is Cai 1704.06611's finite-verification pattern (a bounded, size-independent step domain plus traces), here made possible by compaction.
- *Corollary (softmax globals) [PROVED UNDER A1–A5].* With q softmax heads at margin Δ in place of the hard OR, exactness holds to m* = γe^Δ/(2BqΛ_R): **each e-fold of length costs one nat of margin.**
- *Prediction:* far-key mass at n ≈ (m_n/m_8) × its value at n 8 (MARGIN-SCALE).

**Not proved.**
- A3 (CANON approximates it; ε_C is measured).
- A4 (bounded by Lemma 4; ε_B is measured).
- A5's LOC for unbanded GPT-2 heads (attention mass beyond |i − j| > 4 is measured).
- Whether M2 moves the derived position features as well as the direct one [ASSUMED; POS-PROBE].
- LIT2 (§D): no proof exists for any SGD-trained looped step. Guarantees come only from finite verification, from matching a proved construction (Ito), or from idealised learners. This section is of the first kind.

---

## 4. Goodhart table

| route | closure | caught by |
|---|---|---|
| Pass counter (bank size, ages, V_0) | STALL-AUG makes "reduce while t ≤ n" lose on-distribution: t ≠ c_t on a positive-probability set [PROVED]. V0-OFF removes the fixed-origin clock; Lemma 4 | PF (k), CLOCK-PROBE |
| Content clock (VOID count, copy-age) | Compaction removes the VOID count [PROVED]. Under A3, V̂_t = Φ(τ_t) carries no c_t [PROVED UNDER A3]. **Open in a no-CANON D2** | CLOCK-PROBE (decode c_t at matched τ_t), RS |
| Length / absolute position | Compaction plus restamp give the fresh stamps (direct component) [PROVED]. The first n − 8 steps still see long versions | PF (s), LONG-STEP, MARGIN-SCALE |
| History pull-back (2609.19521) | Age floor μ ≥ 2; the W = 1 variant | HIST-TRUNC, RS |
| Pass-1 precompute | L1 / CAP-aff. CANON's target is the fresh encoding, so V̂_1 holds only the prelude's own ≤ c₀ levels | PROBE(1) / (2) |
| Input recompute | V0-OFF: no path from V_0 to pass ≥ 2 except via V̂_1 [PROVED] | VALUE-SWAP (ii) own |
| Literal copying | Required. Under CANON it is the fixed point, not a bypass | R2 split |
| Coda solving | As VB (≤ c₀ levels) | PROBE vs coda |
| NEW stall detection | Intended behaviour, but **it voids VB's STALL / NO-CLOCK**; PF and RS replace it | — |
| NEW CANON shortcut / collapse | Φ_t's value slot needs v computed. R2 and the answer need information; rank monitor; EMA fallback | SERIAL-STEP, effective rank |
| NEW keep errors | Teacher κ in training gives no incentive; a keep error counts as a failure | KEEP-ACC by n |
| NEW sink counting (2202.12172) | Bounded by Lemma 2; M6d if needed | MARGIN-SCALE |

---

## 5. Decisive readings (comparator-free, per cell), OC block, censoring

All readings are local fp32 at eval_bs 32, on the training card's class and the §170 test set (1,000 per n), with R's own keep decisions.

1. **DEPTH-GENERAL** (§170's C2 rule, unchanged): mean n 9–12 at K′ = n + 2 ≥ 0.80. Every nested model so far is ≤ 0.688.
2. **SELF-SIMILAR** (RS on this model): n 9–16, r ∈ {2, 4, 6, 8} pooled, |RESTART − INTACT-CONT| ≤ 0.05; ≥ 100 conditioned items per cell; control within 0.03.
3. **STEP-FLAT** (per-step accuracy at matched local context vs n):
   - On states with an exact input, a step is correct if the next compacted state is exact.
   - Stratify by the handle's structural radius-4 window, and reweight n 13–16 to the n ≤ 8 window mix.
   - Pooled per-step accuracy must be ≥ the n ≤ 8 value − 0.02, and ≥ 0.98.
4. **Carried over from VB-C1:** PROBE-GAIN and SERIAL-STEP (the depth must be serial use). NO-CLOCK is dropped.

**SS-C2 = 1 ∧ 2 ∧ 3 ∧ PROBE-GAIN ∧ SERIAL-STEP.**

**Labels:**
- HISTORY-KEYED / HEAD-LIMITED / HISTORY-HURTS / CLOCK-KEYED / POSITION-KEYED;
- BEYOND-BUDGET: pooled n 13–16 at K′ = n + 2 ≥ 0.5. Each such item needs ≥ 13 passes, more than any trained count; this is NORTH_STAR C2's "more latent steps past the trained budget";
- NO-ENTRY and LOSES-ONE-PASS (as VB).

**Descriptive:** CANON-GAP ε_C(n, t), CLOCK-PROBE, POS-PROBE, MARGIN-SCALE, KEEP-ACC, attention mass beyond |i − j| > 4.

**OC (one seed per cell) [ASSUMED, from the project's history].**
- **No effect:**
  - P(DEPTH-GENERAL | size-keyed) ≈ 0.05: 0 of 14+ nested models reached 0.80; the best is 0.688.
  - P(SELF-SIMILAR | history-keyed) ≈ 0.05.
  - P(STEP-FLAT | size-keyed step) ≈ 0.1.
  - Jointly, P(SS-C2 | no invariance) ≤ 0.02.
- **Effect:** P(SS-C2 | invariant step learned) ≈ 0.6 (losses from dilution at long versions and from ε_C).
- P(learned | entry) ≈ 0.5 and P(entry) ≈ 0.65, so per-cell P(SS-C2) ≈ 0.2.
- A positive is strong; a null is weak. A rate claim needs ≥ 3 seeds and the user's OK. The realised $ replaces the projection.

**Censoring (rule b):**
- A dead run (< 48k) or NO-ENTRY fails every reading.
- Never-completed items, keep errors and unparseable decodes are failures.
- RS / PF cells with < 100 conditioned items are UNREADABLE = not met.
- STEP-FLAT strata with < 30 states are dropped; if more than 20% of n 13–16 states are dropped, UNREADABLE.
- Event-only values are reported beside.

---

## 6. Minimal first experiment

**Stage 0 ($0, local).**
- **(a) The §1.2 battery.** RS / PF on §190 s1 / s2 and §193 s1 / s2 now; RS / PF / HIST-TRUNC on N1–N3 at landing. The harness is about 150 lines (Experiments).
- **(b) Gates for `boxes/ssvb.py`:**
  - G-C1: teacher compaction = frame(cot_steps[c_t]) on 100k items, stalls included.
  - G-C2: K = 1 equals VB CARRY in fp64 (≤ 1e-9).
  - **G-C3, a code-level Theorem 1 check:** with V̂_t := Φ_t substituted and older versions masked, the continuation equals the fresh run on τ_t in fp64. That proves there is no hidden pass index in the code.
  - G-C4: path-specific gradflow (L_can into core 6–9, L_ans, L_rec, bank K/V; decorators checked by ast).
  - G-C5: G1-cal at μ ≥ 2 (m₉₉ ≤ 0.02).
  - G-C6: mem_check at n 16, K′ 18.
  - G-C7: plain-path smoke.
  - G1b 2k drift smoke; G-TIME on a forced final-stage batch.

**Stage 1 (RTX 3090, seed 1, 48k, n 1..8 train, §170 test set, one GPU class, off-host checkpoint mirror).**

| cell | design | question |
|---|---|---|
| **D1** | SS-VB: M1–M5 | Does a self-similar, clock-free version bank give SS-C2? |
| **D2**, fixed by rule before N1's HIST-TRUNC is read | **if N1 reads HISTORY-HURTS:** D1 with the training read restricted to {V̂_{t−1}} (W = 1). **Otherwise:** D1 with λ_can = 0 | W = 1: is a recent-1 read (2609.19521) needed beyond the age floor? λ_can = 0: must canonical states be enforced? |

- **Launch rule:** D1 launches when Stage 0(a) on the INJECT models does not read all-HISTORY-FREE (§1.2's go rule). N1 is a descriptive in-family reference.
- **Cost.** VB measured 0.28–0.48 s/step (launch-bound, 211-3). Compaction halves the mean version and bank sizes, CANON adds one no-grad prelude per version, and stalls add about 10% passes, so assume VB's rate [ASSUMED; G-TIME decides]. That is 3.7–6.4 h, plus about 1 h overhead, at $0.18–0.26/h: **$0.9–1.9 per cell; pair $1.8–3.8.** Proposed caps: per-cell admission ≤ $2.5, family ≤ $5, with the user's OK via Audit.
- **Predictions [CONJECTURE].**
  - D1: P(entry) 0.65, P(DEEP) 0.5, P(DEPTH-GENERAL | DEEP) 0.5, P(SS-C2) ≈ 0.2. DEPTH central 0.80 if DEEP; n 13–16 about 0.55.
  - D2 (λ = 0): HISTORY-KEYED via copy-age, P 0.5.

**What follows.**
- **D1 SS-C2:** seed 2; the relative-gather PA variant; GSM8K "compaction" (dropping consumed quantity slots), open and needing its own derivation.
- **HEAD-LIMITED:** D3 = D1 + M6b STRUCT-PE (Ito-style, the strongest prior on this exact task family) or M6a LOC; M6d if MARGIN-SCALE shows dilution.
- **TRUNC-LONG** (R2-only training on the first ≤ 8 steps of n 9–16 items) is a fallback that changes the C2 wording to "serial depth beyond the trained chain length, with long inputs seen". It is never used for the pure claim.
- **PARALLEL-READY targets** (reduce every ready handle per pass) are exactly local with no OR [PROVED: ready handles are disjoint], and their pass count is about tree height: mean 3.3 / 5.0 / 6.2 / 7.0 at n 4 / 8 / 12 / 16, max 8 at n 8 [MEASURED]. Most n 16 items then need no more passes than training ran, so they test length more than depth. Held back.

---

## 7. Top risks and falsifiers

1. **Optimisation (NO-ENTRY / LOSES-ONE-PASS).** N3's per-version teacher never recovered (211-12), and CANON is a 768-d co-trained target. *Falsifier at this budget:* D1 and D2 both NO-ENTRY while N1 entered.
2. **HEAD bottleneck.** The first n − 8 steps see long versions with unbanded heads. *Falsifier for "self-similarity suffices":* SELF-SIMILAR holds while DEPTH-GENERAL fails, with failures at versions of more than 8 operators.
3. **A3 not reached.** *Falsifier for the CANON lever:* D1's CLOCK-PROBE ≥ chance + 0.2, or D1 ≈ D2 (λ = 0) on RS.
4. **The bank re-injects history.** *Falsifier for Lemma 4's sufficiency:* HISTORY-KEYED in D1 with CANON-GAP ≤ 0.05 and a large HIST-TRUNC gain.
5. **Keep errors.** KEEP-ACC < 0.995 at n 13–16 caps depth by itself.
6. **Constraint acceptance.** CARRY + V0-OFF + a steep age floor (or W = 1) may read as "less attention-like". It is flagged; the PA-relative variant is the remedy.
7. **Wrong target (premise falsifier).** Stage 0(a) reads HISTORY-FREE + HEAD-LIMITED everywhere, or N1's RESTART − INTACT-CONT ≤ 0.05 at n 12, r 4. Self-similarity then fixes nothing that matters, and D3 comes first.

**Project sources:**
- 187-3 / 187-6 (COUNT-OOD), 187-5 (CONTROL-BROKEN), 187-6a / 190-2 / §190 / §194 (counters removed; the content limit; nesting ≤ 2 scope), §193 (0.688 / 0.647).
- PROOFS_v0 §5, 211-1 pins, 211-3, 211-11 / 211-12.
- §64–§66 / §71 / §106-k2: a restart from a compact state with the history cut extended range on S5 and multiplication. SS-VB builds that cut into the architecture.

**Literature (as cited in LIT2 / the repo's LIT docs):** Ito 2608.31067; Csordás NDR 2110.07732; Liang 2609.33144; Kavuncu 2609.19521; Bohde 2403.04929; Singh 2607.18759; Veličković 2410.01104; Chiang 2202.12172; SSMax 2501.19399; Cai 1704.06611; Ruoss 2305.16843; Zhou 2402.09371; Kuo 2606.29983 (the project's randk, P22, did not help); Fan 2409.15647.
