# PROOFS2 v0: rigour pass on VB-2 (SYNTH2_spec_v0)

*Opus proof subagent for Theory, 2026-10-16. The repo was read only. CPU checks (1 thread; the probe used 3), at idle priority, are in `scratchpad/research2/proofs2/`:*
- *`p2_checks.py` (+ .json): Lemma 0 per operator, the in-place radius, the re-render gate, the reduced-tree law, the STALL-AUG bands;*
- *`p2_statecov.py`, `p2_factored.py`: state-level coverage;*
- *`p2_posprobe.py`: GPT-2 derived position features;*
- *`ident/`: a re-run of H's ident_check.*

*Note 205-5 is applied: HH-FIT is not used as evidence anywhere below.*

## Verdict table

| # | Claim | Verdict | Fix |
|---|---|---|---|
| 1 | D Lemma 0: the leftmost-ready step is a radius-2 rule plus one prefix-OR (compacted) | **[PROVED]**. 0 mismatches in 1,024,297 operators (170,480 states, n ≤ 16), covering per-operator readiness, bracket status and leftmost choice | "Radius 2" is the operator predicate. The **per-slot in-place write is radius 4**, and that is tight (conflicts: R2 89,146, R3 35,853, R4 0), with b_j = OR ready(i′ ≤ j−3) |
| 2 | D Lemma 1: self-similarity | **[PROVED]** (support, cot suffix, re-render 0 / 5,000 items) | τ_t is in Alg. 1's *support*, not its *law*. At r 4 from n 12: height 3.70 vs 3.33, brackets 1.87 vs 1.50. Say "in-support", not "in-distribution", for RESTART |
| 3 | Counterexample to the naive lift | **[PROVED]** | It also defeats A2 and CANON. A5's global features must be existential (prefix-OR), never counts |
| 4 | A2 (logits depend on content, clipped relative position and age) is achievable in GPT-2 with M2 | **[FALSE]** | wpe sits inside h. After removing wte + wpe, position is still linearly decodable from h_6 (held-out R² 0.9986, MAE 0.34 slots), and that code does not extrapolate past slot 33. A2 fails even on fresh problems. Drop it from Thm 1; fold it into A5′ for Thm 2 |
| 5 | M2 "exactly shifts the direct embedding component" | **[FALSE as stated]** | N rescales per slot, so s⊙c − a of wpe[p] remains. Exact only with a tracked per-slot coefficient, and Thm 1 does not need exactness (§2.1) |
| 6 | Thm 1's proof chain (A3 vectors, plus M2 stamps) | **[FALSE: M2 and A3 are inconsistent]** | The bank stores the un-restamped V̂_t while the input is restamped. No V̂_t matches both the fresh bank entry and the fresh input for shifted slots, so **G-C3 fails by construction**. Fix: restamp the *stored* entry Ṽ_t, and put CANON on Ṽ_t (A3′) |
| 7 | D M3: G1 holds at K = 1 | **[FALSE under M2's N]** | U_1 = g⊙V_0·ρ_cls/rms(V_0[j]) ≠ V_0. Fix: r(j) = rms(Φ(current content)[j]), which is the identity at pass 1 and consistent with Thm 1 |
| 8 | D Thm 1 (tail lift) | **[PROVED UNDER A1, A3′, A4 exact, history-free N and restamp]** | Holds for m ≥ 2 (the coda differs at m = 1). The quantitative form needs δ below the readout margin |
| 9 | D Thm 2 (full lift) | **[PROVED UNDER A1, A3′, A4, factored A5′]** | Under a monolithic radius-4 slot rule, state coverage is **0.93 → 0.71** (n 9 → 16), so "total up to 0.997" is false. Factored keys (248 / 169) cover **1.000** of states at n 9–16 |
| 10 | Corollary m* = γe^Δ/(2BqΛ_R) | **[PROVED UNDER first-order composition]** | — |
| 11 | D Lemma 2 (dilution) | **[PROVED]** | — |
| 12 | D Lemma 2′ (log-n scaling) | **[PROVED UNDER a max-logit margin]** | Z_N ≥ e^λ is not preserved by scaling. Monotone iff Δ ≥ ln m₀ (3.5 nats at m₀ = 33) |
| 13 | D Lemma 3 (margin transfer) | **[PROVED UNDER pair coverage + local Φ]** | Single-window COV is insufficient, and GPT-2's prelude is not local |
| 14 | D Lemma 4 (age floor) | **[PROVED]**; the numeric value drops 2B | 0.0029·T₀e^{λc−λ} → **0.0057·B·T₀e^{λc−λ}**. At μ = 2 and zero content gap, ages 1–3 are *not* negligible (§3) |
| 15 | S C1 (COMMUTE is Markov) | **[PROVED]** as σ(dec V_{t−1})-measurability on matched skeletons | "Requires V_0 or older versions" → "requires information beyond dec V_{t−1}", which includes V_{t−1}'s null space. The forcing set vanishes as errors → 0 |
| 16 | S X1 (what L_X = 0 forces) | **[PROVED on the OPX support]** | V_0-invariance is moot under V0-OFF; natural transfer stays [OPEN] |
| 17 | COMMUTE × compaction × stalls × OPX | **Well-defined under 4 conditions** (§4) | Index by the original slot o. OPX by reduction index. L_X's other slots → Ẽ. **CANON's truth target conflicts with COMMUTE** → Φ(dec) |
| 18 | S race model (drift signs) | **[PROVED UNDER common-p]**. The sign is exact near uniform p (early) | Exact CE gives event-weighted drift. π_s, "recompute right iff hh ≤ c₀", is **[ASSUMED]** (205-5) |
| 19 | S stationary transfer (S1–S3) | **[PROVED UNDER S1–S3]**, with S2 strengthened | The full law of LN(V_{t−1}) given dec, not only the null-space part. Windows in compacted coordinates; covered states only |
| 20 | S product bound | **[PROVED]** (equality for all-correct) | Absorbing up to ×0 / ÷ coincidences |
| 21 | S drift (κ > 1 ⇒ t*) | **[DERIVED; needs a lower expansion bound]** | ‖A‖ ≤ κ bounds growth from above only |
| 22 | S coverage t ≤ 9 − r + O; 76.2% | **[PROVED]** (32/42) | — |
| 23 | H Prop 1 | **[PROVED]**; holds in VB-2 | — |
| 24 | H Prop 2 | **[PROVED]**; counts reproduced (4,926–5,053; 0; 0; 0; 1.000) | A compacted read leaves g(VOID) unidentified, but VOID is absent there |
| 25 | H Thm 3 | **[PROVED]** | Strengthen: *skeleton* exactness suffices for timing |
| 26 | H Prop 4a–e | **[PROVED]** | 4c's threshold is 7.6 nats at T = 55 |
| 27 | H Prop 5 | **[DERIVED]**; the closed forms are slightly off | P(t_halt = n) = (1−σ−δ)^{n−1}(1−σ)(1−ν), which matches the simulation (0.633 / 0.505 vs 0.635 / 0.507) |
| 28 | H Prop 8(i, ii) | **[PROVED]**, but incomplete | Under H's form, **"t ≥ 11 ⇒ hold" is exactly zero-loss** (max reduce pass 10 in 2×10⁵ items). The duplicate-corrected count rule K* is zero-loss under both forms |
| 29 | H Prop 8(iii) band | **[PROVED for rules f(content, t)]** | H: a + t ≤ 11. D: unbounded support with geometric weight (table in §6). "Cap 12 vs 12 + S" is immaterial: the cap never binds |
| 30 | V0-OFF closes input recompute | **[PROVED with M2's N]** (pin 5's N leaks V_0) | Two routes remain: pass-1 precompute in V̂_1, and older-version recompute at ≥ μ(h−1) nats for a height-h handle (§7) |
| 31 | Clock inventory | §8 | **Decisive held-out clock test: PF's k-axis** (with a HIST-TRUNC split), plus RS. REWIND is descriptive (FROZEN-1 only). Triple stall is not decisive |

---

## 1. D Lemma 0 and Lemma 1

**Setting.** nested170's render brackets a child iff prec(child) < prec(parent), or it is a right child with equal precedence. z = [EOT] + render(τ) + [" ="].

**Lemma 0 [PROVED].** The node N(i) at operator i has two leaf children iff:
- z[i±1] are numbers;
- L = z[i−2] ∈ {EOT, "("} or prec L < prec z[i];
- R = z[i+2] ∈ {" =", ")"} or prec R ≤ prec z[i].

N(i) is bracketed iff L = "(" and R = ")".

*Proof.*
- **Left, ⇒ (contrapositive).** Suppose the left child C is internal with z[i−1] a number. Then C is unbracketed, and z[i−2] is the operator of the bottom node D of C's unbracketed right spine. Unbracketed right children need strictly higher precedence, and C (an unbracketed left child) needs ≥. So prec z[i−2] ≥ prec z[i].
- **Left, ⇐.** Climb N's unbracketed left spine (prec non-increasing upward) to the first node A that is:
  - bracketed, so z[i−2] = "(";
  - the root, so EOT;
  - or an unbracketed right child of B, so z[i−2] = op_B with prec A > prec B.
  
  Hence prec L < prec z[i].
- **Right side.** The mirror argument, with the strict and non-strict inequalities swapped, gives ≤.
- **Brackets.** A "(" from an ancestor A whose left spine reaches N, and a ")" from an ancestor B whose right spine reaches N, force a path that is all-left and all-right at once, so A = B = N. ∎
- **Leftmost.** `_leftmost_ready` is in-order, and ready nodes are non-nested, so the handle is the smallest ready i: a prefix-OR.

**In-place radius [PROVED + VERIFIED].** VB writes in place: the span goes VOID, and the last span slot takes the value. Slot j's next class is a function of z[j−4..j+4] and b_j = OR_{i′≤j−3} ready(i′).
- j can lie only in the span of an operator i ∈ [j−2, j+2], and ready(i) reads z[i±2].
- b_j = 1 puts the handle at ≤ j−3, so its span ends at or before j−1.
- It is tight: the "(" slot at h−2 needs z[h+2].
- Measured on 1.7M slots: 0 conflicts at R4; 35,853 at R3.

**Lemma 1 [PROVED].**
- The re-render gate: 0 failures in 5,000 items.
- Alg. 1's support is every tree whose ÷ nodes have a nonzero right *value*. Reduction preserves values, and top-down expansion reaches any such tree, since PAIRS lists all preimages.
- cot_steps(τ_t) = cot_steps(x)[t:] by determinism.
- **The law differs** [MEASURED]. Mean height / brackets, fresh vs reduced: 3.33 / 1.50 vs 3.70 / 1.87 (r 4 from n 12), and 4.27 / 2.49 vs 4.94 / 2.94 (r 6 from n 16).
- So RESTART is in-support accuracy under the τ-law. That is the right pairing for RS, since INTACT-CONT sees the same law, but it is not "in-distribution".

## 2. D Theorems 1–2 and the assumptions

**Counterexample [PROVED].**
- A sink at logit ℓ_s with the other logits at 0 has share e^{ℓ_s}/(e^{ℓ_s} + m − 1), which is monotone in m.
- At n ≤ 8, m ≤ 2·8 + 3 + 2·7 = 33; at n 16, m ≥ 35.
- F′ satisfies A2 and can satisfy CANON. Only A5 excludes it.
- So A5's global features must be existential aggregates (prefix-OR). An average-hard head counts.

### 2.1 A2 and the restamp

- **There is no relative position term in GPT-2.** Logits are q(LN h_i)·k(LN h_j)/8, with ω only on the latest version.
- **Position lives in h** [MEASURED]. h_6 − wte − wpe decodes the slot index with held-out R² 0.9986. A probe fit on j ≤ 33 reads a mean of 23.8 where the truth is 40.2 at j > 33. So **A2 is false for the pretrained prelude even on fresh problems**.
- **What M2 guarantees.** Write V̂[j] = c ⊙ wpe[p] + rest. Then N(V̂)[j] + a ⊙ (wpe[j] − wpe[p]) leaves s ⊙ c − a on wpe[p], with s = gρ/rms(V̂[j]).
  - It is exact only if a = s ⊙ c per slot and per pass.
  - Even then it moves one pre-LN additive term. LN mixes everything, and derived features keep p (and, through V̂_1's ancestry, o(j)).
- **What Theorem 1 needs** is only that ψ_δ(V)[j] = V[j] + a ⊙ (wpe[j] − wpe[j+δ]) be a **history-free** map of the current vector and compaction. M2 with a fixed learned a is.

### 2.2 Theorem 1, restated

**Why D's chain breaks.**
- The bank stores V̂_t, but only the input is restamped.
- If V̂_t ∝ Φ(z_t) (A3), the bank entry is fresh, but N(V̂_t) + shift ≠ N(Φ(z_t)) wherever δ ≠ 0. Conversely, if the input is fresh, the bank entry is not.
- **G-C3 fails by construction.**
- **G1 fails too:** U_1 = g ⊙ V_0·ρ_cls/rms(V_0[j]) ≠ V_0, because ρ_cls is a class mean.

**Fixed model.**
- Store **Ṽ_t = ψ_{δ_t}(V̂_t)** for both the bank K/V and the carry.
- U_{t+1} = N(Ṽ_t), with N(V)[j] = g ⊙ V[j]·r(j)/rms(V[j]) and **r(j) = rms(Φ(frame(content))[j])**.
  - At pass 1 this is the identity (G1 ✓).
  - Later it is content-only: one no-grad prelude on the decode, which CANON computes anyway.
- B1 (no CANON) may instead use the identity at pass 1 and ρ_cls afterwards, making no Theorem 1 claim.

**Assumptions.**
- **A1:** exact compaction.
- **A3′:** on exact chains, Ṽ_t[j] = Λ_t(j)·Φ(z_t)[j] with Λ > 0. bank_kv's LN and N remove Λ.
- **A4:** older versions are hard-masked (W = 1).

**Theorem 1 [PROVED UNDER A1, A3′, A4].** Exactness on every problem of size ≤ m (m ≥ 2) ⇒ every item with n − m exact steps is exact thereafter, for every K′ ∈ [n, n + H].

*Proof.*
- At t₀ = n − m: U = N(ΛΦ(τ)) = g ⊙ Φ(τ) = N(V_0^{fresh}).
- The latest entry has V_0^{fresh}'s K/V and bias (β + ω, no age), and older entries are masked.
- So pass t₀+1 equals fresh pass 1, and the keep decisions and ψ (history-free) agree. Induct.
- The coda after fresh pass 1 has no bank, so answers could differ only at K′ = t₀+1, which is relevant only when m = 1. ∎

**Remarks.**
- "Every problem of size ≤ m" is an idealisation (about 2×10¹⁷ problems at n = 8). The empirical quantity is E_{τ∼law(τ_{t₀})} err(τ), which is what RS measures.
- **Quantitative form.** δ_j ≤ Λ^j ε_C + ε_B(Λ^j − 1)/(Λ − 1), valid only while δ_j stays below the R-margin (compaction is discrete). With finite μ, ε_B is governed by ages 1–3 (§3), which HIST-TRUNC measures.

### 2.3 Theorem 2, restated

**A5′ (factored).** Under A1, A3′ and A4 the pass is a symbol map. It factors as:
- *stage 1:* operator features are a function of the radius-2 window;
- *stage 2:* a slot's class is a function of its own class, the stage-1 flags at offsets −2..2 (operator / ready / own brackets), and b_j, plus the 484-entry OP table.

**COV′.** Every test key of both stages occurs in training.

**Theorem 2 [PROVED UNDER A1, A3′, A4, A5′, COV′].**
- *Proof:* the true rule factors this way (Lemma 0, in-place radius). Exactness on training states fixes each realised key, and COV′ covers the test keys. Induct. ∎
- **Measured:** 248 stage-1 and 169 stage-2 keys; state coverage **1.000** at n 9–16.
- A **monolithic** radius-4 slot function (A5 as written) covers only 0.934 / 0.923 / 0.887 / 0.822 / 0.792 / 0.802 / 0.736 / 0.712 of states at n 9–16. D's 0.997 is a window share.
- A5′ subsumes A2.

**Corollary [PROVED UNDER first-order composition].** Each softmax OR is off by ≤ 2Bme^{−Δ} (Lemma 2). Then q heads × Λ_R < γ ⇔ m < γe^Δ/(2BqΛ_R).

**Lemma 3 [PROVED UNDER pair coverage].** Logits on window pairs that never co-occur at n ≤ 8 are unconstrained, which is a counterexample under single-window COV. The lemma also needs a local Φ, which the causal prelude is not.

## 3. D Lemma 2 / 2′ / 4: constants

- **Lemma 2 [PROVED].**
  - α_F ≤ |F|e^{λ−Δ}/(e^λ + |F|e^{λ−Δ}) = ε/(1+ε).
  - o − o_N = α_F(ō_F − o_N), so the deviation is ≤ 2Bε/(1+ε).
  - With bounded logits, α_F → 1 as |F| → ∞ (the finite horizon).
- **Lemma 2′ [PROVED UNDER max-logit margin].** Assume some relevant key has logit ≥ λ. Then ε ≤ m·e^{−κΔ} = m^{1−Δ/ln m₀}, non-increasing iff Δ ≥ ln m₀. A sum condition Z_N ≥ e^λ is not preserved by κ-scaling.
- **Lemma 4 [PROVED].**
  - Z_old ≤ T₀e^{λc−μA}/(1−e^{−μ}), so the movement is ≤ 2B·T₀e^{λc−λ−μA}/(1−e^{−μ}).
  - At μ = 2, A = 3 this is **0.0057·B·T₀e^{λc−λ}** (D dropped the 2B).
  - With T₀ = 33 and μ = 2, C(A) = T₀e^{−μA}/(1−e^{−μ}), and the content gap needed for a movement ≤ 0.02B:

| ages ≥ A | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C(A) | 5.17 | 0.70 | 0.095 | 0.013 | 0.0017 |
| gap needed (nats) | 6.3 | 4.3 | 2.3 | 0.3 | 0 |

  - **At zero gap, only ages ≥ 4 are negligible.** A4 does not follow from μ = 2. Add 0.5 nats at T₀ = 55.

## 4. S C1, X1 and the interaction check

**C1 [PROVED, restated].**
- On matched skeletons the handle is a function of skel(dec V_{t−1}), so **Ẽ_t is σ(dec V_{t−1})-measurable**.
- Every zero-loss model outputs F(dec V_{t−1}) on the reached set. V_0 may be used, but it must lose wherever it conflicts.
- Free-running E_t[b] is not decode-measurable on the error set. Its minimiser needs information beyond the decode, which **V_{t−1}'s null space** supplies as well as V_0 or older versions.
- The forcing set is the model's current error set plus the OPX edits. Its mass → 0 as the model converges, and COMMUTE ≡ R2 at an exact model and on fallback inputs.

**X1 [PROVED on the edit support].**
- Edits are uniform over (a′, b′) ≠ (a, b), so every (literal, version value) pair occurs.
- L_X = 0 then makes the b-output OP[op, a′, b′] as a function of the operand slots' row-space coordinates in versions s_k..t−1. That function is non-constant except in the ×0 and 0÷ cells.
- V_0-invariance is structurally moot under V0-OFF. Natural transfer is [OPEN].

**Interaction.** Well-defined if all four conditions hold:
1. **Training uses the teacher keep mask.** On matched skeletons, Ẽ_t's VOIDs are E_{c_t}'s, and a kept slot decoded as VOID falls back.
2. **OPX tracks operand slots by original index o**, since compaction shifts them by 2 or 4. Stalled copies are edited together.
3. **Index by reduction.** OPX draws a reduction, not a pass, and s_k is the creating *reduction's* pass. D's trained stall pass and its successor share the target F(dec V̂_{t−1}), which is consistent.
4. **One semantics.**
   - L_X's other slots must use Ẽ′ = F(dec V′_{t−1}), not E_t.
   - **CANON (B2) targets the truth Φ(cot_steps[c_t])**: after a value error it pulls V̂_t to the correct value while COMMUTE follows. That reintroduces the free-running signal in 768 dimensions.
   - Use **Φ(frame(dec Ṽ_t))** (with a truth fallback on parse failure). This is also what A3′ needs.

## 5. S's race model and the transfer results

**Race [PROVED UNDER common-p].**
- The arithmetic holds under S's rule: 2(1−p)(π_c − π_s); 2(1−p)(1 − π_sπ_c); and ≈ 2ρ_X/K̄ for OPX.
- The exact two-voter CE gradient is E[1[a_C = y] − p(a_C)]. The drift difference is w_A·P(cmp right, rec wrong) − w_B·P(rec right, cmp wrong), with event-dependent weights.
  - Near uniform p (early training), w_A = w_B, so the signs are exact.
  - Once one circuit dominates, its wins are down-weighted, and the later signs are not derived.
- **π_s is [ASSUMED]** (205-5). Only the stage-1 hh shares (0% ≥ 4, 11% = 3) are measured.
- Tilt arithmetic: Φ⁻¹(.95)/Φ⁻¹(.6) = 6.5, and 6.5² = 42 ✓ [UNDER 199-A].
- Under V0-OFF, C_rec's source is attenuated (§7), so COMMUTE's sign margin only grows.

**Stationary transfer [PROVED UNDER S1–S3′].**
- S2′: the whole law of LN(V_{t−1}) given dec V_{t−1} = E (row-space margins included) is free of t and path. Then the chain rule gives P(all clean | x) = Π_t(1 − h(π(E_{t−1}))) exactly.
- The bound holds only on covered states. Raw-slot windows are unbounded (VOID gaps); compacted factored keys cover every state to n 16.

**Product bound [PROVED].** Π(1 − h_t) = P(all clean) ≤ P(final correct). It is tight under COMMUTE up to ×0 coincidences.

**Drift.**
- κ < 1 needs ‖u_0‖ ≤ ρ as well.
- "κ > 1 ⇒ competence lost at t*" needs ‖Au‖ ≥ κ‖u‖ on u_0's component; ‖A‖ ≤ κ is only an upper bound.

**Coverage [PROVED].** t + r = n + 1 at test. O = 4 covers r ≤ 8 for n ≤ 12: 32/42 = 0.762 ✓.

## 6. H's propositions

- **Prop 1 [PROVED].** t_halt is in no loss term, R1-C, COMMUTE and STALL-AUG included.
- **Prop 2 [PROVED].**
  - Re-run: 5,021 / 4,926 / 5,000 / 5,053, then 0, 0, 0, and 1.000.
  - E_{n−1} always has one operator and no bracket (a bracket needs a parent), so these are #items by root.
  - If DONE reads the compacted V̂_t, g(VOID) is unidentified but VOID is absent. If it reads the pre-compaction V_t, it is as in VB.
- **Thm 3 [PROVED, strengthened].** DONE reads only the skeleton, so skeleton exactness through n gives t_halt = n. COMMUTE's absorbing value errors never mistime the halt.
- **Prop 4 [PROVED].**
  - (c) 1 − p_top ≤ (C−1)e^{−m}. A done-state flip needs some m_j ≤ ln(2(C−1)T): 7.16 at T = 34, 7.64 at T = 55. A live-state flip needs m ≤ 3.6 at an operator slot.
  - (d) LN is affine-invariant.
  - (e) the construction holds when the code noise is < ½.
- **Prop 5 [DERIVED].** The simulation's collapse at one alive operator counts as on time.
  - P(t_halt = n) = (1−σ−δ)^{n−1}(1−σ)(1−ν) gives 0.633 / 0.505, vs simulated 0.635 / 0.507; H's form gives 0.630 / 0.502.
  - H's premature formula is the collapse probability (0.036). The table counts halt < n (0.034).
  - acc_halt ≈ (1 − μ′ − (10/11)δ′)ⁿ = 0.884 ✓; σ ≲ 0.018 ✓ (0.905).

**Prop 8, precisely.** Let a ≥ 1 be the operators alive at a pass's input and t the pass index; at test, a + t = n + 1. Training is 2×10⁵ items, n ~ U{1..8}.

| form | reduce cells seen | last reduce pass | P(reduce at pass ≥ T), T 9 / 10 / 11 / 12 / 13 | test steps on seen cells, n 9 / 10 / 11 / 12 / 13 |
|---|---|---|---|---|
| none | a + t ≤ 9 | 8 | 0 | 0 |
| **H** (0.3 per item, S ∈ {1, 2}, stall skips the core) | a + t ≤ 11, a ≤ 8 | **10** | .056 / .019 / **0** / 0 / 0 | .89 / .80 / 0 / 0 / 0 |
| **D** (0.1 per pass t ≥ 2, unbounded, stall trained) | all a ≤ 7 (geometric weight); a = 8 only at t = 1 | 15 | .088 / .029 / .0076 / .0017 / .0003 | .78 / .70 / .64 / .50 / .39 |

- **(i), (ii) [PROVED], but incomplete.**
  - Under H's form, **"hold iff t ≥ 11" is exactly zero-loss**: it is CLOCK-HOLD shifted, freezing n ≥ 11 items with n − 10 operators alive.
  - Under both forms, **K*** ("reduce iff t − #(latest = age-1 entry) ≤ n") is zero-loss. Copies are detectable, since the age-1 entry is readable (§3).
- **(iii) [PROVED for f(content, t)].** It says nothing about history-reading rules such as K*.
  - Under D's form no "t ≥ T ⇒ hold" is exactly zero-loss, but its weight falls about 4× per pass.
  - **The cap is immaterial:** τ + 4 ≤ 12 + S whenever n ≤ 8.
  - VB-2's "p_s 0.1–0.3, S ∈ {1, 2}, no cap" is therefore H's band (n 9–10). **Name one form.**
- **(iv)** holds with "band" read as "weight ≳ 10⁻³ per item".

## 7. V0-OFF: what remains after pass 1

**Closure [PROVED, with D's N].** With V_0 masked in the core for t ≥ 2, the coda never reading V_0, and N history-free, every V_0 → pass-≥2 path runs through V̂_1. VB's pin-5 N (rms(V_0[o(j)])) is a direct V_0 path into every pass.

**What remains.**
1. **Pass-1 precompute.**
   - V̂_1 holds n − 1 operators of content plus anything pass 1 writes into its null space, and CARRY propagates it.
   - CANON to Φ(dec) limits it to the prelude's levels. In B1, PROBE(1) / (2) measure it.
2. **Older-version recompute [PROVED].**
   - Let k be the number of reductions inside a handle's operand subtrees. Each takes a pass, and every version holding the whole subtree predates all of them, so the youngest source has **age ≥ k ≥ h − 1** (stalls only add age).
   - Mass ≥ α on it needs Δ_c ≥ μ·age + logit(α) − ln(#matching keys): **≥ 2(h−1) nats** at α = ½ with one key.
   - Without that gap, ages ≥ 4 move a head by ≤ 0.026B (Lemma 4).
   - So the recompute cost now *grows with handle height* instead of being constant, as it was for V_0.
3. **V̂_1 itself** costs μ(t − 2) nats at pass t. It is the fixed-origin entry (§8).

**Constraint note.** At μ ≥ 2 an age-a read costs 2a nats. With GPT-2-scale logits, the bank behaves like a window of 3–5 versions unless content logits grow. This bears on open question 1, and it is the user's call.

## 8. The clock inventory and its tests

| clock | status in VB-2 | detected by (stalls trained) |
|---|---|---|
| VOID count; gather γ_q on V_0 | removed (compaction; CARRY base) | — |
| "t ≥ 9 ⇒ hold" | closed, but **shifts to t ≥ 11** under H's form | DEPTH-TIMING n 11–12 (n − 10 alive) |
| bank normaliser | **saturating** under the floor: the t-dependence beyond age t − 1 is ≤ z̄e^{−μ(t−1)}/(1−e^{−μ}) [PROVED], so it resolves only t ≲ 3–4 | PF at small k; removed by HIST-TRUNC |
| oldest kept version (V̂_1, age t − 2; ages present {1..t−2}) | an exponentially attenuated absolute clock: the signal ∝ e^{Δ_c−μ(t−2)} | PF k-axis; removed by HIST-TRUNC |
| null-space content / progress counters | open in B1, pinned in B2 (ε_C). **Copy-form stalls carry the counter, so STALL-AUG trains t-invariance, not c-invariance** (c + r = n ≤ 8 always) | **RS**, **PF at k + r > 8**. They survive HIST-TRUNC. ADVANCE if W3; CLOCK-PROBE |
| stall detection → K* | zero-loss | **REWIND** (K* freezes with 1 alive) |
| derived absolute position | untouched by M2 (R² 0.9986); slots up to 54 at n 16 | PF s-axis, POS-PROBE, R-POS |

**Is any reading still a comparator-free clock test?** Yes: **PF's k-axis.**
- It runs inside one model on natural inputs, at a matched remainder and span (exactly Q at step k, per D's 20,000 / 20,000), with a built-in in-support control (k + r ≤ 8).
- Neither STALL-AUG form trains it: stalls change t − c, never c + r.
- Add one arm, **PF × HIST-TRUNC**. A k-effect that survives is a null-space counter; one that vanishes is a bank-age or normaliser clock.
- **RS** is the second decisive reading. It detects history keying of any kind.

**REWIND** is held out under both forms and specific to bank-index counting (K* included). But its history is off-support: an age-1 version *ahead* of the latest, readable at μ = 2. So only **FROZEN-1** counts as the clock signature, and REWIND is descriptive.

**Triple stall** is not decisive:
- it is trained under D's form (P(≥ 3 stalls) ≈ 0.05 at n 8);
- under H's form a generic duplicate counter (K*) passes it.

STALL-SHIFT / NO-CLOCK are trained-for and no longer decisive.

**Required before implementation:** fixes 6, 7, 9, 17, 28–29 of the table. G-C2 and G-C3 would otherwise fail by construction, and STALL-AUG's form must be named.
