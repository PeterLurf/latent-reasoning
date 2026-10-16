# H — Learned stopping (C5) on VERSION-BANK: CONTENT-HALT and STALL-AUG

*Opus research subagent for Theory, 2026-10-16 ~03:10 EDT (clock read 02:51). Bottleneck (3), learned stopping. The repo was read-only. CPU checks are in `scratchpad/research2/hsim/`: `ident_check.py` (20k generator items) and `halt_sim.py` (with `halt_sim.json`). Nothing here has read any §211 output; the cells land at about 05:00.*

**Tags:** [PROVED], [PROVED UNDER X], [DERIVED], [DERIVED; simulated], [MEASURED] (repo data), [ASSUMED], [CONJECTURE].

**Summary.**
- The primary halt is a **zero-parameter content predicate** read through the existing linear recode head R: stop at the first t at which the decoded newest version Ê_t has no live operator and exactly one value slot.
  - It needs no architecture change, adds no loss term and sends no gradient anywhere.
  - Its whole depth-transfer question reduces to whether the step rule is exact (Thm 3).
- A learned head is unnecessary: the learnable version is *identified* to the predicate (Prop 2).
- Decisive in-model readings:
  - **STALL-SHIFT**, which also rules out the trivial "count the input's operators" halt;
  - **DEPTH-TIMING** at n 9–10, with two signatures that separate a bank-size clock (**CLOCK-HOLD**) from a size-edge collapse (**EDGE-COLLAPSE**).
- **STALL-AUG** is a training-side lever. It is a pass-schedule augmentation, proved to make the count-keyed rule and VB's own zero-loss "t ≥ 9 ⇒ HOLD" shortcut suboptimal.

---

## 1. Diagnosis: what each past halt could read

**Principle** [DERIVED from the rows below]: *a halt is only as clock-free as the least clock-free signal it reads, and only as depth-free as the computation whose completion it detects.*

| halt (section) | signal read | clock or size cue available to it | outcome [MEASURED] | why |
|---|---|---|---|---|
| answer2 (§173) | 3 equal decoder answers | none needed | PREMATURE (§184 s1 n 10: 0.112 vs 0.641; mean halt 2.5–4.5) | Before completion the decoder emits a *stable default*, so output stability carries no information before completion (§201 Part II) |
| MOVED2 (173-3) | as above, ≠ a(0) | none needed | PREMATURE on held-out s2 (n 10: 0.708 vs 0.902) | Wrong answers move off a(0), then settle |
| memory-change ε (§173) | ‖ΔM‖/‖M‖ | additive growth | never fires (capped) | The state never stops moving; under additive growth relative change ~1/t, which the 1% test cannot separate from real change |
| AUX-DONE, §184 / §175 | aux decoder emits "number + EOT" | **pass token** in the aux prompt | exact t = n for n ≤ 8; pinned at t = 8 past the edge (976 / 1,000 at n 9) | T-SHIFT: "Step 8" was always final in training, so the token alone predicts done |
| AUX-DONE, §187 NOPASS | same | **additive-read magnitude** (norm probe 0.60–0.65) | pinned at 8 (991 / 1,000) | A state counter replaced the token |
| AUX-DONE, RMS | same | token only | fires at 8 or never (419 / 581 at n 9) | Token edge |
| AUX-DONE, §190 RMS × NOPASS | same | **neither** | DONE-TRACKS-N: n 9 996 / 1,000 at 9; n 10 halted mean 9.66; replicated in §194 s2 | A content predicate works once its inputs carry no counter |
| SHIRYAEV-HEAD (§201 T2) | logistic on confidence, change flags, AUX-DONE bit, β̄ / ρ, read increments | whatever those features inherit | NO-OVERTHINK on 7 / 7 in range; RMS models track n (event mean 9.7–9.8 at n 10); NOPASS / reweight pinned at ~8.1–8.4 (PREMATURE on §187 s2) | The head inherits its features' counters |

So the evidence already favours (a). The one halt that transferred (§190 AUX-DONE) is a **content predicate on a decoded scratchpad whose decoder sees no clock**. VB's R is exactly that kind of decoder, in its strictest form: linear, per slot, with its own LayerNorm and no prompt.

**Clocks still reachable by VB's *core*** (from the code: boxes/vbank.py `bias_row`, `gather`):
1. **Bank size.** Every core layer's and the gather's softmax normaliser grows with t.
2. **ALiBi age.** It reaches the core through two fixed-origin entries:
   - In the core layers V_0 has no age term, but **V_1 is a fixed-origin entry too**: its logit is β + γ(t − 2). A relative-age bias over an append-only bank is an absolute clock (PROOFS §5(b), generalised) [PROVED: the set of ages present at pass t is {0..t−2}].
   - The **PA gather applies γ_q·(t − 1 − s) to V_0 as well** (`gather`: s_idx includes 0). So N2 and N3 carry an absolute V_0 age clock in the pass input; N1 (CARRY) does not. This is the registered design, not a bug, but it bears on halting.
3. **A training-set shortcut.** Under n ≤ 8 and K_i ≤ 12, every pass at bank index t ≥ 9 is a HOLD pass, so **"t ≥ 9 ⇒ hold" is zero-loss on-distribution** (Prop 8). This is the VB analogue of §182 HOLD's "Step ≥ 9 → done".
4. **Content counters** that the core writes into R's null space.

The magnitude clock is removed from R by its LN (Prop 4d).

---

## 2. Mechanism

### 2.1 The predicate (CONTENT-HALT)
Let p_{t,j} = softmax(R(V_t)[j]) over the 20 classes, Ê_t[j] = argmax_c p_{t,j}(c), 𝒪 = the 4 operator classes, ℬ = the brackets and 𝒱 = the 11 values. Then:
- alive(Ê) = #{j : Ê[j] ∈ 𝒪};
- nval(Ê) = #{j real : Ê[j] ∈ 𝒱};
- **DONE(Ê) = [alive(Ê) = 0] ∧ [no j with Ê[j] ∈ ℬ] ∧ [nval(Ê) = 1]**;
- **t_halt(x) = min{t ≥ 1 : DONE(Ê_t(x))}**, capped at T_max = 24 (never-halting = failure).

The answer is the coda's answer after pass t_halt, the registered answer path. The **R-answer**, the value class at Ê's single value slot, is reported beside it as a coda-free self-consistency check.

**Guards** (label-free, no clock):
- **G-PROG (collapse guard).** If the decoded alive count drops by ≥ 2 in one pass (alive(Ê_{t−1}) − alive(Ê_t) ≥ 2), the trajectory has left the trained one-reduction-per-pass policy. A DONE on that trajectory is flagged UNRELIABLE: the item continues to T_max and is reported as an abstention (counted as failure). This turns an edge collapse into an honest abstention, not a confident wrong halt. It does not use n.
- **STUCK-STOP r = 3** (compute only). Ê_t = Ê_{t−1} = Ê_{t−2} with alive > 0 means the step rule is stuck. Stop and report never-halt; this saves T_max − t passes on hopeless items.

### 2.2 VB integration and where the gradients go
- It reads only the newest bank entry V_t through R, which already exists. There are no new parameters, and the bank, gather and coda are untouched. It is compatible with the pass-global memory constraint: memory is unchanged, and halting is a readout of its newest write.
- **Parameters reached:** none [PROVED: L = L_ans + λL_rec does not contain t_halt, so ∂L/∂θ is independent of the halting rule].
- For the descriptive comparators in §5, the learned heads are fit on **detached** features, so each reaches only its own weights.
- STALL-AUG (§2.4) reaches every §211 parameter through the existing L_ans and L_rec; only the pass schedule changes.

### 2.3 The options, evaluated
- **(a) The content predicate: adopted as primary.**
- **(b) A learned head on version content.**
  - An R-factored head (pooled class posteriors) is identified to (a) (Prop 2), so learning adds nothing.
  - A raw-V head can read clocks (Prop 4e). It is kept only as a *descriptive leak meter* (HEAD-STALL, CLOCK-GAIN).
- **(c) Convergence (V_t ≈ V_{t−1} or Ê_t = Ê_{t−1}).**
  - It is dominated (Prop 7): one pass late on time, premature at every stall, and it converts a stuck count-keyed trajectory into a confident wrong halt.
  - It survives only inside STUCK-STOP.
- **(d) Decoupling from training depth.**
  - VB already draws K_i = n + U{0..4} and has state-only done labels (HOLD = E_n).
  - The remaining coupling is the clock × remaining-work cross term and "t ≥ 9 ⇒ hold". **STALL-AUG** removes both on a band (Prop 8).
- **(e) Overthinking and failed steps.** Covered in Props 5, 6 and 9 and the guards.

**Not adopted:** a supervised done loss BCE(P_done(R(V_t)), d_t) with gradient. d_t = D(E_t) deterministically, so any model with exact L_rec already has exact L_done: the term adds no constraint at the optimum, and away from it, it only reweights the operator-slot CE terms [PROVED]. Its gradient would also add a "look done" pressure on V_t (cf. Popescu 2607.20519, joint gate damage).

### 2.4 STALL-AUG (training lever, conditional; §6 step 3)
**The schedule.** With probability p_s = 0.3 per item, draw S ∈ {1, 2} stall positions s_1 < s_2, uniform in {1..n}. At a stall pass, skip the core and append a copy: stored V_s := V_{s−1}, with gradient through both uses.

**Targets** follow content, not bank index:
- the R2 target at bank index t is E_{t − S_<t}, where S_<t is the number of stalls at or before t;
- the copy entry itself has no R2 term;
- τ_i = n + S and K_i = min(τ_i + U{0..4}, 12 + S);
- the curriculum, λ and everything else are as §211.

**Cost:** E[extra passes] = 0.45 per item, about +6% compute at mean K ≈ 7 [DERIVED].

---

## 3. Mathematics

**Prop 1 (halting as a readout: no incentive to halt early) [PROVED].**
- t_halt enters no loss term, so no gradient pushes toward early or late halting.
- Collapse to one loop would need R(V_1) to decode E_n. But L_rec targets E_1 at t = 1, so collapse strictly raises L_rec on every item with n ≥ 2.
- None of the published collapse routes is present. They need a ponder cost (ACT / PonderNet, RecurTrace 2609.03379), an exit distribution (Ouro 2510.25741, entropy-dependent) or joint gate training (Popescu 2607.20519). ∎

**Prop 2 (identification) [PROVED; preconditions verified].**
- Take the slot-symmetric OR class h(Ê) = 1 − max_j g(Ê[j]), with g : classes → {0, 1} and no position or pass input.
- Then zero error on the done label d_t over training trajectories with exact R forces:
  - g(c) = 0 for c ∈ {VOID, EOT, " =", every value};
  - g(o) = 1 for each of the 4 operators.
  Only g(bracket) is free, and it is irrelevant on well-formed states.
- *Proof.*
  - Every done state E_n contains only VOID / EOT / " =" plus one value, and all 11 values occur. So g = 0 on those classes.
  - For each operator o, some non-done state contains o as its only live operator and no bracket: E_{n−1} of any item with root o. So g(o) = 1.
  - Verified [MEASURED, ident_check.py, 20k items, n 1..8]: lone-operator, no-bracket states 4,926–5,053 per operator; done states that are not a single value 0; non-done states with no operator 0; brackets without an operator 0; exactly one operator removed per step 1.000. ∎
- **Scope.** A general function of the class *multiset* is identified only on the count vectors seen in training. A VOID-count term, for example, would break at deep n, where T > 34. That is why the predicate's form is fixed rather than learned.

**Thm 3 (depth transfer reduces to recode exactness) [PROVED UNDER exact recode].**
- For any item x of any depth n (including n > 8 and T > 34): if Ê_t(x) = E_t(x) for all t ≤ n, then t_halt(x) = n.
  - Proof: E_t has an operator for t < n; E_n is a single value (re-render gate).
  - G-PROG never triggers, since exactly one operator is removed per step.
- If, in addition, the coda's answer is correct at t = n, the halted answer is correct.
- **Corollary.** The halt has no generalisation problem of its own. Every DEPTH-TIMING failure is a failure of the step rule or of R's decoding, and §5's decomposition names which one.

**Prop 4 (clock leakage bounds).**
- **(a) Stalled pass [PROVED].** Any head h(V_t) gives h(V_t) = h(V_{t−1}) at a STALL pass, because the input is an exact copy. A head that reads only the newest version cannot see bank size at that pass. It sees a clock only if the core *writes* one into V_{t+1}, V_{t+2}, ….
- **(b) Hard R-factored head [PROVED].**
  - h = f(Ê_t) gives I(h; t | Ê_t) = 0.
  - On HOLD versions Ê_t = E_n for every t > n. So the decoded content carries no information about t beyond min(t, n).
  - The head's only access to a clock is through the core's decoded writes.
- **(c) Soft R-factored head [PROVED].**
  - Let m_j be slot j's top-1 margin. Then 1 − p_j(Ê_j) ≤ (C − 1)e^{−m_j}, with C = 20.
  - For a content-matched pair (V, V′) (same Ê, different t), and g L-Lipschitz in Σ_j ‖p_j − p′_j‖₁: |h(V) − h(V′)| ≤ 2L(C − 1) Σ_j (e^{−m_j} + e^{−m′_j}), since ‖p_j − δ_Ê‖₁ = 2(1 − p_j(Ê_j)).
  - For the product form P_done = Π_j (1 − q_j), with q_j the alive mass at slot j: on a done state, 1 − P_done ≤ (C − 1) Σ_j e^{−m_j}.
  - So a 0.5 threshold can flip only if Σ_j e^{−m_j} ≥ 1/(2(C − 1)): **some slot's margin must fall below about ln(2(C − 1)T) ≈ 7.2 nats** (T = 34).
- **(d) Magnitude invariance [PROVED].** LN(αv + β𝟙) = LN(v) for α > 0 (up to ε). So R, and every R-factored halt, is blind to per-slot scale and mean shifts: §187's magnitude counter cannot reach it.
- **(e) A raw-V linear head can read a clock [PROVED by construction].**
  - Suppose pooled features carry affine codes u₁ = a·t + ξ₁ and u₂ = b·n + ξ₂ (as probes found in earlier families: n at 0.85, 187-7). Then w·u = u₁/a − u₂/b + ½ thresholds t ≥ n. It reaches the BCE infimum on training trajectories.
  - If the core writes a bank-size code, the head fires at bank index n after a stall, with one operator still alive: premature.
  - No bound is possible without restricting the features. This is measured, not assumed (CLOCK-GAIN, §5).

**Prop 5 (error model: halting with an imperfect step rule) [DERIVED; simulated, halt_sim.py, 2×10⁵ items per cell].**

*Per-pass outcomes* while operators are alive: progress, stall σ, wrong value μ (the timing still progresses), or collapse δ (all remaining operators written as one value). After done, each pass leaves a decode residue (a false live operator) with probability ν. Let δ′ = δ/(1 − σ) and μ′ = μ/(1 − σ).

*Content-keyed core (C):*
- P(t_halt = n) = (1 − σ − δ)ⁿ(1 − ν);
- P(premature) = 1 − (1 − δ′)^{n−1};
- E[t_halt | no collapse] = n/(1 − σ) + ν/(1 − ν);
- **acc_halt ≈ (1 − μ′)ⁿ(1 − δ′)ⁿ, independent of σ**, against acc(K′ = n) ≈ that × (1 − σ)ⁿ. So halting absorbs stalls.
- acc_halt − acc(K′ = n + 2) = P(≥ 3 stalls, otherwise correct) ≥ 0.

*Count-keyed core (K: reduce iff t ≤ n):*
- P(never) = 1 − (1 − σ)ⁿ. With one forced stall, P(never) = 1.
- acc_halt = acc(K′ = n + 2) = acc(K′ = n).

*Edge-pinned core (E8: collapse at t = 8 if more than one operator is alive):* for n ≥ 9, t_halt = 8 always and the accuracy is about 1/11.

| core | n | σ / μ / δ / ν | exact | premature | late | never | acc_halt | acc K′n | acc K′n+2 |
|---|---|---|---|---|---|---|---|---|---|
| C | 8 | .02 / .01 / 0 / .01 | .842 | 0 | .158 | 0 | .921 | .784 | .920 |
| C | 8 | .05 / .01 / .005 / .01 | .635 | .034 | .331 | 0 | .884 | .588 | .874 |
| C | 12 | .05 / .01 / .005 / .01 | .507 | .053 | .440 | 0 | .831 | .452 | .806 |
| K | 8 | .02 / .01 / 0 / .01 | .842 | 0 | .009 | .150 | .784 | .784 | .784 |
| K | 12 | .05 / .01 / .005 / .01 | .507 | .053 | .005 | .434 | .452 | .452 | .452 |
| E8 | 10 | any row | 0 | 1.000 | 0 | 0 | .085 | .085 | .085 |
| C, forced stall | 8 | .02 / .01 / 0 / .01 | .843 at n + 1 | 0 | .157 | 0 | .921 | | |
| K, forced stall | 4–8 | .02 / .01 / 0 / .01 | 0 | 0 | 0 | **1.000** | 0 | | |

**Reading.**
- The accuracy gap against the K′ oracle is **not diagnostic** of keying: it is 0 under K and E8 and ≥ 0 under C.
- The halt-step *distribution* is diagnostic:
  - C fails **late** (late / premature ≈ σ/δ);
  - K fails **never**, with a FROZEN-1 signature after a stall;
  - E8 fails **premature at exactly t = 8**.
- HALT-EXACT ≥ 0.90 pooled over n 3..8 needs σ ≲ 0.018 per pass [DERIVED: the mean over n 3..8 of 0.99ⁿ·0.995 is 0.941; at σ = 0.02 it is about 0.89].

**Prop 6 (decoding errors make the predicate fail late) [PROVED UNDER independent per-slot decode errors e].**
- A false DONE on a state with a live operators needs every operator slot misread *and* a of the a + 1 value slots misread as non-values: P ≤ C(a+1, a)·e^{2a}, so about 2e² at a = 1.
- A false not-done on a done state needs one spurious operator or bracket among T − 1 slots: about (T − 1)e.
- The ratio late / premature ≈ (T − 1)/(2e), more than 10³ at e = 0.01.
- Without the single-value conjunct, the premature rate at a = 1 would be about e, not 2e². That is why the conjunct is in DONE.

**Prop 7 (convergence halting is dominated) [PROVED under exact recode].**
- Decoded convergence (Ê_t = Ê_{t−1}) first holds at t = n + 1: one pass late.
- Under a stall at s < n it fires at s: **premature by construction**.
- Under a stuck K-type trajectory it fires at the first stuck pass, with an operator alive, so the coda reads an unfinished version: a confident wrong halt. CONTENT-HALT instead never fires (an honest censor).
- Raw-V convergence ‖V_t − V_{t−1}‖/‖V_t‖ < ε is not identified: R2 pins only argmax margins, and the null space is free [PROVED]. [CONJECTURE] It behaves like §173's memory-change rules (never fires).

**Prop 8 (the "t ≥ 9 ⇒ hold" shortcut, and the STALL-AUG band) [PROVED].**
- **(i)** Under §211 training (n ≤ 8, K_i ≤ 12), every training pass at bank index t ≥ 9 is a HOLD pass. So the rule "hold if t ≥ 9, else reduce until done" agrees with E on every training trajectory and is zero-loss.
  - On n 9–12 it **freezes with exactly n − 8 operators alive**: CLOCK-HOLD, the never-halt signature.
  - The count-keyed rule K is zero-loss as well (PROOFS §5).
- **(ii)** Under STALL-AUG with p_s > 0 and S ≥ 1, take an n = 8 item with a stall at s ≤ 8. Its last reduction falls at bank index 9, and its target there is E_8 ≠ E_7.
  - So both "t ≥ 9 ⇒ hold" and rule K pay L_rec ≥ p_s·P(s ≤ n)·ℓ_min > 0 at that slot. Neither is a minimiser.
- **(iii) Band theorem.** Any zero-loss rule f(content, t) must equal the content rule C on the support band ℬ = {(E_p(x), t) : p ≤ t ≤ p + S_max, t ≤ 12 + S_max}.
  - With S_max = 2, reductions occur at bank indices up to 10 in training. So (operators alive, bank index) pairs with alive + t ≤ n + 2 become in-distribution: the clock × remaining-work cross term of 187-6a.
  - Outside ℬ nothing is constrained. That includes any version with more live operators, or more slots, than any training item (n ≥ 9 at t = 0; T > 34). With S_max = 2, n 11–12 lie outside the band.
- **(iv) Consequence.** If STALL-AUG raises DEPTH-TIMING at n 9–10, the deep failure was clock × remaining work. If it does not, the deep failure is content length (§190's "done + remaining > 8"), which needs longer items (e.g. 190-4's SELF-DEPTH-ANS).

**Prop 9 (overthinking under the halt) [PROVED].**
- Stopping the loop at t_halt makes the accuracy at any budget B ≥ t_halt equal the accuracy at t_halt for halted items. This is the VB form of §201's FD-on-signal (no further writes).
- C6 risk is then confined to never-halting items, which the censoring rule already counts as failures. The dual risk (premature) is confined to collapses, which G-PROG flags.

---

## 4. Goodhart table

| route | channel | status in VB + CONTENT-HALT | reading that catches it |
|---|---|---|---|
| Pass token | prompt | Absent by construction | — |
| Magnitude counter | ‖V_t‖ | Invisible to R (LN, Prop 4d); the core reads through LN'd bank K/V and N(·) | — |
| Bank-size normaliser | every core softmax, gather | **Open** | STALL-SHIFT (FROZEN-1) |
| Fixed-origin age | V_1 age t − 2 (core); V_0 age in the PA gather | **Open** (N2 / N3 more than N1) | STALL-SHIFT; N1 vs N2 descriptive |
| "t ≥ 9 ⇒ hold" | training support | **Open, zero-loss** (Prop 8i) | CLOCK-HOLD at n 9–12; closed by STALL-AUG |
| Count keying "t ≤ n" | size × clock | Zero-loss on-distribution | STALL-SHIFT |
| Input-count halt (trivial: n is countable from the text) | input tokens | A degenerate "C5" that passes HALT-EXACT and DEPTH-TIMING | **STALL-SHIFT fails it** (halts at n with one operator alive) |
| Edge collapse | write the final value at the size edge | Possible | EDGE-COLLAPSE (t_halt = 8 at n ≥ 9); G-PROG flags it |
| Content counters in R's null space | V_t | Invisible to the halt (4b); the core may use them | CLOCK-PROBE, ADVANCE (descriptive) |
| Premature halting to save compute | — | No compute cost, zero gradient (Prop 1) | — |
| Collapse to one loop | ponder cost | Absent; also raises L_rec | PROBE(1) / HALT-EXACT |
| Exit-distribution collapse (Ouro) | trained exit | None trained | — |
| Joint-gate trajectory damage (Popescu) | gate gradient | None | — |
| Never-halt residue | HOLD margin | No incentive | NO-DONE label, censored |
| Coda lag (DONE before the coda is ready) | readiness window from τ = n | Trained on time | NOT-PREMATURE; R-answer beside |
| Raw-V head reads a clock | V_t content | Not primary | HEAD-STALL, CLOCK-GAIN |

---

## 5. Decisive readings (comparator-free, per cell, inside one model)

**Conditions.** All readings are fp32 at eval_bs 32 (211-7), on the card class used for eval_vb, on the §170 test set (1,000 items per n, n 1..16), with T_max = 24.

**C5-VB = HALT-EXACT ∧ NOT-PREMATURE ∧ STALL-SHIFT ∧ DEPTH-TIMING:**
1. **HALT-EXACT.** Pooled n 3..8, P(t_halt = n) ≥ **0.90** (never-halting and G-PROG abstentions count as failures).
2. **NOT-PREMATURE.** acc_halt ≥ acc(K′ = n + 2) − **0.02** at every n 3..8, and ≥ −**0.05** at every n 9..12 (173-2's PREMATURE line). acc_halt is the coda's answer at t_halt; censored items are read at T_max *and* also reported as wrong.
3. **STALL-SHIFT.** This is the registered §211 STALL intervention (the stored V_t := V_{t−1} at t = ⌈n/2⌉), rolled to 24 with R decoded at every t, on n 4..8 pooled.
   - Among items with intact t_halt = n: P(stalled t_halt = n + 1) ≥ **0.85**.
   - Fail-mode split: FROZEN-1 (stuck with exactly one operator alive), other never-halt, n, ≥ n + 2.
   - **UNREADABLE** with fewer than 200 qualifying items (= not met).
4. **DEPTH-TIMING.** At n = 9 and n = 10 each: P(t_halt = n) ≥ **0.60**, P(t_halt = 8) ≤ 0.10, and P(frozen at T_max with n − 8 operators alive) ≤ 0.10.

**Labels:**
- **CONTENT-HALT:** all four.
- **HALT-IN-RANGE:** 1–3 only.
- **CLOCK-HOLD:** at least 0.5 of n 9–12 items frozen with n − 8 operators alive.
- **EDGE-COLLAPSE:** at least 0.5 of n 9–12 items halt at t = 8 (or are G-PROG-flagged at t = 8).
- **STALL-COUNT:** FROZEN-1 ≥ 0.5 under STALL.
- **STALL-FRAGILE:** STALL-SHIFT fails without FROZEN-1, i.e. the duplicate breaks the trajectory.
- **NO-DONE:** never-halting > 0.10 at n 3..8.
- **CODA-LAG:** HALT-EXACT holds while NOT-PREMATURE fails in range.

**Descriptive** (never decisive):
- the t_halt histogram by n 1..16 (n 13–16 needs bank sizes above 13, never seen in training: a pure bank-size OOD test);
- the event-only mean;
- HALT-REGRET = max_K′ acc_k(n) − acc_halt(n);
- compute: mean t_halt / (best fixed K′ chosen on training items);
- R-answer accuracy and agreement with the coda;
- the COLLAPSE rate (passes with Δalive ≥ 2) by (n, t);
- **R-POS:** R's unchanged-slot accuracy at t = 1 by slot position (separates R's position OOD from step failure at T > 34);
- decoded-convergence and answer2 / MOVED2 halts (Prop 7 predictions);
- **double STALL** (two consecutive copies) → n + 2;
- **ADVANCE:** MCT edit of the stored V_t to E_{t+1} at ⌈n/2⌉ → n − 1. This separates reading the version from recompute-from-V_0 plus a clock, but it is valid only where W3 holds;
- **HEAD-STALL:** a raw-V logistic head (pooled LN(V_t) plus the " =" slot, 1,536 features) and an R-factored head (20 pooled class maxima), each fit on 2,000 training items (t ≤ K_i, detached) and read intact, stalled and deep;
- **CLOCK-PROBE:** a multinomial probe for t ∈ {n+1..n+4} on HOLD versions of training items (content constant), from raw features vs R logits; chance is 0.25;
- **CLOCK-GAIN:** the raw head's logit slope in t on HOLD versions, divided by its done / not-done logit gap, in passes of premature shift per pass of clock.

**OC block (rule a; one seed per cell)** [DERIVED from Prop 5 and the simulation; magnitudes ASSUMED].
- **No effect** (the halt follows a clock or the edge):
  - under K-type keying, P(STALL-SHIFT) ≈ 0 (simulated: never = 1.000 after a forced stall);
  - under the input-count halt, P(STALL-SHIFT) = 0;
  - under EDGE-COLLAPSE or CLOCK-HOLD, P(DEPTH-TIMING) ≈ 0;
  - so **P(C5-VB | no effect) ≤ 0.02**.
- **Effect** (a content-keyed core in a DEEP cell):
  - P(HALT-EXACT) ≈ 0.8 (needs σ ≲ 0.018);
  - P(STALL-SHIFT) ≈ 0.6 (duplicate robustness unknown);
  - P(DEPTH-TIMING) ≈ 0.5 (VB at n 9 also has T > 34; §190's analogue read 996 / 1,000 at n 9);
  - jointly, positively correlated, **P(C5-VB | effect) ≈ 0.35**.
- **Per cell.** P(content-keyed | DEEP) ≈ 0.5 [CONJECTURE: the "t ≥ 9" shortcut is zero-loss] and P(DEEP) ≈ 0.45–0.55, so **P(C5-VB per cell) ≈ 0.08–0.10** and P(≥ 1 of 3) ≈ 0.25.
- **A positive is strong evidence; a null is weak.** HALT-IN-RANGE alone is not a C5 claim: it is the input-count-equivalent regime without the depth part.

**Censoring (rule b):**
- never-halting by T_max, and G-PROG abstentions, fail every halting reading;
- a dead or NO-ENTRY cell fails all four;
- event-only values are reported beside each reading;
- realised $ is reported with the result.

---

## 6. Minimal experiment

**Step 1 ($0, eval-only, local, after landing; pre-register before any eval_vb.json is read).**
- **What:** a new `halt` part, e.g. boxes/vbank_halt.py, writing `<run>/eval_vbhalt.json`. The registered eval_vb parts and the score_s211 contract stay unchanged. Experiments implements it; it is not to be edited into a running module.
- **Per n 1..16:** one rollout to 24 passes with R decoded at every t, and the coda answer at every t 0..24 (the acc part does 18 K′ values; this is about 1.4× that). Then the STALL and double-STALL rollouts to 24 on n 4..8 (and 9..12, descriptive), plus ADVANCE on n 4..8.
- **Per-item fields:** t_halt, the G-PROG flag, alive(Ê_t) for t 0..24, the coda answer at each t, the R-answer, t_conv_dec, and answer2 / MOVED2. Keeping per-item rows also lets HALT-REGRET and the heads be recomputed with no rerun.
- **Heads and probes:** pooled features for 2,000 training items (t ≤ K_i) and the test rows, dumped once. They are then fit on CPU (≤ 3 threads, minutes).
- **Size:** about 1.5–2× eval_vb's acc part per cell. The memory peak is as in 211-7 (4.47 GiB at bs 32, n 16, K′ 24).
- **Where:** on a card **without a co-tenant**, not the card running eval_vb.
- **Cells:** N1, N2, N3 (N3 descriptive if NO-ENTRY by rule b).
- **Gates:**
  - (i) at t = n + 2, the coda answers reproduce eval_vb's acc_n2 bitwise in item counts;
  - (ii) a fixture: synthetic Ê sequences for C, K, E8, input-count and stuck trajectories score as Prop 5 predicts;
  - (iii) every read field is grepped in the module before data (rule).

**Step 2 (conditional training; a separate registration with the user's OK; derivation = §2.4 and Prop 8).**
- **Trigger:** the best cell reads CLOCK-HOLD, STALL-COUNT, EDGE-COLLAPSE or HALT-IN-RANGE, and is DEEP.
- **STALL-AUG-FT:** two arms on that cell's model.pt, each 8k steps at bs 64, n ≤ 8, LR 3e-5 constant, RTX 3090, the same card class:
  - **AUG** (p_s 0.3, S ∈ {1, 2});
  - **CTRL** (p_s 0; it separates continued training from the augmentation).
- **Cost:** 8k × about 0.5 s ≈ 1.1 h per arm, so **≈ $0.25–0.35 per arm, about $0.6 with deploy overhead**.
- **Readings:**
  - DEPTH-TIMING, CLOCK-HOLD and NOT-PREMATURE at n 9–12;
  - held-out stall forms: a **triple** stall, REWIND (append V_{t−2}, predicted halt n + 2) and ADVANCE. Single and double stalls are now trained.
- **Pre-stated predictions** [CONJECTURE]:
  - P(AUG removes CLOCK-HOLD | CLOCK-HOLD) ≈ 0.6;
  - P(DEPTH-TIMING at n 9 | AUG) ≈ 0.5, against ≈ 0.2 in CTRL;
  - n 11–12 lie outside the S_max = 2 band, so a gain there would show the content rule generalising beyond the band; none is predicted (P 0.7).
- **A full 48k STALL-AUG cell** (about $1.2–1.7) only if FT-AUG beats FT-CTRL on DEPTH-TIMING.

---

## 7. Top risks and falsifiers

1. **R's HOLD decoding is sloppy (NO-DONE).** Residue ν is too high and items never halt in range. *Falsifier of (a) as a usable halt:* never-halting > 0.10 at n 3..8 in a DEEP cell.
2. **The core reads a clock** (CLOCK-HOLD / STALL-COUNT). This is the zero-loss shortcut. It falsifies "content halting transfers in VB as trained", and STALL-AUG is the lever.
3. **Edge collapse (EDGE-COLLAPSE)** is the only route to a *confident* premature halt. G-PROG flags it, but it still fails C5.
4. **R's position OOD at T > 34.** Deep-n failures from R itself, separated by R-POS. If R-POS < 0.95 at slots above 34, DEPTH-TIMING is uninterpretable for the core.
5. **Duplicate fragility (STALL-FRAGILE).** A content-keyed core may break on the log-2 duplicate offset. This is not a clock signature, and the fail-mode split keeps the two apart.
6. **The PA gather's V_0 age clock** (N2 / N3 only). If N1 passes STALL-SHIFT and N2 does not, it is the candidate channel (descriptive, one seed).

**Falsifiers of this document's theory:**
- **Prop 2.** The R-factored learned head disagrees with DONE on more than 1% of in-range decisions.
- **Prop 6 / Prop 5's asymmetry.** In range, premature halts outnumber late halts in a DEEP cell (collapse would then dominate stalls).
- **Prop 5's stall invariance.** HALT-EXACT holds but acc_halt < acc(K′ = n) on stalled items.
- **Prop 8iv.** AUG removes STALL-COUNT but not CLOCK-HOLD. The deep freeze is then not the bank-size shortcut.

## 8. Predictions [CONJECTURE, before any §211 output]
- In a DEEP cell: HALT-EXACT P 0.8; NOT-PREMATURE in range P 0.75.
- Decoded convergence fires at n + 1 in range and at the stall pass under STALL (P > 0.9 each).
- answer2 is PREMATURE: there is no answer loss before τ, so pre-τ answers are unconstrained.
- At n 9–12, the leading failure is **CLOCK-HOLD over EDGE-COLLAPSE** (P 0.45 vs 0.25). R2 makes collapse costly, while "t ≥ 9 ⇒ hold" is free.
- N1 shows STALL-SHIFT at least as high as N2 (P 0.6).
- CLOCK-PROBE on raw features is above chance (≥ 0.5, P 0.6), and on R logits about chance (P 0.8).
