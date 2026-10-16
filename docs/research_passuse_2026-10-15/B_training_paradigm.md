# B. STAGGER + SLOT-IIT: timed-window counterfactual training for a pass-global attention memory

*Opus research subagent for Theory, 2026-10-15 (after 22:00 EDT). A draft for Theory and Audit, not a registration. Repo read-only; the coverage checks are CPU scripts in `scratchpad/B_scripts/` (`stagger_nested*.py`, `stagger_gsm*.py`, with JSON outputs). Tags used: [PROVED] [DERIVED] [ASSUMED] [EMPIRICAL] [CHECK] (computed on the generators) [CONJECTURE].*

## 0. Summary

- **The idea.** Rather than rewarding passes through answer likelihoods, which is gameable (Prop M), we change *when* each piece of the problem is true.
  - In a **window row**, each literal shows its real value only during a pass window W(ℓ) set by the CoT's consumption order. Before and after the window, the re-read text shows independent decoy values.
  - The label is the item's own answer. Rows where any single pass's text (a "snapshot") would give that answer are dropped.
- **What the optimum forces [PROVED, Props H/D/B].** An exact solution must do three things:
  - read the text on at least h(τ) ≥ 2 distinct passes (the hitting number of the windows);
  - ignore every literal outside its window, which kills pass-1 precompute, recompute from the re-read text, and decoder-side solving;
  - carry each literal's information in the slots written during its window.
- **Any single-snapshot policy scores 0 on kept rows** [PROVED]. That gives a decisive reading that needs no comparator.
- **SLOT-IIT** (the coordinator's natural unit). Splice the slots written during W(ℓ) from a donor run into the recipient's memory. The answer must follow the donor's ℓ.
  - This asks for more than STAGGER: each pass's slots must hold its information in a *history-independent re-coding*, usable on top of another run's history.
  - That is the operational meaning of "re-coding" here.
- **G8 is closed** [PROVED]. Every term is a CE to a label that does not depend on θ, on full-K rows. No term compares the model with its own truncation.
  - Marginal-pass rewards are rejected (Prop M). With gradient, they are met by "precompute plus clock-gated release". In a global memory the slot count is a free clock. Detached, they reduce to a reweighted CE.
- **GSM8K labels are gold answers** (C4). The windows need only the teacher-CoT's operand grounding (`gsm8k_data.parse_recomputable`). The exclusion rule needs the answers of the decoy problems (`G.render`, or a teacher model).
- **Coverage [CHECK]:**
  - nested: 32–63% of n 3–8 items kept;
  - GSM8K-Aug: about 12% of train items eligible, and only about 99 test items.
- **Honest limits:**
  - Schedule-robust windows force only about 2 reading passes (h ≈ 2.0–2.3). Tighter windows force 2–3.5 but lock in the CoT's order.
  - *Where* the arithmetic happens is not identified by any answer-level target. This is copy-then-compute; it is closed only by capability (A-CAP).
- **First steps:**
  - $0 G-ALIGN on the 16 kept nested models;
  - Stage 1: four 12k-step DeltaMem fine-tunes, a failed-seed rescue plus a converged-seed depth test (≈ $3–4.5, cap $5), runnable before the new architecture exists;
  - Stage 1b: a pilot on the pass-global architecture;
  - Stage 2: GSM8K, either on our harness (≈ $1.2 + a control) or as **CODI-STAGGER**. CODI is already a pass-global attention memory: latents in context. That costs ≈ $3.

---

## 1. The paradigm

### 1.1 Setting

**Architecture (assumed, per the user's constraint):**
- Pass p = 1..K reads a text x^(p) (normally the same problem every pass) and the memory M_{p−1} = (Z_1, …, Z_{p−1}) by content attention. It then appends its own slot set Z_p ∈ ℝ^{s×d}.
- The decoder is "pass K+1": it reads x^(K+1) and M_K and emits the answer.
- Writes are append-only, so Z_p is a function of (x^(p), M_{p−1}) only.
- DeltaMem is the special case where only the last state is kept. On it, "the slots of passes in W" degrade to the whole state after pass max W, which is §205's swap unit.

**The task as a program.** A template τ has literal slots Λ (nested: the leaves; GSM8K: the question literals that the equations use). The program P maps literal values to the answer. The literal values of item x are u.

**Consumption interval of literal ℓ** (union of two schedules):
- nested: κ(ℓ) is the leftmost-ready reduction index of ℓ's parent (the aux schedule), and η(ℓ) is the parent's height (the level-parallel schedule);
- GSM8K: κ ranges over the indices of the equations that use ℓ, and η over their DAG depths.

**Window:**

  W(ℓ) = [ max(1, min(κ,η) − c_a), max(κ,η) + c_l ],

with hi(W) := K+1 when ℓ is consumed by the final operation (the root or the last equation). That lets the decoder do the last op, as the current models do (§205 D4). ℓ is *switchable* iff W(ℓ) ≠ [1, K+1].

Defaults: c_a = 1, c_l = 0 on nested; c_a = c_l = 0 on GSM8K. Both union rules are chosen for schedule robustness, see §3 CHECK.

### 1.2 Window rows (STAGGER)

**Values at pass p**, for each switchable ℓ:
- v_p(ℓ) = u_ℓ if p ∈ W(ℓ);
- v_p(ℓ) = b_ℓ if p < lo(ℓ);
- v_p(ℓ) = a_ℓ if p > hi(ℓ).

**Decoys** b, a are drawn **i.i.d. from the base distribution, independent of u**:
- nested: uniform on Z_11, nonzero for a divisor;
- GSM8K: `draw_literal` at the literal's format.
- A decoy can therefore equal u (a "null decoy"). Independence is what makes Prop H exact. Decoys forced to differ from u would leak "u ≠ b".

**The rest of the row:**
- Texts: x^(p) = render(τ, v_p). Every x^(p) is a valid problem.
- **Label y = P(u)**, the item's own answer.
- **Keep iff** every snapshot is defined, y ∉ {P(v_p) : p = 1..K+1}, and no x^(p) is in the test-exclusion set (else redraw the decoys).
- **Loss:** L_W = CE( dec(M_K, x^(K+1)), y ), where M_K is built by passes 1..K on x^(1..K). K per row is as in the base recipe (WINDOW's K = n + U{lo..hi}).

**Sampling:**
- **Nested:** up to B_W = 32 window rows per step, built from the step's own items (own rng, seed + 21100, no training-rng draw, as `iit_serial.make_pairs`). Their clean versions stay in the clean batch, so the clean path stays paired to the control.
- **GSM8K:** every eligible training draw is presented *only* as a window row, and the eligible share of each batch is oversampled to 25%. This removes the "clean eligible item" cue that a dual-mode policy would need (N1, §2). On GSM8K the aux targets ` <<e_t>>` are evaluated at u, so they are well defined on window rows. On nested, window rows carry no aux, because e_{t+1}'s string contains unconsumed literals.

### 1.3 Slot-interchange rows (SLOT-IIT)

**Construction:**
- Recipient x. Pick a switchable ℓ with hi(ℓ) ≤ K, plus, where one exists, a late literal ℓ_L with lo(ℓ_L) > hi(ℓ).
- Donor x_d = x with ℓ ← ℓ′ and ℓ_L ← ℓ_L′.
- Run the donor's passes 1..hi on x_d, which gives Z^d.
- Run the recipient's passes 1..lo−1 on x, which gives Z.
- Splice: M = (Z_1..Z_{lo−1}, Z^d_lo..Z^d_hi). Continue passes hi+1..K on x, and the decoder on x.
- **Label y_hyb = P(u; ℓ ← ℓ′).** Keep iff y_hyb ∉ {y, P(x_d)}.
- L_I = CE(output, y_hyb). B_I = 8 rows, which is 16 row-forwards.

**What it rules out.**
- Reading ℓ before lo puts u into the recipient's own slots, giving y.
- Re-reading ℓ after hi gives u again, so y.
- Reading ℓ_L inside W gives the donor's ℓ_L′, so P(x_d).
- A slot that stores a pointer into its own run's history, rather than a self-contained value, resolves into the recipient's history after the splice and fails.

**Evaluation form, SERIAL-SLOT** (held out from training): single-pass swaps W = {t} at t ≥ 2, plus swaps of *intermediate* subtrees (multi-literal donors). Neither family is trained.

### 1.4 Total loss, schedule, gradient paths

  **L = L_ans + λ·L_aux + ω(s)·( w_W·L_W + w_I·L_I )**,  with w_W = 1, w_I = 0.5.

ω(s) ramps 0 → 1 over steps 2k–6k from scratch. That is §205's Prop O′ logic: before the memory carries anything, window labels would train decoder-side guessing. In fine-tunes, ω = 1.

| term | rows | flows into | stop-gradients |
|---|---|---|---|
| L_ans, L_aux | clean batch (base recipe; NOPASS aux on GSM8K) | all params | as base |
| L_W | window rows, one K-pass forward with per-pass texts | all params through **every** pass. The early passes' gradient is exactly the reward for their timed reads (Prop D) | none (exact label) |
| L_I | recipient prefix, donor 1..hi, recipient hi+1..K, decoder | write projections of the donor's passes (the spliced content); the recipient's prefix (penalising read-ahead); the late passes' and decoder's reads | none. The text tokens are inputs. The donor's history is kept in the graph: re-coding must be learned by the passes that write Z^d |

**No term is evaluated at a truncated K.** K′ comparisons appear only in the readings.

### 1.5 Audit's three levers

**(1) Per-pass information gain, adopted in causal form.** Prop D gives it directly: for each switchable ℓ, u_ℓ's information enters memory only through the slots of W(ℓ), and the output must depend on it.
- That is a per-pass information-gain floor, I(M_{hi(ℓ)}; u_ℓ) > 0 *and used*.
- It is two-sided: reading outside W is penalised by decoys.
- It is tied to task information (the answer).
- A probe-based "slot t must decode new content" reward is rejected as presence-not-use (§203) and G6/G7-prone (§204).

**(2) Read-side limits: optional architecture hook.** With Q slot reads per pass, "copy every literal, compute at the end" needs about ⌈L/Q⌉ post-window passes to gather L operands. So combining early is favoured [DERIVED under A-READ: one read retrieves about one slot's content. Soft attention can blend, so this is heuristic]. It is not needed for Props H/D/B.

**(3) Training-only past-slot dropout: rejected** [DERIVED].
- On clean rows, a dropped slot's information is still in the text, so dropout trains re-reading. That is G2, the failure mode itself.
- On window rows the label then becomes unreachable (label noise).
- It also pushes toward redundant carry-forward, which is RNN-like and against the user's constraint.

### 1.6 What is deliberately not used

**Prop M (marginal-pass rewards) [PROVED].**

Setup. Let R = log p_K(y) − log p_{K−j}(y) (or any increasing φ of it) for random K, j.

(a) **With gradient through both terms**, R is met without iteration. The construction:
- pass 1 computes logits ℓ* for y and writes them;
- the decoder's readout is g(c)·ℓ*, where c = |M| (the slot count is free in a global memory; DeltaMem uses its additive-norm counter, §185);
- g is increasing.
Then p_t(y) is strictly increasing in t whenever ℓ* is right, so R > 0 for every K, j. This "timed release" is G8 by construction.

(b) **With the truncated term detached**, ∇R = ∇ log p_K. That is a reweighted answer CE, with no term that separates iterative from one-pass solutions of equal p_K.

**Anchoring** p_{K−j} to a frozen pass-1 reference is case (b) with a constant. **RL per-pass advantages** built from likelihood gains inherit (a). ∎

The repair is to make the per-pass credit interventional, with exact labels: windows and splices.

**Curriculum on depth vs K.** STAGGER subsumes it: pass 1 never sees a late literal's true value, whatever its capability.

---

## 2. Goodhart table

| route | status under STAGGER + SLOT-IIT | why |
|---|---|---|
| G1 pass-1 precompute (global memory: "pass 1 writes everything") | **CLOSED** on kept rows [PROVED, Prop B] | late literals show decoys at pass 1, and P(x^(1)) ≠ y. Within one-pass capability, outside any kept row, precompute stays legal (as C1 allows) |
| G2 recompute from the re-read input (the failed-seed / §207 mode) | **CLOSED** [PROVED] | consumed literals show decoys after W, and P(x^(K+1)) ≠ y. SLOT-IIT: re-reading gives y ≠ y_hyb |
| G3 decoder-side solving | **CLOSED** [PROVED] | the decoder's text is the last snapshot, so its answer ≠ y |
| G4 copy literals, compute later | **OPEN in principle** [PROVED: copying and computing are observationally equivalent at the answer]. Closed for deep items under A-CAP [DERIVED, Prop S(iii)] | no answer-level target identifies where the arithmetic happens. Readings: INT-PROGRESS-type probes and SERIAL-SLOT on intermediate subtrees |
| G5 last-pass answer register | harmless | a register must now be computed from carried information. It is not a bypass |
| G6 per-pass targets met by a pass-index readout | **CLOSED** for window rows; NOPASS aux on GSM8K | pass 1 cannot know late true values, so there is nothing to index |
| G7 one-sided dynamics targets | **N/A** | no dynamics target is used |
| G8 early-pass sabotage | **CLOSED** [PROVED, Prop G8] | no likelihood enters with a negative sign. Degrading early passes lowers P(F = y) (Prop H) |
| G9 (global) later passes read pass-1 slots directly | allowed | addressability is the design goal |
| G10 (global) "idle" later passes | **CLOSED for hitting passes** [PROVED, Prop D] | each window must deposit its literal's information. Passes after the last window may idle, which is natural |
| N1 switch detection / dual mode (precompute on clean, carry on windowed) | **OPEN**, mitigated | on nested there is no incentive: a one-pass answer is at chance for n ≥ 3. On GSM8K, eligible items are never shown clean in training. Screened by ITER-GAIN on clean items (§5) |
| N2 record every reading, pick the in-window one later | harmless [PROVED] | equivalent to timed consumption |
| N3 schedule lock-in (CoT order imposed) | **OPEN by design**, bounded | the union windows accept both the sequential and the level-parallel schedules [CHECK: parallel-in-window 1.0]. The tighter sequential windows do not (0.48 at n 8). G-ALIGN chooses |
| N4 history-coded slots (pointers into the run's own history) | **CLOSED by SLOT-IIT** (training) | splices put the slots on a foreign history |
| N5 decoy confusion lowering clean accuracy | risk | failure label LOSES-ONE-PASS |

---

## 3. Math

**Assumptions:**
- **A-DEP:** each switchable ℓ matters: P_others( y is non-constant in u_ℓ ) > 0. In nested this fails only through a ×0 ancestor.
- **A-IND:** decoys are independent of u given the template.
- **A-CAP [EMPIRICAL / ASSUMED]:** one pass computes at most c_max ≤ 2 tree levels from memory plus text, and the decoder at most d_R ≈ 1–2 (§205 D4; SEEDSTAB2 d_R ≈ 3–4 as an upper value).

**Prop H (hitting-set necessity) [PROVED].**
- Let F be any K-pass system (DeltaMem, a slot memory, or CODI latents). Let R(F) be the set of passes whose text can change F's output on the row support.
- If P(F = y) = 1 on the unconditioned window-row distribution, then R(F) ∩ W(ℓ) ≠ ∅ for every switchable ℓ, so **|R(F)| ≥ h(τ)**, the minimum hitting set of {W(ℓ)}.
- Proof: if R(F) misses W(ℓ), F is a function of texts in which ℓ's value is an independent decoy (A-IND). So F ⊥ u_ℓ given the rest, while y is not (A-DEP). Hence P(F ≠ y) ≥ E[1 − max_v P(y = v | rest)] > 0. ∎
- The same argument gives **out-of-window invariance**: the exact F does not depend on b_ℓ or a_ℓ.

**Prop D (memory mediation; per-pass information gain) [PROVED].**
- For p > hi(ℓ), x^(p) is independent of u_ℓ. So the output's dependence on u_ℓ factors through M_{hi(ℓ)} (data processing).
- With append-only writes, u_ℓ's information *enters* memory only in Z_lo..Z_hi.
- So for an exact F, the slots written during W(ℓ) carry u_ℓ-information that is used (output dependence, not decodability). ∎

**Prop B (snapshot bypasses) [PROVED].** Any policy whose output is the correct answer of one pass's text has accuracy exactly 0 on kept rows, by the exclusion rule.
- A policy that is a function of one snapshot but computes something else is correct only by coincidence: about 1/11 on nested if near-uniform [DERIVED].
- G1, G2 and G3 are snapshot policies at passes 1, any p, and K+1.

**Prop S (nested: what the optimum rules out).**

If an exact solution exists in the model class, the population optimum of L (clean rows plus window rows, ω = 1) consists only of policies that satisfy:
- (i) [PROVED] they read the text on ≥ h(τ) passes; they are invariant to out-of-window values (no read-ahead beyond c_a, no re-reading after the window, no decoder-side reading of consumed literals); and every literal's effect passes through the slots of its window. They are therefore *state-carrying and timed*.
- (ii) [PROVED, with SLOT-IIT at w_I > 0] the information in Z_W is history-composable: it survives a splice onto another run's prefix.
- (iii) [DERIVED, under A-CAP] if the last window that closes before K+1 does so at pass p*, the arithmetic of at least H − c_max·(K − p*) − d_R tree levels must happen by pass p*. So copy-then-compute-at-the-end fails deep items.

What remains unidentified: the arithmetic timing within that bound, and behaviour on items outside the kept set.

**Strength vs robustness [CHECK, 4,000 draws per n, K = 8; `stagger_nested*.json`].** Each cell is "kept share / mean h(τ)". The last row is the share of switchable leaves whose level-parallel read pass falls inside the sequential window (1.0 by construction for the union windows).

| n | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| union (1,0) | .49 / 2.0 | .32 / 2.0 | .41 / 2.2 | .39 / 2.1 | .36 / 2.2 | .35 / 2.2 |
| sequential (1,0) | .49 / 2.0 | .63 / 2.0 | .56 / 2.4 | .50 / 2.8 | .44 / 3.2 | .40 / 3.5 |
| parallel read inside the sequential window | 1.0 | .91 | .78 | .65 | .55 | .48 |

- Kept rows start at n = 3, exactly DIRECT's boundary n* = 3.
- Union windows force about 2 timed reading passes and accept both schedules. Sequential windows force up to 3.5 but penalise level-parallel reading on half of the deep leaves.

**Prop G8 [PROVED].**
- Every term is −log q_θ(label | row), with the label fixed independently of θ, and every row runs the full K.
- A "sabotage" change that lowers early passes' information about in-window literals lowers sup P(F = y) on window rows (Props H/D). It cannot lower L.
- No term rewards a gap between K and K−j, so G8 and its timed-release variant (Prop M) have no gradient incentive. ∎

**Prop GSM (does STAGGER make pass-1 precompute suboptimal on GSM8K?) [DERIVED].**

Take a 0–1 proxy on the eligible stratum E, with window-row share w_E of E's loss. Let a_1 be the accuracy of the best one-shot policy and a_C that of the best band-respecting (carry) policy.
- **Single-policy lemma [PROVED]:** a band-respecting policy outputs on a window row exactly what it outputs on the clean problem with values u. So its window accuracy equals its clean accuracy a_C. One-shot policies score 0 on kept rows (Prop B).
- Carry is preferred iff **a_C > (1 − w_E)·a_1**.
  - With §1.2's GSM8K setting (eligible items only windowed), w_E = 1, so carry is preferred whenever a_C > 0.
  - With an even mix (w_E = 0.5), carry needs a_C > 0.5·a_1, i.e. ≈ 0.055 against a one-pass ≈ 0.11 at m ≥ 3 (§209 A).
- **The dual mode** (detect windowing, precompute when clean) beats carry by (1 − w_E)(a_1 − a_C)⁺. That is zero at w_E = 1, which is why eligible items are never shown clean.
- **The binding condition is existence and learnability** of a carry policy with a_C > 0 at this budget. That is not guaranteed: §208 found passes 2–10 add nothing. STAGGER makes the one-pass optimum worth 0 on E; it does not make the carry optimum easy to find.

**GSM8K coverage [CHECK; 8,000 train draws, K = 10; recomputable share by m 1–6: .74 / .69 / .56 / .43 / .29 / .29].** Cells are "kept share of all items / mean h".

| windows | m 2 | m 3 | m 4 | m 5 | m 6+ | train eligible overall |
|---|---|---|---|---|---|---|
| union (0,0), default | .24 / 2.0 | .08 / 2.3 | .06 / 2.1 | .03 / 2.1 | .02 / 2.0 | ≈ 12% |
| sequential (0,0) | .24 / 2.0 | .21 / 2.1 | .13 / 2.6 | .05 / 3.3 | .05 / 2.6 | ≈ 20% |
| union (1,0) | 0 | .04 | .02 | .02 | .01 | ≈ 2% |

- **Test (union (0,0)):** m 2 .17, m 3 .05, m 4 .05, m 5+ .03, i.e. about **99 eligible test items**.
- GSM8K's DAGs are shallow. Most literals enter at depth 1, so the schedule-robust windows mostly force "the depth-2 literal is read at pass ≥ 2".
- The windowed stratum is therefore mostly m ≤ 3, where one pass partly succeeds. The forcing is weakest exactly where text CoT's 2.2× deep-item advantage (§209-3) lives.

---

## 4. Supervision scalability (C4)

**What the paradigm needs, per item:**
- the gold final answer (the label);
- the operand grounding of a text-CoT teacher's steps (which question literal each step uses, and step dependencies), for the windows;
- the final answers of decoy problems, for the exclusion rule.

**Where each comes from:**
- On GSM8K-Aug all three exist now: `parse_recomputable` grounds 56–74% of items with m ≤ 3, and `G.render` executes decoys.
- At scale, the grounding comes from any CoT teacher whose steps cite numbers (calculator-annotated CoT or program-of-thought), as GSM8K-Aug's GPT-4 equations did.
- The decoy answers can come from re-executing that program, or from **the teacher model answering the decoy problem**. That is a final answer to a natural problem, not an intermediate state.

**No hand-built intermediate states are used.** SLOT-IIT's y_hyb is also the final answer of a natural problem (x with ℓ ← ℓ′).

**The limit** is the grounding: items whose literals are words ("twice", "half") or ambiguous are not eligible (§3: 26–71% lost by m). This is the same population restriction as §200 B's R_x.

---

## 5. Predictions, decisive readings, failure labels

**Comparator-free readings:**
- **WINDOW-ACC** on window rows built from *test* items (n 3–8, and n 9–12 as held-out depth; GSM8K: the ≈ 99 eligible test items × 5 decoy draws, scored per item).
  - No single-snapshot policy exceeds 0 (Prop B), so **WINDOW-ACC ≥ 0.5 proves timed multi-pass reading with carry on held-out items.**
  - It is trained directly, so it is a check that the objective was met, not a C1 claim.
- **C1 on clean items (the claim):**
  - nested: THINK-NECESSARY (K′0 / K′1 ≤ 0.2 at n ≥ 3), DEEP, and SERIAL-SLOT at untrained t ≥ 2 (clean-ccc hybrid ≥ 0.6);
  - GSM8K: **ITER-GAIN_rr = ACC − K′1 ≥ T = 0.028 pooled and ≥ T_D = 0.034 at m ≥ 3** (§208's T's), with K′1 = pass 1 and then the decoder on the clean text.

**Labels:**
- **STAGGER-SERIAL** (success) = WINDOW-ACC ≥ 0.5 ∧ the clean C1 reading ∧ SERIAL-SLOT.
- **DUAL-MODE** = WINDOW-ACC ≥ 0.5 but clean ITER-GAIN / THINK-NECESSARY not met. N1 realised; this falsifies STAGGER as a C1 lever.
- **WINDOW-FAIL** = WINDOW-ACC ≤ 0.15. The objective is not met: capacity, alignment or budget.
- **LOSES-ONE-PASS** = clean ACC (pooled; GSM8K m ≥ 3) below the paired control by more than T (0.05 on nested), without the clean C1 reading. The objective cost accuracy without inducing iteration.
- **READ-AHEAD** (G-ALIGN only) = converged models' WINDOW-ACC under sequential windows < 0.6 × their union value. They copy future literals early; choose the union windows.

**Priors [CONJECTURE]:**

*G-ALIGN ($0, existing nested models).*
- Converged (rms s1/s2, window s1/s2 etc.): WINDOW-ACC (union) 0.35–0.75.
- Failed (rms s3, den, hold, …): ≤ 0.10, since they follow the re-read input (203-8) and so output the last snapshot.
- Separation ≥ 0.25: P ≈ 0.75.

*Stage 1, F pair (rescue of the failed rms s3).*
- **RESCUE** = SERIAL t = 1 clean-ccc hybrid ≥ 0.6 in F1 (now 0.003, 203-4) and ≤ 0.2 in F0, plus INPUT-FOLLOW ≤ 0.4 in F1.
- P(RESCUE) ≈ 0.45. F1 WINDOW-ACC central 0.55. LOSES-ONE-PASS P ≈ 0.15.

*Stage 1, C pair (converged s193_rmswin_s1).*
- DEEP is retained (n 3–8 ≥ 0.90): P ≈ 0.85.
- n 9–12 pooled C1 − C0 ≥ +0.05: P ≈ 0.35. Central +0.03 from about 0.5–0.7 (the count limit is not mainly a recompute problem).
- WINDOW-ACC at n 9–12 ≥ 0.4: P ≈ 0.5.

*GSM8K (Stage 2).*
- ACC central 0.13 (0.09–0.18).
- P(ITER-GAIN met) ≈ 0.25; P(WINDOW-FAIL) ≈ 0.35; P(LOSES-ONE-PASS) ≈ 0.35; P(DUAL-MODE) ≈ 0.10.
- CODI-STAGGER: ACC central 0.18 (§209: 0.189); P(ITER-GAIN met) ≈ 0.3.

**OC block (rule a):**
- **Stage 1 RESCUE:** one paired start.
  - P(| no effect) ≈ 0.05–0.10. That is the chance that a fresh-LR kick alone moves a failed seed while F0 does not; F0 measures the kick.
  - P(| effect) ≈ 0.45. So LR ≈ 5–9: a pilot. A second failed start (den s1 or hold s1, ≈ $2 more) gives 2/2: ≈ 0.005 vs ≈ 0.2.
- **GSM8K ITER-GAIN** (test SE ≈ 0.012):
  - P(| no effect) ≈ 0.01–0.02 one-sided;
  - P(| a true +0.05) ≈ 0.95.
  - A positive is strong evidence at one seed; a null is weak.
- **WINDOW-ACC on 99 GSM8K items:** SE ≈ 0.05, so a rate claim needs the n 3–8 nested analogue as well.

**Censoring (rule b):**
- A cell that dies, is aborted, or ends before its step count reads every label as **failed**, not missing.
- WINDOW-ACC with fewer than 80 eligible items is UNREADABLE, which counts as not met.
- Event-only values are reported beside each label.

---

## 6. Experiments

**Stage 0 ($0; 3070, bf16, the original eval bs).**
- **G-ALIGN:** WINDOW-ACC under the union and sequential (1,0) windows on the 16 D7 models (203-4's set), on test items n 3–8. Also SERIAL-SLOT's DeltaMem form (a whole-state swap at hi(ℓ)), as a check against 203-4.
- **Decision:** proceed iff converged median ≥ 0.35 and failed max ≤ 0.20. Choose union vs sequential by READ-AHEAD.
- **Gates:**
  - G-SNAP: 10k rows, each snapshot re-evaluated by `eval_tokens`, all ≠ y, the label = y;
  - G-EXCL;
  - G-PLAIN: windows of width K+1 reproduce the clean eval bitwise.

**Stage 1 (DeltaMem fine-tunes; runnable now, independent of the new architecture).**
- **Cells.** All are 12k steps from a kept final model.pt: fresh AdamW, lr 5e-5, warm-up 200, cosine to 0; γ = fade = 1 from step 1 (a `--init_model` entry, engineering-only); RTX 3090; one GPU class.
  - **F0** s185_rms_s3 + the base recipe (control);
  - **F1** = F0 + STAGGER (w_W 1, B_W 32, windows from G-ALIGN; no SLOT term on DeltaMem, to keep attribution clean; §205 covers the whole-state swap);
  - **C0** s193_rmswin_s1 + its recipe (per-row K via the masked kmatch path);
  - **C1** = C0 + STAGGER.
- **Code:** a new `boxes/stagger.py`, imported only under `--stagger`. Per-pass texts go through `think_inputs` per snapshot. Renders keep the length (every value is one token), so the masks align. `--cache_lower` is off on window rows. The pairing / rng pattern follows `iit_serial.make_pairs`.
- **Gates:**
  - G-PLAIN (the flag off is bitwise to the control for 200 steps);
  - per-term gradflow (L_W nonzero into mem.qkv / wo / trunk; no `@torch.no_grad` on the path, checked by ast);
  - mem_check at B_W 32 with the largest shape;
  - G-TIME.
- **Cost:** base ≈ 0.89 s/step (§205 D3); 32 window rows × K add ≈ 0.35 s, so ≈ 1.25 s/step. F1 / C1 ≈ 4.2 h each, F0 / C0 ≈ 3 h each, ≈ 14.4 h ≈ **$2.6–4.0, cap $5.**
- **Readings:** §5 (RESCUE; DEEP / n 9–12; WINDOW-ACC; LOSES-ONE-PASS), plus 203-6 INPUT-SWAP and path_indep on F1.

**Stage 1b (the pass-global attention memory, when built).**
- Nested from scratch: **A0** = the arch with the base recipe; **A1** = A0 + STAGGER + SLOT-IIT (B_I 8, w_I 0.5).
- Seed 1, 24k steps (entry is decided by about 19k, §205 D3), 3090.
- Cost ≈ (the arch's s/step) × 24k × (1 + 1.35). At ≈ 1 s/step that is ≈ 16 h ≈ **$3–4.5**.
- Positive (A1 STAGGER-SERIAL, A0 not) → 3 seeds × 48k (≈ $15–20, which needs the user's OK).

**Stage 2 (GSM8K; after Stage 1 or 1b reads RESCUE or STAGGER-SERIAL).**
- **(a) Our harness:** `gsm_stagger_s1` = §200 A's flags + `--aux_nopass` + `--stagger` (union (0,0); eligible draws windowed only, at a 25% share). 24k steps ≈ 4.5 h ≈ **$1.2**. The control is A + `--aux_nopass` (≈ $0.9) if no such cell exists.
- **(b) CODI-STAGGER:** §209's CODI4 design, with the question's KV recomputed from x^(t) at latent step t, and the answer prompt on x^(K+1). The teacher sees the clean u-problem, so its CoT is `G.render`'s equations.
  - CODI is already a pass-global attention memory (each latent is one pass's slot), so SLOT-IIT is a latent-KV splice.
  - Comparator: §209 CODI4 (0.189). ≈ **$3** (re-encoding the question per latent ≈ 2× §209's $1.51).
- **(b) is the recommended GSM8K test** under the user's constraint. Total ≤ $5.

---

## 7. Top 3 risks, and what would falsify the design

1. **LOSES-ONE-PASS / unlearnable carry on GSM8K.**
   - STAGGER sets the one-pass solution to 0 on the eligible stratum but supplies no easier path to the carry solution. §208 shows passes 2–10 learned nothing at this budget.
   - If GSM8K ACC falls by more than T with WINDOW-ACC ≤ 0.15, the objective is right but the architecture or budget cannot meet it. Stop on GSM8K; the nested result stands alone.
2. **DUAL-MODE (N1).**
   - Window rows teach "carry when the text changes" without changing clean behaviour.
   - **Falsifier:** WINDOW-ACC ≥ 0.5 in F1 / A1 / G-cells while the clean reading (SERIAL t = 1, THINK-NECESSARY, ITER-GAIN) is unchanged from the control. That falsifies STAGGER as a C1 lever: it would train a switched-input skill, not clean multi-pass use.
3. **Weak forcing vs schedule lock (N3), and thin GSM8K coverage.**
   - Schedule-robust windows force only h ≈ 2 reading passes. Tighter windows impose the CoT order (48% of deep leaves conflict with level-parallel reading).
   - On GSM8K only ≈ 12% of train items and ≈ 99 test items are eligible, mostly m ≤ 3.
   - **Falsifier for the nested value-add:** G-ALIGN shows converged models already at WINDOW-ACC ≥ 0.8 and failed ≤ 0.1, *and* §205's IIT rescues s3. STAGGER then adds nothing over §205 on nested, and its case rests on GSM8K and on SLOT-IIT's re-coding in the new architecture.

**Account falsifier (any stage):** a STAGGER cell meets WINDOW-ACC ≥ 0.5 while SERIAL-SLOT on untrained intermediate-subtree swaps stays at the control's level. Timed reading would then be achieved by copying, without composable re-coding (G4 / N4). The "re-coding" claim fails, and the read-budget hook (§1.5 (2)) becomes the next lever.
