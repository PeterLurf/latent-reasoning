# VERSION-BANK + READY-RECODE: a pass-global attention memory and a training paradigm that force and reward the use of passes (DRAFT, 2026-10-15 22:59 EDT)

**Status.** This is a draft from the deep research session, not a registration. Registration and any spend need the user's OK, relayed by Audit.

**Requests.** The user's request (typed in Audit's session at 21:58 EDT, relayed verbatim): "invent a training paradigm and recurrent memoery archtiecture that forces and rewards uses of passes and recoding information. maybe do something regards attention, up to u". The constraint (22:07 EDT): "make sure the memory can be is pass global … dont make it like rnn memory make it more like attention memeory".

**Process.** All subagents were Opus.
- Prior art: docs/LIT_PASSUSE_PRIORART_2026-10-15.md (81 IDs verified, plus full-text addenda A / B / C).
- Three independent inventors: architecture (A), objective (B) and forced re-coding (C).
- A synthesis spec, then a proof pass and a red team over the synthesis.
- Everything is in docs/research_passuse_2026-10-15/ (A / B / C, SYNTH_spec_v0, REDTEAM_v0, PROOFS_v0).

This draft integrates all of it. It adopts every red-team fix (REDTEAM §G, items 1–8) and every proof correction (PROOFS verdict table).

**Tags:** [PROVED], [PROVED UNDER X], [DERIVED], [MEASURED], [ASSUMED], [CONJECTURE].

---

## 1. Why: what the evidence says must be forced

1. **GSM8K:**
   - Every latent recipe recomputes from the re-read question (§207).
   - When the question is shown only once, pass 1 computes everything and later passes are idle (§208, PASS1-PRECOMPUTE).
   - Per-pass targets are met by precompute plus a pass-token readout (§204).
   - The coupling is met by the decoder (§206).
2. **Nested seeds:**
   - Failed seeds recompute from the input (203-6 / 203-8).
   - **§205 s3 (2026-10-15):** an interchange loss made a failed seed carry and use its state (hybrid 0.996 vs 0.003) without giving it depth. It never entered, and its accuracy equals the control's.
   - So carrying is not the limit. **The per-pass step computation on deep items is** [CONJECTURE, Audit-agreed].
   - **A design must reward each pass for computing something new, not only for carrying.**
3. **Literature (LIT doc):**
   - Retrofitted loops keep the input reachable and plateau at the trained depth (McLeish 2511.07384).
   - Huginn recomputes from the re-injected input (2609.01924, confirmed in full text).
   - ACT and PonderNet collapse to one loop (RecurTrace 2609.03379).
   - Improvement rewards get gamed by degrading the first attempt (Cho 2505.23060).
   - Overwrite-only loops forget, while attention over all iterations does not (InfiLoop 2610.11570).
   - LOTUS's loop is used (3B: 22.7 at R = 1 → 70.0 at R = 6).
   - No unconditional finite-n depth lower bound applies to GPT-2 at n ≤ 16, so "one pass is insufficient" must be measured inside the model.

## 2. The design in one paragraph

**VERSION-BANK (VB)** splits GPT-2 small into three parts:
- a **prelude** (blocks 0–5, run once on the problem);
- a **shared core** (blocks 6–9, looped K times);
- a **coda** (blocks 10–11 + ln_f + LM head, which never sees problem tokens).

**Each pass writes a whole new version V_t of the problem**: T slots, one per position. Versions are **appended to a bank, never overwritten.** Every core layer of every later pass, and the coda, reads the bank by attention. Under the main variant **PURE-ANNEAL**, a pass's starting state is itself an attention read over the bank. So, by the end of training, the only cross-pass channel is attention over stored versions, which is the user's pass-global memory.

**READY-RECODE** trains three things:
- an answer loss only once the item could be done (the readiness window);
- a **linear per-slot recode target**: version t must decode as the expression after t reductions, weighted toward the newly computed values, so each pass is rewarded for one new reduction rather than for copying;
- optionally, **minimal-correction teacher forcing**, which trains the step map on correct inputs at every depth (the §205 diagnosis).

The decisive readings come from inside the model:
- **PROBE(t):** what a fresh readout can decode after t passes;
- **VALUE-SWAP:** splice one pass's reduced value from a donor;
- **STALL:** a copied pass, to detect a pass counter.

## 3. Architecture

**Notation:** P = 6, d = 4, c = 2; D = 768; nested T ≤ 34; K_max = 12.

### Prelude
x = [EOT, problem, " ="] gives V_0 = h_6 (`deltamem.gpt_lower`, causal). The bank starts as 𝔅_0 = {V_0}.

### Core, at pass t = 1..K
**Pass input.**
- **PURE-ANNEAL** (main; REDTEAM §E):

  U_t⁰[j] = Σ_{s<t, i} softmax_{(s,i)}( ℓ_{j,(s,i)} − c(step)·1[(s,i) ≠ (t−1, j)] ) · N(V_s[i])

  - ℓ_{j,(s,i)} = (W_q wpe[j])·(W_k LN(V_s[i]))/√64 + γ_q·(t−1−s), with **content-free position queries**.
  - The values are the per-slot RMS-reset stored vectors N(V_s[i]) themselves (no value or output projection).
  - c is a **scheduled buffer**, not a parameter: 1e4 up to 1k steps, linear to 0 by 8k. A learned −∞ init would self-lock (Shapiro's zero-gate lock, LIT addendum A).
  - At c = 1e4 this is bitwise CARRY; at c = 0 it is an attention read over every version.
  - [PROVED, REDTEAM §E] ‖U‖ ≤ max ‖N(V)‖, independent of t, so there is no additive growth.
- **CARRY** (control): U_t⁰ = N(V_{t−1}) (per-slot RMS reset, gain g init 1).
- **PURE** (raw, q + wpe) is dropped. At init it writes the same version every pass (cos(V_t, V_{t−1}) = 1.000 [MEASURED, REDTEAM F4]): the degenerate fixed point of 2604.11791.

**Core layer ℓ ∈ {6..9}.**
- Keys / values = [bank entries (s, i), s < t, all i] ∪ [current version, causal].
- Bank K/V come from block ℓ's own ln_1 + c_attn applied to the stored V_s[i]. They are cached per (s, ℓ) **with gradient**, in a preallocated, masked bank (not a growing torch.cat; REDTEAM F9e).
- Bank logit bias: β_{ℓ,h} + γ_{ℓ,h}·(t − 1 − s) for s ≥ 1, and **a constant β⁰_{ℓ,h} for V_0, with no age term.** With an age term, V_0's logit is an absolute pass clock (PROOFS §5).
- γ is initialised to ALiBi slopes (2108.12409) on 6 heads and to 0 on the other 6 (REDTEAM F7).
- **β is calibrated per head at step 0**, so the 99th-percentile bank mass is ≤ μ = 0.02 on a calibration batch (PROOFS §9: "β = −5 ⇒ ≈ 1.3%" is false as a bound).
- This needs a new block function with a float mask: `deltamem._block` is causal-only [VERIFIED deltamem.py:256].

**Write.** V_t = U_t⁴, appended.

**No pass token or pass position.** The remaining implicit clocks (the bank normaliser, content) are tested by STALL (§7).

### Coda after pass t
- Blocks 10–11 run causally over V_t (plus decoder tokens on GSM8K, prelude-encoded among themselves only), with the bank at a calibrated β^𝒟.
- The nested answer is read at the per-item " =" slot (len_i − 1).
- K′ = 0 is the coda on V_0.

### Recode head
R = LN + Linear(768 → 20) per slot (VOID, EOT, " =", " (", " )", 4 ops, 11 values). It is linear, so it cannot compute.

### How VB differs from CODI and the looped LMs
**CODI / Coconut** [PROVED, PROOFS Thm 2.2]:
- Same-layer KV access adds width, fan-in and aggregation, but **no depth**.
- All of CODI's depth growth runs through its one fed-back vector per step. That channel is narrow but persistent, so "RNN-like" is an overclaim.
- VB's depth-carrying channel is a **whole T-slot version per pass**, stored as outputs and read raw at every core layer (the Feedback Transformer pattern, 2002.09402).
- Under PURE-ANNEAL that channel is the attention memory itself (L1 holds through attention for PURE-type inputs; PROOFS §1).

**Versus RecurTrace (2609.03379), InfiLoop (2610.11570) and Think-at-Hard (2511.08577):**
- RecurTrace: layer ℓ reads its own earlier loop outputs, same position, W = 3, with a carry.
- InfiLoop: one scalar-scored mixture of all iterations' states.
- Think-at-Hard: same-layer KV over depths.
- VB reads **stored core outputs at every core layer, all positions, all versions**, with a content-addressed per-position gather as the pass input.
- VB also adds a bounded core, a question-free 2-layer coda and READY-RECODE, all in a pretrained retrofit.
- **Novelty is by combination.** Literature should confirm when online (LIT doc M2 / M7).

## 4. Training paradigm: READY-RECODE

For item i: τ_i = n_i (the number of reductions) and K_i = min(τ_i + U{0..4}, 12).

**(R1) Readiness-window answer loss.**
- L_ans = mean_i mean_{t ∈ [τ_i, K_i]} CE(coda_t, y_i). There is no answer loss before τ.
- [PROVED, PROOFS 5.1] No G8 incentive: every term is −log q_θ(fixed target).

**(R2) Reweighted per-slot recode.** E_t is the slot-aligned expression after t Feng reductions (consumed span → VOID; the reduced node's last slot → its value; HOLD E_n for t > n).
- It passes the re-render gate: 0 failures in 20k items [MEASURED, REDTEAM].
- L_rec = mean_t [ ½·mean_{changed slots} CE + ½·mean_{unchanged slots} CE ], with value slots weighted ×2.
- Unweighted, new-value slots are 3.6% of slot terms at n = 8, so about 96% of the loss would be free copying [MEASURED, REDTEAM F3a].
- **This is the "compute something new each pass" term.**

**(R3, cell N3 only) Minimal-correction teacher forcing (MCT; PROOFS §8).**
- With probability p_TF per bank entry (per-version scheduled sampling; 1 until 4k, linear to 0 at 24k), entry V_s is replaced by sg(V_s + Δ_s).
- Δ_s is the minimum-norm edit in R's row space (LN coordinates, pseudo-inverse on the logit error, margin μ) that makes R decode E_s.
- Δ_s = 0 when V_s is already right: a fixed point, so there is no exposure bias at convergence.
- The carry or gather reads the same corrected entry.
- **Why MCT and not the filler-rendered teacher of the synthesis spec.** That teacher was one step ahead (index W1), built from a different circuit (W3), on a filler token that is off-distribution (§203-8).
- **The point of R3:** every pass, at every depth, gets trained from a correct input. That removes the "repair from the input" pressure that free-running targets create once a version is wrong. With per-step accuracy 0.9, that pressure covers 27% / 52% of terms at pass 4 / 8 [DERIVED, REDTEAM F3b].
- Gates: W1, R(V^T_s) decodes E_s by construction; W3, patching along R's row space vs its null space changes the next pass.

**(R4) Curriculum.** n_max 3 → 5 → 8, gated on **answer CE** (not total loss), with the final stage running to 48k.
- The existing AnnealGate does not do this: it reads `tot` (nested170.py:1101), ends at final_len (293–297, 1026–1027), and asserts arm == "inject" (841). A new gate is needed (REDTEAM F9b).

**Total:** L = L_ans + λ·L_rec, with λ = 1 constant (no fade, so the reading stays clean).

**Not adopted, and why:**
- **Marginal-pass rewards (K vs K − j).** With gradient through both terms they are met by timed release: E[R] > 0 via under-confident early passes [PROVED in expectation, PROOFS §6]. Detached, they are a reweighted CE [PROVED]. Published sabotage also exists (Cho 2505.23060).
- **Destructive, fading or lifetime-limited memory:** the user's constraint, and it removes the early-pass gradient paths.
- **Per-pass rotations:** a no-op, a hidden lifetime, or a free pass clock (C §0.5).
- **Training-only past-slot dropout:** it trains re-reading of the input (B §1.5).
- **STAGGER timed windows (B):** proved bypass closures [PROVED, PROOFS §6, with the A-KEEP restatement on kept rows]. Kept as the GSM8K add-on (§9), because on nested the depth argument already closes the snapshot routes.

## 5. Mathematics (claims the draft relies on, after the proof pass)

1. **L1, critical path** [PROVED]. depth(V_t) = P + t·d, and the output after K passes ≤ P + K·d + c, as a layered block network with frozen positions. The per-layer TC⁰ simulation holds.
2. **L1-b** [PROVED]. Same-layer bank reads add no depth without an input feedback edge. **CODI theorem** [PROVED]: latent j has depth j(L + π) + L, and every depth-carrying path uses the feedback chain.
3. **L1-c′, interventional reads** [PROVED]. A pass computes at depth P + d·L_eff + c, where L_eff counts non-copy hops in its read chain (carry included).
   - Passes that ignore every V_{s≥1} stay at depth P + d + c, at width (K+1)T + L_dec.
   - Under CARRY an idle pass *is* a read.
4. **Chaining** [PROVED UNDER CAP-aff].
   - CAP-aff: a chain of m hops closes ≤ c₀ + (m−1)c tree levels; c₀ ≈ 2 (DIRECT, note 170-3), c ≈ 1.
   - Then exact recode at pass t needs L_eff ≥ 1 + ⌈(h_t − c₀)/c⌉, with h_t the reduced tree height.
   - [PROVED UNDER TC⁰ ≠ NC¹] L_eff is unbounded in n.
   - A's "r₁ alone forces a growing chain" is **false**: counterexample M2 has chain 2 at every n.
   - The empirical content is CAP-aff for the trained model's own sub-computations, measured by PROBE (§7).
   - The chain grows with **tree height** (mean 2.67 / 5.04 / 7.03 at n = 3 / 8 / 16), not with n.
5. **Function preservation** [PROVED; fp64 Δ ≤ 1.2e-13 on random nets]. CARRY (or PURE-ANNEAL at c = 1e4) at K = 1 with the bank masked is exactly GPT-2. Conditions: per-item readout slot, causal coda, bank masked in core and coda.
6. **No timed-release incentive under R1 / R2** [PROVED: no negative-sign likelihood].
   - **Count keying is not excluded on-distribution.** "Reduce until done" and "reduce while t ≤ n" are both zero-loss [PROVED].
   - So STALL is required (§7). "Any op alive" is the simpler circuit [CONJECTURE].
7. **What MCT guarantees** [DERIVED]: a fixed point at correct versions. Its forcing is limited to R's 20-dimensional row space, which is weaker at init.

## 6. Goodhart routes (status)

| route | status |
|---|---|
| G1 pass-1 precompute | Bounded on deep nested items by L1 + CAP-aff; open at height ≤ 2 (legal under C1). **Measured by PROBE(1) / PROBE(2).** Open on GSM8K m ≤ 2 (§9). |
| G2 recompute from V_0 | Bounded by depth. Free-running targets reward it once a version is wrong (F3b); R3 removes that. Measured by VALUE-SWAP's own-answer share. |
| G3 coda solving | 2 layers over ≤ 340 entries can do the last reduction. R pins V_n = E_n; watch R at t = n vs the coda at K′ = n − 1 under VALUE-SWAP. |
| G4 literal copying | Not a bypass (V_0 is global); the value slots carry the weight in R2. |
| G5 / G10 idle / carry-only | Idle-before-done is bounded by L1-c′. "Carry-only" is the Markov optimum, not a failure. BANK-ABLATE is descriptive only (REDTEAM F4). |
| G6 / G11 clocks | V_0 has no age term. The bank normaliser and content clocks remain, so it is tested by **STALL** and by DEPTH / REACH. |
| G7 one-sided dynamics targets | None used. |
| G8 sabotage | No incentive [PROVED]. |
| G9 width-for-depth | Closed in practice at n = 8 (height 5.04 vs DIRECT's ~2); weak at n 3–4. |
| CARRY-NO-STEP (the §205 s3 pattern in VB) | **Primary risk.** Reweighted R2 plus R3 address it. The new label measures it. |
| TF format reliance | MCT has no separate dialect; TF-ONLY / CODE-DRIFT are labelled. |

## 7. Decisive readings (comparator-free, inside one model)

All readings run in local fp32 at the original eval_bs, on the training card's class.

**PROBE(t)** (REDTEAM F1, the MEAN-ITER analogue).
- Freeze the trained model. For each t ∈ {0, 1, 2, 3, 4, 6, 8, 12}, train a fresh 2-block coda (initialised from the trained coda, LM head tied and frozen) on V_t + bank → answer.
- Training: 20k training items, n 3..8, on the fly (no cached features), about 2k steps each; local, $0.
- **PROBE-GAIN** = ACC(K′ = n + 2, pooled n 3..8) − max(PROBE(0), PROBE(1)) ≥ **0.50**.
- **PROBE-ONSET:** onset_i = the smallest t with PROBE(t′) correct for every t′ ≥ t in the set, up to n + 4. SERIAL-ONSET requires:
  - coverage ≥ 0.8 at every n 3..8 (never-reached = failure);
  - an OLS slope of median onset on tree height ≥ 0.8 (on n ≥ 0.6, reported beside);
  - med(8) − med(3) ≥ 3.
- **ONESHOT-PRECOMPUTE** = PROBE(2) ≥ 0.8 at n = 8. This falsifies the depth premise, and it can now fire.

**VALUE-SWAP** (REDTEAM D3; replaces SKIP-DELAY, which is exactly K′ − 1 [PROVED], note 170-4).
- Donor: the recipient with only the two operands of the handle reduced at pass t ∈ {2..n−1} changed (the value changes, no ×0 ancestor). Then y_hyb = y_donor.
- Splice arms: (i) slot b only; (ii) the whole V_t; (iii, CARRY only) the bank entry vs the carry, separately.
- Clean-ccc pairs (recipient, donor and hybrid each solved clean), n 4–8 pooled.
- **SERIAL-STEP:** follow(ii) ≥ 0.6 and own ≤ 0.3. Chance about 1/11. §203-4 references: failed class 0.003–0.18, converged 0.80–0.97.

**STALL** (PROOFS §5). At t = ⌈n/2⌉, append V_t := V_{t−1}.
- "Reduce until done" recovers at K′ = n + 1; a count-keyed rule does not.
- **NO-CLOCK:** recovery ≥ 0.9 × intact.

**DEEP:** accuracy ≥ 0.90 at every n 3..8 at K′ = n + 2 (inside the trained window; REDTEAM F7).

**VB-C1** = DEEP ∧ PROBE-GAIN ∧ SERIAL-ONSET ∧ SERIAL-STEP ∧ NO-CLOCK.

**Descriptive (never decisive):**
- the trained-coda ITER-GAIN and onset (trained artefacts, REDTEAM F1);
- STEP-CAP(t, a): the one-step accuracy of the newly reduced slot when V_{t−1} decodes E_{t−1} exactly, by pass and alive count;
- STEP-STATIONARY: slope in t ≤ 0.01 per pass at fixed a;
- CARRY-LIKENESS: final gather mass on (t−1, j). A value near 1 is the Markov optimum;
- BANK-ABLATE;
- DEPTH (n 9–12 at K′ = n + 2) and n 13–16; STABLE (K′ 24);
- entry steps (the pinned basin-entry flag); R accuracy by t.

**Labels:**
- VB-C1;
- ONESHOT-PRECOMPUTE;
- NO-ENTRY (the answer-CE bin is never ≤ 0.5);
- LOSES-ONE-PASS (not DEEP and PROBE-GAIN < 0.2);
- **CARRY-NO-STEP** (copy-slot R accuracy ≥ 0.99 and value STEP-CAP at n 6–8 ≤ 0.8: VB's MET-NOT-ENTERED);
- COUNT-KEYED (STALL fails, or DEEP with REACH(16) ≤ 8 and DEPTH < 0.5);
- TF-ONLY / CODE-DRIFT (N3);
- DILUTION (accuracy falls past K while newest-version mass decays and mean γ is about 0).

## 8. OC block, censoring, cost

**OC (one seed per cell; REDTEAM §F, PROOFS):**
- **No effect:**
  - P(PROBE-GAIN | pass-1 precompute of n = 8) ≈ 0, since PROBE(1) would be high;
  - P(SERIAL-STEP | the recompute-from-V_0 null) ≈ 0.05 (failed-class swap shares 0.003–0.18);
  - P(VB-C1 | no serial use) ≤ 0.05 [ASSUMED from these].
- **Effect:**
  - P(VB-C1 | real serial use) ≈ 0.6–0.7 (losses come from distributed value storage weakening arm (i), and from swap disruption);
  - P(the cell learns serial use at one seed) ≈ 0.45–0.6 (the lottery, §199);
  - so P(VB-C1) per cell ≈ 0.3–0.4.
- **A positive is strong evidence; a null is weak.** A rate claim needs ≥ 3 seeds and the user's OK.

**Censoring (rule b):**
- A run dying before 48k fails every reading; NO-ENTRY is a failure.
- A never-reached onset is a coverage failure.
- A PROBE that does not train (a probe-CE floor not met) fails PROBE-GAIN.
- VALUE-SWAP with < 100 clean-ccc pairs is UNREADABLE, which counts as not met.
- Event-only values are reported beside each reading.

**Cost (nested Stage 1, 48k × bs 64, RTX 3090).**
- §198 cells measured 0.63–0.86 s/step [MEASURED].
- VB central is 0.35–0.8 s/step, i.e. **$0.9–2.4 per cell**; N3 adds 30–60%.
- **Family central about $5, range $3–8.5; cap $9.**
- **G-TIME** runs on a forced final-stage batch (n 8, K 12, T 34), because step 200 falls in curriculum stage 1 and under-projects by 1.5–2× (REDTEAM F10). Per-cell guard $3.

## 9. Experiments

### Stage 0 (local, $0, short and memory-checked; no registration needed for gates)
1. **Implementation:** a new boxes/vbank.py (VBankN), with no edits to latent/*.py or to the plain paths. Then a plain-path smoke (INJECT construct + 1 step) and py_compile. Audit's diff review covers decorators (by ast), return shapes, and the bank cache's gradient.
2. **G1:** the fp64 exact identity at the masked bank (≤ 1e-9).
   **G1-cal:** the per-head β calibration (m₉₉ ≤ 0.02), plus the ratio test Δz(β)/Δz(β − 5) ∈ [0.5, 2]·e⁵.
3. **G1b drift smoke** (REDTEAM F5). At init, P(number) at " =" falls 0.53 → 0.008 by pass 8, so loops leave the pretrained manifold. Run 2k steps locally and report P(num) by pass, cos(V_t, V_{t−1}), and answer CE at K′ = n vs K′ = 1 on n = 1. Pass if the drift falls below its init value and the K = n CE is falling.
   - A re-entry LayerNorm variant is held back. It would need its own derivation.
4. **Path-specific gradflow** (REDTEAM F9a): ∂L/∂(bank K/V) ≠ 0, and ∂L_ans/∂V_1 ≠ 0 with the carry / gather stop-gradiented. Plain per-group gradflow is blind to a dead bank.
5. **Checks:** E_t re-render (100k items); MCT gates W1 / W3 (N3); mem_check at the largest shape (n 16, K′ 24) with a preallocated bank; masked-K vs unmasked equality in fp64.
6. **Optional, descriptive:** B's G-ALIGN, WINDOW-ACC on the 16 kept nested models. It informs STAGGER for GSM8K.

### Stage 1 (cloud RTX 3090, seed 1, 48k, n 1..8 train, §170 test set)

| cell | design | question |
|---|---|---|
| **N2 (main)** | PURE-ANNEAL + R1 + reweighted R2 + R4 | Does a pass-global attention-only memory with a bounded core learn serial step use? |
| N1 (control) | CARRY + the same | Does the attention-only end state cost anything against the carry? (CARRY-LIKENESS and the readings compared) |
| N3 | N2 + MCT teacher forcing (R3) | Does training the step map from correct inputs remove CARRY-NO-STEP (the §205 s3 pattern)? |

- If G-TIME projects the family above $9, drop N1 first.
- **Seed 2** for a VB-C1 cell is a separate decision.

### Stage 2 (GSM8K; only after VB-C1 in some cell; a separate registration)
- **Setup:** VB-12 (and VB-8, one-pass depth 8) on GSM8K-Aug, 24k × bs 64.
- **Fixes from REDTEAM F8:**
  - A **value-first recode target**, "<<v = a op b>>" with the result first, or §204 F-P's teacherless value. The "<<a op b = v>>" form lets 8 decoder-own layers compute v from operands in the context.
  - PROBE-GAIN / MEAN-ITER as the in-model control (K′1 is not trained for m ≥ 2).
  - Items with m > K_max get the answer at K_max.
  - A STAGGER add-on (B) if PASS1-PRECOMPUTE recurs. Its snapshots **replace** the input entry each pass rather than being banked, or Prop D's re-coding implication fails (PROOFS §6).
- **Expectation:** PASS1-PRECOMPUTE on m ≤ 2 is expected (pass 1 is the full network).
- **Cost:** ≤ $1.5 per cell.

## 10. Predictions [CONJECTURE, stated before any run]

| cell | P(entry) | P(DEEP) | P(VB-C1) |
|---|---|---|---|
| N2 | 0.6 | 0.45 | 0.3 |
| N1 | 0.65 | 0.5 | 0.35 |
| N3 | 0.65 | 0.55 | 0.4 |

- N2 CARRY-LIKENESS ≥ 0.8 (the Markov task): P 0.7.
- If DEEP: DEPTH central 0.65 (INJECT 0.49–0.69). BEYOND-INJECT (DEPTH ≥ 0.75): P 0.25.
- CARRY-NO-STEP in N2 / N1: P 0.3; in N3: P 0.15.
- ONESHOT-PRECOMPUTE: P 0.05.

## 11. Top risks and falsifiers

1. **The loop is not learned at this budget (NO-ENTRY / LOSES-ONE-PASS).** Precedents: §198(c) READ-SLOTS, which entered at 33k and was not DEEP; and init drift (REDTEAM F5). *Falsifier at this budget:* N1–N3 all fail this way.
2. **CARRY-NO-STEP:** perfect copying with a deep-step deficit, the §205 s3 pattern. *Falsifier for R2 / R3 as the capability lever:* N3 reads CARRY-NO-STEP.
3. **A count-keyed schedule (COUNT-KEYED / STALL failure).** It would mean the passes run a clock, not a computation.

**Premise falsifier:** ONESHOT-PRECOMPUTE (PROBE(2) ≥ 0.8 at n = 8).

**Constraint check:** N2 VB-C1 with CARRY-LIKENESS about 1 meets the constraint in form, since the attention memory is the only channel. A model that needs older versions would show it on GSM8K (non-Markov).

## 12. Before registration (what is still needed)

1. The user's OK (via Audit) for Stage 1 at cap $9.
2. Stage 0 gates (all $0, local).
3. Literature's novelty check when it is online: Feedback Transformer, RecurTrace, InfiLoop, Think-at-Hard against VB.
4. A scorer, theory/score_s2xx.py, fixture-checked on synthetic panels: VB-C1, ONESHOT, CARRY-NO-STEP, COUNT-KEYED, NO-ENTRY, a dead run, pending.
5. The registration text in docs/THEORY.md, with the OC block and censoring as above.
