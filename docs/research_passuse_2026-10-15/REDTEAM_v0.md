# Red team v0: VERSION-BANK + READY-RECODE (SYNTH_spec_v0)

*Opus red-team subagent for Theory, 2026-10-15 (after 22:35 EDT). Repo read-only. CPU checks (≤ 3 threads) are in `scratchpad/redteam/`:*
- *`slot_targets.py`: A's slot-aligned E_t, the re-render gate, and per-pass slot shares (20k items);*
- *`loop_init.py`: pretrained GPT-2 at init, VB-12 CARRY / PURE loops of blocks 6–9 with a β = −5 bank, 36 nested items, n 3–8.*

*Tags: [PROVED], [MEASURED] (my scripts or repo logs), [VERIFIED file:line], [DERIVED], [CONJECTURE].*

## Top 10 findings (ranked by likelihood × damage)

### 1. BLOCKER: three of VB-C1's four conjuncts are produced by the objective, and the premise falsifier cannot fire

**ITER-GAIN.** Under (R1), the answer loss applies only at t ∈ [n, K_i].
- K′ = 0 is never trained.
- K′ = 1 is trained only on n = 1 items. [VERIFIED: A §2 "never applied at t < τ_i"]
- So for n ≥ 3, ACC(K′0) and ACC(K′1) are readouts of an untrained coda on states that the recode target (R2) keeps visibly "not done" (several alive slots).
- They are low whatever V_1 contains.
- Hence ITER-GAIN ≥ 0.5 follows from ACC(12) ≳ 0.65 and adds nothing to DEEP.

**SERIAL-PROFILE.** The coda is only ever trained on done-looking states (t ≥ n), and R2 pins V_t's linear content to E_t, which looks done exactly at t = n. So onset ≈ n with slope ≈ 1 is trained in.
- A states this itself in its OC block: "P(VB-C1 | serial use) ≈ 0.8 *since the recode targets pin onset ≈ n*".
- A model that precomputes shallow items in pass 1 and releases them at the clock (the bank size: bank mass rises monotonically, 1.7% → 7.3% from t = 1 to 8 at init [MEASURED]) reads exactly like a serial one.

**ONESHOT-PRECOMPUTE** (onset ≤ 2 at n = 8) needs the coda to answer at t ≤ 2 on n = 8 items, which it is never trained to do. So the stated falsifier of the depth premise has probability ≈ 0 whatever the truth.

**Net effect:**
- VB-C1 ≈ DEEP + trained readings.
- The C1 inference then rests entirely on the depth calibration, which comes from an external comparator (DIRECT, note 170-3). So the readings are not comparator-free as labelled.
- The OC line "P(VB-C1 | no serial use) < 0.01" is unsupported for the relevant null (DEEP reached with a trained-in onset).

**Fix (pre-register before any run):**
- **PROBE(t):** freeze the trained model. Train a fresh 2-block coda (initialised from the trained coda, LM head tied and frozen) on V_t + bank → answer, for each t ∈ {0, 1, 2, …, 12}, on 20k training items with n 3–8. Evaluate on test in fp32. This is the MEAN-ITER analogue of §208. Cost: local, $0.
- **PROBE-GAIN** = ACC(K′12) − max(PROBE(0), PROBE(1)) ≥ 0.5 pooled. This is the in-model, trained K′1 control that LIT summary item 9 says is required ("one pass is insufficient must be shown empirically").
- **PROBE-ONSET:** onset defined on PROBE(t), not on the trained coda.
- **ONESHOT-PRECOMPUTE** becomes PROBE(2) ≥ 0.8 at n = 8, so it can actually fire.

### 2. BLOCKER: SKIP-DELAY is exactly K′ − 1; it is note 170-4's degenerate skip, rediscovered

**Proof [PROVED].** VB has no pass index. The pass map is autonomous on the bank: 𝔅_t = 𝔅_{t−1} ∪ {Φ(𝔅_{t−1})}.
- CARRY: U = N(newest).
- PURE: U = q + wpe, the same every pass.
- Age bias: |𝔅| − 1 − s.

"Skip pass t" sets 𝔅_t := 𝔅_{t−1}, so the next pass computes Φ(𝔅_{t−1}), which is the intact pass t. By induction, skipped-at-K′ ≡ intact-at-(K′ − 1), bitwise.
- If the implementation uses a global pass counter for the age term instead, the only difference is γ·1 per entry (γ initialised to 0).
- This is the same degeneracy that THEORY note 170-4 recorded for index-free identical-input arms [VERIFIED THEORY.md:10756–10766]. C's design notes it for LEDGER (C §6: "the skip test is vacuous here").

**Consequences:**
- The recovery clause (K′ = n + 1 ≥ 0.9 × intact) reduces to ACC(n)/ACC(n) = 1, so it is automatically met.
- The delay clause reduces to ACC(n − 1) ≤ 0.3·ACC(n), which is one point on the trained onset curve (finding 1).
- In PURE, "skip" has no other meaning.

**Fix:**
- Drop SKIP-DELAY. Report "skip ≡ K′ − 1" as an identity gate only.
- The step-specific test becomes **VALUE-SWAP** (§D3): single-handle donors spliced at the reduced slot, untrained and chance-floored.

### 3. MAJOR (Theory's added question): R2 gives each pass a reason to compute, but a seed can still sit at "carries but can't reduce"

**What VB adds over the recipe that failed in s185_rms_s3** (which already had §175's one-reduction-per-pass aux):
- **The target is decoder-proof.** R is linear, so it cannot compute. The failed pass-token seeds met their aux with the shared 12-layer trunk doing the reductions itself (199-M3 / M3a: STRUCT-RIGHT errors 0.99–1.00, Cor. 1.2).
- **Presence → use for chained values [DERIVED].** A value written at pass s is needed for its parent's target at a later pass s′. Recomputing it from V_0 is bounded by depth, so for subtrees taller than about 2 the written value must be read. This is stronger than §203's SVR probe, which had no chain and read G-use 0/9.

**Why a seed can still plateau at "copies perfectly, reduces deep items badly" (the §205 s3 pattern under IIT):**
- **(a) Dilution [MEASURED].** Per pass, the share of slot terms that change is 0.326 / 0.220 / 0.147 at n = 3 / 5 / 8; new values are only 0.085 / 0.055 / **0.036**. Under CARRY copying is nearly free (identity residual), so about 96% of L_rec at n = 8 is carry, and the compute signal is about 1/26 of a slot-mean CE. The newly-VOID part (0.11) is a structural plan that a bank-size clock can meet in parallel (§204's PLAN-PARALLEL lesson).
- **(b) Free-running targets reward repair-from-input [DERIVED].** Once V_{t−1} is wrong, the true E_t is unreachable by one reduction. The gradient at pass t then teaches correcting values from V_0, which is the recompute (Huginn / failed-seed) route of §203-8.
  - With conditional per-step accuracy p, the share of pass-t terms with a wrong input is 1 − p^{t−1}: 0.27 / 0.52 at t = 4 / 8 for p = 0.9.
  - So the pressure toward the bad basin is largest early in training and on deep items.
- **(c) Dialect drift [MEASURED at init].** R constrains about 20 of 768 directions; the rest of V_t can drift with t.
  - At init under CARRY: cos(V_t, V_{t−1}) is 0.90–0.94 per pass, and the between-item variance of the " =" slot grows ×2.9 by t = 2 and ×14 by t = 8.
  - A step map trained at pass 1 on V_0 (every expression length is seen there with correct input) then fails at pass t on an equally easy but drifted input.
  - This is s3's signature: conditional per-step success 0.97 → 0.72 over t 3–8, even though late steps are on shorter expressions (§199).
- **(d) Per-pass capacity.** 4 layers per reduction is plausible: MEM-HIGH re-ran only 3 blocks per pass and was DEEP [VERIFIED THEORY.md:14926–14930]. But early on, the right context needed for +/− evaluability enters only through the 1.7%-mass bank (causal current attention under CARRY).

**Does TF (R3) change this?**
- Partly. It removes (b) on TF passes: every target becomes a consistent one-step map from a correct version.
- It does **not** fix (c). V^T_s = core(prelude(render_VOID(e_s))) is always a pass-1 code of re-rendered text (a stationary dialect), so TF trains the step map only on the pass-1 dialect.
- At p_TF → 0 the student's own drifted codes return. That is the TF-ONLY route (finding 6).
- R3 is the cell most aligned with the §205 capability diagnosis, but only if a dialect anchor is added (below).

**Fixes:**
- **L_rec reweighting:** mean over changed slots + mean over unchanged slots, weighted 1 : 1. Value slots get an extra ×2. This is the slot analogue of §175's handle vs copy split.
- **R3 + DIALECT-ANCHOR:** L_cons = mean over real slots of 1 − cos(V_t, sg V^T_t). This is C's ℓ^cos; R prevents collapse. An EMA teacher is the fallback.
- **New readings:**
  - **STEP-CAP(t, a):** one-step accuracy of the newly reduced slot when V_{t−1} decodes E_{t−1} exactly, by pass t and alive-slot count a.
  - **STEP-STATIONARY:** slope in t at fixed a ≤ 0.01 per pass.
- **New label CARRY-NO-STEP:** copy-slot accuracy ≥ 0.99 and value STEP-CAP at n 6–8 ≤ 0.8. This is the VB form of §205's MET-NOT-ENTERED.

### 4. MAJOR: the user's constraint. N1 (the main cell) is an RNN carry, CARRY-ONLY is mis-specified, and PURE starts at a degenerate fixed point

**CARRY.** U_t⁰ = N(V_{t−1}) carries the full residual. That is the RNN channel the user excluded (22:07).

**BANK-ABLATE / CARRY-ONLY** masks versions 1..t−2 but keeps V_0 and V_{t−1}.
- The task is Markov in E_t, and V_{t−1} contains every alive literal. So the optimal serial solution shows about 0 drop.
- CARRY-ONLY therefore fires for the correct solution, under CARRY and PURE alike.
- It tests non-Markov use, which nested never needs. It is not a test of the constraint.

**PURE at init [MEASURED]:**
- cos(V_t, V_{t−1}) = **1.000** for t = 2..8: every pass writes the same version. This is 2604.11791's degenerate fixed point, present from step 0, and pass-blind like §199's γ = 0 phase.
- The current version's content-free keys take 88–95% of attention (bank mass 5.4% → 12.2%).
- PURE is the closest analogue of §198(c) READ-SLOTS (fresh queries each pass, input readable every pass), which entered at 33k and was not DEEP. A's P(DEEP) ≈ 0.3 looks high.

**Proposal: PURE-ANNEAL** (attention-only, function-preserving; details in §E).
- A residual-space gather U_t⁰[j] = Σ_{s<t, i} softmax(ℓ_{j,(s,i)} − c(step)·1[(s,i) ≠ (t−1, j)]) · N(V_s[i]):
  - content-free position queries;
  - no value / output projection, so the hard phase is an exact copy;
  - c a **scheduled buffer**, 1e4 → 0 over 1k–8k steps, not a learned −∞ init. Masked logits get e^{−c} gradient, which is Shapiro's zero-init self-lock.
- At c = 1e4 it is bitwise CARRY, so G1 holds. At c = 0 it is an attention read over every version.
- Make it the main cell, with CARRY as the control.
- Report CARRY-LIKENESS (final mass on (t−1, j)) descriptively. A value near 1 is the Markov optimum, not a failure.

### 5. MAJOR: function preservation covers only K′ = 1; the losses read looped, off-distribution states

The answer loss is read at K ≥ n (≥ 3 for most items) and R2 at t ≥ 2.

**At init [MEASURED, CARRY, β = −5]:**
- The coda's P(number token) at " =" is 0.534 / 0.466 / 0.273 / 0.058 / 0.008 at t = 1 / 2 / 3 / 4 / 8.
- Entropy collapses from 4.5 to 2.4 nats onto non-number tokens.
- Item differences expand (×14).
- So blocks 6–9 fed layer-10 statistics leave the pretrained manifold within 2–3 passes, even with the per-slot RMS reset.

The retrofits keep a re-entry adapter for this reason: McLeish's concat-adapter, Shapiro's identity-biased bridge, Huginn's sandwich norm. An expanding loop at init plus clip 1.0 is §199's basin-selection setting (the init regime picks the basin).

A's "fallback is DIRECT-level, not chance" is also illusory: K′1 is not trained for n ≥ 2 (finding 1).

**Fix:**
- **$0 gate G1b:** a 2k-step local smoke reporting P(num) by pass, cos(V_t, V_{t−1}), and answer CE at K′ = n against K′ = 1 on n = 1 items. Pass if the per-pass drift at 2k is below the init value and the answer CE at K = n is falling.
- Variant to test in the smoke, not in the first cell: a re-entry LayerNorm on N(V_{t−1}), initialised to block 5's ln_2 statistics. This needs its own derivation per the math rule.

### 6. MAJOR: R3's teacher is a different dialect, and the consistency term was dropped

**The two dialects differ.**
- V^T_s = core(prelude(render_VOID(e_s))) is the depth-10 code of *text* with a filler token at VOIDs.
- The student's V_s is the depth-(6 + 4s) code of the original problem.
- §203-8 showed filler masking is off-distribution for trained models (MASK-DISRUPTS / filler-OOD).

**The likely failure.** The core learns "filler = consumed" and the teacher's stationary dialect. When p_TF → 0, student VOIDs are not filler, the leftmost-ready decisions break, and the run reads TF-ONLY / CODE-DRIFT. C kept ℓ^cos exactly to make codes substitutable; SYNTH adopted C's teacher forcing without it.

**Ambiguities to settle:**
- Does the CARRY input also use V^T_{t−1}?
- Is TF per item or per version?
- The sg teacher uses the current weights, so it is a moving target.

**Fix:**
- DIALECT-ANCHOR (finding 3).
- Per-version scheduled sampling (each bank entry independently teacher with probability p_TF).
- The CARRY input follows the newest bank entry.
- Labels TF-ONLY and CODE-DRIFT as in C §6, plus STEP-CAP in both modes.

### 7. MAJOR: ONSET coverage depends on stability far outside training, and the age bias at γ = 0 invites dilution

onset_i needs correctness at every K′ ∈ [t, 16].
- n = 3 items train only to K ≤ 7 (K_i = n + U{0..4}), so 9 or more of the 13 post-completion passes up to 16 are never trained for them.
- §201 / §193 show post-completion drift is the norm unless WINDOW / HOLD covers it.
- A coverage failure at small n then sinks SERIAL-PROFILE for stability reasons, not seriality.

Separately:
- Bank mass grows with t (1.7% → 7.3% by t = 8 at init [MEASURED]).
- With γ initialised to 0, old versions dilute the read past K_max (C §4.2: newest mass ∝ 1/t without an age bias). This predicts STABLE failures at K′ 24.

**Fix:**
- Onset over K′ ∈ [t, n + 4] (the trained window); STABLE and K′ 24 reported separately.
- γ initialised to ALiBi slopes on 6 heads and 0 on the other 6 (C's choice).
- DILUTION label.

### 8. MAJOR: GSM8K transfer recreates §204 / §208, with an inflated ITER-GAIN

**(a) The text recode "<<e_t>>" is D1-leaky (§204).**
- The decoder tokens get blocks 0–5 among themselves, then 2 coda layers: 8 layers of their own over a teacher-forced "<<48/2=".
- So the result token can be computed from in-context operands, and V_t need not hold results.
- Nested N3 is safe: the reduced value replaces the consumed operands in e_{t+1}.

**(b) G1 stays open.** VB-12's pass 1 is the full 12-layer network, and §208 showed one pass learns everything learnable at this budget (onset 1 in every stratum). PASS1-PRECOMPUTE on m ≤ 2 should be expected, not hoped against.

**(c) ITER-GAIN is inflated.** K′1 is trained only on m = 1 items, so ITER-GAIN reads high for any m ≥ 2 precompute that is released at τ = m. That gives a **false VB-SERIAL** route.

**(d) STAGGER does not fit a banked V_0.**
- If the per-pass snapshots V_0^(p) are banked, a literal's true value stays readable after its window.
- That breaks Prop D's data-processing step, which needs x^(p > hi) to be independent of u_ℓ.
- Snapshots must replace the input entry per pass, not be appended.

**(e) Empty windows.** The recurrence curriculum K_max 2 → 5 gives empty answer windows for m > K_max, so those items are silently dropped from L_ans.

**Fix:**
- A result-first or value-only target ("<<24=48/2>>" or §204 F-P's teacherless value).
- PROBE-GAIN or MEAN-ITER as the GSM8K comparator.
- Non-banked snapshots.
- An explicit rule for m > K_max.

### 9. MINOR → MAJOR: implementation traps that pass the stated gates

**(a) The gradflow gate is blind to a dead bank [DERIVED].**
- β and γ receive gradient through the softmax normaliser even if the bank K/V are detached: ∂o/∂β = Σ_b p_b(v_b − o) ≠ 0.
- The core parameters receive gradient through the carry and the current-version keys.
- So a no_grad or detached bank cache passes "nonzero gradient per parameter group".
- Required instead:
  - ∂L/∂(bank K/V tensors) ≠ 0;
  - a bank-only test with the carry stop-gradiented, giving ∂L_ans/∂V_1 ≠ 0.

**(b) Reusing AnnealGate is wrong on three counts.**
- It is fed the total loss `tot` (answer + λ·aux), not the answer CE [VERIFIED nested170.py:1101].
- Its final stage runs exactly final_len and the run stops at s_full + final_len − 1 [VERIFIED nested170.py:293–297, 311–313, 1026–1027], while the LR cosine is over the cap [VERIFIED nested170.py:1039]. An early gate means the run ends before 48k at nonzero LR. A's "the final stage runs to 48k" is not what the code does.
- It asserts arm == "inject" [VERIFIED nested170.py:841–842].

**(c) Code pointers that need new code.**
- `deltamem._block` is causal-only SDPA with no mask argument [VERIFIED deltamem.py:256]. A's "`_block` with this mask" pointer therefore needs a new block function.
- `rs_run` builds a boolean mask [VERIFIED nested170.py:434–435, 446]; β needs a float mask.
- `kmatch_aux_loss` masks per-item K by a state-level torch.where [VERIFIED nested170.py:387]. VB must mask bank entries s ≥ t per (item, t) row in the coda instead.

**(d) Decorated helpers.** `aux_greedy` and `dec_reads` are `@torch.no_grad` [VERIFIED nested170.py:274, 581–582]. None may be reused on a training path, and only the R3 teacher may be sg. Check by ast, not by eye (lesson: insert-above-decorated-def).

**(e) Bank memory.** torch.cat of a growing bank saves O(K²) tensors: about 2.1 GB of K/V at K 12, T 34, bs 64 in bf16. Fine on 24 GB, but the 8 GB gates need micro-batching. Preallocate the bank and mask it instead.

### 10. MINOR: cost is plausible, but G-TIME at step 200 under-projects

**Repo logs [MEASURED from boxes/runs_fsq/s198_*_s1/log.jsonl]** (block count is a weak predictor):

| cell | s/step | note |
|---|---|---|
| cached INJECT | 0.855 | the comparator |
| MEM-HIGH | 0.651 | re-runs half the blocks |
| READ-SLOTS | 0.631 | |

There is a floor from launch overhead.

**VB:**
- It drops the delta-head token loop and the 12-block aux decoder, but runs 12 passes plus a per-(item, t) coda.
- Central 0.35–0.8 s/step, i.e. **$0.9–2.4 per 48k cell** at $0.18–0.23/h, against A's $1.3–1.8. Host speed varies about 2×.
- N3 / R3 add the aux and the teacher (+30–60%).

**The G-TIME problem.** Step 200 falls in curriculum stage 1 (n ≤ 3: K_max 7, T ≈ 11), so it under-projects the final stage (K 12, T ≈ 26–34) by about 1.5–2×.

**Fix:** G-TIME on a forced final-stage batch; a per-cell guard at $2.6.

---

## A. Comparison with §198(c) READ-SLOTS

**The same:**
- The input is encoded once and readable at every pass (V_0 / input KV). That is the Huginn recompute route that failed seeds take (203-8: INPUT-FOLLOW 0.90–0.99).
- Per-pass queries carry no item content in PURE (learned slots vs q + wpe).
- The cross-pass channel starts weak (fade-in vs β = −5).
- (c) ended contractive (delta_cos 0.841) [VERIFIED THEORY.md:15003–15016]. PURE at init is already at cos 1.000.

**Different:**
- The write is T slots wide, not 8.
- Memory is attention over outputs, not DeltaMem.
- The per-pass target is linear and cannot compute; (c) used the pass-token aux on the shared trunk.
- Per-pass depth is 4, not 12.
- CARRY additionally keeps the full residual.

**Reading:** CARRY's precedent is INJECT + aux (entered). PURE's precedent is (c) (failed).

## B. Bypass routes (status after this review)

| route | verdict |
|---|---|
| G1 pass-1 precompute | Bounded on deep nested items by depth. Open at n ≤ 4 (heights 2.7–3.4) and on GSM8K. **Not measurable with the trained coda** (F1). PROBE(1) measures it. |
| G2 recompute from V_0 | Bounded by depth, but **rewarded** by free-running targets once a version is wrong (F3b). Detected by VALUE-SWAP's own-answer share. |
| G3 coda-side solving | The coda has 2 layers over ≤ 340 entries. One final reduction ("a o b", 3 alive slots) is within reach of 2 layers (gather, then a 484-entry MLP lookup). The core lagging by one pass is prevented only by R pinning V_n = E_n. Watch R accuracy at t = n against the coda's answer at K′ = n − 1 under VALUE-SWAP. |
| G4 copying | Not a bypass. |
| G5 / G10 idle / carry-only | G10 as tested is mis-specified (F4). |
| G6 / G11 clock | A clock exists at init: bank mass is monotone in t [MEASURED]. It can meet the VOID-pattern part (0.11 of slot terms at n = 8) as a parallel plan. Values still need chaining. Test: COUNT-KEYED via DEPTH n 9–16, as A says. |
| G8 sabotage | No incentive (no term compares K with K − j). Agreed. |
| G9 width-for-depth | Closed in practice. 12 depth-10 feature sets plus 2 layers form a depth-12, ×12-width circuit, and DIRECT (depth 12) closes about 2 levels, while n = 8 needs a mean height of 5.04. The argument is not vacuous at n = 8, but is weak at n = 3–4. |
| Slot-recode "precompute then shift" | Possible only for values within one-pass depth (height ≤ about 2); legal under C1. It makes the onset slope at n 3–4 uninformative (F1). |
| Positional timed release (Prop M) | No reward, so no incentive, but it is observationally identical to serial use under the trained readings (F1). PROBE-ONSET separates them. |
| PURE degenerate fixed point | Present at init [MEASURED] (F4). |
| R3 format reliance | Likely (F6). |

## C. Trainability notes beyond F3–F5

- **β = −5 is not a vanishing gradient.** The init bank mass is 1.7–7.3% (CARRY) and 5.4–12.2% (PURE), measured on real logits, sinks included. A's 1.3–10% stands. The real problems are a Markov task that never needs the bank (CARRY) and starvation behind content-free keys (PURE).
- **The readiness window** puts the n = 8 answer gradient through 8+ looped passes. R2 supplies dense per-pass gradient, but only about 4% of it is value signal (F3a).
- **The curriculum gate's threshold** (answer CE < 0.3, two bins) must read the answer CE, not `tot` (F9b). Under R1 the stage-1 answer CE covers t ∈ [n, n + 4] only.

## D. Measurement fixes in detail

**D1. PROBE(t).** Defined in F1; frozen model, fp32, fixed eval_bs, card recorded.
- Readings: PROBE-GAIN ≥ 0.5; PROBE-ONSET slope on n ≥ 0.6 (or on tree height ≥ 0.8, reported beside); ONESHOT-PRECOMPUTE = PROBE(2) ≥ 0.8 at n = 8.
- **Over-reading:** PROBE(1) is a 2-block solver on depth-10 / depth-6 features, i.e. a trained 12-layer path. That is the right comparator.
- **Under-reading:** information that 2 layers cannot decode is not usable by the model's own readout either.

**D2. Efficiency.** With no pass index, every K′ readout is a prefix of one 24-pass rollout, so the ONSET sweep costs one rollout per item.

**D3. VALUE-SWAP** (replaces SKIP-DELAY and SERIAL-SLOT as specified):
- **Donor:** the recipient with only the two operands of the handle reduced exactly at pass t ∈ {2..n−1} changed, value changed, no ×0 ancestor.
- **The hybrid label is well defined only in this form.** Then y_hyb = y_donor.
  - With B's two-literal donors (a late leaf ℓ_L also changed), the label is **ill-defined in VB**. Every version re-states all alive literals, so the spliced V_t carries the donor's late leaf too. A CARRY model would give y_donor, and a PURE model's answer depends on which version it reads.
  - If the handle was reduced before t, the recipient's V_s..V_{t−1} hold the recipient's value, and two values conflict in the bank.
- **Arms:**
  - (i) slot b only;
  - (ii) the whole V_t;
  - (iii, CARRY only) bank entry swapped with the recipient's carry kept, and the reverse. This separates the carry from the bank channel.
- **Pairs:** clean-ccc only (recipient, donor and hybrid each solved clean).
- **SERIAL-STEP:** follow(ii) ≥ 0.6 and own ≤ 0.3, on n 4–8 pooled.
- **Chance and reference:** chance ≈ 1/11. Failed seeds on the analogous §203-4 swap read 0.003–0.18; converged seeds 0.80–0.97.

**D4. STEP-CAP / STEP-STATIONARY / CARRY-NO-STEP** (F3).

**D5. BANK-ABLATE:** descriptive only, never a constraint test (F4).

**D6. bf16:** all readings in local fp32 at the original eval_bs on the training card's class. The per-item masked-K training path must equal the unmasked path at equal K (an fp64 gate, as 170-series exactness).

## E. Satisfying the user's constraint while keeping function preservation

PURE-ANNEAL (F4), stated for the derivation the math rule requires:

**Read.**
- ℓ_{j,(s,i)} = (W_q p_j)·(W_k LN(V_s[i]))/√d + γ(t − 1 − s).
- p_j = wpe[j]: a content-free query, so no carried content enters through the query.
- Values are N(V_s[i]) themselves.

**Exactness [DERIVED].** For c ≥ c₀, the gather deviates from V_{t−1}[j] by at most (N − 1)e^{−c}·max‖N(V)‖.
- At c = 1e4 this underflows to 0 in fp32 and fp64, so G1 holds bitwise.
- The hard phase is gate-checked bitwise against CARRY.

**Boundedness [PROVED].** At c = 0, U_t⁰[j] is a convex combination of stored per-slot-RMS-reset vectors, so ‖U‖ ≤ max_{s,i} ‖N(V_s[i])‖, independent of t. There is no additive growth (the §192 lesson).

**Schedule.** c is a buffer (not learnable), 1e4 to 1k steps, linear to 0 by 8k. A learned −∞ init would self-lock.

**End state.** The only cross-pass channels are attention reads over the append-only bank: this gather plus every core layer's bank attention. No state is privileged architecturally.

**Prediction [CONJECTURE].** CARRY-LIKENESS ≥ 0.8 on nested, because the task is Markov. That meets the constraint in form, and in function wherever older versions are needed.

**Cells.** Main N2′ = PURE-ANNEAL + reweighted slot recode. Control N1 = CARRY. Raw PURE is dropped: it is degenerate at init, and the anneal subsumes it.

## F. Revised OC (nested, one seed)

- **P(PROBE-GAIN | pass-1 precompute of n = 8):** ≈ 0 (PROBE(1) would be high).
- **P(VALUE-SWAP SERIAL-STEP | the recompute-from-V_0 null):** ≈ 0.05, from §203-4's failed class at 0.003–0.18.
- **P(both | real serial use):** ≈ 0.6–0.7. Some loss comes from distributed value storage reducing arm (i), and from swap disruption.
- **Readings:** the trained ITER-GAIN / SERIAL-PROFILE are kept as descriptive only.
- **Seeds:** one seed remains a pilot. A null is weak, given the 0.4–0.55 lottery.

## G. Recommended changes before registration (in order)

1. Replace the decisive set with DEEP ∧ PROBE-GAIN ∧ PROBE-ONSET ∧ VALUE-SWAP (F1, F2, D).
2. Make L_rec reweight changed and value slots (F3a).
3. Main cell PURE-ANNEAL; CARRY as the control (F4).
4. Add DIALECT-ANCHOR to R3 and keep R3; it is the lever matching the §205 diagnosis (F3, F6).
5. Gate G1b, the init drift smoke (F5).
6. Onset over the trained window, and γ initialised to ALiBi slopes (F7).
7. Path-specific gradflow, a new gate class, G-TIME on the final stage (F9, F10).
8. GSM8K: a value-first target, PROBE-GAIN, non-banked STAGGER snapshots (F8).
