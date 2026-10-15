# QUESTION-ONCE (with the ANSWER-BOTTLENECK) on GSM8K-Aug: design + derivation DRAFT (2026-10-15 10:50 EDT)

Opus subagent for Theory, after the ANSWER-BOTTLENECK draft (docs/THEORY_ANSBOTTLE_DRAFT_2026-10-15.md, "AB draft"; committed by Theory). **Not registered.** It needs Theory's and Audit's review and **the user's OK**. No git, no theory/score_*.py run, no boxes/ edit.
- **Scripts (CPU, Idle, one thread, no model loaded):**
  - theory/ansbottle/qonce_calc.py → qonce_calc.json: literal counts, the copy-fidelity cap, A by literal count, token-block cost;
  - theory/ansbottle/qonce_pairs.py → qonce_pairs.json: transplant-hybrid kept fractions;
  - theory/ansbottle/pairs_calc.py (the AB draft's pair machinery).
- **Inputs read, beyond the AB draft's:** THEORY.md §153 (PURE, note 153-3), §155, §165, §169, §170, §171, §179, and the index rows 150–179.
- **Coordinator items folded in:** the 8 derivation points; Audit's QO-CAPACITY-BOUND as its own named reading; the internal precedent (§1).

**Tags** as in the AB draft: PROVED (code), PROVED, DERIVED|A…, CONJECTURE, DATA / DATA (POST-HOC), calc.

---

## 0. Summary

1. **AB+QO:** pass 1 reads [EOT, q]; passes 2–10 read a fixed 17-token filler ([EOT] + FILLER-N k16); the decoder is AB's. Then y = Ψ(F⁹(Φ₁(q))) with F item-independent (PROVED).
2. **It removes every recompute-from-input route, in the decoder and in the passes,** including AB's upstream one (PROVED). §207's swap arms become identities. Re-injection, Huginn's design and our INJECT's, is the channel §207 found.
3. **Remaining route: PASS1-PRECOMPUTE**, at most a₁ ≲ 0.13. **Decisive reading ITER-GAIN:** ACC beats the matched one-pass control MEAN-ITER(1), K′1 and K′1-RESCALED by ≥ T (deep ≥ T_D). Passes 2–10 have no item input, so any gain is computation on memory.
4. **Serial vs two-stage:** ONSET(m) on the matched K′ grid, plus a PROGRESS-TRANSPLANT hybrid (0.61 strict-kept at t = 2) and a value probe.
5. **The copy is the cost INJECT avoided (§169 / §170).** Slots and bits do not bind (≤ 15 elements vs 102 addressed slots). Fidelity and retention do: the cap is 0.12 at A's per-operand fidelity (0.68), 0.27 at 0.8. Named reading **QO-CAPACITY-BOUND**.
6. **Precedent:** §165 (problem once, constant passes, blind decoder) gave 0.0 under FINAL-only, before the step aux existed. QO + step aux is untested.
7. **Prior:** ACC 0.075 (HURTS 0.72); ITER-GAIN P 0.35; QO-SERIAL P 0.14. Labels: PASS1 0.45 / TWO-STAGE 0.15 / SERIAL 0.14; CAPACITY-BOUND 0.35 alongside.
8. **Recommendation: AB+QO alone, one cell vs A, $2.0 central / $3.0 high.** AB becomes a later control only if ITER-GAIN passes. **Main risk:** a lossy copy (ρ < 0.72) caps the iteration route below PASS1.

---

## 1. Internal precedent (Audit's required part)

| § | design | outcome | lesson |
|---|---|---|---|
| **153 PURE** (2026-10-07) | pass 0 = the problem (write only); passes j = item-independent [EOT, step j, 4 register tokens] | **never ran.** Note 153-3: with input-only writes, item-independent passes store nothing, so PURE fails by construction. Moved to §155's write-after-read memory and gated on §155's proof, which failed SERIAL | "C1 holds by design, L1 cannot apply". **A whole-memory swap cannot discriminate** (it moves the problem with the state), so PURE's SERIAL was a pass-skip test. Both carry over to QO |
| **165 FIXED-CHUNK** (2026-10-08/09) | problem once (4-token chunks), then K constant think passes [EOT, " ."], **blind** decoder [EOT] → digits; FINAL-only, 4×n mult, 48k | **0.0** at every K′ and skip; memory unused | QO + AB's shape, but FINAL-only on multiplication |
| **169 INJECT** (2026-10-09) | the full problem re-presented every pass | memory unused (FINAL-only, 48k) | **Chosen** because: (a) looped models generalise better with input injection (Fan 2409.15647's no-injection ablation; Huginn 2502.05171); (b) Audit: "the memory never needs to buffer operands", removing 155-4's input-buffer shortcut; (c) §165's passes "cannot re-consult the operands". §170: "the memory never needs to buffer the input or a stack, so stack capacity does not bind" (THEORY.md:10603) |
| **171 READ-SLOTS** | input encoded once; 8 slots attend to its KV **every pass** | memory unused; BLIND 0.0 | the question is re-read through attention: **not** QO |
| **175 STEP-AUX** (on INJECT, nested) | + a per-pass read-only aux decoder | passes **used**: one reduction per pass, K′0 at chance | **per-pass supervision, not the input regime, made identical-input passes trainable.** §191 / §200 inherit INJECT + aux |
| **179 READOUT** | a **blind** end decoder on the mult scaffold | **CAPACITY**: 0.0; columns 1–2 exact from memory alone | **an AB precedent the AB draft omitted:** a blind decoder could not carry an 8-digit product. GSM8K's answer is one token for 90.6% of items (≈ 7.6 bits mean), a lighter load, but the risk applies to AB and AB+QO alike |

**Why revisit INJECT now.**
* **(i) §207:** on GSM8K, re-presentation is the recompute channel. INJECT's virtue ("the memory never needs to buffer") is exactly what lets every GSM8K model leave the memory item-content-free.
* **(ii) The trainability problem that sank §165 / §169 was solved by per-pass supervision (§175), not by INJECT.** QO with the step aux has never run.
* **(iii)** On ProsQA / nested, INJECT is harmless because the step target is the chain *state* (STEPFACTOR D6). On GSM8K the step target is the *plan* (parallel, PLAN-LEAK ≈ 0.9–1.0), so re-presentation hands every pass a one-pass shortcut.
* **QO deliberately reintroduces the buffering INJECT removed.** Hence §4 and QO-CAPACITY-BOUND.

---

## 2. Facts

**2.1 Code (PROVED (code); line numbers as in the AB draft §1.1).**
* Think pass = `ProsMem.write(S, x, mask, h_low)` = gpt_forward_mem_r. The read is at block-6 entry at every position, queries from mem.ln(h_6) (deltamem.py:276–279). Writes run over every unmasked token from u = (1−γ)LN_6(h_6) + γ·LN_w(h_11) (:179–223).
* cache_lower computes blocks 0–5 once per item and asserts the identical input (prosqa168s2.py:124–130).
* The training loop calls m.write per pass (gsm8k_stepaux.py:239–245); eval calls m.think (:327–335).
* The aux decoder reads M_t after pass t with a question-free prompt (prosqa168s2.py:161–170). γ = 0 to 2k, then a ramp to 1 by 6k (gsm8k_stepaux.py:399).
* **FILLER-N k16** = "The morning was calm and quiet, and the road ran along the river. ": exactly 16 GPT-2 tokens, no digits or names (bench/data/gsm8k_aug/s200_test_meta.json). The latent arm was LENGTH-ROBUST to it (§200 R6, excess −0.040 at k32).

**2.2 Data (calc / DATA, POST-HOC).**
* **Test literals** (distinct question numbers): mean 3.12, p90 5, max 9. Shares 1 / 2 / 3 / 4 / 5 / 6+: 0.10 / 0.22 / 0.32 / 0.20 / 0.10 / 0.05.
* **Copy elements** (literals + one plan entry per equation): mean 6.36, p90 10, max 15.
* **Question tokens:** test mean 56.8 (p90 86, max 182); train mean 42.4 (+ EOT 43.4).
* **A's accuracy by literal count** (eval_items.json, 3090 bf16): 1: 0.236 (n 127); 2: 0.187 (284); 3: 0.164 (420); 4: 0.097 (267); 5: 0.069 (130); 6+: 0.059 (68).
  - So R_A = acc(n_lit ≥ 4) / acc(n_lit ≤ 2) = 0.084 / 0.198 = **0.42**, the difficulty slope.
* **A's memory-held operand fidelity** (STEPFACTOR §1.4, Aug valid, teacher-forced top-1 from M_t): q-operand 0.642 (0.717 candidate-restricted), operator 0.699. This is A's realised "copy" in its aux slots: **ρ_A ≈ 0.64–0.72**.

---

## 3. Mechanism

**QO-0. The change.** Pass 1's input is [EOT, q], unchanged. Passes t ≥ 2 get x_f = [EOT] + ENC(k16), 17 positions, mask all true, identical for every item and pass. The decoder is AB's ([EOT, ANS] + answer, reading M_K). Everything else is A's recipe.

**QO-1. After pass 1 the model is an autonomous map on the memory (PROVED, code given the flags).**
* For t ≥ 2, blocks 0–5 see only x_f, so h_6 and every read query at the 17 positions are item-independent constants.
* The pass is M_t = F(M_{t−1}), with F the **same function for every item**. The decoder Ψ is item-independent (AB draft B1). So **y = Ψ(F^{K−1}(Φ₁(q)))**, with Φ₁ = pass 1 from M_0 = 0.
* **Consequences:**
  - (a) no pass ≥ 2 and no decoder can recompute from q;
  - (b) every item-dependent bit after pass 1 lives in M;
  - (c) any accuracy that passes 2..K add over a matched pass-1 control is **computation by F on the memory**.

**QO-2. In the open-loop window, F is an item-independent affine map (PROVED).**
* At γ = 0, the writes of passes ≥ 2 come from LN_6(h_6(x_f)): constant k, v, α, β. So S_del ← S_del·P_f + C_f and S_add ← S_add + A_f.
* For steps ≤ 2000 passes ≥ 2 compute nothing item-specific, and AB+QO is a pass-1 model.
* At γ = 0 **no arm composes** (A, AB and AB+QO alike: writes do not depend on reads; AB draft B7). So the window does not handicap QO's composition relative to A or AB. It does train a pass-1 solution first (risk 2).

**QO-3. The filler (DERIVED; precedents note 204-2 and §203-8).**
* **EOT-only (1 position):**
  - The pass reads one fixed query, 768 floats, and attention over one position is trivial, so the pass is a 6-block MLP on one vector.
  - It writes one token, rank 1 per head. The rest of the table is reachable only through the delta erase term −αβ(S·k)(k·q̂), i.e. content addressing by the write key.
  - Too narrow for several carried values, and it cuts the attention-based circuits GPT-2 uses to gather operands.
* **A fixed filler of F distinct tokens:**
  - F fixed queries give F addressed slots per head. Blocks 6–11 attend across the F positions, a latent workspace like LOTUS's latent positions or Perceiver latents. F tokens are written per pass.
  - **F = 16** matches the need: ≤ 15 copy elements, ≤ 16 key-value items per test item (LOTUS contrast §2).
* **Which tokens:**
  - Note 204-2's lesson is that an input the trunk never saw (an all-zero latent) is off-distribution, while MEAN is in-distribution.
  - A repeated token (" ." × 16) gives queries that differ only by wpe: near-collinear addresses, plus GPT-2's repetition attention patterns.
  - A natural-language filler keeps h_6 in GPT-2's normal range at step 0. The filler is then trained on every pass, so it is in-distribution by construction.
  - §203-8 (failed seeds follow a swapped input) cannot apply: there is no item input after pass 1 to follow.
  - **Choice: [EOT] + FILLER-N k16**, in-house, frozen, already shown harmless as a prefix.
  - Consequence for the FILLER-N eval: k16 prepended to the question at pass 1 now repeats the filler. Under QO, FILLER-N reports k32 as the reading and k16 flagged.
* **Not chosen: learnable filler embeddings** (LOTUS's ⟨lat⟩). They add parameters, a third change.

**QO-4. What C1 becomes (Theory's (1)).**
* **Preserved:** the memory is the only cross-pass channel. Passes ≥ 2 have identical input. K′0 is chance by construction (AB draft B2; ≤ 0.030).
* **Gained, PROVED by QO-1:** "the thoughts do the work" becomes structural. After pass 1 the only computation that can change the answer is F on M. Step-specific blocking means mean-ablating or skipping pass t of F (§5.2). The overthinking question (K′ > K) is now about the fixed points of an **autonomous** map.
* **Lost:**
  - (a) **Pass 1 differs from passes ≥ 2.** cache_lower now holds one per-item cache (pass 1, used once, so no saving) and one filler cache, computed once per batch.
  - (b) **The question as a re-readable tape.** A legitimate serial algorithm that re-fetches a literal at step t must now store it from pass 1, which is the buffering §169 avoided (§4).
  - (c) **The input-based interchange test.** With the program (the question) re-presented and the register in memory, a swap gives a hybrid (§207, AB's TEMPLATE-SWAP). Under QO the program is in the memory too, so a whole-memory swap moves both (§153 PURE's observation).
  - §207's arms all collapse to identities (§5.1). The replacement is the PROGRESS-TRANSPLANT (§5.4), which needs a separability assumption, plus the matched K′ profile.

**QO-5. Why QO removes the recompute route: the re-injection framing (Theory's (2)).**
* **The injection family:**
  - Huginn 2502.05171 re-injects the embedded input e every recurrence, s_t = R(e, s_{t−1}). The abstract does not state it; this is from the method section via Literature's reads.
  - SCSE 2607.27656 (verified): "in additive-injection looped Transformers, an input-conditioned signal is reintroduced at every recurrent step", and it builds a learned anchor that stays a fixed point under repeated injection.
  - OrthoInj, "Towards Looped Models Done Right, Part II", 2610.06833 (verified): "existing injection schemes let the state's component along the input amplify or cancel the injection; we remove this component with orthogonal injection."
  - Wang & Reid 2609.01924 (verified ID; Literature's full text): re-injected content overrides a swapped state within about 2 recurrences.
* **Our INJECT is the strongest member of that family:** the whole forward re-runs on the input, so the state is a side input.
* **§207 is the GSM8K instance of "re-injection overrides the state":** every model follows a swapped input at 0.87–0.92.
* **QO sets the injection to zero after the first step**, so the recurrence is autonomous: s_t = F(s_{t−1}), s_1 = Φ₁(e). The recompute route needs e at step t, and e is absent (PROVED, QO-1).
* **The cost, DERIVED from the same literature:** injection exists to anchor the recurrence (SCSE's fixed point; path independence). Without it, F must itself preserve the problem across 9 passes (retention, §4) and must not drift past K. So STABLE-12/16 become informative readings, not formalities.

**QO-6. The remaining route, and the decisive reading (Theory's (3)).**
* **PASS1-PRECOMPUTE:** pass 1 sees q and can compute y in one forward, write it, and F holds it.
  - Its ceiling is a₁ ≲ K′1(A / W) = 0.125 / 0.129 (AB draft B5(b), A-NEST).
  - It is at most AB's a₁, because F must also hold the answer through 9 item-free passes.
* **DECISIVE: ITER-GAIN** (§5.2): ACC − max(MEAN-ITER(1), K′1, K′1-RESCALED) ≥ T pooled **and** the same on deep items ≥ T_D.
  - It is attributable without a comparator: QO-1(c) says passes ≥ 2 carry no item input, so a gain is latent computation.
  - Beside it goes the cross-model check Theory named: ACC vs the one-pass level of A / W (0.125 / 0.129 + T = 0.157), descriptive.
* **Two other routes give ITER-GAIN without serial computation:**
  - **TWO-STAGE:** pass 1 copies or reduces one level, then pass 2 finishes the computation in one shot from the copy and F holds the result;
  - **DENOISE:** F cleans a noisy pass-1 answer code.
  - Both show an onset that does not depend on m. Serial computation shows an onset that grows with m. Hence ONSET(m) (§5.3) separates SERIAL from TWO-STAGE / DENOISE.

---

## 4. Capacity and the copy (Theory's (4); Audit's QO-CAPACITY-BOUND)

**4.1 What pass 1 must write (DERIVED|QO-1).** Unless pass 1 computes y outright (PASS1), M_1 must hold everything F and Ψ will use: the needed literals, the plan (operators, which operand goes where), and any value pass 1 already computed. Copy elements: mean 6.36, p90 10, max 15 (calc). Values computed later add ≤ m more, so ≤ 23 key-value items at the maximum.

**4.2 Capacity does not bind by count or bits (DERIVED|A-CAP; LOTUS contrast §2).**
* **Delta heads:** ≈ DH = 64 near-orthogonal keys per head (Schlag 2102.11174), 6 heads, 384 items in the table.
  - Under QO, passes ≥ 2 read **17 fixed queries per head** (QO-3), so **102 directly addressed delta slots**, plus the erase-addressing path. The decoder reads 2 fixed queries per head (AB draft B3).
* **Additive heads:** dense positive features make each head a bag, ≈ 1 retrievable item (≈ 6 in all; 64 only with disjoint supports).
* **Needs:** ≤ 23 items against 102 addressed or 384 table slots. Bits: ≈ 6 per literal (STEPFACTOR D3) × ≤ 9 literals ≈ 54 bits, plus the plan (≈ 29 mean), against kilobits.
* **A verbatim token copy would not fit the addressed slots** for the 4.4% of test questions with L + 1 > 102. It is also unnecessary: pass 1 can compress to the elements (the gathering work, AB draft B3).

**4.3 What can bind: fidelity and retention (DERIVED + calc).**
* **Fidelity:** model each needed element as surviving into the representation F uses independently with probability ρ. The iteration route's accuracy is then **≤ cap(ρ) = E_x[ρ^{n_lit(x)+m(x)}]** (an upper bound with perfect computation).

  | ρ | 0.60 | 0.68 | 0.70 | 0.80 | 0.90 | 0.95 | 0.99 |
  |---|---|---|---|---|---|---|---|
  | cap(ρ) (test) | 0.068 | **0.120** | 0.138 | 0.273 | 0.526 | 0.727 | 0.938 |

  - By literal count at ρ 0.8: 0.44 / 0.36 / 0.27 / 0.19 / 0.14 / 0.09 for n_lit 1 / 2 / 3 / 4 / 5 / 6+.
* **The sharp consequence:**
  - At A's realised operand fidelity (ρ_A 0.64–0.72) the iteration route is capped at about 0.10–0.15. That is **at or below the PASS1 ceiling a₁ ≈ 0.13**, which needs no copy at all.
  - The iteration route can only beat PASS1 if **ρ ≳ 0.72**, and it needs ρ ≳ 0.8 to reach about 0.27.
  - Gradient descent will prefer PASS1 unless the copy becomes much more faithful than A's aux slots are today. The answer CE through F and the aux CE (whose 74% operand share is now a copy teacher) both push it up.
* **Retention across 9 filler passes:**
  - The delta channel decays by α per written token: at the initial α ≈ 0.993, 0.993¹⁷ ≈ 0.89 per pass, 0.34 after 9. Filler writes also erase along their keys.
  - The model can learn α → 1 and small β on filler positions (the gates come from u), the progress-gated hold of 199-E1's converged models. It must learn it.
  - The additive channel never decays, but is a bag.
* **Answer to "what if it can't copy":** accuracy is capped at cap(ρ_eff) on the iteration route, and the optimiser falls back to PASS1 (a₁). A lossy copy pushes toward the Goodhart route.

**4.4 The named reading QO-CAPACITY-BOUND (separate from a computation failure).**
* **Measurements:**
  - **LIT-RECALL(t):** a probe of each needed literal (first-token class, top 200 + OTHER) from the **pass-visible read** of M_t. The features are the 17 filler-position reads, PCA to 1,024 on train, then L2 logistic: fit on 4,000 fresh-R_x Aug-train items, read on Aug valid. Taken at t = 1 and t = K, by n_lit ≤ 2 vs ≥ 4. Floors: M_0, y-only, length-only (AB draft §3.6).
  - **R_QO** = acc(n_lit ≥ 4) / acc(n_lit ≤ 2) on the test, against R_A = 0.42.
* **QO-CAPACITY-BOUND** if R_QO ≤ 0.5 × R_A **and** either:
  - **(FIDELITY)** LIT-RECALL(1) at n_lit ≥ 4 ≤ LIT-RECALL(1) at n_lit ≤ 2 − 0.10, beyond the same drop in A's aux-slot operand top-1 (the difficulty control); or
  - **(RETENTION)** LIT-RECALL(K) ≤ LIT-RECALL(1) − 0.10.
* **QO-COMPUTE-FAIL** (the copy is fine, the computation is not): LIT-RECALL(1) and LIT-RECALL(K) ≥ floor + 0.30 at every n_lit stratum, ITER-GAIN fails, and ACC ≤ A − T.
* **Noise:** R_QO's relative SE ≈ 27% at the prior's accuracy (acc ≈ 0.04 on n_lit ≥ 4, n 465). So 0.5 × R_A is about 2 SE from equality. LIT-RECALL has n ≈ 500 items × literals.

---

## 5. Readings re-derived under AB+QO (Theory's (6))

**5.1 §207's arms and AB's TEMPLATE-SWAP collapse (PROVED, QO-1).**
* Any SWAP(t; a → b) with t ≥ 1 never feeds b's question to anything, because passes ≥ 2 and the decoder are item-free. So it is **≡ clean(a)**, and P(y_a) = 1 on a-correct items: an identity gate.
* The only item-mixing point is pass 1's input. Swapping it is trivially clean(b).
* So §207's labels (RECOMPUTES-FROM-INPUT, CARRIES-STATE, 207-1's SERIAL / PASS1-CARRY), AB's UPSTREAM-RECOMPUTE and AB's TEMPLATE-SWAP all become **gates, not readings**. G-SWAP in §12 checks the identity bitwise.

**5.2 MEAN-ITER(k), the matched truncation, and ITER-GAIN.**
* **Definition:** run passes 1..k normally. Replace passes k+1..K by their **item-mean** effect, the note 204-2 MEAN condition: in-distribution statistics with the item-specific computation removed.
  - **Additive:** S_add(K) − S_add(k) = Σ_{s>k} W_s exactly (a FAST_ADD sum), replaced by its batch mean.
  - **Delta:** with the realised gates recorded, S_del(K) = S_del(k)·P_{k+1..K} + C_{k+1..K} (affine; PROVED). P and C are obtained by replaying on I and 0 and are replaced by their item means (P̄, C̄).
* **Why it beats K′1 and K′1-RESCALED:** it keeps the decoder at K-pass statistics (the AB draft's B6 truncation artefact) while removing everything item-specific after pass k. K′1 and K′1-RESCALED are reported beside it.
* **ITER-GAIN (decisive)** = ACC − max(MEAN-ITER(1), K′1, K′1-RESCALED) ≥ T = 0.028 pooled, **and** the deep version (m ≥ 3, n 879) ≥ T_D = 0.034.
* Beside it: **ITER-GAIN-ONLY-RESCALED** (passes the rescaled control but not MEAN-ITER), reported, not credited.

**5.3 ONSET(m) (serial vs two-stage).**
* ONSET(m) = the smallest k ∈ {1, 2, 3, 4, 6, 8, 10} with MEAN-ITER(k) accuracy on stratum m ≥ 0.8 × ACC(m), for m = 1, 2, 3, 4, 5+ with n ≥ 50.
* **SERIAL-PROFILE:** ONSET is non-decreasing in m, and ONSET(m ≥ 4) ≥ ONSET(m ≤ 2) + 2.
* **ONESHOT-PROFILE:** the same ONSET for every stratum: 1 for PASS1; 2 or 3 for TWO-STAGE or DENOISE.
* An unresolvable stratum (ACC(m) < 2 / n) is excluded and listed. Fewer than 3 resolvable strata makes the profile UNREADABLE, which counts as SERIAL-PROFILE not met.

**5.4 PROGRESS-TRANSPLANT (the QO interchange; PROVED construction, A-SEP for its meaning).**
* **Pairs:** AB's TRAIN-REDRAW template pairs (r, d), the same template with different literals (rng 20711, 4,000 sampled templates), both clean-correct.
* **The hybrid state after pass t (t = 2, 3):**
  - additive: W₁(r) + Σ_{s=2..t} W_s(d);
  - delta: S_del^r(1)·P^d_{2..t} + C^d_{2..t}.
  - In words: the recipient's pass-1 content plus the donor's passes 2..t. Passes t+1..K then run normally (item-free) and Ψ decodes.
* **A-SEP (assumption):** the copy is written at pass 1 and progress at passes ≥ 2, in slots that agree across one template's instances.
* **Hybrid targets, both schedule conventions (calc, TRAIN-REDRAW, 23,758 pairs):**

  | convention | t = 2 hybrid (strict-kept) | t = 3 hybrid (strict-kept) |
  |---|---|---|
  | **C** (pass 1 copies; pass s computes step s−1): donor steps {1..t−1} | **0.608** (m ≥ 2: 0.69–0.79) | 0.164 (m ≥ 3: 0.34–0.68) |
  | **S** (pass 1 also computes step 1): donor steps {2..t}, recipient's v_1 | 0.287 (m ≥ 3: 0.73–0.77) | 0.058 (m ≥ 4) |

* **Predictions by route:**
  - SERIAL → y_hyb under its convention;
  - PASS1 → y_own (the answer is in the recipient's pass-1 content);
  - TWO-STAGE (pass 2 computes from the copy) → y_don if F then holds, y_own if F recomputes from the copy every pass;
  - DENOISE → y_own.
* **TRANSPLANT-SERIAL:** P_hyb ≥ 0.30 at t = 2 under the better convention, **and** P_hyb > max(P_own, P_don) there.
  - These are 4 targets in all, and the accidental hit rate is ≤ 0.03 each, so P(any ≥ 0.30 at n ≥ 100) is still < 10⁻¹⁵.
* **Identities (gates):** t = 1 is clean(r); the full donor state is clean(d); r = d is clean(d).

**5.5 Probes.** LIT-RECALL(t) (§4.4). **INT-PROGRESS**: the §204 VALUE-EXACT / VALUE-LEAK logic as probes. On ℓ ≥ 2 R-steps, v_t from the pass-visible read of M_t (t ≥ ℓ+1) ≥ floor + 0.10, and from M_1 ≤ floor + 0.05. ANS-LIN at the decoder (AB draft §3.6), descriptive.

**5.6 Labels (one seed; on the final checkpoint).**
* **QO-SERIAL (the C1 success):** ITER-GAIN ∧ SERIAL-PROFILE ∧ (TRANSPLANT-SERIAL ∨ INT-PROGRESS).
  - Both of the last two are reported. One is required because each rests on its own assumption (A-SEP, probe linearity).
* **TWO-STAGE:** ITER-GAIN ∧ ONESHOT-PROFILE with ONSET ≥ 2. Multi-pass use (C1's weak form), not serial; not success.
* **PASS1-PRECOMPUTE:** ITER-GAIN fails while MEMORY-CARRIES holds (ACC ≥ P_test(y = c) + T). Not success.
* **QO-CAPACITY-BOUND / QO-COMPUTE-FAIL** (§4.4): reported alongside any of the above.
* **MEMORY-FAILS:** ACC < chance + T.
* **STABLE-12 / STABLE-16** (A's rules), read as the autonomous map's drift test (QO-5).

---

## 6. Accuracy prior for AB+QO (Theory's (5); CONJECTURE)

* **The routes and their caps:**
  - PASS1 ≲ a₁ (0.09–0.13), minus the holding loss;
  - the iteration route ≤ cap(ρ), which at ρ_A is 0.12 (§4.3);
  - both ride on AB's routing loss (AB draft §4: central 0.095).
* **What raises the iteration route:**
  - there is no competing recompute route at all;
  - the step aux makes the plan copy a trained target at every pass (it is a copy teacher);
  - the answer CE reaches pass 1 only through F⁹, so the copy and the hold are trained end to end.
* **What lowers it:**
  - §165's 0.0 (though FINAL-only, before the aux);
  - 2k open-loop steps in which only PASS1 can learn (QO-2);
  - F must learn arithmetic on vector-coded operands at 17 filler positions, a new circuit, not GPT-2's token arithmetic;
  - retention must be learned.
* **ACC central 0.075** (80% interval 0.03–0.14): **HURTS P 0.72**, MATCHES 0.25, LIFTS 0.03.
* **Mechanism:** ITER-GAIN P 0.35; SERIAL-PROFILE given ITER-GAIN 0.45; **QO-SERIAL P ≈ 0.14**.
  - Labels: PASS1 0.45, TWO-STAGE 0.15, QO-SERIAL 0.14, MEMORY-FAILS 0.10, the rest MIXED.
  - QO-CAPACITY-BOUND P 0.35 alongside.
* **QO-SERIAL's P exceeds AB's SERIAL-CARRY (0.12)** only modestly. QO removes the recompute route, but a lossy copy makes PASS1 the cheaper basin (§4.3).

---

## 7. Goodhart routes and detectors

| route | why reachable | detector | counts as |
|---|---|---|---|
| PASS1-PRECOMPUTE | pass 1 is a full forward over q (QO-6); the only route in the window (QO-2) | ITER-GAIN fails; ONSET 1; transplant y_own | not success |
| TWO-STAGE / DENOISE | F can finish in one shot from the copy | ITER-GAIN with ONESHOT-PROFILE (ONSET ≥ 2 for every m) | not success |
| Decoder computes from the copy | Ψ reads M_K; s_B ≤ 1 (AB draft B4) | needs no iteration, so MEAN-ITER(1) ≈ ACC; ITER-GAIN fails | not success |
| Stepwise copy via aux slots | the aux installs per-pass operand slots | same as the row above (the decoder executes from slots written by pass 1 and carried): no ITER-GAIN | not success |
| Clock-scheduled register (PASSK) | the additive counter (∝ t, no RMS) | ONSET fixed at a late k for every m: ONESHOT-PROFILE at k ≥ 6 | not success |
| Recall | pass 1 recognises a train question | fit_fix − fit_rs reported; transplant and probes on fresh R_x instances | descriptive |
| Truncation artefact | the decoder is trained on K-pass statistics | MEAN-ITER(1) and K′1-RESCALED beside K′1; ITER-GAIN takes the max | guard |
| Lossy copy | §4.3 | QO-CAPACITY-BOUND | named, not success |

* **COPIES-LITERALS is no longer a Goodhart route under QO:** a copy is *required*. The Goodhart question becomes whether the copy is *computed on* by F (ITER-GAIN) or only decoded.
* **An unevaluable detector is not clear** (rule b).

---

## 8. Gradient flow (PROVED, code, for the AB+QO path; gate G3)

* **L_ans reaches:**
  - the decoder path (AB draft §7 (i), (ii));
  - **through M_K → F⁹ → M_1:** every filler pass's write parameters (qkv[D:3D], gate, mem.ln at weight 1−γ, ln_w at weight γ), the read queries at the filler positions, and trunk blocks 6–11 at the filler positions (γ > 0 only);
  - **pass 1's** writes and trunk blocks 0–11 on q (blocks 6–11 only for γ > 0; blocks 0–5 at every γ via h_6).
* **The question's tokens receive gradient only through M_1** (QO-1). At γ = 0 the filler passes' parameters get **exactly zero** gradient from L_ans through item content (QO-2). They still get constant-write gradients.
* **L_aux:** reads M_t after pass t. Under QO, M_t for t ≥ 2 is F-produced, so the aux gradient reaches F and, through it, pass 1's copy (the copy-teacher path).

---

## 9. Design, OC and censoring (Theory's (6))

**Recommendation: one cell, AB+QO vs A (A's base), with AB as a later control.**
* **(i)** The decisive reading (ITER-GAIN with matched controls) is **within-cell**, and it is attributable by QO-1 without a comparator.
* **(ii)** AB alone was declined (likely null). Its diagnostic value, which upstream route appears, is mostly moot once AB+QO is run:
  - AB+QO reads PASS1 → AB also has PASS1, plus its recompute route, and adds little;
  - AB+QO reads ITER-GAIN or SERIAL → AB becomes the control that shows QO was necessary, run with stage 2.
* **(iii)** It halves the cost. The price: AB+QO vs A is a **two-change** comparison for ACC, worded as "the combined architecture". ACC attribution between AB's routing and QO's copy relies on the QO-CAPACITY-BOUND probes, not on a cell.
* **QO without AB is pointless:** a decoder that sees q recomputes (§207), and the memory goes unused.

**Cell (RTX 3090, seed 1, bs 64, lr 1e-4, warm-up 100 + cosine, 24k; final checkpoint; test never used for selection; the Aug-valid-selected checkpoint beside, labelled):**
* **QA** `gsm_lat_ansbn_qonce_s1` = §200 A's flags + `--ans_bottleneck --question_once` (filler = [EOT] + FILLER-N k16).
* **Comparator:** A, with its eval-only re-run for the new fields; acc and acc_by_kprime bitwise to eval.json.
* All new eval-only readings run on the 3070 at eval_bs 64, card / dtype / bs recorded.

**Readings (T 0.028, T_D 0.034):**
* ACC vs A (LIFTS / MATCHES / HURTS; two-change wording);
* MEMORY-CARRIES;
* **ITER-GAIN (decisive)**;
* ONSET(m) / SERIAL-PROFILE; PROGRESS-TRANSPLANT; INT-PROGRESS;
* QO-CAPACITY-BOUND / QO-COMPUTE-FAIL;
* THINK-NECESSARY (the K′1 clause with the matched control; K′0 reported, vacuous); THINK-NECESSARY-DEEP;
* STABLE-12 / 16; recall share.
* **Descriptive:**
  - the cross-model one-pass check (ACC vs 0.157);
  - the energy split on the transplant, per channel;
  - channel-split swaps (delta-only / additive-only state swaps at t = 2: P_hyb, P_own, P_don; layout unknown);
  - answer_ce entry;
  - FILLER-N (k32; k16 flagged);
  - IDLE-PASSES.

**OC block (rule a; one seed).**
* **P(QO-SERIAL | no effect) ≈ 0.005–0.01.**
  - Under a PASS1 model, ACC − MEAN-ITER(1) is ≈ 0, and the paired discordance SE is ≈ 0.008, so P(≥ T) ≈ 0.02.
  - A spurious SERIAL-PROFILE from noise: ≈ 0.1.
  - TRANSPLANT-SERIAL ≈ 0 by sampling, and INT-PROGRESS false positive ≈ 0.05.
  - The product is about 0.0002; systematic artefacts dominate, ~0.005–0.01.
* **P(QO-SERIAL | effect) ≈ 0.2–0.4:** a carrying fraction p_s 0.3–0.6 of seeds, times detection ≈ 0.7.
* **LR(positive) ≥ 20; LR(null) ≈ 0.7.** A one-seed null is weak evidence. A rate claim needs ≥ 3 seeds plus the AB control (stage 2, a new decision).
* **ITER-GAIN alone:** P(| no effect) ≈ 0.02 by sampling; ≈ 0.2 if "no effect" means "no serial use", since TWO-STAGE also passes it. That is why it is the decisive reading for **"passes ≥ 2 do work"**, and QO-SERIAL is the C1 label.
* **ACC:** P(LIFTS | no effect) ≈ 0.02 (between-recipe SD 0.009).
* **Realised $:** replaces the projection.

**Censoring (rule b).**
* A run ending before 24k, dying or aborted: every reading fails, with the event-only values beside.
* MEAN-ITER replay failing its identity gate (G6): ITER-GAIN not credited.
* ONSET unreadable (< 3 resolvable strata): SERIAL-PROFILE not met.
* Transplant unreadable (< 100 pairs at t = 2): TRANSPLANT-SERIAL not met, but QO-SERIAL may still pass through INT-PROGRESS. If both are unavailable, QO-SERIAL fails.
* Probes unfittable: INT-PROGRESS not met, and QO-CAPACITY-BOUND / QO-COMPUTE-FAIL not determined (reported as such).
* answer_ce never ≤ 0.92 per token: NO-ENTRY, a failure, with the event-only step beside.

---

## 10. Predictions and falsifiers (one seed; P's subjective)

| id | prediction | central / P | falsified if |
|---|---|---|---|
| P-Q1 | ACC HURTS vs A | 0.075; HURTS 0.72 | ACC ≥ 0.147 falsifies the central (the copy-cap / routing account) |
| P-Q2 | the swap-identity gates hold (PROVED) | — | any non-identity (a bug) |
| P-Q3 | ITER-GAIN | P 0.35 | — |
| P-Q4 | ONSET profile | PASS1 (ONSET 1) P 0.45; TWO-STAGE 0.15; SERIAL 0.14 | — |
| P-Q5 | QO-CAPACITY-BOUND | P 0.35 | — |
| P-Q6 | LIT-RECALL(1) ≥ A's aux-slot operand top-1 (0.64) + 0.10 (the copy improves under QO) | P 0.4 | — |
| P-Q7 | STABLE-16 fails (autonomous drift, QO-5) | P 0.4 (A held both) | — |
| P-Q8 | answer_ce at 2k ≥ 1.5 per token, then entry during the ramp | P 0.7 | — |
| P-Q9 | recall share | 0.30 [0.15, 0.45] | — |

* **Derived falsifiers:**
  - **(a)** ITER-GAIN passing while LIT-RECALL(K) is at floor would falsify QO-1's attribution, the claim that F's gain needs item content in M. It would point to a bug or a leak. A gate catches the leak.
  - **(b)** cap(ρ) is an upper bound. ACC > cap(LIT-RECALL-measured ρ) + T on the iteration route falsifies the independence model of §4.3.
* **No derivation predicts a lift.**

---

## 11. Cost

* **Forward token-blocks per item** (calc; train question mean 43.4 tokens with EOT):
  - A 3,836; AB 3,315 (0.86 × A); **AB+QO 1,889 (0.49 × A, 0.57 × AB)**.
  - Delta token-writes per item: 434 → 196.
  - The filler's blocks 0–5 run once per batch.
* **s/step:** the number of passes (10 think + aux) and so of launches is unchanged. With half the step time taken as compute-proportional: s/step ≈ AB × (0.5 + 0.5 × 0.57) ≈ 0.79 × AB.
  - From the realised A-family 1.27–1.82 (AB central 1.40): **central ≈ 1.10 s/step**, low 0.93, high 1.43.
* **Cell:** central 24k × 1.10 s = 7.3 h + eval 0.25 h = 7.6 h × $0.25 + $0.10 ≈ **$2.0**. High 9.8 h × $0.28 + $0.10 + one STUCK host $0.20 ≈ **$3.0**.
  - G7 timing (--time_steps 200 on a 3090) replaces the projection before launch.
* **Realised comparables:** F-P $2.56 (24k, A base); W $1.10 / J1 $1.63 (12k).
* **Eval-only:** local on the 3070, $0. Fewer think tokens make it faster than AB's: ≈ 1 h for QA plus A's re-run.
* **Proposed cap $3.2.** Two cells (AB + AB+QO): $4.6 central / $6.8 high.

---

## 12. Engineering contract and gates (Theory's (7); every new parameter optional; no module edit while any running job lazily imports these modules)

**12.1 Flags.** `--question_once` asserts `--ans_bottleneck`. Optional `--qo_filler` (default "k16", the s200_test_meta string). Both are recorded in args.json and asserted under --eval_only.

**12.2 One think-input helper** in ProsMem (or FactorMem), used by every think path.
* `m.pass_input(t, x, mask) → (x_t, mask_t, h_low_t)`:
  - t = 1 → the question;
  - t ≥ 2 → x_f broadcast to B rows, mask all true, with h_low_f = gpt_lower(x_f) computed once per batch (cache_lower).
* The training loop (gsm8k_stepaux.py:239–245), think_states / m.think, path_indep, idle, fit, struct and filler evals must all call it. The diff review greps for any remaining `m.write(S, x,` outside the helper.
* **cache_lower's assert** (prosqa168s2.py:125) becomes per source: pass 1 caches its own question, and passes ≥ 2 the filler.
* **path_indep under QO:** the donor-start segment re-runs pass 1 (the question pass) on the donor state, then the filler passes. This is a redefinition, descriptive only.
* **KPRIMES under QO** add 3: {0, 1, 2, 3, 4, 6, 8, 10, 12, 16}.

**12.3 New eval-only fields / script** (theory/ansbottle or scratch, as AB draft §11.3):
* MEAN-ITER(k) on the grid, with ONSET(m);
* K′1-RESCALED;
* PROGRESS-TRANSPLANT, both conventions, t = 2, 3;
* channel-split swaps;
* LIT-RECALL(t), INT-PROGRESS, ANS-LIN;
* R_QO and acc by n_lit.
* The hybrid targets come from pairs_calc.chain with the fixed-step sets of qonce_pairs.py.

**12.4 Gates.**
* **G1 plain path bitwise** (both flags off): a 200-step smoke and A's eval-only re-run, as in AB draft G1. **G1b:** with only `--ans_bottleneck` on, the AB path is unchanged by the QO edit (a 100-step smoke equal before and after).
* **G2 no question after pass 1.**
  - (a) Assert every pass ≥ 2 input row equals x_f, and the decoder rows start [EOT, ANS].
  - (b) Invariance: with M_1 fixed, M_K and the answers are **bitwise identical** when the item records handed to the loop after pass 1 have their questions scrambled.
  - (c) A positive control on the plain path, which must differ.
* **G3 gradflow** (fade 1, W_o small random, γ = 1):
  - L_ans alone gives nonzero grad on the filler-pass write parameters and on pass 1's writes;
  - the gradient w.r.t. the question tokens' input embeddings (a hook on pass 1's embed output) is nonzero, and **exactly zero with M_1.detach()** (the question reaches the loss only through M_1);
  - at γ = 0, L_ans gives zero grad on the filler passes' blocks 6–11 (QO-2).
* **G4 mem_check** at the longest question (lower than A's).
* **G5:** the K′0 constant (one distinct string). The filler lower cache equals a fresh gpt_lower bitwise.
* **G6 replay identities** (fp32, ≤ 1e-5 relative; fp64 impossible on the branch, note 204-4):
  - MEAN-ITER with per-item maps (no averaging) reproduces S_K;
  - on a batch of one repeated item, the mean map = the item map;
  - transplant t = 1 = clean(r); r = d = clean(d);
  - SWAP(t ≥ 1) = clean(pre) bitwise (P-Q2);
  - a scorer fixture covers the labels (SERIAL / TWO-STAGE / PASS1 / CAPACITY-BOUND / unreadable).
* **G7 timing** (HOSTPROBE + the first 200 steps) with a running guard at $3.2.
* **Then** construct + 1 step on the plain path (the post-edit smoke rule).

---

## 13. Prior art (Theory's (8); IDs verified on the arXiv abstract pages, 2026-10-15)

* **Re-inject every step** (the norm for looped latent reasoning):
  - Huginn, Geiping et al., **2502.05171** (verified; injection per its method section);
  - Fan, Du, Ramchandran & Lee, "Looped Transformers for Length Generalization", **2409.15647** (verified; the injection ablation is from Literature's read, not the abstract);
  - SCSE, Kim et al., **2607.27656** (verified; quoted in QO-5);
  - OrthoInj, Huang et al., **2610.06833** (verified; quoted in QO-5);
  - Wang & Reid, **2609.01924** (verified ID).
  - Saunshi et al., **2502.17416** (verified) loop the whole model; its injection is not checked here.
* **Question kept in context for the answer and the latents:** Coconut 2412.06769, CODI 2502.21074, PCCoT 2506.18582, SIM-CoT 2509.20317, LOTUS 2606.31779 (AB draft §12).
* **"Encode once, then iterate on a state":**
  - The classic case is the encoder–decoder "fixed-dimensional vector" (Sutskever, Vinyals & Le, **1409.3215**, verified), but its decoder is a generator, not a reasoner.
  - Universal Transformers (Dehghani et al., **1807.03819**, verified) iterate on the whole sequence of positions, so the question's own positions are the state. Not a bottlenecked memory.
* **Internal:** §153 PURE (designed, never ran); §165 FIXED-CHUNK (0.0, FINAL-only); §171 (not QO).
* **Novelty, marked honestly:** "question at the first pass only, then item-free passes over a matrix memory, a blind answer decoder, and per-pass step supervision" is **not found** for latent reasoning (CONJECTURE; Literature to confirm). The principle is the opposite of the field's injection practice, deliberately so, given §207.

---

## 14. Claim status

| claim | status |
|---|---|
| QO-1: y = Ψ(F^{K−1}(Φ₁(q))) with F item-independent; recompute impossible after pass 1 | PROVED (code, given the flags) |
| QO-2: F is an item-independent affine map at γ = 0 | PROVED |
| QO-3: filler choice (F = 16 natural-language tokens) | DERIVED + precedent (204-2, R6) |
| QO-4: C1 preserved / gained / lost; the input-based hybrid lost | PROVED (structure) + §153's observation |
| QO-5: re-injection is the recompute channel; QO zeroes it | PROVED (QO-1) + DATA (§207) + literature framing |
| QO-6: PASS1 ceiling ≤ AB's a₁ ≲ 0.13 | DERIVED\|A-NEST |
| ITER-GAIN attributable to computation on memory | PROVED (QO-1), modulo the MEAN-ITER in-distribution CONJECTURE |
| Capacity does not bind by count or bits (≤ 23 items vs 102 / 384 slots) | DERIVED\|A-CAP + calc |
| cap(ρ) table; iteration route competitive only if ρ ≳ 0.72 | calc + DERIVED (independence model) |
| R_A = 0.42; A by literal count | DATA, POST-HOC |
| Transplant affine construction; kept fractions | PROVED + calc; meaning DERIVED\|A-SEP |
| Prior and P's; OC numbers | CONJECTURE |
| §179 is a missed AB precedent (blind decoder → CAPACITY) | DATA |

---

## 15. Risks

1. **The lossy-copy basin (§4.3).** At today's operand fidelity the iteration route cannot beat PASS1. The cell then reads PASS1 + QO-CAPACITY-BOUND, an informative null that names the next lever: copy fidelity, e.g. a pass-1 literal-reconstruction aux, its own derivation.
2. **The open-loop window trains PASS1 first** (QO-2), for 2k steps, and the ramp to 6k is partial. γ = 1 from step 0 would remove that, but it is another change. It is the first suspect if ONSET stays at 1.
3. **No anchor** (QO-5). The autonomous F may drift. STABLE-12/16 are read, and overthinking past K is likelier than in A.
4. **MEAN-ITER's in-distribution claim** is a CONJECTURE (note 204-2's MEAN succeeded on SIM-CoT's decoder). That is why the max over three controls is used.
5. **A-SEP** (transplant) and **probe linearity** (INT-PROGRESS) are separate assumptions, which is why QO-SERIAL needs either one. If both are unavailable or fail, a true serial model reads TWO-STAGE: conservative.
6. **§179's blind-decoder CAPACITY failure** (multiplication) is a standing risk for AB and AB+QO. GSM8K's mostly one-token answer is lighter. The AB draft should cite §179 (Theory's edit).
7. **One seed, two changes vs A.** ACC attribution between AB and QO is by probes only. Stage 2 (≥ 3 seeds plus the AB control) is a separate decision.
8. **The filler string is reused** from the FILLER-N eval. Its eval is flagged (QO-3). A learned filler is out of scope.
