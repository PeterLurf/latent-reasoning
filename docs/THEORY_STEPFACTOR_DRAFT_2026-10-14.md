# STEP-TARGET-FACTOR on GSM8K-Aug: conditional design + derivation DRAFT (2026-10-14)

> **Theory review (2026-10-14 18:10 EDT):** D1 (value tokens memory-free under teacher forcing: val top-1 M0 0.92 / 0.92 / 0.88 vs M_t 0.94 / 0.92 / 0.90 at t 1-3), the plan leak (exact from M_1 0.128 / 0.035 vs M_t 0.147 / 0.045 at t 2 / 3) and D5's code premise (ProsMem builds DeltaMem with add_norm None, prosqa168s2.py:119) verified against theory/stepfactor/diag_A.json and the code. Not registered; it runs only after §202 lands and needs the user's OK (~$3-3.9). Scripts moved from the scratchpad to theory/stepfactor/.

Opus subagent for Theory. **Not registered.** It would run only after §202 (WARM-START) lands, and only with **the user's OK** (a peer's OK is not that). D:\Latent was read only: no repo edit, no git, no score script. CPU scripts and JSON are in this folder (§8.5). Literature should re-check the citations and the novelty marks (§9).

**Tags.**
* **PROVED (code):** by inspection of the cited lines.
* **PROVED:** a full argument given here.
* **DERIVED|A…:** follows from the named assumptions, which are not verified.
* **CONJECTURE:** an argument or a transfer from another setting.
* **DATA:** read from files. **DATA, POST-HOC:** read from landed runs while drafting.
* **calc:** arithmetic done here.

---

## 0. Summary

1. **The string target's loss is operand identity.** In A, 74% of the teacher-forced aux CE sits on operand tokens (DATA, POST-HOC). The value tokens are memory-free: 0.880 top-1 with zero memory against 0.894 (D1, PROVED + DATA). Only 2.2% of string tokens carry serial content (calc).
2. **The plan is parallel.** Operators and pointers are a parse of the question (D2, PROVED). A already reads step t from M_1 at 0.86–0.92 of M_t (the plan leak, measured). Only values on R-chains (48% of steps) force passes.
3. **Pointers for literals, values for results (D3).** The answer decoder sees q, so a pointer (≈ 1.5 bits) replaces the literal (≈ 6 bits). In A, though, 79–86% of operand errors are selection, not precision, so the pointer's precision gain is about +0.08.
4. **The pooled THINK-NECESSARY test checks decoder self-computation, not seriality (D4, calc).** A serial loop whose decoder finishes one step gives K′1/K′10 = 0.51–0.78. On m ≥ 3 the ratio is ≤ 0.38. So Audit's pooled prediction is not supported as central; the m ≥ 3 reading is added.
5. **NOPASS is not leak-closing here.** The non-RMS additive read is a free pass counter ∝ t (D5, PROVED), so Lemma 4's floor does not bind. The replacement tests are a probe and a counter swap.
6. **Design.** WARM-HELPS: F (value from memory + operator + 2 content pointers) vs W, and F-NOPASS vs F, for $2.97 / $3.85 high. Otherwise F vs A, for $2.82 / $3.70.
7. **Central predictions.** Value exact 0.35 / 0.22 / 0.13 by pass (Audit's ≥ 0.5 at pass 3 is not predicted). Accuracy +1.5 (−3 to +6). Pooled THINK-NECESSARY P ≈ 0.5 (W base) / 0.3 (A base); the m ≥ 3 version P ≈ 0.6.

---

## 1. Facts

### 1.1 Code (PROVED (code))
* **F1. The aux decoder sees no question token.**
  - `ProsMem.aux_logits` (boxes/prosqa168s2.py:161–170) builds the prompt `[EOT, " Step", pass_tok(t), " :"]` (:163), then the teacher-forced `step_ids[t-1]`. `fwd_read` runs on that sequence only, with S (:169).
  - `aux_greedy` (:173–177) uses the same prompt (:174) and returns `x[:, 4:]` (:177).
  - gsm8k_stepaux.py calls it after each pass t ≤ min(m, K) (:194–205). The per-item aux CE is a token mean (:203), averaged over the item's passes and then over items (:209). λ = 1 throughout (`--aux_fade 24000,24000`; `lam` :185–186).
* **F2. The pass token is hard-coded; there is no NOPASS flag on GSM8K.**
  - `pass_tok(t)` is the single token " t" (prosqa168s2.py:55–56), hard-coded at :163 and :174. The aux_logits assert (:168) and the `[:, 4:]` slice (:177) assume a 4-token prompt.
  - gsm8k_stepaux.py's argparse (:122–142) has no `--aux_nopass`.
  - The nested version: `InjectN.nopass` (boxes/nested170.py:226); `aux_prompt` (:228–232, asserting no pass token); the flag (:655); set from args (:679).
* **F3. The memory: plain additive reads, memory-independent open-loop writes.**
  - ProsMem builds `DM.DeltaMem(g, 6, "top")` (prosqa168s2.py:119), so `add_norm = None` (deltamem.py:52, :57).
  - The six additive heads' read is the raw φ(q)ᵀS_add (`add_read` :142–155): **not RMS-normalised**. This corrects the brief's "RMS-read heads".
  - Write source: u = (1−γ)·LN_6(h_6) + γ·LN_w(h_w) (:187). h_6 is the residual entering block 6 *before* the read (:278–279), so for γ = 0 the writes do not depend on the memory.
  - FAST_ADD: S_add += Σ_tok φ(k)vᵀ, with no decay and no normaliser (:194–198).
  - Delta heads: S ← αS(I − βkkᵀ) + βvkᵀ (:205–211; WY form `delta_chunk` :226–248).
* **F4. The answer decoder sees the question.** It reads S_K over `[EOT, question, "\n", "### " + answer, EOS]` (prosqa168s2.py:138–144).
  - Eval: greedy after `[EOT] + pre + [ANS]` (gsm8k_stepaux.py:270–275).
  - The think input `[EOT, question]` is identical on every pass (:121–122; the cache_lower assert is at :125).
* **F5. Step strings.** `step_ids = ENC.encode(" <<e_t>>")` (gsm8k_data.py:122–126). BPE splits them into " <<", number tokens, operator tokens, "=", value tokens and ">>", plus the EOT target.
* **F6. Operand grounding (gsm8k_data.parse_recomputable, :168–212).** Each number token in the LHS is one of:
  - **"r":** it equals a previous result (the latest such result);
  - **"q":** a distinct question value, indexed into `qvals` in order of first appearance (:171–177);
  - **"k":** a number-word value or a CONST.

  The item is not recomputable if an operand is ambiguous between r and q or between q and a number word, if an operand is ungrounded, if the answer ≠ the last result, or if the identity render fails. §202's eval split is R = any r operand, Q = otherwise (gsm8k_stepaux.py:409–412).
* **F7. Identical batches across cells.** The main rng is used only for batch indices (gsm8k_stepaux.py:143, :302); transforms use `xr_rng` (:155). So F cells see the batches their comparator saw.

### 1.2 Data (calc; 20k Aug-train items, rng 7, and the 1,319 test items; step_stats.json, calc_bits_serial.json, serial_share.json)

| quantity | train | test |
|---|---|---|
| mean m; tokens per step target (incl. EOT) | 2.61; 8.8 | 3.25; 8.6 |
| token share: format (" <<", ">>", "=", EOT) / operator / operand / value | 45.3 / 12.9 / 26.9 / 14.6% | 46.4 / 12.9 / 26.8 / 13.4% |
| content tokens: operand / operator / value | 49 / 24 / 27% | |
| recomputable (parse_recomputable ok) | 60.3% (200-2: 60.5%) | 43.2% |
| operand grounding on recomputable items, q / r / k | 58.8 / 28.6 / 12.7% | 51.4 / 33.7 / 14.9% |
| equations with an r operand | 48.2% | 55.9% |
| one-BPE-token literals / values; equations with 2 operands | 94% / 88%; 87% | |
| distinct question numbers per item | mean 2.68 (max 11) | 2.96 |
| operator signatures (top 16 cover) | 94 (98.9%; "*" 38%) | |
| step level ℓ = 1 / 2 / 3 / ≥ 4 | 51.8 / 34.7 / 10.7 / 2.8% | |
| item critical path d*: mean; d* = 1 | 2.11; 20.5% | 2.63; **1.9%** |
| test m mix (n_by_m): m ≤ 2 / m ≥ 3 | | 33.4% / 66.6% (n 879) |

* **Step level:** ℓ(t) = 1 + the maximum level of the results that e_t references (level 1 = only literals and constants).
* **Entropies** (plug-in bits per equation; lower bounds for large supports):
  - the first-operand literal token: **6.0** (q 5.6, r 6.6; 369 distinct);
  - **its pointer: 2.46** (question slot 1.5, previous-result index 1.0; mean log₂ |candidates| 2.16);
  - operator signature: 2.6;
  - value: 7.5 (string; 1,500 distinct), 6.8 for its first token.
* **Serial content per target** (serial_share.json, recomputable items):
  - R-operand tokens are 8.1% of string-target tokens;
  - **operand tokens carrying a level ≥ 2 result are 2.2% of all tokens (4.2% of content tokens);**
  - in the value target, **48.2%** of steps are level ≥ 2.

### 1.3 §200 A (DATA, POST-HOC; boxes/runs_fsq/gsm_lat_aug_s1/eval.json)
* **K′ curve:** K′ 0 / 1 / 2 / 4 / 10 / 16 = 0.083 / 0.125 / 0.131 / 0.137 / 0.147 / 0.136.
* **aux exact by pass:** 0.322 / 0.172 / 0.077 / 0.028 / 0.
* **acc_by_m:** m1 0.138 (n 65), m2 **0.241** (357), m3 0.162, m4 0.093, m5 0.058.
* **Other:** path_indep answer_agree 0.458, delta_cos 0.855. fit_fix 0.655 vs fit_rs 0.175 (recall share 0.48). clip_rate 1.0. eval_sec 192.
* **Speed:** train_sec 43,678 s = **1.82 s/step** on A's host, above b286's 1.482.

### 1.4 What A's aux decoder actually fails on (DATA, POST-HOC; diag_A.py → diag_A.json; A's final model.pt, CPU fp32)
* **Fidelity:** greedy aux exact by pass on eval.json's 500 test extras reproduces bf16 within drift: 0.324 / 0.170 / 0.074 / 0.019 vs 0.322 / 0.172 / 0.077 / 0.028.
* **Design numbers are on Aug valid** (500 items, 240 recomputable, 1,573 pass-items). Teacher-forced (TF) unless "exact". The same prompt P_t reads three memories: M_t, M_1 (step t's content already in pass 1?), and M_0 = 0 (the memory-free floor).

| quantity | M_t | M_1 (prompt t) | M_0 |
|---|---|---|---|
| **share of the TF aux CE**: operand / operator / value / format | **74.4% / 14.8% / 6.9% / 2.4%** | | 76.2 / 17.7 / 3.6 / 0.7% |
| TF top-1, value tokens | **0.894** | 0.892 | **0.880** |
| TF top-1, operator tokens | 0.699 (pass 1: 0.761) | 0.660 | 0.344 |
| TF top-1, q-operand / r-operand (first token) | 0.642 / **0.206** | 0.618 / 0.179 | 0.046 / 0.027 |
| same, candidate-restricted (mean \|C\| 3.8 / 4.8) | 0.717 / 0.316 | 0.705 / 0.280 | 0.350 / 0.175 |
| greedy exact, pass 1 / 2 / 3 / 4 | 0.322 / 0.147 / 0.045 / 0.017 | 0.322 / 0.128 / 0.035 / 0.012 | 0 / 0.002 / 0 / 0 |
| t ≥ 2 (M_1 ÷ M_t): exact; operator; q; r | 0.081; 0.671; 0.580; 0.219 | 0.070; 0.616; 0.535; 0.196 (**ratios 0.86 / 0.92 / 0.92 / 0.90**) | 0.001; 0.297; 0.062; 0.042 |

* **Reading (POST-HOC):**
  - **(a) The string's loss is operand identity.** 74% of the TF aux CE sits on operand tokens, and the R-operand (previous result) is the weakest token class at 0.21.
  - **(b) The value tokens are memory-free.** Zero memory reaches 0.880 against 0.894. D1 holds in practice.
  - **(c) Operand errors are mostly selection, not precision.** Restricting the softmax to the item's candidates recovers 21% of q errors and 14% of r errors, so **79–86% of operand errors pick a wrong candidate**. D3(a)'s precision lever is real but bounded here, at about +0.08 / +0.11 top-1 at fixed memory content.
  - **(d) A stores step t's content at pass 1 and indexes it by the pass token.** For t ≥ 2, M_1 with prompt t gives 0.86–0.92 of M_t's performance on every class, including r-operands, well above the M_0 floor.
    - This is the plan leak (L1), measured. It is Lemma 4's "the reader computes the missing reductions itself" (SEEDSTAB:817).
    - It accounts for A's K′1 ≈ K′10. Passes 2+ add about 0.01–0.02 absolute.

---

## 2. Derivation

**D0. Notation.**
* An item is a question q with equations e_t = a_t "=" v_t (t ≤ m) and answer y = v_m. The memory after pass t is M_t; the aux prompt is P_t; the answer decoder is f_ans(q, M_K).
* **Level:** ℓ(t) = 1 + max{ℓ(j) : e_t references result j} (ℓ = 1 if it references none). The item's depth is d* = max_t ℓ(t).
* **A-OP (assumption):** one think pass evaluates at most one operation level.
  - Support (CONJECTURE + DATA): K′0 ≈ the no-CoT level (0.083 vs 0.15–0.19 cited); ProsQA's one hop per pass (§191: K′0 / K′1 at chance at depth ≥ 3); and 2503.07604 (one-pass multi-step arithmetic relies on shortcuts).

**D1. Under teacher forcing the value tokens are memory-free (PROVED).**
* For an exact equation, v_t is a deterministic function of a_t. (Train keeps only exact / round equations; for round ones, add the rounding convention.)
* The string CE at the value positions conditions on the teacher-forced prefix P_t, " <<", a_t, "=". So a predictor that ignores M_t and evaluates a_t in context reaches the Bayes-optimal value CE. At the optimum I(v_t; M_t | prefix) = 0 is admissible, so the value tokens exert no lasting pressure on M_t to hold v_t.
* What the string does demand of M_t is the plan tokens a_t (operator, operand identities) through a query that is item-independent at the first operand (WARMCOT Lemma R).
* **The one exception:** an R-operand at step t+1 is v_t's identity, so M_{t+1} must hold v_t one pass late. That is 8.1% of tokens, and only 2.2% when the referenced result is level ≥ 2 (§1.2).
* **Measured on A** (§1.4, DATA, POST-HOC): value top-1 is 0.880 with zero memory against 0.894 with M_t.

**D2. The plan is parallel; seriality lives only in R-chain values.**
* **(a) PROVED, information statement.** The plan sequence (sig_t, ptr_t)_{t≤m} is a deterministic parse Π(q) of the question under the Aug annotation convention.
  - Pointers name question slots, constants or previous-result *indices*. Indices are not values, so no plan term needs information that first exists after pass 1.
  - A memory with M_1 ⊇ Π(q), plus any pass index, satisfies every plan head at every t.
* **(b) DERIVED|A-OP.** v_t = f_t(literals, v_refs) needs its references' values, so v_t needs ≥ ℓ(t) passes. Of the steps, 48.2% are level ≥ 2 (train), and d* = 1 holds for only 1.9% of test items.
* **Consequence:** a plan target (operator, pointers) can make M_1 hold the plan; it cannot make passes 2..m necessary. A value target makes pass ℓ(t) necessary for v_t.

**D3. Bits, and the division of labour (DERIVED|A-G).**
* **(a) What a target demands of M_t at one operand.**
  - String target: the aux decoder has no question in context, so the read must pick the literal's token out of the number vocabulary (≈ 6.0 bits; 369 distinct first tokens).
  - Content pointer: the read must pick it out of the item's candidates C_t (≈ 1.5–2.2 bits).
  - **A-G:** each wrong candidate's logit is independent noise N(0, σ²) and the right one has margin μ.
  - Then the top-1 error ≈ (N−1)·Φ(−μ/(σ√2)), so the μ/σ needed at a fixed error rate grows like √(2 ln N). The required signal power ratio, string to pointer, is ≈ ln 369 / ln 4.5 ≈ **3.9×**.
  - **This is precision, not selection.** The think pass must still select the operand (Lemma R).
  - **In A, selection binds** (§1.4(c)): 79–86% of operand errors are a wrong candidate. So the pointer's precision gain is bounded at about +0.08 (q) / +0.11 (r) top-1.
  - The larger argument for a candidate-restricted pointer is about the gradient (DERIVED). Under the full softmax most early probability mass sits on non-candidate tokens, so most of the operand gradient goes to suppressing them, which is precision. A restricted softmax spends all of it discriminating among the candidates, i.e. on selection in the think pass: the bottleneck, and the half of the plan the warm start does not supply (WARMCOT P-W0).
* **(b) Downstream.** f_ans attends to q (F4), so a literal named by a pointer can be retrieved by content-addressed attention at no memory cost.
  - A computed result cannot be retrieved from q.
  - So the efficient memory content is **pointers for literals and values for results**: per step about 1.5 bits per Q operand plus ≥ 6.8 bits per result, against ≥ 6 bits per literal token for the string.
  - The factored target asks for exactly this split. The string asks for literal identities, which f_ans never needs, and never for the current result (D1).

**D4. The one-pass shortcut, and when multi-pass use is necessary (DERIVED|A-OP, A-DEC).**
* **The shortcut:** f_ans reaches its accuracy from (q, M_1), i.e. from the question plus a one-pass summary. A shows K′1 = 0.85 × K′10.
* **What is needed instead:** passes beyond 1 are necessary on an item only if I(y; M_K | q, M_1) > 0, i.e. f_ans needs content that only later passes create. The aux target acts only on what M_t holds. It cannot change the decoder's own one-forward capacity.
* **A-DEC (assumption):** f_ans finishes the last s operation levels itself, with q in context, and has a direct path of accuracy a_D.
* **Model:** a strictly serial memory with per-step fidelity ρ, scored on the test m-mix (calc_bits_serial.json):

  | s (levels f_ans finishes itself) | pooled K′1/K′10 (ρ 0.5–0.7; a_D 0–0.083) | THINK-NECESSARY (pooled) | m ≥ 3 stratum K′1/K′10 (ρ 0.6–0.9) |
  |---|---|---|---|
  | 0 | 0.13–0.31 (0.50 at a_D 0.083, ρ 0.5) | passes, except at a_D ≥ 0.083 with ρ ≤ 0.5 | 0.0–0.38 |
  | 1 | **0.51–0.78** | **fails** | 0.0–0.38 |
  | 2 | 0.78–0.93 | fails | 0.46–0.75 |

* **So the pooled rule tests s (how much the decoder computes itself), not seriality.** Items with m ≤ 2 are 33% of the test but carry most of the solvable mass when ρ < 1.
  - A's 0.85 exceeds every s ≤ 1 serial model. Its accuracy comes from the decoder plus a one-pass summary (s ≥ 2), not from the loop.
  - §1.4(d) shows the mechanism on the aux side: step t's content is already in M_1 (0.86–0.92 of M_t), read out through the pass token.
* **What the target can do:** a value target puts v_m = y into M_m for d* ≥ 2 items. Wherever that is more accurate than self-computation, the answer CE gradient moves f_ans toward copying (s ↓).
  - A plan-only target does the opposite: f_ans must execute the plan in one forward pass (s ↑, an "executor" decoder).
  - The THINK-NECESSARY lever is therefore the value term. Even so, s ≥ 1 on easy last steps is the likely equilibrium (CONJECTURE).
  - Hence the added reading **THINK-NECESSARY-DEEP** on m ≥ 3 (§5.4).

**D5. A free pass counter in this arm (PROVED code + algebra).**
* For γ = 0 the writes do not depend on the memory (F3), so S_add(t) = t·A(x) exactly (FAST_ADD sum, no decay). The read is then φ(q)ᵀS_add(t) = t·r(q, x).
  - This is THEORY_SEEDSTAB_2026-10-13.md §5.1(iii), l.452–455: "the none read is ∝ t (a counter)". Under rms / den it is t-invariant.
* For γ > 0 the additive read is a sum of the pass writes' values with positive weights φ(q)ᵀφ(k_tok) > 0, so it still grows ~t unless the values cancel.
* The injected vector x + fade·W_o·read changes direction relative to the residual as t grows, so later blocks can decode t.
* **So every decoder has the pass index with or without the pass token.** Under RMS (not this arm) only the geometric delta transient Δ*Pᵗ would carry it (SEEDSTAB §5.1(iv)).

**D6. Relation to ProsQA (DATA + CONJECTURE).**
* **Correction to the brief:** ProsQA step strings are 8–11 tokens, not one (" Every sc rom p us is a rem p us ."). Each pass does add only **one new concept**: 38 names with 32 distinct first tokens, ≤ 5.2 bits. The subject repeats the previous object.
* **The property that mattered:** the step content *is the chain state*, node_t = next(node_{t−1}).
  - Every step has ℓ(t) = t (a pure chain).
  - The minimum depth, 3, is beyond the decoder's one-forward reach, which gives K′0 / K′1 at chance (§191).
* **On GSM8K the string's content is mostly level 1** (the plan, plus level-1 literals), and the decoder can finish ≥ 1 step itself.
* **The GSM8K analogue of ProsQA's targets is the value (state) target, not the plan target.**

**D7. Why factoring still helps aux exactness (DERIVED|WARMCOT A1 + Lemma S).**
* Within a pass, the string's exactness is a product: selection × identity routing × the token sequence.
* Separate heads (operator, pointer, value) give each factor its own first-order CE. By Lemma S, factors trained separately leave the slow product.
* This predicts higher plan-head accuracy than the string's plan-token exactness. **It implies nothing about serial use (D2).**

---

## 3. Goodhart and leak analysis

* **L1. Plan leak through the pass token (possible by D2(a)).**
  - **PLAN-LEAK(t)** = plan-head accuracy from M_1 with prompt P_t, divided by the same from M_t (t = 2, 3).
  - The prediction is **PLAN-PARALLEL** (ratio ≥ 0.8). For plan heads this is the derivation's expectation, not a failure.
  - **A already reads 0.86–0.92** on the string target (§1.4(d), DATA, POST-HOC), including r-operands whose referenced result is pass-1 computable.
* **L2. Value leak.**
  - **VALUE-LEAK(t)** = value exact from M_1 with prompt P_t, on R-steps with ℓ(t) ≥ 2. One-pass multi-level computation bounds it, so it is predicted at the M_0 floor + ≤ 0.05.
  - If it is near M_t's value, pass 1 computed the chain, which is the one-pass shortcut.
* **L3. Recall dilutes the serial pressure (CONJECTURE + DATA).**
  - On a memorised train item, every v_t is one lookup, so M_1 can hold all of them. A's recall share is 0.48; W's is predicted at 0.37 (P-W6).
  - So the value target's serial pressure acts on about the unmemorised half of training items.
  - Test readings are unaffected (unseen items), but the training signal is diluted. R_x (Theorem X) would fix this; it is a second change and is not stacked.
  - **Detect:** VALUE-LEAK on 500 fit_fix train items against test items.
* **L4. NOPASS (the leak-closing arm, Audit). What it closes and what it does not.**
  - **Removes:** the prompt index. P is identical across passes.
  - **Does not remove:** the index itself. The additive counter re-supplies it at zero learning cost (D5), so plan-at-pass-1 + counter is reachable from step 0.
  - **Is it cheaper than serial computation?** For plan content, the serial solution *is* a counter plus the parse, because the plan has no serial dependency (D2), so they coincide. For level ≥ 2 values the counter does not help: the content is absent at pass 1.
  - So F-NOPASS is predicted ≈ F on the serial readings (DERIVED|D5, A-OP). The cell's value is to test that claim.
  - **Lemma 4** (THEORY_SEEDSTAB:808–818, PROVED; the derivation behind NOPASS): without the token, two passes with indistinguishable states and different targets pay ≥ 2 ln 2 nats per differing token.
    - Its premise is a contraction (L < 1, indistinguishable states).
    - The additive map A ↦ A + a is a translation (Lipschitz 1), and it keeps passes distinguishable.
    - **So the floor does not bind in this arm** (DERIVED). It would bind under RMS reads, a separate architecture change (F3).
  - **Evidence grading:**
    - "4/4 NOPASS models in the converged class vs 6/11 with the token" is **DATA, POST-HOC, P = 0.15** (THEORY.md:14358; SEEDSTAB:692) and carries no reading.
    - §187's depth claim **FAILED at 2/2** under a basin-entry confound (THEORY.md:13747).
  - **Diagnostics under NOPASS.** The M_1-vs-M_t test is undefined there, because reading M_1 always means "the current step". It is kept for the pass-token arms. Replacements:
    1. **PROBE-LEAK(t).** An L2-logistic probe for step t's targets: the value's first token (top 200 + OTHER), the operator signature, and pointer 1.
       - Features: the 768-d pre-W_o read vector [g_add·φ(q)ᵀS_add,h ; g_del·S_del,h·q̂_h] at the last token of the fixed prompt [EOT, " Step", " :"]. It is a fixed linear map of S, so it is arm-agnostic and applies to all cells and the comparator.
       - **Why these features (PROVED (code), Lemma R extended):** every F term is predicted from h_last. h_last's only memory input is the read at the ≤ 4 prompt positions, and those positions' queries are item-independent (prosqa168s2:163, deltamem read_fixed :158–176).
       - So all the decoder can use is a fixed linear image of S_t. The probe is therefore (up to the pass token's own query) the decoder's information bottleneck, not a stronger reader.
       - Separate probes are fit on M_1 and on M_t, on 4,000 Aug-train items (rng fixed), with λ chosen on an inner split. They are reported on Aug valid, with test descriptive.
       - Reading: **PROBE-LEAK = acc(M_1)/acc(M_t) on ℓ ≥ 2 R-steps.**
    2. **COUNTER-SWAP.** Decode with the trained aux decoder from:
       - M̃_t = (t·S_add(M_1), S_del(M_1)): pass-1 content plus a synthetic magnitude counter;
       - the head-type swaps (S_add(M_t), S_del(M_1)) and (S_add(M_1), S_del(M_t)).

       The plan + counter account predicts plan heads from M̃_t ≥ 0.7 × from M_t. Serial values predict value exact on ℓ ≥ 2 steps from M̃_t ≤ the M_0 floor + 0.05.
* **L5. Heads satisfied by priors.** The operator majority ("*", 0.38), the pointer "first number" rule (0.645 on held-out, WARMCOT probe), and the value marginal. Every head reading is reported against the M_0 (zero-memory) floor and these rules.
* **L6. New parameters and C1 (PROVED (code), conditional on the §8 contract).**
  - The new parameters: W_op (768 → 17), and W_p1, W_p2 (768 → 768, init identity); about 1.19M.
  - They read only the aux decoder's h_last. They are not on the eval path (`answers()` never calls them), and they add no cross-pass variable.
  - The think input is unchanged and identical every pass. The aux decoder stays read-only and is dropped at test.
  - The question's numbers enter only the pointer loss's normaliser (candidate set), never a decoder input.
  - **C1's three clauses hold:** an identical input, memory as the only cross-pass channel, and an aux that is read-only and dropped at test.

---

## 4. Gradient flow

| term | parameters reached | notes |
|---|---|---|
| L_ans (unchanged) | LM head / wte, trunk, read params (mem.ln, qkv q-rows, s_add, s_del, W_o), S_K → write params and think passes | as §200 A |
| L_val,t (value from M_t) | LM head / wte (tied); trunk blocks 0–11 on the aux sequence; read params; S_t → write params (qkv k / v rows, gate, mem.ln; ln_w for γ > 0); S_{t−1}..S_1 (cross-pass); think-pass activations (h_6 via (1−γ)LN_6 incl. cached blocks 0–5; h_w for γ > 0; the think pass's own read at block 6 = the closed loop) | |
| L_op,t | W_op, plus everything upstream of h_last (as L_val except the LM head) | |
| L_ptr,j,t | W_pj, plus upstream of h_last. **wte is detached in the candidate scorer**: no gradient to embeddings from pointers | |

* **Lemma O (WARMCOT, PROVED (code)) holds for every new term:**
  - ∂L_aux/∂S = 0 while fade·W_o = 0;
  - no aux gradient reaches the think pass's top while γ = 0.
* **Smoke assertions (fp32, CPU; the scratch/gradflow_check.py pattern; one backward per term with zero_grad between).**
  - Setup: fade 1, γ 0.5, W_o perturbed by N(0, 1e-3); 4 items with m ≥ 3; terms at t ∈ {1, 3}; S_1 with retain_grad.
  - **L_val (t = 3):** ‖∂/∂S_1‖ > 0, mem.qkv.weight[D:3D].grad ≠ 0, mem.wo.weight.grad ≠ 0.
  - **L_op (t = 3):** W_op.weight.grad ≠ 0 and ‖∂/∂S_1‖ > 0.
  - **L_ptr,1 / L_ptr,2 (t = 3):** W_pj.weight.grad ≠ 0, ‖∂/∂S_1‖ > 0, and the wte grad from this term == 0 (the detach check).
  - **L_ans:** ‖∂/∂S_K‖ > 0, and W_op / W_pj grads == 0 (aux-only heads).
  - **Decorator check (ast):** the new training path is not under a `@torch.no_grad` def. `aux_greedy` is one, which is the insert-above-decorated-def lesson.

---

## 5. Design (conditional on §202)

### 5.1 Branches

| §202 reading (W − A, registered) | base init | cells | comparator |
|---|---|---|---|
| **WARM-HELPS** (≥ +2.8) | C's model.pt via `--init_from` (W's recipe, 12k) | **F-P** and **F-N** | F-P vs **W**; F-N vs F-P |
| NEUTRAL or WARM-HURTS | raw GPT-2 (A's recipe, 24k) | **F-P** only | F-P vs **A** |
| §202 voided / not landed | — | nothing runs | — |

* **If W passes the pooled THINK-NECESSARY itself** (P-W2), STEP-TARGET-FACTOR's shortcut motive is moot for the pooled rule.
  - F-P is then re-scoped as an accuracy and serial-depth lever: DEEP readings first, one cell, re-approval.
* Each comparator is an existing run with the same init, steps, flags, seed 1, RTX 3090 class and batch sequence (F7).
  - It needs only an **eval-only re-run** for the new eval fields: same --eval_bs 64, a 3090. The registered fields must match its eval.json bitwise (the bf16 batch-size rule).

### 5.2 Cells (seed 1; RTX 3090; GPU and driver recorded; bs 64, lr 1e-4, warm-up 100 + cosine; the base's flags otherwise exactly)
* **F-P (factor, pass token kept):** `--aux_target factor`. At each pass t ≤ min(m, K), all terms read M_t through the aux decoder with P_t = [EOT, " Step", " t", " :"]. **No expression in context.**
  1. **Value:** LM-head CE on " " + v_t (the RHS token string as written) + EOT, teacher-forced within the value only, right after P_t.
  2. **Operator:** W_op on h_last (the final hidden at P_t's last position), 17 classes: the top-16 train signatures (98.9%) + OTHER.
  3. **Pointers j = 1, 2:** logits ⟨W_pj·h_last, wte[c]⟩ (wte detached) over C_t = {first tokens of the question's literals} ∪ {first tokens of previous results} ∪ {the target token}; the label is the operand's first token.
     - Pointers are defined on recomputable items only (60.5%) and masked otherwise. The value is defined wherever the RHS is a number token.
  - **Aggregation:** L_aux,t = the mean of the terms present, so there is one mean CE per pass as in the string arm. λ = 1 as the base.
* **F-N (factor, NOPASS):** F-P plus `--aux_nopass`, with P = [EOT, " Step", " :"] for every term, in training and in every aux eval.
* **Not run:**
  - **String × NOPASS** (§187's question on GSM8K): the 2×2's fourth corner. The derivation gives NOPASS a role only once the target has serial content, and the counter neutralises it here (L4). A third W-base cell would also cross $4 (§7).
  - **V (value only, zero new parameters):** the minimal alternative, and the substitute second cell if Audit prefers isolating the plan heads (F − V) over NOPASS.
  - **WINDOW / halting:** not stacked, for three reasons:
    - one change per comparison;
    - idle passes are harmless in A (IDLE-PASSES: K′ = m+2 never above K′10; DATA);
    - a 2×2 with the K schedule costs ≥ $5.6.
    The value target does raise a hold question (no target for t > m), which IDLE-PASSES and STABLE-12 / 16 watch. If F shows idle harm, the follow-up is HOLD (target v_m for t > m), the natural complement of a state target, as its own decision.

### 5.3 Checkpoint rule
* The **FINAL** checkpoint is used for every reading. The Aug-valid-selected checkpoint is reported beside it, labelled.
* **The test is never used for selection.** Design numbers here come from Aug valid only (§1.4), except the fidelity check.

### 5.4 Readings (paired on the 1,319 test items)
* **Thresholds:** T = 0.028 (2 SE, pooled); T_D = 0.034 (= T·√(1319/879), the m ≥ 3 stratum).
* **ACC:** F − comparator reads LIFTS (≥ +T) / MATCHES / HURTS (≤ −T). F-N − F-P is read the same way.
* **LOOP-FORMED (per cell):** K′10 − K′0 ≥ T.
* **THINK-NECESSARY (pooled, as registered):** K′0 ≤ 0.5·K′10 and K′1 ≤ 0.5·K′10. **SHORTCUT-RATIO** = K′1/K′10 is reported per cell, with F's minus the comparator's.
* **THINK-NECESSARY-DEEP (new):** the same rule on m ≥ 3 items (n 879), from a new eval field `acc_by_m_kprime`.
  - **SERIAL-GAIN-DEEP** = (K′10 − K′1) on m ≥ 3, as F minus the comparator: MORE-SERIAL if ≥ T_D.
* **VALUE-EXACT(t):** greedy value from M_t, by pass and by R / Q grounding.
  - The comparator's like-for-like proxy is the value tokens right inside its own greedy string (VALUE-IN-STRING).
* **PLAN-ACC(t):** operator-head and pointer accuracies by pass. The comparator's proxies are the signature and the first-operand token in its greedy string.
* **Leak diagnostics:**
  - PLAN-LEAK and VALUE-LEAK for the pass-token cells and the comparator (M_1 with P_t);
  - PROBE-LEAK for all;
  - COUNTER-SWAP for all, read primarily for F-N;
  - every one against the M_0 floor.
* **Descriptive:** acc, acc_by_m, aux_exact_by_grounding, IDLE-PASSES, STABLE-12 / 16, path_indep, recall share, the valid curve, entry(0.5 / 0.1), clip rate and the per-term aux CE logs.

---

## 6. Predictions and falsifiers (one seed; central values; W base unless stated)

| id | prediction | from | falsified if |
|---|---|---|---|
| P-F1 | PLAN-ACC(1) ≥ the comparator's proxy + 0.05 for pointer 1 (central +0.08, the precision bound of §1.4(c), plus the selection-focused gradient of D3), and ≥ proxy + 0.00 for the operator (central +0.03) | D3, D7 | pointer 1 < proxy + 0.02 (factoring buys no operand exactness) |
| P-F2 | **PLAN-PARALLEL:** PLAN-LEAK(2), PLAN-LEAK(3) ≥ 0.8 (F-P; A reads 0.86–0.92 post hoc); PROBE-LEAK on plan targets ≥ 0.8 (both cells) | D2(a) | < 0.5 (the plan is computed serially: D2's parse premise fails in practice) |
| P-F3 | **VALUE-SERIAL:** on ℓ ≥ 2 R-steps at t = 2–3, VALUE-EXACT from M_t ≥ M_0 floor + 0.10, and VALUE-LEAK ≤ floor + 0.05 | D2(b), A-OP | VALUE-LEAK ≥ 0.7 × VALUE-EXACT from M_t (a one-pass chain: A-OP fails) |
| P-F4 | VALUE-EXACT by pass ≈ 0.35 / 0.22 / 0.13 (range at pass 3: 0.05–0.25); A base 0.30 / 0.17 / 0.09 | calc. Pass 1 ≈ A's plan-right rate (string exact 0.32 ÷ in-context value 0.94 ≈ 0.34) × latent arithmetic (0.8–0.9). Later passes: × the R-identity rate (A: 0.21–0.25 from M_t; F trains it directly) | — (central only) |
| P-F4a (**Audit's**, registered beside) | VALUE-EXACT(3) ≥ 0.5 | Audit's lead | < 0.5. **This derivation predicts the falsification (P ≈ 0.05).** Audit's 0.08 baseline is the full-string exact; the TF value token is already 0.89 and memory-free (§1.4(b)), so the two are not comparable |
| P-F5 | SHORTCUT-RATIO(F) ≤ SHORTCUT-RATIO(comparator) − 0.10 (central −0.2) | D4 (s ↓ under the value target) | F's ratio ≥ the comparator's − 0.03 |
| P-F5a (**Audit's**) | pooled THINK-NECESSARY passes for F | Audit's lead | K′1 > 0.5·K′10. The derivation gives P ≈ 0.5 (W base) / 0.3 (A base): **not supported as a central outcome** |
| P-F6 | THINK-NECESSARY-DEEP passes for F (central m ≥ 3 ratio 0.40), and MORE-SERIAL vs the comparator | D4 | F's deep ratio > 0.5 (the value target did not force serial use where D2 / D4 say it must) |
| P-F7 | ACC: MATCHES or LIFTS; central +1.5 (range −3 to +6) | D3(b), D4 | HURTS (≤ −2.8) |
| P-F8 | F-N ≈ F-P: ACC within ±T, \|Δ deep ratio\| ≤ 0.10, and COUNTER-SWAP plan heads from M̃_t ≥ 0.7 × from M_t | D5, L4 | F-N − F-P ≥ +T with deep ratio −0.10 or more (then NOPASS mattered: the counter was not used) |
| P-F9 (descriptive) | VALUE-EXACT is ≥ 0.10 lower on R-steps than on Q-steps | D2(b) | the gap is ≤ 0.02 |

**No derivation predicts SOTA** (budget-limited, as 200-4). A one-seed reading is worded "on seed 1".

---

## 7. Cost (calc; 1.482 s/step at micro 4 (b286); $0.255/h (§200 host class, WARMCOT 5.7); + $0.10 per host; eval incl. new fields ≈ 0.2 h; high = × 1.35; A's realised 1.82 s/step shown beside)

| item | hours (central / high) | $ central / high / at A's 1.82 s/step |
|---|---|---|
| one 12k cell (W base) | 5.1 / 6.9 | **1.41 / 1.85 / 1.70** |
| one 24k cell (A base) | 10.1 / 13.5 | **2.67 / 3.55 / 3.25** |
| comparator eval-only re-run (3090; A's eval was 192 s) | ≈ 0.2 | 0.15 |
| gates (CPU / local smokes; optional `--time_steps 200` on the cell's host) | — | ≈ 0 (+0.03) |

* **W base (F-P + F-N + W eval-only): $2.97 central / $3.85 high / $3.55 at A's rate.** That is under $4, with only $0.15 of margin at high.
* **A base (F-P + A eval-only): $2.82 / $3.70 / $3.40.**
* **Flags (> $4):**
  - adding the 2×2's fourth W-base cell: $4.38 / $5.70;
  - two A-base cells: $5.49 / $7.25.
  Neither is proposed.
* The value target's aux sequences are shorter than the string's, so s/step should be ≤ the base's; reusing b286 is conservative. Realised $ replaces the projection per cell at landing.

---

## 8. Engineering contract and gates (Experiments implements; Audit reviews the diff; every new parameter optional; no edit while a running job lazily imports these modules)

### 8.1 boxes/prosqa168s2.py (ProsMem, shared with §168s2 / §191: the default path must stay byte-for-byte equivalent)
* `self.nopass = False` in `__init__`.
* A helper `aux_prompt(t)`: `[EOT, STEP_ID, COLON_ID]` if `nopass` (assert no pass token, as nested170:229–231), else `[EOT, STEP_ID, pass_tok(t), COLON_ID]`.
* `aux_logits`: `pre = self.aux_prompt(t)`; the assert compares `x[:, :len(pre)]`.
* `aux_greedy`: the prompt from `aux_prompt(t)`; returns `x[:, len(pre):]`.
* **Alternative with zero risk to ProsMem:** a subclass `FactorMem(ProsMem)` in a new boxes/gsm8k_factor.py, holding nopass and the factor heads. gsm8k_stepaux constructs it only when a new flag is on.

### 8.2 boxes/gsm8k_stepaux.py: new flags
* `--aux_target {string, value, factor}` (default `string` = today's path):
  - `value`: L_val only.
  - `factor`: L_val + L_op + L_ptr,1 + L_ptr,2, mean of the present terms.
* `--aux_nopass`: sets `m.nopass`. It applies to aux_logits, aux_greedy, the factor terms and every aux eval.
* `--op_vocab <json>`: the frozen top-16 signature list, computed once from the full filtered train set (counts and tie order recorded). It is written to the out dir and its sha256 to args.json.
* **Heads:** constructed only when `--aux_target factor`, with a dedicated torch generator (seed + 30000) so that the global RNG stream is untouched.

### 8.3 Target construction (a pure-data function in gsm8k_data, unit-tested)
* **Value:** rhs = e.split("=")[1].strip(), which must match `[\d,]*\.?\d+` (as :186). ids = ENC.encode(" " + rhs) + [EOT]. Otherwise the step's value term is masked; the masked rate is logged.
* **Operator signature:** "".join(ch for ch in lhs if ch in "+-*/%()"). Class = its index in op_vocab, else OTHER.
* **Pointers:** from `parse_recomputable` (non-recomputable items are masked).
  - Operand j ∈ {0, 1} of `eqt["ops"]` in LHS order.
  - Target = ENC.encode(operand token)[0].
  - C_t = sorted set of {ENC.encode(tok)[0], ENC.encode(tok.replace(",", ""))[0] for each question literal in `qn`} ∪ {ENC.encode(rtok_j)[0] for j < t} ∪ {target}.
  - Log the collision rate (two different values sharing a first token).

### 8.4 eval.json additions (additive; registered fields unchanged)
* `acc_by_m_kprime` (on every stepaux run): per-item correctness at each K′, stratified by m.
* `aux_value_exact_by_pass[_by_grounding]` for value / factor; `aux_value_in_string_by_pass` for string.
* `aux_op_acc_by_pass`, `aux_ptr_acc_by_pass` (factor), and the string proxies.
* `*_from_M1_by_pass` (pass-token arms), `counter_swap` (all), and `*_M0_floor`. All are on eval.json's 500 test extras, plus Aug valid.
* **PROBE-LEAK** is a separate eval-only module (boxes/stepfactor_probe.py) on model.pt, so no lazy import of it in training.

### 8.5 Gates (before launch)
1. **Plain path bitwise:** without the new flags, for a fixed batch (fp32 CPU, then the GPU):
   - `loss_items` equals the pre-edit code's loss bitwise;
   - `aux_logits` logits and `aux_greedy` outputs are identical in both aux paths;
   - the model state_dict keys are unchanged.
   Also construct + 1 step with no new flags (the plain-path smoke lesson).
2. **NOPASS:** an assert that the prompt holds no pass token, in training and in every aux eval. With the flag off, the result is identical to gate 1.
3. **Targets:** on 10k train items:
   - each value label reproduces the RHS;
   - each pointer label reproduces its operand's value (q → qvals, r → that result, k → the constant);
   - report coverage (value, pointer, signature OTHER) and the collision rate.
4. **Gradflow per term** (§4).
5. **Eval-path identity:** with the heads deleted after training (or never called), `answers()` is identical. The K′ grid and test items are unchanged (200-1).
6. **mem_check** at the longest item with `--aux_target factor`, which must stay at micro 4.
7. **Fixtures:** the scorer reads every new field by name on a synthetic eval.json (the scorer field-path rule).

Scripts in this folder:
* step_stats.py → step_stats.json;
* calc_bits_serial.py → calc_bits_serial.json;
* serial_share.py → serial_share.json;
* diag_A.py → diag_A.json.

All are CPU, ≤ 3 threads, Idle priority.

---

## 9. Prior art (IDs verified on the arXiv abstract pages, 2026-10-14)

| work | what it does | relation | novelty of ours vs it |
|---|---|---|---|
| Coconut, Hao et al. **2412.06769** | continuous thoughts by a CoT-removal curriculum | the init convention (WARM) | — |
| CODI, Shen et al. **2502.21074** | self-distils the CoT teacher's hidden state at the answer token | a latent-arithmetic lever, not a per-step target | — |
| iCoT-SI, Deng et al. **2405.14838** | internalises CoT by removing tokens step by step | no per-pass latent target | — |
| Implicit CoT via KD, Deng et al. **2311.01460** | distils teacher hidden states across layers | intermediate supervision, not factored | — |
| PCCoT, Wu et al. **2506.18582** | Jacobi-parallel latent thoughts (CODI family) | the SOTA bar 49.48 | — |
| **SIM-CoT**, Wei et al. **2509.20317** | "an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step", removed at inference (abstract) | **the closest:** our string aux is this design | replacing the step **string** by (value from memory; operator; content pointers): **PARTLY KNOWN** |
| **MathQA**, Amini et al. **1905.13319** | operation programs whose arguments are question numbers (n_i), previous results (#k) and constants | **the factored representation itself, for explicit program generation** | as a per-pass latent target: PARTLY KNOWN |
| Pointer Networks, Vinyals et al. **1506.03134** | output = a pointer to an input position | the pointer mechanism | KNOWN (mechanism) |
| Physics of LMs 2.1, Ye et al. **2407.20311** | probes GPT-2's hidden reasoning on GSM-like data | related to PLAN-PARALLEL (Literature to confirm the exact probe claims before we lean on them) | — |
| Implicit reasoning via shortcuts **2503.07604** | one-pass multi-step arithmetic relies on shortcuts | supports A-OP and the one-pass shortcut | — |
| Reasoning by superposition **2505.12514** | continuous thoughts as a superposed BFS frontier (graph reachability) | ProsQA's state-target reading | — |
| Looped transformers, Saunshi et al. **2502.17416** | k-step reasoning by about k loops | the A-OP analogue | — |

**What is NEW (as far as checked; Literature to confirm):**
* D1: the string target's value tokens are memory-free under teacher forcing, so a per-step aux string never asks the memory for the current result.
* D4: the pooled removal test fails for a perfectly serial loop under GSM8K's m-mix when the decoder finishes a step; the m ≥ 3 test.
* D5 / L4: NOPASS is neutralised by a non-normalised additive counter; the counter-swap and probe-leak diagnostics.
* The division "pointers for literals, values for results" (D3(b)) is a NEW framing of KNOWN parts (MathQA's argument types and the attention-dereference argument).

---

## 10. Claim status

| claim | status |
|---|---|
| F1–F7 (aux sees no question; hard-coded pass token; non-RMS additive read; open-loop writes memory-independent; answer decoder sees q; grounding; identical batches) | PROVED (code) |
| Token shares, levels, entropies, serial shares, test m-mix | DATA (calc) |
| §1.4 diagnostics on A | DATA, POST-HOC |
| D1 value tokens memory-free under TF | PROVED (exact equations; + rounding convention for round ones) |
| D2(a) the plan is a parse of q (parallel) | PROVED (information statement; capacity is empirical) |
| D2(b) v_t needs ≥ ℓ(t) passes | DERIVED\|A-OP |
| D3(a) 3.9× SNR ratio string vs pointer | DERIVED\|A-G; its realised gain in A is bounded at about +0.08 / +0.11 (DATA, POST-HOC: selection dominates) |
| restricted-softmax gradient focuses on selection | DERIVED (softmax algebra) |
| A stores step content at pass 1 (PLAN-LEAK 0.86–0.92); operand CE 74%; value memory-free | DATA, POST-HOC (Aug valid, CPU fp32) |
| D3(b) pointers for literals, values for results | DERIVED (attention dereference; F4) |
| D4 pooled THINK-NECESSARY tests s, not seriality; grid | DERIVED\|A-OP, A-DEC + calc |
| D4 value target lowers s; plan target raises it | CONJECTURE (gradient-preference argument) |
| D5 free additive counter | PROVED (code + algebra) for γ = 0; DERIVED for γ > 0 |
| L4 Lemma 4's floor does not bind here | DERIVED (premise fails: translation, not contraction) |
| L4 F-N ≈ F-P | DERIVED\|D5, A-OP |
| D6 ProsQA = state target | DATA + CONJECTURE |
| D7 factoring helps exactness | DERIVED\|WARMCOT A1 + Lemma S |
| L6 C1 preserved | PROVED (code), conditional on the §8 contract |
| P-F4 central values | CONJECTURE + calc |
| §187 NOPASS 4/4 vs 6/11 | DATA, POST-HOC, P = 0.15: no reading |

---

## 11. Risks
1. **The value target is the hardest latent ask.** It needs latent arithmetic, which neither init supplies (WARMCOT D4). If VALUE-EXACT(1) is low (< 0.2), F's serial readings are uninformative.
   - A's r-operand top-1 (0.21 from M_t) is the closest existing anchor for a routed result.
   - **Mitigation:** the plan heads (D7). The falsifiers are stated relative to the M_0 floor.
2. **Selection, not precision, is A's operand bottleneck** (§1.4(c)). If the restricted-softmax gradient does not improve selection, F's plan heads add little (P-F1's falsifier).
3. **Recall dilutes the serial pressure (L3).** About half of training items are memorised under either base. Readings are on test; VALUE-LEAK on train items is reported.
4. **Hold:** there is no target after m, and the state could drift over idle passes. IDLE-PASSES and STABLE watch this; HOLD is a follow-up decision.
5. **Pointer coverage is 60.5%** (recomputable items only), and 43% of test items are recomputable. Pointer readings are on recomputable items and labelled.
6. **The loss scale changes** (mean of up to four terms vs one token-mean CE), and the clip rate is already 1.0 in A. The interaction with the global-norm clip (199-A5 crowding) is logged per term, not fixed.
7. **Single seed, one GPU class** (3090), and the comparators ran on other 3090 hosts. Same-class pairing is accepted; it is not bitwise.
8. **Post-hoc:** A's eval.json and §1.4's diagnostic were read while drafting, and P-F4 / P-F7 were set with that knowledge. **§202's outcome is not assumed**: the branches are fixed in §5.1.
9. **Shared-module edit:** if ProsMem is edited rather than subclassed, a running §202 / §191 eval could lazily import the new code. Use the subclass, or edit only when no job imports prosqa168s2.
