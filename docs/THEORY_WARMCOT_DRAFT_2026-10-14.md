# WARM-START-FROM-COT / JOINT-COT: conditional registration DRAFT (2026-10-14)

Opus subagent for Theory, reviewed by Theory. **Not registered:** it registers only if §200 A / B read LOOP-NOT-FORMED or below at landing. WARM pair user-pre-approved ≤ $4 via Audit; JOINT-COT is not pre-approved. Probe scripts are in theory/warmcot/. Literature to verify citations.

---

# WARM-START-FROM-COT and JOINT-COT: two CoT-supervised initialisation routes for §200's latent cell (CONDITIONAL registration DRAFT, 2026-10-14; Opus subagent for Theory; not registered)

**Status.**
* Drafted by a Theory subagent. D:\Latent was read only; no GPU, no launch. CPU probes at ≤ 3 threads, Idle priority (§1.4; scripts and JSON in this folder, §8).
* **Conditional.** Opens only if §200 A or B lands reading **LOOP-NOT-FORMED**, or R1 reads **below** (A − C ≤ −2.8 or B − D ≤ −2.8). If A forms its loop and matches C, nothing here is triggered.
* **Launch gate.** This is new GSM8K-scale spend ($1.7–8.6, §6.7). It needs **the user's explicit approval**; a peer's OK is not that.
* **Inputs folded in:** the brief (WARM-START, with a resampled pair from D); the coordinator's addition (cover **both** routes: WARM-START, Coconut-style, and JOINT-COT, CODI-style); docs/LIT_LATENT_INIT_STAGING_2026-10-14.md (95a7fb15).

**Tags.**
* **PROVED:** a full proof under the stated assumptions. For "PROVED (code)" the assumption is the harness as committed, with the lines cited.
* **DERIVED|A…:** follows from the named assumptions, which are not verified.
* **CONJECTURE:** an argument or a transfer from another setting.
* **DATA:** read from existing files. **(post hoc)** marks numbers seen while drafting.
* **calc:** arithmetic done here.

---

## 0. Summary

1. **The two published routes.** Every published GPT-2 GSM8K-Aug latent number has explicit-CoT supervision, either before latent training (Coconut family: a CoT-tuned start, about 6 epochs, about 40% valid) or alongside it (CODI family: a joint CoT teacher CE from raw GPT-2 for all 40 epochs). Both ablations are large: 31.1 vs 22.3 without Coconut's stage 0, and 43.7 vs 27.1 for CODI with a static teacher. §200 A / B use neither. These arms add each route to §200 A's exact recipe. **This matches the field's initialisation; it is not a new method.** What it buys is a like-for-like comparison.
2. **Mechanism (§2).** Split the latent step task into (i) arithmetic, (ii) the plan (word problem → next expression), and (iii) routing through the memory.
   * **Two harness facts (PROVED, code).** The aux decoder's read query is item-independent where it must emit the first operand (Lemma R). And the think pass's upper blocks get exactly zero aux gradient until γ > 0 at 2k, with W_o = 0 at step 0 (Lemma O).
   * **So the plan must be computed in the think pass and written,** and its gradient is the product (plan present) × (route present).
   * **From raw GPT-2, both factors start near 0.** A product of small factors escapes slowly (Lemma S, PROVED: a polynomial time in 1/ε for N ≥ 3 small factors, logarithmic for N = 2). Meanwhile the direct answer path and the plan-free "bag of literals" channel grow at first order.
   * In SEEDSTAB Part II's selection law that is b_P ≈ 0, which makes LOOP-NOT-FORMED the modal outcome (DERIVED|A1–A4).
   * **From the CoT trunk, (ii) is present at step 0** (probe, §1.4), so the plan-specific routing gradient is first order as soon as γ·fade > 0. That is a **tilt**: μ_Z grows linearly in the plan content π, not a √-lever.
   * **JOINT-COT** supplies π(t) on C's own learning curve. Its tilt, relative to a WARM cell that forgets nothing, is **0.63** through the decision window (2–6k) and 0.75 overall (calc). So JOINT out-tilts WARM only if WARM keeps less than about 63–75% of its CoT competence through 6k. A new descriptive, COT-RETAIN, measures exactly that.
3. **C1 is preserved by construction in both arms** (PROVED, code). The eval path is §200 A's: identical think input, S the only cross-pass variable, and no CoT text at test. JOINT's CoT CE is training-only. K′0 is the empty-memory trunk (the read adds only W_o's constant bias), so it measures the trunk's direct, no-CoT ability. K′1 catches a one-shot internalised computation.
4. **Predictions (§3)**, at one seed:
   * WARM forms the loop (K′10 − K′0 ≥ 2.8; central gap about 20 points) with THINK-NECESSARY.
   * Final test 0.24–0.35, central **0.28–0.30** (calc: C's per-m accuracy × a per-pass routing fidelity ρ^m, with ρ 0.95 → 0.280). Reaching C's 0.321 within 12k is **not** predicted (P ≈ 0.3).
   * Against an equal-exposure text control C+ (C continued 12k), the reading is matches or below, never above.
   * C's memorisation (recall share 0.42) **transfers**: WARM's recall share ≥ 0.20, central 0.35. The resampled pair lowers it by ≥ 0.10 without moving test (RS-NEUTRAL).
   * JOINT forms the loop with lower confidence; JOINT − A ≥ +2.8 and JOINT vs C reads below.
5. **Recommendation, if only one arm runs: WARM, with its $0.33 C+ control.** It costs $1.70 central / $2.23 high against JOINT's $3.06 / $4.10 and takes 5 h against 11.6 h. It needs one init flag rather than a new loss term, and gives the larger tilt unless COT-RETAIN shows heavy forgetting.
   * JOINT runs second, conditionally: if WARM forms the loop but reads below C+ and COT-RETAIN(6k) < 0.22, or if the user wants the CODI-family (SOTA-holding) convention.
   * If WARM does **not** form the loop although the probe shows the plan present, the account is falsified, and JOINT should not run on its strength.

---

## 1. Facts used

### 1.1 §200 status (DATA; relayed, unscored where marked)
* **C, TEXT-COT** (gsm_txt_aug_s1, 24k × 64 = 4.0 Aug epochs, landed):
  - test **0.3215** (n 1,319);
  - acc_by_m 0.292 / 0.529 / 0.368 / 0.200 / 0.138 / 0.018 / 0.048 / 0.000 at m 1–8;
  - valid 0.16 → 0.35, flat from 16k; final train loss 0.076;
  - fit_fix **0.829** vs fit_rs **0.409** (recall share 0.42);
  - GSM-Symbolic main 0.156 / p1 0.059 / p2 0.022;
  - struct gap 0.188; clip rate 0.064 (1.0 over steps 100–1,000).
* **A / B (latent, raw GPT-2 init), interim, relayed:** valid about 0.13 at 14–16k, below GPT-2 no-CoT (about 0.15–0.19). LOOP-NOT-FORMED looks likely. Not landed or scored.
* **D (TEXT-COT + R_ν + R_x):** HELD for budget (CURRENT_TASKS 10:5x). The resampled pair here is conditional on D landing.

### 1.2 Timing anchors (DATA; b286, RTX 3090 / 595.84, the §200 timing host; commit b75abedd)
* **Latent, A's exact flags:** **1.482 s/step** at micro 4. The mem_check forced micro 4 (16.9 GiB at the longest item).
* **TEXT-COT:** 0.265 s/step timed; C's **realised** figure is 6,105 s / 24k = **0.254 s/step**.
* A's own run (b287, $0.248/h) has not landed, so its realised s/step is not in a local log. The b286 figure (same flags, same GPU class) is used. The draft's 0.89 central was too low by 1.7×, because micro 4 is forced at GSM8K's longest item.

### 1.3 Harness facts (PROVED by code inspection; boxes/gsm8k_stepaux.py, prosqa168s2.ProsMem, deltamem.py, deltas5.fwd_read)
* **F1, the step-0 identity.** `read_fixed` returns `x + fade · W_o([oa, od])` (deltamem l.176). At step 0, fade = 0 and W_o = 0 (weight and bias, l.70–72), so every latent path equals the plain trunk exactly. **A warm cell at step 0 is C's model, and its K′ answers are C forced to answer directly.**
* **F2, the item-independent aux query.** `aux_logits` builds [EOT, " Step", " t", " :"] + the teacher-forced " <<e_t>>" (prosqa168s2 l.161–170). `fwd_read` injects the read only at block 6, with a query computed from the decoder's own block-6 input.
  - Positions 0–4 ([EOT, " Step", " t", " :", " <<"]) contain no item token. So the query that predicts the **first operand token** is a function of t alone.
* **F3, the open loop.** The write input is u = (1 − γ)·LN_6(h_6) + γ·LN_w(h_w) (deltamem l.274–288), with h_w the think pass's residual after block 11. γ = 0 for steps ≤ 2,000 and ramps linearly to 1 at 6,000 (gsm8k_stepaux l.282). For γ = 0, ∂u/∂h_w = 0.
* **F4, the answer decoder sees the question.** It reads [EOT] + question + "\n" + "### " + answer (ProsMem.dec_loss). So a memory-free direct path exists. The think input is [EOT] + question with **no "\n"** (think_inputs).
* **F5, the branch init copies the trunk.** `DeltaMem.__init__` copies block 6's ln_1 / c_attn into the memory's LN / qkv and block 11's ln_1 into LN_w (l.61–83), and zero-inits W_o and the gates.
  - **So a warm cell must load C's weights into g before ProsMem is constructed,** for the copies to come from the trunk the branch is attached to (§5.2).
* **F6, the schedule.** lr = 1e-4 × min(1, step/100) × ½(1 + cos(π·step/steps)) (fsq120.lr_at). That is warm-up 100, then **cosine to 0 over --steps**.
  - **Correction to LIT_GSM8K_COT_BUDGETS Reading 2:** §200 C did **not** run at a constant lr. Its valid plateau from 16k coincides with lr ≤ 0.25 × base. C+ (§5.1) re-warms the lr, so it also tests the budget-limitation reading directly.
* **F7, identical batches across arms.** The main rng (default_rng(seed)) is used only for the batch indices. Transforms use their own xr_rng, and the g_add calibration uses rng 16801. So at seed 1, **A, B, C, D, WARM, C+ and JOINT all draw the same index sequence**: the first 12k batches of each are identical.
* **F8, the eval path (C1).** For the latent arm, answers() = think_states (K passes, `m.think`, the identical [EOT, question] input every pass; `--cache_lower` asserts identity) → greedy_after on [EOT] + pre + [ANS] reading S_K. No CoT text is generated or read. **K′0 uses S = 0:** both reads are linear in S, so the read adds only fade·b_o, an item-independent constant.

* **F9, where C plans (PROVED, code).** C's TEXT-COT labels start at the first CoT token (`lab[r, n0:]` with n0 = len(pre)). Position j predicts token j + 1, so the earliest labelled prediction is made **at the "\n" position**.
  - Block 11's output at a question position feeds only the unlabelled LM-head prediction there. Question positions are trained only as attention sources (the K / V inputs of blocks 1–11) for the "\n" / CoT positions.
  - **So C decides its plan at "\n" and after.** At question positions it carries per-token features that later positions read, not a pooled decision.
  - The latent think input ends at the last question token (F4). The derivation-matched measure is therefore **per-number-token** (a fixed-query read is a key-matched retrieval over per-token writes; Lemma R). Pooled features are a weaker proxy.

### 1.4 Precondition probes (DATA, post hoc: run while drafting, after the A / B interim; CPU, 3 threads, Idle; `probe_plan.py`, `probe_nl.py`)
* **Items:** 2,400 Aug-train items (rng 31415) to fit, 800 held-out Aug-train items, and the 408 Aug-valid items of 500 whose e_1 has an operator and a first operand among the question's distinct numbers (slot ≤ 4).
* **Targets:**
  - **OP:** e_1's first operator (4 classes; majority "*": held-out 0.556 / valid 0.490);
  - **SLOT:** the index of e_1's first operand among the question's distinct numbers (5 classes; majority "the first number": 0.645 / 0.566).
* **Features:** from the latent think input exactly ([EOT] + question), layer-normalised:
  - h_6 (the open-loop write source) and h_top (after block 11, the γ = 1 write source);
  - at the last token and as the token mean.
* **Probe:** L2 logistic regression with λ chosen on an inner split.
* **Models:** raw GPT-2 vs §200 C.

**Pooled features (probe_plan.json; accuracy valid / held-out):**

| target | features | raw GPT-2 | C | C − raw |
|---|---|---|---|---|
| OP | h_top last | 0.488 / 0.576 | 0.551 / 0.681 | **+0.064 / +0.105** |
| OP | h_top mean | 0.537 / 0.591 | 0.596 / 0.677 | **+0.059 / +0.086** |
| OP | h_6 mean | 0.554 / 0.635 | 0.586 / 0.669 | +0.032 / +0.034 |
| OP | h_6 last | 0.493 / 0.589 | 0.517 / 0.636 | +0.025 / +0.047 |
| SLOT | h_top last | 0.537 / 0.639 | 0.569 / 0.641 | +0.032 / +0.002 |
| SLOT | h_top mean | 0.561 / 0.651 | 0.556 / 0.647 | −0.005 / −0.004 |
| SLOT | h_6 last / mean | 0.566 / 0.639, 0.547 / 0.644 | 0.569 / 0.645, 0.556 / 0.651 | ≈ 0 |

**Per-number-token and "\n" features** (probe_nl.json; extended λ grid 0.03–3; valid / held-out). Per-token SLOT is an "is the first operand" logistic on each distinct number's last BPE token, then the argmax per item, on items with ≥ 2 numbers (739 held-out, 358 valid). That is the fixed-query retrieval form of Lemma R.

| target | features | raw GPT-2 | C | C − raw | reference |
|---|---|---|---|---|---|
| SLOT | per-token h_top | 0.578 / 0.651 | 0.606 / 0.656 | +0.028 / +0.005 | first-number rule 0.536 / 0.617; chance 0.36 |
| SLOT | per-token h_6 | 0.589 / 0.641 | 0.595 / 0.651 | +0.006 / +0.010 | as above |
| SLOT | "\n" position, h_top | 0.566 / 0.645 | 0.571 / 0.648 | +0.005 / +0.003 | majority 0.566 / 0.645 |
| OP | "\n" position, h_top | 0.510 / 0.555 | 0.522 / 0.661 | +0.012 / +0.106 | majority 0.490 / 0.556 |
| OP | h_top mean (λ re-tuned) | 0.556 / 0.584 | 0.596 / 0.678 | +0.040 / +0.094 | as above |

**Other measurements:**
* **C's in-context e_1** (greedy first CoT line vs "<<e_1>>", Aug valid 500): exact **0.346** (C's format) / 0.350 (EOT-prefixed). Its value is self-consistent with its own expression in **0.934**. So in-context arithmetic is strong, and the expression (plan) is the weak part.
* **K′0 at WARM's step 0** (C forced to answer directly after [EOT] + question + "\n###", Aug valid): **0.022** (0.020 without EOT). Raw GPT-2: 0.000.

**Reading (DATA, post hoc):**
* **(a) The operator transfers.** C's think-pass top features carry e_1's operator: +0.06 valid / +0.09–0.11 held-out over raw. Raw's are at the majority level.
* **(b) The operand selection does not.** The first-operand slot is no more decodable from C's think-pass features than from raw's, at any probed position: pooled, per-token, or C's own planning position "\n" (C − raw ≤ +0.03, about 1 SE). Both sit at or near the first-number rule.
* **This fits F9 + Lemma R.** C selects its operands by attention from the position that **emits** them: an item-dependent query at "<<". Our aux decoder's query at that position is item-independent (F2), so it cannot run C's selection mechanism. The selection has to be moved into the think pass as written content, and the warm start does not supply that.
* **(c) THINK-NECESSARY has room.** K′0 starts at 0.022.

PROBE3_PLACEHOLDER

---

## 2. Derivation

### D0. The decomposition
* **Item:** question q, equations e_1..e_m with e_t = a_t "=" v_t (an expression a_t, a value v_t), and answer y.
* **The latent cell (A's recipe):**
  - K = 10 passes S_t = F_θ(S_{t−1}; x) over x = [EOT, q];
  - the aux at pass t ≤ min(m, K) is CE(" <<e_t>>" EOT | D_aux(S_t; t));
  - the answer is CE("### y" EOS | D_ans(S_K; q));
  - L = L_ans + L_aux.
* **The aux tokens split** into format tokens (" <<", "=", ">>", EOT), expression tokens a_t, and value tokens v_t:
  CE_t = CE(format) + CE(a_t | ·) + CE(v_t | a_t, ·).
* **Three competences:**
  - **(i) ARITH.**
    - *In-context:* v_t given a_t in the context. Under teacher forcing this needs **no memory**: v_t is a function of a_t for the 97.8% exact equations, so the aux decoder can learn it at first order in either init.
    - *Latent:* computing v_{t−1} inside a think pass, without writing tokens. This is needed whenever a later step uses a previous result: 30.7% of Aug operands are previous results (R) (DATA, draft §1.3).
  - **(ii) PLAN.** The map (q, v_<t) → a_t: which literals or previous results, and which operations. This is the GSM8K-specific competence, word problem → equation.
  - **(iii) ROUTE.** Carry the plan's output into S_t in a form D_aux can render, and carry v_{t−1} into pass t's think computation.
* **What each init supplies (DERIVED; the measured parts are §1.4):**

* (ii) splits into **(ii-a) the operation** (which operator or step type) and **(ii-b) the operand selection** (which literal or previous result fills each slot).

| competence | raw GPT-2 | C's trunk (WARM) | JOINT (raw + CoT CE) |
|---|---|---|---|
| (i) in-context arithmetic | learned at first order (memory-free) | present (value self-consistent 0.934; DATA) | learned at the text rate |
| (i) latent arithmetic | absent | partly: the circuits exist, but on attended tokens, not on reads (CONJECTURE) | as WARM, growing |
| (ii-a) operation, in the think pass | ≈ majority (DATA) | **present**: +0.06 / +0.10 over raw (DATA) | grows on C's curve (CONJECTURE) |
| (ii-b) operand selection, in the think pass | ≈ the first-number rule (DATA) | **≈ raw** (≤ +0.03; DATA): C selects by attention at the emitting position (F9) | as WARM: the CoT CE trains selection at emitting positions only |
| (iii) route | must be learned | must be learned | must be learned |

### D1. Lemma R: the step content must be written by the think pass (PROVED, code)
* By F2, the logits for the first operand token of a_t depend on the item only through r_t = R(S_t; q_t). Here q_t is the fixed, t-indexed query, and R is linear in S per head: φ(q_t)ᵀS_add (6 × 64) and S_del q̂_t (6 × 64).
* So an item-specific first operand is predictable only to the extent that a **fixed linear projection of S_t** carries it. The decoder cannot recover from the question what S_t's projection does not hold, because it never sees the question.
* **Consequence:** the item-specific part of (ii), which operand comes first, must be selected **before the read**, i.e. in the think pass, and written.
  - What remains possible without a plan in the think pass is a plan-free summary: a fixed-query bag of the question's features (which numbers occur). Call this **ρ_B**.
* Later tokens of a_t are queried with the teacher-forced earlier tokens in context, so they can use content-addressed recall ("what was paired with 600").
* The answer decoder's query is item-dependent (F4).

### D2. Lemma O: when the plan in the think pass can receive gradient (PROVED, code)
* **(a)** At step 0 (F1), ∂L/∂S ≡ 0, so the write path and the think-pass activations get zero gradient from both losses. W_o's own gradient is ∝ fade, which rises 0 → 1 over 500 steps.
* **(b)** While γ = 0 (steps ≤ 2k), ∂u/∂h_w = 0 (F3). The think pass's blocks 6–11 output is unused, so a plan computed at the top of the think pass gets **zero aux gradient** and cannot be written. Blocks 6–11 are trained only through their shared use in the decoders.
* **(c)** For γ > 0, ∂L/∂h_w = γ · LN_w′ᵀ · ∂L/∂u, and ∂L/∂u passes through the state and the read. Its magnitude is ∝ γ · fade · ‖W_o‖ · ‖∂R/∂S‖.
* So the plan-in-think factor is gated by γ·‖W_o‖, and W_o's item-specific gradient is the cross-covariance of the read with the decoder's backprop signal. That is nonzero in a plan-specific direction **only if the read already carries plan content**.

### D3. Lemma S: escape time of a product of small factors (PROVED, elementary)
* **Setup:** gradient flow on ℓ(Π_{i=1}^N u_i) with −ℓ′ ≡ g > 0 near 0, and a balanced init u_i(0) = ε. Symmetry keeps the u_i equal, so u̇ = g·u^{N−1}.
* **Time for the product to reach c:**
  - N = 1: T = (c − ε)/g;
  - N = 2: T = ln(√c/ε)/g;
  - **N ≥ 3: T = (ε^{2−N} − c^{(2−N)/N}) / (g(N − 2))**, polynomial in 1/ε.
* **If k of the factors are already O(1) (= u\*),** the remaining N − k obey the same law with g → g·u\*^k.
* N = 2 is Saxe et al.'s linear-network mode dynamics (1312.6120), and the N ≥ 3 plateau is their depth effect. Proof: integrate du/u^{N−1} = g dt.
* **Reading:** a component whose gradient is a product of several near-zero factors waits polynomially long. A component with a first-order gradient learns at once. Pre-training one factor to O(1) removes it from the product and multiplies the rest's gain.

### D4. The factor model, raw vs warm (DERIVED|A1–A4)
**Assumptions.**
* **A1 (separable gain near init).** The memory-dependent aux gain is G = s_P·π·ρ + s_B·ρ_B, with:
  - π = the plan content linearly present in the think pass's write source;
  - ρ = the routing gain for that content (write → S → read → W_o → decoder), with γ(t)·fade(t) as a scheduled factor inside it;
  - ρ_B = the routing of the plan-free bag; s_B < s_P.
* **A2 (a direct competitor).** L_ans has a memory-free part reducible by the trunk alone (F4), with gain s_D·π_D, first order in its own parameters.
* **A3 (no plan from raw in the decision window).** π_raw(2–6k) ≪ π_C. Support (DATA): GPT-2 after 2 steps answers 0.000 at every K′ (s200_evaltime_lat); text CoT needs about 4 epochs to reach 0.35 valid with a direct gradient; and the probe (§1.4).
* **A4 (the mapping to SEEDSTAB Part II).** The selection coordinate m is the share of the loss carried by pass-specific memory content (P), against input-determined memory or the direct path (C). The P drift is b_P(t) ∝ γ(t)·s_P·π(t), and the competitor's drift is first order in its own parameters.

**Raw init.**
* The plan-specific gradients are ∂G/∂ρ = s_P·π ≈ 0 and ∂G/∂π = s_P·ρ ≈ 0: a two-factor saddle in (π, ρ).
* By Lemma O, ρ itself contains γ (0 until 2k) and W_o's plan-specific part, which is ε-small until the read carries plan. So the effective N is ≥ 3 near the origin, and Lemma S makes the escape polynomially slow.
* Meanwhile ρ_B (the bag, an input-determined memory: C) and π_D (the direct answer) grow at first order. They lower the loss first, which is gradient starvation (Pezeshki 2011.09468) and Part II's rich-get-richer gain a.
* In Part II's race formula (C4, docs/THEORY_SEEDSTAB Part II §2), b_P ≈ 0, β_C is large, and the head start T_h covers the decision window. So **z ≤ 0, and LOOP-NOT-FORMED is the modal outcome.**
* This matches the relayed A / B interim (valid about 0.13, at the no-CoT level). It is post hoc for that datum.

**Warm init.** Split π = (π_op, π_sel) after (ii-a) / (ii-b), with s_P = s_op + s_sel.
* **The operation part.** π_op,C is O(1) from step 0 (§1.4 (a)). As soon as fade·γ > 0 (from step 2,001), ∂G/∂ρ ⊇ s_op·π_op,C is first order and points in the plan-specific direction. The only small learned factor left for it is ρ (Lemma S, N = 1–2).
* **The selection part.** π_sel,C ≈ π_sel,raw (§1.4 (b)). So the selection is still a two-factor product (selection-in-think × route), and the warm start does not shorten it.
  - It does supply the format, the in-context arithmetic and the operation. Those were memory-free or first order for raw too, except the operation.
  - So **the warm start removes the (ii-a) leap, not the (ii-b) leap.**
* **In Part II this is a partial tilt:** b_P(t) ≈ k·γ(t)·(s_op·π_op,C + s_sel·π_sel,raw)·r(t), where r(t) ≤ 1 is the retention while the latent losses train the shared trunk.
  - μ_Z is linear in it: a tilt, the only practical lever class (√-levers need 15–42×), but smaller than the ideal of a fully present plan.
* **Where the remaining signal sits.** Operator tokens carry at most ln 4 nats with a 0.56 prior; the first-operand slot carries about ln 5 with a 0.65 prior.
  - That puts the information the warm start supplies at roughly a third to a half of pass 1's plan-specific information (calc, order of magnitude only).
  - No calibrated z follows. The prediction is directional: P(loop) for WARM is greater than for raw, but well short of 1.
* **Signature (Part II P3):** a tilt gives an earlier, tighter entry than raw, and the order parameter is never on the C side. For one seed this is a timing prediction: the valid accuracy rises during or just after the γ ramp (2–6k), not at the 17–25k of raw-init entries on nested.

**The selection gap and the THINK-PREFIX variant (DERIVED from F2 / F9 / Lemma R; precondition DATA in §1.4 probe 3).**
* C selects e_1's first operand at the position that emits it, after "<<" (F9). Its residual there must linearly encode the next token: that is its LM head's job.
* So if the think input ends with C's own prefix up to that position, x′ = [EOT] + q + "\n" + "<<", then the last think token's write source **is** C's operand selection. The delta heads store it under that token's key, and a fixed query can retrieve it (Lemma R) once the route aligns: first order, with no selection factor left in the product.
* **C1 is unchanged:** x′ is identical on every pass (a fixed suffix), and the memory is still the only cross-pass channel.
* **What it does not fix:** at pass t ≥ 2 the previous steps are in memory, not in context, so the same position must read them through the injected read (the latent-arithmetic / route residual).
* The variant changes A's think input, so it is a second change. It is reported as **W-PFX**, beside W (A's input exactly), so that W − W-PFX isolates the format.

**The staircase reading (CONJECTURE; a transfer of Abbe et al. 2108.10573 / 2202.08658 / 2302.11055).**
* Their theorems concern SGD on two-layer networks learning sparse Boolean functions under isotropic inputs: a function is learned efficiently iff each new monomial adds few new coordinates (a small leap). We borrow only the qualitative notion.
* Pass t's content depends on pass t−1's result. With the plan present, each pass adds one new link (route v_{t−1} into pass t): a staircase in t, learned in pass order.
* From raw, the first stair needs plan × route together: a leap of ≥ 2.
* The per-pass aux is intermediate supervision, which removes the leap across passes (Wies et al. 2204.02892; Kim & Suzuki 2410.08633 for parity). But it supervises the **composite** plan ∘ route ∘ decoder, so **within** a pass the leap stays.
* **The CoT stage supervises the plan factor directly** (text tokens in context, no route), which removes it from the within-pass product.

**What the warm start does not supply: latent arithmetic (CONJECTURE).**
* For a step with an R operand, v_{t−1} must be computed inside a forward pass that has no a_{t−1} tokens, and then written.
* C's arithmetic runs on attended tokens (at the "=" position). Its transfer to memory-read features is partial.
* This is the residual per-pass fidelity ρ < 1. It is why WARM is predicted **below** C+ at 12k (P-W3), and why aux exact should be lower on R-operand steps than on question-literal-only steps (P-W8, descriptive).
* It is also consistent with the published family: Coconut from CoT reaches 34.1 / 42.9 = 0.79× of its CoT.

### D5. Consistency check: why ProsQA and nested formed loops from raw GPT-2 (CONJECTURE + DATA)
* On ProsQA (§191: K′10 0.950, K′0 / K′1 at chance) the per-pass plan is a premise lookup: find the premise whose subject is the current node, then copy its object. That is attention-native: GPT-2's copy / induction circuitry supplies it, so π_raw = O(1).
* On nested arithmetic the plan is syntactic (reduce the innermost operation) over single-digit tables.
* On GSM8K the plan is semantic (word problem → equation). Even text CoT, with a direct gradient, needs epochs for it.
* **The model therefore predicts:** a raw-init loop forms when the step plan is attention-native, and fails when the plan is the hard part. GSM8K is the first task in this project where the plan, not the route, is the hard part. It is the first task where raw init (A) appears not to form the loop.

### D6. JOINT-COT: the plan supplied concurrently (DERIVED|A1–A4 + calc)
* **The arm:** raw GPT-2 and A's recipe, with L = L_ans + L_aux + λ_cot·CE_text on the same drawn items, through the plain trunk (no memory), from step 0, λ_cot = 1.
* **The plan in the shared trunk follows the text rate.** Proxy: π_J(t) ≈ C's valid curve (0.16 / 0.21 / 0.26 / 0.29 at 2 / 4 / 6 / 8k, against 0.35 final; DATA).
* **The tilt ratio** (calc, `calc_warm.py`): with b_P ∝ γ·π, Part II's swept gain (a = a_max·γ, 1/a_max = 3.3k steps) and equal σ_Z,

  z_J / z_W = ∫e^{−A}γπ_J / ∫e^{−A}γπ_C = **0.63 through 6k, 0.69 through 8k, 0.75 overall**, for a WARM that forgets nothing (r = 1).

* **So JOINT has the larger tilt iff WARM's retention r over the window is < ≈ 0.63–0.75.** COT-RETAIN(t), WARM's trunk text-CoT valid accuracy, measures r: the bar is COT-RETAIN(6k) ≈ 0.63 × 0.35 = **0.22**.
* **JOINT's advantages:** no forgetting, and the plan is trained in the same trunk throughout (A4 holds for all t); it is matched-exposure with A and C (F7); and it is the SOTA-holding family's convention (PCCoT 49.48).
* **JOINT's costs:**
  - **Gradient crowding under the global-norm clip** (the 199-A5 mechanism). The text CE alone saturates the clip early (C: clip_frac 1.0 over steps 100–1,000), so the latent terms' share of each early update is cut.
  - Trunk capacity is shared.
  - The cost is 2.2× WARM's.
  - It is **CODI without the hidden-state distillation**: only CODI's joint teacher CE. CODI also distils hidden states at the answer-prompt token, which targets latent arithmetic directly. That is not part of this arm.
* JOINT's plan grows inside the decision window rather than before it, so its entry is predicted later and less tight than WARM's (Part II P3), but earlier than raw's.
* **The (ii-b) gap applies to JOINT too.** The CoT CE trains selection at the positions that emit operands (F9), not in the think pass. So the tilt ratio above concerns the operation part, and JOINT also leaves the selection to the route. Its THINK-PREFIX analogue (JOINT-PFX) is named here, not costed.

### D7. Exposure and memorisation (DERIVED + CONJECTURE)
* **WARM's total exposure:** C's 24k text steps plus 12k latent steps, i.e. 36k steps = 6 Aug epochs. The latent cell sees the same first 12k batches C saw (F7).
  - The text control at equal exposure is **C+** (C continued 12k from its model.pt, TEXT-COT, re-warmed lr). The pair WARM / C+ has the same start, the same added steps and the same batches. Only the objective of the added 12k differs.
* **Memorisation transfers (CONJECTURE).** C's recall share is 0.42 (fit_fix 0.829 vs fit_rs 0.409).
  - Memorised questions' plans live in the trunk, and the think pass computes them on the same tokens. So the loop routes memorised plans as readily as computed ones, and WARM inherits recall, discounted by the routing fidelity ρ^m (train mean m 2.6).
  - calc, ρ = 0.95: fit_fix ≈ 0.829 × 0.875 ≈ 0.73 and fit_rs ≈ 0.409 × 0.875 ≈ 0.36, so the recall share is ≈ 0.37.
  - The 2 extra epochs on the same items can only raise fit_fix.
* **The resampled pair.**
  - Under R_x, only about 45% of draws keep an item's original numbers (200-2's 55% resample dose). So each original number vector is seen about 4 × 0.45 ≈ 1.8 times in D, against 4 times in C.
  - Theorem X (PROVED, §200 draft D2) bounds the recall worth on resampled items by the collision probability.
  - **Predicted:** D's recall share < C's, and WARM-RS (init D, B's flags) inherits the lower share. Test is unchanged (RS-NEUTRAL, as P2): memorised training answers carry no test information (0 verbatim overlap; 3 near-duplicates).

### D8. C1: the clean-test property is preserved in both arms (PROVED, code; F1, F2, F8)
* **(a)** The eval path is §200 A's:
  - K passes on the identical [EOT, question], with no pass index on the think path (the pass token exists only in the training-only aux prompt);
  - S is the only variable carried across passes;
  - the answer decoder reads S_K with the question in context.
  WARM changes only the initial weights, and JOINT only adds a training-time loss. Neither adds a variable that crosses passes, so **memory remains the only cross-pass channel**.
* **(b)** No CoT text is produced at test. The answer is ≤ 12 greedy tokens after a forced "###", stopping at a newline or EOS (ANS_MAX). Any CoT-shaped output would be an invalid answer, so a CoT habit in the trunk cannot carry computation at test.
* **(c)** K′0 runs with S = 0. The read adds only fade·b_o, which is item-independent. So **K′0 is the trunk's own direct (no-CoT) ability**, including anything the CoT training internalised.
  - At step 0, WARM's K′0 equals C forced to answer directly (F1; the probe's direct_eot).
  - THINK-NECESSARY (K′0 and K′1 ≤ 0.5 × K′10) and LOOP-NOT-FORMED read on the same 1,319 items and the same K′ grid (200-1). K′1 catches a CoT trunk that computes the whole answer in one think pass.
* **(d)** For JOINT specifically: the CoT CE uses the plain trunk with no memory read, on [question, \n, CoT, ### y]. It shapes shared weights only. At test the latent path is evaluated exactly as A's, and the trunk's text-CoT ability is reported only as the descriptive COT-RETAIN, never as the arm's answer.

### D9. Claim status

| claim | status |
|---|---|
| F1–F8 (step-0 identity, item-independent aux query, open loop, direct path, copy-init, cosine schedule, identical batches, eval path) | PROVED (code) |
| Lemma R: the first-operand content must be in a fixed linear projection of S_t | PROVED (code) |
| Lemma O: zero aux gradient to the think pass's top while γ = 0; gating by γ·fade·‖W_o‖ | PROVED (code) |
| Lemma S: product escape times | PROVED (elementary) |
| raw init: b_P ≈ 0, LOOP-NOT-FORMED modal | DERIVED\|A1–A4 |
| warm init: a tilt, b_P ∝ γ·π_C·r | DERIVED\|A1–A4; π_C > π_raw is DATA (probe, post hoc) |
| tilt ratio JOINT / WARM = 0.63 (to 6k) – 0.75 | calc under A1–A4 + the C-curve proxy (CONJECTURE) |
| staircase / leap reading | CONJECTURE (qualitative transfer) |
| latent arithmetic is the residual (ρ < 1) | CONJECTURE |
| ProsQA / nested formed loops because the plan is attention-native | CONJECTURE + DATA (post hoc) |
| memorisation transfers to WARM | CONJECTURE |
| C1 preserved | PROVED (code) |

---

## 3. Predictions (one seed, RTX 3090; final checkpoint; ±2.8 points = 2 SE at n 1,319)

| id | prediction | from | falsified if |
|---|---|---|---|
| P-W0 (precondition) | The plan probe's best feature set on C's think pass beats raw GPT-2's by ≥ 0.05 on Aug valid, for the first-operand slot or the operator | D4 (A1, A3) | no feature set separates C from raw (then A1's warm half is unsupported at pass 1, and P-W1 is downgraded to CONJECTURE before launch) |
| **P-W1** | WARM **LOOP-FORMED**: K′10 − K′0 ≥ 2.8 points; central gap ≈ 20 points (K′10 ≈ 0.28, K′0 ≤ 0.08) | D4 (tilt) | K′10 − K′0 < 2.8 with P-W0 satisfied: **the staircase / tilt account is falsified for GSM8K** |
| P-W2 | THINK-NECESSARY for WARM: K′0 ≤ 0.5 × K′10 and K′1 ≤ 0.5 × K′10 (central K′0 ≤ 0.08, K′1 ≤ 0.10) | D8 + probe direct_eot | either fails (the CoT trunk answers directly or in one pass) |
| **P-W3** | WARM final test in [0.24, 0.35], central 0.28–0.30 (calc: 0.321·E[ρ^m] = 0.280 at ρ 0.95, 0.304 at 0.98). **Reaching 0.321 is not predicted** (P ≈ 0.3). R1-W (WARM − C+) reads **matches or below, not above** (central −3 to −4) | D4 + D7 | test < 0.24 (below 0.75 × C, under the worst published latent / CoT ratio 0.79); or R1-W reads **above** (then latent arithmetic is not the residual) |
| P-W4 (timing, the tilt signature) | WARM's Aug-valid accuracy at 6k ≥ A's final valid + 3 points, and at 8k ≥ 0.20 | D4 (Part II P3: tilt ⇒ early entry) | valid(8k) < 0.20 while the final valid is ≥ 0.24 (a late, escape-like rise: the tilt is not the mechanism) |
| P-W5 | COT-RETAIN(6k) ≥ 0.22 (= 0.63 × 0.35) | D6 (retention needed for WARM's tilt to exceed JOINT's) | < 0.22 (heavy forgetting; then JOINT's derived tilt is the larger) |
| **P-W6** | Memorisation transfers: WARM fit_fix ≈ 0.73 (range 0.55–0.85), fit_rs ≈ 0.36 (0.25–0.42), recall share ≥ 0.20 (central 0.37) | D7 | recall share ≤ 0.10 (the loop does not route memorised plans) |
| P-W7 (if D lands) | WARM-RS recall share ≤ WARM's − 0.10, and WARM-RS − WARM test within ±2.8 (RS-NEUTRAL) | D7 + Theorem X | WARM-RS − WARM ≥ +2.8 (then recall competed with generalisation, as on ProsQA §191) |
| P-W8 (descriptive) | WARM's aux exact (teacher-forced, test) is ≥ 0.10 lower on steps with a previous-result operand than on question-literal-only steps | D4 (latent arithmetic) | the gap is ≤ 0.02 |
| **P-J1** | JOINT LOOP-FORMED (K′10 − K′0 ≥ 2.8), with lower confidence than P-W1 | D6 (tilt 0.63–0.75 of WARM's) | K′10 − K′0 < 2.8 (the concurrent plan is not enough within 24k) |
| P-J2 | THINK-NECESSARY for JOINT | D8 | either fails |
| P-J3 | J-LEVER: JOINT − A ≥ +2.8 (paired, same batches) | D4 / D6 | JOINT − A < +2.8 |
| P-J4 | J-C3: JOINT − C reads **below** (central −5; JOINT ≈ 0.25–0.30) | D4 (ρ^m) + crowding | JOINT − C ≥ +2.8 |
| P-J5 | No large interference: JOINT's trunk text-CoT valid at 24k ≥ 0.30 (C 0.35) | D6 | < 0.30 (INTERFERES: the latent losses slow plan learning) |

* **No derivation predicts SOTA.** A 12k WARM cell is 6 epochs in total, against the bars' 25–40. "Budget-limited" applies to any R2-style comparison, as in note 200-4.
* **Between the arms (descriptive):** WARM vs JOINT has unequal latent steps (12k vs 24k). No ordering is predicted; the order of entry is reported.

---

## 4. Prior art (IDs verified via arxiv.org/abs on 2026-10-14) and what is new

| work | init | CoT signal during latent training | relation |
|---|---|---|---|
| **Coconut**, Hao et al. **2412.06769** | stage 0 = CoT training (about 6 epochs; the README selects a checkpoint at about 40% valid) | curriculum: the remaining language steps stay as targets | WARM is Coconut's stage-0 convention applied to our loop. RiM's re-run: 31.1 with stage 0 vs 22.3 without |
| **iCoT-SI**, Deng, Choi & Shieber **2405.14838** | an explicit-CoT-trained model | CoT tokens removed step by step | the same "CoT model first" principle, internalising by removal |
| **Implicit CoT via KD**, Deng et al. **2311.01460** | a CoT-trained teacher | the student distils the teacher's hidden states | the teacher-first family |
| **CODI**, Shen et al. **2502.21074** | raw GPT-2 + LoRA | a **joint** CoT teacher CE throughout, plus L1 hidden-state distillation | JOINT is CODI's joint teacher CE without the distillation. Static-teacher ablation 27.1 vs 43.7 (Literature) |
| **PCCoT**, Wu, Teng & Tu **2506.18582** | as CODI | as CODI (Jacobi-parallel latents) | the SOTA bar 49.48, CODI family |
| **SIM-CoT**, Wei et al. **2509.20317** | a trained Coconut checkpoint (itself from CoT) | step-level aux on the latents | the closest to our step aux: 44.8 vs its CoT 42.7 (1.05×). Its config validates on **test** |
| **Zhang et al. 2607.20594** | — | — | budget annealing transfers a found algorithm across loop contracts (4/4) and bypasses the lottery; operator-first masters the operator before chaining. The **in-house analogue of "learn (ii) first, then route"** |
| Abbe et al. **2108.10573 / 2202.08658 / 2302.11055** | — | — | staircase / merged staircase / leap complexity (theory, borrowed qualitatively) |
| Saxe et al. **1312.6120**; Pezeshki et al. **2011.09468** | — | — | product (saddle) dynamics; gradient starvation |
| Wies et al. **2204.02892**; Kim & Suzuki **2410.08633** | — | — | intermediate supervision makes compositional / parity tasks learnable |

**What is new: little, and it is framed that way.** Starting a latent model from a CoT-trained checkpoint (WARM), or training it beside a CoT loss (JOINT), is standard practice. Every published GPT-2 GSM8K-Aug latent number does one or the other. These arms **match the field's initialisation**, so that §200's latent-vs-text comparison becomes like-for-like. What this design adds:
1. The routes applied to an identical-input recurrent memory with per-pass step aux, together with a **removal test** (K′0 / K′1). The published GPT-2 latent results report none (RiM: none; LIT_GSM8K_AUG_BARS).
2. **An equal-exposure text control (C+)** that shares the start, the added steps and the batches. The published comparisons quote CoT bars trained under different budgets.
3. **Falsifiable mechanism predictions:**
   - the plan-probe precondition;
   - the tilt's timing signature;
   - the derived crossover between the routes, COT-RETAIN(6k) ≈ 0.22;
   - the R-operand deficit;
   - memorisation transfer, which the resampled pair tests.
4. JOINT isolates CODI's joint teacher CE **without** distillation. As far as we read, CODI's tables do not report that single-component arm; Literature to confirm.

---

## 5. Design

### 5.1 Cells (RTX 3090 for every cell; seed 1; GPU and driver recorded; bs 64, lr 1e-4, warm-up 100 + cosine over --steps; run dirs in boxes/runs_fsq/)

| cell | dir | init | arm and flags | steps | role |
|---|---|---|---|---|---|
| **W** | `gsm_lat_warm_s1` | C's model.pt (TEXT-COT, Aug) | §200 A's flags exactly + `--init_from`, `--cot_valid` | 12k | WARM-START (Coconut convention) |
| **C+** | `gsm_txt_aug_cont_s1` | C's model.pt | TEXT-COT + `--init_from` | 12k | equal-exposure text control for W; also tests C's budget limitation (F6) |
| W-RS | `gsm_lat_warmrs_s1` | D's model.pt | B's flags (R_ν + R_x) + `--init_from`, `--cot_valid` | 12k | resampled pair; **only if D lands** |
| D+ | `gsm_txt_augrs_cont_s1` | D's model.pt | TEXT-COT + R_ν + R_x + `--init_from` | 12k | its control; only if D lands |
| **J** | `gsm_lat_joint_s1` | raw GPT-2 | §200 A's flags + `--joint_cot 1.0`, `--cot_valid` | 24k | JOINT-COT (CODI convention); paired with A and C (F7) |

* **Steps.** 12k for the warm cells: 2 Aug epochs of latent training, and 6k closed-loop steps after the γ ramp. 8k would leave only 2k closed-loop steps (it saves $0.42).
  - λ = 1 throughout (`--aux_fade 12000,12000`; A's λ is also 1 throughout).
  - A **descriptive** valid read at 8k (logged every 2k anyway). No early stopping.
  - J is 24k, matching A and C step for step on the same batches.
* **Everything else is A's recipe:**
  - DeltaMem-R 6 + 6 after block 6, top writes, γ 2k → 6k, W_o fade 500, g_add calibrated at step 0 (on the loaded trunk), fast_add, chunk_delta 64;
  - K = 10, --cache_lower, the pass token kept;
  - a fresh AdamW.
  - **No FAIR-START or other tilt is stacked:** the init is the only change.

### 5.2 Harness changes (Experiments implements; Audit reviews the diff; lazy-import care: every new parameter optional)
1. **`--init_from <path>`.**
   - Load the state dict strictly into `g` **between** `s5.load_gpt2` and `ProsMem(g)`, so DeltaMem's copies (block 6 ln_1 / c_attn, block 11 ln_1) come from the loaded trunk (F5).
   - The text arm loads into `g` likewise.
   - Record the source file's sha256 and path in args.json.
2. **`--joint_cot λ`** (stepaux only).
   - Per micro-batch, after the latent loss's backward, a **second backward** of λ·CE_text on the same drawn items: seq = pre + cot + a1 + EOS, through the plain trunk, the textcot arm's existing code.
   - Gradients add, so the update is identical to backpropagating the sum, and peak memory is the max of the two graphs, not the sum. That keeps micro 4 safely: about 16.9 GiB, against about 20.8 GiB summed, which is at the 21-GiB auto_micro budget (calc: C's text activations, about 15.5 GiB per 64 items, so about 3.9 GiB per micro-4 chunk, on top of A's 16.9).
   - Log text_ce in every record.
3. **`--cot_valid`.** At each valid eval, also the plain trunk's TEXT-COT accuracy on Aug valid (cot_batched; about 10 s on a 3090, C's valid_sec), logged as `cot_valid_acc` = COT-RETAIN(t).
4. **eval.json addition:** aux_exact_by_pass split by operand grounding (steps with any R operand vs question-literal-only steps, from gsm8k_data's grounding), for P-W8.

### 5.3 Gates (before launch)
* **INIT-IDENTITY (fp32, CPU or local card, 200 Aug-valid items).** A W construction at step 0 gives:
  - acc_by_kprime constant over K′;
  - answers bitwise equal to C's plain trunk on [EOT] + pre + [ANS];
  - and equal to the probe's direct_eot (F1).
* **COPY-INIT:** W's mem.qkv / mem.ln / mem.ln_w equal C's block 6 / 11 tensors (not raw GPT-2's).
* **JOINT-EXACT (fp64, 2 items):** the gradient of two separate backwards equals that of one backward of the sum (max |Δ| ≤ 1e-12 relative).
* **gradflow (JOINT):** nonzero gradient from each of answer CE, aux CE and text CE (memory: the §116 no_grad lesson).
* The **plain-path smoke** (no new flags; construct + 1 step) after the edit.
* **mem_check** at the longest item with both terms.
* `--time_steps 200` for J on a 3090; W reuses b286's 1.482 s/step (same flags).
* **The projection is restated to Audit and the user before launch.**

### 5.4 Checkpoint rule
* **FINAL checkpoint for every reading.** The Aug-valid-selected checkpoint (every 2k) is reported beside it, labelled.
* **The test is never used for selection.** C+ and D+ follow the same rule.
* Kept: model.pt and model_validbest.pt only (data retention).

### 5.5 Readings (fixed at registration; a new `theory/score_s20x_warm.py`, fixture-checked before any output)
* **W-LOOP / J-LOOP:** LOOP-FORMED iff K′10 − K′0 ≥ 2.8 points (200-1: the 1,319 test items, acc_by_kprime); otherwise LOOP-NOT-FORMED. K′0 is printed beside it.
* **THINK-NECESSARY** (per cell): K′0 ≤ 0.5 × K′10 and K′1 ≤ 0.5 × K′10.
* **W-INIT** (the derivation's test, against A's registered reading):
  - **INIT-WAS-THE-BARRIER:** W formed and A not formed;
  - **INIT-NOT-SUFFICIENT:** W not formed (with P-W0 met, this falsifies D4);
  - **INIT-NOT-NEEDED:** A formed.
* **W-C3:** R1-W = W − C+ (paired, the same 1,319): above (≥ +2.8) / matches / below (≤ −2.8).
  - Beside it: W − C and W − A.
  - **Quoted, not compared:** the family ratios (Coconut 0.79×, SIM-CoT + Coconut 1.05×) and the bars (CoT 41.8 / 42.7 / 44.1; PCCoT 49.48), with the budget caveat.
* **J-C3:** J − C at the R1 thresholds. **J-LEVER:** J − A, where JOINT-HELPS needs ≥ +2.8.
* **W-RS** (conditional): W-RS − W and D+ − C+, read as §200 R3.
* **W-MEM:** recall share (fit_fix − fit_rs) for W, W-RS, J, against C / D / A.
* **COT-RETAIN:** W's and J's cot_valid_acc at each 2k; P-W5's bar is 0.22 at 6k.
* **Descriptive, as §200:**
  - the valid curve;
  - entry(0.5 / 0.1) on answer_ce;
  - path_indep at the final checkpoint;
  - K′ by m; IDLE-PASSES;
  - R6 FILLER-N; R7 STABLE-12 / 16;
  - the struct gap; dedupe; GSM-Symbolic (eval-only, local);
  - the aux_exact split (P-W8);
  - Meter FLOPs and timed s/item (on one card).
* **Single-seed rules:**
  - **same-seed only** (note SEED-1);
  - every W-vs-A contrast carries entry steps;
  - CONFOUNDED-BY-ENTRY is flagged where both have entries > 5k apart (by construction W's entry is expected far earlier, so that flag describes the effect rather than confounding it, and is reported as such).

### 5.6 OC note (calc, `calc_warm.py`)
* **The binary loop reading at one seed.** The joint outcome (W formed, A not formed) has probability:
  - 0.63 if P_warm = 0.9 and P_raw = 0.3 (the derivation's picture);
  - 0.25 if both are 0.5 (init irrelevant);
  - 0.21 if both are 0.3.
* **The likelihood ratio is about 2.5–3:** a direction, not a rate. **A one-seed INIT-WAS-THE-BARRIER is worded as "on seed 1".** A rate claim needs ≥ 3 seeds per arm (all-of-3 power 0.73 at p 0.9; size 0.125 at p 0.5).
* **Paired accuracy readings** (paired SE 1.25–1.5 points):
  - P(matches | true Δ 0) = 0.94–0.97;
  - P(below | Δ −4) = 0.79–0.83;
  - P(below | Δ −2) = 0.26–0.30.
  So "matches" does not exclude a 2-point deficit, and the wording stays "within ±2.8".
* **One GPU class (3090).** Same-seed, same-card runs reproduce. A cross-card flip is a separate draw (the GPU-class memory note).

### 5.7 Cost (calc; b286 anchors: latent 1.482 s/step at micro 4, text 0.254 s/step realised; $0.255/h, the §200 host class; + $0.10 per host + about 0.05 h of eval; high = × 1.35 for host-speed variance)

| cell | hours (central / high) | $ central / high |
|---|---|---|
| W, 12k | 5.0 / 6.7 | **1.37 / 1.82** |
| C+, 12k | 0.9 / 1.2 | 0.33 / 0.41 |
| W-RS + D+ (if D lands) | 5.9 / 7.9 | 1.70 / 2.23 |
| J, 24k | 11.6 / 15.7 | **3.06 / 4.10** |

* **Blocks:**
  - **WARM first (W + C+): $1.70 / $2.23**, about 5 h wall on two hosts;
  - + the resampled pair: $3.40 / $4.46;
  - + JOINT: **$6.46 / $8.56**.
* J's micro stays 4 only with the two-backward implementation. If micro 8 is forced, s/step rises (inside the high band only if the rise is ≤ 1.35×). Its `--time_steps` gate decides.
* For scale: §200's four cells were projected at $4.28 central (the 0.89 s/step anchor). A realises about $2.45 at 1.48 s/step.
* **Realised $ replaces the projection per cell at landing. A hard cap for the approved block is the user's (via Audit).**

### 5.8 Run order and interpretation
* **If only one arm runs: W (+ C+).**
  1. It is cheaper (×0.55) and faster (5 h vs 11.6 h).
  2. It needs one init flag, not a new loss term, so the implementation risk is low (no crowding, and no gradflow exposure beyond A's).
  3. It is the convention with the clearest published ablation (31.1 vs 22.3).
  4. Its tilt is the larger unless COT-RETAIN(6k) < 0.22 (D6).
* **JOINT second, conditionally:**
  - W formed but reads below C+ **and** COT-RETAIN(6k) < 0.22. The derivation then says the plan was forgotten before the loop closed, and JOINT's concurrent plan is predicted to tilt more.
  - Or the user wants the CODI-family (SOTA-holding) comparison at matched 24k.
  - **Not** if W reads INIT-NOT-SUFFICIENT with P-W0 met: the account behind JOINT is then falsified.
* **What the outcomes mean for the GSM8K line:**

| outcome | meaning | next (each a new decision) |
|---|---|---|
| W formed, A not; R1-W matches | the barrier was the plan leap, and our loop matches text at like-for-like init and exposure: **the first GSM8K C3 evidence** (one seed) | seeds 2–3 of W + C+; then a budget-matched run (25 CoT epochs, the published exposure) as the SOTA-relevant item for the user |
| W formed; R1-W below | init removes the barrier, but routing costs accuracy at 12k (ρ < 1; P-W8 locates it) | longer latent training if STILL-RISING; JOINT if COT-RETAIN is low; a latent-arithmetic lever (e.g. CODI-style distillation, with its own derivation) |
| W formed; R1-W above | latent-specific gain at equal exposure; P-W3's ceiling is falsified (latent arithmetic was not the residual) | replication before any wording |
| W not formed, P-W0 met | **the staircase / tilt account is falsified** for GSM8K: the plan is present, yet no loop forms. The barrier is routing itself (e.g. the single read layer with an item-independent aux query, or idle passes) | pause latent-on-GSM8K spend; derive a routing change first; JOINT not run on this account |
| P-W0 not met (probe) | the warm trunk's think pass does not carry e_1's plan any more than raw | revisit before launch: P-W1 becomes CONJECTURE; the think input lacks C's "\n" planning position (F4), so check the probe at that position before any format change |

---

## 6. Risks
1. **Forgetting before the loop closes.** C ended its cosine at lr 0. W re-warms to 1e-4 and spends 2k open-loop steps (γ = 0) training the shared trunk on latent losses, which can erode π before the plan can be written. COT-RETAIN measures it; FAIR-START (γ = 1 from step 0) would remove the window but is not stacked, to stay like-for-like with A.
2. **The think-input format.** C plans at the "\n" after the question. The think input ends at the question's last token (F4), and the latent paths prefix EOT, which C never saw. The probe uses the latent format exactly.
3. **The exposure confound.** W has 6 epochs to A's 4 and C's 4. C+ removes it for the text comparison. W vs A is "init + exposure", reported as such; J vs A and J vs C are step-matched.
4. **JOINT's crowding and capacity share** (D6); P-J5 and the logged text_ce read it.
5. **Seed lottery and GPU class:** one seed, one GPU class (§5.6).
6. **Inherited memorisation** inflates fit_fix and GSM-Symbolic drops (C: main 0.156 vs test 0.321). P-W6 / P-W7 read it, and GSM-Symbolic is quoted beside every headline.
7. **Budget-limited:** 6 epochs total, far below the published 25–40. No SOTA reading is possible; R2-style statements are "budget-limited" (200-4).
8. **Post hoc:** A / B's interim and the probe were seen while drafting. The registration must say so, and the predictions above were set with that knowledge.

---

## 7. Corrections and notes for Theory
1. **LIT_GSM8K_COT_BUDGETS, Reading 2:** "constant lr" is wrong for §200 C. The harness uses warm-up 100 + cosine to 0 (fsq120.lr_at; F6). C's flat valid from 16k coincides with lr ≤ 0.25 × base. C+ settles the budget question in-house for $0.33 as a by-product.
2. **The draft's 0.89 s/step central** for latent GSM8K cells was low. b286 measured 1.482 s/step (micro 4 is forced by the longest Aug item), so §200 A realises about $2.45, not $1.54.
3. **The brief's (i) "per-step arithmetic"** splits in two:
   - in-context arithmetic (v_t | a_t), which is memory-free under teacher forcing and so not a barrier for either init;
   - latent arithmetic (computing v_{t−1} inside a think pass), which neither init supplies directly. The latter is the predicted residual.
4. **Lemma R is a design fact worth recording beside §200:** the aux decoder cannot plan, because its query at the first operand is item-independent. This is what forces the computation into the think pass, and so into the memory (C1 by design).

## 8. Files (this folder)
* `probe_plan.py` → `probe_plan.json`, `probe_plan.log`: the plan probe, direct (K′0 at step 0), and C's in-context e_1.
* `calc_warm.py` → `calc_warm.json`: acc(ρ), the tilt ratio, OC, cost.
* Rerun with `python -I <script>` from this folder (CPU; the probe takes about 30–40 min at 3 threads, Idle priority).
