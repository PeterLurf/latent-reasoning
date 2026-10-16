# A. VERSION-BANK (VB): a bounded looped core over a pass-global attention memory

*Theory research subagent A (architecture-first: bounded per-pass compute + attention memory). Written 2026-10-15, after the user's design constraint relayed by Audit: "make sure the memory is pass global … not like rnn memory … like attention memory". Draft only: nothing is registered, and every novelty claim needs Literature's check.*

**One-paragraph summary.**
* GPT-2 small is split into a **prelude** (blocks 0–5, run once on the problem), a **shared core** (blocks 6–9, looped K times) and a **coda** (blocks 10–11 + ln_f + LM head).
* **Each pass t writes a new version V_t of the whole problem**: one 768-d slot per problem position, plus scratch slots on GSM8K.
* **Every version is kept in a bank.** Every core layer of every later pass, and the coda, can read any of them through content-based attention. Nothing is overwritten, and there is no fixed-size recurrent state.
* Because bank entries are pass **outputs** read by **every** core layer (Feedback-Transformer style, 2002.09402), depth builds up *through the attention memory itself*. Pass t's entries have depth exactly 6 + 4t. One pass is the depth-12 pretrained network, and the coda adds only 2 layers.
* **Training:** a **readiness-window answer loss** (the answer is supervised only at passes τ..K, with τ = the item's step count), plus a **per-pass recode target** (version t decodes, slot by slot, as the expression after t reductions on nested; the t-th CoT equation is decodable from it on GSM8K).

---

## 1. Mechanism

### 1.1 Layer map and why

| part | VB-12 (default) | VB-8 (tight, stage 2) | runs |
|---|---|---|---|
| prelude 𝒫 | blocks 0–5 | blocks 0–3 | once, on the problem |
| core 𝒞 (shared) | blocks 6–9 (d = 4) | blocks 4–5 (d = 2) | K times |
| coda 𝒟 | blocks 10–11 + ln_f + head (c = 2) | blocks 10–11 (c = 2) | after any pass t |
| one-pass depth P + d + c | **12** | 8 | |

**Why VB-12 for a pretrained init:**
* **Pass 1 is the pretrained network.** Prelude → core once → coda runs blocks 0..11 in order on [EOT, problem, " ="]. This is exact with the bank biases at −∞ (gate G1, §7) and near-exact at their init of −5 (§1.2). It is the strongest LOSES-ONE-PASS mitigation: the K′ = 1 path starts at DIRECT's function, not at a layer-skipped one.
* **The loop is the middle third** (blocks 6–9). The project's memory layer is already block 6, and --cache_lower already caches blocks 0..5. McLeish et al. (2511.07384, verified) convert pretrained LMs into prelude / recurrent-core / coda models by looping middle layers, with "a curriculum of recurrences". Their core keeps one overwritten state per token and, Huginn-style, re-injects the input (Literature to confirm the second point for McLeish). VB does neither (§1.4).
* **The coda is too shallow to compute much** (c = 2), and it never sees raw problem tokens: only bank entries (§3, G3).

VB-8 tightens the one-pass bound to 8 layers, at the price of dropping blocks 6–9 (more LOSES-ONE-PASS risk). It enters only if VB-12 shows pass-1 precompute (not expected on nested; §4) or on GSM8K (§7.3).

### 1.2 Forward pass (B items, D = 768, H = 12 heads, hd = 64)

**Pass 0: prelude, once.**
* x = [EOT, problem tokens, " ="] (nested), length T (mean 26, max 34 at n = 8; research/heights.py).
* GSM8K: x = [EOT, question] + M_s = 16 scratch positions. The scratch inputs are learned embeddings (init = wte(" .") + N(0, 0.02²)) that pass through the prelude like tokens, so their V_0 entries are at h_6 scale.
* V_0 = 𝒫(wte(x) + wpe[0..T−1]) ∈ ℝ^{B×T×D}, the residual after block 5 (`deltamem.gpt_lower`, causal).
* The **bank** starts as 𝔅_0 = {V_0}.

**Pass t = 1..K: core.**
* **Input U_t⁰ ∈ ℝ^{B×T×D}:**
  * **CARRY** (default): U_t⁰[j] = N(V_{t−1})[j] = g · V_{t−1}[j] · rms(V_0[j]) / rms(V_{t−1}[j]). This per-slot RMS reset returns each slot to its own pass-0 scale (the project's RMS lesson). g is learned, init 1, so pass 1 is the exact identity (U_1⁰ = V_0). No detach.
  * **PURE** (the user's strict reading): U_t⁰[j] = q + wpe[j] · ρ̄/rms(wpe[j]), with q learned (init = the mean h_6) and ρ̄ = the step-0 median rms(h_6). The slot knows only its position; everything else must be read from the bank. PURE is not function-preserving at K = 1.
* **Core layer ℓ ∈ {6, 7, 8, 9}:** queries are the current version's slots. Keys / values are
  **[ bank entries (s, i) for s = 0..t−1, all positions i ] ∪ [ current-version slots ]**.
  * Bank keys / values come from block ℓ's *own* ln_1 + c_attn applied to the stored output vectors V_s[i]. They are cached per (s, ℓ), because V_s is fixed once written. The cache must keep its gradient: no `no_grad`.
  * The additive logit mask (B, H, T, N_bank + T) is:
    * bank entry from version s: **β_{ℓ,h} + γ_{ℓ,h}·(t − 1 − s)**. β_{ℓ,h} is learned, init −5: at equal raw logits, with about T/2 causal current keys, the bank then holds ≈ 1.3% of attention mass at t = 1 and ≈ 10% at t = 8. γ_{ℓ,h} ≤ 0 is learned, init 0: a relative-age recency bias.
    * current version: 0 where allowed. Allowed means causal (j′ ≤ j) under CARRY (pretrained-compatible) and all slots under PURE (fresh queries, no pretrained pattern to keep).
    * padding: −∞.
  * The rest of the block is the pretrained one (`deltamem._block` with this mask through `scaled_dot_product_attention`, as `nested170.rs_run` already builds masks).
* **Write:** V_t = U_t⁴ (after block 9). It is appended: 𝔅_t = 𝔅_{t−1} ∪ {V_t}. Bank size after K passes: (K + 1)·T entries (nested ≈ 340 at K = 12).

**No pass index anywhere:** every version reuses positions 0..T−1, and there is no pass token. The only temporal signal is the *relative* age t − 1 − s, which is shift-invariant across passes (ALiBi-like, for C2). The bank's growing size still reveals t implicitly, so G6 is closed by content, not by hiding t (§3).

**Coda after pass t (answer and text aux).**
* Blocks 10–11 run over [V_t's positions (causal) ; decoder tokens], with bank keys for V_0..V_{t−1} (bias β^𝒟_{ℓ,h}, init −5). Then ln_f and the tied LM head.
* **Nested answer:** read at the " =" position (the last slot of V_t), with CE over the 11 value tokens (`NUM_IDS`), as in `InjectN.dec_logits`.
* **GSM8K answer / any text aux:** the decoder tokens (e.g. ["###", a_1..a_k, EOS], or [EOT, " Step", " :"] + "<<e_t>>") are first encoded by blocks 0..5 *among themselves only*: they see no problem token and no bank entry. They then join at block 10 behind V_t.
  * So the decoder has 2 layers of contact with the memory. This layer skip (h_6 → block 10) is trained.
* **K′ = 0** is the coda directly on V_0: an 8-layer path (prelude + coda).

**Recode heads.**
* **Slot recode (nested only): R = LN + Linear(768 → 20)** on every slot of V_t. The 20 classes are VOID, EOT, " =", " (", " )", 4 ops and 11 values. It is linear, so it **cannot compute**.
* **Text recode (nested N3, GSM8K):** the coda path above with the NOPASS prompt [EOT, " Step", " :"] + the teacher-forced target (e_{t+1} on nested, "<<e_t>>" + EOT on GSM8K), reading [V_t ; bank].

**Slot-aligned E_t (definition, nested).**
* Follow Feng's CoT order (`nested170.cot_steps`: the leftmost evaluable reduction per step).
* When the reduced node's rendered span [a, b] collapses to its value v (the span includes any brackets dropped at that step), slot b gets class "value v" and slots a..b−1 become VOID. Every other slot keeps its class from E_{t−1}.
* For t > n (HOLD): E_t = E_n (the root value at its slot, EOT and " =" unchanged, the rest VOID).
* **Gate:** dropping VOIDs from the slot targets must re-render `cot_steps(root)[t]` exactly, on 100k generated items.
* Causal attention within a version is no obstacle. Reduction t is decided from E_{t−1}, including the token to the right of b that a +/− decision needs, and the bank shows all of V_{t−1} to every slot.

### 1.3 Batching and cost (nested, B = 64, K_max = 12)

* Masked per-item K as `kmatch_aux_loss` (bank entries of passes > K_i masked for item i). All coda calls are batched after the last pass, over every (item, t ∈ [τ_i, K_i]) pair, with the bank masked to s < t.
* About 56 block launches per step (6 + 4·12 + 2; + 8 for the text aux) against about 150–300 for INJECT with its 12-block per-pass aux. §198(a) cached INJECT took 0.861 s/step, and VB-12 should take ≈ 0.5–0.7 of that.

### 1.4 What is new (Literature to check every item)

| | **CODI (§209)** | Huginn 2502.05171 / McLeish 2511.07384 | Feedback Transformer 2002.09402 | **VB** |
|---|---|---|---|---|
| cross-pass memory | KV of all latents, **same layer** | one overwritten state per token | all past steps' merged states, read by every layer | **all versions' output vectors, read by every core layer and the coda** |
| what carries depth across passes | a **single fed-back 768-vector** (last hidden → next latent's input) | the overwritten residual | the memory | **the attention memory itself** (L1) |
| per-step width | 1 vector | T | 1 token | **T (+ M_s) slots: a whole re-coded version** |
| per-pass depth | 12 (full model) | core only | full | **4 (2 in VB-8)** |
| input during passes | question KV at every layer (G2 open) | re-injected every step (2609.01924: recompute) | n/a | **only as V_0, an addressable depth-6 entry** |
| decoder | 12 layers + question (G3 open) | coda on the state | n/a | **2 layers, bank only** |
| objective | teacher CE + L1 at one token | LM loss | LM loss | **readiness window + per-pass recode** |

Against our own designs: §208 QO kept a 12×64×64 DeltaMem written by full 12-layer passes; §198(c) READ-SLOTS re-read a frozen input KV every pass through 8 slots and a DeltaMem. VB keeps no overwritten state.

**The precise difference from CODI.** CODI's latents also attend to all earlier latents through the KV cache. But a latent reads latent j's layer-ℓ states *at layer ℓ*, and same-layer attention adds width, not depth (L1-b, §4).
* All of CODI's serial composition runs through the one-vector input feedback. That is the RNN-like channel the user does not want.
* In VB, the serial channel *is* the pass-global memory.
* VB adds three further differences: a whole version is written per pass; per-pass depth is bounded; and the decoder cannot see the question.

---

## 2. Training objective

Let τ_i be the item's step count: n_i reductions on nested, m_i CoT equations on GSM8K (from the CoT text, which is scalable). K_i = min(τ_i + U{0..4}, K_max) (the WINDOW form), with K_max = 12 on nested and 10 on GSM8K.

* **L_ans (readiness window):** (1/B) Σ_i mean_{t ∈ [τ_i, K_i]} CE(𝒟(V_t, 𝔅_{t−1})_answer, y_i).
  * It is never applied at t < τ_i. Nothing rewards or penalises early answers.
* **L_rec, slot form (nested N1 / N2):** (1/B) Σ_i mean_{t=1..K_i} mean_{j real} CE(R(V_t[j]), E_t^{slot}[j]). HOLD applies for t > n.
* **L_rec, text form (nested N3 / GSM8K):** §175's token-mean CE of the aux decoder at t ≤ min(τ_i, K_i), plus §182 HOLD (target e_{n+1}) for t ∈ (n, K_i] on nested.
  * GSM8K uses no HOLD, as §200.
* **L = L_ans + λ L_rec, with λ = 1 constant to the end.**
* **Curriculum (nested; existing `--anneal_gate`, note 196-1):** n_max 3 → 5 → 8. Advance when the answer-CE bin is < 0.3 for 2 consecutive 1k bins, with at most 8k steps per stage; the final stage runs to 48k.
  * On GSM8K: a curriculum of recurrences (McLeish): K_max 2 → 5 → 10 at 2k / 6k steps.
* **Optimiser:** AdamW at lr 1e-4, wd 0.01, `lr_at` cosine, clip 1.0 (nested170 defaults). The new scalars (β, γ, g) and R are in the same group.

**Gradient reach.** No detach and no stop-gradient anywhere; the bank's K/V cache is built with grad.

| term | reaches |
|---|---|
| L_ans | coda blocks 10–11, ln_f, wte (tied head), β^𝒟. Through V_t and every bank entry s < t: core blocks 6–9 at all passes ≤ t, β, γ, g, q / scratch init. Through V_0: prelude blocks 0–5 and wpe. |
| L_rec slot | R; through V_t: the core at passes ≤ t, then the prelude, wpe and wte. **Not** the coda. |
| L_rec text | the aux path (prelude blocks on the aux tokens, coda blocks, ln_f, wte, β^𝒟), plus everything L_ans reaches through V_t and 𝔅. |

**Gradflow gate:** per term, a nonzero gradient on prelude, core, coda, β, γ, g and R, where applicable (`scratch/gradflow_check.py`).

---

## 3. Goodhart table

| route | status in VB | why / test |
|---|---|---|
| **G1** pass-1 precompute | **closed on nested by capacity** | Pass 1 = 12 layers, and DIRECT at 12 layers fails from n* = 3 [EMPIRICAL, note 170-3]; depth-bounded in general (L1). **GSM8K: open for what 12 layers solve** (m ≤ 2; 0.109 at m ≥ 3); VB-8 tightens this. Test: ONSET. |
| **G2** recompute from the input | **closed by depth, not by structure** | V_0 is readable at every pass, as in CODI. A pass that reads only V_0 writes depth-10 entries, so recompute-only = the one-pass class (L1-c). Unlike Huginn, the input is never re-added to the residual. Test: ONSET, SKIP-DELAY. |
| **G3** decoder-side solving | **bounded** | The coda has 2 layers and no problem tokens; at K′ = 0 it is an 8-layer path. Test: K′ 0. |
| **G4** literal copying | **not a bypass** | No copying is needed (V_0 is global). Copies have depth 6, and the coda cannot finish from them (G3). |
| **G5** pass 1 writes everything, later passes copy or idle (made *easier* by a global memory) | **closed by L1** | Copying is unnecessary, but writing the answer at pass 1 needs depth > 12 for n ≥ 3. Idle passes **after** completion are allowed: that is C6. Idle passes **before** completion leave a critical path too short (L1). Test: SKIP-DELAY. |
| **G6** pass-index readout | **closed by content** | No pass token or pass position. The targets are item content, which an index cannot produce. The slot head is linear; the text aux has 2 contact layers, so it can at most do one step itself (§4 iii). |
| **G7** one-sided dynamics targets | **none used** | Every target is a two-sided CE on task content. |
| **G8** early-pass sabotage | **no incentive** | No term compares K with K − 1 or rewards wrong early answers. Early passes are unsupervised by L_ans and supervised toward *correct* partial content by L_rec. |
| **G9** (new) parallel fan-in: passes compute independent shallow features, and the coda combines them | **closed by depth asymptotically; tested** | Without chaining, every entry has depth ≤ 10 and the answer is in the one-pass class (L1-c). At finite n, width can partly stand in for depth. Test: SKIP-DELAY. |
| **G10** (new) carry-only use: the bank is ignored and the model becomes an RNN over V_{t−1} | **open under CARRY** (it violates the user's intent, not C1) | Test: BANK-ABLATE (mask versions 1..t−2 at every pass). PURE closes it by construction. |
| **G11** (new) count-keyed schedule (τ = n invites "answer at pass n") | **partly open** | Under NOPASS plus the recode target, the cheap algorithm is "reduce until done". Test: DEPTH at n 9–16, and REACH(K′) beyond 8. |

---

## 4. Math

### (i) Depth lemma for a bounded core reading a growing memory

**Model.** Each layer is an attention + MLP block over the vectors it is given, with log precision (the standard idealisation). Per-pass fan-in is unbounded: pass t may attend to every entry of 𝔅_{t−1}. The bound is therefore stated for **critical-path depth**, not for how many earlier results a pass can combine.

**Lemma L1 (critical path) [PROVED, structural induction].**
* Let a_t^{(ℓ)} be the longest path, in layers from the input embeddings, to the pass-t state entering core layer ℓ = 0..d. For PURE, a_t^{(0)} = 0 and a_t^{(ℓ+1)} = 1 + max(a_t^{(ℓ)}, max_{s<t} depth(V_s)), with depth(V_0) = P. So depth(V_1) = P + d and, by induction, **depth(V_t) = P + t·d**. CARRY (a_t^{(0)} = depth(V_{t−1})) gives the same.
* With the coda: **anything output after K passes is computed by a masked, weight-tied transformer of depth P + K·d + c over ≤ (K + 1)·T + L_dec positions** (unrolled, pass s's slots are extra positions frozen after their last layer).

**L1-b (same-layer KV memory adds no depth) [PROVED].** If pass t's layer ℓ read only the bank's layer-ℓ states (the KV-cache pattern), then a_t^{(ℓ+1)} = 1 + max(a_t^{(ℓ)}, P + ℓ) and depth(V_t) = P + d for every t. Iteration then adds nothing without an input-feedback chain. That is why CODI's depth runs through its single fed-back vector, and why VB stores outputs.

**L1-c (no progress without chaining) [PROVED].** A pass that reads no V_s with s ≥ 1 (it recomputes from V_0, or idles) writes entries of depth ≤ P + d. A model none of whose passes read later versions computes, at any K, a function of depth ≤ P + d + c: the one-pass class. This covers G2, G5 and G9.

**Corollary C1-asym [DERIVED from cited theorems, conditional].**
* Assume TC⁰ ≠ NC¹. Feng et al. (2305.15408) show that bounded-depth, log-precision, poly(n)-width transformers cannot solve Arithmetic(n, p) for all n (their direct-answer theorem; Literature to confirm its number). Their argument rests on the TC⁰ upper bound (Merrill & Sabharwal 2207.00729, verified), which also covers fixed non-causal masks and weight tying [ASSUMED from the simulation's generality].
* So for every fixed (P, d, c, K) there is an n̄(P + Kd + c) above which VB fails: **K must grow with n.**
* With unbounded fan-in the requirement is the expression's *critical path*, not n. Feng trees have mean height 2.67 / 5.04 / 7.03 at n = 3 / 8 / 16 (max 12 at n = 16; research/heights.py, 4,000 items per n).
* **Precondition check:** Chen, Peng & Wu (2412.02975, verified) give an unconditional depth bound, but for composing functions over a domain that grows with the input. Feng's domain is Z₁₁, so it does **not** transfer.

**Concrete instantiation [EMPIRICAL].**
* DIRECT (12 layers, 48k) is ≥ 0.98 at n ≤ 2 (height ≤ 2) and 0.64–0.75 at n = 3 (note 170-3): a trained depth-12 pass reliably closes about 2 levels. Predictions: VB-12 at K′ 1 (depth 12) ≤ 0.35 pooled over n 3..8; VB-8 at K′ 1 (depth 8) ≤ 0.8 at n = 2.
* **The gap:** no unconditional finite-n lower bound for 768-wide softmax transformers is known. The n(d) numbers are learnability facts at our budget, not impossibility results.

### (ii) Capacity condition

* **Storage never binds.** E_t needs log₂20 ≈ 4.3 bits per slot against 768 floats, and the bank keeps every version, so nothing has to survive by being rewritten (unlike DeltaMem). GSM8K intermediates (≤ ~7 digits ≈ 23 bits) fit the same way.
* **The binding condition is per-pass compute:** one Feng reduction must be computable in d layers given V_{t−1}.
* **Construction sketch [DERIVED, idealised attention], d = 3 suffices:**
  * Layer 1 (CARRY: own state is already in the carry): read the nearest *alive* (non-VOID) left and right neighbours in V_{t−1}. This is a position-monotone score restricted to alive keys; it relies on GPT-2's linearly decodable positions.
  * Layer 2: the evaluability flag, from the 2-hop context and the precedence rule (a 5-symbol window on the current expression), plus a prefix OR over evaluable ops to the left ("am I the leftmost?").
  * Layer 3: the MLP table lookup o(a, b) (4 × 11 × 11 = 484 entries) and the status update. Consumed slots see the decision through the bank and set VOID.
* So d = 4 has one layer of slack. **Named gaps:** softmax sharpness over ≤ 60 positions (an error ≈ e^{−βΔ}), and learnability. VB-8 (d = 2) fails the sketch and would need τ = 2n, another reason it is stage 2.
* On GSM8K the condition is *one equation per 4 layers*, which is unproven for multi-digit arithmetic (risk 3, §8).

### (iii) What the objective's optimum guarantees

* **Zero L_rec(slot) ⇒ presence [PROVED].** V_t linearly decodes E_t for every t ≤ n.
* **Zero L_rec ⇒ chaining [PROVED, conditional on (i)'s calibration].**
  * Let r₁ be the largest t for which x ↦ E_t is in the one-pass class 𝒞(P + d) (≈ 2 by DIRECT).
  * For t > r₁, pass t's output cannot be computed from V_0 alone (L1-c). So it must read some V_s with s ≥ 1, and there is a dependency chain whose length grows with n.
* **What it cannot pin:** *which* earlier version a pass uses (pass t could compute E_t from E_{t−2}, two reductions in one pass, which is still serial use); learnability; beyond-training depth.
* **Zero L_ans alone** forces the same chaining in principle, by L1 + C1-asym. The recode term is there for learnability (§175's lesson: per-step targets are load-bearing).
* **Text aux:** a decoder with 2 contact layers can solve at most one step itself [ASSUMED], so the core may lag by one pass. It must still advance one step per pass: the aux at pass t + 1 can add at most one step to what the bank already holds.

---

## 5. The LOSES-ONE-PASS risk

**Definition.**
* **Nested:** pooled accuracy over n 3..8 at K′ = 12 is ≤ 0.5, *or* ITER-GAIN < 0.2 (iteration not learned in 48k). The model falls back to its one-pass solution: about 0.30 pooled, i.e. K0 / DIRECT.
* **GSM8K:** ACC ≤ A − T = 0.119, with ITER-GAIN ≤ T.

**Internal evidence:** narrowing the per-pass channel slows entry. Against §198(a) cached INJECT (entry(0.5) 18k, DEEP, DEPTH 0.570): §198(b) MEM-HIGH entered at 24k (DEEP, DEPTH 0.454); §198(c) READ-SLOTS (8 slots) entered at 33k and was **not DEEP** (n 8 0.781). §171 READ-SLOTS without the aux never used its passes, and in §208 QO one pass did everything.

**Quantification [CONJECTURE]:**
* **VB-12 CARRY + slot recode:** P(entry by 48k) ≈ 0.65, P(DEEP) ≈ 0.45, so **P(LOSES-ONE-PASS) ≈ 0.5 per seed.**
* **PURE:** P(DEEP) ≈ 0.3. It is closest to READ-SLOTS, but its slots are T wide and read *outputs*.
* **Text recode:** P(DEEP) ≈ 0.35.
* **GSM8K VB-12:** the K′ 1 floor ≈ the pretrained one-pass (≈ 0.14), minus the layer-skip decoder and looping drift (0–0.03). So P(ACC ≤ 0.119) ≈ 0.25–0.3.
* **GSM8K VB-8:** P(ACC ≤ 0.119) ≈ 0.6.

**Mitigations, all in the default recipe unless marked:**
1. **A function-preserving pass 1** (CARRY; gate G1), so the fallback is DIRECT-level, not chance.
2. **A dense, ungameable slot recode target at every pass** (nested): a stronger anti-recompute signal than INJECT's gameable 12-layer aux.
3. **Curricula:** gated n on nested; recurrences on GSM8K (McLeish).
4. **The RMS reset and the β = −5 bank fade-in:** the bank carries about 1–10% of attention mass from step 0, so its gradient is not e⁻⁸-small.
5. *Fallback:* warm-start from a converged §193 trunk (nested) or the §202 CoT trunk (GSM8K; a low prior, since §202's DeltaMem arm did not gain). Both are C4-legal.
6. *Fallback:* τ = 2n (two passes per reduction), if per-pass capacity binds (the slot-recode CE plateaus on the step-t classes after entry).

---

## 6. Readings, predictions, labels (nested stage 1)

**Readings (all within one model, comparator-free; local fp32 eval at a fixed eval_bs and the training card's class):**

* **ITER-GAIN_N** = ACC(K′ 12) − max(ACC(K′ 0), ACC(K′ 1)), pooled over n 3..8. Bar: **≥ 0.50**. The INJECT family's gap is about 0.88; DIRECT-like one-pass is about 0.30.
* **ONSET:**
  * onset_i = min{t ∈ 0..16 : the readout is correct at every K′ ∈ [t, 16]}; never correct gives onset_i = ∞.
  * **SERIAL-PROFILE** requires all three:
    * coverage (finite onset) ≥ 0.8 at every n 3..8 (never-reached counts as failure);
    * the OLS slope of median onset on n ≥ 0.6;
    * med(8) − med(3) ≥ 3.
  * **ONESHOT** = the range of median onset over n is ≤ 1.
* **SKIP-DELAY** (step-specific use; it replaces SERIAL-NEST's donor swap, which is trivial when the bank is the only state):
  * Skip pass t = ⌈n/2⌉: no V_t is written, and CARRY continues from V_{t−1}.
  * **SERIAL-STEP** requires both: accuracy at K′ = n falls to ≤ 0.3 × intact, *and* accuracy at K′ = n + 1 recovers to ≥ 0.9 × intact (n 5..8 pooled).
  * Precompute, fan-in and recompute all predict no one-pass delay.
* **Descriptive:** DEPTH (n 9–12 at K′ = n + 2); REACH(K′) = max n with acc ≥ 0.8; STABLE (K′ 24 − K′ 12); BANK-ABLATE (versions 1..t−2 masked at every pass); slot-recode accuracy by t; entry steps (the pinned basin-entry flag).

**Labels:**
* **VB-C1** = DEEP (≥ 0.90 at every n 3..8, K′ 12) ∧ ITER-GAIN_N ∧ SERIAL-PROFILE ∧ SERIAL-STEP.
* **TWO-STAGE** = ITER-GAIN with ONESHOT.
* **ONESHOT-PRECOMPUTE** = ONSET ≤ 2 at n = 8 together with DEEP. This **falsifies the depth premise** (§8).
* **NO-ENTRY:** the answer-CE bin is never ≤ 0.5. This is a failure.
* **LOSES-ONE-PASS** (§5).
* **CARRY-ONLY** (G10): BANK-ABLATE drop < 0.05 at every n. Descriptive for CARRY; it flags the user's constraint as unmet in practice.
* **COUNT-KEYED** (G11): DEEP but REACH(16) ≤ 8 and DEPTH < 0.5.

**OC block (one seed per cell):**
* P(VB-C1 | no serial use) < 0.01: ITER-GAIN ≥ 0.5 is unreachable when ACC(12) ≈ ACC(1) ± 0.02, and the median-onset slope over 1,000 items per n has an SE well under 0.1.
* P(VB-C1 | the model learns serial use) ≈ 0.8, since the recode targets pin onset ≈ n. P(learns at one seed) ≈ 0.45.
* **A positive is strong evidence. A null is weak:** P(null | design capable) ≈ 0.55 from the seed lottery. A rate claim needs ≥ 3 seeds.

**Predictions (N1, VB-12 CARRY + slot recode) [CONJECTURE]:**
* **If DEEP:** K′ 0 / 1 pooled ≤ 0.2 / 0.35; median onset = n ± 1 (slope 0.8–1.2); SERIAL-STEP (P 0.8); DEPTH central 0.65 (0.45–0.85; INJECT 0.49–0.69), with **BEYOND-INJECT** if ≥ 0.75 (P 0.25); K′ 24 − K′ 12 ≥ −0.03 (P 0.6); BANK-ABLATE drop ≥ 0.2 at n ≥ 6 (P 0.5, a coin flip). Entry(0.5) 20–30k.
* **Censoring:** a run dying before 48k fails every reading; a never-reached onset is a coverage failure; an unreadable SKIP-DELAY means SERIAL-STEP is not met. Event-only values are reported beside each.

---

## 7. Experiments

### 7.1 Nested stage 1 (seed 1, RTX 3090, 48k × bs 64, n 1..8 train, §170 test set)

| cell | design | question | est. $ |
|---|---|---|---|
| **N1** `s2xx_vb12_carry_slot_s1` | VB-12, CARRY, slot recode, readiness window, curriculum | Does a bounded looped core over a pass-global bank learn serial use? (the main cell) | 1.3–1.8 |
| **N2** `s2xx_vb12_pure_slot_s1` | N1 with PURE | The user's strict form: is a pure attention memory (no carry) learnable? | 1.3–1.8 |
| **N3** `s2xx_vb12_carry_text_s1` | N1 with the text recode (NOPASS aux, HOLD) instead of the slot head | The GSM8K-transferable supervision | 1.6–2.2 |

* **Family central about $4.5–6, cap $8.** The drop order, if a G-TIME projection exceeds the cap, is N2, then N3.
* **Basis:** §198(a) cached INJECT took 41k s for 48k at $0.179/h ($2.15). VB-12 has about 0.6× of its launches, plus bank attention.
* **G-TIME** at step 200 per cell: projected ≤ $2.6.

**Gates (before launch):**
* **G1:** CARRY at K = 1 with β, β^𝒟 = −∞ and g = 1 reproduces plain GPT-2's answer logits on [EOT, problem, " ="] in fp64 (max |Δ| ≤ 1e-9); report Δ at β = −5.
* **G2:** gradflow per term (§2). **G3:** mem_check at K 12 on the longest n = 8 items, and the K′ 24 eval on n = 16.
* **G4:** the slot-target self-check (100k items). **G5 (N3):** no problem token in the aux prompt.
* **G6:** the plain path is untouched (a new module, boxes/vbank.py; a 1-step INJECT smoke). **G7:** Audit's diff review (decorators, return shapes, grad on the bank cache).

**Implementation pointers.**
* boxes/vbank.py: VBankN(g, P = 6, d = 4, mode) reuses `nested170.train_items / test_set / pad / NUM_IDS / AnnealGate`, the masked-K scheme of `kmatch_aux_loss`, and `deltamem.gpt_lower` for the prelude.
* `vb_block(g, i, h, bank_kv_i, bias)` = `deltamem._block` with keys / values = cat(bank, own) and a float `attn_mask` (B, H, T, N + T), as `rs_run` builds its boolean mask. Bank K/V are cached per (version, layer), with grad.
* The eval needs a K′ sweep 0..16 and 24 per item (ONSET), skip-pass hooks (SKIP-DELAY) and bank-mask hooks (BANK-ABLATE).

### 7.2 Next steps by outcome (not registered)

* **VB-C1 in any cell:** seed 2, and VB-PAR (a parallel-innermost recode with τ = tree height: a C2 bet, since mean heights at n 9–12 are 5.3–6.2, within K′ 12). Then GSM8K with the passing configuration.
* **All NO-ENTRY / LOSES-ONE-PASS:** one warm-start cell (§5 item 5, about $1.5) before abandoning. **ONESHOT-PRECOMPUTE:** stop the family (§8).

### 7.3 GSM8K follow-up (GSM8K-Aug, 24k × bs 64, §200 data, test and eval; seed 1; 3090)

* **Cell `gsm_vb12_<carry|pure>_s1`:** a version is the question positions plus 16 scratch slots. The text recode "<<e_t>>" runs at pass t ≤ min(m, K); the readiness-window answer "### a EOS" at t ∈ [m, K_i], with K_i = min(m + U{0..2}, 10); plus a curriculum of recurrences.
* **Supervision:** final answers plus the dataset's CoT equations (the text-CoT teacher's outputs), and m is read from the CoT, as in §200. This is C4-legal; there are no hand-built states.
* **Optional second cell:** `gsm_vb8_*` (one-pass depth 8, the tight G1 closure). **Cost:** 0.8–1.2× §200 A ($0.9) each, so **≤ $1.5 per cell, ≤ $3 for both.**
* **Decisive readings, as §208 for comparability:** ITER-GAIN = ACC(10) − max(ACC(K′ 0), ACC(K′ 1)) ≥ 0.028 pooled ∧ ≥ 0.034 at m ≥ 3; SERIAL-PROFILE by note 208-2's operational definition (strata 1, 2, 3, 4, 5+); SKIP-DELAY at K′ = m (descriptive).
* **Labels:** **VB-SERIAL** (ITER-GAIN ∧ SERIAL-PROFILE), TWO-STAGE, PASS1-PRECOMPUTE (no ITER-GAIN; pass-1 answers on m ≤ 2 are expected and do not fire it by themselves), LOSES-ONE-PASS. ACC vs A (LIFTS / MATCHES / HURTS) and vs CODI4 0.189 are reported.
* **OC:** P(ITER-GAIN | none) ≈ 0.02 (§208's computation). P(| true gain 0.04) ≈ 0.5.
* **Prior [CONJECTURE]:** ACC central 0.14 (0.10–0.19); P(ITER-GAIN) ≈ 0.35; P(VB-SERIAL) ≈ 0.2; P(LOSES-ONE-PASS) ≈ 0.25–0.3. P(ITER-GAIN) is about §208's prior (0.36), which failed. The case for doing better: another pass here is *more layers on the same residual stream*, not a re-derivation through a 49k-float memory.

---

## 8. Top 3 risks and what falsifies the design

1. **LOSES-ONE-PASS / NO-ENTRY:** the looped 4-block core over a growing bank is not learned within the budget (precedent §198(b)/(c)). *Falsifier at this budget:* N1–N3 all fail this way, and so does the warm-start fallback.
2. **The pass-global memory is decorative (G10) or unlearnable:** CARRY reduces to an RNN and PURE never enters, so the user's constraint is met in form, not in function. *Falsifier for "pass-global matters":* N1 is VB-C1 with a BANK-ABLATE drop < 0.05 at every n, *and* N2 is not DEEP.
3. **GSM8K per-pass capacity and an open G1:** VB-12's pass 1 is the full pretrained model, so §208's PASS1-PRECOMPUTE can recur on m ≤ 2, and 4 layers may not hold one multi-digit equation. *Falsifier:* GSM-VB12 with ONSET = 1 in every stratum and ITER-GAIN ≤ 0 while nested N3 (the same supervision type) was VB-C1. Serial use then does not transfer to natural arithmetic at 4 layers per pass; only VB-8 / τ = 2m remain.

**What would falsify the design's premise outright:** **ONESHOT-PRECOMPUTE on nested.** That is, the answer is correct from pass ≤ 2 at n = 8 (depth ≤ 16). It would mean the depth calibration (12 layers ≈ 2 levels) is badly wrong for this architecture, so the bounded core does not force iteration.

---

**Verified references** (arXiv abstract pages / Semantic Scholar, 2026-10-15):
* 2002.09402: Fan et al., Feedback Transformer;
* 2511.07384: McLeish et al., retrofitted recurrence;
* 2502.17416: Saunshi et al., looped transformers;
* 2207.00729: Merrill & Sabharwal;
* 2412.02975: Chen, Peng & Wu (precondition fails here, §4).

Taken from the brief and not re-checked: 2305.15408, 2502.05171, 2609.01924.

**Scratch:** research/heights.py in this scratchpad (the tree-height and length table used in §1.2 and §4).
