# ANSWER-BOTTLENECK on GSM8K-Aug: design + derivation DRAFT (2026-10-15 10:33 EDT)

Opus subagent for Theory, on Audit's lead after §207 (GSM-SWAP). **Not registered.** It needs Theory's review, Audit's review and **the user's OK** (a peer's OK is not that). No git, no theory/score_*.py run, no boxes/ edit.
- **Scripts:** theory/ansbottle/pairs_calc.py → theory/ansbottle/pairs_calc.json (CPU, numpy + tiktoken, Idle priority, one thread, 31 s; no model loaded).
- **Inputs read:** THEORY.md §200 / §202 (note 202-1) / §204 / §206 (note 206-2) / §207 (note 207-1); docs/THEORY_STEPFACTOR_DRAFT_2026-10-14.md (D1, D4, D5b); docs/THEORY_LOTUS_CONTRAST_2026-10-14.md (A-CAP); docs/LIT_ANSWER_BOTTLENECK_2026-10-15.md (Literature, f4c7c197); boxes/gsm8k_stepaux.py, prosqa168s2.py, deltamem.py, deltas5.py, gsm8k_data.py; A / W / F-P / J1 / B eval.json, args.json and A's log.jsonl.
- **Folded in (coordinator, 4 messages):** COPY-THE-QUESTION as the named reading **COPIES-LITERALS** (never success); R_x template pairs for an exact GSM8K hybrid; the last-pass swap validator; the read-energy split by channel; the literal probe as the primary copier detector with a LITERAL-ONLY-SWAP control; the kept rule; the 3-way label set with the t = 1 vs t = 2 contrast.

**Tags.** PROVED (code) = by inspection of the cited lines. PROVED = full argument here. DERIVED|A… = follows from named, unverified assumptions. CONJECTURE. DATA / DATA, POST-HOC = read from landed files. calc = arithmetic here (pairs_calc.json).

---

## 0. Summary

1. **Closes §207's decoder bypass (PROVED, code):** the decoder reads [EOT, "###"] + answer; its first token (the whole answer for 90.6% of items) sees M_K only through two fixed-query reads.
2. **Not the think-side bypass (DERIVED):** passes still read q, so §207's mode can move upstream as RECOMPUTE-INTO-MEMORY or PASS1-PRECOMPUTE.
3. **Serial pressure is unchanged to first order (A-1F):** pass 1 inherits the decoder's one forward. The gain is attribution: ACC above a₁ (≲ 0.13) must be multi-pass.
4. **Exact GSM8K hybrids from R_x template pairs (calc):** 0.61 strict-kept at t = 1, 0.16 at t = 2; 310–1,190 readable TRAIN-REDRAW pairs; test pool unreadable.
5. **3-way TEMPLATE-SWAP label** (hybrid / donor / own at t = 1, 2), plus COPIES-LITERALS (probe + ANS-LIN + LITERAL-ONLY-SWAP), a SWAP(K−1) validator, the energy split, K′1-RESCALED.
6. **Prior:** ACC 0.095 vs A 0.147 (HURTS 0.6); label RECOMPUTE-INTO-MEMORY (0.45); SERIAL-CARRY 0.12. F-P got pass-1 values under a stronger demand.
7. **Base: A's recipe** (24k, vs A). W's better direct answering (K′0 0.111) feeds the upstream bypass, and its 12k leaves 6k steps after the γ ramp.
8. **Cost $2.6 central / $3.8 high** (F-P realised $2.56). OC at one seed: P(pass | null) ≈ 0.01, P(pass | effect) 0.25–0.5; a null is weak (LR ≈ 0.6).
9. **Main risk:** a likely upstream-recompute null. The derived next lever is QUESTION-ONCE (q at pass 1 only), which is its own derivation.

---

## 1. Facts

### 1.1 Code (PROVED (code))
* **F1. Today's answer decoder sees the question.** `ProsMem.dec_loss` builds [EOT] + pre + a1 + [EOS], with pre = question + "\n" and a1 = "### " + answer (prosqa168s2.py:138–144; gsm8k_data.encode :122–126). Labels cover a1 + EOS.
  - Eval is greedy after [EOT] + pre + [ANS] (gsm8k_stepaux.py:337–341).
  - fwd_read runs all 12 blocks with the read at block-6 entry (deltas5.py:91–98).
* **F2. The read.** read_fixed returns x + fade·W_o([g_add·φ(q)ᵀS_add ‖ g_del·S_del·q̂]) at every position, with q from mem.ln(h_6) at that position (deltamem.py:158–176).
  - **W_o has a bias** (nn.Linear, :71–72), so the zero-state read is the constant fade·b_o.
* **F3. The writes.** u = (1−γ)·LN_6(h_6) + γ·LN_w(h_w), with h_w = the residual after block 11 ("top", :285). k, v and the gates come from u (:187–188).
  - Additive: S_add += Σ_tok φ(k)vᵀ (FAST_ADD, :194–198), with no decay.
  - Delta: S ← αS(I − βkkᵀ) + βvkᵀ (:205–211; WY chunks :226–248).
  - γ = 0 to 2k, then a linear ramp to 1 at 6k. Fade 0 → 1 over 500 steps (gsm8k_stepaux.py:398–399).
* **F4. Think passes.** The input [EOT, question] is identical every pass (prosqa168s2.py:121–122). --cache_lower shares blocks 0–5 across passes and asserts the identical input (:125, :128–130).
  - **K is fixed at 10 in training** (gsm8k_stepaux.py:240). K′ varies only at eval.
* **F5. The aux decoder is question-free and read-only.** Its prompt is [EOT, " Step", pass_tok(t), " :"] + the teacher-forced " <<e_t>>" (prosqa168s2.py:161–170). It goes through fwd_read only, so it never writes.
* **F6. The pair machinery.** gsm8k_data.parse_recomputable gives a template: question literals qvals (each unique), and per-equation operand grounding q / r / k, where r is the latest earlier result with that value (:168–212).
  - render(raw, tpl, newq) recomputes every equation exactly (:223–257).
  - draw_literal keeps the digit count, decimals and integrality (:268–280).
  - resample asserts train-only (:283–303). Eval-time pairs need a copy of its loop without that assert; nothing trains on them.

### 1.2 Data (DATA, POST-HOC unless calc)
* **A** (gsm_lat_aug_s1): acc 0.147.
  - K′ 0 / 1 / 2 / 4 / 10 / 16 = 0.083 / 0.125 / 0.131 / 0.137 / 0.147 / 0.136.
  - Deep (m ≥ 3) K′0 / 1 / 10 = 0.064 / 0.097 / 0.110.
  - fit_fix 0.655, fit_rs 0.175 (recall share 0.48).
  - answer_ce 2.87 at step 100, 1.22 at 2.1k, 0.448 at 24k (log.jsonl).
  - 1.82 s/step (3090, micro 4).
* **W:** acc 0.146; K′0 0.111, K′1 0.129; 1.10 s/step over 12k.
* **F-P:** acc 0.139; VALUE-LEAK ≈ 0.9 of VALUE-EXACT (pass-1 values); 1.27 s/step over 24k; realised about $2.56.
* **J1:** acc 0.138; K′1/K′10 0.64; DEEP-MULTIPASS +0.035; 1.64 s/step.
* **§207:** INPUT-FOLLOW(1) 0.87–0.92; CARRIED(1) 0.000–0.005; OWN-KEPT(2) 0.80–0.88 vs ZERO-STATE 0.20–0.51. INPUT-SWAP-THINK own is 0.13–0.45.
* **External no-CoT GPT-2** (25 epochs, Coconut's checkpoint_25 on our bench): 16.5 (docs/GSM8K_REPRO.md).
* **calc (pairs_calc.json):**
  - The test's most frequent answer ("5") covers **0.030**. The train mode ("20") covers 0.024 of the test.
  - An item-independent decoder's answer CE (test vs the train marginal): **5.74 nats per item = 1.85 per labelled token** (3.11 labelled tokens per item: "###", the answer tokens, EOS).
  - **90.6%** of test answers are one BPE token after "###" (8.1% two).
  - Recomputable test items: 570 (43.2%), with mean 2.82 used literals and mean m 3.03.

---

## 2. Mechanism

**Notation.** Think map Φ: (q, M) ↦ M′ with q identical on every pass; M_t = Φ(q, M_{t−1}) and M_0 = 0. The bottleneck decoder is Ψ: M_K ↦ y. Today's decoder is Ψ′: (q, M_K) ↦ y.

**B1. The decoder's only item-dependent input is M_K (PROVED, code).**
* The decoder ids are [EOT, ANS] + the answer prefix.
* Positions 0 and 1 hold item-independent tokens, and blocks 0–5 are causal. So h_6 at positions 0 and 1, and the queries mem.ln(h_6) built from it, are the same for every item.
* The first answer token is the argmax at position 1. After block 6 it depends only on r₀ = R(q₀; M_K) and r₁ = R(q₁; M_K): 12 heads × 64 dims each, at most 1,536 floats.
* Later answer tokens add the answer prefix, an item-dependent query. That prefix is y's own tokens, not q.
* So I(ŷ; q | M_K) = 0. The answer must cross the memory.
* By calc, the first token fixes the answer for 90.6% of test items.

**B2. K′0 is an item-independent constant (PROVED).**
* With M = 0, both reads are the bias constant (F2). The decoder emits one string c for every item.
* So acc(K′0) = P_test(y = c) ≤ **0.030**, and answer CE ≥ 1.85 nats per labelled token (calc).
* This is "chance by construction". THINK-NECESSARY's K′0 clause and LOOP-NOT-FORMED become vacuous, and **are not evidence**.

**B3. The decoder-visible capacity is two fixed addresses, not A-CAP's table (PROVED (code) + DERIVED|near-orthogonal keys).**
* LOTUS-contrast A-CAP's third assumption, that the decoder's queries are item-dependent and so content-addressing, **fails here** for the first answer token.
* Per delta head, S_del·q̂_p is the value at one soft slot along q̂_p. Per additive head, φ(q_p)ᵀS_add is a positively weighted bag.
* So the first token sees about 12 delta slots plus 12 bags of 64 floats.
* **The answer needs ≤ 23 bits** (max; mean 7.6, from LOTUS contrast §2). One slot suffices.
* **A question copy fits too:** about 17 bits of used literals (2.82 × 6.0) plus about 29 bits of plan on average, roughly 150 at most, against ≥ 1.5k floats.
* So capacity does **not** exclude COPIES-LITERALS (answering Literature's question). It only forces any copy to be **pre-assembled at the write**: gathered by attention in the think pass and written along q̂₀ / q̂₁.

**B4. What the think passes must compute (PROVED + DERIVED).**
* By B1, M_K must hold y, or content from which blocks 6–11 at two positions compute y, at the two fixed addresses.
* **A-DEC6 (assumption):** six blocks over two positions, with operands only as vector codes (not tokens), finish at most one arithmetic level. For comparison, one 12-block forward finishes about one level with operands as tokens (A-OP).
* Under A-DEC6, the decoder's self-computation drops from s ≥ 2 (D4, today) to s_B ≤ 1. The cheapest decoder is a register read-out (s_B = 0).
* So the think passes must deliver y, or y less its last level, into a register.

**B5. The one-pass ceiling, and the bypass moving upstream (DERIVED|A-NEST, A-1F).**
* **(a) PROVED.** M_1 = Φ(q, 0) is a one-forward function of q: the pass-1 read is the bias constant. Any route whose answer content after pass t is a function of q alone yields accuracy ≤ a₁, the one-pass bottleneck ceiling (pass 1, then Ψ).
  - **PASS1-PRECOMPUTE:** M_1 holds y, and later passes hold it.
  - **RECOMPUTE-INTO-MEMORY:** every pass rewrites Φ's answer from q, a fixed point.
  - **COPIES-LITERALS:** M holds q's content, and Ψ computes.
  - So **ACC > a₁ ⇒ some pass uses item content written by an earlier pass** (composition). This is the bottleneck's real gain: attribution.
* **(b) A-NEST.** Today's decoder can approximately emulate Ψ by ignoring q. So the bottleneck's one-pass class sits inside today's K′1 class: pass 1, then a 12-block decoder with q.
  - At equal budget: **a₁ ≲ K′1(A / W) = 0.125 / 0.129**.
  - Beside it: the external no-CoT GPT-2 at 25 epochs, 0.165; today's direct path, K′0 0.07–0.11.
* **(c) A-1F, first-order serial pressure.** Today, D4's decoder finishes s ≈ 2 levels from q. Under the bottleneck, pass 1 has the same 12-block forward over q and can do the same s levels before the write.
  - Passes 2..K face the same requirement as before: produce what one forward cannot.
  - **The bottleneck removes the decoder's competing route and creates an identical one in pass 1.**
  - The second-order change is real: the loop must now finish the **last** level too (s_B ≤ 1 instead of s ≈ 2), so a serial loop's ratios move to D4's s = 0 row (pooled K′1/K′10 0.13–0.31; deep 0.0–0.38). That helps only if a serial solution is found at all.
* **(d) DATA prior.** §204's F-P put value pressure on **every** pass (v_t from M_t) and still got VALUE-LEAK ≈ 0.9: the values were written at pass 1. The bottleneck puts value pressure on **one** value, y, at pass K, a weaker serial demand.
  - So the DATA-backed central is the upstream bypass (CONJECTURE + DATA).

**B6. The bottleneck creates a K′ truncation artefact (PROVED at γ = 0; DERIVED at γ = 1).**
* At γ = 0, S_add(t) = t·A(q) exactly (D5). The decoder is trained only on S_K, and the question can no longer anchor it. So K′ < K reads an additive state K/K′ times too small.
* A PASS1 model can then **fail** K′1 on scale alone, and spuriously pass THINK-NECESSARY.
* **Guard:** K′1-RESCALED, the decoder on (K·S_add(1), S_del(1)).
  - Exact at γ = 0 for the additive part: K·S_add(1) = S_add(K).
  - The delta part is the pass-1 state, so it is a lower bound when the delta transient matters.
* THINK-NECESSARY is credited only if **both** K′1 and K′1-RESCALED are ≤ 0.5 × K′10.
* The swap readings (§3) always decode after K = 10 passes, so they avoid the artefact.

**B7. The open-loop window copies by construction (PROVED, code).**
* For steps ≤ 2000 (γ = 0), every write comes from LN_6(h_6), computed by blocks 0–5 before the read. The K passes write identical sequences.
* So M_K is a deterministic function of q's block-6 features: a copy.
* The think-pass blocks 6–11 get **exactly zero** gradient: u's h_w weight is 0, and the final hidden is discarded (prosqa168s2.py:126 takes [1]).
* So for 2k steps the model is "6-block encoder → fixed-query pooling → 6-block decoder on 2 positions". COPIES-LITERALS is the forced regime there.
* Predicted signature: answer CE stays near chance through 2k and drops during the γ ramp (§9 P-B11).

**B8. The clock carries no item information to the decoder; question length does (PROVED + DERIVED).**
* K = 10 for every training item (F4), so the pass index is a constant at the decoder: I(y; K) = 0. A clock can only help the think passes schedule (PASSK-REGISTER, §6).
* But the additive read's magnitude is Σ over the question's tokens, so it encodes question length L, an item-specific scalar correlated with m and y.
* So the probe floors include a length-only floor (§3.5).

**Relation to D4.** Today s ≥ 2: A's deep K′1/K′10 0.88 exceeds every s ≤ 1 serial model. The bottleneck sets s_B ≤ 1 by architecture, B4. D4's prediction for a strictly serial loop with s = 0 is pooled K′1/K′10 0.13–0.31 and deep ≤ 0.38. For the upstream bypass it is ≈ 1. **The K′ profile then discriminates the loop, not the decoder**, given the B6 guard.

**Not proposed here: QUESTION-ONCE (derived next lever).** Feed [EOT, q] at pass 1 and a fixed input at passes ≥ 2.
* Then passes ≥ 2 have no item input except M. RECOMPUTE-INTO-MEMORY becomes impossible (PROVED by the same B1 argument applied to Φ).
* What remains is PASS1-PRECOMPUTE, bounded by a₁, and serial state updates.
* It breaks cache_lower's identical-input premise and is a second change. It needs its own derivation, and is the natural follow-up if this cell reads RECOMPUTE-INTO-MEMORY.

---

## 3. Re-derived §207 readings under the bottleneck

**3.1 What each §207 arm becomes (PROVED, B1).** Write SWAP(t; a → b) for passes 1..t on item a, passes t+1..K on item b, then Ψ, which reads no item.

| §207 arm | definition there | under the bottleneck |
|---|---|---|
| STATE-SWAP(t) | 1..t on x_d, t+1..K on x_r, decoder x_r | SWAP(t; d → r) |
| INPUT-SWAP(k) | 1..k on x_r, k+1..K on x_d, decoder x_d | SWAP(k; r → d): **the same arm with roles exchanged** |
| INPUT-SWAP-THINK | as INPUT-SWAP, decoder keeps x_r | ≡ INPUT-SWAP(k): the decoder reads no text |
| ZERO-STATE (K′ 0) | decoder x_r with M = 0 | the constant c; own-kept = P(c = y_r) ≤ 0.030 |
| a STATE-SWAP at t = K | — | ≡ the clean run on x_d: **an identity gate** (P(y_d) = 1 on donor-correct, bitwise) |

* So §207's two families collapse into one, SWAP(t), read as P(y_pre) = the answer of the item the first t passes saw, and P(y_post).
* **§207's RECOMPUTES-FROM-INPUT** (OWN-KEPT(2) ≥ 0.8 and INPUT-FOLLOW(1) ≥ 0.8) becomes P(y_post | t) ≥ 0.8 at t = 2 and t = 1. This is **UPSTREAM-RECOMPUTE**, the same mode moved into the think passes.
* **CARRIED(t)** becomes P(y_pre | t). 207-1's SERIAL-CARRY / PASS1-CARRY contrast is CARRIED(2) − CARRIED(1).
  - On random donor pairs, a serial model's P(y_pre | t) is ≈ 0 unless m_pre ≤ t, so that contrast is weak.
  - **That is why the template pairs (3.2) are primary.** The random-donor arms are kept for comparability with §207 (descriptive), with §207's recipients (clean-correct test items, rng 20701, readable if ≥ 100).
* **"Presence ≠ use" (§207's zero 0.20–0.51 vs swapped 0.80–0.88) cannot recur:** zero memory gives c by B2. Presence and use are no longer separable by those arms, and the template arms replace them.

**3.2 TEMPLATE-SWAP: exact hybrids from R_x pairs (PROVED as a construction; calc for counts).**
* **Pairs.** x_r and x_d are two R_x instances of **one** recomputable template: same equations, operators and grounding; different used literals.
* **The hybrid after a swap at t (step schedule, matching the aux binding of step s to pass s):** v_s^hyb = v_s^d for s ≤ t. For s > t, evaluate e_s with x_r's literals, the constants, and r-operands taken from v^hyb. Then y_hyb(t) = v_m^hyb.
  - The level schedule (every step of level ≤ t done by pass t) goes beside it. It agrees with the step schedule on 0.80 of pairs at t = 1.
  - A serial carrier gives y_hyb. Recomputing from the post-swap question gives y_own (= y_r). A pass-1 answer or copy, held, gives y_don.
* **Kept rule (Audit):** y_hyb ∉ {y_own, y_don} with y_own ≠ y_don. **Strict** adds that y_hyb renders as a GSM8K answer (≥ 0, ≤ 2 decimals). Strict is the primary set.
* **Kept fractions (calc; 16 draws per template):**

  | pool | pairs | strict t = 1 | strict t = 2 | by m, strict t = 1 / t = 2 |
  |---|---|---|---|---|
  | TRAIN-REDRAW (3,000 sampled → 1,828 recomputable templates; both items fresh draws) | 23,758 | **0.608** | 0.164 | m2 0.72 / 0; m3 0.78 / 0.34; m4 0.79 / 0.68; m5+ 0.69 / 0.66 |
  | TEST (test recipient, redrawn donors) | 7,552 | 0.709 | 0.322 | m2 0.66 / 0; m3 0.73 / 0.39; m4 0.77 / 0.69 |

  - m = 1 pairs are never kept (y_hyb = y_don by definition).
  - Coincidences y_hyb = y_own are 4.9% (TRAIN-REDRAW, t = 1). Pairs strict at both t with y_hyb(1) ≠ y_hyb(2): 0.117.
* **Readability.** One pair per recipient: the first clean-correct instance as recipient, the first other clean-correct, strict-kept instance as donor.
  - **TEST pool:** expected 23–74 pairs at p_r 0.08–0.147 and p_d 0.06–0.175. A has 102 correct recomputable test items. **Unreadable (< 100), so descriptive only.**
  - **TRAIN-REDRAW:** expected **309 / 644 / 1,188** pairs from 1,828 templates at p 0.06 / 0.10 / 0.175 (A's fit_rs = 0.175). Pre-stated at 4,000 sampled templates (rng 20711): ≈ 410 / 860 / 1,580.
  - **Primary pool = TRAIN-REDRAW.** Recall is impossible by Theorem X (fresh literals; worth at most the collision probability). Template memorisation is present by Prop. S, equally for both arms, and A is evaluated on the identical instances.

**3.3 The 3-way label (on clean, strict-kept pairs; P_x(t) = P(answer = y_x) at swap t; readable if n ≥ 100 at t = 1 and n ≥ 50 at t = 2, on m ≥ 3 pairs).**
* **SERIAL-CARRY (the success label) needs all of:**
  - P_hyb(1) ≥ 0.30, and P_hyb(1) > max(P_don(1), P_own(1));
  - P_hyb(2) ≥ 0.20;
  - the SWAP(K−1) validator (3.4): P_don(K−1) ≥ 0.5;
  - COPIES-LITERALS clear (§6).
* **PASS1-PRECOMPUTE (not success):** P_don(1) ≥ 0.5, P_don(2) ≥ 0.5, |P_don(2) − P_don(1)| ≤ 0.15, the validator P_don(K−1) ≥ 0.5, **and** the LITERAL-ONLY-SWAP P^L_don(1) ≥ 0.5.
  - If instead P^L_own(1) ≥ 0.5, the reading is **COPIES-LITERALS (LITERAL-CARRY)**: literals were held and the computation came later.
* **RECOMPUTE-INTO-MEMORY (not success):** P_own(1) ≥ 0.5 and P_own(2) ≥ 0.5. A pass-1 precomputer is fixed after pass 1, while a re-copier or recomputer returns y_own at either t, which is Audit's separating contrast. Sub-labels:
  - **PASSK-REGISTER:** the K′ onset is late and does not depend on m (§6);
  - **MULTIPASS-RECOMPUTE:** DEEP-MULTIPASS also holds (iterative refinement without carry; a C2-type use, not C1).
* **MIXED:** otherwise. All of P_hyb, P_don and P_own are reported at t = 1, 2 and K−1, overall and for m = 2 / 3 / ≥ 4.
* **The accidental rate:** P(answer = a specific off-target number) is about 0.01–0.03. That is checked as P(answer = y_hyb) on **unswapped** recipients, the "HYB-NULL" control, reported beside.

**3.4 Validators and controls (Literature, Audit).**
* **SWAP(K)** = clean(x_d): a gate, PROVED (3.1).
* **SWAP(K−1)** on template pairs: nine donor passes, then one recipient write.
  - A serial carrier with m ≤ 9 has finished, and a held pass-1 answer survives: both give y_don.
  - A per-pass rewriter gives y_own if one pass overwrites (β ≈ 0.85–0.9 per pass in pass-invariant models, 199-E1). Otherwise it gives a mixture.
  - **SERIAL-CARRY and PASS1-PRECOMPUTE require P_don(K−1) ≥ 0.5** (the hold is real). It does not separate those two; it validates them.
* **LITERAL-ONLY-SWAP (L) (calc).** The donor is another recomputable train template with a different operator signature and the same number of used literals, carrying the recipient's used literal values in slot order. The draw is rejected unless R_x's distinctness rule and exact rendering hold.
  - The construction succeeds for **98.6%** of recipients within 300 candidates (median 2 tries); y_own ≠ y_don on 97.9%.
  - The hybrid at t = 1 is the recipient's chain with v_1 := the donor's v_1, under **A-SLOT**: a serial memory stores result s in a template-independent slot s.
  - Kept 0.795, strict 0.658, y_hyb = y_own 0.136.
  - **Predictions:**
    - literal-copy then compute later → y_own;
    - full pass-1 precompute (or a full copy, held) → y_don;
    - serial → y_hyb^L;
    - per-pass recompute → y_own.
  - Pairs per readable panel: the same yield model as 3.2, × 0.66.

**3.5 Read-energy split (descriptive; Literature's superposition caveat).** At the decoder's position-1 (q₁) and position-0 (q₀) reads after SWAP(t):
* **Additive:** S_add(K) = Σ_s W_s, a FAST_ADD sum, so r_add = r_add^pre + r_add^post exactly. The pre share is E_add = ⟨r_add^pre, r_add⟩ / ‖r_add‖².
  - **PROVED at γ = 0: E_add = t/K** (0.1 / 0.2 / 0.9 at t = 1 / 2 / K−1). The additive channel cannot hold donor content against later identical-input writes.
* **Delta:** given the realised post-swap (α, β, k, v), the recurrence is affine in the state: S_del(K) = S_del(t)·P_post + C_post (PROVED). Replaying the recorded gates on S = 0 gives C_post. The pre share is E_del = ⟨(S_del(K) − C_post)q̂, S_del(K)q̂⟩ / ‖S_del(K)q̂‖².
  - This attributes the state, not the causal origin of the post-swap writes (those are closed-loop).
* **Also reported:** the channel split ‖W_o[oa; 0]‖² vs ‖W_o[0; od]‖².
* **Expected (DERIVED from E_add = t/K):** PASS1 and SERIAL need E_del(1) ≥ 0.3, through a held delta slot (the progress-gated β ↓ of 199-E1's converged models). Recompute and re-copy give E_add(1) ≈ 0.1 and E_del(1) ≤ 0.2.
  - Consistency flag: a SERIAL or PASS1 label with E_del(1) < 0.2 is reported as "label without a held slot".

**3.6 Probes (L2 logistic, as §204 PROBE-LEAK).**
* **Features:** (r₀, r₁) at K = 10, pre-W_o with gains. By B1 this is everything the first answer token can use.
* **Fit and read:** 4,000 Aug-train items (fresh R_x draws where recomputable), λ by an inner 80 / 20 split, read on Aug valid.
* **Targets (first BPE token class, top 200 + OTHER):**
  - **LIT:** each used question literal, by slot;
  - **INT:** v_1 (m ≥ 2);
  - **ANS:** y.
* **Floors:**
  - M_0 (constant features → the majority class);
  - **y-only** (the target from a one-hot of y's class: what knowing the answer implies about a literal);
  - **length-only** (from log L, B8).
* **Derived:** LIT-EXCESS = acc_LIT − max(floors). **ANS-LIN** = acc_ANS / the decoder's own accuracy on the same items.
* Reported on r₁ alone (answer-only), on r₀ alone, and on both. **r₀ is shared with the aux prompt** (both start with EOT at position 0, so its query is identical; PROVED), so the aux-installed operand identities show up there.
* A's decoder-position reads are reported beside, descriptively; the positions are not equivalent.

**3.7 The route table (what each upstream route predicts; DERIVED from B1, B5, B6 and 3.5).**

| route | TPL-SWAP t = 1 / t = 2 | SWAP(K−1) | L-SWAP t = 1 | K′ profile | probes on (r₀, r₁) | E_del(1) |
|---|---|---|---|---|---|---|
| SERIAL-CARRY | y_hyb(1) / y_hyb(2) | y_don | y_hyb^L | onset rises with m; deep K′1/K′10 ≤ 0.4 | ANS-LIN high; INT above floor at M_t | ≥ 0.3 |
| PASS1-PRECOMPUTE (held) | y_don / y_don | y_don | y_don | flat from K′1-RESCALED | ANS-LIN high | ≥ 0.3 |
| RECOMPUTE-INTO-MEMORY | y_own / y_own | y_own or mixed | y_own | flat from K′1(-RESCALED) | ANS-LIN high | ≤ 0.2 |
| COPIES-LITERALS, re-copy each pass | y_own / y_own | mixed | y_own | flat | LIT-EXCESS high, ANS-LIN low | ≤ 0.2 |
| COPIES-LITERALS, copy once + decoder computes | y_don / y_don | y_don | y_don | flat | LIT-EXCESS high, ANS-LIN low | ≥ 0.3 |
| COPIES-LITERALS, LITERAL-CARRY (literals held, later passes compute) | y_don / y_don | y_don | **y_own** | flat | LIT-EXCESS high | ≥ 0.3 |
| STEPWISE-COPY (aux operand slots held; decoder executes) | y_hyb / y_hyb | y_don | y_hyb^L | flat | LIT-EXCESS high, **ANS-LIN low** | ≥ 0.3 |
| PASSK-REGISTER | y_own / y_own | y_own | y_own | ≈ 0 until a fixed t_c, then K′10 | ANS only at K′ ≥ t_c | ≤ 0.2 |

* **STEPWISE-COPY is the one route that fakes a hybrid.** The aux CE installs step-t operand identities at pass t (74% of aux CE is operands, STEPFACTOR §1.4). If those slots are held, the decoder could execute the chain.
* It is excluded by **ANS-LIN ≥ 0.5** (the answer is stored, not computed) and by A-DEC6. That is why SERIAL-CARRY requires COPIES-LITERALS to be clear.

---

## 4. Accuracy prior (CONJECTURE, with reasons)

* **What is lost:** A's direct path (K′0 0.083) and its decoder finishing s ≈ 2 levels from q. Of A's 0.147, B5(b) says about 0.125 is one-pass.
* **What replaces it:** pass 1's one forward over q, then the write, two fixed-query reads, and six blocks. A-NEST bounds that route at ≲ 0.125–0.13 at this budget, minus a routing efficiency η < 1.
  - η reflects that the answer must be computed at question positions with no "###" cue, written along fixed keys, and decoded from a vector.
* **Optimisation (B7):** for 2k steps only a 6-block encoder feeds the memory. The trunk's blocks 0–5 receive the whole answer gradient through h_6, which may tilt the trunk before γ ramps (risk 3).
* **Central ACC 0.095** (80% interval 0.05–0.14):
  - **HURTS** (≤ 0.119) P 0.60; **MATCHES** P 0.35; **LIFTS** (≥ 0.175) P 0.05.
  - LIFTS needs serial composition (B5(a)) at a level no GSM8K recipe has shown.
* **What would change the prior:** routing all answer gradient through M_K trains the read path hard. §207 showed today's decoder ignores memory content. If composition was blocked by read-side neglect rather than write-side difficulty, the bottleneck fixes that, and passes ≥ 2 become usable. That is the LIFTS route (P 0.05, plus part of MATCHES).

---

## 5. Base choice: A's recipe (raw GPT-2, 24k, vs A)

1. **One change against the richest comparator.** A has §200's fields, note 202-1's deep fields, §204's eval-only reference on a 3090 (bitwise to eval.json) and §207's swap. It gets the new eval-only fields on the 3070 at $0.
2. **W's trunk favours the Goodhart route.** W's direct accuracy (K′0 0.111 vs A's 0.083) is exactly the one-forward ability pass 1 needs for PASS1-PRECOMPUTE and RECOMPUTE-INTO-MEMORY (B5). A W base tilts the test toward the upstream bypass.
3. **Budget versus the γ schedule.** The bottleneck's difficulty is optimisation (B7: copy-by-construction to 2k, ramp to 6k). W's 12k leaves 6k steps after the ramp, A's 24k leaves 18k. A NEUTRAL or HURTS on W would be budget-confounded. "Faster, same ceiling" (§202) concerns the plain decoder's ceiling, which the bottleneck removes.
4. **Cost:** about +$1.3 over a W base. **Alternative:** a W base (12k, vs W, about $1.3 central) if the budget binds, with points 2 and 3 stated as its confounds.

---

## 6. Goodhart routes and detectors

| route | why it is reachable | detector (pre-stated) | counts as |
|---|---|---|---|
| **PASS1-PRECOMPUTE** (answer register at pass 1) | M_1 = Φ(q, 0) is a full forward over q (B5a); §204 F-P wrote values at pass 1 | TEMPLATE-SWAP P_don(1), P_don(2) ≥ 0.5, flat; SWAP(K−1); L-SWAP y_don; K′1-RESCALED ≈ K′10 | not success |
| **RECOMPUTE-INTO-MEMORY** (§207 upstream) | identical input every pass; a fixed point needs no read | P_own(1), P_own(2) ≥ 0.5; random-donor UPSTREAM-RECOMPUTE; E_add ≈ t/K, E_del low | not success |
| **PASSK-REGISTER** (answer at pass K via a clock) | the additive counter is ∝ t (D5; no RMS on this arm), so the think passes can schedule | ONSET(m) = the smallest K′ ∈ {1, 2, 4, 6, 8, 10} with acc_m ≥ 0.8 × acc_m(10), per m stratum with n ≥ 50. PASSK-REGISTER if ONSET is the same value ≥ 6 for every m ≤ 4 stratum (needs acc_by_m_kprime on the full grid) | not success |
| **COPIES-LITERALS** (Literature / Audit) | capacity allows it (B3); forced in the open-loop window (B7) | **DECODER-COMPUTE form:** LIT-EXCESS ≥ 0.10 (on r₀ ⊕ r₁; r₁ and r₀ beside) **and** ANS-LIN ≤ 0.5. **LITERAL-CARRY form:** TEMPLATE-SWAP P_don(1) ≥ 0.5 **and** L-SWAP P^L_own(1) ≥ 0.5 | never success |
| **STEPWISE-COPY** (the aux colluding) | the aux CE installs operand slots per pass; r₀ is a shared slot | ANS-LIN ≤ 0.5 together with P_hyb ≥ 0.30 (a hybrid without a stored answer) | not success |
| **Recall** (share 0.46–0.48 in A / F-P) | the think pass recognises a train question and writes its answer, i.e. PASS1 on fixed items | fit_fix − fit_rs (reported); every swap and probe runs on fresh R_x instances (Theorem X) | descriptive |
| **Clock as answer source** | none: K is constant at the decoder (B8, PROVED) | only question length leaks, covered by the length-only probe floor | n/a |
| **Truncation artefact** (a false THINK-NECESSARY) | B6 | THINK-NECESSARY credited only if K′1 and K′1-RESCALED are both ≤ 0.5 × K′10 | guard |

* Precedence when labels co-occur:
  - COPIES-LITERALS overrides any non-serial label, reported as e.g. "RECOMPUTE-INTO-MEMORY / COPIES-LITERALS".
  - SERIAL-CARRY with COPIES-LITERALS is **not** success (STEPWISE-COPY is possible).
* **An unevaluable detector is not clear.** The success label then fails (rule b).

---

## 7. Gradient flow (PROVED, code, for the bottleneck path; checked by gate G3)

* **L_ans (bottleneck)** reaches:
  - **(i) Through the decoder forward** on [EOT, ###, answer, EOS]: every trunk block at those positions, wte (EOT, ###, answer tokens; the LM head is tied), wpe[0:n], ln_f.
  - **(ii) Through the read:** mem.ln, qkv rows 0:D (queries), s_add, s_del, wo (weight + bias). fade is a buffer.
  - **(iii) Through S_K into every pass's writes:** qkv rows D:3D (k, v), gate (α, β), mem.ln (weight 1−γ) and ln_w (weight γ). Trunk blocks 0–5 via h_6 = h_low (cache_lower keeps the graph). Think-pass blocks 6–11, and the earlier passes' reads, **only for γ > 0** (B7).
  - **No term reaches the decoder through question tokens; there are none.**
* **L_aux:** unchanged from A. It is read-only on M_t, and its prompt is question-free.
* **Step 0:** W_o = 0 and fade = 0, so ∂L_ans/∂(writes, reads) = 0 at the first step. Only wo (scaled by fade) is nonzero, as in A. The gradflow gate therefore runs with fade = 1 and W_o ~ N(0, 0.02²), or after 50 steps.

---

## 8. Design, OC and censoring

**Cell (RTX 3090, seed 1, bs 64, lr 1e-4, warm-up 100 + cosine, 24k; final checkpoint; the test is never used for selection; the Aug-valid-selected checkpoint beside, labelled):**
* **AB** `gsm_lat_ansbn_s1` = §200 A's flags exactly + `--ans_bottleneck`.
* **Comparator:** A (gsm_lat_aug_s1), landed. Its eval-only re-run with the new fields on the same card and bs: acc and acc_by_kprime must be bitwise equal to its eval.json.
* All new eval-only readings run on the 3070 at eval_bs 64 for both models (the bf16 batch-size rule); card, dtype and bs are recorded.

**Readings (T = 0.028; T_D = 0.034; deep = m ≥ 3, n 879):**
* **ACC:** AB − A reads LIFTS / MATCHES / HURTS at ±T.
* **MEMORY-CARRIES** (replaces LOOP-NOT-FORMED, vacuous by B2): ACC ≥ P_test(y = c) + T.
* **THINK-NECESSARY** (the K′1 clause; the K′0 clause holds by construction and is reported, not counted) with the B6 guard. **THINK-NECESSARY-DEEP**; **DEEP-MULTIPASS** (deep K′10 − K′1 ≥ T_D, and the same with K′1-RESCALED). K′0 is printed beside: chance is ≤ 0.030 in accuracy and 1.85 nats per labelled token in answer CE.
* **MECH:** the TEMPLATE-SWAP 3-way label (3.3) with all P's, plus HYB-NULL, the SWAP(K−1) validator, L-SWAP and the energy split.
* **COPIES-LITERALS** (§6) and PASSK-REGISTER (§6).
* **§207-comparable** random-donor SWAP(1), SWAP(2) (3.1): UPSTREAM-RECOMPUTE if P(y_post | 1) ≥ 0.8 and P(y_post | 2) ≥ 0.8. 207-1's CARRIED(2) − CARRIED(1) is reported.
* **Recall share** (fit_fix − fit_rs).
* **A's same fields beside:** A's TEMPLATE-SWAP uses §207's STATE-SWAP semantics (the decoder reads the post-swap question). Its central label is RECOMPUTE (§207's 0.866).
* **Descriptive:** answer_ce entry (the first log record with answer_ce ≤ 0.5 × 1.85 = 0.92; "> 24k" if never), IDLE-PASSES, STABLE-12/16, the K′ grid by m, FILLER (under the bottleneck the filler enters the think input only).
* **Method labels:**
  - **BOTTLENECK-SERIAL (the C1 success):** SERIAL-CARRY (3.3, with COPIES-LITERALS clear) ∧ DEEP-MULTIPASS.
  - **BOTTLENECK-WIN:** BOTTLENECK-SERIAL ∧ ACC MATCHES or LIFTS.
  - One seed: "on seed 1".

**OC block (rule a; one seed):**
* **P(BOTTLENECK-SERIAL | no effect) ≈ 0.01.**
  - With no carry the hybrid hit rate is the accidental rate p₀ ≤ 0.03. P(P_hyb ≥ 0.30 at n ≥ 100 | p₀) < 10⁻¹⁵, so sampling cannot pass it.
  - The residual is systematic (STEPWISE-COPY slipping past ANS-LIN, a scorer artefact), put at ~0.01.
  - DEEP-MULTIPASS alone passes under no carry at about 0.2 (1 of 5 recipes, J1), so it is not used alone.
* **P(BOTTLENECK-SERIAL | effect) ≈ 0.25–0.5.**
  - "Effect" means the bottleneck induces carry in a fraction p_s ∈ [0.3, 0.6] of seeds (the nested converged-seed rates).
  - P(detected | a carrying seed) ≈ 0.8 (the hybrid fidelity needs the remaining steps right after a novel value).
  - **LR(positive) ≥ 25; LR(null) ≈ 0.6.** A one-seed positive is strong evidence that the mechanism exists in that seed. A one-seed null barely moves the prior. A rate claim needs ≥ 3 seeds (stage 2, a new decision, about +$2.6 per seed).
* **ACC:**
  - Between-recipe SD on this harness ≈ 0.009 (A / B / W / F-P / J1: 0.138–0.160). So P(LIFTS | no effect) ≈ 0.02 and P(HURTS | no effect) ≈ 0.02.
  - Under the central derived effect (−0.05), P(HURTS) ≈ 0.6.
* **Prior P(BOTTLENECK-SERIAL)** ≈ 0.2 × 0.4 + 0.8 × 0.01 ≈ **0.09**.
* **Realised $:** replaces the projection at landing.

**Censoring (rule b):**
* A run whose log ends before 24k, or that dies or is aborted: every reading is a failure, with the event-only values beside.
* **TEMPLATE-SWAP unreadable** (< 100 pairs at t = 1 or < 50 at t = 2 on m ≥ 3): SERIAL-CARRY is NOT MET (failure), with the event-only P's beside. The pool is fixed at 4,000 sampled templates before any output, with no top-up after seeing counts.
* **Probes unfittable** (< 500 valid items with the target class, or a failed fit): COPIES-LITERALS is NOT CLEARED, so the success label fails.
* **K′1-RESCALED missing:** THINK-NECESSARY is not credited.
* **answer_ce never enters (≤ 0.92) by 24k:** NO-ENTRY ("> 24k"), a failure, with the event-only step beside.
* **Not-landed:** pending, never censored.

---

## 9. Predictions and falsifiers (one seed; central values; P's subjective)

| id | prediction | central / P | falsified if |
|---|---|---|---|
| P-B1 | ACC HURTS | 0.095; HURTS 0.60, MATCHES 0.35, LIFTS 0.05 | AB ≥ 0.147 falsifies the central (the routing-loss account), not the method |
| P-B2 | K′0 = P_test(y = c) ≤ 0.030; a single distinct K′0 answer string | PROVED | any second string (a bug) |
| P-B3 | pooled K′1/K′10 high: the upstream one-pass | 0.85; THINK-NECESSARY (guarded) P 0.15 | ≤ 0.50 with the guard |
| P-B4 | deep K′1/K′10 | 0.85; DEEP-MULTIPASS P 0.30 | — (descriptive P) |
| P-B5 | TEMPLATE-SWAP label | RECOMPUTE-INTO-MEMORY 0.45 (P_own(1) ≈ 0.7, P_don(1) ≈ 0.1, P_hyb(1) ≈ 0.05); PASS1-PRECOMPUTE 0.25; SERIAL-CARRY 0.12; MIXED 0.18 | the "bypass moves upstream" account (B5c) is falsified by SERIAL-CARRY |
| P-B6 | COPIES-LITERALS | P 0.10 (A-DEC6: the decoder is a weak computer) | — |
| P-B7 | PASSK-REGISTER | P 0.05 | — |
| P-B8 | recall share | 0.35 [0.2, 0.5] (recall moves into pass 1) | < 0.10 means recall did not survive the routing |
| P-B9 | energy split | E_add(1) within 0.05 of 0.1 at the final model; E_del(1) ≤ 0.2 under RECOMPUTE | E_add(1) ≥ 0.3 (the additive channel holds donor content against B5's algebra; the γ = 1 closed loop would then matter) |
| P-B10 | §207-comparable UPSTREAM-RECOMPUTE | P 0.45 | — |
| P-B11 | answer_ce stays ≥ 1.5 per token to step 2000 (B7), then enters during the ramp; final ≥ A's 0.448 | P 0.7 / 0.85 | answer_ce at 2k ≤ A's 1.22 (an open-loop copier as good as A's direct path) |
| P-B12 | A on TEMPLATE-SWAP: RECOMPUTE (P_own(1) ≥ 0.8) | P 0.8 | P_hyb(1) ≥ 0.20 in A (A carried after all; §207 then needs revisiting) |

* **No derivation predicts SOTA or a lift.** B5(a)'s attribution holds whatever the outcome.
* **Decision value:**
  - RECOMPUTE-INTO-MEMORY → QUESTION-ONCE is the matching next lever (§2);
  - PASS1-PRECOMPUTE → QUESTION-ONCE will not help (pass 1 still sees q), and a per-pass value demand without the pass-1 leak is needed (F-NR-type closing, D5b);
  - SERIAL-CARRY → stage 2 seeds.

---

## 10. Cost

* **Realised s/step on 3090 hosts, A-base family (24k, micro 4):** A 1.82, B 1.51, F-P 1.27. W-base family (12k): W 1.10, J1 1.64.
* **The decoder saving:** the decoder shrinks from ≈ 63 to ≈ 4 tokens, about 14% of forward token-blocks per item (calc: think 3,828, aux about 400, decoder 757 → 49). The loop is launch-bound, so take 7%.
* **Central:** 1.51 × 0.93 = 1.40 s/step × 24k = 9.4 h, + eval 0.3 h, = 9.7 h × $0.25 + $0.10 per host ≈ **$2.5–2.6**.
* **High:** 1.82 s/step with no saving: 12.1 h + 0.3 h at $0.28/h + $0.10 + one STUCK host ($0.20) ≈ **$3.8**.
* **Realised comparables:** F-P $2.56 (+ $0.16 STUCK) at 24k; W $1.10 (+ $0.20); J1 $1.63; §200's three landed cells plus D's idle host ≈ $7.8.
* **Eval-only:** local on the 3070, $0. That covers AB and A: TEMPLATE-SWAP (≈ 2,440 templates × 16 clean evals each), L-SWAP, the random-donor SWAPs, probes, energy split and K′ grid, ≈ 1–2 h per model. The C: drive scratchpad .pt cleanup rule applies.
* **Proposed cap: $4.0.** W-base alternative: $1.3 central / $1.9 high.

---

## 11. Engineering contract and gates (Experiments implements; Audit reviews the diff; every new parameter optional; no module edit while any running job lazily imports these modules)

**11.1 boxes/prosqa168s2.py (ProsMem; shared with §168s2 / §191: the default must stay byte-for-byte equivalent).**
* `ProsMem.__init__(g, ans_bottleneck=False)` stores `self.ans_bn`. In `dec_loss`, when self.ans_bn: seqs = [EOT] + it["a1"] + [EOS], labels on a1 + EOS. That is the same labelled tokens as the plain path, n0 = 1.
* `margins()` (ProsQA candidate scoring): `assert not self.ans_bn` (GSM8K-only flag).
* The same switch goes in FactorMem's dec_loss if it overrides it (boxes/gsm8k_factor.py), so the flag composes with §204's arms only by a later decision. Add `assert not (ans_bottleneck and joint_cot)` (GJ.dec_layers builds the plain sequence).

**11.2 boxes/gsm8k_stepaux.py.**
* `--ans_bottleneck` (store_true; recorded in args.json and asserted equal under --eval_only, as add_norm is at :204–207).
* **One helper, `ans_prefix(it)`:** [EOT, ANS] under the flag, else [EOT] + it["pre"] + [ANS]. It is used by **every** answer path: answers(), valid_acc, acc_by_kprime, idle, path_indep, fit, struct, filler, dedupe, the validbest selection. A grep in the diff review confirms no other "pre" + [ANS] site remains on a stepaux path.
* **eval.json additions (additive):**
  - `bn` = {"k0_answer": c, "k0_distinct": n_strings, "k0_acc", "acc_k1_rescaled", "acc_by_m_kprime_grid" (K′ ∈ {0, 1, 2, 4, 6, 8, 10} × m, with n_by_m)};
  - per-item rows for K′ 1 / 10 (paired bootstrap).

**11.3 New eval-only script (theory/ansbottle/ or scratch/, no boxes edit; §207's gsm_swap.py pattern).** For a model dir and a mode (bottleneck or plain decoder):
* TEMPLATE-SWAP on TRAIN-REDRAW (rng 20711, 4,000 sampled templates, 16 draws; pair rule 3.2) at t ∈ {1, 2, K−1, K}, with HYB-NULL;
* L-SWAP (rng 20712, ≤ 300 candidates, ≤ 8 donors per recipient);
* random-donor SWAP(t) on §207's recipients (rng 20701);
* the energy split (3.5; it needs a hook recording the post-swap (α, β, k, v) per token and replaying them on a zero state);
* probes (3.6);
* K′1-RESCALED.
* Output: `<run>/eval_ansbn.json`. The hybrid calculator is theory/ansbottle/pairs_calc.py's `chain` / `hybrid_stats`, imported, not copied.

**11.4 Gates (before launch):**
* **G1 plain path bitwise.** Without the flag: a 200-step smoke's log records (loss, answer_ce, aux_ce) are bitwise equal to the pre-edit code at the same seed and card. A's eval-only re-run reproduces acc and acc_by_kprime bitwise at eval_bs 64. Then construct + 1 step on the plain path (the post-edit smoke rule).
* **G2 no question tokens.**
  - (a) Assert in dec_loss and in ans_prefix that every bottleneck row starts [EOT, ANS] and that its length is ≤ 2 + len(a1) + 1 (training) or 2 + ANS_MAX (eval).
  - (b) Invariance: for 64 test items, rebuild each item's record with its question replaced by another item's question while keeping S_K and the answer. Decoder logits and greedy answers must be **bitwise identical**.
  - (c) The same test on the plain path must differ (a positive control).
* **G3 gradflow** (scratch/gradflow_check.py; fade 1, W_o small random):
  - with λ = 0, L_ans alone gives nonzero grad on qkv[D:3D], gate, ln_w (γ = 1), mem.ln, wo, s_add, s_del, qkv[0:D], trunk blocks 0 and 11;
  - **DETACH control:** with S_K.detach(), L_ans gives **exactly zero** grad on qkv[D:3D], gate and ln_w, so the answer reaches the writes only through M_K;
  - at γ = 0, L_ans gives exactly zero grad on ln_w (B7);
  - L_aux alone stays nonzero on its known set.
* **G4 mem_check** at the longest question (the decoder is shorter, but run it anyway).
* **G5 K′0 constant:** exactly one distinct K′0 answer string over the test set.
* **G6 swap script.**
  - SWAP(K) with the donor reproduces the donor's clean answers bitwise; x_d = x_r reproduces clean bitwise.
  - The hybrid calculator returns y_own when given the recipient's own values on every pair.
  - The energy decomposition sums to the actual read within 1e-5 relative in fp32 (fp64 is impossible on the branch, as note 204-4; the property check stands).
  - A fixture runs on synthetic SERIAL / PASS1 / RECOMPUTE answer tables, and on a fixture where the probe fails (NOT CLEARED).
* **G7 timing:** HOSTPROBE + the first 200 steps (as §206 G8), with a running guard at $4.

---

## 12. Prior art (IDs verified on the arXiv abstract pages, 2026-10-15)

* **SELR**, "Think in Latent, Explain in Language: Self-Explainable Latent Reasoning", Zhao, Cao, Wang & Gui, **2608.13570** (verified).
  - The abstract alone does not say what the answer decoder sees.
  - Literature's full-text read (f4c7c197): the answer loss runs on [Image][Question] <bot> latents <eot> [Answer], so **the answer decoder sees the question**. Only its CoT decoder is question-free ("forces the latent thoughts to become a self-contained information bottleneck"), and there is no visibility ablation.
  - **Not a precedent** for a question-free answer decoder.
* **Markovian Transformers**, Viteri, Lamparth, Chatain & Barrett, **2404.18988** (verified): "all information flowing from question to answer must pass through a bounded-length CoT". **The text-CoT precedent** for an answer-side bottleneck. They use RL with a KL term against steganography; our COPIES-LITERALS is the latent analogue of that worry.
* **RiM**, "Unlocking the Working Memory of Large Language Models for Latent Reasoning", Aichberger & Hochreiter, **2605.30343** (verified). Its Figure 2 draws a "target cannot attend to question" variant, but reports no result for it (Literature). The closest design, unreported.
* **The others keep the question in the answer's context** (Literature's full-text reads; LOTUS eq. 4 per THEORY_LOTUS_CONTRAST §1.1):
  - Coconut, Hao et al., **2412.06769** (verified);
  - CODI, Shen et al., **2502.21074** (verified);
  - PCCoT, Wu, Teng & Tu, **2506.18582** (verified);
  - SIM-CoT, Wei et al., **2509.20317** (verified);
  - LOTUS, Fan, Svete & Lee, **2606.31779** (verified).
* **Wang & Reid, "Looped Transformers under the Jacobian Lens", 2609.01924** (verified): the re-injected-input override of a swapped state is from Literature's full-text reading. The abstract does not state it.
* **Novelty, marked honestly:**
  - An answer decoder with the question hidden, so the answer must cross the latent memory, is **not found for latent CoT on GSM8K**. It is **new for latent reasoning**, with the principle shared with Markovian Transformers (text).
  - The R_x template-pair hybrid test (an exact interchange target on GSM8K) is **not found** (CONJECTURE of novelty; Literature to confirm).
  - B5(c), the argument that the bottleneck relocates the one-pass computation rather than removing it, is ours.

---

## 13. Claim status

| claim | status |
|---|---|
| B1: the decoder's only item input is M_K; first-token reads at fixed queries; ≤ 1,536 floats | PROVED (code) |
| 90.6% of test answers are one token after "###" | calc |
| B2: K′0 is a constant; acc ≤ 0.030; answer CE ≥ 1.85 nats per labelled token | PROVED + calc |
| B3: A-CAP's addressing fails; capacity still admits a copy | PROVED (code) + DERIVED\|near-orthogonal keys + calc |
| B4: s_B ≤ 1 | DERIVED\|A-DEC6 |
| B5(a): ACC > a₁ ⇒ composition | PROVED (given the a₁ definition) |
| B5(b): a₁ ≲ 0.125–0.13 at this budget | DERIVED\|A-NEST + DATA |
| B5(c): first-order serial pressure unchanged; the bypass moves upstream | DERIVED\|A-1F |
| B5(d): the F-P analogy (pass-1 values under a stronger demand) | DATA, POST-HOC + CONJECTURE |
| B6: the K′ truncation artefact; RESCALED exact at γ = 0 (additive part) | PROVED (γ = 0) / DERIVED (γ = 1) |
| B7: the open-loop window copies by construction; zero grad to think-pass blocks 6–11 | PROVED (code) |
| B8: the clock carries nothing to the decoder; length does | PROVED + DERIVED |
| 3.1: the §207 arms collapse into SWAP(t) | PROVED |
| 3.2: hybrid construction; kept fractions; readable pools | PROVED (construction) + calc |
| 3.5: E_add = t/K at γ = 0; the delta affine decomposition | PROVED |
| 3.7 route table; §6 detectors' discriminations | DERIVED from B1–B8 + A-SLOT, A-DEC6 |
| L-SWAP generability 98.6%; kept 0.66 strict | calc |
| accuracy prior, label P's, OC numbers | CONJECTURE (+ DATA for seed SD) |
| QUESTION-ONCE closes RECOMPUTE-INTO-MEMORY by construction | PROVED (sketch, B1 applied to Φ); not designed here |

---

## 14. Risks

1. **The predicted outcome is a null for the method.** B5(c) and the F-P datum put about 0.7 on PASS1 or RECOMPUTE. The cell's value is diagnostic: it picks the next lever (§9), and makes attribution clean. Theory should decide whether $2.6 for a likely null that decides between QUESTION-ONCE and a pass-1-leak closer is worth it, or whether to derive QUESTION-ONCE first and run both together (two cells, about $5).
2. **ANS-LIN calibration.** A register read-out need not be linear in (r₀, r₁). ANS-LIN ≤ 0.5 could misfire on a nonlinearly coded register, which is a false COPIES-LITERALS. That is conservative: it only blocks success. A logit-lens onset over decoder blocks 7–12 at position 1 is reported beside as a second view.
3. **The open-loop window (B7) may tilt the trunk.** For 2k steps the whole answer gradient lands on blocks 0–5 through h_6, so later behaviour may be path-dependent (a copy basin). Removing the window (γ = 1 from step 0) is a second change and is not proposed. It is the first suspect if P-B11's entry never comes (NO-ENTRY).
4. **A-SLOT** (L-SWAP's hybrid) **and the step-vs-level schedule** (TEMPLATE-SWAP) are assumptions about a serial memory's layout. They agree on 0.80 of pairs at t = 1. A serial model on the level schedule would score as hybrid on that 80% only. P_hyb by both schedules is reported, and SERIAL-CARRY reads the step schedule as registered.
5. **The TRAIN-REDRAW pool sits on train templates** (structure memorisation, Prop. S). A serial-looking hybrid could come from template-specific circuits. That is still carry, and the TEST pool reading (n < 100, descriptive) gives the unseen-template direction.
6. **bf16 and batch size.** Every swap and probe reading runs at eval_bs 64 on the 3070 for both models. Clean-correct sets differ by card (§207's B borderline), so card / dtype / bs are recorded and never mixed between a cell and its comparator.
7. **The γ = 1 closed loop** makes E_add deviate from t/K. P-B9's falsifier catches a large deviation. The energy split is descriptive.
8. **One seed** (LR ≈ 0.6 for a null). A null does not show that the bottleneck cannot induce carry in some seed. Stage 2 (≥ 3 seeds) is a separate decision.
