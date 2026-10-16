# PROOFS_D3: rigour pass on D3 (SLVA + FROZEN-WPE + COPY-ANCHOR)

*Opus proof subagent for Theory, 2026-10-16. The repo was read only. CPU checks (2–3 threads) are in `scratchpad/research2/proofs_d3/`, with numbers in `results.json`:*
- *`slva_mock.py`: an SLVA subclass of VBankN on real GPT-2 weights. It is my reading of D3 §2.1, not `boxes/slva.py`.*
- *`simdecay.py`, `ratio.py`, `decay.py`: the wpe decay.*
- *`oc_read.py`: N1–N3's `eval_steppos.json`.*
- *`anchor_comb.py`: exact reduction chains.*
- *`cov_fresh.py`: fresh-state COV′.*

## Verdict table

| # | Claim | Verdict | Fix |
|---|---|---|---|
| 1 | Thm 1(a): SLVA is CARRY at pass 1 | **[PROVED]**; mock bitwise (V_1, coda(1); fp32 / fp64; c = 0) | Never pad-mask SLVA's own-slot keys: NaN reaches *real* slots at V_1 [VERIFIED]. Add a padded-vs-alone gate |
| 2 | "LONG-STEP measures only what training did to the shared weights"; later passes | **[PROVED for pass 1]**. Nothing for t ≥ 2 beyond (c)–(e) | N1's tolerance is a weight property, not an architecture property. Add the near-CARRY bound (§1) and the position-keyed-α route |
| 3 | Thm 1(b): c ≥ 1e4 ⇒ the rollout equals CARRY bitwise | **[PROVED UNDER** logit spread < c − 104 (fp32) / c − 745 (fp64), finite versions, SLVA path with autocast off and no TF32**]**; mock bitwise at K 12 | State the conditions. G-D1 in the training dtype too |
| 4 | Thm 1(b): "G1 holds bitwise" | **[FALSE as worded]**. CARRY K = 1, bank masked, vs GPT-2: rel 5.4e-8 (fp32), 1.6e-15 (fp64) | "Exact in real arithmetic, ≤ 1e-9 in fp64 (gate_fp). SLVA adds zero error" |
| 5 | Thm 1(c): slot-locality | **[PROVED]** | Slot-local ≠ position-free. The slot's own versions carry derived position, so α can depend on j (and on j ≥ 33) |
| 6 | Thm 1(d): ‖U‖ ≤ max_s ‖N(V_s)[j]‖ | **[FALSE: per-head counterexample, ‖U‖ = √2·max]** | ≤ √min(H, t)·max_s ‖N(V_s)[j]‖. Still t-free. The same fix applies to the base draft's PA bound |
| 7 | Thm 1(e): dilution only in t | **[PROVED]** (per head) | — |
| 8 | SLVA bias γ(t−1−s) on s = 0 | **Design inconsistency.** V_0's mass relative to the newest is ∝ e^{γ(t−1)}, an absolute pass clock | Age term on s ≥ 1 only, as in the bank |
| 9 | Prop 2: decay-only rows | **[PROVED + VERIFIED bitwise]**. 48k simulated fp32 decays reproduce N1's rows 33–1023 exactly | Real factor 0.97634 (0.0237). The fp32 factor 0.97618 (0.0238) is the measured value |
| 10 | Prop 2: "FROZEN-WPE removes the seam" | **[PROVED for Stage 1b; FALSE for Stage 1a]**. FT-SLVA freezes N1's table, which already has N1's seam | Say so. Neither stage removes the functional seam [OPEN]. Freeze before building the optimizer |
| 11 | Prop 3: incentive | **[DERIVED; valid incentive]**. The stated key-side bound is not rigorous | Replace it with Prop 3′ (query side) **[PROVED]**. The ratio is ≈ 6×, not 7× |
| 12 | Thm 4(a) | **[PROVED UNDER A5′, COV′, P-FREE, hard aggregation]** | Fresh-state COV′ is 1.000 at n 9–13 and 0.993–0.997 at n 14–16 (not 1.000) |
| 13 | Thm 4(b) | **[PROVED UNDER first-order composition]** | At pass 1 a head sees ≤ 2T keys (bank + causal), so T* halves in the worst case |
| 14 | Thm 4(c) bar ε_ℓ ≤ 0.23% | **[PROVED]** (0.2329%; n_ℓ = T − 33 = 22) | With N1's in-range errors: ≤ 0.12%. All ~50 copy slots: ≤ 0.10%. Pooled m 16: ≤ 0.30% |
| 15 | Thm 4(c): N1 predicts 0.86–0.95 | **[FALSE: mixes regions]** | With region rates: **0.84–0.88**. Measured 0.80 (m 16, T 55) |
| 16 | Lemma 5: zero copy drift | **[PROVED UNDER exact decodes]**. Also U_t[j] = N(V_0)[j] exactly for any α | — |
| 17 | Lemma 5: drift only via value slots, lifetime ≤ height | **[FALSE]**. VOID slots are never anchored, and there is a lifetime counterexample (§5) | Bound: n − t_create. Measured 1.5–5.9% of values outlive the height |
| 18 | COPY-ANCHOR interactions | **Four new constraints**: R never trained on V_0; the coda's V_0 path re-opens (frame slots); duplicate K/V; vacuous history readings | §5 fixes |
| 19 | OC: LENGTH-FREE P ≤ 0.02 under no effect | **[ASSUMED]**, consistent with the margin. Not derivable from 0 of 2 cells | State the non-bimodality assumption |
| 20 | OC: LENGTH-FREE ≈ 0.8 under effect | **[FALSE as defined: 0]**. NO-KNEE is UNREADABLE at m 11 for every model (35 items at T ≤ 29), so N1 fails, and FT-SLVA's stop rule fires by construction | NO-KNEE on m 9–10, or ≥ 6k draws at m 11. Then ≈ 0.6–0.85 |
| 21 | OC: DEPTH-GENERAL ≈ 0.05 under no effect | **[DERIVED]**: rule of succession, 1/19 | 95% upper bound 0.16 |
| 22 | P(D3-C2) ≈ 0.12–0.15 | **[PROVED UNDER strong positive dependence]** | 0.079 under independence; Fréchet cap 0.165. Quote 0.08–0.15 |
| 23 | H-DILUTION: "Lemma 2 allows ≤ 1.14×" | **[FALSE inference]**. It bounds the perturbation, not the error rate | Rest the refutation on the changed-slot and N1 evidence |

---

## 1. Theorem 1 against vbank.py

**What pass 1 runs (`rollout`).**
- **Normaliser.** CARRY's U_1 = `nrm(V_0, rms0)` = gain ⊙ V_0·(rms0 / rms(V_0)). The ratio is x/x = 1.0 exactly, so U_1 = gain ⊙ V_0 (V_0 bitwise at init).
- **SLVA at t = 1:** one entry per slot and head, with a finite logit (β⁰, no c term). softmax = exp(0)/exp(0) = 1.0 exactly, so U_1 = N(V_0)[j] exactly.
- **Core:** `bias_row` with nver = 1 gives β⁰ on real V_0 slots in both modes. **Coda:** t ≤ 1, so no bank.
- So pass 1 is the same function of (wte, wpe, blocks, ln_f, gain, core β⁰, R) in both modes. SLVA's parameters, β / γ for s ≥ 1 and the coda's bank rows do not enter. **[PROVED]** Mock: V_1 and coda(1) are bitwise equal to CARRY at c = 0, in fp32 and fp64.

**Padding hazard [VERIFIED].**
- Masking SLVA's own-slot keys by `pad` gives padded slots all −inf keys, hence NaN.
- NaN + (−inf) logits then poison block_bank's SDPA rows: the mock gets NaN in *real* slots of V_1. (PA's gather is safe because every query has real keys at other slots.)
- **Fix:** no pad mask on own-slot keys, plus a padded-vs-alone fp64 gate.

**(b) [PROVED UNDER conditions].**
- An off-newest weight is exactly 0 iff ℓ_s − ℓ_{t−1} < c − 103.97 (fp32) or c − 745.13 (fp64).
- The read is then exactly N(V_{t−1})[j], provided the stored versions are finite (0·NaN = NaN) and the product avoids TF32 / bf16 (autocast off; the trainer relies on PyTorch's allow_tf32 default of False).
- Induction over passes gives the rest. Mock: bitwise at K 12, γ_q × 3, fp32 and fp64.
- The hold's SLVA gradient is exactly 0 (w(δ − w) = 0 at one-hot).

**"G1 holds bitwise" is false as worded.**
- gate_carry_bitwise is PA at 1e4 ≡ CARRY. G1 is CARRY at K = 1, bank masked ≡ GPT-2, which runs block_bank's float mask (with T extra −inf columns) against `_block`'s `is_causal`.
- Mock: 5.4e-8 (fp32) / 1.6e-15 (fp64) relative. Exact in real arithmetic; passes gate_fp's ≤ 1e-9. SLVA adds zero error to it.

**(c) [PROVED], with a caveat.**
- U_t[j] is the same function of (V_0[j], …, V_{t−1}[j]) at every j and T.
- But those vectors carry derived position (PROOFS2 row 4, R² 0.9986). Position × version-content cross terms in ⟨W_q LN(N(V_{t−1})[j]), W_k LN(V_s[j])⟩ do not cancel over s, so α can be position-keyed, including at j ≥ 33.
- That Goodhart route is missing from §4. Read it as version mass by age, split j ≤ 32 / j ≥ 33, at passes ≥ 2.

**(d) [FALSE].**
- Per-head normalisation: two norm-10 versions on different heads' slices, each picked by its own head, give ‖U‖ = 14.14.
- Correct: ‖U_t[j]‖² ≤ Σ_h max_s ‖N(V_s)[j]_h‖² ≤ min(H, t)·max_s ‖N(V_s)[j]‖², with ‖N(V_s)[j]‖ ≤ ‖gain‖_∞√D·rms(V_0[j]). Still t-free.
- REDTEAM §E's PA bound has the same flaw.

**(e) [PROVED]** per head. ε can be divided by |S*|.

**What "CARRY at pass 1" guarantees later.** Nothing about values at t ≥ 2 beyond (c)–(e). Two additions:
- **Near-CARRY [PROVED from (e), S* = {t−1}].** α^h_{j,t−1} ≥ 1 − η ⇒ ‖U_t[j]_h − N(V_{t−1})[j]_h‖ ≤ 2ηB.
- **N1's tolerance belongs to N1's weights.** LONG-STEP is an architecture-controlled reading of D3's weights, not a prediction of them. Passes ≥ 2 can be length-keyed through the cross-slot bank reads (as in CARRY) and through position-keyed α.

**Design note.** §2.1 puts γ(t−1−s) on s = 0. V_0's logit relative to the newest is then c_0 − c_{t−1} + β⁰ + γ(t−1): its mass is an absolute pass clock, which the base draft removed from the bank. **Fix:** γ on s ≥ 1 only.

## 2. Proposition 2

**AdamW [PROVED].** A row with zero gradient at every step keeps m = v = 0, so its Adam update is 0/ε = 0, and only p(1 − η_tλ) acts. Clipping keeps zeros, there were no non-finite steps, and rows ≥ 33 are never indexed (T ≤ 33).

**Arithmetic.**
- §211 args: lr 1e-4, wd 0.01, 48k steps, a 100-step warm-up, then a cosine to 0. Σ η_t = 2.395.
- Real-arithmetic ∏ = 0.976335 (0.02367).
- **Verified bitwise:** 48,000 fp32 `p.mul_(1 − η_tλ)` on the pretrained rows 33–1023 reproduce N1's rows exactly (761,088 / 761,088 elements). The fp32 factor is 0.976177 (0.0238, the measured value); N2's is identical.
- **Sharp edge:** rows 29–32 (≤ 0.26% of items) drifted 0.10–0.16, like rows 20–28. Adam's normalisation lets one isolated gradient event move a coordinate ≈ 32 lr-units. So the table does not thin out at 28–32; only the blocks' exposure does.

**Freezing: which table.**
- **Stage 1b** (from pretrained, frozen from step 0): every row is pretrained. No table seam. **[PROVED]**
- **Stage 1a** (FT-SLVA, "N1's table frozen"): it keeps N1's seam (rows 1–32 drift 0.04–0.16; rows ≥ 33 at 0.976 × pretrained; adjacent cos 0.993–0.995 → 0.999). Freezing **only stops growth**. That is a sound premise control, since N1 tolerates this seam, but Prop 2's "no seam" does not describe it.
- **The functional seam** (blocks fine-tuned only on positions < 33) is removed by neither stage **[OPEN]**.
- **Implementation:** requires_grad False before building the optimizer. A zeroed `.grad` still decays (0.36% over the FT, 2.38% over 48k); G-D2 catches it.

## 3. Proposition 3

**The draft's key-side bound (‖W_k‖ L_φ ρ‖p‖) is not rigorous.** k_i comes from a 6-block causal prelude, so V_0[i] is not φ(p_i) + content, and the key difference carries item content that ρ does not bound. The binding side is the query, which depends on j only through the raw row.

**Prop 3′ [PROVED].**
- Take a head with q_j = W_q^h p_j + b^h. Suppose that in one item, query j prefers key a over b by Δ and query j+1 prefers b over a by Δ. This covers any pattern in which adjacent queries pick different keys.
- Summing the margins (logits /8) and applying Cauchy–Schwarz:

  ‖W_q^h(p_j − p_{j+1})‖·‖k_a − k_b‖ ≥ 16Δ, so σ_max(W_q^h)·‖p_j − p_{j+1}‖ ≥ 16Δ/‖k_a − k_b‖.
- Pretrained ‖p_j − p_{j+1}‖ ≈ 0.16 (cos 0.999, norm 3.5). So either σ_max(W_q^h) grows (global: every row) or the row differences grow (rows < 33 only, by Prop 2).

**Incentive [DERIVED; first-order, not an SGD theorem].**
- **Assumptions:** the loss rewards a slot-relative gather (measured: self + right neighbours); content varies across items; steps raise the realised margins.
- **Which route won is measured:** N2's queries have cos 0.999 at j ≥ 33 vs 0.80–0.84 in range. The separation is **row-borne**.
- "CARRY needs no q_j ≠ q_{j′}" is vacuous. The contentful claim is "**no term needs raw-row separation**": CARRY's local reads use LN'd h, already sharp.
- **Ratio:** mean 1 − cos over rows 20–32 is 0.038 vs 0.0065, so ≈ 6×.

## 4. Theorem 4

**(a) [PROVED UNDER A5′, COV′, P-FREE, hard aggregation].**
- At t = 1, A1, A3′ and A4 are vacuous. The proof is valid, but the content is in the assumptions.
- **COV′ for fresh states** (t = 0 keys only; train 60k at n 1–8; test 300 per n): 1.000 at n 9–13, then 0.997 / 0.993 / 0.997 at n 14 / 15 / 16. PROOFS2's 1.000 was for compacted states of every t.

**(b) [PROVED UNDER first-order composition].** At pass 1 a core head sees T bank keys plus ≤ j + 1 causal keys. So |F| ≤ 2T, and T* halves in the worst case.

**(c)**
- **Bar [PROVED]:** e ≤ 1 − 0.95^{1/22} = **0.2329%**. Here n_ℓ = T − 33 exactly at a fresh m 16 step (22 at T 55; no fresh handle at j ≥ 33 in 4,000 draws).
- **Alternative bars:**
  - ≤ 0.10% if all ~50 copy slots share one error rate;
  - ≤ 0.12% if N1's in-range errors stay (j 0–9: 0.13%; j 29–32: 0.25–0.34%);
  - ≤ 0.30% for STEP-GENERAL's pooled m 16 (mean n_ℓ 17).
- **The prediction is off.** The draft applies 0.2–0.7% to all 22 late slots, but 0.2–0.3% is N1's rate at j 29–32; at j ≥ 33 it is 0.47–0.69%. With region rates: 0.9931^22·0.9966^4·0.9986^10 = 0.835 to 0.881, i.e. **0.84–0.88**. Measured at m 16, T 55: **0.80** (134 items).
- **The independence model fits N1's buckets:** 0.961 / 0.931 predicted vs 0.964 / 0.942 measured. It is mildly conservative, consistent with positive association.
- **Consequences:** at N1-quality copying, STEP-GENERAL fails on copy errors alone, and CORE-BAND's 0.2% trigger fires on N1 itself.

## 5. COPY-ANCHOR

**Setup.** A_t = {j : ĉ_t(j) = ĉ_0(j)}. Ṽ_t[j] = V_0[j] on A_t, and V_t[j] otherwise.

**[PROVED UNDER exact decodes].**
1. **Copy slots are drift-free.** If j ∈ A_s for every 1 ≤ s < t, every stored version of slot j is V_0[j]. Then:
   - SLVA gives U_t[j] = N(V_0)[j] exactly, for *every* α;
   - every bank K/V of slot j is identical.
   - So copy slots have zero drift, both stored and as pass input.
2. **A_t also holds changed-back slots.** A_t = {j : E_t[j] = E_0[j]} includes values equal to the input class: 0.14–0.57% of value creations. Harmless to a class-only map, but those slots store V_0, not the value.

**[FALSE as stated]: "drift only via value slots, lifetime ≤ height".**
- **VOID slots are never anchored** (ĉ = VOID). Consumed slots are re-stored as recomputed V_t every pass and read by every bank reader: a drift channel, and §4's VOID-scratch route.
- **Counterexample to lifetime ≤ height:** a + b + ( c + d + ( e + f ) ), with n 5 and height 3. a + b reduces at pass 1, and its value is consumed at pass 5 (a lifetime of 4).
- Measured: 1.5–5.9% of value creations at n 5–16 outlive the height, by up to 9 passes. **Bound:** n − t_create.

**Interactions.**
- **R's targets.** R never trains on V_0, so R(Ṽ_t) at anchored slots is unconstrained. Stored versions then lose "decodes as E_t", which PROBE, MCT and VALUE-SWAP assume.
  - *Fix:* add L_rec at t = 0, and have G-D3 test R(Ṽ_t) = E_t.
- **The coda's bank (pin 4).** `coda` drops V_0 but reads Ṽ_1..Ṽ_{t−1}.
  - EOT and " =" never change, so they are always anchored, and every stored version carries V_0[" ="] (the prelude's summary of the whole problem) plus V_0 of every copy slot.
  - This **re-opens the V_0 → coda path**. VALUE-SWAP (iv) and RECOMPUTE-V0, which mask only V_0's own entry, stop removing V_0 content.
  - *Fix:* exempt frame slots and mask anchored entries in the coda and in arm (iv), or withdraw pin 4. The coda already reads V_1, so this may be harmless, but the choice should be explicit.
- **SLVA.** α is inert on always-anchored slots. Report HISTORY-USE and version mass on non-anchored slots.
- **Duplicates.** An anchored slot's bank weight is scaled by e^{β⁰} + Σ_s e^{β + γ(t−1−s)}, a saturating count of the passes it has stayed unchanged. CARRY does this approximately; the anchor makes it exact.
- **Credit.** Because the switch is an argmax, later-pass gradients through anchored entries reach the prelude, not the core's copy outputs.
- **COPY-REFRESH on N1** puts N1 off-distribution (stale context in its core, V_0 vectors in its coda's bank). A low g can therefore be distribution shift rather than layout; the n ≤ 8 control is what protects LAYOUT.

## 6. OC block

N1–N3's `eval_steppos.json` (the same fixed 2,000 items per m):

| | fresh exact, m 9 / 10 / 11 / 12 | T ≤ 29 items, m 9 / 10 / 11 |
|---|---|---|
| N1 | 1.000 / 0.998 / 0.991 / 0.966 | 1,329 / 307 / **35 (cells of 3 and 32)** |
| N2 | 0.782 / 0.246 / 0.010 / 0.000 | same grid |
| N3 | 0.791 / 0.131 / 0.003 / 0.000 | same grid |

**The defect [PROVED].**
- At m 11 both T ≤ 29 cells have < 100 items, so by §5's censoring NO-KNEE is **UNREADABLE (not met) for every model**.
- LENGTH-FREE therefore fails for N1, and **P(pass | effect) = 0 as written**.
- LENGTH-KEPT inherits this, so FT-SLVA's stop rule **fires by construction**.
- m 8 is no escape: its T ≥ 33 side has 1 item.
- **Fix:** NO-KNEE on m 9–10 (sides 1,329 / 160 and 307 / 914), or ≥ 6,000 draws at m 11.

**P(pass | no effect) ≤ 0.02 [ASSUMED].** The PA cells miss by ≥ 0.95, but 0 of 2 cells gives a 95% upper bound of only 0.78. The figure rests on the cliff not being bimodal across seeds; say so.

**P(pass | effect) after the fix: ≈ 0.6–0.85.**
- The binding cell is m 12 (N1 0.966). N1's m 12 cells *improve* with T (T 31 / 33: 0.64 / 0.87; T 39–41: 0.985–0.99), so clause 1 measures count decay, not length.
- The margin is 0.40 logit, which is 0.31 operators at N1's −1.32 logit per operator. A D3 curve shifted N(0, σ) operators passes with P 0.85 / 0.73 / 0.62 at σ 0.3 / 0.5 / 1.
- Make NO-KNEE the length reading, or label clause 1 as a count reading.

**Other readings.** DEPTH-GENERAL's 0.05 is the rule of succession on 0 of 17 (1/19; the 95% upper bound is 0.16). SELF-SIMILAR's 0.05 is [ASSUMED] from one cell.

**P(D3-C2).**
- Inputs: DEEP 0.55, LENGTH-FREE | DEEP 0.75, DG | DEEP 0.3, VB-C1 | DEEP 0.64.
- Independence: **0.079**. With DG ⇒ LENGTH-FREE: 0.105. Fréchet cap: 0.165.
- **0.12–0.15 needs P(VB-C1 | DEEP, DG) = 0.73–0.91.** Quote 0.08–0.15.
- Note that P(LENGTH-FREE | DEEP) = 0.75 implies P(effect) ≈ 0.94, above FT-SLVA's 0.75 prior.
