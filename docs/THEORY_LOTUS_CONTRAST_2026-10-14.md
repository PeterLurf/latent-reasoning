# LOTUS contrast: why does LOTUS show graded per-loop use on GSM8K while our loop is one-pass dominant? (2026-10-14 20:21 EDT)

Opus subagent for Theory, on Audit's question. Not a registration. The §4.3 note is a proposed post-freeze note for §204, to be filed before any §204 output.
- **Scripts:** theory/stepfactor/lotus_capacity.py → lotus_capacity.json. Older inputs: calc_bits_serial.json, diag_A.json, clock_A.json.
- **Rules followed:** CPU only, no git.
- **Tags:** PROVED, PROVED (paper), DERIVED|A…, CONJECTURE, DATA, DATA (POST-HOC), calc.

## 0. Summary
1. **Capacity is not the cause (§2).** Our channel's usable capacity is about 384 delta-head items, or a key-indexed table of min(L, 64) entries per head. The additive heads add ≈ 6 items when dense.
   - GSM8K needs ≤ 16 items and ≤ 194 bits per test item (answer alone ≤ 23 bits).
   - The margin is 24× on items, and 2× / 12× on bits at 1 / 6 bits per item. C < needed nowhere in the test m-mix.
   - A's M_1 already holds step 1–4 content at once.
2. **LOTUS's latent objective is teacherless (§3, PROVED from its eqs. 1–3).** Inputs are learnable ⟨lat⟩ + h⁽ᵗ⁻¹⁾; no gold token is ever an input; the main recipe supervises post-loop only.
   - A result token at level ℓ then needs its chain computed across positions, so ⌈ℓ/h_L⌉ iterations: per-loop necessity.
   - Our string aux is teacher-forced, so the result is one in-context operation (D1): no necessity.
3. **The derivation favours (ii), supervision form,** over (i)-as-width. It does not exclude (i-b), per-loop geometry: one read at block 6, and our answer decoder reads one injection rather than latent rows at all layers.
4. **§204 F-P is already the discriminating test:** teacherless values at fixed width. §4.3 gives a ready-to-paste LOTUS-SEP note with thresholds. Audit's "fails THINK-NECESSARY ⇒ width" rule has likelihood ratio ≈ 1.
5. **Wording:** LOTUS is a counterexample to one-pass dominance being inherent to latent loops on GSM8K, at 3B. Its own ablations do not attribute it, and there is no GPT-2 R-curve.

## 1. Facts

**1.1 LOTUS 2606.31779 v2 (PROVED (paper): read from the arXiv HTML full text, §3, Tables 1 and 4–8, App. D; the code was not read).**
* **Input:** [Q, ⟨BoT⟩, K·c ⟨lat⟩ positions, ⟨EoT⟩, A]. ⟨lat⟩ positions carry learnable embeddings E (eq. 1).
* **Loop:** h⁽⁰⁾ = f(E | C_pre), then h⁽ᵗ⁾ = f(E + h⁽ᵗ⁻¹⁾ | C_pre) for t = 1..R (eq. 2).
  - f is "the base causal LM". The question's KV cache C_pre is computed once. Gradients flow through all R iterations.
* **L_step:** per-position CE of f_head(h⁽ᴿ⁾_{i,j}) against gold token T_{i,j} (eq. 3), applied **post-loop only** in the main recipe.
  - **Correction to the brief's "every loop":** per-iteration supervision is an ablation, at 68.2 vs 70.0 (Table 4).
* **Answer:** a separate forward over [⟨EoT⟩, A], conditioned on [C_pre, h⁽ᴿ⁾] (eq. 4). The answer attends to all latent rows at every layer.
* **Training:** Coconut's staged curriculum, where epoch 0 is standard CoT fine-tuning (App. D), so it is a warm start plus a curriculum.
* **Sizes:** K = 6, R = 6; c = 25 (150 positions) on Llama, **c = 13 (78 positions) on GPT-2** (App. D).
* **Results:**
  - Table 1 is reported over 3 seeds; GPT-2 LOTUS reads 44.1.
  - Every R / c / supervision ablation is at **3B**:
    - inference R 1..7 = 22.7 / 40.0 / 55.0 / 63.5 / 68.7 / 70.0 / 69.3;
    - training R 2..6 = 14.6 / 23.2 / 52.6 / 68.1 / 70.0;
    - **answer CE only (no latent supervision): 63.3**; CODI-only 64.4;
    - c = 1 / 5 / 10 / 25: 49.7 / 67.5 / 68.4 / 70.0.
* **Confounds in its ablations:**
  - **c is confounded with the target:** T_i is truncated to c tokens, so c = 1 supervises only each step's first token.
  - **Truncation:** inference R < 6 truncates a model trained at R = 6, and R = 1 (22.7) sits below that backbone's no-CoT (38.3).

**1.2 Ours (DATA, POST-HOC).**
* §200 A: pooled K′1/K′10 = 0.85 (floor-corrected 0.66); m ≥ 3 0.88 (0.72). B, m ≥ 3: 0.70 (0.38). ProsQA §191: 0.004.
* The string aux is teacher-forced, and its value tokens are memory-free: top-1 0.880 with M = 0 vs 0.894 (D1, STEPFACTOR §1.4).
* A's M_1 with prompt t carries step t's content at 0.86–0.92 of M_t for t ≤ 4.
* 79–86% of operand errors are wrong-candidate (selection) errors.
* §204 F-P decodes the value from M_t with no expression in context.

## 2. Capacity bound (DERIVED|A-CAP; calc lotus_capacity.json; DATA where marked)

**A-CAP (assumptions):**
* keys can be learned near-orthogonal within a head;
* a retrieved item carries b ≥ 1 bit at the read's SNR;
* the answer decoder's queries are item-dependent (it sees q), so it can address items.

**Usable capacity, not the 49,152 floats.**
* **Delta heads (6, DH 64; L2-normalised keys, delta rule):**
  - Exact key → value retrieval holds for ≤ DH near-orthogonal keys per head; beyond that, interference. Schlag et al. 2102.11174 derive the linear-attention capacity limit and the delta-rule fix.
  - So ≈ 6 × 64 = **384 items**, each a 64-d value.
  - **The write geometry here:** keys come from the question's tokens (L mean 57, p90 86, max 182) and are re-written every pass. Values become pass-dependent once γ > 0.
  - So the state is a key-indexed value table of ≈ min(L, 64) entries per head, overwritten serially in its values. Content is updated, not lost: the delta rule's erasure along k *is* the overwrite.
* **Additive heads (6; φ = elu + 1 ≥ 0):**
  - Every stored item has a positive weight φ(q)ᵀφ(k) > 0 in every read, and dense positive features give a non-zero mean cross-term. Retrieval is then ≈ a bag: **≈ 1 item per head**, about 6 in all.
  - The best case, disjoint-support features, is 64 per head.
  - RMS changes the read's magnitude, not its item count (D5b).
* **Totals:** C ≈ 384–768 items. That is 384–768 bits at b = 1, 2.3–4.6 kbit at b = 6 (one number-token identity among about 64 candidates), and ≥ 12 kbit at b = 32.

**What the test needs** (per item, from the gold equations; worst-case digit coding; plan = 17-way operator + two pointers over the candidates):

| per test item | mean | p99 | max |
|---|---|---|---|
| answer alone (what f_ans must read if the loop computes serially) | 7.6 bits | 19.9 | 23.3 |
| every step's value | 24.7 | 85.8 | 110 |
| values + plan | 53.7 | 144 | **194** (m = 8) |
| key-value items (one value + one plan entry per step) | 6.5 | 14 | **16** |

* **Margins:** items 384 / 16 = **24×**. Bits 384 / 194 = **2.0× at b = 1**, **11.9× at b = 6**. So C < needed holds **nowhere** in the test m-mix (by m, the max needs are 24 / 55 / 88 / 113 / 141 / 163 / 181 / 194 bits for m = 1–8).
* **Realised, not just available (DATA):**
  - A's M_1 carries step 1–4 content at once (0.86–0.92 of M_t), so the trained channel holds ≥ m items.
  - Interference would show up as precision (near-candidate) errors, but those are 14–21% of operand errors; selection dominates.
* **Raw width for scale:** ours is 49k floats of state. LOTUS GPT-2's latent input is 78 × 768 = 60k, and its answer decoder sees 78 × 768 × 2 × 12 = 1.4M K / V entries.
* **Conclusion: width-as-bits is not the cause** (DERIVED|A-CAP + DATA).
* **What it does not exclude, (i-b) geometry:**
  - our pass reads once, at block 6, with 6 blocks after the read, against LOTUS's full 12-layer recurrence on the carried latent residuals;
  - our answer decoder gets one injected read at block 6 per position, against LOTUS's answer attending to 78 latent rows at every layer.

  That is per-loop depth and read path, not capacity.

## 3. Supervision form: LOTUS is teacherless, and that gives per-loop necessity

**Prop. T (PROVED (paper), eqs. 1–3).**
* At every iteration the latent positions' inputs are E + h⁽ᵗ⁻¹⁾: learnable, item-independent embeddings plus the model's own previous states. No gold CoT token is ever an input, so no gold prefix is revealed inside the latent block.
* L_step scores h⁽ᴿ⁾ only. Each gold token T_{i,j} must therefore be produced from (C_pre, the latent states).
* This is the teacherless multi-token objective of Bachmann & Nagarajan 2403.06963 (dummy inputs, all targets at once), made recurrent.
* **Unverified:** the latent attention mask, assumed causal ("base causal LM"). Prop. T holds under any mask, because no gold token is input.

**Lemma TL (DERIVED|A-OPL).**
* **A-OPL (assumption):** one iteration of an L-layer transformer composes at most h_L dependent arithmetic levels. Each level needs layers to fetch operands from other positions and to compute.
* **Argument:**
  - Block i's result token at level ℓ(i) needs the values of the results it references. Those are produced at other positions, as computed latent states, not gold tokens.
  - Within one iteration the chain costs ≥ ℓ(i)/h_L of the forward's depth. The carried state h⁽ᵗ⁻¹⁾ lets each iteration advance it by ≤ h_L levels.
  - So the post-loop target at block i is reachable only after ⌈ℓ(i)/h_L⌉ iterations: **per-loop necessity**, with gains spread over iterations up to R ≈ d*/h_L.
* LOTUS's graded training-R and inference-R curves are what this predicts. That is a consistency check, not a test: 3B, and truncation confounds R < 6.

**The D1 contrast (PROVED, STEPFACTOR D1).**
* Our string aux feeds the gold expression a_t as input before the value tokens (teacher forcing). The result is then one in-context operation on gold operands, whatever its level, and is satisfied at pass t with no chain in memory.
* This is exactly the Clever Hans leak that the teacherless objective closes.
* Computed (non-gold) operands in context are legitimate computation, not a leak: in LOTUS, a result position may read same-iteration operand positions only after the model has computed them.

**F-P is teacherless for values.** v_t is decoded from M_t with only the prompt (STEPFACTOR §5.2).
* So Lemma TL applies to F-P's value term under A-OP. Its plan heads are level 1 (parallel, D2) and carry no necessity.
* **Remaining differences from LOTUS:**
  - F-P binds step t to pass t, where LOTUS supervises post-loop (LOTUS's own per-iteration vs post-loop gap is 1.8 points at 3B);
  - the init / curriculum (W is a CoT init without a curriculum);
  - (i-b) geometry;
  - scale.
* **What LOTUS shows about supervision:** at 3B, gold-CoT supervision adds +6.7 accuracy over answer-only (63.3 → 70.0). The R-curve exists only for the supervised model, so **LOTUS itself does not separate (i) from (ii) for per-loop use.**

## 4. Separating predictions in our setting

**4.1 What each hypothesis predicts for F-P vs its comparator** (W, or A on the A base; registered fields; central values; deep = m ≥ 3):

| reading | H(ii): teacherless targets suffice | H(i): channel width is the cause |
|---|---|---|
| SHORTCUT-RATIO (pooled) | comparator − 0.25 (A base 0.85 → 0.60; floored at ≥ 0.51 by D4's s ≥ 1) | Δ within ±0.05 |
| deep raw ratio | ≤ 0.6 and ≤ comparator − 0.20 (A base 0.88 → 0.50; W base ≈ 0.40) | Δ ≥ −0.10 (0.85–0.90 on A base) |
| VALUE-EXACT from M_t, ℓ ≥ 2 R-steps, t 2–3 | ≥ floor + 0.10 (central 0.20) | ≤ floor + 0.05 (results never carried) |
| VALUE-LEAK (M_1 with prompt t, same steps) | ≤ floor + 0.05 | uninformative (both at floor) |
| interference signature: pass-1 VALUE-EXACT on level-1 steps, m ≤ 2 vs m ≥ 4 | flat (\|Δ\| ≤ 0.05) | drops ≥ 0.10 (more stored items, more interference) |

* **A width test** (string target, pass token kept, the channel doubled, e.g. a second DeltaMem branch at block 9; one change vs the comparator):
  - H(i) predicts the deep ratio falls ≥ 0.10 and ACC rises.
  - H(ii) predicts no change (|Δ| ≤ 0.05), because the teacher-forcing leak sits in the target (D1) whatever the width.
  - §2's bound predicts the null.
* **The registration that already discriminates is §204 F-P vs its comparator:** it changes the supervision to teacherless values at fixed width. F-NR vs F-P tests the index channels, not width.
* **One extra cell would add:** the width test above isolates (i) directly.
  - Cost: W base 12k ≈ $1.4 central / $1.9 high at b286's s/step, plus ≈ 30–50% for the second write and read, so ≈ $1.9–2.8. Plus new code and a mem_check.
  - Not registered here. Its expected information is low given §2.

**4.2 Why "F fails THINK-NECESSARY ⇒ (ii) insufficient ⇒ width next" discriminates weakly.**
* §204 predicts pooled failure at P ≈ 0.6 (W base) / 0.8 (A base) under its own (ii)-type derivation (P-F5a), because of the decoder's self-computation s (D4).
* Under H(i) failure is ≈ certain. The likelihood ratio is therefore ≈ 1.2–1.7: weak evidence for width.
* The deep-ratio delta, with VALUE-SERIAL and the interference signature, separates the hypotheses better (4.3).

**4.3 Ready-to-paste note (proposed Note 204-2; post-freeze, to be filed before any §204 output):**

> **Note 204-2 (proposed; post-freeze, before any §204 output; docs/THEORY_LOTUS_CONTRAST_2026-10-14.md). LOTUS-SEP: is LOTUS's per-loop use (2606.31779) explained by teacherless supervision (ii) or by channel width (i)?**
> * **Inputs:** read from §204's registered fields on F-P vs its comparator (W, or A on the A base), on paired test items.
>   - deep = m ≥ 3 (n 879); "ratio" = the raw K′1/K′10 on deep items;
>   - "floor" = the M_0 reading;
>   - VALUE-SERIAL = P-F3's letter (VALUE-EXACT from M_t ≥ floor + 0.10, and VALUE-LEAK ≤ floor + 0.05, on ℓ ≥ 2 R-steps at t = 2–3).
> * **One added descriptive eval field, $0:** VALUE-EXACT and PLAN-ACC at pass 1 on level-1 steps, by m (≤ 2 vs ≥ 4).
> * **Readings:**
>   - **(ii)-SUFFICIENT:** ratio(F-P) ≤ ratio(comparator) − 0.20, and VALUE-SERIAL.
>   - **DECODER-SHORTCUT:** VALUE-SERIAL, but ratio(F-P) > ratio(comparator) − 0.10. The memory carries results that the answer path does not use, which points to (i-b) read geometry, not width.
>   - **WIDTH-CANDIDATE:** all three of:
>     - VALUE-EXACT from M_t on ℓ ≥ 2 R-steps ≤ floor + 0.05;
>     - ratio(F-P) > ratio(comparator) − 0.10;
>     - INTERFERENCE: pass-1 VALUE-EXACT on level-1 steps lower by ≥ 0.10 on m ≥ 4 than on m ≤ 2, **and** that drop exceeds the same drop in PLAN-ACC by ≥ 0.05 (the difficulty control).
>   - **CARRY-FAILS:** as WIDTH-CANDIDATE without INTERFERENCE. Results are not carried, but not by interference (A-OP or learnability).
>   - **INCONCLUSIVE:** otherwise, e.g. a delta in (−0.20, −0.10].
> * **Noise:** the deep ratio's SE is ≈ 0.08 per cell (0.03–0.12), so a paired difference has SE ≈ 0.11, and −0.20 ≈ 1.8 SE. The per-item rows (§204 gate) give the bootstrap, reported beside.
> * **Pre-stated P (A base / W base):**
>   - (ii)-SUFFICIENT 0.25 / 0.35;
>   - DECODER-SHORTCUT 0.25 / 0.25;
>   - WIDTH-CANDIDATE 0.05 / 0.05;
>   - CARRY-FAILS 0.15 / 0.10;
>   - INCONCLUSIVE 0.30 / 0.25.
> * **Consequence:** only WIDTH-CANDIDATE motivates a width cell. DECODER-SHORTCUT motivates a read-geometry change (the answer reading memory at more layers) as its own derivation.
> * Descriptive, one seed. It changes no §204 rule.

**4.4 Wording.**
* **Can claim:** "LOTUS (2606.31779) is a counterexample to one-pass dominance being inherent to latent loops on GSM8K. A looped, teacherless, post-loop-supervised latent workspace at Llama-3.2-3B shows graded per-loop use (inference R 1 → 6: 22.7 → 70.0)."
* **Cannot claim:**
  - (a) that it holds at GPT-2 scale: only the main number, 44.1 over 3 seeds, with no GPT-2 R-curve;
  - (b) which of LOTUS's differences causes it: the ablations do not cross supervision with R; c is confounded with target truncation; answer-only reaches 63.3;
  - (c) that our memory would show it with teacherless targets: that is §204 F-P;
  - (d) that a graded truncation curve means one step per loop: R < trained R is off-distribution in LOTUS, as K′ < K is for us.

## 5. Claim status

| claim | status |
|---|---|
| LOTUS's inputs E + h⁽ᵗ⁻¹⁾, post-loop L_step, GPT-2 at 78 positions, the 3B-only ablations, answer-only 63.3, the c truncation | PROVED (paper, v2 HTML); code unread; mask assumed causal |
| Prop. T: the LOTUS latent objective is teacherless | PROVED (from eqs. 1–3) |
| Lemma TL: per-loop necessity ⌈ℓ/h_L⌉ under the teacherless post-loop CE | DERIVED\|A-OPL |
| D1 contrast: TF string results need no chain | PROVED (STEPFACTOR D1) + DATA |
| usable capacity 384–768 items; needs ≤ 16 items / ≤ 194 bits; margins 24× / 2–12× | DERIVED\|A-CAP + calc |
| the channel already holds ≥ m items (M_1 0.86–0.92); errors mostly selection | DATA, POST-HOC |
| width-as-bits is not the cause; (i-b) geometry not excluded | DERIVED |
| §4.1 predictions; §4.3 thresholds and P's | DERIVED + CONJECTURE (central values) |
| LOTUS does not itself separate (i) from (ii) for per-loop use | DATA (paper) + reading |

## 6. Risks
1. **LOTUS was read from its v2 HTML only, not its code.** The latent attention mask and the curriculum details (E_stage) are unverified. Prop. T is mask-robust.
2. **A-CAP is optimistic for the additive heads,** and the realised-capacity datum covers plan content, not computed values. If F-P's values fail with an interference signature, §2 is wrong in practice. The LOTUS-SEP WIDTH-CANDIDATE reading catches that.
3. **(i-b) geometry is a live alternative** that neither the capacity bound nor F-P removes. DECODER-SHORTCUT is its signature.
4. **Confounds between LOTUS and us:** scale (3B vs 124M), init / curriculum and budget. A null F-P result would not refute LOTUS, and a positive one would not reproduce it.
5. **The interference test is confounded with difficulty** (longer problems may have harder first steps). The PLAN-ACC control is included, but it is imperfect.
6. **The probabilities are subjective,** at one seed and with a ±0.11 paired SE.
