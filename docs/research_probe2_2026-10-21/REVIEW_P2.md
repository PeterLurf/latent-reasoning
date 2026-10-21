# REVIEW_P2: proof check and red team of P2_draft.md (PROBE2) (2026-10-21 06:25 EDT)

Reviewer: proof checker + red team (subagent). Read-only on the repo apart from this file. No GPU runs. No boxes/runs*/lvb_* output opened.
Sources read: P2_draft.md (6bb096e8); THEORY.md Notes 218-17, 218-18, 218-22, 218-23, 218-24; LANG_v1.md §2.2–2.6, §3.1–3.5, §6;
boxes/langvb_eval.py (Probe, probe_forward, plant_V, part_probe); boxes/langvb.py (coda_ls, dec_logprobs, rollout_ls); boxes/vbank.py
(block_bank); scratch/s218_gl5_known.py (plant_V_anti, k2antixor, probe2, the working tree); theory/score_s218.py ~1405–1450;
v1_checks/v1_oc.py and v1_oc.json. Arithmetic: scratchpad p2_oc.py (it reproduces v1's 0.063 / 0.004 / 0.398 before any change).

## Summary

| # | Item | Grade | One-line finding | Fix |
|---|---|---|---|---|
| 1 | D1: why the registered read path misses xor | **FAIL → BLOCKER** | The stated cause (capacity, a near-monotone readout, the name prior) is not derived, and a provable cause exists. The anti-xor label is tied to the display order, and the registered readout cannot see that order. At t = 0 the registered probe reads **exactly ½** on anti-xor for any recipe, strength or budget (Frame Lemma, §1). | Replace D1 with the Frame Lemma. Correct 218-18 item 3's "capacity limit" diagnosis by Note. Pre-state the predictions: R1 / R3 xor ≈ 0.5, and a display-order probe q ≈ ½. |
| 2 | D2: representability of sign(a)·sign(b) | **CORRECTED (MAJOR)** | The prediction is right, but the stated reason is wrong. PROBE2 can read anti-xor because its head takes (hA, hB) in display order, which gives it the frame. The ⊙ feature does not form (Δ·u)(Δ·u2) unless u and u2 are axis-aligned. XOR on reals needs 4 ReLU units, not 2. LN keeps the signs to first order. Attention needs routing matched to the option's content, which only the registered path has shown so far (anti-lin 1.000). | Rewrite D2 as "frame + sign product of two linear content features → 4 ReLU units". Drop ⊙, or keep it without the claim. |
| 3 | Order of the head's inputs at the readings | **BLOCKER** | The draft does not say how (hA, hB) are ordered for PROBE2(t). The existing PROBE path (A = gold, B = other, target 0) with a learned order-aware head lets the head win by always answering 0, giving PROBE2 = 1.0 at every t. The feasibility code randomises the order for probe0 only. | Use an **antisymmetric head** g(A,B) = f(A,B) − f(B,A), which makes any order shortcut impossible. Run one code path for every t, with an assert. Read PROBE2(0) on the test set. |
| 4 | D3 direction claims | **PARTLY FAIL (MAJOR)** | VALID being easier: PASS. The final-state check being "easier": FAIL, since a randomly initialised head replaces a readout that starts at about ACC, so it may be harder. PROBE2-GAIN being "harder": only [ASSUMED], and it is harder under serial use too. "Onset slope unchanged under serial use": FAIL. With h_p hops computed inside the probe, ON(D) = max(1, D − h_p). A truly serial model fails SERIAL-ONSET2 at h_p ≥ 5 (slope 0.5, span 2), sits exactly at the span bar at h_p = 4, and fails PROBE2-GAIN on band 5–8 at h_p ≥ 4 (gain 0.32). | Add a **SERIAL-ORACLE calibration**: PROBE2 on a planted gold-REACH frontier, with a registered margin. Add a **SELFCOMP null**. Retag the directions as [ASSUMED] and carry them into the OC. |
| 5 | PROBE(0) = ½ for any probe (§3.5) | **PASS (wording CORRECTED)** | Holds for any probe whose t = 0 inputs are question-free: V_0 slot rows come before the question, and coda_ls at t ≤ 1 has no bank. Condition: the (A,B) order must be a function of Z or an independent coin, never of Y. The draft's reason ("no option-asymmetric information") is imprecise. At n = 1,000, the ±0.03 band false-fails 5.8 % per model. | State the condition, read PROBE2(0) on the test set (n ≈ 10,000, false-fail ≈ 0), and keep it as the leak check. |
| 6 | Leakage via names, order, positions | **MINOR (given fix 3)** | Name identity, name-frequency priors, option positions and display order are all in Z, so by SYM v1 exchangeability they give exactly ½. Memorising name pairs from the pool cannot transfer to the test set. The shuffled-label control does **not** detect input-side leakage. | Sufficient: PROBE2(0) = 0.50 ± 0.015 on the test set, through the identical code path, plus the antisymmetric head. Shuffled-label is a plumbing smoke; order-swap and name-relabel are descriptive. |
| 7 | Certification design | **MAJOR** | The untrained model and the 200-step OVERFIT reference are nearly the same near-initial model, so "certified on two reference models" overstates it. The binding certificate is PROBE2-VALID on A1. That needs A1's own PCs / ε, stated explicitly, and an m-rule that also checks the **xor ceiling**, or the 218-17 trap (bar at the ceiling) repeats. Streams conflict: the draft says registered, the code says fresh. The plants sit only at t 0–2 while ON reads t ≤ 12. | Use the m-rule on A1 with K2-lin ≥ 0.985 ∧ K2-xor ≥ 0.97. Use fresh pre-stated streams for certification. Add plants at t 4 / 8 (required, or at least descriptive). Call the untrained / reference step a "recipe feasibility" check. |
| 8 | OC | **MAJOR (missing)** | Composite: P(pass \| no serial) ≈ 0.004, set by CF-L, not the probe. P(pass \| serial) falls from 0.398 to 0.27 / 0.17 / 0.10 for probe-component pass rates (GAIN, ONSET, VALID-A1) of (0.7, 0.6, 0.9) / (0.6, 0.5, 0.8) / (0.5, 0.4, 0.7). Headline 0.043 / 0.027 / 0.016. | v1 computes these from feasibility + SERIAL-ORACLE + A1's VALID priors, with v1_oc.py's mixture (numbers in §8). |
| 9 | Successor or rescue; honesty | **MAJOR** | It is a legitimate successor only if it is frozen before **any** A1 eval output (non-probe parts included; A1 ETA ~15:30 today) and labelled as a redesign after a positive-control failure. 218-18 item 3 says a new probe is "a new design **after §218 is scored**", so this earlier timing needs its own Note. A simpler successor exists (the registered read path + content-equivariant plants). Choosing between successors after seeing which passes certification is a forking path. | Pre-state the successor choice rule before any certification output. Add the wording in §9. |
| 10 | Open items / small | MINOR | Open 3: DIRECT-SYM is plain GPT-2 with no V_t, so PROBE2 is undefined there. Drop it. Open 1: redundant (the coda attends every slot). Open 2: freeze every hyperparameter (head width 256, lr 1e-3 / 1e-4, wd, init seed, joint training, extension rule). D1(i): the name-prior factor applies to the symmetric plant, and anti-lin overcame it. | As stated. |

**Top v1 changes (order of importance):**
1. Replace D1 with the Frame Lemma (§1), and correct 218-18 item 3 by Note.
2. Make the head antisymmetric, or pin a fair-coin order, on one code path. Read PROBE2(0) on the test set.
3. Certify a content-equivariant xor (definition in §2.3) beside anti-lin. Anti-xor reports as descriptive only, since an antisymmetric head cannot learn it.
4. Add the SERIAL-ORACLE calibration with a registered margin before A1, and the SELFCOMP null.
5. Use the m-rule on A1 with the xor ceiling, A1's own plant_dirs, fresh certification streams, and plants at t 4 / 8.
6. Freeze before any A1 output, compute the OC, and write the honesty wording.

---

## 1. D1: the Frame Lemma replaces the capacity story [FAIL → BLOCKER]

**What the draft says.** The xor plant fails because the frozen readout is near-monotone. On this account the product has to be computed in two blocks and written along name-likelihood directions, against a name prior 25× the shift. The draft tags this [DERIVED], but the only support is K3's empirics. As written it is **[MEASURED-only]**: it restates that anti-xor is not found and supplies no mechanism.

**A provable mechanism.** Write the display coin as c: o1 = X if c = 0, else Y, where X and Y are the two option concepts.

- **Registered read path (code-checked):** Δ = s(V_t, name_A) − s(V_t, name_B).
  - s is the same function for both strings: coda_ls with the decoder string run through blocks 0–9 alone, no question keys, at positions T_i + 1….
  - V_t is the same planted tensor in both runs.
  - Swapping the two names therefore negates Δ. The readout is **exchange-equivariant**: it chooses a *name* from content.
  - Its only access to c, the display order, is through V_t's encoding of the question.
- **Anti-xor plant:**
  - Rows: o1 += δ(s), o2 −= δ(s), with δ(s) = ε(s1 u + s2 u2). Label: o1 iff s1 s2 = +1.
  - In X's frame, X's row carries +δ(s′) with s′ = s·(1 − 2c), and Y's row carries −δ(s′).
  - (s′, c) are independent and uniform, and the content-computable product is s′1 s′2 = s1 s2 = y.
  - The target is X iff (c = 0) ⇔ (y = +1). So given the content, the target is X with probability ½, whatever the content.

**Lemma F (frame) [PROVED for t = 0; DERIVED bound for t ≥ 1].**
- Any exchange-equivariant readout whose inputs carry no information about c reads an exchange-invariant plant label at exactly ½. Exchange-invariant labels: symmetric lin, symmetric xor, anti-xor. Anti-lin's label is equivariant: the target is the row carrying +u.
- **At t = 0** the registered coda reads V_0 slot rows, which come before the question in the causal prelude, plus question-free decoder strings at σ-invariant positions. So c is absent and **registered PLANT-xor(0) = ½ exactly for every recipe, lr, budget and ε**. G-L5's "lin and xor ≥ 0.90 at t 0–2" was unpassable by construction, under both the registered symmetric plant and the anti-xor candidate.
- **At t ≥ 1**, if c can be decoded from the coda's inputs with accuracy q, the best accuracy is p·q + (1 − p)(1 − q) ≤ max(q, 1 − q), where p is the content-xor accuracy.
  - In the untrained model and the near-initial reference, the core reads the question only through the version-0 bank, which β0 calibrates to ≤ 0.02 mass at the 99th percentile. So q ≈ ½.
  - That matches the measured 0.522 / 0.515 / 0.532, and symmetric chance even at ×100.
  - A1 has no training incentive to encode display order: its target is the name "### o_R.". So the same holds on A1 [ASSUMED].
- **The K2 oracles never faced this.** They predict z from rows given *in order* [r1, r2], which supplies the frame. The probe must output an index.

**Consequences.**
- 218-18 item 3 ("a capacity / architecture limit of this probe") is wrong. The failure is a **plant-definition flaw**: a label tied to the display order, read by a readout that cannot see that order. This correction should be recorded by Note. It does not change any §218 consequence: (iii) stands and VB-C1-L stays not met.
- The 218-23 route is **closed by construction**. Even if R1 or R3 passed at t = 1, the required G-L5 re-run includes t = 0, where anti-xor = ½ exactly.
- **Predictions to pre-state now,** before R1 and R3 land: R1 xor and R3 xor at t = 1 read ≈ 0.5.
- **Optional $0 descriptive test of the lemma:** train the registered probe to always choose o1 (no plant). It reads q, predicted ≈ ½ at t 0–2 on the untrained model.

## 2. D2: why PROBE2 can read xor [CORRECTED, MAJOR]

**2.1 What actually makes it work.**
- In the plant arms, the feasibility head takes hA from the "### o1." run and hB from the "### o2." run, in that order (scratch/s218_gl5_known.py, probe2). The head is order-aware, so it *has* the frame.
- Anti-xor then reduces to sign(p)·sign(q), where p = (hA − hB)·w1 and q = (hA − hB)·w2 are two linear content features (the images of u and u2).
- That is the whole story. The ⊙ term, "capacity" and the absent name prior are not needed for xor.

**2.2 Are a and b separately present in hA and hB, and in what form?** hA and hB are the mean ln_f output at the name-token decoder rows of two coda runs, and each run attends to every real slot of V_t.
- **Attention route.** The plant enters hA through attention from the name rows of o1 to the slot rows. If attention were name-agnostic (the same weights in the A and B runs), the plant would enter hA and hB identically. Then hA − hB would carry no plant signal, and hA alone would carry (α_o1 − α_o2)·Wδ, whose sign varies from item to item.
  - The registered path's anti-lin at 1.000 shows that the coda's name rows do route to their own option's slot, by content match. So hA ⊃ +W_A δ(s) and hB ⊃ −W_B δ(s).
  - Each of hA and hB therefore carries **both s1 and s2 linearly**, with the opposite overall sign.
  - This is shown for the registered readout, not yet for PROBE2's fresh head: the feasibility arm lin ×1.5 is the test.
- **LN.**
  - LN(x + δ) ≈ LN(x) + J_x δ, with J_x = diag(γ/σ_x)(I − 11ᵀ/d − x̂x̂ᵀ). That is a positive per-coordinate gain after removing two directions, so no sign is flipped.
  - |δ(s)| = ε√2 is the same for all four sign patterns (u ⟂ u2), so the σ rescaling is sign-independent to first order. The sign-dependent second-order term is O(ε_rel) (ε_rel = 0.045 at ×1, 0.067 at ×1.5).
  - LN does not destroy the information. At ×10 (ε_rel 0.45) the linearisation is loose, but anti-lin at ×10 still learns at 1.000.
- **Attention dilution.** Diffuse attention scales the signal by about α_match. That is a power question for the feasibility test, not a representability one.

**2.3 Is the product feature needed?**
- No. sign(pq) = sign(|p + q| − |p − q|), and |x| = relu(x) + relu(−x). So 4 ReLU units on [hA, hB] represent sign(a)·sign(b) **exactly on all of R²**.
- "XOR needs 2 hidden units" holds only for Boolean corners. On unbounded real margins, 2 ReLU units cannot carve out Q1 ∪ Q3: the affine-region argument forces a linear term with an unbounded wrong sign in one quadrant. With width 256 this is moot, but the sentence should read "4".
- **The ⊙ claim is wrong as stated.**
  - The needed bilinear form is (Δ·u)(Δ·u2) = Δᵀ(u u2ᵀ)Δ, a rank-1 *off-diagonal* form.
  - A linear readout of hA ⊙ hB gives only diagonal forms Σ_i w_i x_i y_i. The a·b-type part it can reach is Σ_i u_i u2_i (…), of order ε²/d for random unit u and u2 (here u and u2 are random combinations of PCs 40–60). It is buried under Σ_i w_i hA_i hB_i from the base rows.
  - Drop ⊙, or keep it as plain extra capacity without the claim.

**2.4 Content-equivariant xor (needed for change 3).** Draw a fair coin for the target row T ∈ {o1, o2}.
- T += ε(s1 u + s2 u2) with s1 s2 = +1. The other row gets ε(s3 u + s4 u2) with s3 s4 = −1. Signs are otherwise uniform.
- Label: o1 iff T = o1.
- Every row's projections on u and u2 have mean zero under both labels, so linear accuracy is ½ (a genuine XOR). The rule "the row whose u- and u2-signs agree" depends only on content, so it is equivariant.
- An antisymmetric head g = φ(A) − φ(B) with φ(row) = sign-agreement learns it with 4 units. So does anti-lin. **Anti-xor is unlearnable for an antisymmetric head (Lemma F),** so under change 2 it can only be descriptive.

## 3. Order of the head's inputs: a constant shortcut [BLOCKER]

- **Registered PROBE mode:** in probe_forward, A = gold (dR), B = other (dP), and the target is always index 0. That is harmless for the registered readout, which is a shared per-string score and so equivariant.
- **The draft's head** MLP([hA, hB, hA − hB, hA ⊙ hB]) → 2 logits is **not** symmetric. If the PROBE2(t) readings reuse A = gold, the head learns the constant "index 0", and PROBE2(t) = 1.0 at every t, including t = 0.
- **Current coverage:** the feasibility code randomises the order only for kind 'probe0' (rng 22593 / 22594). The plant arms use (o1, o2), and no PROBE2(t ≥ 1) path exists yet.
- **v1 must:**
  - (a) Preferably make the head antisymmetric by construction, g(hA, hB) = f(hA, hB) − f(hB, hA). With it, any order-based shortcut is impossible and the fixed-order bug is harmless: at t = 0 the σ-partner item presents the same content with gold and other swapped, so the head is wrong on exactly one of the two. Or else (b) use a fair-coin order on a registered rng.
  - Use **one function for every t, including t = 0**, with an assert that the target is not constant within a batch.
  - Make PROBE2(0) the code-level leak check (§5).

## 4. D3: direction claims [PARTLY FAIL, MAJOR]

| claim | verdict | reason |
|---|---|---|
| PROBE2-VALID easier | PASS | Learnable by construction (frame + 4 units); the feasibility test is the evidence. |
| PROBE2(12) ≥ ACC − 0.05 easier | **FAIL** | The registered probe starts from A1's trained coda, which at t = 12 ≥ D already reads ≈ ACC at step 0. PROBE2's head starts random (lr 1e-3) and must reach ACC − 0.05, including D 9–12, which lies outside the probe's D 3–8 training stream. The direction is unknown and may be harder. This is a validity censor, so it lowers P(readable \| serial). |
| PROBE2-GAIN harder | [ASSUMED] | "Weakly more capacity" does not mean a higher trained accuracy at a fixed budget, and no registered PROBE(1) comparator exists. Where the claim holds, it is harder under serial use as well, which lowers P(pass \| effect). |
| SERIAL-ONSET2 "unchanged under serial use" | **FAIL as stated** | See below. |
| PROBE2(0) = ½ | PASS (see §5) | |

**Could a stronger probe raise PROBE2(t) at mid t under true serial use?** Yes. The coda copy has 2 trainable blocks over V_t and the bank, and slot rows carry each concept's local edges from H_6. So the probe can itself extend REACH by some h_p hops: forward from the marked frontier, backward from the options, or meet in the middle.
- About 1 hop per attention layer gives h_p ≈ 2–3 [ASSUMED]. A1's trained edge features could raise it. Partial accuracy below the 0.9 threshold matters too.
- Under ideal serial use (the frontier at t), PROBE2(t, D) ≈ ACC once D − t ≤ h_p. With score_s218's rule (grid 0–8, 10, 12; PROBE2(0) = ½ forces ON ≥ 1):

| h_p | ON(4..8) | slope | span | SERIAL-ONSET2 | gain, band 5–8 | gain, band 6–8 | gain, band 7–8 |
|---|---|---|---|---|---|---|---|
| 0–3 | D − h_p | 1.0 | 4 | pass | 0.43 (0.34 with an 80 % ramp at gap h_p + 1, h_p = 3) | 0.43 | 0.43 |
| 4 | 1, 1, 2, 3, 4 | 0.8 | **3 (at bar)** | pass | **0.32** ✗ | 0.43 (0.32 ramp) | 0.43 |
| 5 | 1, 1, 1, 2, 3 | **0.5** ✗ | **2** ✗ | **fail** | 0.22 ✗ | 0.29 ✗ | 0.43 |
| 6 | 1, 1, 1, 1, 2 | 0.2 | 1 | fail | 0.11 | 0.14 | 0.22 |

(ACC 0.98 and an off-reach accuracy of 0.55 are [ASSUMED]; scratchpad p2_oc.py.) So **a truly serial model fails both components once the probe computes ≥ 4–5 hops itself.** That is the Hewitt–Liang selectivity problem, and a stronger head makes it more likely. The same exposure existed in §218 but was never calibrated.

**Guards (both $0, known-answer, before any A1 probe output):**
- **SERIAL-ORACLE (required).**
  - On the untrained and reference models, plant a per-slot frontier code into V_t: +ε u_F on slots with dist(R, c) ≤ t, −ε u_F elsewhere, using FD-plant's ε. Train PROBE2 on the gold target per t, under the registered recipe.
  - Compute ON_oracle(D) and gain_oracle per candidate band.
  - **Register:** ON_oracle slope ≥ 0.8, span ≥ 4, and gain_oracle ≥ 0.40 for every candidate C1 band. Otherwise SERIAL-ONSET2 and PROBE2-GAIN are declared non-informative (not met, reported) before A1.
  - Disclose the residual: A1's trained features may raise h_p beyond the oracle's.
- **SELFCOMP null (required, descriptive bar).** PROBE2(1) on the gold target on the untrained model's V_1, read on bands 5–8. It measures what the probe computes from input-level information alone. If it is ≥ 0.60 on any band depth, PROBE2-GAIN is flagged non-selective.

## 5. PROBE(0) = ½ for any probe [PASS; wording CORRECTED]

- **The §3.5 argument:** V_0's slot rows come before the question (causal prelude, question last) and the coda is question-free. So PROBE(0)'s output is a function of Z, and by §3.3(a), P(Y = 1 | Z) = ½.
- **Code check:**
  - V_0 = H_6 gathered at slot_idx, which are premise first-mention positions (every concept is mentioned in the premises).
  - coda_ls at t ≤ 1 runs block_bank with no bank.
  - Decoder rows are question-free and sit at positions T_i + 1…, and T_i is σ-invariant (one-token person names).
  - So the theorem holds for **any** readout of these inputs, PROBE2's head included, and any trained parameters, since training items are independent of test items.
- **Conditions v1 must state:**
  - (i) The (A,B) presentation is a function of Z (display order) or an independent coin, never of Y. Otherwise the premise fails: §3's constant shortcut would read 1.0.
  - (ii) Same code path as at t ≥ 1.
  - (iii) "Exactly ½" holds in expectation over the σ-invariant test law. The finite-sample band must match n: ±0.03 at n = 1,000 false-fails 5.8 % per model (≈ 11 % over two models), while on the test set D 3–12 (n ≈ 10,000, SE 0.005) it is ≈ 0. Use the test set, or ±0.05 at n = 1,000.
- **Correct the reason:** not "V_0 carries no option-asymmetric information", but "every input at t = 0 is a function of Z".
- **Scope:** this certifies input-side leak-freedom only. A bank-path bug at t ≥ 2 is not covered (minor; diff review).

## 6. Leakage via names, order, positions [MINOR given fix 3]

- **Name identity** (hA and hB come from the name-token rows), **name-frequency priors**, **option slot positions** and **display order** are all part of Z. SYM v1 draws names uniformly per item without replacement, independent of role (S-σ3), and statement order uniformly (S-σ1/σ2). So every such shortcut gives exactly ½ on test, at every t, because these inputs do not change with t.
- **Memorisation:** over a 200,000-item pool, about 44,850 option-name pairs each occur ≈ 4.5 times with random roles. Memorising them cannot transfer to test items, which are excluded by signature.
- **Controls:**
  - **Sufficient:** PROBE2(0) = 0.50 ± 0.015 on the test set through the identical code path, plus the antisymmetric head. Every input-side shortcut is already present at t = 0.
  - **Shuffled-label** (as in feasibility): detects train / eval plumbing errors only. A head trained on permuted labels cannot learn a leak, so it reads ½ even when the inputs leak. Keep it as a smoke test, not as leak evidence.
  - **Order-swap** (re-render the question as "o2 or o1"; the prediction must flip): descriptive. Under the antisymmetric head it reduces to an identity check.
  - **Name-relabel** (consistent per-item redraw; accuracy unchanged): descriptive.

## 7. Certification design [MAJOR]

- **What the two certification models certify.** The untrained model and the G-L1c reference (200 steps, OVERFIT-256) are both near-initial: β0 is calibrated and PC spectra are similar (ε 6.86 vs 7.86). They certify that **the recipe can learn the plants**, not that it has power on A1's statistics. Wording: "recipe feasibility on two near-initial models". The certificate that binds is **PROBE2-VALID on A1 itself**, which the draft already has. Keep it as the gate.
- **A1's plant.**
  - State that plant_dirs is computed on **A1's own** pool rows (its PCs 40–60 and ε at PC 50), as §218's PLANT did per model.
  - "At the certified m" is ambiguous. Use m = m-rule(A1).
  - **Extend the m-rule:** the smallest m ∈ {1.5, 2, 3} with K2-lin ≥ 0.985 **and** a K2-xor oracle ≥ 0.97, computed on A1's V_1. Use content-xor if change 3 is adopted. Otherwise the 0.90 xor bar can sit at A1's oracle ceiling, which is 218-17's trap again.
  - The K2 computation on A1 reads A1's rows with a plant and no answer information. Pre-state that it may run before the PROBE2 parts.
- **Streams.**
  - The draft says certification uses "the registered streams". The code says feasibility consumed 22500–22554 and certification uses FRESH streams.
  - Design choices made after the feasibility test on those items are mildly adaptive. Use fresh, pre-stated streams for certification.
  - Keep the registered PROBE streams for A1's readings, as in §218.
- **Coverage over t.**
  - The plants are certified at t 0–2, but ON(D) is decided at t 3–8. Under-power there delays the onsets.
  - Add PLANT at t 4 and t 8 to PROBE2-VALID on A1: required, or at least descriptive with an extension rule.
  - Also pre-state the extension rule (20k) for the head.
- **Power at ×m.** LATE is read on 1,000 held-out items. At a true 0.92, P(LATE ≥ 0.90) is 0.99, so the bar is fine.

## 8. OC [MAJOR: missing; computed where possible]

v1_oc.py's mixture is reproduced exactly (P 0.063, | no serial 0.0043, | serial 0.398). PROBE2 changes only the probe components of the serial classes: k = (P_GAIN2 · P_ONSET2)/(0.8 · 0.7) × P(VALID2 on A1).

| P(GAIN2 \| serial) | P(ONSET2 \| serial) | P(VALID2_A1) | P(VB-C1-L2 \| no serial) | P(VB-C1-L2 \| serial) | headline |
|---|---|---|---|---|---|
| 0.8 | 0.7 | 1.0 | 0.004 | 0.398 | 0.063 |
| 0.7 | 0.6 | 0.9 | 0.004 | 0.269 | 0.043 |
| 0.6 | 0.5 | 0.8 | 0.004 | 0.170 | 0.027 |
| 0.5 | 0.4 | 0.7 | 0.004 | 0.099 | 0.016 |

- **Where the no-serial bound comes from.** P(pass | no serial) ≈ 0.004 is set by the decode-free CF-L (0.02 on non-serial classes) and the other non-probe components, not by PROBE2. The draft's "or lower (a stronger probe)" is [ASSUMED], since it needs PROBE2 to be at least as strict on every non-serial class. State it as ≤ the CF-bound.
- **Numbers v1 needs:**
  - P(PROBE2-CERT) from the feasibility test;
  - P(VALID2 on A1) [ASSUMED prior];
  - P(GAIN2 | serial) and P(ONSET2 | serial) from the SERIAL-ORACLE's h_p (§4 table: both ≈ 1 if h_p ≤ 3, onset ≈ 0 if h_p ≥ 5);
  - P(final-state check | serial).
- **Sampling level:**
  - PROBE2-GAIN at ≈ 1,000 items per band depth: under "answer readable after pass 1", gain ≈ 0 ± 0.011, so P(pass) ≈ 0.
  - PROBE2(0) band false-fail ≈ 0 at n 10,000.
- **Realised $:** $0.

## 9. Successor or rescue; honesty [MAJOR]

PROBE2 is a **legitimate successor**, not a rescue, if all of the following hold.
1. **Frozen before any A1 eval output is read,** non-probe parts included.
   - A1's ETA is ~15:30 today, and its non-probe parts run at landing.
   - VB-C1-L2 is a conjunction with those components. Registering after seeing them makes "whether to register" depend on outcomes, which is optional stopping.
   - If they are scored first, seal them, or disclose this.
2. **A Note addresses 218-18 item 3's timing** ("a new design after §218 is scored"). Registering earlier, blind to A1's probe outputs, is the more conservative choice, but it departs from recorded text, so Audit must OK it.
3. **The correct diagnosis is recorded** (Lemma F): the positive-control failure was a plant-definition flaw, not a probe-capacity limit. The draft's §0 / D1 framing would otherwise mislead.
4. **The successor choice is pre-stated before any certification output.** Candidates:
   - (S1) the registered read path + anti-lin + content-xor (the minimal change; Lemma F predicts it can pass);
   - (S2) PROBE2 with the antisymmetric head + anti-lin + content-xor;
   - (S3) the draft's ordered head + anti-xor.

   Picking whichever certifies is a forking path. Pre-state an order, e.g. S1 → S2 with the first to certify adopted (as 218-23 did), or justify one choice by derivation alone.
5. **Wording:** "VB-C1-L2: a successor reading under a probe read path and plant redesigned after §218's registered positive control failed by construction (Lemma F). Registered before any A1 output; one seed; recipe feasibility on two near-initial models, power certified on A1 itself (PROBE2-VALID); serial-onset calibration SERIAL-ORACLE; never C2 / C4; §218's VB-C1-L stays NOT MET." No headline or Stage B role, as the draft says.

## 10. Minor

- Open 3: DIRECT-SYM is a plain 12-block GPT-2 with no passes or V_t, so PROBE2(t) is undefined. Drop it.
- Open 1: redundant (the decoder rows attend every real slot in both blocks). Drop it.
- Open 2: freeze every hyperparameter (head 4D or 2D → 256 → 2, lr 1e-3 head / 1e-4 coda, wd 0.01, head init seed, joint training, extension rule, group / lockstep replay as part_probe).
- D1(i): the 25× name-prior factor is K1's *symmetric* plant number. Anti-lin learned at 1.000 despite the prior, so the prior is not the xor obstacle.
- §2: "PROBE2(0) without a plant is 0.50 ± 0.03": name the n (see §5).
