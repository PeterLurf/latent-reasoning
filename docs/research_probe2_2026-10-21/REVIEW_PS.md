# REVIEW_PS: targeted review of PS_v1.md (PROBE-S, §222 candidate) (2026-10-21)

Reviewer: targeted proof check (subagent). Read-only on the repo apart from this file. No GPU or model runs. No boxes/runs*/lvb_* opened.
Sources read: PS_v1.md; REVIEW_P2.md §1, §2.4, §7, §8; THEORY.md Notes 218-24 and 218-25 (and 218-18); boxes/langvb_eval.py (Probe.scores, plant_dirs, plant_V, probe_forward, part_probe); boxes/langvb.py (coda_ls, dec_hidden, dec_logprobs); boxes/vbank.py (block_bank); LANG_v1.md §2.5, §3.5, §6; v1_checks/v1_oc.py and v1_oc.json; rt_lang/oc_shade.py; scratch/s218_gl5_known.py (cexor_draw, plant_V_cexor, k2cexor, k2anti, k3gen); untrained-model feasibility JSONs under scratch/gate_s218/g_l5_known/ (k2_anti*, feas2_S1_*; no A1 file exists or was opened).
Toy checks (numpy, CPU): scratchpad `ps_review/sim_ps.py` / `sim_ps2.py` (outputs in sim_ps.json / sim_ps2.json, same folder): C:\Users\peter\AppData\Local\Temp\claude\D--\6440ebeb-beed-5a82-930f-3ec2470822c0\scratchpad\ps_review\

## Summary

| # | Item | Grade | Finding | Fix |
|---|---|---|---|---|
| 1a | Label independence of PLANT-xor-S | **PASS (text CORRECTED, minor)** | T is a fair coin drawn on its own stream, so P(T = o1 \| item, display order, names, gold) = ½. Any readout of the unplanted inputs reads exactly ½ in expectation. NOPLANT is an **exact** null for any trained probe (binomial at n 1,000: P(LATE > 0.55) ≈ 0.001 per cell). Shuffled-label is ½ **only in expectation**: a label-blind readout of planted rows can correlate with T, because the two classes differ in plant axis (u + u2 vs u − u2). With per-batch permutation, as the feasibility code does it, toy \|bias\| ≤ 0.01. In the code (cexor_draw) T, s1 and s3 are independent. The design's stream mapping is **unpinned**: 22751–22758 have no roles; "PCs 22750" contradicts "rng 22550"; A1's cexor streams are undefined. | Pin the stream table (E1). Pin a per-batch permutation for the shuffled control. State the NOPLANT exact-null sentence (E4). |
| 1b | Positive construction: an order-blind readout can read cexor | **PASS** | Exact construction within blocks 10–11: 4 ReLU units in block 10's MLP write φ(r_j) = \|a_j + b_j\| − \|a_j − b_j\| = 2·min(\|a_j\|, \|b_j\|)·sign(a_j b_j) into each slot row. Then a block-11 head at the shared "###" decoder row attends with logit ∝ φ_j and copies name(j). So Δ ∝ φ(r_A) − φ(r_B), which is > 0 iff A is the target. Assumption: the per-name score is name-bound, with own-slot weight κ_own > cross weight κ_cross. "Predominantly own slot" is needed only for the route that applies φ after retrieval. **anti-lin 1.000 evidences exactly κ_own > κ_cross**: without binding, anti-lin reads ½. Toy: φ readout = 0.972 / 0.997 / 1.000 at m 1.5 / 2 / 3, at or above the rule oracle; a best linear order-blind readout gets 0.50; anti-xor through the same φ gets 0.498, as the Frame Lemma predicts. A shared MLP trained by gradient on raw rows reaches 0.959 / 1.000. | Add the construction and the assumption to §5 (E5). |
| 1b-LN | Sign products through LayerNorm | **PASS** | LN is positively homogeneous in the centred row. For any readout w, sign(w·(LN(x+δ) − β)) = sign(ŵ·(x+δ)) with ŵ = P(γ ⊙ w), **exactly**. Sign products relative to the β-offset survive exactly. A unit with a nonzero threshold c sees it scaled by 1/σ(x+δ): jitter O(c·CV(σ_row)), plus a second-order term σ² ∋ 2x·Pδ that depends on the signs (≈ 1 % of σ² at m 3, ε_rel ≈ 0.13). Toy with GPT-2-like outlier dims: per-row sign agreement before vs after LN is 0.976 at m 1.5 and 0.9994 at m 3. A learned readout loses nothing (0.958 vs 0.959; 1.000 vs 1.000). | None. |
| 2 | A1's own plant directions (u, u2 from A1's PCs 40–60, rng 22550; ε = √λ50) | **CORRECTED** | **Validity needs no orthogonality assumption.** (T, s1, s3) are independent of the item and of A1, so every function of A1's unplanted inputs reads T at exactly ½. A VALID pass therefore cannot come from A1's own signal: it requires reading the plant. In the toy, "always pick the gold row" reads 0.488–0.524 for every gold-signal size. "Harder, never easier" is **not** a theorem: a structured (non-Gaussian) base along u can make a plant easier than a Gaussian base of the same variance (toy 1.000 vs 0.726). It is not needed, and holds only when the signal is independent of the rest of the base along u. The K2 guard handles swamping, and conservatively: toy K2-cexor rule 0.9996 → 0.978 → 0.904 at a gold signal of 0 / 2 / 3 base-sd along (u+u2)/√2, while a learned probe still reads 0.999. So the guard over-fires before the probe loses power. It fails safe (PROBE-INVALID), and mostly in early-answer (precompute-like) models. Gaps: the guard's t is not stated; its centring and streams are not stated. | State the assumption precisely (E3). K2 guard at **each** t ∈ {0, 1, 2}, centred on A1's own pooled unplanted option-row mean, on pinned streams (E2). Add the descriptive DIAG-ALIGN, read only with A1's probe parts (E2). |
| 3 | OC | **CORRECTED** | Inputs check against v1_oc.json: 0.063 / 0.398 / 0.0043, reproduced. (i) **Partial double count:** v1's 0.063 already prices a PLANT certificate at ≈ 0.95 (LANG_v1 §6 row "PROBE-VALID / FD-VALID ≈ 0.95"; it sits inside RT's "fixed" serial rate 0.50 vs 0.55 "current"). The replacement factor is 0.9 / 0.95, not 0.9. P(CERT) is a genuinely new gate and is correctly multiplied. (ii) The stated "\| serial use 0.36" **omits P(CERT)**. Under the censoring rule (no certifiable probe = UNREAD = fail) it is 0.23. (iii) The S2 branch (S1 fails, S2 certifies) is not counted: ≈ +0.003. Corrected: **P(VB-C1-L-S) ≈ 0.036 (S1 branch; ≈ 0.039 with S2's branch at P(CERT S2 \| ¬S1) 0.3); \| serial ≈ 0.23 (0.38 given CERT); \| no serial ≈ 0.0025 (≤ 0.004 holds).** Sensitivity over P(CERT) 0.3–0.8: P 0.018–0.048. | Replace §6's S1 bullet (E6). |

**Required v1 edits:** E1–E6 below (exact text). None changes a registered bar. E2 adds one descriptive row and states the K2 guard's t, centring and streams. E6 changes only OC numbers.

---

## 1. FRAME check for PLANT-xor-S

### 1(a) The label is not recoverable from the item [PASS]

**Setup (code-checked: s218_gl5_known.cexor_draw / plant_V_cexor).**
- T ~ Bernoulli(½) is drawn from generator rc. s1 and then s3 are two sequential draws from generator rs; s2 = s1 and s4 = −s3.
- Target row += mε(s1u + s1u2); the other row += mε(s3u − s3u2).
- Items come from PS.Stream(22500) and batches from 22501. Coins and signs come from other seeds and are drawn by batch position, not by item identity: a pool item drawn twice gets fresh coins.

**Claim [PROVED].** Let Z = (item text, display order, names, slot positions, gold, A1's parameters, u, u2, ε). The plant randomness (T, s1, s3) is independent of Z by construction. So for any measurable f of the **unplanted** inputs, P(f = T) = ½ exactly.
- In particular, display order, name content and gold carry nothing about T.
- In the toy, an order-aware logistic fit on rich unplanted features (both rows, their projections, products and squared difference) reads 0.498 test vs 0.537 train (overfit only), and agreement(gold, T) = 0.501.

**Controls.**
- **cexor-NOPLANT is an exact null.** The inputs are independent of the training and held-out labels, so for **any** trained probe f the held-out correctness indicators are i.i.d. Bernoulli(½) given f. At n = 1,000, P(acc > 0.55) = 0.0008. LATE averages three reads of the same items, so P(LATE > 0.55) ≤ 0.0024 per cell, or ≈ 0.014 over the 6 (t, model) cells. Fine.
- **The shuffled-label null is ½ only in expectation.** The probe is trained on permuted targets but read on true T over **planted** rows. A label-blind function of planted rows is not forced to ½: the two classes differ in the *axis* of the plant (±(u + u2) vs ±(u − u2)). So a readout whose second-order response distinguishes those axes (a nonzero uᵀHu2 term) correlates with T without ever seeing a label.
  - Over the direction draw the bias averages zero, since u2's sign is random. For the one fixed draw it is a fixed number.
  - Toy magnitude: random 2-layer readouts on planted rows give |bias| mean 0.002, max 0.007 (m 3 and 10). With a name-prior term 5× the plant-driven spread, max 0.010.
  - A **fixed, memorisable** permutation fit on features that include a·b deviated to 0.44 at n 20k. The feasibility code uses a fresh **per-batch** permutation (rng 22597), which avoids memorisation. Pin that.
  - Risk to CERT: small, but it is not a binomial null. The bar ≤ 0.55 is fine if the per-batch permutation is pinned. Its role is a plumbing smoke (REVIEW_P2 §6), not leak evidence. NOPLANT is the leak check.
- **Stream mapping is unpinned (must fix).**
  - §3 lists "plant z / coins 22751–22758" and "PCs 22750" without roles.
  - §1 says u, u2 use "rng 22550". boxes/langvb_eval.plant_dirs hard-codes 22550, so a fresh PCs seed needs a code parameter.
  - No stream is named for the shuffled permutation or for the NOPLANT coin.
  - **A1's PLANT-xor-S needs three arrays (T, s1, s3)**, but §218's registered xor draw (zdraw on 22552 / 22554) yields two.
  - Fix E1. Minor hygiene: 22700 is also §216's pool_seed (scratch/s216_build_pool.py). That is a different generator and different data, so it is harmless, but record it.
- Wording: "linear accuracy on either row is ½" is exact for the order-blind comparative linear score w·(r_A − r_B): its law is symmetric about 0 given either label, and the toy's best w in span(u, u2) reads 0.50. A single-row linear classifier *with an offset* can sit slightly off ½, through the different spreads along w. That is irrelevant to an order-blind readout.

### 1(b) Positive construction [PASS]

**Read path (code-checked).** Δ = Σ_k log p(tok_k | ·) over "### name_A." minus the same over "### name_B.".
- block_bank is causal over [slots | decoder] (vbank.block_bank, triu mask). So slot rows never see decoder rows, and the "###" decoder row is identical in the A and B runs: same token, same position T_i, same slots.
- So Δ = (W_U[name_A] − W_U[name_B])·h_### + [log p("." | A's name row) − log p("." | B's name row)] (+ further name tokens if a name has several).
- The first term is a logit difference from a **shared** hidden state.
- Probe parameters: all of blocks 10–11 (h_10_*, h_11_*) plus β / γ. ln_f and the tied unembedding are frozen.

**Construction (exact representability).**
1. **Block 10, MLP at each slot row j.** Read a_j = ŵ_u·LN₂(h_j) and b_j = ŵ_u2·LN₂(h_j), with ŵ pulled back through LN as in the next subsection. Write φ_j = relu(a+b) + relu(−a−b) − relu(a−b) − relu(b−a) = |a_j + b_j| − |a_j − b_j| into an unused residual direction e_g. This is 4 units of 3,072.
   - φ_j = 2·min(|a_j|, |b_j|)·sign(a_j b_j): positive iff the signs agree.
   - Block-10 attention cross-talk from earlier slots into h_j is a power issue. It can be zeroed on span(u, u2) by W_OV, since the parameters are free.
2. **Block 11, one head at the "###" row.** The query is constant (the "###" token) and the key reads e_g, so the attention logit is κ·φ_j/σ_j. The OV circuit copies the slot's name identity to W_U[name(j)]. That is a copy head through the tied embedding, since the slot row is the residual at the concept's first-mention token.
   - Then logit(name_A) − logit(name_B) = c·(α_A − α_B) + (plant-free terms). α_A > α_B iff φ_A/σ_A > φ_B/σ_B, and σ_A ≈ σ_B (toy CV of σ_row ≈ 0.014–0.017).
   - The target row has φ > 0 and the other row φ < 0 (at m 3 each row's sign pattern is right with probability ≈ 0.9997). So Δ > 0 iff A = T.
3. **Alternative route.** Each name row retrieves its own slot through attention in block 10, and the MLP at the name row computes φ of the retrieved vector, which moves the "." logit.

**Assumption, stated precisely.** Write the per-name score's plant-dependent part as s(V, n_X) = Σ_j κ_{X,j} φ(r_j). Exchange symmetry gives κ_{A,A} = κ_{B,B} = κ_own and κ_{A,B} = κ_{B,A} = κ_cross, so Δ = (κ_own − κ_cross)(φ(r_A) − φ(r_B)) + plant-free terms.
- Route 1 (φ before the binding) needs only **κ_own > κ_cross**.
- Route 2 (φ after retrieval) needs retrieval dominated by the own slot, because φ(Σ α_j r_j) ≠ Σ α_j φ(r_j). That is where "attends predominantly to its own option slot" belongs.
- **Evidence:** anti-lin LATE 1.000 on the untrained model and on the reference (218-24).
  - Anti-lin's Δ = (κ′_own − κ′_cross)·2mε(2z − 1) + noise. Without a name-bound read (κ′_own = κ′_cross), anti-lin is exactly ½.
  - So 1.000 shows the binding exists and is learnable within budget. Cexor needs the same binding plus the 4-unit nonlinearity upstream of it.
  - Learnability at m ≤ 3 is empirical: ×10 → 1.000 at t 0. In the 2k-step feasibility, ×1.5 → 0.748 at t 0 and 0.943 at t 1. The t 1 file landed at 08:38, after PS_v1 (08:34); it is read here descriptively only.

**Toy (sim_ps.json, ε/σ_row = 1.13, calibrated to K2-anti diff_proj 0.9465 / 0.991 at ×1 / ×1.5):**

| m | rule oracle | analytic p² + (1−p)² | order-blind φ (4 ReLU) | attention-copy surrogate | best linear order-blind | anti-xor through φ |
|---|---|---|---|---|---|---|
| 1.5 | 0.915 | 0.914 | 0.972 | 0.972 | 0.502 | 0.498 |
| 2 | 0.977 | 0.976 | 0.997 | 0.997 | 0.502 | 0.498 |
| 3 | 0.9993 | 0.9993 | 1.000 | 1.000 | 0.501 | 0.498 |

- A shared-weight MLP s(row), trained only on Δ = s(r_A) − s(r_B), with u unknown, raw rows, and outlier dims on, reads 0.959 / 1.000 at m 1.5 / 3.
- **Prediction (descriptive, not a rule):** with per-row SNR ≈ 1.13m, the K2-cexor rule reads ≈ 0.915 / 0.977 / 0.9993 at m 1.5 / 2 / 3, so **the m-rule likely selects m = 3**.
  - If the two option rows share a common-mode base component, the per-row SNR is lower than the differenced K2-anti calibration implies, and "ceiling below 0.985" at m 3 becomes possible.

### 1(b)-LN: sign products through the coda's LayerNorms [PASS]

- LN(x) = γ ⊙ Px / σ(x) + β, with P = I − 11ᵀ/d and σ(x) = |Px|/√d > 0. For any w, w·(LN(x+δ) − β) = ŵ·(x+δ) / σ(x+δ), with ŵ = P(γ ⊙ w).
- So **sign(w·(LN(x+δ) − β)) = sign(ŵ·(x+δ)) exactly**, and products of such signs survive exactly. Choosing w = P u ⊘ γ makes ŵ ∝ Pu ≈ u: u's component along 1 is O(d^-½) for a random combination of PCs 40–60.
- **Where it is not exact:**
  - (i) A unit with a threshold c ≠ 0 (for example population centring, when the row mean projects onto ŵ) has its threshold rescaled by 1/σ(x+δ), which varies by row. The jitter is O(c·CV(σ)).
  - (ii) σ² contains 2x·Pδ, which depends on the signs. It is ≈ 1 % of σ² at m 3, so second order in ε_rel.
  - |Pδ| is the same for all four sign patterns (u ⟂ u2 by Gram–Schmidt in plant_dirs).
- Toy with outlier dims (+60 / −40, GPT-2-like):
  - population-centred rule: 0.916 before LN vs 0.904 after LN at m 1.5; 0.9992 vs 0.9987 at m 3;
  - per-row sign agreement 0.976 / 0.9994;
  - a learned readout after LN: 0.958 vs 0.959 raw (m 1.5), 1.000 vs 1.000 (m 3).
- The plant stays in the slot row's residual through both blocks, so block 11's ln_1 and ln_f see it too. The same argument applies.

---

## 2. A1's own plant directions [CORRECTED]

**2.1 Can u or u2 carry A1's real option information, so that VALID passes on A1's signal?**
- No, by 1(a). The plant label T is an independent coin. A1's real signal is a function of the item (gold, reach, side), which is independent of T.
- Any accuracy above ½ on T therefore requires a dependence on the planted perturbation. A PROBE-VALID-S pass certifies that the probe recipe reads the plant in A1's coda and rows, **whatever u's alignment**.
- Toy: with a gold signal of 0–3 base-sd along (u+u2)/√2, "always pick the gold row" reads 0.488–0.524 (sampling).
- The plant probes are separate Probe instances (part_probe: one Probe per key), so no gold-trained weights enter them.

**2.2 Does A1's signal make the plant harder, never easier?**
- **Harder: yes, in the case that matters.** Gold is independent of T, so for the plant task it is nuisance variance along u and u2. The probe must also override A1's trained gold readout, which starts inside the coda copy.
  - The toy shows monotone loss for fixed rules (K2 rule 0.9996 / 0.999 / 0.9955 / 0.991 / 0.978 / 0.904 at μ = 0 / 0.5 / 1 / 1.5 / 2 / 3 base-sd) and very little for a learned probe (≥ 0.999 throughout).
- **"Never easier" is not a theorem.** If the nuisance along u is structured, e.g. discrete ±μ with little noise, a plant can be read *better* than under a Gaussian base of equal variance (toy: 1.000 vs 0.726). Likewise, "adding" a signal that cancels other base content along u reduces the conditional variance.
  - The claim holds only if the u-component of A1's signal is independent of the rest of the base along u, given the probe's other inputs.
- **Neither direction threatens validity**, because "easier" still means reading the plant. What VALID cannot certify, under any alignment, is power on codes in *other* directions or forms (LANG_v1 §3.5: a power certificate for one planted code class).
- Drop "harder, never easier" as a premise. Use 2.1 instead.

**2.3 Does the K2 guard handle swamping?**
- Yes, conservatively. The rule oracle uses fixed signs of centred projections, so a large nuisance along u lowers it first.
  - In the toy it crosses 0.97 at μ ≈ 2.2 base-sd, while the learned probe still reads 0.999. The guard over-fires (PROBE-INVALID = not met), which fails safe.
  - The plants sit at t ≤ 2, where large option-row gold signal for D ≥ 3 is a precompute-like trait. So over-firing mostly censors non-serial models, and P(pass | no serial) can only fall.
  - K2-anti-lin is a logistic fit on full [r1, r2] (k2anti). It can partly regress out a nuisance that other dims make decodable, so it is the less conservative of the two guards.
- **At t ≥ 2** the coda also reads the **unplanted** bank versions 1..t−1 of the same slots. The probe can then difference r_t − r_{t−1} and remove stable base content, including stable gold signal. K2, which uses V_t rows only, is a lower bound on the available information there, so the guard is again conservative.
- **Gaps to close (E2):**
  - (i) The guard is stated "on A1's own planted rows at m" with no t. It must hold at **each** t ∈ {0, 1, 2}, each paired with that t's VALID.
  - (ii) The centring must be A1's own pooled unplanted option-row mean at that t, as k2cexor does. plant_dirs' mean is over all real slots, not option rows.
  - (iii) Pinned streams.
  - (iv) k2anti in the feasibility code reads V_1 only (rows_V1). The m-rule's t 0 needs a rows_Vt version: an implementation note for score/eval code.
  - The K2 runs read no answer information, so they may run before A1's probe parts (REVIEW_P2 §7).

**2.4 The precise assumption.**
- (I-1) For every item, (T, s1, s3) come from registered generators restarted by role (E1), drawn by draw position. They are independent of the item text, display order, names, gold and A1, and mutually independent.
- (I-2) (u, u2, ε) are functions of A1's unplanted pool rows and the coefficient generator only, never of coins or labels.
- **Under (I-1) and (I-2), every function of A1's unplanted inputs reads T at exactly ½, for any alignment of u and u2 with A1's answer subspace.** No orthogonality is assumed.
- Power is not assumed: it is measured by the K2 guards on A1 at each t.

**2.5 Cheap diagnostic: DIAG-ALIGN (descriptive, $0).**
- On A1's unplanted held-out option rows at t ∈ {0, 1, 2}, centred by the guard's μ, with a_X = (r_X − μ)·u and b_X = (r_X − μ)·u2 for X ∈ {gold, other}, report:
  - (i) d′_u and d′_u2 = mean(a_gold − a_other) / sd(a_gold − a_other), and the gold-classification accuracy of sign(a_o1 − a_o2);
  - (ii) the cexor-rule "accuracy" for gold on unplanted rows: does sign agreement pick gold?
  - (iii) cos²(u, ĝ_t) and cos²(u2, ĝ_t), with ĝ_t the difference-of-means direction mean(r_gold − r_other), against the chance level 1/768;
  - (iv) var(option rows along u) / λ50, the plant's per-row SNR calibration on option rows.
- **Built-in null:** at t = 0, (i)–(ii) are 0 / ½ in expectation (V_0 is a function of Z; §3.5).
- Toy: d′-type correlation 0.00 / 0.42 / 0.70 / 0.84 at μ = 0 / 0.5 / 1 / 1.5.
- **Timing:** (i)–(iii) read answer information. They are a two-direction linear probe, so they belong to A1's **probe parts**: computed and read only with them, after CERT. They never gate.

---

## 3. OC [CORRECTED]

**Inputs checked.**
- v1_oc.json headline: P(VB-C1-L) 0.063, | no serial 0.004 (0.0043), | serial 0.398. Reproduced in sim_ps.json from v1_oc.py's mixture: P(entry) 0.55 × P(DEEP | entry) 0.60; classes 0.35 / 0.10 / 0.05 / 0.30 / 0.10 / 0.10; pass rates 0.50 / 0.45 / 0.02 / 0.02 / 0.02 / 0.10; CF factors 0.9 / 0.9 / 0.9 / 0.02 / 0.02 / 0.2; band factor 0.95.
- The no-serial 0.0043 is dominated by the mixed one-pass class (0.10 × 0.2 → 0.002 of 0.0022 class-weighted mass). "Set by CF-L and the non-probe components" is fair wording.

**Is multiplying by P(VALID-S on A1) a double count?** Partly.
- RT's per-class pass rates (oc_shade.py) are single composite numbers for the "fixed" design, which **includes** PLANT-certified probes. Serial_qcond went from 0.55 ("current", no PLANT) to 0.50 ("fixed").
- LANG_v1 §6 lists "PROBE-VALID / FD-VALID (PLANT) … ≈ 0.95" under effect.
- So 0.063 already contains a ≈ 0.95 VALID factor, priced for a plant that in fact had P(VALID) = 0 by construction (218-25). The new VALID-S factor replaces it: × 0.9/0.95 = × 0.947, not × 0.9.
- REVIEW_P2 §8's k used the same convention (VALID = 1.0 reproduces 0.398), so its S2 figures carry the same ≈ 5 % understatement.
- **P(CERT) is new and correctly multiplied.** It is a gate on separate models, independent of A1's mechanism class, so it multiplies every class, including | no serial.
- P(VALID_A1) should be read as conditional on CERT, because the two are positively correlated.

**The "| serial use" figure.**
- The design's 0.36 = 0.398 × 0.9 omits CERT. By §7 ("no certifiable probe → successor readings UNREAD") and the user's censoring rule, never-certified counts as a fail.

**Corrected (S1 branch; sim_ps.json):**

| P(CERT S1) | P(VALID-S_A1 \| CERT) | P(VB-C1-L-S) | \| serial (CERT fail = fail) | \| serial, given CERT | \| no serial | design formula 0.063 × 0.9 × CERT |
|---|---|---|---|---|---|---|
| 0.3 | 0.9 | 0.018 | 0.11 | 0.38 | 0.0012 | 0.017 |
| 0.5 | 0.9 | 0.030 | 0.19 | 0.38 | 0.0020 | 0.028 |
| **0.6** | **0.9** | **0.036** | **0.23** | **0.38** | **0.0025** | 0.034 |
| 0.6 | 0.8 | 0.032 | 0.20 | 0.34 | 0.0022 | 0.034 |
| 0.7 | 0.9 | 0.042 | 0.26 | 0.38 | 0.0029 | 0.040 |
| 0.8 | 0.9 | 0.048 | 0.30 | 0.38 | 0.0033 | 0.045 |

- **The S2 branch:** P(¬CERT S1) × P(CERT S2 | ¬CERT S1) × P(pass | S2). With 0.4 × 0.3 [ASSUMED; S2 feasibility cexor 0.496 at t 1 ×1.5] × 0.027 × (1/0.95) (REVIEW_P2's middle row, de-double-counted), that is **≈ +0.003 headline**, or ≈ +0.02 on | serial.
- **Headline ≈ 0.036–0.039; | serial ≈ 0.23–0.25; | no serial ≈ 0.0025 (≤ 0.004).**
- The numerical change to the headline is immaterial (0.034 → 0.036). The material corrections are the | serial figure (0.36 → 0.23) and the explicit S2 term.
- **P(CERT S1) ≈ 0.6 [ASSUMED] is plausible.** At 2k steps, ×1.5 gave 0.748 at t 0 (plateau) and 0.943 at t 1. The likely m is 3, with 10k steps. CERT needs 12 plant cells (2 plants × 3 t × 2 models) ≥ 0.90 plus 3 controls.
- Realised $: $0, unchanged.

---

## 4. Exact v1 edits

**E1 (§3, replace the "Fresh streams for certification" bullet):**
> - **Fresh streams for certification (roles pinned; held-out / test generators restarted at every read; no stream serves two roles):** pool / held-out PS.Stream 22700 (also §216's pool_seed, under a different generator and data; recorded, harmless); batches 22701; plant_dirs coefficient generator 22750 (a parameter replacing the hard-coded 22550 for certification only); PLANT-lin-S z: train 22751, held-out / test 22752; PLANT-xor-S coin T: train 22753, held-out / test 22754; PLANT-xor-S signs (s1, then s3, two sequential draws of one generator; s2 = s1, s4 = −s3): train 22755, held-out / test 22756; cexor-NOPLANT coin: train 22757 (held-out reads 22754's coins); shuffled-label control: a fresh permutation of each training batch's targets, generator 22758 (never a fixed permutation of the pool). Never used in feasibility.
> - **A1's plant streams:** u, u2, ε from rng 22550 (§218's plant_dirs) on A1's own pool rows; PLANT-lin-S z as §218 (22551 train / 22553 held-out and test); PLANT-xor-S coin T = the first array and s1 = the second array (0 → −1, 1 → +1) of §218's xor draw (22552 train / 22554 held-out and test); s3 from a new generator, 22558 train / 22559 held-out and test.

**E2 (§4, PROBE-VALID-S bullet; replace its second sub-bullet and add a descriptive row):**
> - **and**, at each t ∈ {0, 1, 2} paired with that t's plants, K2-anti-lin ≥ 0.985 and K2-cexor ≥ 0.97 on A1's own planted option rows at m. K2-cexor is the rule oracle, centred on A1's pooled unplanted option-row mean at that t from its K2 train split. K2 streams: coin / signs 22611 / 22612 train and 22613 / 22614 test, lin z 22571 / 22572; as in feasibility, and no answer information is read. Below either bar at any t, that t reads "plant at ceiling on A1" = PROBE-INVALID(t). The K2 runs may run before A1's probe parts.
> - **DIAG-ALIGN (descriptive, never gates; computed and read only with A1's probe parts):** on A1's unplanted held-out option rows at t 0–2, centred as K2: d′ of (r_gold − r_other)·u and ·u2; the gold accuracy of sign((r_o1 − r_o2)·u) and of the cexor sign-agreement rule; cos²(u, ĝ_t) and cos²(u2, ĝ_t) for ĝ_t = mean(r_gold − r_other), against 1/768; var(option rows along u, u2) / λ50. At t = 0 the gold readings are ½ / 0 in expectation (§3.5 of LANG_v1).

**E3 (§5, add a bullet after "D3 directions"):**
> - **Plant validity on A1 needs no orthogonality [PROVED].** The coin T and the signs (s1, s3) come from registered generators, independent of the item (text, display order, names, gold) and of A1, and of each other. u, u2 and ε depend only on A1's unplanted pool rows. Hence every function of A1's unplanted inputs reads T at exactly ½, however u and u2 align with A1's own answer information, and a PROBE-VALID-S pass requires reading the plant. A1's own signal along u and u2 is nuisance for the plant task. It can make the plant harder (the K2 guards at each t catch swamping, conservatively), and it can never produce a pass by itself. "Harder, never easier" is not claimed: it holds only for a nuisance independent of the rest of the base along u, and validity does not need it.

**E4 (§3, controls; append to the cexor-NOPLANT and shuffled-label lines):**
> (cexor-NOPLANT is an exact null for any trained probe: labels are independent of every input, so P(LATE > 0.55) ≤ 0.003 per cell at n 1,000. Shuffled-label is ½ only in expectation, because a label-blind readout of planted rows can correlate with T through the plant axes (u + u2 vs u − u2). It is a plumbing smoke with per-batch permutation, not leak evidence.)

**E5 (§5, replace the "Content-equivariant plants are readable by it [DERIVED]" bullet):**
> - **Content-equivariant plants are readable by it [DERIVED; construction in REVIEW_PS §1(b)].** 4 ReLU units in block 10's MLP write φ(r_j) = |a_j + b_j| − |a_j − b_j| (a = r·u, b = r·u2; > 0 iff the signs agree) into each slot row. A block-11 head at the shared "###" decoder row attends with logit ∝ φ_j and copies name(j), so Δ ∝ (κ_own − κ_cross)(φ(r_A) − φ(r_B)). **Assumption:** the per-name score is name-bound, κ_own > κ_cross. This is evidenced by anti-lin LATE 1.000 on both models (218-24), which is ½ without such a binding. LayerNorm preserves signs relative to its β-offset exactly (positive homogeneity), and up to second order in ε_rel otherwise. **[MEASURED in feasibility]** cexor at t 0 ×10 → 1.000.

**E6 (§6, replace the "Under S1" bullet):**
> - **Under S1:** v1's 0.063 / 0.398 already price a PLANT certificate at ≈ 0.95 (LANG_v1 §6), so PROBE-VALID-S on A1 enters as × P(VALID-S_A1 | CERT) / 0.95 ≈ 0.9 / 0.95 [ASSUMED]. P(CERT(S1)) ≈ 0.6 [ASSUMED from feasibility] is a new gate, independent of A1's mechanism, and a failed certification counts as a fail (§7). **P(VB-C1-L-S | serial use) ≈ 0.398 × 0.947 × 0.6 ≈ 0.23** (0.38 given CERT); **P(VB-C1-L-S | no serial use) ≈ 0.0025 (≤ 0.004)**; **P(VB-C1-L-S) ≈ 0.036** (0.018–0.048 for P(CERT) 0.3–0.8). The S2 branch (S1 not certified, S2 certified; P(CERT S2 | ¬CERT S1) ≈ 0.3 [ASSUMED]) adds ≈ 0.003, for a total ≈ 0.039.
