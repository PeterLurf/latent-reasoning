# PROOFS_SELFDEPTH: rigour pass on SELFDEPTH_draft.md (maths, OC, numerics)

Clock 2026-10-17 02:04 EDT. Opus proof subagent for Theory. $0, CPU only (1 thread), no training, no repo edits.
Draft reviewed: `docs/research_selfdepth_2026-10-17/SELFDEPTH_draft.md` (ac1f854d; identical to the scratchpad copy) and its five scripts.
Raw files read (explicit paths only): `boxes/runs_fsq/s213_d3{t,f}_s1/{eval_chainlabel,eval_chainlabel_fresh,eval_vb,eval_steppos}.json`, `s213_d3t_s1_mem/mem_check.json`, `s213_d3t_s1_gtime/timing.json`. Code read: `boxes/vbank.py` (recode_targets, rec_targets, vb_loss, CurGate, main/draw), `boxes/nested170.py` (generate, render, test_set, train_items), `boxes/vbank_diag.py` long_step, `boxes/fsq120.py` lr_at.

Check scripts and outputs (`research4/proofs_sd/`):

| script | output | what it checks |
|---|---|---|
| gate_oc_exact.py | gate_oc_exact.json | exact joint GATE OC (convolution DP), integer thresholds, φ Clopper–Pearson bounds, measured window π, stratum heterogeneity |
| support_check.py | support_check.json | Prop S1 exact (rationals) under the draft's law, the trainer's real plain law, stage A, loss weights; E[Kb]; pool budget under CurGate |
| noise_shift_oc.py | noise_shift_oc.json | Prop S2 bounds and κ thresholds, two-round factor; H_shift numbers, SE; STOP-LOC on base and exact H_shift; §6 OC arithmetic |
| compose_memT_checks.py | compose_memT_checks.json | chain-exact ⇏ COMPOSE-VALID; one-wrong-step chains; mem_check batch T; analytic item-space bound |
| cost_check.py | cost_check.json | cost / memory with the corrected T reference and AdamW state; simulated padded T |

Tags as in the draft. Verdicts: **[PROVED]**, **[PROVED UNDER X]**, **[FALSE: …]**, **[UNVERIFIED]**.

---

## 0. Verdict table

| claim | verdict | what changes |
|---|---|---|
| S0 support identity (one reduction per pass, ρ + r = n ≤ B_q) | **[PROVED]** (needs COMPOSE-VALID; KEEP alone does not check one operator per step) | none |
| S0(a) old stop pays loss | **[PROVED for the R2 term]**; R1 positive only if the coda does not finish the remaining reductions [ASSUMED] | wording |
| S0(b) "stop at B_q" is zero-loss | **[PROVED]** | none |
| S1 table | **[PROVED]**, reproduced exactly under the draft's law; under the trainer's real plain law P(R \| t ≥ 9) = **0.494**, not 0.500; stage A reads 0.369 | G-S3 target and per-stage scope |
| S2(a) π = P / (P + φ(1 − P)) ≥ 1 − φ(1 − P)/P | **[PROVED]** (identity). Per-n numeric bounds **[PROVED UNDER φ_n ≤ 0.0097 ∀n]**; that premise fails in point estimate at fresh n 15 (bound ≥ 0.9994, measured **0.9968**) | use direct window counts |
| "post-filter precision ≥ 0.997 at every n" (213-38, draft) | **[FALSE by rounding]**: min 0.99682 (fresh n 15); "≥ 0.999 at every n" (test) min 0.99889 | write ≥ 0.9968 / ≥ 0.9989 |
| S2(b) recursion | **[PROVED UNDER A1′, A2, A3′]** after two fixes: w is the share among **kept** (×1/P), and A1 must cover the **untrained** lengths m ∈ W_{q+1} the next labeller uses | restate |
| "avalanche needs κ ≥ 8 at n 12" | **[PROVED UNDER A1′–A3′]**: κ ≥ P/(φn) = 8.1–8.4 | ok |
| "κ ≥ 5 at n 20" | **[FALSE]**: 3.9–4.0 with the 1/P factor | fix |
| two-round growth ≤ 2.7× / 1.6× | **true but mislabelled**: it is the bound for eight +1 rounds (Σn = 100); two W-4 rounds give **≤ 1.32× / 1.15×** | fix |
| S2(c) class mass ≤ 2e-3 | **[PROVED]** (measured: ≤ 6.4e-4) | none |
| S3(a) "pass-1 term cannot flip its own argmax to first order" | **[PROVED UNDER a per-example step at the labeller's weights]**; in batch SGD **[ASSUMED]** | retag |
| S4 H_shift curve / mean 0.48 / THINK-GAIN 0.303 | **[COMPUTED]** given the hypothesis (itself [CONJECTURE]) | SE is 0.0062, not 0.008 (conservative) |
| S4 "under H_shift STOP-LOC ≈ 13–14"; base STOP-LOC "10" | **[FALSE under §5's own definition]**: both read **unreadable** (never finishing); SHIFT can never be labelled under H_shift | **fix the reading** |
| GATE-q OC 0.998 / ≈ 0.5 / ≈ 0.01 | **[PROVED, exact]**: 0.9986 / 0.504 / 0.0125 (R1) | covers the two value terms only |
| "state precision ≥ 0.99" | well-defined once the denominator (KEEP ∧ CV) and t-range (0..n) are fixed; **π^E ≤ π [PROVED]** | spec text |
| R2 composition "on chain-exact chains E_self = recode_targets" | **[FALSE as worded]** for 213-33's string-level chain-exactness (counterexample 1,000/1,000). **[PROVED]** for COMPOSE-VALID ∧ all step values correct | fix premise |
| "COMPOSE-VALID extra drop ≈ 0 because every value-correct chain is chain-exact [MEASURED]" | **non sequitur → [CONJECTURE]**; slot placement and frame are unmeasured at m ≥ 9 | S0-1(a) decides |
| "every value-correct chain is chain-exact" | **[FALSE as worded]** (right-value-wrong-steps chains); true: strict ⇔ chain-exact | wording |
| item space: 0 hits / 0 dups "in 50k at every n 4–20" | counts reproduce, but n 4 had **5k** draws; **[PROVED]** analytically for n ≥ 9 at any pool size | wording |
| test_set(n_max 20) ≡ test_set on n ≤ 16 | **[PROVED]** (code) and measured | none |
| K budgets, R1 window vs vb_loss | **[VERIFIED]**; three spec gaps (R1 label source, rng for self K, mem_check timing) | add to §2.4 |
| 480k pool suffices, fresh-once | **[FALSE: counterexample]** at n 9–10 when stage A lasts ≥ 3k steps (128k needed at the 4k cap) | resize per n |
| cost / memory | arithmetic reproduces; **inputs wrong** (T reference 31 not 33; AdamW state omitted). Fallback kextra 2 needs micro 2 ($0.48–0.91); round-2 kextra 2 needs micro 4 ($1.04–2.55 > $2.0 cap) | fix table |
| OC P(SD-SELF) 0.45, MARGIN 0.25, SD-C2 0.075, \|no effect ≤ 0.01, \|full 0.62 | **[COMPUTED]** exactly from the stated priors (0.448 / 0.256 / 0.0763 / 0.0069 / 0.620); priors [ASSUMED] | add P(SD-SELF \| no effect) ≈ 0 |

---

## 1. Prop S0: support identity under self-labelling

**Statement (precise).** Let the round-q mixture contain gold items (n ≤ 8, targets recode_targets(root)) and self items (n ∈ 9..B_q, KEEP ∧ COMPOSE-VALID chain, targets E_self). Let ρ*_t(i) be the number of reductions in R2 target E_t(i). Then ρ*_t(i) = min(t, n_i) for every t ∈ 0..K_i, and n_i ≤ B_q. Hence every target state has ρ + r = n_i ≤ B_q.

**Proof.** Gold: recode_targets reduces one leftmost-ready handle per t, then HOLDs (vbank.py l.32–47). Self: SPAN ∧ ORDER at step k say that dec_k VOIDs exactly the owned slots of one node whose children are both leaves, except the last, which takes one value class. So the live operator count drops by exactly 1. STEPS gives exactly n steps to reach zero operators. E_self HOLDs after n. Values never enter the count. ∎ **[PROVED]**

- KEEP alone is not enough: KEEP = PARSE ∧ STEPS, and compose.py computes ONE_OP but does not use it. A chain with a 0-operator step and a 2-operator step passes KEEP. This does not matter for the draft, which builds R2 only from COMPOSE-VALID chains. An R1-only use of KEEP-but-not-CV items would not touch S0.
- **(a)** At a pass t ∈ (8, n_i], E_t has one more reduction than E_{t−1}. A model whose argmax state is the HOLD of E_8 is wrong on the changed slots. Its per-slot CE is ≥ log 2 (target probability ≤ ½ when the argmax is wrong), and the changed-slot half carries half the R2 term. So the loss is positive. **[PROVED for R2]**. For R1 (window t ∈ [n_i, K_i]) the loss is positive unless the coda completes the remaining 1–4 reductions in one shot (the §4 one-pass precompute route). That is **[ASSUMED]**, not proved. "Unlearned at the optimum" stays [ASSUMED], as the draft says.
- **(b)** "ρ ≥ B_q ⇒ finished" and any pass deadline t_d ≥ B_q + 1 agree with every target, because no support item has n > B_q. **[PROVED]**
- **(c)** n 15–16 are read at K′ 17–18, past the self kmax of 16. The same holds for the base: B 8, kmax 12 = B + 4, n 11–12 read at K′ 13–14. **[PROVED]**. The H_shift analogy is structurally consistent: same B + 4 margin, same "n B+3 and B+4 need K′ > kmax", and trained T 33 vs 47 against test T means 29 / 41 at n B + 1.

## 2. Prop S1: pass-level support (support_check.py, exact rationals)

**Checked against vb_loss** (vbank.py 462–503, draw 641–646):
- per-item K_i = min(n_i + U{0..kextra}, kmax) from rng_k;
- R1 at t ∈ [n_i, K_i];
- R2 at t ∈ 1..K_i;
- the rollout runs to Kb = max K_i, and passes t > K_i carry no loss.

So the draft's per-item pass accounting is the correct one. kmax never binds: n + 4 ≤ 12 (plain), ≤ 16 (round 1), ≤ 20 (round 2).

| t* / t_d | 9 | 10 | 11 | 12 | ≥ 13 |
|---|---|---|---|---|---|
| draft law (plain U{1..8}), P / E# | 0.500 / 1.25 | 0.400 / 0.75 | 0.283 / 0.375 | 0.145 / 0.125 | 0 / 0 ✓ |
| **trainer law** (train_items redraws n on excluded draws; n 1 at 11.5 % weight), P | **0.494** | 0.397 | 0.282 | 0.145 | 0 |
| trainer law, loss-weighted (R2 averaged over t per item) | 0.490 | 0.394 | 0.281 | 0.145 | 0 |
| deadline cost in R2 units (share of an item's R2 term) | 9.8 % | 5.7 % | 2.8 % | 0.9 % | 0 |
| **stage A** (self n 9–10), P / E# | 0.369 / 0.75 | 0.180 / 0.25 | 0 | 0 | 0 |

- Round 2 reproduces (0.643 … 0.143 at t* 16 under the draft's law, 0.637 … under the trainer's; 0 at ≥ 17).
- The zeros at t ≥ B + 1 are exact under every law **[PROVED]**.
- **G-S3 fix:** the target is **0.494** (stage B only). Stage-A draws must be excluded or checked against 0.369.
- Cosmetic: support_oc.py's "D3-T base" row prints max_rho_plus_r 9 because a weight-0 self n = 9 is in the law.

## 3. Prop S2: label precision and noise propagation

### (a) Post-filter precision

**Identity [PROVED].** Every strict chain passes KEEP: it parsed at every step and took n steps. So kept = P + φ(1 − P), π = P/kept, and 1 − π = φ(1 − P)/kept ≤ φ(1 − P)/P.
- For the gate's KEEP ∧ CV filter, replace P by Pψ, with ψ = P(CV | strict) (unmeasured), and φ by φ_CV ≤ φ.

**φ [COMPUTED from raw counts].**
- D3-T: 2 / 285 (test) + 4 / 939 (fresh) = 6 / 1,224 = **0.0049**. Clopper–Pearson one-sided 95 % upper **0.00965** (the draft's "≈ 0.010" ✓).
- D3-F test is **0 / 541**, not 0 / 528. D3-F fresh is 1 / 2,821 (n 14).

**Per-n bounds.** With φ ≤ 0.00965 the bound gives π ≥ 0.99976 (n 12), 0.99913 (n 16), 0.9973 (n 20) **[PROVED UNDER φ_n ≤ 0.00965 at that n]**.
- That premise is **contradicted in point estimate at fresh n 15**: 3 kept-wrong of 60 non-strict chains (φ̂ 0.05; test + fresh 3 / 128, 95 % upper 0.060). The bound says π_15 ≥ 0.9994; measured 940 / 943 = **0.99682**.
- Per-stratum p = 0.0033. Family-wise over the 20 strata, p = 0.24 under φ-homogeneity and 0.07 under per-kept homogeneity. So it is not significant, but the per-n bound is a model, not a guarantee.
- **The evidence to cite is the direct window count** (test + fresh, D3-T):

| window | kept | kept-wrong | π̂ | 95 % lower |
|---|---|---|---|---|
| R1 n 9–12 | 7,919 | 1 | 0.99987 | **0.99940** |
| R2 n 13–16 | 7,542 | 4 | 0.99947 | **0.99879** |
| R3 n 17–20 | 3,321 | 1 | 0.99970 | **0.99857** |

- "≥ 0.997 at every n" (213-38 and the draft) is **false by rounding**: the minimum is 0.99682. "≥ 0.999 at every n 9–16" (test) has minimum 0.99889. Write ≥ 0.9968 and ≥ 0.9989.

### (b) Recursion: corrected statement

- **Definitions.** e_q(m) = 1 − s_q(m), and ē_q(n) = (1/n) Σ_{m ≤ n} e_q(m).
- **A3′** (weaker than A3): the fresh-step error at a chain's k-th state equals e_q(n − k + 1). Independence is not needed. The union bound gives 1 − P_q(n) ≤ n ē_q(n) under any dependence. A3′ is supported by the product fit (D3-T predicted 0.997 … 0.913 reproduces from eval_steppos long_step).
- **w is the kept-wrong share among KEPT items.** The pool is filled to a kept count. So w_q = φ(1 − P)/(P + φ(1 − P)) ≤ φ n ē_q / P_q. The draft's w ≤ φ(1 − P) is **[FALSE]**: it is the share among drawn items. The factor is ×1.02 at n 12 and ×1.28 at n 20.
- **A1′ (coverage):** fine-tuning on share w raises e at **every m ≤ B_{q+1}** by ≤ κw. As written, A1 covers only "the trained lengths". But round q trains fresh steps only at m ≤ B_q, and the round-(q+1) labeller needs steps at m ∈ W_{q+1}. Those lengths are untrained (S3(d) is a conjecture about exactly them). So the recursion needs A1′ **[ASSUMED]**. GATE-(q+1) and LABELLER-KEPT are what actually guard it.
- **Result [PROVED UNDER A1′, A2, A3′]:** ē_{q+1} ≤ ē_q (1 + κφ n_q / P_q), so ē_Q ≤ ē_0 Π_q (1 + κφ n_q / P_q).
- **Avalanche** (the bound at least doubles per round) iff κ ≥ P_q / (φ n_q):

| n | draft 1/(φn) | corrected, φ 0.010 | corrected, φ 0.00965 | per wrong transition κ′ = κn |
|---|---|---|---|---|
| 12 | 8.3 | **8.1** | 8.4 | ≥ 101 |
| 16 | 6.3 | 5.7 | 5.9 | ≥ 95 |
| 20 | **5.0** | **3.9** | 4.0 | ≥ 81 |

- **Units.** κ is per unit of item-level wrong share. A kept-wrong CV chain carries one wrong transition among n, plus a wrong R1 target. So in per-transition units the margin is about 100×. The draft's "8× its own mass" understates it, which is the conservative direction.
- **Two W-4 rounds** (n_q = 12, 16), κ = 1: the factor is ≤ **1.32** (φ 0.010) or ≤ **1.15** (φ 0.0049). The draft's 2.7× / 1.6× is exp(φ · 100), with Σ over every depth 9..16. That is the bound for eight +1 rounds: valid, loose, mislabelled. S1-C (four +1 rounds, n 9..12) gives ≤ 1.52 / 1.23.
- **Base-labelled fallback [PROVED UNDER A1]:** ē_Q ≤ ē_0 + κ Σ_q w_q^base. The accumulation is additive, not compounding.
- **R2 noise is 1 − π^E ≥ 1 − π.** Compensated chains (a wrong step, the right final value) give wrong R2 targets. They are 3.6 % of single-wrong-step CV chains in fixture (b). Not material.

### (c) Bias class mass

P(c | kept) ≤ P(wrong | kept) / P(wrong | kept, c) ≤ 2(1 − π) **[PROVED]**. At the measured pooled 6 / 18,782 = 3.2e-4, any majority-wrong class has mass ≤ 6.4e-4 (the draft says ≤ 2e-3 ✓). The fresh n 15 cluster (3 of the 6) is the one place to look. LABEL-AUDIT should print the kept-wrong items' handle patterns.

## 4. Prop S3(a)

One SGD step on the batch-mean loss changes item j's loss by −η ∇L_j · ∇L̄ + O(η²). That is ≤ 0 only if the inner product is ≥ 0.
- It holds for a per-example step (−η‖∇L_j‖²).
- It is not guaranteed in a 64-item, ~40-slot batch.
- The target equals the current argmax only at FT step 0.
- "Pass 1 = the labeller's fresh step" holds exactly only at equal dtype and padding: training is bf16 with padded batches, the labeller fp32 and length-grouped.

**[PROVED UNDER a per-example step at the labeller's weights]**; otherwise [ASSUMED]. No reading depends on it.

## 5. Prop S4: SHIFT prediction and STOP-LOC

**Numbers [COMPUTED, given H_shift].**
- From eval_vb: H_shift curve n 13–16 = 0.923 / 0.603 / 0.266 / 0.127, mean **0.47975**.
- acc(K′ 8) at base n 9–12 = 0.329 / 0.133 / 0.127 / 0.118, so THINK-GAIN-BEYOND = **0.303** ✓.
- The sampling SE of the 4-n mean is **0.0062** (the draft's 0.008 is conservative).
- 0.63 lies 24 SE and 0.80 lies 52 SE above the H_shift mean. So P(MARGIN-GROWS | H_shift) = 0.07 and P(DBS | H_shift) ≤ 0.01 are pure curve-uncertainty priors [ASSUMED]; by sampling alone they are ≈ 0.

**STOP-LOC [FALSE under the §5 definition].** The definition: G(k) = mean over readable n ∈ [k+1, min(k+4, 16)] of acc(K′ n+2) − acc(K′ k); readable means acc ≥ 0.18 at ≥ 2 n; STOP = the smallest k ∈ 10..14 with G ≤ 0.02.
- **Base D3-T:** k 10 has one readable n (n 11, 0.266; n 12–14 ≤ 0.141), and k 11–14 have none. So STOP is **unreadable (never finishing)**, not the "10" in the table.
- **Exact H_shift** (base acc_k translated by +4, ABSORB-level n ≤ 12): G(10..13) = 0.777 / 0.585 / 0.401 / **0.081** > 0.02, and k 14 has one readable n (n 15). So STOP is **unreadable**, STOP-TO-SUPPORT fails, and the **SHIFT label (¬MARGIN ∧ STOP-TO-SUPPORT ∧ mean ≥ 0.40) cannot be assigned under the hypothesis it names**. With mean 0.48 it is not SHRINK either: the partition has a hole exactly at H_shift. The same applies to §9 risk 1's falsifier.
- **Fix (either):**
  - readable ≥ 1 n per k: base reads 10, H_shift reads 14 = B + 2;
  - or the unfiltered 213-26-style G: base 10; H_shift G(13) 0.061, G(14) −0.002, so 14.
- Add an explicit UNLABELLED class, counted as not-GROW, plus a fixture panel that is H_shift exactly (G-S12).

## 6. GATE-q: operating characteristic and state precision

**Exact joint OC [PROVED; gate_oc_exact.py].** Both value events are decreasing in (K_n), so by Harris–FKG, gate_oc2's product is a lower bound and min(P_pool, P_all_n) an upper bound. The exact value comes by convolution. Thresholds are integer-exact: c_pool = ⌊0.005 N_pool⌋, c_n = ⌊0.01 N_n⌋; the float floors agree.

| window (N_n) | c_pool / c_n | π 0.999 | **0.997** | 0.995 | **0.993** | 0.99 |
|---|---|---|---|---|---|---|
| R1 2000 / 1992 / 1976 / 1952 | 39 / 20, 19, 19, 19 | 1.0000 | **0.9986** | 0.504 | **0.0125** | 0.0000 |
| R2 1944 / 1904 / 1886 / 1834 | 37 / 19, 19, 18, 18 | 1.0000 | 0.9980 | 0.489 | 0.0129 | 0.0000 |
| R3 1784 / 1678 / 1618 / 1562 | 33 / 17, 16, 16, 15 | 1.0000 | 0.9974 | 0.531 | 0.0234 | 0.0000 |

- The draft's 0.998 / ≈ 0.5 / ≈ 0.01 hold (R3 is ≈ 0.02 at 0.993). The OC is conditional on the kept counts and on equal π across n.
- **It covers only the two value terms.** The π^E and keep terms add constraints, ≈ 1 at D3-T's measured values *if* CV ≈ KEEP, which is unmeasured (§7).
- With the window counts of §3a, P(GATE-1 pass) and P(base-labelled GATE-2 pass) ≈ 1.

**State precision, well-defined form.**

π^E_W = #{i ∈ gate_W : KEEP_i ∧ CV_i ∧ E_self,i(t) = recode_targets(root_i)(t) ∀ t ∈ 0..n_i} / #{i ∈ gate_W : KEEP_i ∧ CV_i}, item-weighted over the window.

- E_self exists iff CV. Both lists have n + 1 entries on the same original frame. t > n is HOLD on both sides, so "every t" must mean 0..n; K is a training-time draw.
- **π^E ≤ π on the same denominator [PROVED]:** E_self = gold at every t ⇒ E_n's single live slot is the gold value. π − π^E is the compensated share.
- The gate's keep_n must be the **KEEP ∧ CV** rate, the pool's actual filter, not 213-38's KEEP rate.

## 7. R2 composition, COMPOSE-VALID and chain-exactness

**Theorem [PROVED, induction].** Let L_k be the live (non-VOID, non-frame) slots of E_k in order.

Invariant: the tokens of E_k on L_k are render(cur_k).

Step:
- CV at step k + 1 means dec_{k+1} is x(cur_k) with the owned slots H of the leftmost-ready node h of cur_k VOIDed, except H's last slot, which holds a value v (SPAN ∧ ORDER).
- E_{k+1} overwrites L_k by dec's interior. The surviving live slots carry rerender(dec_{k+1}) = render(cur_{k+1}) (CANON, with cur_{k+1} = parse(rerender)).
- recode_targets applies the same operation to the same slots: _leftmost_ready depends only on the tree shape, which equals cur_k's whatever the values. It uses the true value v*.

Hence E_self = gold at t < k*, the first wrong step, and differs from k* on. **E_self = gold at every t ⇔ every step value is correct.** ∎

- The 100k gate (850,429 steps, 0 failures) is consistent.
- So is fixture (b): 3,000 CV chains with one wrong step, 0 equal to gold, 107 (3.6 %) compensated.

**Counterexample to the draft's premise [COMPUTED, compose_memT_checks (a)].** A decoder that puts the value on the span's *first* slot is chain-exact, in 213-33's string-level sense (rerender(dec) = render(true reduction)), at every step on **1,000 / 1,000** items. It passes KEEP on 1,000 and COMPOSE-VALID on **0**. Hence:
- "On chain-exact chains E_self = recode_targets(root)" is **[FALSE as worded]**. Replace "chain-exact" with "COMPOSE-VALID with every step value correct".
- "Predicted extra drop ≈ 0, because every value-correct chain is chain-exact [MEASURED]" is a **non sequitur**. chain_exact and STEP-GENERAL (vbank_diag.long_step, l.107–127) compare *rerendered strings*, which drop VOIDs and the frame. SPAN, FRAME and value-slot placement of fresh decodes at m ≥ 9 have never been measured. Retag as **[CONJECTURE]**. S0-1(a) decides it, and GATE-1's keep ≥ 0.70 on KEEP ∧ CV is the safety net.
- "Every value-correct chain is chain-exact" is **[FALSE as worded]**. right_value_wrong_steps chains are value-correct and not chain-exact (3 at D3-T test n 15; ≤ 0.003 per n elsewhere). The measured fact is **strict ⇔ chain-exact** (chain_exact == precision_strict at every n in both cells; chain-exact ⊂ strict by definition).

## 8. Item space, exclusion, test-set extension

- **Counts.** item_space.json reproduces the §1 rows, but **n 1–4 used 5,000 draws** (NPER applies only for n ≥ 5). The draft's "0 hits / 0 dups in 50k at every n 4–20" should read "5k at n 4, 50k at n 5–20".
- **Analytic bound [PROVED].** render is injective (minimal brackets with precedence and associativity). Each generator split picks op 1/4 and a pair 1/|PAIRS| ≤ 1/10 (min |PAIRS| = 10). So P(render = s) ≤ (1/11) · 40⁻ⁿ.
  - Expected within-pool duplicate pairs among D draws ≤ D²/2 · (1/11) · 40⁻ⁿ: **≤ 3.2e-6 at n 9 with D = 135k**, ≤ 5e-11 at n 12.
  - Exclusion hits ≤ D · |EXCL_n| · (1/11) · 40⁻ⁿ ≈ 1e-7.
  - So zero collisions and fresh-once *within the generator* hold at n ≥ 9 at any planned pool size.
- **test_set(1000, n_max 20) ≡ test_set(1000) on n ≤ 16 [PROVED].** One rng is consumed stratum by stratum, n = 1..n_max (nested170.py l.148). Also measured True.
- **Training T** rows ✓. T 47 at n 12 is a 50k-sample maximum; the attainable maximum is 49 (11 bracket pairs).

## 9. Per-n budgets, R1 window, trainer spec gaps (vs vbank.py)

**[VERIFIED]:**
- K = min(n + U{0..4}, kmax) per item;
- R1 window [n, K] and R2 at 1..K exactly as vb_loss;
- self kmax 16 = B + 4 never binds (K = n + U);
- lr_at: 100-step warm-up, cosine over --steps;
- --c_sched 0,1 gives c = 0 from step 1.

**Gaps to add to §2.4:**
1. vb_loss reads `it["t"]` (R1 target, l.482) and rec_targets reads `it["root"]`. G-S0's loader rejects both keys, so the trainer edit must also route the **record's label to the R1 target** (and n from syms). The draft lists only the R2 change.
2. Self K_i must be drawn from **rng_s** or a new stream, never rng_k. Otherwise CTRL's plain-stream identity (G-S3 hash check) breaks.
3. vbank's mem_check peak comes from the first step, before AdamW state exists. G-S9 needs a second step, or +~0.93 GiB (D4 v1 adds 1.0). Also, the draft's 21.5 GiB budget ≠ vbank's default --mem_budget 21.0.

## 10. Fresh-once pool budget [FALSE: counterexample]

With the existing CurGate class counting global steps (12k steps, coin ½, bs 64), a stage A of s_A steps (n ∈ {9, 10}, cap 4k) draws **96,000 + 8 s_A** items at each of n 9 and n 10.
- 120,000 kept per n suffices only if s_A ≤ 3,000.
- At the 4k cap the need is **128,000** (99.9 % quantile 131,800), so fresh-once is violated by about 8–12k items at n 9–10.
- n 11–12 need ≤ 80,000.
- If the gate counted self steps, the need would be 160,000.

**Fix:** kept pools of ≈ 135k at n 9–10 and 85k at n 11–12 (440k total, about 0.9 h of labelling instead of 1.0 h). Alternatively, cap stage A at 3k steps.

## 11. Cost and memory (cost_check.py; the model forms stay [ASSUMED], with corrected inputs)

**Two input errors.**
- c = 0.0258 was **fitted to the 11.28 GiB mem_check** (PROOFS_D4 l.231). That batch (the 64 longest of 512 draws, rng 0) has padded **T 31**, not 33.
- AdamW state (~0.93 GiB) is absent from that peak.

**Simulated typical padded T:** plain 29; self R1 stage B 42, stage A 36; R2 54.

| shape | draft GiB / micro | corrected GiB (forced) / micro | $ (draft → corrected) |
|---|---|---|---|
| R1 self K 16, T 47 | 23.2 / 2 | 26.0 / 2 | 0.52–0.96 → **0.52–1.00** |
| R1 self kextra 2, K 14 | 19.8 / **1** | 22.3 / **2** | 0.34–0.58 → **0.48–0.91** (the fallback no longer saves money) |
| R2 self K 20, T 61 | 43.7 / 4 | 48.7 / 4 | 1.13–2.64 → 1.13–2.78 |
| R2 self kextra 2, K 18 | 38.0 / **2** | 42.4 / **4** | 0.67–1.43 → **1.04–2.55** (above the $2.0 round-2 cap at the high end) |

- **Pair S1-A** projects to $0.82–1.43, so the $1.5 cap holds.
- At G-TIME's per-pass rate (0.462 s per 12 passes vs 0.407 per 11.84; E[Kb] 11.84 reproduced from the trainer's law), the high end is **$1.59 > $1.5**. G-S10 decides, as the draft already says.
- Labelling time reproduces: round 1 3,615 s ≈ 1.0 h; round 2 6,424 s ≈ 1.8 h.

## 12. §6 OC arithmetic [COMPUTED from the stated priors; the priors are ASSUMED]

| quantity | value |
|---|---|
| kept factor | 0.9 · 0.9 · 0.85 = 0.6885 |
| P(SD-SELF) | 0.65 · 0.6885 = **0.448** |
| P(MARGIN-GROWS) | 0.65 · 0.3935 = **0.256** |
| P(DBS) | 0.111 |
| P(SD-C2) | **0.0763** |
| P(SD-C2-MEM) | 0.053 |
| P(STOP-MOVED) | 0.726 |
| P(SD-C2 \| H_shift) | **0.0069** (≤ 0.01 ✓) |
| P(SD-C2 \| H_grow) | 0.172 |
| P(SD-C2 \| H_full) | **0.620** |
| P(SD-C2-MEM \| H_full) | 0.434 |

**Implicit assumptions to state:**
- P(DBS | ¬ABSORB) = P(MARGIN | ¬ABSORB) = 0.
- The kept readings are independent of H. They are more likely positively correlated, which makes the conditionals slightly conservative.
- The OC-block rule also wants **P(SD-SELF | no effect) ≈ 0**: ABSORB needs ≥ 0.95 at n 12, and the base is 0.127.
- Until §5's STOP-LOC fix, the SHIFT mechanism label has P ≈ 0 under H_shift.

## 13. Fix list for the draft (by weight)

1. **STOP-LOC readability** (§5): the base and exact H_shift both read unreadable, so SHIFT is unlabellable. Use readable ≥ 1 n or the unfiltered G, add UNLABELLED and an H_shift fixture.
2. **Pool sizing** (§10): ≈ 135k kept at n 9–10 and 85k at n 11–12, or a stage-A cap of 3k. Make G-S3 per-stage, with target 0.494.
3. **Memory / cost** (§11): use T_ref 31 and +0.93 GiB AdamW. The fallback becomes micro 2 at $0.48–0.91. Round-2 kextra 2 becomes micro 4 at $1.04–2.55. G-S9 must measure after step 2.
4. **COMPOSE-VALID premise** (§7): retag "extra drop ≈ 0" as [CONJECTURE]. Replace "chain-exact" with "COMPOSE-VALID ∧ all values correct" in the composition claim. Write "strict ⇔ chain-exact".
5. **S2** (§3): w among kept (1/P); A1′ for untrained m ∈ W_{q+1}; κ thresholds 8.1 / 5.7 / **3.9**; two-round factor ≤ 1.32×; cite direct window π (R1 ≥ 0.9994 at 95 %); min per-n 0.9968; D3-F 0 / 541.
6. **Trainer spec** (§9): R1 label source; rng_s for self K; G-S9 timing; mem budget 21.0 vs 21.5.
7. **Small items:** item-space n 4 = 5k draws; S3(a) retag; H_shift SE 0.006; GATE OC covers the value terms only; π^E denominator and t-range as in §6.
