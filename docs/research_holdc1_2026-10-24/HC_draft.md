# HOLD-C1 v0 (HC): do serial causal use (C1) and the §230 long-horizon hold (C6) coexist in one model? (2026-10-24, draft, NOT registered)

**Status:** draft v0 for Theory. Not registered, not red-teamed, no Literature check yet. Proposed § number: 232. Origin: the user's 10-24 decision "hold fix into C1 first" (CURRENT_TASKS, Theory 10-24); §230 pass scope (Note 230-1 (3), 230-2): "hold without depth cost: 1 confirmed (s1) + 1 motivating (s2) seed, 12k post-formation anchored far fine-tune at λ 1.6; may be carried into C1 work on the synthetic line".

**$0 checks run for this draft (scripts in this folder, CPU, ≤ 2 threads, idle priority; committed files only, no model loaded):**
- `hc_ckpt_check.py` → `hc_ckpt_check.out.json`: every candidate model.pt is on disk; args diffs as expected (§2).
- `hc_c1_noise_cpu.py` → `hc_c1_noise_cpu.out.json`: FT / seed / card noise of every C1 statistic; the margins in §4.3 and the OC in §6.

**Disclosure (what was seen before this draft).** For the FT cells (s230_fhftfix_s1, s224_fhft_s2, s224_ftctrl_s1_re, s224_ftctrl_s2, s224_rdft_s1 / s2), the 3090 files hold acc / stall / carry_likeness / bank_ablate / stepcap / khorizon. I read those, and they shaped §3 F5 and the arm (v) in §4.5. **No PROBE(t) and no VALUE-SWAP exist for any FT cell**, so none of the decisive C1 statistics has been seen. HOLD / DEPTH / NOLOSS on s1 are §230's own readings (230-2), already seen; they are not re-counted as new evidence (§4.6).

## 1. Which models carry the best C1 record, and why the VB family

| family | registered C1 battery | met | checkpoints on disk | §230 FT applies |
|---|---|---|---|---|
| §184 INJECT (nested170 + DeltaMem-R) | §170 conjunction: DEEP + THINK-NECESSARY + SERIAL-NEST (donor transplant) | 2 / 2 seeds (SERIAL-NEST 0.849 / 0.943) | s184_reweight_s2 only (s1 model.pt deleted by retention) | no: different module (deltamem); the far branch, gates and G-D would have to be ported |
| VERSION-BANK D3-T (§213 / §215 / §216) | **VB-C1** = DEEP ∧ PROBE-GAIN ∧ SERIAL-ONSET ∧ SERIAL-STEP (VALUE-SWAP) ∧ NO-CLOCK (STALL) (VERSIONBANK draft §7, score_s211) | every D3-T-lineage model (s213_d3t_s1, s216_d3t_s2, s215_ftctrl_s1, s215_ftself_s1, s216_r1_s2, s216_r2_s1) | bases s1 + s2, FT-CTRL s1 + s2, RD-FT s1 + s2, FH-FT s2, FH-FT-FIX s1: **all present** | yes (boxes/vbank.py --far --fh --fh_lambda_fixed, G-D'd in §230) |

- **Choice: the D3-T lineage, seeds 1 and 2.** VB-C1 is the stricter battery: five parts, two of them causal-use tests (VALUE-SWAP, STALL), plus a frozen-model probe. INJECT's conjunction has a single causal test. The FT needs no port. Both seeds have every checkpoint the design needs. [MEASURED: `hc_ckpt_check.out.json`, ok.all_model_pt True]
- INJECT is not used. Porting the far branch to deltamem would cost a code change, gradflow and bitwise gates, a G-D, and a new base for s1 (its model.pt is gone). That buys a weaker C1 battery. It is the pre-named follow-up only if HC passes and a second architecture is wanted.

## 2. Models and roles (per lineage L ∈ {s1, s2})

| role | s1 | s2 | what it isolates |
|---|---|---|---|
| BASE (pre-FT) | s213_d3t_s1 | s216_d3t_s2 | the original C1 model |
| FT-CTRL (the same 12k FT, lr 3e-5, no far branch) | s215_ftctrl_s1 (= s224_ftctrl_s1_re) | s224_ftctrl_s2 | the fine-tune alone |
| RD-FT (FT + far CE, no anchor), descriptive | s224_rdft_s1 | s224_rdft_s2 | far CE vs anchor |
| **HC** (FT + far CE + anchor, λ fixed 1.6 = the §230 recipe) | **s230_fhftfix_s1 (exists)** | **s232_fhftfix_s2 (new, Stage B)** | the recipe |
| HC-near, descriptive | – | s224_fhft_s2 (dose rule, λ 0 to step 500 then 1.618) | FT-replicate noise vs s232 |

- Args diffs [MEASURED, hc_ckpt_check]: FT-CTRL vs HC = far / fh flags only; RD-FT vs HC = {fh, fh_lambda_fixed}; **planned s232 vs s230 = {seed, init_from}**, and planned s232 vs FT-CTRL s2 = {far, fh, fh_lambda_fixed}.
- **s1 needs no new training:** s230_fhftfix_s1 is already "D3-T s1 + the §230 FT". Only its C1 parts are unread.
- **s2 needs one cell:** the motivating s2 model (s224_fhft_s2) differs from the recipe by λ schedule. s232 is the exact recipe. It also gives the §230 hold a second run on the s2 lineage, labelled "exact-recipe re-run on the motivating lineage", not an independent seed.

## 3. Derivation: why the anchored post-formation FT should or should not disturb serial use

Notation: core map F_θ(V_{t−1}, B_{<t}) = V_t with SLVA bank B = versions V_0..V_{t−1}. Computing regime C: t ≤ n (an operator is alive). Idle regime I: t > n + 2.

- **F1 [PROVEN, code: vbank.rollout grad_from, far_branch; §224 K1].** L_far + λ·L_FP send no gradient into any stored state V_s, s < T_far (the passes are no-grad, bank K / V detached). The gradient reaches θ only through pass T_far's core and its coda. **The anchor never targets, and never back-propagates into, a mid-chain state V_t (t ≤ n).**
- **F2 [PROVEN, index ranges].** Every C1 reading evaluates passes ≤ n + 2 ≤ 10:
  - VALUE-SWAP at K′ n + 2;
  - STALL at K′ n + 1;
  - ACC / DEEP at n + 2;
  - PROBE(t) at t ≤ 12, through a freshly trained coda.

  Far gradients act only at T_far ≥ 13. So a C1 statistic can change only through shared-weight transfer.
- **F3 [DERIVED, linearised; ASSUMED small-step regime: lr 3e-5, 12k steps].** After the FT, the change at a computing input x_c is ΔF(x_c) ≈ J_θF(x_c)·(Δθ_ctrl + Δθ_far).
  - Δθ_ctrl is the plain-FT part. It is present in FT-CTRL, and FT-CTRL moved no decisive C1 statistic by more than 0.0025 [MEASURED: s213_d3t_s1 → s215_ftctrl_s1, `hc_c1_noise_cpu.out.json`].
  - The far part is ΔF_far(x_c) ≈ −η Σ_steps K_θ(x_c, x_I)·∂L_far+FP/∂F(x_I): an NTK cross-term between computing and idle inputs.
  - **HC − FT-CTRL isolates exactly this term**, plus the noise of two FT replicates. HC − RD-FT splits off the anchor's share.
- **F4 [MEASURED, FH_v1 §0 oracle].** V_{n+2} is a high-margin fixed point: freezing the stored state holds 1.00 to K′ 128, and bank_freeze holds 1.00. So the anchor asks for a property the done-state branch nearly has already (step-0 L_FP 0.039, 230-2). That argues for a small cross-term.
  - Against that: λ·L_FP dominated the loss late (median 5.9×, 29.3× the rest in the last window, 230-2), so the far gradient was the FT's main signal.
- **F5 [MEASURED, committed 3090 files, descriptive]. The cross-term is not zero.**
  - Newest-version mass on computing passes 2..n, n 3–8 (`carry_likeness`): FT-CTRL 0.825 / 0.833 → **HC 0.602 (s1) / HC-near 0.694 (s2)**. RD-FT stays at 0.813 / 0.801.
  - In-range bank dependence (`bank_ablate.acc_n2`, versions 1..t−2 masked), n 8: FT-CTRL 1.000 / 0.999 → HC 0.985 (s1) / HC-near **0.641** (s2).
  - So **the anchor, not the far CE, moved the computing passes' version reads off the newest version**, and in s2 the n 7–8 computation now depends on older versions.
  - This fits a "restore from stored versions" hold (224-5's restoring reading; the bank_freeze oracle), carried into the computing passes by the shared, content-keyed SLVA queries.
  - Consequence: **the C1 battery is exposed.** The computing passes of HC differ from FT-CTRL's. A pass will not be "kept because nothing changed" (memory: controls must be exposed).
- **F6 [DERIVED]. Why a version-mixing read need not hurt serial use on natural rollouts.**
  - On a natural rollout the history is consistent: V_t = F(V_{t−1}). At an unreduced slot, V_{t−1}[j] ≈ V_t[j]; only the just-reduced slot b differs.
  - A content-keyed read that mixes versions where they agree, and takes the newest where they differ, computes the same next state.
  - acc_n2 n 1–8 ≥ 0.999 in HC / HC-near [MEASURED] shows the mixing is harmless on natural rollouts.
- **F7 [DERIVED]. Where it can bite: spliced (counterfactual) rollouts.**
  - VALUE-SWAP (ii) replaces V_t with the donor's but keeps the recipient's V_{<t}, so the history is inconsistent: V_{t−1} still holds the recipient's operands of the handle.
  - A version-mixing reader can partly restore the recipient's value: own ↑, follow ↓. That is strongest in arm (i), the slot-only splice, and possible in (ii).
  - This is a property of the intervention under history reads, not necessarily a loss of serial use. Precedent: §211 N2, the one HISTORY-USED VB model, read follow_i 0.658 / own_i 0.307 but **follow_ii 1.000** [MEASURED].
  - The disambiguating arm is (v), a *consistent* donor history V_1..V_t (§4.5).
- **F8 [DERIVED]. What a real Goodhart would look like.** FH_v1 G1 is the anchor collapsing computation by jumping to the done state early through shared weights. Its signatures:
  - PROBE(2) ↑ (precompute);
  - onset slope ↓;
  - VALUE-SWAP own_iv ↑ (recompute from V_0);
  - STALL ↓ if passes became count-keyed.

  PROBE(t) reads V_t plus the bank, so information redistributed across versions stays visible to it. **Prediction: PROBE / onset / STALL unchanged; follow_i down; follow_ii stays ≥ 0.98 in s1; s2 uncertain** [ASSUMED priors, §6]. Evidence against G1 so far: DEPTH parity (230-2), acc intact, and no ahead-of-schedule loss in FH-FT s2 (REVIEW_HD, p = 1.0).
- **F9 [DERIVED from F1–F3]. Answer to "could the anchor damage mid-chain states?"** Not directly (F1). Only through the shared-weight cross-term (F3), which is visible as F5. The design therefore tests whether F5 changes the serial causal structure (PROBE / VALUE-SWAP / STALL), with FT-CTRL and RD-FT separating the fine-tune, the far CE and the anchor.
- **F10 [DERIVED]. The joint risk C1 × C6.** The anchor was trained on natural rollouts only. On a spliced rollout, a restoring dynamic may pull, over 120 idle passes, toward a state consistent with the whole bank: the recipient's V_0..V_{t−1}, i.e. the recipient's answer.
  - Then the held answer would not be the latent-computed one. SWAP-HOLD (§4.5) reads this directly.
  - FT-CTRL drifts to chance by K′ 128 (H 0.117 / 0.096), so it is the baseline.

## 4. Readings (all registered unless marked descriptive)

### 4.1 Measurement
- **C1 battery:** `boxes.vbank_eval --parts acc,probe,vswap,stall`, fp32, eval_bs 32, §170 test set, on **one card class for every model: the local RTX 4060** ($0).
  - Each model is evaluated in a fresh staged dir boxes/runs_fsq/hc_<run>_4060/ (copies of model.pt / args.json / calibration.json), **never in the committed run dir** (vbank_eval merges into <run>/eval_vb.json and would overwrite 3070 / 3090 data).
  - The model.pt copies are deleted after Audit reproduces the readings (retention).
- **Fallback (pre-stated):** a 4060 OOM or a failed G-BASE (§5) moves **all** models to one cloud 3090. Classes are never mixed within a reading.
- **HOLD / DEPTH / NOLOSS:** the 3090 files as §230 (khorizon to K′ 128, acc, stall).
  - s1: s230_fhftfix_s1 vs s224_ftctrl_s1_re (exist).
  - s2: s232 vs s224_ftctrl_s2, on the Stage-B host.

### 4.2 Primary, per lineage: COEXIST(L) = HOLD ∧ VB-C1 ∧ C1-KEPT ∧ DEPTH ∧ NOLOSS
- **HOLD:** H(K′) = mean over n 1–8 of eval_khorizon `acc.intact[K′][n]` ≥ 0.95 at every K′ ∈ {24, 32, 48, 64, 96, 128}.
- **VB-C1:** the registered conjunction, score_s211.cell_read logic on the 4060 eval_vb of the HC model:
  - DEEP: acc_n2 ≥ 0.90 at n 3–8;
  - PROBE-GAIN ≥ 0.50, with PROBE-VALID;
  - SERIAL-ONSET: coverage ≥ 0.8, slope_h ≥ 0.8, gap_8_3 ≥ 3;
  - SERIAL-STEP: n_pairs ≥ 100, follow_ii ≥ 0.6, own_ii ≤ 0.3;
  - NO-CLOCK: stall.recovery ≥ 0.9.

  The log-based censoring of score_s211 (48k, NO-ENTRY) is replaced by §6 censoring.
- **C1-KEPT:** for every decisive statistic S in §4.3, d = S(HC) − S(ref) in the harm direction must stay within δ_S, **for ref = BASE and for ref = FT-CTRL** (same lineage, 4060).
- **DEPTH:** mean acc_n2 n 9–11 (3090) ≥ FT-CTRL − 0.03.
- **NOLOSS** (3090):
  - acc_n2 n 1–8 ≥ 0.99;
  - stall.recovery ≥ FT-CTRL − 0.02;
  - C1 floor: acc_k["0"] n 1–16 and acc_k["1"] n 2–16 ≤ FT-CTRL + 0.05 per n.
- **COEXIST** (the claim) = COEXIST(s1) ∧ COEXIST(s2).
  - Wording on a pass: "on the synthetic VERSION-BANK nested-arithmetic recipe, the §230 hold fix (12k post-formation anchored far FT, λ 1.6) keeps every registered VB-C1 statistic within measured noise of the pre-FT model and of a matched FT, while holding the answer to K′ 128, in 2 lineages".
  - Not C2, not a rate claim.

### 4.3 C1-KEPT statistics and margins (from `hc_c1_noise_cpu.out.json`)
- **Margin rule:** δ_S = max(floor_S, 2 × max |d_S| over the four same-lineage FT pairs, 3·√2·SE_S).
  - The FT pairs (3070) are: the matched plain FT s213 → s215_ftctrl; and three self-item FTs (s213 → s215_ftself, s215_ftself → s216_r2, s216_d3t_s2 → s216_r1). The self-item FTs are larger perturbations, so they bound a matched FT from above.
  - SE: binomial over the item counts (probe 6,000; vswap ≈ 4,184 clean pairs; stall 5,000; DEEP 1,000 per n), and for slope_h a 300-draw item bootstrap.

| S | harm | max FT \|d\| | max SE | δ_S | HC passes if |
|---|---|---|---|---|---|
| PROBE-GAIN | ↓ | 0.0068 | 0.0044 | **0.030** (floor) | ≥ ref − 0.030 |
| SERIAL-ONSET slope_h | ↓ | 0.000 (1.2857 in all 8 models) | 0.086 (bootstrap) | **0.364** | ≥ ref − 0.364 |
| VALUE-SWAP follow_ii | ↓ | 0.000 (1.000 in all) | 0.0002 | **0.020** (floor) | ≥ ref − 0.020 |
| VALUE-SWAP own_ii | ↑ | 0.000 (0.000 in all) | 0.0002 | **0.020** (floor) | ≤ ref + 0.020 |
| STALL recovery | ↓ | 0.0002 | 0.0004 | **0.020** (floor) | ≥ ref − 0.020 |
| DEEP min acc_n2 n 3–8 | ↓ | 0.000 | 0.0007 | **0.020** (floor) | ≥ ref − 0.020 |
| PROBE(2) pooled n 3–8 (precompute) | ↑ | 0.0025 | 0.0057 | **0.030** (floor) | ≤ ref + 0.030 |

- **Reading of the noise [MEASURED]:** a 12k FT, or even a self-item FT, moves no decisive C1 statistic by more than 0.007, and the 3070 vs 3090 card effect is ≤ 0.0012. The decisive statistics are saturated or nearly so. **Floors bind every δ except slope_h.**
  - The seed difference is larger only for the unsaturated statistics: PROBE(2) 0.270 vs 0.230; follow_i 0.996 vs 0.857.
- **Descriptive, not decisive** (F7: their movement is predicted and ambiguous under history reads): follow_i, own_i, own_iv, PROBE(3 / 4 / 6), coverage, gap_8_3, carry_likeness, bank_ablate.
  - Their FT-pair spread is reported beside them. follow_i's largest FT move is 0.043 (s2 self-FT), and its seed difference is 0.139.

### 4.4 Attribution (per lineage, per decisive statistic; labels, not verdicts)
- **FT-CHANGED-C1:** FT-CTRL − BASE beyond δ (the fine-tune, not the far branch).
- **FAR-CHANGED-C1:** HC − FT-CTRL beyond δ. Then split it, descriptively, by HC − RD-FT (anchor) vs RD-FT − FT-CTRL (far CE).
- C1-KEPT needs both references, so FT-CHANGED-C1 also fails COEXIST(L). It is reported as such, and the far loss is not blamed.

### 4.5 Secondary registered reading and one diagnostic arm (new eval part, §5 K2)
- **SWAP-HOLD(L)** (C1 × C6 jointly; HC only, FT-CTRL / BASE descriptive):
  - The VALUE-SWAP class-A pairs (rng 21120, as vbank_eval), arm (ii), read at K′ n + 2 **and at K′ 128** from one rollout with coda rows at both.
  - Pass: follow_ii(K′ 128) ≥ 0.95 ∧ agree ≥ 0.95, where agree = the per-pair share, over clean pairs, of answer(K′ 128) == answer(K′ n + 2).
  - Meaning: the held answer is the latent-computed (spliced) result, not one re-derived from V_0 or the recipient's older versions over 120 idle passes (F10).
- **Arm (v), diagnostic:** splice the donor's V_1..V_t (consistent history; V_0 stays the recipient's), read at K′ n + 2. It is used only to label a C1-KEPT failure on follow_ii / own_ii:
  - **SWAP-REPAIR:** follow_v ≥ 0.90 ∧ own_v ≤ 0.05. The model rejects inconsistent splices through its history, and serial use is intact under a consistent splice. It reads **COEXIST-SHIFTED**: reported, not COEXIST.
  - **C1-LOST-STEP:** otherwise.
- Not part of COEXIST. Prediction: SWAP-HOLD passes in HC with P ≈ 0.7 given COEXIST.

### 4.6 Seeds, stages, what is new evidence
- **Stage A ($0, local 4060):** the C1 battery on 8 existing models (BASE / FT-CTRL / RD-FT × s1 / s2, HC s1, HC-near s2), plus SWAP-HOLD / arm (v) on HC s1, HC-near s2 and FT-CTRL s1 / s2.
  - Order: s1 trio first (BASE, FT-CTRL, HC; ≈ 4–5 h), then the rest.
  - **Read COEXIST(s1).** Its new evidence is VB-C1 + C1-KEPT. HOLD / DEPTH / NOLOSS are §230's readings (seen).
- **Stage B (≈ $1.0, one cloud 3090):** s232_fhftfix_s2, run **only if COEXIST(s1) passes.** If s1 fails, the coexistence claim cannot pass, and s2 spend cannot rescue it (no problem → fix loop).
  - Host evals as §230: khorizon to K′ 128, acc, stall. Then the C1 battery + SWAP-HOLD on the 4060 → COEXIST(s2), all new.
- **Seed split (s1 pass, s2 fail):** report SPLIT, with HC-near s2 shown beside s232 (FT-replicate spread).
  - **Contingent s3** (new D3-T s3 base 48k + FT-CTRL s3 + HC s3, ≈ $3.0–3.4, cap $4) only by Audit's ruling at that time, disclosed as decided after s1 / s2.
  - No other s3 path. No D3-T s3 base exists today.

## 5. $0 checks and gates before any GPU reading or spend
- **K0 (done):** hc_c1_noise_cpu.py: margins and OC (§4.3, §6).
- **K1 (done):** hc_ckpt_check.py: all 8 model.pt present (502,609,294 B each); args diffs as §2; planned s232 diff = {seed, init_from}. Optional `--sha` hashes to pin before staging.
- **K2, build (Experiments; Audit G-D):** new part `c1hold` in boxes/vbank_diag.py → <dir>/eval_c1hold.json, with the unique top-level key "c1hold" (vbank_diag main() key-collision lesson).
  - It reuses vbank_eval.make_pairs (class A, rng 21120) and the khorizon fits() / eval_bs fallback with ks kept.
  - Arms: (ii) at K′ n + 2 and 128; (v) at K′ n + 2.
  - **Identity gate:** (ii) at K′ n + 2 equals vbank_eval part_vswap's per-pair answers on the same card and bs (bitwise share 1.0) on one base model.
  - Plain-path smoke; mem_check at n 8, K′ 128, bs 32 on the 4060.
- **K3, G-BASE (card):** on the 4060, BASE s1 / s2 must re-read VB-C1, and every decisive statistic must be within 0.01 of its committed 3070 value. Otherwise use the 3090 fallback for all models.
- **K4, scorer:** theory/score_s232.py (Theory), with fixtures for each branch (COEXIST, COEXIST-SHIFTED, FT-CHANGED, FAR-CHANGED, SPLIT, pending, censored). Field paths are grepped in vbank_eval / vbank_diag; every raw value is tested (no smoothing).
- **K5, Stage B only:** --fh_lambda_fixed 1.6 path as §230 (built, G-D'd); G-FH-EQ v2 on the host; ARGS-DIFF = {seed, init_from} vs s230; guards with GUARD_STOP, marks to the final step; drain waiter; off-host ckpt copy as §230 practice; one 3090 class.
- **K6:** stray CUDA contexts on the 4060: check memory / util before the chain, and run waits on 'PART x done' lines, not on key names.

## 6. OC block, censoring, cost

**OC (2 lineages; "effect" = a C1 harm of size h_S on a decisive statistic; numbers from hc_c1_noise_cpu.out.json, null sd = max(max FT |d| / 1.5, √2·SE) [ASSUMED]):**

| quantity | value |
|---|---|
| P(C1-KEPT, all 7 statistics, both refs, both seeds \| no harm) | **0.954** (independence; slope_h contributes 0.964) |
| P(C1-KEPT passes \| harm: PROBE-GAIN −0.10, follow_ii −0.15, own_ii +0.15, STALL −0.10, DEEP −0.05, PROBE(2) +0.15) | **< 0.001** each |
| P(C1-KEPT passes \| onset slope −0.40) | 0.146 (slope_h is coarse: medians on T_SET) |
| P(VB-C1 absolute \| no harm) | ≈ 0.99 per model (bars far from the values: PROBE-GAIN 0.87 vs 0.50; follow_ii 1.000 vs 0.6; stall 1.000 vs 0.9; slope 1.29 vs 0.8; gap 6 vs 3) |
| P(HOLD s2 exact \| §230 effect real) | ≈ 0.85 [ASSUMED: HC-near s2 0.990 at K′ 128, FT-replicate spread unknown] |
| P(HOLD s2 \| no far effect) | < 0.001 (FT-CTRL s2 H(128) 0.096) |
| P(DEPTH s2 \| no harm) | ≈ 0.85 (σ_Δ 0.05, 230's OC; HC-near +0.049) |
| P(NOLOSS s2) | ≈ 0.97 |
| **P(COEXIST, 2 lineages \| no C1 harm, §230 real)** | ≈ 0.977 (s1) × 0.85 × 0.85 × 0.97 × 0.99 × 0.977 ≈ **0.66** |
| **P(COEXIST \| a C1 harm of the stated size in either lineage)** | **≤ 0.15** (≤ 0.001 unless the harm is only an onset-slope shift) |
| P(SWAP-HOLD \| COEXIST) | ≈ 0.7 [ASSUMED] |

- **Theory's prior (F5 lowers it):** P(C1-KEPT s1) 0.65, s2 0.5. The main expected failure is follow_ii / own_ii (→ arm (v) labels it), not PROBE.
  - **P(COEXIST, 2 lineages) ≈ 0.25–0.30** (s1 / s2 C1 outcomes correlated). P(COEXIST(s1)) ≈ 0.62. P(Stage B launches) ≈ 0.62.
- **Censoring (rule b):**
  - a dead Stage-B run (log < 12,000) fails every reading of that lineage;
  - a missing K′ 128 is pending, and FAIL at --final;
  - an OOM falls back to a smaller eval_bs with every K′ kept, recorded;
  - PROBE-INVALID or a probe that does not train fails every PROBE reading;
  - VALUE-SWAP with < 100 clean pairs is UNREADABLE = not met;
  - a never-reached onset is a coverage failure;
  - a crashed eval part is re-run once, same card and bs (a crash is not a reading); a second crash = FAIL;
  - a run stopped by the cost rule reads STOPPED-COST;
  - a failed G-BASE moves all models to the 3090, and never mixes classes;
  - event-only values are reported beside every reading.
- **Cost:**
  - Stage A: $0 (4060, ≈ 14–16 h of card time: ≈ 1.3 h per C1 battery × 9 models + ≈ 0.75 h per c1hold × 5).
  - Stage B: ≈ $1.0 (§230 realised $1.00 for the same cell type), cap **$2.5**.
  - Stage-A 3090 fallback: ≈ $1.9 (≈ 8.5 h × $0.22), cap $2.0.
  - Contingent s3: cap $4.
  - **Family cap $8.5** (≤ $30: auto-approval by Audit under the user's standing rule). Projected central **$1.0**. Realised at registration $0.

## 7. Novelty sentences (for Literature; no "new" / "first" until checked; grep LITERATURE.md + LIT_FIXEDHOLD_NOVELTY_2026-10-21 first)
- S1: "A post-formation, correctness-gated, anchored far-pass fine-tune extends answer hold to 16× the trained pass count while keeping frozen-model probe onset, value-swap and stall readings of serial latent computation within measured noise." (Relative to Viakhirev 2608.18222 terminal fixed-point penalties; post-horizon decay 2608.11233, 2609.33144.)
- S2: "SWAP-HOLD: a counterfactual hold test that splices a mid-chain latent and asks whether the spliced result, not the original, is held for 128 passes." (Check against activation-patching persistence work and Huginn-style long-recurrence stability, 2502.05171.)
- S3 (descriptive): "The hold objective shifts the computing passes' version reads off the newest version (carry_likeness 0.82 → 0.60) with no accuracy cost; a stability objective induced history use that the base recipe lacked (§213 HISTORY-USED 0.000)."

## 8. Red-team questions
1. Saturation: the decisive statistics sit at 1.000 / 0.000. Can C1-KEPT detect anything short of a large harm? Is the F5 exposure enough, or should an unsaturated statistic (follow_i, PROBE(3–6)) be decisive despite F7's ambiguity?
2. Is the floor δ (0.02 / 0.03) justified when measured FT noise is ≤ 0.007? Too loose (harm hidden) or too tight (a one-item flip on DEEP min at 1,000 items is 0.001, so it is safe)?
3. slope_h is quantised (1.2857 in every model). Should onset be read on slope_n, or on a per-item onset mean, as an added decisive statistic?
4. F7 / arm (v): does COEXIST-SHIFTED let a real loss of serial use through? Arm (v) keeps V_0 = recipient; should a (v) + V_0-masked arm be required?
5. s1 confirmatory content is C1 only (HOLD / DEPTH / NOLOSS already seen in §230). Is that adequate for a 2-lineage claim, with s2 carrying all-new readings?
6. Is the exact-recipe s232 needed, or would HC-near s2 do, saving $1? Conversely, should s232 run regardless of s1, as the §230 second seed?
7. Multiplicity: 7 statistics × 2 references × 2 lineages. Is the joint 0.954 under independence optimistic, given that the statistics share items?
8. C1 on the 4060 and HOLD on the 3090: does the cross-class split of readings matter (card effect ≤ 0.0012 measured on probe / stall / acc)?
9. SWAP-HOLD's spliced rollouts are off-distribution for the anchor (trained on natural rollouts). Is a fail informative about C1 × C6, or only about off-distribution holding? Should it stay secondary?
10. Does the F5 shift threaten the line's other readings (DEPTH-GENERAL, LENGTH-FREE, KEEPS-SELF-STAB), which HC does not re-read?
11. Lit: is "anchor-induced history use" (S3) a known effect of fixed-point / consistency regularisers in looped models?

## 9. Files
- This draft: docs/research_holdc1_2026-10-24/HC_draft.md.
- $0 checks: hc_ckpt_check.py (+ .out.json), hc_c1_noise_cpu.py (+ .out.json).
- To build after review: vbank_diag `c1hold` part (K2), theory/score_s232.py (K4), the Stage-B queue / filler (Experiments).
