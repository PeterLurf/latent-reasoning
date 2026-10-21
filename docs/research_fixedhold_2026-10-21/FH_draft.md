# FIXED-HOLD (FH): a gated idle-pass fixed-point lever for the C6 hold failure (draft v0, 2026-10-21 18:37 EDT)

Theory draft, not registered. Addresses Note 216-15 (every VB model loses the held answer past K′ ≈ 32). Lit: LIT_IDLE_HORIZON_PRESCOUT_2026-10-21.md.
- Status: **DRAFT, for red team + Literature.** No § number yet, no spend, nothing launched.
- $0 checks (this folder; CPU, 2 threads, no GPU):
  - `fh_check_json.py` → `fh_check_json.out.json` (A, from the six committed eval_khorizon.json);
  - `fh_check_cpu.py <run>` → `fh_check_cpu.<run>.json` (B, CPU fp32 rollouts of model.pt to K′ 64; 48 items per model, n {1, 4, 8} × 16);
  - `fh_check_summary.py` → `fh_check_summary.out.json` (C, per-model summary of B).
  - B's validity check: the CPU answers equal eval_khorizon items.intact at K′ 24 / 32 / 48 / 64 for 48 / 48 items in all six models (`cpu_vs_3070_share_equal` 1.0).

## 1. Why the hold decays: hypothesis H-COHERENT-DRIFT

**Measured ($0, descriptive):**

| reading (field) | D3-T s1 | FT-CTRL | FT-SELF | R2-s1 | D3-T s2 | R1-s2 |
|---|---|---|---|---|---|---|
| intact mean a(n 1–8), K′ 64 (`acc.intact`) | 0.476 | 0.590 | 0.693 | 0.608 | 0.391 | 0.405 |
| per-pass step, t 24–64 (B `step_ang_v`, 1 − cos) | 2.5e-4 | 2.2e-4 | 2.2e-4 | 2.5e-4 | 4.1e-4 | 4.4e-4 |
| drift coherence (C `coherence`; random walk ≈ 0.13) | 0.66 | 0.67 | 0.64 | 0.60 | 0.68 | 0.66 |
| answer margin at t 12 (B `margin`, nats) | 12.4 | 13.2 | 12.4 | 12.2 | 12.1 | 11.1 |
| margin slope, t 16–48 (C `margin_slope_16_48`, nats / pass) | −0.29 | −0.28 | −0.26 | −0.28 | −0.27 | −0.26 |
| predicted 0-crossing, 12 + m12 / ‖slope‖ (median) | 55 | 58 | 61 | 55 | 58 | 57 |
| rms(V_64) / rms(V_0), value slot (B `rms_v`) | 1.81 | 1.85 | 1.69 | 1.72 | 1.91 | 1.81 |
| state decodes the held value at t 24 (B `rec_ok_v`) | 0.85 | 0.85 | 0.92 | 0.69 | 0.75 | 0.77 |
| wrong answers in the top-2 classes, K′ 64 (A2) | 0.96 | 0.92 | 0.73 | 0.72 | 0.83 | 0.68 |
| monotone items (once wrong, wrong later; A3) | 0.995 | 0.996 | 0.994 | 0.996 | 0.997 | 0.997 |
| non-attractor-y acc, K′ 64 (A2 `acc_nonattr`) | 0.421 | 0.533 | 0.631 | 0.568 | 0.346 | 0.293 |

- The D3-T s1 trajectory (B, n 4): margin 13.2 / 9.4 / 4.4 / −0.13 at t 12 / 24 / 40 / 64. Over the same passes, `drift_ang_v` (1 − cos vs V_{n+2}) is 0.06 / 0.14 / 0.26 / 0.46. Both fall almost linearly.
- The pass-index clock (213-27 AGE-PROJ, refit in B) reads 11.4 at t 12, 15 at t 24 and 17–19 at t 64 (`clock64_mean_over_n`). It saturates; it does not grow linearly.
- The state route alone dies early (216-15 / eval_hist24):
  - bank_ablate_slva is 1.0 at K′ = n + 2 and 0.15–0.27 at K′ 24;
  - dup_newest is ≈ 0.10 at K′ 24.
  - So the coda holds the answer through older bank versions after the current state has drifted (`rec_ok_v` → 0 by t 48–64).
- A7: items fail together across models. Both D3-T s1 and FT-CTRL are wrong on 0.091 of items at K′ 48, against 0.020 if independent. They pick the same wrong answer on only 0.28 of those items (0.12 at K′ 64), so each model has its own attractors.
- A6 (a logistic fit on K′ and n) separates "idle count" from "absolute pass" poorly: −b_n / b_K ranges from −0.23 to 2.27. Deeper n holds better at the same K′ in 5 / 6 models.

**H-COHERENT-DRIFT.** The idle map has no answer-specific fixed point.
- After the settling steps at n + 1 and n + 2 (the n + 2 step is ≈ 0.03–0.05), the state moves a small, nearly constant step per pass, in a mostly fixed direction (coherence 0.60–0.68).
- The answer margin falls linearly, by ≈ 0.27 nats per pass from ≈ 12 nats, and crosses 0 near t 55–60. That matches the observed K′ 48–64 decay.
- Items then absorb into 1–2 answer-independent classes, chosen per model.
- The drift is already present inside the trained range (D3-T s1 at t 12: step 5e-4 – 1.5e-3). Past 12 its rate does not rise; it stays flat or falls. **Nothing switches on beyond the training horizon; the in-range drift simply accumulates.**

**Rivals:**
- **H-NORM (state norm growth): refuted.**
  - [PROVEN, code] The pass input is U_t = Σ α N(V_s), a convex mix of per-slot rms-normalised vectors. Its rms cannot accumulate.
  - [measured] rms(V_t) is flat: +6–10 % from t 12 to t 64 in D3-T s1, and 1.69–1.91 × rms0 at t 64 in all six models.
- **H-POS (pass-index / position extrapolation as the trigger): disfavoured.**
  - The step rate is the same in range and beyond, and the clock saturates.
  - The time-homogeneous route (bank_ablate: V_0 has no age term, the newest version is at age 0) loses the answer sooner: 0.15–0.27 at K′ 24. That arm is off-distribution.
  - The bank's age / size terms may still shape when the anchor fades. That is not tested here.
- **H-ROT (slow rotation): absorbed.** The "rotation" is the coherent drift itself. It is not periodic: recovery from wrong at K′ 48 to right at K′ 64 is 0.006–0.054 (A3).

**Why the base recipe allows the drift [DERIVED from vbank.vb_loss + the B numbers]:**
- Idle passes are trained only for K = min(n + U, 12), U ~ U{0..4}, so ≤ 4 idle passes.
- On those passes, L_rec (HOLD targets) and L_ans are classification losses. Any trajectory that stays inside the answer cell is nearly free.
- At a margin of ≈ 12 nats the CE is ≈ e^−12 ≈ 6e-6. The final answer_ce in s213_d3t_s1 log.jsonl at 48k is 5.0e-6.
- 4 idle passes × 0.27 nats ≈ 1.1 nats of the 12-nat margin, so the gradient pressure against the drift is effectively zero.

**Optional extra mechanism check (Experiments):** none needed, because B runs on CPU. A full-test-set version of B to K′ 128 can run as part of the v1 evals (`fh_check_cpu.py` takes a run dir).

## 2. Method

**Primary lever: FH.** Over the plain D3-T recipe (s213 args, 48k, seed s, 3090) it adds two things.
- **(a) Far sub-batch (RD's CE).** The first q = 16 items of each plain batch are reused, so no new draws.
  - Draw T_far ~ U{13..48} from a separate rng_far = default_rng(seed + 7100).
  - Run passes 1..T_far − 1 with no gradient (rollout `grad_from = T_far`; bank K/V detached), then pass T_far with gradient.
  - L_far = answer CE of the coda at T_far, weight 1.
- **(b) The gated relative fixed-point penalty.**
  - L_FP = λ · mean over (item, t) ∈ IDLE of g_{i,t} · r_{i,t}.
  - r_{i,t} = mean over real slots j of ‖V_t[j] − sg V_{t−1}[j]‖² / ‖sg V_{t−1}[j]‖². This is relative, with the denominator detached (the 192-2 perturbation-metric rule).
  - IDLE = {t ∈ [n_i + 3, K_i]} in the main batch (only items with U ≥ 3; gradient as the existing loss) ∪ {T_far} in the far sub-batch. The settling steps at n + 1 and n + 2 are excluded.
  - The gate g_{i,t} = 1[the coda answer after pass t − 1 == y_i] (argmax, no-grad). The penalty only holds a state that still decodes the right answer.
  - λ = 0.05: Viakhirev's headline weight. **It does not transfer**, because our r is relative (base r ≈ 5e-4 at idle passes). See Q1.
- **Rival / ablation arm: RD (random depth, Anil 2211.09961).** Identical to FH except λ = 0: (a) only.
  - So FH − RD isolates the state penalty. This is Viakhirev's answer-only control, on our model.
- **Plain control: the existing s213_d3t_s1 / s216_d3t_s2.** Same recipe, seeds 1 / 2, RTX 3090.
  - Dedupe rule: the same seed + card reproduces a run exactly, so a plain re-run would be a duplicate. **No new plain runs.**
  - G-FH-EQ (§6 K5) verifies that the FH code with the far rollout and λ = 0 / L_far weight 0 reproduces D3-T s1's first 200 log records bitwise on a 3090.
  - Fallback if G-FH-EQ is not bitwise: a plain run on the same host, about $1.1 per seed.
- **Rules carried over:**
  - §214 CurGate: `GATE.update` keeps reading the plain batch's answer_ce. L_far and L_FP are logged separately and asserted out of it.
  - One GPU class: 3090 for training (as the controls). One eval card for every model in the registered readings.
  - New flags are default-off: `--far "16,13,48"`, `--far_ce 1`, `--fh_w 0.05`. Every pre-FH args.json stays loadable unchanged.

## 3. Derivation

**Assumptions:**
- A1 (as measured): idle dynamics ≈ z_{t+1} = z_t + ω d̂_t + noise, with ω (angular speed) nearly constant and d̂ coherent. The margin falls at rate ṁ ∝ ω (the margin slope and step size move together in B).
- A2: the coda's answer is a monotone function of the margin. The answer is lost when the margin < 0.
- A3: the penalty's effect transfers from the visited idle states (t ≤ 48) to t ≤ 128.

**Claims:**
- C1 [PROVEN, code] The base losses act on idle passes only for t ≤ 12, as cell-membership terms (see §1).
- C2 [DERIVED, A1 + A2] Horizon K_fail ≈ t_0 + m(t_0) / |ṁ|. On the base this gives 55–61, against observed acc 0.39–0.69 at K′ 64 (item-level corr of predicted and observed crossing 0.04–0.75, 48 items per model).
- C3 [DERIVED] Holding to K′ 128 from m(12) ≈ 12 needs |ṁ| ≤ 12 / 116 ≈ 0.10 nats per pass, i.e. ω down ≥ 2.7×.
  - Under A1, r ≈ ω² (equal rms), so that means r down ≥ 7× (to r ≲ 7e-5 at idle passes).
- C4 [DERIVED] FH pushes r → 0 directly. The gradient of g·r w.r.t. V_t is 2g(V_t − c) / ‖c‖² with c detached: it points toward the previous state, and nothing pushes the scale.
- C5 [DERIVED] RD raises the margin at T_far only. It can be met either by lowering ω or by making the readout invariant to d̂.
  - Only lowering ω carries past 48. RD's horizon is therefore uncertain past 48.
  - Viakhirev 2608.18222 reports that an answer-only penalty does not give safety [ASSUMED transferable].
- C6 [PROVEN, Banach] If the idle map is κ-Lipschitz (κ < 1) on a ball around z* that lies inside the answer cell, the hold is permanent.
  - FH does not enforce κ < 1, only near-stationarity at visited states. Contraction is not claimed (Q3).
- C7 [ASSUMED] Penalising at t ≥ n + 3 leaves the computing passes free.
  - Weights are shared, so leakage into the computation is possible. The §4 guards measure it.

**Predictions (P = Theory's prior):**
- P1: HOLD(FH) in both seeds: **0.45**.
- P2: HOLD(RD) in both seeds: **0.25**.
- P3: NO-LOSS(FH) and DEPTH(FH), both seeds: **0.70**.
- P4: MECH, step v(FH) ≤ v_base / 3 at t 24–64: **0.70**.
- P5: v(RD) ≥ v_base / 2: **0.60**.
- P6 (descriptive): FH bank_ablate_slva at K′ 24 ≥ 0.5: **0.35**.

**Goodhart analysis (incl. Audit's point: a fixed-point target is met by freezing everything):**
- **G1, freeze the computation.**
  - Blocked in the loss:
    - L_FP acts only at t ≥ n + 3 (and at T_far ≥ 13 > n);
    - R2's changed-slot CE requires E_t ≠ E_{t−1} transitions on computing passes, so motion is required there and stillness only after;
    - the gate is 0 until the answer is right, so the penalty cannot fire before the computation succeeds.
  - Shared weights can still leak contraction into the computation. Guarded by §4 R-NOLOSS (acc at K′ = n + 2, STALL, PROBE(t), early finish) and R-DEPTH.
- **G2, collapse to an answer-independent attractor.** This is today's failure mode, and a constant state satisfies r = 0 perfectly.
  - Blocked by the gate (a wrong state is never pulled still), L_far, and the in-range L_ans. A constant state cannot be right for all y.
  - Read by A2: the top-2 share of wrong answers and the non-attractor accuracy, per arm.
- **G3, scale.** The denominator is detached and readers are LN / N()-normalised, so there is no gradient route to inflating ‖V‖.
  - `rms_v` is reported. More than ±25 % vs base at t 64 is flagged.
- **G4, gate dodging (making answers wrong to escape the penalty).** It costs L_far and in-range L_ans, so the net incentive is negative. This is why FH keeps L_far.
- **G5, one-sided dynamics target (§199 CSR lesson).** By phase, the target is two-sided: motion on computing passes (R2), stillness on idle passes (L_FP). The phase comes from n at training time only; at test time the model must infer idle-ness from its state.
- **G6, trained readings.** K′ 24–48 lie inside the trained range of L_far, so they are trained readings for both arms. The primary pass needs K′ 64, 96 and 128 as well, which lie beyond any trained pass.

## 4. Registered readings (proposed)

Evals: eval_vb / eval_deep / eval_hist24 as for D3-T, and part_khorizon with ks (24, 32, 48, 64, 96, 128).
- K′ 128 does not fit an 8 GB card at bs 32 (K′ 64 peaked at 0.83 of a 3070).
- So all six models (FH s1 / s2, RD s1 / s2, D3-T s1 / s2) run on one cloud 3090, fp32, eval_bs 32 (the fallback to 16 is recorded).
- G-FH0: the bases' K′ ≤ 64 answers on that card vs the committed 3070 files, share equal ≥ 0.998 (cross-card, not bitwise).
- Scorer: a new `theory/score_fh.py` built on score_s216_ih's loader. It must be synthetic-run on fixtures first.

| reading | definition (fields) | pass |
|---|---|---|
| **R-HOLD (primary)** | H(K′) = mean over n 1..8 of eval_khorizon `acc.intact[K′][n]` | HOLD(A, s): H ≥ 0.95 at every K′ in {24, 32, 48, 64, 96, 128} |
| R-HOLD-GAIN | H_A(128) − H_base,s(128) | ≥ 0.30 |
| **R-NOLOSS** | eval_vb `acc_n2[n]`, n 1..8; `stall.recovery`; early finish = mean over n 2..8 of `acc_k[n−1][n]`; PROBE: `probe["4"].acc_by_n` | acc_n2 ≥ 0.99 at each n; recovery ≥ base − 0.02; early finish ≤ base + 0.05 (base s1 0.204, s2 0.174); PROBE(4) mean n 7–8 ≤ base + 0.05 and n 1–4 ≥ 0.98 |
| **R-DEPTH (C1-relevant)** | mean eval_vb `acc_n2` n 9–11 (base s1 0.597, s2 0.666); C1 floor `acc_k["0"]`, `acc_k["1"]` | ≥ base − 0.10; K′ 0 / 1 ≤ chance + 0.05 |
| R-MECH (supporting, not a gate) | B on the new model: v = `step_ang_v` t 24–64, `margin_slope_16_48`, `coherence`, `rms_v` | FH: v ≤ v_base / 3 |
| R-EXPOSURE | logged `fh_gate_share` (share of IDLE terms with g = 1) and `fh_r_far`, `fh_r_in`, `far_ce` | mean `fh_gate_share` over steps 8k–48k ≥ 0.05, else UNEXPOSED |
| R-HIST (descriptive) | khorizon bank_ablate_slva / self_ablate / other_ablate per K′; hist24 dup_newest / dup_foreign; A2 top-2 share and non-attractor acc; A3 monotone share | — |

**Verdicts:**
- **FH-RECIPE:** HOLD ∧ NO-LOSS ∧ DEPTH in both seeds.
- **RD-RECIPE:** the same for RD.
- If both pass: RD SUFFICES (no penalty needed; the simpler recipe). If FH-RECIPE passes and RD fails HOLD in ≥ 1 seed: PENALTY NEEDED.
- FH with HOLD-GAIN in both seeds but not HOLD: PARTIAL.
- FH fails HOLD with v ≥ v_base / 1.5 and NO-LOSS intact: UNDERDOSED. This is a null at λ 0.05 and stops the design; any new dose is a new registration (no problem → fix loop).

## 5. OC block, censoring, $

**Seeds:** 2 per arm (s1, s2), paired with the D3-T control of the same seed. Sampling SE of H(K′) over 8,000 items at 0.95 is 0.0024, which is negligible; seed variance dominates.

**HOLD, under no effect.** Take the six no-lever models' H(64) as the null: mean 0.527, sample SD 0.122.
- P(H(64) ≥ 0.95) = P(z ≥ 3.47) ≈ 2.6e-4 per seed. H(128) can only be lower.
- **P(pass | no effect) ≤ 2.6e-4 per seed, ≈ 7e-8 for both seeds** (independent); ≤ 1e-3 allowing fat tails.

**HOLD, under effect.** Model the effect as a true H(128) ~ N(0.97, 0.03) per seed (Viakhirev: settling runs lose ≤ 0.004).
- P(≥ 0.95) = Φ(0.67) = 0.75 per seed.
- **P(pass | effect) for both seeds = 0.56 (independent) to 0.75 (correlated).**

**HOLD-GAIN.** The null is the same-seed training perturbation (FT / self rounds vs their base at K′ 64: +0.11 / +0.22 / +0.13 / +0.01) ≈ N(0.12, 0.08).
- P(≥ 0.30) = 0.012 per seed, so 1.5e-4 for both.
- Under effect: ≈ 1.

**DEPTH, under no harm.** Assume a perturbation SD of 0.05 (s1 vs s2 bases: 0.597 vs 0.666).
- P(Δ ≥ −0.10) = Φ(2) = 0.977 per seed, so 0.955 for both.
- Under a true harm of −0.15: detected in ≥ 1 seed with P = 1 − 0.16² = 0.97.

**NO-LOSS:** P(pass | no harm) ≈ 0.95 for both (the bases are 1.0 at every n 1..8).

**FH-RECIPE:** P(pass | effect, no harm) ≈ 0.56 × 0.95 × 0.955 ≈ **0.51**; P(pass | no effect) < 1e-3.

**Censoring:**
- A run that NaNs, dies or stops before 48k counts as FAIL for that arm and seed. It is not dropped.
- A K′ that OOMs at eval_bs 32 is re-run at 16 (recorded). It is never dropped.
- Every one of the 8,000 test items counts; argmax answers, no abstention.
- UNEXPOSED (gate share < 0.05) = FAIL for the recipe claim, reported as "penalty never applied".
- A missing eval file = pending, never a pass.

**Realised-$ estimate (cloud 3090 at $0.22 / h):**
- Base D3-T 48k = 17,865 s ≈ 4.96 h ≈ $1.09 (s213_d3t_s1 log.jsonl `sec`).
- Far sub-batch: 16 items × mean 30.5 no-grad passes + 1 grad pass, against the base step (64 items × ≈ 8 passes × 3 for forward + backward). That is ≈ +35 %, +10 % for longer bank attention, so ≈ ×1.45 ≈ **$1.6 per run; 4 runs $6.4.**
- mem_check / gtime / G-FH-EQ: ≈ $0.3.
- Evals: 6 models × ≈ 40 min (the 3070's 26 min at K′ 64, × 2.5 for K′ 128, ÷ 1.6 for a 3090) ≈ **$0.9**. eval_vb / eval_deep run locally at $0 (or ≈ $0.5 on cloud).
- **Total ≈ $8–10 with 30 % host overhead. Cap $15** (≤ $30, so Audit can auto-approve).

## 6. $0 known-answer checks before v1 (Experiments implements; Audit reads)

- **K1, gradflow** (scratch/gradflow_check.py, fp64 CPU):
  - L_FP and L_far each give a nonzero grad on the core params;
  - L_FP's grad is 0 on items with g = 0;
  - grad is 0 through the no-grad prefix (assert the prefix bank tensors have requires_grad False).
- **K2, penalty known answers:**
  - via an edit hook: V_t := V_{t−1} → r = 0; V_t := 2 V_{t−1} → r = 1.0; a rotation by θ at equal norm → r = 2(1 − cos θ) (|err| ≤ 1e-9, fp64);
  - the denominator carries no grad.
- **K3, gate:** `fh_gate_share` equals the share of correct coda answers after pass t − 1, recomputed separately on a fixture batch.
- **K4, plain-path equivalence** (refactor rule): `rollout(grad_from=None)` and the trainer with every new flag off are bitwise equal to the pre-FH commit (fp64 CPU, 1 step), plus construct + 1 step without new flags.
- **K5, streams:**
  - a T_far histogram over 10k draws (U{13..48}, mean 30.5 ± 0.2);
  - the plain draws_log `plain_hash` for steps ≤ 200 is identical to D3-T s1's draws_log ($0 CPU replay);
  - G-FH-EQ on the launch 3090: λ 0 and far_ce 0, with the far rollout on, gives log records bitwise equal to s213_d3t_s1 for steps 1–200.
- **K6, CurGate isolation:** a named assertion that `GATE.update` receives the plain answer_ce only. A 20-step CPU fixture with far_ce 100 must leave CurGate.state() identical to far_ce 0.
- **K7, memory** at the largest shape (n 8, the 16 longest items, T_far 48, plus the main batch) via mem_check on the 3090; the peak is recorded.
- **K8, scorer:** score_fh.py on synthetic eval files with known verdicts (each verdict branch hit). Grep vbank_diag / vbank_eval for every field it reads.
- **K9, eval extension:** part_khorizon with ks to 128 on D3-T s1 reproduces the committed K′ ≤ 64 answers on the 3070 at bs 32 (bitwise) before any cloud eval. Its memory fallback is exercised.
- **K10, mechanism baseline:** `fh_check_cpu.py` on each new model ($0 CPU, about 2 min each), compared with this folder's base values.

## 7. Novelty sentences for Literature (no "new" / "first" until checked)

- S1: "An idle-pass fixed-point penalty gated on the decoded answer being correct." Check against Viakhirev 2608.18222 (terminal, gated?), STARS 2605.26733, Movahedi 2606.18206 and the DEQ literature.
- S2: "A fixed-point penalty applied only after the computation completes, with motion required on computing passes (a phase-split dynamics target)." Check Huginn / Ouro analyses, Kolawole 2609.39277.
- S3: "In a version-bank loop, the held answer outlives the current state: the state's own decode is lost by K′ 24–48 while the coda still reads the answer from older bank versions." Check memory-augmented looped models.
- S4: "Past the horizon, held answers fall linearly in margin (≈ 0.27 nats per pass, coherent drift) and absorb into 1–2 answer-independent classes per model." Check Shapiro 2608.11233, Liang / Jin / Pan 2609.33144, Kolawole 2609.39277.
- S5: "A matched random-depth vs state-penalty contrast with identical rollout schedules." This is Viakhirev's answer-only control; check the exact protocol.

## 8. Open questions for the red team

- **Q1, dose.** λ 0.05 on a relative r of ≈ 5e-4 gives λ·r ≈ 2.5e-5. That is near the final CE (5e-6) but far below the mid-training CE (0.01–0.1).
  - Is a single dose acceptable, or should λ be set by a grad-norm ratio rule at a fixed step (e.g. ‖∇L_FP‖ / ‖∇L_ans+rec‖ = 0.2 at step 16k)?
- **Q2, per-pass vs anchor target.** Coherence 0.65 means small steps add up. Should r be measured against V_{n+2} (the anchor) rather than V_{t−1}?
  - The anchor forbids cumulative drift, but it fixes the attractor at a possibly unsettled state.
- **Q3, contraction.** A 1-pass gradient window at T_far constrains stationarity, not the Jacobian. Should a relative-gap perturbation-growth reading (192-2 rule) be added, as a descriptive GPU check for Experiments?
- **Q4, depth risk.** AGE-PROJ proj_out cost n 9 −0.056 (0.923 → 0.866): the drift / clock may carry depth-extrapolation information that FH would remove. Is R-DEPTH's −0.10 margin too loose for C1 (priority C1 > C6)?
- **Q5, range.** T_far ≤ 48, read to 128 (2.7×). Base: trained to 12, holds to ≈ 32–48. Is extending T_far to 64 worth about +25 % cost?
- **Q6, gate.** The gate uses the coda answer. Should it also require R to decode the held value at the value slot, so the penalty doesn't freeze a state the coda reads only through the bank?
- **Q7, bank regime.** At K′ 128, ≥ 90 % of bank versions are idle copies. dup_newest (all copies) is at chance at K′ 24 in the base. Does FH's fixed point survive a copy-dominated bank, or does the far range need to cover it?
- **Q8, scratch vs fine-tune.** A 12k fine-tune from the D3-T checkpoints costs ≈ $0.4 per run, with FT-CTRL s1 as an existing paired control (no FT-CTRL s2 exists). Is it a cheaper pilot, or does the base-recipe claim need 48k from scratch?
- **Q9, seeds.** v differs about 1.7× between the lineages (s1 2.2–2.5e-4, s2 4.1–4.4e-4). Are 2 seeds enough, or is s3 needed?
- **Q10, settling boundary.** IDLE starts at n + 3; the step into n + 2 is still ≈ 0.03–0.05, about 100× the idle step. Is that the right boundary? It leaves in-range IDLE terms only for items with U ≥ 3.
