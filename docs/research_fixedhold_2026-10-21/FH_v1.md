# FIXED-HOLD v1 (FH): an anchored far-pass fixed-point lever for the C6 hold failure (2026-10-21)

**Status:** v1 = FH_draft.md v0 (1314f24c), changed as listed here; where this file and v0 differ, this file wins. Red team: REVIEW_FH.md (GO to v1 with fixes 1–13, all adopted). Literature: LIT_FIXEDHOLD_NOVELTY_2026-10-21.md (ed051765). Origin: Note 216-15 (every VB model loses its held answer past ≈ 2–3× its trained passes).

## 0. Evidence the design rests on ($0)
- **H-COHERENT-DRIFT** (v0 §1, measured): after pass n + 2, the current state V_t moves a small, coherent step every idle pass (ω ≈ 0.020 rad s1 / 0.024 s2; coherence 0.60–0.68), and the answer margin falls ≈ 0.22–0.27 nats / pass.
- **Oracle arms** (REVIEW_FH §0; review_scripts/rv_oracle_*; 48 items per model, intact answers equal to eval_khorizon at K′ ≤ 64):
  - **freeze** (stored S_t := V_{n+2} for t > n + 2): acc 1.00 at K′ 64 / 96 / 128 in D3-T s1 and s2, margin flat at 14.6–14.7.
  - cur_freeze (s1): the coda's current version frozen, bank drifting → 1.00 to 128.
  - bank_freeze (s1): the bank frozen, current computed → 1.00.
  - freeze32 → 0.96 at a reduced margin of ≈ 6.2.
  - Inference damping (¼ step) → 0.25 / 0.44 at 128.
- **Reading:** V_{n+2} is a high-margin fixed point the coda can hold forever. The failure is drift of the current state, fed back through the bank. A target that pulls the far state back to V_{n+2} has a verified ceiling. v0 S3's bank-version causal clause is withdrawn.

## 1. Method (single change per arm)
- **Far branch (both FH and RD):**
  - one T_far ~ U{13..48} per step, shared by the far sub-batch (rng_far = default_rng(seed + 7100));
  - the first 16 plain items of the step; passes 1..T_far − 1 with no gradient (bank K / V detached), pass T_far with gradient;
  - L_far = answer CE of the coda at T_far, weight 1.
- **FH = RD + one penalty (anchor form, fix 5):** L_FP = λ · g · r_anchor, with r_anchor = ‖V_{T_far} − sg V_{n_i+2}‖² / ‖sg V_{n_i+2}‖² per item.
  - V_{n+2} comes from the same no-grad prefix; n_i is the item's operator count (known at train time).
  - g = 1 if the coda answer at T_far − 1 is correct, else 0. g is ≈ 1 at most states (fix 10), so the protection against an answer-independent attractor comes from L_far, not from g.
  - **No in-range penalty terms (fix 2):** L_FP acts only at T_far ≥ 13 > n + 2, behind a no-grad prefix. It cannot send gradient into computing-pass activations, only into the shared weights.
- **Dose rule, not a number (fix 4):**
  - L_FP is computed at λ = 1 from step 1 (logged, not in the loss).
  - λ is set once so that the median over the dose window of ‖∇θ λ L_FP‖ / ‖∇θ (L_ans + L_rec + L_far)‖ = 0.2; capped at 10, fixed afterwards.
  - Dose window: Stage 1 steps 250–500; Stage 2 steps 7.5k–8k (end of the c-anneal).
  - The ratio is logged every 100 steps. RD logs r_anchor (fh_r_far) at λ = 0.
- **CurGate (fix 11):** GATE.update reads only the plain batch's answer_ce. L_far and L_FP are logged apart (K6).
  - Advance steps are reported per arm. |Δ| > 2k vs the same-seed comparator ⇒ CONFOUNDED flag on the HOLD / DEPTH contrast.

## 2. Stages (fix 13; a staged pre-registration, not a problem → fix loop)
- **Stage 1, fine-tune pilot:** FH-FT and RD-FT × s1 / s2, 12k steps from D3-T s1 / s2 model.pt, lr 3e-5 (as FT-CTRL), plain D3-T batch mix.
  - New comparator **FT-CTRL s2** (same FT recipe, no far branch). FT-CTRL s1 = s215_ftctrl_s1 if its args match the FT recipe except the far flags (args diff checked); otherwise a new FT-CTRL s1 is run too.
- **Stage 2, base recipe from scratch** (runs only if Stage 1's TRIGGER passes): FH and RD × s1 / s2, the s213 D3-T args, 48k, on a 3090. Plain controls = the existing D3-T s1 / s2 (G-FH-EQ, fix 7).
  - **Contingent s3** for FH and RD, triggered only by a seed split on any Stage-2 verdict (fix 6).

## 3. Derivation (v0 §3 kept, with these changes)
- C3 is re-tagged [DERIVED under A1; lineage-specific constant] (slope / ω −14 s1 vs −11 s2). Every threshold is stated in r ≈ 2(1 − cos) or ω, never in 1 − cos (fix 3).
- **C7 corrected:** with no in-range terms, ∂L_FP reaches computing passes only through shared weights [DERIVED from the no-grad prefix]. K1 asserts it.
- **C8 [MEASURED, oracle]:** the stored state V_{n+2} is a fixed point the coda holds to K′ 128 (§0). The anchor pulls toward a state already verified to hold.
- **Goodhart:**
  - G1, freezing the computation: blocked in the graph (no-grad prefix). Through shared weights it is guarded by STALL (`stall.recovery` ≥ base − 0.02, exposed: t 2–3), PROBE(4) n 7–8, early-finish and DEPTH.
  - G2, an answer-independent attractor: blocked by L_far (a constant state cannot be right for all y).
  - G3, rms collapse: `rms_v` within ± 25 % of base.
  - G4, gate dodging: costs L_far.
  - G6: K′ ≤ 48 lie inside the trained T_far range (trained readings), so the primary needs K′ 64 / 96 / 128.
- **Predictions (Theory's prior):**
  - P1 Stage-1 TRIGGER: 0.45.
  - P2 FH-RECIPE (Stage 2, both seeds): 0.40.
  - P3 PENALTY NEEDED given FH-RECIPE: 0.50.
  - P4 RD alone holds to 128 in both seeds: 0.20.

## 4. Registered readings
H(K′) = mean over n 1..8 of eval_khorizon `acc.intact[K′][n]`, ks {24, 32, 48, 64, 96, 128}, on one 3090 class, fp32, bs as recorded.

**Stage 1:**
- **TRIGGER:** H_FH-FT(128) − H_FT-CTRL,s(128) ≥ 0.30 in both seeds ∧ FT-NOLOSS.
- FT-NOLOSS: acc_n2 n 1–8 ≥ 0.99, stall.recovery ≥ comparator − 0.02, DEPTH-FT ≥ comparator − 0.10.
- Descriptive: the RD-FT gain; r_FH / r_RD (fh_r_far at 12k); the mechanism (fh_check_cpu: ω, coherence, margin slope).

**Stage 2:**
- **R-HOLD (primary):** HOLD(A, s) ⇔ H ≥ 0.95 at every K′ in ks.
- **R-NOLOSS:** acc_n2 n 1–8 ≥ 0.99; stall.recovery ≥ base − 0.02; PROBE(4) n 7–8 ≤ base + 0.05; early-finish ≤ base + 0.05.
  - C1 floor (fix 10): `acc_k["0"]` over n 1..16 and `acc_k["1"]` over n 2..16, each ≤ base + 0.05 per n.
- **R-DEPTH:** mean `acc_n2` n 9–11 ≥ base − 0.10 (FAIL line). A two-seed mean < base − 0.05 is a registered WARN: no recipe claim carries into C1 work without a depth re-test.
- **R-MECH:**
  - FH: ω_FH ≤ ω_base / 2.7 (r ≤ r_base / 7.3) over t 24–64.
  - PENALTY needs r_FH ≤ r_RD / 3.
  - Margin read against the oracle-freeze ceiling 14.6–14.7.

**Verdicts:**
- **FH-RECIPE:** HOLD ∧ NOLOSS ∧ DEPTH in both seeds (labelled "the recipe beats the base").
- **PENALTY NEEDED (fix 1):** FH-RECIPE ∧ H_FH(128) − H_RD(128) ≥ 0.15 in both seeds ∧ r_FH ≤ r_RD / 3.
- **RD SUFFICES:** RD passes HOLD ∧ NOLOSS ∧ DEPTH in both seeds.
- **PARTIAL:** FH HOLD-GAIN (H_FH(128) − H_base(128) ≥ 0.30) in both seeds without HOLD.
- **UNEXPOSED:** logged fh_r_far(FH) is not ≥ 3× below RD's by the end ⇒ the penalty never bit; no reading of the penalty.
- **UNDERDOSED** (only after exposure passes): FH fails HOLD with r_FH ≥ r_RD / 1.5 and NOLOSS intact. A null at the registered dose rule that stops the design; any new dose is a new registration.

## 5. OC, censoring, $
- **Seed variance:** 2 seeds; each seed is its own lineage (H(64) 0.48 vs 0.39; ω 1.2×). A 1 / 2 split triggers s3 (Stage 2) or is reported as SPLIT (Stage 1, no Stage 2).
- **Stage-1 TRIGGER | no effect:**
  - FH-FT and FT-CTRL are independent fine-tunes of the same base, so null difference ~ N(0, √2 × 0.08) (FT perturbation SD from v0 §5).
  - P(≥ 0.30) = 0.004 per seed → **1.6e-5** both.
- **TRIGGER | effect** (FH-FT reaches H(128) ≥ 0.9, FT-CTRL ≈ 0.2–0.35): ≈ 0.6 per seed [ASSUMED: 12k may be too short to settle] → **≈ 0.4–0.5** both.
- **HOLD | no effect (Stage 2):** null H(64) N(0.527, 0.122), and H must be ≥ 0.95 at 128 as well → < 1e-3.
- **HOLD | effect:** true H(128) ~ N(0.97, 0.03) [ASSUMED, not grounded in our data] → 0.56–0.75 both seeds; at a mean of 0.95, ≈ 0.25.
- **DEPTH | no harm:** independent runs, so difference SD = √2 × 0.049 = 0.069: P(Δ ≥ −0.10) = 0.926 per seed, **0.86** both. A true −0.15 harm is missed per seed with P 0.23.
- **FH-RECIPE | effect, no harm:** ≈ 0.56 × 0.95 × 0.86 = **0.46**. **| no effect:** < 1e-3.
- **PENALTY NEEDED | FH ≡ RD:** difference ~ N(0, √2 × 0.05) → P(≥ 0.15) = 0.017 per seed → **3e-4** both (needs a 1-run SD of H(128) ≤ 0.05; checked from the s1 / s2 spread, reported).
- **Censoring (rule b):**
  - a dead run (log < its steps) fails every reading of that arm;
  - a missing K′ 128 is pending, and FAIL at --final;
  - part_khorizon must not drop K′ (fix 9): an OOM falls back to a smaller eval_bs with ks kept, recorded;
  - a never-exposed penalty reads UNEXPOSED (no PENALTY / UNDERDOSED reading);
  - a run stopped by the cost rule reads STOPPED-COST (not a pass);
  - never-firing gates count as g = 0 items (reported).
- **Realised-$ (cloud 3090, $0.22 / h; one T_far per step, mean 30.5 passes, launch-bound loop ⇒ about +100 % per step, fix 8):**
  - Stage 1: 5 FT runs × ≈ $0.35 + evals ≈ **$2.5**, cap $4.
  - Stage 2: 4 × ≈ $2.3 + evals ≈ $0.9 + contingent s3 ≈ $3.4 ≈ **$13.5**, cap $16.
  - **Family cap $20** (Audit's ≤ $30 rule).
  - Measured gtime at K7 goes into the launch note. A Stage-2 run projecting > $2.4 is not launched (STOPPED-COST) without a Note.
  - Realised at registration: $0.

## 6. $0 checks before launch (Experiments implements; Audit G-D)
- K1: gradflow (fp64 CPU). L_far and L_FP give nonzero grad on the core params, and **zero grad on V_{≤ n+2} from L_FP**.
- K2: anchor known answers. r_anchor = 0 when V_{T_far} = V_{n+2}; scale-invariant; gate masks wrong items.
- K3: the gate share equals a separate recompute.
- K4: plain-path equivalence: rollout(grad_from=None) and every new flag off are bitwise equal to the pre-FH commit (fp64 CPU), plus construct + 1 step without new flags.
- K5: streams. A T_far histogram (U{13..48}); one T_far per step.
  - **G-FH-EQ** on the launch host: steps 1–1200 (12 records, crossing the c hold) bitwise vs D3-T; micro equal; the weight-0 far loss runs backward; a 0 × non-finite guard.
  - If not bitwise, re-run the pre-FH commit on the same host: host-caused → the control stands with a note; code-caused → fix, or plain re-runs ($1.1 per seed).
- K6: CurGate isolation (far_ce 100 vs 0 ⇒ an identical CurGate.state() on a 20-step fixture).
- K7: memory and gtime at the largest shape on the 3090 (auto_micro's probe includes the far branch).
- K8: score_fh.py fixtures, one per verdict branch; field paths grepped in vbank_diag / vbank_eval.
- K9: part_khorizon to K′ 128 reproduces the committed K′ ≤ 64 answers on the 3070 at bs 32 (bitwise); the eval_bs fallback is exercised.
- K10: fh_check_cpu baselines on each new model.
- K11: resume bitwise at a 200-step boundary, rng_far restored. Off-host full-state mirrors for Stage-2 runs (≈ 7 h), one mirror dir per run.

## 7. Novelty (Literature ed051765)
- S1 OK: "a relative, correctness-gated, anchored variant of terminal fixed-point penalties (Viakhirev 2608.18222)".
- S2 NARROW: per-item idle boundary + far rollout; Viakhirev penalises the terminal step.
- S3 reworded: descriptive only (`rec_ok_v` falls while the coda answer holds); the causal clause is dropped per §0.
- S4 NARROW: post-horizon decay is reported (2608.11233, 2609.33144).
- S5 OK as a design.
- Damping (Guo / Wu / Du 2609.16665) is recorded as an oracle-phase descriptive arm (§0), not a v1 training arm. The answer-only readout arm is deferred, contingent on FH passing and RD failing.
