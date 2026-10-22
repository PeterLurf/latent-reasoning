# REVIEW_FH: red team of FH_draft.md v0 (1314f24c), 2026-10-21 19:09 EDT

Reviewer: red-team helper for Audit. Nothing committed, nothing launched; no blinded files opened.
New $0 checks (CPU fp32, 2 threads, idle priority) are in `review_scripts/`:
- `rv_recompute.py` → `rv_recompute.out.json`: recomputes the draft's numbers from committed files;
- `rv_oracle_freeze.py` → `rv_oracle_freeze.<run>.json`: phase-oracle edits of the idle rollout to K′ 128 on D3-T s1 and D3-T s2;
- `rv_oracle_split.py` → `rv_oracle_split.<run>.json`: arms that separate the current state from the bank.
- Items as check B: 48 per model (n {1, 4, 8} × 16). Intact answers equal eval_khorizon `items.intact` at K′ 24 / 32 / 48 / 64 (`intact_vs_khorizon_share_equal` 1.0 in both models).

## Verdict: GO to v1 with listed fixes

The $0 oracle supports the core idea. Fixes 1–5 change the verdict logic, the loss graph, the thresholds or the dose, so v1 must carry them before registration. Fixes 6–12 are needed before launch.

## 0. New $0 evidence (decides Q2, Q6 and Q7; answers the "$0 diagnostic between FH and RD" question)

Per arm: acc (fields `acc[arm][t]`) and mean margin (`margin_mean[arm][t]`, nats), 48 items.

| arm (what it edits after pass n + 2, oracle n) | s1 acc K′ 64 / 96 / 128 | s1 margin 12 → 128 | s2 acc K′ 64 / 96 / 128 | s2 margin 12 → 128 |
|---|---|---|---|---|
| intact | 0.48 / 0.15 / 0.13 | 12.4 → −1.0 | 0.58 / 0.31 / 0.17 | 12.1 → −0.5 |
| **freeze**: stored S_t := V_{n+2}; the coda reads S_t | **1.00 / 1.00 / 1.00** | 14.6 → 14.6 (flat) | **1.00 / 1.00 / 1.00** | 14.7 → 14.7 |
| freeze_win: freeze, and the banks keep only V_0, V_1..V_{n+2} and the last 9 versions | 1.00 / 1.00 / 1.00 | 14.6 → 14.7 | 1.00 / 1.00 / 1.00 | 14.7 → 14.7 |
| damp: S_t := S_{t−1} + ¼ (V_t − S_{t−1}) (Guo/Wu/Du quarter step) | 1.00 / 0.54 / 0.25 | 13.6 → −0.6 | 0.94 / 0.75 / 0.44 | 13.9 → 0.1 |
| cur_freeze: intact (drifting) bank; the coda's current version := V_{n+2} | 1.00 / 1.00 / 1.00 | 14.6 → 11.2 | S2_CUR |
| bank_freeze: frozen bank; the coda reads the computed V_t | 1.00 / 1.00 / 1.00 | 13.5 → 13.4 | S2_BANK |
| freeze32: stored S_t := V_32 (already drifted) for t > 32 | 0.98 / 0.96 / 0.96 | 6.5 at t 32 → 6.2 | S2_F32 |

**Readings:**
- **A perfect idle fixed point holds the answer to K′ 128 in both lineages** (48 / 48; one-sided 95 % lower bound ≈ 0.94). The margin is flat at ≈ 14.6–14.7.
  - The bank's age terms and a copy-dominated bank do no harm once the state is still. freeze_win is identical to freeze.
  - Q7 is answered (the fixed point survives a copy-dominated bank). H-POS / age-extrapolation is refuted as a cause of the loss.
  - **FH's target is the right one.** This is the ceiling FH can reach.
- **The cause is drift of the current state, fed back through the bank.**
  - With a still current state, the drifting bank costs only ≈ 3.4 nats over 116 passes (cur_freeze, s1).
  - With a frozen bank, the one-step map is still (bank_freeze).
  - **Correct the draft:** §1's "the coda holds the answer through older bank versions after the current state has drifted" is not the causal picture. The bank_ablate / dup arms are off-distribution. Withdraw or reword novelty sentence S3 (see 12).
- **Stillness reached late also holds** (freeze32 at margin ≈ 6.2), but at less than half the margin.
  - So a per-pass penalty works only if it stops the drift before the margin crosses 0. An anchor target gains margin (fix 5).
- **Inference damping is a weak lever.** A nominal 4× smaller idle update slows the margin decay over t 12–64 only 1.4× in s1 (0.237 → 0.170 nats / pass) and 1.2× in s2 (0.221 → 0.178). It fails at K′ 128 (0.25 / 0.44).
  - The step under damping was not measured, so A1's linearity is not tested directly. Either way, C3's "ω down 2.7× suffices" is not a safe sufficiency number (fix 3).

## 1. Issues, ranked

**1. HIGH: the PENALTY NEEDED verdict has a ≈ 0.19 false-positive rate; the OC null is the wrong arm.**
- The OC computes P(pass | no effect) against the bases (≤ 1e-3). But FH = RD + penalty, so the null for the penalty claim is **RD**, whose HOLD prior is 0.25 (P2).
- If the penalty does nothing, FH and RD are exchangeable. Then P(FH passes both seeds ∧ RD fails ≥ 1) = p²(1 − p²): 0.08 / 0.19 / 0.25 at per-seed p 0.3 / 0.5 / 0.7.
- **Fix:**
  - PENALTY NEEDED := FH-RECIPE ∧ (H_FH(128) − H_RD(128) ≥ 0.15 in both seeds) ∧ R-MECH r_FH ≤ r_RD / 3.
  - Add an OC row P(PENALTY NEEDED | FH ≡ RD), using the 1-run SD of H(128), which must be ≤ 0.05.
  - The draft's "< 1e-3" holds only for "the recipe beats the base", and should be labelled so.

**2. HIGH: the in-range L_FP sends gradient into the computing passes through the graph, not only through shared weights.**
- The trainer builds the bank K / V and the SLVA input with gradient (VBankN docstring; `vb_loss` rolls out with BPTT). For t ∈ {n + 3, n + 4}, V_t depends on V_1..V_{n+2}.
- So ∂L_FP flows into the computing-pass activations. It can be met by changing the computation so that pass n + 3 moves little.
- C7 and G1's "blocked in the loss" are false as stated. The sg on V_{t−1} removes only the direct target path.
- **Fix (preferred): drop the in-range terms.** They are only 2 passes, on the 40 % of items with U ≥ 3, and n ≤ 9. Keep L_FP at T_far only, where the prefix is no-grad, so the window is one pass.
  - This also makes Q10 (the n + 3 boundary) moot: T_far ≥ 13 > n + 2 always (n ≤ 8 in training).
- Alternative: compute in-range terms on a one-pass re-run with every input detached. Then K1 must assert zero grad on V_{≤ n+2} from L_FP.

**3. HIGH: the thresholds mix 1 − cos (∝ ω²) with ω, and C3 is s1-specific.**
- B's `step_ang_v` is v = 1 − cos ≈ ω² / 2. C3 needs ω / 2.7, i.e. **v / 7.3**.
- P4 / R-MECH (v ≤ v_base / 3) is ω / 1.73. Under the draft's own A1 that predicts a crossing at 12 + 12 / (0.27 / 1.73) ≈ **89**, a HOLD fail at 96 / 128.
- UNDERDOSED's v ≥ v_base / 1.5 is ω / 1.22.
- The "≈ 1.7× between lineages" is in v. In angle, `omega_mean` is 0.0205 (s1) vs 0.0244 (s2), only 1.2×.
- `slope_over_omega` differs by lineage: −13.9 to −14.5 nats / rad (s1 lineage) vs −11.1 / −10.4 (s2). A1's constant is not shared.
- Within a model, A1 is supported: `pearson_item_omega_slope` is −0.34 to −0.84.
- **Fix:**
  - State every threshold in r or ω (r ≈ 2v).
  - R-MECH FH: r ≤ r_base / 7.3 at t 24–64, and r_FH ≤ r_RD / 3 for PENALTY NEEDED.
  - Re-tag C3 as [DERIVED under A1, lineage-specific constant; the damping arm shows the sufficiency number is not safe].
  - Add the oracle-freeze ceiling (§0) as the reference for the margin reading.

**4. HIGH: the dose is a guess, and the UNDERDOSED stop makes a likely null uninformative.**
- λ 0.05 on a relative r ≈ 5e-4 gives λ·r ≈ 2.5e-5. Viakhirev settles at 0.005 on an absolute residual (Literature: it does not transfer).
- R-EXPOSURE (gate share ≥ 0.05) is vacuous: the bases are correct on ≥ 0.92 of items to K′ 32, so g ≈ 1 at most visited states. It cannot detect an under-dose.
- **Fix:** register a dose rule, not a number.
  - λ is set once per run at step 8k (the end of the c-anneal) so that the median over steps 7.5k–8k of ‖∇θ λL_FP‖ / ‖∇θ (L_ans + L_rec)‖ = 0.2. The penalty is logged at λ = 1 from the start for this purpose.
  - λ is fixed after 8k, capped at 10.
  - Same rule in both seeds; log the ratio every 100 steps.
- Replace R-EXPOSURE with "logged `fh_r_far` falls ≥ 3× below RD's `fh_r_far` (also logged in RD at λ = 0) by 24k; else UNEXPOSED".
- Keep the UNDERDOSED stop (no problem → fix loop), but define it against RD (r_FH ≥ r_RD / 1.5) and only after this exposure check.

**5. MEDIUM-HIGH: per-pass vs anchor target (Q2). §0 favours the anchor.**
- A one-pass window at the drifted far state asks for stillness *there*. It slows ω but leaves neutral drift; freeze32 shows that holds only at the reduced margin.
- V_{n+2} is a verified high-margin fixed point in both lineages (freeze: flat 14.6–14.7).
- **Fix:** use r_anchor = ‖V_{T_far} − sg V_{n+2}‖² / ‖sg V_{n+2}‖² as the single penalty. V_{n+2} is free from the no-grad prefix, and n is known at train time only.
  - It adds a restoring force toward a state that §0 shows holds.
  - Keep exactly one penalty form (single-change rule).
  - K2 gets an anchor known-answer.
  - The guards are unchanged. STALL becomes more important (see 10).

**6. MEDIUM: the OC numbers.**
- P(pass | effect) uses N(0.97, 0.03), which is not grounded in our data. Say so, and give the sensitivity at a mean of 0.95 (both seeds 0.25).
- DEPTH: FH and base are independent runs; the same seed is not a pairing, because the trajectories diverge at step 1. So the SD of the difference is √2 × 0.049 = 0.069, not 0.05:
  - P(Δ ≥ −0.10 | no harm) is 0.926 per seed and **0.86** for both (not 0.955);
  - a true −0.15 harm is missed per seed with P 0.23;
  - FH-RECIPE P(pass | effect, no harm) ≈ 0.56 × 0.95 × 0.86 = **0.46**.
- Recomputed and confirmed:
  - null H(64): mean 0.527, SD 0.1216; z 3.48, 2.5e-4;
  - HOLD-GAIN null: N(0.120, 0.083), P(≥ 0.30) = 0.015.
- **2 seeds (Q9):** each seed is its own lineage (ω differs 1.2×, slope / ω differs 1.3×, H(64) 0.48 vs 0.39). A 1 / 2 split is uninterpretable.
- **Fix:** pre-register a contingent s3 for FH and RD (+≈ $3.4), triggered only by a seed split on any verdict, and include it in the cap.

**7. MEDIUM: the plain control is valid, but G-FH-EQ is mis-specified.**
- Reusing D3-T s1 / s2 is fine: no new plain runs are needed, because a seed is not a pairing (6), and both bases are re-evaluated on the same card (G-FH0).
- Problems:
  - §2 says "first 200 log records" (20k steps at log_every 100); K5 says steps 1–200 (2 records).
  - Steps 1–200 are inside the c hold (c = 1e4 to 1000) and CurGate stage 0, so they exercise neither the anneal nor the gather.
  - `auto_micro` probes memory with `vb_loss`. If the far branch is in the probe, micro can change and nothing is bitwise. If it is not, real FH runs may OOM.
  - A cross-host 3090 / driver difference (the base ran on 595.58.03) can break bitwise equality with no code fault.
- **Fix:**
  - G-FH-EQ := steps 1–1200 (12 records, crossing the c hold) on the launch host; assert micro equal; assert that the weight-0 far loss still runs backward; guard against 0 × non-finite.
  - If not bitwise, first re-run the pre-FH commit on the same host. If that also differs, the host is the cause, and the control stands with a "non-bitwise host" note.
  - The plain re-run fallback ($1.1 per seed) applies only to a code-caused mismatch.
  - K4 (fp64 plain-path equivalence) remains the code gate.

**8. MEDIUM: the realised-$ estimate ignores the launch-bound loop.**
- "Host speed variance" (the DeltaMem loop is launch-bound): time scales with passes, not items.
- If T_far is drawn per item, the far rollout runs to max T_far over 16 items, ≈ 46 passes. The cost is then roughly +100–130 % (about 36 pass-equivalents per base step), not +35 %. That puts it at ≈ $2.3–2.5 per run and ≈ $13–15 in total, at the cap before s3.
- **Fix:**
  - One T_far per step, shared by the 16 items (mean 30.5).
  - Measure gtime on the 3090 at K7 and put the measured $ in the OC block.
  - Pre-register a stop if the measured cost is > $2.4 per run.
  - Spend order (fix 13): fine-tune pilot, then scratch.

**9. MEDIUM: the part_khorizon OOM fallback drops K′, which contradicts the censoring rule.**
- The code (vbank_diag.py `part_khorizon`) falls back to "the largest K′ ≥ 48 that fits" and drops the K′ above it. It has no bs fallback. The draft says an OOM'd K′ "is re-run at 16 … never dropped".
- On a 24 GB 3090, K′ 128 should fit (≈ 2 × the 3070's 6.6 GB at K′ 64), but the code must not silently drop the primary K′.
- **Fix:** add an eval_bs fallback with ks kept, and record it. K9 exercises it. K′ 128 missing = pending.

**10. MEDIUM: the guards. Most are exposed; one is broken as written.**
- **C1 floor:** "K′ 0 / 1 ≤ chance + 0.05" fails the bases themselves:
  - `acc_k["0"]["1"]` is 0.217 (s1) / 0.204 (s2);
  - `acc_k["1"]["1"]` is 0.997 / 1.0 (one pass solves n 1).
  - Fix: K′ 0 over n 1..16 and K′ 1 over n 2..16, each ≤ base + 0.05 per n.
- **STALL** (`stall.recovery`, stall at t 2–3, bases 1.0 / 0.9998) is the exposed test for the main Goodhart risk: a learned "still ⇒ stay still" absorbing rule firing in the computing phase. It is more important under the anchor form (5). Keep −0.02.
- **PROBE(4)** n 7–8 (bases 0.13 / 0.12) and **early finish** (0.204 / 0.174) are informative, and exposed to premature stillness.
- **NO-LOSS acc_n2** is saturated at 1.0. It can detect harm, but it is not a sensitive guard.
- **DEPTH (Q4):** C1 > C6, and AGE-PROJ's proj_out cost was −0.056. Fix: keep −0.10 as the FAIL line, and add a registered WARN at −0.05 on the two-seed mean that blocks a "recipe" claim from being carried into C1 work without a depth re-test.
- **G2 (gate)** is effectively inert: the gate is ≈ 1. The protection against an answer-independent attractor comes from L_far and the in-range L_ans, so say so.
- **G3 (`rms_v` ± 25 %)** is fine.

**11. MEDIUM: CurGate interaction.**
- K6 (isolation) is right. But L_far and L_FP change the plain answer_ce trajectory that CurGate reads (the §214 lesson). The bases advanced at 6000 (to n 5) and 9000 (to n 8) by the answer-CE rule in both seeds.
- The far sub-batch takes the first 16 plain items, so its n follows the stage. State this.
- **Fix:** register "advance steps per arm; |Δ| > 2k vs the same-seed base ⇒ CONFOUNDED flag on the HOLD / DEPTH contrast" (basin-entry rule).

**12. MEDIUM-LOW: missing launch hygiene.**
- **Off-host checkpoints:** the 48k runs take ≈ 7 h at ×1.45, so they need a periodic off-host mirror of the full state, and a tested resume that restores rng_far.
- Name a separate mirror directory per run.
- Add K11: resume bitwise at a 200-step boundary.

**13. LOW (recommendation, Q8): run the fine-tune pilot first, as Stage 1 of the same registration.**
- Arms: FH-FT and RD-FT × s1 / s2, 12k steps from D3-T model.pt, lr 3e-5 as FT-CTRL. Add a new FT-CTRL s2 ($0.4); FT-CTRL s1 exists.
- Comparator: FT-CTRL, not D3-T, because a plain fine-tune alone gains +0.11 at K′ 64.
- ≈ $2.5 with evals.
- Stage 2 (48k from scratch) runs only if FH-FT shows HOLD-GAIN vs FT-CTRL ≥ 0.30 at K′ 128 in both seeds. This is a staged pre-registration, not a problem → fix loop.
- Both stages ≈ $11–15. The base-recipe claim still needs Stage 2.

**14. LOW: Literature's suggested arms.**
- **(a) Damping:** a $0 inference arm was run (§0). It extends the hold from ≈ 64 to ≈ 96 but fails at 128.
  - A trained damping needs a learned idle-phase gate. That is a new architecture, not a single change vs RD, and it is weaker than a still state.
  - Not in v1. Record §0 as descriptive evidence. Literature's "rival lever" framing holds only for the oracle-phase version.
- **(b) Answer-only readout-stability arm:** RD already carries an answer-level far objective. Its correctness CE at T_far is stronger than readout stability.
  - A third training arm (+$3.4) would break the cap together with s3 and Stage 1.
  - Defer it, contingent on FH passing and RD failing. It would then separate settle from freeze, as in Viakhirev.
  - The §0 freeze / cur_freeze arms already show, at eval level, that a still state, not a frozen readout, is what holds.
- **Weight tying:** moot. The core blocks 6–9 are already shared across passes.
- **Learned halting / absorbing state:** FH trains an absorbing state implicitly. An explicit halting head needs a halting criterion, which is not simpler and runs against C6's step-free reading.

**15. LOW: wording and tags.**
- "H-NORM refuted [PROVEN, code]" is fine for U_t's rms, but V_t is not normalised: rms(V_64) / rms(V_0) is 1.69–1.91. Keep it [measured].
- C6 [PROVEN, Banach] is correct but unused. Mark it as context.
- S3 (the "held answer outlives the current state" reading): reword to the descriptive fact (`rec_ok_v` falls while the coda answer holds). Drop the bank-version causal clause per §0.
- S4 is NARROW per Literature.
- `t_cross_obs_median` is 65 (never crossed by 64) in 5 / 6 models, against a predicted 55–61. The linear model is pessimistic on the 48-item sample, so C2's agreement is weaker than stated (corr 0.04–0.75).

## 2. Answers to Q1–Q10 in one line each
- Q1: grad-ratio dose rule (fix 4).
- Q2: anchor (fix 5, §0).
- Q3: add the relative-gap growth reading as descriptive only.
- Q4: WARN at −0.05 (fix 10).
- Q5: T_far 48 is enough: freeze32 / freeze hold to 128 once still, and the cost matters more.
- Q6: not needed. The coda reads the current state (cur_freeze), and an R-decode gate adds a second route to Goodhart.
- Q7: survives (freeze_win).
- Q8: pilot first (fix 13).
- Q9: contingent s3 (fix 6).
- Q10: moot once in-range terms are dropped (fix 2).
