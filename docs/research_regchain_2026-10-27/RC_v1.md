# REG-CHAIN v1 (2026-10-27)

**Status:**
- v1 = RC_draft.md (4035dcde) with the red team's fixes 1–7 (REVIEW_RC.md); where they differ, this file wins.
- Chosen by Theory under the user's standing rules (relayed by Audit 10-26). Audit approves.
- Literature (LIT_STEPTOLD addendum) says to present this as evaluation hygiene, not a contribution: "To remove the pass-position shortcut a fixed final operation allowed (§234–235), each step's operation is drawn iid in training and test, as in standard synthetic multi-step arithmetic (e.g. Lin 2503.07604), so no pass index predicts its operation."
  - The op-position / step-merging finding is ours; kin are Kuo 2606.29983 and Liang & Pan 2602.00449.

## 1. REG law and build (fixes 1, 2, 4)
- OW.Stream, OW.test_set and OW.pair_candidates take `mul_s_p` ∈ {0, 1} (default 1 = current, bitwise). At 0 no program has the forced mul_s (no ts draw).
  - Assert mul_s_p3 = 1 whenever mul_s_p = 0.
- **Scope of independence:** over passes t ≥ 2 at d ≥ 2, I(op_t; t, d) ≈ 0.0003 bits (value-driven: mul is redrawn when S_{t−1} = 0). Pass 1 differs (no add_s at t 1; d 1 forced mul_v; I = 0.049 bits over all (t, d)).
  - The gated passes 2..d are clean. The build logs the (t, d) op table from 50k training draws.
- **Disclosed:**
  - REG is close to a Markov chain: only 7.5% (d 3) and 16.7% (d 4) of programs read a computed earlier state (add_s, j ≥ 1);
  - every REG program is affine, with no cross-state product. The status line must say "no cross-state product".
- **Streams:**
  - training: REG;
  - TEST2-REG, TRANSFER-REG, pairs / floors, swap_pairs_d and probe: REG with new, unused seeds (grep);
  - calibration (inject_aw.py:282) and memcheck (:208): REG (d ≤ 2 streams are identical under both laws).
- **Gates:**
  - flags-off bitwise at mul_s_p 1, including test_set / pair_candidates;
  - zero mul_s in every REG set;
  - 0 sig overlap between training and every eval set;
  - new set shas in args.json (asserted at eval);
  - inject_ow exports the REG test-set seed so the scorer regenerates TEST2-REG.
- **Runs (one RTX 3090 each; off-host ckpts every 1k):**
  - rc_plaw4_s1 and rc_plaw4_s2 (LAT-AW), rc_pcd4_s1 (CD-AN);
  - args diff vs aw_plaw4_s1 / aw_pcd4_s1 = {mul_s_p 0} + seed + name;
  - d ~ U{1..4}, 16k, γ = 1, askname init.
- **Eval-only references on the REG sets (fixes 5, 7):**
  - s228_pdir2c_s1 → s228_pdir2c_s1_reg (acc; the CUE-ONLY bar);
  - aw_plaw4_s1 / aw_plaw4_s2 / d4d_plaw4_s1 → *_reg (ws, acc, cue_only, swap_kd; descriptive H-POS baseline).
- **Eval change:** swap_kd at d 3 under REG at t 1 (binding, as before) and also at t 2 (descriptive; 1000/1584 pairs kept).

## 2. Derivation
- As the draft, with [A2] scoped as above.
- H-POS predicts per-op WS at pass 3 ≥ 0.8 on every op class.
- H-CAP predicts ≤ 0.5 on the ops that need reading.
- Pass 4 is still trained only at d 4 (25% of steps), and it failed in all three earlier runs, including mul_s items (WS 0.051 / 0.025 / 0.265). So "pass 4 lags" is a live world independent of the op prior.

## 3. Readings (theory/score_rc.py, fixtures before output)
- **Per LAT seed, binding:**
  - LP0 (both that seed's arm and CD-AN);
  - **ACC:** TEST2-REG d 3 ≥ 0.60 and d 4 ≥ 0.50, each ≥ CD-AN + 0.25;
  - **CHAIN:** WS ≥ 0.50 at every pass 2..d, d 3 and d 4;
  - **OP-ALL:** pass-3 WS pooled over d 3–4 ≥ 0.80 for every op class (6 regular ops + add_s j = 0 + add_s j ≥ 1). Any class with n < 50 → VOID;
  - **LP5-4:** pooled drop ≥ 0.30;
  - **SWAP-MID d 3, t 1:** follow_U ≥ 0.50 and ≥ floor + 0.30;
  - **CUE-ONLY guard** on d 3–4 (bar max(majority, s228_pdir2c_s1_reg d) + 0.05).
- **Seed labels, first match:**
  1. VOID (guard, sig mismatch or n < 50);
  2. PASS (all gates);
  3. CAP (OP-ALL < 0.80; "partial" if min ≥ 0.55, else "none");
  4. MERGED (at some pass t < d, d 3–4, WS of v_{t+1} ≥ 0.50);
  5. PASS4-LAG (CHAIN fails only at d 4 pass 4);
  6. CHAIN-OTHER (CHAIN fails otherwise);
  7. GATE-OTHER (ACC / LP5-4 / SWAP fails).
- **Overall:**
  - PASS: both seeds PASS;
  - SPLIT: exactly one PASS, the other FAIL or VOID;
  - FAIL: no PASS and at least one non-VOID;
  - VOID: both VOID.
- **Status mapping:**
  - PASS → "C1, scaffolded (plan given), d ≤ 4, 2 seeds, op-unconfounded, no cross-state product";
  - SPLIT → "1 of 2 seeds".
  - CAP in both seeds → the op-target aux design next (pre-named, not run here).
- **Descriptive:**
  - SWAP-MID d 3 t 2 and d 4 t 2;
  - d 5–6 accuracy and WS;
  - the substitution table per pass;
  - CUE-WRONG; the aux staircase;
  - the references on REG;
  - REG models on P-4 TEST2 d 1–2 (comparable) and d ≥ 3 (out of distribution).

## 4. OC (fix 6)
- **Per seed:** P(VOID) ≈ 0.15 (§235 precedent).
  - Given non-VOID: P(PASS | H-POS and pass 4 learns) ≈ 0.7; P(PASS | H-POS and pass 4 lags) ≈ 0.05; P(PASS | H-CAP) ≈ 0.01.
- **Priors:** H-POS 0.7 × pass 4 learns 0.5.
  - Per-seed P(PASS) ≈ 0.85 × (0.35 × 0.7 + 0.35 × 0.05) ≈ 0.22.
  - Overall P(PASS) ≈ 0.1; P(≥ SPLIT) ≈ 0.35.
- The informative split is CAP vs PASS4-LAG vs PASS.

## 5. Censoring
As P-4: a crash means resume (dedupe); missing parts mean pending, FAIL at --final. A missing reference makes the CUE-ONLY bar pending.

## 6. Cost
- Three training runs ≈ $1.5, plus four eval-only references ≈ $0.3 (or $0 on local cards if they fit memory). Admit ≤ $2.2, guard $6.8.
- Running: realised (§234 + §235 + seed 2, Experiments to confirm, ≈ $4.9) + $6.8 ≈ $11.7 ≤ $30.
