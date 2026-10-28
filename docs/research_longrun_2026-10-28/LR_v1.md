# LONG-RUN v1: C6 stability of the §238 latent loop over long latent runs (2026-10-28)

**Status:** Theory's choice under the user's standing rules (north star C1 > C6 > C4 = C2). Eval-only, $0 (local 4060). Registered before any K′ > 14 output exists.
- Known so far (Note 237-4, descriptive): accuracy holds to K′ 14 on §237's models, against a trained think_k of 8.
- No red-team subagent (eval-only, no training, usage rule); Audit reviews.

## 1. What C6 asks
"No drift or collapse over long latent runs; low seed-to-seed variance in the outcome" (docs/NORTH_STAR.md).
- After step d the hold passes (t > d) re-run the loop on the S_d cue. DeltaMem's additive state can grow with each pass (memory: the additive state grows), so drift or collapse at long K′ is a live failure mode, not an assumed pass.

## 2. Runs (eval-only, no training)
- **Models:** df_plaw4_s1, df_plaw4_s2 (§238; the binding pair) and rcb_plaw4_s1, rcb_plaw4_s2 (§237; descriptive).
- **Sets:** TEST2-DF d 2–4 for df_*; TEST2-REG d 2–4 for rcb_*; all items (1,000 per d).
- **K′ ∈ {8, 16, 32, 64}** (8 = the trained think_k). New eval option `--kprime_list 8,16,32,64` (default off = current KMAX 14 sweep, bitwise). It writes eval_long.json {by_d {d: {acc {K′}, s_norm {K′}: mean ||S_K′||, s_cos {K′}: mean cos(S_K′, S_8)}}, items {sig, ans, pred {K′}}}.
- fp32, the cloud runs' eval_bs where memory allows (else 100, batch-invariant per 236-4's check).

## 3. Readings (binding, per df seed; PASS = both)
- **HOLD-LONG:** for each d ∈ {2, 3, 4} and K′ ∈ {16, 32, 64}, acc(K′) ≥ acc(8) − 0.02.
- **SEED-VAR:** at every (d, K′), |acc_s1 − acc_s2| ≤ 0.05.
- **Labels:**
  - PASS;
  - DRIFT (some acc(K′) < acc(8) − 0.02 while the min stays ≥ 0.5);
  - COLLAPSE (some acc(K′) < 0.5);
  - SEED-SPLIT (HOLD-LONG passes in both seeds, SEED-VAR fails).
- **Descriptive:**
  - s_norm growth (s_norm(64) / s_norm(8)) and s_cos(64) per d;
  - the same table for rcb_*;
  - d 5 at K′ 8..64 (C2-adjacent, descriptive only).
- s1's §238 LP0 failure is a training-stability gate. It doesn't bear on this eval-only reading, and it is stated beside the result.

## 4. OC
- P(PASS | the hold is a fixed point of the loop) ≈ 0.9 (K′ 14 already holds at ≥ 0.999).
- P(PASS | additive growth eventually swamps the read) ≈ 0.05.
- Prior 0.6, so P(PASS) ≈ 0.56.

## 5. Censoring
A missing K′ or d entry means pending. Any non-finite value in s_norm means COLLAPSE.

## 6. Cost
$0 (local eval). The build is a small eval option plus a flags-off gate (default path bitwise vs the current eval on one model / d).
