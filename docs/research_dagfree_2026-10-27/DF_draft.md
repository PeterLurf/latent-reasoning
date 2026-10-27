# DAG-FREE draft: C1 with cross-state products at position-independent passes (2026-10-27)

**Status:** draft for red team + Literature. Theory's choice under the user's standing rules (relayed by Audit 10-26); C1 first. Follows §237 PASS (Note 237-4).

## 1. Why
- §237 passes C1 on the REG law: affine steps, no cross-state product, near-Markov. Audit's scope (236-1 (1)) leaves the TOPO dag with state references open.
- The default law's forced mul_s fixes the product's position (d 3: always step 3), and that produced the position shortcuts (234-7, 234-9).
- A law where a cross-state product can occur at any pass t ≥ 3, with probability independent of t and d, tests product chaining without a position cue.

## 2. Law (one change vs §237: the program law)
- **DF law:** mul_s_p = "free" (new value; 0 / 1 unchanged, bitwise).
  - Per step t ≥ 3, with probability q = 1/8, the step is mul_s with partner j ~ U{1..t−2} among valid partners (s_j ≥ 2, s_{t−1} ≥ 2). If none is valid, fall back to a regular op.
  - Otherwise it is a regular op from OPS_REG as REG. Steps 1–2 are REG.
  - The coin is drawn per step, before the op draw.
- **Expected properties** (to verify by simulation in the red team): P(mul_s | t, d) ≈ q at every t ≥ 3 and d ≥ t (minus fallbacks). d 3 programs have 0 or 1 product; d 4 can have two.
- **Training:** warm start LAT from rcb_plaw4_s1 (seed 1) and rcb_plaw4_s2 (seed 2), which are two different models, though both trace to aw_plaw2_s1. CD-AN from rcb_pcd4_s1.
  - DF law, d ~ U{1..4}, 16k, γ = 1, as §237 otherwise; calibration skipped (warm start); fade restart as 237-2 (a).
- **Eval:** P-4 parts on DF sets with new seeds (TEST2-DF, pairs / floors / swap_pairs_d / probe), plus s228_pdir2c_s1 on the DF sets (CUE-ONLY bar).
  - Step-0 refs: the two inits on DF sets (237-1 (1) practice).
  - Exclusion and set-sha gates as §236.

## 3. Derivation (sketch)
- [A] §237's passes read the op and one state (the previous one or, by add_s, an older one) and compose them (OP-ALL 1.0, including add_s j ≥ 1).
- mul_s needs two states, S_{t−1} and S_j, combined multiplicatively. The DeltaMem read at pass t returns a superposition keyed by the cue, so the question is whether one read can carry both factors.
- H-EXT: the add_s j ≥ 1 skill (reading an older state) extends to a product with S_{t−1}, giving the mul_s class at pass 3 / 4 ≥ 0.8.
- H-READ2: a single read can't deliver two states for a product, giving mul_s ≤ 0.3 while the other classes stay ≈ 1.0.
- The labels separate these.

## 4. Readings (score_rc-style; binding, per seed; PASS = both)
- LP0; ACC (TEST2-DF d 3 ≥ 0.60, d 4 ≥ 0.50, each ≥ CD-AN + 0.25); CHAIN (WS ≥ 0.50 at every pass 2..d, d 3–4).
- **OP-ALL+:** pass-3 and pass-4 WS per op class, pooled over d 3–4 at pass 3 and d 4 at pass 4, ≥ 0.80 for every class with n ≥ 50 (mul_s included). A class with n < 50 → VOID.
- LP5-4; SWAP-MID d 3 t 1 and t 2 (t 2 binding now: it tests the product step directly); CUE-ONLY guard.
- **Labels:** VOID / PASS / READ2 (only mul_s < 0.8) / CAP (another class < 0.8) / MERGED / PASS4-LAG / CHAIN-OTHER / GATE-OTHER.
- Step-0 deltas are reported; the init has never seen a product.

## 5. OC
- Per seed P(PASS | H-EXT) ≈ 0.7; P(PASS | H-READ2) ≈ 0.02. Prior H-EXT ≈ 0.5.
- Per seed P(PASS) ≈ 0.36; overall P(PASS) ≈ 0.2.

## 6. Cost
- Three runs ≈ $2.5 (§237: $2.45 incl. bad hosts) plus local refs. Admit ≤ $2.6, guard $7.
- Running: $8.76 + $7 ≈ $15.8 ≤ $30.
