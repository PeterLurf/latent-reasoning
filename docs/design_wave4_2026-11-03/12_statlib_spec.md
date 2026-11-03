# 12 STATLIB: one stdlib statistics module for every scorer (spec + listing)

2026-11-03 17:21 EST. Wave-4 lead design (ST1, with ST3 / ST4 / ST5 parts). Read-only design: nothing here was computed from stored outputs. Every number below is a closed form or a number quoted from THEORY / the backlog. Cost: $0, CPU only.

## 1. Goal and backlog IDs

There is one module, `theory/statlib.py` (stdlib only, Python >= 3.8, safe on the hosts' 3.11). It replaces the private `wilson` / `near` copies in the scorers and adds the statistics that the standing rules now require:

| ID / rule | What statlib provides |
|---|---|
| ST1 STATLIB (backlog §13.3) | wilson, clopper_pearson (incomplete beta by continued fraction + bisection), margin_se, near, paired_diff, cluster_boot, seed_rate, n_items_for, conf_seq, sprt, e-values + combine, holm, bh, conj_label |
| ST4 PAIRED-ARMS | `join` on item `sig` (asserts equal answers) + `paired_diff` (McNemar, discordant SE) |
| ST3 SEQ-STOP | `conf_seq` (beta-binomial mixture, valid at every look), `cs_first_exclusion`, `sprt` / `sprt_stream` |
| ST5 MULTIPLICITY | `conj_label` (PASS / FAIL / NEAR-k), `holm`, `bh`, e-value pooling (`combine_log_e`) |
| ST2 / Note 269-1 | `seed_sigma_pairs`: the exact sigma_seed definition of read_s269_logs.py:9 and :84-95 |
| Note 269-2 (2), (3) | `cell(..., c2=True)`: the NEAR-SEED label; `seed_cell`: per-seed margins + mean ± SD |
| Audit 11-03 standing rule | Wilson 95% CI + NEAR on every binding cell, in one per-cell schema |
| OC-block rule (10-14) | `oc_pass(s, need, p)` = P(rule fires) at the panel's seed count |

## 2. Design

### 2.1 Per-cell schema (every binding cell in every new scorer's JSON)

```
{"value": 0.857, "n": 1000, "ci95": [lo, hi], "bar": 0.84, "margin_se": 0.0116, "label": "NEAR-SEED",
 "raw": "PASS", "margin": 0.017, "dir": ">=", "near": false, "near_seed": true, "c2": true, "near_def": "score-z2+bar_se0"}
```

The required fields are `{value, n, ci95, bar, margin_se, label}`. The extra fields keep the letter apart from the claim:
- **`raw`** is the registered letter: `value >= bar`, or `value <= bar`.
  - Verdict strings are built from `raw` only ("score on raw values").
  - Nothing in statlib changes a registered bar (Note 269-2 (1)).
- **`label`** is the claim the write-up may make. Precedence:
  1. **NEAR-SEED** if `c2` and |value - bar| < sigma_seed (Note 269-2 (3); strict "<"; sigma_seed defaults to 0.090);
  2. else **NEAR** if `near()` holds;
  3. else **PASS** / **FAIL** = raw.

  NEAR and NEAR-SEED are "claimed neither way".
- **`c2`** is set with `is_c2(d)` (d >= 6; Note 269-1: C2 claims live at d >= 6). It is fixed by depth, never chosen per cell.

### 2.2 Equations

- **Wilson** (z 1.959964; clamped; n = 0 gives (0, 1)): (p + z²/2n ± z·sqrt(pq/n + z²/4n²)) / (1 + z²/n).
- **Clopper-Pearson.**
  - lo = B⁻¹(α/2; k, n-k+1); hi = B⁻¹(1-α/2; k+1, n-k).
  - B⁻¹ is found by bisection on I_x(a, b) (Lentz continued fraction, with symmetry above (a+1)/(a+b+2)).
  - `cp_lower` at k = n equals α^{1/n}: 0.22 / 0.37 / 0.55 / 0.72 for 2/2, 3/3, 5/5, 9/9 seeds, the backlog §13.3 numbers.
- **margin_se** = sqrt(bar(1-bar)/n + bar_se²), the score SE at the bar.
- **near** ⇔ |p - bar| ≤ 2·margin_se.
  - With bar_se = 0 this is exactly "the bar lies inside Wilson(z 2)": the Wilson interval inverts the score test, so NEAR and the CI never disagree.
  - The legacy `near_wald` uses 2·sqrt(max(pq, 1e-12)/n) at the *reading* (score_dd.py:55-56, score_tx.py:40-42, score_em.py:52-53, score_x7.py:38-39).
  - That degenerates at p ∈ {0, 1}: 1.000 vs 0.997 at n 1000 is not NEAR (fixture).
- **paired_diff** (0/1 per item, aligned by `join` on `sig`):
  - diff = (n10 - n01)/n;
  - SE = sqrt(n10 + n01 - (n10-n01)²/n)/n;
  - exact McNemar p = min(1, 2·P(Bin(n10+n01, ½) ≤ min(n10, n01))).
- **n_items_for.**
  - Unpaired (Fleiss): (z_α·sqrt(2p̄q̄) + z_β·sqrt(p1q1 + p2q2))²/δ².
  - Paired (Connor): (z_α·sqrt(disc) + z_β·sqrt(disc - δ²))²/δ².
  - These reproduce the backlog's ≈ 1,900 unpaired / ≈ 1,000 paired items for a 0.02 bar.
  - `n_seeds_for` = ⌈arms·((z_α + z_β)σ/δ)²⌉, a normal-approximation floor.
- **seed_sigma_pairs:** σ² = mean((p1-p2)²)/2 - mean((p1q1 + p2q2)/2n), floored at 0 (= read_s269_logs.py). The NEAR-SEED width stays 0.090 unless Audit re-rules.
- **seed_cell** (Note 269-2 (2)):
  - value = the mean over S seeds;
  - ci95 = mean ± t_{.975,S-1}·SD/√S (t from the same incomplete beta);
  - the per-seed margins are listed;
  - NEAR if the bar lies inside ci95.
  - **Proposal (Audit ack):** NEAR-SEED on a mean uses σ/√S.
- **cluster_boot** (seeds, then items within each drawn seed; percentile; equal seed weight) is descriptive only (§8, risk 5).
- **seed_rate** = k/s with a CP CI. **oc_pass(s, need, p)** = Σ_{j ≥ need} C(s,j) p^j q^{s-j}.
- **conf_seq.**
  - C_t = {θ : M_t(θ) < 1/α}, with M_t = B(a+k, b+t-k)/B(a,b) / (θ^k (1-θ)^{t-k}).
  - M_t(θ*) is a nonnegative martingale (M_0 = 1), so Ville gives P(∃t : θ* ∉ C_t) ≤ α, and every look is valid.
  - log M is convex with its minimum ≤ 0 at k/t, so each bound comes from one-sided bisection.
  - At k = t (a = b = 1): lo = (α/(t+1))^{1/t}.
- **sprt:** llr = k·log(p1/p0) + (n-k)·log(q1/q0). H1 if llr ≥ log((1-β)/α); H0 if llr ≤ log(β/(1-α)).
- **e-values.**
  - `log_e_lr` (p1 > p0) is valid for H0 θ ≤ p0, since E_θ[LR] = θp1/p0 + (1-θ)q1/q0 ≤ 1 there.
  - `log_e_mix` = [B(a+k, b+n-k)/B(a,b)]·[1 - I_{p0}(a+k, b+n-k)]/[1 - I_{p0}(a,b)]/[p0^k q0^{n-k}], a Beta prior truncated to (p0, 1), with E_{p0}[e] = 1 exactly.
  - `combine_log_e`: "product" for independent seeds, "mean" under any dependence. A product over seeds tests the intersection null ("every seed ≤ bar"), not "the recipe passes".
- **holm** (step-down running max of (m-r)·p_(r)); **bh** (step-up running min of p_(r)·m/r).
- **conj_label:**
  - FAIL if any cell is FAIL;
  - PASS if all are PASS;
  - else NEAR-k (k = the NEAR / NEAR-SEED count).

  `conj_raw` is the letter; `conj_tag` gives e.g. "PASS [NEAR-1]".

## 3. Code touch points (all read)

| File:lines | Current | Finding |
|---|---|---|
| theory/score_dd.py:51-56 | wilson(p, n), near (Wald) | n = 0 → ZeroDivisionError. **:90 hard-codes n 1000** (also in Audit SCORER_BUGS, dd LOW). The d 7 C2 row (:80) has no CI / NEAR. **No NEAR-SEED, although Note 269-2 (3) names §265** (new) |
| theory/score_tx.py:33-42 | wilson(p, n) with an n-guard, near | Bars `bun - 0.10` (:62, :66) are estimated; bar_se is ignored |
| theory/score_em.py:48-53 | wilson(p, n), near | n = 0 crash; reads n from eval_em (good, :88, :102) |
| theory/score_x7.py:34-39 (§268; moved from score_ls.py after Audit LS-1; re-read 17:25) | wilson(p, n), near | **:79-80 hard-code n 1000**; C2 at :74. Audit LS-4 already asks for a seed-level NEAR; whether Note 269-2 (3) covers §268 is Audit's call |
| theory/score_dg.py:42-45 | **wilson(k, n)**: a different signature | Same expression with p = k/n, so `wilson_p(k/n, n)` is bitwise-equal |
| theory/score_sc.py | no CI | Labels are thresholds on medians or accuracy deltas; leave as is (descriptive §264 schedule labels) |
| theory/score_of.py:42, :48 (§263) | C2 = d 6, d 7 ≥ bun - 0.05 | **No Wilson, no NEAR, no NEAR-SEED**, although the 11-03 standing rule and Note 269-2 (3) both name §263 (new; not in SCORER_BUGS) |
| theory/c1_acq_score.py:18, score_a01.py:25 | wilson(k, n) (z 1.96 / 1.959964) | Two more copies; archived sections, leave |
| boxes/inject_aw_eval.py:99-114 (`items_rec`, `part_acc`) | eval_acc.json `by_d[d]` = {acc, k_main, acc_kprime} with **no n**; `items[d]` = {sig, ans, pred[N][K+1], L, s} | n must be len(items[d]["ans"]). The per-item `pred` allows `join` + `paired_diff` with no new eval |
| boxes/inject_ow.py:52 | TEST_N = 1000 | Why the hard-coded 1000 is right today |
| theory/read_s269_logs.py:9, 84-95 | sigma_seed definition | Mirrored by `seed_sigma_pairs` |

**Adoption path (drop-in, three phases):**
- **P0 ($0).** Land theory/statlib.py exactly as listed in §10.
  - Audit G-D.
  - `--fixtures` must pass on 3.11 (host) and 3.14 (local).
  - `--legacy-check` must report 0 mismatches. It imports score_dd / tx / em / x7 / dg and compares their wilson / near with `wilson_p` / `near_wald` by `==` on a grid of 6 n × ~50 k × 7 bars.
- **P1 (registered §263 / §265; letters frozen).** Additive only, pre-stated and acked, and committed *before* of_s* / dd_s* outputs land:
  - in score_of.seed and score_dd.full_seed, add `"c2_cells": {d: SL.cell(acc_d, n_d, bar_d, ">=", c2=SL.is_c2(d))}` for d 6 / 7 (score_of) and d 7 / 8 (score_dd), with n_d = len(items[d]["ans"]);
  - append `" [C2 NEAR-SEED]"` to the seed's label when any c2 cell is NEAR-SEED.
  - The binding `crit` dict is untouched.
  - The local wilson / near copies stay as they are. Swapping them for the legacy shims gains nothing and adds diff risk.
- **P2 (§270+ registrations, template v7).** Scorers `import statlib as SL` and build every binding cell with `SL.cell` / `SL.seed_cell`.
  - The verdict is `conj_raw`, with `conj_tag` appended.
  - JSON gets `"cells": {name: cell}`.
  - n comes from items.
  - Arms on the same TEST2 items report `paired_diff` vs the reference (by sig).
  - Each registration states z_s, sigma_seed, the OC block (`oc_pass`) and any sequential rule (`conf_seq` α, look schedule).

```python
import statlib as SL                                    # theory/ is already on sys.path in score_dd / score_of
it = jl(os.path.join(runs, run, "eval_acc.json"))["items"]["7"]; n7 = len(it["ans"])
rit = jl(os.path.join(runs, ref, "eval_acc.json"))["items"]["7"]; nr7 = len(rit["ans"])   # same-seed reference (bun_sX)
c7 = SL.cell(a["7"]["acc"], n7, r["7"]["acc"] - 0.02, ">=", c2=SL.is_c2(7), bar_se=SL.margin_se(r["7"]["acc"], nr7, r["7"]["acc"]))
pd = SL.paired_diff(*SL.join(it, rit, ka=8, kb=8)); pd["scope"] = "checkpoint"   # same TEST2 items, pred[:, K' 8]; never a recipe claim at d >= 6
crit["C2-EXT"] = SL.conj_raw([c7, c8]) == "PASS"; tag = SL.conj_tag([c7, c8])
```

## 4. Gates

| Gate | Pass condition |
|---|---|
| G1 fixtures | `python -B theory/statlib.py --fixtures` prints 60/60 (closed forms: §6) on 3.11 and 3.14 |
| G2 legacy bitwise | `--legacy-check` reports 0 mismatches (the p = 0 equivalence analogue: the frozen shims equal the five live copies). Pre-verified in memory 2026-11-03 ~17:35 against the live score_dd / tx / em / x7 / dg: 0 mismatches |
| G3 P1 invariance | before/after the P1 edit, each of score_of / score_dd `--fixtures` prints byte-identical `FIXTURE` lines, apart from the appended tag; `crit` dicts are equal on every fixture world |
| G4 3.11 API | grep the module for `binomialvariate`, `batched`, `sumprod`, `isjunction`: none (memory: host-python-311) |
| G5 determinism | `cluster_boot(..., seed=0)` returns identical dicts twice; `--fixtures` runs in < 15 s |
| G6 field paths | for each P1 field read (`items[d]["ans"]`, `pred`, `sig`), grep the writer (inject_aw_eval.py:99-114) (memory: scorer-field-path-check) |

There is no gradflow or mem_check: no model code is touched. There are no new training flags, so the p = 0 equivalence rule maps to G2 / G3.

## 5. Criteria (pre-registration-ready), OC block, censoring

| Criterion | Type | Threshold |
|---|---|---|
| G1 fixtures | binding | 60/60 within each fixture's stated tolerance (1e-12 to 1e-6); measured 0.4 s on 3.14 in-memory |
| G2 legacy | binding | 0 mismatches by `==` |
| G3 invariance | binding | identical `crit` on all fixture worlds of score_of and score_dd |
| NEAR-SEED frequency on §263 / §265 C2 rows | descriptive | reported as the count per seed |
| bootstrap vs t-interval width ratio (S = 2-3) | descriptive | reported only |

**OC block (closed form, seed noise σ = 0.090, Note 269-1):**
- **NEAR-SEED rate of a C2 row** whose true margin is m: P(|m + σZ| < σ) = Φ(1 - m/σ) - Φ(-1 - m/σ).
  - m = 0: 0.683.
  - m = 0.05 (a recipe exactly equal to bun under §263's "- 0.05" bar): Φ(0.444) - Φ(-1.556) ≈ 0.612.
  - m = 0.10: ≈ 0.43.
  - **Consequence:** §263's C2 retention rows will mostly read NEAR-SEED at one seed per row. This is by design; the binding letter is unchanged.
- **Per-seed pass probability** at margin m: Φ(m/σ).
  - "3/3 seeds pass" fires with Φ(m/σ)³: 0.125 at m = 0 and ≈ 0.65 at m = 0.10 (`oc_pass(3, 3, 0.867)`).
  - At 2/2 seeds it is 0.25 under no effect, which is why ≥ 3 seeds are required.
- **Seeds for a mean margin δ = 0.10 vs a fixed bar** (α 0.05, power 0.8): ⌈(2.80·0.9)²⌉ = 7. Against a second panel: 13 per arm. Smaller panels detect only δ ≳ 0.15-0.2 at d 6.

**Censoring.** A cell on a time-to-event (halt, onset) counts never-firing / capped items as failures in k, with n = all items. The event-only rate goes beside it as a descriptive `event_only` field (memory: censored-items-in-criteria). `conf_seq` / `sprt` stopped early record `n_used` and the look count. A sequential read that never crosses a boundary reports CONTINUE, never PASS.

## 6. Scorer sketch

- **Fields read:**
  - statlib reads no files.
  - P1 / P2 scorers read `eval_acc.json` `by_d[d].acc` and `items[d]` {sig, ans, pred[N][K'+1]} (inject_aw_eval.py:99-114), for both the run and its same-seed reference;
  - P2 adds `n` from items.
- **Labels:**
  - per cell: PASS / FAIL / NEAR / NEAR-SEED;
  - per conjunction: PASS / FAIL / NEAR-k, beside the raw letter.
- **Fixtures (60, all closed forms):**
  - incomplete beta: I_x(1,1) = x, x^a, 1 - (1-x)^b, I_½(a,a) = ½, the binomial-tail identity, the symmetry branch;
  - CP: CP k/k = 0.05^{1/k} (k 2/3/5/9, two- and one-sided), CP 0/n;
  - Wilson: Wilson 0/n = z²/(n+z²), the mirror, Wilson ⇔ near on a ~20k-point grid, the near_wald p = 1 degeneracy;
  - t quantiles: t_{.975,1} = tan(.475π), t_{.975,2} = 4.303;
  - paired / seeds: McNemar 5/0 = 1/16 with the SE, 3/3 = 1; join permuted + mismatch raise; seed_sigma_pairs = .18/√2;
  - OC and sample size: oc_pass(3,3,½) = ⅛; n_items_for at power ½ = z²·2p̄q̄/δ², plus the backlog's ~1,900 / ~1,000; n_seeds_for = ⌈z²⌉;
  - sequential: CS lower (α/(t+1))^{1/t}, the CS mirror, CS first exclusion t = 8; SPRT 5/5 CONTINUE, 6/6 H1, 0/2 H0;
  - e-values: e_lr = 1.8, e_mix = 1.5, E_{p0}[e_mix] = 1 exactly, < 1 below p0, mean-combine = 2;
  - multiplicity: Holm and BH adjusted p on {.01, .04, .03, .005};
  - cells: cell / seed_cell / conj_label cases; the margin = σ edge; n 0 raises; cluster_boot reproducible and degenerate.
- **P1 fixtures to add in score_of / score_dd:**
  - a C2 row at bun + 0.03 must read raw PASS and label NEAR-SEED, with `crit` unchanged;
  - a row at bun + 0.12 must read PASS / PASS.

## 7. Cost

$0. CPU seconds locally and on hosts. There are no GPU, model loads or stored-output reads. P1 edits are about 15 lines per scorer.

## 8. RED TEAM

**Disclosure.** The required separate red-team sub-agent could not be spawned: the Agent tool returned "concurrent subagent limit reached (20); do not retry" twice. This is the lead's own adversarial pass, backed by in-memory probes of the listing on synthetic edge cases (n 0, k 0 / n, n 1e6, α 1e-12, Beta(5e5, 5e5), S 1, identical seeds, bar > 1, ties, empty lists). **Audit should run an independent red team before G-D.**

Fixed after the probes:
- (a) `cell(n=0)` returned FAIL; it now raises (an empty cell is pending, never FAIL).
- (b) A margin of exactly σ (0.84 vs 0.75) read NEAR-SEED through float error. Fixed by the `EDGE` guard, with a fixture.
- (c) `combine_log_e([])` and `n_items_for(p, p)` now raise named errors.
- (d) `near_def` is recorded on every cell.
- (e) `seeds_ok` (S ≥ 3) is recorded on C2 seed cells.
- (f) `is_c2(d)` fixes C2 tagging by depth.

Open risks (ranked):
1. **Item-level tests answer checkpoint questions, not recipe ones** (σ_seed ≈ 9× the item SE at d ≥ 6).
   - McNemar / conf_seq / sprt / e-values never back a C2 recipe claim. `seed_cell` over per-seed margins does.
   - paired_diff at d ≥ 6 carries `scope: checkpoint`.
2. **Gaming knobs** (zs, bar_se, sigma_seed).
   - Fix them at zs 2, σ 0.090, and bar_se ∈ {0, margin_se(ref, n_ref, ref)/√(#ref seeds)}, stated in the registration. Anything else fails G-D.
   - Cheaper: make zs and σ module constants only.
3. **Two NEAR definitions coexist.** §263-§269 keep the Wald one (frozen) and §270+ use the score one. `near_def` shows which.
4. **seed_cell at S = 3 is strict.** t_{.975,2}·SD/√3 ≈ 2.5 SD (≈ 0.22 at SD 0.09), so most 3-seed C2 means read NEAR.
   - This goes in each C2 OC block.
   - The σ/√S proposal is then usually moot; Audit may strike it.
5. **cluster_boot at S = 2-3 is not an interval.** Seeds of 0/1 give [0, 1], and S = 2 has 3 distinct resamples. It stays descriptive; cheaper to cut it.
6. **conf_seq / sprt need exchangeable order.** TEST2's per-d order was not checked. Shuffle with a logged seed and record `n_used`.
7. **P1 timing.** If of_s* outputs land before the P1 edit is acked and committed, the NEAR-SEED block is post-hoc descriptive and must say so.
8. **G2 binds to today's scorer source.** A later local fix (e.g. an n = 0 guard) fails G2 by design. Re-baseline only with Audit.
9. **holm / bh families must be listed in the registration.** BH assumes PRDS, which is plausible for one model's depth cells but unproven. Holm is the binding default.

## 9. Verified arXiv ids (abs pages fetched 2026-11-03; titles match)

| id | Title (first author, year) | Use |
|---|---|---|
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (Miller 2024) | clustered SEs, paired differences, power for evals |
| 2108.13264 | Deep Reinforcement Learning at the Edge of the Statistical Precipice (Agarwal 2021) | few-seed intervals; stratified bootstrap caveats |
| 1806.08295 | How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments (Colas 2018) | n_seeds_for |
| 1904.06979 | A Hitchhiker's Guide to Statistical Comparisons of Reinforcement Learning Algorithms (Colas 2019) | seed-level test choice at small S |
| 2103.03098 | Accounting for Variance in Machine Learning Benchmarks (Bouthillier 2021) | seed vs data variance (ST2) |
| 2304.01315 | Empirical Design in Reinforcement Learning (Patterson 2023) | seeds-then-items design |
| 1606.04316 | Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis (Benavoli 2016) | paired comparison alternatives |
| 1810.08240 | Time-uniform, nonparametric, nonasymptotic confidence sequences (Howard 2018) | conjugate-mixture CS (conf_seq) |
| 2010.09686 | Estimating means of bounded random variables by betting (Waudby-Smith 2020) | tighter CS alternative (not implemented) |
| 1906.07801 | Safe Testing (Grünwald 2019) | e-values / optional stopping |
| 1912.06116 | E-values: Calibration, combination, and applications (Vovk 2019) | product / mean combination |
| 2210.01948 | Game-theoretic statistics and safe anytime-valid inference (Ramdas 2022) | Ville, e-processes |
| 2402.14992 | tinyBenchmarks: evaluating LLMs with fewer examples (Maia Polo 2024) | sequential / small-n evals context |

Wilson (1927), Clopper-Pearson (1934), McNemar (1947), Wald (1945), Holm (1979), BH (1995), Fleiss and Connor (1987) predate arXiv.

## 10. Listing: theory/statlib.py

```python
"""theory/statlib.py: stdlib-only statistics for Latent scorers (backlog ST1 / ST3 / ST4 / ST5; spec docs/design_wave4_2026-11-03/12_statlib_spec.md).
Pure functions; no numpy / scipy; Python >= 3.8 (hosts run 3.11: no 3.12+ APIs).
Per-cell schema {value, n, ci95, bar, margin_se, label} (+ raw, margin, dir, near, near_seed, c2): verdicts use raw (the registered letter);
label is the claim (PASS / FAIL / NEAR / NEAR-SEED; Audit 11-03 standing rule + Note 269-2 (3)).
    python -B theory/statlib.py --fixtures       closed-form self-test (exit 0 iff all pass)
    python -B theory/statlib.py --legacy-check   bitwise check of wilson_p / near_wald vs the copies in score_{dd,tx,em,x7,dg}.py
"""
import argparse, json, math, os, random, sys
from statistics import NormalDist

Z95 = 1.959963984540054            # NormalDist().inv_cdf(0.975)
NEAR_Z = 2.0                       # Audit standing rule 11-03: NEAR = the bar within 2 SE
EDGE = 1e-12                       # float guard: a margin of exactly sigma_seed (k / 1000 readings) is NOT NEAR-SEED ("less than")
SIGMA_SEED = {4: 0.011, 6: 0.090}  # Note 269-1 ST2 (21 s1 / s2 pairs; read_s269_logs.py)
SIGMA_SEED_C2 = 0.090              # Note 269-2 (3): NEAR-SEED width for every C2 row (d >= 6)
_N = NormalDist()


def zq(q):
    return _N.inv_cdf(q)


# ------------------------------------------------------------------ special functions
def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a, b, x, itmax=100000, eps=3e-16):
    """Continued fraction for I_x(a, b) (modified Lentz; Numerical Recipes 6.4)."""
    tiny = 1e-300; qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap; d = 1.0 / (d if abs(d) > tiny else tiny); h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
            de = d * c; h *= de
        if abs(de - 1.0) < eps:
            return h
    raise ArithmeticError("betacf: no convergence (a=%g b=%g x=%g)" % (a, b, x))


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), a, b > 0; symmetry I_x(a, b) = 1 - I_1-x(b, a) above the mean."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lf = a * math.log(x) + b * math.log1p(-x) - lbeta(a, b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lf) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lf) * _betacf(b, a, 1.0 - x) / b


def bisect(f, lo, hi, tol=1e-14, it=400):
    """Root of f, increasing on (lo, hi); f is only evaluated at interior midpoints."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def betainv(q, a, b):
    return bisect(lambda x: betainc(a, b, x) - q, 0.0, 1.0)


def t_cdf(t, nu):
    tail = 0.5 * betainc(nu / 2.0, 0.5, nu / (nu + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_ppf(q, nu):
    return bisect(lambda t: t_cdf(t, nu) - q, -1e7, 1e7, tol=1e-13)


# ------------------------------------------------------------------ binomial intervals, margins, NEAR
def wilson(k, n, z=Z95):
    """Wilson score interval for k successes of n, clamped to [0, 1]; n 0 -> (0, 1)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n; dn = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / dn; h = z * math.sqrt(max(p * (1.0 - p), 0.0) / n + z * z / (4.0 * n * n)) / dn
    return (max(0.0, c - h), min(1.0, c + h))


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided 1 - alpha interval: lo = B^-1(alpha/2; k, n-k+1), hi = B^-1(1-alpha/2; k+1, n-k)."""
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k <= 0 else betainv(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k >= n else betainv(1.0 - alpha / 2.0, k + 1, n - k)
    return (lo, hi)


def cp_lower(k, n, alpha=0.05):
    """One-sided 1 - alpha Clopper-Pearson lower bound (k = n: alpha ** (1 / n))."""
    return 0.0 if k <= 0 else betainv(alpha, k, n - k + 1)


def wilson_p(p, n, z=1.96):
    """LEGACY, frozen: the exact expression of wilson(p, n) in score_dd / tx / em / x7 (and score_dg's wilson(k, n) with p = k / n):
    z 1.96, float p, no clamp. Only the n 0 guard is added (score_dd / em / x7 raise ZeroDivisionError there)."""
    if not n:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return (c - h, c + h)


def near_wald(p, n, bar):
    """LEGACY, frozen: near() of score_dd / tx / em / x7, |p - bar| <= 2 sqrt(max(p(1-p), 1e-12) / n) at the READING.
    Degenerate at p in {0, 1}. New scorers use near()."""
    return bool(n) and abs(p - bar) <= 2 * math.sqrt(max(p * (1 - p), 1e-12) / n)


def margin_se(p, n, bar, bar_se=0.0):
    """SE of the margin p - bar at the bar (score-test variance bar(1 - bar) / n) plus an estimated bar's own SE.
    With bar_se 0: |p - bar| <= z margin_se  <=>  bar inside wilson(k, n, z)."""
    if n <= 0:
        return float("inf")
    return math.sqrt(max(bar * (1.0 - bar), 0.0) / n + bar_se * bar_se)


def near(p, n, bar, zs=NEAR_Z, bar_se=0.0):
    return n > 0 and abs(p - bar) <= zs * margin_se(p, n, bar, bar_se)


# ------------------------------------------------------------------ paired items
def _key(s):
    return json.dumps(s, sort_keys=True)


def correct(items, k=None):
    """0/1 per item from an items block {sig, ans, pred}; pred[i] is a scalar or a per-K' list (index k, required then)."""
    if k is None and items["pred"] and isinstance(items["pred"][0], list):
        raise ValueError("correct: pred holds per-K' lists; pass k (e.g. the run's think_k)")
    return [int((p[k] if isinstance(p, list) else p) == y) for p, y in zip(items["pred"], items["ans"])]


def join(items_a, items_b, ka=None, kb=None, key="sig"):
    """Align two items blocks on key (unique keys, equal answers asserted); returns (xa, xb) 0/1 lists over the common items."""
    sa = [_key(s) for s in items_a[key]]; sb = [_key(s) for s in items_b[key]]
    if len(set(sa)) != len(sa) or len(set(sb)) != len(sb):
        raise ValueError("join: duplicate %s" % key)
    ib = {s: j for j, s in enumerate(sb)}; ca, cb = correct(items_a, ka), correct(items_b, kb); xa, xb = [], []
    for i, s in enumerate(sa):
        if s in ib:
            j = ib[s]
            if items_a["ans"][i] != items_b["ans"][j]:
                raise ValueError("join: answer mismatch at %s" % s)
            xa.append(ca[i]); xb.append(cb[j])
    return xa, xb


def paired_diff(xa, xb, z=Z95):
    """Paired difference on the same items: diff = (n10 - n01) / n, SE = sqrt(n10 + n01 - (n10 - n01)^2 / n) / n,
    exact two-sided McNemar p = min(1, 2 P(Bin(n10 + n01, 1/2) <= min(n10, n01)))."""
    n = len(xa)
    if n == 0 or n != len(xb):
        raise ValueError("paired_diff: need equal non-empty vectors")
    n10 = sum(1 for a, b in zip(xa, xb) if a and not b); n01 = sum(1 for a, b in zip(xa, xb) if b and not a); m = n10 + n01
    diff = (n10 - n01) / n; se = math.sqrt(max(m - (n10 - n01) ** 2 / n, 0.0)) / n
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(n10, n01) + 1)) / 2 ** m)
    return {"n": n, "p_a": sum(xa) / n, "p_b": sum(xb) / n, "diff": diff, "n10": n10, "n01": n01, "se": se,
            "ci95": [diff - z * se, diff + z * se], "mcnemar_p": p}


# ------------------------------------------------------------------ seeds
def seed_sigma_pairs(pairs):
    """Note 269-1 definition (read_s269_logs.py): pairs [(p1, p2, n)];
    sigma^2 = mean((p1 - p2)^2) / 2 - mean((p1 q1 + p2 q2) / (2 n)), floored at 0."""
    if not pairs:
        return None
    s2 = sum((a - b) ** 2 for a, b, _ in pairs) / (2 * len(pairs)) - sum((a * (1 - a) + b * (1 - b)) / (2 * n) for a, b, n in pairs) / len(pairs)
    return math.sqrt(max(0.0, s2))


def seed_rate(k, s, alpha=0.05):
    return {"k": k, "s": s, "rate": k / s if s else None, "ci95": list(clopper_pearson(k, s, alpha)), "lo1": cp_lower(k, s, alpha)}


def oc_pass(s, need, p):
    """OC block: P(Bin(s, p) >= need), the chance a 'need of s seeds pass' rule fires when each seed passes w.p. p."""
    return sum(math.comb(s, j) * p ** j * (1 - p) ** (s - j) for j in range(need, s + 1))


def n_items_for(p1, p2, alpha=0.05, power=0.80, disc=None):
    """Items per arm, two-sided alpha. Unpaired (Fleiss, no continuity correction) or paired (disc = P(discordant), Connor 1987)."""
    za, zb, dl = zq(1 - alpha / 2), zq(power), abs(p1 - p2)
    if dl == 0:
        raise ValueError("n_items_for: p1 == p2")
    if disc is None:
        pb = (p1 + p2) / 2; v = (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    else:
        v = (za * math.sqrt(disc) + zb * math.sqrt(max(disc - dl * dl, 0.0))) ** 2
    return math.ceil(v / (dl * dl) - 1e-9)


def n_seeds_for(sigma, delta, alpha=0.05, power=0.80, arms=1):
    """Seeds per arm for a mean margin delta under seed SD sigma (normal approx; a floor: t > z at small S)."""
    return math.ceil(arms * ((zq(1 - alpha / 2) + zq(power)) * sigma / delta) ** 2 - 1e-9)


def _quant(xs, q):
    h = (len(xs) - 1) * q; i = int(math.floor(h)); j = min(i + 1, len(xs) - 1); return xs[i] + (h - i) * (xs[j] - xs[i])


def cluster_boot(table, B=2000, alpha=0.05, seed=0):
    """Two-stage percentile bootstrap (seeds, then items within each drawn seed) of the equal-seed-weight mean.
    table = {seed_id: [per-item values]}. DESCRIPTIVE: at S < 5 the seed stage has few distinct resamples (S = 2: 3) and the
    interval is anti-conservative; binding seed-level intervals come from seed_cell."""
    keys = sorted(table); S = len(keys); rg = random.Random(seed)
    est = sum(sum(table[s]) / len(table[s]) for s in keys) / S; reps = []
    for _ in range(B):
        tot = 0.0
        for _ in range(S):
            v = table[keys[rg.randrange(S)]]; L = len(v)
            tot += sum(v[rg.randrange(L)] for _ in range(L)) / L
        reps.append(tot / S)
    reps.sort()
    return {"est": est, "lo": _quant(reps, alpha / 2), "hi": _quant(reps, 1 - alpha / 2), "B": B, "S": S, "seed": seed}


# ------------------------------------------------------------------ sequential: confidence sequence, SPRT
def _logM(k, t, th, a, b):
    return lbeta(a + k, b + t - k) - lbeta(a, b) - k * math.log(th) - (t - k) * math.log1p(-th)


def conf_seq(k, t, alpha=0.05, a=1.0, b=1.0):
    """Beta-binomial mixture confidence sequence after t i.i.d. Bernoulli draws with k successes:
    C_t = {theta : M_t(theta) < 1/alpha}, M_t = B(a+k, b+t-k) / B(a, b) / (theta^k (1-theta)^(t-k)).
    Ville: P(exists t, theta* not in C_t) <= alpha, so every look is valid. Items must arrive in a pre-stated random order."""
    if t <= 0:
        return (0.0, 1.0)
    L = math.log(1.0 / alpha); th = k / t
    lo = 0.0 if k == 0 else bisect(lambda x: L - _logM(k, t, x, a, b), 0.0, th)
    hi = 1.0 if k == t else bisect(lambda x: _logM(k, t, x, a, b) - L, th, 1.0)
    return (lo, hi)


def cs_first_exclusion(xs, bar, alpha=0.05, a=1.0, b=1.0):
    """Running-intersection CS over a 0/1 stream; first t at which bar leaves it ('above': lo > bar; 'below': hi < bar)."""
    k, lo, hi = 0, 0.0, 1.0
    for t, x in enumerate(xs, 1):
        k += int(x); l_, h_ = conf_seq(k, t, alpha, a, b); lo, hi = max(lo, l_), min(hi, h_)
        if lo > bar or hi < bar:
            return {"t": t, "side": "above" if lo > bar else "below", "lo": lo, "hi": hi, "k": k}
    return {"t": None, "side": "CONTINUE", "lo": lo, "hi": hi, "k": k, "n_used": len(xs)}


def sprt(k, n, p0, p1, alpha=0.05, beta=0.20):
    """Wald SPRT, H0 p = p0 vs H1 p = p1: llr = k log(p1/p0) + (n-k) log((1-p1)/(1-p0));
    H1 if llr >= log((1-beta)/alpha), H0 if llr <= log(beta/(1-alpha)), else CONTINUE."""
    llr = k * math.log(p1 / p0) + (n - k) * math.log((1 - p1) / (1 - p0))
    A, B_ = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    return {"llr": llr, "A": A, "B": B_, "decision": "H1" if llr >= A else "H0" if llr <= B_ else "CONTINUE"}


def sprt_stream(xs, p0, p1, alpha=0.05, beta=0.20):
    k = 0
    for t, x in enumerate(xs, 1):
        k += int(x); r = sprt(k, t, p0, p1, alpha, beta)
        if r["decision"] != "CONTINUE":
            return dict(r, n=t, k=k)
    return dict(sprt(k, len(xs), p0, p1, alpha, beta), n=len(xs), k=k)


# ------------------------------------------------------------------ e-values
def log_e_lr(k, n, p0, p1):
    """log likelihood-ratio e-value, p1 > p0; valid for the composite H0 theta <= p0."""
    if not p1 > p0:
        raise ValueError("log_e_lr: needs p1 > p0")
    return k * math.log(p1 / p0) + (n - k) * math.log((1 - p1) / (1 - p0))


def log_e_mix(k, n, p0, a=1.0, b=1.0):
    """log e-value for H0 theta <= p0, Beta(a, b) prior truncated to (p0, 1):
    e = [B(a+k, b+n-k) / B(a, b)] [1 - I_p0(a+k, b+n-k)] / [1 - I_p0(a, b)] / [p0^k (1-p0)^(n-k)]."""
    up = betainc(b + n - k, a + k, 1.0 - p0); pm = betainc(b, a, 1.0 - p0)
    if up <= 0.0:
        return float("-inf")
    return lbeta(a + k, b + n - k) - lbeta(a, b) + math.log(up) - math.log(pm) - k * math.log(p0) - (n - k) * math.log1p(-p0)


def combine_log_e(log_es, mode="product"):
    """product: independent e-values (separate seeds) -> sum of logs. mean: arbitrary dependence -> log of the average.
    A product over seeds tests the INTERSECTION null (every seed <= bar); it is not a recipe-level pass."""
    if not log_es:
        raise ValueError("combine_log_e: no e-values")
    if mode == "product":
        return sum(log_es)
    m = max(log_es)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(x - m) for x in log_es)) - math.log(len(log_es))


def e_reject(log_e, alpha=0.05):
    return log_e >= math.log(1.0 / alpha)


# ------------------------------------------------------------------ multiplicity
def holm(ps, alpha=0.05):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return {"adj": adj, "reject": [x <= alpha for x in adj]}


def bh(ps, q=0.05):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i], reverse=True); adj = [0.0] * m; run = 1.0
    for r, i in enumerate(order):
        run = min(run, ps[i] * m / (m - r)); adj[i] = run
    return {"adj": adj, "reject": [x <= q for x in adj]}


# ------------------------------------------------------------------ cells and labels
def _lab(raw, nr, ns):
    return "NEAR-SEED" if ns else "NEAR" if nr else ("PASS" if raw else "FAIL")


def cell(value, n, bar, direction=">=", c2=False, zs=NEAR_Z, bar_se=0.0, sigma_seed=SIGMA_SEED_C2, k=None):
    """One binding proportion cell. raw = the registered letter; label: NEAR-SEED if c2 and |value - bar| < sigma_seed
    (Note 269-2 (3)), else NEAR if near(), else raw. k (successes) defaults to round(value n)."""
    if n <= 0:
        raise ValueError("cell: n = 0 (an empty cell is pending / VOID, never FAIL)")
    kk = round(value * n) if k is None else k
    raw = value >= bar if direction == ">=" else value <= bar
    nr = near(value, n, bar, zs, bar_se); ns = bool(c2) and abs(value - bar) < sigma_seed - EDGE
    return {"value": value, "n": n, "ci95": list(wilson(kk, n)), "bar": bar, "margin_se": margin_se(value, n, bar, bar_se),
            "label": _lab(raw, nr, ns), "raw": "PASS" if raw else "FAIL", "margin": (value - bar) if direction == ">=" else (bar - value),
            "dir": direction, "near": nr, "near_seed": ns, "c2": bool(c2), "near_def": "score-z%g+bar_se%g" % (zs, bar_se)}


def is_c2(d):
    """C2 rows are fixed by depth (Note 269-1: C2 claims live at d >= 6), never hand-picked per cell."""
    return int(d) >= 6


def seed_cell(values, bar, direction=">=", c2=False, sigma_seed=SIGMA_SEED_C2, alpha=0.05):
    """Seed-level cell (Note 269-2 (2)): value = mean of per-seed readings, n = seeds, ci95 = Student t on S - 1 df,
    margin_se = SD / sqrt(S), per-seed margins listed. NEAR if the bar is inside ci95; NEAR-SEED (c2) if
    |mean - bar| < sigma_seed / sqrt(S) (proposal; needs Audit ack)."""
    S = len(values); m = sum(values) / S
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (S - 1)) if S > 1 else None
    se = sd / math.sqrt(S) if sd is not None else None
    ci = None if se is None else (m - t_ppf(1 - alpha / 2, S - 1) * se, m + t_ppf(1 - alpha / 2, S - 1) * se)
    raw = m >= bar if direction == ">=" else m <= bar
    nr = ci is not None and ci[0] <= bar <= ci[1]; ns = bool(c2) and abs(m - bar) < sigma_seed / math.sqrt(S) - EDGE
    return {"value": m, "n": S, "ci95": list(ci) if ci else None, "bar": bar, "margin_se": se, "label": _lab(raw, nr, ns),
            "raw": "PASS" if raw else "FAIL", "margin": (m - bar) if direction == ">=" else (bar - m), "dir": direction, "near": nr,
            "near_seed": ns, "c2": bool(c2), "seeds_ok": (S >= 3) if c2 else None, "sd": sd, "seed_margins": [(v - bar) if direction == ">=" else (bar - v) for v in values]}


def conj_label(cells):
    """FAIL if any label FAIL; PASS if all PASS; else NEAR-k (k = number of NEAR / NEAR-SEED cells)."""
    labs = [c["label"] for c in cells]
    if "FAIL" in labs:
        return "FAIL"
    k = sum(l in ("NEAR", "NEAR-SEED") for l in labs)
    return "PASS" if k == 0 else "NEAR-%d" % k


def conj_raw(cells):
    return "PASS" if all(c["raw"] == "PASS" for c in cells) else "FAIL"


def conj_tag(cells):
    """The registered letter plus the claim tag when they differ, e.g. 'PASS [NEAR-1]'."""
    r, l = conj_raw(cells), conj_label(cells)
    return r if r == l else "%s [%s]" % (r, l)


# ------------------------------------------------------------------ self-tests
def fixtures():
    res = []

    def chk(name, got, want, tol=1e-9):
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        res.append((name, bool(ok), got, want))
    chk("I_x(1,1) = x", betainc(1, 1, 0.3), 0.3)
    chk("I_x(a,1) = x^a", betainc(3.5, 1, 0.7), 0.7 ** 3.5)
    chk("I_x(1,b) = 1-(1-x)^b", betainc(1, 4.2, 0.2), 1 - 0.8 ** 4.2)
    chk("I_.5(a,a) = .5", betainc(7.3, 7.3, 0.5), 0.5)
    chk("binomial tail identity", betainc(7, 14, 0.3), sum(math.comb(20, j) * 0.3 ** j * 0.7 ** (20 - j) for j in range(7, 21)), 1e-12)
    chk("symmetry branch", betainc(2, 3, 0.9), 1 - betainc(3, 2, 0.1), 1e-12)
    for k in (2, 3, 5, 9):
        chk("CP %d/%d two-sided a.10 lower = .05^(1/k)" % (k, k), clopper_pearson(k, k, 0.10)[0], 0.05 ** (1 / k), 1e-10)
        chk("cp_lower %d/%d = .05^(1/k)" % (k, k), cp_lower(k, k, 0.05), 0.05 ** (1 / k), 1e-10)
    chk("CP 0/10 upper = 1 - .025^(1/10)", clopper_pearson(0, 10)[1], 1 - 0.025 ** 0.1, 1e-10)
    chk("Wilson 0/n upper = z^2/(n+z^2)", wilson(0, 40)[1], Z95 ** 2 / (40 + Z95 ** 2), 1e-12)
    chk("Wilson mirror", wilson(13, 50)[0], 1 - wilson(37, 50)[1], 1e-12)
    bad = 0
    for n in (50, 1000):
        for k in range(0, n + 1, max(1, n // 100)):
            lo, hi = wilson(k, n, NEAR_Z)
            for bar in [0.005 * i for i in range(1, 200)]:
                if min(abs(bar - lo), abs(bar - hi)) > 1e-9:
                    bad += near(k / n, n, bar) != (lo <= bar <= hi)
    chk("near <=> bar inside Wilson(z 2)", bad, 0)
    chk("near at p = 1 (score SE)", near(1.0, 1000, 0.997), True)
    chk("near_wald degenerate at p = 1", near_wald(1.0, 1000, 0.997), False)
    chk("t .975, 1 df = tan(.475 pi)", t_ppf(0.975, 1), math.tan(0.475 * math.pi), 1e-6)
    chk("t .975, 2 df closed form", t_ppf(0.975, 2), math.sqrt(2 * 0.95 ** 2 / (1 - 0.95 ** 2)), 1e-8)
    chk("t_cdf(0) = .5", t_cdf(0.0, 5), 0.5)
    pdx = paired_diff([1] * 5 + [1] * 50 + [0] * 45, [0] * 5 + [1] * 50 + [0] * 45)
    chk("McNemar 5/0 p = 1/16", pdx["mcnemar_p"], 0.0625)
    chk("paired SE closed form", pdx["se"], math.sqrt(5 - 25 / 100) / 100, 1e-12)
    chk("McNemar 3/3 p = 1", paired_diff([1, 1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1, 1])["mcnemar_p"], 1.0)
    A = {"sig": [[1, 2], [3, 4], [5, 6]], "ans": [1, 2, 3], "pred": [[0, 1], [0, 2], [0, 0]]}
    Bx = {"sig": [[5, 6], [1, 2], [9, 9]], "ans": [3, 1, 0], "pred": [[3, 3], [1, 0], [0, 0]]}
    chk("join on sig (permuted, partial)", join(A, Bx, 1, 0), ([1, 0], [1, 1]))
    try:
        join(A, {"sig": [[1, 2]], "ans": [7], "pred": [[7, 7]]}, 1, 0); chk("join answer mismatch raises", False, True)
    except ValueError:
        chk("join answer mismatch raises", True, True)
    chk("seed_sigma_pairs = .18/sqrt 2", seed_sigma_pairs([(0.59, 0.41, 10 ** 12), (0.41, 0.59, 10 ** 12)]), 0.18 / math.sqrt(2), 1e-6)
    chk("oc_pass(3,3,.5) = 1/8", oc_pass(3, 3, 0.5), 0.125)
    za = zq(0.975)
    chk("n_items_for at power .5 = za^2 2 pq / d^2", n_items_for(0.96, 0.94, power=0.5), math.ceil(za * za * 2 * 0.95 * 0.05 / 0.0004 - 1e-9))
    chk("n_items_for unpaired ~1900 (backlog)", 1800 <= n_items_for(0.96, 0.94) <= 1950, True)
    chk("n_items_for paired 5% ~1000 (backlog)", 950 <= n_items_for(0.96, 0.94, disc=0.05) <= 1000, True)
    chk("n_seeds_for sigma = delta, power .5", n_seeds_for(0.09, 0.09, power=0.5), 4)
    al, t = 0.05, 20
    chk("CS k = t lower = (a/(t+1))^(1/t)", conf_seq(t, t, al)[0], (al / (t + 1)) ** (1 / t), 1e-10)
    chk("CS k = t upper = 1", conf_seq(t, t, al)[1], 1.0)
    chk("CS mirror", conf_seq(7, 30)[0], 1 - conf_seq(23, 30)[1], 1e-10)
    tstar = next(s for s in range(1, 100) if (al / (s + 1)) ** (1 / s) > 0.5)
    chk("CS first exclusion of .5 on all-ones (closed form t)", cs_first_exclusion([1] * 50, 0.5)["t"], tstar)
    chk("SPRT 5/5 continue", sprt(5, 5, 0.5, 0.9, 0.05, 0.05)["decision"], "CONTINUE")
    chk("SPRT 6/6 H1", sprt(6, 6, 0.5, 0.9, 0.05, 0.05)["decision"], "H1")
    chk("SPRT 0/2 H0", sprt(0, 2, 0.5, 0.9, 0.05, 0.05)["decision"], "H0")
    chk("e_lr 1/1 = 1.8", math.exp(log_e_lr(1, 1, 0.5, 0.9)), 1.8, 1e-12)
    chk("e_mix 1/1 p0 .5 = 1.5", math.exp(log_e_mix(1, 1, 0.5)), 1.5, 1e-12)
    nn, p0 = 12, 0.3
    chk("E_p0[e_mix] = 1", sum(math.comb(nn, j) * p0 ** j * (1 - p0) ** (nn - j) * math.exp(log_e_mix(j, nn, p0)) for j in range(nn + 1)), 1.0, 1e-9)
    chk("E_.2[e_mix(p0 .3)] < 1", sum(math.comb(nn, j) * 0.2 ** j * 0.8 ** (nn - j) * math.exp(log_e_mix(j, nn, p0)) for j in range(nn + 1)) < 1, True)
    chk("mean-combine {1, 3} = 2", math.exp(combine_log_e([0.0, math.log(3)], "mean")), 2.0, 1e-12)
    ps = [0.01, 0.04, 0.03, 0.005]
    chk("Holm adjusted", max(abs(x - y) for x, y in zip(holm(ps)["adj"], [0.03, 0.06, 0.06, 0.02])), 0.0, 1e-12)
    chk("BH adjusted", max(abs(x - y) for x, y in zip(bh(ps)["adj"], [0.02, 0.04, 0.04, 0.02])), 0.0, 1e-12)
    chk("cell PASS", cell(0.86, 1000, 0.75, c2=True)["label"], "PASS")
    chk("cell NEAR-SEED keeps raw letter", (cell(0.80, 1000, 0.75, c2=True)["label"], cell(0.80, 1000, 0.75, c2=True)["raw"]), ("NEAR-SEED", "PASS"))
    chk("cell margin exactly sigma_seed is not NEAR-SEED", cell(0.84, 1000, 0.75, c2=True)["label"], "PASS")
    try:
        cell(0.0, 0, 0.5); chk("cell n 0 raises", False, True)
    except ValueError:
        chk("cell n 0 raises", True, True)
    chk("cell NEAR (in range, not C2)", cell(0.975, 1000, 0.98)["label"], "NEAR")
    chk("cell FAIL (<= bar)", cell(0.35, 1000, 0.20, "<=")["label"], "FAIL")
    sc = seed_cell([0.8, 0.9, 1.0], 0.7)
    chk("seed_cell t interval", sc["ci95"][0], 0.9 - math.sqrt(2 * 0.95 ** 2 / (1 - 0.95 ** 2)) * 0.1 / math.sqrt(3), 1e-8)
    chk("seed_cell NEAR (bar inside t CI)", sc["label"], "NEAR")
    P, Nr, F = {"label": "PASS", "raw": "PASS"}, {"label": "NEAR", "raw": "FAIL"}, {"label": "FAIL", "raw": "FAIL"}
    chk("conj PASS / NEAR-2 / FAIL", (conj_label([P, P]), conj_label([P, Nr, dict(Nr, label="NEAR-SEED")]), conj_label([F, Nr])), ("PASS", "NEAR-2", "FAIL"))
    chk("conj_tag", conj_tag([P, Nr]), "FAIL [NEAR-1]")
    T = {"s1": [1, 0, 1, 1, 0] * 20, "s2": [1, 1, 1, 0, 1] * 20, "s3": [0, 1, 1, 1, 1] * 20}
    chk("cluster_boot reproducible", cluster_boot(T, B=200, seed=3) == cluster_boot(T, B=200, seed=3), True)
    cb = cluster_boot({"a": [1] * 30, "b": [1] * 40}, B=100)
    chk("cluster_boot degenerate on constants", (cb["lo"], cb["hi"], cb["est"]), (1.0, 1.0, 1.0))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "->", r[2] if r[1] else "%r (want %r)" % (r[2], r[3]))
    k = sum(r[1] for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


def legacy_check():
    """Bitwise: wilson_p / near_wald == the live copies in score_dd / tx / em / x7 (wilson(p, n), near) and score_dg (wilson(k, n))."""
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); bad = 0
    grid = [(k, n) for n in (1, 7, 50, 100, 500, 1000) for k in range(0, n + 1, max(1, n // 50))]
    bars = (0.2, 0.5, 0.75, 0.8, 0.9, 0.95, 0.98)
    for name in ("score_dd", "score_tx", "score_em", "score_x7"):
        M = importlib.import_module(name)
        for k, n in grid:
            bad += M.wilson(k / n, n) != wilson_p(k / n, n)
            bad += sum(M.near(k / n, n, b) != near_wald(k / n, n, b) for b in bars)
    G = importlib.import_module("score_dg")
    bad += sum(G.wilson(k, n) != wilson_p(k / n, n) for k, n in grid)
    print("legacy-check mismatches: %d" % bad); return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--legacy-check", action="store_true")
    a = ap.parse_args(); ok = True
    if a.fixtures:
        ok = fixtures() and ok
    if a.legacy_check:
        ok = legacy_check() and ok
    if not (a.fixtures or a.legacy_check):
        print(__doc__)
    sys.exit(0 if ok else 1)
```
