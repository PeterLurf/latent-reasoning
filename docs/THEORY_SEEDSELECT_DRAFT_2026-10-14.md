# SEEDSELECT: seed stability by selection. A blind $0 retrospective of early training-curve detectors (draft, 2026-10-14 20:52 EDT)

**Math / methodology subagent for Theory. NOT registered; Audit reviews before any feature is computed.**
- **Inputs:** THEORY_SEEDSTAB Part II §§2–3, 5, 6, 8; THEORY_SEEDSTAB2 F4, Prop O; THEORY.md 123-3, 125-2, §130, §131, 131-9, 190-1, 199-T1, §203; LIT_EARLY_PREDICT_LOTUS_LOOPS (35b6a795).
- **Files (theory/seedselect/):**
  - `inventory.py` → `labels.json`, `inventory.json` (metadata and outcome labels);
  - `derive.py` → `derive.json` (theory numerics and label counts);
  - `seedselect.py`: the registered scorer. `--selftest` has run and passed. `--freeze` and `--run` are not run.
- **Blinding:** no early-window value of any run (loss, answer_ce, aux_ce, write_grad_norm before a cutoff, or any slope, spectrum or variance of them) was computed, plotted or printed for this draft.
  - What was read from logs: record counts, field names, step grids, and resume / duplicate markers.
  - Labels: basin_entry entry(0.5), entry(0.1) and last bin; 123-3 FLAG and 131 FLAG4 as booleans; eval long_acc.
  - Earlier exposures by other notes are listed in §1.3.
- **Tags:**
  - **PROVED:** a complete argument is given here, or in the cited section.
  - **DERIVED(A):** derived under the listed assumptions.
  - **MC:** checked by Monte Carlo (`derive.py`).
  - **DATA:** labels only.
  - **CONJ:** conjecture.

## 0. Summary

1. **Nested cannot test this alone.** One mixed within-recipe stratum (RMS s1 and s2 entered at 19k, s3 never) gives 2 pairs, best p = 1/3. Four 2-seed entry-time strata add up to 6 ordinal pairs, where p < 0.05 needs a perfect 6/6 (p = 1/48). RMS's 8k CE level was already seen (Part II §3.3).
2. **Primary panel R1:** fsq120 FINAL S5 2k escapers, outcome FLAG4 (§131's "escaped but did not learn" proxy, held-out precision 8/8). 28 runs in 2 card strata, 92 pairs, window [500, 2000].
3. **Secondary R2:** §123–§130's 2k plateau FLAG on 226 runs (7 strata, 2150 pairs), window [300, 900]. It is coupled to the outcome by construction.
4. **Frozen primary feature:** the OLS slope of ln(loss) over the window, with the pre-stated hare sign (faster early descent predicts failure; Part II C4), tested two-sided by permutation within strata.
5. **Verdicts:** USEFUL iff p < 0.05 and C ≥ 0.72 (REVERSED if C ≤ 0.28); NOT USEFUL iff the 95% CI lies inside (0.28, 0.72).
6. **Power:** R1 detects C = 0.80 with power 0.82 (0.40 at C = 0.70). R2 resolves C ≈ 0.62. Nested R3 and a 4-seed prospective check cannot reach p < 0.05 two-sided.
7. **Window (nested):** a detector ending at or before 2k is blind by construction (PROVED). The γ ramp has accrued 85% of the decision variance by 6k and 96% by 8k (DERIVED), so information does not limit a detector at 6–8k.
8. **Observability limits it instead:** at 8k the loss carries about 10⁻³ nats of sign information against about 10⁻¹ nats of spread in C's progress (DERIVED(A)). Predicted nested loss-only C is about 0.55 (the hare channel only).
9. **fp sensitivity (DATA, labels only):** the same fsq120 seed on another card class flips its 2k outcome in 35–48% of 40 seeds (independence: 44–50%). Tetrachoric ρ is 0.09–0.33, so at least about 70% of the outcome is card-specific.
10. **Cost (48k runs, p = 2/3):** "2 seeds to t1, keep the better" breaks even at AUC 0.75 (t1 = 8k) or 0.72 (7k). A perfect sequential detector costs 1.08 C; a method reaching p = 0.9 costs 1.11 C with no selection. For 24k runs, selection needs AUC 0.87–0.94 at t1 = 7k (p 0.6–2/3) and never pays at 8k for p = 2/3.
11. **Recommendation:** run the $0 retrospective (minutes of CPU); I expect NOT USEFUL for nested. Fund the 4-seed prospective check (about $9.5, a direction check) only if R1 reads USEFUL and R3's direction agrees.
12. **At no extra cost:** keep a full-state 8k checkpoint in every future nested run.

---

## 1. Samples and labels

### 1.1 Outcome definitions (adopted from each family's own sections)
- **Nested (DeltaMem, nested170):**
  - ENTERED = basin_entry `entry(0.5)` is not None, on a complete 48k log (the §187-pinned definition; deduped by step, last occurrence wins).
  - Ordinal outcome: the entry step, with "never" counted as ∞.
- **fsq120 S5 (§123–§131):**
  - **PLATEAU** = 123-3's FLAG: loss(2k) ≥ 0.9 · loss(1k) and loss(2k) ≥ 0.25 · loss(100), at the nearest logged steps. §123, §125, §127, §130 and §128's restart-on-plateau all used this.
  - **ESCAPE-FAIL** = §131's FLAG4: loss(4k) ≥ 0.9 · loss(2.6k), plus the floor clause, among runs with FLAG = false.
    - FLAG4 is a proxy for "escaped and did not learn" (long_acc < 0.99).
    - Held out: precision 8/8 (CI 0.63–1.0) and recall 8/11 (score_s131.json, PASS).
  - Where eval.json has long_acc, the true outcome (exact = long_acc ≥ 0.99) is used.

### 1.2 Samples (labels.json; S = success, F = failure)

| reading | recipe and outcome | harness, log interval | stratum | units | S / F | pairs | role |
|---|---|---|---|---|---|---|---|
| **R1** | fsq120 FINAL S5, 2k escapers; ESCAPE-FAIL | fsq120.py, 100 steps | RTX 3070: flag4proto 10 + rampproto_plain 5 | 15 | 5 / 10 | 50 | **primary** |
| | | | RTX 4060: flag4proto 9 + rampproto_plain 4 | 13 | 7 / 6 | 42 | |
| R1b | same recipe, 24k escapers with long_acc | fsq120.py | 3090 / 3090 Ti: s120 final s1, s128 s102, s104 \| s127 ctrl s3 | 4 | 3 / 1 | 3 | descriptive |
| | | | A4000: s130chk s12 \| s17, s34 | 3 | 1 / 2 | 2 | |
| | | | 2080 Ti: s131x s403 \| s401 (0.987), s402, s404 | 4 | 1 / 3 | 3 | |
| **R2** | FINAL S5; PLATEAU | replay_basin.py, 50 steps | replay130 4060 (§130 FIT + SAME) | 40 | 26 / 14 | 364 | secondary |
| | | | replay130 3070 (§130 CROSS) | 17 | 8 / 9 | 72 | |
| | | | replay_x 2070 (fp32) | 60 | 36 / 24 | 864 | |
| | | | replay_x 3070 | 40 | 27 / 13 | 351 | |
| | | | replay_x 4060 | 40 | 20 / 20 | 400 | |
| | | fsq120.py, 100 steps | flag4proto 3070 | 19 | 10 / 9 | 90 | |
| | | | flag4proto 4060 | 10 | 9 / 1 | 9 | |
| R3 | nested RMS (s185), ENTERED | nested170.py, 100 steps | RTX 3090: s1 (19k), s2 (19k) \| s3 (never) | 3 | 2 / 1 | 2 | descriptive |
| R3ord | nested; entry step | nested170.py | RMS as above; WINDOW s1 25k / s2 17k; REWEIGHT s1 42k / s2 20k; NOPASS+RW s1 19k / s2 28k; RMS×NOPASS s1 22k (resumed at 27k, after the window) / s2 26k | 11 | ordinal | 6 | descriptive |

- **Totals:**
  - mixed (informative) strata: R1 2, R1b 3, R2 7, nested binary 1;
  - pairs: R1 92, R2 2150, nested 2 binary + 6 ordinal.
- **Excluded, with reasons:**
  - **proto136_d8 (24 runs, 2070):** the same seeds as replay_x 2070, with FLAG agreeing 24/24. They are presumed duplicates (G2 tests this), so R2 keeps the 50-step replays.
  - **rampproto_plain, from R2 only:** these seeds were chosen because their same-card replay was not flagged, i.e. selected on the outcome. They stay in R1, which conditions on escape anyway.
  - **Known duplicates:** s128_s5_s1 = s120_s5_final_s1 and s128_s5_s3 = s127_s5_ctrl_s3 (131-8). s175_auxk8_s1_pathindep_2070 is a copy of s175_auxk8_s1's log.
  - **s198_injc_s1:** cache_lower makes it a numerics twin of rms s1, so it is not a new seed. Both entered (18k vs 19k), which is n = 1 of numerics-flip evidence.
  - **Other exclusions:** *.void*, timing and smoke dirs (≤ 200 steps), the 8k proto previews (no outcome), the 4060 Ti stratum (2 of 2 failed), and other families (multiplication, ProsQA, GSM8K), which have only single seeds per recipe.
  - **Strata without mixed binary outcomes** (WINDOW, REWEIGHT, NOPASS, RMS×NOPASS: 2/2 entered each; s170 DIRECT 0/2) contribute only to R3ord.

### 1.3 Prior exposure (blinding audit)

| panel | already seen by earlier notes | consequence |
|---|---|---|
| R3 nested | Part II §3.3: answer and aux CE over 7.5–8.5k for rms s1–s3, window s1 / s2, rmsnopass s1 / s2, nofade, den, hold. s3 was lowest on both. | level features are **unblinded**; slopes are partly informed. R3 is descriptive only. |
| R1b | 131-9: s12_a4000's loss at 2k (1.939, the deepest) and its jump at L_max 4. score_s131.json holds loss(4k)/loss(2.6k) ratios. | descriptive only |
| R2 | §130: the zb_std_in markers, PULL and COLLAPSE steps, FIT seeds' loss(1k) and loss(2k). 131-9: dmax at 2k on 20 replay_x replays. | loss shapes in [300, 900] were never computed. The probes, which correlate with the loss, were seen; this is disclosed. |
| **R1** | the ramp-prototype readings used loss(4k)/loss(2.6k), loss(7k)/loss(5.6k) and the 2k flag | **the pre-2k slope and spectrum are unread: the cleanest panel** |

### 1.4 Label-only facts used in §2 (DATA)
- **Same seed on three card classes** (replay_x, seeds 501–540; FINAL S5 2k replays):

| card pair | FLAG rates | discordant | expected if independent | tetrachoric ρ (bootstrap 95%) |
|---|---|---|---|---|
| 2070 (fp32) – 3070 | 0.325 / 0.325 | 0.350 | 0.439 | 0.33 (−0.25, 0.73) |
| 2070 – 4060 | 0.325 / 0.500 | 0.475 | 0.500 | 0.09 (−0.38, 0.54) |
| 3070 – 4060 | 0.325 / 0.500 | 0.425 | 0.500 | 0.26 (−0.31, 0.68) |

  - All three cards agree for 15 of the 40 seeds.
- **The same seed on a different card (s130chk vs replay130):** FLAG is discordant in 4 of 7 seeds: A4000 vs 3070 for s17 and s34; 4060 Ti vs 4060 for s5 and s7.
- **flag4proto, 3070 vs 4060 (9 seeds):** FLAG is discordant in 4 of 9, and FLAG ∨ FLAG4 in 4 of 9.

---

## 2. Why the window (derivation)

### 2.1 Nested (DeltaMem; write_mix 2000 → 6000, fade over 500 steps; SEEDSTAB2 F4)

**(a) Before 2k a detector is blind by construction (PROVED; Part II Lemma 1, SEEDSTAB2 Prop O, deltamem 278–280).**
- For γ = 0, u = LN_6(h6) does not depend on the read. The loop gain is G ≡ 0, LN_w gets zero gradient, and the selection variable m is frozen.
- In the reduced model Z_{2k} = 0 deterministically. Through the m channel, any feature whose window ends at or before 2k has AUC exactly 0.5.
- What such a window can see: C's open-loop progress (the hare channel, (d)) and data-order noise.

**(b) Information accrual is not binding at 6–8k (DERIVED(A1–A3), integrals in derive.json).**
- Assumptions:
  - A1: Part II C3's swept gain, a(t) = a_max (t − 2k)/4k on the ramp and a_max after it;
  - A2: 1/a_max = 3.3k steps, from Part II's entry-time calibration;
  - A3: constant σ.
- Define R(t) = ∫_{2k}^{t} e^{−2A}σ² / ∫_{2k}^{∞} e^{−2A}σ², the fraction of Var Z accrued by t:

| t | 4k | 6k | 7k | 8k | 10k |
|---|---|---|---|---|---|
| R (A1–A3) | 0.55 | 0.85 | 0.92 | 0.96 | 0.99 |
| R (σ ∝ γ on the ramp) | 0.12 | 0.58 | 0.77 | 0.88 | 0.96 |
| R (a_max halved) | 0.37 | 0.65 | 0.74 | 0.81 | 0.90 |
| R (a_max doubled) | 0.73 | 0.97 | 0.99 | 1.00 | 1.00 |

- **The ideal-observer ceiling (MC).** Suppose a feature measured the partial decision variable Z_t exactly. At z = 0.43 its AUC would be:
  - R = 0.1: 0.65;
  - R = 0.5: 0.84;
  - R = 0.85: 0.95;
  - R = 0.95: 0.985.
- So information does not limit a detector at t1 ≥ 6k.
- **The earliest justified cutoff** (R ≥ 0.9, central model) is about 6.6k.
- **Registered:** t1 = 8000 as primary (R ≥ 0.81 in every variant; also Part II's O1 window and P2 time) and t1 = 6000 as secondary (the end of the ramp).
- **Window start:** t0 = 2100, the first record whose block lies entirely in γ > 0.

**(c) Observability is the binding limit (DERIVED(A4–A6)).**
- Assumptions:
  - A4: the selection SDE is a noisy gradient flow on the loss, dm = −∂_m L dt + σ dW. Then L(m) = L_0 − b m − (a/2) m² + (c/4) m⁴.
  - A5: entry happens at commitment |m| = m_c, at T_e ≈ 19k for RMS, with 1/a = 3.3k and z = 0.43 (Part II §2).
  - A6: the loss drop through commitment is ΔL_c ≈ 2 nats: answer plus aux CE at 8k is about 2.3 (Part II §3.3 table), and the final bin is about 0.01 (labels).
- **The sign-informative part of the loss is L(m) − L(−m) = −2bm. The part −(a/2)m² is sign-blind.**
- Order of magnitude at 8k:
  - m(8k)/m_c ≈ e^{−(T_e − 8k)a} ≈ 0.036;
  - m_c/μ_Z ≈ e^{a(T_e − T_d)}/z ≈ 50/0.43, so the linear share of ΔL_c is 1/(1 + m_c/(2μ_Z)) ≈ 1/59;
  - sign-informative loss signal ≈ ΔL_c · (1/59) · 0.036 ≈ **1 × 10⁻³ nats**;
  - sign-blind part ≈ ΔL_c · 0.036² ≈ 3 × 10⁻³ nats.
- **Compare** the across-seed spread of C's progress at 8k: RMS answer CE 1.84–2.03 over 3 seeds, **about 10⁻¹ nats** (Part II table).
- The m channel's signal-to-noise ratio is therefore about 10⁻², and its contribution to the AUC is about +0.005.
- **Consistent with 199-T1 (DATA):** the state order parameter co-moves with the loss and does not detect at 8k in two entering runs.

**(d) The one usable loss channel at 8k is the hare (DERIVED(A), MC).**
- In C4 a faster C lowers both the early loss and z.
- Part II MC S2 (lognormal SD 0.5 in r_C, which is generous for within-recipe data-order noise) gives P(success) by C-speed tercile of 0.511 / 0.442 / 0.393, i.e. **AUC ≈ 0.55**.
- **Prediction for nested loss-only detectors at t1 = 8k:** C ≈ 0.55. That is below the 0.72–0.75 break-even (§5).
- This is why nested is not expected to pass. It is also why passing in R1 would be informative: there the schedule makes the decision visible in the loss early (§2.2).

**(e) fp sensitivity bounds what any pre-divergence feature can know (DATA §1.4, PROVED (i), DERIVED(A7) (ii)).**
- **(i) Ties are unavoidable (PROVED).** Two runs with the same seed have identical logged curves until their trajectories differ at logging precision. Any feature computed before that point is identical across a discordant pair, so the pair is a tie (contributes ½) for every detector.
- **(ii) The seed-fixed share of the outcome is small (DERIVED(A7)).**
  - A7: a threshold-normal latent outcome, Z = Z_seed + Z_card.
  - The replay_x flips give ρ = Var(Z_seed)/Var(Z) = 0.09–0.33. So the seed fixes about 10–33% of the 2k outcome; the rest is generated by the card-specific trajectory after divergence.
  - Ideal-observer ceiling for seed-determined information: AUC ≤ 0.63–0.77.
  - Hence the window must end **after** cross-card divergence. Gdiv, a registered descriptive, measures when that happens.
- **(iii) For nested** no cross-card pair exists. Part II P10 predicts a flip rate of about 2p(1 − p) ≈ 0.45 if decorrelation precedes the ramp (Λ ≳ 7×10⁻⁴ to 2×10⁻³ per step; Part II §3.2).
  - The implication for the protocol: anything that changes numerics between screening and continuation (card class, cache_lower, fast_add, a resume) re-draws the undecided part of the outcome (§6.1).

### 2.2 fsq120 (R1, R2)
- **Schedule** (fsq120.py l. 11–12, 518–519, 1676):
  - soft codes until 2000, then a soft-to-hard ramp over 2000–4000;
  - L_max 2 until 2000, 4 from 2001, 8 from 5001;
  - warmup 100.
- **R1 = [500, 2000]:**
  - It starts after warmup and the marginal fit, and covers the first pull attempts (about 650; 127-7, §130).
  - It ends at the last record before the L_max 2 → 4 step, and before FLAG4's reference points (2.6k and 4k). The outcome's measurement window is therefore disjoint from the detector's.
  - Conditioning on FLAG = false uses loss(1k) and loss(2k), which a deployed two-stage rule also has at 2k.
  - Cost fraction κ = 2k/24k = 0.083.
- **Why R1 can differ from nested.** At L ≤ 2, a short-chain shortcut and a state-carrying code both lower the loss. The L_max 4 step at 2.1k then reveals which one was learned (the 0.4–0.8 jump, §131). The schedule forces the decision into the loss early, so (c)'s observability limit need not apply.
  - The hare mapping (shortcut = C) is **CONJ**.
  - Counter-anecdote, disclosed: s12_a4000 was the deepest run at 2k and ended exact (131-9).
- **R2 = [300, 900]:** 900 is the last record that does not overlap the FLAG's 1k record.
  - **Coupling:** FLAG is a ratio to loss(1k). An early pull lowers that denominator, and §130 found the flag firing after early pulls (s17, s30). The window level is therefore mechanically linked to the outcome in both directions, which is why R2 is secondary.

---

## 3. Features (frozen; `seedselect.py` SPEC, sha256 recorded by `--freeze`)

- **Series.** y_k = ln v_k at the run's logged record steps s_k ∈ [t0, t1], deduped by step (the last occurrence wins). Every grid step must be present. u = s/1000.
- **Success score** = sign × feature, where a higher score predicts success.

| feature | definition | sign | basis |
|---|---|---|---|
| **slope:f** (primary on f = loss) | OLS slope of y on u | R1, R3: **+1** (hare: faster descent ⇒ failure); R2: −1 (descent ⇒ escape) | Part II C4, §3.3 (nested: DERIVED + DATA); fsq120: CONJ |
| level:f | mean of y over (t1 − 1000, t1] | as for slope | Part II O1; reference only (unblinded in R3) |
| lfa:f | e = y − Bβ̂, with B = {1, u, u²} (+ (u − 6)₊ in R3); E_j = DFT(e)_j; band J = {j ≥ 1 : T/j ≥ 500 steps}, T = NΔ; LFA = √((2/N²) Σ_{j∈J} \|E_j\|²) | +1 (the Notsawo direction); also read two-sided | Notsawo 2306.13253 (EMPIRICAL, across hyper-parameters) |
| lff:f | Σ_J \|E_j\|² / Σ_{j=1}^{⌊(N−1)/2⌋} \|E_j\|² | +1 | scale-free variant of lfa |
| vw:write_grad_norm (R3) | residual SD of ln write_grad_norm after B | −1 (stochastic collapse) | Part II §5: PROVED (scalar), DERIVED (network) |
| slope:aux_ce, slope:answer_ce, lfa:answer_ce, lfa:aux_ce (R3) | as above | +1 | Audit's list |
| composite (R3) | within-stratum rank mean of the signed slope:aux_ce and vw | +1 | uses no outcome |

- **Resolution.**
  - Logged values are means over the preceding interval Δ (100 steps; 50 for the replays). That is a boxcar filter (gain sinc(Δ/P): 0.94 at P = 500, Δ = 100) followed by decimation. The Nyquist period is 2Δ.

| window | N | T (steps) | Δf (per step) | LF band (periods) | residual dof |
|---|---|---|---|---|---|
| R1 [500, 2000] | 16 | 1600 | 1/1600 | j = 1–3 (1600, 800, 533) | 13 |
| R2, replays | 13 | 650 | 1/650 | j = 1 (650) | 10 |
| R2, flag4proto | 7 | 700 | — | — | 4: not resolvable |
| R3 [2100, 8000] | 60 | 6000 | 1/6000 | j = 1–12 (6000–500) | 56 |
| R3b [2100, 6000] | 40 | 4000 | 1/4000 | j = 1–8 | 37 |

  - The rule: lfa and lff are computed only if residual dof ≥ 8.
- **Why the slope is primary and LFA is not.**
  - In the reduced model, the selection mode's excursions are sign-blind.
  - Its slow time, 1/a ≈ 3.3k steps, exceeds the R1 and R2 windows, so within them it shows up as trend, and detrending removes it.
  - Notsawo predicts grokking **across hyper-parameters**, so using it per seed within a recipe is an extrapolation (Literature 35b6a795).
  - The hare slope has a mechanism (C4) and one post-hoc nested observation behind it.
- **ILDR (2604.20923)** is out of scope:
  - it needs penultimate-layer representations on held-out data at checkpoints, which these runs do not have;
  - it times an eventual transition and does not show separation of runs that never enter.
- **The prior stands:** a within-recipe, per-seed early detector is unproven.

---

## 4. Readings, thresholds, power

- **Statistic: pooled within-stratum concordance.**
  - C = Σ over pairs (i, j) in the same stratum with y_i > y_j of [1(x_i > x_j) + ½ · 1(x_i = x_j)], divided by the number of such pairs.
  - Binary outcomes compare success with failure. Ordinal outcomes compare an earlier entry with a later one (never = ∞).
- **Null:** outcomes are exchangeable within strata.
  - Exact enumeration when there are at most 2×10⁵ arrangements; otherwise 10⁵ Monte Carlo permutations, with p = (k + 1)/(n + 1).
  - 95% CI from a stratified bootstrap (4000 draws).
- **Primary: R1, slope:loss, two-sided, α = 0.05.**
  - **USEFUL:** p < 0.05 and C ≥ 0.72 (the pre-stated hare direction).
  - **USEFUL-REVERSED:** p < 0.05 and C ≤ 0.28. This falsifies the hare mapping for fsq120, and prospective use would need a new registration.
  - **NOT USEFUL (falsifier):** the 95% CI lies inside (0.28, 0.72).
  - **DETECTABLE BUT BELOW THE BAR:** p < 0.05 with 0.28 < C < 0.72.
  - Otherwise **INCONCLUSIVE**.
  - **Why the bar is 0.72:** for 48k nested runs at p = 2/3, the N = 2 parallel break-even is 0.718 at t1 = 7k (0.75 at 8k), and a sequential detector gains 5% at 0.70 (§5).
- **Secondary readings (descriptive):**
  - level, lfa and lff in each reading, Holm-adjusted;
  - R2 under the same verdict rules;
  - R1b;
  - R3 and R3b: one-sided exact p with the pre-stated signs;
  - **R2m:** a matched cross-card test. For replay_x seeds whose FLAG differs across cards, the feature is ranked within card; a sign test asks whether the unflagged card's run scores higher;
  - **Gdiv:** the cross-card divergence step.

**Power (derive.json; equal-variance binormal scores; permutation p):**

| reading | pairs | null SD of C | min attainable p | C detectable at 80% power | power at C = 0.6 / 0.7 / 0.8 |
|---|---|---|---|---|---|
| **R1 (primary)** | 92 | 0.117 | 2×10⁻⁷ | ≈ 0.83 | 0.14 / 0.40 / **0.82** |
| R2 | 2150 | 0.043 | ≪ 0.001 | ≈ 0.62 | 0.67 / 1.0 / 1.0 |
| R1b | 8 | 0.22 | 0.021 (one-sided) | — | one-sided: 0.16 / 0.29 / 0.54 at C = 0.7 / 0.8 / 0.9 |
| R3 binary | 2 | 0.41 | 1/3 | impossible | 0 |
| R3 ordinal | 6 | — | 1/48 at 6/6 (two-sided 0.042) | — | P(6/6) ≈ C⁶: 0.26 at 0.8 |
| prospective: 4 new + RMS s1–s3 (≈ 5 S / 2 F) | ≈ 10 | 0.26 | 0.048 one-sided, only with a perfect order | two-sided impossible | 0 |
| 12 new RMS seeds (≈ 10 S / 5 F) | ≈ 50 | 0.16 | 3×10⁻⁴ | ≈ 0.96 | 0.05 / 0.22 / 0.50 |

- **Power statement.**
  - The nested sample alone reaches α = 0.05 only with a perfect 6/6 ordinal ranking (power about 0.26 even at C = 0.8), so **R1 is primary**.
  - R1 certifies a useful detector (C ≥ 0.8 with power 0.82) but misses marginal ones (C = 0.7: 0.40).
  - R2 resolves C ≈ 0.62, but on a coupled outcome and a no-read failure mode.
- **Pre-stated predictions:**
  - **R3 (nested):** C(slope:loss) ≈ 0.55 (§2.1(d)).
  - **R1:** C > 0.5 under the hare mapping (CONJ); a reversal is allowed by the counter-anecdote.
  - **R2:** C(slope, progress sign) > 0.5 if early pulls predict escape, but the coupling can pull it toward 0.5.
  - **R2m:** above 0.5 only if the post-divergence trajectory carries the outcome.

---

## 5. Cost derivation

- **Sequential detector + restart (PROVED, Part II §6).** With κ = t1/C, TPR T and FPR F:

  E/C = [κ + (pT + (1 − p)F)(1 − κ)] / (pT).

- **When a detector pays (PROVED here).** It beats no detector (1/p) iff

  κ + (1 − p)F(1 − κ) < T[(1 − p) + pκ].

  - For an uninformative detector (T = F) the difference is κ(1 − T) ≥ 0, so it never pays.
  - Example, p = 2/3 and κ = 1/6: the condition is 0.444T − 0.278F > 0.167.
- **Parallel-N with in-process continuation (PROVED; P(success) DERIVED(A: binormal scores)).**
  - N seeds run to t1; the top-scored one continues; the rest are killed.
  - Cost per attempt = Nκ + (1 − κ).
  - P(success) = P(the top-scored of N succeeds).

| family, t1, p | κ | no detector | perfect (seq.) | seq. at AUC 0.7 / 0.8 / 0.9 | N = 2: cost per attempt; P(success) at AUC 0.5 / 0.8 / 1 | break-even AUC, N = 2 / N = 3 |
|---|---|---|---|---|---|---|
| nested 48k, 7k, 2/3 | 0.146 | 1.50 | 1.07 | 1.42 / 1.32 / 1.22 | 1.15; 0.67 / 0.80 / 0.89 | 0.72 / 0.81 |
| nested 48k, 8k, 2/3 | 0.167 | 1.50 | 1.08 | 1.43 / 1.33 / 1.23 | 1.17; 0.67 / 0.80 / 0.89 | **0.75** / 0.86 |
| nested 48k, 7k, 0.6 | 0.146 | 1.67 | **1.10** | 1.54 / 1.41 / 1.27 | 1.15; 0.60 / 0.74 / 0.84 | 0.68 / 0.75 |
| nested 48k, 8k, 0.6 | 0.167 | 1.67 | 1.11 | 1.55 / 1.43 / 1.30 | 1.17; 0.60 / 0.74 / 0.84 | 0.71 / 0.79 |
| GSM8K-style 24k, 7k, 2/3 | 0.292 | 1.50 | 1.15 | 1.47 / 1.40 / 1.31 | 1.29; 0.67 / 0.80 / 0.89 | 0.94 / never |
| GSM8K-style 24k, 8k, 0.6 | 0.333 | 1.67 | **1.22** | 1.63 / 1.55 / 1.43 | 1.33; 0.60 / 0.75 / 0.84 | 0.92 / never |

- **What a useful detector must look like.**
  - At t1 = 8k, p = 2/3, a sequential detector needs AUC ≥ 0.70 for a 5% gain and ≥ 0.78 for a 10% gain. Its optimal operating point at AUC 0.8 is about TPR 0.77, FPR 0.33.
  - GSM8K-style 24k runs gain little from selection at 7–8k: a 5% gain needs AUC ≥ 0.76 (p = 0.6) to 0.80 (p = 2/3). A shorter t1 would require an earlier write_mix ramp.
- **Reliability for registered k-seed claims.** P(3/3) is:
  - 0.30 at p = 2/3;
  - 0.51 with N = 2 screening at AUC 0.8 (each slot at p 0.80, costing 1.17 C per slot);
  - 0.70 with a perfect detector;
  - 0.73 for a method with p = 0.9, at 1.0 C and with no post-selection of seeds.
- **Why fp sensitivity bounds detectability before the decision window:** see §2.1(e). Features computed before divergence tie on discordant same-seed pairs. The seed-fixed share of the outcome is at most about 10–33% in the one family where it can be measured. A detector earns its cost only by reading the realised, card-specific trajectory late enough to see the decision.

---

## 6. Prospective check (only if R1 = USEFUL and R3's direction agrees)

### 6.1 Protocol
- **Runs.** 4 fresh seeds (4–7) with s185_rms_s1's args exactly, on the RTX 3090 class only.
- **Detector.** Fixed before launch: R3's SPEC, slope:loss over [2100, 8000] with the hare sign; the composite is secondary. t1 = 8000.
- **At t1 (Audit's addition).** The screened seeds are **continued, not restarted**, on the same host and card. Preferably the process is simply not stopped, so the continuation is bitwise the scored trajectory (same seed + same card reproduces record for record; 131-8).
- **Commitment.** The predictions are committed in a dated note before any record after step 8000 is read:
  - the ranking;
  - the N = 2 and N = 3 picks;
  - the lowest-scored seed as the predicted failure.
- **Checkpoints.**
  - ckpt_every ≤ 8000 (s185 used 1000), and **a full-state copy at 8000 is kept** (model, opt, np_rng, gn_hist).
  - nested170's `--keep_ckpt_steps` saves the model only, which is enough for eval-only probes but not for a resume. Keeping a copy of ckpt_last at t1 is a small engineering change, exactness-gated (flags-off bitwise).
- **What a resume costs (DERIVED(A1–A3, Part II §3.2's chaos assumption)).**
  - A resume restores the model, optimiser and np_rng (190-1). GPT2KV has no dropout. Training draws only from numpy; torch RNG is used only at construction and in eval probes (nested170 l. 472, 664; CHECKED by grep).
  - So the residual nondeterminism is reduction order and kernel choice at relaunch: a rounding-level perturbation.
  - That perturbation decorrelates after k* ≈ 350–3000 steps and re-draws the remaining decision variance:

    P_flip = 2[Φ(z) − Φ₂(z, z; R(t1 + k*))].

  - Resulting flip probabilities:

| t1 | central model | slower-accrual variants |
|---|---|---|
| 8k | 3.5–7.8% | 11–16% |
| 7k | 4.7–10.5% | — |
| 6k | 6.4–14.3% | — |

  - That is comparable to a detector's margin, so in-process continuation is required, and any resumed seed is reported with its resume step.
- **Cost** (D11: 35.4–42.6 ks per 48k on a 3090, at $0.179–0.228/h):
  - to 8k: 4 × 1.6–2.0 h ≈ $1.2–1.8;
  - all four to 48k: about $7.0–10.8, central about $9.3, plus evals: **≈ $9.5 (high $11)**.
  - **Needs the user's approval** (relayed via Audit).
- **Readings.** The new seeds alone cannot reach p < 0.05.
  - **CONSISTENT:** every new failure scores below every new entering seed. Under the null this has probability 1/4 with 1 failure and 1/6 with 2.
  - **INCONSISTENT:** C_new ≤ 0.5.
  - Pooled with RMS s1–s3 (about 7 seeds, about 2 failures), the best one-sided p is 1/21, and only with a perfect order. It is a **direction check**.
  - A decisive prospective test would need about 25 seeds (80% power at C = 0.8; about $58). That is not proposed.

### 6.2 The $0 alternative
- Keep the 8k full-state checkpoint and log in every future nested multi-seed run.
- Commit the frozen detector's 8k prediction in each launch note.
- The prospective panel then accumulates at no extra spend.
- The kept checkpoints also allow detectors that read the state rather than the loss (λ_eff O3, the step ratio O2, step_recover at t = 2), which are not limited by §2.1(c). 199-T1 is a caution for delta_cos.

---

## 7. Engineering and gates

- **Order:**
  1. Audit review.
  2. Register (a THEORY.md section).
  3. Re-run `--selftest` if SPEC changed.
  4. `--freeze` writes frozen.json (the SPEC sha256); commit it with the registration.
  5. `--run` writes results.json.
  6. A scored note.
- **Compute:** CPU, one thread, Idle priority (psutil inside each script); a few minutes. No GPU, no writes outside theory/seedselect/.

**Gates:**
- **G0, fields.** Checked per run and per field. A missing field (e.g. aux_ce when aux_step = 0, or write_grad_norm in logs that predate it) drops that feature for that run, and the drop is listed.
- **G1, window integrity.**
  - Dedupe by step (the last occurrence wins; basin_entry's rule).
  - Every grid record in [t0, t1] must be present, finite and > 0.
  - Only phase-1 records are used (§129's continuations carry phase 2).
  - A resume marker at or before t1 excludes the run. None is expected; s190 s1 resumed at 27k.
- **G2, duplicates.**
  - The 52 candidate pairs (same seed, card class and recipe; listed in labels.json) are compared record for record over [0, 2000] at 100-step resolution; replay 50-step records are averaged in pairs.
  - Pairs identical to within 1e-6 relative count once within a reading (keeping the replay or the first dir), never across readings.
- **G3, labels.** Units are built from labels.json, which holds outcomes only. The replay130 FLAG recompute agreed with score_s130 in 57/57.
- **G4, freeze.** `--run` refuses unless frozen.json's hash equals SPEC's and selftest.json passed on that SPEC.
- **G5, selftest (PASSED 2026-10-14 20:59 EDT on SPEC sha256 d85ded6c65b1…; selftest.json):**
  - a planted hare effect gives C 1.0 (p 5×10⁻⁴);
  - 40 null panels give mean C 0.48 and SD 0.12 (analytic 0.12), with 2.5% rejection at α 0.05;
  - exact p 1/3 (binary 2 S / 1 F) and 1/48 (R3's ordinal design) are reproduced;
  - the resume and window gates fire; dedupe keeps the last occurrence.
- **Gdiv (descriptive; printed before any reading).** For each card pair, the median first record at which |Δ ln loss| > 0.01 across replay_x cards.
- **results.json contents:**
  - the SPEC hash and the gates;
  - for each reading and feature: C, pairs, p (exact or MC), the CI, the verdict and the strata;
  - per-run feature rows. These are written only after the run, when no rule can change.

---

## 8. Claim status

| claim | status |
|---|---|
| A nested window ending at or before 2k is blind through the m channel (G ≡ 0, m frozen) | PROVED (Lemma 1, Prop O) |
| R(6k) 0.85, R(7k) 0.92, R(8k) 0.96 (central); variants 0.58–0.97 at 6k | DERIVED(A1–A3) |
| Ideal-observer AUC ceiling as a function of R | MC |
| Sign-blindness of L(m) at quadratic order; about 10⁻³ nats of sign signal at 8k vs about 10⁻¹ nats of C-speed spread | DERIVED(A4–A6) |
| Hare-channel AUC ≈ 0.55 | DERIVED(A) + MC (Part II S2) |
| Prediction: nested loss-only C ≈ 0.55 at t1 = 8k | DERIVED(A); tested descriptively by R3 |
| Same-seed cross-card FLAG flips 0.35–0.475; tetrachoric ρ 0.09–0.33 | DATA (labels) + DERIVED(A7) |
| Discordant same-seed pairs tie for any pre-divergence feature | PROVED |
| Sequential cost formula; the condition for a detector to pay | PROVED |
| Parallel-N success and break-even AUCs | DERIVED(binormal) + MC |
| Resume flip probability at t1 | DERIVED(A1–A3 + Part II §3.2 chaos) |
| fsq120 hare mapping (shortcut = C) | CONJ |
| LFA sign (Notsawo, across hyper-parameters) | EMPIRICAL (literature); CONJ here |
| A within-recipe per-seed early detector exists | unproven (prior) |

---

## 9. Risks

1. **Transfer.** R1 is another architecture (the FSQ carrier) with another failure mode and schedule.
   - A USEFUL R1 does not license use on nested; it only motivates §6.
   - A NOT USEFUL R1 argues against nested loss-curve selection *a fortiori*, since nested's decision is less observable (§2.1(c)).
2. **Outcome proxies.**
   - FLAG4's recall is 0.73, so about 12% of R1's "successes" may be failures (§131: 3 of 26). That attenuates C toward 0.5 by about that fraction.
   - FLAG fires after early pulls (§130), and R2 is coupled (§2.2).
3. **Strata.** Drivers and hosts vary within a card class and are not stratified. R1 has only two card classes.
4. **Prior exposure** (§1.3). R3 and R1b are descriptive for that reason. R2's probes were seen.
5. **Model dependence.** R(t), the observability estimate and P_flip use the 1-D reduced model with Part II's calibration. The variants bracket R(8k) at 0.81–1.00. The m_c and ΔL_c estimates are order-of-magnitude.
6. **Literature.** Notsawo 2306.13253 predicts grokking across hyper-parameters, so per-seed use within a recipe is an extrapolation. ILDR 2604.20923 times an eventual transition and does not show separation of never-entering runs (both EMPIRICAL; 35b6a795). The prior that a within-recipe per-seed detector is unproven is kept.
7. **Small effective n.** R1's 92 pairs come from 28 runs. Pairs are not independent; the permutation null accounts for this, the normal approximations in §4 do not.
8. **Post-selection.** Any detector used to drop seeds must itself be registered before use (Part II §6). This draft fixes the detector before any feature is computed.
