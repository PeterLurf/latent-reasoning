# PROOFS_SD2: proof and arithmetic pass on SD2_draft.md (SELF-DEPTH-CHAIN round 2, proposed §216)

Clock 2026-10-18 00:40 EDT (`date`). Opus proof-and-arithmetic checker for Theory. $0: CPU only (1 thread, idle priority), no GPU, no repo edits. `find -newer` on boxes/ and theory/ after every run: nothing written (all scripts run with `-B`, `PYTHONDONTWRITEBYTECODE=1`).
Target: research5/SD2_draft.md (00:16 copy) and research5/scripts/*.
Raw files read (explicit paths only): boxes/runs_fsq/{s213_d3t_s1, s215_ftself_s1, s215_ftctrl_s1}/eval_vb.json; {s213_d3t_s1, s215_ftself_s1}/{eval_chainlabel_fresh, eval_hist24, args}.json; s215_ftself_s1/{eval_selfdepth.json, log.jsonl, draws_log.json}; s215_ftctrl_s1/log.jsonl; s215_ftself_s1_{mem,gtime}/*; s213_d3t_s1_gtime/timing.json; s215_pool/{pool_meta.json, pool_r1.jsonl} (T lengths only); theory/results/score_s215.json.
**Not read:** s215_ftctrl_s1/eval_hist24.json and eval_d3hist.json.
Code read: vbank.py l.117–141 (bias_row), 200–345 (gather, coda, rollout), 455–520 (batch_x, vb_loss), 545–580 (CurGate), 593–909 (main); slva.py l.40–80; vbank_eval.py l.102–150 (part_acc); nested170.py test_set / train_items; theory/score_s213.py hist24.

Scripts and outputs (research5/proofs_sd2/, JSON beside each):

| script | checks |
|---|---|
| p1_panels.py | crossings at 30/50/70/90 %, margins/B; H_shift / H_prop / SHRINK and a rigid "constant-growth" H_add panel, per-n and on a 2-D (K′, n) table; M2, M_far, n50, STOP-LOC-2, KB2, RISE, SEs; retro-test on round 1 |
| p1b_kb2_interp.py | KB2 / STOP sensitivity to the interpolation axis ((K′, n) vs (lag, n)) |
| p2_support_s1.py (+ support_check_rerun.json) | S1 table, independent exact-rational rewrite; the PROOFS script re-run (output identical to support_check_r2.json); G-S3 SE; per-micro-batch Kb |
| p3_noise_gate.py | kept-wrong counts, CP bounds, φ̂, κ in both PROOFS §3b forms, two-round factor; GATE-2 / GATE-1′ exact OC, π^E term, keep term, LABEL-AUDIT-2, RETAIN, K′24 binomial |
| p5_mem_time.py | memory model inputs (round-1 pool T, n-16 T over 5 seeds), time model, G-TIME chunk shapes, eval slopes, §8 sums |
| p6_oc_exposure.py | seed-1 exposure, round-1 absorption point, OC recompute + sensitivities, time-model refit |
| p8_testset_tsplit.py | test_set bitwise (re-run), T maxima, FT-SELF T-split with tests |

Verdicts: **PASS** (reproduced), **CORRECTED** (number or wording wrong; fix given), **FAIL** (claim false as stated; replacement given).

---

## 0. Verdict table

| # | claim (draft §) | verdict | correct value / fix |
|---|---|---|---|
| 1 | crossings D3-T n50 10.31 (2.31, 0.288), n90 9.07 (1.07, 0.134); FT-SELF n50 15.46 (3.46, 0.288), n90 13.93 (1.93, 0.161) (§3.4) | PASS | all eight numbers reproduce |
| 2 | "margin/B fits 0.288 / 0.288" (row N1, §0) | PASS at 50 % **only** | 0.134 → 0.161 at 90 %, 0.212 → 0.229 at 70 %, 0.288 → 0.288 at 50 %, 0.362 → 0.356 at 30 % (round-1 value needs the assumed a_1(17)); §1.2 |
| 3 | H_shift row 0.999 / 0.893 / 0.633 / 0.344 / 0.18 / 0.12 / 0.11 / 0.11; M2 0.717; M_far 0.13; n50 19.46; KB2 0.155 | PASS | KB2 0.155 exactly from round 1's (0.633 − 0.517 + 0.344 − 0.150)/2 |
| 4 | H_shift STOP-LOC-2 "19–20"; "G(19) 0.025" | PASS / CORRECTED | G(19) = **0.023** with the measured a_1(16) = 0.344 (0.025 came from a K′-interpolated 0.351); STOP 20 under the extension, 19 if a_1 − a_1,15 ≈ 0 at n 17–19 |
| 5 | H_prop row 0.999 / 0.946 / 0.828 / 0.633 / 0.416 / 0.262 / 0.166 / 0.124; M2 0.852; M_far 0.24; STOP 21; KB2 0.29 | PASS | KB2 0.29 ((K′, n) bilinear) – **0.32** ((lag, n) bilinear): 4–7 SE above 0.25 |
| 6 | H_prop n50 20.63 (§3.4 table, §4.3) | CORRECTED | **20.61** (exact: 16 + 3.4602 / 0.75 = 20.614; per-n table gives the same). 20.63 is the 2-D panel's K′-interpolation artefact |
| 7 | SHRINK 0.480 / 18.31 / STOP 18 / KB2 −0.002 | PASS | |
| 8 | H_full floor "M2 ≥ 0.94, M_far ≥ 0.84" | CORRECTED | ≥ **0.9375** / ≥ **0.8375** (a floor must not round up) |
| 9 | SEs per n; SE(M2) 0.0059 / 0.0052; gaps 14 / 10 / 11.5 / 13 SE; KB2 ≈ 10 SE | PASS | KB2 SE 0.008 (one-way discordance, minimum) → 11.8 SE; 9.5 at 0.01 |
| 10 | "For this +4 step H_prop coincides with 'margin grows by a constant 1.15 per round' … they would differ only under a different window" (§3.4) | **FAIL** | They coincide **only at the 50 % crossing** (n50 20.61 both). A rigid +1.15 growth (round 1 translated by 5.15) gives M2 **0.895** (vs 0.852), KB2 **0.51–0.56** (vs 0.29–0.32), M_far 0.21 (vs 0.24), n90 19.09 (vs 18.41), STOP 21 (same). §1.3 |
| 11 | "DBS-2 is a direct test of H_prop" (§0, §5.1) | CORRECTED (wording) | DBS-2 passes under H_prop (0.852) **and** under constant growth (0.895). It tests margin growth. M2's level (≈ 8 SE apart) and KB2 separate the two growth forms |
| 12 | "the scorer replaces it by FT-SELF's eval_deep, which affects only n 21–24 under H_shift" (§3.4) | CORRECTED | also H_prop n 22–24 (a_1(16.5), a_1(17.25), a_1(18)), so M_far, and H_shift's STOP (G(18), G(19)) |
| 13 | S1 table P(R \| t ≥ t*) 0.637 … 0.143, 0 at ≥ 17; deadline costs 14.8 % … 0.35 % (§3.2, §2.3) | PASS | exact; P(R \| t ≥ 13) = **5/11**; re-run JSON identical to support_check_r2.json; independent rewrite agrees |
| 14 | G-S3 target 0.455 ± 0.01 "over 20k logged draws" | PASS / CORRECTED (wording) | SE 0.0022 at 20k (±0.01 = 4.5 SE); the log covers all 12,000 steps (≈ 383k self items, SE 0.0005). Under the seed-1 replay the value is computable **before launch** from pool_r2 + round-1's draws_log U's |
| 15 | kept-wrong 2 / 3,929 (FT-SELF, both at n 15), π 0.99949, CP lower 0.9984; D3-T 3 / 3,784, 0.99921, 0.9980 (§3.3a) | PASS | one-sided 95 %; two-sided 0.9982 / 0.9977 (score_s215 uses two-sided CIs; state the convention) |
| 16 | φ̂ 2/73 = 0.027, κ ≥ 2.2; φ upper 0.084 → 0.71; D3-T 4.2 / 1.6; round 1 8.1 (§3.3b) | PASS (arithmetic) | the direct-w form (same PROOFS recursion) gives **5.3 / 1.7** (W_2) and **10.7 / 3.4** (whole n 9–16 pool, w 2 / 7,926); §3 |
| 17 | two W-4 rounds, κ 1: "≤ 1.31 (φ 0.010) and ≤ 1.55" | CORRECTED | the φ 0.010 variant's premise is contradicted at round 2 (φ̂_2 0.027). Valid: **1.55** (point φ_1 0.0049, φ̂_2 0.027), **2.68** (both 95 % uppers), 1.43 (direct-w form) |
| 18 | bias-class mass "≤ 2(1 − π) ≤ 1e-3" (§3.3d) | CORRECTED | 2(1 − π) = **1.02e-3** (point), ≤ **3.2e-3** at π's 95 % lower. Cumulative over the lineage ≈ 1.8e-3 (§3) |
| 19 | 200× below Lee's 0.1; 100× below 5 % (§3.3d) | PASS | 196× / 98× |
| 20 | [PROVED UNDER A1′, A2, A3′] carried to round 2 (§3.3b) | CORRECTED | the first compounding round needs per-round φ_q, P_q (A4) and no error-class reinforcement or a cumulative class bound (A5); A3′ is unverified on FT-SELF. §3 |
| 21 | GATE-2 exact OC at 1,994 / 1,978 / 1,972 / 1,914, c_n 19, c_pool 39: 0.9987 / 0.52 / 0.014; = GATE-1's 0.9986 / 0.504 / 0.0125 (§2.1) | PASS | 0.99874 / 0.5237 / 0.01439. GATE-1′ (2,000 / 1,998 ×3; c 20, 19, 19, 19; 39): 0.9983 / 0.481 / 0.0105 |
| 22 | π^E term (7,858 kept, ≤ 78 wrong): ≈ 0.05 at 0.988, ≈ 1 at 0.993 (§6, "not recomputed") | PASS | recomputed: **0.048** / **0.9987**; 0.50 at 0.990 |
| 23 | keep ≥ 0.70 term ≈ 1 | PASS | ≈ 1 at keep 0.957 (0.978 even at 0.72). Kept counts are KEEP-only projections (CV unmeasured on FT-SELF); at CV = 0.98 × KEEP the OC is 0.9984 / 0.511 / 0.014 |
| 24 | LABEL-AUDIT-2 "π 0.985: P(≤ 50 wrong) ≈ 0.002" (§6) | CORRECTED (minor) | exact **0.0013** pooled; per window (≈ 2,500, ≤ 25 wrong) **0.019** |
| 25 | RETAIN: ≈ 1 intact, 0.01 under 2 % erosion; ABSORB-2 ≈ 40 SE (§6) | PASS | 1.0 / 0.0102; 40.3 SE |
| 26 | memory model; round-1 validation 14.47 projected vs 13.91 measured; m 1 26.0 vs OOM (§3.5) | PASS | round 1 used T 47 = the **maximum** T of the pool's n-12 records (also the 32nd longest) |
| 27 | "T(64th longest of 57k n-16 draws) = 59" → "m 4: 13.8 GiB" at the deepest shape (§3.5, row 8) | PASS (arithmetic) / CORRECTED (use) | 59 in 5 of 5 seeds, but vb_loss pads each micro-batch to its own max and G-S9's micro-batch 0 holds the longest record: peak T = **max = 61** (4/5 seeds; 63 in 1/5) → **14.4 GiB** (15.0 at T 63). Micro 2 at T 61 = 25.8 > 21. **Micro 4 stands** |
| 28 | m 1 46.5 / m 2 24.7 / m 4 13.8; T 61 14.4; T 63 15.0 | PASS | |
| 29 | time inputs 0.896 / 0.439 s; 7,260 / 4,877 s; 0.804 s (0.897 ×); 0.406 s (§3.6) | PASS | |
| 30 | compute-bound 1.52 s, launch-bound 2.24 s; 3.7–4.9 h → $0.62–1.27; real 3.4–4.4 h → $0.57–1.15 | PASS (arithmetic) | model inputs: G-TIME's forced self chunks are pool-order records, mean chunk max-T **41.5** (r1, 32 per chunk) and **51.6** (r2, 16 per chunk), not 47 / 59. Refit: compute-bound **1.73 s** → 4.0 h → $0.68–1.04. Range becomes **$0.68–1.27** (upper bound unchanged) |
| 31 | "P(Kb = 20) ≈ 0.80 per batch, so real ≈ 0.9 × forced" | CORRECTED | 0.802 per 64-batch, but the rollout runs per micro-batch (vb_loss Kb = max over the chunk): at micro 4 (16 items) P(Kb = 20) = **0.33**, E[Kb] **18.96**. Real/forced is likely < 0.9 (conservative direction) |
| 32 | eval slopes 0.136 / 0.236 GiB per item; B 4 → 8 within 3 %; bs 32 8.03, bs 16 4.25, bs 8 2.36 (§3.7) | PASS | 2.0 % / 2.5 %. No-intercept projection is conservative (fitted intercept −0.55 GiB → 7.48 / 3.70 / 1.81) |
| 33 | "64 × 0.136 = 8.69 vs the 3090 mem_check's 8.93: reproduced within 3 %" (§3.7, row 7, §10) | CORRECTED | arithmetic right, but not like-for-like: the 3090 eval is **bf16 autocast with one coda (K′ 24)** (vbank.py l.853–857); CPU is fp32 with codas at every K′. Agreement is a consistency check, not a validation. G-S16 is the only validation (bs 16 has 1.6× slope headroom to 7.0 GiB) |
| 34 | round 1's 3070 bs-32 n-16 eval 4.82 GiB "ran"; ≈ 8.4 "OOMed" | PASS | bracketing consistency only |
| 35 | test_set(1000, 24) n 1–20 bitwise = test_set(1000, 20); n 1–16 = test_set(1000); T max 61 / 73 / 83 at n 17 / 20 / 24, mean 73.9 at n 24 (§3.1c, §10) | PASS | re-run on CPU (syms, t, n, depth). eval_selfdepth's T lists align with test_set order at n 9–16 |
| 36 | "rollout has no pass cap, no table indexed by pass" (§3.1d) | PASS | bank / NV / gk are Python lists appended per pass (l.282–345); bias_row and SLVA's bias are β0 on V_0 + γ·(t − 1 − s) (vbank l.125–141, slva l.69–73); KMAX only defaults kmax and calibrate() |
| 37 | round-1 rows up to 59 at n 15–16, "≥ 10 untrained rows" (§3.1c) | PASS | the round-1 trained max T = 47 (pool n 12), so 12 untrained rows |
| 38 | tsplit "no within-n T effect" (n 16 0.319 vs 0.352; n 15 0.658 vs 0.592) (§3.1c, §10) | CORRECTED | numbers PASS, conclusion overstated: n 15 drops **−0.066 at T ≥ 49 (untrained rows; z −2.08, nominal p 0.04)**, n 14 −0.055 (z −1.37), n 16 +0.033 (z +0.92); pooled within-n slope −0.0032 per row, stratified permutation **p 0.13**. Write "no significant pooled within-n T effect (p 0.13); n 15 nominally negative" |
| 39 | exposure "6,000 ± 55 self steps → 384,000 ± 3,505; 48k per n; z(exhaust) 18.3" (§2.3) | CORRECTED | seed 1 replays rng_s exactly (1 coin per step + 64 U per self step in both rounds; vbank l.744–766), so the count is **deterministic: 5,992 self steps, 383,488 items, 47,936 ± 78 per n** (hypergeometric), pool margin 64,512 records; z is moot |
| 40 | "self_answer_ce 0.06 at 1k and 0.0095 at 3k (≈ 24k per n). So 48k per n is 2× round 1's absorption point" (§2.3; reused in §6's P(ABSORB-2) 0.75) | **FAIL** | 0.0604 / 0.0095 PASS (cursor 94,464 → 23.6k per n), but 3k is a one-window dip: CE ≥ 0.03 in 7 windows between 3.2k and 5.7k (max 0.067 at 5.6k). Sustained < 0.02 only from step 5.8k (**46.4k per n**); < 0.01 from 9.3k (**74.2k per n**). Round 2's 48k per n is **1.03×** / **0.65×** those points, 2.1× only the first touch of 0.01 |
| 41 | OC: priors 0.375 / 0.542 / 0.025 / 0.058; kept 0.539; P(SD2-SELF) 0.40; P(SD2-C2) 0.111; \| H 0.020 / 0.242 / 0.384; DBS 0.21, MG 0.27, KB 0.16, C2-MEM 0.028; sensitivity 0.085 / 0.138; 2/2 ≈ 0.02–0.04 (§6) | PASS (arithmetic) | 2/2 = 0.0176–0.0371 |
| 42 | independence "kept readings independent of H and of ABSORB-2" (§6) | CORRECTED (assumption) | inconsistent for K′24-STABLE: round 1's n-1 dip (0.990) occurred in FT-SELF only (FT-CTRL 1.000), so it is driven by the same self-item dose that drives ABSORB. Treat P(K′24 \| ABSORB-2) as lower (§7) |
| 43 | §8 family "$2.9–5.0"; guards sum $6.25 ≤ cap $6.5; option (b) "$2.2–3.6"; option (a) $1.1–2.0; alt (a) total $1.7–3.3 | CORRECTED (minor) / PASS | family **$2.81–4.97** → "$2.8–5.0" ($2.9 only after #30's refit); guards 6.25 ✓; admissions sum $5.40; (b) $2.19–3.65 → "$2.2–3.7"; (a) $1.07–1.95 ✓; alt (a) $1.69–3.22 ✓ |
| 44 | D3-T s2 6.2 h forced (s1 real 4.96 h); R1-s2 "2.2–2.6 h" → $0.45–0.68; labelling 1.2 h (4,250 + 150 s) (§8) | PASS / CORRECTED (minor) | 6.16 / 4.96 h ✓; R1-s2's cost uses 2.63 h forced incl. overhead (2.4 h real), so write "2.6 h"; labelling ✓ |
| 45 | STOP-LOC-2 / n50 data source (§4 notation vs §4.1 table) | CORRECTED (spec) | ambiguous: the notation sends n ≤ 16 to eval_vb, while §4.1 lists STOP-LOC-2 and n50 under eval_deep. Fix in §4 below |
| 46 | other numeric claims: 56,512 unused / ~192k needed; 0 kept-wrong of 2,512 at n 17–20; keep table; collision bound at n ≥ 9; seeds unused (grep); init_from l.666, CurGate l.557 / l.812; MEM history 0.782 → 0.489, dup_newest 0.877, self-ablated a_24 0.43–0.60 vs 0.21–0.23; chance 0.101–0.108; RISE ≈ 0; π^E = π on D3-T 1,892 / 1,892 (RT) | PASS | collision bound with the larger exclusion set: ≤ 2e-6 at n 9 |

**Tally: 46 rows. PASS 25** (#1–3, 5, 7, 9, 13–16, 19, 21–23, 25, 26, 28, 29, 32, 34–37, 41, 46). **CORRECTED 19** (#4, 6, 8, 11, 12, 17, 18, 20, 24, 27, 30, 31, 33, 38, 39, 42–45; mixed rows such as "PASS (arithmetic) / CORRECTED (use)" are counted as CORRECTED). **FAIL 2** (#10, #40).

---

## 1. §3.4 hypothesis panels (p1_panels, p1b_kb2_interp)

### 1.1 Reproduced numbers

| | M2 (17–20) | M_far (21–24) | n50 | STOP-LOC-2 | KB2 | RISE |
|---|---|---|---|---|---|---|
| H_shift (per-n = 2-D) | 0.7172 (SE 0.0059) | 0.1311 | 19.460 | 20 (G(19) 0.0231) | 0.1550 (SE ≥ 0.008) | −0.0065 |
| H_prop per-n / 2-D | 0.8516 / 0.8529 (SE 0.0052) | 0.2421 / 0.2429 | **20.614** / 20.624 | 21 / 21 | 0.2906 / 0.2920; (lag, n) axis **0.3174** | −0.0075 |
| SHRINK | 0.4798 | 0.1295 | 18.306 | 18 | −0.0015 | −0.0005 |
| **H_add** (round 1 translated by 4 + δ, δ = 1.1546) | **0.8954** | 0.2098 | 20.609 | 21 | **0.512** / (lag, n) **0.562** | +0.0012 |

- Gap to 0.80: H_shift +14.06 SE, H_prop −9.99 SE. Gap to 0.785: +11.51 / −12.89 SE. All as drafted.
- My 2-D round-1 table interpolates K′ 17–23 through K′ 16, K′ n + 2 (acc_n2) and K′ 24. The draft's interpolates K′ 16 → 24 linearly, which gives a(16) = 0.351 at K′ 18 against the measured 0.344. That is the source of its 0.719 / 0.1587 / 0.0248 / 20.634 variants. The per-n tables (sd2_tables) are the right ones to quote.

### 1.2 Robustness of the H_prop fit at other levels

| level L | D3-T n_L (m, m/B) | FT-SELF n_L (m, m/B) | m ratio r1/r0 | H_prop predicts m_1 | constant growth predicts m_1 |
|---|---|---|---|---|---|
| 90 % | 9.072 (1.072, 0.134) | 13.934 (1.934, **0.161**) | **1.80** | 1.61 | 2.23 |
| 70 % | 9.697 (1.697, 0.212) | 14.742 (2.742, **0.229**) | **1.62** | 2.55 | 2.85 |
| 50 % | 10.306 (2.306, 0.288) | 15.460 (3.460, **0.288**) | **1.50** | 3.46 | 3.46 (fitted) |
| 30 % | 10.899 (2.899, 0.362) | 16.269* (4.269, 0.356) | 1.47* | 4.35 | 4.05 |

*needs the assumed a_1(17) = 0.180.

- **Margin/B is constant only at the 50 % level** (and ≈ at 30 % under the assumed extension). The top of the curve moved out more than proportionally (×1.62–1.80 vs B's ×1.5).
- The n50 → n90 width grew ×**1.24** (1.234 → 1.526). H_prop predicts ×1.5; a rigid translation predicts ×1.0. Round 1 sits between them.
- Retro-test on round 1 (D3-T → FT-SELF at n 13–16, measured 0.999 / 0.893 / 0.633 / 0.344, M 0.717):
  - H_prop: 0.949 / 0.816 / 0.603 / 0.378, M 0.687, RMSE **0.051**;
  - rigid growth δ: 1.000 / 0.935 / 0.653 / 0.318, M 0.726, RMSE **0.026** (δ is fitted on round 1's own n50; H_prop has no free parameter);
  - H_shift: RMSE 0.261.
- So the "2-point fit at face value" (§6 sensitivity 0.50) rests on one level. The data do support growth at every level (≥ ×1.47).

### 1.3 Fix text for §3.4 (replace the H_prop sub-bullet)

> For this +4 step H_prop and a constant +1.15 margin growth predict the **same n50 (20.61)**, since 0.288 × 4 = 1.15. Their curves differ: H_prop widens the transition by 16/12, a constant growth keeps its width. Constant growth gives M2 0.895, KB2 0.51–0.56, M_far 0.21 and n90 19.1, against H_prop's 0.852, 0.29–0.32, 0.24 and 18.4 [DERIVED, proofs_sd2/p1_panels]. Round 1's own width grew ×1.24, between the two. Margin/B is constant at the 50 % level only (0.134 → 0.161 at 90 %, 0.212 → 0.229 at 70 %). DBS-2 tests margin growth (both forms pass). M2's level and KB2 separate the two forms; n50 does not.

Also: H_prop n50 20.63 → **20.61** (§3.4 table, §4.3 n50 row, G-S12 fixture); H_full floor M2 ≥ **0.9375**, M_far ≥ **0.8375**; H_shift G(19) **0.023**. The a_1 replacement by FT-SELF's eval_deep affects H_shift n 21–24 and STOP, **and** H_prop n 22–24 / M_far.
- Optional fixture for G-S12: H_add (→ GROW or FULL? M2 0.895, M_far 0.21 → GROW; KB2 0.51 ✓) to exercise the KB2 split.

---

## 2. §3.2 S1 for round 2 (p2_support_s1, support_check_rerun)

- PROOFS' script (identical to support_check_r1copy.py, `diff` clean) re-run: JSON **identical** to support_check_r2.json.
- Independent rewrite (my own enumeration of (n, U) under p_self ½, self U{9..16}, K = min(n + U, 20); plain exclusion-weighted law, K ≤ 12): P(R | t ≥ t*) = 0.6370 / 0.6053 / 0.5660 / 0.5168 / **0.4545 (= 5/11)** / 0.3750 / 0.2727 / 0.1429 / 0. Deadline costs 0.1475 / 0.1117 / 0.0817 / 0.0570 / 0.0372 / 0.0218 / 0.0107 / 0.0035 / 0. **PASS**.
- G-S3: at t ≥ 13 only self items contribute (plain K ≤ 12), so the target is plain-law-free. Delta-method SE of the empirical ratio: 0.0022 at 20k self items, 0.0005 at 383,488. Under seed 1 the coin and U sequences equal round 1's, so the empirical value is fixed by the pool order and can be computed before launch. Fix: "over the 12,000 logged steps (383,488 self items); precomputed from pool_r2.jsonl + s215_ftself_s1/draws_log.json U's, then checked."

## 3. §3.3 label noise and the κ recursion (p3_noise_gate)

**Counts.** FT-SELF n 13–16: kept-wrong 0 / 0 / 2 / 0 of 997 / 989 / 986 / 957; non-strict 3 / 11 / 16 / 43. n 9–12: 0 of 3,997. n 17–20: 0 of 2,512. D3-T n 13–16: 0 / 0 / 3 / 0 of 3,784; non-strict 219. All draft numbers PASS.

**Re-derivation (PROOFS §3b).**
- kept = P + φ(1 − P) (every strict chain is kept). A kept non-strict chain has exactly n steps, so it is kept-wrong, and φ = kept-wrong / non-strict.
- w_q = φ(1 − P_q)/kept ≤ φ(1 − P_q)/P_q. A3′ gives 1 − P_q(n) ≤ n ē_q(n). So w_q ≤ φ n_q ē_q / P_q.
- A1′: an FT on pool share w raises e at every m ≤ B_{q+1} by ≤ κw. A2: clean labels do not raise it. Hence ē_{q+1} ≤ ē_q + κ w_q ≤ ē_q (1 + κ φ n_q / P_q).
- Avalanche (the bound doubles) needs κ ≥ ē_q / w_q. Two valid lower bounds on that threshold:
  - **φ form** P/(φ n): FT-SELF 0.957 / (0.0274 · 16) = **2.18**; 0.71 at φ's upper 0.0837. D3-T 4.18 / 1.64. Draft PASS.
  - **direct-w form** (1 − P)/(n w) ≥ P/(φ n) (since w ≤ φ(1 − P)/P): FT-SELF W_2 0.0026875 / 5.09e-4 = **5.3** (1.7 at w's upper); whole round-2 pool (w 2 / 7,926 = 2.5e-4) **10.7** (3.4).
  - The φ form is the loosest: it multiplies the pooled φ̂ (both events at n 15) by n 16's non-strict mass, where n 16 itself had 0 / 43 kept.
  - Per-n caveat: at n 15 alone both forms give κ ≥ **0.53** (2 of 16 non-strict kept). GATE-2's π_n ≥ 0.99 rule is the guard there (π_15 = 0.998).
- **Two rounds, κ 1** (per-round φ): (1 + φ_1·12/0.976)(1 + φ_2·16/0.957) = **1.55** at φ_1 0.0049, φ̂_2 0.0274; **2.68** at both 95 % uppers (0.00965, 0.0837); 1.43 in the direct-w form (round 1: LABEL-AUDIT's 2 / 5,000). The "≤ 1.31 (φ 0.010)" variant assumes φ_2 ≤ 0.010, which FT-SELF's point estimate contradicts. Drop it.

**Is A1′ / A2 / A3′ enough for the first compounding round? No. Two additions and one open premise:**
- **A4 (round-indexed parameters).** PROOFS' product used a single φ. FT-SELF's φ̂ (0.027; D3-T 0.0049 pooled, 0.0137 at n 13–16) shows φ is a property of each labeller. State the bound as ē_Q ≤ ē_0 Π_q (1 + κ φ_q n_q / P_q) with each round's measured φ_q and P_q. A better labeller can have a larger φ: fewer failures, a larger share of them clean-looking.
- **A5 (no error-class reinforcement).** The additive step assumes w_2's errors add to, and do not multiply, the errors FT-SELF absorbed from w_1 (2 / 5,000). For a self-trained labeller the natural failure is the same pattern class re-labelled wrong in round 2 and trained twice. Then the S2(c) class bound must be applied to the cumulative lineage mass, ≤ 2(1 − π_1) + 2(1 − π_2) ≈ 8e-4 + 1.02e-3 = **1.8e-3** (point). Fix: LABEL-AUDIT-2 compares its kept-wrong handle patterns with round 1's (score_s215 label_audit.kept_wrong) and with FT-SELF's 2 fresh n-15 kept-wrong chains.
- **A3′ on FT-SELF: unverified.** The per-state ↔ fresh-step identification was supported on D3-T's long_step product fit, not on FT-SELF. The union bound itself needs only A3′, but A3′ needs a FT-SELF check (eval_steppos long_step, $0, local) or a [ASSUMED] tag.
- The draft's conclusion "the κ bound is not the guard; the direct measurements are" stands under every form.

Fix text for §3.3(b), last two sub-bullets:
> Direct-w form (same recursion, measured w): κ ≥ 5.3 (W_2; 1.7 at w's 95 % upper), 10.7 / 3.4 for the whole n 9–16 pool. Two W-4 rounds with κ = 1 and per-round φ: ≤ 1.55 (point), ≤ 2.68 (95 % uppers) [DERIVED, proofs_sd2/p3_noise_gate]. Round 2 adds A4 (per-round φ_q, P_q) and A5 (no error-class reinforcement; the class bound is cumulative, ≈ 1.8e-3); A3′ is unverified on FT-SELF.

§3.3(d): "≤ 2(1 − π) ≤ 1e-3" → "2(1 − π) = 1.0e-3 (≤ 3.2e-3 at π's 95 % lower)".

## 4. §2.1 GATE-2 / GATE-1′ OC (p3_noise_gate)

| window (N_n) | c_n / c_pool | π 0.9995 | 0.997 | 0.996 | 0.995 | 0.994 | 0.993 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| GATE-2 1,994 / 1,978 / 1,972 / 1,914 | 19 ×4 / 39 | ≈ 1 | **0.9987** | 0.921 | **0.524** | 0.130 | **0.0144** | 5e-7 |
| GATE-1′ 2,000 / 1,998 ×3 | 20, 19, 19, 19 / 39 | ≈ 1 | 0.9983 | 0.905 | 0.481 | 0.107 | 0.0105 | 3e-7 |
| GATE-2 at CV = 0.98 × KEEP | 19, 19, 19, 18 / 38 | ≈ 1 | 0.9984 | 0.914 | 0.511 | 0.126 | 0.0141 | 6e-7 |

- π^E pooled (7,858 kept, ≤ 78): 0.9987 at 0.993, 0.504 at 0.990, **0.048** at 0.988. Keep ≥ 0.70: ≈ 1 at 0.957.
- All draft values PASS. The kept counts are 2,000 × the KEEP-only fresh rate; KEEP ∧ CV on FT-SELF is unmeasured (RT measured CV = KEEP only on D3-T). The OC barely moves at CV = 0.98 × KEEP.

## 5. §3.5–3.7 and §8: memory, time, cost (p5_mem_time, p6_oc_exposure)

- **Training memory.** Round-1 pool n-12 T: max 47, 16th 47, 32nd 47, 64th 45. The 14.47-vs-13.91 validation used the padded T of micro-batch 0 = the maximum.
  - Round 2's analogue is the max T of the n-16 records: 61 (4/5 seeds of 57k), 63 (1/5) → **14.4 / 15.0 GiB** at micro 4.
  - Fix row 8 / §3.5: "the 64 longest n-16 records pad micro-batch 0 to the longest record's T, 61 in 57k draws (63 in 1 of 5 seeds; 59 is the 64th): m 2 25.8 > 21.0, **m 4 14.4 GiB** (15.0 at T 63)".
  - Also: NOT-SIZE-KEYED-2's split "T ≤ 61 vs ≥ 63" should read "T ≤ T_train vs > T_train, T_train = max T of the consumed n-16 records (pool_meta)", since the pool max may be 63.
- **Time.** Arithmetic PASS. The fit used T 47 / predicted at T 59 (deepest records). G-TIME's forced self steps are pool-order chunks, mean max-T 41.5 (round 1, 32 per chunk) and 51.6 (round 2, 16 per chunk). Refit: compute-bound 1.73 s, forced 4.0 h, **$0.68–1.04**. Launch-bound 2.24 s / $1.27 is unchanged, so per-cell **$0.68–1.27**. Real/forced: per-micro-batch E[Kb] 18.96 (P(Kb = 20) 0.33 at micro 4), so ≤ 0.9 is conservative.
- **Eval memory.** Slopes and projections PASS. The 3090 comparison is bf16 + one coda vs fp32 + all codas: reword to "consistent with", not "reproduces". G-S16 decides; bs 16 has ≈ 1.6× slope headroom to 7.0 GiB.
- **§8.** Per-cell lows sum to **$2.81**, highs $4.92 (+ ≤ $0.05 egress): write **$2.8–5.0** (or $2.9–5.0 after the refit's +$0.06 per R2 cell). Guards 1.60 + 2.00 + 1.05 + 1.60 = **$6.25 ≤ $6.5** ✓; admissions $5.40. Option (b) s2 $2.19–3.65 → "$2.2–3.7". D3-T s2 6.16 h forced + 0.4 → $1.11–1.70 ✓. R1-s2 2.63 h forced → $0.45–0.68 ✓ (hours column "2.6", not "2.2–2.6").

## 6. §2.3 exposure and the round-1 absorption point (p6_oc_exposure)

- vbank.py draw_step: rng_s = default_rng(seed + 21800) is consumed by one coin per step and, on a self step, 64 U draws (self_kextra 4 in both rounds). The plain draw (rng, rng_k) happens every step first. So **seed 1 reproduces round 1's coin and U sequences exactly**: 5,992 self steps (round-1 draws_log), 383,488 items, **47,936 ± 78 per n**, 191,744 ± 117 per window, and 64,512 unused records. z(exhaust) is moot.
- Fix text: "Exposure: exactly round 1's 5,992 self steps (seed-1 replay of rng_s) → 383,488 items, 47,936 ± 78 per n (pool order), 64,512 records spare."
- Absorption point: replace "0.0095 at step 3k (≈ 24k items per n). So 48k per n is 2× round 1's absorption point" with:
  > "Round 1's self_answer_ce first fell below 0.01 at step 2.9k (22.9k per n) but stayed ≥ 0.03 in 7 windows up to 5.7k. It held < 0.02 from step 5.8k (46.4k per n) and < 0.01 from 9.3k (74.2k per n) [MEASURED log.jsonl]. Round 2's 48k per n at n 13–16 is ≈ 1× round 1's sustained point (0.65× at < 0.01). Without T2 the n 13–16 CE is unread (the logged CE pools both windows)."
- This removes one of §6's stated reasons for P(ABSORB-2) 0.75 (§7).

## 7. §6 OC arithmetic and independence (p6_oc_exposure)

- Every number in §6 reproduces: priors 0.375 / 0.5417 / 0.025 / 0.0583; kept 0.53865; P(SD2-SELF) 0.404; P(DBS | ABSORB) 0.2758; P(SD2-C2) **0.1114**; | H 0.0202 / 0.2424 / 0.3838 / 0; DBS 0.207, MG 0.270, KB 0.164; C2-MEM 0.0279; sensitivity 0.0852 / 0.1376; 2/2 0.0259 (indep.) / 0.0546 (corr.) × 0.68 = **0.018–0.037**. PASS.
- **Inconsistent independence.** The kept factor multiplies P(ABSORB-2) by K′24-STABLE 0.70 as if independent. Round 1's K′24 edge (a_24(1) = 0.990) appeared in FT-SELF only (FT-CTRL 1.000), so it is caused by the self-item dose that also produces ABSORB. Round 2 adds the same dose again, with deeper K.
  - A binomial floor makes the point concrete: if the true n-1 error stays at 0.010, P(≥ 990 / 1,000) = **0.58**. At 0.015 it is 0.12; at 0.005 it is 0.99.
  - The "independent" 2/2 variant also treats K′24 as independent across lineages, though the dip is recipe-driven. The correlated variant covers that.
- **Sensitivity rows to add** (priors still [ASSUMED]):

| change | P(SD2-SELF) | P(SD2-C2) | 2/2 |
|---|---|---|---|
| as drafted | 0.404 | 0.111 | 0.018–0.037 |
| P(ABSORB-2) 0.60 (§6 exposure correction) | 0.323 | 0.089 | 0.011–0.030 |
| P(K′24 \| ABSORB-2) 0.55 | 0.317 | 0.088 | 0.011–0.029 |
| both | **0.254** | **0.070** | **0.007–0.023** |
| H_grow prior 0.25 / 0.50 | 0.404 | 0.085 / 0.138 | 0.013–0.028 / 0.022–0.047 |

- The conditionals P(SD2-C2 | H) scale the same way (| H_grow 0.24 → 0.15 in the "both" row). The discrimination statements ("a DBS-2 pass is strong evidence for growth") are unchanged.

## 8. §3.1(c)/(d) (p8_testset_tsplit, code)

- test_set bitwise and T maxima: PASS (rows 35, 37). Code claims: PASS (row 36). T-split: CORRECTED (row 38). It matters for H_T: n 15's T ≥ 49 items are exactly the untrained-row items, and they read lower (z −2.1). So round 1 does not rule out a row-keyed component; it only fails to show a significant pooled one.

## 9. One source for every reading that touches n ≥ 17

**What the draft says.**
- §4 Notation: "n ≤ 16 readings come from eval_vb.json (bs 32); n 17–24 readings come from eval_deep.json (bs 16)".
- §4.1 assigns STOP-LOC-2 and n50 to eval_deep. But STOP-LOC-2's windows k 13..15 span n 14..19, and n50 is read over n 13..24.
- A scorer that follows the notation literally would mix the files in G(13), G(14) and G(15) (n 14–16 from eval_vb, n 17–19 from eval_deep), and in an n50 crossing between 16 and 17.

**Would a mix bias G(k)?** Not systematically:
- each per-n term a(n) − a_k(n) comes from one rollout in one file (acc_n2 and acc_k share the K′ 24 / K′ 28 rollout; codas are causal);
- the n 13–16 items are bitwise identical (row 35);
- both evals run on the 3070 in fp32.

The only difference is the chunking (32 vs 16), which in fp32 flips near-ties only. But BS-CHECK bounds |Δa(n)| ≤ 0.005 and not Δa_k(n). A straddling G(k) can then move by up to ≈ ¾ (|Δa| + |Δa_k|) ≈ 0.0075. That flips STOP when G(k) ∈ (0.0125, 0.0275), and STOP ≤ 16 vs ≥ 17 is exactly the SHIFT / MIXED boundary. It also makes BS-CHECK failures leak silently into a mechanism label.

**Fix (replace the §4 notation bullets):**
> - Readings wholly inside n ≤ 16 (ABSORB-2, RETAIN, DEPTH-GENERAL-2, K′24-STABLE, FT-DIVERGED, VB-C1, THINK-NECESSARY-DEEP-2, K′24-WIDE's n 9–16 half) come from eval_vb.json (bs 32, unchanged).
> - **Every reading whose n-range includes any n ≥ 17** takes **all** of its n and K′ values, including n 13–16, from eval_deep.json only. These are DBS-2, MARGIN-GROWS-2, M_far, STOP-LOC-2 (all k 13..23, n 14..24), K′-BEYOND-2, RISE-PAST-K-2, n50 (n 13..24), C2-TAIL-2, NOT-SIZE-KEYED-2 and K′24-WIDE's n 13–20 half. score_s216.deep_read asserts that it opens only eval_deep.json.
> - BS-CHECK compares eval_deep with eval_vb at n 13–16 on a(n) **and** on a_k(n) for k 12..16. It is reported beside every eval_deep reading and is never a substitute source.

Add a G-S12 fixture: an eval_vb whose n 13–16 differ from eval_deep's by 0.01 must leave STOP-LOC-2 and n50 unchanged.

---

## 10. Fix list by weight

1. **§3.4 H_prop ≡ constant growth (FAIL):** replace the sub-bullet (§1.3 text). Reword "DBS-2 is a direct test of H_prop" to "a test of margin growth; M2's level and KB2 separate H_prop from constant growth". Margin/B constancy holds at 50 % only.
2. **§2.3 absorption point (FAIL) and exposure:** §6 text above. P(ABSORB-2) 0.75 loses its "2× absorption point" support; add the sensitivity rows (P(SD2-C2) 0.070–0.111).
3. **§4 single-source rule** for every reading touching n ≥ 17 (§9 text); BS-CHECK on a_k too.
4. **§6 independence:** K′24-STABLE is dose-coupled to ABSORB-2; state it and add the sensitivity.
5. **§3.3:** add A4 / A5 and the A3′ caveat; drop the φ 0.010 two-round variant; add the direct-w κ (5.3 / 10.7); class mass 1.0e-3 (3.2e-3).
6. **§3.5 memory:** peak T is the record maximum (61–63): 14.4–15.0 GiB, micro 4 unchanged. H_T split threshold from the realised T_train.
7. **§3.6 / §8 cost:** compute-bound refit at G-TIME shapes → $0.68–1.27 per R2 cell; family $2.8–5.0 (or $2.9 after the refit); per-micro-batch Kb.
8. **§3.1(c):** T-split wording (p 0.13; n 15 z −2.1 on untrained rows).
9. **Small:** n50 20.61; H_full floors 0.9375 / 0.8375; G(19) 0.023; LABEL-AUDIT-2 OC 0.0013 (per window 0.019); CP convention (one-sided); G-S3 precomputable; 3090 eval check is "consistent with", not "reproduces"; extension caveat covers H_prop n 22–24.

**Nothing here changes a registered threshold.** The OC headline P(SD2-C2) ≈ 0.11 is arithmetically right under the stated priors. It falls to ≈ 0.07 if P(ABSORB-2) and K′24-STABLE are re-shaded as §7 suggests (a prior choice for Theory).

*Written 2026-10-18 00:40 EDT (Opus proof subagent for Theory; $0; nothing committed or launched).*
