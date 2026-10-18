# SELF-DEPTH-CHAIN round 2 v1 (SD2, proposed §216): registration-ready text

Clock 2026-10-18 01:08 EDT (`date`). Opus design-revision subagent for Theory. $0.
**Status: registration-ready text, NOT registered.** It requires **the user's explicit OK** regardless of cap: round 1 read MEM-TRADED (STAB_self 0.489 < 0.5), and the frozen round-1 text says "MEM-TRADED → only with the user's explicit OK" (SELFDEPTH_v1 §9). The ≤ $30 Audit auto-approval does not apply. **The §215 pair (SELF-ATTRIBUTED / FT-EFFECT and the MEM attribution) is scored before the user is asked** (D12); this text states only how SD2's wording depends on that result (§2.5).
Inputs: research5/SD2_draft.md (00:16), REDTEAM_SD2.md (+ rt_sd2/), PROOFS_SD2.md (+ proofs_sd2/), research5/scripts/; the frozen round-1 text docs/research_selfdepth_2026-10-17/SELFDEPTH_v1.md (5ded8703); THEORY notes 214-5 and 215-7 (there is no note 215-6 in THEORY.md at 6c49b973; RT found the same); theory/score_s215.py (after 215-7) and theory/results/score_s215.json (SELF block); boxes/vbank.py, vbank_diag.py (part_hist24, part_selfdepth, _acc_n2_items), theory/score_s213.hist24.
Theory's decisions D1–D12 (relayed 2026-10-18) are applied and listed in §1.
**Not read, on purpose:** boxes/runs_fsq/s215_ftctrl_s1/eval_hist24.json and eval_d3hist.json.
$0 checks (research5/v1_checks/, JSON beside each script; CPU, ≤ 2 threads, idle priority, `python -B`, no file written under D:/Latent): listed in §10.
Tags: [PROVED], [PROVED UNDER X], [DERIVED] (exact arithmetic on measured inputs; script named), [MEASURED] (raw file named), [ASSUMED], [CONJECTURE].

---

## 0. One screen

| item | SD2 v1 |
|---|---|
| Question | Round 1 absorbed n 9–12 and grew the margin past support (M 0.717, n50 15.46). If the model now trains on its **own** round-1 chain at n 13–16, does it (a) absorb n 13–16 and (b) grow the margin **again** (any growth form), so that n 17–20 reaches ≥ 0.80 (C2 proxy)? Is the boundary still binding at n 22–24 (literal C2)? |
| What SD2 tests | **DBS-2 tests renewed margin growth vs H_shift, not H_prop specifically**: H_prop (M2 0.852) and constant growth (M2 0.895) both pass, H_shift (0.717) does not. H_prop's own discriminator is the descriptive WIDTH-2 = n50 − n90 (panels 1.53 / 1.57 / 2.22 for H_shift / constant growth / H_prop) [DERIVED v1_panels]. **LIT-C2-2** = mean_{n 22..24}[a(n) − a_20(n)] ≥ 0.10 is the C2-literal row (passes only if the boundary stops binding) |
| Cells | **s1:** `s216_r2_s1`, FT from `s215_ftself_s1/model.pt`, seed 1. **s2 lineage (D1):** `s216_d3t_s2` (D3-T args, seed 2, 48k) → GATE-1-s2 + pool → `s216_r1_s2` → GATE-2-s2 + pool → `s216_r2_s2`. RTX 3090 only. No R2-CTRL (D6) |
| Labeller | Each lineage's round-1 model's re-rooted single-step chain at both windows (s1: FT-SELF). Filter KEEP ∧ COMPOSE-VALID. GATE-2 (n 13–16) and GATE-1′ (n 9–12, relabelled), 2,000 per n, GATE-1's rule. Pre-stated fallback per window: the lineage's base (BASE-LABELLED) |
| Self law (D10) | p_self 0.5. Self items **2/3 n 13–16, 1/3 n 9–12** (pool composition 74,000 / 37,000 records per n; one fixed interleaved order). K = min(n + U{0..4}, 20). Seed-1 replay: exactly 5,992 self steps = 383,488 items, **63,915 ± 85 per new n (1.38× round 1's sustained CE < 0.02 point, 0.86× its < 0.01 point)**, 31,957 per old n [DERIVED v1_support_d10] |
| Recipe | FT-SELF's args with only init_from, self_pool, self_kmax 20, self_micro (G-S9: projected **4**, 14.4–15.0 GiB) and out changed; 12k steps, lr 3e-5. s2 cells: + seed 2 |
| Sources (D3) | Every reading that touches any n ≥ 17 takes **all** its values from `eval_deep.json` (n 13–24, K′ 0..28, bs 16). n ≤ 16-only readings stay on eval_vb (bs 32). BS-CHECK is descriptive |
| Bars | s1: DBS-2 0.80, MARGIN-GROWS-2 0.785, SHRINK < 0.60, FULL M2 ≥ 0.80 ∧ M_far ≥ 0.60, LIT-C2-2 0.10. **s2: the same formulas on the s2 lineage's own measured curves** (§4.3); s1's numbers are the formulas' s1 instance |
| Headline OC (D7) | **P(SD2-C2) ≈ 0.067** (0.046–0.099); P(SD2-SELF) ≈ 0.26; **P(SD2-C2 \| no effect = H_shift) ≈ 0.031** (0.022–0.064 for lineage σ 0.06–0.12); \| H_grow 0.14; \| H_full 0.24. **×2 ≈ 0.005** (0.001–0.013). LIT-C2-2 with SD2-SELF ≈ 0.010. Cross-checks: RT 0.064, PROOFS 0.070; ×1.5 GROW shift (sensitivity only) 0.077 [DERIVED v1_oc] |
| Cost | Projected **$0.69–1.27 per R2 cell**; family (s1 R2 + s2 lineage) **$2.9–5.0**; **family cap $6.5** (guards sum $6.45). Realised at registration: $0 |
| Approval | **The user's explicit OK** (MEM-TRADED rule), asked **after** the §215 pair is scored (D12), for the whole family in one decision (§2.4). Plus Audit's gates |

---

## 1. Change table (every review finding, with source)

The draft's §1 rows against SELFDEPTH_v1 §9 stand unless changed below: W_2 n 13–16 (row 1), the FT-SELF labeller with GATE-2 + GATE-1′ (rows 2–3), K law (row 5), extrapolation n 17–24 with DBS-2 on n 17–20 (row 6), eval_deep (row 7), micro 4 (row 8, re-derived), the round-2 condition met (row 10), the MEM-TRADED rule (row 12), S1-C unfunded (row 14). Changed: row 4 (mixture → D10), row 9 (cost), row 11 (second seed → D1), row 13 (×1.5 shift → sensitivity only), N1–N5 (below).

### 1.1 Theory's decisions

| # | decision | applied in |
|---|---|---|
| D1 | Second seed = option (b), lineage `s216_d3t_s2` → GATE-1-s2 + pool → `s216_r1_s2` → GATE-2-s2 + pool → `s216_r2_s2`. **RT B1 fixed:** per-lineage cell readers (`r1_cell(lineage)`, `r2_cell(lineage)` read the lineage's own base, gate, pool and selfdepth paths); ARGS-DIFF reference per round with `seed` allowed for s2; every s2 mechanism / H bar from the **same formula** on the s2 lineage's own measured curves (s1's numbers are the formula's s1 instance). **RT B2 fixed:** COMPETENT-BASE = the 48k run completed ∧ acc_n2 ≥ 0.99 ∧ a_24 ≥ 0.97 at n 1–8; any later s2 failure fails the ×2 row | §2.4, §4.3, §4.5, §4.6; formulas in v1_panels.py |
| D2 | N5 / G-S18 baselines (eval_deep on D3-T s1, FT-SELF, FT-CTRL) are read **only after `s216_r2_s1` has launched**. FT-SELF's is required (H_shift reference at n 21–24, descriptive); D3-T's and FT-CTRL's optional | §2.2 item 6, §7 G-S18 |
| D3 | eval_deep at eval_bs 16 (G-S16 decides; bs 8 pre-stated fallback). Every reading touching any n ≥ 17 takes **all** its n and K′ values from eval_deep (STOP-LOC-2, n50, n90, WIDTH-2, M2, M_far, KB2, LIT-C2-2, C2-TAIL-2, mechanism). n ≤ 16-only readings stay on eval_vb (bs 32). BS-CHECK: a(n) **and** a_k(n) at every shared K′, n 13–16, max \|Δ\| and item flips, descriptive, beside every eval_deep reading | §4.0, §4.2, §4.4 |
| D4 | T1 yes (self auto-micro tries {1, 2, 4, 8}; G-S7 plain-path check required). T2 no. **No checkpoint evals** (RT M4's 4k / 8k learning curve is not run) | §2.6, §7 |
| D5 | MEM sub-label: neutral names, **POST HOC**, descriptive, never used to soften MEM-TRADED in the user's OK framing; KEEPS-SELF-STAB-2 unchanged. Per-item store for hist24 = eval edit **E4** + gate G-S19 | §4.4, §5.3, §2.6, §7 |
| D6 | R2-CTRL: never by default, not offered | §2.4 |
| D7 | OC headline = the red team's shaded priors, recomputed exactly (v1_oc.py) after D9 / D10; lineage noise in P(pass \| no effect) (σ 0.06–0.12); the K′24-STABLE ⟂ ABSORB-2 independence dropped (dose-coupled conditional, bounded); FT-DIVERGED in the OC; ×1.5 GROW shift only as a sensitivity row; PROOFS' re-shade 0.070 as a cross-check | §6 |
| D8 | K′24-STABLE unchanged (a_24 ≥ 0.99 at n 1..8) | §4.4 |
| D9 | s1 thresholds kept (DBS-2 0.80, MARGIN-GROWS-2 0.785, SHRINK < 0.60, FULL M_far ≥ 0.60). DBS-2 / MARGIN-GROWS-2 test "the margin grew" (growth family), **not H_prop**; H_prop stays a named hypothesis with its own descriptive discriminator **WIDTH-2 = n50 − n90**. **LIT-C2-2** = mean_{n 22..24}[a(n) − a_20(n)] ≥ 0.10 added as the C2-literal row, on SD2-C2's line, not inside the SD2-C2 conjunction. KB2 kept but **descriptive** ("mostly passes ≤ 20"). Margin/B constancy holds only at the 50 % level (stated) | §3.4, §4.3, §5.1 |
| D10 | Self mixture **2/3 n 13–16 / 1/3 n 9–12** (deviation from v1 §9's 50/50), justified by PROOFS' exposure correction; by pool composition only (no trainer change); counts sized for 383,488 consumed with a stated margin; one fixed interleaved order; S1 table, G-S3 target, memory / time / cost recomputed | §2.3, §3.2, §3.5, §3.6, §8; v1_support_d10.py, v1_memtime.py |
| D11 | Seed 1 for `s216_r2_s1`, with the disclosure that its plain stream repeats round 1's plain batches exactly | §2.3 |
| D12 | Branch P2 (FT-CTRL STOP-MOVED) halts SD2. The §215 pair (incl. MEM attribution) is scored **before** the user is asked; this text states only the pair-result dependence of the wording | §2.5 |

### 1.2 Red team (REDTEAM_SD2.md)

| # | finding | v1 | verdict |
|---|---|---|---|
| B1 | s2 scoring path voids (ARGS-DIFF `seed`; s215 paths) or mislabels (s1-anchored bars) | D1: `r1_cell` / `r2_cell(lineage)`; ARGS-DIFF references and allowed sets per round (§4.6); bars by formula (§4.3). Formulas reproduce s1's 0.63 / 0.40 (round 1) and 0.785 / 0.60 (round 2) [DERIVED v1_panels, asserted]. R1-s2 fixture in G-S12 | accepted |
| B2 | ×2 drops failed s2 stages from the denominator | D1: only a **completed** base below the accuracy floor reads "not competent (C6 data point)". A dead base (incomplete 48k, guard stop, NaN), a failed GATE-1-s2, GATE-2-s2 with both labellers, any R1-s2 / R2-s2 censor, and R1-s2 failing its round-2 condition each **FAIL** the ×2 row. A rerun of a dead base needs the user's OK and is disclosed | accepted. Reading of D1: "completed (not dead)" makes a dead base a ×2 failure, not "not competent" (dead = fail, rule b); see §9 item 1 |
| M1 | K′-BEYOND-2 is not a K′ > K reading | KB2 descriptive ("pass gain at out-of-support depth, mostly passes ≤ 20"): panels 0.155 / 0.29–0.32 / 0.51–0.56 for H_shift / H_prop / constant growth, of which passes > 20 give −0.007 / −0.008 / +0.001 [DERIVED v1_panels]. LIT-C2-2 added (D9) | accepted |
| M2 | H_prop is a one-level coincidence; DBS-2 does not test it | D9 rewording (§0, §3.4, §5.1); WIDTH-2 descriptive; "0.50 face-value" sensitivity dropped | accepted |
| M3 | One lineage: lineage noise, not SE 0.006 | §3.4 discrimination table by σ; §6 P(pass \| no effect) with σ 0.06–0.12; wording "an instance", mechanism "consistent with" (§5.1) | accepted |
| M4 | New-window dose ≈ 0.85× round 1's absorption point, not 2× | "2×" dropped. Under D10 the new window gets 63,915 per n = 1.38× round 1's sustained CE < 0.02 point (46,416) and 0.86× its < 0.01 point (74,176). P(ABSORB-2) 0.65 (RT 0.60 was set at the 50/50 dose; sensitivity 0.60 / 0.70). **¬ABSORB-2 follow-up pre-stated** (§5.2). The 4k / 8k checkpoint evals are not run (D4) | accepted in part (checkpoint evals rejected by D4; P(ABSORB-2) re-shaded for D10) |
| M5 | Censors pass when their input is missing | score_s216 returns **pending** while any registered file is missing (the 215-7 rule) and never reads a missing eval_selfdepth / gate file as a pass; after the chain is declared done (`--final`), a missing file fails by rule b (§4.6). One fixture per missing file | accepted |
| M6 | MEM "robust / lost" is a post-hoc rescue | D5: neutral names, POST HOC, descriptive, out of the user-OK framing; the raw arms are shown to the user (§5.3) | accepted |
| M7 | No per-item store behind KEEPS-SELF-STAB-2 | E4 + G-S19. **Scope:** every arm score_s213.hist24 reads (intact, bank_ablate_slva, self_ablate, other_ablate, dup_newest, dup_foreign) at n 3..8, the argmax answer at **both K′ n + 2 and K′ 24** plus y and the test-order index. D5 named the K′ 24 arms at n 3..6; STAB_arm is a difference of (a_{n+2} − a_24) terms and the hist24 gate reads n 3..8, so the wider set is the minimum that lets Audit recompute the registered reading | accepted (scope widened to what the rule reads) |
| m1 | P1 already decidable from CTRL's eval_vb (−0.021) | Recorded as RT's derived value. The formal PAIR reading is score_s215's, pending until FT-CTRL's registered files land (215-7). D12 | accepted |
| m2 | N5 scope | D2 | accepted |
| m3 | One file per K′-grid reading; per-item BS-CHECK | D3; score_s216.deep_read asserts it opens only eval_deep.json; G-S12 fixture (eval_vb differing by 0.01 at n 13–16 leaves STOP-LOC-2 / n50 unchanged) | accepted |
| m4 | H_T power, threshold, confound | T split at **T_train** = max T of the consumed n-16 records (pool_meta), not a fixed 61 / 63. The ≥ 100-items-per-band rule qualifies at about n 20–21 only (stated). Within-n T tracks bracket structure, so the flag is descriptive. Added: M2 on items with T ≤ T_train (descriptive) | accepted |
| m5 | K′24 granularity; FT-DIVERGED missing from the OC | §6: P(K′24 \| ABSORB-2) 0.55 (binomial floor 0.583 at a true n-1 error 0.010; 415 distinct n-1 items); FT-DIVERGED ≈ 0.12 given ABSORB-2, without double counting its a_24 < 0.97 clause | accepted |
| m6 | s2 gating | R2-s2 runs whatever R1-s2's mechanism label (overrides v1 §9's "SHIFT → optional"; disclosed). The user's OK is asked for the whole family, explicitly covering an R1-s2 MEM-TRADED reading (§2.4) | accepted |
| m7 | D3-T s2 needs its own resume smoke | G-S4b (§7) | accepted |
| m8 | pool_r1 retention | Keep `s215_pool/pool_r1.jsonl` until every s216 pool is built and G-S1-verified (its syms enter the exclusion set); then delete per the retention rule | accepted |
| m9 | eval_deep contract details | items_sha256 serialisation fixed (§4.2); scorer asserts card class (3070), dtype fp32, tf32 off; per-item test-order index stored | accepted |
| m10 | Admission headroom; refusals; 3070 queue | R2 admission ≤ **$1.45** (14.6 % above the projected upper $1.27), guard **$1.70**; family cap unchanged $6.5 (guards sum $6.45). "Not admitted → not run, user told; the cell's readings read UNREAD (never a pass)". 3070 queue ≈ 19–22 h for the family (§8; RT 20–25 h) | accepted |
| m11 | Fallback wording | BASE-LABELLED wording pre-stated (§5.1) | accepted |
| m12 | testset_ext read generator statistics of n 17–24 | Disclosed in §2.2 item 6 | accepted |
| m13 | Seed 1 replays round 1's plain stream | D11 disclosure (§2.3) | accepted |

### 1.3 Proofs (PROOFS_SD2.md; PASS rows unchanged and not listed)

| # | finding | v1 |
|---|---|---|
| 2 | margin/B constant at the 50 % level only (0.134 → 0.161 at 90 %, 0.212 → 0.229 at 70 %) | stated in §3.4 (D9) |
| 4 | H_shift G(19) 0.023 (not 0.025) | §3.4 [DERIVED v1_panels: 0.0231] |
| 6 | H_prop n50 20.61 (not 20.63) | §3.4, §4.3, G-S12 [v1_panels: 20.614] |
| 8 | H_full floors M2 ≥ 0.9375, M_far ≥ 0.8375 | §3.4 |
| 10 (FAIL) | H_prop ≠ constant +1.15 growth except at the 50 % crossing | §3.4 replaced by PROOFS' fix text; constant growth (H_add) is its own panel |
| 11 | "DBS-2 is a direct test of H_prop" | D9 rewording |
| 12 | the extension caveat also covers H_prop n 22–24 / M_far and H_shift's STOP | §3.4 |
| 14 | G-S3 SE; the replay makes it precomputable | §3.2, G-S3: target **1/2** under D10, precomputed from pool_r2 + round-1 U's, then checked |
| 15 | CP convention | one-sided 95 % throughout §3.3 (two-sided stated beside where score_s215 uses two-sided Wilson CIs) |
| 16 | direct-w κ form | §3.3: κ ≥ 5.3 (W_2; 1.7 at w's upper); pool form 7.9 / 2.0 under D10's mix (PROOFS' 10.7 / 3.4 at 50/50) [DERIVED v1_noise_d10] |
| 17 | two-round factor | 1.31 dropped; 1.55 point / 2.68 upper / 1.43 direct-w kept |
| 18 | class mass | 2(1 − π) = 1.0e-3 (≤ 3.2e-3 at π's 95 % lower); cumulative lineage bound ≈ 1.8e-3 |
| 20 | assumptions for the first compounding round | A4 (per-round φ_q, P_q), A5 (no error-class reinforcement; cumulative class bound) added; **A3′ unverified on FT-SELF** (tagged [ASSUMED]) |
| 21 | GATE-1′ OC | §2.1: 0.9983 / 0.481 / 0.0105 |
| 22 | π^E term recomputed | §6: 0.048 at π^E 0.988, 0.504 at 0.990, 0.9987 at 0.993 |
| 23 | kept counts are KEEP-only projections | stated (CV unmeasured on FT-SELF); OC at CV = 0.98 × KEEP 0.9984 / 0.511 / 0.014 |
| 24 | LABEL-AUDIT-2 OC | recomputed under D10's window split (§6) |
| 27 | peak T = the longest record (61; 63 in 1 of 5 seeds) | §3.5: micro 4 at **14.4–15.0 GiB**; re-checked on 77,325 n-16 draws per seed (D10's count) [DERIVED v1_memtime] |
| 30 | compute-bound refit at G-TIME shapes | §3.6: recomputed for D10's chunks (mean max-T 52.2): 1.76 s, $0.69–1.06; launch-bound $0.83–1.27 |
| 31 | per-micro-batch Kb | §3.6: under D10 at micro 4, P(Kb = 20) 0.42, E[Kb] 19.20 |
| 33 | the 3090 eval check is "consistent with", not "reproduces" | §3.7 |
| 38 | T-split wording | §3.1(c): "no significant pooled within-n T effect (p 0.13); n 15 nominally negative (−0.066, z −2.1) on untrained rows"; the H_T flag rule kept (descriptive) |
| 39 | exposure deterministic under the replay | §2.3, recomputed for D10 |
| 40 (FAIL) | "2× round 1's absorption point" | dropped; D10 numbers (§2.3) |
| 42 | K′24-STABLE dose-coupled to ABSORB-2 | §6 (D7) |
| 43–44 | family $; R1-s2 hours | §8 ($2.9–5.0; R1-s2 2.6 h) |
| 45 | single source for readings touching n ≥ 17 | D3 (§4.0) |

### 1.4 New in v1 (not in any review)

| # | item | why |
|---|---|---|
| V1 | Bar formulas round **up to the next 0.005** (ceil005) | reproduces all four registered s1 bars exactly (0.63, 0.40, 0.785, 0.60); v1 §4.2's 0.40 had no stated derivation, so its formula (H_shift − 0.08) is pre-stated here so that its s1 instance equals the registered number |
| V2 | FULL needs M2 ≥ max(0.80, MARGIN-GROWS-2) ∧ M_far ≥ 0.60 (absolute); a measured-only flag "FULL not separable from H_prop" when H_prop's M_far upper bound ≥ 0.60; degenerate bars (SHRINK bar ≥ MG bar) → UNLABELLED (bars degenerate) | keeps the partition exhaustive for any s2 curve without reading unread n > 16 values |
| V3 | R1-s2's eval_deep (after R2-s2 launches): required, descriptive | the s2 analogue of FT-SELF's baseline (D2) for the s2 H_shift panel at n 21–24 |
| V4 | BS-CHECK item flips from E2's per-item store | eval_vb keeps per-n aggregates only, and its harness stays unchanged |
| V5 | The user's OK is asked once for the whole family | avoids an outcome-dependent second ask between R1-s2 and R2-s2 (m6) |

---

## 2. Design (registration text)

### 2.1 Labeller and gates

- **Chain, KEEP, COMPOSE-VALID (CV), kept, R2 targets:** exactly as SELFDEPTH_v1 §2.1 (`boxes/selfdepth.label_chains`, `compose_E`). Kept = KEEP ∧ CV. R2 targets HOLD-padded to the batch Kb.
- **Labeller (pre-stated by the round-1 §9 rule, never chosen on pool or test items):** the lineage's round-1 model. s1: `boxes/runs_fsq/s215_ftself_s1/model.pt` (sha256 in pool_meta.json). s2: `s216_r1_s2/model.pt`. Local 3070, fp32, bs 32, step cap n + 4. The same labeller labels both windows.
- **Why FT-SELF** [MEASURED s215_ftself_s1 / s213_d3t_s1 eval_chainlabel_fresh.json, 1,000 fresh items per n]:

  | n | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | FT-SELF keep | 1.000 | 0.999 | 0.999 | 0.999 | **0.997** | **0.989** | **0.986** | **0.957** | 0.882 | 0.751 | 0.518 | 0.361 |
  | D3-T keep | 1.000 | 0.996 | 0.988 | 0.976 | 0.972 | 0.952 | 0.943 | 0.917 | 0.892 | 0.839 | 0.809 | 0.781 |
  | FT-SELF prec_kept | 1 | 1 | 1 | 1 | 1 | 1 | 0.998 | 1 | 1 | 1 | 1 | 1 |

  FT raised the single step's reach at n ≤ 16 and narrowed it at n ≥ 18 (coverage, not precision: 0 kept-wrong of 2,512 kept chains at n 17–20). That matters for a round 3, not for SD2. The Literature note's "n 16 keeps about a third" used the loop's accuracy; the yield is the chain's keep, 0.957.
- **GATE-2 (n 13–16) and GATE-1′ (n 9–12).** 2,000 held-out fresh items per n; s1 rng default_rng(22660) / (22661), s2 (22860) / (22861); §2.2 exclusion; never trained; kept = KEEP ∧ CV; scored against gold in a separate process. Rule per window (= GATE-1's): **π_pool ≥ 0.995 ∧ π_n ≥ 0.99 at every n ∧ π^E_pool ≥ 0.99 ∧ keep_n ≥ 0.70 at every n**.
  - Exact OC [DERIVED, PROOFS p3_noise_gate]: GATE-2 at FT-SELF's projected kept counts (1,994 / 1,978 / 1,972 / 1,914; c_n 19, c_pool 39): **P(pass \| π 0.997) 0.9987, \| 0.995 0.524, \| 0.993 0.014**. GATE-1′ (2,000 / 1,998 ×3): 0.9983 / 0.481 / 0.0105. Same OC as GATE-1 (0.9986 / 0.504 / 0.0125).
  - The kept counts are KEEP-only projections; KEEP ∧ CV is unmeasured on FT-SELF (CV = KEEP was measured on D3-T only). At CV = 0.98 × KEEP the OC is 0.9984 / 0.511 / 0.014.
  - Measured window π 0.99949 (2 kept-wrong / 3,929, both at n 15), so P(pass) ≈ 1.
  - Descriptive at GATE-2: **KEPT-VS-DRAWN** (height / T / depth of kept vs all drawn at n 16; Literature failure mode 1).
- **Fallback per window (pre-stated):** if a window's gate fails under the round-1 model, the lineage's **base** (s1: D3-T s1; s2: D3-T s2) labels that window, after passing the same gate on the same gate items (separate eval_gate entry). Both labellers fail a window's gate → no round 2 in that lineage (no cell, no spend). The wording then changes (§5.1).

### 2.2 Contamination rules (registration text)

1. **Training items are fresh generated items only.** s1 round-2 pool: `nested170.generate(rng, n)`, rng = default_rng(22600), drawn until **74,000 kept per n at n 13–16 and 37,000 per n at n 9–12** (444,000; D10); one fixed interleaved order, default_rng(22601) permutation. Fresh-once: exhaustion or reuse = void.
2. **Exclusion by exact rendered syms** against: test_set(1000, **n_max 24**) (24,000 items); SERIAL (rng 17300, 2,200); the 213-37 fresh items (rng 21337, n 9–20, and n 21–24 if E3 extends them); GATE-1 (s215_gate1/gate_items.json), GATE-2, GATE-1′; **all 440,000 round-1 pool records** (s215_pool/pool_r1.jsonl, kept until every s216 pool is built and G-S1-verified, m8). Per n: draws, collisions and duplicates logged. Expected 0 at n ≥ 9 [PROVED, PROOFS §8: P(render = s) ≤ (1/11)·40⁻ⁿ; with this exclusion set ≤ 2e-6 at n 9].
3. **No gold reaches training** (as SELFDEPTH_v1 §2.2 item 3): builder gold patches before generation; record keys exactly {syms, n, chain_label, steps}, asserted at load; the SD-LABEL-MAP grep; no `s216_gate` string in boxes/vbank.py.
4. **Gold is used only for evaluation, the gates, LABEL-AUDIT-2, MEMO-GAP-2 and eval_deep**, each in a separate process.
5. **Round 1's extrapolation items (test n 13–16) are round 2's in-support test items**, excluded from every pool. Their round-1 readings (FT-SELF a(13..16)) were read before this text and inform only the predictions and the formula bars (§4.3).
6. **n 17–24 of test_set(1000, n_max 24) is SD2's extrapolation set.**
   - Before SD2's registration commit it is used only as an exclusion list. **Disclosure (m12):** the draft's testset_ext.py read the T lengths and y-class shares of the n 17–24 items (generator statistics, no model output).
   - **After `s216_r2_s1` has launched (D2)** it is read on FT-SELF (required) and on D3-T s1 / FT-CTRL (optional) as descriptive baselines (G-S18), and on `s216_r1_s2` after `s216_r2_s2` has launched (V3). No threshold, labeller choice, launch, stop or go / no-go of any SD2 cell depends on them.
7. **The labeller is fixed in advance** (§2.1). Pool, GATE-2 and GATE-1′ use the same card class and settings (3070, fp32, bs 32).
8. **Post hoc:** pool ∩ (all of item 2) = ∅ is re-verified on the pool file. Any hit, gold key, reuse (`self_reuse` > 0) or exhaustion → **CONTAMINATED / void**.
9. **s2 seeds:** round-1 pool 22700 / 22701, GATE-1-s2 22760, audit 22770; round-2 pool 22800 / 22801, GATE-2-s2 / GATE-1′-s2 22860 / 22861, audit 22870. s2's exclusion set is item 2's plus every s216 pool and gate that exists at build time (its round-2 pool also excludes its round-1 pool), test n_max 24 from the start. Every seed named here (22600 / 22601 / 22660 / 22661 / 22670, 22700 / 22701 / 22760 / 22770, 22800 / 22801 / 22860 / 22861 / 22870) is unused in docs/THEORY.md, boxes/*.py, theory/*.py and scratch/*.{py,sh} [re-checked by grep at 6c49b973, 0 hits].

### 2.3 Cells and training (s1 lineage)

| element | spec |
|---|---|
| Init | `--init_from boxes/runs_fsq/s215_ftself_s1/model.pt`: strict load, missing = [] (asserted, vbank.py l.666), calibrate() not called, `cal` = FT-SELF's calibration.json (= D3-T's), fresh AdamW |
| Recipe | **s215_ftself_s1/args.json** with only `--init_from`, `--self_pool boxes/runs_fsq/s216_pool_s1/pool_r2.jsonl`, `--self_kmax 20`, `--self_micro <G-S9>` and `--out` changed. Unchanged: lr 3e-5, 12k steps, warm-up 100 + cosine, cur 8, c_sched 0,1, p_self 0.5, self_kextra 4, kmax 12, kextra 4, bs 64, **seed 1**, draws_log_steps 20000, ckpt_every 2000, mem_budget 21.0 |
| Seed 1 (D11) | rng_s = default_rng(seed + 21800) is consumed by one coin per step and 64 U per self step in both rounds, and the plain draw (rng, rng_k) happens first every step [VERIFIED vbank.py draw_step l.738–770]. So round 2 replays round 1's plain batches, coins, self step indices and self U values exactly, with new pool records. **Disclosure:** the 383,488 plain-step items (n ≤ 8) are seen a second time; plain n ≤ 8 is saturated (a(n) = 1.000 at n 1–8 on FT-SELF [MEASURED eval_vb]), so no reading depends on new plain items. G-S3 checks plain_hash identity at steps 1..200 |
| CurGate | Moot: `--cur 8` is a single stage, CurGate.update returns None at the final stage (vbank.py l.557), and self steps are excluded (l.812) [VERIFIED]. plain_answer_ce is logged (l.827). The 214-5 lesson is met by construction |
| Self law (D10) | Pool order uniform within each window, 2 : 1 by window. P(R pass \| t ≥ t*) and deadline costs: §3.2 (**1/2 at t* 13, 0 at t ≥ 17**) |
| Exposure (D10) | Exactly round 1's **5,992 self steps → 383,488 items** (seed-1 replay; PROOFS #39). Expected per n: **63,914.7 ± 85.2 at n 13–16, 31,957.3 ± 63.2 at n 9–12** (hypergeometric, random-permutation order); new-window share 2/3 ± 0.0003. Round 1's self_answer_ce held < 0.02 from step 5.8k (46,416 items per n) and < 0.01 from 9.3k (74,176 per n) [MEASURED log.jsonl, PROOFS #40]. So the new window gets **1.38× / 0.86×** those points (the 50/50 law gave 1.03× / 0.65×). The old window gets 1/3 of round 1's 95,872 per n; RETAIN guards it [DERIVED v1_support_d10]. The logged self_answer_ce pools both windows (T2 not built), so the n 13–16 CE curve is unread |
| Pool size | 444,000 records; **60,512 spare** after the deterministic 383,488. For a non-replayed seed (s2, Binomial(12000, ½) self steps, 384,000 ± 3,505 items) z(exhaust) = **17.1** [DERIVED] |
| Micro | plain 1; self from G-S9 over {1, 2, 4, 8} (T1; projected **4**: 14.4 GiB at T 61, 15.0 at T 63) |
| Logging | round 1's fields (step, answer_ce, plain_answer_ce, self_answer_ce, self_rec_ce, self_frac, pool_cursor, n_self_steps, self_reuse, c); draws_log.json steps 1..20,000 |
| Checkpoints | Full state every 2k (model, AdamW, all four rngs, cursor, step, elapsed, n_nonfinite, calibration), mirrored off-host, sha-checked after the first copy; the resume is tested before launch (G-S4 / G-S11). **No checkpoint evals** (D4). Checkpoints and mirrors deleted at landing; model.pt kept (the round-3 labeller) |

### 2.4 The second lineage (D1, option b)

| stage | spec | gate / reading |
|---|---|---|
| `s216_d3t_s2` | **s213_d3t_s1/args.json with only `seed 2` and `out` changed**, 48k, RTX 3090, no init_from (calibrate() at step 0) | ARGS-DIFF vs s213_d3t_s1 allowing {seed, out} ∪ META; resume smoke G-S4b; basin entry by score_s211.entry_on, reported with the §187 confound rule vs s1's 11k (descriptive) |
| COMPETENT-BASE | evaluated **only on a completed 48k run**: acc_n2 ≥ 0.99 **and** acc_k[24] ≥ 0.97 at every n 1–8 (eval_vb). VB-C1 descriptive | completed and below the floor → **"s2 base not competent (C6 data point)"**: the only exclusion. Not completed (dead, guard stop, NaN, host loss) → **×2 FAIL** (rule b) |
| GATE-1-s2 + round-1 pool | D3-T s2's chain; 2,000 per n 9–12 (rng 22760); pool 110,000 kept per n 9–12 (rng 22700 / 22701; 440k, z 16.0) | GATE-1 rule. Fail → **×2 FAIL** |
| `s216_r1_s2` | **s215_ftself_s1/args.json** with only `init_from s216_d3t_s2/model.pt`, `self_pool`, `self_micro`, `seed 2`, `out` changed | `r1_cell("s2")`: v1 §4–§5's readings with the lineage's paths and the round-1 bar formulas (§4.3). Round-2 condition (v1 §9): ABSORB ∧ ¬FT-DIVERGED ∧ LABEL-AUDIT ∧ ¬CONTAMINATED ∧ ¬dead ∧ ARGS-DIFF ok. Fail → **×2 FAIL** ("lineage stopped at round 1"; its round-1 label is reported) |
| GATE-2-s2 + GATE-1′-s2 + round-2 pool | R1-s2's chain; §2.1 rule; D10 composition and counts (444k) | both labellers fail a window → **×2 FAIL** |
| `s216_r2_s2` | s1's R2 recipe with `seed 2`, `init_from s216_r1_s2/model.pt`, its own pool | `r2_cell("s2")` with the round-2 bar formulas on R1-s2's and D3-T s2's own eval_vb. Any censor → **×2 FAIL** |

- **R2-s2 runs whatever R1-s2's mechanism label** (overrides SELFDEPTH_v1 §9's "SHIFT → optional", so that the second lineage is not outcome-selected; disclosed, m6). It needs only the round-2 condition.
- **One user decision covers the family**, explicitly including an R1-s2 MEM-TRADED reading and R2-s2 under any R1-s2 label (V5).
- **A cell never admitted** (no host passes G-TIME admission within the allowed tries, §8) is not run and the user is told. Its readings read **UNREAD**, never a pass; ×2 then reads UNREAD.
- **No R2-CTRL** (D6).

### 2.5 Dependence on the §215 pair (scored before the user is asked; D12)

SD2's recipe, readings and thresholds do not change across branches. Only the wording and the KEEPS-SELF-STAB-2 prior move. Theory scores the pair with score_s215 once FT-CTRL's registered files exist (215-7: pending until then).

| pair result (score_s215 PAIR) | SD2 consequence |
|---|---|
| **P2 FT-EFFECT** (CTRL STOP-MOVED) | **SD2 halts.** Plain FT alone moved the stop; Theory re-plans. (RT m1 derived CTRL's acc_gain_after10 from its eval_vb at −0.021, so P2 is not expected; the registered reading is score_s215's.) |
| P1 SELF-ATTRIBUTED, MEM "traded by the self items" | The user's OK question says: round 1's own-slot share fell with the self items (STAB_self 0.782 → 0.489), not with the FT; a deeper round may trade more. KEEPS-SELF-STAB-2 prior 0.20 |
| P1, MEM "traded by the FT (lr re-heat)" | The OK question says: FT-CTRL shows the trade without self items; a second FT likely re-heats again. Prior 0.25 |
| P1, MEM "kept" in both (not expected: SELF read MEM-TRADED) | not applicable as stated; the question reports the raw arms |
| PAIR UNREAD (CTRL censored or its files never land) | The OK question says "MEM attribution unread". Prior 0.25 |

In every branch the user sees the raw hist24 arms (intact, bank_ablate_slva, self_ablate, other_ablate, dup_newest; FT-SELF's are in §5.3) and the registered label MEM-TRADED; the POST HOC sub-label (§4.4) is not part of the question (D5).

### 2.6 Edits needed (list only; none written; Audit's diff review G-S13)

| # | where | edit |
|---|---|---|
| T1 | boxes/vbank.py l.706, l.719 | self auto-micro loop `(1, 2, 4)` → `(1, 2, 4, 8)` and its error text. G-S7 plain-path check required (D4). Optional: help-string "n 12" → "n_top" |
| B1 | scratch/s216_build_pool.py (new; from s215_build_pool.py) | labeller path; per-n counts {9..12: 37,000, 13..16: 74,000} (s2 round 1: {9..12: 110,000}); seeds §2.2; exclusion incl. test n_max 24 and the round-1 pool syms; GATE-2 + GATE-1′ items; pool_meta adds labeller_sha256, per_n, excl breakdown, window tags, T_max_by_n; `--lineage s1|s2 --round 1|2` |
| B2 | scratch/s216_gate2.py | GATE-1's scorer generalised to a window list and a labeller list (fallback); eval_gate.json with by_n 9..16 + `window` + `labeller`, per-item kept / label / gold / state-ok |
| E1 | boxes/vbank_diag.py `part_deep` (or boxes/vbank_deep.py) | §4.2 contract |
| E2 | vbank_diag.py `part_selfdepth` | parameterised by lineage and round: pool / gate / cursor paths (today hard-coded to s215, l.1377–1410), test n_max 24, memo n 9..16, label_audit by window, contamination vs test n_max 24 and the round-1 pool; **items_n2 and items_k24 per item at n 1..16** plus T and the test-order index. Round-1 behaviour bitwise-unchanged with round 1's arguments |
| E3 | vbank_diag.py `part_chainlabel_fresh` | n 9..20 → 9..24, the same rng 21337 consumed sequentially (n 9..20 stays bitwise); descriptive LABELLER-KEPT-2 |
| **E4** | vbank_diag.py `part_hist24` (D5 / RT M7) | add `items` {arm: {n: {"ans_n2": [...], "ans_k24": [...]}}} for every arm at n 3..8, plus `y` and the test-order index per n. The existing per-n means are unchanged (G-S19 checks they equal the stored items' means) |
| S1 | theory/score_s216.py (new) | §4.6 |
| Q1 | scratch/s216_*.sh | queue / fill / post-landing chains; drain waiters armed at launch; eval chains pinned to the 3070 |

T2 (per-window self CE logging) is not built (D4).

---

## 3. Mathematics

### 3.1 S0, support identity for round 2 [PROVED; SELFDEPTH_v1 §3 / PROOFS_SELFDEPTH §1 with B_2 = 16]

- Every target state of the round-2 mixture has ρ + r = n ≤ 16: CV ∧ STEPS gives one VOIDed handle per step for n steps, then HOLD; values never enter the count. The 2 : 1 window mix (D10) changes weights, not support.
- (a) "Stop at ρ ≥ 16" and any pass deadline t_d ≥ 17 are zero-loss on the support. **H_shift is still irreducible by loss**, one boundary further out.
- (b) ABSORB-2 (n 13–16) is in support; n 17–24 is out of support. n 19–24 are read at K′ n + 2 = 21–26, above the trained max K 20 [PROVED: vb_loss windows, PROOFS_SD2 §2]. At n ≤ K + 1 = 21 a pass gain past K cannot be non-zero by saturation (a_n(n) ≈ a(n) [MEASURED FT-SELF eval_vb]); **literal C2 needs n ≥ K + 2 = 22** (RT M1), hence LIT-C2-2 on n 22–24.
- (c) Positions: round 2 trains rows up to **T_train** = the max T of the consumed n-16 records (61 in 4 of 5 generator seeds, 63 in 1 [DERIVED v1_memtime, 77,325 draws per seed]); eval_deep reads up to T 83 at n 24 [MEASURED testset_ext.json]. Rows above T_train are pretrained GPT-2 rows the loop never trained (only weight-decayed).
  - Round-1 precedent (FT-SELF, trained T ≤ 47; read rows to 59 at n 15–16): **no significant pooled within-n T effect** (stratified permutation p 0.13; pooled slope −0.0032 per row), but **n 15 is nominally negative on the untrained rows** (T ≥ 49: −0.066, z −2.1; n 14 −0.055, z −1.4; n 16 +0.033, z +0.9) [DERIVED PROOFS p8_testset_tsplit]. Round 1 does not rule out a row-keyed component.
  - The nuisance hypothesis **H_T** (failure keyed on rows > T_train) is read by NOT-SIZE-KEYED-2 (§4.4), descriptive.
- (d) The rollout has no pass cap: the bank is a Python list, SLVA's age bias is γ·age with β0 on V_0, no table is indexed by pass [VERIFIED vbank.py l.262–345, slva.py l.69–73; PROOFS #36]. K′ 28 is admissible.

### 3.2 S1 under the D10 law [DERIVED, v1_checks/v1_support_d10.py (PROOFS_SD2's exact-rational construction with the self law replaced; its 50/50 run reproduces p2_support_s1.json, P(R \| t ≥ 13) = 5/11, asserted) and v1_support_check_d10.py (PROOFS_SELFDEPTH's own support_check `mixture()` run on the D10 law: identical to 1e-12)]

| t* | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | ≥ 17 |
|---|---|---|---|---|---|---|---|---|---|
| P(R pass \| t ≥ t*), D10 | 0.668 | 0.646 | 0.614 | 0.567 | **0.500** | 0.400 | 0.283 | 0.145 | **0** |
| deadline cost at t_d (share of an item's R2 term, loss-weighted), D10 | 16.4 % | 13.0 % | 10.0 % | 7.3 % | 5.0 % | 2.9 % | 1.4 % | 0.47 % | **0** |
| for reference, 50/50 law | 0.637 / 14.8 % | 0.605 / 11.2 % | 0.566 / 8.2 % | 0.517 / 5.7 % | 0.455 / 3.7 % | 0.375 / 2.2 % | 0.273 / 1.1 % | 0.143 / 0.35 % | 0 / 0 |

- **G-S3 target: P(R \| t ≥ 13) = 1/2 exactly**, 0 at t ≥ 17. Only self items contribute at t ≥ 13 (plain K ≤ 12), so the target is plain-law-free. Delta-method SE 0.0004 over 383,488 self items. Under the seed-1 replay the realised value is fixed by pool_r2's order and round 1's U values, so it is **precomputed before launch** (pool_r2.jsonl + s215_ftself_s1/draws_log.json) and then checked on the run's own draws_log.
- D10 raises the cost of a stop at the boundary: a deadline at t_d 16 now costs 0.47 % of an item's R2 term (0.35 % at 50/50), at t_d 15 1.4 % (1.1 %). Still a weak gradient against stopping exactly at the boundary, which is the H_shift mechanism.

### 3.3 S2, label noise in the first compounding round

Round 2 is the first round whose labeller was itself trained on self-labels (round-1 LABEL-AUDIT: 2 / 5,000 kept-wrong, π_state 0.9996 [MEASURED score_s215.json]). Confidence bounds below are one-sided 95 % Clopper–Pearson; score_s215 / GATE files use two-sided Wilson intervals, stated where quoted.

- **(a) Direct counts** [DERIVED from MEASURED eval_chainlabel_fresh, v1_noise_d10]:
  - FT-SELF chain at W_2 = n 13–16: **2 kept-wrong of 3,929 kept** (both at n 15), π 0.99949, one-sided lower **0.9984** (two-sided 0.9982). W_1 = n 9–12: 0 of 3,997.
  - Pool under D10 (2/3 W_2, 1/3 W_1): w_pool = 3.4e-4, π 0.99966; expected kept-wrong among LABEL-AUDIT-2's 5,000: **1.7**.
  - D3-T (fallback): 3 / 3,784, π 0.99921, lower 0.9980. π^E ≤ π [PROVED, PROOFS_SELFDEPTH §6]; GATE-2 measures π^E directly.
- **(b) κ thresholds at n 16** [PROVED UNDER A1′, A2, A3′, A4, A5; PROOFS_SD2 §3 forms]:
  - φ form P/(φn): FT-SELF P(16) 0.957, φ̂ = 2/73 = 0.027 → **κ ≥ 2.2** (0.71 at φ's 95 % upper 0.084). D3-T 4.2 / 1.6.
  - Direct-w form (1 − P)/(n w): **κ ≥ 5.3** on W_2 (1.7 at w's upper); **7.9** on D10's pool mix (2.0 with each window at its upper; 10.7 at the 50/50 mix).
  - Per-n caveat: at n 15 alone both forms give κ ≥ 0.53 (2 of 16 non-strict kept). GATE-2's π_n ≥ 0.99 rule is the guard there (π_15 0.998).
  - **Two W-4 rounds, κ = 1, per-round φ:** ≤ **1.55** (point: φ_1 0.0049, φ̂_2 0.027), ≤ **2.68** (both 95 % uppers), 1.43 (direct-w form). The draft's "1.31 (φ 0.010)" is dropped: its premise φ_2 ≤ 0.010 is contradicted by FT-SELF's point estimate.
  - **Assumptions for the first compounding round.** A1′ (an FT on pool share w raises e at every m ≤ B_{q+1} by ≤ κw), A2 (clean labels do not raise it), A3′ (1 − P_q(n) ≤ n ē_q(n)), plus **A4** (per-round parameters: ē_Q ≤ ē_0 Π_q (1 + κ φ_q n_q / P_q) with each labeller's own φ_q, P_q; a better labeller can have a larger φ) and **A5** (no error-class reinforcement: round-2 errors add to, and do not multiply, the errors FT-SELF absorbed from round 1). **A3′ is unverified on FT-SELF** (it was supported on D3-T's long_step fit only) [ASSUMED for FT-SELF].
  - **The κ bound is not the guard; the direct measurements are** (GATE-2 / GATE-1′ at 2,000 per n, LABEL-AUDIT-2 at 5,000).
- **(c) Pattern-class mass** [PROVED, PROOFS_SELFDEPTH §3c]: a majority-wrong class has mass ≤ 2(1 − π) = **1.0e-3** at round 2's point π (≤ 3.2e-3 at π's 95 % lower). Under A5's failure mode the bound applies to the **cumulative lineage mass**, 2(1 − π_1) + 2(1 − π_2) ≈ **1.8e-3** (point). LABEL-AUDIT-2 compares its kept-wrong handle patterns with round 1's (score_s215 label_audit.kept_wrong) and with FT-SELF's 2 fresh n-15 kept-wrong chains.
- **(d) Empirical record:** round-1 FT on w_1 ≈ 4e-4 produced a labeller whose fresh step **improved** at n 13–16 (strict 0.997 / 0.989 / 0.984 / 0.957 vs D3-T's 0.972 / 0.952 / 0.940 / 0.917); its loss at n ≥ 18 is coverage with precision 1.000 [MEASURED]. No sign of compounding; the noise-only κ is not identified without a gold-labelled twin (stated, not claimed).
- **(e) Lee 2502.01612** (Fig 5): the avalanche starts when **unfiltered** self-label accuracy is below ~0.9. Ours: unfiltered strict ≥ 0.957 at n 16, filtered 1 − π ≈ 5e-4 (196× below 0.1); Lee's 5 % structured noise (Fig 24) is 98× our kept-wrong share. Biased errors hurt more (App B.4): (c)'s class bound.
- **LABEL-AUDIT-2 must show** π ≥ 0.99 and π^E ≥ 0.99 pooled **and per window**, each window's CI overlapping its gate's CI; else LABEL-NOISY-2 (censor).

### 3.4 S4: hypotheses and pre-stated predictions for n 17–24 [CONJECTURE for each H; DERIVED numbers, v1_checks/v1_panels.py]

**Round-1 inputs** [MEASURED eval_vb acc_n2; DERIVED crossings, PROOFS p1_panels]:

| | B | n90 (m, m/B) | n70 (m/B) | n50 (m, m/B) | width n50 − n90 |
|---|---|---|---|---|---|
| D3-T s1 | 8 | 9.07 (1.07, 0.134) | 0.212 | 10.31 (2.31, **0.288**) | 1.23 |
| FT-SELF s1 | 12 | 13.93 (1.93, 0.161) | 0.229 | 15.46 (3.46, **0.288**) | 1.53 |

**Margin/B is constant only at the 50 % level** (0.134 → 0.161 at 90 %, 0.212 → 0.229 at 70 %); the top of the curve moved out more than proportionally (×1.62–1.80 vs B's ×1.5), and the width grew ×1.24 (H_prop predicts ×1.5, a rigid translation ×1.0). Retro-test on round 1: H_prop has RMSE 0.051 with no free parameter, a rigid growth 0.026 with δ fitted on round 1's own n50, H_shift 0.261 [DERIVED PROOFS §1.2].

**Hypotheses** (d = n − 16; a_1 = FT-SELF's acc_n2, linear in n):
- **H_shift**: a_2(16 + d) = a_1(12 + d). Round 1's growth was one-off; the margin stays 3.46.
- **H_prop**: a_2(16 + d) = a_1(12 + 0.75 d). Margin/B stays 0.288 at the 50 % level; the transition widens by 16/12.
- **Constant growth (H_add)**: round 1's curve translated by 4 + δ, δ = m50_1 − m50_0 = 1.155. **For this +4 step H_prop and constant growth predict the same n50 (20.61), since 0.288 × 4 = 1.15, but different curves**: H_prop widens the transition, constant growth keeps its width (PROOFS #10, FAIL of the draft's "coincide" claim).
- **H_full**: no stop ≤ 24, stated as a floor.
- **SHRINK**: the margin falls back to the base's: a_2(16 + d) = a_0(8 + d).
- **H_T** (nuisance): failure keyed on rows > T_train, read by the T-split, not by M.
- a_1 at n 17–20 is unread; the assumed extension (excess over a 0.11 floor decaying at the last ratios' trend) is 0.180 / 0.124 / 0.110 / 0.110 [ASSUMED]. It enters H_shift at n 21–24 and STOP, **and** H_prop at n 22–24 (a_1(16.5), a_1(17.25), a_1(18)) and M_far (PROOFS #12). FT-SELF's eval_deep (G-S18) replaces it in the descriptive reference panel; no bar uses it.

| n | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | **M2** (17–20) | M_far | n50 | n90 | WIDTH-2 | STOP-LOC-2 | KB2 (desc.) | **LIT-C2-2** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H_shift | 0.999 | 0.893 | 0.633 | 0.344 | 0.180 | 0.124 | 0.110 | 0.110 | **0.717** | 0.131 | 19.46 | 17.93 | **1.53** | 20 (G(19) 0.023) | 0.155 | **0.005** |
| H_prop | 0.999 | 0.946 | 0.828 | 0.633 | 0.416 | 0.262 | 0.166 | 0.124 | **0.852** | 0.242 | **20.61** | 18.39 | **2.22** | 21 | 0.29–0.32 | **0.036–0.046** |
| constant growth | 1.000 | 0.999 | 0.909 | 0.673 | 0.389 | 0.206 | 0.133 | 0.112 | **0.895** | 0.210 | 20.61 | 19.04 | **1.57** | 21 | 0.51–0.56 | **0.030–0.038** |
| H_full (floor) | ≥ 0.97 | ≥ 0.95 | ≥ 0.93 | ≥ 0.90 | ≥ 0.88 | ≥ 0.85 | ≥ 0.82 | ≥ 0.80 | ≥ **0.9375** | ≥ **0.8375** | > 24 | — | — | 24+ | ≈ 0.6–0.8 | **≈ 0.71** |
| SHRINK | 0.923 | 0.603 | 0.266 | 0.127 | 0.141 | 0.127 | 0.135 | 0.115 | **0.480** | 0.130 | 18.31 | 17.07 | 1.23 | 18 | −0.002 | −0.004 |

- Ranges are the two interpolation axes of round 1's (K′, n) table ((K′, n) bilinear vs (lag, n) bilinear; PROOFS p1b). Per-n readings (M2, M_far, n50, n90, WIDTH-2) use the per-n tables, the hypotheses' own form; the 2-D panel's K′-interpolation artefacts (n50 20.63, WIDTH 1.54 for H_shift) are not quoted. The H_full LIT floor takes a_20(n) at n ≥ 22 at chance (the coda finishes at most one reduction) [ASSUMED].
- KB2's decomposition: passes > 20 contribute −0.007 / −0.008 / +0.001 (H_shift / H_prop / constant growth); ≥ 99 % of KB2 is gain at passes ≤ 20 = K (RT M1). So KB2 is descriptive.
- **Sampling SE** at 1,000 items per n: SE(M2) 0.0059 / 0.0052 / 0.0044 (H_shift / H_prop / constant growth). **Sampling is not the error bar; lineage noise is** (RT M3). No FT-seed replicate exists; Kuo 2606.29983's 0.09–0.12 is from scratch. At a Gaussian lineage σ on M2 [DERIVED v1_panels lineage_noise_M2]:

  | σ | P(DBS-2 \| H_shift) | P(DBS-2 \| H_prop) | P(DBS-2 \| const. growth) | LR (H_prop / H_shift) | MARGIN-GROWS-2 mislabel \| H_shift / \| H_prop |
  |---|---|---|---|---|---|
  | 0.03 | 0.003 | 0.96 | 1.00 | 330 | 0.012 / 0.013 |
  | 0.06 | 0.084 | 0.80 | 0.94 | 9.6 | 0.13 / 0.13 |
  | 0.09 | 0.18 | 0.72 | 0.86 | 4.0 | 0.23 / 0.23 |
  | 0.12 | 0.25 | 0.67 | 0.79 | 2.7 | 0.29 / 0.29 |

- **Which readings split which hypotheses:** renewed growth (H_prop or constant growth) vs H_shift: M2 (DBS-2, MARGIN-GROWS-2), n50. H_prop vs constant growth: **WIDTH-2** (2.22 vs 1.57) and M2's level (0.852 vs 0.895, ≈ 8 SE apart by sampling, not by lineage noise); n50 does not split them. Growth vs H_full: M_far, STOP-LOC-2, **LIT-C2-2**. SHRINK: M2 below the SHRINK bar. H_T: the within-n T-split.
- **One lineage claims "an instance"**; a mechanism label reads "consistent with H", never "H confirmed / refuted".

### 3.5 Training memory [DERIVED, v1_memtime; form ASSUMED, validated once]

- Model (PROOFS_SELFDEPTH §11, T_ref 31): GiB(K, T, m) = 2 + 0.93 + [0.58·K·T/31 + 0.0258·(T/31)²·Σ_{t=1..K}(t + 1)] / m at bs 64. Validation on round 1: K 16, T 47 (the **longest** n-12 record, which pads micro-batch 0), m 2 → 14.47 projected vs **13.91 measured** (s215_ftself_s1_mem); m 1 → 26.0, measured OOM.
- **Round-2 peak shape:** vb_loss pads each micro-batch to its own max T, and G-S9's micro-batch 0 holds the longest record, so the peak T is the **maximum** T of the n-16 records: **61** in 4 of 5 seeds, 63 in 1 (77,325 draws per seed, D10's count / FT-SELF's keep 0.957; the 64th longest is 59).
  - m 1: 48.7 (51.0 at T 63); m 2: **25.8 > 21.0** (27.0); **m 4: 14.4 GiB (15.0)**; m 8: 8.7 (8.9).
  - So **micro 4**; T1's 8 is insurance for a host with less free memory. D10 does not change the peak shape (it is the 64 longest n-16 records, K 20, either way).

### 3.6 Time and cost [DERIVED, v1_memtime; models ASSUMED]

- Measured round-1 inputs: forced G-TIME self (K 16, chunks of 32, mean chunk max-T 41.5) 0.896 s, plain 0.439 s; real FT-SELF 7,260 s, FT-CTRL 4,877 s; real self step 0.804 s (0.897 × forced), real plain 0.406 s [MEASURED, PROOFS p5].
- **Forced deep self step at D10 (K 20, micro 4, pool-order chunks of 16, mean chunk max-T 52.2):** compute-bound refit (PROOFS' two-point fit at G-TIME shapes) **1.76 s**; launch-bound (∝ K × micro) **2.24 s**. Micro-8 contingency: 1.76–4.48 s.
- Real / forced: per-micro-batch rollouts (vb_loss Kb = max K in the micro-batch): under D10 at 16 per chunk P(Kb = 20) 0.42, E[Kb] 19.20 (50/50: 0.33, 18.96) [DERIVED v1_support_d10]. Calibrated on round 1's measured 0.897 (E[Kb] 15.80 of 16 at 32 per chunk): **≈ 0.87**.
- **Projection** (G-S10's registered form: 0.5 forced self + 0.5 forced plain, 12k steps, + 0.4 h): **4.1–4.9 h → $0.69–1.27** at $0.17–0.26/h. Real expected 3.6–4.3 h → $0.62–1.13. Micro 8 (launch-bound) would project 8.6 h → $1.46–2.24, above the admission bound: such a host is refused (§8).
- Not chosen: window-homogeneous 64-blocks (≈ $0.15 per cell cheaper; a different batch composition from round 1; RT open question 10).

### 3.7 Eval memory on the 3070 [DERIVED, research5/scripts/evalmem_cpu.jsonl; PROOFS #32–33]

- CPU fp32 rollout peaks with codas at every K′ on the longest items; slopes from B 8 → 16 (B 4 → 8 agrees within 2.0 % / 2.5 %): **n 16, K′ 24: 0.136 GiB/item; n 24, K′ 28: 0.236 GiB/item.**
- eval_deep at n 24, K′ 28 (+ 0.47 GiB weights): bs 32 → **8.03 GiB (over 8 GB)**; **bs 16 → 4.25**; bs 8 → 2.36 (no-intercept projection, conservative; the fitted intercept gives 7.48 / 3.70 / 1.81).
- The 3090 mem_check (bf16 autocast, **one** coda at K′ 24) is **consistent with** the slope (64 × 0.136 = 8.69 vs 8.93), not a validation of the fp32 all-codas case. **G-S16 is the only validation** (bs 16 has ≈ 1.6× slope headroom to 7.0 GiB).
- fp32 padded batches equal per-item results to 5.7e-15 in fp64 (vbank_eval docstring); bs-dependence is at near-ties only. Round-1 precedent: E2's items_n2 (length-sorted chunks, rollout to n + 2) equals eval_vb acc_n2 (test-order chunks, rollout to 24) exactly at every n 9–16 [MEASURED, RT m3]. BS-CHECK measures it on SD2's own files.

---

## 4. Readings, definitions and the scorer's field contract

Notation: a(n) = accuracy at K′ n + 2; a_k(n) = accuracy at K′ k. Chance ≈ 0.10–0.11 [MEASURED testset_ext majority share 0.101–0.108 at n 21–24]. Every reading is per cell and comparator-free; the bars are fixed by formula from the lineage's own earlier files before its R2 cell is scored.

### 4.0 One source per reading (D3)

- **Readings wholly inside n ≤ 16 come from `eval_vb.json`** (bs 32, harness unchanged): ABSORB-2, RETAIN, DEPTH-GENERAL-2, K′24-STABLE, FT-DIVERGED, VB-C1 parts, THINK-NECESSARY-DEEP-2, K′24-WIDE's n 9–16 half.
- **Every reading whose n-range includes any n ≥ 17 takes all of its n and K′ values, including n 13–16, from `eval_deep.json` only** (bs 16): DBS-2 / M2, MARGIN-GROWS-2, M_far, LIT-C2-2, STOP-LOC-2 (all k 13..23, n 14..24), n50, n90, WIDTH-2, KB2, RISE-PAST-K-2, C2-TAIL-2, NOT-SIZE-KEYED-2, K′24-WIDE's n 13–20 half, and the mechanism label's M2 / M_far / STOP inputs. score_s216.deep_read asserts that it opens only eval_deep.json.
- ABSORB-2 gates the mechanism label (UNLABELLED when it fails); it is a condition, not a value mixed into the label.
- **BS-CHECK** compares the two files at n 13–16 on a(n) **and** a_k(n) for every shared K′ (0..16, 24): max \|Δ\| (with its n, K′) and item flips (eval_deep items vs E2's items_n2 / items_k24 at K′ n + 2 and K′ 24). Descriptive, reported beside every eval_deep reading, never a censor and never a substitute source; values > 0.005 are flagged in the report.

### 4.1 Files

**Per R2 cell** (boxes/runs_fsq/s216_r2_s1/, s216_r2_s2/):

| file | producer | fields read | readings |
|---|---|---|---|
| log.jsonl | trainer | step, answer_ce, plain_answer_ce, self_answer_ce, self_reuse, pool_cursor, n_self_steps, c | dead (last step < 12,000), c = 0, reuse = 0 |
| args.json, args_diff.json | trainer | vs **s215_ftself_s1/args.json** (§4.6 allowed sets) | ARGS-DIFF |
| draws_log.json | trainer | coin, self_n, U, plain_hash | G-S3 (P(R \| t ≥ 13) = 1/2, precomputed value for s1; s1 plain_hash == round 1's at steps 1..200) |
| eval_vb.json | vbank_eval (all parts) | acc_n2, acc_k (0..16, 24), probe, test_meta, vswap, stall, stepcap | §4.0 first list |
| **eval_deep.json** (§4.2) | part `deep` | meta, acc_n2, acc_k (0..28), items, chance | §4.0 second list, BS-CHECK |
| eval_steppos.json, eval_lfk.json | vbank_diag | as §213 | LENGTH-FREE |
| eval_hist24.json (+ E4 items) | vbank_diag hist24 | intact, bank_ablate_slva, self_ablate, other_ablate, dup_newest, dup_foreign; items | KEEPS-SELF-STAB-2, MEM history, MEM sub-label (POST HOC) |
| eval_selfdepth.json (E2) | part `selfdepth --lineage --round` | cursor, memo (n 9..16), label_audit (pooled + by window, kept_wrong records), items_n2 / items_k24 + T + idx (n 1..16), contamination {pool_test_hits, pool_excl_hits} | MEMO-GAP-2, LABEL-AUDIT-2, CONTAMINATED, BS-CHECK flips |
| eval_chainlabel_fresh.json (E3, n 9..24) | vbank_diag | by_n keep, prec_kept, precision_strict | LABELLER-KEPT-2 (descriptive; round 3's question) |
| eval_rs, eval_d3hist, eval_vbhalt, eval_chainlabel | vbank_diag | as §215 | descriptive (never gate) |

**Shared per lineage:** `s216_pool_<s>/pool_meta.json` {seeds, per_n, ns, draws_n, collide_n, dup_n, keep_all, sha256, builder_commit, labeller, labeller_sha256, card, dtype, bs, excl {set: hits}, windows, T_max_by_n}; `s216_gate2_<s>/eval_gate.json` {by_n 9..16 {window, labeller, items, n_kept, keep_kv, pi, pi_E, pi_ci95, pi_E_ci95, records}, pooled_by_window, GATE_2, GATE_1p, fallback}. s2 round 1: `s216_pool_r1_s2/`, `s216_gate1_s2/` (round-1 formats).
**R1-s2** (s216_r1_s2/): round 1's file set (SELFDEPTH_v1 §4.1) with E2's lineage paths. **s2 base** (s216_d3t_s2/): log.jsonl, args.json, eval_vb.json (the `acc` part is registered, for competence; the full panel is optional, for descriptive VB-C1), eval_chainlabel_fresh.json (descriptive), eval_hist24.json (MEM history, descriptive). **Baselines (G-S18 / V3):** eval_deep.json in s215_ftself_s1 (required), s213_d3t_s1 / s215_ftctrl_s1 (optional), s216_r1_s2 (required).

### 4.2 eval_deep.json contract (new part; no code here)

- **Producer:** `python -m boxes.vbank_diag --run <cell> --part deep --eval_bs 16 --device cuda:<3070>` (bs 8 if G-S16 fails), local 3070, fp32 (no autocast), tf32 off (matmul and cudnn).
- **Items:** `N.test_set(1000, n_max=24)` restricted to n 13..24 (12,000 items, test seed 17100), in test order; chunks of eval_bs within each n (as part_acc).
- **Rollout:** one K′ 28 rollout per chunk, codas at every K′ ∈ 0..28 (K′ n + 2 ≤ 26 is in the grid).
- **Fields:**
  - `meta` {run, mode, card, dtype "fp32", tf32 false, eval_bs, per_n 1000, n_lo 13, n_hi 24, K_grid [0..28], test_seed 17100, test_n_max 24, **items_sha256**, model_sha256, commit, sec, **peak_gib** (max_memory_allocated)};
  - **items_sha256** = sha256 of the UTF-8 bytes of `json.dumps([[it["syms"], int(it["t"])] for it in items], separators=(",", ":"), ensure_ascii=True)` over the 12,000 items in test order (m9). The scorer recomputes it from N.test_set and asserts equality (mismatch → void);
  - `acc_n2` {n: acc at K′ n + 2}; `acc_k` {K′: {n: acc}} for K′ 0..28;
  - `items` {n: {"idx": [test-order index], "y": [...], "T": [...], "height": [...], "ans": [[argmax answer at K′ 0..28] per item]}} (348k ints ≈ 1.5 MB): every per-item output behind every eval_deep reading (store-items rule). The scorer asserts acc_k[K′][n] == mean(ans[K′] == y) for every K′, n;
  - `chance` {n: majority-class share}; `def` (text).
- The scorer asserts meta.card contains "3070", dtype fp32, tf32 false, eval_bs ∈ {16, 8} (the G-S16 outcome), n range and K grid.
- **Not computed:** bank-off / ablate arms, probes, vswap. **Cost:** ≈ 10–30 min on the 3070 [ASSUMED]; peak 4.25 GiB projected at bs 16.

### 4.3 Primary and mechanism readings, with the per-lineage bar formulas

**Bar formulas** (registered; reference implementation `r1_bars` / `r2_bars` in v1_checks/v1_panels.py, which score_s216 must reproduce; ceil005(x) = the smallest multiple of 0.005 ≥ x). Only measured eval_vb values at n ≤ 16 enter a bar.

| bar | formula (lineage ℓ: base a_0 = D3-T ℓ, round 1 a_1 = R1 ℓ) | s1 instance [DERIVED, asserted] |
|---|---|---|
| R1 MARGIN-GROWS | ceil005(M_shift,1 + 0.15), M_shift,1 = mean_{n 9..12} a_0(n) | 0.47975 → **0.63** (v1) |
| R1 SHRINK | M < ceil005(M_shift,1 − 0.08) | **0.40** (v1) |
| R1 FULL | M ≥ max(0.80, R1 MARGIN-GROWS) | 0.80 (v1) |
| R2 M2_shift | mean_{n 13..16} a_1(n) | 0.71725 |
| R2 M2_prop | mean_{d 1..4} a_1(12 + 0.75 d), linear in n | 0.85156 |
| R2 M2_shrink | mean_{n 9..12} a_0(n) | 0.47975 |
| **MARGIN-GROWS-2** | ceil005((M2_shift + M2_prop)/2) | **0.785** |
| **SHRINK bar** | ceil005((M2_shrink + M2_shift)/2) | **0.60** |
| **FULL** | M2 ≥ max(0.80, MARGIN-GROWS-2) ∧ M_far ≥ **0.60** (absolute) | 0.80 / 0.60 |
| FULL-separability flag (descriptive) | H_prop's M_far upper bound mean_{d 5..8} a_1(12 + 0.75 d), with a_1(x > 16) := a_1(16) (monotone [ASSUMED]), ≥ 0.60 → "FULL not separable from H_prop" | 0.362 → no flag |
| degenerate bars | SHRINK bar ≥ MARGIN-GROWS-2 (possible only if round 1 shrank the lineage's margin) → mechanism "UNLABELLED (bars degenerate)" | not degenerate |
| absolute bars | DBS-2 0.80, LIT-C2-2 0.10, ABSORB-2 0.95, RETAIN 0.99 | — |

**R2 readings (per cell, `r2_cell(ℓ)`):**

| reading | exact definition | source | role | FT-SELF (round-1 model) |
|---|---|---|---|---|
| **ABSORB-2** | min_{n 13..16} a(n) ≥ 0.95 | eval_vb | in support; enabler; **never C2 / C4** | 0.344 (n 16) |
| **RETAIN** | min_{n 9..12} a(n) ≥ 0.99; failure = ERODED (part of SD2-SELF) | eval_vb | must not break | 1.000 |
| DEPTH-GENERAL-2 | mean_{n 13..16} a(n) ≥ 0.80 | eval_vb | descriptive, IN-SUPPORT | 0.717 |
| **DBS-2** | M2 = mean_{n 17..20} a(n) ≥ **0.80** | eval_deep | **the C2-proxy reading** (one lineage: an instance) | baseline after launch (D2) |
| **MARGIN-GROWS-2** | M2 ≥ the lineage's MARGIN-GROWS-2 (s1 0.785) | eval_deep | mechanism: renewed growth (any form) | — |
| M_far | mean_{n 21..24} a(n) | eval_deep | enters FULL | — |
| **LIT-C2-2** | mean_{n 22..24}[a(n) − a_20(n)] ≥ **0.10** (accuracy gained by passes 21..n + 2 > K 20 on items that need ≥ K + 2 reductions) | eval_deep | **C2-literal row on SD2-C2's line**, not in the SD2-C2 conjunction | panels: H_shift 0.005, H_prop 0.04, const. growth 0.03, H_full ≈ 0.71 |
| **STOP-LOC-2** | G(k) = mean over n ∈ [k + 1, min(k + 4, 24)] of a(n) − a_k(n), k 13..23, no readability filter; STOP = min k with G(k) ≤ 0.02, else "24+". STOP-TO-SUPPORT-2 iff STOP ≥ 17 | eval_deep | descriptive; enters SHIFT / MIXED | round 1: 15 |
| n50, n90 | first downward crossing of 0.5 (0.9) by a(n) over n 13..24, linear interpolation; never crossing → "> 24"; a(13) already below → "< 13" | eval_deep | descriptive; n50 panels 19.46 / 20.61 / 20.61 (H_shift / H_prop / const. growth) | 15.46 / 13.93 |
| **WIDTH-2** | n50 − n90 (unreadable if either is "> 24" or "< 13") | eval_deep | descriptive; **H_prop's discriminator** (panels 1.53 / 2.22 / 1.57) | 1.53 (base 1.23) |
| KB2 | mean_{n 19, 20}[a(n) − a_18(n)]; "≥ 0.25" reported | eval_deep | **descriptive** ("pass gain at out-of-support depth, mostly passes ≤ 20") | round 1's analogue 0.155 |
| RISE-PAST-K-2 | mean_{n 19, 20}[a(n) − a_20(n)] | eval_deep | descriptive (≈ 0 under every H) | — |
| C2-TAIL-2 | a_28(n) ≥ a(n) − 0.05 at every n 17..20 | eval_deep | descriptive | — |
| THINK-NECESSARY-DEEP-2 | max(a_0(n), a_1(n)) ≤ 0.15 at every n 13..16 | eval_vb | descriptive (C1 at the new depth) | — |

**Mechanism label** (exhaustive; only under ABSORB-2, else **UNLABELLED (¬ABSORB-2)**; "UNLABELLED (bars degenerate)" as above; MG = MARGIN-GROWS-2, SB = SHRINK bar):
- **FULL**: M2 ≥ max(0.80, MG) ∧ M_far ≥ 0.60 (with the separability flag beside it when raised);
- **GROW**: M2 ≥ MG ∧ ¬FULL (contains H_prop and constant growth);
- **SHIFT**: SB ≤ M2 < MG ∧ STOP ≥ 17;
- **MIXED**: SB ≤ M2 < MG ∧ STOP ≤ 16;
- **SHRINK**: M2 < SB.

Panels [DERIVED v1_panels]: H_shift → SHIFT (M2 0.717, STOP 20), H_prop → GROW (0.852, M_far 0.24), constant growth → GROW (0.895, M_far 0.21), H_full floor → FULL, base margin → SHRINK (0.480). The label is "consistent with", never "confirmed" (§3.4 lineage noise: 13–29 % mislabel at σ 0.06–0.12).

**R1-s2 readings (`r1_cell("s2")`):** SELFDEPTH_v1 §4.2–§4.3 and §5.2 as registered (ABSORB, DBS, MARGIN-GROWS, STOP-LOC, K′-BEYOND, the kept readings, LABEL-AUDIT, MEMO-GAP, CONTAMINATED, censors), with the lineage's own paths, contamination vs test n_max 24, and the R1 bar formulas above (MARGIN-GROWS, SHRINK, FULL) on D3-T s2's eval_vb. K′-BEYOND is read as registered; per RT M1 it is mostly gain within K.

### 4.4 Kept and descriptive readings (per R2 cell)

| reading | definition | round-1 FT-SELF |
|---|---|---|
| LENGTH-FREE | score_s213.no_knee ∧ score_s212.lfk MEETS (as registered) | ✓ |
| VB-C1 | score_s211.cell_read parts, dead = 12,000, no NO-ENTRY censor (as §215) | ✓ |
| **K′24-STABLE** | a_24(n) ≥ 0.99 at every n 1..8 (eval_vb), **unchanged** (D8) | ✓ at the edge: 0.990 at n 1 (FT-CTRL 1.000, so the dip is self-item-specific) |
| K′24-WIDE (descriptive) | a_24(n) ≥ a(n) − 0.01 at n 9..16 (eval_vb); a_28(n) ≥ a(n) − 0.05 at n 13..20 (eval_deep) | — |
| **KEEPS-SELF-STAB-2** | score_s213.hist24: gate ∧ CONTENT-STABILISES ∧ STAB_self ≥ 0.5 (n 3..6), **unchanged**. Else **MEM-TRADED-2**. Recomputed by Audit from E4's per-item store (must equal the per-n reading) | 0.489 (MEM-TRADED) |
| MEM history (descriptive) | STAB_self base → R1 → R2: s1 0.782 → 0.489 → x; **MEM-ERODING** iff x < (R1's value − 0.05) (s1: 0.439). s2: the same formula on its own hist24 files | — |
| MEM sub-label (**POST HOC**, descriptive; D5) | defined after round 1's readout; only when MEM-TRADED-2: **"STAB_self low with dup_newest destabilising"** iff CONTENT == CONTENT-STABILISES (STAB_bank ≥ 0.10 ∧ STAB_dup_newest ≥ 0.10) ∧ STAB_self < 0.5; **"content channel not stabilising"** iff the gate passes and CONTENT ≠ CONTENT-STABILISES; **"unreadable (gate)"** iff the hist24 gate fails. Exhaustive. Never used to soften MEM-TRADED in the user's OK question; the registered KEEPS-SELF-STAB-2 rule is unchanged | round 1 would read the first (STAB_dup_newest 0.877) |
| **LABEL-AUDIT-2** | 5,000 random consumed records (s1 rng 22670, s2 22870): π ≥ 0.99 ∧ π^E ≥ 0.99, pooled and per window, each window's Wilson 95 % CI overlapping its gate's CI; kept-wrong patterns printed and compared with round 1's. Else **LABEL-NOISY-2** (censor) | 0.9996 / 0.9996 |
| MEMO-GAP-2 | \|acc(K′ n + 2) on the first 2,000 consumed pool items of n − on the 2,000 gate items of n\| ≤ 0.02 at n 9..16 | 0 gap at n 9–12 |
| CONTAMINATED | any consumed record in test_set(1000, 24), SERIAL, 213-37, any gate, or the round-1 pool; or self_reuse > 0 → censor | 0 / 0 |
| LABELLER-KEPT-2 | fresh chain keep / prec_kept at n 13..24 (E3) | descriptive |
| **NOT-SIZE-KEYED-2** | within n 17..22: a(n) for T ≤ T_train vs T > T_train (T_train from pool_meta T_max_by_n[16]), and the within-n OLS slope of correct on T (eval_deep items). **H_T flag** iff the T > T_train deficit is ≥ 0.10 at ≥ 2 of n 19–22 with ≥ 100 items per band. The band rule qualifies at about n 20–21 only; within-n T tracks bracket structure, so the flag is confounded and descriptive. Also M2 on items with T ≤ T_train | round 1: no significant pooled effect (p 0.13); n 15 nominally negative |
| BS-CHECK | §4.0 | round 1: items_n2 = eval_vb exactly at n 9–16 |

### 4.5 Lineage rows

- **SD2-C2 ×2** = SD2-C2 in `s216_r2_s1` **and** in `s216_r2_s2`. Wording: "in both competent lineages (2/2; base seeds 1 and 2)". It is not a rate (rates need ≥ 3 seeds and the user's OK).
- **Outcomes, exhaustive:**
  - **PASS** (2/2).
  - **FAIL**, worded "SD2-C2 in k of 2 lineages" with the failing stage named: s1 R2 censored or ¬SD2-C2; s2 base dead (incomplete 48k, guard stop, NaN, host loss); GATE-1-s2 fails; R1-s2 fails its round-2 condition ("lineage stopped at round 1"; its round-1 label is reported, GROW replicated or not); GATE-2-s2 fails with both labellers; R2-s2 censored (dead, FT-DIVERGED, LABEL-NOISY-2, CONTAMINATED, ARGS-DIFF, items_sha256 mismatch, or a registered file still missing in `--final`) or ¬SD2-C2.
  - **NOT COMPETENT**: a **completed** s2 base below the accuracy floor (§2.4) reads "s2 base not competent (C6 data point); SD2-C2 instance in 1 lineage; ×2 not evaluable". This is the only exclusion.
  - **UNREAD**: a cell never admitted (§8), or pending. Never a pass.
- Also per lineage: SD2-SELF, LIT-C2-2, the mechanism label, WIDTH-2, n50, MEM history; and SD2-SELF ×2 (descriptive).

### 4.6 Scorer theory/score_s216.py (to write; fixture-checked before any output)

- **Lineage table** (one dict): s1 {base s213_d3t_s1, r1 s215_ftself_s1, r2 s216_r2_s1, pool_r1 s215_pool, pool_r2 s216_pool_s1, gate1 s215_gate1, gate2 s216_gate2_s1}; s2 {base s216_d3t_s2, r1 s216_r1_s2, r2 s216_r2_s2, pool_r1 s216_pool_r1_s2, pool_r2 s216_pool_r2_s2, gate1 s216_gate1_s2, gate2 s216_gate2_s2}.
- **ARGS-DIFF** (fixes RT B1; META = score_s213.META):
  - s2 base vs s213_d3t_s1/args.json: allowed {seed}; seed must equal 2.
  - R1-s2 vs s215_ftself_s1/args.json: allowed {seed, init_from, self_pool, self_micro, args_ref, mem_check_self, mem_check_steps, keep_ckpt_steps, stop_at, draws_log_steps}; init_from must be the lineage's base model.pt, self_pool the lineage's round-1 pool, seed 2.
  - R2 cells vs s215_ftself_s1/args.json: allowed {init_from, self_pool, self_kmax, self_micro, args_ref, mem_check_self, mem_check_steps, keep_ckpt_steps, stop_at} (+ seed for s2); self_kmax must be 20, init_from the lineage's round-1 model.pt, self_pool the lineage's round-2 pool, seed 1 (s1) / 2 (s2).
  - Any other shared difference, an unexpected new key, or pause on → ARGS_BAD (censor).
- **Pending rule (215-7; RT M5).** A cell reads **"pending (missing [...])"** while any registered file is missing, and reads nothing else first. Registered files: R2 cells: log.jsonl, args.json, draws_log.json, eval_vb.json (all parts), eval_deep.json, eval_steppos.json, eval_lfk.json, eval_hist24.json (with E4 items), eval_selfdepth.json, and the lineage's pool_meta.json and eval_gate.json (with GATE_2 / GATE_1p decided). R1-s2: SELFDEPTH_v1 §4.1's set with the lineage paths, plus GATE-1-s2's eval_gate.json. s2 base: log.jsonl, args.json, eval_vb.json (acc part). **A missing eval_selfdepth.json or gate file is pending, never a pass** of LABEL-AUDIT-2, CONTAMINATED or a gate. Pending never counts as a pass of any reading or row.
- **`--final <lineage>`** (run once Theory records the lineage's chain as done): a registered file still missing then **fails by rule b**: readings that need it fail; LABEL-AUDIT-2 / CONTAMINATED / gates read as censored. Descriptive files (eval_rs, eval_d3hist, eval_vbhalt, eval_chainlabel(_fresh), baselines) never gate.
- **Reuses:** score_s215 (stop_loc generalised to a k-range and a file; the censors with 215-7's pending rule; gate1 generalised to windows and labellers), score_s213 (no_knee, hist24, jl, META), score_s212 (lfk), score_s211 (cell_read's VB-C1 parts, entry_on, log_recs).
- **New:** deep_read (single-source assert, items_sha256, meta asserts, acc / items consistency; M2, M_far, LIT-C2-2, STOP-LOC-2, n50, n90, WIDTH-2, KB2, RISE, C2-TAIL-2, T-split), r1_bars / r2_bars (identical to v1_panels.py), mechanism2, retain, gate2, label_audit2, memo2, contamination2, hist24_items (STAB recomputed from E4; must equal score_s213.hist24 to 1e-12), mem_history, mem_sublabel, bs_check, competence, lineage rows, descriptive reference panels per lineage (the round-1 extension replaced by that lineage's R1 eval_deep when present).
- **Fixtures (G-S12):** exact H_shift (→ SHIFT, M2 0.717, STOP 20, ¬DBS-2, LIT 0.005); H_prop (→ GROW, DBS-2 ✓ 0.852, n50 20.61, WIDTH 2.22); constant growth (→ GROW, M2 0.895, WIDTH 1.57, KB2 0.51); H_full floor (→ FULL, LIT-C2-2 ✓); base margin (→ SHRINK); MIXED (M2 0.70, STOP 16); ¬ABSORB-2 (UNLABELLED); RETAIN fail (ERODED, ¬SD2-SELF); dead; FT-DIVERGED; LABEL-NOISY-2 (one window 0.985); CONTAMINATED (a pool record equal to a test n-22 item); ARGS-DIFF (self_kmax 24; s2 seed 1); **s2 bars by formula** (an R1-s2 curve translated +1 in n: the bars move, and that curve's exact H_shift reads SHIFT); degenerate bars (UNLABELLED); **R1-s2 with seed 2 and its own gate** (not ARGS_BAD; reads GATE-1-s2); dead s2 base → ×2 FAIL; completed s2 base below the floor → NOT COMPETENT; GATE-1-s2 fail → ×2 FAIL; R1-s2 stopped → ×2 FAIL; **single source** (eval_vb differing by 0.01 at n 13–16 leaves STOP-LOC-2 / n50 / WIDTH-2 unchanged); **each registered file missing → pending, and with --final → failure**; eval_selfdepth missing → pending (never LABEL-AUDIT-2 pass); items_sha256 mismatch → void; E4 items inconsistent with the per-n means → hist24 gate fails; MEM-TRADED-2 sub-label cases (all three); H_T flag; BS-CHECK flag; not admitted → UNREAD.

---

## 5. Claims, wording, censoring, Goodhart

### 5.1 Claims and pre-stated wording

- **SD2-SELF** = ABSORB-2 ∧ RETAIN ∧ LENGTH-FREE ∧ VB-C1 ∧ K′24-STABLE, uncensored. Wording (only this): *"latent depth extended from 12 to 16 by distilling the round-1 model's own re-rooted single-step chain into the loop. The labeller was itself self-trained; labels are gold-equivalent at π = [GATE-2's measured pooled π] (rule ≥ 0.995; LABEL-AUDIT-2 π = [value]); no gold labels beyond n 8; in support; not C2, not C4."* (The bracketed values are filled from eval_gate.json / eval_selfdepth.json; the draft's fixed "≥ 0.999" is not pre-claimed.) An enabler, never a headline.
  - **Fallback wording (m11):** if a window is BASE-LABELLED, the sentence names the labeller per window: *"… labels for n 13–16 [or 9–12] from the base model (BASE-LABELLED fallback after the round-1 model failed GATE-2 [GATE-1′]) …"*, and drops "the labeller was itself self-trained" for that window.
- **SD2-C2** = SD2-SELF ∧ DBS-2: *"depth extrapolation past the self-labelled range (n 17–20, untouched §170-extension items; C2 proxy; one lineage: an instance)"*. On the same line: **LIT-C2-2** (met / not met: *"accuracy rises with latent passes past the trained budget at n 22–24 (literal C2)"*), the mechanism label ("consistent with"), n50, WIDTH-2, MEM-TRADED-2 status and BS-CHECK.
- **SD2-C2-MEM** = SD2-C2 ∧ KEEPS-SELF-STAB-2. MEM-TRADED-2 and the MEM history are stated beside any positive.
- **DBS-2 is worded as a test of renewed margin growth**, never as a test of H_prop. "Consistent with H_prop" needs WIDTH-2 near H_prop's panel (2.22) rather than the translation panels (1.53 / 1.57), and is descriptive.
- **Never:** ABSORB-2, RETAIN, DEPTH-GENERAL-2, MARGIN-GROWS-2, STOP-LOC-2, KB2, WIDTH-2 or THINK-NECESSARY-DEEP-2 worded as C2 or C4. KB2 is never called C2-literal (RT M1).
- **C4** (strict and revised) stays unmet: round 2's labeller inherits a step learned from simulator R2 states at n ≤ 8. "Self-trained labeller" is a procedural property. **C1:** the decode / re-render is training-time only; test-time computation is latent.
- **Plainly:** under H_shift each round buys +4 in support at a constant margin of ≈ 3.5, and C2 at n 17–20 needs another round. Under renewed growth (proportional or constant) round 2 is predicted to clear DBS-2. Literal C2 needs the boundary to stop binding, which LIT-C2-2 asks. SD2's main expected product is SD2-SELF (P ≈ 0.26) plus one mechanism instance; P(SD2-C2) ≈ 0.07. One lineage is an instance; 2/2 is the north-star direction ("every competent seed").

### 5.2 Censoring (rule b; each fails every reading of that cell)

- dead (log last step < 12,000 for FT cells, < 48,000 for the s2 base; guard stops included);
- **FT-DIVERGED**: a(n) < 0.99 or a_24(n) < 0.97 at any n 1–8;
- LABEL-NOISY-2; CONTAMINATED; an ARGS-DIFF mismatch;
- **eval_deep missing at `--final`, or items_sha256 / meta mismatch** → every n ≥ 17 reading fails (DBS-2, MARGIN-GROWS-2 and LIT-C2-2 count as failures) and the mechanism reads UNLABELLED;
- **pending never passes**: a missing registered file reads pending until `--final`, then fails (§4.6);
- never-reached events are values, not missing: n50 / n90 "> 24", STOP "24+"; chance-level items count as wrong;
- both labellers fail a window's gate → no round 2 in that lineage (no cell, no spend); for s2 that fails the ×2 row.
- **¬ABSORB-2 follow-up (pre-stated; RT M4):** an uncensored ¬ABSORB-2 in `s216_r2_s1` starts nothing automatically. Theory may propose one disclosed re-run (24k steps, or lr 1e-4), which needs its own note, Audit and the user's OK; otherwise the SD line stops with round 1 plus this result. In s2, ¬ABSORB-2 (or R1-s2's ¬ABSORB) fails the ×2 row; no re-run inside SD2.

### 5.3 Goodhart analysis (targets new in SD2)

| target / route | finding |
|---|---|
| Stop at the new boundary (pass ≥ 17) | Zero loss on the support (§3.1a). This **is** H_shift, irreducible by loss. Read by M2 / STOP-LOC-2, not prevented. D10 raises the boundary deadline cost only slightly (0.35 % → 0.47 % at t_d 16) |
| Copying the self-trained labeller's systematic errors (Lee App B.4) | kept-wrong ≈ 5e-4 per W_2 item; class mass ≤ 1.0e-3 per round, ≈ 1.8e-3 cumulative under A5's failure mode (§3.3c). LABEL-AUDIT-2 prints the patterns per window and compares them with round 1's |
| Selection by the labeller (kept items easier; Literature failure mode 1) | ABSORB-2's denominator is **all** §170 test items, so selection cannot inflate it. KEPT-VS-DRAWN is descriptive. Round 1's ceiling argument did not bind (keep 0.979 at n 12, loop 1.000) |
| Memorising pool items | Fresh-once; MEMO-GAP-2 at n 9–16 |
| Oracle loop count (Fan's max-confidence, Kuo's best-K) | No per-item or per-n K′ selection: readings at fixed K′ n + 2 and fixed K′ 18 / 20 / 28; the full grid is stored |
| **LIT-C2-2's difference form** | It rises if a_20(n) falls **below** chance at n 22–24 (a systematically wrong class at K′ 20) as well as if a(n) rises. No objective targets it, but the reading is two-sided only through a(n). Guard (descriptive, beside it): LIT-C2-2-floored = mean_{n 22..24}[a(n) − max(a_20(n), chance(n))], and a(n) − chance(n) at n 22–24. The registered row is D9's formula |
| Per-lineage bars | Fixed from the lineage's earlier eval_vb files before its R2 cell exists; no training objective reads them. A weaker R1-s2 margin lowers R2-s2's growth bar by design (lineage-relative H_shift) |
| KEEPS-SELF-STAB's direction | STAB_self = 1 − a_24(self-ablated) when intact = 1, so it falls when the model becomes **more robust without own-slot history**, not only when memory is lost. FT-SELF's arms [MEASURED eval_hist24, n 3–8 at K′ 24]: self-ablated 0.43 / 0.46 / 0.55 / 0.60 / 0.65 / 0.65, bank-ablated 0.28 / 0.29 / 0.38 / 0.46 / 0.52 / 0.53, dup_newest 0.13 / 0.12 / 0.13 / 0.11 / 0.10 / 0.10, intact 1.000. Less use of history in general is **the trade itself** (RT M6). The registered rule is kept; the sub-label is POST HOC and descriptive and stays out of the user's question (D5) [interpretation CONJECTURE] |

---

## 6. OC (one lineage per cell; ×2 for both; realised spend at registration $0) [arithmetic DERIVED, v1_checks/v1_oc.py; every prior ASSUMED]

**Headline inputs (D7):** RT's shaded priors given ABSORB-2, **H_shift 0.60 / H_grow 0.30 / H_full 0.02 / SHRINK 0.08**; P(DBS-2 \| H) 0.12 / 0.55 / 0.90 / 0 (lineage noise; RT M3); P(MARGIN-GROWS-2 \| H) 0.18 / 0.70 / 1.0 / 0; P(KB2 ≥ 0.25 \| H) 0.10 / 0.40 / 0.85 / 0; P(LIT-C2-2 \| H) 0.01 / 0.05 / 0.85 / 0 (panels 0.005 / 0.03–0.05 / ≈ 0.71). Recomputed after D9 / D10 with these changes, each stated:
- **P(ABSORB-2) 0.65** (RT 0.60 was set at the 50/50 dose of 1.03× / 0.65× round 1's sustained points; D10 gives 1.38× / 0.86×; the window is still harder: 16 serial passes, T to 61–63). Sensitivity 0.60 / 0.70.
- **RETAIN \| ABSORB-2 0.93** (RT 0.95): the old window's rehearsal falls to 31,957 per n under D10 (round 1 absorbed it at 95,872 per n). Sensitivity 0.95 / 0.90.
- **K′24-STABLE \| ABSORB-2 0.55**, no longer independent (PROOFS #42: round 1's n-1 dip came with the self items; D10 adds more deep self dose). Binomial floor: P(≥ 990 / 1,000) = 0.583 at a true n-1 error 0.010, 0.987 at 0.005, 0.117 at 0.015 [DERIVED v1_support_d10]; 415 distinct n-1 items (RT m5). Bounded over [0.45, 0.65]. With P(K′24 \| ¬ABSORB-2) 0.65 the joint P(ABSORB-2 ∧ K′24) is 0.358 vs 0.380 under independence at the same marginal (Fréchet bounds 0.235–0.585).
- **¬FT-DIVERGED \| K′24-STABLE 0.97**: FT-DIVERGED's a_24 < 0.97 clause already implies ¬K′24-STABLE, so given K′24-STABLE only its a(n) < 0.99 clause remains (FT-SELF reads 1.000 at every n 1–8). P(FT-DIVERGED \| ABSORB-2) ≈ 0.12, a whole-cell censor (RT m5; RT's independent 0.90 factor double-counted the a_24 clause).
- LENGTH-FREE 0.90, VB-C1 0.90 (RT). Kept factor given ABSORB-2: 0.90 · 0.90 · 0.93 · 0.55 · 0.97 = **0.402**.
- KEEPS-SELF-STAB-2 0.20 (RT; 0.20–0.25 by pair branch, §2.5).
- s2: P(base competent) 0.75 (RT) × P(GATE-1-s2) 0.95 × P(R1-s2 meets the round-2 condition) 0.70 (RT) × P(GATE-2-s2, either labeller) 0.98 = 0.489 reach R2-s2; correlation factor 1.2 on the conditional pass (shared hypothesis and recipe; RT).

| reading | P(pass \| no effect) | P(pass \| effect) | **headline** |
|---|---|---|---|
| ABSORB-2 | ≈ 0 (no round-2 effect: FT-SELF n 16 0.344, 40 SE) | 0.65 [ASSUMED] | 0.65 |
| RETAIN | ≈ 1 if intact (p_err 0.001: all four n ≈ 1.000); 0.010 per n under 2 % erosion | — | 0.93 \| ABSORB-2 |
| K′24-STABLE | 0.987 / 0.583 / 0.117 at a true n-1 error 0.005 / 0.010 / 0.015 | — | 0.55 \| ABSORB-2 |
| DBS-2 | **H_shift: 0.084 / 0.18 / 0.25 at lineage σ 0.06 / 0.09 / 0.12** (headline 0.12) | H_prop 0.80 / 0.72 / 0.67; const. growth 0.94 / 0.86 / 0.79; H_grow family 0.55; H_full 0.90 | 0.166 |
| MARGIN-GROWS-2 | H_shift 0.13 / 0.23 / 0.29 (headline 0.18) | H_prop 0.87 / 0.77 / 0.71; family 0.70; H_full 1.0 | 0.220 |
| LIT-C2-2 (per cell given ABSORB-2; headline jointly with SD2-SELF) | boundary binding: 0.01 \| H_shift, 0.05 \| H_grow (panels 0.005–0.046 vs bar 0.10) | 0.85 \| H_full (floor ≈ 0.71) | **0.010** |
| KB2 ≥ 0.25 (descriptive) | H_shift 0.10 | H_grow 0.40, H_full 0.85 | 0.128 |
| SD2-SELF | ≈ 0 | — | **0.261** |
| **SD2-C2** | **0.031 \| H_shift** (0.022 / 0.047 / 0.064 at σ 0.06 / 0.09 / 0.12) | **0.144 \| H_grow**, 0.17–0.21 \| H_prop, **0.235 \| H_full** | **0.067** |
| SD2-C2-MEM | ≤ 0.006 | 0.03–0.05 | 0.013 |
| **SD2-C2 ×2** | **0.0006** (H_shift in both) | — | **0.0046** (0.0062 \| s2 base competent) |
| MG-2 mechanism mislabel | 0.13–0.29 \| H_shift | 0.13–0.29 \| H_prop | — |
| GATE-2 value terms | π 0.993: 0.014; π 0.995: 0.52 | π 0.997: 0.9987; measured 0.9995: ≈ 1 | ≈ 1 |
| GATE-1′ value terms | π 0.993: 0.0105; π 0.995: 0.48 | π 0.997: 0.9983 | ≈ 1 |
| GATE-2 π^E term (7,858 kept, ≤ 78 wrong) | π^E 0.988: 0.048; 0.990: 0.504 | π^E 0.993: 0.9987 | ≈ 1 |
| LABEL-AUDIT-2 (D10 split: ≈ 3,333 new / 1,667 old records, SD 33) | π 0.985: pooled 0.0013; new window 0.0067; **old window 0.037**; π 0.99: 0.50–0.54 | π 0.9995: ≈ 1 (expected kept-wrong 1.7 of 5,000) | ≈ 1 |
| KEEPS-SELF-STAB-2 | fails at an unchanged 0.489 | — | 0.20 [ASSUMED] |

**Sensitivity and cross-checks** (P(SD2-SELF) / P(SD2-C2) / ×2):

| variant | SD2-SELF | SD2-C2 | ×2 |
|---|---|---|---|
| **headline** | 0.261 | **0.067** | 0.0046 |
| pre-stated ×1.5 GROW shift (priors 0.375 / 0.542 / 0.025 / 0.058; sensitivity only, D7) | 0.261 | 0.077 | 0.0057 |
| P(ABSORB-2) 0.60 / 0.70 | 0.241 / 0.281 | 0.062 / 0.072 | 0.0039 / 0.0054 |
| P(K′24 \| ABSORB-2) 0.45 / 0.65 | 0.214 / 0.309 | 0.055 / 0.079 | 0.0031 / 0.0065 |
| RETAIN 0.95 / 0.90 | 0.267 / 0.253 | 0.068 / 0.065 | 0.0048 / 0.0043 |
| lineage σ 0.06 (DBS \| H_shift 0.084, \| H_grow 0.60) / σ 0.12 (0.245, 0.50) | 0.261 | 0.065 / 0.082 | 0.0051 / 0.0051 |
| range low (P(A) 0.55, K′24 0.45, RETAIN 0.90, DBS \| H_shift 0.18, \| H_grow 0.45; s2 0.70 / 0.60) | 0.175 | **0.046** | 0.0014 |
| range high (0.70, 0.65, 0.95, 0.08, 0.60, ×1.5 priors; s2 0.80 / 0.80) | 0.340 | **0.099** | 0.013 |
| cross-check: RT's shaded OC (rt_oc.json), reproduced exactly | 0.249 | 0.0636 | 0.0045 |
| cross-check: PROOFS' re-shade ("both" row), reproduced exactly | 0.254 | 0.0700 | — |

- **How to read it.** A DBS-2 pass is evidence for **renewed growth** (likelihood ratio vs H_shift ≈ 10 / 4 / 2.7 at σ 0.06 / 0.09 / 0.12), not for H_prop specifically, and not literal C2. A null is weak against growth (0.55). MARGIN-GROWS-2 and WIDTH-2 are the informative splits; LIT-C2-2 is the literal-C2 row and is expected to fail unless the boundary stops binding. SD2-SELF passes under H_shift, so it says nothing about C2.
- **Rule check (OC block + censoring rule):** every registered reading above carries P(pass \| no effect) and P(pass \| effect) at its seed count (one lineage; ×2 for both); never-reached events count as failures (§5.2); realised $ at registration is $0.

---

## 7. Gates (all green before any spend on the cell they guard; Audit reviews)

| gate | check |
|---|---|
| **G-S0 NO-GOLD** | as SELFDEPTH_v1: builder gold patches before generation; key-set assert at load; SD-LABEL-MAP grep; no `s216_gate` in vbank.py; self-batch call-counter smoke |
| G-S1 EXCLUSION | pool ∩ (test n_max 24 ∪ SERIAL ∪ 213-37 ∪ every gate ∪ round-1 pool(s)) = ∅ by exact syms; per-n draws / collisions logged; n(syms) == generator n; per lineage and round |
| G-S2 COMPOSE | trainer E_self (HOLD-padded to **Kb 20**) == recode_targets(root, 20) bitwise on GATE-2 / GATE-1′ kept items and the first 1,000 pool records (separate process); only the listed wrong-state items may differ |
| **G-S3 STREAMS** | s1: P(R \| t ≥ 13) over all 12,000 logged steps **equals the value precomputed** from pool_r2.jsonl + s215_ftself_s1/draws_log.json U's (law: 1/2, SE 0.0004); 0 at t ≥ 17; consumed new-window share 2/3 ± 0.001; self U only from rng_s; **plain_hash at steps 1..200 == s215_ftself_s1/draws_log.json** (D11). s2: 0.500 ± 0.002, 0 at t ≥ 17 |
| G-S4 INIT + RESUME | FT cells: `--init_from <R1 or base> --keep_ckpt_steps 0 --stop_at 2 --ckpt_every 1`: ckpt_model_0 == init model.pt on every key; INIT_FROM prints missing []; calibration.json == the init's; ckpt_last at steps 1 and 2; a resume runs step 3 with cursor / rng_s / AdamW restored |
| **G-S4b BASE RESUME** (RT m7) | `s216_d3t_s2`: `--stop_at 2 --ckpt_every 1`, calibrate() at step 0 writes calibration.json; a resume runs step 3 with all rngs, CurGate stage and AdamW restored, and calibrate() not re-run |
| G-S5 ARGS-DIFF | in each launch log, the §4.6 allowed sets only |
| G-S6 LOAD | VE.load strict on each new model.pt |
| **G-S7 PLAIN-PATH** (required, D4) | after T1: p_self 0, construct + 1 step, CPU-bitwise vs the pre-edit tree, on the final tree |
| G-S8 GRADFLOW | scratch/gradflow_check.py on a round-2 self batch at Kb 20 (n 16 items): nonzero R1 and R2 grads on SLVA, core, R, coda; decorators by ast; `--log_every 1` runs mass_stats |
| **G-S9 MEM** | on each FT host: `--mem_check_self --mem_check_steps 2` at the 64 longest n-16 records, K 20, bs 64; peak after step 2; self micro = the smallest in **{1, 2, 4, 8}** with peak ≤ 21.0 GiB (projected 4 at 14.4–15.0 GiB) |
| G-S10 TIME | forced G-TIME (odd steps deep self, even plain) on each host → projection → §8 admission |
| G-S11 OFF-HOST | full-state mirror every 2k for every cell incl. the 48k s2 base (user rule 2026-10-15), sha-checked after the first copy; resume per G-S4 / G-S4b |
| G-S12 FIXTURES | score_s216 fixtures (§4.6) pass before any output |
| **G-S13 DIFF REVIEW** (Audit) | T1; B1 (labeller path, exclusion set, counts, no simulator call); B2 (windows, fallback labeller); E1 (part_deep: items, K′ grid, per-item store, items_sha256 serialisation, fp32, tf32 off, bs); E2 / E3 (paths, ranges; round-1 behaviour bitwise-unchanged with round 1's arguments); **E4** (store added, per-n means unchanged) |
| G-S14 LAUNCH | 3090 only (card recorded per cell); drain waiters armed at launch; c = 0 logged; no module edit while any deploy is in flight; requeue watchers checked |
| **G-S15 GATES** | GATE-2 / GATE-1′ (and GATE-1-s2) per §2.1 on the 3070 (fp32, bs 32), eval_gate.json written **before** the pool is used |
| **G-S16 EVMEM** | on the 3070: part_deep's deepest chunk (the 16 longest n-24 items, K′ 28, fp32, bs 16) peak ≤ 7.0 GiB, **output to scratch/_tmp** (never into a cell directory), before any eval_deep output; else bs 8 (pre-stated). Also record that bs 32 OOMs or exceeds 7.0 |
| **G-S17 SMOKE-AT-SCALE** | the builder's exclusion set at full size (≈ 470k syms incl. 440k round-1 records) and the trainer's pool load / G-S9 deep-shape selection on the full 444k pool, locally, before deploy, timing logged (b310's O(N²) lesson) |
| **G-S18 BASELINES** (D2) | eval_deep on FT-SELF (**required**), D3-T s1 and FT-CTRL (optional), run and read **only after `s216_r2_s1` has launched**; eval_deep on `s216_r1_s2` (required, V3) only after `s216_r2_s2` has launched. BS-CHECK on each (aggregate Δ at every shared K′; FT-SELF's existing eval_selfdepth has items_n2 at n 9–16 only, so its flips are at K′ n + 2 only) |
| **G-S19 HIST24-STORE** (D5 / E4) | part_hist24 with E4 on FT-SELF, **output to scratch/_tmp**: per-n means bitwise equal to the committed s215_ftself_s1/eval_hist24.json; score_s216.hist24_items recomputes STAB from the items to 1e-12 of score_s213.hist24; before any R2 hist24 output |
| COMPETENT-BASE | s2 base, on its completed eval_vb, before GATE-1-s2 |
| s2 cells | G-S0 … S19 per s2 cell as applicable; G-S5 per §4.6 |

---

## 8. Cost, caps, admission, guards, wall-clock [DERIVED v1_memtime; price band $0.17–0.26/h ASSUMED]

| item | where | hours | projected $ | admission (G-TIME projection at the host's price) | in-flight guard (realised, incl. refused / BAD hosts) |
|---|---|---|---|---|---|
| s1 round-2 pool (444k; 77,325 draws at n 16) + GATE-2 / GATE-1′ (16k), FT-SELF labeller | 3070 | **1.3** (1.27 + 0.04) | 0 | — | — |
| **s216_r2_s1** | 3090 | 4.1–4.9 forced / 3.6–4.3 real | **0.69–1.27** | ≤ **$1.45** | **$1.70** |
| s216_d3t_s2 (48k) | 3090 | 6.2 forced + 0.4 (s1 real 4.96 h) | 1.11–1.70 | ≤ $1.75 | $2.00 |
| GATE-1-s2 + r1-s2 pool (440k, D3-T s2 labeller; D3-T s1's rate as proxy) | 3070 | ≈ 0.95 | 0 | — | — |
| s216_r1_s2 | 3090 | 2.6 forced (2.4 real) | 0.45–0.68 | ≤ $0.95 | $1.05 |
| GATE-2-s2 + GATE-1′-s2 + r2-s2 pool (444k) | 3070 | ≈ 1.3 | 0 | — | — |
| s216_r2_s2 | 3090 | as r2_s1 | 0.69–1.27 | ≤ $1.45 | $1.70 |
| Mirror egress | | | ≲ 0.05 | | |
| Evals: full panel per FT cell ≈ 4.5–5 h (eval_vb 65 min, deep, steppos, lfk, rs, d3hist, hist24, vbhalt, chainlabel ± fresh, selfdepth) × 3; s2 base ≈ 0.5 h (eval_vb acc part, chainlabel_fresh, hist24); baselines eval_deep × 2–4 (≈ 0.5–2 h); smokes G-S16 / G-S19 | 3070 | ≈ 15–18 | 0 | — | — |
| **Total (s1 R2 + s2 lineage)** | | | **$2.9–5.0** | admissions sum $5.60 | **family cap $6.5** (GUARD_STOP; guards sum $6.45) |

- **Confirmed ≤ $30**: family cap $6.5. **The user's explicit OK is required anyway** (MEM-TRADED, SELFDEPTH_v1 §9). Round 1 realised ≈ $1.07 for the pair [RELAYED by Theory].
- **Admission and retries** as SELFDEPTH_v1 §8 with notes 215-2 / 215-4: a projection above its bound means no launch on that host; at most 2 more **measured** refusals per cell; BAD hosts (never start) do not use a try but count toward the guard. **Steps are never reduced** (12k / 48k are the registered budgets). A guard stop is a dead run. **A cell no host admits is not run; the user is told; its readings read UNREAD** (m10). The R2 admission bound sits 14.6 % above the projected upper $1.27 (RT m10: the launch-bound 3090 speed varies up to ≈ 2× across hosts); a micro-8 host (projected $1.46–2.24) is refused.
- **Parallelism:** `s216_d3t_s2` needs no new code (G-S5: seed only) and can launch right after registration and the user's OK, alongside s1's engineering and labelling. The R2 cells run on separate 3090 hosts.
- **The 3070 is the bottleneck:** ≈ 19–22 h across the family (labelling ≈ 3.6 h + evals ≈ 15–18 h; RT estimated 20–25 h), plus §214's pending evals and FT-CTRL's remaining parts. One eval card per experiment (same-card rule).
- **Wall-clock:** s1 ≈ 11 h after code review (labelling 1.3 h, train ≈ 4 h, evals ≈ 5 h); the s2 lineage ≈ 24 h from its launch.
- **Retention:** keep model.pt of every FT cell and of `s216_d3t_s2` (labellers); keep evals, logs, args, pool_meta, eval_gate (with per-item records); delete checkpoints and mirrors at landing; keep `s215_pool/pool_r1.jsonl` until every s216 pool is built and G-S1-verified (m8); delete each s216 pool after its LABEL-AUDIT-2 / MEMO-GAP-2 have been scored (pool_meta + seeds + builder commit regenerate it).

---

## 9. Open items

1. **Reading of D1 on a dead base.** D1 writes COMPETENT-BASE = "the 48k run completed (not dead) ∧ …" and asks for RT B2's fix. I read it as RT B2 does: a dead base **fails** the ×2 row (dead = fail, rule b); only a completed base below the accuracy floor is "not competent". If Theory meant a dead base to read "not competent", one line changes in §2.4 and §4.5 (and the B2 blocker returns).
2. **E4 scope** is wider than D5's literal list: all six hist24 arms, n 3..8, both K′ n + 2 and K′ 24. STAB_arm is a difference of (a_{n+2} − a_24) terms and the hist24 gate reads n 3..8, so this is the minimum that lets Audit recompute the registered reading. Confirm.
3. **P(ABSORB-2) 0.65 and RETAIN 0.93** depart from RT's 0.60 / 0.95 because D10 changed the dose premise RT used. With RT's values the headline is 0.062 / 0.068 (sensitivity rows). Theory's call.
4. **New pre-statements (V1–V5):** ceil005 rounding of formula bars; FULL = M2 ≥ max(0.80, MG) ∧ M_far ≥ 0.60 with a separability flag; degenerate bars → UNLABELLED; R1-s2's eval_deep required (descriptive); one user decision for the family. Confirm.
5. **Admission $1.45 / guard $1.70 per R2 cell** (was $1.35 / $1.60): RT m10's headroom. The family cap stays $6.5.
6. **LIT-C2-2's floored variant** is descriptive (Goodhart, §5.3). If Theory prefers the floored form as the registered row, it should be decided before registration.
7. **A3′ on FT-SELF** is unverified (PROOFS #20); an eval_steppos long_step check on FT-SELF ($0, local 3070) would replace the [ASSUMED] tag. Not required by any gate.
8. **`--final` trigger:** who records a lineage's chain as done (a Theory note naming the lineage) should be named in the registration commit.
9. **Not resolved here by design:** the §215 pair (score_s215 PAIR + MEM attribution). While this text was being written, Theory committed the pair result (a51f4cff / 2d3fec56, 01:24; per the commit subjects only: SELF-ATTRIBUTED, MEM traded by the self items; nothing behind them was read). Per §2.5 that selects the row "P1 SELF-ATTRIBUTED, MEM traded by the self items" (KEEPS-SELF-STAB-2 prior 0.20, the headline's value), so no number in §6 moves. The user's OK question is Theory's to write from the scored pair (D12).
10. COMPETENT-BASE uses a_24 ≥ 0.97 (D1), the FT-DIVERGED level, while K′24-STABLE uses 0.99. Intended: competence is a floor, K′24-STABLE a kept reading. Stated so nobody "fixes" it.

---

## 10. Verification log (this revision)

- **v1_checks/v1_support_d10.py → v1_support_d10.json** [DERIVED, exact rationals]: the D10 S1 table and deadline costs; **G-S3 target 1/2 exactly** (SE 0.0004 at 383,488); the 50/50 run reproduces PROOFS' p2_support_s1.json (5/11 asserted; every P and deadline cost equal to 4 d.p.); per-micro-batch Kb under D10 (P(Kb = 20) 0.886 / 0.662 / 0.419 / 0.238 at 64 / 32 / 16 / 8 per chunk; E[Kb] 19.20 at 16); exposure (63,914.7 ± 85.2 per new n; 31,957.3 ± 63.2 per old n; share SD 0.0003; 1.377× / 0.862× round 1's sustained points; 2.80× its first-touch 0.01 point); pool 444,000, 60,512 spare, z(exhaust) 17.1 (s2 R2), 16.0 (s2 R1); LABEL-AUDIT-2 exact OC pooled and per window; RETAIN and K′24 binomials.
- **v1_checks/v1_support_check_d10.py → v1_support_check_d10.json**: PROOFS_SELFDEPTH's original support_check (`mixture()`, executed from research5/scripts/support_check_r1copy.py's source, not imported) on the D10 and 50/50 laws; both agree with v1_support_d10.json to 1e-12 (asserted).
- **v1_checks/v1_panels.py → v1_panels.json** [DERIVED from MEASURED s213_d3t_s1 / s215_ftself_s1 eval_vb]: `r1_bars` / `r2_bars` with s1 instance asserted (0.63, 0.40, 0.785, 0.60; H_prop M_far upper 0.362, no flag; not degenerate); per-n panels (H_shift / H_prop / constant growth δ 1.1546 / SHRINK / H_full floor): M2 0.7172 / 0.8516 / 0.8954 / 0.4798 / ≥ 0.9375; n50 19.460 / 20.614 / 20.609 / 18.306; WIDTH-2 1.526 / 2.224 / 1.569 / 1.234; 2-D readings on PROOFS' table (replicated, not imported) on both axes: STOP 20 (G(19) 0.0231) / 21 / 21 / 18; KB2 0.155 / 0.291–0.317 / 0.512–0.562 / −0.002; LIT-C2-2 0.005 / 0.036–0.046 / 0.030–0.038 / −0.004 / 0.713; lineage-noise table (reproduces rt_curves.json's σ rows exactly for H_shift / H_prop, within 3e-4 for constant growth: RT used δ 1.1532, this script the exact 1.1546).
- **v1_checks/v1_oc.py → v1_oc.json**: headline, joint ABSORB-2 / K′24 model and bounds, every sensitivity row; **RT's shaded OC (0.06357609, ×2 0.0045266) and PROOFS' re-shade (0.0700437) reproduced exactly (asserted)**.
- **v1_checks/v1_memtime.py → v1_memtime.json** [CPU generator draws, read-only import of boxes.nested170, no bytecode]: n-16 T max 61 / 61 / 61 / 61 / 63 over 5 seeds × 77,325 draws (64th longest 59); GiB at K 20 for micro 1 / 2 / 4 / 8; D10 G-TIME chunk mean max-T 52.24 (16 per chunk); compute-bound refit 1.763 s (PROOFS' 50/50 refit 1.728 s reproduced); launch-bound 2.241 s; real / forced 0.872; projections and costs; labelling hours (s1 R2 pool 1.27 h + gates 0.04 h; s2 R1 pool 0.92 h at D3-T s1's rate); family $2.94–4.96, admissions $5.60, guards $6.45.
- **v1_checks/v1_noise_d10.py → v1_noise_d10.json** [DERIVED from MEASURED s215_ftself_s1/eval_chainlabel_fresh.json]: kept-wrong 2 / 3,929 (W_2), 0 / 3,997 (W_1); D10 pool w 3.4e-4; κ direct-w 5.28 / 1.68 (W_2), 7.92 / 2.04 (D10 pool), 10.65 (50/50, PROOFS reproduced); class mass 1.02e-3 / 3.20e-3; cumulative 1.82e-3.
- **Reproduced from the reviews without change:** PROOFS' GATE-2 / GATE-1′ / π^E OCs, κ φ-form, two-round factors, T-split statistics, eval-memory slopes; RT's lineage-noise table and P1 observation (CTRL acc_gain_after10 −0.021, RT's derived value; not re-read).
- **Code read** [VERIFIED]: vbank.py l.553–560 (CurGate final stage), l.700–770 (self auto-micro loop `(1, 2, 4)` at l.706 / raise l.719; draw / draw_step: rng_s coin + 64 U per self step), l.808–830 (self steps excluded from CurGate; §215 log fields); vbank_diag.py part_hist24 (per-n means only, l.948–949), _acc_n2_items, part_selfdepth (s215 paths, items_n2 n 9..16 only, contamination vs n_max 20), main (output path = the run dir, hence the scratch-output rule in G-S16 / G-S19); theory/score_s215.py (after 215-7: ARGS-DIFF BASE / ALLOWED, pending rule, cell(), gate1()); score_s213.hist24 (STAB_arm definition, gate on n 3..8), score_s213.META.
- **Raw files read** (explicit paths only): s213_d3t_s1 / s215_ftself_s1 eval_vb.json; s215_ftself_s1 eval_hist24.json, eval_chainlabel_fresh.json; s213_d3t_s1 eval_chainlabel_fresh.json; theory/results/score_s215.json (SELF block, read at ≈ 01:05 when PAIR still read UNREAD). FT-CTRL: none (directory listing only, to explain a `find -newer` hit). Repo commits after 01:05 (a51f4cff, 2d3fec56: the §215 pair result) were seen by subject line only (§9 item 9).
- **Not read, on purpose:** s215_ftctrl_s1/eval_hist24.json, eval_d3hist.json.
- **No repo write:** `find D:/Latent/boxes D:/Latent/theory -newer v1_oc.py` after the CPU runs showed only s215_ftctrl_s1/eval_vbhalt.json (01:13:38), written by FT-CTRL's running eval chain on the 3070, not by these scripts (all output paths are research5/v1_checks/; `python -B`, PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1, nice 19 / idle priority).
- **THEORY.md:** notes 214-5 and 215-7 read (lines 18733–18758 at 6c49b973); there is no note 215-6.
- **Still unbuilt (flagged):** T1, B1, B2, E1–E4, score_s216, queue scripts. Memory, time and eval-memory projections are model extrapolations; G-S9 / G-S10 / G-S16 decide.

*v1 written 2026-10-18 01:08–01:29 EDT (Opus design-revision subagent for Theory; $0; nothing committed, registered or launched).*

---

## Addendum 1 (2026-10-18 01:31 EDT; Audit's review of 3afb01e3, before registration): pre-stated eval-card fallback

The 3070 dropped off the bus at ~22:56 10-17 ("GPU is lost") and runs hot. eval_deep (n 13–24, K′ 0..28) is the longest eval in the family.
1. **3070 lost mid-chain, then recovered** (reboot): the parts not yet complete re-run **on the 3070** after recovery, as on 10-17/18. Partial outputs are kept as `*.void_gpulost`, never deleted. Completed parts stand if their gates (the d3hist gate, BS-CHECK, items_sha256) pass after the re-run.
2. **3070 not recoverable** (fails again after a reboot, or is removed): **every registered eval of both lineages moves together to one named card, the local RTX 4060 (8 GB), fp32, the same eval_bs per part** (eval_vb 32, eval_deep 16, G-S16 re-run on the 4060 first).
   - **Reference re-evals come first on that card:** eval_vb + eval_deep on s213_d3t_s1 and s215_ftself_s1 (and, for s2, its own base and round-1 model). No SD2 cell output is read on the 4060 before they exist.
   - **The s1 bars are recomputed by the registered §4.3 formulas** from FT-SELF's 4060 curves (the B1 per-lineage formula machinery, applied per card). Every reading compares on that card only, and the card is recorded per file.
   - 3070 outputs of SD2 cells already landed stay on record and are reported beside, never mixed into a reading.
3. A partial move (some parts on each card) is not allowed within a cell's registered reading set.
