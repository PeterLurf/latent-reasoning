# SELF-DEPTH round 3 (R3) draft v0: is the memory loss in self rounds carried by one parameter group? Freeze it and find out

Clock 2026-10-21 14:33 EDT (`date`), repo HEAD 4b3c3be7. Draft subagent for Theory. $0.
**Status: DRAFT v0. Not reviewed, not registered.** Next: red team + proofs → v1. Stage B needs **the user's explicit OK** whatever its cap: MEM-TRADED-2 read in §216, and SELFDEPTH_v1 §9's rule carries over to any round 3 (216-5).
**Inputs:** NORTH_STAR (C1 > C6 > C4 = C2 > C3 > C5 > C7); THEORY §215, §216, Notes 216-1…216-13, §217 close, §219 + Note 219-8; SD2_v1.md (structure, machinery); LIT_MUX_FORGETTING_2026-10-20 §2; LIT_INTERNAL_ONLY_FORGETTING_2026-10-21; boxes/vbank.py, boxes/slva.py, boxes/vbank_diag.py (part_hist24, part_stabhist), theory/score_s213.hist24, theory/score_s216.py (function list).
**Not read, on purpose:** any §218 lvb_direct_s1 file, the s130 TEST replays, any s162 output, the sealed n 25–30 items (216-5; only their published hash and T_max are used).
**$0 checks for this draft** (this folder; CPU, 2 threads, read-only on committed files; outputs are JSON files next to each script): r3_logsplit.py, r3_param_drift.py, r3_oc_sim.py (§10).
**Tags:** [PROVEN] (by construction / exact), [DERIVED] (arithmetic on measured inputs; script named), [MEASURED] (raw file named), [ASSUMED], [CONJECTURE].

---

## 0. One screen

| item | R3 v0 |
|---|---|
| Question | Self rounds cut STAB_self (the drop in K′24 accuracy when SLVA's own-slot versions 1..t−2 are masked) from 0.782 to 0.489 to 0.437, while outputs stay put (216-13 INTERNAL-ONLY). **Is that change carried by one localisable parameter group?** If so, does freezing that group during self-training keep the mechanism *and* still let the round add depth? |
| Hypothesis | **H_LOC:** the self-item-specific part of the STAB loss (FT-CTRL 0.696 → FT-SELF 0.489, Δ 0.207) is carried by one non-core parameter group G. A priori G = SLVA addressing (`slva_*`: W_q, W_k, LN_q, LN_k, beta0, gamma). Rival **H_DIFF:** the change is spread through the core / coda / prelude and co-adapted, so no single-group freeze keeps it |
| Single change (lever) | **FREEZE-G:** `requires_grad False` on G before the optimiser is built, in the self round only. Nothing else changes (args, pool, seed, steps, lr). G is chosen by a pre-stated $0 PARAM-SWAP rule (§7.2) before v1. Prior choice: `slva_*` (0.94 % of parameters) |
| Stage A (the mechanism test) | Re-run **round 1** with FREEZE-G in both lineages, each paired with its existing no-lever twin (same seed, pool, coins and plain batches): **A-s1** `r3a_frz_s1` (twin FT-SELF, control FT-CTRL); **A-s2** `r3a_frz_s2` (twin R1-s2, which was FT-DIVERGED). There is no new control cell |
| Stage B (round 3 proper; only if Stage A carries) | From `s216_r2_s1`, labelled by R2-s1's own chain: **R3-FRZ** (the lever) and **R3-PLAIN** (the one ablation arm, lever off). Self items ⅔ n 17–20 / ⅓ n 13–16, self_kmax 24. Extrapolation: n 21–24 (eval_deep) + **the sealed n 25–30** (216-5, sha 3938fdf3…) |
| Primary readings | A: **MECH-CARRIES** = in both seeds ΔSTAB = STAB_self(FRZ) − STAB_self(twin) ≥ 0.08 ∧ no OVERSHOOT ∧ ABSORB. B: **R3-STABLE-DEPTH** = ABSORB-3 ∧ RETAIN-3 ∧ K′24-STABLE ∧ MEM-HELD-3 ∧ DBS-3 on R3-FRZ |
| OC | A (2 seeds, σ_rep 0.04): **P(pass \| no effect) 0.006** (0.000–0.027 for σ_rep 0.02–0.06); **P(pass \| full effect) 0.50** (0.41–0.61); half effect 0.17. B: 0.13–0.17 \| no effect vs 0.30–0.41 \| effect. B is weakly discriminating by design; A is the test [DERIVED r3_oc_sim] |
| Cost | Projected A $0.9–1.4, B $2.8–4.0, **family $3.7–5.4; family cap $7.0** (GUARD_STOP). PARAM-SWAP, pools, gates and evals run locally ($0). Realised at draft time: $0 |
| Approval | A: Audit, under the user's $30 standing rule. B: the user's explicit OK (MEM-TRADED-2 rule), asked after A is scored |

---

## 1. Origin: what failed, and the constraint each failure sets

| # | failure / finding (source) | constraint on R3 |
|---|---|---|
| O1 | **MEM-TRADED, attributed to the self items** (§215 pair): STAB_self D3-T 0.782 → FT-CTRL 0.696 → FT-SELF 0.489, on identical plain batches (G-S3) | The effect to explain is **0.207, self-specific**, measured in round 1. A mechanism test should run where that effect is largest and a paired control already exists: round 1 from D3-T |
| O2 | **MEM-ERODING** (§216 R2): 0.489 → 0.437 (Δ 0.052) | A round-3-only design starts from 0.437. The no-lever drop there may be about 0.05, about one model-noise σ. **A round-3 pair alone cannot test the mechanism** (§5) |
| O3 | **R1-s2 FT-DIVERGED** (216-8): a_24(1) 0.921, a_24(2) 0.973 on a base already at the edge (0.994 / 0.987). FT-CTRL stays at 1.000 | The lever must be read for small-n K′24 output stability as well. s2 is where the harm reached outputs, so s2 is the stress seed |
| O4 | **INTERNAL-ONLY** (216-13): KL(D3-T ‖ ·) at n 3–6, K′24: FT-SELF 0.00187 vs FT-CTRL 0.00179, ratio 1.04, CI95 [0.76, 1.41]. R2 shows output drift beginning (0.0151) | **Output-level levers do not touch round 1's loss** (on-policy relabelling, forward-KL replay, restart-from-base as a drift fix). The lever must act on internals |
| O5 | **Causal use preserved** (§217): VALUE-SWAP follow 1.000, STALL ≥ 0.99 in every competent band | The lever must not cost C1. VB-C1 stays a conjunct; vbank_causal is an optional descriptive row |
| O6 | **n50 / support 1.28–1.31** over 5 points (216-9) | Expected R3 depth at support 20: n50 ≈ 25.7 (H_prop) or 24.5 (H_shift). The sealed n 25–30 straddles both, so it is where an n50 is readable |
| O7 | **Text baseline** (§219 T1, 219-8): n50 14.606, n50 / support 1.217 vs latent R1 1.29; Δ_1 −0.854; T2 pending | R3 is latent-only. The text n50 constants are cited on the R3 n50 line, descriptively only |
| O8 | **$0, this draft — weights don't localise** (r3_param_drift): per group, ‖θ_SELF − θ_CTRL‖² shares follow the share of gradient-receiving parameters (slva 1.45 % vs 1.37 %; core attn 12.1 %; prelude MLP 29 %); cos(θ_SELF − θ_D3T, θ_CTRL − θ_D3T) = 0.13–0.16 in **every** matrix group | Adam's update norms carry no localisation. **G must be chosen functionally (PARAM-SWAP, §7.2), not from weights** |
| O9 | **$0, this draft — gradient by batch type** (r3_logsplit): self batches carry 20–650× plain batches' gn_preclip (FT-SELF 1–4k: 2.40 vs 0.0128). SLVA's share of the grad norm is 0.04–0.08 on **both** types. On plain batches, SLVA off-newest mass on copy slots is unchanged (FT-SELF 0.267 → 0.272; FT-CTRL 0.275 → 0.281) | The self items' push does not single out SLVA at the trained horizon (K ≤ 12). H_LOC with G = slva gets a **low prior (0.20)**. Hence the functional selection rule |

---

## 2. Hypothesis and the single method change

### 2.1 What STAB_self measures, and why "forgetting" is the wrong word [MEASURED eval_hist24, n 3–8]
- self_ablate (SLVA's own-slot versions 1..t−2 masked; V_0 and V_{t−1} kept) gives acc_k24 at n 3–8 of:

| model | acc_k24 under self_ablate, n 3 … 8 | STAB_self |
|---|---|---|
| D3-T s1 | 0.206 / 0.206 / 0.231 / 0.230 / 0.275 / 0.244 | 0.782 |
| FT-CTRL | 0.317 / 0.283 / 0.292 / 0.323 / 0.327 / 0.279 | 0.696 |
| FT-SELF | 0.429 / 0.460 / 0.552 / 0.602 / 0.645 / 0.650 | 0.489 |
| R2-s1 | 0.465 / 0.516 / 0.618 / 0.644 / 0.731 / 0.753 | 0.437 |
| D3-T s2 → R1-s2 | 0.307 … 0.505 → 0.412 … 0.794 | 0.684 → 0.489 |

- Intact acc_n2 = 1.000 everywhere, and intact acc_k24 ≥ 0.997 at n 3–8.
- **So the self rounds do not lose stability at n 3–8.** They become *less dependent* on intermediate history: holding from {V_0, V_{t−1}} alone works more often. The registered loss is of the history channel's causal role in holding (KEEPS-SELF-STAB).
- **The output harm sits at the longest idle stretch** (n 1–2, 22–23 idle passes; training never has more than 4): R1-s2 a_24(1) 0.921.
- **[CONJECTURE] HOLD-SHIFT:** self items move idle-pass holding from history-anchored to state-anchored (Markov) holding. A Markov hold is unconstrained past the trained idle horizon, so its drift shows first at n 1–2. R3 asks *where* that shift lives in the network.

### 2.2 Hypothesis
- **H_LOC:** the self-specific part (Δ 0.207, O1) is carried by one non-core group G. Freezing G in the self round keeps STAB_self near FT-CTRL's level while ABSORB still holds.
- **H_DIFF (rival):** the change is spread out and co-adapted. A G-freeze keeps no more than half of it, or blocks ABSORB.
- **H_NEED (a second rival, a finding either way):** the depth gain needs G to move. FREEZE-G blocks ABSORB → FREEZE-BLOCKS-DEPTH.
- This is a test of *why* the rounds degrade, namely *which part carries it*. A positive gives a lever that is also the fix. A negative closes the single-group route without spending on a patch loop.

### 2.3 The lever and why it beats the other Literature candidates
| candidate (Literature 838991c2) | fit to O1–O9 | decision |
|---|---|---|
| **Freeze the bank read / write** (Zheng 2501.13453; "freeze the mechanism's parameters") | In SLVA the history read is `slva_*`. No new loss means no Goodhart target. One flag, with a precedent (D3-F froze wpe with a sha check). **Directly tests H_LOC** | **PRIMARY**, as FREEZE-G with G from §7.2 |
| **W_V-only updates** (Lee, Sohn & Lee 2602.23197) | SLVA has **no W_V**: values are N(V_s) straight from the core (slva.py docstring). "Fix key / query addressing" = freezing `slva_wq/wk/ln_q/ln_k` = the freeze above. Precondition check: the theorem is for linear attention; SLVA is a softmax read, so **the theorem does not transfer**, only its empirics | Merged into the primary (G = slva_addr) |
| **LDIFS feature distance on K′24 slot states** (Mukhoti 2308.13320) | (i) Near an idle fixed point the intact pass input does not depend on the history weights (§3, D1), so state matching cannot see the change in history reliance; (ii) a one-sided closeness target on dynamics (the CSR lesson) can block depth learning on shared parameters; (iii) needs a frozen-base forward at K′24 every step (≈ +40 % step time [ASSUMED]) plus a λ search | Rejected for v0 |
| **216-9 small-n K′24 rollout mix** | Its targets were **already met** at n 3–8 (FT-SELF a_24 1.000) while STAB fell 0.207. With CE ≈ 0 there [ASSUMED], it adds no gradient where STAB lives (§3, D2). It acts only at n 1–2, so it is an *output* guard for O3, not a test of O4 | Not the lever. Open question Q4 (a Stage B guard would be a second change) |
| On-policy relabelling / forward-KL replay / restart from base | Ruled out by O4 (output-level) | Rejected |

### 2.4 Cells
- **Recipe:** each FRZ cell = its twin's args.json + `--freeze_groups <G>` + `out`. Nothing else differs; ARGS-DIFF allows only {freeze_groups, out} ∪ META.
- **A-s1 `r3a_frz_s1`:** init s213_d3t_s1/model.pt; pool = s215_pool/pool_r1.jsonl, rebuilt bitwise (scratch/s215_pool_restore.sh; sha c2e226ae… matched once before, CURRENT_TASKS 10-18); seed 1.
- **A-s2 `r3a_frz_s2`:** init s216_d3t_s2/model.pt; pool s216_pool_r1_s2/pool_r1.jsonl (on disk; sha in pool_meta); seed 2.
- **Pairing [PROVEN by construction, VERIFIED vbank.py draw_step l.744–770]:** the plain draw (rng, rng_k) and rng_s's coin and U do not depend on parameters. With the same seed and pool, the FRZ cell replays its twin's plain batches, coins and self records exactly. G-S3 (plain_hash identity at steps 1..200) is the check.
- **B `r3_frz_s1` / `r3_plain_s1`:** init s216_r2_s1/model.pt (sha in its pool_meta); R2's args with only init_from, self_pool, `--self_kmax 24`, self_micro (from G-S9), out and (FRZ only) freeze_groups changed. 12k steps, lr 3e-5, seed 1. The two cells share pool, coins and plain stream (paired).
- **B labeller:** R2-s1's re-rooted chain (fresh keep n 17–20 0.993 / 0.977 / 0.973 / 0.956, prec_kept ≥ 0.9956, LABELLER-KEPT-2). **GATE-3** (n 17–20) and **GATE-2′** (n 13–16) use GATE-1's rule: π_pool ≥ 0.995 ∧ π_n ≥ 0.99 ∧ π^E_pool ≥ 0.99 ∧ keep_n ≥ 0.70. Fallback per window: FT-SELF; then no Stage B.
- **B pool:** 74,000 kept per n at 17–20 and 37,000 per n at 13–16 (444k), one fixed interleaved order. Candidate seeds (grep: 0 hits in THEORY / boxes / theory / scratch): pool 23100 / 23101, gates 23160 / 23161, audit 23170.
- **B exclusion:** test_set(1000, n_max 30) (incl. the sealed n 25–30), SERIAL, 213-37 fresh items, every gate, and every earlier pool (r1 s1 / s2, r2 s1). Stage A's pools are n 9–12 only, so they cannot contain any n ≥ 13 test item [PROVEN by n-disjointness].

---

## 3. Derivation

**Assumptions**
- A1 [ASSUMED] STAB_self (score_s213.hist24 `STAB['self_ablate']`, mean over n 3–6) measures how much K′24 holding causally relies on SLVA's versions 1..t−2.
- A2 [ASSUMED] Model noise: the same recipe on another 3090 host gives STAB_self ± σ_rep, with σ_rep 0.04 central and 0.02–0.06 for sensitivity. Item SE ≈ 0.008 (4,000 items) is folded in.
- A3 [ASSUMED] s2's FT-only STAB drop equals s1's (0.086). So s2's CTRL level ≈ 0.598 and its self-specific effect ≈ 0.109.
- A4 [ASSUMED] A STAB change scales with the movement of the parameters that carry it (used only in D4).

**Claims**
- **D0** [PROVEN by construction] With θ_G frozen, θ_G is bitwise the init's at every step. Any STAB change in an FRZ cell must therefore flow through the other groups. In particular, SLVA's weights α still move through the keys W_k LN_k(V_s), because V_s comes from the core. **Freezing fixes the addressing *function*, not the addressing *behaviour*.**
- **D1** [DERIVED under a fixed-point assumption] If V_s[j] = V*[j] for every idle version s ∈ S_idle, then U_t[j] = Σ_s α_s N(V_s)[j] = N(V*)[j] for any α supported on S_idle ∪ {t−1}. So intact idle-pass states are invariant to how attention is spread over idle versions. A state-matching penalty (LDIFS) at K′24 on intact states cannot constrain the reliance that self_ablate reveals. It holds only approximately: history versions 1..n−1 are compute versions, and idle states drift.
- **D2** [DERIVED from MEASURED eval_vb + ASSUMED CE ≈ 0] FT-SELF has a_24 = 1.000 at n 3–8. A K′24 replay loss there has ≈ zero gradient, so it cannot oppose the STAB fall in the bins STAB averages. Its only lever is at n 1–2.
- **D3** [DERIVED, r3_param_drift] The self contribution's weight displacement ‖θ_SELF − θ_CTRL‖² is distributed like the gradient-receiving parameter count: slva 1.45 % vs 1.37 %, core attn 12.1 % vs 11.0 %. The SLVA scalars barely move: γ heads 9–11 are D3-T −0.0784 / −0.0853 / −0.0866, CTRL −0.0773 / −0.0862 / −0.0873, SELF −0.0795 / −0.0877 / −0.0887, R2 −0.0819 / −0.0914 / −0.0920. At age 22 SELF − CTRL is ≤ 0.05 logits. **Weights neither support nor refute H_LOC.**
- **D4** [DERIVED under A4] Freezing slva_addr removes ≤ 1.5 % of the self contribution's squared displacement. "Less total drift" cannot then explain a ΔSTAB ≥ 0.08 (≥ 39 % of 0.207). **A Stage A pass with G = slva is group-specific.** For G = prelude (≈ 44 % of the self contribution's displacement) this specificity argument does not hold; Q2 asks for a size-matched control.
- **D5** [DERIVED, r3_oc_sim] OC (§5).
- **D6** [DERIVED from R2's committed curve; H forms CONJECTURE] Panels for R3 at support 20:
  - H_shift (R2's margin carried +4): M3 = mean a(21..24) = R2's mean a(17..20) = **0.8745**; mean a(25..28) = R2's mean a(21..24) = **0.208**; n50 = **24.54**.
  - H_prop (R2's curve scaled ×20/16, linear interpolation): a(21..24) = 0.998 / 0.987 / 0.932 / 0.821, M3 = **0.935**; a(25..28) = 0.659 / 0.423 / 0.269 / 0.180, mean **0.383**; n50 = **25.68**.
  - **DBS-3 (≥ 0.80) passes under both**, so it reads "depth maintained", never "growth". Growth form is read only descriptively, from N50-3 on n 21–30.
- **D7** [DERIVED] Training memory and time at R3 shapes, from SD2 §3.5–3.6 scaled by K × T (self K 24 vs 20; mean chunk max-T ≈ 52 × 75 / 61 ≈ 64 [ASSUMED T(n 20) ≈ 75]): self step ×1.48 (compute-bound) to ×1.73 (∝ K² T) → 2.6–3.9 s; plain 0.41 s. G-S9 picks micro from {1, 2, 4, 8}.

**Predictions** (rough; every number [ASSUMED])

| # | prediction | P |
|---|---|---|
| P1 | PARAM-SWAP: some non-core G has RECOVERY ≥ 0.5 (§7.2) | 0.40 (slva 0.20, prelude 0.12, coda 0.08) |
| P2 | Only core reverts recover ≥ 0.5 (→ no Stage A; re-plan) | 0.30 |
| P3 | Given a G and Stage A run: s1 SEED-PASS | 0.60 |
| P4 | MECH-CARRIES (both seeds) | 0.40 |
| P5 | FREEZE-BLOCKS-DEPTH in either seed | 0.15 (slva) / 0.25 (prelude, coda) |
| P6 | DIV-PREVENTED (s2 a_24(1) ≥ 0.97), given MECH-CARRIES | 0.60 |
| P7 | Stage B R3-STABLE-DEPTH, given Stage A carried | 0.35 |
| P8 | N50-3 ≥ 24.5 on R3-FRZ | 0.60 |
| P9 | FREEZE-COST (n50 R3-FRZ − R3-PLAIN < −0.5) | 0.25 |
| — | End to end, P(R3-STABLE-DEPTH) ≈ 0.40 × 0.6 × 0.35 | **≈ 0.08** |

**Goodhart analysis**
- **FREEZE-G is a constraint, not a target.** Nothing is optimised toward a measured quantity. The model can route around it through the states (D0); that is the H_DIFF outcome, and Stage A reads it.
- **STAB_self as a pass criterion is one-sided in §215 / §216 (≥ 0.5).** A model made fragile to *any* masking would score high without "better memory". Mitigations (new here):
  - (a) **two-sided:** OVERSHOOT = STAB_self(FRZ) > STAB_base + 0.10 is not a pass;
  - (b) intact acc_k24 ≥ 0.99 at n 1–8 in s1 (K′24-STABLE);
  - (c) the whole hist24 profile is reported: bank_ablate_slva, dup_newest, dup_foreign. A fragility artefact would also inflate dup_foreign's drop, which is at chance in every model now.
- **A trivial pass by not learning** (frozen model ≈ base) is blocked by the ABSORB conjunct. A cell that does not absorb never reads MEM kept.
- **PARAM-SWAP picks G on the same statistic Stage A reads.** It does so on spliced post-hoc models, though, and Stage A is a fresh training run. Selection cannot inflate Stage A's own D. Disclosed: the effect-size priors (A3) use the same STAB values.

---

## 4. Registered readings and pass criteria (reusing §215 / §216 machinery)

**Scorer:** theory/score_r3.py (name to follow the § number), importing theory/score_s216 (r1_cell / r2_cell logic, args_diff + args_ref_check, label_audit, contamination2, deep_read, stab_self, the 215-7 pending rule) and theory/score_s213.hist24. New LIN entries: `s1frz`, `s2frz`, `r3frz`, `r3plain`. Fixture-checked before any output. Every missing registered file reads pending.

**Eval contract:** local fp32, tf32 off, on the 3070 (the §215 / §216 card; Addendum-1 fallback to the 4060 with reference re-evals first). Parts:
- eval_vb (bs 32), eval_hist24 (bs 32, with E4 items), eval_steppos, eval_lfk, eval_selfdepth (round-parameterised), eval_chainlabel_fresh.
- Stage B adds eval_deep (n 13–24, K′ 0..28, bs 16) and the new **eval_deep30**: n 25–30 of test_set(1000, n_max 30), K′ 0..34, per-item answers, items_sha256 must equal 216-5's `3938fdf3…`; eval_bs from G-EVMEM (T_max 105; 8 expected).

### 4.1 Stage A (per FRZ cell; twin = FT-SELF for s1, R1-s2 for s2)
| reading | definition (field) | role |
|---|---|---|
| ABSORB | min acc_n2 n 9–12 ≥ 0.95 (eval_vb) | conjunct |
| ΔSTAB | STAB_self(FRZ) − STAB_self(twin) (hist24 `STAB['self_ablate']`) ≥ **0.08** | conjunct |
| OVERSHOOT | STAB_self(FRZ) > STAB_base + 0.10 (s1 0.882, s2 0.784) → not a pass | guard |
| K24-s1 | s1 only: acc_k['24'] ≥ 0.97 at every n 1–8 (twin met it) | conjunct (s1) |
| FROZEN-INTACT | sha256 of G's tensors in model.pt = the init's; else **FREEZE-BREACH** (censor) | check |
| **SEED-PASS** | ABSORB ∧ ΔSTAB ≥ 0.08 ∧ ¬OVERSHOOT (∧ K24-s1 in s1) | per seed |
| **MECH-CARRIES** | SEED-PASS in both seeds | **primary** |
| MECH-PARTIAL | exactly one seed | label |
| FREEZE-BLOCKS-DEPTH | ¬ABSORB in FRZ while the twin absorbed (both did: 1.000 ×4) | label (H_NEED) |
| MECH-NOT-CARRIED | ABSORB in both, SEED-PASS in neither | label (H_DIFF) |
| RECOVERY_s1 | (STAB_FRZ − 0.489) / (0.696 − 0.489) | descriptive |
| DIV-PREVENTED (s2) | acc_k['24'][1] ≥ 0.97 ∧ [2] ≥ 0.97 (R1-s2 0.921 / 0.973) | secondary (C6 output) |
| ΔM | M(FRZ) − M(twin), M = mean acc_n2 n 13–16 (twins 0.717 / 0.688) | descriptive (depth cost) |
| VB-C1, LENGTH-FREE, LABEL-AUDIT, CONTAMINATED, MEMO-GAP | as score_s216.r1_cell | as registered |

### 4.2 Stage B (R3-FRZ; R3-PLAIN read with the same rows)
| reading | definition | role |
|---|---|---|
| ABSORB-3 | min a(n), n 17–20 ≥ 0.95 (eval_deep; every n ≥ 17 value comes from eval_deep) | conjunct |
| RETAIN-3 | min a(n), n 9–16 ≥ 0.99 (eval_vb) | conjunct |
| K′24-STABLE | a_24 ≥ 0.99 at n 1–8 (eval_vb); FT-DIVERGED = a < 0.99 or a_24 < 0.97 at any n 1–8 → censor | conjunct |
| MEM-HELD-3 | STAB_self(R3-FRZ) ≥ STAB_self(R2-s1) − 0.03 = **0.407**, ∧ ¬OVERSHOOT (≤ 0.537) | conjunct |
| DBS-3 | M3 = mean a(n), n 21–24 ≥ 0.80 ("depth maintained past support"; non-discriminating vs H_shift, D6) | conjunct (C2 proxy) |
| **R3-STABLE-DEPTH** | ABSORB-3 ∧ RETAIN-3 ∧ K′24-STABLE ∧ MEM-HELD-3 ∧ DBS-3 | **primary** |
| N50-3 / n90 / WIDTH-3 | over n 21–30 (eval_deep + eval_deep30), the SD2_v1 §4.3 n50 rule; panels 24.54 / 25.68 | descriptive |
| FAR-30 | mean a(n), n 25–28; panels 0.208 / 0.383 | descriptive |
| LIT-C2-3 | mean over n 26–28 of [a(K′ n+2) − a(K′ 24)] ≥ 0.10 | descriptive (literal C2) |
| Pair rows | ΔSTAB-3 = FRZ − PLAIN; FREEZE-COST = Δn50 < −0.5; Δa_24 at n 1–2 | descriptive |
| TEXT line | N50-3 / support beside §219 T1 1.217 (T2 when scored) | descriptive |
| GATE-3 / LABEL-AUDIT-3 / CONTAMINATED / MEMO-GAP-3 / VB-C1 / LENGTH-FREE | as SD2_v1 §4.4 with the windows shifted | as registered |

**Wording:**
- A is a mechanism reading at round-1 shape: never C2, never C4.
- B is ABSORB-3 in support. DBS-3 is a C2 proxy (one lineage: an instance). It is never worded as growth.

---

## 5. OC (r3_oc_sim.py → r3_oc_sim.json; every prior [ASSUMED])
**Model.** Per seed, D = δ + e with e ~ N(0, √2 σ_rep), since D compares two independent FT runs.
- SEED-PASS = D ∈ [0.08, STAB_base + 0.10 − STAB_twin] ∧ ABSORB, with p_abs = 0.95 / 0.90 / 0.85 under none / half / full.
- δ_s1 = 0.207 (full) / 0.1035 (half) / 0; δ_s2 = 0.109 / 0.0545 / 0 (A3).

**Arithmetic, σ_rep 0.04** (√2 σ = 0.0566):
- No effect: P(Z ≥ 0.08 / 0.0566 = 1.41) = 0.079, × 0.95 = **0.075 per seed → 0.0056 for both**.
- Full effect, s1: P(Z ≥ (0.08 − 0.207) / 0.0566 = −2.24) ≈ 0.987 × 0.85 = **0.839**. s2: P(Z ≥ −0.51) ≈ 0.695 × 0.85 = **0.591**. Both **0.496**.

| σ_rep | P(MECH-CARRIES \| none) | \| half | \| full | P(MECH-PARTIAL \| none) |
|---|---|---|---|---|
| 0.02 | 0.000 | 0.119 | 0.612 | 0.004 |
| **0.04** | **0.006** | **0.175** | **0.496** | 0.138 |
| 0.06 | 0.027 | 0.187 | 0.411 | 0.274 |

- Monte Carlo (200k) matches the closed form to ≤ 0.003.
- **Reading power:** a pass is strong evidence for H_LOC (likelihood ratio ≈ 90 at σ 0.04). A fail is weak evidence against it: s2's effect is small (A3), so s2 limits power (Q3).
- MECH-PARTIAL is common under no effect at σ 0.06 (0.27), so it is never worded as support. It only gates Stage B when s1 passes (§6).

**Stage B, one lineage.** R3-STABLE-DEPTH:

| case | AR | K24 | MEMH | DBS (σ_lin 0.06–0.12, μ 0.8745–0.935) | product |
|---|---|---|---|---|---|
| no lever effect | 0.75 | 0.60 | 0.386 | 0.73–0.99 | **0.13–0.17** |
| lever effect | 0.75 | 0.80 | 0.692 | 0.73–0.99 | **0.30–0.41** |

- **B discriminates weakly** (likelihood ratio ≈ 2.4). It is the application step; the mechanism claim rests on A.

---

## 6. Censoring, launch rules, cost

**Censoring (rule b: each fails every reading of its cell):**
- dead (last step < 12,000, guard stops included), nonfinite, CONTAMINATED, LABEL-NOISY, ARGS-DIFF (anything beyond {freeze_groups, out} ∪ META vs the twin, plus the 216-7 args_ref rule), FREEZE-BREACH.
- A Stage B cell is also censored on FT-DIVERGED or a GATE-3 failure of both labellers.

**Never-reached events (the user's 10-14 rule):**
- ABSORB never reached in an FRZ cell → SEED-PASS false, labelled FREEZE-BLOCKS-DEPTH. **Never read as MEM kept.**
- In Stage A, FT-DIVERGED is a *reading* (DIV), not a censor: s2's twin diverged, and the lever is meant to act on it. s1 needs K24-s1.
- n50 never crossed by n 30 → reads **"> 30"**: it passes any n50 ≥ bar, the ratio reads "> 1.50", and it is never missing. A crossing at n ≤ 20 → ABSORB-3 fails anyway.
- n90 not crossed → "> 30"; WIDTH-3 then reads "undefined (censored)". STOP-LOC not found within K′ 34 → "> 34".
- eval_deep30 missing, or items_sha256 ≠ 3938fdf3… → every n ≥ 25 reading fails (N50-3 then uses n 21–24 only and is flagged).
- A never-admitted cell → UNREAD (never a pass); MECH-CARRIES then reads UNREAD.
- A PARAM-SWAP arm failing SWAP-VALID → that G reads "not localised".

**Launch rules:**
- Stage A launches only if §7.1 passes and §7.2 selects a G.
- Stage B launches only if A reads MECH-CARRIES, or MECH-PARTIAL with s1's SEED-PASS and no FREEZE-BLOCKS-DEPTH, **and** the user OKs it.
- MECH-NOT-CARRIED or FREEZE-BLOCKS-DEPTH → no Stage B with the lever, and **no automatic fallback lever**: Theory re-plans (avoids the problem → patch loop).
- No module edit while any R3 deploy is in flight.

**Cost** (RTX 3090 ≈ $0.22/h; transfer and storage ≈ $0.1–0.2 per cell [218-19]):

| item | basis | projected | admission / guard |
|---|---|---|---|
| A-s1, A-s2 | round-1 shape: FT-SELF 7,260 s; realised $0.56 (excl. its bug host) and R1-s2 $0.54 | $0.45–0.70 each | $0.75 / $0.95 |
| R3-FRZ, R3-PLAIN | R2 realised $1.02 × self-step factor 1.48–1.73 (D7): 6k × 2.6–3.9 s + 6k × 0.41 s = 5.0–7.2 h | $1.40–2.00 each | $2.10 / $2.50 |
| PARAM-SWAP, pool rebuild (s1 ≈ 1 h), R3 pool (3–5 h), gates, evals | local 3070 / 4060 | $0 | — |
| **Family** | | **$3.7–5.4** | **cap $7.0** (guards sum $6.90) |

- Realised at draft: **$0**. Steps are never reduced. Admission tries as §215 (215-2).
- Checkpoints: full state every 2k with off-host mirrors and a tested resume (the 10-15 rule). A mirror dir that already exists is moved aside first (ckpt-mirror rule).

---

## 7. $0 checks before v1

### 7.1 Known-answer checks (all must pass; Audit reads the JSON)
- **K1 SPLICE-IDENTITY:**
  - Tool `splice(body, donor, G)` = load body's state_dict, then overwrite G's keys from donor.
  - For each G ∈ {slva_addr, prelude0-5, coda10-11, core6-9}, splice(D3-T s1, D3-T s1, G) → eval_hist24 equals the committed D3-T s1 file: bitwise on the 3070 (fp32, bs 32), or within ±0.005 per n on another card (KL-DRIFT's gate practice).
- **K2 SPLICE-FULL:**
  - The groups must be a disjoint cover of every state_dict key except c_anneal (asserted).
  - splice(D3-T s1, FT-SELF, all groups) reproduces FT-SELF's committed STAB_self **0.489** and intact rows.
- **K3 FREEZE-SMOKE** (CPU, OMP 2, 20 steps, the edited trainer):
  - with `--freeze_groups slva`: G's sha is unchanged at steps 0 / 10 / 20; every other trainable group has a nonzero grad (scratch/gradflow_check.py); the AdamW param count = total − |G|;
  - **without the flag:** loss trajectory bitwise equal to the pre-edit trainer (plain-path smoke);
  - a G not present in the mode asserts.
- **K4 PAIRING** (CPU, 30 steps each, tiny micro): FRZ-s1 and FRZ-s2 draws_log plain_hash / coin / U = the twins' committed draws_log.json at steps 1..30.
- **K5 SCORER FIXTURES:** synthetic hist24 / eval_vb files with known STAB and acc give the right SEED-PASS / OVERSHOOT / FREEZE-BLOCKS-DEPTH / MECH-* labels, and pending on a missing file. Also an n50 "> 30" case and a deep30 sha-mismatch case.

### 7.2 PARAM-SWAP localisation (informative; picks G; descriptive otherwise)
- **Arms (s1):** body FT-SELF with donor FT-CTRL, for G ∈ {slva_addr, prelude0-5, coda10-11, core6-9}. FT-CTRL is the donor because it shares base, plain stream and seed.
  - Reverse arms (body FT-CTRL, donor FT-SELF): IMPLANT, descriptive.
  - s2 replication: body R1-s2, donor D3-T s2, descriptive.
- **SWAP-VALID:** the spliced model has intact acc_n2 ≥ 0.99 and acc_k24 ≥ 0.97 at n 3–8. Otherwise the arm reads UNREADABLE.
- **RECOVERY(G)** = (STAB_self(FT-SELF ⊕ CTRL_G) − 0.489) / (0.696 − 0.489).
- **Selection rule (pre-stated):** G = the non-core group with the largest RECOVERY ≥ 0.50 among the SWAP-VALID arms; a tie within 0.05 goes to the smaller parameter count. If only core6-9 reaches 0.50, or nothing does → **no Stage A**; Theory re-plans and records the result as an H_DIFF instance.
- **Cost:** 8–10 eval_hist24 runs, ≈ 1–2 h on the 4060; $0. It needs the splice tool's G-D review and K1–K2 first.

---

## 8. Open questions for the red team
- **Q1** Is a round-1 redo the right place to test a round-3 lever? It gives power (O1, O2), but the R3 lever then acts on R2-s1, whose G already moved in two rounds.
- **Q2** For G = prelude (≈ 44 % of the self contribution's displacement), D4's specificity argument fails. Add a size-matched random-freeze arm (that would be the one ablation arm in A, and R3-PLAIN would then drop)? Or accept the confound in the wording?
- **Q3** s2's effect size rests on A3 (CTRL-s2 not run). Should s2 use its own bar (e.g. 0.06), or a ΔSTAB scaled by its base gap (0.684 − 0.489)? Should A2 be measured by a **REPLAY** cell (FT-SELF's recipe re-run, ≈ $0.5) instead of assumed?
- **Q4** Should Stage B carry the 216-9 n 1–3 K′24 rollout mix as a guard against O3? It is a second change; this v0 says no.
- **Q5** D0: freezing fixes the addressing function only. Should Stage A add a descriptive SLVA mass-at-idle-passes readout (part_stabhist's gather_split_log at K′24, n 3–6) to show whether the addressing *behaviour* still moved?
- **Q6** Is DBS-3 worth a conjunct, given that it passes under H_shift (D6)? Alternatives: drop it, or replace it with N50-3 ≥ 24.5.
- **Q7** MEM-HELD-3's −0.03 tolerance against R2's 0.437: is it too loose, given that R2's own drop was 0.052?
- **Q8** The R3 self_kmax 24 equals eval's K′ 24 anchor. K′24-STABLE at n 1–8 is still 16–23 idle passes past training. Should Stage B also read a_32?
- **Q9** HOLD-SHIFT (§2.1) is post hoc from the hist24 table. Does any registered reading depend on it? (Intended: no; it only motivates.)
- **Q10** Card: the 3070 has been lost twice (§215 / §216). Pre-state the Addendum-1 fallback for every R3 eval, including deep30's memory on the 4060.

## 9. Novelty sentences (for Literature; none is claimed in this draft)
- N1 "Freezing a recurrent latent model's slot-local history-attention parameters during self-training preserves its long-horizon memory mechanism" (grep THEORY.md: no slva freeze before; check Zheng 2501.13453, 2602.23197, partial-freeze continual-learning work).
- N2 D1's "fixed-point insensitivity": equal values make attention weights unidentifiable from outputs. Probably known (attention identifiability, e.g. Brunner et al. 1908.04211); check before citing D1 as ours.
- N3 PARAM-SWAP (group-wise weight reversion to localise a fine-tuning effect): likely known (skill localisation / weight-reversion work, e.g. Panigrahi 2302.06600); check.
- N4 "HOLD-SHIFT: self-training moves idle-pass holding from history-anchored to state-anchored": an observation on our models. Check against Literature's 838991c2 verdict (NARROW) before any "first".
- Grepped for this draft: the THEORY.md section list; Notes 215-x / 216-x / 217-x / 219-8; vbank_diag part names (part_hist24, part_stabhist, part_hold, part_ageproj …); the eval_*.json names in s213_d3t_s1 / s215_ftself_s1. **No "unmeasured" claim is made.**

## 10. $0 checks run for this draft
| script | output | finding |
|---|---|---|
| r3_logsplit.py | r3_logsplit.json | per third of training, by coin (draws_log): FT-SELF self gn_preclip 2.40 / 0.089 / 0.0147 vs plain 0.0128 / 0.0017 / 0.0007. SLVA's share of the grad norm is 0.061–0.065 (self) and 0.041–0.062 (plain). Plain off-newest copy mass is 0.267 / 0.268 / 0.272 (CTRL 0.275 / 0.277 / 0.281). R2-s1 and R1-s2 have the same pattern |
| r3_param_drift.py | r3_param_drift.json | per-group relative displacement ≈ 1.1e-2 in every matrix group for both CTRL and SELF. SELF − CTRL shares follow the gradient-receiving parameter shares. cos(SELF − D3T, CTRL − D3T) = 0.13 (slva), 0.13–0.16 (core / coda / prelude), 0.98 (bank biases). Total 125,641,988 parameters; slva 1,184,280 (0.94 %) |
| r3_oc_sim.py | r3_oc_sim.json | §5 tables |
