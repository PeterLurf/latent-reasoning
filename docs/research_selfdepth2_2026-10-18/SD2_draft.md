# SELF-DEPTH-CHAIN round 2 (SD2, proposed §216): DRAFT pre-registration

Clock 2026-10-18 00:09 EDT (`date`). Opus design subagent for Theory. **$0: no training, no GPU, no repo edits** (CPU checks only, ≤ 2 threads, nice 19; no bytecode written into D:\Latent, checked with `find -newer`).
**Status: DRAFT, not registered.** It needs Theory's review, red team / proofs as Theory chooses, Audit, and **the user's explicit OK regardless of cap**. Round 1 read MEM-TRADED (STAB_self 0.489 < 0.5), and the frozen round-1 text (SELFDEPTH_v1 §9) says "MEM-TRADED → only with the user's explicit OK". So the ≤ $30 Audit auto-approval does not apply.
Sources: SELFDEPTH_v1.md (frozen §215 text, 5ded8703), PROOFS_SELFDEPTH.md, REDTEAM_SELFDEPTH.md; THEORY §215, notes 215-1…215-5, the §215 FT-SELF result block, notes 214-4 / 214-5; theory/score_s215.py and theory/results/score_s215.json; boxes/{vbank, selfdepth, vbank_eval, vbank_diag, nested170, slva}.py; scratch/s215_build_pool.py; research5/LIT_iterated_selftrain.md.
Raw files read: boxes/runs_fsq/{s213_d3t_s1, s215_ftself_s1, s215_ftctrl_s1}/{eval_vb, log.jsonl, args.json}; s215_ftself_s1 and s213_d3t_s1 {eval_chainlabel(_fresh), eval_hist24, eval_selfdepth}; *_mem/mem_check.json, *_gtime/timing.json, auto_micro(_self).json; s215_pool/pool_meta.json.
**Not read, on purpose:** FT-CTRL's eval_hist24 / eval_d3hist (on disk since 22:43–22:51). §2.5 below is written blind to the pair's MEM attribution.
$0 checks (research5/scripts/; JSON outputs beside each script): `testset_ext.py`, `tsplit.py`, `evalmem_cpu.py`, `sd2_numbers.py`, `sd2_tables.py`, `sd2_oc.py`, `support_check_r1copy.py` (PROOFS' support_check, unchanged; it includes the round-2 law).
Tags: [PROVED], [PROVED UNDER X], [DERIVED] (exact arithmetic on measured inputs; script named), [MEASURED] (raw file named), [ASSUMED], [CONJECTURE].

---

## 0. One screen

| item | SD2 (draft) |
|---|---|
| Question | Round 1 absorbed n 9–12 and grew the margin past support: M 0.717, n50 15.46, margin/B 0.288, which equals the base's 0.288. If the model trains on its **own** round-1 chain at n 13–16, does it (a) absorb n 13–16, and (b) grow the margin again, **proportionally** (H_prop: margin/B constant, n50 ≈ 20.6) rather than by a constant (H_shift: n50 ≈ 19.5)? Is n 17–20 then ≥ 0.80 (C2 proxy)? |
| What SD2 tests | **DBS-2 is a direct test of H_prop.** Under H_prop, round 2 is predicted to clear the C2 bar: M2 = mean acc n 17–20 = **0.852** (H_shift 0.717; bar 0.80; SE ≈ 0.006) [DERIVED, sd2_tables] |
| Cells (recommended) | **s216_r2_s1**: FT from `s215_ftself_s1/model.pt`. **Second lineage (option b):** `s216_d3t_s2` (D3-T recipe, seed 2, 48k) → `s216_r1_s2` (round 1 as §215 FT-SELF) → `s216_r2_s2`. No R2-CTRL by default. RTX 3090 only |
| Labeller | FT-SELF's re-rooted single-step chain at **both** windows (one labeller). Filter KEEP ∧ COMPOSE-VALID. GATE-2 (n 13–16) and GATE-1′ (n 9–12, relabelled), each 2,000 per n, with GATE-1's thresholds. Pre-stated fallback per window: D3-T (BASE-LABELLED) |
| Self law | p_self 0.5. Self items are **50 % n 13–16 / 50 % n 9–12**, all fresh, uniform within each window, in one interleaved pool order. K = min(n + U{0..4}, 20). Gold n ≤ 8 is 50 % of steps, as in round 1 |
| Recipe | Round-1 FT-SELF args, with only init_from, self_pool, **self_kmax 20** and self_micro (G-S9: projected **4**) changed. 12k steps, lr 3e-5, seed 1 |
| Extrapolation set | `test_set(1000, n_max 24)` n 17–24. The n 1–20 items are bitwise test_set(1000, 20) [MEASURED, testset_ext.json]. **n 17–20** is DBS-2's window (sealed in round 1); n 21–24 is new (M_far, FULL, n50, STOP-LOC) |
| New eval part | `eval_deep.json`: n 13–24, K′ 0..28, per-item answers at every K′, local 3070, fp32, **eval_bs 16** (bs 32 projects 8.0 GiB at n 24 / K′ 28; bs 16 projects 4.25 GiB) [DERIVED] |
| Headline OC (one lineage) | **P(SD2-C2) ≈ 0.11** (0.085–0.14 over the H_grow prior); P(SD2-SELF) ≈ 0.40; P(SD2-C2 \| H_shift) ≈ 0.02; P(SD2-C2 \| H_grow) ≈ 0.24; P(SD2-C2 \| H_full) ≈ 0.38. **2/2 lineages ≈ 0.02–0.04** |
| Cost | Projected **$0.62–1.27 per R2 cell**. Family (s1 R2 + s2 lineage) projected **$2.9–5.0**; **family cap $6.5** (guards per cell). Round 1 realised ≈ $1.07 for the pair [RELAYED by Theory]; FT-SELF ≈ $0.68 incl. the $0.118 bug host [§215 result header] |
| Approval | **The user's explicit OK** (MEM-TRADED rule, v1 §9), plus Audit's gates. Not auto-approved, although the cap is ≤ $30 |

---

## 1. Change table vs round-1 §9 (the pre-stated round-2 plan)

| # | v1 §9 pre-statement | SD2 | why |
|---|---|---|---|
| 1 | W_2 = n 13–16, B_2 = 16 | **kept** | Round 1's +4 worked. Raw loop accuracy at n 13–16 (0.999/0.893/0.633/0.344) mirrors the base's at n 9–12 (0.923/0.603/0.266/0.127), from a better start [MEASURED eval_vb]. Labeller keep at n 13–16 is ≥ 0.957 (§2.1). A wider window would put n 17–20 in support and spend the extrapolation set. A narrower one gives up the H_prop test (§3.4) |
| 2 | Labeller = FT-SELF's chain; GATE-2 = same rule, 2,000 held-out per n 13–16 | **kept**, plus **GATE-1′** (2,000 per n 9–12 under the same labeller) | The old window is now relabelled by FT-SELF (row 4), so it needs its own gate |
| 3 | Fallback BASE-LABELLED (D3-T) | **kept, per window** | GATE-2 fails → D3-T labels n 13–16. GATE-1′ fails → D3-T labels n 9–12 (GATE-1 already passed for D3-T). Both labellers fail GATE-2 → no round 2 |
| 4 | Self mixture: newest 50 % / round-1 window 50 %, all fresh | **kept.** Round-1 window **re-drawn fresh** and labelled by FT-SELF | The round-1 pool has only 56,512 unused records (cursor 383,488 / 440,000 [MEASURED eval_selfdepth cursor]), against the ~192k needed. Those records carry D3-T labels; one labeller is simpler. Fresh draws keep MEMO-GAP-2 clean |
| 5 | K = min(n + U{0..4}, 20) | **kept** (self_kmax 20, self_kextra 4) | — |
| 6 | Extrapolation = the sealed n 17–20 extension | **extended to n 17–24**; DBS-2 stays on n 17–20 | H_shift and H_prop differ in where the 50 % crossing falls (19.5 vs 20.6), and FULL needs n > 20 (Literature note §Implications). Adding n 21–24 to test_set is bitwise-safe for n ≤ 20 [PROVED: one rng, consumed stratum by stratum, nested170.py l.148–154; MEASURED: testset_ext.json eq_n1_20 true] |
| 7 | "New eval contract; memory at n 20, K′ 24 on the 8 GB card to check" | `eval_deep.json` (§4.2): n 13–24, K′ 0..28, **eval_bs 16** | At bs 32, n 24 / K′ 28 projects 8.03 GiB, over the 3070. At bs 16 it projects 4.25 GiB [DERIVED, evalmem_cpu slopes; the slope reproduces the 3090 mem_check within 3 %]. The registered n ≤ 16 readings stay on eval_vb at bs 32 (unchanged) |
| 8 | Memory K 20, T 61 → 48.7 GiB forced → micro 4 | **re-derived: micro 4, 13.8 GiB** at the deepest shape | The 64th-longest of 57k n-16 draws has T 59 (max 61) [DERIVED]. The model was validated on round 1: 14.47 GiB projected vs **13.91 measured** at micro 2 (s215_ftself_s1_mem) |
| 9 | Cost $1.13–2.78 | **$0.62–1.27 per R2 cell** (forced projection) | Re-based on round 1's measured G-TIME (self 0.896 s, plain 0.439 s) and its real/forced ratio 0.897. The range is two bounding time models, compute-bound and launch-bound (§3.6) |
| 10 | Runs only if ABSORB ∧ ¬FT-DIVERGED ∧ LABEL-AUDIT ∧ ¬CONTAMINATED | **met** [MEASURED score_s215.json SELF] | — |
| 11 | GROW / FULL → round 2 plus a second seed | **GROW read**, so round 2 plus a second seed. **Second seed = new lineage from a D3-T seed-2 base (option b)**; option (a) is priced as the alternative (§2.4) | "In every competent seed" is about training seeds. Under (a), round-1 labels are identical to s1's (same D3-T s1 labeller) |
| 12 | MEM-TRADED → only with the user's explicit OK | **applies** | Round 1 read MEM-TRADED (0.489, by 0.011) |
| 13 | OC shift: GROW → P(H_grow \| round 2) × 1.5 (cap 0.8) | **applied**: 0.25 → 0.375; the rest renormalised in round-1 proportions | §6 |
| 14 | S1-C (Lee-faithful +1 rounds) stays unfunded | unchanged | — |
| N1 | — | **H_prop** is a named hypothesis | The base → round-1 margins fit margin/B = 0.288 / 0.288 at the 50 % level [DERIVED sd2_numbers crossings]. Huang 2511.07378 Thm 5.1 and Yang 2610.14456 Thm 2 predict a margin proportional to the trained length (Literature note) |
| N2 | — | Thresholds re-anchored on round 1's own curve: MARGIN-GROWS-2 0.785 (the H_shift/H_prop midpoint), SHRINK < 0.60, FULL needs M_far ≥ 0.60 | Round 1's 0.63 / 0.40 bars were anchored on the base's curve |
| N3 | — | RETAIN (n ≤ 12 kept), LABEL-AUDIT-2 per window, MEMO-GAP-2 at n 9–16, NOT-SIZE-KEYED-2 (the H_T nuisance), BS-CHECK, MEM history and a MEM sub-label | §4 |
| N4 | — | G-S16 EVMEM, G-S17 SMOKE-AT-SCALE | Memory rules: "smoke at full data scale" (the b310 O(N²) hang) and "check memory for the deepest shape" |
| N5 | — | Baseline reads of eval_deep on D3-T s1, FT-SELF and FT-CTRL, **after** SD2's registration commit | They give the exact H_shift panel at n 21–24 and FT-SELF's own n 17–24 level. No threshold depends on them |

---

## 2. Design

### 2.1 Labeller and gates

- **Chain, KEEP, CV, kept, R2 targets:** exactly as v1 §2.1 (`boxes/selfdepth.label_chains`, `compose_E`). The only change is the labeller: **`s215_ftself_s1/model.pt`**, sha256 recorded in pool_meta.json. It is local 3070, fp32, bs 32, cap n + 4.
- **Why FT-SELF** (pre-stated primary). On 1,000 fresh items per n [MEASURED eval_chainlabel_fresh.json]:

  | n | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | FT-SELF keep | 1.000 | 0.999 | 0.999 | 0.999 | **0.997** | **0.989** | **0.986** | **0.957** | 0.882 | 0.751 | 0.518 | 0.361 |
  | D3-T keep | 1.000 | 0.996 | 0.988 | 0.976 | 0.972 | 0.952 | 0.943 | 0.917 | 0.892 | 0.839 | 0.809 | 0.781 |
  | FT-SELF prec_kept | 1 | 1 | 1 | 1 | 1 | 1 | 0.998 | 1 | 1 | 1 | 1 | 1 |

  - FT **raised** the single step's reach at n ≤ 16 and **narrowed** it at n ≥ 18. The narrowing is coverage (FAIL), not precision: 0 kept-wrong among 2,512 kept chains at n 17–20.
  - This matters for a round 3, not for SD2.
- **GATE-2** (n 13–16) and **GATE-1′** (n 9–12). 2,000 held-out fresh items per n:
  - rng default_rng(22660) for n 13–16 and default_rng(22661) for n 9–12; §2.2 exclusion; never trained.
  - kept = KEEP ∧ CV; scored against gold in a separate process.
  - Rule, per window (= GATE-1's): **π_pool ≥ 0.995 ∧ π_n ≥ 0.99 at every n ∧ π^E_pool ≥ 0.99 ∧ keep_n ≥ 0.70 at every n**.
  - Same thresholds because the exact OC at FT-SELF's kept counts (1,994 / 1,978 / 1,972 / 1,914; c_n 19, c_pool 39) is **P(pass \| π 0.997) 0.9987, \| 0.995 0.52, \| 0.993 0.014** [DERIVED, sd2_numbers gate2_oc]. That is the same operating characteristic as GATE-1 (0.9986 / 0.504 / 0.0125).
  - The measured window π is 0.99949 (2 kept-wrong / 3,929), so P(pass) ≈ 1.
  - Descriptive, at GATE-2: **KEPT-VS-DRAWN**, the height / T / depth distributions of kept vs all drawn at n 16 (Literature failure mode 1).
  - Correction to the Literature note: its "n 16 keeps about a third" uses the loop's accuracy. The yield is the chain's keep, 0.957.

### 2.2 Contamination rules (registration text)

1. **Training items are fresh generated items only.**
   - s1 pool: `nested170.generate(rng, n)`, rng = default_rng(22600), n 9–16, until **56,000 kept per n** (448,000).
   - One fixed interleaved order: default_rng(22601) permutation.
   - Fresh-once: exhaustion or reuse = void.
2. **Exclusion by exact rendered syms** against:
   - test_set(1000, **n_max 24**) (24,000 items);
   - SERIAL (rng 17300, 2,200);
   - the 213-37 fresh items (rng 21337, n 9–20, and n 21–24 if eval_chainlabel_fresh is extended, §2.6);
   - GATE-1 (s215_gate1/gate_items.json), GATE-2 and GATE-1′;
   - **all 440,000 round-1 pool records** (cheap; it keeps fresh-once across the lineage).

   Per n, draws, collisions and duplicates are logged. Expected 0 at n ≥ 9 [PROVED, PROOFS §8: P(render = s) ≤ (1/11)·40⁻ⁿ].
3. **No gold reaches training.** As v1 §2.2 item 3:
   - builder gold patches before generation;
   - record keys exactly {syms, n, chain_label, steps}, asserted at load;
   - the SD-LABEL-MAP grep;
   - no `s216_gate` string in boxes/vbank.py.
4. **Gold is used only for evaluation, GATE-2 / GATE-1′, LABEL-AUDIT-2, MEMO-GAP-2 and eval_deep**, each in a separate process.
5. **Round 1's extrapolation items (test n 13–16) are round 2's in-support test items.** They are excluded from every pool. Their round-1 readings (FT-SELF a(13..16)) were read before this draft and inform only the predictions.
6. **n 17–24 of test_set(1000, n_max 24) is SD2's extrapolation set.**
   - Before SD2's registration commit it is used only as an exclusion list.
   - After the commit it is read on D3-T s1 / FT-SELF / FT-CTRL as descriptive baselines (row N5).
   - No threshold uses those baselines.
7. **The labeller is fixed in advance** by the round-1 §9 rule, never chosen on pool or test items. Pool, GATE-2 and GATE-1′ use the same card class and settings (3070, fp32, bs 32).
8. **Post hoc:** pool ∩ (all of item 2) = ∅ is re-verified on the pool file. Any hit, gold key, reuse (`self_reuse` > 0) or exhaustion → **CONTAMINATED / void**.
9. **The s2 lineage uses its own seeds.** Round-1 pool 22700 / 22701, GATE-1-s2 22760, audit 22770. Round-2 pool 22800 / 22801, GATE-2/1′-s2 22860 / 22861, audit 22870.
   - Its exclusion list is the same plus its own pools and gates, with test n_max 24 from the start.
   - All 226xx–228xx seeds are unused in THEORY.md, boxes, theory and scratch [checked by grep].

### 2.3 Cells and training (s1 lineage)

| element | spec |
|---|---|
| Init | `--init_from boxes/runs_fsq/s215_ftself_s1/model.pt`: strict load, missing = [] (asserted, vbank.py l.666), calibrate() not called, `cal` = FT-SELF's calibration.json (= D3-T's), fresh AdamW |
| Recipe | **s215_ftself_s1/args.json** with only `--init_from`, `--self_pool boxes/runs_fsq/s216_pool_s1/pool_r2.jsonl`, `--self_kmax 20`, `--self_micro <G-S9>` and `--out` changed. Unchanged: lr 3e-5, 12k steps, warm-up 100 + cosine, cur 8, c_sched 0,1, p_self 0.5, self_kextra 4, kmax 12, kextra 4, bs 64, **seed 1**, draws_log_steps 20000, ckpt_every 2000, mem_budget 21.0 |
| Seed 1 | Seed 1 replays round 1's plain stream and coin sequence: the same plain batches, and self steps at the same step indices with new pool records. **G-S3 then checks plain_hash at steps 1..200 against s215_ftself_s1/draws_log.json** (identity). No reading depends on new plain items |
| CurGate | Moot. `--cur 8` is a single stage, CurGate.update returns None at the final stage (vbank.py l.557), and self steps are excluded anyway (l.812) [VERIFIED code]. plain_answer_ce is logged (l.827). So the 214-5 lesson is met by construction |
| Self law | The pool order is uniform over n 9–16 (1/8 each = 50/50 windows). Plain steps follow D3-T's law. P(R pass \| t ≥ t*) = 0.455 / 0.375 / 0.273 / 0.143 at t* 13–16, **0 at t ≥ 17** [DERIVED, support_check_r2.json "R2 (self 9-16, kmax 20)"] |
| Exposure | 6,000 ± 55 self steps → 384,000 ± 3,505 items; 48k per n. Round 1 gave 96k per n, but its self_answer_ce was 0.06 at step 1k and **0.0095 at step 3k (≈ 24k items per n)** [MEASURED s215_ftself_s1/log.jsonl]. So 48k per n is 2× round 1's absorption point. Pool 448,000: z(exhaust) = 18.3 [DERIVED] |
| Micro | plain 1; self from G-S9 (projected 4; §3.5) |
| Logging | Round 1's fields. Optional descriptive addition: self_answer_ce split by window (§2.6 E3) |
| Checkpoints | Full state every 2k, mirrored off-host. The resume is tested before launch (G-S4 / G-S11). Checkpoints are deleted at landing; model.pt is kept (round-3 labeller) |

### 2.4 The second seed: options weighed (recommendation: b)

| | (a) FT replicate from D3-T s1 | **(b) new lineage from D3-T seed 2 (recommended)** |
|---|---|---|
| Cells | s216_r1_s2 (FT-SELF recipe from D3-T **s1**, seed 2, the round-1 pool re-permuted, rng 22701) → its pool → s216_r2_s2 | **s216_d3t_s2** (D3-T args, seed 2 only, 48k, 3090) → GATE-1-s2 + pool (D3-T s2's chain) → s216_r1_s2 → GATE-2-s2 + pool (r1-s2's chain) → s216_r2_s2 |
| Tests | FT optimisation noise at a fixed base. **Round-1 labels are identical to s1's** (same labeller, same records), so round 1 is not an independent replicate. Lee App B.2.2 ("each seed generates its own data") is met only from round 2 | **The whole pipeline:** the base's basin lottery (Kuo 2606.29983: looped-model seed SD 0.09–0.12), its own labels at both rounds, its own margin. This is the unit "in every competent seed" refers to. D3-T s2 is also a C6 data point (does D3-T's STOP-AT-10 / VB-C1 replicate?) |
| Cost (projected) | R1 $0.45–0.68 + R2 $0.62–1.27 = **$1.1–2.0** | D3-T s2 $1.12–1.70 + R1 $0.45–0.68 + R2 $0.62–1.27 = **$2.2–3.6** |
| Wall-clock | ≈ 16 h (R1 2 h, R1 evals 4 h, pool 1.3 h, R2 4 h, evals 5 h) | ≈ 24 h. D3-T s2 needs **no new code** and can launch right after registration, in parallel with s1's engineering and labelling |
| Risk | Little information if the FT noise is small | D3-T s2 may not be competent (P ≈ 0.8 [ASSUMED]). If so, the lineage stops at ≈ $1.3, is reported as a C6 data point, and is **not** counted against SD2-C2 ("competent" is pre-stated, below) |

- **COMPETENT-BASE (pre-stated, s2):**
  - not dead (48k);
  - basin entry by score_s211.entry_on (the §187 rule: the entry step is reported, and a gap > 5k vs s1's 11k is CONFOUNDED, descriptive);
  - acc_n2 ≥ 0.99 and a_24 ≥ 0.97 at every n 1–8;
  - GATE-1-s2 passes.

  VB-C1 is descriptive for the base.
- **Gating inside s2:** s216_r2_s2 runs only if s216_r1_s2 reads ABSORB ∧ ¬FT-DIVERGED ∧ LABEL-AUDIT ∧ ¬CONTAMINATED (v1 §9's rule), independent of s1's round-2 reading.
- **R1-s2 scoring:** score_s215's registered cell() unchanged, with fixed thresholds. Its own-base H_shift panel is descriptive.
- **R2-CTRL** (FT from FT-SELF with only round-1-window self items): **not recommended.**
  - Its attribution row (SELF-ATTRIBUTED-2) follows from ABSORB-2 under every hypothesis, the same non-discrimination as round 1's STOP-MOVED (RT §4).
  - Its MEM value is a dose question.
  - It is offered as an optional add-on (+$0.70 cap) only in branch M-S (§2.5).

### 2.5 Branches on the pending round-1 pair (FT-CTRL attribution not yet scored)

Pre-stated: **SD2's recipe, readings and thresholds do not change across branches**. Only wording, the R2-CTRL offer, and the KEEPS-SELF-STAB-2 prior move.

| branch (score_s215 PAIR) | SD2 consequence |
|---|---|
| **P1 SELF-ATTRIBUTED** (CTRL ¬STOP-MOVED) | As designed. CTRL's acc_n2 at n 9–16 is at base level (0.938/0.582/0.258/0.157/…) [MEASURED], so P1 is expected, but not assumed |
| **P2 FT-EFFECT** (CTRL STOP-MOVED) | **SD2 is not launched.** Plain FT alone moved the stop, so the self items were not what moved depth. Theory re-plans |
| **M-S** "traded by the self items" (CTRL keeps STAB_self ≥ 0.5) | Deeper self items may trade more. The user's OK names this. **R2-CTRL offered** as a MEM-dose control (+$0.70). KEEPS-SELF-STAB-2 prior 0.20 |
| **M-F** "traded by the FT (lr re-heat)" (CTRL also < 0.5) | A second FT likely re-heats again. R2-CTRL not useful (CTRL already shows FT alone does it). Prior 0.25 |
| **C-X** CTRL censored / evals never land | MEM attribution UNREAD. SD2 proceeds with the user's OK, worded "MEM attribution unread". Prior 0.25 |

### 2.6 Edits needed (list only; none written; Audit's diff review G-S13)

| # | where | edit |
|---|---|---|
| T1 | vbank.py | **No change required** for SD2-s1 [VERIFIED by reading]. draw_step takes any pool n, K = min(n + U, self_kmax). G-S9's deep shape uses n_top = max pool n (= 16). The init_from asserts hold. Optional safety: let the auto_micro self loop try {1, 2, 4, **8**} (it raises after 4, l.718–719); projected need is 4. Optional: help-string "n 12" → "n_top" |
| T2 | vbank.py (optional, descriptive) | log.jsonl `self_answer_ce_w1` / `_w2` (self items n ≤ 12 / ≥ 13). This needs a per-item answer CE split in vb_loss's stats. If not built, the window split is UNREAD |
| B1 | scratch/s216_build_pool.py (new; from s215_build_pool.py) | labeller path; NS 9..16; PER_N 56,000; seeds 22600/22601/22660/22661; §2.2 exclusion incl. test n_max 24 and the round-1 pool syms; GATE-2 + GATE-1′ items; pool_meta adds labeller_sha256, excl breakdown and window tags. A `--lineage s2` mode covers the s2 round-1 pool (D3-T s2 labeller, n 9–12, 110k per n) and round-2 pool |
| B2 | scratch/s216_gate2.py | GATE-1's scorer generalised to a window list; one eval_gate.json with by_n 9..16 plus a `window` field; stores per-item kept / label / gold / state-ok |
| E1 | vbank_diag.py `part_deep` (or a new boxes/vbank_deep.py) | §4.2 contract |
| E2 | vbank_diag.py `part_selfdepth` | Parameterised by round: pool / gate paths, test n_max 24, memo n 9..16, label_audit by window, contamination vs test n_max 24. **items_n2 and items_k24 per item at n 1..16** (store-items rule). Today the paths are hard-coded to s215 (l.1377, 1390, 1407, 1410) |
| E3 | vbank_diag.py `part_chainlabel_fresh` | Extend n 9..20 → 9..24, with the same rng 21337 consumed sequentially, so n 9..20 stays bitwise (same argument as test_set). Descriptive LABELLER-KEPT-2 |
| S1 | theory/score_s216.py (new) | §4.6 |
| Q1 | scratch/s216_*.sh | queue / fill / post-landing chains. Drain waiters armed at launch; eval chains pinned to the 3070 |

---

## 3. Mathematics

### 3.1 S0, support identity for round 2 [PROVED; v1 §3 / PROOFS §1 with B_2 = 16]

- Every target state of the round-2 mixture has ρ + r = n ≤ 16. The argument is the same: CV ∧ STEPS gives one VOIDed handle per step for n steps, then HOLD; values never enter the count.
- (a) "Stop at ρ ≥ 16" and any pass deadline t_d ≥ 17 are zero-loss on the support. **H_shift is still irreducible by loss**, one boundary further out.
- (b) ABSORB-2 (n 13–16) is in support. n 17–24 is out of support. n 19–24 are read at K′ n + 2 = 21–26, above the trained max K of 20 [PROVED: vb_loss windows, PROOFS §2].
- (c) Positions: round 2 trains rows up to T ≈ 61 (n 16). eval_deep reads up to T 83 at n 24 (≤ 1,024 wpe rows) [MEASURED testset_ext.json].
  - Rows 62–83 are pretrained GPT-2 rows that the loop never trained (they are only weight-decayed).
  - Round-1 precedent: FT-SELF read rows up to 59 at n 15–16, ≥ 10 untrained rows, with no within-n T effect. n 16: T ≤ 47 0.319 vs T ≥ 49 0.352; n 15: 0.658 vs 0.592 [DERIVED tsplit.json].
  - The decline is n-keyed, not T-keyed. Whether that holds at 20+ untrained rows is the nuisance hypothesis **H_T** [CONJECTURE]. NOT-SIZE-KEYED-2 reads it (§4.4).
- (d) The rollout has no pass cap. The bank is a Python list, and SLVA's age bias is γ·age (continuous) with β0 on V_0. No table is indexed by pass [VERIFIED vbank.py l.262–345, slva.py l.69–73]. K′ 28 is admissible.

### 3.2 S1 for round 2 [DERIVED, support_check_r2.json; PROOFS' exact-rational script, unchanged]

| t* | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | ≥ 17 |
|---|---|---|---|---|---|---|---|---|---|
| P(R pass \| t ≥ t*), trainer law | 0.637 | 0.605 | 0.566 | 0.517 | 0.455 | 0.375 | 0.273 | 0.143 | **0** |
| deadline cost at t_d (share of an item's R2 term, loss-weighted) | 14.8 % | 11.2 % | 8.2 % | 5.7 % | 3.7 % | 2.2 % | 1.1 % | 0.35 % | **0** |

- G-S3's target is P(R \| t ≥ 13) = 0.455 ± 0.01 over 20k logged draws, and 0 at t ≥ 17.
- A stop or deadline at t_d ≤ 16 now costs loss. The smallest costs are at t_d 15–16 (1.1 % / 0.35 %). A weak gradient against stopping exactly at the boundary is the H_shift mechanism.

### 3.3 S2, label noise in the first compounding round

Round 2 is the first round whose labeller was itself trained on self-labels (w_1 = 2 / 5,000 kept-wrong in LABEL-AUDIT, π_state 0.9996 [MEASURED score_s215.json]).

- **(a) Direct counts** [DERIVED from MEASURED eval_chainlabel_fresh]:
  - FT-SELF chain at W_2 = n 13–16: **2 kept-wrong among 3,929 kept**, π = 0.99949, Clopper–Pearson 95 % lower **0.9984**. Both errors are at n 15.
  - D3-T (fallback): 3 / 3,784, π 0.99921, lower 0.9980.
  - State level: π^E ≤ π [PROVED, PROOFS §6]. GATE-2 measures π^E directly. RT measured π^E = π on D3-T at n 13–16 (1,892 / 1,892); it is unmeasured on FT-SELF.
- **(b) κ thresholds at n 16** [PROVED UNDER A1′, A2, A3′; PROOFS §3b forms]:
  - Avalanche (ē at least doubling per round) needs κ ≥ P/(φn).
  - FT-SELF: P(16) 0.957 strict; φ̂ = 2/73 (kept-wrong among non-strict chains, pooled n 13–16) = 0.027 → **κ ≥ 2.2**; at φ's 95 % upper (0.084), **κ ≥ 0.71**.
  - D3-T: 4.2 / 1.6. Round 1 (D3-T at n 12): 8.1.
  - The FT-SELF margin is thinner because a better labeller fails less, and its few failures are relatively more often kept-wrong. With 2 events, φ is barely determined.
  - **So the κ bound is not the guard; the direct measurements are** (GATE-2 at 2,000 per n, LABEL-AUDIT-2 at 5,000).
  - Two W-4 rounds with κ = 1: ≤ 1.31 (φ 0.010) and ≤ 1.55 (φ̂ FT-SELF at n 16) [DERIVED, sd2_numbers two_round_factor_kappa1].
- **(c) The empirical record:**
  - Round-1 FT on w_1 ≈ 4e-4 produced a labeller whose fresh step **improved** at m 13–16: strict 0.997 / 0.989 / 0.984 / 0.957 vs D3-T's 0.972 / 0.952 / 0.940 / 0.917.
  - Its loss at m ≥ 18 is coverage, with precision 1.000 [MEASURED].
  - No sign of compounding. The noise-only κ cannot be identified without a gold-labelled twin (RT's GOLD-DEPTH); stated, not claimed.
- **(d) Lee 2502.01612** (Fig 5): the avalanche starts when **unfiltered** self-label accuracy is below ~0.9.
  - Ours: unfiltered chain strict accuracy ≥ 0.957 at n 16, and the filtered π 0.9995 (1 − π ≈ 5e-4, about 200× below Lee's 0.1).
  - Lee's 5 % structured noise (Fig 24) is about 100× our kept-wrong share.
  - Biased errors hurt more (App B.4). The bias-class mass is ≤ 2(1 − π) ≤ 1e-3 [PROVED, PROOFS §3c]. LABEL-AUDIT-2 prints the kept-wrong handle patterns.
- **LABEL-AUDIT-2 must show:** π ≥ 0.99 and π^E ≥ 0.99 pooled **and per window**, each window's CI overlapping its GATE's CI. Else LABEL-NOISY-2 (censor).

### 3.4 S4: hypotheses and pre-stated predictions for n 17–24 [CONJECTURE for each H; DERIVED numbers, sd2_tables.json / sd2_numbers.json]

Round-1 inputs [MEASURED eval_vb acc_n2]: a_0 (D3-T, B 8) and a_1 (FT-SELF, B 12).

| | B | n50 (margin, margin/B) | n90 (margin, /B) |
|---|---|---|---|
| D3-T | 8 | 10.31 (2.31, **0.288**) | 9.07 (1.07, 0.134) |
| FT-SELF | 12 | 15.46 (3.46, **0.288**) | 13.93 (1.93, 0.161) |

Hypotheses (d = n − 16):
- **H_shift**: round 1's curve translated by +4 in n and K′: a_2(16 + d) = a_1(12 + d). Round 1's growth was one-off; the margin stays 3.46.
- **H_prop**: the picture scales with B about the boundary: a_2(16 + d) = a_1(12 + 0.75d), linear interpolation. Margin/B stays 0.288; n50 20.6.
  - For this +4 step H_prop coincides with "margin grows by a constant 1.15 per round", since 0.288 × 4 = 1.15. They would differ only under a different window.
- **H_full**: no stop ≤ 24, stated as a floor.
- **SHRINK**: the margin falls back to the base's: a_2(16 + d) = a_0(8 + d).
- **H_T** (nuisance): failure keyed on untrained position rows (T > 61), read by the T-split, not by M.
- a_1 at n 17–20 is unread. The extension used [ASSUMED]: 0.180 / 0.124 / 0.110 / 0.110, i.e. the excess over the 0.11 floor decays at the last ratios' trend. The scorer replaces it by FT-SELF's eval_deep (row N5), which affects only n 21–24 under H_shift.

| n | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | **M2** (17–20) | M_far (21–24) | n50 | STOP-LOC-2 | K′-BEYOND-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H_shift | 0.999 | 0.893 | 0.633 | 0.344 | 0.18 | 0.12 | 0.11 | 0.11 | **0.717** | 0.13 | 19.46 | 19–20 | 0.155 |
| **H_prop** | 0.999 | 0.946 | 0.828 | 0.633 | 0.416 | 0.262 | 0.166 | 0.124 | **0.852** | 0.24 | 20.63 | 21 | 0.29 |
| H_full (floor) | ≥ 0.97 | ≥ 0.95 | ≥ 0.93 | ≥ 0.90 | ≥ 0.88 | ≥ 0.85 | ≥ 0.82 | ≥ 0.80 | ≥ 0.94 | ≥ 0.84 | > 24 | 24+ | ≈ 0.6–0.8 (a_18 at n 19–20 can finish at most one reduction in the coda) |
| SHRINK (base margin) | 0.923 | 0.603 | 0.266 | 0.127 | 0.14 | 0.13 | 0.14 | 0.12 | 0.480 | 0.13 | 18.31 | 18 | −0.002 |
| SE at 1,000 per n (H_shift / H_prop) | .001/.001 | .010/.007 | .015/.012 | .015/.015 | .012/.016 | .010/.014 | .010/.012 | .010/.010 | **0.0059 / 0.0052** | | | | ≈ 0.01 (paired) |

- **Discrimination by sampling** is not the issue.
  - DBS-2's bar 0.80 is 14 SE above H_shift and 10 SE below H_prop.
  - MARGIN-GROWS-2's 0.785 (the midpoint 0.784) is 11.5 SE / 13 SE away.
  - K′-BEYOND-2's 0.25 is ≈ 10 SE above H_shift; H_prop sits at 0.29, near the bar.
  - The open question is the curve family, and seed variance (Kuo: 0.09–0.12 across from-scratch runs).
- **Which readings split which hypotheses:**
  - H_shift vs H_prop: M2 (DBS-2, MARGIN-GROWS-2), n50, K′-BEYOND-2.
  - H_prop vs H_full: M_far, STOP-LOC-2.
  - SHRINK: M2 < 0.60.
  - H_T: the within-n T-split at n 19–22 (T ≤ 61 vs ≥ 63).
- **STOP-LOC-2 under H_shift** is 19 if round 1's G(15) window (n 16 only) carries over exactly. The assumed extension gives G(19) 0.025, so 20. Descriptive either way.

### 3.5 Training memory [DERIVED, sd2_numbers memory_train; form ASSUMED, validated once]

- Model (PROOFS §11, T_ref 31):

  GiB(K, T, m) = 2 + 0.93 + [0.58·K·T/31 + 0.0258·(T/31)²·Σ_{t=1..K}(t + 1)] / m, at bs 64.

- **Validation on round 1:** K 16, T 47, m 2 projects 14.47 GiB; measured 13.91 (2-step mem_check with AdamW, s215_ftself_s1_mem). m 1 projects 26.0; measured OOM.
- **Round-2 deepest shape** (the 64 longest n-16 pool records, K 20):
  - T(64th of 57k draws) = 59 [DERIVED].
  - m 1: 46.5; m 2: **24.7 > 21.0**; **m 4: 13.8 GiB**. At T 61: 14.4. At T 63: 15.0.
  - So micro 4. The trainer's loop {1, 2, 4} suffices; T1's "8" is a safety margin only.

### 3.6 Time and cost [DERIVED, sd2_numbers time; models ASSUMED]

- **Measured round-1 inputs:**
  - forced G-TIME: self (K 16, T 47, micro 2) 0.896 s; plain 0.439 s;
  - real FT-SELF 7,260 s and FT-CTRL 4,877 s for 12k steps, so a real self step is 0.804 s (0.897 × forced) and a real plain step 0.406 s.
- **Round-2 forced deep self step** (K 20, T 59, micro 4), two bounding models:
  - compute-bound (a·K·T + b·T²·ΣK, fitted to both measured points): **1.52 s**;
  - launch-bound (∝ K × micro): **2.24 s**.
  - Mixed-window batches put almost every self batch near the deep shape (P(Kb = 20) ≈ 0.80 per batch), so real ≈ 0.9 × forced.
- **Projection** (G-S10's registered form: 0.5 forced-self + 0.5 forced-plain, 12k steps, + 0.4 h overhead): **3.7–4.9 h → $0.62–1.27** at $0.17–0.26/h. Real is expected at 3.4–4.4 h → $0.57–1.15.
- Not chosen (open question 10): window-homogeneous 64-blocks in the pool file would save ≈ 0.5–0.7 h per cell (≈ $0.15) with no trainer change, at the price of a different batch composition from round 1.

### 3.7 Eval memory on the 3070 [DERIVED, evalmem_cpu.jsonl]

- CPU fp32 rollout peaks, with codas at every K′, on the longest items. The per-item slope comes from the B 8 → 16 difference, because small-B deltas are biased low by heap reuse after torch.load. The B 4 → 8 slopes agree within 3 %.
  - **n 16, K′ 24: 0.136 GiB/item; n 24, K′ 28: 0.236 GiB/item.**
- **Checks against GPU facts:**
  - 64 × 0.136 = 8.69 GiB vs the 3090 mem_check's 8.93 GiB (10.33 minus weights and AdamW; bf16 bs 64 at n 16 / K′ 24).
  - Round 1's 3070 bs-32 n-16 eval: 4.82 GiB projected (it ran). Holding it plus a K′ 18 rollout projects ≈ 8.4 (it OOMed, vbank_eval l.132's comment).
- **eval_deep at n 24, K′ 28** (+ 0.47 GiB weights): bs 32 → **8.03 GiB (over 8 GB)**; **bs 16 → 4.25 GiB**; bs 8 → 2.36.
- So eval_deep uses eval_bs 16, gated by G-S16. fp32 padded batches equal per-item results to 5.7e-15 in fp64 (vbank_eval docstring). bs-dependence is therefore at near-ties only, and BS-CHECK measures it.

---

## 4. Readings, definitions and the scorer's field contract

Notation:
- a(n) = acc at K′ n + 2; a_k(n) = acc at K′ k.
- **n ≤ 16 readings come from eval_vb.json** (unchanged harness: bs 32, K′ 0..16 and 24).
- **n 17–24 readings come from eval_deep.json** (bs 16, K′ 0..28).
- Chance ≈ 0.10–0.11 [MEASURED testset_ext majority share 0.101–0.108 at n 21–24].
- Every reading is per cell and comparator-free. The H_shift reference M_shift2 = 0.717 is a fixed number from FT-SELF's own eval_vb.

### 4.1 Files per R2 cell (boxes/runs_fsq/s216_r2_s1/, s216_r2_s2/)

| file | producer | fields read | readings |
|---|---|---|---|
| log.jsonl | trainer | step, answer_ce, plain_answer_ce, self_answer_ce, self_reuse, pool_cursor, c | dead (< 12,000), c = 0, reuse = 0 |
| args.json, args_diff.json | trainer | diff vs **s215_ftself_s1/args.json** (s1) / s216_r2_s1/args.json (s2: seed, init_from, self_pool, out, self_micro) | ARGS-DIFF |
| draws_log.json | trainer | coin, self_n, U, plain_hash | G-S3 (P 0.455 at t ≥ 13; s1 plain hashes == round 1's) |
| eval_vb.json | vbank_eval | acc_n2, acc_k (0..16, 24), probe, test_meta, vswap, stall, stepcap | ABSORB-2, RETAIN, FT-DIVERGED, K′24-STABLE, VB-C1, DEPTH-GENERAL-2 (descriptive), THINK-NECESSARY-DEEP-2 (descriptive) |
| **eval_deep.json** (new, §4.2) | part `deep` | acc_n2, acc_k, items, meta | DBS-2, MARGIN-GROWS-2, M_far, STOP-LOC-2, K′-BEYOND-2, RISE-PAST-K-2, n50, C2-TAIL-2, NOT-SIZE-KEYED-2, BS-CHECK |
| eval_steppos.json, eval_lfk.json | vbank_diag | as §213 | LENGTH-FREE |
| eval_hist24.json | vbank_diag | intact, bank_ablate_slva, self_ablate, other_ablate, dup_newest | KEEPS-SELF-STAB-2, MEM history, MEM sub-label |
| eval_selfdepth.json (round-parameterised, E2) | part `selfdepth` | memo (n 9..16), label_audit (pooled + by window, per-item flags), items_n2 / items_k24 (n 1..16) + T, contamination | MEMO-GAP-2, LABEL-AUDIT-2, CONTAMINATED, per-item store |
| eval_chainlabel(_fresh).json (fresh to n 24, E3) | vbank_diag | by_n keep, prec_kept, precision_strict | LABELLER-KEPT-2 (descriptive; round-3 labeller) |
| eval_rs, eval_d3hist, eval_vbhalt | vbank_diag | as §215 | descriptive |

Shared: `s216_pool_s1/pool_meta.json` {seeds, per_n, ns, draws_n, collide_n, dup_n, keep_all, sha256, builder_commit, labeller, labeller_sha256, card, dtype, bs, excl}; `s216_gate2_s1/eval_gate.json` {by_n 9..16 {window, items, n_kept, keep_kv, pi, pi_E, CIs, per-item records}, pooled_by_window, GATE_2, GATE_1p}. Baselines (row N5): eval_deep.json in s213_d3t_s1 / s215_ftself_s1 / s215_ftctrl_s1.

### 4.2 eval_deep.json contract (new part; no code here)

- **Producer:** `python -m boxes.vbank_diag --run <cell> --part deep --eval_bs 16 --device cuda:<3070>`, local, fp32 (no autocast), tf32 off.
- **Items:** `N.test_set(1000, n_max=24)` restricted to n 13..24 (12,000 items, seed 17100).
- **Batching:** chunks of eval_bs within each n, in test order (as part_acc).
- **Rollout:** one K′ 28 rollout per chunk, codas at every K′ ∈ 0..28. K′ n + 2 ≤ 26 is in the grid; n 24 reaches K′ n + 4 = 28.
- **Fields:**
  - `meta` {run, mode, card, dtype "fp32", eval_bs, per_n 1000, n_lo 13, n_hi 24, K_grid [0..28], test_seed 17100, test_n_max 24, **items_sha256** (of the concatenated syms + y of the 12,000 items, which the scorer asserts), model_sha256, commit, sec, **peak_gib** (max_memory_allocated)};
  - `acc_n2` {n: acc at K′ n + 2};
  - `acc_k` {K′: {n: acc}} for K′ 0..28;
  - `items` {n: {"y": [...], "T": [...], "height": [...], "ans": [[the argmax answer at K′ 0..28] per item]}}. This is 348k ints (≈ 1.5 MB) and stores every per-item output behind every threshold reading;
  - `chance` {n: majority-class share};
  - `def` (text).
- **Not computed:** bank-off / bank-ablate arms, probes, vswap.
- **Cost:** ≈ 10–30 min on the 3070 [ASSUMED]; peak 4.25 GiB projected at the deepest chunk.

### 4.3 Primary and mechanism readings (per R2 cell)

| reading | exact definition | role | FT-SELF (round-1 model) |
|---|---|---|---|
| **ABSORB-2** | min_{n 13..16} a(n) ≥ 0.95 (eval_vb) | in support; enabler; **never C2 / C4** | 0.344 (n 16) |
| **RETAIN** | min_{n 9..12} a(n) ≥ 0.99 (eval_vb). Failure = ERODED (a reading, part of SD2-SELF) | must not break | 1.000 |
| DEPTH-GENERAL-2 | mean_{n 13..16} a(n) ≥ 0.80 | descriptive, IN-SUPPORT | 0.717 |
| **DBS-2** | M2 = mean_{n 17..20} a(n) ≥ **0.80** (eval_deep) | **the C2 reading** (one lineage: an instance, not a rate) | (baseline read post-registration) |
| **MARGIN-GROWS-2** | M2 ≥ **0.785** (the H_shift 0.717 / H_prop 0.852 midpoint) | mechanism: did the margin grow again? | — |
| M_far | mean_{n 21..24} a(n) | enters FULL | — |
| **STOP-LOC-2** | G(k) = mean over n ∈ [k + 1, min(k + 4, 24)] of a(n) − a_k(n), k 13..23, no readability filter. STOP = min k with G(k) ≤ 0.02, else "24+". STOP-TO-SUPPORT-2 iff STOP ≥ 17 | descriptive; enters SHIFT / MIXED | round 1: 15 |
| **K′-BEYOND-2** | KB2 = mean_{n 19, 20}[a(n) − a_18(n)] ≥ **0.25**. a(n) is read at K′ 21–22, above trained K 20 (the round-1 form shifted by +4) | C2-literal row, on DBS-2's line | round 1: 0.155 |
| RISE-PAST-K-2 | mean_{n 19, 20}[a(n) − a_20(n)] | descriptive (≈ 0 under every H, saturation at K′ = n) | — |
| n50 / m50 | first 0.5 crossing of a(n) over n 13..24 (linear interpolation); m50 = n50 − 16, and m50/16 beside 0.288 / 0.288. Never crossing = "> 24" | descriptive; H_shift 19.46, H_prop 20.63 | 15.46 |
| C2-TAIL-2 | a_28(n) ≥ a(n) − 0.05 at every n 17..20 | descriptive | — |
| THINK-NECESSARY-DEEP-2 | max(a_0, a_1) ≤ 0.15 at every n 13..16 | descriptive (C1 at the new depth) | — |

**Mechanism label** (exhaustive; only under ABSORB-2, else **UNLABELLED (¬ABSORB-2)**):
- **FULL**: M2 ≥ 0.80 ∧ M_far ≥ 0.60.
- **GROW**: M2 ≥ 0.785 ∧ ¬FULL. Contains H_prop.
- **SHIFT**: 0.60 ≤ M2 < 0.785 ∧ STOP ≥ 17.
- **MIXED**: 0.60 ≤ M2 < 0.785 ∧ STOP ≤ 16.
- **SHRINK**: M2 < 0.60.

The panels read H_shift → SHIFT, H_prop → GROW, H_full → FULL, base-margin → SHRINK [DERIVED sd2_numbers panels]. The bars sit at the midpoints between adjacent hypotheses.

### 4.4 Kept and descriptive readings

| reading | definition | round-1 FT-SELF |
|---|---|---|
| LENGTH-FREE | score_s213.no_knee ∧ score_s212.lfk MEETS (as registered) | ✓ |
| VB-C1 | score_s211.cell_read parts, dead = 12,000, no NO-ENTRY censor (as §215) | ✓ |
| **K′24-STABLE** | a_24(n) ≥ 0.99 at every n 1..8, **unchanged** | ✓ **at the edge**: 0.990 at n 1. FT-CTRL reads 1.000 there, so the dip is self-item-specific [MEASURED eval_vb] |
| K′24-WIDE (descriptive) | a_24(n) ≥ a(n) − 0.01 at n 9..16 (eval_vb); a_28(n) ≥ a(n) − 0.05 at n 13..20 (eval_deep) | — |
| **KEEPS-SELF-STAB-2** | score_s213.hist24: gate ∧ CONTENT-STABILISES ∧ STAB_self ≥ 0.5 (n 3..6). Else **MEM-TRADED-2** | 0.489 (MEM-TRADED) |
| MEM history (descriptive) | STAB_self D3-T 0.782 → R1 0.489 → R2 x; **MEM-ERODING** iff x < 0.439 (= 0.489 − 0.05) | — |
| MEM sub-label (descriptive; Goodhart note §5.3) | **MEM-TRADED (robust)** iff STAB_self < 0.5 ∧ STAB_dup_newest ≥ 0.10 ∧ intact a_24 = 1.0 at n 3..6. **MEM-TRADED (lost)** iff CONTENT ≠ CONTENT-STABILISES | robust (dup_newest 0.877; self-ablated a_24 0.43–0.60 vs D3-T 0.21–0.23) [MEASURED eval_hist24] |
| **LABEL-AUDIT-2** | 5,000 random consumed records (rng 22670): π ≥ 0.99 ∧ π^E ≥ 0.99, pooled and per window, each inside its GATE's 95 % CI. Else LABEL-NOISY-2 (censor) | 0.9996 / 0.9996 |
| MEMO-GAP-2 | \|acc(K′ n+2) on the first 2,000 consumed pool items − on the 2,000 GATE items\| ≤ 0.02 at n 9..16 | 0 gap |
| LABELLER-KEPT-2 | fresh chain keep / prec_kept at n 13..24 (E3) | descriptive (round 3's question) |
| **NOT-SIZE-KEYED-2** | within n 17..22: acc for T ≤ 61 vs T ≥ 63, plus the within-n OLS slope of correct on T (eval_deep items). **H_T flag** iff the T ≥ 63 deficit is ≥ 0.10 at ≥ 2 of n 19–22 with ≥ 100 items per band | round 1: no within-n T effect |
| BS-CHECK | \|eval_deep a(n) − eval_vb a(n)\| ≤ 0.005 at each n 13..16 (bs 16 vs 32). Failure is reported beside every eval_deep reading, not a censor | — |

### 4.5 Lineage rows (s1, s2)

- **SD2-C2 ×2** = SD2-C2 in s216_r2_s1 **and** s216_r2_s2. Wording: "in both competent lineages (2/2; base seeds 1 and 2)". It is not a rate: rates need ≥ 3 seeds and the user's OK.
- A non-competent s2 base reads "s2 base not competent (C6 data point); SD2-C2 instance in 1 of 1 competent lineage".
- If s216_r1_s2 fails v1 §9's round-2 condition, s2 reads "lineage stopped at round 1", and its round-1 label is reported (GROW replicated or not).

### 4.6 Scorer theory/score_s216.py (to write; fixture-checked before any output)

- **Reuses:**
  - score_s215: args_diff (BASE parameterised), stop_loc (k-range parameterised), the censors, gate1 (window list);
  - score_s213: no_knee, hist24, jl;
  - score_s212: lfk;
  - score_s211: cell_read's VB-C1 parts, entry_on (s2 base).
- **New:** deep_read (M2, M_far, STOP-LOC-2, KB2, RISE, n50, C2-TAIL-2, T-split, items_sha256 assert), mechanism2, retain, gate2 (two windows), label_audit2 (by window), memo2, contamination2 (vs test n_max 24 and the round-1 pool), MEM history and sub-label, BS-CHECK, competence (s2 base), lineage rows, and the H_shift reference panel from FT-SELF's eval_deep (descriptive).
- **Fixtures (G-S12):**
  - exact H_shift (→ SHIFT, M2 0.717, KB2 0.155, ¬DBS);
  - H_prop (→ GROW, DBS ✓ 0.852, KB2 0.29 ✓);
  - H_full floor (→ FULL);
  - base-margin (→ SHRINK);
  - MIXED (M2 0.70, STOP 16);
  - ¬ABSORB-2 (UNLABELLED);
  - RETAIN fail (ERODED, ¬SD2-SELF);
  - dead; FT-DIVERGED; LABEL-NOISY-2 (one window 0.985);
  - CONTAMINATED (a pool record equal to a test n-22 item);
  - ARGS-DIFF (self_kmax 24);
  - MEM-TRADED-2 robust vs lost;
  - eval_deep missing (every n ≥ 17 reading fails);
  - items_sha256 mismatch (void);
  - s2 base not competent; s2 stopped at round 1;
  - H_T flag; BS-CHECK fail.

---

## 5. Claims, wording, censoring, Goodhart

### 5.1 Claims and pre-stated wording

- **SD2-SELF** = ABSORB-2 ∧ RETAIN ∧ LENGTH-FREE ∧ VB-C1 ∧ K′24-STABLE. Wording (only this): *"latent depth extended from 12 to 16 by distilling the round-1 model's own re-rooted single-step chain into the loop. The labeller was itself self-trained; labels are gold-equivalent at π ≥ 0.999 (GATE-2); no gold labels beyond n 8; in support; not C2, not C4."* It is an enabler, never a headline.
- **SD2-C2** = SD2-SELF ∧ DBS-2: *"depth extrapolation past the self-labelled range (n 17–20, untouched §170-extension items; C2 proxy, one lineage)"*. K′-BEYOND-2 goes on the same line (C2-literal: K′ 21–22 > K 20). The mechanism label and n50 are stated beside it.
- **SD2-C2-MEM** = SD2-C2 ∧ KEEPS-SELF-STAB-2. MEM-TRADED-2 and the MEM history are stated beside any positive.
- **Never:** ABSORB-2, RETAIN, DEPTH-GENERAL-2, MARGIN-GROWS-2, STOP-LOC-2 or THINK-NECESSARY-DEEP-2 worded as C2 or C4.
- **C4** (strict and revised) stays unmet. Round 2's labeller inherits a step learned from simulator R2 states at n ≤ 8. "Self-trained labeller" is a procedural property.
- **C1:** the decode / re-render is training-time only; test-time computation is latent.
- **Plainly:**
  - Under H_shift each round buys +4 in support at a constant margin of ≈ 3.5. C2 at n 17–20 then needs another round.
  - Under H_prop the margin grows ∝ B and round 2 is predicted to clear DBS-2.
  - SD2's DBS-2 is therefore a direct H_prop test. One lineage is an instance; 2/2 is the north-star direction ("every competent seed").

### 5.2 Censoring (rule b; each fails every reading of that cell)

- dead (log last step < 12,000, guard stops included);
- **FT-DIVERGED**: a(n) < 0.99 or a_24(n) < 0.97 at any n 1–8;
- LABEL-NOISY-2; CONTAMINATED; an ARGS-DIFF mismatch;
- **eval_deep missing or items_sha256 mismatch** → every n ≥ 17 reading fails (DBS-2, MARGIN-GROWS-2, K′-BEYOND-2 count as failures), and the mechanism reads UNLABELLED;
- never-reached events: n50 not crossed reads "> 24"; STOP "24+". Neither is ever treated as missing. Chance-level items count as wrong;
- a failed GATE-2 with both labellers means no round 2 (no cell, no spend).

### 5.3 Goodhart analysis (targets new in SD2)

| target / route | finding |
|---|---|
| Stop at the new boundary (pass ≥ 17) | Zero loss on the support (S0a). This **is** H_shift, irreducible by loss. Read by M2 / STOP-LOC-2, not prevented |
| Copying the self-trained labeller's systematic errors (Lee App B.4: biased errors hurt more) | kept-wrong ≈ 5e-4; class mass ≤ 2(1 − π). LABEL-AUDIT-2 prints the patterns per window |
| Selection by the labeller (kept items easier; Literature failure mode 1) | ABSORB-2's denominator is **all** §170 test items, so selection cannot inflate it. KEPT-VS-DRAWN is descriptive. Round 1's ceiling argument (keep 0.979 at n 12) did not bind (loop 1.000) |
| Memorising pool items | Fresh-once; MEMO-GAP-2 at n 9–16 |
| Oracle loop count (Fan's max-confidence, Kuo's best-K) | No per-item or per-n K′ selection. Readings at fixed K′ n + 2 and fixed K′ 18 / 20 / 28. The full grid is stored |
| KEEPS-SELF-STAB's direction | STAB_self = 1 − a_24(self-ablated) when intact = 1. It falls when the model becomes **more robust without own-slot history**, not only when memory is lost. Round 1's drop came with self-ablated a_24 rising from 0.21–0.23 to 0.43–0.60, while dup_newest still destroys stability (0.877). The registered rule is kept. The robust / lost sub-label is descriptive and pre-stated here [interpretation CONJECTURE] |

---

## 6. OC (one seed per cell; realised spend at registration $0) [arithmetic DERIVED, sd2_oc.json; every prior ASSUMED]

**Priors given ABSORB-2:** round-1 shaded (0.65 / 0.25 / 0.03 / 0.07), shifted by the pre-stated GROW rule. H_grow is 0.375 (contains H_prop), H_shift 0.542, H_full 0.025, SHRINK 0.058.

**Other assumed values:**
- P(ABSORB-2) 0.75: a better start than round 1 (min 0.344 vs 0.127), half the per-n exposure at 2× round 1's absorption point, deeper K and longer T.
- Kept factor = LENGTH-FREE 0.9 × VB-C1 0.9 × K′24-STABLE **0.70** (at the edge in round 1) × RETAIN 0.95 = 0.539.
- P(DBS-2 \| H): 0.05 / 0.60 / 0.95 / 0. P(MG \| H): 0.10 / 0.75 / 1.0 / 0. P(KB \| H): 0.05 / 0.45 / 0.90 / 0. The order is H_shift / H_grow / H_full / SHRINK.
- KEEPS-SELF-STAB-2 0.25 (branch M-S 0.20).
- Independence as v1 §6: the kept readings are independent of H, and the beyond-support readings are 0 without ABSORB-2.

| reading | P(pass \| no effect) | P(pass \| effect) | **headline** |
|---|---|---|---|
| ABSORB-2 | ≈ 0 (no R2 effect: FT-SELF n 16 0.344, ≈ 40 SE) | 0.75 [ASSUMED] | 0.75 |
| RETAIN | ≈ 1 if intact (p 0.999: P(≤ 10 errors / 1,000) ≈ 1); 0.01 under 2 % erosion | — | 0.95 |
| DBS-2 | H_shift: ≈ 0 by sampling (14 SE); **0.05** curve prior | H_prop ≈ 1 by sampling (10 SE); **0.60** H_grow family; H_full 0.95 | 0.21 |
| MARGIN-GROWS-2 | H_shift: ≈ 0 by sampling (11.5 SE); 0.10 prior | H_prop ≈ 1 (13 SE); 0.75 family; 1.0 H_full | 0.27 |
| K′-BEYOND-2 | H_shift 0.155: ≈ 0 (≈ 10 SE); 0.05 prior | H_prop 0.29: 0.45 (near the bar); H_full 0.90 | 0.16 |
| SD2-SELF | ≈ 0 | 0.539 | **0.40** |
| **SD2-C2** | **0.020** (\| H_shift) | **0.24** (\| H_grow), **0.38** (\| H_full) | **0.111** |
| SD2-C2-MEM | ≤ 0.005 | 0.06–0.10 | 0.028 |
| SD2-C2 ×2 (both lineages) | ≤ 0.001 | — | **≈ 0.02–0.04**: 0.026 (independent) to 0.055 (correlated), × P(s2 competent ∧ its R1 passes) ≈ 0.8 × 0.85 |
| GATE-2 value terms | π 0.993: 0.014; π 0.995: 0.52 | π 0.997: 0.9987; measured 0.9995: ≈ 1 | ≈ 1 |
| GATE-2 π^E term (≈ 7,858 kept, ≤ 78 wrong) | π^E 0.988: ≈ 0.05 (round-1 analogue at 7,920 kept: 0.051, v1 §6; not recomputed) | π^E 0.993: ≈ 1 (0.9989 at 7,920) | ≈ 1 |
| LABEL-AUDIT-2 (5,000) | π 0.985: P(≤ 50 wrong) ≈ 0.002 | π 0.9995: ≈ 1 | ≈ 1 |
| KEEPS-SELF-STAB-2 | ≈ 1 if the FT does no damage (but R1 is already at 0.489) | — | 0.25 [ASSUMED] |

- **Sensitivity** (H_grow prior): 0.25 (no shift) → P(SD2-C2) 0.085; 0.375 → 0.111; 0.50 (taking H_prop's 2-point fit at face value) → 0.138.
- **How to read it:**
  - A DBS-2 pass is strong evidence for growth: P 0.02 under H_shift vs 0.24 / 0.38.
  - A null is weak against H_grow (0.60).
  - MARGIN-GROWS-2 and n50 are the informative splits.
  - SD2-SELF passes under H_shift, so it says nothing about C2.

---

## 7. Gates (all green before any spend; Audit reviews)

| gate | check |
|---|---|
| **G-S0 NO-GOLD** | as v1: builder gold patches before generation; key-set assert at load; SD-LABEL-MAP grep; no `s216_gate` in vbank.py; self-batch call-counter smoke |
| G-S1 EXCLUSION | pool ∩ (test n_max 24 ∪ SERIAL ∪ 213-37 ∪ GATE-1 ∪ GATE-2 ∪ GATE-1′ ∪ round-1 pool) = ∅ by exact syms; per-n draws / collisions logged; n(syms) == generator n |
| G-S2 COMPOSE | trainer E_self (HOLD-padded to **Kb 20**) == recode_targets(root, 20) bitwise on GATE-2 / GATE-1′ kept items and the first 1,000 pool records (separate process). The only allowed differences are the wrong-state items, listed |
| G-S3 STREAMS | from draws_log: P(R \| t ≥ 13) = 0.455 ± 0.01, 0 at t ≥ 17; self U only from rng_s; **s1: plain_hash at steps 1..200 == s215_ftself_s1/draws_log.json** |
| G-S4 INIT + RESUME | `--init_from FT-SELF --keep_ckpt_steps 0 --stop_at 2 --ckpt_every 1`: ckpt_model_0 == FT-SELF model.pt on every key; INIT_FROM prints missing []; calibration.json == FT-SELF's; ckpt_last at steps 1 and 2; a resume runs step 3 with cursor / rng_s / AdamW restored |
| G-S5 ARGS-DIFF | in the launch log: only init_from, self_pool, self_kmax, self_micro, out, args_ref and the META keys differ from s215_ftself_s1/args.json (s2 cells: plus seed) |
| G-S6 LOAD | VE.load strict on the R2 model.pt |
| G-S7 PLAIN-PATH | only if T1 / T2 land: p_self 0, construct + 1 step, CPU-bitwise vs the pre-edit tree, on the final tree |
| G-S8 GRADFLOW | scratch/gradflow_check.py on a round-2 self batch at Kb 20 (n 16 items): nonzero R1 and R2 grads on SLVA, core, R, coda; decorators by ast; `--log_every 1` runs mass_stats |
| **G-S9 MEM** | on the R2 host: `--mem_check_self --mem_check_steps 2` at the 64 longest n-16 records, K 20, bs 64; peak after step 2; self micro = the smallest in {1, 2, 4(, 8)} with peak ≤ 21.0 GiB (projected 4 at 13.8 GiB) |
| G-S10 TIME | forced G-TIME (odd steps deep self, even plain) on each host → projection → §8 admission |
| G-S11 OFF-HOST | full-state mirror every 2k, sha-checked after the first copy; the resume is G-S4's, tested before launch (user rule 2026-10-15) |
| G-S12 FIXTURES | score_s216 fixtures (§4.6) pass before any output |
| **G-S13 DIFF REVIEW** (Audit) | B1 (builder: labeller path, the new exclusion set, no simulator call), B2, E1 (part_deep: items, K′ grid, per-item store, fp32, bs), E2 / E3 (paths, ranges; s215 behaviour bitwise-unchanged when run with round 1's args), any T1 / T2 |
| G-S14 LAUNCH | 3090 only (card recorded per cell); drain waiters armed at launch; c = 0 logged; no module edit while any deploy is in flight; requeue watchers checked (memory: requeue ships the dirty tree) |
| **G-S15 GATE-2 / GATE-1′** | §2.1 rule on the 3070 (fp32, bs 32), file eval_gate.json, **before** the pool is used |
| **G-S16 EVMEM** (new) | on the 3070: part_deep's deepest chunk (the 16 longest n-24 items, K′ 28, fp32, bs 16) peak ≤ 7.0 GiB, before any eval_deep output; else bs 8 (pre-stated). Also confirm bs 32 OOMs or exceeds 7.0 (records the projection) |
| **G-S17 SMOKE-AT-SCALE** (new) | the builder's exclusion set at full size (≈ 495k syms) and the trainer's pool load / G-S9 deep-shape selection on the full 448k pool, locally, before deploy, with timing logged (b310's O(N²) hang lesson) |
| G-S18 BASELINES (new) | eval_deep on D3-T s1 / FT-SELF / FT-CTRL **after** SD2's registration commit (row N5); BS-CHECK on each (vs their eval_vb at n 13–16) |
| s2 only | G-S5 for s216_d3t_s2 (seed only vs s213_d3t_s1/args.json); COMPETENT-BASE before GATE-1-s2; G-S0…S17 per s2 cell |

---

## 8. Cost, caps, admission, guards, wall-clock

| item | where | hours | projected $ | admission (G-TIME projection at the host's price) | in-flight guard (realised, incl. refused / BAD hosts) |
|---|---|---|---|---|---|
| Baselines eval_deep × 3 | 3070 | ≈ 1–1.5 | 0 | — | — |
| s1 pool (448k) + GATE-2 / GATE-1′ (16k), FT-SELF labeller | 3070 | **1.2** (4,250 s + 150 s at FT-SELF's measured sec/item) [DERIVED] | 0 | — | — |
| **s216_r2_s1** | 3090 | 3.7–4.9 (forced) / 3.4–4.4 (real) | **0.62–1.27** | ≤ **$1.35** | **$1.60** |
| s216_d3t_s2 (48k) | 3090 | 6.2 forced (s1 real 4.96 h) + 0.4 | 1.12–1.70 | ≤ $1.75 | $2.00 |
| GATE-1-s2 + r1-s2 pool (440k, D3-T s2 labeller) | 3070 | ≈ 1.0 | 0 | — | — |
| s216_r1_s2 | 3090 | 2.2–2.6 | 0.45–0.68 | ≤ $0.95 | $1.05 |
| GATE-2-s2 + r2-s2 pool | 3070 | ≈ 1.2 | 0 | — | — |
| s216_r2_s2 | 3090 | as r2_s1 | 0.62–1.27 | ≤ $1.35 | $1.60 |
| Mirror egress | | | ≲ 0.05 | | |
| Evals: full panel per FT cell ≈ 4.5–5 h (eval_vb 65 min, deep, steppos, lfk, rs, d3hist, hist24, vbhalt, chainlabel ± fresh, selfdepth); s2 base ≈ 0.3 h (competence: acc part + chain fresh) | 3070 | ≈ 15–16 | 0 | — | — |
| **Total (recommended: s1 R2 + lineage b)** | | | **$2.9–5.0** | | **family cap $6.5** (GUARD_STOP; the sum of the guards is $6.25) |
| Alternative (a) instead of (b) | | | $1.7–3.3 | | family cap $4.5 |
| Optional R2-CTRL (branch M-S only, user's choice) | 3090 | ≈ 2.3 | 0.45–0.68 | ≤ $0.60 | $0.70 (family cap + $0.70) |

- **Confirmed ≤ $30**: family cap $6.5 (7.2 with R2-CTRL). But **the user's explicit OK is required** (MEM-TRADED, v1 §9). Round 1 realised ≈ $1.07 of $1.50 for the pair [RELAYED].
- **Admission and retries** as v1 §8 with notes 215-2 / 215-4:
  - a projection above its bound means no launch on that host; at most 2 more measured refusals per cell; BAD hosts don't use a try but count toward the guard;
  - **steps are never reduced** (12k / 48k are the registered budgets);
  - a guard stop is a dead run.
- **Parallelism:** s216_d3t_s2 needs no new code (G-S5: seed only) and can launch immediately after registration, alongside s1's engineering. The R2 cells run on separate 3090 hosts.
- **The 3070 is the bottleneck:** ≈ 20 h of labelling + evals across the family, plus §214's pending evals. One eval card per experiment (same-card rule).
- **Wall-clock:** s1 ≈ 11 h after code review (labelling 1.2 h, train ~4 h, evals ~5 h); the s2 lineage ≈ 24 h from its launch.
- **Retention:**
  - keep model.pt of every R2 cell (the round-3 labeller) and of s216_d3t_s2 / s216_r1_s2;
  - keep evals, logs, args;
  - delete checkpoints and mirrors at landing;
  - delete pools after LABEL-AUDIT-2 / MEMO-GAP-2 (pool_meta + seeds + builder commit regenerate them).

---

## 9. Open questions for Theory's decision

1. **Second seed: (b) new lineage (recommended) vs (a) FT replicate.** (b) costs ≈ $1.1–1.6 more and ≈ 8 h more wall-clock (mostly overlapped). It is the only option that tests "every competent seed".
2. **Reading the sealed n 17–20 on round-1 models after registration** (row N5). The round-1 text says "sealed … round 2's extrapolation set"; this draft reads it as allowing round 2's descriptive baselines once SD2's thresholds are frozen. Confirm, or drop N5 and keep the assumed a_1 extension (affects only H_shift's n 21–24 panel and STOP).
3. **eval_bs 16 for eval_deep** (a departure from the bs-32 convention, with BS-CHECK) vs a K′ grid only to 26 at bs 32 (projected 7.7 GiB, still over budget) vs a larger local card.
4. **T1:** add micro 8 to the self auto_micro loop (safety only; needs G-S7)? **T2:** per-window self CE logging (descriptive)?
5. **MEM sub-label (robust / lost)** as a pre-stated descriptive split. It changes no registered rule, but it reframes round 1's MEM-TRADED for the user's OK question.
6. **R2-CTRL:** offer it to the user in branch M-S, or never?
7. **Priors:** P(ABSORB-2) 0.75, P(DBS \| H_grow) 0.60, K′24-STABLE 0.70 (edge risk), P(s2 base competent) 0.8. Red-team shading as round 1? Also whether to keep the pre-stated × 1.5 shift given H_prop's 2-point fit (sensitivity 0.085–0.138).
8. **K′24-STABLE stays exactly as registered** (n 1, 0.990 in round 1). A second FT may push n 1 below 0.99, which fails SD2-SELF. No relaxation is proposed; confirm.
9. **Thresholds:** MARGIN-GROWS-2 at 0.785 sits 0.015 below DBS-2's 0.80, so M2 ∈ [0.785, 0.80) reads GROW but not C2. SHRINK < 0.60; FULL needs M_far ≥ 0.60; the H_T flag rule. Confirm.
10. **Pool order:** mixed windows (round-1 semantics, chosen) vs window-homogeneous 64-blocks (≈ $0.15 per cell cheaper, no trainer change, different batch composition).
11. **Seed 1 for s216_r2_s1** (replays round 1's plain stream; enables the G-S3 identity check) vs a fresh seed.
12. **Pending §215 pair:** branch P2 (FT-EFFECT) halts SD2. Is that the right pre-statement, or should SD2 run anyway with the attribution worded away?

---

## 10. Verification log (this draft)

- **testset_ext.json** [MEASURED, CPU]:
  - test_set(1000, 24)[:20000] == test_set(1000, 20) and [:16000] == test_set(1000) (syms and y);
  - T by n: max 61 / 73 / 83 at n 17 / 20 / 24, mean 73.9 at n 24;
  - 593 duplicate syms in the 24k, all at small n (the §170 convention);
  - 0 SERIAL hits; majority-class share 0.101–0.108 at n 21–24.
- **tsplit.json** [DERIVED from eval_selfdepth items_n2 + T]: FT-SELF n 13–16 accuracy by T. No within-n T effect (n 16: ≤ 47 0.319, ≥ 49 0.352).
- **evalmem_cpu.jsonl** [MEASURED, CPU fp32, B 4 / 8 / 16]: slopes 0.136 GiB/item (n 16, K′ 24) and 0.236 (n 24, K′ 28). The 3090 bf16 mem_check is reproduced within 3 %.
- **sd2_numbers.json:**
  - crossings, panels (H_shift / H_prop / SHRINK on the 2-D acc_k table; H_full floor), SEs;
  - GATE-2 exact OC (convolution);
  - label noise and κ (FT-SELF and D3-T);
  - pool z 18.3;
  - memory (round-1 validation 14.47 vs 13.91; round-2 micro 4 at 13.8 GiB);
  - time (compute- and launch-bound bounds);
  - labelling 1.2 h.
- **sd2_tables.json:** per-n tables from acc_n2 directly (M2: H_shift 0.7172, H_prop 0.8516, SHRINK 0.4798; SE 0.0059 / 0.0052).
- **sd2_oc.json:** OC arithmetic and sensitivity. **It supersedes sd2_numbers.json's `oc` block**, whose kept factor (0.567) omits RETAIN.
- **Fixed during drafting:** sd2_numbers' two-round factor first used D3-T's P(16) (a loop-variable leak); fixed and re-run, giving 1.31 / 1.55.
- **support_check_r2.json:** PROOFS' exact-rational S1 script, unchanged (it already contains the round-2 law).
- **Code read** [VERIFIED]:
  - vbank.py l.545–580 (CurGate), 593–899 (main: init_from asserts, pool load, the G-S9 deep shape from n_top, draw_step, the l.812 self exclusion, logging, checkpoint);
  - rollout l.262–345 (no pass cap); slva.py l.48–73 (age bias continuous);
  - vbank_eval part_acc (n 1..16, KS 0..16 + 24);
  - vbank_diag part_selfdepth (s215-hard-coded paths), part_chainlabel_fresh (rng 21337, n 9..20);
  - selfdepth.py (label_chains / compose_E generic in n);
  - nested170 test_set / generate;
  - scratch/s215_build_pool.py (exclusion composition).
- **Still unbuilt (flagged):** B1, B2, E1–E3, score_s216, T1 / T2 if chosen, queue scripts. Memory, time and eval-time projections are model extrapolations; G-S9 / G-S10 / G-S16 decide.

*Draft written 2026-10-18 (Opus design subagent for Theory; $0; nothing committed, registered or launched).*
