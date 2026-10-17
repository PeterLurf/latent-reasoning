# SELF-DEPTH-CHAIN v1 (§215 round 1): registration-ready text

Clock 2026-10-17 12:12 EDT. Opus design-revision subagent for Theory. **$0: no training, no GPU, no repo edits** (CPU checks only, 1 thread).
User approval (relayed in §214's header, 11:58 EDT): SELF-DEPTH-CHAIN round 1 = the FT pair FT-SELF + FT-CTRL from D3-T, **cap $1.5, not poolable with §214**, conditional on a v1 that fixes every review finding.
Sources: docs/research_selfdepth_2026-10-17/{SELFDEPTH_draft.md (ac1f854d), PROOFS_SELFDEPTH.md, REDTEAM_SELFDEPTH.md, scripts/}; THEORY notes 213-19, 213-26, 213-30, 213-33…213-39, §214; docs/LIT_SELF_DEPTH_2026-10-11.md addendum (a84321b9); boxes/vbank.py (working tree, 12:01), slva.py, vbank_diag.py (part_chainlabel / _fresh), nested170.py, vbank_eval.py; theory/score_s211/212/213/214.py.
New $0 checks: `research4/v1_checks/v1_checks.py` → `v1_checks.json` (STOP-LOC, K′-BEYOND, mechanism partition, π^E OC, pool sizing, OC arithmetic).
Tags: [PROVED], [PROVED UNDER X], [DERIVED] (exact arithmetic on measured inputs; script named), [MEASURED] (raw file named), [ASSUMED], [CONJECTURE].

---

## 0. One screen

| item | v1 |
|---|---|
| Question | Does FT on self-labelled n 9–12 items (labels from D3-T's own re-rooted fresh single step; no gold field beyond n 8) (a) **absorb** n 9–12 into the latent loop, and (b) move the depth margin **past** the new support (n 13–16, untouched §170 test items)? |
| What round 1 tests | **Mainly ABSORB** (in support, capacity). C2 needs either many rounds (each buys +4 in-support depth; ≈ linear in $) or a boundary that stops binding (H_full). DEPTH-BEYOND-SELF is the one C2 reading |
| Cells | `s215_ftself_s1` (p_self 0.5) and `s215_ftctrl_s1` (p_self 0; SD's own control). RTX 3090 only, seed 1, 12k × bs 64, FT from `s213_d3t_s1/model.pt` |
| Labeller | D3-T, RE-ROOT CHAIN (213-33), local 3070, fp32, bs 32. Filter **KEEP ∧ COMPOSE-VALID** |
| Pool | 110,000 kept fresh items per n 9–12 (440k), seed 21900, one fixed interleaved order (perm seed 21901), fresh-once, no gold field, exhaustion = void |
| Self law | **Uniform n 9–12 from step 1 (no self curriculum)**; per step coin p_self 0.5; K = min(n + U{0..4}, 16); coin and U from rng_s = seed + 21800 |
| Headline OC | **P(SD-C2) ≈ 0.03** (red-team shaded; draft 0.075); P(ABSORB) 0.65, **P(SD-SELF) ≈ 0.45**; P(SD-C2 \| no effect) ≈ 0.007 |
| Cost | Pair projected **$0.82–1.43**; cap **$1.5**; G-TIME above a cell's admission bound → **no launch on that host** (steps never reduced) |

---

## 1. Change table (every review finding)

| # | draft | v1 | source |
|---|---|---|---|
| P1 | STOP-LOC with "readable = acc ≥ 0.18 at ≥ 2 n"; base table says 10 | Unfiltered G(k), k 9..15, class "≥ 16" if none; **base reads 10, exact H_shift reads 14 (= B + 2)**; mechanism partition made exhaustive (+ MIXED, UNLABELLED); H_shift fixture | PROOFS §5; RT §6; v1_checks |
| P2 | 120k kept per n; stage A n 9–10 up to 4k steps | **Stage A dropped** (RT §4 option): uniform n 9–12; 110k per n; use 96k ± ~0.9k per n; P(exhaust) ≈ 0 (z 16); exhaustion or reuse = void | PROOFS §10 (135k / 85k under stage A); RT #10 |
| P3 | GiB model at T_ref 33, no AdamW, budget 21.5; kextra-2 fallback micro 1 | T_ref 31, + 0.93 GiB AdamW, budget **21.0** (vbank default); self K 16 forced 26.0 GiB → micro 2; **kextra-2 fallback deleted** (22.3 GiB → micro 2 too, $0.48–0.91: saves ≤ $0.09 and changes the design) | PROOFS §11; RT #15–16 |
| P4 | "COMPOSE-VALID extra drop ≈ 0 because every value-correct chain is chain-exact" | Premise replaced by the measurement: **CV == KEEP on 3,960 / 3,960 kept chains (n 9–12) and 1,892 / 1,892 (n 13–16); π^E 3,959 / 3,960 = 0.99975**. Composition theorem restated: E_self = gold at every t ⇔ every step value correct. "Strict ⇔ chain-exact" (test items) replaces "every value-correct chain is chain-exact" (false on fresh n 9) | PROOFS §7; RT #1, #3, #4 |
| P5 | S2: w ≤ φ(1 − P); κ ≥ 5 at n 20; two rounds ≤ 2.7× / 1.6×; "≥ 0.997 every n" | w among **kept** (×1/P); A1′ covers untrained m ∈ W_{q+1}; κ ≥ **8.1 / 5.7 / 3.9** at n 12 / 16 / 20; two W-4 rounds ≤ **1.32× / 1.15×**; window π 95 % lower **0.9994 / 0.9988 / 0.9986**; per-n min 0.9968 (fresh n 15); D3-F test 0 / 541; S2 stated on states (π^E ≤ π) | PROOFS §3; RT #3 |
| P6 | Trainer edit list: R2 only | vb_loss reads `it["t"]` (R1) and `it["n"]` (window): the loader maps `chain_label → it["t"]` in one named line; n from syms; **self K offsets from rng_s**; G-S9 peak **after step 2** | PROOFS §9 |
| P7 | G-S3 target P(R \| t ≥ 9) = 0.500 | **0.494** (trainer's plain law; loss-weighted 0.490); 0 at t ≥ 13 | PROOFS §2 |
| P8 | small items | S3(a) [PROVED UNDER a per-example step at the labeller's weights]; S0(a) R1 term [ASSUMED]; H_shift SE 0.0062; item space n 1–4 = 5k draws; GATE value-term OC exact (0.9986 / 0.504 / 0.0125) + π^E-term OC computed here | PROOFS §0, §4, §8, §6 |
| R1 | pool size; cap units unstated | P2 (no self curriculum, so no cap units) | RT blocker 1 |
| R2 | `cal` unbound under init_from → NameError at the first checkpoint (vbank.py ~710) | init_from binds `cal` from the base calibration.json (**present in the uncommitted working-tree vbank.py at 12:01 EDT, ll. ~605–609**; Audit reviews the committed version); G-S4 smoke `--stop_at 2 --ckpt_every 1` + resume before launch | RT blocker 2 |
| R3 | slva.mass_stats → rec_targets → recode_targets(it["root"]) on self batches (crash / gold read) | rec_targets branches on `E_self`; mass_stats calls VB.rec_targets, so it inherits the branch; **self items carry no `root`** (any gold path is a KeyError, never a silent read); G-S8 smoke `--log_every 1` on a self batch with a recode_targets call counter | RT blocker 3 |
| R4 | STOP-LOC unassignable under H_shift | P1 | RT blocker 4 |
| R5 | LABEL-AUDIT and S2 label-level | LABEL-AUDIT state-level (π^E) and label-level; S2 on states; compensated chains reported (1 / 5,852 measured; 274 / 2,875 in the draft's own fixture) | RT blocker 5 |
| R6 | streams and gates | coin + U from rng_s; self n from the fixed pool order; G-S3 hashes (syms, K) incl. U; **no self CurGate** (moot); `--mem_check_steps 2` at the deepest round shape; E_self HOLD-padded to the **batch** Kb (G-S2 at Kb 16); **KEEPS-SELF-STAB on n 3..6** (scorer code and the 0.782 baseline) | RT blocker 6 |
| R7 | K′-BEYOND unthresholded (RT: vs K′ 16, e.g. ≥ 0.10) | RT's form is **≈ 0 under every hypothesis** (in-range accuracy saturates at K′ = n: acc 1.000 at K′ = n for n 3–8 [MEASURED eval_vb]), so v1 reads **K′-BEYOND = mean_{n 15,16}[acc(K′ n+2) − acc(K′ 14)] ≥ 0.25** (reads at K′ 17–18 > K 16); RT's form kept as descriptive RISE-PAST-K | RT #19; v1_checks |
| R8 | "primary C2" / "beyond gold-supervised depth" / THINK-GAIN-BEYOND "C2-literal" | Wording rules §5.1; THINK-GAIN-BEYOND, C2-TAIL, DEPTH-GENERAL, STOP-MOVED, SELF-ATTRIBUTED labelled non-discriminating (pass under H_shift) | RT #8, #9, #19 |
| R9 | OC 0.075 headline | Shaded headline 0.032; draft beside; P(pass \| no effect) per reading | RT §6 |
| R10 | pool on host unspecified | pool + GATE on the 3070, fp32, bs 32; pool sha256 in pool_meta.json and asserted at load; shipped with the deploy | RT §5 |
| A1 | ABSORB "beyond gold-supervised depth" | **ABSORB is in support: never C2, never C4**; only "enabler" / RT's honest phrase | Audit (i) |
| A2 | — | "Round 1 mainly tests ABSORB; C2 needs many rounds or B never binding" stated (§0, §5.1) | Audit (ii) |
| A3 | S1-A: CTRL shared with D4 | **No sharing**: §214 is option B (a 48k scratch cell, no FT). FT-CTRL is SD's own | Audit (iii); §214 |
| A4 | gates | + Audit diff review (init_from w/o calibrate + `cal`, the chain labeller in the data path, the R1 window), ARGS-DIFF in the launch log, off-host mirror, drain waiters, 3090 only, c = 0 logged | Audit |
| N1 | — | **n 17–20 of the extended §170 set sealed in round 1** (round 2's extrapolation set) | new |
| N2 | — | vbank.py is co-edited by §214's pause path: one reviewed commit or sequenced commits; G-S7 on the final tree; no module edit while any deploy is in flight | new (memory: deploys ship the working tree) |
| N3 | GOLD-DEPTH (RT §7.3) | Not chosen (the user approved SELF). Predicted to read the same within ≤ 2.4 % selection and ~1e-3 state noise [DERIVED, RT §2.2]; stated so readers know the labels are gold-equivalent | RT #9 |

---

## 2. Design (registration text)

### 2.1 Labeller and filter

- **Chain** (213-33; part_chainlabel_fresh's loop): cur ← input; repeat: x = [EOT] + render(cur) + [" ="]; rollout K = 1; dec = argmax R(V_1); s = rerender(dec); cur ← _parse_tree(s); stop at no operator (label = value) or step cap n + 4 (FAIL). Always the model's own decode, never the simulator.
- **KEEP** = parses at every step ∧ exactly n steps (n = # operator tokens of the input syms).
- **COMPOSE-VALID (CV)** = KEEP ∧ FRAME ∧ SPAN ∧ ORDER ∧ CANON at every step (compose.py definitions; ORDER from the parsed input's tree shape only).
- **Kept = KEEP ∧ CV.** On D3-T's fresh decodes CV adds no drop: 3,960 / 3,960 (n 9–12), 1,892 / 1,892 (n 13–16) [MEASURED rt_sd/real_compose_n9_12.json, _n13_16.json].
- **R2 targets** E_self,t: E_0 = x's classes; E_k = E_{k−1} with step k's span VOIDed over the live slots and the value on the span's last slot; HOLD after n, **padded to the batch Kb**.
  - [PROVED, PROOFS §7] Under CV, E_self = vbank.recode_targets(gold root) at every t ⇔ every step value is correct; else equal before the first wrong step.
  - [MEASURED] 100,000-item composition gate: 850,429 steps, 0 failures (draft); 20k reproduction 170,011 steps, 0 failures (RT).

### 2.2 Contamination rules (registration text)

1. **Training items are fresh generated items only.** Pool: `nested170.generate(rng_pool, n)`, rng_pool = default_rng(21900), n 9–12, drawn until 110,000 kept per n. No §170 test item, SERIAL item, gate item or 213-37 item (seed 21337) is ever a training item.
2. **Exclusion by exact rendered syms** against test_set(1000, **n_max 20**) ∪ the SERIAL pool (rng 17300, 2,200 items) ∪ the GATE-1 items. An excluded draw is redrawn. **Per n: draws, collisions, collision rate are logged** in pool_meta.json. Expected 0 at n ≥ 9: P(any hit) ≤ 1e-7 [PROVED, PROOFS §8].
3. **No gold reaches training.**
   - Builder: N.apply, N.value, N.eval_tokens, N.cot_steps, VB.recode_targets and vbank_diag.reduce_k are monkeypatched to raise **before generation** (generate uses the PAIRS table built at import).
   - The generator's (root, t) are dropped right after render. n is computed from syms and asserted equal to the generator's n, which is then dropped.
   - **Pool record keys are exactly {syms, n, chain_label, steps}** (steps = per step (j_lo, j_hi, j_val, value) in the original frame). The trainer's loader asserts this key set on every record.
   - **Assert: no training-pool record has a gold field** (root, t, y, gold, E, recode). The G-S0 grep allows `chain_label` in exactly two places: the builder's writer and the loader's line `it["t"] = rec["chain_label"]  # SD-LABEL-MAP`.
   - The trainer never opens the gate directory (grep: 0 hits for `s215_gate` in boxes/vbank.py).
4. **Gold is used only for evaluation and GATE-q**, in separate processes: GATE-1, LABEL-AUDIT, MEMO-GAP, the eval files. Gold for an audited record is recomputed from its syms (`_parse_tree` → recode_targets / value), never stored in the pool.
5. **The extrapolation test is the untouched §170 test set at n 13–16** (1,000 per n, test_set seed 17100). "Untouched": never in a pool, gate or labeller-selection set. The base D3-T's earlier evaluations on it (213-19 / 26 / 35) were read before registration and inform only the predictions.
6. **The n 17–20 extension of test_set(1000, n_max 20) is sealed in round 1** (used only as an exclusion list): it is round 2's extrapolation set.
7. The labeller is fixed in advance (D3-T model.pt). It is never chosen on pool or test items. Pool and GATE-1 are labelled on the same card class and settings (3070, fp32, bs 32).
8. Post hoc: pool ∩ (test ∪ SERIAL ∪ gate) = ∅ re-verified on the pool file; any hit, any gold key, any reuse (`self_reuse` > 0) or exhaustion → **CONTAMINATED / void**.

### 2.3 Cells and training

| element | spec |
|---|---|
| Init | `--init_from boxes/runs_fsq/s213_d3t_s1/model.pt`: strict load, **missing keys = [] asserted** (the working-tree clause tolerating `pause_vec` is D4's), calibrate() / ratio_test **not** called, `cal` = the base calibration.json (copied to out), fresh AdamW |
| Recipe | D3-T's args.json **+** `--lr 3e-5 --steps 12000 --cur 8 --c_sched 0,1` (c = 0 from step 1), lr_at 100-step warm-up then cosine to 0, wd 0.01, bs 64, seed 1, plain kmax 12, kextra 4 |
| FT-SELF only | `--self_pool <pool_r1> --p_self 0.5 --self_kmax 16 --self_kextra 4 --self_micro <G-S9>` |
| FT-CTRL | the same without self flags (p_self 0): every step is a plain D3-T-law batch |
| Streams | Every step draws the plain batch from rng (items) and rng_k (U), as D3-T. Then rng_s = default_rng(seed + 21800) draws the coin. On a self step the plain batch is discarded, the next 64 pool records are taken at the cursor, and their U offsets come from rng_s. **Plain batches of FT-SELF equal FT-CTRL's at the same step index** |
| Self law | the pool's fixed interleaved order (perm 21901) ≈ uniform n 9–12; P(R pass \| t ≥ t*) = 0.494 / 0.397 / 0.282 / 0.145 at t* 9–12, **0 at t ≥ 13** [PROVED, PROOFS §2] |
| Losses | vb_loss unchanged: R1 at t ∈ [n, K] on `it["t"]` (= chain_label for self items), R2 at t 1..K on E (E_self for self items) |
| Micro | plain micro 1 (D3-T's); self micro from G-S9 (projected 2) |
| Logging (log.jsonl, every 100) | D3-T's fields + `self_frac`, `self_answer_ce`, `self_rec_ce`, `plain_answer_ce`, `pool_cursor`, `n_self_steps`, `self_reuse` (must be 0), `c` |
| Draw log | `draws_log.json`: per step 1..20,000 coin, self n list, U list, sha1 of (syms, K) of the plain batch |
| Checkpoints | full state every 2k (model, AdamW, rng / rng_k / rng_tf / rng_s, pool cursor, step, elapsed, n_nonfinite, calibration), mirrored off-host; deleted at train end (model.pt kept) |

### 2.4 Trainer edits (Audit diff review; none written here)

1. init_from as §2.3 (the working-tree edit binds `cal`; SD adds `assert missing_keys == []` when `--self_pool` or SD's tag is set, or relies on G-S4's printed line).
2. Self-pool loader: sha256 check against pool_meta.json, key-set assert, SD-LABEL-MAP, E_self built from `steps` (compose), `n` from syms.
3. `rec_targets`: `if "E_self" in it:` HOLD-pad E_self to Kb; else recode_targets(it["root"], Kb). mass_stats inherits it.
4. Per-batch-type micro; self K from rng_s; the cursor and rng_s in ckpt_last.pt and in the resume branch.
5. `--mem_check_self` (with `--mem_check_steps 2`): the 64 longest n 12 pool records by T, K 16, bs 64, two optimiser steps, peak after step 2; then the n 16 K′ 24 eval forward.
6. No model change: SLVA's parameters are unchanged, so VE.load works with the existing factory [VERIFIED, draft].

---

## 3. Mathematics (corrected; full proofs in PROOFS_SELFDEPTH)

- **S0 support identity** [PROVED]. Every target state of the round-1 mixture has ρ + r = n ≤ 12 (CV ∧ STEPS: one VOIDed handle per step for n steps, HOLD after; values never enter the count).
  - (a) The old stop "ρ ≥ 8 ⇒ finished" pays loss on the R2 term at passes 9..n [PROVED]. On R1 only if the coda does not finish the remaining reductions in one shot [ASSUMED].
  - (b) "Stop at ρ ≥ 12" and any deadline t_d ≥ 13 are zero-loss on the support [PROVED]. The D3-T identifiability problem recurs one boundary out: this is H_shift, irreducible by loss.
  - (c) ABSORB's range n 9–12 is **in support**. The C2 test is n 13–16. n 15–16 are read at K′ 17–18 > trained max K 16 [PROVED].
- **S1** (exact, trainer law) [PROVED]: as §2.3. Any pass or age deadline at t_d ≤ 12 now costs loss (deadline cost 9.8 % / 5.7 % / 2.8 % / 0.9 % of an item's R2 term at t_d 9–12); t_d ≥ 13 is free. SD cannot tell a progress stop from a pass/age stop (§214 / D4 does that).
- **S2 noise** (on states).
  - (a) [PROVED, identity] π = P / (P + φ(1 − P)). The cited evidence is the direct window counts (D3-T, test + fresh): R1 n 9–12, 7,919 kept / 1 wrong, 95 % lower **0.9994** [MEASURED]. State level: π^E ≤ π [PROVED]; measured π^E 0.99975 at n 9–12 [MEASURED].
  - (b) [PROVED UNDER A1′, A2, A3′] ē_{q+1} ≤ ē_q (1 + κφ n_q / P_q). Avalanche needs κ ≥ 8.1 (n 12) / 5.7 (16) / 3.9 (20); per wrong transition ≈ 100× that. Round 1 is base-labelled, so noise enters once: w ≤ φ(1 − P)/(P + φ(1 − P)) ≤ 3e-4 at n 12 [PROVED UNDER φ ≤ 0.0097].
  - (c) A majority-wrong pattern class has mass ≤ 2(1 − π) ≤ 6.4e-4 [PROVED]. LABEL-AUDIT prints kept-wrong handle patterns (the fresh n 15 cluster is the place to look).
- **S3(a)** pass 1's target is the model's own argmax at FT step 0 [PROVED UNDER a per-example step at the labeller's weights; else ASSUMED]. No reading depends on it.
- **S4 predictions** [CONJECTURE for the hypotheses; DERIVED numbers, v1_checks]:

| hypothesis | n 13 / 14 / 15 / 16 | mean M | STOP-LOC | K′-BEYOND |
|---|---|---|---|---|
| **H_shift** (exact translation n + 4, K′ + 4) | 0.923 / 0.603 / 0.266 / 0.127 | **0.480** (SE 0.0062) | **14** (G(13) 0.061, G(14) −0.0015) | **−0.0015** |
| H_grow (δ(12) ≈ 1.5 δ(8)) | 0.95 / 0.82 / 0.60 / 0.38 | 0.69 | ≈ 15 | 0.25–0.34 (straddles) |
| H_full (no stop ≤ 16) | ≥ 0.9 / ≥ 0.9 / ≥ 0.85 / ≥ 0.80 | ≥ 0.85 | ≥ 16 | ≈ 0.67 |

---

## 4. Readings, definitions and the scorer's field contract

Notation: a(n) = eval_vb `acc_n2[n]` (K′ n + 2); a_k(n) = eval_vb `acc_k[k][n]` (k ∈ 0..16, 24). Chance 0.10. All evals local fp32, eval_bs 32, **3070 only**, §170 test set n 1–16 (1,000 per n). Every reading is per cell, comparator-free, except §4.4.

### 4.1 Files per cell (boxes/runs_fsq/s215_ftself_s1/, s215_ftctrl_s1/)

| file | producer | fields read | readings |
|---|---|---|---|
| log.jsonl | trainer | step, answer_ce, self_answer_ce, self_reuse, pool_cursor, c | dead (last step < 12,000), c = 0, reuse = 0, first 1k-bin self_answer_ce (reported) |
| args.json, args_diff.json, calibration.json | trainer | diff vs s213_d3t_s1/args.json | ARGS-DIFF |
| draws_log.json | trainer | coin, self_n, U, plain_hash | G-S3 |
| **eval_vb.json** (parts acc, probe, vswap, stall) | boxes.vbank_eval | acc_n2, acc_k, probe, test_meta, vswap, stall, stepcap | ABSORB, DEPTH-GENERAL, DBS, MARGIN-GROWS, STOP-LOC, K′-BEYOND, RISE-PAST-K, STOP-MOVED, THINK-GAIN-BEYOND, C2-TAIL, THINK-NECESSARY-DEEP, VB-C1, K′24-STABLE, FT-DIVERGED |
| eval_steppos.json | vbank_diag steppos | fresh_m_T, fresh_m | NO-KNEE, STEP-GENERAL |
| eval_lfk.json | vbank_diag lfk | (score_s212.lfk) | LFK |
| eval_rs.json, eval_d3hist.json | vbank_diag | as §213 | SELF-SIMILAR, HISTORY-USED (reported) |
| eval_hist24.json (DUP arms) | vbank_diag hist24 | intact, bank_ablate_slva, self_ablate, other_ablate, dup_newest, dup_foreign | KEEPS-SELF-STAB / MEM-TRADED |
| eval_vbhalt.json | vbank_diag | coda per t | FIN-K (descriptive) |
| eval_chainlabel.json, eval_chainlabel_fresh.json | vbank_diag | by_n precision_strict, keep, prec_kept | LABELLER-KEPT (descriptive in round 1) |
| **eval_selfdepth.json** (new) | new part `selfdepth` | memo {n: {acc_pool, acc_fresh, items}}, label_audit {n_audit, pi_label, pi_state, kept_wrong [{syms, step, pattern}]}, items_n2 {n: [0/1]} + T per item (n 9–16), contamination {pool_test_hits} | MEMO-GAP, LABEL-AUDIT, CONTAMINATED, T-split |

Shared (not per cell): `boxes/runs_fsq/s215_pool/pool_meta.json` {seed, perm_seed, per_n, draws_n, collide_n, keep_n, sha256, builder_commit, card, dtype, bs}; `boxes/runs_fsq/s215_gate1/eval_gate.json` {by_n {n: {items, n_kept, keep_kv, kept_wrong_label, kept_wrong_state, pi, pi_E}}, pooled {n_kept, pi, pi_E}, GATE_1}.

### 4.2 Primary and mechanism readings (FT-SELF; computed identically on CTRL)

| reading | exact definition | role | base D3-T |
|---|---|---|---|
| **ABSORB** | min_{n 9..12} a(n) ≥ 0.95 | in support; enabler; **never C2 / C4** | 0.127 (n 12) |
| DEPTH-GENERAL | mean_{n 9..12} a(n) ≥ 0.80 (§213) | reported, labelled IN-SUPPORT | 0.480 |
| **DEPTH-BEYOND-SELF (DBS)** | M = mean_{n 13..16} a(n) ≥ **0.80** | **the C2 reading** (one seed: an instance, not a rate) | 0.130 |
| **MARGIN-GROWS** | M ≥ **0.63** (H_shift 0.480 + 0.15) | mechanism: GROW vs SHIFT | — |
| **STOP-LOC** | G(k) = mean over n ∈ [k + 1, min(k + 4, 16)] of a(n) − a_k(n), **no readability filter**; STOP = min k ∈ 9..15 with G(k) ≤ 0.02, else "≥ 16". **STOP-TO-SUPPORT iff STOP ≥ 13** | descriptive location; enters the SHIFT / MIXED split | 10 (G(9) 0.050, G(10) −0.003) [DERIVED] |
| **K′-BEYOND** | KB = mean_{n 15,16}[a(n) − a_14(n)] ≥ **0.25**; a(n) is read at K′ 17–18 > K 16 | C2-literal row beside DBS | analogue (n 11–12 vs K′ 10) −0.0015 |
| RISE-PAST-K | mean_{n 15,16}[a(n) − a_16(n)] | descriptive; ≈ 0 under every H (saturation at K′ = n) | analogue (vs K′ 12) −0.0005 |
| STOP-MOVED | score_s213.acc_gain_after10 mean_gain ≥ 0.10 | descriptive / attribution (≈ 1 under ABSORB) | −0.003 |
| THINK-GAIN-BEYOND | mean_{n 13..16}[a(n) − a_12(n)] | descriptive; ≈ 0.30 under H_shift | — |
| C2-TAIL | a_24(n) ≥ a(n) − 0.05 at every n 13..16 | descriptive | — |
| THINK-NECESSARY-DEEP | max(a_0(n), a_1(n)) ≤ 0.15 at every n 9..12 | descriptive (C1 at the new depth) | — |

**Mechanism label** (exhaustive; only under ABSORB, else **UNLABELLED (¬ABSORB)**): **FULL** if M ≥ 0.80; **GROW** if 0.63 ≤ M < 0.80; **SHIFT** if 0.40 ≤ M < 0.63 ∧ STOP-TO-SUPPORT; **MIXED** if 0.40 ≤ M < 0.63 ∧ STOP ≤ 12; **SHRINK** if M < 0.40. The exact-H_shift panel reads M 0.480, STOP 14 → SHIFT [DERIVED, v1_checks].

### 4.3 Kept readings (must not break)

| reading | definition | base |
|---|---|---|
| LENGTH-FREE | score_s213.no_knee NO_KNEE (m 9, m 10; now in the trained range: a kept check, not a test) ∧ score_s212.lfk MEETS | ✓ |
| VB-C1 | score_s211.cell_read DEEP ∧ PROBE-GAIN ∧ SERIAL-ONSET ∧ SERIAL-STEP ∧ NO-CLOCK (n 3–8), with dead = last step < 12,000 and no NO-ENTRY censor | ✓ |
| K′24-STABLE | a_24(n) ≥ 0.99 at every n 1..8 (score_s214 logic) | ✓ (1.000) |
| **KEEPS-SELF-STAB** | score_s213.hist24: gate (intact == eval_vb bitwise) ∧ CONTENT == CONTENT-STABILISES (STAB_bank ≥ 0.10 ∧ STAB_dup_newest ≥ 0.10) ∧ STAB.self_ablate ≥ 0.5, **mean over n 3..6**. Else **MEM-TRADED** ("depth gained, memory channel lost") | 0.782 |
| STEP-GENERAL / LABELLER-KEPT | fresh_m exact ≥ 0.95 at m 12–16; chain strict n 13–16 ≥ base − 0.02 | descriptive in round 1 |
| NOT-SIZE-KEYED | PROBE(12) extra-size slope (REDTEAM_D4 F7 method, rt_probe_size.py) > −0.10 at n 9–16; T-split of items_n2 at n 13–16 | descriptive |
| MEMO-GAP | \|acc(K′ n+2) on the first 2,000 consumed pool items − on the 2,000 GATE-1 items\| ≤ 0.02 at n 9..12 (gold from syms, eval only) | descriptive |
| **LABEL-AUDIT** | 5,000 random consumed records (rng 21970): **π^E ≥ 0.99 and π ≥ 0.99**, each inside GATE-1's 95 % CI (overlap). Else **LABEL-NOISY** (censor) | — |
| **GATE-1** (pre-launch, eval only) | 2,000 held-out fresh items per n 9–12 (rng 21960, §2.2 exclusion, never trained), kept = KEEP ∧ CV, vs gold: **π_pool ≥ 0.995 ∧ π_n ≥ 0.99 at every n ∧ π^E_pool ≥ 0.99 ∧ keep_n ≥ 0.70 at every n**. π^E = #{kept ∧ E_self(t) = gold E(t) ∀ t ∈ 0..n} / #kept, item-weighted | expected pass |

### 4.4 FT-SELF vs FT-CTRL (SD's own control)

- **SELF-ATTRIBUTED** iff FT-SELF STOP-MOVED ∧ CTRL ¬STOP-MOVED. **FT-EFFECT** iff CTRL STOP-MOVED (attribution void). It attributes the **in-support** absorption to the self items, not extrapolation (it passes under H_shift).
- **MEM attribution** (as D4 v1): MEM-TRADED in SELF with KEEPS-SELF-STAB in CTRL → "traded by the self items"; both → "traded by the FT (lr re-heat)".
- **Erosion:** FT-DIVERGED in CTRL too → the FT itself.
- If CTRL is not run (§8 admission), these rows read UNREAD and every FT-SELF reading stands.

### 4.5 Scorer `theory/score_s215.py` (to write, fixture-checked before any output)

- Reuses **score_s213**: `no_knee`, `hist24` (n 3..6), `acc_gain_after10`, `chainlabel`, `chainlabel_fresh`, `jl`. **score_s212**: `lfk`, `rs_read`, `halt_read`. **score_s211**: `cell_read`'s VB-C1 parts, with FULL = 12,000 and the NO-ENTRY censor off (FT from an entered model). **score_s214**: `fin_k`, `curve_from_acc_k`, the K24 / KEEPS logic. ARGS-DIFF vs `s213_d3t_s1/args.json` over the §213 META whitelist.
- New: STOP-LOC, K′-BEYOND, RISE-PAST-K, the mechanism partition, GATE-1, LABEL-AUDIT, MEMO-GAP, CONTAMINATED, the CTRL rows.
- Fixtures (G-S12): exact H_shift (→ SHIFT, STOP 14, KB < 0.25), GROW, FULL, SHRINK, MIXED, ¬ABSORB (UNLABELLED), CTRL FT-EFFECT, dead, FT-DIVERGED, LABEL-NOISY, CONTAMINATED, MEM-TRADED, CTRL missing.

---

## 5. Claims, wording, censoring

### 5.1 Claims and pre-stated wording

- **SD-SELF** = ABSORB ∧ LENGTH-FREE ∧ VB-C1 ∧ K′24-STABLE. Wording (only this): *"latent depth extended from 8 to 12 by distilling the model's own re-rooted single-step chain into the loop; no gold labels beyond n 8; labels gold-equivalent at ≥ 0.999; in support; not C2, not C4."* It is an **enabler** (a depth-12 base for round 2 or D4), never a headline.
- **SD-C2** = SD-SELF ∧ DBS: *"depth extrapolation past the self-labelled range (n 13–16, untouched §170 test items; C2, one seed)"*. K′-BEYOND is reported on the same line as the C2-literal row.
- **SD-C2-MEM** = SD-C2 ∧ KEEPS-SELF-STAB, on the same line. MEM-TRADED is stated beside any positive.
- **Never:** ABSORB, DEPTH-GENERAL, STOP-MOVED, SELF-ATTRIBUTED or THINK-GAIN-BEYOND worded as C2 or C4. **C4** (strict and revised) stays unmet: the labeller's step was learned from simulator R2 states at n ≤ 8.
- **Plainly:** one round mainly tests ABSORB. Under H_shift (the default) each round buys +4 in-support depth while the margin past support stays ≈ +1–2, so depth d costs ⌈(d − 8)/4⌉ rounds at growing cost (memory ∝ T²K). C2 needs many rounds, or a boundary that stops binding (H_full), which is what DBS asks.
- **C1:** the decode / re-render is a training-time labeller only; test-time computation is fully latent (VB-C1 kept; THINK-NECESSARY-DEEP descriptive).

### 5.2 Censoring (rule b; each fails every reading of that cell)

- dead (log last step < 12,000; includes a guard stop, §8);
- **FT-DIVERGED**: a(n) < 0.99 at any n 1–8, or a_24(n) < 0.97 at any n 1–8 (base 1.000 / 1.000);
- **LABEL-NOISY** (LABEL-AUDIT fails); **CONTAMINATED** (§2.2 item 8);
- an ARGS-DIFF mismatch (void).
- A failed GATE-1 means no round 1 (no cell, no spend).
- Chance-level items count as wrong in every DEPTH reading. STOP-LOC is always defined (no unreadable state).

---

## 6. OC (one seed per cell; realised spend at registration $0)

**Independence assumptions** (stated): (a) the kept readings (LENGTH-FREE 0.9, VB-C1 0.9, K′24-STABLE 0.85; product 0.6885) are independent of H and of each other; positive correlation with ABSORB would make the conditionals conservative. (b) DBS = MARGIN-GROWS = K′-BEYOND = 0 without ABSORB. (c) KEEPS-SELF-STAB (0.7) is independent of H. (d) GATE-1 and LABEL-AUDIT pass with P ≈ 1 (measured π, π^E) and independently of everything else. All priors are [ASSUMED]; the arithmetic is [DERIVED, v1_checks.json; it reproduces PROOFS §12 and RT §6].

Priors given ABSORB: draft H_shift 0.55 / H_grow 0.30 / H_full 0.10 / SHRINK 0.05; **RT-shaded 0.65 / 0.25 / 0.03 / 0.07**, P(DBS | H_grow) 0.25 → 0.15 (no VB cell has shown a margin beyond +2).

| reading | P(pass \| no effect) | P(pass \| effect) | **shaded (headline)** | draft |
|---|---|---|---|---|
| ABSORB | ≈ 0 (no FT effect: base min 0.127) | — | **0.65** [ASSUMED] | 0.65 |
| SD-SELF | ≈ 0 | — | **0.45** | 0.45 |
| DBS | ≤ 0.01 (H_shift; 52 SE by sampling) | 0.9 (H_full) / 0.15 (H_grow) | 0.046 | 0.111 |
| MARGIN-GROWS | 0.07 (H_shift curve prior; ≈ 0 by sampling, 24 SE) | 1.0 / 0.85 | 0.19 | 0.26 |
| K′-BEYOND | ≤ 0.02 (H_shift −0.0015) | 0.95 (H_full) / ≈ 0.5 (H_grow) | 0.11 | — |
| **SD-C2** | **0.0069** | 0.62 (H_full) | **0.032** | 0.076 |
| SD-C2-MEM | ≤ 0.005 | 0.43 | 0.022 | 0.053 |
| SHIFT label | — | ≈ 1 under exact H_shift (fixture) | ≈ 0.42 (0.65 × 0.65) | — |
| STOP-MOVED / SELF-ATTRIBUTED | ≈ 1 under H_shift (non-discriminating); ≤ 0.02 for CTRL | — | ≈ 0.73 / 0.71 | 0.73 |
| THINK-GAIN-BEYOND, C2-TAIL, DEPTH-GENERAL | ≈ 1 under H_shift given ABSORB | — | descriptive | — |
| KEEPS-SELF-STAB | ≈ 1 if the FT does no damage (base 0.782) | — | 0.7 (MEM-TRADED 0.3) [ASSUMED] | 0.7 |
| GATE-1 value terms | P(pass \| π 0.993) **0.0125**; \| 0.995 0.504 | P(pass \| π 0.997) **0.9986** | ≈ 1 (window π ≥ 0.9994) | same |
| GATE-1 π^E term (N_kept ≈ 7,920, ≤ 79 wrong) | P(pass \| π^E 0.988) **0.051**; \| 0.99 0.52 | P(pass \| π^E 0.993) **0.9989** | ≈ 1 (measured 0.99975) | not computed |

- **Sensitivity:** with P(ABSORB) 0.45 (selection ceiling ≈ 0.976 at n 12 leaves 0.026 slack; 6k self steps at peak lr 3e-5), SD-SELF 0.31, SD-C2 **0.022**. The figure "≈ 0.45" quoted for ABSORB in CURRENT_TASKS is P(SD-SELF).
- **How to read it:** an SD-C2 positive is strong (P 0.007 under H_shift, 0.62 under H_full). A DBS null is weak against H_grow. MARGIN-GROWS and K′-BEYOND are the informative splits. Every other positive passes under H_shift. Rate claims need ≥ 3 seeds and the user's OK.

---

## 7. Gates (all green before any spend; Audit reviews)

| gate | check |
|---|---|
| **G-S0 NO-GOLD** | §2.2 item 3: builder patches before generation; record key set asserted at load; grep allows `chain_label` only in the writer and SD-LABEL-MAP; no `s215_gate` in vbank.py; a self-batch smoke asserts recode_targets is called only for plain items (call counter) |
| G-S1 EXCLUSION | pool ∩ (test_set(1000, n_max 20) ∪ SERIAL ∪ gate) = ∅ by exact syms; per-n draws / collisions logged; n(syms) == generator n asserted, then dropped |
| G-S2 COMPOSE | compose gate 0 diffs (100k, done); **trainer E_self (HOLD-padded to Kb 16) == recode_targets(root, 16) bitwise** on the GATE-1 kept items and on the first 1,000 pool records (roots re-derived from syms in a separate gate process) |
| G-S3 STREAMS | from draws_log: P(R pass \| t ≥ 9) = 0.494 ± 0.01 over 20k draws, 0 at t ≥ 13; plain (syms, K) hashes of FT-SELF == FT-CTRL on every plain step in 1..200 (U included); self U only from rng_s |
| G-S4 INIT + RESUME | `--init_from --keep_ckpt_steps 0 --stop_at 2 --ckpt_every 1`: ckpt_model_0.pt bitwise == base model.pt on every key; INIT_FROM line prints missing [] and "calibrate() NOT called"; calibration.json == base's; **ckpt_last.pt written at steps 1 and 2 (no NameError)**; a resume from it runs step 3 with cursor, rng_s and AdamW state restored |
| G-S5 ARGS-DIFF | printed in the launch log: only lr, steps, cur, c_sched, init_from, args_ref and (FT-SELF) the self flags differ from s213_d3t_s1/args.json |
| G-S6 LOAD | VE.load strict on the FT model.pt with the existing factory |
| G-S7 PLAIN-PATH | p_self 0, no pool: construct + 1 step on slva / pa / carry, CPU-bitwise vs the pre-edit code, **on the final tree after §214's edits** |
| G-S8 GRADFLOW | scratch/gradflow_check.py on a self batch: nonzero R1 and R2 grads on SLVA, core, R, coda; decorators by ast (no no_grad on the new paths); **`--log_every 1` on a self batch runs mass_stats without KeyError** |
| **G-S9 MEM** | on the FT host: `--mem_check_self --mem_check_steps 2` at the **deepest round-1 shape** (64 longest n 12 pool items, K 16, bs 64); peak after step 2 (AdamW allocated); self micro = the smallest in {1, 2, 4} with peak ≤ **21.0 GiB**; projected 26.0 GiB at micro 1 → micro 2 [ASSUMED model] |
| G-S10 TIME | G-TIME on forced self (K 16, micro from G-S9) and plain (K 12) batches on each cell's host → projection; §8 admission |
| G-S11 OFF-HOST | full-state mirror every 2k (§2.3), sha-checked after the first copy; the resume is G-S4's, tested before launch |
| G-S12 FIXTURES | score_s215 fixtures (§4.5) pass before any output |
| **G-S13 DIFF REVIEW** (Audit) | (i) init_from without calibrate() and with `cal` bound; (ii) the chain labeller in the data path (builder = part_chainlabel_fresh's loop + compose checks, no simulator call); (iii) the R1 window [n, K] on self items with n from syms and `t` from SD-LABEL-MAP; (iv) the rec_targets branch and mass_stats; (v) streams and per-type micro |
| G-S14 LAUNCH | drain waiters armed at launch for both cells; both cells on **RTX 3090** (one class; card recorded per cell); log shows c = 0 at every step ≥ 1; no module edit while any deploy (§214 or §215) is in flight |
| G-S15 GATE-1 | §4.3 rule on the 3070 (fp32, bs 32), file eval_gate.json, before the pool is used |

---

## 8. Cost and cap

Projection inputs: PROOFS cost_check (T_ref 31, AdamW 0.93 GiB, padded T plain 29 / self 42, micro 2 at K 16, 0.4 h overhead, $0.18–0.26/h) [DERIVED from one measured point; model ASSUMED].

| item | where | hours | $ |
|---|---|---|---|
| Pool (440k kept) + GATE-1 items (8k), base labeller | local 3070 | ≈ 0.9 (measured rate) – 1.3 (with compose checks) | 0 |
| GATE-1 scoring, G-S0…S8, S12 | local CPU / GPU | < 1 | 0 |
| **FT-SELF** 12k (self micro 2, K ≤ 16) | 3090 | 2.91–3.86 | **0.52–1.00** |
| **FT-CTRL** 12k | 3090 | 1.66 (1.84 at G-TIME's rate) | **0.30–0.43** (0.33–0.48) |
| **Pair** | | | **0.82–1.43**; a slow host at G-TIME's rate: 1.59 |
| Mirror egress | | | ≲ 0.02 |
| Evals (eval_vb ≈ 65 min, steppos, lfk, rs, d3hist, hist24 ≈ 8 min, vbhalt, chainlabel ±fresh, selfdepth) | local 3070 | ≈ 3.5–4 per cell | 0 |

- **Cap $1.5 for the pair** (guard), not poolable with §214. Realised $ replaces the projection.
- **Admission (pre-stated, G-S10):** a cell is launched on a host only if its G-TIME projection at that host's price is ≤ **$0.95 (FT-SELF)** / **$0.40 (FT-CTRL)**.
  - **If a projection is above its bound: no launch on that host** (destroy; the G-S9 / G-S10 spend counts toward the cell). Up to 2 more 3090 hosts per cell. If none admits, that cell is not run and the user is told.
  - **Steps are never reduced** (12k is the registered ABSORB budget).
- **In-flight guards:** FT-SELF stops at $1.05, FT-CTRL at $0.45 realised, including that cell's rejected-host spend. Total ≤ $1.50. A guard stop is a dead run (§5.2).
- The cells may run in parallel on two 3090 hosts (same cost, faster).
- If FT-CTRL is not admitted, FT-SELF still runs. The §4.4 rows then read UNREAD.
- Retention: keep model.pt (round-2 labeller / D4-on-SD base), evals, logs, args; delete checkpoints and the mirror at landing. Delete the pool after LABEL-AUDIT and MEMO-GAP (pool_meta.json + seeds + builder commit make it regenerable).

---

## 9. Lee's +1 per round vs this +4 window; round 2 (not registered, not funded)

| | Lee et al. 2502.01612 (LIT addendum a84321b9) | §215 round 1 |
|---|---|---|
| Labeller | the previous-round model's own greedy outputs at the new difficulty | the **base's fresh single step**, re-rooted per step (no counterpart in Lee or Huang) |
| Step per round | **+1** difficulty unit (≈ 60–100 rounds) | **+4** (n 9–12) |
| Why | the labeller *is* the extrapolating model; its accuracy at d + 1 is what gets bootstrapped | the labeller's reach (horizon 20, window π ≥ 0.9994) does not depend on the loop's depth, so no bootstrapping is needed in round 1 [MEASURED 213-38; RT] |
| Filter | length filter + multi-model majority vote; values never checked | KEEP ∧ CV (single model); values never checked label-free; GATE-1 / LABEL-AUDIT measure them vs gold (eval only) |
| Mixture | D_0 ∪ … ∪ D_r, newest up-sampled to 50 % | 50 % gold n ≤ 8 + 50 % self (one window in round 1) |
| Init | fine-tune from the previous round | FT from D3-T |
| Failure mode | error avalanche without filtering; model-made errors hurt more than random noise | avalanche needs κ ≥ 8.1 at n 12 [PROVED UNDER A1′–A3′]; round 1 is base-labelled (noise enters once) |

- **The +4 window's real risk is not labels but capacity:** the loop must learn 4 new depths at once (¬ABSORB, P ≈ 0.35). Dense per-pass R2 targets are the mitigation. The ¬ABSORB follow-up below is pre-stated.
- Huang 2511.07378 (a token-CoT theorem: doubling under a proven constant-factor margin) is **motivation only**. The latent loop's measured margin is +1–2 (0.923 / 0.603 at B + 1 / B + 2), which licenses no doubling.
- Steps per round: the draft's "Lee 1.5k–5k steps per round" is not in Literature's addendum. **Unverified**; not cited.

**Round 2 plan (a follow-up; needs its own registration, user OK and cap):**
- W_2 = n 13–16, B_2 = 16. Labeller = the FT-SELF model's chain (GATE-2: same rule on 2,000 held-out items per n 13–16). Pre-stated fallback **BASE-LABELLED** (D3-T: π^E 1,892 / 1,892 at n 13–16 on 500 per n [MEASURED RT]).
- Self mixture: newest window 50 % / round-1 window 50 %, all fresh items. K = min(n + U{0..4}, 20).
- Extrapolation: the sealed §170 extension n 17–20 (new eval contract; memory at n 20, K′ 24 on the 8 GB card to check).
- Memory: K 20, T 61 → 48.7 GiB forced → micro 4. **Cost $1.13–2.78** [DERIVED, PROOFS §11; model ASSUMED].
- Runs only if round 1 reads ABSORB ∧ ¬FT-DIVERGED ∧ LABEL-AUDIT ∧ ¬CONTAMINATED. GROW / FULL → round 2 plus a second seed. SHIFT → the ratchet is linear in $, optional (or D4-style pauses on the SD model to ask whether the residual stop at ~B + 2 is age-keyed). MEM-TRADED → only with the user's explicit OK.
- **Pre-stated OC shift:** SHIFT → P(H_grow | round 2) × 0.5; GROW → × 1.5 (cap 0.8).
- ¬ABSORB → one disclosed re-run (24k, or lr 1e-4) with the user's OK; else stop.
- A Lee-faithful S1-C (4 rounds × 3k at +1 each) stays an alternative, not funded.

---

## 10. Verification log (this revision)

- v1_checks.json [DERIVED from s213_d3t_s1/eval_vb.json]:
  - STOP-LOC base G(7..15) = 0.565 / 0.303 / 0.050 / −0.003 / … → **STOP 10**;
  - exact H_shift G(9..15) = 0.843 / 0.746 / 0.565 / 0.303 / 0.061 / −0.0015 / 0.001 → **STOP 14**;
  - K′-BEYOND −0.0015 (base analogue and H_shift), H_full panel 0.67, H_grow panel 0.25–0.34;
  - in-range saturation a_n(n) = 1.000 at n 3–8;
  - mechanism panels SHIFT / GROW / FULL / SHRINK / MIXED / UNLABELLED;
  - π^E OC; pool P(exhaust) 0 in 20k sims (max use 398,336 of 440,000; z 16);
  - OC 0.0763 (draft) / 0.0318 (shaded) / 0.0220 (shaded, P(ABSORB) 0.45).
- Code read [VERIFIED]:
  - vbank.py working tree (uncommitted at 12:01 EDT): `--init_from` with `cal` bound, the `pause_vec`-only missing-key clause; vb_loss reads it["t"], it["n"]; rec_targets → recode_targets(it["root"]); draw() U from rng_k;
  - slva.mass_stats → VB.rec_targets; nested170.generate uses PAIRS only;
  - score_s213.hist24 averages n 3..6; score_s211 FULL = 48,000 (override needed); vbank_eval acc_k K′ 0..16, 24.
- **Still unbuilt (flagged):** every trainer edit (§2.4), the pool builder, eval part `selfdepth`, score_s215, `--mem_check_self`. Memory and time projections are model extrapolations; G-S9 / G-S10 decide.

*v1 written 2026-10-17 (Opus subagent for Theory; $0; nothing committed or registered).*
