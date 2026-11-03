# 09 LINEAGE COMPRESSION: BRIDGE2, ONE-RUN SCHEDULE, STATE-DISTILL (wave-4 lead design, 2026-11-03)

Status: design, not registered. Read-only; no outputs computed. Every number quoted comes from the THEORY.md note named inline.

## 1. Goal and backlog IDs

bun (§259) ends a 68k-step lineage (from the six args.json files): s233_askname (8k, direct, d_max 2) -> aw_plaw2 (12k, inject K 8, d <= 2) -> rcb_plaw4 (16k, REG d <= 4) -> dis (16k, distract_p 1.0) -> dch (8k, distract_mix s256,chain,givop) -> bun (8k, hold_inc 1.6, hold_start 500). Every stage restarts warmup + cosine at lr 1e-4 (fsq120.lr_at: `min(1, step/100) * 0.5 (1 + cos(pi step / steps))`), a fresh AdamW (inject_aw.py:508) and the read fade-in `fade = min(1, step/500)` (line 549). Note 236-5: REG d <= 4 straight from askname never formed pass 2 (0/2 at 16k). Note 258-5: the merged stage aw_plaw2 -> REG + §256 distract, 16k (bridge_s1) works: d 6 0.949, d 7 0.789.

Covered: **LC5 BRIDGE2** (aw_plaw2 -> one 16k stage with REG d <= 4 + §257 mix + hold: 28k after askname instead of 68k); **LC1 ONE-RUN** (one 36k run from the askname trunk with step schedules; single vs restarted cosine); **LC4 STATE-DISTILL** (bun_s1 as a frozen teacher for a fresh small student, and the same loss as the GPT-2 medium pass-2 route after Note 258-7/8); and the **schedule flags** all three need.

Why: §268 LINEAGE-SEED costs ~$5 and ~12 h per independent lineage, and every C2 claim needs >= 3 seeds (Note 269-2). A 2-stage or 1-run recipe makes seeds ~2.5x cheaper and removes four restart confounds. Formation is invisible at 2k (Note 269-1 FD1; aw_plaw2 formed at 7-8k, Note 236-5), so formation gates read at >= 4k and bind at the end of their phase.

**Order:** BRIDGE2 arm B first. ONE-RUN's phase layout is fixed after BRIDGE2 reads (if BRIDGE2-OK, phases 2-4 collapse into one and ONE-RUN becomes 28k with one boundary). DISTILL-S runs in parallel; DISTILL-M only if §258 S2b reads NO-SEED.

## 2. Exact design

### 2.1 Schedule flags (shared build; boxes/inject_aw.py)

Schedules are piecewise-constant in `step` and derived from the step alone, so `--resume` needs no new ckpt fields (lines 514-528). Every boundary sits at `k * ckpt_every + 1`, so a resume from the ckpt at `k * ckpt_every` replays it exactly.

| flag | semantics | touch point |
|---|---|---|
| `--d_max_sched "1:2,12001:4"` | `dmax_at(step)`; the draw becomes `rng_d.integers(1, dmax_at(step) + 1)` | line 552 |
| `--distract_sched "1:0,12001:1.0"` | `p_at(step)` replaces `a.distract_p` in the per-item coin `rng((seed, SEED_DISTRACT_COIN, step, k)).random() < p` | line 325 |
| `--mix_start S` | `IR.distract_mix` (MIXT) from step S; `IR.distract` (§256 rule) before | line 326 |
| `--lr_restarts "12001,28001"` | at a listed step: `opt = make_opt(params, a.lr)` (fresh AdamW) and `lr_at(a.lr, gr, step - r_k + 1, len_k)` with len_k the segment length (warmup 100 + a fresh cosine) | lines 545-546, 508 |
| hold | no new flag: `inc_on = hold_inc > 0 and step >= hold_start * sc` is already a step schedule | line 558 |

Rules: `a.d_max` and `a.distract_p` are the schedule maxima, so EXCL / INS_TEXTS / WS_ITEMS / mem_check and the asserts at lines 306-313 and 449-451 are built for the final phase; `if a.distract_p:` (line 319) is taken throughout and `p_at = 0` never inserts (the items are the same `stream.draw(d)` as the plain branch). With every flag absent, `dmax_at ≡ a.d_max`, `p_at ≡ a.distract_p`, `mix_start = 1`, no restarts: today's call sequence and rng consumption. Each 1k-bin record (line 612) gains `sched: {d_max_now, distract_p_now, mix_on, lr_segment, lr_now}`; the scorer VOIDs a run whose logged schedule disagrees with its args.

Disclosed: (i) a restart replays the lineage's LR + AdamW reset but not its read fade-in re-ramp (`fade` is absolute-step; re-fading a formed model's read is a perturbation the lineage carried by accident, not built); (ii) a changed `integers` bound can shift the later d-sequence (dseq / ihash / exposure_by_d are logged per bin).

### 2.2 LC5 BRIDGE2 (2-stage recipe)

Init: aw_plaw2_s1_init/model.pt via `--init_model` (sha asserted, lines 292-295; g_add calibration skipped). 16k, d_max 4, bs 64, micro 2, lr 1e-4, REG law; args_ref = dis_s1's frozen args as bridge_s1 used (bridge_s1 differs from dis_s1 only in init_model and the absent mix), plus:

- **Arm B (staged-in-one; first):** `--distract_p 1.0 --distract_mix s256,chain,givop --mix_start 8001 --hold_inc 1.6 --hold_start 8000 --hold_log --keep_final_ckpt` (§256 inserts 1-8k; the §257 mix and the penalty 8-16k: the dch + bun order inside one stage).
- **Arm A (all-on; second, only if B is not OK):** the same with `--mix_start 1 --hold_start 500` (bun's hold_start on a model without passes 3-4).

Comparators: bun_s1 / s2 (the §259 scorecard); the norm guard's no-penalty control is arm B's own 1-8k window (`hold.norm_guard_by_d` per bin before and after onset) and bridge_s1 (same init, no mix, no hold; disclosed mix mismatch). Seed 1 in stage 1; seeds 2-3 of the chosen arm in stage 2 (same aw_plaw2_s1 init, as bun_s1 / s2 share rcb; §268's aw_plaw2_s3 gives the first independent init). Budget rule (§13.0 rule 3): a C2 FAIL at 16k gets one 24k re-run (fresh cosine, ~$2.7) before any redesign; a NO-FORM gets none.

### 2.3 LC1 ONE-RUN (36k from the askname trunk)

Init: `--init_trunk s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe` (as aw_plaw2: fresh DeltaMem, calibration runs). `--steps 36000 --d_max 4 --d_max_sched "1:2,12001:4" --distract_p 1.0 --distract_sched "1:0,12001:1.0" --distract_mix s256,chain,givop --mix_start 20001 --hold_inc 1.6 --hold_start 28000 --hold_log --ckpt_every 2000 --keep_final_ckpt`. Phases: P-2 (1-12k, d <= 2) -> REG d <= 4 + §256 inserts (12-20k) -> §257 mix (20-28k) -> + hold (28-36k). `aux_fade 48000,48000` keeps λ_aux = 1 throughout (lam_at, line 299-302).

- **Arm A:** one warmup + cosine over 36k (lr at 12k / 20k / 28k = 0.75 / 0.41 / 0.12 x 1e-4; the hold phase runs at a low LR).
- **Arm B:** `--lr_restarts "12001,20001,28001"` (the lineage's restarts in one process, without ckpt reloads, host changes or fade re-ramps).

P-2 cannot be shortened: onset is 4-7k (Note 269-1). Phase gates read at 12k (FORM-2) and 20k (FORM-4); a NO-FORM at 12k is a NO-SEED result (as in §268), not a void. Both arms draw identical items (same schedule and rng streams), so A vs B differs only in LR / AdamW.

### 2.4 LC4 STATE-DISTILL

Teacher T = bun_s1, built by `build_aw` from bun_s1/args.json as a Namespace (`--distill_model boxes/runs_fsq/bun_s1/model.pt --distill_sha ...`; its init_trunk sha asserted, so askname's model.pt ships too), loaded strict, `requires_grad False`, eval mode, outside `opt.param_groups`. Per chunk the teacher runs the K passes under `torch.no_grad()` on the same `cue_inputs` (line 383) and keeps S^T_t (B, 12, 64, 64; deltamem.py:86) and the ':' hidden h^T_t (pass_aw, lines 47-56). Student terms, per pass t = 1..K:

- L_kl = τ² KL( softmax(z^T_t / τ) || softmax(z^S_t / τ) ), z = `colon_logits(h, pos)[:, NUM_IDS]` (the 11 number tokens, the WS readout, lines 61-63), τ = 2; dimension-free, so it runs on medium.
- L_state = mean_t ||S^S_t - S^T_t||² / ||S^T_t||² (same-D students; the FitNets hint).
- L = CE_ans + λ_aux L_aux + λ_kl L_kl + λ_st L_state; λ_kl = 1, λ_st = 1 (small), λ_st = 0 (medium: 8 + 8 heads, build_aw line 118).

Plainly: the teacher's readouts carry per-pass targets of aux origin. This compresses replication (seeds, ports), not discovery, and is **not C4**.

- **Arm S (small student):** `--init_trunk s233_askname_s1 --d_max 4 --distract_p 1.0 --distract_mix s256,chain,givop --hold_inc 1.6 --hold_start 8000 --hold_log`, 16k. Controls: §236 REG-CHAIN (the route minus inserts, hold and teacher: 0/2 at 16k) and **S-ctrl** = arm S with λ_kl = λ_st = 0, run only if S forms.
- **Arm M (medium, the port route):** `--base gpt2-medium --init_trunk askm_lr1e4` (§258 S1's lr 1e-4 arm), the P-2 stage exactly (aw_plaw2's args: d <= 2, 12k, K 8, lr 1e-4) + λ_kl = 1, λ_st = 0. Gate = §268's aw stage gate. Teacher targets on d <= 2 are exact (bun d 1-4 1.000, Note 259-4).

Implementation: in `loss_reg` (lines 383-404) a new `elif DIST:` branch uses `m.pass_aw` (its `[0]` is `m.write`, lines 58-59) to keep h_c / pos_c; flags-off never enters it. The terms join `loss` beside the `cue_ans` term (line 418). Teacher cost ≈ K no-grad passes ≈ +35% step time on small, ~0.5 GB of weights.

## 3. Code touch points (read)

boxes/inject_aw.py: `InjectAW.pass_aw / write / colon_logits` 45-63; `build_aw` 109-126 (`--base`, heads = D/128); `ws_readout` 135-152; `hold_step` 159-200; argparse 211-246; setup 250-298 (`rng_d` 271, `init_model` 292-295); `lam_at` 299-302; INS_TEXTS / MIXT 305-313; `draw_items` 316-338 (coin 325, draw 326); `loss_reg` 365-425; mem_check 452-490; `make_opt` / resume 508-528; loop 544-566 (`lr_at` 546, fade 549, d draw 552, hold 556-559); 1k-bin record 612-645; ckpt 655-663. boxes/fsq120.py:731-744 `make_opt`, `lr_at`. boxes/deltamem.py:86-87 `zero_state`, 179-200 `write_seq`. boxes/insert_render.py:176, 191-194. theory/score_bu.py, score_ls.py headers (bars and field paths).

## 4. Gates (before any launch; G-D by Audit)

1. **Flags-off bitwise:** new tree with no schedule / distill flag vs the pre-change tree, `--stop_at 300`, on bun_s1's and aw_plaw2_s1's args, fp32, deterministic SDPA wrap: log.jsonl, ckpt_last.pt tensors and args.json (minus shas) byte-equal (the §262 / §265 form).
2. **Schedule identity:** `--d_max_sched "1:4" --distract_sched "1:1.0" --mix_start 1` reproduce the flags-off run bitwise.
3. **Restart identity:** `--lr_restarts "201"` at `--stop_at 400`: the logged lr at 201-400 equals fp64 `lr_at(step - 200, 200)`; a unit test asserts `opt.state == {}` after step 201's reset and that a resume from the ckpt at 200 replays it.
4. **Distill off = plain:** λ_kl = λ_st = 0 with the teacher loaded reproduces flags-off bitwise; `requires_grad False` and absence from `opt.param_groups` asserted.
5. **Gradflow:** a 20-step CPU smoke with CE_ans = aux = 0 and λ_kl = 1, then λ_st = 1 alone: non-zero grads on `mem.qkv`, `mem.gate` and trunk blocks >= `mem.layer`; zero on the teacher. Then the plain-path smoke without any flag (§265 (c)).
6. **mem_check** on the host at the final-phase shape (d_max 4, K 8, hold on, teacher loaded), `--mem_budget 21`, 1 GiB headroom, `--micro 2` pinned.
7. **Step-0 reproduction** for `--init_model` arms (§237); host shas, frozen args_ref, WALLW guards as in §268; every set built on the host.
8. No new parameters, so no zero-init usage monitor; a width-mismatch hint regressor is not built.

## 5. Pre-registration table (per seed; Wilson 95% CIs; NEAR = bar inside the CI; NEAR-SEED = |acc - bar| < 0.09 on C2 rows)

| criterion | field | bar | applies to |
|---|---|---|---|
| RETAIN | eval_acc by_d[1-4].acc | >= 0.98 | all |
| NOT-DECODER | eval_acc by_d['4'].acc_kprime['2'] | <= 0.20 | all |
| SWAP4 / LP5-D4 / CUE-ONLY | eval_swap_kd, eval_lp5h, eval_cue_only (score_bu rules) | pass | all |
| FORM-2 (mechanism; phase gate) | log 1k-bin ws_readout['2']['2'] | >= 0.90 by the end of phase 1 (12k) and in every later bin (no drift); BRIDGE2: in every bin from 1k (a present code retained) | ONE-RUN, DISTILL-S/M, BRIDGE2 |
| FORM-4 (mechanism at the trained depth) | eval_ws by_d['4'] passes 1-4; ONE-RUN also log ws_readout['4'] at 20k | >= 0.90 | all small arms |
| C2-6 / C2-7 | eval_acc d 6 / d 7 | >= mean(bun_s1, s2) - 0.05 (0.914 / 0.805 from Note 259-4) | all; binding only with >= 3 seeds |
| WS-DEEP | eval_ws_deep by_d['6'] passes 1-6 | >= 0.90 | all |
| C6 | eval_long_k128 d 1-6: acc[64], acc[128] | >= acc[8] - 0.02; norm guard vs the no-penalty window / control (score_lh, GAMED iff > 1.25) | all with hold |
| CONTENT | eval_insert_ho H-CHAINSRC-OP acc8 / min(by_pos) | >= 0.85 / >= 0.80 | all small arms |
| P2-GATE (medium) | eval_acc d 2 and eval_ws by_d['2']['2'] | >= 0.95 and >= 0.90 at 12k | DISTILL-M |
| STABLE | score_ao | pass | all |

Descriptive: d 8-9, the K' curve, exposure_by_d and gn_preclip_by_d per phase (the Note 236-5 swamp hypothesis), norm_guard_by_d before / after hold onset, distract_types per bin, onset t* per arm, S vs S-ctrl margins.

Labels: **BRIDGE2-OK / ONE-RUN-OK / DISTILL-OK** = every row passes (C2 rows with 3 seeds, each reported as its margin vs the bar plus the seed mean ± SD); **-PARTIAL** = RETAIN + FORM pass, listed rows fail; **NO-FORM** = FORM-2 or FORM-4 fails at its phase gate; **FAIL**. Arm A vs B is reported as per-seed margin differences; with sigma_seed 0.09 and 3 seeds per arm (SE ~0.07) "restarts matter" is claimed only above a 0.15 mean difference at d 6 or d 7. The binding question is whether each arm reaches the bars.

**OC (priors):** BRIDGE2-OK arm B 0.50, arm A 0.35 (the penalty from step 500 on a model without passes 3-4 may teach "freeze after pass 3": NO-FORM in A 0.2). ONE-RUN: FORM-2 by 12k 0.7; ONE-RUN-OK arm A 0.3, arm B 0.45. DISTILL-S FORM-4 0.55 vs the §236 0/2 control; DISTILL-OK 0.35. DISTILL-M P2-GATE 0.4.

**Censoring:** crash or host fault = VOID-RERUN once; NO-FORM at a phase gate is a result (the run continues for the descriptive reads; DISTILL-M stops at 12k); a logged schedule disagreeing with the args = VOID-BUILD; a failed step-0 reproduction = VOID; NEAR-SEED rows are claimed neither way; no seed rescues. A run cut by the guard is CENSORED-BUDGET, not FAIL.

**Shortcut-exclusion arms:** NOT-DECODER + SWAP4 + LP5-D4 on every arm (a student matching the teacher's logits from text alone fails SWAP4); the norm guard against a hold by norm inflation; S-ctrl exposes the teacher's contribution; CUE-ONLY.

## 6. Scorer sketch (theory/score_lc.py)

Reads per run: eval_acc (by_d[d].acc, acc_kprime), eval_swap_kd (by_d['4'].swap_mid['4'].follow_U), eval_lp5h (by_d['4']), eval_cue_only, eval_ws (by_d[d][t]), eval_ws_deep (by_d['6']), eval_long_k128 (acc[8/64/128]), eval_insert_ho (H-CHAINSRC-OP acc8, by_pos), log.jsonl 1k-bin records (ws_readout, hold.norm_guard_by_d, gn_preclip_by_d, distract_types, exposure_by_d, `sched`). Refs: bun_s1 / s2 eval_acc; bridge_s1 and bunc_s* (norm-guard controls). Wilson CIs via score_ls's helper; NEAR / NEAR-SEED; a per-seed margin table. Fixtures (12/12 before launch): OK, PARTIAL-C6, NO-FORM, FAIL-RETAIN, VOID-BUILD, VOID-MISSING, NEAR-SEED, restart-identity, GAMED, CENSORED-BUDGET, P2-GATE pass / fail.

## 7. Cost (RTX 3090 ~$0.2/h; anchors: bridge_s1 16k $1.41, bun 8k $0.99, p2m 12k medium $3.03 on A6000; a 12k d <= 2 phase ≈ $0.8, estimated)

| stage | cells | $ |
|---|---|---|
| BRIDGE2 stage 1 | arm B seed 1 (16k, hold 8-16k ≈ $1.6) + evals; arm A only if B is not OK (+$1.8) | 1.8-3.8 |
| BRIDGE2 stage 2 | chosen arm, seeds 2-3 | ~3.6 |
| BRIDGE2 budget arm | one 24k re-run on a C2 FAIL | ~2.7 |
| ONE-RUN stage 1 | arms A, B to 20k (reads at 12k / 20k) | ~3.2 |
| ONE-RUN stage 2 | passing arms to 36k (+$1.7 each) | ~3.4 |
| ONE-RUN stage 3 | +2 seeds of the chosen arm (36k ≈ $3.3 each) | ~6.6 |
| DISTILL-S stage 1 | arm S (16k, +35% teacher ≈ $2.3); S-ctrl only if S forms (+$1.8) | 2.3-4.1 |
| DISTILL-S stage 2 | seeds 2-3 of S | ~4.6 |
| DISTILL-M | 12k medium (A6000) + small teacher | ~3.5 |

Three registrations: BRIDGE2 ~$8-10 (guard $12), ONE-RUN ~$13 (guard $16), DISTILL ~$11-12 (guard $15). Each first stage <= $4; a stage 2 runs only after Audit scores stage 1.

## 8. RED-TEAM

Disclosure: the required red-team sub-agent could not be spawned (the session's concurrent sub-agent limit of 20 was reached at both attempts, with the instruction not to retry). The review below was done by the lead against the code and notes; the caller should re-run an independent red-team pass before registration.

1. **Exposure confound.** bun saw 40k d <= 4 steps after aw; BRIDGE2 sees 16k, so a C2 miss can be budget, not recipe. Mitigation: the pre-stated 24k budget arm before any redesign; the label stays "16k recipe".
2. **Seed-vs-recipe.** Stage-1 cells are one seed; bun_s1 / s2 spread 0.016 at d 6 and 0.126 at d 8 (Note 259-4). C2 rows bind only at 3 seeds; NEAR-SEED; arm comparisons descriptive below 0.15.
3. **Hold from step 500 on an unformed model (arm A).** The penalty acts on passes t >= d + 3 (hold_step, lines 159-170): passes 4-8 on d 1 items, before passes 3-4 exist. It can teach a global freeze; hence arm B first and A's NO-FORM prior.
4. **FORM-2 "every bin from 4k" was wrong in the first draft** (aw_plaw2 formed at 7-8k); fixed to a 12k phase gate plus no drift afterwards.
5. **rng stream shift.** A changed `integers` bound can consume the bit generator differently; the post-12k d-sequence is logged (dseq, ihash, exposure_by_d) and shared by arms A / B; nothing binding matches a reference d-sequence.
6. **Restart ≠ stage restart.** A stage boundary also reloads a ckpt, may change GPU class (Note 258-5) and re-ramps `fade`. Arm B isolates LR + AdamW. If B passes and A fails, the reading is "a restart of LR + moments suffices", not "the lineage's restarts were necessary".
7. **Teacher leakage.** L_kl / L_state are per-pass supervision by proxy (the Note 266-1 ruling): no C4 wording. Medium's decoder shortcut (Note 264-1) is caught only by the WS pass-2 mechanism gate inside P2-GATE.
8. **Build risks checked.** (a) line 312's `distract_p > 0` assert holds because `a.distract_p` is the schedule max; (b) `--resume` loads `opt` (line 515) and boundaries at `k * ckpt_every + 1` replay the reset exactly; (c) `hold_log` runs no_grad stats from step 1 (HOLD, line 289) and changes no gradient before hold_start; (d) the teacher under the student's bf16 autocast differs slightly from bun's fp32 eval, which soft targets tolerate; (e) `mem_train` calls `loss_items`, so the teacher is inside the measured peak; (f) the 24k budget arm is a fresh run, since a `--steps` extension recomputes the cosine.
9. **Cheaper falsifiers.** BRIDGE2 arm B alone ($1.8) answers the highest-value question and fixes ONE-RUN's layout and DISTILL-S's recipe. S-ctrl is conditional; §236 gives the no-insert, no-teacher control for free. The 12k ONE-RUN read ($1.6 for both arms) is the cheapest test of FORM-2 under a schedule.
10. **Stats.** Bars and seed rules follow Notes 269-1 / 269-2. "Restarts matter" is underpowered at 3 seeds per arm (SE ~0.07): descriptive below 0.15.
11. **Added after review:** CENSORED-BUDGET; the in-run norm-guard window; conditional S-ctrl; the fade disclosure; the 24k budget arm; no zero-init pathway.

## 9. Verified arXiv ids (abs pages fetched 2026-11-03; titles checked)

- 1608.03983 Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts (arm B).
- 1904.03626 Hacohen & Weinshall, On The Power of Curriculum Learning in Training Deep Networks (the d_max ramp).
- 2412.06769 Hao et al., Training Large Language Models to Reason in a Continuous Latent Space (Coconut's staged curriculum).
- 2410.05464 Panigrahi et al., Progressive distillation induces an implicit curriculum (a progressive-teacher arm, DISTILL stage 2).
- 2502.21074 Shen et al., CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation (hidden-state distillation at the answer position).
- 1503.02531 Hinton et al., Distilling the Knowledge in a Neural Network (τ-softened KL).
- 1805.04770 Furlanello et al., Born Again Neural Networks (same-architecture student).
- 1412.6550 Romero et al., FitNets: Hints for Thin Deep Nets (the state hint).
- 2604.04902 Dilgren et al., Are Latent Reasoning Models Easily Interpretable? (latent tokens often unnecessary: the decoder-shortcut gates stay binding).
- 1711.02257 Chen et al., GradNorm (listed under LC2 by wave 3; loss balancing, not self-paced weighting; LC2 not designed here).
- 1603.06560 Li et al., Hyperband (HX1's halving; unused since FD1 found no 2k marker).
