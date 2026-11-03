# 16. RL-HALT (RL2), RL-MECH-GUARD (RL3) and RL-DEPTH (RL1): registration-ready design

Wave-4 lead design, 2026-11-03. Backlog IDs: **RL2 (H8')**, **RL3 (X8)**, **RL1 (K13)**; absorbs **H8 ANSWERABLE-NOW** (as the reward readout, not the halt) and **K1b EM-SAMPLED** (RL1 on the §267 model). Read-only; nothing computed from stored outputs.

## 1. Goal

C5 on bun is a training-free rule (answer unchanged for 3 passes, Note 259-11) that halts at d - 1 with ~0.98 determinism (Note 264-4) and costs +3 passes. §264 will say whether that halt is the model's own or clocked by the given plan (OC 0.95 ER PLAN after Note 264-3). RL2 makes the halt a **model output** trained from the answer only (a STOP action per pass, reward = correct-at-stop - lambda * passes) and tests whether it is content-based with §264's schedules (ER / LFI / CY) plus one new schedule (PT) that separates "the cue names the target" from "the target is answered". RL3 is the guard every RL cell carries. RL1 asks whether answer-only RL lifts d 7-8 (C2) on a stochastic latent, gated by a pass@64 pre-check (RLVR sharpens what the base can already sample, 2504.13837).

## 2. Design

### 2.1 RL2 RL-HALT

**Policy.** After pass t the write position ':' has hidden h_t (post ln_f, the row `colon_logits` reads; inject_aw.py:61-63). A STOP head gives p_t = sigmoid(w . h_t + b), t = 1..K_max, K_max = 12. The trajectory stops at the first t with STOP = 1, else at K_max (censored; reward uses S_12). P(stop = t) = p_t prod_{s<t} (1 - p_s).

**Reward.** R(t) = 1[R_phi(S_t, q) = v_d] - lambda * t, lambda = 0.03 (a wrong answer costs 1; running to the cap costs 0.36; the marginal pass is cheaper than any accuracy it buys above 0.03). R_phi is the **S-only reward readout** (2.2); the training decoder (text + S) never enters the reward, so "stop at d - 1 and let the decoder do the last hop" (Note 264-4 DECODER-LAST-HOP) earns nothing.

**Objective (exact, no sampling).** The loop is deterministic given its weights, so the 8 "rollouts" of an RLOO group would share every state and differ only in the stop time. The expected reward is therefore computed exactly:

J = sum_{t=1}^{12} P(stop = t) R(t),   L_halt = -J + beta * KL(pi_stop || pi_0)

with pi_0 the step-0 head (beta 0.01, a Dr.GRPO-style length-bias-free form, 2503.20783; no per-token length normalisation). RLOO (group 8, leave-one-out baseline, 2402.14740) is kept only as the sampled fallback for stage H2 if the exact sum is too costly with the loop unfrozen (it is not at K_max 12). Stage **H1** trains the head alone on the frozen loop; stage **H2** unfreezes the loop with the plain loss (answer CE + aux CE + hold penalty, exactly bun's `loss_reg`) kept and L_halt added with weight 1, under the RL3 guard. In H2 the gradient reaches the loop only through h_t -> p_t: the loop is not trained to change answers, only to make its halting state readable.

**Zero-init rule.** b is initialised to logit(0.05) (not 0), w ~ N(0, 0.02). Usage monitor: mean p_t per pass per 1k bin and P(stop = t) histogram in log.jsonl. A non-zero-init arm (b = logit 0.5) runs as one extra cell (rule 8).

**Cue schedules.** Training items use the STD cue S_min(t, d) only (bun's cue). Eval runs the policy under STD, **ER** (S_min(t, d - 2)), **LFI** (S_t for t <= d, then the inserts' entities; Note 264-3 items, seed 26420), **CY** (S_((t-1) mod d) + 1), and the new **PT PREMATURE-TARGET**: S_d, S_1, S_2, ..., S_d (the target is cued at pass 1 before its parent exists, then the plan runs). Under PT a "cue names the asked entity" policy stops at t = 1 with a pre-target answer; a content policy stops at t = d + 1. The token swap is the §263 / §264 `CUE_NAME_OFF` mechanism (6 before ':', original token asserted; inject_aw.py:91-92).

**Composition with §264.** If §264 reads **C5 PLAN-CLOCKED**, RL2's binding C5 claim needs the schedules in 5.1 (ER COMP-HALT, LFI CONTENT, PT CONTENT); STD-only halting would be a re-description of the plan. If §264 reads **COMP**, the schedules stay binding as the shortcut-exclusion arm, and the claim is efficiency (fewer passes than r = 3 at equal accuracy). A third stage **H-PF** (only on H1 PASS): the same head on **gb_s1** (§260 generic cue; the loop follows reachability and does not stop at S_d, H-EXTEND 0.08, Note 260-5). There the cue carries no plan, so a halt at the target cannot be cue-read; this is the C5 claim a plan-free setting needs (backlog Lane C).

**Composition with §267.** On the EM model every pass's ':' output is an 11-way categorical, so log pi(path, stop) = sum_t log p_t(v_t) + log P(stop = t) is exact. RL2 is not run on EM here (stage A has d_max 3 and no hold); the head and schedules port unchanged once stage B exists.

### 2.2 R_phi, the S-only reward readout (stage H0)

A copy of bun's trunk + DeltaMem, frozen, with a **question-only decoder input** dec_S = [EOT] + enc(" " + QUESTION(u, S_d)) + SFX run through `fwd_read` (deltas5.py:91-97: the block-6 read on S, no write; `dec_logits`, inject_ow.py:86-90, with it["dec"] replaced). No op sentence is in its input, so it can only answer from S. Head: a new 12-way linear on the ':' hidden (11 values + NOT-YET), trained by CE on states produced by frozen bun at t = 1..12 under STD and PT schedules on 6,000 fresh REG items d 1-4 (the §248 HALT-HEAD fit stream, seed 24750 + 2, excl = the eval sets), labels: v_d if t >= t*(schedule) else NOT-YET, t* = d (STD) / d + 1 (PT). Only the new head and ln_f-scale train (the trunk stays bun's so the state is read as bun reads it). 1,000 steps on cached states.

**REWARD-VALID gate (binding for H1 to start):** on TEST2 d 1-4 at t = 12, acc(R_phi) >= 0.95; on t < t* states, P(NOT-YET) >= 0.90; d 5-6 at t = 12 >= acc_k8(bun) - 0.05 (descriptive if lower: R_phi then under-rewards deep items and H1 halts late on them, which is conservative).

### 2.3 RL3 RL-MECH-GUARD (every 200 steps, any cell with an unfrozen loop)

On the first 200 TEST2 items d 1-4 (the WS_ITEMS block, inject_aw.py:451), fp32, no_grad:
1. **FROZEN-READOUT:** the pre-RL R_phi (frozen) on the current model's S_12: acc >= 0.95 at d 1-4. A drop means the state code moved under RL.
2. **WS:** `ws_readout` (inject_aw.py:134-151) per pass vs the step-0 record: every (d, t <= d) cell within 0.03.
3. **SWAP-V:** the §267 `swap_v` form on bun = `part_swap_kd` SWAP-MID at d 3, t 1 (inject_aw_eval.py:391-435): follow >= step-0 follow - 0.05.
4. **KL:** mean KL(pi_stop,now || pi_stop,0) <= 2.0 (RL2) / mean KL of the ':' categorical <= 0.5 (RL1).
Two consecutive failures of 1-3 stop the cell (censor label **MECH-DRIFT**, the step named); the last passing ckpt is the cell's model. Arms (run beside every RL1 cell, one each on RL2 H2): **RANDOM-REWARD** (reward replaced by a seeded Bernoulli(0.5) per rollout, same pipeline; any gain is a format or optimiser effect) and **1%-MASK** (a seeded random 1 % of parameters trainable, 2602.01599; a match with full RL says RL is sharpening, 2504.13837, 2505.11711, 2511.08567).

### 2.4 RL1 RL-DEPTH

**Stochastic latent.** bun's loop is deterministic, so a group of rollouts needs a noise source. Two policies:
- **bun:** Dropout-GRPO (2606.10184): one dropout mask on the write input u (deltamem.py:185, after `mem.ln_w`) with p 0.1, drawn once per rollout and held for all passes; the reward uses the greedy decoder answer under that mask; the policy term is log p_dec(answer_g | mask_g).
- **EM (§267 stage B model, if it exists):** the per-pass categorical is the policy; v_1..v_{d-1} are sampled, log pi = sum_t log p_t(v_t), exact. This is K1b EM-SAMPLED.

**pass@64 pre-check ($0.1, binding GO / PARK).** 300 fresh items per d at d 6-8, 64 masks each (greedy answer per mask). pass@1 = the no-mask answer. GO iff pass@64 - pass@1 >= 0.10 at d 8 (bun d 8 ~0.5, so there is room); PARK otherwise: nothing to sharpen. Reported: pass@k curve k in {1, 4, 16, 64}; the within-item answer entropy.

**Objective.** RLOO, group G = 8 per item, items d 5-8 (gold answers only, aux CE off, hold penalty kept so C6 is not traded away), A_g = R_g - mean_{h != g} R_h, L = -(1/G) sum_g A_g log p_dec(answer_g | mask_g) + beta KL(p_dec || p_dec,0), beta 0.02, lr 2e-5, 1,000 steps, bs 32 x 8. RL3 every 200 steps. Cells: rl1_s1 (bun_s1), rl1_rr_s1 (RANDOM-REWARD), rl1_m1_s1 (1%-MASK). Stage 2 (on GAIN only): bun_s2 + §268's fresh-lineage seed, so the C2 claim has >= 3 seeds on per-seed margins (Note 269-1).

## 3. Code touch points (read)

- boxes/inject_aw.py: `InjectAW.pass_aw` 47-56 (returns S, h, pos; the STOP head reads `Fn.layer_norm(h)[ar, pos]`, the input of `colon_logits` 61-63); `ws_readout` 134-151 (the frozen-loop rollout template); `hold_step` 159-199 (an extra loss with its own graph, backward'd after the plain backward); `loss_reg` 361-419 (pass loop 372-404; the detached `Ssv` 371/393 give S_1..S_K; `dec_logits` 404); args 211-249 (new flags `--rl_halt`, `--rl_lambda`, `--rl_kmax`, `--rl_freeze_loop`, `--rphi`, `--rl_depth`, `--drop_fixed`, `--rl_group`, `--rl_guard_every`, `--rl_random_reward`, `--rl_mask_frac`); step loop 544-608 (the RL step goes at 556-560 beside `hold_step`; the 1k record 609-645 takes the halt histogram and guard fields); `--init_model` 293-296.
- boxes/inject_ow.py: `dec_logits` 86-90, `aux_logits` 92-103 (R_phi reuses `fwd_read` with the dec_S token list).
- boxes/deltas5.py: `fwd_read` 91-97.
- boxes/deltamem.py: `write_seq` 179-215 (the Dropout-GRPO mask multiplies `aw` after `mem.ln_w`, 185; FAST_ADD / CHUNK_DELTA untouched).
- boxes/inject_st.py: `cue_inputs` 70-80 (the schedules build `think_cues` per pass; one padding asserted).
- boxes/inject_aw_eval.py: `passes` 55-83 (`seqs_of_t` for ER / LFI / CY / PT; the halt part cannot use it because the head needs h, so a new `part_rlhalt` follows `ws_readout`'s loop); `part_halt` 812-828 (the output format {acc_k8, acc_halt, mean_khalt, khalt_hist, never_halt_n, items}); `part_halthead` 884-918 (the fit stream and the fit-set / TEST2 disjointness assert reused by R_phi); `part_swap_kd` 391-435.
- theory/score_sc.py (schedule definitions and labels; §264 layout `items[sch][d]`), theory/score_c5c.py (the r = 3 rule and the C5 IN-RANGE / EXTRAP bars), theory/score_hx.py (censoring text).
- Stored inputs: boxes/runs_fsq/bun_s1 (args.json: 8k, bs 64, lr 1e-4, K 8, d_max 4, aux_step 1.0, hold_inc 1.6, distract_mix s256,chain,givop, micro 2; train_summary.json train_sec 13230 on a 3090 = 1.65 s/step), bun_s2, gb_s1 (H-PF).

## 4. Gates

1. **p = 0 equivalence:** `--rl_halt 0 --rl_depth 0 --drop_fixed 0` on a 20-step CPU fp32 smoke from bun_s1: log.jsonl and model.pt sha byte-identical to the unmodified module (GPU bitwise gates need the det wrap; CPU avoids it).
2. **Exact-expectation check (fp64, 4 items):** J equals the enumeration over the 12 stop times; P(stop) sums to 1 with the cap term.
3. **gradflow_check** (read from the decorators down): in H1 only the head gets a gradient (trunk / DeltaMem grads asserted None); in H2 the loop's grad from L_halt is non-zero only through h_t; in RL1 the policy term reaches the write projections and the trunk, and `aux_logits` is asserted off.
4. **mem_check** at d 4 (bun) with K_max 12 and the R_phi pass inside the peak, micro pinned (the RL1 group of 8 at bs 32 = 256 rollouts per step: micro 8).
5. **200-step smoke at full data scale** on the host (3090), then the §256-style guard .out check before the run continues.
6. **REWARD-VALID** (2.2) before H1; **pass@64 GO** before RL1.
7. Scorers committed with fixtures before launch; Wilson CIs + NEAR on every binding cell; fresh draws disjoint from TEST2, fit sets and EXCL (asserted, Note 259-11's rule).

## 5. Pre-registration table

### 5.1 RL2 binding (per seed, fresh REG draw n 1,000 per d, K_max 12)

| Criterion | Rule | Threshold |
|---|---|---|
| HALT-IN-RANGE (d 1-4, STD) | acc_halt >= acc_k8 - 0.02 and mean khalt <= d + 0.5 | both |
| HALT-EXTRAP (d 5-7, STD) | acc_halt >= acc_k8 - 0.02 each d; mean khalt <= d + 0.5; khalt increasing over d 1-7 | all |
| EFFICIENT | mean (passes_run_r3 - khalt_RL) over d 1-7 >= 1.5 at equal accuracy (acc_halt within 0.02 of the r = 3 rule's acc_halt) | >= 1.5 |
| PT CONTENT (d 3-6) | P(khalt <= 2) <= 0.10 and P(khalt in [d, d + 2]) >= 0.80 | both; CUE-MATCH if P(khalt <= 2) >= 0.5 |
| ER COMP-HALT (d 4-6, r = d - 2) | P(khalt <= r + 1) <= 0.20 | PLAN-HALT if >= 0.8 (Note 264-3: d 3 reported only) |
| LFI CONTENT (d 3-6) | mean khalt <= d + 1 and acc_halt >= acc24(STDI) - 0.02 | CUE-CHANGE if P(khalt = 12) >= 0.5 |
| CY | acc_halt >= acc(d) - 0.02; khalt <= d + 1 | descriptive |
| USAGE | mean p_t at t < d <= 0.2 and at t >= d + 1 >= 0.5 by the end; the non-zero-init arm reaches the same label | monitor |
| VALIDITY | STD acc at K' 8 within 0.015 of bun's eval_acc; REWARD-VALID passed | else VOID |

**Label OWN-HALT** = HALT-IN-RANGE + HALT-EXTRAP + PT CONTENT + ER COMP-HALT + LFI CONTENT, both seeds; **PLAN-HALT** if PT CUE-MATCH or ER PLAN-HALT in both; else MIXED. C5 wording on OWN-HALT: "met, scoped: a learned stop trained from the answer only, content-based under premature / early-repeat / late-fresh cues; plan given in the cue". H-PF on gb_s1 adds "plan-free" if HALT-IN-RANGE + H-EXTEND (halt at the target) >= 0.8 hold there.

### 5.2 RL3 (per 200-step check): FROZEN-READOUT >= 0.95; WS within 0.03; SWAP-MID follow >= step-0 - 0.05; KL bounds. RANDOM-REWARD arm: its change in the primary metric must be <= 0.02, else the main cell's gain is labelled FORMAT-EFFECT. 1%-MASK: descriptive (SHARPEN if within 0.03 of full RL).

### 5.3 RL1 (stage 1, bun_s1): GAIN if d 8 acc >= bun_s1 d 8 + 0.10 and d 7 >= bun - 0.02 and d 1-4 >= 0.98 and every RL3 check held; NULL otherwise. Stage 2 C2 claim: three seeds, per-seed margins all >= 0.05 at d 8, K'-curve reported (K' 8 / 12 / 16).

### 5.4 OC (outcome priors)

- RL2 H1 OWN-HALT 0.35; PLAN-HALT 0.45 (the state may carry a pass marker rather than "answered", Note 264-2 PC 0.35; the cue-match feature is the cheapest); MIXED 0.2. If §264 reads COMP, OWN-HALT 0.5.
- REWARD-VALID 0.8 (the aux head already reads v_t from S; the question-only decoder input is new).
- H-PF on gb_s1 HALT-IN-RANGE 0.3 (the loop overwrites past S_d; a halt at d must beat the drift).
- RL1 pass@64 GO 0.5; GAIN | GO 0.3; RANDOM-REWARD within 0.02 0.7.

### 5.5 Censoring

- A never-halting item counts at K_max 12 with the S_12 answer (acc_halt) and 12 (khalt), as §247 / score_hx do; never_halt share reported beside it.
- MECH-DRIFT (RL3 twice in a row): the cell stops, its last passing ckpt is scored, the label carries DRIFT@step; it never counts as a PASS.
- VOID (validity) reruns once on a fresh host; a build failure VOID-RERUN once; OOM re-pins micro (no design change).
- H-PF and RL1 stage 2 run only on their GO conditions; a parked stage is reported as PARKED with its gate value.

## 6. Scorer sketch

`theory/score_rl.py` (fixtures before launch), reading per cell:
- `eval_rlhalt.json`: `items[sch][d] = {sig, ans, pred[0..12] (R_phi answer at every t, and the decoder answer at K' 8 for acc_k8), p_stop[1..12], khalt, halted}` for sch in STD / ER / LFI / STDI / CY / PT; `def`.
- `eval_stab_fresh.json` of the same draw (the r = 3 rule's passes_run for EFFICIENT; score_c5c's rule re-used).
- `eval_acc.json` by_d[d].acc (VALIDITY), `eval_rphi.json` {acc_by_d_t, notyet_prec} (REWARD-VALID).
- `log.jsonl` 1k bins: `rl: {p_stop_by_t, halt_hist, J, kl}`; `guard: {step, frozen_readout, ws_max_dev, swap_follow, kl, pass}` every 200 steps.
- RL1: `eval_acc.json` d 1-8 vs bun's, `eval_pass64.json` {d: {pass_at: {1, 4, 16, 64}, entropy}}, `eval_swap_kd.json`.
Labels: OWN-HALT / PLAN-HALT / MIXED / VOID; GAIN / NULL / PARK / FORMAT-EFFECT / MECH-DRIFT. Fixtures: one synthetic cell per label (10).

## 7. Cost (RTX 3090, ~$0.2/h; bun 1.65 s/step)

- H0 R_phi: state cache (6k items x 12 passes, forward only) + 1k head steps: ~20 min, $0.07.
- H1 (head only): 12 passes + 12 R_phi reads per item, bs 64, 2k steps at ~1.0 s: $0.12 per seed; x2 + the non-zero-init arm: $0.36.
- H2 (unfrozen, 1k steps at ~2.6 s with the guard): $0.15 per seed; x2 + RANDOM-REWARD: $0.45.
- Evals (6 schedules x d 3-7 x 1,000 items at K' 12 + the fresh stab draw): $0.17 per seed. H-PF on gb_s1: $0.3. **RL2 ~$1.3; guard $2.**
- RL1: pass@64 $0.1; stage 1 three cells (8 rollouts x bs 32, ~8 s/step, 1k steps) 3 x $0.45; stage 2 two seeds $0.9. **RL1 ~$2.4; guard $3.** First falsifying stage $0.1.

## 8. RED TEAM (separate sub-agent)

(See section 8 below; filled from the red-team agent's return.)

## 9. Verified arXiv ids (abs pages fetched, titles checked)

- 2504.13837 Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (pass@k: RL sharpens)
- 2402.14740 Back to Basics: Revisiting REINFORCE-Style Optimization (RLOO)
- 2503.20783 Understanding R1-Zero-Like Training (Dr.GRPO, length bias)
- 2402.03300 DeepSeekMath (GRPO)
- 2606.10184 Dropout-GRPO: Variational Stochasticity for Continuous Latent Reasoning (fixed dropout mask per rollout for deterministic latents)
- 2502.04463 Training Language Models to Reason Efficiently (RL with a compute penalty)
- 2503.04697 L1: Controlling How Long a Reasoning Model Thinks with RL
- 2505.13379 Thinkless: LLM Learns When to Think
- 2607.20519 Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts (post-hoc confidence readouts often match learned gates: the comparison our r = 3 rule vs RL2 makes)
- 2107.05407 PonderNet (expected-loss halting); 1603.08983 Adaptive Computation Time (Graves)
- 2511.08567 The Path Not Taken: RLVR Provably Learns Off the Principals; 2602.01599 The Multiple Ticket Hypothesis: Random Sparse Subnetworks Suffice for RLVR (the 1 % arm); 2505.11711 RL Finetunes Small Subnetworks in LLMs; 2509.04259 RL's Razor: Why Online RL Forgets Less (KL-closest solutions; the RL3 KL bound)
