# 17. DSR: DEEP-SUPERVISED RECURSION (HRM / TRM transplant) on the bun loop — registration-ready design (2026-11-03, wave 4, lead agent)

**Backlog IDs covered:** AP1 (A8) DEEP-SUPERVISED RECURSION; touches CA1 SG-WRITE (detach), S11 / M3 S-ONLY DECODER, M8, H6 / RL2 (learned halt), K8 (answer-only ladder). Lane B (C4), parallel to §267 EXACT-EM-11.

## 1. Goal

Test whether the HRM / TRM training recipe (final-answer CE after **every** pass, carried state **detached** between passes, n unsupervised inner micro-passes, a learned halt head q trained on its own correctness) forms a multi-pass loop in the bun architecture **from answers only**, with the memory left as it is (continuous DeltaMem, accumulating, no reset, no categorical write). It is a C4 route **without a designed state space**, the contrast to §267 (Z_11 categorical, reset memory, 1 + 11(d − 1) forwards per item). Here each pass costs one forward, as in bun.

Why it could beat the AO record (AO 1/4, AEX0 0/2, GATE-CURRIC parked, BLIND-DEC withdrawn, ACUE PARK; Notes 251-8..11): the measured failure is that the final-answer CE, back-propagated through the text-reading decoder and the 8-pass graph, pushes the pass-t ':' code off v_t on d > t items at onset (AW-SOURCE-DRIFT, Note 251-9 addendum). Two of the transplant's ingredients remove that gradient direction:

1. **S-only decoder** (the output head reads the memory and the question, not the op sentences; HRM / TRM have no text path at all). On Z_11 every op is a bijection, so given only v_t and no text, p(a | S_t) is near-uniform for t < d (an add_c step excludes a = v_t, a mul_c step excludes 0; nothing more). The per-pass CE on incomplete items therefore has no preferred direction for the value code; its only informative gradient is on d = t items (read v_t) and on passes t > d (keep it). The decoder cannot do the last hop from text (Note 264-1's V1+TEXT shortcut; bun's NOT-DECODER gate).
2. **Detach between passes (HRM's one-step gradient).** The CE at pass t trains pass t's forward only (its block-6 read of sg S_{t−1}, the trunk, the ':' write) and the decoder. No gradient reaches pass t's write from later passes, so no later-pass answer pressure can pull the code. The HRM bet is that decoder-readability of S_t is enough for pass t + 1 to read it: here that is structurally true, because the decoder and the think passes share the same read (`DM.read_fixed` at block 6, the same qkv / wo / gains; inject_aw.py:47-56, inject_ow.py:86-90).

Scope line if it passes (mirrors §267's): "C4 (no step value or prefix answer in any loss) with an undesigned continuous memory; plan given by the cue; askname init (one-hop questions over intermediate entities, the §110b-1 qualifier, as in AO_v1 §1 and §267); chain-order text". A GPT-2-base-init stage removes the qualifier (Stage C).

## 2. Design

Notation as bun (§259): text x (op sentences), question Q_d = "How many <u> does <S_d> have?", answer a ∈ Z_11, cue c_t = " " + QUESTION(u, S_min(t,d)) + " The answer is:" (plan given), K = 8, pass input x_t = [EOT] ⊕ enc(x ⊕ c_t) (tok_aw, inject_aw.py:76-93).

**Pass (unchanged, pass_aw):** h^(t) = Trunk(x_t; read_6(S_{t−1})); S_t = write_seq(h6^(t)[':'], h11^(t)[':'], S_{t−1}) (inject_aw.py:47-56; deltamem.py:179-230, gamma = 1).

**S-only decoder (new):** p_S = [EOT] ⊕ enc(" " + QUESTION(u, S_d)) ⊕ SFX (14-16 tokens; no op sentences; built in tok_aw as o["dec_s"], asserted to end in ':'). Dec_S(S) = LM logits at ':' of fwd_read(trunk, p_S, read_fixed(S)) — inject_ow.dec_logits (86-90) with seqs = dec_s. The 11 number tokens NUM_IDS are the answer classes (as dec_logits).

**Deep supervision (HRM §deep supervision; TRM Fig. 3):** for t = 1..K
- ℓ_t = CE(Dec_S(S_t), a); S_t ← sg(S_t) before pass t + 1;
- L_DS = (1/K) Σ_t w_t ℓ_t with w_t = 1 (HRM's uniform weighting). Passes t > d have a reachable target (a) and learn "already answered → keep it readable", i.e. the hold without using d. Passes t < d get the near-uniform target of §1.
- Backward per pass (ℓ_t.backward() right after its decode, graph freed), one AdamW step per batch (clip 1.0 as bun). Deviation from HRM / TRM, which step the optimizer per segment: kept so the LR schedule and bun's step count mean the same thing.
- **Inner micro-passes (TRM: T − 1 recursions without gradient, 1 with):** segment t = n writes of x_t under no_grad from sg S_{t−1}, then one write with gradient: S'_{t,0} = S_{t−1}; S'_{t,j} = write(S'_{t,j−1}, x_t) (no_grad, j = 1..n); S_t = write(S'_{t,n}, x_t). n = 0 in the main arm (the step cue is one hop per cue, so an inner micro-pass can only re-apply the same hop; TRM's 1-step-gradient ablation shows the gradient through inner recursions matters there, 56.5 → 87.4, which argues against n > 0 with no gradient on our loop). n = 2 is a Stage-B arm under the identical-input target cue (§261 form), where segments are the natural unit.

**Halt head (TRM's replacement for HRM's Q-learning ACT; TRM §4.6: 86.1 → 87.4 and no second forward):** q_t = σ(w_q · sg(h_dec,t) + b_q), h_dec,t = the decoder's answer-position hidden state (post ln_f, as hh_hidden, inject_aw_eval.py:831-837); ℓ_q,t = BCE(q_t, 1[argmax_NUM Dec_S(S_t) = a]). L = L_DS + λ_q (1/K) Σ_t ℓ_q,t, λ_q = 1. Main arm: sg on the feature, so q never shapes the loop (single-change discipline; the loop's training signal is the answer CE only); `--ds_q_grad 1` is a Stage-B arm (TRM lets q's gradient into the network). Labels are derived from the final answer only (C4-clean). Eval halting: t* = min{t : q_t ≥ 0.5}, answer = pred_{t*}, else K' 14 (score_hx's rule); the training-free r = 3 rule is read alongside. The S-only decode is ~12 tokens vs a ~120-token prompt, so halting-by-decode costs ~0.1 pass, not the ~2x of Note 266-1.

**Aux:** off and not run (the DS path skips aux_logits; the `aux_step > 0` assert at inject_aw.py:259 is relaxed under `--ds 1`). The hold penalty (uses d to pick hold passes) is off in Stages A-B (`--hold_log` for the stats only). WS-READOUT (inject_aw.py:135-151, no_grad, never fed back) stays as the formation monitor.

**Why the decoder shortcut cannot pass the gate here.** With the text decoder, per-pass CE at t < d rewards "decoder does the remaining hops from text" (the K' 2 → d 4 shortcut, Note 264-1); a DS-TEXT arm is run to show it. With the S-only decoder the only shortcut left is **in-pass**: pass t reads the whole text and computes a directly (CD-AN: d 3 / 4 = 0.085 / 0.090 at 12k, Note 237-4; d 2 above chance). It is excluded by NOT-ONE-PASS (d 2 at K' 1 ≤ 0.20 under Dec_S; at d ≤ 4 Stage B: d 3 / 4 at K' 1 ≤ 0.20, K' 2 ≤ 0.20 for d 4) plus the causal swaps.

**Plan cue.** The step cue names S_t (restriction 1). Audit's C4 ruling (Note 266-1) concerns step values in the loss; the cue is a disclosed scope restriction, not a loss term, and §267 carries the same restriction. The plan-free version is Stage B's target-cue arm (identical cue every pass naming S_d, §261's TGT_FMT; n = 2 inner micro-passes).

## 3. Code touch points (read)

- **boxes/inject_aw.py** (695 lines). Flags beside line 249: `--ds {0,1}` (0 = bitwise bun), `--ds_dec {sonly,text}`, `--ds_inner n`, `--ds_q λ`, `--ds_q_grad {0,1}`. Asserts 259-268: `--ds 1` requires `--write_src answer --cue step|target`, no `--ckpt_passes`, no `--cue_ans`, no `--ask_p`, `--hold_inc 0`; aux_step may be 0. `loss_reg` 361-419 untouched; a new `loss_ds(items, stat)` is dispatched from `loss_items` 433-440 (ask items are not allowed under ds). The micro loop at 553-555 (`(lk / a.micro).backward()`) becomes `lk = loss_ds(...)` doing its own per-pass backward scaled by 1/micro and returning a detached float, under `if a.ds` only. tok_aw 76-93: add `o["dec_s"]`. Logging 594-645: per 1k bin `ds_ce_by_pass {t: ce}`, `ds_acc_by_d_pass {d: {t: acc}}` (the in-run decode), `q_acc_by_pass`, `q_mean_by_d_pass`; `ws_readout` as now. mem_train 453-459 runs the ds path when `--ds 1` (per-pass backward lowers the peak; micro still pinned).
- **InjectAW** (45-63): `dec_sonly(S, items, dev)` = inject_ow.dec_logits's body on `it["dec_s"]`; `self.dec_mode` so `dec_logits` dispatches, which keeps every eval part working (passes() calls `m.dec_logits`, inject_aw_eval.py:55-83). q head: `self.q = nn.Linear(D, 1)` registered on the model so it lives in model.pt (eval `load`, 1291-1296, builds from args.json; strict load).
- **boxes/inject_aw_eval.py:** new part `ds_halt` (eval_ds_halt.json in part_halthead's output format, 884-916, with p = q_t and HH_KMAX 14, fit-free), TEST2 d 1-7. Existing parts reused unchanged: acc (104), swap_1 (271-279), swap_kd (391), lp5 (118), probe (313-350; u1|v1 is the drift probe), ws (352).
- **theory/score_dsr.py** (new; `score_ds.py` is §256's).

## 4. Gates (build; Audit G-D)

1. **p = 0 equivalence:** `--ds 0` replays the first 100 steps of ao_s1 (AO recipe) and of bun_s1 bitwise (log loss at step 100; the AO_v1 §3 replay convention).
2. **Decoder path:** Dec_S on a batch equals dec_logits on hand-built dec_s prompts in fp64; with the same (u, S_d) and S = 0 two items give identical logits (text-independence assert).
3. **gradflow_check** (read from the decorators down): per pass t, nonzero grads on mem.qkv / gate / ln_w / wo and the trunk from ℓ_t; `S_{t−1}.requires_grad is False` at every pass start; exactly K grad-carrying write_seq calls per batch under `--ds_inner n` (hook count); with `--ds_q_grad 0`, ℓ_q's grad on every trunk / mem parameter is exactly 0; with `--ds_dec text`, the gradient of ℓ_1 on d 4 items reaches the trunk's attention over op-sentence rows (the shortcut path exists, so the arm is informative).
4. **Forward identity:** `--ds 1` with a debug no-detach switch gives the same ℓ_t values at step 1 (detach changes gradients only).
5. **mem_check** at the d_max shape with micro pinned (1 GiB headroom); eval peak unchanged.
6. **200-step smoke at full data scale** on the real stream, locally (Python 3.11 compatibility of any new API).
7. **Eval identity:** `ds_halt`'s decode at K' 8 reproduces eval_acc's K' 8 predictions bitwise; the in-run q_t on a batch equals the eval part's q_t (fp32).
8. Zero-init usage monitor: write_grad_norm (logged) and the mean-state ablation (LP5) at eval; Stage B (bun init, W_o ≠ 0) is the non-zero-init arm.

## 5. Stages, arms, criteria

**Stage A — formation (the C4 test; the AO-REP harness so the record is comparable).** Init s233_askname_s1 (`--init_trunk` + sha), AO recipe (`--args_ref aw_plaw2_s1/args.json`: REG d ~ U{1..2}, K 8, bs 64, lr 1e-4, 12k cosine, mul_s_p 1, `--aux_step 0`), RTX 3090.
- DSR dsr_s1 / dsr_s2: `--ds 1 --ds_dec sonly --ds_inner 0 --ds_q 1 --ds_q_grad 0`.
- SAO sao_s1 (single-change control for the decoder): AO loss (CE at K = 8 only, full graph, no detach) with `--ds_dec sonly`. Separates "S-only decoder stops the drift" from "deep supervision + detach forms the loop".
- DST dst_s1 (shortcut arm): `--ds 1 --ds_dec text`. Predicted to pass accuracy and fail NOT-ONE-PASS / NOT-DECODER.
- Comparators (stored, no new numbers computed by this document): ao_s1-s4, aex0_s1/2, aw_plaw2_s1 (aux), §267's aoc_s1 when it lands.
- **4k gate** (`--stop_at 4000`, resume to 12k on GO; Note 269-1 FD1: formation markers read ≥ 4k; read on the raw 4k-bin values, no smoothing):
  - GO per DSR seed: ws_readout['2']['1'] ≥ 0.80 (NO-DRIFT; AO 0.11-0.27 at 2k, final ≤ 0.13 in 3/4) AND (ds_acc_by_d_pass['2']['2'] ≥ 0.40 OR the slope of the last 3 d 2 train bins > 0.05/bin).
  - PARK the line if ws['2']['1'] ≤ 0.30 in both DSR seeds (the drift recurs without the text path: the mechanism claim of §1 is false).
  - Else NOTE: continue to 12k on one seed only.
- **Binding at 12k, per DSR seed (score_dsr.py):**

| Criterion | Field | Threshold |
|---|---|---|
| LP1 | last-bin by_d['1'].acc | ≥ 0.90 |
| ACC2 | eval_acc by_d['2'].acc (K' 8, Dec_S) | ≥ 0.70 and ≥ CD-AN d 2 + 0.30 (aw_pcd2_s1, stored) |
| NOT-ONE-PASS | eval_acc by_d['2'].acc_kprime['1'] | ≤ 0.20 |
| NO-DRIFT | eval_probe probes['u1\|v1'].test_acc on d 2 | ≥ 0.90 (G-AW0 init 1.000; AO 0.11-0.25) |
| STATE-USE | eval_swap_1 by_k['8'].follow_ca | ≥ 0.50 and ≥ floor + 0.30, n_pairs ≥ 1000 |
| LP5 | eval_lp5 d 2 drop | ≥ 0.30 |
| CUE-ONLY | score_ae rule | guard → VOID |
| STABLE | score_ao.stable | guard |
| Q-SIGNAL (desc. → binding in B) | eval_ds_halt d 1-2 acc_halt | ≥ acc_k8 − 0.02, mean_khalt d 2 > d 1 |

Labels, first match: VOID / UNSTABLE / DRIFT (NO-DRIFT fails) / ONE-PASS (NOT-ONE-PASS fails) / NOT-FORMED (ACC2 fails) / ACC-ONLY (STATE-USE or LP5 fails) / **DSR-FORMED** (all pass). Arm: FORMED (both) / SPLIT / FAIL. SAO and DST are descriptive and reported on the same table (DST is expected ONE-PASS). STILL-RISING: AO_v1's one +6k extension rule applies, re-scored at 18k, final. Wilson 95% CIs on every binding rate; NEAR flag if the CI covers the bar.

**Stage B — from bun (retention, extension, C5; only on Stage A FORMED or SPLIT with NO-DRIFT in both).** Init bun_s1 / s2 (`--init_model` + sha), bun's args (REG d ≤ 4, §257 insert mix, 8k, `--aux_step 0 --hold_inc 0 --hold_log`), arms: DSR-B (step cue, n 0), DSR-B-TC (`--cue target --ds_inner 2`), DSR-B-QG (`--ds_q_grad 1`), CTRL bunx (`--aux_fade 0,0`, text decoder, CE at K only = AEX0 from bun). Binding per seed: RETAIN d 1-4 ≥ 0.95 at K' 8; NOT-ONE-PASS d 3 / 4 at K' 1 ≤ 0.20 and d 4 at K' 2 ≤ 0.20; SWAP4 (swap_kd d 4 t 2 follow_U ≥ 0.50, ≥ floor + 0.30); LP5-D4 ≥ 0.30; C2-RETAIN d 6 ≥ bun − 0.05 (bun 0.972 / 0.956; ≥ 3 seeds and per-seed margins before any C2 claim, Note 269-1 ST2 — so C2 is descriptive in B); C5 via score_hx.halt on eval_ds_halt (C5-EXTRAP required for the C5 line; C5-IN-RANGE reported); C6 descriptive (eval_long_k128 hold; expected DRIFT without the penalty; a `--hold_inc` arm is Stage B2 and needs an Audit ruling on d-dependence under C4). Mechanism gate at the claimed depth: WS / in-run decode at d 4 passes 3-4 ≥ 0.90.

**Stage C (strict C4, after B):** GPT-2 base init, no askname, d ≤ 2 → 4 curriculum; the same gates. Learnability pilot first (process rule 5).

**OC block.** Stage A per DSR seed: P(NO-DRIFT at 4k | S-only removes the drift direction) 0.7, | it does not 0.1; prior 0.35 → P(NO-DRIFT) ≈ 0.31. P(DSR-FORMED | NO-DRIFT) ≈ 0.45 (the detach may block chaining; the askname decoder must learn the read from W_o = 0), so P(DSR-FORMED per seed) ≈ 0.14, both ≈ 0.08 (ρ ≈ 0.5). Under the null (the decoder change does nothing), P(DSR NO-DRIFT and SAO DRIFT in the same lineage) ≤ 0.05. Pre-named reads: SAO NO-DRIFT alone → "the S-only decoder suffices; deep supervision not needed" (then CE-at-K + S-only is the cheaper route); DSR DRIFT with SAO DRIFT → PARK the transplant and the AO line stays closed; DST ONE-PASS → the shortcut account of Note 264-1 confirmed on small. Stage B: P(RETAIN ∧ NOT-ONE-PASS ∧ SWAP4) ≈ 0.4 given A; P(C5-EXTRAP) ≈ 0.25 (ACT / PonderNet collapse to one loop on looped LMs, RecurTrace 2609.03379; §248's head halts early out of range).

**Censoring.** An unfinished run or a crashed eval is VOID-RERUN once (fresh host, `--keep_final_ckpt`, off-host ckpt every 2k); never scored on partial logs; the 4k read is on complete bins only; a seed PARKed at 4k is not extended; no third extension. Eval parts missing → pending, not FAIL. Binding evals on cloud hosts only (local RAM bit flips).

## 6. Scorer sketch (theory/score_dsr.py, fixtures before launch)

Reads: `log.jsonl` 1k-bin records (`ws_readout`, `by_d`, `ds_acc_by_d_pass`, `q_acc_by_pass`), `eval_acc.json` (`by_d[d].acc`, `acc_kprime`), `eval_probe.json` (`probes['u1|v1'].test_acc`), `eval_swap_1.json` (`by_k`, `n_pairs`; floor from `items`), `eval_lp5.json`, `eval_cue_only.json`, `eval_ds_halt.json` ({d: {acc_k8, acc_halt, mean_khalt, khalt_hist, never_halt_n}}), `train_summary.json` (score_ao.stable), `aw_pcd2_s1/eval_acc.json` (CD-AN). Stage-A labels as §5; `--gate4k` mode emits GO / PARK / NOTE from the 4k bin. Stage B adds score_hx.halt / hold, swap_kd d 4, lp5h. Fixtures (≥ 8): formed, drift, one-pass, acc-only, still-rising, pending, void (cue-only), unstable, gate4k GO / PARK. Wilson CIs with NEAR.

## 7. Cost (RTX 3090, ~$0.2/h)

AO recipe at d_max 2: ~0.6 s/step → 12k ≈ 2 h ≈ $0.45 per run (AO_v1 §7); DS adds 8 short decodes per step (+~8%) and per-pass backward lowers peak memory. Stage A: 4 runs to 4k ($0.6) + 3 continued to 12k ($1.2) + evals (~$0.1 each, $0.4) ≈ $2.2; extensions ≤ $0.4. Stage B: bun 8k ≈ $1.0-1.1 per run, 2 seeds × 3 arms + CTRL ≈ $7.5. Guard: Stage A $3, Stage B $9 (each < $30; Stage B registered separately after A reads).

## 8. RED TEAM (separate sub-agent)

(pending)

## 9. Verified arXiv ids (each abs page fetched 2026-11-03, title checked)

- 2506.21734 Hierarchical Reasoning Model (Wang et al.): deep supervision with detached segments, one-step gradient, ACT Q-head with M_min / M_max and Q-learning targets.
- 2510.04871 Less is More: Recursive Reasoning with Tiny Networks (Jolicoeur-Martineau): single network, n = 6 / T = 3, N_sup = 16, halting by BCE on correctness (no continue loss), 1-step-gradient ablation 56.5 vs 87.4, EMA 0.999.
- 2211.09961 Path Independent Equilibrium Models Can Better Exploit Test-Time Computation (the carried-state / extra-iteration reading).
- 2202.05826 End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking (progressive loss, recurrence extrapolation).
- 2502.05171 Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (truncated backprop through the last recurrences).
- 1603.08983 Adaptive Computation Time for Recurrent Neural Networks (Graves); 2107.05407 PonderNet (Banino): the halting precedents.
- 2409.15647 Looped Transformers for Length Generalization (Fan et al.): answer supervision at the depth-matched loop count (ACUE's precedent, Note 251-10).
- 2609.03379 RecurTrace (Wang et al.): ACT and PonderNet collapse to one loop on a looped LM.
- 2603.21676 (Chen; backlog BM4 counter-claim: per-step supervision hurt extrapolation; abstract confirms final-output-only supervision).
