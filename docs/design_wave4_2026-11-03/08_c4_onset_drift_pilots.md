# 08 C4 ONSET-DRIFT PILOTS: one registration-ready cell bundling CA1 SG-WRITE, CA2 ONSET-FREEZE (LP-FT, EMA), CA4 SHARED-STEP SIAMESE; CA3 MULTI-LEAF and K18 EM-ACCUM staged behind rulings

Wave-4 lead design, 2026-11-03. Backlog IDs: §13.2 C4 lane / §13.3 CA1 (K13), CA2 (K14), CA3 (K15), CA4 (K16), CA6 (K18). Lane B (restriction 2). Status: draft for Theory, not registered; no build, no spend.

## 1. Goal

The answer-only line (§244 / §245 / §251 AEX0 / ACUE) fails at one measured site (Note 251-9 addendum): on d > t items the pass-t ':' code stops carrying v_t within the first 1k steps (ws_readout d 2 pass 1 at 1k = 0.22 / 0.09 / 0.16 / 0.27 in ao_s1 / s3 / s4 / aex0_s1; probe u1|v1 0.11-0.25; final <= 0.13 except the one formed seed), while d 1 pass 1 stays 1.0. The code is present at init (G-AW0 1.000). The answer CE pulls it toward the final answer on items whose answer is not v_t. Each entry here removes or weakens one gradient path to that code without putting any step value into the loss (Note 266-1), on the Note 251-10 harness. One cell answers (a) WHICH path carries the drift (direct write gradient vs shared-weight interference), and (b) whether holding the seeded code lets pass 2 form from answers alone (the analogue of §251 AEX, where held seeded passes let d 3-4 form from answers 2/2 on accuracy).

Scope of any pass: "answer-only d 2 formation given the askname init and the step cue (plan given); one lineage". Not a strict C4 (the init saw per-entity values; §244's disclosure stands).

## 2. Harness and the measured loss

Harness H0 = the §244 AO recipe = ao_s1's args (boxes/runs_fsq/ao_s1/args.json): inject_aw --arm inject --cue step --write_src answer --init_trunk s233_askname_s1 (sha c00f7ec0) --think_k 8 --d_max 2 --aux_step 1 --aux_fade 0,0 --aux_reweight --fast_add --cache_lower --write_mix_start 0 --write_mix_end 0 --bs 64 --lr 1e-4 --steps 12000, mul_s_p 1, micro 1, --args_ref aw_plaw2_s1. hostprobe 0.56 s/step on an RTX 3090.

Loss (boxes/inject_aw.py loss_reg, lines 361-419): for pass t = 1..8, S <- m.write(S, x_t, msk, hl) (line 386; InjectAW.pass_aw lines 47-56 runs the trunk, gathers the ':' row and calls DM.write_seq(mem, h6[':'], h11[':'], S)); then lg = m.dec_logits(S_K) (404), la = CE(lg, ans) (405), aux = 0 in-graph under aux_fade 0,0 (lam_at, 299-302), loss = la (407). In deltamem.write_seq (179-227) the write source is u = LN_w(h_w) with gamma = 1 (gamma fill, 550), from which k, v, alpha, beta are projected. So the answer gradient reaches pass t's ':' code through two paths: P-direct, d loss / d u_t via S_t -> S_K (additive heads accumulate writes; delta heads contract) and the block-6 reads at passes t+1..8 and the decoder (read_fixed 158-176); P-shared, the same trunk weights computing the ':' of pass d and the reads. The entries below cut P-direct (CA1), slow the weights (CA2), or add an invariance term on u_1 (CA4).

## 3. Entries, applied to Note 266-1 one by one

**CA1 SG-WRITE, --sg_write 1.** In loss_reg's pass loop, for pass t, every item i with d_i > t gets its write source detached: pass_aw takes an optional per-item mask and applies `hw = torch.where(sg[:, None, None], hw.detach(), hw)` (and the same on h6) on the gathered ':' rows before write_seq. Items with d_i <= t (the pass that answers, and the hold passes 3..8) write live. DeltaMem's own projections (qkv k/v rows, gate, ln_w) stay in the graph; only the trunk's ':' activations are cut, so "the answer gradient reaches the pass-t trunk code only via shared weights". Ruling: no value enters the loss; C4-eligible. Mask default None = bitwise.

**CA2a ONSET-FREEZE LP-FT, --freeze_trunk_until 2000 --trunk_lr_mult 0.1.** Two optimiser groups (fsq120.make_opt 731-737 builds one AdamW; the lr loop at inject_aw 545-546 calls lr_at, which multiplies by gr["mult"] (fsq120 740-743)). The trunk group gets mult 0 for steps <= 2000 (AdamW with lr 0 applies neither update nor decay), then 0.1; the DeltaMem group (mem.* : qkv, gate, ln, ln_w, wo, s_add, s_del, add_gain) stays at 1.0. Kumar et al. (2202.10054): fitting the new head first, then fine-tuning at a lower trunk rate, avoids distorting pretrained features. Here the "feature" is the askname ':' code. No new parameters, so no zero-init pathway is introduced; write_grad_norm is already logged every 100 steps (565, 604) and serves as the usage monitor of the DeltaMem group during LP. C4-eligible.

**CA2b EMA WRITE SOURCE, --ema_write 0.999.** An EMA copy of the trunk (not of DeltaMem), built in build_aw after the init load (109-125), computes h6 / h11 at ':' under no_grad for passes with d_i > t; the live trunk computes the reads, the pass-d write and the decoder. EMA update theta_ema <- tau theta_ema + (1 - tau) theta after opt.step (570). Mean-teacher form (1703.01780; BYOL 2006.07733). Contrast with CA1: SG = live weights, no gradient; EMA = lagged weights, no gradient. If SG drifts and EMA holds, interference is fast relative to 1 / (1 - tau) = 1k steps. Cost: +0.5 GB fp32 copy, one extra no-grad trunk forward on d 2 steps (cache_lower cannot be shared). C4-eligible.

**CA4 SHARED-STEP SIAMESE, --sia 1.0.** For each d 2 item a partner is built with the same S_0 given, givens, distractors, names, frames and S_1 sentence, and only step 2 re-drawn (a new op / constant / S_2 name via ow_words' _draw_program law for one step, re-rendered through NW._finish as make() does, 108-122; OW.solve asserted; the partner is a new helper in inject_aw, ow_words byte-identical). The partner runs pass 1 only. Term on the write source u = LN_w(h_w) at pass 1 (D 768): L_sia = ||u_a - u_b||^2 / D + mean_j max(0, 1 - std_j(u)) (VICReg invariance + variance, 2105.04906; no projector, no new parameters, so no collapse-prone zero-init path). Pairing uses the generator's structure (which sentence is step 2), never a value; it is the same information the step cue already supplies. Ruling: no step value enters the loss; C4-eligible with the pairing disclosed. Known limit: invariance to the suffix does not force the code to be v_1 (a prefix-text code also satisfies it); the DRIFT reading decides.

**CA3 MULTI-LEAF, staged, needs a ruling.** insert_render's 'chain' insert (194-233) already adds "X has c more than S_j" after S_j's sentence, j in 1..d-1. A leaf item asks "How many u does X have?" over the same text (answer v_j + c mod 11; spine' = S_0..S_j, X; cue plan d' = j + 1) and is co-batched with the original. Under Note 266-1 the leaf answer is a bijective image of v_j with c in the text, i.e. a prefix answer by proxy. As drafted it is therefore NOT C4; it may run as descriptive / mechanistic work ($0.6) only after Audit rules on the narrower question "does a different question's final answer count as a step value when it is a bijective image of one?". Not in the stage-1 cell.

**K18 EM-ACCUM, the §267 follow-up (CA6).** Runs only on EM-C4 2/2. inject_em's forward DP with the accumulating state S_t = write_seq(emb(v_t), S_{t-1}) instead of the zero state. The chain is no longer Markov in v_{t-1}, so exact enumeration at d_max 3 costs 1 + 11 + 121 = 133 upper-half forwards (5.8x §267, ~$8 / seed); at d_max 2 it equals RESET-DM exactly (S_0 = 0), so it must run on OPS_CHAIN d_max 3. Proposed: beam-partial enumeration (top-3 at pass 1 x 11 at pass 2 = 45 forwards, ~2x §267) plus RLOO (2402.14740) on sampled off-beam v_1 with the beam mass as control variate; a gate that the beam-partial objective equals the exact 133-path sum within 1e-3 nats on 4 d 3 items in fp64. The DISTILL-TO-DM sub-arm (the EM posterior q(v_t) as a soft target for the continuous ':' logits) puts a distribution over step values into the loss; model-derived, not hand-built, but it needs a ruling before any C4 wording. ~$3 with one CTRL; registered separately.

## 4. What FD1 (Note 269-1) allows at 2k

At step 2000 every stored d_max 2 run, formed or not, has pass-2 WS 0.02-0.13 and aux 0.08-0.11; onset is 4-7k. So at the 1k / 2k / 4k rows only readings of an ALREADY-PRESENT code are valid: ws_readout['2']['1'] (the pass-1 code on d 2 items, 1.000 at step 0) with ws_readout['1']['1'] as the still-present control. NOT valid at <= 4k: ws d 2 pass 2, train d 2 acc, any K' read; they are reported as trends and read bindingly at 12k. For LP-FT the trunk cannot move before 2k, so its DRIFT rows are 3k and 4k (>= 1k after unfreeze); its 1k / 2k rows are trivially 1.0 and are not evidence.

## 5. Stages, cells, cost (RTX 3090, ~$0.2 / h)

- **Stage 0, $0 local:** build gates (section 6).
- **Stage 1, ~$1.0, guard $1.5:** sg_s1, sg_s2 (--sg_write 1), lpft_s1, ema_s1, sia_s1; H0 args + the one flag each (args_ref ao_s1; args diff = {flag, seed, out, stop_at, ckpt_every 1000}); --stop_at 4000 (exit 9, ckpt kept). 4k x 0.56 s = 37 min + build/drain ~10 min = ~$0.17 (sia +15%, ema +25%).
- **Stage 2, <= $1.75, guard $2.0:** resume to 12k (--resume ckpt, same --steps so the cosine is unchanged; memory rule: ship args.json + ckpt) only for cells labelled NO-DRIFT or NEAR at stage 1; 8k x 0.56 s = 75 min ~$0.3 + eval parts acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws (~$0.05).
- Comparators, no new spend: AO ao_s1-s4 (DRIFT rows and finals), AUX aw_plaw2_s1 / aux_s2 (formed at 7k), CD-AN aw_pcd2_s1 for LP3's floor.
- Family total <= $3.5 (within the $30 auto-approve rule).

## 6. Gates (stage 0)

1. **p = 0 equivalence, every flag:** --sg_write 0 / --freeze_trunk_until 0 / --ema_write 0 / --sia 0 together = the current module bitwise on a 100-step local replay of ao_s1's args under the det wrap (same card, module-before vs module-after; the §251 / §252 form). deltamem.py, ow_words.py, fsq120.py stay byte-identical (sha asserted in args.json: deltamem_sha256, ow_words_sha256 at 494-496).
2. **gradflow (scratch/gradflow_check.py's term() pattern, 24-27; read from the decorators down):** on one d 2 batch with la alone: under SG the grad of la w.r.t. the pass-1 ':' h11 rows (retain_grad) is exactly 0 and nonzero at pass 2 and on mem.qkv / gate / ln_w; on a d 1 batch nonzero at pass 1. L_sia alone: nonzero on the trunk and ln_w, zero on wo / s_add (it never touches the read). EMA: the EMA copy holds no parameter in the graph; the live trunk's grad on a d 2 batch is nonzero (reads, pass 2, decoder). LP-FT at step <= 2000: trunk group lr 0 -> trunk params bitwise unchanged after 10 steps; mem group changed.
3. **mem_check** (--mem_check, 466-490) at the d 2 shape, micro 1 pinned, each flag on; EMA budget 21 GiB with the extra copy.
4. **Smoke** 200 steps at full data scale per flag on the local card (set build 27 s), then the ws_readout row at step 200 present and the d 1 row ~1.0.
5. **Partner gate (sia):** 2,000 partners: text differs from the original only in the step-2 sentence and the question; OW.solve matches; prefix tokens identical up to S_1's sentence end; v_1 equal; the sig excluded from EXCL-overlap by construction (the stream's own items).
6. **Resume gate (stage 2):** a 4k -> 4,100 resume reproduces the un-stopped run's step-4100 loss row bitwise on the local card (resume path 514-528).

## 7. Pre-registration table

Per cell. n = 200 items for log ws rows (Wilson 95% half-width 0.055 at 0.80); NEAR = |p - bar| <= 2 sqrt(p(1-p)/n) (score_em.near). Full-set evals n 1,000.

| Criterion | Stage | Field | Bar | Kind |
|---|---|---|---|---|
| PRESENT-1 | 1 | log ws_readout['1']['1'] at 1k, 2k, 4k | >= 0.95 each (else VOID: the harness lost d 1) | guard |
| DRIFT | 1 | log ws_readout['2']['1'] at 1k, 2k, 4k (lpft: 3k, 4k) | NO-DRIFT if every row >= 0.80; DRIFT if any row <= 0.30 at >= 2k; else NEAR | binding |
| DRIFT-vs-AO | 1 | the same rows vs ao_s1-s4 pooled (0.09-0.27 at 1k) | descriptive: margin per row | descriptive |
| TREND-2 | 1 | log ws_readout['2']['2'] at 4k; by_d['2'].acc bin 4 | reported, never a label (FD1) | descriptive |
| FORM-ACC | 2 | eval_acc by_d['2'] (TEST2 d 2, K' 8) | >= 0.95 | binding |
| FORM-WS2 | 2 | eval_ws by_d['2']['2'] | >= 0.90 (the §268 P-2 stage gate; mechanism at the claimed depth) | binding |
| NOT-DECODER-0 | 2 | acc_kprime d 2 at K' 0 | <= 0.20 (K' 1 is 1.0 in every working loop, REVIEW_AC 3) | binding |
| LP5-1 | 2 | lp5 pass-1 mean-state drop on d 2 | >= 0.30 | binding |
| STATE-USE | 2 | swap_k follow_U, floor, n_pairs >= 1000 | >= 0.50 and >= floor + 0.30 | binding |
| CUE-ONLY | 2 | cue_only d 2 | <= max(majority, CD-AN) + 0.05, else VOID | guard |
| STABLE | 2 | score_ao.stable (28-55) | n_nonfinite <= 10, no unrecovered 3x rise | guard |
| PROBE | 2 | probe u1|v1, read1|v1 test acc | >= 0.80 (descriptive: write vs readout) | descriptive |

Labels per cell, first match: VOID / UNSTABLE / DRIFT / NEAR-DRIFT / NOT-FORMED (NO-DRIFT, FORM fails) / ACC-ONLY (FORM-ACC, not FORM-WS2 or STATE-USE) / AO-FORMED (all binding). Cell wording: "answer-only d 2 formation with <flag> (askname init, plan given, one lineage)".

Decision rules (pre-stated): SG NO-DRIFT in 2 / 2 and AO-FORMED >= 1 / 2 -> register 2 more SG seeds + a d_max 4 SG extension (the AEX design with aux off). SG NO-DRIFT but NOT-FORMED everywhere -> the direct path carries the drift, but formation needs it: next is SG released at 4k (pre-named, not run here). SG DRIFT in 2 / 2 -> the drift is shared-weight interference; CA2 leads; if lpft also DRIFTs after unfreeze, the answer-only line in this harness closes and K18 is the remaining C4 route. SIA is read for DRIFT only unless it also forms.

**OC.** Per SG seed: P(NO-DRIFT) 0.55 (the direct path is the only one specific to d > t items; interference from pass-2 training is plausible but acts through a different cue and state); P(AO-FORMED | NO-DRIFT) 0.45 (§251 AEX: held seeded passes let new depth form from answers 2 / 2 on accuracy, 1 / 2 on the causal bar); per seed ~0.25; P(SG AO-FORMED >= 1 / 2) ~0.44. lpft NO-DRIFT (post-unfreeze) 0.40, AO-FORMED 0.15; ema NO-DRIFT 0.40, AO-FORMED 0.15; sia NO-DRIFT 0.35, AO-FORMED 0.10. Under "the drift is interference" P(SG NO-DRIFT) ~0.15. P(any cell AO-FORMED) ~0.55. The 12k final is read once; no STILL-RISING extension (§244 s2 needed 18k + a re-warm, so a NOT-FORMED here says "not by 12k", disclosed).

**Censoring.** A cell without its 4k ws row is VOID-RERUN once (fresh host). A stage-2 resume that crashes is VOID-RERUN once from the kept 4k ckpt; a crashed eval is re-run on the kept model.pt (keep_final_ckpt not needed). Cells labelled DRIFT at stage 1 are not resumed; their ckpts are deleted after the stage-1 score is committed, logs kept. ONSET rows are the raw log values (Note 267-1 (3)). No bar is edited after the first output.

## 8. Scorer sketch

theory/score_c4b.py (fixtures before any output): reads boxes/runs_fsq/<cell>/log.jsonl rows step in {1000, 2000, 3000, 4000} -> ws_readout['1']['1'], ['2']['1'], ['2']['2'], by_d['2'].acc; stage 2: eval_acc.json (by_d, acc_kprime), eval_ws.json, eval_lp5.json, eval_swap_k.json (n_pairs, follow_U, floor), eval_cue_only.json, eval_probe.json, train_summary.json; comparators ao_s1-s4, aw_plaw2_s1, aw_pcd2_s1 (CD-AN floor). Imports score_ao.stable and wilson / near as in score_em. Per-cell dict {rows, DRIFT, NEAR flags, FORM..., label}; pooled "SG k/2". Fixtures (8): no-drift + formed; no-drift + not-formed; drift at 2k; NEAR row; lpft read at 3k/4k only; missing 4k row -> VOID; unrecovered rise -> UNSTABLE; acc without WS2 -> ACC-ONLY. Output theory/results/score_c4b.json; run twice, byte-identical.

## 9. Code touch points (read)

- boxes/inject_aw.py: InjectAW.pass_aw 47-56 and write 58-59 (sg mask / EMA source); colon_logits 61-63; build_aw 109-125 (EMA copy after strict init load, sha asserted 113); ws_readout 135-151 and the 1k-bin log 632-633 (unchanged, the DRIFT field); args 211-249 (four new flags, defaults off); loss_reg 361-419, pass loop 372-393 (per-t sg mask; sia partner forward after 386), la 405, loss 407-409; make_opt call 508 and the lr loop 545-546 (groups for LP-FT); opt.step 570 (EMA update); resume 514-528; stop_at 671-672.
- boxes/deltamem.py: read_fixed 158-176, write_seq 179-227 (u = LN_w(h_w); the detach lands on the caller's h rows, deltamem unchanged).
- boxes/fsq120.py: make_opt 731-737, lr_at 740-743 (gr["mult"]).
- boxes/ow_words.py: _draw_program 45-97, make 108-122 (the sia partner law); boxes/insert_render.py distract_mix 194-233 (CA3 leaf).
- boxes/inject_aw_eval.py: part_swap_k 255, part_probe 313-349 (u1|v1, read1|v1), part_ws 352.
- theory/score_ao.py stable 28-55; theory/score_em.py wilson / near / onset; scratch/gradflow_check.py term 24-27.

## 10. Literature (ids fetched 2026-11-03, titles checked)

2502.05171 Geiping et al., Scaling up Test-Time Compute with Latent Reasoning (recurrent depth, truncated backprop through the loop: the SG analogue). 2506.21734 Wang et al., Hierarchical Reasoning Model; 2510.04871 Jolicoeur-Martineau, Less is More: Recursive Reasoning with Tiny Networks (deep supervision with detached segments: answer loss per segment, gradient cut between them, the closest training-side precedent for SG-WRITE). 2202.10054 Kumar et al., Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT). 2006.07733 Grill et al., BYOL; 1703.01780 Tarvainen and Valpola, Mean teachers (EMA target). 2011.10566 Chen and He, SimSiam (stop-gradient prevents collapse); 2105.04906 Bardes et al., VICReg. 2402.14740 Ahmadian et al., Back to Basics (RLOO). 2312.02179 Phan et al., Training Chain-of-Thought via Latent-Variable Inference (TRICE). 2409.15647 Fan et al., Looped Transformers for Length Generalization (loss only at T(n)). 2007.01919 Correia et al., Efficient Marginalization of Discrete and Structured Latent Variables via Sparsity. 2407.20311 is Ye et al., Physics of Language Models Part 2.1 (hidden reasoning on grade-school math): it shows models precompute intermediate values needed for later questions over the same context, the motivation for CA3, not a multi-question training method. Not found: a stop-gradient on a recurrent-memory write source used against onset drift of a pretrained code (narrow novelty; Literature to spot-check).

## 11. RED TEAM

The required red-team sub-agent could not be launched (the session's concurrent sub-agent limit was reached at both attempts); the critique below is the lead's own, written against the draft with the project's failure taxonomy (§13.0), and must be redone by an independent agent before registration.

1. **SG may stop formation, not only drift.** With P-direct cut on d 2 items, pass 1's write is readable only if the askname code is already linearly stored by the untrained DeltaMem projections; AO probes say storage follows the code (read1 = u1), so it is, but the read-use at block 6 of pass 2 must form from the decoder gradient alone, which aex0_s2 did not do for d 3 in 16k. Mitigation: the pre-named SG-release follow-up; the NOT-FORMED label is informative either way.
2. **Hold passes 3..8 write live** (t >= d). They re-read the S_2 cue and can overwrite S with a code trained toward the answer, i.e. the decoder shortcut can re-enter through them. NOT-DECODER-0 and FORM-WS2 catch it; a cheaper falsifier is to log ws_readout['2']['3'] (free, same row).
3. **LP-FT's 2k frozen phase is dead time for DRIFT** and the 0.1x trunk rate may simply slow the same drift (a DRIFT label at 4k is then ambiguous between "slower" and "none"); the 12k final decides, so lpft should resume on NEAR as well as NO-DRIFT (adopted above).
4. **EMA: the live trunk's pass-1 code is never read by anything**, so its drift is invisible to the DRIFT row (ws_readout uses the live model's pass_aw). Fix: log both the live and the EMA ':' argmax on the 200 items (two keys), and define DRIFT on the EMA row (the write source) with the live row descriptive.
5. **SIA pairs are the same item twice up to the last sentence**: the invariance term can be satisfied by attending less to the suffix (a position prior), which also helps order-bound selection; the variance hinge on 768 dims of a LayerNormed vector is weak. The DRIFT row is still the measurement; no C4 wording rests on SIA.
6. **Bars copied from P-2 (acc 0.95, WS2 0.90) at 12k** may be too high for an answer-only pilot whose one previous success needed 18k; ACC-ONLY / NOT-FORMED at 12k is then "not by 12k", disclosed, not "cannot".
7. **Seeds:** only SG has 2 seeds; lpft / ema / sia are single-seed and can be read only as DRIFT localisers, never as recipe comparisons (Note 269-1 ST2).
8. **Cheaper falsifier:** stage 1 could stop at 2k for sg / ema / sia ($0.1 each) since DRIFT is visible by 1k in AO; 4k is kept because lpft needs it and the 4k ckpt is the stage-2 init.
