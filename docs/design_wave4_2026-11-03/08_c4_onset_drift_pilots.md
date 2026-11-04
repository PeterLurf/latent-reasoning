Status: completed 2026-11-04 by Theory's agent; not red-teamed yet

# 08 C4 ONSET-DRIFT PILOTS: where does the answer gradient knock the seeded pass-1 code off v_1, and does holding it let pass 2 form from answers alone?

Wave-4 lead design. Draft 2026-11-03 (cut off before completion); completed 2026-11-04 01:59 EST. Backlog IDs: §13.2 C4 lane; §13.3 CA1 (K13) SG-WRITE, CA2 (K14) ONSET-FREEZE, CA4 (K16) SHARED-STEP SIAMESE, plus a new arm ANC (Note 269-4(d) self-distillation). CA3 (K15) MULTI-LEAF and CA6 (K18) EM-ACCUM are out of this design's cells (sec. 4.7). Lane B (restriction 2). Not registered; no build, no spend. Every code citation below was re-read on 2026-11-04 against the working tree (sec. 15).

## 0. Summary and relation to §267

- **Question.** In every failed answer-only run on the askname / P-2 lineage, the pass-1 ':' code on d 2 items stops carrying v_1 within the first 1k steps (Note 251-9 addendum; G-AW0 1.000 at init). Two questions follow. (a) Which gradient path carries this drift: the direct write path, or interference through shared trunk weights? (b) If the code is held, does pass 2 form from answers alone?
- **Stage 1 (decisive, cheapest): 5 cells x 4k steps on Vast RTX 3090, about $1.4 expected, guard $2.5.** The cells are ctrl_s1, sg_s1, sg_s2, anc_s1 and lpft_s1. Each runs the §244 AO recipe with one change. The binding read is the drift of an already-present code (ws_readout['2']['1'] at 1k / 2k / 4k), which Note 269-1 FD1 allows at <= 4k.
- **Stage 2:** non-drifting cells resume to 12k (about $2.3 expected, guard $4). Formation is read only at 12k, with the §244 binding set plus a mechanism gate (FORM-WS2).
- **What this adds over §267 (EXACT-EM-11 stage A).** §267 removes the continuous write. Its pass state is a categorical Z_11 value on reset memory, it uses OPS_CHAIN d_max 3, a new module (inject_em) and exact marginal likelihood. Note 267-1 (2) scopes it as "NOT C4 for the bun architecture; says nothing yet about DeltaMem's accumulating state". This design keeps the bun / of architecture unchanged: a continuous ':' write into accumulating DeltaMem, on the §244 REG d_max 2 harness, with ao_s1-s4 as $0 comparators. It therefore answers what §267 cannot:
  - where the drift enters the continuous write;
  - whether an answer-only route works on the architecture the SOTA uses (of_s1 / of_s2, Note 263-3).

  No cell, module, data law or budget is shared with §267. K18 EM-ACCUM (the §267 follow-up) is removed from this doc and stays behind §267's EM-C4 2/2 (sec. 4.7). Both designs read their onset rows the same way (raw 1k / 2k values against a 0.80 bar, Note 267-1 (3)), so their results compare directly.

## 1. Goal and scope

The measured failure is in the answer-only line (§244 / §245 / §251 AEX0 / the Note 251-10 ACUE pilot):
- On d 2 items, ws_readout d 2 pass 1 at 1k is 0.22 / 0.09 / 0.16 / 0.27 in ao_s1 / s3 / s4 / aex0_s1; P-CTRL in Note 251-11 is 0.110 / 0.125 at 1k / 2k.
- The u1|v1 probe reads 0.11-0.25.
- The final value is <= 0.13, except in the one seed that formed.
- d 1 pass 1 stays at 1.0 throughout.

The code is present at init (G-AW0 1.000). Each arm removes or counteracts one route by which the answer gradient reaches that code. The strict-route arms add no step value to the loss (Note 266-1). ANC adds only a model-derived target (Note 269-4(d)).

Scope of any pass: "answer-only d 2 formation given the askname init and the step cue (plan given); one lineage; REG word programs, chain-order text". This is not strict C4: the askname init was trained on one-hop entity questions whose answers are per-entity values (the §244 / §267 disclosure stands). This design is a mechanism pilot. It makes no claim about C4's operative standard (latent vs text CoT at equal compute; NORTH_STAR row C4).

## 2. Init decision: askname (s233_askname_s1, sha c00f7ec0)

| Candidate | What a drift / formation result would mean | C4 reading possible? | Harness cost | Verdict |
|---|---|---|---|---|
| **askname** | The only init where pass 1 carries v_1 (G-AW0 1.000) and pass 2 has never been trained. This is exactly the measured drift site: all four AO drift readings and the ACUE pilot were taken here. ao_s1-s4 and aw_plaw2_s1 / aux_s2 are $0 comparators on the same args. | Yes, scoped as in sec. 1 (the lineage disclosure only). | §244 harness, realised 0.49 / 0.77 / 0.85 s/step (ao_s1 / s3 / s4 train_summary train_sec 5900 / 9214 / 10155 over 12k) | **chosen** |
| of_s1 | Every pass at d <= 4 is already formed, so there is nothing to form. Answer-only training here tests retention of a trained loop, not onset drift of a seeded pass followed by formation. | No: every stage of the of lineage (askname -> P-2 -> ... -> bun -> of) had per-pass value aux in the loss, so a pass is "retention under answers", not C4. | REG d <= 4 + inserts, ~1.65-1.7 s/step (Note 268-1 rates), 2-3x | rejected for stage 1-2. Pre-named transfer (sec. 12, stage 4): SG on the seeded passes for an answer-only d 5-6 extension from of_s1 (an AEX0 variant), only if SG is AO-FORMED here. |
| bun | As of_s1, and also order-bound (Note 257-3). of_s1 supersedes it as lineage head (Note 263-3). | No | same as of | rejected |

Justification (DATA): Note 269-1 FD1 shows that formation is invisible before 4k, and that 2k gates only detect drift of an already-present code. A formed init (of / bun) has no present-but-unformed boundary to test. askname has exactly one: pass 1 is present and pass 2 is absent.

## 3. Harness H0 and every term of its loss (C4 audit)

**H0 = ao_s1's args** (boxes/runs_fsq/ao_s1/args.json, read 2026-11-04):
- arm inject, cue step, write_src answer;
- init_trunk s233_askname_s1 (init_sha c00f7ec0bd7977fe);
- think_k 8, d_max 2, aux_step 1.0, aux_fade "0,0", aux_reweight, fast_add, cache_lower;
- write_mix_start 0, write_mix_end 0;
- bs 64, lr 1e-4, steps 12000, micro 1, mul_s_p 1, mul_s_p3 1.0, seed 1;
- ckpt_every 1000, keep_final_ckpt;
- eval_parts acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws;
- args_ref aw_plaw2_s1.

The one-d-per-step law: d = default_rng(seed + SEED_D).integers(1, d_max + 1) (inject_aw 271). Every step is therefore all-d-1 or all-d-2.

**Loss in code (boxes/inject_aw.py loss_reg, 361-419), term by term:**

| Term | Code | Target | Step value in the gradient? |
|---|---|---|---|
| la, answer CE | lg = m.dec_logits(S, items, dev) (404; InjectOW.dec_logits, inject_ow 86-90: the decoder input is think + SFX, read-only on S_K); la = CE(lg, tgt) (405); tgt = NUM_IDS[ans] (363) | the final answer | no |
| lam * aux | 394-403 build the aux CE on R_step[t] (tok_ow, inject_ow 69: aux_t = enc(' ' + R_step[t]) + EOT, i.e. "S_t has v_t u"). loss = la + lam_now[0] * aux (407). | per-pass values v_t | **weight exactly 0, but the term is in the graph.** lam_at (299-302) with f0 = f1 = 0: step <= 0 is false for step >= 1, so lam = aux_step * max(0, (0 - step) / max(1, 0)) = 0.0. The gradient is 0 * d aux = +/-0 when aux is finite. A non-finite aux makes tot non-finite and the step is skipped (566-569; counted in n_nonfinite). Note: inject requires aux_step > 0 (assert 259), so H0 cannot simply drop the term. |
| a.cue_ans * acue | 408-409, only `if a.cue_ans` | the answer (Note 251-10) | off (cue_ans 0) |
| loss_ask | loss_items 433-441; NASK = round(ask_p * bs) (265) | v_X of one-hop asks | off (ask_p 0, so NASK = 0) |
| HOLD penalty | HOLD = hold_inc > 0 or hold_log (289); hold_step 559-560 | none (norms) | off |
| g_add calibration | 529-541: no_grad, sets s_add from 64 items | none | not a loss |
| ws_readout / vacc / wsp logs | 134-151, 397-401, 633: no_grad, never fed back | v_t | logging only |

**Strict-C4 fix used by every cell here, `--c4_noaux 1`.** When lam_now[0] == 0.0, the aux block (394-403) runs under torch.no_grad(), so the vacc logs are kept, and aux := la * 0.0 in-graph. After this, no tensor derived from R_step is reachable from `loss`. This is PROVED by construction; gate G-C4a checks it.
- Numerically the change is a no-op. Adding +/-0 to a finite gradient leaves it bitwise unchanged. An AdamW step with a +/-0 component is bitwise unchanged. Gate G-C4a asserts parameter equality after 100 steps under the det wrap.
- The ctrl cell uses the same flag, so every cell meets Note 266-1 literally rather than "up to a zero weight".

**Gradient paths to the pass-1 code u_1 = LN_w(h11[':']) on a d 2 item (H0).**
- The write is DM.write_seq(mem, h6[':'], h11[':'], S) (pass_aw 47-56).
- In write_seq (deltamem 179-223), u = (1 - gamma) LN(h6) + gamma LN_w(h_w). gamma = 1 from step 1 (inject_aw 550 with write_mix 0, 0).
- k, v and the gate (alpha, beta) are projected from u.
- P-direct: d la / d u_1 flows via S_1 -> S_2 ... S_8 -> the decoder's read of S_K (fwd_read), and via the block-6 reads (read_fixed 158-176) at passes 2..8.
- P-shared: the same trunk weights compute u_1, the pass-2 code, the hold passes, the decoder and all reads.

## 4. Arms (each is ONE change on H0 + c4_noaux; flags default off = bitwise)

### 4.1 CTRL, ctrl_s1 (stage 1)

H0 + c4_noaux on the current module and the same host class as the other cells. Why it is needed:
- ao_s1-s4 ran on module 461458f5 with excl_n 160598. The current module adds later features, and §254 recorded excl_n 192598 after §250's frame sets were added. The stream is OW.Stream(OW.SEED_TRAIN, EXCL, ...) (280, a fixed seed), so a different EXCL changes the item sequence.
- So only ctrl_s1 is exactly paired with the seed-1 arms (same d sequence, same items, same DeltaMem init via torch.manual_seed(seed), 271).
- **Validity guard:** ctrl_s1 must read DRIFT (sec. 8). If it does not, the harness has changed and the family is VOID-HARNESS. No localisation claim is then made, and stage 2 waits for a Note.

### 4.2 CA1 SG-WRITE, `--sg_write 1` (sg_s1, sg_s2; stage 1)

- **Change.** In loss_reg's pass loop (372-393), at pass t, every item with d_i > t writes from detached sources. pass_aw gains an optional bool mask sg (B,) and applies `h6r = torch.where(sg[:, None, None], h6r.detach(), h6r)`, and the same on the h11 ':' row, before DM.write_seq.
- Items with d_i <= t (the answering pass and hold passes 3..8 on d 2) write live. Since one d per step, a d 2 step detaches pass 1 for the whole batch and a d 1 step detaches nothing.
- DeltaMem's own projections (qkv k/v rows, gate, ln_w) stay in the graph. Only the trunk's ':' activations are cut.
- mask None means unchanged code path.
- Under gamma = 1 the h6 detach is redundant, since (1 - gamma) = 0 exactly. It is kept for clarity.
- **C4 check:** no new loss term. Loss = la + 0 (c4_noaux). Eligible under Note 266-1.
- **Args diff vs ctrl_s1:** {sg_write 1}; sg_s2 also has {seed 2}.

### 4.3 ANC INIT-ANCHOR, `--anc 1.0` (anc_s1; stage 1; new arm, Note 269-4(d) route)

- **Change.** A frozen copy g_0 of the askname trunk is loaded strict (sha asserted, as build_aw 111-113) and runs under no_grad. On d 2 steps only, at pass 1, g_0 runs the pass-1 input xs[0] as a plain GPT-2 forward (no memory) and gives teacher logits z_0 over NUM_IDS at ':'.
- The student logits are z = m.colon_logits(h, pos)[:, NUM_IDS] from the live pass_aw at pass 1 (61-63; pass_aw already returns h, pos).
- Loss term: L_anc = mean_i KL(softmax(z_0,i) || softmax(z_i)) over the 11 number tokens. Total loss = la + anc * L_anc + 0.
- Equality at step 0 (DERIVED from code): read_fixed returns x + fade * W_o(...) (176), and fade = min(1, step / 500) (549). At step 0 the live pass-1 logits equal g_0's up to bf16, so L_anc ~ 0 at init. Gate G-ANC checks this.
- **C4 check, Note 269-4(d) conditions:**
  1. The target is the model's own earlier checkpoint (step 0 = the askname init). It is never a simulator value, and the solution graph is not used.
  2. The sub-question is the pass-1 cue input (text + QUESTION(u, S_1) + ' The answer is:'), i.e. the plan cue, which is test-time information.
  3. A pre-registered log of target accuracy vs the true v_1, `anc_tgt_acc` per 1k bin, is reported. Expected ~1.000 (G-AW0).

  Wording if it passes: **"C4-SD (Note 269-4(d) self-distillation), plan given"**, a weaker tier than the strict-route arms. Unresolved item U1: Audit must confirm that a step-0 teacher whose own lineage saw per-entity answers counts under 269-4(d) cond. 1.
- **What ANC is for.** ANC is the direct test of question (b). It holds the readout of the pass-1 code while P-direct stays open. If pass 2 still does not form from answers when the code is held, drift is not the only blocker.
- **Args diff vs ctrl_s1:** {anc 1.0}.

### 4.4 CA2a ONSET-FREEZE LP-FT, `--freeze_trunk_until 2000 --trunk_lr_mult 0.1` (lpft_s1; stage 1)

- **Change.** make_opt (fsq120 731-737) builds one AdamW group when lr_new == 1.0. Under the flag the trainer instead builds two groups:
  - mem: every m.mem parameter, mult 1.0;
  - trunk: every other parameter, {"mult": 0.1, "warm_from": 2000}.
- No new lr logic: lr_at (fsq120 740-743) already gives w_ = min(1, max(0, step - warm_from) / 300). So the trunk lr is exactly 0 for step <= 2000 and ramps linearly to 0.1x over 2001-2300.
- AdamW with lr 0 multiplies by 1 - 0 * wd and adds -0 * (m / denom), so the trunk is bitwise unchanged. Gate G-LPFT checks this.
- Pass 2 cannot form while the trunk is frozen: G-AW1 (Note 234-2) found that a frozen trunk does not use a stored v_1 (0.118 vs 0.107). The arm therefore only tests whether the drift is an onset shock: the first-bin grad norm is 7-9 under a fresh AdamW (Note 251-8). It does not test formation during the frozen phase.
- Disclosed: clip_grad_norm_ (566) still runs over all parameters, so during the frozen phase the mem group's clip scale includes the trunk's (unapplied) gradient, as in H0.
- **C4 check:** no new loss term. Eligible.
- **Args diff vs ctrl_s1:** {freeze_trunk_until 2000, trunk_lr_mult 0.1}.

### 4.5 Stage 1b (conditional: runs only if SG reads DRIFT in 2 / 2)

- **CA2b EMA WRITE SOURCE, `--ema_write 0.999` (ema_s1).**
  - An EMA copy of the trunk computes h6 / h11 at ':' under no_grad for the passes with d_i > t. Update: theta_ema <- tau theta_ema + (1 - tau) theta after opt.step (570). The EMA copy is added to the ckpt dict (torch.save 659-663) for resume.
  - DRIFT is defined on the EMA trunk's pass-1 ':' readout (new log key ws_readout_ema). The live row is descriptive, since nothing reads the live pass-1 code on d 2 (draft red team 4).
  - Not resumed to stage 2: at test, the model would need two trunks. The arm is a timing localiser only. If SG drifts and EMA holds, the interference is fast relative to 1 / (1 - tau) = 1k steps.
  - C4: no loss term.
- **CA4 SHARED-STEP SIAMESE, `--sia 1.0` (sia_s1).**
  - The partner of a d 2 item has the same S_0, givens, distractors, names, frames and S_1 sentence; only step 2 is re-drawn. It is built by a new helper in inject_aw from ow_words' _draw_program (45-85) law for one step and re-rendered as make() (108-122) does; OW.solve is asserted, and ow_words stays byte-identical.
  - The partner runs pass 1 only. L_sia = ||u_a - u_b||^2 / D + mean_j max(0, 1 - std_j(u)) on u = LN_w(h11[':']) at pass 1 (VICReg invariance + variance, 2105.04906).
  - The pairing uses which sentence is step 2, never a value. C4-eligible, with the pairing disclosed.
  - Read for DRIFT only. Invariance to the suffix does not force the code to be v_1.

### 4.6 What each arm can and cannot show

- **SG** removes P-direct exactly and leaves P-shared. NO-DRIFT under SG localises the drift to P-direct. DRIFT under SG localises it to P-shared or to op-coverage erosion (M4).
- **ANC** leaves both paths open and adds a restoring force on the readout. It reads question (b).
- **LPFT** delays and slows everything on the trunk. It tests the onset-shock account.

### 4.7 Not in this design's cells

- **CA3 MULTI-LEAF.** A leaf question's final answer v_j + c is a bijective image of v_j with c in the text. Under Note 266-1 that is a prefix answer by proxy, so it is NOT C4 as drafted. It may run only as descriptive work ($0.6), after Audit rules on "does a different question's final answer count as a step value when it is a bijective image of one?" (unresolved U4).
- **K18 EM-ACCUM.** This is §267's stage-B follow-up (EM with the accumulating state, beam-partial enumeration + RLOO 2402.14740; the DISTILL-TO-DM sub-arm needs a ruling). It belongs to §267's own registration after EM-C4 2/2 and is not costed here.

## 5. Mechanism and mathematical backing

Tags: PROVED (follows from the code or algebra), DERIVED-UNDER-ASSUMPTIONS (assumption named), CONJECTURE, DATA (measured, cited).

- **M1 (PROVED, graph).** Under SG on a d 2 step, the only pass-1 trunk outputs consumed downstream are the ':' rows h6[pos] and h11[pos] passed to write_seq (pass_aw 54-56). The block-11 hidden states at other positions are discarded, and the lower cache xs[0] is used only by pass 1 on d 2 items. So after the detach, d la / d theta_trunk receives no term from the pass-1 trunk forward. The grads equal those of running pass 1's trunk forward entirely under no_grad. Gate G-SG asserts this bitwise.
- **M2 (DERIVED-UNDER-ASSUMPTIONS; assumption: pass 2's read of S_1 is unformed at onset, DATA FD1: ws d 2 pass 2 0.02-0.13 at 2k).**
  - With fast_add, the additive heads accumulate S_a,K = S_a,0 + sum_t phi(k_t) v_t^T (write_seq 194-200). The decoder reads S_K (fwd_read).
  - Hence d la / d v_1 = phi(k_1)^T (d la / d S_a,K) != 0 whether or not pass 2 reads S_1. While pass 2 ignores S_1, this is the only memory route from u_1 to the answer.
  - Gradient descent then moves u_1 so that the decoder's read predicts a = v_2 directly. This is the observed drift, and it acts only on d > t items, since on d = t items the answer is v_t.
  - Prediction: cutting P-direct (SG) removes the force. ANC opposes it in the readout subspace only.
- **M3 (CONJECTURE).** The P-shared interference is small at onset.
  - First order: the pass-1 code moves by Delta u_1 = J_1(theta) Delta theta, with Delta theta = -eta P(clip(g_live)). Here g_live collects d 1 steps, pass 2 and hold passes, and the decoder; P is Adam's preconditioner.
  - No bound is available without measuring J_1 J_live^T.
  - For: pass-1 and pass-2 inputs differ in the cue name, and in the trained of loop the cue name selects the computed entity (NAME-SWAP j = 1 follow 1.000, Note 263-3; this is DATA, but on of, not askname).
  - Against: the pass-2 input on d 2 is the same text as pass 1 with another cue, so any cue-insensitive feature trained toward v_2 bleeds into pass 1. The new log key ws_bleed (P(pass-1 argmax == v_2) on d 2) measures this.
- **M4 (PROVED from code; consequence DERIVED).** Op coverage of the d 1 control:
  - ow_words _draw_program line 70: `op = "mul_v" if d == 1 else reg[...]`. Every d 1 item's step 1 is mul_v.
  - d 2 items draw step 1 from OPS_REG minus add_s (add_s is rejected at t < 2, lines 72-74): six ops, so ~5/6 of d 2 items have a non-mul_v step 1.
  - Hence ws_readout['1']['1'] (the "still-present" control) covers only mul_v. Under SG, the pass-1 code for the five other ops gets no direct gradient at any step, so nothing anchors it except the init.
  - Consequence: if the drift is erosion of an unanchored code rather than an answer pull, SG will still drift, and the drift will concentrate on non-mul_v items.
  - Descriptive log key ws_op1 = pass-1 accuracy on the d 2 WS items split by step-1 op {mul_v, other}. About 33 / 167 items, so its CIs are wide and it is descriptive only.
  - Stage 0a pre-states the same split on ao_s1-s4's stored eval_ws items at $0 (sec. 12).
- **M5 (DERIVED; assumption: KL in the 11-token readout subspace transfers to the write).** ANC's gradient reaches h11[':'] through ln_f and the LM head (colon_logits 61-63). The write reads LN_w(h11[':']), the same row. Holding the 11-way readout holds h's projection on the number-token directions of the LM head, not all of h. NO-DRIFT under ANC is therefore a statement about the readout. probe u1|v1 at 12k (inject_aw_eval part_probe 313-349) is the descriptive check that the write source carries v_1.
- **M6 (PROVED, optimizer).** LP-FT's frozen trunk is bitwise frozen (sec. 4.4). After unfreeze, the trunk's Adam step size is 0.1x H0's at equal moments, so DRIFT at 4k under LPFT is ambiguous between "slower" and "none". FINAL-HOLD at 12k decides (sec. 8).
- **M7 (DATA, base rates).**
  - Answer-only formation by 12k on this harness: 0 / 4 seeds. ao_s2 formed only at 18k with an LR re-warm, Note 244-5.
  - Formation with per-pass aux: aw_plaw2_s1 and aux_s2 formed at 7k (Note 269-1).
  - Answer-only extension with held seeded passes: rare, EXTEND 1 / 4 and d 4 accuracy 2 / 4 (Note 255-1, which supersedes the draft's "2 / 2").

## 6. Read windows (Note 269-1 FD1)

- **Stage 1 (1k / 2k / 4k), binding:** only readings of the already-present code. These are ws_readout['2']['1'] (DRIFT), with ws_readout['1']['1'] as the d 1 guard.
- **Not binding at <= 4k (trend only):** ws_readout['2']['2'], train by_d['2'].acc, aux vacc, any K' read.
- **Formation:** read bindingly only at the 12k final, from eval_* files on n = 1000.
- **lpft_s1:** its DRIFT rows are 3k and 4k (>= 1k after unfreeze). Its 1k / 2k rows are trivially 1.0 and are not evidence.
- **Raw values:** all rows are the raw logged values (no smoothing, Note 267-1 (3)).

## 7. Predictions (Theory's, before any output)

| # | Prediction | P | Tag |
|---|---|---|---|
| P1 | ctrl_s1 reads DRIFT (a row <= 0.30 at 2k or 4k) | 0.90 | DATA-based (AO 4/4, P-CTRL 0.125 at 2k) |
| P2 | sg_sX NO-DRIFT per seed | 0.45 | M2 vs M3/M4 |
| P2b | SG NO-DRIFT in 2 / 2 | 0.38 | seeds correlated |
| P3 | anc_s1 NO-DRIFT | 0.85 | M5 |
| P4 | lpft_s1 NO-DRIFT at 3k / 4k | 0.35 | M6 |
| P5 | If an SG cell drifts, ws_op1 other - mul_v <= -0.30 (the drift concentrates on non-mul_v ops) | 0.60 | M4 conjecture |
| P6 | ws_bleed >= 0.30 in a drifting cell at 2k (pass 1 prints v_2) | 0.35 | M3 |
| P7 | AO-FORMED at 12k, per resumed cell | 0.25 | M7 |
| P7b | Any of sg_s1 / sg_s2 / anc_s1 AO-FORMED | 0.40 | |
| P8 | A resumed cell is ACC-ONLY (accuracy without FORM-WS2 or STATE-USE: the decoder last-hop route, as in ao_s2_ext) | 0.15 | DATA (Note 251-9) |

## 8. Binding criteria, labels and decision rules

**Statistics.**
- Wilson 95% CIs: wilson(p, n) in theory/score_em.py 48-50, or theory/statlib.py once it lands.
- NEAR: near(p, n, bar) = |p - bar| <= 2 sqrt(p(1 - p) / n) (score_em 53-54).
- Every proportion-bar cell gets its value, n, CI, bar and NEAR flag.
- A NEAR cell is claimed neither way.
- Differences (LP5 drop) and STABLE are raw booleans (the §270 convention).

Reference widths: n 200 at 0.80 gives a CI of [0.739, 0.850] and a NEAR half-width of 0.057; at 0.90, [0.851, 0.934]; at 0.30, [0.241, 0.367]. n 1000 at 0.90 gives [0.880, 0.917].

**Stage 1 (per cell; log.jsonl 1k-bin rows; n = 200 per row, WS_ITEMS = TEST2's first 200 per d, inject_aw 451):**

| Criterion | Field | Rows | Bar | Kind |
|---|---|---|---|---|
| PRESENT-1 | ws_readout['1']['1'] | 1k, 2k, 4k (lpft 3k, 4k) | >= 0.95 every row, else VOID (the harness lost d 1) | guard |
| DRIFT | ws_readout['2']['1'] | same | **NO-DRIFT** if every row >= 0.80 and none NEAR(0.80). **DRIFT** if any row at >= 2k is <= 0.30 and not NEAR(0.30). **NEAR** if no DRIFT and every row is >= 0.80 or NEAR(0.80). Else **PARTIAL**. | binding |
| ANC-TGT | anc_tgt_acc (anc only) | every bin | reported (269-4(d) cond. 3); < 0.95 at any bin means VOID-ANC (the teacher is not the init) | guard |
| ctrl validity | ctrl_s1 DRIFT label | | must be DRIFT, else VOID-HARNESS for the family | guard |
| descriptive | ws_readout['2']['2..8'], by_d['2'].acc, ws_op1, ws_bleed, write_grad_norm, gn | 1k-4k | trends only (FD1) | descriptive |

Pooled SG: "SG NO-DRIFT k / 2". Paired margin per row vs ctrl_s1 (same items, same d sequence), descriptive.

**Stage 2 (per resumed cell, final at 12k; evals from eval_*.json, n 1000; field paths as theory/score_ao.py seed 58-104):**

| Criterion | Field | Bar | Kind |
|---|---|---|---|
| LP1 | last log bin by_d['1'].acc | >= 0.90 | binding |
| LP3 | eval_acc by_d['2'].acc | >= 0.70 and >= CD-AN (aw_pcd2_s1) + 0.30 | binding (Wilson, NEAR on 0.70) |
| FINAL-HOLD | eval_ws by_d['2']['1'] | >= 0.80 | binding (the 12k drift check, all cells) |
| FORM-WS2 | eval_ws by_d['2']['2'] | >= 0.90 | binding (mechanism at the claimed depth; memory "gate on the mechanism") |
| LP5 | eval_lp5 by_d['2'] intact[kb] - mean_at_1[kb] | >= 0.30 | binding (raw) |
| MEAN-STATE | eval_mean_s by_d['2'].pooled | <= CD-AN + 0.10 | binding |
| STATE-USE | eval_swap_k: n_pairs >= 1000, follow_U | >= 0.50 and >= floor + 0.30 | binding (Wilson on follow_U, n = n_pairs) |
| CUE-ONLY | eval_cue_only by_d['2'].answer_acc | <= max(majority, DIRECT) + 0.05, else VOID | guard |
| STABLE | score_ao.stable (28-55), base = the stage-1 dir | n_nonfinite <= 10, no unrecovered 3x rise | guard |
| descriptive | probe u1 / v1, read1 / v1 (eval_probe), eval_ws passes 3-8, acc_kprime d 2, cue_wrong drop, by-bin curves vs ao_s1-s4 / aw_plaw2_s1 / aux_s2 | | descriptive |

**Labels per cell, first match:**
1. VOID
2. UNSTABLE
3. LOST-CODE (FINAL-HOLD fails)
4. NOT-FORMED (LP3 or LP1 fails)
5. ACC-ONLY (FORM-WS2, LP5, MEAN-STATE or STATE-USE fails)
6. **AO-FORMED** (all binding pass)

A NEAR on any binding proportion turns the label into NEAR-<criterion>. Cell wording: "answer-only d 2 formation with <arm> (askname init, plan given, one lineage)". For ANC the wording is prefixed "C4-SD".

**Decision rules (pre-stated):**
- **R0, stage gate:** resume a stage-1 cell to 12k iff it is NO-DRIFT or NEAR. lpft_s1 resumes unless it is DRIFT (M6). ctrl_s1 is never resumed; ao_s1-s4 are the 12k answer-only comparators.
- **R1:** SG NO-DRIFT 2 / 2 with ctrl DRIFT gives "the onset drift is carried by the direct write path (P-direct)" (DATA, scoped).
- **R2:** SG DRIFT 2 / 2 with ctrl DRIFT gives "the drift survives without P-direct: shared-weight interference or unanchored-op erosion". ws_op1 tells which, descriptively, and stage 1b (EMA, SIA) runs.
- **R3:** SG split (one NO-DRIFT, one DRIFT / PARTIAL) gives SPLIT. No localisation claim. Two more SG seeds need a Note.
- **R4 (formation):** any AO-FORMED cell leads to stage 3, two more seeds of that arm from scratch to 12k (registered separately, pre-named here). An AO-FORMED SG cell also pre-names the stage-4 of_s1 transfer (sec. 2).
- **R5:** if every resumed cell is FINAL-HOLD-pass but NOT-FORMED / ACC-ONLY, then "holding the seeded code is not sufficient for answer-only formation of pass 2 by 12k in this harness (not 'cannot': ao_s2 needed 18k)". The continuous-DeltaMem answer-only line parks, and §267 (categorical EM) is the remaining strict C4 route. The pre-named follow-up is SG released at 4k (SG on for 1-4k, then off), with no registration here.
- **R6:** ctrl_s1 not DRIFT gives VOID-HARNESS. No stage 2. A Note investigates the module difference against ao_s1.

## 9. OC block

Simulated in the scratchpad (oc_c4b.py, 20k draws per row profile, n 200 per row, the sec. 8 rule including NEAR).
- Per seed, P(NO-DRIFT):
  - true rows like AO (0.20 / 0.15 / 0.12): **0.000**;
  - weak partial (0.6 / 0.5 / 0.45): 0.000 (PARTIAL);
  - flat 0.80: 0.000 (NEAR 0.95);
  - flat 0.85: 0.10 (NEAR 0.90);
  - flat 0.88: 0.69;
  - flat 0.90: 0.95;
  - held 0.97 (the aux comparator's level): **1.000**;
  - slow drift (0.95 / 0.85 / 0.70): 0.000 (PARTIAL 0.88).
- So the NO-DRIFT claim effectively needs a true row of about 0.90 or more. A code that drifts to the 0.80s reads NEAR, never NO-DRIFT.
- **SG primary claim (R1, NO-DRIFT 2 / 2):**
  - P(pass | no effect, SG = AO) < 0.001;
  - P(pass | effect, code held at >= 0.95) ~ 1.00;
  - P(pass | partial effect, true 0.85) ~ 0.01.
  - Under Theory's prior (held 0.45 / partial 0.20 / drift 0.35), P(NO-DRIFT) per seed is 0.45, and P(2 / 2) ~ 0.38 with the seed correlation assumed in P2b.
- **ANC (single seed):** P(NO-DRIFT | no effect) < 0.001; P(NO-DRIFT | anchored) ~ 1.00.
- **AO-FORMED per cell (stage 2, n 1000):**
  - P(pass | no formation effect) ~ 0.05-0.10. Answer-only base rate by 12k is 0 / 4; Wilson 95% upper bound 0.49, Jeffreys mean 0.10.
  - P(pass | forms like aux_s2: acc 1.0, WS2 1.0, swap 0.95+) = 1.000 (simulated).
  - P(pass | forms like ao_s2_ext at 18k: acc 0.967, WS2 ~0.85, swap 0.656) = 0.000. It fails FORM-WS2, so a late, partial, decoder-route formation is labelled ACC-ONLY, never AO-FORMED.
  - With one or two seeds per arm, an AO-FORMED reading is "formed in k / n seeds", and stage 3 is required before any rate claim.
- **Family error budget:** binding claims are R1 (SG localisation) and AO-FORMED per arm. ANC and LPFT are single-seed and are read as localisers or existence checks, never as recipe comparisons (Note 269-1 ST2).

## 10. Censoring (conservative)

- A stage-1 cell missing any of its binding rows (1k / 2k / 4k; lpft 3k / 4k), or crashing before 4k: VOID-RERUN once on a fresh host. A second failure is scored **DRIFT**, the unfavourable label, never NO-DRIFT.
- A stage-2 resume that crashes: VOID-RERUN once from the kept 4k ckpt. A second failure is **NOT-FORMED**.
- A crashed eval is re-run once on the kept model.pt. If it is still missing, every criterion that eval feeds counts as failed.
- STATE-USE with n_pairs < 1000 fails (score_ao.seed 76).
- More than 10 non-finite steps means UNSTABLE (fails the arm claim).
- OOM: rerun with the same micro 1 pinned. No micro change without a Note.
- Cells labelled DRIFT at stage 1 are not resumed. Their 4k ckpts are deleted after the stage-1 score is committed and read (memory "keep gate ckpts until read"); logs and args are kept.
- No bar is edited after the first output.

## 11. Seeds

| Cell | Seed | Pairing |
|---|---|---|
| ctrl_s1 | 1 | exact pair for every seed-1 cell (same d sequence, item stream and DeltaMem init) |
| sg_s1, sg_s2 | 1, 2 | sg_s1 vs ctrl_s1 exact; sg_s2 vs ao_s2 approximate (module / EXCL differ; ao_s2 is the one AO seed that formed, at 18k) |
| anc_s1, lpft_s1 | 1 | vs ctrl_s1 exact |
| stage 1b ema_s1, sia_s1 | 1 | vs ctrl_s1 |
| stage 3 (pre-named) | 3, 4 | new data orders, two seeds of an AO-FORMED arm |

All cells run on RTX 3090 only (GPU class flips basins, memory "basin entry"); train_gpu is recorded per cell. One lineage (askname). Disclosed: Note 269-2's >= 3-seed rule is for C2 claims and does not bind here; any formation-rate wording waits for stage 3.

## 12. Stages, cost, guards (Vast RTX 3090 at ~$0.21 / h)

Step-time basis:
- **expected** 0.94 s/step: the worst realised H0 rate of 0.85 s/step (ao_s4) + 10% arm overhead (ANC teacher forward on d 2 steps; SG ~0);
- **ceiling** 1.65 s/step (the caller's rate for the larger harness), used for every guard.

Fixed overhead per cell is ~20 min: set build 27 s, host probe, mem_check, and the ckpt rename + 1.5 GB drain.

| Stage | Content | Expected | Ceiling | Guard |
|---|---|---|---|---|
| 0a | $0 pre-read (pre-stated now, pushed to Audit, computed only after Audit's ack; memory "pre-state, push, wait for ack"). On ao_s1-s4's stored eval_ws d 2 items: pass-1 accuracy split by step-1 op (mul_v / other) and P(pass-1 pred == v_2). Reading: ANCHOR-GAP if other <= 0.20 while mul_v >= 0.80; ANSWER-PULL if both <= 0.30 and the v_2 share >= 0.30; else MIXED. Predictive only, never a label. | $0 | $0 | - |
| 0 | Build + gates G0-G8 (sec. 13) on a Vast 3090, since local GPU work is on hold after the RAM bit flips (Notes 255-6 / 255-7) | $0.25 | $0.40 | $0.5 |
| 1 | ctrl_s1, sg_s1, sg_s2, anc_s1, lpft_s1; --stop_at 4000 (exit 9; the host renames ckpt_last.pt to ckpt_4000.pt so the drain fetches it, since the drain skips ckpt_last.pt, inject_aw 686) | 5 x $0.29 = $1.4 | 5 x $0.46 = $2.3 | **$2.5** |
| 1b | only on R2: ema_s1, sia_s1, 4k | 2 x $0.33 | 2 x $0.50 | $1.2 |
| 2 | <= 4 resumed cells, 4k -> 12k in a new dir <cell>_r (--resume ckpt_4000.pt, args.json shipped, same --steps 12000, stop_at 0; memory "resume extension in a new dir") + the 9 eval parts | 4 x $0.58 = $2.3 | 4 x $0.91 = $3.6 | **$4.0** |
| 3 | pre-named, separate registration: 2 seeds x 12k from scratch of an AO-FORMED arm | 2 x $0.85 | 2 x $1.30 | $3.0 |

This registration's family guard is stages 0 + 1 + 1b + 2 = **$8.2** (expected ~$4.6 if 1b runs, ~$3.9 if not). With stage 3 it is $11.2. Both are within Audit's $30 auto-approve rule.

**Exact args diff (vs boxes/runs_fsq/ao_s1/args.json, args_ref = that file):**
- **Every cell:** {seed, out, stop_at 4000, auto_micro false (micro 1 pinned, as ao_s1 resolved; auto_micro headroom rule), c4_noaux 1}.
- **Module keys added after §244 at inert defaults (disclosed non-semantic; each passed its own default-off bitwise gate):** base gpt2, perm_p 0, cue_ans 0, aux_d_max 0, train_frames '', distract_p 0, distract_mix '', distract_nmax 2, hold_* defaults, init_model '', nk 0, ask_p 0, ask_perm_p 0.5 (inert at ask_p 0).
- **Disclosed:** excl_n differs from ao_s1's (later frame test sets in EXCL, as in §254's disclosure). This is why ctrl_s1 exists.
- **Per arm:**
  - ctrl: none;
  - sg: {sg_write 1};
  - anc: {anc 1.0};
  - lpft: {freeze_trunk_until 2000, trunk_lr_mult 0.1};
  - ema: {ema_write 0.999};
  - sia: {sia 1.0}.
- **Stage 2 vs the stage-1 cell:** {resume, out, stop_at 0}.

The args diff is committed with each cell, and a single-change check is run against ctrl_s1's args.json (memory "single-change claims need an args diff").

## 13. Gates (stage 0; one JSON per gate, incremental, memory "gate scripts")

- **G0 default-off bitwise.** All new flags off (sg_write 0, anc 0, freeze_trunk_until 0, ema_write 0, sia 0, c4_noaux 0) equals the current module bitwise on a 100-step replay of ao_s1's args under the det wrap (same card, module before vs after; the §251 / §252 form). deltamem.py, ow_words.py and fsq120.py stay byte-identical (deltamem_sha256 / ow_words_sha256 asserted in args.json, inject_aw 494).
- **G-C4a.** c4_noaux 1 vs 0: bitwise-equal parameters after 100 steps (det wrap). Under c4_noaux, `aux.requires_grad is False` and no autograd node holds an R_step-derived tensor (walk loss.grad_fn on one batch per d).
- **G-C4b, every arm.** On 3 batches per d, grad(loss) == grad(la + the arm's declared term) bitwise. This lists every term in `loss` (scratch/gradflow_check.py term() pattern, 26-28; read from the decorators down).
- **G-SG.** On a d 2 batch, SG grads == grads with pass 1's trunk forward under no_grad (bitwise). The retain_grad on pass-1 h11[':'] is exactly 0. Grads on mem.qkv k/v rows, gate and ln_w are nonzero. On a d 1 batch, SG == H0 bitwise.
- **G-ANC.**
  - At step 0, teacher logits == live pass-1 colon logits over NUM_IDS (rel. diff <= 1e-3, bf16); L_anc < 1e-3.
  - anc_tgt_acc on WS_ITEMS d 2 >= 0.99 (G-AW0).
  - L_anc alone has nonzero trunk grads and zero grads on mem.qkv k/v rows, gate and ln_w (it never reaches the write).
- **G-LPFT.** Trunk params bitwise unchanged after 20 steps at step <= 2000; mem params changed. Per-group lr at steps 2000 / 2001 / 2150 / 2300 equals 0 / 0.1 x lr_at x 1/300 / ... / 0.1 x lr_at (the warm_from ramp).
- **G-mem.** mem_check (466-490) at the d 2 shape, micro 1 pinned, every flag. Peak <= 20 GiB (1 GiB headroom).
- **G-smoke.** 200 steps at full data scale per flag. The 1k-bin keys exist in the smoke log under --sched_scale: ws_readout, ws_op1, ws_bleed and (anc) anc_tgt_acc.
- **G-resume.** 4000 -> 4100 resume reproduces the un-stopped run's step-4100 loss row bitwise (det wrap), for ctrl, sg, anc and lpft (the last one checks the restored param-group mult / warm_from; resume path 514-528).
- **G-host.** Python 3.11 (no 3.12+ APIs; memory "hosts run Python 3.11").

## 14. Scorer (theory/score_c4b.py; fixtures before any output)

- **Inputs:**
  - stage 1: boxes/runs_fsq/<cell>/log.jsonl rows step in {1000, 2000, 3000, 4000}, giving ws_readout['1']['1'], ['2']['1'], ['2']['2'], by_d['2'].acc, ws_op1, ws_bleed, anc_tgt_acc;
  - stage 2: <cell>_r eval_acc / eval_ws / eval_lp5 / eval_mean_s / eval_swap_k / eval_cue_only / eval_probe / train_summary, with comparators aw_pcd2_s1 (CD-AN) and s228_pdir2c_s1 (DIRECT) as in score_ao 17.
- **Imports:** score_ao.stable, and wilson / near from score_em (statlib once it lands).
- **Output:** per-cell {rows with CI and NEAR, DRIFT label, stage-2 criteria, label}, the pooled "SG NO-DRIFT k / 2", and the R0-R6 routing line, written to theory/results/score_c4b.json. The scorer is run twice and must be byte-identical.
- **Fixtures (12):**
  1. no-drift + formed;
  2. no-drift + not-formed;
  3. drift at 2k;
  4. NEAR row at 0.80;
  5. PARTIAL;
  6. lpft read at 3k / 4k only;
  7. missing 4k row -> VOID-RERUN, then DRIFT on the second miss;
  8. unrecovered rise -> UNSTABLE;
  9. accuracy without WS2 -> ACC-ONLY;
  10. FINAL-HOLD fail -> LOST-CODE;
  11. ctrl not DRIFT -> VOID-HARNESS;
  12. anc_tgt_acc < 0.95 -> VOID-ANC.
- Every field path read is grepped in the module before the fixtures are committed (memory "scorer field-path check").

## 15. Code touch points (verified 2026-11-04 by reading the lines)

- **boxes/inject_aw.py:**
  - InjectAW.pass_aw 47-56 (sg mask / EMA source on the ':' rows before write_seq 56); write 58-59; colon_logits 61-63 (ANC student);
  - build_aw 109-125 (strict init load with sha 111-113: the pattern for g_0 and the EMA copy);
  - ws_readout 134-151 (per_item preds -> ws_op1 / ws_bleed);
  - argparse 212-250 (new flags, defaults off); inject asserts aux_step > 0 at 259; HOLD 289; lam_at 299-302;
  - WS_ITEMS 451; loss_reg 361-419 (pass loop 372-393, m.write 386, aux block 394-403 under c4_noaux, la 404-405, loss 407); loss_items 433-441;
  - make_opt call 508 (param groups for LPFT); resume 514-528; g_add calibration 529-541;
  - lr loop 545-546; fade 549; gamma 550; clip 566; opt.step 570 (EMA update); 1k-bin ws log 632-633; ckpt torch.save 659-663 (+ EMA state);
  - stop_at exit 9 671-672; keep_final_ckpt rename 686.
- **boxes/inject_ow.py:** tok_ow 63-71 (aux = R_step); dec_logits 86-90; aux_logits 92+.
- **boxes/deltamem.py:** read_fixed 158-176 (fade at 176), write_seq 179-223 (u 186-187; fast_add 194-203). deltamem stays unchanged.
- **boxes/fsq120.py:** make_opt 731-737, lr_at 740-743 (mult, warm_from).
- **boxes/ow_words.py:** _draw_program 45-85 (line 70: d 1 is mul_v only; 72-74: add_s skipped at t < 2), make 108-122 (SIA partner law).
- **boxes/inject_aw_eval.py:** part_swap_k 255, part_probe 313-349, part_ws 352-356, run 1297.
- **theory/score_ao.py:** stable 28-55, seed 58-104 (stage-2 field paths), CD / DIR 17.
- **theory/score_em.py:** wilson 48-50, near 53-54, onset 57-62.
- **scratch/gradflow_check.py:** term 26-28.

## 16. Literature (ids from the 2026-11-03 draft, titles checked then; new ones marked)

- 2502.05171 Geiping et al., recurrent depth with truncated backprop through the loop: the SG analogue.
- 2506.21734 Wang et al., Hierarchical Reasoning Model, and 2510.04871 Jolicoeur-Martineau, Tiny Recursive Models: deep supervision with detached segments, the closest training-side precedent for SG-WRITE.
- 2202.10054 Kumar et al., LP-FT.
- 2006.07733 BYOL; 1703.01780 Mean teachers.
- 2011.10566 SimSiam; 2105.04906 VICReg.
- 2402.14740 RLOO; 2312.02179 TRICE; 2409.15647 Fan et al., looped transformers (loss only at T(n)); 2007.01919 Correia et al.
- 2407.20311 Ye et al., Physics of LMs 2.1 (the motivation for CA3, not a method).
- 2610.27628 PAR, prefix self-distillation (the ANC route; Note 269-4(d)).
- **New, NOT fetched this session (Literature to verify ids and titles before registration):** 1606.09282 Li and Hoiem, Learning without Forgetting (KL to the old model's outputs, the ANC form); 1802.01483 Li et al., L2-SP (anchoring to the pretrained point).
- Novelty NARROW (Literature, docs/LIT_C4DRIFT_SG_2026-11-04.md; replaces the draft's "not found"): SG-WRITE detaches the pass-t write source for items not yet answered at pass t, a per-item variant of the truncated backprop of 2502.05171 / 2506.21734 / 2510.04871 / 2609.19717, used here as a localisation test of answer-gradient drift. Looped Flows 2609.11801 (App. B) conjectures the same mechanism without testing it. Ours adds the per-item mask (d_i > t), drift localisation, and a loss-free detached pass (M4's erosion branch rests on it). ANC is LwF-style KL (1606.09282) and LPFT is 2202.10054: no novelty claimed for either. 1802.01483 (L2-SP) verified.

## 17. Known risks (the lead's own critique; an independent red team is still required)

1. **SG may block formation as well as drift.** With P-direct cut, pass 2's use of S_1 must form from the decoder gradient alone. aex0_s2 did not form d 3 that way in 16k. NOT-FORMED is then informative, and SG-release is pre-named.
2. **Hold passes 3..8 write live** and re-read the S_2 cue, so the decoder shortcut can re-enter through them. FORM-WS2 and STATE-USE catch it. ws_readout['2']['3..8'] is already logged at every 1k bin.
3. **The ws_readout['1']['1'] guard covers only mul_v (M4).** A PRESENT-1 pass does not show that the other five ops' pass-1 codes are intact; ws_op1 and the stage-0a read cover this descriptively.
4. **ANC holds the readout, not the full write source (M5).** A NO-DRIFT ANC with probe u1|v1 < 0.80 at 12k is reported as "readout held, write source drifted".
5. **The 12k horizon may be too short** (ao_s2 needed 18k + re-warm). NOT-FORMED means "not by 12k". There is no STILL-RISING extension; any extension needs a new Note.
6. **lpft's 0.1x trunk may simply slow the drift (M6).** FINAL-HOLD at 12k decides.
7. **Single seeds** (anc, lpft, 1b) localise only.
8. **Dead-zone of the NO-DRIFT rule:** a true 0.80-0.88 code reads NEAR (sec. 9). This is accepted as conservative.

## 18. Unresolved items

- **U1:** Audit ruling that ANC's step-0 teacher (askname lineage) satisfies Note 269-4(d) cond. 1. Until then ANC reads "self-distillation route, C4 status pending".
- **U2:** Audit ack for the stage-0a $0 read before it is computed.
- **U3:** an independent red team of this completed doc. The draft's red-team agent never ran (concurrent sub-agent limit, 2026-11-03).
- **U4:** the CA3 ruling (sec. 4.7).
- **U5:** Literature verification of 1606.09282 / 1802.01483 and the novelty spot-check.
- **U6:** the build owner and the queue slot relative to §267 (both are lane B; neither is built as of CURRENT_TASKS 01:5x 11-04).
- **U7:** confirm on the first stage-1 host that the realised s/step is <= 1.65. If it is higher, the guards scale and Audit is told before stage 2.
