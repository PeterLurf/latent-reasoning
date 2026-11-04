Status: completed 2026-11-04 by Theory's agent; not red-teamed yet

# Wave-4 design 07: the plan-free lane (P9 CUE-FADE, H8 ANSWERABLE-NOW, TH5 DATAFLOW-ALL-WRITE; TH2 decision rule)

Draft by the lead design agent, 2026-11-03 (cut off). Completed 2026-11-04 01:3x EST by Theory's agent. Read-only on the repo: no training, no numbers computed from stored run outputs. The OC numbers come from a synthetic simulation (§10).

Covers backlog IDs **P9**, **H8** (H7 TC-HALT is H8 on a target-cue model), **TH5**, and **C1 PLAN-PROBE** (re-specified per red team 06 finding 11). **TH2** is the decision rule. This is Lane C of docs/EXPERIMENT_BACKLOG.md §1 (l. 56-62).

## 0. What changed since the draft

1. **Base is of_s1, not bun_s1.**
   - §263 scored ORDER-FREE-C1 2/2 (Note 263-3): TEST-PERM d 4 0.982 / 0.961. NAME-SWAP shows the cue name selects among already-computed entities: re-cue of S_1 follows 1.000, jump-ahead is at the floor. Full all-sentence shuffle gives d 4 0.46 / 0.38, so order-freedom covers op order only.
   - The draft's own rule ("all from of_s* if §263 reads ORDER-FREE-C1") applies. Every arm's args are written as a diff against boxes/runs_fsq/of_s1/args.json (§5).
2. **The stop is model-decided, never harness-decided (red team 06 finding 1).**
   - Every plan-free arm runs passes whose inputs are a function of the text alone. They are byte-identical across passes, which is asserted (§3.4 D-BLIND).
   - A stop is read only from model outputs:
     - the hold (the answer at K' 8 / 16 / 64);
     - the C5 r = 3 rule on the decoder's own answers;
     - the H8 head on the model's ':' hidden.
   - The harness never sees it["d"], the spine or A_q when it builds an eval input.
   - The draft's TH5 halt "r_t = ||S_n q_{S_d}|| >= theta" had the harness query the state at A_q's key. It is demoted to a descriptive mechanism read.
3. **Binding branch rows are on TEST-PERM renders, and question use is measured by TWIN pairs (red team 06 findings 1 and 4; backlog §12.5 l. 443).**
   - On chain-order text, "next sentence's subject" solves the plan. So every binding family row is built on TEST-PERM d 4 items (perm, then insert). Clean renders are descriptive.
   - A twin pair is the same text with the question re-pointed at the competing branch. It makes the TH2 bound exact (M1).
4. **PLAN-PROBE is re-specified fork-split on the of_s1 trunk (red team 06 finding 11).** It is stage 1, the cheapest decisive step. P9 parks on its failure, under assumption M4.
5. **§261 TARGET-CUE stays parked.**
   - Notes 261-1 and 262-3: parked by Audit pending §263.
   - P9-T at p = 1 is §261's condition in a 7-token form. The pilot's TC-ABRUPT arm (p = 1 from step 1, no KD) is §261's no-curriculum control on of_s1.
   - Proposed: §261 superseded by this design's S2. That is an Audit ruling.
6. **§270 FULL-PERM and §271 N1 were registered since the draft.**
   - Both take of_s1 as init (§270 runs from bun_sX_init with of_sX as the control; §271 runs LOOP-SI from of_s1).
   - This lane does not change if §270 passes. Its args freeze at registration with init of_s1. A rebase on fp_s* would be a separate follow-up (one change at a time). `full_perm` is added as a descriptive eval part.
   - Queue order: after §271 S1 and §270, which were registered first. The lane runs in parallel when hosts allow.
7. **Seeds and claims.** Every C2 claim needs 3 seeds. The full P9 / TH5 stages run 3 seeds from of_s1 (seed = data order and coins; one lineage). C1 claims at >= 2/3 seeds; C2-RETAIN needs 3/3.
8. **Seed family 26300 collides.** It is already SEED_NS in boxes/of_parts.py:15. Every new seed is from 27070-27079: grep over boxes/*.py, scratch/*.py, theory/*.py and docs/THEORY.md on 2026-11-04 found no use.

## 1. Goal and the TH2 decision rule (kept; sharpened)

Note 260-5: gb_s1 / s2 have no plan in the cue.
- Clean-chain C1 holds: TEST2 d 1-4 1.000, SWAP4 0.985.
- But the loop selects by **forward reachability**:
  - H-EXTEND 0.08 / 0.02;
  - H-LIVE 0.40 / 0.41;
  - H-CHAINSRC-LIVE 0.06 / 0.41;
  - d 7 0.25 / 0.07.
- The bun lineage, with the plan, reaches 0.92 / 0.99 / 1.00 on these.

**The lane's question.** Which plan-free design, on the order-free of base, restores stopping at the asked entity, rejection of live distractor chains, and extrapolation? The model must make these decisions itself.

**TH2 mirror-pair bound.**
- Model setup: a plan-free policy at pass t picks one of the b_t sentences that read already-computed entities. It decides from (text, S_{t-1}) without using the question.
- On the branch families, the correct branch and the distractor branch are exchangeable under renaming. The question is the only thing that tells them apart. The families are boxes/insert_render.py:121-123 (H-CHAINSRC), 135-138 (H-EXTEND), 139-143 (H-LIVE), and 233 / 248-250 (H-CHAINSRC-LIVE).
- So a forward-only policy has P(correct chain) <= prod_t 1/b_t, averaged over the exchangeable pair. "Stop at S_d" against "continue to X1" is also such a choice. The per-family bounds:

| Family | Structure | Branch point | Per-item bound | gb (s1 / s2) |
|---|---|---|---|---|
| H-EXTEND | X1 = S4 + c1, X2 = X1 + c2, after S4 | stop vs continue at pass 5 | 1/2; 0 for a policy that never stops | 0.08 / 0.02 |
| H-LIVE | given Y; X1 = Y + c1; X2 = X1 + c2 | start: S_0 vs Y at pass 1 | 1/2 | 0.40 / 0.41 |
| H-CHAINSRC-LIVE | X1 = S_j + c right after S_j; X2 = X1 + c' | b_{j+1} = 2 | 1/2 | 0.06 / 0.41 |

- A **question-conditioned** policy can reach 1:
  - "current entity == target" for the halt;
  - "is an ancestor of S_d" for selection.
- The text already ends with the question (nested_words.py:156, `text = " ".join([render(x) for x in seq] + [q])`), so every plan-free model sees S_d. gb did not use it.
- The TWIN construction (§3.2) turns the bound into an exact pair statement (M1).

**Decision rule.** A design gets a full stage on the branch families only if its selection or halt is question-conditioned. That can come by construction (TH5: nothing to select; H8: a one-hop halt check) or by training pressure (P9: the teacher's plan is distilled into a student that has only the question).

**Pre-stated routing:**
- H8 repairs H-EXTEND but not H-LIVE / CHAINSRC-LIVE. The halt is one-hop; selection is not.
- P9 can pass all families only if one of_s1 pass can compute is-ancestor-of-S_d at d <= 4. The stage-1 PLAN-PROBE-F reading is the shared falsifier (M4).
- TH5 passes the families without any selection. Its risks are capacity and formation.

## 2. Math backing (claim tags: PROVED / DERIVED-UNDER-ASSUMPTIONS / CONJECTURE / DATA)

**M1 PROVED: the twin null.**
- Setup: a twin pair (a, b) is one text with the question naming the end of branch A or of branch B (§3.2). Take any eval-time map whose pass inputs are identical for a and b. That holds for any question-blind policy, and exactly for Q-ABLATE inputs (§3.3).
- Then the pass-t ':' readout is identical for both twins. If v_A(t*) != v_B(t*) at the decision pass t*, at most one twin is right there. So **':' pair-accuracy = 0**, and the per-item accuracy averaged over the two twins is <= 1/2.
- This is pigeonhole. The draft's prod 1/b_t is the expectation of this bound over a random tie-break.
- Corollary (PROVED): under Q-ABLATE, ':' pair-accuracy is exactly 0. G-TWIN asserts it as a builder check, so the control is exposed and can fail only through a build error.

**M2 DATA (Note 260-5): gb behaves as the M1 null.**
- H-LIVE per item 0.40 / 0.41, about 1/2: a coin at the start fork.
- H-EXTEND 0.08 / 0.02, far below 1/2: deterministic "continue", one twin of each pair.
- This is what "no effect" means in the OC block (§10).

**M3 PROVED: twins discriminate at every fork position.**
- Without twins, at H-CHAINSRC j < 3 the spine branch S_{j+1} is read by S_{j+2} while the insert X is a sink. A blind policy that prefers the continuing branch then solves the item.
- With twins, the twin asks X, so that policy is wrong on it. Pair-accuracy is 0 at every j.
- At j = 3 (= d - 1) both candidates are sinks. The single-item rows already need the question there, so they bind at j = 3 and pool j 1-3 for twins.

**M4 DERIVED-UNDER-ASSUMPTIONS: what P9's selection must compute in one pass.**
- Assumptions:
  - (A1) S_{t-1} carries only the values of computed entities, with no ancestor-set register. WS / aux training pushes it toward this (inject_aw.py:394-403 aux; ws_readout 134-152).
  - (A2) the pass-t selection is computed inside pass t from (text, cue, S_{t-1}).
- Then at a fork, pass t must evaluate "which candidate is an ancestor-or-self of the cue's S_d". That is the PLAN-PROBE-F fork head's function on the same trunk with S = 0 (§4).
- Hence probe FAIL implies P9 cannot pass the fork rows.
- The assumption fails if earlier passes store an ancestor set (backlog P4, 2505.12514). That route is disclosed and not excluded. A PLAN-PROBE-F FAIL therefore parks P9; it does not prove P9 impossible.

**M5 PROVED: fading the target cue is a no-op on passes t >= d.**
- cue_str(it, min(t, d)) = cue_str(it, d) for t >= d (inject_st.py:44-46; cue_inputs 70-80 reads think_cues[min(t, len) - 1]).
- So P9-T changes only passes t < d. The hold passes and hold_step's xs[-1] (inject_aw.py:179-181) are unchanged.
- P9-G changes every pass. It needs think_cues extended to K entries, one coin per pass.

**M6 PROVED: the fade schedule.**
- f(s) = clip((s - w0) / (w1 - w0), 0, 1); p_t(s) = clip(2 f(s) - (K - t)/(K - 1), 0, 1), with K = 8.
- p_t is non-decreasing in s and t, and p_t = 1 for every t once f = 1, since 2 - (K - t)/(K - 1) >= 1.
- Values:
  - f = 0.25: p_8 = 0.5, p_5 = 0.071, p_1..4 = 0;
  - f = 0.5: p_t = (t - 1)/7.
- With M5, P9-T's effective fade at d 4 covers passes 1-3 only, and p_3 = 2/7 at f = 0.5.

**M7 DERIVED: KD on the ':' is a soft value label.**
- For teacher p̄ and student p over NUM_IDS, KL(p̄ || p) = CE(p̄, p) - H(p̄), and d/dlogits = p - p̄.
- If p̄ is close to one-hot on v_min(t, d), the gradient is the CE gradient on v_t up to the teacher's off-mass.
- DATA: of-lineage conditional WS 0.95-0.97 per pass (Note 262-3, CTRL arm at 2k).
- So KD's information beyond the per-pass aux (already a simulator target; C4 scope unchanged, Note 263-2) is (i) a direct target at the write source ':' (inject_aw.py:47-56), and (ii) the teacher's off-mass.
- The KD-vs-NOKD contrast measures (i) + (ii). It is descriptive.

**M8 DERIVED: the frozen teacher equals the student at step 0.**
- The teacher is of_s1 frozen, which is the init_model. So at s < w0 (p = 0 everywhere) the faded set is empty and the KD term is an in-graph zero. The loss equals of's recipe exactly (gate G2).
- A frozen teacher replaces the draft's EMA teacher (1703.01780), because of_s1 is already converged under the step cue. EMA-of-student adds drift with no expected gain. The EMA variant is a disclosed alternative, not an arm.

**M9 PROVED: H8's halt carries no harness information.**
- The halt is sigmoid(w · LN(h_11[':', t]) + b) >= 0.5, on pass inputs asserted identical across t (D-BLIND).
- So khalt is a function of (text, model) only.

**M10 DERIVED-UNDER-ASSUMPTIONS: H8 alone can repair H-EXTEND twins but not H-LIVE twins.**
- Assumption: the trajectory writes one entity per pass and follows the frontier.
- H-EXTEND: a blind trajectory passes S_4 at pass 4 and X1 at pass 5. A question-conditioned halt (at 4 for the S_4 twin, 5 for the X1 twin) gives pair-accuracy up to 1.
- H-LIVE: the trajectory visits one branch only, so pair-accuracy at halt is <= P(visits both), about 0.
- This is TH2's prediction for halting.

**M11 TH5 capacity and order.**
- **M11a PROVED, given rank 40:** if the 40 name embeddings are linearly independent (G-RANK checks rank(wte[NAME_IDS]) = 40 at build), some W_n: R^768 -> R^64 maps them to orthonormal keys (40 <= 64).
- **M11b PROVED:** with orthonormal keys, beta = 1 delta writes S <- S - (S k) k^T + v k^T (nk.py:56-61) to distinct keys commute. So the sentence writes of one pass are order-free, and a read returns the last value written to each key.
- **M11c DERIVED-UNDER-ASSUMPTIONS:**
  - Define depth(given) = 0 and depth(op) = 1 + max depth of its sources.
  - Assume an op sentence's write at pass t is correct whenever all its sources are givens (read from the text) or were written by pass t - 1.
  - Then entity s is correct from pass depth(s) on. The spine's S_t resolves at pass t, the same timing as of's aux. Distractor chains resolve in parallel and nothing is selected.
  - The question enters only at the decoder's NK read at the question's name token, which is one-hop. So the TH2 bound does not apply.
- **M11d CONJECTURE: formation.** Formation from of_s1 under the gate pressure is a bet. §262's NK binding went unused (Note 262-3, ablation drop 0.004).

**M12 DATA (synthetic simulation, §10): the OC numbers.**

## 3. Shared eval instruments (new part module boxes/lane07_parts.py; registered in PARTS / AW_ONLY, inject_aw_eval.py:36-37)

### 3.1 HO-PERM renders
- Base: PR.test_perm_sets on REG TEST2 d 4 (boxes/perm_render.py:44-56, seed 25150 + 4). Then IR.held_out / held_out2 (insert_render.py:107-151, 236-250) apply the insert families relative to the permuted labels (`_put` by label, l. 97-98).
- Families: H-EXTEND, H-CHAINSRC (j 1-3), H-LIVE (before / inside / after), H-CHAINSRC-OP, H-CHAINSRC-LIVE (j 1-3).
- OW.solve == answer is asserted in `_finish_ho` (l. 101-104).
- Set shas come from two identical clean builds, committed before output (G-RENDER).
- Clean renders: the existing part_insert_ho (inject_aw_eval.py:1125-1148), descriptive.

### 3.2 TWIN pairs (part `twin_ho`, perm + clean)
- The twin of a family item has the same text with the final question QUESTION(u, X_end) (nested_words.py:83). X_end is:
  - X1 for H-EXTEND (depth 5);
  - X for H-CHAINSRC j (depth j + 1);
  - X2 for H-LIVE (depth 2 from Y);
  - X2 for H-CHAINSRC-LIVE j (depth j + 2).
- Asserts:
  - the twin texts differ only after the last ". ";
  - OW.solve(twin) == the generator value.
- Pairs whose answers are equal, or whose decision-pass values are equal, are excluded. Their count is reported. About 10/11 are kept if values are near-uniform mod 11 (ow_words.py:37-38; DATA at build).
- Decision pass t*:
  - H-CHAINSRC(-LIVE) j: t* = j + 1;
  - H-EXTEND: t* = 5 (base holds v_4, twin writes v_X1);
  - H-LIVE: t* = 1 (base v_{S1}, twin v_{X1}).
- Fields per (family, pos, render):
  - acc_a, acc_b;
  - **pair_acc** (both decoder answers right at K' 8);
  - pair_acc_k16;
  - **ws_pair** (both ':' argmaxes at t* right; computed as in ws_readout, inject_aw.py:143-147);
  - per item: pred at K' 0..16, so the C5 r = 3 rule can be scored.
- On of_s1 under its own step cue (the plan given, twins with their own spines), the part runs as a descriptive gold-plan reference.

### 3.3 Q-ABLATE (part `q_ablate`)
- The pass inputs drop the question sentence, and any cue name becomes " someone" (GPT-2 id 2130, one token, checked 2026-11-04). So the target is absent from every pass input.
- The decoder input is unchanged (dec = think + SFX, inject_ow.py:71).
- Fields:
  - acc on HO-PERM H-EXTEND and H-CHAINSRC j 3;
  - the twins' ws_pair under ablation. It must be exactly 0 (M1 corollary, gate G-TWIN).
  - decoder pair_acc under ablation (DECODER-RESCUE, descriptive): selection by the decoder from a question-blind state.

### 3.4 Model-decided stop, D-BLIND and the PLAN-FREE assert
- **Inputs:** for every plan-free arm the eval inputs come from `it["text"]` alone. For P9-T, the cue name is parsed from the text's final question and asserted equal to spine[-1] at build only. An item copy with names.spine, values and d deleted must give byte-identical tensors (gate D-BLIND).
- **PLAN-FREE assert:** passes()' assert (inject_aw_eval.py:65-66) is widened to o["cue"] in ("generic", "target", "fade_t", "fade_g").
- **Stop readings** (none uses d):
  - HOLD: the answer at K' 8 vs K' 16 / 64;
  - C5 r = 3: the first K' whose decoder answer is unchanged for 3 passes (Note 259-11), on the twin_ho per-K' preds; **never-halt by K' 16 = failure**;
  - H8: the head's first K' with p >= 0.5; **never-halt by K' 8 = failure** on acc-at-halt (stricter than part_halthead's censor-at-K' convention, inject_aw_eval.py:905).

## 4. Stage 1: PLAN-PROBE-F (fork-split, of_s1 trunk; shared with doc 06's P1 / P2 decision)

- **Model:** of_s1's full state (`--init_model boxes/runs_fsq/of_s1/model.pt`), one pass from S = 0. This is exactly of's pass-1 computation through pass_aw (inject_aw.py:47-56). All params are fine-tuned on a copy. The tied-head name CE moves wte (red team 06 finding 9); that is harmless here because the probe model is discarded.
- **Input:** [EOT] + enc(" " + text + Q_c) + SFX, with:
  - Q_c = " Which name comes after <c>?" for a current entity c;
  - Q_c = " Which name comes first?" for the start query.
- **Label:** the next spine entity's name token, or " none" (GPT-2 id 4844, single) when c = S_d. Readout: colon_logits (inject_aw.py:61-63) restricted to the 40 NAME_IDS + " none". An argmax outside that set counts as a failure.
- **Training:** 1.5k steps, bs 64, lr 1e-4.
  - Items from of's stream law: REG, perm 0.5, distract_mix s256,chain,givop, p 1.0, own stream seed 27070.
  - c drawn uniformly from {start, S_0..S_d} on the spine. Training includes the chain-insert type, so forks occur in training.
  - Disjointness from TEST2 / TEST-PERM / insert / HO-PERM texts is asserted, as in inject_aw.py:337-338.
- **Eval queries** (HO-PERM, §3.1):
  - FORK_j (j 1-3): c = S_j on H-CHAINSRC pos j and H-CHAINSRC-LIVE pos j, n = 2,000 per j; label S_{j+1}. The competitor is X (or X1).
  - STOP: c = S_4 on H-EXTEND, n 1,000; label " none". The competitor is X1.
  - START: the first query on H-LIVE (3 pos), n 3,000; label S_1. The competitor is X1.
  - QMASK_j3: FORK_3 with the question sentence deleted.
  - Descriptive: clean renders; plain TEST-PERM d 4 at every c.
- **PASS (binding, one run, scope "one lineage, probe"):** FORK_1, FORK_2, FORK_3, STOP, START >= 0.90 each, AND FORK_3 - QMASK_3 >= 0.30 (question used).
- **Routing (decision column, as red team 06 finding 10):**

| Reading | Decision |
|---|---|
| PASS | S2 P9 pilot launches |
| FAIL with FORK_3 and STOP >= 0.90 but FORK_1 < 0.90 ("short reach") | P9 parked (M4). H8 keeps its route, since STOP is a one-hop equality. Doc 06's P2 BACKWARD keeps its route if its own parent head >= 0.9 |
| FAIL on the QMASK clause (the head uses non-question cues, e.g. insert op-type statistics) | P9 parked. The probe's fork accuracy is reported as a shortcut |
| Any FAIL | S5 TH5 pilot becomes the lane's lead (it needs no selection) |

- **Cost:** about 0.3 s/step one pass x 1.5k = 8 min, + setup and evals about 20 min. ~$0.12; **guard $0.5**.

## 5. Designs and arms (args diff vs boxes/runs_fsq/of_s1/args.json)

**Common base: of_s1's args.** arm inject, think_k 8, d_max 4, steps 8000, bs 64, lr 1e-4, micro 2, mem_write top, write_mix 0 / 0, cache_lower, aux_step 1.0, aux_fade 48000,48000, aux_reweight, aux_target value, fast_add, src ow, mul_s_p 0 (REG), mul_s_p3 1.0, distract_mix s256,chain,givop, distract_p 1.0, perm_p 0.5, cue step, write_src answer, init_trunk s233_askname_s1 (c00f7ec0bd7977fe), init_model bun_s1_init (1ad3517b7635fddf), hold_inc 1.6, hold_start 500, hold_far 16 / 16 / 40, hold_log, ask_p 0.25, ask_perm_p 0.5, nk 0, eval_parts acc,test_perm,insert_ho,swap_kd,lp5h,stab. RTX 3090 (train_gpu of of_s1).

**Every arm below sets:**
- `--init_model boxes/runs_fsq/of_s1/model.pt --init_model_sha <prefix, at build>` (fine-tune from the SOTA head; §271 does the same);
- `--args_ref` = a frozen copy of of_s1/args.json;
- `--eval_parts` += late_premise, long_k128, name_swap, ho_perm, twin_ho, q_ablate, full_perm (descriptive).

**New flags:**
- every new flag defaults to off, and at the default it is bitwise of's path (G1);
- each arm's diff lists only these flags, plus seed / steps / stop_at / out.

### 5.1 P9 CUE-FADE

**Mechanism.**
- Pass t's cue is cue_str(it, min(t, d)) = " How many <u> does <S_min(t,d)> have?" (7 tokens; inject_st.py:44-46, 49-63). The name sits 6 tokens before ':' (tok_aw's assert, inject_aw.py:91-92; nk.py:28 CUE_NAME_OFF).
- CUE-FADE replaces that token at pass t by a fade token with probability p_t(s) (M6). The coin is per item and per pass: `default_rng((seed, 27071, step, k, t))`.
  - **P9-T:** the fade token = S_d's name. At p = 1 the cue duplicates the text's own question, and every pass input is byte-identical (§261's condition in a 7-token form). By M5 it acts on passes t < d only.
  - **P9-G:** the fade token = " someone" (2130). At p = 1 the cue carries no information, and the target is only in the text. think_cues are extended to K entries.
- Both keep the cue at 7 tokens, so cue_inputs' one-padding assert (inject_st.py:77) and the name assert (inject_aw.py:91-92, widened to "name or fade token") hold.
- The coins are applied after the perm and the inserts. The DISTRACT "cues unchanged" assert (inject_aw.py:329-330) compares unfaded cues.
- Ask items (inject_aw.py:99-106, 350-359) are unchanged: they are one-hop target questions already.

**KD.**
- Teacher: frozen of_s1 (M8), no-grad, on the **unfaded** inputs.
- L = L_dec + λ_aux L_aux + L_inc + λ_kd · mean over faded (item, t) of KL(p̄_t || p_t). Here p = softmax of colon_logits over NUM_IDS at pass t; λ_kd = 1, temperature 1.
- No KD on unfaded passes.
- Insertion points:
  - teacher passes alongside the student's pass loop (inject_aw.py:372-403);
  - KD added after the aux (404-407).
- The teacher adds ~0.5 GB (mem_check G4).

**Monitors per 1k bin:** faded share by pass; KD loss; student WS split faded / unfaded; teacher WS (d 3-4, passes 2-3). A cell whose fade monitors are silent for 2k steps is a build fault.

**Arms (args diff vs of_s1, plus the common init / eval lines):**

| Arm | Flags | Stage |
|---|---|---|
| P9-T | `--cue_fade t --fade_win 250,1500 --kd 1.0 --kd_teacher boxes/runs_fsq/of_s1/model.pt --stop_at 2000` (pilot); full: `--fade_win 500,5500`, no stop_at | S2 / S3 |
| P9-G | `--cue_fade g`, otherwise as P9-T | S2 / S3 |
| TC-ABRUPT | `--cue_fade t --fade_win 0,0 --kd 0 --stop_at 2000`. p = 1 from step 1, no teacher: §261's no-curriculum control | S2 |
| P9-x-NOKD | the S3 winner with `--kd 0` | S3, 1 seed, descriptive |

**Pilot step-0 reference:** of_s1 evaluated under the p = 1 inputs of P9-T and P9-G (eval only).
- Prediction: TEST-PERM d 3 < 0.30 with probability 0.8. NAME-SWAP says a jump-ahead cue does not compute (Note 263-3: j >= 3 follow at the floor).

**S3 arm choice (pre-stated):** P9-G if its pilot is GO (the stronger scope: no target in the cue slot); else P9-T if GO; else TC-ABRUPT if GO; else P9 PARKs.

### 5.2 H8 ANSWERABLE-NOW head (H7 TC-HALT on a P9 model)

**Mechanism.**
- a_t = w · LN(h_11[':', pass t]) + b: one logit on the ':' hidden that pass_aw already returns (inject_aw.py:47-56).
- Label y_t = 1[argmax dec_logits(S_t) over NUM_IDS == ans], from the model's own decoder (no-grad; inject_ow.py:86-90). Training label only.
- Computed on a seeded quarter of each batch: coin seed 27072; 16 items x K passes.
- L_h = λ_h BCE(a_t, y_t), λ_h = 0.5.
- Init: w ~ N(0, 0.02) with generator seed 27073, b = -2. Non-zero init; prior about 1/K.
- Usage monitor: train AUC per 1k bin.
- Halt at eval: the first t with sigmoid(a_t) >= 0.5. **Never-halt = failure** (§3.4).

**Arms.**

| Arm | Model and flags | Stage |
|---|---|---|
| H8-gb | gb_s1's args (bun lineage, cue generic: the only stored plan-free-input model; Note 260-5 keeps its model.pt) + `--init_model boxes/runs_fsq/gb_s1/model.pt --h8 1 --h8_lam 0.5 --h8_frac 0.25 --h8_frozen 1 --lr 1e-3 --steps 1000`. Its diff is against gb_s1/args.json, not of_s1 (disclosed) | S1b |
| H8-P9 | S3's three seeds' model.pt + the same H8 flags; diff vs that seed's args = {h8*, lr, steps, init_model} | S4 |

- H8-gb tests M10 (TH2's halting prediction) on a known blind trajectory. Its renders are clean, because gb is order-bound (Note 251-16 lineage). Its label is a routing label, not a C1 claim.

### 5.3 TH5 DATAFLOW-ALL-WRITE

**Mechanism.**
- Pass input: of's generic plan-free input (`--cue generic`, inject_aw.py:73, 80-83: [EOT] + enc(" " + text + " What is the next value?") + SFX), byte-identical every pass. The text ends with the question.
- At every pass, the model writes at every sentence end p_s (the '.' tokens; the question's '?' is not a write position) into an NK block (nk.py:34-61, packed as in InjectNK 71-91).
  - Key: the entity the sentence defines (B for ops, the name for givens; nested_words.py:128-144). This is a text-parse oracle; scope note in §14.
  - Value: v_s = g_s W_vn ln_w(h_11[p_s]).
  - Gate: g_s = sigmoid(w_g · h_11[p_s] + b_g), with b_g = -2 and non-zero w_g (seed 27074).
  - Writes go in sentence order. By M11b the order does not matter once keys are orthonormal.
- Read: at every name-token position at block 6 (nk.py:46-54). The decoder reads at the question's name token, which is the question conditioning.
- The ':' DeltaMem write stays.
- The aux is unchanged: S_t's value at pass t, which is the M11c timing.
- The NK key / write code needs T > 1 writes per pass: a loop over sentence ends with distinct keys.
- Halt: H8 head flag (§5.2) or the C5 r = 3 rule. The key-norm r_t = ||S_n k(A_q)|| is descriptive only (§0.2).

**Arms.**

| Arm | Flags | Stage |
|---|---|---|
| TH5 | `--cue generic --dataflow 1 --stop_at 2000` (pilot) / no stop_at (full). Strict load of of_s1 except th5.* (W_n, W_vn, W_on, w_g, b_g), as §262's strict-except-nk.* (inject_aw.py:295-296) | S5 / S6 |
| TH5-ABL | eval-only: the NK read zeroed (nk.py:50-51 `ablate`) on the same model | S5 / S6 |

## 6. Code touch points (each line range read 2026-11-04)

- **boxes/inject_aw.py**
  - InjectAW.pass_aw 47-56; colon_logits 61-63; TGT_FMT / tgt_cue 66-70; GEN_CUE 73.
  - tok_aw 76-93 (generic 80-83, target 84-86, the name assert 91-92). Fade coins and the fade_t / fade_g cue tags go here or in draw_items.
  - tok_ask 99-106.
  - ws_readout 134-152; hold_step 159-199 (xs[-1] at 179-181).
  - args 211-249 (new flags `--cue_fade`, `--fade_win`, `--kd`, `--kd_teacher`, `--h8`, `--h8_lam`, `--h8_frac`, `--h8_frozen`, `--dataflow`).
  - asserts 255-268, 290; strict-load-except 293-296; lam_at 299-303 (f(s) beside it).
  - draw_items 316-346 (DISTRACT assert 329-330, disjointness 337-338); ask_items 350-359.
  - loss_reg 361-419 (pass loop 372-403, decoder / aux 404-407); loss_ask 421-431; loss_items 433-440; the step loop's hold_step call 556-559.
- **boxes/inject_st.py:** cue_str 44-46, tok_st 49-63, cue_inputs 70-80 (padding assert 77).
- **boxes/nk.py:** CUE_NAME_OFF 28; NAME_IDS 29-31; NK 34-61 (key 43-44, read 46-54, write 56-61); DeltaMemNK 64-68; InjectNK 71-123 (pass_aw 79-91).
- **boxes/deltamem.py:** read_fixed 158-176; write_seq 179-223; _block 251-260 (is_causal 256).
- **boxes/inject_aw_eval.py:** PARTS / AW_ONLY 36-37; seq_std 44-47; passes 54-82 (PLAN-FREE assert 65-66); part_cue_only 159-171 (the Q-ABLATE input-builder template); part_halt 812-824; hh_* 831-881; part_halthead 883-912; part_test_perm 1044-1070; part_insert_ho_s256 1077-1107; part_insert_ho 1125-1148.
- **boxes/insert_render.py:** statements 18-28; HO_FAMS 79-80; _put 97-98; _finish_ho 101-104; held_out 107-151 (H-CHAINSRC 121-123, H-EXTEND 135-138, H-LIVE 139-143); distract_mix 194-230 (the 'chain' type 198, 210-212); HO2_FAMS 233; held_out2 236-250.
- **boxes/perm_render.py:** permuted 22-35; check 38-41; test_perm_sets 44-55; SEED_PERM_EVAL 12.
- **boxes/nested_words.py:** NAMES 33; GIVEN / QUESTION 82-83; _statements 128-144; _finish 146-170 (text 156).
- **Other boxes files:**
  - boxes/inject_ow.py: tok_ow 63-71, dec_logits 86-90, aux_logits 92-103;
  - boxes/ow_words.py: apply mod 11 37-38, solve 178-184;
  - boxes/of_parts.py: SEED_NS 26300 at l. 15, the collision; name_swap 51 ff.
- **theory/:** score_of.py (RETAIN definitions: NOT-DECODER = test_perm by_d['4'].acc_kprime['2'] <= 0.20, l. 6; SWAP4 / LP5-D4 field paths l. 45-50); score_fp.py:36 wilson. **scratch/gradflow_check.py** exists.

## 7. Gates (before any training output)

1. **G1 OFF-BIT:**
   - every new flag at its default reproduces of_s1's recipe;
   - args_diff vs of_s1 = {} plus out / seed;
   - the step-0 loss and a 20-step `--sched_scale` smoke are bitwise equal on of's path, including ask items + perm, so loss_items routes through loss_ask (inject_aw.py:433-440).
2. **G2 fp64 equivalence with each flag on:**
   - P9 at s < w0: p = 0 and the teacher equals the student, so the loss is bitwise of's and the KD term is an in-graph zero;
   - H8: loss - λ_h · BCE == of's loss;
   - TH5: with g_s forced to 0 and W_on = 0, the loss equals of's in fp64, and the DeltaMem blocks after pass K are bitwise equal.
3. **G3 gradflow_check on every new term:**
   - KD reaches the trunk through faded passes only;
   - BCE reaches w, b only under h8_frozen;
   - TH5's g_s, W_vn and W_n get gradient from the decoder and aux.
4. **G4 SMOKE at the real data scale and mem_check:**
   - smoke: 20 steps, bs 64, d 4 items with 2 inserts, the fade at p = 1 on every pass;
   - mem_check at d_max with the teacher, micro pinned at 2, 1 GiB headroom (memory: auto_micro headroom).
5. **G5 asserts:**
   - P9: at p = 1 the inputs are byte-identical across passes and name S_d only (P9-T) or no entity (P9-G);
   - D-BLIND (§3.4);
   - H8 labels never read it["d"];
   - TH5: every write key is a name token, and the question is not a write position;
   - G-RANK (M11a).
6. **G-RENDER:** HO-PERM and TWIN set shas from two identical builds. Every twin pair differs only in the final question; OW.solve agrees.
7. **G-TWIN:** under Q-ABLATE, the twins' pass inputs are byte-identical and ws_pair = 0 on a 200-pair smoke (M1 corollary).
8. **G-DISJOINT:** PLAN-PROBE-F training texts are disjoint from every eval text.
9. **Scorers** score_pp.py, score_p9.py, score_h8.py, score_th5.py, with fixtures (§13), committed and G-D'd by Audit before any output. The OC script (§10) is committed beside this doc.

## 8. Criteria (Wilson 95% CI on every proportion-bar cell; NEAR when the bar lies inside the CI; letters on raw values; a claim resting on a NEAR cell is worded NEAR)

### S1 PLAN-PROBE-F (one run; one lineage, probe)

| Cell | n | Bar | Type |
|---|---|---|---|
| FORK_1, FORK_2, FORK_3 (HO-PERM; H-CHAINSRC + H-CHAINSRC-LIVE) | 2,000 each | >= 0.90 | binding |
| STOP (H-EXTEND, c = S_4) | 1,000 | >= 0.90 | binding |
| START (H-LIVE) | 3,000 | >= 0.90 | binding |
| FORK_3 - QMASK_3 | 2,000 / 2,000 | >= 0.30 | binding (question use) |
| clean renders, TEST-PERM d 4 by c | | | descriptive |

### S1b H8-gb (routing label TH2-HALT)

- Binding: TWIN pair-acc at halt, clean renders. H-EXTEND >= 0.60 AND H-LIVE (pooled) <= 0.20.
- Descriptive: HALT-OWN (Spearman(mean khalt, d) over TEST2 d 1-6 >= 0.8; acc_halt >= acc_k8 - 0.02 at d 1-4) and val AUC at d 5-6.

### S2 P9 pilot (per arm, seed 1, 2k, eval at p = 1)

- **GO:** all of:
  - TEST-PERM d 2-4 at K' 8 >= 0.85 each;
  - WS (test_perm's per-pass ws) d 3 passes 2-3 >= 0.50;
  - TWIN pair_acc (perm) on H-EXTEND >= 0.50 and on H-CHAINSRC j 3 >= 0.50;
  - PLAN-FREE assert and D-BLIND true.
- **PARK:** TEST-PERM d 3 < 0.20, or both twin pair_acc < 0.15. That is no better than the M1 / M2 null plus noise.
- **Else NOTE.**
- Descriptive: TC-ABRUPT vs P9-T; the step-0 reference.

### S3 P9 full (3 seeds of the chosen arm from of_s1; PLAN-FREE-CONTENT-C1 per seed)

| Row | Cells | Bar |
|---|---|---|
| RETAIN-PF | TEST2 d 1-4 acc K' 8 | >= 0.95 |
| | NOT-DECODER (test_perm d 4 K' 2) | <= 0.20 |
| | SWAP4 follow_U | >= max(0.50, floor + 0.30) |
| | LP5-D4 | as score_of |
| ORDER | TEST-PERM d 2-4 | >= 0.90 |
| | LATE-PREMISE d 2-4 | >= 0.85 |
| CONTENT (HO-PERM, single items) | H-EXTEND, H-LIVE (pooled), H-CHAINSRC-LIVE (pooled) | >= 0.80 each |
| | H-CHAINSRC-OP | >= 0.85, min pos >= 0.80 |
| TWIN (HO-PERM) | pair_acc on H-EXTEND, H-CHAINSRC j 1-3 pooled and j 3, H-LIVE pooled, H-CHAINSRC-LIVE pooled | >= 0.80 each |
| MECH | ws_pair on H-EXTEND and H-CHAINSRC j 3 (null = 0 exactly, M1) | >= 0.70 |
| Q-USE | Q-ABLATE drop on H-EXTEND and on H-CHAINSRC j 3 | >= 0.30 |
| STOP (model-decided) | H-EXTEND K' 16 acc | >= K' 8 - 0.02 |
| | C5 r = 3 on TEST-PERM d 2-4: acc_halt (never-halt = failure) | >= acc_k8 - 0.02 |
| | Spearman(mean khalt, d) over d 1-6 | >= 0.8 |
| C6 | K' 64 / 128 vs K' 8 at d 1-4, TEST2 and TEST-PERM | >= K' 8 - 0.02 |
| PLAN-FREE | passes() assert + D-BLIND | true |

- **Arm label:** PLAN-FREE-CONTENT-C1 at k >= 2 of 3 seeds ("one lineage; plan-free at test, steps supervised in training").
- **C2-RETAIN (separate label; 3/3 seeds raw):** TEST2 d 6 >= of_s1 d 6 - 0.05 (0.921) and d 7 >= of_s1 d 7 - 0.05 (0.859). NEAR-SEED rows use sigma_seed 0.090 (Note 269-2). Retention only: no C2 improvement claim (standing rule 5).
- **Descriptive:**
  - clean renders and clean twins;
  - full_perm (§270's part);
  - name_swap;
  - DECODER-RESCUE;
  - the KD effect (winner vs NOKD, 1 seed, twin pair_acc);
  - the gold-plan twin reference on of_s1.

### S4 H8-P9 (per seed; HALT-OWN-PF)

- Spearman(mean khalt, d) over TEST2 d 1-6 >= 0.8.
- acc_halt >= acc_k8 - 0.02 at d 1-4 (never-halt = failure).
- TWIN H-EXTEND pair-acc at halt >= 0.80.
- PARK if val AUC < 0.70 at d 5-6.
- Arm at >= 2/3.

### S5 TH5 pilot (seed 1, 2k)

- **GO:** all of:
  - TEST-PERM d 2-4 >= 0.50;
  - NK-ablation drop at d 3 >= 0.20;
  - TWIN H-LIVE pair_acc >= 0.50 (no selection needed, M11c);
  - gate usage two-sided: mean g_s >= 0.5 on resolved and <= 0.2 on unresolved (depth, pass) cells.
- **PARK:** TEST-PERM d 3 < 0.20 or ablation drop < 0.05.
- **Else NOTE.**

### S6 TH5 full (3 seeds; DATAFLOW-C1 per seed, arm >= 2/3)

- the S3 rows, except that MECH is replaced by:
  - NK-ablation at K' 8 on TEST-PERM d 4 <= chance + 0.10;
  - TWIN pair_acc on every family >= 0.80, H-LIVE included;
  - C5 r = 3 khalt = d on >= 0.80 of clean d 1-6 items.
- C2-RETAIN as S3.
- Descriptive: d 7 / 8 at K' 9 / 10; r_t; NK read accuracy by entity depth and pass.

## 9. Predictions (pre-stated, before any build)

| ID | Prediction | P |
|---|---|---|
| PR1 | S1 PLAN-PROBE-F PASS | 0.50 |
| PR1a | STOP >= 0.90 and FORK_3 >= 0.90 (one-hop equality) | 0.85 |
| PR1b | FORK_1 >= 0.90 (three-hop backward reach) | 0.50 |
| PR2 | S1b TH2-HALT (H-EXTEND twin at halt >= 0.6 AND H-LIVE twin <= 0.2) | 0.45 |
| PR2a | H-LIVE twin at halt <= 0.2 alone (M10) | 0.85 |
| PR3 | step-0 of_s1 under the p = 1 cue: TEST-PERM d 3 < 0.30 | 0.80 |
| PR4 | S2 GO for P9-T, given S1 PASS | 0.40 |
| PR4a | S2 GO for P9-G, given S1 PASS | 0.20 |
| PR4b | S2 GO for TC-ABRUPT, given S1 PASS | 0.20 |
| PR5 | S3 per seed PLAN-FREE-CONTENT-C1, given GO | 0.50 |
| PR5a | S3 arm >= 2/3, given GO | 0.45 |
| PR6 | S3 C2-RETAIN 3/3, given GO | 0.30 |
| PR7 | KD effect >= 0.05 on twin pair_acc (descriptive) | 0.50 |
| PR8 | S4 HALT-OWN-PF per seed | 0.60 |
| PR9 | S5 TH5 GO | 0.20 |
| PR10 | S6 DATAFLOW-C1 arm >= 2/3, given GO | 0.40 |
| PR11 | DECODER-RESCUE (decoder pair_acc under Q-ABLATE) <= 0.10 on any P9 model (the decoder does not select from a blind state) | 0.75 |

## 10. OC block (DATA-SIM)

**Script.** Theory's scratchpad lane07_oc.py, output lane07_oc_out.txt. numpy only, `python -I`; two runs byte-identical; 20,000 panels, seed 70700. Both files are committed to docs/design_wave4_2026-11-03/ at registration.

**Model.**
- Each cell's true rate moves by a seed effect sigma (sqrt(rho) z_shared + sqrt(1 - rho) z_cell), with rho 0.5 (C2: 0.8). Items are binomial at the cell's n. Letters are read on raw values.
- Sigma per cell type:

| Cell type | Sigma | Source |
|---|---|---|
| in-range | 0.011 | DATA: Note 269-1 d 4 |
| insert families | 0.03 | assumed 3x in-range |
| extrapolation d 6 / 7 | 0.090 | Note 271-2 |
| pilots | 0.05 | assumed |

- The Q-ABLATE and hold rows enter as independent factors (an approximation; they are near-deterministic under both hypotheses).
- The sim uses SWAP4 >= 0.90, stricter than the registered max(0.50, floor + 0.30). That makes it conservative for P(pass | effect).

| Stage / rule | "No effect" (M1 / M2 null: question-blind passes, gb-like) | Effect | Other worlds |
|---|---|---|---|
| S1 PLAN-PROBE-F PASS | 0.000 (blind head 0.5 at forks) | 0.963 (0.95 everywhere, QMASK 0.5) | marginal 0.92: 0.557; short reach (j1 0.75): 0.000; non-question cue (QMASK drop < 0.30): 0 by construction |
| S1b TH2-HALT | 0.000 (halt unused) | 1.000 (q-conditioned halt, blind selection) | selection also q-conditioned: 0.000 (this label is TH2-specific by design) |
| S2 P9 GO (1 seed) | 0.000 (twin 0.05) | 0.837 (perm 0.92, ws 0.7, twins 0.65) | slow former (twins 0.50): 0.265 |
| S3 PLAN-FREE-CONTENT-C1 per seed / arm >= 2/3 | 0.000 / 0.000 | 0.930 / 0.989 (twins 0.88, mech 0.85, families 0.90, perm 0.95) | weak (0.84 / 0.80 / 0.86 / 0.93): 0.521 / 0.527; STOP-ONLY (H8-world): 0.000 / 0.000 |
| S3 C2-RETAIN 3/3 | lost (d 6 0.85, d 7 0.70): 0.000 | of-level (0.971 / 0.909): 0.245 at sigma 0.09; 0.766 at sigma 0.03 | retained 0.95 / 0.89: 0.150 / 0.445 |
| S5 TH5 GO | 0.000 (NK unused, drop 0) | 0.998 (drop 0.4, H-LIVE twin 0.7) | |

**Reading.**
- Every binding stage has P(pass | no effect) = 0. The TWIN / MECH rows make the null fail by construction (M1), so the false-positive risk is in the build, not in noise (gates G-TWIN, D-BLIND).
- Power is high for a strong effect and about 1/2 for a weak one.
- C2-RETAIN at 3 seeds has low power if the fixed-init seed spread is as large as the lineage spread (0.09). That spread is unmeasured for fixed-init fine-tunes. S3 reports it, and the scope is stated before output.

## 11. Censoring (conservative; never-reached = failure)

- **Stops:**
  - an item whose C5 r = 3 or H8 halt never fires by its K' cap fails the halt rows. It is not credited with its K' 8 answer;
  - a twin pair with a missing or failed member fails.
  - A PLAN-PROBE-F argmax outside the 40 names + " none" fails.
- **Runs:**
  - a cell that never lands, or lands without a binding eval file, gets **one** VOID-RERUN (host failure before the first eval only). After that it is FAIL;
  - a cell whose fade / g_s / AUC monitors are silent for 2k steps is a build fault: VOID, fixed, rerun once.
  - No seed rescue: a failed seed is not replaced.
- **Exclusions:** excluded twin pairs (equal values) are counted and reported, and are not failures. They come from the generator, not the model. If more than 15% are excluded in a cell, G-RENDER fails.
- **P9 scoring:** at p = 1 only; intermediate-p evals are descriptive.

## 12. Staged plan, costs, guards, routing (Vast RTX 3090 ~$0.21/h; ~1.65 s/step = $0.096 per 1k steps; teacher +35%, TH5 writes +15%)

| Stage | Contents | Expected $ | Guard | Decision |
|---|---|---|---|---|
| S0 | this doc → independent red team → Literature brief → registration; builds; gates G1-G9; scorers + fixtures; OC script committed | 0 | — | Audit G-D |
| **S1** | **PLAN-PROBE-F** on of_s1 (1.5k one-pass steps + evals) | **0.12** | **0.5** | PASS → S2; FAIL → P9 parked, S5 leads |
| S1b | H8-gb (frozen head, 1k steps on gb_s1) | 0.15 | 0.5 | TH2-HALT routes H8's scope; independent of S1 |
| S2 | P9 pilot: P9-T 2k ($0.26 + $0.13 evals), P9-G ($0.39), TC-ABRUPT ($0.32), step-0 refs ($0.06) | 1.16 | 2.0 | arm choice §5.1; none GO → P9 PARK |
| S3 | P9 full: winner x 3 seeds x 8k ($1.04 + $0.19 evals + mirror = $1.25 each) + NOKD x 1 ($0.96) | 4.7 (5.95 with one VOID-RERUN) | 7.0 | PLAN-FREE-CONTENT-C1 k/3; C2-RETAIN 3/3 |
| S4 | H8-P9 on the 3 S3 seeds (frozen, 1k each) | 0.45 | 1.0 | HALT-OWN-PF |
| S5 | TH5 pilot 2k (+ TH5-ABL eval) | 0.35 | 0.8 | after S2, or right after an S1 FAIL; GO → S6 |
| S6 | TH5 full 3 seeds x 8k ($1.07 each) | 3.2 (4.3 with one rerun) | 5.0 | DATAFLOW-C1 k/3 |

- **Guards per design:**

| Design | Stages | Guard |
|---|---|---|
| PLAN-PROBE-F | S1 | $0.5 |
| P9 | S2 + S3 | $9.0 |
| H8 | S1b + S4 | $1.5 |
| TH5 | S5 + S6 | $5.8 |

  Each is <= $30. Lane guard total $16.8.
- **Cost by path:**
  - all-GO path ~$10.1;
  - probability-weighted with §9's priors: ~$3.0.
- Wall cap per 8k cell 7 h; off-host checkpoint mirror per run name.
- **Cheapest decisive stage first:** S1 ($0.12) can park the lane's main arm before any $1+ spend. S1b is equally cheap and tests TH2 on an existing model.

## 13. Scorer sketch

theory/score_pp.py, score_p9.py, score_h8.py, score_th5.py, in score_of.py's form: read / score / fixtures; results in theory/results/; Wilson as score_fp.py:36.

**Fields:**
- eval_acc.json by_d[d].acc;
- eval_test_perm.json by_d[d].acc_kprime['8'], ['2'], and ws;
- eval_late_premise.json by_d[d].acc;
- eval_swap_kd / eval_lp5h / eval_long_k128, as score_of.py l. 45-50 reads them;
- eval_ho_perm.json fam[F].acc8 / by_pos;
- eval_twin_ho.json fam[F][render][pos].{n_pairs, n_excluded, acc_a, acc_b, pair_acc, pair_acc_k16, ws_pair, items.pred_k0_16};
- eval_q_ablate.json fam[F].{acc, ws_pair, dec_pair_acc};
- eval_h8_halt.json {d: acc_k8, acc_halt, mean_khalt, never_halt_n, head.val_auc} and twin fields;
- eval_nk_ablate.json by_d[d].acc;
- eval_plan_probe.json {FORK_j, STOP, START, QMASK_3, n};
- gates_pass.json {plan_free, d_blind, fade_identical, twin_ablate_zero, h8_no_d, th5_keys, rank40}.

Before scoring, grep each field path in the writing module (memory: scorer field-path check).

**Labels:** PASS / FAIL (S1); TH2-HALT; GO / PARK / NOTE (pilots); PLAN-FREE-CONTENT-C1 k/3; C2-RETAIN (3/3, NEAR-SEED rows); HALT-OWN-PF k/3; DATAFLOW-C1 k/3; NEAR flags per cell.

**Fixtures (at least 12 per scorer):**
- GO;
- PARK by chain loss;
- PARK by a twin floor;
- NOTE partial;
- a missing assert gives not-GO;
- pending (missing file);
- NEAR on a bar inside the CI;
- never-halt counted as failure;
- excluded pairs > 15% → G-RENDER fail;
- a ws_pair > 0 under Q-ABLATE → build-fault;
- C2 3/3 vs 2/3;
- STOP-ONLY world → FAIL: TWIN.

## 14. Unresolved (for the red team and Audit)

1. **M4's assumption (no ancestor register)** decides whether an S1 FAIL truly parks P9. A P4-style ancestor pre-pass would reopen it. Not in this design.
2. **TH5's key oracle.** Write keys come from the generator's "which entity this sentence defines" parse, a text-local fact. That is a C1 scope caveat (plan-free, but parse given). A learned-key arm (k = normalize(W_n h_11[p_s])) is the follow-up.
3. **Fixed-init seed spread is unmeasured.** C2-RETAIN power ranges from 0.25 to 0.77 (§10). If S3's three seeds show spread > 0.05 at d 6, C2 is reported as NEAR-SEED-limited, with no rescue.
4. **The of_s1 vs fp_s1 base.** If §270 reads FULL-ORDER-C1 2/2 before S3 launches, S3 stays on of_s1 (args frozen at registration). An fp rebase is a follow-up.
5. **H8-gb runs on the bun lineage** (clean renders, order-bound). Its label routes; it does not claim.
6. **The KD-vs-NOKD contrast is one seed (descriptive).** Upgrading it to a claim needs 3 more seeds (~$2.9).
7. **Exact build line counts and `--init_model_sha` prefixes** for of_s1 / gb_s1 are filled at build. No sha was computed here (read-only rule).
8. **Interaction with doc 06.** PLAN-PROBE-F is one run serving both docs. If doc 06's runner registers first, its QUERY-HANDOFF read does not change this lane's S1-S3, because they need no runner: inputs are text-only and passes() suffices.

## 15. RED TEAM (independent sub-agent)

Pending. Required before registration (README caveat). Points to attack first:
- M1's use of exchangeability on REG items: spine ops are drawn from the REG mix, while inserts are add_c on FRAME_TRAIN. Twins remove the asymmetry for pair rows, but the single-item rows still carry it.
- D-BLIND's coverage of the decoder input.
- Whether P9-T's duplicated question is a leak of anything beyond the question.
- TH5's parse oracle.

## 16. arXiv ids (each fetched at arxiv.org/abs/<id>, title checked, by the draft's lead)

2405.14838 From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step (P9 schedule); 1511.03643 Unifying distillation and privileged information (P9 teacher); 1703.01780 Mean teachers are better role models (EMA teacher; replaced by a frozen teacher, M8); 1503.02531 Distilling the Knowledge in a Neural Network; 2207.07061 Confident Adaptive Language Modeling (H8 halt); 2107.05407 PonderNet; 1603.08983 Adaptive Computation Time; 2402.06457 V-STaR (self-labelled verifier); 2607.20519 Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts; 2505.12514 Reasoning by Superposition (TH2 / TH5 parallel frontier; P4 ancestor sets); 2312.04927 Zoology (MQAR recall capacity); 2310.17191 How do Language Models Bind Entities in Context?; 2412.06464 Gated Delta Networks; 2505.20896 How Do Transformers Learn Variable Binding in Symbolic Programs?; 2502.19907 Order Doesn't Matter, But Reasoning Does; 2410.05464 Progressive distillation induces an implicit curriculum; 2604.04902 Are Latent Reasoning Models Easily Interpretable?
