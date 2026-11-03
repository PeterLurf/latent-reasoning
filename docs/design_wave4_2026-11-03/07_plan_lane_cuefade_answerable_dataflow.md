# Wave-4 design 07: the plan-free lane (P9 CUE-FADE, H8 ANSWERABLE-NOW, TH5 DATAFLOW-ALL-WRITE)

Lead design agent, 2026-11-03. Read-only on the repo; no numbers computed from stored outputs. Covers backlog IDs **P9**, **H8** (and H7 TC-HALT as H8 on a target-cue model), **TH5**, using **TH2** as the decision rule; Lane C of docs/EXPERIMENT_BACKLOG.md §1.

## 1. Goal and the TH2 decision rule

Note 260-5: without the plan in the cue (gb_s1 / s2), clean-chain C1 holds (TEST2 d 1-4 1.000, SWAP4 0.985) but the loop selects by **forward reachability**: H-EXTEND 0.08 / 0.02, H-LIVE 0.40 / 0.41, H-CHAINSRC-LIVE 0.06 / 0.41, d 7 0.25 / 0.07 (bun lineage 0.92 / 0.99 / 1.00). The lane's question: which plan-free design can restore stopping and live-distractor rejection, and extrapolation.

**TH2 mirror-pair bound (math backing).** Model a plan-free policy at pass t as a choice among the b_t sentences that read the last written entity, made from (text, S_{t-1}) without the question. On the branch families (insert_render.py:135-143, 233) the correct branch and the distractor branch are exchangeable under renaming, so a forward-only policy has P(correct chain) <= prod_t 1/b_t, and P(stop at S_d) is a coin because "stop" and "continue to X1" are also a branch:
- H-EXTEND (X1 = S4 + c1, X2 = X1 + c2 after S4): one stop / continue branch at pass 5, bound 1/2 for the halt, 0 for a policy that never stops (gb: 0.08 / 0.02).
- H-LIVE (given Y; X1 = Y + c1; X2 = X1 + c2): the start branch at pass 1 (S_0 vs Y), bound 1/2 (gb: 0.40 / 0.41).
- H-CHAINSRC-LIVE (X1 = S_j + c right after S_j, X2 = X1 + c'): b_{j+1} = 2, bound 1/2 (gb: 0.06 / 0.41).
A **question-conditioned** policy (uses S_d: "current entity == target" for the halt, "is-ancestor-of-S_d" for selection) can reach 1. The text already ends with the question (nested_words.py:155, `text = " ".join([...] + [q])`), so every plan-free model sees S_d; gb did not use it.

Decision rule: a design is worth a full stage on the branch families only if its selection or halt is question-conditioned by construction (TH5: nothing to select; H8: one-hop halt check) or by training pressure (P9: the teacher's plan is distilled into a student that only has the question). Pre-stated predictions: H8 fixes H-EXTEND but not H-LIVE / CHAINSRC-LIVE (the halt is one-hop, selection is not); P9 passes all three only if a one-pass GPT-2 can compute is-ancestor-of-S_d at d <= 4 (the Lane-C PLAN-PROBE head >= 0.9 is the shared falsifier; run it first, $0.15); TH5 passes all three without any selection, and its risk is capacity and formation.

Order: PLAN-PROBE -> §261 pilot (parked, $0.3) -> H8 on gb_s1 ($0.25 / $0.4) -> P9 (if §261 PARKs / NOTEs) -> TH5 pilot. All from of_s* if §263 reads ORDER-FREE-C1 (the branch-family gates require TEST-PERM, which bun fails for a known reason); else from bun_s* with the TEST-PERM cell descriptive.

## 2. Designs

Common base: bun_s1's args (runs_fsq/bun_s1/args.json: K 8, d_max 4, bs 64, lr 1e-4, aux_step 1.0 reweighted, distract_p 1.0 mix s256,chain,givop, hold_inc 1.6 from step 500, write_src answer, cue step, init_model dch_s1_init), 8k steps.

### 2.1 P9 CUE-FADE

Pass t's cue is `cue_str(it, min(t, d))` = `" How many <u> does <S_min(t,d)> have?"`, 7 tokens, the name 6 before ':' (inject_st.py:44-46, 49-63; nk.py:25 CUE_NAME_OFF). CUE-FADE replaces the name token at pass t by a **fade token** with probability p_t(s) (per item, per pass, seeded coin `default_rng((seed, 26300, step, k, t))`):
- arm **P9-T**: fade token = S_d's name, i.e. the cue becomes `cue_str(it, d)` (the hold passes already see it). At p = 1 the pass input is byte-identical across passes and names exactly S_d: the §261 target_only condition in a 7-token form.
- arm **P9-G**: fade token = `" someone"` (GPT-2 id 2130, single token; "How many eggs does someone have?"). At p = 1: an information-free cue; the target is only in the text.
Both keep the cue length 7, so cue_inputs' one-padding assert (inject_st.py:76-79) and tok_aw's name assert (inject_aw.py:91-92, relaxed to "name or fade token") hold.

Schedule (late passes first): f(s) = clip((s - 500) / 5500, 0, 1); p_t(s) = clip(2 f(s) - (K - t)/(K - 1), 0, 1). f = 0.25: p_8 0.5, p_5 0.07, p_1..4 0; f = 0.5: p_t = (t-1)/7; f >= 1: all passes faded. Eval always at p = 1 (a new `o["cue"] = "fade_t" / "fade_g"` arms passes()' PLAN-FREE assert, inject_aw_eval.py:65-66).

Self-distillation to the plan-cued EMA teacher (1511.03643 privileged information; 1703.01780 mean teacher): teacher weights theta_bar <- 0.999 theta_bar + 0.001 theta per step, initialised to the init model; teacher forward no-grad on the **unfaded** inputs; L = L_dec + lambda_aux L_aux + L_inc + lambda_kd * mean over faded (item, t) of KL(p_bar_t || p_t), with p_t = softmax of the ':' logits over NUM_IDS (colon_logits, inject_aw.py:61-63) at pass t, lambda_kd = 1, temperature 1. Unfaded passes get no KD term (inputs identical, the term would only be EMA lag). Arms: P9-T, P9-G, and P9-T-NOKD (lambda_kd 0; the backlog's no-distil arm). Monitors (log per 1k bin): teacher WS under the step cue (d 3-4, passes 2-3), the faded share, KD loss, and the student's WS split faded / unfaded.

Shortcut-exclusion (exposed, eval-only): **QUESTION-ABLATE**: the same model on H-EXTEND / H-LIVE with the question sentence deleted from the text (cue_only-style input builder; the fade cue kept). If branch-family accuracy does not drop by >= 0.3, the model is not using the target and the pass is a sentence-order artefact.

### 2.2 H8 ANSWERABLE-NOW head

a_t = w . LN(h_11[':', pass t]) + b, one logit on the ':' hidden that pass_aw already returns (inject_aw.py:47-56). Label y_t = 1[argmax dec_logits(S_t) over NUM_IDS == ans] from the model's own decoder (no-grad, inject_ow.py:86-90), on a seeded quarter of each batch (16 items x K passes; ~+20% step time). L += lambda_h BCE(a_t, y_t), lambda_h 0.5; the head is non-zero-init (w ~ N(0, 0.02), b = -2, prior ~1/K), and its usage monitor is the train AUC per 1k bin (rule 8). Halt at eval: the first t with sigmoid(a_t) >= 0.5, else K (censored at K with the K answer; never_halt_n reported, halthead's convention, inject_aw_eval.py:884-916). Arms on gb_s1 (gb_s2 fails d 6 and is excluded): **H8-FROZEN** (everything frozen but the head, 1k steps, $0.25) and **H8-FT** (gradient into the trunk, 2k steps from gb_s1, $0.4). Prediction from the bound: H-EXTEND acc-at-halt rises to >= 0.8 (one-hop "current == target"), H-LIVE / CHAINSRC-LIVE stay <= 0.55.

H8 is a flag on any AW model: on bun it is the model's own halt that §264 needs if it reads PLAN-CLOCKED; on a §261 or P9-T model it is H7 TC-HALT.

### 2.3 TH5 DATAFLOW-ALL-WRITE

No cue. Pass input = [EOT] + enc(" " + text) + SFX (text ends with the question). At every pass the model writes at every sentence end p_s (the '.' tokens; the question's '?' excluded) into the §262 NK block (nk.py:44-62: k = normalize(W_n e(name)), delta rule beta 1), keyed by the entity the sentence defines (B for ops, the name for givens; nested_words.py:128-144 gives both), value v_s = g_s W_vn ln_w(h_11[p_s]) with a resolvability gate g_s = sigmoid(w_g . h_11[p_s] + b_g) (b_g = -2 init, non-zero w_g; usage monitor = mean g_s by (depth, pass)). The ':' DeltaMem write stays (the decoder's value stream). The NK read at every name-token position (block 6, nk.py:48-56) gives an op sentence its sources' values once written, so a sentence resolves at pass max(1, depth(s) + 1) with depth(given) = 0 and depth(op) = 1 + max depth of its sources: **passes = DAG depth + 1**, independent of order and of distractor chains (they resolve in parallel). Halt = the question key is resolved: r_t = ||S_n q_{S_d}|| >= theta (theta = the 5th percentile of resolved keys on the fit stream, pre-stated).

Aux (hand-built, same C4 scope as bun's per-pass aux): per pass one seeded sentence s of the item, target `GIVEN.format(A=name(s), v=v_s if resolved_t(s) else "?", u)` through aux_logits (inject_ow.py:92). Decoder target unchanged. Gates below force NK use (the §262 lesson: with a ':' value path the binding went unused; here the ':' write has no plan, so the decoder must look the target up by name). Capacity: 64-dim keys, <= ~16 names per item (d 8 + givens + distractors + inserts); MQAR-style crosstalk (2312.04927) is the pilot's risk, read as the NK-ablation drop by name count.

## 3. Code touch points (read)

- boxes/inject_aw.py: tok_aw 76-97 (cue builders; the fade token replaces `q[-7]`, assert 91-92 widened); args 211-249 (new `--cue_fade {"", "t", "g"}`, `--fade_win 500,5500`, `--kd 1.0`, `--kd_tau 0.999`, `--h8 0/1`, `--h8_lam`, `--h8_frac 0.25`, `--dataflow 0/1`); asserts 261-268, 290; lam_at 299-303 (f(s) beside it); draw_items 316-346 (the per-item, per-pass fade coins applied after perm / insert, with the DISTRACT "cues unchanged" assert 329-330 compared on the unfaded cue); loss_reg 361-419 (teacher no-grad loop and KD at 373-390; H8 BCE at 399-407 after the aux); loss_ask 421-431 (ask items keep their names: they are one-hop target questions already); hold_step 159-200 (the hold passes read xs[-1]: under P9-T that is the S_d cue, unchanged; under P9-G the faded cue at p = 1); ws_readout 135-152; pass_aw 47-56, colon_logits 61-63.
- boxes/inject_st.py: cue_str 44-46, tok_st 49-63, cue_inputs 70-80.
- boxes/nk.py: NK.key / read / write 44-62 (TH5 writes T > 1 keys per pass: a loop over sentence ends or a batched erase-then-write with distinct keys); InjectNK 70-80.
- boxes/deltamem.py: write_seq 179-220 (mask semantics for padded positions; TH5's ':' write unchanged).
- boxes/inject_aw_eval.py: passes 55-83 (assert 65-66); part_halt 812-826, hh_* 831-882, part_halthead 884-916 (H8's eval format); part_test_perm 1044-1075; part_insert_ho_s256 1077-1116 and part_insert_ho 1127-1140 (new `insert_ws`: ws_readout on the family items, X1's value = (v_j + c1) mod 11 from meta; new `h8_halt`; new `q_ablate`).
- boxes/insert_render.py: HO_FAMS 79-80, held_out 107-151, HO2_FAMS 233. boxes/nested_words.py: _statements 128-144, _finish 146-170. boxes/ow_words.py: _draw_program 45-86. boxes/inject_ow.py: tok_ow 63-71, dec_logits 86-90.
- theory/score_tc.py: the scorer template (read / score / fixtures). scratch/gradflow_check.py exists.

## 4. Gates (before any training output)

1. **Flag-off bitwise**: every new flag at its default reproduces bun's step-0 loss and the 20-step `--sched_scale` smoke bitwise (offbit_aw, the §261 build condition, Note 261-1).
2. **fp64 step-0 equivalence with the flag on**: P9 at step < 500 has p = 0 and teacher == student, so the KD term is exactly 0 and the loss equals bun's in fp64; H8 and TH5 add their terms, so the gate is "loss - lambda * new term == bun's loss" in fp64, and the DeltaMem state after pass K bitwise equal (TH5: the DeltaMem blocks; the NK block new).
3. **gradflow_check** on every new term (KD reaches the trunk through the faded passes only; BCE reaches w, b and, in H8-FT, the trunk; TH5's gate g_s and W_vn receive gradient from the decoder and aux).
4. **Smoke on the real data scale** (20 steps, bs 64, d 4 items with 2 inserts) and **mem_check** at d_max with the teacher (P9 holds a second parameter copy, ~0.5 GB) within the 21 GB budget, micro pinned.
5. Asserts: P9 the faded cue is byte-identical across passes at p = 1 and names S_d only (P9-T) / no entity (P9-G); H8 labels never read `it["d"]`; TH5 every write key is a name token and the question is not a write position.
6. Fixtures 6/6 or more per scorer, committed before launch; eval parts registered in PARTS / AW_ONLY.

## 5. Criteria (per seed; Wilson 95% CI on every binding cell; NEAR if the bar lies inside the CI)

| Design / stage | Cell | Bar | Type |
|---|---|---|---|
| P9 pilot (2k, seed 1, fade window 250-1500, eval at p = 1) | TEST2 d 2-4 | >= 0.90 each | GO |
| | WS d 3 passes 2-3 | >= 0.50 | GO |
| | H-EXTEND, H-LIVE acc8 | >= 0.60 each | GO |
| | PARK | d 3 < 0.20, or H-EXTEND < 0.30 and H-LIVE < 0.50 | |
| P9 full (2 seeds x 3 arms, 8k) | PLAN-FREE-C1: TEST2 d 1-4 >= 0.95, NOT-DECODER <= 0.10, SWAP4 >= 0.90, LP5-D4 >= 0.80, PLAN-FREE assert | binding | |
| | CONTENT: H-CHAINSRC-OP >= 0.85 (min pos >= 0.80); H-EXTEND >= 0.80; H-LIVE >= 0.80; H-CHAINSRC-LIVE >= 0.80 | binding | |
| | MECH: H-EXTEND ':' argmax at passes 5-8 == answer on >= 0.80 of items (insert_ws) | binding (gate on the mechanism) | |
| | ORDER: TEST-PERM d 2-4 >= 0.85 (binding if init = of_s*, else descriptive) | | |
| | C2: d 6 >= 0.80; C6: K' 64 = K' 8 - 0.02 at d 1-4 | binding | |
| | QUESTION-ABLATE drop >= 0.30 on H-EXTEND | binding (shortcut exclusion) | |
| | Label PLAN-FREE-CONTENT-C1 per seed; arm k/2; KD effect = P9-T minus P9-T-NOKD on CONTENT, descriptive | | |
| H8 (gb_s1) | HALT-OWN: Spearman(mean khalt, d) over d 1-6 >= 0.8; acc_halt >= acc_k8 - 0.02 at d 1-4 | binding | |
| | H-EXTEND acc-at-halt >= 0.80 | binding | |
| | H-LIVE, CHAINSRC-LIVE acc-at-halt | predicted <= 0.55 (bound); reported | descriptive |
| | PARK if val AUC < 0.70 at d 5-6 | | |
| TH5 pilot (2k, seed 1, from bun_s1 strict except nk.* / gate) | TEST2 d 2-4 >= 0.50; NK-ablation drop >= 0.20 at d 3; resolved-at-depth: mean g_s >= 0.5 for resolved and <= 0.2 for unresolved (depth, pass) cells, two-sided | GO | |
| | PARK: d 3 < 0.20 or ablation drop < 0.05 | | |
| TH5 full (2 seeds, 8k) | C1 battery as above + NK-ablation at K' 8 <= chance + 0.10; CONTENT all four families >= 0.80; TEST-PERM d 2-4 >= 0.85; C2 d 6 >= 0.80 at K' 8, d 7 at K' 9 descriptive; C6 K' 64 hold; halt r_t: khalt = d + 1 on >= 0.8 of clean d 1-6 items | binding | |

**OC (outcome priors, stated before any build):** P9-T pilot GO 0.35 (it is §261 with a curriculum and a teacher; §261's own prior 0.4 is conditional on the §263 fix), P9-G GO 0.15 (the target must be fetched from the text; gb shows no use); full CONTENT pass given GO 0.5; KD effect >= 0.05 on CONTENT 0.5. H8: HALT-OWN 0.6, H-EXTEND >= 0.8 at halt 0.5, H-LIVE <= 0.55 0.8 (the bound). TH5 pilot GO 0.25 (formation of T > 1 keyed writes from a ':'-trained init is the risk; §262's binding went unused), full pass given GO 0.4.

**Censoring:** VOID-RERUN once per cell on host failure before the first eval; a cell whose fade / g_s monitors are silent for 2k steps is a build fault, not a result; never-halt items count at K with the K answer; P9 arms are scored at p = 1 only (intermediate-p evals descriptive); no seed rescue (rule 4); a NEAR cell is reported NEAR, not passed.

## 6. Scorer sketch

theory/score_p9.py, score_h8.py, score_th5.py in score_tc.py's form (read / score / fixtures; results in theory/results/). Fields: eval_acc.json by_d[d].acc; eval_ws.json by_d["3"][t]; eval_insert_ho.json fam[F].acc8 and by_pos; eval_insert_ws.json fam["H-EXTEND"].hold_frac (new); eval_test_perm.json by_d[d].acc_kprime[K]; eval_swap_kd / lp5h / long_k128 / stab as score_of.py reads them; eval_q_ablate.json fam[F].acc8; eval_h8_halt.json {d: acc_k8, acc_halt, mean_khalt, never_halt_n, head.val_auc} and fam[F].acc_halt; eval_nk_ablate.json by_d[d].acc; gates_pass.json {fade_identical, h8_no_d, th5_keys}. Labels: GO / PARK / NOTE (pilots); PLAN-FREE-CONTENT-C1 k/2, HALT-OWN, DATAFLOW-C1 k/2; NEAR flags. Fixtures: GO; PARK by chain loss; PARK by family floor; NOTE partial; assert-missing not GO; pending; NEAR on a bar inside the CI.

## 7. Cost (RTX 3090 ~$0.2/h; bun's 8k ~ $0.8)

P9: pilot $0.3; full 2 seeds x (P9-T, P9-G at ~+35% for the teacher, P9-T-NOKD) ~ $6.5 + evals $0.6; guard $9. H8: $0.25 + $0.4 + evals $0.2; guard $1. TH5: pilot $0.4; full 2 x ~$1.2 + evals $0.5; guard $4. PLAN-PROBE first, $0.15. Lane total <= $15 on the pilot-gated path.

## 8. RED TEAM (separate sub-agent)

(pending)

## 9. arXiv ids (each fetched at arxiv.org/abs/<id>, title checked)

2405.14838 From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step (P9 schedule); 1511.03643 Unifying distillation and privileged information (P9 teacher); 1703.01780 Mean teachers are better role models (EMA teacher); 1503.02531 Distilling the Knowledge in a Neural Network; 2207.07061 Confident Adaptive Language Modeling (H8 halt); 2107.05407 PonderNet; 1603.08983 Adaptive Computation Time; 2402.06457 V-STaR (self-labelled verifier); 2607.20519 Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts; 2505.12514 Reasoning by Superposition (TH2 / TH5 parallel frontier); 2312.04927 Zoology (MQAR recall capacity); 2310.17191 How do Language Models Bind Entities in Context?; 2412.06464 Gated Delta Networks; 2505.20896 How Do Transformers Learn Variable Binding in Symbolic Programs?; 2502.19907 Order Doesn't Matter, But Reasoning Does; 2410.05464 Progressive distillation induces an implicit curriculum; 2604.04902 Are Latent Reasoning Models Easily Interpretable?
