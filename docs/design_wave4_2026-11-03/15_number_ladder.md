# NUM-100: integers 0-99 in one token, backward-uniform values (wave-4 design; 2026-11-03)

**Backlog IDs:** Lane D3 NUM-100 (GSM8K ladder stage 1, §12.6); NR1 NUM-CODE-PROBE; NR5 CARRY-STRATIFIED (fields now); next rung NR4 R2L-SPACED, DIGIT-AUX-10K, NR2 FONE-AUX. One change vs bun (§259): the value domain. Lead: this file only. Status: draft for Theory → Audit registration; nothing here has been run.

## 1. Goal

Leave restriction 3 (mod-11) with one change: every quantity is an integer 0..99 rendered as its single GPT-2 token (verified: all of " 0".." 99" are one token, 100 distinct ids 352..10495; 495 of " 100".." 999" are not). Ops stay the bun REG law (add_c, sub_c, mul_c, add_v, sub_v, mul_v, add_s; no forced mul_s) with no modular wrap, so a step is real 2-digit arithmetic with carries. The question is whether the bun loop (K 8, cue, ':' write, hold penalty, §257 inserts) carries over when the per-pass write must hold a 100-way value and the step needs carry handling, and how the write encodes the number (Fourier vs digit code), which decides the next rung.

History check: the 0..99 range was the original NESTED-WORDS domain (§223). §223/§226 showed the one-pass d 2 failure was composition, not value size (mod-11 one-pass d 2 also 0.53-0.57; THEORY.md §226 row). So the range was never tested on the loop recipe; this is the test.

## 2. Design

### 2.1 Generator `boxes/ow_num100.py` (new; wraps nested_words like ow_words.py does)
Same API as ow_words (Stream, make, redraw, test_set, pair_candidates, solve, set_sha, P, NAMES, GIVEN, QUESTION, FRAME_*, _decorate, NW), so the trainer / eval bind it by module rebinding (§3). Draw order per item, so the op law is value-independent:

1. **Ops first.** ops_t for t = 1..d uniform over OPS_REG (ow_words.py:54, 69-79 rule: d 1 → mul_v; add_s only at t ≥ 2 with j ~ U{0..t-2}). Identical to bun's REG law (mul_s_p 0, dag True, order_mode topo).
2. **Answer uniform.** S_d ~ U{0..99}.
3. **Backward-uniform pairs.** For t = d..1: draw (S_{t-1}, b_t) uniformly from PRE[op_t][S_t] = {(x, b): x ∈ 0..99, b ∈ B(op), op(x, b) = S_t, x ≠ 0 if mul} with the nested_words operand ranges (nested_words.py:91-92): add/sub constants and givens B = 1..20, mul B = 2..9. add_s: (x, v) uniform over x + v = S_t, x, v ∈ 0..99, and S_j is **pinned** := v. When the recursion later produces S_j (as the x of step j+1), the pair draw at step j+1 is restricted to x = pinned value; S_0 pins likewise.
4. **Restart** the whole program if any PRE set (or pinned set) is empty; count restarts (Stream.stats, as ow_words.py:112).
5. Names, unit, frames, 0-2 distractors (values U{1..99}), topo order: exactly ow_words.make's draw sequence (ow_words.py:108-122).

Equations: add x + b, sub x − b (≥ 0 by construction), mul x·b (≤ 99 by construction); no `% P`. Leak statement: the op is drawn before any value; values condition the op only through the restart event, i.e. PRE empty: add needs S_t ≥ 1; sub needs S_t ≤ 98; mul needs a divisor of S_t in 2..9 with x ≠ 0 (so S_t ∈ {0, 1} ∪ primes > 9, 23 values, never follow a mul). This is the arithmetic itself, not a sampling artefact; forward rejection (nested_words.py:107-119) adds the artefact that P(mul accepted | x) collapses for x > 49 and values drift small. Both are measured in stage 0 (G-MI), not assumed.

New item keys: `vals: 100`, `mod: None`, `carry: [c_1..c_d]` with c_t = carries in step t (add: 1 if x%10 + b%10 ≥ 10; sub: 1 if x%10 < b%10 (borrow); mul: number of positions i ∈ {0, 1} where digit_i(x)·b + carry_in ≥ 10), `n_carry = sum`, `carry_last = c_d`; `wrap_final / wrap_any` kept as False (read by part_frames, inject_aw_eval.py:980, and the §241 re-render, :578-590). R_step / cot unchanged in form ("<S_t> has <v> <u>."), so the sequence-scored aux (inject_ow.py:92-102) and vidx (tok_ow, inject_ow.py:70: the token after " has") need no change.

Inserts (bun: --distract_p 1 --distract_mix s256,chain,givop): the only mod-11 literals are the value draws v U{1..9}, c U{1..10 − v} (insert_render.py:36, 211, 215; held-out families :114-125, :244-247). Under vals 100: v U{1..99}, c U{1..20}. Inserted X values are never rendered or asked, so no range constraint is needed (the chain insert "X has c more than S_j" stays unrendered as a value).

Pairs (build_sets, inject_ow.py:137-155): redraw = the backward draw on the recipient's skeleton (ops, js, order fixed); the exact-token-length rule (:148) holds because every value is one token; cf (inject_ow.py:117-124) uses OW.apply, which is the integer apply here. Kept rate is a stage-0 statistic (bun: ~2050/3000 per d).

### 2.2 Model: no architecture change
The ':' readout is the full-vocab LM head at ':' (colon_logits, inject_aw.py:61-63) sliced by NUM_IDS (ws_readout :138, :147; eval pred_of, inject_aw_eval.py:41; loss targets inject_aw.py:363, :378, :389, :417, :424). With NUM_IDS of length 100 the readout width is 100 and chance is 0.01. The write stays the ':' hidden state; decoder and aux unchanged; hold penalty unchanged.

### 2.3 Arms
| Arm | Run | Init | Change vs bun_s1 args (args_diff must show only these) |
|---|---|---|---|
| step-0 ref | n100_ref_s1 (eval only) | bun_s1 model.pt | --vals 100 sets; no training (the mod-11 model can only emit 0..10) |
| pilot | n100_pilot_s1 | --init_model bun_s1 (sha 798f9259...) | --vals 100, --steps 8000 --stop_at 4000 |
| full | n100_s1 (resume the pilot ckpt), n100_s2 (from bun_s2) | bun_s* | --vals 100, 8k |

Everything else is bun_s1/args.json as stored (think_k 8, d_max 4, lr 1e-4, bs 64 micro 2, aux_step 1 reweight, cue step, write_src answer, mul_s_p 0, distract_p 1 mix s256,chain,givop, hold_inc 1.6, init_trunk s233_askname_s1). Warm start from bun is the point: the selection / write / hold mechanism exists; only the value code must be re-learned. A cold arm (from the askname trunk) is not run: the lineage took 68k steps and would confound range with formation.

## 3. Code touch points (read)

- **NUM_IDS copies.** nested170.py:13-21 defines P = 11, VOCAB, NUM_IDS (len 11) and asserts one token each; it stays. `boxes/num100.py` (new) defines NUM_IDS_100 with the same assert. Four modules hold a copy by `from boxes.nested170 import NUM_IDS`: inject_ow.py:39, inject_st.py:35, inject_aw.py:36, inject_aw_eval.py:33. A new `inject_aw.set_vals(100)` rebinds `NUM_IDS` in all four namespaces and `OW` (inject_aw.py:33, inject_ow.py, inject_st.py, inject_aw_eval.py:32, insert_render.py, perm_render.py, ask_render.py) to ow_num100; called in main after parse (before build_sets, inject_aw.py:277) and by the eval loader from args.json `vals`. With `--vals 11` (default) nothing is rebound.
- **Hard-coded 11s.** inject_aw_eval.py:768 `P = np.zeros((N, K + 1, 11))` → len(NUM_IDS); :795 `xe = (xh + 1 + rng.integers(0, 10)) % 11` → `% len(NUM_IDS)` with integers(0, len − 1); :587 already uses OW.P (→ 100); docstrings "11 number tokens" (inject_aw.py:11, :136; inject_aw_eval.py:767, :1118). inject_ow.py:43 asserts " 0".." 10" single tokens: unchanged, still true.
- **Cue guard.** inject_st.py:54-57 G-CUE asserts no number token (set(tk) & NUM_IDS) and CUE_LEN in every cue; with 100 ids the assert is the gate (names / units are words). inject_aw.py:92 and :105 name-token checks unchanged.
- **Sets.** build_sets (inject_ow.py:127-171) gets `vals`: new seed constants SEED_*_N100 (grep-unused), `shas["set_law"] = "reg100"`; ws_deep's sha assert (inject_aw_eval.py:1114-1115 vs scratch/s256_ho_shas.json) branches to a new committed scratch/n100_shas.json; insert_ho / held-out families (IR.insert_eval_sets) under the 100 draws.
- **Generator-side constraints that encode mod 11:** ow_words.py:26 (P), :37-38 (apply % P), :41-42 (_operand 2..P / 1..P), :45-85 (_draw_program, forward), :117 (dis_vals), :125-144 (redraw), :99-105 (wrap flags). All replaced by ow_num100's versions; ow_words.py is untouched.
- **Eval parts (bun's set + new):** acc (:104-117; items_rec gains `carry`, `n_carry` per item — the CARRY-STRATIFIED fields, logged now), ws (:352), swap_kd (:391-420), lp5h, cue_only, halthead, long_k128, and new `num_probe` (§5). parts c7 / probe / normctr / halt_inc need only the literal fixes above and are not run in stage 1.
- **Scorers that assume 11:** score_bu.py has no 11 literal (grep), nor do the aw-era scorers; the 20 older scorers that do (score_s170 … score_dc / score_df / score_em, grep list) belong to closed sections and are not touched. score_n100.py is new.

## 4. Gates

1. **G0 generator selfcheck** (local CPU, $0): 100k items d 1..8: OW.solve(text) == answer (the regex solver with integer apply), all values in 0..99, every carry field recomputed by an independent digit routine, no restart loop > 10^4, R_t solve == answer for t < d.
2. **G1 stats (binding on the generator, pre-stated):** on 200k training-law items: (a) answer histogram: every value frequency in [0.007, 0.013]; (b) G-MI: I(op_t; S_t) ≤ 0.10 bit and I(op_t; S_{t−1}) ≤ 0.10 bit for every t ≤ 4 (reported next to the same statistics for nested_words' forward-rejection draw, as the contrast); (c) restart rate ≤ 0.5 per item; (d) pair kept rate ≥ 1000 per d for d 3..8; (e) carry share: P(n_carry ≥ 1 | d 4) reported (expected ~0.6-0.8; the stratified read needs ≥ 150 items per stratum, else the stratum is merged and labelled).
3. **G2 flags-off equivalence:** with --vals 11 (default) the args.json set shas equal bun_s1's (test2_sha 3b9101c3…, pairs2_sha 2fe9f87e…), a 20-step CPU smoke (--sched_scale, fp32) gives bitwise-identical log.jsonl to the pre-edit module, and `eval_acc` on bun_s1 (cloud, $0.1, det wrap) is byte-identical before / after the literal edits. The rebinding is inert at 11 by construction; this gate catches the literal rewrites.
4. **G3 tokenization:** every rendered text's token count equals its mod-11 skeleton's (one token per value); G-CUE assert passes on 10k items; no rendered sentence contains a 3-digit number.
5. **G4 mem_check** at vals 100 (same lengths → same peak; assert within 1%); micro pinned at 2 as bun.
6. **G5 gradflow / no_grad:** no new loss term, so no gradflow check; num_probe is ast-checked @no_grad (the copied-module rule).
7. **G6 scorer fixtures:** score_n100.py fixtures PASS / FAIL / NEAR / NO-FORM / CLIFF before launch.

## 5. NUM-CODE-PROBE (eval part `num_probe`, no_grad, pre-stated here)
Inputs: 2,000 TEST2-100 d-2 items (and d-4, pass 4). u_t = ln_w(h[':']) at pass t (the write input, as part_probe reads it, inject_aw_eval.py:321-323). Split 1,000 fit / 1,000 test, seed fixed.

- **Decoders of v_t from u_t:** (100) a 100-way multinomial ridge-logistic probe; (F) Fourier: ridge to [cos, sin](2πv/T), T ∈ {2, 5, 10, 50, 100} (10 dims), decoded by nearest point on the T = 100 ∘ T = 10 helix; (D) digit: two 10-way logistic probes (tens, units), decoded as the digit pair; (L) linear: ridge to v, decoded by rounding.
- **Use test (causal):** project the fitted F subspace (10-d), the D subspace (top-20 of the digit probes' weights), a magnitude direction (L, 1-d) or a random subspace of matched dimension (3 draws) out of u_t before the write (a `write_edit` hook on the gathered ':' row in pass_aw, inject_aw.py:47-56), and read the WS at pass t + 1 and the decoder answer at K. Drops relative to intact.
- **Labels:** UNRESOLVED if (100) test acc < 0.80. FOURIER if acc(F) ≥ acc(D) + 0.15 and drop(F) ≥ max(2·drop(rand), 0.10). DIGIT symmetric. MIXED otherwise. The label is descriptive and routes the next rung; the caveat that a T = 10 Fourier pair and a units-digit one-hot are near-equivalent is disclosed, so FOURIER vs DIGIT hinges on the T = 50 / 100 (magnitude) components and the use test, not on the T = 10 circle.
- **Error histogram (from eval_acc items, scorer-side):** errors split into unit-digit-only, tens-digit-only, ±10 (carry) and other, overall and by n_carry.

## 6. Criteria (TEST2-100, n 1,000 per d; Wilson 95% CIs; NEAR when the CI straddles the bar; both seeds unless stated)

| ID | Kind | Measure | Bar |
|---|---|---|---|
| P0 | pilot gate (4k, seed 1) | TEST2-100 d 1 acc; d 2 acc; WS pass 2 at d 2 (part ws) | GO: d 1 ≥ 0.90, d 2 ≥ 0.70, WS_2(d 2) ≥ 0.60. PARK: d 1 < 0.80 (arithmetic not learned → FONE / DIGIT-AUX rung first). Else CONTINUE to 8k, no re-draw of bars |
| B1 | binding | d 1 acc | ≥ 0.97 |
| B2 | binding | d 2, 3, 4 acc | ≥ 0.90 each |
| B3 | binding (mechanism at depth) | WS pass t at d 4, t = 1..4 | ≥ 0.85 each |
| B4 | binding (C1) | swap_kd d 4: swap_k follow_U; swap_mid at K' = K follow_U | ≥ 0.80 and ≥ floor + 0.50; ≥ 0.70 and ≥ floor + 0.40 |
| B5 | binding (carry) | pooled d 1-4: acc(n_carry ≥ 2) − acc(n_carry = 0) | ≥ −0.10 (else CLIFF) |
| B6 | binding (retain) | TRANSFER frames 6-7 d ≤ 4; cue_only d 4 | ≥ 0.85; cue_only ≤ 0.15 (chance 0.01) |
| D1 | descriptive | d 5-8 acc, WS d 6 passes 5-6 | reported; "C2 retained" label if d 6 ≥ 0.80 in both seeds (no C2 claim at 2 seeds) |
| D2 | descriptive | halthead r = 3 (C5), long_k128 hold at d 1-4 (C6) | reported vs bun |
| D3 | descriptive | num_probe label; error histogram | FOURIER / DIGIT / MIXED / UNRESOLVED |
| D4 | descriptive | step-0 ref acc; lp5h | reported |

Verdict: NUM-100 PASS = B1-B6 in both seeds; NEAR items re-read on a 2,000-item TEST2-100 draw (pre-stated seed) once.

**OC (outcome priors).** G0-G1 pass 0.9. Pilot GO 0.55 (PARK 0.2, CONTINUE 0.25). B1-B2 | GO 0.6. B3 | B2 0.75. B4 | B3 0.8. B5 no-cliff 0.45 (a cliff is the modal outcome: 2510.00184, 2402.14903 show carry / long-range digit dependence is the hard part). D3: FOURIER 0.35, DIGIT 0.30, MIXED 0.30, UNRESOLVED 0.05 (2406.03445 / 2502.00873 vs 2410.11781). D1 d 6 ≥ 0.80 both seeds 0.3. P(NUM-100 PASS) ≈ 0.2; P(a decision-changing read: PARK, CLIFF or a probe label) ≈ 0.85.

**Decision rules.** PASS + no cliff → BRANCH-DAG (D4) on 0..99. CLIFF or MIXED/DIGIT → DIGIT-AUX-10K next. PARK or UNRESOLVED → FONE-AUX. NO-FORM in one seed → no seed rescue; the third seed only with Audit approval.

**Censoring.** VOID-RERUN once per cell on host failure / non-finite loss (the aborted logs kept). A seed with WS pass 2 at d 2 < 0.30 at 8k is NO-FORM (a fail, reported, no rescue). The 4k pilot read is a gate, not a binding read; binding reads are at 8k only. If G1 fails the generator is redesigned and the registration amended before spend; bars are not moved after any output. Items with a merged carry stratum are labelled, not dropped.

## 7. Scorer sketch: theory/score_n100.py
Reads: `eval_acc.json` (acc_by_d; items[d].pred / gold / carry / n_carry / sig), `eval_ws.json` (by_d[t]), `eval_swap_kd.json` (by_d["4"].swap_k.follow_U / floor, swap_mid[K].follow_U / floor), `eval_transfer.json`, `eval_cue_only.json`, `eval_halthead.json`, `eval_long_k128.json`, `eval_num_probe.json` (acc100, accF, accD, accL, drops{F, D, L, rand[3]}), `args.json` (vals == 100, init_model_sha256, args_diff keys ⊆ {vals, steps, stop_at, resume, out, init_model*}). Per-seed table B1-B6 with Wilson CIs and NEAR; labels CLIFF, NO-FORM, FOURIER / DIGIT / MIXED / UNRESOLVED; the error histogram; the final line "NUM-100 k/2 (labels)". Fixtures: pass, fail-B2, near-B3, cliff, no-form, unresolved, args-diff-violation (7).

## 8. Cost (RTX 3090, ~$0.2/h, bun's 0.62 steps/s)
Stage 0: $0 local CPU + $0.1 eval-bitwise on bun_s1. Pilot: 4k steps ≈ 1.8 h ≈ $0.36 + eval $0.05 + step-0 ref $0.05. Full: s1 resumes 4k more + s2 8k = 12k steps ≈ 5.4 h ≈ $1.1; evals (8 parts × 2) ≈ $0.4. Total ≈ $2.1; guard $3. Build: ~450 lines (ow_num100 ~180, num100 + set_vals ~60, build_sets / seeds / shas ~40, literal fixes ~10, num_probe + write_edit ~120, score_n100 ~200 + fixtures).

## 9. Next rung (not registered here)
- **R2L-SPACED (NR4), prerequisite for > 99:** values rendered as reversed spaced digits (" 4 2 1" for 124) in givens, R_step and the answer; the aux is already sequence-scored, but the ':' readout and decoder become 3-token greedy decodes (nested170.aux_greedy pattern, :274-281) scored by exact match; the one-token-per-value length invariants (pairs :148, G3) are dropped. 2402.14903 / 2307.03381 motivate right-to-left.
- **DIGIT-AUX-10K:** a zero-init linear head on u_t to 4 × 10 digit logits (CE on the digits of v_t); rule 8 → usage monitor (the head's gradient norm share) and a non-zero-init arm; plus a digit-split arm (rendered digits).
- **FONE-AUX (NR2):** MSE from u_t to [cos, sin](2πv/T), T ∈ {10, 100, 1000} (2502.09741; 2006.10739); the NUM-100 probe label picks between the two.

## 10. Verified arXiv ids (abs pages fetched 11-03; titles match)
2406.03445 Zhou, Pre-trained LLMs Use Fourier Features to Compute Addition · 2502.00873 Kantamneni, Language Models Use Trigonometry to Do Addition · 2410.11781 Levy, LMs Encode Numbers Using Digit Representations in Base 10 · 2605.01148 Feucht, Arithmetic in the Wild (Llama base-10 addition for cyclic concepts) · 2301.05217 Nanda, Progress measures for grokking · 2410.21272 Nikankin, Arithmetic Without Algorithms (bag of heuristics) · 2502.09741 Zhou, FoNE · 2006.10739 Tancik, Fourier Features · 2405.17399 McLeish, Transformers Can Do Arithmetic with the Right Embeddings · 2310.13121 Quirke, Understanding Addition in Transformers · 2307.03381 Lee, Teaching Arithmetic to Small Transformers · 2402.14903 Singh, Tokenization counts (R2L) · 2510.00184 Bai, Why Can't Transformers Learn Multiplication (running-sum aux) · 2305.15408 Feng, Towards Revealing the Mystery behind CoT (Alg. 1 backward sampling) · 2503.07604 Lin, Implicit Reasoning in Transformers is Reasoning through Shortcuts.

## 11. RED-TEAM
(pending: a separate sub-agent's critique is appended below)
