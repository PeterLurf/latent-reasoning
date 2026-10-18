# TEXT-SELF (proposed §218): a matched text chain-of-thought self-training baseline for the latent self-depth line — DRAFT

Clock 2026-10-18 12:04 EDT (`date`), repo HEAD 598e75de. Opus design subagent for Theory. **$0: no GPU, no repo edits** (CPU checks ≤ 2 threads, idle priority, `python -B`; every output under research6/).
**Status: DRAFT, not registered, not red-teamed, not proof-checked.** Written section by section.
Inputs: THEORY §215, §216, §217 and notes 215-1…215-8, 216-1…216-7, 217-1, the §215 FT-SELF / pair results and the §216 s1 round-2 result; SELFDEPTH_v1 (5ded8703, frozen); SD2_v1 (e9ade2ec, frozen); LIT_ITERATED_SELFTRAIN_2026-10-17, LIT_FAITHFULNESS_SELFIMPROVE_2026-10-18 (11966136), LIT_STEPFACTOR_D1_2026-10-14; boxes/nested170.py (§170 TEXT-COT arm), boxes/core.py (GPT2KV), boxes/selfdepth.py, scratch/s216_build_pool.py, boxes/vbank_eval.py, theory/score_s215.py / score_s216.py; boxes/runs_fsq/s170_cot_s1 / s170_cot_s2 (args.json, args_eval.json, eval.json, log.jsonl, mem_check.json).
Tags: [PROVED], [PROVED UNDER X], [DERIVED script] (exact arithmetic on measured inputs; script named, JSON beside it in research6/scripts/), [MEASURED file], [ASSUMED], [CONJECTURE].

---

## 0. One screen

| item | TEXT-SELF draft |
|---|---|
| Question | Run the latent self-depth schedule (§215 round 1, §216 round 2) on a **text chain-of-thought** model of the same task, round for round: the same windows (n 9–12, then ⅔ n 13–16 + ⅓ n 9–12), form-only filter, gate rule, mixture (p_self 0.5, gold n ≤ 8 replay), 12k steps, lr 3e-5 and **identical self exposure** (5,992 self steps = 383,488 items at the same step indices). Does text-CoT self-training (a) absorb each window and (b) move its 50 % depth by as much as the latent loop did (n50 10.31 → 15.46 → 20.54; n50 / support 1.29 / 1.29 / 1.28)? Literature found no such matched comparison (LIT_FAITHFULNESS_SELFIMPROVE) |
| Base | `s170_cot_s1` (§170 TEXT-COT: GPT-2 small, Feng chain, gold n ≤ 8, 48k, RTX 3090). Its bf16 curve: a(9..12) 0.995 / 0.707 / 0.234 / 0.123, **n50 10.44**, i.e. **parity with D3-T at round 0** (10.31). The fp32 re-eval (G-T19) replaces the bf16 curve for every bar |
| Cells | `s218_t1_s1` (FT from the base) → `s218_t2_s1` (FT from T1); eval-only `s218_base_s1`. Recommended second lineage from `s170_cot_s2`, in parallel, pre-stated; the primary comparison is s1 vs s1 |
| Labeller | The round's text model's **own greedy single step, re-rooted at every step** (REROOT: the latent labeller's protocol), computed exactly by **TF-GREEDY** (teacher-forced verification of the unique form-valid candidate; 2 batched forwards per step; 90 / 90 agreement with KV-cache generation on CPU). Base precondition [MEASURED, CPU, 200 fresh items per n]: REROOT keep **1.000 / 0.995 / 0.975 / 0.970 at n 9–12** (1 kept-wrong of 788), so round 1 is labellable. The whole-chain protocol (FULL, Lee's) keeps only **0.54 / 0.05 / 0.00 at n 10 / 11 / 12**, so it is a descriptive row only. **Round-2 risk:** the base's single step has a copy-length reach (keep 0.955 / 0.875 / 0.775 / **0.630** at n 13–16, against D3-T's 0.972–0.917), so round 2 rests on T1's own reach. A failed GATE-T2 reads **TEXT-LABEL-STALL** (P ≈ 0.15) |
| Filter | KEEP-T ∧ CV-T: every step is exactly one leftmost-ready reduction, with an exact copy elsewhere, any value token in the slot, canonical, correctly terminated, exactly n steps. **Values and the final answer are never checked** (D1: the visible operands make a value check trivial for text; it would be a gold check the latent arm never had) |
| Context | The chain needs P + C ≈ 1.5 n² + 3n rows; GPT-2 has 1,024. **Only n 24 overflows** in the test set (29 / 1,000; n ≤ 23 all fit; max 1,085). Overflow items are failures (an intrinsic token cost), with CTX-FEASIBLE accuracy beside. A third text round is infeasible (n 25: 55 % fit, n ≥ 28: 0 %): **CTX-BOUND, two rounds only** |
| Readings | Per cell: T-ABSORB / T-ABSORB-2 (≥ 0.95), T-RETAIN, T-DBS / T-DBS-2 (≥ 0.80), n50, WIDTH, mechanism by the SD2 bar formulas on the text curves. **Comparison:** Δ_r = n50_text − n50_latent; **PARITY iff \|Δ\| ≤ 0.5 at both rounds**; TEXT-AHEAD / LATENT-AHEAD by round 2's sign (round 1 not contradicting); CROSSED otherwise. Descriptive: TOKEN-COST per correct answer (text sequential steps ≈ 1.5 n² vs latent n + 2; FLOPs ≈ ½ of latent), DEPTH-PAIRED per item, FAIL-POS. Never worded as C1 / C2 / C4 |
| Hypotheses | Additive text extrapolation (N-SHIFT / POS-ADD: Δ ≈ −1, then −2) vs proportional (N-PROP / POS-PROP: \|Δ\| ≤ 0.34) [DERIVED panels]. Precondition: at n 10–14 the base's FULL chains break near a fixed **row ≈ 170–200** (≈ 1.2–1.35 × the 147 trained rows) while the failing stage falls from 8 to 6, i.e. failure is keyed to training frontiers (rows, copy length), not to step count [DERIVED ts_failpos] |
| OC headline | Given both rounds readable (P 0.68) and the stated priors: **LATENT-AHEAD 0.59, PARITY 0.19, TEXT-AHEAD 0.18, CROSSED 0.03** (lineage s 0.35). Unconditional: 0.40 / 0.13 / 0.12 / 0.02, with ROUND-1-ONLY / TEXT-LABEL-STALL 0.22 and a censored text cell 0.10. P(PARITY \| no difference) 0.92 / 0.61 / 0.39 at s 0.2 / 0.35 / 0.5; P(PARITY \| 1-n difference) ≤ 0.07; P(LATENT-AHEAD \| additive panels) ≥ 0.98. POOLED (two lineages): P(PARITY \| no difference) 0.82 at s 0.35. Per cell: P(T-SELF) 0.85, P(T-DBS) 0.15, P(T-SELF-2) 0.55, P(T-DBS-2 ∧ T-SELF-2) 0.21 |
| Cost | **$1.7–5.0 per lineage** (Plan A: one 3090 per lineage for labelling, gates, training and fp32 evals, 10–19 h); **family (s1 + s2) $3.5–10.0, cap $12**; or Plan B, local evals: $0.6–1.9 per lineage plus 8–11 h of 3070. Under the $30 auto-approval; no MEM-TRADED user OK needed |

---

## 1. Design (draft registration text)

### 1.1 What is matched, and what cannot be

| element | latent lineage (§215 / §216, s1) | TEXT-SELF (proposed) | matched? |
|---|---|---|---|
| Base | D3-T s1 (`s213_d3t_s1`), SLVA loop, gold n ≤ 8, 48k × bs 64, lr 1e-4, RTX 3090 | §170 TEXT-COT s1 (`s170_cot_s1`), GPT-2 small, Feng chain, gold n ≤ 8, 48k × bs 64, lr 1e-4, RTX 3090 [MEASURED s170_cot_s1/args.json] | yes (task, generator, gold range, steps, bs, lr, card class) |
| Base depth | n50 10.31, n90 9.07 [MEASURED, SD2_v1 §3.4] | n50 **10.44**, n90 9.33 (s1); 10.60 / 9.60 (s2) [DERIVED ts_panels from MEASURED eval.json, **bf16**, RTX 3070] | n50 / B 1.29 vs 1.30 (s1): **parity at round 0** (descriptive; the fp32 re-eval, §1.6, replaces the text numbers) |
| Labeller | the round's model, **single step re-rooted at every step** (SELFDEPTH_v1 §2.1) | the round's text model, **single greedy step re-rooted at every step** (REROOT), computed by TF-GREEDY (§1.2) | yes: the same procedure (own single step, fresh root each step); the step itself is text, not a latent decode |
| Filter | KEEP ∧ CV (form only; values never checked) | KEEP-T ∧ CV-T (form only; values never checked) (§1.3) | yes |
| Gate | GATE-1 / GATE-2 / GATE-1′: 2,000 held-out per n; π_pool ≥ 0.995 ∧ π_n ≥ 0.99 ∧ π^E_pool ≥ 0.99 ∧ keep_n ≥ 0.70 | the same rule and counts (π^E ≡ π for text, §1.3) | yes |
| Round-1 window / mix | n 9–12 uniform, p_self 0.5, gold n ≤ 8 plain stream | the same | yes |
| Round-2 window / mix | ⅔ n 13–16 / ⅓ n 9–12 by pool composition | the same (64,000 / 32,000 per n) | yes |
| FT recipe | 12k steps, bs 64, lr 3e-5 (0.3 × the base lr), warm-up 100 + cosine to 0, AdamW wd 0.01, init_from, fresh AdamW | the same numbers; lr 3e-5 is also 0.3 × the text base's 1e-4 | yes, absolute and relative lr |
| Self exposure | 5,992 self steps = 383,488 self items per round (seed-1 coin stream) [MEASURED §215 / §216 results] | **the same 5,992 self-step indices**: the text trainer replays the latent's rng_s consumption exactly (§1.5) | yes (exact) |
| Plain stream | D3-T's law, replayed from seed 1 each round (disclosed, D11) | §170 TEXT-COT's law (uniform n 1–8, test ∪ SERIAL excluded), replayed from seed 1 each round (disclosed) | each arm's own base law |
| Training target | R1 answer + R2 per-pass states E_self (from the chain) | the chain string e_1 ' =' … e_n EOT composed from the chain (compose_T, §1.3) | the same chain object, each arm's own target format |
| Test items | test_set(1000, n_max 24): n 13–16 after round 1, n 17–20 (+ 21–24) after round 2 | **the same items** (paired per item) | yes |
| Eval card / dtype | local RTX 3070, fp32, bs 32 (eval_vb) / 16 (eval_deep) | RTX 3090 (cloud; every text-arm GPU stage), fp32, tf32 off, cot_eval_bs 32 (§1.6, §8; open question Q3) | no (cross-architecture comparison, read through aggregate n50 only) |
| Test-time computation | K′ = n + 2 latent passes over T ≈ 3n + 4 rows | free greedy generation of C(n) ≈ 1.5 n² tokens with the whole chain in context | **the object of comparison** |

**Not matchable, stated:**
- (i) The text model sees every intermediate expression at test time; the latent loop must hold them (D1, §1.4).
- (ii) Text positions grow ~n² (§3.3). At round 2 the text arm trains rows ≤ 527 and is read on rows up to 1,085, past GPT-2's 1,024 positions at n 24 (§1.7).
- (iii) The latent arm has a memory channel that self-training trades (MEM-TRADED). The text arm has no counterpart; its counterpart cost is tokens and context.

### 1.2 Labeller: the text model's own re-rooted greedy single step (REROOT), computed by TF-GREEDY

- **REROOT chain** (the text analogue of SELFDEPTH_v1 §2.1; never the simulator):
  - cur ← input; repeat: prompt x = [EOT] + render(cur) + [' =']; the model's greedy continuation up to its first ' =' or EOT is the step s;
  - s must pass CV-T (§1.3); cur ← parse(s);
  - stop at a single number (label = its value) or at the step cap n + 4 (FAIL).
- **Why REROOT, not the model's whole chain (FULL, Lee 2502.01612's protocol)** [MEASURED research6/scripts/ts_label_check_gen.json, ts_label_check_s1.json; CPU fp32; fresh items, rng 23180]:
  - FULL keep (the whole greedy chain is form-valid) at n 9 / 10 / 11 / 12: **0.995 / 0.540 / 0.050 / 0.000** (200 per n). The FULL chain breaks near rows 170–200, past the 147 rows the base trained (§1.2a).
  - So GATE-1's keep_n ≥ 0.70 fails at n 10–12 under FULL. FULL cannot run the +4 window, and its kept items would be a ≤ 5 % selected subset at n 11 (Literature failure mode 1).
  - REROOT keep at n 9 / 10 / 11 / 12: **1.000 / 0.995 / 0.975 / 0.970**, with 1 kept-wrong of 788 (§1.2a).
  - REROOT is also the matched choice: the latent labeller is the latent model's single step from a fresh root, never its multi-pass rollout. FULL is the precedent's protocol (Lee). It is reported at every gate as the descriptive **LABELLER-FULL** row (keep, π on the gate items), so readers see what Lee's protocol would have given.
- **TF-GREEDY (teacher-forced greedy verification) is the labeller's computation.**
  - **Uniqueness [PROVED].** Given the current tree, a CV-T step is unique up to its value token. CV-T pins the handle (the leftmost-ready node; shape only), its replacement by one leaf, and the canonical rendering. Values are single GPT-2 tokens, and leaves never carry brackets (nested170.render). So the CV-T step is c_k(v) = render(tree with the handle → leaf v) for some v ∈ {0..10}.
  - **Equivalence [PROVED in exact arithmetic].** By induction over the emitted tokens: greedy decoding emits a CV-T step iff, teacher-forced on its own accepted prefix, the argmax equals c_k's token at every fixed position and is a value token at the value slot. When it holds, the greedy step is c_k(v) with v = that argmax. Two batched forward passes per step (prompt + prefix → v; v + suffix + terminator) therefore decide exactly what greedy decoding decides. Nothing is generated after a deviation; the item is not kept either way.
  - **fp32 caveat [ASSUMED negligible; measured].** The parallel and KV-cache forwards sum in different orders, so an argmax can differ at a near-tie. On CPU fp32, TF-GREEDY's kept flag and label equal plain KV-cache greedy generation (nested170.cot_answers' loop) on **90 / 90** items (n 9: 10, all kept; n 11: 40, 2 kept; n 12: 40, 0 kept) [MEASURED ts_label_check_smoke.json, ts_label_check_gen.json]. G-T16 EQUIV re-measures it on the GPU at 250 gate items per n (§7).
  - **Gold-freeness.** The candidate uses the tree shape only (N._leftmost_ready, N._render_pos, a leaf swap); values are always the model's argmax. N.apply / N.value / N.eval_tokens / N.cot_steps / N.cot_ids are patched to raise in the builder (G-T0). Gold enters only the separate gate and audit processes.
  - **Cost.** About 2 × Σ_k L_k token-positions per item, all in parallel batches: **365 per T1 item, 575 per T2 item** [DERIVED ts_cost]. Generation instead needs about 1.5 n² **sequential** decode steps per item.
    - Projected **0.6–1.0 h per T1 pool** and **0.9–1.6 h per T2 pool** (392k items incl. gates) at 17 / 10 achieved fp32 TFLOPS [DERIVED ts_cost; throughput ASSUMED].
    - Naive KV-cache generation of the same pools at bs 32: about 12,250 batches × ≈ 190 / 300 decode steps at 12–16 ms per step (20 ms at T2's longer contexts) ≈ **8–11 h (T1) and 13–20 h (T2)** on a 3070-class card [DERIVED from the §170 eval's measured per-step latency, ts_cost eval_calib; batch maxima ASSUMED].
    - On the local 3070 (≈ 10 achieved TFLOPS), TF-GREEDY needs ≈ 1.0 h (T1) / 1.6 h (T2) per pool [DERIVED ts_cost label_hours].
    - TF-GREEDY is what makes the matched exposure (383,488 self items) affordable without shrinking it.
- **Settings:** fp32, tf32 off (as the latent labeller); label batch 128, fixed; step cap n + 4; one card class for pool and gates (RTX 3090, §8).
- **Labeller identity** (pre-stated; never chosen on pool or test items):
  - round T1: `s170_cot_s1/model.pt`;
  - round T2: `s218_t1_s1/model.pt`, with the pre-stated **BASE-LABELLED** fallback per window (`s170_cot_s1`) if T1 fails a window's gate (SD2_v1 §2.1's rule). Both labellers failing a window's gate → no T2 (no cell, no spend).

### 1.2a REROOT precondition numbers (CPU, base s170_cot_s1, fresh items rng 23180)

[MEASURED research6/scripts/ts_label_check_s1.json, summarised in ts_summary.json; CPU fp32; 200 fresh items per n (rng 23180), excluded against test_set(1000, 24) + SERIAL. These are precondition data that inform the design; they are not registered readings and their items join every exclusion set.]

| n | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|
| **REROOT keep** | 1.000 | 0.995 | 0.975 | 0.970 | 0.955 | 0.875 | 0.775 | **0.630** |
| REROOT kept-wrong / kept | 0 / 200 | 0 / 199 | 1 / 195 | 0 / 194 | 0 / 191 | 0 / 175 | 0 / 155 | 0 / 126 |
| REROOT failing step | — | step 1 (1) | step 1 (4), 3 (1) | step 1 (5), 3 (1) | step 1 (9) | step 1 (23), 2–3 (2) | step 1 (41), 2–3 (4) | step 1 (66), later (8) |
| FULL keep | 0.995 | 0.540 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FULL kept-wrong / kept | 0 / 199 | 2 / 108 | 1 / 10 | — | — | — | — | — |
| FULL: median failing stage / median row of the failure window | 6 / 132 (1 item) | 8 / 170 | 8 / 186 | 7 / 196 | 6 / 199 | 6 / 198 | 5 / 174 | 4 / 146 |
| for reference: latent D3-T chain keep (SD2_v1 §2.1) | 1.000 | 0.996 | 0.988 | 0.976 | 0.972 | 0.952 | 0.943 | 0.917 |

**Readings** (descriptive; they shape the design):
- **Round 1 is labellable.** A GATE-T1 preview on these counts: keep_n ≥ 0.70 at every n 9–12 ✓; π_n 1.000 / 1.000 / 0.995 / 1.000; π_pool 787 / 788 = 0.9987 ≥ 0.995 ✓. At 200 per n the per-n π cannot reach the 0.99 one-sided bound (lower 95 % ≈ 0.976–0.985); the registered gate uses 2,000 per n.
- **REROOT labels are cleaner than FULL labels.** REROOT has 1 kept-wrong of 1,435 kept (n 9–16), against 3 of 317 for FULL (n 9–11). Long contexts produce value errors too; short re-rooted prompts avoid most of them. That is D1's prediction (§1.4) with one qualification: values are a local read, but not a perfect one at rows near the frontier.
- **The base's single step has a copy-length reach.** 148 of the 165 REROOT failures are at **step 1**, the longest expression. Keep falls from 0.955 to 0.630 as the mean input grows from 39 to 48 tokens (n 13 → 16), against a trained maximum of 31 [DERIVED from test_set lengths: max L_0 31 at n 8]. The latent single step reaches much further (D3-T 0.917 at n 16). So **the text base cannot label n 16 under GATE's keep_n ≥ 0.70**, and the BASE-LABELLED fallback for T2's n 13–16 window is expected to fail at n 16. **Round 2 therefore rests on T1's own single-step reach** (T1 trains copies up to 43 tokens). LABELLER-KEPT-T and GATE-T2 measure it, and a T2 gate failure reads ROUND-1-ONLY with **TEXT-LABEL-STALL** (§4.4).
- **FULL chains break near a fixed row, not a fixed step.** At n 10–14 the median failure window sits at rows ≈ 170–200 (≈ 1.2–1.35 × the 147 trained rows), while the failing stage falls from 8 to 6 as n grows. At n 15–16 a second mode appears: early failures on long inputs (the copy-length reach above), which pulls the median row down to 146–174 [DERIVED ts_failpos_s1.json]. **The base's failure is keyed to training-distribution frontiers in rows and copy length, not to the step count**, which favours the POS-* panels over the N-* panels as a mechanism (§3.5). The comparison itself reads additive vs proportional, which both families contain.
- **The base's REROOT reach beyond n 16** (100 per n; ts_label_check_s1_deep.json): keep **0.62 / 0.41 / 0.18 / 0.14 at n 17–20** (kept-wrong 2 / 62 at n 17, 0 after), against D3-T's chain keep 0.892 / 0.839 / 0.809 / 0.781 (SD2_v1 §2.1) [MEASURED]. Almost every failure is at step 1. **At equal training (gold n ≤ 8), the latent single step reaches about 6 operations further than the text single step at the 0.78 keep level** (D3-T at n 20 vs text at n 15). This is a labelling-reach difference between the arms that exists before any self-training, and the proposed LABELLER-REACH row (Q13) would report it.

### 1.3 Filter and target (the text analogues of KEEP ∧ COMPOSE-VALID)

Per step k, with tree_{k−1} the parsed previous expression (tree_0 = the parsed input) and s_k the model's emitted tokens up to the terminator:
- **PARSE-T:** s_k parses as an expression over the §170 vocabulary.
- **FRAME-T:** |s_k| = |c_k|, and every token outside the handle's span equals e_{k−1}'s (an exact copy).
- **SPAN-T:** exactly one contiguous span of e_{k−1} is replaced, by exactly one token.
- **VALUE-SLOT:** that token is a value token (' 0' … ' 10'). **Its value is never checked.**
- **ORDER-T:** the span is the leftmost-ready handle of tree_{k−1} (operands, operator and its own brackets; shape only).
- **CANON-T:** render(parse(s_k)) == s_k.
- **TERM-T:** the terminator is ' =' iff s_k has an operator, and EOT iff s_k is a single number.
- **KEEP-T** = PARSE-T ∧ TERM-T at every step ∧ exactly n steps (n = # operator tokens of the input) ∧ the last step is a single number.
- **CV-T** = KEEP-T ∧ FRAME-T ∧ SPAN-T ∧ VALUE-SLOT ∧ ORDER-T ∧ CANON-T at every step.
- **Kept = KEEP-T ∧ CV-T** ⟺ s_k = c_k(v_k) at every k [PROVED, §1.2 uniqueness]. The decomposed flags are logged per failure, for reporting which check broke first.
- **Record** (pool): keys exactly **{syms, n, chain_label, values}** (values = the model's v_1..v_n; chain_label = v_n). No gold field (root, t, y, gold, cot) ever.
- **Target:** compose_T(syms, values) = ids(c_1(v_1)) + [' ='] + … + ids(c_n(v_n)) + [EOT], from shape and the model's values only.
  - Under CV-T it equals the model's own greedy REROOT chain token for token. It equals N.cot_ids(gold root) iff every v_k is the gold value [PROVED: uniqueness + induction; the analogue of PROOFS_SELFDEPTH §7].
  - Loss = CE on every target token, prompt masked (the base's loss_items textcot, unchanged).
- **π^E ≡ π for text [PROVED]:** a kept chain's every intermediate expression is c_k(v_k), so "every state correct" ⟺ "every value correct". GATE's π^E term is kept for form and reads π.

### 1.4 The D1 value-token shortcut, stated precisely, and how this design handles it

- **D1 for text** [PROVED, an information argument; LIT_STEPFACTOR_D1, parent Bachmann & Nagarajan 2403.06963]. When the text model emits v_k, the handle's operator and operands are in its context: in e_{k−1}, which is the prompt under REROOT and the previous step under FULL. So v_k = apply(op, a, b) is a function of visible tokens. Teacher-forced CE on v_k can be driven to zero by a local read, and no hidden state has to carry v_k.
- **It is not a confound of the comparison.** At test time the text model also sees the operands. D1 is the text arm's mechanism, the thing being compared with the latent loop, which must hold intermediate values across passes (VB-C1). It is a difference in what the two arms must compute; it is stated, not removed.
- **Where it would become a gold leak, and the rule that prevents it.**
  - The visible operands make a *value check* trivially available to the text arm's filter: recompute apply(op, a, b) from the step's own tokens with the simulator. That is a gold step-value check. It would set π = 1 by construction and filter on correctness.
  - The latent filter never checks values (CV checks frame / span / order / canon only; GATE / LABEL-AUDIT are eval-only).
  - **So the text filter checks form only (VALUE-SLOT accepts any value token), the final answer is never compared with gold, and the target values are the model's own argmax.** This is Literature's recommendation made explicit (LIT_FAITHFULNESS_SELFIMPROVE: the same final-answer / exact-step-count / form filter in both arms). G-T0's patch makes N.apply raise in the builder, so no value check can run there.
- **The remaining asymmetry is measured, not assumed.**
  - (i) A wrong v_k in a kept text chain is one wrong arithmetic target. It is then copied faithfully into e_{k+1..n} and the answer. Copying a wrong value is a *correct copy* under the format, so it trains copying, not arithmetic. A latent kept-wrong chain likewise carries one wrong step value into its later R2 states. Both arms' kept-wrong rates are measured at the gates (π) and in LABEL-AUDIT.
  - (ii) **Value tokens are a small share of the text target:** n of C(n) tokens (n 12: 12 / 218 = 5.5 %; n 16: 16 / 386 = 4.1 %) [DERIVED ts_tokens]. The text arm's CE is dominated by copy tokens whose targets are fixed by shape (gold-free). The trainer logs the self items' value-token CE separately (`self_value_ce`), so the arithmetic and copy parts of the text loss are visible.
- **Prediction from D1** [CONJECTURE]: text labels are harder to get wrong than latent labels (a local read vs a held value), so π_text ≥ π_latent. The text arm's failures beyond support should be **form** failures (copy / position), not value failures. The fail-step histograms in ts_label_check are the first evidence (§1.2a).

### 1.5 Cells and training (s1 lineage; RTX 3090 only)

| element | spec |
|---|---|
| Cells | **`s218_t1_s1`**: FT from `boxes/runs_fsq/s170_cot_s1/model.pt`. **`s218_t2_s1`**: FT from `s218_t1_s1/model.pt`. Base reference **`s218_base_s1`**: an eval-only directory holding the fp32 re-eval of `s170_cot_s1/model.pt` (model_sha256 recorded); `s170_cot_s1/` itself is never written |
| Trainer | `boxes/nested170.py --arm textcot` plus the edits of §1.9 (nested170.py has no `--init_from` / self pool today [VERIFIED grep]) |
| Init | `--init_from <model.pt>`: strict load (missing = unexpected = [] asserted), fresh AdamW, no calibration (the textcot arm has none) |
| Recipe | **s170_cot_s1/args.json** with only `--init_from`, `--steps 12000`, `--lr 3e-5`, `--self_pool`, `--p_self 0.5`, `--self_micro <G-T9>`, `--out` (+ `args_ref` and the smoke-only keys) changed. Unchanged: arm textcot, bs 64, n_max_train 8, seed 1, warm-up 100 then cosine to 0 (fsq120.lr_at), AdamW wd 0.01, clip_grad_norm 1.0, bf16 autocast, ckpt_every 2000, auto_micro (plain micro 1), mem_budget 21.0 |
| Plain stream | rng = default_rng(seed 1) and train_items(rng, 64, 8, EXCL) as the base: **each round replays the base's first 12,000 plain batches** (disclosed, as SD2's D11; a(n) = 1.000 at n 1–8 [MEASURED eval.json], so no reading depends on new plain items). G-T3 checks plain_hash at steps 1..200 |
| Coin stream (exposure match) | rng_s = default_rng(seed + 21800); per step `coin = rng_s.random() < 0.5`; on a self step **64 × rng_s.integers(0, 5) are drawn and discarded** (the latent's self-U draws, vbank.py l.755 / l.764). The text coin sequence then equals the latent rounds' exactly: **5,992 self steps = 383,488 self items, at the same step indices** [PROVED by stream identity; G-T3 asserts coin equality with s215_ftself_s1/draws_log.json at all 12,000 steps]. On a self step the plain batch is drawn first and discarded (so plain batches stay aligned with the base) |
| Self batch | the next 64 pool records at the cursor; target compose_T(syms, values) (§1.3); loss = the base's textcot CE (prompt masked); no root key exists, so any gold path is a KeyError |
| Pool sizing (reuse-free, exact) | consumption is deterministic: **383,488**. T1 pool: **96,000 kept per n 9–12 = 384,000** (spare 512 records = 8 self steps). T2 pool: **64,000 per n 13–16 + 32,000 per n 9–12 = 384,000**. One fixed interleaved order (permutation seed = pool seed + 1). Expected consumed per n (hypergeometric, 383,488 of 384,000): T1 95,872 (SD 9.8); T2 63,914.7 per new n (SD 8.4) and 31,957.3 per old n (SD 6.2) [DERIVED]; the realised counts are fixed by the pool order and precomputed before launch. Fresh-once; exhaustion or reuse = void |
| Micro | plain 1 (the base's); self from G-T9 over {1, 2, 4, 8}: projected **T1 2 (13.6 GiB), T2 4 (11.8 GiB)**; micro 1 projects 25.4 / 41.7 GiB, T2 micro 2 21.8 GiB > 21.0 [DERIVED ts_cost, linear-in-T model calibrated on the base's one mem_check point (12.99 GiB at T 147); ASSUMED form] |
| Logging (log.jsonl, every 100) | step, loss, plain_loss, self_loss, **self_value_ce**, self_copy_ce, self_frac, pool_cursor, n_self_steps, self_reuse (must be 0), lr, grad-norm summary |
| Draw log | draws_log.json, steps 1..12,000: coin, self n list, plain_hash (sha1 of the plain batch's syms) |
| Checkpoints | full state every 2k (model, AdamW, np rng, torch rng, rng_s, cursor, counters, step, elapsed), **mirrored off-host** and sha-checked (user rule 2026-10-15); resume tested before launch (G-T4); deleted at landing; model.pt kept (T1 is T2's labeller) |

### 1.6 Eval contract (per text model: `s218_base_s1`, `s218_t1_s1`, `s218_t2_s1`)

**eval_text.json** (new eval part; nested170's generation loop with the changes listed):
- **Items:** test_set(1000, n_max 24), n 1–24 (24,000 items, test seed 17100), test order. **items_sha256** as SD2_v1 §4.2 (json.dumps of [[syms, int(t)]] over the 24,000 items, separators (",", ":"), ensure_ascii); the scorer recomputes and asserts it.
- **Generation:** greedy, KV cache, nested170.cot_answers' loop (batched by equal prompt length), **fp32, tf32 off, cot_eval_bs 32** (§170's base eval value).
  - **Budget, gold-free:** per batch min(max C_struct + 50, 1025 − P), with C_struct the item's shape-only chain length (ts_tokens.struct_len; equal to len(cot_ids) on all 24,000 test items [DERIVED, asserted]). The current loop reads the gold chain for its budget; the values coincide, but the new part never calls cot_ids.
  - **CTX-OVERFLOW:** an item with P + C_struct > 1025 is **not generated** and counts as a failure (§1.7).
  - **Non-termination:** no EOT within the budget, or a tail that is not one number, gives answer −1 = a failure (as §170).
- **Per-item store** (store-items rule): idx, n, y, P, C_struct, overflow, answer, ntok (generated incl. EOT), eot, **form_valid** (KEEP-T ∧ CV-T of the generated chain), **strict** (form_valid ∧ answer == y), first_bad_step, **first_bad_pos** (absolute position of the first token that deviates from the form-valid candidate given the model's own earlier values).
- **Aggregates:** acc_by_n (answer == y over all 1,000 items, overflow = failure: **this is a_T(n)**), acc_feasible_by_n, form_valid_by_n, strict_by_n, ntok_mean_by_n, first_bad_pos quantiles by n, chance_by_n (majority-class share).
- **Meta:** run, model_sha256, card, dtype fp32, tf32 false, cot_eval_bs 32, test_seed, test_n_max 24, items_sha256, budget rule, commit, sec, peak_gib.
- **Projected time:** 1.7–2.4 h per model on a 3070-class card, ≈ 1.2–1.7 h on a 3090 [DERIVED ts_cost: 293k–(293k + 50 × batches) sequential steps; per-step latency calibrated on the §170 eval's measured 1,411 s for n 1–16 (3070, bf16, bs 32), fp32 KV traffic doubled; ASSUMED form].

**eval_textself.json** (per FT cell; gold only here, separate process):
- **LABEL-AUDIT-T:** 5,000 random consumed records (T1 rng 23270, T2 23370): π (all values equal gold), pooled and per window, Wilson 95 % CIs overlapping the gate's; kept-wrong records with (syms, step, the wrong value, the gold value).
- **MEMO-GAP-T:** |a on the first 2,000 consumed pool items of n − a on the 2,000 gate items of n| ≤ 0.02 per window n (free generation, the eval settings).
- **CONTAMINATION:** pool ∩ the §1.8 exclusion set = ∅ (re-checked on the pool file), self_reuse = 0.
- **LABELLER-KEPT-T** (descriptive; a round-3 preview): REROOT keep / π of the cell's model on 500 fresh items per n 9–24 (T1 rng 23280, T2 23380).

**Base re-eval first (G-T19):** `s218_base_s1/eval_text.json` exists before any T1 output is read. It fixes every text bar (§4.3) and the round-0 comparison row. BF16-VS-FP32 (descriptive): the committed bf16 `s170_cot_s1/eval.json` acc_by_n against the fp32 re-eval at n 1–16.

### 1.7 Context overflow (pre-stated)

- **GPT-2 has 1,024 position rows** (core.GPT2KV adds `wpe[p0:p0 + T]`; a generation past row 1,023 cannot run). Generating a chain of C tokens after a prompt of P tokens feeds positions 0..P + C − 2, so **an item is context-feasible iff P + C ≤ 1025**.
- **C is a function of the input's shape only** (§1.3), so feasibility is a gold-free item property, fixed before any model runs.
- **Measured on the test items** (generator statistics, no model output; disclosed) [DERIVED ts_tokens.json]:

| n | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| mean P + C | 435.5 | 490.7 | 545.8 | 605.7 | 667.7 | 730.4 | 797.6 | 871.1 | 942.0 |
| max P + C | 519 | 559 | 639 | 697 | 767 | 843 | 933 | 995 | **1,085** |
| feasible share | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.971** (29 of 1,000 overflow) |

  The draft brief's estimates (max 539 / 793 / 1,105 at n 16 / 20 / 24) were close; the measured maxima are 519 / 767 / 1,085. **Only n 24 overflows (29 items); n 21–23 all fit.**
- **Rule:** an overflow item is not generated and **counts as a failure** in a_T(n): the intrinsic token cost of the text format. **CTX-FEASIBLE accuracy** (over the 971 feasible n-24 items) is reported beside it, and every reading touching n 24 (M_far, n50 / n90 if they reach 24) states the overflow share. The latent arm has no overflow (rows ≤ 83 at n 24).
- **Round 3 is context-infeasible for the text arm** [DERIVED ts_panels, on fresh draws rng 23192, never the sealed n 25–30 test items]: feasible share n 24 0.959, **n 25 0.551, n 26 0.086, n 27 0.003, n ≥ 28 0.000**. A text round 3 (support 20, extrapolation read at n 21–28) would lose most of its test items to the context. So TEXT-SELF is registered for **two rounds only**, and the comparison ends at round 2 (**CTX-BOUND**, stated beside every comparison reading).

### 1.8 Contamination rules (draft registration text)

1. **Training items are fresh generated items only.** Pools: `nested170.generate(rng, n)`; T1 rng default_rng(23200), T2 23300; one interleaved order each (23201 / 23301). Fresh-once; exhaustion or reuse = void.
2. **Exclusion by exact rendered syms** against: **test_set(1000, n_max 30)** (30,000 items; 216-5's rule for any later pool or gate; superset of n_max 24); SERIAL (rng 17300, 2,200); the 213-37 fresh items (rng 21337, n 9–24); **every latent gate** (s215_gate1, s216_gate2_s1, s216_gate1_s2, and s216_gate2_s2 if it exists at build time); **every latent pool** present at build time (s215_pool/pool_r1.jsonl, s216_pool_s1/pool_r2.jsonl, s216_pool_r1_s2, s216_pool_r2_s2); **the TEXT-SELF precondition items of this draft** (rng 23180, n 9–20, 200 per n: read by the base text model on CPU); every TEXT-SELF gate and pool already built (any lineage). A latent pool already deleted under its retention rule is covered by the collision bound P(render = s) ≤ (1/11) · 40⁻ⁿ per draw [PROVED, PROOFS_SELFDEPTH §8]; this is stated in pool_meta. Per n: draws, collisions and duplicates logged.
3. **No gold reaches training.** Builder gold patches (N.apply, N.value, N.eval_tokens, N.cot_steps, N.cot_ids raise) installed **before** generation; record keys exactly {syms, n, chain_label, values}, asserted by the loader; no root key in any self item; a self-batch smoke counts cot_ids / cot_steps calls on self items = 0 (G-T0). The trainer never opens a gate directory (grep: no `s218_gate` in nested170.py).
4. **Gold is used only for evaluation, the gates, LABEL-AUDIT-T and MEMO-GAP-T**, each in a separate process.
5. **Test items:** the text arm reads the same test_set(1000, n_max 24) items as the latent arm. Text models have been evaluated on n 1–16 before (s170 eval.json, bf16; read before this draft; they inform predictions only). **n 17–24 have never been read on a text model.**
6. **The sealed n 25–30 (216-5) are never evaluated and never used**, except as the exclusion list of item 2.
7. **The labeller is fixed in advance** (§1.2). Pools, gates and evals use one card class (RTX 3090) and fixed settings (fp32, tf32 off, label batch 128; cot_eval_bs 32).
8. **Post hoc:** pool ∩ (item 2) = ∅ re-verified on the pool file. Any hit, any gold key, any reuse (`self_reuse` > 0) or exhaustion → **CONTAMINATED / void**.
9. **Seeds** (all unused in docs/THEORY.md, docs/research_*/*.md, boxes/*.py, theory/*.py, scratch/*.{py,sh} [VERIFIED grep at 598e75de, 0 hits]): s1: precondition 23180; shape-only draws 23190–23193 (never shown to a model); T1 pool 23200 / 23201, GATE-T1 23260, LABEL-AUDIT-T1 23270, LABELLER-KEPT 23280; T2 pool 23300 / 23301, GATE-T2 23360, GATE-T1′ 23361, LABEL-AUDIT-T2 23370, LABELLER-KEPT 23380. s2 lineage (if run): the same pattern at 23400–23580 (to be grep-checked at registration).

### 1.9 Code edits (list only; none written; Audit's diff review G-T13)

| # | where | edit |
|---|---|---|
| T1 | boxes/nested170.py, textcot arm | `--init_from` (strict load, fresh AdamW); `--self_pool` + loader (sha256 vs pool_meta, key-set assert, n from syms); `--p_self` with the rng_s coin replay incl. the 64 discarded integers(0, 5) draws per self step; compose_T targets for self items (`SD-LABEL-MAP-T`: `it["values"] = rec["values"]`); per-batch-type micro (`--self_micro`); log fields (self_value_ce / self_copy_ce); draws_log; rng_s + cursor in the checkpoint and the resume branch; `--mem_check_self` (the 64 longest pool records of the deepest n, two optimiser steps, peak after step 2); `--stop_at`. **Default behaviour without the new flags bitwise unchanged** (G-T7, all four §170 arms construct + 1 step) |
| T2 | boxes/textself.py (new) | candidate (shape only), TF-GREEDY REROOT and FULL labellers (batched, right-padded, per-row positions and key masks), CV-T decomposition, compose_T, record format, load_pool. CPU reference: research6/scripts/ts_label_check.py (fwd / label_reroot / label_full) |
| B1 | scratch/s218_build_pool.py (new; from s216_build_pool.py) | gates stage (GATE items, exclusion, gold patch) and pool stage (per-n targets, interleave, pool_meta with seeds, draws, collisions, labeller sha256, card, dtype, label_bs, T_max_by_n) |
| B2 | scratch/s218_gate.py (new) | gold scoring of GATE-T1 / GATE-T2 / GATE-T1′ (π, keep, CIs, per-item records), LABELLER-FULL, KEPT-VS-DRAWN, the BASE-LABELLED fallback, G-T16 EQUIV |
| E1 | eval part `text` (in nested170.py `--eval_text` or a new boxes/text_eval.py) | §1.6 contract (gold-free budget, CTX-OVERFLOW, per-item store, items_sha256) |
| E2 | eval part `textself` | §1.6 (LABEL-AUDIT-T, MEMO-GAP-T, contamination, LABELLER-KEPT-T) |
| S1 | theory/score_s218.py (new) | §4.6 |
| Q1 | scratch/s218_*.sh | one-host-per-lineage stage chain (§8), drain waiters, off-host mirrors at every stage boundary |

**nested170.py is shared with the §170 family** (DIRECT / INJECT / READSLOTS): no edit while any job importing it is running or deploying (deploys ship the working tree; lazy imports at eval).

---

## 2. Seeds and lineages

| lineage | text | latent partner | status of the partner |
|---|---|---|---|
| **s1 (primary)** | `s170_cot_s1` → `s218_t1_s1` → `s218_t2_s1` | D3-T s1 → FT-SELF → R2-s1 | complete: n50 10.31 / 15.46 / 20.54 [MEASURED §216 s1 result] |
| s2 (recommended, secondary) | `s170_cot_s2` → `s218_t1_s2` → `s218_t2_s2` | D3-T s2 → R1-s2 → R2-s2 | in progress (base COMPETENT, 216-3; GATE-1-s2 PASS, 216-4; ×2 pending) |

- **Cost / benefit of s2** (cost from §8; OC from §6):
  - Cost: one more lineage at the same price as s1, about **$1.7–5.0** (Plan A), on its own 3090 host in parallel. No new code.
  - Benefit 1: the comparison's error is **lineage noise**, not sampling. Sampling SE of n50 at 1,000 items per n is about 0.03–0.05 n [DERIVED from the per-n slope near 0.5 of ≈ 0.3–0.5 per n]. Lineage spread at round 0 is |Δn50| 0.20 between the latent bases (10.31 / 10.51) and 0.16 between the text bases (10.44 / 10.60) [DERIVED ts_panels; s2 latent from 216-3's a(n)], and it can grow over rounds (Kuo 2606.29983: across-run SD 0.09–0.12 in accuracy at the frontier). With one lineage per arm, σ_Δ ≈ 0.2–0.5 n per round [ASSUMED]. Averaging two lineages per arm divides it by √2 (§6 POOLED rows).
  - Benefit 2: a text replicate tells whether the text arm's round-1 margin (the H_pos / H_n split, §3.5) is a lineage property.
- **Recommendation:** run s2 **in parallel and pre-stated**, whatever s1 reads, so the second lineage is not outcome-selected (SD2's m6 rule).
  - **The primary comparison is the s1 pair.** A POOLED row (mean text n50 − mean latent n50 over the lineages that reached each round) is secondary.
  - If the latent s2 lineage stops before R2 (any ×2 FAIL stage), the text s2 lineage's round-2 rows read **UNPAIRED** (descriptive: text lineage noise only).
- **Seeds inside a lineage:** the FT cells use the base's seed (1 or 2) for the plain stream and rng_s, as the latent R1 / R2 cells did. So **the s2 coin stream equals R1-s2's / R2-s2's** (default_rng(2 + 21800) with the same consumption). G-T3 checks it against `s216_r1_s2/draws_log.json` once that file exists. If it does not exist at s2 text launch, it is checked against a stream replay computed in a separate process and logged.

---

## 3. Mathematics

### 3.1 S0-T, the support identity for text [PROVED]

- **Steps.** Every kept chain of round r has exactly n ≤ B_r steps (B_1 12, B_2 16): KEEP-T. Every target string is the composition of n CV-T steps.
- **Rows.** Every target token sits at a position ≤ T_train,r = max (P + C) over the consumed records. That is ≈ **311** at B 12 and ≈ **527** at B 16 [DERIVED ts_tokens: the max over 12,800 fresh draws per mixture; the registered value is pool_meta's T_max_by_n]. The base trained rows ≤ 147.
- **(a) Zero loss past support.** A model that fails on any item with n > B_r, or on any token at position > T_train,r, pays no loss on the round-r mixture: the loss touches only the support. **Stopping at the support is irreducible by loss in text too** (the text H_shift). Unlike the latent arm, where rows grow ~3n, the text arm's step frontier and position frontier move together (positions ∝ n², §3.3), so its training cannot separate them.
- **(b) In-support vs past support.** T-ABSORB (n 9–12; n 13–16 in round 2) is in support. n 13–16 after T1 and n 17–24 after T2 are past it in both steps and rows: at n 17–20, read positions run to 767, against 527 trained rows.
- **(c) Pretrained rows.** GPT-2's rows > T_train,r carry pretraining only for this task (learned absolute position embeddings, core.GPT2KV: `wpe[p0:p0 + T]`). The latent arm's nuisance H_T (rows 61 trained vs 83 read) is the small version of what is **intrinsic** here.

### 3.2 S1-T, exposure under the replayed coin stream [PROVED by stream identity; DERIVED counts]

- rng_s = default_rng(seed + 21800); per step one `random()`; on a coin of 1 (< 0.5), 64 `integers(0, 5)` draws. This is the latent trainer's exact consumption [VERIFIED vbank.py l.755, l.764]. So the text coin sequence equals the latent's at every step, and **both arms take their 5,992 self steps at the same 5,992 step indices**, in round 1 and in round 2 (the seed-1 replay of D11).
- Self items per round 383,488; per n as in §1.5. Plain items: the base's first 12,000 batches, each round.
- **Deadline / position costs** (the text analogue of S1's deadline table): the share of an item's target tokens at positions > t* is a pure function of the shape law. It is not computed here because no reading uses it. A "stop past the trained rows" policy costs nothing (S0-T a).

### 3.3 Positions and context grow as n² [constant MEASURED ts_tokens; form DERIVED under uniform bracket drops]

- **Form.** e_0 has L_0 = 2n + 1 + 2b tokens (b bracket pairs). Step k removes one operator and one operand (−2) and, at some steps, one bracket pair (−2): L_k = L_0 − 2k − 2b_k, with b_k the brackets dropped by step k (b_n = b). Then C = Σ_{k=1..n} (L_k + 1) = n(L_0 + 1) − n(n + 1) − 2 Σ_k b_k. With b_k ≈ b k / n and b = βn, this gives C ≈ (1 + β) n² + (1 − β) n.
- **Measured** (test items): C̄ / n² = **1.49 (n 8), 1.51 (n 16), 1.51 (n 24)**, i.e. β ≈ 0.47 at n 16; P̄ = 2n + 3 + 2b ≈ 2.94 n + 3 (50.0 at n 16, as measured).
- **Test-time cost per item** [DERIVED]:

| | text (free greedy generation) | latent (K′ = n + 2 passes) |
|---|---|---|
| sequential steps | C(n) ≈ 1.5 n²: 386 at n 16, 606 at n 20, 868 at n 24 | n + 2: 18 / 22 / 26 |
| token-positions processed | P + C ≈ 1.5 n² + 3n: 436 / 668 / 942 | (n + 2) × T ≈ 3n²: ≈ 18 × 50 = 900 / 1,364 / 1,924 |
| memory at test | KV cache over P + C rows | bank of (n + 2) × T states |

  So the text format uses **about half the token-positions (FLOPs)** but **21–33× more sequential steps** at n 16–24. TOKEN-COST (§4.4) reports both per correct answer.
- **Context bound** (§1.7): feasible share 0.971 at n 24 and **0.551 / 0.086 / 0.003 / 0 at n 25 / 26 / 27 / ≥ 28** (fresh draws). Two rounds are the most the text arm can run on GPT-2 (CTX-BOUND).

### 3.4 Why the text arm can extrapolate differently: error accumulation

- **(a) Per-token iid errors cannot explain the base curve** [DERIVED from MEASURED eval.json].
  - If every emitted token failed independently at rate ε, then a(n) ≈ (1 − ε)^{C(n)}.
  - a(8) = 1.000 on 1,000 items gives a 95 % upper bound ε ≤ 3.0 / (1,000 × 95.3) ≈ 3.1e-5 per token, which predicts a(11) ≥ (1 − 3.1e-5)^{181} ≈ 0.994. The measured a(11) is 0.234.
  - So the base's failure is **keyed** to something that changes between n 8 and n 11: either the **step count** (> 8 never trained) or the **position** (> 147 never trained). These are the two families of §3.5.
- **(b) Error accumulation in the self-labels.**
  - Under REROOT each step is a short, fresh context, so the labeller's step error is the single-step error at that expression length. It does not grow with the chain's position, which is why REROOT keeps at n where FULL breaks (§1.2a).
  - The FT target then asks the model to produce the whole chain at rows ≤ T_train,r. Absorption is learning to copy at longer rows, and the 383,488 items per round are its dose.
- **(c) Per-step rewrite.** Each text step re-emits the whole remaining expression: L_k tokens, of which L_k − 1 are copies. A copy error anywhere breaks FRAME-T (the chain is no longer form-valid). Its answer may still be right by luck: the free-generation a(n) counts the answer only, with a chance floor of about 0.04–0.10 [MEASURED base n 13–16 0.100 / 0.088 / 0.082 / 0.040].
- **(d) Value errors are rare by D1.** In-range a(n) = 1.000 at n 1–8 on 8,000 items, with ≈ 36k value tokens emitted. Past support, form-valid chains still carry value errors: FULL 3 of 317 kept (n 9–11), REROOT 1 of 1,435 kept (n 9–16) [MEASURED ts_label_check_s1.json]. So value errors exist past support, but most failures are form failures (§1.2a): copies broken at a row frontier (FULL) or at a copy-length frontier (REROOT's step 1).
- **(e) Two frontiers, both moved by self-training.** The base's failures sit at the edges of what it trained: rows (≤ 147) and copy spans (≤ 31 tokens) (§1.2a). T1's targets extend both (rows ≤ 311, copies ≤ 43), and T2's extend them again (rows ≤ 527, copies ≤ 57). The row frontier binds first at test time: past n ≈ 15 under POS-PROP's 1.31 × 311 ≈ 407 rows, against a copy frontier ≈ 1.3 × 43 ≈ 56 tokens ≈ n 18 [DERIVED; scaling ASSUMED]. The copy frontier decides T1's **labelling** reach at n 13–16 (every step of a re-rooted chain is a fresh copy of up to 57 tokens).

### 3.5 Hypotheses and pre-stated panels for the text lineage [CONJECTURE for each H; DERIVED numbers ts_panels.json from the bf16 base curve; the fp32 re-eval replaces them before any text cell is read]

Base s1: a_0(9..12) = 0.995 / 0.707 / 0.234 / 0.123; floor c = mean a_0(13..16) = 0.077; n50 10.44, n90 9.33.

| panel | rule (round r, support B_r, trained rows T_r) | r1: M (n 13–16) | r1 n50 | Δn50_1 vs 15.46 | r2: M2 (n 17–20) | r2 n50 | Δn50_2 vs 20.54 |
|---|---|---|---|---|---|---|---|
| **N-SHIFT** (step-keyed, additive) | a_r(n) = a_0(n − 4r) | 0.515 | 14.44 | **−1.02** | 0.515 | 18.44 | **−2.10** |
| **N-PROP** (step-keyed, proportional) | a_r(n) = a_0(8 + (n − B_r) · 8 / B_r) | 0.749 | 15.66 | +0.20 | 0.888 | 20.88 | +0.34 |
| **POS-PROP** (row-keyed, proportional) | P(correct) = c + (1 − c) σ((1.31 − tot / T_r) / 0.075), fitted on n 9–12 | 0.742 | 15.56 | +0.10 | 0.873 | 20.51 | −0.03 |
| **POS-ADD** (row-keyed, additive) | P(correct) = c + (1 − c) σ((45.5 − (tot − T_r)) / 11), fitted on n 9–12 | 0.513 | 14.48 | **−0.98** | 0.531 | 18.54 | **−2.00** |
| BETTER (pre-stated offset) | text margin grows faster than the latent's | — | — | +1.5 | — | — | +3.0 |
| WORSE (pre-stated offset) | margin shrinks below N-SHIFT | — | — | −2.0 | — | — | −4.1 |

- s2 base (sensitivity): N-SHIFT −0.86 / −1.94, N-PROP +0.44 / +0.65, POS-PROP +0.36 / +0.34, POS-ADD −0.86 / −1.92.
- Both position fits reproduce the base at n 9–12 with the same error (SSE 0.0020), so **the base alone cannot split them**. The rounds can: additive panels fall about 1 n behind per round, proportional ones track the latent.
- **What the comparison discriminates:**
  - **Additive** text extrapolation (in steps or rows) predicts **LATENT-AHEAD** at both rounds (Δn50 ≈ −1, then ≈ −2).
  - **Proportional** extrapolation predicts **PARITY** (|Δn50| ≤ 0.34 in both panels).
  - TEXT-AHEAD needs a text margin growing faster than the latent's ×1.29 law.
- **The H_pos vs H_n split inside a family** is descriptive. FAIL-POS (§4.5) asks whether failing items' first deviation sits at a fixed absolute row (≈ ρ T_r) or at a fixed step index (≈ B_r + c).
- **Literature priors:**
  - Lee 2502.01612: reverse addition's margin stays ≈ 3–5 digits over ≈ 90 rounds. That is additive, and it is the closest text precedent, though with NoPE and from scratch.
  - Huang 2511.07378: text CoT generalises a constant **factor** of the training length (Thm 5.1, ≥ 2× at accuracy → 1). That is proportional, under its one-layer NoPE assumptions.
  - The latent lineage measured ×1.29 at 50 % across three supports (an in-house empirical constant).
  - GPT-2's learned absolute positions are the classic length-generalisation failure, which favours row-keyed failure.
  - Priors [ASSUMED]: N-SHIFT 0.25, POS-ADD 0.20, N-PROP 0.15, POS-PROP 0.20, BETTER 0.10, WORSE 0.10 (additive 0.45, proportional 0.35).

### 3.6 S2-T, label noise

- **(a) Identity** [PROVED, as SELFDEPTH_v1 S2(a)]: π = P / (P + φ(1 − P)), with P the labeller's strict-correct rate and φ the share of its wrong chains that pass CV-T. For text, φ counts **value-only** errors, because every copy or ordering error fails CV-T (§1.3).
- **(b) Direct counts** are the guard, as in SD2 §3.3: GATE-T* at 2,000 per n, LABEL-AUDIT-T at 5,000. They are not a κ bound.
  - The latent κ argument (A1′–A5) carries over with "state" replaced by "value". Its premises are unverified for text [ASSUMED], so no bound is quoted.
  - Precondition data: §1.2a.
- **(c) Class mass** [PROVED, PROOFS_SELFDEPTH §3c]: a majority-wrong pattern class has mass ≤ 2(1 − π). LABEL-AUDIT-T prints each kept-wrong record's handle (op, a, b → wrong v) so that a systematic arithmetic error (one (op, a, b) cell wrong) is visible. This is the text form of Lee's App B.4 biased-error risk.
- **(d) Compounding across rounds.** T2's labeller is T1, which was itself trained on base-labelled chains with value errors at rate 1 − π_1. A wrong (op, a, b) → v fact learned in T1 would recur in T2's labels. LABEL-AUDIT-T2 compares its kept-wrong handles with T1's (the A5 check).

### 3.7 Training memory and time [DERIVED ts_cost; forms ASSUMED, each validated by a gate]

- **Memory.** GiB ≈ 1.9 (weights + grads + AdamW) + 0.0754 · T / micro at bs 64, calibrated on the base's single mem_check point (12.99 GiB at T 147, micro 1). The logits (bs × T × 50,257) dominate the per-token term.
  - T1 deep shape (T 311): micro 1 25.4 → **micro 2 13.6**.
  - T2 deep shape (T 527): micro 2 21.8 > 21.0 → **micro 4 11.8**.
  - G-T9 decides on each host.
- **Time.** Per-step t(T) = 0.27 s · (T / 134) · (1 + T / 4,608) / (1 + 134 / 4,608), calibrated on the base's steady 0.26–0.27 s/step (3090, plain batches with a mean batch-max T of 134) [MEASURED log.jsonl steps 12k–48k; the first 12k ran at 0.48–0.74 s/step, a host-speed effect].
  - T1 self step **0.59 s**, T2 self step **1.03 s**, plain 0.27 s.
  - **T1 1.43 h, T2 2.16 h** of training at the steady rate. Hosts differ by up to ≈ 2× (memory note: host speed variance), so 1.4–2.9 h / 2.2–4.3 h.
- **Eval memory.** At fp32 the KV cache is 12 layers × 2 × 768 × 4 B = 73.7 KB per token per item. At cot_eval_bs 32 and 1,024 rows that is 2.4 GB, plus the torch.cat copy and 0.5 GB of weights: < 8 GB [DERIVED]. G-T18 measures the longest n-24 batch.

---

## 4. Readings, definitions and the scorer's field contract

Notation: a_T(n) = eval_text `acc_by_n[n]` (free greedy generation, answer == y over all 1,000 items; overflow and non-termination are failures). Chance ≈ 0.04–0.11 (the base's n 13–16 floor 0.077; majority-class share 0.10–0.11). Every per-cell reading is comparator-free. The latent numbers in §4.4 are the committed s1 readings and are never recomputed here.

### 4.1 Files

**Per FT cell** (boxes/runs_fsq/s218_t1_s1/, s218_t2_s1/):

| file | producer | fields read | readings |
|---|---|---|---|
| log.jsonl | trainer | step, loss, plain_loss, self_loss, self_value_ce, self_reuse, pool_cursor, n_self_steps | dead (last step < 12,000), reuse = 0 |
| args.json | trainer | vs s170_cot_s1/args.json (META whitelist + §4.6 allowed sets) | ARGS-DIFF |
| draws_log.json | trainer | coin, self_n, plain_hash | G-T3 (coins == the latent round's; plain_hash == the base's first 200) |
| **eval_text.json** | eval part `text` | meta, items (per item), acc_by_n, acc_feasible_by_n, form_valid_by_n, strict_by_n, ntok_mean_by_n | §4.2, §4.3, §4.4 |
| eval_textself.json | eval part `textself` | label_audit, memo, contamination, labeller_kept | LABEL-AUDIT-T, MEMO-GAP-T, CONTAMINATED, LABELLER-KEPT-T |

**Per round, shared:** `s218_pool_t<r>_s1/pool_meta.json` {seeds, per_n, draws_n, collide_n, dup_n, keep_n, sha256, builder_commit, labeller, labeller_sha256, card, dtype, label_bs, excl {set: hits}, T_max_by_n}; `s218_gate_t<r>_s1/eval_gate.json` {by_n {window, labeller, items, n_kept, keep, pi, pi_ci95, records}, pooled_by_window, GATE_T1 / GATE_T2 / GATE_T1p, fallback, labeller_full {n: keep, pi}, kept_vs_drawn, equiv {items, agree, disagreements}}.
**Base:** `s218_base_s1/eval_text.json` (fp32 re-eval of s170_cot_s1; required before any T1 reading).
**Latent side (read only):** s213_d3t_s1 / s215_ftself_s1 eval_vb.json; s216_r2_s1 eval_deep.json; s215_ftself_s1/eval_selfdepth.json (items_n2 at n 9–16, for DEPTH-PAIRED); s216_r2_s1/eval_deep.json items.

### 4.2 Per-cell readings (comparator-free)

| reading | exact definition | role | base s170_cot_s1 (bf16; fp32 re-eval replaces) |
|---|---|---|---|
| **T-ABSORB** (T1) | min_{n 9..12} a_T(n) ≥ 0.95 | in support; enabler; **text only, never a C-criterion** | 0.123 (n 12) |
| **T-DBS** (T1) | M_T = mean_{n 13..16} a_T(n) ≥ 0.80 | text depth past the self-labelled range (descriptive of text) | 0.078 |
| **T-ABSORB-2** (T2) | min_{n 13..16} a_T(n) ≥ 0.95 | in support | — |
| **T-RETAIN** (T2) | min_{n 9..12} a_T(n) ≥ 0.99 | must not break | — |
| **T-DBS-2** (T2) | M2_T = mean_{n 17..20} a_T(n) ≥ 0.80 | past support | — |
| M_far_T (T2) | mean_{n 21..24} a_T(n) (n 24 with overflow = failure; CTX-FEASIBLE beside it) | enters T-FULL | — |
| **n50 / n90** | first downward crossing of 0.5 (0.9) by a_T(n) over n 1..24, linear interpolation; never crossing → "> 24", already below at n 1 → "< 1" | **the comparison input** | 10.44 / 9.33 |
| WIDTH_T | n50 − n90 (unreadable if either is a sentinel) | descriptive | 1.11 |
| n50 / support | n50 / B_r | descriptive (latent 1.29 / 1.29 / 1.28) | 1.30 |
| **T-SELF** (T1) | T-ABSORB ∧ ¬censored | the text round-1 claim (in support) | — |
| **T-SELF-2** (T2) | T-ABSORB-2 ∧ T-RETAIN ∧ ¬censored | the text round-2 claim | — |
| Round-2 condition | T-ABSORB ∧ ¬FT-DIVERGED-T ∧ LABEL-AUDIT-T ∧ ¬CONTAMINATED ∧ ¬dead ∧ ARGS-DIFF ok (T1) | T2 runs iff it holds | — |

**Text mechanism label** (the SELFDEPTH / SD2 bar formulas applied to the text lineage's own fp32 curves; §4.3). It applies only under T-ABSORB (T-ABSORB-2), else **UNLABELLED**.
- T1: **T-FULL** if M_T ≥ max(0.80, MG_T); **T-GROW** if MG_T ≤ M_T and not T-FULL; **T-SHIFT** if SB_T ≤ M_T < MG_T; **T-SHRINK** if M_T < SB_T.
- T2: the same, with M2_T, the round-2 bars, and T-FULL additionally needing M_far_T ≥ 0.60.
- STOP-LOC is not defined for text (no pass index), so SHIFT and MIXED are not split.

### 4.3 Text bars by formula (registered; s1 instance on the bf16 curve, recomputed on the fp32 re-eval before any T1 reading)

| bar | formula (a_0 = text base fp32, a_1 = T1) | s1 instance on the bf16 base [DERIVED ts_panels] |
|---|---|---|
| R1 MARGIN-GROWS MG_T | ceil005(mean_{n 9..12} a_0(n) + 0.15) | ceil005(0.515 + 0.15) = **0.665** |
| R1 SHRINK SB_T | ceil005(mean_{n 9..12} a_0(n) − 0.08) | ceil005(0.435) = **0.435** |
| R2 M2_shift,T | mean_{n 13..16} a_1(n) | from T1 |
| R2 M2_prop,T | mean_{d 1..4} a_1(12 + 0.75 d), linear in n | from T1 |
| R2 MG2_T | ceil005((M2_shift,T + M2_prop,T) / 2) | from T1 |
| R2 SB2_T | ceil005((mean_{n 9..12} a_0(n) + M2_shift,T) / 2) | from T1 |
| absolute bars | T-ABSORB 0.95, T-RETAIN 0.99, T-DBS / T-DBS-2 0.80, M_far 0.60 | — |

The latent R1 / R2 bars came from the same formulas (SD2_v1 §4.3, V1). Degenerate bars (SB ≥ MG) → UNLABELLED (bars degenerate).

### 4.4 Comparison readings (pre-stated; read only after the text round's eval_text.json exists; the latent values are fixed committed numbers)

- **Inputs:** n50_L(0, 1, 2) = **10.31 / 15.46 / 20.54** (s1: D3-T eval_vb, FT-SELF eval_vb, R2-s1 eval_deep; THEORY §216 s1 result). n50_T(r) from eval_text. Δ_r = n50_T(r) − n50_L(r).
- **PARITY** iff |Δ_1| ≤ 0.5 ∧ |Δ_2| ≤ 0.5.
- **TEXT-AHEAD** iff ¬PARITY ∧ Δ_2 > 0.5 ∧ Δ_1 ≥ −0.5.
- **LATENT-AHEAD** iff ¬PARITY ∧ Δ_2 < −0.5 ∧ Δ_1 ≤ 0.5.
- **CROSSED** = every other case, i.e. the two rounds disagree: opposite signs outside the band, or round 2 inside the band with round 1 outside it (caught up or fell back). Reported with both Δ values. The four outcomes are exhaustive and exclusive [PROVED by case split; fixture].
- **ROUND-1-ONLY** (T2 not run because T1 failed the round-2 condition, uncensored): PARITY-1 iff |Δ_1| ≤ 0.5, else TEXT-AHEAD-1 / LATENT-AHEAD-1 by the sign of Δ_1, with the failing condition named.
- **TEXT-LABEL-STALL** (T2 not run because GATE-T2 or GATE-T1′ failed under T1 and under the base fallback): the comparison reads ROUND-1-ONLY, plus the row *"the text arm could not label its round-2 window under the matched gate rule"*, with each failing n and its keep / π. The latent arm passed the same gate (GATE-2, 216-1: keep ≥ 0.954 at n 13–16). This is a pre-stated outcome, not a censor. P ≈ 0.15 [ASSUMED]; the base's own REROOT keep at n 16 is 0.630 (§1.2a).
- **Censored text cell** (dead, FT-DIVERGED-T, LABEL-NOISY-T, CONTAMINATED, ARGS-DIFF, eval meta / items_sha256 mismatch): **every comparison row at and after that round fails** (rule b; none of PARITY / TEXT-AHEAD / LATENT-AHEAD passes) and reads "TEXT-CENSORED (r, reason)".
- **Round 0** (descriptive): Δ_0 = n50_T(0) − 10.31 on the fp32 base re-eval (bf16: +0.13).
- **POOLED** (secondary; only if the s2 text lineage is run): Δ̄_r = mean_T n50(r) − mean_L n50(r) over the lineages with a readable round r, the same four-way rule on Δ̄. If the latent s2 R2 cell is missing, the round-2 POOLED row reads UNPAIRED (the text s2 values are reported).
- **RATIO** (descriptive): n50 / support per round, text vs latent.
- **DEPTH-PAIRED** (descriptive; the same test items): per item at n 13–16 (T1 vs FT-SELF; latent items from s215_ftself_s1/eval_selfdepth.json items_n2) and n 17–20 (T2 vs R2-s1; eval_deep items at K′ n + 2), the 2 × 2 correct table and McNemar's exact p per n.
- **TOKEN-COST** (descriptive), per n 13–16 (round 1) and 17–20 (round 2), both arms:
  - sequential steps per correct answer: text Σ ntok / #correct; latent (n + 2) · 1,000 / #correct;
  - token-positions per correct answer: text Σ (P + ntok) / #correct; latent (n + 2) · Σ T_item / #correct, with T_item = len(syms) + 2 (the vbank frame).
- **CTX-FEASIBLE** (descriptive): the feasible share per n (n ≤ 23: 1.000; n 24: 0.971) and CTX-FEASIBLE accuracy at n 24; **CTX-BOUND** stated beside every comparison row (a third text round is context-infeasible, §1.7).
- **Never:** any text reading worded as latent C1 / C2 / C4 or as any north-star criterion. The north star is a latent model. TEXT-SELF is a baseline, and the comparison rows are about the latent arm's standing against it.

### 4.5 Descriptive readings (per text model)

| reading | definition | why |
|---|---|---|
| FORM-VALID f(n), STRICT s(n) | eval_text form_valid_by_n, strict_by_n | the strict analogue of chain exactness (chance ≈ 0) beside the answer accuracy |
| **FAIL-POS** | per n, among failing items: the median first_bad_pos and first_bad_step; the share of first deviations at positions > T_train,r | splits row-keyed from step-keyed failure (§3.5) |
| LABELLER-FULL (at each gate) | FULL-protocol keep / π on the gate items | what Lee's protocol would have kept |
| KEPT-VS-DRAWN (at each gate) | mean P + C, height and bracket count of kept vs all drawn items per n | selection bias (Literature failure mode 1) |
| LABELLER-KEPT-T | REROOT keep / π of the cell's model at n 9–24 on fresh items | the next labeller's reach |
| BF16-VS-FP32 (base) | committed bf16 acc_by_n vs the fp32 re-eval at n 1–16 | dtype sensitivity of the text eval |
| self_value_ce / self_copy_ce curves | log.jsonl | the arithmetic vs copy parts of the text loss (D1) |
| EQUIV (at each gate) | TF-GREEDY vs KV-cache generation agreement | labeller exactness |

### 4.6 Scorer theory/score_s218.py (to write; fixture-checked before any output)

- **Lineage table:** s1 {base s218_base_s1 (eval) / s170_cot_s1 (model), t1 s218_t1_s1, t2 s218_t2_s1, pool_t1 s218_pool_t1_s1, pool_t2 s218_pool_t2_s1, gate_t1 s218_gate_t1_s1, gate_t2 s218_gate_t2_s1, latent {r0 s213_d3t_s1, r1 s215_ftself_s1, r2 s216_r2_s1}}; s2 likewise.
- **ARGS-DIFF:** each FT cell vs s170_cot_s1/args.json (s2: s170_cot_s2), over score_s213's META whitelist.
  - Allowed: {init_from, steps, lr, self_pool, p_self, self_micro, out, args_ref (content-checked per 216-7's rule), mem_check_self, mem_check_steps, stop_at, keep_ckpt_steps, draws_log_steps}.
  - Values asserted: steps 12000, lr 3e-5, p_self 0.5, init_from = the lineage's previous model, self_pool = the round's pool. Anything else → ARGS_BAD (censor).
- **Pending rule** (215-7 / SD2 M5): a cell reads "pending (missing [...])" while any registered file is missing (log, args, draws_log, eval_text, eval_textself, the round's pool_meta and eval_gate, and the base's eval_text). `--final <lineage>` turns a missing file into a failure (rule b).
- **New functions:**
  - text_read: items_sha256 recomputed; meta asserts (card class 3090, fp32, tf32 false, cot_eval_bs 32, n 1–24); acc_by_n == mean(answer == y) per n; overflow items exactly the C_struct rule's.
  - n_cross (as score_s216's n50 / n90 over n 1..24);
  - bars_text (the §4.3 formulas; the latent instance must reproduce SD2's 0.63 / 0.40 / 0.785 / 0.60 when fed the latent curves, asserted);
  - mech_text, the round-2 condition, label_audit_t, memo_t, contamination_t;
  - compare (the four-way rule, ROUND-1-ONLY, TEXT-CENSORED, POOLED / UNPAIRED);
  - depth_paired, token_cost, ctx_feasible, fail_pos.
- **Fixtures (G-T12)**, one per panel and per rule:
  - the four §3.5 panels: N-SHIFT → LATENT-AHEAD, N-PROP → PARITY, POS-PROP → PARITY, POS-ADD → LATENT-AHEAD; BETTER → TEXT-AHEAD; a crossed curve → CROSSED;
  - |Δ| exactly 0.5 → inside the band;
  - T1 ¬T-ABSORB → ROUND-1-ONLY; T1 dead → TEXT-CENSORED and every comparison row fails; LABEL-NOISY-T; CONTAMINATED; ARGS-DIFF (lr 1e-4);
  - overflow items scored as failures; an items_sha256 mismatch → void; a missing eval_text → pending, then failure with --final;
  - the n50 sentinels; T-FULL needing M_far ≥ 0.60;
  - the latent bars reproduced from the latent curves; degenerate bars → UNLABELLED;
  - the POOLED / UNPAIRED rows.

---

## 5. Claims, wording, censoring, Goodhart

### 5.1 Claims and pre-stated wording

- **T-SELF** (T1) = T-ABSORB ∧ ¬censored. Only this wording: *"text-CoT depth extended from 8 to 12 by fine-tuning a §170 TEXT-COT model on its own re-rooted greedy chain (form-filtered, values never checked; labels gold-equivalent at π = [GATE-T1 pooled π]; no gold labels beyond n 8; in support)."*
- **T-SELF-2** (T2): the same sentence, *"from 12 to 16 … the round-1 text model's own chain …"*, with GATE-T2's π (or the BASE-LABELLED wording per window, as SD2's m11).
- **T-DBS / T-DBS-2:** *"text-CoT accuracy past the self-labelled range (n 13–16 / 17–20, the same untouched test items as the latent arm): mean [M]"*. Pass or fail is stated against 0.80. Each is a text-arm description, **never C2**.
- **Comparison wording** (exactly one applies):
  - **PARITY:** *"on Arithmetic(n, 11), round-matched text-CoT self-training and latent self-training moved the 50 % depth by the same amount (|Δn50| ≤ 0.5 at rounds 1 and 2: Δ = [Δ_1] / [Δ_2]; one lineage each)."*
  - **LATENT-AHEAD:** *"latent self-training moved the 50 % depth further than round-matched text-CoT self-training (Δn50 = [Δ_1] / [Δ_2])."*
  - **TEXT-AHEAD:** the mirror of LATENT-AHEAD.
  - **CROSSED:** *"the arms' order changed between rounds (Δ = [Δ_1] / [Δ_2])."*
- **On the same line, always:**
  - the latent cost: MEM-TRADED / MEM-ERODING (STAB_self 0.782 → 0.489 → 0.437);
  - the text cost: TOKEN-COST at n 17–20 (sequential steps and token-positions per correct answer, both arms), and **CTX-BOUND** (a third text round is context-infeasible on GPT-2);
  - "one lineage each", or the POOLED row with its lineage count.
- **Never:**
  - text readings worded as latent C1 / C2 / C4 or as any north-star criterion;
  - PARITY worded as "latent reasoning is as good as text CoT" in general. It is one task, one model size and two rounds.
  - LATENT-AHEAD worded as "latent beats CoT" without the matched-design qualifiers. The text arm was form-filtered exactly as the latent arm, it reads its own operands (D1), and the latent arm traded memory.
- **What it says for the north star:** a comparison row is context for C2 (the latent arm's depth extrapolation measured against the natural text baseline under identical self-training). It is not itself a C-criterion.

### 5.2 Censoring (rule b; each fails every reading of that cell and every comparison row at and after its round)

- **dead:** log last step < 12,000 (including a guard stop, §8).
- **FT-DIVERGED-T:** a_T(n) < 0.99 at any n 1–8 (the base reads 1.000 at every n 1–8).
- **LABEL-NOISY-T:** LABEL-AUDIT-T fails (π < 0.99 pooled or per window, or its CI does not overlap the gate's).
- **CONTAMINATED:** §1.8 item 8.
- **ARGS-DIFF:** a mismatch, per the §4.6 allowed sets.
- **eval_text:** missing at `--final`, or an items_sha256 / meta mismatch → every reading of that model fails.
- **Values, not missing:**
  - CTX-OVERFLOW items and non-terminating generations are **failures** in a_T(n);
  - never-crossed n50 / n90 read as sentinels ("> 24"), never as missing;
  - chance-level answers count as given (the free-generation a_T(n) keeps its chance floor, as latent a(n) does).
- **Pending** never passes (215-7).
- **Gates:** a failed GATE-T1 means no T1 (no cell, no spend). For T2, both labellers failing a window means no T2, and the comparison reads ROUND-1-ONLY with the reason.
- **Not in the text arm** (stated so nobody "fixes" it): LENGTH-FREE, VB-C1, K′24-STABLE, KEEPS-SELF-STAB and THINK-NECESSARY are latent-loop readings with no text counterpart. The text arm's only "kept" reading is FT-DIVERGED-T.

### 5.3 Goodhart analysis

| target / route | finding |
|---|---|
| **Stop at the support** (fail past n > B_r or past row T_train,r) | Zero loss on the support (S0-T a); irreducible by loss in text as in the latent arm. It is what the panels measure (N-SHIFT / POS-ADD → LATENT-AHEAD), not something the design prevents |
| **The filter rewarding short / easy items** (selection) | REROOT keep 0.97–1.00 at n 9–12 (§1.2a); its few failures are first steps on the longest inputs, so selection in round 1 is ≤ 3 % and toward shorter inputs. KEPT-VS-DRAWN is reported at every gate. T-ABSORB's denominator is **all** test items, so selection cannot inflate it. Under FULL (not used) keep is 0.05 / 0.00 at n 11 / 12: the selection Literature warned about, and the reason FULL is descriptive only |
| **A value check sneaking into the filter** (D1) | VALUE-SLOT accepts any value token; the builder's N.apply raises (G-T0). An accidental value check would show as π = 1.000000 with zero kept-wrong at every n *and* keep < strict-correct share; the gate script reports both |
| **Copying a systematic arithmetic error** (Lee App B.4) | LABEL-AUDIT-T prints each kept-wrong record's (op, a, b → v). A repeated cell is a systematic error; class mass ≤ 2(1 − π) |
| **Memorising pool items** | Fresh-once; MEMO-GAP-T |
| **The eval budget** | Gold-free (C_struct + 50, context-capped); a model that never emits EOT fails; no oracle stopping, no per-item budget selection |
| **The free-generation answer metric** | It credits a non-canonical chain that reaches the right number, and lucky guesses (floor ≈ 0.04–0.10), as latent a(n) credits chance answers. STRICT s(n) (form-valid ∧ correct, chance ≈ 0) is reported beside it; n50 is insensitive to a floor ≤ 0.1 [DERIVED: a crossing of 0.5 moves by < 0.1 n under a floor change of 0.05 at the measured slopes] |
| **Overflow** | It lowers a_T(24) by ≤ 0.029 by construction. It is a counted cost, reported as CTX-FEASIBLE beside, and enters only M_far_T and sentinel n50s |
| **Cross-arm bars** | No training objective reads n50 or Δ. The latent n50s are fixed committed numbers, so the text arm cannot move its comparator |
| **Card / dtype** | The text evals are fp32 on one card class. The latent evals are fp32 on the 3070. Near-tie flips move a(n) by ≤ 0.005 (the latent BS-CHECK saw 0 / 8,000 flips across bs), far below the 0.5-n band [ASSUMED for the cross-card case] |

---

## 6. OC (one lineage per arm unless POOLED; realised spend at registration $0) [arithmetic DERIVED ts_oc.json / ts_panels.json; every prior ASSUMED]

**Structural priors** [ASSUMED]:
- T1 uncensored 0.95; T-ABSORB 0.90 (the base learned n ≤ 8 by step ≈ 600 [MEASURED log.jsonl: loss 0.017 at step 500]; REROOT labels at keep ≈ 1 and π ≈ 1);
- gates and LABEL-AUDIT 0.99; **GATE-T2 / GATE-T1′ pass 0.85** (T1's own single-step reach at n 13–16; the base fallback is expected to fail at n 16, keep 0.630 [MEASURED §1.2a]); T2 uncensored 0.95;
- T-ABSORB-2 0.85 (rows to 527, ⅔ dose); T-RETAIN 0.95;
- panel priors as §3.5 (N-SHIFT 0.25, POS-ADD 0.20, N-PROP 0.15, POS-PROP 0.20, BETTER 0.10, WORSE 0.10);
- lineage noise on M / M2: sd 0.06–0.12 (Kuo); on Δn50: s at round 1 and s√2 at round 2, rounds correlated 0.5, s ∈ {0.2, 0.35, 0.5} (s 0.2 ≈ the round-0 spread measured between seeds).

**Per-cell readings:**

| reading | P(pass \| no effect) | P(pass \| effect) | headline |
|---|---|---|---|
| T-ABSORB | ≈ 0 (no FT effect: base min 0.123) | 0.90 [ASSUMED] | 0.85 (with censoring) |
| T-DBS (M_T ≥ 0.80) | ≈ 0 (base M 0.078); ≤ 0.009 under N-SHIFT / POS-ADD (sd 0.12) | 0.17–0.33 \| proportional panels (M 0.74–0.75); 0.80–0.95 \| BETTER | **0.15** (sd 0.09) |
| T-SELF-2 | ≈ 0 | — | **0.55** |
| T-DBS-2 (M2_T ≥ 0.80) | ≤ 0.012 under the additive panels (M2 0.52–0.53) | 0.73–0.93 \| proportional (M2 0.87–0.89); 0.89–0.99 \| BETTER | **0.21** (with T-SELF-2, sd 0.09) |
| TEXT-LABEL-STALL (given the round-2 condition) | — | — | 0.15 |
| both rounds readable (comparison) | — | — | **0.68** |

**Comparison readings, one lineage per arm, given both rounds readable** (× 0.68 for the unconditional rate; the rest is ROUND-1-ONLY ≈ 0.09 (¬T-ABSORB), TEXT-LABEL-STALL ≈ 0.13 and a censored text cell ≈ 0.10):

| truth (Δ_1 / Δ_2) | s | PARITY | TEXT-AHEAD | LATENT-AHEAD | CROSSED |
|---|---|---|---|---|---|
| **no difference** (0 / 0) | 0.20 / 0.35 / 0.50 | **0.92 / 0.61 / 0.39** | 0.04 / 0.15 / 0.23 | 0.04 / 0.16 / 0.23 | 0.01 / 0.08 / 0.15 |
| latent ahead by 1 n per round (−1 / −1) | 0.20 / 0.35 / 0.50 | 0.002 / 0.035 / 0.07 | 0.00 / 0.00 / 0.01 | **0.96 / 0.84 / 0.76** | 0.04 / 0.12 / 0.16 |
| text ahead by 1 n per round (+1 / +1) | 0.20 / 0.35 / 0.50 | 0.002 / 0.035 / 0.07 | **0.96 / 0.84 / 0.76** | 0.00 / 0.00 / 0.01 | 0.04 / 0.12 / 0.16 |
| at the band edge (+0.5 / +0.5) | 0.35 | 0.31 | 0.50 | 0.02 | 0.17 |
| additive panels (N-SHIFT / POS-ADD: ≈ −1 / −2) | 0.20–0.50 | ≤ 0.01 | 0 | **≥ 0.98** | ≤ 0.01 |
| proportional panels (N-PROP +0.20 / +0.34; POS-PROP +0.10 / −0.03) | 0.20 / 0.35 / 0.50 | 0.70–0.90 / 0.51–0.59 / 0.35–0.38 | 0.03–0.28 / 0.14–0.37 / 0.22–0.40 | 0.00–0.05 / 0.05–0.17 / 0.11–0.24 | 0.02 / 0.08–0.10 / 0.14–0.16 |
| **prior-weighted headline** | 0.35 | **0.19** | 0.18 | **0.59** | 0.03 |
| prior-weighted, unconditional | 0.35 | 0.13 | 0.12 | 0.40 | 0.02 |

**POOLED** (two lineages per arm, s / √2): P(PARITY | no difference) **0.99 / 0.82 / 0.60** at s 0.2 / 0.35 / 0.5, and P(LATENT-AHEAD | −1 / −1) 0.99 / 0.92 / 0.84. The prior-weighted headline (s 0.35) is PARITY 0.26, LATENT-AHEAD 0.57.

**How to read it:**
- **LATENT-AHEAD is the most likely outcome under the stated priors (0.59)**, because the additive families (0.45 prior) predict about −1 n per round and the comparison detects that with P ≥ 0.98.
- **PARITY is a weak reading at one lineage:** P(PARITY | true parity) is only 0.61 at s 0.35. A null on PARITY is weak evidence of a difference. A PARITY pass is moderately strong: P(PARITY | 1-n difference) ≤ 0.07. The POOLED row (with s2) lifts P(PARITY | parity) to 0.82.
- **The informative split is additive vs proportional text extrapolation.** The latent lineage was proportional (n50 / support 1.29 / 1.29 / 1.28). A LATENT-AHEAD reading says the latent's proportional margin is something text self-training on GPT-2 does not reproduce. A PARITY reading says the proportional law is not specific to the latent loop.
- **Rule check** (OC block + censoring rule): every registered reading above carries P(pass | no effect) and P(pass | effect) at its seed count (one lineage; POOLED for two). Never-reached events (overflow, non-termination, never-crossed n50) count as failures or sentinels (§5.2). Realised $ at registration: $0.

---

## 7. Gates (all green before any spend on the stage they guard; Audit reviews)

| gate | check |
|---|---|
| **G-T0 NO-GOLD** | Builder gold patches (N.apply, N.value, N.eval_tokens, N.cot_steps, N.cot_ids raise) before generation. Record key set {syms, n, chain_label, values} asserted at load; no root key in a self item. Grep: `chain_label` / `values` only in the writer and the loader's SD-LABEL-MAP-T line. A self-batch smoke counts cot_ids / cot_steps calls on self items = 0. No `s218_gate` string in nested170.py |
| G-T1 EXCLUSION | pool ∩ §1.8 item 2 = ∅ by exact syms; per-n draws / collisions / duplicates logged; n(syms) == generator n asserted, then dropped |
| **G-T2 COMPOSE-T** | compose_T(syms, values) == the TF-GREEDY accepted token sequence, bitwise, on every gate-kept item. == N.cot_ids(gold root) on every kept item whose values equal gold, and ≠ on every kept-wrong item (separate gold process). The decomposed CV-T flags agree with the candidate-equality test on 100k synthetic perturbed steps (unit test: drop, duplicate, swap, wrong span, wrong bracket, non-value token) |
| **G-T3 STREAMS** | coins == s215_ftself_s1/draws_log.json coins at all 12,000 steps (so 5,992 self steps; s2: vs R1-s2's draws_log or a separate-process replay); plain_hash at steps 1..200 == the base stream's (rng seed 1, train_items, recomputed separately); the realised per-n self counts == the values precomputed from the pool order |
| G-T4 INIT + RESUME | `--init_from --stop_at 2 --ckpt_every 1`: model at step 0 == the init model.pt bitwise; INIT_FROM prints missing [] / unexpected []; ckpt_last at steps 1 and 2; a resume runs step 3 with cursor / rng / rng_s / AdamW restored, and its step-3 loss equals an uninterrupted run's (CPU fp32 smoke) |
| G-T5 ARGS-DIFF | printed in each launch log: only the §4.6 allowed keys differ from s170_cot_s1/args.json |
| G-T6 LOAD | each new model.pt loads strictly into make_trunk's GPT2KV; `--eval_only` runs on it |
| **G-T7 PLAIN-PATH** | without the new flags: construct + 1 step, CPU-bitwise vs the pre-edit tree, for **all four §170 arms** (direct / inject / textcot / readslots), on the final tree |
| G-T8 GRADFLOW | scratch/gradflow_check.py on a self batch: nonzero grad on every trunk parameter tensor; decorators by ast (no no_grad on the new paths) |
| **G-T9 MEM** | on each FT host: `--mem_check_self --mem_check_steps 2` on the 64 longest pool records of the deepest n (T1 n 12, T2 n 16) at bs 64; peak after step 2 incl. AdamW; self micro = the smallest of {1, 2, 4, 8} with peak ≤ 21.0 GiB (projected 2 / 4) |
| G-T10 TIME | forced G-TIME (odd steps deep self, even plain) → training projection; plus the measured throughput of the base re-eval stage → the lineage projection → §8 admission |
| G-T11 OFF-HOST | full-state checkpoint mirror every 2k (sha-checked after the first copy). **Every stage's outputs** (pool + pool_meta, eval_gate, eval_text, eval_textself) are mirrored at the stage boundary, before the next stage starts. The resume is G-T4's |
| G-T12 FIXTURES | score_s218 fixtures (§4.6) pass before any output |
| **G-T13 DIFF REVIEW** (Audit) | T1 (trainer: init_from, loader, coin replay incl. the 64 discarded draws, compose_T, per-type micro, logging, ckpt / resume); T2 (TF-GREEDY: candidate from shape only, masks / positions / padding, REROOT and FULL loops, no simulator call); B1 / B2; E1 (gold-free budget, CTX-OVERFLOW, per-item store, items_sha256); E2 |
| G-T14 LAUNCH | RTX 3090 only for every text-arm GPU stage (card in every file); drain waiters armed at launch; no edit of nested170.py while any job importing it runs or deploys |
| **G-T15 GATES** | GATE-T1 (T1), GATE-T2 + GATE-T1′ (T2) per §1.2 on the lineage's host before the pool is used. A pool stage starts only if eval_gate.json reads PASS for its window (mechanical check in the stage script; fallback labeller per window as §1.2) |
| **G-T16 EQUIV** | on 250 gate items per n, plain KV-cache greedy generation (fp32, the eval loop) reproduces TF-GREEDY's kept flag and label on ≥ 0.995 of items; every disagreement listed with its top-2 logit margin. Fail → the labeller reverts to generation (slow; Theory re-plans the cost) |
| **G-T17 SMOKE-AT-SCALE** | locally before deploy: the builder's exclusion set at full size (≈ 1.4M syms incl. the latent pools) and the trainer's pool load on a full-size 384k synthetic pool; timings logged (b310's O(N²) lesson) |
| **G-T18 EVAL-SMOKE** | on the eval card: eval_text on the base at n 1–3 plus the 32 longest n-24 feasible items (peak ≤ 21 GiB on a 3090; ≤ 7 GiB if an 8 GB card is used) and the 29 overflow items flagged and not generated; output to scratch/_tmp, never into a cell directory |
| **G-T19 BASE FIRST** | s218_base_s1/eval_text.json (fp32) exists, and its bars (§4.3) are recorded in a Theory note, before any T1 output is read |

---

## 8. Cost, caps, admission, guards, wall-clock [DERIVED ts_cost.json; price band $0.17–0.26/h and throughputs ASSUMED]

**Plan A (recommended): every text-arm GPU stage on one RTX 3090 host per lineage** (one GPU class for labelling, gates, training and evals). The local 3070 stays free for §216 s2 and §217.

| stage (per lineage) | hours | notes |
|---|---|---|
| deploy, smokes (G-T4, T7–T9, T16, T18) | 0.4–0.6 | |
| base re-eval (eval_text, n 1–24) | 1.2–2.4 | 293k–3xxk sequential decode steps (ts_cost); a slow host or a 3070-class rate gives the upper end |
| GATE-T1 (8k) + LABELLER-FULL + EQUIV | 0.2–0.4 | |
| T1 pool (384k, REROOT TF-GREEDY) | 0.6–1.0 | 365 token-positions per item |
| T1 training (12k) | 1.4–2.9 | self 0.59 s, plain 0.27 s per step; ×2 host variance |
| T1 eval_text + eval_textself | 1.5–2.7 | |
| GATE-T2 + GATE-T1′ (16k, labeller T1) + EQUIV | 0.3–0.5 | |
| T2 pool (384k) | 0.9–1.6 | 575 token-positions per item |
| T2 training (12k) | 2.2–4.3 | self 1.03 s per step |
| T2 eval_text + eval_textself | 1.5–2.7 | |
| **per lineage** | **10.2–19.1 h** | **$1.73–4.97** |
| mirror egress | | ≲ $0.05 per lineage |
| **s1 + s2 (recommended)** | parallel hosts | **$3.5–10.0** |

- **Caps:** **family cap $12.0** (GUARD_STOP), per-lineage guard **$6.0** (realised, incl. refused / BAD hosts). **Admission per lineage:** a host is used only if its G-T10 projection for the whole chain is ≤ **$5.5** at its price. Otherwise there is no launch on that host, with up to 2 more measured refusals per lineage (215-2's counting). **Steps are never reduced.** A guard stop is a dead run.
- **≤ $30:** the family cap is under the user's $30 auto-approval rule (Audit approves). Unlike §216 this needs no MEM-TRADED user OK (the text arm has no memory channel). The cost is stated here as the rule requires.
- **Plan B (local evals):** training only on the cloud (**$0.6–1.9 per lineage**: T1 + T2 training plus deploys), with pools, gates and evals on the local 3070 (fp32). That costs about **8–11 h of 3070 time per lineage** (pools 1.6–2.6 h + gates 0.5 h + three eval_text 5–7 h at 1.7–2.4 h each), against the 3070's ≈ 19–22 h §216 queue. It saves $1–3 per lineage and moves the text evals onto the latent arm's eval card. Q3 asks whether the idle local RTX 2070 (2 % utilisation at 12:0x, 8 GB, fp32 only) could take this queue at $0.
- **Wall-clock:** one lineage ≈ 10–19 h end to end on one host; the base re-eval can run on a second host in parallel with GATE-T1 + the T1 pool (same cost, about 2 h sooner).
- **Retention:** keep model.pt of T1 and T2 (labellers; possible round-3 text labeller is moot by CTX-BOUND, so T2's model.pt may be deleted after its evals are scored, with Theory's OK), all eval / log / args / draws_log / pool_meta / eval_gate (with per-item records). Delete checkpoints and mirrors at landing. Delete each pool after LABEL-AUDIT-T and MEMO-GAP-T are scored (pool_meta + seeds + builder commit regenerate it).

---

## 9. Open questions for Theory

1. **Labeller protocol.** REROOT (the latent labeller's procedure) is primary; FULL (Lee's) is descriptive at the gates. FULL cannot pass GATE-1's keep_n ≥ 0.70 at n 11–12 (keep 0.05 / 0.00 measured). Confirm. The alternative is a Lee-style +1-per-round FULL arm, which is a different schedule and so not matched.
2. **TF-GREEDY as the labeller's computation** (exact in exact arithmetic; 90 / 90 CPU agreement; G-T16 on GPU) instead of KV-cache generation (≈ 10–20× slower per pool). Confirm, including the fallback if G-T16 fails.
3. **Card plan.**
   - Plan A: one RTX 3090 per lineage for every stage ($1.7–5.0 per lineage).
   - Plan B: local 3070 for pools / gates / evals ($0.6–1.9 per lineage, 8–11 h of 3070 per lineage).
   - Is the idle local RTX 2070 an approved eval card?
   - Plan A puts the text evals on a different card class from the latent's 3070. It is read only through n50, at fp32 [ASSUMED negligible].
4. **The s2 text lineage:** recommended in parallel (+$1.7–5.0), with the s1 pair primary and a POOLED row. Or s1 only?
5. **The PARITY band ±0.5** (as briefed) gives P(PARITY | true parity) only 0.61 at one lineage and s 0.35. Keep ±0.5, or widen to ±0.75? (Not computed here; it would trade sensitivity to a 1-n difference.)
6. **a_T(n) = free-generation answer accuracy** (§170's protocol, chance floor ≈ 0.04–0.10, matched to the latent a(n), which also scores the answer), with STRICT s(n) descriptive. Or STRICT as the registered metric? n50 differs by < 0.1 n between them at the measured slopes [DERIVED §5.3].
7. **Base re-eval at fp32** replaces the committed bf16 curve for every text bar (G-T19). Confirm that the committed bf16 eval.json is descriptive only.
8. **CTX-BOUND:** the text arm stops at round 2 by construction (n 25: 55 % feasible; n ≥ 28: 0 %). State it in the registration as a pre-stated end, not a failure.
9. **The text mechanism label** reuses the SELFDEPTH / SD2 bar formulas on text curves without STOP-LOC (no pass index), so SHIFT and MIXED are merged. Acceptable?
10. **Coin-stream replay:** the 64 discarded rng_s draws per self step make the text arm's self steps land at the latent's step indices. This is cosmetic for the readings but makes G-T3 an exact check. Keep, or let the text trainer draw only the coin, with exposure then equal in law (5,992 ± 55 self steps)?
11. **Latent baselines are fixed numbers** (10.31 / 15.46 / 20.54). FT-SELF's n50 15.46 comes from eval_vb (bs 32) and R2's 20.54 from eval_deep (bs 16), both 3070 fp32. Their BS-CHECK showed max |Δ| 0.000, so mixing those two sources needs no correction.
12. **The round-2 text labeller risk** (§1.2a): the base's REROOT keep at n 16 is 0.630 < 0.70, so the fallback cannot rescue T2's window, and T2 rests on T1's single-step reach (P(GATE-T2 passes) 0.85 [ASSUMED]).
    - Recommended: keep the matched gate rule, with TEXT-LABEL-STALL as a pre-stated reading (it is itself a finding: the latent single step reaches further).
    - Alternatives, both unmatched: a text-only bridge (e.g. a T1.5 round at n 13–14), or a keep_n bar lowered for text. Decide before registration, not after GATE-T2.
    - A cheap check before registration is not possible: T1 does not exist yet. The base's numbers are the only precondition.
13. **LABELLER-REACH as a descriptive cross-arm row:** text base REROOT keep vs D3-T chain keep per n (§1.2a's last two rows), and the same for T1 vs FT-SELF (LABELLER-KEPT-T vs FT-SELF's eval_chainlabel_fresh). Add it to §4.4?

---

## 10. Verification log (this draft)

**$0 CPU checks** (research6/scripts/, JSON beside each; CPU, ≤ 2 threads, Idle priority, `python -B`, PYTHONDONTWRITEBYTECODE=1; no file written under D:/Latent):
- **ts_tokens.py → ts_tokens.json** [DERIVED; generator statistics of test_set(1000, n_max 24)]:
  - struct_len (shape only) == len(N.cot_ids) on all 24,000 test items (asserted);
  - per-n P / C / P + C means, maxima and quantiles; feasible share (n 24: 0.971, 29 items); training shapes for the T1 / T2 mixtures (rng 23190).
- **ts_label_check.py → ts_label_check_smoke / _gen / _s1 / _s1_deep.json** [MEASURED; base s170_cot_s1, CPU fp32, fresh items rng 23180]:
  - FULL and REROOT keep / π / fail-stage histograms;
  - TF-GREEDY vs KV-cache greedy generation 90 / 90 (smoke n 9: 10; _gen n 11–12: 80).
- **ts_failpos.py → ts_failpos_gen.json, ts_failpos_s1.json** [DERIVED from the MEASURED FULL results]: where FULL chains break, in stages and rows.
- **ts_summary.py → ts_summary.json**: the §1.2a table, with one-sided 95 % Clopper–Pearson lower bounds on π per n and protocol.
- **ts_cost.py → ts_cost.json** [DERIVED; forms ASSUMED]:
  - eval decode steps under cot_answers' batching, calibrated on the §170 eval's 1,411 s;
  - TF-GREEDY token-positions per item;
  - training-time and memory models.
- **ts_panels.py → ts_panels.json** [DERIVED from MEASURED s170_cot_s1 / s2 eval.json]:
  - base n50 / n90; four hypothesis panels per round, with M, n50, Δn50 vs the latent;
  - the position-logistic fits; the comparison OC by Monte Carlo (400k draws per case);
  - context feasibility at n 21–30 on fresh draws (rng 23192; the sealed n 25–30 test items were never generated or read).
- **ts_oc.py → ts_oc.json** [DERIVED; priors ASSUMED]: per-cell OC (T-DBS / T-DBS-2 per panel and lineage sd), structural headline (incl. P(GATE-T2) 0.85, set after §1.2a's n 16 keep), comparison OC for one lineage and POOLED.

**Files read** (explicit paths only):
- docs/THEORY.md §215–§217 and their notes (lines 18563–19140 at 598e75de);
- SELFDEPTH_v1.md; SD2_v1.md (all, incl. Addendum 1);
- LIT_ITERATED_SELFTRAIN_2026-10-17.md, LIT_FAITHFULNESS_SELFIMPROVE_2026-10-18.md, LIT_STEPFACTOR_D1_2026-10-14.md;
- boxes/nested170.py (task, render, cot_steps, cot_ids, test_set, train_items, loss_items textcot, cot_answers, mem_check, the training loop's lr / clip, eval);
- boxes/core.py (GPT2KV.forward: `wpe[p0:p0 + T]`, SDPA attention); boxes/s5.load_gpt2; boxes/memcap.py; boxes/selfdepth.py; scratch/s216_build_pool.py (head); boxes/vbank.py l.641–659, l.744–770 (rng_s);
- boxes/runs_fsq/s170_cot_s1 / s2: args.json, args_eval.json, eval.json, log.jsonl (step times), mem_check.json, train_oom.log (tail); s213_d3t_s1 / s215_ftself_s1 args.json; s215_ftself_s1/draws_log.json (record count only).
- **Directory listings only:** boxes/runs_fsq/s215_pool, s216_pool_s1 and the s21[56]_gate / pool directory names.

**Seeds grep** (docs/THEORY.md, docs/research_*/*.md, boxes/*.py, theory/*.py, scratch/*.py, scratch/*.sh at 598e75de): 23180–23182, 23190–23193, 23200 / 23201, 23260 / 23261, 23270, 23280, 23300 / 23301, 23360 / 23361, 23370, 23380, 23400 / 23401, 23460, 23470, 23500 / 23501, 23560 / 23561, 23570, 23580: 0 hits each.

**Still unbuilt (flagged):** every code edit of §1.9; score_s218; the stage scripts. Memory, time, labelling and eval projections are model extrapolations. G-T9 / G-T10 / G-T16 / G-T18 decide.

*Draft written 2026-10-18 12:04–13:17 EDT (Opus design subagent for Theory; $0; nothing committed, registered or launched).*
