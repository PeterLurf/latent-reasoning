# 04 GSM8K STAGED DATA + EVAL PLAN (wave-4 lead design, 2026-11-03 17:2x EST; read-only draft, not registered; no numbers computed from stored outputs)

## 1. Goal and backlog IDs

One staged ladder from bun's mod-11 word programs to GSM8K, **one change per rung**, each rung carrying the same matched arms and the same gates, so that a C-claim at rung k is comparable to rung k-1. Covers backlog §12.6 (NUM-100 = D3, DIGIT-AUX-10K, O1, BRANCH-DAG = D4, IGSM-LOOP, L1 = T4, LLM-REWRITE, SOCRATIC-CUE PROBE, GSM8K-SYMBOLIZED-TRAIN, E PROTOCOL, SOCRATIC SELF-CUE), §13.3 numbers NR1-NR5, hybrid HY2 / HY4, scale S10 / SC2, and says where C4 (§267 EM) and the plan-free P-lane (§260 / §261) enter. Priors: §209 / §210 (CODI in our pipeline reaches 0.379 at 20 epochs; the 4-epoch gap was budget; depth m >= 3 is the weak axis at 0.265), §200 / the 10-13 template draft (Aug train is shorter and shallower than the test: mean m 2.60 vs 3.25; 77.2% of Aug is exactly recomputable; 0 verbatim test questions in Aug train).

## 2. The rungs (generator spec, tokens, aux / readout, what changes)

**Common skeleton (bun, §259):** ow_words chain programs, forward order, frames 0-5, 0-2 distractors (§257 mix), K 8, step cue `How many <u> does <S_t> have? The answer is:` at every pass (inject_st.py:40-46 CUE_FMT / cue_str; inject_aw.py:76-93 tok_aw), the ':' row writes DeltaMem, per-pass aux `[EOT, " Step", " t", " :"] + " <S_t> has <v_t> <u>." + EOT` teacher-forced and **sequence-scored** (inject_ow.py:92-103 aux_logits; nested170.py:260-272), answer = argmax over NUM_IDS at the decoder's ':' (nested170.py:20 `NUM_IDS = [TID[" %d"] for v in range(11)]`; inject_ow.py:251, 278; nested170.py:407). inject_ow.py:43 asserts " 0".." 10" are single tokens.

| Rung | Change (only this) | Generator spec | Tokens / readout |
|---|---|---|---|
| **R0 NUM-100** (D3) | value alphabet Z_11 -> integers 0-99 | ow_words with `--num_p 100`: ops add_c / sub_c (c in 1..20), mul_c (c in 2..9), add_v / sub_v / mul_v (givens 1..20 for add/sub, 2..9 for mul), no mod reduction, **backward-uniform**: draw S_d ~ U{0..99}, then for t = d..1 draw op_t and operand uniformly among the pairs with an in-range exact pre-image (sub needs S_t + c <= 99, add needs S_t - c >= 0, mul needs c divides S_t); restart if a step has no pre-image (restart count logged as ow_words does). Givens / distractor values 1..20 or 2..9 by op. | " 0".." 99" are single GPT-2 tokens (checked with tiktoken 11-03: all 100). NUM_IDS -> NUM_IDS_P of length 100; readout unchanged (one token). Aux unchanged. |
| **R1 DIGITS** (NR4 R2L format + DIGIT-AUX-10K) | numbers 0-999, rendered as **fixed-width zero-padded spaced digits, least-significant digit first** (`--val_fmt digits_r`), W 3: 7 -> " 7 0 0", 123 -> " 3 2 1" (each " 0".." 9" one token; the carry runs with the decode direction, 2307.03381; fixed width = the R2L alignment of 2402.14903); standard order is a descriptive arm | R0's backward-uniform draw with S_d ~ U{0..999}, add/sub c in 1..99, mul c in 2..9 | aux target = the step sentence with the W digit tokens (sequence-scored already; GIVEN.format at ow_words.py:104 and nested_words.py:82 gets a value formatter); the **answer** = W-token greedy decode after ':' (aux_greedy-style, nested170.py:275-283) replacing the one-token argmax in dec_logits (inject_ow.py:86-90); the ':' write source is unchanged. The train-log value-token acc `vidx = q.index(HAS_TOK) + 1` (inject_ow.py:70, 267-270) reads one token: it becomes a W-token all-correct check. Arm DIGIT-HEAD (NR3): W x 10 per-digit heads on the ':' row with a per-digit swap gate; its write uses write_seq's T > 1 path (deltamem.py:179-183). |
| **R2 O1** | exact division and 2-digit multiplication | mul c in 2..99, div_c with c dividing S_{t-1} exactly (backward: S_{t-1} = c S_t), range 0-9999, W 4 | as R1 |
| **R3 BRANCH-DAG** (D4) | two spines merged by one add_s / mul_s (parallel branches, not back-edges) | ow_words `dag=True` extended with a second spine of length d_2 in 1..d-1, topological forward order; the cue names the step entity in a fixed topological order | first at Z_11 (one change), then at R0's alphabet. **Precondition** MC2 KEEP-k RECALL (two live values in memory). |
| **R4 IGSM-LOOP** | iGSM med-pq items (2407.20311; generator MIT), mod 23 | iGSM's own generator; the cue = iGSM's "Define X" sentence order | 0..22 single tokens; readout as R0 |
| **R5 L1 = T4** | paraphrase families: 40 train / 20 held frames per op | nested_words FRAMES (nested_words.py:40-82) extended | unchanged |
| **R6 LLM-REWRITE** | Claude rewrites R2 / R3 items as prose; each rewrite re-solved by the independent regex solver (ow_words.py:178-184 solve) and kept only if the solution matches | 2309.12284-style rephrase; rejection logged | unchanged |
| **R7 SOCRATIC-CUE PROBE** (HY4 stage 1) | real GSM8K items; the cue = GSM8K's Socratic sub-questions (2110.14168 release, MIT) aligned 1:1 to the `<<e_t>>` equations | Aug train (gsm8k_data.py:100-122 load_aug; filter_bad_eq) restricted to items that are in GSM8K train (93% verbatim) with a Socratic sub-question count equal to m and all intermediates integers in the rung's range | K 10 (gsm8k_stepaux.py:1-20), aux " <<e_t>>", answer decode after "###" scored by bench.tasks.norm_number (bench/tasks.py:142-148) |
| **R8 SYMBOLIZED-TRAIN** | in-house templates from GSM8K TRAIN only (the 10-13 draft's R_nu / R_x: gsm8k_data.py:137-215, 283) | per-draw name relabel + exact number resample | as R7 |
| **R9 GSM8K-Aug full** | budget: 20 epochs (§210's lesson) | GSM8K-Aug 385k | its own approval (> $30) |

**HY2 LATENT SUB-CALL** (calculator) is an arm at R2: the pass emits a tool query (2302.04761 / 2211.12588 / 2211.10435 style) and reads the exact product back as text; it is a C3 / C1 comparator, not a latent claim. **S10 / SC2** sit beside R1: Qwen3-0.6B-Base (2505.09388) tokenises digits individually, so the R1 digit format is native there; SC2's shortcut-propensity index (2k direct FT per base, SPI(d)) runs on R0 data before any port.

## 3. Where C4 and the P-lane enter

- **C4 (§267 EXACT-EM-11):** the forward DP marginalises an 11-state Markov chain. At **R0** it becomes EM-100: 100 states, 1 + 100(d-1) upper-half forwards per item (d 2: 101; d 3: 201), tiled 100x. The ops are injective but no longer bijections on the alphabet, so the posterior collapses faster (fewer consistent paths), and the fp64 gate "DP == explicit path sum" is 100 paths at d 2. From **R1** on (1000 states) the exact DP is out; the C4 lane switches to K2 SAMPLED-VALUE RLOO on the same categorical head (8 samples), with the gradient cosine against EM-100 at R0 as its oracle. **C4 enters only on EM-C4 2/2 at §267 stage A**, first as EM-23 on R4's modulus (23 states, 1 + 23(d-1) forwards), then EM-100 as a learnability pilot (bs 16, d_max 2, 2k steps; cost §9).
- **P-lane:** every binding cell is plan-given (step cue). Plan-free arms (`--cue generic` §260, `--cue target` §261; inject_aw.py:233) are descriptive at R0-R3 and become binding at **R7**, where the model's own sub-question (SOCRATIC SELF-CUE = P1 on GSM8K) is the test: the binding cell there is the given-Socratic-cue one; SELF-CUE is a second cell gated on §261's TC-HALT reading PLAN-FREE. Order-free (§263) is a prerequisite for R6 / R7 (prose order is not chain order).

## 4. Matched arms at every rung (E PROTOCOL)

0. **Learnability pilot** (process rule 5): 2k steps of DIRECT and LAT on the rung's stream ($0.3); PARK the rung if d 2 < 0.50 at 2k in both.
1. **LAT** (binding): bun's recipe, 8k steps, 2 seeds (3 seeds whenever a C2 row is claimed: Note 269-1), **two init cells**: from the previous rung's s1 and from bun_s1 (`--init_model`), so the alphabet change is read with the lineage held fixed; the registration carries the args diff against bun_s1.
2. **TEXT-CoT, same data** (§266 X3 recipe: sentence steps and value-only steps, 24k, CurGate vbank.py:578-600): VALID if d 1-4 >= 0.95; LATENT-SPECIFIC if t7 <= lat7 - 0.10 and t8 <= lat8 - 0.10.
3. **PAUSE** (B1 PAUSE-PASS): the same K passes, the ':' write replaced by a learned constant vector (`--pause_write 1`, non-zero init, a usage monitor on its norm): PAUSE-SOLVES if d 1-4 >= 0.95, which voids C1 at that rung.
4. **DIRECT** (`--arm direct`, one pass): the shortcut-exclusion floor; NOT-DECODER <= 0.20 = LAT at K'0 vs K'8.
5. **Answer-marginal / last-number / answer-in-question baselines** on the rung's test set (the GSM8K_REPRO baselines): the backward-uniform draw targets max answer frequency <= 0.03 and answer entropy >= 6.4 bits at R0 (log2 100 = 6.64).
6. **C3:** generated tokens -> FLOPs by the §12.1 formulas; LAT vs TEXT at matched FLOPs per d (descriptive).

## 5. Decontamination and licences

- **13-gram filter** (GPT-3 2005.14165 App. C): lower-cased, whitespace-normalised word 13-grams of every training question (R6-R9, and R0-R5 for the record) against the 1,319 GSM8K test questions and every OOD eval set used (GSM-Plus 2405.00332, SVAMP, GSM-Hard, GSM-Symbolic main / p1 eval-only). Any training item sharing one 13-gram is dropped; `decon.json = {n_in, n_dropped, share, examples[:20]}`. Gate: share < 0.01 at R6-R8, else investigate before launch. New CPU script theory/gsmtpl/decon13.py beside audit_aug.py / seen_split.py (listed, not read). Also reported: exact-match share and the structure-fingerprint seen share (seen_split.py).
- **Licences:** GSM8K MIT; GSM8K-Aug (2311.01460; the da03/Internalize_CoT_Step_by_Step repo shows MIT, the 385k file's own licence to be confirmed); iGSM MIT (facebookresearch/iGSM LICENSE, checked 11-03); TinyGSM MIT; TemplateGSM (2411.18104) CC-BY-4.0 (optional, fidelity-filtered); **GSM-Symbolic CC-BY-NC-ND and test-derived: eval only, never committed**; ASDiv NC: not trained on; GSM-Plus (2402.19255; its GitHub page shows no LICENSE file, 11-03) / GSM-Hard / SVAMP: eval only, licences to be verified by Literature at Stage 0. Claude-rewritten items (R6) are our own data; the prompt and the solver-check rate go in the registration.

## 6. Dependency DAG and the 3-week schedule

```
Stage 0 ($0 CPU) ─► R0 NUM-100 ─► R1 DIGITS ─► R2 O1 ─► R6 LLM-REWRITE ─► R7 SOCRATIC PROBE ─► R8 SYMBOLIZED ─► R9 Aug full
                     │   │            │                       ▲                 ▲
                     │   └─ EM-100 (C4, after §267 EM-C4)    │                 │
                     ├─► R3 BRANCH-DAG (Z_11 first; needs MC2 KEEP-k) ─────────┘   §263 ORDER-FREE ─┘
                     ├─► SC2 SPI (bases) ─► S10 Qwen3 port (digit-native)
                     └─► R4 IGSM-LOOP (independent, after R3)      R5 L1 paraphrase after §263
```
Rung k+1 is launched only when rung k reads LEARN + MECH (§7) in 2/2 seeds; a MECH fail at rung k is itself the finding ("the write holds Z_11 but not 0-99") and routes to NR1-NR3.

| Week (design <= $30 each) | Content | $ |
|---|---|---|
| **W1 D-A** | Stage 0: generator (backward-uniform + forward-rejection test draw, P=11 path byte-identical shas), tokeniser table, decon13 on the existing Aug set, baselines incl. P(op_d \| answer); $0.1 bun_s1 ':'-row probe on 11..99 values (part_probe); 2k learnability pilot; **R0 NUM-100**: lat 2 seeds x 2 inits, text 1, pause 1, direct 1; NR1 NUM-CODE-PROBE on the R0 ':' rows ($0.2); EM-100 learnability pilot (bs 16, 2k; only if §267 has read EM-C4) | 0.1 + 0.3 + ~5.5 + 0.2 + 1.5 ≈ 7.6; guard 10 |
| **W2 D-B** | **R1 DIGITS** (lat 2 seeds + DIGIT-HEAD arm + text + pause), NR5 carry-stratified read, SC2 SPI on 3 bases (R0 data); **R3 BRANCH-DAG** at Z_11 (after MC2) | ~5 + 1 + 1 + 1.5 ≈ 8.5; guard 12 |
| **W3 D-C** | **R2 O1** (lat 2 + HY2 sub-call arm + text), **R4 IGSM-LOOP** pilot, R6 LLM-REWRITE build + solver-check ($0 API-side), R7 Stage 0 (Socratic alignment counts, decon, $0) | ~4 + 1.5 + 2 ≈ 7.5; guard 10 |

R7-R9 (~$2 + $3 + $25-45) are week 4+ and need their own approvals; R9 needs the user's GSM8K-scale approval (> $30).

## 7. Gates, criteria, OC, censoring (R0 NUM-100 as the registrable template; later rungs substitute their alphabet)

**Build gates.** (G1) `--num_p 11` and `--val_fmt one` and `--pause_write 0` are defaults and the flags-off run is **bitwise** bun_s1's smoke (p = 0 equivalence; OW.set_sha of TEST2 / PAIRS2 / TRANSFER unchanged, ow_words.py:187). (G2) The independent solver (ow_words.py:178-184 via NW.solve) re-derives the answer on 100k generated items at P 100 (nested170.py:122-129's self_check pattern). (G3) gradflow_check read from the decorators down (digit decode path, pause vector). (G4) mem_check at the d_max shape with micro pinned, 1 GiB headroom. (G5) 200-step smoke at full data scale with the answer CE falling. (G6) EM-100: fp64 DP == explicit path sum on 4 items at d 2. (G7) usage monitor on the pause vector's norm and on the digit heads' logit margin. (G8) the scorer's fixtures pass before launch.

| Criterion | Field | Bar | Kind |
|---|---|---|---|
| LEARN | eval_acc d 1-4 (TEST2-100 backward-uniform AND forward-rejection draws, n 1000/d each) | >= 0.95 each, both seeds, both draws | binding |
| MECH-WS | eval_ws d 4 passes 1-3 | >= 0.90 | binding (mechanism at the claimed depth) |
| MECH-SWAP | swap_k d 4 follow | >= 0.80 and >= floor + 0.30 | binding |
| NOT-DECODER | acc K'0 d 4 | <= 0.20 | binding |
| NUM-SHORTCUT | answer-marginal / divisibility-aware marginal / last-number / in-question baselines | each <= 0.10 | binding (data gate) |
| PAUSE-SOLVES | pause arm d 1-4 | >= 0.95 -> C1 void at this rung | binding control |
| LATENT-SPECIFIC | text t7 / t8 vs lat | t7 <= lat7 - 0.10 and t8 <= lat8 - 0.10 | descriptive at 2 seeds, binding at 3 |
| C2-RETAIN | d 6 / d 7 | >= bun mean - 0.05 | descriptive (3 seeds for a claim) |
| C6 | K' 64 / 128 at d 1-4 | >= K' 8 - 0.02 | descriptive |
| EM-100 (C4) | ws_readout d 2 pass 1 at 1k / 2k; acc d 2 | >= 0.80; >= 0.90 | binding for the C4 cell only |
| DECON | decon.json share | < 0.01 | gate |

All bars on raw values with Wilson 95% CIs; a cell whose CI contains the bar is flagged **NEAR** and counted as not passed. Label **R0-PASS** = LEARN + MECH-WS + MECH-SWAP + NOT-DECODER + NUM-SHORTCUT in both seeds and not PAUSE-SOLVES.

**OC block (stated now).** P(LEARN) ~0.65 (bun's lineage formed on 11 values; the LM head already separates " 0".." 99", but the write must carry ~6.6 bits instead of 3.5). P(MECH | LEARN) ~0.7. P(PAUSE-SOLVES) ~0.15 (the chain needs a carried value, but a 100-value alphabet with small constants may let the decoder read the text). P(TEXT NOT-LATENT-SPECIFIC) ~0.4 (as §266). P(EM-100 pass) ~0.3 x P(§267 EM-C4). Misread zone: a true value within 1 Wilson half-width of a bar.

**Censoring.** A run that dies before the final step: VOID-RERUN once under a new name; logs kept. A text cell with parsed share < 0.9 at d <= 4 is a build bug (VOID-RERUN once). A generator gate failure (G2 < 1.0) blocks launch, it is not a result. A NEAR cell stays NEAR; no re-eval on more items (seed spread, not item SE, binds: Note 269-1). Missing fields are reported as missing, never defaulted.

## 8. Scorer sketch (theory/score_ladder.py, one scorer for all rungs)

Reads per cell dir: `eval_acc.json {d: {acc, n}}`, `eval_ws.json {d: {t: acc}}`, `eval_swap_k.json {d: {follow, floor, n}}`, `eval_k0.json {d: acc}` (NOT-DECODER), `eval_text.json {d: {acc, parsed, ntok}}`, `baselines.json {marginal, last_number, in_question}`, `decon.json {share}`, `args.json {num_p, val_fmt, pause_write, init_model}`; EM cells: §267's `eval_em.json` + `log.jsonl ws_readout`. Labels: R{k}-PASS / FAIL:<criteria> / PAUSE-SOLVES / LATENT-SPECIFIC / NOT-LATENT-SPECIFIC / UNINFORMATIVE / NEAR. Reuses score_em.py's wilson / near (theory/score_em.py:48-56). Fixtures (10): pass; NEAR on LEARN; MECH-WS fail; PAUSE-SOLVES; text parsed < 0.9 -> VOID; missing eval_ws -> missing; decon share 0.02 -> gate; EM onset fail; 3-seed C2 margin; args flags mismatch -> refuse. Eval parts reused from inject_aw_eval.py: part_acc (:104), part_ws (:352), part_swap_k (:255), part_swap_1 (:271), part_lp5 (:118), part_test_perm (:1044), part_transfer (:938), part_frames (:966); new: part_k0, part_digits (W-token decode), part_baselines.

## 9. Cost (RTX 3090, ~$0.2/h)

bun-style 8k run ~$0.8-1.2 (R0 token lengths equal mod-11's; R1 adds 2 tokens per value, ~+10%); text 24k ~$1.2; pause ~$1; direct ~$0.3; EM-100 learnability pilot at bs 16, d_max 2, 2k steps (101 upper-half forwards per item, micro-chunked) ~$1.5; evals ~$0.3 per cell. Totals per week in §6: ~$7.6 / $8.5 / $7.5, each under $30 with guard; the whole ladder to R8 ~$30-35 plus R9 $25-45 (§12.6's $45-70 stands).

## 10. Verified arXiv ids (abs pages fetched 11-03, titles matched)

2110.14168 GSM8K (Cobbe); 2311.01460 implicit CoT / GSM8K-Aug (Deng); 2502.21074 CODI (Shen); 2506.18582 PCCoT (Wu); 2509.20317 SIM-CoT (Wei); 2412.06769 Coconut (Hao); 2407.20311 iGSM (Ye); 2312.09241 TinyGSM (Liu); 2410.05229 GSM-Symbolic (Mirzadeh); 2411.18104 TemplateGSM (Zhang); 2402.19255 GSM-Plus (Li); 2405.00332 is GSM1k, "A Careful Examination of LLM Performance on Grade School Arithmetic" (H. Zhang), the contamination-vs-generalisation reference, **not GSM-Plus as §12.6 has it**; 2309.12284 MetaMath (Yu); 2211.10435 PAL (Gao); 2005.14165 GPT-3, the 13-gram filter (Brown); 2402.14903 Tokenization counts, R2L (Singh); 2405.17399 Abacus embeddings (McLeish); 2307.03381 Teaching arithmetic to small transformers, reversed digits (Lee); 2410.21272 bag of heuristics (Nikankin); 2502.09741 FoNE (Zhou); 2406.03445 Fourier features for addition (Zhou); 2510.00184 multiplication long-range dependency, running-sum aux (Bai); 2302.04761 Toolformer (Schick); 2505.09388 Qwen3 (Yang); 2211.12588 Program of Thoughts (Chen); 2305.15408 Feng et al. (the mod-11 generator); 2604.04902 latent models skip the latents (Dilgren); 2505.18962 System-1.5 (Wang). **Correction to §12.6:** the 13-gram decontamination is GPT-3's (2005.14165), not PAL's (2211.10435).

## 11. RED TEAM

**11a. Lead's own red-team pass (fallback, written while the sub-agent slot was unavailable; items adopted into the plan are marked ADOPTED).**

1. **Backward-uniform draws leak the op from the answer.** A mul pre-image needs c | S_t, so a prime answer > 9 means the last op was add/sub; chains ending in mul over-represent composite answers. A decoder can read op-by-divisibility priors. ADOPTED: Stage 0 reports P(op_d | answer) and the "divisibility-aware marginal" baseline; the test set is drawn **both** backward-uniform and forward-with-rejection (nested_words v1.2's rule), and LEARN must hold on both (a distribution-shift check for free).
2. **"One change per rung" is false as written:** rung k inits from rung k-1's s1, so lineage and alphabet both move relative to bun. ADOPTED: each rung carries a from-bun_s1 init cell beside the from-previous-rung cell, and the registration lists the args diff (the single-change-args-diff rule).
3. **Digit order.** Standard-order decoding puts the carry against the generation direction; reversed (LSD-first) digits are the known fix (2307.03381). ADOPTED: `--val_fmt digits_r` (LSD-first, fixed width) is the primary R1 format; standard order is a descriptive arm; NR5's carry-stratified read separates carry-free from carry items.
4. **Zero-padding costs tokens in the text / pause arms** (every value is W tokens there too). ADOPTED: FLOPs are counted on generated tokens per arm; the C3 row is at matched FLOPs, not matched steps.
5. **NUM_IDS is an import-time constant** (nested170.py:20) consumed at inject_ow.py:251 / 278 and nested170.py:407; inject_ow.py:43 asserts only " 0".." 10". `--num_p` must rebind it before the first use, and the P = 11 path must stay bitwise (G1). ADOPTED as a named touch point.
6. **EM-100 memory and cost.** 100x tiling at bs 64 is 6,400 upper-half forwards per step; the $3 figure assumes micro chunks and 4k steps. ADOPTED: EM-100 is a learnability pilot at bs 16, d_max 2, 2k steps (~$1.5), and the alphabet step is 11 -> 23 (iGSM's modulus, R4) before 100 for the C4 lane.
7. **No learnability pilot** (process rule 5). ADOPTED: every rung starts with a 2k-step DIRECT + LAT pilot ($0.3) whose PARK rule is d 2 < 0.5 at 2k in both arms.
8. **$0 falsifier first:** read bun_s1's ':' row on d 1 items whose values are 11..99 (the LM head already knows the tokens; does the write separate them?) via part_probe (inject_aw_eval.py:313) before any R0 training ($0.1, cloud host, pre-stated). ADOPTED into W1 D-A.
9. **R7's Socratic alignment** is per solution line, not per `<<e_t>>`; the "count equals m" filter keeps a shallow subset. ADOPTED: report the kept share by m; use the alignment only where counts match, else the step cue is the equation's left side (descriptive cell).
10. **Licences:** GSM8K-Aug's 385k file and GSM-Plus have no visible licence (11-03); Literature confirms before R7 / any OOD eval is registered.

**11b. Separate sub-agent red team:** see below (appended when the sub-agent returns; if absent, the spawn failed on the 20-concurrent-sub-agent limit and the caller is told so).
