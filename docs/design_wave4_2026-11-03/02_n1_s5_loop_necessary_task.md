# N1 / TH6 / B4 — SHUFFLED-OBJECTS: the "loop is necessary" task (design, wave 4, 2026-11-03)

Lead: Fable design agent (read-only; no runs, no stored-output reads). Red team: §10.

## 1. Goal and backlog IDs

Backlog **N1** (§11; Lane D2), **TH6** (§13.3: only non-solvable value groups force the loop), **B4** (§12.8: N1 with baselines). Targets: a public small-benchmark number (BBH tracking_shuffled_objects), **C2** (train swaps m ≤ 4, test 5-12), **C1** at the claimed depth (SWAP at pass t, m = 8), and the paper's expressivity argument: a one-pass model is a fixed-depth TC⁰ circuit; following one pointer through transpositions of 5 objects is NC¹-complete (Barrington 1989: width-5 permutation branching programs = NC¹, and a BP follows exactly one pointer, so the single-holder chain is the NC¹-hard object; transposition words lose nothing since every permutation is ≤ 4 transpositions); S3 / S4 are solvable, so shortcuts exist (ACC⁰, Barrington–Thérien; 2210.10749 empirically). Prediction: DIRECT extrapolates farther at n 3 / 4 than at n 5; the loop does not care about n.

Caveat up front: for fixed m everything is finite, so "necessary" is a length-generalisation claim backed by the circuit-class contrast, not a theorem about m ≤ 12 (§13.0 already reframed mod-11 this way; N1 is the task where the contrast is at least real).

## 2. The task

**Generator** `boxes/so_words.py` (new; modelled on `boxes/ow_words.py` 108-188: `make` / `redraw` / `Stream` / `test_set` / `pair_candidates` / `solve` / `set_sha`; excl protocol of `inject_ow.build_sets` 127-186).

Draw n ∈ {3, 4, 5} people from `NW.NAMES` (40 single-token names, nested_words.py 33-34), a scenario family f ∈ {balls, books, presents, partners, positions} (BBH's five), n distinct objects O_f (colour words for balls / presents; titles, positions, partner names otherwise — multi-token allowed, objects occur only in text and cue), an initial bijection h_0: O → P, m swaps (a_t, b_t), a_t ≠ b_t uniform over pairs, and one asked object o*. The state is o*'s holder:

    S_0 = h_0(o*);  S_t = b_t if S_{t-1} = a_t;  a_t if S_{t-1} = b_t;  else S_{t-1}   (t = 1..m)

answer = S_m (a person token); `values` = [S_0..S_m] as NAMES indices; `k_touch` = #{t : S_t ≠ S_{t-1}} (P(touch) = 2/n); `d` = m. An independent regex `solve(text)` checks every item (ow_words.py 178-184).

**Text**, two render modes (50 / 50 in training; swaps in chain order):

- NUM: "Ann, Bob, Kim, Leo and Mia are playing a game. At the start each holds one ball: Ann has the red ball, Bob the blue ball, Kim the green ball, Leo the white ball and Mia the black ball. Swap 1: Kim and Ann trade balls. Swap 2: Leo and Mia trade balls. Swap 3: Ann and Bob trade balls. Swap 4: Leo and Kim trade balls. At the end, who has the red ball?"
- ORD (BBH-shaped): "First, … Then, … Then, … Finally, …" — no numbers; locating swap t needs counting.

Frames: ≥ 4 paraphrases per family for setup and swap ("trade / swap / exchange / switch"), none sharing a 13-gram with any of the 750 BBH inputs (a $0 decontamination script runs at build, part of G0). NAMES exclude BBH's Alice / Claire / Dave / Eve / Fred / Gertrude, so BBH is also a name transfer.

**Per-pass aux target** (hand-built, as bun; C4 not claimed): `R_step[t]` = "After swap t, <S_t> has the <o*>." in `tok_ow`'s form (inject_ow.py 63-71), `vidx` = the holder token; `aux_logits` (nested170.py 260) unchanged.

**Cues** (`tok_so` in a new `boxes/inject_so.py` importing inject_aw / inject_st byte-identical, the inject_st → inject_aw precedent; `tok_st`'s G-CUE "no digit, 7 tokens" assert, inject_st.py 57-58, does not apply):

- **SI (step-index, main):** pass t input = [EOT] + enc(" " + text + " After swap {min(t, m)}, who has the {o*}? The answer is:"). It names an index, not an entity: no entity plan; the model locates swap min(t, m) itself (NUM: number-token match; ORD: counting). Passes t > m repeat the swap-m cue — bun's hold semantics (`hold_step`, inject_aw.py 159-200, reads xs[-1]; unchanged).
- **EN (entity cue; plan-given comparator, 1 seed):** " After {a_t} and {b_t} trade, who has the {o*}? The answer is:" — the analogue of bun's `cue_str` naming S_t (inject_st.py 44-46).
- Write source = the ':' token (`InjectAW.pass_aw`, inject_aw.py 47-59); decoder [EOT] + text + " The answer is:" → holder.

**Value vocabulary.** `NUM_IDS` (nested170.py 20) becomes a per-task `VAL_IDS` = the 40 name tokens; `ans` indexes NAMES; eval argmax is restricted to the item's own `people_ids`; training CE stays 40-way.

**Splits.** Train: n ~ U{3,4,5}, m ~ U{1..4}, NUM / ORD, all families, frames 0-3. TEST (held out by `sig`): 500 items per cell (n, m, mode), n ∈ {3,4,5}, m ∈ {1..12}; frames 4-5 = TRANSFER. Strata recorded: k_touch, S_m = S_0. BBH: the 3 × 250 public items verbatim (three / five / seven objects, m = n; repo licence MIT, LICENSE fetched 2026-11-03).

**Recipe** = bun_s1's frozen args (`boxes/runs_fsq/bun_s1/args.json`: 8k, bs 64, lr 1e-4, aux_step 1, aux_fade 48000, aux_reweight, cache_lower, write_src answer, hold_inc 1.6, hold_start 500, hold_far 16 / 16-40, init_trunk s233_askname) with `--task so --cue index --think_k 12 --d_max 4 --distract_p 0` (the untouched swaps are the holds; no insert mix) and **init_model = bun_s1/model.pt** — the formed loop. Note 269-1: formation is invisible at 2k and the lineage took 68k steps, so a fresh-trunk start is a formation experiment; one FROM-TRUNK arm is kept, descriptive. K = 12 keeps test m ≤ 12 inside the trained pass count (§259 scope). `AUX_W` (nested170.py 370) has 8 entries but is indexed only for t ≤ d ≤ 4 (the `idx` filter, inject_aw.py 395): no change, `--d_max 4` must stay. With 8-11 hold passes per item (bun: 4-7) `hold_inc` is re-calibrated once from the pilot's `hold_log` so the penalty's loss share matches bun's; declared before S2.

## 3. Arms and stages

| arm | init | what | seeds | $ |
|---|---|---|---|---|
| LOOP-SI | bun_s1 | the design above | 3 | 3 × 0.6 |
| LOOP-EN | bun_s1 | entity cue (plan given) | 1 | 0.6 |
| DIRECT | s233 trunk | one pass: text + question + " The answer is:", same 8k / data | 3 | 3 × 0.25 |
| PAUSE-12 | s233 trunk | DIRECT + 12 learned pause tokens before the answer (2310.02226; 2404.15758), mean-embedding init + attention-mass usage monitor | 1 | 0.3 |
| TEXT-CoT | s233 trunk | value-only chain "1 Bob 2 Bob 3 Kim … = Kim" via the §266 X3 trainer on the same items; greedy at test | 1 | 0.5 |
| BETA-POS | bun_s1 | LOOP-SI with the delta gate clamped to (0, 1), `--beta_pos` (§4) | 1 | 0.6 |
| FROM-TRUNK | s233 trunk | LOOP-SI from the trunk, fresh DeltaMem | 1 | 0.6 |

Matched compute = same items, steps, batch, base. Serial depth at test: loop and text CoT m × 12 layers, DIRECT / PAUSE 12. FLOPs: the loop re-reads the prompt each pass vs text CoT's cached L + m tokens — the C3 deficit of §12.1, stated. Coconut is skipped (§119: Coconut / CODI / SIM-CoT at chance on S5).

**S1** = the learnability pilot (LOOP-SI seed 1, `--stop_at 4000`, $0.3) **plus DIRECT × 3** ($0.75), together. Pilot gate: GO if NUM n 5, m ≤ 4 ≥ 0.90 and m 5 MOVED ≥ 0.50 (Wilson lower bound above `floor_hold`); PARK if m ≤ 4 < 0.50 (switch to EN before spending more); else NOTE. DIRECT gate: if DIRECT NUM n 5 MOVED m 8 ≥ 0.80 in 2/3 seeds, the "necessary" framing is dead at m ≤ 12 and S2 shrinks to LOOP-SI × 3 + TEXT-CoT (C2 / BBH only). **S2** = the rest. `eval_parts` = acc_so, swap_so, ws_so, long_k64, bbh, beta_rec only (the OW parts lp5h / cue_only / probe read `names.spine`).

## 4. Code touch points (read)

- `boxes/ow_words.py` 37-188; `boxes/nested_words.py` 33-36, 83-85 (NAMES, GIVEN, QUESTION).
- `boxes/inject_st.py` 44-63 (`cue_str`, `tok_st`, G-CUE), 70-80 (`cue_inputs`: all pass inputs share one padding — holds: " 1".." 12" are single tokens and the object phrase is fixed per item).
- `boxes/inject_aw.py` 45-63 (`pass_aw` / `colon_logits`), 76-97 (`tok_aw`; the §262 name-token assert 91-92 is `cue == "step"`-only), 138 (`ws_readout` NUM_IDS), 159-200 (`hold_step`), 233 (`--cue` choices + index / entity), 361-420 (`loss_reg`: NUM_IDS at 363 / 385 / 416 → VAL_IDS), 453-461 (`mem_train` / `mem_eval`).
- `boxes/inject_aw_eval.py` 33 / 41 / 321 / 670 / 768-777 (every NUM_IDS site), 104-116 (`part_acc` → `acc_so` by (n, m, mode, k_touch)), 352-357 (`part_ws` → `ws_so`), 365-433 (`swap_pairs_d` / `part_swap_kd` → `swap_so`, §6).
- `boxes/deltamem.py` 5, 98, 188: `be = 2 * torch.sigmoid(...)` — the gated-delta transition α(I − β k kᵀ) already has eigenvalue α(1 − β) ∈ (−1, 1) along k: the negative range of 2411.12537 is in place. `--beta_pos` replaces the factor 2 by 1 in both lines behind the flag; `alpha_rec` (77, 101-102) gets a sibling `beta_rec`. Honest framing: 2411.12537's condition is for a *linear* recurrence; here the trunk computes the transition and the memory only needs clean overwrite (β ≈ 1), so E1 is a monitor plus one falsifier arm, not a hypothesis the design rests on.
- `boxes/s5.py`, `boxes/deltas5.py` headers, THEORY §162: DeltaMem-R on symbolic S5 (whole-arrangement state): exact in range, 0.0 at 512, BUFFER-LIKE — a prior for "the memory buffers", not a replication (N1: single-pointer state, one ':' write per pass, hold penalty).
- `theory/score_bu.py` 1-30 — scorer conventions.

## 5. Gates before launch

- **G0 generator:** `solve(text) == answer` on 20k items; k_touch histogram vs the 2/n law (χ² p > 0.01); NUM / ORD of one item equal as sentence multisets modulo markers; 13-gram decontamination vs BBH = 0 hits; the BBH swap-sentence parser and the first-BPE-token distinctness of the ≤ 7 BBH names asserted on all 750 items; set shas recorded; excl ∩ train = ∅.
- **G1 equivalence (p = 0):** `inject_so --task ow` with bun_s1's args reproduces inject_aw's `format` string byte-identically and the first 50 CPU fp32 step losses bitwise. `--beta_pos` off: deltamem byte-identical (module sha in args).
- **G2 cue build:** every SI cue ends in ':'; its number token == enc(" %d" % min(t, m)); the shared-padding assert passes at n 5, m 12, K' 12.
- **G3 gradflow:** `gradflow_check` on one LOOP-SI micro-batch — non-zero grads at DeltaMem `qkv` / `gate` / `wo` and the trunk from both the answer CE and the aux CE (no new loss terms; the existing check on the new inputs).
- **G4 mem_check:** training n 5, m 4, K 12, micro 2, bs 64 ≤ 20 GiB (1 GiB headroom, micro pinned); eval n 5, m 12, K' 12, eval_bs 250 (fallback 125).
- **G5 smoke:** the plain path (`--task so`, no flags) runs 20 CPU steps and every eval part on 8 items with fixtures.
- **Zero-init:** none new except the pause tokens (usage monitor, rule 8).

## 6. Criteria (per seed unless stated; Wilson 95% CIs; NEAR = CI crosses the bar)

Floors fixed from the test set at build ($0, pre-stated): `floor_hold(n, m)` = share with S_m = S_0; `floor_major` = 1/n. Binding cells use the **MOVED stratum**: S_m ≠ S_0 and k_touch ≥ 2 (the pointer moved and did not merely return), which defeats "answer the initial holder" and "answer the other party of the one touching swap". X1 is read only with X2: DIRECT also meets number tokens 5-12 for the first time at test, so a DIRECT failure at n 5 alone is uninformative; the n 3 vs n 5 asymmetry shares that shift and is the expressivity reading.

| # | criterion | cell | bar | type |
|---|---|---|---|---|
| L1 | in-range learnability | LOOP-SI NUM, n 5, m 1-4 | ≥ 0.95 each m | binding |
| L2 | ORD learnability | LOOP-SI ORD, n 5, m 1-4 | ≥ 0.90 | binding |
| C2-1 | depth extrapolation | LOOP-SI NUM, n 5, MOVED, m 6 / 8 | ≥ 0.85 / ≥ 0.80 in ≥ 2/3 seeds; per-seed margins | binding |
| C2-2 | depth curve m 9-12; C2-3 ORD n 5 m 8 ≥ 0.60 | | | descriptive |
| X1 | one pass fails at n 5 | DIRECT NUM, n 5, MOVED, m 8 | ≤ 0.50 in 3/3 | binding (with X2) |
| X2 | solvable-group asymmetry | DIRECT NUM, MOVED, m 8: acc(n 3) − acc(n 5) | ≥ 0.15 in ≥ 2/3 seeds | binding |
| X3 | pause ≠ loop | PAUSE-12 NUM, n 5, MOVED, m 8 | ≤ DIRECT + 0.10 | binding |
| X4 | text CoT reference | TEXT-CoT NUM, n 5, m 8 | reported; C3 row: loop ≥ CoT − 0.05 | descriptive |
| M1 | mechanism at depth (WS) | LOOP-SI n 5, m 8: ':' argmax over people_ids at pass t = S_t | passes 5-8 ≥ 0.90 | binding |
| M2 | SWAP-MID at depth | n 5, m 8, t 4: S_4 := donor's | follow_cf ≥ 0.50 and ≥ MEAN floor + 0.30 | binding |
| M3 | SWAP-MID in range | m 4, t 2 | same bars; precondition for reading C2-1 | binding |
| M4 | NOT-DECODER | n 5, m 4, acc at K' 2 | ≤ floor(S_2 = S_4) + 0.05 | binding |
| M5 | strata | LOOP-SI m 8: acc(k_touch ≥ 6) − acc(k_touch ≤ 2) | ≥ −0.10 | binding |
| H1 | hold (C6) | n 5, m 1-4, K' 64 | ≥ acc(K' 12) − 0.02 | binding |
| B1 | BBH five_objects | 250 items, option-wise (§7) | target ≥ 0.60 (random 0.20); CI reported | descriptive → binding if L2 passes |
| B2 | BBH three / seven (seven: n 7 OOD) | 250 each | reported | descriptive |
| E1 | eigenvalue use | share of writes with β > 1 on m 5-12; BETA-POS vs LOOP-SI at C2-1 | within 0.05 → "sign range not load-bearing" | descriptive |
| P1 / F1 | LOOP-EN vs LOOP-SI at C2-1; FROM-TRUNK at L1 | | reported | descriptive |

Label: **N1-PASS** = L1, L2, C2-1, X1, X2, M1-M5, H1 all pass (≥ 2/3 seeds where seeded); **N1-LOOP-ONLY** = L1 + C2-1 + M1-M5 pass but X2 fails (the loop extrapolates; the asymmetry is invisible at m ≤ 12); else FAIL with the failing rows. A loop failing M3 while passing L1 is a decoder, and C2-1 is not read. BBH never gates the label.

**OC block (priors):** P(L1) 0.60 (new task on a formed loop; the unseen-number lookup is the main risk; the pilot catches it at $0.3). P(C2-1 | L1) 0.35. P(X1) 0.80. P(X2) 0.45 (m ≤ 12 may be too short for the asymmetry; TH6 predicts it, 2210.10749 makes it plausible). P(M2 | C2-1) 0.55. P(B1 ≥ 0.60) 0.25 (ORD + name + frame transfer). P(E1 within 0.05) 0.70. Value if N1-PASS: the first public depth split with a C1 battery on a pretrained LM (B4), on the one task where "loop needed" has circuit-class backing.

**Censoring:** VOID-RERUN once per cell for non-finite loss, OOM, host loss or a drain fault (sha-verified ckpts). A seed failing L1 is a FAIL row, not censored. S1 PARK voids S2. Cells are read only after the scorer and fixtures are committed; BBH per-item outputs are read by the scorer only (grep-only blinding on logs).

## 7. Scorer sketch (`theory/score_n1.py`)

Reads per run: `eval_acc_so.json` by_cell[n][m][mode] {acc, n_items, by_k_touch, floor_hold, items{pred, ans, k_touch, s0_eq_sm}}; `eval_ws_so.json` by_m["8"].passes[t]; `eval_swap_so.json` by_m[m].t {n_pairs, follow_cf, floor, stay_own, intact_acc}; `eval_long_k64.json`; `eval_bbh.json` {subset: {acc, n, per_item{pred_option, target, margin}}}; `eval_beta.json` {share_gt1_by_m, mean_beta}; DIRECT / PAUSE / CoT: `eval_acc_so.json` only. Labels as §6; per-seed rows; Wilson CI + NEAR per cell; `--fixtures` with 8 synthetic eval sets (pass, each single-row failure, NEAR, censored): 8/8 before launch.

**BBH protocol (pre-stated):** for "X has the ___ / Options (A)…": run the loop once per option o_j with the SI cue "After swap t, who has the <o_j>?" (family verb: has / is dancing with / is playing); answer = argmax_j of the ':' logit of X's first BPE token at pass m; m = the swap-sentence count. One rule, no ties; n loop runs per item, reported as compute.

**SWAP pairs law (`swap_so`):** donor = the recipient's text with a different asked object o' (same swaps and people, so token lengths are equal and cues differ only in the object); kept iff own ≠ cf(t) ≠ donor-final, cf(t) = the recipient's swaps t+1..m applied to the donor's S_t (cf_a = cf_b: a strict chain). Floor = the donors' mean S_t (as part_swap_kd). 1,000 pairs per cell.

## 8. Cost (RTX 3090, $0.2/h)

bun_s1's hostprobe is 0.62 s/step at K 8 on ~70-token items; N1 items are 60-110 tokens at K 12 → ~1.0 s/step → 8k ≈ 2.3 h ≈ $0.45, plus eval (36 cells × 500 × K' 12, swaps, K' 64 hold, 750 BBH × ≤ 7 options × K' 7) ≈ 0.7 h ≈ $0.15: **$0.6 per loop run**. S1 $1.05; S2 3 × 0.6 + 0.6 + 0.6 + 0.6 + 0.3 + 0.5 = $4.4; with one VOID-RERUN allowance **≈ $5.5, guard $7**, inside the $30 auto-approval.

## 9. Verified arXiv ids (titles fetched 2026-11-03)

2210.09261 Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them · 2411.12537 Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues · 2502.10297 DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products · 2404.08819 The Illusion of State in State-Space Models · 2210.10749 Transformers Learn Shortcuts to Automata · 2402.12875 Chain of Thought Empowers Transformers to Solve Inherently Serial Problems · 2412.06769 Training Large Language Models to Reason in a Continuous Latent Space · 2310.02226 Think before you speak: Training Language Models With Pause Tokens · 2404.15758 Let's Think Dot by Dot: Hidden Computation in Transformer Language Models · 2305.02363 Entity Tracking in Language Models · 2502.21074 CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation. Non-arXiv: Barrington, JCSS 38(1), 1989.

## 10. RED TEAM

**Process note.** The rules require a separate red-team sub-agent. Every Agent spawn in this session was refused ("concurrent subagent limit reached, 20; do not retry"), so this is a self-run red team by the lead, written after the draft against the required checklist. Audit should treat it as unreviewed and may commission the independent pass before registration. ✓ = folded into §2-§7.

**Confounds.** (1) ✓ DIRECT meets number tokens 5-12 only at test, so X1 alone cannot separate "TC⁰ cannot" from "unseen token": X1 is read with X2 (shared shift). Cleaner: a NUM arm whose training swaps are numbered from a random offset window within 1-12 — an S2 add-on ($0.6) if X2 is NEAR. (2) BBH's "X has the ___" is the inverse of the pointer chain; the option-wise protocol converts it into n forward chains — fair, n× compute per item, reported. (3) m ≤ 4 is DIRECT-solvable, so the loop may recompute S_t from text every pass and ignore the memory; only M3 / M4 and the OOD cells see this. ✓ M3 is a precondition for reading C2-1. (4) ✓ MOVED now requires S_m ≠ S_0 and k_touch ≥ 2. (5) ✓ hold-pass ratio vs bun: one declared hold_inc re-calibration.

**Shortcuts.** Initial holder — MOVED; "the other party of the last swap naming the initial holder" — right iff k_touch = 1, excluded by k_touch ≥ 2, reported by M5; parity-like shortcuts at n 3 — the X2 prediction; aux decodability from text at m ≤ 4 — confound 3, M1 binds at m 8.

**Bugs / code.** ✓ NUM_IDS sites inject_aw.py 138 and inject_aw_eval.py 33 / 41 / 321 / 670 / 768-777 were missing at first; the shared-padding assert holds (single-token numbers, fixed object phrase); ✓ OW-only eval parts excluded; `AUX_W` safe only with `--d_max 4`; ✓ BBH first-BPE-token distinctness in G0; eval memory at K' 12 on 110-token items: G4, fallback eval_bs 125.

**Cheaper falsifiers.** ✓ DIRECT × 3 moved into S1 ($0.75): if DIRECT extrapolates at n 5, the framing dies before any loop run. A $0 step-0 eval of frozen bun_s1 under the SI cue is a chance reference only. G5 smokes the SI lookup on CPU.

**Statistics.** 500 items: Wilson half-width ≈ 0.035 at p 0.8, under the 0.15 X2 bar; seed spread binds (Note 269-1, σ 0.09): 3 seeds, per-seed margins. X2 is unpaired, per DIRECT seed, 2/3. BBH 250 items: half-width ≈ 0.06; descriptive.

**Process rules.** Shortcut exclusion: DIRECT, PAUSE-12, MOVED, M4 / M5. Mechanism at depth: M1 / M2 at m 8. p = 0: G1 for `--task ow` and `--beta_pos`. Zero-init: pause tokens only. ≥ 3 seeds: LOOP-SI, DIRECT. Wilson + NEAR everywhere binding. No stored-output numbers; floors come from the new test set, pre-stated.

**Verdict (self):** GO-with-fixes (folded); open risk = confound 1, settled by the pilot and the offset-window arm.
