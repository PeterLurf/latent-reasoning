# MT1 JOINT-4 / MT4 / MT2: one bun loop on four families, no task flag (wave-4 design, 2026-11-03)

Lead: multitask design agent. DRAFT for Theory / Audit; nothing built or run, no stored output read.

## 1. Goal and backlog IDs

**Question.** Does the bun loop (§259: GPT-2 small + DeltaMem, K 8, per-pass cue, per-pass aux, hold penalty) serve four structurally different serial families with one set of weights and no task token, keeping its WP mechanism gates, and is the memory's use shared or partitioned across families?

- **MT1 JOINT-4** (backlog §13.2 / §13.3): word programs (WP, bun's REG stream) + shuffled objects (SO) + ProsQA (PQ) + kinship-lite (KIN); per-task vs shared cue; per-family mechanism gates; gradient cosines.
- **MT4** cross-task state swap + head-necessity matrix + task-id probe (eval-only, after MT1).
- **MT2** leave-one-family-out: this document delivers its two missing generators; MT2 itself is stage 3, conditional.
- **§174** (THEORY.md:11142, registered 2026-10-09, never launched: INJECT on multiplication + nested) is absorbed; its MATCH / BUDGET-vs-INTERFERENCE reading, cross-task swap with Audit's length-matched null, task probe and `grad_cos` are carried onto bun's harness.

## 2. Design

### 2.1 Common raw item format

`tok_ow` (boxes/inject_ow.py:63-71) reads `text, R_step, answer, d, sig`; `tok_st` (inject_st.py:49-63) reads `d, text` and `cue_str` = QUESTION(unit, names.spine[c]); `tok_aw` (inject_aw.py:76-93) asserts the cue's name token 6 before ':'; `ws_readout` (inject_aw.py:146) reads `raw.values[min(t, d)]`; `swap_pairs_d` / `redraw` need `ops, src_j, names, unit, frames, order, n_distractors`. Every family emits one dict with these keys plus `family`, `value_toks`, `value_cands`, `answer_tok`, `cues`:

| key | WP (OW.Stream, unchanged) | SO | KIN | PQ |
|---|---|---|---|---|
| `d` (train) | 1-4 | swaps 1-4 | hops 1-4 | gold path 3-4 |
| `names.spine` | S_0..S_d | [P] + swap ids | [P_0, slot_1..d] | [root] + path |
| `values[t]` | int 0..10 | P's object after swap t | person after hop t | concept after hop t |
| `value_cands` | NUM_IDS | the item's n colours | the item's persons | {opt_true, opt_false} |
| `R_step[t]` (aux) | "<S_t> has <v_t> <u>." | full assignment after swap t | "<slot_t> is <v_t>." | "<root> is a <v_t>." |
| `cues[t]` (PT arm) | "How many <u> does <S_t> have?" | "What does <P> have after <A_t> and <B_t> swap?" | "Who is <slot_t>?" | target cue (2.3) |

All candidate words are single GPT-2 tokens (checked with tiktoken gpt2 today: 14 colours, 10 object nouns, 25 relation nouns, slot letters A-H, 39 single-token animal nouns for PQ's 38 concepts; the build asserts `len(ENC.encode(" " + w)) == 1`). The decoder stays `[EOT] + enc(" " + text) + " The answer is:"` with the full-vocabulary CE (inject_aw.py:363, 405); accuracy = argmax restricted to `value_cands` (`pred_of`, inject_aw_eval.py:40). For WP, `answer_tok = NUM_IDS[ans]`, `value_cands = NUM_IDS`: bitwise the current path.

### 2.2 The two new generators (boxes/so_words.py, boxes/kin_words.py; NW's Stream / test_set / set_sha / redraw / solve / selfcheck interface)

**SO (BBH tracking form; the non-solvable value group, backlog TH6 / §13.0).** n_obj in {4, 5} (p 0.5; n 3 cannot give 4 distinct pairs), persons from NAMES, one unit noun with n colours. Initial sentences "<A> has the <red> <u>." (n), then d swap sentences "<A> and <B> swap <u>s." in time order, every unordered pair used at most once (the cue names one sentence; n 4 allows d <= 6, test d 7-8 use n 5). The order is the plan: SO is order-bound by construction and never permuted. Question "What does <P> have at the end?"; at least one swap involves P. State = the permutation sigma_t; `values[t]` = P's object under sigma_t. Distractors: 0-2 swaps of a disjoint group with another unit at uniform slots. Aux = the full assignment "<A> has the <c1> <u>, <B> the <c2> <u> and ..." (a single-value aux under-specifies what the next swap needs); `vidx` = P's colour token, found by prefix encoding and asserted == `value_toks[t]` (inject_ow.py:71's `index(HAS_TOK) + 1` is wrong here). `redraw`: same persons, swaps, P, order; a deranged initial assignment (identical token length). `solve`: regex simulator. Note: at d <= 4 one pass can express the answer (§13.0); SO's claim is C1 (the pass-t write carries sigma_t) and length generalisation, not necessity.

**KIN (two-hop-per-step pointer chasing with named slots; tests value-keyed retrieval, backlog MC1 / MC2).** Persons from NAMES (6-10), relations REL8 = {mother, father, sister, brother, aunt, uncle, cousin, friend} (lite: an arbitrary partial function (Y, r) -> X, unique per key, no gender logic). Facts "<X> is the <r> of <Y>."; chain: P_0, then d definitions "Let <slot_t> be the <r_t> of <slot_{t-1}>." (slots A-H, slot_0 := P_0's name); question "Who is <slot_d>?". Step t = read the definition (keyed by the cue's slot), take v_{t-1} from memory, retrieve the fact keyed by (v_{t-1}, r_t): selection keyed by the memory's VALUE, which WP never requires. Chain facts sit in random order among 2-6 extra facts about the same persons (keys unique; subjects shared so name matching alone cannot pick a fact) plus 0-2 facts about unused names; definitions in chain order. `R_step[t]` = "<slot_t> is <v_t>."; `cands` = the item's persons. `redraw`: same persons / slots / relations / order, the table re-drawn so the chain resolves differently. `solve`: regex + lookup.

**PQ (p8_na as stored; depth 3-4 trained, 3-8 tested).** Train items relabelled per draw by §191 `relabel()` (prosqa168s2.py:77-99) onto ENT17 and a new single-token concept vocabulary CON1T (38 animal nouns; the CON38 assert at line 84 becomes an argument). `text` = premises + question; `values[t]` = gold node t (`node`, line 68); `R_step[t]` = "<root> is a <node_t>."; `answer_tok` = opt_true; `value_cands` = [opt_true, opt_false] (chance 0.5, prosqa168s2's `acc_def`). The test file is relabelled ONCE with a fixed rng (27420; premise and option order kept) through a `fixed=True` variant, since line 82's `_split == "train"` assert and the test-vocabulary asserts (295-297) are CON38-specific. EXCL = the relabel-invariant graph signature.

### 2.3 Cues (both arms derive the cue from the item text; no task token)

- **PT (per-task, binding):** WP = bun's step cue S_min(t, d); SO = "What does <P> have after <A_t> and <B_t> swap?" (never a colour); KIN = "Who is <slot_min(t, d)>?" (never a person); PQ = "Is <root> a <X> or a <Y>? What is the next value?" at every pass (PQ's plan-giving cue would contain the value, so PQ is plan-free). Within an item every pass has one cue length (cue_inputs' one-padding assert, inject_st.py:70-79, holds); lengths differ across families.
- **SH (shared, one seed, descriptive):** one template for all: " <question> What is the next value?" (the item's question + §252's GEN_CUE suffix, constant over passes; §261's target form generalised). Plan-free for WP / SO / KIN; its prior is tied to §261's unread pilot.
- G-CUE per family: no digits, no `value_cands` token in the cue (inject_st.py:57 generalised); PQ's cue names both options symmetrically, so its assert is "both present, neither alone". `CUE_NAME_OFF[family]` replaces the name-position assert at inject_aw.py:91-92.

### 2.4 Training recipe (bun's args; one change: `--families wp,so,kin,pq`)

- Init = bun_s<k>/model.pt (sha asserted), bun_s1's args.json exactly (K 8, d_max 4, lr 1e-4, bs 64, aux 1.0 reweighted, hold_inc 1.6, hold_start 500, distract mix p 1.0 **WP only**, mul_s_p 0, cue step, write_src answer, fast_add, cache_lower), 8k steps, RTX 3090. The loop exists for WP; the question is absorption of three families without a flag and without WP loss. (A from-scratch joint lineage is MT1b, ~$6, only if the warm start reads INTERFERENCE everywhere.)
- **Per-step family draw** (keeps the one-d-per-step law, inject_aw.py:552, and hold_step's assert at 168): `f = rng_f.choice(families)` uniform, then `d` from f's law (WP / SO / KIN U{1..4}; PQ U{3, 4}). Batches are single-family; shares logged per 1k bin (`family_n`).
- Per-family micro (`--micro_by_family pq:8`): PQ items are ~500 tokens (p8_na_train mean 219, max 357 words) against WP's ~100; mem_check runs per family on its longest item (the memcheck list at inject_aw.py:442-450 becomes per family).
- IR.distract / PR.train_perm / ask items (inject_aw.py:319-346) apply to WP only (SO / KIN carry their own distractors). WS_ITEMS (line 451): the first 200 TEST-f items of d 1..4 per family; `ws_readout` takes the argmax over `value_cands` against `value_toks[min(t, d)]`.
- **Gradient-cosine diagnostic** (every 500 steps, no optimiser step, four fixed probe batches of 32, one per family, seeded 27410+f, excluded from training; the plain `loss_items`, hold penalty off): g_f over all trainable parameters; `cos(g_f, g_g)` for the 6 pairs, split by module group {blocks 0-5, blocks 6-11, DeltaMem write, W_o read}, plus ||g_f||; to log.jsonl as `grad_cos`. Reading: CONFLICT(f, g) if the median full cosine over steps 4k-8k < -0.10; ORTHOGONAL if |cos| <= 0.10; ALIGNED above. Descriptive; routes MT3 (PCGrad 2001.06782) and the grouping read (2109.04617).

### 2.5 Cells and stages

| stage | cells | purpose | $ |
|---|---|---|---|
| 0 learnability + matched reference | sg_so_s1, sg_kin_s1, sg_pq_s1: bun_s1 + 2k steps on one family alone | process rule 5; the matched-examples reference for MATCH (each family gets ~2k steps inside JOINT) | 0.25 + 0.25 + 0.8 |
| 0 WP control | ctl_wp_s1: bun_s1 + 2k WP-only | WP matched-budget reference | 0.3 |
| 1 JOINT-PT | jt_s1, jt_s2 (8k) | binding | 2 x 1.2 |
| 2 JOINT-SH | js_s1 (8k), only if jt_s1 reads TASK-AGNOSTIC | cue contrast | 1.2 |
| MT4 | eval parts on jt_s* | swap / heads / probe | 0.3 |
| 3 MT2 (only on TASK-AGNOSTIC 2/2) | jt3_minus_f (8k, three families) + 1k adaptation on f vs bun + 1k on f, f in {so, kin} | zero-shot and sample-efficiency ratio | 2 x 1.5 |

Stage-0 GO per family: d 1-4 (PQ 3-4) accuracy >= 0.90 at K' 8 and WS-f >= 0.80 at 2k; a miss gets one 2k extension (budget before redesign, rule 3), then the family is dropped (JOINT-3, disclosed). Stage 1 launches with >= 2 new families GO.

### 2.6 Per-family mechanism gates (binding on jt_s*, at the trained depth 4)

The bun / §263 forms (theory/score_of.py:40-41; inject_aw_eval.py part_swap_kd:391, part_lp5h:510) on TEST-f (1,000 per d, fixed seeds, EXCL from training; PQ = p8_na_test by depth, n reported) with the family's `value_cands`:
- **ACC-f:** d 1-4 (PQ 3-4) >= 0.95 at K' 8.
- **WS-f:** trunk argmax over `value_cands` at ':' of pass t == `value_toks[t]`, t = 1..4 on d 4, each >= 0.90.
- **SWAP-f:** SWAP-MID at t 2 on d 4 family pairs (recipient / `redraw` donor; U = {cf_a(2), cf_b(2)} from the family's `solve` on the donor-prefix counterfactual): follow_U >= 0.50 and >= floor + 0.30 (floor = the donors' mean S_2).
- **LP5-f:** MEAN-AT-2 on d 4: intact - mean_at_t >= 0.30.
- **NOT-DECODER-f (SO, KIN):** acc at K' 2 on d 4 <= chance_f + 0.10 (SO 1/n pooled, KIN 1/|persons|); K' 0 by d reported.
- **THINK-GAIN-pq:** one-pass no-CoT GPT-2 already reads 0.63 at depth 3 on this data (THEORY.md:10439), so a chance-based decoder bar fails by construction. Bar: acc(K' 8) - acc(K' 0) >= 0.20 at depth 4, with the K' 0..14 curve (the §168s2 form).
- SO at n 5 reported separately (lower chance; S_5 is the non-solvable case).

## 3. Code touch points (read today)

- boxes/inject_aw.py: `tok_aw` 76-93 (family dispatch `tok_f`, same dict keys + `answer_tok`, `value_cands`); `draw_items` 316-346 (family gate on IR / PR; the `(f, d)` draw replaces line 552); `loss_reg` 363 (`tgt = answer_tok`) and 417 (per-item candidate argmax); memcheck 442-451; `ws_readout` 135-151; args 213-249 (`--families`, `--micro_by_family`, `--grad_cos_every`); the 1k bin 609-645 (`family_n`, `ws_readout_<f>`); `hold_step` 159-199 unchanged (xs[-1] is the last cue for every family).
- boxes/inject_st.py: `tok_st` 49-63 (takes `it["cues"]`; G-CUE against `value_cands`); `cue_inputs` 70-79 unchanged.
- boxes/inject_ow.py: `tok_ow` 63-71 (`vidx` from the family's value word); `build_sets` 127 (`sets[f]`).
- boxes/inject_aw_eval.py: `pred_of` 40, `part_acc` 104-116, `swap_pairs_d` 370 (OW-specific seeds -> `pairs_f`), `part_swap_kd` 391-433, `part_lp5h` 510-531, `part_ws_deep` 1111, `part_long` 534, `run` 1296-1349 (loops families, writes `eval_<part>_<f>.json`; WP outputs keep their names so score_bu / score_of read unchanged).
- boxes/prosqa168s2.py: `encode` 63-70, `relabel` 77-99, `load_train_raw` 102.
- New: boxes/so_words.py, boxes/kin_words.py, boxes/mt_parts.py (part_mt_swap / mt_heads / mt_probe), theory/score_mt.py.

## 4. Gates before launch

1. **p = 0 equivalence:** `--families wp` with `--grad_cos_every 0` bypasses the family draw entirely (so `rng_d` at inject_aw.py:271 is consumed identically) and reproduces bun_s1's first 50 steps bitwise (loss, gn_preclip, ihash); the family-looped eval parts on bun_s1 reproduce eval_acc / eval_swap_kd / eval_lp5h / eval_ws_deep byte-for-byte (fp32, same card, det wrap).
2. Generator self-checks: `--selfcheck` on 3,000 items per family (solve == answer; `redraw` pairs token-length identical; single-token vocabularies; no digit in any text; no `value_cands` token in any cue except PQ's symmetric pair; `aux[t][vidx[t]] == value_toks[t]`); PQ: graph signature invariant under relabel, CON1T coverage.
3. CPU smokes (`--steps 20 --sched_scale 0.002`): each family alone, the 4-mix, then the plain WP path (post-edit smoke rule).
4. gradflow (scratch/gradflow_check.py): per family the aux CE, decoder CE and hold penalty each reach DeltaMem's write parameters and the trunk; the grad_cos hook leaves `params.grad` None and no rng state changed (ihash equal with and without it).
5. mem_check per family at its longest item; PQ at micro 8, budget 21 GiB; a 100-step 4-family timing slice fixes the cost projection.
6. Family-share assert: every 1k bin's shares in [0.15, 0.35].
7. Audit's diff review; args_diff vs bun_s1 = {families, micro_by_family, grad_cos_every, init_model} (single-change rule).

## 5. Pre-registration table

Per seed (jt_s1, jt_s2); every criterion with a Wilson 95% CI and a NEAR flag when the bar lies inside the CI.

| # | criterion | type | definition / bar |
|---|---|---|---|
| B1 | RETAIN-WP | binding | TEST2 d 1-4 >= 0.98; NOT-DECODER (K' 2 at d 4 <= 0.20); SWAP4; LP5-D4; d 6 >= bun(same seed) - 0.05; C6 (K' 64 / 128 >= K' 8 - 0.02 at d 1-6) |
| B2 | MECH-f, f in {so, kin, pq} | binding | ACC-f, WS-f, SWAP-f, LP5-f, NOT-DECODER-f / THINK-GAIN-pq all pass |
| B3 | MATCH-f | binding | acc_f(d 4, K' 8) >= sg_f(2k) - 0.02 on the same 1,000 items (paired; McNemar p beside the CIs); a miss reads INTERFERENCE-f if < sg_f - 0.05, else NEAR-MATCH-f. Disclosed confound: sg_f's 2k cosine schedule vs JOINT's ~2k family-f steps spread over 8k |
| B4 | TASK-AGNOSTIC(seed) | label | B1, B2, B3 for every family that passed stage 0 |
| B5 | STABLE | binding | score_ao.stable |
| D1 | C2-f | descriptive | acc_f at d 5-8 (K' 8, 12); SO n 5 separately (no C2 claim: 2 seeds, one lineage) |
| D2 | CUE-CONTRAST | descriptive | js_s1 vs jt_s1 per family on B2's measures |
| D3 | GRAD-COS | descriptive | 6 pairwise cosines by module group; CONFLICT / ORTHOGONAL / ALIGNED |
| D4 | MT4 XSWAP(f -> g) | descriptive | S_2 of a family-f donor into a family-g d 4 recipient (t 2), recipient acc vs the in-task random-donor swap and the length-matched (+-2 tokens) in-task null; SHARED if within 0.05 of the in-task swap; PARTITIONED if below the length-matched null by >= 0.10 |
| D5 | MT4 HEADS | descriptive | 12 x 4 matrix: acc_f(d 4) with head h replaced at every pass by its per-family mean (pooled mean as a second condition); floor = a random 64-d direction of head h at the same norm; NECESSARY-f if the drop exceeds the floor's by >= 0.20; SHARED-HEAD if necessary for >= 3 families; overlap = Jaccard |
| D6 | MT4 TASK-ID | descriptive | logreg (nested170.logreg_acc) family from the flattened delta state after pass 1 (4,000 / 1,000 balanced), with a token-count baseline and the same probe on the un-adapted bun_s1 state |
| D7 | MT2 (stage 3) | descriptive | zero-shot acc_f of jt3_minus_f; adaptation ratio = steps to acc_f >= 0.90 from jt3_minus_f vs from bun (1k cap; CENSORED-AT-1k if neither reaches it) |

**Scope.** Plan given for WP / SO / KIN, PQ plan-free; per-pass aux (no C4); REG frames 0-5; one lineage (§268 pending); d <= 4 trained.

**OC block.** P(stage-0 GO): SO 0.6, KIN 0.4 (value-keyed retrieval is the untested mechanism), PQ 0.35 (K 8 on 3-4 hops over ~500-token premises from a WP-formed loop). P(TASK-AGNOSTIC per seed | all GO) 0.25; P(RETAIN-WP) 0.6 (the hold penalty and distractor training are WP-specific pressure); the likeliest failure is INTERFERENCE-kin. P(SH matches PT) 0.15 (tied to §261). MT4: P(PARTITIONED on >= 1 pair) 0.5; P(TASK-ID >= 0.95) 0.8 (expected, proves nothing alone). A null everywhere is informative: it bounds "one loop, many tasks" at this scale and routes MT1b / MT3.

**Censoring.** Crash before 8k: VOID-RERUN once from the last ckpt. A stage-0 family failing twice: DROPPED (label TASK-AGNOSTIC-3). PQ OOM at micro 8: premise count capped at the 90th percentile, disclosed. PQ test depths with n < 100: reported without a bar. STABLE false: FAIL, not VOID. No post-hoc bar changes; extra analyses labelled post-hoc.

## 6. Scorer sketch (theory/score_mt.py, fixtures before launch)

Reads per run: `eval_acc.json` (by_d.*.acc / acc_kprime), `eval_swap_kd.json` (by_d.4.swap_mid.4.follow_U / floor), `eval_lp5h.json` (by_d.4.intact.4 / mean_at_t.4), `eval_long_k128.json`, their `_<f>` siblings, `eval_ws_<f>.json` (by_d.4.{1..4}), `eval_mt_swap.json` (pairs.{f}->{g}.{acc_swap, acc_intask, acc_lenmatched}), `eval_mt_heads.json` (matrix[h][f], floor[h][f]), `eval_mt_probe.json` (acc, acc_len_baseline, acc_bun_init), `log.jsonl` (`grad_cos`, `family_n`), sg_f's `eval_acc_<f>.json`, bun_s*'s eval_acc. Labels: TASK-AGNOSTIC / FAIL: <criteria> per seed, k/2 per arm; MATCH / NEAR-MATCH / INTERFERENCE per family; XSWAP SHARED / PARTITIONED / MIXED per pair; HEADS overlap; GRAD-COS per pair. Fixtures (>= 10): pass, each binding miss, pending, NEAR, a dropped family, a censored PQ depth. Wilson CIs from theory/statlib.py if it exists, else inline.

## 7. Cost (RTX 3090, ~$0.2/h; bun 0.62 s/step at micro 2)

WP / SO / KIN steps ~0.65 s; PQ steps ~4-7 s (prosqa168s2's basis; no `--ckpt_passes` under `--cue step`, inject_aw.py:261). JOINT 8k: 6k x 0.65 s + 2k x 5 s ~ 3.9 h ~ $0.8-1.2 per seed. Stage 0 $1.6, stage 1 $2.4, stage 2 $1.2, MT4 $0.3, per-family evals $0.5: **~$6.0, guard $7**; MT2 +$3.0 if run. Local: CPU smokes and self-checks only.

## 8. RED-TEAM

**Process note.** The Agent pool was saturated (20/20 concurrent) at three attempts to spawn the required red-team sub-agent (the tool said not to retry). The list below is the lead's separate critical read of the v0 draft against the code; ADOPTED items were folded into sections 2-7. A sub-agent red-team is still owed before registration and should start from this list.

**BLOCKER (all adopted).** (1) v0's NOT-DECODER-pq <= 0.60 fails by construction: no-CoT GPT-2 reads 0.63 at depth 3 (THEORY.md:10439) -> THINK-GAIN-pq. (2) SO at n 3, d 4 must repeat a swap pair, so the cue "after <A> and <B> swap" names two sentences -> n in {4, 5}, unique pairs. (3) PQ's target cue names both options, which the generalised G-CUE forbids -> symmetric-presence assert. (4) `relabel()` refuses test items (prosqa168s2.py:82) and the test asserts are CON38-specific -> fixed-seed test relabel.

**MAJOR.** (5) Warm start from bun makes TASK-AGNOSTIC asymmetric (WP's 68k-step lineage vs ~2k steps per new family); INTERFERENCE-f can read as "too few steps". Kept: sg_f(2k) as the matched reference, one extension; MT1b named as the from-scratch follow-up. (6) Schedule mismatch in MATCH, disclosed (the `--stop_at 2000` alternative mismatches the other way). (7) KIN admits a textual two-lookup chain at d <= 2 inside one pass; NOT-DECODER-kin at K' 2 on d 4 covers the claimed depth, the K' 0 curve by d exposes a partial decoder path, SWAP-kin binds the mechanism. (8) KIN's candidates all appear in the text, so a "last-mentioned subject" readout beats chance; ACC >= 0.95 and WS >= 0.90 exceed any positional heuristic on random-order facts; descriptive add: WS split by whether v_t is the most recent subject. (9) The task-id probe is near-vacuous (any text-dependent write carries family identity) -> the un-adapted bun_s1 probe added. (10) Mean-ablating an accumulator head shifts the norm for every family alike and inflates "shared" -> the random-direction floor and per-family means added. (11) Two seeds, one lineage: k/2 labels, no C2 claim; MATCH's 0.02 bar at n 1,000 (Wilson half-width ~0.015) will often read NEAR -> paired items adopted.

**MINOR / cost.** (12) PQ dominates cost; if mem_check needs micro >= 16, PQ wall time doubles; the premise cap is the pre-stated fallback; JOINT-3 without PQ is the cheaper design, kept out because PQ is the one public family. (13) The p = 0 gate must bypass the family draw so `rng_d` is consumed identically (adopted in 4.1). (14) `tok_ow`'s `vidx` rule breaks on SO's aux (adopted: prefix encoding + selfcheck). (15) Cheapest falsifier: stage 0 alone ($0.5 for SO + KIN) answers whether the WP-formed loop absorbs any new family in 2k steps; two double failures make JOINT moot and route MT1b / LC4 STATE-DISTILL. Residual risk accepted: CON1T's animal priors ("Every cat is a dog") may slow PQ relative to nonsense names (no nonsense single tokens exist in GPT-2's BPE).

## 9. Verified arXiv ids (abs pages fetched 2026-11-03; titles matched)

1807.03819 Universal Transformers (Dehghani; joint bAbI, per-task ponder). 2209.11142 A Generalist Neural Algorithmic Learner (Ibarz; one processor, 30 algorithms). 2502.08482 Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning (Yu; RELAY). 2110.14056 How to transfer algorithmic reasoning knowledge to learn new algorithms? (Xhonneux; MT2's prior). 2604.07822 Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers (Kohli). 2001.06782 Gradient Surgery for Multi-Task Learning (Yu; PCGrad, MT3). 2109.04617 Efficiently Identifying Task Groupings for Multi-Task Learning (Fifty; gradient affinity, the grad-cos reading). 2210.09261 Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them (Suzgun; tracking shuffled objects). 2210.10749 Transformers Learn Shortcuts to Automata (Liu; S_n word problems). 1908.06177 CLUTRR (Sinha; kinship-lite's ancestor). 2412.06769 Training Large Language Models to Reason in a Continuous Latent Space (Hao; Coconut, ProsQA). 2502.05171 Scaling up Test-Time Compute with Latent Reasoning (Geiping; Huginn, task-dependent saturation). 2602.08332 Latent Reasoning with Supervised Thinking States (Amos; variable tracking).
