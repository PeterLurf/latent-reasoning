# Wave-4 design 20: BM1 PRONTOQA-DEPTH and BM2 WEB-OF-LIES-DEPTH on the bun recipe

Lead agent, 2026-11-03. Read-only; nothing here is registered or run. Covers backlog IDs **BM1**, **BM2** (§13.2 / §13.3 "public benchmark scout"), the headline-H2 pairing with **D1 PROSQA-OOD** (§12.8, Lane D), and B1 PAUSE-PASS / NO-WRITE as the shortcut-exclusion arm.

## 1. Goal

Two public depth-split benchmarks with public generators, run on the bun recipe (§259) with the same arms, gates and statistics as D1 PROSQA-OOD, so that headline H2 ("H1 + a public depth split") rests on three tasks: two with public generators, one (BM2) with 250 public test items (BBH). Both keep bun's restrictions 1-2 (a plan in the cue where the task has one; per-pass aux from the public proof trace, allowed under revised C4) and leave restriction 3 (the mod-11 template). Neither is a C4 route. BM2 is a Z_2 value chain (keep / negate), the closest public relative of the mod-11 loop; BM1 is a linear is-a chain whose per-step state is a concept name, the plan-free target-cue form (§261) that D1 also uses, so BM1 and D1 share one runner.

## 2. Data

### 2.1 BM1 PRONTOQA-DEPTH

Generator `asaparov/prontoqa` (Apache-2.0; `run_experiment.py`, fetched 2026-11-03). Verified options: `--model-name json` (writes examples, no model), `--ontology fictional`, `--deduction-rule ModusPonens`, `--distractors / --test-distractors {none, relevant, irrelevant}`, `--ordering / --test-ordering {postorder, preorder, random}`, `--min-hops / --max-hops / --hops-skip`, `--proof-width 2`, `--num-trials`, `--few-shot-examples`, `--seed` (default 62471893, seeded once per hop count), `--no-adjectives`, `--proofs-only`, `--OOD`. Output JSON: `example<i>.test_example` with `question` (context sentences), `query` (`"True or false: <entity> is <property>."`), `chain_of_thought`, `answer` (`True` / `False`; the conclusion is negated with p 0.5). Entities: Fae, Rex, Sally, Max, Alex, Sam, Polly, Stella, Wren; concepts are nonce words (wumpus, jompus); properties are adjectives.

Pre-stated generation (one command per hop count, `--few-shot-examples 0`, only `test_example` kept, ordering random, distractors relevant):

| split | hops | n | seed |
|---|---|---|---|
| train | 1-4 uniform | on demand, 20k per hop per chunk, dedupe on the sorted sentence set | 71001+ |
| ID test | 1-4 | 1,000 per hop | 71901 |
| **DEPTH test** | **6-10** | 1,000 per hop | 71910+h |
| ORDER test (descriptive) | 6, 8, postorder | 500 per hop | 71920+h |

Stage-0 gate: the generator must produce hops 9-10 under `--max-hops 10`; if it caps lower, the range is cut and written into the registration before any run.

Item format (new `boxes/bm_data.py`, the keys of `tok_aw`'s dict so the trainer loop is untouched): `text` = `question`; `think = [EOT] + enc(" " + text)`; `dec = [EOT] + enc(" " + text + " " + query) + SFX` (`SFX = enc(" The answer is:")`); `think_cues[t] = dec` for every t (the **target cue**: the query repeated, byte-identical per pass, plan-free, so `--cache_lower` hashes one input); state at hop t = the concept reached after t implications, read from `chain_of_thought` (sentence 2t+1 is "<entity> is a <C_t>."); `aux[t] = enc(" " + C_t) + [EOT]`, **sequence-scored** (the `[EOT] Step t :` aux decoder, `nested170.aux_prompt:228`, already scores token sequences); `ans` in {0, 1} indexing `ANS_IDS = [enc(" False")[0], enc(" True")[0]]`; `d` = hops; `sig` = the sorted sentence set.

A step-naming cue ("Every <C_{t-1}> is a what?") is excluded by design: with the concept named, C_t is a one-sentence lookup and the memory carries nothing. The target cue forces pass t to read C_{t-1} from memory, the forward-reachability form of §260.

**$0 tokenisation pre-check** (`bm/tok_check.py`, GPT-2 BPE via `boxes.nested170.ENC`): over 2,000 generated items, the single-token share and the distinct-first-token share of concept names, properties, entity names and answer tokens. Rule, fixed now: the single-token argmax readout (bun's `NUM_IDS` path) only if >= 95% of the state alphabet is single-token with distinct tokens; otherwise sequence-scored. Spot-check done with tiktoken (a tokenizer property, not a stored-eval read): " True" / " False" / " Yes" / " No" / " truth" / " lies" are 1 token; nonce concepts are 2-3 tokens (" wumpus" 3, " dumpus" 2); adjectives mostly 1 (" feisty" 3); entities 1-2. So BM1's state readout is **sequence-scored**: the candidate log-prob margin over the item's concept pool (`prosqa168.candidate_margins` form), and WS-READOUT at ':' is that margin's argmax.

**PROP-SHORTCUT.** The query names a property; if one context sentence alone mentions it, its polarity answers the item without any chain. Stage 0 counts this rate on the generator's output. The shortcut-exclusion split **PROP-FOIL** appends, in the generator's own template form, one sentence giving the queried property with the opposite polarity to an off-chain (distractor) concept, so a property lookup alone is at chance. The plain split is reported for comparability, PROP-FOIL for every C1 reading.

### 2.2 BM2 WEB-OF-LIES-DEPTH

Generator: BIG-bench `web_of_lies/task.py` (Apache-2.0; `WebOfLiesTask._get_expression(length)`, fetched 2026-11-03): 50 hard-coded first names, shuffled per item; first sentence `"{X_1} tells the truth."` / `"{X_1} lies."`, then `"{X_i} says {X_{i-1}} tells the truth."` / `"... lies."`, fair coins; the answer starts True and flips at each "lies"; question `" Does {X_length} tell the truth?"`; targets `["Yes", "No"]`; `shuffle_sentences` default False; seed 42; `num_sentences (1, 2, 4, 8)`, 400 trials each. BBH (`suzgunmirac/BIG-Bench-Hard`, MIT) ships 250 items of 5 people (1 root + 4 "says"), Yes / No.

Regeneration (`bm/gen_wol.py`, ~60 lines; v_1 = truth(X_1), v_t = v_{t-1} XOR says_lies(X_t)):

| split | d | distractors | order | n |
|---|---|---|---|---|
| train | 2-5 uniform | 0-3 off-chain statements (new names: a root "{Y} lies." plus "{Z} says {Y} tells the truth." chains of length 1-2, never touching the main chain) | chain | on demand |
| ID test | 2-5 | 0-3 | chain | 1,000 per d |
| **DEPTH test** | **6, 8, 10, 12, 16, 20** | 0-3 | chain | 1,000 per d |
| **BBH-250** | 5 | 0 | chain | 250, scored once, blinded |
| SHUFFLE (descriptive) | 5, 8 | 0-3 | shuffled | 500 per d |

Distractors are the shortcut-exclusion device: without them the answer is the parity of the word "lies" over the whole text (a count, no chain). Stage 0 reports the parity-shortcut agreement rate per split (BBH-250: 1.0 by construction, so BBH-250 is the public number and DEPTH-with-distractors the mechanism number).

Item format: cue at pass t names the step's person (the plan, as bun's cue names S_t): `think_cues[t] = [EOT] + enc(" " + text + " Does X_t tell the truth?") + SFX` for t <= d, the S_d cue after (bun's `spine[min(t, d)]` rule); `dec` = the d-th cue; `ans = v_d`; `aux[t] = [ANS_IDS[v_t], EOT]`, `ANS_IDS = [enc(" No")[0], enc(" Yes")[0]]` (single tokens: bun's `NUM_IDS` argmax path with P = 2). The §262 cue-name assert (`inject_aw.py:92`, name token 6 before ':') becomes a span assert (names are 1-3 tokens).

## 3. Model and training

The bun recipe: GPT-2 small; DeltaMem 6 + 6 heads at block 6, written from the ':' state (`InjectAW.pass_aw`, `inject_aw.py:47-56`); K = 8, `--cache_lower`, aux_step 1 reweighted, hold-increment penalty (`hold_step`, `inject_aw.py:159-199`: state norms only, no token alphabet, so it ports unchanged; lambda 1.6 from step 500). Loss = answer CE (full-vocab CE with the `ANS_IDS` target, the `inject_aw.py:363` path) + sequence aux CE + L_inc. Decoder: the final direct-format pass on `dec` reading S_K.

Lineage: bun's 68k-step lineage does not transfer; learnability pilot first (process rule 5). Stage 1a ONE-HOP (the askname analogue): 1k steps of plain GPT-2 fine-tuning on one-hop items (BM1 hop 1; BM2 d 1-2 with the person cue). Stage 1b: the loop on d ~ U{1..4} (BM1) / U{2..5} (BM2), 8k steps from that init, one seed. Formation markers read at >= 4k (Note 269-1 FD1): WS-READOUT at pass 2 on d 2 >= 0.6 by 6k, else STALL -> the staged P-2 -> REG recipe (LC5 form) before any redesign (rule 3).

Flags, in a new module `boxes/inject_bm.py` (a §234-style copy of `inject_aw.py` with asserted edits; `inject_aw.py` untouched): `--task {ow, prontoqa, wol}` (ow = bitwise `inject_aw`), `--bm_data <dir>`, `--prop_foil 0|1` (BM1 eval split), `--distract_nmax` reused for BM2's inserts. No new loss term and no new zero-init pathway (W_o = 0 is bun's, with its hold-stats monitor).

## 4. Code touch points (read)

- `boxes/inject_aw.py:76-93 tok_aw` (cue forms, SFX append, the ':' and name-token asserts); `:159-199 hold_step` (reads `xs[-1]` as the hold cue: BM1's identical cues and BM2's S_d cue both satisfy it); the `NUM_IDS` readers `:363` (answer CE target), `:378`, `:389` (the ':' argmax), `:417` (train acc), `:424` -> one `ANS_IDS` per task; `:213-249` args.
- `boxes/inject_st.py:44-63 cue_str / tok_st` (G-CUE: 7-token cue, no digits -> replaced by the BM cue builder); `:70-80 cue_inputs` (one padding per batch: holds, each item's cue length is fixed).
- `boxes/inject_ow.py:63-71 tok_ow` (the item dict), `:80-102 InjectOW.aux_logits` (sequence-scored aux).
- `boxes/inject_aw_eval.py:36 PARTS`; `:255-267 part_swap_k` and `:370-387 swap_pairs_d` (OW streams + `IO.cf` counterfactuals: BM needs its own pair builder, §6); `:313-348 part_probe` (P = 11, OW stream: BM2 with P = 2; BM1 drops it); `:352-356 part_ws`.
- `boxes/prosqa168.py candidate_margins`; `boxes/prosqa168s2.py:63-70 encode` (D1's text-state item form); `theory/score_s191.py` (scorer template).

New code: `bm_data.py` ~220 lines, `gen_wol.py` ~60, `inject_bm.py` ~150 of edits, `inject_bm_eval.py` ~250, scorer ~130.

## 5. Gates

1. **G-EQ (p = 0):** `inject_bm --task ow` on the 200-step smoke = `inject_aw` bitwise (log.jsonl losses and model.pt sha under the det wrap); `tok_bm(--task ow) == tok_aw` on 1,000 items.
2. **G-ALPHA:** fp64: the answer CE on an `ANS_IDS` target equals `inject_aw`'s formula with `NUM_IDS` swapped; the aux CE on a 3-token concept equals the explicit token log-prob sum.
3. **G-CUE-BM:** every pass input ends in ':'; BM1 cues byte-identical across passes (asserted); BM2 cue t's name span = X_t.
4. **Smoke the plain path** per task at the real data scale, then `--mem_check` at the longest eval input (BM1 hop 10 ~250 tokens x K' 16; BM2 d 20 + 3 inserts ~190 tokens), 1 GiB headroom on 24 GB.
5. **Gradflow:** `gradflow_check` on the `ANS_IDS` CE and the sequence aux.
6. **Stage-0 generator gates:** hop-10 production, PROP-SHORTCUT and parity-shortcut rates, dedupe counts, the tokenisation table, all before Stage 1.
7. Guard marks cover the final step.

## 6. Arms, criteria, OC block, censoring

Arms (Stage 2, 3 seeds each unless marked): **LOOP**; **NO-WRITE** (same K passes, write zeroed: the compute-matched pause control, B1 form); **CoT-SFT** (the generator's `chain_of_thought` / a BBH-style trace for BM2, greedy decode then answer); **No-CoT**; **LOOP-NOAUX** (1 seed, descriptive). Same train split and steps for every arm.

C1 battery at the claimed depth (6 and 8; PROP-FOIL for BM1, the distractor DEPTH split for BM2): NOT-DECODER (K' 0), SWAP-K (recipient's S_K := a donor's with equal text length and a different chain; `part_swap_k`'s follow_U >= 0.5 and >= floor + 0.3), MEM-ABLATE, CUE-WRONG (BM2: an off-chain person). Pair builder: donors by regenerating the chain under the recipient's names (BM2: re-flip the statements; BM1: re-draw the chain over the same concept pool), kept iff own != donor final.

| # | criterion | B/D | threshold | NEAR |
|---|---|---|---|---|
| 1 | LEARN: ID test, LOOP, each seed | B | >= 0.95 | CI contains 0.95 |
| 2 | FORMED: WS-READOUT pass 2 on d 2 at 8k | B (mechanism) | >= 0.8 per seed | - |
| 3 | C1-NOTDEC at depth 6 and 8 | B | <= 0.60 | CI contains 0.60 |
| 4 | C1-SWAP-K at depth 6 | B | follow_U >= 0.5 and >= floor + 0.3 | CI contains 0.5 |
| 5 | C2-DEPTH: LOOP at K' = d + 2, per seed | B | min over 3 seeds: depth 6 >= 0.80, depth 8 >= 0.65 | CI contains the bar |
| 6 | C2-MARGIN vs NO-WRITE and CoT-SFT at depth 6-8, per-seed margins | B | every seed > 0, mean >= 0.15 | 0.10-0.15 |
| 7 | K'-CURVE at depth 8 non-decreasing from K' 4 to K' d+2, every seed | B | yes / no | - |
| 8 | C6: K' 128 at depth 6 vs K' d+2 | B | within 0.03; norm ratio <= 1.2 | 0.03-0.05 |
| 9 | C5: stability-rule halt pass | D | median in [d-1, d+3] | - |
| 10 | BBH-250 (BM2, LOOP, K' 8) | D (public number) | Wilson CI; parity caveat | - |
| 11 | SHORTCUT-RATE: NO-WRITE and No-CoT at depth 6 on the exclusion split | B (control validity) | <= 0.60 | - |
| 12 | ORDER / SHUFFLE | D | report | - |
| 13 | C3: FLOPs per item and bs-1 s/item vs CoT-SFT; serial layers | D | report | - |

Wilson 95% CIs on raw values, n = 1,000 per depth, NEAR when the CI contains the bar. The C2 claim is per task, needs 3 seeds on per-seed margins (Note 269-1 ST2), scope "one task, one recipe" until X7 exists.

**OC block (priors, before any run).** BM2: P(LEARN) 0.85; P(FORMED | LEARN) 0.65 (bun's write formed at P-2 from a lineage, here from a 1k one-hop init); P(C1 at d 6 | FORMED) 0.6; P(C2 d 6 >= 0.8 in all 3 seeds | C1) 0.5; P(d 12 >= 0.65) 0.25. BM1: P(LEARN) 0.75 (multi-token states, plan-free cue; §191 reached 0.95 on ProsQA chains in 12k); P(FORMED) 0.55; P(C1 at hop 6) 0.5; P(C2 hop 6 >= 0.8 all seeds) 0.35; P(hop 10 >= 0.5) 0.2. P(CoT-SFT at depth 6-8 < 0.5) 0.7 (SubLAT's CoT-SFT 11.6 on 7-hop ProsQA). H2 outcome: BM2 passes C1 + C2 at d 6-8 with ~0.25, BM1 ~0.15; either plus D1 makes H2.

**Censoring.** A run that dies (OOM, host loss, sha mismatch) is CENSORED and relaunched with the same seed and bs on a 24 GB card; partial logs kept. A seed failing criterion 2 at 8k is NEVER-FORMED: it stays in the 3-seed denominator, is not replaced, and gets no mechanism rescue (rule 4); a declared fourth seed launches only if 2/3 formed. Eval parts dump per part; a crashed part is re-run alone. Generator items failing dedupe or the parse assert are dropped before training and counted. BBH-250 is scored once, after the DEPTH reads.

## 7. Scorer sketch (`theory/score_bm.py`)

Per cell: `args.json` (asserts task, think_k 8, hold_inc, aux_step, seed, steps, bm_data sha); `eval.json` fields `acc.by_d.<d>.acc_kprime.<k>` (LOOP, NO-WRITE), `cot.by_d.<d>.acc`, `direct.by_d.<d>.acc`, `ws.by_d.2.pass2` (FORMED), `acc_kprime.0` (NOT-DECODER), `swap_k.by_d.<d>.{follow_U, floor}`, `long.by_d.<d>.acc_kprime.128` + `norm_ratio`, `halt.by_d.<d>.median_pass`, `bbh250.{acc, n}`, `flops.by_d.<d>.{loop, cot}`, `shortcut.{parity_rate, prop_rate}`; `log.jsonl` key `ws_pass2_d2` per 1k bin. Labels: LEARN-FAIL / NEVER-FORMED / C1-FAIL / C2-MET-d6 / C2-MET-d8 / C2-NEAR / CONTROL-INVALID (criterion 11) / SHORTCUT-SPLIT-ONLY. Fixtures before any run: three synthetic `eval.json` (pass, NEAR, control-invalid) and one with a missing part; `wilson()` from `theory/statlib.py` if ST1 lands, else a local copy.

## 8. Cost (RTX 3090, ~$0.2/h)

Basis: bun 1.65 s/step at bs 64, K 8, ~100-token items; BM2 ~70 tokens -> ~1.3 s/step; BM1 ~130 tokens -> ~2.0 s/step.

| stage | content | $ |
|---|---|---|
| 0 | generators, tokenisation table, shortcut rates, dedupe (CPU) | 0 |
| 1 | ONE-HOP 1k + LOOP 8k, one seed per task (BM2 ~3.2 h, BM1 ~4.8 h) + evals | 1.9 |
| 2 BM2 | LOOP x3, NO-WRITE x3 (2.9 h each), CoT-SFT x3 (~1 h), No-CoT x3, NOAUX x1, evals incl. K' 128 and BBH-250 | ~5.5 |
| 2 BM1 | LOOP x3, NO-WRITE x3 (4.4 h each), CoT-SFT x3 (~1.5 h), No-CoT x3, NOAUX x1, evals on plain + PROP-FOIL | ~8.5 |
| total | Stage 1 is reused as seed 1 if it passes LEARN + FORMED | **~$16** |

## 9. Pairing with PROSQA-OOD for headline H2

One protocol for the three tasks: the same arms, the same C1 battery at depth >= 6, the same K' curve, 3 seeds, Wilson CIs, per-seed margins, one table per task in SubLAT's Table-2 layout. Rules: (a) a task enters H2 only if its C1 gates pass at depth 6 on its shortcut-exclusion split; (b) H2's C2 sentence names every task where criterion 5 holds in all 3 seeds and says "partial" where NEAR; (c) comparability is own-run only (D1's splits are ours; BM1 / BM2 regenerated with published seeds; BBH-250 is the only externally fixed set, with its parity caveat); (d) C3 stays split and is reported, not claimed. BM1 and D1 share the target-cue runner and the sequence-scored readout, so a D1 Stage-0 REDESIGN (set-valued state) applies to BM1 too; BM2 is independent. Spend order: BM2 Stage 1 (cheapest, Z_2 is bun's closest relative), then BM1 Stage 1, then D1 Stage 1 as queued in Lane D.

## 10. Verified arXiv ids

Abs pages fetched 2026-11-03, titles checked: 2210.01240 (Saparov & He, ProntoQA); 2305.15269 (Saparov et al., ProntoQA-OOD); 2210.09261 (Suzgun et al., BBH); 2206.04301 (Zhang et al., LEGO); 2412.06769 (Hao et al., Coconut); 2610.40283 (Zeng & Zhang, SubLAT); 2602.08332 (Amos et al., Thinking States); 2604.04902 (Dilgren & Wiegreffe); 2603.21676 (Chen); 2609.35643 (Cheng & Zhang, ProsQA-Ext).

## 11. RED TEAM

(pending: the red-team sub-agent's report)
