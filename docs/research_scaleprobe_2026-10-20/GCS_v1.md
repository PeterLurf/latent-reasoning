# GSM-CACHE-SWAP v1 (formerly SCALE-PROBE): where does CODI's latent computation live on natural GSM8K, and does it port? (2026-10-20)

**Status:** v1, for Audit (under the user's $30 rule, cap **$0**). Theory's research step after §215–219, ranked first by Theory and Audit (C1 on language is the open gap).
- **Reviews:** SP_draft.md (667f648e) → REVIEW_SP.md (Opus proof check + red team, 2 BLOCKERs, all fixed here); Literature LIT_SCALEPROBE_CHUNKSNAP_2026-10-20.md (2d35f1dc, plus the SCIT follow-ups); Audit's conditions (leak control, identity, floor, OC, SCIT bridge, stage 0 first).

## 0. Question, prior work, what is new

- **On GPT-2, 210-4 / 210-5 read CODI20's latents as necessary but context-bound** (DONOR-SWAP follow 0.040).
  - That swap replaced the latent *input embeddings* z and re-ran the recipient's forward (`latent_override`). That is SCIT's "h-only" family.
- **SCIT 2608.27265** (Ding, Huang, Yang; Literature read it in full):
  - The counterfactual computation travels in the **latent positions' KV cache**, not in the h vector.
  - Synthetic two-hop templates. Latent-tail cache transplant: target win 0.70–1.00 on GPT-2; 0.93–1.00 on LLaMA3-1B (CODI / Sim-CoT). h-only: 0.000–0.246.
  - Metric: a teacher-forced closed set {recipient, donor, oracle-cf, random}, with oracle-cf = the recipient's program on the donor's intermediate.
  - Leak controls: the oracle-vs-donor split; competing prompt-prefix / non-late segments (a carrier call needs oracle ≥ 0.80 and the competing segment ≤ 0.25); semantic sources.
  - At 8B the carrier moves to the prompt prefix.
  - SCIT lists GSM8K as future work.
- **New here** (the novelty claim is limited to these):
  - (a) natural GSM8K pairs;
  - (b) free-generation donor-final follow, not a closed set, plus SCIT's metric as a bridge;
  - (c) the h-only (re-run) and cache granularities paired on the same items;
  - (d) stage 1 on the released CODI-1B with the same arms.
- **Decision value** (outcome → action, §6): whether CODI-style distillation on natural language produces latent computation that carries an answer portably, and where. That bears on whether §218's explicit state-target route is needed for language C1, and whether a CODI-based 1B line is a C1 candidate.

## 1. Stages, models, pins

| stage | model | role | pin |
|---|---|---|---|
| 0 | **gsm_codi_20ep_s1** (our CODI20, boxes/codi_port.py, trainable.pt) | **stage-0 primary** (the 210-4 model) | trainable.pt sha256; codi_port.py git sha |
| 0 | zen-E/CODI-gpt2 (official, MIT; prosqa/ckpt_ext/hf/codi) | matched-recipe GPT-2 comparator (descriptive) | HF revision sha + file sha |
| 1 | **zen-E/CODI-llama3.2-1b-Instruct** (official, MIT) | **stage-1 primary** | HF revision + file sha. Config / tokenizer from meta-llama/Llama-3.2-1B-Instruct or the byte-identical alpindale mirror (HF file hashes equal). Strict key load: only `dynamic_cls.*` unexpected. Unmerged bf16 LoRA r128 / α32, projection 2048 + LN |
| 1 | MisakiTaro0414/MUX (1B), connordilgren/llama32-1b-gsm8k-coconut | descriptive contrasts, only if their released loaders run unmodified | HF revision + file sha; the MUX code repo verified first (its card's URL is a placeholder) |

- bcywinski/codi_llama1b-answer_only is **dropped**: it is not zen-E's weights, and it carries no licence. The Cywiński averaging result is on that model, not on CODI-1B.
- **Code:** the official zhenyi4/codi inference path, ported locally:
  - `[BOS] question <bot>`, z_1 = prj(h_last), each z_j fed as an input embedding, z_{j+1} = prj(h_last);
  - then `<eot>`, greedy decode (logit of `<eot>` excluded), max 256 new tokens, stop at the model's EOS;
  - **parser = the last number in the text, float equality to gold** (ported verbatim);
  - no chat template.
- The download goes into its own directory and is read before use. transformers version and attention implementation are pinned and recorded. On transformers 5.x, DynamicCache exposes `.layers[l].keys / .values`, shape (B, H, T, d).
- **Items:** HF `openai/gsm8k` main test (1,319). Its sha is recorded; it is identical in order and golds to prosqa/ckpt_ext/hf/data/gsm8k_test.json (checked by the reviewer).

## 2. Execution rules

- **Batch size 1 for every arm** (no padding). The official eval attends to left pads with no mask, so outputs at bs > 1 depend on batch composition (REVIEW 1b).
- **Card:** one local 8 GB card per model, recorded.
  - Stage 0 fp32: GPT-2 small fits easily.
  - Stage 1 bf16, as the recipe. Peak ≈ 4.2 GiB at bs 16 per the reviewer, so it is lower at bs 1. **G-M** memory-checks the longest prompt (185 tokens) at K 10 first.
- Time ≈ 2.5–3 h per model per full battery.
- **Order:** stage 0 (CODI20, then zen-E GPT-2), then stage 1 on its trigger (§6).

## 3. Pairs

- **Recipients:** items correct under INTACT (K 6).
- **Primary pairs: EXACT-LENGTH donors.**
  - For each recipient, the donor is drawn with rng 22020 among INTACT-correct items with a different gold and **the same question token length**, so every position, including RoPE's, is identical (Δ = 0).
  - Available for 96–98 % of recipients (reviewer). Recipients with no exact-length donor are excluded and counted.
  - Each recipient is used once. pool_size and n_pairs are reported.
- **Descriptive pair sets:**
  - (i) 210-4's original 500 pairs (unequal length; the h-only arm only, as 210-4);
  - (ii) GRAPH-MATCHED pairs: a dedicated draw (rng 22021) of pairs with the same executable solution-graph signature (from GSM8K's `<<a op b = c>>` annotations). One leaf intermediate differs, and pairs whose oracle-cf equals either gold are dropped. Expected ≈ 179 CODI20 / ≈ 287 1B recipients.
  - (iii) stage 1 only: unequal-length pairs with RoPE key re-rotation by Δ in fp32 (the model's own llama3-scaled rotary), values unrotated. It is gated on CPU: the key error and the logit match against intact at Δ = 0 within 1e-5.

## 4. Arms (per-item outputs stored for every arm; the share-gate rule)

| arm | definition |
|---|---|
| **INTACT** | K 6, the §1 path |
| MEAN-Z | each z_j replaced by its per-position mean over the test set (K 6). Necessity is read against this, not K 0 (REVIEW 5h) |
| K-SWEEP (descr.) | K ∈ {0, 1, 2, 3, 4, 5, 6, 8, 10}; K 0 = `<bot>` followed directly by `<eot>` |
| NOISE (descr.) | z_j ~ Gaussian with per-position norm matched (rng 22022) |
| **STATE-SWAP** | the recipient's z_1..z_6 replaced by the donor's, recipient forward re-run (210-4's h-only granularity) |
| **CACHE-SWAP (primary)** | the recipient's INTACT cache, with the **K/V rows at the 6 latent positions, all layers**, overwritten by the donor's (exact-length pair, Δ = 0). Then `<eot>` and greedy decode. **Assert:** every cache row outside the written slots is bitwise the recipient's intact row |
| **LEAK-SEG** | the same, but the K/V rows overwritten are the donor's **last 6 question-content tokens** (just before `<bot>`), and the recipient's latent rows are kept. The competing segment, Audit's non-latent-span control and SCIT's competing-segment rule. Assert as above |
| CACHE-SWAP variants (descr.) | K-only; V-only; the last latent row only; layer halves |
| PREFIX-SWAP (descr.) | STATE-SWAP of z_1..z_k only, k ∈ {1, 3, 5} |
| QUESTION-DROP (descr.) | `[BOS] <bot> z_1..z_6 <eot>` with the recipient's own z and no question tokens. Floors: constant answer ≤ 0.030, permutation floor ≈ 0.011 |
| QMASK (descr.) | the answer phase's attention to question **content** tokens masked; BOS, `<bot>` and positions kept |
| LOGIT-LENS (descr.) | the top number token from each z_j / latent-row residual through the LM head |
| **SELF-SWAP (identity)** | donor = recipient in STATE-SWAP and CACHE-SWAP. It must equal INTACT **bitwise** (tokens) |
| SCIT BRIDGE (descr.) | teacher-forced log-prob of "The answer is: N" for N ∈ {recipient gold, donor gold, random gold (rng 22023), oracle-cf (GRAPH-MATCHED only)} under CACHE-SWAP and STATE-SWAP. target-win = the donor gold (and oracle-cf) wins. Multi-token numbers are summed |

**Readouts per swap arm:**
- **follow** = output == donor gold;
- **own** = output == recipient gold;
- **neither**;
- on GRAPH-MATCHED, **int-follow** = output == oracle-cf.

**The floor (pre-stated):** the **permutation floor** of each arm's outputs. For each arm, the mean over 1,000 random re-pairings (rng 22024) of P(output == gold of a random other item with a different gold). With GSM8K's answer distribution (Σp² 0.011) it is expected at 0.01–0.05. The 210-3 coincidence floor (0 by construction) is reported beside it only.

## 5. Readings (one primary per stage; everything else descriptive)

- **NECESSARY:** acc(INTACT) − acc(MEAN-Z) ≥ 0.10.
- **CACHE-CARRIER-GSM (primary, per stage):** all four of:
  - NECESSARY;
  - CACHE-SWAP follow ≥ floor + 0.30;
  - LEAK-SEG follow ≤ 0.25;
  - SELF-SWAP bitwise identity.
- **Labels:**
  - **CARRIER-PORTS** = CACHE-CARRIER-GSM.
  - **CARRIER-LEAK** = CACHE-SWAP follow ≥ floor + 0.30 but LEAK-SEG > 0.25: the question relays through the cache.
  - **SCAFFOLD** = own ≥ 0.70 under CACHE-SWAP: the answer is recomputed from the recipient's question (210-2's RECOMPUTES).
  - **CONTEXT-BOUND** = NECESSARY ∧ CACHE-SWAP follow < floor + 0.10.
  - **UNNECESSARY** = ¬NECESSARY.
  - **MIXED** = otherwise.
- **Pre-stated descriptive contrasts:**
  - **GRANULARITY:** CACHE-SWAP follow − STATE-SWAP follow on the same pairs (McNemar, exact).
  - **LATE vs DISTRIBUTED:** PORTABLE-LATE-CACHE if last-row-only follow ≥ 0.8 × all-rows follow, else PORTABLE-DISTRIBUTED (with PREFIX-SWAP and LOGIT-LENS).
  - **INTERMEDIATE:** int-follow on GRAPH-MATCHED. A donor-final null with high int-follow reads "the latents carry intermediates (SCIT-like); the final answer is recomputed with the recipient's own numbers".
- **Wording:**
  - CARRIER-PORTS: *"on natural GSM8K, transplanting the 6 latent positions' KV rows from a donor makes <model> output the donor's final answer at <follow> (floor <f>; a same-size question-span transplant <leak>), and swapping the latent input vectors alone gives <state>. The answer-carrying computation lives in the latent positions' cache. One model, eval-only."*
  - This is a portability property of CODI's latents, **not** a serial-C1 claim (no hop-wise test). Never C3 / C4.

## 6. Outcome → action (pre-stated)

| outcome | action |
|---|---|
| stage 0 CARRIER-PORTS (CODI20) | a Note on 210-4 / 210-5: the context-bound reading was a granularity effect, and the carrier is the cache. Stage 1 runs |
| stage 0 not CARRIER-PORTS, zen-E GPT-2 CARRIER-PORTS | a recipe difference: our port vs the official recipe, reported. Stage 1 runs |
| stage 0 not CARRIER-PORTS in both GPT-2 models | SCIT's carrier does not reproduce on natural GSM8K in our harness. Reported; **stage 1 needs a new Note with a fresh justification** |
| stage 1 CARRIER-PORTS | a language-portability reading for CODI-style training at 1B. A CODI-based 1B line becomes a C1 candidate, for which the next design adds VB-style serial tests (hop-wise swaps on graph-matched pairs) |
| stage 1 CARRIER-LEAK / SCAFFOLD / CONTEXT-BOUND with stage 0 PORTS | scale moves the carrier (SCIT's 8B prompt-prefix shift, earlier on natural data). The explicit state-target route (§218) stays the language-C1 route |
| UNNECESSARY at any stage | the latents are not needed on natural GSM8K for that model. Reported; no further CODI-based C1 work on it |

## 7. OC (exact binomial; the reviewer's oc_sp.py / oc_cache.py)

- **Pairs:** stage 0 ≈ 480–490 (500 recipients × 0.96–0.98); stage 1 ≈ 700 (733 × 0.96).
- **P(CACHE-CARRIER-GSM | no effect)** (follow = floor 0.00–0.05) **≈ 0** (< 1e-6).
- **P(follow bar | true follow = floor + Δ)** at 480 pairs: Δ 0.40 → 1.000; 0.35 → 0.98–0.99; 0.32 → 0.81–0.84; 0.30 → 0.52; 0.25 → 0.006–0.016.
- **The leak gate** (≤ 0.25): P(pass) = 0.997 at a true 0.20; 0.52 at 0.25; 0.008 at 0.30.
- **Priors** [ASSUMED]:
  - P(CARRIER-PORTS, stage 0) ≈ 0.35. SCIT's GPT-2 cache arm reached 0.70–1.00 on templates with the oracle metric, but donor-*final* follow on natural pairs with the recipient's question visible is harder, and the latents may carry intermediates only.
  - P(stage 1 CARRIER-PORTS | stage-0 PORTS) ≈ 0.6.
  - A null is informative through the label it reads.
- **Realised $: $0** (local).

## 8. Censoring (rule b)

- **UNREAD (every reading not met):**
  - the G-R gate fails;
  - a load or pin failure;
  - a test sha mismatch;
  - card / dtype / batch-size mismatch within a model;
  - SELF-SWAP not bitwise;
  - any cache-row assertion fails.
- Unparseable or no-number outputs count as wrong / neither (the official parser).
- Fewer than 300 pairs → the swap arms are UNREADABLE, which fails the primary.
- A missing file is pending (215-7). A stage that does not run (§6 trigger) reads UNREAD = not met, never pending.

## 9. Gates ($0)

- **G-M:** memory at the largest shape first.
- **G-R INTACT reproduction:**
  - the bs-1 INTACT against the official-semantics run of the same checkpoint (virtual bs 128: each item padded to its official batch's max length, no mask, as test.py does): per-item agreement ≥ 0.97 and |Δacc| ≤ 0.025;
  - for CODI20, also |acc − 0.379| ≤ 0.025 (210-4's committed value);
  - for zen-E 1B, 0.556 ± 0.03 as a sanity flag only (single-run figure in the paper).
- **G-I:** SELF-SWAP bitwise identity, plus "donor-own" (re-running the donor with its own rows) bitwise.
- **G-C:** the cache-row assertions, run on 20 pairs with hand-checked slot indices.
- **G-RoPE** (stage 1 descriptive set only): the CPU re-rotation check.
- **G-D:** Audit's diff review of the port, the swap / cache hooks, LEAK-SEG and QMASK spans.
- **G-S:** theory/score_s220.py, fixture-checked before any output: pending rule, labels, floors, McNemar, the action table.

## 10. Cost

**$0.** Local cards, ≈ 3 h per model. Stage 0 needs no download. Stage 1 downloads ≈ 5 GB once.

## Addendum 1 (2026-10-20; Audit's four approval conditions, before registration and before any output)

1. **Label precedence:** UNREAD > UNNECESSARY > CARRIER-PORTS > CARRIER-LEAK > SCAFFOLD > CONTEXT-BOUND > MIXED. The first label whose condition holds is the label. score_s220's fixtures cover every overlap: SCAFFOLD with CONTEXT-BOUND; ¬NECESSARY with high follow; PORTS with high own; LEAK with SCAFFOLD.
2. **Donor reuse:**
   - **Each donor is used at most 3 times.** Donors are drawn with rng 22020 among eligible exact-length donors whose use count is below 3. A recipient with no eligible donor left is excluded and counted.
   - Beside every binomial reading, report the max reuse and an **item-cluster bootstrap CI** (clusters = donor items; 10,000 resamples, rng 22025).
   - The bar is read on the point estimate, as registered. The cluster CI is descriptive.
3. **GRAPH-MATCHED Δ:**
   - Pair set (ii) is restricted to **Δ = 0** (same question token length) where an exact-length graph-matched donor exists.
   - Graph-matched pairs with Δ ≠ 0 are a separate descriptive row with Δ reported per item. Their int-follow is confounded by position, so it is never pooled with the Δ = 0 row.
4. **LEAK-SEG floor:** LEAK-SEG's own permutation floor (§4 method) is recorded beside its follow. The gate stays absolute (≤ 0.25).
