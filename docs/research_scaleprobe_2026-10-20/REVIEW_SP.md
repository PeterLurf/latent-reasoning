# REVIEW_SP: proof check and red team of SP_draft.md (draft 667f648e)

Written 2026-10-20 22:55 EDT by the Opus proof-check / red-team subagent.
- **Scope:** read-only on the repo. The only outputs are this file and `review_scripts/` (oc_sp.py = the OC / GSM8K numbers, pkl_index.py = the checkpoint key / dtype index). Nothing was committed and no GPU was used.
- **Sources read:**
  - zhenyi4/codi@main: `scripts/test_llama1b.sh`, `test.py`, `src/model.py`, `scripts/train_llama1b_gsm8k-aug.sh`;
  - the HF API listings (sha, files, LFS sha256) for the four models and the meta-llama, alpindale and unsloth base repos;
  - the first 2 MB (HTTP range) of each checkpoint, which holds its torch-zip `data.pkl`: keys, shapes, dtypes;
  - HF `openai/gsm8k` main test parquet, sha256 prefix ee7b8da9e381df27;
  - the Llama-3.2 tokenizer.json from alpindale (git blob sha 5cc5f00a…, equal to Meta's);
  - arXiv 2502.21074 HTML and the Cywiński LessWrong post;
  - in the repo: boxes/codi_port.py, theory/codi_probe.py, scratch/s210_latent_use.py, THEORY §207 and Notes 210-2 to 210-5.

## Summary table

| # | finding | grade | fix (v1) |
|---|---|---|---|
| 1a | **The swap object is unspecified.** "Latent states are replaced" can be read as a KV-cache transplant. A raw transplant of KV computed at the donor's positions into the recipient is invalid under RoPE: the relative offsets are wrong by L_r − L_d (median 18 tokens). | **BLOCKER** | Define DONOR-SWAP as a transplant of the K latent **input embeddings** z_1..z_K (prj outputs; z_1 = prj(h at `<bot>`)). Re-run the recipient's own forward at the recipient's positions. This is exactly 210-4 (`codi_probe.run(latent_override)`). Allow a KV transplant only as a descriptive arm, with keys re-rotated by R(L_r − L_d) (exact for RoPE, including llama3 scaling) |
| 1b | **Pads and batches.** The official inference passes **no attention mask and no position_ids** in the latent and decode phases, so left pads are attended there. Per-item output then depends on the item's pad count, i.e. on batch composition (official bs 128). The draft's eval_bs 16, run separately on recipient subsets, gives each arm different padding, and the G-I bitwise gate cannot hold | **BLOCKER** | Run every arm at **eval_bs 1** (no pads; deterministic per item; ~2.5–3 h on one card). Or, if batched, run every arm over the full 1,319 in the same batches with per-row overrides and fixed per-item padding. Also reproduce the official semantics ("virtual bs 128": left-pad each item to its official batch-128 max, physical bs 16) for the 55.6 check only |
| 1c | **Test file.** test_llama1b.sh reads no released file. test.py loads HF `gsm8k` main test via `datasets`. | MINOR (CORRECTED) | Pin `openai/gsm8k` main test (parquet sha256 recorded). It is identical in order and gold to our `prosqa/ckpt_ext/hf/data/gsm8k_test.json` (checked) |
| 1d | **Decoding and parser.** Official decoding is greedy (`--greedy True`): argmax over `logits[:, :V−1]` (eot excluded), stop at `<\|eot_id\|>`, **max_new_tokens 256** (the draft says ≤ 32). The parser takes the **last number anywhere** in the decoded text (not after "The answer is:", as codi_probe does), and the gold is float(split '####', drop ','), scored by exact equality | MAJOR (CORRECTED) | Port test.py's `extract_answer_number` / `compute_accuracy` verbatim. Use max_new 256 with an all-finished break. Use the raw question with BOS and no chat template |
| 1e | **K 0** is under-specified | MINOR (CORRECTED) | K 0 = `[BOS] question <bot> <eot>` → decode. prj(h_bot) is computed and discarded, so no latent position exists. At K, z_{K+1} is computed and unused |
| 1f | **Swap-equivalence lemma** "[PROVED by construction]" | **CORRECTED** | It is a relabelling identity, true only for the **full** swap (k = K) under input-transplant semantics. The latent KV is recomputed in the recipient context, so the answer = f(recipient question, donor z_1..z_K). §207's INPUT-SWAP(k < K) is a **prefix** swap, which the draft lacks: add PREFIX-SWAP k ∈ {1, 3, 5} (descriptive) |
| 1g | **The 55.6 reference is weakly pinned** | MAJOR | In the paper's text it appears only in a figure: the body gives 90 % of CoT-SFT, and an ablation gives 51.9 at the default β 20. It is a single run, and nothing says it was measured on the released file. test_llama1b.sh points at an `ep_3` directory while the training script runs 10 epochs. Gate on reproducing the **official code path on this checkpoint** (pinned env, official semantics), and keep 0.556 as a sanity band (see 5c) |
| 2 | **QUESTION-MASK as drafted** | MAJOR | Its "answer prompt" does not exist: only `<eot>` is fed, and "The answer is:" is generated. Masking BOS (the attention sink) breaks generation, so the floor would be uninformative. The masked answer still reads the question through the latent KV (latents attend to the question). v1: (i) **QMASK-KV**: for `<eot>` and generated queries, all layers, mask the question content tokens only. Keep BOS, `<bot>`, latents, `<eot>` and generated tokens visible, with positions unchanged (masking = dropping the KV with the positions kept; never renumber). (ii) a new **QUESTION-DROP**: re-run `[BOS] <bot> z_1..z_K <eot>` with no question, using the captured z (pure "is the answer in z" test). Floors: K 0 under the same mask/drop (≤ max_a p_a = 0.030 for a constant answer) and a permutation floor (≈ 0.011). Both arms descriptive |
| 3 | **Memory** | PASS | bf16 weights 2.30 GiB (1.236 B params; tied embeddings). Unmerged LoRA +0.17 GiB (test.py casts it to bf16). Max prompt 185 tokens with `<bot>`. At bs 16: KV ≈ 0.23 GiB, full-T logits 0.71 GiB, hidden states 0.21 GiB, context ≈ 0.4 GiB, so the peak is ≈ 4–4.5 GiB, which fits. At bs 128 the official run OOMs on 8 GB (logits 5.7 GiB) unless the virtual-bs trick is used. **fp32** (4.6 GiB of weights) fits at bs ≤ 4, or at bs 16 with last-position logits only (`logits_to_keep=1`, or calling lm_head on the last position yourself) |
| 4a | **OC numbers** | CORRECTED | Pool 733 (acc 0.556) needs ≥ 220 follows. P(pass \| 0.32) = **0.884** (not 0.86). P(pass \| 0.40) = 1.000. P(pass \| 0.28) = 0.12. P(pass \| 0.25) = 0.001. P(pass \| no effect, ≤ 0.10) ≈ 0. SE 0.0052 at 0.02, 0.0172 at 0.32, 0.0181 at 0.40. At a 300 pool, P(pass \| 0.32) = 0.79 |
| 4b | **The coincidence floor is vacuous** | MAJOR | P(recipient's intact pred == donor gold) = 0 by construction: recipients are correct and golds differ (210-4 read 0.000). The control is never exposed. Replace it with a **permutation floor**: P(swap pred == gold of a random non-donor with gold ≠ the recipient's), averaged over ≥ 1,000 draws, from the stored outputs. GSM8K golds: 351 distinct among 1,319; top answers 5 (40), 2 (37), 10 / 4 (35); 36 % are integers ≤ 20. Null floors: Σp² = 0.011 for marginal-like garbage, and ≤ 0.054 if failures copy a number from the recipient's question. The 0.30 bar is unaffected |
| 4c | **The context-bound 0.10 bound** | MINOR | Sensible: P(read < 0.10 \| true 0.08) = 0.98, \| 0.12 = 0.05. Define it on follow and report the permutation floor beside it. The draft drops 210-2's **RECOMPUTES** (own ≥ 0.70), which is a distinct outcome (latents as content-free slots). Add SCALE-SCAFFOLD (necessary ∧ own ≥ 0.70) |
| 5a | **SCALE-PORTABLE can pass on an answer cache.** CODI distils the ':' hidden state, and Dilgren finds the answer is matched after ~31 % of the latents. A late latent holding the final answer as a token-like vector would be portable without serial use | MAJOR | Add pre-stated disambiguation, as 207-1: **PORTABLE-LATE-CACHE** if follow(POS-SWAP K only) ≥ 0.8 × follow(all), else PORTABLE-DISTRIBUTED. Report a logit-lens of each z_j and h_j (top token = donor gold?). Wording: "answer-carrying, portable latents", not "C1" (C1 needs drop-to-chance plus serial hop-by-hop use, per NORTH_STAR) |
| 5b | **Position confound in the scale contrast.** GPT-2 (210-4) has learned absolute positions: the donor's z, taken from a late hidden state at the donor's positions, can carry donor-position information that clashes at the recipient's positions. Llama (RoPE, no absolute position in the residual stream) has much less of this. A 1B "portability gain" could be an architecture artefact, not scale | MAJOR | Add a **LENGTH-MATCHED DONOR-SWAP** arm (donor with \|L_d − L_r\| ≤ 2 tokens and a different gold, rng 21004; ~55 candidates per recipient, so the pool is unchanged). Run it on 1B **and** on GPT-2 (CODI20 and zen-E/CODI-gpt2), $0 |
| 5c | **The intact gate tolerance** | MINOR | ±0.025 is not "1.8 SE": a re-run of a fixed greedy model on fixed items has no sampling error. Its spread comes from numerics and eval semantics (precedent: our port scored zen-E/CODI-gpt2 at 42.5 vs 43.7 published, −1.2 pts). Keep ±0.025 vs the *official-path* run on this checkpoint. Add per-item agreement of the port vs the official path ≥ 0.97, and keep ±0.03 vs 0.556 only as a sanity flag |
| 5d | **Base-weight provenance via the mirror** | PASS / MINOR | The CODI .bin is a **full** state dict: bf16 base, separate lm_head and embed (128,259 rows = +pad/bot/eot), fp32 LoRA / prj, and an unused `dynamic_cls.*` head. So the gated base supplies only config and tokenizer. alpindale's config.json, tokenizer files and model.safetensors have **the same git blob / LFS oids as meta-llama's** (HF API), so the mirror is byte-identical. v1: build from the pinned config, then load_state_dict with a check that missing == ∅ (non-persistent rotary buffers aside) and unexpected == {dynamic_cls.*} |
| 5e | **The answer-only control is mis-described** | MINOR | bcywinski/codi_llama1b-answer_only is **not** zen-E's weights. It is Cywiński et al.'s own three-mode CODI (GSM8k + CommonsenseQA, special tokens `<\|bocot\|>` / `<\|eocot\|>`, chat template, all-bf16 incl. prj). Its 36.72 / 41.60 do not come from a stated GSM8K-test protocol, and it carries no licence (default all rights reserved). Drop it. The Cywiński averaging result is also on *their* model and templated prompts, not "CODI-1B" (fix §0) |
| 5f | **The scale contrast is confounded by recipe** | MAJOR | CODI20 is our port: our split, 20 epochs, γ 2, accuracy 0.379. CODI-1B is the official recipe at 0.556. Add **zen-E/CODI-gpt2** (official, MIT, already at prosqa/ckpt_ext/hf/codi, loadable by codi_probe, 42.5) on the identical battery: official vs official. The scale reading needs zen-E/CODI-gpt2 context-bound **and** CODI-1B portable |
| 5g | **Gate G-I(b) is unrunnable** | MINOR | GSM8K test has **0 duplicate questions**, and the different-gold donor rule would exclude one anyway. Replace it with: (i) self-swap reproduces INTACT bitwise (at bs 1); (ii) the donor's question with its own captured z reproduces the donor's intact prediction bitwise |
| 5h | **LATENTS-NECESSARY via K 0 conflates slots with content.** K 0 is out of distribution (`<bot><eot>`, never trained) | MAJOR | Add **MEAN-Z** (each z_j replaced by its per-position mean over items; Dilgren's arm). Define necessary-by-content = acc(K 6) − acc(MEAN-Z) ≥ 0.10, and report it beside K 0 and NOISE |
| 6 | **The decision rule is overstated** | MAJOR | CONTEXT-BOUND does not imply "state-target supervision is the route": it removes "scale alone fixes it" and is silent on VB transfer. PORTABLE (one checkpoint, other recipe, RoPE) does not imply "move to a 1B base" without 5b and 5f. UNNECESSARY, MIXED and SCAFFOLD have no stated actions. See the decision table in §6 |
| 8i | **GSM-CACHE-SWAP stage 0: is a KV transplant implementable on the GPT-2 path?** | PASS | Yes. codi_probe.run holds `pkv` (transformers 5.18: `DynamicCache.layers[l].keys` / `.values`, shape (B, H, T, d); there is no `key_cache` attribute). The latent tail is the last K rows before `<eot>` (index −K..−1 after the latent loop), which is batch-invariant under left padding. run() has no hook for editing the cache, so the transplant needs a new run variant in the stage-0 script (no boxes/ or theory/ edits). 210-4 swapped the **inputs** z (`latent_override`) and recomputed the cache: SCIT's "h-only" family, not a cache transplant |
| 8ii | **Stage-1 RoPE handling** | **BLOCKER** until specified | **Re-rotate keys by Δ = p_r − p_d in fp32.** Use the model's own llama3-scaled rotary embedding; leave values unrotated. Checked on a tiny llama3-scaled Llama on CPU: max\|k(p+Δ) − R(Δ)k(p)\| = 6.6e-7, and transplant-after-re-rotation logits vs intact differ by 1.2e-7, while the raw unrotated cache differs by 1.5e-2. Reverse-distance position ids are equivalent under RoPE, but the recipient must be re-run at shifted positions, so its question cache is no longer bitwise its own (breaks Audit's assertion). Reject left-padding: official semantics attend to pads, and with masking it reduces to a shift. **Primary pairing for both stages: exact-length donors** (Δ = 0). Available for 96 % of CODI20 recipients and 97.5 % of 1B recipients (median 7 / 12 candidates), so GPT-2's uncorrectable absolute positions are also exact. Keep 210-4's original 500 pairs as the unaligned descriptive set |
| 8iii | **Is "donor-final-follow" meaningful on natural GSM8K pairs?** | MAJOR | It is meaningful as a **carrier-reliance** test: the recipient's question stays in context and competes, which makes the test harder than SCIT's templates. It is not a state test, because a cache transplant moves deep-layer latent rows that attended to the donor's question: an answer register or a question-content relay. The oracle-cf split, which tells an intermediate carrier from a final-answer carrier, needs matching solution graphs. Only **~1.5 of 210-4's 500** random pairs match (P = 0.0031). A dedicated graph-matched draw gives ~179 (CODI20) / ~287 (1B) recipients. Exclude pairs where oracle-cf ∈ {recipient gold, donor gold} |
| 8iv | **OC: follow ≥ floor + 0.30 at 500 pairs** | PASS | Floor 0.01–0.05. P(pass \| floor + 0.40) = 1.000, \| +0.35 = 0.98–0.99, \| +0.32 = 0.81–0.84, \| +0.30 = 0.52, \| +0.25 = 0.006–0.016, \| no effect ≈ 0. At 150 graph-matched pairs: \| +0.35 = 0.89–0.92, \| +0.40 = 0.99. Leak gate (competing ≤ 0.25): P \| true 0.20 = 0.997, \| 0.25 = 0.52, \| 0.30 = 0.008. Oracle ≥ 0.80: P \| 0.85 = 0.999, \| 0.75 = 0.005 |
| 7 | **Minor items** | MINOR | MUX's card says 57.01 (paper 56.7 ± 0.5), and its card's code URL is a placeholder: verify MisakiTaro0414/mux before building. Pin transformers (local 5.18; CODI was written for 4.4x) and `attn_implementation`. Add a "follow-intermediate" row (pred ∈ donor's `<<…=c>>` results \ {donor gold}, permutation floor 0.028) |

## 1. How CODI-1B inference actually works (test.py, src/model.py)

1. **Model.**
   - `AutoModelForCausalLM.from_pretrained(Llama-3.2-1B-Instruct, bf16)`, then `resize_token_embeddings(V0 + 3)` with pad = 128256, bot = 128257, eot = 128258.
   - Then `get_peft_model` (LoRA r 128, α 32, dropout 0.1, on q/k/v/o/gate/up/down) and prj = Dropout(0) → Linear(2048, 2048) → GELU → Linear(2048, 2048) → LayerNorm(2048).
   - `load_state_dict(ckpt, strict=False)`, `tie_weights()`, then the **whole model is cast to bf16**, LoRA and prj included, although the checkpoint stores them in fp32.
   - `model.eval()` turns dropout off.
2. **Tokenizer.** `padding_side="left"`, `[PAD]` added as id 128256. The raw question text is used (`.strip().replace('  ',' ')`), BOS is added by the Llama-3 post-processor, and there is **no chat template**.
3. **Encode.** `[pads] BOS question <bot>` (remove_eos True, so no EOS delimiter), with an attention mask but no position_ids. Then z_1 = prj(h_last at `<bot>`).
4. **Latents.**
   - For i in 1..K: feed z_i as `inputs_embeds` (with **no attention_mask**), then z_{i+1} = prj(h_last).
   - So there is a projection after **every** latent, including the first.
   - K inputs are fed, and z_{K+1} is computed and discarded.
5. **Answer.**
   - Feed the `<eot>` embedding (no mask), then decode greedily: argmax over `logits[:, :V−1]`, so eot is never generated.
   - Stop per row at `<|eot_id|>`, max 256 tokens.
   - The model generates "The answer is: N" itself; no answer prompt is fed.
6. **Parser.** The last number in `decode(skip_special_tokens=True)`, with commas removed; float equality against the gold.
7. **Batching.** bs 128 in dataset order. Pads are masked only as keys of the question pass. During latents and decoding they are visible, and positions count them. RoPE makes the uniform position shift harmless, but the pad keys are not harmless, so an item's output depends on its batch's max length. Training ran with `fix_attn_mask False`, so the model saw the same effect in training.
8. **The 55.6.**
   - Its source is a paper figure, from a single run.
   - The eval is this script: greedy, HF gsm8k test (1,319), bs 128.
   - Whether it was measured on the released file is not stated (ep_3 path vs 10-epoch training).

**Swap-equivalence lemma.** Latents are functions of the question: z_1 through h_bot, and every z_{j+1} through attention to the question KV. The answer phase reads question KV, `<bot>` KV and latent KV (and pads, officially). Under the **input-transplant** swap, the latent KV is recomputed from (recipient question, donor z). So "recipient question + donor z" is identical to "donor z moved to a new question". That makes DONOR-SWAP equal to §207's QUESTION-SWAP at k = K by relabelling: INPUT-FOLLOW ↔ own, CARRIED ↔ follow. This is **definitional, not a theorem**, and it does not cover §207's k = 1, 2 prefix swaps. Retag it [DEFINITION] and add PREFIX-SWAP.

**Our 210-4 harness** (codi_probe.run):
- It overrides the latent **inputs** at positions 1..6 and recomputes everything at the recipient's mask-derived positions (`pos = cumsum(mask) − 1`).
- Pads are masked in all phases, a deliberate deviation from official that matches official at bs 1.
- So 210-4 is an input transplant: correct, and the definition v1 should copy.
- Its residual risk is 5b: GPT-2's late-layer hidden states carry absolute-position information, so a donor z can carry the donor's positions.

## 2. QUESTION-MASK

- **What it can and cannot show.** Masking the question only for the answer phase leaves the latent KV, which attended to the question at every layer. So acc_qmask ≈ acc(K 6) shows only that the answer does not need **direct** access to the question; the latent KV may relay it.
- **Don't renumber positions.** Dropping KV while keeping positions is the same as an additive −inf key mask. Renumbering would change RoPE offsets, so never renumber.
- **Keep BOS.** Llama puts large attention mass on BOS (the sink). Masking it gives garbage that says nothing about the latents.
- **Which tokens to mask.** The question tokens strictly between BOS and `<bot>`; keep `<bot>`, since it is a marker the model saw in training.
- **Cleaner companion: QUESTION-DROP.** A fresh forward of `[BOS] <bot> z_1..z_K <eot>` with the item's own captured z. If acc is high, the answer is recoverable from z alone. Caveat: the positions are unseen in training (latents normally sit after ~56 question tokens), so a low reading is weaker evidence than a high one.
- **Floors.**
  - The K 0 version of each arm gives a constant answer, at most 0.030 (the share of the modal gold "5").
  - The permutation floor ≈ 0.011.
  - Report QUESTION-DROP with donor z scored against the donor's gold (follow) as well.
- **The two readings together.**
  - QUESTION-DROP high with DONOR-SWAP follow low means the latents carry the answer but the recipient's question overrides them ("conflict-bound").
  - QUESTION-DROP at the floor means the latents need their own question to be decoded ("decode-bound", the 210-5 reading).

## 3. Memory (Llama-3.2-1B: embed 262.7 M; 16 layers × 60.8 M; total 1.236 B)

| component | bf16, bs 16 | fp32, bs 16 |
|---|---|---|
| weights (tied) | 2.30 GiB | 4.60 GiB |
| LoRA unmerged (90.2 M) / prj (8.4 M) | 0.17 / 0.02 GiB | 0.34 / 0.03 GiB |
| KV (32 KiB per token in bf16; 16 × 453 tokens) | 0.23 GiB | 0.45 GiB |
| question-pass logits, full T 186 | 0.71 GiB | 1.42 GiB (use last-position logits: ~0) |
| output_hidden_states (17 layers) | 0.21 GiB | 0.42 GiB |
| CUDA context + allocator | ~0.5 GiB | ~0.5 GiB |
| **peak** | **≈ 4.2 GiB** | ≈ 7.8 GiB full logits (no); **≈ 6.4 GiB** with last-position logits, or bs ≤ 4 |

- Merged vs unmerged: official is unmerged peft in bf16. A bf16 merge rounds W + BA. Use unmerged (+0.17 GiB), or merge in fp32 and cast, and let G-R's per-item agreement decide.
- At bs 1 everything is < 3.5 GiB.
- Timing at bs 1: ~22 forward calls per item, ~0.3–0.5 s per item on a 4060.
  - About 9 min per 1,319-item arm and 5 min per 733-pair arm.
  - Full battery (K-sweep 9, swaps ~10, noise / mean / qmask / drop ~5) ≈ 2.5–3 h on one card.

## 4. OC (review_scripts/oc_sp.py; exact binomial)

| intact acc | pool | follows needed | P(pass \| 0.25) | P \| 0.28 | P \| 0.30 | P \| 0.32 | P \| 0.35 | P \| 0.40 |
|---|---|---|---|---|---|---|---|---|
| 0.530 | 699 | 210 | 0.001 | 0.12 | 0.50 | 0.875 | 0.998 | 1.000 |
| 0.556 | 733 | 220 | 0.001 | 0.12 | 0.51 | 0.884 | 0.998 | 1.000 |
| 0.580 | 765 | 230 | 0.001 | 0.11 | 0.50 | 0.883 | 0.998 | 1.000 |

- **No effect.** At follow ≤ 0.10 (any realistic floor), P(pass) < 1e-10.
- **Context-bound bound.** P(follow_obs < 0.10 \| true 0.08) = 0.976, \| 0.10 = 0.52, \| 0.12 = 0.047.
- **Floors** (GSM8K test, 1,319 golds, 351 distinct):
  - gold collision Σp² = 0.0111;
  - P(donor gold ∈ the recipient question's numbers) = 0.054, an upper bound for "failure = copy a question number";
  - P(own gold ∈ own question) = 0.064.
- **Follow-intermediate.** P(recipient gold ∈ donor's `<<…=c>>` set) = 0.028, so a follow-intermediate row needs its own permutation floor. Mean 3.1 intermediates per item.
- **Intact gate.** Under a (wrong) binomial model, P(pass \| true 0.556) = 0.93, P(\| 0.531) = 0.50, P(\| 0.58) = 0.53. See 5c for the right noise model.
- **Draft vs recomputed.** The draft's 0.86 should read 0.88, and its "SE ≤ 0.006" holds (0.0052). P(pass \| effect) should be stated jointly with P(LATENTS-NECESSARY). Cui's noise drop (55.6 → 27.5) makes that likely, but it is not certain under MEAN-Z.

## 5. Red-team details

**5a. Trivial passes.**
- **Similar-question donors.** Rare in GSM8K, and they would make following *harder* to separate from own, not easier. Report follow by question-overlap tercile (token Jaccard) as a descriptive row; no exclusion is needed.
- **Answer cache (the real risk).** CODI's objective aligns the student's ':' hidden state with the teacher's at every layer. If z_5 or z_6 already hold the final answer, swapping them moves the answer with no serial content.
  - The 207-1-style split (LATE-CACHE vs DISTRIBUTED via POS-SWAP K and PREFIX-SWAP) plus a logit-lens of z_j and h_j makes this visible.
  - SCALE-PORTABLE's wording must name which one it is.
  - Neither reading is C1 by itself. NORTH_STAR C1 = drop to chance plus serial, hop-by-hop use. This battery measures **portability of answer-relevant content**, a C1-relevant property.

**5b. Artefacts behind CONTEXT-BOUND.**
- A raw KV transplant would manufacture context-boundness under RoPE: wrong relative offsets, and deeper-layer latent KV carries the donor's question. Hence BLOCKER 1a.
- With the input transplant, Llama is position-clean at the input level: z is a LayerNorm output with no explicit position term, and RoPE is applied inside the recipient's forward.
- GPT-2 is the more exposed model: wpe is added at the input, so late hidden states encode absolute position. CODI20's 0.04 may partly reflect this.
- The LENGTH-MATCHED arm on both architectures resolves it. If GPT-2's follow rises sharply when lengths match, then 210-5's "context-bound" needs a position caveat, and any 1B gain is architecture, not scale.

**5c. Intact tolerance.** See the table. Also require that the port and the official path agree item by item.

**5d. Provenance.**
- zen-E rev b2c88ba2, pytorch_model.bin LFS sha256 c6d4b4b0…ed23b (3,408,265,790 bytes).
- Keys: `codi.base_model.model.*` (base weights in bf16 plus LoRA A/B in fp32), `prj.{1,3,ln}` in fp32, `dynamic_cls.{1,3}` in fp32 (a 7-way head unused by test.py).
- The mirror check is already satisfied by HF oids:
  - config.json 3e3aaf51;
  - tokenizer.json 5cc5f00a (re-hashed locally);
  - tokenizer_config 4ff488a1;
  - model.safetensors LFS 1ff795ff… is identical in alpindale and unsloth; meta-llama's LFS sha is hidden (gated), but its git oid 5daf23ed matches.
- The llama3 rope_scaling block in config.json matters: a wrong config would silently shift RoPE. Pin config.json by its oid.

**5e. Licences.**
- zen-E: MIT. MUX: Apache-2.0. Dilgren Coconut: MIT. zen-E/CODI-gpt2: MIT.
- bcywinski: none stated, so no grant. It is also not a control for zen-E. Drop it.

**5f. Recipe confound.** Covered in the table: add zen-E/CODI-gpt2 as the matched-recipe GPT-2 point. SIM-CoT 3B (internlm/SIM_COT-LLaMA3-CODI-3B, 15 GB of fp32 shards) would give a third scale, but it is too big for 8 GB in bf16 without care. Treat it as a later cloud option, not v1.

## 6. Decision value: what each outcome licenses

| outcome (v1 labels) | what a skeptic says | what it licenses |
|---|---|---|
| PORTABLE-DISTRIBUTED at 1B, and zen-E/CODI-gpt2 context-bound under length matching | "One checkpoint; RoPE vs wpe" | Scale or architecture makes CODI latents portable. That justifies a costed proposal for a 1B language line (NORTH_STAR item 5), not a switch |
| PORTABLE-LATE-CACHE | "An answer register, not a state chain" | No language-line change. It suggests CODI-1B stores the answer early. Feed this to C1 serial tests (PREFIX / POS) |
| PORTABLE at 1B, but zen-E/CODI-gpt2 also portable | "It's the recipe; CODI20 was an under-trained port" | Re-read 210-5 as a port artefact. Re-test our GPT-2 line with the official recipe before any scale claim |
| CONTEXT-BOUND, with QUESTION-DROP at the floor | "Fine, but VB transfer to language is untested" | Removes "scale alone" as a route. Raises the relative value of state-target supervision without establishing it |
| CONTEXT-BOUND, with QUESTION-DROP high | "Latents carry the answer; the question overrides them" | A conflict, not decode-boundness. Motivates a gating / override test rather than a new objective |
| SCAFFOLD (own ≥ 0.70) | "Slots, not content" | The same as §207 at 1B. Strengthens the case for explicit state targets |
| UNNECESSARY (MEAN-Z drop < 0.10) | "Contradicts Cui / Dilgren; check the port" | Treat it as a harness failure until G-R and per-item agreement pass. If those pass, CODI-1B latents are inert under mean replacement |
| MIXED (0.10 ≤ follow < 0.30) | "Partial transfer" | Report the profile. Run the MUX contrast as a tie-breaker. No direction change |

## 7. Recommended v1 changes (in order)

1. Specify DONOR-SWAP, POS-SWAP and PREFIX-SWAP as **latent-input (z) transplants with the recipient's forward re-run**, as 210-4 did. Forbid raw KV transplants; a KV-SWAP with RoPE re-rotation is descriptive only.
2. **Eval semantics.** Run every arm at bs 1 (no pads). Add an official-semantics reproduction (virtual bs 128) for the 55.6 check. Port test.py's parser and stopping verbatim (max_new 256, last number, float equality), use the HF gsm8k test pinned by sha, and use no chat template.
3. **G-R.**
   - The official-path accuracy on this checkpoint, with the port within ±0.025 of it and per-item agreement ≥ 0.97.
   - 0.556 ± 0.03 as a sanity flag.
   - State the noise model as numerics, not sampling.
4. **Replace the coincidence floor** with a permutation floor, applied to follow, follow-intermediate and the QMASK / DROP arms.
5. **Add arms:**
   - MEAN-Z (necessity by content);
   - LENGTH-MATCHED DONOR-SWAP;
   - PREFIX-SWAP {1, 3, 5};
   - QUESTION-DROP;
   - logit-lens of z_j and h_j;
   - follow-intermediate.
6. **Add labels:** SCALE-SCAFFOLD (own ≥ 0.70), and PORTABLE-LATE-CACHE vs PORTABLE-DISTRIBUTED. Map every label to an action (§6).
7. **Add zen-E/CODI-gpt2** (local, $0) as the matched-recipe GPT-2 comparator on the identical battery. Re-run 210-4's swap with length-matched donors on CODI20.
8. **Model list.** Drop bcywinski (not a zen-E control; no licence). Fix the Cywiński attribution in §0. Fix MUX's figure to 57.01 (card) / 56.7 ± 0.5 (paper).
9. **Loader.** Build from the pinned config, then a strict key check (unexpected = dynamic_cls.* only). Use unmerged LoRA in bf16 as the official path. fp32 is descriptive only, with last-position logits.
10. **G-I.** Replace the duplicate-question gate (0 duplicates exist) with self-swap bitwise plus donor-own bitwise at bs 1. Fix the OC number: 0.86 → 0.88.

## 8. Addendum: GSM-CACHE-SWAP re-scope (Theory via the coordinator, after SCIT 2608.27265; folded in 2026-10-20 ~23:20 EDT)

**(i) Is the transplant implementable on stage 0's GPT-2 path, and what did 210-4 do?**
- **Implementable.**
  - `codi_probe.run` already threads `past_key_values` through the question pass, the K latent steps and decoding.
  - On the local transformers 5.18 this is a `DynamicCache`, with `.layers[l].keys` / `.values` of shape (B, H, T, 64) for GPT-2. Checked on a tiny random GPT-2 on CPU (review_scripts/cache_api_check.py). There is no `key_cache` attribute, so code written for 4.x tuples or `key_cache` lists will break.
  - After the latent loop, the latent rows are the last K along T (positions P..P+K−1, P = padded question + `<bot>` length). Indexing from the end is the same for every row under left padding: this is SCIT's reverse-distance alignment.
- **How to write it.**
  - Capture the donor's K latent rows at every layer (6 × 12 × 2 × 768 fp32 ≈ 0.44 MB per item; 0.22 GB for 500).
  - In the recipient's run, after its own latent loop, overwrite `keys[:, :, -K:]` and `values[:, :, -K:]` in place at every layer, before `<eot>`.
  - Assert that rows `[:, :, :-K]` are `torch.equal` to the recipient's intact cache.
  - `run()` has no cache hook. Write a new run variant in the stage-0 script; per §207's "new script, no boxes/*.py edits", never patch theory/codi_probe.py in place.
- **What 210-4 did** (scratch/s210_latent_use.py → codi_probe.run(latent_override)):
  - At each latent position j in `pos`, the input z_j was replaced by the donor's captured z_j.
  - The recipient's forward then recomputed every latent KV row, attending to the **recipient's** question, at the recipient's mask-derived positions.
  - So 210-4 is the **re-run state swap** (SCIT's h-only family). Its 0.040 is consistent with SCIT's GPT-2 / 1B h-only 0.000–0.246.
- **The paired comparator.**
  - 210-4's arm is deterministic. Re-running it at the same card / bs / fp32 must reproduce 210-4's per-pair outputs bitwise on the original 500.
  - Make that a gate before any cache-arm output.
- **Note (check, not a finding).**
  - codi_port.codi_loss passes no `position_ids`, so if HF GPT-2 defaults to arange over the padded length, CODI20 trained with pad-inclusive absolute positions.
  - codi_probe evaluates with positions starting at 0.
  - If so, CODI20 saw varied absolute offsets in training, which weakens (but does not remove) the GPT-2 position confound.

**(ii) RoPE handling for stage 1.**

| option | exact? | recipient question cache bitwise its own? | verdict |
|---|---|---|---|
| **Re-rotate the transplanted keys by Δ = p_r − p_d** (fp32, then cast; values unchanged) | yes: R(a)R(b) = R(a+b) per frequency, including llama3 scaling. Tiny-model check: key error 6.6e-7, logits 1.2e-7 vs intact; raw unrotated 1.5e-2 | yes | **use** |
| Align by reverse-distance position ids (re-run the recipient so its latent tail sits at the donor's positions) | equivalent under RoPE in exact arithmetic; bf16 cos/sin rounding differs | **no**: its question cache is recomputed at shifted positions | descriptive at most |
| Left-pad to equal length | official semantics attend to pads (changes the computation); masked, it reduces to a shift | no | reject |

- **Δ** = (recipient's absolute index of latent slot j) − (donor's index of slot j). Both are counted on the same convention: bs 1, no pads; or, under official semantics, pad-inclusive cache_position.
- **Prefix/competing-segment controls.** Re-rotate any transplanted non-latent rows to their destination positions the same way.
- **Gates.**
  - (a) A shifted self-transplant: run the recipient at positions +Δ, take its latent rows, re-rotate by −Δ, write them into its unshifted cache. Its output must equal intact (fp32: bitwise preds; bf16: preds equal on ≥ 99 % of a 64-item check).
  - (b) Exact-length pairs give Δ = 0, a no-op that must match the unrotated path bitwise.
- **Primary pairing for both stages: exact-length donors.**
  - GPT-2 has learned absolute positions, so no correction exists. The fix is pairing.
  - A random-but-exact-length donor draw (rng 21005) covers 95.8 % of CODI20 recipients (pool ~495, median 7 candidates) and 97.5 % of 1B recipients (median 12).
  - Positions are then identical, with no correction in either architecture.
  - Keep 210-4's original 500 pairs as an unaligned descriptive set, so the length effect itself is measured.

**(iii) Natural GSM8K pairs and "donor-final-follow".**
- **Meaningful as a carrier test.** Under a latent-tail transplant, the recipient's question rows are intact and visible to the answer. A donor-final output therefore means the answer phase reads the transplanted latent rows *over* the question. That is a stronger reading than in SCIT's same-template pairs, where the recipient's text does not contradict the donor's quantities.
- **Not a state or serial test.**
  - The transplanted deep-layer latent rows attended to the donor's question, so they can relay donor question content, or hold the final answer in a register (CODI distils the ':' state).
  - SCIT finds the carrier shifts to the prompt prefix at 8B. Hence Audit's competing-segment control: the same number of donor rows from a non-latent span (e.g. the last K question rows before `<bot>`), re-rotated in Llama, written into the same destination slots. Pass needs competing ≤ 0.25.
- **The oracle-cf split** (recipient program applied to the donor intermediate) is undefined on unmatched graphs.
  - Graph signatures come from the `<<expr=c>>` annotations, with operands mapped to q / r_i.
  - 855 distinct signatures among 1,319 items. P(random pair matches) = 0.0031, i.e. ~1.5 of 210-4's 500 pairs.
  - A dedicated graph-matched draw gives ~179 CODI20 and ~287 1B recipients (assuming correctness is independent of signature).
  - Define oracle-cf by substituting the donor's step-k result into the recipient's step k, then recomputing the recipient's later steps (pre-state k ∈ {1, last − 1}).
  - Drop pairs where oracle-cf ∈ {recipient gold, donor gold}.
- **The intermediate-follow row** belongs on this matched set only. Permutation floor ≈ 0.028 for any-intermediate.
- **The SCIT closed-set metric on GSM8K** is a descriptive bridge only.
  - Candidates are multi-token numbers of different lengths. Score the full continuation "The answer is: N<eos>" by summed log-prob, and report a per-token-normalised version as well.
  - Free-generation follow remains primary.

**(iv) OC (review_scripts/oc_cache.py).**
- Pass iff follow_obs ≥ floor_obs + 0.30. The floor is modelled conservatively as an independent binomial; the permutation floor's real SE is smaller.

| pairs | floor | +0.25 | +0.28 | +0.30 | +0.32 | +0.35 | +0.40 |
|---|---|---|---|---|---|---|---|
| 500 | 0.00 | 0.006 | 0.17 | 0.52 | 0.84 | 0.99 | 1.000 |
| 500 | 0.03 | 0.012 | 0.19 | 0.52 | 0.82 | 0.99 | 1.000 |
| 500 | 0.05 | 0.016 | 0.21 | 0.52 | 0.81 | 0.98 | 1.000 |
| 300 | 0.03 | 0.042 | 0.26 | 0.52 | 0.77 | 0.96 | 1.000 |
| 150 | 0.03 | 0.117 | 0.34 | 0.53 | 0.71 | 0.90 | 0.99 |

- **No effect** (follow = floor): P(pass) ≈ 0 at every n.
- **SCIT-like effect** (0.93–1.0): P(pass) = 1.
- **Carrier call** (oracle ≥ 0.80 ∧ competing ≤ 0.25) at 500:
  - P(oracle_obs ≥ 0.80 \| 0.85) = 0.999, \| 0.80 = 0.53, \| 0.75 = 0.005;
  - P(competing_obs ≤ 0.25 \| 0.20) = 0.997, \| 0.25 = 0.52, \| 0.30 = 0.008.
  - At the graph-matched n ≈ 179, these thresholds are ~1.5× noisier. Pre-state that the oracle split reads only if n_matched ≥ 150.

**Net for the re-scope.**
- Stage 0 and stage 1 are sound once three things are fixed in the text:
  - (1) exact-length pairing as primary, with re-rotation for any Δ ≠ 0 in Llama;
  - (2) the oracle / intermediate rows on a dedicated graph-matched draw, not on 210-4's random pairs;
  - (3) the whole-cache bitwise assertion outside the written slots, in every arm.
- The 1B items in §1–7 still apply to stage 1: bs 1 / pad semantics, official parser and decoding, the G-R reference, MEAN-Z, the permutation floor, and dropping bcywinski.
- QUESTION-DROP / QMASK become optional descriptive arms.
