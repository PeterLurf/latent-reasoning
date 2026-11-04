Status: completed 2026-11-04 by Theory's agent; not red-teamed yet

# 19 SCALE PORT: S-ONLY repair test on GPT-2 medium, then a Qwen3-0.6B-Base port of the P-2 seed (design, 2026-11-03 / 11-04)

**Provenance.** The 11-03 draft (SC2 SPI + S11 S-only decoder + S10 Qwen3 port) was cut off at its red-team section. This file rewrites it in place as one complete, staged design. Read-only work: no training, no download, no computation on stored run outputs. The only stored files opened were `args.json` of p2m_s1b, askm_lr1e4, aw_plaw2_s1 and s233_askname_s1 (recipe flags), plus two docs-side JSONs from the earlier Qwen2.5 line (`docs/research_largebase_2026-10-23/lb_tok_check.json`, `lb_p1_eager.json`).
**Backlog IDs:** S11 S-ONLY DECODER (= backlog M3's S-only half); S10 QWEN3-0.6B PORT + READINESS; S12 Qwen P-2 seed (absorbed here as Stage 4). SC2 SPI is deferred (§0.2).
**Priority fit (NORTH_STAR: C1 > C6 > C4 = C2 > C3 > C5 > C7).** This is a C1 lane: it asks whether the latent loop's per-pass computation can be seeded on a larger, modern base at all. No C2 / C6 / C5 claim is made. A seed here only unlocks a later REG d <= 4 registration on Qwen.
**Scope line (pre-stated):** one seed, one lineage (askname -> P-2 on each base), one GPU class (RTX A6000), d <= 2, plan given (cue step), chain-order text, REG frames 0-5.

## 0. Summary

### 0.1 One-paragraph plan

§258 parked GPT-2 medium because pass 2's ':' readout (WS) never showed v_2. Yet the memory did hold v_2 (aux 1.00, read through a prompt that contains no problem text), and the decoder solved d 2 from v_1 plus the text (Notes 258-9, 264-1). Nothing about Qwen3 removes that decoder shortcut. A bigger, better-trained base should make the shortcut stronger (§1.3). So a plain port would most likely repeat §258.

The design therefore tests the remedy first, where it is cheapest. **Stage 1** (~$2.5) warm-starts medium's p2m_s1b with an S-only decoder: the decoder sees only the question, so the last hop must come through S. If pass 2 does not then compute the hop causally (SWAP2 at K' 2), the larger-base line **PARKS** at Stage 1, and the next step is M8 / M2.

On GO, Qwen3-0.6B-Base is ported through new, default-preserving modules (Stages 2-3: gates and an askname one-hop init, ASK-Q). Stage 4 then runs the P-2 recipe with the S-only decoder (P-2-Q, 12k + one 6k extension). Its formation gate is the pass-2 WS read at >= 4k (log bins 4k-12k, binding at 12k / 18k), paired with a causal pass-2 swap.

Family guard **$20**, expected spend ≈ **$6.5**, P(park at Stage 1) ≈ 0.5.

### 0.2 Changes from the 11-03 draft

- **SC2 SPI (30 direct-FT cells, $1.4) is deferred and not registered here.**
  - It is not decisive for the port: the draft's own stated weakness is that medium's shortcut is one hop from memory's v_1, not two hops from text.
  - Medium's one-hop-from-memory capacity is already shown behaviourally (Note 264-1 M1: follow_ca 1.000).
  - The SC2 build, `numread.py`, survives as the Qwen readout (§5.3).
- **S11 SO-F (fresh medium, 12k) is deferred.** If Stage 1 passes, the Qwen run is the fresh-formation test, on the base we actually want.
- **CTRL-W becomes conditional** on SO-W's GO, because it only attributes a pass. SO-WS (small comparator) runs in parallel with SO-W, because it is what interprets a fail.
- **The S10 readiness checks R1-R3 become Stages 2-3.** S12 becomes Stage 4. A replication (seeds 2-3) is pre-named as a separate registration (§11), so this design stays under $30.
- **Added:**
  - a formation gate on pass-2 WS, read at >= 4k (FD1, Note 269-1);
  - an exact multi-token WS read for Qwen (WS-Q, §3 M5);
  - a $0 pre-stated Stage 0;
  - the math backing (§3), the OC block (§9) and conservative censoring (§10).

## 1. Diagnosis: why GPT-2 medium did not seed (§258), and what Qwen3 does and does not change

### 1.1 What §258 measured (DATA)

| fact | source |
|---|---|
| Small P-2 recipe (aw_plaw2_s1 args, d <= 2, aux 1.0, cue step, write_src answer, lr 1e-4, 12k): WS d 2 pass 2 ≈ 1.0; aux formed 0.9999 by 6-7k | Notes 258-7, 269-1 |
| Medium, same recipe from askm_lr1e4 at lr 1e-4 (p2m_s1b; args diff vs aw_plaw2_s1 = base, init, micro 2, bookkeeping only; read this session): TEST2 d 2 1.000; **WS d 2 pass 2 0.017 / 0.015** at 6k / 12k (below chance) | Note 258-9 |
| Medium log aux_vacc d 2 pass 2 = 0.11 / 0.13 / 0.14 / 0.40 / **1.00 / 1.00** at 2k-12k: the write forms at ~9-10k | Note 258-9 |
| d 2 acc at K' 1 = 0.974 / 1.000: the answer is already there after one pass | Note 258-9 |
| K = 8 whole-state swap: follow_ca 1.000, follow_dfin 0.000. The decoder applies the recipient's op 2 to the donor's v_1; the donor's v_2 in S is never used. Pass 2's ':' echoes v_1 (0.84 / 1.00) | Note 264-1 M1 |
| At lr 3e-5 (p2m_s1) the aux crept to 0.18 by 12k: never formed | Note 258-7 |
| Formation is invisible at 2k (all 15 d_max-2 runs at chance); onset 4-7k at small, ~9-10k for medium's aux | Note 269-1 FD1 |
| Small's frozen askname trunk cannot use a value read into its residual: G-AW1 v_2 0.118 vs majority 0.107 | Note 234-2 |
| Qwen2.5-0.5B few-shot is at chance on CHAIN (0.100); frozen Qwen2.5 has v_1 at chance at the write sources (probe 0.112 vs majority 0.116) | Note 252-4; lb_p1_eager.json |

### 1.2 Reading (claims tagged in §3)

**(a) The write did form at medium.**
- The aux prompt is `[EOT, " Step", " t", " :"]` + target prefix, with **no problem text**: `nested170.InjectN.aux_logits` 260-262, read only on S; inject_ow.py 92-103 uses `fwd_read` on S. So aux 1.00 at pass 2 means S_2 carries v_2 (§3 M1).
- Some pass in 1-2 computed op 2. §258's NO-SEED therefore does not show that medium cannot form the write.

**(b) The decoder never needed it.**
- The text decoder sees every op sentence plus S_K (tok_aw asserts `dec == think + SFX`, inject_aw.py 90).
- Medium's trunk can apply one op to a value read from S in a single forward. Small's trunk cannot (G-AW1 0.118), and that was small's lucky constraint.
- Once the decoder route fits, the answer-CE gradient into the pass-2 write vanishes (§3 M2). Only the aux keeps pressing S_2 → v_2, which explains why the aux formed late and through a non-LM subspace.

**(c) WS is an LM-head readout of the pass's own ':' state, never trained in P-2.**
- The ':' readout appears in `ws_readout`, inject_aw.py 135-152 ("never fed back").
- At small the askname-trained head reported the computed v_2. At medium it reports the value read from S_1 (the echo).
- So the registered mechanism gate (WS) and the aux decodability came apart. Note 258-9 states this; the verdict stands as registered.

**(d) Two failure readings remain.**
- ECHO-READOUT: pass 2 computes v_2 into the write, but the head shows the read.
- NO-PASS-2-COMPUTE: pass 1 computes v_2 eagerly, or the passes accumulate it.
- Stored data cannot separate the two, because the only swap that exists is the K = 8 one, where the text decoder recomputes. Stage 1 separates them causally (§3 M3).

### 1.3 What Qwen3-0.6B-Base changes

Config, from the draft's 11-03 fetch of the HF config: 28 layers, D 1024, 16 q / 8 kv heads, head_dim 128, QK-norm, RoPE θ 1e6, RMSNorm, SwiGLU, tied embeddings, vocab 151,936. Non-embedding params ≈ 0.44B (Qwen3 report 2505.09388), vs ≈ 0.30B for GPT-2 medium.

| difference | effect on the §258 failure | tag |
|---|---|---|
| Capacity / pretraining (0.44B non-embedding, 28 layers, ~36T tokens incl. synthetic maths) | a stronger one-forward hop from read value + text, so the decoder shortcut gets **stronger**, not weaker (simplicity bias, 2006.07710; shortcut learning, 2004.07780). A plain-decoder port is predicted to fail like medium | CONJECTURE (P 0.65) |
| No in-context competence on this task (Qwen2.5-0.5B few-shot 0.100 on CHAIN; v_1 not eager at the write sources) | the askname one-hop init is still needed; the raw base gives no foothold | DATA for Qwen2.5; transfer to Qwen3 ASSUMED |
| Tokenizer: digits split one per token, with a separate space token (" 7" = [" ", "7"], " 10" = [" ", "1", "0"]); first-token collision 1 / 10; 40/40 names single-token | the write must sit on the standalone-space token (the position whose LM head emits the first digit). WS must be read over the restricted 11-class trie (§3 M5), otherwise 1 and 10 merge. The aux target index must hit digit tokens, not the space (REVIEW_LB 9a). Names are no problem | DATA for the Qwen2.5 tokenizer (lb_tok_check.json); Qwen3 = same BPE ASSUMED, asserted by G-Q5 before any spend |
| RoPE (no wpe) | memory reads and writes carry no position. Right padding plus explicit position_ids keeps the real positions exact (§3 M6). The GPT-only wpe adds (`DM.wpe_slice` deltamem.py 43-48; `gpt_lower` 263-269; `fwd_read` deltas5.py 91-98; `pass_aw` inject_aw.py 49) must never run. POS_OFFSET / KEEP-SINK evals are N/A. Neutral for the shortcut; position-bound lookup (Note 252-4) is still possible under relative positions | PROVED (exactness); CONJECTURE (neutral) |
| GQA 16 q / 8 kv, head_dim 128 | the memory needs 16 distinct 64-dim heads from 8 kv heads: split each kv head into two halves (§5.2). This avoids LB's 7x-replication collapse (REVIEW_LB #4). RoPE partner dims (i, i + 64) land in different memory heads; at relative position 0 the pretrained q·k is the sum of the two halves' products | PROVED (sum identity); CONJECTURE (adequate init) |
| QK-norm (RMSNorm on each 128-dim q / k head, before RoPE) | copied as learnable per-128 weights onto the memory's q / k sources before the 64-split. Keys are better conditioned than GPT-2's (whose anisotropy blocked acquisition, NORTH_STAR lessons §72b / §73). A mild help to the write's key geometry | CONJECTURE |
| RMSNorm (no bias, no mean subtraction) | DeltaMemQ uses RMSNorm copies of layers.14.input_layernorm (read / write input) and layers.27.input_layernorm (ln_w). Known massive-activation dims in Qwen residuals may dominate un-normalised writes, so G-Q4 and the read_ratio monitor watch for it | CONJECTURE (risk) |
| Tied embeddings (as GPT-2) | the decoder and WS logits use embed^T, as on GPT-2. No change | PROVED (config) |
| Depth 28, L/2 = 14 | the write source is block 27's output, with 14 blocks above the read. More room for both the loop and the decoder shortcut | CONJECTURE |

**Conclusion.** Qwen3 changes the plumbing (tokenizer, positions, norms, heads), and §5 handles that exactly. It does not change, and probably strengthens, the cause of §258's NO-SEED: a decoder that can do the last hop itself. The port is therefore only worth running with the S-only decoder, and only after Stage 1 shows that this remedy makes pass 2 compute on a base where the shortcut is known to exist.

## 2. Init strategy (options considered; the choice is pre-stated)

| option | verdict | reason |
|---|---|---|
| **A. Askname one-hop pretraining (ASK-Q), the s233 recipe on Qwen, 8k, then P-2-Q with an S-only decoder** | **chosen** | It is the init every formed loop descends from (askname -> aw_plaw2). It trains the ':' head to answer "How many u does X have?" for one hop, with class (iv) "S_2 with S_1 given" -> v_2, which is exactly pass 2's job once v_1 comes from memory. Medium passed it at both lrs (Note 258-3: classes i-iv 1.000, TEST2 d 1 1.000) |
| A'. Askname with perm_p 0.5 (s252_askperm, INIT-LOOKUP, Note 252-5) | not now | a second factor beside the base change. It belongs to the later order-free stage (of lineage, Note 263-3) |
| B. STATE-DISTILL from bun / of (teacher per-pass ':' soft targets) | rejected | 09_REDTEAM F3: its KL trains the WS readout the formation gate reads (a trained reading), its targets are hand-built by proxy, and the teacher has a different tokenizer. NO-GO as written |
| C. Direct P-2 from the raw base | rejected | Qwen2.5 few-shot on CHAIN at chance (0.100) and no eager v_1 (0.112), so there is no one-hop foothold. §233 shows P-2 needs the askname head |
| D. M2 COLON-CE (CE on the ':' at every pass) / `--cue_ans` (Note 251-10) | fallback only | it trains the quantity the WS gate reads. If ever used, the formation gate must move to the causal swap |
| E. Ask co-training (`--ask_p`, §263's of recipe) inside P-2-Q | not in P-2-Q | it CE-trains the ':' head on one-hop ask items, partly training the WS readout, and it adds a factor. Kept for the post-seed REG stage |

**lr.** ASK-Q and P-2-Q use **lr 1e-4**. Note 258-8: lr 3e-5 confounded medium's S2, while 1e-4 formed the aux. ASK-Q gets one fallback arm at 3e-5, only if 1e-4 fails R2 or goes non-finite (§7.3). P-2-Q runs at the lr of the ASK-Q arm that passed.

## 3. Math backing (assumptions, mechanism, tagged claims)

Notation: pass t reads S_{t-1} at block L/2 (read_fixed, deltamem.py 158-176) and writes S_t from the last real token (':' on GPT-2, the space token on Qwen) via write_seq (179-223). The decoder reads S_K, read-only (fwd_read). Items have d ∈ {1, 2}; v_t is the value of S_t; op_t maps v_{t-1} to v_t.

**M1 (PROVED, given the code): aux above chance at pass t means S_t carries information on v_t.**
- The aux input is `[EOT, " Step", " t", " :"]` + the target prefix, with no problem text (nested170 260-262; inject_ow 92-103), and its only other input is S_t through read_fixed.
- If S_t were independent of v_t, the aux prediction would be independent of v_t, and its accuracy could not exceed the majority share.
- So medium's aux 1.00 at d 2 pass 2 (Note 258-9) means S_2 determines v_2 almost surely on the 200-item log subset. Some pass ≤ 2 applied op_2.

**M2 (DERIVED-UNDER-ASSUMPTIONS): under the text decoder, the answer gradient into the write vanishes once the decoder route fits.**
- Assumption A1: the decoder computes p(v_2 | text, S_K) through a route R that uses v_1 from S_K plus the op-2 sentence (Note 264-1 M1 is the DATA for A1 at medium).
- The answer CE gradient with respect to the logits is p − y. As p(y) → 1 through R, ‖∂L/∂S_K‖ ≤ ‖p − y‖ · ‖∂logits/∂S_K‖ → 0, whatever S_K holds about v_2. The only remaining pressure on S_2 → v_2 is the aux, which reads S through a text-free prompt and does not constrain the ':' LM readout of the pass.
- **Prediction (already met, POST-HOC):** at medium the aux forms while WS does not. At small, A1 fails (G-AW1: the trunk cannot use a read value, 0.118), so the answer gradient stays and has to flow through the write.

**M3 (PROVED, given the code change): with `--dec_src cue` the decoder's output depends on the item only through (question tokens, S_K).**
- dec = `[EOT] + enc(" " + cue_str(it, d)) + SFX` contains no op sentence. cue_str (inject_st.py 44-46) is a function of names and unit only.
- **Corollary 1 (SWAP2 identification):** in `swap_1`, the recipient's S_1 is replaced by the donor's, and pass 2 then runs on the recipient's own input (text + cue + SFX). For K' = 2, the answer is f(q_rec, S_2) with S_2 = W(S_1^don, x_rec). follow_ca(K' 2) = P(answer = op_2^rec(v_1^don)) above floor_ca can only arise if pass 2 combined the donor's S_1 content with the recipient's op-2 sentence. The decoder cannot see that sentence, so the hop happened in pass 2. A pass-1 eager computation cannot produce it either, since the recipient's own S_1 was discarded.
- **Corollary 2 (no text route):** a model with NOT-DEC failing (K' 1 high) under S-only must have S_1 carrying v_2, i.e. pass 1 computed both hops (EAGER-P1). That is labelled separately.

**M4 (DERIVED-UNDER-ASSUMPTIONS): under S-only, the answer pressure on the write cannot vanish until S_K carries v_d.**
- Assumption A2: in the generator, v_d is independent of the question tokens (names, unit). TEST2's majority share ≈ 0.10 is DATA consistent with A2; the generator check is a fixture (G-DS2).
- Then any decoder that ignores S has expected CE ≥ H(v_d | q) ≈ ln 11 − δ nats and accuracy ≤ majority. So ∂L/∂S_K stays bounded away from 0 until S_K determines v_d, which is the opposite of M2.
- **Prediction:** S-only removes the shortcut equilibrium. It does **not** by itself force the hop into **pass 2**: passes 3..8 see the same text and cue, so a LATE-HOP solution is admissible. The per-pass aux (S_2 → v_2) is the only term that places it at pass 2 (DERIVED; LATE-HOP is a pre-stated label).

**M5 (PROVED): the WS-Q restricted-trie read is exact, and equals WS on GPT-2.**
- Each class v has a token sequence c_v after the write position. s(v) = Σ_i log p(c_{v,i} | prefix) + 1[c_v is a proper prefix of some c_w] · log(1 − Σ_{children ch} p(ch | c_v)).
- With causal attention and a read that uses only the pass-start S_{t-1} (read_fixed adds wo(read(S_{t-1})) at every position), appending the probe token "1" after the write position leaves every earlier position's logits unchanged. So one extra eval-only forward on x_t + ["1"] with S_{t-1} gives all terms. The pass's write is taken from the unappended forward.
- On GPT-2 no c_v is a prefix of another (" 0".." 10" are single distinct tokens; lb_tok_check `answer_single_token_after_space` true). The indicator term is then 0 and argmax s(v) = argmax over NUM_IDS logits, i.e. ws_readout. Gate G-WSQ0 checks this bitwise.

**M6 (PROVED): RoPE port exactness.**
- With position_ids = arange(T) for each sequence, right padding and a causal mask, the real tokens at positions < len attend only to real tokens with the same positions as unpadded. So padded = unpadded on real positions.
- With wo = 0 at install, read_fixed returns x + fade·0 = x. So the step-0 model equals the plain trunk (G-Q3: max|Δ| = 0).
- Lower-cache identity: blocks 0..13 on identical inputs per pass are shared exactly (the same argument as deltamem.gpt_lower's docstring, 263-265).

**M7 (PROVED): the GQA split keeps the pretrained relative-position-0 score.**
- For kv head j and its first q head 2j, q·k = q_{[0:64]}·k_{[0:64]} + q_{[64:128]}·k_{[64:128]}. Memory heads (2j, 2j + 1) take the two halves after the shared q_norm / k_norm.
- Using q head 2j only (not 2j + 1) is a choice. Head distinctness is gated (G-Q4), not proved.

**M8 (CONJECTURE): shortcut propensity grows with one-forward capacity.**
- The faster-learned route captures the answer gradient (2004.07780, 2006.07710); medium > small is DATA (Notes 234-2, 264-1).
- **Prediction:** a text-decoder Qwen port would not seed (P 0.65; not run, disclosed).

**M9 (DATA → DERIVED): formation markers need reads at >= 4k.**
- FD1: every d_max-2 run is at chance in WS and aux at 2k, and onset was 4-7k (small) or ~9-10k (medium aux). So no 2k gate. The binding read is at the run's end (12k / 18k), and the 4k-11k log bins are the pre-stated trajectory.

**M10 (CONJECTURE): the warm S-only switch on medium adapts within 6k.**
- S_2 already determines v_2 (M1), and the decoder reads S through the same fwd_read path the aux was trained on. Only the decoder's mapping "question + S → answer" must be learned. The step-0 S-only accuracy at d 2 is predicted near chance, since the decoder never read v_2 from S.
- This is why 6k (not 12k) suffices for a warm test. A fresh formation (Stage 4) keeps 12k + 6k.

## 4. Design: stages, arms, exact args

**One GPU class: RTX A6000 (48 GB) for every cell.** p2m_s1b and askm_lr1e4 were trained on A6000 (Note 258-2 pinned the class), so the warm starts stay on their class. The basin-flip lesson (Note 258-5) forbids mixing classes inside a comparison. The small comparator SO-WS is the one cross-class warm start (aw_plaw2_s1 was a 3090 run); it is descriptive, and the class change is disclosed. Price assumption: **$0.42/h** (b397's A6000 dph_total 0.4167, CURRENT_TASKS 11-01 22:54); the live dph replaces it. Local GPUs are not used (box RAM faults, open 11-03).

### Stage 0 ($0; pre-stated reads; each needs push → Audit ack before it is computed; descriptive, no binding effect)

- **R0a LAG-ECHO:** p2m_s1b `eval_ws.json` by_d['2'][t] for t = 3..8 (stored, n 1,000).
  - Does the ':' head show v_2 once a pass reads S_2? Prediction: WS pass 3-8 ≥ 0.90 (LAG-ECHO: the head reports the read, not the pass's computation) with P 0.55; all ≤ 0.20 (NO-LAG) with P 0.35.
  - Routing: interpretation only. The Stage 1 / 4 gates are fixed now.
- **R0b:** p2m_s1b `eval_swap_1.json` by_k['2'] and ['8'] (text decoder; follow_ca / follow_dfin / stay_own).
  - Prediction: follow_ca(K' 2) ≥ 0.90 via decoder recompute. Non-diagnostic for pass 2 under the text decoder (M3 does not apply). It is the text-decoder reference for SO-W's swap.

### Stage 1 (the cheap PARK stage; ~$2.5): does the S-only decoder make medium's pass 2 compute the hop?

| arm | init | dec_src | steps | role |
|---|---|---|---|---|
| **SO-W** | p2m_s1b/model.pt (`--init_model`, sha asserted) | cue | 6k | **binding: GO / PARK** |
| SO-WS | aw_plaw2_s1/model.pt (`--init_model`) | cue | 6k | descriptive comparator (does the S-only switch break a loop that already formed?), run in parallel |

**Exact args, SO-W.** p2m_s1b's args.json with only the listed diffs. The args diff vs p2m_s1b is checked by script before admission (single-change rule). Every other flag is unchanged: fast_add, cache_lower, micro 2 pinned, mem_budget 21, mul_s_p 1, γ = 1 from step 0.

```
python -B boxes/inject_aw.py --arm inject --base gpt2-medium --think_k 8 --d_max 2 --steps 6000 --bs 64 --lr 1e-4 --seed 1 \
  --micro 2 --mem_budget 21.0 --cache_lower --fast_add --aux_step 1.0 --aux_fade 48000,48000 --aux_reweight --aux_target value \
  --cue step --write_src answer --write_mix_start 0 --write_mix_end 0 --mul_s_p 1 --ckpt_every 1000 --log_every 100 --eval_bs 250 \
  --init_model boxes/runs_fsq/p2m_s1b/model.pt --init_model_sha <sha8 of p2m_s1b/model.pt> --dec_src cue \
  --eval_parts acc,ws,swap_1,swap_1f --out boxes/runs_fsq/so_w_s1
```

- **Diff vs p2m_s1b:** init_trunk → '' (init_model carries every weight; calibration skipped by design, inject_aw 565); init_model (+ sha); dec_src cue (new); steps 12000 → 6000; stop_at 6000 → 0; keep_final_ckpt → off; eval_parts.
- The schedule is a fresh 300-step warmup + cosine over 6k (a warm restart; disclosed).
- **SO-WS:** aw_plaw2_s1's args with the same diffs (base gpt2, micro 1 pinned, `--init_model boxes/runs_fsq/aw_plaw2_s1/model.pt`, out so_ws_s1).
- **Step-0 eval (both arms, pre-stated, descriptive):** `inject_aw_eval` on the init under dec_src cue, parts acc,ws.
  - Predictions for SO-W: d 1 acc > 0.5 (P 0.6); d 2 acc K' 8 < 0.30 (P 0.7).
  - WS equals p2m_s1b's eval_ws per item, since the think inputs are unchanged (a consistency gate, G-DS3).

**Conditional CTRL-W** (only on SO-W GO; ~$1.8): SO-W's exact args with `--dec_src text`, out ctrl_w_s1.
- It is the exposed control: the same 6k warm restart with the shortcut still available.
- Under the text decoder its SWAP2 is non-diagnostic (M3 does not apply). It is read on FORM-WS and NOT-DEC. If CTRL-W also reaches the SO-W label, "the S-only decoder was needed" is not claimed, but the GO stands, because S-only is sufficient.

### Stage 2 (build + port gates, GO only; ~$0.3 on the host)

- Build `trunk_qwen3`, `deltamem_q`, `numread`, `inject_q`, `inject_q_eval` and `so_parts` (§5). Run every §6 gate on an A6000 host before any training: fp64 gates, G-TIME, mem_check with micro pinned, the resume gate, and the host Python 3.11 smoke.
- Weights come from `Qwen/Qwen3-0.6B-Base` at a pinned revision. The safetensors sha256 is asserted on every load, as for medium (s5.py 114-151).
- READY-R1 = every gate passes. A fail gets one fix and a re-run; a second fail → PORT-FAIL → PARK (the build is reported).

### Stage 3 (ASK-Q askname init, GO only; ~$0.5)

```
python -B scratch/s19_askname_q.py --base qwen3-0.6b --d_max 2 --steps 8000 --bs 64 --lr 1e-4 --seed 1 --micro <pinned> \
  --mem_budget 44.0 --eval_parts acc,st_classes --out boxes/runs_fsq/askq_lr1e4
```

- s19_askname_q.py = the s258_askname.py semantics: P-DIR2's item stream, d ~ U{1, 2} at seed + 22331, EXCL, the four ask_render.ask_of classes, the target v_X.
- It runs on inject_q `--arm direct` with the Qwen tokenizer.
- The house make_opt / lr_at apply (300-step warmup, cosine to 8k; fsq120 731 / 740).
- **R2 gate:** ST classes i-iv pooled ≥ 0.98 each AND TEST2 d 1 ≥ 0.98 (as Note 258-3).
- **Fallback:** if R2 fails or n_nonfinite > 0, one arm at lr 3e-5 (askq_lr3e5) runs. If it passes, P-2-Q uses lr 3e-5, and the 258-8 confound is disclosed. If both fail → ASK-FAIL → PARK.
- **R3 INSTALL smoke** (300 steps of the P-2-Q args, `--stop_at 300`): n_nonfinite 0; write_grad_norm > 0; read_ratio ∈ (0, 0.5]; max pairwise head-state cosine < 0.99; mem_check peak ≤ 43 GiB at the pinned micro.

### Stage 4 (P-2-Q seed, GO + READY only; ~$5.7 + $2.8 extension)

```
python -B boxes/inject_q.py --arm inject --base qwen3-0.6b --think_k 8 --d_max 2 --steps 12000 --bs 64 --lr 1e-4 --seed 1 \
  --micro <pinned> --mem_budget 44.0 --cache_lower --fast_add --aux_step 1.0 --aux_fade 48000,48000 --aux_reweight --aux_target value \
  --cue step --write_src answer --write_mix_start 0 --write_mix_end 0 --mul_s_p 1 --dec_src cue \
  --init_trunk boxes/runs_fsq/askq_lr1e4/model.pt --init_sha <sha8> --ckpt_every 1000 --log_every 100 --eval_bs 125 --keep_final_ckpt \
  --eval_parts acc,ws_q,swap_1,swap_1f --out boxes/runs_fsq/p2q_s1
```

- The flags are p2m_s1b's except base / init / dec_src / mem_budget / micro / eval_bs; that diff is checked by script.
- keep_final_ckpt is needed: the extension resumes from it, and a seed becomes the REG-stage init.
- **In-training WS-Q log:** `ws_readout_q` on WS_ITEMS (TEST2 d 1-2, the first 200 each, no_grad, never fed back) in each 1k-bin record, as `ws_readout` (inject_aw 675-676).
- **Extension (one, final):** `p2q_s1_ext` resumes the 12k ckpt with `--steps 18000` (the cosine is recomputed, disclosed as in Note 258-4b) and runs in a new directory with args.json + ckpt shipped (resume-extension rule). The trigger is in §7.2.

### Stage 5 (pre-named, NOT in this registration)

- **P2Q-REP:** seeds 2 and 3, each with its own ASK-Q seed (whole-lineage seeds), in a separate registration with its own guard (≈ $13 point / $17 guard).
- It runs only if Stage 4 is SEED-OK or SEED-CAUSAL. Any later C2 claim on Qwen needs ≥ 3 seeds (Note 269-2).

## 5. Minimal port path (default-preserving; GPT-2 path bitwise-gated)

### 5.1 Edits to existing modules (two files; defaults reproduce the current behaviour)

**boxes/inject_aw.py** (739 lines; line numbers verified this session):
- New `--dec_src {text, cue}`, default text, in argparse (`main` 211-252; e.g. beside `--write_src`, 234).
- Module-level `DEC_SRC = "text"`, set by `main()` right after `parse_args` (252).
- `tok_aw` (76-96) reads DEC_SRC. With cue, after the write_src block it sets `o["dec"] = [EOT] + ENC.encode(" " + cue_str(it, it["d"])) + SFX`. The line-90 assert `o["dec"] == o["think"] + SFX` becomes dec_src-conditional; with cue it asserts that no op-sentence token sequence of the item occurs in dec.
- Every `tok_aw(` call site picks the flag up through the module global, with no signature change: inject_aw 103, 351, 364, 373, 374, 482, 484, 485, 486, 569; inject_aw_eval 96, 231, 319, 386, 474, 503, 602, 642, 890, 1185, 1217, 1273. (The draft listed only five of these; the module-global fix covers all of them.)
- `inject_aw_eval.load` (1291-1294) and `run` (1297) set `AW.DEC_SRC = a.get("dec_src", "text")` from args.json.
- A runtime assert in an `InjectAW.dec_logits` override (base: inject_ow.py 86-90): when DEC_SRC == "cue", every item's dec equals the cue form.
- `read_mask` (inject_aw_eval 993-1043) assumes the text decoder; it asserts DEC_SRC == "text" if called.
- No other line changes. `build_aw` (109-125), its geometry rule (118) and G12 (124) are untouched.

**boxes/inject_aw_eval.py:**
- Register `swap_1f` in PARTS (36) and in the dispatch dict (`run`, 1316-1341). The part lives in a new `boxes/so_parts.py`.
- swap_1f = `part_swap_1` (271-279) plus floor_ca (S_1 := the mean of the donors' S_1, MEAN-1, as part_swap_k's floor, 255-268) and follow_v1don with n_sep (an echo of the donor's v_1, on recipients with v1_don ∉ {own, dfin, ca}). It reuses `swap_pairs` (217-232; SWAP_CAP 1000, seeds 23244 / 23245, line 197), `grab_states` (235-242) and `passes` (55-82).

**Not edited:** deltamem.py (lazy-import rule; Qwen code goes in deltamem_q.py), s5.py, nested170.py, inject_ow.py, deltas5.py, fsq120.py. Edits land only when no deploy is in flight (deploys ship the working tree).

### 5.2 New modules (Qwen only; none imported on the GPT-2 path)

- **boxes/trunk_qwen3.py:** plain torch. HF only loads weights and serves as the G-Q1 reference.
  - fp32 master ParameterDict; interface `w`, `embed`, `logits` (tied embed^T after model.norm), L 28, D 1024, H 16, KV 8, HD 128.
  - block(i, h, pos): x = RMS(h)·w_in; q (16 × 128), k / v (8 × 128); q ← RMS_128(q)·q_norm, k ← RMS_128(k)·k_norm, **then** RoPE (rotate_half, inv_freq = θ^(−2i/128), θ 1e6); k / v repeated ×2 (q heads 2j, 2j + 1 share kv j); SDPA causal, scale 1/√128; h += o W_oᵀ; h += W_down(silu(W_gate x') ⊙ W_up x') with x' = RMS(h)·w_post. RMS eps from the config (1e-6).
  - Explicit position_ids = arange(T) in every forward. No wpe anywhere. HF decoder layers are not called, which avoids the version skew in their return types.
- **boxes/deltamem_q.py:** DeltaMemQ mirrors DeltaMem.__init__ (deltamem.py 52-84), read_fixed (158-176) and write_seq (179-223), with these changes:
  - layer 14; 8 additive + 8 delta heads of 64;
  - k_mem / v_mem = k_proj / v_proj rows (8 kv heads × 128), each kv head split into memory heads (2j, 2j + 1);
  - q_mem = the q_proj rows of q heads 0, 2, …, 14, split the same way;
  - q_norm / k_norm applied per 128-dim source head before the split (learnable copies);
  - no biases;
  - additive heads = memory 0-7 (kv 0-3), delta heads = memory 8-15 (kv 4-7);
  - ln = RMSNorm ← layers.14.input_layernorm; ln_w = RMSNorm ← layers.27.input_layernorm (write source = block 27's output, pre-norm);
  - gate W = 0, α bias +5, β 0; **wo = 0**; s_add calibrated by the §228 rule (inject_aw 565-576); s_del 0.
  - FAST_ADD and the mask semantics as in write_seq.
- **boxes/numread.py:**
  - class token sequences c_v = tok(" The answer is: v")[len(tok(" The answer is:")):], with prefix stability asserted;
  - the restricted-trie score s(v) (§3 M5); acc = argmax_v s(v); CE = −s(ans);
  - ties broken by the lowest class index (deterministic).
- **boxes/inject_q.py:** `inject_aw.main` copied with asserted edits (the house mk_* pattern), plus InjectQ (overrides pass_aw / write / lower / dec_logits / aux_logits / colon_logits; never calls wpe_slice / gpt_lower / fwd_read / `_block`). Qwen specifics:
  - the leading token is `<|endoftext|>` (id asserted from the pinned tokenizer), the analogue of GPT-2's EOT 50256;
  - SFX_Q = tok(" The answer is:") + [space]; the write token = the space token (the pass_aw COLON assert becomes `x[pos] == SPACE_ID`);
  - the answer loss is the full-vocabulary sequence NLL over c_ans (LB_v1 §5), with the first-digit CE at the write / decoder position plus the "1" → "0" continuation for 10 and the end-mass term for 1 (the same s(·) as numread);
  - the aux prompt is [EOT, " Step", " ", t, " :"] with the target index on digit tokens only;
  - `--dec_src` as in inject_aw.
- **boxes/inject_q_eval.py:** the inject_aw_eval parts acc / ws_q / swap_1 / swap_1f / lp5h on InjectQ. ws_q = WS-Q exact (§3 M5) per pass t, d, from one extra no-grad forward on x_t + ["1"] reading S_{t−1}.
- **scratch/s19_askname_q.py:** the s258_askname.py semantics on inject_q `--arm direct`, writing eval_st_classes.json ({'pooled': {i..iv: {'acc', 'n'}}}, the Note 258-1 (2) format).

## 6. Gates (Experiments builds; Audit G-D before any spend)

**GPT-2 path, p = 0 equivalence (Stage 1 needs only G-DS0 to G-DS3):**
- **G-DS0:** `--dec_src text` (default) is bitwise equal to the pre-change inject_aw (the step-50 model + AdamW state, resume 20 → 40) on p2m_s1b's args at `--sched_scale` small and on bun_s1's args. eval_acc and eval_swap_1 are byte-equal on a 50-item subset.
- **G-DS1:** `swap_1f` on a dec_src-text model reproduces `swap_1`'s follow_ca / follow_dfin / stay_own exactly.
- **G-DS2:** dec_src cue fixtures:
  - 20 decoded items: dec = EOT + question + SFX, with no op sentence;
  - gradflow: the decoder CE reaches the write params through S, and no op-sentence token is in the decoder graph;
  - the generator check for A2: chi-square of v_d against the question's name / unit on 10k stream items is not significant at 0.01.
- **G-DS3:** SO-W's step-0 eval_ws equals p2m_s1b's eval_ws per item (the think inputs are unchanged).

**Qwen path:**
- **G-WSQ0:** numread / ws_q on GPT-2 equals ws_readout bitwise (§3 M5).
- **G-Q7:** inject_q `--base gpt2` (GPT-2 trunk through the same InjectQ code with the GPT-2 tokenizer) equals inject_aw bitwise (step-50 + AdamW, resume) on P-2 args. This ties the copied main to the original.
- **G-Q1:** trunk_qwen3 vs HF Qwen3ForCausalLM (eager attention, fp64) on 16 OW items: max|Δlogit| / max|logit| ≤ 1e-10.
- **G-Q2:** right-padded batch = per-item unpadded forwards on real positions (fp64, ≤ 1e-10); lower(14) + upper = full at d 1 / 2 (fp64, ≤ 1e-10).
- **G-Q3:** install no-op at step 0: max|Δlogits| = 0 vs the plain trunk in decoder and aux forwards.
- **G-Q4:** mapping asserts (memory q / k / v rows == the mapped q_proj / k_proj / v_proj slices; the q_norm / k_norm copies: the G12 analogue). Head distinctness: max pairwise cosine of the per-head k-weight blocks < 0.9, and pairwise state correlation after one write on 64 items < 0.9.
- **G-Q5 (tokenizer and readout fixtures, pre-spend):**
  - the Qwen3 tokenizer's sha is recorded;
  - the 40 OW names single-token;
  - " 0".." 9" = [space, digit], " 10" = [space, "1", "0"];
  - SFX_Q ends in the space token;
  - aux vidx covers digit tokens only;
  - s(·) on hand-built logits for 1 vs 10;
  - a nonzero CE gradient at the second-digit position;
  - an untrained-memory model gives LP2 ≈ majority (not ≈ 1 from reading the space token).
  - If any name is multi-token, the build reports it; names matter only through the cue.
- **G-Q6 gradflow:** gradflow_check per term: answer CE → trunk + memory; aux → write k / v / gate / ln_w / q_norm / k_norm, after wo is perturbed to N(0, 1e-3).
- **G-RES:** resume bitwise (ckpt at 20 → 40 vs straight 40) for inject_q, plus a mirror round-trip of one ckpt (§12).

**Smokes and resources:**
- the plain-path smoke (`--arm direct`, 1 step) for gpt2, gpt2-medium and qwen3-0.6b;
- mem_check with micro pinned (≥ 1 GiB headroom) at the SO-W shape, the ASK-Q shape and the P-2-Q shape (d 2, K 8);
- G-TIME on the A6000 before each admission;
- the host Python 3.11 smoke (no 3.12+ APIs);
- transformers ≥ 4.51 on hosts only for G-Q1 and loading;
- the step-0 read_ratio = ‖fade·wo(o)‖ / ‖h‖, logged per 100 steps in R3 and Stage 4 (the zero-init usage monitor).

## 7. Pre-registered criteria

**Statistics, every binding proportion:** a Wilson 95% CI with its n. **NEAR** = the bar lies within 2 SE of the reading (Audit 11-03 rule); the label gets the suffix "-NEAR". For routing (spend), a NEAR reading is decided by its point value; a NEAR reading never supports a claim. Raw values only, compared to the bars as registered. One seed per stage, one lineage.

### 7.1 Stage 1 (SO-W binding; SO-WS / CTRL-W descriptive comparators read with the same fields)

| ID | rule | field (n) |
|---|---|---|
| ACC2 | d 2 acc at K' 8 ≥ 0.90 | eval_acc by_d['2'].acc_kprime['8'] (1,000) |
| NOT-DEC | d 2 acc at K' 1 ≤ 0.20 | by_d['2'].acc_kprime['1'] (1,000) |
| SWAP2 | follow_ca at K' 2 ≥ 0.90 AND ≥ floor_ca + 0.50 | eval_swap_1f by_k['2'].follow_ca / floor_ca (n_pairs ≤ 1,000) |
| AUX2 | the last 1k bin's aux d 2 pass 2 ≥ 0.90 | log.jsonl aux_vacc_by_d_pass['2']['2'] (bin n) |
| **FORM-WS** | WS d 2 pass 2 ≥ 0.90 at 6k (the formation gate; the 4k / 5k log bins on 200 items are the pre-stated ≥ 4k trajectory) | eval_ws by_d['2']['2'] (1,000); log ws_readout['2']['2'] per bin |
| COV | args dec_src == "cue" and eval meta dec_src == "cue" | args.json, eval_*.json meta |

**Labels (first match):**
1. COV-FAIL (void → VOID-RERUN);
2. NO-READ (ACC2 fails);
3. EAGER-P1 (NOT-DEC fails);
4. ECHO (SWAP2 fails with follow_v1don(K' 2) ≥ 0.5 on n_sep ≥ 100);
5. LATE-HOP (SWAP2 fails AND follow_ca(K' 8) ≥ 0.90);
6. NO-PASS2 (SWAP2 fails otherwise);
7. AUX-ONLY (AUX2 fails);
8. **S11-PASS** (all pass, FORM-WS included);
9. **SEED-CAUSAL** (all but FORM-WS pass).

**Decision (pre-stated):**
- **GO** iff S11-PASS or SEED-CAUSAL. Both show that pass 2 computes op 2 from S_1 under the S-only decoder (M3).
- **PARK the larger-base line** on any other label. Qwen is not built; the route is M8 TEXT-DROPOUT / M2 COLON-CE (with a causal, not WS, gate).
- **SO-WS reading:** if SO-WS fails with the same label as SO-W, the failure is attributed to the S-only switch itself, not to scale (no "scale-specific" wording). If SO-WS passes and SO-W fails, the wording is "S-only does not repair medium's pass 2 in 6k (1 seed)".
- **CTRL-W reading (GO only):** if CTRL-W reaches FORM-WS and NOT-DEC, the decoder change is not shown necessary (disclosed); the GO stands.

### 7.2 Stage 4 (P-2-Q, binding)

- **Fields as §7.1**, on p2q_s1 (and on p2q_s1_ext if extended). FORM-WS uses **eval_ws_q** by_d['2']['2'] (exact WS-Q, n 1,000).
- **Formation trajectory (pre-stated, descriptive):** the log ws_readout_q['2']['2'] and aux_vacc_by_d_pass['2']['2'] in each 1k bin from 4k to 12k (n 200). Onset t* = the first bin with WS-Q ≥ 0.50; a separate aux onset uses the same rule. No decision is taken before 12k (FD1: no 2k / early marker; medium's aux was 0.14 at 8k and 1.00 at 10k).
- **Extension rule (one +6k to 18k; final):** EXTEND if any of:
  - 0.50 ≤ WS-Q(12k) < 0.90;
  - WS-Q(12k) − WS-Q(6k bin) ≥ 0.10;
  - aux bin(12k) ≥ 0.50 with WS-Q(12k) < 0.90.
  The same gates are read at 18k.
- **Labels:** COV-FAIL / NO-READ / EAGER-P1 / ECHO / LATE-HOP / NO-PASS2 / AUX-ONLY / **SEED-OK** (all, incl. FORM-WS) / **SEED-CAUSAL** (all but FORM-WS).
  - A run that reaches neither by its last step is **NO-SEED** (censored = failure, §10), with its label recorded.
- **Decision:**
  - **SEED-OK** → the Stage 5 registration may be drafted, and a Qwen REG d ≤ 4 design may use p2q_s1 as init.
  - **SEED-CAUSAL** → the same, but any later Qwen gate on per-pass computation uses aux plus the causal swaps (Note 258-9's lead), not WS.
  - **NO-SEED** → the Qwen line PARKS, worded "not formed by 18k on one seed (S-only decoder, askname init, lr 1e-4)". A per-seed formation rate ≤ 0.5 is not excluded (§9).

### 7.3 Stages 2-3 (build checks; a fail means one fix plus one re-run, not a science null)

| ID | rule | field |
|---|---|---|
| R1 PORT | G-Q1 to G-Q7, G-WSQ0, G-RES all pass | gates_pass.json booleans + max_rel_err |
| R2 ASK-Q | ST classes i-iv pooled ≥ 0.98 each AND TEST2 d 1 ≥ 0.98 | eval_st_classes.json pooled[c].acc; eval_acc by_d['1'].acc (1,000) |
| R3 INSTALL | the 300-step smoke rules of §4 Stage 3 | smoke log.jsonl |

- **Labels:** READY = R1 ∧ R2 ∧ R3; else PORT-FAIL / ASK-FAIL / INSTALL-FAIL. A second fail → PARK.

## 8. Predictions (stated before any build or output)

| # | prediction | P |
|---|---|---|
| P0a | R0a LAG-ECHO (p2m_s1b WS d 2 passes 3-8 ≥ 0.90) | 0.55 |
| P1 | SO-W step-0 d 2 acc (K' 8, cue decoder) < 0.30; d 1 > 0.50 | 0.70; 0.60 |
| P2 | SO-W ACC2 at 6k | 0.75 |
| P3 | SO-W NOT-DEC (pass 1 does not compute both hops) | 0.75 |
| P4 | SO-W SWAP2 | 0.55 |
| P5 | SO-W FORM-WS | 0.30 |
| P6 | SO-W label: S11-PASS 0.25 / SEED-CAUSAL 0.25 / EAGER-P1 0.15 / LATE-HOP 0.10 / NO-READ 0.10 / ECHO + NO-PASS2 + AUX-ONLY 0.15 → **GO 0.50** | — |
| P7 | SO-WS (small) stays S11-PASS | 0.65 |
| P8 | CTRL-W (text decoder) reaches FORM-WS | 0.10 |
| P9 | READY (R1 0.90 × R2 0.85 × R3 0.90, after one fix each) | ≈ 0.70 |
| P10 | P-2-Q SEED-OK / SEED-CAUSAL / NO-SEED (given GO and READY) | 0.25 / 0.20 / 0.55 |
| P11 | P-2-Q aux onset at 4-10k, given it forms by 18k | 0.60 |
| P12 | A text-decoder Qwen port would NO-SEED (M8; not run) | 0.65 |

**Overall:** P(a Qwen seed, SEED-OK or SEED-CAUSAL) ≈ 0.50 × 0.70 × 0.45 ≈ **0.16**. P(PARK at Stage 1) ≈ 0.50.

## 9. OC block (seed-rate, at this panel's seed count = 1 per stage)

**Item-level noise.** The bars are absolute at n 1,000 (TEST2 d 2; WS 1,000; swap n_pairs ≤ 1,000). At 0.90 the Wilson half-width is ≈ 0.019 (2 SE ≈ 0.019, so NEAR spans ≈ 0.88-0.92).

**P(pass | no effect):**
- **SWAP2 under no pass-2 computation:** follow_ca is at the MEAN-1 floor (≈ 1/11 for an echo or an S-ignoring model). P(observed ≥ 0.90 | p 0.10, n 1,000) < 1e-100.
- **FORM-WS under the echo:** WS ≈ 0.015-0.10, so P(≥ 0.90) ≈ 0.
- **P(false GO)** ≈ 0. Under an S-only decoder no text route exists (M3), so the only way to pass SWAP2 is the pass-2 hop.

**P(pass | effect):**
- With true follow_ca 0.95 and WS 0.95: P(pass both) ≈ 1 − 2·Φ(−7.3) ≈ 1.
- With a marginal true effect of 0.91: P(pass) ≈ 0.70, and it carries -NEAR.

**Seed-rate.** One seed per stage cannot estimate a per-seed formation rate. If the true per-seed rate is r, P(NO-SEED | r) = 1 − r, i.e. 0.5 at r = 0.5. So:
- **NO-SEED / PARK wording** is restricted to "not formed on one seed in 18k". A rate ≤ 0.5 is not excluded (the §258 precedent: Note 258-8 wording).
- **A SEED claim** is "one seed, one lineage". The exact Clopper-Pearson 95% lower bound on r after 1/1 is 0.025. Stage 5 (2 more seeds) gives 3/3 → lower bound 0.29.
- **Stage 1** is a warm continuation of one formed-aux run. Its pass / fail speaks to that model only. The PARK it triggers is a cost decision under P8 / P12, not a claim that S-only cannot work at scale.

**Realised $** is reported per stage against §12's points and guards.

## 10. Censoring (conservative; pre-stated)

- **A crashed or host-lost cell** gets one VOID-RERUN on the same GPU class, from the last off-host ckpt (G-RES tested). If it is still unreached, the cell FAILS the gates it feeds (Stage 1 → PARK; Stage 4 → NO-SEED). Missing cells are never imputed.
- **Non-finite runs** (n_nonfinite > 0 in any bin) count as FAIL. A non-finite ASK-Q arm triggers the 3e-5 fallback.
- **A formation event not reached by the run's last step** (12k, or 18k after extension) is a failure (NO-SEED), not missing. The event-only mean onset is reported beside it, as descriptive.
- **No TEST2 item is dropped after the fact.** Swap pairs are filtered only by `swap_pairs`'s pre-existing rule (217-232). The ECHO subset (v1_don ∉ {own, dfin, ca}) is defined now; its n_sep is reported, and ECHO needs n_sep ≥ 100.
- **WS-Q ties** take the lowest class index (deterministic, the same in every arm).
- **Aborted logs and run dirs** are kept (renamed *.void), never deleted.

## 11. Seeds and scope

- **Stage 1:** seed 1. Inits p2m_s1b (lineage askm_lr1e4 → p2m_s1b) and aw_plaw2_s1 (s233_askname_s1 → aw_plaw2_s1).
- **Stage 3 / 4:** seed 1 (askq_lr1e4 → p2q_s1). The d-sequence law is `default_rng(seed + 22331)`, as p2m_s1b.
- **Stage 5 (separate registration):** seeds 2 and 3, each with its own ASK-Q seed (whole-lineage seeds, the §268 rule). It is needed before any Qwen claim beyond "one seed formed".
- **Scope of any positive result:** one lineage; d ≤ 2; plan given (cue step); S-only decoder; chain-order text; REG frames 0-5; RTX A6000.

## 12. Cost, guards, off-host checkpoints

**Rate basis.**
- A6000 at $0.42/h (assumption; b397 dph_total 0.4167). G-TIME and the live dph replace every rate before admission.
- Medium at the d ≤ 2 P-2 shape: p2m_s1b cost $3.554 for 12k incl. two evals and setup, ≈ $0.28 per 1k steps.
- Qwen per step ≈ 1.6 × medium (non-embedding params 0.44B / 0.30B ≈ 1.45; think length ≈ 1.09x, from Qwen2.5's 48.96 tokens at d 2 in lb_tok_check). This is ASSUMED until G-TIME.

| item | basis | point | guard |
|---|---|---|---|
| Stage 0 | $0 reads | $0 | $0 |
| **Stage 1: SO-W** | 6k × $0.28/1k + evals (incl. step 0) $0.12 | $1.80 | $2.35 (×1.3) |
| Stage 1: SO-WS (parallel) | small d ≤ 2 ≈ 0.75 s/step (LB cost model pilot_d2 0.72) × 6k on A6000 + evals | $0.65 | $0.85 |
| Stage 1b: CTRL-W (GO only) | as SO-W | $1.80 | $2.35 |
| Stage 2: build gates on the host | download 1.2 GB, fp64 G-Q1/2, G-TIME, mem_check, G-RES ≈ 0.7 h | $0.30 | $0.50 |
| Stage 3: ASK-Q lr 1e-4 | 8k × ≈ 0.40 s/step (medium 0.252 × 1.6) ≈ 0.9 h + eval + R3 smoke | $0.50 | $0.70 |
| Stage 3: ASK-Q fallback lr 3e-5 (conditional) | as above | $0.50 | $0.70 |
| **Stage 4: P-2-Q 12k** | 12k × $0.45/1k + WS-Q double-forward evals and swaps $0.20 + mirrors $0.10 | $5.70 | $8.00 (×1.4, rate uncertain) |
| Stage 4: extension 6k (conditional) | 6k × $0.45/1k + eval | $2.80 | $3.90 |
| **family** | | **≈ $14.1 if every branch runs** | **cap $20** (sum of guards $19.35) |

- **Expected spend:**
  - Stage 1 is $2.45, always.
  - With P(GO) 0.5: CTRL-W 1.80 + build 0.30 + ASK-Q 0.50 + 0.15 × 0.50 fallback, then with P(READY) ≈ 0.8: P-2-Q 5.70 + 0.35 × 2.80 extension.
  - Total ≈ 2.45 + 0.5 × 8.0 ≈ **$6.5**. Worst case is the $20 cap. This is under the user's $30 per-design line (Audit auto-approve rule, user 10-17, reaffirmed 10-26).
- **Projection stops:**
  - If G-TIME projects P-2-Q 12k above $7.5 point (> 5.3 s/step at $0.42/h), Stage 4 is not admitted; it is re-planned with Audit (e.g. micro / cache, or a 24 GB class only if a whole new design pins it).
  - If any cell's live cost passes its guard, it stops at its last ckpt.
- **Guards:** per-cell cost guards at the guard column; wall guards at dph_total; GUARD_MARKS reaching the final step (guard-marks rule); family guard $20.

**Off-host checkpoints (user 10-15 rule):**
- Every Stage 1 and Stage 4 cell has `--ckpt_every 1000` and `ckpt_mirror.py <n>:<run> --optional log.jsonl,args.json`.
  - Medium cells mirror every 0.36 h (as §258).
  - P-2-Q mirrors every 1.0 h: its ckpt holds fp32 params + AdamW ≈ 0.6B × 12 B ≈ 7.2 GB, which is heavy to mirror more often. Worst-case loss is about 1 h ≈ $0.42.
- Mirrors land on D: (never /tmp on C:). The resume from a mirrored ckpt is tested once in G-RES before Stage 4 admission.
- keep_final_ckpt only on P-2-Q (the extension and later-init needs). Its ckpt_final is deleted after the extension or the scoring is verified. SO-W / CTRL-W / SO-WS keep model.pt only until their scoring is reproduced.

## 13. Scorer sketches (committed with fixtures before any output)

- **theory/score_so.py** (Stage 1 and Stage 4; `--stage S1 | S4`).
  - Reads: eval_acc by_d['2'].acc_kprime['1'/'8'] (+ n); eval_swap_1f by_k['2' / '8'].{follow_ca, floor_ca, stay_own, follow_dfin, follow_v1don, n_sep, n_pairs}; eval_ws / eval_ws_q by_d['2']['2'] (+ n); log.jsonl's last-bin aux_vacc_by_d_pass['2']['2'] and the 4k-12k bins (ws_readout / ws_readout_q, aux) for the trajectory and t*; args / meta dec_src.
  - Outputs: Wilson CIs, NEAR, the label, GO / PARK (S1), the extension trigger and SEED label (S4), and the CTRL-W / SO-WS descriptive rows.
  - **Fixtures (14):** one per label (9 for S1, incl. COV-FAIL and both GO labels); the extension trigger by each of its three clauses and a no-trigger case; a NEAR case; a censored (missing 12k eval) case → NO-SEED.
- **theory/score_q_ready.py:** reads gates_pass.json, askq eval_st_classes / eval_acc and the smoke log.jsonl (n_nonfinite, write_grad_norm, read_ratio, head_cos, mem peak). **Fixtures (5):** one per label plus a NaN log.
- **The field-path check** (grep each module for every key a scorer reads) is part of G-D. The keys that are new here: swap_1f's follow_v1don / n_sep / floor_ca; ws_readout_q; eval_ws_q; the meta dec_src.

## 14. Unresolved items (for the red team and Audit)

1. **Is pass-2 WS the right mechanism marker beyond small?** The 258-9 dissociation and M2 suggest that at medium / Qwen the ':' LM head may report the read (LAG-ECHO), not the pass's computation, even when the pass computes the hop (M1). This design keeps FORM-WS binding, as instructed, but lets SEED-CAUSAL (causal SWAP2 without WS) carry a GO. Should SEED-CAUSAL count, or should the formation gate move to the causal swap entirely?
2. **LATE-HOP.** The S-only decoder plus a repeated cue lets passes 3..8 do the hop (M4). Only the per-pass aux places it at pass 2. Is LATE-HOP a failure for the later REG d ≤ 4 stage? Pre-stated here as "not a seed".
3. **The warm-to-fresh gap.** Stage 1 is a warm repair on a model whose S_2 already holds v_2. A Stage 1 GO does not show that S-only lets a write form from scratch. Stage 4 is the first fresh test, on Qwen, without a medium fresh arm (SO-F deferred).
4. **The Qwen3 tokenizer** is assumed to be Qwen2.5's BPE (digit splitting, 40/40 single-token names). G-Q5 asserts this before spend; if it differs, the readout fixtures change and the build reports it.
5. **The A6000 price and the Qwen speed** ($0.42/h, 1.6x medium) are assumptions. G-TIME and the projection stop guard them. 48 GB-class availability on Vast may force a wait; one class per design forbids switching mid-family.
6. **Massive-activation dims in Qwen residuals** could dominate the RMS-normalised write source (ln_w). G-Q4 and read_ratio monitor this; there is no pre-registered fix.
7. **SO-WS crosses GPU class** (a 3090-trained init on an A6000). It is descriptive only.
8. **The lr choice** (1e-4, not the §258 tie-break) is a deliberate deviation based on Note 258-8. It is disclosed.
9. **SC2 SPI** (the shortcut-propensity index across 5 bases) stays deferred. If the Qwen line seeds, a 3-cell Qwen SPI could quantify M8 cheaply (~$0.3), as a separate registration.

## 15. Red team

Not red-teamed yet. A separate sub-agent should review in particular: §3 M3's identification argument, the GO rule's use of SEED-CAUSAL, the cost basis (§12) and the G-DS0 / G-Q7 coverage of every tok_aw call site.

## 16. Verified arXiv ids (fetched 2026-11-03 for the draft; titles matched then; not re-fetched)

- 2505.09388: Qwen3 Technical Report.
- 2304.01373: Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling (SC2, deferred).
- 2604.04902: Are Latent Reasoning Models Easily Interpretable? (latents often unused).
- 2004.07780: Shortcut Learning in Deep Neural Networks.
- 2006.07710: The Pitfalls of Simplicity Bias in Neural Networks.
- 2402.16837: Do Large Language Models Latently Perform Multi-Hop Reasoning?
- 2406.12775: Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries.
- 2405.15071: Grokked Transformers are Implicit Reasoners.
- 2305.13245: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.
- 2104.09864: RoFormer: Enhanced Transformer with Rotary Position Embedding.
- 2302.05442: Scaling Vision Transformers to 22 Billion Parameters (QK-norm).
- 2412.06464: Gated Delta Networks: Improving Mamba2 with Delta Rule.
- 2608.11233: Retrofitting Recurrent Depth into a Pretrained Language Model (base: Qwen2.5-0.5B-Instruct).
- 2506.18582: Parallel Continuous Chain-of-Thought with Jacobi Iteration (the backlog §10 GSM8K table).
