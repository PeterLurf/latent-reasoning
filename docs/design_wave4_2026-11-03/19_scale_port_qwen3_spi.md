# 19 SCALE LANE: SC2 SPI + S11 S-ONLY DECODER (medium) + S10 QWEN3-0.6B PORT (design, 2026-11-03)

**Status:** lead design, registration-ready draft. Nothing built, run or read. Read-only on the repo. No stored eval output or checkpoint was read; the only stored files opened were `args.json` of p2m_s1b / askm_lr1e4 (recipe flags).
**Backlog IDs:** SC2 (S15) SHORTCUT-PROPENSITY INDEX; S11 S-ONLY DECODER ON MEDIUM; S10 QWEN3-0.6B PORT + READINESS R1-R4 (§12.7, §13.0 "Scaling"); absorbs M3's S-only half; pre-names S12.

## 1. Goal

**Why one family.** Note 264-1: on GPT-2 medium (p2m_s1b, lr 1e-4) the decoder computes the last hop from v_1 + text. The K = 8 swap gives follow_ca 1.00 and follow_dfin 0.00, and pass 2's ':' echoes v_1. Note 258-9 adds that the aux decodes v_2 from S_2 at 1.00, and d 2 at K' 1 is 0.974.

**Correction to the backlog framing.** The decoder last hop is not medium-specific.
- Small's P-2 also carries the DEC-STEP2 label (Note 234-4: swap follows cf_a 1.000), and the backlog's "halt is effectively at d−1" says the same thing.
- What separates medium from small at d 2 is pass 2 itself: small's pass-2 ':' shows v_2 (WS ≈ 1.0, aw_plaw2_s1), while medium's echoes v_1 (WS 0.015) even though v_2 is aux-decodable.
- So S11 needs a **small S-only arm** to say anything about scale, and SC2 must predict WS formation, not "the decoder shortcut".

Three questions decide the scale recipe:

| ID | Question | Decision it drives |
|---|---|---|
| SC2 | Does a base's memory-free one-pass hop capacity (SPI) order the bases the way the observed shortcut does (medium > small)? At which depth d* does one-pass learning fail? | the shortcut guard and the starting d_max for each larger base (S12 Qwen, SC4 Pythia) |
| S11 | With the text removed from the decoder (S-only), does medium's pass 2 compute the hop? Warm (p2m_s1b), then fresh (askm_lr1e4) | the larger-base recipe: askname → P-2 with an S-only decoder, or a two-stage switch, or M8 / M2 instead |
| S10 | Can Qwen3-0.6B-Base host the loop exactly (equivalence), learn the askname foothold, and run the latent loop finitely? | whether S12 (Qwen P-2 with guard, ~$4-5) may register |

**Order:** SC2 ∥ S10 build → S11 stage 1 → (S11 stage 2 ∥ S10 cloud) → S12 registration (separate).

## 2. Design

### 2.1 SC2: SHORTCUT-PROPENSITY INDEX

**Bases:**

| base | L / D / H (KV) | notes |
|---|---|---|
| GPT-2 small | 12 / 768 / 12 | |
| GPT-2 medium | 24 / 1024 / 16 | |
| Pythia-160M | 12 / 768 / 12 | rotary_pct 0.25, parallel residual, untied |
| Pythia-410M | 24 / 1024 / 16 | rotary_pct 0.25, parallel residual, untied |
| Qwen3-0.6B-Base | 28 / 1024 / 16 (8) | head_dim 128, QK-norm, RoPE θ 1e6, tied, vocab 151,936 |

- **Sources:** configs fetched today from HF; the Qwen3 config lists transformers 4.51.0.
- **Pythia:** numbers are single tokens. Three names (Mia, Zoe, Fay) are multi-token, which is irrelevant for direct FT (lb_tok_check.json).

**Cell:** memory-free direct full FT from the raw pretrained base.
- Data: TOPO stream `OW.Stream(OW.SEED_TRAIN, EXCL, frames 0-5, topo, dag)`, d ~ U{1..4} per step (SEED_D law), bs 64, 2,000 steps, the house `make_opt` / `lr_at` (300-step warmup, cosine to 2k).
- **Format:** `dec = BOS + " " + text + " The answer is:"`, with the text ending in the final question. This is inject_aw `--arm direct`.
- **Grid:** lr ∈ {3e-5, 1e-4} (§258's S1 bracket, the same for every base) × seeds {1, 2, 3}. 5 × 2 × 3 = 30 cells, all on one GPU class (RTX 3090).
- **Why raw bases:** askname inits exist only for GPT-2.

**Readout (shared module `numread`, also used by S10).**
- Each class v ∈ Z_11 has a token sequence c_v = tok(" The answer is: v")[len(tok(" The answer is:")):]. Prefix-stability is asserted.
- **Score:** restricted-trie sequence score. s(v) = Σ_i log p(c_v,i | prefix) + 1[c_v is a proper prefix of some c_w] · log(1 − Σ_children p(child | c_v)).
  - For GPT-2 / Pythia (single distinct tokens) this is exactly the house full-vocabulary token log-prob.
  - For Qwen (c_7 = [Ġ, 7]; c_10 = [Ġ, 1, 0]), it is LB_v1's prefix-factorised read (REVIEW_LB #10 judged it sound). One forward on dec + [Ġ, "1"] gives every term. The read is causal, so the appended tokens never reach q₁.
- **Accuracy** = argmax_v s(v). **CE** = −s(ans) (REVIEW_LB fix 9d).

**Index.** acc(b, d) is read on TEST2 (1,000 items per d, d 1-4) at step 2,000.
- SPI(b, d) = (acc − maj_d) / (1 − maj_d), with maj_d the TEST2 majority share (≈ 0.10).
- **SPI2 = SPI(b, 2)** is primary.
- A base's value is the 3-seed mean at its better lr (max over the bracket, pre-stated: the propensity under any bracket lr).
- **Descriptive:** a learning-curve AUC from no-grad evals at steps 500 / 1,000 / 1,500 / 2,000, on fixed 500-item TEST2 subsets.

**Second index, UI (USE-INDEX).** The one-hop-from-memory capacity behind the observed route. It reuses the §234 G-AW1 FROZEN-USE harness (`docs/research_answerpos_2026-10-25/aw_frozen_use.py`):
- a frozen askname trunk, plus a single pass-1 ':' write S_1;
- only W_o (+ s_add / s_del) is fitted;
- the score is the v_2 test accuracy on d 2.
- Small's value is already known: **0.118 vs majority 0.107** (Note 234-2).
- **New cell:** medium's askm_lr1e4 (8 + 8 heads at layer 12). This is a copy with three asserted edits (INIT path + sha, `--base gpt2-medium`, geometry from `build_aw`), run on a cloud host for ≈ $0.05. Because it is a new computation on a stored checkpoint, it needs a pre-statement → push → Audit ack first.
- **Qwen:** UI(Qwen) on ASK-Q follows in S10 (descriptive).
- **Normalised index:** UI_n = (acc − maj) / (1 − maj).

**Mechanism link (stated before data).** The latent recipe gives two routes to the d-hop answer: the write chain, and one-pass recomputation in the decoder from text (+ v_{d-1}). SPI measures how fast the trunk learns the one-pass route from text alone. Under shortcut / simplicity-bias accounts (2004.07780, 2006.07710), the faster route captures the answer gradient. Prediction: SPI2(medium) > SPI2(small).
- **Known weakness, stated:** medium's observed shortcut is ONE hop from memory's v_1, not two hops from text. A FLOOR result (all bases ≈ 0) would say SPI does not capture it.

### 2.2 S11: S-ONLY DECODER on medium

**New flag `--dec_src {text, cue}`** (default text, which is bitwise current behaviour).
- With cue, `dec = [EOT] + enc(" " + q) + SFX`, where q = raw["question"] (the text's final "How many <u> does <S_d> have?").
- The decoder sees no op sentence. It reads S_K at block L/2 (fwd_read) as before.
- Think inputs, aux, cue schedule and K = 8 are unchanged.

**Arms** (one RTX 3090 class; base args = p2m_s1b args.json):

| arm | init | dec_src | steps | role |
|---|---|---|---|---|
| SO-W | p2m_s1b model.pt (`--init_model`, sha asserted) | cue | 6k | binding (repair) |
| CTRL-W | p2m_s1b | text | 6k | exposed control: more training, shortcut available |
| SO-WS | aw_plaw2_s1 (small P-2, its args) | cue | 6k | scale comparator: the same intervention where WS formed |
| SO-F (stage 2) | askm_lr1e4 (`--init_trunk`, as p2m_s1b) | cue | 12k (+6k extension rule) | binding (formation, the larger-base recipe) |

- p2m_s1b's recipe: d ≤ 2, aux 1.0, reweight, cue step, write_src answer, lr 1e-4, bs 64, fast_add, cache_lower, mul_s_p 1, γ = 1 from step 0.
- SO-F runs only if SO-W is not NO-READ.
- **Step-0 read (pre-stated, descriptive):** SO-W's init evaluated with dec_src cue before training. Prediction: d 1 > 0.5, d 2 near chance (the decoder never read v_2 from S).

**New eval part `swap_1f`** (S_1 := donor's S_1, then the recipient's own passes 2..K'). Pairs follow the `swap_pairs` law (d 2, seeds 23244 / 23245). Fields at K' ∈ {2, 8}:
- follow_ca (cf_a(1) = recipient op 2 on the donor's v_1);
- stay_own;
- follow_dfin;
- follow_v1don (an echo of the donor's v_1, read on the echo-separable subset v1_don ∉ {own, dfin, ca}, with n_sep);
- floor_ca (S_1 := the mean of the donors' S_1, i.e. MEAN-1).

With the S-only decoder, a follow_ca at K' 2 can arise only if pass 2 applied op 2 to S_1's value.

### 2.3 S10: Qwen3-0.6B-Base port + readiness

**Trunk adapter `boxes/trunk_qwen3.py` (plain torch; HF only loads weights and serves as the gate reference).**
- Weights come from the pinned safetensors revision into a ParameterDict (fp32 master). The interface mirrors GPT2KV: `w`, `embed`, `logits` (tied), `L = 28`, `D = 1024`, `H = 16`, `KV = 8`, `HD = 128`.
- **block(i, h, cs):**
  - x = RMS(h) ⊙ w_in;
  - q = x W_qᵀ → (B, T, 16, 128), k / v → (B, T, 8, 128);
  - q ← RMS_128(q) ⊙ q_norm, k ← RMS_128(k) ⊙ k_norm, **before** RoPE (confirmed in modeling_qwen3.py);
  - RoPE with rotate_half;
  - k / v repeated ×2 (GQA group = q heads 2j, 2j+1);
  - SDPA, causal, scale 1/√128;
  - h += o W_oᵀ; h += W_down(silu(W_gate x') ⊙ W_up x') with x' = RMS(h) ⊙ w_post.
- **Explicit positions:** every forward (think pass, aux, decoder, lower cache) uses position_ids = arange(T) from 0.
  - cos / sin come from inv_freq = θ^(−2i/128), θ = 1e6, computed once per T.
  - No wpe add anywhere. POS_OFFSET / KEEP-SINK evals are N/A (asserted).
  - Right padding with a causal mask leaves the real positions exact (gate G-Q2).
  - We do not call HF decoder layers: their return type and mask handling changed across versions (4.51 returns a tuple; main returns a tensor). The plain-torch block removes that skew.

**DeltaMemQ (`boxes/deltamem_q.py`; deltamem.py stays byte-identical because of the lazy-import rule).**
- Geometry: layer l = L/2 = 14; D / 128 = 8 additive + 8 delta heads of DH 64.
- **Mapping from block 14:**
  - **k_mem** = k_proj (8 kv heads × 128 = 1,024 rows), with each 128-dim kv head split into two 64-dim memory heads (mem head m ↔ kv head ⌊m/2⌋, half m mod 2);
  - **v_mem** = v_proj, the same way;
  - **q_mem** = the q_proj rows of the first q head of each GQA group (q heads 0, 2, …, 14) → 1,024 rows, split the same way, so q_m pairs with its own kv head;
  - bias 0 (attention_bias false);
  - q_norm / k_norm are copied as learnable per-128 RMSNorm weights and applied to the 128-dim source head before the 64-split;
  - **additive heads** = mem 0-7 (kv 0-3); **delta heads** = mem 8-15 (kv 4-7);
  - ln = RMSNorm ← layers.14.input_layernorm; ln_w = RMSNorm ← layers.27.input_layernorm (write source = the output of block 27, pre-final-norm);
  - gate W = 0 with α bias +5, β 0; **wo = 0**; s_add calibrated (§228 rule); s_del 0.
- **Why split rather than repeat:** splitting yields 16 distinct heads, avoiding LB's 7×-replication collapse (REVIEW_LB #4).
- **Disclosed:** RoPE partners (i, i + 64) land in different memory heads. At relative position 0 the pretrained q·k equals the sum of the two halves.

**Trainer and eval.** `boxes/inject_q.py` = inject_aw.main copied with asserted edits (the house mk_* pattern). The tokenizer is abstracted (`--base {gpt2, qwen3-0.6b}`), and `numread` reads the answers. The Qwen-specific pieces:
- the write token = the last token of SFX_Q = tok(" The answer is:") + [Ġ], i.e. the position whose LM head emits the first digit (the GPT-2 analogue of ':');
- the decoder input carries + ["1"] for q₂;
- aux vidx = every digit token of the value (REVIEW_LB 9a);
- the aux prompt is [BOS, " Step", Ġ, t, " :"].

`boxes/inject_q_eval.py` ports only acc / cue_only / swap_1f / swap_kd / lp5h / ws_q. ws_q is descriptive: the first-digit argmax, with 1 and 10 merged.

**Readiness:**
- **R1 PORT:** G-Q1–G-Q7 (§4).
- **R2 ASK-Q:** the s233 askname recipe (`scratch/s258_askname.py` semantics, `ask_render.ask_of` classes) on inject_q `--arm direct`, 8k steps, at SC2's better Qwen lr (by 3-seed TEST2 d 1; a tie takes the lower lr).
- **R3 INSTALL smoke:** 300 steps of P-2-Q (d ≤ 2, K 8, aux, dec_src cue) from ASK-Q.
- **R4 = SPI(Qwen)** from SC2.

## 3. Code touch points (read this session)

**boxes/inject_aw.py:**
- `tok_aw` 76-96: add the `dec_src` param. The line-90 assert `o["dec"] == o["think"] + SFX` becomes dec_src-conditional.
- `main` argparse 211-252 (+ `--dec_src`).
- `build_aw` 109-125: the line-118 geometry rule and the G12 assert at 124 stay as they are for GPT-2.
- Every `tok_aw(` call site: draw_items 316-347, tok_ask 103, MC_TR / MC_EV 447-450, WS_ITEMS 451, calibration 533.
- `loss_reg` dec_logits at 404 (unchanged).
- **Coverage trap:** inject_aw_eval builds items through `tokd` (95), `swap_pairs` (217-232) and `swap_pairs_d` (370-388), all calling `AW.tok_aw(...)` without the new arg.
  - Fix: a module-level `AW.DEC_SRC`, set by `main()` and by `inject_aw_eval.load()` (1291) / `run()` (1297) from args.json (`a.get("dec_src", "text")`), read by tok_aw when the param is None.
  - Plus a runtime assert in `InjectOW.dec_logits` (inject_ow.py 86-90, through an InjectAW override) that every item's dec matches the cue form when DEC_SRC == "cue".
- `read_mask` (inject_aw_eval 993-1043) assumes the text decoder. It is not used here and asserts if called.

**boxes/inject_aw_eval.py:** register `swap_1f` in PARTS (36) and the dispatch dict (1316-1341), living in a new `boxes/so_parts.py` (the sc_parts / of_parts pattern). It reuses `swap_pairs` 217, `grab_states` 235, `passes` 55-82 (edit hook) and `part_swap_1` 271-279 as its template.

**boxes/deltamem.py (read, NOT edited):**
- DeltaMem.__init__ 52-84: the ln_1 / c_attn / ln_w copy and the zero inits mirrored in DeltaMemQ;
- read_fixed 158-176 and write_seq 179-223: mirrored with RMSNorm + QK-norm;
- `_block` 251-260, `gpt_lower` 263-269, `wpe_slice` 43-48: the GPT-only position add that InjectQ must never call.

**Other GPT-specific code that InjectQ overrides:**
- InjectAW.pass_aw (inject_aw 47-57) adds `DM.wpe_slice`;
- `colon_logits` 61-64 uses ln_f;
- nested170 InjectN.write / lower / aux_logits (238-273) and deltas5.fwd_read add wpe.
- InjectQ overrides pass_aw / write / lower / dec_logits / aux_logits / colon_logits.

**Loading:** nested170.make_trunk 411-425 → s5.load_gpt2 (141-151; `--base` choices). inject_q adds its own loader and leaves s5 untouched.

**New modules:** `boxes/numread.py`, `boxes/spi_direct.py` (SC2), `boxes/trunk_qwen3.py`, `boxes/deltamem_q.py`, `boxes/inject_q.py`, `boxes/inject_q_eval.py`, `boxes/so_parts.py`.
- spi_direct reuses OW.Stream / build_sets (inject_ow 127-171), SEED_D, and fsq120 make_opt 731 / lr_at 740.
- Its backends: `house` for gpt2 / gpt2-medium (GPT2KV, tiktoken) and `hf` for Pythia / Qwen3 (AutoModelForCausalLM at pinned revisions; the LM head applied only to gathered answer positions).

## 4. Gates (Experiments builds; Audit G-D before any spend)

**p = 0 equivalence for every new flag:**
- **G-DS0:** `--dec_src text` (default) matches the pre-change inject_aw bitwise (step-50 model + AdamW, resume 20 → 40) on p2m_s1b's args at `--sched_scale` small, and on bun_s1's args. eval_acc and eval_swap_1 are byte-equal on a 50-item subset.
- **G-DS1:** `swap_1f` on a dec_src-text model reproduces `swap_1`'s follow_ca / follow_dfin / stay_own exactly.
- **G-SPI0:** spi_direct `--backend house --base gpt2` matches inject_aw `--arm direct --d_max 4` bitwise at step 50 (same items, ihash, d sequence).
- **G-SPI1:** the hf backend on `openai-community/gpt2` matches the house backend's step-0 logits in fp64 (rel ≤ 1e-10). This ties the HF code path to the house path.
- **G-Q7:** inject_q `--base gpt2` matches inject_aw bitwise (step-50 model + AdamW, resume) on P-2 args.

**Port gates (S10 R1):**
- **G-Q1:** the trunk_qwen3 forward vs HF Qwen3ForCausalLM (eager attention, fp64) on 16 OW items: max|Δlogit| / max|logit| ≤ 1e-10.
- **G-Q2:** a right-padded batch equals per-item unpadded forwards on the real positions (fp64, ≤ 1e-10). cache_lower: lower(14) + upper = full (fp64, ≤ 1e-10) at d 1 / 4 / 8.
- **G-Q3:** the memory install is a no-op at step 0 (wo = 0): max|Δlogits| = 0 vs the plain trunk, in the decoder and aux forwards.
- **G-Q4:** the mapping assert (qkv rows == the mapped q_proj / k_proj / v_proj slices; q_norm / k_norm copies, the G12 analogue). Head distinctness: max pairwise cosine of the per-head k-weight blocks < 0.9; the pairwise state correlation after one write on 64 items < 0.9.
- **G-Q5 readout fixtures:**
  - tokenization asserts (SFX_Q ends in Ġ; c_10 = [Ġ, 1, 0]; vidx covers the digit tokens only);
  - s(·) on hand-built logits for 1 vs 10;
  - a nonzero CE gradient at the q₂ position;
  - an untrained-memory model gives LP2 ≈ majority (not ≈ 1 from reading the space token).
- **G-Q6 / gradflow:** `gradflow_check` per term (answer CE → trunk + memory; aux → write k / v / gate / ln_w / q_norm / k_norm, after wo is perturbed to N(0, 1e-3)).
  - S11: decoder CE reaches the write params through S, and no path runs from the decoder to any op-sentence token (they are absent).

**Smokes and memory:**
- the plain-path smoke (`--arm direct`, 1 step) for every new base, plus the GPT-2 default;
- **mem_check** with micro pinned (auto_micro headroom ≥ 1 GiB) at:
  - S11 SO (d 2, K 8, 3090);
  - SPI Qwen (d 4, bs 64);
  - R3 (d 2, K 8);
- G-TIME on the 3090 before admission;
- host Python 3.11 smoke (no 3.12+ APIs);
- transformers pinned (≥ 4.51) on hosts for G-Q1 / G-SPI1 and loading only.

**Zero-init rule:** DeltaMemQ's wo = 0 gets a usage monitor: read_ratio = ‖fade·wo(o)‖ / ‖h‖ per 100 steps (logged in R3, binding-logged in S12). S12 is pre-named with a **WO-NZ arm** (wo ~ N(0, σ), σ set for read_ratio ≈ 0.02 at step 0). S11 adds no new pathway.

## 5. Pre-registration criteria

**Statistics:**
- Wilson 95% CI on every accuracy cell (n per cell stated). NEAR = the threshold lies inside the CI; the label is then suffixed "-NEAR".
- Seeds are compared on per-seed values, never on pass counts.
- No C2 claim is made in this family. The ≥ 3-seed rule is applied to SC2's index anyway.

### SC2 (binding = B; descriptive = D)

| ID | Rule | Fields |
|---|---|---|
| B-ORDER | mean SPI2(medium) − mean SPI2(small) ≥ 0.15 AND min over medium seeds > max over small seeds (each at the base's better lr) → CONSISTENT; both bases' acc Wilson-upper(d 2) < maj + 0.10 in every seed → FLOOR; else INCONSISTENT | eval_spi.json by_d['2'].acc, .n, majority['2'] |
| B-DSTAR | d*(b) = the smallest d with mean SPI(b, d) ≤ 0.10 and every seed's Wilson-upper acc ≤ maj_d + 0.15; none in 1-4 → d* = ">4" | by_d[d] |
| B-UI | UI(medium) test acc ≥ 0.50 with the G-AW1 controls valid ((a) S := 0 ≤ maj + 0.05; (b) LP2 path ≥ 0.80) → USE-CAPABLE (the trunk can compute from a read value: the route medium's pass 2 / decoder could take); ≤ maj + 0.10 → USE-INCAPABLE, the same as small; between → PARTIAL; a control fails → UNREADABLE (one fix + re-run) | aw_frozen_use_medium.json (n 1,000) |
| D-RANK | the Spearman ranking of the 5 bases by SPI2; Pythia-160M vs small and 410M vs medium (data / arch at matched size) | — |
| D-LR | the SPI2 gap between lrs per base | — |

**Pre-registered predictions passed downstream:**
- **P-S12:** if CONSISTENT, τ = the midpoint of the small and medium SPI2 means. Bases with SPI2 > τ are predicted SHORTCUT, meaning the text-decoder control in S12 / SC4 shows decoder recomputation (K' 1 ≥ 0.5 and swap follow_ca via the text).
- **P-S11:** if SPI2(medium) ≥ 0.30, SO-W is predicted TEXT-RECOMPUTE rather than PASS.
- **S12's training d_max = max(2, d*(Qwen)).**

**OC:**
- **Under exchangeable seeds** (no true difference), P(min-medium > max-small) = 1/20. Combined with the 0.15 mean bar, P(false CONSISTENT) < 0.05.
- **True gap 0.30 and per-seed σ 0.09** (Note 269-1's d 6 spread, used as a conservative proxy): P(pass) ≈ 0.85 [ASSUMED normal].
- **Priors:**
  - CONSISTENT 0.35 / FLOOR 0.30 / INCONSISTENT 0.35;
  - SPI2(Qwen) ≥ 0.5 with P 0.45 (GSM8K 59.6 per backlog §12.7; GPT-2 small's CD-AN d 2 = 0.097, Note 233-4).
- **Outcome map:**
  - FLOOR → a pre-named SC2-L at 8k for medium and Qwen only (~$1.5, a separate registration);
  - INCONSISTENT → "one-pass hop capacity does not explain medium's shortcut". The S-only guard then becomes mandatory for every base > small, independent of SPI.

### S11 (per binding arm SO-W, SO-F; CTRL-W descriptive)

| ID | Rule | Field |
|---|---|---|
| ACC2 | d 2 acc K' 8 ≥ 0.90 | eval_acc by_d['2'].acc_kprime['8'] (n 1,000) |
| NOT-DEC | d 2 acc K' 1 ≤ 0.20 (the house NOT-DECODER bar; ≤ 0.10 would sit at chance ≈ 0.10) | by_d['2'].acc_kprime['1'] |
| SWAP2 | follow_ca at K' 2 ≥ 0.90 AND ≥ floor_ca + 0.50 | eval_swap_1f by_k['2'].follow_ca / floor_ca |
| AUX2 | the last-1k-bin aux value acc, d 2 pass 2 ≥ 0.90 | log.jsonl aux_vacc_by_d_pass['2']['2'] |
| COV | args dec_src == "cue" and eval meta dec_src == "cue" | args.json, eval_*.json meta |

**Labels** (first match):
- COV-FAIL (void);
- NO-READ (ACC2 fails);
- EAGER-P1 (NOT-DEC fails);
- TEXT-RECOMPUTE (SWAP2 fails with stay_own(K' 2) ≥ 0.5);
- ECHO (SWAP2 fails with follow_v1don ≥ 0.5 on n_sep ≥ 100);
- AUX-ONLY (AUX2 fails);
- **S11-PASS**;
- else OTHER.

**SO-F extension:** one +6k (Note 258-4 form) if 0.50 ≤ follow_ca(K' 2) at 12k < 0.90, or if it is < 0.50 but rose ≥ 0.10 from the 6k read. The extension is final.

**SO-WS (small) is read with the same gates and is binding as a comparator.** "Scale-specific" wording needs SO-WS PASS ∧ SO-W fail. If both fail with the same label, the failure belongs to the S-only intervention, not to scale.

**Decision:**
- SO-F PASS → the larger-base recipe = askname → P-2 with an S-only decoder (S12).
- Only SO-W PASS → two-stage (text-decoder formation, then the S-only switch).
- Both fail → M8 TEXT-DROPOUT / M2 COLON-CE next; the S-only decoder is not sufficient.

**OC:** the bars are absolute on n 1,000 (Wilson half-width ≈ 0.02 at 0.9).
- Priors: SO-W PASS 0.45, TEXT-RECOMPUTE 0.25, ECHO 0.10, EAGER-P1 0.10, NO-READ 0.10; SO-F PASS | SO-W PASS 0.5.
- **Null** (S holds v_2 but pass 2 does not compute it from S_1): SWAP2 needs follow_ca ≥ 0.90, unreachable by an echo or text model, so P(false PASS) ≈ 0.
- **Proposed $0 pre-stated read (needs push → Audit ack):** p2m_s1b's stored eval_swap_1.json by_k['2'].follow_ca. This is CTRL's step-0 analogue under the text decoder; the prediction is ≥ 0.9 via decoder recompute.

### S10

| ID | Rule |
|---|---|
| R1 PORT | G-Q1–Q7 all pass (gates_pass.json booleans + max_rel_err) |
| R2 ASK-Q | ST classes i-iv pooled ≥ 0.98 each AND TEST2 d 1 ≥ 0.98 (eval_st_classes.json, eval_acc) |
| R3 INSTALL | 300 steps: n_nonfinite = 0; write_grad_norm > 0 after the fade; read_ratio ∈ (0, 0.5]; max head-state pairwise cosine at 300 < 0.99; mem_check peak ≤ 21 GiB at the pinned micro |

- **Label:** READY = R1 ∧ R2 ∧ R3; else PORT-FAIL / ASK-FAIL / INSTALL-FAIL. R4 = SPI(Qwen), descriptive here.
- **Priors:** R1 0.9, R2 0.85, R3 0.85 → READY ≈ 0.65. These are build checks: a fail means a fix plus one re-run, not a science null.

### Censoring (all)
- A crashed or host-lost cell gets one VOID-RERUN on the same GPU class. If still unreached → FAIL for the gates it feeds; an SPI cell → that base is UNREAD (B-ORDER needs all 6 small / medium cells, else INCONCLUSIVE).
- Non-finite runs count as FAIL.
- No TEST2 item is dropped after the fact.
- Aborted logs are kept (renamed *.void).

## 6. Scorer sketches (committed with fixtures before any output)

**theory/score_sc2.py**
- Reads `spi_<base>_lr<lr>_s<k>/eval_spi.json`: by_d[d].{acc, n}, majority[d], curve[step][d].acc, args (base, lr, seed, revision sha).
- Computes SPI, Wilson CIs, the better-lr selection, B-ORDER / B-DSTAR / D-RANK, τ, and the P-S12 / P-S11 prediction strings.
- **Fixtures (8):**
  1. CONSISTENT;
  2. equal bases → INCONSISTENT;
  3. overlapping seeds → INCONSISTENT;
  4. gap exactly 0.15 → NEAR;
  5. FLOOR;
  6. a missing medium seed → INCONCLUSIVE;
  7. an lr selection tie;
  8. d* = ">4".

**theory/score_s11.py**
- Reads eval_acc by_d['2'].acc_kprime, eval_swap_1f by_k['2'].{follow_ca, floor_ca, stay_own, follow_v1don, n_sep}, log.jsonl's last bin aux_vacc_by_d_pass, and the dec_src meta. CTRL-W is reported beside.
- **Fixtures (9):** one per label, plus COV-FAIL, plus the extension trigger and no-trigger.

**theory/score_s10.py**
- Reads gates_pass.json, askq eval_st_classes / eval_acc, and the smoke log.jsonl (n_nonfinite, write_grad_norm, read_ratio, head_cos).
- **Fixtures (5):** one per label, plus a NaN log.

**Rule:** a field-path check (grep the module for every key the scorer reads) is part of G-D.

## 7. Cost (RTX 3090, $0.20/h; G-TIME replaces every rate)

| item | basis | point | guard (×1.3) |
|---|---|---|---|
| SC2 30 cells | 2k steps; ~0.12 / 0.25 / 0.13 / 0.27 / ~0.45 s/step (small / medium / P160 / P410 / Qwen, direct; medium from Note 258-2's 0.252 on an A6000) + set build / eval ~2 min per cell ≈ 5.1 GPU-h | $1.05 | $1.40 |
| S10 | G-TIME + mem_check $0.10; ASK-Q 8k ≈ 1 h $0.20; R3 smoke $0.05; download / setup $0.10 | $0.45 | $0.80 (×1.8, uncertain rate) |
| S11 stage 1 | SO-W + CTRL-W, 6k each at ~2.4 s/step (p2m_s1b: $3.55 / 12k on an A6000, incl. eval) + eval $0.10 each; SO-WS small 6k at ~1 s/step $0.35 | $2.15 | $2.80 |
| SC2 UI medium | aw_frozen_use copy, ≤ 2k fit steps on blocks 12-23 | $0.05 | $0.10 |
| S11 stage 2 | SO-F 12k (+6k extension $0.85) | $1.75 (+0.85) | $3.40 |
| **family** | | **≈ $5.5 (+0.85)** | **cap $8.5** |

- **Expected spend** (stage 2 run with P 0.9; extension 0.3): ≈ $4.9.
- All cells run on one GPU class (basin-flip lesson, Note 258-5). The local 4060 runs only CPU / fp64 gates.

## 8. Red team

*(Section filled from a separate sub-agent; see below.)*

## 9. Verified arXiv ids (fetched 2026-11-03; titles match)
- 2505.09388: Qwen3 Technical Report.
- 2304.01373: Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.
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
