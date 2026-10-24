# LB-OW (draft): the §228 word-program band on a larger pretrained base, Qwen2.5-0.5B (2026-10-24 00:19 EDT)

**Status:** draft for Theory → red team → Literature (§10) → Audit / user. Not registered. Nothing built or trained. Only the $0 CPU checks in this folder were run (no GPU, ≤ 2 threads); `lb_bind_probe.py` was run as `--smoke` only (no accuracy printed or written).
- **Follows:** 229-1 (§228 language line CLOSED at GPT-2 small; base-model question to the user) and the user's decision of 10-23 (via Audit / Theory): re-test the band on a ~0.5–1B pretrained base. Anything over $30 total goes back to the user.
- **Task and recipe fixed by §226–§229:** TOPO generator (boxes/ow_words.py, `order_mode topo`, `dag on`, mod 11), train frames 0–5, full FT, inject_ow's §184 port (LAT_v1.md at e78d7211, value target, deviations (a)–(g)). **The only intended change is the base model.** Every forced side change is listed in §4 (L1–L13).
- **Checks run here ($0):** `lb_tok_check.py` → `lb_tok_check.json`; `lb_cost_model.py` → `lb_cost.json`; `lb_weight_rms.py` → `lb_weight_rms.json`; `lb_oc.py` → `lb_oc.json`.

## 0. Claim, scoped
- **C1 only:** "On the TOPO word programs, Qwen2.5-0.5B trained with the §184 INJECT recipe answers depths that one pass cannot learn at equal exposure. Removing the latent passes drops accuracy to the one-pass level, and a mid-chain state swap makes the answer follow the donor."
- Readings, band rule and C1-LANG conjunction: LAT_v1 §2–§3 unchanged (R1 ∧ R2 ∧ R3 on band B), re-labelled **C1-LANG-Q**.
- One seed; a second only on C1-LANG-Q (the user decides; over-cap).
- A null speaks about **this recipe at this base**, not about latent reasoning on language. Never "first", never GSM8K.

## 1. Base choice

| base | params | L / D / heads (KV) | pos / tokenizer | full-FT states (16 B/param) | INJECT pilot d ≤ 2, 8k (3090) | Stage 1 d ≤ 8, 48k (LAT) | verdict |
|---|---|---|---|---|---|---|---|
| GPT-2 medium | 354.8M | 24 / 1024 / 16 | learned wpe; GPT-2 BPE | 5.29 GiB | $0.94–1.46 | $12.0–23.9 | below the user's range; WebText-only base |
| GPT-2 large | 774.0M | 36 / 1280 / 20 | learned wpe; GPT-2 BPE | 11.53 GiB | $2.2–3.2 | $31–78 | over $30 for Stage 1; 36 layers × K 8 |
| **Qwen2.5-0.5B** | **494.0M** | **24 / 896 / 14 (2)** | **RoPE θ 1e6; Qwen2 BPE, digits split** | **7.36 GiB** | **$1.22–1.78** | **$14.9–27.4** | **chosen** |
| Llama-3.2-1B | 1235.9M | 16 / 2048 / 32 (8) | RoPE θ 5e5; Llama-3 BPE, "0".."10" single | 18.42 GiB | $3.7–8.1 | $54–128 | over $30 alone; meta-llama repo gated (401 without a token) |

(lb_cost.json `bases.<b>.cells.*.inject*.usd`, point estimates at $0.22/h; ranges = with / without CHUNK_DELTA. Param counts from the downloaded configs.)

- **Decision: Qwen2.5-0.5B** (Apache-2.0, ungated; revision `060db6499f32faf8b98477b0a26969ef7d8b9987`).
  - It is the only candidate in range whose full Stage 1 is near $30 and whose full-FT states fit a 24 GB 3090 with INJECT activations (§3).
  - It has the largest jump in pretrained arithmetic and binding competence per dollar: 18T training tokens, math-heavy [ASSUMED from the Qwen2.5 report, arXiv 2412.15115; Literature to verify the GSM8K base score]. The GPT-2 family is near 0 on GSM8K.
- **GPT-2 medium is kept as a $0 comparator only** (L0 probe, §5): a within-family scale point with zero tokenizer or position change.
- **Llama-3.2-1B is rejected on cost** (states 18.42 GiB leave ≈ 2.2 GiB of the 21 GiB budget for activations; pilot micro 32). Its tokenizer would avoid deviation L3 (lb_tok_check: `answer_single_token_after_space` true).

## 2. Why scale should help pass-1 binding (derivation and predictions)

**The failure to explain.** 228-2 / 229-1: LAT d = 1 answer 1.000, but pass-1 aux value acc is only 0.51 (value target) and 0.55 (R_j target). So d = 1 is answered from the text by the decoder, and the state does not carry step 1. That is the precondition for d = 2.

**Setup.** The pass-t write stores S_t from the think-token write source u = (1 − γ) LN(h_{L/2}) + γ LN_w(h_top); γ ramps 0 → 1 over steps 2,000–6,000. The aux reads S_t at block L/2 of a forward over "[BOS] Step t :" and must emit "<S_t> has <v_t> <u>.". DeltaMem's output projection wo is zero-initialised (boxes/deltamem.py, `self.wo.weight.zero_()`).

- **D1 [DERIVED].** With wo = 0, the read adds nothing to the residual stream, so ∂L_aux/∂S = W_oᵀ(·) = 0 at init. The write parameters and the trunk get no aux gradient through S until wo grows.
  - wo's own gradient is ∝ E[δ ⊗ read(S_1)], with δ the residual error at the read layer and read(S_1) = φ(q)ᵀS_1 (+ the delta-head read).
  - Linearised (additive heads S = Σ φ(k) vᵀ are linear in v; delta heads in the small-state regime), the aux path is a two-layer linear map wo ∘ (read ∘ write) from write-source features φ(u) to the target direction. That is the deep-linear-network setting of Saxe et al. (arXiv 1312.6120).
- **D2 [DERIVED under D1's linearisation and whitened features].** Mode i of that map grows sigmoidally, with escape time τ_i ≈ ln(s_i / a_0) / (η s_i). Here s_i is the singular value of the cross-covariance Σ_{y,φ} between the pass-1 target and the write-source features, and a_0 is the initial mode strength.
  - If v_1 (or the operand bindings it is computed from) is not linearly present in φ (s ≈ 0), the linear path never learns it. The trunk must first build the feature through higher-order gradient terms.
  - Meanwhile the decoder solves d = 1 from the text without S, which removes the d = 1 answer gradient's pressure on S. This matches 228-2.
- **D3 [DERIVED from D2].** Within a fixed budget, pass-1 learnability is governed by s_1 at the start of training: how linearly decodable the step-1 binding is in the frozen base's write-source features.
- **D4 [ASSUMED, Literature to verify].** Linear binding features become more present and robust with scale and training tokens:
  - binding-ID vectors (Feng & Steinhardt, arXiv 2310.17191);
  - entity tracking, weak in small models without code-heavy pretraining (Kim & Schuster, arXiv 2305.02363);
  - fine-tuning reuses an existing tracking circuit rather than building one (Prakash et al., arXiv 2402.14811);
  - read side: a deeper stack above the read (12 vs 6 blocks), and injected-vector steering of a model that works better at scale (prompt tuning, Lester et al., arXiv 2104.08691; transfer to a mid-stack read is [ASSUMED]).
- **D5 [DERIVED, the lr consequence].** D2's escape rate on the memory side depends on η for the zero-initialised parameters (wo, gate, s_add, s_del). Those keep §228's 1e-4 (L7), so the memory-side rate is unchanged and only s_1 differs between bases.

**Predictions (registered with the draft; probabilities [ASSUMED]).**
- **P1 (L0, frozen, $0):** OPD binding probe (§5) test acc at the val-selected layer: Qwen ≥ 0.80 **and** ≥ GPT-2 small + 0.15. P ≈ 0.55. EAGER Qwen > GPT-2 small (descriptive), P ≈ 0.6.
- **P2 (A1):** pass-1 aux value acc (LP2) ≥ 0.80. P ≈ 0.40 overall; ≈ 0.55 if P1 holds, ≈ 0.20 if not.
- **P3 (A1):** LP3 passes (where applicable, §6). P ≈ 0.25.
- **Falsifiers:**
  - P1 holds but LP2 fails → binding is present but not stored. D3's sufficiency fails, and LB closes with that statement.
  - P1 fails but LP2 passes → the scale effect is not via pre-existing binding (reported; D4 unsupported).

## 3. Memory feasibility (lb_cost.json; all projections replaced by F2 before admission)
- **Calibration on GPT-2 small (measured):**
  - s228_plat2_s1_mem `peak_gib_train` 18.758 (micro 1 = 64 items per micro-batch, d ≤ 2, K 8);
  - f2.json `lat_d2` 6.232 (16 items per micro-batch, 4060);
  - ⇒ 0.261 GiB per item + 2.06 GiB fixed; analytic model scale c = 0.857.
- **Held-out checks:**
  - f2 `lat_d4` (8 items, d ≤ 4): predicted 5.214 vs measured 5.229 GiB;
  - s229 cumulative target: 22.62 vs 20.19 (over-predicts long aux targets, i.e. conservative);
  - time: s229 `train_sec` 7,398 / 8,000 steps = 0.925 vs predicted 0.795 s/step (−14 %; another host, `hostprobe_sps` 0.567 vs 0.504).
- **Qwen2.5-0.5B, full FT, fp32 master + bf16 autocast (as §228):**
  - states 16 B × 494.0M = 7.36 GiB. **No local 8 GB training is possible for any base ≥ 0.5B before activations** (`fits_8gb_fullft_states` false).
  - INJECT d ≤ 2: 0.59 GiB/item → 16 items per micro-batch (inject_ow `--micro 4`), peak 17.0 GiB on a 3090. With `--ckpt_passes`: 0.21 GiB/item, `--micro 2`, 14.4 GiB.
  - INJECT d ≤ 8 (Stage 1): 1.53 GiB/item → `--micro 8`, 19.8 GiB; with ckpt `--micro 4`, 17.0 GiB.
  - DIRECT: `--micro 1`, 12.4 GiB (d ≤ 2) to 18.9 GiB (d ≤ 8). TEXT: `--micro 2` (codi_ps semantics), 16.6 GiB.
  - Compared with §228 F2 (18.8 GiB at micro 1, d ≤ 2, GPT-2 small): Qwen INJECT per item is 2.3× GPT-2 small (24 vs 12 layers, 1.4× s_layer, 3× vocab).
- **Not used:** LoRA (fits easily, but replaces §184's full FT, a second change) and pure-bf16 weights / Adam (halves states; a numerics change). Both are red-team questions.

## 4. Cells and deviations
**Cells (one GPU class: RTX 3090, as §228's pilots):**

| cell | module / flags | role |
|---|---|---|
| DIRECT-Q | inject_ow `--arm direct --base qwen25_05b` | one-pass, same harness and same d-sequence and items as LAT-Q (exposure-matched by construction: LAT_v1 draft D2) |
| LAT-Q | inject_ow `--arm inject --base qwen25_05b` + §228's frozen LAT args (s228_plat2_s1/args.json) | the test cell (INJECT port, value target) |
| TEXT-Q | codi_ps `--direct --text_cot --full_ft --base qwen25_05b` + §227's recipe (CurGate stages 1..8, cg_thr 0.45, cap 6000, cg_signal step_value_ce) | the §227 serial control repeated (band admission); C3 comparator later |

- **Why the INJECT port again and not another arm:** the single-factor question is "does the same recipe pass at a larger base?". A different arm would confound base and architecture. A CODI arm stays a separate future registration (no-loop rule).

**Deviations vs §228 (all forced by the base, plus L7 / L8 for cost and lr):**
- **L1 trunk.** Qwen2 blocks: RMSNorm (no bias), SwiGLU (I 4864), q / k / v biases, GQA 14 q / 2 kv heads, tied 151,936-row embedding.
  - New `boxes/trunk_qwen.py` exposing inject_ow's trunk interface (`embed`, `logits`, `block(i, h)`, `lower`, `final_norm`, `L`, `H`).
  - deltamem's `_block` / `gpt_lower` / `gpt_forward_mem_r` / `fwd_read` dispatch to `g.block` only when present (GPT-2 path bitwise unchanged; G-B0).
- **L2 positions.** RoPE θ 1e6 inside attention at positions 0..T−1 on every forward (think, aux, decoder), replacing GPT-2's added wpe[:T].
  - `--cache_lower` stays valid (identical input every pass), re-gated in fp64 (G-B2).
  - POS_OFFSET / KEEP-SINK evals (note 187-5) are not applicable.
  - The memory reads and writes stay position-free.
- **L3 tokenizer.** Digits are split, and a separate " " token precedes every number. " 10" → [" ", "1", "0"], so the answer classes 1 and 10 share a first token (lb_tok_check `first_token_collisions` [[1, 10]]).
  - **Answer read = prefix-factorised 11-way score** (one forward, decoder + " " + "1" appended):
    - s(v) = log q₁(v) for v ∈ {0, 2..9};
    - s(1) = log q₁("1") + log(1 − q₂("0" | "1"));
    - s(10) = log q₁("1") + log q₂("0" | "1");
    - CE = −log softmax₁₁(s)[ans]; acc = argmax.
  - [PROVEN] This is the sequence probability restricted to the 11 classes, with any non-"0" continuation after "1" counted as termination. Chance stays 1/11; for GPT-2 it reduces to the existing read.
  - **Aux value-token acc (LP2) = every digit token of the value correct** (teacher-forced argmax). The extra " " target token is kept in the CE.
- **L4 special token.** `<|endoftext|>` (151643) replaces GPT-2's EOT (50256) as the think / aux start, aux end and padding.
- **L5 aux prompt.** enc(" Step %d :") under Qwen = [" Step", " ", "1", " :"] (4 tokens + BOS; GPT-2: 3 + EOT).
- **L6 memory geometry.**
  - h_add = h_del = 7 (7 + 7) × 64 = 896; memory layer L_m = 12 (= L/2, §184's rule); write source after block 23 (pre-final-norm).
  - ln / ln_w are RMSNorm, initialised from block 12 / block 23 `input_layernorm`.
  - qkv init: q from q_proj (+ bias); k / v from k_proj / v_proj **repeat-interleaved 7×** (the effective per-q-head keys and values of the pretrained GQA layer).
  - wo / gate / s_add / s_del init as §184. g_add calibration rule as §228.
- **L7 learning rate (two groups).**
  - [DERIVED] AdamW's per-element step is ≈ lr (|m/√v| ≲ 1), so one step moves W by ≈ lr / RMS(W) relative to its size.
  - Holding §228's per-step relative update fixed gives lr_trunk = 1e-4 × RMS_Qwen / RMS_GPT-2 = 1e-4 × 0.0204 / 0.1275 = **1.6e-5** (lb_weight_rms.json `lr_qwen_matched`; median over hidden matrices, n = 168 / 48).
  - The rule applies to the trunk and to DeltaMem's copied ln / qkv / ln_w.
  - Zero- or constant-initialised memory parameters (wo, gate, s_add, s_del) keep **1e-4** (no scale to match; D5).
  - DIRECT-Q and TEXT-Q use 1.6e-5.
  - Corroboration [ASSUMED]: common Qwen-0.5B SFT practice is 1e-5–2e-5. µP (arXiv 2203.03466) would give 8.6e-5 (red-team Q2).
- **L8 cost levers.**
  - `--chunk_delta 64` for LAT-Q, if its gates pass (G-B5; §168 precedent, Note 168-11: 8.1× loop speed).
  - `--ckpt_passes` only where F2 shows it cheaper.
  - `--micro` from mem_check (micro-batched gradient = full-batch up to fp order; one d per step for all micro-batches).
- **L9 vocab.** 151,936 vs 50,257: aux logit memory ×3 (in the §3 model). The decoder gathers one position.
- **L10 lengths.** Think inputs +6–9 % (lb_tok_check `len_ratio_vs_gpt2.think_mean` 1.088 / 1.062 / 1.064 at d 1 / 4 / 8). Value targets +18 %.
- **L11 TEXT decode budget.** max_new_tokens = ⌈1.12 × (24·d + 16)⌉ (`text_cot_mean` ratio 1.12 at d 1–2). Unparseable / non-terminating = wrong, as §227.
- **L12 sets.** TEST2 / TRANSFER / exclusion are text-level (shas unchanged). **PAIRS2 is re-filtered for Qwen token-length match** (R3 swaps need equal lengths; new sha, Stage B only).
- **L13 eval dtype.** As §228 (fp32 eval, cached lower blocks).

## 5. $0 local work (before any spend)
- **No local training cell:** full FT does not fit 8 GB for any candidate base (§3).
  - A frozen-trunk "memory-only" learnability run was considered and rejected: a frozen trunk cannot do the mod-11 step, so pass-1 value acc would be at chance for both bases (uninformative).
- **L0 BIND-PROBE** (`lb_bind_probe.py`; frozen, forward-only, bf16 on the 4060, ≈ 10–20 min per base; or CPU at 2 threads): GPT-2 small, GPT-2 medium, Qwen2.5-0.5B.
  - 4,500 TOPO items (d ∈ {1, 2} alternating, seed 23111, train frames).
  - Targets:
    - **OPD:** S_0's value at S_0's mention inside S_1's defining sentence (lookup among distractors);
    - EAGER: v_1 at the end of that sentence;
    - ANSQ: d = 1 answer at "?".
  - Logistic probe per layer; layer chosen on val (500); read on test (1,000).
  - Target extraction checked on all 4,500 items: 0 non-unique S_1 sentences.
  - **Reading = P1** (§2). **Not a spend gate** (A1 costs ≤ $1.84 guard). It sets the wording of the A1 outcome and is reported before A1 launches.
- **Build gates G-B0–G-B8** (§9) on CPU / 4060 at micro shapes, then **F2 on one cloud 3090** (mem_check + 60-step timing at the A1 and Stage-B shapes, plus the chunk trajectory gate; ≈ $0.15).

## 6. Cloud pilots (LP gates as §228 / §229)
**A1 = P-DIR2q + P-LAT2q** (d ~ U{1, 2}, 8k steps, one 3090; §228's P-LAT2 form).
- **LP0:** no non-finite value. Plus no 100-step mean gn_preclip after the first two windows > 2 × the max of windows 3–10 of the same run.
  - This is a divergence rule self-calibrated to the base, stated before any output. §228's 10 and §229's 38.1 were codi_ps / GPT-2 scales.
- **LP1:** LAT d = 1 train acc (last 1k bin) ≥ 0.90.
- **LP2 (state carries step 1):** pass-1 aux value acc (last 1k bin, all digit tokens) ≥ 0.80.
- **LP3 (binding):** TEST2 d = 2 (1,000 items, K′ 8) LAT ≥ 0.70 **and** ≥ DIRECT-Q d = 2 + 0.30.
  - **Applicability (new, stated before output):** if DIRECT-Q d = 2 > 0.40, one pass partly learns d = 2 on this base. LP3 is then NOT-APPLICABLE, and the A1 decision is LP0–LP2. The binding test moves to A2's band depths.
  - Reason: OC with DIRECT at 0.62 gives P(LP3 | working LAT 0.90) = 0.14 (lb_oc.json).
- Descriptive: DIRECT-Q d = 1 (§228's DIRECT reached only 0.623 at 8k), entry_d1, per-d curves, clip rate, aux name-token acc.

**A2 = P-DIR4q + P-LAT4q** (d ~ U{1..4}, 16k), only after an A1 pass, **in parallel with TEXT-Q** (§227 recipe, `--stop_at 22000` so that d 6 gets ≥ 4k steps; TEXT pass rule as §227: free-decode d = 2 ≥ 0.80 and d = 3 ≥ 0.70).
- **LP4 (binding):** LAT TEST2 d = 3 ≥ 0.60 and d = 4 ≥ 0.50, each ≥ DIRECT-Q + 0.25.
  - Read only on depths where DIRECT-Q ≤ 0.50, else that depth is NOT-APPLICABLE.
  - LP4 needs ≥ 1 applicable depth to pass.
- **Futility (spend guard; counts as FAIL):** at 8k, stop A2-LAT if pass-1 aux < 0.50 **and** its OLS slope over the last 4 bins ≤ +0.02 / bin.

**Outcome map (pre-committed):**
- A1 fails LP1 → harness fault; one G-D'd diagnosis, never a retune.
- A1 fails LP2 or an applicable LP3 → **LB CLOSES:** "the §184 INJECT recipe does not carry step 1 on the TOPO band at Qwen2.5-0.5B either", worded with P1's result. **No fallback** (§229 was the recipe fallback; scale was the remaining lever). The next lever goes to the user.
- A1 passes → A2 ∥ TEXT-Q.
  - TEXT-Q fails → band undefined on this base → stop (A1 / A2 descriptive).
  - LP4 fails → **LAT-SHALLOW-Q** (carries step 1, does not compose deeper) → close.
  - LP4 passes → args-freeze Note → Stage B (Audit approves within the $30 line) or the full Stage 1 (user).

## 7. Stage B (the ≤ $30 version) and the full design (over $30)
- **Full design** = §228's Stage 1 on Qwen (DIRECT-Q + LAT-Q, U{1..8}, 48k steps, 6k steps per depth): **$23.50 point / $30.53 guard with CHUNK_DELTA; $37.21 / $48.36 without** (lb_oc.json `full_design_d8`). **This exceeds the $30 line at guard → the user decides.**
- **Stage B (≤ $30)** = U{1..6}, 36k steps (6k per depth, as §228). Band B ⊆ {2..6}, still ≥ 3 depths with ≥ 2 of them ≥ 4.
  - **Staged total (build + A1 + A2 + TEXT + Stage B): $17.17 point / $22.32 guard with CHUNK_DELTA.**
  - **If CHUNK_DELTA fails its gate:** U{1..5}, 30k, $19.57 / $25.44 (staged d ≤ 6 would be $26.56 / $34.52).
- **Readings:** LAT_v1 §2–§4 unchanged (R1 DEEP-GAIN, R2 mean-state REMOVE, R3 STATE-SWAP on cf(a) ∪ cf_b at d ≥ 4 with follow_U ≥ 0.50, G-W0 with 223-6 guards and DIRECT-STILL-RISING).
  - Added: a depth enters B only if TEXT-Q acc(d) ≥ 0.70 at its stop.
  - score_lat runs unchanged (field paths identical; the answer read is internal to the eval module).
- Admission per cell: F2 × 1.3 ≤ share, else an over-cap Note (re-scope, never a silent step cut). Guard marks reach the final step; off-host full-state checkpoint mirrors with a tested resume; drain waiters armed at launch.

## 8. Cost, OC, censoring
**Spend (3090 at $0.22/h; point / guard ×1.3; lb_oc.json `spend_point_guard`).**

| item | with CHUNK_DELTA | without |
|---|---|---|
| build: F2 + chunk trajectory gate | $0.15 / $0.20 | $0.15 / $0.20 |
| **A1 pilot (DIR + LAT, d ≤ 2, 8k)** | **$1.42 / $1.84** | $1.98 / $2.57 |
| A2 pilot (DIR + LAT, d ≤ 4, 16k) | $3.83 / $4.97 | $5.27 / $6.85 |
| TEXT-Q (22k) | $1.10 / $1.43 | $1.10 / $1.43 |
| Stage B (U{1..6}, 36k) | $10.67 / $13.88 | $18.06 / $23.47 (use U{1..5}: $11.07 / $14.39) |
| **staged total (all pass)** | **$17.17 / $22.32** | $19.57 / $25.44 (d ≤ 5) |
| full design (U{1..8}, 48k) | $23.50 / $30.53 | $37.21 / $48.36 |
| **expected spend (priors below)** | **$3.93 / $5.11** | $5.67 / $7.37 |

- Host variance (2× on the launch-bound loop) is mostly removed by CHUNK_DELTA (loop share of A1-LAT: 0.14 of 2.17 s/step). Without it, the loop is 1.14 of 3.17 s/step.
- Realised at drafting: **$0** (downloads only).

**OC (exact binomial, 1,000 TEST2 items per d per arm; lb_oc.json).**
- **LP3:**
  - P(pass | null, both arms at chance) = 0;
  - P(pass | LAT 0.72, DIR 0.09) = 0.92;
  - P(pass | LAT 0.70, DIR 0.09) = 0.52 (boundary);
  - P(pass | LAT 0.68, DIR 0.09) = 0.09;
  - P(pass | LAT 0.90, DIR 0.62) = 0.14 (hence the applicability rule).
- **LP4:**
  - P(pass | null) = 0;
  - P(pass | working 0.85 / 0.75) ≈ 1.00;
  - P(pass | marginal 0.62 / 0.52) = 0.82;
  - P(pass | shallow 0.80 / 0.30) ≈ 0.
- **P1 (PB form, test n 1,000 each):**
  - P(pass | equal at 0.60 or 0.95) ≈ 0;
  - P(pass | Qwen 0.85 vs GPT-2 0.65) = 0.996;
  - P(pass | Qwen 0.82 vs GPT-2 0.66) = 0.70.
- **TEXT-Q:** P(pass | §227-like) ≈ 1.00; P(pass | weak 0.75 / 0.65) = 0.005.
- **LP2:** a training-bin mean over ≈ 64k item-passes. Sampling noise is negligible; its OC is set by the true value.
- **Stage B:** as LAT_v1 §7. P(C1-LANG-Q | no effect) ≈ 0 under the mechanism nulls (pause, relay, recompute, floor collapse, leak). A hop-per-pass model passes R2 with P ≈ 0.9.
- **Priors [ASSUMED]:**
  - P(A1 pass) = 0.95 (LP1) × 0.40 (LP2) × 0.6 (LP3 | LP2) ≈ 0.23;
  - P(A2 pass | A1) = 0.5;
  - P(C1-LANG-Q | Stage B) = 0.45;
  - **overall ≈ 0.05.**
  - Value of the null: it closes the "too small a base" explanation for 228 / 229 at ≈ $1.6.

**Censoring.**
- A cell that cannot reach its stop step within its guard (crash, host loss after one resume from the off-host checkpoint, a futility stop) counts as **FAIL** for every gate it feeds. Never-reached = fail.
- TEXT: unparseable, missing, non-terminating or overflowing decodes = wrong.
- LP2 is read on all items (every item has pass 1).
- No TEST2 item is excluded after the fact.
- Aborted runs keep their logs (renamed *.void, "aborted unread").

## 9. Build gates (Experiments builds; Audit G-D before any spend)
- **G-B0 flags-off bitwise:**
  - inject_ow with default `--base gpt2` reproduces §228's port and resume gates bitwise (step-50 model + AdamW) at the §228 module sha;
  - codi_ps with default `--base gpt2` reproduces the §227 text_cot step-50 checkpoint bitwise;
  - deltamem's dispatch adds no op on the GPT-2 path.
- **G-B1 trunk equivalence:** trunk_qwen vs transformers Qwen2ForCausalLM at the pinned revision, fp64, 16 OW items: max |Δlogit| / max |logit| ≤ 1e-10. fwd_read with S = 0 and wo = 0 equals the plain trunk exactly.
- **G-B2 cache_lower fp64** on Qwen OW shapes, d 1 / 4 / 8 (as cache64).
- **G-B3 gradflow per term** (answer CE → trunk + memory; aux → write k / v / gate / ln_w; checked after wo is perturbed to N(0, 1e-3) or after 50 steps, since wo = 0 zeroes the aux path at init). Plus the prefix-factorised read: nonzero grad into the q₂ position.
- **G-B4 plain-path smoke:** `--arm direct --base qwen25_05b`, 1 step without new flags; and the GPT-2 default.
- **G-B5 CHUNK_DELTA:**
  - new inject_ow `--chunk_delta` (default 0 ⇒ bitwise);
  - fp64 final-state equality vs the loop on Qwen shapes (≤ 1e-10 relative);
  - §168's g5 trajectory gate: 200 steps, bs 32, one 3090, d_chunk ≤ max(2 × d_reorder, 0.02), no non-finite.
  - Fail → the no-chunk column.
- **G-B6 resume** (step 20 → 40 bitwise), **mem_check** at the deepest train / eval shapes, **F2** timing on the 3090.
- **G-B7 args single-change diffs** (`--args_only`, diffed against the frozen references):
  - LAT-Q vs s228_plat2_s1 = {base, lr groups, micro, chunk_delta, ckpt_passes, hashes / identity};
  - LAT-Q vs DIRECT-Q = the latent flags only (as §228);
  - TEXT-Q vs §227 TEXT = {base, lr, stop_at, decode budget}.
  - No new args after the freeze.
- **G-B8 answer-read fixture:** on hand-built logits, s(·) for 1 vs 10 and a GPT-2-path identity. score_lat fixtures re-run (no scorer change).
- **Pins:** HF revision sha; tokenizer.json sha (lb_tok_check `tokenizer_sha256`); `module_sha256` for trunk_qwen / inject_ow / codi_ps in args.json.

## 10. Novelty sentences for Literature (no "new" / "first" until checked)
- **N1:** "The §184 recurrent-memory latent loop (a DeltaMem state written per think pass, read by a question-free step decoder) is ported unchanged onto a modern pretrained LM (Qwen2.5-0.5B) and tested for causal serial use on templated multi-step word programs, against exposure-matched one-pass and serial text-CoT baselines on the same base."
  - Check against: Coconut 2412.06769; CODI 2502.21074 (which reports GPT-2 and Llama-3.2-1B); recurrent depth 2502.05171; looped transformers 2502.17416; and the house bib (LITERATURE.md, LIT_*.md, incl. the §228 / §229 addenda).
- **N2:** "A frozen-base linear binding probe is pre-registered as the predictor of whether a latent write can carry step 1, derived from deep-linear escape dynamics at a zero-initialised read projection."
  - Check against: Feng & Steinhardt 2310.17191; Kim & Schuster 2305.02363; Prakash 2402.14811; Saxe 1312.6120; Yang 2402.16837; Biran 2406.12775.
- **N3:** "The same latent recipe fails to carry step 1 at GPT-2 small and is re-tested at a larger base with the trunk lr set by relative-update matching." This is a scale-dependence claim for one recipe and one task. Literature: any prior latent-CoT scale ablation on a fixed task (CODI, Coconut follow-ups, iCoT 2405.14838).
- Expected verdicts [ASSUMED]: N1 narrow (the delta is the INJECT loop on this band), N2 possibly not found, N3 narrow.

## 11. Open questions for the red team
1. **Band shift:** Qwen's DIRECT may learn d = 2 (or d = 3) one-pass. Are the LP3 applicability rule and Stage B's U{1..6} enough to keep a ≥ 3-depth band? Should Stage B go to U{1..8} at the user's cost instead?
2. **lr:** is relative-update matching (1.6e-5 trunk, 1e-4 zero-init memory) the right transfer, versus §228's 1e-4 everywhere (a cleaner "single change", but 6.3× the relative update) or µP (8.6e-5)?
3. **D1–D3:** does the deep-linear escape argument survive the fade and γ ramps, the delta heads' nonlinearity, and the decoder's text shortcut? Is the frozen OPD probe the right proxy for s_1, or should the probe sit on the write source u (LN of h_{L/2} / h_top) rather than raw hidden states?
4. **Prefix-factorised read (L3):** is it acceptable, or should the base be Llama-3-tokenized despite the cost? Does the extra appended "1" token leak class-1/10 information into the q₁ read? (It is causal, so q₁ never sees it; confirm.)
5. **GQA init (L6):** is repeat-interleaving k / v the right "init from the layer's attention", versus random init or q-only copy?
6. **Memory layer 12 of 24:** keep §184's L/2, or match the absolute depth of GPT-2's read (6 blocks)?
7. **CHUNK_DELTA (L8):** acceptable as a family-wide kernel given that the base already changes? Does its 8.1× apply at h_del 7 and T ≈ 65–162?
8. **Contamination:** Qwen2.5 was trained on synthetic math word problems; can DIRECT exploit surface templates in ways GPT-2 could not (shortcut panel re-run on Qwen tokens)?
9. **Tied 152k embedding under full FT:** freeze it (−2.2 GiB, a deviation) or keep it trainable (as GPT-2's tied wte in §228)?
10. **One seed and one GPU class** (3090, §228's class), with the "GPU class changes the basin" lesson: is a second seed at A1 worth $1.4 before any Stage-B spend?

## 12. Main risks
- **Pass-1 may fail again** (prior P(LP2) ≈ 0.40). A null closes the line at ≈ $1.6.
- **Band shift:** a stronger base may solve d ≤ 2–3 one-pass, leaving too few band depths under ≤ $30 (U{1..6}).
- **Build risk:** trunk_qwen + the deltamem dispatch + a GQA-expanded DeltaMem + a new answer read is the largest port since §228. G-B0 / G-B1 carry the weight; the lazy-import and dirty-tree deploy rules apply.
- **Cost model:** time is calibrated on one GPT-2 small host pair (−14 % on the held-out s229 point). F2 must replace every number, and CHUNK_DELTA decides whether Stage B fits.
- **lr transfer** is [DERIVED] from a sign-like Adam assumption. A wrong lr can destroy the very binding features the premise relies on (too high) or stall learning (too low).

## Files (this folder)
- `lb_tok_check.py` / `.json`: tokenizer facts per base (GPT-2, Qwen2.5-0.5B, Llama-3.2-1B via the ungated unsloth mirror, SmolLM2-360M, Pythia-410M).
- `lb_cost_model.py` / `lb_cost.json`: memory and time model, calibration and held-out checks, per base / cell / variant.
- `lb_weight_rms.py` / `.json`: hidden-matrix RMS and the matched trunk lr.
- `lb_oc.py` / `.json`: gate OC and the spend tree.
- `lb_bind_probe.py`: the L0 probe (smoke-tested only on GPT-2 small and Qwen2.5-0.5B, CPU; no accuracy read).
