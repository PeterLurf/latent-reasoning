# 03 — C3: KV-shared and cue-row-cached passes (implementation sketch)

Wave-4 lead design, 2026-11-03. Read-only design: no build, no runs, no spend. All FLOP numbers below are analytic, computed from the wave-2 formulas (backlog §12.1, REVIEW_CC #4, theory/c3_flops_cc.py). The prompt lengths P are backed out of §12.1's prefill row (P ≈ 35 at d 1, ≈ 74 at d 4, ≈ 126 at d 8, including bun's distractor inserts). E0 recomputes everything with real P.

## 1. Goal and backlog IDs

- **E0 C3-RESCOPE + G-TIME** (§1 A4, §12.1): measured wall-clock and counted FLOPs for bun against text CoT on the same items.
- **E1 BUN-SHARED** (§12.1): exact speedups on the trained bun, with no retraining.
- **CR (cue-row cached passes)**: the trained variant of CACHED-CUE.
  - Parked at Note 251-13; REVIEW_CC #1-#8 are carried in.
  - Optional **E3 top-k row refresh** (§7; de-ranked at k = 16) is allowed only at k ≤ 4.
- **Context** (Notes 246-0 / 251-12 / 251-13, §12.1): bun is 2.4-9.1x the FLOPs of value-only text CoT. Its serial-depth edge is only 1.2-1.6x against value-only text. Masking prompt-row reads breaks hops ≥ 3 (rcb d 4: 0.10).

**Headline expectation (pre-stated).**
- E1 cuts bun's FLOPs by 28-41% exactly, which still leaves bun at 1.7-5.3x value-only text.
- CR with the long 11-row cue reaches about parity with *sentence* CoT and stays above value-only text.
- Only a ≤ 2-row cue (CR-S2) gets near value-only parity (≈ 1.05x).
- So the defensible C3 claim stays serial depth / latency, and E0 is what decides it.

## 2. Design

### 2.0 Notation and the bun forward (as read)

**Pass input.** Pass t's input is x_t = [EOT] + enc(" " + text) + enc(" " + QUESTION(u, S_min(t,d))) + SFX.
- The prompt prefix p = [EOT] + enc(" " + text) has length P. The text ends with the final question.
- The 7 cue tokens are asserted at inject_st.py:57/60, and SFX = " The answer is:" is 4 tokens (inject_ow.py:42).
- So L = P + 11.

**Decoder input.** dec = p + SFX (inject_ow.py:73), of length P + 4.

**Padding.** Padding is on the right (nested170.pad:166-170), and every pass input shares one padding (cue_inputs, inject_st.py:70-81).
- Each item's prefix rows therefore sit at the same absolute positions in all d cue inputs and in the decoder input.

**One pass** (InjectAW.pass_aw, inject_aw.py:47-56):
- h_low = blocks 0-5 over all L rows (cached once per *distinct cue*: `m.lower`, nested170.py:241-243; eval inject_aw_eval.py:67; train inject_aw.py:370);
- read_fixed(S_{t-1}) adds W_o[g_a φ(q)ᵀS_a ‖ g_d S_d q̂] at every row (deltamem.py:158-176);
- blocks 6-11 over all L rows;
- write_seq on the ':' row only (deltamem.py:179-223, T = 1).

**Decoder.** inject_ow.dec_logits (inject_ow.py:86-90) → deltas5.fwd_read (deltas5.py:91-98) runs all 12 blocks over P + 4 rows with the read at every row. Under passes() it is called at every K′ in kset (inject_aw_eval.py:80); only the reported K′ counts toward FLOPs.

**FLOPs.** blk(T) = 24D²T + 2DT(T+1), D = 768, head = 2DV.
- bun at K′ (Note 246-0 form, d distinct cues):
  - F_BUN = d·6blk(L) + K′·[6blk(L) + 8D²L] + 12blk(P+4) + 8D²(P+4) + head.
  - The 8D² per row is the read: the full 3D qkv projection (6D²) plus W_o (2D²).
- Check against §12.1: at d 1 this gives ≈ 15.0 G (§12.1: 15.3), and at d 8 ≈ 217 G (§12.1: 219).

### 2.1 E1 BUN-SHARED (exact; no weight change)

**E1a, prefix lower sharing.** Blocks 0-5 on p are computed once per item and shared by all d cue inputs *and* the decoder.
- By causality, prefix rows never attend to cue or SFX rows.
- Cue rows (11 per distinct cue) and decoder SFX rows (4) then run blocks 0-5 against the cached prefix K/V of blocks 0-5.
- Position ids are per row: wpe[P_b + j]. P varies within a batch, so an explicit key mask allows key k only if k < P_b or k is a causal own-suffix key.

**E1b, decoder lower.** The decoder starts at block 6 from the assembled h_low, using a fwd_read variant with h_low.

**E1c, q-only read.** In read_fixed, `mem.qkv(a)[..., :D]` (deltamem.py:163) computes all 3D outputs and keeps D. E1c uses `Fn.linear(a, mem.qkv.weight[:D], mem.qkv.bias[:D])` instead.
- This is exact in exact arithmetic, and k/v received no read-path gradient anyway.
- It saves 4D² per row per read, in passes, the decoder and aux.

**Cost and saving.**
- F_E1 = 6blk(P) + d·6[24D²·11 + 4D·11·L] + 6[24D²·4 + 4D·4(P+4)] + K′·[6blk(L) + 4D²L] + 6blk(P+4) + 4D²(P+4) + head.
- At K′ = d: d 1 15.0 → 10.8 G (−28%); d 4 73.7 → 47.4 G (−36%); d 8 217 → 128 G (−41%).
- What remains is K′·6blk(L): the upper half re-run over the whole prompt every pass (97 of 128 G at d 8).

**Value-only text** (≈ 6.4 / 13.9 / 24.0 G) puts E1 at **1.7x / 3.4x / 5.3x**.

**Not shareable:** blocks 6-11 on prompt rows, because they read S_{t-1}. Removing them is §2.2.

### 2.2 CR: cue-row cached passes (trained variant)

**Forward.**
1. **Prompt cache.** Run all 12 blocks once on p, with the read on **S_0**, and keep K/V for every block.
   - read_fixed on the zero state gives oa = od = 0, so W_o contributes its bias only.
   - Pass 1 is therefore **bitwise-in-exact-arithmetic identical to bun's pass 1**: prompt rows in bun's pass 1 also read S_0.
   - Note 251-13's S0 mask zeroed the W_o *output*, bias included (hook on mem.wo, inject_aw_eval.py:1017-1019, 1036), so S0 was *not* the CR forward. S0′ below fixes this.
2. **Pass t ≥ 2: cue rows only.** The c cue rows (long: 11 = 7 cue + 4 SFX) are processed at positions P_b..P_b + c − 1:
   - blocks 0-5 attend to the cached prompt K/V (S-independent, so once per distinct cue);
   - h6 ← read_fixed_q(h6, S_{t-1}) **at cue rows only**;
   - blocks 6-11 attend to the cached (S_0-read) prompt K/V plus causal own rows;
   - write_seq(h6[':'], h11[':']) as pass_aw.
3. **Decoder.** The 4 SFX rows over the prompt cache, reading S_K at those rows only (bun reads at all P + 4 rows).
4. **Top-k refresh (optional, k ≤ 4).**
   - **Row choice:** at pass t, the k prompt rows with the largest head-mean attention mass from the ':' row of pass t − 1 at blocks 6-11. This is the model's own choice (the plan is not consulted).
   - **Upper-block refresh:** those rows are re-run through blocks 6-11 with read(S_{t-1}), from their cached block-6 input.
   - **Splicing:** their K/V are copied, per pass, into a clone of the block-6..11 cache (copy-on-write). Queries = refreshed rows + cue rows, with a causal mask on absolute positions.
   - **Exactness:** with R_t = all prompt rows this is *exactly* bun (gate G-CR-ALL).

**Cost.**
- F_CR = 12blk(P) + Σ_distinct-cues 6[24D²c + 4D·c(P+c)] + K′·{6[24D²c + 4D·c(P+c)] + 4D²c} + 12[24D²·4 + 4D·4(P+4)] + head, plus k·6(24D² + 4D·P) per pass for refresh.

| d (K′ = d) | text val | text sentence | bun | E1 | CR-long (c 11) | CR-S2 (c 2) | CR-long + k 4 |
|---|---|---|---|---|---|---|---|
| 1 | 6.4 | 8.4 | 15.0 | 10.8 | 8.7 | 6.9 | 8.7 |
| 4 | 13.9 | 18.7 | 73.7 | 47.4 | 21.2 | 14.8 | 22.6 |
| 8 | 24.0 | 33.1 | 217 | 128 | 38.1 | 25.3 | 41.0 |

(GFLOPs per item, analytic.)

**What the table says.**
- The 7-token plan cue costs about one sentence-CoT step per pass. So CR-long sits at **0.9-1.15x sentence CoT** and **1.4-1.6x value-only**.
- A cue of ≤ 2 rows is required for value-only parity: REVIEW_CC #4 gives 13.98 vs 13.47 at d 4.
- Cue tokens are oracle plan input that text CoT must generate itself. The C3 wording must say so.

**Serial layers.**
- bun = 12 + 6K′ + 6.
- E1 and CR are the same: the lower half is S-independent and runs in parallel, and pass 1 is fused with the prompt pass.
- Value-only text = 12 + 12d.

**Training (CR-FT).** Warm from bun_s1 / bun_s2 (args: boxes/runs_fsq/bun_s1/args.json).
- **Unchanged from bun:** data (REG, distract_p 1.0, mix s256 / chain / givop), aux CE, hold_inc 1.6, K 8, lr 1e-4, bs 64, micro 2, bf16. Only `--cr long` changes.
- **Prompt cache:** recomputed every step, with gradient.
- **hold_step** (inject_aw.py:159-199): it builds its own graph, so it recomputes its own prompt cache, exactly as it now recomputes `hlp = m.lower(xs[-1])` (line 179).
- **aux_logits:** unchanged. It has a separate prompt.
- **Steps:** 4k (FD1: markers read ≥ 4k).
  - Pre-stated futility look at 2k: if d 3 < 0.50 in every seed, stop.
- **Single change per stage (red-team R1).**
  - **Stage 1 = CR-P:** cached passes, with bun's decoder kept unchanged (full-row read; its lower half is E1-shared). This adds ≈ 6blk(P+4) back: d 4 ≈ 27 G, d 8 ≈ 49 G.
  - **Stage 2 = CR-long:** the 4-row SFX decoder. Conditional on stage 1 passing.
- **Not in this stage:** CR-S2 needs a cue change (" So <S> has" or " <S>:") on top of CR. That is two changes, so it is a later stage only if CR-long passes (REVIEW_CC #1 staging).

### 2.3 E0 G-TIME harness

**Script.** New boxes/c3_gtime.py. Random weights: nested170.make_trunk with `scratch "12,768,12"` (core.random_gpt2_sd, nested170.py:411-423), and DeltaMem built on it with W_o re-drawn N(0, 0.02) and fade 1. No model.pt.

**Items.** 64 REG TEST2 items per d ∈ {1, 2, 4, 6, 8}, built by tok_aw with bun's args (real P, distractors).

**Configs.**
- BUN: the passes() path.
- BUN-E1.
- CR-long, CR-S2, CR-long + k 4.
- TEXT-VAL and TEXT-SENT: core.GPT2KV prefill, then a teacher-forced, KV-cached 1-token loop of the gold CoT length, with the head on every generated token (core.py:123-157).
  - Fairness (R4): GPT2KV grows its cache by torch.cat every step (core.py:140-141), which handicaps the text arms. So a preallocated-cache loop is the binding TEXT, and HF GPT2LMHeadModel(config) with use_cache (random init) is the reference row.
- K′ ∈ {d − 1, d, 8}. The C5 stability rule is a separate "+ decoder every pass" row.

**Protocol.**
- bs ∈ {1, 8, 64}, fp32 and bf16 autocast.
- 5 warm-up + 20 timed reps using CUDA events after synchronize; median and IQR.
- The whole grid runs twice in separate processes to measure the run-to-run spread s.
- Logged: clocks, driver, nvidia-smi (no co-tenants).

**G-FLOP.** torch.utils.flop_counter.FlopCounterMode totals vs the analytic F_* formulas, within 2% for every config.

**Output.** c3_gtime.json rows {cfg, d, Kp, bs, dtype, flops_analytic, flops_counted, lat_ms_med, lat_ms_iqr, items_per_s, serial_layers, run}.

## 3. Code touch points (all read)

| File:lines | Change |
|---|---|
| deltamem.py:18-23, 158-176 | add `Q_ONLY = False` (module flag, precedent FAST_ADD / POS_OFFSET / READ_CLAMP); line 163 branches to the q-slice linear when on |
| deltamem.py:251-269 | new `_block_kv(g,i,h,hd,past,attn_mask)` and `lower_prefix / lower_suffix` in a **new module** boxes/c3_cache.py; `_block`/`gpt_lower` untouched |
| deltas5.py:91-98 | `fwd_read_low(g, h_low, mem, S)` (blocks mem.layer..11) in c3_cache.py |
| inject_aw.py:45-66 | `InjectCR(InjectAW)` in c3_cache.py: `prompt_cache`, `pass_cr`, `dec_cr`, `write`; pass_aw's ':' assert reused on the last cue row |
| inject_aw.py:361-419, 159-199, 421-431 | `--cr {0,long}` and `--shared_lower`; line 370 (hls), 386 (write), 404 (dec), 179 (hlp) switch; asserts: not nk / ask_p / perm_p / cue_ans / ckpt_passes, `POS_OFFSET == 0` |
| inject_aw.py:134-151 (ws_readout), inject_aw_eval.py:55-82 (passes), 85-92 | E1 / CR lower and dec swap behind `--e1` / `--cr`; per-item longest-common-prefix detection, asserted == len(it["think"]) for seq_std; custom seqs_of_t (cue_only / cue_wrong / test_perm) fall back to the old path |
| inject_aw_eval.py:993-1041 (read_mask) | new keep mode `bias_kept`: prompt rows read S_0 (not zero) = the CR k = 0 forward (S0′) |
| theory/c3_flops_cc.py | extended into the analytic F_* functions imported by c3_gtime.py (single source) |

## 4. Gates

- **E1 gates:**
  - **G-E1-OFF:** det-wrapped (scratch/s222_det_wrap.py), the old commit vs the new tree at flags off. part acc on TEST2 d 1-12 plus a 50-step train: per-item preds, logits sha and final params bitwise.
  - **G-E1-PREFIX:** det-wrapped fp32 GPU. gpt_lower(x_t)[b, :P_b] is bitwise equal across t (same B×L shape). The decoder input has a different T, so its rows are checked in fp64 only. This verifies the sharing premise (causality, right padding).
  - **G-E1-FP64:** max relative error ≤ 1e-12 on logits, S_t, ':' logits and (for training use) every gradient. 64-item mixed-P batches, d 1-8, K′ 0..14.
  - **G-E1-FP32:** answer flips vs the old path on TEST2 d 1-12 at all K′; max |Δlogit| reported. A strict bitwise ON gate is not expected: GEMM shapes change.
    - Binding use of E1 requires 0 flips.
    - Otherwise E1 is timing / accounting only, and never mixed with a comparator (exactness memo).
- **CR gates:**
  - **G-CR-OFF:** as G-E1-OFF.
  - **G-CR-ALL (fp64):** refresh = all prompt rows ≡ bun (forward, S_t, decoder, and the gradient of one hold-on step), ≤ 1e-12.
  - **G-CR-P1 (fp64):** CR's S_1 ≡ bun's S_1.
  - **G-CR-MASK (fp64):** CR k = 0 ≡ read_mask(bias_kept).
  - **G-CR-KV:** the prompt K/V hash is identical across passes within a step (S-independence; REVIEW_CC #6).
  - **G-CR-PAD (fp64):** an item's outputs are invariant to batch-mates (P_max changes).
- **Common:**
  - gradflow_check on one CR step: nonzero gradient on blocks 0-11 (via the prompt cache), on mem q rows (read), k/v/gate (write) and W_o.
  - A plain-path smoke after the edits.
  - mem_check / auto_micro with 1 GiB headroom.
  - No new zero-init pathway. The usage monitor still logs per-pass cue-row read norms (mem.norms) and the ':' attention share on prompt vs cue rows. A non-zero-init arm is n/a.
  - All CR gates on a cloud 3090 (local RAM bit-flip hold).

## 5. Pre-registration table

Statistics: Wilson 95% CIs (n = 1000 per d); NEAR = within 0.02 of a threshold (or ±10% for timing ratios); per-seed values reported.

| ID | Kind | Criterion | PASS | FAIL |
|---|---|---|---|---|
| E1-EXACT | binding (use) | G-E1-OFF, -PREFIX, -FP64 | all pass | any fail → E1 not used |
| E1-FLIP | binding (use) | fp32 flips, TEST2 d 1-12 | 0 | > 0 → accounting only |
| E0-FLOP | binding (harness) | counted vs analytic | ≤ 2% every config | stop; fix the formula |
| E0-LAT1 | descriptive label | bs 1, fp32, K′ = d, BUN-E1 latency / TEXT-VAL | ≤ 0.83 LAT-WIN | ≥ 1.2 LAT-LOSE; else NEAR |
| E0-THR64 | descriptive label | bs 64 items/s ratio, same rule | | |
| CR-S0′ | descriptive ($0.1) | bun_s1/s2 bias_kept, TEST2 d 1-8, K′ d − 1 / 8 + WS | — | prediction: d 4 ≤ 0.3 |
| CR-ACC | binding | stage-1 CR-P 4k, TEST2 d 1-4 at K′ 8, each of 3 FT seeds | Wilson LB ≥ 0.93 | < 0.90 |
| CR-MECH | binding (claimed depth) | swap_kd SWAP-MID follow at d 3-4 ≥ 0.80; WS passes 2..d at d 4 ≥ 0.90 | both | either < 0.70 |
| CR-SHORTCUT | binding | NOT-DECODER (K′ 0 acc ≤ majority + 0.10) and MEAN-K floor at d 2-4 ≤ floor + 0.05 | both | either |
| CR-CTRL | binding | CR-ACC minus CTRL-STD (bun + 4k, standard path, same seed / data) at d 1-4 ≥ −0.02 | | |
| CR-C2 | descriptive (3 seeds) | d 6 / 7 at K′ 8 vs bun (0.97 / 0.86); per-seed margins | | |
| CR-C3F | binding (analytic + G-FLOP) | the passing stage's F(K′ = d) ≤ F_TEXT-SENT at every d 1-8 | all d | any d |

**Censoring.**
- An eval crash, OOM or missing field → the cell is VOID and re-run once on another host. A second failure is FAIL.
- An FT run ending before 4k → FAIL (no extension) unless it is a host fault (re-run once).
- pred −1 counts as wrong.
- G-TIME configs that OOM at bs 64 are reported as OOM with their largest feasible bs, and the label is censored as LOSE.
- Timing differences < 2s (run-to-run spread) → NEAR.

**OC block (outcome priors).**

| Outcome | P |
|---|---|
| E1-EXACT | 0.9 |
| E1-FLIP = 0 | 0.6 |
| E0-LAT1 at d 8: LAT-WIN | 0.55 |
| E0-LAT1 at d 1-2: LAT-WIN | 0.3 (BUN is launch-bound) |
| E0-THR64: LAT-LOSE at every d | 0.8 |
| CR-S0′ d 4 ≤ 0.3 | 0.8 (Note 251-13) |
| CR-ACC | 0.35 |
| CR-MECH given CR-ACC | 0.7 |
| CR-SHORTCUT given ACC | 0.75 |
| CR-C3F | 0.15 (the table predicts a FAIL at d 4-8) |
| CR-C2 retains d 6 ≥ 0.9 | 0.25 |

**Decision rule.**
- Order: S0′ and I2 first ($0.2). If S0′ gives d 4 ≥ 0.90 at K′ 8 in both models, there is no FT: E0 times CR-P at bun's weights.
- If CR-ACC / MECH / SHORTCUT pass but CR-C3F fails, CR's C3 claim is "sentence-CoT FLOP parity + latency". CR-S2 is then the next single change.
- If CR-ACC fails, CR stays parked; C3 = E0's measured latency, with E1 as the exact accounting.

## 6. Scorer sketch (theory/score_c3.py, writes theory/results)

**Reads:**
- `acc.by_d[d].acc_kprime[k]`, `acc.items[d].pred` (inject_aw_eval.py:104-115);
- `ws_deep.by_d[d][t]` (1111-1119);
- the swap_kd follow fields: names to be grepped from part_swap_kd (391+) before the scorer is written;
- g_e1.json / g_cr.json {off_bitwise, prefix_rows_equal, fp64_max_rel, fp32_flips, fp32_max_abs_dlogit};
- c3_gtime.json rows (§2.3).

**Labels:** E1-EXACT / E1-ACCOUNTING-ONLY; LAT-WIN / NEAR / LAT-LOSE per (d, bs, dtype); CR PASS / FAIL / NEAR per row; VOID.

**Fixtures:** synthetic JSONs per label (incl. NEAR at ±0.02, VOID on a missing key, OOM rows), a Wilson unit test, and a field-path check against the module.

## 7. Cost (RTX 3090, ~$0.2/h)

| Item | $ |
|---|---|
| E0 G-TIME | 0.10 |
| E1 gates | 0.25 |
| CR-S0′ | 0.10 |
| CR build gates | 0.10 |
| CR-P FT, 3 seeds x 4k (CR step ≈ 0.5x bun's ≈ 0.55 s) | ≈ 0.40 |
| CTRL-STD 4k | 0.25 |
| 4 evals | 0.40 |
| **Total** | **≈ $1.6**; guard $3 |

## 8. RED-TEAM

**Provenance.** The required separate red-team sub-agent could not be spawned: the session hit its concurrent sub-agent limit and the tool said not to retry. The list below is the lead's own adversarial pass, re-checked against the code. **An independent red team must run before registration.**

**R1 (fixed): two changes in CR.**
- As first drafted, CR changed both the passes and the decoder's read rows (all P + 4 → 4).
- Fix: stage 1 is CR-P (cached passes, bun decoder); stage 2 is the SFX-only decoder (§2.2).

**R2 (fixed): G-E1-PREFIX vs the decoder cannot be bitwise.** The decoder input has a different T, so the GEMM and SDPA shapes change. Fix: bitwise only across the same-shape cue inputs; fp64 for the decoder.

**R3: W_o bias.**
- wo is nn.Linear with a trainable bias (deltamem.py:70-72), so prompt rows of every bun pass receive fade·b_o.
- The read_mask S0 (Note 251-13) removed it. Its collapse at d ≥ 3 may therefore partly reflect a bias shift, not missing state.
- S0′ (bias_kept) separates the two at $0.1 and is the **cheapest CR falsifier**: if S0′ d 4 ≥ 0.9, CR needs no FT.

**R4 (fixed): text-arm latency handicap.** The torch.cat cache growth is fixed via the preallocated cache plus the HF reference row.

**R5: oracle-plan accounting.**
- The cue supplies S_t's name, which text CoT must generate. Its tokens are counted as processed input, but the planning is free for the loop.
- Any C3 wording must say "plan given", or compare against text CoT that is also given the plan in its prompt. TEXT-SENT-PLAN is an optional E0 row: same cues inline, teacher-forced.

**R6: deployed K′.**
- bun runs think_k 8, so K′ = d presumes an oracle halt.
- E0 must lead with K′ 8 and the C5 stability-rule row (+ a decoder every pass; §12.1 ≈ 2x). The K′ = d / d − 1 rows are bounds.

**R7: decoder shortcut (Note 264-1).**
- In CR, the decoder can still compute the last hop from the prompt text plus S_{d−1}.
- CR-SHORTCUT (NOT-DECODER and the MEAN-K floor) and CR-MECH at d 3-4 cover this. CR-MECH must be read at the claimed depth: d 3-4 for CR-ACC, d 6 if CR-C2 is quoted.

**R8: seeds and lineage.**
- 3 FT seeds from 2 inits of one lineage. CR-C2 stays descriptive, and every CR label is scoped "one lineage".
- CTRL-STD controls "4k more steps", not the routing.

**R9: cheaper falsifier before any FT.** I2 READ-USE MAP ($0.1; §13.1) on bun.
- If ablating the read at all prompt rows (bias kept) already costs < 0.05 at d 3-4, CR is nearly free.
- If the read is consumed at the operand sentence rows, re-routing to cue rows is the hard part. Then raise CR-ACC's bar for proceeding, or try top-k first.

**R10: path exclusions.** E1 / CR must assert off READ_CLAMP (deltamem.py:23, 165), POS_OFFSET / POS_KEEP0 (21-22, 43-48), nk (inject_aw.py:118-120), ask_p, perm_p, cue_ans and ckpt_passes. mem.norms / rec record only the last row: in CR the last row is still ':', so they are unchanged.


## 9. arXiv ids (fetched 2026-11-03, titles matched)

- 2412.06769 Coconut;
- 2510.02312 KaVa;
- 2506.18582 PCCoT (Jacobi);
- 2211.05102 Efficiently Scaling Transformer Inference;
- 2110.12894 The Efficiency Misnomer;
- 2310.02226 Pause Tokens;
- 2309.06180 PagedAttention;
- 2207.06881 Recurrent Memory Transformer;
- 2502.05171 Huginn;
- 2404.02258 Mixture-of-Depths.
