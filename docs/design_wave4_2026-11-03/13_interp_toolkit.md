# 13. INTERP TOOLKIT for bun + §269b parts I1 / I3 / I4 / TH2 / RB3 (design, wave 4, 2026-11-03)

## 1. Goal and backlog IDs
- **Toolkit** (`boxes/interp.py`, eval-only, no trainer code): per-head hooks on blocks 0-11 (c_attn / c_proj), per-memory-head state and read patching, position-class masks, path patching and EAP / EAP-IG over the unrolled K passes plus the decoder, and attribution-vs-patching validation (2310.10348).
- **Parts** (`boxes/ip_parts.py`, registered in `inject_aw_eval.run()`; queued as §269b):
  - I1 OP-SELECT HEADS;
  - I3 VALUE CIRCUIT;
  - I4 CROSS-PASS EAP;
  - TH2 MIRROR-PAIR;
  - RB3 PASS-TOKEN-OCCLUSION.
- §269 deferred I1 / TH2 / RB3 to §269b. I3 / I4 are backlog §13.3. I2 READ-USE stays in §269 and shares the position classes below.
- **Targets:** C1 scope (selection mechanism, leakage), C2 diagnosis (d 8 failure locus), C7 groundwork (the mechanism figure). Every label is descriptive and routes lanes C / D / E. **None is a C-claim.**

## 2. What the code fixes (read 2026-11-03)
**Trunk.** Every forward path goes through the module function `deltamem._block(g, i, h, hd)` (deltamem.py L251-260). This covers:
- the think pass: `InjectAW.pass_aw`, inject_aw.py L47-56;
- the lower cache: `gpt_lower`, deltamem.py L263-269;
- the decoder and aux: `deltas5.fwd_read`, L91-98.

In `_block`:
- q / k / v = `LN1(h) @ c_attn.weight + bias`, split, then `.view(B, T, 12, 64)`.
- `o = SDPA(is_causal=True)` with shape (B, 12, T, 64).
- `h += o.transpose.reshape(B, T, 768) @ c_proj.weight + bias`.
- Conv1D (in, out) layout, so head j's output enters through rows `c_proj.weight[64j:64j+64, :]`.
- The c_proj bias is added once and is not attributed to any head.

**Precedent.** sc_parts.py L153-165 (§265 attn_fade) already rebinds `DM._block`. Because `gpt_lower` resolves `_block` as a module global, the rebinding reaches the lower blocks too. The bitwise read-only assert is at L203-212.

**Memory.** DeltaMem sits at layer 6 (`build_aw`, L118). bun has add_norm None, so the state is exactly `(B, 12, 64, 64)`:
- **Blocks 0-5 are additive heads,** stored k×v: `Sa[b,h,k,v] += phi(k_h) v_h^T` (L105). Read `phi(q)^T Sa` (L145).
- **Blocks 6-11 are gated-delta heads,** stored v×k: `Sd = alpha (Sd - beta (Sd k) k^T) + beta v k^T` (L110-111, L208-210). Read `Sd q̂` (L167).

**Read.** `read_fixed` (L158-176) adds `fade * W_o([exp(s_add) oa || exp(s_del) od])` at every position of every pass and of the decoder. Memory head h enters through the columns `W_o[:, 64h:64h+64]`.

**Write.** bun has gamma = 1 (write_mix 0). `pass_aw` therefore calls `write_seq` on the single ':' row, with `u = LN_w(h_11[':'])` (L54-56, L184-188, FAST_ADD L194-200). Consequences:
- one rank-1 update per head per pass;
- state head h's update depends only on `S[:, h]` and u.
- **Per-head state patching between passes is therefore an exact, clean intervention.**

**Inputs.** The pass-t input is `[EOT] + enc(' ' + text + ' ' + QUESTION(u, S_min(t,d))) + enc(' The answer is:')`:
- ':' is at L-1;
- the cue name is at L-7 (asserted in `tok_aw` L91-92; `CUE_NAME_OFF` 6, of_parts.py L16);
- the cue occupies L-11..L-5;
- the suffix occupies L-4..L-1.

The decoder input is `[EOT] + enc(' ' + text) + SFX` (`tok_aw` L90), where the text already ends with the question about S_d. Sentence spans come from `sc_parts._spans` (L125-138, which asserts the tiling).

**Runner trap.** `EV.passes` (inject_aw_eval.py L54-81) computes `m.lower(x)` once per distinct input (L67). The hold passes t > d all reuse `xs[d-1]`. A hook on blocks 0-5 at "pass t" would therefore leak into every pass that shares the input. **The toolkit runner recomputes the lower blocks for every pass that carries a lower-block hook.**

## 3. Toolkit design (`boxes/interp.py`)
**3.1 Context and dispatch.**
- `with IP.hooks(m, spec) as cap:` wraps the instance methods `m.pass_aw` and `m.dec_logits`, as `read_mask` does (L992-1040). Each wrapper sets `cur_pass ∈ {1..K, 'dec'}`.
- It rebinds `DM._block`, `DM.read_fixed` and `DM.write_seq`, and restores them in `finally`.
- `spec` maps `(pass, layer, kind)` to an op. With no entry for the current `(pass, layer)`, the original function runs, which gives a bitwise no-op.

**3.2 Hook kinds (per block i).**

| kind | site | op | exactness |
|---|---|---|---|
| `z` | o (B, 12, T, 64) after SDPA, before c_proj | capture / replace(donor, heads, posmask) / mean / zero | same ops as `_block`: bitwise at identity |
| `mlp` | the MLP output | capture / replace / mean | bitwise |
| `resid` | h entering block i | capture / replace | bitwise |
| `pattern` / `ov` | manual softmax(QKᵀ/8 + causal)·V | pattern-from-donor with clean V, or clean pattern with donor V | tolerance gate (G-EQ1) |
| `knock` | additive -inf on (query ∉ span, key ∈ span) | attention knockout of a span, all heads, layer set | tolerance gate |
| `qkv_split` | q, k and v from three LN copies of h | separate gradient sites for EAP | tolerance gate |

**Memory hook kinds.**
- `memread`: the pre-W_o read `o_h` (B, T, 12, 64), split per memory head and gated by a position mask.
- `memstate`: `S_t[:, H] := donor[:, H]`, applied through `passes(edit=…)`.
- `write_u`: u at ':'.

**3.3 Position classes** (`IP.classes(it, t)`). Each class is a (T,) bool mask. Names and numbers are located by token id inside each span and asserted to be single tokens. The classes for pass t:
- EOT;
- S0 (the given sentence of S_0);
- STEP (S_t's op sentence), with sub-classes STEP_SUBJ (the S_t name token), STEP_SRC (the S_{t-1} or S_j name token) and STEP_OPND (the number token);
- GIV_t (G_t, for *_v ops);
- PREV (S_{t-1}'s op sentence);
- OTHEROPS (S_k with k ∉ {t-1, t});
- DISTR (distractor and insert sentences);
- QSENT (the text's final question);
- CUE (the cue tokens other than the name);
- CUENAME;
- SFX (" The answer is");
- COLON.

**Partition rule:** the classes tile [0, L). The decoder pass uses the same classes with t = d. The I2 classes of §269 (COLON / CUE / STEP / OTHEROPS / REST) are unions of these.

**3.4 Memory-head patching.** `IP.patch_state(S, D, heads)` returns a clone with `S[:, heads] = D[:, heads]`. Heads 0-5 are additive and 6-11 are delta, and the two orientations differ (k×v vs v×k), so whole blocks are swapped and never transposed. The per-head read contribution is `fade · W_o[:, 64h:64h+64] o_h`. The W_o bias stays in the clean run.

**3.5 Path patching** (2304.05969, 2211.00593). Run three forward passes: clean, corrupt, and patched.
- In the patched run, the sender's z at `posmask` is set to its corrupt value.
- Every other head output and MLP output on the path is frozen to its clean cache.
- Only the receivers recompute.
- Receiver sets:
  - R_out: the ':' residual after block 11, which is both the write source and the WS readout;
  - R_qkv(l', h'): a head's q, k or v input;
  - R_mem: u, then the next pass's state.
- The metric is LD = logit(y_clean) − logit(y_corr) over the 11 number tokens at the ':' readout (pass-t WS), or at the decoder ':' for the final answer.
- Normalised effect: NE = (LD_patch − LD_clean) / (LD_corr − LD_clean).

**3.6 EAP / EAP-IG across passes** (2310.10348, 2403.17806, 2403.00745).
- Under `torch.enable_grad()` with the parameters frozen, the hooks retain the activations of every node.
- **Nodes per pass:**
  - the embedding;
  - 144 heads with separate q / k / v inputs (`qkv_split`);
  - 12 MLPs;
  - 12 memory-read heads at block 6 (inputs: the block-6 residual and the state block S_{t-1}[h]);
  - the write node u_t;
  - 12 state nodes S_t[h].
- **Cross-pass ("memory") edges:**
  - u_t → S_t[h];
  - S_{t-1}[h] → S_t[h], the decay / accumulate carry;
  - S_t[h] → read_{t+1,h};
  - S_K[h] → the decoder reads.
- **Edge score:** ΔA_up · ∂M/∂(input_down), summed over positions; for a state edge, ⟨ΔS, ∂M/∂S⟩_F.
- **EAP-IG** uses m = 8 steps on embedding interpolations between the clean and corrupt inputs. All cross-pass edges are scored, and the lower cache is not shared under grad.
- About 32k edges per pass × 9 forwards, plus 12 × 3 × 8 memory edges.

**3.7 Validation subset (G-ATP, binding for I4 labels).**
- **Exact patching** on 200 items for:
  - all 96 state nodes S_t[h] (t 1..8);
  - all 144 head-z nodes at passes t and t-1 under C-SEL;
  - the 12 MLPs;
  - 300 edges stratified by EAP-IG decile (100 top, 200 random).
- **Metrics:** Spearman ρ between attribution and patching, sign agreement, and recall@20.
- **Pass:** node ρ ≥ 0.7 on state nodes, ρ ≥ 0.6 on head nodes, and recall@20 ≥ 0.6.
- If EAP (m = 1) fails, EAP-IG is used. If EAP-IG also fails, the part is labelled ATTRIBUTION-UNRELIABLE and only the patching ranks are reported.

## 4. Counterfactual families (built in-part; the run's own tokenization; same token length asserted)
- **C-SEL (selection):** at pass t, the cue name S_t is replaced by S_j, with j ≠ t drawn by `default_rng((27010, d, i))`. The item is kept iff the clean and corrupt readouts differ.
- **C-VAL0 (memory value):** S_0's number token is set to v0' (all 11 values swept). The text is otherwise identical, and `OW.solve` is asserted.
- **C-OPND:** S_t's number token is set to c' (1..10, or 2..10 for mul), and solve is asserted.
- **Mirror pairs** (TH2) and **occlusion** (RB3): see the part specs.
- New seed family: 27000-27099. Grep boxes/*.py and scratch/*.py for unused seeds at build time, as of_parts does.

## 5. Part specs

**I1 OP-SELECT HEADS** (`ip_i1`):
- **Conditions:** TEST2 d 4 at t 2 / 3 / 4, and d 8 at t 4 / 7; the first 500 items each, under C-SEL.
- **Per head (144) × posmask {COLON, CUENAME, ALL}:** noising NE (corrupt → clean) and denoising NE (clean → corrupt).
- **Top-5 heads by ALL-noising:** a joint patch, path patching to R_out vs R_qkv, and a pattern / OV split. The clean attention mass from ':' and from CUENAME onto STEP is also recorded.
- **Split rule (selection bias):** items are split 50 / 50 by the sig hash. The top-k heads are chosen on half A; joint5, path and split are scored on half B only.
- **JSON:**
  - `{cond: {n, n_kept, head_ne: [12][12][3] noise, head_ne_den, top5: [[l,h]], joint5: {flip_share, ci}, path: {l_h: {out, qkv}}, split: {l_h: {pattern, ov}}, mass_step: [12][12]}}`;
  - `items` (sig, j, clean / corrupt readout, LD);
  - `def`.
- **Labels:**
  - **LOCALIZED** if joint5 flip_share ≥ 0.6 (d 4 pooled); **DISTRIBUTED** if the top-20 joint < 0.4; else MIXED.
  - Per top-5 head: **PATTERN** if pattern / total ≥ 0.6; **OV** if ov / total ≥ 0.6; else MIXED.
  - **Depth (d 8 t 7 vs d 4):**
    - SAME-CIRCUIT if Jaccard(top-10) ≥ 0.5 and joint5 ≥ 0.5;
    - DILUTED if SAME heads and mass_step ≤ 0.7 × the d 4 value;
    - SHIFT if Jaccard < 0.3.
- **Routing:**
  - LOCALIZED + PATTERN + DILUTED → D1b SSMax / FD4 on the named layers;
  - DISTRIBUTED → head-targeted fixes are dropped and TH3 (selection margin) leads.

**I3 VALUE CIRCUIT** (`ip_i3`):
- **Items:** TEST2 d 4 at t 2-4 and d 8 at t 5-7 (untrained passes); 300 items each.
- **(a) Memory sweep:** S_{t-1} is replaced by the state from the C-VAL0 donor run, for every v0'. follow = P(the pass-t readout = op_t(v'_{t-1}, c_t) mod 11). It is tabulated per (op class, v', c) cell, with v' = 0 for mul flagged as an untrained cell. Also recorded: **stay** = P(readout = the recipient's own v_t), the text-recompute signature. v'_{t-1} is affine in v0', so items whose coefficient is 0 mod 11 (possible through add_s) are listed and excluded from the table.
- **(b) Localisation:** per-component patching (heads 6-11 × 12 and MLPs 6-11 at COLON / STEP_OPND) under C-VAL0 and C-OPND.
- **(c) Circuit faithfulness:** keep the top components explaining ≥ 0.8 of LD and mean-ablate the rest of blocks 6-11 at COLON.
- **(d) Representation (descriptive):** per layer at ':', the class-mean residuals over v_t ∈ Z_11. Report the share of variance in the top-2 DFT frequencies (2301.05217) and a linear-probe accuracy.
- **JSON:** `{cond: {follow, ci, table: {op: [[acc]]}, comp_ne: {..}, circuit: [..], faith, fourier_share: [12], probe_acc: [12]}}` + `items`.
- **Labels:**
  - **MEMORY-COMPUTE** if follow ≥ 0.8 and stay ≤ 0.1; **TEXT-RECOMPUTE** (shortcut) if stay ≥ 0.5; else MIXED.
  - **ALGO-COMPLETE** if the 10th-percentile cell acc ≥ 0.8; **HEURISTIC-HOLES** if ≥ 10% of cells are < 0.5 (2410.21272).
  - **PASS-INVARIANT** if Jaccard(circuit_d4, circuit_d8) ≥ 0.6 and faith ≥ 0.8 on both; else PASS-SPECIFIC.
- **Routing:** HEURISTIC-HOLES or PASS-SPECIFIC → the computation arm of I5 / TH3 (value-table coverage data); ALGO-COMPLETE + PASS-INVARIANT → C2 failures are not computational.

**I4 CROSS-PASS EAP** (`ip_i4`):
- **Items:** TEST2 d 4 (and d 7 descriptive), 500 items, K' 8, metric = the decoder LD. Corruptions: C-VAL0 (value-flow graph) and C-SEL at t 2 (selection graph). Includes the §3.7 validation.
- **JSON:**
  - `{corr: {valid: {rho_state, rho_head, recall20, sign}, method, edges_top: [[src,dst,score]] (top 2000), mem_share: {t: [12]}, dec_text_share, circuit: {n_edges, share, faith}}}`.
  - `dec_text_share` is the attribution share of the decoder's own text path vs the S_K reads (the shortcut-exclusion arm; cf. Note 264-1).
- **Labels:**
  - **ATTRIBUTION-VALID** / **UNRELIABLE** per §3.7.
  - **SPARSE** if a ≤ 5% edge set reaches faith ≥ 0.8; else DENSE.
  - Memory channel: **DELTA-CARRIED** if the delta heads hold ≥ 0.7 of the memory-edge attribution; **ADD-CARRIED** if the additive heads hold ≥ 0.7; else BOTH.
  - **DECODER-SHORTCUT** if dec_text_share ≥ 0.3.
- **Routing:** DELTA- / ADD-CARRIED informs D10 / D11 and MC3; DECODER-SHORTCUT confirms M8 / S11 for all bases.

**TH2 MIRROR-PAIR** (`ip_th2`):
- **Families:** d ∈ {2, 3, 4}, branch point j ∈ {0..d-2}, branch length L = d − j, b = 2. Each family has 300 pairs.
- **Item construction:**
  - The donor item is TEST2 d (REG).
  - A mirror chain M_{j+1..d} is added on unused names, with ops from OPS_CHAIN and frames from FRAME_TRAIN. M_{j+1} reads S_j, and M_k reads M_{k-1}.
  - The sentences of the two branches are interleaved in a random branch-first order: `default_rng((27030, d, j, i))`.
  - Member A asks S_d; member B asks M_d. The text differs only in the question; the cues follow each member's own path.
  - solve is asserted for both members, and v^A_d ≠ v^B_d and v^A_{j+1} ≠ v^B_{j+1} are required.
- **Models:**
  - gb_s1 / gb_s2 (plan-free generic cue; the decisive model);
  - bun_s1 / s2 under STD (control) and under the generic cue (descriptive).
- **Measures:** at the branch pass j+1, the pass-t readouts, the final answer, and the decoder with S_K := the mean state (decoder-rescue arm).
  - QSENS = P(readout_A ≠ readout_B);
  - PAIR-BOTH = P(both members' readouts equal their own branch value);
  - PAIR-BOTH-FINAL;
  - first-mentioned-branch share.
- **Bound (stated before output):** a question-blind deterministic policy gives identical readouts, so PAIR-BOTH ≤ QSENS. A forward-only policy's member accuracy is ≤ Π 1/b_t. A superposed state (2505.12514) could still answer both members at the end.
- **JSON:** `{model: {fam: {n, qsens, pair_both, pair_both_final, mean_k_final, first_share, ci...}}}` + `items` (pair id, branch order, readouts).
- **Labels, on gb (pooled):**
  - **FORWARD-BLIND** if PAIR-BOTH ≤ 0.10 and the QSENS upper CI ≤ 0.15;
  - **Q-CONDITIONED** if PAIR-BOTH ≥ 0.5;
  - else PARTIAL;
  - plus **SUPERPOSED-OR-DECODER** if PAIR-BOTH-FINAL − PAIR-BOTH ≥ 0.3. The MEAN-K arm splits the two: decoder-only if mean_k_final stays.
- **Validity:** bun STD PAIR-BOTH ≥ 0.85, and gb acc ≥ 0.9 on the twins with the mirror chain removed. Otherwise the label is VOID-RENDER.
- **Routing:** FORWARD-BLIND → P1 SELF-CUE is capped on branch families, so P2 BACKWARD-PLAN, §261 TC and TH5 run first. Q-CONDITIONED → P1 is viable.

**RB3 PASS-TOKEN-OCCLUSION** (`ip_rb3`):
- **Items:** TEST2 d 2-4 (1,000 each) and d 6 / 8 (500 each, descriptive).
- **Occlusion:** at pass t ≤ d only, each sentence span is knocked out (`knock`, blocks 0-11, keys of the span hidden from every query outside it; EOT never knocked).
- **Secondary arm:** 3σ embedding noise (2202.05262), 10 draws, descriptive.
- **Readouts:** the pass-t WS readout and the K' 8 answer.
- **Licensed set L_t:** {STEP; GIV_t if *_v; S0 if t = 1 or (add_s with j = 0)}.
  - Leak classes: PREV (recompute from text), OTHEROPS-past, OTHEROPS-future, DISTR, QSENT, unread givens.
- **Measures:**
  - EXPOSED = flip(STEP);
  - LEAK = P(any non-licensed occlusion flips the readout | intact readout correct);
  - a decoder arm (knockout at the decoder pass only).
- **JSON:** `{d: {t: {n, exposed, leak, leak_by_class: {...}, ci}}, dec: {d: {by_class}}, noise: {...}}` + `items`.
- **Labels:**
  - **VOID-INERT** if EXPOSED < 0.8;
  - **CLEAN-SELECT** if the LEAK upper CI < 0.10;
  - **SELECT-WITH-LEAKAGE** if LEAK ≥ 0.20 (the backlog rule rewords C1);
  - else MINOR-LEAK;
  - **DECODER-READS-TEXT** if the decoder flip(S_d) ≥ 0.3.
- **Prediction (wave 3):** the footprint spreads with op count.

## 6. Gates (each is its own queue line; gates_pass.json before any part)
- **G-EQ0 (hooks off):** `IP.run_passes` with an empty spec vs `EV.passes`, d 4 and d 8, 200 items, K' 0..14. Answers and S_K must be `torch.equal`. This doubles as the "p = 0 equivalence" gate, since the toolkit adds no trainer flag.
- **G-EQ1 (manual attention):** identity `pattern` / `knock(∅)` / `qkv_split` on all layers. fp32: max |Δ dec logits| ≤ 1e-4 and answer agreement 1.000 on 1,000 items. fp64 CPU on 20 items: rel ≤ 1e-9 (memory notes: fp64, not the reorder floor; deterministic wrap).
- **G-EQ2:** a self-patch (clean → clean) for every kind is `torch.equal` to the hooked identity run. The z / mlp / resid hooks are bitwise vs the unhooked run.
- **G-LIN:** Σ_h W_o,h o_h + b vs `wo(cat)`: rel ≤ 1e-5. A 12-head `patch_state` equals a full state swap, bitwise.
- **G-LOWER:** a lower-block hook at pass t changes no other pass. Shown by a diff vs the hooked-at-t-only reference on passes ≠ t.
- **G-GRAD:** gradflow through the unrolled graph:
  - every S_t[h] (t < K) and every read node has a finite, nonzero gradient on ≥ 1 item;
  - the pass-1 read q-grads are exactly 0 (S_0 = 0), as expected;
  - and the EAP completeness check: Σ edge scores into the output ≈ the first-order LD change (rel ≤ 0.1 on the C-VAL0 subset).
- **G-CLS:** the position classes tile [0, L) for every item. The name and number tokens are asserted, including insert labels.
- **G-RENDER (TH2):** solve on both members; the only text difference is the question; the token-length equality is asserted.
- **Smokes:** CPU, at full item scale for the set builds and the JSON / scorer reads (random-init trunk). Host-side first lines on bun_s1: the G-EQ set.
- **mem_check:** the I4 grad peak at bs 8; auto-halve; the budget is 21 GiB with 1 GiB headroom.
- **Not applicable:** zero-init usage monitors (no training).

## 7. Pre-registration table
| part | criterion | threshold | type |
|---|---|---|---|
| all | G-EQ0..G-CLS | as §6 | binding validity |
| I1 | joint5 flip_share (d 4) | ≥ 0.6 LOCALIZED / < 0.4 (top-20) DISTRIBUTED | descriptive label |
| I1 | pattern / OV share | ≥ 0.6 | descriptive |
| I1 | depth Jaccard / mass ratio | ≥ 0.5 & joint5 ≥ 0.5 / ≤ 0.7 / < 0.3 | descriptive |
| I3 | follow & stay | ≥ 0.8 & ≤ 0.1 / stay ≥ 0.5 | descriptive |
| I3 | cell p10 / hole share | ≥ 0.8 / ≥ 10% < 0.5 | descriptive |
| I3 | circuit Jaccard & faith | ≥ 0.6 & ≥ 0.8 | descriptive |
| I4 | ρ_state / ρ_head / recall20 | ≥ 0.7 / ≥ 0.6 / ≥ 0.6 | binding for the I4 labels |
| I4 | edge share at faith 0.8 | ≤ 5% | descriptive |
| I4 | memory-edge share | ≥ 0.7; dec_text ≥ 0.3 | descriptive |
| TH2 | bun STD PAIR-BOTH; gb twin acc | ≥ 0.85; ≥ 0.9 | binding validity |
| TH2 | gb PAIR-BOTH / QSENS | ≤ 0.10 & ≤ 0.15 / ≥ 0.5 | descriptive (routes P1 vs P2) |
| RB3 | EXPOSED | ≥ 0.8 | binding validity |
| RB3 | LEAK | < 0.10 (upper CI) / ≥ 0.20 | descriptive (C1 wording) |

**Statistics.**
- Every share carries a 95% Wilson CI. NEAR is set when the CI contains the threshold.
- Labels are read on bun_s1. bun_s2 confirms; a disagreement is reported as MIXED with no routing.
- The d 8 comparisons are descriptive. They are not C2 claims, so the ≥ 3-seed rule is not triggered, and §268's bun_s3 is added if it exists.

**OC (outcome priors):**
- P(I1 LOCALIZED) 0.55; P(PATTERN | LOCALIZED) 0.6; P(DILUTED at d 8) 0.4.
- P(I3 MEMORY-COMPUTE) 0.75; P(HEURISTIC-HOLES) 0.35; P(PASS-INVARIANT) 0.6.
- P(I4 ATTRIBUTION-VALID) 0.6 (linear-attention states are near-linear; trunk saturation is the risk, 2403.00745); P(DELTA-CARRIED) 0.5; P(DECODER-SHORTCUT) 0.35.
- P(TH2 gb FORWARD-BLIND) 0.65 (Note 260-5: H-EXTEND 0.08 / 0.02); P(VOID-RENDER) 0.15.
- P(RB3 SELECT-WITH-LEAKAGE) 0.35; P(DECODER-READS-TEXT) 0.6.

**Censoring.**
- A crashed part gets VOID-RERUN once, as its own line.
- A failed gate means no part runs (VOID-BUILD).
- Items with a wrong intact readout leave the flip denominators (the count is reported).
- C-SEL pairs with equal clean and corrupt readouts leave the NE metrics. If fewer than 30% are kept, the label is INSUFFICIENT.
- Non-finite gradients drop the item. If more than 5% of items drop, the part is VOID.

## 8. Scorer sketch (`theory/score_ip.py`, stdlib + statlib Wilson)
- **Fields read:** exactly the JSON keys above (`eval_ip_i1.json` … `eval_ip_rb3.json`, each with `def`) plus `gates_pass.json`. A field-path check greps `ip_parts.py` for each key (memory rule).
- **Fixtures:** 2 per label (≈ 40). Each validity gate gets a failing fixture, and a NEAR case is included.
- The scorer is G-D'd by Audit before launch.

## 9. Cost (RTX 3090, $0.2/h)
**Throughput [ASSUMED]:** ~2 ms per 120-token GPT-2-small pass-forward in fp32. The estimates below are in pass-forwards.

| part | pass-forwards | time | cost |
|---|---|---|---|
| I1 | 5 cond × 500 × (144 × 3 × 2 + top-5 extras) ≈ 2.3M | ≈ 1.3 h | $0.26 |
| I3 | ≈ 0.4M | 0.25 h | $0.05 |
| I4 | EAP-IG ≈ 0.1M + validation ≈ 0.7M | 0.6 h | $0.12 |
| TH2 | 4 model × arm cells | 0.15 h | $0.03 |
| RB3 | ≈ 0.45M | 0.3 h | $0.06 |

- With bun_s2 confirmation, set builds (44 s each) and G-TIME, the total is **≈ $1.1**; guard **$2**.
- A G-TIME line (100 items per part) projects the cost and aborts above the guard.

## 10. RED TEAM
**Not done by a separate sub-agent.** The required red-team spawn was refused twice by the session's concurrent-subagent limit (20). The items below are the lead's own adversarial pass against the code. **Audit must commission an independent red team before registration.**

1. **The lower cache leaks hooks across passes.** `EV.passes` L67 computes `m.lower` once per distinct input, so a hook on blocks 0-5 would reach every pass sharing that input, including all hold passes t > d. **Fix:** G-LOWER, plus per-pass lower recompute in `IP.run_passes`.
2. **Selection bias in I1.** Picking the top-5 heads out of 432 head × posmask cells and scoring them on the same items inflates joint5. **Fix:** the half-A / half-B split (§5).
3. **The manual attention is not SDPA-bitwise.** **Fix:** only `pattern`, `knock` and `qkv_split` use it; z / mlp / resid reuse `_block`'s exact ops (bitwise, G-EQ2); G-EQ1 sets the tolerance in fp32 and fp64.
4. **RB3's EXPOSED control is partly trivial.** Knocking out STEP also removes the subject name that the cue matches. **Fix:** report STEP_OPND-only knockout as the informative control (flip expected only for the value); EXPOSED stays as the inertness check.
5. **I3 text-recompute is confounded with memory use.** Under the state patch, a readout equal to the recipient's own v_t means text recompute; equal to op(v') means memory use. Without both shares the label is ambiguous. **Fix:** report stay and follow; MEMORY-COMPUTE needs follow ≥ 0.8 and stay ≤ 0.1.
6. **The C-VAL0 sweep has holes.** It does not cover Z_11 when the affine coefficient is 0 mod 11 (add_s). Mul with input 0 is an untrained cell. **Fix:** exclude and flag (§5).
7. **The decoder last hop (Note 264-1) can mask latent effects** in the K' 8 metrics of RB3 and TH2. **Fix:** every part reports the pass-t WS readout alongside the final answer; the MEAN-K arm in TH2; dec_text_share in I4.
8. **The TH2 mirror chain looks like a trained "chain" distractor.** bun rejects those under STD, so a PAIR-BOTH failure on member B could be distractor rejection rather than rendering. **Fix:** the validity bar uses both members separately (each ≥ 0.85), and the branch-first order is randomised.
9. **TH2 on bun under the generic cue is out of distribution** (bun never saw it). **Fix:** descriptive only; the routing label is read on gb_s1 / s2.
10. **EAP linearity may fail at the softmax saturation of selection heads** (2403.00745) and through the 8-pass recurrence. **Fix:** G-ATP is binding, with the patching fallback; the AtP* QK-fix is used for the attention-pattern nodes.
11. **Path-patching semantics in a loop.** "Freeze the others to clean" applies inside pass t only. Passes after t recompute from the patched state, which mixes direct and mediated effects. **Fix:** report R_out at pass t (local) separately from the final-answer effect (global).
12. **Cheaper falsifiers to run first:**
    - I1: the 144-head ALL-noising at d 4 t 2 alone (n 200, ~$0.01). If no head has NE ≥ 0.1, the label is DISTRIBUTED and the rest is skipped.
    - I4: the G-ATP subset before the full EAP-IG.
    - TH2: gb_s1 alone (~$0.01).
    - RB3: d 4 only before d 6 / 8.
13. **Multiplicity.** Five parts × several labels on one lineage. **Fix:** labels route only, not claim; bun_s2 must agree before any routing.
14. **Right padding.** The class masks and the ':' position must use each item's own length (`mask.sum - 1`). Knockout masks are built per item. Asserted in G-CLS.

## 11. arXiv ids (each abs page fetched 2026-11-03; titles match)
2310.10348 Attribution Patching Outperforms Automated Circuit Discovery; 2403.17806 Have Faith in Faithfulness (EAP-IG); 2403.00745 AtP*; 2304.05969 Localizing Model Behavior with Path Patching; 2211.00593 Interpretability in the Wild (IOI); 2304.14997 Towards Automated Circuit Discovery (ACDC); 2309.16042 Towards Best Practices of Activation Patching; 2404.15255 How to use and interpret activation patching; 2202.05262 Locating and Editing Factual Associations in GPT; 2305.15054 A Mechanistic Interpretation of Arithmetic Reasoning ... Causal Mediation Analysis; 2410.21272 Arithmetic Without Algorithms (bag of heuristics); 2301.05217 Progress measures for grokking; 2303.02536 Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS); 2505.12514 Reasoning by Superposition (continuous thought holds several frontiers; the TH2 SUPERPOSED escape); 2412.06464 Gated Delta Networks; 2102.11174 Linear Transformers Are Secretly Fast Weight Programmers.
