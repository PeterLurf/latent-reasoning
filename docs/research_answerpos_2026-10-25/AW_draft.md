# ANSWER-WRITE (AW) draft v0: write the INJECT state from the answer-position state of a per-pass sub-question (2026-10-25)

**Status:** draft for Theory → red team → Literature → Audit. Not registered; no § number (cells named `aw_*`). Nothing committed by the drafter.
**Basis:** the user's 10-25 decision (the next language latent arm writes the memory from the ANSWER position of a "How many <u> does <S_t> have?" decode, where the model computes v_t; GPT-2 small; ≈ $1–2 pilot). Evidence: Audit's locus probe (scratch/audit_s233_locus/locus.json, d500ca4c) and answer-position probe (scratch/audit_answerpos_probe/probe.json, af0f0eed); Notes 226-2, 227-2, 228-2, 229-1, 231-3, 233-2, 233-4; ST_v1.md / REVIEW_ST.md.
**$0 files in this folder (CPU, ≤ 2 threads, idle priority):**
- `aw_checks.py` → `aw_checks.json` (run, ALL_PASS; base GPT-2 only): token / position asserts, single-token write exactness (fp64), leak-fixture prototype, swap pairs and their training overlap, lengths.
- `aw_gate.py`: the binding gate G-AW0 (§2). **Not run on the init model**: it runs only after this rule is committed. Smoke on base GPT-2 (48 items, plumbing only) → `aw_gate_smoke.json`; the descriptive base-GPT-2 reference (`--init base`) → `aw_gate_base.json`. A first base run under a 0.90 control bar is kept as `aw_gate_base.json.void_bar090` (§2.2 disclosure).
- `aw_oc.py` → `aw_oc.json` (run): OC of every binding gate and the cost model from measured §228 / §233 files.

## 1. Mechanism (a)

### 1.1 The one change vs §233 LAT-ST: the write source
- **§233 LAT-ST:** pass t reads `[EOT] + enc(" " + text + " " + QUESTION(u, S_min(t,d)))`. **Every** non-pad token writes: S_t = W(S_{t−1}, {u_j}_j), u_j = (1 − γ) LN_6(h_6,j) + γ LN_w(h_w,j), ≈ 52 tokens at d = 2 (`aw_checks.json` LEN `think_st_mean` 52.4).
- **AW:** pass t reads the same input plus `enc(" The answer is:")` (4 tokens), i.e. a sub-question decode:
  - `x_t = [EOT] + enc(" " + text + " " + QUESTION(u, S_min(t,d))) + enc(" The answer is:")`.
  - **Only the last token (':') writes:** S_t = W(S_{t−1}, u_{':'}). Nothing is decoded: no logits, no token, no sampling on think passes.
  - At t = 1 this is exactly the P1-ASK+NAME model's class-(i) input [PROVEN, `aw_checks` TOK: 278 / 278 class-(i) items, `pass1_equals_askname_class_i_dec`].
- Implementation: the ':' rows of h_6 and h_w are gathered and `write_seq` runs on one token. It equals the one-hot-masked full-length write: fp64 relative error 2.1e-15 (state) and 8.1e-16 (∂/∂ qkv), over FAST_ADD × γ {0, 0.5, 1} × cached / uncached (`aw_checks` W2). The fp32 GEMM-reorder floor is 4.8e-7.
- **The name-token source (0.855) is not used.** Its forward cost is the same (blocks 6–11 must run over all positions either way), so it saves only the 4 suffix tokens (≈ 7 %). It carries less v_1 (0.855 vs 1.000). The gate reports it as descriptive.

### 1.2 Facts
- **E1 [MEASURED, locus.json, st_plst2_s1, d = 2, chance_majority 0.112]:**
  - v_1 at the pass-1 cue end, `i_hw_cue|v1` test_acc 0.247;
  - written state `iii_S1|v1` 0.287 (`iii_S1|v0` 0.672);
  - aux read vector `iv_read_aux1|v1` 0.183;
  - positive control `v_hw_s0end|v0` 1.000.
  - So the §233 write source barely carries v_1. S_1 holds the GIVEN v_0 better than the computed v_1.
- **E2 [MEASURED, probe.json, s233_askname_s1, class-(i) input]:**
  - `ans_pre_lnf|v1` 1.000 and `ans_post_lnf|v1` 1.000 at ':';
  - `qname_pre_lnf|v1` 0.855 at the question's name token;
  - layer-0 controls `ans_layer0|v1` 0.088 and `qname_layer0|v1` 0.083;
  - the model's own `model_argmax_acc_v1_restricted` 1.000.
- **E3 [MEASURED, 233-2]:** P1-ASK+NAME gives 1.000 in every class. That includes (iv), S_2 with S_1 GIVEN in the text (n 257): pass t's job in text form is done in one pass.
- **E4 [MEASURED, 233-2 / 233-4]:** one pass cannot chain two steps. CUED-DIRECT d = 2 0.089 (local) and 0.097 (cloud); plain DIRECT 0.093.
- **E5 [MEASURED, 233-4]:** §233's state and cue were both unused: LP5 drop −0.007, CUE-WRONG drop 0.001, pass-1 aux on d = 2 0.33.
- **E6 [PROVEN, generator; F5 of ST_draft]:** at d ≤ 2, S_t's defining sentence needs only v_{t−1} plus text operands. A state carrying the 11-way value v_{t−1} is sufficient input for pass t.

### 1.3 Derivation: why the state should now carry v_t
- **D1 [DERIVED from E2] The init write source carries v_1 at ≈ 1.00.**
  - At step 0, fade = 0 and W_o = 0, so the read adds exactly 0 (asserted in aw_gate). h_w at ':' is therefore the one-pass model's pre-ln_f residual (E2's feature).
  - u_{':'} at γ = 1 is LN_w(h_w) = n(h_w) ⊙ w_11 + b_11, and ln_f(h_w) = n(h_w) ⊙ w_f + b_f. Both are per-dimension affine maps of the same normalised vector.
  - The probe standardises every feature, which removes any per-dimension affine map with nonzero scale. So probe(u1) = probe(`ans_post_lnf`) = 1.000, up to the 1e-6 sd epsilon and LBFGS.
  - **So G-AW0 (§2) is predicted to pass and is chiefly a build check** (right position, right init, right construction). The real uncertainty is in training.
- **D2 [DERIVED] One write puts v_1 into every later read, linearly.**
  - From the zero state, an additive head writes φ(k) vᵀ and a delta head writes β v k̂ᵀ, with v = W_v u.
  - Any later read gives φ(q)ᵀφ(k) · v (additive; the scalar is > 0 because φ = elu + 1 > 0) and β (k̂ · q̂) · v (delta). The read at each position is a position-dependent scalar times the same v; then W_o.
  - So v_1 is linearly present at block 6 of every position of pass 2, of the aux sequence and of the decoder, once W_o ≠ 0, iff it is linearly decodable from W_v u. [ASSUMED: W_v, initialised from block 6's 768 × 768 c_attn V rows, keeps the v_1 code; aw_gate's descriptive `s1|v1` probe measures this at init.]
- **D3 [DERIVED] Why §233's write diluted it.** S_1 was a sum of ≈ 52 per-token writes, none of which carried v_1 well (E1: 0.247 at the best position). AW makes S_1 one term from a source at 1.000.
  - LP2 (the pass-1 aux reading v_1 from S_1) becomes a storage-and-copy problem like E1's given-value case (`iii_S1|v0` 0.672 with diluted writes; F3 / 231-3: v_0 1.000 held at write sources).
- **D4 [DERIVED] What enters the state is what the answer head would emit.** ln_f(h_w) at ':' unembeds to v_1 (E2: 1.000), and u1 is an invertible per-dimension affine map of the same normalised vector (D1). The write is continuous (W_k, W_v of u), never discretised.
- **D5 [DERIVED] The gradient path to the read opens.** W_o = 0 at init, so ∂L/∂S_1 = 0 at step 1. But ∂L/∂W_o ∝ φ(q)ᵀS_1 ≠ 0 once fade > 0 (step ≥ 1 of 500). The decoder-CE and aux paths into S_1 then open, as in §228 / §233 (G3 checks each term).
- **D6 [DERIVED] At d = 2 a pass can come from "pass 1 + decoder".** The decoder input is the S_2 question, which is pass 2's own job. The decoder can apply op 2 to v_1 read from S_K. That counts as serial use: two serial computations, with the state carrying the intermediate (ST_draft §7).
  - The descriptive label is **DEC-STEP2**: LP3 passes but WS-READOUT at pass 2 (d = 2) is < 0.50.
  - Pass-to-pass chaining is first required at P-4 (d 3–4).

**Assumptions the pilot tests.**
- **A1 [ASSUMED; the main risk] Read-into-compute.** Blocks 6–11 of pass t learn to use v_{t−1}, supplied by the block-6 read, in place of the text sentence "S_{t−1} has v u." (E3 shows they compute op t when the value is a text token reached by attention). This is untested on language; it is §233's A2, never reached because §233's S_1 was empty.
- **A2 [ASSUMED]** The INJECT objective (decoder CE on S_K + aux CE on S_t, t ≤ d) gives enough pressure for A1 within 12k steps.
- **A3 [ASSUMED]** The trunk's ':' code survives 12k INJECT steps. No think-pass CE holds it; descriptive WS-READOUT tracks it.

**Risks named in the brief.**
- **R1, not read (most likely failure; truth `pass1_only`):** S_1 carries v_1 (LP2 passes) but nothing uses it (LP3 / LP5 fail). Reading: **AW-NOT-READ**, i.e. the bottleneck is reading back, not writing. That inverts 233-4's "bottleneck is writing … (and reading it back)" and costs ≈ $1.8.
- **R2, readout-only:** the ':' state may encode the *answer token* (an unembedding direction), not a quantity the op circuits accept through a residual-added read. D2 shows the code is linearly present; whether the compute circuits use it in that subspace is A1. For d ≤ 2 the 11-way value is all the next step needs (E6), so "readout-only" fails only through A1, not through missing content. At d ≥ 3 (mul_s back-references) the earlier passes' additive writes must stay readable (ST's A4).
- **R3, one-pass gain without the state** (multi-task trunk improvement): caught by LP5, MEAN-STATE and SWAP-K.
- **R4, early writes from mid-layer:** steps ≤ 2,000 write from u0 = LN_6(h_6) at ':' (γ = 0; v_1 content unknown, reported by the gate). The γ ramp 2,000 → 6,000 is kept for single-change discipline (open question 3).

**Predictions [ASSUMED]:**
- LP2 (`aux_vacc_by_d_pass["2"]["1"]`, last bin): point 0.90, 80 % interval 0.55–0.99, P(≥ 0.80) ≈ 0.65.
- P(LP3 ∧ LP5 ∧ MEAN-STATE ∧ SWAP-K | LP2) ≈ 0.30, so **P(P-2 PASS) ≈ 0.19** (aw_oc `P_P2_pass` 0.189).
- WS-READOUT at pass 1 ≥ 0.90 at the end (P ≈ 0.7).
- CD-AN d = 2 < 0.40 (P ≈ 0.75).

## 2. The $0 gate G-AW0 (Audit's condition), the init, and single-change discipline (b)

### 2.1 Init: the P1-ASK+NAME trunk s233_askname_s1
`boxes/runs_fsq/s233_askname_s1/model.pt`, sha256 c00f7ec0bd7977fe…; GPT-2 small, full FT, 8k steps on the 4060.
- **Why this init:**
  - (i) The gate needs a trunk that computes v_1 at ':'. Pretrained GPT-2 (§228 / §233's init) computes no intermediate at any write source (231-2: G_LN6 0.106, G_HTOP 0.102).
    - **The answer position too [MEASURED, `aw_gate --init base`, N 3,000, chance_majority 0.113]:** `u1|v1` 0.102, `u0|v1` 0.083, `name|v1` 0.100, `s1|v1` 0.113, `iv_u1|v2` 0.082 (all chance); `l0|v1` 0.088; v_0 controls `v0ctl|v0` 0.828 and `v0ctl_u0|v0` 1.000. The verdict file reads FAIL.
    - So G-AW0 fails on §233's init by measurement, and AW needs a trunk that computes v_1.
  - (ii) It already does pass t's job in text form (E3) and reads the question's name (233-2; the §228 trunks did not, F4: ASK0 0.084).
  - (iii) It isolates the question §233 left open: given a write source that carries v_t, does the loop store it and use it?
- **Rejected:**
  - pretrained GPT-2 (`u1|v1` 0.102: G-AW0 would FAIL);
  - s228_pdir2c_s1 (ignores names, F4; descriptive reference `--init pdir2c`);
  - §228's base (= pretrained GPT-2).
- **How the init is built:** the trunk is loaded strict before DeltaMem is built. DeltaMem's ln / qkv are copied from the loaded block 6 and ln_w from the loaded block 11's ln_1, as InjectN does; g_add is calibrated on the loaded trunk.
- **Exposure, disclosed [DERIVED from code]:** askname ran inject_ow.main (seed 1, d_max 2), so its 8k steps drew the identical (d, item) sequence that AW and CD-AN draw in their first 8k steps. AW re-sees those training items with a different objective. TEST2 is excluded from all three. CD-AN has the same exposure, so the comparison is matched.

### 2.2 G-AW0 rule (pre-stated; `aw_gate.py`, binding, $0, before any spend)
- **Model:** built as the arm builds it (2.1); fp32, no_grad. S = zero state, fade = 0, and read == 0 is asserted.
- **Items:** 3,000 d = 2 items, OW.Stream(23471, EXCL), seed unused (grep 10-25). Split by sig 60 / 20 / 20.
- **Probe:** Audit's protocol (standardised multinomial LR, LBFGS 200 iterations, wd ∈ {1e-4, 1e-3, 1e-2} on val). Per-item test predictions are stored.
- **Binding feature:** u1 = mem.ln_w(h_w) at ':' of x_1, the write source from step 6,000 on (γ = 1). Target v_1.
- **PASS iff all four hold:**
  - `u1|v1` test_acc ≥ 0.80;
  - `l0|v1` ≤ majority + 0.05;
  - `v0ctl|v0` ≥ 0.80 (u1 at the end of S_0's GIVEN sentence; 231-2's positive-control bar);
  - `read_max_abs` == 0.
- **Disclosed:** the drafted control bar was 0.90. The descriptive base-GPT-2 reference then read `v0ctl|v0` 0.828, and the gate script marked it UNREADABLE. That is consistent with 231-3's frozen-GPT-2 top-layer value (G_HTOP 0.861, d = 2).
  - The bar was lowered to 231-2's 0.80 before any askname reading. `v0ctl_u0|v0` (mid-layer, where 231-3 read 1.000) was added as descriptive.
  - The first base output is kept as `.void_bar090`, and the base reference was re-run under the amended script.
- **Outcomes:**
  - `u1|v1` < 0.80 with valid controls → **FAIL: AW stops at $0** and goes to the user.
  - A control fails → UNREADABLE: fix once, re-run once.
- **Descriptive:**
  - u0 (γ = 0) and uhalf (γ = 0.5);
  - s1, the single-token write of u1 (D2);
  - `name|v1` (the 0.855 source);
  - `iv_u1|v2` (pass 2's job in text form);
  - the trunk's argmax at ':' for x_1 → v_1 and for iv → v_2;
  - the base / pdir2c references.
- **Prediction:** u1 ≈ 1.00 (D1).
- **OC** (n_test 600; aw_oc `G_AW0`): P(pass) = 1.000 at a true 1.00; 0.9998 at 0.855; 0.52 at 0.80; 0.003 at 0.75.
- **Run cost:** ≈ 15 min CPU at 2 threads (base reference: 818 s for N 3,000).

### 2.3 Deviations from §233 LAT-ST (st_plst2_s1)
- (w1) think input + `" The answer is:"` (+4 tokens; passes t > d hold the S_d input, as §233);
- (w2) write from the last token only (one-hot write; gathered implementation, W2 above);
- (w3) init = s233_askname_s1 (§233: pretrained GPT-2), with DeltaMem built from the loaded trunk;
- (w4) binding comparator CD-AN = §233's CUED-DIRECT recipe from the same init;
- (w5) new eval parts: WS-READOUT, SWAP-1, SWAP-K (§3); the build fixture FX-LEAK (§4);
- (w6) new args `--write_src {all, answer}`, `--init_trunk <path>` (+ its sha256), and the format string.
- **The treatment** is (w1 + w2), "the write source". (w3) is the gate's precondition, and CD-AN controls it. The attribution control ST-AN (w3 alone; §5) runs only on a pass.
- **Unchanged:** K = 8, the cue form and hold, the decoder input (§228's `text + " The answer is:"`, no cue), the aux (read-only on S_t, R_step target, AUX_W reweight, fade 48000), lr 1e-4, bs 64, 12k steps (+ ST_v1's still-rising extension to 16k), write mix 2,000 → 6,000, FAST_ADD, the per-distinct-input lower cache, stream / EXCL / d law, seed 1, TEST2.

### 2.4 Build packet (Experiments; Audit G-D; new module `boxes/inject_aw.py`, imports inject_st / inject_ow read-only)
- **G1:** `--write_src all --init_trunk ''` is bitwise inject_st `--cue step` (both arms, 50 steps).
- **G2:** the single-token write vs the one-hot-masked write: fp64 state + gradient ≤ 1e-10 relative (prototype passes: W2).
- **G2b:** the per-distinct-input lower cache, batch-of-one, fp64.
- **G3:** per-term gradflow; decoder CE and the aux at each pass give nonzero gradient into DeltaMem's write projections and W_o.
- **G4:** a plain-path smoke (no new flags).
- **G5:** mem_check at the d = 2 shape (micro chosen; the T-token write activations are gone, so micro 1 may fit [ASSUMED]).
- **G6:** args diff vs st_plst2_s1 = {write_src, init_trunk, init_sha, format, module shas}; CD-AN vs st_pcd2_s1 = {init_trunk, init_sha}.
- **G7:** FX-LEAK (§4).
- **G8:** score_aw fixtures + a field-path check.
- **G9:** decoded inputs, 3 per d ∈ {1, 2, 4, 8}, including the hold passes.
- **G10:** `aux_vacc_by_d_pass` pools to `aux_vacc_by_pass`.
- **G11:** an ast check that every eval part keeps `@torch.no_grad` (233-4 lesson).
- **G12:** a strict init load and sha assert. mem.qkv.weight equals the loaded block-6 c_attn weight transposed.
- **G-EXP:** the per-1k-bin `ihash` of LAT-AW and CD-AN equals st_plst2_s1's (same items).

## 3. Comparators and readings (c): pilot P-2
**Cells (one RTX 3090 host; 12k; d ~ U{1, 2}):**
- `aw_plaw2_s1` (LAT-AW);
- `aw_pcd2_s1` (CD-AN: `--arm direct --cue all`, init askname);
- plain DIRECT `s228_pdir2c_s1`, reused iff the host is a 3090 (else fresh, +$0.05; descriptive, and the CUE-ONLY bar);
- `st_pcd2_s1` (base-init CD), descriptive.

| gate | rule | role |
|---|---|---|
| LP0 (each arm) | no non-finite row; no 100-step mean gn_preclip after window 10 > 2 × max(windows 3–10) (score_st `lp0`) | binding |
| LP1 | LAT d = 1 train acc, last 1k bin ≥ 0.90 | binding |
| LP2 | `aux_vacc_by_d_pass["2"]["1"]`, last 1k bin ≥ 0.80 | binding |
| **LP3** | LAT TEST2 d = 2 acc (K′ 8) ≥ 0.70 **and** ≥ CD-AN + 0.30 | **binding** |
| **LP5** | MEAN-AT-1 on TEST2 d = 2 at K′ = d: acc(intact) − acc(S_1 := TEST2-d-2 mean of S_1; passes 2..K own, cue kept) ≥ 0.30 | **binding** |
| MEAN-STATE | pooled (S := TEST2 mean of S_K): acc ≤ CD-AN + 0.10 | binding |
| **STATE-USE** | (a) FX-LEAK passes at build (§4); (b) **SWAP-K** on the 1,000 same-skeleton d = 2 pairs (st_cue_check C5 law; cue identical by C2): recipient's S_K := donor's S_K, decoder on the recipient's text; U = {donor final, cf_a(t = 1)} (= cf_b in 1,000 / 1,000 kept; own ∉ U by the kept rule); **follow_U ≥ 0.50 and ≥ floor + 0.30**, floor = share in U of the MEAN-K answers on the same pairs | **binding** |
| CUE-ONLY | think = `[EOT] + enc(" " + cue_t) + SFX` (no text), decoder unchanged; answer on d = 2 ≤ max(majority, plain DIRECT d = 2) + 0.05 | guard (VOID) |
| CUE-WRONG | at pass t the cue names S_{t+1} (t < d), else S_0; drop at K′ = d ≥ 0.20 → label C1-TOLD, else plain C1 | label |
| descriptive | WS-READOUT (the trunk's argmax at ':' of pass t vs v_min(t,d), per pass, no_grad, never fed); SWAP-1 (S_1 := donor's S_1, passes 2..K own: follow cf); pass-2 aux on d = 2; K′ 1; LP5 at K′ 8; a v_1 probe on the trained S_1; CD-AN d = 1; plain / base-init CD d = 2 | — |

- **Pairs [PROVEN, aw_checks PAIRS]:** 1,000 kept of 1,613 tried. Overlap with the 768,000 replayed training items is 0, and with TEST2 d = 2 it is 0.
- **Outcomes:**
  - **PASS** = LP0–LP3 ∧ LP5 ∧ MEAN-STATE ∧ STATE-USE ∧ ¬VOID. Then the attribution control ST-AN, then P-4 (§5).
  - **AW-NOT-STORED:** LP2 fails. If WS-READOUT at pass 1 is ≥ 0.80, the source still carries v_1, so the failure is write / storage.
  - **AW-NOT-READ:** LP2 passes, LP3 fails, LP5 < 0.10 (R1).
  - **AW-RECOMPUTE:** LP3 passes but LP5, MEAN-STATE or SWAP-K fails.
  - **INIT-ONE-PASS:** CD-AN d = 2 ≥ 0.70, so LP3 cannot be met. The plan plus the init make one pass suffice; this is a finding, not an arm fail.
  - Each is a stop. There is no fallback inside AW (no problem → fix loop).
- **Censoring (pre-stated):**
  - A dead or non-finite cell → UNREADABLE, and one re-run on the same class. A crash → resume. A missing file reads pending.
  - Every TEST2 item and every kept pair counts. No item is dropped.
  - Answers are an argmax over the 11 number tokens, so no item can be "unparsed".
  - A part that crashes late is re-run eval-only from model.pt (233-4 precedent), with per-item identity checked against any partial file.

## 4. Leak control (d): the final answer sees v_t only through the memory
**Channels from a think pass to the decoder:**
- (T) tokens: think passes emit none. No `g.logits`, argmax or sampling in the training think path; WS-READOUT runs in eval only, under no_grad, and its outputs are stored, never fed.
- (Q) sequence: dec = `[EOT] + enc(" " + text) + enc(" The answer is:")`, a function of the item text only (asserted equal to tok_ow's dec). No think-pass token is in any decoder sequence.
- (K) cache: the lower cache belongs to the think inputs; the decoder runs fwd_read with no cache.
- (S) state: S_K, the intended channel.

**Claim [PROVEN by construction]:** decoder logits = F(θ, dec_tokens(text), S_K).
- θ is shared across items and holds no per-item content at test time, and dec_tokens do not depend on any think pass.
- Hence any per-item information computed in a think pass, v_t included, reaches the final answer only through S_K. v_t is never a token anywhere: the text holds only v_0, operands and distractor values, and the cue and suffix hold no number [PROVEN, aw_checks TOK: 18,000 inputs, 0 number tokens and 0 digits after the text].
- So AW is not text-CoT: there is no discretised intermediate in any context.
- **No detachment:** it would cut the write's learning, and leak control does not need it. The aux sequences are teacher-forced on R_step (v_t is a target token, as in §228 / §233). They read S_t, return logits only, never modify the S chain and are dropped at test.

**FX-LEAK (build fixture G7, binding before spend; prototype passes on base GPT-2, aw_checks LEAK):**
- (a) With S_K fixed, the decoder logits are bitwise invariant across think inputs: real cues, wrong cues, another item's text.
- (b) Positive control: a planted leak (the think pass's ':' argmax token appended to dec) is detected.
- (c) The state channel is live: different think inputs give different S_K and different logits, so the fixture is not vacuous.
- (d) S_K is bitwise equal with the aux computed and not computed.
- (e) dec tokens equal tok_ow's.

## 5. OC, censoring, cost (e)
**OC (aw_oc.json; one seed; 20,000 reps; P(P-2 PASS | truth)):**

| truth | LP2 | LP3 | LP5 | MEAN-STATE | SWAP-K | **PASS** |
|---|---|---|---|---|---|---|
| no effect (aux 0.33, LAT = CD = 0.10) | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | **0.000** |
| pass1_only (stored, unread) | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | **0.000** |
| pass 1 + decoder (D6) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** |
| strong serial | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** |
| marginal (LAT 0.75, CD 0.40, aux 0.82, drop 0.35, follow 0.60) | 0.88 | 0.99 | 1.00 | 1.00 | 1.00 | **0.877** |
| at bars | 0.52 | 0.38 | 0.50 | 0.51 | 0.51 | 0.020 |
| weak read (LAT 0.45) | 1.00 | 0.00 | 0.51 | 1.00 | 0.00 | **0.000** |
| recompute (state unused) | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | **0.000** |
| init solves one pass (CD-AN 0.75) | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.000** |

- CUE-ONLY: false trip at majority 0.000; power at 0.25 1.000.

**Cost** (measured: st_plst2_s1 train 19,426 s at micro 2; st_pcd2_s1 1,160 s; s228_plat2_s1 G-TIME 0.729 s/step at micro 1; st_plst2_s1 eval 759 s; rate $0.20–0.27 per host-hour; §233 P-2 realised ≈ $1.61):
- LAT-AW train time is 3.05–5.75 h. High = ST's measured time × AW's +7 % tokens. Low = s228's micro-1 rate × 1.25 tokens [ASSUMED fits at micro 1].

| stage | host-h | $ low–high | guard ×1.3 |
|---|---|---|---|
| G-AW0 + build gates (local) | — | 0 | 0 |
| **P-2** (LAT-AW + CD-AN; +$0.05 fresh DIRECT if not a 3090) | 3.9–6.6 | 0.78–1.78 | **2.31** |
| ST-AN attribution (on a pass only) | 5.9 | 1.17–1.58 | 2.05 |
| P-4 (d ≤ 4, 16k, two hosts) | 6.9–11.9 | 1.38–3.21 | 4.17 |
| Stage 1 (U{1..8}, 48k, two hosts) | 35.9–60.6 | 7.19–16.37 | 21.28 |
| **family** | | | **29.81 → cap $30** |

- **P-2 admission:** G-TIME on the host (60 steps). Admit iff the projected P-2 is ≤ $1.85 (233-3's bound); guard $2.31; the 4-BAD rule.
- **Stage 1 is re-costed from P-4's measured s/step.** If ×1.3 exceeds the cap left, it goes to the user. The high anchor is §233's slow micro-2 rate.
- **Priors [ASSUMED]:** G-AW0 0.97; LP2 | gate 0.65; the rest of P-2 | LP2 0.30 → P(P-2) 0.19; P-4 | P-2 0.5; C1 | P-4 0.4 → **P(C1-TOLD-AW) ≈ 0.04. Expected spend ≈ $4.2** at the high rate; the P-2 spend alone is ≤ $1.78.
- **Only on a P-2 PASS:**
  - **ST-AN** = §233 LAT-ST (all-token writes, no suffix) from the askname init, 12k, same class. If it fails LP3 → **WRITE-SOURCE-CREDITED**. If it passes → **INIT-SUFFICES**, reported plainly. P-4 runs on AW either way.
  - **P-4** = ST_draft §6.3 with AW / CD-AN on two 3090s: LP4 (d 3 ≥ 0.60, d 4 ≥ 0.50, each ≥ CD-AN + 0.25); LP5-4 (MEAN-AT-⌈d/2⌉ drop ≥ 0.30, pooled over d 3–4); CUE-ONLY and SWAP-K on d 3–4.
  - **Stage 1** = ST_v1 §4 / ST_draft §6.4 with AW and CD-AN (R1, R2, R2′, R3; G-W0 band from CD-AN). It needs two hosts, HOSTPROBE (redeploy if the projection is > 40 h), off-host full-state checkpoints every 2k with a tested resume, and guard marks to step 48,000. A second seed only with separate approval.

## 6. Novelty sentences for Literature (no "new" / "first" before the check)
- **N1:** "In a memory-state latent loop on a pretrained LM (INJECT / DeltaMem, GPT-2 small), each pass answers a sub-question about intermediate S_t, and **only the answer-position hidden state** (never decoded to a token) is written to the memory. The state therefore carries the value the LM would answer, and a cue- and init-matched one-pass comparator credits any depth gain to the loop."
  - Nearest known (ours): Coconut 2412.06769 (the last hidden state fed back as the next input embedding); least-to-most 2205.10625, successive prompting 2212.04092, self-ask 2210.03350 (sub-questions answered in text); planning tokens 2310.05707; §233 ST.
  - To locate: writes from answer / readout positions into a recurrent or matrix memory; sub-answer hidden-state feedback; RMT-style memory tokens (2207.06881, Literature to verify the ID).
- **N2 (diagnosis):** "The INJECT write failed on word programs because its source did not carry the intermediate (v_1 0.247 at the cue end, 0.287 in S_1). The same quantity is linear at 1.000 at the answer position of a one-pass model asked for it." Kin: Biran 2406.12775, Yang 2402.16837.
- **N3 (method, narrow):** single-token writes make each pass's contribution to the state attributable to one position, which makes MEAN-AT-1 / SWAP-1 / SWAP-K cleaner readings of serial use. Mean ablation and swaps are standard (SCIT 2608.27265 / §221).

## 7. Open questions for the red team
1. **Init as a second change.** Is CD-AN enough to control (w3)? Or must ST-AN (w3 alone) run inside P-2 (+≈ $1.6), so that a pass is attributable to the write source at once?
2. **Exposure.** AW re-sees askname's 8k training items (same stream). Does that bias LP3 vs CD-AN, which re-sees them too?
3. **The γ ramp.** Steps ≤ 2,000 write from LN_6(h_6) at ':' (unknown v_1; the gate reports u0). Should γ start at 1 (one more deviation)? Or is the ramp harmless because LP2 is read in the last bin?
4. **A1 pressure.** No loss asks pass 2's ':' for v_2 except the pass-2 aux read through S_2. Should a pre-named arm add a think-pass answer CE at ':' (target v_t; "SOURCE-CE")? Would that make the write source a supervised text-free CoT, and does that matter for C3 / C4?
5. **D6.** At d = 2, "pass 1 + decoder" passes P-2 without pass-to-pass chaining. Should P-2 require WS-READOUT pass 2 ≥ 0.50 (chaining), or is P-4's LP4 the right place?
6. **SWAP-K.** Is U = {donor final, cf} with a MEAN-K floor sound when the decoder can recompute from the recipient's text? (Its own answer is excluded by the kept rule.)
7. **Readout-only (R2).** Is there a $0 test that the read vector can be *used* (not only decoded), e.g. splice the read of a d = 1 item's S_1 into a text where S_1's sentence is masked, on the askname trunk with a briefly fitted W_o?
8. **Hold passes.** Passes 3..8 at d = 2 write the S_2 answer state (≈ v_2) seven times. Do repeated additive writes saturate or swamp v_1 for back-references at d ≥ 3?
9. **Cost.** The ST-measured micro-2 rate is the high anchor. Is a two-host P-2 (LAT on one host, CD-AN on another) worth it for wall clock alone?

## 8. Main risks
1. **Not read (R1 / A1):** S_1 carries v_1 but pass 2 / the decoder never learns to use the read. This is the most likely fail (truth `pass1_only` → PASS 0.000), labelled AW-NOT-READ.
2. **The init carries the result (attribution):** a pass may come from the askname trunk rather than the write source; ST-AN decides, on a pass only.
3. **Readout-only code (R2):** the stored code may be an answer-token direction that the op circuits do not take as input through a residual read.
4. **D6 scope:** a d = 2 pass may be "pass 1 + decoder", not a chain; P-4 must show chaining.
5. **Code drift (A3):** the ':' code may erode under the INJECT losses (WS-READOUT shows it). Early writes come from mid-layer (R4).
6. **Scope:** C1-TOLD (plan given by the generator), one seed, GPT-2 small, TOPO word programs. It adds nothing to C4.
