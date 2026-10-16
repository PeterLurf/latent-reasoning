# NOVELTY_4: VERSION-BANK + READY-RECODE against prior art (2026-10-15 23:19 EDT)

**Scope.** Pre-registration novelty check of docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md (read in full, with Addendum 1).
- Full texts read (arXiv HTML; AttnRes from the PDF): 2002.09402, 2609.03379, 2610.11570, 2511.08577, 2609.35748.
- Also read in full text: 2601.21582, 2510.07739, 2603.15031, 2510.14095, 2511.07384, 2608.11233, 2502.05171, 2510.25741, 2507.10524, 2409.15647, 2311.12424, 2603.08391, 2609.33144, 2205.15659, 2209.11142, 2404.03592.
- Every ID was checked on its arXiv abs page today (curl, citation_title). Also checked: 2606.24898, 2211.09961, 2106.02997, 2303.02536, 1506.03099, 2112.00826.
- **Tags:** [THEOREM], [DERIVATION] (a definition or a design fact), [EMPIRICS]. Quotes are verbatim.

## Verdicts

| # | Component | Verdict | Closest prior art, and what differs |
|---|---|---|---|
| 1 | Output-version bank, read by every core layer at all positions, in a pretrained retrofit | **PARTIALLY ANTICIPATED** | **TaH 2511.08577 / TaH2 2609.35748:** every layer of a pretrained Qwen3 attends to all earlier iterations at all causal positions. But they store **same-layer K/V**, loop the **full stack**, and start each pass from a logit-weighted embedding (TaH) or an input-injection updater (TaH2). **Feedback 2002.09402:** an output memory read by every layer, but along tokens, not passes, and from scratch. **RecurTrace:** own-layer, same position, W = 3. **What is left:** a bank of *bounded-core outputs*, re-projected by each core layer's own c_attn, unwindowed, in a 6/4/2 GPT-2 retrofit. |
| 2 | PURE-ANNEAL gather as the pass input | **PARTIALLY ANTICIPATED** | **InfiLoop 2610.11570:** a softmax mixture over *all* earlier iterations as the next input, with a content-free shared pseudo-query, RMSNorm keys and an age decay. **AttnRes 2603.15031:** the same over layers. **MeSH 2510.07739:** a per-position softmax read over whole-sequence slots as the next state. **New:** cross-position addressing (slot j may pull slot i ≠ j), wpe-only queries, and annealing from bitwise CARRY through a scheduled buffer. |
| 3 | Question-free shallow coda | **ANTICIPATED** (nested) / **PARTIALLY** (GSM8K decoder tokens) | **Huginn (2,4,2) and McLeish's retrofit:** p = C(s_r), a 2-layer coda that reads only the last recurrent state. Prelude-encoding the decoder tokens "among themselves only" was not found. **Flag:** VB's coda also reads the bank, which appears to include V_0, so it is not question-free as specified. |
| 4 | Readiness-window answer loss | **ANTICIPATED in substance** | **Fan 2409.15647:** the answer is supervised "only after a certain number of steps", T(n). **Yang 2311.12424:** a truncated loss window [b−T, b]. **Altabaa 2510.14095:** a 1{Depth(x_i) ≤ t} mask. VB's [n_i, n_i + U{0..4}] window combines these; the difference is minor. |
| 5 | Linear per-slot recode target | **PARTIALLY ANTICIPATED** | **Altabaa:** "a shared linear readout layer" per token, at every iteration, with factored targets. **Shapiro 2608.11233:** exact per-loop intermediate CE in a pretrained retrofit. **CLRS:** per-step hints. **New:** a slot-aligned rendering of serial reductions (consumed span → VOID, HOLD), plus the changed / unchanged / value reweighting. |
| 6 | Minimal-correction teacher forcing (MCT) | **PARTIALLY ANTICIPATED** | Per-step teacher forcing exists: CLRS hints (p = 0.5) and Altabaa's discrete re-embedding. The edit's algebra is the distributed interchange intervention b + Rᵀ(Rs − Rb) (ReFT 2404.03592). Using a minimum-norm, row-space correction of the student's own continuous latent as *scheduled teacher forcing in a looped LM* was **not found**. |
| 7 | PROBE / VALUE-SWAP / STALL | **PARTIALLY ANTICIPATED** (PROBE, VALUE-SWAP); **STALL not found** | **Per-loop readers:** Shapiro's frozen same-reader protocol; the logit lens in 2609.33144. **Interchange:** 2106.02997 / DAS 2303.02536. **STALL's nearest relative:** path independence (Anil 2211.09961). A copied-pass test against count keying was not found. |

**Overall.** "Novelty is by combination" (draft §3) is confirmed. No checked paper has VB's architecture. The nearest architecture is TaH / TaH2; the nearest objective is Altabaa.

## 1. The four requested papers (plus TaH2), axes (a)–(h)

| | (a) stored | (b) readers / positions | (c) pass start | (d) per-pass depth | (e) decoder sees input | (f) per-pass supervision | (g) retrofit | (h) task / scale |
|---|---|---|---|---|---|---|---|---|
| **VB** | core output V_t (T slots) per pass; all kept | core layers 6–9 + coda; **all positions, all versions** | gather over bank (PURE-ANNEAL), or N(V_{t−1}) | 4 blocks | no raw tokens; V_0 via bank (flag) | R2 linear recode every t; answer for t ≥ τ | GPT-2 124M | nested arithmetic n ≤ 8 → GSM8K |
| **Feedback** 2002.09402 | m_t = Σ_l softmax(w^l) x_t^l, one vector per *token* | all layers; all past tokens | token embedding; attention context = memory | full model per token | seq2seq: decoder only | none (LM / task loss) | scratch | algorithmic toys, WMT14, enwik8, WT103, RL |
| **RecurTrace** 2609.03379 | layer ℓ's own x_ℓ^(t′), last W = 3 loops | each looped layer; **same position only** | carry h + α·RMSNorm(e) (α init 0) + gated LMA term | 3 middle layers | yes (frozen decoder layers over all positions) | none per loop; LM loss at sampled depth; halting BCE vs a loss-gain oracle | frozen Qwen3 0.6–8B, ≤ 2.2% trained | 1.15M direct-answer items; MathQA 56.9 at 2.0 loops |
| **InfiLoop** 2610.11570 | candidates z_i as a running N_t / D_t (O(d)) | next-iteration input only; same position | h_t = Σ_i α_{t,i} z_i | 2-layer block (nested TRM) | yes (f_θ(h; x)) | TRM deep supervision | scratch, 7M | Sudoku-Extreme 97.9, Maze, ARC |
| **TaH** 2511.08577 | **same-layer KV** of every depth | every layer; positions j ≤ i, depths k ≤ d | logit-weighted embedding p^(d)E + residual; LoRA for d > 1 | full stack | yes | NTP at the oracle depth d_i^π | Qwen3-Base 0.6–4B | Open-R1 SFT; math / code / QA, d_max = 2 |
| **TaH2** 2609.35748 | executed KV, s ≤ t, j ≤ m | every layer (extended duo-causal) | U_ψ(e_t, h^(m)) injection updater | full stack | yes | stopping-weighted NTP + decider BCE (online lookahead labels) | Qwen3-Base 1.7–8B | AIME, code, QA, tools; M ≤ 8 |

**Definitions (verbatim):**
- **Feedback** [DERIVATION]: "z_t^l = Attn(x_t^l, {m_{t−τ}, …, m_{t−1}})"; "m_t = Σ_{l=0}^{L} Softmax(w^l) x_t^l".
  - [EMPIRICS] "top-only memory" "closely matched" the full design; "when all layer attend to the first layer output, it works as good as the standard Transformer".
  - This is VB's pattern (store outputs, read raw at every layer), but on the token axis.
- **RecurTrace** [DERIVATION]: "M_ℓ^(t) = {x_ℓ^(t′) | max(1, t−W) ≤ t′ ≤ t−1} is a sliding window of the layer's own outputs"; "reads only same-position states from a fixed three-loop window, which … excludes older states and cross-position memory".
  - The loop-distance bias "−β_h(t−t_j)" is ALiBi-initialised. That is VB's γ_{ℓ,h} init.
- **InfiLoop** [DERIVATION]: "h_t = Σ_{i=0}^{t} α_{t,i} z_i, α_{t,i} = exp(s_i − λ(t−i)) / Σ_j exp(s_j − λ(t−j))", with "s_i = q^⊤ RMSNorm(z_i)". Here q is "a learned pseudo-query shared across iterations", and "The aggregation operates independently at each token position".
  - "‖h_t‖ ≤ max_{0≤i≤t} ‖z_i‖". This is VB's [PROVED, REDTEAM §E] bound, already stated there; cite it rather than claim it.
- **TaH** [DERIVATION]: "X_{≤i}^{(≤d)} = {x_j^{(k)} | j ≤ i, k ≤ d}"; "KV^{(≤d)} = [KV^{(1)}; …; KV^{(d)}]".
- **TaH2** [DERIVATION]: "At depth m, token t attends to executed KV states at positions s ≤ t and depths j ≤ m".

**Precise difference from TaH on axis (a)** [DERIVATION, mine]:
- Under CARRY, U_{s+1}⁰ = N(V_s), and LN is scale-invariant. So layer 6's bank K/V for V_s *equal* the TaH-style same-layer K/V of pass s+1.
- VB differs from duo-causal attention only at layers 7–9. There, end-of-core outputs are projected by lower layers' c_attn, which is Feedback's "higher-level memory" idea applied to passes.
- The draft's line "Think-at-Hard: same-layer KV over depths" undersells TaH. TaH already gives every layer all positions and all depths.

## 2. Other close prior art

| paper | (a) / (b) | (c) | (d) | (f) | (g) / (h) |
|---|---|---|---|---|---|
| **Dreamer** 2601.21582 | depth attention (DA) over all previous depths, **same token** | carry (RMSNorm on residual) | 1 layer per depth | LM | scratch ~1–2B, 100B tokens |
| **MeSH** 2510.07739 | B slots, each L×D; per-token softmax read and write | read from buffer | core block (Pythia, e.g. 3+6R3+3) | LM | scratch |
| **AttnRes** 2603.15031 | all earlier **layer** outputs, per token | h_l = Σ α_{i→l} v_i | n/a (depth, not loop) | LM | scratch, 48B / 3B-active, 1.4T tokens |
| **Altabaa** 2510.14095 | carry; discrete re-embed | E^(t+1) = Block(E^(t)) | 2–4 layers | per-iteration linear value readout, depth-masked | scratch; DAG mod-23, N ≤ 32 → 128 |
| **McLeish** 2511.07384 | carry + concat-adapter injection | s_i = R(e, s_{i−1}) | 4 layers | LM, curriculum | TinyLlama / OLMo / Llama retrofit |
| **Shapiro** 2608.11233 | carry via bridge; "no attention connects loop t to loop t−1" | bridged state | 12-layer block | exact per-loop intermediate CE, then outcome-only | Qwen2.5-0.5B retrofit; pointer chasing |
| **Huginn** 2502.05171 | carry; KV entry i mod k at decode | s_i = R(e, s_{i−1}) | 4 layers, (2,4,2) | LM, random r | scratch 3.5B |
| **Ouro** 2510.25741 | carry; KV sharing only at decode | full stack | full | Σ_t p_φ(t∣x) L^(t) − βH | scratch |

**Key definitions:**
- **Dreamer** [DERIVATION]: "depth attention (DA) applies a separate instance of attention over the previous depths of the same token".
- **MeSH** [DERIVATION]: "h^(t+1) = Σ_b m_b^(t+1) ⊙ w_read,b^(t)", with routers that "have unique parameters for each iteration t". That is an explicit pass clock, which VB forbids.
- **AttnRes** [DERIVATION]: "q_l = w_l, k_i = v_i = h_1 (i = 0) / f_i(h_i)", "φ(q,k) = exp(q^⊤ RMSNorm(k))". So values are raw outputs with no projection, as in PURE-ANNEAL.
- **Frey 2603.08391:** its "memory banks are static learnable parameters". This is not activation memory, so it is not close.

**Not found:** a pretrained-LM retrofit whose recurrent core reads a bank of its own *outputs* at all positions in every core layer, or whose pass input is a *cross-position* attention gather over stored iterations.

## 3. READY-RECODE prior art (definitions)

- **Fan 2409.15647** [DERIVATION]: "we supervise the output of the model to match the target data only after a certain number of steps"; the loss is at "f_{T_i}(M_θ, ·)", i.e. T_i iterations.
- **Yang 2311.12424** [DERIVATION]: "the truncated loss window starts at [1,T] and then shifts over time to [b−T, b]".
- **Altabaa 2510.14095** [DERIVATION]:
  - "AlgorithmAlignmentLoss = Σ_t Σ_i 1{Depth(x_i) ≤ t} · ℓ(W_value E_i^(t), Value(x_i))".
  - There is one linear read-out per factor (syntax / variable / operation / value).
  - "This discretization enables us to train the model with a type of teacher-forcing across recurrent iterations".
  - Self-correction: "randomly corrupting the model's latent state".
- **Shapiro 2608.11233** [DERIVATION]: "for loop t, the decoded state receives cross-entropy against the registered intermediate target".
  - [EMPIRICS] Shapiro's abstract: "computes one task step per loop, persists when only final answers are graded".
- **CLRS 2205.15659** [DERIVATION]: "at each step we feed back ground-truth hints with probability 0.5".
- **ReFT 2404.03592** [DERIVATION]: "DII(b, s, R) = b + R^⊤(Rs − Rb)", where R has orthonormal rows. MCT's Δ_s is this edit, with the gold readout as the source and a pseudo-inverse for a non-orthonormal R.

**What READY-RECODE adds:**
- one *serial* reduction per pass, rendered over the original slots (Altabaa reduces one graph layer per iteration, in parallel);
- copy-aware reweighting of the loss;
- MCT's correction kept in the continuous space and limited to the readout's row space, so the student's null-space content survives;
- all of this inside a pass-global attention memory in a pretrained LM.

## 4. Reported failure modes that bear on the design

1. **Age weighting is load-bearing** (InfiLoop Table 2, fixed β, H3L6) [EMPIRICS]: reverse decay 2.0, uniform 38.4, forward 74.0 (learned from β₀ = 0.55: 82.1).
   - VB's gather term γ_q·(t−1−s) has no stated initialisation.
   - **Initialise it to a forward (recency) decay, with the sign stated.** Otherwise the c → 0 endpoint is the uniform-weight case.
2. **First-iteration heads** [EMPIRICS]: TaH (A.3.5) has "a head that mainly attends to first-iteration keys"; TaH2 (C.5) has heads preferring "first-iteration states, later-iteration states, or both".
   - In VB this is the V_0 route, which Addendum 1's RECOMPUTE-V0 label measures.
   - TaH: "Causal@current" (no cross-depth reads) scores −8.5; "Causal@iter1" scores −5.4.
3. **Pass-1 does everything** [EMPIRICS]: MeSH (Pythia-410M prelude-recurrent-coda) reports "The first core loop … accounts for the vast majority of the work". Its fix uses per-step routers, a pass index that VB excludes.
4. **Step-matched training learns a clock** [EMPIRICS]: 2609.33144 studied MR-Loop (Fan's scheme, "K = T = n") and found "a transferable progress cue", with an attention peak that "shifts forward by one relation per loop".
   - R1 sets τ_i = n_i, and the gather queries are wpe-only, so a position-advancing clock is a live route.
   - This supports keeping STALL decisive.
5. **Pretrained loops drift** [EMPIRICS]: in RecurTrace, plain looping raises NLL by +0.07 / +0.05 / +0.06 at 1.7 / 4 / 8B, and memory reverses this. "Gates open at initialization" is worse (+0.02, 53.1 vs 54.7).
   - Removing QK-normalization is the largest single ablation: +0.05 NLL, 52.8.
   - **VB's bank K/V reuse GPT-2's c_attn without QK-norm.** Consider it, or report bank-logit scale.
   - W = 3 was best of {1, 2, 3, 5}, but the gaps are ≤ 1.2 pp at one configuration. That is weak evidence on VB's unwindowed bank; the DILUTION label covers it.
6. **Readout blind spot** [EMPIRICS]: Sharma & Vu 2606.24898 found that "dense per-loop cross-entropy controls the variables exposed by the readout". RMSNorm / LayerNorm readouts "hide radial scale".
   - R = LN + Linear, so R2 constrains about 20 directions after LN.
   - The per-slot RMS reset N(·) is their "scale-removing recurrence" fix. Confirm it applies to V_t on the coda path too.
7. **Teacher forcing and shift** [EMPIRICS]: in 2209.11142, "feeding ground-truth hints at training time hurts generalisation". MCT's Δ = 0 fixed point is the answer to this, but keep TF-ONLY decisive for N3.
8. **Continuous per-step supervision is not enough out of distribution** [EMPIRICS]: Altabaa's continuous variant "slowly degrades"; the discrete anchor plus correction "required a deeper recurrent block" (4 layers, not 2). VB's core has 4 blocks.
9. **Tied attention projections across depths** [EMPIRICS]: Dreamer found that forcing reuse "leads to worse performance than without such reuse". This was from scratch; it is a mild caution for VB's tied bank projections.
10. **Ouro** [EMPIRICS]: reusing KV across steps at prefill costs ">10 points on GSM8K". Per-pass versions are not interchangeable, which is consistent with keeping all of them.

## 5. Suggested draft edits (before registration)

1. **§3:** restate TaH (all positions, all depths, every layer, pretrained) and give the layer-6 equivalence. Cite Feedback for top-memory, and AttnRes / InfiLoop / MeSH for the gather.
2. **§3:** cite InfiLoop for the convex-hull norm bound.
3. **§3:** state the γ_q initialisation (forward decay).
4. **§3 coda:** say whether the coda's bank includes V_0. If it does, drop "never sees problem tokens" and add RECOMPUTE-V0's coda analogue.
5. **§4:** cite Fan / Yang / Altabaa for R1, Altabaa / Shapiro / CLRS for R2, and CLRS / Altabaa / ReFT-DII for R3.
6. **§7:** cite path independence (Anil 2211.09961) beside STALL, and 2609.33144 as the empirical motive for it.
