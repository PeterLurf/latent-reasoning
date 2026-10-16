# C. FORCED RE-CODING in a pass-global attention memory: LEDGER + RE-POSE

*Theory research subagent (angle C), 2026-10-15. The design is re-aimed for the user's constraint (relayed by Audit): the memory must be **pass-global attention memory, not RNN memory**. Repo read-only; the numbers quoted come from THEORY.md §170–§210 (via a digest with line citations) and boxes/nested170.py, boxes/deltamem.py and boxes/deltas5.py. arXiv IDs were checked on arxiv.org or Semantic Scholar.*

## 0. The design in five lines, and what changed

- **LEDGER (the architecture).** An append-only, content-addressed bank of slot vectors.
  - Every pass appends one *group* of w slots, written from its top layer.
  - Every later pass, and a question-blind decoder, reads the bank by softmax attention at block 6.
  - Each reader is limited to k groups, chosen by content plus a relative pass-age bias.
  - Nothing is ever erased.
- **RE-POSE (the training objective).** The group written by pass t must equal, in the model's *own* slot code, the group that a fresh pass writes when it is *given the re-posed problem e_{t+1} as text*.
  - The contrast is against re-posings that differ **only in a computed value**. This rewards each pass for writing new, task-decodable content, and for re-coding the still-live older values.
  - During training, passes are first teacher-forced on the teacher codes, then annealed to free-running.
  - With probability 1/2, past groups are masked except the newest (training-only).
- **The decisive reading** follows from the design: an **EDIT test**.
  - Replace pass t's group with Enc(ẽ_{t+1}), the code of a re-posed problem with one value changed.
  - Then check whether the answer follows the edit. The chance floor is 1/11, so the reading needs no comparator.
  - The same test is the C7 round-trip / edit test, applied to C1.

**What changed.** My first draft used lifetime-limited / destructive memory (exact per-pass clearing). The user's constraint rules that out, and the draft also had problems of its own (§0.5). What survives is the core idea: re-coding is *forced* only where it is *necessary*. Here it is necessary because of:
- the per-pass compute bound;
- read budgets that bound fan-in;
- a training-only Markov mask;
- value-specific targets.

## 0.5 Critique of the angle's starting ideas

| Idea | Verdict | Why |
|---|---|---|
| Lifetime-L / clear-each-pass / destructive read | **Dropped** | It violates the pass-global constraint. [DERIVED] It also removes the direct decoder→early-pass gradient paths. With W_o zero-initialised, the chain Jacobian is exactly 0 at init (cf. §199: loop gain 0 for γ = 0). **Soft decay** does not bound lifetime when norms are free: a write of norm A survives ρ^k·A, and an RMS read normalises away the decay. **Partial consumption** (only the read slots are consumed) leaves unread slots as registers (G5). |
| Per-pass rotation of writes | **Rejected** | If the architecture applies R_s⁻¹, it is a no-op. If R_s is fixed per pass and not inverted, it is a free absolute pass clock (a pass-indexed dialect, G6), and old values need learned per-age inverses: a soft lifetime in disguise. A hidden per-example key makes old values unreadable, which is the same as erasure. Rotations also preserve norms, so a norm side-channel survives. **Replacement:** a *relative* pass-age attention bias (ALiBi over passes) gives the recency structure without any absolute pass identity. |
| Softmax competition on writes (slot attention) | **Not needed** | Capacity is not the binding constraint (§4.3). §208 read "not capacity-bound" (R_QO 0.42 = R_A). Competition adds a trainability risk and closes nothing. |
| Per-pass bandwidth limit | **Reformulated** | It is ill-posed as stated. The answer (log₂11 = 3.46 bits) always fits, and on nested the partial state starts *larger* than the answer. The only budget that forces re-coding is **read fan-in** (Lemma 2), plus the training-only Markov mask (Lemma 3). |

## 1. Mechanism

### 1.1 Skeleton

LEDGER is built on §198(c)'s `ReadSlotsN` (nested170.py L461) and `rs_run` (L427):
- The question [EOT, x] is encoded **once** per batch. Its per-layer KV is cached, and there is no memory read while encoding.
- A think pass is w = 8 write tokens that attend to the cached question KV, and causally to each other.
- Measured cost: READ-SLOTS ran at 0.853× cached INJECT's wall-clock (note 198-2).

**New code** (proposed `boxes/ledger.py` plus an `--arm ledger` in nested170):
- the memory read, which is attention over a growing bank, replacing DeltaMem's `read_fixed`;
- the write, which appends the group instead of `write_seq` into S;
- the blind decoder;
- the RE-POSE losses.

### 1.2 State

The ledger after pass t is 𝓛_t = {(m_{s,j}, s) : 1 ≤ s ≤ t, 1 ≤ j ≤ w}, with m ∈ ℝ^768. Plus one learned **null slot** (k_∅ learned, v_∅ = 0), so a reader can read nothing.

### 1.3 Pass t (t = 1..K)

- **Input.** u_j = e_j + wpe[P_j], j = 1..w.
  - e_j is learned, initialised to wte(" .") + 0.02·ε_j, as in ReadSlotsN.
  - P_j = 96 + j − 1 are **fixed positions, identical in every pass**. Assert L(x) < 96 for n ≤ 16.
  - So no positional pass clock exists.
- **Blocks 0..5.** Standard GPT-2 blocks attending to C(x) (the cached question KV) and causally to the write tokens.
- **Read at block 6 entry:**

  h ← h + f · W_o · Read(h; 𝓛_{t−1})

  - f is the fade buffer: 0 → 1 over 500 steps.
  - W_o is zero-initialised.
  - So at step 0 the pretrained function is unchanged, as with DeltaMem.
- **Blocks 6..11**, then the **write.** Pass t appends the group G_t = (m_{t,1..w}), with tag t:

  m_{t,j} = LN_f(h^{(12)}_j)

  This is GPT-2's ln_f output, the vector the LM head reads.

### 1.4 Read(h; 𝓛)

The read uses 12 heads, d = 64, with LN_r and W_q, W_k, W_v initialised from block 6's ln_1 and c_attn (DeltaMem's convention).

- **Projections.** For each head η:
  - q_i^η = W_q^η LN_r(h_i);
  - k_{s,j}^η = W_k^η LN_r(m_{s,j});
  - v_{s,j}^η = W_v^η LN_r(m_{s,j}).
- **Group selection (the read budget).** The age of group s at pass t is a_s = t − 1 − s (the newest has a = 0), and q̄^η is the mean query over the reader's tokens. Score each group:

  σ_s = (1/12) Σ_η max_j (q̄^η · k^η_{s,j})/√d − λ̄·a_s

  - The selected set is G = top-k_s σ_s.
  - k = 2 for passes and k_dec = 1 for the decoder.
  - For steps < 4k, k = ∞ (dense warm-up); then k = 2.
- **Attention within G:**

  α^η_{i,(s,j)} = softmax over {(s, j) : s ∈ G} ∪ {∅} of [ q_i^η · k^η_{s,j}/√d − λ_η·a_s ]

  Read^η_i = Σ α v

  The age slopes λ_η ≥ 0 are learned, initialised to ALiBi's geometric slopes (Press et al. 2108.12409) on 6 heads and to 0 on the other 6.
- **What a pass sees:**
  - the question (all blocks);
  - its own write tokens;
  - at block 6, ≤ 2 ledger groups plus the null slot, chosen by content and relative age.
- **What a pass never sees:**
  - a pass index;
  - other passes' tokens at other layers;
  - the ledger's size (the top-k makes the read set size-invariant).

### 1.5 Decoder

The decoder is **question-blind**: [EOT, " ="] at positions 0, 1.
- It reads the ledger at block 6 with k_dec = 1 group.
- The LM head at " =" gives a full-vocabulary CE on the value token (nested170's convention).
- This follows §208's answer bottleneck. It closes G3 by construction.

### 1.6 The teacher encoder

Enc(e) is the same pass machinery, run with:
- question = [EOT, e], the re-posed text;
- an empty ledger (null-only read);
- the same write tokens at the same P_j.

It returns a group of w vectors. It is run under no_grad (sg) with the current weights; an EMA copy is a fallback flag. Because the student and teacher groups are produced by **identical machinery at identical positions**, they live in the same space. This is the precondition teacher forcing needs, and it holds by construction.

### 1.7 Window and training-only masks

- **Window.** K = n + U{1..4} (§182 WINDOW), with **no hold target** past n: §182 read HOLD − WINDOW = −0.192.
- **TOM (training-only mask).** In training, with probability p_m = 0.5:
  - a free-running pass t ≥ 2 gets G := {t − 1}, the newest group only;
  - the decoder gets G := {K}.
- At inference the full ledger and the top-k read are used.

### 1.8 How LEDGER differs from CODI / Coconut

CODI (2502.21074) and Coconut (2412.06769) also keep every latent globally addressable. LEDGER differs in four ways:

1. **Reads of top-layer codes at block 6.** A same-layer KV read gives a later pass only the *last* layer's worth of compute on an earlier pass's final result, unless there is a hidden→input feedback path. This is the Feedback Transformer point (2002.09402). LEDGER's top-down read gives 6 blocks per hop with no RNN path.
2. **Budgeted group selection.**
3. **Targets in the model's own problem code**, which build in the C7 substitution test.
4. **No absolute pass identity anywhere.**

## 2. Training objective

For an item x with CoT states e_1 = x, e_2, …, e_{n+1} = " v" (Feng's leftmost reduction; nested170 `aux_steps`):

### 2.1 Teacher forcing (TF)

- Per item, with probability p_TF(step): 1 until 4k steps, then linear to 0 at 24k.
- Passes t ≤ n then read the teacher ledger 𝓛^T_{t−1}, whose group s is sg Enc(e_{s+1}).
- Window passes t > n and the decoder read 𝓛^T_n, followed by the student's own groups from n + 1.
- In TF mode the t ≤ n passes are independent given 𝓛^T. They run as **one batched forward** of B·n rows × w tokens, like text-CoT teacher forcing.

### 2.2 RE-POSE, for each pass t ≤ n (TF or free-running)

- **Similarity:** sim(G, G′) = (1/w) Σ_j cos(m_j, m′_j)/τ, with τ = 0.1.
- **Candidates:** C_t = {e_{t+1}} ∪ {ẽ¹, ẽ², ẽ³}. Each ẽ is e_{t+1} with **one live computed value** replaced by a wrong value:
  - ẽ¹ and ẽ² alter the newest value v_t;
  - ẽ³ alters an older value that is still present in e_{t+1}, if one exists, otherwise v_t again;
  - each must be valid (no divisor 0).
- **Contrastive term:**

  ℓ^NCE_t = −log [ exp sim(G_t, Enc(e_{t+1})) / Σ_{c∈C_t} exp sim(G_t, Enc(c)) ]

- **Cosine term:** ℓ^cos_t = 1 − (1/w) Σ_j cos(m_{t,j}, Enc(e_{t+1})_j).
- **Combined:**

  L_RP = mean over items, then over t ≤ n, of AUX_W[t]·(β·ℓ^cos_t) + AUX_W[t*]·ℓ^NCE_{t*}

  - t* is one sampled pass per item, which saves 3 encoder passes on every other pass.
  - β = 0.5.
  - AUX_W is §184's reweight (nested170 L370).

**Answer loss:** L_ans = CE(decoder | free-running 𝓛_K) + CE(decoder | TF 𝓛_K).

**Total:** L = L_ans + λ_R·L_RP. λ_R = 1, constant to the end; there is no fade in the first experiment, so the decisive reading stays clean.

### 2.3 Gradient paths

| Term | Path | Stops at |
|---|---|---|
| CE(free-running) | decoder → its top-1 group → the pass that wrote it → that pass's read → earlier groups (serial), **and** direct to any selected group | the zero-init W_o at step 0 (the fade) |
| CE(TF) | decoder pass only, plus trunk / read branch | the sg teacher groups |
| ℓ^cos, ℓ^NCE (TF passes) | **one hop**: pass t's write tokens, trunk, read branch | the sg teacher ledger, the sg Enc targets |
| ℓ^cos, ℓ^NCE (free-running passes) | pass t, then back through the student groups it read | the sg Enc targets |
| TOM | no new term; it changes only the read set | — |

**Gate (required before launch).** The gradient-flow check gives nonzero gradient per term on:
- W_o, W_q, W_k, W_v;
- e_j;
- λ_η;
- the trunk.

The W_q, W_k, W_v gradient is 0 at step 0 *by design* (W_o = 0). The check runs at fade = 1 on a perturbed W_o.

## 3. Goodhart table

| Route | Status under LEDGER + RE-POSE | Mechanism |
|---|---|---|
| **G1** pass-1 precompute | **Bounded / not rewarded** | Pass 1 sees x once; it determines ≤ c₀ ≈ 2 reductions (DIRECT 0.64–0.75 at n 3). Its target is Enc(e₂), so extra precompute earns nothing. Deeper values cannot be precomputed (Lemma 1). |
| **G2** recompute from input every pass | **Closed beyond c₀** | A pass ignoring the ledger writes groups determining ≤ c₀ reductions. It fails ℓ^NCE for t > c₀ (Prop. 4) and fails the answer for n > c₀. Measured by EDIT-FOLLOW. |
| **G3** decoder-side solving | **Closed by construction** | The decoder is question-blind and reads one group, so it can only decode a consolidated code. |
| **G4** copying literals | **Closed** | All candidates share every literal. A literal-copying student is value-blind, so E[ℓ^NCE] ≥ log 4 (Prop. 4). ℓ^cos alone *is* G4-gameable, which is why it is paired with ℓ^NCE. |
| **G5** static answer register + idle repeaters | **Permitted where harmless** | The ledger keeps a register addressable, as the user requires. A register needs a chain to *compute* it (Lemma 1). Idle passes before n fail ℓ^NCE (the target changes every pass). Idling after n is the intended hold, with no hold target. |
| **G6** pass-index readout | **Closed** | There is no pass token, the positions are fixed, the age is relative and the read set is size-invariant (top-k). The target depends on content (e_{t+1}), not on t. The honest pass needs no t: it reduces the newest code once. |
| **G7** one-sided dynamics target | **Closed** | There is no displacement / novelty / Lyapunov floor (§199 CSR's failure). The "information gain" is a two-sided CE on the identity of a *task value*. |
| **G8** early-pass sabotage | **Closed** | The targets are sg codes of fixed texts. Sabotage cannot make the positive easier to pick out among the value-counterfactuals, and it raises L_ans. |
| **G9 (new) TARGET-BLIND encoder** | **Gated** | If the Enc codes ignore values, ℓ^NCE carries no signal. Step-0 gate TARGET-SEP (§7); a mid-run monitor tracks the candidate cosine gap. |
| **G10 (new) identity-matching in InfoNCE** | **Closed** | The negatives are the same item with only the value changed; in-batch negatives are never used. |
| **G11 (new) value-prior artefact** | **Measured** | A counterfactual value may be atypical under the generator. Report the information gain over a context-only prior. |
| **G12 (new) TF-ONLY / exposure bias** | **Mitigated, measured** | p_TF → 0 by 24k; ℓ^cos makes the student codes substitutable. Labels TF-ONLY and CODE-DRIFT. |
| **G13 (new) TOM evasion** | **Measured** | A model could rely on old groups when they are visible. The masked half forces newest-group sufficiency (Lemma 3); the mask-old eval checks it. |
| **G14 (new) norm channel** | **Closed** | m = LN_f(·) has bounded norm, and the read re-normalises with LN_r. This is irrelevant to persistence anyway (nothing is erased). |
| **G15 (new) dilution / overthinking** | **Mitigated** | Junk groups written after n could capture the decoder's read. The age bias plus content top-1, trained under WINDOW; label DILUTION. |
| Rotation-inverse / aliasing routes | **N/A / benign** | No rotation is used. Superposition within a group is legitimate; cross-group reliance is penalised by TOM. |

## 4. Mathematics

**Notation.** W(x) is the item's work, measured in reductions (n); the height-based variant is in A1′. ρ(G) is the number of leading reductions *determined* by group G: the largest r such that e_{r+1} is a function of G (and of x, for readers that see x).

### 4.1 Chain lemma under the budgets

- **A1 (per-pass compute bound).** [ASSUMED; EMPIRICAL support]
  - A pass that sees x and reads groups R writes a group with ρ ≤ max(c₀, max_{G∈R} ρ(G) + c).
  - The blind decoder reading a group G is reliable only if ρ(G) + c_dec ≥ n.
  - Calibration:
    - c₀ ≈ 2: DIRECT reads 0.98–0.99 at n 2 and 0.64–0.75 at n 3 (note 170-3).
    - c ≈ 1: one reduction per pass in converged models (§175 / §184 / P0b).
    - c_dec ≈ 0–1.
  - A1′ (variant): the same bound with tree height in place of n, for fan-in-combined reads.
- **Lemma 1 (chain).** [PROVED from A1 by induction on the read DAG]
  - Let L(t) be the length of the longest read chain ending at pass t. A chain is a sequence of passes in which each reads the previous one's group.
  - Then ρ(G_t) ≤ c₀ + (L(t) − 1)·c.
  - A correct answer on a work-n item therefore needs the decoder's group to end a read chain of length

    **f(n) ≥ 1 + ⌈(n − c₀ − c_dec)/c⌉**

    Each link is a re-coding step: a group written from a read group.
  - With the calibration, f(n) ≈ n − 1 to n − 2.
  - Idle passes may sit anywhere (the memory is global), but at most K − f(n) of them.
  - This holds for any pass-global memory. The read budget and TOM make the chain *identifiable* and *Markov* (below).
- **Lemma 2 (fan-in, the read budget).** [PROVED, counting]
  - Suppose every pass reads ≤ k groups and the decoder reads ≤ k_dec.
  - Then the number of passes that can influence the answer through a read DAG of depth ≤ D is ≤ k_dec·(k^D − 1)/(k − 1).
  - So an answer that aggregates values written by N distinct passes needs D ≥ log_k(N(k − 1)/k_dec + 1) consolidation steps.
  - This bites on breadth. A GSM8K-style "compute many shallow values in parallel passes, let the reader combine them" pattern needs log₂(N + 1) re-codings at k = 2 and k_dec = 1.
  - **Combined: f(n, N) = max(Lemma 1, Lemma 2).**
- **Lemma 3 (TOM gives Markov sufficiency).** [DERIVED]
  - Suppose a masked pass (probability p_m) reads only G_{t−1}, and the masked-pass loss is the same objective.
  - Then any parameter setting with excess risk ε has

    I(task state at t ; everything not in G_{t−1}, x | G_{t−1}, x) ≤ ε/p_m

    measured in loss units, via the CE ≥ conditional-entropy bound.
  - So the minimiser makes the newest group a sufficient statistic of progress, and every live value is re-coded into it each pass.
  - Unlike §203 SVR (decodable value features that the decoder *did not use*: G-use 0/7 valid models), the value information here lies on the **only path available to masked passes**. Use is structurally required on half the passes, and the EDIT test verifies it.
- **Prop. 4 (RE-POSE is a value-information bound).** [PROVED]
  - Let the candidates differ only in a value v*, and let the student's group be independent of v* given its value-free content.
  - Then the true candidate's index is exchangeable, E[p_true] = 1/(N + 1), and by Jensen, E[ℓ^NCE] ≥ log(N + 1).
  - More generally, E[ℓ^NCE] ≥ H(v* | G_t) restricted to the candidate set. So I(G_t; v*) ≥ log(N + 1) − E[ℓ^NCE] (in nats, for uniform candidates).
  - CPU check (scratchpad `research/ledger_sanity.py`): a value-blind student with arbitrary similarities averages 3.40 at N = 3 and 5.25 at N = 10, against the bounds log 4 = 1.39 and log 11 = 2.40.

### 4.2 Stability: bounded, no blow-up, no dilution; relative metrics

- **Norm.** [PROVED] ‖m_{s,j}‖ = ‖γ_f ⊙ ẑ + β_f‖ ≤ ‖γ_f‖_∞·√768 + ‖β_f‖, where ‖ẑ‖ = √D.
  - The read is a convex combination of values from LN_r-normalised slots, so ‖Read‖ ≤ max_{s,j} ‖W_v LN_r(m_{s,j})‖.
  - This bound is **independent of t and of the ledger size**.
  - Contrast the additive DeltaMem state: S_t = Σ_passes Σ_tokens φ(k)vᵀ grows with t, which is what RMS normalisation patched (§185).
- **No decay to zero.** The null slot is the only "read nothing". The top-k bound limits averaging-to-cancel to ≤ 2w slots.
- **No dilution beyond the trained K** (the C2 / C6 precondition). [PROVED]
  - With content logits bounded |s| ≤ B and age bias −λa, the newest group's mass is ≥ e^{−2B}(1 − e^{−λ}), for every t.
  - Without the age bias it falls like 1/t.
  - CPU check (w = 8, B = 2):

    | λ | newest-group mass at t = 4 / 8 / 16 / 32 |
    |---|---|
    | 0 | 0.47 / 0.28 / 0.15 / 0.08 |
    | 0.5 | 0.67 / 0.63 / 0.62 / 0.62 |

  - **Prediction:** a model whose learned λ̄ ≈ 0 shows DILUTION past K.
- **Relative metrics.** Slot norms are near-constant, so all perturbation probes are **angular**:
  - rotate slot j of group s by an angle θ in a random orthogonal direction;
  - measure the transmission T_{s→t} = ∠(m′_t, m_t)/θ, and the decoder's relative margin change.
  - These are scale-free by construction: no absolute-gap growth artefact (the 192-2 lesson).
  - Two-sided reading (*descriptive, never a target*):

    | Region | Reading |
    |---|---|
    | 0.3 ≤ T_{t→t+1} ≤ 3 for t < n | relay |
    | T ≪ 0.3 | contraction trap (the read is ignored) |
    | T ≫ 3 | expanding motion (cf. §199's FLOOR-GAMED) |
    | T_{t→t+1} < 0.5 for t > n | stable hold (C6) |

  - Sanity-check the probe on a converged §193 model's DeltaMem read, using the relative read gap, before reading LEDGER.

### 4.3 Capacity

- **Need.** [DERIVED] With the question visible to every pass, the Markov-sufficient progress state is the set of live computed values plus their positions.
  - In leftmost reduction this is ≤ the tree height h, which is ≤ ~7 for n ≤ 16.
  - That gives ≤ h·(log₂11 + log₂64) + 4 ≈ 70 bits.
  - The RE-POSE target, however, codes the *whole* re-posed text: ≤ ~60 symbols × log₂19 ≈ 255 bits at n = 16.
- **Supply.** [ASSUMED] w = 8 slots × 768 dims, with ≥ 32 reliably learnable bits per slot, gives ≥ 256 bits. This is borderline at n = 16, t = 1, and ample for n ≤ 8.
- **Gate.** TARGET-SEP must also report a linear probe of e_{t+1}'s symbols from the Enc codes at n = 16. If it is < 0.9 per symbol, use w = 16, at about +15% cost.
- **Ledger read cost.** ≤ (K = 20) × 8 = 160 slots, which is negligible.

## 5. Pretrained init and trainability

- **Function preservation.** With W_o = 0 and fade = 0, every pass and the decoder compute exactly ReadSlotsN's or the plain trunk's outputs. Gate: max |Δ logits| = 0 at step 0.
- **Does a global ledger worsen the seed lottery? No, the opposite.** [DERIVED]
  - The ledger gives **direct** decoder→any-pass paths, so no term needs a K-fold Jacobian product.
  - The lifetime-limited design I dropped *would* have worsened it: its zero-init chain has no gradient at init.
- **Why LEDGER should beat the lottery.**
  - §199 models basin selection as a pitchfork, P(enter) = Φ(μ_Z/σ_Z).
  - §203-7 found GPU/driver numerics carry about 70% or more of the outcome. This means μ_Z ≈ 0: at init a free-running pass reads its own earlier groups, which hold the *same x-features it already has*. There is no deterministic drive to read, so numerics decide.
  - **Teacher forcing supplies a non-redundant ledger from step 0.** Enc(e_t) contains values that the reader cannot compute from x for t > c₀. With the ledger, each pass's task becomes "read a code, do one reduction (an n = 1 problem, which DIRECT solves at 0.998), write a code".
  - Hence μ_Z = O(1) from the end of the fade. [DERIVED direction; ASSUMED magnitude]
- **Gradient through K passes.**
  - TF RE-POSE terms are one-hop.
  - The free-running CE has direct paths plus serial paths.
  - Only TOM-masked free-running passes depend on serial paths, and those are trained after TF has built the step map.
- **Mitigations:**
  - the fade-in;
  - a dense → top-2 anneal at 4k;
  - p_TF annealed over 4k → 24k;
  - AUX_W per-pass reweighting;
  - the base recipe's gradient clip;
  - an EMA teacher flag, if the teacher's cosine gap drifts.
- **Pretrained mismatch (risk).** The read branch feeds top-layer ln_f vectors to block 6 through a new attention branch, which is as new as DeltaMem's read. It is initialised from block 6's own c_attn.

## 6. Predictions, decisive reading, labels, OC block

### Predictions (nested, per seed, 48k steps)

All numbers are [ASSUMED] priors stated before any run.

| # | Prediction | Value | Comparison |
|---|---|---|---|
| P1 | ENTRY: free-running answer-CE 1k-bin < 0.5 | by 8k; final < 0.1 | lottery base rate about 10/19 |
| P2 | Competence: acc at n 3–8, K′ = n + 2 | ≥ 0.95 | — |
| P2 | K′0 (pooled n ≥ 4) | ≈ 0.09 | — |
| P2 | K′1 (pooled n ≥ 4) | ≤ 0.15 | — |
| P3 | **EDIT-FOLLOW** (n 4–8, t = 2..n−1) | ≥ 0.80 | chance 0.09; a recomputing model about 0.1 |
| P3 | ROUND-TRIP (substitute Enc(e_{t+1})) accuracy drop | ≤ 0.03 | — |
| P4 | Markov: mask-old eval accuracy | ≥ full − 0.02 | — |
| P4 | Newest group selected (passes t ≤ n) | ≥ 0.9 of the time | — |
| P5 | DEPTH (n 9–12 at K′ = n + 2) | ≥ 0.75 | §193: 0.647 / 0.688 |
| P5 | n 13–16 | ≥ 0.35 | §193: about 0.13–0.16 |
| P6 | STABLE: acc at n 3–8, K′ 32 | ≥ K′(n + 2) − 0.03 | §193 s2: 0.862 vs 0.978 |
| P7 | Angular T_{t→t+1} for t < n (descriptive) | median in [0.3, 3] | — |
| P7 | Angular T_{t→t+1} for t > n (descriptive) | < 0.5 | — |

P5's mechanism is self-similarity. Once ≤ 8 operators remain, the newest group is Enc of an in-distribution expression, so the errors concentrate in the first n − 8 passes.

### Decisive reading (comparator-free): RELAY-MET

A seed reads RELAY-MET when all four hold:
- (a) acc at n 3–8 ≥ 0.90;
- (b) K′0 and K′1 ≤ 0.20 at n 4–8;
- (c) EDIT-FOLLOW ≥ 0.70 over about 1,000 edits;
- (d) the ROUND-TRIP drop ≤ 0.05.

Every bar is against a chance floor or the run's own unedited accuracy. No other arm enters the reading.

Notes:
- The skip test is vacuous here, as in note 170-4: the pass inputs are identical, so a skip equals K − 1 passes. The step-specific C1 test is therefore EDIT-at-t.
- **Censoring.** Edits count as non-follow when the output matches neither value. Edits are scored on *all* items, and the conditioned-on-correct rate is reported beside. A NO-ENTRY seed fails every reading.

### Failure labels

| Label | Condition |
|---|---|
| NO-ENTRY | the bin never < 0.5 by 24k (never-reached = failure) |
| TF-ONLY | TF CE < 0.1 but free-running acc at n 3–8 < 0.6 |
| CODE-DRIFT | ROUND-TRIP drop > 0.10 |
| RECOMPUTE | EDIT-FOLLOW ≤ 0.2 with (a) met |
| LOSES-ONE-PASS | acc(n, K′ = n) < acc(n, K′ = n + 1) − 0.2, pooled n 4–8 (pass 1 spent on encoding) |
| DILUTION | acc falls past K while the newest-group mass decays and λ̄ < 0.1 |
| TARGET-BLIND | the step-0 or mid-run cosine gap is below the noise floor |
| NOT-DEEP | — |

### OC block (per registration rule)

- **P(RELAY-MET | no effect).**
  - Under the recompute null (EDIT-FOLLOW ≈ 0.10–0.15, as in failed seeds' INPUT-FOLLOW 0.90–0.99), P(≥ 0.70 of 1,000) < 10⁻¹⁰⁰.
  - Under a "carries, but not in Enc's code" null, the prior is EDIT-FOLLOW 0.2–0.5, so P ≤ 0.05. [ASSUMED]
  - Design level, 1 of 2 seeds: ≤ 0.10.
- **P(RELAY-MET | effect).**
  - Per seed: P(entry) 0.85 × P(substitutable | entry) 0.75 ≈ 0.64.
  - ≥ 1 of 2 seeds: 0.87.
  - 2 of 2 seeds: 0.41.
- **Realised $:** reported per cell when it lands (projection in §7).

## 7. Minimal first experiment, then GSM8K

### Stage 0: local 8 GB card, short and memory-checked, $0

1. **Implementation.** `boxes/ledger.py` plus `--arm ledger` in nested170.py (ReadSlotsN skeleton). Then py_compile, and **smoke the plain path** (readslots and inject construct, plus 1 step) after the edit.
2. **Gates:**
   - G0a: function preservation, max |Δ logits| = 0.
   - G0b: gradient flow, nonzero gradient per term.
   - G0c: **TARGET-SEP**. On 2,000 re-posed texts at step 0:
     - a linear probe recovers v_t from Enc codes at ≥ 0.9 held-out accuracy;
     - the median 1 − cos(Enc(e), Enc(ẽ)) is ≥ 3× the bf16 re-run noise floor;
     - the n = 16 symbol probe is ≥ 0.9.
   - G0d: memory check at the largest shape (n 16, K′ = 2K).
   - G0e: a 2k-step TF smoke, where ℓ^NCE must fall below log 4 − 0.5, with no non-finite values.
3. Delete the scratch model.pt files after the smokes.

### Stage 1: cloud RTX 3090, 24 GB, checkpointed

- **LEDGER-full seed 1:** 48k steps, n ≤ 8, WINDOW 1..4, bs 64 (§193 schedule otherwise).
- **Seed 2** launches only if s1 enters by 12k **and** the realised rate projects the pair ≤ $10. If s1 is NO-ENTRY by 24k, stop: no s2.
- **Cost projection:**
  - ReadSlots runs at about 0.853 × cached INJECT's 0.861 s/step, i.e. about 0.73 s/step.
  - × 1.3 for the window passes, × 1.4 for the encoder and negative passes (TF batching offsets part of this).
  - That gives about 1.3 s/step, so 48k steps ≈ 17–18 h ≈ **$3.2–5.0 per seed**.
  - **Pair cap: $10.**
  - One GPU class for both seeds (the basin-dependence lesson); record the GPU per cell.
- **Eval:**
  - acc_by_n at K′ ∈ {0, 1, n, n + 2, 12, 16, 24, 32};
  - EDIT / ROUND-TRIP at t = 2..n−1 (n 4–8), re-run at the original --bs;
  - the mask-old eval;
  - selection logs (the read graph);
  - angular transmission;
  - the INPUT-SWAP and donor STATE-FOLLOW from §203, for continuity.

### Stage 2: GSM8K-Aug follow-up (only if RELAY-MET in ≥ 1 seed)

- **Setup:**
  - Port the ledger arm to gsm8k_stepaux (a ReadSlots-style question cache).
  - w = 8, K = m_eq + U{1..2} (cap 10).
  - Blind decoder [EOT, "###"].
- **RE-POSE target:** Enc(question ⊕ "<<e₁>>…<<e_t>>"), the model's code of the question with t equations written. Negatives are the same text with equation t's result (or one earlier live result) replaced by a wrong number.
- **Schedule:** TF anneals 1 → 0 by 12k, over 24k steps.
- **Cost:** about 1.5–2.0 s/step, ≈ 10–13 h, ≈ **$2.0–3.7** (cap $4).
- **Predictions:**

  | Measure | Prediction | Reference |
  |---|---|---|
  | acc | ≥ 0.20 | latent 0.14–0.16; CODI-4ep 0.189 |
  | ITER-GAIN = acc(K′ = K) − acc(K′ = 1) | ≥ +0.03 | §208: −0.005 |
  | m ≥ 3 stratum | ≥ 0.17 | CODI 0.132 |
  | VALUE-LEAK ratio, t ≥ 2 | ≤ 0.5 | §204: 0.90 |
  | EDIT-FOLLOW on results | ≥ 0.5 | — |

- **Design total:** ≤ $14 = $10 + $4. This is above $10 overall, so **Stage 2 needs the user's OK**. Stage 1 alone stays within $10.

## 8. Top three risks, and what falsifies the design

1. **Code substitutability fails (TF-ONLY / CODE-DRIFT).**
   - The free-running codes satisfy the relative InfoNCE but sit far from the teacher codes, so reads trained on teacher codes fail at free-running time.
   - *Falsifier:* ROUND-TRIP drop > 0.10, or TF-ONLY, in both seeds.
2. **Moving-target or encoder collapse (G9).**
   - Enc shares the trained trunk. Its value-sensitivity could erode mid-run, and ℓ^NCE would then become uninformative.
   - *Falsifier:* the candidate cosine gap falls below the noise floor after the step-0 gate passed. The EMA-teacher fallback exists, but its use would be a new registration.
3. **The compute bound is weaker than assumed on GSM8K.**
   - Shallow steps let pass 1 compute most values, the VALUE-LEAK route. Then Lemma 1's forcing has little bite, and only Lemma 2 (fan-in) acts.
   - *Falsifier:* GSM8K ITER-GAIN ≤ 0.01 with VALUE-LEAK ratio > 0.8 despite RELAY-MET on nested. That would mean the design forces re-coding where depth demands it, but does not create depth on GSM8K.

**What falsifies the design as a whole:**
- **EDIT-FOLLOW ≤ 0.30 with competence (a) met**, in a seed that entered. The model solves the task, but not through its re-posed codes, so the forcing failed.
- **NO-ENTRY in 2/2 seeds**: the teacher-forcing trainability claim is false.
- **DEPTH n 9–12 ≤ 0.65 with RELAY-MET**: the self-similarity C2 argument is false, even though C1 / C7 hold.

## References (IDs verified)

| Work | arXiv ID |
|---|---|
| Feng et al. | 2305.15408 |
| Coconut | 2412.06769 |
| CODI | 2502.21074 |
| Feedback Transformer | 2002.09402 |
| ALiBi | 2108.12409 |
| Recurrent Memory Transformer | 2207.06881 |
| Token Turing Machines | 2211.09119 |
| Memorizing Transformers (top-k memory reads) | 2203.08913 |
| Shared Global Workspace (limited-capacity attention memory) | 2103.01197 |
| Slot Attention | 2006.15055 |
| Markovian Transformers | 2404.18988 |
| Huginn | 2502.05171 |
| Looped transformers (Saunshi) | 2502.17416 |
| Gated DeltaNet | 2412.06464 |
| Fast-weight programmers | 2102.11174 |
| uRNN | 1511.06464 |
| DAS | 2303.02536 |
| NTM | 1410.5401 |
| Titans | 2501.00663 |
