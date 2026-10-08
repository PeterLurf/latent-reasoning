# Prior check for §153 THINK-PASS: item-agnostic latent passes with DeltaMem as the sole cross-pass channel (VISIBLE / PURE) (2026-10-07)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-07) or earlier notes (LIT_INDEX).

**Verdict:** new as scoped. The components are known (memory-augmented segment recurrence, looped models with input injection,
step-aligned latent supervision, MQAR-capable delta memories). Not found: K *latent thinking passes* on a pretrained LM whose only
cross-pass channel is a matrix-valued associative memory, with per-pass step decoding and removal.

## (1) Precedents for K passes with a matrix memory as the only cross-pass channel
| Lead | Memory | Passes | Closest gap |
|---|---|---|---|
| ARMT 2407.04841 | layerwise associative (linear-attention) memory | **input segments** of text, on pretrained **GPT-2** | segments carry new input; RMT memory tokens also cross segments; retrieval tasks |
| Infini-attention 2404.07143 | compressive linear-attention memory (with a delta variant) as the only long-range path | input segments; continual pretraining 1B / 8B | long context, not reasoning passes |
| RMT 2207.06881 | memory *tokens* (vectors) | input segments | not matrix-valued |
| LM2 2502.06049 | a gated memory bank, read by cross-attention | per token, alongside attention | not the sole channel |
| Titans 2501.00663 | a neural memory updated by test-time gradient steps | per token / segment; from scratch | not the sole channel; no latent passes |
| Huginn 2502.05171 / Saunshi 2502.17416 | **the residual state** (looped depth) | latent iterations on the same input | the state is a vector stream, not a matrix memory; input re-injected |
| Pause tokens 2310.02226 | none (attention to pause positions) | extra positions | no recurrent state |

**Not found:** latent passes with *no new input* (PURE), or with the whole problem re-shown (VISIBLE), where a delta / linear memory is the
*only* thing that crosses passes in a pretrained LM.

## (2) Per-pass step supervision decoded from the state, then removed
- **SIM-CoT 2509.20317:** step-aligned decoding of single-vector latents; on throughout training, discarded at inference. No removal
  schedule.
- **Closest with removal: the recurrent-depth retrofit 2608.11233.** Intermediate-step supervision with one task step per loop, then
  **outcome-only annealing, and the mechanism persists.** That's the same arc as A → B-EARLY, but its state is the looped residual
  stream, not a memory.
- ICoT-SI 2405.14838 (token removal); Coconut (stage replacement).
- **Memory-state step decoding plus removal: not found.** New.

## (3) PURE: associative recall of operands inside the loop
- **MQAR evidence:**
  - Zoology 2312.04927 introduced multi-query associative recall and showed gated-convolution / linear models lag attention as the
    number of pairs grows.
  - Based 2402.18668: a recall vs state-size trade-off for linear attention.
  - DeltaNet 2406.06484 / Gated DeltaNet 2412.06464: the delta rule markedly improves MQAR over plain linear attention.
  - **8 operand digits is a small key set, well within the reported regime** for d_k ≥ 64 per head.
- **Failure modes to register against:**
  1. **Crosstalk in the additive heads.** No-decay additive heads accumulate every write. With non-orthogonal keys, interference grows
     with the number of writes (capacity ≈ d_k orthogonal keys).
     - If operands *and* the accumulator share additive heads, recall degrades over K passes.
     - Log per-head roles: operand recall vs accumulator decodability per head.
  2. **Delta-head overwrite.** Gated-delta writes replace values at a key. Later accumulation writes can erase operand entries if their
     keys overlap. Log operand-recall accuracy *at each pass* (decode b_j from memory at pass j with a held-out probe), not only at
     pass 1.
  3. **Step-index confusion.** Pass j must query b_j's key. The keys come from pass-0 positions, and the query from step token j. The
     mapping is learned only for j ≤ 4, so **4×5 (j = 5) asks for an untrained key**. That is the clock-out-of-range concern
     (Prop. 9 / §35). Expect C2 to fail unless:
     - the step token is relative ("next digit"); or
     - the memory carries its own pointer (state-based counting); or
     - j is randomised over a wider range in training (Huginn / Kuo 2606.29983).
  4. **Write saturation / norm growth** in the no-decay heads over K + 1 passes (pass 0 writes operands). Log the memory norm per pass
     (as for §152).

## (4) VISIBLE: "the model selects its own step input"
- **Largely covered.** Looped and recurrent-depth models with input injection see the whole input every iteration and must decide what
  to process (Huginn 2502.05171; Fan 2409.15647; Saunshi 2502.17416; Universal Transformers 1807.03819).
- **What is new:** selection combined with a memory as the *sole* cross-pass state, plus a step token. Phrase it as: "item-agnostic
  passes: the designer no longer supplies the per-step operand", not as a novel capability.
- **L1 caveat:** the whole problem is visible every pass, so recomputation is open. One pass can't do 4×4 (direct 0.002), but partial
  recomputation can mix with carriage.
  - Make SERIAL discriminate as §152: donors with a_donor ≠ a_rec and a different b-prefix, with a three-way prediction split
    (carriage / b-prefix / recipient).
  - **In VISIBLE, a swapped memory competes with the visible recipient operands, so expect a higher recipient share than in §152.**
    Pre-state the reading.

## C2 / C7 descriptives
- **4×5 with K = 5:** the no-op passes must leave the memory unchanged. Log the memory change on no-op passes (an attractor test).
- **The reader's per-pass text (C7):** a decoded S_j shows what the memory supports reading, not what the model uses. Pair it with the
  donor split.
