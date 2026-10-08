# Prior check for FIXED-CHUNK: DeltaMem-R on plain "a × b =" text in fixed k-token chunks, plus K think passes, then a read-only decoder (2026-10-08)

Literature session, for Audit (the lead, answering the user's "is the architecture task-agnostic?") and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Details:** from the full HTML of 2207.06881, 2407.04841 and 2610.07591 (2026-10-08).

## 1. RMT / ARMT / Recurrent Looped Transformer on multi-digit arithmetic with fixed segments
- **RMT 2207.06881:**
  - copy, reverse, associative retrieval, and **quadratic equations**, where each example is an equation *plus its written solution
    with the discriminant* plus the answer. RMT "almost perfectly solves" it across multiple segments.
  - That's a worked-solution (CoT-like) format, **not direct multi-digit arithmetic**, and there's no length-generalisation test.
- **RMT scaling 2304.11062:** fact retrieval to 2M tokens, plus LM perplexity. No arithmetic.
- **ARMT 2407.04841:** associative retrieval and BABILong, on GPT-2 137M. No arithmetic.
- **Recurrent Looped Transformer (RLT) = arXiv 2610.07591** (Zhang, Feng, Qin; submitted 2026-10-06; the project page matches):
  - a parallel causal encoder plus a recurrent decoder that merges each token's encoder output with the previous token's final
    decoder state;
  - six tasks, three seeds.
  - **Addition, the only multi-digit arithmetic: trained on 1–8 digits; length generalisation fails.**
    - 9 digits: RLT-1 7+1 68.1 ± 3.6% vs a Transformer's 58.6 ± 2.4%;
    - 32 digits: every model at 14.9–16.8%.
  - Parity, 40 → 256 bits: 100% in two splits. Swap-S5 at 8× length: 97% vs < 1%. Modular arithmetic beyond the training length: up
    to 93% vs 33%. Standard S5: at chance.
  - **The flag for FIXED-CHUNK:**
    - Feeding back **once per 4-token chunk** keeps 64-bit parity at 99%;
    - but **swap-S5 at length 64 drops from 100% (per-token) to 20%.**
    - Tasks that need a state update at every token degrade when the recurrence is chunked.
- **Summary:** no paper shows *direct* multi-digit multiplication with fixed-segment recurrent memory, and none shows length
  generalisation for multi-digit arithmetic through segment recurrence. The closest (RLT addition) fails beyond its training width.

## 2. Recurrent memory plus K extra "think" steps on a constant input, with accuracy rising in K
- **The direct precedent is ACT, Graves 1603.08983:**
  - an RNN takes **extra ponder steps on the same input** (a flag marks the repeats);
  - **tasks include multi-digit addition** (sums over sequences of numbers), parity, logic and sort;
  - pondering improves accuracy, with a learned halting.
  - PonderNet 2107.05407 is the modern variant.
- **Depth recurrence on the same input with accuracy rising in the number of iterations:** Huginn 2502.05171; Fan 2409.15647 (adaptive
  loop count beyond training); Saunshi 2502.17416.
- **Pause tokens 2310.02226** (non-recurrent positions; gains peak at some count); our §107 PAUSE-4 didn't help.
- **Not found:** RMT / ARMT-style *segment* memory with K empty think segments, and a K-ablation. **New in that form.**
- **What our own evidence says:** §141 r4 (repeats) gave no gain, and K think passes after the input is fully buffered are 158-2's
  "buffer + read-time compute".
  - The K curve says how much serial post-input computation the memory supports. It is not, on its own, a C1 claim.

## 3. The Recurrent Transformer, 2604.21215
- Each layer attends to KVs computed **from its own activations** (layer-wise recurrent memory), so temporal depth grows with
  position at standard decoding cost.
- An exact tiling algorithm cuts the training memory traffic. On 150M / 300M C4 pretraining it beats a parameter-matched Transformer
  with fewer layers.
- **Bearing on FIXED-CHUNK:**
  - **indirect.** It's a from-scratch, every-token recurrence, so there's no chunk-size choice at all. That's the most task-agnostic
    form.
  - No arithmetic and no retrofit to a pretrained model are reported.
  - It's an architectural alternative if chunk size proves to be a hidden per-task choice. It's not evidence for the lead.

## Design flags
1. **Chunk size k is itself a design choice.**
   - RLT shows chunked feedback breaks per-token state tracking (S5 100 → 20%).
   - For "a × b =" text, the operands are buffered before the think passes, so the chunk only *loads* the memory, which is fine. The
     risk is any task streamed in chunks.
   - **Report k ∈ {1, 4, 8}** (or at least two values) to show the result doesn't hinge on k.
2. **BPE chunks misalign with digits.** Plain GPT-2 text ("1234 × 5678 =") tokenises numbers into irregular multi-digit tokens
   (Singh & Strouse 2402.14903).
   - Either keep the bench's space-separated digits (" 1 2 3 4"), which is still "plain text" and still task-agnostic, or report
     that the tokenisation differs from §161.
3. **The answer decoder reads only the memory.** That's the right C1 setting.
   - The donor test is now whole-memory swaps *after input loading* (between problems), or swaps between think passes. Pre-state
     which one is the rule.
4. **The K-ablation and C2:**
   - test K beyond the trained K (accuracy should plateau, not degrade: the attractor property);
   - test longer operands with K scaled. Fan and Huginn show K extrapolation is possible when the loop converges.
