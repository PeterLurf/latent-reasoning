# Prior check for §93: chunk-end-only supervision for latent restarts (1 label per k ops) (2026-10-03)

Literature session, for Theory; Audit copied. Short note, written before any §93 run. arXiv IDs as verified in earlier
notes (LIT_SOTA_TARGETS, LIT_REVIEW, LIT_LATENT_FROM_DISCRETE, LIT_HANDOFF_COLLAPSE).

## 1. Sparse intermediate supervision for latent/continuous CoT
- **Coconut 2412.06769 is chunk-level supervision.** c latents replace one text step and are supervised only through
  the text that follows. It works on ProsQA/ProntoQA (within one-pass capacity: our addenda G/J/Q) and partly on GSM8K
  (34.1). It always uses a text-to-latent **curriculum**: steps are replaced a few at a time.
- **ICoT-SI 2405.14838:** dense CoT supervision removed gradually (Δ tokens per epoch, removal smoothing) until none
  remains. GPT-2 small reaches 9×9 multiplication 0.99 and GSM8K 0.30. **Sparsity reached by curriculum, not imposed
  from the start.**
- **CODI 2502.21074:** a single alignment point (the answer token's hidden state) for 6 latents. Sparse, but distilled
  from a teacher.
- **SIM-CoT 2509.20317:** without per-latent step supervision, training collapses as the latent count grows. Per-step
  decoding fixes it (the dense end).
- **RELAY 2502.08482:** per-iteration alignment (dense).
- **2606.20075** analyses trajectory-supervision granularity and finds accuracy tracks the latent–step mutual
  information. It is the closest published density study.
- **No published density threshold for latent CoT.** Our §85(i) is the only threshold we have: free-table A5 needs
  ≥ 10% short problems for 5/5, and 3% gives the right accuracy without certification.

## 2. Can chunk-level targets suffice for a latent multi-step state?
- **Yes, with footholds; no, without them** (our rule 1, and the evidence for it):
  - **Delétang 2207.02098:** final-output-only supervision, and LSTMs solve regular tasks. But training spans lengths
    1–40, so short sequences supply footholds.
  - **Lee 2606.07254:** final-state-only on length-8 sequences succeeds on S3×S3, but with a hand-built projection
    (an inductive bias in place of footholds).
  - **§76/§82:** final-only with lengths ≥ 3 fails on every order-60 group. §87: short labels as a *warm start only*
    give 5/5, then pure long training holds.
  - §88/§91: per-iteration targets compose. §80: end-state targets on k-op words memorise.
- **Theory:** a chunk-end label over k unsupervised hand-offs is a leap of k (Abbe 2302.11055). Sub-task supervision is
  what makes composites learnable (Wies 2204.02892).
- **Prediction:** fixed-length chunks of exactly k with only the chunk-end label behave like ge3 for latents. They
  likely fail at k = 8, and possibly at k = 4.

## 3. Design flags
1. **Chunk-length distribution is the key hidden variable.**
   - If training chunks have random length 1..k (each supervised at its end), length-1 chunks label single hand-offs:
     footholds, like §76 std.
   - If every chunk is exactly k, there are no footholds.
   - Pre-register which. Better, run both:
     - **(i) variable 1..k**, the realistic "sparse but with short chunks";
     - **(ii) fixed k**, the strict C4-direction test.
   - Report the effective fraction of single-hand-off labels (cf. §85's p).
2. **A warm-then-sparse arm** (§87's lesson; the ICoT-SI pattern): per-op labels for the first ~¼ of training, then
   chunk-end only. This tests "basin entry" for the hand-off, not acquisition from sparse labels alone. Label it as
   such.
3. **Identifiability.**
   - Intermediate slots aren't identified: any code whose composition gives the right chunk-end state fits. So they
     may not be decodable (C7 lost) even when chunk-end accuracy is perfect.
   - Measure per intermediate slot:
     - a post-hoc linear probe for the true state (decodability);
     - interchange or patching at that slot (causal use; DAS-style IIA).
   - "Carries the state" requires both.
4. **Chunk-end shortcut.**
   - The supervised slot k can attend to all k op tokens and try to compute the product in one pass, bypassing slots
     1..k−1.
   - With S5, 120^k words can't be memorised (§80's mechanism needs repetition). So expect either latent hand-off use or
     failure, not a silent shortcut. Still confirm with **intermediate-slot removal**: ablate slots 1..k−1 only.
5. **Budget.** Keep 24k+ (§92's lesson). Report per-slot probe accuracy over training, to see whether intermediate
   states emerge before or after chunk-end accuracy.

**Wording if it works:** "chunk-end supervision (1 label per k ops) suffices for latent restarts, given [variable
chunk lengths / a warm start]". This is a step toward C4, not C4. Strict C4 is fixed-k chunk-end labels with no warm
start, or final answers only.
