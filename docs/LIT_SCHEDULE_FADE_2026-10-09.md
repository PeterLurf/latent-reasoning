# Prior check for SCHEDULE-FADE: start from §161's per-pass digit-pair inputs, then fade them into the full problem text (after §171 READ-SLOTS left the memory unused) (2026-10-09)

Literature session, for Audit (the lead) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Details:** Goyal from the PDF; UT from the PDF (2026-10-09).

## 1. Goyal et al. 2310.02226 (pause tokens): confirmed
- **Models:** 1B and 130M decoder-only, pretrained on C4. Nine downstream tasks.
- **Pause-pretraining + pause-finetuning (PausePT_PauseFT)** gives gains on **8 of 9 tasks at 1B**: SQuAD +18% EM, CommonSenseQA +8%, GSM8K +1%
  (over a base of about 7%).
- **Pause only during finetuning on a standard-pretrained model (StdPT_PauseFT): "lukewarm gains", "mixed results", "even hurts in GSM8k".**
  Gains "show up in far fewer instances, and are relatively mild". The 130M model gains on fewer datasets.
- **For us:** this matches §107 PAUSE-4 and §171. Extra identical positions added to a pretrained LM at fine-tune time are mostly not exploited.
  That's a known effect, not a bug in our harness.

## 2. Curricula that remove *input* structure / segmentation while keeping the learned procedure
- **Direct precedent: not found.** Internalisation work removes *output-side* scaffolds:
  - ICoT-SI 2405.14838 removes CoT tokens;
  - Coconut 2412.06769 replaces language steps with thoughts;
  - our §89 / §142 fade reader targets.
- **The nearest input-side analogues:**
  - **Scheduled sampling, Bengio et al. 1506.03099:** progressively replaces the *given* per-step input (ground truth) with the model's own
    prediction. A curriculum from a supplied input schedule to a self-generated one. That's structurally the closest to SCHEDULE-FADE.
  - **Bansal 2202.05826, progressive training:** removes the *iteration-count* scaffold (train continuing from intermediate states), with
    recall (input injection) kept.
  - **Index hints, Zhou et al. 2310.16028 (RASP-L):** structure tokens added to the *input* to make a task learnable and length-general. They
    weren't removed by curriculum. It's a precedent for input scaffolding, not for fading it.
- **New as scoped:** an ICoT-SI-style fade from designer-segmented per-pass inputs to the full plain text, in a recurrent-memory model.
- **The design risk:** the procedure learned under segmented inputs may be *input-driven* (each pass's operand tells it what to do).
  - Once the operands are replaced by the full text, each pass must *select* its operand itself, which is exactly what §171 failed to learn.
  - A fade gives a gradual path to that, but needs a **per-pass selection signal** to survive (see 3).
  - Use a smooth mixture (per-pass input = segment with probability 1 − p, full text with probability p; p ramped), plus ICoT-SI's removal
    smoothing.

## 3. Identical input every pass, and recurrence ignoring the state / step
- **Universal Transformers 1807.03819 add fixed sinusoidal *timestep* coordinate embeddings** (P^t_{i,j} = sin(i/…) + sin(t/…)) at every
  recurrent step, so identical input is never presented identically. The paper doesn't ablate it.
- **Loop-index signals as the standard fix in looped LMs:**
  - **Relaxed Recursive Transformers 2410.20672:** depth-wise LoRA, a different adapter per loop. It improved on pure weight tying.
  - **Think-at-Hard 2511.08577:** depth-aware LoRA in the extra iterations.
  - Huginn instead randomises the start state s₀ and the loop count, and injects the input each step (no timestep embedding).
- **Evidence that repeated identical-input passes add little or collapse:**
  - Think-at-Hard: most tokens are correct after pass 1, and later passes cause "latent overthinking";
  - RecurTrace 2609.03379: ACT / PonderNet collapse to one loop;
  - Bansal: fixed-iteration nets degenerate without recall and progressive training;
  - our §137 note (a fixed point after one repeat without a repeat index), §141 r4 (no gain), §107, §171.
- **For us:** with identical input per pass, nothing distinguishes pass t from pass t+1 except the memory. If the memory's read path starts near zero
  (W_o fade-in), the cheapest solution is to ignore it, which is the "no-read basin" (§111).
- **Two literature-grounded remedies to combine with the fade:**
  - (a) a **pass-index signal** (a UT timestep embedding, or per-pass LoRA as Relaxed Recursive / TaH). With a pass index, the model can learn
    "on pass t, read column t" from the text.
    - Caveat: an absolute pass index extrapolates poorly (the clock issue). Use relative or randomised offsets (LIT_C2_MULT_SCHEDULE) for C2.
  - (b) a **reader / step aux** on the memory (STEP-AUX, LIT_STEP_AUX; §111 READ), which forces the memory into use.

## Verdict
- New as scoped.
- The pause result (fine-tune-only pauses are lukewarm) explains §171's null.
- **Pair the fade with a pass-index signal and / or a memory-reading auxiliary.** Otherwise the fade's endpoint is §171 again.
