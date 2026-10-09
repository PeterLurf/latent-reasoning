# Prior check for INJECT: every latent think pass re-presents the full plain problem text, with a recurrent memory as the only cross-pass state (2026-10-09)

Literature session, for Audit (the lead, on the user's "work on task agnosticness") and Theory. INJECT is also the latent arm of
DEEP-ProsQA.
- **IDs:** verified via arXiv abstract pages.
- **Details:** from the full HTML of 2409.15647, 2502.05171 and 2604.01577.

## 1. Fan et al. 2409.15647, Looped Transformers for Length Generalization
- **Input injection:** "for each step, the original input sequence is injected together with the output from the previous decoder
  block". That is, the full input is re-added every loop iteration (following Bai et al., DEQ).
- **Ablation:** yes. "FOP-Loop-Adaptive-WO" (no injection) generalises **worse in general**. The effect is clearest on **Addition,
  Binary Sum and Unique Set**.
  - "With input injection, the model can maintain a strong connection to the original input, preventing information loss."
- **Tasks and training lengths:**
  - parity, addition, copy, binary sum and unique set, trained on lengths [1, 20);
  - binary multiplication (length = the second number; T = the product of the lengths), trained on [1, 12).
  - Tested beyond: e.g. parity to 40, addition to 30, copy 35, multiplication 16, binary sum 24, unique set 35.
- The number of loops T(n) is tied to the problem length and is adaptive at test. From scratch, and no persistent memory: the state is
  the residual stream.

## 2. Huginn / recurrent depth 2502.05171
- **Per-iteration injection:** the recurrent block starts with an **adapter A: ℝ^{2h} → ℝ^h on the concatenation [s_i ; e]**, where
  e = the prelude's embedding of the *whole input*.
  - So e is re-injected at every recurrence (as in the deep-thinking literature).
  - Addition (s + e) works equally well for small models; **concatenation works best at scale**.
  - The prelude gets gradients from every step.
- Their first training run with a parameter-free additive adapter (plus other choices) failed. A learned adapter plus an embedding
  scale fixed it.

## 3. 2604.01577, "Exploration of Fast-Slow Latent Recurrence for Train-Short, Test-Long Generalization" (Takashiro, Koyama, Miyato, …)
- **The claim:** a **persistent fast-slow recurrence.**
  - The latent Z is **never reset across observations**.
  - For each incoming observation it runs **T weight-shared inner updates** (T = 5 by default) with **the encoded observation C
    injected at every inner step**: Z(t+1) = R(Z(t), C).
  - It improves OOD (train-short, test-long) over recurrent, state-space and Transformer baselines, on symbolic sequence prediction,
    navigation and partially observed RL.
  - Ablations credit **T > 1 inner iterations with weight sharing**, **state-dependent transitions** and **feature-wise non-linear
    mixing**. The default core is AKOrN (oscillatory neurons) + FFN.
- **Does the state persist with input injection?** **Yes:** the state persists across observations, *and* each observation is injected
  at every inner step.
- **One difference from INJECT:** it injects the *current observation* (streaming), not the full problem text, and it's from scratch.
  - Structurally it's the closest precedent to §165 FIXED-CHUNK + think passes: observations → inner loops → a persistent state.
  - INJECT's "re-present everything" is the Huginn / Fan pattern.

## 4. Precedent for a pretrained LM + cross-pass memory + full-input re-presentation per pass
- **Pieces that exist:**
  - **Pretrained + per-loop input re-injection:** the recurrence retrofits 2608.11233 and McLeish 2511.07384 (prelude output
    re-injected each loop, Huginn-style); Relaxed Recursive 2410.20672. **Their state is the residual stream, not a separate
    memory.**
  - **Pretrained + segment memory:** RMT / ARMT 2407.04841 (GPT-2). Each segment is *new* input, not a re-presentation.
  - **Text-level re-presentation:** "Re-Reading" prompting, 2309.06275 (repeat the question; better reasoning in LLMs).
- **Not found:** a pretrained LM whose think passes each see the *full problem text* and share *only* a recurrent memory. **New as
  scoped.**

## Flags for registration
1. **INJECT reopens L1 by design.** With the whole problem visible every pass, recomputation is always available.
   - C1 can't come from swaps: a whole-memory swap is the donor's chain, and a partial swap competes with the visible input.
   - It has to come from **K-removal against a failing K = 0 / single-pass control on problems beyond the measured one-pass boundary**
     (§165's rule; THEORY_C1_FORCING).
   - DEEP-ProsQA beyond L* is the right setting.
2. **The fusion operator matters.**
   - Huginn: concatenation + a learned adapter beat addition at scale, and a parameter-free additive start failed.
   - For a pretrained GPT-2, **re-presenting the text as tokens** (the memory read mid-stack, the text through the normal embedding) is
     itself the injection. No adapter is needed, which keeps it function-preserving.
3. **The loop count:** Fan ties T to the problem size and halts adaptively. Fixing K for all depths would hide whether passes scale
   with depth. Log accuracy vs K per depth.
4. **Per-pass cost:** each pass re-encodes the full text, so compute is K × (text length). Report it vs TC-M (C3); it's dearer than
   streaming (2604.01577's per-observation design).
