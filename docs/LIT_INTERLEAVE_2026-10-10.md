# Prior check for INTERLEAVE: k latent think passes before each answer token, with a DeltaMem memory carried across tokens (conditional on §179 CAPACITY / PARTIAL) (2026-10-10)

Literature session, for Theory; Audit copied.
- **IDs:** verified via arXiv abstract pages.
- **Huginn details:** from its HTML (2026-10-10).

## 1. Huginn 2502.05171: is there state across tokens beyond the KV cache?
- **Training: no.** Each token's recurrence starts from a randomly sampled s₀. Across tokens only the KV cache links positions (per-iteration KV
  entries).
- **Inference, two zero-shot features:**
  - **KV-cache sharing:** "the model is able to attend to the latest cache entry from every previous token, even if computed at different recurrent
    depths".
  - **"Zero-shot continuous chain-of-thought":** "instead of sampling a random initial state s₀ at every generation step, we can **warm-start with the
    last state s_r from the previous token**". This builds "a computation deeper than the current number of recurrence steps", related to Coconut.
- **So a carried cross-token latent state exists in Huginn only as an untrained, inference-time warm start.** INTERLEAVE trains it, through a separate
  memory.

## 2. Quiet-STaR 2403.09629
- Generates a **text** rationale (with start / end-of-thought tokens) after **every** token in parallel, with a learned mixing head blending
  with- and without-thought predictions. Trained by REINFORCE on general text. Mistral-7B zero-shot GSM8K 5.9 → 10.9%.
- **State across tokens:** only the text and KV. The rationales are discarded after each position's prediction. Per-token thinking, but textual and
  not carried.

## 3. Per-token latent computation with cross-token state: the closest designs
| Design | Per-token extra computation | Cross-token latent state | Notes |
|---|---|---|---|
| **Recurrent Looped Transformer 2610.07591** | a recurrent decoder of L_D blocks per token | **the previous token's final decoder state is merged into the current token** (the recurrence path grows with the sequence) | from scratch; parity / S5 / modular arithmetic. **Addition fails beyond 8 digits** (LIT_FIXED_CHUNK). Closest in mechanism |
| **T²MLR 2607.15178** | — | the previous token's cached *middle-layer* state is injected into an earlier layer | retrofit at 1.7B; maths gains |
| **Fast-slow 2604.01577** | **T inner updates per observation** | a **persistent, never-reset latent** | from scratch; streaming tasks. Structurally the closest to "k passes per step + carried state" |
| Feedback Transformer 2002.09402 | — | all layers' past representations, fed back | from scratch |
| CoTFormer 2310.10845 | repeated passes per token, attending to earlier passes | via attention to earlier passes' tokens | adaptive budget |
| Think-at-Hard 2511.08577 | extra iterations on hard tokens only | duo-causal attention across iteration depth and sequence | pretrained, looped; a decider triggers per token |
| Mixture-of-Recursions 2507.10524 / Mixture-of-Depths 2404.02258 | per-token recursion depth | KV only | routers |
| Thinking Tokens 2405.08644; pause tokens 2310.02226 | extra positions | KV only | not recurrent |

- **Not found:** per-token *latent* think passes with a **separately carried fast-weight memory** (not the residual, the KV or a merged hidden state)
  in a pretrained LM.
- **INTERLEAVE is new as scoped.** The closest are RLT (previous-token state merge, from scratch) and the fast-slow recurrence (inner loops plus a
  persistent latent).

## 4. Interleaving latent computation with answer emission on multi-digit arithmetic
- **The separator paper 2610.06605:** one extra (non-recurrent) position **before each answer digit** turns 4×4 no-CoT multiplication from 1% to 89%
  (from scratch). The carry is computed at that slot.
  - That's "computation interleaved with emission" in its simplest form, and it's **the precedent INTERLEAVE generalises.**
- **RLT 2610.07591:** per-token recurrence on addition, trained 1–8 digits. It fails past 8 (68% at 9; about 15% at 32).
- **Pause / thinking tokens:** general-purpose, not arithmetic-specific.
- **Not found:** *latent recurrent passes* before each answer digit with a carried memory on multi-digit multiplication.

## Flags
1. **The k passes see the same context**, so they're identical-input passes (the §171 problem). They differ only via the memory and the current
   ans_<j.
   - Give each pass index a signal (UT timestep embedding / per-pass LoRA; LIT_SCHEDULE_FADE), or k passes collapse to 1.
   - Measure accuracy vs k with k = 0 as THINK-NECESSARY: 2610.06605 shows one extra slot may already suffice.
2. **The memory is never reset across tokens, so its norm grows** with answer length (additive heads). Log per-token norm. For C2 (longer products),
   cover more answer tokens (LIT_S166 state coverage; LIT_RAND_K).
3. **Bypass:** the decoder sees [problem, " =", ans_<j] and so can compute from the context (L1 / §179 SHORTCUT). Pair with DEC-DROP or a blind variant
   if §179 reads SHORTCUT.
4. **Cost:** (k + 1) passes per answer token, so about 8 digits × (k + 1) passes for 4×4. Report vs TC-M and ICoT-SI (C3).
