# Pre-registration check: toy permutation tracking (S3/S5) with a latent loop and written-state restarts (2026-10-01)

Literature session, for Theory; Audit copied. Builds on `LIT_STATE_TRACKING_PREP_2026-10-01.md` (211f6f1). IDs verified
(arXiv API).

## 1. State tracking and length generalisation
- **Theory:** transformers without CoT (and SSMs) cannot track non-solvable group state (S5) in TC⁰: 2310.07923,
  *Illusion of State* 2404.08819. **Shortcuts:** Liu et al. 2210.10749. All semiautomata have log-depth shortcuts and
  solvable groups have shallower ones; the learned shortcuts are brittle OOD.
- **Recurrent fixes:** 2411.12537 (negative eigenvalues), DeltaProduct 2502.10297, Implicit LMs 2502.07827 (fixed point).
- **Looped:** **Zhang et al. 2607.20594**: weight-tied loops on S4/S5/A5 word problems; A5 is learned, but **S5
  deadlocks** without an operator-first curriculum; extrapolation is seed-bimodal. Fan 2409.15647; Giannou 2301.13196.
- **Randomised position offsets:** Ruoss et al. 2305.16843 (randomised positional encodings boost length
  generalisation). The precedent for your offset scheme.
- **Chunked / dynamic-halt recurrence:** Chowdhury & Caragea 2402.00976 (Universal Transformer and chunk-wise Temporal
  Latent Bottleneck, plus dynamic halting, on flip-flop, ListOps and logical inference).

## 2. Latent / continuous-thought models on state tracking
- **Latent Reasoning with Supervised Thinking States** 2602.08332 (Feb 2026), **the closest competitor**. It writes
  thinking tokens every few input tokens and adds them back into the following input. On state-tracking tasks it
  "extrapolat[es] to longer sequences than seen during training", beating CoT. Its thoughts are *supervised* from natural
  language with teacher forcing.
- No Coconut-style (final-answer-only) latent loop on S3/S5 found.

## 3. Restart / segmenting with written states for state tracking
- Thinking States (above) is chunked written state *with supervision*.
- RMT 2207.06881 (memory tokens across segments, continuous); Temporal Latent Bottleneck chunks (via 2402.00976);
  PENCIL 2503.14337; Delethink 2510.06557.
- **Not found:** decoding a latent loop's permutation after K steps, writing it as the initial-arrangement block, and
  restarting on the suffix, trained label-free. **New as scoped.** Distinguish from Thinking States by: no thought
  supervision, a written *state* (the arrangement) rather than free thought tokens, and a restart with history cut.

## 4. Count- or length-keyed failures in looped models on state tracking
- Zhang 2607.20594: the budget law (speed set by the contract), and "roughly a third crack chained A5; the rest never pass
  the second position" (bimodal).
- Kuo 2606.29983: a spurious length–loop correlation on algorithmic tasks.
- Liang 2609.33144: finite-state composition among its tasks; failure beyond the horizon through degraded progress cues.
- **Your Q1 prediction (a count-keyed stop at step 8) is consistent with these.** With K = 8 fixed and N ≤ 8, short
  sequences carry slack 8 − N, which Prop. 4 says installs a clock.

## Design cautions
1. **S3 is solvable**, so shallow shortcuts exist (Liu). The model may solve S3 at N ≤ 8 in one pass, bypassing the loop
   (a C1 risk, as on ProsQA). Use **S5 as primary** for C1; keep S3 as a shortcut control, and report a K′ = 0 accuracy
   for both.
2. **S5 may not train at all** (Zhang's deadlock). Pre-register a fallback: an operator-first curriculum (Zhang) or A5.
3. **The restart writes the decoded permutation, which is checkable by construction.** Score the written-state edit test
   by IIA against the true permutation composition, as on graphs.

## Addendum (2026-10-01): what Zhang et al.'s "operator-first curriculum" is (for §64)
From 2607.20594 (full text, App. D):
- It is "an extended n=2 stage that masters the composition table before any chaining". It trains only length-2
  problems first (one composition of two elements), not single-element problems, generator subsets or pretrained
  embeddings.
- Exact S5 schedule: **"stages 2:20k, 4:10k, 8:10k, 16:10k, 32:12k"**. That is length 2 for 20k steps (~32% of 62k),
  then doubling lengths 4 / 8 / 16 / 32 in discrete stages.
- Reported: joint learning of operator and chain "deadlocks; sequenced learning does not, reliably", and the curriculum
  "dissolves the wall … 10/10 across two machines".

**Compared with §64's revision** (N = 1 for 10%, then maximum N linear 2 → 8 by 40%): the same principle (operator
before chain), but a different start (N = 1 vs n = 2), a shorter operator stage (10% vs ~32%), and linear rather than
doubling stages. Either call it "a length curriculum in the spirit of Zhang et al.'s operator-first curriculum", or
match theirs: a dedicated N ≤ 2 stage for ~30% of steps, then doubling stages.
