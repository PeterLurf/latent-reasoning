# Novelty check for §92's claim line: latent restarts track S5 exactly to 2,048 in GPT-2 (2026-10-03)

Literature session, at the user's request via Audit; Theory copied. It folds in Audit's four priors, missing from
685bef0 and LIT_CHUNK_MACHINE. All arXiv IDs verified via the arXiv API on 2026-10-03.

## The priors (Audit's four, verified, plus two found now)

| Prior | What it does | Difference from §92 |
|---|---|---|
| **Recursive latent space reasoning, 2510.14095** | From-scratch 2-layer recurrent transformer on mod-23 arithmetic over DAGs. Input-adaptive recurrence (one iteration per graph layer); per-iteration supervision of node values ("algorithmic supervision"); a **discrete bottleneck** that argmax-decodes factored symbols and re-embeds them after every iteration; error-correction training (random corruption of latent values). Train N ≤ 32 → test N = 128 (≈ 4×), near-perfect. Mechanistic checks; no exhaustive certificate | **The closest mechanism-level prior**: supervised latent recurrence with discrete snapping. §92 at k = 1 (snap every op) is close to it. New in §92: pretrained GPT-2; Coconut-style per-op slots with continuous hand-off inside chunks (k = 4); a non-solvable group; DeltaProduct protocol to 2,048; exhaustive certificate; removal test |
| **Thinking States, 2602.08332** | Qwen2.5-0.5B. Text thoughts generated every few tokens, compressed into a fixed-size state injected into later tokens; supervised by text; extrapolates parity and variable assignment (train N ≤ 10/40 → 100) | Text thoughts, KV cache kept (no history cut), no S5, no certificate |
| **State tracking from code, 2602.14814** | S5 as REPL traces with state reveals, length 64. **"Transformers require dense state supervision… Accuracy drops as reveal spacing increases, with full permutations failing under any sparsity"**, including a fine-tuned pretrained **Qwen3-0.6B**. DeltaNet[−1,1] works at all spacings and extrapolates | A pretrained transformer *fails* S5 in their in-context text setting; §92 succeeds only with per-op latent slots, snapping and a history cut. **It also bears on §93: sparse supervision failing for transformers is partly a replication** |
| **RMT 2207.06881; Block-Recurrent 2203.07852** | Continuous state carried across segments | No snap, no S5, no certificate |
| **LIFT 2609.38149** | Pretrained 135M–1B LMs fed back a predicted continuous state per token, trained teacher-forced (states derived from an off-the-shelf LM's next-token distribution). A controlled state-tracking study in which a tiny LIFT beats same-size transformers trained on 8× more data | Continuous feedback with no snap or restart; the task is unspecified in the abstract; no S5 certificate |
| **Continuous mixture collapse, 2609.02049** (new) | Theory and experiments: softmax readout plus autoregressive feedback form a dynamical system that either amplifies one mixture component or contracts mixtures until indistinguishable. Pretrained rollouts sit on the amplifying side | Explains why unsnapped continuous hand-off drifts (§86, the §89 α → 1 collapse) and why §92's snap is load-bearing |

Also relevant, already in the notes:
- Exact S5 in non-pretrained recurrent models: DeltaProduct 2502.10297; Howe 2609.18966 (1.00 at 512); PD-SSM.
- A pretrained LM cuts off near its training length: Li et al. 2503.02854.
- Retrofitted recurrence in pretrained LMs: 2608.11233; T²MLR 2607.15178 (middle-layer recurrence retrofitted into a
  1.7B model, math reasoning; no S5).

## Any 2026 work with certified latent S5 tracking in a pretrained LM?
**None found.** Searched 2026 arXiv for latent / continuous / looped / recurrent-state work on S5 or A5 with pretrained
LMs. Hits are:
- from-scratch models (2510.14095, NextLat 2511.05963 on A5, LRNNs);
- pretrained LMs that fail S5 (2602.14814) or aren't tested on it (LIFT, T²MLR, Thinking States).

## Verdict: Audit's reading confirmed, with two sharpenings
- **"New as scoped, not as mechanism."** The mechanism (supervised latent recurrence plus discrete re-embedding, plus
  restart) is anticipated by 2510.14095 (from scratch, non-group task) and, in text form, by our own §71/CLR and
  PENCIL-style chunking.
- **Scoped claim:** pretrained GPT-2, per-op continuous latent slots, a snap to a structured state code with a history
  cut, S5 under DeltaProduct's protocol, certified exact at every tested length (to 2,048), a removal test (latents
  needed), and fewer FLOPs than the written-state scaffold.
- **Sharpening 1.** State the FLOPs comparator explicitly in the claim line: §71's written-state WC vs text CoT. The
  0.29–0.39× figure should name which one.
- **Sharpening 2.** Cite 2602.14814 as the contrast: a fine-tuned pretrained transformer fails S5 tracking in-context
  even with dense reveals. §92's success depends on the latent-slot + snap + cut scaffold, not on GPT-2 tracking state
  natively.
- **Conditions:** C1 (use) + C2 + C3; **not C4** (per-op labels).
- **Suggested claim line:** "To our knowledge, the first pretrained-LM latent state tracker certified exact on S5 to
  2,048 (DeltaProduct protocol). Its mechanism is a pretrained-LM, non-solvable-group instance of supervised latent
  recurrence with discrete re-embedding (cf. 2510.14095); per-op state labels are used (not C4)."
