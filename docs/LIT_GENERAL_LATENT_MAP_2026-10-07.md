# Generalised latent reasoning models: the field mapped to C1–C7, and what to borrow (2026-10-07)

Literature session, at the user's request via Audit; Theory copied. IDs verified via the arXiv API (2026-10-07) or
earlier notes (LIT_INDEX).

C4 as revised (NORTH_STAR e51dac47): **C4** = supervision that exists at scale (final answers, or CoT text where it
naturally exists), with no hand-built intermediate states. **Strict C4** = final answers / outcomes only.

| Method | Supervision (C4) | One recipe across families? | C1 causal test? | Main claims (C2 / C3 / C5 / C6 / C7) |
|---|---|---|---|---|
| Coconut 2412.06769 | CoT curriculum (revised-C4 ✓) | GSM8K, ProsQA, ProntoQA, with per-task c and stages | 0-thought ablation only. External tests find bypass (Aswal 2606.12689; our ProsQA 0.738 → 0.732) | BFS-like probes (C7, correlational) |
| CODI 2502.21074 | self-distillation from CoT | GSM8K plus OOD maths | none. SCIT 2608.27265: computation sits in the V-cache, not the latents | C3 (3.1× compression) |
| Huginn 2502.05171 | LM pretraining, random recurrence | general LM (multi-task by pretraining) | none. Lu 2507.02199 finds little interpretable latent CoT | test-time depth scaling (C2 in depth); convergence exit (C5) |
| Ouro / LoopLM 2510.25741 | LM pretraining (7.7T) | general LM | none | C3 (1.4 / 2.6B ≈ 12B); learned depth (C5); degrades beyond trained depth (C2 ✗) |
| Attractor Models 2605.12466 | task supervision; fixed point, implicit differentiation | LM + Sudoku + Maze | none | **C6:** a stable fixed point, constant memory; Sudoku-Extreme 91.4% (27M) |
| LOTUS 2606.31779 | gold CoT-step token CE per latent (revised-C4 ✓) | maths expressions → NL maths | none | C3 (matches CoT at 3B, 2.5–6.9× faster); **C7: latents projected through the LM head recover gold steps** |
| Abstract-CoT 2604.22709 | CoT bottleneck SFT ↔ self-distillation, then RL; **discrete reserved vocabulary** | maths, instruction following, multi-hop | not in abstract | C3 (up to 11.6× fewer tokens); power-law code use |
| PLaT 2601.21358 | latent planner + separate decoder | maths | none | answer-aware stopping (C5); pass@k gains |
| SLPO 2607.19691 | **outcome-reward RL** on existing latent reasoners (a surrogate policy + correctness-supervised stopping head) | 2 latent reasoners, 2 backbones, 3 held-out benchmarks | none in abstract | **C5: difficulty-adaptive horizons**; Pass@8 / 16 up to +12 points |
| Latent Thought Flow 2606.16222 | GFlowNet sampler matching a reward-induced posterior over answer quality and cost | reasoning benchmarks | none | cost-aware compute (C3 / C5); stochastic transitions |
| Related (earlier notes) | SIM-CoT 2509.20317; KaVa 2510.02312; HRPO 2505.18454; SofT-GRPO 2511.06411; SWITCH 2606.13106; Thinking States 2602.08332; 2510.14095 | — | **SWITCH** exposes latent blocks to probing and causal intervention via boundary tokens | — |

**Analysis papers that do test C1:** Aswal 2606.12689, Li 2602.08783, SCIT 2608.27265, Zhang 2512.21711, Dilgren
2604.04902, Liang & Pan 2602.00449. All find bypass, cache carriage or non-serial use. Background surveys: 2507.06203,
2505.16782.

## (a) Generality across task families from one recipe
- Generality in the field means **pretraining-scale generality** (Huginn, Ouro) or **multi-domain fine-tuning**
  (Abstract-CoT: maths, instruction following, multi-hop; Attractor: LM + puzzles).
- **No paper claims one fixed recipe giving exact, certified state tracking across held-out algorithmic families.**
  Our §128 (5 families, 3 held out) has no direct counterpart. The comparison class is Delétang-style from-scratch
  suites.

## (b) Causal use (C1)
- **None of the method papers runs donor / interchange tests.** They report accuracy, ablations or projections.
- The C1 evidence in the field comes from analysis papers, and it is mostly negative.
- Our donor SERIAL protocol is a clear differentiator. SWITCH's boundary tokens are the only method design built for
  intervention.

## (c) RL on latents for strict C4 (§137)
- **Candidates:**
  - SLPO 2607.19691 (a surrogate policy + stopping head; outcome RL; **requires a working latent reasoner as the
    start**);
  - Abstract-CoT's RL stage over **discrete** codes;
  - HRPO 2505.18454 (a gated hybrid);
  - SofT-GRPO 2511.06411 / Soft Tokens Hard Truths 2509.19170 (Gumbel or noise exploration);
  - LTF (GFlowNet).
  - Discrete-program precedents: MAPO 1807.02322, RandoMer 1704.07926.
- **Our evidence:** RL from scratch failed (§69), and reward rarely teaches state (Allchin 2607.11953). Every
  successful latent-RL paper starts from a supervised or self-distilled latent model.
- **Recommendation for §137:**
  - Our FSQ codes are discrete. Make the code choice **stochastic** (a categorical over each dimension's levels) to get
    an **exact per-step log-likelihood**, then run GRPO with final-answer reward.
  - Warm-start from a carrier that has entered (§128 / §134 recipe), as Abstract-CoT and SLPO do.
  - This avoids SLPO's surrogate, because the latent is already discrete.

## What to borrow (ranked)
1. **Abstract-CoT's pipeline** (bottleneck SFT ↔ self-distil, then RL over a discrete vocabulary). It is the closest
   analogue to our FSQ carrier, and the template for §137.
2. **SLPO's stopping head** (correctness-supervised, refined by outcome RL) for C5 on our carrier.
3. **Huginn / Kuo randomised depth** (2502.05171; 2606.29983) for C6 reliability. Untested in our harness.
4. **Attractor-model fixed-point refinement** (2605.12466) as a no-snap stabiliser (C6).
5. **LOTUS's LM-head projection** of latents as a cheap C7 readability check (correlational; our edit / donor tests stay
   primary).
