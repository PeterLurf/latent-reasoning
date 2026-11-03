# External SOTA: latent / looped / recurrent-depth reasoning on small models (2024–2026)

Audit research pass, 2026-11-03. Web check against docs/LITERATURE.md, EXPERIMENT_BACKLOG.md and LIT_*.md. Numbers are from each paper's arXiv HTML unless marked. Status: **listed** = in LITERATURE.md or EXPERIMENT_BACKLOG.md; **LIT only** = only in dated LIT_*.md notes; **NEW** = not cited anywhere in docs/.

## 1. Comparison table (base ≤ 1.5B unless noted)

| Method (id) | Base | Benchmarks, accuracy | Compute claim | Depth extrap. | Halting | Causal latents | Status |
|---|---|---|---|---|---|---|---|
| Coconut (2412.06769) | GPT-2 124M | GSM8k-Aug 34.1 (CoT 42.9); ProntoQA 99.8; ProsQA 97.0 (CoT 77.5, No-CoT 76.7, iCoT 98.2) | 8.2 vs 25 tokens; 0.09 vs 0.26 s | none | no | contested: w/o-thought 95.5 on ProsQA; 2604.04902: Coconut = No-CoT on ProsQA/ProntoQA with identical data | listed |
| CODI (2502.21074) | GPT-2; Llama-3.2-1B | GSM8k-Aug 43.7 / 55.6; ProsQA 81.6, ProntoQA 95.1 (replication 2604.04902) | 2.7x speed-up, 6 latents | none | no | SCIT 2608.27265: computation in KV suffixes | listed |
| PCCoT (2506.18582) | GPT-2; 1B | GSM8k-Aug 49.48 ± 0.31 (CoT 44.1); 1B 53.35 (CoT 61.6) | 24 latents, 3 Jacobi iters; 0.199 vs 1.353 s | none | no | none | listed |
| SIM-CoT (2509.20317) | GPT-2 … 8B | Coconut +8.2 on GPT-2 (44.8); beats CoT by 2.1 at 2.3x token efficiency | tokens only | none | no | readout decoder only | listed |
| CoLaR (2505.16552) | Llama-3.2-1B-Inst, LoRA | GSM8k-Aug 40.1 (CoLaR-2) vs CoT 49.4, Coconut 23.1; 4-set avg 48.8 vs 53.6 at −53 % chain | chain length only | none | compression factor at test | none | listed (backlog) |
| KaVa (2510.02312, ICLR 2026) | 1B / 3B / Qwen2.5-0.5B, LoRA | GSM8k-Aug 56.5 at 1B (CODI 55.6, PCCoT 53.35, CoT ~63) | −62–92 % forward passes | none | no | none | LIT only |
| LOTUS (2606.31779) | GPT-2; 1B; 3B | GSM8k-Aug 44.1 ± 0.7 / 57.3 / 70.0 vs CoT 42.7 / 58.4 / 71.5 | thought phase 2.5x–6.9x faster; K=6, R=6 | R=7 "dips slightly" | no | readouts only | listed |
| SLT (2605.25745) | Llama-3.2-1B; Qwen3-4B | GSM 46.5 vs CoT 53.5 (Coconut 24.2, CoLaR-2 17.2 rerun); −58 % tokens | fig only | none | no | none | NEW |
| SubLAT (2610.40283, 29 Oct 2026) | GPT-2; 1B; R1-Qwen-1.5B | **ProsQA + OOD splits** (GPT-2, test / 7-hop / 8-hop / DAG / Dense): SubLAT 88.9 / 77.0 / 77.2 / 50.7 / 72.5; Coconut 98.0 / 73.3 / 72.5 / 58.7 / 65.7; CODI 82.5 / 69.9 / 70.1 / 59.3 / 65.8; CoT-SFT 71.9 / 11.6 / 10.5 / 36.4 / 22.0. 1B Coconut 98.1 / 53.4 / 49.6; CODI 97.5 / 66.3 / 65.6. GSM8k-Aug GPT-2 28.2 | 6 latents; 6.9 GPU-h/epoch vs CoT 0.35 | OOD hops only; T fixed at 6 | no | none | NEW |
| ProsQA-Ext latents (2609.35643) | 4-layer GPTNeoX d256, scratch | train H 3–6, test 7–12: bottleneck-latent > full-latent > pause > direct > CoT (figures only) | not reported | yes, K fixed 6 | no | yes: transplant, circuit pruning, head interventions | LIT only |
| Thinking States (2602.08332) | Qwen2.5-0.5B / 1.5B | Vars (modular variable tracking) train N=10 → test [10,100]: 33.8 (CoT 6.8); N=20 87.2 (CoT 35.5); N=40 97.7 | matches CoT on 2-hop QA, lower latency | length | no; steps = chunks | none | listed (NORTH_STAR) |
| Huginn (2502.05171) | 3.5B, 800B tok | GSM8K CoT 34.8 / 42.1 flex/strict at r=32; r=1 gives 0; r=64 EMA 47.2 | KL exit 5e-4 | trained r ~ Poisson mean 32; saturates 8–32 | zero-shot KL exit | 2507.02199: little latent CoT | listed |
| Ouro / LoopLM (2510.25741) | 1.4B, 2.6B, 7.7T tok | GSM8K 78.9 (1.4B R4) vs Qwen3-4B 72.9; synthetic 3-hop QA learned faster looped | trained exit gate ~66 % MMLU at mean 2.5 exits | "moderate degradation" at T=8 | yes, entropy-regularised gate | first-step KV cache collapses GSM8K to 18.7 | listed |
| Recurrent-depth retrofit (2608.11233) | Qwen2.5-0.5B, 6M adapter / 180M | step-supervised chain: 84 vs scratchpad 72; beyond depth 10 retains 53 vs 2.5 | 7.6x faster | ~1.5x supervised depth | open | none | listed |
| Loop, Think & Generalize (2604.07822) | 4-layer block, scratch | parametric multi-hop: train ≤12 hops, R=8 reaches ~19; "overthinking" past ~15 iters | none | yes | KL+entropy rule | patching found shortcuts | listed |
| DiscoLoop (2607.00341) | scratch; 440M LM | parametric 2-hop OOD ~100 vs looped 8.3; 3-hop OOD ~65 | fewer steps | not tested | no | realignment intervention lifts OOD 8→~100 | LIT only |
| RecurTrace (2609.03379) | 0.6B–8B looped | MathQA 56.9 at 2.0 mean loops, +2.2 over best fixed depth; ACT/PonderNet collapse to one loop | matched | — | oracle-trained head | none | LIT only |
| Halting diagnostics (2607.20519; 2611.02333) | Ouro + synthetic | post-hoc confidence readouts match or beat learned gates; probe vs native gate flips with the selection rule; untrained KL ties | not timed | — | subject | — | listed; NEW |
| Adaptive loops + memory (2603.08391) | iso-FLOP small | PonderNet-style per-layer looping helps math only | iso-FLOP | — | learned | none | NEW |
| Extrapolation theory (2610.14456; 2606.29983) | looped, scratch | finite extrapolation window; unreliable across seeds unless stopping is stochastic | — | theory | RL-halting | — | LIT only; listed |

## 2. What the table says

- GPT-2 GSM8k-Aug leaderboard: PCCoT 49.5 > LOTUS 44.1 ≈ CoT 44.1 > CODI 43.7 > Coconut 34.1. At 1B every latent method trails CoT (best KaVa 56.5 vs 61.6–63). No pretrained-base method gains from extra latent steps at inference; LOTUS, Ouro and Huginn flatten or degrade past the trained budget.
- ProsQA in-distribution is saturated and discredited (Coconut 97–98; 2604.04902 and 2606.12689 show the latents are not needed). The only live numbers are the OOD splits of 2610.40283: GPT-2 7/8-hop 77.0 / 77.2, DAG 59.3, Dense 72.5, same-base CoT-SFT ~11 on 7/8-hop.
- Halting is unsolved: learned gates collapse (RecurTrace) and untrained readouts tie them (2607.20519, 2611.02333). Our answer-stability rule (C5) is such a readout: defensible, not novel.
- Causal use of latents on a pretrained base rests on cache-swap tests only (Ouro, SCIT); the fullest causal battery (2609.35643) is from scratch.

## 3. Best next public target

**ProsQA with the 7-hop / 8-hop / DAG / Dense OOD splits of 2610.40283, on GPT-2 124M.**

1. Task match. Every edge is stated in the prompt ("A is B."); the question is reachability over ~23 nodes / 36 edges: content-selected multi-hop over stated facts, our §257 mechanism, no arithmetic. Backlog T3 already equates it with the §260 loop.
2. Headroom at matched base. Published GPT-2 accuracy beyond training depth tops out at 77 % at 7–8 hops; our model trained on d ≤ 4 holds 0.86 at d7 and 0.97 at d6. Same-base text CoT is ~11 % there, so C3 is winnable by a wide margin. Dense (72.5) and DAG (59.3) stress branching, the real test of whether DeltaMem can hold a frontier rather than one value.
3. Fresh and unclaimed. Posted 2026-10-29; every arm uses a fixed six latents, so accuracy against K' scaled with hop count would be a new kind of result here.

Caveats. The in-distribution 98 % is not worth beating (shortcuts). ProsQA needs set-valued state (branching); our memory carries one running value under a step cue. Whether SubLAT released its OOD splits is unverified; if not, regenerate from the Coconut ProsQA generator with the stated hops and the two-disjoint-paths recipe of 2609.35643, and say so.

Runner-up: the Vars task of 2602.08332 on Qwen2.5-0.5B, train N=10 → test [10,100]: 33.8 vs CoT 6.8. Almost our word-program task, but not GPT-2 (S7/S8 lines) and 100 ops is far past our d8.

## 4. Fair comparison protocol

- Same base and data: GPT-2 124M (secondary Llama-3.2-1B-Instruct), the Coconut ProsQA train split, no extra items; 3 seeds, mean ± sd, as in 2610.40283.
- Baselines run by us from official code on the same split: CoT-SFT, No-CoT, Coconut, CODI; SubLAT and KaVa cited. Report ID test plus all four OOD splits.
- Matched compute: FLOPs per item and bs-1 wall-clock. Our cost is K passes × prompt length (blocks 0–5 cached once) against CoT's ~49 generated tokens; give the full K' curve.
- Depth: train on the ProsQA hop distribution only; evaluate K' = 6 (every published arm) and K' scaled with hops. A C2 claim needs accuracy rising with K' past trained K in every seed.
- Mechanism gates, since Coconut's ProsQA gains survive removing the thoughts: NOT-DECODER / KV-blocking and step-swap on the OOD splits, pre-registered with the OC block and censoring rule.
- Supervision: per-step aux from ProsQA's proof traces is allowed under revised C4; no simulator-built frontier states.
- Halting: answer-stability rule against an untrained KL exit and a fixed-K oracle (2611.02333).

## 5. Verification

Every id in the table's first column plus 2604.04902 was fetched on its arXiv abs page on 2026-11-03 (title, authors, date confirmed). Ids cited only via existing project notes (2606.12689, 2608.27265, 2507.02199, 2606.29983) were not re-fetched.
