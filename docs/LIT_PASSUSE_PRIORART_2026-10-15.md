# LIT sweep: forcing and rewarding multi-pass latent use (2026-10-15)

*Written 2026-10-15 by a Theory Opus subagent standing in for the offline Literature session (deep research session, user request 21:58 EDT; pass-global memory constraint 22:07 EDT). Rows marked (abs) are abstract-only and need a full-text check before a registration relies on their loss definitions.*

Stand-in for the Literature session. **IDs:** all 81 verified on their arXiv abstract pages (curl, citation_title / citation_author meta tags). "abs+full" means I also read the arXiv HTML full text. Definitions marked (abs) come from the abstract only: they need a full-text check before a registration relies on them. Tags: [THEOREM] / [DERIVATION] / [EMPIRICS], with the setting.

## Summary
1. **M1 (small core + question-once):** the closest are the pretrained retrofits (McLeish 2511.07384, Shapiro 2608.11233, ETD 2510.07358, Penelope 2607.25915). All of them keep the input reachable on every loop, through injection or a prompt KV. **New:** cutting input access after pass 1. **Known against:** recall / no-injection ablations (Bansal 2202.05826, Fan 2409.15647, 2604.11791) and the project's own §165.
2. **M2 (append-only per-pass slots + progress term):** **LOW novelty.** Coconut / CODI in-context latents, TaH duo-causal attention 2511.08577 and RecurTrace loop-memory attention 2609.03379 already do this; step terms come from SIM-CoT / LOTUS / Shapiro. A pass-global memory makes pass-1 precompute *easier* [DERIVATION].
3. **M3 (write-to-keep / rotation):** write-to-keep exists by construction in RMT / TTM, with gated decay in InfiLoop 2610.11570, Expire-Span and AGCLR 2606.07720. **Not found:** rotation used as a forcing device. Under attention, re-writing is a cheap copy, so it forces re-writing, not re-computing [DERIVATION].
4. **M4 (marginal-pass reward):** the iteration gain ΔL is used **only to train halting gates**: Ouro Stage II I^(t) with the LM frozen; TaH2 labels δ = ℓ^(m) − ℓ^(m+1). The text-RL analogues are PAV progress, MRT and the SCoRe bonus. **New:** putting the gain on the backbone. **Gameable** by sabotaging k < K [DERIVATION].
5. **M5 (IIT across passes on GSM8K):** IIT 2112.00826 (including an LSTM), SIIT 2407.14494 and causal distillation 2112.02505. Nothing found on looped models or GSM8K. **HIGH novelty.**
6. **M6 (input removal / perturbation after pass 1):** the closest are word dropout 1511.06349 and Markovian Transformers 2404.18988, both text-side. The perturbation form is the SIIT input-side term. **New for latent passes.**
7. **M7 (pass-global KV + bounded core + R/W budgets + non-redundancy reward):** the closest is RecurTrace (loop memory + loss-gain halting). Budgets have precedents in MoD 2404.02258, MoR 2507.10524 and sparse R/W 1610.09027. Non-redundancy losses exist only for vision / soft-CoT latents (ProLaViT 2607.02907, SoftCoT++ 2505.11484). **New by combination**, and the non-redundancy term is Goodhart-prone.
8. **Strongest published failure evidence:**
   - Huginn's "Bad Run 2" "learned early to ignore the incoming state s" (val. ppl same at 1 or 32 recurrences);
   - 2609.01924 (Huginn recomputes from the re-injected input);
   - "ACT and PonderNet collapse to one loop" (RecurTrace);
   - Ouro's exit distribution self-collapses to T_max without an entropy term;
   - later loops worsen 15.9–21.7% of tokens (TaH2); "latent overthinking" (TaH).
9. **Lower bound:** Feng 2305.15408 is about arithmetic expressions mod p, which is the project's nested task (Feng mod 11). It is asymptotic (TC0 ≠ NC1). It **does not** show that one 12-layer pass fails at n ≤ 8, so "one pass is insufficient" must be shown empirically (K′1 control).
10. **Pending:** three full-text helper checks (exact loss equations for Saunshi, Fan, Bansal, CODI, Coconut, LOTUS, SCoRe, PAV, SIIT, Feng) hadn't returned by hand-off. The rows marked (abs) need them.

## Verified IDs
| id | title | first author | year | verified |
|---|---|---|---|---|
| 2502.17416 | Reasoning with Latent Thoughts: On the Power of Looped ... | Saunshi | 2025 | abs |
| 2502.05171 | Scaling up Test-Time Compute with Latent Reasoning: A ... | Geiping | 2025 | abs+full |
| 2510.25741 | Scaling Latent Reasoning via Looped Language Models | Zhu | 2025 | abs+full |
| 2409.15647 | Looped Transformers for Length Generalization | Fan | 2024 | abs |
| 1807.03819 | Universal Transformers | Dehghani | 2018 | abs+full |
| 2511.07384 | Teaching Pretrained Language Models to Think Deeper with ... | McLeish | 2025 | abs |
| 2608.11233 | Retrofitting Recurrent Depth into a Pretrained Language ... | Shapiro | 2026 | abs |
| 2510.07358 | Encode, Think, Decode: Scaling test-time reasoning with ... | Koishekenov | 2025 | abs |
| 2607.25915 | Penelope: Localized Latent Recurrence for Efficient ... | Chen | 2026 | abs+full |
| 2610.06833 | Towards Looped Models Done Right, Part II: Rethinking at ... | Huang | 2026 | abs |
| 2607.27656 | Looped Transformers with Source-Centered State Evolution | Kim | 2026 | abs |
| 2609.01924 | Looped Transformers under the Jacobian Lens: Does the ... | Wang | 2026 | abs |
| 2604.11791 | A Mechanistic Analysis of Looped Reasoning Language Models | Blayney | 2026 | abs+full |
| 2601.10242 | Loop as a Bridge: Can Looped Transformers Truly Link ... | Chen | 2026 | abs+full |
| 2609.35748 | Improving Test-Time Scaling with Adaptive Looped ... | You | 2026 | abs+full |
| 2511.08577 | Think-at-Hard: Dynamic Looped Transformers for Improved ... | Fu | 2025 | abs+full |
| 2609.03379 | RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory | Wang | 2026 | abs |
| 2610.11570 | Scaling to Tens of Thousands of Test-Time Iterations ... | Li | 2026 | abs |
| 2606.18023 | LoopCoder-v2: Only Loop Once for Efficient Test-Time ... | Yang | 2026 | abs |
| 2605.07721 | Memory-Efficient Looped Transformer: Decoupling Compute ... | Vendrell | 2026 | abs |
| 2607.15456 | Looped Latent Attention: Cross-Loop KV Compression for ... | Neill | 2026 | abs |
| 2610.07591 | Recurrent Looped Transformer | Zhang | 2026 | abs |
| 2507.10524 | Mixture-of-Recursions: Learning Dynamic Recursive Depths ... | Bae | 2025 | abs+full |
| 2412.06769 | Training Large Language Models to Reason in a Continuous ... | Hao | 2024 | abs |
| 2502.21074 | CODI: Compressing Chain-of-Thought into Continuous Space ... | Shen | 2025 | abs |
| 2606.31779 | Bridging the Gap Between Latent and Explicit Reasoning ... | Fan | 2026 | abs |
| 2509.20317 | SIM-CoT: Supervised Implicit Chain-of-Thought | Wei | 2025 | abs |
| 2606.07720 | Why Limit the Residual Stream to Layers and Not Tokens? ... | Farhan | 2026 | abs |
| 2512.21711 | Do Latent Tokens Think? A Causal and Adversarial ... | Zhang | 2025 | abs |
| 2602.00449 | Do Latent-CoT Models Think Step-by-Step? A Mechanistic ... | Liang | 2026 | abs |
| 2602.08783 | Dynamics Within Latent Chain-of-Thought: An Empirical ... | Li | 2026 | abs |
| 2604.06374 | The Illusion of Superposition? A Principled Analysis of ... | Rizvi-Martel | 2026 | abs |
| 2203.08913 | Memorizing Transformers | Wu | 2022 | abs |
| 2207.06881 | Recurrent Memory Transformer | Bulatov | 2022 | abs |
| 2203.07852 | Block-Recurrent Transformers | Hutchins | 2022 | abs |
| 2002.09402 | Addressing Some Limitations of Transformers with ... | Fan | 2020 | abs |
| 2211.09119 | Token Turing Machines | Ryoo | 2022 | abs |
| 2105.06548 | Not All Memories are Created Equal: Learning to Forget ... | Sukhbaatar | 2021 | abs |
| 2310.02226 | Think before you speak: Training Language Models With ... | Goyal | 2023 | abs |
| 2404.15758 | Let's Think Dot by Dot: Hidden Computation in ... | Pfau | 2024 | abs |
| 2602.08332 | Latent Reasoning with Supervised Thinking States | Amos | 2026 | abs |
| 2510.14095 | Unlocking Out-of-Distribution Generalization in ... | Altabaa | 2025 | abs |
| 2402.02622 | DenseFormer: Enhancing Information Flow in Transformers ... | Pagliardini | 2024 | abs |
| 2502.12170 | MUDDFormer: Breaking Residual Bottlenecks in ... | Xiao | 2025 | abs |
| 2409.19606 | Hyper-Connections | Zhu | 2024 | abs |
| 2502.09245 | You Do Not Fully Utilize Transformer's Representation ... | Gerasimov | 2025 | abs |
| 2309.17453 | Efficient Streaming Language Models with Attention Sinks | Xiao | 2023 | abs |
| 1603.08983 | Adaptive Computation Time for Recurrent Neural Networks | Graves | 2016 | abs |
| 2107.05407 | PonderNet: Learning to Ponder | Banino | 2021 | abs |
| 1409.5185 | Deeply-Supervised Nets | Lee | 2014 | abs |
| 2506.21734 | Hierarchical Reasoning Model | Wang | 2025 | abs |
| 2510.04871 | Less is More: Recursive Reasoning with Tiny Networks | Jolicoeur-Martineau | 2025 | abs |
| 2202.05826 | End-to-end Algorithm Synthesis with Recurrent Networks: ... | Bansal | 2022 | abs |
| 2211.09961 | Path Independent Equilibrium Models Can Better Exploit ... | Anil | 2022 | abs |
| 2606.29983 | Stabilizing Extrapolation in Looped Transformers via ... | Kuo | 2026 | abs |
| 2607.20519 | Adaptive Depth in Looped Transformers: Diagnosing ... | Popescu | 2026 | abs |
| 2602.10520 | Prioritize the Process, Not Just the Outcome: Rewarding ... | Williams | 2026 | abs |
| 2410.08146 | Rewarding Progress: Scaling Automated Process Verifiers ... | Setlur | 2024 | abs |
| 2503.07572 | Optimizing Test-Time Compute via Meta Reinforcement ... | Qu | 2025 | abs |
| 2409.12917 | Training Language Models to Self-Correct via ... | Kumar | 2024 | abs |
| 2404.16710 | LayerSkip: Enabling Early Exit Inference and ... | Elhoushi | 2024 | abs |
| 2207.07061 | Confident Adaptive Language Modeling | Schuster | 2022 | abs |
| 1810.07052 | Shallow-Deep Networks: Understanding and Mitigating ... | Kaya | 2018 | abs |
| 2112.00826 | Inducing Causal Structure for Interpretable Neural Networks | Geiger | 2021 | abs |
| 2407.14494 | InterpBench: Semi-Synthetic Transformers for Evaluating ... | Gupta | 2024 | abs |
| 2303.02536 | Finding Alignments Between Interpretable Causal ... | Geiger | 2023 | abs |
| 2112.02505 | Causal Distillation for Language Models | Wu | 2021 | abs |
| 2404.18988 | Markovian Transformers for Informative Language Modeling | Viteri | 2024 | abs |
| 1511.06349 | Generating Sentences from a Continuous Space | Bowman | 2015 | abs |
| 2305.15408 | Towards Revealing the Mystery behind Chain of Thought: A ... | Feng | 2023 | abs |
| 2207.00729 | The Parallelism Tradeoff: Limitations of Log-Precision ... | Merrill | 2022 | abs |
| 2310.07923 | The Expressive Power of Transformers with Chain of Thought | Merrill | 2023 | abs |
| 2402.12875 | Chain of Thought Empowers Transformers to Solve ... | Li | 2024 | abs |
| 2503.03961 | A Little Depth Goes a Long Way: The Expressive Power of ... | Merrill | 2025 | abs |
| 2402.09268 | Transformers, parallel computation, and logarithmic depth | Sanford | 2024 | abs |
| 2502.02393 | Lower Bounds for Chain-of-Thought Reasoning in ... | Amiri | 2025 | abs |
| 2402.08164 | On Limitations of the Transformer Architecture | Peng | 2024 | abs |
| 2404.02258 | Mixture-of-Depths: Dynamically allocating compute in ... | Raposo | 2024 | abs |
| 1610.09027 | Scaling Memory-Augmented Neural Networks with Sparse ... | Rae | 2016 | abs |
| 2607.02907 | ProLaViT: Learning Progressive Latent Visual Thoughts in ... | Li | 2026 | abs |
| 2505.11484 | SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought ... | Xu | 2025 | abs |

## Per-family notes

### (a) Looped / recurrent depth
- **Saunshi 2502.17416 (abs):** a k-layer model looped L times "nearly matches" a kL-layer model on addition, p-hop and math. Theory: many such problems are iterative, so looped models reach near-optimal depth; looped models simulate CoT. [THEOREM + EMPIRICS; synthetic + LM pretraining.]
- **Huginn 2502.05171 (full):** prelude / core / coda. The core input is "an adapter A: R^{2h}→R^h mapping the concatenation of s_i and e" (the input is re-injected every step). Training details:
  - r ~ log-normal Poisson;
  - backprop through "only the last k" iterations (k = 8).
  - **Failure:** in Bad Run 2, "validation perplexity is the same whether 1 or 32 recurrences are used … the model has learned early to ignore the incoming state s". Fixed by the sandwich norm and a 10× lower LR. [EMPIRICS; 3.5B pretraining.]
- **Ouro 2510.25741 (full):** Stage I loss is Σ_t p_φ(t|x) L^(t) − β H(p_φ), which is KL to a uniform prior over exits.
  - Without the entropy term, "this self-reinforcement collapses p_φ onto t = T_max". Geometric priors show "premature collapse … onto shallow steps".
  - Reusing only the first step's KV gives GSM8K 78.92 → 18.73. [EMPIRICS; 1.4B / 2.6B, 7.7T tokens.]
- **Fan 2409.15647 (abs; project notes):** input injection every loop; the no-injection ablation is worse. **Bansal 2202.05826 (abs):** recall + progressive loss, against "overthinking". **2604.11791 (full):** without injection, pre-norm reaches a "'degenerate'" fixed point (every layer converges to the same point), and injection helps the final recurrence "to the detriment of earlier recurrences". [EMPIRICS; small random-init.]
- **Retrofits.** McLeish 2511.07384 (abs): recurrence curriculum; beats post-training the base at equal compute.
  - **Shapiro 2608.11233 (abs):** "identity-preserving one-loop path" + re-entry bridge. "With intermediate-step supervision, the model computes one task step per loop and persists when only final answers are graded". Extrapolates ≈1.5× the supervised depth. [EMPIRICS; Qwen2.5-0.5B.]
  - **ETD 2510.07358 (abs):** iterates selected middle layers of OLMo-2 1B during mid-training.
  - **Penelope 2607.25915 (full):** lower prefix run once; recurrence on blocks [11,16) of Llama-3.2-1B, but "memory queries attend[ing] to cached prompt keys and values" every step. So it is not question-once.
- **Input bypass (abs; repo note LIT_INPUT_BYPASS_LOOPED):**
  - 2609.01924: "Huginn carries content forward across all sixteen recurrences, while reads, writes, and ablations act only within a sliding window of roughly two recurrences"; "Ouro reconstructs workspace content in every loop".
  - SCSE 2607.27656 and OrthoInj 2610.06833: injection acts as a forcing term on every step.
- **Diminishing / negative loops:**
  - **TaH2 2609.35748 (full):** after the extra iterations, "21.7% and 15.9% [of tokens] become worse" (Ouro / Huginn).
  - **TaH 2511.08577:** "latent overthinking" (correct after pass 1, revised to wrong).
  - **LoopCoder-v2 2606.18023 (abs):** loops ≥ 3 regress; later loops give "reduced representational diversity".
  - **2601.10242 (full):** injected concepts are noticed "only when the concept is injected in the final loop". [EMPIRICS.]
- **UT 1807.03819 (full):** adds fixed (position, time-step) coordinate embeddings P^t every step, an explicit pass index (the route our per-pass targets were gamed through).

### (b) Memory / latent slots
- **Coconut 2412.06769, CODI 2502.21074 (abs):** continuous thoughts in context with the question visible. CODI self-distils by aligning hidden states at one token (exact layer / norm pending).
- **Failure evidence:**
  - 2512.21711: Coconut tokens "show minimal sensitivity to steering" and "promote shortcut usage".
  - 2602.00449 (CODI): "the final input follows a separate near-direct route". At depth it "does not reliably sustain a full latent rollout".
  - SIM-CoT 2509.20317: more latents → "latent representations becoming homogeneous"; the fix is a per-step aux decoder.
  - 2604.06374: in the fine-tuned regime, superposition "collapses or is not used". [EMPIRICS; GPT-2 / Llama.]
- **LOTUS 2606.31779 (abs; repo THEORY_LOTUS_CONTRAST):** K latent blocks for R iterations. The abstract says the CE is "on each latent position's gold CoT-step token"; the repo derivation says the inputs are teacherless (learnable ⟨lat⟩ + h^(t−1)). It closes the gap to CoT at 3B.
- **Thinking States 2602.08332 (abs):** thoughts are generated every few input tokens, re-embedded, "and add[ed] to the following input tokens". NL-supervised with teacher forcing.
- **2510.14095 (abs):** input-adaptive recurrence + algorithmic supervision + "anchored latent representations via a discrete bottleneck" + error correction, on GSM8K-style modular arithmetic over graphs. This is the closest design to per-pass re-coding (discrete re-embedding each step).
- **RMT 2207.06881, TTM 2211.09119 (abs):** memory tokens are re-produced by the model at each segment / step, so content survives only if re-written. **Expire-Span 2105.06548:** a learned expiry. **Block-Recurrent 2203.07852:** LSTM-style gated state. **Memorizing 2203.08913:** non-differentiable kNN over cached (k, v). **Feedback 2002.09402:** "exposes all previous representations to all future representations".
- **Pause 2310.02226 / filler 2404.15758 (abs):** extra tokens help, but the filler case "is difficult and requires specific, dense supervision to converge". Expressivity is characterised by quantifier depth. [THEOREM + EMPIRICS.]

### (c) Per-iteration progress / halting
- **ACT 1603.08983, PonderNet 2107.05407 (abs):** a ponder cost / a KL to a geometric prior. RecurTrace reports both "collapse to one loop" on MathQA (abs).
- **Deep supervision (DSN 1409.5185):** final-answer CE at every step. That's also what HRM 2506.21734 / TRM 2510.04871 use: answer CE per supervision step. Early-exit losses (LayerSkip 2404.16710, CALM 2207.07061, SDN 1810.07052 "overthinking") reward answering early, the opposite of M4.
- **Gain-based signals (all for halting only):**
  - Ouro Stage II: "I_i^(t) = max(0, L_stop^(t−1) − L_stop^(t))"; label w = σ(k(I − γ)); BCE on the gate; "we freeze the LM".
  - TaH2: "δ_t^(m) = ℓ_t^(m) − ℓ_t^(m+1)"; labels keep the top-ρ share of positive gains; the backbone gets only the stopping-weighted next-token CE (eq. 7, α_D = 0.05).
  - RecurTrace: oracle "when additional depth still reduces loss".
  - 2607.20519 (abs): joint gate training damages "the trajectory" itself; post-hoc confidence readouts match learned gates.
- **Text RL analogues (abs):** PAV 2410.08146 rewards "progress" (a change in success probability under a prover); MRT 2503.07572 adds a dense progress bonus; SCoRe 2409.12917 adds an improvement-shaping bonus across attempts. RLTT 2602.10520 spreads reward over Ouro's latent loop trajectory. Kuo 2606.29983: stochastic stopping for extrapolation.
- **Path independence:** Anil 2211.09961.

### (d) Causal-abstraction training
- **IIT 2112.00826** (repo full-text note): the loss is zero ⇒ causal abstraction [THEOREM]. Tasks included a CNN-Bi-LSTM on ReaSCAN.
- **SIIT (InterpBench 2407.14494):** also intervenes on non-aligned nodes and trains them to have no effect, because "IIT never intervenes on low-level nodes that are not aligned".
- **DAS 2303.02536:** learned rotation for distributed interchange.
- **Causal distillation 2112.02505.**
- **Across passes / recurrent iterations as a training objective:** NOT FOUND (search, plus the repo note LIT_IIT_TRAINING). The analysis-only work is 2602.08783 (do-interventions on Coconut / CODI steps: "staged functionality with non-local routing").

### (e) Forcing intermediate use; depth bounds
- **Markovian Transformers 2404.18988:** the answer is predicted from a bounded CoT alone (text). Word dropout 1511.06349: decoder-input dropout forces latent use (VAE). Answer-from-latent-only on GSM8K: not found (repo LIT_ANSWER_BOTTLENECK).
- **Lower bounds (abs):**
  - **Feng 2305.15408:** bounded-depth transformers can't solve arithmetic / linear equations unless size grows super-polynomially; constant-size models can with CoT. [THEOREM; log-precision, TC0 ≠ NC1.]
  - Merrill & Sabharwal 2207.00729 (log-precision ⊆ TC0), 2310.07923 (CoT length vs power), 2503.03961 (log depth suffices for regular languages / connectivity).
  - Li 2402.12875 (constant depth + T CoT steps = serial circuits); Sanford 2402.09268 (k-hop needs log k depth); Amiri 2502.02393 (CoT length lower bounds); Peng 2402.08164 (composition); Liu 2210.10749 (shallow shortcuts to automata).
  - **Caveat [DERIVATION]:** a chain of affine updates mod p is a word problem over a solvable group (in TC0). Only general +/× formulas give NC1-hardness. All the bounds are asymptotic.

### (f) Pass-global attention memory (user addition)
- **Closest LM instance:** Coconut / CODI (each thought's KV is visible to all later thoughts and the answer). AGCLR 2606.07720 (GPT-2, GSM8K / ProsQA) adds a persistent write / read / forget memory across passes because "intermediate hidden states are overwritten".
- **Cross-loop KV:**
  - TaH duo-causal attention (iteration-depth KV).
  - RecurTrace: "each looped layer attend[s] to its own states from previous iterations along the loop-time axis". MathQA 56.9% at 2.0 loops; 0.6–8B.
  - Ouro: a per-loop KV cache.
  - MELT 2605.07721: one gated KV shared across loops.
  - LLA 2607.15456: the loop-axis KV is low-rank "but not safely collapsible to a single state".
  - RLT 2610.07591: KV replay.
  - MoR 2507.10524: recursion-wise caching vs sharing.
  - InfiLoop 2610.11570: content weighting + temporal decay over all past loop states, after "noisy state updates overwrite correct intermediate deductions".
- **Depth-dense within a pass:** DenseFormer 2402.02622 ("strong and structured reuse of activations from distant layers"), MUDDFormer 2502.12170, Hyper-Connections 2409.19606, LIMe 2502.09245 (heads read the KV of earlier layers; the single-state residual "creates pressure toward representation collapse").
- **Failure modes:**
  - **Early slots dominating:** attention sinks 2309.17453 (abs); the first-loop states dominating isn't directly shown (pending the RecurTrace / TaH head analyses).
  - **Later passes copying / homogenising:** SIM-CoT, LoopCoder-v2 diversity loss, and 2601.10242 (final-loop-only).

## Novelty scoring
| | closest prior art | what would be new (pretrained LM) | known failure / risk |
|---|---|---|---|
| M1 | Shapiro, McLeish, ETD, Penelope, Huginn; Saunshi | no input route after pass 1 in a pretrained retrofit | no-recall overthinking / forgetting (Bansal, Fan); degenerate fixed point (2604.11791); §165 = 0.0; Feng's bound is asymptotic only |
| M2 | Coconut / CODI, TaH, RecurTrace, Ouro KV; SIM-CoT / LOTUS step terms | little | [DERIVATION] an append-only global memory puts the pass-1 slots in the decoder's reach, so precompute + ignore is still optimal |
| M3 | RMT / TTM, Expire-Span, BRT gates, InfiLoop, AGCLR, 2510.14095 discrete re-embedding | rotation as a forcing device (not found) | [DERIVATION] a rotation R_t known to the model (via pass index) is invertible: it forces nothing. Write-to-keep under attention is a one-head copy |
| M4 | Ouro Stage II, TaH2, RecurTrace (gates only); PAV / MRT / SCoRe (text) | gain loss on the backbone | [DERIVATION] the margin max(0, m − (log p_K − log p_k)) falls by worsening k. With k detached it is just a reweighted CE at K. Pair it with a frozen / EMA k-baseline |
| M5 | IIT (incl. LSTM), SIIT, causal distillation, DAS | IIT over pass boundaries on GSM8K (equation-chain counterfactuals) | un-aligned paths (SIIT); precompute passes whole-state swaps unless post-swap ops differ (repo rule) |
| M6 | word dropout, Markovian, answer bottleneck (not found for latents) | masking or perturbing q for passes ≥ 2, with an invariance target | a lossy pass-1 copy caps accuracy (QO draft's QO-CAPACITY-BOUND); the model copies q into memory (a "bag of literals") |
| M7 | RecurTrace + MoD / MoR budgets + sparse R/W; ProLaViT / SoftCoT++ diversity | the combination, with a pass-level non-redundancy reward in a looped LM | non-redundancy is met with noise or irrelevant distinct content (the repo's dynamics-Goodhart lesson). Tie it to new decodable task information, e.g. conditional probe gain I(slot_t; target \| slots<t) |

## IDs that failed verification
- **None of the IDs given failed.** Every one matched the claimed paper.
- **Naming notes:**
  - 2002.09402 is titled "Addressing Some Limitations of Transformers with Feedback Memory" (Angela Fan).
  - 2602.08332's first author is Amos; title "Latent Reasoning with Supervised Thinking States".
  - 2510.14095 is Altabaa et al., "Unlocking OOD Generalization … Recursive Latent Space Reasoning".
  - LOTUS = 2606.31779 (Fan, Svete, Lee).
  - SIIT = InterpBench 2407.14494 (Gupta et al.).
  - Fan's length-generalisation paper = 2409.15647.
- **Not found:** IIT across passes as a training objective; question-once retrofits; rotation as a forcing device; a backbone-side marginal-pass loss.
