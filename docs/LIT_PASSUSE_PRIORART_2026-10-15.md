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

## Full-text addendum A (2026-10-15; helper subagent; arXiv HTML full text read for every paper; extracted texts kept in the Theory scratchpad research/fullA/; titles and first authors match)

1. **Saunshi 2502.17416.**
   - [THEOREM] Thm 5.1: n-element group composition needs only "a 1-layer transformer block, T=⌈log2 n⌉ loops".
   - [THEOREM] Thm 5.2: an L-layer model with at most R distinct layers is simulated by "loops a 1-layer transformer block for L times" (embedding d+R+2).
   - [THEOREM] Cor 5.3: p-hop needs ⌊log2 p⌋+2 loops.
   - [THEOREM] Thm 5.4: m CoT steps of an L-layer model are matched by a looped model with L+O(1) layers and Ω(log(n+m)) more embedding dimension, run with "m dummy tokens with m loops".
   - [DERIVATION] Regulariser: R_G(k) = (1/(L−k)) Σ cosine between corresponding layers of consecutive k-blocks. The loss is L_xent + λ_reg|G|⁻¹ Σ_G R_G(k), with k=4 and λ_reg=10.
   - [EMPIRICS] Synthetic tasks (n-ary addition, p-hop, i-GSM): a looped k-layer model ≈ a kL-layer model. In LM pretraining (1B, from scratch, 250B Pile tokens), perplexity is worse but reasoning better. **No pretrained retrofit.**
2. **Fan 2409.15647.**
   - [DERIVATION] Injection is additive ("input embeddings are added to the output embeddings of the previous step"). The loss is CE only after T_i iterations, with no intermediate supervision; T(n)=n for parity and copy, n+1 for addition.
   - [DERIVATION] Test-time stop: by oracle, or by maximum confidence T = argmin_{t≤T_max} L(f_t, ŷ^t) against the model's own greedy decode.
   - [EMPIRICS] No-injection ablation: injection is "generally better", qualitatively (Fig 7). NoPE GPT-2-style block, from scratch.
3. **Bansal 2202.05826.**
   - [DERIVATION] Recall: "the input to r at iteration k is [φ_{k−1}, x]".
   - [DERIVATION] Progressive loss (Alg 1): n∼U{0,m−1}, k∼U{1,m−n}; "compute φ_n with n iterations w/o tracking gradients", then k more iterations with gradient, with the loss at n+k. L = (1−α)L_max_iters + αL_progressive; α = 1 (prefix sums), 0.01 (mazes), 0.5 (chess); m = 30.
   - [EMPIRICS] "without recall all models fail to extrapolate" (59×59 mazes); without recall, accuracy shows a "sharp decline" with more iterations. Small convnets, from scratch.
4. **McLeish 2511.07384.**
   - [DERIVATION] Prelude = early layers; recurrent block and coda = later layers; the middle layers are removed (TinyLlama (4,8,4) = [0-3], [10-17], [18-21]).
   - [DERIVATION] s_0∼N(0,σ²); s_i = R(e, s_{i−1}), where R begins with a linear adapter on concat(s, e) (2h→h). Recurrences are sampled Poisson-Lognormal, with backprop through at most the last 8 passes and a curriculum ramping the mean.
   - [EMPIRICS] No analysis of later-recurrence use. Table 6 GSM8K plateaus at the training mean (4): 25.6 / 44.1 / 51.2 / 51.8 / 51.9 / 52.0 at 1 / 2 / 4 / 8 / 16 / 32.
5. **Shapiro 2608.11233** (Qwen2.5-0.5B, split at layers 6 and 18).
   - [DERIVATION] Bridge: work + gate·(prelude_proj(p) + state_proj(work) − work), with an identity-biased gate; "the Prelude contribution is applied only on re-entry".
   - [EMPIRICS] An earlier zero-init gate "self-locked" (all gate and bridge gradients exactly zero).
   - [DERIVATION] Per-loop CE against the registered intermediate target, then 1,000 steps of final-answer-only training.
   - [EMPIRICS] Afterwards 625/640 intermediate states are still correct, and 93.0% of extra loops continue the chain ("essentially never parked"). PonderNet "collapsed to always choosing the maximum depth". Synthetic chain task.
6. **Wang & Reid 2609.01924** (inference-time hooks on Huginn-0125, 3.5B, 16 recurrences).
   - [EMPIRICS] Confirmed. §1: "continuous recomputation from the re-injected input, not a memory buffer".
   - [EMPIRICS] §5 injection swap from recurrence k onward "flips the final output to the new input's own answer in all 30 configurations, including k=12"; the old answer decays within about 2 recurrences, "with a ∼2-recurrence memory".
7. **RecurTrace 2609.03379** (frozen Qwen3 0.6B–8B, post hoc).
   - [DERIVATION] Loop Memory Attention: h ← h + α·RMSNorm(e) (α=0 at init); h ← h + LMA_ℓ(x_ℓ^(t−1), M_ℓ^(t)), where M_ℓ^(t) = layer ℓ's own outputs from up to W=3 earlier loops, at the same position only. Weights softmax(QᵀK/√d − β_h(t−t_j)).
   - [DERIVATION] Halting oracle: y_t = 1[min_{t′>t} L_{t′} < L_t − δ], δ = 0.01.
   - [EMPIRICS] ACT / PonderNet: "drives both methods to stop after one loop" (mean depth 1.0 in all 6 settings).
   - [EMPIRICS] Removing loop memory (plain looping) raises NLL by 0.07 / 0.05 / 0.06 nats. Beyond the training depth, memory adds +17.7 (state tracking) and +16.7 (arithmetic), and costs −4.7 on pointer chasing. Which loops are attended is not analysed.
8. **Think-at-Hard 2511.08577.**
   - [DERIVATION] Duo-causal attention over all positions ≤ i and depths ≤ d (KV concatenated per depth).
   - [EMPIRICS] Heads attend first-iteration, second-iteration or both keys; deeper layers show more cross-depth reuse. "Latent overthinking": most predictions are correct after pass 1 but sometimes revised to errors. Ouro-1.7B: always-1 73.1, always-2 79.7, oracle 81.8.
9. **TaH2 2609.35748** (Qwen3-1.7B).
   - [DERIVATION] Token t attends to KV at positions ≤ t and depths ≤ m.
   - [EMPIRICS] Extra iterations make 21.7% (Ouro) and 15.9% (Huginn) of tokens worse.
10. **RLTT 2602.10520** (Ouro-1.4B / 2.6B-Thinking).
    - [DERIVATION] GRPO's log P^(T_max) is replaced by Σ_t ω_t ∇log P^(t)·Â_i, with the same advantage for every loop; ω is the exit PDF, progressive t^α/Σs^α, or uniform.
    - [EMPIRICS] The weightings differ "consistently modest[ly]".
    - [THEOREM] Thm A.5: RLTT's optimal length ≤ GRPO's, assuming concave S(L) and c_RLTT ≥ c_GRPO.
11. **Popescu 2607.20519.**
    - [DERIVATION] Joint objective Σ_t q_t(x)ℓ_t + βKL(q‖π).
    - [EMPIRICS] Joint gate training gives "accuracy at early loops is often lower", with gains concentrated late. Refitting the gate on the frozen trajectory "does not improve performance": "the main issue is the trajectory induced by the joint objective". A large β collapses to near-fixed depth; a geometric prior with λ ≥ 0.5 puts almost all weight on loop 1 and degrades later states.
    - **Setting: MANO (integers mod 23, prefix notation, up to 10 operations; a task close to our nested one)**, a 4-layer 512-d model, T = 6, from scratch; also Ouro.
12. **InfiLoop 2610.11570.**
    - [EMPIRICS] Overwrite failure: TRM (newest state only) peaks at 62.2% at 468 layers and falls to 58.1% at 1,872. "TRM turns 20.9% of these predictions into an incorrect value at least once, and InfiLoop turns 2.4%."
    - [DERIVATION] h_t = Σ_i α_{t,i} z_i with α ∝ exp(s_i − λ(t−i)), content score s_i = qᵀRMSNorm(z_i) (a shared pseudo-query, scored once), λ learned. It has an exact constant-memory recurrence.
    - [EMPIRICS] The step size γ is about 0.40 before solving and "surges to 0.91" at the solving step.
13. **Kuo 2606.29983.**
    - [DERIVATION] L = E_{K∼π_stop(·|x)} ℓ(θ; x, y, K). Hand schedule K = clip(n+Δ, 1, T_max); RL-halting with REINFORCE, a baseline and entropy.
    - [EMPIRICS] Run-to-run OOD std 0.087 (K=20) and 0.122 (K=n), vs ≤ 0.009 for non-looped models, traced to "the spurious correlation ... between sequence length and number of loops". It "can also stabilize a suboptimal computation". 3-layer block, from scratch.

**Not stated in the papers:**
- (4) whether later recurrences are used (only the Table 6 plateau);
- (5) any loop-bypass / T=1-dominance finding;
- (7) which loop slots the memory attends to;
- (8) a number for how often predictions are revised to wrong (Fig 8 only).

## Full-text addendum B (2026-10-15; helper subagent; 20 papers read in full from the arXiv HTML; local copies in the Theory scratchpad research/fullB/)

**Latent-reasoning LMs**

1. **Coconut 2412.06769 (Hao; pretrained GPT-2).**
   - [DERIVATION] The question stays in context for thoughts and answer: E_t = [e(x_1..x_i), h_i, …, h_{t−1}].
   - [DERIVATION] Stage k replaces the first k CoT steps with k×c thoughts (optimizer reset at each stage). c = 1 for ProsQA / ProntoQA; c = 2 for GSM8K, plus a final stage that removes all language.
   - [DERIVATION] Loss: CE on the remaining tokens after the thoughts. "The objective does not encourage the continuous thought to compress the removed language thought."
   - [EMPIRICS] Table 1, ProsQA: Coconut 97.0, pause 96.6, w/o thought 95.5.
   - [EMPIRICS] Table 1, GSM8K: Coconut 34.1, pause 24.1, w/o thought 21.6, No-CoT 16.5, CoT 42.9.
   - [EMPIRICS] Without the curriculum: 14.4. c = 3 gives a loss spike.
2. **CODI 2502.21074 (Shen; GPT-2 and LLaMA-3.2-1B; LoRA r128; GSM8k-Aug).**
   - [DERIVATION] L = αL_student + βL_KD + γL_teacher.
     - L_teacher: CE on [CoT, y] | Q.
     - L_student: answer CE | Q, Z. The question is visible.
     - L_KD (Eq. 5): L1 at the ":" of "The answer is:", all layers, sg on the teacher, normalised per layer by the teacher's batch std.
   - [DERIVATION] Setup details:
     - the teacher drops the final CoT step (keeping it: 31.7 vs 43.7);
     - a 2-layer MLP + LN projection; 6 latents;
     - α = β = 1; γ = 1 (GPT-2), γ = 20 (LLaMA).
   - **Inconsistency:** Appendix F sweeps β 1 → 30 on LLaMA (46.5 → 51.9%).
   - [EMPIRICS] GPT-2 ablations: w/o L1 24.5 (No-CoT-SFT 19.1); separate static teacher 27.1.
3. **LOTUS 2606.31779 (Fan, Svete, Lee; GPT-2 and Llama-3.2-1B / 3B-Instruct; initialised from explicit-CoT checkpoints; full fine-tune).**
   - [DERIVATION] The latents are learnable (shared <lat> token, learnable E ∈ R^{Kc×d}) and never teacher-forced.
   - [DERIVATION] Loop: h^(0) = f(E | C_pre), h^(t) = f(E + h^(t−1) | C_pre), where **C_pre is the KV cache of [Q, <BoT>]** (the question stays readable). Gradients flow through all R iterations.
   - [DERIVATION] Losses:
     - L_step = (1/N_step) Σ CE(f_head(h^(R)_{i,j}), T_{i,j}), a **post-loop** step readout through the base LM head;
     - L_ans is a separate forward over [<EoT>, A] attending to [C_pre, h^(R)];
     - L = L_ans + 0.05 L_step; K = 6, c = 25 (Llama) or 13 (GPT-2), R = 6.
   - [EMPIRICS] 3B GSM8K: LOTUS 70.0 ± 0.9; CoT 71.5; CODI 60.8; CODI+SIM-CoT 62.3; No-CoT 38.3.
   - [EMPIRICS] 3B supervision ablations:
     - no latent supervision: 63.3;
     - per-iteration supervision: 68.2, vs post-loop 70.0;
     - LOTUS-aux: 69.9 per-iteration vs 68.4 post-loop;
     - c = 1: 49.7.
   - [EMPIRICS] **3B loop-count ablations, showing the iterations are used:**
     - trained R: 14.6 (R = 2), 68.1 (R = 5), 70.0 (R = 6);
     - **inference R on the R = 6 model: 22.7 (R = 1) → 70.0 (R = 6) → 69.3 (R = 7)**.
     - No zero or shuffle test on h^(R).
4. **Thinking States 2602.08332 (Amos; Qwen2.5-Base 0.5B / 1.5B).**
   - [DERIVATION] Mechanism:
     - the input is chunked; X̃_i = X_i + S_i after layer 1;
     - a 1-layer Thinking Block generates natural-language reasoning (or <EOS>);
     - a Compression Block (a copy of layer 1) keeps the last c representations as S_{i+1};
     - thoughts are never appended to the context.
   - [DERIVATION] Fully teacher-forced (S_i* = C(Z_i*)); no BPTT.
   - [EMPIRICS] GSM 42.22 vs CoT 60.50, Coconut 32.65. Parity (trained N = 10): 98.37 vs CoT 12.35.
   - [EMPIRICS] Bypass is not tested. "State ambiguity": prepending the question raises 42.22 → 48.65.
5. **Altabaa 2510.14095 (from scratch; DAG modular arithmetic p = 23; train N ≤ 32, test up to 128).**
   - [DERIVATION] T scales linearly with graph depth. Per-step targets: Σ_t Σ_i 1{Depth(x_i) ≤ t} ℓ(W E_i^(t), Value(x_i)).
   - [DERIVATION] After each iteration the state is projected to a discrete factored symbolic space and re-embedded (allowing teacher forcing across iterations). Error correction: random corruption of value components; needs a 4-layer block.
   - [EMPIRICS] Continuous supervision "slowly degrades" out of distribution; discrete + correction is near-perfect on every OOD split; CoT reaches only N ≈ 40.
6. **SIM-CoT 2509.20317.**
   - [EMPIRICS] Coconut collapses at 5 latents (12.5%); latents become "nearly identical", with content "mainly numbers, lacking operators".
   - [DERIVATION] A training-only step decoder reads z_k alone: L_step = −Σ_k Σ_t log p(y_{k,t} | z_k, y_{k,<t}); L = λ_step L_step + λ_lm L_ans (λ values not found).
   - [EMPIRICS] GPT-2 Coconut+SIM 44.8 (+8.2).
7. **AGCLR 2606.07720 (GPT-2 124M, one seed).**
   - [DERIVATION] Gated persistent memory: h′_t = (1−f)⊙h_t + r⊙c_{t−1}; c_t = LN(c_{t−1} + w⊙h′_t).
   - [EMPIRICS] Cosine to the pass-1 state falls 1.0 → 0.126 by pass 6 (AGCLR about 0.22). This is not a use test, and Coconut keeps earlier thoughts in attention.
   - [EMPIRICS] Results:

     | | CoT | Coconut | AGCLR |
     |---|---|---|---|
     | GSM8K | 40.6 | 31.4 | 34.0 |
     | HotpotQA EM | 11.0 | 10.4 | 14.0 |
     | ProsQA | 55.0 | 92.0 | 96.0 |

**Do latents do the work?**

8. **Zhang 2512.21711 (7–8B, Coconut 5 latents on ProntoQA).**
   - [EMPIRICS] Steering: perturbing CoT tokens gives 50% AdvBench success vs 5% for the latents.
   - [EMPIRICS] MMLU swap: Coconut stays at about 60% vs CoT dropping to 43.4%. "latent tokens exert very limited influence."
   - [EMPIRICS] Shortcut uptake is strong.
9. **Liang 2602.00449 (CODI on s_t = s_{t−1}x_t + b mod m; 3-layer from scratch).**
   - [EMPIRICS] At 2–3 hops bridge states form, but the final input goes straight to the answer: patching latents under an x_{n+1} corruption gives 0% recovery.
   - [EMPIRICS] At n ≥ 4, only the late s_n is decodable; for prime m accuracy drops sharply.
   - [THEOREM 1] "Fully bijective FSC has non-contracting long-horizon dynamics".
10. **Markovian Transformers 2404.18988 (Llama-3.1-8B, Mistral-7B; LoRA).**
    - [DERIVATION] The answer is generated from the CoT alone ("we do not allow the answer generation to attend back to the question").
    - [DERIVATION] R = ln π_θ(ans | CoT) − ln π′(ans | CoT′), with π′ the **frozen baseline**; L = L_PG + L_AR + 0.1 KL, with the advantage standardised per batch.
    - [EMPIRICS] GSM8K 19.6 → 57.1. In the perturbation test, Markovian minus non-Markovian sensitivity grows +0.235 → +0.981 with severity.
11. **Pause tokens 2310.02226.**
    - [EMPIRICS] Gains need pretraining and fine-tuning with delays; fine-tuning-only is "lukewarm" and hurts GSM8K. Zero pauses at inference "breaks its performance spectacularly".
    - [THEOREM 6.1, informal] 2 layers can implement N > L independent computations iff they use pauses.
12. **Filler tokens 2404.15758 (34M, random init).**
    - [EMPIRICS] 3SUM 100%; 2SUM 93.6 vs 78.7 without filler. It needs dense supervision (a 50/50 mix of parallelisable CoT and filler); filler-only training stays at about 71%.
    - [THEOREM] (citing Merrill & Sabharwal): still TC^0 with polynomial filler.

**Memory architectures**

13. **Feedback Transformer 2002.09402.**
    - [DERIVATION] m_t = Σ_l softmax(w^l) x_t^l; every layer attends to past m's.
    - [EMPIRICS] Sequential, so 84.4K vs 296K words/s training. Higher layers work better as memory.
14. **RMT 2207.06881.**
    - [DERIVATION] Memory is fully rewritten (H^mem_{τ+1} := H^write_τ), no gating; BPTT over 0–4 segments.
    - [EMPIRICS] Instability with large memory and deep unrolls.
15. **Token Turing Machines 2211.09119.**
    - [DERIVATION] Write = learned re-selection ("erased if not re-selected"); m = 96.
    - [EMPIRICS] Charades: TTM 26.34, NTM 25.86, memory zeroed 22.65.
16. **Expire-Span 2105.06548.**
    - [DERIVATION] e_i = Lσ(wᵀh_i + b), set once; soft mask; expired memories are deleted (never rewritten); L_task + αΣe_i/T.
17. **Cross-layer reads:**
    - **LIMe 2502.09245:** reads earlier layers' K/V (current-layer queries). Early layers attend to the embeddings. GSM8K fine-tune 0.167 vs 0.140.
    - **DenseFormer 2402.02622:** scalar averaging of earlier outputs. High weight on the initial embedding; later layers give it negative weight.
    - **MUDDFormer 2502.12170:** dynamic per-position weights per Q / K / V / residual stream. The value stream benefits most, and leans on the first layer's output.
    - **Hyper-Connections 2409.19606:** n copies mixed by a learned matrix, in a Λ-shaped pattern. With n = 1 it is worse than baseline.

## Full-text addendum C (2026-10-15; helper subagent; full text read; IDs checked on the abs pages; dumps in the Theory scratchpad research/fullC/)

**Improvement / progress objectives:**
- **SCoRe 2409.12917.**
  - [DERIVATION] Stage I maximises r̂(y₂) − β₂·KL on attempt 1 to the base model (β₂ 0.1 / 0.25). Stage II adds a bonus to attempt 2 only, α·(r̂(y₂) − r̂(y₁)) with α = 10, under γ = 0 (instantaneous reward).
  - [EMPIRICS] Collapse to non-correcting behaviour when α = 1. Attempt 1 is not degraded (52.6 → 60.0), with no discussion of sabotage.
- **PAV 2410.08146.**
  - [DERIVATION] Progress = A^μ = Q^μ(s_h, a_h) − Q^μ(s_{h−1}, a_{h−1}) under a separate prover policy μ; using π's own advantages ≡ outcome reward.
  - [EMPIRICS] With a strong prover, Q^μ-trained policies "only produce re-phrasings of the question".
  - [THEOREM, informal] Thm 3.1.
- **MRT 2503.07572.**
  - [DERIVATION] r_prg(z_j) = J_r(μ(·|z_j, c)) − J_r(μ(·|c)); contexts come from π_old, so the sum does not telescope.
- **Cho 2505.23060.**
  - [EMPIRICS] With a whole-trajectory reward r₂ − r₁ (γ = 0), training maximised the gap "often by deliberately decreasing r₁". The fix is R = (r₂ − r₁) + 0.5·r₁. **Published evidence of the early-attempt sabotage route (our G8).**
- **Other gaming:** ACT's ponder cheat (inflating late-state magnitudes, 1603.08983); PURE 2504.15275 (summation-form PRM credit "induces LLMs to hack steps").
- **No ranking / margin loss enforcing acc(K) > acc(K′ < K) was found.** The closest:
  - Bansal 2202.05826 (progressive loss);
  - Asadulaev 2511.16886 (Deep Improvement Supervision: per-step targets "strictly closer to y*");
  - Ouro Stage II (gate only);
  - RLTT 2602.10520.

**Causal abstraction:**
- [DEFINITION] IIT 2112.00826: Eq. 1 / 4. A zero loss ⇒ causal abstraction (the converse does not hold). The CNN-LSTM use intervenes on an encoder token, not across steps.
- [DEFINITION] SIIT 2407.14494: strictness loss on non-aligned nodes; best W_SIIT ≈ 0.5. Non-aligned nodes can still contribute constantly.
- [DEFINITION] DAS 2303.02536: a soft rotation intervention; the models are frozen (an analysis tool).
- **No IIT across recurrent iterations or latent steps as a training objective was found.** The nearest:
  - Ruiz 2507.02782 (state initialised with noise or another input's final state);
  - type-level IIT 2212.09897;
  - causal distillation 2112.02505;
  - LIFT 2509.13334;
  - counterfactual simulation training 2602.20710.
  - Analysis only: DAS on GRU/LSTM 2501.06141; SCIT 2608.27265 (cache interchange on CODI); 2608.15156.

**Depth / CoT bounds:**
- [THEOREM] Feng 2305.15408:
  - Thm 3.1: assuming TC⁰ ≠ NC¹, for any depth L there is an n such that no log-precision transformer solves Arithmetic(n, p). This is asymptotic.
  - Thm 3.3: CoT with d constant, L = 5 and 5 heads suffices.
  - Each CoT step reduces the leftmost handle (our format).
- [THEOREM] Merrill 2503.03961: looped depth ⌈log₂ n⌉ suffices for regular languages and connectivity. There is no proven matching lower bound (conditional on TC⁰ ≠ NC¹). Θ(log n) is EMPIRICAL.
- [THEOREM] Sanford 2402.09268: hop_k at L = ⌊log₂ k⌋ + 2; L = Ω(log k) under the MPC conjecture.
- [THEOREM] Li 2402.12875: S₅ is in CoT[n, log n, 1] but not in T[poly, log n], assuming TC⁰ ⊊ NC¹.
- [THEOREM] Amiri 2502.02393: UHAT CoT lower bounds Ω(N) (parity, median, …).
- [THEOREM] Peng 2402.08164: one attention layer cannot compose functions if H(d+1)p < n log n.
- **Reading:** no unconditional finite-n depth lower bound applies to GPT-2 at n ≤ 16. "One pass is insufficient" must be shown empirically (K′1).

**Input removal across loops:**
- [EMPIRICS] Recall needed / injection helps: Bansal, Schwarzschild 2106.04537, Yang 2311.12424 ("without input injection … deteriorates beyond the trained loop iterations").
- [EMPIRICS] Marchenko 2610.00673: looped Qwen3 without re-injection is only 0.01–0.02 nats worse; from scratch, concatenation injection was 0.12–0.13 nats *worse* than no injection.
- No input-dropout-across-loops or removal-curriculum work was found.

**Halting / early exit definitions:**
- ACT 1603.08983 (Eq. 10–12); PonderNet 2107.05407 (Eq. 2–3).
- Deeply-Supervised Nets 1409.5185 (the companion term "often goes to zero").
- LayerSkip 2404.16710 (all layers at all iterations "reduces the accuracy of the last layer").
- Shallow-Deep 1810.07052 (overthinking, destructive in up to 50% of errors).
- CALM 2207.07061.

**Rotation / lifetime / bottlenecks:**
- No per-iteration random re-encoding used to force recomputation was found.
- Nearest: Henaff 1602.06662, shuffling RNN 2007.07324, RotLSTM 2105.00357, Expire-Span 2105.06548, Zoneout 1606.01305.
- Bottlenecks: VIB 1612.00410, IBRO 2507.18391, 2606.20075 ("dual collapse"), Token Assorted 2502.03275, Hourglass 2607.11696.
