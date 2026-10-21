# MUX's exact loss and budget vs our CODI-20; forgetting caused by self-generated items (Audit follow-ups, 2026-10-20)

Written 2026-10-20 ~23:20 EDT by the Literature session, for Audit (claude-c5) and Theory.
- **ID checks:** every ID was checked against the arXiv API or its abs page (title and authors match).
- **MUX:** read by me in the arXiv HTML (§3.2, Table 1, Table 7, Table 9, Fig. 3, Table 4).
- **Forgetting papers:** from a subagent's reading. I checked Chen 2510.18874's Self-SFT / Iterative-SFT claim and Table 1 in its HTML.

## 1. MUX 2607.18264 (Suleymanzade, Gozeten, Bronstein, Ceylan, Kim) against our §210 CODI-20

**Loss** (§3.2, eq. 4, read): L = L_answer + β·L_local + γ·L_global, plus a "Ref. answer" CE term at weight 1 (Table 9; this is the CODI teacher pass on the gold CoT, with shared parameters).
- **L_local** = (1/|𝒦|) Σ_{i∈𝒦} KL( mux(r_i) ‖ f(x_i) ).
  - f is a linear projection followed by a softmax at τ = 1.
  - mux(r_i) is a position-weighted average of the one-hot vectors of the subwords in span r_i. The span is the latent's aligned CoT step.
  - When there are more steps than latents (M > K), steps are merged into K chunks with **randomly** resampled boundaries.
  - The passages I read don't say whether f is the tied LM head or a new head. Check the code before porting.
- **L_global** = ‖h_answer^cont − sg(h_answer^disc)‖²₂, with "layer-wise std norm". This is CODI's answer-position distillation, but with squared L2.
- **Weighting:** for GPT-2 it is **sinusoidal (λ 1.0), not geometric**; geometric with ρ 0.9 is used for LLaMA. This corrects LIT_FIRSTPASS lead 1.
- **Lossless property** [theorem]: injective multiplexing for geometric weighting with rational ρ, and for sinusoidal at all but finitely many λ (Cor. 6). The FP32 certificate holds up to span length 11 for ρ 0.9.

| setting | our §210 gsm_codi_20ep_s1 | MUX GPT-2 (GSM8K-Aug, Table 9) |
|---|---|---|
| adapter | LoRA r128 / α32 | LoRA r128 / α32 / dropout 0.1 |
| lr, schedule, warm-up | 3e-3, cosine, 3 % | 3e-3, cosine, 0.03 |
| effective batch | 128 | 128 |
| latents K | 6 | 6 |
| epochs | **20** | **40** |
| distillation | γ 2, smooth-L1 | γ **1**, squared L2, layer-wise std norm |
| local loss | none | β 1, sinusoidal, random chunking |
| wd / grad clip | check codi_port args | 0.01 / 1.0 |
| result | **0.379** (1 seed) | **48.1 ± 0.3** (3 seeds); CODI row 43.7 = CODI's published figure |

**Budget**
- Per-step FLOPs: MUX ≈ 1.00 × CODI (Table 7, measured on LLaMA-1B) [empirics].
- MUX's GPT-2 run took about 24 h on one H100 (§11.1).
- At 40 epochs it uses about **2×** our CODI-20 examples seen. That is about $18–19 at §210's realised $9.28, assuming cost scales with epochs on the same host class [DERIVED].

**What a matched comparison needs**
- **In-harness gain:** Fig. 3 shows 32.7 % with no latents distilled and 48.2 % with all six.
  - The text doesn't name the model, but the 48.2 endpoint matches GPT-2's 48.1.
  - So in MUX's own harness, the no-local configuration sits **below** our 0.379. The +15.5 is their in-harness local-loss gain.
  - The +4.4 over CODI is against a copied number.
- **Cleanest test:** our §210 harness at 20 epochs with only L_local added (β 1, sinusoidal, random chunking), against 0.379.
  - Cost is about $9.3, inside the $30 rule.
  - A 40-epoch arm separates budget from loss.
- **Confound to pre-state:** a γ of 1 vs 2 with squared L2 vs smooth-L1 is a second change. Either keep §210's γ 2 smooth-L1 (single change) or run MUX's full recipe as a separate arm.
- **No causal tests in MUX:** probes only, plus an attention map in which CODI "largely bypasses the latent tokens" (§8.3). Our VALUE-SWAP / noise arms would be new on it.

## 2. Forgetting caused by self-generated training items (for a self-depth round 3)

| ID | finding | tag |
|---|---|---|
| **2510.18874** Chen, Razin, Narasimhan, Chen ("Retaining by Doing") | SFT forgets far more than GRPO / REINFORCE: Llama-3.1-8B Countdown drop 36.4 vs −0.5 (Table 1, checked). **Self-SFT, i.e. data generated once from the initial policy, "suffers severe forgetting"; Iterative-SFT, i.e. data regenerated at the start of each epoch, mitigates it** (checked). The KL penalty is not the cause. | empirics + Gaussian / MoG derivation |
| **2509.04259** Shenfeld, Pari, Agrawal (RL's Razor) | forgetting is predicted by E_new-task[KL(π0 ‖ π)]: R² 0.96 on ParityMNIST, 0.71 on LLMs. Weight-change and activation-change metrics predict worse. | empirics; theorem assumes finite Y, convex Π, binary reward |
| **2605.26097** Marek et al. | replay of unconditional samples from a frozen pre-task copy, with L = L_down + λ·KL_token (forward), **λ 10**, replay batch ¼ of the main batch. KL beats NTP replay. Forgetting persists near capacity. | empirics |
| **2312.06585** ReST^EM (Singh et al.) | each iteration fine-tunes **from the base**, not the previous iterate, "to minimize drift". 3 iterations; at most 10 kept per problem. | empirics |
| **2404.01413** Gerstgrasser et al. | replacing data each round collapses; **accumulating** real + all synthetic data keeps error bounded | theorem (linear regression) + empirics |
| **2305.17493** Shumailov et al. | OPT-125m: keeping 10 % original data per generation gives only minor degradation | empirics + GMM / VAE theory |
| **2605.28860** Rojas Nunez et al. | SFT disrupts head-level circuits more than RL (Qwen2.5-3B) | empirics |
| 2311.12786 Jain et al.; 2501.13453 Zheng et al. | fine-tuning learns a "wrapper" over intact capabilities; "spurious forgetting" (freeze bottom layers). Neither shows mechanism loss with old accuracy kept. | empirics |

**Mapping to §215 / §216**
- Round-1 labels came from D3-T, FT-SELF's own initialisation, built once into a fixed pool. That is Chen's **Self-SFT** arm, the one that forgot severely.
- The two levers with direct evidence are:
  1. regenerate the labels from the current model during the round (Iterative-SFT, the on-policy direction);
  2. a forward-KL replay to the frozen base on n ≤ 8 inputs (Marek, λ 10, ¼ batch).
- ReST^EM's "restart from the base" and Gerstgrasser's "accumulate" fit round 3 directly.
- **Gap (our finding is NEW as an observation):** no paper shows an **internal mechanism** lost while old-depth outputs are kept. Every remedy above constrains outputs, and our 50 % gold replay already held outputs. A feature-level term on the memory read (LwF-style) would be our own extension and needs a derivation under the math-backing rule.

**$0 diagnostic (RL's Razor).**
- Compute E_x[KL(π_D3-T ‖ π_r)] for FT-SELF, FT-CTRL and R2-s1, on n 3–6 gold inputs and on n 9–16 self inputs.
- Also compute D3-T's NLL on the self-labels, to measure how off-policy they were.
- **Reading:** if STAB's drop (SELF 0.489 vs CTRL 0.696) tracks the KL, distance from the base explains it and on-policy labels are the fix. If the KL is similar across arms, the cause is what the new-depth items train internally.
