# Overnight scout: task-agnostic designs where a recurrent / latent model chooses what to read from raw input (2026-10-09)

Literature session, for Audit and Theory (the user's overnight brief: deep ProsQA, nested arithmetic, and new task-agnostic latent
methods).
- **IDs:** all verified via arXiv abstract pages (2026-10-09).
- **Numbers:** from abstracts or HTML where stated.
- **★** = a GSM8K-relevant result at GPT-2 scale.

## 0. Benchmarks first (affects DEEP-ProsQA)
- **ProsQA-Ext, Cheng & Zhang 2609.35643:** shortcut-controlled (a shortcut ≈ 50%), trained on 3–6 hops, OOD on 7–12.
  - From-scratch GPTNeoX 4L / d256. Latent variants beat CoT / pause OOD.
  - Latent-state transplants redirect the answer, and there's a sparse recurrent search circuit.
  - It corrects my DEEP-ProsQA note (84322b5a). **Use it as the DEEP generator / baseline.**
- **Nested arithmetic:** Feng 2305.15408 (fd630b87 facts). Mind the stack depth.

## 1. Pointer / attention-over-input recurrences (a latent state that queries the raw input)
| Design | Mechanism | Relevance |
|---|---|---|
| Perceiver 2103.03206 | a small latent array cross-attends to the raw input repeatedly (weight-shared iterations) | the canonical "latent chooses what to read": a fixed latent size, any input |
| **RIN, Recurrent Interface Networks 2212.11972** | latent tokens alternately *read* from and *write* to an interface of input tokens, with compute concentrated on the latents; adaptive across iterations | closest to "a carrier that reads the input when it wants"; from scratch (diffusion) |
| Pointer Networks 1506.03134; NTM 1410.5401; NPI 1511.06279 | attention outputs indices into the input; content / location addressing; a recurrent controller calls sub-programs | the classic pointer-reading recurrence; algorithmic tasks |
| fast-slow persistent recurrence 2604.01577 | a never-reset latent; T inner updates per observation, with the observation injected | streaming read; OOD length gains (LIT_INJECT) |
| **BDH-CQ 2608.09888** | inputs continuously update a recurrent memory, then iterative latent computation answers a query | "think after reading" with a memory; ARC-AGI-1 29.5% pass@2 at 150M |

- **For us:** the Perceiver / RIN pattern is the task-agnostic generalisation of DeltaMem.
  - Each think pass, the memory (or a few latent slots) **cross-attends to the full plain text** and writes back.
  - So the model, not the designer, picks which digits or edges to read at each step.
  - It's an alternative to INJECT's "re-run GPT-2 on the full text each pass": it's cheaper (cross-attention, not a full forward over the text) and makes the read pattern inspectable (C7).

## 2. Learned segmentation / chunking
| Design | Mechanism | Notes |
|---|---|---|
| **H-Net 2507.07955** (dynamic chunking) | a **routing module** predicts boundaries from the cosine similarity of adjacent encoder outputs; a **smoothing module** interpolates uncertain boundaries for learnability; down / up-sampling around a main network | byte-level LM; matches BPE Transformers at > 1B; end to end, no external boundary predictor |
| BLT 2412.09871 | entropy-based patching (a small LM's next-byte entropy sets the boundaries) | patches scale better than tokens |
| Dynamic token pooling 2211.09761; MEGABYTE 2305.07185; Large Concept Models 2412.08821 | learned or fixed pooling; sentence-level latent units | — |

- **For us:** H-Net's router (similarity-based boundaries plus smoothing) is a ready, task-agnostic way to let the model choose **where
  one think step ends** on plain text. It would replace §165's fixed k-token chunks.
- **Caveat:** these are pretraining-scale tokenisation methods, with no reasoning or C1 results. Retrofitting a router to pretrained
  GPT-2 is untested.

## 3. Adaptive computation / halting on LMs
| Design | Result | Pitfall |
|---|---|---|
| ACT 1603.08983; PonderNet 2107.05407; Universal Transformers 1807.03819 | the halting classics | **RecurTrace 2609.03379: in a controlled MathQA test on a looped LM, ACT and PonderNet *collapse to one loop*** |
| CALM 2207.07061 (confident early exit); Mixture-of-Depths 2404.02258; Mixture-of-Recursions 2507.10524 | token-level depth routing | CALM 54.1% at 5.6 loops vs RecurTrace 56.9% at 2.0 (MathQA) |
| **RecurTrace 2609.03379** | loop-memory attention over the loop's own past states, plus a halting head supervised by an *oracle* (does more depth still cut the loss?) | beats fixed depth at matched compute |
| Think-at-Hard 2511.08577 | a decider triggers extra iterations only on hard tokens; depth-aware LoRA; "latent overthinking" (correct after pass 1, broken later) | an oracle iteration policy is worth up to +7.3% |
| AdaAnchor 2603.15051 | latent anchor vectors refined until their dynamics converge (convergence halting) | +up to 5% vs fixed steps, 48–60% fewer steps (GSM8K / SVAMP / MultiArith) |
| 2511.21581 | post-SFT RL to shorten latent reasoning | Llama-3.2-1B, GSM8K-Aug: −52% length, no accuracy loss |
| FR-Ponder 2509.24238 | a < 1M-parameter controller halts or adds a steering step on frozen states (GRPO) | GSM8K / MATH500 |

- **For us:**
  - The pitfall to pre-register for any halting arm is the **collapse to one loop under ACT / PonderNet**.
  - **Supervise halting with an oracle** (RecurTrace), or use convergence (AdaAnchor / attractor ρ) rather than a learned cost penalty.
  - Our ρ / fixed-point measurements give a convergence-halting signal for free.

## 4. Input injection for pretrained LMs
- The recurrence retrofits 2608.11233, McLeish 2511.07384 and Relaxed Recursive 2410.20672 (prelude output re-injected each loop,
  Huginn-style).
- Text-level: Re-Reading 2309.06275; echo / repetition embeddings 2402.15449.
- Details in LIT_INJECT (6b019d1e).

## 5. "Think while reading" memory models
- **Quiet-STaR 2403.09629:** the model generates rationales after *every* token while reading text, learned from LM data alone.
  Mistral-7B zero-shot GSM8K **5.9 → 10.9%**. Task-agnostic by construction.
- Thinking States 2602.08332 (text thoughts re-embedded per chunk); fast-slow 2604.01577; BDH-CQ 2608.09888.
- **Long-input readers:** ReadAgent 2402.09727 (gist memory plus look-up); MemWalker 2310.05029 (a tree of summaries, navigated
  interactively). LLM-agent level, not latent.

## ★ GSM8K-relevant at GPT-2 scale
- **PCCoT 2506.18582** (parallel continuous CoT, Jacobi iteration). **GPT-2 Small on GSM8K-Aug: 49.48 ± 0.31** (GSM8K-Aug-NL 49.23), vs
  CoT 44.1, Continuous CoT 48.24, CODI 43.7, Coconut 34.1, iCoT 30, pause tokens 39.27.
  - **The highest GPT-2-small latent GSM8K number found**, above text CoT. Our leaderboard references (CODI 42.5 / SIM-CoT 44.8)
    should add it.
- Everything else with GSM8K is ≥ 1B (2511.21581, AdaAnchor, FR-Ponder, Quiet-STaR, CoLaR 2505.16552) or doesn't report GSM8K (H-Net,
  RIN, Perceiver, MoD / MoR).

## Ranked suggestions for task-agnostic latent methods (C1 > C6 > C4 = C2)
1. **READ-SLOTS (Perceiver / RIN-style read pass).** K think passes in which DeltaMem (or a few latent slots) cross-attends to the full
   plain text and writes back, with no designer-chosen operand per pass.
   - It's task-agnostic, cheaper than INJECT, and the read-attention map is a C7 window.
   - C1 is by K-removal beyond L* (DEEP-ProsQA / Feng).
2. **Convergence-halted think passes:** stop when the memory's per-pass change falls below a threshold (AdaAnchor / attractor).
   Register the ACT / PonderNet one-loop collapse as the risk of learned halting.
3. **H-Net-style learned boundaries** for streaming tasks: a router decides when to close a think step. It's riskier as a retrofit.
4. **A PCCoT comparison point** for any GSM8K return (49.5 at GPT-2 small).
