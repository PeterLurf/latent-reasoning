# ID checks: latent-step faithfulness papers, and whether a matched text-CoT vs latent self-improvement comparison exists (2026-10-18)

Literature session, for Audit (the request) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Sources:** 2607.06648 checked in full text; the rest from abstracts.
- **Evidence tag:** all EMPIRICAL.

| ID | verified title / authors | one-line summary (with corrections to the brief) |
|---|---|---|
| **2607.06648** | "Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories"; Jin, Yang & Wang. **Verified.** | Tracks counterfactual responsiveness and activation-intervention sensitivity across training checkpoints (COCONUT and CODI on GPT-2, with Llama-1B replications). **Accuracy can rise while responsiveness falls.** On ProsQA both decline by late checkpoints (replacement-specific at Llama-1B). **On GSM8K the two answer formats go opposite ways:** under norm-noise, intervention sensitivity **rises** on GSM-open and **falls** on GSM-Choice. The brief's "shrinks except for open-ended answers" is right in direction, with that replacement caveat. |
| **2512.21711** | "Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought"; Zhang, Tang, Ju, Duan & Liu. **Verified.** | Steering shows COCONUT tokens are insensitive to perturbation and "lack reasoning-critical information" (unlike CoT tokens). Biased / OOD shortcut tests on **MMLU and HotpotQA** (not GSM8K / ProsQA) show COCONUT exploiting dataset artefacts. So: placeholders plus shortcuts, as the brief says. |
| **2603.03475** | "When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning"; Sahoo, Chadha, Jain & Chaudhary (ICLR 2026 Workshop on Latent & Implicit Thinking). **Verified.** | **Qwen2.5-Math-7B** at 61% accuracy. **81.6% of *correct* predictions** come through "computationally inconsistent pathways" (18.4% stable); 8.8% of all predictions are confident silent failures. **Caveat:** evaluated on a subset (6% of GSM8K), and the model isn't a latent-CoT-trained model. The brief's "81.6% inconsistent on GSM8K" needs "of correct predictions, on a 6% subset". |
| **2606.27378** | "Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs"; Seddik & Fard. **ID confirmed.** | Four axioms (Causality, Minimality, Separability, Stability), measured directly on **frozen open-weight LLMs'** thought representations across 23 tasks. No candidate satisfies all four. The representations separate task type but not two questions within a task, and "encode little information beyond what is already present in the input embedding". So "a frozen-model thought-representation probe" is accurate. |

## Matched text-CoT vs latent self-improvement comparison (same task, same rounds)
- **Not found.**
- The self-improvement precedents are **text-CoT or answer-label only**:
  - Lee 2502.01612 (own outputs, length filter / majority vote);
  - Huang 2511.07378 (own CoT traces, doubling);
  - Shafayat 2505.21444 (RL with a majority-vote self-reward).
  - See LIT_ITERATED_SELFTRAIN / LIT_SELF_DEPTH.
- **Nearest latent-side items, none of them matched:**
  - **Abstract-CoT, Ramji, Naseem & Astudillo 2604.22709:** a policy-iteration-style warm-up alternating bottlenecked SFT and **self-distillation** of discrete "abstract" tokens, then RL. Compared with verbalised-CoT post-training, but not round-for-round self-improvement on a shared task.
  - **C-MTP, Yerram, He & Choi 2607.16972:** direct supervision of continuous CoT. Both direct and indirect methods drop about 65% on long-trace tasks. No self-training.
- **So a matched TEXT-SELF vs LATENT-SELF arm** (same task, the same round schedule and filter, a text-CoT model vs our latent model labelling its own data) **would be new.**
  - Design point: the text arm's labels are its own CoT strings. Teacher forcing on those strings has D1's value-token shortcut (LIT_STEPFACTOR_D1), so filter by final-answer / step-count validity in both arms for a like-for-like comparison.

## Addendum (2026-10-18): do 2602.08783 / 2602.00449 already have a single-hop input swap with a pass-time sweep? (Audit's CONSUMPTION-FRONT lead for LANG-VB)
- Both verified on arXiv abstract pages and read in full text (pdftotext).

**2602.08783:** Li, Bai, Chen, Li, Yang, Lin & Zhang, "Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure". **Verified.**
- **Venue:** the abs page says "**Accepted to ICML 2026**". An ICLR 2026 workshop listing also exists, so cite ICML 2026.
- **Interventions:** step-wise **do-interventions on latent STATES**, do(h_t ← h̃_t), with replacement operators such as mean or noise used as robustness variants. Coconut and CODI, maths and general reasoning.
- **Findings:** causal leverage is heterogeneous across steps, effects are non-local ("routed propagation"), and early output bias comes before representational commitment.
- **No input swap, no per-hop analysis, no crossover-vs-hop curve.** Not a precedent for CONSUMPTION-FRONT beyond the general step-resolved framing.

**2602.00449:** Liang & Pan, "Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks". **Verified.**
- **Closest precedent, but not the same experiment.** CODI on polynomial iteration s_t = F(s_{t−1}, x_t). Two relevant protocols:
  1. **Single-input-token corruption × position / layer restoration sweep** (App. activation patching).
     - Corrupt one input token, x̃_i = (2·x_i) mod 50, i.e. one hop. Then patch clean activations into the corrupted run at each phase: encoder tokens, [BoT], latent 1–6, [EoT], [Ans]; per layer.
     - Two-hop result (Fig. 4): x2 corruption is rescued at [BoT] (100% at L2-post), l1 (about 80%) and l2 (about 70%), and not at later latent steps. **x3 (the last input) is rescued only at [Ans]**, a direct late-fusion route that bypasses the latents.
     - General n: corruption at earlier inputs "yields nontrivial recovery at specific latent reasoning steps, with larger i generally producing stronger recovery". That's **qualitative, with no hop-indexed crossover curve**.
  2. **State swap swept over latent position** (App. H, length 3 only): suffix-matched pairs (A, B) differing in the bridge state s2.
     - Flip-to-B is 90.1 / 80.1 / 71.5% at [BoT] / l1 / l2, about half at l3, 0.5% at l5, and 0 at l6 / [EoT].
     - That's a **pass-time crossover curve for one STATE swap at one depth**, not an input swap and not across hops.
- **What it doesn't have:**
  - a looped model with the input **re-presented every pass** (CODI's "time" axis is the latent-token position within one forward);
  - an **input** swap applied from pass t onward;
  - the full **hop × pass** grid with a consumption front (crossover pass as a function of hop index).

**Verdict for CONSUMPTION-FRONT:**
- **Partly anticipated in design, new in setting and in the curve.**
- Frame it as extending 2602.00449's single-token corruption × position patching, and its state-swap pass sweep, to a looped re-presented-input model, with the full hop-indexed crossover curve. Don't call it a replication.
- **Prior expectation from 2602.00449 (CODI):** early hops are consumed at early latent positions, and the **final hop bypasses the latents** (late fusion at the answer). A consumption front would show the last hop off the diagonal.
- 2609.01924 (Huginn injection swap: 30 / 30 flips, about a 2-recurrence memory; LIT_INPUT_BYPASS_LOOPED) is the looped-model precedent for a whole-input swap, though not single-hop.
