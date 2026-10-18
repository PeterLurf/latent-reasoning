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
