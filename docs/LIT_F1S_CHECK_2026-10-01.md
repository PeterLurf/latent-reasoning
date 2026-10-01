# Pre-registration check: F1-S (mixed-depth final-answer training of the entity-label channel) (2026-10-01)

Literature session, for Theory; Audit copied. Committed before the F1-S registration. IDs verified (arXiv API or
Semantic Scholar).

## 1. Easy-to-hard / length curricula for iterative models without step labels
- **Fan et al. 2409.15647: the closest design precedent.** Looped transformers trained with "only end-to-end
  supervision (input and output) and the number of steps needed": the loss is applied only at T(n), with T(n) given per
  example, over mixed lengths, and the model extrapolates. F1-S (P = d, final answers, depths 1–6) is the same recipe
  applied to an entity-label channel in GPT-2.
- **Ye et al. 2510.19753:** training data *within the model's capacity* drives learning of the algorithm, while
  beyond-capacity data drives heuristics. This directly predicts that shallow questions teach the one-hop rule better
  than deep-only ProsQA, matching your §54b collapse.
- Schwarzschild 2106.04537 / Bansal 2202.05826: easy-to-hard with output-only loss and extra iterations at test.
- Kohli 2604.07822: recurrent depth trained on mixed hops, extrapolating 5 → 10.
- Saparov 2412.04703: search is learned only with a balanced distribution over lookahead depths.
- Lee 2502.01612: self-improvement on progressively harder instances.
- Zhang 2607.20594: T = n training sets speed.
- Multi-hop curricula: Coconut's stage curriculum 2412.06769; Grokked Transformers 2405.15071 (atomic plus 2-hop
  composition).
- **Novelty of F1-S as a method: low.** This is an established recipe. What is new is its application to a *discrete,
  readable per-entity label channel* in a pretrained LM.

## 2. Does shallow-question augmentation count as step supervision (C4)?
- In this literature, shallow questions are **separate terminal tasks**, i.e. outcome supervision:
  - Fan frames them that way;
  - SCRL 2605.22074 uses sub-problems as standalone outcome-reward questions;
  - Rajaraman 2606.27721 analyses decomposition into shorter sub-problems in an RL-with-verifiable-outcomes setting;
  - our boundary-memory proposal (§8) allows independent short tasks "disclosed and separately counted".
- **Two disclosures needed for a C4 claim:**
  - **P = d leaks the depth.** Fan supplies T(n) too and calls it out. That is a per-example step count tied to the
    target. By Prop. 4(b) (Kuo's K = n) it can become a count shortcut. Test at P > d and with learned halting, not only
    P = d.
  - Depth-1 questions over the *same graphs* as the deep ones make the label rule effectively supervised per hop. That
    is legitimate outcome supervision on other tasks, but state it, and run an arm with depth-1 questions on *disjoint*
    graphs to show transfer rather than per-graph label leakage.

## 3. LM work with a discrete per-entity state learned from shallow questions and iterated
- **Not found.** Nearest:
  - EntNet 1612.03969 (per-entity memory slots updated end-to-end from bAbI answers; continuous);
  - entity-tracking probes in LMs 2305.02363;
  - NEE masks 2006.08084 and CLRS hints (supervised per-node state);
  - DiscoLoop 2607.00341 (a discrete channel in a loop);
  - masked diffusion (discrete per-position refinement).
- **New if scoped as:** a discrete, readable per-entity label channel in a pretrained LM, learned from mixed-depth final
  answers and iterated beyond the trained depth.
