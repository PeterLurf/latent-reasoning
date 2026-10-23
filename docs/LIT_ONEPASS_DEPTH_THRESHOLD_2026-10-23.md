# When do one-pass (no-CoT) transformers learn k-step word problems or symmetric reachability? (for the §223 decision; 2026-10-23)

Written 2026-10-23 by the Literature session, for Theory and Audit. Kept light under the usage limit.
- **Sources:** mostly our own notes. I re-checked Lin 2503.07604 and Physics-of-LMs 2.1 2407.20311 in their HTML today, and the Balesni 2411.16353 ID.
- **Tags:** all [empirics] unless marked.

## Short answer
- **No published threshold fits our setting** (pretrained GPT-2 small, LoRA, 2.5k steps per stage).
- **The closest evidence says one-pass multi-step arithmetic is learnable at GPT-2 scale only under conditions §223's NESTED-WORDS breaks:**
  - **fixed premise order** (one surface layout);
  - **small-number arithmetic;**
  - **long from-scratch training.**
- Symmetric reachability without a gated latent curriculum stays at chance in every published arm.

## Arithmetic / word-program composition
- **Lin et al. 2503.07604** (GPT-2 trained from scratch; templated multi-step arithmetic, 1–5 training steps):
  - Implicit stepwise reasoning reaches high in-distribution and out-of-distribution accuracy **only with fixed premise order** (premises presented sequentially).
  - With unfixed premise order the model learns a shortcut (chaining numbers via commutativity) and decays with steps: 2 / 3 / 4 / 5 / 6 steps = 1.00 / 0.87 / 0.57 / 0.43 / 0.23 (forward test order, Table 1).
  - Calculation size is deliberately restricted, to "focus on reasoning rather than calculation".
  - **NESTED-WORDS samples sentence order and uses values 0–99. Both are the conditions Lin identifies as harmful.**
- **Arun, Zhao & Arulkumaran** (OpenReview IwMfUz9KNj, NeurIPS 2026 workshop, abstract only):
  - Layout diversity "imposes a large convergence cost"; per-example layouts "prevent learning the task altogether".
  - The next-token objective is a barrier under fixed budgets; reweighting ≥ 25 % of the loss onto answer tokens fixes it.
  - Same direction as Lin.
- **Physics of LMs 2.1 (Ye, Xu, Li, Allen-Zhu 2407.20311)** (from scratch, iGSM, **with** a solution trace):
  - **Depth matters more than width.** A 4-layer model at 1920 dimensions underperforms; a 20-layer model at 576 dimensions performs very well (Fig. 9).
  - Layers implement layer-by-layer dependency planning.
  - It is not one-pass, but it supports "layers bound reasoning depth".
- **Wang 2405.15071** (grokked implicit reasoners): two-hop composition is learned only after grokking (very long training) and generalises in-distribution only.
- **Dziri 2305.18654:** fine-tuned GPT-3 fails out of distribution as composition depth grows ("linearised subgraph matching").
- **Balesni 2411.16353:** latent two-hop over synthetic facts is hard even for Llama-3-8B / GPT-4o without CoT.
- **Our own record:** §80 P0, where GPT-2 memorised k-op words (unseen 0.03–0.12), is consistent with Lin.

## Symmetric reachability (§218)
- **Capacity is not the limit.**
  - Ye 2510.19753: L layers handle diameter ≤ 3^L. With GPT-2's 12 layers that is ≫ our D.
  - Log-depth theory: Merrill & Sabharwal 2503.03961; Sanford 2402.09268 [theorems].
- **Learning is the limit:**
  - Saparov 2412.04703: plateau; converging becomes vanishingly likely with size.
  - Mirtaheri 2505.21825 [theorem]: ½ with too few queries.
  - Gatmiry 2609.19717: every ungated arm is at chance; gated latent curricula solve it.
- So §218 DIRECT at chance is **expected**.

## For the §223 decision and the next task (implications, not prescriptions)
- **A d ≥ 2 null at 2.5k LoRA steps per stage is weak evidence of "untrainable".** Lin needs long training, Wang needs grokking-scale training, and our budget is small. Physics-2.1-style depth claims are about from-scratch models with traces.
- **Before closing §223, one cheap discriminating arm:** a **fixed premise order** (dependency order, one layout) with **small values** (e.g. mod 11, as §170) at the same budget.
  - If d ≥ 2 then trains, the null was layout / number-size (Lin, Arun), not depth.
  - Optionally reweight the loss onto answer tokens (Arun, abstract only).
- **For the next language task:** start where Lin's conditions hold (forward order, small values, depth ≤ 5 trained, OOD to 6+), then add layout diversity as a separate, later axis. Model size: GPT-2 small's 12 layers are not the binding constraint according to Ye's or Physics 2.1's depth results. Depth bounds bind well above d = 2.

## Addendum (2026-10-23): Lin 2503.07604 fact check (read in its HTML) and the OW_draft §12 verdicts
**Lin's exact regimes**
- **Values:** arithmetic **mod 23** ("to avoid numbers being split into multiple tokens and prevent errors from large number calculations"), with + and − only.
- **Format:** formulas, **not natural language**.
- **Model:** a 12-layer GPT-2 **from scratch with RoPE**.
- **Data:** 25,000 templates per length (2–5 steps), K = 2 variable-name groups. The test set filters out templates whose earlier steps overlap with training.
- **Structure:** a **linear chain**. Each step = the previous step's variable + an **inline number operand** + an operator ("a=1+2,b=3-a,c=b+5,c>>?"). So every given sits **inside the premise that uses it**.
- **Locality:** restricting attention to the current step (window 6) "completely loses" reasoning. Including the previous step recovers it.
- **"Fixed vs unfixed" is premise ORDER only** (forward vs reverse vs random shuffle of premises, the question excluded). Reverse and random **do** put premises before the premises defining their variables, which answers D2: yes.
- **Variable-as-subtrahend:** the variable is subtrahend in 25 % of steps. The shortcut fails there.

**Correction to my message of 2026-10-23:** I said NESTED-WORDS "samples sentence order". Theory's $0 check shows v1.2 was already in dependency order (0 / 20k violations). Against Lin, v1.2 therefore **did not violate the order condition**. It differed on:
1. **givens far from their use** (Lin's operands are inline, and its dependencies are distance-1);
2. **values 0–99 vs mod 23** (a larger answer space, and English add/sub semantics);
3. **natural-language frames and paraphrase / layout diversity** (Lin uses formulas);
4. **DAG reads of earlier quantities** vs Lin's strict previous-step chain;
5. **recipe:** pretrained absolute-PE GPT-2 + LoRA at 2.5k steps per stage vs from-scratch RoPE with full training.
- Of these, (1) and (4) are the closest to Lin's measured locality mechanism (the window experiment). OW's given-placement change targets them.

**Arun IwMfUz9KNj and paraphrase (abstract only):** the task ladder introduces "complex grammatical form and paraphrasing **through layout diversity**". Grammatical form and length are fine, but layout diversity costs convergence, and per-example layouts block learning. So **OW's paraphrase frames are likely within Arun's "layout diversity" axis.** Keep the frame count small and fixed (not per-example) until the PDF is read.

**OW_draft §12 verdicts**
| # | verdict | basis |
|---|---|---|
| N1 (one-pass vs CODI-style latent on a templated multi-step word task; removal + mid-chain latent-prefix swap scored against the donor-state counterfactual, past a measured one-pass boundary) | **NARROW** | The parts exist: CODI / Coconut / SIM-CoT (latent CoT on word problems), Lin (one-pass boundary on formulas), SCIT 2608.27265 (counterfactual cache transplant, synthetic templates), Liang & Pan 2602.00449 (position-swept state swaps). New only as the combination on a natural-language templated task, with the counterfactual-scored prefix swap past a measured one-pass boundary. |
| N2 (counterfactual-follow as the serial-vs-relay guard) | **NARROW** | It is SCIT's oracle-counterfactual target. Our §221 INT-FOLLOW applied it to natural GSM8K. Cite SCIT as the protocol. |
| N3 (premise-order shuffle as a separate axis for **latent** reasoning) | **OK** | Order sensitivity is known one-pass (Lin) and in CoT LLMs (Chen et al. 2402.08939, "Premise Order Matters": best when premise order matches the proof order). For latent CoT on shuffled-premise arithmetic: not found. ProsQA lists facts in shuffled order, but has no order ablation. |
| N4 | fact check | answered above |
