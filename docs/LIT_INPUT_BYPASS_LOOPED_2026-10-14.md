# ID checks on Audit's "re-presented input as the bypass route" lead: three looped-transformer papers, their seed variance and any bypass / recompute failure mode (2026-10-14)

Literature session, for Audit (the request) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Sources:** all three read in full text (pdftotext; searched for seeds, bypass, recompute and reasoning tasks).

| ID | verified title / authors | what it is | seed variance? | bypass / recompute mode? |
|---|---|---|---|---|
| **2607.27656** | "Looped Transformers with Source-Centered State Evolution"; Kim, Hayashi, Kamiya, Koyama, Iwasawa & Matsuo. **Verified.** | SCSE. In additive-injection looped transformers the re-injected input signal moves the state even at the reference (a "zero-deviation forcing bias"). SCSE makes an input-conditioned anchor an exact one-step fixed point while nonzero deviations still drive computation. The theory shows the forcing bias can be harmful, neutral or beneficial. Evaluated on **language modelling** (WikiText-2 / 103, web pretraining, LAMBADA, adaptive depth). | Mentions "stochastic-replication counts" in the appendix. The case study uses **one** replication. **No seed-variance result on reasoning.** | **Mechanism-level only:** the injected input acts as a forcing term on every step. **No reasoning tasks; no bypass reported.** |
| **2610.06833** | "Towards Looped Models Done Right, Part II: Rethinking at Fixed Points"; Huang, Shi, Chen, Wen, Liu, Xing & Ma. **Verified.** | Learned depth prior plus **orthogonal injection (OrthoInj)**: "existing injection schemes let the state's component along the input amplify or cancel the injection; we remove this component". Pretraining from 100M to 1.6B, with GSM8K / MATH500 evaluations. Fixed points enable terminal-KV sharing; at 1.6B, fixed-depth full-BPTT training plus sharing drops GSM8K **50.6 → 21.2**. | **Explicitly single-seed:** "these single-seed comparisons do not estimate seed variance"; "they condition on one training seed and so exclude seed variance"; "multiple seeds … remain untested". | Not reported as such. It cites Huginn vs Ouro: terminal-KV reuse "barely changes" Huginn's GSM8K but "collapses" Ouro's, tracking convergence to fixed points. |
| **2609.01924** | "Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?"; Wang & Reid. **Verified.** | Jacobian-lens workspace analysis of released **Ouro-2.6B** and **Huginn-0125**, with Qwen3.6-27B as the baseline. | n/a: released checkpoints, no training. | **Yes, the closest published evidence for the lead.** Huginn's apparent persistence across 16 recurrences "is continuous recomputation from the re-injected input, not a memory buffer". **Injection swap:** replacing the injected embedding with a one-word-different prompt's from recurrence k onward "flips the final output to the new input's own answer in all 30 configurations, including k = 12". The old answer's rank decays within 2 recurrences: "a 2-recurrence memory". Ouro "re-derives its workspace content each loop". The authors note a definitive answer "needs controlled training runs (matched small looped models with and without injection)". |

## Reading for the lead
- **The published evidence that a pretrained looped LM recomputes from the re-injected input instead of carrying state is 2609.01924** (EMPIRICAL; released checkpoints; Huginn's state has about a 2-recurrence memory).
- **SCSE and OrthoInj give mechanism-level reasons why additive input injection shapes or dominates the state** (EMPIRICAL plus derivations), but neither tests reasoning-task bypass or seed variance.
- **None of the three reports seed variance on reasoning tasks.** 2610.06833 explicitly disclaims it.
- **A $0 in-house test suggested by 2609.01924:** the **injection swap**, the converse of our SERIAL state swap.
  - Keep the memory, and replace the re-presented think input with a different item's from pass k onward. Measure how fast the output follows the new input.
  - **Prediction under the lead:** failed seeds follow the new input within one or two passes (recompute); converged seeds follow the memory (carried state).
  - With the SERIAL whole-state swap (converged 0.53–0.96, failed 0.07–0.12), this gives a 2 × 2 dissociation: state-following vs input-following.
- **2609.01924 calls for "matched small looped models with and without injection"; our setup is that controlled run.** Our INJECT design re-presents the full problem every pass, and the memory is the only other channel. Cite it as the direct motivation.

## Addendum (2026-10-16): SCSE 2607.27656 details, for the D3 §2.3 trajectory-drift lead (Audit; full text)
- **Mechanism, confirmed:**
  - The input-conditioned anchor h̄ = e + a(e) is computed **once** per input.
  - The state evolves as a deviation δ_t = h_t − h̄ through a bias-free recurrent core (G(0) = 0).
  - A per-example **zero-deviation mask** m_{b,t} = 1{‖δ_t‖_F² > ε} gives T_t(0; e) = 0, so **the anchor is an exact one-step fixed point by construction**. The "zero-deviation forcing bias" b_t(e) = T_t(0; e) is zero.
  - Ablations name the learned anchor and the anchor-coordinate deviation recurrence as the main contributors.
- **Training regime: FROM SCRATCH:** "we train decoder-only LMs from scratch" on WikiText-2 / 103, OpenWebText and C4 (GPT-2 tokenizer only). Shared-block models of 22M, 50M, 95.6M and up to 139.2M. **Language modelling only, no reasoning tasks.**
- **Depth extrapolation:**
  - Training samples loop depth uniformly from {1..8}. Evaluation runs at T ∈ {4, 8, 12, 16, 24, 32, 48}, with T > 8 called "extra-loop".
  - At 50M (PPL), the looped baseline goes 151.1 (T 8) → 178.9 (T 48); the step-conditioned adapter 125.7 → 160.1; **SCSE 123.1 (T 8) → 135.5 (T 24) → 156.4 (T 48)**.
  - So SCSE has the best frontier at every depth, but it **still degrades with extra loops**: it reduces drift's cost, it doesn't remove it.
  - At 22M, ± across replications is reported (e.g. 156.7 ± 0.5).
  - Adaptive depth: a 48-loop ceiling, with a calibrated hidden-update stopping rule targeting mean budgets of 8 / 16 / 24.
- **Drift evidence:** in additive-injection models, trajectories from the anchor drift along input-specific "forcing rays" t·b(e), and an injection-gain dose test scales the displacement. That's one WikiText-2 model, one replication, six inputs (Fig. 4): illustrative.
- **Transfer precondition for us (flag):**
  - SCSE targets **additive re-injection into a residual hidden state**.
  - Our carried state is the DeltaMem memory S. The input is re-presented as text each pass and enters through the trunk, not added to S. Our "drift" lives in S's additive heads (no decay) and the delta heads' α^T contraction.
  - **An SCSE analogue would need an anchor on S** (e.g. S after pass 1, or a learned function of the input) and a zero-at-anchor update. That's a new design, not a port.
