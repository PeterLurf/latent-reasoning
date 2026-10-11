# Two depth-generalisation papers (Chen 2603.21676; He 2512.02677): verified, and how they relate to our step-supervision results (2026-10-11)

Literature session, for Audit (the request) and Theory. Both IDs verified via arXiv abstract pages (2026-10-11). Neither was previously in docs/.

## Chen 2603.21676, "Thinking Deeper, Not Longer: … Depth-Recurrent Transformers for Compositional Generalization" (Hung-Hsuan Chen; v1 and v2 on arXiv)
- **Model:** a depth-recurrent transformer (a shared-weight block iterated), **with no token generation**.
- **Three stabilisers for 20+ thinking steps:**
  - (i) a **"silent thinking" objective that supervises only the final output**;
  - (ii) **LayerScale initialisation**;
  - (iii) an **identity-biased gate** ("a gradient highway across steps").
- **Domains:** graph reachability (adjacency masking), nested Boolean logic (relative positioning), unstructured relational text.
- **A "computational frontier":** accuracy climbs once thinking steps ≥ the task's complexity. It's near-perfect on the two structured tasks.
- **Extrapolation:** it extrapolates beyond the training range on the graph task (where fixed depth barely does). On the two sequence tasks it comes within
  2 points of fixed-depth models with 4–6.4× more parameters.
- **The supervision claim, scoped exactly as stated:** "**On the graph task**, whose adjacency mask makes propagation depth verifiable, intermediate
  per-step supervision … consistently *harms* this extrapolation."

## Does our data contradict it? Not directly. State the scope.
- Their harm result is **on the graph-reachability task only** (depth verifiable through the adjacency mask), in **small from-scratch** recurrent-depth models.
  It's not reported for their nested-Boolean task.
- Our evidence: §184 per-pass reweighted aux on **Feng nested arithmetic** gives n 9–12 = **0.486**, vs **0.201** for faded §175, on **pretrained GPT-2 with a
  DeltaMem** carrier.
  - That's a different domain, a pretrained model with a memory state (not the residual), and a different comparison (reweighted aux kept vs aux faded,
    not aux vs final-only from the start).
- **Fair wording:** "Chen (2603.21676) finds per-step supervision harms extrapolation in from-scratch depth-recurrent models on graph reachability. On nested
  arithmetic with a pretrained LM and a fast-weight memory, our per-pass step aux helped (§184 vs §175-faded). The difference may reflect domain
  (verifiable propagation vs value computation) or pretraining."
- **Consistent with both:** ProsQA-Ext 2609.35643 (trace-free latents beat CoT on longer graph hops) agrees with Chen on graphs. The recurrence retrofit
  2608.11233 (step-per-loop supervision on a pretrained model, ~1.5× extrapolation) agrees with our pretrained result.
- **Borrowable stabilisers:** LayerScale init and an identity-biased gate on the think-pass update. Both are cheap and orthogonal to NORM-CAP / ADD-NORM.

## He 2512.02677, "Exploring Depth Generalization in LLMs for Solving Recursive Logic Tasks" (Zhiyuan He; AAAI per Audit)
- **The finding:** standard transformers decay rapidly with **recursion (nesting) depth** beyond training, even when they handle longer non-nested sequences.
  The cause is the lack of **stack-like behaviour**.
- **The method:** a **looped locate-and-replace pipeline**. A *locator* model finds a solvable sub-expression and a *replacer* evaluates it, keeping the
  structure. That's explicit decomposition into text.
- **Domains:** Boolean algebra, **recursive arithmetic** and propositional logic, each with controllable depth. It reduces decay on OOD depth.
- **For us:**
  - an explicit-decomposition baseline, close to our TEXT-COT (Feng-style one reduction per step) but with two models and structural replacement;
  - its "stack-like behaviour" diagnosis matches LIT_TC0_SERIAL's flag: nested expressions need a stack, so a fixed memory must hold the pending partial
    results up to the nesting depth;
  - **a useful external comparator for depth extrapolation on recursive arithmetic.**
