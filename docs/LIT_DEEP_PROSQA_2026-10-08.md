# Prior check for DEEP-ProsQA: depth-extended graph reachability past the single-pass boundary (2026-10-08)

Literature session, for Audit (the lead) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Numbers:** from the full HTML of 2412.06769 and 2604.06374, and the GitHub API for the issue (2026-10-08).

## 1. Depth-extended ProsQA / deep reachability benchmarks with latent or CoT baselines
- **ProsQA itself** (Coconut 2412.06769): answer paths of up to **6 reasoning steps**. Coconut uses c = 1 and 6 stages because "the
  maximum number of reasoning steps is 6". **No published depth-extended ProsQA was found.**
- **Synthetic graph reachability with depth / size control:**
  - **Saparov et al. 2412.04703, "Transformers Struggle to Learn to Search":** small transformers on graph connectivity with a
    controlled (balanced) distribution. They learn a parallel per-vertex reachability expansion: each layer expands reachable sets,
    so search covers a number of vertices exponential in the layer count.
  - **But difficulty grows with graph size, isn't fixed by more parameters, and in-context search (CoT) doesn't resolve it.** That's
    the closest "accuracy vs depth / size" reference, and it's from scratch.
  - **Reasoning by Superposition 2505.12514:** theory plus experiments. A 2-layer transformer with D continuous thoughts
    (D = diameter) solves directed reachability by a superposed BFS; discrete CoT needs O(n²) steps.
  - **2509.23365:** the training dynamics of that superposition (index-matching logits, two stages).
  - **ProntoQA, Saparov & He 2210.01240:** 1–5-hop proofs. Greedy proof planning is the failure mode.
- **For us:** the Saparov result implies a **single pass handles depth up to about 2^layers in expanded reachable sets**. A measured
  single-pass boundary for GPT-2 small (12 layers) could therefore be large. Measure it on the DEEP generator itself (as §162's L*)
  before committing to a depth range.

## 2. The "pause-token control matches Coconut on ProsQA" claim
- **The primary source is Coconut's own Table 1 (2412.06769v4)**, not the GitHub issue.
  - ProsQA: **Coconut 97.0 ± 0.3; "pause as thought" 96.6 ± 0.8**, i.e. <pause> tokens in place of continuous thoughts with the
    *same* multi-stage curriculum.
  - Also on ProsQA: "w/o thought" 95.5 ± 1.1; iCoT 98.2; CoT 77.5; No-CoT 76.7; the plain "Pause Token" baseline (no curriculum)
    75.9; "w/o curriculum" 76.1.
- **So the curriculum, not the continuous thought, carries Coconut's ProsQA gain.** That's visible in the original table.
- **microprediction/latentreasoning issue #32** (opened 2026-09-17) quotes the same Table 1 and 2604.06374. It's a secondary source.

## 3. What 2604.06374 and 2607.06648 show about ProsQA mechanisms
- **2604.06374, "The Illusion of Superposition?"** (Rizvi-Martel, Rabusseau, Mosbach). Logit lens plus entity probing, in three
  regimes.
  - **Fine-tuned pretrained models (Table 2a): a counterfactual no-latent evaluation drops accuracy by ≤ 1.0%** (GPT-2 124M: 99.0 →
    99.0; SmolLM2-135M / 360M / 1.7B: −1.0 / −0.7 / 0.0). **Shortcut: the answer is obtained in a single forward pass, then copied
    through the latents.**
  - **From scratch (Table 2b, 8 heads, 768-d), w/ / w/o latents:** 2 layers 94.5 / 13.8; 4 layers 96.2 / 16.0; 8 layers 95.2 / 53.0;
    12 layers 91.2 / 34.1. **Latents are used, and only from-scratch models show superposition.**
  - The causes: pretraining biases late layers to commit to a token, and capacity decides the solution.
- **2607.06648, "Final Checkpoints Are Not Enough":** tracks verified counterfactual edits and interventions along training.
  - **High accuracy can coexist with *low and declining* counterfactual responsiveness.**
  - On ProsQA, the output's sensitivity to norm-noise replacement falls alongside responsiveness (depending on the replacement).
  - Different latent methods follow different trajectories.
- **Also:** Aswal 2606.12689 (Coconut / CODI vs controls lacking recurrence or curriculum). BFS-like frontiers appear in the controls
  too, and utilisation is graded, so decodable patterns ≠ mechanism.
- **For us:** with pretrained GPT-2, *standard-depth* ProsQA is solved in one pass, which is the shortcut. That supports the
  DEEP-ProsQA premise: go past the single-pass boundary, or C1 can't be forced (cf. THEORY_C1_FORCING).

## 4. Is there a ProsQA-NA / shortcut-free variant in the literature?
- **Not found. Ours (§57; LIT_PROSQA_SHORTCUT) appears to be the only one:** the person-agnostic proximity heuristic scoring 0.911,
  and the shortcut-free set on which our Coconut reproduction falls 0.974 → 0.738 (0.732 with no thoughts).
- The literature critiques ProsQA via ablations and controls (Coconut's own Table 1; 2604.06374; Aswal 2606.12689; Dilgren
  2604.04902), not via a de-shortcutted generator.

## Design flags for DEEP-ProsQA
1. **Combine depth extension with the §57 shortcut removal.** Otherwise deeper graphs may still be solved by the proximity cue.
2. **Measure the one-pass boundary** with a matched direct (no-latent) GPT-2 on the new generator. Saparov predicts parallel reachability
   reaches far with 12 layers. Pick depths beyond the measured L*.
3. **Balance the distribution of path lengths and distractor branching** (Saparov: the training distribution decides whether search is
   learned).
4. **Controls from the literature:** a no-latent evaluation (2604.06374), a pause-with-curriculum arm (Coconut Table 1), and
   checkpoints along training (2607.06648).
