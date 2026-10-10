# Facts for STEP-AUX-ProsQA: ProsQA-Ext vs our p8 deep ProsQA, the baselines to beat, and other small NL multi-step benchmarks (2026-10-10)

Literature session, for Audit (the user's "smaller language task before GSM8K") and Theory.
- **IDs:** verified via arXiv abstract pages.
- **ProsQA-Ext:** details from its HTML (Sec. 2–3, App. "Dataset", "Training details").
- **Ours:** from prosqa/make_deep.py and bench/LEADERBOARD.md (2026-10-10).

## (a) ProsQA-Ext (Cheng & Zhang 2609.35643) vs our p8 generator
| | ProsQA-Ext | our p8 (`prosqa/make_deep.py`, PROTOCOL8) |
|---|---|---|
| **How depth is made** | **genuinely long proof paths:** OOD graphs built by Alg. 1. Two vertex-disjoint directed paths of length H, plus 13 − H off-path nodes per side attached to the root / internal nodes | ProsQA short problems (3–4 hops) with **chains of m new concepts hung off both options**. "Only the base of each option's chain decides the answer" |
| **Training depths / size** | **400k** new graphs, 3–6 hops (graph size, proof length, root and number of shortest paths quota-matched to ProsQA) | **12,000**, m ~ U{0..6}, so depth 3–10 |
| **Evaluation** | ID val / test = ProsQA's released graphs and queries, **relabelled** with premise and candidate order randomised; **OOD 7–12 hops** | test = every short test problem × m = 0..10, i.e. **depth 3–14** (1,684 items); plus a control set (chains off non-option concepts) |
| **Shortcuts** | labels assigned after construction, premises shuffled, the correct candidate balanced across sides; "no obvious superficial statistics" (≈ 50% for a shortcut) | standard p8 has the proximity shortcut. **p8_na** (§59) neutralises it |
| **Models** | **from scratch**: GPTNeoX, 4 layers, hidden 256, 4 heads, FFN 768; AdamW 4e-4 constant, batch 256, bf16 | pretrained GPT-2 small |
| **Latent setup** | Pause and latent variants use **K = 6 thinking positions**. Latent models are **end-to-end through the recurrence, with no intermediate targets, curriculum or RL**. Text CoT supervises the shortest-proof tokens | Coconut curriculum; ours |

- **One structural difference matters for STEP-AUX:**
  - in ProsQA-Ext the answer requires following an H-hop path;
  - in p8 the hung chains lengthen the *text path* but the answer is decided at the chain base, so "hop t of the gold path" for t > the base depth
    walks a chain that doesn't change the answer.
  - Per-pass hop targets are meaningful on both, but **p8's deep hops may be answer-irrelevant**, a possible shortcut of its own. Check how p8 defines the
    gold path past the base.

## (b) ProsQA-Ext's reported accuracies by depth
- **Only in figures** (Fig. 1C / 1D: free-generation accuracy, ID 3–6 and OOD 7–12, SEM error bars). **No numeric table in the HTML.** Read the values off
  the figure or ask the authors if numbers are needed.
- **What the text states:**
  - **ID:** Bottleneck-latent and Full-latent reach "nearly perfect accuracy", Pause is "closely behind", and CoT is "high overall" but declines as H
    grows.
  - **OOD (7–12):** Full- and Bottleneck-latent "retain the highest accuracy" (Bottleneck slightly ahead), clearly beating Pause and Direct. CoT and Pause
    rely on local graph features (candidate in-degree strongly affects Direct and Pause).
  - **K = 5 / 6 / 7** thinking positions are reported for OOD accuracy (one fewer, the trained count, one more).
- **No pretrained-GPT-2 rows.** All five variants are from scratch.

## (c) Pretrained-GPT-2 latent results on deep ProsQA that we'd have to beat
- **Ours (bench/LEADERBOARD.md, prosqa_deep, 1,684 items, depth 3–14):**
  - **text CoT (p8c_cot s1–s4) 74.8–78.1%**, roughly flat across depth;
  - **Coconut (p8c / g / q variants, K 6–16) 60.3–63.2%**, near 75–98% at d3–4 and ≈ chance (50%) from about d7;
  - **F1-S (ours; P = 16) 100.0%** at every depth.
- **Shortcut-free p8_na:** CoT 73.9–76.2%; Coconut 53.1–56.3%.
- **External:** only the HF card bmarti44/coconut-curriculum-checkpoints (Coconut GPT-2: 7 hops 66.0%, 8 hops 67.5%; a *different generator*, not
  peer-reviewed; already in literature.json).
- **Nothing peer-reviewed reports a pretrained-GPT-2 latent result on deep ProsQA.** Our CoT and Coconut rows are the bars, and F1-S (100%) is our own
  ceiling.
- **The bar for STEP-AUX-ProsQA:**
  - beat **text CoT (~76%)** on p8_na at depth > 10 (OOD vs train 3–10);
  - show THINK-NECESSARY;
  - F1-S already shows 100% is reachable on p8, so a step-aux latent result must be read on the **na** split and with causal tests to add anything.

## Other small natural-language multi-step benchmarks with GPT-2-scale latent baselines
- **ProntoQA** (Saparov & He 2210.01240; 1–5 hops):
  - Coconut's GPT-2 Table 1: CoT 98.8, No-CoT 93.8, Coconut 99.8, pause-as-thought 100.0. **Saturated** at GPT-2.
  - Dilgren 2604.04902 releases matched GPT-2 ProntoQA checkpoints (cot / coconut / codi; LIT_GSM8K_STEP2).
  - **PrOntoQA-OOD 2305.15269** tests deeper or compositional deduction OOD (LLM prompting); no GPT-2 latent results.
- **CLUTRR 1908.06177** (kinship, relation-chain length k; trained short, tested longer):
  - a classic text length-generalisation benchmark;
  - **no latent-reasoning GPT-2 results found** (its baselines are LSTM / GNN / BERT-era).
- **AGCLR 2606.07720:** GPT-2 + Coconut + persistent memory: ProsQA 96.0 vs Coconut 92.0 (standard depth), and HotpotQA EM 14.0.
- **Recommendation:**
  - **p8_na (ours) is the best small target with GPT-2 bars already measured** (CoT ~74–76; Coconut ~53–56; F1-S 100).
  - ProsQA-Ext is the external, from-scratch reference with a cleaner long-path generator. **Consider adding a ProsQA-Ext-style generator** (Alg. 1) as a
    second test set, because p8's deep hops may not change the answer.
  - ProntoQA is saturated. CLUTRR is an untested option with no latent baseline.
