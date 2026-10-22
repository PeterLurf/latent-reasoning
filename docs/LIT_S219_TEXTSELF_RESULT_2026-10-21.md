# §219 result (Note 219-13, CROSSED) against the self-improvement literature (2026-10-21)

Written 2026-10-21 by the Literature session, for Theory and Audit.
- **Our own earlier checks:**
  - LIT_FAITHFULNESS_SELFIMPROVE_2026-10-18 (matched text vs latent self-improvement: not found → new);
  - LIT_FIRSTPASS_2026-10-20 (still new);
  - LIT_ITERATED_SELFTRAIN_2026-10-17 (Lee / Huang / Fan / Yang margins).
- **Refresh today:** two web searches (extended + standard) for latent or looped self-training with depth extrapolation, matched to text. Nothing new found.
- **Numbers:** read from Note 219-13 (ca6e4b54).

## (a) Verdict on "a matched text-CoT vs latent-recurrent self-training comparison on depth extrapolation": **OK**, with scope stated
- **Not found anywhere:** the same task, round schedule, windows, steps, mixture and form-only filter, with a text-CoT model and a latent-recurrent model each labelling its own deeper items.
- **Text-only self-improvement:** Lee 2502.01612, Huang 2511.07378 [theorem], Liu 2602.10014 [theorem], STaR 2203.14465, ReST^EM 2312.06585.
- **Latent-side items are not round-for-round self-training:** Abstract-CoT 2604.22709 (self-distillation warm-up, then RL), C-MTP 2607.16972, Latent-GRPO 2604.27998 (RL).
- **Required scope in the wording:** one lineage per arm, GPT-2 small, Arithmetic(n, 11), two rounds, and train tokens not compute-matched (text 1.19e8 vs latent 2.99e8 in r2). Note 219-13's §5.1 wording already carries this.
- **Updated 2026-10-22 (§225, Note 225-1, 189cd6c2):** 219-13's TOKEN-COST overcounted latent steps by 3×. At matched compute on both axes, latent matches text-CoT with ~20–50× fewer sequential steps and ~0.65× FLOPs per correct answer (n 13–20; one lineage each, GPT-2, Arithmetic(n, 11)).
- The "~20–35× fewer sequential steps" point is the familiar latent-efficiency argument (Coconut 2412.06769, CODI 2502.21074 report fewer generated tokens). Cite it as consistent with them, not as a new kind of claim.

## (b) Where our numbers bear on cited claims [descriptive; two rounds, one lineage, so weak]
**Margin past support (n50 − support).** Support is 8 / 12 / 16 at base / R1 / R2.

| arm | base | R1 | R2 |
|---|---|---|---|
| text | 2.43 | 2.61 | 4.16 |
| latent | 2.31 | 3.46 | 4.54 |

- n50 / support is nearly constant: text 1.30 / 1.22 / 1.26, latent 1.29 / 1.29 / 1.28.
- **That is closer to a margin proportional to the trained depth** (Huang 2511.07378's constant-factor generalisation per stage; Yang 2610.14456's extra depth Θ(H)) **than to the near-constant +k margin** in Lee 2502.01612's reverse-addition curve (Fig. 17: about 3–5 extra lengths flat across 95 rounds).
- **Caveats:**
  - Lee uses +1 per round over many rounds and from-scratch NoPE models; ours is +4 windows with pretrained GPT-2.
  - Two points can't separate the forms.
  - Cite this as "consistent with proportional margins", not as a test.
- **Lee's error-avalanche threshold** (self-label accuracy ≲ 0.9 → collapse) is not approached in either arm. Text labeller keep / π ≥ 0.98 / 0.999 on the windows (219-10 / 219-11), and the latent labeller was gold-equivalent (π ≥ 0.9994). This is consistent with Lee's "filtering is needed only when labels are noisy".
- **Chen 2510.18874** (Self-SFT forgets in outputs) vs our latent arm: forgetting is internal-only in round 1 (216-13) and starts reaching outputs in round 2. The text arm's retention is T-RETAIN met (min a(9..12) = 1.000). This is the first matched data point that self-labelled text CoT kept old-depth outputs here.

## Lead (descriptive)
- An unindexed NeurIPS 2026 listing, "What Breaks Depth Extrapolation in Looped Language Models?", reports that reweighting ~25 % of the loss onto answer tokens restores depth extrapolation under fixed budgets.
- It bears on §218's answer-vs-template-token imbalance and on §223's R_j-length risk (LIT_S218_CLOSE_S223_AUX_2026-10-21).
- **Not yet verified:** I found no arXiv ID. Don't cite it until it is found and read.

## Addendum (2026-10-21 ~15:30 EDT): the lead pinned down, partly
**Citation**
- Arun, Zhao & Arulkumaran, "What Breaks Depth Extrapolation in Looped Language Models?"
- **NeurIPS 2026 Workshop on the Linguistic Principles for Foundation Models**, a workshop paper, not the main track.
- OpenReview **IwMfUz9KNj**. No arXiv ID was found (export.arxiv.org author / abstract search).
- The OpenReview PDF and API are behind a browser challenge, so **the full text was not read**.

**Abstract (from the NeurIPS page, verbatim excerpts):**
- "a task ladder that gradually moves from pure symbolic composition towards natural language by introducing complex grammatical form and paraphrasing through layout diversity".
- "Grammatical form and length still allow for depth extrapolation, while layout diversity imposes a large convergence cost on learning, and per-example layouts prevent learning the task altogether".
- "the standard next-token prediction training objective … creates a major optimization barrier to depth extrapolation under fixed budgets, while reweighting even 25% of the loss on answer tokens restores it".
- The authors recommend controlling layout diversity and "focus[ing] the loss on task-important tokens".

**Not available without the PDF:**
- the exact meaning of "25 %" (most likely the share of total loss weight on answer tokens, but unconfirmed);
- whether a curriculum is used;
- model sizes, tasks and budget.

**Use:** [empirics, abstract only, workshop]. Cite as "consistent with" for §218's diagnosis (template tokens absorbing the loss) and as a candidate remedy for §223. Don't quote a recipe until the PDF is read. A user with an OpenReview login could download it.

## Addendum (2026-10-22): MB_draft N1 (matched-budget text-CoT vs latent self-training)
- **Verdict on N1** ("a registered matched-budget comparison (sequential steps and token-position compute, both reported) between matched text-CoT and latent-recurrent self-training lineages"): **OK**, scoped to GPT-2, a synthetic task and our lineages.
- **No compute-matched latent-vs-CoT comparison was found:**
  - Prior efficiency claims are generated-token counts (Coconut 2412.06769, CODI 2502.21074) or **latency** (LOTUS 2606.31779: matches explicit CoT at 3B with 2.5–6.9× lower thought-phase latency).
  - Ouro 2510.25741 and iso-depth 2604.21106 scale recurrence against parameters or depth, **not against CoT**.
  - Our own 10-01 check (LIT_PERM_TRACKING_CHECK: compute-matched CoT vs latent on state tracking) found none either.
  - One web search (2026-10-22) found none.
- **Cite LOTUS** as the nearest efficiency comparison, but it is latency-framed and not budget-matched.
