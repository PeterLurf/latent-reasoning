# Literature review for new ideas (2024–2026): latent reasoning, state tracking, discrete latents, composition (2026-10-02)

Literature session, at the user's request via Audit. All arXiv IDs verified (Semantic Scholar batch, 2026-10-02).
Details for most entries are in the dated LIT_* notes and `docs/LITERATURE.md`. This page is the synthesis.

## Field state in one line per area
1. **Latent / continuous thought.** It works at scale as *depth* (looped or recurrent-depth LMs), and on narrow tasks
   with *step supervision* (SIM-CoT, CODI distillation). Under outcome-only supervision the latent tokens are often
   causally unnecessary: shortcuts or bypass (Aswal, Dilgren, Zhang 2512.21711). Latent RL (LEPO, Latent-GRPO,
   SofT-GRPO) improves benchmarks but never certifies what the latents carry.
2. **Length / state tracking.**
   - Recurrent models with input-dependent multiplicative transitions solve S5 to 512 and beyond (DeltaProduct, Howe,
     PD-SSM, NC-SSM).
   - Transformers and pretrained LMs extrapolate only about 2× without scaffolds (Kraus, Zhang looped, Li 2503.02854).
   - Written-carry-over methods (Delethink, PENCIL, InftyThink, MEM1) scale reasoning length with fixed chunks.
3. **Discrete latents / automata from outcomes.**
   - VQ and FSQ give discrete codes, but usually learned by reconstruction, not from outcomes.
   - Automaton induction from final labels is classical (second-order RNNs, RPNI, spectral WFA) and fragile under SGD.
     Reward rarely teaches state (Allchin). Learned solutions often stop at quotients (Zhang & Li cosets).
4. **Composition vs storage in pretrained LMs.** Transformers store facts and compose them in-distribution, but fail
   OOD unless grokked or the composition is seen (Wang 2405.15071; two-hop, Balesni 2411.16353; Yang 2402.16837).
   Apparent lexical generalisation comes from pretraining exposure (Kim/Linzen/Smolensky 2212.10769).

## Top papers and how each relates to our findings
| Paper | Key result | Relation to our findings |
|---|---|---|
| Aswal et al. 2606.12689 | Coconut ProsQA: removing thoughts barely changes accuracy; thought positions have ≈ 0 causal recoverability | Independent confirmation of our **ProsQA shortcut** (proximity heuristic 0.91; deep ProsQA at chance without the cue) |
| Dilgren & Wiegreffe 2604.04902 | Latent tokens often unnecessary on logical tasks; matched GPT-2 GSM8K CoT/Coconut/CODI checkpoints released | Same conclusion as our C1 controls. Their checkpoints are our §68 references |
| SIM-CoT 2509.20317 | Implicit CoT collapses as the number of latent steps grows unless steps are supervised | Matches F1: unlearnable from final answers, exact with per-pass targets (§54b) |
| Zhu et al. (Ouro) 2510.25741 | Looped LMs pretrained at scale use latent depth productively | Where latents *do* work: depth, pretrained in, not grafted on. Contrast with our GPT-2 grafts |
| Ruiz & Gu 2507.02782 | Recurrent length failure = unexplored states; state-passing / noise init fixes it | The **range principle** in other words. Their fix extends the range; ours keeps quantities in range by restarts |
| Aghajohari et al. (Delethink) 2510.06557 | Fixed-size chunks with text carry-over give linear-cost long reasoning under RL | The nearest at-scale analogue of our written-state restarts (§66, §71); fixed chunk size (C5 open) |
| Siems et al. (DeltaProduct) 2502.10297 | S5 at 1 layer, n_h = 4, train 128 → 512, all-position labels | The comparator for §71 (our WC: 1.000 at 512 and 2,048, scaffolded) |
| Howe 2609.18966 | Removing the additive input path flips a Householder RNN from shortcut to exact S5 automaton (1.00 at 16×) | Same direction as §70: multiplicative 10/10 vs additive 0/10 |
| Zhang & Li 2609.29951 | Group-product learners stall at coset (quotient) solutions | Explains our **partial-automaton optima** (§67 table 1/6 at 6 codes) |
| Xu, Hsu & Maleki 1810.11344 | Over-parameterised EM converges globally where exact-size EM gets stuck | The theory anchor for **table-only scaling**: M = 2\|G\| certifies S4/A5/S5, 12 codes for S3 |
| Wang et al. 2405.15071 (+ Balesni 2411.16353) | Transformers learn composition in-distribution but fail OOD composition; two-hop facts compose only if co-seen | Matches **GPT-2 stores but doesn't compose**: §76(B) held-out rows at chance (0.008/0.011), §74 transfer null |
| Allchin 2607.11953 | Reward-only RL: 86/103 permutation automata fail the state probe | Matches §69 (RL doesn't fix acquisition) |
| Lee 2606.07254 | Held-out generator pairs; a projected recurrent model trained on final states is exact to 2²⁰ on S3×S3 | The only shared-parameter, final-state learner that passes a held-out-transition test. Its projection is the inductive bias our shared cells lack |

## Open problems where we could contribute
1. **Shared-parameter acquisition (storage → composition).**
   - Only the free table acquires S4/A5/S5 from final answers. Every shared cell fails, and amortised GPT-2 cells don't
     fill held-out rows.
   - Open: which *structural* bias (orthogonal/permutation-matrix transitions as in Howe and PD-SSM, Lee's projection,
     group-equivariant parameterisation) lets a shared cell acquire from final answers *and* fill held-out (state, op)
     entries.
   - We have the certificate and held-out-row instruments ready. A positive result would be the first shared-parameter
     latent automaton that composes, certified.
2. **Learned restart points (C5) under the range principle.**
   - Every working fix restarts from a written state at fixed k (Delethink, PENCIL, our WC).
   - Open: learning *where* to cut, e.g. with a halting signal tied to in-range-ness, without the scaffold. Recurrent
     halting work (PonderNet, Kuo 2606.29983) doesn't test this.
   - Contribution: learned cut points that keep the certificate.
3. **A shortcut-free causal benchmark for "do latents do work?"**
   - The field disagrees (Aswal; Dilgren; Zhang 2512.21711; SCIT 2608.27265), partly because benchmarks leak shortcuts
     (our ProsQA 0.91 heuristic).
   - We have a shortcut audit, IIA-scored edit tests, depth-matched controls and certificates. Contribution: a
     shortcut-audited generator plus a standard causal test suite.
4. **Transformer / pretrained-LM state tracking past 2× without hand chunking.** The published transformer bests are
   about 2× (Kraus 1.7×, Zhang ≥ 0.92 at 2×). Our scaffolded WC reaches 2,048. Closing this with a learned
   loop/restart (problem 2) on DeltaProduct's protocol would be a SOTA-relevant transformer result.
5. **Exact-likelihood latents beyond toys.**
   - Exact marginalisation beats RL-style updates for discrete-latent acquisition (§69/§70/§76). Latent-RL papers on
     LMs never certify.
   - Open: bring exact-marginal discrete codes to a real step domain, e.g. GSM8K-Aug steps as op codes. Pair with the
     §78 step-count split, so latent vs written-state vs CoT can be compared on length generalisation.
