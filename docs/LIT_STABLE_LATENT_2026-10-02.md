# Prior check: "stable latent" candidates (a) acquire then amortise into GPT-2, (b) restart selection (2026-10-02)

Literature session, for Theory; Audit copied. Committed before registration. arXiv IDs verified (Semantic Scholar
batch). Classics without arXiv IDs are marked.

## (a) Teacher automaton (free table / spectral WFA) → GPT-2 transition cell
**Prior work.**
- **WFA ↔ RNN equivalence:** Rabusseau, Li & Precup 1807.01406 (AISTATS'19). Linear second-order RNNs compute exactly
  the WFA functions, and spectral learning extends to learning linear 2-RNNs. So a spectral WFA *is* a
  (linear, second-order) neural cell. Moving it into a multiplicative cell is close to a re-parameterisation, not a
  distillation.
- **Distillation runs the other way:** RNN → WFA (Eyraud & Ayache 2009.13101, spectral) and RNN → DFA (Weiss
  1711.09576, L*). Transformer → DFA: Zhang et al. 2406.05564. Mechanistic FSA inside CoT transformers: Zhang et al.
  2502.20129.
- **Automaton → neural (rule insertion):** Omlin & Giles 1990s, programming DFA rules into second-order RNN weights
  (non-arXiv; journal). Tracr 2301.05062 compiles programs into transformer weights.
- **Imitating a symbolic teacher's step transitions:** Neural Algorithmic Reasoning 2105.02761 and CLRS 2205.15659
  (neural processors supervised on classical algorithms' intermediate states). Expert iteration: Anthony 1705.08439
  ("acquire with a slow reliable learner, amortise into a fast network"). McCoy & Griffiths 2305.14701 distil Bayesian
  priors into networks.
- **The student side is already standard:** training a transformer on *per-step state labels* for automata (Liu et al.
  2210.10749, shortcuts to automata; Li, Guo & Andreas 2503.02854, pretrained Pythia/GPT-2 with all-prefix S5 labels).
  Stage 2 is that, with the labels coming from a learned teacher instead of the generator.
- **Not found:** the pipeline "final-answer-only acquisition (exact marginalisation or spectral WFA) → pretrained-LM
  student cell with an exhaustive certificate" for S3/S4/A5/S5. **New as a pipeline.** Neither stage is new alone.

**Framing risks.**
1. **Cayley-table memorisation.** With codes as embeddings and ops as tokens, Stage 2 learns a finite lookup
   |Q| × |Σ| (S3: 6 × |Σ|; S5: 120 × |Σ|), and every entry appears in training. Exactness plus snapping then gives
   length generalisation by construction.
   - So the certificate certifies *imitation*, not acquisition.
   - Optional test that would mean more: hold out (state, op) pairs from Stage-2 training and check whether GPT-2 fills
     them. That requires structure, e.g. ops as compositional text such as "swap 2 4".
2. **C1 "from final answers only"** holds for the *system*. GPT-2 itself is trained on teacher-supplied intermediate
   states. Honest wording: "the system acquires from final answers; GPT-2 hosts the acquired automaton."
   - That claim is weaker than GPT-2 acquiring the automaton itself. It is still defensible if the teacher never sees
     intermediate labels and its states are unsupervised (not identified with the true permutation).
3. **What the LM adds.**
   - At test the teacher table is strictly better, being smaller and exact.
   - Justify the student by C3–C7: readable codes, edit tests, running inside the LM's forward pass.
   - Report teacher-alone as an arm. Otherwise Audit will ask why GPT-2 is needed.
4. **Teacher identifiability.** A spectral WFA is identified only up to change of basis; the free table only up to
   state relabelling. Use the minimal-DFA quotient as the code set and certify against it, not against the true
   group labels.

## (b) Best-of-n restarts selected by loss, counted as one seed
**Prior.**
- Random restarts are standard practice in EM/HMM training: pick the highest likelihood. Seed lotteries appear in
  algorithmic and fine-tuning work:
  - Zhang 2607.20594: 8/22 from scratch on A5.
  - Dodge et al. 2002.06305: BERT fine-tuning varies widely across seeds; bad seeds can be spotted and stopped early
    from early validation loss.
  - Mosbach 2006.04884: instability is an optimisation issue, fixable by longer training and bias correction.
  - Picard 2109.08203 (a seed-selection critique).
- Lottery Ticket (Frankle 1803.03635) concerns subnetworks and initialisation, not restart selection. Don't cite it for
  (b).

**Reporting standard.**
- **Dodge et al. 1909.03004 (Show Your Work):** report *expected best validation performance as a function of the
  number of trials n*, since best-of-n is legitimate model selection when the budget is stated. Directly applicable:
  report the certification rate of best-of-n for n = 1, 2, 4, 8 from the same pool of runs (bootstrap over runs).
- **Henderson 1709.06560, Bouthillier 2103.03098, Agarwal 2108.13264:**
  - report all runs, plus variance over seeds and selection;
  - use enough independent seeds;
  - give interval estimates such as IQM with bootstrap CIs, not cherry-picked bests.

**Rules to pre-register so (b) is fair.**
- The selection criterion uses training or validation loss only, never the certificate, held-out test, or OOD lengths.
  The certificate stays the outcome.
- A "seed" is the full best-of-n procedure, with disjoint restart seeds per outer seed. State n and the total compute
  (n×).
- Also report the per-restart certification rate p₁. If restarts are independent and selection were perfect,
  P(best-of-n certifies) = 1 − (1 − p₁)^n. If the observed rate is near that bound, loss selection works; if it falls
  well below, loss doesn't track the certificate.
  - §69's 0–1/10 → 5/10 from best-of-4 is consistent with p₁ ≈ 0.15 and good selection.
  - Even then, ≥ 9/10 needs n ≈ 13 at p₁ = 0.15 (1 − 0.85¹³ ≈ 0.88; 0.9 at n ≈ 14). The budget follows from p₁.
- Report the loss-vs-certificate rank correlation across restarts. That is the evidence that selection is
  label-free.

**Bottom line.**
- (b) is accepted practice if reported in Dodge's style (curve over n, compute stated, selection blind to the
  certificate). It is a reliability procedure, not a method claim.
- (a) is new only as a pipeline. Its C1 claim must be worded at the system level, with a teacher-alone arm.
