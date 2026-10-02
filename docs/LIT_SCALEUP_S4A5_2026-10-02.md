# Prior check for §75: scaling the certified frozen-GPT-2 latent cell to S4, then A5 (2026-10-02)

Literature session, for Theory; Audit copied. Very short note. arXiv IDs verified (Semantic Scholar batch).

## 1. New since 2026-10-01, and the final-answer-only line
- **Zhang & Li 2609.29951 (2026-09-24), "Tracking States or Tracking Cosets?":** transformers and RNNs trained on group
  products, including A5, often learn **quotient (coset) solutions**. They recover the coset and predict nearly
  uniformly within it, with partial accuracy ≈ 1/|class|. Transformers find only normal-subgroup cosets; RNNs pass
  through non-normal coset stages during training.
  - **Directly relevant to the A5 step.** Expect plateaus at accuracy levels set by subgroup sizes. A plateau there
    means a coset solution, not noise. Log the code partition against A5's subgroup lattice.
- **Lee 2606.07254:** a held-out *transition-pair* falsifier. Selected ordered generator pairs are forbidden in
  training and required at test. A projected recurrent model trained on length-8 **final states** is error-free to
  2²⁰ steps on S3×S3 (5 seeds); GRU and SSMs stay near chance.
  - This is the nearest final-answer-trained neural prior, but on a solvable group, with a hand-built projection and
    no certificate.
  - It also answers the Cayley-table concern from LIT_STABLE_LATENT: hold out transition pairs.
- **Jia 2610.00329 (NC-SSM, 2026-09-29):** Lie-group SSMs reach 100% on S3, D4, Q8 and A5 tracking. Per the abstract,
  4× extrapolation; no certificate.
- **Allchin 2607.11953:** reward-only RL on hidden automata. **86/103 permutation automata fail a state probe** whatever
  the reward. A warning that final-answer or reward signals rarely teach state in permutation automata.
- **Neural DFA learning from final labels** (accept/reject): second-order RNNs (Giles et al. 1992, non-arXiv) and
  DeepDFA 2408.08622. Both extract automata on small regular languages, not 24–60-state groups.
- **Not found:** a neural or LM-hosted machine that learns S4, A5 or S5 **from final answers only** with an exhaustive
  certificate. DeltaProduct, Grazzi, Merrill, Howe, PD-SSM and Li 2503.02854 all use per-position labels. Zhang
  2607.20594 uses curriculum and exact match. None certifies.

## 2. Comparator and novelty framing
- **No published neural number matches the protocol** (final answers only, certificate).
- Report two kinds of comparator, labelled by supervision:
  - **Non-neural, same supervision:** your spectral WFA (69-2; Balle et al. spectral WFA learning). It is exact and
    sets the bar for sample count.
  - **Neural, stronger supervision:**
    - Zhang 2607.20594: A5 from scratch 8/22 seeds; operator-first 100% of attempted seeds. Exact match, curriculum.
    - Howe 2609.18966: A5 4/5 seeds when zeroing W_b restores generalisation; 1.00 at 512 for the −b cells.
      Per-position labels.
    - DeltaProduct: A5 at n_h = 2, train 128 → 512, per-position.
- **The proposed framing is fair, with three qualifiers:** "the first certified neural A5 automaton learned from final
  answers only, hosted by a frozen pretrained LM".
  1. **"Hosted by a frozen pretrained LM"** should read **"frozen GPT-2 features with learned code embeddings and a
     standardised bilinear readout".** The transition is computed by the trained head on GPT-2 features. Your toy
     one-hot bilinear cell (10/10 on S3) is the control that shows what GPT-2 adds; run it at A5 too.
  2. **Training-length disclosure.** If training includes length-1 problems ("code + op → next code"), final answers
     directly label Cayley-table entries, and "from final answers" reduces to per-transition supervision.
     - State the length distribution.
     - The strong claim needs problems of length ≥ 2 only, from the identity or a fixed start, so that no single
       transition is labelled.
     - Or add Lee's held-out transition-pair split.
  3. **Non-neural precedent:** spectral WFA already does A5 from final answers exactly. The claim is about the neural,
     LM-hosted cell, not about acquisition being possible.
- **Reliability bar:** ≥ 9/10 seeds, with seeds counted as in LIT_STABLE_LATENT (b) if restarts are used. That is
  directly comparable to Zhang's 8/22 from scratch, the only published from-scratch A5 seed rate found.
