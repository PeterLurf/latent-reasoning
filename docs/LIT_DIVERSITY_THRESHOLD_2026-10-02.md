# Prior check for §90: is §88's looped-GPT-2 composition a word-diversity threshold? (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch / arXiv API).

## 1. Prior on diversity or data-size thresholds for generalisation
- **Raventós et al. 2306.15063.** A task-diversity threshold separates memorising (Bayesian over the pretraining tasks)
  from generalising in-context regression.
- **Wang et al. 2405.15071.** Grokked implicit composition depends on the *ratio* of inferred to atomic facts, not on
  raw data size, and OOD composition fails.
- **Grokking critical data size.** Power 2201.02177 (a critical training *fraction* of the S5 composition table);
  Omnigrok 2210.01117; Zhu et al. 2401.10463 (a critical data size marks the switch from memorisation to
  generalisation, and it grows with model size).
- **Compositional generalisation and data:**
  - Patel et al. 2203.07402: modest changes to the training distribution make seq2seq models near-perfect on SCAN
    primitive generalisation.
  - Zhou et al. 2311.04420: more complex, diverse datasets improve compositional generalisation by giving diverse
    examples *and* by reducing repetition-driven memorisation.
  - Oren et al. 2109.02575: structurally diverse training sets.
  - Okawa et al. 2310.09336: compositional abilities emerge multiplicatively with component exposure.
- **Coverage theory:** Wiedemer 2307.05596 (compositional generalisation needs training support covering each
  component). Abbe 2301.13105 (with limited support, models learn minimum-degree interpolators).

## 2. Looped / recurrent models and a minimum number of distinct compositions
- **Not found** as a stated threshold.
- **Nearest:**
  - RELAY 2502.08482 (iteration-aligned step supervision gives length generalisation).
  - Zhang 2607.20594 (operator-first curriculum: train the shared step on single operations before composing;
    reliable on S5/A5).
  - Saunshi 2502.17416 (looping = implicit CoT).
  - Fan 2409.15647 (looped length generalisation).
  - None measures the number of distinct compositions needed.

## 3. A confound in §90 as designed: words vs transition-pair coverage
- With a **shared** step and per-iteration supervision, iteration i learns transitions (state before i, op_i) →
  state. What an unseen word needs is that **each of its transitions** was seen at that iteration (or anywhere, if
  the weights are tied). The word count alone doesn't measure this.
  - **§88, k = 2 (25 of 36 S3 words):** the second iteration's (state, op) pairs are exactly the 25 seen words. Every
    unseen word contains an **unseen transition**, so failure (0.06) is §76(B)'s held-out-row result (storage, not
    composition), not a diversity shortfall.
  - **§88, k = 3 (151 of 216):** the (state, op) pairs at iterations 2–3 almost surely cover all 36. Unseen words are
    new *orders* of seen transitions, which is sequence-level novelty only.
  - **(ii) S4, k = 2 (576 words total):** each two-op word *is* one second-iteration transition. Every unseen word is
    an unseen transition **at any training size**, so the curve over 25/100/400 measures held-out-transition
    generalisation (expected ≈ chance, per §76(B)), not a diversity threshold.
- **Fix: control coverage explicitly** (Lee 2606.07254's held-out transition-pair logic).
  - (a) Few words that **cover all (state, op) pairs** (e.g. S3, k = 3, 25 words chosen for full transition coverage)
    vs 25 random words. This separates word count from coverage.
  - (b) Report, for each test word, whether all its transitions were seen in training. Score accuracy separately for
    "all transitions seen" and "≥ 1 unseen transition".
  - (c) For a genuine held-out-transition test, hold out specific (state, op) pairs at every iteration. Chance there
    reproduces §76(B); above chance would be real generalisation of the step.
- **Expected reading:** §88's success is composition of *seen* transitions in new orders (sequence-level). That is
  still the C1-relevant in-pass result §80 lacked, because §80 used end-state targets. It is not transition-level
  generalisation.
