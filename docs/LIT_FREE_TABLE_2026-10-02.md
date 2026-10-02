# Prior check for §76: over-complete free-table latent automaton (exact marginalisation, final answers) at S4/A5/S5; amortisation into GPT-2 (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch); non-arXiv
classics are marked.

## 1. Prior for the method
- **What it is:** a tabular probabilistic automaton (IO-HMM / PFA with a deterministic start and an emission decoder)
  fitted by gradient ascent on the **exact marginal likelihood**. That is Baum–Welch / GEM by gradient.
  - Classic references (non-arXiv): Bengio & Frasconi 1995, IO-HMM (NIPS); Rabiner 1989 (Baum–Welch); the PAutomaC PFA
    learning competition (Verwer et al., *Machine Learning* 2014), where EM-style learners were competitive.
  - Differentiable WFST machinery: GTN, Hannun 2010.01003.
- **Over-parameterisation helping EM escape local optima:**
  - Xu, Hsu & Maleki 1810.11344 (NeurIPS'18): over-parameterised EM for Gaussian mixtures converges globally where the
    exact-size model gets stuck. This is the theory anchor for M = 2|G|.
  - Chiu & Rush 2011.04640: HMMs with very large state spaces train well.
  - Petrov et al. 2006 (non-arXiv): split–merge latent states.
  - Your earlier result, 12 codes certifying 10/10 at S3, fits this pattern.
- **Non-neural acquisition from labelled strings:** RPNI (Oncina & García 1992, non-arXiv; passive DFA learning) and
  spectral WFA (Balle et al. 2014; your 69-2). Exact learning is classically possible with enough data.
- **Tabular learning of group tables, but with direct labels:** grokking (Power et al. 2201.02177) learns S5 x∘y
  composition from *pair* labels, i.e. direct Cayley entries, and generalises to held-out entries only through shared
  parameters.
- **Not found:** EM or exact-marginal-likelihood learning of a permutation automaton for **A5 or S5 from final answers
  only, with a certificate and a seed-reliability panel.** The method is classical; **the empirical result is new as
  scoped**: reliable, certified acquisition of a non-solvable group by a local likelihood learner.

## 2. Fair framing and qualifiers
- Supportable claim: **"a gradient-trained over-complete tabular latent automaton, by exact marginal likelihood, acquires
  A5/S5 from final answers in ≥ 9/10 seeds with an exhaustive certificate."** Qualifiers:
  1. **Name the model class honestly.** It is a tabular PFA/IO-HMM with no shared parameters (864k at A5, 6.9M at S5),
     not a neural network. Comparisons with Zhang 2607.20594 (8/22) and Howe (4/5) cross model class *and*
     supervision. Label them "neural, per-position or curriculum supervision". The same-class, same-supervision
     comparator is the **spectral WFA** (exact, non-iterative).
  2. **M = 2|G| uses |G|, which is oracle knowledge.**
     - Report an M sweep, e.g. M ∈ {|G|, 1.5|G|, 2|G|, 4|G|}, or a rule that doesn't need |G|. Example: grow M until
       validation likelihood plateaus. Spectral WFA picks rank from the Hankel spectrum without the oracle.
     - Certify against the minimal-DFA quotient (codes merged by behaviour), as in LIT_STABLE_LATENT.
  3. **Answer = full final group element** (24/60/120-way). That is full-state supervision at the last step. Say so. A
     weaker observation ("which box holds X", 5-way) is the harder, more realistic variant. The free table can be run
     on it too, as a stretch arm.
  4. **Local-optimum framing.** The reliability claim is about the likelihood landscape under over-parameterisation and
     annealing (τ 4→1). Report the τ-ablation and M = |G| failures so the over-completeness effect is shown, not
     assumed.
  5. **Sample and compute accounting.** Report the problems seen to certification against spectral WFA's ≤ 10⁶.

## 3. Strong variant: lengths ≥ 3, or held-out transition pairs?
- **For the free table (A), "lengths ≥ 3 only, from the fixed start" is the right strong variant.** The only shortcut a
  table can exploit is a final answer that labels one entry directly (length 1, and length 2 from a fixed start). With
  lengths ≥ 3 every label is a composition, so entries must be inferred.
- **Lee 2606.07254's held-out *ordered generator pairs* does not bite on a free table.** Every entry T[c, a] still gets
  trained whenever state c meets op a. The falsifier targets models that memorise local *bigram* patterns, which a
  table cannot exploit anyway. Holding out (code, op) *entries* is impossible for a free table by construction: an
  entry never seen stays untrained.
- **For amortisation (B), use held-out (code, op) entries.** The GPT-2 cell has shared parameters, so a held-out entry
  tests whether the student composes instead of copying the teacher's table (the Cayley-memorisation concern from
  LIT_STABLE_LATENT). Hold out a fixed fraction (e.g. 10%) of teacher entries and certify agreement on them separately.
  Lee's pair falsifier fits (B) too, as a secondary check.
- **Also report a lengths-≥ 3 panel with a random start state** (if the start is a learned or given code), or at least
  confirm that reachable-set coverage from the fixed start is complete at each N.
