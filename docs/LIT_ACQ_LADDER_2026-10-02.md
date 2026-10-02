# Prior check for §82: acquisition-ladder hypothesis (final-answer learning climbs quotient structure; A5 has no foothold) (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified.

## Prior
- **Order-60 trio precedent:** Merrill et al. 2404.08819 used A5, A4×Z5 and Z60 (order matched) for *expressivity*
  (depth needed vs length). §82 adds D30 and asks about *learning dynamics*. Liu et al. 2210.10749: solvable groups
  admit Krohn–Rhodes cascade shortcuts.
- **Learning dynamics on groups:**
  - Chughtai et al. 2302.03025: networks learn group composition through irreducible representations, with
    representation order varying by run.
  - Nanda et al. 2301.05217 (progress measures; modular addition); grokking 2201.02177.
  - **Zhang & Li 2609.29951:** learned state tracking passes through **coset** stages. Transformers find only
    normal-subgroup cosets; recurrent nets go through **both normal and non-normal** coset stages. Partial accuracy
    ≈ 1/|class|.
- **Staircase / foothold theory:**
  - Abbe et al. 2202.08658 (merged-staircase property) and 2302.11055 (leap complexity, saddle-to-saddle): SGD learns
    hierarchical targets step by step when each step is reachable from the last, and stalls otherwise.
  - Parity-type hardness: Shalev-Shwartz 1703.07950; Hahn & Rofin 2402.09963.
  - The "foothold" framing is this theory transplanted to group automata. **New as scoped**: no paper tests acquisition
    dynamics across order-matched groups by quotient structure.

## The key correction: A5 does have footholds, just not normal ones
- For the word-problem automaton (state g, update g → g·a), **right cosets Hg of *any* subgroup H** form a
  well-defined quotient automaton, because Hg·a = H(ga). Every transitive G-set is a deterministic coarsening of the
  regular automaton. Normality is needed only for the *quotient group*, not for the *quotient automaton*.
- A5 has proper subgroups giving transitive actions:
  - A4 (index 5, the natural 5-point action);
  - D5 (index 6);
  - S3 (index 10);
  - Z5 (index 12);
  - plus smaller ones.
- So "A5 is simple, hence no foothold" is not implied. A likelihood learner can climb the 5-point or 6-point action
  first, with partial accuracy ≈ 1/|H|: 1/12 for A4, 1/10 for D5.
- Zhang & Li report recurrent learners passing through non-normal coset stages, and a free table is more like an RNN
  than a transformer here.
- **The hypothesis to register should therefore be:**
  - "plateaus correspond to coset spaces G/H (any subgroup), and the acquisition difficulty depends on the subgroup
    lattice's chain of footholds (index steps)";
  - not "normal quotients only".
- **Predictions to compare** (plateau accuracy = 1/|H|):
  - A5 via A4 → 1/12, D5 → 1/10.
  - Z60 via its quotient chain (Z2, Z3, Z4, Z5, …, many small steps).
  - A4×Z5 via Z5 and A4 quotients.
  - D30 via Z2 and the rotation subgroup.
  - Measure the code partition at each plateau against *all* subgroup coset partitions (computable for order 60), not
    just normal ones.
- **Alternative explanation to rule out:** step size, not existence. A5's largest proper subgroup has index 5, while
  Z60 offers index-2 steps. The staircase view predicts difficulty grows with the **smallest available index jump**
  (leap complexity), not with simplicity per se.
- **Design suggestion: separate solvability from footholds.**
  - **Z59 or Z61** (prime order ≈ 60) is abelian, hence solvable, but has **no proper non-trivial subgroups**, so no
    foothold at all. If solvability is what helps, Z59 is easy. If footholds are what help, Z59 is as hard as or harder
    than A5.
  - Optionally add a non-simple group whose smallest index is ≥ 5, e.g. A5 × Z7 (order 420), as a larger-order check.
  - Report each group's minimal subgroup index alongside solvability.
