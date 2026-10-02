# Prior check for §81: doubly-stochastic (Sinkhorn) transitions against quotient traps and for held-out entries (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified.

## Prior
- **Sinkhorn relaxations of permutations:** Adams & Zemel 1106.1925 (Sinkhorn propagation); Mena et al. 1802.08665
  (Gumbel-Sinkhorn, with temperature τ → 0 approaching hard permutations); Linderman et al. 1710.09508 (Birkhoff-polytope
  reparameterisations). Birkhoff–von Neumann: doubly-stochastic matrices = the convex hull of permutation matrices.
- **Norm-preserving or permutation-structured recurrent transitions:** uRNN 1511.06464; expRNN 1901.08428 (orthogonal
  parameterisation); PD-SSM 2509.22284 (one-hot column × diagonal); Howe 2609.18966 (pure Householder transitions
  without additive input give exact S5).
- **Doubly-stochastic HMM transitions:** I found no standard line of HMM work that constrains transitions to the
  Birkhoff polytope for automaton learning. **New as scoped** for final-answer acquisition of group automata.

## Two flags that matter for the predictions
1. **Doubly-stochastic does not exclude quotient traps.**
   - A coset (quotient) solution that predicts uniformly within a class corresponds to block-averaging matrices.
     Example: J/|H| blocks composed with a permutation of the classes.
   - Those are doubly stochastic, so they lie inside the Birkhoff polytope (Zhang & Li 2609.29951's within-coset
     uniform solutions).
   - Sinkhorn removes only *non-injective* traps: merging states, unequal in-degree.
   - To exclude coset traps the transitions must be pushed to the **vertices**:
     - Gumbel-Sinkhorn with τ annealed to 0 (Mena);
     - an entropy penalty on the rows;
     - or a hard assignment at test with a certificate under hard permutations.
   - Pre-register which one, and report the row entropy at plateaus.
2. **Held-out-entry recovery under a bijection is constraint completion, not generalisation.**
   - If a (code, op) entry never occurs in any training path, the only information about it is the bijectivity
     constraint.
   - With exactly one missing row per op, that row is forced: the single unused column. With h ≥ 2 missing rows in the
     same op, the completion is ambiguous (h! ways) unless training paths pass through those entries.
   - Pre-register the held-out design: missing rows per op, and whether held-out entries appear inside longer training
     paths. Report recovery by h.
   - "Recovered with h = 1" should be called **constraint-forced**, not compositional.
- **Size consistency:** a doubly-stochastic matrix needs a square M × M. With over-complete M = 2|G|, the true solution
  is a permutation on 2|G| codes (two copies of the regular representation, or one copy plus unused codes forced into a
  permutation among themselves). Check that the certificate's minimal-DFA quotient treats duplicated copies correctly.
