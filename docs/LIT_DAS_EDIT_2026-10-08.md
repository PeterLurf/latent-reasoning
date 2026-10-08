# Prior check for DAS-EDIT: distributed alignment search on DeltaMem-R's memory state (§161 column sum; §162 S5 permutation) (2026-10-08)

Literature session, for Audit (the lead) and Theory. IDs verified via arXiv abstract pages; DAS details from its full HTML (2026-10-08).

**The motivation:** every rank-1 *probe-direction* edit has been null, while probes show the state is carried. That's a known gap:
probe-decodable ≠ causally used (Ravichander et al. 2005.00719; amnesic probing, Elazar et al. 2006.00995). A probe finds a
*correlational* direction; DAS optimises the *interchange* objective directly. So DAS can succeed where probe edits fail, and that's
the right next tool.

## 1. DAS, 2303.02536 (Geiger, Wu, Potts, Icard, Goodman): confirmed
- **Method:**
  1. Learn an **orthogonal change of basis R** (by gradient descent; PyTorch's orthogonal parameterisation) for a chosen neural
     representation N.
  2. Split the rotated space into orthogonal subspaces Y₀, …, Y_k, one per high-level causal variable (plus the rest).
  3. **Distributed interchange intervention:** rotate the base and source representations, set the base's Y_j to the source's Y_j,
     rotate back, and run the model.
  4. Train R so the output matches the high-level causal model's counterfactual.
- **Hyperparameters that stay discrete:** N (which representation) and the **dimensionality |Y_j| of each variable's subspace**.
- **IIA (interchange intervention accuracy):** the share of interchange interventions whose output matches the causal model's
  counterfactual. **It's reported on held-out base / source pairs.** The paper finds 100% IIA alignments that brute-force,
  neuron-aligned search misses.
- **Typical k:** small subspaces relative to the representation, chosen per variable and swept. The paper's toy cases use a few
  dimensions out of a small hidden layer.

## 2. Boundless DAS, and DAS / interchange on recurrent or memory states
- **Boundless DAS (Wu et al. 2305.08809):** it replaces the remaining brute-force choices (the subspace boundaries) with **learned
  boundaries** (a soft mask with temperature). It was applied to Alpaca-7B, which on a numerical reasoning task implements two
  interpretable boolean variables, robust to input and instruction changes.
- **Tooling:** pyvene 2403.07809 (intervention library; supports trainable rotated-subspace interventions on arbitrary modules,
  recurrent ones included).
- **On recurrent states specifically:** Csordás et al. 2408.10920 trained interventions on **GRU hidden states**.
  - The smallest RNNs store tokens by **order of magnitude ("onion" representations), not directions**, and these are impossible to
    isolate in linear subspaces. Larger RNNs are linear.
  - Mamba activation patching (2404.03646) works at the state level, without DAS.
- **DAS on SSM / linear-attention memory states: not found.** It would be new here.
- **Note for DeltaMem:** the additive store holds sums of φ(k)vᵀ with φ > 0. An accumulator may well be coded by *magnitude along a
  direction*.
  - A single-value magnitude code is still linearly editable (scale along the direction).
  - But several values sharing one direction at different scales (the onion case) aren't.
  - So a DAS null on the additive heads doesn't rule out a magnitude code: also try Csordás-style scaling interventions.

## 3. Pitfalls and controls
- **Makelov, Lange & Nanda 2311.17030 (the interpretability illusion):**
  - a subspace patch can reproduce the counterfactual behaviour by **activating a dormant parallel pathway**: a component causally
    disconnected from the output in normal operation;
  - it's linked to rank-1 fact editing.
  - Their recommended evidence: check that the subspace is used in the *unpatched* model.
  - **Wu et al.'s reply, 2401.12631:** argues that the illusion is largely mitigated by proper held-out evaluation and appropriate
    subspace sizes.
- **The non-linear representation dilemma, Sutter et al. 2507.08802:**
  - with *unrestricted* (non-linear) alignment maps, **any network can be mapped to any algorithm**, including **randomly initialised
    LMs at 100% IIA** (IOI).
  - **So keep the map linear (orthogonal R)**, and run the random-model baseline.
- **Controls to pre-register:**
  1. **Held-out base / source pairs** for IIA (DAS standard). Fit the rotation on one set of items and score on others.
  2. **A random / untrained-model baseline:** the same DAS on a randomly initialised (or pre-training-checkpoint) DeltaMem gives the
     IIA floor (Sutter).
  3. **A random-subspace control:** the IIA of random orthogonal subspaces of the same k.
  4. **A k sweep (1, 2, 4, 8, …).**
     - **Critical for us:** as k approaches the full state dimension, a subspace swap becomes a *whole-state* swap. Under the Markov
       property that's behaviourally the donor's chain (LIT_FREE_PAIRS), so IIA → 1 by construction.
     - **Only IIA at k ≪ the state dimension is evidence.** Report IIA vs k, and the k where it saturates.
  5. **Necessity (against Makelov):** mean- or zero-ablate the found subspace in *unpatched* runs; the output must degrade.
     Optionally, check that the subspace has a large component in the row space of the downstream read (φ(q)ᵀ S, and the head's
     output projection), not in its null space.
  6. **Propagation (§155's test):** intervene at hand-off t and score the counterfactual at the column after next (multiplication) or
     at the final state (S5). That's the C1-relevant use beyond same-step IIA.
- **Practicality:** DeltaMem's state is 12 heads × d_k × d_v (tens of thousands of dimensions). A full rotation is infeasible. Learn a
  **low-rank orthonormal basis (n × k, Stiefel)**, as pyvene's low-rank rotated interventions do, per head or over the flattened
  heads, with k ≤ 32.

## Verdict
- **DAS on a linear-attention / delta memory state is new.** The method and the controls are well established.
- **It's the right replacement for probe-direction edits.**
- **The most important control here is the k sweep.** At large k it collapses into the whole-state swap, which passes by
  construction.
