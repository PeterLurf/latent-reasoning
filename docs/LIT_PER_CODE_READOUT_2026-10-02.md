# Prior check for §77: per-code readout (one linear map per latent code over frozen GPT-2 op features) (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch).

## 1. Prior for state-indexed maps
- **It is a second-order RNN with a one-hot state.** In Giles-style second-order cells (h'_i = σ(Σ_jk W_ijk h_j x_k)),
  a one-hot h = e_c selects the slice W_c, giving logits = W_c x. §77 is exactly that, with x = std(GPT-2 op
  features). Refs: Giles et al. 1992 and Goudreau et al. 1994 (non-arXiv; already in LIT_ACQ_AB). ISAN 1611.09434 is the
  *dual*: matrices indexed by the input symbol, not the state.
- **Switching / regime-indexed linear maps:** switching linear dynamical systems (Ghahramani & Hinton 2000,
  non-arXiv); recurrent SLDS, Linderman 1610.08466 (discrete state selects the linear dynamics).
- **Hard routing by a discrete variable:** sparsely-gated MoE 1701.06538, Switch 2101.03961, hard MoE 1704.06363. In
  §77 the router is the latent code itself: deterministic, one expert per code. Hypernetworks 1609.09106 generate the
  weights instead. Recurrent Independent Mechanisms 1909.10893 select modules per step.
- **Symbolic automata (non-arXiv; D'Antoni & Veanes):** each state's transitions are predicates over a rich input
  space. The per-code readout is a learned, linear-in-features version of that.

## 2. LM or neural input encoder + automaton
- **Neurosymbolic automata:** NeSyA 2412.07331 (symbolic automata plus neural perception of propositions, end-to-end
  differentiable) and Neural Reward Machines 2408.08677 (probabilistic Moore-machine relaxation with semi-supervised
  symbol grounding); reward machines 2010.03950.
  - In this line the neural part typically *grounds input symbols* for a specified or partly specified machine.
  - §77 reverses the roles: the machine is learned from final answers and the encoder is a frozen LM. (Their exact
    split of given vs learned should be checked in the full papers before claiming the contrast in print.)
- **Not found:** "frozen pretrained LM as op encoder + per-state linear transition maps, learned from final answers,
  with a certificate". New as scoped. The architecture class is the classical second-order / switching family.

## 3. Framing risk and the right controls
- **Your reading is right.**
  - If GPT-2's standardised op features φ(c, a) are linearly independent across the |Σ| ops for each code, every W_c
    row can realise any next-code assignment. The model is then a **reparametrised free table**, and its acquisition
    behaviour should match the table's.
  - **Parameter count is larger, not smaller:** M × 768 × M. That is 11M at A5 (M = 120) and 44M at S5 (M = 240),
    against the free table's 0.86M / 6.9M. Report this.
- **Which control:**
  1. **One-hot op features in place of GPT-2's (primary).** W_c over one-hot ops *is* the free table. If PT-per-code ≈
     one-hot per-code on acquisition, GPT-2 contributes nothing to acquisition, which is the honest C1 statement. This
     isolates the feature map with the readout fixed.
  2. **Random-init frozen GPT-2 features.** Also linearly independent generically, so expected to acquire too. It
     tests pretraining only where pretraining could matter (below), not acquisition.
  3. **Code-free op features:** h_op computed without the code in context. This tests whether GPT-2's contextualisation
     of the op by the code matters, or whether only W_c does the state work.
- **Where GPT-2 can contribute, so measure it there:** one-hot features give zero by construction, so any non-zero
  result is GPT-2's.
  - (a) Paraphrase or held-out phrasing agreement (§74 design, with its negative control).
  - (b) **Held-out (code, op) entries.** With independent features W_c is unconstrained on an unseen op, so expect
    ≈ chance unless φ carries group structure. A non-chance result would be a genuine C1 contribution.
  - (c) Sample efficiency to certificate against the free table.
- **Recommended wording if acquisition ties the table:** "the per-code readout lets a frozen LM host the acquired
  automaton (a table reparametrised through LM features). The LM's contribution is limited to the transfer properties
  measured in (a)–(c)."
