# Prior check for FREE-PAIRS: training on deliberately inconsistent digit-pair sequences (data-level IIT) (2026-10-08)

Literature session, for Audit (the lead after §155 R s1: exact 1.000, not SERIAL) and Theory.
- **IDs:** verified via arXiv abstract pages (2026-10-08).
- **IIT details:** from its PDF.

## 1. IIT, 2112.00826: confirmed, plus the one result that matters most here
- **"Inducing Causal Structure for Interpretable Neural Networks"**, by Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa
  Kreiss, Thomas Icard, Noah D. Goodman and Christopher Potts.
- **IIT:** align causal-model variables with neural representations, and train the network so that **interchanging those
  representations** (base ← source) reproduces the causal model's counterfactual output. A zero IIT loss guarantees causal
  abstraction.
- Tasks: MNIST-PVR, ReaSCAN, MQNLI.
- **The paper compares IIT with *data augmentation built from the same causal model*.** That baseline creates hybrid inputs (MNIST-PVR:
  swap a quadrant from a source into the base; MQNLI: swap a phrase) labelled by the causal model. **That's exactly FREE-PAIRS's
  construction.**
- **Its result:** data augmentation gives **excellent, even perfect, behavioural accuracy but "very low interchange intervention
  accuracy"** (MQNLI), and it doesn't reach IIT's alignment elsewhere.
  - Their wording: "IIT involves interchanging internal network representations, while data augmentation methods only involve the
    creation of inputs."

## 2. What this means for FREE-PAIRS (an important nuance)
- **For a *whole-state* swap in a Markov chain, data augmentation and the intervention coincide.**
  - Swapping the entire carried state after segment t is behaviourally identical to running the recipient on the hybrid input (the
    donor's prefix, then the recipient's suffix), because the state is a function of the prefix.
  - FREE-PAIRS puts such hybrids in the training distribution, so **whole-state SERIAL becomes an in-distribution accuracy test. It
    will pass if training succeeds, almost by construction.**
- **The IIT caveat bites at the *sub-state* level.**
  - Data augmentation doesn't make specific internal variables (the carry, a column accumulator) align with the causal model's
    variables.
  - So after FREE-PAIRS, **C1 must be read from partial interventions:** §155's rank-1 carry edit, DAS-style aligned subspaces
    (2303.02536), or per-head / per-key swaps. Whole-memory donor consistency is no longer informative evidence of mechanism.
- **Pre-register that reading.** Otherwise a FREE-PAIRS pass on whole-state SERIAL reads as "C1 met" when it only shows "trained on
  hybrids".
- **If a mechanism-level guarantee is wanted, IIT proper is the stronger tool:** an interchange loss on the memory (or the aligned
  carry subspace) between two items. It needs a differentiable swap op, which the Markov memory already has.

## 3. Precedents for training on inconsistent or recombined sub-problems
- **Counterfactually augmented data:** Kaushik et al. 1909.12434 (minimal label-changing edits; better robustness out of
  distribution).
- **Recombining fragments to force compositional structure:** GECA, Andreas 1904.09545 (good-enough compositional augmentation;
  improves SCAN-style compositional generalisation).
  - The closest *sequence*-level precedent: valid fragments in new combinations, labelled consistently.
- **Shortcut learning in general:** Geirhos 2004.07780. Pointer Value Retrieval 2107.12580 (tasks built to expose shortcut vs
  compositional solutions; the source of IIT's MNIST-PVR).
- **Arithmetic data design:** Lee 2307.03381 balances carries and digit counts (*distribution* design, still consistent problems).
  Neural GPU curricula 1511.08228 / 1611.00736 grow length, but problems stay consistent.
- **Not found:** arithmetic / algorithmic training on *deliberately inconsistent* sub-problems (independent partial products, each
  labelled by the actual sum) to break operand recovery. **New in this form.**
  - The closest are IIT's data-augmentation baseline (hybrid inputs, but not arithmetic) and GECA.

## 4. Evidence on interchange metrics and length generalisation
- **IIA:** the direct evidence (2112.00826) is **negative for data augmentation at the sub-variable level** (high behaviour, low IIA).
  It's positive for IIT proper.
- **Compositional / out-of-distribution generalisation:** positive for recombination augmentation (GECA on SCAN; Kaushik on robustness).
- **Length generalisation:** no evidence found either way for arithmetic.

## 5. Design flags
1. **The p = 0.5 mixture changes the task.** Independent pairs make the sequence a pure *stream accumulation* (Σ of arbitrary products,
   as RUNNING-SUM), not multiplication.
   - Keep the consistent half, and report accuracy on consistent items separately. That's the target task.
2. **Ground truth for "follows neither"** (§155 s1): after FREE-PAIRS, re-run the §155 swap. If the answer now follows the donor
   prefix + recipient suffix, that's expected by construction (section 2). The informative test is the rank-1 carry edit with its
   nulls.
3. **The shortcut may move.** With operands not recoverable from consistency, the model could store *the pairs themselves* (all
   ≤ 16 digits) and sum at the end: capacity allows it.
   - Probe the memory for raw pair digits vs the accumulator. In FREE-PAIRS the accumulator is the minimal sufficient state, but
     storing pairs isn't excluded.
