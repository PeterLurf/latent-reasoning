# Check of EXTERNAL_IDEAS_C1_2026-10-01: cited priors and Design 1 novelty (2026-10-01)

Literature session, for Audit and Theory. Committed before any Design 1 registration. IDs verified (arXiv API);
non-arXiv items marked.

## Cited priors — verified
| Cited | ID / venue | Status |
|---|---|---|
| State-Regularized RNNs | 1901.08817 (Wang & Niepert, ICML 2019) | ✔ |
| Differentiable finite-state machines | Google Research self-organising-systems blog (2022) | ✔ exists (not arXiv). Probabilistic automaton (softmax transitions, einsum state propagation), trained on outputs with an entropy penalty, then **hard FSM extracted and tested** |
| GraphFSA | 2408.11042 (Grötschla et al. 2024) | ✔ |
| Universal Transformers for Circuit Computations | 2608.31067 (Ito et al., Aug 2026; provably correct 280-parameter parameterisation, perfect length generalisation on Boolean circuits) | ✔ |
| State-Denoised RNNs | 1805.08394 | ✔ |
| Hopfield | 2008.02217 | ✔ |
| VQ-VAE / FSQ | 1711.00937 / 2309.15505 | ✔ |
| Sanford et al. ICML 2024 / CoT expressivity | 2402.09268 / 2310.07923 | ✔ |

## Closer priors the proposal did not cite (Design 1)
- **Input–Output HMMs**, Bengio & Frasconi, NeurIPS 1994 (not arXiv): neural networks parameterise *input-conditioned*
  transition probabilities between discrete states, trained by EM / forward-algorithm likelihood. **Design 1's training
  method is an IOHMM with GPT-2 as the transition network.** This is the must-cite.
- Neural HMMs (Tran et al. 1609.09007); Structured Attention Networks (Kim et al. 1702.00887): exact marginalisation over
  discrete latent structure inside deep nets.
- Automata extraction (Weiss 1711.09576); quantised bottlenecks (Koul 1811.12530).
- State tracking in LMs: Li, Guo & Andreas 2503.02854 (associative scan); Liu 2210.10749.

## Novelty of Design 1
- **The method is classical:** exact forward marginalisation over a learned finite transition table from sequence-level
  labels, then hard execution and transition extraction (IOHMM 1994; diff-FSM 2022; State-Regularized RNNs).
- **New if scoped as:** a *pretrained GPT-2* as the transition function of a 6-state categorical latent on S3 under a
  streaming interface, with hard GPT-2 execution, a complete 36-transition audit (an arbitrary-length certificate), and
  the proposal's causal controls (donor and irrelevant swaps, early budgets, resets).
- The proposal's own caveats are right:
  - the **tiny table/MLP control** is essential, since IOHMM-style tables learn S3 trivially;
  - the **full-prompt GPT-2** baseline is needed for the stronger claim. S3 is solvable, and LMs learn associative scans
    (2503.02854), so a full-prompt baseline may succeed at moderate lengths.

## Designs 2 and 3 (brief)
- **Design 2's nearest work is Ito et al. 2608.31067:** provably correct, perfectly length-generalising circuit
  evaluation in tiny transformers, learning from shallow examples. Also GraphFSA. Claim it narrowly, as the proposal
  says.
- **Design 3's nearest:** State-Denoised RNNs 1805.08394 (clean-up trained on corrupted self-states); State Reification
  1905.11382.
