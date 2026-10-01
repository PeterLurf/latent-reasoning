# Pre-registration novelty check: PROPOSAL_BOUNDARY_MEMORY_2026-10-01 (external D1, detailed) (2026-10-01)

Literature session, for Audit and Theory. Committed before any Phase A registration. IDs verified (arXiv API or
Semantic Scholar).

## Nearest work by component

| Component | Nearest prior work | What they show |
|---|---|---|
| Learned merger of region summaries | **Divide and Conquer Networks** 1611.02401 | Learned scale-invariant *split* and *merge* applied recursively, trained from input–output pairs (or reward), with a complexity regulariser |
| Order consistency (associativity) | **Learnable Commutative Monoids** 2212.08541 (Ong & Veličković); **MLP-LDRU** 2605.26035 | Learnable associative/commutative binary operators with regularisers penalising violations, giving an O(log V) aggregator. MLP-LDRU: associativity-biased parallel-reduction operators reach **100% OOD length generalisation on 18/21 regular-language tasks** |
| Boundary-response (port) interfaces | Kron reduction / Schur complement 1102.2950 (exact algebra); **Mosaic Flows** 2104.10873; **LatentDDM** 2609.03069 | Mosaic Flows: a subdomain solver over arbitrary boundary conditions, assembled iteratively with overlapping subdomains (alternating Schwarz). LatentDDM (Sept 2026): a learned local operator plus a learned composition module; error 36–56% lower on larger domains than full-domain models |
| Learned graph reduction / path aggregation | Neural coarsening 2102.01350; NBFNet 2106.06935 | (cited by the proposal) |
| Public-state-only reasoners | Markovian Transformers 2404.18988 (all information through text); Delethink 2510.06557; MEM1 2506.15841; PENCIL 2503.14337; our CLR (written checkpoints with isolation) and THEORY F1 (public per-token labels) | — |
| Full-state language round trip with semantic edits | NLA (activation-level, reconstruction plus ~50% edits); our C7 protocol | **No reasoning system with a complete persistent-state round trip found** |
| Predictive / response states | PSRs (2001), PSRNN 1705.09353 | (cited by the proposal) |

## Verdict
- **New as a combination**, as the proposal itself scopes it: an exclusively public, quantised boundary-response state;
  a learned order-consistent merger; a complete-state controlled-language round trip; and age-equivalence tests against
  a diagnosed count-keyed fallback.
- **Not new as components.** Learned composition (DCN), associativity regularisation (LCM, LDRU) and boundary-condition
  composition (Mosaic Flows, LatentDDM) all exist. The proposal's §17 should add **DCN 1611.02401, LCM 2212.08541, MLP-LDRU
  2605.26035, Mosaic Flows 2104.10873 and LatentDDM 2609.03069**.

## Implications for the Phase A registration
- **Directional prediction available:** MLP-LDRU's result (associativity bias → length generalisation) predicts the
  order-consistency arm beats the no-consistency arm. Register it as directional, not two-sided.
- **Known failure modes from the priors:**
  - learned associativity is hard to get exactly (LCM needs explicit regularisers; report the residual associativity
    error);
  - DCN needed a complexity regulariser under weak supervision;
  - Mosaic Flows needed *overlap* between subdomains for consistent boundaries. That bears on shared-port wiring, so
    consider an overlap ablation.
- **Ceiling and controls:** the exact Kron/Schur composer is the right diagnostic ceiling, as proposed. The proposal's
  float-vs-quantised and sham-merger arms match the steganography and C1 concerns in `LIT_NORTHSTAR`.
