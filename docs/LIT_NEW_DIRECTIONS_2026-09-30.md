# Seeds for genuinely new directions — adjacent fields not yet mined (2026-09-30)

Literature session, for Theory's step-back proposal round (requested by the user via Audit); Audit copied. Fields not
covered in `IDEAS_ADJACENT_2026-09-29.md`. arXiv IDs verified (Semantic Scholar or arXiv); non-arXiv classics are
standard references, not API-verified. "Novelty" = whether I found it applied to latent / continuous-thought reasoning.

| # | Idea (field) | Source | What it would give a latent reasoner | Target | Novelty in latent reasoning |
|---|---|---|---|---|---|
| 1 | **Vector-symbolic set state with cleanup** (hyperdimensional computing) | Kanerva 2009 (Cogn. Comput. 1:139); Plate 1995 HRR (IEEE TNN 6:623); resonator networks 2007.03748; attention-as-binding view 2512.14709 | The thought is a *bundle* of bound node hypervectors, with a known crosstalk ≈ √(n/d) capacity law and a cleanup memory that restores each item. A principled set state, where CLR-1's 3-anchor set lost reach | Sets, C3, C7 (items decodable by unbinding) | Not found (2512.14709 is a lens on attention, not a latent loop) |
| 2 | **The latent loop as an iterative decoder** (error-correcting codes) | Learned unrolled belief propagation, Nachmani et al. 1607.04793; ECOC (Dietterich & Bakiri 1995, JAIR 2:263) | The thought is a redundant codeword over node ids; each step is message passing that *corrects* errors, using the graph's edges as parity checks. Converges to a fixed point (count-free); "syndrome = 0" is a natural halt | C2, C3, C5 | Not found |
| 3 | **An explicit instruction-pointer register** (program execution) | IPA-GNN 2010.12621 (soft instruction pointer; extrapolates to longer programs); Neural Programmer-Interpreters 1511.06279 (call/return = learned per-subroutine halt) | Make the pointer you discovered a *dedicated* state (a distribution over node ids, updated by attention), separate from the scratch residual. Call/return as the checkpoint/halt policy | C2, C5, C7 | Not found for latent CoT |
| 4 | **Macro-thoughts and test-time latent program search** (program synthesis) | DreamCoder 2006.08381 (library learning, wake–sleep); Latent Program Network 2411.08706 (gradient search in a latent program space at test time) | Learn reusable k-hop "macro" operators that compose; at test time, optimise the checkpoint latent by gradient against a consistency objective (an inference scheme, no new weights) | C2, C4 | Not found |
| 5 | **Search over checkpoints** (search and planning) | MuZero 1911.08265 (search in learned latent dynamics); MCTSnets 1802.04697; Value Iteration Networks 1602.02867 (count-free planning by repeated operator); Searchformer 2402.14083 (search-dynamics bootstrapping shortens traces, a ladder precedent) | Treat each written checkpoint as a search node: expand the top-k checkpoints, score them with a learned value head, keep the best. Replaces "one anchor right ≈ 1/W" with search | C2, C5 | Not found for latent loops (Multiverse/ToT are text-level) |
| 6 | **Restart schedules with guarantees** (SAT solving, randomised algorithms) | Luby, Sinclair & Zuckerman 1993 (IPL 47:173; universal restart sequence optimal for unknown run lengths); iterative deepening, Korf 1985 (AI 27:97) | A *non-learned* restart and halt policy with optimality guarantees: Luby-scheduled episode lengths, or iterative deepening of segment depth. A strong baseline any learned C5 policy must beat | C5 | Not found |
| 7 | **Norm-preserving advance plus cleanup** (dynamical systems) | Deep Koopman embeddings 1712.09707 (linear, time-invariant dynamics in learned coordinates); unitary RNNs 1511.06464; Neural ODEs 1806.07366 (solver-chosen steps) | Constrain the advance operator to unit-modulus spectrum on the pointer subspace (no fade, no count), then a nonlinear cleanup (cf. #1). An ODE form gives compute adaptive to solver tolerance | C3, C6, C5 | Not found for latent CoT |
| 8 | **Successor-representation state** (neuroscience) | Dayan 1993 (Neural Comput. 5:613); hippocampus as predictive map, Stachenfeld et al. 2017 (Nat. Neurosci. 20:1643); TEM, Whittington et al. 2020 (Cell 183:1249); transformers ↔ hippocampal formation 2112.04035 | The thought holds discounted multi-step reachability, Σγ^k T^k, rather than a single frontier. Iterating refines it toward a fixed point (count-free, idempotent), and reachability is read off directly. TEM-style *structural* position codes are a content-tied clock | C2, C4, sets | Not found for latent CoT |

**Cheapest first tests.**
- #6: an inference-only Luby/ID restart schedule on existing re-anchoring models.
- #3: a pointer-register readout from existing sweep heads; re-train with a dedicated register.
- #1: a d-scaling probe of the set-checkpoint capacity law.
- #2 and #8 are new architectures (GPU pilots).

**Novelty-check protocol for Theory's 3–5 proposals** (I will run this on each before registration):
- the three nearest works with verified IDs;
- the exact scope under which "new" holds;
- the closest competing result (Markovian Transformers 2404.18988, Ouro 2510.25741, Shapiro 2608.11233 by default);
- known failure modes;
- one line per item, committed before the registration commit.
