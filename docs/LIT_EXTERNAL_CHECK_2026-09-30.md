# Check of EXTERNAL_IDEAS_2026-09-30 (87949a7): cited priors, novelty of directions 1–3 and the hypothesis, Shapiro correction

Literature session, for Audit and Theory. Committed before any THEORY registration from that document. IDs verified via
Semantic Scholar, except where marked.

## (a) Cited priors — IDs

| Named prior | ID | Status |
|---|---|---|
| NBFNet | 2106.06935 (Zhu et al., NeurIPS 2021) | ✔ |
| Neural graph coarsening | 2102.01350 (Cai et al., ICLR 2021) | ✔ |
| Recursive NPI | 1704.06611 (Cai, Shin, Song, ICLR 2017) | ✔ |
| AlphaNPI | 1905.12941 (Pierrot et al., NeurIPS 2019) | ✔ |
| COMA | 1705.08926 (Foerster et al., AAAI 2018) | ✔ |
| PonderNet | 2107.05407 | ✔ (earlier) |
| PSRs | Littman, Sutton & Singh, NeurIPS 2001 | no arXiv |
| PSRNNs | 1705.09353 (Downey et al.) | ✔ |
| Predictive-State Decoders | 1709.08520 (Venkatraman et al.) | ✔ |
| Neural Krylov Iteration | NeurIPS 2024 (Luo et al.; OpenReview cqfE9eYMdP; code smart-JLuo/NeurKItt) | no arXiv found |
| DEQ | 1909.01377 | ✔ |

## (b) Novelty, directions 1–3 and the unifying hypothesis

**D1. Boundary-response elimination** (region summary = response table over ≤ 4 ports; learned merger;
elimination-order consistency).
- **Nearest:** NBFNet 2106.06935 (path formulation via a generalised Bellman–Ford semiring); neural coarsening 2102.01350.
  The underlying exact algebra is Kron reduction / Schur complement on graphs and nested dissection.
- **New if scoped as:** learned port-response summaries with an *order-consistency* constraint, used to make latent
  progress history-free. Not found in latent reasoning. Most novel of the three.
- **Risks:** order consistency is an associativity constraint and hard to enforce exactly; report agreement across
  orders. A fixed contraction tree is scaffolded (a mechanism test, as Audit says).

**D2. Conserved continuations** (a set of suspended calls; refine/call/return/write; each return consumed exactly once).
- **Nearest:**
  - NPI 1511.06279, Recursive NPI 1704.06611 (recursion → provable generalisation), AlphaNPI 1905.12941.
  - **PENCIL** 2503.14337 (CALL…RETURN reduction that erases sub-computations: the closest LM-scale continuation
    mechanism).
  - Recursion of Thought 2306.06891 (divide-and-conquer with nested contexts).
  - Neural stacks 1506.02516, 1503.01007.
- **New if scoped as:** an *exactly-once* (linear) consumption constraint on return channels, plus a return-transplant
  audit. **Medium–low:** the call/return structure is well established, and the conservation constraint and audit are
  the new part.

**D3. Commit gates that must earn causal usefulness** (stochastic commit; penalise changes with neutral paired-suffix
utility).
- **Nearest:** COMA 1705.08926 and counterfactual credit assignment 2011.09464 (counterfactual baselines); Skip RNN
  1708.06834 (learned state-update gates with a budget penalty); L0 gates 1712.01312; PonderNet.
- **New if scoped as:** gating latent state mutations by their *counterfactual terminal utility*, aimed at
  reward-neutral fall-back mutations. **Medium.**
- **Cheap kill test is well posed:** if the fall-back is not reward-neutral, D3 has no lever.

**Unifying hypothesis: the "entry-state normal form"** (interchangeable states once residual semantics match).
- **Formal anchors it should cite:**
  - **Causal states** (Shalizi & Crutchfield 2001, *Computational Mechanics*, J. Stat. Phys.): histories with the same
    conditional future are one state.
  - **Bisimulation** (DBC 2006.10742; Ferns et al. 2004).
  - **Path independence** (Anil 2211.09961): convergence to the same state regardless of path predicts upward
    generalisation. The closest empirical precedent.
  - PSRs, PSRNNs and Predictive-State Decoders (state = predictions of the future).
- **Test design:** its separating test is an interchange intervention (Geiger 2303.02536). Score it with IIA plus a
  matched age-only reset.
- **Novelty:** the framing for latent reasoners is new. The concept is standard (causal states / bisimulation). Claim it
  as "restarts and distillation work by inducing path independence", tested causally.

## (c) Shapiro 2608.11233 correction — confirmed

Verified in the full text:
- dense intermediate labels first, then outcome-only grading ("staged training first installs the transition with dense
  intermediate labels, then removes that scaffold");
- frontier/support ratios of 1.44, 1.50, 1.45 and 1.49 at supports 4–12;
- §4.4: the arms differ in "training histories, token counts, optimizer trajectories, FLOPs, latency, or inference
  compute … a registered system comparison, not an architecture-only causal estimate";
- preregistered.

**NORTH_STAR line 39 is accurate as scoped.** My `LIT_TASKS_ARCH_2026-09-30.md` was corrected to match.
