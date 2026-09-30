# Prior work for the set-valued restart pilot (THEORY §33/§44) (2026-09-30)

Literature session, for Theory; Audit copied. IDs verified (Semantic Scholar/arXiv).

**1. Conditioning on a self-chosen, set-valued, discrete written state.**
- **CoT2** 2505.23648: samples and composes K discrete tokens per step and feeds them back. The closest self-chosen,
  set-valued discrete state, but no restart and no history cut.
- **Text-level parallel branches** (the model writes a branch structure, then continues from it): Multiverse 2506.09991
  (map / process / reduce; branches share only the prefix), APR 2504.15466 (spawn/join child threads), Hogwild!
  2504.06261, ThreadWeaver 2512.07843.
- **Tree of Thoughts** 2305.10601: a search over written states, run externally.
- **Carried state tokens** (continuous, not written): Memory Transformer 2006.11527; Recurrent Memory Transformer
  2207.06881 (segment recurrence via memory tokens, the nearest "restart a segment from carried tokens"). Slot
  Attention 2006.15055 (a set of slots refined iteratively). Pointer Graph Networks 2006.06380 (pointer/set state
  inside a graph network).
- **Not found:** a latent loop restarted from a *set of self-chosen node-id tokens* with the latent history cut. The
  claim holds scoped to that.

**2. Keeping branches alive through a discrete bottleneck.**
- Theory: Zhu 2505.12514. Discrete CoT must commit to one path (O(n²) steps), whereas a continuous thought superposes
  the frontier. A *written set* restores parallelism by listing the frontier, like BFS's queue.
- CoT2: achievable parallelism grows with embedding dimension (K composed tokens).
- Gu 2609.13747: uneven weights → "weakest-object routing bottleneck"; cumulative states need less dimension.
- Text-level parallelism (Multiverse, APR) keeps branches as separate streams, not a single state.
- Cost trade-off: a W-token written set restores every branch's amplitude but costs W tokens; nobody has quantified this
  for latent loops (open).

**3. Failure modes when a model must attend to several start tokens.**
- **Collapse onto one component:**
  - Wu 2508.03440: soft thinking is "single-threaded", following the top token.
  - Backour 2609.02049: feedback amplifies one mixture component.
  - Rizvi-Martel 2604.06374 and Coconut's own analysis: spread, then commit.
- **Position and sink effects:**
  - Attention sinks (Xiao 2309.17453): initial tokens absorb attention regardless of content, so extra START tokens
    can act as sinks. This fits §44's "extra START tokens ignored".
  - Lost in the middle (Liu 2307.03172): first and last positions favoured.
- **Dispersion:** Veličković 2410.01104 (per-anchor mass falls as anchors are added).

**Mitigations the literature suggests:**
- Permutation-invariant anchors: a shared position id, as in position coupling (2405.20671).
- Randomised anchor order and anchor dropout in training.
- A per-anchor coverage target, so each anchor's branch must be answerable (CoT2's matching of a *distribution* over
  target traces is the precedent).
- Report the per-anchor attention share (an effective number) to detect collapse.
