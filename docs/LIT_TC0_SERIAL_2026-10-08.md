# When is serial latent use forced? TC⁰ vs inherently serial tasks: precedent for note 158-2 (2026-10-08)

Literature session, for Theory (note 158-2, b304d697); Audit copied. IDs verified via arXiv abstract pages (2026-10-08). The circuit
results are classical (journal papers, not arXiv).

**Theory's claim:** multiplication ∈ TC⁰, so a memory that can hold the inputs admits "buffer + read-time compute", and C1 is forced
only on inherently serial tasks or beyond the training length. **The literature supports it, with two refinements (below).**

## 1. Precedents
- **Transformers without CoT are TC⁰-bounded:**
  - Merrill & Sabharwal 2207.00729 (log-precision transformers ⊆ uniform TC⁰);
  - Li, Liu, Zhou, Ma 2402.12875 (constant depth, constant-bit precision ⊆ AC⁰). **With T CoT steps, any size-T circuit.**
  - **Empirically, CoT helps most on tasks hard for parallel computation:** permutation-group composition, iterated squaring, the
    circuit value problem, especially at low depth.
- **CoT expressivity:**
  - Merrill & Sabharwal 2310.07923 (linear CoT steps → regular languages; polynomial → P);
  - Feng et al. 2305.15408 (bounded-depth transformers can't evaluate arithmetic expressions or solve linear equations directly unless
    TC⁰ = NC¹; with CoT they can).
- **Gradient descent prefers parallel shortcuts when they exist:** Liu et al. 2210.10749, "Transformers Learn Shortcuts to Automata".
  - Shallow transformers learn shortcut (parallel) simulations of semiautomata: constant depth for solvable groups, O(log T) for S5,
    instead of step-by-step recurrence.
  - **These shortcuts are brittle out of distribution (length).**
  - It is the closest evidence for "default to a non-serial solution when one exists".
- **Direct evidence of "buffer + read-time compute" in ICoT-SI's multiplier:** Bai et al. 2510.00184. The ICoT model uses attention
  to build a DAG that **caches and retrieves pairwise partial products**: parallel read-time computation, not a serial carried state.
- **Our own:** §114 / §121 / §122 (the carrier holds the multiplier prefix, the b-prefix, and recomputes); §155 R s1 (exact, but not
  SERIAL).
- **Not found:** a study that directly contrasts recurrent / memory-augmented models on a TC⁰ task vs an NC¹-hard task and reports
  buffering vs serial-state use. The closest pair of results is Liu (shortcuts, by group class) + Li (CoT gains by seriality).
  **This contrast would be a new, clean experiment for us:** RUNNING-SUM / multiplication vs S5 on the same carrier, with donor
  SERIAL.

## Two refinements to 158-2
1. **NC¹ tasks also admit parallel solutions, given log depth** (Liu: S5 in O(log T) layers; Sanford 2402.09268: k-hop needs about
   log k depth). So "inherently serial" is relative to the depth available at read time.
   - In our harness each pass and the final read have *constant* depth (12 blocks).
   - **The forcing condition is "the task's depth exceeds what one constant-depth read can compute over the buffer"**: true for S5 /
     NC¹-complete tasks at long T, false for fixed-size multiplication (TC⁰).
2. **Buffer capacity is a second forcing condition, independent of complexity.** A fixed d_k × d_v memory buffers O(d) symbols.
   - When the input information exceeds the buffer (long chains, or 4×n with large n), a compressed carried state is forced even for
     TC⁰ tasks.
   - That's why length beyond training (C2) and long chains are where C1 becomes non-optional.
- **And one caveat: membership ≠ learnability.**
  - Multiplication ∈ TC⁰, yet standard training fails at 4×4 (Bai: < 1%), and ICoT-SI needs a removal curriculum.
  - So "buffer + compute" is *admitted*, not guaranteed. A per-step reader can tip the optimiser to a serial state (as §111).

## 2. Citation for "integer multiplication ∈ uniform TC⁰"
- **Multiplication of two n-bit integers is in TC⁰** via iterated addition of the n partial products. Iterated addition is in TC⁰
  (Chandra, Stockmeyer & Vishkin, "Constant depth reducibility", SIAM J. Comput. 1984). That is the classical source.
- **Hesse, Allender & Barrington, "Uniform constant-depth threshold circuits for division and iterated multiplication"** (JCSS 65(4),
  2002) gives **DLOGTIME-uniform TC⁰ for division and iterated multiplication**, the stronger statement. It's the standard modern
  citation when uniformity matters.
- **Textbook:** Vollmer, *Introduction to Circuit Complexity* (Springer, 1999).
- **Recommended wording:** "integer multiplication is in (DLOGTIME-)uniform TC⁰ (Chandra–Stockmeyer–Vishkin 1984; Hesse–Allender–
  Barrington 2002)".

## 3. Inherently serial benchmarks that fit GPT-2 scale and have text-CoT baselines
- **Permutation composition / the S5 word problem** (NC¹-complete, Barrington). Ours, with a text-CoT baseline (§119), and in Li
  2402.12875.
- **Arithmetic expression evaluation mod p** (NC¹-complete): Feng 2305.15408 trains small transformers with and without CoT; there's
  a direct CoT vs no-CoT gap.
- **Circuit value problem and iterated squaring:** Li 2402.12875 (small transformers, CoT vs none).
- **Graph reachability / ProsQA** (Coconut; our shortcut-free ProsQA, §57): GPT-2 text-CoT baselines exist. Continuous-CoT theory:
  2505.12514.
- **k-hop induction / pointer chasing:** Sanford 2402.09268 (log-depth lower bounds; synthetic, cheap).
- **Recommendation:** arithmetic-expression evaluation (Feng's format) is the closest to "serial maths". It's NC¹-complete, has a
  published CoT vs direct gap at small scale, and has no b-prefix-style shortcut if the expression is streamed.
