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

## Addendum (2026-10-08): Feng 2305.15408's arithmetic task exactly, plus looped / recurrent precedents (for option 2(a); prepared early, as Theory asked "no rush")
Read from the PDF (Sec. 3, 5; App. B, H; Alg. 1).

**Task, Arithmetic(n, p):**
- **Field:** Z_p with **p = 11**. Tokens are the numbers 0–10, the operators **+ − × ÷** (÷ = the modular inverse), brackets, and "=".
- **Generation (Alg. 1):** it builds the expression backwards from the answer.
  - Sample the answer t uniformly from Z_11.
  - Repeat n times: pick a random number position in the current string, sample an operator o uniformly, sample t₁, t₂ with
    o(t₁, t₂) = that number, and replace it by "t₁ o t₂", adding brackets when precedence requires (e.g. after ÷ or ×).
  - So **n = the number of operators.** Nesting depth isn't set directly; it emerges from random positions.
  - **The answer is uniform over 11 values, so chance = 9.1%.**
- **CoT format:** reduce the leftmost "handle" (one computable operation) per step, rewriting the whole expression each time, e.g.
  "1 + 5 × (1 − 2) = 1 + 5 × 10 = 1 + 6 = 7".
- **Datasets:** n ∈ {4, 5, 6} for the main panel; **1M training / 0.1M test** per setting, deduplicated. Direct datasets = CoT
  datasets with the steps removed (loss on the answer tokens only).
- **Models:** from-scratch transformers, d = 256, H = 4, L ∈ {3, 4, 5}; AdamW lr 1e-4, wd 0.01, batch 512, 100 epochs, dropout 0.1.
- **Results:**
  - **3-layer + CoT is near-perfect at all n.**
  - **Direct fails.** Deeper helps direct somewhat, but accuracy drops significantly with length. The per-bar values are only in
    Fig. 2 (a bar chart), not in the text, so read them off the figure if a numeric bar is needed.
  - **Extrapolation:** CoT trained on n = 1–15 (with relative PE) still performs well at n = 16–18 (Fig. 3).
  - CoT stays > 95% with 30% of the steps missing plus a single-token corruption.
- **Theory:** no bounded-depth log-precision transformer solves Arithmetic(n, p) for all n (unless TC⁰ = NC¹). A depth-5 constant-size
  transformer generates the CoT.

**Design flags for a latent carrier on this task:**
1. **Bracketed expressions need a stack.** Streamed left to right, one token per pass, the carried state must hold the pending partial
   results and operators up to the **nesting depth**. That is pushdown-like, not finite-state.
   - Unlike S5 / §128, the state size grows with depth. Bound the depth in generation (and report it), or the buffer-capacity
     condition (refinement 2) dominates.
2. **Don't flatten it.** A bracket-free, left-associative chain (x ← x op c) is a word problem over the affine maps of Z₁₁, a
   *solvable* monoid. So it's parallelisable (TC⁰ / ACC⁰), not inherently serial: it would lose the property 2(a) is chosen for.
   **Inherent seriality needs the nesting** (Buss's formula-value result).
3. **If finite state *and* NC¹-hard is wanted:** the S5 word problem remains the clean case (Barrington). Expression evaluation is the
   "maths-flavoured" one, at the cost of a stack.
4. **GPT-2 tokenisation:** " 0" to " 10" are single tokens; operators and brackets are separate tokens. That's fine.

**Looped / recurrent models on this task, or on CVP / iterated squaring, with K-pass ablations: not found on Feng's or Li's exact
tasks.** The closest, with loop-count ablations:
- Saunshi 2502.17416: a k-layer looped L times ≈ a kL-layer model on addition, p-hop induction and i-GSM synthetic maths, with
  loop-count curves;
- Fan 2409.15647: adaptive loop counts on RASP-L tasks, with length generalisation;
- 2604.07822: depth extrapolation in recurrent-depth transformers on multi-hop composition;
- implicit / fixed-point RNNs 2502.07827: state tracking vs the number of fixed-point iterations;
- theory: Giannou 2301.13196 (looped transformers as programmable computers).
- **New if run:** a Markov latent carrier on Feng-style expressions with a K-pass / extra-pass ablation and donor SERIAL.
