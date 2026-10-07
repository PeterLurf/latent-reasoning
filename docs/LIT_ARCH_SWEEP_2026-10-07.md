# Major architecture / paradigm changes for a pretrained transformer: a sweep mapped to our lessons and C1–C7 (2026-10-07)

Literature session, at the user's request via Audit; Theory and Audit copied.
- **IDs:** all verified via the arXiv API (2026-10-07). Abstract-level claims only, unless marked.
- **Priority:** C1 > C6 > C4 = C2 > C3 > C5 > C7.

## Leads (Audit's list, verified) and additions (marked +)

| Family | Papers | What it changes | C1 test? | Main C-claims / lesson mapping |
|---|---|---|---|---|
| **Retrofit recurrent depth** | 2608.11233 (Qwen2.5-0.5B prelude / loop / coda; a 6M adapter over a frozen base matches the 180M full block); McLeish 2511.07384 (a curriculum of recurrences); Ouro 2510.25741; Huginn 2502.05171; iso-depth 2604.21106 | loops a middle block per token (depth, not across items) | none run. The abstract claims one task step per loop. A donor swap of the loop state at loop k is directly applicable | **C2:** extrapolates to about 1.5× the supervised depth, then decays (Ouro also degrades beyond trained depth). **C4:** step supervision, then outcome-only annealing persists, the same arc as our READ → fade. **C3:** φ = 0.46, so a loop is worth about half a unique block, and truncated BPTT lowers φ to 0.38 |
| Loop damping | "Right direction, wrong step" 2609.16665 | a fixed ¼ step on loop updates | — | **C6:** loop failures are overshoot along a useful direction. This matches our ρ / expansion reading of continuous carriers. A damped carried update (e ← e + α Δ) is untested in our harness |
| **Cross-token middle-layer recurrence** | T²MLR 2607.15178 | caches the previous token's *middle-layer* state and injects it into an *earlier* layer; retrofitted into a 1.7B model | none | **Directly comparable to our carrier,** which taps the last layer into the input embedding. T²MLR finds a localised middle-layer loop (about 20% of layers) beats full-depth recurrence |
| **Hybrid SSM / linear RNN** | Priming 2605.08301 (GKA > GDN > Mamba-2, initialised from a transformer, < 0.5% of pretraining tokens); 2604.21454 (hybrids help on strictly sequential chained updates, not on multi-hop; reasoning tokens dominate) | replaces some attention layers with recurrent layers | none. **In hybrids, attention layers bypass the recurrent state** (the SCIT / KV lesson), so C1 needs a Markov cut | C3 (cache / speed). State tracking is task-dependent |
| + State tracking in linear RNNs | Grazzi 2411.12537 (**verified**: eigenvalues in [−1, 1] → parity, modular counting; Householder products → any regular language); Sarrof 2405.17394 (SSMs exact on star-free languages, as flip-flop; no parity with eigenvalues ≥ 0); Merrill 2404.08819 (diagonal SSMs can't do S5 tracking); DeltaProduct 2502.10297 (n_h Householders → S_n, length extrapolation improves sharply from 1 to 2–3); DeltaNet 2406.06484; Gated DeltaNet 2412.06464; RWKV-7 2503.14456; Mamba-2 2405.21060 | **a state transition with a bounded spectrum built into the architecture** | an explicit fixed-size state, so **donor-swappable by design** | **C6 / C2:** bounded eigenvalues make the transition non-expanding by construction (no RAW-H drift), and group tracking is exact (isometric). Our §128 families (S5, FF, CYC, MOD, RDFA) are all regular, so DeltaProduct covers them in theory |
| + Fixed-point / implicit RNNs | 2502.07827 (implicit LMs are RNNs: iterating to a fixed point gives non-linear state tracking); Fixed-point RNNs 2503.10799 | a dense transition via fixed-point iteration | — | links to Attractor Models / our contraction lesson |
| + Conversion recipes | LoLCATs 2410.10254; MOHAWK 2408.10189; Mamba-in-the-Llama 2408.15237; RADLADS 2505.03005 (350–700M tokens at 7B) | distil attention into linear / recurrent layers | — | cost references for (c) |
| + Test-time memories | Titans 2501.00663; TTT layers 2407.04620; fast weights 2102.11174; ARMT 2407.04841; MARCH 2608.12435 (cached state anchors) | a learned memory updated by online gradient steps | explicit state | recall-oriented, no arithmetic evidence |
| **Energy / attractor** | EBT 2507.02092; ERM 2607.10128 (Hopfield energies pick trajectories and depth; Sudoku 98.97%, Maze 99.30%); Attractor Models 2605.12466 | inference as energy descent / a fixed point | none | **C6:** fixed points and energy-based selection of depth. ERM: deeper recurrence isn't always better (C5) |
| **Tiny recursive** | TRM 2510.04871 / HRM 2506.21734 | trained from scratch, a few-M-parameter recursion on z | none | puzzles, not a pretrained LM. The input is visible every iteration, so the loop state may not be needed (our §121 input-carriage lesson) |
| **Latent diffusion / distributional thoughts** | ATF 2609.33271 (a diffusion head samples the next continuous thought; the backbone is kept); LaDiR 2510.04573; CCDD 2510.03206 | stochastic, multimodal continuous thoughts | none | C3 / pass@k diversity. **C6 risk:** continuous, sampled, no correction. Low priority for our C1 > C6 ordering |
| **Arithmetic embeddings** | Abacus 2405.17399; position coupling 2405.20671 | digit-significance IDs in token space | n/a | **C2 for arithmetic:** the length generalisation comes from making significance explicit. This maps to the MULTI-SLOT slot-ID design |

## (a) Which have, or could take, a C1-style causal test?
- **None of the method papers runs donor / interchange tests.** The evidence is accuracy, probes and ablations, as in the earlier map
  (LIT_GENERAL_LATENT_MAP).
- **Testable by design:**
  - architectures with one explicit fixed-size carried state and no other cross-step path: pure linear-RNN layers, DeltaProduct, a
    T²MLR cached vector *if* attention is cut, our Markov carrier;
  - loop-state swaps at loop k for recurrent depth.
- **Not cleanly testable:**
  - hybrids with attention: attention bypasses the state unless segments are Markov;
  - TRM / HRM: the input is re-visible each iteration;
  - diffusion thoughts: stochastic, so swaps need a fixed noise seed.

## (b) Do delta-rule / linear-recurrent states carry arithmetic accumulators that non-linear bottlenecks fail at?
- **Theory, yes for counters and sums of bounded range:**
  - a running sum is an integrator (eigenvalue 1);
  - modular counting needs negative or complex eigenvalues, and non-triangular transitions for mod 3 (Grazzi 2411.12537);
  - mod 10⁴ is a regular (finite-state) language, so it is in principle covered by Householder products.
- **Experiments:** Grazzi's modular-arithmetic and parity tasks, and DeltaProduct's permutation groups, at small scale.
- **Not found:** evidence that linear-RNN states learn **multi-digit carries** or multiplication accumulators where non-linear
  bottlenecks fail.
- **Counter-evidence:**
  - 2510.14826 proves fixed-state SSMs can't do "truly long-form" generation. Its 1,000-digit addition needs *external tool memory*.
  - Grazzi notes non-linear RNNs handle parity easily, so the linear property buys parallel training, not more expressivity.
- **For us:** 140-3's failure (accumulators ≥ 100 absent even in h) is in-pass learning, not a carrier limit. A linear-recurrent
  carrier wouldn't obviously fix it. The arithmetic literature points to **digit layout with significance IDs** (Abacus, position
  coupling) and **more in-pass depth**.
- **RUNNING-SUM (A) is the discriminating test.** Adding a DeltaProduct-cell arm there would answer (b) for our setting directly.

## (c) Minimum retrofit cost on a GPT-2-small-scale model (124M), cheapest first
1. **Move the carrier's tap point (T²MLR-style):** read layer ℓ (for example, 6) and inject at layer ℓ′ (for example, 2) instead of
   last-layer → input. It's a code-path change in fsq120 with the same budget as current runs (about $0.4–0.5).
2. **A DeltaProduct / Gated-DeltaNet cell as the carried state** (eigenvalues in [−1, 1], n_h 2–3), replacing W_down / FSQ / W_up.
   There's no attention conversion, so no distillation. Kernels exist (the flash-linear-attention library). Same budget.
3. **Loop a middle block per item** (prelude / loop / coda), with a recurrence curriculum (McLeish) and a ¼-step damping option
   (2609.16665). An adapter-only variant (2608.11233's 6M-over-frozen) is cheapest. Compute grows with loops (×r on the looped
   layers). This adds in-pass depth, which addresses 140-3.
4. **A full hybrid conversion** (LoLCATs / MOHAWK / Priming) of GPT-2: a distillation of order 10⁸ tokens. Costlier, and attention
   remains a bypass. **Not recommended** for C1.
5. EBT / diffusion / TRM-style: training from scratch or a large redesign. Not a retrofit at this scale.

## Recommendation, ranked by C1 > C6 > C4 = C2
1. **DeltaProduct carrier** (option 2). Explicit state (C1), non-expanding by construction (C6), theory covering all §128 families,
   and length-extrapolation evidence (C2). The first test is RUNNING-SUM + S5 against FSQ-8.
2. **Looped middle block per item + our Markov carrier** (option 3): a two-level recurrence (slow across items, fast within an
   item, as HRM's structure in a pretrained LM). It targets multiplication's in-pass bottleneck.
3. **The T²MLR tap point** (option 1). The cheapest ablation of a design choice we never varied.
4. Keep Abacus-style slot / significance IDs for any multi-slot arithmetic design.
