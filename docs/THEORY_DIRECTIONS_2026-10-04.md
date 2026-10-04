# Theory directions, 2026-10-04: hand-off convergence, learning without step labels, latent chains without snaps

A research memo for the user (requested via Audit, 2026-10-04). It contains no registrations. Every candidate below
would get a Literature check and a pre-registration before any run. Priors are verified by Literature in a214bb6d (`docs/LIT_DIRECTIONS_LEADS_2026-10-04.md`). §
numbers refer to `docs/THEORY.md` and its archive.

## What the evidence says (the constraints any solution must respect)

* **Per-step targets are load-bearing** for latent hand-offs inside the LM.
  - §93: sparse chunk-end labels give 0/3. §95: single-op footholds give 0/9.
  - With per-step targets, hand-offs are learned, but **slot by slot (a staircase)** and slowly: S5 needed 3× budget
    (§86 → §92).
* **Shortcuts compete with carriage.** On multiplication, slot 2 is recomputed from visible operands (98-3: mean-ablated
  latents still decode it at 0.79–0.99). §97's text fork also recomputes partly under mean ablation.
* **Depth recurrence with per-iteration targets is reliable** (§88/§91: 5/5 composition once budget suffices).
* **The C4 barrier is optimisation, not information.**
  - §85: an oracle-initialised table re-certifies with no short labels.
  - §96: a final-answer-trained table supplies exact per-op labels.
  - In-LM routes failed: RL (§69), annealing (§87), sparse labels and footholds (§93, §95).
* **Every success snaps; every unsnapped chain drifts or collapses**: §64's single loop past its trained steps, §66's CL,
  §86 at 1×, §89 without a decoder. 2609.02049 and 2604.06374 explain the mixture collapse. The **range principle**
  (§66 synthesis): failures are quantities leaving their trained range, and snaps keep them in range.

## (A) Making latent hand-offs converge fast and reliably

| # | Candidate | Mechanism, and why it fits our evidence | Closest prior | Cheapest decisive test | Likely failure |
|---|---|---|---|---|---|
| A1 | **Carry the state in depth, not in sequence** | Each op updates a recurrent residual state through a looped middle block with input injection (the op's embedding), instead of slot j *retrieving* slot j−1 through attention. Retrieval must be learned per position, which is our toy-era "pointer retrieval" mechanism (§2) and a candidate cause of the staircase. §88/§91's loop composes reliably. | T²MLR 2607.15178 (middle-layer state fed to the next token, retrofitted into a 1.7B model); Huginn, TRM | S5 k = 4 restarts (§97 recipe, text target), looped block per op vs §97 GEN-4 (a T²MLR-style state path). 3 seeds, about $3. **Pass:** per-slot ≥ 0.95 at all slots in ≤ 0.5× GEN-4's steps, with removal at the floor. | Loop depth per op is too shallow for a 120-state update; looped weights interfere with pretrained features (L2-SP helped in §91). |
| A2 | **Round-trip consistency loss** | Add ‖slot_j − Enc(Dec(slot_j))‖, where Enc is the hidden state the model produces when reading its own decoded checkpoint text. Every slot gets a dense latent-space target tied to readable text. It also directly optimises C7's round trip. | CODI; cycle consistency. Our F4 translator never trained, so this is an auxiliary loss only (Literature flag) | 4×4 mult GEN-M (fixed code) ± the loss. 3 seeds, about $3. **Pass:** slot-3 threshold step ≤ 0.5× control, removal at the floor. | Collapse to a trivial fixed point, if Enc and Dec co-adapt; needs a stop-gradient on one side. |
| A3 | **Compact state target** | Decode only the carried state (S_j), digit-structured, not "+ P_j ( S_j )". There is less to carry and decode, and S5's 7-bit state learned faster than multiplication's ~30 bits. | 2510.00184 (running-sum auxiliary target) | 4×4 GEN-M with the target "( S_j )". 3 seeds, about $3. **Pass:** M1 at 24k. | It is task-specific formatting (a soft "hand-built" cost); gains may only shift the staircase. |
| A4 | **Parallel teacher-forced state training** | All slots learn in parallel from gold state inputs, annealed to the model's own latents. That attacks the staircase directly. | LIFT 2609.38149, SST V2 2605.00206, PCCoT 2506.18582, C-MTP 2607.16972 | §99's TF-SLOT is its small-scale test (running). | Exposure bias when the gold share ends. |
| (in §99) | TF-SLOT, MARKOV, K-CURR, STD, TEXT-FIRST | Parallel slot learning; shortcut removal; a curriculum; standardisation; a CoT warm start. | §99 | §99 (fixed code) | — |

## (B) Learning step states without step labels (C4)

| # | Candidate | Mechanism, and why it fits our evidence | Closest prior | Cheapest decisive test | Likely failure |
|---|---|---|---|---|---|
| B1 | **Self-training by length** | Round 0: train on final answers of length-1–2 problems; their answers *are* step states, a ≤ p fraction. Round r: on problems up to length 2^r, the model's own decoded checkpoints become targets **only if its final answer is right**; then iterate. It grows the competent range like the std curriculum that certifies tables (§76/§82). Final-answer verification **plus a vote/length filter** (2502.01612; our §62 label-noise result) removes wrong states. Pilot L/M did this label-free on toys (19/20). | Lee 2502.01612 and 2511.07378 (recursive self-training extends length; needs a vote/length filter); STaR 2203.14465 | S5 GEN-4 (§97 recipe), 4 rounds, 3 seeds, about $6–8. **Pass:** §97's G4 with ≤ 5 % length-1/2 problems and no other step targets. | Early-round errors on rare prefixes get locked in (§96's uncertified teachers); verification at the final answer misses compensating errors. |
| B2 | **Restart-consistency objective** | No step targets. Final-answer CE, plus a term that rewards a chunk's decoded checkpoint if **restarting from it reproduces the full run's final answer**. A good checkpoint is one that carries the state; the final answer anchors its meaning. | Markovian Transformers 2404.18988; LaTRO 2411.04282, BoLT 2503.18866. Flag: our §50 reward halt collapsed | S5 k = 4, final answers only (problems of length 4–128). 3 seeds, about $4. **Pass:** checkpoint decode ≥ 0.95 at the chunk boundaries and G4. | Degenerate private codes (consistent but unreadable), so C7 fails while C4 passes; a symmetric saddle at the start (§85). |
| B3 | **Amortised teacher, generalised (§96 → beyond finite state)** | The pipeline works for finite-state tasks. Replace the free table with a teacher class that fits arithmetic (e.g. a learned carry automaton), then the LM hosts. | §96 | Multiplication teacher feasibility on CPU, $0. | It's not general: every task needs a teacher class, so it's a side bet. |
| (closed) | RL from outcomes (§69), annealing (§87), sparse labels (§93), footholds (§95) | — | — | — | Already failed. Not repeated (fix-loop risk). |

## (C) Fully latent chains without snaps (stable over hundreds of steps)

| # | Candidate | Mechanism, and why it fits our evidence | Closest prior | Cheapest decisive test | Likely failure |
|---|---|---|---|---|---|
| C1 | **Snap-strength dose-response (soft attractor)** | Replace the hard snap with h ← (1−β)·h + β·Proj(h) (the projection onto the decoded checkpoint's embedding). β is annealed from 1 down to a kept **β_min ∈ {0.1, 0.01, 0}**, one arm each. The smallest β_min that keeps the chain exact measures how much discrete correction the latent really needs. Literature flags that β → 0 removes the carrier, so a small kept β is the realistic target. | Attractor Models 2605.12466. Flag: annealing β → 0 removes the carrier, so keep a small β (Literature) | S5 §97 GEN-4, 3 β_min arms × 3 seeds (9 runs), about $6, each tested at its β_min to 2,048. **Pass:** ≥ 0.99 at 512 in ≥ 2/3 at β_min ≤ 0.01, removal at the floor. | Accuracy decays as β → 0, the range principle reasserting itself. A kept small β is then the honest fallback ("soft attractor"). |
| C2 | **Denoising / contractive training, late** | Add Gaussian noise to the latents in the last third of training, and require the next state to decode correctly. The map becomes contractive around valid states (attractors), so errors decay instead of compounding. Early noise broke learning (§23), so it is added late. | 2609.26487 (anisotropic stability of finished recursions); 2510.14095's error-correction noise | As C1, with noise σ ramped over the last 8k steps, no snaps at test. 3 seeds, about $4. **Pass:** as C1. | Contraction only within the noise scale; drift beyond it persists. Combine with C3. |
| C3 | **Norm / manifold constraint** | Project each latent onto a fixed-norm shell, or onto the convex hull of the token embeddings. This removes amplitude drift, the toy-era Mechanism 2 (attenuation, §4/§14), and the out-of-distribution input norms behind §72's anisotropy. | nGPT 2410.01131 (hypersphere; 4–20× faster training) | An add-on arm to C1/C2 (+3 runs, about $2). | Necessary but not sufficient: direction drift remains. |
| C4 | **Orthogonal / Householder state channel** | A dedicated recurrent state updated by LM-produced Householder products (norm-preserving, so no drift). It represents group actions exactly, and DeltaProduct solves S5 tracking this way. GPT-2 writes the op parameters and reads the state. Stability holds by construction for group-structured state. | Howe 2609.18966 (norm-preserving; exclude any additive input path), DeltaProduct, NC-SSM 2610.00329 | S5 at 512 → 2,048, unsnapped, 3 seeds, about $4. **Pass:** ≥ 0.99 at 2,048, and removing the channel drops to chance. | It moves state-tracking into a special module (is that still "the LM thinking"?). Non-group state (arithmetic carries) does not fit orthogonal maps. |
| C6 | **Per-op fixed point (implicit recurrence)** | Each op's latent update is the fixed point z* = f(z*; state, op), solved by iteration, with implicit gradients. Stability comes from contraction at the solution, so depth adapts to need and drift is bounded by construction. | DEQ 1909.01377, Implicit LMs 2502.07827, Fixed-point RNNs 2503.10799 | S5 unsnapped to 2,048, 3 seeds, about $5. **Pass:** ≥ 0.99 at 512 with no snaps. | Solver cost; no contraction on hard ops; training instability. |
| C5 | **Long self-generated rollouts** | Train on chains much longer than 4 with the model's own latents (no snaps or teacher forcing), with truncated backprop. The model sees its own drift in training. | LIFT 2609.38149 | S5 chains of 32 ops unsnapped. 3 seeds, about $6. **Pass:** ≥ 0.99 at 128 and 512. | It is expensive, and the range principle predicts failure past the trained chain length unless the dynamics are stationary (needs C1/C2). |

**What would count as success for (C):**
* **C1:** zero/mean removal → chance.
* **C2:** accuracy holds past the trained chain length with **no discrete checkpoint at test** (train ≤ 128 ops, test
  512–2,048).
* **C6:** the per-step decode stays exact along the chain, latent norm and nearest-valid-state distance stay bounded
  (no drift), and ≥ 4/5 seeds pass.

## Ranking (expected information per dollar toward the north star)

| Rank | Candidate | Why first | Cost |
|---|---|---|---|
| 1 | **C1 snap-strength dose-response** | It directly measures how much discrete correction our working chains need (C2 + C6 toward no snaps), on the §97 code, cheaply. Every β_min outcome is informative. | about $6 |
| 2 | **A1 state in depth** | It tests a mechanism hypothesis for the staircase (retrieval vs recurrence) and unifies §88's reliable loops with §92's restarts. If it is right, every later rung trains faster. | about $3 |
| 3 | **B1 self-training by length** | It is the most direct in-LM route to C4 consistent with §85/§96 (information present; growing range); there are toy precedents. | about $6–8 |
| 4 | A2 round-trip consistency | It helps (A) and C7 at once; cheap as an extra loss. | about $3 |
| 5 | C2 + C3 contraction + norm | The fallback for C1, if annealing fails. | about $4–6 |
| 6 | B2 restart-consistency | A strict-C4 attempt; C7 risk. | about $4 |
| 7 | C6 per-op fixed point / C4 norm-preserving channel | Literature's pick for (C) if the soft attractor fails. C4 narrows generality to group state. | about $4–5 each |

**Not proposed:** further label-thinning variants (§93/§95 closed), RL from outcomes (§69), or annealing of free tables
(§87). These would be fix loops.

* **Literature's own picks:** A = LIFT-style parallel state training (TF-SLOT in §99 now); B = filtered self-training by
  length; C = a norm-preserving channel or a per-op fixed point.

**Suggested order** (given credit of about $67 with a $30 floor, and §98/§100/§99 reruns at about $26): **C1 and A1
next**, about $7, each on one proof host first. B1 follows the user's go-ahead on C4 self-training.
