# Prior check for Theory's three big bets: B1 DELTA-STATE, B2 MID-STACK HAND-OFF, B3 RETROFITTED RECURRENT DEPTH (+ paradigms) (2026-10-07)

Literature session, for Theory (the user's 16:40 sweep request); Audit copied. Builds on LIT_ARCH_SWEEP_2026-10-07 (87cd2af5).
- **IDs:** all verified via the arXiv API (2026-10-07).
- **Audit's IDs:** all verify. Grazzi et al. = **2411.12537**; DeltaProduct = 2502.10297 (correct).

## B1 DELTA-STATE: a gated DeltaNet layer mid-stack as the only cross-segment channel
**Not new as an architecture. New in its use (state tracking / arithmetic, negative eigenvalues, donor C1).**
- **Precedents for a linear-recurrent memory as the cross-segment path of a (pretrained) transformer:**
  - **Infini-attention 2404.07143:** local attention within a segment plus a compressive linear-attention memory (with a
    delta-rule variant) as the long-range path. Retrofitted into 1B / 8B models by continual pretraining.
  - **ARMT 2407.04841:** a layerwise associative memory with segment recurrence, **built on pretrained GPT-2 (137M)**. It still
    passes RMT memory tokens too.
  - TransformerFAM 2404.09173; RMT 2207.06881.
- **What differs:** all of these are evaluated on retrieval / long context. None tests state tracking, arithmetic or causal use. The
  closest results on state tracking with linear states are Grazzi 2411.12537 and DeltaProduct 2502.10297, trained from scratch.
- **Design flags:**
  1. **Accumulation and the delta rule pull in opposite directions.**
     - The delta rule's write *replaces* the value at key k: v ← v + β(v_new − S k). Accumulating then still needs the network to
       compute old + product in-pass, which is the 140-3 failure.
     - A plain **additive (linear-attention) write, S ← S + v kᵀ with no decay, is an exact integrator.** Products written to a
       column key give carry-save column sums with no in-pass addition.
     - So keep an additive path: β → 0 along accumulation keys, or a separate ungated linear-attention head. Initialise the gate α
       ≈ 1, since any decay below 1 corrupts sums over 16+ steps.
     - Negative eigenvalues (β ∈ (0, 2)) serve the finite-state families, not sums.
  2. **Carry resolution moves to the readout.** Converting column sums (up to about 300) to product digits is the carry-cascade
     problem (2310.13121).
     - Emit LS-first, one product digit per output step reading its column key. The decoder must then hold a carry (≤ about 30)
       across output steps. Score per-column readout separately.
  3. **Zero-init W_o gives q / k / v exactly zero gradient at install** (as in GROW), so W_o moves first. Use a short warm-up, and
     μP-style lr care on W_o (fan-in = the head value dim).
  4. **C1 by design** holds only if the delta state is the *only* path: attention is segment-local, and no residual stream or
     memory token crosses segments. **Donor test:** swap S between items at hand-off t.
  5. **State size and position:** one mid-stack layer reads what that layer's input has. Pair it with B2's probe to place it.

## B2 MID-STACK HAND-OFF (ℓ_out of step t → ℓ_in of step t+1)
**Not new as an architecture. New in our Markov carrier with donor C1.**
- **Precedents:**
  - **Feedback Transformer 2002.09402:** the lowest layer at step t reads the *highest* representations of the past; it addresses
    transformers' limits on tracking tasks. Trained from scratch.
  - **TransformerFAM 2404.09173:** a feedback loop, retrofittable, with no extra weights.
  - **T²MLR 2607.15178:** middle layer → earlier layer, about 20% of the network; retrofitted at 1.7B; beats full-depth recurrence.
  - CoTFormer 2310.10845: CoT seen as re-applying the model, with attention to earlier passes.
- **The $0 probe test is sound,** with two caveats:
  - (a) **Probe the untuned pretrained GPT-2 as well as s137 / s140.** The trained carriers have adapted their late layers to the
    last-layer tap, which biases the profile toward ℓ = 12.
  - (b) **Use a non-linear probe and bin by accumulator magnitude** (140-3's binning). The question is whether *any* layer holds
    values ≥ 100. If none does, B2 alone doesn't escape L2.

## B3 RETROFITTED RECURRENT DEPTH for direct-answer 4×4
**Not new as an architecture. The novelty would be the pairing with stepwise internalisation (below).**
- **Retrofit precedents:**
  - 2608.11233: prelude / loop / coda on Qwen2.5-0.5B; one task step per loop under step supervision; persists after
    outcome-only annealing; about 1.5× depth extrapolation.
  - McLeish 2511.07384: a recurrence curriculum.
  - Relaxed Recursive Transformers 2410.20672: pretrained → looped, with layer-wise LoRA.
  - Mixture-of-Recursions 2507.10524: per-token depth (C5).
- **From scratch:** Huginn 2502.05171; Ouro 2510.25741; iso-depth 2604.21106 (φ 0.46).
- **Looped arithmetic and length generalisation:** Saunshi 2502.17416 (k layers × L loops ≈ kL layers on addition / p-hop); Fan
  2409.15647; 2604.07822 (depth extrapolation, from scratch).
- **Damping:** 2609.16665 (¼ step).
- **Energy / attractor stopping (C5):** EBT 2507.02092; ERM 2607.10128 (deeper isn't always better); Attractor Models 2605.12466.
- **Baselines (GPT-2 small, 4×4):**
  - No-CoT **0.29** (ICoT-SI 2405.14838, Table 2; verified in LIT_BENCH_VERIFY). Our own no-CoT at our budget: 0.003 (§98).
  - **ICoT-SI 1.00 with no loops.** That is the bar a looped direct-answer model must beat for C3. B3's distinctive claims are C2
    (loops beyond training) and C5 (convergence halting).
- **Flags:**
  - **No serial chain across items.** C1 means loop-state swaps at loop k (donor at depth), not item hand-offs. State which C1 is
    claimed.
  - §114 LOOP-R failed under L1's shortcut. Direct-answer 4×4 has all inputs visible to every loop, so the loop *may* recompute from
    the inputs. The test of real iteration is accuracy and probes **by loop index** (a monotone per-loop progress).
  - Truncated BPTT lowers φ (2604.21106). Prefer full backprop through the loops at r ≤ 8.

## Paradigms
- **TRM 2510.04871 / HRM 2506.21734 deep supervision:** supervise the answer after each outer refinement, with the carried z
  detached between refinements. It fits B3 directly (the answer after each loop group), and gives a convergence signal for C5.
- **ATF 2609.33271:** a diffusion head samples the next continuous thought, then RL. For strict C4 it's still warm-started (as
  Abstract-CoT / SLPO; LIT_GENERAL_LATENT_MAP). Our evidence (§69; Allchin) says RL needs a working latent model first. **C6 risk:**
  sampled continuous thoughts.
- **Stepwise internalisation *into depth* (removing CoT while adding loops):**
  - Not found as one schedule.
  - Closest: 2608.11233 (step-per-loop supervision, then outcome-only annealing); Log-ICoT 2605.28600 (proves ICoT internalises
    k-parity in log₂ k stages, absorbed into deeper *layers*); McLeish's recurrence curriculum.
  - **New as scoped:** an ICoT-SI removal schedule coupled to a loop-count schedule (remove step j ↔ add a loop).

## What is NOT new (summary)
- B1's architecture (Infini-attention, ARMT on GPT-2).
- B2's architecture (Feedback Transformer, FAM, T²MLR).
- B3's retrofit (2608.11233, McLeish, Relaxed Recursive, MoR).
- TRM deep supervision.
- **New:** their use as the *only* carried channel in a Markov latent chain with donor C1 on state-tracking / arithmetic tasks;
  B1's additive-key carry-save accumulator; and the removal ↔ loop coupled schedule.

## Ranking by our priorities (C1 > C6 > C4 = C2 > C3)
1. **B1**, with the additive path kept: C1 by design, and it attacks L2 directly via carry-save keys.
2. **B2's $0 probe first.** It decides whether B2 is worth building, and where B1's layer should sit.
3. **B3:** mainly C2 / C3 / C5. ICoT-SI's 1.00 without loops sets a high C3 bar.
