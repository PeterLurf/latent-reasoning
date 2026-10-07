# Where SOTA latent-reasoning architectures fall short, and where we can improve on them (2026-10-07)

Literature session, at the user's direct request; Theory and Audit copied.
- **IDs:** verified via the arXiv API (2026-10-07) or earlier notes (LIT_INDEX).
- **Builds on:** LIT_GENERAL_LATENT_MAP (070703ea), LIT_ARCH_SWEEP (87cd2af5), LIT_BIG_BETS (75f51298).
- **Ordered by our priorities:** C1 > C6 > C4 = C2 > C3 > C5 > C7.

**The SOTA families:**
- **Recycled hidden states:** Coconut 2412.06769; CODI 2502.21074; SIM-CoT 2509.20317; LT-Tuning 2602.10229; LOTUS 2606.31779;
  MUX 2607.18264.
- **Discrete latent vocabularies:** Abstract-CoT 2604.22709.
- **Recurrent depth:** Huginn 2502.05171; Ouro 2510.25741; retrofit 2608.11233.
- **Tiny recursive / attractor:** HRM 2506.21734; TRM 2510.04871; Attractor Models 2605.12466.
- **Latent RL / sampling:** SLPO 2607.19691; ATF 2609.33271; 2510.07745.

## Shortcomings, the evidence, and our position

| # | Shortcoming in SOTA | Evidence | Our position |
|---|---|---|---|
| 1 | **C1: the latents are often not causally used.** The computation rides the KV cache, or the thoughts are ablatable | Aswal 2606.12689; SCIT 2608.27265 (CODI's computation sits in the V-cache); Coconut / SIM-CoT 0-thought ablations; **2607.06648: accuracy rises while counterfactual responsiveness *falls* during training**, so endpoint checks mislead | **Ahead (by design).** Markov segments make the code the only cross-step path. Donor SERIAL is scored per model (§108 / §111 / §128). No SOTA method paper runs a donor test |
| 2 | **C6: the latent chain is unstable as it lengthens** | Coconut collapses at 5 latents (12.5%, as reported in SIM-CoT 2509.20317); feature collapse from recycling raw hidden states, severe at 8B with untied embeddings (LT-Tuning 2602.10229); **2607.09698: Coconut is an *expanding* dynamical system, CODI a stable attractor** (Lyapunov analysis) | **Ahead on finite state.** A bounded FSQ code plus a measured contraction ρ: exact to 2,048 steps on S5 and the §128 families. Our RAW-H (raw hidden, 0.85 at 512) reproduces Coconut's failure mode, and the bounded code fixes it. 2607.09698's attractor / expanding split is our ρ reading, found independently |
| 3 | **C2: fixed latent counts and depth limits** | Coconut / CODI use a fixed c or 6 latents per task; looped models degrade beyond trained depth (Ouro; retrofit ≈ 1.5×, 2608.11233) | **Ahead on finite state:** trained at ≤ 32, exact at 2,048 (64×). Not shown on maths |
| 4 | **C4: dependence on a CoT curriculum or teacher** | every SOTA method distils or curricula from CoT; 2602.01148 argues a curriculum is *theoretically necessary* (distribution mismatch); latent RL needs a supervised warm start (SLPO, Abstract-CoT) | **Ahead on finite state:** FINAL-only (strict C4) with a length curriculum, which is a curriculum over *length*, not CoT, consistent with 2602.01148. **Behind on maths:** multiplication isn't reached on the task-agnostic carrier |
| 5 | **C3: latent methods are weak at *computation*** | GPT-2 GSM8K-Aug: Coconut 33.1 vs CoT 41.6 (Dilgren 2604.04902); 2602.01148: "exploration 97% (ProsQA) vs computation 34% (GSM8K)"; MUX 2607.18264: one latent per step is a bandwidth bottleneck, and multiplexing several subwords per latent losslessly prevents collapse shortcuts | **Shared weakness.** 140-3: our single-vector code (and even h) fails to hold accumulators ≥ 100. This is the same failure as SOTA, seen more cleanly |
| 6 | **C5 / exploration: deterministic latents can't sample or branch** | 2510.07745 (no sampling mechanism in continuous space; adds dropout / noise + a latent reward model); ATF 2609.33271 (a diffusion head for multimodal next thoughts); 2602.01148 (the certainty trade-off: certain latents execute, uncertain ones explore but accumulate error) | **Structural advantage, unused.** FSQ codes give an exact per-dimension categorical (LIT_CAND5_6). Rounding = certainty for execution; sampling levels = exploration. One carrier can switch between them, which is 2602.01148's "adaptive certainty" made concrete |
| 7 | **C7: readability is correlational** | LM-head projections (LOTUS), auxiliary decoders (SIM-CoT, SELR 2608.13570), probes; superposed traces (2607.09698) | **Ahead on finite state:** discrete codes allow exact automaton extraction (CERT / Moore minimisation), not just probes |
| 8 | **Evaluation practice** | shortcut-prone benchmarks (ProsQA proximity heuristic 0.911, our §57); endpoint-only faithfulness (2607.06648); accuracy hides unreliable paths (2603.03475: 81.6% of correct Qwen2.5-Math answers come from inconsistent pathways) | **Ahead:** shortcut-free task design (L1), scoring on every raw step, and donor tests through training |

## Where we're behind (state it)
- **Scale and domain:** GPT-2 small, synthetic finite-state tasks. No natural-language or maths result on the task-agnostic carrier
  (GSM8K is paused by the user).
- **Arithmetic computation** (row 5) is unsolved for us too.
- **C1 is design-forced** (Markov segments), not emergent. Each item costs a full GPT-2 pass, so C3 on long inputs is worse than
  token-level latents.
- **Task-agnostic only in part:** the segment = item layout is a per-task design choice.

## Where we can improve on SOTA (concrete, ranked)
1. **Bring our evaluation to SOTA checkpoints** (C1 / C6; cheap, no training).
   - Run donor SERIAL + ρ on the released GPT-2 Coconut / CODI / SIM-CoT checkpoints (Dilgren 2604.04902's matched public set).
   - That is the first donor-level and contraction-level comparison in the field. It extends 2607.09698's Lyapunov reading with a
     causal test.
2. **A bounded discrete carrier as a drop-in for Coconut-style recycling.** Rows 2 / 3 say raw recycling expands and collapses;
   our FSQ-bounded code doesn't.
   - The first test that needs no GSM8K: Coconut's own ProsQA / ProntoQA with an FSQ + COPY carrier vs raw recycling, scored on
     latent count beyond training (C2) and donor tests (C1).
3. **Fix the shared computation bottleneck (row 5) with a different state shape, not a wider vector:**
   - (a) B1's **additive, no-decay accumulator** (carry-save column keys; LIT_BIG_BETS);
   - (b) **MUX-style lossless multiplexing** (several digits superposed per carried code with a geometric-decay weighting). It's
     a principled form of MULTI-SLOT, with a recoverability guarantee.
   - RUNNING-SUM (A) decides between them.
4. **Stochastic FSQ for exploration + outcome RL** (rows 4 / 6).
   - A categorical over levels gives exact log-likelihoods: no surrogate (SLPO), no diffusion head (ATF), no dropout (2510.07745).
   - Warm-start from an entered carrier.
   - Add a learned certainty gate (round vs sample) to test 2602.01148's trade-off directly.
5. **Track faithfulness through training** (2607.06648's lesson). Log donor SERIAL at checkpoints, not only at the end. Our
   restart-on-plateau and fade schedules are exactly where responsiveness could quietly decay.

**Prediction:** items 1 and 2 are where we can show a clear advantage soonest. Item 3 is where SOTA and we fail alike, so a win
there would be the largest step beyond the field.
