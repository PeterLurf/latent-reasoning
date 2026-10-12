# Prior check for §188: a per-pass aux (equal expected weight per pass position) combined with a per-item randomised K = n + U{1..4} (2026-10-11)

Literature session, for Theory; Audit copied. Every ID below was verified earlier (LIT_INDEX); none is new.

## 1. Has anyone combined per-step supervision, randomised loop counts and pass-count extrapolation?
- **Not as one recipe.** The components appear separately.

| Paper | Per-step supervision | Randomised loop count | Extrapolation in passes |
|---|---|---|---|
| Huginn 2502.05171 | no (LM pretraining) | yes: log-normal Poisson, r̄ 32, truncated BPTT 8 | test-time recurrence scaling; task-dependent saturation |
| Fan 2409.15647 | no | per-example T(n) (size-matched), adaptive at test | yes (length) |
| Kuo 2606.29983 | no | stochastic loop counts; RL halting | stabilises across inference loop counts |
| Fixed-Point Reasoners 2606.18206 | no (final output) | fixed-point halting | adaptive compute; stability via pre-norm + residual scaling |
| Kohli 2604.07822 | no | R ~ clip(Poisson(λ), 2, 8) | depth extrapolation |
| Retrofit 2608.11233 | **yes**: one step per loop | not stated as randomised | ~1.5× supervised depth, then decay |
| CLRS hints 2205.15659; NPI 1511.06279 | **yes**: per-step hints / traces | the step count follows the algorithm | larger inputs (more steps); hints help some tasks only |

- **§188's combination is new.** The closest pairs are the retrofit (step supervision, then extrapolation to ~1.5×) and Huginn / Kuo (randomised depth, no
  step supervision).

## 2. A known negative interaction: step supervision → pass-index fixation → worse iteration extrapolation?
- **Yes, two direct pieces of evidence and one mechanism:**
  - **Chen 2603.21676:** in from-scratch depth-recurrent models, **intermediate per-step supervision "consistently harms" extrapolation**. That's stated for
    the graph-reachability task (verifiable propagation depth). The closest published negative.
  - **Liang, Jin, Pan 2609.33144:** under final-answer supervision, MR-Loop advances through a **transferable "progress cue"**, a learned pass-position
    signal, which **degrades past the training depth**. Errors then compound (limited self-correction). This is the pass-index fixation mechanism even
    without step supervision; step targets at fixed pass positions would strengthen such a cue.
  - **Bansal 2202.05826:** fixed-iteration training learns "behaviours specific to the iteration number". The fix: progressive training (start from detached
    intermediate states) so behaviour is repeatable indefinitely.
- **And one confound specific to §188:** K = n + U{1..4} ties loop count to problem size. That's exactly **Kuo's spurious size ↔ loop-count correlation**, the
  cause of erratic OOD behaviour.
  - A within-item window of 1–4 is a small randomisation; Kuo's fix is randomising K *independently of n*.

## Recommendations
1. **Decouple the step target from the absolute pass index:**
   - supervise "the next reduction from the current state" (target = the step after the one the memory currently encodes), rather than "step t at pass t";
   - or add Bansal-style progressive passes (some chains start from a detached mid-chain memory with the step targets shifted).
   - Either one tests whether the aux teaches a procedure or a schedule.
2. **Add a K-independent arm** (K ~ U{n_min .. n_max+4} regardless of n) beside K = n + U{1..4}, so Kuo's correlation can be separated from the aux effect.
3. **Read C6 at 2–3× K with per-pass step accuracy, not only the final answer.** Liang's degradation shows up as progress-cue / readout drift before the
   answer flips.
