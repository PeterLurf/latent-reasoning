# Looped / recurrent-depth latent reasoning and related hidden-state techniques: proposals (2026-10-04)

A research memo for the user (directive relayed by Audit, 15:40). It contains no registrations. Each proposal would get
a pre-registration with Literature's check before any run. Priors are verified in c70d55e3 (`docs/LIT_LOOPED_PROPOSALS_2026-10-04.md`). § numbers refer to
`docs/THEORY.md` and its archive.

## The two walls these proposals target

1. **The no-read basin (reliability, C6).** Forced-carriage designs work in some seeds and never start in others: §108
   S5 Markov is 1/4, §110 self-training 1/3. The failing seeds' loss sits exactly on the "latent never read" curve
   (111-1). §111 (READ / DAG / 2×) is testing fixes.
2. **Per-hand-off precision on richer states (C1 / C2 on arithmetic).** S5's step is a lookup (about 7 bits).
   Multiplication's is a running sum with carries.
   - k = 2 works at 4×4 (§100); k = 4 collapses (§98 0.16); 5×5 fails.
   - Neither width (§107 LAT) nor extra token positions (§107 PAUSE 0.06 / 0.04 / 0.01) helped.
   - Compounding: 0.99 at 512 needs about 0.9999 per hand-off.
3. Also: snap points are hand-set (C5 unmet), and length generalisation comes from the snap / cut scaffold, not from the
   latents (§106).

## Proposals (ranked by the user's order: C1 > C6 > C4 = C2 > C3 > C5 > C7)

### P1. Looped depth per op: serial depth, not positions (C1 on arithmetic, C6, a C5 handle)
* **Mechanism.** In each Markov segment (forced carriage, §99/§108 harness), GPT-2's middle block (layers 4–7) is looped
  **r times** over the segment's positions.
  - Huginn-style input injection: the layer-3 state is re-added each iteration.
  - The looped layers are tied, with per-iteration LoRA relaxation (Relaxed Recursive Transformers) so GPT-2's
    pretrained function survives at r = 1.
  - Training samples **r at random** (1–8). Each iteration's output, through layers 8–11, gets the same step-text target
    (per-iteration targets, so the extra loops aren't untargeted slack).
* **Why it fits our evidence.** Multiplication's step is a multi-digit multiply-add. Extra *positions* (§107 PAUSE) did
  not help, and carries need *serial depth* (Saunshi). §88/§91's looped block composed reliably with per-iteration
  targets.
* **Closest prior.** Huginn 2502.05171, Ouro 2510.25741, UT, CoTFormer 2310.10845, Relaxed Recursive 2410.20672, MoR
  2507.10524. Pretrained retrofits: Shapiro 2608.11233 (about 1.5× depth), McLeish 2511.07384 (a recurrence
  curriculum), T²MLR 2607.15178.
* **Cheapest decisive test.** 4×4 multiplication, MARKOV segments, r_train ∈ 1..8, 3 seeds, about $6–9.
  - **Pass:** op-3 decode ≥ 0.95 in ≥ 2/3 seeds (CARRY-3) with §104 donor USE and removal at the floor, **where §107
    PAUSE-4 failed** (head-to-head, pre-registered).
  - Descriptive: accuracy against test-time r (r beyond training: C2 in depth).
* **Likely failure.** Test-time depth scaling transfers only modestly to retrofits (Literature); overthinking at large r
  (Kohli); a looped pretrained block destabilising (keep LoRA small, plus a curriculum).

### P2. Redundant state coding: error correction by independent copies (C6, C2 precision)
* **Mechanism.** Each op writes **2–3 latents, each trained to carry the full state** (each gets the per-slot decode
  target). The next segment reads their **average** (or a learned vote).
  - Independent errors cancel: per-hand-off error ε → about ε² for 2 copies if errors are uncorrelated.
  - Distinct from §107 LAT-m, a wider single carrier with one read.
* **Why it fits our evidence.** §108 s3 carries serially but at 0.99845 per hand-off, which compounds to 0.59 at 512.
  The fix needed is precision, not mechanism.
* **Closest prior.** None in latent CoT (Literature: new). Related: error-correction noise 2510.14095, nGPT 2410.01131,
  ensembling.
* **Cheapest decisive test.** S5 MK-4 (§108 recipe) with 2 copies, 5 seeds, about $6. **Pass:** per-hand-off ≥ 0.9999
  and cumulative ≥ 0.99 at 2,048 in the seeds that escape the no-read basin, against §108's 1/4.
* **Likely failure.** The copies' errors are correlated (shared weights), so the gain is small. The copy-decorrelation
  statistic is logged.

### P3. Straight-through Gumbel snaps: end-to-end training through discrete checkpoints from final answers (toward strict C4)
* **Mechanism.** A k = 1 text-snap chain (§97 GEN-1: each op is a chunk, the state written as text) trained with **only
  the final answer** supervised. Gradients pass through each snap by a **straight-through Gumbel** estimator: a hard
  sampled checkpoint forward, a relaxed backward.
  - This is a fully differentiable discrete chain, with no step labels and no exact marginal.
* **Why it fits our evidence.** Exact marginalisation (tables) needs a foothold (§85). RL on outcomes failed (§69).
  Soft forward mixtures collapse (2609.02049, 2604.06374, 2508.03440), hence a hard forward. ST-Gumbel is an estimator
  we have not tried.
* **Closest prior.** Gumbel-softmax 1611.01144; SofT-GRPO; MUX (lossless alternative).
* **Cheapest decisive test.** S5 GEN-1, problems of length ≥ 3 only (strict C4), 3 seeds, about $4. **Pass:**
  cumulative ≥ 0.95 at 512 with no short problems. A pass is the first strict-C4 result.
* **Likely failure.** Biased gradients over long chains; a symmetric start (§85's saddle). A ge3-strict null is likely.
  A positive would be decisive.

### P4. Self-triggered snapping by round-trip discrepancy (C5, C2)
* **Mechanism.** At each latent step, compute d = ‖latent − Enc(Dec(latent))‖, where Enc re-reads the decoded
  checkpoint text. **Snap only when d > τ**, with τ calibrated in range. This is a label-free trigger and not confidence
  (§83 showed confidence fails out of range).
* **Closest prior.** ReLAT 2606.06252; §83 (confidence routing).
* **Cheapest decisive test: $0, eval-only** on existing §97/§105/§108 models.
  - Calibrate τ on training-range chunks. On test chains, measure (i) first-error detection rate (as §83), and (ii) the
    accuracy and snap count of adaptive snapping against fixed k = 4.
  - **Pass:** ≥ 0.95 first-error detection out of range, with fewer snaps at equal accuracy.
* **Likely failure.** Enc and Dec share weights, so consistent errors give a small d. Report the miss rate.

### P5. Adaptive depth by fixed-point halt (C5; conditional on P1)
* **Mechanism.** In P1's looped block, halt when ‖z_{i+1} − z_i‖ < ε (label-free). Do harder ops (larger carries) get
  more iterations?
* **Prior.** ACT, PonderNet, DEQ, Huginn's divergence exit. Direct support: 2609.26487 (latent motion drops after
  completion), but it is anisotropic, so calibrate ε in range and test out of range.
* **Test.** Eval-only on P1's models ($0). **Pass:** iterations correlate with carry count (ρ ≥ 0.5), at equal
  accuracy.

### P6. Early triage of no-read seeds (C6, at the procedure level)
* **Mechanism.** Use Audit's loss/null-curve signature (111-1): kill or restart a seed that has not left the null curve
  by about 1k steps after k = 2.
* **Prior.** Dodge 2002.06305 (bad seeds are identifiable early).
* **Test.** Already pre-registered as §111's early_escape analysis. If its hit rate is high, adopt it as the reported
  recipe ("train n, restart the non-escapers").

## Ranking (information per dollar under C1 > C6 > C4 = C2)

| rank | proposal | conditions | cost |
|---|---|---|---|
| 1 | **P1 looped depth per op** (vs PAUSE, head-to-head) | C1 (arithmetic), C6, C5 handle | about $6–9 |
| 2 | **P2 redundant state coding** | C6, C2 precision | about $6 |
| 3 | **P4 round-trip-triggered snapping** | C5, C2 | **$0** (eval-only) |
| 4 | **P3 straight-through Gumbel from final answers** | strict C4 | about $4 (likely null; decisive if not) |
| 5 | P5 adaptive depth (after P1) | C5 | $0 eval |
| 6 | P6 early triage (from §111) | C6 procedure | $0 |

**Suggested next:** P4 now ($0, eval-only, on existing models); P1 and P2 registered once §111 shows whether the
no-read basin is fixed (they share its harness).
