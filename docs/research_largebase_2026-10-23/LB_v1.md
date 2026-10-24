# LARGE-BASE v1: the §228 latent C1 band on Qwen2.5-0.5B (2026-10-24)

**Status:** v1 = LB_draft.md (9c38f364), changed as listed here; where they differ, this file wins. Red team: REVIEW_LB.md (REVISE; fixes 1–10 adopted, no new red team needed). Literature: LIT_LARGEBASE_CHECK_2026-10-24.md (IDs verified; Qwen2.5-0.5B GSM8K 41.6, 4-shot; N1 narrow citing Shapiro 2608.11233, which retrofits a loop on this same base; N2 OK with the linearisation tagged [ASSUMED]; N3 narrow).

## 0. What actually failed at GPT-2 small (REVIEW_LB #1, $0 on the trained §228 model)
- Pass-1 aux is 0.51 overall: 0.77 on d = 1 items, 0.25 on d = 2.
- On d = 2 items, v_1 is at chance in the write source u (linear probe 0.11–0.16) and in S_1 (0.13–0.16).
- **The think pass never computes the unrequested intermediate.** S_1 just mirrors u. This is not a storage or name-lookup failure.
- The premise for scale is therefore: **does a larger base compute the intermediate eagerly at the write sources?**

## 1. Base and the premise test (fix 3)
- Qwen2.5-0.5B (revision 060db649…).
- **P1′ launch condition ($0, CPU, frozen base, before any spend):** EAGER = a linear probe for v_1 at the actual write sources on d ≥ 2 TOPO items (the same probe as REVIEW_LB #1, run on frozen Qwen and frozen GPT-2 small).
  - **EAGER_Qwen ≤ 0.30 → no spend;** the line closes as "the 0.5B base does not compute the intermediate eagerly".
  - The draft's OPD lookup probe becomes descriptive.
- D2's "escape law" is withdrawn as derived. The zero-init wo makes it a regression regime. Only the "presence" argument stands [ASSUMED].

## 2. Cells
- DIRECT-Q, LAT-Q (the §228 INJECT port, value target, same d-sequence) and TEXT-Q (the §227 control, run to 30k; fix 7).
- **Arm (fix 8):** INJECT is kept as the single-factor test. A null closes "INJECT at 0.5B" only.
- **Pre-named next registration if LAT-Q fails:** a step-told arm (the pass is told which step to compute) or a Shapiro-style loop retrofit, tested first on GPT-2 small.

## 3. Learning rate and init (fixes 4, 5, 6)
- **The trunk lr is unanchored, so A1 runs TWO LAT arms:** 1.6e-5 (weight-RMS transfer) and 8.6e-5 (µP). A1 closes only if both fail.
  - If Literature extracts Shapiro 2608.11233's trunk lr on this base before launch, that value replaces 8.6e-5 (a dated Note before output).
  - Saved checkpoints are re-probed (P1′ probe) after A1, to tell feature drift from failure to build.
- **GQA memory init:** per-head orthogonal rotations of the copied k / v, so the 7 + 7 heads start distinct. Plus a symmetry gate **G-B9** (pairwise head-state cosine at init < 0.9).
- **Memory-side rate:** wo's lr is scaled to 3.8e-5 to match GPT-2's read / residual ratio (0.37 vs 0.14). The read / residual ratio is logged.

## 4. Gates (fixes 2, 7)
- **LP2 is read on d = 2 items only** (in general, items with d > t), via a new by-(d, pass) log key.
- **LP3's not-applicable cut is DIRECT-Q d = 2 > 0.55** (was 0.40).
- If A2's DIRECT-Q d = 3 > 0.50, Stage B does not run, and the d ≤ 8 design goes to the user.
- LP0 is self-calibrated as in the draft (replays cleanly on the §228 / §229 logs).

## 5. Build traps (fix 10)
- inject_ow's aux value index would land on Qwen's standalone space token. Fixture **G-B8b** asserts it reads the digit token.
- **boxes/deltamem.py is not edited** (running jobs lazily import it). The Qwen paths go in a new module.
- The two-group optimiser gets a flags-off bitwise gate.
- The answer loss is the full-vocab sequence NLL (as GPT-2), not a softmax over 11. The prefix-factorised read is used for evaluation only. Chance ≈ 0.10, since answer 0 never occurs at d = 1; this is stated.
- A censored DIRECT cell means one re-run, not a fail.

## 6. Design and cost (fix 9): the staged design is TERMINAL
- **No d ≤ 8 follow-up under this registration** (no splitting to get under the line).
  - Stage A: A1 (d ≤ 2, 8k; DIRECT-Q + two LAT-Q arms) → A2 (d ≤ 4, gate LP4) → TEXT-Q to 30k.
  - Stage B (U{1..6}, 36k): DIRECT-Q + LAT-Q, only on an A pass.
- **Cost (fixes included):** $18.79 point / **$24.42 guard → cap $25** (≤ $30: Audit). Expected spend ≈ $5.71 point / $7.42 guard, under the draft's priors plus the applicability branch.
- F2 on the cloud 3090 replaces every modelled rate before admission.
- CHUNK_DELTA is used only if its gate passes. Otherwise Stage B uses U{1..5} (cost in LB_draft §9).

## 7. OC and predictions (unchanged except as fixed)
- P(P1′ passes) ≈ 0.45.
- P(A1 LP2 on d = 2 ≥ 0.80 in either lr arm | P1′ pass) ≈ 0.45.
- P(LP3) ≈ 0.3.
- Overall P(C1) ≈ 0.04 [ASSUMED].
- Under no effect, P(pass) ≈ 0 (exact binomial, LB_draft lb_oc.json).
