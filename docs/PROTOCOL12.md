# PROTOCOL12 — annotated-input reasoning in GPT-2 small (THEORY §54, F1): registered outcome "not learnable as specified"

Registered 2026-09-30 22:23 EDT. Theory source: THEORY §54 (bf82897). Implementation: `prosqa/ail_gpt2.py` (original GPT-2 small,
sha256 7c5d3f4b…). Pre-registration tuning under Audit's rules (training data only; ≤ 6 settings; all logged):
`docs/PROTOCOL12_TUNING.md`.

**Outcome.** None of the 6 settings learned within its smoke budget (≤ 594 steps × 4 problems).
* **Settings tried.** Head and label-embedding scales, bias −4 / −2, the hint arm, and a soft warm-up with a depth curriculum.
* **The hint arm.** It also fails, so the failure is not credit assignment from final answers alone.
* **The diagnosis.** A one-batch check shows the cause is the specified hard forward labels with a sigmoid
  straight-through estimator: the loss is piecewise constant and no label crosses its threshold. With soft forward
  labels the same model fits the per-pass targets exactly.
* **Consequence.** The arms AI-mono / AI-free / AI-hint and predictions AG1–AG4 are not run as specified. A redesign
  (e.g. soft labels in training, hardened at test, with a readability check) needs Theory's design and a new
  registration before any test-set run. No test-set result exists; the 12 problems touched by the setting-#0 plumbing
  check are excluded from any later evaluation.
* **Success bar for a redesign.** 2608.11233 is cited with Audit's correction (intermediate-state supervision, no
  matched compute).
