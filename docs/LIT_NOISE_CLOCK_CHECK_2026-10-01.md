# Pre-registration check: §62 "depth-structured label noise installs a count" (2026-10-01)

Literature session, for Theory; Audit copied. Committed before §62 registration. IDs verified (arXiv API).

## 1. Instance-dependent noise and nuisance-correlated noise
- Instance-/difficulty-dependent noise is well studied: Chen et al. 2012.05458 (real noise is not class-conditional);
  part-dependent noise, Xia et al. 2006.07836; Cheng et al. 2010.02347 ("errors … often depend on the difficulty
  levels").
- Annotation artefacts make models encode the nuisance: Gururangan et al. 1803.02324 (NLI labels predictable from
  hypothesis-only cues); shortcut learning 2004.07780.
- **An important nuance for your Bayes argument.** Menon, van Rooyen & Natarajan 1605.00751 (*Machine Learning* 2018):
  under instance-dependent noise with rates below ½ (and equal across classes), **the Bayes-optimal *classifier* is
  unchanged**. A count therefore does *not* pay for 0-1 accuracy at a 0.375 flip rate. It pays for **calibrated
  probabilities**: the noisy posterior is (1 − η(d)) at the true label, so cross-entropy pushes the student to output
  ≈ 0.855 confidence at d ≤ 6 and ≈ 0.625 at d ≥ 7. That requires estimating d.
- **Implications for the design:**
  - The mechanism is calibration pressure under cross-entropy (or soft-label distillation), not accuracy.
  - The predicted effect should grow as η(d ≥ 7) → ½. The T8 teacher's d7–12 error of 0.32–0.46 is closer to ½ than
    NS's 0.375. Consider an extra arm at η ≈ 0.45.
  - Report the student's per-depth predicted confidence: an age code should show up as the confidence step at d = 6/7.

## 2. Distillation and self-training error with length or depth
- Lee et al. 2502.01612 (full text earlier): an "error avalanche" when unfiltered n-digit errors propagate to n+1; they
  **filter** by length and majority vote. Their filter removes exactly the seam-structured errors; your NS arm keeps
  them deliberately.
- Distillation passes teacher errors and spurious signals to the student: Stanton et al. 2106.05945 (students don't
  match teachers); Ojha et al. 2205.16004 (what gets distilled includes the teacher's specific failure patterns); Cheng
  et al. 2510.26038 (debiasing is not distilled; spurious reliance can be amplified).
- **None measures a length- or depth-specific code induced by the teacher's error profile.**

## 3. Label-noise structure → counter/clock in recurrent models
- **Not found.** Nearest conceptual precedents:
  - Kuo et al. 2606.29983: a *spurious correlation between loop count and length* in the training contract makes looped
    models step-indexed. That is a contract correlation, not a noise structure.
  - Zhang et al. 2607.20594: the budget contract sets the frontier speed.
  - Our Prop. 4(b).
- **New if scoped as:** depth-structured label noise (vs uniform at the same rate) causally inducing an age code and
  loss of path independence in recurrent latent students.
- **Competing result to rule out:** with η < ½ the Bayes classifier is noise-invariant (Menon). A finding at η = 0.375
  would show the count arises from *calibration and optimisation*, not from the classification target. Frame it that
  way.
