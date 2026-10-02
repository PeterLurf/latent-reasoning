# Prior check for §83: uncertainty-triggered restarts in place of fixed chunk size (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch).

## 1. Confidence-triggered cuts, retrieval or exits
- **Trigger on low-probability tokens:** FLARE 2305.06983 (active retrieval when a generated token's probability
  falls below a threshold, then regenerate) and DRAGIN 2403.10081 (uncertainty-driven retrieval timing). These are
  the closest in *mechanism*: a confidence drop triggers a discard-and-redo.
- **Confidence-gated computation with calibrated thresholds:** CALM 2207.07061 (early exit per token). Its thresholds
  are set by **Learn-then-Test risk control** (2110.01052), giving a finite-sample guarantee of consistency with the
  full model. **This is the precedent for calibrating τ in range**, and the method to use: LTT, or conformal
  (2306.10193) for "≤ 1% false restarts in range" with a stated confidence level.
- **Learned halting:** ACT 1603.08983 and PonderNet 2107.05407 learn when to stop. DEER 2504.15895 stops long CoT on
  confidence. Delethink, InftyThink and PENCIL cut on fixed budgets or rules.
- **Selective prediction:** Geifman & El-Yaniv 1705.08500 (risk–coverage curves).
- **Not found:** confidence-triggered history-cut restarts on a state-tracking task with an exactness certificate.
  New as scoped.

## 2. Is confidence reliable out of range?
- **The general evidence says no, especially far from training data.**
  - Modern networks are overconfident (Guo 1706.04599).
  - ReLU networks give arbitrarily high confidence far from the training data (Hein 1812.05720).
  - LMs are reasonably calibrated in distribution on their own answers (Kadavath 2207.05221), but that is in-range
    evidence.
- **I found no paper measuring LM confidence calibration specifically under length or context-length extrapolation.**
  Length-generalisation papers report accuracy collapse (Anil 2207.04901; Zhou 2402.09371; Kazemnejad 2305.19466),
  not confidence. So §83's detection rate is itself a new measurement, and the main risk is **confident errors**.
- **Failure modes to expect:**
  1. **Confident wrong states.** A positional or OOD shift can flip the argmax while keeping a peaked distribution.
     Then the error is accepted, and all later states are confidently consistent with it.
  2. **Detection only at the first error matters.** After one undetected error, later confidence says nothing about
     correctness. Score detection on the *first* wrong state per run.
  3. **Product-of-token-probabilities scale.** States with more tokens get systematically lower products. Use a
     per-state normalised score (min-token or mean log-prob) or calibrate τ per state length.
- **Exactness arithmetic.** With n accepted states and a miss probability q per erroneous state (error rate p per state
  OOD), P(no undetected error) ≈ (1 − p·q)^n. At 2,048 states, holding 0.99 needs p·q ≲ 5·10⁻⁶. Because restarts
  reset to in-range contexts, p should stay near the in-range error rate. So report p and q separately.

## 3. Framing for C5
- It is neither learned halting (ACT/PonderNet) nor a fixed budget.
- Suggested wording: **"self-gated restarts: chunk boundaries are chosen at inference by the solver's own predictive
  confidence, with a threshold calibrated on in-range data only (risk-controlled, as in CALM/LTT); no parameters are
  trained for segmentation."**
- C5 credit: "adaptive, model-determined boundaries (C5, unlearned)". A trained cut policy remains the stronger C5
  version.
- **Report against fixed k:**
  - the risk–coverage curve (accepted chunk length vs accuracy) as τ varies;
  - accuracy at 512 and 2,048;
  - first-error detection rate;
  - the false-restart rate on fresh in-range data, with the LTT guarantee.
