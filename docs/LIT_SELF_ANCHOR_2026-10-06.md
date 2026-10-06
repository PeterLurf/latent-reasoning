# Prior check for Variant 2b, SELF-ANCHOR: §118's anchor with the model's own greedy decode as the text path (2026-10-06)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-06) or earlier notes.

## 1. Prior art
- **Aligning free-running to text-fed dynamics:** professor forcing 1610.09038 (the teacher path is gold-forced);
  our §118 ANCHOR (gold state text).
- **Self-generated targets through a discrete channel:**
  - back-translation 1511.06709 and dual learning 1611.00179 (a model's own outputs, translated back, become training
    targets);
  - cycle consistency 1703.10593;
  - self-conditioning 2208.04202;
  - iterated learning, Ren et al. 2002.01365 (agents trained on their own previous generation's discrete messages:
    compositional languages emerge through a learning bottleneck).
- **Self-distillation:** CODI 2502.21074 (gold-CoT teacher in the same model); BYOL / SimSiam 2006.07733 /
  2011.10566 (stop-grad self-targets).
- **Hard self-decoded restarts:** our §112 / §116 (final-answer training through self-decoded restarts); 2510.14095
  (argmax re-embedding each iteration).
- **New as scoped:** a soft, training-only consistency pull of a wide carried latent toward the carried state computed
  from its *own decoded* "answer if queried now", label-free, with no test-time snap. The closest are §118 (gold) and
  §112 / §116 (hard, test-time).

## 2. Failure modes
1. **The text path must stay competent, and nothing in 2b trains it** (the §121 / §122 lesson: ANCHOR bound to a dead
   text path).
   - In §118 the text path b = [EOT, gold text, op] got competent training from somewhere. In 2b, with sg(b) and FINAL
     supervision, **nothing directly trains b to compute the next state from text.** b can decay into a dead target, and
     the anchor then ties the latent to it.
   - **Fix (FINAL-only, label-free):** also train **text-path rollouts to the final answer**. Restart from the
     self-decoded text and continue to the end with final-answer CE: §116's mechanism, the CODI γ·L_teacher analogue.
     That keeps b competent without labels.
   - Log b's own next-decode accuracy (truth-scored, log only).
2. **Constant-decode collapse at init:** if decodes are near-constant, b is near-constant, and the anchor pulls every
   latent to one point. Your gate (weight 0 until decodes are valid) is necessary.
   - Add a **diversity condition** to the gate: decode entropy across the batch / active units 1509.00519. Valid but
     constant decodes must not open it.
3. **Confirmation of wrong decodes:** a wrong decode becomes the anchor target, so the latent is pulled to a
   self-consistent wrong state (Arazo 1908.02983; iterated learning *selects* for learnable rather than correct
   codes).
   - Only the final-answer loss corrects it. Keep the anchor weight modest.
   - Optionally anchor only on **final-answer-filtered chains** (§110's filter), i.e. chains whose final answer was
     right.
4. **The sg asymmetry is adequate.** It is BYOL-like with input asymmetry (text vs latent), and the decoder losses
   block the constant solution. Keep §118's cosine + log-norm form, and monitor the effective rank, as you do (about
   150–200 in §118; a falling rank is the collapse signal).
5. **Family scope:** "answer if queried now" = the full state for S5 / CYC / MOD / RDFA, but not FF (only the read
   bit), so the anchor is partial there.
