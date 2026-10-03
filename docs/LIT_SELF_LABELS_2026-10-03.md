# Prior check for §96: per-op targets from a final-answer-trained free-table teacher (amortised self-labels) (2026-10-03)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified via the arXiv API (this note and earlier
ones); speech classics are non-arXiv journal references.

## 1. Closest prior
- **Hybrid DNN-HMM forced alignment (speech), the closest structural precedent.** A GMM-HMM trained by EM on
  utterance-level transcripts (no frame labels) produces Viterbi *forced alignments*. These become frame-level targets
  for a neural acoustic model, often iterated (realign with the DNN, retrain). Hinton et al. 2012, *IEEE Signal
  Processing Magazine*; Dahl et al. 2012, *IEEE TASLP* (non-arXiv). The same shape as §96: a latent-variable model
  trained by marginal likelihood on sequence-level labels supplies per-step targets to a neural student.
- **Amortising a slow, reliable learner into a network:** expert iteration (Anthony 1705.08439); the
  teacher-automaton → GPT-2 cell candidate in LIT_STABLE_LATENT (a), with priors Rabusseau 1807.01406, NAR, CLRS.
  - The difference here is that the teacher was trained on **final answers only** (plus short problems), and the
    student is the latent-restart pipeline, not a transition cell.
- **Self-generated intermediate labels filtered by final answers:** STaR 2203.14465 (rationales kept if the answer is
  right); noisy-student pseudo-labelling 1911.04252. Our own Pilot L/M (restart-teacher distillation; agreement
  filtering helped, M-2) and §65 (restart labels = true labels).
- **Latent-variable reasoning via EM:** TRICE 2312.02179, BoLT 2503.18866. Their posterior samples are the labels.
  None certifies.

## 2. Has "C4 via a final-answer-trained discrete teacher" been claimed for latent reasoning?
- **Not found.**
  - RELAY 2502.08482 uses a looped teacher to produce CoT for an autoregressive student, but the teacher itself trains
    with iteration-wise intermediate supervision.
  - Abstract-CoT 2604.22709 self-distils from its own verbal-CoT-trained model.
  - Speech forced alignment is the domain precedent for the pipeline shape.
- **New as scoped:** "per-op latent targets from a certified automaton acquired from final answers (with ≤ p short
  problems), amortised into an LM's latent restarts".

## 3. Pitfalls
1. **Uncertified teachers give wrong labels exactly where they matter.**
   - §85: at p = 0.03 all 5 seeds score 1.000 at N 128, but only 2/5 certify. Rarely visited codes stay wrong, and
     those are the rare prefixes.
   - At p = 0.05, expect a mix. Use **only certified teachers**, or report per-position label agreement *and* the
     student's error on disagreeing positions.
   - §62 showed that *any* label noise installs count/age dependence in students. Near-clean labels are required, not
     just high average agreement.
2. **Over-complete codes.** Supervising in the decoded-state space D(code) (not code ids) is right. Duplicate codes per
   state then don't matter, provided D is exact on every code the teacher visits on §92's training distribution,
   including the longest training sequences. The certificate checks reachable (code, op) pairs, so it covers this if it
   was run on the reachable set at those lengths.
3. **Redundancy of the student run.**
   - If label agreement on §92's training set is exactly 100%, the student's training data are *identical* to §92's,
     so retraining reproduces §92 by construction (bit-identical given seeds).
   - The informative measurements are then:
     - (a) teacher certification at p = 0.05 / 0.1;
     - (b) label agreement.
   - Consider running the student only if agreement < 100%, where it tests robustness to label noise. That is a
     useful C4-direction result in its own right (§62's lesson).
4. **Claim wording.** "C4 for the pipeline, qualified by the teacher's short-problem fraction p." The acquisition
   result belongs to the free table (§76/§85). The LM contributes hosting and latent execution (C1/C2/C3 from §92),
   not acquisition. This is Audit's earlier "system-level C4" point (LIT_STABLE_LATENT (a)).
