# Prior check for §84: the written-state GSM8K arm halts at the trained value count; oracle-stop and decoy padding (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch).

## 1. Halting / stop-decision failure under length generalisation
- **Newman et al. 2010.07174 (BlackboxNLP'20), the direct prior.** Models trained to emit EOS (+EOS) extrapolate far
  worse than −EOS models.
  - −EOS models are evaluated with an **oracle length** (forced to the gold length) and extrapolate up to 10× longer.
  - Mechanisms: hidden states organised by linear position ("length manifolds"), and "length attractors" (once EOS is
    top, generation is trapped).
  - **§84's diagnosis, stopping at the trained count of values, is the same failure** with "#### answer" as EOS. Plan
    (1) is Newman's oracle-length design.
- **Anil et al. 2207.04901** (scratchpad fine-tuning fails to length-generalise; failures include premature
  termination). **Zhou et al. 2310.16028** (RASP-L; length generalisation needs a length-independent algorithm).
- **Premature termination in long CoT:** Wang et al. 2501.18585 (o1-like models abandon reasoning paths early,
  "underthinking"). Not a length-range mechanism, but the same symptom at scale.

## 2. Decorrelating a length cue from the target
- **Positional analogue:** randomised positional encodings, Ruoss et al. 2305.16843. Training with randomly sampled
  position ids so the model can't key on absolute position, which boosts length generalisation. **Decoy padding is the
  same idea applied to the value count.**
- **Distractors in math word problems:** GSM-IC 2302.00093 (irrelevant context degrades LLMs); GSM-Symbolic's NoOp
  clauses 2410.05229.
- **Counterfactual augmentation for spurious features:** Kaushik et al. 1909.12434. Shortcut learning: Geirhos
  2004.07780.
- **Ye et al. 2407.20311 (Physics of LMs 2.1):** GPT-2-size models on synthetic GSM data learn to compute *only
  necessary* parameters. So unneeded values in the state are a recognised distractor regime, and the model can learn
  to ignore them.
- **Not found:** decoy *computed values* in a written state used to decorrelate the halting decision from the count.
  New as scoped.

## 3. Oracle-stop / forced-continuation diagnostics
- Newman's oracle-length evaluation is the standard form: force generation to the gold length and score.
- Faith and Fate (Dziri 2305.18654) decomposes errors into local (step) vs propagation errors. Use the same split:
  - **halting error** = the answer emitted before the true n;
  - **step error** = a wrong equation given a correct state;
  - **propagation** = a correct step on an already-wrong state.

## Design flags
1. **Decoys must cover the test count range.** If padded states never exceed, say, 4 total values while long items
   need 6–8, the count is still bounded and the halting cue moves to "total ≤ 4". Sample the decoy count so that total
   values in training span ≥ the test range (≥ 8). Otherwise it is the range principle again.
2. **Decoys must be marked or distributed so the halting target stays identifiable.** The stop rule must depend on
   *which* values the question needs. Make decoys plausible (binary ops on question numbers, as planned), and make sure
   a decoy never equals a needed intermediate or the answer, or the label becomes ambiguous.
3. **Decoys may also hurt step selection.** The model might pick a decoy as an operand. Report step accuracy on
   in-range items with and without decoys, as a cost check (GSM-IC predicts some degradation).
4. **Oracle-stop first.** If forcing continuation to the true n still fails, the steps themselves don't extrapolate,
   and decoy padding can't fix that. Pre-register: decoys are informative only if the oracle-stop arm recovers a
   substantial part of the long-item accuracy.
