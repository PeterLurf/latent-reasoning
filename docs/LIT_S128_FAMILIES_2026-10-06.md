# Prior check for §128: one fixed recipe across state-tracking families (S5, flip-flop, CYC, MOD) (2026-10-06)

Literature session, for Theory; Audit copied. Delétang et al. specs read from the paper's PDF (2207.02098,
appendix and §4); other IDs as in LIT_INDEX.

## (a) Delétang et al. 2207.02098: exact specifications
- **Protocol:**
  - Train lengths ℓ ~ U(1, N), N = 40, sampled fresh each batch (batch size 128, Adam, 1M steps).
  - Test ℓ ~ U(41, 500).
  - **Score = per-sequence accuracy of the final output, averaged over unseen lengths.**
  - A model "generalises" if the score is ≥ 90%. Chance is 20% for the 5-class tasks.
  - Code: github.com/deepmind/neural_networks_chomsky_hierarchy.
- **Cycle Navigation (R):** a cycle of length 5. Moves STAY / INCREASE / DECREASE, encoded {0, 1, 2}. The agent
  starts at 0; the output is the end position (5 classes). It is listed as **permutation-invariant**.
  - Scores: RNN, Stack-RNN, Tape-RNN and LSTM 100.0; Transformer 61.9.
  - **Your CYC matches it** (word tokens instead of {0, 1, 2}).
- **Modular Arithmetic (Simple) (R):** numbers {0..4} and operations {+, −, ×}, result mod 5. The input alternates
  number and operator (odd length, e.g. "1 + 2 − 4" → 4). The paper says it is solvable by a 5-state FSM "with the
  state being equal to the current result", i.e. a left-to-right fold, not operator precedence.
  - Scores: RNN, Stack-RNN, Tape-RNN and LSTM 100.0; Transformer 24.2.
  - **Your MOD is the same task with an implicit leading "0"** (start value 0, items "op d"). It's fair to call it
    "Modular Arithmetic (Simple), Delétang et al." with that encoding note. Their DCF "Modular Arithmetic" (with
    brackets) is a different task, so don't confuse the names.
- **Protocol mismatch, to report:** they test final-output accuracy averaged over lengths 41–500. You use cumulative
  all-positions accuracy at 512. **Also report Delétang's score** (final output, U(41, 500)) for CYC and MOD, so the
  published RNN / LSTM 100 and Transformer 61.9 / 24.2 numbers are directly comparable.

## (b) Priors and novelty for "one fixed recipe, exact and causally serial discrete latent state, across families, in a pretrained LM"
- **One architecture across tasks, from scratch:**
  - Delétang 2207.02098;
  - Ruoss 2305.16843 (randomised positional encodings across the suite);
  - MLP-LDRU 2605.26035 (near-saturates the regular tasks);
  - DeltaProduct 2502.10297 / Grazzi 2411.12537 (groups, parity, modular arithmetic);
  - Liu 2210.10749 (transformers learn shortcuts to automata across several semiautomata);
  - Merrill 2404.08819.
- **One recipe across tasks with explicit scratchpads:** Turing Programs 2407.03310.
- **Pretrained LMs:**
  - Li 2503.02854 (S3 / S5 fine-tuning; cut-off at the training length);
  - 2602.14814 (Qwen fails S5 in code traces).
  - **None reports a certified, causally serial latent state across several families.**
- **Verdict: new as scoped.** The from-scratch "one architecture across the Chomsky suite" line is the comparison
  class. Cite Delétang and Liu as the cross-task precedents.

## (c) Traps
1. **The item-identity code can't satisfy either task.** The next position or value depends on the current state for
   every item except the identities ("stay", "plus 0", "minus 0", "times 1").
2. **CYC has a counting shortcut.** It is abelian and permutation-invariant: position = (#INC − #DEC) mod 5.
   - A code that stores the *unreduced* count (a counter, not mod 5) and reduces only at the readout is exact as long as
     the count stays in range.
   - With 8⁸ FSQ codes, counts up to ±512 fit, so **"exact at 512" does not show a 5-state automaton**. This mirrors
     §120 cell 1's open machine (537k codes).
   - **Make the closure / minimal-automaton check primary for CYC:** reachable codes vs length, and Moore-minimised
     states (should be 5). Add a test at lengths ≫ 512 (e.g. 4,096 or 8,192); a counter fails when it overruns its range.
   - Transformers' partial 61.9 on CYC comes from this permutation invariance (Delétang §5).
3. **MOD's resets shorten the effective dependency.**
   - "times 0" (1 of 15 items) resets to 0, so only the suffix after the last reset matters. Its typical length is about
     15 whatever the total length.
   - Long-length accuracy is therefore much easier than the nominal length suggests (the same for flip-flop writes).
   - Report the effective-suffix length distribution at test, and the accuracy binned by it. Optionally a variant
     without "times 0" (pure group part on nonzero values plus …).
4. **Curriculum vs Delétang's U(1, 40):** your 2 → 32 curriculum is a different training distribution. Fine, but state
   it when citing their numbers.

## (d) Are two held-out families enough?
- **The family set is well chosen in Krohn–Rhodes terms:**
  - S5 (a non-solvable group);
  - CYC (an abelian group, Z5);
  - flip-flop (a reset, the aperiodic primitive);
  - MOD (a monoid mixing a group action with a reset: ×0 resets, ×2/3/4 permute the nonzero values, ± translate).
  - Every finite automaton decomposes into group and reset components, so this covers both primitive kinds and one
    mixture.
- **Recommendation: fix one more held-out family now, a *random* minimal DFA** (generated from a fixed seed: e.g.
  6–10 states, 3–4 input symbols, rejected if it is a group, a pure reset or not minimal). It tests "agnostic" beyond
  hand-picked algebraic tasks.
  - It has no published baseline, so it is descriptive for the TASK-AGNOSTIC rule, or included in it, but decide now.
  - Allchin 2607.11953 used held-out random permutation automata the same way.
- **Scope of any pass:** "task-agnostic across regular-language state tracking (group, reset and mixed)". Not beyond
  regular languages (Delétang's DCF / CS tasks need a stack or tape).
