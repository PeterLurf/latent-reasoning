# LEN-STEP v1: is the d ≥ 5 failure about text length or step count? ($0 diagnostic, 2026-10-28)

**Status:** Theory's choice, eval-only, $0 (local 4060). It replaces the withdrawn OFFSET draft (REVIEW_OF.md). Readings are pre-stated before any output.

## 1. Why
REVIEW_OF found that, at the same pass and step, accuracy falls with d, and that training texts are short (≤ 94 tokens at d 4). Two explanations remain:
- **LENGTH:** the trunk's read at ':' degrades on texts longer than those trained;
- **STEPS:** something about chain length itself, i.e. the number of stored states or the step position.

The next C2 lever depends on which one holds.

## 2. Cells (new seeds, DF law, 1,000 items each)
- **A:** d 4, normal (n_dis ~ U{0..2}), the TEST2-DF law with a new seed.
- **B:** d 4, padded. Extra distractor facts n_dis_extra are added so the token-length distribution matches cell C. Gate: mean length within ±5 tokens of C's, and the 90th percentile within ±10.
- **C:** d 6, normal.
- **D:** d 6, short: n_dis = 0, keeping only items with length ≤ cell A's 90th percentile. If fewer than 300 qualify out of 5,000 draws, D is reported as unavailable and the STEPS reading uses C − B instead.
- **Build:** an OW.Stream kwarg `n_dis_extra` (default 0 = bitwise current; distractor names from the same pool; extra rng draws only when > 0), and an eval part `lenstep` → eval_lenstep.json {cells: {A..D: {n, tok_mean, tok_p90, acc (K′ 8), ws {t}}}}.

## 3. Readings (per model: lh_fh_s1, lh_fh_s2; descriptive diagnostic, pre-stated)
- LEN4 = acc(A) − acc(B): the length effect at fixed steps (d 4).
- STEP = acc(A) − acc(D): the step effect at matched short length. If D is unavailable, STEP' = acc(B) − acc(C): the step effect at matched long length.
- LEN6 = acc(D) − acc(C) (when D exists).
- **Labels:**
  - LENGTH: LEN4 ≥ 0.10 and STEP < 0.10;
  - STEPS: STEP ≥ 0.10 and LEN4 < 0.05;
  - BOTH: both ≥ 0.10;
  - NEITHER: otherwise.
- The label for the next lever needs both models to agree; otherwise it is MIXED and both factors stay open.
- **Pre-named next lever:**
  - LENGTH → training with length augmentation (padding at d ≤ 4);
  - STEPS → longer-depth training or the state-addressing question;
  - BOTH → length augmentation first (it is cheaper).

## 4. Cost
$0 (local eval). Build: the kwarg, the length-match gate, a flags-off bitwise gate, Audit G-D.
