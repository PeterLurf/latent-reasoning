# HALT-HEAD v1: a learned halt head on the frozen §240 loop (C5) ($0, 2026-10-29)

**Status:** Theory's choice under the user's standing rules; option (1) from Audit's two suggestions after §247. Eval-only plus a tiny local head fit, $0 (4060 / CPU). Rules are pre-stated before any output.
- Audit's option (2), C7 TEXT-WRITE, is declined for now. To make the write follow the sentence it must be trained on x ≠ gold, i.e. on edits, so §247's edit test would largely measure its own training data (circular). A non-circular C7 design needs held-out edit types and is left for later.

## 1. Why
- §247: the decoder's own max-prob is confidently wrong at K′ 0–1, so readout halting fails (COSTLY).
- C5 asks for **learned** halting: "spends more steps on harder inputs".
- **H-SIGNAL:** the loop's states carry a decodable "chain complete" signal that a small learned head can read, even though the answer softmax is miscalibrated.

## 2. Head
- **Features at each K′ ∈ 0..8:** the decoder's final-layer hidden state at the answer position (the state that produces the answer logits), detached. The models are frozen.
- **Label (self-consistency, no gold):** y_K′ = [pred(K′) == pred(8)]. pred(8) is the model's own fully thought answer, so no hand-built correctness label is used.
- **Head:** logistic regression on the 768-d feature (L2 1e-3, standardised).
- **Fit split:** 6,000 fresh items from the DF training stream, d ~ U{1..4}, excluding TEST2-DF sigs; 20% held out for the threshold sanity check only.
- One head per model.
- **Halt rule (fixed now):** the first K′ with p_head ≥ 0.90, else K′ 8; censored items count at K′ 8 with the K′-8 answer (Note 247-1).

## 3. Readings (TEST2-DF; binding at d 1–4, as §247, per model)
- **HALT-ACC:** acc at halt (vs gold) ≥ acc(K′ 8) − 0.02 at each d 1–4.
- **HARDER-MORE:** mean K′_halt strictly increases from d 1 to d 4.
- **SAVING:** mean K′_halt ≤ d + 1 at d 1–4.
- **Labels, first match:** COSTLY / NO-SIGNAL / NO-SAVING / C5-LEARNED. Overall: both models, else MIXED.
- **Scope:** "C5-partial (post-hoc learned halt head on a frozen loop, self-consistency labels; externalisation not tested)".
- **Descriptive:**
  - d 5–6 (more passes on harder? halting on wrong items?);
  - never-halt share and event-only mean;
  - head calibration;
  - pass savings vs K′ 8.

## 4. Build
- An eval part `halthead`. It dumps the per-item decoder answer-position hidden state at K′ 0..8 for the fit items and TEST2-DF d 1–6, fits the head (fixed seed), and writes eval_halthead.json in §247's eval_halt format {d: {acc_k8, acc_halt, mean_khalt, khalt_hist, never_halt_n}} plus {head: {train_acc, val_acc, val_auc}}.
- Gate: the hidden dump at K′ 8 reproduces eval_acc's K′ 8 predictions bitwise.
- Local 4060; fp32.

## 5. OC
- The decoder is at 1.000 from K′ = d − 1 and at chance before, so pred(K′) == pred(8) separates cleanly by K′ relative to d.
- If the hidden state encodes progress, a linear head should read it: P(C5-LEARNED per model | H-SIGNAL) ≈ 0.7; prior 0.6, so per model ≈ 0.42 and both ≈ 0.3.
- Under no signal (head = K′-independent): P(HARDER-MORE) ≈ 0.04.

## 6. Censoring
Missing parts mean pending. The fit must reach val_acc ≥ 0.80 or the run is VOID (an uninformative head).

## 7. Cost
$0.
