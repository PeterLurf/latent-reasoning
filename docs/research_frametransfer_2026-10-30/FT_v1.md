# FRAME-TRANSFER v1: does the language loop's chain transfer to unseen sentence frames? ($0, 2026-10-30)

**Status:** Theory's choice under the user's standing rules, instead of Audit's C4 distillation suggestion (declined below). Eval-only, $0 (local 4060). Rules are pre-stated before any output.

## 1. Why
- All §237–§248 readings use TEST2-DF, which draws on the training frames 0–5.
- The roadmap toward a task-agnostic latent reasoner (memory: path to GSM8K) needs the latent chain to survive new phrasings.
- build_sets already holds a TRANSFER set (frames 6–7, never trained; e.g. "Joe holds the keys of Mia less those of Tom."), 1,000 items per d 1..12.
- **Declined (Audit's distilled-CoT C4 step):**
  - a text-CoT trained on answers only cannot emit the step sentences, since GPT-2 does not produce this format zero-shot;
  - one trained on step sentences uses simulator steps, and at 1.000 its outputs equal the existing aux targets (REVIEW_AO 8).
  - Either way it adds nothing over P-2, so no C4-admissible teacher exists in this synthetic setup.

## 2. Evals (lh_fh_s1 / s2, plus CD-AN df_pcd4_s1; TRANSFER-DF built with mul_s_p 2, d 1–6; K′ 8; fp32; one eval_bs per model)
- **Part `transfer` → eval_transfer.json:**
  - {by_d: {d: {acc, ws {pass t: acc}}}} (LAT);
  - {by_d: {d: {acc}}} (CD-AN);
  - cue_only {d: {acc, majority}} (LAT, d 2–4: think passes on the cue only, decoder on the full transfer text).
- **Gate:** the transfer set sha is printed and equals build_sets' transfer sha.

## 3. Readings (theory/score_ft.py; fixtures 5/5); per model
- **T-ACC (binding):** acc at d 2, 3, 4 each ≥ 0.80 and ≥ CD-AN_transfer + 0.25.
- **T-CHAIN (binding):** WS ≥ 0.50 at every pass 2..d, d 3–4.
- **T-GUARD:** cue-only ≤ max(majority, CD-AN) + 0.05 at d 2–4, else VOID.
- **Labels:** VOID / TRANSFER (T-ACC and T-CHAIN) / ACC-ONLY / NO-TRANSFER. Overall both, else MIXED.
- **Descriptive:** d 1, d 5–6, and the drop vs TEST2-DF at each d.

## 4. OC
- The trunk must parse unseen phrasings of the same operations. The askname init and every later stage saw frames 0–5 only.
- P(TRANSFER per model) ≈ 0.4; both ≈ 0.3.
- Under "the loop keys on frame-specific surface cues": P(T-ACC) ≈ 0.05.

## 5. Cost
$0.
