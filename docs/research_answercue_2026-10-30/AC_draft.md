# ANSWER-AT-CUE draft: supervise the final answer at the pass whose cue asks the final question (C4) (2026-10-30)

**Status:** draft for red team + Literature. Theory's choice after Note 251-9 (with its probe addendum).

## 1. Why
- **Note 251-9:** in every failed answer-only run, the pass-1 ':' code on d 2 items stops carrying v_1 (probe u1|v1 0.11–0.25 vs init 1.000), while d 1 items keep it.
  - The answer CE reaches the codes only through the hold passes and a decoder that can take the last step itself.
  - So nothing pushes a d 2 item's pass 2 to compute v_2, and the pass-1 code drifts.
- **Key fact about the plan-given format:** the pass-d cue *is* the final question ("How many <u> does <S_d> have? The answer is:").
  - The model's own LM head at that ':' answering it is the askname competence the loop was built on (G-AW0).
  - Supervising that ':' with the item's **final answer** uses no intermediate value. Each item contributes one label, its answer, at the pass that asks its question.
- With d ~ U{1..D}, every pass t ≤ D gets labels from the depth-t items, so every pass is shaped by final answers alone.
  - The pass-d ':' cannot shortcut: one pass cannot do two steps (CD-AN 0.097 at d 2).
  - So a correct pass-d answer requires reading v_{d−1} from S_{d−1}. The gradient path to the pass-(d−1) code is one hop, not a route through the hold passes and the decoder.
- **Question:** does answer-at-cue (ACUE) form the loop from askname without per-pass targets, and extend it?

## 2. Arms (askname init s233_askname_s1; REG law; d ~ U{1..4}; K 8; 16k; RTX 3090; micro pinned)
| Arm | Runs | Loss |
|---|---|---|
| ACUE | acue_s1 / s2 | decoder answer CE + λ · CE(colon_logits at pass d's ':' , answer), λ = 1, no aux (`--aux_fade 0,0`) |
| AO4 (comparator) | ao4_s1 / s2 | decoder answer CE only, the same recipe (the §244 recipe at REG d ≤ 4) |

- Args diff ACUE vs AO4 = {cue_ans 1.0}.
- The ':' logits restricted to the 11 number tokens (as G-AW0 reads them)? To be fixed in v1.
- **Build:** `--cue_ans λ`. On each item it adds CE at the pass-d ':' position. inject_aw already computes colon_logits for WS (inject_aw.py:58-60). Default 0 = bitwise current.

## 3. Readings (score_ae gates, d 2–4; per seed)
- **Formation:** d 2 acc ≥ 0.9; d 2 at K′ 1 ≤ 0.2 (the pass-1 code alone does not answer); probe u1|v1 ≥ 0.8 (descriptive).
- **Extension:** the §251 d 4-centred gates (ACC4, NOT-DECODER, SWAP4, LP5-D4).
- **Labels:** EXTEND / FORM-D2-ONLY / NOT-FORMED / VOID / UNSTABLE.
- **Caveat:** WS at pass d is trained in ACUE, so it is not evidence there. SWAP-MID and LP5 are the causal gates.

## 4. OC (sketch)
- P(EXTEND per seed | a one-hop answer path suffices) ≈ 0.35; P(EXTEND | per-pass targets on d > t items are needed) ≈ 0.05.
- AO4 is expected to fail (§244 / §245 / AEX0).

## 5. Cost
- 4 × 16k at ~$0.8 ≈ $3.2; guard $4.

## Open questions for the red team
- Does pass-1 code drift on d ≥ 2 items still happen when only the depth-1 items label pass 1?
- Is this C4-admissible: is the answer at the cue pass an "intermediate state"?
- Is the comparator needed, given AEX0 and AO?
