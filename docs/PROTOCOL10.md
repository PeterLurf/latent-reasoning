# PROTOCOL10 — segmented Coconut with written entity checkpoints (inference only)

Registered 2026-09-30 18:31 EDT, before any PROTOCOL10 output exists and before the PROTOCOL8b step-1 verdict (ctl_s2 still in its
probe). Design: Theory, THEORY §47/§48 (message of ~18:15), implemented by Experiments. North-star tags: **C2** (reach
beyond the trained depth), **C5 v0** (heuristic write and halt, not learned), **C7** (edit test on the written text),
**C3** reference (the PROTOCOL8 text-CoT arm on the same depths, descriptive, once scored).

## Question
Does chaining isolated latent segments, each restarted from an entity checkpoint written as text, carry a
checkpoint-trained Coconut GPT-2 past its trained depth? On the toy one segment covers its trained horizon at any
absolute depth (§46), and single-anchor chains reach d19–24 (§48).

## Models and data
The 6 PROTOCOL8b step-1 models (`prosqa/ckpt/p8b1_{ckpt,ctl}_s{1,2,3}_checkpoint_48`); the Phase 0 model
(`p8_coconut_checkpoint_48`) as a descriptive reference. Data `prosqa/p8data/p8_test.json` (md5 ff5baa26e4), answer depth
11–14 (beyond the trained ≤ 10): the first 100 problems per depth for 11–13 and all 34 at depth 14 (334 problems).

## Procedure (`prosqa/segchain_p10.py`, K′ = 10 per segment, S = 4, seed 0)
1. Sweep head calibrated as in `reanchor_p8.py` (first 100 problems with ext ≤ 2; uses gold-path labels on those shallow
   problems, a disclosed C4 caveat of the v0 write policy).
2. Segment 1 = a plain K′-thought episode. Write: at the segment's last thought, the top statement under the sweep head
   whose object is neither the root nor already written; its object Y becomes "<root> is a Y." at a random statement
   position. Segment s + 1 = a fresh K′-thought episode on the new text: **isolation** — no earlier latent thought is
   visible, the written text is the only channel between segments. Halt when Y is an answer option or no candidate is
   left; larger S reuse the halting segment's answer.
3. Answer at S = the margin of segment S (or of the halting segment). Accuracy = share of margins > 0.
4. Edit (C7): on problems whose first write is reachable from the root, re-run segment 2 with the written entity
   replaced by a distractor at the same hop distance that reaches the negative option but not the target (skipped if none).

## Predictions and decision rules (Theory's, fixed now)
Gain G(model) = acc(S = 4) − acc(S = 1), depths 11–14 pooled over the 334 problems.
- **P10-1 (primary):** supported if mean G over the 3 ckpt seeds ≥ 0.10 and every ckpt seed has G > 0; not supported
  if mean G(ckpt) ≤ 0; otherwise inconclusive.
- **P10-2 (control):** mean G over the 3 ctl seeds ≤ 0.02 (as the pilot's null predicts); > 0.02 is reported as a
  chaining gain without checkpoint training.
- **P10-3 (arm difference):** mean G(ckpt) − mean G(ctl) ≥ 0.05 with complete separation (every ckpt G above every
  ctl G; exact one-sided Mann–Whitney p = 1/20).
- **Secondary (descriptive):** G at S = 2 and 3; accuracy by depth × S; edit-follow rate = share answered with the
  negative option when edited minus unedited (Theory's claim threshold 0.5); write quality (first write on the gold
  path, its hop distance, the share of unreachable writes); halting rate. If step-1 models use checkpoints (P8b1-1) but
  P10-1 fails, the reading is that the write policy limits (stage B next), not reading.

## Blinding
Runs start before the P8b1 verdict. The script prints progress only; `prosqa/res_p10/*.json` are read only after the
P8b1 verdict is committed. Local cards only (3070, then the 4060 when Theory frees it); Vast freeze holds.

## Amendment 1 (2026-09-30 19:21 EDT, before any PROTOCOL10 output was read): exploratory segment-length variants
Not part of the decision rules. The same script and models, with (a) K′ = 5 thoughts per segment, S = 8 (Theory: a
segment should span ≤ 4–5 hops) and (b) K′ = 10, S = 8 (longer chains). Outputs `prosqa/res_p10/<model>_k<K′>_s8.json`,
blind under the same rule. Run on the idle 3070/2070 (`scratch/p10v_worker.sh`).

## Amendment 2 (2026-09-30 20:18 EDT, before any PROTOCOL10 output was read): exploratory shallow-depth check
Not part of the decision rules. Theory (§48) saw shallow accuracy erode with more segments on the toy. The same
models and the K′ = 5 / 10, S = 8 settings of amendment 1 are run on answer depths 3–6 (100 per depth). New flag
`--exclude_cal` drops the 100 calibration problems (depths 3–6) from the evaluation; the main run (d11–14) is unaffected,
since no calibration problem is that deep. Outputs `prosqa/res_p10/<model>_k<K′>_s8_d3-6.json`, blind under the same rule.
