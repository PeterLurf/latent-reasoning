# PROTOCOL8b pilot — re-anchoring a trained Coconut GPT-2 with a written checkpoint (exploratory)

Written 2026-09-30 11:14 EDT, committed before the pilot was run on real data (only a 6-problem CPU smoke test for mechanics).
Exploratory: it informs PROTOCOL8b's confirmatory registration and makes no confirmatory claim.

**North-star conditions:**
- C2: restarting resets the loop's iteration count.
- C7: the checkpoint is a human-readable sentence the model reads back.
- C5: a hand-built form of "write a checkpoint".

**Why.** THEORY §26 and §32: on the toy, restarting from the frontier's entity rescued 31/34 failures, and distilling
that teacher fixed 19/20. PROTOCOL8 Phase 0: Coconut GPT-2 learns deep ProsQA only to about 5–6 hops (0.55 at depth 8,
chance from 10).

**Design** (`prosqa/reanchor_p8.py`, sha256 b168369a71857a45…):
- **Model and data:** the Phase 0 checkpoint `prosqa/ckpt/p8_coconut_checkpoint_48` (answer mask on), K′ = 10, on
  `p8_test.json`, 150 problems per m.
- **Sweep head:** the (layer, head) is fixed on 100 calibration problems (m ≤ 2, excluded from evaluation). It is the
  one putting the most attention mass on gold-path statements from the thought positions.
- **Anchors:** the objects of the top-3 statements under the sweep head at the last thought.
- **Restart:** each anchor Y is written as the text statement "\<root\> is a Y." and a fresh 10-thought episode is
  run. The answers are max-combined per option with the baseline episode.
- **Controls:** random anchors (3 random concept entities) and an oracle anchor (the gold-path entity 4 hops before the
  target).

**Expectations (not decision rules):**
- If the account transfers, re-anchoring lifts depth 7–10 over the baseline, the oracle lifts it more, and random
  anchors do not help (or hurt, since writing a false checkpoint can make the wrong option reachable).
- If re-anchoring and the oracle both fail, the model cannot use a written checkpoint, and PROTOCOL8b needs training
  on checkpoints (distillation) before inference can use them.

## Result (2026-09-30 12:24 EDT; `prosqa/res_p8b/p8b_pilot.json`, sha-checked; Vast 53544379 destroyed after the fetch)

- **Sweep head:** layer 10, head 11 (from calibration). Its top-1 anchor is on the gold path in **56 %** of problems,
  so the decode is informative.
- **Accuracy by answer depth, K′ = 10 (150 per m):**

  | Depth | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | baseline | .74 | .76 | .73 | .70 | .62 | .58 | .51 | .52 | .48 | .50 |
  | re-anchor (top-3) | .75 | .70 | .68 | .62 | .61 | .53 | .52 | .48 | .52 | .51 |
  | random anchors | .74 | .75 | .73 | .70 | .58 | .57 | .50 | .49 | .49 | .52 |
  | oracle (4 hops before target) | .74 | .76 | .73 | .70 | .61 | .51 | .46 | .43 | .46 | .51 |

- **Reading:**
  - A written checkpoint does not help this model, even when it is correct (oracle).
  - Re-anchoring slightly hurts the shallow depths, where the max-combination admits false checkpoints.
  - The model does not use the new shortcut edge ("root is a Y") to shorten its latent search. That fits a
    one-pass conduit that reads fixed positions rather than doing search over statements (Phase 0; addenda G, J, Q).
  - This is the second pre-stated branch: **PROTOCOL8b has to train checkpoint use** (re-anchor distillation, or
    training with written checkpoints in the data) before inference-time restarts can work. THEORY 41-2 also points
    to deeper tight-budget training as a lever.
