# PROTOCOL8b step 1 — can Coconut GPT-2 be taught to use written checkpoints?

Written 2026-09-30 12:37 EDT, committed before any step-1 run started. Append-only.

**North-star conditions:**
- C5 / C7: a readable written checkpoint the model uses.
- C2: a route past the fall-back.
- C4: final-answer labels only.

**Why.** The PROTOCOL8b pilot (`docs/PROTOCOL8B_PILOT.md`) found that the Phase 0 Coconut model ignores written
checkpoints, even correct ones: oracle gain ≈ 0 at depth 7–10. Re-anchor distillation (THEORY §32) needs a teacher
that beats the student, so checkpoint use must be trained first (theory session, 2026-09-30 ~12:25).

## Design

- **Base model:** the Phase 0 checkpoint `prosqa/ckpt/p8_coconut_checkpoint_48` (sha-checked on upload).
- **Fine-tune:** at the final stage with 10 latents and the answer mask (`resume: 44` → 48, i.e. 4 epochs, fresh
  optimizer, lr 1e-4, the Phase 0 recipe otherwise). Configs `prosqa/p8b1_<arm>_s<1..3>.yaml`; pod script
  `prosqa/pod_p8b1.sh`.
- **Arm `ckpt`:** trains on `p8data/p8_train_ckpt.json` (`prosqa/make_ckpt_data.py`, sha256 51a5cddf8b3c557f…; md5 801f3d54bf, seed 0).
  - 50 % of the problems carry an inserted **true** written checkpoint "\<root\> is a \<Y\>", where Y is the node k gold
    hops in, k ~ U{1..depth−1}. In ProsQA's (transitive) semantics this statement is true.
  - The checkpoint goes at a random statement position. The label is the unchanged final answer.
  - No false checkpoints in step 1.
- **Arm `ctl`:** the same fine-tune on `p8_train.json`, with no checkpoints.
- **Seeds:** 3 fine-tune seeds per arm (6 runs on 24 GB Vast cards, ~).
- **Held-out labelled probe:** `p8data/p8_test.json` (never used in training), 150 problems per m. The first 100 m ≤ 2
  problems pick the sweep head for decoded anchors and are excluded. There is no other filter. Evaluated with
  `prosqa/reanchor_p8.py --random_pos` (sha256 545098eee3b43a1c…) at K′ = 10.
- **Conditions:**
  - base (no checkpoint);
  - **oracle** checkpoint (the gold node 4 hops before the target, inside the training support U{1..depth−1});
  - random checkpoints: 3 random concepts, i.e. **false** statements;
  - the model's own decoded anchors (top-3).

  Each is scored at a random statement position (primary; the training distribution) and end-appended (as in the
  pilot). Answers are max-combined with the base episode, as in the pilot.

## Outcomes (`prosqa/score_p8b1.py`, sha256 a2703358d99cc829…, written before data)

Per seed, on held-out problems at answer depth 7–10:
- OG = acc(oracle, random position) − acc(base);
- RG = the same for random checkpoints;
- REG = the same for decoded anchors.

**P8b1-1 (primary):** supported if
- mean OG(ckpt) ≥ 0.10;
- mean OG(ckpt) − mean OG(ctl) ≥ 0.05;
- every ckpt seed's OG exceeds every ctl seed's (complete separation; exact one-sided MW p = 1/20 = 0.05).

Not supported if the mean difference ≤ 0; otherwise inconclusive.

**P8b1-2 (control, two-sided), per arm:**
- mean RG within ±0.02: no gain from false checkpoints;
- RG > 0.02: spurious gain;
- RG < −0.02: harm, meaning blind trust in written statements, which would motivate step 2's verification.

**Secondary:** end-appended OG; REG (do the model's own anchors now help? The precondition for §32 distillation);
plain deep accuracy by arm; all of it by depth.

**Next.** Only if P8b1-1 is supported: step 2, re-anchor distillation with the model's own decoded anchors
(label-free), plus false-checkpoint training for verification if P8b1-2 shows harm.

**Note (2026-09-30 13:17, file naming only).** `prosqa/pod_p8b1.sh` writes `res/p8b1_<cfg>.json` with
cfg = `p8b1_<arm>_s<seed>`, i.e. `p8b1_p8b1_<arm>_s<seed>.json`. The scorer reads `p8b1_<arm>_s<seed>.json`. Fetched files
are renamed to the scorer's names before scoring. No content changes.
**Note (2026-09-30 13:27).** `ckpt_s1` (53555197) hung for 15 minutes while loading the base checkpoint: GPU idle,
~0 CPU. It was killed and restarted from scratch on the same host (same seed and config; the base-checkpoint sha had
already been verified there), and now trains normally. No data existed from the hung attempt; its log is kept as
`prosqa.out.hung_attempt1`.

**Descriptive reference (2026-09-30 15:22, before any step-1 result was read).** The un-fine-tuned Phase 0 model is
evaluated with `reanchor_p8.py --random_pos` (same probe, same seed) on the local RTX 2070. This gives OG, RG and REG
**before** any checkpoint training, including the random-position variant the pilot lacked. It is a reference only; no
decision rule uses it.
