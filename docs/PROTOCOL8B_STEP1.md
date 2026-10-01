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

## Exploratory follow-up: hop sweep (added 2026-09-30 18:07 EDT, before any sweep output was read; not part of the registration)

`prosqa/hopsweep_p8b.py --skip_probe` (`scratch/p8b_hopsweep.sh`, local 2070, started 17:44): for the 6 step-1 models, on
p8_test problems disjoint from the P8b1 probe (40 per depth 7–10), write the oracle checkpoint k gold hops in, for every
k = 1..depth−1, at one random position, and record accuracy by depth × k and by hops remaining (depth − k). It prints no
accuracies; the outputs `prosqa/res_p8b/hopsweep_*.json` are read only after the P8b1 verdict is committed (Audit guard).
Stated expectation, from Theory's toy result (THEORY §46): accuracy is high while the hops remaining are within the model's
in-distribution reach (≈ 5–6 hops, going by Phase 0), is near chance beyond it, and is flat across absolute depth at fixed hops remaining.
Ckpt-trained models are expected to show this more strongly than ctl models. Descriptive only; no decision rule.

## Deviation note (2026-09-30 19:01 EDT): ctl_s2 probe re-run locally

ctl_s2's host (Vast 53555209, RTX A5000) finished training at ~18:05 (TRAIN_EXIT 0) but ran the probe ~10× slower than
the other hosts: 200 of ~1,650 problems in ~65 min, so ~9 h in total. Its checkpoint_48 was fetched at 18:58 and
verified against the host by sha256 (b40721ad…). The unchanged probe command (`reanchor_p8.py --n_per_m 150 --Kp 10
--random_pos`, seed 0) runs locally on the 4060 from 18:59 (`scratch/p8b1_ctl2_local.sh`, output
`prosqa/res_p8b1/p8b1_ctl_s2_local.json`). Rule, fixed now: the local output is ctl_s2's P8b1 result. The Vast probe
is stopped and the host is destroyed once the local run has finished. The GPU differs (4060 vs A5000; TF32 is off in both),
so near-tie margins could differ in principle; no result of either run has been read.
Update 19:03: host 53555209 destroyed at 19:02, before its probe finished (DESTROYED_VERIFIED; its prosqa.out saved in
`prosqa/res_p8b1/logs_ctl_s2/`). Fallback: if the local run fails, it is re-run locally from the same verified
checkpoint. The 4060 re-run is fp32 on a different card, so small numeric differences from the other five seeds' hosts
are possible.

## Result (2026-09-30 20:47 EDT; `python prosqa/score_p8b1.py` → `results/p8b1.json`; all 6 seeds, ctl_s2 from the local re-run)

**P8b1-1: inconclusive.** OG(ckpt) = −0.025 / −0.052 / −0.038 (mean −0.038), OG(ctl) = −0.080 / −0.102 / −0.117
(mean −0.100). Difference +0.061 ≥ 0.05 with complete separation (exact one-sided MW p = 0.05), but mean OG(ckpt) is
negative, far below the 0.10 bar. A written correct checkpoint still lowers accuracy; checkpoint training only
reduces the harm (Phase 0 model: −0.133).
**P8b1-2:** RG(ckpt) mean −0.004 (no gain from false checkpoints); RG(ctl) mean −0.027 (harm: blind trust).
**Secondary (d7–10).** BASE ckpt 0.530 / 0.545 / 0.535 vs ctl 0.570 / 0.563 / 0.573 (checkpoint training costs ≈ 0.03
of plain accuracy; both near chance = 0.5). OGe ckpt mean −0.018, ctl −0.031. REG (own anchors) ckpt −0.016, ctl −0.080.
Oracle-at-random-position accuracy by depth (ckpt / ctl): d7 0.54 / 0.53, d8 0.52 / 0.52, d9 0.48 / 0.40, d10 0.46 / 0.43.
Phase 0 reference (`res_p8b/p8b1_base_ref.json`, descriptive): BASE 0.558, oracle_rp 0.425.
**Reading.** Four epochs of fine-tuning with checkpoints on half the training problems taught the model to be hurt less
by written statements, not to use them: even the oracle checkpoint four hops before the target, which leaves a problem
the model solves at d3–5, does not help. One caveat: the registered combination takes the max per option over the base
and checkpoint episodes, which can add false positives (THEORY §50). **Step 2 (conditional on P8b1-1 supported) is not run.**

## Exploratory hop sweep: result (2026-09-30 20:48 EDT; read after the P8b1 verdict; 5 of 6 models, ctl_s2 still running)

Single-episode accuracy (no max-combination) with the oracle checkpoint written k hops in, by hops remaining (depth − k),
mean over seeds (`prosqa/res_p8b/hopsweep_*.json`, probe-disjoint p8_test problems):

| hops remaining | 0 (plain) | 1 | 2 | 3 | 4 | 5 | 6 | 7–9 |
|---|---|---|---|---|---|---|---|---|
| ckpt (3), d7–10 | 0.51 | 0.48 | 0.42 | 0.40 | 0.44 | 0.49 | 0.50 | 0.50–0.52 |
| ctl (2), d7–10 | 0.55 | **0.13** | 0.21 | 0.28 | 0.40 | 0.49 | 0.53 | 0.46–0.53 |
| ckpt (3), d11–13 | 0.50 | 0.48 | 0.44 | 0.47 | 0.49 | 0.49 | 0.46 | 0.49–0.61 |
| ctl (2), d11–13 | 0.45 | 0.24 | 0.29 | 0.39 | 0.39 | 0.45 | 0.43 | 0.45–0.56 |

**The stated expectation (THEORY §46 toy shape: high when few hops remain) is not met.** A correct checkpoint close to the
target drives the untrained-to-use (ctl) models *below chance* (0.13 at one hop remaining: they systematically pick the
other option), and checkpoint training only lifts this to chance (≈ 0.45–0.50). Neither arm reads the checkpoint as a
premise to reason from. The ctl pattern looks like a learned cue that works against a statement linking the root to
an entity near an answer option. To diagnose (Theory): which option and which statements the checkpoint episodes attend
to. Consequence: restart-from-text schemes on these Coconut models (PROTOCOL10, F2 on GPT-2) start from a model that
misreads written statements.
