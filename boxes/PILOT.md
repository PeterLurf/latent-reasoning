# THEORY §66 — throughput pilot and the frozen steps / batch (recorded 2026-10-01 15:37 EDT, before any panel run)

Pilot: seed 99 (not a panel seed), 1000 steps × bs 32, RTX 3090 hosts 53743423 (driver 550.90.07) and 53743427
(570.195.03); runs in `boxes/runs/pil_*` (eval_train.json, log.jsonl; train only).

| arm | s/step (3 concurrent per GPU) | final loss | acc N = 8 "all" (steps 500 / 1000) |
|---|---|---|---|
| NC | 0.124 | 1.18 | 0.031 / 0.016 |
| TC | 0.167 | 0.04 | 0.047 / 1.000 |
| TC2 | 0.125 | 0.11 | 0.047 / 0.172 |
| CL | 0.210 | 1.15 | 0.047 / 0.016 |
| WC k=4 | 0.265 | 0.33 | 0.016 / 0.063 (edit IIA 0.057) |
| WC0 k=4 | 0.137 | 0.24 | 0.063 / 0.109 (edit IIA 0.063) |

**Frozen for the panel (the pilot may fix only steps and batch): 8000 steps × bs 32 for every arm** (256k problems per
arm; same task-data budget). Reason: TC converges by 1000 steps, while the latent / chunked arms are still improving at
1000 steps (WC solver loss 0.33). 8000 steps keeps the wall-clock at ≈ 0.3–0.6 h per run at 4 concurrent runs per GPU.
lr 1e-4 (cosine, 100 warm-up), AdamW wd 0.01, bf16 autocast (unchanged from the implementation).
