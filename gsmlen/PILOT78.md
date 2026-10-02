# THEORY §78 throughput pilot (2026-10-02 12:29 EDT; local RTX 4060, 8 GB; `bash scratch/p78_pilot.sh cuda:0`)

The pilot fixes epochs (≤ 6) and batch only. Each arm ran 400 optimiser steps from base GPT-2, seed 1, short-valid
eval at the end. Logs: `logs/p78_pilot_*.log`; runs `gsmlen/pilot/`. Two earlier attempts at bs 32 ran out of
memory (8 GB: fp32 logits of 32 × L × 50,257); their logs are kept in `logs/void/`.

**Batch.** bs 16 × grad_acc 2 (effective 32), bf16 autocast.

| arm | s / step | steps / epoch | h / epoch (4060) | loss at steps 100 / 400 | short-valid acc at 400 |
|---|---|---|---|---|---|
| TC bf16 | 0.260 | 6,449 | 0.47 | 1.218 / 0.623 | 0.072 |
| TC fp32 (parity) | 0.389 | 6,449 | 0.70 | 1.203 / 0.615 | 0.072 |
| WS bf16 | 0.238 | 17,386 (556,369 calls) | 1.15 | 0.723 / 0.544 | 0.033 |
| WS-pad bf16 | 0.258 | 17,386 | 1.25 | 0.712 / 0.659 | 0.033 |

* **bf16 parity confirmed.** The same valid accuracy, and losses within 0.01. bf16 is used (1.5× faster).
* **Epochs: 6** (the registered maximum), with the checkpoint chosen by short-valid accuracy after each epoch.
* **Range gap on record:** the fraction of WS / WS-pad training states with true length ≥ 3 is 0.0.
* **Cost per seed at 6 epochs:** about 17.2 GPU-h on a 4060, about 8.7 GPU-h on a 3090 (about 2× the 4060, from §68's
  3090 bf16 measurement).
