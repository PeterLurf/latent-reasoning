# THEORY §68 throughput measurements (2026-10-01 21:25 EDT; before any §68 run)

**Official Coconut code** (facebookresearch/coconut @ 27273cb, unmodified on Linux; data GSM8K-Aug from
Internalize_CoT_Step_by_Step @ e06a32ee, `gsm_train.json` sha256 93a266b1…, 385,620 items).
* **Setup.** Vast RTX 3090 (580.159.03), 1 GPU, batch 32, fp32. Debug epoch of 5,000 items (157 batches). Logs in
  `logs/gsmtime/`.
* **CoT stage:** 0.516 s per batch → **1.73 h per epoch** (+ ~0.05 h validation).
* **Coconut stage 3 (6 latents):** 3.81 s per batch → **12.7 h per epoch**. Stages 1 / 2 (2 / 4 latents), interpolated
  per pass: ~5.4 / 9.1 h per epoch.

**Local proxy** (our GPT-2, `scratch/gsm_time_proxy.py`, bs 8): RTX 4060 CoT 1.79 h per epoch; a 6-pass latent proxy
2.8 h. Batch 32 fp32 does not fit in 8 GB.

**Projection for the registered §68** (fp32, 3090-class, 1 GPU per run):

| arm | per seed | 3 seeds |
|---|---|---|
| TC, 25 epochs | 44 h | 133 GPU-h |
| CO to epoch 25 (official num_epochs) | 247 h | 742 GPU-h |
| CO to epoch 15 (the official eval checkpoint) | 133 h | 398 GPU-h |
| WS (token-matched to TC) | ~44 h | ~133 GPU-h |

* **Total:** ≈ 660–1,000 GPU-h, i.e. **≈ $110–170 at $0.17/h**, against the registration's 40–80 GPU-h / $10–25 and
  about $110 of credit.
* **Wall time.** Even with one host per run, CO is sequential after TC: ≥ 44 + 133 h ≈ 7+ days.

**Not launched.** Re-scoping is for Theory, Audit and the user. Levers the pilot may use (batch, precision):
* bf16 (to be measured; typically 1.5–2× on Ampere);
* fewer seeds;
* CO to the epoch-15 checkpoint;
* faster GPUs (cost per FLOP is roughly flat).
