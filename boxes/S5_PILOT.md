# THEORY §71 pilot and frozen panel settings

**Pilot (seed 99, local RTX 2070, fp32).** WC0₁ all-position, 8,000 steps, bs 64, lr 1e-4 (cosine, 100 warm-up):
* loss 7.9e-7 by step 2,000;
* certified (k = 1: 0 errors / 14,400 inputs, batch 100, fp32);
* cumulative accuracy 1.000 at t = 128 / 256 / 512 on 200 test sequences.

One configuration tried.

## Panel settings, frozen (2026-10-02 01:14 EDT; before any panel run; amendment 71-1 applies)
* **WC0ₖ, k ∈ {1, 2, 4}, seeds 1–5:** `--arm wc0all --k <k> --steps 8000 --bs 64 --lr 1e-4 --eval_bs 500`. The
  certificate for k = 1, 2; the chunk check for k = 4.
* **Tagger, seeds 1–3:** `--arm tagger --steps 8000 --bs 32 --lr 1e-4` (all-position labels on length-128 sequences).
  Not piloted: the steps match WC's; batch 32 sequences × 128 positions.
* **P3, seeds 1–3:** `--arm p3 --k 4 --latent 0` (WC0₄) and `--latent 4` (WC₄), `--steps 8000 --bs 64 --lr 1e-4`. k′ ~
  U{1..4} per amendment 71-1; final-only.
* **Precision:** bf16 on Ampere+ (Vast 3090s), fp32 on the local 2070. The certificate and chunk check run at the same
  dtype and eval batch as the test rollout (recorded in eval.json).
* **DeltaProduct / GRU:** as in the job script (the paper's command; dpvar per 71-1).
