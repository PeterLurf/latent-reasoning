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

## DeltaProduct comparator: amendment 71-2 log (2026-10-02 04:02 EDT)
* **First S5 runs (torch 2.5.1, triton 3.1.0, transformers 4.46.2, the bundled fla 0.2.2).**
  - 3 concurrent runs OOM'd a 24 GB card; the runs were then made sequential.
  - dp_s1: best sequence accuracy 0.0. dp_s2: loss ≈ ln 120 (chance). dpvar_s1: 0.0.
  - All voided (`*.void_oom`, `*.void_env0`); the dpvar host was stopped.
* **Environment check: the README's own S3 Configuration 2 (expected to train), 100 epochs.** Sequence accuracy at 512 is
  0.18, token accuracy 0.961. The environment does not reproduce the reference. The README asks for triton > 3.1.0.
* **Diagnosed configuration 1 (of 2): the repo's stated environment.**
  - torch 2.6.0 + triton 3.2.0 (with matching torchvision 0.21 / torchaudio 2.6).
  - The S3 check gives best sequence accuracy 0.73 (final 0.36), token accuracy 0.989. Better, but not "trains".
  - Decision: run the S5 dp seeds 1–3 in this environment, sequentially, with the paper command unchanged.
  - Attempt 2 is held in reserve.
  - Interface-only patch: their error handler's wandb calls are guarded.
* **71-2 closed (2026-10-02 05:23 EDT): DeltaProduct NOT reproduced on our setup.** In configuration 1 (torch 2.6.0 /
  triton 3.2.0), S5 dp_s1 reached best sequence accuracy 0.0 after its full run (paper command unchanged), as in the
  original environment. dp_s2 was stopped.
  - The repo's own S3 check reached 0.73 best (0.18 before the upgrade): partial.
  - Attempt 2 was not spent on a guess.
  - Logs in `logs/dp_s71/`; host destroyed.
  - Per 71-2, §71's S3 comparison is reported as "comparator not reproduced on our setup", citing only the published
    figure (DeltaProduct n_h = 4, 1 layer: robust to 512, Fig. 5).
