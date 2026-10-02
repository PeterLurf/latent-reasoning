# THEORY §67b pilot log (seed 99; may fix updates, batch and lr only)

Started 2026-10-01 22:12 EDT. Fixed by §67b + 67b-2:
* M 24; τ annealed linearly 4 → 1 over the first half of updates; second-order readout W·ReLU((U·LN(h_code)) ⊙
  (V·LN(h_op))); LN before the answer head; distinct code init. Product-automaton certificate; counter check on
  certified runs.
* **Pilot 1 (D1b):** 8,000 updates, bs 512, lr 1e-4 (GPT-2 and new parameters), on the RTX 3070.
* **Pilot 1 result:** certified (product automaton, 0 bad pairs), 1.000 at N 8–128. One configuration tried.

## Panel settings, frozen (2026-10-01 22:32 EDT; before any panel run)
* **Seeds** 1–5.
* **D1b:** `--arm d1 --M 24 --tau0 4 --readout bil --head_norm ln --code_init distinct --updates 8000 --bs 512
  --lr 1e-4`.
* **RESET:** the same with `--arm reset`.
* **FP** (the init rule applied identically to the GPT-2 arms: LN head input): `--arm fp_ex` (example-matched) and
  `--arm fp_cmp` (compute-matched: rows per update so GPT-2 token positions per update equal d1's), `--head_norm ln
  --updates 8000 --bs 512 --lr 1e-4`.
* **TAB:** `--arm tab --M 24 --tau0 4 --updates 8000 --bs 512 --lr 1e-2` (the toy recipe's lr; a free table).
* **Eval:** n_test 2,000 per depth, nulls 10,000; certificate, audit, nulls and mechanism on every certified D1b.
* **Placement:** TAB on the local 3070. D1b / RESET / FP_ex / FP_cmp of seed s run together on Vast host s (RTX 3090
  class, 4 concurrent).

## Implementation amendment 67b-impl-1 (2026-10-01 23:38 EDT; Audit / external agent; before any panel result was read)
* **Bug.** `c1/train.py` set the d1 compute budget to a constant 120 GPT-2 token positions per update (36 × 3 + 6 × 2).
  That is right only for M = 6; the true count is 6M × 3 + M × 2 = 20M, i.e. 480 at M = 24 (240 at M = 12).
* **Consequences.**
  - §67b's fp_cmp (compute-matched FP) got ~4× too few rows per update, so it was under-trained in D1b's favour.
  - Logged `gpt2_tokens` for d1 arms at M ≠ 6 (67-2, 67b) are 2–4× low. fp_ex is unaffected.
* **Fix.** Budget = 20 · M from args (`d1_tokens(M)`).
* **Re-run.** fp_cmp for §67b seeds 1–5 with the corrected budget (`--M 24`), runs `c1/runs/p67b_fpcmp2_s<k>`, on local
  GPUs.
* **Scoring.** C1-5's compute-matched comparison uses the corrected runs. The original fp_cmp runs are kept and reported
  as descriptive.
* Measured training GPU-seconds per arm are reported next to token positions (positions are a proxy). Vast runs shared
  a GPU 4 ways, local runs share with bench evals, so wall-seconds are indicative only.
