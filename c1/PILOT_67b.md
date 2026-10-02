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
