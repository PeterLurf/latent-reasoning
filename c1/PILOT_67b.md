# THEORY §67b pilot log (seed 99; may fix updates, batch and lr only)

Started 2026-10-01 22:12 EDT. Fixed by §67b + 67b-2:
* M 24; τ annealed linearly 4 → 1 over the first half of updates; second-order readout W·ReLU((U·LN(h_code)) ⊙
  (V·LN(h_op))); LN before the answer head; distinct code init. Product-automaton certificate; counter check on
  certified runs.
* **Pilot 1 (D1b):** 8,000 updates, bs 512, lr 1e-4 (GPT-2 and new parameters), on the RTX 3070.
