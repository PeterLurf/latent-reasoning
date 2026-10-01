# §60b — settings frozen for the seed panel (2026-10-01 03:25 EDT, before any §60b panel run)

* Code: `bm/data_b.py`, `bm/model_b.py`, `bm/run_b.py` at adb6575 (+ Vast tooling). Pool `bm/data_pool_b` (seed 0).
* Development pilot (dev only, seed 0, 10k × 128, on Vast; `bm/runs/pb_pilot_lam*/`):
  λ 1 → dev probe 0.5945 (entry 0.853, laws 0.374); λ 0.1 → 0.588 (0.851, 0.375). Identity / chance baseline on non-trivial
  labels: 0.50. **λ_order = 1** (higher dev accuracy).
* **Steps 10,000, batch 128**, lr 3e-4 cosine (200 warm-up), identical for both arms; the pilot's train accuracy is flat
  (0.57–0.61) from step 1,000 to 10,000, so no longer schedule was chosen.
* Seeds 1–4, BM-oc (λ 1) and BM-no (λ 0); failed seeds kept, no extra training.
