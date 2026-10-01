# §60 Phase A development log

* 2026-10-01 00:38 — development pilot (BM-oc λ 1 on cuda:1, λ 0.1 on cuda:2; seed 0; dev set only) started on the
  pre-60-3 code (commit c8621dc). Stopped at ~00:57 for implementation note 60-3 (root slots permuted; labels over all
  a ≠ 0 entries). Its outputs (bm/runs/pilot_oc_lam*/ and the pool from c8621dc) were deleted unread; the logs
  logs/bm_pilot_lam*.log were overwritten by the restart. Disclosed here; no result was read from it.
* 2026-10-01 00:58 — data pool regenerated on the fixed code (seed 0) and the development pilot restarted (1ffaeff).
* 2026-10-01 02:25 — development pilot result (dev set only): BM-oc λ = 1 and λ = 0.1 give **identical** dev metrics
  (probe 0.874, entry 0.839, laws 1.000, nP2 0.918, nP34 0.852). Train accuracy plateaus at 0.88 from step 2000 to
  10000. That is exactly the identity map f(a) = a: trivial entries (37 % of labels) right, non-trivial ones predicted 0
  (80 % of them negative), 0.37 + 0.63·0.80 = 0.874. The identity map satisfies all four laws trivially.
  **Plumbing test** (`bm/test_runner.py`): exact operators that see only the model's inputs, run through the same
  batched runner with slot permutations, local ids and the canonical root remap, score probe 1.000 and entry 1.000. So it
  is not a wiring bug; the learned model collapses to the identity. Diagnostic run (2–3-edge regions only, λ 0, 2000
  steps, not a setting) launched to separate "cannot learn one merge" from deep credit assignment.
