`# Boundary memory (user's proposal; THEORY §60 / §60b) — Phase A results
`
`Updated 2026-10-01 04:39 EDT. Details: `bm/runs/score_panel_b.json`, `bm/runs/DEV_LOG.md`, THEORY §60b (de25858).
`
`* **§60b panel (registered revision, bdc70bc): Gate A NOT PASSED in 4/4 seeds.**
`  - Identity / chance line ≈ 0.51 per split.
`  - BM-oc probe (instances / motifs / orders): s1 0.63/0.56/0.61, s2 0.63/0.55/0.61, **s3 0.88/0.86/0.84, s4 0.79/0.73/0.71**.
`  - BM-no stays at 0.55–0.63 in every seed.
`  - Laws 0.19–0.41. PA-dir 1/4: oc is better on held-out orders in s3/s4, but with higher associativity error.
`* **Decision: the boundary-memory line STOPS at Phase A** (§60b's fixed rule, after the one revision): no Phase B, no further revision.
`* **Descriptive pointer for future work.** With order consistency, the learned merger acquired non-trivial composition
`  in 2/4 seeds, and BM-no never did. Held-out orders behave like held-out instances (s3 0.84 vs 0.88), so this is an operator,
`  not a sweep. Its confidence |T − 0.5| fades with merge depth (s3 0.48 → 0.38; s4 0.48 → 0.35), a confidence clock (THEORY).
`* **§60 panel (registered original, note 60-3 code): Gate A NOT PASSED in 4/4 seeds.** All 8 runs (BM-oc, BM-no) equal each
`  split's identity-map baseline exactly: instances 0.877, motifs 0.911, orders 0.891, size 0.895. That is the collapse to
`  f(a) = a predicted by the dev pilot. Laws 0.99–1.00 only because the identity map satisfies them; PA-dir 1/4
`  (\`bm/runs/score_panel_a.json\`). Consistent with the stop.
