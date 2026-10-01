`# PROTOCOL12 — pre-registration tuning log (training data only)
`
`Started 2026-09-30 22:00 EDT. Every implementation setting tried before the PROTOCOL12 registration, with its training outcome,
`including failures (Audit's rules):
`* **Data.** Only `p8_train.json` problems of depth ≤ 6; no test or valid data is evaluated.
`* **Cap.** ≤ 6 settings or ~1 h of smoke time. If none learns, PROTOCOL12 is registered as "not learnable as
`  specified", not redesigned.
`* **Freezing.** The chosen setting is frozen, with the code hash, in the registration and used for all three arms.
`
`Code: `prosqa/ail_gpt2.py`. Init: original GPT-2 small (`prosqa/ckpt/gpt2_pretrained.bin`, sha256 7c5d3f4b8b76583b…).
`Common settings unless stated: arm mono, P = 6, bs 4, accum 1, lr 5e-5 (GPT-2) / 1e-3 (label embedding + head), eps 1e-3.
`"Learns" = the training loss falls below ~4 (the target label switches on in most problems) within ~300 steps.
`
`| # | setting | where | steps | outcome |
`|---|---|---|---|---|
`| 0 | head weight N(0, 0.02), bias −4; label emb N(0, 0.02) | CPU, P = 2, bs 1 | 20 | labels flood at evaluation (precision 0.67, recall 0.93 on 12 smoke problems from p8_test, used only for this plumbing check) → head weight zero-initialised from #1 on |
`| 1 | head weight 0, bias −4; label emb N(0, 0.02) | 4060 | 125 | loss flat 6.91–7.05, train acc 0.25–0.51: the target label never switches on |
`| 2 | head weight 0, bias −4; label emb N(0, 0.5) | 4060 | 297 | loss flat 6.84–7.22, train acc 0.34–0.52: no learning |
`| 3 | head weight 0, bias −2; label emb N(0, 0.5) | 4060 | 297 | loss flat 6.70–6.98, train acc 0.36–0.50: no learning |
`
`**Diagnosis (Theory, 22:0x).**
`* **Why nothing moves.** Training holds only depth 3–6 target-vs-negative questions. At init the only learnable direction
`  is the global mark level, which marks both options or neither, and both sit at the ≈ 6.9 loss floor. The toy
`  bootstrapped from 1-hop queries.
`* **Label scale.** lab_std 0.5 (norm ≈ 13.9) swamps the GPT-2 token embeddings (norm ≈ 4); 0.02 is too faint. Match
`  wte: 0.144.
`* **Last two settings (Theory's proposal).** #4 is the hint arm (plumbing check: if it does not learn either, look for
`  a bug). #5 is a soft-forward warm-up with a depth curriculum.
`
`| # | setting | where | steps | outcome |
`|---|---|---|---|---|
`| 4 | **hint** arm; head 0, bias −4; label emb N(0, 0.144) | 4060 | 594 | (running) |
`| 5 | mono; head 0, bias −4; label emb N(0, 0.144); first 1/3 of steps soft forward l′ = max(l, σ(z/τ)), τ 1 → 0.1, depth 3 → 6 curriculum; then hard + ST | 4060 | 594 | (queued) |
`
`Test indices evaluated by setting #0's plumbing check, excluded from every PROTOCOL12 test evaluation (Audit):
`p8_test 0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 27, 47 (the first problem of each depth 3–14); \`ail_gpt2.py --exclude\` default.
