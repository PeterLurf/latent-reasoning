# REVIEW_C3L: red team of C3L_draft.md (148d3682), 2026-10-29

Opus subagent, read-only. Analytic FLOPs on TEST2-DF (scratchpad c3l_flops.py).
Verdict: **GO-with-fixes** (1 and 5 before registration). Theory's decision: **the run is withdrawn** and the C3 reading is recorded analytically (Note 246-0), because finding 1 fixes the outcome in advance.

1. **BLOCKER (the outcome is set by arithmetic).**
   - One LAT pass reruns blocks 6–11 over every prompt token, with a read at each position (inject_aw.py:47-51), and writes only at ':' (:53-54). Blocks 0–5 are cached per distinct cue (d inputs; inject_st.py:59). The decoder is a full 12-block forward (inject_ow.py:70, 85-89).
   - TEXT = one prefill plus a KV-cached step per token (codi_ps.py:346-351).
   - Executed GFLOPs at the cheapest K′ reaching 1.000 (K′ = d − 1), LAT vs TEXT full chain: d 2 18.6 vs 12.0; d 3 35.6 vs 15.5; d 4 57.5 vs 19.1. Counting the cue-shared prefix once: d 4 36 vs 19, d 2 ≈ 14 vs 12.
   - Serial layers: 24 / 30 / 36 vs 204 / 264 / 324.
   - So S goes to LAT and F to TEXT-AHEAD (P ≈ 0.95). The only empirical unknown is TEXT's ceiling.
2. **FIX:** use analytic per-item formulas, not instrumentation (padded batches and an all-K′ decode make instrumented counts wrong). Check each path once with FlopCounterMode.
   - F_LAT = min(K′, d)·6·blk(L) + K′·(6·blk(L) + 8D²L) + 12·blk(L−7) + 8D²(L−7) + head.
   - F_TEXT = 12·blk(P) + Σ_i [12(24D² + 4D(P+i)) + head].
   - blk(T) = 24D²T + 2DT(T+1).
   - Use serial layers for (S).
3. **FIX:** TEXT's length is exactly 5d + 6 tokens, so truncation gives a 0/1 step. Add answer forcing, define acc(B) as the largest config with cost ≤ B (no oracle), and use K′ 0..14 + 16. s1 dips at K′ = d.
4. **FIX:** give TEXT the cue-all prompt (LAT gets the plan from the per-pass cue); raise the ceiling gate to ≥ 0.98 per d; keep d 5–8 descriptive; disclose s1's LP0-FAIL.
5. **FIX:** the build is not small. It needs a key-map loader for the askname init (vocab 50257, `p.h_0_…`), askname's own format, mul_s_p in the stream, TEST2-DF in the eval, LAT's EXCL, and --d_stages 4.
6. **FIX:** define MIXED and a seed-aggregation rule; add the OC P(pass | no effect) / P(pass | effect); pre-state the reading of the likely "S-LAT / F-TEXT" outcome.
7. **NOTE:** the cost is plausible; off-host checkpoints are needed.
