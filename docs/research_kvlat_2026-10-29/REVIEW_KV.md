# REVIEW_KV: red team of KV_draft.md (8c8ec19f), 2026-10-29

Opus subagent, read-only (scratchpad kvlat_flops.py, kvlat_norms.py).
Verdict: **GO-with-fixes** (fixes 1, 2, 4 and 5 before registration). Theory's decision: **not registered, deferred** (Note 246-1). The value / effort / prior balance is too weak for the remaining ~$7.

1. **BLOCKER:** with P-2's think_k 8, KV-LAT costs 92+ cached token-steps, 1.8–2.1× TEXT at d 2. Parity holds only at K′ ≤ d.
   - P-2's K′ robustness comes from DeltaMem's fixed-size state; KV-LAT's decoder trained on 8 latents shifts at K′ 1–2.
   - Fix: train K = d or K ~ U{d..8}, and disclose the recipe change.
2. **FIX (scale):** the block-11 state at ':' has norm 225 (ln_f 90) vs wte ~4 / wpe ~3.5, so an identity W_lat injects 23–57× too large.
   - Fix: take ln_f(h) with a learned gain ~0.044 (Coconut / CODI); pin the position ids.
3. **FIX (FLOPs):** the formula omits L_t's own forward. Fuse it as [L_t, cue_{t+1}].
   - d 2: K′ 1 10.5 G, K′ 2 12.6 G vs TEXT 11.7 G (plain prompt).
   - Serial depth 24–36 vs TEXT 204. The "trades latency" sentence is stale.
4. **FIX (not one change):** the aux path (P-2 decodes the step sentence from memory; the draft uses a CE at ':'), the decoder (P-2 re-reads S at every prompt position), dropped fade / γ / g_add calibration, new masks and positions, W_lat's optimiser group, and CUE-WRONG dropped.
5. **FIX (gates):**
   - L_1 interventions must recompute KV and L_2..L_K′ from the recipient's cues; a swap with them kept is near-vacuous at d 2.
   - CUE-ONLY needs two caches.
   - C3-F at a 0.70-accuracy config vs TEXT at 1.000 is unfair: use ≥ 0.98, name TEXT's prompt, and compute FLOPs analytically with one FlopCounterMode check.
6. **FIX (build):** a new module boxes/inject_kv.py (inject_aw's sha is asserted by §245; lazy-import skew); reuse rs_run (nested170.py:427-452). Gates: fp64 cache equivalence, padding invariance, gradflow into L_1 and W_lat, the FLOP check, G-TIME (launch-bound risk), an args diff.
7. **NOTE (priors):** the project's own cheap-pass failures, §171 / §198 READ-SLOTS (memory unused, read mass on EOT) and §42 (Coconut models stopped using latents). ~0.45 per seed is realistic, not 0.6.
8. **NOTE (novelty):** KV-LAT = Coconut + a per-step cue. It drops P-2's narrow claim (a compact recurrent memory, not a KV slice); STATE-USE falls under SCIT 2608.27265. The only new piece, the FLOP-matched measurement, is arithmetic once K′ is known.
