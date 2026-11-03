# REVIEW_NK: red team of NK_draft.md (Opus subagent, 2026-11-03; no model.pt loaded). Verdict: GO-with-fixes, 4 blockers, all adopted in NK_v1.md

**$0 read (dch_s* eval_test_perm items vs regenerated REG TEST2):**
- pass 1 is purely positional (WS 1.000 with S_1 in slot 1, 0.31-0.38 when moved; at d 2 S_1 is always moved → 0.10);
- passes >= 2 are order-bound even after a correct previous pass (S_t moved: 0.20-0.37; unmoved 0.78-0.91).

So selection fails at every pass; NK as drafted fixes only transport.

1. **BLOCKER:** `assert not (a.distract_p and a.perm_p)` (inject_aw.py:278), and bun carries distract_p 1.0. Build perm-then-insert and keep both disjointness asserts.
2. **BLOCKER:** the gate (permuted d 2 pass-2 WS) mostly measures pass-1 lookup, which NK cannot touch, and passes without the binding (§252 NP reached 0.40 with no slot).
   - Fix: a conditional pass-t WS (given pass t − 1 correct) on d 3-4 permuted items; an NK-ablation causal gate; the name-swap test binding; pass 1 reported separately.
   - Either scope to transport, or add askperm-style one-hop co-training as a shared change.
3. **BLOCKER:** a separate S_n escapes SWAP4, LP5, the hold penalty / norm guard and halt_inc. Pack it into S (pack_state precedent) and audit the slices.
4. **BLOCKER:** the read path has no token ids (read_fixed, fwd_read for decoder and aux, ws_readout, passes, read_mask). Thread the ids and assert them. The aux must read NK.
5. Tie q and k (otherwise own-match ~ crosstalk); one head suffices with 40 names; an unwritten name reads crosstalk.
6. Use the global 40-name token set (inserted side names are outside it["names"]); assert write key = spine[min(t, d)].
7. W_on = 0 without a fade buffer (a zero fade kills gradient); value = block-11 ':' h via ln_w; the delta erase is needed for hold idempotence.
8. Hold passes get a copy-based fixed point (helps C6). The decoder can copy S_d's slot, so NOT-DECODER is weaker evidence. "Plan-light" is unsupported (the key needs the plan).
9. Retention:
   - C2 vs dch − 0.05 plus d 7 and WS d 6 passes 5-6;
   - the C6 ||S|| ratio incl. NK;
   - pilot --steps 8000 --stop_at 2000;
   - CTRL vs bun.
10. Cost: ~$0.3-0.45 per pilot cell, full ~$6-8, within $15.
11. Novelty narrower:
    - EntNet;
    - Entities-as-Experts / Mention Memory (TOME): mention-position injection;
    - EntityNLM.
    - New: per-step dynamic writes in a latent loop.
    - Alternative worth running: perm FT + one-hop co-training (it is the CTRL arm in v1).
