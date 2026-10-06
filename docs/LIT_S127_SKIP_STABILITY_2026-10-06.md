# Lead for the §127 SKIP rerun: stabilising a skip path injected into a pretrained LM (note 127-3) (2026-10-06)

Literature session, unrequested lead for Theory; Audit copied. IDs verified via the arXiv API (2026-10-05).

**Context (note 127-3).**
- SKIP s2 escapes at 2k: the code moves off the item and onto the state.
- The loss then drifts back towards the plateau (4.9 at 3k).
- Training goes NaN when L_max switches to 16.

**The pattern.** The skip adds W_skip·q into blocks 6–11 at every decoder position. As L_max grows, more positions
receive an unnormalised additive term. Large skip weights push the residual stream off GPT-2's distribution (cf. §72b
massive activations; HRPO's raw-injection failure).

**Standard remedies for a new residual branch into a deep network:**
1. **A learnable per-channel scale, initialised small but non-zero:** LayerScale (Touvron et al. 2103.17239,
   diag(ε) with ε ≈ 1e-4 … 1e-1); ReZero (Bachlechner et al. 2003.04887, a scalar α).
   - This keeps the early-gradient benefit (unlike zero-init, which gives the code no gradient at step 0; e925cf3e)
     while bounding the injected magnitude.
   - A rerun-safe form: x ← x + diag(s)·W_skip·q, with s initialised at 0.1 and a separate LR.
2. **Normalise the injected term** to the residual stream's scale (LayerNorm on W_skip·q, or the §73 / §125
   standardisation) so it can't grow unboundedly.
3. **Bound the training dynamics:** gradient-norm clipping on the new modules; fp32 for the skip projection and the
   attention softmax (bf16 overflow is a common NaN source at longer contexts).
4. The non-finite guard in 127-3 detects NaN but doesn't cure it. Pair it with (1) or (2), otherwise the rerun will
   likely trip the > 1% rule again at L16.

**On the post-2k regression.** Once the decoder can read the code through the skip, the in-stream route
(GPT-2 reading the carried code as its input embedding) can stay unused. This is a skip-path bypass of the carrier's
own use. Dieng et al. 1807.04863 keep skips permanently, but our C1 test is on the carried code.
- Report the donor / ablation effect of the skip path vs the in-stream carrier at the end.
- Consider **decaying the skip scale to 0 after entry** (it was an entry device, like VAR's 6k decay), then checking
  whether the carrier alone holds.
