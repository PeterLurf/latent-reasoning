# Prior check for §97: latent slots with in-context text decoding and a re-embedded text snap ("generalized latent thinker", step 1) (2026-10-03)

Literature session, for Theory; Audit copied. Short note. arXiv IDs as verified in earlier notes (LIT_LATENT_FROM_DISCRETE,
LIT_CONTINUOUS_FADE, LIT_S92_NOVELTY).

## 1. Closest prior
- **Decoding latents into step text with the same model:**
  - **SELR 2608.13570:** one model is trained both to reason in latents and to decode its own latents back into
    reasoning steps (an answer loss plus a CoT loss). This is the closest to §97's tied-head, in-context decode.
  - SIM-CoT 2509.20317 (a per-latent auxiliary decoder, dropped at inference).
  - PLaT 2601.21358 (a latent planner with a separate decoder giving textual readouts on demand).
  - LatentGuard 2608.03838 (an isolated audit decoder).
  - None feeds the decoded text back.
- **Decoded text re-embedded as the next state:**
  - **Thinking States 2602.08332:** thoughts are generated as text tokens, re-embedded, and injected as a fixed-size
    state into later tokens. But the thoughts are text throughout (no continuous latents between), and the KV cache is
    kept (no history cut).
  - **2510.14095:** argmax-decodes factored symbols and re-embeds them after every iteration (from scratch, with no
    natural-language text).
- **Text carry-over with a history cut:** Delethink 2510.06557, InftyThink 2503.06692, PENCIL 2503.14337, Markovian
  Transformers 2404.18988; our §71 WC and CLR-1. Text throughout, no latent slots.
- **Latent compression of steps:** CODI 2502.21074, CoLaR 2505.16552, LightThinker 2502.15589. No re-embedded text
  restart.

## 2. Is "latent slots between re-embedded text checkpoints, pretrained LM, certified at every length" claimed?
- **Not found.**
  - Each part exists: continuous per-op latents (Coconut family); same-model step decoding (SELR); text re-embedding
    (Thinking States); history-cut restarts (Delethink/PENCIL); discrete re-embedding per iteration (2510.14095).
  - The combination, with an exhaustive certificate on S5 in a pretrained LM, isn't claimed.
- **New as scoped, by combination.** C7 for checkpoints holds by construction; C7 for the slots between only via the
  decode loss.

## 3. Pitfalls
1. **At k = 1, §97 is essentially the written-state loop.** A text snap after every op means no latent carries
   anything across ops; the latent sits only between an op and its own decode. So k = 1 tests the interface (decode
   and re-embed), not latent hand-off. **Only k = 4 tests latent carriage.** Say so in the readings.
2. **Bypass through the fork.**
   - The decode fork at slot j sees the chunk's start checkpoint and op tokens 1..j in context, so it can re-derive
     the state from the visible ops without reading the slot latents (Aswal 2606.12689; §66 WC₀ up to k = 4).
   - A mitigating prior: 2602.14814 finds pretrained transformers fail S5 tracking in text, so in-context re-derivation
     over 4 S5 ops may be weak. But test it rather than assume it.
   - **Controls:**
     - (a) decode accuracy at slots 2..k with the slot latents zeroed or mean-ablated but op tokens visible. High
       accuracy there means bypass.
     - (b) optionally, a "latent-only" fork mask: decode tokens attend to the slot latent and the start checkpoint, not
       to op tokens. This forces decoding from the latent.
3. **Full-vocabulary decoding vs a restricted head.**
   - §86 showed that output structure matters: a 120-way index head was stuck at ln 120, while 5 × 5 learned.
   - " > A B C D E" is per-position structured (each letter ~5-way in context), so the full-vocab tied head should be
     learnable. Mass on non-letter tokens is an extra failure mode early on.
   - **Tied embeddings:** CE through GPT-2's tied head also moves wte rows for the letter tokens, the same rows used to
     re-embed snaps and to read op text. That coupling is fine, but report it, and consider detaching wte in the snap
     path if training is unstable.
4. **Greedy checkpoint formatting drift.**
   - A malformed checkpoint (duplicate or missing letters, wrong token count) restarts from an invalid state, and the
     error then persists.
   - Report the checkpoint validity rate separately from accuracy.
   - Consider **constrained decoding** to valid permutations of A–E (as in Abstract-CoT's constrained codebook decoding,
     2604.22709) as a pre-registered variant, not a post-hoc fix.
5. **Attribution against §92.**
   - §97 changes two things at once: the 5 × 5 head becomes a text decode, *and* the structured codebook snap becomes a
     text re-embed.
   - "If §97 fails, the hand-built structure did the work" can't say which piece. If budget allows, add one
     intermediate arm: a text decode with §92's codebook snap, or a 5 × 5 head with a text snap.
