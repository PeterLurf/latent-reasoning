# Prior check for §80: latent chunk machine (discrete code carried across k-op chunks, exact marginalisation) (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch).

## Prior
- **Segment-level carry-over in transformers (continuous):** Transformer-XL 1901.02860 (cached segment states),
  Block-Recurrent Transformers 2203.07852 (a recurrent state block updated per segment), RMT 2207.06881 (memory tokens
  passed between segments). All carry a *continuous* state and train by backprop through segments, with no discrete
  bottleneck and no marginalisation.
- **Discrete bottlenecks:**
  - Discrete Key-Value Bottleneck 2207.11240 (quantised memory between encoder and decoder).
  - VQ latent tokens mixed into CoT: Token Assorted 2502.03275.
  - Thinking States 2602.08332 (text thoughts re-embedded per chunk; supervised, no history cut, no marginalisation).
- **Exact or near-exact marginalisation over a discrete latent feeding a transformer call:**
  - REALM 2002.08909 and RAG 2005.11401 marginalise over retrieved documents (a discrete latent) from final answers.
    This is the closest training analogue, with one latent per call rather than a chain.
  - Segment models with exact marginalisation over segmentations: SWAN 1702.07463.
  - Latent-rationale EM: TRICE 2312.02179 (MCMC-EM, not exact).
- **Composition of unseen op tuples within one pass:**
  - Liu et al. 2210.10749 (transformers compute k-step automaton compositions via shortcuts).
  - Li, Guo & Andreas 2503.02854 (pretrained LMs learn an associative scan for permutation words).
  - Hupkes 1908.08351 (systematicity and productivity taxonomy).
  - None tests held-out (state, op-tuple) combinations under a certificate.
- **Not found:** a discrete code carried between transformer calls, trained by *exact* marginalisation over codes at
  chunk boundaries from final answers. New as scoped. The pieces exist separately: chunked carry-over (RMT/BRT),
  marginalisation over a discrete latent (REALM/RAG), and in-pass composition (Liu; Li).

## Design flags
1. **Two levels of "held-out"; report them separately.**
   - (a) Held-out (code, k-op word) *pairs*, where the word appears with other codes. GPT-2 must apply a known word's
     permutation to a code it never saw it with. That is the §76(B) "fill a held-out row" test, which failed (chance).
   - (b) Held-out k-op *words* never seen at all. GPT-2 must compose new op tuples in its forward pass. This is the
     real composition test, and the one the one-hot-chunk table can't touch.
   - (a) can pass while (b) fails, and vice versa. Pre-register both.
2. **The premise check is right, and it must run on the same held-out sets.** §71's in-text k-op accuracy is over a
   sampled distribution. Measure it on exactly the held-out combinations of (a) and (b), in text (written-state form).
   If GPT-2 can't compose them in text, the latent version has no reason to.
3. **The marginalisation is over codes at chunk boundaries only.** Within a chunk the transition is a deterministic
   readout of one forward pass, so composition must happen inside GPT-2 (C1 proper). Good for attribution. But with
   k-op words the effective alphabet is |Σ|^k, and the code×word table grows multiplicatively. Report coverage:
   training occurrences per (code, word).
4. **Controls:**
   - The one-hot-chunk table gives 0 on (a) and (b) by construction.
   - Random-init GPT-2 is the pretraining control.
   - Add a **k = 1 GPT-2 cell iterated k times** (the per-op cell chained). It separates "GPT-2 composes inside one
     pass" from "the chain composes".
