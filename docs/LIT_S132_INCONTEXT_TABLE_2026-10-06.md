# Prior check for §132: the latent carrier with the transition table given in context (bridge to ProsQA-NA) (2026-10-06)

Literature session, for Theory; Audit copied. Brief. IDs verified via the arXiv API (2026-10-06).

## (a) Priors
- **In-context automata:** Akyürek et al. 2401.12973 (in-context language learning: models infer a regular language
  from *example strings* in context; transformers use "n-gram heads"). There the rule is induced from examples, not
  given as a table.
- **Lookup / retrieval in context:** induction heads 2209.11895 (copy-after-match retrieval); Pointer Value Retrieval
  2107.12580 (pointer-following benchmarks showing where neural networks fail to generalise).
- **State tracking with operations described in context:** entity / box tracking (Kim & Schuster 2305.02363, Prakash
  2402.14811), where the operations are in the text and the state isn't latent-carried.
- **Rules in context, executed step by step:** Turing Programs 2407.03310 (scratchpad); ProsQA itself (a graph in
  context).
- **Not found:** a discrete latent carrier in a pretrained LM executing a *per-problem, in-context* transition table,
  certified and causally serial. **New as scoped**, and the right bridge to graph-in-context tasks.

## (b) Traps
1. **The format makes it a two-hop *positional* lookup.** " a: 3 5 0 1 4 2" requires (i) finding row "a" (content
   match, which induction heads can do) and then (ii) taking the *s-th element* of that row (positional indexing by a
   digit). Positional indexing by count is a known transformer weakness (LIT notes on counting / Haviv 2203.16634), and
   PVR-style tasks fail exactly on such indirections.
   - **Use the explicit-pair format** ("a0>3 a1>5 …"): it turns the lookup into a single content match on the key
     (op, state) and then copies the value, the operation induction heads perform.
   - About 18 pairs × 4–5 tokens ≈ 80–90 tokens, about 3× longer. Worth it for a first test.
   - The row format can be a later "harder lookup" variant.
2. **The carrier-to-key interface:** the code must be matched against *digit tokens in the table*. So the code must
   encode the absolute state identity in a form GPT-2 can use as a match key. The different-table donor (planned) tests
   exactly that. Good.
3. **The short-length shortcut (§121's input-carriage trap):** at L ≤ 4 the 8-d code can hold the *op history*
   (3⁴ = 81 sequences), and the last segment can compose it through the table in one pass. That is exact in range,
   broken at length.
   - The curriculum to 32 plus SERIAL donor and length tests catch it.
   - Also add an **op-history probe** on the code (descriptive), as in §121.
4. **Cost:** with the table re-read in every segment, segments are about 7× (format: about 10×) longer. Fine on a 3090.
   Report FLOPs.

## (c) Size
- k = 6, m = 3 permutations is a sensible target: no synchronisation, Moore 6 per table, and an S6 subgroup.
- **For entry, pilot at k = 4, m = 2** (fewer pairs, shorter context, smaller lookup), with the same format. If it
  enters, go to k = 6, m = 3 under the same registration.
- Pre-register the size ladder now, so the move up isn't a post-hoc choice.
