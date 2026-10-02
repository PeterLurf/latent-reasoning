# Prior check for §74: zero-shot transfer of the certified latent cell to held-out op phrasings (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch).

## 1. Zero-shot lexical or paraphrase transfer of heads on frozen LM features
- **The closest analogue is cross-lingual zero-shot transfer:** a task head trained on mBERT features in one language
  works in another with no target-language training (Pires et al. 1906.01502). This is the same pattern as "train on
  `swap 1 3`, test on `exchange boxes 1 and 3`": the head is fixed, and the frozen encoder supplies the equivalence.
- **Numeracy in embeddings:** Wallace et al. 1909.07940 probe frozen embeddings (GloVe, ELMo, BERT, …) for number
  value. Value is linearly decodable, but probes extrapolate poorly outside the training range. Levy & Geva 2410.11781:
  LMs represent numbers digit by digit in base 10. Razeghi 2202.07206: numerical behaviour tracks pretraining term
  frequency. Survey: Thawani 2103.13136. **Not found: a published measurement of GPT-2's last-layer similarity between
  number words and digits ("one" vs "1").** Measure it directly (below).
- **Compositional generalisation with pretrained models:**
  - SCAN 1711.00350, COGS 2010.05465. Furrer 2007.08970: pre-training vs specialised architectures.
  - **Kim, Linzen & Smolensky 2212.10769:** pretrained models' "lexical generalisation" on COGS is inflated by
    *pretraining exposure* to the held-out words. Replacing them with novel embeddings collapses performance (inverse
    scaling with pretraining data).
  - For §74 this cuts both ways. It is exactly the mechanism you want to demonstrate, so pretraining knowledge is doing
    the work. But it also means the random-init failure is close to **guaranteed by construction**: unseen word
    embeddings are random and untrained. The (i) vs (ii) contrast confirms an expected effect; it doesn't discover one.
- **Instruction-paraphrase robustness:** Sun et al. 2306.11270 (instruction-tuned LMs degrade on unseen instruction
  phrasings); PromptRobust 2306.04528.

## 2. LM-hosted state machines tested on unseen op phrasings
- **Boxes / entity tracking:** Kim & Schuster 2305.02363 (fine-tuned models track entities across a different entity
  set and longer sequences; operation-phrasing transfer is not in the abstract). Prakash et al. 2402.14811 (fine-tuning
  enhances an existing entity-tracking circuit). Li et al. 2106.00737 (probes of implicit world state in Alchemy and
  TextWorld).
- **Not found:** a *discrete or latent transition table hosted by an LM*, certified exhaustively, then tested for
  per-entry agreement under held-out op phrasings. **New as scoped.** Exhaustive agreement over every (code, op) is a
  stronger measure than accuracy on sampled sequences.

## 3. Failure modes to expect
1. **Reversed order ("swap 3 1").**
   - If training phrasings always have i < j, the reversed form is OOD *syntax*, not a lexical substitution.
   - Nothing in LM pretraining guarantees that "swap" is symmetric in its arguments. Causal features (order-sensitive,
     and dominated by the last token) will likely map "3 1" to a different op representation.
   - Expect this family to fail unless training included both orders. If it did, it isn't held out. Report it
     separately from the lexical families.
2. **Ordinals and number words.**
   - " one" and " 1" are different BPE tokens. Any equivalence comes from contextual features in middle or late layers.
   - Ordinals ("the first and third") also move the number into a different syntactic slot and token position.
   - Expect partial transfer; number words are more likely than ordinals.
3. **Frozen standardisation statistics (§73).**
   - The per-coordinate statistics come from the training phrasings' features.
   - Coordinates that are near-constant across training ops (template tokens such as "swap") get a large 1/σ gain.
   - A paraphrase changes exactly those coordinates, so the standardised features can be thrown far off the training
     manifold even when the raw features are close.
   - Pre-register an ε or σ-floor, and report agreement both with and without dropping low-σ coordinates.
4. **Readout position.** If op features come from a fixed position, such as the last op token, phrasings of different
   lengths or argument orders change *which* token is read. State the pooling rule, and consider mean-pooling over the
   op span as a pre-registered variant.

## Suggested additions (cheap)
- **Diagnostics before the agreement test:** cosine and nearest-neighbour rank between raw (unstandardised) op features
  for each paraphrase and its training form. For example, is "swap one and three" closer to "swap 1 3" than to the
  other 5 ops? This predicts agreement and separates "features don't match" from "the head doesn't transfer".
- **A positive-control family:** whitespace or casing variants ("Swap 1 3"), which should transfer.
- **A negative-control family:** a semantically different op with similar surface form ("swap 1 4" where 1 3 is
  expected; or "copy 1 3"). Agreement should be *low*, showing the test isn't trivially satisfied.
- **Random-init framing:** report (ii) as the expected floor (Kim et al. 2212.10769), not as the main evidence. The
  informative result is *which families* transfer under pretrained GPT-2, and whether the agreement is exactly 100 %.
