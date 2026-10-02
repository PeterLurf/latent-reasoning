# Prior check for §79: latent automaton with natural-language op paraphrases, held-out phrasing families (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified. It builds on LIT_PARAPHRASE_TRANSFER
(614e6ef).

## 1. Prior: generalisation to unseen instruction phrasings through frozen/pretrained LM features
- **Instruction following with pretrained text encoders, the direct analogue:**
  - Hill et al. 2005.09382: a BERT encoder lets an RL agent trained on synthetic template commands transfer zero-shot,
    substantially above chance, to natural human instructions.
  - Lynch & Sermanet 2005.07648: pretrained sentence embeddings make a policy robust to many out-of-distribution
    synonym instructions without new demonstrations.
  - The same pattern as §79: train on a few phrasings, transfer through the pretrained feature map.
- **Systematic-generalisation benchmarks:** gSCAN 2003.05161, SCAN 1711.00350, COGS 2010.05465. Pretraining helps
  lexical splits more than structural ones (Furrer 2007.08970).
- **Instruction-paraphrase robustness:** Sun et al. 2306.11270, where instruction-tuned LMs degrade on unseen
  phrasings.
- **Not found:** paraphrase transfer to a *certified latent automaton's* transitions, measured exhaustively per
  (code, op). New as scoped.

## 2. Is "train on multiple phrasings, test held-out ones" a known recipe?
- Yes in spirit: paraphrase or synonym augmentation of instructions is standard in grounded language learning (Hill;
  Lynch). Kim, Linzen & Smolensky 2212.10769 show the generalisation comes from pretraining exposure to the words, and
  novel embeddings collapse it.
- So §79's mechanism is expected to work where held-out phrasings use words GPT-2 has seen in similar contexts. What
  would be new is exact, certified per-entry agreement.
- **Controls:**
  - The phrasing-as-symbol table gives 0 by construction.
  - Random-init GPT-2 gives ≈ 0 by construction (2212.10769).
  - **The bag of pretrained word embeddings is the informative control.** If BoW matches GPT-2, transfer is keyword
    matching, not contextual understanding.

## 3. Pitfalls in choosing held-out families
1. **Lexical leakage.**
   - Define families by their *operative* words (verb plus argument encoding), and make the held-out families
     disjoint from the training families in those words.
   - Example: if "swap" appears in training, "swap the first and third" tests ordinals, not paraphrase.
   - Report the n-gram or word overlap between each held-out family and the training set.
2. **Template overlap.** Shared frames ("…boxes i and j") let features match on structure alone. Include at least one
   held-out family with a different frame, e.g. "the items in boxes 1 and 3 trade places".
3. **The BoW-solvable trap.**
   - S3's ops are the identity, 3 transpositions and 2 three-cycles. Transpositions are symmetric, so word order
     doesn't matter.
   - The two 3-cycles ("rotate left" vs "rotate right", "1→2→3" vs "1→3→2") differ only in order or direction words.
   - Include **minimal pairs** that a bag of words can't separate: direction-reversed 3-cycles, and negation or
     distractor phrasings ("swap 1 and 3, then swap them back" = identity). That makes BoW vs GPT-2 informative.
4. **Convention ambiguity.** "Rotate left/right" and "move each item to the next box" have direction conventions that
   annotators and paraphrase generators get wrong. Generate phrasings from a formal spec and verify each phrasing's
   label programmatically (or by two-annotator agreement). Otherwise label noise caps agreement below 100%.
5. **Per-family reporting.** Report exhaustive agreement per held-out family (as in §74), with:
   - a positive control (casing or whitespace);
   - a negative control (a different op with similar wording; agreement must be low);
   - the §73 frozen standardisation statistics and σ-floor check, since new phrasings shift low-σ coordinates.
6. **Number of training phrasings.** Transfer usually rises with phrasing diversity in training. Pre-register a
   sweep over the number of training families (e.g. 2/4/8) to show the trend, not just one point.
