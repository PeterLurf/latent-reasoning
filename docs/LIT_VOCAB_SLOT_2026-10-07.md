# Token-shaped working memory between passes: a prior check for B4 VOCAB-SLOT (2026-10-07)

Literature session, for Audit (relaying the user: "improve the state / latent space between transformer passes … working memory that mimics
discretised tokens") and Theory (B4's prior check). IDs verified via the arXiv API (2026-10-07).

**B4:** K = 4 write positions; slot_k = Σ softmax(LM-head(h(w_k)) / τ) · wte; soft → straight-through hard argmax (p_hard 0 → 1); Markov
segments, so the slots are the only cross-step channel; FINAL-only training on §140's digit pairs.

## (a) Audit's leads: all 10 verify

| ID | Paper | What it does | Relevance to B4 |
|---|---|---|---|
| 2505.15778 | Soft Thinking | training-free; next input = probability-weighted mixture of token embeddings | B4's soft phase, without training |
| 2509.19170 | Soft Tokens, Hard Truths | RL with soft tokens (mixtures + input-embedding noise), no CoT distillation; **train soft, infer hard is best** | the hardening question, at 8B |
| 2607.18264 | MUX | each latent = a geometric-decay-weighted superposition of a *span* of tokens, lossless by construction, decodable | the lossless target (c) |
| 2509.25020 | MarCos | layers map a continuous thought to the *distribution* of the next thought (a Markov chain of thoughts) | Markov thought transitions; continuous |
| 2502.03275 | Token Assorted | VQ-VAE latent tokens as a *new* vocabulary, mixed with text | discrete codes, but a new codebook |
| 2606.29712 | DLR | a discrete latent vocabulary built by clustering rendered-CoT image features; up to 20× compression | discrete, new vocabulary |
| 2604.22709 | Abstract-CoT | a reserved discrete vocabulary; bottleneck SFT ↔ self-distil, then RL with constrained decoding | discrete latent policy; reserved, not natural, tokens |
| 2502.06049 | LM2 | a memory bank, read by cross-attention and gate-updated, beside the stream | a continuous memory; not token-shaped |
| 2606.13862 | SuperThoughts | two CoT tokens compressed per latent, decoded by an MTP head | superposition of 2 tokens with supervision |
| 2602.14077 | GTS | a learned Gaussian sampler over latent states (GRPO), not heuristic noise | exploration over continuous latents |

## (b) Closest priors to "vocabulary-constrained slots as the only carried state"
- **Real text as the only channel (autoregressive, not parallel slots):**
  - Markovian Transformers 2404.18988: the CoT is the sole bottleneck to the answer, with the question removed; RL.
  - The Markovian Thinker / Delethink 2510.06557: bounded chunks with a text carryover.
  - Thinking States 2602.08332.
  - Our own §92 / §97 / §100 text-state chunks: supervised, and §100 reached 1.000 on 4×4 with k = 2 text restarts.
- **Learned discrete messages over a vocabulary, end to end (Gumbel / STE):** emergent communication. Havrylov & Titov 1705.11192
  (ST-Gumbel over symbol sequences); Lewis et al. 1706.05125; Lee et al. 1909.04499.
- **A frozen LM vocabulary as a quantisation codebook:** LQAE 2302.00902 (image → nearest RoBERTa token embeddings); SPAE 2306.17842
  (a frozen LLM vocabulary as a pyramid codebook); PEZ 2302.03668 (continuous prompts projected to the nearest tokens, with
  straight-through updates).
- **Not found:** K parallel slots on a pretrained LM's *own vocabulary simplex*, hardened by STE, as the *sole* cross-segment state,
  trained from final answers. **New as scoped.**
  - The closest combination is MULTI-SLOT (LIT_RUNSUM_MULTISLOT) with a VQ codebook fixed to GPT-2's wte (as LQAE / SPAE), plus
    Markovian Transformers' "text is the only channel".

## (c) Known failure modes
1. **Soft tokens collapse to the argmax.**
   - 2508.03440: under Soft Thinking, LLMs are "single-threaded"; they follow the top-probability token in the mixture (the Greedy
     Pitfall). **Gumbel-softmax noise** fixes much of it.
   - For B4: the soft phase may give little more than the hard one. Use Gumbel-softmax (1611.01144 / 1611.00712), not plain softmax.
2. **Off-manifold mixtures.**
   - A high-entropy mixture of 50k embeddings sits near the *mean* embedding (GPT-2's wte is anisotropic), so many different states
     map to near-identical inputs (blurring, the same symptom as feature collapse in LT-Tuning 2602.10229).
   - Low τ keeps mixtures near the vertices. **Log the slot entropy and the distance to the nearest wte row.**
3. **Hard-token information loss.**
   - A hard slot holds ≤ log₂ 50257 ≈ 15.6 bits; K = 4 gives about 62 bits, far above §140's need (an accumulator < 400 plus
     position).
   - So the hard capacity isn't binding. The risk is *learning* through STE, not capacity.
   - 2509.19170: training soft and inferring hard was best (at scale, with RL). Our §120: FSQ needed a p_hard ramp.
4. **Language drift (C7).** When real tokens become a task channel optimised end to end, they **drift from their natural meaning**
   (Lewis 1706.05125; Lee 1909.04499) and can **encode information steganographically** (Roger & Greenblatt 2310.18512). Slots will
   be *real tokens* but not necessarily *meaningful* ones.
5. **What MUX's lossless superposition solves.**
   - Each latent must be decodable back to its span (a geometric-decay weighting makes the superposition invertible). That gives a
     per-latent **local target**, which prevents collapse shortcuts (latents that carry nothing distinct).
   - B4 trains FINAL-only, so it has no local target. That's the collapse risk MUX names.
   - A MUX-style reconstruction of each slot's own input would be a label-free local target. It would need a defined span, which
     §140's segment provides (the digit pair plus the carried slots).

## (d) Do token-shaped states help arithmetic or number carriage?
- **When the tokens are supervised to be numbers, yes:**
  - scratchpads 2112.00114, Lee 2307.03381, Abacus 2405.17399;
  - our §100 (text states, 4×4 = 1.000).
  - Tokenisation matters: digit-wise > BPE chunks (Singh & Strouse 2402.14903). GPT-2's BPE has single tokens for many but not all
    1–3-digit numbers, so a slot holding "347" may be one token or not.
- **With FINAL-only training, no evidence that emergent slots adopt number semantics.** Emergent-communication codes are arbitrary
  (drift), so expect an arbitrary 4-symbol code.
  - The arithmetic still has to happen *in-pass* (140-3's barrier). Token shape changes the *carrier*, not the computation.
- **The cheap way to get number semantics: restrict the slot vocabulary to the 10 digit tokens (plus a blank).**
  - Each slot is then a digit, and 4 slots = a 4-digit accumulator: an abacus layout.
  - It tests (d) directly and makes C7 meaningful. It also turns the 50k-way STE into a 10–11-way choice (pitfall 1).

## Pitfalls specific to B4 (Theory's list)
1. **50k-way STE from scratch.** Emergent-communication results use vocabularies of 10s–1000s. Full-vocabulary ST-Gumbel has high
   variance and tends to use few tokens.
   - Run a **restricted sub-vocabulary arm**: digits + blank; or the top-N frequent tokens.
   - Use **Gumbel-softmax ST** rather than deterministic STE, for exploration (2508.03440).
2. **All slots collapse to one token.**
   - Positions differ, but nothing forces distinct content. Add a slot-ID (positions already provide one) and log the per-slot
     token histograms.
   - A batch-level usage-entropy bonus (codebook-utilisation style) is the standard fix if collapse appears.
3. **The schedule:** τ annealing (Gumbel papers: exponential decay to about 0.5) plus a p_hard ramp after entry (§120). Don't harden
   before the read path forms (§126's lesson: hardening too early recreated the plateau).
4. **The decoder seeing only slots:** keep it (the GEN-M-MASK lesson).
5. **The tied head is shared with the LM:** gradients through the LM head and wte for the slot path also move GPT-2's token geometry.
   Consider freezing wte for the slot read, or log the drift in the LM's own perplexity.

## Fair C7 claims when slots are real tokens
- **Fair:**
  - "the carried state is a sequence of K vocabulary tokens";
  - the state machine can be enumerated and certified over tokens (as with FSQ codes);
  - single-token swaps causally change the output (donor / C1).
- **Not fair without evidence:** "the slots are human-readable reasoning". Claim semantic readability only if the tokens match a
  task quantity (e.g., the digit slots equal the accumulator's digits) on held-out items, measured, not eyeballed.
- With the digit-restricted vocabulary that test is direct.

## Verdict
- **New as scoped.**
- **Recommended additions:**
  1. Gumbel-ST, not plain STE;
  2. a digit-restricted vocabulary arm (it tests (d) and makes C7 meaningful);
  3. slot-entropy, nearest-token-distance and per-slot-histogram logs;
  4. optionally, a MUX-style local reconstruction target against collapse.
