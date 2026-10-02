# Prior check for §72 (frozen pretrained vs frozen random GPT-2 cell; position-0 readout; head learning rate) (2026-10-02)

Literature session, for Theory; Audit copied. A short note: §72 is a diagnostic, not a method. arXiv IDs verified
(Semantic Scholar batch). Specific claims were read from the papers.

## 1. Frozen pretrained vs frozen random vs trainable, with a small trainable interface
- **Lu et al. 2103.05247 (Frozen Pretrained Transformer):** GPT-2 with self-attention and FFN frozen; input layer,
  output layer, LayerNorm (and positional embeddings) trained. Tasks include bit memory/XOR, ListOps and CIFAR. The
  language-pretrained frozen model beats the frozen *random-init* one in performance and compute. **The direct
  precedent for arms (i) vs (ii).** Note that FPT trains the LayerNorm parameters. If §72 freezes LN, say so, because
  that differs from FPT.
- **Zhong & Andreas 2410.04368 (NeurIPS'24):** random-init transformers with **only embeddings and unembeddings
  trained** solve modular arithmetic, in-weights/in-context associative recall, decimal addition and parenthesis
  balancing. So arm (ii) is a real baseline, not a strawman: some discrete functions are reachable through
  embeddings alone.
- Also: Reservoir Transformers 2012.15045 (frozen random layers interleaved with trained ones). Probe-baseline
  practice: Hewitt & Liang 1909.03368 control tasks (selectivity: compare against a representation or task that
  should *not* support the function).
- **Not found:** a frozen-pretrained vs frozen-random vs trainable comparison on *learning an automaton transition
  with an exhaustive certificate*. §72 is a diagnostic within established practice. No novelty claim is needed.

## 2. Position-0 readout in GPT-2: the risk is real
- **Sun et al. 2402.17762:** GPT-2 is in the group with **massive activations at the starting token only**. Massive
  activations "act as fixed but important biases", and deep-layer attention concentrates on those tokens.
- **Gu et al. 2410.10781 (ICLR'25):** the sink is **positional, not token-specific**. With a random first token in
  pretraining the sink still forms at position 1; fixing a token at position 2 moves the sink there. GPT-2 was
  pretrained without a forced BOS, so position 0 is the sink whatever token sits there.
- Also Xiao 2309.17453 (attention sinks; StreamingLLM keeps the first tokens) and Barbero 2504.02732 (why LLMs attend
  to the first token). Darcet 2309.16588 is the ViT analogue: registers absorb the artefact.
- **Implication for h_code at position 0:**
  - In a causal model, position 0's hidden state depends only on token 0 (the code) and pos-0, so it is not
    literally input-independent.
  - But from the middle layers on, the residual there is dominated by a few huge, near-constant sink dimensions.
  - After LayerNorm (dividing by a norm set by those dimensions), the code-specific component is compressed toward a
    common vector, so LN(h_code) can be nearly identical across codes.
  - This is an inference from the papers above, not a published measurement for trainable code embeddings. **Measure
    it:** the cosine and the between-code / within-code variance ratio of LN(h_code) at each layer, with and without
    a leading sink token.
- **Fix used in practice:** put a sink/BOS token (`<|endoftext|>`) first and read the code at position 1.
  TransformerLens prepends BOS by default for this reason (library documentation, not a paper). Alternatively read
  from an earlier layer.
  - **Recommend arm (i′):** BOS + code. If (i′) ≫ (i), the 1/5 failure is a readout artefact, not a pretraining
    deficit.
  - Arm (ii) (random-init) has no learned sink, so a pos-0 artefact would *favour* (ii) over (i). Keep that confound
    in mind when you interpret (i) vs (ii).

## 3. Learning rate for new heads and embeddings vs the backbone
- **Kumar et al. 2202.10054 (LP-FT, ICLR'22):** fine-tuning with a randomly initialised head distorts pretrained
  features. Linear-probe first, then fine-tune. The head's state at the start of fine-tuning matters.
- **Discriminative learning rates:** ULMFiT 1801.06146 (per-layer learning rates). LoRA+ 2402.12354: equal learning
  rates for the A and B adapters are provably suboptimal at large width; set η_B = λ·η_A (λ ≫ 1).
- **New embeddings need far larger learning rates:** prompt tuning (Lester 2104.08691) trains soft-prompt embeddings
  at learning rate **0.3**, against 0.001 for full model tuning (both Adafactor). That is a 300× gap.
- **µP (Tensor Programs V, 2203.03466):** the optimal learning rate differs by parameter type (embeddings, hidden,
  readout), so a single learning rate tuned for one type mis-scales the others.
- **Bearing on §72:**
  - If the GPT-2 cell's new codes and head were trained at the backbone's fine-tuning learning rate (≈ 100× below the
    toy cell's), the 1/5 vs 10/10 gap is confounded.
  - Give each arm's *new* parameters (codes and head) their own learning rate. Use a short sweep, or the toy cell's
    optimum.
  - Report a matched-learning-rate cell, (iii) with head/codes at the toy learning rate and the backbone at a low one,
    before attributing the gap to representations.
