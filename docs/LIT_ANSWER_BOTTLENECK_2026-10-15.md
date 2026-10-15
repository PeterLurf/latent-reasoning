# ANSWER-BOTTLENECK lead: does any latent-CoT method decode the final answer with the question hidden? (SELR 2608.13570 check plus a search) (2026-10-15)

Literature session, for Audit (the request) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Sources:** SELR and RiM read in full text (pdftotext); the rest from abstracts or earlier full-text reads.
- **Evidence tag:** all EMPIRICAL.

## SELR 2608.13570: "Think in Latent, Explain in Language: Self-Explainable Latent Reasoning"; Zhao, Cao, Wang & Gui. VERIFIED.
- **The ANSWER decoder is NOT question-free.** The answer loss runs on one sequence, [Image] [Question] <bot> latents <eot> [Answer], so the answer is decoded **with the question in context** (§3.2, Fig. 2; Algorithm 1 builds E from the image, the question embeddings E(X_q) and <bot>).
- **Only the CoT decoder is question-free:** "Crucially, during CoT decoding, the model is not shown the original image or question--only the latent thoughts. This self-regularization forces the latent thoughts to become a self-contained information bottleneck." The CoT is decoded from [decoding prompt] <bot> latents <eot> [steps].
- **No ablation on question visibility** (for the answer or the CoT decoder). The ablations (LLaMA-3.2-1B, GSM8K) cover the CoT loss, variable vs fixed latent length, the sampling distribution and the single-step loss:
  - single-stage SELR 35.03; without the full CoT loss 33.97; with a variable latent length 32.83;
  - multi-stage SELR 42.46.
- **Its question-free CoT decoder** is the same design as SIM-CoT's step decoder (conditioned on z_k only) and LOTUS-aux's decoder: CoT-side bottlenecks, not answer-side.

## Search: is a final answer decoded from latents alone (question hidden) on GSM8K?
**None found for latent CoT.** Every GSM8K latent method we've read decodes the answer with the question in context:
- Coconut, CODI, PCCoT and SIM-CoT (same sequence);
- LOTUS (the question's KV cache plus the post-loop latents);
- RiM 2605.30343. Its readouts "may attend to the question and the memory blocks"; the default mask lets targets see the question.
  - RiM's Figure 2 draws a **variant 2: "Target cannot attend to Question"**, but I found **no reported result** for it in the text.
  - That's the closest design and **unreported**.
- SELR (above) and Heima 2501.19201 (a separate interpreter for the thinking tokens, VLM): CoT-side only.

**Text-CoT analogue (answer from the reasoning alone):**
- **Markovian Transformers, Viteri et al. 2404.18988:** "all information flowing from question to answer must pass through a bounded-length CoT", an autoencoder-style bottleneck with the answer predicted from the CoT alone. GRPO-style RL with a KL penalty to discourage steganography.
- That's the published precedent for an answer-side bottleneck, in **text**, not latents. It was already listed as a nearest competitor in our literature-role notes.

## Reading for ANSWER-BOTTLENECK
- **An answer decoder that cannot see the question,** so every answer-relevant bit must pass through the latent memory, is **not found for latent-CoT / GSM8K**.
  - The closest are RiM's unreported "target cannot attend to question" mask variant and Markovian Transformers' text bottleneck.
  - **Novelty: NEW for latent reasoning;** the principle is shared with Markovian Transformers.
- **Our harness fact (F4, earlier notes):** our answer decoder f_ans attends to q, so literals can be fetched from the question at no memory cost. Hiding q forces literal identities through the memory too, which raises the bandwidth the memory must carry.
  - LIT_STEPFACTOR_D1 / §204 D3: the string target's operand identities are ≈ 6 bits per literal token.
- **Risk to state:** with q hidden, a memory that copies the question (a "bag of literals") satisfies the bottleneck without serial computation. The bottleneck forces *use of the memory*, not *serial use*.
  - Pair it with the SERIAL / INPUT-SWAP dissociation (203-6) so a copy-the-question solution is distinguishable from computed progress.
