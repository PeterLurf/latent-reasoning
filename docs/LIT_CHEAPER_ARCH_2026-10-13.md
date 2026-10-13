# Prior check: cheaper looped / latent architectures, and which keep the memory as the only cross-pass channel (2026-10-13)

Literature session, for Theory (the request, for the user's "research an architecture change that is cheaper") and Audit.
- **IDs:** all verified via arXiv abstract pages or HTML (2026-10-13).
- **Mechanics:** Penelope, LLoCoT and LOTUS from their HTML; the rest from abstracts and method sections.

## Short answer
1. **The main constraint-safe saving is one we already have.**
   - Penelope 2607.25915 is the published prior for our `--cache_lower` (062fd45d, note 168-8).
   - Under INJECT every think pass sees identical input, and DeltaMem enters at block 6 (`boxes/deltamem.py` l.3). So blocks 0–5 run once and are shared by all passes.
   - It's exact (fp64 agreement 6.9e-15), saves about 45% at K = 10, and the cache depends only on the input, so the memory stays the only cross-pass channel.
   - **It's held only because it can't be mixed with uncached comparators.** Running whole new families cached (cell *and* comparator) costs nothing in validity.
2. **The next lever is to move the recurrent interval up.**
   - Penelope loops only blocks [11, 16) of a 16-layer Llama-3 backbone and caches the lower 69%. Ours caches 6 / 12 = 50%.
   - With l_m = 9 we would cache 75%, so a pass costs 3 blocks instead of 6 (**about 2× cheaper per pass on top of `--cache_lower`**).
   - That's a real architecture change (the memory reads and writes later), so it needs its own arm. No paper tests where the read layer should sit for depth generalisation.
3. **Penelope's second cut:** within the interval, recur only on **8 interface + 2 readout slots** that attend to a **prompt-side KV cache computed once**. Per-pass tokens drop from T to 10.
   - The prompt cache is input-only, so it's constraint-safe.
   - But it's a slot design: our READ-SLOTS (§171) failed. Treat it as a third step, not the first.
4. **These violate "memory = the only cross-pass channel", so don't adopt them:**
   - **MELT 2605.07721:** one KV cache per layer shared across loops via a learned gate;
   - **LLA 2607.15456:** a KV cache compressed and carried across loops;
   - **RLT 2610.07591:** replays the sliding-window KV across iterations. That's the closest match to "recurrent KV replay", which wasn't found under that name;
   - **T²MLR 2607.15178:** feeds the previous token's mid-layer state into an earlier layer, a second (cross-token) recurrent channel.
5. **These are cheaper only relative to token-by-token CoT, not per pass:** LLoCoT 2610.11472, LOTUS 2606.31779, LRT 2609.01117.
   - Their savings come from parallel latent slots, or from a tiny separate reasoner instead of the LM.
   - They don't cut the cost of an INJECT pass.
6. **Inference-only, not relevant to our training cost:** FlashLoop 2609.29812 (training-free) and MoR 2507.10524 (from-scratch pretraining with routers).
7. **Caveat:** `--cache_lower` needs bitwise-identical think input. A pass index on the think token (note 177-4's e_t, added at the input embedding) breaks it. Any pass signal must enter at or above l_m. This is already asserted in prosqa168s2.

## Per paper: (a) recomputed per loop; (b) cached or frozen; (c) any cross-loop channel besides an explicit state; (d) depth results

| paper | (a) per loop | (b) cached | (c) extra channel? | (d) depth results |
|---|---|---|---|---|
| **Penelope** 2607.25915 | blocks [11, 16) on 8 interface + 2 readout states (time-modulated GRU update) | the lower prefix [0, 11) once; a prompt-side KV cache inside the interval | none; the prompt cache is input-only | Deep ListOps, ProsQA, PrOntoQA (and Deep ListOps on a Qwen3 backbone): competitive with Coconut on the same backbone, lower latency; training ramps K 1 → 6, then fixes K = 6 |
| **LLoCoT** 2610.11472 | the full shared transformer on the prompt plus K learned slots, for R iterations; latent means are fed back into the slots | nothing beyond the prompt | the slots *are* the state | code only (HumanEval / MBPP): on par with Reasoning SFT, about 36× faster TTFT |
| **LOTUS** 2606.31779 | the full model on the prompt plus K parallel latent blocks, for R iterations; CE on each latent's gold CoT-step token | none | the latent blocks are the state | GSM8K: within 1 point of explicit CoT at 3B (Llama-3.2-3B-Instruct); 2.5–6.9× lower thought latency |
| **LRT** 2609.01117 | a tiny recurrent reasoner over the latents | the frozen LLM runs once | none (the reasoner's state) | not extracted |
| **MELT** 2605.07721 | the looped layers | **a per-layer KV cache shared across loops (gated)** | **yes: the KV** | not extracted |
| **LLA** 2607.15456 | the looped layers | **a KV compressed and carried across loops** (memory, not FLOPs) | **yes: the KV** | not extracted |
| **RLT** 2610.07591 | the recurrent block | **a sliding-window KV replayed across iterations** | **yes: the KV** | not extracted |
| **T²MLR** 2607.15178 | the standard forward | the previous token's mid-layer state | **yes: cross-token, mid → early layer** | not extracted |
| **FlashLoop** 2609.29812 | only the tokens whose state still changes (token-sparse), with sparse attention | a KV residual, quantized | none (inference-time) | lossless at 1.64× speed-up and 6× less KV memory |
| **MoR** 2507.10524 | a per-token number of recursions, chosen by a router | KV for active tokens only (variants) | none | from-scratch pretraining (perplexity / few-shot), no depth tasks |

Perceiver-style slot priors (Perceiver 2103.03206, RIN 2212.11972, Q-Former 2301.12597, Resampler 2204.14198, AutoCompressors 2305.14788) are covered in LIT_S171_S174.

## Suggested order
1. **`--cache_lower` as the default** for every *new* family, with comparators cached too. It's free.
2. **An l_m = 9 arm** (cache blocks 0–8). It's the Penelope interval choice, applied to us.
3. **Only after that, Penelope-style slots-over-prompt-cache.** It's cheapest, but it's the riskiest given §171.
