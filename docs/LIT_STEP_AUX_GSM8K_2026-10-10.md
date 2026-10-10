# Prior check for STEP-AUX-GSM8K: §175's per-pass aux decoder + DeltaMem + INJECT passes on GSM8K-Aug, GPT-2 small (2026-10-10)

Literature session, for Audit (the lead) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Tables:** read from the HTML (2026-10-10).

## 1. SIM-CoT 2509.20317, GPT-2 Table 1 (accuracy %; in-domain GSM8K-Aug; OOD GSM-Hard / MultiArith / SVAMP)
| Method | SIM-CoT | GSM8K-Aug | GSM-Hard | MultiArith | SVAMP | OOD avg | # tokens |
|---|---|---|---|---|---|---|---|
| SFT-CoT | ✗ | 42.7 | 9.0 | 85.0 | 41.6 | 45.2 | 27.6 |
| No-CoT | ✗ | 19.1 | 4.3 | 41.1 | 16.4 | 20.6 | 2.2 |
| iCoT | ✗ | 30.1 | 5.7 | 55.5 | 29.4 | 30.2 | 2.2 |
| **Coconut** | ✗ | **36.6** | 8.1 | 83.5 | 36.2 | 42.6 | 12.2 |
| **Coconut + SIM-CoT** | ✓ | **44.8 (+8.2)** | 9.3 | 90.8 | 40.7 | 46.9 | 12.2 |
| **CODI** | ✗ | **42.0** | 9.4 | 93.0 | 41.7 | 48.0 | 12.2 |
| **CODI + SIM-CoT** | ✓ | **42.6 (+0.6)** | 9.4 | 92.8 | 42.6 | 48.3 | 12.2 |
- **These are their own re-runs.** Coconut 36.6 vs the paper's 34.1; CODI 42.0 vs 43.7.
- **The split:** trained on **GSM8K-Aug (Deng et al.)**, evaluated on the "GSM8k-Aug test set (Cobbe et al.)" = the 1,319 GSM8K test items.
  - Same as Coconut / iCoT and our bench (LIT_BENCH_VERIFY).
  - Our bench reproduced Coconut + SIM-CoT 44.8 (checkpoint_28) and CODI + SIM-CoT 39.4.
- Llama-3.2-1B (Table 2): CODI + SIM-CoT 56.1 vs SFT-CoT 58.4.

## 2. Step-aligned aux decoding with a *persistent memory* (not latent tokens in context) on GSM8K
- **Persistent memory in latent reasoning, at GPT-2 scale, on GSM8K: AGCLR, 2606.07720** (Farhan & Chaudhary, 2026-06-05).
  - "Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning."
  - It adds to Coconut a **Gated Concept Stream**: a persistent residual memory maintained across all reasoning passes, with learned write / read /
    forget gates. It targets the "concept bottleneck" (intermediate states overwritten across passes).
  - **GPT-2: GSM8K 34.0 vs vanilla Coconut 31.4 (their run)**; CoT 40.6. HotpotQA EM 14.0 vs 10.4; ProsQA 96.0 vs 92.0.
  - All three gates are necessary (Table 2 ablation), and the forget gate matters most.
  - **No step-aligned aux decoder.** It uses Coconut's stage curriculum.
  - **The closest prior to the memory half of STEP-AUX-GSM8K.**
- **Step-level supervision with "memory blocks": RiM, 2605.30343** ("Unlocking the Working Memory of LLMs for Latent Reasoning").
  - Fixed blocks of special tokens act as a working memory, processed in one forward pass.
  - A two-stage curriculum: (1) **predict explicit reasoning steps after each memory block** (step-aligned supervision); (2) drop step supervision and
    refine the final answer after each block.
  - Trained on GSM8K-Aug, tested on GSM8K / GSM-Hard, Llama-family models at several scales.
  - **Its memory is in-context tokens, not a persistent fast-weight memory.**
- **Not found:** step-aligned aux decoding **plus** a persistent (fast-weight) memory as the cross-pass state on GSM8K. **STEP-AUX-GSM8K is new as
  scoped.** AGCLR (persistent memory, no step aux) and SIM-CoT / RiM (step supervision, latent or in-context tokens) are the two halves.

## 3. Does PCCoT's 49.48 use the same split?
- **Yes.** PCCoT's dataset whynlp/gsm8k-aug has **train 385,620** (= Deng et al.'s GSM8K-Aug), validation 500, and **test 1,319**. The first test items
  match our bench file (LIT_PCCOT_PROBE addendum; b173fcc7).
- **One caveat:** PCCoT's "CoT 44.1" row is copied from CODI, and SIM-CoT's SFT-CoT is 42.7. Use our bench's own CoT (41.8) as the matched reference.

## Notes for the lead
1. **The GPT-2 GSM8K bar for STEP-AUX-GSM8K:**
   - latent methods: PCCoT 49.48, Coconut + SIM-CoT 44.8, CODI 42.0–43.7, AGCLR 34.0;
   - text CoT: 41.8 on our bench.
2. **Cite AGCLR as the closest memory prior.** Their gate ablation suggests a **forget / decay** path matters on GSM8K. DeltaMem's 6 gated-delta heads
   cover it, but the 6 additive no-decay heads don't. Log per-head-type usage.
3. **GSM8K steps are DAG-structured, with variable count** (LIT_S149): SIM-CoT pools the tail steps into the last latent. Decide how passes beyond the
   step count are supervised (HOLD, LIT_STEP_AUX_HOLD).
