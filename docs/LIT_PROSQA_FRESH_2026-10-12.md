# ProsQA-FRESH and data efficiency: a generator, priors on latent methods' data needs, and GSM8K data (after §168's memorisation: train 1.000, test 0.490) (2026-10-12)

Literature session, for Theory; Audit copied. IDs verified via arXiv abstract pages / the GitHub API (2026-10-12) or earlier notes.

## 1. Is a ProsQA generator available?
- **Coconut (facebookresearch/coconut, MIT):** the repo ships only the **fixed** splits (`data/prosqa_{train,valid,test}.json`) plus preprocessing for GSM8K and
  ProntoQA. **No ProsQA generator.**
  - **But the paper's appendix gives the construction in full (Alg. 1):**
    - incrementally add nodes, each with poisson(1.5) in-edges from existing nodes;
    - keep the node families of the two candidates separate with some probability, so they stay balanced (an anti-shortcut device);
    - parentless nodes get entity names, the others concept names (ProntoQA-style fictional names);
    - node 0 = the entity, leaf nodes labelled 1 / 2 = Concept A / B;
    - randomly permute the answer order.
  - **It's re-implementable from the paper.**
- **ProsQA-Ext 2609.35643: no code link** found (abstract page or HTML). Its appendix also specifies its generators:
  - 400k fresh 3–6-hop training graphs, quota-matched to ProsQA's size / proof-length / root / path-count distribution;
  - the OOD Alg. 1: two disjoint H-paths + 13 − H off-path nodes;
  - relabelled ID sets.
- **ProntoQA's generator is public:** asaparov/prontoqa (**Apache-2.0**; Saparov & He 2210.01240). Coconut's names follow it.
- **Ours:** `prosqa/make_long.py` / `make_deep.py` extend the *fixed* base problems (12k subsample), drawing chain nodes from the pool of names unused in
  each problem.
- **The cheapest "fresh" option, with no new generator needed:** **resample every step**:
  - (a) the hung chains (m and the new names);
  - (b) **a relabelling of all concept / entity names** (a random bijection over the name pool, as ProsQA-Ext relabels ID graphs);
  - (c) the premise order and candidate order.
  - That removes surface memorisation while keeping the 12k base graph structures.
  - **A true ProsQA-FRESH** (new graphs every step) needs re-implementing Coconut's Alg. 1 (about 1 page), or ProsQA-Ext's.

## 2. Priors on data efficiency of latent methods
- **Fresh or large data is the norm where latent or implicit reasoning generalises:**
  - ProsQA-Ext trains on **400k** fresh graphs (vs ProsQA's 17,886);
  - **Saparov 2412.04703:** transformers learn graph search **"when given the right training distribution"**, with "effectively limitless high-coverage data". The
    difficulty grows with graph size and isn't fixed by scale;
  - ICoT-SI uses 808k multiplication items, up to 200 epochs;
  - Feng 2305.15408: 1M samples per setting (CoT learns; direct fails).
- **Implicit (no-text) composition is data-hungry and grokking-dependent:**
  - **Wang et al. 2405.15071, "Grokked Transformers are Implicit Reasoners":** implicit multi-hop composition is acquired only through **grokking** (far
    past overfitting). **Generalisation depends on the ratio of inferred to atomic facts in training** (the data distribution, not just size). **OOD
    composition stays poor**, while comparison tasks generalise.
  - **Kohli 2604.07822:** systematic generalisation emerges in a **three-stage** process (memorisation → in-distribution → systematic) under a 95% threshold
    curriculum.
  - §168's train 1.000 / test 0.490 is **stage 1 (memorisation)** of that picture on a small fixed set.
- **Why text CoT transfers on the same data:** it gets dense token-level supervision of every hop, so it needs fewer examples per procedure. Implicit / latent
  composition gets the procedure only through the answer (or a decoded aux), which the grokking literature finds needs far more data or training.
- **Regularisers / curricula that push toward a procedure:**
  - **curriculum:** Coconut "w/o curriculum" collapses to 76.1 on ProsQA (Table 1); Kohli's threshold curriculum;
  - **stage mixing:** Coconut's config has `uniform_prob` (mixing samples from other stages; default 0.0);
  - **random latent / text mixing:** Token Assorted 2502.03275;
  - **removal smoothing:** ICoT-SI 2405.14838;
  - **dropout of the direct view:** Bowman word dropout (LIT_DEC_DROP);
  - **self-improvement on fresh harder items:** Lee 2502.01612.
  - **For memorisation specifically: data freshness / relabelling beats any of these.** The structure of 1(b) is the standard anti-memorisation device in
    synthetic reasoning (ProntoQA / ProsQA-Ext use fictional, relabelled names).

## 3. GSM8K training data in latent-reasoning papers
- **The standard is GSM8K-Aug** (Deng et al. 2311.01460): **385,620** training problems (GPT-4-generated augmentations of the 7,473-item GSM8K train set),
  with **equation-only CoT**.
  - Used by iCoT, Coconut, CODI, SIM-CoT, PCCoT, RiM and others. The test set = the 1,319 GSM8K items.
  - **GSM8K-Aug-NL** is the natural-language-CoT variant (CODI / PCCoT report both).
- It's already in our bench: `bench/data/gsm8k_aug/{train,valid,test}.txt`, plus Coconut's / PCCoT's HF copy (whynlp/gsm8k-aug: 385,620 / 500 / 1,319).
- **No latent-reasoning GPT-2 result trains on the 7.5k original alone.** All use the augmented 385k.

## Recommendation
**Before a full ProsQA-FRESH generator, run a cheap "ProsQA-RELABEL" arm:** the same 12k base graphs, with names relabelled, chains resampled and premises /
candidates reshuffled every step.
- If test rises well above 0.49, memorisation of surface forms was the barrier.
- If not, structural data volume is, and the Coconut Alg. 1 generator (or ProsQA-Ext's 400k-style sampler) is the next step.
