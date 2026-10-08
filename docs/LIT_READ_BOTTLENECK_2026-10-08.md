# Read-side remedies for linear memories (CCQ, multi-query reads) and read interference: a check for the §157 READ-BOTTLENECK panel (2026-10-08)

Literature session, for Audit (the panel lead) and Theory.
- **IDs:** verified via arXiv abstract pages (2026-10-08).
- **CCQ details:** from its full HTML.

## 1. CCQ, 2606.01294 (Le, Nguyen, Nguyen, Luu …; EMNLP 2026 camera-ready): verified
- **Claims as stated, all correct:**
  - a second-order Taylor expansion of the softmax log-partition at the isotropic-attention point. Its curvature is the **running
    key covariance**, maintained by the same recurrent / chunkwise mechanism as the state;
  - the query is **contracted along the high-variance directions of memory before it reads the state**;
  - **read-step only**, composable with any write rule;
  - attached to **GLA and Gated DeltaNet**, it improves perplexity, zero-shot accuracy, **S-NIAH at and beyond the training context**,
    **4K → 20K extrapolation perplexity** and **LongBench**.
- **Design details** (HTML): it uses the **unweighted** running covariance C̄_t (the exact Hessian would be attention-weighted); a
  learned gate (a linear projection of the query) scales the correction; the appendix discusses the interaction with decayed backbones.
- **Does it apply to an additive no-decay elu+1 head?**
  - **In form, yes:**
    - it needs only a running second moment of the keys, updated by the same rule as the state;
    - with no decay, that is a cumulative mean, arguably the *simplest* case;
    - the paper's "kernel feature maps" section covers φ(k)ᵀφ(q) heads.
  - **Two caveats:**
    - (a) The derivation is for the exp kernel on raw keys. With elu+1, use the covariance of **φ(k)** in feature space, which is a
      heuristic transfer and **not tested in the paper** (only GLA / GDN, both decayed).
    - (b) With no decay, normalise C̄ by the count, or the contraction strength grows with the number of segments.
- **What it targets:** read *dilution* (interference: every stored key contributes to the read). That's not read *bandwidth*. 152-5's
  "single 768-d read is the bottleneck" is a bandwidth claim, so CCQ and MULTI-READ test different hypotheses. Keep them as separate
  arms.

## 2. A multi-query read (R > 1 read vectors per head, inserted as extra positions)
- **Precedents for several reads of one memory:**
  - **every token is a reader:** Infini-attention 2404.07143 and ARMT 2407.04841 (each position's query reads the associative
    memory);
  - LM2 2502.06049 (all input tokens cross-attend to the memory bank);
  - **Perceiver IO 2107.14795** (an array of learned output queries decodes a latent array: the cleanest "R read queries" design);
  - **Memory Mosaics 2405.06394** (a network of associative memories, each read by its own query);
  - **RMT 2207.06881** (read / write memory tokens);
  - TransformerFAM (feedback-token block).
- **Read bandwidth ablated separately from memory size: not found.**
  - In RMT, FAM and the Block-Recurrent Transformer the number of memory tokens *is* the memory size, so the two are confounded.
  - Zoology / Based ablate state size, not read count.
  - In matrix memories the two are separable (fixed d_k × d_v, varying R). **Ablating R at a fixed state would be new and clean.**
- **Design notes for MULTI-READ:**
  - Give each read position a distinct learned query and a slot-ID embedding (else the R reads are identical, the same symmetry issue
    as LIT_RUNSUM_MULTISLOT).
  - The compute-matched control is R extra positions with *no* memory read (pause-style). Otherwise "more read bandwidth" is
    confounded with "more positions".

## 3. Sanity check of the two capacity / interference papers
- **2609.33093, "How Linear Attention Remembers" (pretrained GLA / GDN, causal interventions): directly relevant to read
  interference.**
  - Facts enter through *concentrated, content-dependent writes* and are read through *concentrated query-time pathways*.
  - Several facts stay accessible, but with **cross-fact causal coupling** (not independent KV storage).
  - **Recall and editability degrade with memory load, much more than with elapsed context.** Interference from later writes scales
    with their **overlap** with existing memory.
  - **For us:** it predicts that interference in DeltaMem's read grows with the number of stored items, not the number of passes. The
    rank-1 carry edit (§155) may also show coupling to other stored columns. Pre-state that.
- **2607.17419, KATA, "Breaking the Capacity Wall with Symmetric Cones": relevant mainly as a capacity bound.**
  - It treats recall as spherical packing, with interference floored by the **Welch bound**. Rank-1 PSD features give a better
    capacity / interference trade-off, with 0.985 MQAR at 16× OOD length.
  - **For us:** with about 8 column keys (+ 8 operands in PURE) at d_k ≥ 64, we are **below** the Welch regime, so capacity
    shouldn't bind.
  - It becomes relevant only if the store grows (more columns, longer operands), or if elu+1 keys are far from orthogonal. A key
    cosine-similarity log would show that.

## Verdict for the panel
- CCQ (read dilution) and MULTI-READ (read bandwidth) test distinct hypotheses, and both are applicable to DeltaMem.
- A bandwidth ablation at a fixed state, with a pause-position control, would be new.
- Log key cosine similarity and per-item recall vs memory load (2609.33093's axis).
