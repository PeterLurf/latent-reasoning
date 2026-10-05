# Prior check for the task-agnostic proposals P-A to P-D (no hand-built snap templates or schedules; one design on ≥ 2 of 3 families) (2026-10-04)

Literature session, for Theory; Audit copied. IDs verified (arXiv API / abstract pages; LIT_INDEX). It builds on
LIT_TASK_AGNOSTIC (b284f980).

## (a) Discrete bottlenecks on recurrent state; is P-A (FSQ at every hand-off) new?
- **Priors:**
  - **QBN 1811.12530:** quantised bottlenecks inserted into trained recurrent policies, extracting and *minimising*
    Moore machines (post hoc).
  - **SR-RNN 1901.08817:** learned centroids during training, easy DFA extraction (from scratch).
  - L* extraction: Weiss 1711.09576. Saturated RNNs are automata: 2004.08500. DeepDFA 2408.08622.
  - Argmax re-embedding of factored symbols each iteration: 2510.14095 (task-specific factors).
  - In latent CoT: VQ codes for *text spans* (Token Assorted 2502.03275), learned discrete vocabularies (Abstract-CoT
    2604.22709, DLR 2606.29712), Gumbel in latent SFT (2510.15522). None quantises the carried state at every
    hand-off.
  - FSQ 2309.15505 itself is for VQ-VAE image and audio codes.
- **Verdict: new as scoped.** "FSQ at every hand-off of a pretrained LM's latent chain, task-agnostic, certified by
  enumerating reachable (code, item) transitions" is not found. Closest: SR-RNN (from scratch, learned centroids) and
  QBN (post hoc).
- **Why the certificate is sound under Markov segments:** the next code is a deterministic function of (code, item
  tokens), so the reachable transition table *is* the model.
  - Exactness on that table implies exactness at every length.
  - **Moore minimisation** of the induced machine (outputs = the answer / reader decode), as QBN did, gives
    codes-per-state and a minimal automaton you can compare with the true one.

## (b) P-B: denoising / contractive continuous dynamics
- **Priors:**
  - error-correction noise (2510.14095; one task);
  - nGPT hypersphere 2410.01131;
  - Stable Recurrent Models 1805.10369 (stable RNNs are well approximated by feed-forward models; stability limits
    long memory);
  - Attractor Models 2605.12466 / DEQ (a fixed-point restoring map);
  - mixture-collapse dynamics 2609.02049.
- **Is the gain over READ plausible?** READ's no-snap per-hand-off decays (§115: 0.94 → 0.86 on s6). A continuous
  carrier with noise plus normalisation can slow drift, but reaching about 0.9999 per hand-off at 512 without any
  discretisation has no precedent in pretrained LMs.
- **Expectation:** P-A (drift impossible by construction) > P-B. Stable Recurrent Models also warns that contractive
  maps forget, so the tension between stability and memory is real.

## (c) P-C: learned-notation checkpoints (free m-token writes at a learned gate, with a history cut)
- **Priors:**
  - InftyThink 2503.06692 (learned summaries);
  - Delethink 2510.06557 (learned content, fixed timing, RL);
  - PENCIL 2503.14337 (learned timing; per-task traces);
  - LightThinker 2502.15589 (learned compression timing);
  - gisting 2304.08467, AutoCompressors 2305.14788.
- **Emergent discrete notation:** Havrylov & Titov 1705.11192 (sequences of discrete symbols between agents;
  **straight-through Gumbel converges much faster and better than RL**).
- **Cost:** emergent languages are effective but **not interpretable or compositional by default** (Kottur et al.
  1706.08502). So C7 isn't free. Use ST-Gumbel over RL.

## (d) The non-group task family: which is most standard and reviewer-proof?
- **Delétang et al. 2207.02098 (Chomsky hierarchy).** The standard length-generalisation protocol (train 1–40, test
  41–500) with many published baselines (DeltaProduct, Grazzi, MLP-LDRU, Ruoss). Its regular tasks include group-like
  ones (parity, cycle navigation, modular arithmetic) and non-group DFAs (e.g. even pairs). The most reviewer-proof
  choice.
- **Flip-flop language modelling 2306.00946.** A **reset (non-invertible) automaton**, the canonical non-group
  Krohn–Rhodes component, with known transformer "attention glitches" on long sequences. It cleanly complements S5
  (pure group).
- Liu 2210.10749 automata (Dyck, gridworld, semiautomata), and random DFAs (Allchin 2607.11953 uses held-out
  permutation automata), are also usable.
- **Recommendation:** flip-flop (non-group, reset) plus one Delétang regular task (shared protocol). Avoid random DFAs
  as the *primary* family, since they have no published baselines.

## (e) Traps
1. **ST-FSQ inside GPT-2.** FSQ works on a few bounded dimensions (tanh, then round). You need a learned projection
   from 768 dimensions down to d (6–10) and back up. Standardise first (anisotropy, §73).
   - ST gradients are biased, and quantisation noise early on can reproduce the **no-read basin**.
   - Use the §111 reader from step 0, and consider soft-to-hard annealing of the rounding. But don't anneal *away*
     from a working continuous carrier, which is the posterior-collapse lesson.
2. **Collapse:** FSQ has no learned codebook, so there's no dead-code collapse, but *usage* can still be low. Report
   code usage and perplexity, and codes per true state.
3. **State counts.** Streaming multiplication's carry is bounded (≤ len(a) + 1 digits), but with a = 4 that is up to
   about 10⁵ states. The FSQ grid (L^d) must exceed that comfortably. Enumeration certification is then about
   10⁵ × 10 transitions: feasible, but log it.
4. **Is a dataset step trace a "per-task template"?** Scope it honestly. The *snap* (FSQ) and the *timing* are
   task-agnostic; the *supervision* is task-specific if the reader uses a trace format we designed (S5 state text,
   the streaming-multiplication format).
   - Cleanest claim: **final-answer-only** (reader off) as the primary arm.
   - The reader arm counts as "task-agnostic architecture, per-task supervision", and is acceptable only with each
     dataset's native trace.
