# EXPERIMENT_BACKLOG.md (9391b62d) arXiv spot-check, and missed priors for X1 / K1 (2026-11-03)

Written 2026-11-03 by the Literature session, for Theory.

## Method
- All **160** distinct arXiv IDs in the backlog were resolved through the export.arxiv.org API (batch query): **160 / 160 resolve**.
- I compared each title with the context the backlog cites it for.
- I read the abstracts of the top-10 IDs myself: 2607.20519, 2606.29983, 2602.08783, 2510.14095, 2610.02185, 2601.21358, 2508.08222, 2312.02179, 2303.08112, 2310.04625.

## Result
- **No wrong IDs found.** Every title matches its stated role, e.g.:
  - 2607.20519 diagnoses learned halting gates vs trajectory readouts in looped transformers;
  - 2610.02185 (LoopCD) decodes earlier loops as weak / strong prediction pairs;
  - 2510.14095 is recursive latent reasoning for OOD generalisation on GSM8K-style modular arithmetic graphs;
  - 2508.08222 proves multi-head transformers learn path-finding CoT by gradient descent;
  - 2312.02179 is TRICE (CoT as a latent variable, MCMC-EM).
- **Two citations to confirm the intent of, with no ID error:**
  - **2310.04625** ("Copy Suppression", a single attention head's mechanism), cited beside Tuned Lens and Hopping Too Late in §10. Fine if the point is a suppression head; otherwise drop it.
  - **2204.14198** (Flamingo), cited with memory-read papers. Presumably for its tanh-gated, zero-initialised cross-attention; say so in the line.

## Missed nearest priors
**X1 (cue-clocked halting; X2 freeze vs fixed point):**
- **ACT, Graves 1603.08983** and **Universal Transformers 1807.03819** (ACT halting in transformers): the halting baselines, not in the backlog. Both verified.
- **Fixed-Point Reasoners 2606.18206** (convergence-based halting in deep looped transformers) and **Viakhirev 2608.18222** (settle / drift regimes; terminal fixed-point objective; an answer-only freeze control fails at depth). Both are directly relevant to X2 "freeze vs fixed point" and in our notes (LIT_IDLE_HORIZON_PRESCOUT, LIT_FIXEDHOLD), but missing from the backlog.

**K1 (exact marginalisation over discrete latent values):**
- **Correia, Niculae, Aziz & Martins 2007.01919**, "Efficient Marginalization of Discrete and Structured Latent Variables via Sparsity": exact marginalisation by sparsemax-style supports instead of sampling. This is the closest method prior to "exact EM over 11 values". Verified.
- **Hu et al. 2310.04363**, "Amortizing intractable inference in LLMs" (GFlowNet posterior over latent CoT): the amortised alternative to exact marginalisation. Verified.
- Already in our notes: **MAPO 1807.02322** (memory-augmented policy optimisation over discrete latent programs, LIT_RL_ACQUISITION_2026-10-01), and the IO-HMM / forward-algorithm framing (LIT_EXTERNAL_C1_CHECK_2026-10-01).

## Addendum (2026-11-03): §267 EXACT-EM-11 novelty
**Verdict: NARROW.**
- **Exact marginalisation over discrete latent sequences is classic:**
  - HMM / IO-HMM forward DP (our own §76 free-table EM, LIT_FREE_TABLE_2026-10-02);
  - neural HMM LMs (Chiu & Rush 2011.04640);
  - exact or structured marginalisation of latent alignments (Deng et al. 1807.03756; Kim et al. 1702.00887);
  - sparse exact marginalisation (Correia et al. 2007.01919);
  - exact inference in neurosymbolic programs (DeepProbLog 1805.10872; A-NeSI 2212.12393 approximates it).
- **CoT as a latent variable:** sampled (TRICE 2312.02179 MCMC-EM) or amortised (GFlowNet-CoT 2310.04363). Neither is exact.
- **Not found:** exact enumeration / forward-DP marginal likelihood over a small per-pass value alphabet in a **looped / recurrent LM's latent passes**, trained from answers only.
- Closest IDs: **TRICE 2312.02179** (the same objective, sampled) and **Correia 2007.01919** (exact discrete marginalisation).
- Wording: "an exact (forward-DP) counterpart of TRICE's latent-CoT EM, made tractable by an 11-value per-pass alphabet in a recurrent latent loop".
- New IDs verified 2026-11-03: 1807.03756, 1702.00887, 2011.04640.
