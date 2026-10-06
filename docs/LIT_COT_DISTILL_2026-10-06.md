# Prior check for candidate 4 (the user's idea): distil a text-CoT teacher's per-step hidden states into the latent carrier (2026-10-06)

Literature session, for Theory; Audit copied. Brief. IDs verified earlier (LIT_INDEX); none new.

## 1. Prior art: per-step hidden-state distillation from a CoT teacher
- **ICoT-KD, Deng et al. 2311.01460:** a teacher with explicit CoT. The student ("emulator") predicts the teacher's
  *hidden states across layers* (vertical reasoning), and a third model reads them out. The original "distil CoT into
  hidden states".
- **CODI, 2502.21074:** self-distillation, L1 on the hidden state of the token before the answer, all layers,
  std-normalised. Single point, not per step.
- **KaVa, 2510.02312:** distils the teacher's **compressed KV cache** *stepwise* into latent tokens. It is notably
  **more robust to natural-language traces than equation-only methods**, which directly supports the user's premise
  that hidden states are a better target than CoT text.
- **SIM-CoT, 2509.20317:** per-latent step supervision, through a *text* decoder, not hidden states.
- **Distilling System 2 into System 1, 2407.06023:** distils System 2 *outputs* (not hidden states).
- **System-1.5, 2505.18962:** two-stage self-distillation (language CoT → latent → shortcut paths).
- **LIFT, 2609.38149:** teacher-forced state targets derived from a pretrained LM's next-token distribution, fed back.
- **Background:** PCCoT 2506.18582, CoLaR 2505.16552, SemCoT 2510.24940 (semantic alignment to ground-truth reasoning).
- **Analysis:** 2606.20075. *Geometric imitation* of teacher states can collapse the latent space, while *generative
  reconstruction* preserves information, and accuracy tracks the latent–step mutual information.

## 2. Teacher layer and token
- **Token:** use the teacher's hidden state at the **last token of each step's written state** (after the state text
  is emitted), where the content is the state.
  - Step-start or op tokens mix in the *operation*, which is the §121 / §132 input-carriage trap. A student matched to
    them is pushed to encode op history.
- **Layer:** mid-to-late. Pick it by a **linear state probe** on the teacher (state accuracy vs layer), not by default.
  Li, Guo & Andreas 2503.02854 locate permutation state in GPT-2 / Pythia's middle-to-upper layers. CODI uses all
  layers, std-normalised.
  - A single probe-chosen layer is reasonable. Report the probe curve.

## 3. Known failures
1. **Flat / bypassed students:** CODI-style distillation gave flat students in our §119 (0/3). SCIT 2608.27265 finds
   CODI's computation lives in the value cache, not the latents. With your Markov segments and masked fork the bypass
   route is closed, which is the right fix.
2. **Distilling nuisance:** the teacher's hidden state at a step encodes the state *plus* token, position and format.
   A linear map from an 8-d code can't reproduce that, so the loss pressures the code to spend capacity on nuisance.
   - **Distil only the state-predictive subspace:** project teacher states onto the probe's state subspace (or a CCA /
     PLS subspace against the true state). Note that this uses labels *for the teacher's probe*, so say so.
   - Or use a **generative / contrastive** target (InfoNCE between the student code and the matching teacher step
     state, against other steps), per 2606.20075, instead of cosine regression.
3. **A discrete student and a continuous teacher:** regressing a W_up(FSQ-8) embedding onto a 768-d continuous state
   can't be exact, and its residual gradient leaks into the FSQ encoder as noise. The contrastive or subspace targets
   above avoid demanding an exact match.
4. **Copying teacher shortcuts:** if the teacher's hidden state already uses a shortcut (e.g. a parallel-scan component,
   Li 2503.02854), the student inherits it. Check the teacher's own per-step state probe before distilling.

## 4. Constant vs annealed distillation weight
- **Constant** is the common choice in hidden-state distillation (CODI keeps β constant). ICoT-SI and Coconut anneal
  *text* supervision away, not a hidden-state distillation term.
- Our §126 ANNEAL and §89 α → 1 collapses argue against removing an anchor. **Keep it constant**, as proposed.

## 5. Novelty
- **New as scoped.** Per-step hidden-state distillation into a *discrete* (FSQ) latent carrier in a Markov chain,
  scored by certificate and donor SERIAL, isn't found.
- The distillation idea itself is known: ICoT-KD (the original), KaVa (stepwise, the closest), CODI.
- The text-target control (§118 ANCHOR) is the right comparator: teacher-state vs self-text targets.
- **Scope:** the teacher is trained on *labelled CoT*, so this is per-step-supervised by proxy, not C4.
