# GSM-Symbolic and template resampling as a GSM8K *training* augmentation (after §191: per-draw relabelling lifts text CoT on fixed-set ProsQA from 0.735 to 0.938) (2026-10-13)

Literature session, for Audit (the request) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Licences:** from the HF / GitHub APIs (2026-10-13).

## GSM-Symbolic, Mirzadeh et al., arXiv 2410.05229 (Apple): verified
- **Authors:** Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, et al.
- **What it does:** builds symbolic templates and instantiates many variants. LLM accuracy varies across instantiations of the same question and drops when only
  the numbers change. Adding clauses (P1 / P2) hurts more.
- **The release:**
  - templates on GitHub (apple/ml-gsm-symbolic);
  - sample instances on HF **apple/GSM-Symbolic**, configs main / p1 / p2, **test splits only**;
  - **licence: CC-BY-NC-ND-4.0** (no derivatives).
- **CRITICAL: the templates are built from the GSM8K *test* set.**
  - "Given a specific example from the test set of GSM8K, we create parsable templates"; 100 templates × 50 samples, with number ranges matched to the test set.
  - **Training on GSM-Symbolic instances would contaminate our GSM8K test evaluation** (the same 1,319-item test the latent baselines use).
  - **The ND licence also bars distributing adapted versions.**
  - **Use it for evaluation only.**

## Template resampling as a *training* augmentation for GSM8K: what exists
- **TemplateGSM / Template-based Data Generation, 2411.18104 (verified; data CC-BY-4.0):**
  - GPT-4-generated **parameterised meta-templates built from the 7,473 GSM8K *train* problems as prototypes**;
  - placeholders for names, quantities, items, dates and locations;
  - programmatically verifiable solutions;
  - **more than 7M** generated problems; code at iiis-ai/TemplateMath.
  - **The closest match to "names / numbers redrawn", and train-derived (safe)**, but released as a large static set, not per-epoch resampling.
- **Other GSM8K training augmentations:**
  - **GSM8K-Aug** (Deng 2311.01460; 385,620, GPT-4 paraphrased new problems): what every latent paper uses; static;
  - **MetaMath 2309.12284** (rephrasing, backward / FOBAR questions);
  - **TinyGSM 2312.09241** (12.3M synthetic problems with Python solutions; small models beyond 80% with a verifier);
  - MuggleMath 2310.05506 (query / response augmentation).
- **Per-epoch template resampling for latent or implicit-CoT models: not found.**
  - All latent GPT-2 GSM8K work (iCoT, Coconut, CODI, SIM-CoT, PCCoT) trains on static GSM8K-Aug.
  - **So a "GSM8K-RELABEL" (resampling names / numbers per draw from train-derived templates) for a latent model would be new.** It's the GSM8K analogue of
    §191 control B.

## Recommendations
1. **Source train-derived templates only:** TemplateGSM's code / data (CC-BY-4.0, from the train split), or our own templating of GSM8K-Aug train items.
   **Never GSM-Symbolic** (test-derived, CC-BY-NC-ND).
2. **For a per-draw relabel:** resample names / items freely. Resampling *numbers* changes every intermediate value, so recompute the equation-format CoT
   (needed for the step aux) programmatically. TemplateGSM's verifiable solutions make that possible.
3. **Report GSM-Symbolic (main / p1) as an evaluation** of robustness to instantiation shifts. That's legitimate as eval-only, and directly tests whether the
   latent model memorised surface forms.
