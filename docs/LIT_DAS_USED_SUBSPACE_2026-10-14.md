# Prior art: locating USED intermediate-value subspaces by interchange intervention (DAS) in recurrent / looped models (pre-emptive, after §203's G-use DO-NOT-LAUNCH) (2026-10-14)

Literature session, for Theory and Audit.
- **IDs:** verified (arXiv abstract pages). **Claims:** from abstracts.
- **In-house precedent:** boxes/das167.py (§167 DAS-EDIT on the DeltaMem-R memory).

## The core methods
- **DAS:** Geiger et al. **2303.02536** (EMPIRICAL + a THEORY frame). It learns a rotation so that an interchange intervention on a low-dimensional subspace aligns with a high-level causal variable. The scored quantity is interchange-intervention accuracy (IIA).
  - Theory: **causal abstraction 2301.04709** (THEOREM-level formal framework).
- **Scaled:** Boundless DAS, "Interpretability at Scale" **2305.08809** (Alpaca).
- **Tooling:** pyvene **2403.07809**.
- **Multi-attribute disentangling:** RAVEL / MDAS **2402.17700**.
- **Interventions as training:** ReFT **2404.03592** (LoReFT edits low-rank subspaces).

## Caveats that bear directly on "used"
1. **The interpretability illusion:** Makelov et al. **2311.17030** (EMPIRICAL + constructed examples).
   - A subspace patch can change the output by activating a **dormant parallel pathway**, without the subspace being the one the model uses. High IIA ≠ the model's own variable.
   - The authors' **reply** is **2401.12631**.
   - **Design implication:** pair any DAS subspace with (a) a rank- and energy-matched random-subspace control, as §203 rev 3 already does, and (b) a check that the patched subspace carries the variable in **unpatched** runs (a probe on clean activations, plus ablation).
2. **Non-linear (magnitude) codes in RNNs:** Csordás et al. **2408.10920** (EMPIRICAL).
   - Gated RNNs trained to repeat sequences store each position "with a particular order of magnitude, rather than a direction" ("onion" representations): "impossible to locate in distinct linear subspaces".
   - **Implication:** a negative linear-DAS / linear-probe result on our recurrent memory doesn't rule out a *used* value code. Our additive heads grow ∝ t (a magnitude clock), so magnitude-indexed codes are plausible. Test a magnitude / norm-band variable alongside linear subspaces.

## Latent-CoT / looped-model applications
- **2606.12689** (Coconut / CODI): causal interventions show latent use is **graded**, and the effect "concentrates in **low-rank directions**" whose step-to-step geometry grows structured. That's the closest precedent for searching *used* low-rank subspaces in latent-reasoning models.
- **2602.00449** (CODI on sequential polynomial iteration): logit lens, linear probes and **activation patching** localise intermediate-state representations and their routing. Short-horizon tasks form "bridge states"; deeper ones compress or lose the latent rollout.
- **2602.08783** (Coconut / CODI): step-wise do-interventions in an SCM view.
- **I found no DAS / interchange-intervention study on an *identical-input looped* or delta-rule-memory model** (DeltaNet-class). Our §167 DAS-EDIT is already ahead of the literature in setting.

## Suggested design for "find the USED value subspace"
- **Fit:** fit DAS on the pass-t memory (or the read), with v_t counterfactuals. Use disjoint fit / eval items, as §167 does.
- **Controls:** score IIA against matched random subspaces, and check that the found subspace is decodable on clean runs (Makelov).
- **Ablation test:** run a **subspace ablation of the DAS direction** (does accuracy drop?) as the use test, beside IIA.
- **Non-linear variant:** add a magnitude / non-linear variant (Csordás) for the additive heads before concluding "no used value code".
- **Expectation from §203's gate:** the D7 probe direction holds about 1% of read energy and is unused. DAS should find a *different*, higher-energy direction if a used code exists.
