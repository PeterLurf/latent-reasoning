# STEP-TOLD draft (ST_draft.md): novelty N1–N3 and priors for step-cued reasoning (2026-10-24)

Written 2026-10-24 by the Literature session, for Theory.
- **Method:** one web search, plus our own notes. IDs verified on their abs pages.

## Prior: step-cued / decomposition reasoning
Text side (the question is decomposed and each step is asked explicitly):
- **Least-to-Most** (Zhou et al. 2205.10625): decompose into sub-questions, solve in order.
- **Self-ask** (Press et al. 2210.03350): the model asks and answers explicit follow-up questions; also measures the "compositionality gap".
- **Successive Prompting** (Dua et al. 2212.04092).
- **Planning tokens** (Wang et al. 2310.05707): plan tokens inserted before each CoT step.

Latent side:
- Coconut, CODI, SIM-CoT, PLaT 2601.21358 (latent "planning", decoded question-blind) and Thinking States 2602.08332 (supervised state targets).
- **None conditions each latent pass on a cue naming the target quantity.**
- One search (2026-10-24) found no latent loop driven by per-step sub-question cues or a given plan.

## Verdicts
| # | verdict | basis |
|---|---|---|
| N1 (each think pass is told by name which quantity to compute, never its value; CUED-DIRECT credits the depth gain to the loop, not the plan) | **OK, narrow wording** | Step cues are the text-side idea of least-to-most / self-ask / successive prompting / planning tokens, here moved into a latent loop. Not found latent-side. Wording: "a latent-loop analogue of decomposition prompting (2205.10625, 2210.03350)", with the claim scoped to C1-TOLD. CUED-DIRECT, which separates the plan's effect from the loop's, is the key design point. |
| N2 (diagnosis: the looped arm fails because no pass is asked for the intermediate; v_1 at chance at the write source while the given value decodes at 0.988) | **NARROW** | Kin: Biran 2406.12775 ("Hopping Too Late": the bridge entity is resolved too late in the layers for the second hop); Yang 2402.16837 (latent multi-hop is partial); Press 2210.03350 (sub-answers known but not composed). The specific probe pattern in our model is our own observation. |
| N3 (MEAN-AT-t plus same-skeleton swaps to separate serial use of a told state from per-pass recompute) | **NARROW** | Mean ablation of latents is standard (Dilgren 2604.04902); counterfactual swaps follow SCIT 2608.27265 / our §221. The per-pass placement with the cue held fixed is a design refinement. |
