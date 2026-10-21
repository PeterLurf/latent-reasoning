# Is "self-training keeps outputs but erodes an internal long-horizon mechanism" reported? (Note 216-13 INTERNAL-ONLY; 2026-10-21)

Written 2026-10-21 by the Literature session, for Theory and Audit.
- **Method:** grepped our own notes first, then ran one search subagent.
- **IDs:** I re-checked every new ID below on its abs page.
- **Numbers:** from the subagent's reading of the HTML. Abstract-only ones are marked "(abs)".

## Verdict: NARROW
- **"Behaviour kept while internal computation changes" is reported** [empirics]:
  - **Haskins & Adams 2505.10822** (distillation GPT-2 → DistilGPT2: broad behaviour preserved, internal circuit restructured, teacher components dropped; abs);
  - **Wang et al. 2502.11812** (math fine-tuning: circuit nodes stay similar while edges change substantially);
  - **Rojas Nunez 2605.28860** (Qwen2.5-3B: SFT keeps ~52 % of the base circuit vs ~68 % for RL, and RL keeps more "despite a larger output-space KL");
  - **Chhabra 2503.01896** (toxic fine-tuning corrupts specific IOI-circuit components; retraining restores them).
- **The more common finding is the reverse** (outputs forget, internals persist): Zheng 2501.13453 "spurious forgetting" (Task-0 ≈ 100 → 10 % within 150 steps while recoverable knowledge stays ≈ 96 %), Jain 2311.12786, Prakash 2402.14811.
- **Not found:** self-training on the model's own filtered labels eroding an internal **memory / long-horizon** mechanism while base-KL on the same items equals a gold-only fine-tune's.
  - The output-level self-generated-data results (Chen 2510.18874 on-policy; Marek 2605.26097 self-replay) would predict no forgetting here. They measure behaviour only.
- **Suggested wording:** *"Consistent with reports that fine-tuning or distillation can restructure internal circuits while preserving behaviour (2505.10822, 2502.11812, 2605.28860), round-1 self-training eroded the memory channel's long-horizon stability at old depths with base-KL equal to a gold-only fine-tune (one seed, one pair). We found no prior report of this for self-generated labels or for a memory mechanism."*
- **Correction to my 10-20 note:** LIT_MUX_FORGETTING said "mechanism loss with outputs kept has no prior". That holds only for self-training and for memory. The general phenomenon has the priors above. Note 216-13's "Literature found no prior work on this" should be narrowed to match.

## RL's Razor (Shenfeld 2509.04259)
- Rojas Nunez is the one report I found of output KL failing to track circuit retention, and it fails in the opposite direction (higher KL, more circuit kept).
- **Caveat before claiming a violation:** Shenfeld's predictor is KL on the **new-task** distribution. Ours is on old-depth items (n 3–6). The matching check is KL(D3-T ‖ FT) on the self items' inputs (n 9–12), FT-SELF vs FT-CTRL.
- Also, KL there is near zero and accuracy / argmax agreement are 1.000 for both models, so the output metric sits at ceiling. Equal KL is weak evidence of equal drift beyond what argmax already shows.

## Levers with evidence for preserving internals
1. **Freeze the lower layers or the mechanism's parameters** (Zheng 2501.13453) [empirics].
   - Recipe: Biography task, 7–8 bottom layers + embeddings frozen, lr 5e-6. Task-0 accuracy 11.18 → 44.22 % with Task-1 at 99.93 %.
   - Safety: 6 layers frozen, jailbreak 99.80 → 1.15 %.
   - For us: freeze the version-bank write / read parameters in the self round.
2. **Feature-space distance to the base, LDIFS** (Mukhoti 2308.13320, TMLR) [empirics; CLIP vision].
   - Recipe: λ · mean ‖Φ_t(x) − Φ_0(x)‖² over features at evenly spaced layers on the fine-tuning data; λ by cross-validation.
   - Results: mean Δ linear-probe accuracy over 10 tasks −2.59 (CE) vs +0.29 (L2-SP 1802.01483) vs +0.86 (LDIFS), at equal target accuracy.
   - For us: apply it to the slot-history states at K′ 24 on old-depth items.
   - Caveat (Goodhart memory): a one-sided closeness target on internal states needs the derivation's gaming analysis.
3. **Update only the value matrix** (Lee, Sohn & Lee 2602.23197, ICML 2026) [theorem on linear attention + empirics, abs].
   - Fine-tuning all attention parameters can harm in-context learning; restricting updates to W_V preserves it.
   - For us: the delta-rule bank is linear attention, so keep key / query addressing fixed in the self round. No numbers verified.
4. **216-9's lead** (plain small-n K′ 24 rollouts mixed in) has no direct prior. It is the closest in spirit to replay at the measured horizon.

**Evidence status:** one seed and one FT pair (216-13 says "an instance"). A "new" claim needs a second seed and the n 9–12 KL check.
