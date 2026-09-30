<!-- Text extracted from the user's handoff document latent-branch-next-steps.docx (24 Sep 2026). Tables are flattened; equations listed at the end. -->

> **Status (2026-09-28): historical.** The user's 24 Sep handoff that framed Parts II–III (kept as the source of the
> Part II/III questions; code and PROTOCOL2.md cite it). Its questions were answered or superseded in Parts II–VI —
> see `report/findings.md` and `docs/RESEARCH_TIMELINE.md`. Current work: `docs/CURRENT_TASKS.md`.

# Next steps for latent branch research
# Research direction and experimental handoff — 24 September 2026
Recommendation. Explain why branch edits weaken as more alternatives are active. Begin with the existing small models and distinguish signal dilution, interference between branches, and reconstruction from cached information. Use a semantic test to establish what a branch edit changes before interpreting its geometric properties.
## The research question
Under what conditions can a model use several alternatives from a shared latent state, and what mechanism limits our ability to change one alternative without disrupting the others? The target is a predictive account of selective causal access in a controlled graph task. Relevance to nonverbal reasoning comes from studying hidden computational states; this setup does not establish scheming, private intentions, or general reasoning capacity.
## What the report gives us
The report describes a two-layer continuous-thought transformer, delayed target queries, three recurrent training seeds, and pause-token and no-latent controls. Its strongest starting point is a reproducible pattern of branch-targeted effects with width-dependent weakening. The mechanism is still open. [1]
Reported observation
Research implication
Targeted probability change is 0.236 at width 2, 0.124 at width 4, and 0.066 at width 8.
Width times effect is 0.472, 0.496, and 0.528. This derived near-constant pattern motivates a dilution hypothesis.
Removing a branch tends to increase scores on other active branches.
Changing the active count may change a shared gain or resource allocation. Fixed-width swaps can test this.
Pause tokens match much shallow performance; recurrence helps mainly at greater depth.
Analyze depth separately and replicate the relevant pause comparison before claiming recurrence-specific behavior.
## How the approaches fit together
Causal abstraction defines a meaningful branch intervention. Local controllability predicts the difficulty of a selective edit. Sparse coding and normalization offer candidate explanations for width effects. Cache recovery is a cross-cutting causal hypothesis. These approaches can support one project; they need not become three separate studies.
A useful result would show that a specified mechanism predicts unseen intervention outcomes and survives a targeted causal test. A failed semantic test is also informative: it would limit what the current branch-edit evidence establishes.
# Shared design and measurement
## Keep the experimental setting small
Reuse the reported architecture, training scaffold, checkpoints, and delayed-query format. Start with widths 2, 4, and 8 and retain depth-specific measurements. Fork all branch queries from the same saved pre-query state. Hold out structural graph families, including renamed or reordered copies. [1,2]
Let the relevant state consist of the chosen activation h and retained information C, including cached keys and values. Specify whether an edit changes h before it feeds the next latent position, cached values at an existing position, or both. These are different intervention interfaces. The report retains earlier cached states when changing the thought vector.
## Preflight before running new hypotheses
Reproduce one published effect table from the saved checkpoints. Check split hashes, donor generation, checkpoint identifiers, query averaging, uncertainty estimates, and the frozen intervention rules.
Record every retained and recomputed position. Compare common sites across seeds as well as the previously selected sites: two reported best sites are the START output before latent recurrence.
Audit donor widths. Removing from width 1 or inserting at width 8 can produce width 0 or 9; if those widths were excluded from training, treat them as a separate distribution-shift test.
Establish clean recipient and counterfactual performance at each width and depth. Publish all examples and a prespecified competence-stratified analysis; do not silently restrict to successful cases.
## Measures that answer different questions
Measure
Definition and use
Readability
Held-out branch-label decoding. Report uncertainty and shuffled-label controls. Probe accuracy alone does not demonstrate computational use.
Targeted effect
Signed change toward the intended branch status. Report probability and reachable-versus-unreachable logit contrasts, separated by depth.
Collateral effect
Mean, total, and worst absolute change on unchanged branches, plus signed changes. Exclude both intended branches in swap tests.
Counterfactual recovery
Compare the patch effect with the clean counterfactual effect. Normalize only when the denominator is adequately large; retain raw effects and do not clip ratios.
Semantic validity
Agreement with the predicted continuation from an edited branch state, including queries that distinguish competing computational explanations.
Treat the graph as the sampling unit because its forked queries are dependent. Bootstrap graph families within each seed and show seed-level results. Three seeds provide limited evidence about the population of training runs. Thresholds, primary contrasts, and any aggregation across depths should be fixed before fresh confirmation.
# Approach one Causal abstraction
Purpose. Determine whether an activation edit changes a branch state that subsequent computation uses. This is the semantic foundation of the project. Causal abstraction formalizes correspondence between an interpretable computation and neural interventions. [4]
## The candidate high level computation
Specify a state variable such as the current search frontier or all nodes reached so far. Choose one explicitly. Define how it should update using the graph, and what it means to insert or remove a branch at that stage. Do not assume one latent step equals one graph hop; establish and freeze any step alignment on discovery data.
Write φ for a restricted map from neural state S to the candidate search state, E for a neural edit, I for its proposed abstract counterpart, F for the neural continuation, and T for the abstract transition. The correspondence to test is:
In words, editing the neural state and continuing should match editing the abstract branch variable and applying the recipient graph transition. Restrict map complexity in advance. A decoder that simply recomputes reachability from the entire cached graph cannot establish that a frontier was represented in the chosen activation.
## The discriminating test
Construct matched donor and recipient graphs in which a candidate branch B has different downstream continuations. Transfer only the proposed branch-state component from the donor, retain the recipient graph context, and query endpoints where donor and recipient futures disagree.
The branch-state account predicts following the recipient continuation from the edited state.
Transferred answer information predicts donor endpoints; a generic reachability bias tends to move both diagnostic and irrelevant queries together.
Repeat with another recipient continuation and with two independent branch edits. Evaluate whether composed edits produce the abstractly predicted outcomes.
Check clean competence on both continuations first. Use matched node identities and stage-specific donor states. Include identity patches, wrong-branch donors, unchanged-branch queries, and natural counterfactual runs. A donor can carry future information even when the patch is small; this is precisely what the conflicting-continuation test addresses. [5]
## Decision and contribution
Advance when restricted interventions predict recipient-specific continuations on held-out families, at more than one branch. Revise when only final-answer steering works or donor futures dominate. Failure limits the proposed state interpretation; it does not prove the model lacks an alternative representation.
The candidate contribution is an operational test of separately usable alternatives. High probe accuracy, a diagonal effect matrix, or commuting linear edits alone is insufficient. Commuting projections can be built without representing the intended computation.
# Approach two Local controllability
Purpose. Predict the cost of changing one branch while preserving the others under a specified intervention interface. This applies established constrained least-norm reasoning to the model’s actual downstream responses. The equations are standard linear algebra; empirical predictive value is the research target. [6]
## A local model of intervention effects
Let R be the vector of branch-query logit contrasts obtained by forking the saved state. Hold C fixed and restrict edits to δ = Uu, where U has orthonormal columns and is fixed using discovery data. Let J be the derivative of R with respect to h, and A = JU. For small edits:
Choose a desired score change d, with zero entries for unchanged branches. A swap has two nonzero entries. The idealized problem is:
For a single target i, project its row of A onto the nullspace of the other rows. Call this residual direction p. If p is zero, a nonzero perfectly selective edit is impossible in this linearized interface. Otherwise, a target change of magnitude α has minimum squared norm:
If branches respond to almost the same directions, p becomes small and selective control becomes costly. Small raw gradients can also raise this cost through dilution. Measure gradient magnitude and residual-to-total gradient ratio separately so that weaker response is not mistaken for increased alignment.
## Implementation and falsifiable prediction
Estimate query gradients one at a time on a modest, stratified subset of graphs. Start in a frozen low-rank subspace; avoid storing every full activation or Jacobian.
Use small dose sweeps to establish where the linear approximation predicts finite effects. Replace exact zero collateral with a prespecified tolerance for practical optimization.
Predict targeted and collateral effects of independently specified donor edits on held-out graphs. Compare with width, donor norm, gradient norm, clean confidence, and probe accuracy as simpler predictors.
Use a fixed score target and a separate clean-counterfactual-relative target. Document activation scaling; raw Euclidean cost is not invariant to arbitrary coordinate changes.
## Decision and limits
Advance if geometry improves held-out prediction beyond these baselines and remains valid for semantically tested, finite edits. Deprioritize it if only infinitesimal edits work or predictions fail at natural donor magnitudes. Constructing an edit from a Jacobian and verifying its tiny first-order effect is a sanity check, not the main result.
A control-rank limit concerns accessible output changes at this interface. It is not a bound on stored bits or the number of thoughts. Large predicted costs can push interventions beyond the local approximation and beyond naturally occurring states.
# Approach three Sparse coding and normalization
Purpose. Explain the width effect using a tractable representation model. Sparse-feature superposition motivates studying overlapping codes, but it does not establish that the trained model uses an additive branch code. That assumption must be tested. [7]
## Separate signal size from interference
Consider a candidate code with branch vectors v, binary active indicators z, a width-dependent gain g, and noise ξ. All symbols refer to a fixed stage and a declared context family:
The sum allows two distinct explanations. Gain may shrink as width W grows even if branch directions remain separable. Alternatively, overlapping directions may produce crosstalk that makes branch information harder to decode or edit selectively. Both mechanisms can coexist.
For unit code vectors, at most W active branches, gain one, maximum pairwise absolute correlation μ, and projected noise bounded by η, a matched-filter decoder has a separating threshold whenever:
This is an elementary sufficient condition from bounding cross-terms. Its failure does not prove information loss: a different decoder may succeed. A prescribed normalized gain changes the signal margin; actual attention and LayerNorm denominators must be measured rather than equated to a toy formula.
## Tests that distinguish the explanations
Gain test. Compare natural donor magnitudes, dose-response curves, clean counterfactual effects, and branch signal amplitudes across widths. Fit simple alternatives such as constant gain, inverse square-root gain, and inverse-width gain using discovery data. Compare held-out predictive error with uncertainty; the three aggregate width points alone cannot select a mechanism.
Fixed-width swap. Exchange one active and one inactive branch while preserving total width. If collateral falls relative to removal or insertion, the changing active count may explain part of the effect. Control edit norm and clean counterfactual size: a swap changes two branch statuses.
Overlap test. Estimate a dictionary from independent examples, measure reconstruction quality and context dependence, and ask whether overlap predicts held-out collateral beyond width and amplitude. Fit encoding geometry separately from downstream response geometry; they need not coincide.
## From an explanatory fit to a causal mechanism
If a shared attention budget becomes the leading account, identify a candidate head or path on discovery data. Patch its branch-specific, value-weighted contribution or a precisely defined normalization component, with matched unrelated-path controls. Test predicted effects on unchanged branches. Attention weights falling with width can arise from averaging and are not a causal demonstration by themselves.
Advance when the model predicts new dose or swap outcomes and a targeted mechanism intervention supports it. Revise when the code changes strongly with context, additive reconstruction is poor, or causal tests contradict the fitted explanation. A normalization mechanism could be a valuable result even if no capacity loss occurs.
# First experiments Calibration and branch swaps
## Experiment one Calibrate the width effect
Start with existing checkpoints. For each width, depth, seed, and declared site, save the clean recipient response, clean donor response, natural activation difference, projected difference, and patched response. Compare the full-vector, branch-pattern, probe-filter, shared-PCA, and matched random interventions used in the report.
The report’s probe-filter projection contains much less of the donor difference than the pattern projection. A tiny projected intervention can have a tiny effect without showing that its direction is intrinsically ineffective. Forward activation patterns and decoding weights also have different interpretations. [1, 8]
Report both natural-dose results and magnitude-controlled results. Prespecify a modest multiplier grid, for example 0, 0.25, 0.5, 1, and 1.5, on discovery data. Treat the upper doses as stress tests if they leave natural activation ranges.
Compare equal-sized perturbations only where the magnitude ranges overlap. Do not enlarge a nearly zero projection by orders of magnitude and then describe it as a natural branch edit.
Plot targeted change against collateral at each dose, alongside donor norm, clean counterfactual gap, and starting confidence. Keep probability effects and logit effects separate.
Interpretation. If modest, supported rescaling largely restores selectivity at high width, pursue dilution. If collateral remains after magnitude and competence controls, investigate interference or cache recovery. A failure to restore selectivity under this particular edit family does not prove that no other selective edit exists.
## Experiment two Hold active width fixed
Choose reachable branch A and shape-matched unreachable branch B. In the donor graph, move A’s hub edge to the decoy and B’s hub edge to the start. Preserve node count, edge count, branch shapes, ordering conventions, and width. Begin at width 4, then check widths 2 and 8.
Patch the candidate state and fork queries for A, B, and the unchanged branches. Compare with A-only removal and B-only insertion. Add a same-status edge change as a structural control. Measure mean, summed, worst, and signed collateral on unchanged branches; do not count either A or B as collateral.
Outcome
Next inference or test
Both intended statuses change and collateral shrinks.
Active-count or shared-gain explanations gain support. Run the semantic continuation test, then a candidate budget intervention.
Collateral persists after magnitude controls.
A count-only explanation is insufficient. Test response alignment and route-specific cache recovery.
Only one branch changes or donor endpoints dominate.
The intervention may not implement the proposed swap. Resolve its semantics before making capacity claims.
Deliverable. One dose-response figure, one swap-versus-removal effect matrix, and a short decision note naming which hypotheses remain consistent with the results. These tests should determine the next investment of effort.
# Cache recovery as a causal hypothesis
Question. Does retained information reconstruct an edited branch? The report observes recovery in some seeds, but the cache explanation requires an intervention on the suspected route. The thought vector is only part of the computational state. [1]
## A matched factorial experiment
Locate a candidate source of the original branch signal on discovery graphs, such as selected graph-token keys or values, or earlier latent cache entries. Freeze the locations and patch rule. Use matched counterfactual donor values rather than zeroing the whole cache.
Condition
Thought state
Candidate cache route
Baseline
Original
Original
Thought only
Counterfactual edit
Original
Cache only
Original
Counterfactual edit
Combined
Counterfactual edit
Counterfactual edit
Continue all runs using the same recomputation rule. Track the branch signal immediately after the intervention and at later latent steps, then evaluate the final forked queries. Use independent diagnostic probes where possible, but require behavioral evidence as well as probe trajectories.
## Evidence that would support recovery
The thought-only intervention should first reduce the branch signal, followed by its return. Editing the proposed cache route should prevent that return and preserve the intended final effect, while identity patches and matched unrelated-cache patches should not. Other branches and clean task performance should remain sufficiently intact to rule out generic damage.
Compare all four conditions, including the cache-only main effect and their interaction on a declared score scale. Probability nonlinearities can create interactions, so a larger combined effect alone is insufficient. Patching graph-token information can also change the effective input problem directly; timing and pathway specificity are needed to support reconstruction of an edited state.
## If the initial cache test is inconclusive
Separate graph-token routes from earlier latent routes and distinguish keys from values. Expand access on discovery data, keeping thought-only edits as the reference. Freeze the candidate routes before confirmation.
If a recovery path is supported, incorporate it into the full-state causal model and repeat semantic continuation tests. Otherwise, report the limit of the tested interface. Distributed branch computation can be a substantive result even when one vector is not independently editable.
## Why an information capacity bound is secondary
If the cache contains the graph and start node, that side information determines reachability. Information theory alone then gives no useful storage lower bound for the thought vector. A memory or access bottleneck needs a separately specified experiment. Defer that extension until the current mechanism is understood. [3]
# Execution plan and research decisions
Use a six to eight week core plan, with the remaining portion of a one to three month project reserved for robustness or writing. These are planning estimates, conditional on reproducible checkpoints and an accessible implementation. Keep the existing small architecture; benchmark a short run before committing a GPU budget.
Stage
Work and reason
Required output
Week 1
Reproduce the report and calibrate effects. Establish whether dose, confidence, or depth explains the width trend.
Verified table; dose-response plots; frozen definitions.
Week 2
Run fixed-width swaps and construct semantic continuation cases. Separate active-count effects from branch identity.
Swap matrix; diagnostic graph set; initial hypothesis decision.
Weeks 3 to 4
Establish semantic validity and test candidate cache routes. Resolve what the intervention changes and where recovery originates.
Continuation results; four-condition recovery comparison.
Weeks 5 to 6
Fit the most supported quantitative account. Test controllability or encoding predictions against simple baselines.
Held-out prediction results; one targeted causal mechanism test.
Weeks 7 to 8
Freeze the final protocol; use fresh graphs and additional matched controls; write the result.
Confirmation figures; reproducible release; claim and limitations.
## Choose the contribution from the evidence
If effect loss largely follows magnitude and active-count changes, build a mechanistic account of dilution or normalization and test the responsible path.
If semantic edits remain valid but their collateral or required norm grows with response alignment, test whether control geometry predicts that loss of selectivity beyond amplitude.
If a specific cache route restores edited branches, explain reconstruction and how intervention access changes apparent branch editability.
If interventions fail the semantic test, document the failure of the branch-state interpretation. This can be useful evidence about the limits of current assays.
These outcomes are not mutually exclusive. Prefer the smallest combination that predicts held-out outcomes. Avoid choosing the most interesting narrative before the discriminating tests.
## Scope control and stopping rules
Reuse the three recurrent seeds initially. Add matched pause-model seeds for the decisive result, using an appropriate activation or cache interface rather than pretending pause models have recurrent thought vectors. Focus on depths where each model is competent, while retaining the full depth breakdown.
Defer larger models, merging branches, new architectures, and bottleneck training until one mechanism is supported. If semantic validation remains unsuccessful after a bounded discovery phase, narrow the claim to intervention behavior. If geometry adds no predictive value, retain that negative result and stop expanding it.
Minimum valuable result. A reproduced phenomenon, a test that separates at least two plausible explanations, and a fresh confirmation that supports a narrowly stated computational claim or falsifies the proposed interpretation.
# Confirmation checklist and references
## Freeze these choices before confirmation
Graph-family split and donor rules; model checkpoints; intervention sites and cache boundaries; query depths; restricted maps and subspaces; dose ranges and magnitude metrics.
Primary outcome and contrasts; collateral definition; competence strata; counterfactual denominator threshold; graph-level uncertainty; treatment of seed variation and failed runs.
Expected results under dilution, interference, and recovery; baseline predictors; criteria for semantic agreement; limits on further discovery. Select numerical thresholds on discovery data and justify them.
Keep the original report’s selectivity threshold available for comparison, but do not let one threshold replace the dose-versus-collateral curves. Use a fresh confirmation split for newly selected tests. Archive code version, configurations, checkpoints, split hashes, per-graph measurements, and the analysis that rebuilds each figure.
## References and source material
[1] Branch edits in continuous thought. Report dated 23 September 2026. Supplied as latent_branch_report.pdf; numerical findings and model details are attributed to this report.
[2] Mechanistic interpretability research handbook. Supplied as mechanistic-interpretability-research-handbook.md. Research design guidance on precise claims, competing explanations, and discriminating tests.
[3] Theoretical frameworks for readable and usable alternatives in latent reasoning. Research synthesis dated 23 September 2026. Earlier project memo; contains derivations and the side-information discussion. Proposed applications are hypotheses, not established results.
[4] Geiger and colleagues. Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability. JMLR 26, 2025. Formal foundation for correspondence between high-level and neural interventions. Source
[5] Shih, Winnicki, and Darve. When Does Activation Steering Change What a Model Computes From? arXiv:2606.29522v2, revised 6 September 2026. Motivates state-use tests and natural-scale intervention controls; earlier versions used a different title. Source
[6] Boyd and Lall. Controllability and state transfer. Stanford EE263 lecture notes. Background on reachable sets and minimum-norm control. The branch-specific local formulation here is a proposed application. Source
[7] Elhage and colleagues. Toy Models of Superposition. 2022. Background on sparse features, shared dimensions, and interference; does not establish an additive branch code in this experiment. Source
[8] Haufe and colleagues. On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage 87, 2014. Distinguishes decoding weights from forward activation patterns. Source
[9] Training Large Language Models to Reason in a Continuous Latent Space. 2024. Coconut architecture background; use it to distinguish the general method from the report’s small trained system. Source
The standard algebra and sufficient bounds provide tools for testing the model. The prospective contribution is their validated connection to branch computation, not a claim to have introduced these mathematical frameworks.

[tables: 5 | rows: 50 ]

## Equations (as written in the document)


## Candidate PROTOCOL8 (2026-09-29, from the Q result and the theory session)

**Multi-step latent work in a pretrained model.** Addenda P and Q show Coconut GPT-2 computes ProsQA in one pass: no
training-depth wall, extra thoughts never help, and its limit is following chains added past the answer options. The
toy model's slack / schedule-end account (THEORY.md) applies only to models whose thoughts carry a step schedule. The
next question is whether a pretrained model can be made to do genuinely multi-step latent work: train on problems
deeper than one pass resolves (ProsQA-style graphs whose answer requires following long added chains, where P shows
one-pass accuracy is at chance from 4 added hops), with Coconut's recipe vs anytime (minimum-progress) supervision
(theory/train_arch.py AnytimeMixin, to be adapted to the Coconut codebase). Pre-register before launch; ~$10–20 of
cloud credit. PROTOCOL7 T12 (Q-deep) was contingent on Q1 and does not run.

**Candidate outcomes (2026-09-29 14:35, from the theory session, THEORY §13 addendum 9e02cd6; for the pre-registration).**
The theory session re-read its saved GPT-2 CODI GSM8K accuracy by gold step count s:
- At s = 3 and s = 4, two latents score below zero latents (0.15 vs 0.24, and 0.11 vs 0.19).
- The gain from 6 latents over none is largest at s = 2–3 and gone by s ≥ 6.
- So a truncated chain is worse than none: the non-anytime signature.

Suggested outcomes:
- (1) Per difficulty stratum, accuracy is non-decreasing in the number of inference latents L = 0..K′: predicted for
  anytime arms, violated for Coconut arms.
- (2) Test-time gains at L > K occur only above the one-pass capacity c.

Constraint from our own data: the Coconut GPT-2 ProsQA model's budget sweep is flat, 0.976 at K′ = 0 and 0.974 for
K′ = 2..12 (`prosqa/res/kprime_test_e50.json`). It is one-pass, so it has no truncation penalty. Outcome (1) therefore
passes trivially below c. It must be scored only on strata above c, with a tolerance, and together with a non-zero
latent benefit (acc at K minus acc at 0).

Anytime training supervises intermediate readouts, so monotonicity inside the trained range is partly built in. The
informative test is at depths and budgets beyond training. CODI's dip at L = 2 may also be a readout
distribution shift (it never saw a 2-latent context). A Coconut arm trained with random budgets would separate that
from dynamics keyed to the budget.

**Third arm, agreed with the theory session (2026-09-29 ~14:40): Coconut with random tight training budgets.**
The theory session calls this the strongest single addition: it tests Prop. 4 (random tight budgets and anytime
supervision are one identification mechanism), and it separates "keyed to the budget" from "never saw that length".
For a fair test:
- (a) The budget support reaches at or below the smallest step demand at each curriculum stage, e.g. K ~ U{1..K_max} per
  batch.
- (b) K is drawn independently of the example (not K = hops; Kuo's K = n failed; THEORY §22.7 refinement (b)).

Predictions:
- Random-tight ≈ anytime beyond training, and both beat fixed-K Coconut.
- Random-tight removes the L2 dip.
- Random-tight failing where anytime succeeds would refute Prop. 4's equivalence claim.

If cost forces two arms, keep Coconut vs anytime. Cost of three arms: about 1.5× the two-arm estimate, ~$15–30; the
user decides.

**Literature input for PROTOCOL8 (2026-09-29 ~19:40; Literature session, `docs/LIT_REQUESTS_2026-09-29.md` Part A):**
- **Anytime arm is new in form.** No prior method uses a depth-masked per-step answer target (a lower bound on progress,
  early arrival never penalised). Closest:
  - CLRS hints / NEGA supervise the exact reached set, which pins the speed to 1 hop per step.
  - RELAY, SIM-CoT and PMPS align content per step.
  - Fan 2409.15647 puts a loss only at T(n).
  - Yang 2311.12424 puts a full-answer loss over late iterations.
  - PonderNet, TRM and multi-exit nets put the full target at every step.

  Wording: "per-step supervision of the answer with a depth-masked target: minimum progress without fixed speed."
- **Random-budget arm must be example-independent, with minimum 1** (as Bansal 2202.05826, Huginn). Coconut's
  `uniform_prob` caps thoughts by the example's own step count, and Kuo's K = n + Δ is centred on n, so neither is the
  arm we want. Use K ~ U{1..K_max} per batch.
- **Caveat on Prop. 4(b).** Fan's budget is deterministic in n, yet extrapolates, because the test budget is also T(n).
  Kuo's K = n failure may partly be a clipped test budget (K ≤ 20 tested at n = 20–60). Theory has been told.
- **Checkpoints.** `bmarti44/coconut-curriculum-checkpoints` (MIT, GPT-2 ProsQA; model card: 7-/8-hop out-of-distribution
  Coconut 66.0/67.5 vs pause 75.4/75.1). We already evaluated one checkpoint of it (`prosqa/res/ext_bmarti44_coconut_e49.json`).
  CODI and SIM-CoT GPT-2 checkpoints are GSM8K only.
- **Deeper datasets for later generality.** CLUTRR (2–10 hops), StepGame (k ≤ 10), the Saparov 2412.04703 generator.

**Random-tight arm, exact rule (theory session, 2026-09-29 ~19:45; THEORY §22.7, 340e558), to register as is:**
- **Per stage:** at every latent curriculum stage s ≥ 1, draw K ~ U{1..K_s} per example, independent of the example.
  K_s = c·s is the stage's nominal latent count, not capped by the example's step count n. The final stage uses
  K ~ U{1..K_max}.
- **Format:** Coconut's stage-K format. K latents replace the first K steps, explicit CoT resumes at step K + 1, and
  if K ≥ n the answer follows the latents directly. The loss is unchanged.
- **Implementation:** in `get_cot_latent_dataset`, redrawn every epoch because the dataset is rebuilt per epoch.
- **Prediction:** random-tight ≈ anytime > fixed-K Coconut on beyond-training outcomes.
- **Optional sub-arm, if seeds allow:** random-tight at the final stage only, to test whether circuit selection in
  Coconut happens in the early stages.

**THEORY Pilots D, F and H scored (2026-09-30 ~00:10, commit 440d6ae), and what they mean for PROTOCOL8:**
- **Jitter off.** Pilot F: jitter alone broke learnability (1/8 competent). Jitter + anytime was learnable (6/8) but
  gave no d8 gain over anytime (median 0.568 vs 0.584). `patch_p8`'s jitter option stays unused in the confirmatory
  arms.
- **Anytime helps mostly at the first distance past training.** Pilot D, PD1 supported (anytime 8/10 extrapolate vs
  control 3/9, Fisher p 0.055). Mechanism: anytime removes the in-range brake, so the pointer moves one hop per step
  in 6/10 runs vs 0/9. But the pointer falls back within 1–3 steps after K. The gain is at d7, while d8–10 barely
  move.
- **Consequence for PROTOCOL8.** Score beyond-training accuracy separately at the first strata past training (m = 7)
  and deeper (m = 8), not pooled only. Expect anytime/random-tight effects at m = 7 more than at m = 8.
- **Pilot H.** Post-hoc anytime fine-tuning of trained failures does not repair them (1/18). The PROTOCOL8 rules
  already apply from the first latent stage, which fits.

**THEORY Pilot E final (2026-09-30 ~00:30, commit 55c0bf4; §20 Result, §35 mechanism).**
- **Verdicts.** PE1 "depth wall supported" by the registered rule: 0/12 competent anytime runs in E8 ∪ E3 cross beyond
  depth. PE2 is reversed at K = 8. PE3 and PE4 are not met.
- **Mechanism (§35).** The wall is a learned fall-back, not a representational limit.
  - Pushing the pointer 2 hops ahead moves it 2 hops deeper, and it falls back at the same step (8/17 models pass
    depth 8).
  - Re-feeding an older thought delays the fall-back by one step.
  - So the fall-back is keyed to an iteration count carried inside the thought.
- **Prediction to register for PROTOCOL8** (Theory's line): anytime or tight-budget training at depth n gains at
  m = n + 1 and gains nothing at m ≥ n + 2 for any K′. Deeper answers need the count reset: restarts from written
  checkpoints (§33), or count-clamping at inference (§36, pending).
- **What this means for the plan after PROTOCOL8.** The checkpointed design is the route to C2 beyond n + 1. It needs a
  readable entity checkpoint in GPT-2 (Patchscopes-style decode), which is also the first C7 test. Pilot K (toy
  random-tight, ~02:00) will say whether random-tight behaves like anytime.
- **Correction (00:35):** "m = 7" above should read answer depth. Trained depth = orig (3–4) + m (0–6) ≤ 10, so m = 7
  mixes depth 10 (in range) with depth 11. PROTOCOL8's primary beyond-training contrast is at answer depth d = 11
  (`by_path_len` in long_eval2), with d ≥ 12 as the "no gain" prediction stratum. m strata stay descriptive.
- **Refined prediction (Theory, 00:40):** key it to answer depth. The fall-back trips after as many latent iterations
  as the deepest trained problem needed, so the right key is depth, not m. The toy's "+1" is the answer pass's own hop.
  For GPT-2, let c1 = the answer pass's hop capacity, measured from Phase 0 Coconut at K′ = 0 as the largest answer
  depth with accuracy ≥ 0.75 (c1 ≥ 1).
  - Gain is possible up to d = D_train + c1 (D_train = 10).
  - Firm prediction: no gain at d ≥ D_train + c1 + 1 for any K′.
  - If c1 cannot be measured, use c1 = 1 (the toy value) and treat d = 12 .. 10 + c_model as a grey zone, not a test.
  - The primary contrast stays at d = 11.

**THEORY Pilot L (§32, 7a6b330):** distilling the model's own re-anchored inference makes 19/20 trained failures
extrapolate (fine-tuning control 0/20; own single-episode labels 0/20). **Planned follow-up (PROTOCOL8b, own
registration):** re-anchoring plus distillation on PROTOCOL8's coco-arm checkpoints (~Thu 00:00). Anchor decode:
attention-based and/or text (Patchscopes; also the first C7 test). Recipe to come from Theory.
- **Pilot K (§31, b19711a; corrected per Audit):** random-tight is *consistent with* anytime on the toy (only 4 competent runs; PK1 unresolved; PK3 split, 2/4 still show the fixed-K brake), not established (4/4 competent RT runs extrapolate at d7; median d8
  0.571 vs 0.584; learnability lower, 4/8). PROTOCOL8 registers the same prediction for both arms, with lower
  confidence for random-tight.
- **PROTOCOL8b recipe** (Theory; toy code `theory/distill_reanchor.py`, `theory/exp12_reanchor.py`):
  - Teacher: K thoughts, then the top-3 statements by sweep-head attention at thought K become anchors. Each anchor
    gets a fresh K-thought episode from its target entity, and the answer is the max logit over the episodes.
  - Student: fine-tune at lr ≈ train/5 for 3000 steps. Each step pairs a normal batch with a deep batch of K′ = 2K,
    with CE on the teacher's answer.
  - Arms: T vs self (the model's own K′ = 2K answer) vs GT.
  - Decode: attention (primary) and text (secondary, the C7 test, with an input-only control).
  - Toy bar: T 19/20 extrapolate, self 0/20.

**THEORY §41-2 (2026-09-30 ~12:10): reach grows with the deepest training depth** (toy, true labels, tight budgets).
Train to depth 9 and the fade sets in by about 15; depth 12, by about 22; depth 18, flat at ~0.72 to 23 and 0.88 at
25–30. **For PROTOCOL8b and later:** the deepest problems in the training mix may matter more than the loss form, so
consider a deeper tight-budget training arm (deep ProsQA beyond m = 6, which needs a larger name pool) next to
re-anchor distillation.
