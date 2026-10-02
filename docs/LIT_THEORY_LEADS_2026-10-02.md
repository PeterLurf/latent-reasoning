# Literature leads for Theory's open directions (2026-10-02)

Literature session, at the user's request via Audit. All arXiv IDs verified (Semantic Scholar batch, 2026-10-02).
Format: paper, then the finding, then a testable idea for us. ⚑ = already does what a registered § claims.

## a. Acquisition without short-step supervision (§76 ge3, §82: one block at chance)
- **Wies, Levine & Shashua 2204.02892.** Composite seq2seq problems unlearnable end-to-end become learnable when
  intermediate sub-task labels are given.
  - Our ge3 failure is the predicted case. *Idea:* treat short products as the sub-tasks and ask the minimal *fraction*
    of length-1/2 problems that unlocks acquisition (sweep 0.1%–10%). That turns "std vs ge3" into a threshold curve.
- **Abbe et al. 2306.16921 (curriculum on parity, mixed inputs) and 2301.13105 (degree curriculum).** Curriculum
  provably reduces sample complexity when easy instances are mixed in.
  - *Idea:* a ge3 curriculum that starts with length-3 problems drawn from a *small generator subset* (a lower-degree
    proxy) before the full alphabet. This tests whether the "foothold" is alphabet size, not length.
- **Abbe et al. 2302.11055 (leap complexity, saddle-to-saddle).** SGD stalls when the next step needs a large leap.
  - ge3 means a leap of 3 unknown transitions at once, from the uniform saddle. *Idea:* measure the gradient norm at
    uniform initialisation for ge3 vs std. A near-zero symmetric-saddle gradient would explain "one block".
- **Kim & Suzuki 2410.08633.** Transformers learn parity efficiently only with CoT or intermediate supervision.
  - The same lesson in a transformer. Cite it with Wies for the boundary claim.
- **Zhang et al. 1406.3824 (spectral initialisation + EM, provably optimal).** Spectral estimates escape EM's bad
  optima when used as initialisation.
  - *Idea:* build the spectral WFA from **ge3 data only** (Hankel prefixes ≥ 1, suffixes ≥ 2), then initialise the
    free table from it and run exact-marginal EM. If spectral-on-ge3 works, the uniform-saddle failure is an
    optimisation problem, not an information one.
- **Jin et al. 1609.00978.** Local maxima in GMM likelihoods are common and bad, and random initialisation fails with
  high probability. Background for "symmetric init = trap"; motivates symmetry-breaking init, e.g. random sparse
  tables.

## b. Composition in pretrained LMs (§76(B), §80: stores, doesn't compose)
- **Wang et al. 2405.15071 (grokked implicit reasoners).** Composition generalises in-distribution only after grokking,
  and fails OOD. The speed depends on the inferred/atomic data ratio; cross-layer parameter sharing helps.
  - *Idea:* a §80 data-scale sweep (k-op words 10³ → 10⁵, a fixed ratio of seen to unseen) plus a looped GPT-2
    variant, to find whether in-pass composition emerges.
- **Saunshi et al. 2502.17416.** Looped transformers match much deeper models on reasoning and simulate k-step CoT
  with k loops.
  - *Idea:* loop GPT-2's middle block k times for k-op words. This tests "depth, not data" for §80(b).
- **Lin et al. 2503.07604.** GPT-2 implicit multi-step reasoning emerges only on fixed-pattern data and is shortcut
  learning; it fails on variable patterns. ⚑ Consistent with §80 P0. Cite it as prior for "memorises words".
- **Press et al. 2210.03350 (compositionality gap).** LMs answer sub-questions but not their composition; the gap
  doesn't shrink with scale. The large-model analogue of §76(B).
- **Raventós et al. 2306.15063.** A task-diversity threshold separates memorisation from generalising in-context
  algorithms.
  - *Idea:* find the number of distinct training words at which GPT-2 switches from memorising to composing §80
    words. That would be a diversity threshold for in-pass composition.
- **Ye et al. 2407.20311 (Physics of LMs 2.1).** Small GPT-2-size models trained on synthetic GSM data learn
  generalisable multi-step reasoning, given enough diverse data. A data-scale counterpoint to §80.

## c. Confidence under length extrapolation; adaptive segmentation (§83, C5)
- **Ovadia et al. 1906.02530.** Calibration degrades steadily under dataset shift for every method tested.
  - *Idea:* report §83's ECE and first-error detection rate as a function of distance out of range, i.e. chunk length
    offered.
- **Kamath et al. 2006.09462.** Selective QA under domain shift: a calibrator trained on mixed in- and out-of-domain
  data beats raw softmax confidence.
  - *Idea:* a tiny calibrator (features: max prob, entropy, position in chunk) fitted on in-range data with *synthetic*
    out-of-range chunks.
- **CALM 2207.07061 + Learn-then-Test 2110.01052.** Risk-controlled confidence thresholds (already in LIT
  93597e4). Use LTT for τ.
- **Chung et al. 1609.01704 (HM-RNN) and Hwang et al. 2507.07955 (H-Net dynamic chunking).** Learned segment
  boundaries, end to end.
  - *Idea:* the trained-C5 upgrade of §83. A boundary head trained to fire where the next written state would fall
    outside the trained range, with targets from §83's own detections (self-distilled boundaries).
- **Guo 1706.04599; Hein 1812.05720.** Overconfidence in distribution, and arbitrarily high confidence far from data:
  the prior for §83's main risk.

## d. Paraphrase-invariant operation semantics (§79: digit reliance)
- **Todd et al. 2310.15213 (function vectors).** A small set of attention heads carries a compact "function vector" for
  a task that transfers across contexts and triggers the task zero-shot.
  - *Idea:* extract a function vector for each S3 op from GPT-2 across phrasings, and test (i) whether it is
    paraphrase-invariant and (ii) whether feeding it as the op feature raises §79 agreement above the
    standardised-hidden-state readout.
- **Hendel et al. 2310.15916 (task vectors).** In-context demos compress into a single task vector at a middle layer.
  - *Idea:* read op features at the task-vector layer, not the last layer, since digit identity may dominate late
    layers.
- **McCoy et al. 1902.01007 (HANS).** Models adopt lexical-overlap heuristics. Matches §79's digit reliance.
  - *Idea:* adversarial phrasings that keep the digits but change the op ("swap 1 and 3" vs "move 1 to 3") as the
    negative set.
- **Levy & Geva 2410.11781.** LMs represent numbers digit by digit in base 10. Explains why digit tokens dominate the
  features.
  - *Idea:* mask or average the digit-token positions and read the op from the verb span, to measure the
    verb-semantic share.
- **Kim, Linzen & Smolensky 2212.10769.** Lexical generalisation tracks pretraining exposure (in LIT a7cdf69).

## e. Latent vs text CoT on length generalisation in math (§78)
- **Hase et al. 2401.06751.** ⚑ The same GSM8K split (easy 2–3 vs hard 6–11 steps), 7–70B; easy data recovers about
  79% of the gain (LIT 56d524d). Any §78 "first step-count split" claim must be scoped to GPT-2 scale and the
  written-state arm.
- **Deng et al. 2405.14838 (ICoT-SI).** Internalising CoT step by step. GPT-2 small reaches 0.30 on GSM8K and 0.99 on
  9×9 multiplication, trained per size.
  - *Idea:* add an ICoT-SI arm to §78. It is the implicit-CoT baseline for short-to-long.
- **Lin et al. 2503.07604.** Implicit reasoning fails on variable-pattern problems. Predicts latent arms will fail
  short → long more than text.
- **Lee et al. 2307.03381 (teaching arithmetic).** Data format, i.e. the scratchpad/state, decides length
  generalisation for small transformers. Supports the §78 written-state design and bounding the state (LIT 56d524d
  pitfall 5).
- **Amos et al. 2602.08332 (Thinking States).** Chunked text-state injection length-generalises on parity and vars
  (Qwen 0.5B), not GSM8K. The nearest latent-vs-text-state prior.
- **Our own repro by step count (GSM8K_REPRO):** CODI 0.40/0.66/0.49/0.31/0.14/0.02 at 1–6 steps (n = 57 at 6). The
  ≥ 6 tail is thin, as flagged.
