# Literature answers to agent requests (2026-09-29)

From the Literature session. Part A answers the Experiments session (PROTOCOL8 prep, four items), Part B the Audit
session (three items). arXiv IDs were checked against the arXiv API. Training details marked "(full text)" were read from
the paper's HTML or ar5iv; the others are from abstracts or model cards, as noted.

---

# Part A — Experiments (PROTOCOL8)

## A1. Is "anytime / minimum-progress supervision" new?

Our loss: after latent step k, supervise only answers whose required depth is ≤ k + C, mask deeper reachable nodes
(never label them negative), and apply the full loss at K. The ingredient that sets it apart is **masked look-ahead: a
lower bound on progress, with early arrival never penalised**. No prior work below has it.

| Prior method | What is supervised at intermediate steps | Closest to ours? | Difference |
|---|---|---|---|
| **CLRS hints** 2205.15659; **NEGA** Veličković et al. 1910.10593 | The algorithm's exact intermediate *state* per step: for BFS, which nodes are reached at step k (`reach_h`) | **Closest in form** | Exact labels: deeper nodes are *negative* at step k, so speed is pinned to one hop per step (early arrival penalised). Ours supervises the answer head, not the state, and is a lower bound only. |
| **RELAY** 2502.08482 | Each loop iteration aligned to an explicit CoT step | Close in spirit | Supervises the *content* of each step (a target sequence), not a masked answer. |
| **SIM-CoT** 2509.20317; **PMPS** 2609.09928 | Each latent decoded (auxiliary decoder) to its explicit reasoning step; PMPS via prototypes | Per-latent supervision | Content targets from a teacher CoT; no answer-at-every-step, no depth mask. |
| **CODI** 2502.21074 | Hidden state at one token aligned to a CoT teacher (self-distillation) | Far | One alignment point, not per step. |
| **Fan et al.** 2409.15647 (full text) | Loss **only at the final iteration** T(n), where T(n) = ground-truth step count for the example | Far | No intermediate loss; the budget is matched per example (an exact contract, not a lower bound). |
| **Yang et al.** 2311.12424 (full text) | Loss over a *window* of iterations [b − T, b], with b scheduled upwards during training | Moderate | Full answer at several late iterations; nothing masked by depth. |
| **PonderNet** 2107.05407; ACT 1603.08983 | Full answer at every step, weighted by halting probability | Moderate | Full target at every step penalises "not yet reached" as error. |
| **TRM deep supervision** 2510.04871 (full text) | Full final answer at each of up to N_sup = 16 supervision steps; learned halting q̂ | Moderate | "The same loss to the complete predicted answer" at every step; no masking. |
| **Anytime / multi-exit nets**: Hu et al. 1708.06832; MSDNet 1703.09844 | Full answer at every exit (layer depth, not recurrence) | Name only | Full target, feed-forward depth. |
| **Bansal progressive loss** 2202.05826 (full text) | Full answer after a random number of iterations (below) | Moderate | Random stopping point, full target, no masking. |

**One-line claim you can make:** anytime supervision is per-step supervision of the *answer* with a **depth-masked
target** (nodes within k + C hops positive, unreachable nodes negative, deeper reachable nodes masked). It enforces
*minimum* progress without fixing speed. This differs from exact-state hints (CLRS/NEGA), full-answer deep supervision
(PonderNet, TRM, multi-exit) and content alignment (SIM-CoT, RELAY). Cite CLRS/NEGA as the closest form and RELAY as the
closest in spirit.

## A2. Random-budget precedents (for Prop. 4 condition (b))

| Method | Budget during training | Independent of the example? | Minimum vs hops needed |
|---|---|---|---|
| **Coconut** `uniform_prob` (repo README) | "The probability to mix data from other stages": 0 standard, 0.3 in their analysis. Thoughts = c × stage | Mostly (random stage), but capped by the example's own step count unless `pad_latent_to_max` | Stage can be below the example's steps (partial latent) |
| **Pause tokens**, Goyal et al. 2310.02226 (full text) | Pretraining: M_pt = 10% of 2048 at **random positions** (count fixed). Fine-tuning: **fixed** M_ft (10 or 50) appended | Count fixed; positions random | Not a budget sweep; tolerates M_inf ∈ [5, 25] at inference for M_ft = 10 |
| **Kuo et al.** 2606.29983 (full text) | Stochastic: K = clip(n + Δ, 1, 20), Δ ~ Unif{−5..5}; length-matched K = n; fixed K = 20 | **No**: centred on length n | Support reaches n − 5 (below) and n |
| **Fan et al.** 2409.15647 | T(n) = ground-truth steps | **No**: exact per-example | Exactly tight |
| **Bansal et al.** 2202.05826 (full text) | n ~ U{0..m−1} iterations without gradient, then k ~ U{1..m−n} with gradient; L = (1−α)L_max + αL_prog | **Yes** | Down to 1 iteration (well below) |
| **Huginn** 2502.05171 (full text) | r ~ Poisson(e^τ) + 1, τ ~ N(log r̄ − ½σ², σ); "most often … less than r̄", heavy tail | **Yes** | Down to 1 |
| **DR-Loop** 2609.33144 (full text) | R ~ clip(Poisson(4), 2, 8) | **Yes** | 2, below most examples |
| **Yang et al.** 2311.12424 | Scheduled window, not random | — | — |
| **TRM** 2510.04871 | Up to 16 deep-supervision steps, learned halting (ACT-style) | Learned | — |
| **Our P22 randk** | U{4..8} (per batch) | Yes | 4, **above** the stage-3 need (slack 2–3) |

**Caveat to pass to Theory:** Fan's T(n) is a deterministic function of an observable (n), yet Fan *extrapolates* —
because the test budget is also T(n), whether oracle or max-confidence stopping. Kuo's K = n was worst, but Kuo clip K at
T_max = 20 and test on n = 20–60. **Check whether Kuo's test budget could exceed 20.** If not, their K = n failure may be
partly a clipped test budget, not only the length–loop shortcut. Condition (b) should say "deterministic in an
observable *and* not continued consistently at test".

## A3. Checkpoints and deeper datasets

**Checkpoints (Hugging Face):**
- **bmarti44/coconut-curriculum-checkpoints** (MIT): GPT-2 124M on **ProsQA**, seed 0, 50 epochs. CoT baseline (83.0%),
  **Coconut (97.0%)**, pause-curriculum (96.6%), pause-multipass (94.8%). The model card reports OOD evaluations on
  **7-hop and 8-hop** chains, DAG and dense graphs: Coconut 66.0% / 67.5% vs pause-curriculum 75.4% / 75.1% at 7 / 8
  hops; DAG 59.2% vs 51.9%. Its claim is that "recycled content hurts extrapolation", while sequential processing helps
  topological generalisation. That bears on Theory's Mechanism 1. The accompanying write-up ("The Curriculum Is the
  Mechanism") has no arXiv ID I could find; treat the numbers as model-card claims.
- **zen-E/CODI-gpt2**, **zen-E/CODI-llama3.2-1b-Instruct** (GSM8K-Aug).
- **internlm/SIM_COT-GPT2-CODI**, **internlm/SIM_COT-GPT2-Coconut** (GSM8K; SIM-CoT on top of CODI or Coconut).
- The official facebookresearch/coconut repo does not state that checkpoints are released (train from scratch).

**Datasets deeper than one pass:**
- **ProsQA generator** (Coconut repo): regenerate with more hops; your addendum-P sets already exist.
- **bmarti44's 7-/8-hop, DAG and dense ProsQA OOD sets** (with the checkpoints above; check the repo for files).
- **CLUTRR** 1908.06177: kinship chains of 2–10 hops; standard protocol trains k = 2–3 (or 2–4) and tests to 10.
- **StepGame** 2204.08292: spatial reasoning with k up to 10; test sets at k = 6–10 beyond training are standard.
- **Saparov et al.** 2412.04703: graph-connectivity generator with controllable size and lookahead (synthetic tokens).
- Use Biran 2406.12775 to set the "deeper than one pass" threshold, and your P curve to place it.

## A4. PROTOCOL9 novelty: a training fix tested across depth, width and graph shape with same-seed pairs

- **2609.33144** (Beyond the Training Horizon): mechanism only, **no fix, no seeds reported**. Two looped configurations,
  three tasks; nothing paired.
- **2607.20594** (Zhang et al., abstract): training-level fixes that work "in every seed": an operator-first curriculum
  dissolves the S5 wall, and warm-starting across budget contracts transfers the algorithm. The axis is input *length*
  on group word problems (not depth × width × shape in graphs), with 16-seed populations. The abstract doesn't say
  whether controls were same-seed pairs.
- **Kuo et al. 2606.29983**: a fix (stochastic stopping or RL-halting) across four tasks with 18–32 seeds, reported as
  mean ± variance, not same-seed pairs; looped transformers, not continuous thoughts.
- **Kohli et al. 2604.07822**: more recurrence at test time for deeper implicit hops; no fix across width or shape.
- **Verdict:** I found no paper testing a *continuous-thought (Coconut-style)* training fix across depth, width and graph
  shape with same-seed paired controls. Claim it with that scope. Closest: Kuo (many seeds, several tasks, looped) and
  Zhang (every-seed fixes, one task family, length axis).

---

# Part C — Theory (three questions)

## C1. Do trained continuous-thought models superpose the frontier or pick branches?

| Work | What it measures | What it finds on branch selection |
|---|---|---|
| **Coconut** 2412.06769 (full text) | Decodes thoughts through the LM head; forces language mode after k thoughts to read out an "implicit value function" over frontier nodes | **Spread, then commitment.** First thought: probability spread over children (e.g. 0.33 / 0.16 / 0.32 / 0.01); second thought: "mostly ruled out other options". A "transition from parallel exploration to more focused reasoning", **value-weighted**, not uniform. |
| **Zhu et al.** 2505.12514 (full text) | Layer-2 attention to frontier vs non-frontier edges; inner products of thought and node embeddings | Superposition, but **biased**: optimal-path edges score highest (2.54 vs frontier 2.12), attributed to optimal-path curriculum supervision. Frontier attention falls over steps (2.12 → 0.61). No per-branch count. |
| **Zhu et al.** 2509.23365 | Training dynamics of the index-matching logit | Bounded logit keeps *comparable* weights across traces. This predicts no winner-take-all, *given* their training. |
| **Rizvi-Martel et al.** 2604.06374 | Logit lens and entity-level probing (correct-next / wrong-neighbour / target / other) | Soft thinking and fine-tuned Coconut **collapse to one interpretation** ("the target entity dominates"). Only from-scratch models show superposition. |
| **Wu et al.** 2508.03440 | Probing Soft Thinking | "Single-threaded": models follow the top token, a greedy feedback loop that suppresses alternatives. |
| **Zou et al.** 2602.01148 | "Symbolic Index" (decisional certainty), with causal tests | Commit vs explore is governed by certainty: high certainty → commit (execution), low → explore. |
| **Backour** 2609.02049 | Theory plus experiments on mixture dynamics | Softmax feedback **amplifies or erases** differences between components. Winner-take-all can arise from the recurrence itself. |
| **Gu et al.** 2609.13747 | Capacity theory and trained models | Uneven coefficients cause a "weakest-object routing bottleneck" (joint route success 0.906 uniform vs 0.586 recency-heavy). A pointer-like, uneven weighting fails when all branches matter. |
| Kay et al. 2020, Cell (neuroscience) | Hippocampal decoding | Alternatives **time-multiplexed**, one per theta cycle, not superposed. |

**Advice.**
- (1) The literature supports **both** outcomes: superposition under from-scratch training with bounded logits (Zhu),
  and collapse or commitment through value-guidance, certainty or feedback amplification (Coconut's own analysis,
  Rizvi-Martel, Wu, Zou, Backour). A single-branch pointer in our toy is therefore not anomalous. Say which regime we
  are in.
- (2) Reconcile with our own Findings 1 and 3: every branch is causally editable, and per-edge attention halves when the
  frontier doubles (fork/chain 0.49–0.60), which looks like *splitting*. A "≥ 50% of max" count of ~1.3 edges can
  coexist with a skewed-but-spread distribution. Report the **effective number** (exp of attention entropy, or the
  participation ratio) alongside the threshold count.
- (3) Distinguish **commitment** (the same branch favoured at every step: Coconut, Backour) from **multiplexing** (the
  favoured branch rotates across steps: Kay). Also check whether the favoured branch is predictable from edge-token
  order or position: in the delayed format the query is unknown during the thoughts, so there is no value signal to
  select on, and any selection must be arbitrary.

## C2. Novelty checks

**F1a (random budgets whose support goes below the steps needed; final-answer labels only).**
- *Below-need budgets with final labels already exist:* Kuo 2606.29983 (K down to n − 5); Huginn 2502.05171 (mostly
  below the mean, min 1); DR-Loop 2609.33144 (min 2); Bansal 2202.05826 (random k ≥ 1 after random warm-up); **LoopFormer**
  2602.11451 (trains on "variable-length trajectories" including short ones, with a shortcut-consistency loss). Note that
  LoopFormer conditions each loop on the current time and step size, an explicit clock and so against C2. Feed-forward
  analogues: stochastic depth 1603.09382; anytime or multi-exit prediction 1708.06832 (early exits predict the final label
  with insufficient compute).
- *An explicit argument that budget ≤ need is what matters:* the nearest is **Zhang et al.'s budget law** 2607.20594. The
  training contract sets the mechanism's speed (v ≈ n_train / T_train; "the demanded minimum is met, never exceeded"), so a
  tighter contract selects a faster frontier. They frame it as *speed*, not *identification of the continuation*.
- **Verdict: medium–low as a method; the identification claim (Prop. 4 with condition (b)) is ours.** LoopFormer is a
  must-cite and must-distinguish (a clock-conditioned loop).

**F5a (distil the clock-resetting re-anchoring procedure into one uninterrupted latent loop).**
- *Bootstrapping from own outputs:* STaR 2203.14465, Quiet-STaR 2403.09629, self-improving transformers 2502.01612.
- *Distilling a procedure into the model:* iterated amplification and distillation (Christiano et al. 1810.08575, the
  exact pattern); Distilling System 2 into System 1 (2407.06023).
- *Distilling an iterative procedure into the model's own dynamics:* diffusion step distillation, i.e. progressive
  distillation 2202.00512 and consistency models 2303.01469 (the closest structural analogue); LoopFormer's
  shortcut-consistency (aligns short and long trajectories).
- **Verdict: medium–low as a method; new as distilling a *clock-resetting* procedure into continuous-thought
  recurrence.** Cite IDA and consistency models as the pattern.

## C3. Formal anchors for Prop. 9 ("a fractional speed needs a clock")

**Short argument.** Let the pointer's displacement at step k be a function Δ(s_k) of an internal state s_k that evolves
by a depth-independent deterministic map s_{k+1} = f(s_k) (homogeneity). If s ranges over a finite set, which it does in
effect at finite precision, the orbit is **eventually periodic** by pigeonhole with some period q. The displacement
sequence is then eventually periodic, and the average speed is m/q for integers m and q. A non-integer speed needs q ≥ 2:
the state must cycle through q distinct phases, i.e. carry a counter mod q, **a clock**. With exact real states,
irrational speeds need a rotation-like map on a phase variable, again a clock.

**Anchors:**
- **Ultimately periodic outputs of finite automata** (standard; e.g. any automata text). A deterministic finite-state
  system's output sequence is eventually periodic.
- **Sturmian words** (Morse & Hedlund 1940, *Symbolic Dynamics II: Sturmian Trajectories*, Amer. J. Math. 62:1–42;
  Lothaire 2002, *Algebraic Combinatorics on Words*, ch. 2). The "floor(n·α)" advance sequences with irrational α are
  aperiodic, so no finite automaton generates them. Irrational speed is impossible for finite-state homogeneous
  transitions; rational p/q needs period q.
- **Rotation number and mode locking** (Poincaré; Katok & Hasselblatt 1995, *Introduction to the Modern Theory of
  Dynamical Systems*, ch. 11; Arnold tongues). For circle maps, rotation number p/q ⇔ a periodic orbit of period q. In
  smooth parametrised families, rational rotation numbers occupy open parameter intervals (mode locking: the devil's
  staircase). A learned continuous phase map will generically lock to a rational speed with a period-q cycle. This
  predicts the fractional speeds you observe (½, ⅓ …) and not irrational ones, which is testable on the pointer
  trajectories.
- For counters in trained recurrent nets: Weiss et al. 1805.04908 (LSTMs and ReLU-RNNs learn counting; squashing RNNs
  cannot).

---

# Part B — Audit

## B1. Citation check (IDs and titles verified; claims checked against abstracts, full text where noted)

| Citation | ID and title | Attributed claim | Verdict |
|---|---|---|---|
| Haviv et al. | 2203.16634, *Transformer Language Models without Positional Encodings Still Learn Positional Information* | Causal transformers infer position without PE | ✔ ("causal attention enables the model to infer the number of predecessors") |
| D'Amour et al. | 2011.03395, *Underspecification Presents Challenges for Credibility in Modern ML* | Underspecification | ✔ |
| Anil et al. | 2211.09961, *Path Independent Equilibrium Models Can Better Exploit Test-Time Computation* | Path independence → upward generalisation | ✔ (abstract; the specific interventions were not verified) |
| Rodionov & Prokhorenkova | 2402.11628, *Discrete Neural Algorithmic Reasoning* (ICML 2025) | Hard attention and discretisation → size generalisation | ✔ |
| Veličković et al. | 2410.01104, *Softmax is not Enough (for Sharp Size Generalisation)* | Softmax dispersion | ✔ |
| Kuo et al. | 2606.29983, *Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping* | Seed-level instability; random loop counts help | ✔ with nuance: the stochastic schedule is n-centred (K = clip(n + Δ)); the best result is RL-halting (45.0%); length-matched K = n is worst |
| Zhang et al. | 2607.20594, *When Does Recurrence Become an Algorithm?* | In-distribution predictors of seed-level OOD fate | ✔ (head instruments; retrospective) |
| Aswal et al. | 2606.12689, *Observable Patterns Are Not Explanations* | Latent-reasoning shortcuts | ✔ in substance: thoughts removable with ~no loss, BFS signature reproduced by pause controls. "Shortcut" is our word, not theirs. |
| Cui et al. | 2602.22441, *How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?* | Latent-reasoning shortcuts | ✔ ("pervasive shortcut behavior") |
| Liang et al. | 2609.33144, *Beyond the Training Horizon* | Content-based progress in looped models | ✔ (full text: MR-Loop's "live–consumed" progress cue; no step counter reported) |
| Yu et al. (RELAY) | 2502.08482, *Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning* | Iteration-wise supervision → length generalisation | ✔ |
| Wei et al. (SIM-CoT) | 2509.20317, *SIM-CoT: Supervised Implicit Chain-of-Thought* | Step-level supervision stabilises latents | ✔ |

## B2. Prior work on Theory's proposals

**§26 re-anchored latent episodes** (restart a bounded latent loop from a decoded state). Beyond the CTM maze result and
the Markovian Thinker 2510.06557 (fixed chunks with a short carry-over, explicit CoT):
- **InftyThink** 2503.06692: iterative reasoning in segments with summaries carried forward (explicit CoT).
- **PENCIL** 2503.14337: erases intermediate thoughts by learned reduction rules, so the context stays short ("long
  thoughts with short memory").
- **HRM** 2506.21734: the low-level module converges, then is reset by the high-level update each cycle ("hierarchical
  convergence"). The closest *latent* analogue of re-anchoring.
- **Self-improving transformers** 2502.01612: re-apply the model to its own outputs on harder instances.

What is new in §26: re-anchoring a *continuous-thought* loop from a *decoded frontier node* to reset both the count
clock and the amplitude, tested as a training-free repair.

**F5a, distilling the re-anchoring procedure back into one episode.** Precedents:
- Christiano et al. 1810.08575 (iterated amplification and distillation): amplify with a procedure, then distil into
  the base model. The exact pattern.
- Yu et al. 2407.06023 (*Distilling System 2 into System 1*).
- CODI 2502.21074 (self-distillation into latents) and Deng et al. 2405.14838 (stepwise internalisation).
- Growing-NCA sample-pool training (Mordvintsev et al. 2020, Distill), the non-distillation route to persistence.

**F1a, a random budget K ~ U{1..K} with final-answer labels only.** Published random-count schemes that state the
minimum (A2 table): Bansal (k ≥ 1, independent), Huginn (≥ 1, independent), DR-Loop (≥ 2, independent), Kuo (n − 5,
n-centred). Only Kuo and Fan state the budget relative to the steps needed (both via n). Under Prop. 4, U{1..K} reaches
the tight budget, while P22's U{4..8} did not, so F1a is the discriminating test. None of these papers reports a slack
manipulation.

## B3. Standards for reporting a large family of pre-registered tests

Concrete sources to audit against:
1. **Deviation reporting template:** Willroth & Atherton 2024, *Best Laid Plans: A Guide to Reporting Preregistration
   Deviations*, AMPPS (doi 10.1177/25152459231213802). One table row per deviation: original plan, what changed, when
   (before or after which data), why, and the impact on conclusions.
2. **When a deviation is legitimate:** Lakens 2024, *When and How to Deviate From a Preregistration*, Collabra:
   Psychology 10(1):117094 (doi 10.1525/collabra.117094). Five categories: unforeseen events, errors in the
   preregistration, missing information, violated untested assumptions, falsified auxiliary hypotheses. Each deviation
   needs a justification and an evaluation of its consequence for the test's severity.
3. **Multiplicity:** Rubin 2021, *When to Adjust Alpha During Multiple Testing*, Synthese 199:10969 (arXiv 2107.02947).
   Adjust only for *disjunctive* claims ("at least one of these supports X"); not for individual claims, each with its
   own conclusion, or conjunctive ones. Audit question per verdict: is it an individual claim, or is the paper making a
   union claim across the family? If the write-up says "k of N predictions held", report the family explicitly and
   either apply FDR (Benjamini–Hochberg) across that family or state that verdicts are individual.
4. **Outcome switching:** COMPare (Goldacre et al. 2019, *Trials*, doi 10.1186/s13063-019-3173-2). Every pre-specified
   outcome must be reported, and every added outcome must be declared as such. In trials, switches favoured significant
   outcomes in half the cases.
5. **ML-specific:** Hofman et al. 2023, *Pre-registration for Predictive Modeling*, 2311.18807. Agarwal et al. 2021, *Deep
   RL at the Edge of the Statistical Precipice*, 2108.13264: with few seeds, report interval estimates (stratified
   bootstrap CIs, IQM, performance profiles), not point "supported/not supported".

**Suggested audit checklist for the write-up:**
- (a) One master table of *every* pre-registered test across PROTOCOL6 A–R, PROTOCOL7 T1–T13 and THEORY pilots A–J:
  registration commit time, first-data time, verdict, n. Include failed and unresolved tests.
- (b) Re-specified tests (e.g. T1→T1b, T2→T2b, P29→P30) keep the original verdict. The re-specification counts as a
  *new* test with its own registration time, and the report states what data had been seen (Willroth & Atherton row;
  Lakens category).
- (c) Amendments after partial data (P30's test-set growth, Pilot A's commit 4 min after the first run started): state
  whether any outcome-relevant data had been seen. If yes, the affected verdict is exploratory.
- (d) Multiplicity statement per Rubin, plus BH-adjusted values if any family-level claim is made.
- (e) For small-n verdicts, report effect sizes with CIs next to the binary verdict (Agarwal et al.).
