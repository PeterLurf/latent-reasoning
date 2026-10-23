# Literature — readable vs usable alternatives in latent reasoning

> **Historical (to 2026-09-27).** Kept in place because other docs cite it by path. Current literature: `docs/LIT_INDEX.md` (start with LIT_FINDINGS_COMPILED, LIT_LATENT_FROM_DISCRETE, LIT_SEMI_DISCRETE).

Compiled 2026-09-25, expanded 2026-09-27 after a systematic sweep: topic searches, the latentreasoning.net
bibliography, and forward citations of the closest papers. Every arXiv ID was checked against the arXiv API.
Entries are grouped by the question they bear on, and each note says what the paper means for this project
(Parts I–V, `docs/NEXT_STEPS.md`, `report/findings.md`). **Key** marks papers a write-up must engage with.
Where a note says "Finding N", it refers to the numbered findings in `report/findings.md`.

## 0. Closest work (engage with all of these)

| Paper | Relation to this project |
|---|---|
| **Key** Hao et al. 2024, *Training LLMs to Reason in a Continuous Latent Space* (Coconut), [2412.06769](https://arxiv.org/abs/2412.06769) | Source architecture; BFS-like frontiers on ProsQA. Our model is a small 2-layer re-implementation, not their GPT-2 system. |
| **Key** Zhu, Hao, Hu, Jiao, Russell, Tian 2025, *Reasoning by Superposition* (NeurIPS 2025), [2505.12514](https://arxiv.org/abs/2505.12514) | Proves a 2-layer transformer with D continuous thoughts solves reachability with the frontier in superposition (weights 1/√\|frontier\|). This is the construction our clocked sweep resembles. Evidence is observational only: no interventions, no width variation. |
| **Key** Zhu et al. 2025, *Emergence of Superposition* (ICLR 2026), [2509.23365](https://arxiv.org/abs/2509.23365) | Training dynamics of the same setting: a bounded index-matching logit keeps traces at comparable weight. Also: trained on 2 thoughts, generalises to 3–4. |
| **Key** Gu, Liu, Fu 2026, *How Many Thoughts Can a Vector Hold? The Capacity of Reasoning by Superposition*, [2609.13747](https://arxiv.org/abs/2609.13747) | Capacity theory with trained models (2-block and GPT-2) that varies the number of alternatives. Finds cumulative superposition needs less dimension than frontier-only, and weak components bottleneck routing. No per-alternative interventions. Nearest neighbour to Findings 1 and 3. |
| **Key** Aswal, Palmeira Ferraz, Zhou, Peyrard 2026, *Observable Patterns Are Not Explanations*, [2606.12689](https://arxiv.org/abs/2606.12689) | Pause-as-Thought reproduces the BFS signature. Removing thoughts barely changes Coconut ProsQA accuracy, and thought positions have ~0 causal recoverability. Our delayed format and depth-matched controls answer this. |
| **Key** Rizvi-Martel, Rabusseau, Mosbach 2026, *The Illusion of Superposition?*, [2604.06374](https://arxiv.org/abs/2604.06374) | Superposition appears only when training from scratch; fine-tuned Coconut takes shortcuts. Bears on GPT-2 init and the ProsQA transfer. |
| **Key** Zhang, Hu, Peng, Xie 2026, *When Does Recurrence Become an Algorithm?*, [2607.20594](https://arxiv.org/abs/2607.20594) | In-distribution "head instruments" (convergence time, frontier completion) predict seed-level extrapolation (AUROC up to 1.00, 3 × 16 seeds, retrospective). **Precedent for Part V end activity (P26)**; ours is prospective and in a different model class. Recheck that P20's "finish-early" account matches their claim. |
| **Key** Kuo, Chayti, Reizinger, Brendel, Jaggi 2026, *Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping*, [2606.29983](https://arxiv.org/abs/2606.29983) | Extrapolation is unstable across 18–32 seeds; randomised loop counts stabilise it. Our randk (P22) did not, a tension to report. No in-training diagnostic. |
| **Key** Shih, Winnicki, Darve 2026, *When Does Activation Steering Change What a Model Computes From?*, [2606.29522](https://arxiv.org/abs/2606.29522) | Three-part semantic test (use, move-specific, conflicting continuation) and natural-scale controls: steering vectors are 26–77× natural changes. The standard for NEXT_STEPS Approach 1. |
| **Key** Ding, Huang, Yang 2026, *SCIT: Testing Causal Cache Carriers in Latent CoT Models*, [2608.27265](https://arxiv.org/abs/2608.27265) | Suffix-cache interchange test: in CODI-GPT2 the counterfactual computation travels in value-cache suffixes, not hidden states. The ready-made protocol for NEXT_STEPS' cache-recovery experiment. |

## 1. Continuous / latent reasoning: analyses of what the latent state does

- Li et al. 2026, *Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure*, [2602.08783](https://arxiv.org/abs/2602.08783) — step-level do-interventions on Coconut/CODI; influence jumps between steps rather than moving one step at a time. Compare with our irregular clock readers.
- Dilgren & Wiegreffe 2026, *Are Latent Reasoning Models Easily Interpretable?*, [2604.04902](https://arxiv.org/abs/2604.04902) — latent tokens are often unnecessary on logical-reasoning sets; when needed, gold traces decode 65–93% of the time.
- Zhang et al. 2025, *Do Latent Tokens Think? A Causal and Adversarial Analysis of Coconut*, [2512.21711](https://arxiv.org/abs/2512.21711) — latent tokens act as placeholders and promote shortcuts.
- Jin, Yang, Wang 2026, *Final Checkpoints Are Not Enough: Latent Reasoning Faithfulness Along Training Trajectories*, [2607.06648](https://arxiv.org/abs/2607.06648) — counterfactual responsiveness can fall as accuracy rises, including on ProsQA. Supports tracking our measures across checkpoints (`development.py`).
- Liang & Pan 2026, *Do Latent-CoT Models Think Step-by-Step?*, [2602.00449](https://arxiv.org/abs/2602.00449) — CODI forms bridge states on short tasks and fuses late on long ones.
- Chang et al. 2026, *Unlocking the Black Box of Latent Reasoning* (ACL 2026), [2606.01243](https://arxiv.org/abs/2606.01243) — directional edits flip answers. This is answer steering, which our semantic test separates from branch-state editing.
- Wu et al. 2025, *LLMs are Single-threaded Reasoners: Demystifying Soft Thinking*, [2508.03440](https://arxiv.org/abs/2508.03440) — soft-token models follow the top token (a greedy feedback loop), so no parallel paths. A negative case against which our trained superposition stands out.
- Backour 2026, *The Dynamics of Continuous Mixture Collapse in Language Models*, [2609.02049](https://arxiv.org/abs/2609.02049) — softmax feedback amplifies or erases differences between mixture components. Theory for our branch competition (Finding 3).
- Larson 2026, *The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA*, [2609.03090](https://arxiv.org/abs/2609.03090) — rank ablation is flat and conflates rank with position-irrelevance. A cautionary example for subspace-rank arguments.
- Lu et al. 2025, *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer*, [2507.02199](https://arxiv.org/abs/2507.02199) — probing Huginn finds little interpretable latent CoT.
- Duraipandian et al. 2026, *Interpreting Latent CoT Reasoning as Dynamical Systems*, [2607.09698](https://arxiv.org/abs/2607.09698) — CODI shows attractor dynamics, Coconut expanding ones. Relevant to the rewind and phase-shift results.
- Malarkkan et al. 2026, *Invariant Reasoning Directions in Latent Trajectories*, [2606.29164](https://arxiv.org/abs/2606.29164) — contrastive refinement signals are low-rank.
- Yu 2024, *Do LLMs Really Think Step-by-step in Implicit Reasoning?*, [2411.15862](https://arxiv.org/abs/2411.15862); Yang et al. 2025, *Internal Chain-of-Thought: Layer-wise Subtask Scheduling*, [2505.14530](https://arxiv.org/abs/2505.14530) — implicit step structure in standard models.
- Brauer, Mayrink Verdun, Marks 2026, *Reading Between the Dots: Decoding Hidden Computation across Filler Tokens*, [2607.03502](https://arxiv.org/abs/2607.03502) — in frontier models, KV transplants at filler positions swap outputs and intermediate values decode at 80–95%. The pause-token counterpart of our analysis.

## 2. Continuous / latent reasoning: methods and theory

- Goyal et al. 2023, *Think Before You Speak: Training LMs With Pause Tokens*, [2310.02226](https://arxiv.org/abs/2310.02226) — the pause-token control.
- Pfau, Merrill, Bowman 2024, *Let's Think Dot by Dot*, [2404.15758](https://arxiv.org/abs/2404.15758); London & Kanade 2025, *Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers*, [2505.21024](https://arxiv.org/abs/2505.21024) — why pause models can match recurrence at shallow depth.
- Shen et al. 2025, *CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation*, [2502.21074](https://arxiv.org/abs/2502.21074); Wei et al. 2025, *SIM-CoT: Supervised Implicit Chain-of-Thought*, [2509.20317](https://arxiv.org/abs/2509.20317) — SIM-CoT finds latent training collapses as the number of latent steps grows and adds step-level supervision. Compare with our lookup scaffold and stalls.
- Deng, Choi, Shieber 2024, *From Explicit CoT to Implicit CoT: Internalizing CoT Step by Step*, [2405.14838](https://arxiv.org/abs/2405.14838) — the curriculum idea behind Coconut's stages.
- Gozeten et al. 2025, *Continuous Chain of Thought Enables Parallel Exploration and Reasoning* (CoT2), [2505.23648](https://arxiv.org/abs/2505.23648) — parallelism limited by embedding dimension, which our wide variant (P11) did not bear out.
- Xu & Sato 2025, *A Formal Comparison Between Chain of Thought and Latent Thought*, [2509.25239](https://arxiv.org/abs/2509.25239); Xu & Sato 2025, *To CoT or To Loop?*, [2505.19245](https://arxiv.org/abs/2505.19245) — expressivity separations.
- Deng et al. 2025, *LLM Latent Reasoning as Chain of Superposition*, [2510.15522](https://arxiv.org/abs/2510.15522); Suleymanzade et al. 2026, *MUX: Continuous Reasoning via Multiplexed Tokens*, [2607.18264](https://arxiv.org/abs/2607.18264) — superposition as a training target.
- Gatmiry et al. 2026, *Learn Your Own Thoughts: Abstract Token Curriculum*, [2609.19717](https://arxiv.org/abs/2609.19717) — continuous thoughts from a difficulty curriculum without scratchpad supervision; relevant to dropping our lookup scaffold.
- Fan, Svete, Lee 2026, *Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers*, [2606.31779](https://arxiv.org/abs/2606.31779) — latent CoT falls behind explicit CoT above 1B; looping closes the gap.
- Geiping et al. 2025, *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*, [2502.05171](https://arxiv.org/abs/2502.05171) — recurrent depth at scale.
- Surveys and positions: Zhu et al. 2025, *A Survey on Latent Reasoning*, [2507.06203](https://arxiv.org/abs/2507.06203); Wang 2026, *LLM Reasoning Is Latent, Not the Chain of Thought*, [2604.15726](https://arxiv.org/abs/2604.15726).
- Tracking resource: latentreasoning.net ([github.com/microprediction/latentreasoning](https://github.com/microprediction/latentreasoning)), a curated bibliography with issue-tracked annotations.

## 3. Safety motivation (monitorability of hidden reasoning)

- Korbak et al. 2025, *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*, [2507.11473](https://arxiv.org/abs/2507.11473) — why reasoning leaving text matters. NEXT_STEPS limits the claims we may make here.
- Ramjee 2026, *Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models*, [2604.23460](https://arxiv.org/abs/2604.23460) — misalignment decodable from early latent tokens.
- Brauer et al. 2026 (§1) — monitorability depends on the whole computational trace, not surface tokens.

## 4. Graph search and multi-step reasoning in transformers

- Saparov et al. 2025, *Transformers Struggle to Learn to Search* (ICLR 2025), [2412.04703](https://arxiv.org/abs/2412.04703) — without CoT, transformers learn path-merging (reachable sets double per layer). The non-recurrent mechanism behind query-time graph reading.
- **Key** Ye et al. 2025, *Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data*, [2510.19753](https://arxiv.org/abs/2510.19753) — L layers handle diameter ≤ 3^L; beyond that the model falls back on degree heuristics. A concrete depth bound for our 2-layer model and the depth split in Finding 4.
- Roy & Saparov 2025, *Transformers Can Learn Connectivity in Some Graphs but Not Others*, [2509.22343](https://arxiv.org/abs/2509.22343).
- Merrill & Sabharwal 2025, *A Little Depth Goes a Long Way: Log-Depth Transformers* (NeurIPS 2025), [2503.03961](https://arxiv.org/abs/2503.03961); Sanford, Hsu, Telgarsky 2024, *Transformers, Parallel Computation, and Logarithmic Depth*, [2402.09268](https://arxiv.org/abs/2402.09268); Sanford et al. 2024, *Understanding Transformer Reasoning Capabilities via Graph Algorithms*, [2405.18512](https://arxiv.org/abs/2405.18512) — connectivity needs depth growing as log n. This is why recurrence should matter only on deep queries (Finding 5).
- de Luca & Fountoulakis 2024, *Simulation of Graph Algorithms with Looped Transformers*, [2402.01107](https://arxiv.org/abs/2402.01107) — looped constructions for BFS/DFS/Dijkstra.
- Wang et al. 2024, *ALPINE: Planning Capability of Autoregressive Learning*, [2405.09220](https://arxiv.org/abs/2405.09220) — adjacency and reachability learned, transitive closure not.
- Bachmann & Nagarajan 2024, *The Pitfalls of Next-Token Prediction*, [2403.06963](https://arxiv.org/abs/2403.06963) — Clever Hans shortcuts on path-star graphs; background for the scaffold and pilots 1–2.
- Brinkmann et al. 2024, *A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task*, [2402.11917](https://arxiv.org/abs/2402.11917) — backward chaining on trees; compare with our target-first backward walk.
- **Key** Spies et al. 2024, *Transformers Use Causal World Models in Maze-Solving Tasks*, [2412.11867](https://arxiv.org/abs/2412.11867) — activating features is easier than suppressing them, and models handle more active features than in training. Compare with insertion vs removal and the width sweep.
- Wu et al. 2024, *Do Language Models Plan Ahead for Future Tokens?*, [2404.00859](https://arxiv.org/abs/2404.00859) — pre-caching vs breadcrumbs, a frame for whether the thought computes ahead or the answer re-reads (Finding 4).
- Ma & Rui 2026, *Where's the Plan? Locating Latent Planning with Lightweight Interventions*, [2605.07984](https://arxiv.org/abs/2605.07984) — decodable plans are causal only in some models. Another readable ≠ usable case.
- Yang et al. 2024, *Do LLMs Latently Perform Multi-Hop Reasoning?*, [2402.16837](https://arxiv.org/abs/2402.16837); Wang et al. 2024, *Grokked Transformers are Implicit Reasoners*, [2405.15071](https://arxiv.org/abs/2405.15071).

## 5. Looped / recurrent computation, clocks, and extrapolation (Parts III–V)

- Classic halting and recurrence: Dehghani et al. 2018, *Universal Transformers*, [1807.03819](https://arxiv.org/abs/1807.03819); Graves 2016, *Adaptive Computation Time*, [1603.08983](https://arxiv.org/abs/1603.08983); Banino et al. 2021, *PonderNet*, [2107.05407](https://arxiv.org/abs/2107.05407).
- Schwarzschild et al. 2021, *Can You Learn an Algorithm? Easy-to-Hard Generalization with Recurrent Networks*, [2106.04537](https://arxiv.org/abs/2106.04537); Bansal et al. 2022, *Logical Extrapolation Without Overthinking*, [2202.05826](https://arxiv.org/abs/2202.05826) — more test-time iterations solve harder instances. The precedent for "more steps → longer chains".
- Giannou et al. 2023, *Looped Transformers as Programmable Computers*, [2301.13196](https://arxiv.org/abs/2301.13196); Yang et al. 2023, *Looped Transformers are Better at Learning Learning Algorithms*, [2311.12424](https://arxiv.org/abs/2311.12424); Saunshi et al. 2025, *Reasoning with Latent Thoughts: On the Power of Looped Transformers*, [2502.17416](https://arxiv.org/abs/2502.17416).
- Fan et al. 2024, *Looped Transformers for Length Generalization*, [2409.15647](https://arxiv.org/abs/2409.15647) — adaptive loop count gives extrapolation.
- Kohli et al. 2026, *Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers* (COLM 2026), [2604.07822](https://arxiv.org/abs/2604.07822) — 5-hop → 10-hop by adding recurrence; overthinking degrades.
- Overthinking and step failure: Yang et al. 2026, *Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning*, [2605.26733](https://arxiv.org/abs/2605.26733); Guo et al. 2026, *Right Direction, Wrong Step*, [2609.16665](https://arxiv.org/abs/2609.16665) — accuracy peaks then collapses with more iterations. A counterpoint to extrapolating models that keep improving.
- Fixed points and stability: Blayney et al. 2026, *A Mechanistic Analysis of Looped Reasoning Language Models*, [2604.11791](https://arxiv.org/abs/2604.11791); Labovich 2026, *Stability and Generalization in Looped Transformers*, [2604.15259](https://arxiv.org/abs/2604.15259); Kim et al. 2026, *Looped Transformers with Source-Centered State Evolution*, [2607.27656](https://arxiv.org/abs/2607.27656). A converged fixed point vs a still-moving state is exactly what end activity measures.
- **Key** Buehlmaier 2026, *LayerNorm as Implicit Gain Control in Looped Transformers*, [2607.10681](https://arxiv.org/abs/2607.10681) — LayerNorm couples gain inversely to activation scale in recurrence. Candidate mechanism for the width-dependent gain (P10, P13 pre-LN feedback).
- Lam-Muir 2026, *The Ignition Is Real, and It Lives at the Readout*, [2608.03263](https://arxiv.org/abs/2608.03263) — pre-registered: arrival time rises lawfully with problem depth ("difficulty-clocked"), with a withdrawn headline reported. The closest methodological sibling, and a clock result to compare with P23/P24.
- Lam-Muir 2026, *Behaviour Is an Incomplete Measure of Reasoning Development*, [2608.16085](https://arxiv.org/abs/2608.16085) — decodability precedes behavioural arrival; calls for causal intervention.
- Wang & Reid 2026, *Looped Transformers under the Jacobian Lens*, [2609.01924](https://arxiv.org/abs/2609.01924) — Jacobian analysis of recurrence; relevant to NEXT_STEPS Approach 2.
- Kazemnejad et al. 2023, *The Impact of Positional Encoding on Length Generalization*, [2305.19466](https://arxiv.org/abs/2305.19466) — NoPE background for removing the step embedding (P19, P21).
- Zhou et al. 2023, *What Algorithms Can Transformers Learn? (RASP-L)*, [2310.16028](https://arxiv.org/abs/2310.16028); Anil et al. 2022, *Exploring Length Generalization in LLMs*, [2207.04901](https://arxiv.org/abs/2207.04901).
- Scale landscape (context only): Schwethelm et al. 2026, *How Much Is One Recurrence Worth?*, [2604.21106](https://arxiv.org/abs/2604.21106).

## 6. Causal interventions: semantics, methodology, pitfalls

- **Key** Geiger et al. 2025, *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability* (JMLR 26), [2301.04709](https://arxiv.org/abs/2301.04709) — foundation for NEXT_STEPS Approach 1.
- **Key** Sutter, Minder, Hofmann, Pimentel 2025, *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough?* (NeurIPS 2025), [2507.08802](https://arxiv.org/abs/2507.08802) — with unrestricted maps any network aligns to any algorithm. Justifies NEXT_STEPS' "restrict map complexity in advance".
- Geiger et al. 2023, *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations* (DAS), [2303.02536](https://arxiv.org/abs/2303.02536); Wu et al. 2023, *Interpretability at Scale: Identifying Causal Mechanisms in Alpaca* (Boundless DAS), [2305.08809](https://arxiv.org/abs/2305.08809) — learned-subspace interchange interventions.
- **Key** Makelov, Lange, Nanda 2023, *An Interpretability Illusion for Subspace Activation Patching*, [2311.17030](https://arxiv.org/abs/2311.17030) — subspace patches can work through dormant pathways. Caveat for the pattern-subspace edits.
- Grant et al. 2025, *Addressing Divergent Representations from Causal Interventions*, [2511.04638](https://arxiv.org/abs/2511.04638) — interventions push representations off-distribution, with harmless vs harmful cases. Our donor edits are natural by construction; say so.
- Vaidyanathan et al. 2026, *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*, [2606.27510](https://arxiv.org/abs/2606.27510) — the indirect effect contains interaction terms. Directly relevant to the thought × cache factorial.
- Heimersheim & Nanda 2024, *How to Use and Interpret Activation Patching*, [2404.15255](https://arxiv.org/abs/2404.15255); Zhang & Nanda 2023, *Towards Best Practices of Activation Patching*, [2309.16042](https://arxiv.org/abs/2309.16042) — metrics (logit vs probability) and corruption choice.
- Vig et al. 2020, *Causal Mediation Analysis for Interpreting Neural NLP*, [2004.12265](https://arxiv.org/abs/2004.12265); Meng et al. 2022, *Locating and Editing Factual Associations in GPT* (ROME), [2202.05262](https://arxiv.org/abs/2202.05262); Chan et al. 2022, *Causal Scrubbing* (Alignment Forum).
- Self-repair, the frame for cache recovery: McGrath et al. 2023, *The Hydra Effect*, [2307.15771](https://arxiv.org/abs/2307.15771); Rushing & Nanda 2024, *Explorations of Self-Repair*, [2402.15390](https://arxiv.org/abs/2402.15390); Gong et al. 2026, *Conditional Co-Ablation: Recovering Self-Repair Backups*, [2607.01940](https://arxiv.org/abs/2607.01940); Patrawala et al. 2026, *LLM Layers Immediately Correct Each Other*, [2609.07876](https://arxiv.org/abs/2609.07876); Wang et al. 2022, *IOI circuit* (backup heads), [2211.00593](https://arxiv.org/abs/2211.00593).
- Steering: Turner et al. 2023, *Activation Engineering (ActAdd)*, [2308.10248](https://arxiv.org/abs/2308.10248); Park et al. 2023, *The Linear Representation Hypothesis*, [2311.03658](https://arxiv.org/abs/2311.03658).

## 7. Probing: readable ≠ usable (Finding 2)

- **Key** Haufe et al. 2014, *On the Interpretation of Weight Vectors of Linear Models in Multivariate Neuroimaging*, NeuroImage 87:96–110 — filters vs patterns.
- **Key** Marks & Tegmark 2023, *The Geometry of Truth*, [2310.06824](https://arxiv.org/abs/2310.06824) — mass-mean probe directions have more causal effect than logistic-regression directions at similar accuracy. The direct precedent for filter vs pattern: a linear probe's Haufe pattern is close to the class mean difference. Ours is the stronger form (filters fully inert).
- Im & Li 2025, *A Unified Understanding and Evaluation of Steering Methods*, [2502.02716](https://arxiv.org/abs/2502.02716) — mean-of-differences is the optimal steering direction and beats classifier directions.
- Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks*, [1909.03368](https://arxiv.org/abs/1909.03368); Belinkov 2022, *Probing Classifiers*, [2102.12452](https://arxiv.org/abs/2102.12452).
- Elazar et al. 2021, *Amnesic Probing*, [2006.00995](https://arxiv.org/abs/2006.00995); Belrose et al. 2023, *LEACE: Perfect Linear Concept Erasure in Closed Form*, [2306.03819](https://arxiv.org/abs/2306.03819) — erase-then-measure alternatives to donor patching.

## 8. Superposition, shared budgets, and the width law (Findings 1, 3)

- Elhage et al. 2022, *Toy Models of Superposition*, [2209.10652](https://arxiv.org/abs/2209.10652); Hänni et al. 2024, *Mathematical Models of Computation in Superposition*, [2408.05451](https://arxiv.org/abs/2408.05451); Adler & Shavit 2024, *On the Complexity of Neural Computation in Superposition*, [2409.15318](https://arxiv.org/abs/2409.15318).
- **Key** Veličković et al. 2024, *Softmax is not Enough (for Sharp Size Generalisation)*, [2410.01104](https://arxiv.org/abs/2410.01104); Nakanishi 2025, *Scalable-Softmax Is Superior for Attention*, [2501.19399](https://arxiv.org/abs/2501.19399) — softmax attention necessarily disperses as items grow. The null model for the per-edge budget law (P25).
- Hahn 2020, *Theoretical Limitations of Self-Attention*, [1906.06755](https://arxiv.org/abs/1906.06755).
- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* (transformer-circuits.pub); Olsson et al. 2022, *In-context Learning and Induction Heads*, [2209.11895](https://arxiv.org/abs/2209.11895).
- Gu et al. 2026 (§0), Backour 2026 (§1) and Buehlmaier 2026 (§5) are the 2026 theory for competition and gain.

## 9. Controllability (NEXT_STEPS Approach 2)

- Boyd & Lall, *EE263: Introduction to Linear Dynamical Systems*, Stanford lecture notes.
- Wang & Reid 2026 (§5) — Jacobian lens on recurrence.

## 10. Training dynamics, stalls, and seed variation (Parts IV–V)

- **Key** Zhong et al. 2023, *The Clock and the Pizza: Two Stories in Mechanistic Explanation*, [2306.17844](https://arxiv.org/abs/2306.17844) — seeds and hyperparameters select qualitatively different algorithms. The frame for regular vs irregular clock readers and seed-level extrapolation.
- McCoy, Min, Linzen 2020, *BERTs of a Feather Do Not Generalize Together*, [1911.02969](https://arxiv.org/abs/1911.02969); Sellam et al. 2022, *The MultiBERTs*, [2106.16163](https://arxiv.org/abs/2106.16163) — same in-distribution accuracy, very different generalisation across seeds.
- Hoffmann et al. 2024, *Eureka-Moments in Transformers: Multi-Step Tasks Reveal Softmax Induced Optimization Problems* (ICML 2024), [2310.12956](https://arxiv.org/abs/2310.12956) — long plateaus on intermediate steps from softmax. A candidate explanation for our late-curriculum stalls and the lr 5e-4 effect.
- Wei et al. 2025, SIM-CoT (§2) — latent training collapse with more latent steps.
- Bouthillier et al. 2021, *Accounting for Variance in ML Benchmarks*, [2103.03098](https://arxiv.org/abs/2103.03098).
- Nosek et al. 2018, *The Preregistration Revolution*, PNAS 115(11):2600–2606 — framing for PROTOCOL 1–5.

## 11. Added 2026-09-29 — for the Part VI–VII horizon results and the theory stream

Claim-by-claim use of these papers: `docs/ADVISOR_2026-09-29.md`. All IDs verified.

- **Key** Liang et al. 2026, *Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers*, [2609.33144](https://arxiv.org/abs/2609.33144) — looped transformers, including knowledge-graph traversal. A propagating frontier whose confidence decays per hop, a content-based progress cue, limited self-correction; no seeds, no fix. The closest work to the propagation-limit horizon and to Mechanism 2.
- **Key** Yu et al. 2025, *Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning* (RELAY, EACL 2026), [2502.08482](https://arxiv.org/abs/2502.08482) — iteration-wise supervision gives looped transformers length generalisation. The nearest precedent for anytime supervision.
- **Key** Rodionov & Prokhorenkova 2024, *Discrete Neural Algorithmic Reasoning* (ICML 2025), [2402.11628](https://arxiv.org/abs/2402.11628) — discretised states plus hard attention give provable size generalisation, overcoming attention annealing on large graphs. A restoring-map precedent.
- Altabaa et al. 2025, *Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning*, [2510.14095](https://arxiv.org/abs/2510.14095) — discrete-bottleneck anchoring plus error correction.
- Rodionov & Prokhorenkova 2023, *Neural Algorithmic Reasoning Without Intermediate Supervision*, [2306.13411](https://arxiv.org/abs/2306.13411); Bevilacqua et al. 2023, *Neural Algorithmic Reasoning with Causal Regularisation*, [2302.10258](https://arxiv.org/abs/2302.10258); Veličković et al. 2022, *The CLRS Algorithmic Reasoning Benchmark*, [2205.15659](https://arxiv.org/abs/2205.15659) — step-level hints vs none.
- Anil et al. 2022, *Path Independent Equilibrium Models Can Better Exploit Test-Time Computation*, [2211.09961](https://arxiv.org/abs/2211.09961); Merrill et al. 2020, *A Formal Hierarchy of RNN Architectures*, [2004.08500](https://arxiv.org/abs/2004.08500); Weiss, Goldberg, Yahav 2018, *On the Practical Computational Power of Finite Precision RNNs*, [1805.04908](https://arxiv.org/abs/1805.04908) — attractors and saturation.
- **Key** Haviv et al. 2022, *Transformer Language Models without Positional Encodings Still Learn Positional Information*, [2203.16634](https://arxiv.org/abs/2203.16634) — causal attention counts predecessors. The implicit clock behind retrieval schedules.
- **Key** D'Amour et al. 2020, *Underspecification Presents Challenges for Credibility in Modern Machine Learning*, [2011.03395](https://arxiv.org/abs/2011.03395) — the general form of the slack account's Prop. 1.
- Summers & Dinneen 2021, *Nondeterminism and Instability in Neural Network Optimization*, [2103.04514](https://arxiv.org/abs/2103.04514) — one-bit changes lead to different solutions; a frame for the precision null.
- Critical windows: Achille, Rovere, Soatto 2017, *Critical Learning Periods in Deep Neural Networks*, [1711.08856](https://arxiv.org/abs/1711.08856); Ali 2026, *Critical Windows of Complexity Control*, [2605.04396](https://arxiv.org/abs/2605.04396).
- Seed-level strategies: Juneja et al. 2022, *Linear Connectivity Reveals Generalization Strategies*, [2205.12411](https://arxiv.org/abs/2205.12411); Qin et al. 2024, *Sometimes I am a Tree*, [2412.04619](https://arxiv.org/abs/2412.04619); Zhou et al. 2024, *Transformers Can Achieve Length Generalization But Not Robustly*, [2402.09371](https://arxiv.org/abs/2402.09371).
- Latent CoT limits: Cui et al. 2026, *How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?*, [2602.22441](https://arxiv.org/abs/2602.22441); Zou et al. 2026, *Capabilities and Fundamental Limits of Latent Chain-of-Thought*, [2602.01148](https://arxiv.org/abs/2602.01148); Biran et al. 2024, *Hopping Too Late*, [2406.12775](https://arxiv.org/abs/2406.12775).
- Markov and deep-supervision relatives: Aghajohari et al. 2025, *The Markovian Thinker*, [2510.06557](https://arxiv.org/abs/2510.06557); Takashiro et al. 2026, *Exploration of Fast-Slow Latent Recurrence for Train-Short, Test-Long Generalization*, [2604.01577](https://arxiv.org/abs/2604.01577); Asadulaev et al. 2025, *Latent Reasoning in TRMs is Secretly a Policy Improvement Operator*, [2511.16886](https://arxiv.org/abs/2511.16886); Li et al. 2026, *Structural Process Supervision for Latent Chain-of-Thought Reasoning* (PMPS), [2609.09928](https://arxiv.org/abs/2609.09928).

- **Key** Kavuncu 2026, *LSTM-UT and Recurrent-Depth Transformers on Cellular Automata*, [2609.19521](https://arxiv.org/abs/2609.19521) — a current-state-only recurrent transformer extrapolates; one that keeps an expanding cache of earlier recurrent states fails, and retained history undermines state correction. Independent support for Mechanism 1 / A1.
- Recurrent-state perturbation and attractors (for T6): Mozer et al. 2018, *State-Denoised Recurrent Neural Networks*, [1805.08394](https://arxiv.org/abs/1805.08394); Lim et al. 2021, *Noisy Recurrent Neural Networks*, [2102.04877](https://arxiv.org/abs/2102.04877); Lamb et al. 2019, *State-Reification Networks*, [1905.11382](https://arxiv.org/abs/1905.11382); Viakhirev et al. 2026, *Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth*, [2608.18222](https://arxiv.org/abs/2608.18222) (settling vs drifting operators predict depth safety).
- Set/max state for reachability (for A6): Veličković et al. 2019, *Neural Execution of Graph Algorithms*, [1910.10593](https://arxiv.org/abs/1910.10593); Bonet 2026, *Recurrent Graph Neural Networks with Set-Based Aggregation*, [2609.15932](https://arxiv.org/abs/2609.15932).
- Bypass and one-pass sufficiency (for Prop. 8): Fu et al. 2025, *Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning*, [2511.08577](https://arxiv.org/abs/2511.08577); Geirhos et al. 2020, *Shortcut Learning in Deep Neural Networks*, [2004.07780](https://arxiv.org/abs/2004.07780); Baherwani et al. 2026, *Not All LLM Reasoning is Visible in the Chain-of-Thought*, [2607.22925](https://arxiv.org/abs/2607.22925).

## 12. Adjacent fields — ideas to borrow (2026-09-29)

Worked through as proposals in `docs/IDEAS_ADJACENT_2026-09-29.md`. arXiv IDs verified.

- Darlow et al. 2025, *Continuous Thought Machines*, [2505.05522](https://arxiv.org/abs/2505.05522) — internal ticks, neuron-level models, synchronisation as representation (worked through in THEORY §26).
- Clocks: Koutník et al. 2014, *A Clockwork RNN*, [1402.3511](https://arxiv.org/abs/1402.3511); Neil et al. 2016, *Phased LSTM*, [1610.09513](https://arxiv.org/abs/1610.09513); Xu & Sato 2024, *On Expressive Power of Looped Transformers: … Timestep Encoding*, [2410.01405](https://arxiv.org/abs/2410.01405).
- Structure-aligned positions: Cho et al. 2024, *Position Coupling*, [2405.20671](https://arxiv.org/abs/2405.20671); McLeish et al. 2024, *Transformers Can Do Arithmetic with the Right Embeddings*, [2405.17399](https://arxiv.org/abs/2405.17399).
- Control flow and restoring maps: Csordás, Irie, Schmidhuber 2021, *The Neural Data Router*, [2110.07732](https://arxiv.org/abs/2110.07732); Ramsauer et al. 2020, *Hopfield Networks is All You Need*, [2008.02217](https://arxiv.org/abs/2008.02217); Tang et al. 2020, *Iterative Homogeneous GNNs*, [2010.13547](https://arxiv.org/abs/2010.13547); Yan et al. 2020, *Neural Execution Engines*, [2006.08084](https://arxiv.org/abs/2006.08084).
- Energy-based iterative reasoning: Du et al. 2022, *IREM*, [2206.15448](https://arxiv.org/abs/2206.15448); Du et al. 2024, *IRED*, [2406.11179](https://arxiv.org/abs/2406.11179); Gladstone et al. 2025, *Energy-Based Transformers*, [2507.02092](https://arxiv.org/abs/2507.02092).
- Binding by phase: Miyato et al. 2024, *Artificial Kuramoto Oscillatory Neurons*, [2410.13821](https://arxiv.org/abs/2410.13821); Löwe et al. 2022, *Complex-Valued Autoencoders for Object Discovery*, [2204.02075](https://arxiv.org/abs/2204.02075).
- Curricula and search: Lee et al. 2025, *Self-Improving Transformers*, [2502.01612](https://arxiv.org/abs/2502.01612); Gandhi et al. 2024, *Stream of Search*, [2404.03683](https://arxiv.org/abs/2404.03683); Ye et al. 2024, *Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning*, [2410.14157](https://arxiv.org/abs/2410.14157).
- Interpretability tool: Ghandeharioun et al. 2024, *Patchscopes*, [2401.06102](https://arxiv.org/abs/2401.06102).
- Neuroscience and cognitive science (non-arXiv): Mordvintsev et al. 2020, *Growing Neural Cellular Automata*, Distill (sample-pool training); Kay et al. 2020, *Constant Sub-second Cycling between Representations of Possible Futures in the Hippocampus*, Cell 180:552 (doi 10.1016/j.cell.2020.01.014); Reynolds & Heeger 2009, *The Normalization Model of Attention*, Neuron 61:168–185; Carandini & Heeger 2012, *Normalization as a Canonical Neural Computation*, Nat Rev Neurosci 13:51–62; Ma, Husain, Bays 2014, *Changing Concepts of Working Memory*, Nat Neurosci 17:347–356.

## 13. Added 2026-10-21 (Literature; details in docs/LIT_NOVELTY_221_222_PS2_2026-10-21.md)
- Chang et al. 2026, *Unlocking the Black Box of Latent Reasoning*, [2606.01243](https://arxiv.org/abs/2606.01243): GSM8K whole latent-sequence transplant flips 2.5 % (Table 2), the nearest natural-data prior to §221's donor-final follow 0.03.
- Wang et al. 2026, *Latent CoT as Planning* (PLaT), [2601.21358](https://arxiv.org/abs/2601.21358): a question-blind decoder of latent groups to CoT segments; adding the question hurts. Prior for question-blind decoding (PORT-STATE).
- Elesedy & Zaidi 2021, [2102.10333](https://arxiv.org/abs/2102.10333): invariant / anti-invariant orthogonal decomposition (Lemma 1). The Frame Lemma is a corollary.
- Shen et al. 2025, CODI [2502.21074](https://arxiv.org/abs/2502.21074) §5.1: intermediates decoded from latents, separated by placeholder latents (Table 3: 97.1 / 83.9 / 75.0 %). Prior for Note 221-2's lens reading.
- Blogs, not peer-reviewed: Cywiński et al. (LessWrong, 2025-12-22, CODI-1B latents 3 / 5 hold steps; https://www.lesswrong.com/posts/YGAimivLxycZcqRFR); Realmbird (LessWrong, 2026-03-19, compute / store alternation on CODI-1B; https://www.lesswrong.com/posts/rnwjdw5hzddjRvBoC).

- Internal-only forgetting (Note 216-13; docs/LIT_INTERNAL_ONLY_FORGETTING_2026-10-21.md): Haskins & Adams [2505.10822](https://arxiv.org/abs/2505.10822); Wang et al. [2502.11812](https://arxiv.org/abs/2502.11812); Chhabra et al. [2503.01896](https://arxiv.org/abs/2503.01896); LDIFS, Mukhoti et al. [2308.13320](https://arxiv.org/abs/2308.13320); Lee, Sohn & Lee [2602.23197](https://arxiv.org/abs/2602.23197); L2-SP [1802.01483](https://arxiv.org/abs/1802.01483).

- §218 close / §223 aux risk (docs/LIT_S218_CLOSE_S223_AUX_2026-10-21.md): Saparov et al. [2412.04703](https://arxiv.org/abs/2412.04703); Ravichander et al. [2005.00719](https://arxiv.org/abs/2005.00719); Du et al. [1812.02224](https://arxiv.org/abs/1812.02224); PCGrad [2001.06782](https://arxiv.org/abs/2001.06782); GradNorm [1711.02257](https://arxiv.org/abs/1711.02257); Frydenlund [2410.13779](https://arxiv.org/abs/2410.13779); Sanford et al. [2402.09268](https://arxiv.org/abs/2402.09268).

- Round-3 FREEZE-G / PARAM-SWAP (docs/LIT_R3_FREEZEG_2026-10-21.md): Brunner et al. [1908.04211](https://arxiv.org/abs/1908.04211); Panigrahi et al. [2302.06600](https://arxiv.org/abs/2302.06600); Surgical fine-tuning [2210.11466](https://arxiv.org/abs/2210.11466); LoRA forgets less [2405.09673](https://arxiv.org/abs/2405.09673); Layer swapping [2410.01335](https://arxiv.org/abs/2410.01335); Stoehr et al. [2403.19851](https://arxiv.org/abs/2403.19851); Varma et al. [2309.02390](https://arxiv.org/abs/2309.02390); Lubana et al. [2211.08422](https://arxiv.org/abs/2211.08422); TIES [2306.01708](https://arxiv.org/abs/2306.01708); Frankle et al. [1912.05671](https://arxiv.org/abs/1912.05671); Juneja et al. [2205.12411](https://arxiv.org/abs/2205.12411); Kong et al. [2601.14758](https://arxiv.org/abs/2601.14758).

- Idle-pass drift / fixed-point levers (docs/LIT_IDLE_HORIZON_PRESCOUT_2026-10-21.md): Viakhirev et al. [2608.18222](https://arxiv.org/abs/2608.18222); Bai et al. [2106.14342](https://arxiv.org/abs/2106.14342); Kolawole et al. [2609.39277](https://arxiv.org/abs/2609.39277); Bansal et al. [2202.05826](https://arxiv.org/abs/2202.05826).

- §219 result vs self-improvement (docs/LIT_S219_TEXTSELF_RESULT_2026-10-21.md): our matched text-CoT vs latent self-training (Note 219-13, CROSSED; n50 20.16 vs 20.54 at round 2) bears on Lee [2502.01612](https://arxiv.org/abs/2502.01612) (margin form) and Huang [2511.07378](https://arxiv.org/abs/2511.07378) (proportional margin); STaR [2203.14465](https://arxiv.org/abs/2203.14465), ReST^EM [2312.06585](https://arxiv.org/abs/2312.06585). C3 efficiency (§225, Note 225-1): at matched compute, latent ≈ text-CoT with ~20–50× fewer sequential steps and ~0.65× FLOPs per correct answer (scope as above). No compute-matched prior was found; LOTUS [2606.31779](https://arxiv.org/abs/2606.31779) is latency-framed.

- One-pass depth thresholds (docs/LIT_ONEPASS_DEPTH_THRESHOLD_2026-10-23.md): Lin et al. [2503.07604](https://arxiv.org/abs/2503.07604) (implicit multi-step only with fixed premise order); Physics of LMs 2.1 [2407.20311](https://arxiv.org/abs/2407.20311) (depth > width); Balesni et al. [2411.16353](https://arxiv.org/abs/2411.16353); Dziri et al. [2305.18654](https://arxiv.org/abs/2305.18654).

## Internal sources (from NEXT_STEPS.md)

- *Branch edits in continuous thought*, report dated 23 Sep 2026 (`report/latent_branch_report.pdf`).
- *Mechanistic interpretability research handbook* (supplied as `mechanistic-interpretability-research-handbook.md`; not in this repo).
- *Theoretical frameworks for readable and usable alternatives in latent reasoning*, research memo, 23 Sep 2026 (not in this repo).

## Novelty status of the findings (as of 2026-09-27)

| Finding (`report/findings.md`) | Status | Nearest prior work |
|---|---|---|
| 1. Per-branch edits in a shared thought; selectivity falls with width | New | Gu et al. 2609.13747 (capacity, no interventions); Zhu et al. 2505.12514 (observational) |
| 2. Probe filters inert, patterns causal | New in latent reasoning; precedent in general | Marks & Tegmark 2310.06824; Im & Li 2502.02716; Haufe 2014 |
| 3. Branch competition (signed collateral); per-frontier-edge budget | New empirically; dispersion expected in theory | Veličković 2410.01104; Backour 2609.02049; Gu 2609.13747 |
| 4. Width moves mid-depth nodes to query-time reading | New | Aswal 2606.12689 (total bypass on ProsQA); Ye 2510.19753 (depth bound) |
| 5. Recurrence helps only the deepest queries | Consistent with prior work | Aswal 2606.12689; Merrill & Sabharwal 2503.03961 |
| 6. Step-locked sweep by causal blocking; only some seeds | New (causal); clocked arrival also in Lam-Muir 2608.03263 | Zhu 2505.12514 (attention only); Li 2602.08783; Zhong 2306.17844 |
| 7. End activity predicts extrapolation | Precedent (retrospective); ours prospective | Zhang et al. 2607.20594 |
| Extrapolation with more steps; seed dependence | Not new | Zhu 2509.23365; Kohli 2604.07822; Kuo 2606.29983 |
