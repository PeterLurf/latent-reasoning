# Literature — readable vs usable alternatives in latent reasoning

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
