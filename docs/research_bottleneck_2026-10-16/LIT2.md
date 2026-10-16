*Prior-art sweep for the bottleneck session, by a Theory Opus subagent standing in for the offline Literature session (61 IDs verified on abs pages; 50 read in full).*

# LIT2: prior art for the three VERSION-BANK bottlenecks (2026-10-16 02:57 EDT)

**Status.** An Opus literature sweep for Theory, run while Literature is offline. It feeds the "fix bottlenecks" session (user, 02:44 EDT) on VB (THEORY.md §211, docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md).

**Method.**
- Every ID in the table was checked on its arxiv.org/abs page (title and first author).
- "full" means the full text was read (arXiv HTML, or PDF via pdftotext), by me or by four Opus reader subagents. I spot-checked the load-bearing quotes against the downloaded text.
- "abs" means abstract only.

**Tags.** [THEOREM] is proved, with its assumptions. [DERIVATION] is an informal argument. [EMPIRICS] is measured, with its setting. [CONJECTURE] is mine.

---

## Summary (12 lines)

1. **(1) Step: closest prior.** Altabaa 2510.14095 uses a per-step "depth ≤ t ⇒ decoded at iteration t" loss, argmax re-embedding and value corruption. On mod-23 DAGs, N ≤ 32 → 128, it solves 100%, vs 54.8% with continuous latent supervision.
2. **(1) Step: same task.** RELAY 2502.08482 (per-loop CoT-round targets, Feng-style arithmetic, ≤ 15 → 25 operators) scores 0.98 vs 0.77 for an unsupervised loop. The state is continuous, the model is from scratch, and the loop count is an oracle.
3. **(1) Step: lessons.** At depth, the lever was snapping the state to a discrete re-embedding plus corruption training, not the soft target. Ground-truth hint teacher forcing *hurt* OOD in CLRS (2209.11142).
4. **(1) Step: new for us.** A pretrained GPT-2 retrofit, a whole-problem bank, and recode plus MCT (a continuous, minimum-norm snap). No paper snaps or corrupts states inside a pretrained LM.
5. **(1)/(2) Negative for VB.** Full cross-iteration KV history hurts depth extrapolation: CoTFormer scores 50.1% at depth 20 vs 99.4% for recent-1 (2609.19521). CLRS also gains from dropping history (2403.04929). VB's bank is this pattern.
6. **(2) Size-keying: closest prior.** Liang 2609.33144 shows causally that a longer prompt impairs the looped readout state at the *same hop* (transplant: 5.5% → 83.6%).
7. **(2) Size-keying: exact fix.** Ito 2608.31067 uses bracket-depth positions and a hardmax "deepest span" mask. It is exact on nested modular arithmetic from depth 1–2 to depth 30; a Fan-style looped transformer is at chance by depth 3. NDR 2110.07732 takes nested mod-10 arithmetic from depth 5 to 8 at 0.98 vs 0.47.
8. **(2) Mechanisms with proofs:**
   - absolute PE pins heads [THEOREM, 2607.18759];
   - softmax disperses as Θ(1/n) [THEOREM, 2410.01104];
   - normaliser count leakage (2202.12172);
   - loop–n coupling (2606.29983, 2607.20594).
   - No paper proves depth invariance for an SGD-learned looped step.
9. **(2) Size-keying: new for us.** A content-derived structural position (bracket depth or reducible span from R) and a size-free bank read, in a pretrained core.
10. **(3) Halting transfer.** Learned halts transfer past training only from scratch with a size-visible signal: PonderNet parity 48 → 96; Kuo RL-halt < 20 → 60 (45.0 vs 22.1). In pretrained models, ACT and PonderNet collapse to one loop (2609.03379), and TaH2 and Ouro gain nothing past the trained depth.
11. **(3) Counter at the edge.** Correctness and loss-gain halt labels (TRM, Ouro, RecurTrace) exist only up to the trained depth. T = n training fixes the pace at one step per loop (2607.20594). The closest "done" predicate on the state is Ito's hand-built "no parentheses remain".
12. **(3) Halting: new for us.** HALT-R stops when R decodes no live operator. It is a content predicate, defined at every n and checkable with STALL. It needs K decoupled from n and a post-hoc readout, not a joint gate (2607.20519).

---

## Verified IDs

| ID | Paper (first author, year) | Read | Fam. |
|---|---|---|---|
| 2510.14095 | Altabaa 2025, Recursive latent space reasoning | full | A |
| 2502.08482 | Yu 2025, RELAY loop-aligned reasoning | full | A |
| 2202.05826 | Bansal 2022, Algorithm synthesis w/o overthinking | full | A |
| 2209.11142 | Ibarz 2022, Generalist Neural Algorithmic Learner | full | A |
| 2302.10258 | Bevilacqua 2023, NAR with Causal Regularisation | full | A |
| 2403.04929 | Bohde 2024, Markov Property of NAR | full | A |
| 2402.11628 | Rodionov 2024, Discrete NAR | full | A |
| 2306.13411 | Rodionov 2023, NAR w/o Intermediate Supervision | full | A |
| 2006.08084 | Yan 2020, Neural Execution Engines | full | A/C |
| 2604.18839 | Cameron 2026, Denoising Recursion Models | full (§1–4) | A |
| 2609.19521 | Kavuncu 2026, Storing Is Not Remembering (LSTM-UT) | full | A/D |
| 2607.20594 | Zhang 2026, Convergence Selection in Looped Tr. | full | A/D/E |
| 2609.39892 | Wu 2026, How Looped Tr. Route Each Loop | full | A/C |
| 2205.15659 | Veličković 2022, CLRS | abs | A |
| 1905.13211 | Xu 2019, What Can NNs Reason About? (alignment) | abs | A |
| 2502.01612 | Lee 2025, Self-Improving Transformers | abs | A |
| 2602.10014 | Liu 2026, Easy-to-hard self-improvement theory | abs | A |
| 2604.07822 | Kohli 2026, Loop, Think & Generalize | abs | A |
| 2507.10524 | Bae 2025, Mixture-of-Recursions | full | B |
| 2511.08577 | Fu 2025, Think-at-Hard | full | B |
| 2609.35748 | You 2026, Adaptive Looped Transformers (TaH2) | full | B/E |
| 2404.02258 | Raposo 2024, Mixture-of-Depths | full | B |
| 2310.07096 | Tan 2023, Sparse Universal Transformer | full | B/C |
| 1603.08983 | Graves 2016, ACT | full | B/E |
| 2107.05407 | Banino 2021, PonderNet | full | B/E |
| 2506.21734 | Wang 2025, HRM | full | B/E |
| 2510.04871 | Jolicoeur-Martineau 2025, TRM | full | B/E |
| 2402.00976 | Ray Chowdhury 2024, Recurrent Tr. w/ Dynamic Halt | full (§6) | B/E |
| 2405.16039 | Csordás 2024, MoEUT | full | C |
| 2605.09403 | Smithline 2026, Sparsity Moves Computation | full (grep) | C |
| 2410.13964 | Zhao 2024, Sparse MoE for compositional gen. | abs | C |
| 2609.35751 | Wang 2026, How to Loop MoE (Foil) | abs | C |
| 1904.12774 | Rosenbaum 2019, Routing Networks: challenges | abs | C |
| 2206.02713 | Mittal 2022, Is a Modular Architecture Enough? | abs | C |
| 2010.02066 | Csordás 2020, Are Neural Nets Modular? | abs | C |
| 2409.15647 | Fan 2024, Looped Tr. for Length Generalization | full | D/E |
| 2609.33144 | Liang 2026, Beyond the Training Horizon | full | D |
| 2608.31067 | Ito 2026, UT for Circuit Computations | full | D/E |
| 2110.07732 | Csordás 2021, Neural Data Router | full | D |
| 2310.16028 | Zhou 2023, What Algorithms Can Tr. Learn? | full | D |
| 2410.02140 | Huang 2024, Formal Framework for LG | full | D |
| 2305.19466 | Kazemnejad 2023, NoPE | full | D |
| 2410.01104 | Veličković 2024, Softmax is not Enough | full | D |
| 2501.19399 | Nakanishi 2025, Scalable-Softmax | full | D |
| 2202.12172 | Chiang 2022, Limitation of Self-Attention | full | D |
| 1704.06611 | Cai 2017, Generalize via Recursion | full | D/E |
| 2607.18759 | Singh 2026, Relative Generalize, Absolute Memorize | full | D |
| 2606.29983 | Kuo 2026, Learned Stochastic Stopping | full | D/E |
| 2305.16843 | Ruoss 2023, Randomized Positional Encodings | abs | D |
| 2405.17399 | McLeish 2024, Abacus embeddings | abs | D |
| 2405.18719 | Golovneva 2024, Contextual Position Encoding | abs | D |
| 2203.16634 | Haviv 2022, NoPE LMs still learn positions | abs | D |
| 2305.15408 | Feng 2023, Mystery behind CoT (Arithmetic(n,p)) | abs | task |
| 2510.25741 | Zhu 2025, Ouro | full (PDF) | E |
| 2609.03379 | Wang 2026, RecurTrace | full | E |
| 2607.20519 | Popescu 2026, Diagnosing Learned Halting Gates | full | E |
| 2607.14427 | Logan 2026, Per-Token Fixed-Point Convergence | full | E |
| 2606.18206 | Movahedi 2026, Fixed-Point Reasoners | full | E |
| 2602.08864 | Moosa 2026, Dynamic Compute Allocation (ANIRA) | full | E |
| 2207.07061 | Schuster 2022, CALM | full | E |
| 1511.06279 | Reed 2015, Neural Programmer-Interpreters | abs | E |

---

## A. Learning a step skill, then composing it

**Altabaa 2510.14095** (from scratch; ≤ 4 layers, ≤ 384 dimensions; DAG modular arithmetic, p = 23; N ≤ 32 → 128).
- **Per-step loss** (Eq. 3/5): "Σ_t Σ_i 1{Depth(x_i) ≤ t}·ℓ(W_value E_i^(t), Value(x_i))". The linear readout is shared and the loss is summed over 4 factors.
- **Snap** (§4.1): states "are projected into a structured, discrete symbolic space and then immediately re-embedded". B.3: this "enables us to train the model with a type of teacher-forcing across recurrent iterations".
- **Corruption** (B.4): "with some small probability, we corrupt a random selection of the value components". The probability is not stated. T is set from the known depth (an oracle).
- [EMPIRICS] Percent fully solved at N = 128 (Fig. 4 SVG):

  | variant | N = 128 |
  |---|---|
  | snap + correction | 100 |
  | snap only | 94.8 |
  | continuous supervision | 54.8 (94.3 at N = 32) |
  | CoT | 0 (from N = 64) |

  - Correction needed 4 layers, not 2. Relative PE was "by far" the best.
  - Negative: without corruption, "no further computation … after the number of iterations reaches the depth". Seeds are not stated.

**RELAY 2502.08482** (from scratch, 3 layers, RoPE).
- Iteration t predicts CoT round t ("3×2+6÷3=" → "6+6÷3=" → "6+2=" → "8").
- L = L_ans + λ·L_iter, with λ not stated.
- [EMPIRICS] At 25 operators (train ≤ 15): 0.98 vs 0.77 vanilla loop vs 0.15 autoregressive CoT. Modulus and seeds are not stated.

**Bansal 2202.05826** (from scratch, conv nets).
- Progressive loss (Alg. 1): "Choose n ∼ U{0, m−1} and k ∼ U{1, m−n}". The first n iterations run without gradient, then k iterations with gradient, so the step is trained from the model's own state at a random depth. Recall concatenates the input every iteration.
- [EMPIRICS] Prefix sums 32 → 512 bits: 97.1 ± 1.9 vs 0% for plain DT; with n fixed at 0, 90.3.
- Negatives: a loss weighted up with iteration (72.5 / 0); about half the chess runs fail to converge.

**Teacher forcing and single-step training (negatives).**
- Generalist 2209.11142 (GNN, 16 → 64) removed hint teacher forcing, "aligning training with evaluation, and avoiding the network becoming overconfident in always expecting correct hint predictions". [EMPIRICS] "feeding ground-truth hints at training time hurts generalisation" (the largest ablation effect).
- DRM 2604.18839 (ARC-AGI): one-step denoising "substantially underperforms", because test-time states "differ substantially from the noised-target states seen in training". Their fix: corrupt the target, unroll k steps, L = E_τ[−log P(Y | H_k)] (Eq. 6). [EMPIRICS]

**History vs Markov.**
- Bohde 2403.04929 [DERIVATION]: historical embeddings "intrinsically contradict the Markov nature".
  - G-ForgetNet: z = [x̄, g⊙h^(t−1)], with an L1 penalty on g.
  - [EMPIRICS] The gate is open early in training and near 0 at the end. 30/30 CLRS tasks beat baseline (82.9 vs 76.0). Pure ForgetNet is unstable early in training.

**Discrete control.**
- Discrete NAR 2402.11628: one-hot states and hard attention. [EMPIRICS] 100% from 16 → 1600 nodes. [DERIVATION] The "guarantee" is exhaustive unit tests of the finite transitions.
- NEE 2006.08084: subroutines trained one at a time on intermediate states, then composed; state held in attention masks. [EMPIRICS] 100% at sizes 25–100. Plain transformers fail beyond 8 numbers ("difficulty separating 'what' … from 'where'").

**Invariance augmentations.**
- Hint-ReLIC 2302.10258 assumes Y_{t+1} ⊥ X^c_t | X^s_t, i.e. a step depends on a subset of the state. [THEOREM 4.1] Invariance to a refinement implies invariance for the coarser target. Augmentations add nodes that leave the current step unchanged. [EMPIRICS] Heapsort 32 → 95. Negative: no-hint training beats it on BFS and binary search.
- The no-hint contrastive variant is 2306.13411.

**Lock-in and pace.**
- Wu 2609.39892 [EMPIRICS]: a frozen loop runs different transitions depending on its *entering* state, routed through attention patterns. Stepwise supervision locks in one hop: two-hop steering 13.2 vs 50.5 for final-only; Ouro 7.8 vs 98.0.
- Zhang 2607.20594 [EMPIRICS] (from scratch, group prefix products): pace "v ≈ n_tr/T_tr (exponent 0.98 ± 0.04)". Extra loops rescue late positions; over-long T makes tight-budget models overthink.

**Curricula** (abs).
- Algorithm alignment: Xu 1905.13211 argues sample efficiency improves when the network's structure aligns with the algorithm's steps.
- Self-improvement on filtered harder instances: Lee 2502.01612 (10 → 100-digit addition; a pretrained init accelerates it).
- Easy-to-hard: Liu 2602.10014 [THEOREM] gives conditions under which easy-to-hard beats fixed mixtures.
- Kohli 2604.07822: depth extrapolation with more inference loops; three-stage grokking.

**New for us.**
- All of the above is from scratch (≤ 4 layers or a GNN).
- None uses attention over past iterations, except CoTFormer, which is negative.
- None learns its halt.
- Nested depth 8 → 16 is untested (RELAY reaches 15 → 25 operators).
- MCT's minimum-norm correction in R's row space is a continuous snap. Altabaa's 54.8% continuous-supervision arm is the warning.

**Leads** [CONJECTURE]:
- (i) **SNAP arm:** straight-through re-embedding of R's argmax decode into V_t's recode subspace, plus value-slot corruption at small p. Needs a derivation for the manifold shift.
- (ii) **Bansal start:** run t₀ passes detached, then supervise k passes, so the step trains from its own states at every depth.
- (iii) Keep p_TF → 0, as planned.

## B. Adaptive compute inside a pass

**MoR 2507.10524** (from scratch LMs, 135M–1.7B).
- Expert-choice: "each recursion depth becomes an expert and selects their preferred top-k tokens", with a linear router on H^r. Token-choice fixes the depth from H^1.
- Losses: auxiliary BCE (causality), balancing, z-loss.
- [EMPIRICS] Failures:
  - an auxiliary router with "positional bias" (66.7% dead tokens);
  - token-choice imbalance (MaxVio up to 1.28);
  - routers "almost perfectly separated … challenging to adjust top-k values after training".

**MoD 2404.02258.** Expert-choice top-k at 12.5% capacity; the top-k is "non-causal" (fixed by an auxiliary predictor); random routing is "drastically worse". [EMPIRICS]

**TaH 2511.08577** (pretrained Qwen3 0.6–4B).
- Oracle d = 1 + 1[t̂ ≠ t] from a fixed reference model; an imitation-trained decider.
- [EMPIRICS] "the second iteration corrects 8.7% of predictions, it also flips 2.1% of correct ones into errors".
- Restricting attention to the current depth: −8.5. An on-policy oracle collapsed training.

**TaH2 2609.35748** (pretrained 1.7–8B).
- Gain labels δ = ℓ^(m) − ℓ^(m+1), trained jointly.
- [EMPIRICS] "21.7% and 15.9% [of tokens] become worse" (Ouro / Huginn). The Ouro gate: "every token exits after the first iteration". Raising the ceiling from 8 to 12 gives no gain.

**SUT 2310.07096** (from scratch).
- Stick-breaking halt "α_l = α̂_l ∏_{l'<l}(1−α̂_{l'})", with an ACT depth penalty and an MI balance loss.
- [EMPIRICS] Logical inference ≤ 6 → 7–12 operators: 98 → 81, vs about 51 for a Transformer. Halting depth rises with operator count. Compute saved is about 50% "regardless of the threshold". Negative: SCAN length 7%.

**HRM / TRM** add inner L-cycles per outer step, the "inner loop per step" pattern.

**Compute-trigger signals:** a linear hidden-state score (MoR, MoD), a pass-1 mismatch (TaH), the loss gain (TaH2), or a halt MLP plus a depth penalty (SUT, ACT, PonderNet). None reads task structure.

**New for us** [CONJECTURE]: per-slot inner steps keyed on R's "reducible slot" decode. This is NEE's mask, read from a linear recode.

**Negatives:** router positional bias, on-policy oracle collapse, and no gain past the trained depth.

## C. Operator routing and specialisation

- **SUT** [EMPIRICS, one model, PMI only]: "the or operator is dominating expert 9".
- **Smithline 2605.09403** [EMPIRICS, 1-layer models, add-with-carry, mod-113 addition]. **Negative:** "frozen random routing nearly matches learned routing"; "On modular addition, routing shows no correlation with input structure".
- **Zhao 2410.13964** (abs): the optimal number of active experts grows with task complexity.
- **MoEUT 2405.16039:** no algorithmic results; layer-1 routing "mainly depends on the token's identity".
- **Foil 2609.35751** (abs): looped MoE; "routing confidence tracks healthy expert use better than load balance".
- **Wu 2609.39892:** shared loops already route *implicitly*, by entering state through attention.
- **Classic negatives** (abs): routing training pathologies (1904.12774); learned modular systems are sub-optimal (2206.02713); networks "fail to reuse submodules" (2010.02066).

**New for us** [CONJECTURE]: a label-derived hard route, where R's decoded operator at the reducible slot selects a core expert (FFN or LoRA). It has NEE and Discrete NAR support, and needs a frozen-random-route control and a derivation. Learned MoE routing has a direct negative on modular arithmetic.

## D. Depth- and length-invariant step rules

**Size-keying mechanisms:**

1. **Absolute positions.**
   - [THEOREM] (2607.18759 Lemma 1) RoPE makes logits depend on offset alone, so the rule transfers. Prop. 2: with learned absolute PE, "the trained head pins to a fixed absolute position inside the training range".
   - RASP-L disallows index arithmetic (2310.16028; its length-generalisation claim is a "phenomenological" conjecture).
   - Huang [THEOREM 7] covers only "Periodic and Local" translation-invariant terms, for an idealised single-pass learner.
   - **GPT-2's wpe is learned and absolute. PURE-ANNEAL queries W_q·wpe[j] by design.**
2. **Total length in the readout.**
   - [EMPIRICS] Liang 2609.33144 (from scratch, NoPE, Fan-style, trained n ≤ 12): "longer prompts impair the recurrent [EQ] state even at shared hop indices". Transplanting the length-20 state into a length-30 run takes hops 12–19 from 5.5% to 83.6%, since "[EQ] can attend to the entire prompt".
   - The progress cue decays (28 → 9.1). No fix was tested.
   - Aside: the m = 11 polynomial task generalises where other primes fail.
3. **Dispersion.** [THEOREM] (2410.01104 Thm 2.2) With finite features, "attention coefficients must disperse in all global Transformer heads" (Θ(1/n)). The paper's own fix is "ad-hoc". In VB, both the version (T grows with n) and the bank (t·T entries) disperse.
4. **Count leakage.** [DERIVATION] (2202.12172) A learned head carries a (k − n/2)/(e^c + n − 1) term. The fix is logit × log n; SSMax 2501.19399 uses softmax((s·log n)·z).
5. **Loop–n coupling.** [EMPIRICS] Kuo 2606.29983: "spurious correlation … between sequence length and number of loops". The budget law (2607.20594) gives the same coupling.
6. **History.** [EMPIRICS] 2609.19521 (from scratch, Rule 30, trained to depth 12):
   - at depth 20, full-cache CoTFormer scores 50.1%, recent-2 through recent-8 are near chance, and recent-1 scores 99.4%;
   - "retained history can pull a corrected trajectory back toward failure";
   - clearing the cache after clean injection adds 31–41 points.

**Fixes on nested arithmetic:**
- **Ito 2608.31067** (from scratch, 924 parameters; mod-10 +/×, fully bracketed).
  - PE: cumulative bracket sums "A_1 = (L X G_PE)1" forward and backward; a hardmax "selects exactly the tokens belonging to the deepest reducible subexpressions".
  - [THEOREM 1–2] Each iteration reduces all deepest spans (δ → δ − 1), and the model "halts autonomously … when there remain no parentheses". Assumes hardmax, straight-through discrete states and fully bracketed input.
  - [EMPIRICS] Trained at depth 1–2: 10/10 seeds, 100% to depth 30.
  - Fan-style looped transformer: 1.00 / 0.10 / 0.05 at depth 2 / 3 / 10 (chance 0.10); RoPE and FIRE also collapse.
- **NDR 2110.07732.**
  - Copy gate "h^(t+1) = g ⊙ u + (1−g) ⊙ h^(t)" (bias −3).
  - Geometric attention "A_{i,j} = P_{i,j} ∏_{k∈S_{i,j}} (1−P_{i,k})", sigmoid with no normaliser.
  - [EMPIRICS] Nested mod-10 arithmetic, depth ≤ 5 → 7–8: 0.98 vs 0.47 for a Transformer and 0.77 with relative PE. Both parts are needed. The gate makes extra steps harmless.
- **Fan 2409.15647** (NoPE; "T(n) … is available in the training data"). Parity 20 → 40+. Its own limitation: it "requires the ground-truth number of steps".
- Ruoss (randomised positions), Abacus (random offsets) and CoPE (content-counted positions; Ito's PE is one case) are the positional alternatives (abs).

**Proofs.**
- None exists for an SGD-trained looped step.
- Guarantees exist only:
  - by finite verification: Cai 1704.06611, "we only need to test a finite number of inputs to show that the model will work correctly on inputs of unbounded complexity", which needs a step with a bounded, size-independent input domain plus traces;
  - by learned parameters matching a proved construction (Ito);
  - for an idealised learner (Huang).
- NoPE [THEOREM 1–2] can recover absolute positions, so NoPE alone does not remove a size cue (also Haviv 2203.16634).

**New for us** [CONJECTURE]:
- a structural position in a pretrained core (bracket depth or reducible span from R(V_{t−1}), offsetting wpe);
- a size-free bank read (sigmoid or geometric, log-n scaled, or {V_0, V_{t−1}} only);
- a Cai-style finite step (operator, two values, mod 11), so an exhaustive 4 × 11 × 11 STEP-CAP table is a verification.

## E. Learned halting that reads task state

**Definitions.**
- PonderNet: "p_n = λ_n ∏_{j<n}(1−λ_j)", with L = Σ p_n·𝓛_n + β·KL(p ‖ p_G(λ_p)).
- ACT: 𝓛 + τ·P(x).
- Ouro:
  - Stage I: Σ p(t|x)·L(t) − β·H(p).
  - Stage II (LM frozen): I(t) = max(0, L(t−1) − L(t)), w = σ(50·(I − 0.005)), with BCE.
  - Exit at CDF ≥ q.
- RecurTrace: "y_t = 𝟙[min_{t′>t} 𝓛_{t′} < 𝓛_t − δ]".
- TRM: "binary_cross_entropy(q_hat, (y_hat == y_true))".
- HRM: Q-learning with Ĝ_halt = 𝟏{ŷ = y}.
- Kuo: hazard σ(q(Pool(H))); REINFORCE with R = −ℓ at one sampled τ per example.
- Fan: "T = argmin_t 𝓛(f_t(x), ŷ^t)", run to T_max (not online).
- FPRM: residual ‖z − f(z)‖∞/‖f(z)‖∞ < 0.1.
- Logan: successive-output KL < ε.
- Ito: no parentheses remain.
- CALM 2207.07061: softmax top-2 gap, state cosine, or a linear classifier, with Learn-then-Test thresholds.
- NPI 1511.06279 (abs): an end-of-program probability trained from traces. This is the closest *supervised* "done" target.

**Transfer to unseen depth** [EMPIRICS].
- PonderNet (RNN, parity, 1–48 → 49–96 bits): "almost perfect" while ACT stays at chance; steps about 3 → 5.
- Kuo (from scratch, n < 20 → 20–60; 32 seeds on addition, 18 on the other tasks):

  | task | K = n | K = n ± 5 | RL-Halt |
  |---|---|---|---|
  | Addition, mean OOD (std) | 22.1 (9.9) | 30.9 (5.9) | 45.0 (2.7) |
  | Dyck-1 | 66.6 | 88.2 | 97.5 |

  - Copy fails: it "can also stabilize a suboptimal computation".
- FPRM's training-free residual: A5/S5 32 → 128 at 98%. Ito: exact to depth 30.
- A global-mean UT halt on ListOps is still "poor in out-of-distribution generalization", where "higher noise from dense softmax in longer contexts" is the suspected cause (2402.00976).
- Pretrained models:
  - RecurTrace interpolates only, and 97.3% of items stop at the 2-loop floor;
  - TaH2 gains nothing 8 → 12;
  - Ouro degrades past T = 4;
  - ANIRA 2602.08864: "models fail to extrapolate to unseen input sizes despite allocating additional computation".

**Readouts beat joint gates.**
- Popescu 2607.20519 (from scratch; mod-23 prefix arithmetic and parity; in distribution only): a fixed-prior depth loss "𝓛_fixed = E[Σ π_t ℓ_t]" with a post-hoc margin readout reaches 95% at depth 2.19 (geometric prior: 1.38). Joint gates need 4.49 (2.80). "the main issue is the trajectory induced by the joint objective".
- Logan 2607.14427: the learned router "requires nearly full depth and yields no reduction", while the convergence halt saves 38%.

**No proof** that a learned halt transfers; only Cai's verification and Ito's construction give guarantees.

**New for us** [CONJECTURE]: HALT-R. Stop at the first pass where R(V_t) decodes no live operator or bracket slot, with a convergence fallback. It is the learned-recode analogue of Ito's rule. It cannot fire under STALL, because a copied pass leaves operators alive.

Requirements:
- (a) R must not carry the pass index. Decouple K from n (Kuo's window; a Bansal detached start). K_i = τ_i + U{0..4} keeps K correlated with n.
- (b) Validate R's "none-left" reading at n 9–16 first.
- (c) Use it as a post-hoc readout, not a jointly trained gate.

---

## Failure modes and VB analogues

| failure mode | evidence | VB analogue | check |
|---|---|---|---|
| Collapse to one loop | ACT high τ "reduces to 1" (1603.08983); ACT and PonderNet at 1.00 loops, frozen Qwen3 (2609.03379); Ouro gate exits at pass 1 (2609.35748) | early answer-settling halts | no joint halt with a depth penalty; post-hoc readout |
| Collapse to max depth | Ouro without entropy "collapses p onto t = Tmax"; TRM + ACT on S5 exhausts the budget (2606.18206) | always run to K_max | report the mean exit, censored |
| Counter-like halt | pace n_tr/T_tr and T* = ⌈n/v̂⌉ (2607.20594); correctness and gain labels only up to the trained depth | "done" as a pass counter at n = 8 | STALL; decouple K from n; HALT-R |
| Overthinking | 2.1% of correct predictions flipped (2511.08577); 21.7% of tokens worse (2609.35748); no recall ⇒ collapse (2202.05826) | DILUTION past K | copy gate / hold; STABLE at K′ 24 |
| History pulls back | full cache 50.1 vs recent-1 99.4 (2609.19521); history vs Markov (2403.04929) | bank at every core layer | arm: bank = {V_0, V_{t−1}} vs full, n 9–16 |
| Size-keyed readout | 5.5 → 83.6 transplant (2609.33144); Θ(1/n) (2410.01104); APE pinning (2607.18759) | DEPTH 0.5–0.7 at n 9–12 | offsets / structural PE; log-n bank scale |
| Soft-anchor drift | continuous 54.8 vs snap 100 at N = 128 (2510.14095) | CARRY-NO-STEP at depth | SNAP arm + corruption |
| Teacher-forcing reliance | GT hints "hurts generalisation" (2209.11142); one-step denoising underperforms (2604.18839) | TF-ONLY (N3) | p_TF → 0; train from own states |
| Stepwise lock-in | two-hop 13.2 vs 50.5 (2609.39892) | R2 forbids two reductions per pass | let the target accept ≥ 1 reduction |
| Router pathologies | positional bias (2507.10524); random ≈ learned on modular addition (2605.09403) | operator-routed core | label route + random-route control |

**Contradiction to resolve.**
- Access to earlier depths helps in distribution (TaH: −8.5 without it).
- It hurts extrapolation (LSTM-UT; Markov NAR).
- VB's DEEP reading is in distribution and its DEPTH reading is out of distribution, so a recent-1 bank arm separates the two [CONJECTURE].
