# LIT: iterated self-training and the extrapolation margin across rounds

Written 2026-10-17 23:40 EDT (from `date`). Literature subagent; read-only web research.
Every arXiv ID was checked against export.arxiv.org/api/query on this date; title and authors match.
Where a number says "read from figure", it was read off the rendered SVG of the arXiv HTML version, not from text.

## 1. Lee, Cai, Schwarzschild, K. Lee, Papailiopoulos — Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges — arXiv 2502.01612 (verified; ICML 2025) [empirics]
Claim: iterated self-labelling on a +1-difficulty-per-round curriculum extends small transformers far past their labelled range, but harder tasks need filtering.
Assumptions: 14M NoPE LLaMA-style model trained from scratch (Llama-1B/3B with LoRA only in §7.3). Problems are generated with a known difficulty setting. Greedy decoding, exact-match scoring. Each round trains on the gold data (D0) plus all earlier self-labelled sets, with the newest set upsampled to 50%.
- **Schedule** (§4, §6, App C.3.2): +1 per round in digits for addition and copy, +1 hop for mazes (N=30, from ≤9 hops), +3 nodes for mazes by node count. Multiplication starts from 5×5 and adds one digit to one operand per round, sweeping m=1..n+1; reaching 10×10 takes 41 rounds. Each round uses 50k self-labelled items and 500–5000 fine-tuning steps.
- **Results:** reverse addition is near-perfect to 100 digits (Fig 3). The abstract says "10-digit to 100-digit", but §4.1 says training covered 1–16 digits. Forward addition with a length filter stays above 98% up to length 70 after 60 rounds (Fig 10). Multiplication with majority vote plus length filter is near-perfect to 9×9 at round 31 (Fig 13), and the accelerated schedule reaches 10×10 in 19 rounds (Fig 19). Mazes with majority vote reach 30 hops (Fig 14).
- **Filters** (§5.2):
  - Length filter: drop outputs shorter than the batch maximum by more than a threshold (2 tokens for forward addition, 10 for multiplication).
  - Majority vote for multiplication: 5 seeds trained independently, keep an item when at least 4 of 5 outputs are identical. This raises 5×6 data accuracy from a 31% mean to 93.3% (Fig 7).
  - Majority vote for mazes: 3 of 3 must agree.
  - Each seed must generate its own self-labelled data; a shared self-labelled set does worse (App B.2.2, Fig 31).
  - On mazes, majority vote is about as good as an oracle verifier, and checking every move beats checking only the end node (§6.3.3, Fig 16).
- **No filter:** forward addition degrades within the 10 rounds shown (Fig 9), multiplication reaches only 13.7% on 6×6 at round 7 (§6.2), and mazes collapse (Fig 14). Multiplication needed majority vote and length filter together (Fig 12).

**Margin across rounds** (directly on the H_shift / H_full question)
- §7.1, Fig 17 (read from figure): the count of extra lengths at >99% accuracy past the trained length, under the +1-per-round schedule.
  - Copy: about 3 at round 0, rising to about 35–45 by rounds 80–93, roughly linear in the round number.
  - Reverse addition: about 3 for rounds 1–15, about 4 for rounds 20–40, about 5 (range 3–8) for rounds 50–95. This is near-constant with slow growth.
- The "exponential" claim (abstract, §7.2) comes from an accelerated schedule that each round samples every length with >99% *test* accuracy. That is oracle information, and the authors flag it.
- Fig 18 (read from figure): copy turns superlinear, from about 30 at round 20 to about 184 at round 48. Reverse addition goes from about 25 at round 10 to about 100 at round 48, versus 63 under +1 per round. That is about 2 digits per round, roughly linear, not exponential.
- Pretrained Llama-1B/3B extrapolate further per round and accelerate more than the 14M model trained from scratch (§7.3, Figs 20–21).

**Error accumulation** (§8)
- **Error avalanche** (Fig 5, reverse addition): once self-label accuracy drops below about 0.9, accuracy at the next length falls below y=x. In Fig 22 (left), a failing run goes 0.90→0.77, 0.78→0.59 and 0.62→0.41 over rounds 7–9.
- **Injected noise** (Fig 24, read from figure): at round 5, 5% structured noise (±1 on the last digits, or dropped digits) cuts round-10 accuracy to about 0.6–0.75. At round 20, the model tolerates up to about 50% noise. Structured noise does more harm than uniform noise.
- At equal label accuracy, the model's own biased errors hurt more than random labels (App B.4, Fig 33).
- Adding more low-quality self-labelled data makes things worse (§8.3, Fig 26).
- Models memorise their own wrong labels (Fig 25).
- The dominant out-of-distribution error is an output that is too short or drops a CoT step (Fig 6).

## 2. Fan, Du, Ramchandran, K. Lee — Looped Transformers for Length Generalization — arXiv 2409.15647 (verified; ICLR 2025) [empirics + constructive propositions]
Claim: a looped decoder block trained end-to-end with loop count T(n) learns steps that generalise to longer inputs; running more loops at test time gives extrapolation.
Assumptions: the task is n-RASP-L, i.e. P_pre ∘ (P')^{T(n)} ∘ P_post (Def 3.1). T(n) is known for every training item. Full-output prediction rather than next-token prediction. NoPE, input injection, GPT-2 block, length curriculum.
- **Constructions** (Props 3.2–3.4): parity T(n)=n, copy T(n)=n, addition T(n)=n+1.
- **Results** (§6.2, Fig 4): parity trained on ≤20 is near-perfect beyond 40. Addition and copy stay near-perfect at the training maximum +10, where next-token-prediction baselines fail.
- **Adaptive steps** (§4.3, Eq 2): either use the oracle T(n), or use "maximum confidence": run to T_max and pick the step t that minimises the cross-entropy of the model's own decoded output, over a batch of same-length items or per sample. The per-sample rule nearly matches the batch rule on tasks whose iterates converge after solving (addition, copy, multiplication, unique set). It is worse on tasks that don't converge (parity, binary sum) (§6.4).
- The paper does not use self-training or multiple rounds.

## 3. Huang, Wen, Singh, Chi, Chen — Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization — arXiv 2511.07378 (verified) [theorem + small empirics]
Claim: recursive self-training on the model's own greedy CoT traces provably doubles the solvable length at each stage. A model trained to L_k = 2^k reaches accuracy 1−1/poly(d) at 2^{k+1} (Thm 5.1), and Θ(log d) stages reach the maximum length (Cor 5.1).
Assumptions:
- One-layer NoPE transformer.
- LEGO state tracking with S_n acting on Z_n, with n_y = Θ(log log d / log log log d).
- Gradient descent on the population loss.
- The FFN (the one-step rule) is frozen after stage 1; later stages train only attention, each until loss < d^{-E1}.
- Self-labels are greedy and unfiltered; the theorem makes them near-exact.
- **Mechanism:** attention concentrates on the right clause. For symmetric actions it is too weak for long contexts but strong enough for 2× (Thm 1.2: only constant-factor generalisation without self-training).
- **Experiments** (§7, Fig 1b): self-labelled stages L=5→10→20→40. Each stage generalises at least a constant factor past its training length, and the final model is near-perfect up to L=160.
- **Relevance:** predicts a margin proportional to the trained length (multiplicative), not a fixed +k per round.

## 4. Yang, Huang, Wang, Mirzasoleiman — Learning to Extrapolate in Depth with Looped Transformers — arXiv 2610.14456 (verified; posted 2026-10-10) [theorem]
Claim: a looped block trained at depth K and then frozen and unrolled to T>K works only inside a finite window, Hard(G) < T < T_max(G;K) (Prop 1). The guaranteed extra depth is T−K = Θ(H·3^{K−H}) for training hardness H (Thm 2).
Assumptions: all-pairs graph connectivity, exchangeable training distribution, symmetric positive initialisation, population gradient flow (which reduces to one effective parameter, Thm 1), spectral regularity.
- If K−H = O(1), the extra depth is Θ(H): the margin is proportional to training hardness. Training with more loops than the training hardness widens the window exponentially.
- Too many test loops create false connections, the same overthinking failure as in §5 (Fig 2).
- **Relevance:** no self-training, but it is a second theorem in which the margin scales with training hardness, and the loop budget relative to hardness sets that margin.

## 5. Kohli, Parthasarathy, Sun, Yao — Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers — arXiv 2604.07822 (verified) [empirics]
Claim: recurrent-depth transformers extrapolate to deeper k-hop compositions when inference recurrence is raised. The maximum training recurrence sets the range, and overthinking caps it.
Assumptions: synthetic knowledge graph with 200 entities and 10 relations; curriculum that advances at 95% accuracy per hop level; NoPE; training from scratch.
- **Range** (§6.3, Fig 7): all models trained on ≤12 hops. R=6 reaches 14 hops, R=8 reaches 19, and dynamic recurrence reaches 19.
- **Overthinking** (Fig 8): the logit margin rises and then falls as iterations increase, and the peak is lower for harder items. Dynamic recurrence decays more slowly. More than 15 iterations does not help out of distribution.
- **Learning speed** (§6.2, Fig 6): once the rule is learned, each new hop level needs few steps. From a 20-hop checkpoint, 21 hops took as few as 50 steps when training only on 21-hop data.
- **Halting** (Figs 9–10): stopping on KL(p_t‖p_{t−1}) < ε alone (the rule from Geiping et al.) halts too early. Requiring the entropy H(p_t) to be below a threshold as well works better.

## 6. Kuo, Chayti, Reizinger, Brendel, Jaggi — Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping — arXiv 2606.29983 (verified) [empirics]
Claim: how far looped transformers extrapolate depends heavily on the seed, because loop count is spuriously correlated with length. Random loop counts in training reduce run-to-run variance out of distribution.
Assumptions: binary addition, Dyck-1, unique set, copy; trained from scratch; evaluated with the best loop count chosen per length (oracle).
- **Variance** (§4.1): out-of-distribution standard deviation across runs is 0.087 (K=20) and 0.122 (K=n), versus 0.009 and 0.005 for non-looped baselines. Some runs extrapolate far while others collapse just past the training range. Larger K shifts mean extrapolation outward but raises variance (Fig 2).
- **Random loop counts** (§4.2): sampling K from a window around n reduces variance. Learned stopping (RL-Halting) improves the accuracy–stability trade-off but can lock in a suboptimal computation.

Also verified, one line each:
- Shafayat et al., arXiv 2505.21444 [empirics]: RL with a majority-vote self-reward climbs a Reasoning Gym difficulty curriculum (Knights & Knaves trained with gold labels at level 2, reaching near 100% at level 9). Long training with self-reward then collapses suddenly to a fixed template answer (§4.2, Fig 6).
- Geiping et al., arXiv 2502.05171 [empirics]: a 3.5B recurrent-depth LM trained on 800B tokens improves with test-time recurrence and uses a KL-based adaptive exit. No self-training.
- Cai et al., arXiv 2506.09251 [empirics]: length generalisation transfers from a longer related task trained jointly. Not self-training.

## Implications for round 2
The numbers come from the brief. Arithmetic marked [derivation] is mine; its assumption is that the accuracy curve keeps its shape and that crossings are linear interpolations of single-seed accuracies.

**Increments**
- Lee adds one difficulty unit per round. Even the accelerated schedule self-labels only lengths that already have ≥99% test accuracy.
- Our round 1 added 4 (n 9–12) at base raw accuracies of 0.94/0.58/0.26/0.16 and still succeeded. So our per-step filter is doing the job that Lee's "safe range" does.
- Round 2 on n 13–16 has raw accuracies of 0.999/0.893/0.633/0.344, a profile similar to round 1. A +4 step stays within what round 1 already showed works.
- If the filter is close to exact, yield ≈ raw accuracy, so n=16 keeps about a third of drawn items. Oversample per n so the kept counts per n are roughly equal.

**H_shift vs H_full**
- No source shows the boundary stopping binding (H_full) for an arithmetic-like task.
  - Lee's reverse-addition margin stays at about 3–5 digits over about 90 rounds (Fig 17).
  - Only copy's margin grows, and the exponential curve needs oracle schedule information (Fig 18). Addition then stays at about 2 digits per round.
- Two theorems (Huang 2511.07378 Thm 5.1; Yang 2610.14456 Thm 2) instead predict a third option, **H_prop**: the margin is a constant fraction of the trained length or hardness.
- Our two points fit H_prop [derivation]:

  | Accuracy level | Base model (labelled to 8) | Round 1 (labelled to 12) |
  |---|---|---|
  | 50% crossing | n 10.25, margin 2.25, ratio 0.28 | n 15.46, margin 3.46, ratio 0.29 |
  | 90% crossing | n 9.11, margin 1.11, ratio 0.14 | n 13.93, margin 1.93, ratio 0.16 |

  The margin grew ×1.5–1.7 while the labelled range grew ×1.5. Caveat: base → round 1 is not a pure self-training round-to-round comparison.
- Predicted crossings for round 2 (labelled to 16) [derivation]:

  | Hypothesis | 90% crossing at n | 50% crossing at n |
  |---|---|---|
  | H_shift (round-1 margins held constant) | ~17.9 | ~19.5 |
  | H_prop (ratios 0.14–0.16 and 0.28–0.29) | ~18.2–18.6 | ~20.5–20.6 |
  | H_full | ≥0.9 everywhere in 17–20, with the plateau continuing | — |

- Testing n 17–20 separates H_full from the other two. To separate H_shift from H_prop, test **n 17–24** to locate the 50% crossing.
  - Near 50% the curve drops about 0.29 per n, so 200–500 items per n give a crossing SE of about 0.1 n.
  - Seed variance is the larger error: Kuo reports 0.09–0.12 across-run SD for looped models. Run at least 2 seeds per round before ruling on the hypothesis.

**Filters**
- Keep the exact-n-steps check. It is our version of Lee's length filter, which targets the dominant failure: dropped steps or short outputs (Fig 6).
- Keep the per-step compose check. Lee's per-move verifier beat end-only checks (Fig 16).
- If compose-valid recomputes each step exactly, kept labels are correct and label noise is not the risk. If it only checks form or self-consistency, add a vote across at least 2 independently fine-tuned seeds, each generating its own self-labelled data (Lee App B.2.2).
- Keep gold n≤8 and the round-1 self-labelled 9–12 items in the mix, upsampling the newest round (Lee: D0 + all previous rounds, newest at 50%).

**Expected failure modes**
1. **Selection bias.** At n 15–16 the filter keeps only the items the model already solves, likely skewed toward easier substructures. Lee App B.1 notes that hard cases (long carry chains) are rare under random sampling. Compare the structural features of kept versus drawn items; reweight if they differ.
2. **Error avalanche** if wrong labels slip through. Early rounds are fragile: 5% structured noise at round 5 cut later accuracy to about 0.6–0.75 (Lee Fig 24). Biased errors do more harm than random ones (Fig 33), and more low-quality data hurts (Fig 26). Audit a sample of kept n=16 labels against a gold computation.
3. **Loop window / overthinking.** If inference recurrence has to grow with n, expect a finite window (Yang Prop 1; Kohli Fig 8). Sweep the loop count at n 17–24 instead of using one fixed count. The training loop budget relative to hardness sets the margin (Yang Thm 2; Kohli §6.3).
4. **Seed variance** may decide the H_shift/H_full reading if only one seed is run (Kuo §4.1).
5. **Collapse after many self-training rounds** is documented for RL with self-reward (Shafayat), not for filtered SFT with gold anchors as Lee uses. The risk is low over 2 rounds, but track in-support (n ≤ 12) accuracy every round.

## Addendum (2026-10-18): do the cited theorems give the margin as "proportional to trained length", and with what constant? (Audit, after §216 R2-s1)
- **The data, per Audit:**
  - n50 / support = 1.29 / 1.29 / 1.28 for D3-T / R1 / R2 (n50 = 10.3 / 15.5 / 20.5 at support 8 / 12 / 16): a constant ratio, i.e. H_prop.
  - The theorems were re-read in full text (pdftotext).
- **Huang et al. 2511.07378 (THEOREM; one-layer NoPE transformer, LEGO state tracking, gradient descent):**
  - **Thm 1.2 (informal), symmetry actions:** "standard CoT training only generalizes to a **constant factor** of the training length". The constant is not stated.
    - For simply transitive actions, generalisation reaches d^c (0 < c < 1), which is not proportional.
  - **Thm 5.1:** a model trained progressively on lengths L = 1..2^k (Algorithm 2, self-labelled stages) solves length 2^{k+1} with accuracy **1 − 1/poly(d)**. The "constant-factor length generalization" is instantiated as a **factor of 2 at near-perfect accuracy**.
  - It is a **sufficient bound** under its Assumptions 3.1–3.5 and 5.1, not a fitted constant.
  - **Verdict:** "proportional to the trained length" is correct as a multiplicative-margin statement. The only explicit constant is **≥ 2× at accuracy → 1**.
  - That isn't comparable to our 1.28, which is measured at the 50% crossing. Our 90% crossings were about 1.14–1.16 (§215 base / round 1). Ours is well below Huang's factor, as expected when his assumptions (one layer, NoPE, LEGO) don't hold.
- **Yang et al. 2610.14456 (THEOREM; looped transformer, graph connectivity, population gradient flow):**
  - The margin is in **loop depth relative to training hardness**, not input length: a finite window T < T_max(G; K), where T_max involves log_3 of a 3^K-dependent term (Eq. 8).
  - The extra depth is Θ(H·3^{K−H}). It is proportional to the training hardness H **only when K − H = O(1)**, and the constant is hidden in the Θ.
  - **Verdict:** the note's wording ("a constant fraction of the trained length **or hardness**") is acceptable for Yang only with the qualifier "hardness, in depth, when K − H = O(1)". **It gives no numerical constant.**
- **For contrast, the empirical precedent is additive, not proportional:** Lee 2502.01612's reverse-addition margin stays at about 3–5 digits as lengths grow from 10 to 100 (ratio falling toward ≈ 1.05).
  - Our constant 1.28 ratio across supports 8 / 12 / 16 is the multiplicative pattern the two theorems predict qualitatively, **not** Lee's additive one.
- **Bottom line:**
  - H_prop is supported qualitatively by Huang (a constant-factor statement) and, with qualifiers, by Yang.
  - **Neither supplies a constant comparable to ≈ 1.28.** Huang's explicit factor 2 is a near-perfect-accuracy sufficient bound in a different model; Yang's constant is unspecified.
  - Treat 1.28 as an empirical in-house constant. A prediction for round 3 (n50 ≈ 1.28 × the round-3 support) rests on the observed ratio, not on a theorem.

## Addendum 2026-10-21 (our §219 / §216 numbers vs the margins above)
- Text and latent self-training lineages (Note 219-13) hold n50 / support ≈ 1.22–1.30 across base / R1 / R2 (support 8 / 12 / 16). That is consistent with a margin proportional to trained depth (Huang 2511.07378, Yang 2610.14456) rather than Lee's near-constant +3–5 on reverse addition. It is two rounds and one lineage each, so descriptive only. See LIT_S219_TEXTSELF_RESULT_2026-10-21.
