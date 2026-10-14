# Verification pass on the G1–G3 general-improvement notes (THEORY_GENERAL_IMPROVEMENTS, 25f40809), priority claims only (2026-10-13)

Literature session, for Theory (the request) and Audit.
- **Sources:** each paper's own text, from the arXiv PDF via pdftotext (and the arXiv HTML for Moonlight Table 6) (2026-10-13).
- **G3's numbers:** recomputed independently (scratchpad oc_check.py).
- **Evidence tags** per the user's 2026-10-13 rule: THEOREM / DERIVATION / EMPIRICAL, with assumptions.

## Bottom line
- **One misread needs fixing:** Moonlight's Table 6 in G2 §2.9 and G1.
- **One citation-direction problem in note 199-M2:** Frankle 1912.05671 is cited for connectivity, but it shows the opposite at k = 0.
- **Everything else checked is supported by the full text,** and G3's operating characteristics reproduce exactly.

## 1. ERROR: Moonlight 2502.16982, Table 6 (G2 §2.9; G1 l.403)
- **G2 says:** "an AdamW-pretrained model fine-tuned with Muon shows no significant advantage (MMLU 50.2 vs 55.7 for Muon–Muon)".
- **The table actually reads** (columns are pretraining / SFT optimiser):

  | | Muon / Muon | **AdamW / Muon** | Muon / AdamW | AdamW / AdamW |
  |---|---|---|---|---|
  | MMLU | 55.7 | **55.3** | 50.2 | 52.0 |
  | HumanEval | 57.3 | 53.7 | 52.4 | 53.1 |
  | MBPP | 55.6 | 55.5 | 55.2 | 55.2 |
  | GSM8K | 68.0 | 62.1 | 64.9 | 64.6 |

- **50.2 is Muon-pretrained + AdamW-SFT.** The AdamW-pretrained + Muon-SFT cell is **55.3**, which is *above* AdamW / AdamW on MMLU (+3.3), HumanEval (+0.6) and MBPP (+0.3), and below it on GSM8K (−2.5). That is mixed, not "weak".
- **The paper's own sentence:** "when the SFT optimizer differs from the pretraining optimizer, SFT with Muon does not show a significant advantage over AdamW" (§3.5). The Discussion calls the mismatch "suboptimal".
- **Table 7** (Qwen2.5-7B SFT, Adam vs Muon): MMLU 71.4 / 70.8, HumanEval 79.3 / 77.4, MBPP 71.9 / 71.6, GSM8K 89.8 / 85.8. So Muon is on par to slightly worse.
- **Correct wording:** "Muon gave no consistent advantage when fine-tuning AdamW-pretrained models (Moonlight Tables 6 / 7: mixed, GSM8K lower)."
- The branch-only recommendation still follows. The 50.2 figure must go.
- **Tag:** EMPIRICAL. The table shows no error bars.

## 2. CAUTION: note 199-M2 cites Frankle 1912.05671 for "same-init fine-tunes are often linearly connected"
- **Frankle's own finding at k = 0 is the opposite:**
  - "All but the smallest MNIST network are unstable to SGD noise at initialization"; they become stable only after **3% of training (ResNet-20, CIFAR-10) or 20% (ResNet-50, ImageNet)**.
  - "Stable" means an error barrier under 2%. Frankle trains from scratch.
- **The connectivity claim belongs to:**
  - Neyshabur 2008.11687: "when training from pre-trained weights, the model stays in the same basin", whereas "models trained from random initialization do not live in the same basin";
  - Wortsman 2203.05482: fine-tuned models "often appear to lie in a single low error basin".
- **Our case is a hybrid:** a pretrained trunk plus a deterministic zero / copied memory branch (I checked `boxes/deltamem.py`: every branch parameter is copied or constant, so the branch init is identical across seeds).
  - Neyshabur predicts CONNECTED for the trunk part. Frankle's k = 0 instability is the relevant prior for a branch starting at W_o = 0.
- **Fix:** cite Frankle as the *instability-at-init* precedent and Neyshabur / Wortsman as the connectivity precedent. The s1–s2 same-class control in 199-M2 is exactly what's needed to separate these; keep it.
- **Tags:** all three EMPIRICAL (vision / NLP fine-tuning, no theorem).

## 3. Confirmed (full text)

| claim (where) | verdict | tag / assumptions / scope |
|---|---|---|
| **Summers & Dinneen 2103.04514:** a one-bit change in one first-layer weight gives the same run-to-run variability as a new seed; "all sources of nondeterminism have similar effects" (G2, G3, synthesis) | **Confirmed** (a single least-significant-bit change; the accuracy gap grows to 2.33% by epoch 2) | EMPIRICAL. From-scratch image classifiers (CIFAR-10 / ImageNet ResNets). **Linear models stay close** (convex), so the instability needs a non-convex net. Remedies the paper validates: accelerated ensembling (−48% variability) and test-time augmentation. Applying it to a pretrained fine-tune is an extrapolation; Neyshabur suggests pretrained fine-tunes are *more* stable. |
| **TRM 2510.04871:** Sudoku-Extreme 87.4 vs no-EMA 79.9; 1-step gradient 56.5; ACT (Q-learning) 86.1 → 87.4 (BCE halt); EMA 0.999; k = 4 of 7 truncation "did not help"; deep supervision 19 → 39% on ARC (blog, as quoted) | **Confirmed** (Table 1, §4.1, §4.6, §4.7, "Ideas that failed") | EMPIRICAL. A single table of point accuracies (no seeds or error bars), a tiny 5M model trained from scratch, weight decay 1.0, small data. "Full backprop" means through the *last* recursion process (n + 1 evaluations), with T − 1 no-grad processes first. |
| **HRM 2506.21734:** deep supervision detaches z between segments (Fig. 4 pseudocode); "a 1-step approximation of the gradient of the recursive deep supervision process" | **Confirmed** | DERIVATION, from the DEQ implicit-function / 1-step approximation. It assumes a fixed point, which TRM shows is not reached. |
| **DIS 2511.16886:** detached carries; "reducing the number of forward passes by 18x"; 3× fewer supervision steps, 8× fewer latent steps; ARC-AGI-1 24% at 0.8M parameters | **Confirmed** | EMPIRICAL, plus a policy-improvement framing. Note the title change (G1 already flags it). |
| **McLeish 2511.07384:** backprop through at most the last 8 recurrences (as Huginn); Fig. 4: Muon (lr 1e-3) "the most stable" for recurrent models, AdamW (5e-5) spikes to NaN; minimal difference for TinyLlama | **Confirmed** | EMPIRICAL. One data order, one lr per optimiser. |
| **He et al. 1812.01187:** "no bias decay" applies weight decay only to conv / FC weights; biases and BN γ / β are unregularised | **Confirmed** | EMPIRICAL / heuristic (following Jia et al.). One of the large-batch tricks; **no standalone ablation number**, as G2 says. |
| **Wang & Aitchison 2405.13698:** AdamW weights are an EMA of updates, with τ_iter = 1/(ηλ); the optimal τ in epochs is roughly constant | **Confirmed** | THEOREM ("we prove and empirically verify") **under a λ-dependent initialisation and a fully scale-invariant network**; the constant-τ_epoch finding is EMPIRICAL. Our pretrained, non-scale-invariant GPT-2 trunk doesn't meet the theorem's assumption. |
| **ZClip 2504.02507:** a z-score anomaly threshold on EMA statistics of the gradient norm; spikes clipped "while not interfering with convergence otherwise" | **Confirmed** (abstract and §1; LLaMA 1B pretraining, lr 3e-3) | EMPIRICAL. **Scope note for G2 §2.5:** our clip binds on 100% of steps. A z-score threshold tracks the running mean, so ZClip would mostly *stop* clipping. That changes the effective step size, not just the spike handling, so it isn't a small tilt; G2's "relative clipping" mechanism should say so. |
| **McCandlish 1812.06162:** B_simple = tr(Σ)/\|G\|²; the noise scale "increases as the loss decreases"; Adam-preconditioned gradients gave "mixed results" | **Confirmed** | DERIVATION under a local quadratic model with stated assumptions (§2.4). The estimator is exact given those assumptions. |
| **G3 statistics:** Simon optimal 1/10, 5/29 (E[N \| p₀] 15.0) and minimax 1/15, 5/25 (19.5) at p₀ 0.1 / p₁ 0.3; design B Simon (6, 4, 13, 10); A 6/6; S truncated SPRT; Clopper–Pearson 6/6 → 0.607 | **Reproduced exactly** (my code). B: size 0.049 / 0.120, power 0.819 / 0.954, E[N] 7.63 / 12.20. S: 0.048 / 0.121, 0.893 / 0.980, E[N] 6.62 / 10.61. A: 0.047 / 0.088, 0.531 / 0.735. SPRT A = 2.7726, B = −1.5581. 0.05^{1/6} = 0.607. | Simon: an exact binomial design (THEOREM-level computation; assumes i.i.d. Bernoulli seeds). Wald–Wolfowitz: THEOREM (i.i.d., simple hypotheses, untruncated). G3 states the assumptions correctly. **i.i.d. seeds given a recipe and GPU class** is the load-bearing assumption (our GPU-class basin flips). |

## Not re-checked here
- The ~180 other IDs: the agents report abs-page checks.
- I re-checked full text only for the priority list above.
