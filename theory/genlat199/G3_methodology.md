# G3: Experimental methodology: learning more per GPU-dollar, with claims that hold up (2026-10-13)

**Methodology subagent for Theory.** The repo was read only; nothing under D:\Latent was edited, and no GPU or cloud host was used.

**User rule (relayed 2026-10-13):** every adopted change carries mathematical backing. Each rule below has a **Math** block: a derivation, or a cited theorem with its assumptions.

**How this note relates to the existing ones (extended, not repeated):**
- `LIT_GSM_TEMPLATES_2026-10-13`: GSM-Symbolic is eval-only; TemplateGSM is train-derived.
- `LIT_PROJECT_AUDIT_2026-10-04`: reviewer demands, including survival analysis for censored runs (§8).
- `LIT_PLATEAU_SIGNATURE_2026-10-13`: a detector, read NO-SIGNATURE in 192-1.
- `LIT_SEED_STABILITY_2026-10-13`: Frankle 1912.05671.
- `LIT_STABLE_LATENT_2026-10-02` (b): best-of-n reporting (Dodge 1909.03004); "report all runs" (Henderson, Bouthillier, Agarwal).
- `LIT_S130_MARKER`: exact CIs, ≥ 8 per class.
- `LIT_S104_INTERVENTIONS`, `LIT_DAS_EDIT`, `LIT_NORTHSTAR`: patching pitfalls, DAS controls, Lanham.
- `MATH_selection.md` §6: the restart-cost formula.

**Source tags:**
- **[abs]**: the arXiv abstract only. Method details are never stated from these.
- **[full]**: the full text (HTML or ar5iv). "fetch" means a fetched summary checked against the abstract.
- **[std]**: a standard non-arXiv result or reference, not fetched.
- **[DATA]**: post hoc, from existing `log.jsonl` / `eval.json` files. Descriptive, not a test.
- **[calc]**: computed or derived here.

All arXiv IDs were checked against the arXiv API on 2026-10-13. Scripts are in `scratchpad/genlat/tools/`:
- `designs.py` (exact binomial);
- `mc_designs2.py` (100k-rep Monte Carlo of wave designs);
- `sprt.py` (exact dynamic programming);
- `entry_table.py` (replicates `theory/basin_entry.py`, read-only);
- `depth_var.py`;
- `seedstats.py`: the helper to adopt. Its self-test reproduces Simon's published design for p0 0.10 / p1 0.30 (optimal 1/10, 5/29, E[N] 15.0; minimax 1/15, 5/25).

**Cost model [DATA + assumption]:**
- 0.885 s/step on a cloud 3090: `s185_rms_s1/eval.json` gives train_sec 42,491 for 48k steps, and eval_sec is 763.
- About $0.05 per 1k steps, plus $0.10 per host. So c₈ = $0.50 (to 8k), c₃₀ = $1.60 (to 30k), c₄₈ = $2.50 (to 48k).

## Three findings

1. **A 3/3 claim is false 22–30% of the time** when the base rate is 0.6–0.67: P(3/3) = p³ = 0.216 / 0.296 [calc].
   - Even 6/6 leaves P(p ≥ 0.9) at 0.52 under a uniform prior.
   - Certifying p ≥ 0.9 needs 29/29.
   - What is cheap to show is "P_X > 0.6" at a stated α.
2. **A same-seed rescue reads positive with probability p(1 − ρ) under no effect** (Math 3).
   - ρ looks low: Summers & Dinneen 2103.04514 [abs] ("even one-bit changes in initial parameters result in models converging to vastly different values"); our GPU-class flips; seed-1 vs seed-2 entry order 2 / 2 / 1 across 5 recipe pairs [DATA].
   - So §196 ACCELERATES has an FPR of about 0.4–0.6 [DATA: 6 of 15 runs meet "entry ≤ 25k and final < 0.1"].
3. **Depth contrasts are dominated by entry time [DATA, 8 runs].**
   - Between-run SD 0.086 raw, 0.029 within entry strata (the 8.9× variance ratio comes from these two SDs).
   - Eval SE 0.008, and eval is 2% of run time.

## Ranked changes, each with its mathematical backing

### 1. Sequential absolute-rate designs on fresh seeds for every "P(enter)" or "k/k" claim
- **Change.** Replace ad-hoc s1 → s2 → s3 extensions with a registered sequential test of H₀: p_X ≤ p₀ against H₁: p_X = p₁, using Simon's two-stage design, a curtailed all-of-N rule, or a truncated SPRT.
  - Two-arm comparisons are unaffordable [calc]: a one-sided Fisher exact test of 0.6 vs 0.9 needs 30 seeds per arm for power 0.80, about $96.
- **Math.**
  - **(a) The size of a composite null is attained at p₀ (coupling theorem, [std]).**
    - Write X_i = 1{U_i < p} with U_i i.i.d. uniform. Any rule that claims on an up-set of outcome vectors is monotone: more entries never turn a claim into a non-claim. So P_p(claim) is non-decreasing in p.
    - Hence sup over p ≤ p₀ of P_p(claim) = P_{p₀}(claim).
    - Assumptions: seeds i.i.d. given the recipe, and a fixed GPU class.
  - **(b) All-of-N:**
    - size = p₀^N; power = p₁^N.
    - Stopping at the first failure (one seed at a time), E[N | p] = Σ_{i=0}^{N−1} p^i = (1 − p^N)/(1 − p).
    - In waves of w seeds, E[N | p] = w Σ_{j=0}^{N/w − 1} p^{wj}.
    - Curtailment never changes the decision, because it stops only once the final decision is fixed. So the size and power equal the fixed-N values.
  - **(c) Simon (n₁, r₁, n, r)** (Simon 1989, *Controlled Clinical Trials* 10:1–10 [std; PubMed 2702835]):
    - claim iff X₁ > r₁ among the first n₁ seeds and X > r among all n;
    - α(p) = Σ_{x₁ = r₁+1}^{n₁} b(x₁; n₁, p) · P(Bin(n − n₁, p) ≥ r + 1 − x₁);
    - PET(p) = B(r₁; n₁, p);
    - E[N | p] = n₁ + (1 − PET(p))(n − n₁).
    - Simon's optimal design minimises E[N | p₀] subject to α(p₀) ≤ α and α(p₁) ≥ 1 − β; minimax minimises n. Both are found by exhaustive search (`seedstats.simon`).
  - **(d) Wald's SPRT [std: Wald 1945; optimality Wald & Wolfowitz 1948].**
    - Log-likelihood ratio Λ = s·ln(p₁/p₀) + f·ln((1 − p₁)/(1 − p₀)), after s entries and f failures.
    - Claim when Λ ≥ A = ln((1 − β)/α); stop when Λ ≤ B = ln(β/(1 − α)).
    - **Wald–Wolfowitz theorem:** among all tests with error probabilities ≤ (α, β), the SPRT minimises E[N] under both p₀ and p₁. Assumptions: i.i.d. observations, simple hypotheses, no truncation, and boundaries hit exactly. Overshoot makes Wald's approximations α' ≤ α/(1 − β) and β' ≤ β/(1 − α) inexact, so we compute the exact OC.
    - **For p₀ 0.6, p₁ 0.9, α 0.05, β 0.2 [calc]:** A = 2.7726, B = −1.5581; each entry adds +0.4055 and each failure −1.3863.
      - **Claim** at s ≥ 7 / 11 / 14 / 18 entries with f = 0 / 1 / 2 / 3 failures.
      - **Futility stop** at f = 2 with s ≤ 2; f = 3 with s ≤ 6; f = 4 with s ≤ 9.
- **Operating characteristics** (endpoint: entry by 30k; [calc]):

  | design | size at p 0.6 / 0.667 | power at 0.9 / 0.95 | E[N] at p 0.6 / 0.9 | E[$] at p 0.6 / 0.9 | max |
  |---|---|---|---|---|---|
  | today: 3/3 at 48k | 0.216 / 0.296 | 0.729 / 0.857 | 3 / 3 | 7.5 / 7.5 | $7.5 |
  | **A: 6/6, waves of 3, stop at first failure** | **0.047 / 0.088** | 0.531 / 0.735 | 3.65 / 5.19 | 5.8 / 8.3 | $9.6 |
  | A8: 8/8 (if p₀ = 2/3) | 0.017 / 0.039 | 0.430 / 0.663 | 3.7 / 6.3 | 6.0 / 10.0 | $12.8 |
  | **B: Simon (6, 4, 13, 10)** | **0.049** / 0.121 | **0.819** / 0.955 | 7.63 / 12.20 (uncurtailed) | 9.9 / 18.2 | $20.8 |
  | **S: SPRT, one seed at a time, truncated at 30** | **0.048** / 0.121 | **0.893** / 0.980 | 6.62 / 10.61 | 10.6 / 17.0 | P(N ≥ 20) 0.034 / 0.092 |
  | S3: SPRT in waves of 3 | 0.030 / 0.091 | 0.900 / 0.985 | 8.10 / 12.89 | 13.0 / 20.6 | – |

  - Wald's no-overshoot ASN approximation, (L·B + (1 − L)·A)/E[z], gives 4.31 / 8.42. Those are too optimistic, because a failure overshoots B by up to 1.39.
  - S has the highest power at an E[N | p₀] below Simon's, as Wald–Wolfowitz predicts. It needs a hard cap (truncation, counted as NO-CLAIM, is conservative by (a)).
- **Adopt.**
  - Each seed-rate registration has an **OC block**: p₀, p₁, α, β, design, fresh seed list, stop rules, E[$] at p₀ and p₁, and max $. `seedstats.py` prints it and Audit re-runs it.
  - Choose p₀ conservatively: pooled history 10/15 by 48k (exact CI 0.38–0.88), 9/15 by 30k; RMS 2/3 (0.09–0.99). At p₀ = 2/3, use A8 or Simon (7, 5, 24, 19).
- **Already?** No.

### 2. Conservative censoring: truncation and frozen detectors that can only turn successes into failures
- **Change.**
  - Score each seed Y = 1{entered by T = 30k AND not flagged at 8k by a frozen detector}.
  - Stop flagged seeds at 8k.
  - Continue a seed to 48k only if DEEP / DEPTH are needed.
- **Math.**
  - **(a) Validity [calc; coupling].**
    - Y_i ≤ E_i (= 1{entered}) pointwise, and the claim rule is monotone (1a). So claim(Y) ≤ claim(E) pointwise, and P(claim) ≤ P_{p}(claim with E) ≤ α whenever p ≤ p₀.
    - This needs no assumption about the detector's accuracy: only that it is fixed before data and uses only that seed's own trajectory.
  - **(b) Power and cost.**
    - Let sensitivity s = P(flag | fail) and false-flag rate f = P(flag | success).
    - The scored rate is q = p·P(enter by T | enter)·(1 − f).
    - Per seed, P(flag) = φ = p·f + (1 − p)·s, and E[cost] = c₈ + (1 − φ)(c₃₀ − c₈).
    - For design A, one wave costs W = 3c₈(1 − (1 − φ)³) + 3c₃₀(1 − φ)³. It proceeds with probability (p(1 − f))³, so E[$] = W(1 + (p(1 − f))³) and power = (p₁(1 − f))⁶.
    - **Check** at p = 0.6, s = 0.5, f = 0: φ = 0.2, W = 3.19, E[$] = $3.88. The MC gives $3.88.
    - **Specificity matters more than sensitivity:**
      - f = 0, s = 0.5: null cost $5.8 → $3.9, power unchanged at 0.531;
      - f = 0.05, s = 0.8: null cost $2.9, but power falls to (0.9 × 0.95)⁶ = 0.39.
    - **Truncation at 30k [DATA, 15 runs]:**
      - It keeps 9 of 10 eventual entrants (entries at 17, 19, 19, 19, 20, 22, 25, 26, 28 and 42k), so P(enter by T | enter) ≈ 0.9 on this history.
      - The first bin below 0.5 equals the final entry in 15/15 runs (no transient dips).
  - **(c) Post-selection bias, and how pre-registration removes it [calc].**
    - If flagged seeds are discarded or restarted and the rate among kept seeds is reported, the estimand becomes P(enter | not flagged) = p(1 − f) / (p(1 − f) + (1 − p)(1 − s)), by Bayes.
      - At p = 0.6, s = 0.8, f = 0.05 that is 0.877; at s = 0.5, f = 0 it is 0.75.
      - So **a recipe with p = 0.6 would be reported as 0.75–0.88.**
    - Pre-registration fixes the estimand, in either of two ways:
      - (i) flagged seeds count as failures (the bound in (a));
      - (ii) the *procedure* "recipe + detector + restart" is the method. Its success probability and expected cost come from MATH_selection §6: E[cost] = [c + (p·TPR + (1 − p)·FPR)(C − c)]/(p·TPR).
    - **A threshold tuned on the same seeds is optimistic.**
      - Choosing the best of m thresholds on n seeds inflates in-sample accuracy by up to E[max of m] ≈ σ√(2 ln m) [std; Gaussian maxima bound, i.i.d. worst case], with σ = √(a(1 − a)/n).
      - For §130's 24 combinations on about 4 positives: σ = 0.2, so the optimism is up to about 0.50. In-sample recall of 1.0 is nearly guaranteed.
      - Hence: freeze the detector before the confirmatory seeds exist.
- **Citations.** Dodge 2002.06305 [abs] ("stop training less promising runs early"); LC-PFN 2310.20447 [abs] (predictive early stopping, 2–6×).
- **Benefit.** 33–50% saving on null methods at no loss of validity.
- **Cost.** An 8k checkpoint and a minutes-long eval per seed; delete the checkpoint after the read.
- **Already?** Partly. Entry ≤ 25k is read, but runs are not stopped.

### 3. A placebo-perturbation control before any same-seed rescue or one-seed A/B
- **Change.** Before reading a same-seed rescue (§196: Grokfast on RMS s3), run 3 mechanism-free one-bit placebos of s3, e.g. the least-significant mantissa bit of one trunk weight, to 30k ($4.5).
- **Math.**
  - **(a) Rescue false-positive rate [calc].**
    - Let S_b and S_X be the base and treated outcomes on the same seed: Bernoulli(p) each under the null, with correlation ρ.
    - Then P(1,1) = p² + ρp(1 − p), and P(S_X = 1, S_b = 0) = (1 − p) − [(1 − p)² + ρp(1 − p)] = p(1 − p)(1 − ρ).
    - So **P(rescue | S_b = 0, no effect) = p(1 − ρ):** 0.60 / 0.45 / 0.30 / 0.15 for ρ = 0 / 0.25 / 0.5 / 0.75.
    - Assumption: under the null, the treatment perturbs the trajectory like a placebo. Same marginal p, same ρ; Summers & Dinneen 2103.04514 [abs]: "all sources of nondeterminism have similar effects".
  - **(b) What the placebo decides.** The rescue rate r = p(1 − ρ) is near 0 if s3 is intrinsically hard, and near 0.6 if the lottery is chaos.
    - With 3 placebos: P(0/3 | r = 0.6) = 0.064, and P(≥ 1/3 | r ≤ 0.05) ≤ 0.143.
    - Rule: **0/3 → HARD-SEED** (same-seed rescues informative); **≥ 1/3 → CHAOTIC** (one-seed rescues read PROMISING, and need rule 1).
  - **(c) [DATA] supporting low ρ.** Entry order of seed 1 vs seed 2 across 5 recipe pairs is 2 / 2 / 1, an exact sign test on 4 non-ties p = 1.0. That is consistent with outcomes that behave like fresh draws per recipe (MATH_selection P10 predicts a flip rate of about 2p(1 − p)).
- **Adopt.** A dated note for §196 before its output is read: ACCELERATES on one seed = PROMISING. Run the placebo cell in the next free slot.
- **Already?** No. Plain s3 is treated as an exact control, which is true for causation, not for mechanism.

### 4. Two estimands, P(enter) and quality given entry; entry time as survival data
- **Change.**
  - Every quality reading (DEEP / DEPTH / STABLE) states its estimand: (i) total effect; (ii) effect on entry; (iii) effect among runs entered by T, flagged as post-treatment conditioning.
  - Compare within entry strata.
  - Analyse entry time with Kaplan–Meier, log-rank and restricted mean survival time (RMST), censored at T.
- **Math.**
  - **(a) Law of total variance [std].** Var(D) = E[Var(D | S)] + Var(E[D | S]).
    - [DATA]: 0.0074 ≈ 0.00084 + 0.0066, using sample variances, so the split is approximate. Strata explain about 89%.
    - For a two-arm mean contrast with SE target 0.03, n per arm = 2σ²/SE²: **16.4 raw, 1.9 within stratum.**
    - A single-seed gap ≥ +0.10 under no effect has probability 1 − Φ(0.10/(√2·σ)): 0.205 raw, 0.007 within stratum.
  - **(b) Conditioning on a post-treatment variable is biased in general** (principal stratification [std: Frangakis & Rubin 2002]).
    - E[D | entered, X] − E[D | entered, base] mixes the effect on D with a change in *who* enters.
    - It is the causal effect only if X does not change the entering population. So (iii) is descriptive unless entry rates match.
  - **(c) Survival tools [std].**
    - KM estimate Ŝ(t) = Π_{t_i ≤ t}(1 − d_i/n_i). It is consistent under independent censoring, and administrative censoring at T is independent by design.
    - The log-rank test is valid under H₀: S_X = S_base for any censoring pattern, and most powerful under proportional hazards.
    - RMST = ∫₀^T S(t) dt needs no proportional-hazards assumption.
    - These use censored runs, which a binary cut at T discards.
- **Adopt.** `basin_entry.py` gains `km()` / `logrank()`; depth readings print their stratum. **Already?** Partly: CONFOUNDED-BY-ENTRY is flagged.

### 5. Screen, then confirm on fresh seeds and fresh test draws
- **Change.**
  - Label cells SCREEN or CONFIRM.
  - Screen candidates with 1–2 seeds to 30k, dropping a candidate at its first failure.
  - Confirm the survivor with rule 1 on pre-declared fresh seed indices (e.g. 101+), scored on a fresh test draw: free for synthetic tasks, using strata n ≥ 6 where the item space is large.
- **Math.**
  - **(a) Confirmation is unbiased by the screen [calc].** Confirmatory outcomes are independent of the screening data given p_X. So P(claim | X selected) = P_{p_X}(confirm rejects) ≤ α whenever p_X ≤ p₀, however X was selected.
    - Reusing the screening seeds breaks this: the winner's curse E[max_j p̂_j] ≥ max_j E[p̂_j].
    - Seeds 1–3 and `test_set(seed = 17100)` have served about 20 recipe decisions. Dwork 1411.2664 / 1506.02629 [abs]: "reusing a holdout set adaptively multiple times can easily lead to overfitting to the holdout set itself".
    - Recht 1902.10811 [abs] found new-test-set drops "not caused by adaptivity". So our main risk is forking-path selection, not memorisation.
  - **(b) Expected cost of drop-on-first-failure screening (successive rejection on Bernoulli arms) [calc].**
    - For a candidate with rate p_j and cap K: E[N_j] = Σ_{i=0}^{K−1} p_j^i = (1 − p_j^K)/(1 − p_j), and P(survive) = p_j^K.
    - Pipeline: E[$] = c₃₀ Σ_j (1 − p_j^K)/(1 − p_j) + Σ_j p_j^K · E[$_confirm | p_j].
    - At K = 2: a null candidate (p 0.6) costs 1.6 c₃₀ + 0.36 × $5.8 = **$4.7** and passes the screen with 0.36. A good one (p 0.9) costs 1.9 c₃₀ + 0.81 × $8.3 = $9.8.
    - The per-candidate false-claim rate is p₀^K · α ≤ 0.017.
  - **(c) Hyperband / successive halving by loss rank (Li 1603.06560 [abs]; ASHA 1810.05934 [abs]).**
    - Budget identity [calc]: one bracket with n arms, factor η and minimum resource r₀ keeps n/η^i arms at r₀η^i, so it spends n·r₀ per rung and n·r₀(⌊log_η n⌋ + 1) in total.
    - SH keeps arms by intermediate rank, so its value requires early ranks to predict final ones. **Here they anti-predict:** the 4 failures held 8k answer-CE ranks {1, 2, 4, 6} of 10 (MATH_selection, exact p = 0.033) [DATA].
    - So **loss-rank SH at 8k would keep the failing seeds.** Use SH only on a proxy that passes rule 7.
- **Already?** No. The same seeds serve screening and claims.

### 6. Uncertainty in every reading; evidence grades; a replication ledger
- **Change.**
  - Every reading prints:
    - its margin to the bar in seed-SD units;
    - a seed-level interval (exact for binary outcomes; t or bootstrap over seeds otherwise; Multi-Bootstrap over seeds × items, MultiBERTs 2106.16163 [abs]);
    - paired item-level differences for two models on the same items.
  - NORTH_STAR lines carry a grade: **E1** (one seed), **E2** (2–3 seeds), **E3** (rule 1 confirmed).
  - A ledger records each single-seed reading and its replication outcome (e.g. §187 reversed at s2; §192 refuted RMS seed-robustness).
- **Math.**
  - **(a) Clopper–Pearson** [std] inverts exact binomial tails, so its coverage is ≥ 1 − α for every p. Examples: 6/6 → one-sided lower bound 0.607; 2/3 → (0.094, 0.992).
  - **(b) Paired differences [std; Miller 2411.00640 [full, fetch]]:** Var(Ā − B̄) = (σ_A² + σ_B² − 2σ_AB)/n.
    - The paired SE is smaller whenever item-level outcomes correlate.
    - Miller's recommendations: "question-level paired differences, rather than the population-level summary statistics"; clustered SEs "over 3X larger than naive" for grouped items; power analysis before an eval.
  - **(c) Spend on seeds, not items.** For a mean over n_s seeds × n_i items, Var = σ_seed²/n_s + p(1 − p)/(n_s n_i).
    - [calc]: the item share of the variance is p(1 − p)/n_i / (p(1 − p)/n_i + σ_seed²).
      - For DEPTH (4,000 items) it is 0.8% at σ_seed = 0.086 and 6.7% at 0.029.
      - For one per-n accuracy (1,000 items) it is 3.1% and 22%.
    - So extra items help only for within-stratum, per-n readings; seeds dominate everywhere else.
  - **(d) Why the ledger.** The positive predictive value of a reading is PPV = (1 − β)π / ((1 − β)π + α(1 − π)) [std: Ioannidis 2005], where π is the share of true effects among those tested.
    - At π = 0.3 and power 0.7: a single-seed reading with α ≈ 0.2 (finding 3) has PPV 0.60; α = 0.05 gives PPV 0.86.
    - The ledger's replication rate estimates PPV directly, which calibrates how much one seed should move spend.
- **Citations.** Card 2010.06595 [abs] (underpowered studies "increase the chances of exaggerated findings"); Picard 2109.08203 [abs] (outliers are "surprisingly easy to find"); Agarwal 2108.13264 [abs]; Benavoli 1606.04316 [abs].
- **Already?** Partly. GSM8K_REPRO states a binomial interval; scorers print point thresholds only.

### 7. Validate a proxy by decision accuracy before it gates spend
- **Change.**
  - Before a proxy decides a launch (a local-card pilot, an 8k or 16k loss, a small-n curriculum stage, GPT-2-small for a larger model), score its decision accuracy (DA): the share of k already-run full-scale pairs whose order it reproduces.
  - Prefer continuous proxies (aux CE, λ_eff) over binary DEEP.
- **Math.**
  - **(a) The bar [calc].** k/k correct gives a one-sided 95% lower bound of 0.05^(1/k): 6/6 → 0.61; 14/14 → 0.81. A bar of "≥ 0.8 on 6 pairs" is weak.
    - **Register "beats a coin":** 6/6, or ≥ 9/10 (P ≤ 0.016 and 0.011 under DA = 0.5).
    - Pairs that share recipes are not independent, so count a recipe once.
  - **(b) Value of a proxy [calc].** Choosing between two candidates by proxy picks the worse one with probability 1 − DA. So expected regret = (1 − DA)·|Δp|·V, where V is the value per unit of P(enter). Use the proxy only when its saving exceeds this.
  - **(c) Known biases.**
    - Wu 1803.02021 [abs]: short-horizon objectives bias toward small step sizes, "by multiple orders of magnitude". The theory is a noisy quadratic, so it is an existence result, not a bound for us.
    - μTransfer 2203.03466 [abs] covers HP transfer across width for models parametrised in μP. For our pretrained SP trunk it applies at most to the newly initialised memory branch.
    - Wortsman 2309.14322 [abs]: instabilities appear at small scale at high learning rate, and can be "predicted before they emerge" from activation and gradient-norm scaling.
    - DataDecide 2504.11393 [abs]: single small-scale ranking "~80% of comparisons correct"; continuous likelihood metrics are better proxies.
- **[DATA].** Within RMS, the 16k bin separates s3 (1.77) from s1 / s2 (1.16 / 1.20). At 8k s3 is ahead (2.29 vs 2.39 / 2.50). Across recipes the 16k bin overlaps: failures 1.60–2.44, entrants 0.81–2.38.
- **Already?** No formal test. The 4060-vs-3090 caveat is known.

### 8. Measure the decision time with fork tests (bounds every early detector)
- **Change.**
  - From 4k / 8k / 12k checkpoints of 3 parents, run 2 children each with a new data-order seed (`--data_seed_after`), to 30k.
  - Score q̂(k) = the share of children that enter. The linear-path barrier between children is a free extra.
- **Math.**
  - **(a) Martingale [std: Doob].** q_k = P(enter | F_k) is a martingale in k, and "decided by k" means q_k ∈ {0, 1}. Each child from θ_k is an independent Bernoulli(q_k), given an independent data order after k.
  - **(b) Bayes-error bound on any detector [calc].** For any rule g that uses only F_k: P(g ≠ Y | F_k) = q_k 1{g = 0} + (1 − q_k) 1{g = 1} ≥ min(q_k, 1 − q_k). Taking expectations, **error ≥ E[min(q_k, 1 − q_k)]**.
    - Fork tests therefore measure the floor that no 8k detector (rule 2, P2) can beat.
  - **(c) The design discriminates [calc].** The parent's outcome is Bernoulli(q). P(both children agree with it) = q³ + (1 − q)³: 1 if decided, 0.25 at q = 0.5. For 3 parents all agreeing, P = 0.25³ = 0.016 under q = 0.5.
  - **(d) Precedent.** Frankle 1912.05671 [full, ar5iv fetch]: copies spawned at iteration k are trained with "different samples of SGD noise", and stability is judged by the error barrier along the linear path (< 2%). ResNet-20 is stable at 3% of training, ResNet-50 at 20%.
- **Cost.** 6 children × 22k steps ≈ $6.6 (one item). **Already?** No.

### 9. C1 / C3 evaluation upgrades
- **(a) The C1 battery on intermediate checkpoints** (the 8k / 16k / 24k checkpoints already kept for rules 2 and 8). 2607.06648 [abs]: "high task accuracy can coexist with low counterfactual responsiveness", and an endpoint view "can obscure" when responsiveness changes.
  - **Math [calc]:** C1 at the end is one sample of a time series c(t). The entry-relative timing (does c(t) rise before, at, or after entry?) is the quantity that separates cause from consequence for λ (§199). An endpoint cannot identify it.
- **(b) Compute frontiers for C3.** Truncate text CoT at several lengths (early answering, Lanham 2307.13702, in LIT_NORTHSTAR) and compare A_text(c) with the latent A_lat(c) over a shared FLOP range. 2606.17930 [abs]: "report capability as a function of inference-time compute … at matched budgets".
  - **Math:** with A(c) increasing in c for both methods, A_lat(c₁) − A_text(c₂) at c₁ ≠ c₂ confounds method with budget. Only A_lat(c) − A_text(c) on a common grid (interpolated on log FLOPs) is the contrast.
- **(c) The compute-matched null is met by construction.**
  - **Proof:** every pass sees the identical input x. With the memory reset to M₀ before each pass, pass t computes F(M₀; x) for every t, so the end decoder reads S₁. "K passes, memory reset" is therefore output-identical to K′ = 1.
  - This answers the filler-token objection (Pfau 2404.15758 [abs]: "computational benefits independent of token choice").
- **(d) Optimal-constant ablation beside zero and donor ablation.**
  - **Math [calc, by definition]:** damage_OA = min_c E[L(·; component := c)] − E[L]. It is ≤ the damage of zero or mean ablation, both of which are particular constants. So a removal effect under OA is the most conservative constant-ablation claim. (Li & Janson 2409.09951 [abs] for the idea.)
  - For C7 features, MIB 2504.13151 [abs]: DAS performs best, and SAE features are "not better than neurons".
- **Already?** (c) yes; (a), (b), (d) no. Cost is about $0, eval-only.

### 10. GSM8K hygiene beyond LIT_GSM_TEMPLATES
- **Change.**
  - A dev split from train for all selection; the 1,319 test items touched only by registered final cells, with each touch logged.
  - Decontaminate GSM8K-Aug and TemplateGSM-derived training data against the test set (n-gram plus embedding checks).
  - Add GSM1k or GSM-Symbolic (eval-only) when the base model changes.
- **Math.**
  - **(a) Selection optimism [std; Gaussian maxima].** Picking the best of m equal-skill variants on the same test set inflates the winner's expected accuracy by up to σ√(2 ln m)·√(1 − r), where σ = √(p(1 − p)/n) and r is the variants' item-level correlation.
    - At n = 1,319, p = 0.4 and m = 20: 3.3 points if r = 0; 1.8 points if r = 0.7.
    - That is comparable to the gaps between published latent GPT-2 GSM8K results.
  - **(b) Contamination bias** = share contaminated × (accuracy on contaminated − accuracy on clean). It is non-zero only if contaminated items exist, hence the check.
    - Yang 2311.04850 [abs]: n-gram decontamination is "insufficient"; rephrased test samples bypass it; contamination was found "in synthetic dataset generated by GPT-3.5/4".
  - **(c) Base-model contamination.** GPT-2 (2019) predates GSM8K (Cobbe 2110.14168 [abs], submitted 27 Oct 2021), so its pretraining cannot contain the test. A newer base can.
    - GSM1k 2405.00332 [abs]: a GSM8K mirror with drops "up to 8%". Its availability was not checked here.
- **Already?** The binomial CI and FLOPs, yes; (a)–(c), no.

### 11. Numerical-stack fingerprint, host canary, fp32 eval check
- **Change.**
  - Record the torch / CUDA / cuDNN versions and the deterministic flags beside gpu / driver.
  - A 200-step loss-hash canary per new host against its GPU class's reference.
  - Once per model family, compare K′ 16 / 32 evals in fp32 vs bf16, and on the training card vs another card. 192-1 evaluated 3090-trained models on a 3070 in bf16.
- **Math.**
  - **(a) Confounding by stack [calc].** If p depends on the stack s (unknown; P10), then E[p̂_A − p̂_B] under no effect = Σ_s (w_A(s) − w_B(s)) p(s), where w are the arms' stack mixes. Recording s allows stratification (w_A = w_B within a stratum); pinning one GPU class is the special case of a point mass w.
  - **(b) Eval rounding compounds with K′.** A per-pass rounding perturbation δ propagates as δ Σ_t Π L_t. For path-dependent models (L ≈ 1, λ_eff ≈ 0; MATH_selection) this grows like K′δ, so long-K′ evals are the sensitive ones.
    - Yuan 2506.09501 [abs]: bf16 greedy reasoning varies "up to 9%" in accuracy with GPU count, type and batch size; the fix is FP32 compute (LayerCast).
    - Zhuang 2106.11872 [abs]: determinism overheads up to 746% / 241% / 196%. So prefer recording and stratifying over forcing determinism.
- **Already?** Partly: gpu, driver and eval_dtype are logged.

**Registration practice is already strong.** The additions:
- the OC block (1);
- the estimand (4);
- the claim scope ("RMS / RTX 3090");
- SCREEN / CONFIRM (5);
- evidence grades (6).

Precedents: van Miltenburg 2103.06944 [abs] (registered reports); REFORMS 2308.07832 [abs] (32 questions); NeurIPS 2020 Pre-registration in ML workshop (PMLR v148) [proceedings index]; Forde & Paganini 1904.10922 [abs]. If we ever monitor continuously, the anytime-valid alternatives (e-processes and confidence sequences, valid "at all stopping times") are Ramdas 2210.01948 [abs], Howard 1810.08240 [abs] and Johari 1512.04922 [abs].

## Worked example: the cheapest design that can show "X raises P(enter) from about 0.6 toward ≥ 0.9"

**Registered claim:** H₁ "P_X(entry by 30k) > 0.6 on RMS / RTX 3090", tested against H₀: P_X ≤ 0.6 at α ≤ 0.05.
- "≥ 0.9" is reported as a posterior, not claimed. Certifying it needs the lower bound 0.05^(1/n) ≥ 0.9, i.e. n ≥ ln 0.05 / ln 0.9 = 28.4, so **29/29**.

**Design A (one $10 Audit item):**
- Fresh seeds 101–106, cloud 3090s, in waves of 3.
- Each seed is checkpointed at 8k, with a frozen detector λ_eff(8k) ≤ τ, τ deep in the C side, chosen before data.
- Any flag in a wave → stop the wave: **NO-CLAIM**.
- At 30k, any seed not entered (fixed-bin: every bin from entry to 30k < 0.5) → **NO-CLAIM**.
- 3/3 → wave 2 (seeds 104–106), same rules. 6/6 → **CLAIM**.

**Explicit computation [calc]:**

| step | value |
|---|---|
| size, any p ≤ 0.6 (1a coupling; 2a for the flags) | ≤ 0.6⁶ = **0.0467** (0.0878 if the true base is 2/3) |
| power | p₁⁶(1 − f)⁶ · P(enter by 30k \| enter)⁶: at p₁ = 0.9 and f = 0, **0.531**; at p₁ = 0.95, 0.735. If 10% of entrants enter after 30k, power at p₁ = 0.9 is 0.9⁶ × 0.9⁶ = 0.28. So the endpoint must suit X's entry times (P3 predicts earlier, tighter entry for tilt methods). |
| E[seeds \| p], no detector | 3 + 3p³ = 3.65 (p 0.6) / 5.19 (p 0.9) |
| E[$], no detector | c₃₀ × E[seeds] = **$5.84 / $8.30** (MC: 5.84 / 8.31) |
| E[$], detector f = 0, s = 0.5 | W(1 + p³) with W = 3c₈(1 − (1 − φ)³) + 3c₃₀(1 − φ)³ and φ = 0.2 (p 0.6) / 0.05 (p 0.9): **$3.88 / $7.50** (MC agrees) |
| max $ / wall-clock | $9.6 / 1–2 waves of about 7.4 h (8k ≈ 2 h) |
| report after 6/6 | one-sided 95% lower bound 0.607; P(p > 0.6 \| 6/6) = P(Bin(7, 0.6) ≤ 6) = 1 − 0.6⁷ = 0.972; P(p ≥ 0.9 \| 6/6) = 1 − 0.9⁷ = 0.522 (uniform prior). The identity: P(Beta(s + 1, n − s + 1) ≥ x) = P(Bin(n + 1, x) ≤ s). |

**When power matters:**
- **Truncated SPRT (S):** claim at 7 straight entries, or 11 with one failure, or 14 with two; futility at 2 failures with ≤ 2 entries.
  - Truncated at 30: α 0.048, power 0.893, E[N] 6.6 / 10.6 → E[$] $10.6 / $17.0. P(N ≥ 20) is 0.034 / 0.092.
  - **Capped at 20** (no decision by then counts as NO-CLAIM, valid by 1a): α 0.044, power 0.812, E[N] 6.5 / 10.3 → E[$] $10.4 / $16.4.
- **Simon (6, 4, 13, 10):**
  - α = 0.1866 × 0.1586 + 0.0467 × 0.4199 = 0.0492;
  - power = 0.3543 × 0.8503 + 0.5314 × 0.9743 = 0.819;
  - PET(0.6) = 0.767, E[N | 0.6] = 6 + 0.233 × 7 = 7.63;
  - two Audit items, and stage 2 is bought only if stage 1 shows ≤ 1 failure.

**Against today's practice:**
- 3/3 at 48k costs $7.5 and claims falsely with probability 0.22–0.30.
- Design A costs $5.8 in expectation under the null and claims falsely ≤ 0.047 (≤ 0.088 if the base is 2/3).
- A's NO-CLAIM means "not shown", never "X fails": a p = 0.9 method fails A with probability 0.47.

## Caveats
- The [DATA] items are post hoc, pooled over recipes, with n = 5–15. The depth SDs include recipe effects.
- ρ comes from 5 pairs plus the literature; rule 3 measures it.
- Costs assume $0.05 per 1k steps and s185_rms_s1's step time; host speeds vary up to about 2×.
- The detector's sensitivity is unknown, so rule 2's saving is a range; its validity is not affected.
- Rules 1–2 assume i.i.d. seeds on one GPU class.
- The Gaussian-maxima bounds (2c, 10a) are i.i.d. worst cases. Correlated candidates give less optimism.
