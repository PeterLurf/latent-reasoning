# G1: general training ingredients from latent / looped / recurrent-depth reasoning (2024-2026), mapped to our harness

Theory subagent (G1), 2026-10-13 22:10 EDT (clock). Read-only use of D:\Latent\docs (NORTH_STAR, CURRENT_TASKS, THEORY, THEORY_SEEDSTAB_2026-10-13, LIT_*), boxes/nested170.py and boxes/fsq120.py (optimiser lines only), and the seedstab FACTS.md. Nothing under D:\Latent was edited.

**Scope.** Training methods, not architectures. For each ingredient: what it is, the evidence, whether it is architecture-agnostic, which of our problems it targets, whether we already tried it, the cost in our harness, a cheap test, and (user rule, coordinator 2026-10-13) a **math basis**.

**What this note adds to existing notes (no repeats).**
- **Already covered elsewhere, and only cited here:**
  - LIT_GENERAL_LATENT_MAP: the method map, and the RL-over-FSQ recommendation.
  - LIT_RAND_K: Huginn's sampler, Kuo, Bansal PROG.
  - LIT_STEP_AUX / _GSM8K: SIM-CoT, 2606.20075, AGCLR.
  - LIT_PCCOT_PROBE: PCCoT's recipe.
  - LIT_DEPTH_RECURRENT_CHEN_HE: LayerScale and the identity gate.
  - LIT_SELF_DEPTH: filtered self-training.
  - LIT_HALTING_DECOY: halting length cues.
  - LIT_CHEAPER_ARCH: Penelope, MoR, MELT and other KV-sharing designs.
  - LIT_SEED_STABILITY: escape-time theory, stochastic collapse, syre, restarts.
- **New here, from full texts:**
  - the training mechanics of TRM / HRM: detached deep supervision, no-grad recursions, weight EMA, a BCE halt, AdamW settings;
  - their two 2025-26 follow-ups (Augmented HRM, DIS);
  - McLeish's retrofit recipe: a curriculum of the mean recurrence, Muon vs AdamW, healing;
  - Ouro's stability fixes and Stage-II gate;
  - Huginn's failed-run fixes;
  - the iCoT-SI / Coconut stage-switch hygiene;
  - Token Assorted's and CoLaR's randomised-mixture ablations;
  - general weight-averaging and mode-connectivity tools (EMA, WSD, soups, LMC).

**Source tags on every claim.**
- **[full text]:** read in the arXiv HTML (fetched with curl, converted to text locally, the relevant section read).
- **[abstract]:** from the abs page only. Method details are never stated from an abstract alone.
- **[our docs]:** a project file.
- **[LIT x]:** from an existing project note.

**Verification.**
- **All 64 arXiv IDs cited here were verified.** For each, the arxiv.org/abs page was fetched (raw HTML, `<title>` read): 2026-10-13, 21:55-22:10 EDT. The list is at the end.
- **One fabricated search summary was caught and discarded.** A WebSearch summary said STARS 2605.26733 uses "gradient truncation". Its abstract says Jacobian spectral-radius regularisation with random loop sampling.
- **One title differs between versions.** The search engine labelled 2511.16886 "Deep Improvement Supervision" (the method's name). The current abs title is "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator".

## 0. Our harness, in the terms that matter here [our docs: FACTS.md; boxes/fsq120.py make_opt / lr_at; nested170.py]

**Optimiser and schedule.**
- AdamW with default betas (0.9, 0.999) and ε 1e-8. Weight decay 0.01 on **all** parameters. One learning-rate group, lr 1e-4.
- Warm-up 100 steps, then cosine to 0 over 48k steps.
- Global clip 1.0, which binds on 65-100 % of steps.
- **Full BPTT through all K = 8 passes.**

**Seeds and outcomes.**
- **The init is deterministic.** A seed is the data order, plus rounding.
- **The seed lottery:** RMS s1 / s2 enter the basin at about 19k; s3 never does.
- **The selection model** (THEORY_SEEDSTAB Part II; DERIVED in its reduced model):
  - P(P basin) = Φ(μ_Z/σ_Z), with an observed z ≈ 0.25-0.43.
  - **√-levers** (batch, window, Grokfast) would need 15-42× changes.
  - **Tilt levers** (μ_Z) would need 4-7×.
  - The tilt terms are: C's open-loop head start T_h; the pass-token shortcut; P's late-pass reward b_P.
- **Part I bounds:**
  - **Thm 1 resolution bound (PROVED):** n ≤ 1 + log(L‖S₁−S₀‖/μ)/log(1/κ).
  - **Lemma 5.2 differential-mode attenuation (PROVED under smoothness):** in a κ-contractive regime a parameter step moves the pass-specific mode by about Bκ^r‖δθ‖, but the common mode by up to B‖δθ‖/(1−κ).
  - **Thm 2 integrator (PROVED, open loop):** bias ∝ t, noise ∝ √t.
  - **Thm 3:** a code-attractor successor map is exact for any n if η ≤ (1−κ)r.

## 1. Ranked list: general training improvements, best first

The rank weighs: our priority order (C1 > C6 > C4 = C2 > C3 > C5 > C7); whether a mechanism can be stated for *our* setup (the user rule); and cheapness.

### 1. DETACH-PASS: detached per-pass deep supervision (TRM / HRM / DIS) in place of BPTT across passes

**What it is.**
- Each pass's state is detached before the next pass: S_{t−1}.detach() enters pass t.
- Each pass is trained only by its own step-aux loss (and pass K also by the answer CE). Backward runs per pass, with gradients accumulated: one optimiser step per batch.
- HRM-faithful variant: one optimiser step per pass.
- Optional second form (TRM): k₁ no-grad passes, then gradient passes.

**Evidence.**
- **HRM:**
  - deep supervision detaches z between segments, with `loss.backward(); opt.step()` per segment [full text, Fig. 4 pseudocode];
  - it is "a 1-step approximation of the gradient of the recursive deep supervision process", with superior empirical performance and stability [full text];
  - the one-step gradient gives O(1) memory vs BPTT's O(T) [full text].
- **TRM** (Sudoku-Extreme test accuracy) [full text, Table 1]:
  - the same detached deep supervision (N_sup = 16);
  - within a step: T − 1 recursion processes without gradient, then one with full backprop;
  - full backprop through the recursion beats HRM's 1-step gradient: 87.4 vs 56.5;
  - truncating to the last k = 4 of 7 evaluations "did not help" [full text, "Ideas that failed"].
- **ARC Prize analysis, as quoted in TRM:** deep supervision doubled ARC-AGI accuracy, 19 % → 39% [TRM full text, citing the blog].
- **DIS 2511.16886:**
  - per-step intermediate targets with detached carries [full text];
  - "reduce the total number of forward passes by 18x" [abstract];
  - 3× fewer supervision steps and 8× fewer latent steps than TRM, ARC-AGI-1 24% at 0.8M parameters [full text].
- **Truncation in the retrofits:** Huginn and McLeish backprop only the last 8 recurrences [full text, both].
- **Caution:** iso-depth 2604.21106 finds truncation lowers φ [LIT_RAND_K].

**Generality.** Architecture-agnostic: any recurrence with a per-step target.

**Targets.**
- **Cost:** activation memory O(1) in K instead of O(K). Same FLOPs, unless k₁ no-grad passes are added: with k₁ = 4 of 8 that is about 0.67× pass compute.
- **The seed lottery (C6), via tilt.**

**Already tried?** No. Every section uses full BPTT (§155's gate demands it).
- Closest: **§176 PROG** (k₁ no-grad passes, then k₂ with gradient). It ran before step aux, when the memory was unused, so it is uninformative now.
- **§148 used TRM-style answer deep supervision** on a residual loop (not detached). Loops went unused.

**Math basis.**
- **(i) What the sources provide.**
  - HRM justifies its gradient via the IFT / 1-step DEQ gradient (a theorem whose assumption, convergence to a fixed point, TRM shows is violated).
  - TRM and DIS are empirics. DIS adds a policy-improvement derivation for its targets.
- **(ii) Mechanism in our setup.** Write L = Σ_t ℓ_t(S_t) and S_t = F_θ(S_{t−1}; x).
  - The full gradient is Σ_t ∂ℓ_t/∂S_t · Σ_{s≤t} (Π_{r=s+1}^{t} J_r) ∂_θF|_s.
  - DETACH keeps only s = t. The removed part R has norm ≤ Σ_t ‖∂ℓ_t/∂S_t‖ · B · κ/(1−κ) under κ-contraction (Thm 1's setting).
  - By Lemma 5.2, ∂_θF|_s is dominated by the **common (pass-independent) mode**.
    - In the open loop it is *exactly* common for the additive heads (Lemma 1: identical writes).
    - For the delta heads, the S_{s−1}·∂M/∂θ term varies with s, so it is only approximately common there.
  - So R is essentially the input-summary push that the C basin feeds on: late-pass losses reward a better common write by a factor up to 1/(1−κ) ≈ 5-10 at s3's κ of 0.8-0.9.
  - DETACH keeps the local term, which contains the loop-gain (progress) gradient once γ > 0. It deletes the amplified common-mode term.
  - In the race formula z = √(2a)/σ·[(b_P − β_C)/a + β_C e^{−r_C T_h}/(a + r_C)], that is a cut in C's drive β_C / r_C. **A tilt, not a √-lever.**
  - Tags: DERIVED-UNDER-ASSUMPTIONS for "R is common-mode and ×1/(1−κ)" (Lemma 5.2's assumptions: contraction, smooth F). CONJECTURE for "β_C is mainly carried by R".
  - **The cost of losing R: no long-range credit.** Each target e_{t+1} is the whole re-rendered expression, a sufficient statistic for every later target. So a model whose S_t decodes e_{t+1} for all t ≤ n carries all the information the answer needs (DERIVED: Markov targets).
  - **O(1) memory:** PROVED trivially (each pass's graph is freed after its backward).
- **(iii) Predictions.**
  - **(P1.a) [$0 CPU]** On s185_rms_s3's final model, the cross-pass share ‖g_full − g_D‖/‖g_full‖ on memory-branch parameters is ≥ 0.5, and R has cosine ≥ 0.8 with its pass-averaged (common-mode) surrogate. **Falsifier:** cosine < 0.5. Then R is not common-mode and the tilt argument fails.
  - **(P1.b) [one RTX 3090 cell, about $3]** s3 + DETACH (exact same-seed control s185_rms_s3, as in §196):
    - at 8k the donor-start λ_eff is on the P side (≥ −0.009, THEORY_SEEDSTAB Part II), where the control is ≤ −0.023;
    - entry(0.5) ≤ 25k.
    - **Falsifier:** λ_eff(8k) ≤ −0.023 and no entry by 48k.
  - **(P1.c)** An s1 guard stays DEEP (≥ 0.90 at n 3..8, K′ 8). If it is not, the lost long-range credit matters and the Markov-target argument is wrong for this model.

**Cost in our harness.**
- About 20-40 lines in nested170: a `--detach_pass` flag, and per-pass backward when the flag is set.
- Gates:
  - flags-off bitwise;
  - gradflow per term, plus an assert that the cross-pass gradient is exactly 0;
  - a memcheck.
- **A side benefit to check:** bs-64, K = 8 may fit on the 8 GB local cards (full runs locally would be free compute). Memcheck at n = 8, K = 12 before claiming it.

**Cheap test.** (P1.a) on CPU (3 threads, Idle, one batch of 16 items); then the memcheck. Only then the $3 cell.

### 2. Weight interpolation between seeds: linear mode connectivity as a lottery diagnostic, and "seed soups"

**What it is.**
- θ(α) = (1−α)θ_a + αθ_b between two final models fine-tuned from our identical deterministic init.
- Evaluate loss, accuracy and the order parameter along α.
- If two converged seeds are connected, their average is a "soup".

**Evidence.**
- **Neyshabur 2008.11687:** "when training from pre-trained weights, the model stays in the same basin … instances … close in parameter space" [abstract].
- **Model soups 2203.05482:** fine-tuned models "often appear to lie in a single low error basin"; averaging improves accuracy and OOD [abstract].
- **Frankle 1912.05671:** stability to SGD noise is decided early [LIT_SEED_STABILITY].
- **Augmented HRM 2601.10679 (Sudoku-Extreme)** [full text, Table 1]:
  - majority vote over 10 checkpoints from the second half of one run: 55.0 → 64.2 (+9.2);
  - so even same-run weight-space neighbours make complementary "guesses".

**Generality.** Fully general.

**Targets.** C6 (the seed lottery): diagnosis first, a possible fix second. C2: soups improve OOD, which bears on n 9-12.

**Already tried?** No (no "interpolat", "soup" or "mode connect" in THEORY or LIT notes).

**Math basis.**
- **(i) What the sources provide.** Empirics. Soups add an analytic relation: weight averaging ≈ logit ensembling under flatness [abstract].
- **(ii) Mechanism in our setup.** The §199 Part II reduced model has two attractors of the selection variable m (a cubic drift), with the init at the unstable point m₀ ≈ 0 (DERIVED-UNDER-ASSUMPTIONS).
  - If m is approximately linear in θ along the chord (assumption), then a P↔C chord crosses m = 0. **Prediction:** a loss barrier and a sign change of λ_eff at some α*.
  - A P↔P chord stays at m > 0. **Prediction:** no barrier and λ_eff in the P band throughout.
  - The other outcome, s3 ↔ s1 with no barrier and λ_eff moving monotonically, would mean s3 is a *slow P* in the same basin. The lottery is then a speed problem, and budget or extension is the lever, not tilt.
  - Tags: DERIVED-UNDER-ASSUMPTIONS (reduced model plus the linear-m chord); CONJECTURE for real weights.
- **(iii) Predictions** (models s185_rms_s1 / s2 / s3, with s184 / s187 / s190 converged models as extra P points):
  - **P-P:** barrier < 0.1 nats on train loss, and λ_eff ∈ [−0.009, +0.004] at all α ∈ {0, .25, .5, .75, 1}.
  - **P-C:** a barrier ≥ 0.3 nats and a λ_eff sign change.
  - **Soup:** the s1 + s2 soup's pooled n 3..8 accuracy at K′ 8 ≥ min(s1, s2) − 0.01, and n 9-12 at K′ = n + 2 ≥ the mean of s1 and s2.
  - **Falsifiers:** a P-P barrier ≥ 0.3, so the converged seeds sit in different basins; this contradicts the one-attractor-per-class picture. Or no P-C barrier at all.

**Cost in our harness.** Eval only. 3 pairs × 5 α × (loss on 512 train items + 512 test items + path_indep) ≈ 1 h on one idle local card, between runs (no co-tenants). The script is about 60 lines; it loads model.pt state dicts and averages every tensor.

**Cheap test.** As above, $0. Read before §199's methods launch, since it decides between "tilt" and "speed".

### 3. Stage-switch hygiene for BUDGET-ANNEAL (iCoT-SI, Coconut, McLeish, Token Assorted)

**What it is.**
- For the registered-in-waiting BUDGET-ANNEAL (196-1: n ≤ 4 / K 4 → 6 / 6 → 8 / 8, warm-started *with* optimiser state):
  - (a) use a continuous schedule (linear or 1-√) of n_max and K instead of hard stages;
  - (b) a small lookahead tail: a few % of items from beyond the current cap ("removal smoothing");
  - (c) a measured decision on resetting AdamW's moments at each switch.

**Evidence.**
- **iCoT-SI 2405.14838** (7×7 multiplication, GPT-2 small) [full text]:
  - optimiser reset at every removal; without it, accuracy "drops to zero around 100 steps and does not recover";
  - removal smoothing (P(o) ∝ e^{−λo}, λ = 4: 98% o = 0) is "crucial";
  - removing Δ = 16 tokens per epoch fails to converge, while 8 works.
- **Coconut 2412.06769** (GSM8K) [full text]:
  - resets the optimiser at stage switches;
  - adding 3 thoughts per step (c = 3) gives a training-loss spike at the final stage transition, a slight drop and more variance;
  - w/o curriculum 14.4 vs 34.1.
- **McLeish 2511.07384** [full text]:
  - linearly scheduling the *mean* recurrence (Poisson-lognormal) helps loss slightly per step and greatly per FLOP;
  - 1-√ is as good as linear;
  - scheduling the backprop depth instead is better per FLOP, worse per step.
- **Token Assorted 2502.03275** [full text, Table 4.4]: one stage with a random replacement count m per sample beats a staged curriculum.
  - Average over all datasets: 17.8 vs 7.8 (Llama-3.2-1B) and 28.1 vs 10.9 (3B).
  - GSM8K: 48.7 vs 9.8 (1B) and 73.8 vs 14.9 (3B).
- **CoLaR 2505.16552:** a random compression factor per step beats single-factor training at every test factor except c = 5 [full text, §4.7].

**Generality.** General: any curriculum over difficulty or depth.

**Targets.** C6 (the lottery: curriculum is the one lever the theory says can rescue a seed already on the C side, by "escape", THEORY_SEEDSTAB §2 "Not levers"). C2.

**Already tried?**
- A curriculum pilot (n ≤ 4, K 4) made s3 enter by 10k [FACTS].
- BUDGET-ANNEAL is drafted with the optimiser state *kept* [our docs: note 196-1 (a)].
- No smoothing and no reset arm.

**Math basis.**
- **(i) What the sources provide.** iCoT-SI gives a heuristic (Adam's second-moment estimate is stale after a loss change). McLeish, Token Assorted and CoLaR are empirics. Graduated optimisation has a theorem for a function class (Hazan 1503.03712) [LIT_SEED_STABILITY].
- **(ii) Mechanism in our setup.**
  - **(a) Stale moments: DERIVED-UNDER-ASSUMPTIONS** (stationary gradient statistics within a stage, a jump at the switch).
    - With β₂ = 0.999, v̂ has a 1/(1−β₂) = 1000-step memory. After the per-coordinate gradient scale jumps by c, Adam's step is inflated by about c/√(β₂^k + (1−β₂^k)c²): c at the switch, relaxing over about 1000 steps.
    - **But our clip binds on 65-100% of steps.** The global norm entering Adam is about 1 both before and after the switch. So the overshoot comes only from *redistribution* across coordinates, not from global scale.
    - DERIVED: the reset matters much less here than in iCoT-SI.
  - **(b) Smoothing as continuation: CONJECTURE.**
    - Thm 1 makes the resolvable depth n ≤ 1 + log(L d/μ)/log(1/κ) a continuous function of κ.
    - A continuous n_max(step) keeps the required κ just above the achieved κ (a homotopy).
    - A hard 4 → 6 jump raises the requirement from (μ/Ld)^{1/4} to (μ/Ld)^{1/6} at once, and the new items' gradient arrives as a burst.
    - The theory's P5 says curriculum rescues by escape (some seeds visit the C side and cross), which makes the transitions the risky moments.
- **(iii) Predictions** (196-1's local pilot, descriptive, from the kept stage-1 checkpoint, three stage-2 legs: hard + keep moments, hard + reset, smooth + lookahead 3%):
  - (a) the post-switch loss spike (max 200-step loss / pre-switch loss) is < 1.3 in all legs, because the clip neutralises it; reset changes it by < 0.1;
  - (b) the smooth leg keeps handle accuracy at the new top n ≥ 0.9 through the transition, where the hard legs dip ≥ 0.1.
  - **Falsifiers:** a spike ≥ 2 in the keep-moments leg that reset removes, so the clip does not neutralise it; or no difference between smooth and hard.

**Cost in our harness.** `--anneal` exists (196-1 a). Adding a "smooth" schedule kind and `--anneal_reset_opt` is about 15 lines. The pilot runs locally ($0; the 4060's 196 checkpoints are kept).

**Cheap test.** As above, $0. It sets BUDGET-ANNEAL's switch design before the cloud cell.

### 4. A correctness-labelled halt head, trained on a frozen model (TRM's BCE halt; Ouro Stage II)

**What it is.**
- A small head on the memory read predicts whether the answer decoded from S_t is correct (TRM), or whether the next pass still improves the answer loss (Ouro).
- It is trained after the main run, on frozen weights. At test, the model halts at the first t with q_t > τ.

**Evidence.**
- **TRM** [full text]:
  - replacing HRM's Q-learning ACT (halt + continue losses, a second forward pass) with BCE on "reached the correct solution" changes Sudoku-Extreme 86.1 → 87.4 with one forward pass;
  - removing ACT from *training* hurt generalisation.
- **HRM:** Q-halt with reward 1{ŷ = y}, and a randomised minimum M_min [full text].
- **Ouro Stage II** [full text]:
  - freeze the LM; train only the gate with BCE against w = σ(50(I − 0.005)), where I = the detached per-step loss improvement;
  - best accuracy at every compute budget on MMLU (66 % vs about 64% at an average of 2.5 exits).
- **Ouro Stage I:** the entropy regulariser = KL to a uniform exit prior (an exact identity, [full text]). Geometric priors plateau higher and collapse to shallow depths [full text, App. A].
- **Unsupervised ACT / PonderNet collapse to one loop** in RecurTrace's controlled test [LIT_FIVE_LEVERS].

**Generality.** General.

**Targets.** C5, and C6: s1's K′ 16 accuracy at n = 8 is 0.75 vs 0.99 at K′ 8, and halting before the post-completion passes avoids that. C4-clean: the labels come from final answers only.

**Already tried?** Not this form.
- §173 answer-settling halts read PREMATURE (173-2 / 173-3).
- AUX-DONE is supervised by 1{t ≥ n}: a counter at the edge (187-4).
- "#3 halt head" is listed as unregistered in CURRENT_TASKS.

**Math basis.**
- **(i) What the sources provide.** Empirics (TRM, Ouro). The ELBO / uniform-prior identity is exact. HRM's Q-learning stability argument cites bounded-parameter convergence theory.
- **(ii) Mechanism in our setup. DERIVED (trivial bound):**
  - If the halt step t̂ ∈ [n, K_train] for an item, the halted answer is the K′ = t̂ answer.
  - So halted accuracy ≥ min_{K′ ∈ [n, 8]} acc(K′) on items where the head is in range. This avoids Thm 2's post-completion drift.
- **CONJECTURE: the correctness signal is readable.**
  - The Thm 2 closed-loop margin p_t ≈ Φ((μ − t b̄)/(√t σ̄)) is a smooth function of the read state.
  - Outside n ≤ 8 the head inherits COUNT-OOD (187-3), so premature halts are expected at n 9-12.
- **(iii) Predictions** (on s185_rms_s1 / s2, WINDOW s1 / s2, frozen; the head is fit on train items, n ≤ 8):
  - pooled n 3..8 with halting (cap 16) ≥ acc(K′ 8) − 0.01;
  - mean passes ≤ n + 1.5;
  - at n 9-12: halted acc ≤ acc(K′ = n + 2), with the premature share reported (the expected failure).
  - **Falsifier:** halted pooled n 3..8 < acc(K′ 8) − 0.03. The correctness signal is then not readable from S_t.

**Cost in our harness.** Eval only:
- per-pass answer decoding t = 1..16 on about 2k train + 1k test items per model (one decoder pass per t);
- a logistic-regression head on read features on CPU.
- Under 1 h on a local card per model. About 80 lines of scratch script.

**Cheap test.** As above, $0. Pre-register it as eval-only, like §173.

### 5. Pass count drawn independently of problem size, with hold targets (Kuo; CoLaR; Huginn / McLeish)

**What it is.** K ~ U{K_lo..K_hi} with K_lo ≥ n_max = 8 (for example U{8..24}) for **every** item, with HOLD targets for t > n. This replaces WINDOW's K = n + U{1..4}.

**Evidence.**
- **Kuo 2606.29983:**
  - looped-transformer OOD variance is traced to the spurious correlation between sequence length and loop count;
  - stochastic loop counts "sharply reduce OOD variance" [abstract].
- **CoLaR:** random-factor training beats fixed-factor training, and interpolates to unseen factors (it fails beyond the trained maximum) [full text].
- **Huginn / McLeish:** Poisson-lognormal counts [full text].
- **STARS 2605.26733:** random loop sampling [abstract].

**Generality.** General.

**Targets.** C6 (post-completion stability past K′ 24), and C2 seed spread (n 9-12: 0.486 / 0.665 across §184 seeds).

**Already tried?** Partly.
- §176 RK (Huginn sampler) was pre-aux on multiplication, with the memory unused: uninformative now.
- §182 WINDOW is size-coupled. It is STABLE in s1 uncapped, and s2's pooled K′ 32 is 0.996 [our docs].
- §188 / §193 add REWEIGHT / RMS.
- LIT_S188 already flagged WINDOW as Kuo's correlation.

**Math basis.**
- **(i) What the sources provide.** Empirics (Kuo has a causal story, not a theorem).
- **(ii) Mechanism in our setup.**
  - Under WINDOW, the read-out pass index K carries I(K; n) > 0 about size. That is the cue a size-keyed computation (187-3 COUNT-OOD) can use.
  - With K ⟂ n, I(K; n) = 0 by construction (PROVED). Only the input itself still carries size.
  - CONJECTURE: this narrows the seed spread at n 9-12.
  - **For the lottery it is a √-lever** (window length; THEORY_SEEDSTAB Part II), so no lottery gain is predicted (DERIVED-UNDER-ASSUMPTIONS).
- **(iii) Predictions:**
  - STABLE at K′ 24 and 32 (≥ K′ 8 − 0.02) in 2 / 2 seeds;
  - |s1 − s2| at n 9-12 (K′ = n + 2) < 0.10, against §184's 0.18;
  - the entry rate is not better than RMS's 2/3.
  - **Falsifier:** a seed spread ≥ 0.18, so the size cue is not the pass count.

**Cost in our harness.** `--k_window` already exists. A `--k_indep lo,hi` flag is about 10 lines. One 3090 cell is about $3-4: mean K 16 vs WINDOW's about 7, so about 2× per step. **Not under $3 for a full cell.**

**Cheap test.** First an eval-only check on existing WINDOW s1 / s2: does I(K; n) matter? Probe whether the answer decoder's read encodes K vs n (a ridge probe on read features, CPU).

### 6. Weight EMA of the trained model (TRM, Huginn), optionally with a constant / WSD schedule

**What it is.**
- A shadow copy θ̄ ← βθ̄ + (1−β)θ (β = 0.999) is kept and evaluated beside the raw weights.
- WSD (warmup-stable-decay) keeps lr constant and cools down late, or uses the EMA instead of a cooldown.

**Evidence.**
- **TRM:**
  - "no EMA" 79.9 vs 87.4 [full text, Table 1];
  - EMA "prevents sharp collapse", where HRM "tends to overfit quickly and then diverge" on small data [full text].
- **Huginn:** constant lr. A post-hoc EMA over the last 75 checkpoints simulates a cooldown [full text].
- **EMA study 2411.18704:** better generalisation, prediction consistency, calibration and robustness to label noise; it "requires less learning rate decay" [abstract].
- **WSD 2405.18392:** constant lr + cooldown scales like cosine; weight averaging improves along the trajectory; runs become reusable [abstract].
- **LAWA 2209.14981:** averaging the latest checkpoints speeds up progress [abstract].

**Generality.** Fully general.

**Targets.**
- C6 for the *final model* (post-completion drift, K′ 16).
- Cost via WSD:
  - an entered run (entry about 19k) can be cooled at about 28k instead of running a fixed 48k cosine;
  - a non-entered run can be extended without an lr restart.

**Already tried?** No (no weight EMA anywhere; Grokfast's EMA is of gradients, §196).

**Math basis.**
- **(i) What the sources provide.** Empirics. For constant-lr SGD near a quadratic minimum, iterates are a stationary OU process (Mandt 1704.04289 [abstract]), and their EMA has a lower variance (a standard OU computation).
- **(ii) Mechanism in our setup.**
  - **PROVED (trivial):** a shadow EMA never feeds back into training, so **it cannot change P(P basin).** No lottery claim is made.
  - **For C6, CONJECTURE:** Thm 3's margin condition η ≤ (1−κ)r. If part of the per-pass write error η is parameter noise around the basin minimum, the EMA shrinks it by about √((1−β)/(1+β)) relative to the raw iterate (DERIVED for OU).
  - Our cosine already decays lr to 0, so the residual noise is small, and the predicted gain is small (DERIVED).
  - **WSD: window length is a √-lever** (Φ(b√T_w/σ)), so a longer high-lr window buys little selection. Its value is accounting (stop or extend per run), not tilt.
- **(iii) Predictions:**
  - on the next cloud cell with a free-riding EMA: EMA vs raw differ by ≤ 0.01 at K′ 8 and by ≥ 0 at K′ 16 / 24;
  - s3-type runs stay at their raw outcome.
  - **Falsifier:** EMA ≥ raw + 0.05 at K′ 16. The post-completion failure is then noise-dominated, and the next lever is averaging, not HOLD.

**Cost in our harness.** About 15 lines: a shadow state dict, updated after `opt.step()`, saved in the checkpoint and evaluated at the end.
- Memory: +0.5 GB fp32. Time: < 2 %.
- **Training stays bitwise identical**, so it can ride along on any registered cell (gate: flags-off and flags-on training losses identical).

**Cheap test.** Ride along on the next registered 3090 cell: $0 marginal, plus one eval.

### 7. Check the trunk's language ability; a "healing" LM mix before natural-language transfer (McLeish; LoRA forgetting)

**What it is.**
- Measure GPT-2-trunk perplexity on general text after our synthetic fine-tuning.
- If it has degraded, mix 5-10% general-text LM batches into training (a healing mix) before the GSM8K / ProsQA-language steps.

**Evidence.**
- **McLeish:**
  - after model surgery, a "healing" period with minimal distribution shift helps recover basic language modelling, +5% Arc-C for the recurrent model [full text];
  - a pretrained init strongly beats random init per FLOP; the loss curves would cross only around 950B tokens (extrapolated) [full text].
- **Biderman 2405.09673:** LoRA learns less and forgets less, and forgets less than weight decay or dropout [abstract].
- **The frozen-backbone latent reasoners:** CoLaR (frozen Llama-3.2-1B + LoRA r128 [full text]); PCCoT (LoRA r128 [LIT_PCCOT_PROBE]); SoftCoT (frozen LLM, "full-model fine-tuning … catastrophic forgetting" [abstract]).
- **A sibling G-file measured weight drift:** trunk_drift.json shows relative trunk change 0.10-0.12 in every model (converged or failed), with wte 0.09-0.15. These are weights, not behaviour.

**Generality.** General.

**Targets.** GSM8K transfer, and C4 (distilling from a text-CoT teacher needs a trunk that still reads language).

**Already tried?** No (no perplexity or forgetting measurement in THEORY).

**Math basis.**
- **(i) What the sources provide.** Empirics.
- **(ii) Mechanism in our setup. DERIVED-UNDER-ASSUMPTIONS** (θ₀ is a minimiser of L_LM, quadratic regime):
  - L_LM(θ₀ + Δ) ≈ L_LM(θ₀) + ½ ΔᵀFΔ, with F the Fisher (≈ Hessian at θ₀).
  - So a healing mix at weight λ adds λ·½ΔᵀFΔ to our objective. That is EWC's penalty with the exact Fisher, computed implicitly (EWC 1612.00796).
  - **The measurement needs no mechanism.** The fix is principled.
- **(iii) Predictions:**
  - the perplexity ratio vs GPT-2 base on 200k OpenWebText / WikiText tokens, with memory disabled (W_o path off), is reported;
  - if it is ≥ 1.5, a healing mix is pre-registered before any GSM8K cell, with a ≤ 0.01 in-distribution cost.
  - **Falsifier for the derivation:** with the mix on, the perplexity ratio fails to fall below 1.2 while in-distribution accuracy is held. The quadratic regime does not apply.

**Cost in our harness.** CPU eval, about 20-40 min at 3 threads. The healing mix itself is about 30 lines (a second data stream).

**Cheap test.** The perplexity check: $0.

### 8. Muon (orthogonalised updates) for the recurrent matrices; AdamW* settings

**What it is.**
- Muon (Newton-Schulz-orthogonalised momentum) for the 2-D matrices of the memory branch (q / k / v / W_o, gates). AdamW for the rest.
- Or AdamW* (β₂ 0.95, update clipping, no ε).

**Evidence.**
- **McLeish** [full text, Fig. 4]:
  - Muon beats AdamW **for recurrent models**. AdamW at 5e-5 spikes to NaN; Muon at 1e-3 is "the most stable" with the lowest loss;
  - for the non-recurrent TinyLlama the difference is small.
- **Moonshot 2502.16982:** Muon SFT of an AdamW-pretrained model shows no significant advantage [full text, §3.5]. Hence Muon on the memory branch only.
- **Common practice, not ablated:** TRM (β₂ 0.95), Huginn (AdamW* with update clipping, no ε, β₂ 0.95, clip 1), Ouro SFT (β 0.9, 0.95), HRM (Adam-atan2, scale-invariant) [full text, all].

**Generality.** General.

**Targets.** Training stability. The lottery only if a mechanism holds (below).

**Already tried?** No (no Muon or β₂ change). §127 LR10 (10× lr) broke learners. Huginn fixed its collapsed runs partly by *lowering* peak lr 10× [full text], which is consistent.

**Math basis.**
- **(i) What the sources provide.** Empirics only. McLeish's evidence concerns loss spikes and divergence; our runs do not diverge.
- **(ii) Mechanism in our setup. CONJECTURE, and weak.**
  - Orthogonalisation equalises a matrix gradient's singular directions. If the progress (loop-gain) direction were a small but above-noise singular component, Muon would raise b_P relative to C's dominant direction (a tilt).
  - **But a rank-1 signal in noise gains no clear advantage over Adam** in a simple spiked-matrix model. Both resolve it only above the noise floor (my derivation; not robust).
  - So **no mechanism for basin selection can be stated with confidence.** Ranked down per the user rule.
- **(iii) Prediction, if run:**
  - s3 + Muon(memory) with exact control: entry(0.5) ≤ 25k (ACCELERATES), or NO-EFFECT, with the modal outcome NO-EFFECT;
  - the clip fraction falls below 50% (a stability signature) whatever the entry.

**Cost in our harness.** About 40 lines (KellerJordan/Muon is one file) plus param groups. A $3 cell. **Run only if §196 Grokfast reads NO-EFFECT and DETACH also fails.**

### 9. A stochastic memory write + GRPO from a converged model, for depth beyond training (CoLaR; HRPO; SofT-GRPO)

**What it is.**
- Make the delta-head value write Gaussian (v + σε, with a learned σ initialised small).
- Then run outcome-reward GRPO (group 8, no KL) on n 9-10 items, starting from a converged RMS model.

**Evidence.**
- **CoLaR:**
  - a probabilistic latent head; a deterministic head "fits well on simple math datasets but lacks test-time exploration potential";
  - GRPO with a 0/1 reward and no KL [full text];
  - RL gives +5.4% on MATH with 82.8% shorter chains [abstract].
- **HRPO:** the gate starts on token embeddings and progressively mixes in hidden states under RL. Larger weight decay for stability [abstract + full text, App.].
- **SofT-GRPO 2511.06411:** Gumbel-reparameterised soft-thinking RL [abstract].
- **Caution:** 2512.11816 finds RL-trained latent models still lag language CoT on maths [abstract]. Every successful latent-RL paper starts from a supervised model [LIT_GENERAL_LATENT_MAP].

**Generality.** General (any latent with a sampled component).

**Targets.** C2 (n 9-12 beyond training, where no step labels exist), and strict C4 (answers only).

**Already tried?**
- §69: RL from scratch failed.
- The FSQ-code GRPO recommendation (LIT_GENERAL_LATENT_MAP) was not run on DeltaMem.

**Math basis.**
- **(i) What the sources provide.** The policy-gradient theorem: unbiased for a stochastic policy (PROVED, standard). GRPO's clipping is heuristic.
- **(ii) Mechanism in our setup.**
  - At n 9-12 the converged models score 0.49-0.66. So the per-item success rate p sits near ½, where the group-relative advantage variance p(1−p) is maximal: a dense reward signal exactly where step labels are missing (DERIVED).
  - CONJECTURE: the reward credits the passes past n = 8, the ones the step aux never trained.
  - **Risk from the theory:** noise injection lowers z in *selection*. But here the start is already in P, so selection is not at stake (DERIVED-UNDER-ASSUMPTIONS: the P basin is stable to small noise, per Thm 3's margin).
- **(iii) Prediction:**
  - after 500 GRPO steps, n 9-12 at K′ = n + 2 rises by ≥ 0.05;
  - n ≤ 8 at K′ 8 drops by ≤ 0.01;
  - donor SERIAL is kept.
  - **Falsifier:** no gain with p ≈ 0.5. The deep-item failure is then not reachable by local exploration of the write.

**Cost in our harness.** About 120 lines (a sampler, log-probabilities of the Gaussian writes, a GRPO loop). A local short job (an 8 GB card, bs small). $0.

### 10. Final-answer-derived step targets by monotone corruption (DIS), for multi-token answers

**What it is.**
- Pass t's target is the final answer with a decreasing fraction of tokens masked (a linear schedule).
- For multiplication: reveal product digits least-significant first.
- No hand-built intermediate states.

**Evidence.**
- **DIS** [full text]:
  - targets come from a linearly decreasing mask rate over 6 steps;
  - LLM-generated intermediate targets gave only 10% vs 24% for corruption targets on ARC-AGI-1;
  - on Sudoku (MLP backbone) it matches TRM's final score but learns more slowly.

**Generality.** General for multi-token outputs. Not applicable to single-token answers (nested mod 11, ProsQA).

**Targets.** C4: dense per-pass targets without simulator states. This is the revised-C4-safe version of step aux for multiplication (§161's input schedule is hand-given, per NORTH_STAR C1).

**Already tried?** No. ICoT-style removal (§142 / §152) is output-side text removal, not corruption targets.

**Math basis.**
- **(i) What the sources provide.** DIS derives that a supervised improvement target makes each recursion a policy-improvement step (a derivation in the paper, under its assumptions).
- **(ii) Mechanism in our setup.**
  - For long multiplication, output digit j depends only on columns ≤ j and the incoming carry. So an LSB-first reveal is consistent with a serial schedule (DERIVED).
  - In the selection model, a target at every pass raises b_P at late passes (a tilt, per THEORY_SEEDSTAB's "late-pass reward"). CONJECTURE.
- **(iii) Prediction** (on §161's 4 × n mix):
  - a DIS-target arm is EXACT at 4×4 by 48k in ≥ 1 of 2 seeds, with SERIAL row donors;
  - ext 4×8 is reported.
  - **Falsifier:** test_acc ≤ the FINAL-only baseline. The reveal order then does not shape the memory.

**Cost in our harness.** About 30 lines in the multiplication box (target construction). One 3090 cell (about $3).

### 11. Damped post-completion passes as an eval-only diagnostic of the K′ 16 failure (2609.16665; 2605.23872)

**What it is.** At eval, for passes t > n (oracle n), scale the memory increment: S_t = S_{t−1} + η·(F(S_{t−1}) − S_{t−1}), with η ∈ {1, 0.5, 0.25}.

**Evidence.**
- 2609.16665: a fixed quarter step gives positive gains for 72.2-83.2% of selected finite-step failures in looped transformers [abstract].
- 2605.23872: training-free loops work only with damped sub-steps (a forward-Euler view) [abstract].
- §148 used η = 0.25 inside a residual loop [our docs].

**Generality.** General.

**Targets.** C6 diagnosis only (a hand-set, oracle-timed scaffold, not a claimable method).

**Already tried?** Not on the memory loop.

**Math basis.**
- **(i) What the sources provide.** 2609.16665 gives a local quadratic model with curvature-variation bounds [abstract]. 2605.23872 is a heuristic.
- **(ii) Mechanism in our setup.**
  - Thm 2a: post-completion error accumulates as t·b (bias) + √t·σ (noise), with b, σ ∝ the increment size. So damping by η scales the post-completion drift by η (DERIVED-UNDER-ASSUMPTIONS: the post-completion writes are small and roughly open-loop).
  - THEORY_SEEDSTAB already falsified a pure j/n account, so this is a live test.
- **(iii) Prediction:**
  - if the drift is bias-like, K′ 16 pooled accuracy at η = 0.25 recovers to within 0.02 of K′ 8;
  - if it does not recover, the failure is a direction error (wrong computation on the completed code), and Thm 3's margin, not the step size, is the target.

**Cost in our harness.** Eval only, about 20 lines. $0.

### Considered and ranked out (a mechanism exists, but it predicts weakness, or no mechanism)

- **Larger batch / gradient-noise scale** (Ouro: batch 4M → 8M tokens "particularly important for recurrent architectures" [full text]; McCandlish 1812.06162 [abstract]; stochastic collapse 2306.04251 [LIT_SEED_STABILITY]):
  - batch is a **√-lever**: about 15-42× would be needed for P = 0.95 (THEORY_SEEDSTAB lever theorem, DERIVED-UNDER-ASSUMPTIONS);
  - not recommended for the lottery.
- **Checkpoint / seed answer ensembles** (Augmented HRM bootstrapping):
  - accuracy only; an ensemble is not one memory, so C1 tests do not apply;
  - with correlated voters, Condorcet gains are small (DERIVED for positive error correlation).
- **Random initial state / path-independence training** (Huginn s₀ ~ truncated normal "promotes … path independence" [full text]; Anil 2211.09961 [abstract]):
  - **conflicts with our mechanism.** Every converged model is path-*dependent* and every failed one contractive (FACTS);
  - for an additive no-decay integrator, "forget the start" means forgetting progress (THEORY_SEEDSTAB §3.4: harmful pre-completion forgetting along progress).
  - Only transverse contraction is allowed (Thm 3).
- **STARS 2605.26733** (Jacobian spectral-radius regularisation towards stable fixed points [abstract]):
  - the same conflict, if applied pre-completion;
  - §199's CSR is the sign-switched version.
- **Grokfast-style symmetric amplifiers:** §196 is running; the theory classes them as √-levers.
- **β₂ = 0.95 alone, LayerScale / identity-biased gates, FPRM residual scaling:**
  - LayerScale / identity gates and FPRM are covered in LIT_DEPTH_RECURRENT_CHEN_HE and LIT_DEEP_PASS;
  - none has an isolated ablation or a mechanism for selection.

### Already in our recipe (literature counterparts; not re-proposed)

| Ingredient | Where it is in our harness |
|---|---|
| Input injection: identical input every pass (Bansal recall, Huginn, Fan) | INJECT §169 |
| Per-pass generative step supervision (SIM-CoT; LoopMTP 2608.03624 [abstract]; DIS) | §175 STEP-AUX |
| Equal per-pass signal (Ouro's expected loss over exits) | §184 REWEIGHT |
| Scale-normalised read (readout blind spot 2606.24898 [abstract]) | §185 RMS / §189 |
| A hold-the-answer target after completion (TRM / HRM answer deep supervision; Augmented HRM data mixing restores the fixed-point property 55.0 → 59.9 [full text]) | §182 HOLD |
| Function-preserving zero-init branch (HRPO's token-first gate) | W_o = 0, fade |
| Cached lower blocks | Penelope, `--cache_lower` |
| Gradient EMA filter | §196 |
| Teacher-state alignment (CODI) | §135 (FAIL) |
| Pass-index removal | §187 NOPASS |
| Restarts | §126 / §131 |

## 2. Looks good, but would break C1's clean test (identical input every pass; memory the only cross-pass channel)

1. **Feeding latents or the current answer back as input tokens.**
   - Coconut and CODI recycle latents into the context; TRM / DIS carry a separate answer state y, and the input changes per pass.
   - That makes a second cross-pass channel through attention / KV, which is exactly the V-cache bypass SCIT found in CODI [LIT_GENERAL_LATENT_MAP].
   - **OK only if y lives inside the memory.**
2. **Cross-loop KV sharing or replay:** MELT, LLA, RLT, MoR's KV reuse for exited tokens [LIT_CHEAPER_ARCH], and T²MLR's cross-token mid-layer state.
3. **A pass-index embedding at the input** (UT timestep, loop-index LoRA, SIM-CoT's "Step k" prompt in the main path).
   - The input then differs per pass, which also breaks `--cache_lower`.
   - A pass signal may enter only at or above the memory layer [LIT_CHEAPER_ARCH note 7]. NOPASS (§187) shows it also costs depth.
4. **Test-time optimisation of the latent:** LatentSeek 2505.13308 and LTPO 2510.04182 (policy gradient on latent vectors per instance [abstract]); SoftCoT++'s multi-init + contrastive perturbation [abstract].
   - The state is then edited by an outside optimiser, so donor and removal tests no longer measure the model's own serial use.
5. **Hybrid text / latent reasoning in the trace:** Token Assorted, HRPO's sampled tokens, PCCoT's in-context latents.
   - The text or the in-context latents become a channel besides the memory.
   - **Token Assorted is safe only as a *training* augmentation** that writes m solution steps into the (constant) input prefix, with m = 0 at test.
6. **Answer deep supervision at every pass before completion** (TRM / Ouro at t < n).
   - It rewards answering before the serial computation is done, i.e. a one-pass shortcut. That undermines THINK-NECESSARY.
   - Chen 2603.21676's harm is exactly this [LIT_S187_NOPASS].
   - **Keep it masked to t ≥ n (HOLD).**
7. **Ensembles or soups reported as the C1 model.** Ensembling answers across checkpoints or seeds is fine for accuracy. C1 / donor tests must run on single models (a soup is one model and is fine).

Mechanism conflicts that do *not* break C1 but contradict our data: random s₀ / path-independence training and STARS-style global contraction (see "ranked out").

## 3. The three highest-value recommendations

1. **DETACH-PASS** (rank 1):
   - **The case:**
     - the only literature ingredient with a stated *tilt* mechanism for our lottery: it removes the BPTT cross-pass term, which is common-mode and ×1/(1−κ)-amplified under Lemma 5.2;
     - with Markov step targets it should cost nothing in information;
     - it gives O(1)-in-K memory, possibly making full runs fit on the 8 GB local cards.
   - **Sources:** TRM 2510.04871, HRM 2506.21734, DIS 2511.16886 [full text].
   - **Order:** the $0 CPU gradient decomposition (P1.a) and a memcheck first; then one s3 cell with the exact same-seed control (about $3), plus an s1 guard.
2. **Seed-interpolation (LMC) diagnostic, and soups** (rank 2):
   - **The case:** eval only, $0, on existing s185_rms_s1 / s2 / s3. It decides whether s3 sits in a separate basin (the tilt picture) or is a slow member of the same basin (a speed / budget picture) **before** §199's methods spend GPU. A connected P-P pair also yields a seed soup, a free C6 / C2 candidate.
   - **Sources:** 2008.11687, 2203.05482, 1912.05671; 2601.10679's checkpoint bootstrapping.
3. **Stage-switch hygiene for BUDGET-ANNEAL** (rank 3):
   - **The case:** curriculum is the one lever the theory says can rescue a seed already on the C side, and the n ≤ 4 pilot made s3 enter by 10k.
   - **The design rule:** a continuous schedule (linear / 1-√, McLeish 2511.07384) with a small lookahead tail (iCoT-SI removal smoothing, 2405.14838) and a measured, clip-aware decision on AdamW resets (iCoT-SI / Coconut 2412.06769 reset; under our binding clip the derived effect is small). Prefer per-sample mixtures over hard stages (Token Assorted 2502.03275: 48.7 vs 9.8 on GSM8K, 1B).
   - **Order:** $0 local legs from the kept 196 checkpoints.

**Free riders, $0:** a weight-EMA shadow on the next registered cloud cell (rank 6; no lottery claim); the trunk perplexity check before any GSM8K step (rank 7); the eval-only correctness halt head (rank 4).

## Verified IDs (abs page fetched 2026-10-13; exact titles)

| ID | Title |
|---|---|
| 1603.08983 | Adaptive Computation Time for Recurrent Neural Networks |
| 1612.00796 | Overcoming catastrophic forgetting in neural networks |
| 1704.04289 | Stochastic Gradient Descent as Approximate Bayesian Inference |
| 1812.06162 | An Empirical Model of Large-Batch Training |
| 1909.01377 | Deep Equilibrium Models |
| 1912.05671 | Linear Mode Connectivity and the Lottery Ticket Hypothesis |
| 2008.11687 | What is being transferred in transfer learning? |
| 2103.12803 | JFB: Jacobian-Free Backpropagation for Implicit Networks |
| 2107.05407 | PonderNet: Learning to Ponder |
| 2111.05177 | On Training Implicit Models |
| 2202.05826 | End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking |
| 2203.05482 | Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time |
| 2209.14981 | Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging |
| 2211.09961 | Path Independent Equilibrium Models Can Better Exploit Test-Time Computation |
| 2306.04251 | Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2405.09673 | LoRA Learns Less and Forgets Less |
| 2405.14838 | From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step |
| 2405.18392 | Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations |
| 2409.15647 | Looped Transformers for Length Generalization |
| 2410.20672 | Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA |
| 2411.18704 | Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits |
| 2412.06769 | Training Large Language Models to Reason in a Continuous Latent Space |
| 2412.13171 | Compressed Chain of Thought: Efficient Reasoning Through Dense Representations |
| 1503.03712 | On Graduated Optimization for Stochastic Non-Convex Problems |
| 2502.03275 | Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning |
| 2502.05171 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach |
| 2502.12134 | SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs |
| 2502.16982 | Muon is Scalable for LLM Training |
| 2502.21074 | CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation |
| 2505.11484 | SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning |
| 2505.13308 | Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space |
| 2505.16552 | Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains |
| 2505.18454 | Hybrid Latent Reasoning via Reinforcement Learning |
| 2506.18582 | Parallel Continuous Chain-of-Thought with Jacobi Iteration |
| 2506.21734 | Hierarchical Reasoning Model |
| 2507.10524 | Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation |
| 2509.20317 | SIM-CoT: Supervised Implicit Chain-of-Thought |
| 2510.04182 | Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization |
| 2510.04871 | Less is More: Recursive Reasoning with Tiny Networks |
| 2510.07358 | Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts |
| 2510.25741 | Scaling Latent Reasoning via Looped Language Models |
| 2511.06411 | SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization |
| 2511.07384 | Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence |
| 2511.16886 | Latent Reasoning in TRMs is Secretly a Policy Improvement Operator |
| 2512.11816 | Reinforcement Learning for Latent-Space Thinking in LLMs |
| 2601.10679 | Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models |
| 2602.01148 | Capabilities and Fundamental Limits of Latent Chain-of-Thought |
| 2603.21676 | Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization |
| 2604.07822 | Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers |
| 2605.23872 | Training-Free Looped Transformers |
| 2605.26733 | Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models |
| 2606.18206 | Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers |
| 2606.20075 | What Makes Effective Supervision in Latent Chain-of-Thought: An Information-Theoretic Analysis |
| 2606.24898 | Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models |
| 2606.29983 | Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping |
| 2608.03624 | LoopMTP: A looped transformer guided by latent multi-token prediction |
| 2608.11233 | Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets |
| 2608.18230 | Allocating Recurrent Compute in Looped Language Models |
| 2609.16665 | Right Direction, Wrong Step: Geometric Analysis of Finite-Step Failure in Looped Transformers |

| 2408.15495 | Remove Symmetries to Control Model Expressivity and Improve Optimization |
| 2604.21106 | How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models |
| 2605.01288 | A Theory of Saddle Escape in Deep Nonlinear Networks |
| 2607.09698 | Interpreting Latent CoT Reasoning as Dynamical Systems |

The last four are cited only through project notes (LIT_RAND_K, LIT_SEED_STABILITY, LIT_PASS_LYAPUNOV); their titles were re-fetched here.

**Not on arXiv:** the ARC Prize HRM analysis (blog, arcprize.org/blog/hrm-analysis), quoted only as cited in TRM's full text.

**Raw sources:** the fetched pages are in this folder's abs/ and full/ subdirectories, with the scripts in tools/.
