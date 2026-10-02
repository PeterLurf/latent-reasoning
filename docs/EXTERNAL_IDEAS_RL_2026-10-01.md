# External RL / acquisition proposals, 2026-10-01 (reply to EXTERNAL_PROMPT_RL_2026-10-01.md)

Saved verbatim below the triage. The agent read repo snapshot 986b3fd. These are unvalidated proposals; each needs
Literature's check and a THEORY registration before anything runs.

## Audit triage

* **Bug found and verified.** `c1/train.py:32` hardcodes `D1_TOKENS = 120`, which is correct only for M = 6. The GPT-2
  cell evaluates 20·M positions per update. So §67b's compute-matched full-prompt arm (fp_cmp, M = 24) was given about
  4× too little compute, and C1-5 needs a corrected re-run. Sent to Experiments.
* **Method labels in §69.**
  - `rmer` weights successful samples by occurrence, not uniformly over distinct programs, so it is not Guu's RandoMer.
  - `grpo`'s self-inclusive group baseline scales the gradient by 1 − 1/K.
  - Results should be labelled accordingly. The registered definitions stand.
* **Add an exact expected-reward control (max mean J).** It separates "objective weighting" from "sampling" in any RL
  win. Cheap.
* **Proposal A (operator-conditioned exact likelihood).**
  - It tests whether parametrisation, not exploration, is the obstacle, by comparing an additive cell, a multiplicative
    cell, and B_a-conditioned logits, matched.
  - It is the most diagnostic and the cheapest. Run first.
* **Proposal B (tempered-posterior continuation into hard EM / Viterbi).** Classical annealing → hard EM, exact over
  finite codes. Second.
* **Proposal C (stationary-rule sampling: K whole candidate automata, held fixed across examples).** It changes the
  objective to consistent rules. It is the highest variance and should be run only if A and B fail.
* **The scaling warning matters.** The GPT-2 cell's enumeration cost grows as (3|G| + 2)·M: 7k positions per update for
  S4 and 44k for A5 at ratio 4. So a GPT-2 port beyond S3 needs its own compute plan, and the spectral baseline (69-2)
  already learns S4/A5.
* **Seeds proposed:** development 1300–1302 and confirmation 1400–1409. Check they're unused.

---

# Acquire latent rules by changing optimization

**External research response — 2026-10-01.** Reviewed repository snapshot: `986b3fd1778944e4195dd39d288feeef267669aa`. This document proposes experiments; it does not report new training. It is intended for Literature's check and a THEORY registration before execution, as requested in the [external prompt](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/docs/EXTERNAL_PROMPT_RL_2026-10-01.md).

**Test three changes: better state–operation interaction under exact marginal likelihood, genuine posterior continuation into hard EM, and exploration over whole stationary rule tables.** I would run the first two before committing GPT-2 compute to the third. The evidence establishes unreliable acquisition and state aliasing; it does not isolate inadequate exploration or establish that RL is necessary. Exact marginalization removes latent-path sampling variance, but cannot repair an ill-conditioned parameterization, symmetric representation, weak early credit, or disagreement between stochastic training and greedy execution. RL cannot manufacture gradient information absent from the objective. Its useful contribution here would be changing the objective, search distribution, or optimization trajectory. None of the procedures below has demonstrated the requested reliability or S4/A5/S5 scaling; the numerical schedules are concrete hypotheses to register.

## Discrete validity has exposed an acquisition failure

The committed results support the main diagnosis, with several qualifications. Independently aggregating the saved JSON gives **GPT-2 M12: 3/10 certificates, precisely seeds 1, 2 and 8**; free-table M12: 10/10; bilinear M12 at temperature one: 7/10; bilinear M24 at temperature one: 8/10; and M24 with temperature 4→1: 10/10 selection seeds but 8/10 fresh seeds, failing 103 and 108. The overcomplete MLP grid certifies only 1/40. S4 M48 and A5 M120 each certify 0/3. These are different panels, not interchangeable estimates. ([GPT-2 pilot](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/PILOT.md), [table](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/results/c1_diag_tab12.json), [bilinear M12](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/results/c1_diag_bil.json), [bilinear M24](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/results/c1_diag_bil24.json), [held-out panel](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/results/c1_diag_bil24_heldout.json), [MLP grid](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/results/c1_diag_mlp.json), [S4](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/results/c1_group_S4.json), [A5](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/results/c1_group_A5.json))

There is a concrete acquisition witness. Failed GPT-2 seed 3 takes initial code 0 with operation 1 to code 10, decoded as answer 0; the correct one-operation answer is 1. Its certificate reports six reachable codes but twelve reachable `(code, true-state)` pairs, with every reachable code ambiguous. This is hard-state aliasing, already visible at length one. It does not establish a particular subgroup or coset interpretation. Nor does an approximately 0.95 score establish such structure without the transition graph and answer confusion. ([Seed 3 evaluation](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/runs/a672_d1_M12_s3/eval.json))

I retract the earlier recommendation to prefer minimal code capacity: **overcomplete codes are the stronger starting point empirically**. Multiple codes per semantic state are compatible with exact computation. The reachable product certificate permits them and ignores irrelevant unreachable codes; forcing a bijection would reject some correct implementations. The certificate proves correctness at every positive length for the fixed deterministic machine, conditional on stable argmaxes. ([Certificate](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/core.py#L269-L289))

Two evidential limits matter. The toy `loss` field is the last minibatch at one curriculum length, not aggregate evaluation NLL; it cannot alone diagnose a soft-to-hard gap. Older diagnostic files used an incorrect start-code certificate and must not be pooled. The snapshot contains the improved GPT-2 pilot but no scored §67b panel or §69 RL result panel, so I cannot independently verify the prompt's first-three-seed claim or rank the existing RL methods. ([Loss recording](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/c1_diag.py#L54-L98), [certificate correction](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/docs/THEORY.md#L4863-L4868), [frozen tree](https://api.github.com/repos/PeterLurf/latent-reasoning/git/trees/986b3fd1778944e4195dd39d288feeef267669aa?recursive=1))

Why can exact gradients still fail? Let the suffix value be $b_t(c)=P_\theta(y\mid c,a_{t+1:N})$. If later transition matrices mix strongly, these values become almost equal across codes, leaving little advantage for an early transition. Exact summation computes that weak signal exactly. Symmetric initialization, low occupancy, saturated logits and shared-parameter interference provide additional explanations. These are hypotheses to measure, not proven causes of particular failed runs.

## Separate the objective from the sampling estimator

For a complete word $x=a_{1:N}$, final answer $y$, initial code $z_0=0$, define

\[
w_\theta(z)=D_\theta(z_N,y)\prod_{t=1}^N T_\theta(a_t,z_{t-1},z_t),\qquad J_\theta(x,y)=\sum_z w_\theta(z).
\]

Marginal likelihood optimizes log $J$; terminal binary reward optimizes $J$. For each example, $\nabla J=J\nabla\log J$; across examples the weighting differs. **Include exact expected-reward optimization, max mean $J$, as a control.** Comparing it with exact mean log-likelihood isolates objective weighting; comparing it with ordinary trajectory REINFORCE isolates sampling effects. Suffix values and forward occupancies give its exact gradient without a learned critic or intermediate labels.

The repository already implements `marg`, `hem`, `grpo`, `rmer`, best-of-four and a marginal/`hem` hybrid. Preserve their registered definitions. Existing `hem` samples trajectories and reinforces successes; it does not perform Viterbi inference. On-policy successful paths follow the model's answer-conditioned posterior, so their expected score update largely recovers likelihood learning, with additional finite-batch weighting. Existing `rmer` equalizes successful sample occurrences but keeps duplicates: frequent paths still dominate. It is not uniform weighting over distinct discovered programs. Existing `grpo` is a REINFORCE variant without PPO clipping; its self-inclusive group-mean baseline scales the expected reward gradient by $1-1/K$. Use leave-one-out baselines in new arms. ([Implementation](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/c1_rl.py#L107-L139), [existing registration](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/docs/THEORY.md#L5053-L5133))

A genuine meritocratic-search extension would deduplicate successful paths, search beyond ordinary on-policy frequencies, and specify weights over that discovered set. Guu et al. provide the relevant precedent. Memory plus weighted imitation is not automatically MAPO: MAPO accounts for buffer probability mass and its complement. I would defer this fourth arm because exact posterior inference is cheap here, and successful but semantically spurious paths are abundant. ([Guu et al., 1704.07926](https://arxiv.org/abs/1704.07926), [MAPO, 1807.02322](https://arxiv.org/abs/1807.02322))

## Three procedures make distinct, falsifiable changes

### A. Condition transition operators while retaining exact marginalization

**Algorithm.** Preserve final-answer NLL, curriculum, optimizer, code count and temperature. Compare parameter-matched additive and existing multiplicative cells, matching initial logit standard deviation as well as parameter counts. Add an explicitly operation-conditioned logit model:

\[
L(c,a,j)=e_{\mathrm{out}}(j)^\top B_a e_{\mathrm{in}}(c)+b_j.
\]

All embeddings and matrices are learned from terminal answers. No multiplication table, inverse relation or group-equivariant action is supplied. A rank-$r<M$ model restricts each operation's logit matrix; include $r=M$ with full-rank embeddings as a capacity reference capable of representing arbitrary tables. Otherwise failure could simply be a rank bottleneck. The full-rank reference need not match parameter count; label it separately.

Use exact gradients throughout; there is no policy baseline or sampled exploration. Transition temperature starts at 4 and falls linearly to 1 by update 4,000, then remains 1. Use distinct initial codes, normalized features and matched logit scale. These choices test smooth optimization and symmetry breaking rather than reward exploration. The free-table M12 result makes this a serious alternative to the exploration-only diagnosis, although it does not guarantee a scalable factorization.

**Cheapest decisive test.** S3, M24, 8,000 updates, batch 512, Adam learning rate 0.01, existing quarter-stage length curriculum. Screen additive, multiplicative and operator-conditioned variants on three fresh development seeds; freeze one candidate before ten fresh confirmation seeds. The engineering gate is ≥9/10 final-checkpoint certificates. A claim that interaction improved acquisition additionally requires at least three more certificates than the matched additive control and paired outcome reporting; 9/10 versus the historical 8/10 is not convincing evidence of superiority.

For GPT-2, first compare additive and multiplicative heads using exactly the same normalized hidden features. Its existing operation-position hidden state already sees the code, so calling it an independent operation gate would be incorrect. An operation-indexed $B_a$ acting on a GPT-2 code representation is a separate architecture arm. Compare trainable GPT-2 with frozen pretrained features and frozen random features if claiming pretraining is responsible; a sufficiently large head could learn the table itself. ([Current cell](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/core.py#L40-L116))

**Prior and difference.** Multiplicative RNNs explicitly let inputs select recurrent operators, while also identifying conditioning difficulties. This proposal applies that principle to terminal-only finite-state induction and exact certification, rather than claiming multiplicative interaction is new. ([Sutskever, Martens and Hinton, 2011](https://icml.cc/2011/papers/524_icmlpaper.pdf))

**Likely failure.** Richer interactions fit partial machines more efficiently, or success requires nearly independent table parameters and scales poorly. The matched experiment can refute the exploration-only account even if it does not solve GPT-2 acquisition.

### B. Temper the answer posterior, then perform actual hard EM

**Algorithm.** At an E-step snapshot, compute

\[
q_\eta(z\mid x,y)\propto w_{\theta_{\rm old}}(z)^\eta.
\]

Use log-space forward–backward with edge potentials $\eta\log T$ and terminal potential $\eta\log D$. **Do not row-normalize the powered potentials:** that would change the path posterior. Detach exact edge responsibilities $\xi_t(i,j)$ and final responsibilities $\gamma_N(j)$. Minimize

\[
C(\theta)=-\sum_{tij}\xi_t(i,j)\log T_\theta(a_t,i,j)-\sum_j\gamma_N(j)\log D_\theta(j,y).
\]

For a neural cell, hold the detached posterior counts fixed across two Adam updates per E-step, recomputing T, D and J at each update; count both against the update budget. The first update at $\eta=1$ has exactly the marginal-likelihood gradient; fitting the held posterior for a second update changes the optimization path. Use a paired MML control that reuses each batch twice but recomputes its gradient. No policy baseline, rejection sampler or external step label is involved.

**Fixed first schedule.** For $U=8,000$ optimizer updates: the first $U/8$ use MML; over the next $3U/8$, increase $\eta$ linearly from 0.5 to 2; the next $U/4$ use $\eta=4$; the final $U/4$ use max-product/Viterbi paths with backpointers. Outside the initial phase optimize $0.25[-\log J]+0.75C$. Preserve the matched arm's transition-temperature schedule: policy temperature and posterior inverse temperature are distinct. Deterministic tie-breaking must be fixed in advance. A $\eta=1$-throughout generalized-EM control separates posterior continuation from the second inner update.

The proposed mechanism is early distribution over alternative explanations, followed by commitment to coherent paths. This is a substantive change from reinforcing whichever successful paths ordinary sampling happens to find. It remains possible for different examples' Viterbi paths to use contradictory assignments for the same code–operation pair.

**Cheapest decisive test.** S3 MLP M12, 8,000 updates, batch 512, learning rate 0.01; three fresh development seeds, then ten fresh seeds after freezing the schedule. Require ≥9/10 certificates; run the exact-MML, exact-reward and $\eta=1$ controls on the same seeds and example streams. Replicate the frozen procedure on bilinear M24 before scaling. This targets the poorly acquiring MLP rather than relying only on a near-ceiling bilinear cell.

**Prior and difference.** Deterministic annealing and Viterbi training are established latent-variable methods. Their tagging and grammar results do not establish this automaton result. LaTRO's variational rationale learning is relevant conceptually, but uses sampled textual rationales; here posterior inference is exact over finite hidden codes and the outcome is a complete reachable-machine certificate. ([Smith and Eisner, 2004](https://aclanthology.org/P04-1062/), [Spitkovsky et al., 2010](https://research.google/pubs/viterbi-training-improves-unsupervised-dependency-parsing/), [LaTRO, 2411.04282](https://arxiv.org/abs/2411.04282))

**Likely failure.** Flattening preserves symmetry; hardening freezes a wrong partition; or the likelihood anchor prevents leaving it. Neural minibatch generalized EM and the tempered phases have no unconditional ordinary-likelihood monotonicity guarantee.

### C. Explore stationary rules instead of independently resampled actions

**Algorithm.** For each minibatch, independently sample $K=16$ candidate automata. In candidate $k$, draw $H_k[c,a]\sim T(c,a,\cdot)$ once per transition row and $O_k[c]\sim D(c,\cdot)$ once per decoder row. Hold these assignments fixed across repeated visits and all $B=256$ examples. Lazy first-access sampling is equivalent. Execute each candidate and obtain terminal reward $r_{ki}=\mathbf1[O_k(z_N)=y_i]$.

For example $i$, use the independent leave-one-candidate-out baseline $b_{ki}=\sum_{\ell\ne k}r_{\ell i}/(K-1)$. Let $V_{ki}$ contain the distinct transition rows consulted by that example. The score-function estimate is

\[
\widehat g=\frac1{KB}\sum_{ki}(r_{ki}-b_{ki})\nabla_\theta\left[\sum_{(c,a)\in V_{ki}}\log T_\theta(c,a,H_k[c,a])+\log D_\theta(z_N,O_k[z_N])\right].
\]

Each consulted row appears **once**, even when visited repeatedly; it was sampled once. Do not divide by the random number of visited rows. Unused independent rows can be integrated out. This is an unbiased terminal-reward gradient for the distribution over stationary hard automata; sharing each candidate across examples changes variance, not this expectation.

**Schedule.** First $U/4$ updates use exact MML. The remaining $3U/4$ use the estimator above, with the matched 4→1 policy-temperature schedule. Add an exactly differentiated regularizer equal to $\beta$ times the average of mean transition-row entropy and mean decoder-row entropy; $\beta=0.02$ at the switch, falling linearly to zero by $3U/4$. Use zero thereafter. No standard-deviation normalization, clipping, replay or certificate-triggered resets. Differentiate the actual temperature-adjusted sampling distribution.

This changes the objective: ordinary Markov marginalization redraws transitions on each occurrence, whereas a candidate rule must answer repeated queries consistently. Exact marginalization supplies warm-up and a diagnostic comparator, not a claimed exact evaluator of the stationary-rule objective. At deployment use the model's ordinary greedy GPT-2 calls, never substitute a searched table or select candidates using the certificate.

**Cheapest decisive test.** S3 MLP M12, 8,000 updates, three development seeds and ten held-out seeds; require ≥9/10 certificates. Compare exact MML, exact Markov expected reward, and ordinary trajectory REINFORCE with the same $K$, baseline, entropy and warm-up. Include free-table M12 as a capacity reference. Measure total cost: sixteen automata require more rollout work, although GPT-2's table construction is shared across candidates.

**Prior and difference.** Temporally consistent exploration motivates parameter-space noise; neural program induction combines terminal supervision with policy gradients and imitation. Here the explored object is a finite categorical stationary rule system, without an interpreter implementing the target composition. State-regularized RNNs supply a representation precedent. None proves this estimator will improve acquisition. ([Parameter-space noise, 1706.01905](https://arxiv.org/abs/1706.01905), [Neural Symbolic Machines, 1611.00020](https://arxiv.org/abs/1611.00020), [State-Regularized RNNs, 1901.08817](https://arxiv.org/abs/1901.08817))

**Likely failure.** Candidate machines all receive similar chance reward, coordinated rule changes become rare, or greedy rowwise modes fail despite successful sampled candidates. This arm has the clearest change in exploration, but also the highest variance risk.

## Certification and compute must survive the scaling test

Use development seeds 1300–1302 and confirmation seeds 1400–1409 only after checking that they are unused. Preserve §69 as registered; these are new named arms and amendments, not revised definitions of its methods. Select schedules on development data only, freeze configuration and code hashes, and evaluate final checkpoints on all ten confirmation seeds. **≥9/10 is an engineering gate, not proof of ≥90% population reliability, or proof of improvement over 8/10.** Report binomial intervals and paired outcomes. Best-of-four remains a four-attempt pipeline with all four costs reported.

The following fixed progression keeps the input alphabet equal to all group elements and preserves the successful S3 code/group ratio. It must be reported separately from existing ratio-two scaling results.

| Stage | Cell and codes | Training budget | Promotion rule |
|---|---|---|---|
| Acquisition stress test | S3 MLP M12; matched controls | 8,000 updates | Frozen procedure ≥9/10 certificates |
| Interaction/replication | S3 bilinear/operator cell M24 | 8,000 updates | ≥9/10 |
| First scaling | S4, M96 | 16,000 updates | ≥2/3 feasibility seeds, then ≥9/10 fresh seeds |
| Harder group | A5, M240 | 16,000 updates | Same independent gate |
| Final scaling | S5, M480 | 16,000 updates | Only after A5 passes; same gate |
| GPT-2 transfer | First S3 with matched M12 or M24 cell | At most 8,000 updates and three GPU-hours/run | Three feasibility seeds, then ten; ≥9/10 |

Use the existing quarter-stage curriculum for S3 and one frozen operator-first curriculum for all scaling arms, including controls. Register any 32,000-update escalation separately; do not extend individual failures after inspection. For GPT-2 retain the relevant §67b initialization, normalization and learning rate $10^{-4}$, changing only the registered algorithm or head factor. Calibrate timing on an excluded seed, fix the update cap before confirmation, and count cap hits as failure to acquire within budget. ([Existing settings](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/PILOT_67b.md), [existing RL driver](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/theory/c1_rl_driver.py))

One comparison needs correction before new claims: `D1_TOKENS=120` is hardcoded, but S3 evaluates $6M$ three-position transition inputs and $M$ two-position decoder inputs, totaling **20M positions/update**. M12 uses 240 and M24 uses 480. The present counter and `fp_cmp` batch calculation therefore undercount by factors two and four respectively. Amend the accounting and rerun affected controls; do not silently rewrite old registrations. Positions are still only a compute proxy, so report measured accelerator time, table-row evaluations, dynamic-programming work and peak memory too. ([Constant](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/train.py#L30-L32), [uses](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/train.py#L178-L191), [table construction](https://github.com/PeterLurf/latent-reasoning/blob/986b3fd1778944e4195dd39d288feeef267669aa/c1/core.py#L103-L109))

With an all-group alphabet the GPT-2 count becomes $(3|G|+2)M$. At ratio four this is 480 positions/update for S3, 7,104 for S4, 43,680 for A5 and 173,760 for S5, before marginalization costs. **Few-hour GPT-2 convergence beyond S3 is unverified and increasingly implausible under naive full enumeration.** Toy success should determine whether a larger port merits a separate compute plan.

Before training, validate the estimators on an exhaustively enumerable two-code model: posterior marginals; the $\eta=1$ gradient identity; exact reward versus REINFORCE expectations; and stationary-rule scoring on a sequence revisiting a row. During evaluation record length-stratified NLL, soft terminal probability, sampled and greedy accuracy, row entropy, output-answer coverage, state collisions, confusion and minimum argmax margins on the same fixed panel. Keep true-state mappings and certificate witnesses out of training, curriculum adaptation and confirmation checkpoint selection.

Retain the original zero-latent and retrained no-feedback controls, donor-state counterfactuals, same-semantic substitutions where nonvacuous, and corrected example/compute-matched full-prompt baselines. Verify real GPT-2 hard calls agree with the audited table across relevant batching and precision settings. Streaming makes the null controls fail by construction; this supplied interface remains a limitation, not evidence that unrestricted GPT-2 requires multiple passes.

The useful next result is a **frozen final-answer-only procedure repeatedly acquiring a certifiably correct stationary machine**, with the source of improvement isolated. If conditioned exact likelihood wins, revise the exploration diagnosis. If stationary-rule sampling wins its objective and architecture controls, the evidence supports searching coherent rules. If all fail, the failure localizes acquisition more sharply than another broad “RL versus marginalization” comparison.

### Implementation pseudocode for the registered additions

`make_tables` evaluates the learned cell and decoder; it never consults the generator's composition table. The terminal-label generator and certificate remain separate. All log probabilities below use the actual scheduled policy temperature. For A, the only change inside `make_tables` is the registered head; its loss is ordinary exact forward NLL.

```text
A: interaction-conditioned exact likelihood
for update in 1..U:
    words, final_answers = draw_complete_problems(curriculum(update))
    logT, logD = make_tables(theta, head, policy_temperature(update))
    loss = mean(-log_forward_terminal_probability(words, final_answers, logT, logD))
    optimizer_step(loss)

B: exact posterior continuation / generalized hard EM
for outer_step in 1..U/2:
    batch = draw_complete_problems(curriculum(2 * outer_step))
    if in_initial_MML_phase:
        repeat twice: recompute tables and optimizer_step(exact_MML_NLL(batch))
        continue
    old_logT, old_logD = detached(make_tables(theta, scheduled_temperature))
    if in_Viterbi_phase:
        counts = max_product_path_counts(batch, old_logT, old_logD)
    else:
        counts = forward_backward_counts(batch, eta * old_logT, eta * old_logD)
    repeat twice:
        logT, logD = make_tables(theta, scheduled_temperature)
        complete_NLL = cross_entropy_from_fixed_counts(counts, logT, logD)
        optimizer_step(0.25 * exact_MML_NLL(batch) + 0.75 * complete_NLL)

C: stationary-rule policy gradient
for update in 1..U:
    batch = draw_complete_problems(curriculum(update))
    logT, logD = make_tables(theta, scheduled_temperature)
    if update <= U/4: optimizer_step(exact_MML_NLL(batch)); continue
    independently sample K rule tables and decoder tables from exp(logT), exp(logD)
    execute each table on every example; record rewards and DISTINCT queried rows
    for each candidate k and example i:
        baseline = mean(reward[l, i] for l != k)
        score_logprob = sum(logT[row, chosen_target] for each distinct queried row)
        score_logprob += logD[terminal_code, chosen_answer]
        loss[k, i] = -detach(reward[k, i] - baseline) * score_logprob
    optimizer_step(mean(loss) - beta * exact_uniform_row_entropy)
```

For B, `counts` includes terminal decoder responsibilities as well as edges; powered potentials stay unnormalized within rows. For C, candidate assignments persist throughout the whole batch, while candidates are independent. No visited-row-count normalization is applied. Calibrate each group's compute separately, record the GPU model and numerical precision, and fix the confirmation cap before viewing outcomes.
