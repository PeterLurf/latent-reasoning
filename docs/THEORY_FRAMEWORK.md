# When does latent reasoning extrapolate? Four conditions, a clock proposition and three new methods

Theory stream, Latent project, 2026-09-29. A self-contained write-up of the theory in `docs/THEORY.md` §18 and §22–§25:
definitions, propositions with proof sketches, the evidence so far, and the pre-registered tests still running.
Experimental details, amendments and timestamps are in THEORY.md; the proposals for the field are in `docs/PROPOSALS.md` §8.

---

## 0. Summary

A latent reasoner computes by iterating a learned transition on a continuous state before it answers: Coconut-style
continuous thoughts, looped transformers, recursive models. A common hope is that such a model trained with K steps on
problems of depth ≤ D will solve deeper problems when given more steps. In our controlled setting (a 2-layer
Coconut-style model on graph reachability; over 100 trained models analysed) it does so only for some random seeds. The theory
explains why, and it generalises into four conditions that any latent reasoner must meet to extrapolate in the number
of steps:

| condition | statement | violated by | fixed by |
|---|---|---|---|
| **C1 identification** | training contracts pin the transition down: no slack while the circuit is chosen | fixed budgets larger than the steps needed (Coconut, CODI) | tight contracts: anytime supervision, matched or random budgets that reach the tight value (Prop. 4) |
| **C2 homogeneity** | the transition has no usable clock | step embeddings, count-indexed reads of the latent history, phase coordinates in the state | removing the *need* for a clock (C1 ⇒ Prop. 9); Markov steps alone do not |
| **C3 stability** | the map on the computation-carrying coordinates is restoring (a stable non-zero fixed point) | graded maps with gain < 1 (fading pointers) | amplitude jitter (T6, Prop. 5); discrete states |
| **C4 idempotence** | once the computation is complete, further steps leave the answer-relevant state unchanged | dangling final states, overthinking | monotone set states (A6, Prop. 6); fixed-point models |

Main results:
1. **Slack makes the continuation unidentified** (Prop. 1). This is an instance of underspecification: with a
   budget larger than needed, the training loss is indifferent to what the model does after the needed steps.
2. **Tightness unifies three training tricks** (Prop. 4): random loop counts, budget matching and anytime (per-step)
   supervision are one mechanism. They help iff (a) training places mass on budgets equal to the steps needed at the
   stage where the circuit is chosen, and (b) that budget is not a deterministic function of something the model can
   observe.
3. **Slack creates the clock** (Prop. 9). Under slack a least-effort learner moves its frontier at a fractional speed
   (demanded depth ÷ budget; measured: depth ≈ 3 after K = 5, 6, 7 or 8 steps). A transition whose state is only the
   pointer cannot realise a fractional speed, so it needs a phase variable: a clock. What the clock does past the
   training budget is unidentified, so C1 violations produce C2 violations. This is why removing the history channel
   (Markov steps) did not help: the clock moved into the state.
4. **Contraction training forces restoring maps** (Prop. 5). With amplitude jitter no graded (linear) map fits beyond
   depth ln(a₀/b₀)/(2 ln r), and every map that does fit is bistable. This is a training-only fix for C3, with no state
   supervision and no new parameters.
5. **A monotone latent set cannot fade and cannot overthink** (Prop. 6). Errors grow linearly in the number of steps,
   not geometrically. The architecture (A6) fits every least-fixed-point reasoning task.
6. **Slack explains bypass** (Prop. 8). If a problem fits in one forward pass, every latent step is unidentified, and
   bypassing the latents is the shortest zero-loss solution. Extrapolation in latent steps can therefore only be
   measured on problems deeper than one pass.
7. **Re-anchoring fixes failures without training** (§5b). Restarting the latent loop every K steps from the
   decoded frontier lifts 7-hop success from 24 to 55 of 59 models; restarting from the true frontier node gives 59/59.
   The failing models' per-episode computation is intact, and only the continuation past K breaks.

---

## 1. Setting

**Latent reasoner.** Input x (a problem), initial state h₀ = enc(x), transition h_k = F(h_{k−1}, H_{<k}, x) where
H_{<k} is the history of earlier states (Coconut: every latent position's keys and values; Markov: none), and a
readout R(H_{≤T}, q) after a budget of T steps, possibly with a query q that adds up to q̄ hops of its own.

**Task family (the toy).** Graph reachability with a delayed query. A problem of depth n needs the latent computation to
reach depth n − q̄. The state carries a **pointer**: the set of nodes at the current frontier depth π(k). The measured
fact behind this abstraction is that the trained models' thoughts are sharp pointers moving one depth per step (THEORY
§1).

**Contract.** A training distribution over (problem, budget) pairs; μ(T | n) is the budget distribution at depth n.
*Slack* σ = T − (n − q̄). *Tight*: T = n − q̄.

**Schedule.** The sequence π(1), …, π(T). *Zero loss* (in the idealised noise-free model) iff the schedule covers
depths 1..n − q̄ by step T, because the delayed readout reads every latent position.

---

## 2. C1 — identification

**Proposition 1 (slack ⇒ unidentified continuation).** If σ > 0 for every trained contract, then for any prescribed
behaviour after coverage (advance, stall, revert, restart) there is a zero-loss schedule that produces it. The output
of step T is never consumed by the loss.
*Proof.* Coverage needs n − q̄ distinct depths in T > n − q̄ steps. The σ spare steps can be placed anywhere (dwell,
restart), and every step after coverage is free. ∎

**Proposition 2 (anytime supervision ⇒ minimum speed).** Supervise the readout after every step k for all
non-reachable targets and for reachable targets within k + C hops, masking deeper ones. Then every zero-loss schedule
satisfies max_{j≤k} π(j) ≥ k + C − q̄ for all k up to coverage: unit speed from the first step. Early arrival is never
penalised. ∎

**Proposition 4 (tightness).** Let the budget T be unobservable to the latent steps (it is revealed only by when the
readout arrives).
(a) If no trained budget is tight at a stage, the delayed schedule π(k) = max(0, k − 1) has zero loss under every budget
in the support. So random budgets alone do not exclude dwell or restart schedules.
(b) If the support contains the tight budget for every trained depth at the stage, every zero-loss schedule runs at unit
speed on 1..n − q̄.
*Refinement (after Kuo et al. 2606.29983, whose length-matched K = n was their worst arm):* the premise "T
unobservable" fails when T is a deterministic function of an observable of the example. Then the loop count is itself a
clock (C2).
*Proof.* (a) The delayed schedule covers by step n − q̄ + 1 ≤ T. (b) It must succeed when T = n − q̄. ∎

**Corollary.** Random budgets, budget matching and anytime supervision are one identification mechanism. They should
help iff tight mass is present **during circuit selection**. In our curriculum that is stage 3, the first stage that
needs a latent advance (THEORY §10).

*Evidence.*
- Anytime supervision gives slack-free fronts (`1 1 2 3 4 5 6 7 8 9 10 11`) in most runs (Pilot B, PB1 3/4 vs 3/6).
- Budget matching gave the pilot's two best extrapolators (2/2).
- Overrun budgets (never tight) and randk (K ∈ 4..8, slack 2–3 at stage 3) did not help.
- A K dose-response (K5 vs K6 vs K7) did **not** appear (Pilot C, PC1 not supported). All three have slack, which Prop. 9
  treats alike.
- *Pending:* Pilot D/E (28 same-seed pairs) and T7 (anytime only inside vs only after the selection window, Pilot F).

---

## 3. C2 — homogeneity and the clock

**Proposition 9 (fractional speed needs a clock).** Suppose training demands depth d* at step K and the learner meets
the demand exactly. If d*/K is not an integer, the frontier advances on some steps and dwells on others. A transition
whose state is only the pointer maps equal inputs to equal outputs, so it cannot dwell at one step and advance at
another from the same pointer. It needs a phase variable (a clock) read from one of:
- a step embedding;
- the number of earlier thoughts (causal attention always offers this: Haviv et al. 2203.16634);
- a coordinate carried in the state, e.g. a decaying amplitude.
*Proof.* If the state is the pointer p, then F(p) is fixed. Once F(p) = p it stays p forever; if F(p) ≠ p it never
dwells at p. A schedule with speed a/b ∉ ℤ therefore needs extra state that distinguishes visits to the same pointer:
at least b phases for a periodic realisation, or a counter. ∎

**Consequences.**
1. Slack (C1 violated) forces a clock (C2 violated). The clock's behaviour past K is not constrained by training
   (Prop. 1): it may keep braking periodically (cycles, `2 3 2 3`), stop, or release (the frontier then runs at unit
   speed).
2. Removing one clock channel does not remove the need for a clock. Our Markov models built their clocks in the state
   (fade-triggered restarts `1 2 3 | 1 2 3 4`, or a clean period-2 phase bit `1 1 2 2 3 3 | 4 4 5 5`).
3. Enforcing tightness makes the clock unnecessary.

*Evidence (in-sample).*
- p[K−1], the depth of the last fed-back thought, is 3 at K = 6 and at K = 8 (median; 29/42 and 10/17 models), and 3,
  3, 2 at K = 5, 6, 7 in Pilot C.
- Stage by stage, the in-range reach tracks the curriculum's demand (median 2 / 3 / 3.5 at the end of stages 3 / 4 / full).
- Dwell steps are where the history is copied (retrieval share 0.22 in extrapolators and 0.43 in failures, against
  0.04 / 0.14 on advance steps).
- *Pending, pre-registered:* LE1 (arm medians of p[K−1] span ≤ 1 across K), LE2 (anytime: p[K−1] ≥ K − 2), LE3 and LE-R
  (depth caps lower p[K−1]; out of sample on the experiments session's runs), LE4 (anytime models more restoring).

---

## 4. C3 — stability

**Proposition 3 (reach of a homogeneous advance).** Let the pointer amplitude follow a_{d+1} = φ(a_d) + noise, with
matching threshold θ.
(i) If φ has a stable fixed point a* > θ whose basin contains the trained amplitudes (a *restoring* map), reach is
unbounded under noise η < (a* − θ)(1 − L), where L is φ's Lipschitz constant on the basin.
(ii) If φ(a) ≤ g·a with g < 1, reach ≤ d_train + ln(a_{d_train}/θ) / ln(1/g).

**Proposition 5 (jitter excludes graded maps).** Train with *amplitude jitter*: every fed-back thought is rescaled by
s ∈ [1/r, r], and answers must stay correct at every trained depth ≤ D.
(a) A linear map φ(a) = g·a satisfies all-down jitter on the signal, (g/r)^n a₀ > θ, and all-up jitter on a decoy's
leakage b₀, (g·r)^n b₀ < θ, only while r^{2n} < a₀/b₀. So no linear map fits beyond n_lin = ln(a₀/b₀)/(2 ln r).
(b) Any map that fits for D > n_lin sends the jitter interval [a*/r, r·a*] into a subinterval of itself. It therefore
has a fixed point there with average slope < 1 on the interval (intermediate value theorem). Since φ(0) ≈ 0 and
φ(a*/r) > a*/r, it must also expand somewhere below a*/r: it is bistable, with a threshold and a stable level.
(c) With such a basin, (i) of Prop. 3 applies: reach is unbounded. ∎

**Method T6 (amplitude jitter).** During training only, set h ← c + s(h − c) for every thought fed back, with c the batch
mean and s ~ LogUniform[1/r, r] applied with probability p. It is the transfer-curve perturbation we use to *diagnose*
restoring maps, turned into augmentation. It costs one elementwise operation per step, adds no parameters, and ports
unchanged to CODI/Coconut on LLMs. Prior art on denoising or noisy recurrence (Mozer et al. 1805.08394; Lim et al.
2102.04877) motivates stability in general; the new parts are multiplicative amplitude jitter of a fed-back thought with
no auxiliary target, and the statement that it excludes graded maps.

*Evidence.*
- Transfer curves of trained models are restoring (clamped) in extrapolators and linear with gain ≈ 0.7 in fading
  failures (THEORY §14).
- A training-free amplitude boost at inference (α ≈ 1.3, with the history block past K) raises 7-hop success from
  24/59 to 46/59 (THEORY §12).
- *Pending:* Pilot F (J vs C, JAT vs AT; PF1 restoring index, PF2/PF3 reach).

---

## 5. C4 — idempotence

**Proposition 6 (monotone latent set).** Carry a membership vector m ∈ [0, 1]^V with
m_{k+1} = max(m_k, σ(β(u(m_k, x) − τ))), where u is monotone in m.
(a) m_k never decreases: no member fades.
(b) If u is exact on sets, m_k is the k-step closure, and m is invariant after convergence: extra steps are no-ops (no
overthinking, no dangling state).
(c) With per-step false-positive probability ≤ ε per entity, P(any false positive by step k) ≤ k·V·ε. This is linear in
k, against the geometric decay of a graded pointer or a constant hazard (1 − h)^k.
*Proof.* (a) and (b) follow from max and monotonicity (Knaster–Tarski on a finite lattice with inflationary updates);
(c) from the union bound. ∎

**Architecture A6 (latent set memory).** Next to the continuous thought:
- per-entity sigmoid scores (non-competitive, so members do not dilute each other);
- a max update;
- the set fed back into the next thought through a zero-initialised gate.
It suits every inflationary reasoning task: reachability, ProsQA-style entailment, Horn/Datalog inference, dataflow.
Non-monotone reasoning needs stratified sets. Relatives: max aggregation in neural execution of graph algorithms
(1910.10593), set-aggregation recurrent GNNs (2609.15932). It is new as a latent-CoT state.

**Prop. 6 + Prop. 9.** With a monotone set, "full speed, then idle at the fixed point" needs no clock, so a set removes
slack's pressure to build a brake. *Pending:* Pilot G (PG1–PG5) and two feasibility runs.

---

## 5b. Re-anchoring: C2 and C3 at inference (after Continuous Thought Machines)

CTMs reach longer maze paths by re-applying a bounded-horizon policy from the current state. The latent analogue is
**T8, re-anchored episodes**:
1. Run the trained K-step loop.
2. Decode the frontier from the model's own attention (the target of the edge the sweep head reads at step K).
3. Start a fresh K-step episode from that node.
4. Max-combine the answers across episodes.

The step count never leaves the trained range (C2), and every restart begins from a crisp node embedding (C3).

*Result* (pre-registered, THEORY §26; 59 models, no training):
- 7-hop success goes from 24/59 to **55/59**, against 41/59 for the history block.
- 14 of 16 block-resistant (fading) failures are rescued.
- Controls: random-node anchors give 0/59. Random anchors at the frontier's depth match the model's decode (56/59).
  Restarting at the true frontier node gives **59/59 and 0.885** balanced accuracy at 8–10 hops; failing models alone
  reach 0.872.

Failing models are therefore fully competent for K steps from any start. Only the continuation past K was broken, which
is the clearest evidence for the clock account (Props. 1, 9).

*Implication.* A latent reasoner need not extrapolate in the number of steps if it is re-anchored every K steps. The
problem then reduces to decoding the frontier; at LLM scale that means reading the reached entity or state out of the
thought, e.g. with Patchscopes.

*Training-time counterparts under test:*
- **A8, chunk-reset latent attention** (Pilot J): a CTM-style sliding window that restarts every L steps, so every
  test-time chunk is a copy of a trained chunk.
- **T9, a periodic step clock** (Pilot I): a clock that cannot run out.

## 6. Scope — pretrained models (Prop. 8)

Let c be the number of hops one forward pass resolves on the task (the one-pass capacity). For an n-hop problem with K
latent steps, the slack is K − ⌈max(0, n − c)/c_s⌉, where c_s is the hops per latent step. If n ≤ c, every latent is
unidentified and bypass is the shortest zero-loss solution. This is a single account of the reports that latent tokens
are bypassed (Aswal et al.; Cui et al. 2602.22441; Rizvi-Martel et al.; our ProsQA addenda). GPT-2 Coconut on ProsQA
works in one pass: its accuracy is flat in the number of thoughts (0.976 at K′ = 0; 0.974 at K′ = 2..12).

At GPT-2 scale the C1 violation is visible directly. Public CODI (6 trained latents, GSM8K) scores **below no latents**
when its latent chain is truncated: 3-step problems 0.15 with 2 latents vs 0.24 with none. Its intermediate states are
not answer-ready, and 12 latents do not extend its reasoning-depth horizon (4-step problems 0.31 → 0.24). An anytime-
trained reasoner should be non-decreasing in the number of latents. PROTOCOL8, run by the experiments session, tests
this above one-pass capacity with three arms: Coconut, anytime, and random tight budgets (Prop. 4's equivalence claim).

---

## 7. Status of the tests (2026-09-29)

| test | claim | status |
|---|---|---|
| exp5 block (59 models) | Mechanism 1 is causal | **supported**: 17/23 retrieval failures rescued, zero effect in-distribution |
| Pilot A | Markov steps help (A1) | **not supported**; the clock moves into the state |
| Pilot B | anytime supervision helps (Prop. 2) | PB1 supported (n = 4) |
| Pilot C | extrapolation falls with K | **not supported** (PC1); retrieval share does rise with K |
| §21 | stage-3 retrieval marker predicts failure | **failed out of sample** (AUC 0.51) |
| §14 | two-factor rule | **failed out of sample** (8/15) |
| Pilot D/E | anytime vs control, 28 pairs: depth wall or schedule end | running (~01:00–05:00 EDT 2026-09-30) |
| §26 T8 re-anchoring | restarting at the frontier fixes failures (C2, C3) | **supported**: 24 → 55/59; oracle frontier 59/59 |
| Pilot H | fine-tuning a failure with anytime (± jitter) repairs it | queued |
| Pilot I | periodic clock keeps the front moving past K | queued |
| Pilot J | chunk-reset attention (A8) | queued |
| LE1–LE4, LE-R | Prop. 9 | registered; C data in, D/E/R pending |
| Pilot F | T6 jitter, T6 + anytime (T7 window arms withdrawn for budget) | queued |
| Pilot G | A6 set memory | queued; feasibility runs in progress |
| PROTOCOL8 | Prop. 8 and C1 at GPT-2 scale | Phase 0 (one-pass capacity c) running, experiments session |

Negative results are part of the theory's history: the Markov architecture, the K dose-response, the retrieval marker
and the two-factor rule all failed. Each failure narrowed the account from "the retrieval channel" to "slack, and the
clock it creates".
