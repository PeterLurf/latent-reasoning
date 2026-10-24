# FIXED-HOLD draft v0 (FH_draft.md, 1314f24c): novelty lines S1–S5 (2026-10-21)

Written 2026-10-21 by the Literature session, for Theory. Kept light under the usage limit.
- **Checked myself:** our own notes (LIT_IDLE_HORIZON_PRESCOUT, 8bb89d7a) and Viakhirev 2608.18222's HTML (its objective, Appendix J and the Appendix K controls).
- **Searches:** one web search for correctness-gated fixed-point losses.
- **IDs:** 2609.16665 verified on its abs page. The others were verified earlier today.

| # | sentence | verdict | basis |
|---|---|---|---|
| S1 | an idle-pass fixed-point penalty gated on the decoded answer being correct | **OK, narrow wording** | Fixed-point and contraction penalties exist: Viakhirev 2608.18222 (terminal step), Bai 2106.14342, STARS 2605.26733, Movahedi 2606.18206 (no penalty; residual scaling). **Viakhirev's "gate" is the learned update gate a_T scaling the penalty** (Appendix J: "the gate could in principle satisfy the penalty by closing"), not answer correctness. A correctness gate was not found in these or in one search. Wording: "a relative, answer-correctness-gated variant of terminal fixed-point penalties (2608.18222)". |
| S2 | the penalty only after the computation completes, with motion on computing passes (phase split) | **NARROW** | Viakhirev already applies the penalty only at the terminal step T, after a fixed compute budget. New is the phase boundary set **per item from the known n** (idle set t ≥ n + 3), plus the far no-grad rollout. If the draft really adds a two-sided "motion required" target on computing passes, that part is not found. Our Goodhart memory applies to it. |
| S3 | in a version-bank loop the held answer outlives the current state's own decode | **OK** (an observation on our architecture) | Not found for memory-augmented looped models. State the scope: VB models, K′ 24–48. |
| S4 | past the horizon, held answers fall linearly in margin (≈ 0.27 nats / pass, coherent drift) and absorb into 1–2 answer-independent classes | **NARROW** | **Decay past the trained depth is reported:** Shapiro 2608.11233 (97.7 → 10.9 %); Liang / Jin / Pan 2609.33144 (errors persist and compound); Viakhirev's drifting regime (λ_max > 0, depth damage 0.87–0.93 without the objective). **The margin mechanism is the converse of Viakhirev's sufficient condition** (the answer is frozen while per-step displacement is small relative to the decoder margin). **Guo, Wu & Du 2609.16665:** extra iterations reduce support for the reference answer when "a locally improving direction produces a harmful full update"; a fixed quarter step recovers 72–83 % of selected failures. That makes **damping** a rival lever worth naming. New: the measured linear rate and the absorbing answer-independent classes in our models. |
| S5 | a matched random-depth vs state-penalty contrast with identical rollout schedules | **OK as a design** | **Viakhirev's matched control is not random depth.** It is an **answer-only** penalty, ‖readout(z_T) − readout(z_{T−1})‖ with the latent free: Sudoku extrapolation 0.00, reachability overthinking damage 0.99–1.00, vs 0.00 for the latent-settling recipe. Random depth exists as a separate lever (Anil 2211.09961, Huginn, STARS, Kuo 2606.29983), but not as a rollout-matched rival to a state penalty. |

## Other notes for the draft
- **Weight transfer:** our r is relative, Viakhirev's residual is absolute, so 0.05 does not transfer (the draft's Q1). Viakhirev's dose curve does show that settling starts at 1/10 of the headline weight (0.005).
- **Add Viakhirev's answer-only arm** (gate-free, readout penalty) as a cheap descriptive control. It separates "state settles" from "answer frozen", which is exactly S5's question.
- **Correctness signals in loops:** TRM / HRM halting heads are trained on whether the answer is correct. That is a halting target, not a fixed-point penalty, so it is only a cousin of S1.

## Addendum (2026-10-23): HOLD-DEPTH draft (HD_draft.md, ee9537b9), N1–N3
**Bansal 2202.05826, read in the PDF.**
- Progressive loss: run a random number of iterations, then restart the recurrence from those features while discarding the earlier gradients. Train to produce the solution after a random number of further iterations.
- Explicitly "to continue improving the quality of partial solutions, **even when they contain errors**".
- Mixed with L_max_iters at α = 1 (prefix sums) or 0.01 (chess). **There is no correctness gate.**
- Bansal reports extrapolation **gains** (e.g. chess DT-Recall with progressive loss 97 % vs 83 % without).
- **G-FAR is therefore not a known Bansal variant.** It is the opposite design choice: it withholds the repair signal from wrong prefixes, where Bansal trains on them.

| # | verdict | basis |
|---|---|---|
| N1 (progressive / random-depth training fixed hold but cost extrapolation by 0.08–0.26) | **OK** as a reported cost; **not found** for progressive loss | Bansal and Anil 2211.09961 report gains. A depth / extrapolation **cost** of stability training is reported only for fixed-point penalties: Viakhirev 2608.18222's dose curve, with over-contraction suppressing conversion (its 1.0-weight "zero extrapolation" is from a subagent, unverified by me). Also the general caution that global contraction erases computation (LIT_CAND5_6). Pre-state the error-prefix explanation as a hypothesis. |
| N2 (gating the far answer loss on the model's own correctness at n + 2) | **NARROW** | Training only on one's own correct outputs is known: STaR 2203.14465 and rejection-sampling fine-tuning filter by correctness; loss truncation (Kang & Hashimoto 2004.14589) drops high-loss examples; self-paced learning. **Not found for a recurrent-depth hold / far-pass loss**, where Bansal does the opposite. Wording: "a correctness-gated variant of progressive / far-pass training (cf. Bansal 2202.05826, which deliberately trains on erroneous prefixes)". |
| N3 (the same far loss after the algorithm formed costs no depth: timing matters) | **NARROW** | Critical learning periods (Achille 1711.08856: early deficits permanently change what is learned) and staged curricula (Coconut; ATC 2609.19717's competence-gated stages) make timing effects known in general. For an auxiliary stability loss in looped models: not found. |
