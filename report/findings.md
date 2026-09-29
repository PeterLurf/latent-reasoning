# Readable vs usable alternatives in latent reasoning — what stands after Parts I–VI

Consolidated write-up (plan step 6, `docs/RESEARCH_TIMELINE.md`). Written 2026-09-25; Part VI (findings 8–10, the
revision table, Open, Caveats) added 2026-09-28 ~19:00 while addenda J (seed 2), M and O were still running — see
`docs/CURRENT_TASKS.md` for their state. Every number is taken from
`results/` or `docs/RESEARCH_LOG.md`; the detailed reports are `report/latent_branch_report.html` (Parts I–III) and
`report/protocol4_report.html` (Part IV). Part V is `docs/PROTOCOL5.md`, `docs/PROTOCOL5_RESULTS.md` and
`results/protocol5_confirm.json`. Part VI is `docs/PROTOCOL6.md` (addenda A–O) and `docs/PROTOCOL6_RESULTS.md`
(sources: `results/protocol6_confirm.json`, `results/p29.json`, `results/p30.json`, `results/p_l.json`,
`prosqa/res_*/`).

## Setting

A 2-layer decoder (d = 256) with Coconut-style continuous-thought recurrence (K = 6 latent steps unless stated) is
trained on graph reachability: 10 branches × 6 nodes, W of them attached to the start hub and the rest to a
shape-matched decoy hub, 62 nodes and 60 edges at every W. Delayed format: the query node comes after the latent
steps; upfront format: before. Train / discovery / confirm splits are disjoint by structural family, and every
confirmatory prediction was frozen before its data were drawn. Training needs a PARENT/CHILDREN lookup scaffold
(annealed to 0) and a gated depth curriculum; all claims are about models trained this way.

## Findings that stand

**1. Alternatives in one shared thought are separately usable, and less so as width grows.** Replacing one branch's
component of the thought with a donor's moves that branch more than the others at every width, with effects confined
to its deep (≥ 3-hop) nodes and exact-zero identity controls. The targeted effect falls with the number of attached
branches (seed 0: 0.085 → 0.026 from W = 2 to 8; seed 1: 0.300 → 0.059; seed 2: 0.324 → 0.114). A strict selectivity
bar (≥ 0.10 targeted, ≤ 0.03 collateral) is met only by seed 2 at W = 2. *Parts I–II, 3/3 seeds, fresh graphs.*

**2. Readable is not usable.** Linear probe read-out directions (filters) are causally inert: patching them changes
nothing (≤ 0.001 at every W) although the probes decode reachability; they hold 0.3% of the donor-difference energy, less
than a random 6-dim subspace (2.3%). The probes' activation patterns (Cov(h)·w) carry the effect (~88% of a full
patch in seed 0; 6–57× the filter effect at matched norm, P3 on fresh graphs, every seed).

**3. Branches compete for a shared attention budget, split per frontier edge.** Removing one branch makes the other
branches *more* reachable (signed collateral < 0; insertion > 0). In layer 1 the latent step's attention on the
current frontier has a roughly fixed total; per-edge attention halves when the frontier doubles at fixed W (fork vs
chain graphs: 0.49–0.60 in s0–s3 on fresh graphs, P25 primary 14/18). The Part II law "per-branch amplitude ∝ W^−β,
β ≈ 0.6–0.9" is this frontier-size law seen through W. Width changes the sweep's amplitude in all 22 competent models
(per-edge W8/W2 0.13–0.32).

**4. With width, mid-depth nodes move from the thought to query-time reading.** In the delayed format, 1-hop nodes
are always resolved at query time and ≥ 4-hop nodes always through the latent pathway; the boundary moves inward with
W (2–3-hop nodes switch to query-time reading). Part II P1 held in every seed; the 4-layer variant is the exception
(it keeps nodes in the latent pathway). In the upfront format the answer token never reads the graph and target-branch
edits are selective (targeted 0.14–0.74 across seeds, collateral ≤ 0.03 except one cell at 0.053).

**5. Recurrence buys reliability on the deepest queries, not overall accuracy.** Depth-matched 6-hop accuracy:
recurrent 0.59–0.66 vs all six pause-token and no-latent controls 0.50–0.56. Overall accuracy does not separate them
(a no-latent seed reaches 0.91).

**6. The thought runs a clocked breadth-first sweep; in single-step readers it is step-locked.** Layer-1 attention
reads the reachable frontier one depth at a time. In models that read each depth in a single step (9 of 22, fixed on
discovery graphs), blocking a depth's edges at its step cuts the branch's deeper nodes and blocking elsewhere does not
(P23′ 7/9), and the clock is the same at W = 2, 4, 8 (P24′ 7/9) — both just under the 0.80 bar ("mixed"); on four
new competent models 3/4 each (10/13 overall). The other 13 models read a depth over several steps (P23 1/13, P24
2/13). Pause-token models show no advancing sweep.

**7. Whether a model extrapolates is visible inside its training budget: is its sweep still running at the last step?**
Some models follow chains longer than any in training when given more latent steps; which ones is not set by the
latent budget K, the step embedding, randomised budgets or how deep the sweep gets. It is predicted by *end activity*
— the layer-1 frontier attention at the last trained step relative to its peak. Found on 22 models (AUC 0.81) and
frozen before a fresh test on 12 new ones: below 0.25, 1 of 6 extrapolate; at or above, 5 of 6 (AUC 0.83, P26
confirmed). Pooled: 2 of 15 vs 12 of 19. A sweep that has gone quiet by step K almost never extends; one still reading
the frontier usually does. *(Part VI: still close to necessary, but a weak predictor on new runs — see finding 9.)*

**8. The open/closed outcome is decided late in the curriculum and is mostly fixed once full depth is reached.**
15 000 further steps with a restarted optimizer and new data leave the end-activity class unchanged in 11 of 12 runs
(A1 "stable"). Branches restarted from checkpoints with fresh data and optimizer agree with their original's outcome
in 2/4 from stage-2 entry, 5/6 from stage-4 entry and 6/6 from full-depth entry (E1 "set" at full depth). The set point
is not absolute: about 1 run in 8 drifts slowly after full depth (one closes, one reopens without gaining
extrapolation), and a closure can be decided in the weights before end activity shows it. Training precision is not
the cause: same-card bf16/fp32 pairs differ neither in extrapolation (P27 not supported, b = 0, f = 2) nor in
curriculum progress (P28 not supported, 1 vs 1). Small numerical differences (precision, card, data order) steer which
solution a run lands on while the late curriculum is open. *Part VI addenda A, B, E, F, H.*

**9. Among open sweeps, whether the frontier crosses the trained horizon decides extrapolation.** P7 — layer-1
frontier excess on the distance-7 edge at test time, relative to the largest in-horizon excess — ranked extrapolation
on 72 existing models (AUC 0.96, end activity 0.82). As a fixed cut-off (P7 ≥ 0.2, no floor on the denominator) it
failed prospectively (P29: accuracy 0.50, AUC 0.73 on 8 runs). As a threshold-free ranking with a pre-registered floor
it held on 66 new runs (P30: 42 competent, 21 extrapolate; AUC 0.993 vs end activity 0.673; 0.96–1.00 on each of five
cards and both recipes; P7 ≥ 0.2 → 15/15 extrapolate). 41 of those 42 had high end activity, so an open sweep is close
to necessary but far from sufficient. With two spare thoughts (K = 8) P7 still ranks extrapolation (L1, AUC 0.90) and
12 of 13 non-extrapolators stop at distance 6 (L2): a failing sweep stops at the **training graph depth**, not at the
step budget. *Part VI addenda I, K, L; the converse test (training depth 5, addendum M) is running.*

**10. The sweep does not transfer to Coconut GPT-2 on ProsQA: the thoughts carry an answer computed in one pass.**
The official Coconut GPT-2 (reproduced: test 0.974) does not use its thoughts (0 thoughts 0.976). With the answer
blocked from everything before the thoughts (addendum G, 3 seeds) the thoughts become necessary (0 thoughts 0.49–0.52;
6 thoughts 0.94–0.97) but one thought already gives 0.93–0.94: no hop-by-hop sweep, no step-locked use. With the
candidate question moved after the thoughts (addendum J, 2 seeds; delayed format as in the toy model) GPT-2 is
competent (0.89, 0.88), one thought matches six, and there is again no sweep; whether zero thoughts suffice differs by
seed (0.85 vs 0.51; seed 2 running). Truncated to 4 layers (addendum N, 2 seeds) GPT-2 does not learn the delayed task
at all (0.54, 0.55). ProsQA's 3–6 hops fit inside one 12-layer pass, so under Coconut's recipe the continuous thoughts
carry the result rather than compute it. *Part VI addenda D, G, J, N.*

## What was revised or overturned

| Claim | When | Now |
|---|---|---|
| Pause ≈ recurrence to 4 hops; no-latent fails beyond 2 hops | Part I, one seed per control | Seed luck (Part II replication) |
| Budget divides ~1/W among branches | Part I (H-norm) | Per frontier edge; W is a proxy (Part IV) |
| Every recurrent model runs a step-locked clock | Part III | A subset (9/22 single-step readers; Part V) |
| The stop is keyed to the step embedding | Part III (clock freeze; P18, one run) | Not at training level: P19 not replicated, P22 fails (Part IV). The clock-freeze rescue on s3/V2 stands as an intervention result |
| A sweep that finishes inside the budget does not extrapolate | Part III (P16) | Fails at K = 8, 10 (P20, addendum G) |
| Wider thought / more layers change the width law as predicted | Part III (P11, P12) | Both failed; P12 in the opposite direction |
| Extrapolation is a seed-level outcome with no visible cause | Part IV | Still seed-level, but predictable in-budget from end activity (P26, Part V) |
| A restart rule can fix late-stage stalls | Part IV next steps | No: ≤ 8% saving; a lower LR helps more (Part V) |
| End activity predicts extrapolation | Part V (P26) | Near-necessary but weak on new runs (AUC 0.53 in P29's set, 0.67 in P30's); P7 ranks far better (Part VI) |
| Training without full depth keeps the sweep open | Part V (Open) | Not supported: continued stalled runs reached full depth and no open sweep closed (A3, Part VI) |
| fp32 training hurts extrapolation / stalls the curriculum | Part VI exploratory | Not supported on same-card pairs (P27, P28) |
| P7 ≥ 0.2 predicts extrapolation | Part VI exploratory | Fails as a fixed cut-off (P29); holds as a floored ranking (P30, L1) |
| Continuous thoughts in Coconut GPT-2 run the same sweep | Part VI (transfer question) | No: not used (D), a one-pass conduit when forced (G, J), not learned at 4 layers (N) |

## Training reliability

About half of all gated attempts before Part V never reached full curriculum depth, mostly by stalling in stages 2–4.
A late-stage restart rule saves at most ~8% of compute (results/stall_rule.json): slow successes and stalls overlap.
Halving the learning rate (1e-3 → 5e-4) on the no-step recipe gave 7/8 gate passes against 5/8 (R1 holds, but p = 0.28
at n = 8) at a cost in deep-query accuracy (matched 5-hop 0.69 vs 0.77). Every pilot failure was a late-stage stall;
no run stalled at stage 1.

## Open

*Part V's three open items were taken up in Part VI: why the sweep stays open (findings 8–9), training without full
depth (A3, not supported) and transfer to ProsQA (finding 10).* Still open on 2026-09-28:

* **Why the frontier crosses the horizon in some seeds.** P7 ranks extrapolation, but it is itself a seed-level
  outcome. Addendum O tests one training-level cause: a longer lookup scaffold (paired re-training of the P30 seeds,
  48 pairs; runs until Tue 2026-09-29).
* **Training depth vs step budget.** L2 says failing sweeps stop at the training depth; addendum M (training graphs
  capped at distance 5, K = 6) is the converse test (running).
* **ProsQA J1.** Seed 2 of addendum J decides whether the delayed-query model needs any thought at all.
* **Multi-step latent computation in a pretrained model.** Needs problems deeper than one forward pass, or a recipe
  closer to the toy model's (scaffold, gated curriculum); N shows that simply removing layers does not do it.

## Caveats

* One task family, one small architecture, trained with a lookup scaffold; transfer to Coconut GPT-2 on ProsQA was
  tested in Part VI (finding 10) and did not show the sweep.
* Effect sizes vary ~4× across seeds; the qualitative pattern is what replicates.
* The prospective P26 test used one recipe (no step embedding, K = 6, two learning rates, 12 competent runs); the
  22 exploration models span K = 4–10, three architectures and both step-embedding settings. The 0.25 threshold was
  chosen on the exploration set.
* Two of the 16 pilot runs trained in fp32 on a different card (both lr 5e-4, both competent); R1 without them is 5/6
  vs 5/8.
* nostep_s0 (P18, an extrapolating model) ended at curriculum stage 4; PROTOCOL3 had no full-depth clause.
* Two host crashes (bad RAM, one caught bit flip) on 2026-09-24; an integrity re-run found stored results intact.
* Part VI: the P7 floor (in_peak ≥ 0.03) and the ranking form of P30 were chosen after P29 failed, and tested only on
  new runs; P30's test set grew by 36 Vast.ai seeds by amendment before any P30 result was read (`docs/PROTOCOL6.md`).
* Part VI null results (P27, P28) come from 8 same-card pairs each: they show no effect of precision, not its absence.
* ProsQA: one recipe (Coconut's, no scaffold or gated curriculum), GPT-2 only, 2–3 seeds per condition.
