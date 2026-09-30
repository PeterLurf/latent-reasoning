# PROTOCOL6 results (Part VI; compiled 2026-09-26, in progress)

Sources: `results/protocol6_confirm.json` (`python -m latent.protocol6 score`), `results/p6_endact.json`,
`results/p6_strict.json`, `results/p6_traj.json`. Predictions and definitions: `docs/PROTOCOL6.md` (addenda A–R, each
written before the data it scores; amendments dated there). Chronology: `docs/RESEARCH_LOG.md`.
EA = end activity (PROTOCOL5 addendum B); high = EA ≥ 0.25. Extrapolates = d7 ≥ 0.60 with lower bound > 0.55 at K′ = 12.

## Verdicts so far

| Test | Evidence | Reading |
|---|---|---|
| **A1** does further training close the sweep? (12 competent pilot runs, +15 000 steps, fresh optimizer) | EA class unchanged 11/12; high → low 0/6; low → high 1/6 | **stable** |
| **A2** P26 on the continued models | low 1/5 extrapolate, high 4/7; AUC 0.71 | confirmed (weaker than before) |
| A3 stalled runs continued (exploratory) | 3 of 5 reach full depth; no open sweep closes | see below |
| **P26b** P26 on a second recipe (step embedding, lr 5e-4; 12 runs) | high 5/6 extrapolate, low 0/1; AUC not computable | confirmed by the frozen rule, thin |
| **P23″ / P24″** strict clock, pooled prospective sets | 3/5, 4/5 (one new single-step reader) | mixed / confirmed at the bar |
| B1 when is EA visible? (checkpoint trajectories) | final class present 1 000 steps after full depth in 7/8 | exploratory |
| **E1** when is the outcome decided? (18 branches) | agreement from stage-2 entry 2/4, stage-4 entry 5/6, full-depth entry 6/6 | open / partial / **set** |
| **P27** does bf16 training make extrapolation more likely? (8 same-card pairs) | both arms competent in 2 pairs (53, 54); fp32-only extrapolator in both (b = 0, f = 2) | **not supported** |
| **P28** do fp32 runs stall in the curriculum more? (addendum H, 8 new same-card pairs) | full depth 6/8 in both arms; bf16-only 1 (s72), fp32-only 1 (s73) | **not supported** |
| **ProsQA T0** competence (official Coconut GPT-2) | test 0.974 (validation 0.973) | competent |
| **ProsQA T1** hop-by-hop sweep | layer 7 reads hops 1–2 only; thoughts 2–6 identical; clock [1,1,1] | **absent** |
| **ProsQA T2** step-locked use | blocking at all thoughts ≤ 0.09 of 12.2 nats; 0 thoughts 0.976 vs 6 thoughts 0.974 | **thoughts not needed** |
| **P29** does P7 ≥ 0.2 predict extrapolation on new runs? (addendum I; 8 competent test runs, 3 extrapolate) | accuracy 0.50, AUC 0.73 (EA 0.53) | **not supported** |
| **ProsQA G** bypass-blocked Coconut (3 seeds) | competent (0.972, 0.972, 0.940); K′ = 0 at chance (0.49–0.52); no sweep; not step-locked; K′ = 1 already 0.93–0.94 | thoughts **needed**, sweep **absent** — a conduit, not a search |
| ProsQA J delayed query (addendum J; 3 seeds) | competent (0.89, 0.88, 0.88); K′ = 1 ≈ K = 6; no sweep; not step-locked; K′ = 0 0.85 / 0.51 / 0.48 | **prefix-computed** (one pass); J1 thoughts needed (2 of 3 seeds) |
| ProsQA N delayed query on 4-layer GPT-2 (addendum N; 2 seeds) | test 0.544 / 0.552 | **not learned** (N0 fails on both) |
| **P30** does P7 *rank* extrapolators on new runs? (addendum K; 42 competent test runs, 21 extrapolate; 4 cards, 2 recipes) | AUC 0.993 (EA 0.673); per card 0.96–1.00; per recipe 0.995 / 1.00; Spearman with d7 0.87 | **supported** |
| **L1 / L2** K = 8 (two spare thoughts): does P7 still rank extrapolators, and do failing sweeps stop at the training depth? (addendum L; 17 competent, 4 extrapolate) | AUC 0.90 (EA 0.59); non-extrapolators with P7 < 0.10: 12/13 | **supported / supported** |
| **M1 / M2** depth cap 4, K = 6 (one spare thought): does a failing frontier stop at the new training depth (distance 5)? (addendum M; 17 runs, 10 competent, 2 extrapolate by one level) | P6 < 0.10: 7/10 (share 0.70 ≥ 0.30); M2 needs ≥ 3 of each outcome, only 2 extrapolate (descriptive AUC 0.56) | **supported / unresolved** |
| **P1 / P2** ProsQA deeper than training: training-depth wall in GPT-2? do 12 thoughts help? (addendum P; 6 GPT-2 models, long + control sets) | walls 5/6 (a_6 0.48–0.52 with c_6 ≥ 0.80; j_s1 c_6 0.792 → "length failure"); 12-thought gain ≥ 0.05 in 0/6 | **supported / not supported** — but see the depth-matched analysis: the decline tracks the added chain, not the training depth |
| **O1** does a longer lookup scaffold (full depth reached while it is still on) make extrapolation more common? (addendum O; 48 same-card pairs) | successes 13 vs 13; discordant l = 6, b = 6 (sign p 0.61); local 6 vs 4, Vast 0 vs 2 | **not supported** |
| **Q1 / Q2** does GPT-2's depth horizon follow the training depth? (addendum Q; addendum-G recipe trained on 3–4 hop problems only, 2 seeds) | acc 3–4 hop vs 5–6 hop (never trained) at K = 6: s0 0.947 vs 0.951, s1 0.928 vs 0.938 (gap −0.003, −0.010); K′ = 12 gain on 5–6 hop 0.000, 0.000 | **not supported / not supported** |
| **R1 / R2 / R3** depth cap 3 and more cap 4: does the horizon follow the training depth at every depth? (addendum R; cap 3: 12 runs, 8 competent; cap 4 pooled with M: 18 competent) | R1: P5 < 0.10 in 8/8; R2 shares cap 3 1.00, cap 4 0.67, depth 6 0.92; R3: 2/18 extrapolate (unresolved) | **supported / supported / unresolved** — details below |

## Addendum A: continuation

| Run | EA before → after | Extrapolates before → after (d7 after) |
|---|---|---|
| rel_base_s20 | 0.001 → 0.001 | ✘ → ✘ (0.533) |
| rel_base_s24 | 0.166 → **0.321** | ✘ → ✘ (0.596) |
| rel_base_s25 | 0.029 → 0.049 | ✔ → ✔ (0.706) |
| rel_base_s26 | 0.009 → 0.057 | ✘ → ✘ (0.560) |
| rel_base_s27 | 0.621 → 0.571 | ✔ → ✔ (0.800) |
| rel_lr5_s20 | 1.000 → 0.961 | ✔ → ✔ (0.624) |
| rel_lr5_s21 | 0.930 → 0.888 | ✘ → ✘ (0.599) |
| rel_lr5_s22 | 0.629 → 0.600 | ✔ → ✔ (0.711) |
| rel_lr5_s23 | 0.829 → 0.856 | ✔ → ✔ (0.608) |
| rel_lr5_s24 | 1.000 → 0.759 | ✔ → **✘** (0.561) |
| rel_lr5_s26 | 0.013 → 0.011 | ✘ → ✘ (0.502) |
| rel_lr5_s27 | 0.123 → 0.120 | ✘ → ✘ (0.502) |

* Every continuation restarts the optimizer and data stream (amendment: 7 of 17 runs had no `state.pt`).
* A3: rel_base_s23 reached full depth (step 38 500) and kept EA 1.00 and extrapolation (0.711); nostep_s0 reached full
  depth (43 000), EA 0.89 → 0.63, lost extrapolation (K′ 12: 0.631 → 0.543); rel_lr5_s25 reached full depth, EA stayed
  0.02; rel_base_s21/s22 stayed at stage 2. Reaching full depth closed no open sweep, so "never training at full depth
  keeps the sweep open" is not supported.

## Addendum B: second recipe (step embedding, K = 6, lr 5e-4)

| Run | Precision | Final stage | Gate | EA | d7 @K′12 | Extrapolates |
|---|---|---|---|---|---|---|
| dev6_s30 | bf16 | full | ✔ | 1.000 | 0.719 | ✔ |
| dev6_s31 | bf16 | 3 | ✘ | −0.002 | 0.506 | ✘ |
| dev6_s32 | bf16 | full | ✔ | 0.952 | 0.768 | ✔ |
| dev6_s33 | bf16 | full | ✔ | 0.426 | 0.609 | ✔ |
| dev6_s34 | bf16 | 4 | ✘ | 0.710 | 0.529 | ✘ |
| dev6_s35 | bf16 | full | ✔ | 0.151 | 0.582 | ✘ |
| dev6_s36 | bf16 | 4 | ✘ | 0.882 | 0.521 | ✘ |
| dev6_s37 | bf16 | full | ✘ (matched-5 0.594) | 0.278 | 0.530 | ✘ |
| dev6_s38 | bf16 | full | ✔ | 1.000 | 0.637 | ✔ |
| dev6_s39 | fp32 | full | ✔ | 0.366 | 0.498 | ✘ |
| dev6_s40 | fp32 | full | ✔ | 0.674 | 0.667 | ✔ |
| dev6_s41 | fp32 | 4 | ✘ | 0.337 | 0.524 | ✘ |

* Gate 7/12, all failures late (stages 3–4, or accuracy at full depth); this recipe stalls more than the no-step
  recipe at the same LR (PROTOCOL5: 1/8). Most runs end with an open sweep (6 of 7 competent).
* Strict clock: one single-step reader (dev6_s38: P23 ✘, P24 ✔). The step-embedding recipe rarely yields single-step
  readers; the clock claim stays unsettled (pooled prospective P23″ 3/5, P24″ 4/5).
* Extension B′ (added after the 12 were seen; not pooled into P26b): dev6_s42–44 (fp32, 2070) all end with open sweeps
  (EA 0.59, 0.46, 0.52) and none extrapolates — the observation that prompted addendum F.

## Addendum E: branching

| Branch point | Agreement (competent branches) | Reading |
|---|---|---|
| stage-2 entry | 2/4 (dev6_s30's two branches stalled at stage 4) | open |
| stage-4 entry | 5/6 | partial |
| full-depth entry | 6/6 | set |

* dev6_s35 (closed sweep, EA 0.15): branches from stage 2 ended 0.35 / 0.30, from stage 4 0.81 / 0.16, from full-depth
  entry 0.15 / 0.14. Its measured EA at full-depth entry was still 0.57: the closure was determined by the weights at
  that point before EA showed it (B1 trajectory: 0.57 → 0.32 → 0.15 over the rest of the schedule).
* Extrapolation followed the original in 17 of 18 branches.

## Addendum N: delayed-query ProsQA on 4-layer GPT-2 — both seeds (scored 2026-09-28 ~12:05 / ~15:40)

`prosqa/res_n4_s1/` (sha-verified; Vast 53139198 destroyed ~12:05). Validation stayed at chance throughout (0.45–0.55
from stage 3 to epoch 49).

* **N0: not competent** — test 0.552 at K = 6 (bar 0.85): **not learned by 4 layers**. As pre-registered, N1–N3 are
  reported, not scored: K′ = 0–5 all 0.55–0.57, K′ up to 12 at most 0.586; no sweep (layer 2, clock [1, 4, 2]);
  all-thought blocking moves the margin by ≤ 0.35 nats (base margin 0.55).
* **Seed 0** (`prosqa/res_n4_s0/`, sha-verified, Vast 53139194 destroyed ~15:40): test 0.544 — **not competent**;
  K′ = 0–5: 0.53–0.56; no sweep; blocking has no effect (≤ 0.13 nats). **Both seeds agree: N0 fails** (addendum-N
  reading: "not learned by 4 layers"; no seed 2 needed).
* Reading: making one pass too shallow does **not** push GPT-2 to spread the search over its thoughts
  under Coconut's recipe; the 4-layer model fails to learn the delayed task at all, while the 2-layer toy model (trained
  from scratch, with the gated depth curriculum and auxiliary parent/children losses) does learn a stepwise sweep.

## Addendum J: delayed-query ProsQA — seeds 0, 1 and 2 (scored 2026-09-28 ~03:50 / ~04:45 / ~21:55)

`prosqa/res_j_s0/`, checkpoint `prosqa/ckpt/j_s0_checkpoint_50` (sha-verified; Vast 52981944 destroyed ~03:50).
Validation climbed with the curriculum (0 → 0.82 at stage 4 → 0.88 at stage 6).

| Test | Seed 0 | Reading |
|---|---|---|
| J0 competence (≥ 0.85) | test 0.890 | competent |
| J1 thoughts needed | K′ = 0: **0.850**; K = 6: 0.890 | **not needed** |
| J1b work in the thoughts | K′ = 1 0.898 vs K = 6 0.890 | **prefix-computed** |
| J1c depth dependence | K′ = 2: path 3 0.871, 4 0.903, 5 0.928, 6 0.833 (n = 12) | **not depth-dependent** |
| J2 sweep | layer 10, clock [1, 1, 1] valid and test | **absent** |
| J3 step-locked | blocking the hop-h statement at every thought: +0.07, −0.11, −2.4, −5.0, −6.1, −7.4 nats (base 8.5) | **not step-locked** |

Seed 1 (`prosqa/res_j_s1/`, sha-verified, Vast 52981949 destroyed ~04:45): J0 0.876 (competent); J1 K′ = 0 **0.514**
(**needed**); J1b K′ = 1 0.832 vs K = 6 0.876 (+0.04, **prefix-computed**); J1c not depth-dependent (K′ = 2: 0.861 /
0.871 / 0.870 / 1.0); J2 absent (layer 10, clock [1, 1, 2]); J3 not step-locked (all-thought blocking of hops 3–6:
−2.5 to −8.1 nats, base 7.7).

**Seeds agree on J0, J1b, J1c, J2, J3 and disagree on J1** — seed 0 carries the answer through the `<start-latent>`
position (K′ = 0 works), seed 1 through thought 1 (K′ = 0 at chance). As pre-registered, seed 2 (`prosqa_j_s2.yaml`,
Vast 53145506, launched ~04:50) decides J1. Either way the one-pass reading holds on both seeds: one thought (or none)
carries almost all of the accuracy, and nothing is computed step by step.

**Seed 2 — tiebreak (scored 2026-09-28 ~21:55; `prosqa/res_j_s2/`, checkpoint `prosqa/ckpt/j_s2_checkpoint_50`,
sha-verified; Vast 53145506 destroyed ~21:50).** J0 test 0.876 (competent); J1 K′ = 0 **0.478** (**needed**);
J1b K′ = 1 0.814 vs K = 6 0.876 (+0.06, **prefix-computed**); J1c K′ = 2 path 3 0.886 vs ≥ 5 0.852 (−0.03, **not
depth-dependent**); J2 absent (layer 10, clock [1, 1, 2]; exploratory: hops 4–6 peak at thoughts 3, 3, 4); J3 not
step-locked (all-thought blocking of hops 1–6: +0.22, +0.04, −2.5, −4.5, −5.1, −7.9 nats, base 7.7).

**Addendum J final (2 of 3 seeds):** J0 competent, **J1 thoughts needed** (seeds 1, 2; seed 0 hands the answer over
at `<start-latent>`), J1b **prefix-computed**, J1c not depth-dependent, J2 no sweep, J3 not step-locked (all three
seeds agree on everything but J1). Reading, as pre-registered: J0 + J1 with J1b prefix-computed — with a delayed query,
GPT-2 packs the reachable set into the prefix pass and one thought carries it; no stepwise latent computation.

* The answer works with no thoughts at all because the `<start-latent>` position — whose final state becomes thought
  1 and which the mask leaves open for that reason — sees the statements and the start entity. GPT-2 computes the
  reachable set from the start entity in one 12-layer pass over the prefix and hands it over there. The large
  all-thought blocking effects for hops 3–6 show that the graph information does flow through the start/thought
  positions; it is simply not spread over steps. Closing that position would change little: thought 1 *is* its output.
* Reading (pending seed 1): with a delayed query, GPT-2 does the whole ProsQA search in a single forward pass — ProsQA's
  3–6 hops fit inside 12 layers — so continuous thoughts carry the result rather than compute it. Forcing multi-step
  latent reasoning in GPT-2 needs problems deeper than one pass can solve.

## Addendum G: bypass-blocked Coconut on ProsQA — all 3 seeds (scored 2026-09-27 11:45 / 13:15 / 14:35)

Official Coconut GPT-2 recipe, 50 epochs, with the answer mask (`<end-latent>` and answer positions cannot see
anything before `<start-latent>`). Results in `prosqa/res_g_s1/`, checkpoint `prosqa/ckpt/g_s1_checkpoint_50`
(sha-verified; Vast instance destroyed).

| Test | Seed 1 | Rule | Reading |
|---|---|---|---|
| G0 competence | test 0.972 (validation 0.957) | ≥ 0.90 | **competent** |
| G1 thoughts needed | K′ = 0: 0.498; K = 6: 0.972 (diff 0.474) | K′0 ≤ 0.65 and diff ≥ 0.20 | **needed** |
| G2 hop-by-hop sweep | layer 7; clock valid [2, 4, 3], test [2, 3, 3] | increasing, test = valid | **absent** |
| G3 step-locked use | blocking the hop-h path statement at every thought: −0.10, −0.35, −0.08, 0, +0.07, +0.06 nats (base margin 11.7) | as addendum D | **not step-locked** |

Seed 2 (`prosqa/res_g_s2/`, checkpoint sha-verified, instance destroyed 13:10): G0 test 0.940 (competent); G1 K′ = 0
0.516, diff 0.424 (**needed**); G2 clock valid [3, 2, 1], test [2, 2, 1] (**absent**); G3 delta_all −0.18, −0.18, −0.18,
−0.07, −0.06, −0.01 of a 10.6-nat margin (**not step-locked**). Exploratory K′ = 0/1/2/3: 0.516 / **0.932** / 0.946 /
0.946. Seed 0 (`prosqa/res_g_s0/`, sha-verified, instance destroyed 14:30): G0 0.972; G1 K′ = 0 0.494 (**needed**); G2 clock
valid [2, 4, 4], test [2, 3, 5] (**absent**); G3 delta_all −0.07, −0.26, −0.05, −0.07, −0.02, +0.04 of 12.2 nats (**not
step-locked**); K′ = 0/1/2/3: 0.494 / **0.932** / 0.968 / 0.970.

**Addendum-G reading (all 3 seeds agree on every test): competent; thoughts needed; no hop-by-hop sweep; not
step-locked.** With the bypass blocked, GPT-2's continuous thoughts become necessary but act as a one-step conduit for an
answer computed in the prefix pass.

* **Exploratory, the key follow-up (`scratch/kprime_small.py`, `res_g_s1/kprime_small_test.json`).** Accuracy with
  K′ = 0, 1, 2, 3 thoughts: 0.498, **0.938**, 0.962, 0.972; at K′ = 1, path length 3/4/5/6: 0.95/0.94/0.91/1.00. The
  single thought at K′ = 1 is the final hidden state of the prefix pass at `<start-latent>`. The problem is solved in
  GPT-2's forward pass over the question; the continuous thought mostly **carries** the answer past the mask rather
  than computing it step by step. The question sentence ("Is Tom a wumpus or a yumpus?") precedes the thoughts, so the
  prefix can answer it directly — the toy model's *delayed* format (query after the thoughts) is what rules this out.
* Blocking single path statements barely matters (≤ 0.35 of 11.7 nats), consistent with GPT-2 spreading statement
  content across later prefix positions (every later token can attend to it), so no single statement is a bottleneck.

## Addendum D: ProsQA transfer

* The official Coconut GPT-2 (reproduced: 0.973 validation, 0.974 test) does not use its thoughts: accuracy is the
  same with 0 to 12 thoughts (0.974–0.976), blocking a path statement at every thought step barely moves the answer,
  and the layer that attends most to the graph reads the first two hops at every thought without advancing. This
  replicates Aswal et al. (arXiv 2606.12689) with an independent implementation.
* Exploratory, same model: hiding the statements from the answer positions leaves 0.755 (0 thoughts) / 0.73 (6) — the
  question sentence already carries most of the answer; hiding everything before the thoughts gives 0.52 / 0.54
  (chance). The thoughts carry no answer information. The transfer question therefore needs a model whose answer
  must pass through the thoughts: addendum G.

## Addendum F: training precision (scored 2026-09-26 ~22:05)

Seeds 50–57, each trained in bf16 and in fp32 on the same card (same init and data stream); seeds 50–53 also in fp32
on the RTX 2070. The seed-53 pair was the last to finish (21:47).

| Seed | bf16: depth / gate / d7 / EA | fp32 (same card): depth / gate / d7 / EA | fp32 on 2070: depth |
|---|---|---|---|
| 50 | full / ✓ / **0.707** / 0.83 | 4 / – / 0.498 / 0.57 | 4 |
| 51 | full / ✓ / 0.498 / 0.08 | 4 / – / 0.505 / 0.80 | 4 |
| 52 | 4 / – / 0.497 / 0.51 | 4 / – / 0.497 / 0.76 | 3 |
| 53 | full / ✓ / 0.564 / 0.54 | full / ✓ / **0.601** / 0.86 | 3 |
| 54 | full / ✓ / 0.532 / 0.36 | full / ✓ / **0.728** / 0.63 | — |
| 55 | full / – / 0.517 / 0.54 | 3 / – / 0.512 / 0.15 | — |
| 56 | full / ✓ / 0.518 / 1.00 | full / – / 0.503 / 0.68 | — |
| 57 | 4 / – / 0.506 / 0.05 | full / ✓ / 0.514 / 0.43 | — |

(d7 = 7-hop accuracy at K′ = 12; bold = extrapolates by the pre-registered rule.)

* **P27 not supported.** Only two pairs have both arms competent, and in both the fp32 model is the one that
  extrapolates (b = 0, f = 2; one-sided sign test p = 1.0). The test had little power, but the direction is the
  opposite of the prediction. The exploratory pattern it came from (bf16 14/19 vs fp32 3/8 among high-EA models) does
  not survive a same-card, same-recipe design: it was most likely the card/recipe confound named in the addendum.
* **Extrapolation is rare in this recipe whatever the precision.** 3 of 8 competent models extrapolate (bf16 1/5,
  fp32 2/3); among high-EA competent models, bf16 1/4 and fp32 2/3. The seed-53 fp32 model only just passes
  (0.601, lower bound 0.583).
* **P26 within arms (secondary).** No competent model with EA < 0.25 extrapolates (bf16 0/1; fp32 none low), and
  high EA is again far from sufficient (bf16_s56: EA 1.00, d7 0.518). By the frozen rule bf16 reads "confirmed" (thin)
  and fp32 "unresolved".
* **What precision does change is curriculum progress.** Full depth: bf16 6/8, fp32 4/8 on the same card, fp32 on the
  2070 0/4. Discordant pairs: bf16-only 3 (50, 51, 55), fp32-only 1 (57). (Seed 57 was not in the exploratory basis
  of addendum H, which listed seeds 50–52 and 54–56.) Addendum H tests this prospectively on seeds 70–77.
* **Card matters even at fixed precision.** Seeds 50–53 in fp32 reach stages 4, 4, 4, full on the RTX 4060 but 4, 4, 3, 3
  on the 2070: identical init and data, different hardware arithmetic, different curriculum outcome — the same
  sensitivity to small numerical differences seen in the branching experiment (addendum E).

## Addendum H: does fp32 stall the curriculum? (scored 2026-09-27 ~10:30)

| Seed | bf16: depth / gate / extrap / EA | fp32: depth / gate / extrap / EA |
|---|---|---|
| 70 | full / ✓ / – / 1.00 | full / ✓ / **✓** / 1.00 |
| 71 | 3 / – / – / 0.69 | 4 / – / – / 0.52 |
| 72 | full / – / – / 0.60 | 3 / – / – / 0.00 |
| 73 | 4 / – / – / 0.73 | full / ✓ / – / 0.78 |
| 74 | full / – / – / 0.68 | full / – / – / 0.59 |
| 75 | full / ✓ / **✓** / 0.95 | full / ✓ / **✓** / 0.76 |
| 76 | full / ✓ / **✓** / 1.00 | full / ✓ / – / 1.00 |
| 77 | full / ✓ / **✓** / 1.00 | full / ✓ / **✓** / 0.72 |

* **P28 not supported:** b = 1 (s72), f = 1 (s73), sign test p = 0.75. Both arms reach full depth 6/8. The addendum-F
  imbalance (3 vs 1) was chance; pooled over F and H (16 pairs, descriptive) it is 4 vs 2 (p = 0.34).
* Extrapolation, pairs with both arms competent (70, 75, 76, 77): bf16-only 1, fp32-only 1. With addendum F that is
  bf16-only 1 vs fp32-only 3 over 6 pairs. **Training precision affects neither curriculum progress nor
  extrapolation** on the same card. The RTX 2070 fp32 arm (0/4 full depth) remains unexplained (chance vs card; small n).

## Addendum L: two spare thoughts — the horizon stays at the training depth (scored 2026-09-28 ~13:20)

23 runs (`k8_s200..s222`, K = 8, depth-6 training graphs; 4060 and 3070 bf16, 2070 fp32), all trained (the noon
cut-off dropped none). 17 competent (4 extrapolate), 6 not (stalls; one full-depth gate failure). No run below the
floor. `results/p_l.json`, P7 from the frozen code (seed 2033).

* **L1 supported:** AUC of P7 0.90 (EA 0.59). Per card: 4060 0.80, 3070 1.00 (2070: no extrapolator).
* **L2 supported:** 12 of the 13 non-extrapolators have P7 < 0.10 — with two thoughts to spare, their frontier still
  stops at distance 6, the deepest distance seen in training. The exception (k8_s213, P7 0.19, d7 0.578) is borderline.
* Extrapolation is rarer at K = 8 on this recipe (4/17 vs 21/42 at K = 6). More thoughts do not buy longer reach.
* Reading: the horizon a failing sweep stops at is the **training depth**, not the step budget. Addendum M (training
  depth moved to 5 at K = 6) is the converse test.

## Addendum K: P30 — the frontier crossing the horizon predicts extrapolation (scored 2026-09-28 ~03:40)

Test set: 66 new runs (`p30_s100..159` on the RTX 4060, RTX 3070 and two Vast.ai RTX A5000s; `dev6_s51..56` on the RTX
2070), all trained as pre-registered. 42 competent (21 extrapolate), 22 not competent (mostly stage-4 stalls; two
full-depth gate failures), 2 stage-1 failures after the relaunch budget (`p30_s136`, `p30_s159`, no model). No competent
run fell below the floor. P7 from the frozen code on fresh graphs (seed 2033); `results/p30.json`.

| | AUC for extrapolation | Spearman with d7 |
|---|---|---|
| **P7** | **0.993** | **0.87** |
| End activity | 0.673 | 0.34 |

* **P30 supported** (AUC ≥ 0.80 and above EA). Per card: 4060 1.00, 3070 0.96, 2070 1.00, Vast A 1.00, Vast B 1.00; per
  recipe: no step embedding 0.995, dev6 1.00.
* P7 ≥ 0.2: 15 of 15 extrapolate. P7 < 0.1: 3 of 24 extrapolate, all marginally (d7 0.60–0.62; the other near-threshold
  extrapolators have P7 0.09–0.13). The fixed 0.2 cut-off of P29 would have scored 0.86 here; the ranking is what holds.
* 41 of the 42 competent runs have high end activity, and only half of them extrapolate: among open sweeps, **whether the
  frontier crosses the trained horizon decides extrapolation**, prospectively and across cards and recipes. With P29's
  failure (fixed threshold, unfloored ratio) this is the second prospective test of the account, and the first as a
  ranking.

## Addendum I: P29 (scored 2026-09-27 ~13:25)

Test set complete: `stall_{bf16,fp32}_s73`, `dev6_s48..50`, `p29_s80..89` (s90–95 dropped by the stopping rule).
Competent: 8 (3 extrapolate); stage-4 stalls: stall_bf16_s73, p29_s81, s82, s83, s85; gate failures at full depth:
dev6_s49, p29_s89. P7 on fresh graphs (seed 2033, frozen code), `results/p29.json`.

| Run | in-horizon excess | P7 | EA | d7 | extrapolates |
|---|---|---|---|---|---|
| p29_s84 | 0.088 | 0.51 | 0.97 | 0.651 | ✓ |
| p29_s87 | 0.093 | 0.11 | 0.53 | 0.646 | ✓ |
| dev6_s50 | 0.112 | 0.08 | 0.64 | 0.609 | ✓ |
| dev6_s48 | **0.005** | 0.34 | 0.24 | 0.541 | – |
| stall_fp32_s73 | **0.0003** | 0.22 | 0.78 | 0.511 | – |
| p29_s80 | 0.089 | 0.01 | 0.82 | 0.515 | – |
| p29_s86 | 0.026 | 0.02 | 0.17 | 0.502 | – |
| p29_s88 | 0.067 | 0.02 | 0.97 | 0.534 | – |

* **P29 not supported** (accuracy 0.50 < 0.65; AUC 0.73). EA does no better here (AUC 0.53).
* Why, read after scoring (exploratory): the two false positives have a near-zero denominator (below the minimum of the
  72-model basis), and two extrapolators sit just under the cut-off (0.08, 0.11). Ranked, P7 separates every run with
  a measurable in-horizon frontier (extrapolators 0.08–0.51 vs 0.01–0.02). The K = 8/10 models show the same pattern.
  The ranking survives; the fixed threshold and the unfloored ratio do not. Tested again, prospectively, in addendum K.

## Exploratory: after full depth — sharpness, drift, reopening (2026-09-26 ~23:45)

Sources: `results/sharpness.json` (`scratch/sharpness.py`), `results/p6_traj.json`. Not pre-registered.

* **Attention sharpness at full-depth entry does not predict the outcome.** Across 18 checkpointed runs, the
  largest single-edge attention share at full-depth entry is 0.156–0.243 for the 6 extrapolators and 0.108–0.356 for
  the others; the reachable/decoy ratio and the entropy overlap just as much. The "over-committing index-matching
  logit" account (Zhu et al. 2026) is not visible in these summary measures.
* **The class is usually, not always, fixed at full depth.** Of 17 original runs with ≥ 5 checkpoints after full
  depth, 15 keep the EA class they have 1 000 steps after full-depth entry. The two exceptions drift slowly in
  opposite directions: `dev6_s35` closes (0.57 → 0.32 → 0.15) and `prec_bf16_s53` **reopens** (0.06 at full-depth entry,
  step 8 000 → 0.54 by step 22 000). Within the high class, EA also drifts substantially (dev6_s42 0.94 → 0.59,
  dev6_s40 0.41 → 0.67). Branching from full-depth checkpoints (E1, 6/6) and the continuation test (A1, 11/12) agree
  that most runs are set, but the set point is not absolute.
* **Within a run, EA and 7-hop accuracy do not move together.** In the 6 runs whose EA moves ≥ 0.2 after full depth,
  the within-run Spearman correlation of EA and d7 is positive in 3 and negative in 3. The reopened `prec_bf16_s53`
  improves d7 only from 0.50 to 0.57 and never extrapolates; `dev6_s35` closes while d7 slightly rises. EA works as a
  between-run marker of which solution a model found, not as a dial that moves extrapolation inside one run.

## Exploratory: does the frontier cross the trained horizon? (2026-09-27 ~06:50; basis of addendum I)

Source: `results/far_sweep.json` (`scratch/far_sweep.py`, long-chain graphs of `length_gen`, seed 99).
P7 = layer-1 frontier excess on the distance-7 edge (max over steps 1–12) ÷ the largest in-horizon excess.

| Predictor (72 competent K = 6 models, 36 extrapolate) | AUC for extrapolation | Spearman with d7 |
|---|---|---|
| End activity (EA) | 0.82 | 0.51 |
| **P7** | **0.96** (0.98 within high-EA) | **0.84** |

* P7 ≥ 0.2: 28 of 29 extrapolate; P7 < 0.2: 8 of 43. Most high-EA non-extrapolators have P7 ≈ 0 (e.g. stall_bf16_s70
  EA 1.00 / P7 0.03, prec_bf16_s56 1.00 / 0.02, stall_fp32_s76 1.00 / 0.00): their sweep is active at step 6 but the
  frontier never reaches distance 7. Failure to extrapolate is mostly **propagation**, not readout.
* Exceptions: two extrapolators with P7 ≈ 0.03 (rel_base_s25, low EA too; rel_lr5_s23), and two non-extrapolators
  near P7 = 0.2 (sharp_fp32_s60, rel_base_s26_ext).
* Caveats: threshold chosen on these data; same graphs as d7. Tested prospectively on fresh graphs and new runs in
  addendum I (P29).
* **K = 8 and K = 10 models (added 2026-09-27 ~12:15, `results/far_sweep_k.json`).** 8 competent models, 3 extrapolate:
  P7 0.10 (budget_k10_s3), 0.58 (budget_k8_s1), 0.14 (nostep_k8_s1) vs 0.00, 0.00, 0.01, 0.01, 0.14 (randk_s2, d7 0.563)
  for the rest — AUC ≈ 0.90, but the 0.2 cut-off would miss two of the three. These models have thoughts to spare
  beyond step 6, yet the non-extrapolators still never reach the distance-7 edge: the horizon a failing sweep stops at
  is the **training graph depth** (6), not the number of trained thoughts.


### Addendum M (scored 2026-09-28 ~20:10, `scratch/score_m.py`, `results/p_m.json`)

* 10 of 17 runs competent (4 stopped at stage 4, 3 reached the cap but missed matched 5-hop ≥ 0.60); all 10 above the
  floor (in5 0.035–0.108).
* **M1 supported:** 7 of 10 have P6 < 0.10 — the frontier stops at distance 5, the capped training depth, although a
  sixth thought is available. Together with L2 (depth-6 training, 8 thoughts: 12/13 failing sweeps stop at 6) the
  horizon of a failing sweep follows the **training depth**, not the step budget.
* **M2 unresolved** (2 extrapolators by one level). Descriptive: the three runs whose frontier crosses (P6 0.12, 0.36,
  0.42) include one extrapolator (m_s309, matched 6-hop 0.611, d7 0.612); m_s314 crosses (P6 0.36, d7 0.591) but its
  matched 6-hop is 0.552; m_s313 extrapolates by one level (0.667) with P6 = 0.00 — the one clear case of a
  non-frontier route, and its d7 is 0.523.
* Exploratory: extrapolation by one level is rarer under the cap (2/10) than extrapolation to distance 7 in depth-6
  K = 6 models (21/42 in P30).

### Addendum P (scored 2026-09-29 ~10:00, `prosqa/score_p.py`, `prosqa/res_p/p_score.json`)

* Accuracy at K = 6 by extension m (0 / 2 / 4 / 6): G 0.94–0.97 / 0.65–0.73 / 0.48–0.52 / 0.49–0.50; J 0.88–0.89 /
  0.64–0.68 / 0.51–0.53 / 0.48–0.52. Control sets (same added statements, answer path unchanged): G 0.92–0.97 at every
  m; J 0.79–0.82 at m = 6. So added length alone costs G nothing and J ~0.07; the collapse needs the answer to go
  through the added chain. **P1 supported** (5 walls, 1 length failure). **P2 not supported**: K′ = 12 never helps
  (a_6 at K′ 0 / 1 / 6 / 12 all 0.46–0.54). My pre-registered expectation ("gradual decline, no wall") was wrong on
  the wall.
* **Depth-matched secondary (decisive for the reading).** Within m = 2, problems whose paths stay inside the training
  depth (5–6 hops) are no easier than those beyond it (7–8 hops): 0.64–0.74 vs 0.57–0.72 per model; within m = 4,
  7–8 ≈ 9–10 ≈ chance. Original (m = 0) 5–6-hop problems score 0.88–1.00. Accuracy follows the number of *added*
  hops, not the path length, so this is **not a training-depth horizon like the toy model's**: GPT-2 fails to follow
  new chains hung off the answer options whatever the total depth — consistent with a shortcut that searches a few
  hops backwards from the options, and with no multi-step use of the thoughts (P2, addenda G/J). Addendum Q (3–4 hop
  training) tests training-depth dependence directly on unextended problems.

### Addendum O (scored 2026-09-29 ~10:05, `scratch/score_o.py --final`, `results/p_o.json`)

* The manipulation worked: long-scaffold runs reached full depth at median step 18 500 while the scaffold was still at
  weight 1 (until 20 000); base runs at 18 000 after it had faded. Competent 30 vs 29 of 48; P7 ≥ 0.2 in 30 % vs 21 % of
  competent runs (descriptive). Extrapolation did not change, so the exploratory association (early full depth under
  the scaffold → 8/10 extrapolate) was a fast-learner confound, not a cause. 8 of 96 runs (both arms of seeds 133, 138,
  141, 142) were stage-1 failures and count as unsuccessful. dev6 pairs (descriptive): 4 scored after the ~13:00 re-score, long 1/4, base 2/4; o_dev6_s55/s56 were dropped
  early on a wrong premise (see the correction in PROTOCOL6.md addendum O).
* **Note on L2 / M1 (2026-09-29 ~11:45, from the theory session, `docs/THEORY.md`; exploratory).** The latent thought
  acts as a pointer that should advance one depth per step. In failing K = 8 models the front *dwells* (~2 steps per
  hop), reaches depth ~4–5 by step 8 and reverts at step 9 (e.g. k8_s214, k8_s207). So L2 and M1 show that the front
  stops **where the learned step schedule ends**, and training makes that coincide with the training depth. They do
  not by themselves separate a depth wall from a schedule end. Proposed cause: *slack*. When the latent budget exceeds
  the hops needed, any schedule covering each depth once has zero loss, so the continuation past K is not identified.
  This fits O1 (a longer scaffold removes no slack) and the retrospective extrapolating share, 47 % at K = 6 vs 26 % at
  K = 8. The decisive test (anytime / minimum-progress supervision at K = 6 and 8, and on capped graphs, scored on
  distance ≥ 8) is being run and pre-registered by the theory session (THEORY.md §17).

### Addendum Q (scored 2026-09-29 ~12:25, `prosqa/score_q.py`, `prosqa/res_q_s{0,1}/`; Vast 53289329/53289703 destroyed after checksum-verified fetch)

* Both seeds competent (Q0: 3–4 hop 0.947, 0.928). Trained only on 3–4 hops, GPT-2 answers unseen 5–6 hop problems
  **as well as trained ones**: no wall at the training depth (toy models fail exactly there). Extra thoughts add nothing.
  On extended problems (m = 2, path 5–6) accuracy is 0.66 / 0.64, the same drop as the full-depth G models in addendum P.
* Together with P's depth-matched analysis: GPT-2's limit on ProsQA is not the training depth; it is the added chain
  (hops attached beyond the answer options). The toy model's training-depth horizon does not transfer to Coconut GPT-2,
  consistent with one-pass computation (G, J) rather than a learned step schedule.

### Addendum R (scored 2026-09-30 01:16, `scratch/score_r.py`, `results/p_r.json`)

- **R1 (cap 3):** all 8 competent runs above the floor have P5 < 0.10 (share 1.00 ≥ 0.30): **supported**. Not competent:
  r3_s401, s407, s410, s411.
- **R2 (dose-response):** the share whose frontier stops one level past the training depth is ≥ 0.30 at every depth:
  cap 3 1.00 (8/8), cap 4 0.67 (12/18, M pooled with R), depth 6 0.92 (12/13, L2 basis): **supported**.
- **R3 (M2 pooled):** only 2/18 cap-4 runs extrapolate by one level (needs ≥ 3 of each): **unresolved** (descriptive
  AUC 0.44). The exploratory cap-3 version is also unresolved (1/8).
- **Reading:** the frontier stops at the training depth at all three depths (3, 4, 6), and at cap 3 in every competent
  run. THEORY §35 reads this as a learned fall-back keyed to an iteration count.
- **Out-of-sample check by the theory session (THEORY §25, LE-R, registered d75d275 before R was scored):** supported.
  Median p[K−1] is 2 at cap 3 and 3 at cap 4, with the dose ordering 2 < 3 ≤ 3. The pointer's in-range reach tracks the
  trained depth, not the budget.

## Reading so far (updated 2026-09-29 ~13:20)

Whether a model's sweep stays open — and with it, mostly, whether it extrapolates — is decided during the late
curriculum, between stage 4 and the arrival at full depth, and is usually fixed once full depth is reached: later
training, new data order and a restarted optimizer rarely change it (slow drift can, in about 1 run in 8, and a
reopened sweep did not bring extrapolation with it). An open sweep remains close to necessary for extrapolation
but is not sufficient. Training precision is not the missing factor (P27 not supported): same-card fp32 and bf16
models extrapolate equally rarely once competent. Nor does it affect curriculum progress (P28 not supported). Tiny numerical differences — precision, card, a changed data order —
steer which solution a run lands on while the late curriculum is still open.

Addenda I–Q add: among open sweeps, whether the latent frontier crosses the trained horizon (P7) decides
extrapolation, prospectively and on every card and recipe (P30, L1; PROTOCOL7 T4 on 44 new runs). Failing fronts stop
where the learned step schedule ends, which training puts at the training depth (L2, M1; the theory session's
anytime test decides wall vs schedule end). PROTOCOL7 shows the limit is causal and in propagation: blocking the
frontier removes beyond-horizon answers, and supplying it at natural scale restores them. A longer lookup scaffold does
not help (O1). None of this transfers to Coconut GPT-2 on ProsQA, which computes in one pass: its thoughts carry an
answer (G, J), extra thoughts never help (P2, Q2), and it has no training-depth wall (Q1). Its limit is following
chains hung beyond the answer options (P, depth-matched).
