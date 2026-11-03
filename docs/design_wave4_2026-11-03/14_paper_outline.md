# 14. Paper outline for headline H1 / H2 (wave-4 lead design, 2026-11-03)

A paper plan, not an experiment. Read-only; every number below is copied from THEORY.md Notes, sota_arch.json or the backlog, none computed.

## 1. Goal and backlog IDs

- **H1 (mechanism paper, defensible now):** on a pretrained GPT-2 small, a latent loop with a one-token recurrent memory passes a causal battery, extrapolates 2-3 depths past training, and holds a fixed point to 16x the trained pass count; the depth horizon is set by *content selection* (which op sentence a pass reads), not by the pass budget.
- **H2 = H1 + one public depth split** (ProsQA-OOD 7/8-hop, D1; or shuffled objects, D2 / N1) with own-run same-split baselines.
- Backlog IDs: §12.8 H1 / H2 / B1 / B2 / B6; §13.0 (expressivity, statistics rules); §266 X3 / X5 / C7-P1; §268 X7; §264 X1 / X2; §269 (keys, triage); D1 PROSQA-OOD; D2 N1; E0 C3-RESCOPE.

## 2. The paper

### 2.1 Title options
1. "Content Selection, Not Depth Budget: What Limits Depth Extrapolation in a Looped Pretrained LM"
2. "A One-Token Memory Loop on GPT-2: Causal Latent Steps, Depth Extrapolation and a Trained Fixed Point"
3. "Plan-Given Latent Loops: A Causal Battery for Recurrent Reasoning in a Pretrained Language Model"

### 2.2 Abstract draft (scoped)
> Can a pretrained language model carry a multi-step computation in a latent recurrent state rather than in generated text? GPT-2 small is fine-tuned to re-read a templated word problem over K latent passes; each pass receives a step cue naming the entity to compute (the plan is given), writes one token's hidden state into a recurrent DeltaMem memory, and is supervised with the step's value (per-step supervision). At depth d <= 4 the model reaches 1.000, and causal tests (two-pass truncation, mid-chain state swaps, mean-ablation, cue-only controls) show the latent state carries each step's value. Trained to d <= 4 with distractor sentences, the loop reaches d 6 at 0.96-0.97 and d 7 at 0.85-0.86 (chance ~0.09), and a hold penalty keeps accuracy and state norm fixed to K' = 128. The extrapolation came not from a larger pass budget (three nulls) but from training the step to *select* its sentence by content instead of position. A single 12-layer pass could express these programs; the loop buys learnability and extrapolation, not expressivity. We report what the given plan pays for (without it, selection follows any read-chain and extrapolation collapses), seed spread (sigma 0.09 at d 6) and compute (2.4-9.1x the FLOPs of value-only text CoT). [H2: on regenerated ProsQA 7/8-hop structural-OOD splits, the loop reaches X vs own-run CoT-SFT / Coconut / CODI at Y.]

Scope words that must survive editing: *plan given*, *per-step supervision*, *templated mod-11*, *one lineage until §268 reads*, *K' > K does not help*.

### 2.3 Section-by-section content

1. **Introduction.** Narrow claim: causal latent steps + extrapolation + stability in one pretrained model, with what is given (plan, step targets) listed up front. Contrast: Coconut / CODI (no depth extrapolation, contested causal use); Huginn / Ouro (recurrent depth at scale, no causal battery).
2. **Setup.** REG word programs mod 11 (frames 0-5, 40 names, distractor inserts); GPT-2 small + DeltaMem (6 additive + 6 gated-delta heads read at block 6, ':' write, decoder pass); 6-stage lineage, 68k steps; K = 8.
3. **Causal battery (C1).** NOT-DECODER, SWAP4, LP5-D4, CUE-ONLY at d 4; content selection on held-out distractor families (§257); the carried value is ~3.5 bits.
4. **Depth extrapolation (C2).** Three pass-budget nulls (§239 / §240 / §243), slot counting (Note 255-4), the distractor fix (§256 / §257), the late-pass failure (Note 264-1), same-data text CoT (§266 X3).
5. **Fixed point (C6) and stopping (C5).** Hold penalty, K' 128, BUN-CTRL drift; the stability rule as an external readout; §264's plan-clock test decides the wording.
6. **What the plan buys.** §260 (plan-free: C1 on clean chains, forward-reachability selection, d 7 0.07-0.25) and §261 (target-only cue).
7. **Expressivity vs learnability.** §13.0: a 12-layer pass expresses d ~ 64; the loop's value is learnability / extrapolation. N1 (S5) is where a loop is provably needed.
8. **Compute (C3).** The §12.1 analytic table + E0 wall-clock; no efficiency claim.
9. **(H2) Public depth split.** D1 ProsQA-OOD with own-run CoT-SFT / No-CoT / Coconut / CODI, 3 seeds, zero-latent and noise arms.
10. **Related work, limitations, reproducibility** (pre-registration ledger per §, OC blocks, censoring).

### 2.4 Figures and tables, with the experiment that fills each

| # | Figure / table | Content | Experiment | Status |
|---|---|---|---|---|
| F1 | Architecture + pass schematic | cue, ':' write, read at block 6, decoder | sota_arch.json | done |
| T1 | Scorecard C1-C7 with scope column | sota_arch rows reworded per §264 / §266 | §259, §264, §266 | 259 done; 264 / 266 registered |
| F2 | Acc vs d at K' 8: bun vs one-pass vs same-data text CoT | d 1-10, Wilson CIs, both seeds | §266 X3 + bun eval_acc | registered |
| F3 | K' curve at d 3-8 | the "K' > K helps" construct; predicted flat | §264 STD K' 8-24 | registered |
| T2 | Causal battery at d 4 | NOT-DECODER, SWAP4, LP5-D4, CUE-ONLY | §259 | done |
| T2b | Causal battery at d 6-7 | SWAP-mid, LP5 at the claimed depth | new eval part, ~$0.3 | not yet |
| F4 | Held-out distractor families | H-CHAINSRC-OP, H-LIVE, H-EXTEND with step-0 bars | §257 (Note 257-3) | done on dch; bun re-score $0.1 |
| F5 | First-error pass, per-pass reliability d 5-8 | Note 264-1 X4a | §264 $0 read | done |
| F6 | Hold: acc and ||S|| to K' 128, bun vs CTRL | Note 259-4 | §259 | done |
| F7 | Cue schedules ER / LF / CY | plan-clocked or not | §264 cue_clock | registered |
| F8 | Halting: khalt vs d, fresh draw | Note 259-12 | §259 C5-STABLE | done (r = 3 post hoc; gb out-of-sample) |
| T3 | Plan-free / target-only: what the cue buys | §260 gb rows; §261 pilot | §260 / §261 | partial (§261 parked) |
| F9 | Late-pass error triage | SELECTION / MEMORY / COMPUTATION | §269 part 1 | registered |
| T4 | Seeds and lineages | per-seed margins, sigma_seed 0.090, bun_s3 | §268 | registered |
| T5 | FLOPs and serial layers per d | §12.1 table + E0 wall-clock | E0 | analytic done; wall-clock not yet |
| T6 | Pause / no-write control | zero / constant / Coconut-feedback write | B1 | not yet ($2.5) |
| T7 | Held-out structure | frames 6-7, unseen names, TEST-PERM, LATE-PREMISE | B2 + §263 | §263 running; B2 not yet |
| T8 (H2) | ProsQA-OOD vs own-run baselines, 3 seeds | test / 7h / 8h / DAG / Dense | D1 | not yet (S0 $0 / S1 $2 / S2 $25) |
| T9 (H2 alt) | Shuffled objects depth split | text / pause / Coconut arms | D2 / N1 | not yet ($3-4) |
| A1 | One-hop pipeline, C7 text-substitute | X5, P1 | §266 | registered |
| A2 | Memory keys NAME / VALUE / CONSTANT | MC1 | §269 part 2 | registered |

### 2.5 Related work (ids verified 11-03, abs page title checked)
- Latent CoT on small models: Coconut 2412.06769; CODI 2502.21074; PCCoT 2506.18582; SIM-CoT 2509.20317; SubLAT 2610.40283 (ProsQA-OOD splits).
- Recurrent depth: Huginn 2502.05171; Ouro 2510.25741; retrofit 2608.11233.
- Looped length / depth generalisation: 2409.15647; 2502.17416; 2610.14456; 2606.29983 (stochastic stopping); 2609.35643 (from-scratch latent search with causal interventions, the fullest comparable battery).
- Shortcut critiques: 2604.04902; 2604.06374; 2606.12689; 2602.22441; 2507.02199.
- Fixed-point / halting: 2608.18222; 2606.18206 (residual halting); 2611.02333; 2607.20519. The hold penalty is cited as the same idea as the terminal fixed-point objectives, not as new.
- Controls and expressivity: pause tokens 2310.02226, 2404.15758; implicit CoT 2405.14838; CoT expressivity 2305.15408, 2310.07923; supervised thinking states 2602.08332.

### 2.6 Limitations (to print, not to hide)
Plan given by the cue (§260); per-step value targets at every stage (C4 not met; §267 is the only C4-eligible route); templated mod-11, closed 40-name set, frames 0-5, chain order until §263 reads; GPT-2 small only (medium shortcuts the last hop, Note 264-1); one lineage until §268; the C5 rule is external and r = 3 post hoc (gb_s* out-of-sample); C3 and C7 not met; C1 gates at d 4 only until T2b; ~3.5 bits of carried state.

## 3. Code touch points (read)
- `boxes/inject_aw_eval.py:55-83` `passes(m, items, dev, bs, kset, seqs_of_t, Td, edit, aux_ts, on_state)`: every K' sweep or state edit (F3, F7, T2b, F9) runs through it; the PLAN-FREE assert at :65-66 underwrites T3. `part_swap_kd` (:391), `part_lp5h` (:510), `part_long` (:534) fill T2 / T2b / F6.
- `theory/score_bu.py:35-56` `seed()`: the C1 / C2 / C6 conjunction (d 1-4 >= 0.98, acc_kprime['2'] <= 0.20, follow_U >= 0.50 and >= floor + 0.30, LP5 drop >= 0.30, d 6 / 7 >= dch - 0.05, acc[128] >= acc[8] - 0.02). T1 / T2 read its JSON.
- `theory/score_sc.py:62-80` `at_d()` (ER / LFI / CY, tau = sqrt(c h)); `score()` :113-114 the C5 / C6 consequence strings. F7, T1 wording.
- `theory/score_tx.py:33-98`: `wilson()`, `near()`, X3 rule :62, X5 :74, C7 :82-98. F2, A1.
- `theory/score_ls.py:42-81`: stage gates, LINEAGE-REPLICATES / PARTIAL / BREAKS-AT. T4.
- `theory/score_c5c.py:20-35`: halt rule, C5-EXTRAP. F8.
- `docs/sota/sota_arch.json`: T1 is generated from it.

## 4. Gates (what a figure needs before it is typeset)
- Every binding number comes from a committed, fixture-passing scorer's JSON; no figure from a $0 read that was not pre-stated and acked.
- Wilson 95% CIs on every cell; NEAR (bar within 2 SE) printed, not dropped.
- T2b and the F4 re-score: eval-only, parts reused with ds 6-7, local smoke at full item scale, cloud host only.
- B1 PAUSE-PASS adds a write mode: p = 0 bitwise equivalence vs inject_aw, gradflow_check, mem_check, 200-step smoke; the zero write is a zero-init pathway, so the constant-write arm is the non-zero-init arm and the write-norm log the usage monitor.
- B2: eval-only with render asserts (answer unchanged, cue offset recomputed). D1 Stage 0 is the learnability pilot (process rule 5).

## 5. Pre-registration-ready criteria: the submit decision

The "experiment" here is whether H1 or H2 is submittable. Binding labels, each read from the named scorer, both seeds unless stated; ">= 3 seeds" rows read bun_s3 from §268.

| Row | Criterion | Threshold | Source | Binding? |
|---|---|---|---|---|
| S1 | Latent-specific extrapolation | X3 LATENT-SPECIFIC (text max d 7 <= bun7 - 0.10 and d 8 <= bun8 - 0.10), valid cells only | §266 score_tx | binding, H1 |
| S2 | Causal at claimed depth | SWAP-mid follow_U >= 0.50 and >= floor + 0.30, LP5 drop >= 0.30, at d 6, both seeds | T2b | binding, H1 |
| S3 | Pause control | zero- and constant-write arms d 6 <= bun - 0.30; Coconut-feedback reported | B1 | binding, H1 |
| S4 | Replication | LINEAGE-REPLICATES, or PARTIAL with C1 + C6; C2 rows as per-seed margin + mean +- SD over 3 seeds; within sigma_seed 0.09 of a bar = NEAR-SEED, claimed neither way | §268; Note 269-2 | binding for "replicates" |
| S5 | Plan-clock | §264 label printed in T1; C5 wording = score_sc's consequence string | §264 | wording |
| S6 | K' construct | K' 12 / 16 / 24 within 0.02 of K' 8 at d 7-8 -> "depth generalisation at fixed K" | §264 STD | wording |
| S7 | Held-out structure | frames 6-7, unseen names, TEST-PERM with CIs; a row < 0.50 goes to limitations | B2 | descriptive |
| S8 (H2) | Public split | 7h / 8h >= own-run Coconut + 0.05 in >= 2 of 3 seeds with NOT-DECODER + step swap on the OOD split; zero-latent / noise arms <= loop - 0.20; else the port is an appendix | D1 (scorer to write) | binding, H2 |
| S9 | Pipeline framing | X5 PIPELINE-SOLVES -> C1 worded "value transport under a given plan" | §266 | wording |

**Decision rule.** SUBMIT-H1 iff S1, S2, S3 pass and S4 is at least LINEAGE-PARTIAL with C1 + C6; SUBMIT-H2 iff SUBMIT-H1 and S8; else WRITE-UP-SCOPED (an arXiv note with the limitations list and no public-split claim).

**OC block (priors stated now).**
- P(S1) ~0.6 (§266 OC: NOT-LATENT-SPECIFIC 0.4). P(S2) ~0.6: SWAP4 passes at d 4, but late passes are less reliable (Note 264-1). P(S3) ~0.8; the Coconut-feedback arm is the risk. P(S4 >= PARTIAL with C1 + C6) ~0.6 (§268 OC). P(S5 PLAN-CLOCKED) ~0.7, wording only. P(S8 | D1 runs) ~0.3: 7h / 8h are chain-like; DAG / Dense need a frontier.
- P(SUBMIT-H1) ~0.3; P(SUBMIT-H2) ~0.1; P(WRITE-UP-SCOPED) ~0.6.

**Censoring.** A VOID cell reruns once under its §'s rule; a second void leaves the row "not run", never filled from an older checkpoint. UNDERTRAINED text cells (Note 266-1) never count for the latent. §268 BREAKS-AT-<stage> is printed in T4 and every "2/2" stays "one lineage". Each paper section states how many stored-output reads it used (VALIDITY_STATS confound 4).

## 6. Scorer sketch: theory/score_paper.py
- **Reads:** score_bu / sc / tx / ls / dg JSONs (labels, CIs, per-seed margins); eval_swap_kd / eval_lp5h at ds 6-7 (T2b); eval_pause.json (B1, by_d acc per arm); eval_heldout.json (B2); eval_prosqa_ood.json (D1, by_split acc per arm and seed).
- **Fields:** by_d[d].acc, acc_kprime[k], swap_mid[t].follow_U / floor, intact / mean_at_t, verdict strings; each path grepped in its scorer before fixtures are written.
- **Labels:** S1..S9 PASS / FAIL / NEAR / NOT-RUN; verdict SUBMIT-H1 / SUBMIT-H2 / WRITE-UP-SCOPED.
- **Fixtures (>= 8):** all pass -> SUBMIT-H2; S8 missing -> SUBMIT-H1; X3 NOT-LATENT-SPECIFIC -> WRITE-UP-SCOPED; §268 BREAKS-AT -> WRITE-UP-SCOPED; S2 NEAR -> printed; text UNDERTRAINED -> S1 NOT-RUN; §264 seeds disagree -> "MIXED"; D1 zero-latent arm within 0.20 -> S8 FAIL.
- **Output:** T1 and the figure-status table are generated, never hand-typed.

## 7. Minimal remaining experiment set and cost (RTX 3090 ~$0.2/h)

| Item | What | $ | Status |
|---|---|---|---|
| §264 | plan-clock + K' curve (F3, F7, S5, S6) | 0.5 | registered |
| §266 | same-data text CoT, pipeline, C7 (F2, A1, S1, S9) | 4 | registered |
| §268 | lineage seed 3 (T4, S4) | 5.6 | registered |
| §269 | triage + keys (F9, A2) | 0.8 | registered |
| T2b | causal battery at d 6-7 | 0.3 | not yet |
| F4 re-score | insert_ho on bun | 0.1 | not yet |
| B1 | pause / no-write / Coconut-feedback (S3) | 2.5 | not yet |
| B2 | held-out structure (S7) | 0.3 | not yet |
| E0 | wall-clock (T5) | 0.1 | not yet |
| **H1 total** | | **~14** | |
| D1 S0-S2 | ProsQA-OOD port, 3 seeds, own baselines (S8) | 27 | not yet |
| or D2 / N1 | shuffled objects with text / pause / Coconut arms | 3-4 | not yet |
| **H2 total** | | **~41 (ProsQA) or ~18 (N1)** | |

Only B1 (3 arms x ~$0.8) and §268 train; the rest is eval on stored checkpoints.

## 8. RED-TEAM (separate sub-agent)

*Both spawn attempts hit the concurrent-subagent limit (20 running). This is the lead's own red-team, labelled as such; an independent read should replace it before the plan is acted on.*

1. **S1 baseline is under-matched.** Text CoT starts from pretrained weights at 24k steps; bun had 68k lineage steps and an askname init. Cheaper falsifier: one text cell from s233_askname_s1 (~$1.2) before claiming S1.
2. **S2 can pass by a decoder shortcut at d 6.** A swap at t = 2 of 6 leaves hops 3-6 to the decoder. Pair T2b with NOT-DECODER at d 6 (K' 2 and K' 4 <= 0.20) and a swap at t = d - 1.
3. **S3's zero-write arm must be exposed.** Require d 1-4 >= 0.95 at K' 8 in every B1 arm before the d 6 gap counts. The Coconut-feedback arm can pass; if it does, DeltaMem adds nothing and the OC must say so.
4. **"Selection, not budget" rests on three nulls at older recipes and one positive.** Distractor training also changes the op-sentence distribution. §265's pilot (~$0.8) is the direct test; until it reads, say "distractor training, not pass budget".
5. **Seed spread swallows the C2 headline.** sigma_seed 0.090 at d 6 from two seeds of one lineage: print per-seed margins and a range, never "0.97", and wait for §268.
6. **F8 (C5) is post hoc.** r = 3 was chosen after r = 2 missed (Note 259-9); gb_s* is the out-of-sample check. Present it that way or drop C5 from the headline; PLAN-CLOCKED turns F8 into a limitation figure.
7. **The predicted-flat K' curve contradicts the NORTH_STAR C2 definition.** Define C2 as depth generalisation at fixed K on page 1.
8. **ProsQA-OOD splits are regenerated, so SubLAT's numbers are context only;** ProsQA is shortcut-prone (2604.04902, 2606.12689). Zero-latent / noise arms are binding; if D1 Stage 0 fails, switch H2 to N1 ($3-4) rather than spend $25.
9. **Cheapest falsifier of the whole paper:** §266 X3 ($4, registered). If text CoT reaches d 7 within 0.10 of bun, H1 is a negative-result note; launch nothing else in §7 before that read.
10. **Bug risk in T2b:** `swap_pairs_d(a, d, t)` pairs by program signature and the d 6-7 pools are smaller than d 4; count distinct signatures before trusting 1,000 pairs.

## 9. Verified arXiv ids
Fetched https://arxiv.org/abs/<id> on 2026-11-03; titles match the roles in §2.5: 2412.06769, 2502.21074, 2506.18582, 2509.20317, 2502.05171, 2510.25741, 2409.15647, 2502.17416, 2604.04902, 2604.06374, 2606.12689, 2602.22441, 2608.18222, 2606.18206, 2606.29983, 2611.02333, 2607.20519, 2609.35643, 2602.08332, 2507.02199, 2310.02226, 2404.15758, 2608.11233, 2610.14456, 2405.14838, 2305.15408, 2310.07923, 2610.40283. Not fetched (backlog-only, keep out of the paper until Literature checks them): 2607.06648, 2602.08783 (B6 faithfulness).
