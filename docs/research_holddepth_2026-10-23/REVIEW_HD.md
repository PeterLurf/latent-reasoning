# REVIEW_HD: red team of HD_draft v0 (ee9537b9) after Stage 0 (Notes 224-10 / 224-11)

Red team, 2026-10-23. Read-only on the repo; nothing committed. New $0 checks (CPU, 2 threads, Idle priority) are in `review_scripts/`:
- `rv_logs.py` → `rv_logs.out.json`: far-loss trajectories and dose integrals from the committed `log.jsonl`.
- `rv_par.py` / `rv_tans.py` → `rv_par.out.json` / `rv_tans.out.json`: a null-calibrated PAR test, how items finish ahead, and t_ans histograms, from the `hd_check_cpu.*.json` files.
- `rv_check_cpu_ft.py`: an unchanged copy of `hd_check_cpu.py`, with only the root and output folder changed. It was run on 4 FT cells (s224_rdft_s1, rdft_s2, ftctrl_s2, fhft_s2) → `review_scripts/hd_check_cpu.<run>.json`. The FT-CTRL s1 model is not on disk, since `s224_ftctrl_s1_re` was an eval-only re-run.
- `rv_oc.py` → `rv_oc.out.json`: an independent OC computation.

No blinded file was read (no s130 TEST replays, no s162 outputs, nothing from the sealed n 25–30 set of 216-5).

## Verdict: REVISE with a different lever. G-FAR (and DELAY) should not get spend.

Replace the lever with **one post-formation anchored far fine-tune at a fixed λ (FH-FT-FIX)**: ≈ $2.3, cap $4, with a pre-stated stop. If Audit declines that spend, **STOP** and accept RD's hold, with the depth caveat re-scoped as in #2. That is safe for C1: RD passes the in-range C1 floor (NOLOSS).

Why:
- **G-FAR's specific premise has no supporting signature left.** REPAIR_SIG is absent in both seeds.
- **The logs show what G-FAR would actually do: almost nothing during formation.** RD's far CE equals its in-range answer error in every bin from 6k on. So G-FAR is roughly the plain recipe plus a late hold term, which makes it an expensive, scratch-trained version of the FT route. Its HOLD then carries a real D6 risk.
- **The existing data already contain a model with both properties.** FH-FT s2 has H ≥ 0.99 at every K′ to 128, DEPTH +0.049 vs FT-CTRL, and its ahead-of-schedule ability intact (new check below).

## Issues, ranked

### 1. CRITICAL: G-FAR's premise lost its evidence, and G-FAR barely differs from the plain recipe during formation

- **The far loss during formation is just the in-range answer error re-applied at T_far.** far_ce vs answer_ce (2k bins, `rv_logs.out.json`):
  - s1 at 6–8k: 0.336 vs 0.324; at 8–10k: 0.275 vs 0.276.
  - From 6k on, |far_ce − answer_ce| ≤ 0.012 in all three RD seeds.
  - The only hold-specific excess is in 0–4k (+0.46 / +0.38 in s1), when the T−1 gate share was 0.10–0.60.
- **So L_rep ≈ L_far and L_hold ≈ 0 from about 4k on.** G-FAR therefore removes almost all of RD's formation-time far gradient. A G-FAR pass would read "no far gradient during formation" and could not separate repair from the WINDOW mechanism.
- **The repair-pathway version (D2) is not supported.** d PA_nS is +0.04 / −0.04 (224-11).
  - What survives is a weaker "unsatisfiable-target gradient conflict" story, which has no $0 signature.
  - The FT contrast (repair dose 6–10 → no cost) is confounded with lr, dose and timing, as §1.3 itself says.
- **Literature (addendum to LIT_FIXEDHOLD_NOVELTY_2026-10-21):** Bansal 2202.05826 trains the progressive loss on erroneous prefixes with no gate and reports extrapolation GAINS (Anil also reports gains).
  - So "error-prefix far loss harms extrapolation" has no prior support. G-FAR is the opposite choice to Bansal.
  - N2 (correctness gate) and N3 (timing) are NARROW (STaR / RFT / loss truncation / self-paced; Achille 1711.08856 critical periods).
- **Fix:** withdraw G-FAR as the lever.
  - If it is kept anyway: restate D2 as a conflict hypothesis without a mechanism signature, cut the H-REPAIR prior from 0.35 to ≤ 0.15, and state that G-FAR ≈ DELAY in effect (see #6).

### 2. CRITICAL: DEPTH at n 9–11 mostly measures off-schedule compression at a pass horizon that every model shares. PAR is that compression.

**Measured, from the hd_check_cpu files** (`rv_tans.out.json`, `rv_par.out.json`):

- **No model finishes an item at pass ≥ 11.** Of the n = 11 items that are right at K′ = n + 2, the number with t_ans ≥ 11:

| model | base s1 | base s2 | RD s1 | RD s2 | RD s3 | FH s1 | FH s2 | FT cells |
|---|---|---|---|---|---|---|---|---|
| t_ans ≥ 11 / right | 0 / 26 | 1 / 35 | 2 / 24 | 1 / 20 | 2 / 18 | 7 / 23 | 3 / 18 | ≤ 3 |

  - Right answers at n 9–11 are produced by pass 9–10.
  - **More passes past about 10 do not help in any model.** That is a C2 failure, common to base and RD.
- **At n = 10, base's right items finish at pass 9, one pass early:** 57 / 78 (s1) and 76 / 95 (s2). RD: 20 / 52 and 40 / 57.
  - 57–64 % of base's ahead items have the **state itself one reduction ahead** at n − 1 (`ahead[n−1]`). That is a pass doing more than one reduction, which R2 penalises in range.
  - The coda finishing the last op accounts for only ≈ 18–19 % (`exact[n−1]`).
- **RD's lost ahead items end finished with a wrong value.** Arm finished 0.73–0.89, arm exact[n] 0.02–0.13, while base exact[n] is 0.95–1.0. This is a value error in the off-schedule reductions, not idling. It agrees with "no SETTLE".
- **Not all of the cost is compression.** RD s2 / s3 also lose at n = 9 (acc 0.81 / 0.73 vs 1.00), where no compression is needed. Only s1's cost is ≈ all at n = 10 (Δ by n: +0.02 / −0.20 / −0.02).

**Implications:**
- §224's DEPTH-WARN treated DEPTH as a C1 proxy. C1 (NORTH_STAR: K′ 0/1 blocking, serial use in range) is passed by RD.
- What RD loses is extrapolation through a compression shortcut, closer to C2 (which fails everywhere anyway), plus higher seed variance (C6: RD depth 0.52 / 0.51 / 0.37).

**Fix:**
- Re-word the 224-9 caveat: "RD reduces n 9–11 extrapolation, which in all models relies on finishing within ≈ pass 9–10 by multi-reduction passes; no model computes past ≈ pass 10."
- A future depth / C2 lever should target the pass horizon, not the far pass. This is a separate design.

### 3. HIGH: PAR_SIG's 0.60 line is not calibrated against a null. PAR is real in 3 / 4 cells and marginal in RD s2.

- **The null PAR share is P(ahead | base right): 0.61 (s1) and 0.55 (s2).** RD s1 at 0.79 and RD s2 at 0.63 are compared with a line that s1 almost meets under the null.
  - d_ahead ≤ −0.05 is also partly mechanical: losing base-right items at the base ahead rate gives ≈ −0.04 / −0.12.
- **Proper test: P(arm right | base ahead) vs P(arm right | base right, not ahead), exact 2 × 2:**

| cell | P(right \| ahead) | P(right \| not ahead) | p |
|---|---|---|---|
| RD s1 | 0.68 | 0.86 | 0.002 |
| RD s2 | 0.57 | 0.69 | 0.054 |
| FH s1 | 0.61 | 0.87 | 2e-5 |
| FH s2 | 0.48 | 0.79 | < 1e-5 |
| RD-FT s1 | 0.86 | 0.94 | 0.075 |
| RD-FT s2 | 0.82 | 0.80 | 0.75 |
| FT-CTRL s2 | 0.89 | 0.85 | 0.27 |
| **FH-FT s2** | 0.91 | 0.91 | 1.0 |

- **Ahead share:** base 0.35 / 0.37, RD 0.19 / 0.24, FT-CTRL s2 0.43, RD-FT 0.29 / 0.38, **FH-FT s2 0.41**.
- **So the post-formation far loss leaves compression intact, even in a cell that holds 0.99 at K′ 128 (FH-FT s2).** The formation-time far loss removes it.
  - Hold and compression are therefore not intrinsically in conflict, and the trade is not forced.
- **Fix:** report PAR with this test in the Notes. The 224-11 phrase "PAR_SIG in 4 / 4" should read "3 / 4 at p < 0.01; RD s2 p 0.054".

### 4. HIGH (D6): G-FAR has a material HOLD risk, about 0.4–0.6 that both seeds pass, not 0.85

- **RD's hold-specific signal was concentrated where G-FAR keeps the least.** The far − answer CE excess was large only in 0–4k (#1), when 40–90 % of far items were wrong at T−1. G-FAR keeps only the gated part of that.
- **After formation G-FAR is a drift-restoring loss, which is the FT regime.** That regime gave H(96) 0.93 / 0.92 and H(128) 0.66 / 0.74 for RD-FT.
- **More lr-integral may not close the gap.** G-FAR's lr-integral after 16k is 0.93 vs RD-FT's 0.18 (5×). But RD-FT's far CE had already converged (last 1k ≈ 1e-4), and its hold still did not extrapolate past K′ 48. Dose inside the trained K′ ≤ 48 range is not what limits K′ 64–128.
- **FH-FT shows the anchor is what extrapolates after formation.** FH-FT beats RD-FT at K′ 128 by +0.15 / +0.25 (224-4); RD alone suffices only from scratch.
- **Fix:** if G-FAR survives, add a $0 local D6 pre-gate before any cloud spend.
  - RD vs G-FAR to 8k, on one local card, same seed.
  - khorizon at K′ 64 / 128 on n ≤ 5.
  - G-FAR H(128) < RD − 0.10 → stop.
  - Use P(HOLD both) ≈ 0.5 in the OC.

### 5. HIGH: the OC's σ_Δ = 0.03 comes from FT pairs. Scratch-trained arms are noisier.

- **The 5 pairs (rms 0.020) are FT or near-identical runs from a shared 48k init.**
  - A scratch arm trained against a same-seed base diverges chaotically (the GPU-class / basin lesson), so seed pairing buys little.
  - Base s1 vs s2 differ by 0.069 → σ_run ≈ 0.05 and σ_Δ ≈ 0.05–0.07. Eval binomial SE of the difference is ≈ 0.011.
- **Recomputed (`rv_oc.out.json`; bar Δ ≥ −0.03; no effect = RD's mean −0.115, s1-like −0.076):**

| rule | full, σ .03 | full, σ .05 | full, σ .07 | no effect (s1-like), σ .05 |
|---|---|---|---|---|
| both of 2 (draft) | 0.71 | 0.53 | 0.44 | 0.03 |
| mean of 2 ≥ −0.03, each ≥ −0.08 | 0.91 | 0.72 | 0.56 | 0.03 |
| 2 of 3 seeds | 0.93 | 0.82 | 0.74 | 0.08 |

- **P(HD-LEVER | full effect)** = 0.43 (σ .05, HOLD 0.85) or 0.26 (HOLD 0.5). The draft's 0.58 reproduces only at σ .03.
- **With the mechanism prior revised for #1 (H-REPAIR ≤ 0.15), P(HD-LEVER) ≈ 0.05–0.08** for ≈ $14 expected.
- **Fix, for any scratch lever:**
  - use the 2-seed-mean rule with a −0.08 per-seed guard;
  - **or** 3 seeds with same-seed bases (2 of 3). base_s3 + arm_s3 ≈ $5.7.
  - The 2-seed "both" rule is underpowered at realistic σ.
  - For the FT-route lever below, σ_Δ ≈ 0.02 is the right value, because those pairs are FT pairs.

### 6. MEDIUM: DELAY's cosine-lr confound, and its low information

- **DELAY's far term starts at lr 0.75e-4.** The lr-integral after 16k is 0.93, 39 % of RD's 2.40, and the gradient arrives when the model has largely converged.
  - So a DELAY HOLD fail confounds "formation needed" with "less dose / lower lr".
  - A DELAY DEPTH pass mostly re-tests what the FT cells already showed (2 / 2 with no cost).
- **G-FAR ≈ DELAY in effect** (#1): both are about plain during formation, with a far signal after. So the 2 × 2's off-diagonal labels (REPAIR-LATE / WINDOW) are improbable, and the table is mostly diagonal noise.
- **Fix:** drop DELAY. If it is kept: make its HOLD descriptive only, and report each arm's far lr-integral beside the label.

### 7. MEDIUM: cost, censoring, base rule, Goodhart, CurGate (mostly sound)

- **Cost:** the $3.1 per cell and $12.4 core estimates are consistent with §224 (RD-s3 realised $1.61). The problem is value: ≈ $14 expected for P ≈ 0.05–0.08 (#5).
- **Censoring:** sound. Fixed-K′ readings use all items, and never-reached events are censored at 48k and reported as "> 48k".
  - Add: an H(K′) or DEPTH cell that dies after its last guard and cannot resume is VOID, never FAIL. The draft says this; keep it in the scorer fixtures.
- **Same-seed base rule:** sound as a comparator rule. s3 bringing its own base_s3 is correct, but it gives no variance reduction for scratch arms (#5).
- **Goodhart:**
  - G1 is correct: an argmax indicator under no_grad has zero gradient a.e.
  - G2 is reachable: RD's T−1 gate was 0.977 / 0.986 / 0.983 over 14–16k, and the n + 2 gate should be ≥ that.
  - Residual risk: none via gradients. The UNEXPOSED line just needs G-FAR's own trajectory.
- **CurGate:** it reads only the plain-batch answer_ce (vbank.py ≈ l.907), with `cur_gate 0.3:2:8000`. RD advanced by threshold at 6k / 9k.
  - Any lever that changes formation speed can shift those advances. The draft's 2k CONFOUNDED flag covers this.
  - far_ce ≈ answer_ce means the far term does not touch CurGate's input directly.

### 8. LOW: the Stage-0 MIXED label is seed noise

- dU flips sign (+0.23 / −0.13), and the base U_share itself differs 0.17 vs 0.41 between seeds. So U_share is dominated by seed.
- Keep "no SETTLE" (d_idle_pre ≈ 0). Do not read MIXED as a mechanism.

## Replacement lever (Q1): FH-FT-FIX, a post-formation anchored far fine-tune at a fixed λ

- **What PAR points to.** The cost appears when the far loss is present during formation, and is absent when it is applied after formation (RD-FT and FH-FT keep compression: #3).
  - FH-FT s2 is the one §224 cell that passes HOLD (H ≥ 0.99 at every K′ to 128) and DEPTH (+0.049 vs FT-CTRL; +0.019 vs base).
  - FH-FT s1 held only 0.82 at K′ 128 with λ = 0.035: the dose rule gave a 46× lower λ than s2's 1.618.
- **Single change vs registered FH-FT:** λ fixed at 1.6 (dose rule off). Everything else as FH-FT: 12k from D3-T s1 / s2, lr 3e-5, one 3090 class.
  - For s1 this tests the λ explanation. For s2 it is a near-replication.
  - **No ablation** is needed: FT-CTRL s1 / s2 and RD-FT s1 / s2 already exist as comparators.
- **Readings:**
  - HOLD H ≥ 0.95 at K′ 24–128;
  - DEPTH vs same-seed FT-CTRL ≥ −0.03 (FT-pair σ ≈ 0.02 → full-effect power 0.87 both-of-2);
  - NOLOSS as in §224;
  - descriptive: Stage-0 compression (ahead share, P(keep | ahead)).
- **Cost and stop rule:** ≈ $1.15 per cell → ≈ $2.3, cap $4. A FAIL closes the HD line and RD's hold is accepted with the #2 wording.
- **Why not the alternatives:**
  - **An idle-only adapter** is an architecture change, more than one change.
  - **A lower far weight** (Bansal's α ∈ {1, 0.01}, with gains in both) is a valid single change, but from scratch it still acts during formation and cuts the hold dose. Under #4 its HOLD risk is at least G-FAR's.
  - **"Never before the item's own budget"** is already true (T_far > n + 2).

## Q6: stop or move on?

- G-FAR as drafted: **STOP**. Prior P ≈ 0.05–0.08, ≈ $14, and the mechanism reading cannot be decided (#1, #6).
- The trade itself is narrower than "C1 > C6". RD costs n 9–11 compression-extrapolation, not in-range C1.
- Accepting RD now is defensible. FH-FT-FIX is the one cheap step worth taking first, if Audit wants a hold recipe with no depth caveat.
