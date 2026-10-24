# HOLD-DEPTH (HD): why the far pass costs depth, and a gated far pass (draft v0, 2026-10-23 22:11 EDT)

Theory draft, not registered. Follows Note 224-9 (§224 CLOSED: RD fixes the C6 hold, but costs depth). No § number, no spend, nothing launched.
- Status: **DRAFT, for Audit + red team + Literature.**
- Addresses Audit's three additions: (1) why the cost grows s1 → s3 (§1.4); (2) a $0 frozen-model probe that separates capacity from early-halt / settle (§2); (3) DEPTH against the same-seed base (§4).
- $0 files in this folder (CPU, ≤ 2 threads, Idle priority, no GPU):
  - `hd_check_json.py` → `hd_check_json.out.json` (check A: committed eval_vb / eval_khorizon / log / args only; no model loaded). **Run; §1 quotes it.**
  - `hd_check_cpu.py <run>` → `hd_check_cpu.<run>.json` (check B = Stage 0: CPU rollouts of model.pt on n 9–11 test items). **Only `--validate` was run** (K1, K2 on 7 models; no n ≥ 9 item read). The deep read waits for §2's predictions to be committed.
  - `hd_stage0_score.py` → `hd_stage0.out.json` (Stage-0 labels; fixtures 4/4).

## 1. What the committed data already say (check A; descriptive, seen before any prediction here)

### 1.1 The cost, per same-seed pair

DEPTH = mean `acc_n2` n 9–11 (K′ = n + 2). Training n ≤ 8 (cur 3,5,8), so n 9–11 is length extrapolation. In-range `acc_n2` n 1–8 ≥ 0.999 in every cell.

| cell | DEPTH | Δ vs same-seed base | H(128) |
|---|---|---|---|
| base s1 / s2 (`s224_base_s*_re`) | 0.597 / 0.666 | – | 0.107 / 0.134 |
| RD s1 / s2 | 0.521 / 0.512 | −0.076 / −0.153 | 1.000 / 1.000 |
| RD s3 | 0.373 | no base (−0.224 vs base s1, −0.293 vs base s2) | 1.000 |
| FH s1 / s2 | 0.476 / 0.443 | −0.122 / −0.222 | 1.000 / 1.000 |
| FT-CTRL s1 / s2 (12k FT from D3-T, lr 3e-5) | 0.593 / 0.636 | – | 0.116 / 0.096 |
| RD-FT s1 / s2 | 0.600 / 0.603 | +0.007 / −0.033 vs FT-CTRL | 0.664 / 0.740 |
| FH-FT s1 / s2 | 0.615 / 0.685 | +0.022 / +0.049 vs FT-CTRL | 0.818 / 0.990 |

- K4: the deltas reproduce Notes 224-7 / 224-9 (s3 vs the mean base: −0.259). [PROVEN, recomputed]
- Most of the loss is at n 10 (base 0.603 / 0.718, RD 0.406 / 0.477 / 0.282) and, in RD s2 / s3, n 9. n 11 is near floor in every cell (0.16–0.29).

### 1.2 Where the answer is decided

- **The answer at n 9–11 is fixed by pass n.** `acc_k` at K′ = n, n + 1, n + 2, 16, 24 agree within 0.016 in every cell. Base and RD alike: no late arrival, no late loss (`late_gain_24_minus_n2` −0.013 … +0.001). [measured]
- **RD answers early LESS, not more.**
  - early-finish (mean_{n 2..8} `acc_k[n−1][n]`): base 0.204 / 0.174; RD 0.156 / 0.169 / 0.155; FH 0.141 / 0.126.
  - n 10 at K′ = n − 1: base 0.538 / 0.661; RD 0.268 / 0.393 / 0.241.
  - Much of the base's extra depth at n 10 is already there one pass early: base items finish "ahead of the serial schedule".
- **The state itself carries less of the answer.** PROBE(12) (fresh readout of the frozen V_12 + bank, trained n 3–8), mean n 9–11: base 0.585 / 0.667; RD 0.505 / 0.520 / 0.375; FH 0.468 / 0.423. It tracks DEPTH, so the coda is not the bottleneck. [measured]
- **Slot-level state tracking along the serial schedule is nearly identical.** `r_acc_by_t[t][n]` at t 9–n differs ≤ 0.01 (n 11, t 10: base 0.918 / 0.917, RD 0.914 / 0.910 / 0.913). At t = n: base 0.987 / 0.988, RD 0.982 / 0.979 / 0.977. [measured]
- Per-step value accuracy given an exact previous state (`stepcap.by_t_a` pooled over a), t 8 / 9: base 0.858 / 0.940 (s1), 0.881 / 0.982 (s2); RD 0.79–0.81 / 0.82–0.86. But FH s1 is 0.864 / 0.950 with a −0.12 cost. It is pooled over n ≥ t (composition differs by cell), so it is not decisive. [measured]
- BANK-ABLATE (versions 1..t−2 masked) at n 9–11: base 0.515 / 0.580, RD 0.333 / 0.350 / 0.354. RD's deep items lean more on older versions. Consistent with §3's mechanism, but not specific to it. [measured]

### 1.3 Candidate mechanisms against A

- **(b) distribution / "answer early and hold": refuted as stated.**
  - Every far item has n_i + 2 < T_far (`far_branch` asserts it; T_far ≥ 13, n ≤ 8). The far CE never lands on a computing pass of a training item. [PROVEN, code]
  - RD answers early less (above), and does not freeze earlier. [measured]
  - A pass-gated variant (computing switched off at passes ≥ 9, harder in RD) is not refuted by A. That is SETTLE in Stage 0.
- **(d) curriculum: refuted.** CurGate advances at 6,000 / 9,000 in all 7 trained cells and both bases. Basin entry (first 500-step mean `answer_ce` < 0.5) is base 3,500 / 3,300, RD 3,600 / 3,600 / 3,800, FH 3,800 / 3,500: within +300 of the same-seed base, far under the 5k CONFOUNDED line. [measured]
- **(c) optimisation: a real coupling, but it is timed.**
  - Grad clip at 1.0, with `clip_frac_100` ≈ 1 until ≈ 14k in every run.
  - RD's `gn_preclip` over steps 0–1k is 52–68 vs base 23–25. So during formation the far term takes a large share of each fixed-norm update. [measured + DERIVED]
  - No per-term grad share is logged for RD, and the logged `fh_gnorm_*` exist only under --fh, so (c) cannot be correlated per seed.
- **(a) steady-state interference: weak a priori.**
  - Late in training `far_ce` is 1e-6 – 7e-4 (final 1k bin). The CE gradient on the logits is (p − y), of order the CE, so the far term's late gradient is ≈ 0. [DERIVED]
  - Whatever costs depth acts mainly while L_far is large: steps ≲ 16k, and right after each advance.
- **(e) NEW, formation-time repair (H-REPAIR): best supported.**
  - `far_ce` tracks `answer_ce` 1k-bin by 1k-bin through formation (RD s1, 9–10k: 0.453 vs 0.456).
  - The far gate share (`fh_gate_share` = answer right at T_far − 1) is 0.10 → 0.79 over the first 4k and drops to ≈ 0.85 after each advance. So most of the far loss during formation sits on items whose no-grad prefix computed the WRONG state. [measured]
  - Repair dose (Σ over records of (1 − gate share) × 100 steps): RD 3,380 / 3,240 / 3,518, FH 3,587 / 3,365, versus **6–10 in the four FT cells**, whose gate share is ≥ 0.994 from the first 1k steps. **The FT cells show no depth cost.** [measured]
  - FT is confounded with timing and dose (lr 3e-5, 12k, after formation), so A cannot separate "repair items" from "any far gradient during formation". §3–§4 separate them.
- **(f) PAR (ahead-of-schedule loss), NEW, descriptive:** the base's n 10 advantage is largely items already right at K′ = n − 1. Stage 0 tests whether RD's deficit items are those (PAR-SIG).

### 1.4 Audit (1): why does the cost grow s1 → s3?

| seed | cost | entry delay vs same-seed base | r_log | repair dose | ∫far_ce 0–12k | gn 0–1k | answer_ce 2–3k |
|---|---|---|---|---|---|---|---|
| RD s1 | −0.076 | +100 | 1.04e-2 | 3,380 | 9,654 | 60.4 | 1.163 |
| RD s2 | −0.153 | +300 | 1.49e-2 | 3,240 | 9,353 | 52.1 | 1.152 |
| RD s3 | −0.224 / −0.293 (no base) | n/a (3,800 abs.) | 6.5e-3 | 3,518 | 9,892 | 67.8 | 1.392 |
| FH s1 | −0.122 | +300 | 3.1e-3 | 3,587 | 9,974 | 62.0 | 1.277 |
| FH s2 | −0.222 | +200 | 6.0e-3 | 3,365 | 9,474 | 57.2 | 1.125 |

- **r_log does not track the cost.** s3 has the smallest r_log and the largest cost; across the four same-seed pairs the order is not monotone. [measured]
- **Basin entry does not track it either.** Delays are +100 to +300 in all pairs, with FH reversing RD's seed order. [measured]
- **Nothing logged orders the four same-seed pairs consistently** (repair dose, ∫far_ce, early gn, early answer_ce).
  - RD s3 is the extreme on dose, early gn and slow early learning, but it has no same-seed base.
  - The base seed spread is 0.07 (0.597 vs 0.666), so up to ≈ 0.07 of s3's apparent extra cost may be seed, not recipe.
- **Reading:** with 2 same-seed pairs per arm, no correlation is readable. The s1 → s3 growth is not explained by any logged quantity. §4 removes the mean-base comparator, and base_s3 is in the contingent budget.

## 2. Stage 0 ($0, frozen models): settle vs capacity, with predictions (Audit (2))

**Run after this section is committed, before any spend:** `hd_check_cpu.py` (default --per 128: n 9, 10, 11 × 128 items, K′ to 24, coda every pass) on base s1 / s2 (s213_d3t_s1, s216_d3t_s2), RD s1 / s2 / s3 and FH s1 / s2. Then run `hd_stage0_score.py`. Estimated ≈ 8 min per model on 2 threads (validate run: 50 s / model).

### 2.1 Readings (field names as in hd_check_cpu / hd_stage0_score)

- `U_share`: of the items wrong at K′ = n + 2, the share whose final state R(V_{n+2}) is **unfinished** (`alive_fin` ≠ 1 live slot).
- `act_late` / `act_early`: state activity, mean 1 − cos(V_t, V_{t−1}) over real slots, on the computing passes beyond trained n (t 9..n) and in range (t 2..7). Read as the arm / base ratio.
- `idle_pre`: idle-classifier score at t 8..n − 1. A per-model ridge on the real-slot mean of V_t, fit on n 3–8 (t ≤ n − 1 → 0, t ≥ n + 2 → 1). Held-out accuracy 1.000 in all 7 models (K2).
- `PA_S` / `PA_nS`: P(answer right | S) and P(answer right | not S), where S = finished final state holding the gold value.
- `ahead`: share with t_ans ≤ n − 1. **PAR share:** of the items right in base and wrong in the arm (same item), the share with base t_ans ≤ n − 1.

### 2.2 Predictions per mechanism (written before any deep read)

| mechanism | U_share (arm − base) | act_late ratio | idle_pre (arm − base) | other |
|---|---|---|---|---|
| SETTLE (pass-gated idling; (a)-damping / (b)-variant) | ≥ +0.15 (errors are unfinished) | ≤ 0.70 | ≥ +0.15 | act_early ratio ≈ 1 |
| VALUE (capacity / repair / formation shaping) | within ±0.15 | 0.70–1.43 | within ±0.15 | errors are finished with a wrong value |
| H-REPAIR, supporting | – | – | – | PA_nS(RD) − PA_nS(base) ≥ +0.05 (answers from wrong states) |
| PAR, supporting | – | – | – | PAR share ≥ 0.60 and ahead(RD) − ahead(base) ≤ −0.05 |

- Theory's prior from A [ASSUMED]: VALUE 0.6, SETTLE 0.15, MIXED 0.25.
- r_acc_by_t already shows nearly equal slot tracking at t 9–n (§1.2), which argues against strong SETTLE.

### 2.3 Labels and the lever gate (hd_stage0_score)

- Labels per same-seed pair:
  - **SETTLE** if dU ≥ 0.15 and (act_late ratio ≤ 0.70 or d idle_pre ≥ 0.15);
  - **VALUE** if |dU| < 0.15, 0.70 < ratio < 1.43 and |d idle_pre| < 0.15;
  - else MIXED.
- The reading is the RD label if RD s1 and RD s2 agree, else MIXED. RD s3 is shown against both bases and never labelled. FH is descriptive.
- **Gate:**
  - **SETTLE → G-FAR is withdrawn** (its premise is a cost set in the computing weights by repair gradients, not a moved compute / idle boundary). The draft returns to Theory with the SETTLE-matched lever (an idle-only adapter), with no spend under this design.
  - VALUE or MIXED → §3 goes to review.
- Sampling [DERIVED]: ≈ 150–190 wrong items per model, so the SE of dU is ≈ 0.05 and the 0.15 line is ≈ 3 SE.

## 3. Lever: G-FAR, the far CE gated by correctness at the trained read

### 3.1 Change (one flag over RD; default off)

- `--far_gate n2`: in `far_branch`, also read the coda at each item's n_i + 2, inside the no-grad prefix (grad_from = T_far > n_i + 2). Set g_i = 1[argmax over NUM_IDS of that answer == y_i].
- **L_far^G = (1/q) Σ_i g_i · CE_i(T_far), q = far_n = 16** (fixed denominator).
- Everything else is RD: T_far ~ U{13..48} from rng_far, the first 16 plain items, weight 1, the D3-T args, 48k on one 3090 class.
- New log fields: `far_gate2_share`, `far_ce_g2` (CE over g = 1), `far_ce_ng2` (CE over g = 0).

### 3.2 Derivation

- **D1 [PROVEN, code]:** ∇L_far reaches only pass T_far's core weights and the coda. Passes 1..T_far − 1 and the bank run with autograd off (`rollout(grad_from = T_far)`).
- **D2 [DERIVED]: the gate splits the far loss exactly.** Split L_far = L_hold + L_rep with L_hold = (1/q) Σ g_i CE_i and L_rep = (1/q) Σ (1 − g_i) CE_i.
  - For a g_i = 0 item, V_{T−1} holds a wrong reduced expression (it was already wrong at the trained read n_i + 2).
  - So ∇L_rep asks the shared core (one pass) plus the coda to output y_i from a wrong state. It can only do so with information outside the algorithmic state: older bank versions, V_0 or the prelude.
  - That is a gradient toward a non-serial repair pathway in the weights every computing pass uses.
- **D3 [DERIVED]:** the softmax-CE logit gradient has norm ∝ (1 − p_i(y)). Repair items (p small) give near-maximal gradient, settled hold items (p ≈ 1) give almost none. Hence ∇L_rep dominates ∇L_far during formation: gate share 0.10–0.79 over the first 4k, ≈ 0.85 after each advance (§1.3).
- **D4 [PROVEN, algebra]:** with the fixed denominator q, ∇L_far^G = ∇L_far − ∇L_rep, item by item. The hold part of RD is kept unchanged. (A Σg denominator would scale up the few early gated items: rejected.)
- **D5: the predictions follow.**
  - If the depth cost is a functional of the cumulative ∇L_rep (H-REPAIR), G-FAR removes it.
  - If it comes from the far gradient on correct items during formation (WINDOW) or at any time (STEADY), G-FAR keeps it.
- **D6 [ASSUMED, the main risk]:** RD's full hold to K′ 128 comes from L_hold, not L_rep.
  - Against: RD-FT, whose far loss was ≈ all hold (repair dose 6–10), reached only H(128) 0.664 / 0.740 in 12k at lr 3e-5.
  - For: G-FAR keeps the hold signal for all 48k at the full lr, ≈ 5× FT's lr-integral.
  - G-FAR's HOLD reading tests D6 directly.

### 3.3 Goodhart check

- **G1 [PROVEN]:** the gate is an argmax indicator computed under no_grad. It is piecewise constant in θ, so it contributes zero gradient almost everywhere. A gradient-based optimiser therefore gets no push toward failing the gate (being wrong at n + 2 to escape L_far).
- **G2:** n + 2 correctness is the main recipe's own trained read. L_ans trains it on every item.
  - Reading `far_gate2_share` should track the main batch's acc at n + 2 and reach ≥ 0.95 by 16k.
  - If it does not, report **UNEXPOSED** (no lever reading), never a pass.
- **G3:** the verdict never reads the gate. HOLD and DEPTH are fixed-K′ answer accuracies on the held-out §170 test set.
- **G4:** gating removes only items the model already got wrong in range. A model "holding only what it got right" is exactly the C6 target: H is computed on intact answers.

### 3.4 One ablation: DELAY (ungated RD, far branch on from step 16,000)

- `--far_on 16000`. T_far is still drawn every step, so post-onset draws equal RD's. Steps < 16k are the plain recipe.
- Why 16k: the first 2k bin after which base `clip_frac_100` stays < 0.9 (s1 0.707 at 16–18k), 7k after the last advance.
- With G-FAR it forms a 2 × 2 that reads which part of the far gradient costs depth:

| G-FAR DEPTH | DELAY DEPTH | label (descriptive) | meaning |
|---|---|---|---|
| pass | pass | FORMATION | the cost is set by the far gradient during formation; repair vs window not separated |
| pass | fail | REPAIR-LATE | ungated far CE costs depth even after formation; gating avoids it |
| fail | pass | WINDOW | any far gradient during formation costs depth, hold items included |
| fail | fail | STEADY | far CE on shared weights costs depth at any time → architecture separation (idle adapter); this line closes |

- DELAY's HOLD reading also tests D6: DELAY HOLD fails while G-FAR HOLD passes → the formation-time hold signal is needed.

## 4. Registered readings (proposed)

- **Cells:** G-FAR s1 / s2 and DELAY s1 / s2: D3-T args (seed 1 = s213_d3t_s1, seed 2 = s216_d3t_s2), 48k, one RTX 3090 class. Run dirs s2xx_gfar_s{1,2}, s2xx_delay_s{1,2}.
- **Comparator: the SAME-SEED base re-eval only** (s224_base_s1_re / s224_base_s2_re, 3090, fp32, bs 32). No mean-base comparator anywhere (Audit (3)).
- **A seed without a base is excluded** from every binding reading. The contingent s3 (below) launches with its own plain base_s3 (D3-T seed 3), so RD-s3 also gets a same-seed comparator (descriptive for §224).
- **Per (arm, seed):**
  - **HOLD:** H(K′) = mean n 1–8 `acc.intact[K′]` ≥ 0.95 at every K′ ∈ {24, 32, 48, 64, 96, 128}.
  - **DEPTH:** mean `acc_n2` n 9–11 ≥ same-seed base − 0.03.
  - **NOLOSS** (as score_s224_s2, same-seed base): `acc_n2` n 1–8 ≥ 0.99; stall ≥ base − 0.02; PROBE(4) n 7–8 ≤ base + 0.05; early-finish ≤ base + 0.05; C1 floor `acc_k['0']` n 1–16 and `acc_k['1']` n 2–16 ≤ base + 0.05.
- **Verdicts:**
  - **HD-LEVER (G-FAR):** HOLD & DEPTH & NOLOSS in s1 AND s2. This is the only claim that carries the hold into C1 work.
  - **PARTIAL:** G-FAR HOLD & NOLOSS in both seeds and DEPTH misses, but depth(G-FAR) − depth(RD) ≥ +0.05 in both seeds (same seed; RD s1 / s2 exist). Descriptive: "reduces the cost", no carry.
  - **UNEXPOSED:** `far_gate2_share` < 0.95 over the 15–16k bin, or `far_ce_g2` ≡ 0 after 4k. No G-FAR reading.
  - The **2 × 2 label** (§3.4) uses the same DEPTH bar for DELAY. DELAY HOLD is reported.
  - **SPLIT:** s1 and s2 differ on HD-LEVER. Then the contingent s3 = base_s3 + G-FAR s3 (+ DELAY s3 only if DELAY also split); 2 of 3 seeds, labelled "(2/3)".
- **Flags:** CONFOUNDED if any CurGate advance differs from the same-seed base by > 2,000 steps, or basin entry (500-step mean answer_ce < 0.5) by > 5,000. A flagged verdict is reported as "(CONFOUNDED)" and never upgraded.
- **Descriptive:** Stage-0 B readings on the new models; DEPTH at K′ 24; `far_ce_g2` / `far_ce_ng2` curves; early `gn_preclip`.

## 5. OC table, censoring, cost, $0 checks

### 5.1 OC (HD-LEVER, 2 seeds, per-seed Δ = depth − base, noise σ_Δ)

- σ_Δ: same-seed small recipe changes give D3-F − D3-T +0.005, FT-CTRL − base −0.004 / −0.030, RD-FT − FT-CTRL +0.007 / −0.033 (rms 0.020). Also 3 n × 1,000 items: binomial SE ≈ 0.009. Use **σ_Δ = 0.03** [ASSUMED, conservative]; σ_Δ = 0.02 shown in brackets.
- P(HOLD both | lever works) ≈ 0.85 [ASSUMED; D6 risk]; P(NOLOSS) ≈ 0.97 (§224: 5 / 5 far cells passed).

| true state | per-seed true Δ | P(DEPTH pass, both) | P(HD-LEVER) |
|---|---|---|---|
| no effect (G-FAR = RD's cost) | −0.076 / −0.153 | 0.063 × 0.000 ≈ 0.000 [0.01 × 0] | ≈ 0.000 |
| half effect | −0.038 / −0.077 | 0.39 × 0.06 = 0.02 [0.34 × 0.01] | ≈ 0.02 |
| full effect | 0 / 0 | 0.84² = 0.71 [0.93² = 0.87] | 0.71 × 0.85 × 0.97 ≈ **0.58** [0.72] |
| full effect, but no hold (D6 false) | 0 / 0 | 0.71 | 0 (HOLD fails) |

- Prior over mechanisms [ASSUMED]: H-REPAIR 0.35, WINDOW 0.25, STEADY 0.15, SETTLE 0.10 (Stage 0 would stop the line), other 0.15. This gives **P(HD-LEVER) ≈ 0.25**.
- The tighter −0.03 bar costs about 0.16 of power at σ 0.03 versus −0.05 (0.84² vs 0.95²). The no-effect pass rate stays ≈ 0 either way, so the bar is set by the C1-first priority, not by power.

### 5.2 Censoring

- A cell that dies before 48k resumes from its off-host checkpoint (tested resume, per the off-host rule). If it cannot finish, it is VOID (not FAIL) and gets one re-run within the cap.
- HOLD / DEPTH / NOLOSS are fixed-K′ accuracies over all test items. No item is dropped, and there is no event-time reading.
- Event readings:
  - first step with `far_gate2_share` ≥ 0.95: never reached → censored at 48,000, reported as "> 48k" (with the event-only mean);
  - basin entry: never reached → "> 48k", flagged CONFOUNDED.
- Stage 0: t_ans = None (never stably right) counts as not-ahead. act / idle use every item.

### 5.3 Cost (cloud RTX 3090, $0.22/h)

- Per far cell ≈ $3.1 (§224: $2.7–3.2 each for 48k + evals incl. probe + khorizon; RD-s3 realised $1.61). Admit at a G-TIME projection ≤ $3.10, guard $3.70 (as 224-6).
- Core: 4 cells ≈ **$12.4**. Base re-evals: none (s224_base_s*_re exist).
- Contingent s3 (SPLIT only): base_s3 ≈ $2.6 + G-FAR s3 $3.1 (+ DELAY s3 $3.1) ≈ $5.7–8.8.
- G-HD-EQ host gates ≈ $0.3. Stage 0: $0 (≈ 1 h local CPU).
- **Cap $22** (max $21.5): within Audit's $30 rule. Expected ≈ $12.4 + 0.25 × $7 ≈ **$14**.

### 5.4 $0 known-answer checks

- **K1 [done]:** CPU fp32 answers at K′ 24 on n 8 test items equal `eval_khorizon.json` items.intact for 16 / 16 items in all 7 models (`cpu_vs_khorizon_n8_k24_share_equal` 1.0).
- **K2 [done]:** idle-classifier held-out accuracy 1.000 in all 7 models. Caveat: perfect separation may mean it reads the pass clock. d idle_pre is read only as arm − base.
- **K3 [done]:** `hd_stage0_score.py --fixtures` 4 / 4 (synthetic SETTLE → SETTLE, VALUE → VALUE, split → MIXED, s3 never same-seed). The deep code path was also smoked on n 8 items (exact, finished, t_ans ≤ 8, idle 0 → 1 at n + 1..n + 2); that smoke file was not kept.
- **K4 [done]:** check A reproduces the scorer's depth deltas (§1.1).
- **K5 (to build, before deploy):**
  - G-FAR gradflow: L_far^G has nonzero grad on core and coda params when Σg > 0, and exactly zero when all g = 0;
  - fp64 synthetic batch with forced right / wrong items: L^G = Σ g CE / 16 to 1e-12;
  - G-HD-EQ v2 (det-wrapped, 1,200 steps crossing the c hold): `--far_gate n2` with the gate forced to 1 is bitwise RD; with `--far_w 0` it is bitwise plain;
  - construct + 1 step with no new flags (plain-path smoke);
  - an args diff against s224_rd_s{1,2} = {far_gate} only.
- **K6 (DELAY):** `--far_on` 600 in a gate run: steps < 600 bitwise plain, `far_ce` logged from 600. Args diff against s224_rd = {far_on}.
- **K7:** `theory/score_hd.py` with fixtures (HOLD / DEPTH same-seed / UNEXPOSED / SPLIT / CONFOUNDED / no-base exclusion), written before any output.

## 6. Novelty sentences for Literature (to check; no "new" claim before that)

- **N1:** random-depth / progressive training with a detached prefix (Anil 2211.09961; Bansal 2202.05826) fixed the long-horizon hold here, but cost length extrapolation by 0.08–0.26. Is a depth / extrapolation cost of progressive-loss training reported anywhere? Bansal reports gains in extrapolation, so is our cost tied to computing errors in the prefix during formation?
- **N2:** gating the far-pass answer loss by the model's own correctness at its trained read depth, so the deep-pass loss acts only as a hold signal and never as a one-pass repair signal from a wrong prefix. Check self-paced learning / loss truncation / correctness-gated auxiliary losses on recurrent-depth models.
- **N3:** the FT result: the same far loss applied after the algorithm formed (repair dose ≈ 0) costs no depth. Is there prior evidence that the timing of an auxiliary stability loss changes which algorithm a looped model learns?

## 7. Open questions for the red team

1. **D6:** can L_hold alone reach H(128) ≥ 0.95? RD-FT (hold-only, 12k, lr 3e-5) reached 0.66 / 0.74. If not, G-FAR fails HOLD, and HOLD and depth may share a cause (the repair gradient).
2. **σ_Δ = 0.03** rests on 5 same-seed pairs. Is a −0.03 DEPTH bar at 2 seeds sound, or should the bar be on the 2-seed mean?
3. **Gate at n + 2 vs at T_far − 1** (FH's gate): the T − 1 gate would drop drift failures, the very hold signal. Is n + 2 the right read when K = n + U trains reads up to n + 4?
4. **Fixed denominator q** vs Σg: at gate share 0.1, L^G is ≈ 10× smaller than RD early. Is that what we want (D4), or should a matched-magnitude control exist?
5. **DELAY onset 16k:** the cosine lr means the delayed far term trains at ≤ 0.75e-4. Does DELAY's HOLD then confound "formation needed" with "lr too low"?
6. **Stage 0:** is the 0.15 / 0.70 SETTLE line well powered (SE ≈ 0.05)? Is the idle classifier (perfectly separable, maybe a clock) informative?
7. **Seed 3:** run base_s3 now ($2.6) to settle Audit (1) for RD-s3, or keep it contingent?
8. **C1 relevance:** DEPTH is n 9–11 length extrapolation (train n ≤ 8). Is that the right C1 proxy for "carry the hold into C1 work", or should DEPTH also require K′ 24 deep answers held?
