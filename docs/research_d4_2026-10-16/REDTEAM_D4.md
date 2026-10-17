# RED-TEAM of D4_draft.md (computed PAUSE passes) — Opus red-team subagent, 2026-10-16/17

Status: COMPLETE (2026-10-17). $0: no training, no GPU, no repo edits.
Scripts: research3/rt_d4/.

## Findings (appended as they are verified)

### F1 (attack 1, decode faithfulness) — EC / t_fin are R-decode readings, and R's decode is demonstrably unfaithful beyond range [MEASURED, rt_d4/rt_decode_answer2.py -> .json]
- Collapse items (any decoded alive drop >= 2) with decoded t_fin < n + 2, coda answer at t_fin vs at n + 2:
  - N1 n 9 (656 items): right at t_fin 0.369 | WRONG at t_fin but RIGHT at n+2 **0.418** | wrong both 0.213. N2 n 9 (381): 0.525 / **0.365** / 0.110.
  - N1 n 10 (994): 0.409 / 0.076 / 0.514; N2 n 10 (910): 0.263 / 0.178 / 0.559. n >= 11: answer ~chance (0.11-0.17) whatever the decode says.
  - So at n 9, ~40% of "decoded finished early" items keep computing after R says 0 operators (the answer changes from wrong to right), while the decode never goes back up (alive_rises 0.000): R reports "done" before the computation is done.
  - N2's decoded alive count INCREASES at some pass in 40% (n 9) to 71% (n 13) of beyond-range items (N1: 0-9%): the on-schedule conditioning of h(t) selects on a noisy decode.
- Consequence: EC/DG/ORS/t_fin measure R's decode, which can itself carry the deadline (R reads V_t; R2 trained "age >= 9 => finished" decodes). A decode-level deadline is NOT the computation's deadline. D3-T is the extreme case already measured (n 9 exact decode at t 7 0.501 vs acc 0.923).
- Answer-level compression IS real at n 10 (draft's acc ratio K'=n-1 / K'=n+2 = 0.89 for D3-T), so the thesis is not refuted, but the registered mechanism readings must not be decode-only.
- Verdict on "EDGE-COLLAPSE is a pass-indexed deadline": NEEDS-FIX (partly decode artefact at n 9-10; unresolvable at n >= 11 where answers are at chance).

### F2 (attack 1, decode-free check) — an answer-level pass-indexed stop exists in all four runs; it is cleaner than the decode reading [MEASURED, eval_vb acc_k; rt_d4/rt_settle.py]
- acc(K') plateaus at K' = 10 for every n >= 10 in N1, N2, D3-T, D3-F: no new correct answers after pass 10, and no bump at K' = n.
  - D3-T n 11: 0.168 (K' 9) -> 0.273 (10) -> 0.266 (13); n 12: 0.107 / 0.123 / 0.125 (K' 12) ; D3-F n 11: 0.158 -> 0.323 -> 0.326; N2 n 10: 0.277 -> 0.434 -> 0.424.
  - Correct answers at K' = n - 1 (D3-T n 11 at K' 10: 0.273 vs chance ~0.10) need >= 1 correct multi-reduction pass, which training never demands: compression toward a pass-~10 finish is real at the answer level.
- Decode-free settle pass (first t from which the coda answer is constant to n + 2):
  - N2: median 9 at every n 9-16 (slope over n 9-14 = 0.00) -> answer-level deadline, agrees with its decode t_fin (9).
  - N1: median settle 9/9/10/11/12/13 at n 9-14 (OLS slope 0.857, spread 4) while decode t_fin is 8/9/9/9/9/10 (slope 0.286). N1's answer keeps changing to ~n - 1 although R says "done" at 9; the changes add no accuracy (acc flat after K' 10), so they are wrong->wrong drift (N1 is K'24-unstable: acc_24 0.48-0.54 at n 6-8).
  - **Trap for Audit's falsifier (b):** "median t_fin tracks n (slope >= 0.7, spread >= 3)" is MET by N1's answer-settle pass (0.857 / 4) although N1 has the deadline. Any t_fin-tracks-n test must be on correct-and-stable items or paired with accuracy gains at K' = n (see F6).
- Verdict: "a pass-indexed finish-by-~10 limit exists" VERIFIED at the answer level (acc plateau + early-correct answers); "EC as read by the decode" NEEDS-FIX (decode unfaithful at n 9-10, F1). Recommended: make ACC-GAIN-AFTER-10 = acc(K'=n+2) - acc(K'=10), n 11-14, a primary decode-free mechanism reading.

### F3 (attack 1, t vs state size) — N1 is pass-keyed; N2 is SIZE-keyed (slot count T), yet N2 passes the draft's EC rule [MEASURED, rt_d4/rt_hazard_keys.py -> .json]
- Draft conditioning (on-schedule, alive >= 3, n 10-16, t 2-12); weighted logistic fits on (t, n, T-bin) cells; T = slot count from the replicated §170 test set (replica verified: every row's y and n match).
- **N1 (carry):** deviance t-only 538 | n-only 14739 | r-only 8389 | t+n 528 | t+T 537 | t+n+T 498. t dominates (b_t 1.22). At fixed (t, n), high-T tercile hazard is modestly higher (t 6 n 10: 0.167 vs 0.273; t 8 n 12: 0.82 vs 0.91). h(t) is the same across n 10-16 at t <= 6 (e.g. t 6: 0.18-0.23). Draft's "pass-indexed" VERIFIED for N1, with a secondary size modulation.
- **N2 (pa):** deviance t-only 1327 | n-only 1272 | t+n 718 | **t+T 555** | t+n+T 555 (n adds nothing given T; b_T 0.225 per slot, b_t 0.45). N2 collapses from pass 2: h(t 2) = 0.08 / 0.29 / 0.43 / 0.62 / 0.76 / 0.82 at n 10-15. At fixed (t 7, n 10): low-T 0.156 vs high-T **0.725**; (t 6, n 11): 0.244 vs 0.800.
- So the draft's §1.1 row "N1 and N2 read EDGE-COLLAPSE, a pass-indexed deadline" is REFUTED for N2: its over-reduction is keyed on state length (+ a weaker t term). N2 meets the EC rule (h(7-9) 0.61, h(8)/h(5) 2.97, t_fin 9/9/9) — **EC is not specific to pass-indexing (a size-keyed model passes it).** Fixture gap: EC's fixture has no size-keyed trajectory.
- Relevance to D4: pauses keep n <= 8, so they add (t, r) coverage but NO coverage of larger T. If D3-T's hazard is T-keyed (like N2), computed pauses cannot remove it. Required: add the t-vs-T fit + fixed-(t, n) T-split to S0-1 as a branch input (SIZE-KEYED -> D4 not proposed), and a size-keyed fixture to d4_edge_collapse.py that must read EC = not-met.

### F4 (attack 1, size vs pass for D3-T itself; RS confound) [MEASURED from code + eval_vb stepcap]
- **RS restart confounds age with size.** vbank_diag.part_rs runs "the item reduced t times run fresh" via N.render(reduce_k(...)): the restart re-renders (VOIDs dropped), so it resets lineage age AND shrinks the slot count T to an r-operator expression AND clears SLVA history. "RS 1.000 in every cell" (draft §1.1 row 1) therefore cannot separate a pass/age deadline from a state-size limit. Verdict on "limit is in the trajectory, not the step": UNVERIFIED as stated (trajectory OR size).
- **Slot count is out of support beyond n 10.** Training T (n <= 8, 200k draws) max 33, p99 29. Test share with T > 33: n 10 0.17, n 11 0.63, n 12 0.96, n 13+ ~1.0. Pauses keep n <= 8, so they add no T coverage. The draft's per-n predictions under DEADLINE-GONE (n 11 0.55-0.85, n 12 0.3-0.7) take no account of this.
- **But D3-T's STEP-CAP(t, a) points to t, not size** (value-slot acc given exact decode at t-1; a = non-VOID content slots): at t <= 4, 0.99-1.00 for every a up to 51; at fixed a = 21: t 5 0.99, t 6 0.91, t 7 0.84, t 8 0.70; at t 8 accuracy RISES with a (0.61 at a 9 -> 0.85 at a 29). Same in D3-F and N1. LONG-STEP (fresh one-pass inputs) exact to m 16 (draft). So for D3-T the early step is size-robust and the decline is pass-keyed — supporting the draft over a pure size story for D3-T — but on exact-decoded items only, where t = a = rho (no separation of the three clocks).
- Net: N1 and probably D3-T are t-keyed; N2 is T-keyed. The branch rule must test it on D3-T's own vbhalt rows (F3 fix), not assume it from N1.

### F5 (attack 5, can DG / EC fail correctly?) — two false-positive routes for DEADLINE-GONE [MEASURED, rt_d4/rt_fixtures.py through an unchanged copy of d4_edge_collapse.read]
| fixture | EC | DG | median t_fin n 9..16 | Audit slope / spread (n 9-14) | T-split diff |
|---|---|---|---|---|---|
| FREEZE3 (items with >= 3 alive at pass 9 stick forever) | F | **TRUE** | 9, 10, 99, 99, ... | 20.5 / 90 -> **passes** | 0.00 |
| SHIFT13 (draft's deadline fixture moved +4) | F | **TRUE** | 9, 10, 11, 12, 12, 12, 13, 13 | 0.63 / 3 -> fails (catches it) | 0.00 |
| SHIFT12 | F | F | 9, 10, 11, 11, 11, 12, 12, 12 | 0.51 / 3 -> fails | 0.00 |
| TONLY (N1-like t-only hazard) | **T** | F | 8, 8, 9, 9, 10, 10, 11, 11 | 0.46 / 2 | 0.03 |
| SIZEKEY (N2-like t+T hazard) | F (fixture too early-collapsing; real N2 reads T) | F | 8 x 8 | 0 / 0 | **0.22** |
- **Bug (censoring):** d4_edge_collapse codes never-finishing as t_fin = 99 and takes medians, so a STUCK model satisfies "median t_fin(n 12) - median t_fin(n 10) >= 2" and h(7-9) = 0 -> DEADLINE-GONE. That breaks the OC block's censoring rule (never-reached events count as failures). Audit's slope test has the same hole. Fix: t_fin = never counts as failure; DG readable only if >= 80% of n 10-12 items reach alive 0 by K' n + 2; medians over finishers + a finish-rate floor.
- **Shifted deadline:** the draft's DG passes a deadline moved to ~12-13 (SHIFT13). Audit's slope >= 0.7 over n 9-14 catches SHIFT12/13 here but not a hard deadline at >= 13 (analytic: t_fin = min(n, 13) gives slope 0.857, spread 4). Since PAUSE-A's zero-loss deadline is at 17 and n 13-16 are only "descriptive", require a scan: DEADLINE-MOVED(t_d) for t_d in 9..16 = median t_fin pinned within 1 pass of t_d over >= 3 n-values > t_d (finishers only, with the finish-rate floor). Report t_d; DG = no t_d <= 15.
- T-split separates SIZEKEY (0.22) from TONLY (0.03): adopt as the size test (F3).

### F6 (attacks 2-3, Prop D0 / D1 / D2 numbers) [COMPUTED, rt_d4/rt_schedule.py; d4_schedule_stats.py re-read]
- D1 table REPRODUCED for the script as written (P(unfinished | t >= 9/12/14/16/17) = 0.421 / 0.315 / 0.193 / 0.050 / 0).
- **Spec/simulation mismatch (NEEDS-FIX):** the script shuffles the S scattered pauses with the n reductions, so **30.6%** of augmented items end with a pause on FINISHED input; completion then precedes n + P (mean gap 0.51 passes). The draft says "no pauses after completion" and sets R1's window at tau = n + P. Pick one. With the constraint enforced, D1 reads 0.482 / 0.376 / 0.244 / 0.071 / 0 (G-P3 must reproduce whichever is implemented, and the target map / R1 window must use the actual completion pass).
- **D0b VERIFIED numerically:** the marker-free test stratum (t >= 9, unfinished, no pause before t) has **0** training passes; P(unfinished | t >= 9, no pause yet) = 0 vs 0.48 with a pause in history. Max unmarked-pass count on an unfinished pass = 7 (so "unmarked count >= 8 => finished" is zero-loss). So Prop D2's "coverage of the test diagonals 6.7k-57k per 1M" is coverage *conditional on a pause history*; the stratum the marker-free eval actually visits has zero coverage under every pause schedule (Audit (a)). Relabel D2's row and state that transfer needs history-invariance that no loss term supplies.
- **Shifted-deadline pressure is weak (Audit (b)):** an ungated deadline at t_d costs 0.82 unfinished passes/item at t_d = 9, 0.23 at 12, **0.062 at 14, 0.0055 at 16**. In a 12k-step FT the deadline can plausibly settle at ~13-15, which passes DEPTH-GENERAL (n 9-12) and the draft's DG while the clock survives. Needs the DEADLINE-MOVED scan (F5) and a registered C2 reading at n 13-16.
- **The cheapest gated solution is in the new parameter itself.** If the age signal is an additive increment per core application, pi = -(that increment) makes pause passes age-neutral: a 768-parameter solution needing no change to the trained core, and pi starts at 0 with the most direct gradient. The draft lists "marker as a reset" but rates the marker route P ~ 0.3 overall; from a converged deadline model (FT) this route should be the prior favourite. START-D's GATED label (t_fin shift >= +3 in bank index at D 4) does catch it. VERIFIED that the readings catch it; the OC P(deadline removed) ~ 0.45 is UNVERIFIED and in my view optimistic for the FT.

### F7 (attack 1 on D3-T itself, decode-free and R-free) — D3-T is not size-keyed like N2 [MEASURED, rt_d4/rt_probe_size.py; eval_vb probe per-item rows]
- PROBE(12) per-item correctness (a fresh coda trained on n 3-8; tracks acc_n2: D3-T n 10 0.598 vs 0.603) split within n by extra size T - 2n (= 3 + 2 x bracket pairs). Logistic acc ~ n + (T - 2n), n 9-12, slope on extra size (SE):
  - **N2 -0.186 (0.018)** (n 10: low-T 0.558 vs high-T 0.196) — confirms F3 without R;
  - N1 -0.054 (0.017); **D3-T -0.035 (0.017)** (n 10: 0.612 vs 0.530; n 9: 0.898 vs 0.887); D3-F -0.030 (0.017).
- So for the base actually proposed (D3-T), a within-n size effect is small; D3-T resembles N1 (t-keyed), not N2. The draft's choice of D3-T and the pass-keyed reading survive this test; the N2 claim does not.
- The draft's §1.1 "finishes early" ratios VERIFIED from acc_k (D3-T 0.18 / 0.36 / 0.89 / 1.03; N1 0.15 / 0.27 / 0.96 / 1.13; N2 0.16 / 0.20 / 0.65 / 1.01). In range the coda cannot do the last step (n 8 ratio 0.18), so the early-correct answers at n 10 come from compression in the core: VERIFIED as answer-level compression.
- Base rule check (D3-F landed after the draft): D3-F mean acc_n2 n 9-12 = 0.494 vs D3-T 0.480 (+0.014 < 0.05) and acc_k24 n 1 = 0.835 (< 0.99) -> **base = D3-T** by the draft's own rule.

### F8 (attack 1, what the N1 evidence excludes) [MEASURED, rt_d4/rt_lag.py; eval_foreign.json; code]
- **FOREIGN-INSERT excludes only the bank index.** part_foreign pushes o donor versions after V_0, keeps the carry at V_0 and runs n + 2 real passes at absolute indices o+1..o+n+2. N1 n 6/7/8 at o 6 = 0.983/0.974/0.951 (VERIFIED), so the bank index is not N1's clock. It says nothing about **lineage age vs content progress rho**: both live in the carried state, and rho is directly content-readable (the VOID count of the slot-aligned state encodes it). The draft's step "so the deadline is carried by lineage age" relies on CLOCK-KEY. At n 9, FOREIGN o 6 does cost 0.853 -> 0.728, so the bank is not irrelevant beyond range.
- **CLOCK-KEY sign is not a decode-lag artefact in N1.** Only 22.9% of N1's behind->collapse events (849, n 10-16) land exactly back on schedule (N2 47.2%); a lag-catch is the first off-schedule event for just 1-2 items per n in N1. g_dev = +0.53 (VERIFIED; SE 0.035) is real overshoot from behind. But it contradicts a rho-key AND a pure t/age-key (both predict g <= 0); it says the hazard rises with work outstanding at fixed t. Verdict on "age, not rho": UNVERIFIED — S0-3 must decide it, and is therefore the premise gate (F11).
- N1 n 9 items whose first off-schedule decode is an over-reduction still answer 0.693 at n + 2 (clean items 0.980): decoded over-reductions at n 9 are often harmless (consistent with F1).

### F9 (attack 3, lemmas)
- **Prop D0 (a), (b): VERIFIED** (counting; numerically F6). It cuts against the plan harder than the draft's OC admits: (a) means a PROGRESS-ONLY reading in S0-3 makes D4 provably unable to act, so P(deadline removed | PROGRESS-ONLY) should be ~0, not the draft's 0.2 with the FT still run. NEEDS-FIX: PROGRESS-ONLY = stop.
- **Prop D1: VERIFIED** as counting (b-1 = a = rho under D3-T; copy-stall decouples b only; pause decouples b and a; max t + r 9 / 17 / 21); table numbers reproduce, modulo the trailing-pause mismatch (F6).
- **Prop D2: numbers reproduce; the coverage claim NEEDS-FIX** (coverage exists only with a pause history; the marker-free stratum has 0, F6). "P perp n" VERIFIED by construction.
- **Lemma D3: VERIFIED** (softmax identity dU/dl_s = alpha_s (v_s - U) is standard; the quadratic O(||H|| delta^2) is a DERIVED local model; D3-T's slva_gn 8e-5 and loss 2e-5 at 48k fit it). Tangential to the plan.
- **Lemma D4 (a)-(c): VERIFIED** as standard linear-recursion bounds (fixed point delta* = delta_a + eps/alpha_0 for (c)). **(d) NEEDS-FIX:** "state-dependent eps => support effect => both answered by coverage" is a leap. A state-dependent eps can depend on T (out of support from n 11, F4), and pauses add no T coverage. Say "answered by coverage of the variable eps depends on: (t, r) by pauses, not T".
- **Lemma D5: NEEDS-FIX.** The bound is right for sliding-window truncation (drop lags >= k: sum_{t>k}(t-k) = (K-k)(K-k+1)/2 pairs in the rho_b -> 1 case; "relative" is a ratio of upper bounds, as stated). But **a sliding window saves no memory with vb_loss's single backward**: every pass's graph stays alive until the one backward. The version that saves memory (the "~13 / ~16 GiB with TBPTT 12" rows) is chunked TBPTT, which also cuts lag-1 credit at the chunk boundary (pass 13 cannot reach pass 12): exactly the late unfinished passes D4 is about. Neither variant exists in vbank.rollout. **Better: per-pass activation checkpointing** (exact gradients, about 1.3x compute; THEORY already names a --ckpt_passes fallback, also not implemented). For PAUSE-A at Kb 20, full BPTT probably fits (F10), so A needs neither.

### F10 (attack 4, implementation and harness traps) [code read: boxes/vbank.py, slva.py, vbank_eval.py, fsq120.lr_at]
VERIFIED in code:
- `input_edit` is applied in rollout after the gather / SLVA read and after sg_input, before block 6, at every pass t >= 1 (vbank.py 318-321), so a train-time closure U_t + mask_i(t) * pi works. The attribute is documented eval-only, so the trainer's host eval (acc_n2_host) must clear it (G-P9).
- HOLD branch: `term = where(ch.sum(1) > 0, 0.5 tc + 0.5 tu, tu)` (vbank.py 498), so a pause pass with E_t = E_{t-1} takes the unchanged-slot mean, value classes weighted 2.
- `--c_sched 0,1`: c_at(step, 0, 1) = max(0, 20 (1 - step)) = 0 at every step >= 1, and SLVA.anneals is True: G-P4 holds as specified.
- `--cur 8`: a single-stage CurGate (update() returns None at the last stage).
- Cost follows the batch max: rollout runs to Kb = max K_i.
Traps that silently change the recipe (none blocks feasibility; each needs a named gate):
1. **R1 window.** `win = (n_ <= t) & (t <= Kt)` (vbank.py 470). If it is not changed to tau_i <= t, R1 demands the final answer at passes n..n+P-1 of a paused item, which trains answering from an unfinished state: it *trains compression*, the very deadline D4 tries to remove. G-P2 covers the window; make it a named, fixture-tested assertion. tau_i must be the actual completion pass (F6: 30.6% of simulated items finish before n + P).
2. **calibrate overwrites beta / beta0.** The non-resume branch (vbank.py 602-615) always runs m.calibrate() (sets beta.data / beta0.data) and rewrites args / calibration. `--init_from` must bypass it. G-P1 catches it only if it runs through the trainer's real init path. Suggested: --stop_at 0 --keep_ckpt_steps 0, then compare ckpt_model_0.pt with the base model.pt bitwise on every shared key.
3. **Strict loading everywhere.** VE.load does m.load_state_dict(...) strict (vbank_eval.py 42). A new `pause_vec` key breaks every eval script on the FT model.pt unless the factory adds it from args; adding it unconditionally breaks D3-T / D3-F / N1 / N2 loads. Add "old and new model.pt load through VE.load" to G-P0 / G-P5.
4. **mem_check / auto_micro / time_steps force K = a.kmax (12) on the longest n <= 8 items** (vbank.py 622, 646): they would certify the plain shape. G-P6 needs a forced Kb = kcap path. Shortcut: K = min(n + U{0..4}, kmax) never exceeds 12 for n <= 8, so --kmax 20 leaves plain batches unchanged and makes the forced checks run at 20, a conservative proxy (more coda passes than a paused batch).
5. **mem_check under-reads the steady state.** It measures a single step (last = 1); AdamW moments (~1.0 GiB for ~125M params) are allocated after that step's backward, so from step 2 the peak is ~+1 GiB. The projected 19.6 GiB becomes ~20.6, near the 21 GiB rule; it still fits a 24 GB card. Run mem_check for 2 steps.
6. **lr_at** (fsq120.py 740): a hard-coded 100-step linear warm-up, then cosine to 0 over --steps. --lr_const / --warmup 500 are new code (draft flags it). D3-T ended at lr 0, so the FT re-heats to 3e-5 (30% of the base's peak): FT-CTRL is essential.
7. **TBPTT is absent** and a sliding window saves no memory (F9 D5). Use activation checkpointing, or full BPTT for A.
8. **Logging:** slva.mass_stats calls rec_targets unmapped, so slva_offnewest_chg / copy are mislabelled on paused items (diagnostic only); map it too.
9. **Hidden-work route through R2.** R2 constrains only R's 20-class linear readout. The core can satisfy HOLD at a marked pass while still reducing in R's null space, then "release" later. F1 shows R and the computation already come apart beyond range. PAUSE-LEARNED (final acc >= 0.99 at K' n + 8) cannot see this. Add **PAUSE-EARLY**: under test pauses, acc at K' = n + P - 1 must stay near the in-range baseline ratio (0.18 at n 8); a high early accuracy means the pause is only in the decode.

### F11 (attack 5, readings / OC / censoring / Stage 1 / cost)
- **DEADLINE-GONE's OC row "P(pass | no effect) <= 0.05" is UNVERIFIED** and too low until the F5 fixes land. A model that freezes at n >= 12 passes DG (censoring bug), and FT-DIVERGED censors only in-range damage (n 1-8 acc, acc_k24), so a beyond-range freeze is not censored. A deadline shifted to ~13 passes DG too.
- **EC is not specific** (N2, size-keyed, passes it; F3). It is fine as a branch input once the T-split clause is added.
- **Decode-only mechanism readings** (EC, DG, ORS, t_fin, START-D's t_fin) inherit R's unfaithfulness (F1). Add the decode-free pair:
  - **ACC-GAIN-AFTER-10** = acc(K' n + 2) - acc(K' 10) over n 11-14; it is ~0 now in all four runs (F2). Deadline removed => > 0, and gains at K' = n.
  - **PAUSE-EARLY** (F10.9).
- **The marker-free test stratum has zero coverage** (F6). DEPTH-GENERAL therefore tests exactly the history-invariance bet. Positive = strong, null = weak, as the draft says; the OC numbers (P(deadline removed) ~ 0.45) are ASSUMED, and for an FT I would put them lower (F6, cheapest gated route = pi itself).
- **Stage 1 = FT pair from D3-T (Audit c): acceptable as a cheap mechanism probe, but it is the base most favourable to gated solutions.** A converged deadline circuit plus a free 768-d pi makes "pi cancels the age increment" the minimal edit. Required changes:
  - (i) GATED in the FT must not veto 1b. The draft's "1b does not run under GATED" discards the better test because of a base-choice artefact; under FT-GATED run 1b (from scratch), or stop D4 outright, decided before data.
  - (ii) Optional cheap variant: a frozen random-direction pi (not learned, fixed norm), which removes the free-parameter cancel route.
  - (iii) FT-PAUSE and FT-CTRL on the same host / GPU class, sequentially (basin and speed variance), with identical item streams (pause schedule from its own rng) so that AUG - CTRL isolates the pauses.
- **PROGRESS-ONLY must stop D4** (D0a: pauses provably cannot act on a rho-keyed deadline). The draft lowers P to 0.2 and still runs the FT.
- **Delayed-start-heavy A has its own easy gated route.** A clock that starts at the first content change (visible in the bank: the pre-reduction versions equal V_0) discounts every start pause. Only scattered pauses defeat it, and A puts E[D] = P/2 of its pauses at the start. Against content-change-counting clocks (= rho) neither kind helps (D0a). START-D reads this route, but the branch rule's rationale for A ("exactly the test diagonal") holds only in absolute pass coordinates.
- **Cost realism:** the projections are consistent with D3-T's measured 0.407 s/step (augmented ~0.41 x 19.5/11.8 ~ 0.67 s; the FT pair ~$0.72-1.08; the 48k cell $1.30-1.87). With the known ~2x 3090 host spread on launch-bound loops, the 1a cap of $1.5 can bind; set it after a same-host G-TIME on forced augmented batches (G-P7 already planned).
- **D3-F landed:** base = D3-T (F7).

### F12 (attack 6, cheaper $0 tests of the same premise)
1. **Already decisive, $0, done here:** the answer-level plateau (F2) shows D3-T has a pass-~10 stop at the answer level, and PROBE(12) per-item rows show D3-T is not size-keyed (F7). Use both as S0-1 inputs alongside EC / ORD (they do not depend on R).
2. **CLOCK-PROBE + AGE-ERASE on D3-T (eval-only, local, ~minutes).** In-range items rolled to 24 give idle passes t = n+1..24 with identical content E_n and only age differing (D3-T is K'24-stable, acc_k24 1.000).
   - Fit a linear probe for t (or 1[t >= 9]) on idle-pass V_t. Decodable => an explicit age signal exists; not decodable => a lineage-age clock is unlikely and D4's premise weakens.
   - Then project the fitted age subspace out of the pass input U_t (input_edit already exists) on n 9-12 items and read ACC-GAIN-AFTER-10 / acc_n2 / EC.
   - A rise is a causal premise test AND points to a training-free fix (an age-invariant carry); no rise argues against the age carrier.
   - vbank_halt already lists CLOCK-PROBE as "later".
3. **Fix S0-3 AGE-REVERT before running it.** Reverting "slots whose decode changed" to S_{t-1}[j] keeps the handle's slots at their fresh V_0 vectors through every pause: the same handle is attempted each time, so only non-handle slots age. A deadline that reads age locally at the handle would not fire => a false PROGRESS-ONLY (and by F11 that should stop D4).
   - Add an **AGE-MCT** arm: V_t' = m.mct(V_t, E_{t-1}), the existing minimum-norm edit in R's row space. It keeps the computed, aged vectors on every slot. Caveat: content outside R's row space may still carry the reduction; the n 1-4 control guards against that.
   - Pre-state that PROGRESS-ONLY requires both arms null.

---

## Verdicts per claim
| draft claim | verdict | key numbers |
|---|---|---|
| §1.1 limit is in the trajectory, not the step (RS 1.000; acc flat K' 10 -> 24) | numbers VERIFIED; inference NEEDS-FIX | RS re-renders, so it resets age AND shrinks T (F4); STEP-CAP / PROBE favour t over T for D3-T (F4, F7) |
| §1.1 beyond range D3-T finishes early (0.18 / 0.36 / 0.89 / 1.03) | VERIFIED | acc_k; answer-level compression is real (F2, F7) |
| §1.1 / note 213-24 "N1 and N2 read EC, a pass-indexed deadline" | N1 VERIFIED (decode level); **N2 REFUTED** | N2: t+T deviance 555 vs t-only 1327; fixed (t 7, n 10) h 0.156 vs 0.725 by T; PROBE(12) size slope -0.186 (F3, F7) |
| §1.1 N1 GRADUAL-DRIFT is mostly over-reduction (ORS 0.69) | numbers VERIFIED; reading UNVERIFIED | decode-level; R says "done" while answers keep changing (F1, F2) |
| §1.1 N1's clock is not the bank index | VERIFIED (that only) | FOREIGN o 6 n 8 0.951; age vs rho not separated (F8) |
| §1.1 hazard ~ t x remaining work (g_dev +0.53) | VERIFIED; not a decode-lag artefact | 22.9% of behind->collapse events land on schedule (F8) |
| §1.2 carrier = lineage age | UNVERIFIED (the premise) | S0-3 decides it, and needs the AGE-MCT fix (F12) |
| Answer-level pass-~10 stop in D3-T | VERIFIED (new, decode-free) | acc plateau after K' 10 at every n >= 10 (F2) |
| Prop D0 (a), (b) | VERIFIED | test stratum has 0 training passes; P(unf given t >= 9, no pause) = 0 vs 0.48 (F6) |
| Prop D1 | VERIFIED (counting + numbers) | spec mismatch: 30.6% trailing pauses (F6) |
| Prop D2 | numbers VERIFIED; coverage claim NEEDS-FIX | coverage exists only given a pause history (F6) |
| Lemma D3 | VERIFIED | — |
| Lemma D4 (a)-(c) / (d) | VERIFIED / NEEDS-FIX | (d) ignores T (F9) |
| Lemma D5 / TBPTT memory | NEEDS-FIX | sliding window saves no memory; chunked cuts lag 1 (F9) |
| §2 implementability | VERIFIED feasible | 9 traps, each needing a named gate (F10) |
| Memory 19.6 GiB at Kb 20 | plausible, ~+1 GiB low | single-step mem_check omits the AdamW moments (F10) |
| Cost | VERIFIED consistent | host spread can bind the $1.5 cap (F11) |
| DEADLINE-GONE reading / OC <= 0.05 | **REFUTED as specified** | FREEZE3 and SHIFT13 fixtures read DG = True (F5) |
| EC reading | NEEDS-FIX | not specific: size-keyed N2 passes (F3) |
| PAUSE-LEARNED gate | NEEDS-FIX | blind to decode-only pausing; add PAUSE-EARLY (F10) |
| Audit (b) falsifier (slope >= 0.7, spread >= 3) | NEEDS-FIX | passes FREEZE3 (99s) and N1's answer-settle pass (0.857 / 4); catches SHIFT12/13; misses a hard deadline at >= 13 (F2, F5) |
| Stage 1 FT pair from D3-T | acceptable with changes | the FT favours gated / pi-cancel routes; GATED must not veto 1b (F11) |

## Blockers (fix before Audit / registration)
1. **DG / t_fin censoring:** never-finishing is coded 99, so a stuck model reads DEADLINE-GONE (and passes Audit's slope test). Count never as failure, add a finish-rate floor, and add FREEZE and SHIFT fixtures that must read not-met.
2. **Decode-only mechanism readings:** add decode-free ACC-GAIN-AFTER-10 (= acc(K' n + 2) - acc(K' 10), n 11-14) and PAUSE-EARLY. EC / DG become co-requirements, not sole evidence.
3. **Correct the N2 claim** in §1.1 and THEORY note 213-24 (size-keyed). Add the t-vs-T fit and the fixed-(t, n) T-split to S0-1 as a SIZE-KEYED stop clause, with a size-keyed fixture.
4. **S0-3 AGE-REVERT keeps the handle fresh** (false PROGRESS-ONLY risk). Add AGE-MCT, and make PROGRESS-ONLY a hard stop (D0a).
5. **Spec consistency:** decide on trailing pauses. tau_i must be the actual completion pass; add a named R1-window assertion (an unchanged window trains compression).
6. **Shifted-deadline scan** DEADLINE-MOVED(t_d), t_d 9..16, on finishers; a registered C2 reading at n 13-16.

## Other required fixes
- `--init_from` bypasses calibrate; run G-P1 through the trainer's own init path (ckpt_model_0 vs base, bitwise).
- VE.load strict-load compatibility for the old and new model.pt.
- mem_check / auto_micro / time_steps at the forced kcap shape (or --kmax 20), mem_check over 2 steps.
- Replace TBPTT with activation checkpointing, or full BPTT for A.
- Relabel D2's coverage; fix Lemma D4(d) and D5.
- FT-GATED does not veto 1b; run the FT pair on the same host with matched item streams.
- Optional: a frozen-pi variant. Also register START-D at D 8 as a PAUSE-PROTOCOL descriptive C2 reading (if gating is learned, test-time start pauses are a usable, bounded protocol).

## Cheaper premise test (recommended before any spend)
CLOCK-PROBE on D3-T's idle passes (same content, only age differs), then AGE-ERASE: project the age subspace out of U_t on n 9-12 and read ACC-GAIN-AFTER-10. This is eval-only and $0, and gives a causal test of the age carrier plus a possible training-free fix (F12).

## Not available / not waited for
D3-T / D3-F eval_vbhalt, eval_stepbyt and eval_hist24 do not exist yet (checked 2026-10-17 00:0x). Every D3-T statement above uses eval_vb (acc_k, stepcap, probe per-item rows).

## Files (scratchpad research3/rt_d4/)
- testset.py: pure-numpy §170 test-set replica, verified against rows.
- rt_decode_answer.py, rt_decode_answer2.py: F1.
- rt_settle.py: F2.
- rt_hazard_keys.py: F3.
- rt_fixtures.py + d4ec_copy.py (unchanged copy of the draft reader): F5.
- rt_schedule.py: F6.
- rt_probe_size.py: F7.
- rt_lag.py: F8.
- One .json output per script.
