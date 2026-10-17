# PROOFS_D4: rigour pass on D4_draft.md §3 (D0-D5, (v)) and §5 OC

Opus proof subagent for Theory, started 2026-10-17 00:0x EDT. $0: CPU only (OMP_NUM_THREADS=3), no training, no GPU, no repo edits.
Check scripts: research3/proofs_d4/ (exact_sched, n_law, d0_d1_check, lemma_d3_check, lemma_d4_d5_check, falsifier_check, dgone_shift_check, oc_cost_check; each writes the .json of the same name). Status: COMPLETE.

Code read: boxes/vbank.py (rollout, bias_row, vb_loss, draw, c_at), boxes/slva.py (SLVA.gather), boxes/nested170.py (train_items, generate),
docs/research_d4_2026-10-16/scripts/d4_schedule_stats.py (+ .json), d4_edge_collapse.py (+ refs), s0_finishpass.py, s0_clockkey.py.

## 0. Verdict summary (filled in at the end)

See section 12.

## 1. §2.2 schedules: internal consistency and the simulation code

Checks: `proofs_d4/exact_sched.py` = an EXACT enumeration (hypergeometric law of the shuffled mid positions, no Monte Carlo) of BASE / PAUSE-A / PAUSE-B,
in two variants: **free** (scattered pauses uniformly interleaved with the n reductions, trailing pauses after the last R allowed; this is what
`d4_schedule_stats.py`'s `rng.shuffle(mid)` does) and **lastR** (the last mid position forced to be R, so completion = n + P, as §2.1 states).
Outputs `exact_uniform.json` (n ~ U{1..8}, D4's assumption) and `exact_realised.json` (the trainer's realised n law, `n_law.py`).
The exact free/uniform values reproduce D4's 200k-item simulation to MC noise (e.g. A P(F=0 | t>=9) 0.4205 exact vs 0.4218 / 0.4208 MC; I(t;rho) 0.9654 vs 0.958 / 0.965).

| claim | verdict |
|---|---|
| A: completion n + P <= 16, never censored under cap 20 | **[PROVED]** K = min(n+P+U{0..4}, 20) and n+P+4 <= 20, so the cap never binds at all (K = n+P+U exactly; "cap 20" is vacuous) |
| B: cap 24 with tail, never censored | **[PROVED]** n+P <= 20 < 24 so K >= n+P always; the cap binds only on the tail (max n+P+4+8 = 32) |
| plain batches = D3-T's K = min(n + U{0..4}, 12) | **[PROVED]** matches `vbank.main.draw` (kextra 4, kmax 12) |
| D ~ U{0..P} (A), U{0..floor(P/3)} (B) | matches the simulation (D_div) |
| "τ_i = n_i + P_i (the completion pass)"; "no pauses after completion" (§2.1) vs "remaining P − D uniformly interleaved with the n reductions" (§2.2) | **[FALSE: internal inconsistency]** Uniform interleaving puts >= 1 pause after the last R with probability S/(n+S): **0.309 of A items, 0.542 of B items** (exact, uniform n). The simulation (hence every §3 number) uses this free variant, where the completion pass is D + (position of the last R) < n + P and the trailing pauses are marked HOLD passes on a FINISHED input (0.26 per A item, 0.64 per B item, pooled with plain). Fix: pick one. lastR matches §2.1's R1 window and target map; free matches the numbers. G-P3 must test the variant actually implemented (the numbers differ, below). |
| the n law | **[FALSE as assumed]** `nested170.train_items` redraws n AND the item when the item is in EXCL (test set + SERIAL pool). Acceptance a(1) = 0.118, a(2) = 0.975, a(n>=3) ~ 1, so P(n=1) = **0.017** (not 0.125) and P(n>=3) = 0.141 each (`n_law.json`; the trainer's own sampler gives 0.017 / 0.139 / 0.14). Effect on §3 numbers: small (e.g. BASE I(t;rho) 2.357 -> 2.448, A I(t;rho) 0.965 -> 1.013); conclusions unchanged. |
| loss weighting | D4's tables are pass-weighted (each pass 1). `vb_loss` averages R2 over t per item, then over items, so the gradient sees each pass with weight 1/K_i. Loss-weighted values are given below; they do not change any conclusion. |

Exact values (uniform n; pass-weighted unless marked; F = input finished, O = output target finished):

| quantity | BASE | A free | A lastR | B free | B lastR |
|---|---|---|---|---|---|
| P(F=0 \| t>=9) | 0 | 0.4205 | 0.4819 | 0.4120 | 0.5376 |
| P(F=0 \| t>=12) | 0 | 0.3138 | 0.3755 | 0.3231 | 0.4533 |
| P(F=0 \| t>=14) | 0 | 0.1920 | 0.2439 | 0.2419 | 0.3562 |
| P(F=0 \| t>=16) | 0 | 0.0500 | 0.0714 | 0.1560 | 0.2473 |
| P(F=0 \| t>=17) | 0 | **0** | **0** | **0.1135** | 0.1900 |
| P(F=0 \| t>=20) | 0 | 0 | 0 | **0.0127** | 0.0284 |
| P(F=0 \| t>=21) | 0 | 0 | 0 | **0** | **0** |
| P(O=0 \| t>=t*): first t* with 0 | **8** | **16** | **16** | **20** | **20** |
| I(t; rho) bits | 2.357 | 0.965 | 0.938 | 0.883 | 0.818 |
| I(t; F) bits | 0.281 | 0.179 | 0.155 | 0.212 | 0.162 |
| I(t; M) bits | 0 | 0.0346 | **0.0484** | 0.0173 | 0.0248 |
| I(t; M), loss-weighted | 0 | 0.0114 | 0.0192 | 0.0127 | 0.0097 |
| max t + r on computing passes | 9 | 17 | 17 | 21 | 21 |
| E[Kb] per-batch aug (bs 64) | 11.80 | 15.19 | 15.19 | 17.78 | 17.78 |
| E[Kb] augmented batches only | - | 18.59 | 18.59 | 23.76 | 23.76 |
| E[Kb] per-item aug | 11.80 | 17.90 | 17.90 | 23.17 | 23.17 |

(Realised n law: A free P(F=0|t>=9) 0.430, I(t;rho) 1.013, I(t;M) 0.034; A lastR I(t;M) 0.046; B free 0.431 / 0.951 / 0.018; see exact_realised.json.)

**Fix to the D1 table, PAUSE-B row:** D4 prints t* 17 -> 0.22 and t* 20 -> 0.03. Those are the PER-PASS values P(F=0 at t = 17) = 0.217 and
P(F=0 at t = 20) = 0.033 from the JSON's `p_unfinished_by_t`, not P(F=0 | t >= t*). The correct row is
**0.41 / 0.32 / 0.24 / 0.16 / 0.11 / 0.013 / 0** at t* 9 / 12 / 14 / 16 / 17 / 20 / 21 (the printed 0.22 > 0.16 non-monotonicity was the symptom).
The A row is right (0.42 / 0.31 / 0.19 / 0.05 / 0 / 0 / 0). (`d4_schedule_stats.py` computes the >= table only for t* 6..20, so 21 was never computed; it is 0 exactly.)

**Fix (input- vs output-keyed deadline):** EDGE-COLLAPSE is an OUTPUT event (a pass removes >= 2 operators; t_fin = first pass whose output has 0 alive).
The output-keyed rule "pass t >= t* outputs a finished state" pays loss only where the OUTPUT target is unfinished. That rule is zero-loss from
**t* = 8 (BASE), 16 (A), 20 (B)**, one pass earlier than the input-keyed 9 / 17 / 21. So "every deadline at t* <= 16 (A) or <= 20 (B) pays loss"
holds for input-keyed rules only; for the output-keyed collapse the paying range is t* <= 15 (A) / <= 19 (B), and the loss mass near the cap is small
(A: P(O=0 | t>=14) = 0.073, t>=15: 0.028; loss-weighted 0.069 / 0.027). "Pays loss" is a statement about the optimum, not about learning dynamics:
whether 3-7 % unfinished mass suffices to keep a late deadline from forming is **[ASSUMED]**.

**G-P3 thresholds:** I(t;M) <= 0.05 passes for every variant but lastR-A (0.048 pass-weighted, 0.046 realised n) is within 0.004 of it;
|corr(P, n)| <= 0.02 holds exactly at item level (P is independent of n); P(F=0 | t>=9) >= 0.40 passes in all four variants.

## 2. Prop D0 (support identity) and D0b (history-gated deadline)

**Setup (definitions in vb_loss's terms).** An item has n <= 8 Feng reductions. A training schedule is a word over {R (reduction), P (computed
pause), C (copy-stall), I (idle, finished input)} with exactly n R's and I only after the n-th R. rho_{t-1} = #R before pass t (the target index:
R2's target at pass t is E_{rho_t}); r_t = alive operators of the TARGET input state E_{rho_{t-1}}. Every Feng reduction removes exactly one operator
(`recode_targets`: one handle per step), so alive(E_k) = n - k.

- **D0 (rho + r = n <= 8 on every training pass, any schedule) [PROVED]**: r_t = alive(E_{rho_{t-1}}) = n - rho_{t-1}. This is an identity of the
  TARGET schedule; the network's decoded state can leave it (that is what EC measures).
- **D0(a) [PROVED]**: rho >= 8 and rho <= n <= 8 give rho = n, so "rho >= 8 => finished" agrees with every R1 / R2 target under every schedule.
  The stronger and more useful form: **every state with rho + r >= 9 (every pass of every n >= 9 item, from pass 1) is outside the training support
  under every schedule**, so no schedule over n <= 8 items constrains behaviour there. D4's wording "test states with rho >= 9 are never in support"
  is true but weaker than this; replace it. Corollary: any rule keyed on size (n >= 9, readable from the slot count T) is vacuous in training.
- **D0(b) [PROVED UNDER: every non-reducing pre-completion pass is marked]**. Fix a marker pattern Pi. If a training item's marker history before t
  equals Pi_{<t} and its input at t is unfinished, then all passes before t are pre-completion R or P (I only follows completion; trailing
  pauses in the free variant also follow completion), so rho_{t-1} = (t - 1) - |Pi_{<t}| <= n - 1 <= 7, i.e. t <= 8 + |Pi_{<t}|. Hence
  "t > 8 + |Pi_{<t}| => finished" is zero-loss; for Pi = none (the test condition) this is "no marker yet and t >= 9 => finished", D4's
  history-gated deadline. Brute force (`d0_d1_check.py`, 7,984 pause-only schedules): 0 violations. **With unmarked copy-stalls the bound is
  false** (5,920 / 7,984 C-only schedules violate it): the correct general bound is t <= 8 + #(non-reducing passes before t). This does not affect
  D4 (no copy-stalls in PAUSE training) but D0's preamble "under any pass schedule (stalls, pauses, idle passes)" must not be read as covering (b).
- **D0b = the MARKER-GATED clock [PROVED]**: on every unfinished pass, #unmarked passes before t = rho_{t-1} exactly (brute force: 0 violations).
  So the gated count IS progress, and its deadline is zero-loss by D0(a). D0b and the MARKER-GATED route are one route, not two.
- [CONJECTURE] that either is learnable: it needs a carried "marker seen" bit or an unmarked-pass counter. The bank stores the pause outputs V_t,
  so a trace of pi can persist in the stored history even under HOLD decode targets. Zero loss is proved; learnability is not.

## 3. Prop D1 (which clocks each schedule decouples)

**Definitions in vbank.rollout's terms.** St[0] = V_0; pass t reads St[0..t-1] (SLVA: `gather` over NV / gk lists built by `push(st)`; the `carry`
variable is unused under SLVA) and stores `st = edit(t, h)` or h. A computed pause adds pi through `input_edit` to U_t and stores the core output h.
A copy-stall is `edit` returning St[t-1] (stored and carried).
- **b** (bank index) at pass t = number of readable entries = t (the prefix INSERT arm keeps t absolute), so b - 1 = t - 1.
- **a** (lineage age of the newest entry): a(St[0]) = 0; a(St[s]) = a(St[s-1]) + 1 if St[s] is pass s's core output; a(St[s]) = a(St[s']) if St[s]
  is a copy of St[s'] (copy-stall; prestall copies of V_0 have a = 0). Under SLVA the input mixes all entries; a follows the newest-entry chain
  (the SLVA query comes from N(V_{t-1}); c > 0 favours it). This is a DEFINITION; that the trained network's clock is a is **[ASSUMED]**
  (N1 evidence via FOREIGN; D3-T untested until S0-2 / S0-3).
- **rho** = #R before t in the target index map.

On every pass with unfinished input (brute force over 101,120 schedules with C / P / I mixes, `d0_d1_check.json`: 0 violations):
b - 1 = rho + S_C + S_P and a = rho + S_P, where S_C, S_P count copy-stalls / pauses before t. Hence **[PROVED]**:
D3-T (no C, P): b - 1 = a = rho. Copy-stall only: b - 1 = rho + S, a = rho (decouples b only). Computed pause only: b - 1 = a = rho + P_t
(decouples b and a, not rho). LONG-K adds only I passes, so P(F=0 | t >= 9) stays 0 and it adds zero-loss passes for "t >= 9 => finished" [PROVED];
"LONG-K alone can entrench EC" stays [DERIVED -> CONJECTURE] (it is a statement about learning).
Max t + r on R passes = n + #(C,P before the last R) + 1 <= n + P + 1: 9 / 17 / 21 [PROVED, brute force 0 violations].

**Caveats D1 does not cover (tag [CONJECTURE]):** (i) a third clock readable from the bank, the number of DISTINCT versions, equals rho + 1 under
both copy-stalls (bitwise duplicates) and HOLD pauses (decode duplicates). It is a progress clock, zero-loss by D0(a), and neither lever decouples it.
(ii) b is directly readable: `bias_row` / SLVA give V_0 a constant beta0 and the others beta + gamma * age over t entries, so V_0's softmax share
falls with t. The table's "moves b: yes" is then a moving readable clock, not a hidden one.

## 4. Prop D2 (information values, independence, coverage)

- I(t; rho), I(t; F), I(t; M): recomputed exactly (table in section 1). D4's numbers are the free/uniform values to MC noise: 2.36 / 0.96 / 0.87
  (exact 2.357 / 0.965 / 0.883), 0.281 / 0.179 / 0.213 (exact 0.281 / 0.179 / 0.212), I(t;M) 0.036 / 0.018 (exact 0.0346 / 0.0173). **[VERIFIED]**
  Under lastR they are 0.938 / 0.818, 0.155 / 0.162, I(t;M) **0.048** / 0.025.
- **P independent of n [PROVED at item level]** (P ~ U{1..Pmax} drawn without n). It is NOT independent under pass weighting or conditional on
  "unfinished at t": e.g. F = 0 at t = 16 forces n + P = 16. "Leaks no size" is item-level only. Harmless: at a pass, P_{<t} = t - 1 - rho.
- **I(t; M) small is not evidence against the marker route.** The MARKER-GATED count equals rho exactly on unfinished passes (section 2), whatever I(t;M) is.
  I(t;M) only bounds how well a single pass's marker predicts t. Reword "the marker carries almost no pass-index information" to say only that.
- **Coverage** (R passes at (t, r = n - t + 1) per 1M items, A free, exact): n 12 t 5 = 6,585; n 9 t 9 = 56,863 (D4: 6.7k / 57k) **[VERIFIED]**.
  (t, r >= 9) is 0 by D0 **[PROVED]**. Under lastR, n 9 t 9 is 62,500. Coverage is in (t, r) = (a + 1, r), NOT in the full descriptor: a training R
  pass at (t, r) has rho = t - 1 - P_t < t - 1, while the test state has rho = t - 1. So (a, r) is covered and (rho, r) is not, by D0.
  The inductive-bias bet in D4 §3 is exactly that this gap does not matter **[CONJECTURE]**.

## 5. Lemma D3 (SLVA history gradient under LONG-K)

**Code (boxes/slva.py `gather`):** U_t[j]_h = sum_{s<t} alpha^h_{j,s} N(V_s)[j]_h, with alpha = softmax_s(l) and
l = q.k/8 + beta0 1[s=0] + gamma (t-1-s) 1[s>=1] - c 1[s != t-1]; masked entries get alpha = 0.
U_t is exactly a convex combination with no residual term, so the identity applies as written.

- **Identity [PROVED]**: dU/dl_s = alpha_s (N_s - U), hence dL/dl_s = alpha_s <dL/dU_t[j]_h, N(V_s)[j]_h - U_t[j]_h> for ANY downstream loss.
  - Autograd check (`lemma_d3_check.py`): max error 3.6e-15 on a generic nonlinear loss, and 6.7e-16 through the exact SLVA logit / read algebra.
  - Scope: these are the SLVA read's logits only. The core and coda layers also attend over the bank (`bias_row`'s beta / beta0 / gamma rows), and the Lemma says nothing about those history weights.
  - The chain to the parameters (W_q, W_k, LN, gamma, beta0) is via dl/dtheta, which is bounded.
- **"At an idle pass with drift-free post-completion versions, the bracket is 0 for s >= n" [PROVED UNDER an extra condition]**:
  - the bracket is N_fin - U = sum_{s'<n} alpha_{s'} (N_fin - N_{s'});
  - it vanishes only if the read puts ~0 mass on pre-completion versions at that slot, OR the slot's content did not change after its last write. Copy slots meet the second condition, and there the bracket is 0 for every s >= last-change(j), not only for s >= n;
  - for s < n on changed slots, the bracket is the content difference and descent lowers those logits (anti-history) **[PROVED, sign]**.
- **O(delta^2) [PROVED UNDER the local model]**:
  - the model: L(U) = 1/2 (U - c*)^T H (U - c*) + O(|U - c*|^3), with the optimum at the clean content c*, and N_s = c* + delta_s;
  - then dL/dl_s = alpha_s dbar^T H (delta_s - dbar) + O(delta^3), with dbar = sum alpha delta_s;
  - numerically, the gradient's log-log slope in the drift scale is 1.994 -> 2.000, and the remainder's slope is exactly 3.000;
  - so "O(||H|| delta^2)" is right, with an O(delta^3) remainder that D4 should state.
- **"The gradient points toward the least-drifted versions" [FALSE in general: counterexample]**:
  - descent on l is gradient descent of 1/2 dbar^T H dbar over the simplex. It moves mass toward the versions whose drift is least ALIGNED with the mean drift (in the H metric), i.e. toward cancellation;
  - counterexample: H = I, alpha = (0.9, 0.1), delta_1 = +0.01 e, delta_2 = -0.02 e. Descent RAISES version 2, the more drifted one (grad_l = (+1.9e-5, -1.9e-5));
  - the claim is true **under same-sign co-linear (accumulating) drift**: with delta_2 = +0.02 e, descent raises version 1;
  - fix: "toward the versions whose drift cancels, or is least aligned with, the mean read drift; these are the least-drifted versions when drifts accumulate".
- **"D3-T: LONG-K history gradient ~ 0" [DERIVED from MEASURED]**. The dominant reason is dL/dU ~ 0 at training loss 0.0000. The O(delta^2) form adds that drift alone generates only a second-order signal.
- "Anti-drift reads must be trained on aged unfinished states" is **[CONJECTURE]**.

## 6. Lemma D4 (drift vs budget)

**The model:** delta_t = A_t delta_{t-1} + eps_t, with ||A_t|| <= rho (rho < 1, or rho = 1) and ||eps_t|| <= eps.
Symbol clash: this rho is a contraction factor, not progress. Rename it (e.g. lambda).

- **The bound [PROVED]**:
  - delta_t = Phi_{t,0} delta_0 + sum_u Phi_{t,u} eps_u with ||Phi_{t,u}|| <= rho^{t-u}, so ||delta_t|| <= rho^t ||delta_0|| + eps (1 - rho^t)/(1 - rho);
  - it is tight: an adversarially aligned eps attains it (max ratio 1.000, 0 violations; `lemma_d4_d5_check.py`).
- **(a) [FALSE as worded; fix]**:
  - the uniform-in-K bound is max(||delta_0||, eps/(1-rho)), because the bound is a convex combination of the two;
  - "bounded by eps/(1-rho)" needs ||delta_0|| <= eps/(1-rho), e.g. delta_0 = 0 (V_0 exact);
  - counterexample: rho 0.9, eps 0.01, ||delta_0|| = 5 gives ||delta_1|| = 4.51 > eps/(1-rho) = 0.1;
  - "no drift failure at any K iff the drift is below R's margin m" is **[PROVED UNDER a fixed-margin reader]** (failure iff ||delta|| >= m, with m state-independent).
- **(b) [PROVED]**:
  - at rho = 1, ||delta_t|| <= ||delta_0|| + t eps;
  - RMS: E||delta_t - Phi delta_0||^2 <= t sigma^2 D for martingale-difference eps (zero mean, independent of the past), with equality for orthogonal A_t. Numerics: RMS / (sigma sqrt(tD)) = 0.996-1.000;
  - this is an RMS bound, not an almost-sure one. D4's "sqrt(t) sigma sqrt(D)" is this bound.
- **(c) [PROVED UNDER the anchor model]**:
  - the model: delta_t = A_t((1 - alpha_t) delta_{t-1} + alpha_t delta^anc_t) + eps_t, with ||A_t|| <= 1, alpha_t >= alpha0, and ||delta^anc_t|| <= delta_a independently of the recursion;
  - induction: with B = max(||delta_0||, delta_a + eps/alpha0), x_t <= B - alpha_t (B - delta_a) + eps <= B;
  - so sup ||delta|| <= max(||delta_0||, delta_a + eps/alpha0). D4 omits the delta_0 term;
  - the bound is tight: numerically, the limit equals delta_a + eps/alpha0;
  - the anchor's drift must be bounded independently. An anchor taken from the same drifting recursion makes the bound circular;
  - on an unfinished state, (c) is valid only for slots whose content has not changed since the anchor version.
- **(d) [DERIVED, contrapositive valid; conclusion CONJECTURE]**:
  - if A, eps and the margin m are all state-independent, then the drift law at pass t is the same for in-range and beyond-range items, so the failure rates at the same t would match;
  - in-range exactness, measured through K' 24, refutes that. So at least one of A_t, eps_t or m depends on the state;
  - D4 says only "eps depends on the state". It should say "A, eps or m";
  - "a support effect, answered by coverage" is **[CONJECTURE]**.

## 7. Lemma D5 (TBPTT bias)

**Setup:** L = sum_t l_t and g = sum_t sum_{s<=t} J_{t,s}, where J_{t,s} is the contribution of l_t through the use of theta at pass s.

- **The assumption is stated on the wrong object.**
  - D4 assumes ||prod dV_u/dV_{u-1}|| <= rho_b^{t-s}.
  - In VERSION-BANK, pass t reads the stored entries St[s] DIRECTLY: through the SLVA read, through every core layer's bank attention and through the coda's. `push` computes the bank K / V WITH gradient. So J_{t,s} has one-hop paths, weighted by attention mass, that a chain-Jacobian bound does not control.
  - **Fix:** assume ||J_{t,s}|| <= G rho_b^{t-s} on the TOTAL derivative, i.e. the off-newest bank mass at lag t-s must decay too. That is an assumption about the trained model **[ASSUMED]**.
- **The bound for sliding-window TBPTT-k (drop lags >= k) [PROVED]**:
  - ||g - g_k|| <= G sum_{t=k+1}^{K} sum_{l=k}^{t-1} rho_b^l <= (K-k) G rho_b^k/(1-rho_b) <= K G rho_b^k/(1-rho_b);
  - exact sums (`lemma_d4_d5_check.json`): the D4 formula is a valid bound but very loose as rho_b -> 1. At K 20, k 12, rho_b 0.99 it gives 1773 G against an exact 31.2 G;
  - at rho_b = 1 the bound is G (K-k)(K-k+1)/2 = 36 G (K 20) / 78 G (K 24), exact for s in 1..t;
  - use the min of the two bounds.
- **"Relative to the same upper bound on ||g||, <= rho_b^k" [PROVED as a ratio of bounds; FALSE as a relative error]**. Near the base optimum ||g|| can be far below its upper bound (training loss 0.0000), so ||g - g_k|| / ||g|| is not bounded by rho_b^k.
- **"At A's worst case (Kb 20, k 12), only lags 12-19 are cut" [FALSE for chunked TBPTT]**.
  - The statement holds for sliding-window TBPTT only.
  - The usual implementation, and the one G-P6's "zero gradient across the cut" describes, detaches every k passes. Every pair (t >= 13, s <= 12) is then dropped: lags 1-19.
  - At Kb 20: 96 of 210 (t, s) pairs are dropped (46 %), against 36 (17 %) for sliding. The bias-bound share at rho_b 0.9 is 0.30 chunked vs 0.067 sliding.
  - The rho_b^k decay is lost: at rho_b 0.5 the share is still 0.052.
  - At Kb 24: 144 of 300 pairs.
  - Fix: say which TBPTT is meant. Under chunked TBPTT, the losses at passes 13-20 keep their within-chunk (local) credit, which is the credit D4's argument needs.
- **Gate fix (G-P6):**
  - a cut must detach the bank entries St[<= cut] (their K / V, keys and NV) as well as the carry;
  - otherwise gradient crosses the cut through the bank reads and "zero gradient across the cut" fails;
  - and per-chunk backward is needed for any memory saving (section 8).
- "Why it is still acceptable" is **[CONJECTURE]**, not DERIVED: which credit the deadline removal needs is a claim about the learned solution.
- Variance stays [ASSUMED].
- The literature (Huginn 2502.05171: k = 8 at mean recurrence 32, backprop through the last k iterations) is [EMPIRICS]. It is not re-verified here (Literature's role).

## 8. (v) Cost and memory

Checks: `oc_cost_check.py` / `.json`; the Kb values are the exact laws from section 1.

| item | D4 | check | verdict |
|---|---|---|---|
| E[Kb] D3-T / A (per batch) / B (per batch) | 11.8 / 15.2 / 17.9 | exact 11.80 / 15.19 / 17.78; per item 17.90 / 23.17; augmented batches 18.59 / 23.76 | **[VERIFIED]** (B 17.9 is MC noise; exact 17.78) |
| s/step, augmented batches | A ~0.70, B ~0.9 | linear in Kb: A 0.64, B 0.82; quadratic: A 1.01, B 1.65 | **[ASSUMED]**, but plausible. Rough FLOP count: at T ~ 40 and t = 20 the bank attention is ~18 % of a core layer's dense FLOPs (~9 % averaged over passes), so the scaling is near-linear and D4's figures are mildly conservative. |
| 12k FT-A | $0.40-0.62 | with D4's own 0.55 s/step: 2.25 h -> **$0.40-0.58**; linear-quadratic band $0.39-0.72 | arithmetic slip at the upper end (+$0.04) |
| 12k FT-B | $0.47-0.68 | $0.46-0.67 (band $0.44-1.00) | **[VERIFIED]** arithmetic |
| FT-CTRL | $0.32-0.46 | $0.32-0.46 | **[VERIFIED]** |
| Stage 1a projection (A) | $0.72-1.08 | $0.72-1.04 (band up to $1.18) | under the $1.5 cap in every case |
| 48k cell | A $1.30-1.87, B $1.65-2.39 | arithmetic $1.29-1.86 / $1.65-2.38 | arithmetic **[VERIFIED]**. The 0.50 s mean (< 0.55 FT mean) assumes pauses cost less in early curriculum stages; with pauses in every stage the augmented Kb is ~n_max + 12, so 0.50 is optimistic **[ASSUMED]** |
| memory split | 19.6 (A, Kb 20 full) / 24.4 (B, Kb 24 full) | 2 + 0.58 Kb + c sum(t+1), c = 0.0258 fitted to 11.28 at Kb 12: **19.53 / 24.27** | arithmetic **[VERIFIED]**; the split itself is **[ASSUMED]** |
| TBPTT-12 memory | ~13 (A) / ~16 (B) | peak over chunks with per-chunk backward: 11.28 (A) / 14.99 (B) | D4's values are conservative **only if** backward runs per chunk AND pre-cut bank entries are detached. A single backward at the end saves nothing. **State the implementation.** |

## 9. §5 OC block

Checks: `oc_cost_check.json`, `dgone_shift_check.py` (D4's own `d4_edge_collapse.read`, imported verbatim, on synthetic trajectories).

- **P(DG) ~ 0.2 per FT-PAUSE cell [arithmetic VERIFIED; inputs ASSUMED]**:
  - 0.45 x [0.35, 0.45] = [0.158, 0.203];
  - with the no-effect leak 0.55 x 0.05 it is [0.185, 0.230].
  - Assumed: DG depends on the FT only through E = "deadline removed", and P(DG | not E) = 0.05.
  - The 0/18 base rate supports anything up to 0.15 (Clopper-Pearson 95 % upper 0.153; rule of three 0.167), and the 18 are not independent. So 0.05 is a prior, not a measurement.
- **P(D4-C2) ~ 0.15 [needs stated assumptions]**:
  - D4-C2 = DG and LF and VB-C1 and K24, and FT-DIVERGED censors every reading;
  - under independence with P(LF) = 0.85: P(C2) = [0.157, 0.196] x P(VB-C1) P(K24) P(not DIVERGED), which gives 0.13-0.16 at 0.9 each and 0.14-0.18 at 0.95 each;
  - so ~0.15 silently assumes **independence** of DG and LF, and P(VB-C1 and K24 and not DIVERGED) ~ 0.8-0.9;
  - without independence, the Frechet bounds on P(DG and LF) are [0.035, 0.185] to [0.08, 0.23];
  - the dependence is plausibly NEGATIVE **[CONJECTURE]**: removing a late-pass deadline changes exactly the late-pass behaviour that NO-KNEE / LFK read;
  - state the independence assumption and the three omitted factors.
- **DEADLINE-GONE OC: P(pass | no effect) <= 0.05 [TRUE only for a narrow "no effect"]**:
  - sampling false positives are negligible: P(h_hat(7-9) <= 0.10 | true h 0.30, 300 events) = 7e-17. At true h 0.15 it is 0.007, and at 0.12 it is 0.16;
  - so the bound holds iff the no-effect world has h(7-9) >= ~0.15, i.e. **the base is EC**. D4's own prior gives P(D3-T EC) = 0.6;
  - under a non-EC base (P 0.4), the FT can read DEADLINE-GONE with no effect at all;
  - fix: make DEADLINE-GONE count as mechanism evidence only if the base reads EC (S0-1) or FT-CTRL reads not-DEADLINE-GONE. A control must be exposed.
- **DEADLINE-GONE is met by a SHIFTED deadline [FALSE as a removal test]**. Running D4's reader on synthetic trajectories gives:

  | model | DEADLINE-GONE |
  |---|---|
  | SOFT(Delta): N1's hazard shifted by Delta passes | met for Delta >= 4 (h(7-9) 0.029 at Delta 4) |
  | HARD(t_d): one reduction per pass, then collapse at pass t_d | met for t_d >= 12 |
  | 9 + E[P] = 13.5 (A) | met |
  | 15.5 (B) | met |
  | the zero-loss caps 16 / 20 | met |

  - Its two clauses (h(7-9) <= 0.10; t_fin(12) - t_fin(10) >= 2) only see t <= 12.
  - Rename it NO-DEADLINE-BY-12, or add a shift clause (section 10).
  - P(DEADLINE-GONE | shifted, not removed) ~ 1.
- **The pre-stated shifts:**
  - AGE-CLOCK (0.6): P(DG) 0.23-0.29;
  - PROGRESS-ONLY (0.2): 0.11-0.13;
  - not-EC and not-ORD (0.3): 0.14-0.17.
- **The 48k cell:**
  - P(entry) 0.7 x P(DG | entry) 0.2 = 0.14 unconditionally;
  - but 1b runs only on FT outcomes (DEADLINE-GONE, or UNGATED and PAUSE-LEARNED), and the two cells share the mechanism. Its prior should be conditional on the gate and is positively dependent on the FT result;
  - family odds are not additive;
  - P(FT reads DEADLINE-GONE) = 0.45 x 0.8 + 0.55 x FP = 0.36 (FP 0) to 0.50 (FP 0.25).
- **Censoring:**
  - rule (b) is respected: never-reached events count as failures and t_fin = 99 when never finished;
  - median t_fin is well defined only while < 50 % of items never finish. Say so for n >= 13.

## 10. Audit's shifted-deadline falsifier

**The proposal:** F_AUDIT = OLS slope of median t_fin on n over n 9-14 >= 0.7, AND spread (max - min of the medians) >= 3.

Checks: `falsifier_check.py` / `.json`. The models of plain-pass test behaviour:

| model | definition |
|---|---|
| CONTENT(q) | one reduction per pass, plus a state-independent over-reduction with probability q |
| HARD(t_d) | constant output deadline at pass t_d |
| SOFT(Delta) | N1's measured hazard shifted by Delta passes, with N1's measured drop-size law (2: 0.62, 3: 0.25, 4: 0.10, ...) |

SOFT(0) reproduces N1's medians to within 1 pass: [8, 9, 9, 9, 10, 10, 11, 11] vs the measured [8, 9, 9, 9, 9, 10, 10, 11].

| model | medians n 9..16 | F_AUDIT slope / spread | F_AUDIT | slope n 12-16 |
|---|---|---|---|---|
| N1 measured | 8 9 9 9 9 10 10 11 | 0.29 / 2 | fail | 0.50 |
| N2 measured | 9 x 8 | 0 / 0 | fail | 0 |
| CONTENT q 0 / 0.05 / 0.10 | n / n / n - 1 | 1.0 / 5 | **pass** | 1.0 / 0.8 / 1.0 |
| SOFT(Delta 3) | 9 10 11 11 12 12 12 13 | 0.60 / 3 | fail | 0.4 |
| SOFT(Delta 4.5 = E[P_A]) | 9 10 11 12 13 13 13 14 | 0.86 / 4 | **pass** | 0.4 |
| SOFT(Delta 6.5 = E[P_B]) | ... 15 15 | 1.0 / 5 | **pass** | 0.8 |
| HARD(12) | ... 12 12 12 | 0.63 / 3 | fail | 0 |
| HARD(13.5 = 9 + E[P_A]) | ... 13.5 | 0.93 / 4.5 | **pass** | 0.35 |
| HARD(15.5 = 9 + E[P_B]), HARD(16), HARD(20) | n (to 15.5) | 1.0 / 5 | **pass** | 0.9-1.0 |

- **Achievable under the PAUSE support (D0)? Yes [PROVED].**
  - A content-tracking network is zero-loss under every schedule and gives slope 1, spread 5. F_AUDIT reads t_fin, not correctness, so the off-support early passes (r >= 9) do not prevent it.
  - It is robust to state-independent over-reduction (q 0.10 still gives slope 1).
  - Stronger: at the training optimum, an output-keyed constant deadline must sit at t_d >= 16 (A) / 20 (B) (section 1). **Every** zero-loss constant-deadline network therefore PASSES F_AUDIT.
  - The zero-loss networks that FAIL it are exactly D0b's history- or marker-gated deadlines, which collapse at ~9 on pause-free test histories (N1-like slope ~0.3).
- **Does it discriminate a shifted constant deadline (~9 + E[P]) from content-tracking finishing? No [FALSE: counterexample].**
  - HARD(13.5) and SOFT(4.5) pass it, as do B's 15.5 and both caps.
  - The smallest constant deadline that passes is t_d = 12.25. Over n 9-14, min(n, t_d) has slope >= 0.7 once t_d >= 12.25.
  - F_AUDIT is a valid falsifier of "a deadline near 9-12 survives (gated or not)". It does not test a shift.
- **Repair (pre-state all three; [DERIVED] from the models above):**
  - (i) Keep F_AUDIT as the near-9 / D0b-gated falsifier.
  - (ii) A-shift test: slope over **n 12-16** >= 0.7 on plain passes. This rejects HARD(13.5) (0.35) and SOFT(4.5) (0.4) and passes CONTENT (0.8-1.0). It cannot reject B's 15.5 or either cap, because no plain test item has n > 16.
  - (iii) A deadline at 16 (A) / 20 (B) is unobservable on plain passes at n <= 16. Use START-D with large D (eval-only markers, in-distribution under PAUSE-LEARNED):
    - the design: n 9-12, D in {0, 2, 4, 6, 8} (A; up to 12 for B), K' = n + D + 2, and regress median t_fin (bank index) on n + D over cells with n + D in [12, 20];
    - the readings: CONTENT 1.0, HARD(13.5) 0.14, HARD(16) 0.56, HARD(20) 1.0 (so B needs n + D up to 24);
    - D4's START-D uses D in {1, 4}, so n + D <= 16, which is too short;
    - alternatively, add eval-only items with n 17-20 (`generate` takes any n).
  - Together, (i)-(iii) separate CONTENT (passes all), a shifted age / bank deadline (passes (i), fails (ii) or (iii)), a gated deadline at 9 (fails (i), unlocks under (iii)) and the unchanged deadline (fails all).

## 11. Silent assumptions about the trained network (tagged)

- [CONJECTURE] The deadline is keyed on lineage age a (D1 proves only the counting; D3-T is untested until S0-2 / S0-3).
- [CONJECTURE] No content-visible progress clock is used. Candidates are the count of distinct versions, the VOID count, or the size n read from the slot count; all are zero-loss by D0 and none is decoupled by pauses.
- [ASSUMED] A small positive loss mass blocks a late deadline: 5 % at t >= 16 input-keyed, 0 output-keyed at 16 (A). "Pays loss => not learned" is an optimum statement.
- [CONJECTURE] D0b / MARKER-GATED is learnable or not (zero loss proved).
- [ASSUMED] Lemma D5's total-derivative decay; Lemma D4's fixed margin; Lemma D3's quadratic local model.
- [CONJECTURE] Covering (a, r) without (rho, r) suffices (the inductive-bias bet).
- [ASSUMED] The OC independence (DG vs LF, VB-C1, K24, not FT-DIVERGED).
- Minor: §2.2's "h <= 0.02 in N1" at t <= 4 is 0.021 at t 4.

## 12. Verdict summary

| claim | verdict |
|---|---|
| D0 rho + r = n <= 8; D0(a) | **PROVED** (state the stronger "rho + r >= 9 never in support") |
| D0(b) history-gated deadline zero-loss | **PROVED UNDER** all non-reducing passes marked (FALSE with unmarked copy-stalls); = the MARKER-GATED clock |
| D1 counting (b, a, rho per lever; max t + r 9 / 17 / 21) | **PROVED** (brute force, 0 violations) |
| D1 table | A row VERIFIED; **B row FALSE at t* 17 / 20** (per-pass values printed): correct 0.11 / 0.013 |
| "deadline at t* <= 16 / 20 pays loss" | input-keyed PROVED; **output-keyed zero-loss from 16 (A) / 20 (B)** |
| §2.1 vs §2.2 completion pass | **FALSE: inconsistent** (trailing pauses 31 % A / 54 % B). Pick lastR or free; G-P3 for the implemented one (lastR I(t;M) 0.048) |
| D2 information values, coverage | **VERIFIED** (exact). I(t;M) is not evidence against the marker route |
| D3 identity | **PROVED** (autograd 1e-15) |
| D3 O(delta^2) | **PROVED UNDER** the quadratic model (+O(delta^3)) |
| D3 "toward least-drifted" | **FALSE** (cancellation counterexample); true for accumulating drift |
| D4 bound, (b), (c) | **PROVED UNDER** the stated model; (a) and (c) need the delta_0 term |
| D4(d) | contrapositive valid (A, eps or m state-dependent); "coverage answers it" CONJECTURE |
| D5 bound | **PROVED** for sliding-window, on the total derivative (the chain-Jacobian assumption is insufficient with a bank) |
| D5 "only lags 12-19 cut" | **FALSE for chunked TBPTT** (lags 1-19; 96 / 210 pairs); the relative bound is a ratio of bounds |
| (v) cost / memory | arithmetic VERIFIED (FT-A upper end $0.58, not $0.62); TBPTT memory needs per-chunk backward and bank detach |
| OC P(DG) ~ 0.2, P(C2) ~ 0.15 | arithmetic OK; needs independence and omitted factors stated; DEADLINE-GONE FP bound needs an EC base |
| DEADLINE-GONE as a removal reading | **FALSE**: met by shifted deadlines (Delta >= 4 / t_d >= 12) |
| Audit F_AUDIT | achievable; **does not discriminate** a shift to 9 + E[P] (passes from t_d 12.25); repair in section 10 |
