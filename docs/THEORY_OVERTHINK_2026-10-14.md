# THEORY: overthinking, halting (EXIT) and latent <-> text transitions (ENTRY / EXIT): derivation and test plan (2026-10-14)

Opus subagent at the user's direction ("also try to fix over thinking decreases acc"; "teach models when to transition in and out of latent thinking"), reviewed by Theory. Claim tags are the author's, and every number is POST-HOC. CPU scripts are in theory/overthink201/. Registered tests: THEORY §201.

**Theory's review notes:**
* Chung 2605.07755 WAS verified by Theory against its arXiv abstract (2026-10-13); Literature to add it to a LIT note.
* The claim that note 173-4's RMS n 10 per-pass accuracies read one pass late is **unverified**; it is checked before it is used anywhere.
* The ProsQA depth-3 point is verified (correction 200-3).

---

# OVERTHINK / EXIT / ENTRY: why extra passes hurt, when to stop, and when to write (PLAN, 2026-10-14)

Theory subagent, for the Theory session. Read-only on D:\Latent. CPU only (1 thread, nice 19). Nothing was launched. Scripts and outputs are in this folder:
- `kprofile.py` → `kprofile.json`: K′ profiles and acc_by_t from existing eval files;
- `voh.py`: value of halting and horizon maps;
- `beta_by_n.py`: 199-E1 gates by n and pass.

**Tags.**
- **PROVED**: a complete argument, given here, under the stated assumptions.
- **DERIVED (A…)**: follows from named assumptions that are not verified for our models.
- **CONJECTURE**: an argument, not a proof.
- **DATA**: a number read from an existing file. **Every DATA item is POST-HOC**: final models, read after the outcomes were known. None of it is a test.

---

## Summary (relayable, about 250 words)

**Why extra passes hurt (derivation; data POST-HOC).**
- **The additive heads have no fixed point (PROVED).** After completion each pass adds a constant-size write, 30-45% orthogonal to the state, so the RMS read turns toward the write's own direction at about 1/t per pass (exact law).
- **The delta heads have a fixed point for frozen inputs (PROVED),** and converged models learn a hold at t = n+1 (199-E1). But the gate reads the memory: β re-rises after n+1, and the delta increment ratios follow (t-1)/t (0.86 / 0.89 / 0.91 vs 0.90 / 0.91 / 0.92, rms s1). The delta state is being driven by the additive drift.
- **Passes past K get zero gradient.** Each trained pass after completion cuts the K'16 loss about e-fold (rms s1: 0.24 -> 0.004).

**EXIT.**
- Halting is quickest change detection: stop once P(done) >= p*C_FA / (p*C_FA + c_d), about 0.95-0.98 here.
- Answer settling carries no information (the default answer is stable before completion), hence PREMATURE.
- Freezing all writes is exactly halting (PROVED); freezing only the additive writes is a softer halt.
- The β-hold never sees the pass token, so it is a candidate done cue.

**ENTRY/EXIT to text.** A checkpoint pays once the latent passes go past their reliable horizon. One checkpoint reaches n <= H_latent + H_decode ≈ 14.

**$0 tests:** T1, a 2×2 freeze (additive or delta writes after completion), decides the channel. T2 scores a pre-registered halt head on frozen models against the best fixed K' and K'32, with censoring explicit. T4 tests text checkpoints.

**Caveat.** A tuned fixed K' is within 0.03 of the per-n oracle. Halting buys robustness (K'32 loses 0.3-0.6) and compute.

---

## 0. Setup and sources

**The pass map** (boxes/deltamem.py, read):
- Each pass, on the identical think input x, reads S at block 6 (read_fixed: queries from LN₆(h₆), so pass-invariant), runs blocks 6–11, then writes:
  - additive heads: A ← A + Σ_τ φ(k_τ)v_τᵀ, with φ = elu + 1 > 0, no decay;
  - delta heads: Δ ← α_τΔ(I − β_τk̂_τk̂_τᵀ) + β_τv_τk̂_τᵀ, with α = σ(·) and β = 2σ(·).
- The write inputs are u = LN_w(h_w(S)) once γ = 1. **So k, v, α and β all depend on the read of S.**
- The answer decoder ([EOT, expr, " ="], dec_logits) and the step aux decoder read only S. Memory is the only cross-pass channel (C1).

**Training** (boxes/nested170.py, read):
- Fixed K: the answer is read at S_K; the aux is read at passes t ≤ n.
- WINDOW / HOLD: K_i = n + U{1..4}, and S is frozen after K_i by the masked select in kmatch_aux_loss / kmatch_loss.
- HOLD adds the target e_{n+1} on passes n < t ≤ K_i.

**Completion pass.** τ = n for nested: acc_by_t jumps from about 0.1 to about 1 at t ≈ n−1..n in every converged model (DATA, kprofile.json). The first post-completion pass is n+1.

**Sources used (DATA, POST-HOC):**
- eval.json `s170.acc_by_n_kprime` and `eval_s173_halt.json` acc_by_t for 16 nested runs;
- §191 A `s168s2.acc_by_depth_kprime` / normcap / aux_done;
- 199-E1 gate tables (theory/seedstab199/emp/emp_*.json);
- notes 173-2/3/4, 182 results, 184-1, 185, 190-2 and 191 in docs/THEORY.md.

---

## Part I. Why accuracy decays after completion in this architecture

### I.1 The additive integrator

**Prop 1 (no fixed point; PROVED).**
- Per additive head, one pass writes W = Φ_M V_Mᵀ. Φ_M ∈ ℝ^{64×T} holds the columns φ(k_τ) of the real tokens and is entrywise > 0. V_M holds the value vectors.
- (i) S_add is fixed by a pass iff W = 0. If Φ_M has full column rank (generic for T ≤ 64; nested think inputs have T ≈ 8–40), W = 0 iff v_τ = 0 at every real token.
  - So a post-completion fixed point of the additive state needs the trunk to emit all-zero values on a terminal state: a "done ⇒ silence" computation that no loss term asks for.
  - On ProsQA (T ≈ 450 > 64) cancellation is possible but not generic.
- (ii) Under `add_norm rms` the decoder-visible additive read N(φ(q)ᵀA) is invariant iff φ(q)ᵀW = c_q·φ(q)ᵀA with c_q > −1, for every query q in the (pass-invariant) think and decoder query sets.
  - Call this a **radial write in read space**. It is reachable (W = cA), so under RMS a read fixed point exists in the parametrisation. It needs every post-completion write to be aligned with the accumulated state.
- Proof: (i) Φ_M has trivial null space, so each column of V_Mᵀ is 0. (ii) N is invariant to positive rescaling only. ∎

**Prop 2 (drift law under a constant post-completion write; PROVED, exact).**
- Per head and query, let u₀ = φ(q)ᵀA_T be the read at the last trained or completion pass T.
- Let each later pass add the same b = φ(q)ᵀW. Split b into b_∥ and b_⊥ relative to û₀, with ρ_∥ = b_∥/|u₀| and ρ_⊥ = |b_⊥|/|u₀|.
- After j passes the read direction û_j makes the angle θ_j with û₀, where

  tan θ_j = j·ρ_⊥ / (1 + j·ρ_∥).

- So θ_j rises, reaches half of its limit near j ≈ 1/ρ_∥, and saturates at θ_∞ = ∠(b, u₀). **The read converges to the direction of the write itself, not to the terminal code.**
- Since |u₀| ≈ T|b̄| for coherent writes, ρ_∥ ≈ 1/T: the drift depends on the ratio K′/T (scale-free in pass count).
- Proof: u_j = u₀ + jb. ∎

**Lemma 3 (margin; DERIVED under A_L: the decoder margin is L_R-Lipschitz in the post-W_o read).** Per head, the RMS read has norm g√DH, so a turn by θ moves it by 2g√DH·sin(θ/2). The answer margin falls by at most L_R‖W_o^add‖·g_add√DH·Σ_h 2 sin(θ_h/2).

**DATA (POST-HOC; emp_*.json, items n 4..8, passes 9–12, all past completion):**
- The additive increments ‖A_t − A_{t−1}‖ are constant in size: rms s1 3,222 / 3,151 / 3,066 / 3,005, ratios 0.97–0.98 per pass; the same in all 8 models. So the integrator is neutral, step ratio about log 0.98.
- The share of each write orthogonal to the state, √(1 − cos²)/rel_inc, is 0.36 → 0.26 over t 9 → 12 (rms s1), 0.47 → 0.32 (s2), 0.58 → 0.40 (§190 s1).
- That decline is the Prop 2 signature: the state is rotating onto the write direction. A constant write predicts sin∠ ∝ 1/t, so 0.36 × 9/12 = 0.27, against 0.26 measured.
- The state-space turn per pass is about 0.5/t rad (cos values are rounded to 3 decimals, so ±30%).

### I.2 The delta heads: a fixed point exists, but it is slaved to the gate input

**Prop 4 (PROVED; Thm 4 of THEORY_SEEDSTAB + Banach).**
- For frozen inputs u, F_Δ(Δ) = ΔP(u) + C(u) has ‖P‖ ≤ Π_τα_τ < 1.
- So it has a unique fixed point Δ*(u) = C(u)(I − P(u))⁻¹, with ‖Δ_t − Δ*‖ ≤ ‖P‖^{t−t₀}‖Δ_{t₀} − Δ*‖.
- **The terminal code is a fixed point iff Δ_n = Δ*(u(S_n)).** That is self-refresh: each post-completion pass rewrites, along its keys, the content that the α-decay and the erasures remove.

**Prop 5 (the delta state follows a drifting input; DERIVED under A_u: u Lipschitz in the read, Δ* ℓ*-Lipschitz in u, open-loop retention ρ < 1).**
- e_t := ‖Δ_t − Δ*(u_t)‖ obeys e_t ≤ ρ·e_{t−1} + ℓ*‖u_t − u_{t−1}‖.
- (i) If u is frozen, the delta state relaxes geometrically, at ratio ρ, onto Δ*(u_n). That point is the terminal code only under self-refresh; otherwise it is a nearby wrong fixed point (pathway P3 below).
- (ii) If u keeps drifting with ‖Δu_t‖ ∝ 1/t (the additive read of Prop 2), then once the ρ^j transient has died the delta increments are ∝ 1/t. Their consecutive ratios tend to (t−1)/t, not to ρ.
- (iii) The cumulative delta drift then grows like log(K′/T): slowly, sub-linearly, and without bound unless the write direction saturates (Prop 2).
- So the delta channel adds no long-run drift of its own. It passes on the drift of its input, and **the only non-vanishing driver after completion is the additive integrator**, plus the decoders' own additive read.

**DATA (POST-HOC; 199-E1, beta_by_n.py):**
- **Learned hold.** β, averaged over the item's tokens and heads, falls pass by pass and reaches its minimum at t = n+1. rms s2: n 4 / 5 / 6 minimum 0.430 / 0.376 / 0.343 at t = 5 / 6 / 7.
- **Retention jumps at n+1:** rms s2 n 4: 0.744 → 0.869; rms s1 n 4..7: +0.081 / +0.078 / +0.063 / +0.050.
- **The hold erodes after n+1.** rms s2 β n 4: 0.430 → 0.552 by t 12; n 6: 0.343 → 0.436. Retention 0.873 → 0.842. Pooled β in rms s1 / s2 / §190 s1 rises from its minimum at t ≈ 9.
  - Exception: §184 s2's β keeps falling (0.582 → 0.568) and its retention is flat (0.834 → 0.830). It is the most stable fixed-K model at K′24 (n 8: 0.826, against rms s1 0.460).
- **The delta state follows the drift rather than relaxing.** Absolute delta increments past completion, ratios at t 10 / 11 / 12:
  - rms s1: 0.86 / 0.89 / 0.91; rms s2: 0.86 / 0.90 / 0.93. Prop 5(ii) predicts (t−1)/t = 0.90 / 0.91 / 0.92.
  - Pure relaxation would hold near ρ ≈ 0.80 (retention). This supports pathway P2 below; the causal test is T1.
- **Readers.** Every decoder reads mainly the delta state: delta dec-only ablation gives 0.09–0.21. Converged rms s1 / s2 compute almost entirely in the delta heads: additive think-read ablation leaves 0.88 / 0.76 late-pass aux (199-E1).

**Three pathways for post-completion corruption** (the T1 test separates them):
- **P1 (direct):** the decoders' own additive read drifts (Prop 2).
- **P2 (coupled):** additive drift → u drifts → β re-opens and v changes → Δ*(u_t) moves (Prop 5ii).
- **P3 (delta-intrinsic):** Δ_n is not self-refreshing, so it relaxes onto a wrong Δ*(u_n) (Prop 5i), driven partly by the α decay (α_eff^L ≈ 0.78 per pass at n 8).
- **Predicted ranking for converged RMS / NOPASS models (DERIVED):** P2 > P1 > P3.
- **For §184 / §175 (unnormalised additive read): P1 is larger.** NORM-CLAMP on §175 recovered about 80% by fixing the magnitude alone (DATA).

### I.3 Why training up to K leaves the post-completion map unconstrained

**Prop 6 (PROVED, trivially).**
- In every recipe the loss is a function of S_1..S_{K_i} only: the masked select freezes S after K_i, and the answer is read at S_{K_i}. So F evaluated on any state first reached at t > K_i gets zero gradient.
- With fixed K, the post-completion passes of item n enter the loss only through the single answer read at S_K, i.e. j₀ = K − n trained post-completion passes.
- For n = K there are none. And with no HOLD there is no post-completion aux target.

**Lemma W** (THEORY_GSM8K_TEMPLATE_DRAFT, PROVED under SEEDSTAB A2): the margin is at least μ − L_D·d·dist(j, J), where J is the set of trained read offsets.
- Fixed K: J = {K − n}.
- WINDOW: J = {1..4} for every n.
- Drift that is harmless inside J costs no loss, so **no gradient pushes it toward zero**. Stability beyond J comes only from the implicit bias of the learned map, which is seed-dependent (WINDOW K′32: s1 0.739 vs s2 0.996).

**The trained-horizon map (DATA, POST-HOC; voh.py)**, K′16 loss against K′8 by j₀ = K − n:

| model | j₀ = 0 | 1 | 2 | 3 | 4 | fit L₀·e^{−κ_h·j₀} |
|---|---|---|---|---|---|---|
| rms s1 | 0.242 | 0.076 | 0.047 | 0.018 | 0.004 | L₀ 0.24, κ_h ≈ 1.0 |
| rms s2 | 0.050 | 0.015 | 0.001 | 0.000 | 0.002 | — |
| §184 s1 | 0.091 | 0.074 | 0.061 | 0.023 | 0.012 | — |
| §190 s1 | 0.015 | 0.006 | 0.006 | 0 | 0 | — |
| ProsQA §191 (K′16 vs K′10, j₀ = 10 − d) | 0.337 | 0.260 | 0.162 | 0.079 | 0.029 (j₀ 5: 0.017) | L₀ 0.34, κ_h ≈ 0.6 |

**The link (DERIVED / CONJECTURE).**
- By Props 2 / 5(iii), the cumulative drift is about s·ln(K′/T). An item fails once that drift exceeds its margin μ, i.e. at K′_fail ≈ T·exp(μ/s).
- If each trained post-completion pass raises μ by a constant a, then the loss at fixed K′ falls about exponentially in j₀, as observed.

### I.4 The two-term model: why K′ a little past K helps and more hurts

**Model (DERIVED; assumptions: one completion per item; hazard 0 inside the trained horizon, rising past it).**

acc_n(K′) = c + (q_n − c)·C_n(K′)·S_n(K′)

- C_n = P(τ_n ≤ K′) is completion: about 1[K′ ≥ n] on nested.
- S_n = Π_{t=τ_n}^{K′−1}(1 − h_n(t)) is survival under the post-completion hazard h_n.
- For a depth mix π, a fixed K′ is optimal when the completion gain no longer beats the drift loss:

  Σ_n π_n(q_n − c)·[C_n(K′+1) − C_n(K′)]·S_n(K′+1) ≤ Σ_n π_n(q_n − c)·C_n(K′)·S_n(K′)·h_n(K′).

**§191 ProsQA, worked (DATA, POST-HOC; n_by_depth weights):**
- **K′10 → 12:** completion gain at depths 11–14 = +0.018 / +0.102 / +0.205 / +0.41 × (397, 326, 166, 34) ≈ 88 item-units.
  - Drift loss at depths 3–10 ≈ 44 units, mostly depths 8–10 (j₀ ≤ 2: −0.031 / −0.021 / −0.031).
  - Net +0.011: pooled 0.950 → 0.961.
- **K′12 → 14:** no completion mass is left (depths 13–14 already read 0.934 / 0.912 at K′12), and the j₀ ≤ 2 hazard jumps (0.10–0.25). Net −0.111.
- **So K′* = K + 2 for this mix**, as the inequality predicts. Nested is the same: RMS s1 n 10 peaks at t = 10 (0.87; acc_by_t indexed from t = 0, matching K′8 = 0.352 at t = 8) and falls to 0.45 at t = 16.
  - Note 173-4's "0.14 / 0.35 / 0.84 at t 8 / 9 / 10" reads one pass late against the same file. Flagged for Theory.

**Value of halting (DATA, POST-HOC; voh.py).** Per-item stopping attains the per-n oracle Σ_nπ_n·max_t acc_n(t). Its gain over the best single fixed K′ is small in every model:
- nested, mixed n 3..12: +0.00 to +0.026 (RMS s1 0.837 → 0.863);
- nested, deep n 9..12: up to +0.059 (RMS s1);
- ProsQA depths 3..14: +0.011 over K′12, +0.022 over the trained K′10.
- Against overthinking the gain is large: pooled n 3..8 at K′32 is 0.345 / 0.620 / 0.692 / 0.709 (§184 s1 / RMS s1 / §190 s1 / RMS s2), against ≥ 0.97 at K.
- **Consequence:** halting's value here is robustness when K′ is unknown or large, plus compute. It is not a higher peak. The pre-registered comparator must therefore be the best fixed K′ **chosen on training items**, not the trained K.

**A side correction (DATA, POST-HOC; for the GSM8K draft's risk 0).** ProsQA depth 3 reads 0.930 / 0.945 / 0.955 / 0.955 / 0.960 at K′ 4 / 6 / 8 / 10 / 12. It is flat from K′4, so its deficit is not drift from idle passes. On this profile the IDLE-PASSES risk is unsupported for ProsQA, though still worth measuring on GSM8K.

### I.5 What this implies for fixes (summary)
- **The root is the additive integrator's post-completion writes (Prop 1 / 2). The delta channel passes them on (Prop 5).** Hence:
  - (i) fixes on the additive channel's post-completion writes, or a stop to all writes (halting), act on the cause;
  - (ii) the learned β-hold is a completion signal the model already computes, from the memory and without the pass token;
  - (iii) any learned post-completion fixed point needs a loss term that sees drift directly (CSR's ceiling, or an idempotence target). Answer-loss-only training is blind to drift that is harmless inside its trained horizon (Prop 6).
- **This is a hypothesis about converged models**, with one POST-HOC quantitative support (the (t−1)/t ratios). T1 is its causal test, and it can flip the ranking.

---

## Part II. EXIT: halting as optimal stopping

**Setting.**
- Per item there is an unknown completion time τ. After each pass t there is an observation o_t (decoder posterior, aux output, gates). Stopping at t returns a_t.
- C_FA = q(0) − c is the loss from stopping before completion (about 0.9 on nested, about 0.45 on ProsQA).
- c_d(j) = h(j) + c₀ is the cost of one more pass after completion: hazard plus compute.

**Theorem E1 (one-step look-ahead threshold; PROVED in the monotone case).**
- Let π_t = P(τ ≤ t | o_{1:t}) and p_t = P(τ = t+1 | τ > t, o_{1:t}).
- Stop now: expected cost (1 − π_t)·C_FA.
- Take exactly one more pass, then stop: (1 − π_t)(1 − p_t)·C_FA + π_t·c_d.
- Continuing is better iff π_t < A_t := p_t·C_FA / (p_t·C_FA + c_d).
- If the stopping region {π_t ≥ A_t} is absorbing (once done, the posterior stays high and A_t does not rise), the one-step rule is optimal among all stopping rules. This is the monotone case of Chow–Robbins–Siegmund (1971, *Great Expectations*).
- With a geometric prior on τ and i.i.d. observations before and after τ, this is Shiryaev's (1963) Bayesian quickest-detection problem, whose optimal rule is a posterior threshold.
- Both are classical results; Literature to confirm the citation details.

**Corollaries (DERIVED; the numbers are DATA, POST-HOC).**
- (a) **Overthinking is continuing where E1 says stop.** For t ≥ τ the expected one-pass gain is −h − c₀ < 0, and "always K′ = 32" is the never-stop rule. Its loss Σ_j h(j) is what the K′32 column measures.
- (b) **The threshold must be high.** With C_FA ≈ 0.9 and c_d ≈ 0.01 (DWELL K′n+2 − K′n ≤ 0.012, note 188-1; ProsQA K′12 vs 10 ≤ 0.03), p = 0.2 / 0.5 gives A ≈ 0.95 / 0.98. **Stop only when at least 95% sure the computation is done.**
- (c) **Why answer settling is PREMATURE (PROVED).** Before completion the decoder returns a stable default (173-3: acc ≈ 0.1 at t < n − 1). So P(settled | τ > t) ≈ P(settled | τ ≤ t), the likelihood ratio is about 1, π_t stays at the prior, and a rule on "settled" is a fixed-time rule set by the prior.
  - DATA: false alarms are 70–80% at n = 10 (answer2: 0.112 / 0.187 / 0.171 at the halt vs 0.64 / 0.90 / 0.60).
  - MOVED2 adds "moved off a(0)", but wrong answers move too (173-3a premise failed).
  - The memory-change rules (0.01) never fire: no false alarms, but unbounded delay.
- (d) **AUX-DONE is a good change detector inside training and a counter beyond it (DATA).**
  - It fires at exactly t = n for n 2–8 with accuracy 1.0.
  - It is pinned at t = 8 or never by the token (RMS) or the magnitude counter (§187).
  - It tracks n to about 10 only when neither counter exists (§190 s1: n 10 0.889 vs 0.897, mean halt 9.66).
  - On ProsQA its object-match proxy fires early on 30% of items (0.892 vs K′12 0.961).

**Prop 7 (a frozen-state halt is exactly halting; PROVED).** The decoder reads only S and a fixed prompt. So freezing every write after t̂ gives S_{K′} = S_t̂, and the answer is a(t̂) for every K′ ≥ t̂. This is also what the masked training select does.
- **A frozen-state halt cannot avoid PREMATURE.** PREMATURE belongs to the trigger, not to halting.
- **What can soften a premature trigger is a partial freeze.** Freezing only the additive heads (FA-on-signal) leaves the delta channel computing.
  - DERIVED bound, using 199-E1's ablations as proxies: in rms s1 / s2 the late computation survives additive think-read ablation at 0.88 / 0.76. So a premature FA costs at most about 0.12–0.24 of the late passes, against about 0.9 for a premature hard halt.
  - In §190 s1 / §184 s2 (0.24 / 0.31) the soft halt helps little.

**Candidate completion signals for π_t (ranked by the derivation; each predicted size-free or counter-keyed):**
1. **The β-hold / retention jump** (the gates of pass t+1, computed from the read of S_t):
   - the gate never sees the pass token (the think input has none), so it is predicted size-free on RMS and §190 models (CONJECTURE);
   - predicted counter-keyed on §184 / §187, whose unnormalised additive magnitude reaches u.
   - Detecting it costs one look-ahead pass.
2. **AUX-DONE** (2 aux greedy tokens per pass): token-keyed except on NOPASS models.
3. **The answer posterior** (max-prob, entropy, top-2 margin): unknown before completion. It is PREMATURE if the default answer is confident.
4. **A trained head on frozen features** (TRM / Ouro Stage II; G1 item #4), labels "done" 1[t ≥ n] or "correct" 1[a_t = y]:
   - **features exclude t and the additive read norm**, or the head learns the training prior "done by 8";
   - unsupervised ACT / PonderNet collapse to one loop (RecurTrace 2609.03379), so the head is trained post hoc on a frozen model with oracle labels, not with a cost penalty.

---

## Part III. ENTRY/EXIT between latent passes and written text (when to write a checkpoint)

**Decision model (DERIVED).**
- At each pass there are three actions: L (one more latent pass), W (write a text checkpoint), S (stop).
- **W** greedily decodes the step text e_{t+1} with the aux decoder, then restarts with think input [EOT, e_{t+1}, " ."] and S = 0. The answer decoder sees e_{t+1} as the expression.
  - On nested this needs no training: every intermediate e_t is itself a valid problem of depth n − t + 1 ≤ n.
- **Payoffs:**
  - a latent pass at segment pass s succeeds with p_L(s) (≈ 1 inside the reliable horizon H_L, falling past it);
  - a checkpoint succeeds with q_W(ℓ), where ℓ = n − t is the remaining length (≈ 1 for ℓ ≤ H_W, the longest step string seen in training, about 7 ops for n ≤ 8);
  - after a successful W the error is 0 and the pass counter restarts at 1. **A checkpoint also resets the integrator**, so no segment ever enters the untrained region past K. Text checkpoints are an overthinking fix for long computations.
- **Policy:**
  - in the good class (Thm 3: bounded error η/(1−κ) < r) never write; this is the pure latent, C1 ideal;
  - past H_L write while still readable, i.e. as late as p_L stays at about 1. Writing early only lengthens the decoded string (q_W smaller), and writing late loses p_L. This is the role the coordinator assigns to Chung 2605.07755's readability threshold: that ID is **NOT in our LIT notes and is unverified here; Literature to check before any citation**.

**Prop 8 (single-checkpoint reach; PROVED given the two horizons).**
- A checkpoint at pass k on an n-item needs k ≤ H_L (the latent is reliable) and n − k ≤ H_W (the remainder is decodable). It is feasible iff n ≤ H_L + H_W.
- More checkpoints do not extend this: the first one is binding, since later remainders are shorter.
- With H_L ≈ 7–8 and H_W ≈ 7, **n_max ≈ 14–15**. Inside the feasible window [n − H_W, H_L], the write-late rule picks k* = H_L, the latest reliable pass, since it gives the shortest decoded string.

**Relation to prior work.**
- §115 TRIGGER (6/6) snapped on the round-trip disagreement between the latent and text paths. It detected first errors at 1.0 but cost a text-path pass at every hand-off (115-1).
- §97: re-embedded text checkpoints.
- SWITCH 2606.13106: learned `<swi>` boundary tokens via RL.
- HRPO 2505.18454: a gated token / latent blend, RL.
- Abstract-CoT 2604.22709: a discrete reserved vocabulary plus RL.
- None decides *when to checkpoint and restart* from the latent state's own reliability (LIT_CLR_REVIEW: "None learns when to emit a checkpoint and restart").
- The minimal design below is eval-only and uses a label-free trigger.

**Minimal design: SNAP-CHAIN (eval-only, $0; T4).**
- Latent passes until EXIT fires. If no exit has fired by pass k_w, decode e_{k_w+1}; if it parses as an expression of the grammar, restart on it.
- The triggers compared:
  - TIMEOUT, k_w = H_L − 1 = 7;
  - LENGTH-IN-RANGE: write at the first pass whose decoded string has ≤ H_W operators;
  - fixed k_w ∈ {2, 4, 7}.
- The final segment halts by Part II's rule.
- **ENTRY** in this minimal form is re-entering latent mode right after every checkpoint. Learned entry (which steps to do in text at all; SWITCH-style routing) is later work.

---

## Part IV. Candidate fixes, ranked by the derivation

| rank | fix | guarantee (tag) | cost | what it breaks | C1 clean test |
|---|---|---|---|---|---|
| 1 | **Halting on a good completion signal** (EXIT rule, eval-only) | Prop 7 + E1: if t̂ ∈ [τ, τ + j_safe], acc = acc(t̂), and drift never happens (PROVED). Gain bounded by VOH; robustness to large K′ (DATA) | $0; saves passes (mean ≈ n + 1 vs 12–32) | nothing in training; risk = trigger false alarms (PREMATURE) | untouched (K′ = 0 / removal / serial run as before) |
| 2 | **Soft halt: freeze the additive heads on the signal** | removes P1 + P2 (Prop 5i); the delta channel relaxes to Δ*(u_t̂); tolerant of premature triggers in delta-computing models (DERIVED, II) | $0 | if P3 dominates, it leaves the delta-intrinsic drift | untouched |
| 3 | **WINDOW (+HOLD) training** | Lemma W: margin ≥ μ inside J = {1..4} for every n; beyond J only Lipschitz extrapolation (PROVED / DERIVED); STABLE 2/2 (DATA: K′32 0.739 / 0.996) | about $5–7 per cell (27 h) | HOLD cost −0.08 within K (s1; no reweight), and its "Step ≥ 9 → done" token shortcut | untouched |
| 4 | **CSR's ceiling (running; read only)** | Thm A1(ii): post-completion read drift ≤ e^λ/(1 − e^λ) × the last progress increment, on the trained passes (PROVED) | running | see Prop 9 | untouched (Thm A3) |
| 5 | **A strict ceiling or idempotence target** (λ_post ≤ log(1/3), on WINDOW so every n has post passes; IDEM is the limit λ → −∞) | Prop 9: drift < ½ the code distance, Thm 3's nearest-code margin (PROVED); a contraction-type property, so it extrapolates if met at the code (Thm 3) | one window cell, about $5–7 | possible within-K cost if the ceiling fights the integrator (199-A risk ii) | untouched |
| 6 | **A learned additive write gate**, A ← A + g_t W_t (g from u, init ≈ 1) | g → 0 after completion gives the correct fixed point (A frozen at the terminal value; PROVED); no pre-completion contraction (g = 1 is exact integration) | small | **needs a drift-seeing loss** (Prop 6): with answer loss alone the gate gets no gradient to close | untouched |
| 7 | **A decay / forget gate on the additive heads**, A ← γA + W | Prop 10: a fixed point exists, but it is the **write attractor** W*/(1 − γ), not the terminal code, unless the terminal write is self-consistent (PROVED) | small | pre-completion progress forgetting at rate γ (Thm 1 horizon; it re-creates the CSR floor's problem unless γ ≥ 0.95, where γ^8 = 0.66 barely changes K′16) | untouched |

**Prop 9 (what CSR's registered ceiling certifies; PROVED from 199-A's definitions).**
- With d_t = ‖R_{t+1} − R_t‖ and λ_t ≤ λ_post for t ≥ n, the total drift after completion is Σ_{j≥0} d_{n+j} ≤ d_{n−1}·e^{λ}/(1 − e^{λ}).
- d_{n−1} is the last progress increment, i.e. the read distance between the codes of e_n and e_{n+1}.
- At **λ_post = log 0.5 the bound is exactly one code distance**. Nearest-code correctness (Thm 3, r < δ/2) needs < ½, i.e. **λ_post ≤ log(1/3)**. So the registered ceiling bounds drift but does not certify P-A4.
- Under fixed K = 8 the post range [n, K − 1] is **empty for n = 8**, the row carrying most of rms s1's K′16 loss (0.242 of the pooled −0.065). CSR can reach n = 8 only by generalisation from n ≤ 7.
- The bound is on the think-query read; the decoder queries are not bounded directly (caveat).

**Prop 10 (decay does not fix overthinking by itself; PROVED).**
- With A ← γA + W* after completion, A_j → W*/(1 − γ) and the RMS read tends to N(φ(q)ᵀW*). That is the same limit as Prop 2's undecayed integrator (û → b̂).
- Decay changes only the rate (γ^j against about 1/j), and it adds pre-completion forgetting.
- It helps only if training makes the terminal write reproduce the terminal code. That again needs a drift-seeing loss.
- AGCLR 2606.07720's forget gate (its largest ablation) acts in a gated concept stream, not a fast-weight integrator, so it does not transfer as evidence.

**(e) Hold-tied additive gate (CONJECTURE; the "sharpest" candidate, and testable at eval first).**
- Gate the additive write by the model's own retention jump: g_t = 1 − σ(k(ρ_t − ρ_{t−1} − δ)).
- **Self-consistency (PROVED, conditional):** if g = 0 at the hold pass, A stops; u then varies only through Δ, which relaxes (Prop 5i); β stays at the hold, so the lock is self-sustaining.
- A level-based gate on β would also shrink late pre-completion writes (β falls 0.95 → 0.44 before completion), which hurts models that use the additive read late (§190 / §184). Hence a jump detector.
- **At eval this gate is exactly the soft halt of rank 2 with the β signal.** So it is tested at $0 before any training.

**Ranking logic.** Ranks 1–2 act on the root cause at $0 with no training and no C1 risk. They depend only on whether a completion signal exists, and T1 / T2 measure both. Ranks 3–5 are the trained routes to a post-completion attractor. Of these only 5 sees drift directly; 3 is proven in-window; 4 is running and is read, not stacked. Ranks 6–7 need a drift-seeing loss anyway, or are harmful by Prop 10.

---

## Part V. Tests (cheapest decisive first; nothing stacked on the running CSR or GSM8K cells)

**Common to all.**
- Eval-only on final model.pt of landed runs; an idle local card (Experiments' 2070), no co-tenants; fp32 eval, card recorded.
- 1,000 test items per n (test_set, as halt173), n 2..12 and 16. Training-distribution items: 2,000 (n 2..8; 199-M2's train_items rng 19900) for every threshold or head fit.
- Censoring: a rule that never fires stops at T_max = 32 with answer a_32. It is counted in every accuracy, and the event-only accuracy, mean halt and censored share are reported beside it.
- Scorers are written and fixture-checked before any output is read. Thresholds are fixed here.

### T0 (done now, $0, POST-HOC; not a test)
VOH table, horizon maps, 199-E1 by-n gates and increment ratios (Part I; voh.py, beta_by_n.py). They set the comparators and the expected effect sizes.

### T1 ($0): PASSDUMP + FREEZE 2×2. Which channel drives overthinking?

**Models.**
- Nested: s185_rms_s1 / s2, s190_rmsnopass_s1, s187_nopass_s1 / s2, s184_reweight_s1 / s2.
- Negative control: s182_window_s2 (K′32 0.996: nothing to recover).
- ProsQA s191_stepaux_relabel_s1: descriptive.

**Arms** (the oracle freeze point f = n + 1 nested, depth + 1 on ProsQA; passes up to 32 nested and 24 ProsQA):
- CLEAN;
- FA: after pass f, S_add (and z for den) is reset to S_add,f after every write, while the delta heads write normally;
- FD: after f, S_delta is reset to S_delta,f, while the additive heads write;
- FB: both frozen. By Prop 7 this equals halting at f.

**Gates.**
- FB's answers must equal CLEAN's a(f) bitwise at every K′ ≥ f (the identity check).
- CLEAN must reproduce eval_s173_halt acc_by_t within 0.005.

**Logged per item and pass:**
- the answer logits over the answer vocabulary;
- AUX-DONE bit and probability;
- β̄ and α-mass per delta head;
- retention ρ_t;
- additive and delta read increments at the decoder's query;
- the additive read angle θ_t to the freeze point.

**Readout.**
- Recovery R_X(K′) = (acc_X − acc_CLEAN)/(acc_FB − acc_CLEAN), pooled n 3..8 and n 9–10, at K′ ∈ {16, 24, 32}.
- Computed only where the CLEAN loss is ≥ 0.03; otherwise the model reads "nothing to recover" and serves as a no-harm control: |acc_X − acc_CLEAN| ≤ 0.01.

**Readings per model (at K′24):**
- **ADD-COUPLED:** R_FA ≥ 0.7 and R_FD ≥ 0.5 (P2 dominant).
- **ADD-DIRECT:** R_FA ≥ 0.7 and R_FD < 0.5 (P1).
- **DELTA-INTRINSIC:** R_FD ≥ 0.7 and R_FA ≤ 0.3 (P3).
- **JOINT:** both ≤ 0.3.
- Otherwise mixed.
- Pathway shares under additivity: P1 ≈ 1 − R_FD, P3 ≈ 1 − R_FA, P2 ≈ R_FA + R_FD − 1. The interaction is reported openly.

**Predictions (from Part I):**
- **T1-a:** rms s1 / s2 and §190 s1 read ADD-COUPLED.
- **T1-b:** §184 s1 / s2 read ADD-DIRECT or ADD-COUPLED, with R_FD < R_FA (larger P1: unnormalised read; the NORM-CLAMP precedent).
- **T1-c (mechanism):**
  - under FA, the delta increment ratios at t ≥ f + 3 fall to ≤ 0.85 (relaxation; Prop 5i), against CLEAN within 0.03 of (t−1)/t;
  - under FA, β stays within 0.03 of its minimum through t = 24, while under CLEAN it rises by ≥ 0.05 by t = 16 at n 4..6;
  - under CLEAN, θ_t is concave and increasing in t (Prop 2).
- **T1-d (ProsQA, weak):** R_FA ≥ R_FD. With T ≈ 450 the delta state is nearly rewritten every pass unless α → 1, and NORM-CAP recovered about ⅓ of depth 10's K′16 loss.

**Falsifiers.**
- In ≥ 2 of {rms s1, rms s2, §190 s1}, R_FA ≤ 0.3: the additive-integrator account of converged-model overthinking is false, and Part IV's ranks 2 / 5–6 drop below delta-side fixes (WINDOW plus a ceiling on the delta read).
- If T1-c's ratio under CLEAN is ≈ ρ rather than (t−1)/t, Prop 5(ii)'s reading of the POST-HOC ratios is wrong.

**Cost.** About 30–40 min per nested model (4 arms × 11 n × 32 passes; halt173 timings scaled) and about 1 h for ProsQA: about 5–6 h on one local card, $0.

### T2 ($0, then $0–1): halting rules scored offline from the T1 CLEAN dump; one primary rule

**PRIMARY: SHIRYAEV-HEAD.**
- A logistic π̂_t for label 1[t ≥ n], fit on the training-item dump, on features:
  - answer max-prob, entropy and top-2 margin;
  - answer-change flags at t−1 and t−2;
  - AUX-DONE bit and probability;
  - β̄_t and ρ_t per head;
  - their one-pass differences;
  - additive and delta read relative increments.
- **No t, and no additive norm.**
- Stop at the first t with π̂_t ≥ A = p̂·C_FA/(p̂·C_FA + c_d) (E1), where:
  - p̂ = the training hazard of completion per pass (geometric fit);
  - C_FA = 1 − acc(t < n − 1) on training items;
  - c_d = the mean one-pass post-completion hazard on training items + 0.002.
- The answer is a(t).

**Descriptive rules (same scorer):**
- AUX-DONE and AUX-DONE+1;
- MIN-BETA: stop at t when β̄_{t+1} > β̄_t + δ_β, with δ_β fit on training items;
- RET-JUMP: stop at t when ρ_{t+1} − ρ_t ≥ δ_ρ;
- CONF: stop at the first max-prob ≥ θ_c;
- TRM-HEAD: logistic on 1[a_t = y], stop at the first q̂ ≥ τ_q;
- answer2 / MOVED2 (references).

**Comparators.** Best fixed K′ **chosen on training items**; trained K; per-n oracle; K′ = 32.

**Readings (per model, on test, cap 32):**
- **NO-OVERTHINK:** pooled n 3..8 ≥ acc(best fixed K′ on train) − 0.01.
- **SIZE-FREE:** at n 9 and 10, ≥ acc(K′ = n + 2) − 0.03, with event-only mean halt in [n − 1, n + 3] and censored share ≤ 0.10.
- **PREMATURE:** ≥ 0.05 below K′ = n + 2 at any n ∈ 3..10.
- **COMPUTE:** mean passes, censored at 32, against the best fixed K′.
- Also reported: AUROC of π̂ for done vs not-done at each n.

**Predictions.**
- **T2-a:** NO-OVERTHINK for the primary rule on all 7 nested models. This rests on AUX-DONE features being exact in distribution.
- **T2-b (the genuine test):** on rms s1 / s2 the primary is SIZE-FREE at n 9, and at n 10 on s2, even though AUX-DONE there is token-pinned (fires at 8 or never). The prediction is that the β / ρ features carry a token-free completion signal.
  - Its descriptive twin: MIN-BETA's event-only mean halt at n 10 is ≥ 10 on rms s1 / s2.
  - **Falsifier:** on both RMS seeds, ≥ 50% of n = 10 halts fall at t ∈ {8, 9} with accuracy ≥ 0.05 below K′ = 12. Then the hold is pass-keyed (through the additive dilution counter), and the β cue is not a size-free EXIT.
- **T2-c:** on §184 s1 / s2 the primary is PREMATURE or censored at n 10. Their additive magnitude counter reaches the gates.
- **T2-d (E1's high threshold):** CONF is PREMATURE unless θ_c ≥ 0.99, and answer2 / MOVED2 stay PREMATURE (replicates 173-x; a reference only).
- **T2-e (soft halt):** FA-on-signal (FA starting at the primary rule's t̂ instead of the oracle f) is ≥ the hard halt at every n on rms s1 / s2, and ≥ 0.03 better at any n where the hard halt is PREMATURE.

**Trained halt head on frozen models ($0–1, after T2's scalar version).**
- The same labels, plus per-head read statistics (12 heads × norm / cos to the previous pass) as features. A small MLP, CPU or local card.
- **Registered falsifier (G1 #4):** pooled n 3..8 with halting < acc(K′8) − 0.03 means the correctness or done signal is not readable from these features.

### T3 ($0, when §199-A's CSR cells land; nothing stacked)
Read and report P-A4 as registered. Pre-stated descriptives (fixed now):
- (i) Per-n K′16 loss against the paired control (s185_rms_s{1,2,3}). **Prediction (Prop 9):** the loss ratio CSR / control is largest at n = 8 (no post term) and smaller at n 5..7.
- (ii) K′24 and K′32 against the control: does the ceiling extrapolate past K?
- (iii) An eval-only extension of CSR's λ_t logger to t ≤ 32, per head type. **"Extrapolates"** = the median post-completion λ over t ∈ [8, 24] ≤ log 0.8 and K′32 ≥ K′8 − 0.05.
- (iv) T1's 2×2 on the CSR models, to see whether CSR removed the additive driver (R_FA small because the CLEAN loss is small).

### T4 ($0): SNAP-CHAIN text checkpoints (ENTRY/EXIT)

**Models.** §190 s1 (the size-free done signal), rms s1 / s2, §184 s2.

**Items.** n 9, 10, 11, 12, 14, 16 (1,000 each).

**Policies.**
- Latent-only: the best fixed K′ and the T2 halt;
- TIMEOUT k_w = 7;
- fixed k_w ∈ {2, 4, 7};
- LENGTH-IN-RANGE;
- up to 3 checkpoints.

**Measured.**
- q_W: greedy whole-string aux exact at the write pass, plus the parse-valid share;
- per-segment accuracy;
- total passes and decoded tokens.

**Readings.**
- **SNAP-HELPS:** n 11–12 accuracy ≥ the latent-only best K′ + 0.15, at mean passes ≤ that K′ + 4.
- **SNAP-NEUTRAL:** within ±0.05.
- **SNAP-HURTS:** ≤ −0.05.

**Predictions (Prop 8 + the write-late rule).**
- **T4-a:** SNAP-HELPS at n 11–12 on §190 s1 (latent-only 0.54 / 0.24) whenever q_W at k_w = 7 is ≥ 0.6.
- **T4-b:** k_w = 7 ≥ k_w = 4 ≥ k_w = 2 at n 11–12. **Falsifier:** k_w = 4 beats 7 by ≥ 0.05.
- **T4-c:** a reach boundary. n 14 is still helped (the window [n − H_W, H_L] = [7, 7]); n 16 is not (empty window).
  - **Falsifier:** n 16 is helped by ≥ 0.15 with one checkpoint. Then H_W > 8 (the decoder copies longer strings than trained) or H_L > 8.
- **T4-d (C1 per segment):** each segment's K′ = 0 removal is at chance. Logged descriptively.

### T5 (≤ $10, conditional; each needs its own registration with a derivation; only after T1 / T2 and the CSR read)
- **If T2's primary is NO-OVERTHINK + SIZE-FREE on ≥ 2 models:** no training for overthinking. The halt rule becomes the K′-free eval protocol (C5 + C6), and up to $1 goes on the MLP halt head.
- **If T1 reads ADD-COUPLED / ADD-DIRECT and T2 is not SIZE-FREE:** one cell, RMS × WINDOW + a strict additive-read ceiling (rank 5, λ_post ≤ log(1/3) on the additive contribution), seed 2, about $5–7, paired with §193 (registered, unrun) as its comparator.
  - Predictions: STABLE and LONG-STABLE (K′24 ≥ K′8 − 0.05), no within-K cost (≥ 0.99 at n 3..8 at K′8), and T1 R_FA ≈ 0 (nothing left to recover).
- **If T1 reads DELTA-INTRINSIC:** the same cell with the ceiling on the delta read instead.
- **If T4 is SNAP-HELPS:** a frozen-model "write head" ($0–1) predicting next-step failure from the T1 features. That is the latent-only predictor of disagreement §115-1 asked for. Training a GSM8K / ProsQA checkpoint format needs the user's approval.

**Order and cost.** T1 → T2 (same dump) → T4 → T3 as CSR lands → T5 (conditional). T0–T4 cost $0 (about 6–8 card-hours locally). T5 is ≤ $10 and conditional.

---

## Part VI. Claim status

| claim | tag |
|---|---|
| Prop 1 additive no-fixed-point (generic full rank); RMS read fixed point iff radial writes | PROVED |
| Prop 2 constant-write drift law and saturation at the write direction | PROVED |
| Lemma 3 margin loss bound | DERIVED (Lipschitz decoder) |
| Prop 4 delta fixed point for frozen inputs | PROVED |
| Prop 5 the delta state follows the input drift: (t−1)/t increment ratios, log-divergent drift | DERIVED (A_u) |
| Prop 6 zero gradient past K_i; fixed-K trains only j₀ = K − n | PROVED |
| exponential horizon map from the log drift | DERIVED / CONJECTURE |
| two-term optimal-K′ condition | DERIVED |
| E1 one-step threshold (monotone case); Shiryaev form | PROVED (classical) |
| answer settling uninformative ⇒ PREMATURE | PROVED (given equal settle rates before and after) |
| Prop 7 frozen-state halt ≡ halt | PROVED |
| soft-halt tolerance bound | DERIVED (ablation proxies) |
| Prop 8 single-checkpoint reach n ≤ H_L + H_W | PROVED (given horizons) |
| Prop 9 CSR log 0.5 = one code distance; empty post range at n = 8 | PROVED |
| Prop 10 decay → write attractor | PROVED |
| β-hold is token-free and size-free | CONJECTURE (T2-b tests it) |
| additive integrator drives converged overthinking via the gate (P2) | CONJECTURE + POST-HOC DATA (T1 tests it) |
| all numbers from eval / emp files | DATA, POST-HOC |

## Caveats
- The 199-E1 gates are means over 32 items per n. Per-item β profiles may be noisier, which risks MIN-BETA false alarms; the primary rule uses a fitted head for this reason.
- The orthogonal-write fractions use cosines rounded to 3 decimals (±30%).
- The VOH and horizon-map numbers come from the same eval files that motivated the account (POST-HOC). The registered tests are T1–T4 on stated thresholds.
- Chung 2605.07755 is unverified (not in any LIT note). Shiryaev 1963 and Chow–Robbins–Siegmund 1971 are classical; Literature to confirm the details.
- T1 / T2 use oracle n only for the freeze point (T1) and for training labels (T2, training items). Test-time rules never see n.
