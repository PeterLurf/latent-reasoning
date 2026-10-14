# THEORY §199 derivations: contraction, selection probability and the seed lottery (2026-10-13)

Written by Opus subagents of the Theory session, at the user's direction ("deligate reseach or math proof/ressoning fomulation to fellow opus 5.5 subagetns", 2026-10-13). Theory reviewed the key proofs (Thm 1 / 1', Lemmas 1-4, Thm 3 / Prop 3.2, Theorem 1 of Part II and its corollaries, the init numbers α^L).
* **Status tags are the authors'.** Every claim is marked PROVED / DERIVED (under stated assumptions) / CONJECTURE / CHECK / DATA. [DATA] items are post hoc.
* **The method "PCLT" (Part I §6.A) is registered as CSR** (completion-switched step-ratio target; THEORY amendment 199-A), renamed per Literature's novelty verdict (docs/LIT_S199_PCLT_NOVELTY_2026-10-13.md, 67329f72):
  - the sign switch at completion is NEW;
  - the step-ratio measure is KNOWN (2607.09698, Eq. 3);
  - using it as a training target is PARTLY KNOWN;
  - avoid "Lyapunov target" unqualified.
* **Scripts** (CPU only; they read run files and the GPT-2 checkpoint and write JSON beside themselves) are in theory/seedstab199/math and theory/seedstab199/sim. The subagent fact sheet is FACTS.md, and its literature scan is LIT_contraction_subagent.md.

---

# Part I: contraction vs serial computation

# Contraction, integration and the seed lottery in the DeltaMem loop

Theory subagent, 2026-10-13 21:49 EDT (clock). Read-only use of `boxes/deltamem.py`, `boxes/nested170.py`, `boxes/deltas5.py:fwd_read`, run `eval*.json` files and `scratch/lyap192.py`. Numerical checks: `scratchpad/seedstab/math/*.py` (CPU, 3 threads, idle priority). Their outputs are quoted with the script name.

**Tags.**
- **PROVED**: a complete proof is given here, under the stated assumptions.
- **DERIVED (A…)**: follows from the named assumptions, which are not verified for our model.
- **CONJECTURE**: an argument or analogy, not a proof.
- **CHECK**: a numerical sanity check.
- **DATA**: a number read from an existing eval file.

---

## Summary

1. **Resolution bound (Thm 1, PROVED).**
   - Suppose the pass map is κ-contractive (in a norm where the decoder is L-Lipschitz), and a pass-free decoder reads every step with logit margin μ.
   - Then n ≤ 1 + log(L·‖S₁−S₀‖/μ) / log(1/κ). Consecutive margins also obey m_t + m_{t+1} ≤ 2Lκ^t‖S₁−S₀‖, and starts are forgotten as κ^t.
   - Four independent s3 statistics give a κ between 0.76 and 0.91:
     - per-step aux success: 0.78;
     - answer ratio by n: 0.76;
     - whole-state step ratio: 0.84–0.91;
     - delta cos: ≈ 0.89–0.91.
   - At init the open-loop delta map has κ = α^L: 0.91 at n = 4 and 0.83 at n = 8 (CHECK). **s3 looks like the init's open-loop contraction never got undone.** The converged models have κ_eff ≥ 1.
2. **Integrator bound (Thm 2, PROVED for the open loop, DERIVED for the closed loop).**
   - A neutral channel never forgets its errors: bias grows as t·b, noise as √t·σ.
   - With an RMS-normalised read and fixed-size writes, the newest step's share of the read decays as 1/√t (CHECK: 1.0, 0.71, 0.50, 0.35, 0.25 at t = 1, 2, 4, 8, 16).
   - Keeping it constant needs geometric write growth. That fits the converged models' positive step ratios and growing additive norm.
   - Corruption after completion: the pure "j/n" scaling is **falsified** by the K′ = 16 table (DATA). What survives is "no intrinsic fixed point, so the model is stable only where it was trained" (fixed-K vs window recipes).
3. **Good class (Thm 3, PROVED).**
   - A code-attractor successor map: locally κ-contractive around each code, moving code → successor code, with the terminal code a fixed point.
   - It computes exactly for **any** n and any number of extra passes if the per-pass write error satisfies η ≤ (1−κ)r.
   - A global contraction cannot do this for unbounded n with separated codes (Prop 3.2). Its Lipschitz constant is ≥ 1 *between* basins and < 1 *inside* them.
   - **Reconciliation with Anil 2211.09961 (§3.4).**
     - Harmful forgetting is *pre-completion forgetting along progress coordinates*.
     - Forgetting transverse to the code is error correction, and good.
     - Our donor test starts **on** the code manifold (another item's completed code). It measures progress-direction forgetting and re-anchoring. It is blind to transverse forgetting.
4. **Delta Jacobian (Thm 4, PROVED).**
   - P = (Π_τ α_τ)·H₁⋯H_L.
   - Singular values:
     - exactly Πα on span(K)^⊥;
     - Πα·σ_i(Q|_K) on span(K).
   - Exact dissipation identity: ‖v‖² − ‖Qᵀv‖² = Σ_τ β_τ(2−β_τ)(k_τ·v_{τ−1})².
   - **For every parameter value the open-loop delta map is a strict contraction**, ‖P‖ ≤ Π_τ α_τ < 1. Isometry needs every β ∈ {0, 2} and α = 1, which the parametrisation cannot reach.
   - At init with GPT-2 keys (CHECK), the slowest key-space direction is barely touched by the projections (σ_max(Q|_K) ≈ 0.97–1.0), so ‖P‖ ≈ α^L. The contraction scales with input length: 0.91 at n = 4, 0.77 at n = 12, 0.049 at ProsQA length L ≈ 450.
   - The closed loop can undo it only if ‖E‖ ≥ 1 − Πα along progress directions. At init the LN_w Jacobian alone attenuates the loop by about 0.05 (CHECK).
5. **Why SGD selects contraction (§5).**
   - **PROVED:**
     - the open-loop phase (γ = 0, steps ≤ 2000) *is* the bad set: input-determined trajectories, a geometric clock, and an additive read that is t-blind under rms and den;
     - in a κ-contractive regime, any parameter step moves the *differential* (pass-specific) mode of late states by at most ‖δθ‖(Bκ^r + rκ^{r−1}K‖Δ₀‖) but the *common* (pass-independent) mode by up to B‖δθ‖/(1−κ). So the loop's gradient enriches the input summary that pass-token decoders exploit, and only weakly builds progress.
   - **CONJECTURE:** a threshold / race model (the closed-loop gain is a multiplicative small mode; Rawal–DeWeese escape law; closing LR window) plus implicit-regularisation drift toward contraction. Noise decides which side of the threshold a seed lands on.
6. **Methods (§6).**
   - **A. PCLT** ("progress-cone Lyapunov target"):
     - a scale-free log-ratio penalty on the *realised read-space progress increments*: a floor λ ≥ log 0.95 before completion and a ceiling λ ≤ log 0.5 after it;
     - zero extra trunk passes;
     - PROVED: it certifies Thm 1's horizon and bounds the post-completion drift; its gradient is not κ^t-attenuated (it is a relative sensitivity), which turns the multiplicative escape into an additive drive.
     - It differs from Gradient Flossing 2312.17306 (uniform λ → 0) and STARS 2605.26733 (convergent throughout): it is restricted to the progress direction and switches sign at completion.
   - **B. Pass-boundary contraction:**
     - decay once per pass (B-light), or a within-pass staged delta rule merged once with a rank-capped soft erase (B-full);
     - PROVED: the open-loop spectrum is independent of L and of key geometry ({α_p} or {α_p, α_p|1−β_p|});
     - within-pass overwrite is kept; read-modify-write against the carried state is lost.
   - **C. Self-encoded state teacher forcing** (riskier candidate): breaks the serial unlock cascade into parallel depth-1 problems.
7. **Falsifiable predictions** (§6.4) are given with numbers, on 3 seeds of RTX 3090:
   - mechanism-level: the 8k log-ratio and entry step;
   - outcome-level: the class on the path test, K′16 stability, and n = 12 per-step handle accuracy;
   - with explicit falsifiers for the theory itself.

---

## 0. Setup: the exact pass map (from the code)

### State
- S = (A, Δ, [z]):
  - A ∈ ℝ^{6×64×64} are the additive heads, indexed (k, v);
  - Δ ∈ ℝ^{6×64×64} are the gated-delta heads, indexed (v, k);
  - z is the extra den block, ignored unless stated.
- The think input x = [EOT, expr, " ."] (real length L = n_syms + 2) is identical on every pass.
- h₆ = blocks 0..5(x) is pass-invariant (`gpt_lower`).

### Read (`read_fixed`, deltamem.py l.158–176)
- At every think token τ: r_τ(S) = fade · W_o [ g_add · N(φ(q_τ)ᵀA) ‖ g_del · Δ q̂_τ ].
  - N is the rms per-head normalisation × add_gain (l.153–154), or the identity (none), or division by φ(q)ᵀz (den).
  - q_τ comes from LN₆(h₆), so **the read queries are pass-invariant**: S enters only through the fixed query matrices.
- R(S) := (r_τ(S))_τ is everything the trunk sees of S.

### Write (`write_seq`, l.179–223)
- h_w(S) = blocks 6..11(h₆ + R(S)).
- u_τ = (1−γ)·LN₆(h₆)_τ + γ·LN_w(h_w(S))_τ.
- (k_τ, v_τ, α_τ, β_τ) are functions of u_τ; α = σ(·), β = 2σ(·) ∈ (0, 2), and k̂ = k/‖k‖ for the delta heads.
- Then, over the real tokens τ = 1..L (padding masked: no change, no decay):
  - A ← A + φ(k_τ)v_τᵀ
  - Δ ← α_τ Δ (I − β_τ k̂_τ k̂_τᵀ) + β_τ v_τ k̂_τᵀ (l.210)

### Pass map
- S_t = F(S_{t−1}; x), autonomous, S₀ = 0.
- Aux decoder at pass t: `fwd_read` on S_t with the prompt [EOT, " Step", (" t"), " :"] plus e_{t+1}, teacher-forced. There is no problem text.
- Answer decoder: [EOT, expr, " ="] reading S_K.

### Schedules
- fade: 0 → 1 over steps 0..500.
- γ: 0 for steps ≤ 2000, then linear to 1 at 6000.
- At init, W_o = 0, the gate weight is 0 (so β ≡ 1 and α ≡ σ(5) = 0.99331), and the q/k/v rows are copied from block 6.
- Recipes: rms / reweight / nopass / den / nofade use a fixed K = 8. window / hold / rwwin use K = n + U{1..4} (args.json, DATA).

### Open loop
- γ = 0, or W_o = 0, or fade = 0.
- Then u does not depend on S, so (k, v, α, β) depend on x only.
- The pass map is affine: F(A, Δ) = (A + a(x), ΔP(x) + C(x)), with
  - P = M₁M₂⋯M_L, M_τ = α_τ(I − β_τ k̂_τ k̂_τᵀ);
  - C = Σ_τ β_τ v_τ k̂_τᵀ M_{τ+1}⋯M_L.

---

## 1. Resolution bound

### Definitions
- A **pass-free decoder** D maps a state (plus a fixed prompt and teacher-forced prefix) to logits. This is the nopass recipe.
- For consecutive targets y = e_{t+1} and y′ = e_{t+2}, let j be the first position where they differ. Their prefixes before j are identical, so D's token input at j is identical for S_t and S_{t+1}.
- Let z(S) ∈ ℝ^V be the logits at j, and L the Lipschitz constant of S ↦ z(S) from ‖·‖ to ‖·‖_∞ on a set U.
- The margin at pass t is m_t = z(S_t)[y_j] − max_{i≠y_j} z(S_t)[i].
- F is **κ-contractive on U** if ‖F(S) − F(S′)‖ ≤ κ‖S − S′‖ for all S, S′ ∈ U, with F(U) ⊂ U.

### Theorem 1 (PROVED)
Let F be κ-contractive (κ < 1) on U ∋ S₀, and let d₀ = ‖S₁ − S₀‖ (= ‖F(0)‖ from the zero start).

- (a) ‖S_{t+1} − S_t‖ ≤ κ^t d₀, and S_t → S*(x) with ‖S_t − S*‖ ≤ κ^t d₀/(1−κ).
- (b) **Path independence.** For two starts S₀, S₀′ ∈ U: ‖S_t − S_t′‖ ≤ κ^t ‖S₀ − S₀′‖.
- (c) **Margin decay.** For a pass-free decoder and consecutive distinct targets: m_t + m_{t+1} ≤ 2Lκ^t d₀.
  - In particular, if steps 1..n are all decoded with margin ≥ μ > 0, then **n ≤ 1 + log(L d₀/μ) / log(1/κ)**.
- (d) Assume additive noise on the margins, i.i.d. N(0, σ²) per pass, and balanced margins m_t ≈ m_{t+1} (a CE-trained decoder has no reason to favour one). Then the per-step success satisfies p_t ≤ Φ(Lκ^t d₀/σ).

### Proof
- (a) S_{t+1} − S_t = F(S_t) − F(S_{t−1}). Iterating gives κ^t‖S₁ − S₀‖. The geometric tail gives the limit (Banach).
- (b) The same argument with two trajectories.
- (c) Write z_t = z(S_t), z_{t+1} = z(S_{t+1}), y = y_j, y′ = y′_j ≠ y.
  - From the margins: z_t[y] − z_t[y′] ≥ m_t and z_{t+1}[y′] − z_{t+1}[y] ≥ m_{t+1}.
  - Adding: (z_t − z_{t+1})[y] − (z_t − z_{t+1})[y′] ≥ m_t + m_{t+1}, so 2‖z_t − z_{t+1}‖_∞ ≥ m_t + m_{t+1}.
  - Then ‖z_t − z_{t+1}‖_∞ ≤ L‖S_t − S_{t+1}‖ ≤ Lκ^t d₀.
  - With all margins ≥ μ for t = 1..n−1: 2μ ≤ 2Lκ^t d₀, hence t ≤ log(L d₀/μ)/log(1/κ) for t = n−1.
- (d) follows from (c) and the noise model. ∎

### Remark 1.1 (the norm matters; PROVED by example)
Thm 1 needs a norm in which *both* F is contractive *and* the decoder has a moderate L. Example 3.3 below (a nilpotent pipeline) is contractive in a weighted norm and computes n exact steps, but its decoder's Lipschitz constant in that norm is w^{−n}, which makes the bound vacuous.

The robust, norm-free version is stated in decoder (or read) space.

### Theorem 1′ (PROVED; the measurable form)
- Let g_t = ‖R(S_{t+1}) − R(S_t)‖ / ‖R(S_t) − R(S_{t−1})‖ be the *realised read-space progress gain*. R is any feature map through which the decoder sees S, with the decoder Lipschitz L_R in R.
- If g_s ≤ κ for t₀ ≤ s < t, then the margins satisfy m_t + m_{t+1} ≤ 2L_R κ^{t−t₀}‖R(S_{t₀+1}) − R(S_{t₀})‖.
- The same telescoping gives the converse **certificate**: if a pass-free decoder reads steps t₀..n with margin μ, then Π_{s=t₀}^{n−2} g_s ≥ μ/(L_R‖R(S_{t₀+1}) − R(S_{t₀})‖). The geometric-mean progress gain cannot be small.

### Corollary 1.2 (pass-token decoders; DERIVED under A1)
- **Assumption A1:** one trunk forward (the decoder) reliably performs at most d reductions beyond what the state makes explicit.
- With the pass token, the decoder D_t sees t. Thm 1(c) does **not** constrain its accuracy, because two passes have different token inputs.
- What Thm 1(a) still gives: |D_t(S_t) − D_t(S*)| ≤ Lκ^t d₀/(1−κ). So for t ≥ N_ε the aux output equals the output computed from the input-determined fixed point S*(x) plus the token t.
- Under A1, accuracy collapses once t + 1 exceeds (the progress stored in S*) + d.
- **Prediction (P1.2):** in a contractive pass-token model, late-pass aux errors are "lag" or arithmetic errors made by the decoder, not off-by-one-pass confusions.
  - Test: on s185_rms_s3's greedy aux outputs at t ≥ 5, measure the fraction of wrong outputs equal to e_{t′} for some t′ ≤ t (lag) vs other strings.
  - A contraction-plus-shortcut account predicts mostly *non-lag* arithmetic errors with the pass token.
  - With nopass, it predicts lag errors (the state stalls at its fixed point).

### 1.3 The integrator part: what "contractive on the content subspace" means
- The additive heads have open-loop Jacobian I (no decay). They are never contractive in state space.
- But the trunk and decoders see A only through the **normalised** read N(φ(q)ᵀA) (rms, den).
- Define content-space contraction in read space, through g_t of Thm 1′.

**Lemma 1.3 (PROVED; CHECK `toys.py` C).**
- Let A_t = Σ_{s≤t} w_s with unit writes, and read ρ(A) = A/‖A‖.
- If the writes are pairwise orthogonal, the newest write's share ⟨ρ(A_t), w_t⟩ = 1/√t. Random unit codes in ℝ^256 give 1.0, 0.707, 0.496, 0.352, 0.253 at t = 1, 2, 4, 8, 16, against 1/√t.
- If the writes are coherent (all ≈ w̄), the read increment ‖ρ(A_{t+1}) − ρ(A_t)‖ ≈ ‖w_{t+1}^⊥‖/‖A_t‖ = O(1/t).
- So a normalised integrator with fixed-size writes is **algebraically** contractive in read space: Π g ≈ (t₀/t)^{1/2 or 1}, not κ^t. Thm 1′ then gives a *polynomial* resolution horizon, N ≲ t₀(L_R d_{t₀}/μ)^{2 or 1}.
- To keep a constant newest-step share ρ*, the writes must grow geometrically: ‖w_t‖ = λ‖A_{t−1}‖. For orthogonal writes λ = 1, the share holds at 0.707 and the state norm grows ×√2 per pass (CHECK).

**Definition (content-space contraction).** The loop is *contractive on the content subspace* over [t₀, t] if the realised read-space gains satisfy g_s ≤ κ < 1 there.
- κ-geometric: g bounded by κ < 1.
- Algebraic: g_s → 1 like 1 − c/s.
- Neutral or expanding: g ≥ 1.

The open-loop delta heads are geometric (Thm 4). Additive heads with fixed writes are algebraic. Growing writes are neutral in read share and expanding in norm.

### 1.4 Numbers: s3 is contractive, near the init's open-loop rate (CHECK `fits.py`, `init_jacobian*.py`)

| statistic (s185_rms_s3) | implied κ | how |
|---|---|---|
| aux conditional per-step success t = 3..8 (0.97, 0.92, 0.90, 0.86, 0.73, 0.72) | **0.78** (a = Ld₀/σ = 4.0, residual 0.10 in log-probit) | Thm 1(d) fit |
| answer acc ratio acc(n)/acc(n−1), n = 3..8 at K′ = 8 (0.962, 0.878, 0.764, 0.755, 0.736, 0.643) | **0.76** | Thm 1(d) fit |
| whole-state step ratio at t = 2, 3, 4 (lyap192: −0.175, −0.150, −0.095) | **0.84, 0.86, 0.91** | Thm 1(a) |
| delta cos 0.978 after 16 passes, donor vs zero start | **0.89–0.91** (cross-item cos 0 to 0.5) | Thm 1(b), 1 − cos ∝ κ^{32} |

Comparison:
- Converged models' step ratios are ≥ 0 (s1: +0.005, +0.057, +0.108), so their effective κ ≥ 1.
- **Init, open loop, delta heads only**, with real GPT-2 block-6 keys on nested inputs (24 items per n):

| n | L | ‖P‖ = α^L | open-loop delta cos16 (donor vs zero) | step ratio t = 2..4 (delta only) |
|---|---|---|---|---|
| 2 | 8.2 | 0.947 | 0.966 | −0.40, −0.29, −0.24 |
| 4 | 13.7 | 0.912 | 0.990 | −0.41, −0.31, −0.26 |
| 8 | 27.0 | 0.834 | 0.9995 | −0.50, −0.39, −0.34 |
| 12 | 38.6 | 0.772 | 1.0000 | −0.57, −0.46, −0.42 |

**Reading (DERIVED, not established).**
- s3's progress contraction (κ ≈ 0.76–0.91) sits in the init's open-loop band: ‖P‖ ≈ 0.83–0.91 for the n ≤ 8 inputs it trains on.
- Every converged model left that band (κ_eff ≥ 1).
- The hypothesis **H_open**: "a failed seed is one whose closed-loop gain along progress never exceeded the open-loop gap 1 − ‖P‖" (Thm 4.3).
  - It is testable by the empirical subagent's per-head-type split.
  - If the contraction sits in the additive heads, H_open is wrong as stated, and §4's gap must be replaced by the additive read's algebraic gap.

---

## 2. Integrator error bound

### Model
A neutral channel: A_t = A_{t−1} + w_t* + ξ_t, where
- w_t* is the ideal write;
- ξ_t is the per-pass write error, with E ξ_t = b and Cov ξ_t = σ²I/m, independent across passes.
- The ideal state is A*_t = Σ w_s*, and the error is E_t = A_t − A*_t.

### Theorem 2

**(a) Open loop (PROVED).** E_t = Σ_{s≤t} ξ_s exactly. So E[E_t] = t·b and Cov E_t = tσ²I/m: **bias ∝ t, noise ∝ √t** (CHECK: ‖E_t‖/σ = 1.0, 2.0, 4.0 at t = 1, 4, 16). Nothing ever decays.

**(b) Closed loop, normalised read (DERIVED under A2).**
- **Assumption A2:**
  - w = W(ρ(A)), with W ℓ_W-Lipschitz;
  - ‖ρ(A) − ρ(A′)‖ ≤ 2‖A − A′‖/‖A*‖ near the ideal path;
  - ‖A*_s‖ ≥ c·s.
- Then ‖E_t‖ ≤ (1 + 2ℓ_W/(c(t−1)))‖E_{t−1}‖ + ‖ξ_t‖, which gives ‖E_t‖ ≤ Σ_s ‖ξ_s‖·(t/s)^{α} with α = 2ℓ_W/c: **polynomial amplification**.
- The read error is ‖E_t‖/‖A*_t‖. For bias this is ≈ b/(c(1−α)), constant in t.
- The newest-step signal is ∝ 1/t (coherent writes). Decoding fails at t* ≈ c(1−α)μ′/b: a linear horizon in the inverse bias.

**(c) Growing-write integrator (DERIVED under A2′).**
- **Assumption A2′:** ‖w_t‖ = λ‖A_{t−1}‖, with per-pass relative error ε_rel (bias b̄ and noise σ̄ in units of the write).
- The newest-step share stays at ρ*.
- The decision variable drifts as t·b̄ (bias) and √t·σ̄ (noise), so p_t ≈ Φ((μ − t b̄)/(√t σ̄)).
- Failure past t* ≈ μ/b̄ or (μ/σ̄)².
- This is the "bias ~ t, noise ~ √t" form. Of the three regimes it is the one the converged models' positive step ratios and growing additive norm point to.

**(d) After completion (DERIVED).**
- If j extra passes write r_j of relative size ρ in random directions, then 1 − cos(R_n, R_{n+j}) ≈ jρ²/(2n). For example, n = 4, ρ = 0.3, j = 8 gives 0.079 (CHECK).
- So the read drifts linearly in j and is diluted by 1/n.
- Without a fixed point, the drift stops only if the writes after completion are learned to vanish or align.

### 2.1 Confronting the data

- **Post-completion (DATA, eval.json acc_by_n_kprime).**
  - s185_rms_s1 (fixed K = 8) at K′ = 16:

    | n | 4 | 5 | 6 | 7 | 8 |
    |---|---|---|---|---|---|
    | K′ = 16 accuracy | 0.996 | 0.981 | 0.953 | 0.920 | 0.752 |

  - At K′ = 8: 0.994 for n = 8.
  - **The error grows with n, not with j/n** (n = 4 has j = 12 and is fine).
  - So the pure dilution law (d) is falsified as the sole explanation.
  - What the data fit: the model was trained with 8 − n post-completion passes (fixed K = 8, answer read at S₈). Error at K′ = 16 falls monotonically with that trained post-completion horizon: 0 → 0.25, 1 → 0.08, 2 → 0.047, 3 → 0.019, 4 → 0.004.
  - Recipes with trained post-completion passes are stable: window_s2 gives n = 8 at K′ = 16 0.997; nopass_s1 0.989.
  - **Surviving claim (DERIVED):** an integrator has no intrinsic fixed point, so post-completion stability exists only where training put it. A design with contraction after completion (§3, §6A) has stability by construction.
- **DEPTH (DATA, eval_s187_deepaux_items_n9_12.json, teacher-forced handle accuracy after pass t).**
  - In converged models the per-step accuracy at a **fixed** pass t ≤ 8 falls with input length.
  - For rms_s1 at t = 4: 0.986, 0.964, 0.945, 0.931 for n = 9, 10, 11, 12.
  - At t = 8: 0.77, 0.76, 0.55, 0.45.
  - So DEPTH is partly size-keyed, not only "beyond the trained horizon". That fits a per-pass contraction growing with L (Thm 4; Audit's α^{L(n)}).
  - **Confound:** the target e_{t+1} is also longer for larger n.
  - A margin-model κ(n) fit (`fit_size.py`) gives no consistent L-slope across models (dlogκ/dL from −0.024 to +0.012). The direct measurement (ᾱ_pass, P-spectrum; the §199 plan's ρ̄ test) decides this.
  - Separate hazard: pass-token decoders collapse at t = 9 (an untrained token; rms_s1 n = 9 t9 0.144), while nopass_s1 holds (0.977).

---

## 3. The good class: serial computation that is stable after completion

### Definition 3.1 (code-attractor successor map, CASM)
- E is a finite set of expressions with successor σ (leftmost-ready reduction). Numbers are terminal: σ(v) = v.
- Item x has the chain e₁ → ⋯ → e_{n+1}; elements before termination are distinct, because each step shortens the string.
- Codes c: E → state space, with δ-separation along chains: ‖c(e) − c(e′)‖ ≥ δ for e ≠ e′ on a chain.
- F(·; x) is a CASM with parameters (κ, r, η) if, for every e on x's chain including the terminal one:
  - S ∈ B(c(e), r) implies ‖F(S; x) − c(σ(e))‖ ≤ κ‖S − c(e)‖ + η, with κ < 1;
  - and ‖S₁ − c(e₂)‖ ≤ r₁ ≤ r.

### Theorem 3 (PROVED)
- If η ≤ (1−κ)r, then for every t ≥ 1: ‖S_t − c(e_{min(t+1, n+1)})‖ ≤ κ^{t−1}r₁ + η(1 − κ^{t−1})/(1−κ) ≤ r.
- Let a decoder have margin μ_c at the codes and be L_D-Lipschitz, with 2L_D r < μ_c (or nearest-code decoding with r < δ/2). Then it reads e_{t+1} correctly at every pass t ≤ n, and the answer for **every** K′ ≥ n.
- There is no horizon in n and no corruption from extra passes. Write errors do not accumulate: they are bounded by η/(1−κ).

**Proof.** Induction. S₁ is in the ball. If S_t ∈ B(c(e_{t+1}), r_t), then S_{t+1} ∈ B(c(e_{t+2}), κr_t + η). Also r_{t+1} = κr_t + η ≤ r whenever r_t ≤ r and η ≤ (1−κ)r. The terminal code maps into its own ball. The decoder margin drops by at most 2L_D r. ∎

### Margin condition (quantitative)
η/(1−κ) + κ^{t−1}r₁ < min(δ/2, μ_c/(2L_D)).

The admissible per-pass write error scales with the **transverse spectral gap** 1 − κ. Strong local contraction tolerates large write errors.

### Proposition 3.2 (a global contraction cannot do this for all n; PROVED)
- Suppose F(·; x) is κ_g-Lipschitz on a set containing the chain's code balls, and computes a chain of length n as in Thm 3.
- Since S_t ∈ B(c(e_{t+1}), r), we have δ − 2r ≤ ‖S_{t+1} − S_t‖ ≤ κ_g^{t−1}‖S₂ − S₁‖ ≤ κ_g^{t−1}(δ_max + 2r).
- Hence κ_g ≥ ((δ − 2r)/(δ_max + 2r))^{1/(n−2)}, which tends to 1 as n → ∞.
- So a family solving all chain lengths with fixed separation has sup Lipschitz ≥ 1. That Lipschitz is attained **between** basins (code → successor code), while inside the balls it is κ < 1.
- "Contraction" and "serial computation" are reconciled by **multi-basin local contraction**, not by a global Banach contraction. ∎

### Example 3.3 (nilpotent pipeline; PROVED)
- F(s₁, …, s_n; x) = (a(x), g(s₁), …, g(s_{n−1})) with g ℓ-Lipschitz.
- In the weighted norm ‖s‖_w = max_i w^i‖s_i‖ with w < 1/ℓ, F is (wℓ)-contractive.
- It is exact after n passes: register i holds g^{i−1}(a(x)). It is path-independent from **any** start after n passes, and stable.
- A decoder of register n has Lipschitz constant w^{−n} in ‖·‖_w, so Thm 1 is vacuous there.
- Lesson: (i) apply Thm 1 in decoder or read space (Thm 1′); (ii) path independence together with exact serial computation is possible when the input is re-injected every pass, which is Anil's setting.

### 3.4 Progress versus transverse forgetting (Anil reconciliation)

Decompose a state near the code manifold as S = c(p) + z:
- p = the progress coordinate (the nearest code, i.e. how far the computation has got);
- z = the transverse offset.

There are four kinds of forgetting:

| kind | what is forgotten | effect | in a CASM |
|---|---|---|---|
| **T** (transverse) | an off-manifold offset z | error correction, desirable (Anil's path independence for arbitrary inits) | z → κ^t z |
| **P** (progress, pre-completion) | how far the chain has got: p_t stalls or converges to an input-determined point before termination | **harmful**: Thm 1 horizon, per-step accuracy decaying in t | forbidden |
| **R** (re-anchoring) | an x-inconsistent code (another item's progress) is reset onto x's chain | Anil-desirable; costs computation (consistency check against x) | optional (G1) |
| **none** | progress is never forgotten, including errors | integrator, Thm 2 | — |

**What the donor test measures (answer to the coordinator).**
- The donor start is another item's 8-pass memory, which is **on** (or near) the code manifold: a completed code of another item.
- So it probes P and R. It **does not probe T**.
- Predicted signatures:

| class | delta cos (donor vs zero) | acc from zero / donor | random small perturbation (≈ transverse) |
|---|---|---|---|
| B (P-forgetting, bad) | high | both mediocre, ≈ equal | anything |
| I (integrator) | low | high / ≈ chance | ≈ neutral |
| G0 (CASM, no re-anchoring) | low | high / ≈ chance (donor's terminal code is a fixed point) | contracting |
| G1 (CASM + re-anchoring) | high | high / high | contracting |

- DATA:
  - s3: 0.978, 0.596 / 0.576. This is B.
  - Converged models: 0.44–0.63, about 0.99 / 0.10–0.14. These are I or G0.
  - G1 is never observed.
- The donor test **cannot** separate I from G0. Post-completion stability and the transverse-perturbation sign do.
- It separates B from G1 only through the accuracy columns.
- The s192 random perturbations (relative slopes: s1 −0.038, s2 −0.004, s3 +0.034) are mostly transverse in a crude sense. But they are also mostly **read-invisible**: S has 49k numbers, while the think read sees a T-dimensional query span per head. Read-invisible directions evolve by the open-loop map alone (I for A, P for Δ), so these slopes largely measure open-loop retention and state-norm growth, not closed-loop error correction.
- **Recommendations:**
  - (i) Normalise the donor test by the cross-item distance: PDI = (1 − cos(zero, donor)) / (1 − cos(zero_i, zero_j)).
    - CHECK: at init, INJECT cross-item cos16 is 0.44–0.61 and PDI16 is 0.057, 0.014, 0.0011, 0.0001 at n = 2, 4, 8, 12.
    - READ-SLOTS has cross-item cos 0.84–0.93, with PDI16 0.08–0.12. **Raw delta cos conflates contraction with cross-item similarity.**
  - (ii) A split-perturbation test:
    - perturb along the realised progress direction S_t − S_{t−1}, and along random directions projected onto the read-visible subspace (the span of the think queries per delta head) but orthogonal to it;
    - report λ_prog and λ_trans.
    - The theory predicts: s3 has λ_prog < 0; converged models have λ_prog ≈ 0 and λ_trans ≤ 0; the target class has λ_prog ≈ 0 pre-completion and both < 0 post-completion.

**Reconciled statement (DERIVED).**
- Anil's finding (path independence ⇒ better upward generalisation) concerns T (and R).
- Our failure is P. Converged models avoid P, but they are I, not G0/G1, which is why they are not STABLE in fixed-K recipes.
- The forbidden contraction is **pre-completion contraction along progress coordinates**. Transverse contraction and post-completion contraction are wanted.

---

## 4. The delta-head Jacobian

### Theorem 4 (open loop, one head; PROVED)
Setting: k̂_τ unit, β_τ ∈ (0, 2), α_τ ∈ (0, 1), τ = 1..L real tokens, and Δ′ = ΔP + C as in §0. Let H_τ = I − β_τ k̂_τ k̂_τᵀ, Q = H₁⋯H_L, K = span{k̂_τ}, r = dim K.

- (a) The derivative is X ↦ XP. As an operator on ℝ^{64×64} (Frobenius norm), its singular values are those of P, each with multiplicity 64.
- (b) P = (Π_τ α_τ)Q. Every H_τ fixes K^⊥ pointwise and maps K into K, so Q = Q|_K ⊕ I_{K^⊥}. The singular values of P are:
  - **Π α exactly**, with multiplicity 64 − r;
  - **Π α·σ_i(Q|_K)**, i = 1..r.
- (c) **Dissipation identity.** With v₀ = v and v_τ = H_τ v_{τ−1} (the row action applies H₁ first):

  ‖v‖² − ‖H_L⋯H₁v‖² = Σ_τ β_τ(2 − β_τ)(k̂_τ·v_{τ−1})².

  (Each step: ‖Hv‖² = ‖v‖² − β(2−β)(k̂·v)².)
- (d) If every β_τ ∈ (0, 2), then ‖Q|_K‖ < 1.
  - Proof: equality ‖Qᵀv‖ = ‖v‖ forces k̂_τ·v_{τ−1} = 0 for all τ, hence v_τ = v for all τ, so v ⊥ K. Compactness of the unit sphere of K finishes it.
  - So ‖P‖ = Πα when r < 64, and ‖P‖ < Πα when r = 64.
- (e) **Isometry iff Πα = 1 and every β_τ ∈ {0, 2}.** This follows from (c), since β(2−β) = 0 iff β ∈ {0, 2}.
  - In the code, α = σ(·) < 1 and β = 2σ(·) ∈ (0, 2) open. So **for every parameter value and every input, ‖P‖ ≤ Π_τ α_τ < 1**.
  - The delta channel is a strict contraction by construction. Near-isometry needs β → 0 or 2 and α → 1, i.e. gate logits → ±∞.
- (f) Special cases.
  - Orthonormal keys: σ(Q|_K) = {|1 − β_τ|}. β = 1 gives exact annihilation; β = 2 gives a reflection.
  - Two keys with cos c and β = 1: σ(Q|_K) = {|c|, 0}. CHECK `thm4_checks.txt`: 0, 0.5, 0.9, 0.99.
  - β = 1 annihilates the **last** written key direction exactly (P k̂_L = 0), but earlier keys only partially when keys are correlated.

### 4.2 At init (β ≡ 1, α ≡ 0.99331; CHECK `init_jacobian.py`, `init_jacobian2.py`, real GPT-2 block-6 delta-head keys)

| n | L | α^L = ‖P‖ | σ_max(P\|_K) | ρ(P\|_K) | mean \|cos\| between keys |
|---|---|---|---|---|---|
| 2 | 8.5 | 0.945 | 0.915 | 0.846 | 0.74 |
| 4 | 13.9 | 0.911 | 0.900 | 0.875 | 0.76 |
| 8 | 25.9 | 0.841 | 0.839 | 0.833 | 0.77 |
| 12 | 38.4 | 0.773 | 0.772 | 0.770 | 0.78 |

- With GPT-2's correlated keys, Q barely contracts its slowest key-space direction (σ_max(P|_K)/α^L = 0.97 → 0.9994).
- For comparison, i.i.d. random unit keys give σ_max(Q|_K) = 0.43 at L = 8 and 0.84 at L = 30, with no α (`thm4_checks.txt`). Random keys are strongly contracted in key space; GPT-2's are not.
- **The init's per-pass contraction is α^L, i.e. it is set by input length.**
- Written content lives in K (the rows of C lie in K). Its early increments shrink faster (log step ratio −0.4 to −0.57) than the slowest rate.
- Length prediction, from α^L at init:

  | L (≈ n) | 14 (4) | 29 (9) | 38 (12) | 64 | 300 | 450 (ProsQA) |
  |---|---|---|---|---|---|---|
  | α^L | 0.91 | 0.82 | 0.77 | 0.65 | 0.13 | **0.049** |

  - **P4.2 (DERIVED):** at fixed recipe, the lottery and DEPTH get worse with think-input length, because the open-loop gap 1 − α^L that the closed loop must beat grows ≈ L·0.0067 (§4.3).
  - A clean causal test: INJECT with k extra writing filler tokens (L → L + k at the same n) should raise the failure rate. A write mask restricted to the last 8 tokens should behave like READ-SLOTS.
- READ-SLOTS (M = 8 slot writes, keys from [EOT, expr] + 8 × (wte(" .") + N(0, 0.02²))):
  - α^8 = 0.948 per pass for **every** L;
  - slot keys have mean |cos| 0.91, and σ_max(P|_K) ≈ ρ ≈ 0.947;
  - open-loop delta cos16 ≈ 0.99 at every n, with PDI16 0.08–0.12 flat in n. INJECT's PDI16 falls from 0.057 (n = 2) to 0.0001 (n = 12).
  - **P4.3 (DERIVED):** READ-SLOTS's delta cos and PDI are flat in n, while INJECT's grow with n among non-converged models. READ-SLOTS's lottery does not worsen with input length. If READ-SLOTS still produces B-class seeds at small n, the α^L account is insufficient on its own.

### 4.3 The closed loop

Full Jacobian of one pass, with u depending on S through h_w (γ > 0):
- J_ΔΔ[X] = XP + ∂_u(ΔP + C)·du[X]
- J_ΔA[Y] = ∂_u(ΔP + C)·du[Y]
- J_AA[Y] = Y + ∂_u(Σφ(k)vᵀ)·du[Y]
- J_AΔ[X] = ∂_u(Σφ(k)vᵀ)·du[X]

Here du[·] = γ · J_LNw · J_{B6..11} · fade · W_o · [g_add J_N(·) ‖ g_del (·)q̂_τ]_τ.

So E := J − blockdiag(I, P⊗) satisfies:
- **E ≡ 0 when γ = 0, or W_o = 0, or fade = 0.** This holds at init, and for all steps ≤ 2000.
- ‖E_ΔΔ‖ ≤ γ·fade·g_del·‖W_o^{del}‖·‖J_B‖·‖J_LNw‖·‖∂_u(ΔP + C)‖·‖Q̂‖, where Q̂ is the fixed (input-only) think-query matrix and ‖Q̂‖ ≤ √T.
- **Invisible directions.** Content in key-space directions orthogonal to every think query q̂_τ never enters R(S). On those, J = P exactly: the closed loop cannot act there.
- **Necessary condition for non-contraction** along a progress direction u: ‖(P + E)u‖ ≥ ‖u‖ forces ‖Eu‖ ≥ (1 − ‖P‖)‖u‖ ≥ (1 − Πα)‖u‖ (PROVED, triangle inequality).
- **Sufficient condition:** ‖Eu‖ ≥ (1 + ‖P‖)‖u‖, or ‖Pu‖ + ‖Eu‖ ≥ ‖u‖ with Eu aligned to Pu.
- **LN_w attenuation at init** (CHECK `ln_atten.py`): σ(h_w) after block 11 is 13–18 at non-sink tokens, so ‖J_LNw‖ = 0.050–0.063 (for comparison ‖J_LN6‖ ≈ 0.27).
  - The remaining factors must therefore supply ≳ (1 − α^L)/0.055: **1.6 at n = 4, 3.2 at n = 8, 4.6 at n = 12** (DERIVED; ln_w is trainable, so this is an init-time statement).
- When the closed loop cannot undo the contraction:
  - (i) while γ·fade·‖W_o‖ ≈ 0;
  - (ii) on read-invisible directions;
  - (iii) when the trunk's write sensitivity is not aligned with the progress direction (Eu ⊥ Pu or anti-aligned).
- When it can: once ‖Eu‖ exceeds the gap along the realised progress direction. Converged models' g ≥ 1 says they reached this.

---

## 5. Why SGD can select the contractive regime

### 5.1 The open-loop phase is the bad set (PROVED)
For steps ≤ 2000 (γ = 0):
- (i) F is affine, F(A, Δ) = (A + a(x), ΔP(x) + C(x)) (§0).
- (ii) From S₀ = 0: A_t = t·a(x) and Δ_t = C Σ_{s<t} P^s = Δ*(I − Pᵗ), with Δ* = C(I − P)^{−1}. From any Δ₀: Δ_t − Δ* = (Δ₀ − Δ*)Pᵗ. The convergence is geometric at rate ‖P‖ ≤ α^L < 1 (Thm 4) and is exactly path-independent in Δ.
- (iii) rms read: N(φ(q)ᵀ·t a) = N(φ(q)ᵀa) for t ≥ 1 (scale invariance, up to the 1e-6). den read: φ(q)ᵀ t a/φ(q)ᵀ t z is also t-invariant. The none read is ∝ t (a counter).
- (iv) Hence under rms or den, the only pass-index information reaching any decoder during steps ≤ 2000 is the pass token (if present) and the decaying transient Δ* Pᵗ.

So for the first 2000 steps the model learns on the bad set: memory = input summary + geometric clock. The cheapest solutions are pass-token decoders computing e_{t+1} from (input summary, t), which is depth-limited. The ramp then starts from inside that basin.

DATA consistent with this: s3 is ahead at 8k (aux 0.357 vs 0.383), plateaus later, and its final state is init-like (κ ≈ 0.8).

### 5.2 Differential-mode attenuation (PROVED under the smoothness assumptions)

**Lemma 5.2.**
- Let S_r(θ) = F(S_{r−1}(θ); θ) with S₀ = 0, and Δ_r := S_{r+1} − S_r.
- Assume on a neighbourhood of the trajectory:
  - ‖∂_S F‖ ≤ κ < 1;
  - ‖∂_θ F‖ ≤ B;
  - ‖∂_S F(S) − ∂_S F(S′)‖ ≤ H‖S − S′‖;
  - ‖∂_θ F(S) − ∂_θ F(S′)‖ ≤ M‖S − S′‖.
- Then for every parameter direction δθ:
  - (a) common mode: ‖δS_r‖ ≤ B‖δθ‖/(1−κ), which is **not attenuated**;
  - (b) differential mode: ‖δΔ_r‖ ≤ ‖δθ‖(Bκ^r + rκ^{r−1}K‖Δ₀‖), with K = M + HB/(1−κ).

**Proof.**
- Let J_r = ∂_S F(S_{r−1}) and b_r = ∂_θ F(S_{r−1})δθ. Then δS_r = J_r δS_{r−1} + b_r, which gives (a).
- For (b): δΔ_r = J_{r+1}δΔ_{r−1} + (J_{r+1} − J_r)δS_{r−1} + (b_{r+1} − b_r).
- Bound the pieces: ‖J_{r+1} − J_r‖ ≤ H‖Δ_{r−1}‖ ≤ Hκ^{r−1}‖Δ₀‖ (Thm 1a), and ‖b_{r+1} − b_r‖ ≤ Mκ^{r−1}‖Δ₀‖‖δθ‖, with δΔ₀ = b₁.
- Unroll the recursion. ∎

**Consequences.**
- (i) In a contractive regime, any update, aux-driven or not, moves late states mostly **together**. That improves the pass-independent content (the input summary pass-token decoders use) far more efficiently than pass-specific progress. The ratio is ≤ (1−κ)(κ^r + rκ^{r−1}K‖Δ₀‖/B).
- (ii) In the open-loop-like regime, F(S) = SP + C, so H = 0 and M = ‖∂_θP‖. The *only* first-order route to more late-pass separation is changing P itself (α → 1, β → 0/2), which yields a slower clock, not progress.
- (iii) The CE gradient on a late margin is ℓ′·∂m_r ∝ rκ^{r−1}: attenuated. The gradient of log‖Δ_r‖ is the relative sensitivity ‖δΔ_r‖/‖Δ_r‖ ≲ ‖δθ‖(B/‖Δ₀‖ + rK/κ) when ‖Δ_r‖ ≈ κ^r‖Δ₀‖: **not attenuated**. This asymmetry is what Method A exploits.

**Honest caveat.** At s3's κ ≈ 0.78, rκ^{r−1} is 1.6, 1.9, 1.8, 1.4 for r = 2, 4, 6, 8. The direct attenuation of the escape gradient is **not** small. The trap cannot be "vanishing gradient in κ" alone. It needs §5.3.

### 5.3 The closed-loop gain is a multiplicative small mode (DERIVED; Rawal–DeWeese analogy)
- Write F_ε = F₀ + εG, with F₀ the open-loop map (linear part P) and ε the closed-loop gain along progress.
- With P = 0, S_{t+1} − S_t = O((ε·Lip G)^t)‖S₁‖. Depth-t pass separation is a **degree-t polynomial** in ε at the open-loop point.
- With P ≠ 0 it is ≤ (‖P‖ + ε Lip G)^t (Thm 1 with κ_eff = ‖P‖ + ‖E‖).
- ε is a product of learnable factors that are small after the ramp: W_o (shared with the decoders, so it grows anyway), the LN_w gain (×0.05 at init), the alignment of the write projections (trained on LN₆ inputs in the open-loop phase), and γ.
- For balanced factors, ε grows as ε̇ ∝ η r ε^{2−2/r} D(t), where D is the residual drive from late-pass errors.
- This is the small-initialisation saddle with Rawal & DeWeese 2605.01288's escape time:
  - Θ(log(1/ε₀)) for r = 2;
  - Θ(ε₀^{−(r−2)}) for r ≥ 3.
- This is an analogy (their result is for feed-forward small-init layers). r is not measured.

### 5.4 Threshold / race model (CONJECTURE, with a PROVED ODE kernel)
- **ODE kernel, r = 2 (PROVED):** ε̇ = 2η(t)D(t)ε gives ε(∞) = ε₀ exp(2∫ηD). Escape (ε(∞) ≥ ε_needed) happens iff ε₀ ≥ ε_c := ε_needed·exp(−2∫ηD): a threshold on ε₀.
- For r ≥ 3 the dependence on ε₀ is polynomially sharper.
- **Mapping (CONJECTURE):**
  - ε₀ is the closed-loop progress gain at the end of the ramp (≈ 6k). It is random across seeds through data order and rounding (Summers & Dinneen 2103.04514: one-bit perturbations; our GPU-class flips).
  - ∫ηD is finite: cosine LR to 0 over 48k. D(t) falls as pass-token decoders and the open-loop clock absorb late-pass loss: a **closing window**.
  - A seed fails iff ε₀ < ε_c. When the ε₀ distribution straddles ε_c, the outcome is a lottery decided early (Frankle 1912.05671).
  - The curriculum (n ≤ 4, K = 4) lowers the needed κ_eff (Thm 1: (μ/Ld₀)^{1/n} with n = 4), which lowers ε_needed. That is consistent with s3 entering by 10k under it.

### 5.5 Noise-driven drift toward contraction (CONJECTURE)
- SGD ≈ GD on L + (η/4)·E‖∇L_B‖² (Smith et al. 2101.12176; Barrett & Dherin 2009.11162).
- Per-batch gradient norms through the loop scale with the trajectory sensitivity Σ_s ‖J^{t−s}‖. That is ~(1 − κ^t)/(1 − κ) for κ < 1, ~t at κ = 1, and exponential beyond.
- So the implicit regulariser increases with κ: a drift toward contraction ∝ η/B.
- Caveats:
  - global clipping binds on 65–100% of steps (normalised updates change the implicit term);
  - Chen & Kunin 2306.04251's stochastic-collapse theorem needs an *invariant set*. {E = 0} is invariant only as a schedule (γ = 0), not under GD for γ > 0, so their theorem does not apply strictly.
- **Prediction (P5.5):** at fixed LR, batch 32 fails more often than batch 128 (more noise, more drift to contraction). This is generic. It is useful only as a discriminating test of §5.5.

### 5.6 Proved vs conjectured, in one place
- **PROVED:**
  - Thm 1, 1′, 3, 4; Lemma 1.3; Thm 2a; Prop 3.2; Lemma 5.2;
  - the open-loop phase = bad set (5.1);
  - the r = 2 ODE threshold;
  - the necessary closed-loop condition ‖E‖ ≥ 1 − Πα.
- **DERIVED:**
  - Thm 2b–d; Cor 1.2 (A1);
  - the s3 κ estimates (the model of Thm 1d);
  - the init-time LN_w attenuation factor and the required amplification;
  - length predictions P4.2 / P4.3;
  - the H_open reading.
- **CONJECTURE:**
  - the race / threshold mapping (5.4);
  - the implicit-regularisation drift (5.5);
  - the Rawal analogy (5.3);
  - that the outcome is decided by ε₀ at about 6k.

---

## 6. Methods

### Design requirements derived above
- **R1.** No pre-completion contraction along progress (Thm 1, §3.4).
- **R2.** An escape signal for the progress gain that is not κ^t-attenuated (Lemma 5.2iii) and not multiplicative in a small gain (§5.3–5.4).
- **R3.** Transverse contraction is allowed (Anil, Thm 3).
- **R4.** Contraction after completion (Thm 3; §2.1).
- **R5.** Keep C1: identical input every pass, memory as the only cross-pass channel, removal and serial tests valid.
- **R6.** Prefer soft over hard constraints (Vorontsov 1702.00071).

### 6.A PCLT: progress-cone Lyapunov target (objective). Recommended first.

**Definition.**
- R_t := R(S_t), the think-pass read features at the pass-invariant think queries, for t = 0..K.
  - Pass t+1 computes this at block 6 anyway; R_K costs one einsum.
  - Variant: the pre-W_o reads [N(φ(q)ᵀA) ‖ Δq̂].
- d_t := ‖R_{t+1} − R_t‖ + ε₀‖R_t‖, with ε₀ = 1e-3 (a floor against log 0).
- λ_t := log d_t − log d_{t−1}. A common scale cancels, so it is scale-free.
- Per item, with known n (training only):

  L_PCLT = mean_{2≤t≤n−1} τ·softplus((λ_min − λ_t)/τ) + mean_{n≤t≤K−1} τ·softplus((λ_t − λ_post)/τ)

  with λ_min = log 0.95, λ_post = log 0.5, τ = 0.05, and weight w_P = 0.1, added to the existing loss.
- The post term needs passes beyond n, so use the window recipe K = n + U{1..4}.
- The pre term needs n ≥ 3.

**Theorem A1 (what the target certifies; PROVED by telescoping).**
- (i) If λ_t ≥ λ_min on 2 ≤ t ≤ n−1, then ‖R_{t+1} − R_t‖ ≥ e^{λ_min(t−1)}‖R₂ − R₁‖ − O(ε₀).
  - So the necessary condition of Thm 1′ holds for every n ≤ 2 + log(L_R‖R₂ − R₁‖/μ)/|λ_min|.
  - Example: with L_R‖R₂ − R₁‖/μ = 10 and λ_min = log 0.95, this is n ≤ 46.
- (ii) If λ_t ≤ λ_post < 0 for t ≥ n, the post-completion read drift is summable: Σ_{j≥0}‖R_{n+j+1} − R_{n+j}‖ ≤ e^{λ_post}‖R_n − R_{n−1}‖/(1 − e^{λ_post}) (+ O(ε₀)).
  - With λ_post = log 0.5, that is one last progress increment.
  - The answer is then correct for every K′ ≥ n wherever the answer margin exceeds L_R times this bound. Stability holds by construction on the passes where the target is met.
  - Beyond the trained K it is an extrapolation of a learned property, but a contraction-type one, which extrapolates (Thm 3).

**Theorem A2 (the escape signal is not attenuated; PROVED in the scalar model, DERIVED in general).**
- Scalar model: Δ_t = κ(θ)^t Δ₀. Then ∂λ_t/∂θ = ∂log κ/∂θ, identical at every t.
  - Compare the aux CE on margin m_t = Lκ^tΔ₀: ∂ℓ_t/∂θ = ℓ′LΔ₀ t κ^{t−1} ∂κ/∂θ.
- General case: |∂λ_t| ≤ ‖∂Δ_{t+1}‖/‖Δ_{t+1}‖ + ‖∂Δ_t‖/‖Δ_t‖ = O(tK/κ) by Lemma 5.2 when ‖Δ_t‖ ≳ κ^t‖Δ₀‖. It is a relative sensitivity.
- In the §5.4 kernel, the floor adds a drive w_P σ(·)/(κ₀ + ε) that is **O(1) at ε → 0**: ε̇ = η[rε^{2−2/r}D + w_P/(κ₀ + ε)].
  - Growth from any ε₀ ≥ 0 is then at least linear, at rate ≥ η w_P/(κ₀ + ε_needed).
  - **The threshold ε_c disappears.** Escape is deterministic and no longer decided by noise.
- This is the precise sense in which PCLT "removes the bad regime": it removes the saddle structure, given A3.
  - **Assumption A3:** the floor's gradient reaches the closed-loop parameters, i.e. the realised increment is read-visible.

**Theorem A3 (clean test; PROVED).**
- n enters only the training loss. The model, the forward pass, the input and the cross-pass channel are unchanged.
- Every C1 test (K′ = 0, removal, serial, path) is unchanged.

**Distinctions (stated as theorem conditions).**
- Gradient Flossing 2312.17306 pushes **all** Lyapunov exponents of the full state toward 0, at all times (QR).
- STARS 2605.26733 pushes toward convergent, stable dynamics throughout.
- DEQ Jacobian regularisation (Bai 2106.14342) bounds the full Jacobian at the equilibrium.
- PCLT constrains:
  - (a) only the **realised progress direction in read space** (the cone of the actual increments R_{t+1} − R_t). Transverse and read-invisible directions are free to contract, which is Anil-compatible (R3);
  - (b) with a **sign switch at completion**: a floor (no contraction) for t < n and a ceiling (contraction) for t ≥ n.
- Neither GF nor STARS is completion-conditioned. Neither targets a ratio of realised increments.

**Cost.**
- Zero extra trunk passes; it stores R_t (B × T × 768 per pass).
- The post term needs the window recipe's 1–4 extra passes (≈ +30% think passes vs K = n).

**What could go wrong.**
- (i) **Motion without progress.** The floor can be met by a clock-like or oscillating read motion that the decoders ignore. The aux loss still has to decode e_{t+1}. The floor only prevents premature convergence (Lemma 5.2iii gives unattenuated late-pass gradients), so the race can still go to the shortcut. Detected by P-A5.
- (ii) **Integrator conflict.** Converged-type solutions meet the floor by write growth (§2c). The ceiling then requires a completion-aware write, which may slow training. Remedy: turn on the post term only after entry (loss < 0.5), or with w_post = 0.03.
- (iii) **Spiky gradients from log of small increments.** The ε₀ floor and the existing clipping should handle them.
- (iv) **Wrong subspace.** The think-query read may miss what the aux decoder needs. Variant: measure increments on the aux decoder's own read at a fixed probe prompt (the nopass prompt), which costs one read-only forward per pass.
- (v) If H_open is false (the contraction sits in the additive read's algebraic decay), the floor acts on the right quantity anyway, because R includes the normalised additive read. This is robust to where the contraction sits.

### 6.B Pass-boundary contraction (parametrisation)

**B-light.**
- Apply the decay **once per pass**: α_τ ≡ 1 for all tokens except one boundary step, Δ ← α_p Δ, where α_p = σ(gate_α(mean_τ u_τ)) with bias 5 (α_p = 0.993 per pass).
- Keep the token-level β erase.

**B-full (staged merge).**
- Within the pass, run the delta rule on a staging state W starting from 0. This keeps within-pass overwrite.
- Take U_p = orthonormal basis of the top-r_max (r_max = 16) principal directions of the pass's keys (a QR of the L × 64 key matrix).
- Merge once: Δ′ = α_p Δ (I − β_p U_p U_pᵀ) + W.
- β_p ∈ [0, 1] is learned, with init 0.5. This is soft: no hard isometry (R6).

**Theorem B1 (PROVED).** The open-loop pass Jacobian has singular values:
- B-light: ‖P‖ = α_p·‖Q‖ ≤ α_p, with exactly α_p on K^⊥. Independent of L.
- B-full: **exactly** {α_p (multiplicity 64 − r_max), α_p|1 − β_p| (multiplicity r_max)}. Independent of L and of key geometry. CHECK `thm4_checks.txt`: {0.97, 0.388} for α_p = 0.97, β_p = 0.6.
- The open-loop resolution horizon and path-independence time become **designable constants**, N_open ≈ log(Ld₀/μ)/|log α_p|.
  - Per-token decay gives L·|log α|, which grows with n: 0.094 at n = 4, 0.26 at n = 12, 3.0 on ProsQA.

**Theorem B2 (what the delta rule loses and keeps; PROVED).**
- Kept: within-pass associative overwrite. Later tokens erase and replace earlier tokens' writes along their keys, in W.
- Lost: read-modify-write against the **carried** state. The token-level error term e_τ = v_τ − α Δk̂_τ becomes v_τ − W_{τ−1}k̂_τ, so a pass cannot write only the *difference* to the carried content. It replaces the content on span(U_p) by β_p.
- What compensates: the carried content is still read at block 6 (Δq̂). The trunk can compute a corrected value and write it, through the closed loop.
- B-light loses nothing within the pass: β is unchanged, and only α moves to the boundary.

**Why not hard β = 2 (Householder).**
- An exact reflection makes P orthogonal on K. Three costs:
  - (i) The open loop never forgets errors (Thm 2: √t noise).
  - (ii) Each pass reflects content in key space, while the read queries are fixed (input-only). The read Δq̂ then sees content through Pʲq̂, a different effective query every pass, which scrambles retrieval.
  - (iii) Vorontsov: hard orthogonality hurts optimisation.
- The theory recommends soft or learned (α_p, β_p), and putting the "no contraction along progress" requirement into the objective (6.A) rather than into a hard spectrum.

**Cost.**
- B-light: none.
- B-full: one QR of L × 64 per head per pass, plus one extra 64 × 64 product. Negligible against the trunk.
- `CHUNK_DELTA` is unaffected for B-light; B-full replaces it by the staging loop.

**Risks.**
- B addresses only the **open-loop, size-keyed** part. If failed seeds' contraction is closed-loop or additive, B alone is predicted not to change the lottery. That makes B a diagnostic as well as a fix.
- Changing α's granularity changes the model at init (α^L → α_p). The fade-in keeps it function-preserving at step 0 (W_o = 0).

### 6.C Self-encoded state teacher forcing (SETF; candidate, riskier)
- Teacher state T_t := sg(F(0; text = e_t)). This is the model's own one-pass state on the intermediate expression's text. In a trained model it decodes as e_{t+1}, and T₁ = S₁ exactly.
- Feed S̃_t = (1−τ)S_t + τT_t into pass t+1 during training, with τ annealed 1 → 0. Test at τ = 0, which is clean (C1).
- **Theorem C1 (PROVED, linearised):** at τ > 0 the pass-specific part of pass t+1's input differs across t by O(τ‖T_t − T_{t−1}‖) = O(τ). That is not κ^t. The serial unlock cascade (depth d needs depth d − 1 first) becomes **parallel depth-1 problems**.
- **Theorem C2 (PROVED):** if θ is a CASM (Thm 3) and every T_t ∈ B(c(e_{t+1}), r), then by convexity of the balls S̃_t ∈ B(c(e_{t+1}), r) for **every** τ ∈ [0, 1]. Good solutions solve the whole homotopy, so annealing cannot lose them.
- Risks:
  - exposure bias and representation mismatch: one pass on e_t's text vs t passes on x, e.g. different additive accumulation;
  - +60–100% think compute (n extra short passes, no backward through T);
  - prior-art proximity: CODI 2502.21074, iCoT 2405.14838, Coconut 2412.06769, GTF 2306.04406 / 2310.17561.
- Recommended only if A fails P-A5.

### 6.4 Falsifiable predictions and design
Common design:
- 3 seeds (1, 2, 3; data-order seeds identical to s185), RTX 3090 only (one GPU class).
- RMS recipe with the window K = n + U{1..4} (needed for A's post term).
- Baselines:
  - the same recipe without A or B;
  - the curriculum (n ≤ 4 / K = 4 first);
  - budget annealing (Zhang 2607.20594);
  - Grokfast (s196).
- Log λ_t (read-space) every 500 steps.

**A (PCLT):**
- **P-A1 (mechanism, early):** at 8k, median λ_t over t = 2..5 is ≥ log 0.95 (= −0.05) for all 3 seeds. The baseline is expected to show ≤ −0.10 in at least one seed (s3's whole-state analogue was −0.15).
  - Curriculum and Grokfast are predicted **not** to raise λ at 8k (curriculum acts by lowering the needed κ, Grokfast by filtering). This is the separation from them.
- **P-A2 (entry):** all 3 seeds reach train loss < 0.5 by 24k, with median entry ≤ 16k. The historical RMS baseline is 2/3 by 48k, with entries at about 19k.
- **P-A3 (path class):** final delta cos ≤ 0.65 and acc_zero ≥ 0.95 for all 3 seeds. No B-class model.
- **P-A4 (stability):** at K′ = 16 and n = 8, accuracy is ≥ 0.97. Fixed-K rms_s1 had 0.75; window_s2 had 0.997, so here A is not expected to beat the window baseline.
- **P-A5 (falsifies the contraction account):** a seed that meets the floor (λ ≥ −0.05 at 8k and 16k) but ends with loss ≥ 0.5, or lands in the B class. If the floor cannot be met at all (its loss stays high), that indicates an inability to escape, not a falsification.

**B:**
- **P-B1 (init, exact):** B-light gives ‖P‖ = α_p for every n. The baseline α^L is 0.91 (n = 4) to 0.77 (n = 12).
- **P-B2 (size):** if size-keyed contraction drives DEPTH, B raises the n = 12 handle accuracy at t = 8 from 0.45–0.50 (rms s1/s2) to ≥ 0.70, and removes the fall with n at t = 4 (0.986 → 0.931). **No change falsifies the α^L account of DEPTH**; the alternatives are target length and capacity.
- **P-B3 (lottery):** if the empirical split puts failed-seed contraction in the delta heads, B-light gives 0 B-class models out of 3. If it puts it in the closed loop or the additive heads, B predicts no change.
- **P-B4 (READ-SLOTS analogue):** α^8 = 0.948 per pass whatever L. Init PDI16 is 0.08–0.12, flat in n; INJECT's goes from 0.057 to 0.0001. Trained READ-SLOTS non-converged models have delta cos flat in n.

**Theory-level falsifiers (cheap, on existing models or evals):**
- **F1:** H_open predicts that in s185_rms_s3 the realised delta-state increment ratio matches the trained open-loop ρ(P) on test inputs within ±0.05, while in s1 and s2 it exceeds ρ(P) by ≥ 0.1. Reversed or no gap falsifies H_open.
- **F2:** P1.2's error-type split on s3's late-pass aux outputs.
- **F3:** the split-perturbation test (§3.4): s3 λ_prog < λ_trans; converged models λ_prog ≈ 0 ≥ λ_trans.
- **F4:** the filler-token length test (P4.2): +20 writing filler tokens raise the failure rate. With 3 seeds this is weak; use the per-seed 8k λ_t distribution as the continuous outcome.

---

## 7. Risks and open issues
- **Cause vs consequence.** All §1.4 evidence is on final models. A model whose writes ignore the read converges trivially *to* a contraction. §5 gives a mechanism in the causal direction (open-loop phase, attenuation, threshold), but the threshold mapping is conjecture. P-A1, P-A2 and P-A5 are the causal tests.
- **Where the contraction sits.** It is not yet known whether s3's contraction is in the delta heads (H_open) or in the additive read (algebraic dilution, Lemma 1.3). The empirical subagent is measuring this. A is robust to the answer; B is not.
- **The pass token.** All 4 nopass models are in the converged class, vs 6/11 for pass-token models. Hypergeometric P(0 of 4 | 5 of 15 failed) = 0.15, so this is suggestive only.
  - Cor 1.2 says pass-token decoders make the contractive basin cheaper (shortcut), and Thm 1′ gives nopass models a certificate.
  - A cheap test: rmsnopass with seed 3 should be converged-class.
  - Pass-token decoders also fail at an unseen t = 9 (§2.1).
- **Anil.** PCLT forbids only progress-direction contraction pre-completion. Transverse contraction is free and post-completion contraction is required. G1 (re-anchoring) is not targeted; it would need an x-consistency term.
- **Numbers that are model-based:** κ fits (the Gaussian margin model), the LN_w factor (init only), the required amplification (init only), and the 1 − cos ∝ κ^{32} inversion (it assumes |S_donor| ≈ |S*|).

## Appendix: numerical checks
All in `scratchpad/seedstab/math/`, CPU, 3 threads, idle; no .pt written.

- `gpt2lib.py`: the generator (a copy of nested170's), GPT-2 blocks 0–5 from prosqa/ckpt/gpt2_pretrained.bin (read-only), and the delta write maps.
- `init_jacobian.py` → `init_jacobian.json`: init P spectrum, dissipation identity (asserted to 1e-9 on every head and item), open-loop cos16 and step ratios, random-key baseline.
- `init_jacobian2.py` / `init_jacobian2b.py` → `.json`: P restricted to span(K), READ-SLOTS slot keys, cross-item cos, PDI16.
- `ln_atten.py` → `ln_atten.json`: σ(h_w) and the LN Jacobian norms.
- `fits.py` → `fits.json`: the s3 κ estimates.
- `fit_size.py` → `fit_size.json`: κ(n) from the deep-aux eval. Inconclusive.
- `toys.py` → `toys.json`:
  - Thm 1 and path-independence bounds: 0 violations in 400 random maps;
  - integrator share 1/√t, noise √t, growing-write share 0.707, post-completion drift.
- `thm4_checks.txt`: the two-key formula, strict contraction for β ∈ (0, 2), isometry at β = 2, near-isometry at β = 0.02 / 1.98, and the B-full spectrum.


---

# Part II: noise-driven basin selection

# Basin selection in the DeltaMem think loop: success probability and the lever it implies

**Stochastic-dynamics subagent for Theory, 2026-10-13.** Inputs:
- `FACTS.md`;
- `docs/LIT_SEED_STABILITY_2026-10-13.md`;
- `boxes/deltamem.py` and `boxes/nested170.py` (read only);
- 15 `log.jsonl` / `args.json` files in `boxes/runs_fsq/` (read only).

Scripts are in `scratchpad/seedstab/sim/`; all are CPU-only:
- `sim_selection.py` and `sim_extra.py`: Monte Carlo (MC) checks of every formula below;
- `init_contraction.py`: the exact init pass map on GPT-2 weights;
- `early_vs_outcome.py`: 8k losses against outcome;
- `lam_eff.py`.

**Status labels:**
- **[PROVED]**: a complete argument is given here.
- **[DERIVED|A]**: derived under the listed assumptions.
- **[MC]**: checked by Monte Carlo.
- **[DATA]**: post-hoc, from existing logs (descriptive, not a test).
- **[CONJ]**: conjecture.

---

## 0. Summary

- **Two basins, one order parameter.**
  - **C (contractive):** the pass map forgets its start. Per-pass information decays like L^t, and the model plateaus.
  - **P (computing):** the pass map is neutral on the content subspace (L ≈ 1) and converges.
  - **The order parameter.** λ is the effective per-pass log contraction on the content subspace. A donor-start estimate, λ_eff = ln(2(1 − delta_cos))/(2K′), separates all 15 models:
    - P: λ_eff ∈ [−0.009, +0.004];
    - C: λ_eff ∈ [−0.098, −0.023];
    - **init: about −0.3**. The exact init pass map gives delta-state increments shrinking by e^−0.42, e^−0.32, e^−0.28 at t = 2, 3, 4.
  - **The designed "symmetric point" (equal-norm reads) is not symmetric in dynamics:**
    - the init lies inside C's defining region (global contraction);
    - for 2000 steps the loop gain is *identically* zero, so P cannot move.
- **Master formula [PROVED for the linearised SDE, MC-checked for the cubic one].**
  - Let m be the selection variable, with dm = [b(t) + a(t) m − c m³] dt + σ(t) dW.
  - Then **P(P basin) = Φ(μ_Z / σ_Z)**, where μ_Z = ∫ e^{−A(s)} b(s) ds, σ_Z² = ∫ e^{−2A(s)} σ(s)² ds and A = ∫ a.
  - Special cases:
    - constant coefficients: Φ(b √(2/a) / σ);
    - a neutral window T_w: Φ(b √T_w / σ);
    - a C competitor with head start T_h: Φ(√(2a)/σ · [(b_P − β_C)/a + β_C e^{−r_C T_h}/(a + r_C)]).
- **The lever.**
  - The observed P ≈ 0.6–0.67 means z = Φ⁻¹(P) ≈ 0.25–0.43.
  - Reaching P = 0.95 needs z × 3.8–6.5. The levers fall into two classes:
    - **√-levers:** batch size, window length, Grokfast's amplification, curriculum-as-window. These would need **15–42×** changes.
    - **Tilt levers:** these change μ_Z linearly, which needs a **4–7×** change in bias.
  - **Only tilt is practical.** For this architecture the tilt terms are:
    - the C head start during the open loop (T_h);
    - C's shortcut through the pass token, which lets the step reader ignore contraction (Lemma 4);
    - P's late-pass reward. Pass t's signal shrinks as (9 − t)/8 × E_t; pass 8 gets 1/22 of pass 1's.
  - An order-parameter feedback ("STEER") drives P → 1 provably *in the reduced model*: P = Φ((m_eff + b/a)/(σ √(1/(2(k−a)) + 1/(2a)))).
- **Two observations the model predicts [DATA, post hoc]:**
  - **The failed seeds were ahead early.** In the 10 non-reweighted models, the 4 failures hold answer-CE ranks {1, 2, 4, 6} of 10 at 8k (exact one-sided p = 0.033). Excluding HOLD, the 3 failures have the 3 lowest aux CEs of 9 (p = 0.012).
  - **Recipes that only tilt in the model's sense** (NOPASS or REWEIGHT) **converged 6/7**, against 4/8 for the rest (Fisher p = 0.18; not significant).

---

## 1. Exact facts about this architecture (the reduced model's anchors)

Notation:
- x is the think input, identical every pass.
- S = (S_a, S_d) is the additive and delta state.
- The pass map is F_θ(S; x).
- u = (1 − γ) LN_6(h6) + γ LN_w(h_w) is the write input.

**Lemma 1 (open loop) [PROVED].** For γ = 0 (steps ≤ 2000):
- **The pass map is affine and does not depend on the read.**
  - u = LN_6(h6) depends only on x (blocks 0..5 never see the memory). So every pass writes the same k, v, α, β.
  - Hence S_a,t = t·W(x) and S_d,t = B(x)(I + M + … + M^{t−1}), with M = Π_tok α(I − β k kᵀ) (token order, right-multiplied).
- **The loop gain G = ∂write_t/∂read_t is identically 0.** LN_w's parameters get exactly zero gradient.
- **No parameter setting reaches the P basin while γ = 0.** Every gradient step in [0, 2000] shapes W_o, the trunk and the decoders for open-loop (input-determined) memories only.

**Lemma 2 (RMS / DEN reads are pass-blind in open loop) [PROVED].**
- **RMS:** φ(q)ᵀ S_a,t = t·φ(q)ᵀ W. RMS normalisation per head removes t exactly, up to the 1e-6 epsilon.
- **DEN:** the ratio (t φᵀW)/(t φᵀz) is also t-free.
- So for every t ≥ 1 the additive read is identical. The pass-1 read of S_0 = 0 is 0.
- **Without normalisation** (the original NOFADE recipe) the read grows ∝ t, so the norm is a pass counter.
- **A donor start D:** the RMS read direction of D + tW forgets D like 1/t. This is a power law: the read is asymptotically path-independent even though the state is not.

**Lemma 3 (delta channel at init) [PROVED bound + computed].**
- **The bound.** With gate weight 0, β = 1 and α = σ(5) = 0.99331. Each factor α(I − kkᵀ) has operator norm α, so ‖M‖ ≤ α^T. The clean-trajectory increments are S_{t+1} − S_t = B M^t.
- **Computed** on pretrained GPT-2 (block-6 ln_1 and c_attn on h6 for 240 training-distribution items, n ~ U{1..8}, T = 5–29 tokens):
  - **Spectral radius:** ρ(M) has median 0.904, which equals α^T. That eigenvalue lives off the key span, which receives no writes.
  - **Increments (key span):** the state step ratio log(‖BM^t‖/‖BM^{t−1}‖) has medians **−0.51, −0.42, −0.32, −0.28, −0.25, −0.24** at t = 1..6. At t = 2 it ranges from −0.36 (n = 2) to −0.46 (n = 8).
  - **Comparison with trained models:**
    - the failed s3 ends at −0.175, −0.150, −0.095 (t = 2, 3, 4);
    - s1 ends at +0.005, +0.057, +0.108.
  - **Training must raise the delta channel's λ from about −0.3 to about 0.** s3 stopped part-way.

**Lemma 4 (contraction ⇒ path independence and a provable aux-loss floor without the pass token) [PROVED].**
- Let F_x be L-Lipschitz with L < 1 (any norm). Then:
  - ‖S_{t+1} − S_t‖ ≤ L^t ‖S_1 − S_0‖;
  - from two starts, ‖S_t − S′_t‖ ≤ L^t ‖S_0 − S′_0‖.
- **Without the pass token (NOPASS).** Let R be the step reader's logit map, ℓ-Lipschitz in S. Its logits at passes t and t+1 differ by at most ε_t = ℓ L^t ‖S_1 − S_0‖ in sup norm.
  - Take any position where the targets e_{t+1} ≠ e_{t+2} differ.
  - Use |log softmax(z)_y − log softmax(z′)_y| ≤ 2‖z − z′‖_∞ and p(y) p(y′) ≤ 1/4.
  - Then the two passes' summed CE is ≥ 2 ln 2 − 4ε_t.
- **The floor grows to 2 ln 2 nats per differing token** as L^t → 0. Consecutive Feng steps differ at almost every position after the reduced operation (the string shortens by 2 tokens).
- **With the pass token the bound fails.** The reader's input differs between passes, so it can emit different outputs from identical states and compute the missing reductions itself. One-shot depth limits this; s3's step-aux accuracy is 0.99 / 0.997 / 0.96 / 0.87 at t = 1–4.
- **So the pass token removes the aux loss's direct penalty on contraction.** NOPASS restores it. This is the cleanest first-principles tilt in §4.

**Order-parameter estimate from the donor test [DERIVED|A].**
- **Assumptions:** the zero-start and donor-start gaps decay at a constant per-pass rate, with comparable state norms. The initial relative gap is 1 (0 vs D).
- **The estimate:** L^{16} ≈ ‖S_16 − S′_16‖/‖S‖ ≈ √(2(1 − cos)), so λ_eff = ln(2(1 − delta_cos))/32.
- **Per model** (`lam_eff.py`):

| group | models | λ_eff |
|---|---|---|
| converged (loss ≤ 0.24) | all 10 | −0.009 … +0.004 (≈ 0, neutral: the donor gap does not shrink in 16 passes) |
| failed | nofade −0.024, rwwin −0.023, hold −0.050, den −0.061, **rms_s3 −0.098** | |

- **Cross-check:** s3's −0.098 matches its own clean step ratio at t = 4 (−0.095).
- **The basins:**
  - **C:** λ ≤ −0.02, a single global attractor;
  - **P:** λ ≈ 0, path-dependent.
  - Locally the converged models are mildly contracting: relative-perturbation slope −0.038 / −0.004. That is the "robust latching" of a multistable state machine. They are *globally* path-dependent.
  - The relevant bifurcation of the pass map is therefore **one attractor → many**. It is crossed by raising the content-subspace loop gain G past ≈ 1.

---

## 2. Reduced model and master formula

**Model.**
- Let m be the selection (share) coordinate:
  - m > 0: content is carried by a read-dependent, non-contractive loop (P);
  - m < 0: content is carried by the input-determined, contractive state (C).
- λ ≈ λ_C for m ≪ 0 and λ ≈ 0 for m ≫ 0, so λ is a monotone readout of m.
- The SDE is:

  dm = [ b(t) + a(t) m − c m³ ] dt + σ(t) dW,   m(0) = m_0 (≈ 0: nothing learned at init).

- Its coefficients:
  - **a(t) ≥ 0 is the rich-get-richer gain** between the two pathways (gradient starvation, Pezeshki et al. 2011.09468). Whichever pathway W_o and the decoders lean on gets more gradient. a = 0 while a pathway is cut.
  - **b(t) is the information bias.**
  - **σ² is the SGD noise in this coordinate.** With continuous time τ = η_eff·k: σ² = η_eff σ_g²/B, where σ_g² is the per-example gradient variance along m.
  - **Adam:** with a slowly varying preconditioner, Adam rescales drift and noise in a coordinate equally; that sets η_eff [DERIVED|A].
  - **Clipping** multiplies η_eff by min(1, 1/‖g‖) on each step.

**Theorem 1 (selection probability) [PROVED for c = 0; DERIVED|A + MC for c > 0].**
- Variation of constants gives m(t) = e^{A(t)}[m_0 + ∫_0^t e^{−A}b ds + ∫_0^t e^{−A}σ dW].
- If A(t) → ∞, sign m(t) → sign Z, with Z = m_0 + ∫_0^∞ e^{−A}b ds + ∫_0^∞ e^{−A}σ dW. Z is Gaussian by Itô isometry.
- Hence

  **P(P basin) = Φ(μ_Z/σ_Z),  μ_Z = m_0 + ∫_0^∞ e^{−A(s)} b(s) ds,  σ_Z² = ∫_0^∞ e^{−2A(s)} σ(s)² ds.**

- **The cubic term** acts only once |m| has outgrown the noise, by which point the sign is fixed (exit from an unstable point, small-noise regime). The MC deviation is ≤ 0.006 in every case below.

**Corollaries** (all checked by MC, `sim_selection.py`, N = 20,000 paths; MC vs formula):

| # | regime | P(P basin) | MC check |
|---|---|---|---|
| C1 | constant a, b, σ | Φ(b √(2/a) / σ) | b = −0.02…+0.05: 0.385/0.389, 0.501/0.500, 0.556/0.556, 0.616/0.611, 0.766/0.760 |
| C2 | neutral window T_w (a = 0), then a | Φ(b (T_w + 1/a) / (σ √(T_w + 1/(2a)))) → Φ(b √T_w/σ) | 0.633/0.633, 0.732/0.740, 0.627/0.626, 0.406/0.410 |
| C3 | swept gain a = κ(t − t_0) (κ t_0² ≫ 1) | Φ(√2 (π/κ)^{1/4} b/σ); from a = 0 at the ramp start: Φ((π/κ)^{1/4} b/σ) | κ 0.1/0.025/0.4: 0.628/0.631, 0.688/0.682, 0.589/0.594 |
| C4 | race: b(t) = b_P − β_C c(t), c = 1 − e^{−r_C(t+T_h)} (C's progress with head start T_h) | Φ( √(2a)/σ · [ (b_P − β_C)/a + β_C e^{−r_C T_h}/(a + r_C) ] ) | 0.503/0.500, 0.412/0.412, 0.557/0.556, 0.385/0.389, 0.470/0.470 |

**How fast P → 1.**
- The tail is P_fail = Φ(−z) ≈ e^{−z²/2}/(z√(2π)).
- In the window regime z = b√T_w/σ, so **P_fail ~ exp(−b² T_w B / (2 σ_g² η_eff))**: Gaussian in the bias, exponential in window × batch.
- In C4 the head start enters as e^{−r_C T_h} *inside* z. That is a double-exponential sensitivity to how long C learns before P can move.

**Calibration [DATA + DERIVED|A].**
- Pooled 9/15 gives z ≈ 0.25; RMS 2/3 gives z ≈ 0.43.
- **The time scale 1/a.** Successes enter at T_entry ≈ T_d + (1/a) ln(m_c/Z), Z | Z > 0. For z_0 = 0.25–0.43, SD(ln Z) = 1.05–1.08 [MC].
  - Observed entry steps of the 9 successes other than reweight_s1: mean 20.6k, SD 3.6k. So **1/a ≈ 3.3k steps**.
  - The 41.7k entry (reweight_s1) would need ln Z about 6 units below typical. That is implausible within one recipe; REWEIGHT's a is presumably different.

**The lever theorem [DERIVED|A, from C1–C4].**

| target | factor in z | √-levers (batch, window, Grokfast's 1+λ) | bias |
|---|---|---|---|
| 0.60 → 0.95 | × 6.5 | × 42 | × 6.5 |
| 0.67 → 0.95 | × 3.8 | × 15 | × 3.8 |
| → 0.99 | × 5.4–9.2 | × 29–84 | × 5.4–9.2 |

- **Noise, window and symmetric amplification cannot reach P → 1 at sane cost.** Only a change in μ_Z (tilt) or removal of the C attractor can.

---

## 3. The observed features

**3.1 Deterministic init, bimodal outcome [PROVED in the model].**
- The init sits at m_0 ≈ 0, an unstable point. Each data order is one draw of W.
- The final state is one of the two attractors, chosen by sign Z, Z ~ N(μ_Z, σ_Z²).
- z ≈ 0.25–0.43 means the tilt is about a third of the noise SD.

**3.2 Floating-point rounding flips outcomes [DERIVED|A].**
- **What the bifurcation alone can do.**
  - A perturbation δ of m at time s shifts Z by δ e^{−A(s)}. It flips the outcome with probability ≈ φ(z)|δ| e^{−A(s)}/σ_Z.
  - Near the symmetric point it is amplified by **e^{A(t) − A(s)} = exp(∫_s^t a)**.
  - With a ≈ 1/3.3k per step over the 4–6k-step decision window, that is only about **e^{1.2–1.8} ≈ 3–6×**.
  - A rounding error of 2⁻⁸ (bf16 trunk) to 2⁻²⁴ (fp32) relative cannot change sign Z through m alone.
- **What the observed flips imply.** The rounding perturbation must first grow in the *transverse* (chaotic) directions of training with Lyapunov exponent Λ, until the two runs' per-batch gradients along m decorrelate:

  Λ ≳ ln(σ_Z/δ_0)/T_d ≈ 7e-4 (bf16) to 2e-3 (fp32) per step.

  That is a doubling time ≤ 350–1000 steps, consistent with a chaotic early phase (Frankle 1912.05671; Summers & Dinneen 2103.04514).
- **Flip probability [PROVED, MC].** Let ρ_Z be the shared fraction of the decision variance (the part accumulated before decorrelation, k* = ln(ℓ/δ_0)/Λ). Two runs with the same data order flip with

  **P_flip = 2[Φ(z) − Φ₂(z, z; ρ_Z)]** (z = 0: arccos(ρ_Z)/π, Sheppard).

  MC: ρ 0 / 0.5 / 0.9 gives 0.480 / 0.322 / 0.140 at z = 0.25. For k* ≪ T_d (ρ_Z → 0), P_flip → 2p(1 − p) ≈ 0.45–0.48.

**3.3 The failed run was ahead at 8k [DERIVED|A + MC + DATA].**
- **In C4:**
  - C's progress is visible in the loss immediately;
  - P's is hidden until commitment, about (1/a) ln(m_c/Z) after T_d;
  - a faster C (larger r_C) both lowers the early loss and lowers z.
- **MC (S2).** Per-seed C speed was varied lognormally (SD 0.5). P(success) by tercile of early C progress: slowest 0.511, middle 0.442, **fastest 0.393**.
  - The sign of m at t = 3/a already predicts the final basin with accuracy 0.984. **The decision is early; the loss shows it late.**
- **[DATA] Answer CE and aux CE at 7.5–8.5k** (`early_vs_outcome.py`; all 3090):

| model | answer CE @8k | aux CE @8k | entry | final | basin |
|---|---|---|---|---|---|
| s185_den_s1 | **1.785** | **0.327** | never | 0.642 | C |
| s185_rms_s3 | **1.840** | **0.349** | never | 1.031 | C |
| s182_window_s1 | 1.848 | 0.361 | 23.5k | 0.239 | P (borderline) |
| s182_nofade_s1 | **1.858** | **0.355** | never | 0.560 | C |
| s185_rms_s1 | 1.907 | 0.379 | 18.3k | 0.006 | P |
| s182_hold_s1 | 1.987 | (0.424, aux targets differ) | never | 1.060 | C |
| s185_rms_s2 | 2.027 | 0.416 | 17.5k | 0.003 | P |
| s190_rmsnopass_s1 | 2.027 | 0.423 | 21.0k | 0.002 | P |
| s182_window_s2 | 2.052 | 0.416 | 16.2k | 0.001 | P |
| s190_rmsnopass_s2 | 2.122 | 0.435 | 25.1k | 0.228 | P |

- **The lowest early loss marks failure:**
  - answer CE: failure ranks {1, 2, 4, 6}, exact one-sided p = 0.033;
  - aux CE (HOLD excluded): failures are the 3 lowest of 9, p = 0.012.
  - At 6k the separation is weaker; at 4k it is absent. That fits a decision made during the γ ramp (2–6k).
- **Caveat: recipes differ.** The only clean within-recipe comparison is RMS s1/s2/s3: s3 is lowest on both CEs and fails.

**3.4 Curriculum (n ≤ 4, K 4) helping [DERIVED|A, two channels].**
- **(i) Barrier.** Leaving C means crossing G ≈ 1 on content directions. There the K-pass error amplification is ~linear in K (critical slowing), so the barrier height is ΔU(K) ∝ K.
  - The Kramers escape rate from C is k_K ≈ (√(U″_C |U″_b|)/2π) e^{−2ΔU/σ²}.
  - Halving K multiplies k_K by e^{ΔU(8)/σ²}. That is exponential, so curriculum and budget annealing act mainly on *escape*.
  - **Data bound.** 0 escapes in 6 failed runs × ~40k steps gives k_K(K = 8) ≤ 1.25e-5 per step (95%), so mean escape time ≥ 80k steps.
- **(ii) Noise.** Items with n > the current one-shot depth contribute high-loss, mostly basin-uninformative gradient. That raises σ_g and saturates clipping (clip rate 1.0 in failures). Removing them raises z ∝ 1/σ_g.
- **Not a pure bias effect.** At K = 4 the C transient covers more of the supervised passes, which by itself would *lower* b_P.
- **Signature that separates it from tilt:**
  - escape gives exponentially distributed entry times (CV ≈ 1), often right after stage changes, with the order parameter first on the C side;
  - tilt gives tight entry times (CV ≈ 0.2) and an order parameter that is never on the C side.

---

## 4. The drift bias at the designed symmetric point, from first principles

**Phases:**

| phase | steps | what happens | effect on b |
|---|---|---|---|
| 0 | 0–500 | Fade 0 → 1. Only W_o is learnable on the branch path, so Rawal's count is r = 1, a fast escape: this is not a polynomial lottery. g_add equalises read norms, so the W_o gradient has no magnitude bias. | **Information bias: b < 0.** By Lemma 2 the RMS integrator's read is pass-blind; by Lemma 3 the delta read carries a decaying pass signal. With the pass token neither matters for the step reader, so both halves serve as input encoders: an input-determined memory, which is C's defining property. [DERIVED|A] |
| 1 | 500–2000 | G ≡ 0 (Lemma 1): P is unreachable. Everything learned (W_o, trunk read-in, decoders) is fitted to open-loop, contractive / pass-blind memories. | **C head start T_h ≈ T_γ + ramp/2 ≈ 4k steps** in C4. Deterministic and identical across seeds, apart from data-order noise in C's speed. [PROVED that P cannot move; DERIVED that this lowers z] |
| 2 | 2000–6000 | γ ramps, so a(t), b_P(t) ∝ γ: a swept bifurcation (C3). Ramp speed is only a quarter-power lever: a 16× slower ramp doubles z. | **The decision window.** |
| 3 | 6000 – ~8k | m's sign is fixed; losses diverge visibly only at 12–20k. | — |

**Sources of b after the loop closes:**
- **b_P, the late-pass reward.** Only late passes discriminate P from C. Their aux signal shrinks with the item count n ≥ t and with E_t. The aux sum is dominated by passes 1–3, where C is already accurate.
- **The pass token lowers b_P.** By Lemma 4, with the token the aux loss does not penalise contraction directly; without it, any contraction pays ≥ 2 ln 2 − 4ℓL^t D per differing token pair.
- **β_C, r_C.** The contractive delta channel (λ_d ≈ −0.3 at init) and the one-shot reasoning of the full-trunk step decoder make C fast. That is the "hare".

**Conclusion [DERIVED|A].**
- **The equal-norm calibration makes the init symmetric in scale, not in information or dynamics.** Every term before 6k tilts toward C, and only the late-pass residual tilts toward P.
- The observed z ≈ 0.25–0.43 > 0 says the late-pass reward slightly outweighs the head start.
- **What makes b strongly positive**, each a term in C4:

| lever | term in C4 | change |
|---|---|---|
| (a) | T_h → 0 | Close the loop from step 0 (γ = 1 once W_o ≠ 0; at W_o = 0 the writes are still input-only, so step-0 function preservation is unaffected). Or do not train the step reader while γ ≈ 0. |
| (b) | b_P ↑, β_C ↓ | **NOPASS.** Contraction is then penalised by the aux loss itself (Lemma 4). |
| (c) | b_P ↑ | **REWEIGHT** (equal per-pass signal), or a stronger late-pass dose w_t ∝ (Ē/E_t)^ω, ω > 1. |
| (d) | r_C, β_C → 0 | A **shallow step reader**: it can only render what is shallowly present in S_t (§7, M3). |

**[DATA] Recipes with (b) or (c):**

| group | models | converged |
|---|---|---|
| (b) or (c) | reweight ×2, reweight+nopass ×2, rms+nopass ×2, rwwin ×1 | 6/7 |
| the rest | window ×2, hold, nofade, den, rms ×3 | 4/8 |

Fisher one-sided p = 0.18. This is direction-consistent, not evidence.

---

## 5. Stochastic collapse (Chen, Kunin et al. 2306.04251)

**The invariant set.**
- Let G_d = u_d v_d be the loop gain on content direction d: the trunk's propagation of the read into d, times the write's sensitivity to d.
- Both gradients vanish at u = v = 0, and per-example noise in each factor is ∝ the other factor, so the noise is multiplicative.
- {G_d = 0} is therefore an (approximately) invariant set of SGD. With γ = 0 it is exactly invariant.

**Condition [PROVED for the scalar recursion; DERIVED|A for the network].**
- Near the set, q_{k+1} = q_k(1 + η(β + ξ_k)), with β = −∂L/∂G (mean growth per unit gain) and Var ξ = s²/B. Then

  E ln|q_{k+1}/q_k| = ηβ − η²(β² + s²/B)/2 + O(η³).

- So the set is **attractive (collapse) iff η_eff s²/B > 2β − η_eff β² ≈ 2β**.
- In continuous time, y = ln q obeys dy = (a − ζ²/2) dt + ζ dW, with ζ² = η_eff s²/B. The Itô drag ζ²/2 is the collapse.

**Success probability with a closing window [PROVED; MC S3].**
- Success means y reaches the commitment level h = ln(G_c/G_0) within the window τ.
- The Bachelier–Lévy formula gives P = Φ((μτ − h)/(ζ√τ)) + e^{2μh/ζ²} Φ((−μτ − h)/(ζ√τ)), with μ = a − ζ²/2.
- MC vs formula as B increases (noise ↓): 0.072/0.077, 0.159/0.163, 0.230/0.236, 0.293/0.296, 0.448/0.455, 0.616/0.622.
- **Noise hurts here when μτ > h.** That is the regime implied by P > 0.5.

**Predictions:**
- **P_fail rises with η_eff/B.**
  - **Under collapse** it rises with a sharp transition at the critical ratio (η/B)* = 2β/s².
  - **Under plain selection with b > 0** it rises smoothly: P = Φ(c√(B/η)).
  - **If successes instead needed noise to escape** (b < 0, Kramers), P_fail *falls* with η/B.
- **Clipping and Adam** enter only through η_eff. Adam is invariant to a constant gradient rescale, so clipping matters only through per-step reweighting.

**Cheap test (≤ 1 GPU-hour per seed):**
1. At a 6k checkpoint, add a scalar knob c on the read-dependent write input: u(c) = (1−γ) LN_6(h6) + γ LN_w(h_w(h6 + c·read)).
2. Compute per-example derivatives ∂ℓ_i/∂c at c ∈ {0.25, 1} over 4 × 64 items, giving β = −mean and s² = variance.
3. Report κ_SC = η_eff s²/(2βB), with η_eff from Adam's second-moment state on W_o / LN_w.

The prediction: failing seeds have κ_SC > 1, successes κ_SC < 1. Also run B = 64 vs 256 at fixed lr, 3 seeds each, read at 25k: predicted P(B = 256) > P(B = 64).

---

## 6. Early detector + restarts (brief; restarts are not new)

**Cost formula [PROVED].**
- Let a try cost c up to the detector (T_d) and C in full. Let the detector accept a run with TPR = P(accept | success) and FPR = P(accept | failure). Tries are i.i.d., so

  **E[cost] = [c + (p·TPR + (1−p)·FPR)(C − c)] / (p·TPR).**

- Checks:
  - no detector: C/p;
  - a perfect one: C + c(1 − p)/p.

**Numbers** (C = 48k, c = 8k):

| strategy | expected cost |
|---|---|
| p = 0.6, no detector | 1.67 C |
| p = 0.6, perfect detector | 1.11 C |
| p = 0.6, TPR / FPR = 0.9 / 0.1 | 1.20 C |
| p = 0.6, TPR / FPR = 0.8 / 0.2 | 1.32 C |
| a method with p = 0.9 / 0.95, no detector | 1.11 C / 1.05 C |

**The decision is early and bounded** (by 6–8k, §3.3). Restart times are therefore not heavy-tailed, and Luby's universal schedule is unnecessary: fixed restart at T_d is optimal.

**Why a method beats restarts here.**
- **Compute is roughly a wash.**
- **Registered multi-seed claims need p^k.** P(3/3) = 0.22 at p = 0.6 vs 0.86 at p = 0.95.
- **Restarts make seed sets post-selected.** Any detector used to discard seeds must itself be pre-registered.

---

## 7. Methods that raise P toward 1 (ranked by what the theory guarantees)

**M1 FAIR-START [DERIVED|A; the cheapest tilt].** Three changes, each a term in C4:
- (i) γ = 1 from step 0, or write_mix (0, 2000). This sets T_h → about 500.
- (ii) NOPASS: b_P ↑, β_C ↓ (Lemma 4).
- (iii) REWEIGHT: b_P ↑.

Prediction and constraints:
- **Predicted:** z rises from about 0.4 to at least 1.6, so P ≥ 0.9. That rests on the post-hoc 6/7 plus the head-start term.
- **C1 is kept:** identical input, memory as the only channel, read-only aux.
- **Risk:** the loop closing before W_o has a useful read may raise σ early. Check it with a dose on write_mix_start.

**M2 LATE-PASS DOSE [DERIVED|A].**
- w_t ∝ (Ē/E_t)^ω with ω ∈ {0, 1, 2}.
- μ_Z is linear in b_P, so **z(ω) ≈ z_0 + cω**: a monotone dose-response in both P and the 8k order parameter.
- Cheap, and isolates the b_P term.

**M3 SHALLOW STEP READER [CONJ for P → 1; the "remove the bad set" option].**
- The aux decoder may only render S_t: a linear or 1–2-block readout of the read, no pass token, dropped at test.
- A contraction then gets *no* aux credit at any t ≥ 1. Even e_2 needs one reduction, which the reader cannot perform. So C stops being a stationary point of the aux loss.
- **This is the recurrent-memory analogue of the weak-decoder remedies for posterior collapse** (Bowman et al. 1511.06349; Chen et al. 1611.02731).
- **Risk:** P may also become slower to learn, because the reader cannot help.

**M4 STEER, order-parameter feedback during the decision window [PROVED in the reduced model; CONJ that a faithful proxy exists].**
- During [T_γ, t_off], add a hinge loss k·relu(λ* − λ̂) on a scale-free estimate λ̂ of the content-subspace contraction. In the model this is drift −k(m − m*).
- **Theorem 2 [PROVED for the linearised model; MC S5′].** With k > a:
  - m is OU around m_eff = (b + k m*)/(k − a), with variance σ²/(2(k − a));
  - after release, **P = Φ( (m_eff + b/a) / (σ √(1/(2(k−a)) + 1/(2a))) )**.
  - MC vs formula: 0.959/0.964, 1.000/1.000, 0.921/0.923, 1.000/1.000. Without control: 0.444.
  - **P → 1 as m* grows**, and release must come after commitment, not before.
- **Prior art:** gradient flossing (Engelken 2312.17306) pushes RNN Lyapunov exponents toward 0 *throughout* training, for gradient stability. M4 is a time-limited basin-selection control whose required strength comes from Theorem 2.
- **Goodhart risks:**
  - β → 2 gives a Householder reflection: non-contractive and path-dependent, but content-free;
  - additive norm growth.
- **Use a decoder-visible statistic,** e.g. the read-level relative increment through W_o of the delta heads (non-expansive by construction, so norm growth cannot fake it). Do not use the raw state.

**Not levers (theory) [DERIVED|A]:**
- **Grokfast** (α 0.98, λ 2) amplifies slow components symmetrically: b, a and r_C. So z ≤ ×√(1+λ) ≈ ×1.7, and it also amplifies C's open-loop head start, e^{−r_C(1+λ)T_h}.
- **Noise injection** lowers z.
- **Batch size** is a √ lever.
- **Changing the ramp speed** is a quarter-power lever.
- **No lever rescues a seed that is already in C at 8k,** except escape (curriculum or annealing, barrier ∝ K).

---

## 8. Falsifiable predictions (≥ 3 seeds, one GPU class: RTX 3090, RMS base recipe)

**Cheap per-seed observables (≤ 8k steps).**
- (O1) Answer CE and aux CE averaged over 7.5–8.5k, ranked within the arm.
- (O2) Clean delta-state step ratio at t = 2–4 at an 8k checkpoint (eval only, minutes).
- (O3) Donor-start delta_cos at K′ = 16, giving λ_eff = ln(2(1 − cos))/32, at 8k.

Predicted **C side:** O2 ≤ −0.08 or λ_eff(8k) ≤ −0.02.

| # | prediction | number | falsified if |
|---|---|---|---|
| P1 | "Hare": the seed with the lowest aux CE (O1) at 8k fails more often | P(fail \| lowest of 3) ≥ 0.5 (base about 0.4). On RMS / 3090, aux CE(8k) < 0.358 ⇒ fail (post-hoc threshold, 9/9) | across ≥ 6 new base seeds the lowest-CE seeds fail no more than the others |
| P2 | Decision by 8k: O2 / O3 at 8k separate eventual P from C | AUC ≥ 0.8 over ≥ 6 seeds; the 8k sign agrees with the final basin in ≥ 5/6 | AUC < 0.65 (then the decision is later, or λ is a consequence) |
| P3 | Entry-time law: successes enter at T_d + (1/a) ln(m_c/Z) | base SD ≈ 3–4k, right-skewed; a tilt arm has SD ≤ 2k and mean ≥ 2k earlier | tilt-arm entries as spread as base |
| P4 | FAIR-START (M1) vs base | M1 ≥ 3/3 (P ≥ 0.9; prob 3/3 ≥ 0.73); all M1 seeds on the P side at 8k (O3 λ_eff ≥ −0.015); base 3/3 has prob 0.22–0.30 | any M1 seed on the C side at 8k |
| P5 | Curriculum (budget annealing) vs tilt | Curriculum rescues by escape: some seeds start on the C side after the full-K stage begins, then cross; entry-time CV ≥ 0.5. Tilt: no seed on the C side, CV ≤ 0.3 | curriculum seeds never visit the C side (then it acts as tilt, not barrier lowering) |
| P6 | Grokfast (from step 0) | P ≤ 0.77; P(3/3) ≤ 0.46; C-side fraction at 8k not reduced; §196 (RMS s3 + Grokfast): NO-EFFECT is the modal outcome (a fresh draw at P ≤ 0.77, but a symmetric amplifier) | Grokfast 3/3 *and* its 8k order parameters all on the P side |
| P7 | Batch 64 → 256 (fixed lr) | Selection: P 0.6–0.67 → 0.69–0.81; collapse near threshold: → ≈ 1; the 8k order-parameter SD across seeds shrinks by ≈ 2 | P falls (then successes need noise: b < 0 plus Kramers) |
| P8 | Open-loop dose: write_mix (0, 4k) / (2k, 6k) / (4k, 8k) | P monotone decreasing; aux CE at the end of the open loop lower for longer loops (C ahead) | P increasing with open-loop length |
| P9 | Late-pass dose ω ∈ {0, 1, 2} (M2) | P and λ_eff(8k) increase monotonically in ω | non-monotone |
| P10 | GPU-class flip: same seed on 3090 vs 4090 | outcome flip rate ≈ 2p(1−p) ≈ 0.45 (if k* ≪ T_d); same seed on the same card reproduces exactly | flip rate ≈ 0 across cards (then decorrelation happens after T_d and rounding is not the noise source) |
| P11 | C-basin per-pass profile | Per-step conditional success declines with t. Under NOPASS the log-probit slope ≈ λ, steeper than with the token. s3 (token): −0.24 per pass, vs a state step ratio of −0.10 to −0.175 | a failed NOPASS run with flat per-pass success |
| P12 | Stochastic-collapse ratio at 6k (§5 test) | κ_SC > 1 in failing seeds, < 1 in succeeding | no separation |

**Minimum discriminating design.** On RMS / 3090, 3 seeds per arm: BASE, M1, NOPASS + REWEIGHT only, CURRIC, GROKFAST (≈ 15 runs).
- Each run stops at 25k for the binary entry readout, with an 8k checkpoint for O2 / O3: about 5 h per run on a 3090 (0.74 s/step).
- **The headline contrast:**
  - **M1:** all 3 seeds on the P side at 8k;
  - **BASE:** about 1 in 3 on the C side;
  - **CURRIC:** C side early, then escape;
  - **GROKFAST:** indistinguishable from base.

---

## 9. Claim status

| claim | status |
|---|---|
| Open loop: affine pass map, G ≡ 0, LN_w zero gradient, P unreachable for γ = 0 | PROVED |
| RMS / DEN additive read pass-blind in open loop; unnormalised read ∝ t; RMS forgets a donor as 1/t | PROVED |
| ‖M‖ ≤ α^T at init; computed ρ(M) = 0.904 and key-span increments −0.51 … −0.24 per pass | PROVED / computed |
| Contraction ⇒ path independence; NOPASS aux floor ≥ 2 ln 2 − 4ℓL^t D per differing token pair | PROVED |
| λ_eff = ln(2(1−cos))/32 separates 15/15 models (P ≥ −0.009, C ≤ −0.023) | DERIVED\|A + DATA |
| Theorem 1, P = Φ(μ_Z/σ_Z), and C1–C4 | PROVED (linear) / MC (cubic) |
| √-levers need 15–42× for P = 0.95; bias needs 4–7× | DERIVED\|A |
| fp flips need transverse chaos with Λ ≳ 7e-4 to 2e-3 per step; flip-rate formula | DERIVED\|A; formula PROVED + MC |
| Failures ahead at 8k (p = 0.033 / 0.012) | DATA (post hoc, recipes confounded) |
| Every pre-6k term tilts toward C; the pass token lowers b_P | DERIVED\|A |
| Curriculum acts by barrier ∝ K (escape) and σ_g, not by bias | CONJ (two-channel argument) |
| Stochastic-collapse condition η_eff s²/B > 2β; Bachelier–Lévy success law | PROVED (scalar) / DERIVED\|A (network) / MC |
| Restart cost formula | PROVED |
| STEER Theorem 2 | PROVED (reduced model); faithful proxy CONJ |
| FAIR-START gives P ≥ 0.9 | CONJ (theory-motivated; post-hoc 6/7) |
| Shallow reader removes C as a stationary point of the aux loss | CONJ |

## 10. Caveats

- **The reduced model is one-dimensional.** The real selection may involve several content directions, each with its own race. Per-direction failures would then compound: P = Π_d Φ(z_d), giving partial C basins (s3 uses passes partially).
- **Cause or consequence?** λ is treated as the slow selected variable. If it is a consequence of an earlier choice (e.g. which heads W_o reads), P2 fails. P2 is the test.
- **The [DATA] items are post hoc** across recipes that differ in more than the tilt terms. They motivate predictions P1 / P4; they do not test them.
- **The barrier ∝ K** and **the decision window ≈ the γ ramp** are modelling assumptions.

**Files:** `scratchpad/seedstab/sim/{sim_selection.py, sim_extra.py, init_contraction.py, early_vs_outcome.py, lam_eff.py}`. No GPU was used, the repo was not modified, and no .pt files were created.
