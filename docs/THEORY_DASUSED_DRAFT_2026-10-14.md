# DAS-USED: find the causally used representation of the intermediate values (draft rev 1, 2026-10-14)

> **Theory review (2026-10-14 20:18 EDT):** the whole-state SERIAL separator was checked against raw eval.json s170.serial["2"]: hybrid 0.53-0.96 in the 9 converged models vs 0.07-0.12 in the 7 failed (failed hybrid = donor = chance). Proof-tagged claims were taken as tagged and not re-derived. IIT 2112.00826 is not verified (Literature). Not registered; a $0 4-model pilot comes first.

> **Status: draft, not registered.** Opus math subagent for Theory; Theory reviews. Eval-only, $0, on the 16 final models of D7. All data are POST-HOC.
> Read-only use of the repo. CPU checks (3 threads, Idle priority, no GPU, no .pt written) are in `theory/dasused/`: `qoverlap.py/.json` (F-Q), `magcheck.py/.json` (M1, M2, M4), `toy_illusion.py/.json/.log` (§3 controls, §6 toy).
> Coordinator constraints folded in (2026-10-14): G-use is NOT-USED 9/9 (§203 result); energy-matched controls must reach the target energy and gate on |E_R − E*|/E* ≤ 0.2; probe-after-ablation validity = "predictions constant or accuracy ≤ majority + 0.02"; necessity by mean or resample ablation, zero only as a descriptive bound.

## 0. Summary

1. **Why.** D7's probe subspace is decodable but unused (G-use NOT-USED 9/9). Yet the converged state does carry progress causally: the whole-state SERIAL swap gives the hybrid answer at 0.53–0.96 at t = 2 (failed 0.07–0.12). The question is where, how compactly, and whether that is the model's own route.
2. **Site: the whole carried state S_t** (49,152-d; 53,248 for den). It is the only cut between pass t and the answer, and an edit to it persists through later passes exactly as the model's own content does (Prop S0, PROVED). DAS: R ∈ St(D, k), k ∈ {4, 16, 64}. Donors are structure-matched: node t's subtree and one late leaf are resampled, so y, y′ (= the counterfactual) and y_d are distinct.
3. **What IIA can and cannot show.** IIA implies decodability of v_t from R's coordinates, conditioned on the recipient (D2, PROVED). The dormant-pathway illusion fools clean-run decodability AND mean / resample necessity (D4b, D4c, PROVED by construction). A low variance share does not tell a genuine low-energy code from the illusion (toy CHECK).
4. **The illusion test that works: the on-support refit** (D5, DERIVED). Refit R inside the span of natural state variation; an illusion's IIA collapses, a used code's does not (toy CHECK).
5. **Controls:** 5 Haar and 5 exact-energy random subspaces (Prop E, PROVED: any energy between the Ky Fan bounds is reachable; gate ≤ 0.2), shuffled targets, necessity (resample primary, mean secondary) on the on-support subspace, per-n floors, transfer across passes.
6. **Magnitude:** rms / den additive reads are invariant to the head's scale (M1, PROVED + CHECK), so magnitude there acts only as an age in later mixing (M2). An equal-step norm code in an un-normalised integrator gives at most 0.17 MAP accuracy for the latest value (M3, DERIVED). β = 1 forbids onion layering at a shared delta key (M4, PROVED). Tests: norm-band decode, norm swap, age rescale, β statistics.
7. **Separator (pre-stated):** min converged vs max failed IIA*_cc at t = 2 (primary) and t = 3. SEPARATES-USED needs a margin ≥ 0.20, ≥ 7/9 converged UR-FOUND and ≤ 1/7 failed. The failed side is largely predicted by contraction (P ≈ 0.85), so the information is on the converged side: compact? on-support? delta or additive?
8. **Method implication:** a presence loss does not protect use, even on a used direction (T1, PROVED). An interchange loss does (T2, PROVED), and it also penalises contraction. But coordinates do not transfer across runs (T3, gauge, PROVED), so a target must be coordinate-free or a designated slot. New Goodhart risk: one-shot precompute into the slot.
9. **Cost:** $0. Local bf16, one card class, about 2.5 GPU-h per model (about 40 h in all), staged: a 4-model pilot of about 6 h first.

---

## 1. Facts

**Code (read-only).**

| # | fact | where |
|---|---|---|
| F1 | Think passes all see the same input [EOT, expr, " ."]. The state S is the only thing carried between passes. `override=(t, S′)` replaces S after pass t. | nested170.py 235–252 |
| F2 | The answer decoder reads S_K at **every** prompt position [EOT, expr, " ="] (fwd_read, no write), then takes the argmax over the 11 value tokens at " =". | nested170.py 254–258, 753–766; deltas5.py 91–98 |
| F3 | The memory enters at block 6. Blocks 0–5 never see it, so the read queries are a fixed function of the text and identical across passes. Writes come after the pass, from u = (1−γ)LN_6(h_6) + γ LN_w(h_w) (γ = 1 after step 6000), so the **keys depend on S** but the queries do not. | deltamem.py 187, 276–288 |
| F4 | Reads: delta r = e^{s_del} Δ q̂ (linear in Δ). Additive: φ(q)ᵀA, then none / ÷ φ(q)ᵀz (den) / per-head RMS × gain (rms). | deltamem.py 142–155, 158–176 |
| F5 | Delta rule: S ← α S (I − β k kᵀ) + β v kᵀ, with α = σ(·) (init ≈ 0.993), β = 2σ(·) ∈ (0, 2) (init 1), k and q unit-norm. | deltamem.py 69, 98, 208–210 |
| F6 | G-use: V_probe from delta reads at q_eq, pooled t = 2..8; ablation at every read_fixed call; swap = rank-1 edit at key q̂_eq after pass 2; answer_cf for t = 2; energy controls Σ^β, β ∈ {0.5, 1, 2}, best 5 of 200. | scratch/svr_guse.py 47–61, 134–146, 189–199, 205–209, 259–264 |
| F7 | §167 DAS engine: R = QR(W).Q; patch S_r + RRᵀ(S_d − S_r); PCA init of S_d − S_r; Adam with early stop on validation IIA; mean-ablation necessity; read_alignment. | das167.py 39–54, 187–241, 244 |
| F8 | s170 SERIAL: a whole-state swap of S_t from a donor in which h's operands and the root's bare operand are changed. hybrid = the recipient with h → v′. | nested170.py 1196–1205, 1274– |
| F9 | path_indep: K′ = 16 from M_0 = 0 vs from another item's 8-pass memory. | nested170.py 1226–1243 |

**Data (POST-HOC).**

| # | fact | source |
|---|---|---|
| D1 | **G-use NOT-USED 9/9.** drop(V) ≤ 0.0004. F − F_null ∈ [−0.0014, 0] with no disruption. V carries 0.6–1.1% of the delta-read energy. The "energy-matched" controls removed about 10× E_V (E 122–132 vs 11.2), so Δ_E < 0 by construction. ALL-heads V_probe on rms s1: 0.06% of the read energy (E_V 0.68 of 1,059), 97% of it in the additive slice; ABL-NULL; IGNORES (edit success 0.854). | THEORY.md §203 result; `<run>/eval_svr_guse.json` |
| D2 | **Whole-state SERIAL hybrid at t = 1 / 2.** Converged 0.38–0.90 / 0.53–0.96 (0.68–0.99 at t = 6). Failed 0.06–0.08 / 0.07–0.12 (null 0.04–0.09). Recipient-answer share at t = 2: converged 0.02–0.16, failed 0.12–0.31. | eval.json `s170.serial` (16 runs) |
| D3 | path_indep acc_from_donor: converged 0.10–0.14 (path-dependent), failed 0.22–0.64. delta_cos: converged 0.44–0.63, failed 0.76–0.98. | eval.json `s182.path_indep` |
| D4 | Accuracy at K′ = K for n = 3 / 4 / 8. Converged ≥ 0.99 at every n except window s1 and rmsnopass s2 (n = 8: 0.76 / 0.77). Failed 0.85–1.00 / 0.65–0.97 / 0.19–0.49. | eval.json `acc_by_n` |
| D5 | D7 step_recover at t = 2: converged 0.73–0.89, failed 0.13–0.37. | SEEDSTAB2 draft D7 |
| D6 | add_norm. rms: 7 runs (converged rms s1/s2 and rmsnopass s1/s2; failed rms s3, csr s1, rmsgf s3). none: 8 (converged nopass s1/s2, window s1/s2, reweight s2; failed rwwin s1, hold s1, nofade s1). den: 1 (failed). K = 12 for window, rwwin and hold (k_window 1,4); K = 8 otherwise. All runs use fast_add. | args.json |
| D7 | §167 precedent. S5: 3/3 DECOMPOSABLE at k ≤ 64 (k_min 16–64; variance share at k = 16 is 0.06–0.26). MULT: 1 positive; the k = 64 fits collapse (train IIA 0.00–0.02). | THEORY.md §167 |
| F-Q | **CHECK** (4 models, 102 items, CPU fp32). Decoder and think inputs give identical residuals entering block 6 at shared prefix positions (max \|Δ\| = 0.0). Per delta head, \|cos(q̂_eq, q̂_" .")\| is 0.21–0.75. 0.38–0.82 of q̂_eq lies inside the item's think-query span, which has rank 14–17 of 64 (99% energy) at a mean length of 18.5 tokens. So G-use's edit was visible to later think reads: its IGNORES is not an invisibility artefact. | theory/dasused/qoverlap.json |

---

## 2. Variable and site

**2.1 Variable.**
- v_t is the value of the t-th leftmost-ready reduction: `nested170.reductions`, the same variable as D7, step_recover and G-use.
- The high-level model is F_t(x, v), x's tree with node t forced to v (generalise `svr_guse.answer_cf` from t = 2 to any t).
- Counterfactual: y′ = F_t(x, v_t(x_d)). Drop a pair if any division by 0 occurs in the recipient, donor or counterfactual evaluation.

**2.2 Site: S_t, the full carried state after pass t** (all 12 heads; den adds its z block).

**Prop S0 (PROVED, from F1–F3).**
- (i) For fixed x, the answer is a deterministic function G_x(S_t): passes t+1..K on the identical think input, then the decoder.
- So S_t is a complete mediator. Every effect that anything computed in passes ≤ t has on the answer passes through S_t.
- (ii) An edit S_t → S′ is a do-intervention on that cut. Passes t+1..K then apply α-decay, β-erase and new writes to S′ exactly as to the model's own state (additive content persists undecayed; delta content decays by α per token and is erased where later keys overlap). Persistence "as the model's own writes do" is automatic: there is no clamp.
- (iii) So any state edit is delta-rule-consistent in the sense that matters. U4's rank-1 delta-write form buys minimal norm; the DAS patch is minimal in another sense: it changes only R's coordinates (D1).

**Why not the alternatives.**
- **The read at q_eq** (G-use's site) is one of about 19 query positions per pass. Each item's queries span only 14–17 of the 64 key dimensions per delta head (F-Q), and the decoder also reads at every prefix position.
- **A read patch in later passes** must be re-applied every pass. That is a clamp, a different intervention (U4's argument).
- **The per-pass write** is S_t − decay(S_{t−1}), a sub-site of S_t.
- **U4's rank-1 edit** Δ + δ q̂_eqᵀ is, up to reading the donor at its own query, the state patch with R = span{(e_{i,h} q̂_{eq,h}ᵀ)_h : e_i an orthonormal basis of V_probe} (orthonormal because each q̂_h is a unit vector). DAS frees both the key side and the value side, and both head types.

**2.3 Donors: structure-matched, with three distinct answers.**
- Copy x's tree. Re-expand node t's subtree top-down by the generator's own conditional (Alg. 1: (t1, t2) uniform among the pairs with o(t1, t2) = the node value), from a new value v_t^d ≠ v_t drawn uniformly.
- Resample one **late leaf** ℓ uniformly to a value ≠ ℓ_r. ℓ is the root's leaf child if it has one; otherwise the leaf operand of the highest reduction > t outside node t's subtree. Items without one are dropped (counted).
- Keep the pair iff y = F(x), y′ = F_t(x, v_t^d) and y_d = F(x_d) are pairwise distinct.
- **Same-value control donors:** v_t^d = v_t, with the late leaf changed. A patch must then keep y.
- **Why matched:**
  - (a) the k = D whole-state swap has a defined target: y′ iff S_t carries node t's progress and the late leaf is read from the text after t. This is s170 SERIAL generalised from h to node t (D2);
  - (b) the query geometry aligns, since only digit tokens differ;
  - (c) late-leaf variation pins the patched content: transplanting the donor's late leaf gives y_d ≠ y′, so IIA cannot be met by carrying "the donor's computation".
- Unmatched donors (any same-n item) are a descriptive transfer test.

**2.4 Sets.**
- **Fit set A:** `train_items` with rng 20301 and the run's exclusion set; n ∈ [t+1, 8]; 4,000 fit pairs and 500 validation pairs.
- **IIA set B:** recipients from the test set TE, n ∈ [t+1, 8], equal per n. Draw until there are ≥ 200 **cc** pairs (recipient and donor both answered correctly unpatched), with at most 3,000 pairs.
- A and B are disjoint by construction (training draws exclude TE). G-DISJ also checks donor texts.
- **Primary statistic: IIA_cc**, pooled over n and reported per n. IIA over all pairs and b0 = P(base answer = y′) are reported beside it.
- **Points:** t = 2 (primary) and t = 3. **Liveness (descriptive):** the whole-state matched swap for v_2 at τ = 1..4, as y′ / y_d / y shares by n. A model that does 2 levels per pass (SEEDSTAB2 D12) may already have consumed v_2 at S_2.

---

## 3. Derivation

**Setup.**
- G_x: R^D → Z_11 is the downstream map (Prop S0).
- Π_R(S, S′) = S + RRᵀ(S′ − S), R ∈ St(D, k).
- IIA(R) = P_𝒫[G_x(Π_R(S_t(x), S_t(x_d))) = F_t(x, v_t(x_d))].
- J = ∂ logits / ∂S at S_t(x). Σ = Cov(S_t) over items. supp = range Σ.

### 3.1 What IIA at rank k measures

**D1 (PROVED).** RᵀΠ_R(S, S′) = RᵀS′ and (I − RRᵀ)Π_R(S, S′) = (I − RRᵀ)S.
- So the patched state is exactly (z_d, c_r): the donor's R-coordinates z_d = RᵀS_t(x_d), and the recipient's complement.
- IIA = 1 on 𝒫 iff G_x(Rz_d + (I − RRᵀ)S_t(x)) = F_t(x, v_t(x_d)) on every pair: "downstream reads v_t from z, and nothing else from the donor".
- Sufficiency is IIA. It is not uniqueness.

**D2 (PROVED: decodability is necessary).**
- Take pairs where v ↦ F_t(x, v) is injective (no non-invertible step on node t's path to the root).
- Define f_x(z) := F_t(x, ·)⁻¹(G_x(Rz + (I − RRᵀ)S_t(x))). It returns v_t(x_d) on every IIA-success pair.
- So IIA on those pairs ≤ sup_f P[f_x(z_d) = v_t(x_d)]: a recipient-conditioned decoder of v_t from z. ∎
- A context-free probe on z can be lower if the code is context-relative. That is reported (flag CONTEXT-RELATIVE), but it is not a gate.

**D3 (DERIVED, U5's linear-evidence model).**
- Suppose v_t has a copy in R with downstream weight w_R and a copy in W ⊄ R with weight w_W.
- The patch then produces conflicting evidence, and IIA ≈ P(w_R e_R(v_d) + w_W e_W(v_r) prefers v_d). This increases with w_R/(w_R + w_W).
- So a subspace IIA reads the **dominance** of R's copy. Larger k can absorb W, and IIA then rises with k.
- Hence the k-curve, the head-type locus, and necessity as a qualifier only (U3).

### 3.2 The illusion, formalised

**D4 (PROVED, constructive; Makelov 2311.17030 in our notation).**
- Let a ∈ supp with Ja = 0, where aᵀS_t carries v_t: correlated but unused, like D7's V_probe.
- Let b satisfy bᵀΣb = 0 and Jb ≠ 0: never populated, but read.
- For w = (a + b)/√2, the first-order logit change is J w wᵀ(S_d − S_r) = ½ (Jb) · aᵀ(S_d − S_r).
- So the donor's value reaches the answer through b, a route clean inputs never use. With enough gain ‖Jb‖, IIA → 1.

**D4b (PROVED, same construction): necessity is fooled too.**
- Resample: S ← S + wwᵀ(S_o − S) changes the logits by ½(Jb)aᵀ(S_o − S).
- Mean: S ← S − wwᵀ(S − μ) changes them by −½(Jb)aᵀ(S − μ).
- Zero: likewise.
- Each ablation injects into b, so accuracy drops although the model uses neither a nor b.

**D4c (PROVED): clean decodability does not separate.** wᵀS_t = aᵀS_t/√2 + const, because b has no variance. That is exactly a's decodability, which is high.

**D4d (CHECK, toy §6):** the variance share tr(RᵀΣR)/tr Σ is about 3 × 10⁻⁴ for both the illusion and a genuine low-energy used code. A Makelov-style low-variance flag fires on both. (D7's V_probe itself is 0.6–1.1% of the energy.)

**On the reply 2401.12631:** it argues that such patches can still be legitimate causal abstractions, and it stresses held-out evaluation. We keep a stricter requirement, the model's *own* on-support route, because a supervision target needs what the model uses on clean inputs.

### 3.3 The on-support test

**D5 (DERIVED(A_supp, A_lin)).**
- If every column of R lies in supp, then the patch displacement RRᵀ(S_d − S_r) ∈ supp, and a b-type route (bᵀΣb = 0) cannot be reached.
- High IIA with R ⊂ supp then means the downstream map responds, along directions the clean computation itself varies in, the way v_t predicts. That is the operational (interventionist) meaning of "the model's own variable".
- **Residuals:**
  - (i) dormancy is graded, so supp is taken at an eigenvalue threshold ε and the result is reported as a profile over ε;
  - (ii) supp is estimated from finite samples (G-SUPP below);
  - (iii) the argument is first-order. The toy is the nonlinear check.

**Statistics.**
- **IIA_proj** = IIA(orth(P_supp R*)) with no refit. This is D4's R_∥: an illusion loses its b-part. Descriptive only: in the toy, DAS's raw R* on the USED model mixed the true variable with a dormant contribution, so its projection fell to 0.10 while the refit recovered 1.00. **So every descriptive field (locus, overlap, decode, necessity) is computed on R_on, not R*.**
- **IIA_on** = IIA of a refit constrained to supp, warm-started at orth(P_supp R*) plus the standard inits. **This is the test.**
- **ω** = the off-support share of the patch displacement energy. Descriptive only: in the toy ω is 0.17 while the effect is wholly off-support, because the dormant gain is high. A threshold of ω ≤ 0.2 would have passed the illusion.

**Estimating supp in 49,152-d.**
- Clean S_t on 6,000 fresh items (rng 20311), centred. Eigenvectors from the 6,000 × 6,000 Gram matrix.
- supp_ε = the eigenvectors with λ > ε λ_1. ε = 10⁻⁶ is primary; the profile is ε ∈ {10⁻³, 10⁻⁴, 10⁻⁵, 10⁻⁶}.
- **G-SUPP:** 1,000 held-out B states keep ≥ 99.9% of their centred energy inside supp_ε. Otherwise the point is SUPPORT-INCOMPLETE and reads at most USED-UNVERIFIED.

### 3.4 Controls

**(a) Random subspaces.**
- 5 Haar (rank-matched, rng 20321) and 5 exact-energy.
- **Prop E (PROVED).** Let G be Gaussian, R(β) = orth((Σ + εI)^{β/2} G), and E(β) = tr(R(β)ᵀΣR(β)).
  - E(β) is continuous. As β → −∞ / +∞ it tends a.s. to the bottom-k / top-k eigen-energy, the Ky Fan bounds.
  - By the intermediate value theorem every E* strictly between those bounds is attained, which covers every non-extremal rank-k subspace.
  - Bisect on β ∈ [−12, 12] to |E − E*|/E* ≤ 10⁻³. If a draw does not bracket E*, redraw.
- **G-EN:** all 5 achieved errors ≤ 0.2, reported.
- G-use's β ∈ {0.5, 1, 2} leans toward high-variance directions and misses low targets. It was about 10× too high there (D1), and 14–19× too high for the toy's low-energy subspaces (0.24–0.32 off for its moderate one). The exact construction reaches ≤ 10⁻⁴ in every case (toy CHECK).

**(b) Clean decodability** of v_t from z = R_onᵀS_t (R*ᵀS_t beside it): linear probe; kNN-15; ‖z‖ alone and z/‖z‖ alone (§4 M-e).
- Necessary (D2) but not diagnostic (D4c), so reported only.
- **Validity rule for any probe after an ablation:** predictions constant, or accuracy ≤ majority + 0.02. Not chance ± 0.02: a probe with a bias predicts its bias class, and 2 G-use models fell outside chance ± 0.02 that way.

**(c) Necessity**, computed for R_on (D4b: R*'s own necessity can be illusion-driven), as the excess over the mean of the 10 random controls.
- **Resample (primary):** S_t ← S_t + RRᵀ(S_o − S_t), with S_o another B item's state.
- **Mean (secondary; the IOI convention, 2211.00593).**
- **Zero:** a descriptive bound only. It is off-distribution (Zhang & Nanda 2309.16042), and note 204-2's zeroed latent broke item-free tokens (0.963 → 0.499).
- Necessity is a qualifier (D3 / U3: redundancy), not required for USED.

**(d) Consistency.**
- **Across items:** IIA_cc ≥ 0.5 in every n-stratum with ≥ 50 cc pairs; otherwise PARTIAL.
- **Across passes (descriptive):**
  - R*(t = 2) applied at the t = 3 point, and to v_2 at S_3 (transfer IIA);
  - principal overlap ‖R_2ᵀR_3‖²_F / k.
  - This separates a single register from per-pass slots. Both count as "used", but they imply different targets (§5.3).
- **Unmatched-donor IIA.**
- D4 holds pair by pair, so consistency is a robustness check, not an illusion test.

**(e) Capacity.**
- **Shuffled targets** (fit to a permuted donor's y′, scored against the permuted targets): ≤ 0.20.
- **Random-init memory** (das167's random_model; §167-1): descriptive here, at k 16 / 64, t = 2 only. The 7 failed models are the trained negatives.

### 3.5 Decision rule (per model × point)

- **Best k** = the highest held-out IIA_cc over {4, 16, 64}, ties to the smaller k. Write IIA* for it, and ctrl for the maximum IIA_cc over the 10 random subspaces.
- **Validity:** G-IDENT, G-WHOLE, G-CF, G-DISJ and G-EN must pass, and the same-value donor patch with R* and with R_on must lower own accuracy by ≤ 0.05 (G-DISRUPT, as G-use's SWAP-DISRUPTIVE). Otherwise VOID.
- **Consumption rule (pre-stated):** if the whole-state y_d share at n ∈ {t+1, t+2} exceeds 0.30 (the state has already used the late leaf, so v_t may be consumed), that model's label at that point is computed on n ≥ t+3 only, and this is flagged.

| label | condition |
|---|---|
| **UR-FOUND** (USED-REPRESENTATION-FOUND) | IIA* ≥ 0.70; IIA* − max(ctrl, b0) ≥ 0.50; shuffled ≤ 0.20; G-SUPP passes and IIA_on ≥ IIA* − 0.15; the per-n floor holds. Suffix **-REDUNDANT** if R_on's resample-necessity excess is < 0.10. |
| **ILLUSION-SUSPECT** | the first three conditions hold, and IIA_on < IIA* − 0.30 |
| **USED-UNVERIFIED** | the first three conditions hold, and G-SUPP fails |
| **NONE-FOUND** | IIA*_cc < 0.30 at every k, and G-FIT passes |
| **FIT-FAILURE** | as NONE-FOUND, but G-FIT fails (uninformative, §167-3) |
| **PARTIAL** | anything else |

- **G-FIT:** train IIA is non-decreasing in k within 0.05, AND train IIA(64) ≥ max_k train IIA − 0.05. This is §167-3 plus its closing caveat.
  - k = 64 is warm-started from k = 16's R padded with PCA directions, so a converged nested fit cannot start below k = 16.
- The whole-state reference IIA_D (k = D) is reported beside every label, with its y_d and y shares.

---

## 4. Magnitude / onion variant (Csordás 2408.10920)

**Setup.**
- Additive head h: A_t = A_{t−1} + Σ_j φ(k_j)v_jᵀ over the pass's unpadded tokens, with φ = elu + 1 > 0, and o(q) = φ(q)ᵀA.
- The read is r = g·o (none), r = g·o/(φ(q)ᵀz) (den), or r = g·γ_h ⊙ o/RMS(o) (rms).

**M1 (PROVED; CHECK magcheck.json).**
- rms: r(q; cA) = r(q; A) for c > 0, up to the 10⁻⁶ regulariser (relative change ≤ 6 × 10⁻⁸).
- den: r(q; cA, cz) = r(q; A, z) (≤ 3 × 10⁻⁹).
- none: r scales by c.
- So **no function of an additive head's overall magnitude is readable at one read in the 8 rms / den models.** Note that rms keeps the *relative* magnitudes of components inside a head, since they are its direction. Onion-like layering *within* a head's direction is not excluded by M1.

**M2 (PROVED for the first pass; DERIVED after).**
- After A_t ← cA_t, pass t+1's additive reads are unchanged (rms; den with z scaled too), so its writes W_{t+1} are unchanged.
- Pass t+2 reads cA_t + W_{t+1}, so the weight of old content against the new write changes.
- In the open loop (A_t ≈ t ā), c acts as an **age shift**: cA_t ≈ (ct)ā.
- CHECK: the second read changes by 9–17% at c = 0.5 / 2.
- So in rms models magnitude is a clock in the mixing, not a value register.

**M3 (DERIVED(A_eq, A_ind): capacity of a norm code).**
- In a none-mode head, let v_s set an equal-step norm increment m(v_s) ∈ {m_0 + jδ} at pass s, so ‖o_t‖ ≈ Σ_{s≤t} m(v_s).
- A_ind: the reader does not recompute the earlier values, which is the converged regime.
- Then the MAP accuracy of v_t from the norm is 21/121 = **0.17** at t = 2 (and at t = 3), and **0.15** at t = 4, against chance 0.09 (exact enumeration).
- A usable norm code would need unequal level spacings whose required precision grows with t (Sidon-type or onion), or the earlier terms cancelled through another channel.

**M4 (PROVED; CHECK error ≤ 10⁻¹⁵).**
- At a shared unit key k, S′k = α(1 − β)Sk + βv.
- At β = 1 the old content is replaced, so there is no layering. At β < 1 it is kept with weight α(1 − β), so a geometric onion is possible only where β is bounded away from 1.
- So the delta-head onion question is the empirical β distribution.

**Tests (eval-only; M-b is the use test, the rest are descriptive).**
- **M-a NORM-BAND DECODE:** v_t from the 12 per-head log read norms at q_eq, and from the 12 per-head log state norms (linear and kNN-15; step_recover's 4,000 / 1,000 protocol; chance and majority baselines). In rms models this is presence only: by M1 the model cannot read it at the next pass.
- **M-b NORM SWAP (the magnitude interchange):** S_h ← (‖S_h^d‖_F / ‖S_h‖_F)·S_h (direction kept; den scales z with A). Head sets: each single head, all additive, all delta. IIA_cc to y′ with matched donors, plus the same-value control.
  - There are 12 scalars and no fit, so there is no capacity for an illusion.
- **M-c AGE RESCALE:** S_add ← c·S_add, with c ∈ {(t−1)/t, (t+1)/t}, at S_t. Report the answer-change rate, and the share of errors equal to F with one reduction fewer (a lag).
- **M-d α / β statistics** per delta head at the think tokens of passes 1..K: the mean, and the share with β < 0.8 (M4).
- **M-e CODE SHAPE inside R_on:** decode v_t from ‖z‖ alone vs z/‖z‖ alone.
- **Not done:** nonlinear-alignment DAS, which fits random models (2507.08802 via §167-1).

---

## 5. Separator reading and implications

### 5.1 Pre-stated reading over the 16 final models

- **SEP(t)** = min over the 9 converged models of IIA*_cc(t) − max over the 7 failed models of IIA*_cc(t).
- **SEPARATES-USED(t)** iff SEP(t) ≥ 0.20, AND ≥ 7/9 converged UR-FOUND, AND ≤ 1/7 failed UR-FOUND.
- t = 2 is primary and t = 3 secondary. SEP on IIA_D and the D-site arm (§7) are reported beside it.

**Predictions (CONJECTURE, central P).**

| # | prediction | P | basis |
|---|---|---|---|
| P1 | failed IIA*_cc(2) ≤ 0.20 in ≥ 6/7 | 0.85 | D2 (hybrid ≤ 0.12), D3 |
| P2 | converged IIA_D(2) ≥ 0.5 in ≥ 8/9 | 0.8 | D2 (≥ 0.53 in 9/9) |
| P3 | converged UR-FOUND(2) ≥ 7/9 | 0.45 | S5 3/3 at k ≤ 64, but MULT's fits failed (D7) |
| P4 | converged ILLUSION-SUSPECT ≤ 1/9 | 0.75 | — |
| P5 | the UR subspace's q_eq read displacement has ≤ 5% of its energy in V_probe | 0.8 | D1 |
| P6 | at t = 3, ≥ 60% of R_on's mass is on the delta heads (UR-FOUND converged models) | 0.55 | SEEDSTAB2 D4 (E1): converged models compute in delta heads |
| P7 | SEPARATES-USED(2) | 0.4 | ≈ P3 × P1 |
| P8 | M-b norm-swap IIA_cc ≥ 0.3 in any model | 0.1 | M1, M3 |

### 5.2 Outcomes and what they mean

- **O1 SEPARATES-USED.** A compact, on-support, used v_t code is a property of the converged class. Its k*, locus and register-vs-slot shape go into §5.3.
- **O2 DISTRIBUTED.** Converged IIA_D ≥ 0.7, but UR-FOUND ≤ 4/9 with converged fits. The code is used but not compact at k ≤ 64, so a subspace target is ill-posed. Only a coordinate-free whole-state interchange is aligned with the good class.
- **O3 ILLUSION.** ≥ 3/9 ILLUSION-SUSPECT. Linear subspaces are not admissible targets here.
- **O4 NO-SEPARATION.** Failed UR-FOUND ≥ 2/7. The failed models carry a used v_2, so the failure lies later (v_3+, consumption, or decoding). Move the separator to t = 3 / 4 before any method.
- **O5 SITE-SWAP** (D-site, descriptive). Failed models' decoder-residual IIA for v_2 ≥ 0.5 at n ∈ {3, 4} where their memory-site IIA ≤ 0.2, and the converged models show the reverse. This would give SEEDSTAB2's "the reader computes" (inferred from error patterns, D3 there) a causal reading.
- **Honest limit.** The failed side is mostly predicted by contraction (P1). The new information is the converged side. DAS-USED measures whether an *interventional* target is aligned with the good class; it is not a new separator in itself.

### 5.3 What this means for a training method (directions only; nothing here is registered)

**T1 (PROVED): a presence loss does not protect use.**
- L_pres(θ) = CE(h(RᵀS_t(θ)), v_t) depends on θ only through S_t. It is identical for two models with the same S_t, whether or not G_x reads R.
- So the minimisers of task + L_pres include "R is an unused register, the task is solved elsewhere". Starting from a *used* R does not prevent drift into that set.
- This is SVR's failure, not removed by choosing a better direction.

**T2 (PROVED): an interchange loss does.**
- L_II(θ) = E CE(G_{θ,x}(Π_R(S_t(x), S_t(x_d))), y′) → 0 only if G reads v_t from R (D1).
- An unused register, or a contractive model (G insensitive to S_t), pays at least the model's own clean −log p(y′), which is large for an accurate model.
- So L_II has a value gap against both failure classes (SEEDSTAB2's S1), and its target has content (S3: y′ depends on the donor's value).
- Its alignment with the good class (S2) is exactly converged IIA: what DAS-USED measures.

**T3 (PROVED): the gauge.**
- The model is unchanged under S_h → V_h S_h O_hᵀ per delta head (V_h ∈ GL(64), compensated in W_o and the v-rows; O_h ∈ O(64) on the q and k rows, which preserves unit norms). The erase, the write and the read all commute with this.
- So R* from one run defines nothing in another run without a fit.
- A target must therefore be **coordinate-free** (the matched whole-state interchange, i.e. SERIAL's hybrid as a loss) or a **designated slot** fixed by the architecture (interchange intervention training, Geiger et al., "Inducing causal structure for interpretable neural networks", arXiv 2112.00826; ID to be verified by Literature).
- DAS-USED's k* and locus size the slot.

**New Goodhart risks (DERIVED unless stated).**
- **GH1 one-shot precompute (route without computing; PROVED).** A single-point interchange constrains G, not how S_t was produced. A model that computes v_t from the text in one pass and writes it into R meets it.
  - Guard: a **chained** interchange (swap v_{t−1} at S_{t−1}, and require R's pass-t content to follow op(v_{t−1}^d, ·)), plus §203's K′ = 1 / pass-1 guard.
- **GH2 schedule lock-in.** y′ assumes v_t is live at S_t, so the loss imposes that schedule. That fights a natural 2-levels-per-pass schedule (SEEDSTAB2 D12). Guard: pick (t, node) from the liveness profile.
- **GH3 over-following.** G learns to trust the state over the text everywhere, which hurts robustness to state noise (C6). Guard: same-value donors must keep y, plus noise evals.
- **GH4 off-manifold training.** Spliced states divert capacity. Guard: clean-loss pairing, as in §203's bitwise pairing.
- **GH5 content pinning (PROVED for matched donors).** Carrying the donor's late leaf gives y_d ≠ y′, so the loss pins R's information content to v_t, not to "everything".

---

## 6. Cost and runtime; toy

**Dtype.**
- The trunk runs under bf16 autocast on the 3070 / 4060 (capability ≥ 8). The memory branch runs in fp32 (autocast off, deltamem.py 92, 162, 183). R and the patch are fp32.
- bf16 is bitwise only at a fixed batch size, so every eval-type quantity (base answers, IIA, controls, necessity) uses **bs 250**. Fits use batch 64 (gradients only).
- **One card class for all 16 models**, card recorded per model. The 2070 (fp32 path) is excluded.

**Fit budget** (das167's caps, tightened).
- ≤ 3,000 steps (cap 5,000), batch 64, lr 3e-3 (1e-3 only on a G-FIT failure). Validation IIA every 200 steps on 500 pairs, patience 3.
- 4,000 fit pairs, 500 validation, and the B set (≥ 200 cc pairs) for IIA.
- Inits: das167's PCA of S_d − S_r, and the class-mean directions of S_t by v_t (top ≤ 10, padded with PCA). Best by validation.

**Fits per model.**
- t = 2: k ∈ {4, 16, 64} with PCA init, plus k = 16 with class-mean init, plus the on-support refit, plus shuffled at k = 16: **6 fits**.
- t = 3: k* (from t = 2) and 64, plus on-support, plus shuffled: **4 fits**.
- Descriptive head-type fits (delta-only, additive-only at k*): +2, Stage B only.

**Runtime (estimate; replaced by a measured `--time_steps 50`).**
- About 0.4–0.8 s per fit step: passes t+1..K with per-pass checkpointing (InjectN.think ckpt), backward through 6–10 passes.
- About 15–25 min per fit with early stop, so about 2.5 GPU-h per model.
- Controls, necessity, magnitude and D-site are forward-only: minutes.
- **Stage A (pilot):** rms s1, nopass s1 (converged), rms s3, nofade s1 (failed), t = 2 only. About 6 GPU-h. It is a go / no-go on fit convergence and G-SUPP.
- **Stage B:** the other 12 models, plus t = 3 for all. About 35 GPU-h, or about 18 h wall on two cards. Queue it on the local backlog; it is not cloud work.

**Memory.**
- States: (4,500 + ≤ 3,000) pairs × 2 × 49,152 fp32 ≈ 2.9 GB, plus 6,000 support states (1.2 GB), in CPU RAM. Support basis: 49,152 × ≤ 6,000 fp32 ≈ 1.2 GB on the GPU.
- **Memory-check the largest shape first:** window s1 (K = 12), t = 2, so 10 passes of backward, on the longest n = 8 rows at batch 64 with pass checkpointing. Estimate ≤ 3 GB, about 13 GB without checkpointing. Use fit_micro 2 if the peak is above 6.5 GiB.

**Toy CHECK (`theory/dasused/toy_illusion.py/.json`; CPU, fp64, 3 threads).**
- State in R^2048: A (2-d), C (2-d), B (2-d, dormant, variance 10⁻¹⁰, read with gain 30), N (18-d, variance 1), F (2,022-d, variance 0.005).
- Value v ∈ Z_7 in a circle code, then 3 carry passes, then y = v + w with the late operand w taken from the text.
- **USED:** read A, with C a decodable but unused copy. **UNUSED:** the answer is recomputed from the text; C is decodable and unused. **USED-LOW:** C is read with gain 10, so a genuine code with about 3 × 10⁻⁴ of the variance.
- DAS: k ∈ {1, 2, 4}, three inits (random, PCA of differences, class means), 1,500 Adam steps each; IIA on 4,000 held-out pairs. Runtime 20 min.

| quantity (best k = 2) | USED | UNUSED (illusion) | USED-LOW |
|---|---|---|---|
| IIA | 1.00 | 0.99 | 1.00 |
| max IIA, 5 Haar / 5 exact-energy | 0.009 / 0.0003 | 0.007 / 0.12 | 0.014 / 0.12 |
| exact-energy rel. error (G-use-style β-family, best 5) | ≤ 8 × 10⁻⁵ (0.24–0.32) | ≤ 7 × 10⁻⁵ (15.8–18.9) | ≤ 1 × 10⁻⁴ (14.4–17.2) |
| shuffled targets (chance 0.143) | 0.145 | 0.148 | 0.140 |
| clean decode of v from Rᵀs, linear / kNN | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| variance share | 0.016 | 3.0 × 10⁻⁴ | 3.2 × 10⁻⁴ |
| necessity of R*, resample / mean / zero | 0.87 / 1.00 / 1.00 | **0.87 / 0.98 / 0.98** | 0.87 / 0.99 / 0.99 |
| necessity, Haar mean (resample) | 0.015 | 0.009 | 0.019 |
| ω (off-support displacement share) | 0.001 | 0.17 | 0.09 |
| IIA_proj / **IIA_on** | 0.10 / **1.00** | 0.00 / **0.00** | 1.00 / **1.00** |
| necessity of R_on (resample) | 0.87 | 0.00 | 0.87 |
| R* mass on A / C / B / F | .39 / .39 / .002 / .22 | 0 / .82 / .17 / .01 | .09 / .89 / .01 / .01 |
| **label (§3.5 rule)** | **UR-FOUND** | **ILLUSION-SUSPECT** | **UR-FOUND** |
| label without the on-support refit | UR-FOUND | UR-FOUND (wrong) | UR-FOUND |
| low-variance (Makelov-style) flag | no | yes | yes (wrong) |

- **Reading (CHECK; first-order D4 / D4b / D4c / D5 confirmed in a nonlinear toy).**
  - Every standard control passes for the illusion: random subspaces, shuffled targets, decodability, and necessity in all three forms. Without the refit the rule would read UR-FOUND.
  - The variance-share flag misfires on the genuine low-energy code.
  - ω stays under 0.2 for the illusion.
  - Only the on-support refit separates all three.
  - Low-energy exact-energy controls sit near the dormant route and read 0.07–0.12, still under the 0.2 bar. Haar controls are ≤ 0.014.

---

## 7. Engineering contract (for Experiments) and gates

**Script:** `scratch/dasused.py <run> [device] [--bs 250] [--points 2,3] [--stage A|B] [--fit_micro m] [--time_steps N] [--eval_only]`.
- Standalone, and **no edits to boxes/*.py**. It loads with halt173.load_model and sets DM.FAST_ADD from args, as svr_guse does, with CHUNK_DELTA 0.
- The patch is applied in the script's own pass loop (no monkeypatch): S_t → Π_R → `m.write` for passes t+1..K → `m.dec_logits`. This is equivalent to `think(override=(t, ·))`.
- It writes `<run>/eval_dasused.json`, **dumped after each point and after each fit** (a crash late in the run must not lose the earlier fields).
- It saves R* only, as fp16 `.npy` at k* per point (≤ 6.3 MB). No .pt and no support basis are kept.

**Fields (eval_dasused.json).**
- **Top level:**
  - run, class (from D7's lists), add_norm, K, card, dtype, bs_eval, bs_fit, fit_micro, ckpt_passes, seeds;
  - sets {A: n_items / n_fit / n_val; B: n_pairs, n_cc, by_n};
  - gates {IDENT, WHOLE, CF, DISJ, SUPP, EN, FIT, DISRUPT, MEM, GRAD, SMOKE};
  - base {acc_by_n, b0_by_n}.
- **points.{2,3}:**
  - whole_state {ycf, ydonor, yown, same_value_disruption, by_n};
  - liveness {τ: {ycf, ydonor, yown}} (t = 2 only, τ = 1..4);
  - k.{4,16,64} {iia, iia_cc, iia_cc_by_n, iia_train, val_best, steps, init, lr};
  - best_k, iia_star, k_min_qualifying;
  - controls {haar_iia_cc[5], energy_iia_cc[5], energy_rel_err[5], E_star, E_total, var_share, shuffled_iia, random_model (desc)};
  - support {eps, m, heldout_capture, omega, iia_proj, iia_on, iia_on_by_n, eps_profile};
  - necessity {resample: {R_on, R_star, random_mean, excess}, mean: {...}, zero_desc: {...}};
  - decode {linear_z, knn_z, linear_norm_z, linear_dir_z, majority, chance};
  - consistency {transfer, principal_overlap, unmatched_iia_cc, per_n_floor_ok};
  - locus {mass_delta, mass_add, mass_z, delta_only_iia_cc (desc), add_only_iia_cc (desc)};
  - vprobe_overlap (desc);
  - label.
- **magnitude:** {norm_decode, norm_swap {head_set: iia_cc}, age_rescale, alpha_beta_stats, code_shape}.
- **dsite (desc):** {block_8, block_10} — R ∈ St(768, k), k ∈ {4, 16}, on the decoder's " =" residual entering that block, with the donor's residual at the same block; IIA_cc to y′ for v_2; Haar control.
- sec, def.

**Gates (all must pass before any label is read).**
- **G-MEM:** mem_check at the largest shape (§6) before any fit. Report the peak in GiB.
- **G-SMOKE:**
  - the plain path (no patch) runs one batch end to end on this model;
  - **The toy is the label rule's fixture.** The script's own `label()`, applied to `theory/dasused/toy_illusion.json`'s fields (with G-SUPP and the per-n floor set to pass), must return USED → UR-FOUND, UNUSED → ILLUSION-SUSPECT, USED-LOW → UR-FOUND. (The toy's necessity excess is taken against its Haar mean.)
- **G-IDENT:**
  - the k = 0 patch path reproduces the run's own `inject_answers`-equivalent predictions **bitwise** at bs 250 on every B recipient, on the same card;
  - an empty-R patch equals no patch bitwise.
- **G-WHOLE:** the R = I patch equals passing S_d whole, bitwise.
- **G-CF:** answer_cf(x, t, v_t(x)) == y on all items. The counterfactual equals an independent evaluator: render the tree with node t replaced by the number v′ and call eval_tokens. y, y′ and y_d are distinct on every kept pair.
- **G-DISJ:** no recipient or donor text is shared between A and B. B recipients ⊂ TE; A recipients ∉ the exclusion set.
- **G-GRAD** (the gradient-flow rule): at the first fit step R.grad has a nonzero norm, and every model parameter has requires_grad False and grad None.
- **G-EN:** every exact-energy control has |E_R − E*|/E* ≤ 0.2.
- **G-SUPP:** held-out capture ≥ 0.999 at the primary ε.
- **G-FIT** (§3.5) gates negative readings only. **G-DISRUPT** (§3.5): the same-value patch costs ≤ 0.05 of own accuracy.
- **Probe-after-ablation validity:** predictions constant, or accuracy ≤ majority + 0.02.
- **Pre-registration:** the rule (§3.5, §5.1) and the label() code are committed before any eval_dasused.json exists. Stage A outputs are read only for G-FIT / G-SUPP / G-MEM go / no-go before Stage B. Any change after that is a dated amendment.

---

## 8. Claim-status table

| claim | status |
|---|---|
| S0: S_t is a complete mediator; state edits persist through the delta rule like the model's own content | PROVED (code F1–F3) |
| Decoder and think queries coincide at prefix positions; queries are memory-free | PROVED; CHECK (F-Q, max \|Δ\| = 0) |
| G-use's edit was visible to later think reads (0.38–0.82 of q̂_eq inside the think-query span) | CHECK (F-Q) |
| D1: after a patch, R-coordinates = donor's, complement = recipient's | PROVED |
| D2: IIA ≤ recipient-conditioned decodability of v_t from z | PROVED |
| D3: under redundancy, subspace IIA reads dominance | DERIVED (linear evidence) |
| D4: dormant-pathway illusion | PROVED (constructive, first order); toy CHECK |
| D4b: resample / mean / zero necessity are fooled by it | PROVED (same construction); toy CHECK |
| D4c: clean decodability cannot separate it | PROVED; toy CHECK |
| D4d: a low variance share cannot separate it from a genuine low-energy code | toy CHECK |
| D5: on-support refit excludes the dormant mechanism | DERIVED(A_supp, A_lin); toy CHECK |
| Prop E: exact-energy random subspaces reach any non-extremal target energy | PROVED (IVT + Ky Fan); toy CHECK ≤ 10⁻⁴ |
| M1: rms / den additive reads are scale-invariant per head | PROVED; CHECK |
| M2: a rescale is invisible to the next read and acts as an age shift after | PROVED (first pass); DERIVED (open loop); CHECK |
| M3: equal-step norm code gives ≤ 0.17 MAP accuracy of the latest value | DERIVED(A_eq, A_ind); exact enumeration |
| M4: β = 1 forbids onion layering at a shared delta key | PROVED; CHECK |
| T1: a presence loss does not protect use | PROVED |
| T2: an interchange loss is minimised only by a used R; value gap vs unused and contractive | PROVED (the gap's size is model-dependent) |
| T3: GL(64) × O(64) gauge per delta head; R does not transfer across runs | PROVED |
| GH1: single-point interchange is met by one-shot precompute | PROVED (constructive) |
| GH2–GH4 | DERIVED / CONJECTURE |
| P1–P8 | CONJECTURE |

**Prior art.**
- DAS 2303.02536 / causal abstraction 2301.04709; Boundless DAS 2305.08809; the illusion 2311.17030 and its reply 2401.12631; onion codes 2408.10920; patching practice 2309.16042; IOI mean ablation 2211.00593; random-model DAS 2507.08802 (all via Literature's 2b6bfd25 / §167, verified there).
- IIT 2112.00826 is new here and **unverified**.
- To our knowledge (Literature's survey), no DAS study exists on an identical-input looped or delta-rule memory. §167 is the in-house precedent.

---

## 9. Risks

1. **Fit failure at large k on a 49k-d state** (§167 MULT: k = 64 collapsed).
   - Mitigations: the nested warm start, two inits, G-FIT.
   - The residual risk is converged FIT-FAILURE, which leaves O1 / O2 undecided. **This is the main risk.**
2. **Schedule ambiguity.** v_t may already be consumed at S_t (2 levels per pass) or not yet computed. A good model could then read low IIA at (v_t, S_t).
   - Mitigations: the liveness profile, IIA by n, the whole-state y_d share, and the consumption rule of §3.5 (n ≥ t+3 when y_d > 0.30 at small n).
3. **Support estimation** from 6,000 states in 49,152-d.
   - If G-SUPP fails, the illusion test is unreadable (USED-UNVERIFIED). A genuine code below ε would be misread as ILLUSION-SUSPECT. That error is conservative for a USED claim, and the ε profile shows it.
4. **The separator is largely pre-determined on the failed side** (contraction), so SEPARATES-USED mostly tests the converged side. Worded so (§5.2).
5. **Spliced states are off-manifold even on-support.** G-DISRUPT (same-value patch costs ≤ 0.05) bounds this. Above that the point is VOID.
6. **bf16 and card class:** a fixed bs of 250, one card class, and the identity gate against the run's own path on that card. Do not compare to eval.json fields produced on 3090s.
7. **Cost creep:** about 40 GPU-h locally. The pilot gates Stage B. Keep the 3 – 4 CPU-thread limit for the CPU-side SVDs, to protect GPU queues.
8. **Interpretation:** UR-FOUND shows the model's own on-support route *for this counterfactual family*, up to first order (A_lin). It does not show uniqueness, and it does not by itself justify a training target (T1–T3, GH1).
