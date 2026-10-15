# DAS-USED: find the causally used representation of the intermediate values (draft rev 2, 2026-10-14)

> **Theory review (2026-10-14 20:18 EDT):** the whole-state SERIAL separator was checked against raw eval.json s170.serial["2"]: hybrid 0.53-0.96 in the 9 converged models vs 0.07-0.12 in the 7 failed (failed hybrid = donor = chance). Proof-tagged claims were taken as tagged and not re-derived. IIT 2112.00826 is not verified (Literature). Not registered; a $0 4-model pilot comes first.

## Changelog

**rev 2 (2026-10-14, after Audit's review, relayed by Theory).**
1. **Wording.** The whole-state SERIAL swap is causal **within converged models** only. It is no longer called a class separator: the s170 pool is n = 8, where the failed models solve only 0.19–0.49 clean, and the swap disrupts them (recipient share 0.12–0.31). The class contrast waits for capability conditioning (D2, §0, §5.2).
2. **Capability conditioning (new §2.5).**
   - Every fit and every IIA / SERIAL readout uses **ccc pairs**: the model solves the recipient, the donor and the hybrid item clean.
   - Per-n clean accuracy and the conditioned pair count are printed beside every IIA.
   - A stratum with fewer than **N_min = 100** ccc pairs (worst-case SE ≤ 0.05, Prop K0) is **N/A**, never "absent".
3. **SEPARATES-USED re-derived (§5.1, Prop K2).** Converged vs failed IIA is compared on **identical common pair sets** (≥ 200 pairs, bootstrap SE ≤ 0.05), so a failed model can neither pass nor fail through low clean accuracy. New outcome NOT-READABLE.
4. **Conditioned whole-state SERIAL, with per-item rows (pilot P0).** It has two parts: the s170 protocol re-run, and an n 3–8 extension.
5. **Pilot fixed (§6):** rms s1 / rms s3 and window s1 / hold s1, all on the RTX 3070. Window s1's memory check comes first.
6. **§7 contract final.**
7. **IIT 2112.00826 is verified** (Literature, docs/LIT_IIT_TRAINING_2026-10-14.md, 253ec5bb). The method follow-up is docs/THEORY_IITSERIAL_DRAFT_2026-10-14.md.

> **Status: draft, not registered.** Opus math subagent for Theory; Theory reviews. Eval-only, $0, on the 16 final models of D7. All data are POST-HOC.
> Read-only use of the repo. CPU checks (3 threads, Idle priority, no GPU, no .pt written) are in `theory/dasused/`: `qoverlap.py/.json` (F-Q), `magcheck.py/.json` (M1, M2, M4), `toy_illusion.py/.json/.log` (§3 controls, §6 toy).
> Coordinator constraints folded in (2026-10-14): G-use is NOT-USED 9/9 (§203 result); energy-matched controls must reach the target energy and gate on |E_R − E*|/E* ≤ 0.2; probe-after-ablation validity = "predictions constant or accuracy ≤ majority + 0.02"; necessity by mean or resample ablation, zero only as a descriptive bound.

## 0. Summary

1. **Why.** D7's probe subspace is decodable but unused (G-use NOT-USED 9/9). The whole-state SERIAL swap is causal **within** converged models (hybrid 0.53–0.96 at t = 2). The class contrast (failed 0.07–0.12) is **pending capability conditioning**: that pool is n = 8, where failed models solve 0.19–0.49 clean, and the swap disrupts them. The question is where the used code sits, how compact it is, and whether it is the model's own route.
2. **Site: the whole carried state S_t** (49,152-d; 53,248 for den). It is the only cut between pass t and the answer, and an edit to it persists through later passes exactly as the model's own content does (Prop S0, PROVED). DAS: R ∈ St(D, k), k ∈ {4, 16, 64}. Donors are structure-matched: node t's subtree and one late leaf are resampled, so y, y′ (= the counterfactual) and y_d are distinct.
3. **What IIA can and cannot show.** IIA implies decodability of v_t from R's coordinates, conditioned on the recipient (D2, PROVED). The dormant-pathway illusion fools clean-run decodability AND mean / resample necessity (D4b, D4c, PROVED by construction). A low variance share does not tell a genuine low-energy code from the illusion (toy CHECK).
4. **The illusion test that works: the on-support refit** (D5, DERIVED). Refit R inside the span of natural state variation; an illusion's IIA collapses, a used code's does not (toy CHECK).
5. **Controls:** 5 Haar and 5 exact-energy random subspaces (Prop E, PROVED: any energy between the Ky Fan bounds is reachable; gate ≤ 0.2), shuffled targets, necessity (resample primary, mean secondary) on the on-support subspace, per-n floors, transfer across passes.
6. **Magnitude:** rms / den additive reads are invariant to the head's scale (M1, PROVED + CHECK), so magnitude there acts only as an age in later mixing (M2). An equal-step norm code in an un-normalised integrator gives at most 0.17 MAP accuracy for the latest value (M3, DERIVED). β = 1 forbids onion layering at a shared delta key (M4, PROVED). Tests: norm-band decode, norm swap, age rescale, β statistics.
7. **Separator (rev 2, capability-conditioned):** only ccc pairs (recipient, donor and hybrid item all solved clean); strata with < 100 such pairs are N/A; converged and failed models are compared on identical pair sets (≥ 200, SE ≤ 0.05). SEPARATES-USED needs Δ ≥ 0.20 on every readable pair, ≥ 5/7 failed readable, ≥ 7/9 converged UR-FOUND and ≤ 1 readable failed UR-FOUND.
8. **Method implication:** a presence loss does not protect use, even on a used direction (T1, PROVED). An interchange loss does (T2, PROVED), and it also penalises contraction. But coordinates do not transfer across runs (T3, gauge, PROVED), so a target must be coordinate-free or a designated slot. New Goodhart risk: one-shot precompute into the slot.
9. **Pilot ($0, approved):** rms s1 / rms s3 (same recipe, seeds differ) and window s1 / hold s1 (s182 family, K = 12, none-mode), all on the 3070. P0 (forward-only gates, clean accuracies, the conditioned SERIAL with per-item rows) runs before any fit. The full 16-model run is about 40 GPU-h, locally.

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
| D2 | **Whole-state SERIAL hybrid at t = 1 / 2** (n = 8 pool, 476 items, unconditioned). Converged 0.38–0.90 / 0.53–0.96 (0.68–0.99 at t = 6): **causal within converged models.** Failed 0.06–0.08 / 0.07–0.12 (null 0.04–0.09), but these models solve only 0.19–0.49 of n = 8 clean, and the swap disrupts them (recipient share at t = 2 is 0.12–0.31, against 0.02–0.16 for converged). So the **class contrast is pending capability conditioning** (§2.5). | eval.json `s170.serial` (16 runs) |
| D3 | path_indep acc_from_donor: converged 0.10–0.14 (path-dependent), failed 0.22–0.64. delta_cos: converged 0.44–0.63, failed 0.76–0.98. | eval.json `s182.path_indep` |
| D4 | **Clean accuracy at K′ = K (8; 12 for window / rwwin / hold), n = 1..8.** Converged ≥ 0.99 at every n except window s1 and rmsnopass s2 (0.95 / 0.89 / 0.84 / 0.76 and 0.95 / 0.88 / 0.85 / 0.77 at n 5–8). Failed: rms s3 1.0 / 1.0 / .96 / .84 / .65 / .49 / .36 / .23; csr s1 1 / 1 / .85 / .65 / .47 / .31 / .23 / .21; den s1 1 / 1 / 1 / .96 / .86 / .71 / .58 / .42; rwwin s1 1 / 1 / 1 / .93 / .81 / .67 / .50 / .36; rmsgf s3 .94 / 1 / .85 / .67 / .50 / .33 / .27 / .19; hold s1 1 / 1 / 1 / .93 / .81 / .65 / .52 / .35; nofade s1 1 / 1 / 1 / .97 / .89 / .74 / .64 / .49. **Failed models are lower but not at chance at K′ = K.** At K′ 0 / 1 they are near chance from n = 3. | eval.json `acc_by_n`, `acc_by_n_kprime` |
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
- **Fit set A:** `train_items` with rng 20301 and the run's exclusion set; n ∈ [t+1, 8]. Draw recipients until there are **4,000 ccc fit pairs and 500 ccc validation pairs**, with at most 20,000 raw recipients.
- **IIA set B:** every eligible TE item at n ∈ [t+1, 8] (≤ 1,000 per n), one matched donor each (rng 20302).
- A and B are disjoint by construction (training draws exclude TE). G-DISJ also checks donor texts.
- **Primary statistic: IIA_ccc**, pooled over the model's eligible strata (§2.5) and reported per n. Beside it: IIA over all pairs, b0 = P(clean answer = y′), and per-n clean accuracy with the ccc counts.
- **Points:** t = 2 (primary) and t = 3.
- **Liveness (descriptive):** the whole-state matched swap for v_2 at τ = 1..4, as y′ / y_d / y shares by n. A model that does 2 levels per pass (SEEDSTAB2 D12) may already have consumed v_2 at S_2.

**2.5 Capability conditioning (rev 2).**
- **Hybrid item.** x_hyb := x with node t's subtree taken from x_d, and x's late leaf kept.
  - **F(x_hyb) = y′ (PROVED).** In x_hyb, node t evaluates to v_t^d, and every other leaf is x's.
  - It is a valid, renderable expression of the same length; G-HYB checks it with eval_tokens.
- **ccc pair:** the model answers x, x_d and x_hyb correctly on its clean path (K′ = K, bs 250). Fits, IIA, controls, necessity and the conditioned SERIAL all use ccc pairs only.
- **Prop K1 (DERIVED(A_serial): why all three).**
  - Suppose the model carries v_t in S_t and uses it.
  - After the patch, it must carry out the remaining operations of x (above node t, with x's late leaf) on the carried value v_t^d. That is the same remaining computation it performs from pass t+1 on x_hyb clean: same text above node t, same value at node t.
  - So on ccc pairs, a low IIA cannot be blamed on missing capability for the remaining operations.
  - A model that cannot solve x_hyb is excluded, not counted as absent.
  - A_serial (the remaining computation depends on node t only through its value) is the serial program itself. If it fails, IIA_ccc measures exactly that failure.
- **Prop K0 (PROVED: N_min).** An IIA estimated from N pairs has SE = √(p(1−p)/N) ≤ 0.5/√N. **SE ≤ 0.05 needs N ≥ 100.**
  - So a (model, point, n) stratum with N_ccc < 100 is **N/A**.
  - A model whose eligible strata pool to fewer than 100 is N/A at that point.
  - For a contrast of two models on identical pairs, the paired difference has Var ≤ p₁₀ + p₀₁ ≤ 1. So the rule is N ≥ 200 **and** a paired-bootstrap SE ≤ 0.05, else N/A.
- **Expected ccc yield (≈ acc³ under independence; P0 measures it).**
  - rms s3: about 0.88 / 0.59 / 0.27 / 0.12 at n 3 / 4 / 5 / 6, so t = 2 is readable at n 3–6 with 1,000 items per n.
  - hold s1: about 1.0 / 0.80 / 0.53 / 0.27.
  - t = 3 (n ≥ 4) may be N/A for the weakest failed models (csr s1, rmsgf s3: n = 5 accuracy 0.47 / 0.50, so ccc ≈ 0.10–0.13).
- **Conditioned whole-state SERIAL (pilot P0, per-item rows).**
  - (i) The s170 protocol re-run: SPOOL n = 8, rng 17300; serial_items rng 17200; t ∈ {1, 2, 4, 6}. Rows record, per item, the recipient, donor and hybrid clean answers and the swapped answer. Failed models are expected to be N/A here (n = 8 ccc ≈ acc³ ≈ 0.01–0.12, against 476 items).
  - (ii) An n 3–8 extension with the same donor construction (h's operands and the root's bare operand; rng 20371) on TE items, ≤ 1,000 per n, swaps at t ∈ {1, 2}. Eligibility is as in IIT-SERIAL §2.1 (root has a bare leaf, h ≠ root, height(root) ≥ 2t + 1), with y_hyb ∉ {y, y_d}, so that it doubles as that method's G-ALIGN.
  - Reading per (model, n, t): hybrid / own / donor / other shares on ccc rows, with N.
  - **This is the conditioned class contrast for SERIAL.** It is also the alignment measurement for IIT-SERIAL (docs/THEORY_IITSERIAL_DRAFT_2026-10-14.md).

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
- **Across items:** IIA_ccc ≥ 0.5 in every eligible n-stratum (N_ccc ≥ 100); otherwise PARTIAL.
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

- **All quantities are on ccc pairs in the model's eligible strata** (N_ccc ≥ 100, Prop K0). No eligible stratum: the label is **N/A**.
- **Best k** = the highest held-out IIA_ccc over {4, 16, 64}, ties to the smaller k. Write IIA* for it, and ctrl for the maximum IIA_ccc over the 10 random subspaces.
- **Validity:** G-IDENT, G-WHOLE, G-CF, G-DISJ and G-EN must pass, and the same-value donor patch with R* and with R_on must lower own accuracy by ≤ 0.05 (G-DISRUPT, as G-use's SWAP-DISRUPTIVE). Otherwise VOID.
- **Consumption rule (pre-stated):** if the whole-state y_d share at n ∈ {t+1, t+2} exceeds 0.30 (the state has already used the late leaf, so v_t may be consumed), that model's label at that point is computed on n ≥ t+3 only, and this is flagged.

| label | condition |
|---|---|
| **UR-FOUND** (USED-REPRESENTATION-FOUND) | IIA* ≥ 0.70; IIA* − max(ctrl, b0) ≥ 0.50; shuffled ≤ 0.20; G-SUPP passes and IIA_on ≥ IIA* − 0.15; the per-n floor holds on every eligible stratum. Suffix **-REDUNDANT** if R_on's resample-necessity excess is < 0.10. |
| **ILLUSION-SUSPECT** | the first three conditions hold, and IIA_on < IIA* − 0.30 |
| **USED-UNVERIFIED** | the first three conditions hold, and G-SUPP fails |
| **NONE-FOUND** | IIA* < 0.30 at every k, and G-FIT passes |
| **FIT-FAILURE** | as NONE-FOUND, but G-FIT fails (uninformative, §167-3) |
| **N/A** | no eligible stratum (fewer than 100 ccc pairs) |
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
- **M-b NORM SWAP (the magnitude interchange):** S_h ← (‖S_h^d‖_F / ‖S_h‖_F)·S_h (direction kept; den scales z with A). Head sets: each single head, all additive, all delta. IIA_ccc to y′ with matched donors, plus the same-value control.
  - There are 12 scalars and no fit, so there is no capacity for an illusion.
- **M-c AGE RESCALE:** S_add ← c·S_add, with c ∈ {(t−1)/t, (t+1)/t}, at S_t. Report the answer-change rate, and the share of errors equal to F with one reduction fewer (a lag).
- **M-d α / β statistics** per delta head at the think tokens of passes 1..K: the mean, and the share with β < 0.8 (M4).
- **M-e CODE SHAPE inside R_on:** decode v_t from ‖z‖ alone vs z/‖z‖ alone.
- **Not done:** nonlinear-alignment DAS, which fits random models (2507.08802 via §167-1).

---

## 5. Separator reading and implications

### 5.1 Pre-stated reading over the 16 final models

- **Common pair sets.** For a converged model c and a failed model f: P(c, f) = the B pairs that are ccc for both (identical pairs). Converged models solve almost everything, so P(c, f) ≈ ccc(f).
- **Δ(c, f)** = IIA_c − IIA_f on P(c, f). Each model uses its own R_on at its own best k, fitted on its own A-ccc pairs. The SE is a paired bootstrap (2,000 resamples).
  - The contrast is **N/A** if |P(c, f)| < 200 or SE > 0.05 (Prop K0).
- **f is READABLE** at t if Δ(c, f) is non-N/A for all 9 converged c.
- **SEPARATES-USED(t)** iff all of:
  - ≥ 5/7 failed models are READABLE;
  - Δ(c, f) ≥ 0.20 for every readable (c, f);
  - ≥ 7/9 converged models read UR-FOUND;
  - ≤ 1 readable failed model reads UR-FOUND.
- **NOT-READABLE(t)** if fewer than 5/7 failed models are READABLE. Then there is no class reading at that t. N/A models are listed, never counted as absent.
- **Prop K2 (PROVED, given K0 and the ccc definition): no trivial pass or fail.**
  - (i) A failed model cannot *fail trivially* (look absent because it cannot compute). Every pair in P(c, f) is one it solves clean on all three items.
  - (ii) It cannot *pass trivially*. A reading from a handful of easy items is N/A (N ≥ 200, SE ≤ 0.05).
  - (iii) Both models are scored on identical pairs, so stratum mix and item difficulty cancel in Δ. The converged comparator is not given easier items.
- t = 2 is primary and t = 3 secondary. Reported beside it: the same contrast on IIA_D (whole state), the conditioned SERIAL contrast (§2.5), and the D-site arm (§7).

**Predictions (CONJECTURE, central P).**

| # | prediction | P | basis |
|---|---|---|---|
| P1 | Δ(c, f) ≥ 0.20 on every readable pair at t = 2, and ≥ 5/7 failed READABLE | 0.7 | D3 (contraction); D2 is not capability-conditioned |
| P1b | conditioned SERIAL (P0 ii, n 3–4, t = 1): failed hybrid_ccc ≤ 0.2 with own-answer share ≥ 0.6; converged hybrid_ccc ≥ 0.6 | 0.65 | D3: failed models recompute from text (acc_from_donor 0.22–0.64) |
| P2 | converged IIA_D(2) ≥ 0.5 in ≥ 8/9 | 0.8 | D2 (≥ 0.53 in 9/9) |
| P3 | converged UR-FOUND(2) ≥ 7/9 | 0.45 | S5 3/3 at k ≤ 64, but MULT's fits failed (D7) |
| P4 | converged ILLUSION-SUSPECT ≤ 1/9 | 0.75 | — |
| P5 | the UR subspace's q_eq read displacement has ≤ 5% of its energy in V_probe | 0.8 | D1 |
| P6 | at t = 3, ≥ 60% of R_on's mass is on the delta heads (UR-FOUND converged models) | 0.55 | SEEDSTAB2 D4 (E1): converged models compute in delta heads |
| P7 | SEPARATES-USED(2) | 0.35 | ≈ P3 × P1 |
| P8 | M-b norm-swap IIA_ccc ≥ 0.3 in any model | 0.1 | M1, M3 |

### 5.2 Outcomes and what they mean

- **O1 SEPARATES-USED.** A compact, on-support, used v_t code is a property of the converged class. Its k*, locus and register-vs-slot shape go into §5.3.
- **O2 DISTRIBUTED.** Converged IIA_D ≥ 0.7, but UR-FOUND ≤ 4/9 with converged fits. The code is used but not compact at k ≤ 64, so a subspace target is ill-posed. Only a coordinate-free whole-state interchange is aligned with the good class.
- **O3 ILLUSION.** ≥ 3/9 ILLUSION-SUSPECT. Linear subspaces are not admissible targets here.
- **O4 NO-SEPARATION.** Failed UR-FOUND ≥ 2/7. The failed models carry a used v_2, so the failure lies later (v_3+, consumption, or decoding). Move the separator to t = 3 / 4 before any method.
- **O5 SITE-SWAP** (D-site, descriptive). Failed models' decoder-residual IIA for v_2 ≥ 0.5 at n ∈ {3, 4} where their memory-site IIA ≤ 0.2, and the converged models show the reverse. This would give SEEDSTAB2's "the reader computes" (inferred from error patterns, D3 there) a causal reading.
- **Honest limit.**
  - The failed side is predicted by contraction (D3), but only capability-conditioned readings count: unconditioned SERIAL (D2) is not evidence of a class difference.
  - The new information is mainly on the converged side.
  - DAS-USED measures whether an *interventional* target is aligned with the good class; it is not a new separator in itself.

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
- A target must therefore be **coordinate-free** (the matched whole-state interchange, i.e. SERIAL's hybrid as a loss) or a **designated slot** fixed by the architecture (interchange intervention training, Geiger et al., "Inducing causal structure for interpretable neural networks", arXiv 2112.00826, verified by Literature 253ec5bb; developed in docs/THEORY_IITSERIAL_DRAFT_2026-10-14.md).
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
- **Each model is pinned to one card.** All 16 run on one card class (RTX 3070) unless §7's cross-card check passes. Card, dtype and bs are recorded in every json block. The 2070 (fp32 path) is excluded.

**Fit budget** (das167's caps, tightened).
- ≤ 3,000 steps (cap 5,000), batch 64, lr 3e-3 (1e-3 only on a G-FIT failure). Validation IIA every 200 steps on 500 pairs, patience 3.
- 4,000 ccc fit pairs, 500 ccc validation pairs, and B's ccc pairs (eligible strata) for IIA.
- Inits: das167's PCA of S_d − S_r, and the class-mean directions of S_t by v_t (top ≤ 10, padded with PCA). Best by validation.

**Fits per model.**
- t = 2: k ∈ {4, 16, 64} with PCA init, plus k = 16 with class-mean init, plus the on-support refit, plus shuffled at k = 16: **6 fits**.
- t = 3: k* (from t = 2) and 64, plus on-support, plus shuffled: **4 fits**.
- Descriptive head-type fits (delta-only, additive-only at k*): +2, Stage B only.

**Runtime (estimate; replaced by a measured `--time_steps 50`).**
- About 0.4–0.8 s per fit step: passes t+1..K with per-pass checkpointing (InjectN.think ckpt), backward through 6–10 passes.
- About 15–25 min per fit with early stop, so about 2.5 GPU-h per model.
- Controls, necessity, magnitude and D-site are forward-only: minutes.
- **Pilot (Audit-approved, $0): 4 models, all on the RTX 3070.**
  - **s185_rms_s1 (converged) / s185_rms_s3 (failed).** Same recipe; only the seed differs, and s3 never entered. This is the pure seed contrast that SVR / IIT-SERIAL target. rms mode: M1 applies.
  - **s182_window_s1 (converged) / s182_hold_s1 (failed).** Same s182 family (k_window 1,4, K = 12; hold adds aux_hold). None-mode additive heads, where a magnitude code is readable (M1 does not apply). Window s1 is also the largest shape and one of the two weakest converged models, a conservative comparator.
  - **Clean accuracy of the failed pair at n 3 / 4 / 5:** rms s3 0.96 / 0.84 / 0.65, hold s1 1.0 / 0.93 / 0.81. This gives ≥ 100 ccc pairs at n 3–5 or 3–6 (§2.5).
  - **Not chosen:** den s1 and nofade s1 are more accurate but have no same-family converged partner; csr s1 and rmsgf s3 are s185-family failures with fewer ccc pairs (n = 4: 0.65 / 0.67).
  - **Order:**
    1. **P1** on window s1 first: G-MEM at its largest shape, plus `--time_steps 50`.
    2. **P0** on all 4 (forward-only; about 20–30 min each).
    3. **P2** at t = 2 (6 fits each; about 2.5–3 h each; window s1 / hold s1 run 10 passes of backward).
  - About 12 GPU-h in all, about 12 h wall on the one card.
  - **Go / no-go for Stage B** (gates only): G-FIT converged on ≥ 3/4; G-SUPP on ≥ 3/4; measured ≤ 3.5 GPU-h per model. Pilot labels are reported as pilot results and cannot change the rule.
- **Stage B:** the other 12 models, then t = 3 for all 16. About 35 GPU-h on the local backlog. It is not cloud work.

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

## 7. Engineering contract (final, rev 2) and gates

**Script:** `scratch/dasused.py`, a new file. **No edits to boxes/*.py.**
- It imports boxes read-only (halt173.load_model, nested170, deltamem). It sets DM.FAST_ADD = args.fast_add, DM.CHUNK_DELTA = 0 and READ_CLAMP None.
- The patch runs in the script's own pass loop, with no monkeypatch: S_t → Π_R → `m.write` for passes t+1..K → `m.dec_logits`. This is equivalent to `think(override=(t, ·))`.
- The loop is not under any `@torch.no_grad` decorator in the fit path (gradient-flow rule).

`python scratch/dasused.py <run> --device cuda:K --stage {P0,P1,P2,B} [--points 2[,3]] [--bs 250] [--fit_bs 64] [--fit_micro 1] [--time_steps 0] [--resume]`

**Stages.**
- **P0 (forward-only):**
  - G-IDENT, G-CF, G-HYB, G-DISJ, G-WHOLE, the plain-path smoke;
  - clean answers (bs 250) for every B recipient, donor and hybrid item → per-n clean accuracy of x / x_d / x_hyb, ccc counts and eligible strata (**G-CAP**);
  - the whole-state matched swap at t ∈ {1, 2, 3} plus liveness;
  - the conditioned SERIAL (§2.5 i and ii).
- **P1:** G-MEM at this model's largest shape (t = 2, longest n = 8 rows, fit batch 64, pass checkpointing), plus `--time_steps 50`. Peak GiB and s/step go into the json. Window s1 is done before any other P2.
- **P2 (per point):**
  - the support set and G-SUPP;
  - fits on A-ccc pairs: k 4 / 16 / 64 (PCA init), k 16 (class-mean init), the on-support refit, and shuffled targets at k 16;
  - IIA on B-ccc pairs; controls; necessity; decode; consistency; locus; magnitude M-a..M-e; D-site.
- **B:** P0–P2 for the other 12 models, then `--points 3` for all 16.
- **--resume** skips blocks marked `"done": true` in eval_dasused.json.

**Seeds** (numpy unless noted):

| use | seed |
|---|---|
| A recipients | 20301 |
| B donors | 20302 |
| A donors | 20303 |
| support items | 20311 |
| Haar controls | 20321 |
| exact-energy controls | 20322 |
| shuffled targets | 20331 |
| resample-ablation partners | 20341 |
| magnitude tests | 20351 |
| D-site | 20361 |
| extended-SERIAL donors | 20371 |
| fit init and batches | torch 20381 + k, numpy 20391 + k |
| s170 SERIAL re-run | its own 17300 / 17200 |

**Files (new, in the run dir).**
- `eval_dasused.json`: dumped after every block and every fit.
- `eval_dasused_rows.npz`: per-pair rows for every arm. Each row has the pair id, point, n, text hashes of x / x_d / x_hyb, y / y′ / y_d, the three clean answers, the ccc flag, and the answer under each patch (whole state, each k, each control, each necessity mode, M-b). It also holds the conditioned-SERIAL rows.
- `dasused_R_t{t}_on.npy`: R_on at k*, fp16 (≤ 6.3 MB).
- Nothing else: no .pt, no support basis, no state dumps.

**Every json block records:**
- card (`torch.cuda.get_device_name`) and dtype ("bf16 autocast trunk, fp32 memory branch");
- bs_eval, bs_fit, fit_micro, ckpt_passes;
- torch version, driver, the git hash of scratch/dasused.py at launch;
- wall seconds.

**Fields.**
- **Top level:**
  - run, class, add_norm, K, the card / dtype / bs block;
  - gates {IDENT, WHOLE, CF, HYB, DISJ, CAP, SUPP, EN, FIT, DISRUPT, MEM, GRAD, SMOKE};
  - base {clean_acc_by_n {x, x_d, x_hyb}, n_ccc_by_n, eligible_strata, b0_by_n};
  - serial_cond {s170: {t: {hybrid, own, donor, other, n_ccc}}, ext: {t: {n: {...}}}}.
- **points.{2,3}:**
  - whole_state {ycf, ydonor, yown, other, n_ccc, by_n};
  - liveness;
  - k.{4,16,64} {iia_ccc, n_ccc, iia_ccc_by_n, n_ccc_by_n, iia_all, iia_train, val_best, steps, init, lr};
  - best_k, iia_star, k_min_qualifying;
  - controls {haar_iia_ccc[5], energy_iia_ccc[5], energy_rel_err[5], E_star, E_total, var_share, shuffled_iia_ccc, random_model (desc)};
  - support {eps, m, heldout_capture, omega, iia_proj, iia_on, iia_on_by_n, eps_profile};
  - necessity {resample: {R_on, R_star, random_mean, excess}, mean: {...}, zero_desc: {...}};
  - decode (on R_on) {linear_z, knn_z, linear_norm_z, linear_dir_z, majority, chance};
  - consistency {transfer, principal_overlap, unmatched_iia_ccc, per_n_floor_ok};
  - locus {mass_delta, mass_add, mass_z, delta_only_iia_ccc (desc), add_only_iia_ccc (desc)};
  - vprobe_overlap (desc);
  - label.
- **magnitude** {norm_decode, norm_swap {head_set: {iia_ccc, n_ccc}}, age_rescale, alpha_beta_stats, code_shape}.
- **dsite (desc)** {block_8, block_10}: R ∈ St(768, k), k ∈ {4, 16}, on the decoder's " =" residual entering that block, with the donor's residual at the same block. IIA_ccc to y′ for v_2, with a Haar control.
- **Every IIA is stored with its n_ccc and per-n counts.** The scorer prints them beside it.

**Gates (all must pass before any label is read; any validity failure stops that model after a dump).**
- **G-MEM:** peak ≤ 6.5 GiB at fit_micro ≤ 2, else stop.
- **G-SMOKE:**
  - the plain path runs one batch end to end;
  - the script's own `label()`, applied to `theory/dasused/toy_illusion.json`'s fields (G-SUPP and the per-n floor set to pass; the toy's necessity excess is against its Haar mean), returns USED → UR-FOUND, UNUSED → ILLUSION-SUSPECT, USED-LOW → UR-FOUND.
- **G-IDENT:** the k = 0 patch path reproduces the run's own `inject_answers`-equivalent predictions bitwise at bs 250 on every B recipient, on the pinned card. An empty-R patch equals no patch bitwise.
- **G-WHOLE:** the R = I patch equals passing S_d whole, bitwise.
- **G-CF / G-HYB:** answer_cf(x, t, v_t(x)) == y on all items. The counterfactual equals eval_tokens on the rendered x_hyb. y, y′ and y_d are distinct on every kept pair.
- **G-DISJ:** no recipient or donor text is shared between A and B. B recipients ⊂ TE; A recipients ∉ the exclusion set.
- **G-CAP:** ccc counts and N/A strata are written before any IIA is computed.
- **G-GRAD:** at the first fit step R.grad has a nonzero norm, and every model parameter has requires_grad False and grad None.
- **G-EN:** every exact-energy control has |E_R − E*|/E* ≤ 0.2.
- **G-SUPP:** held-out capture ≥ 0.999 at ε = 10⁻⁶.
- **G-FIT:** gates negative readings only.
- **G-DISRUPT:** the same-value patch costs ≤ 0.05 of own accuracy (ccc pairs).
- **Probe-after-ablation validity:** predictions constant, or accuracy ≤ majority + 0.02.

**Pinning and cross-card.**
- Each model runs entirely on one card. The pilot runs all four on the RTX 3070.
- Stage B may add a second card only after rms s1 is re-run there with |ΔIIA_ccc| ≤ 0.02 at every k and identical labels. Otherwise Stage B stays on the 3070.

**Pre-registration.** The rule (§2.5, §3.5, §5.1), the scorer and label() are committed before any eval_dasused.json exists. Pilot outputs are read for the Stage-B go / no-go gates; labels are reported as pilot results. Any later change is a dated amendment.

---

## 8. Claim-status table

| claim | status |
|---|---|
| S0: S_t is a complete mediator; state edits persist through the delta rule like the model's own content | PROVED (code F1–F3) |
| Decoder and think queries coincide at prefix positions; queries are memory-free | PROVED; CHECK (F-Q, max \|Δ\| = 0) |
| G-use's edit was visible to later think reads (0.38–0.82 of q̂_eq inside the think-query span) | CHECK (F-Q) |
| F(x_hyb) = y′ for the hybrid item | PROVED |
| K0: N_min = 100 per stratum for SE ≤ 0.05; contrasts on ≥ 200 identical pairs with bootstrap SE ≤ 0.05 | PROVED (binomial bound) |
| K1: on ccc pairs, low IIA is not a capability shortfall | DERIVED(A_serial) |
| K2: no trivial pass or fail of a failed model in SEPARATES-USED | PROVED (given K0 and the ccc definition) |
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
- IIT 2112.00826 is verified (Literature 253ec5bb). Its zero-loss causal-abstraction guarantee is a theorem.
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
4. **Capability conditioning can make the class contrast unreadable.**
   - At t = 3, the weakest failed models (csr s1, rmsgf s3) may have < 100 ccc pairs per stratum, giving NOT-READABLE rather than a separator.
   - Conditioning on solved items also selects the failed models' easiest items. Identical common pair sets keep the contrast fair, but the claim is restricted to that item population and is worded so.
5. **Spliced states are off-manifold even on-support.** G-DISRUPT (same-value patch costs ≤ 0.05) bounds this. Above that the point is VOID.
6. **bf16 and card class:** a fixed bs of 250, one card class, and the identity gate against the run's own path on that card. Do not compare to eval.json fields produced on 3090s.
7. **Cost creep:** about 40 GPU-h locally. The pilot gates Stage B. Keep the 3 – 4 CPU-thread limit for the CPU-side SVDs, to protect GPU queues.
8. **Interpretation:** UR-FOUND shows the model's own on-support route *for this counterfactual family*, up to first order (A_lin). It does not show uniqueness, and it does not by itself justify a training target (T1–T3, GH1).
