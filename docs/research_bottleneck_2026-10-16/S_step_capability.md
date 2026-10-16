# S: per-pass step capability — diagnosis, COMMUTE + OPX (+ SELF-SNAP), and a minimal nested test (research subagent for Theory, 2026-10-16 ~03:45 EDT)

**Status.** Research draft for Theory. It is not a registration, and nothing here has been spent.

The new §1 numbers come from CPU-only analyses (2 threads) of existing eval files.
- Scripts: `research2/step/` in this scratchpad. Run `python feat.py feats.json` first, then `hhfit.py`, `hazard.py`, `hazard2.py`, `logit.py` or `byop.py` with `feats.json <models…>`.
- Inputs read: only `boxes/nested170.py` (via importlib) and `boxes/runs_fsq/*/eval*.json`.
- LIT2's prior-art leads (research2/LIT2.md) are folded in.

**Tags:** [PROVED], [PROVED UNDER X], [DERIVED], [MEASURED], [ASSUMED], [CONJECTURE].

---

## 0. Summary

1. **The failed-seed step is recompute-bounded, not depth-decayed.**
   - In the four failed INJECT seeds, in-range handle accuracy is fit by a *pass-free* function of the handle's subtree height (R² 0.993–1.000). It sits at ≈ 1 for height ≤ 2 and collapses at height 4–5 (0.22–0.73).
   - The "decay with t" is composition: tall handles appear only from pass 3.
   - §205's IIT did not lift this profile.
2. **Converged seeds compose** (a flat height profile). They fail only beyond training, keyed to pass index, remaining size and held-value load.
3. **Mechanism: COMMUTE + OPX,** two loss-only changes to VB.
   - **COMMUTE** trains the step to commute with decoding on the model's own versions.
   - **OPX** applies the same loss to counterfactual operand values written into the bank.
   - Together they remove the signal that rewards recompute and add one that penalises it at every height from step 0.
   - A second cell adds **SELF-SNAP**, after Altabaa 2510.14095: a straight-through canonical code in R's row space for stored versions.
4. **Plan.** $0 diagnostics first, then two nested cells at about $0.8–2.2 each.

---

## 1. Diagnosis

### 1.1 Evidence

**(E1) §199's "conditional per-step success" is not conditional** [MEASURED].
- s185_rms_s3's 0.97 → 0.72 (t 3–8) is the ratio of consecutive *marginal* handle accuracies (`s175.aux_acc_by_role.handle`: 0.9654/0.9989 = 0.966 … 0.3609/0.5019 = 0.719).
- Each t is a different population (items with n ≥ t: 1,000 … 133).

**(E2) Late steps are shorter in tokens, but taller** [MEASURED; enumeration on the §170 test set, n ≥ t, n ≤ 8, nested170's leftmost-ready order].

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| live symbols in E_{t−1} | 13.5 | 12.2 | 10.7 | 9.3 | 7.8 | 6.3 | 4.7 | 3.0 |
| mean handle subtree height hh | 1.00 | 1.65 | 2.05 | 2.44 | 2.79 | 3.27 | 3.99 | 5.01 |
| share with hh ≥ 3 | 0 | 0 | 0.35 | 0.54 | 0.60 | 0.70 | 0.86 | 1.00 |
| both operands computed | 0 | 0 | 0.17 | 0.27 | 0.35 | 0.44 | 0.58 | 0.73 |

**(E3) HH-FIT** [MEASURED].
- Model: acc_t = Σ_h P(hh = h | t)·q(h), with 5 free q's and no t term, fit to the 8 marginal handle accuracies (aux eval population, n ~ U{1..8}).

| model | q(1) | q(2) | q(3) | q(4) | q(5+) | R² |
|---|---|---|---|---|---|---|
| s185_rms_s3 | 0.999 | 0.995 | 0.937 | 0.422 | 0.324 | 0.997 |
| s205_iit_s3 (IIT) | 0.998 | 0.971 | 0.854 | 0.251 | 0.224 | 0.993 |
| s185_den_s1 | 0.999 | 0.997 | 0.976 | 0.727 | 0.359 | 0.999 |
| s188_rwwin_s1 | 0.995 | 0.993 | 0.951 | 0.730 | 0.240 | 1.000 |

- **Converged models (12):** every q ≥ 0.949 in 11. The exception is s182_window_s1, which never got below loss 0.1: q(5+) 0.735.
- **Linear-in-t alternative:** R² 0.83–0.94. It has fewer parameters, so E4 is the discriminating test.
- **Final answers:** failed-seed accuracy tracks E[q(root height)] to within about 0.1.

**(E4) First-error hazard** [MEASURED].
- Data: 187-3 per-item rows, n 9–12, 4,000 items, 13 models; teacher-forced aux handle; prefix-clean steps, t ≤ 8.
- Model: a joint logistic fit, with coefficients per SD.

| group | hh | t | live length | held values | bracket depth |
|---|---|---|---|---|---|
| failed (3) | **+0.71…+1.10** | +1.1…+1.6 | +0.3…+1.0 | +0.04…+0.18 | +0.4…+0.6 |
| converged (9) | **−0.03…+0.39** | +1.6…+4.4 | +0.8…+3.5 | +0.27…+0.53 | −0.05…+0.83 |
| s190_rmsnopass_s2 (under-converged) | +0.67 | +1.65 | +0.75 | +0.26 | +0.75 |

- **Failed, at t = 6:** hh 1 → hh 5 hazard is 0.076 → 0.533 (s185_rms_s3).
- **Converged:** hazard rises with held load instead. s190_rmsnopass_s1 at t = 8: 0.05 / 0.09 / 0.27 / 0.50 for held 0–3.
- **A residual pass-keyed component, everywhere:** s185_rms_s3's hh = 1 handles go 0.006 → 0.198 over t 1–8.

**(E5) Written values vanish after pass 2 in failed seeds** [MEASURED].
- Probe: `step_recover`, a linear probe of the read for step t's value.
- At t = 3: failed 0.13 / 0.24 / 0.23 / 0.19 (majority baseline 0.08–0.11). This includes s205_iit_s3, which reads 0.91 / 0.36 at t = 1 / 2.
- Converged: 0.62–0.77.

**(E6) The operator does not matter** [MEASURED]. Hazard by operator is within 0.05 in each of five models.

### 1.2 Reading

**Failed regime** [MEASURED pattern; mechanism CONJECTURE].
- Each visible step is recomputed from the re-shown input by the 12-layer aux reader, with a capacity of about 3 tree levels.
- Passes ≥ 3 write no usable value, so the step is *absent* beyond pass 2.
- §205 s3's MET was measured at τ1 / τ2 (τ3 had no items), the only passes where s3 writes. So "carrying is not the limit" holds only there.

**Converged regime** [MEASURED]. The step composes, but beyond training it is non-stationary: keyed to pass × size (COUNT-OOD, 187-6a) and to held load.

**Why recompute wins early** [DERIVED, §3.2].
- On shallow handles, recompute and composing tie; wrong-input terms reward only recompute.
- R4's first stage (n ≤ 3) has **0%** of steps at hh ≥ 4 and 11% at hh 3 [MEASURED]. The recompute route meets about 98% of it, so stage 1 hardly separates the circuits.
- The faster circuit gets a head start: §199's "hare" signature.

**In VB.**
- Linear R closes the reader-recompute route (REDTEAM F3). What remains is recompute *inside* pass t: 4 core layers plus V_0 at constant β⁰.
- VB's failed regime is therefore CARRY-NO-STEP with a q(hh) cliff.
- Two VB costs are still open:
  - VOID-gap neighbour lookup;
  - the diffuse gather (off-self 0.80–0.94, 211-9 / 211-12), which spends core depth reassembling E_{t−1}.

**Prior art (LIT2; IDs verified there).**
- **RELAY 2502.08482:** per-loop CoT-round targets on Feng-style arithmetic, ≤ 15 → 25 operators: 0.98 vs 0.77 unsupervised. This supports R2.
- **Altabaa 2510.14095:** mod-23 DAGs, N ≤ 32 → 128. Snap + corruption 100%, snap alone 94.8%, continuous latent supervision 54.8%. That is an S2 failure (§3.3) in a related setting.
- **CLRS 2209.11142 and DRM 2604.18839:** ground-truth teacher forcing and one-step denoising hurt OOD. This fits N3 (211-12).
- **CoTFormer 2609.19521 and Bohde 2403.04929:** full cross-iteration history hurts depth extrapolation (50.1% vs 99.4% for recent-1 at depth 20). VB's bank is this pattern, a C2 risk.

### 1.3 Hypotheses and $0 discriminators

| hypothesis | signature | test |
|---|---|---|
| H-REC: recompute-bounded | hazard ∝ hh at fixed t; re-rooted step fine | D1, D2(i, iii) |
| H-PASS: clock-keyed | hazard rises with t (or bank size) at fixed hh | D2(i, v) |
| H-HOLD: held-load interference | hazard ∝ held at fixed t | D2(i) |
| H-CAP: VB capacity (4 layers, VOID gaps) | low STEP-CAP in every hh class, rising with gap; re-root also low on long inputs | D2(i, iii) |
| H-GATHER: reassembly tax | age-1 operands worse; cross-slot gather mass > 0.3 | D2(iv) |
| H-OP: operator-specific | — | rejected (E6) |

**D1 (INJECT, local GPU, about 1 h).**
- Models: s185_rms_s3, s205_iit_s3, s185_rms_s1.
- 187-3 rows on **n 3..8** (300 per n), plus RE-ROOT rows: the model run fresh on x′ = cot_steps(root)[t−1], whose pass-1 handle is the step-t handle with literal operands.
- **Pre-stated:** the failed seeds reproduce HH-FIT in-trajectory (q(4) < 0.8), with re-root ≥ 0.98 in every original-hh class; s185_rms_s1 is ≥ 0.98 on both.
- *Falsifier of H-REC:* re-root accuracy also falls with hh.

**D2 (§211 finals, fp32, eval_bs 32; about 30–60 min each).**
- **(i)** STEP-CAP rows on exact-input events (R(V_{t−1}) decodes E_{t−1} exactly), with hh, held, length, max VOID gap and operand ages. Then the E4 regression.
- **(ii)** OPERAND-FOLLOW: the §2.2 edit at eval.
- **(iii)** RE-ROOT-VB: pass-1 STEP-CAP on render(E_{t−1}).
- **(iv)** Gather mass on (t−1, j), on (s < t−1, j) and on (·, i ≠ j).
- **(v)** (iii) after o ∈ {2, 4} copies of V_0.
- **Pre-stated:** a §211 cell reading CARRY-NO-STEP or NO-ENTRY shows q(1–2) ≥ 0.97, q(≥4) < 0.8 and OPERAND-FOLLOW < 0.5 [CONJECTURE, P 0.65].

---

## 2. Mechanisms

**Notation.**
- dec(V) = the per-slot argmax of R(V).
- skel(E) = E with every number class replaced by a generic NUM.
- At pass t the handle has operand slots l and r in E_{t−1}, value slot b in E_t, and operator op. OP is the 11×11×4 table.

### 2.1 COMMUTE (replaces R2's targets; R1 relabelled; no extra forward)

**Target for t ≥ 2** (t = 1 keeps E_1, since dec V_0 := E_0):
- Ẽ_t[b] = OP[op, dec V_{t−1}[l], dec V_{t−1}[r]];
- Ẽ_t[j] = dec V_{t−1}[j] for number slots carried unchanged from E_{t−1};
- Ẽ_t[j] = E_t[j] otherwise.

This applies when skel(dec V_{t−1}) = skel(E_{t−1}). Otherwise Ẽ_t = E_t (fallback, logged as `commute_fb`).

**Losses.**
- **L_rec^C** is R2's exact form (½ changed + ½ unchanged, values ×2, HOLD), with Ẽ in place of E.
- **R1-C:** at t ∈ [n_i, K_i] the answer target is dec V_t's value if skel(dec V_t) = skel(E_n), else y_i. The coda *reads* what the version holds.

**Parameters reached:** R2's and R1's (core 6–9, gather, bank rows, gain, R, the prelude via V_0's entries; the coda via R1-C).

**What changes, and when.**
- Only terms whose input version is wrong in a value slot: 27% / 52% at t = 4 / 8 for p = 0.9 (REDTEAM F3b).
- There, free-running R2 demands a repair only recompute can deliver; COMMUTE demands the correct reduction *of what is there*.

**Design question (a), the input distribution.** The only exposure-bias-free inputs are the model's own versions; COMMUTE relabels targets instead of correcting inputs. The alternatives each fail:
- **MCT (N3):** mixes corrected entries into a diffuse gather (211-12 (c)), and is ground-truth teacher forcing (CLRS, DRM).
- **Teacher-rendered versions:** a different dialect (REDTEAM F6).
- **§111's DAG:** kept ground-truth targets and read 1/5. VB already has §111's winning reader (R from step 0).

### 2.2 OPX: operand interchange (dense step-level interventions)

**Per item, with probability ρ_X = 0.5:**
1. Draw a handle pass t ~ U{2..n_i}.
2. Draw (a′, b′) ≠ (a, b) (b′ ≠ 0 for ÷), changing one or both operands.
3. For each operand slot k and each version s ∈ [s_k, t−1] (s_k = 1 for a leaf, else the pass that computed it): V′_s[k] = MCT-edit(V_s[k] → class a′ / b′, μ 1.0). This is VBankN.mct restricted to slot k (gate W1 exists).
4. V_0, other slots and the null-space components are unchanged.
5. Recompute bank K/V and gather keys for the edited versions only (detached), and run **pass t only**: V′_t = core_t(V′_{<t}).

**Loss.**

  L_X = 2·CE(R(V′_t)[b], OP[op, a′, b′]) + ½·mean_{j≠b} CE(R(V′_t)[j], E_t[j]),  λ_X = 1.

- **Parameters reached:** core 6–9 (including ln_1 / c_attn via the edited K/V), the PA gather, core bank rows, g, R. Not the prelude or the coda.
- **Cost:** +15–20% s/step [ASSUMED, launch-bound, 211-3].

**Why edit every version since s_k.** The bank then says "this slot has held a′ since creation", consistent for every pass before t. The only contradictions are V_0's literal and the operand's pre-s_k subtree, which are exactly the recompute sources. So a recompute circuit is wrong on every edited term at every height, including hh = 1.

### 2.3 SELF-SNAP (cell S2; LIT2 lead (i))

**Stored entry:**

  S_t = V_t + α·sg(Δ_t)

- **The edit Δ_t:** the row-space edit (LN coordinates, pinv of W_R, ≤ 3 iterations) that sets every real slot's R-logits to L*(dec V_t) = μ_s·(onehot − 1/20), with μ_s = 4.
- **Reads:** the bank, gather and carry read S_t; R2 / COMMUTE / R1 read V_t.
- **Gradient:** straight-through.
- **Schedule:** α ramps 0 → 1 over 4k–8k; ‖Δ‖/‖V‖ is clipped at 0.05 per slot, and the clip rate is logged.
- **OPX under SNAP** ("snap corruption", Altabaa B.4): the same sampling, with the operand slots' snap targets set to a′ / b′ in versions s_k..t−1. The target is COMMUTE's (*follow*).
- **No new parameters.**

**Full snap is not proposed** (S_t = emb(dec V_t)). It would make the inter-pass state a 20-class slot sequence, a discrete scratchpad, which undercuts the user's latent-over-text standard (NORTH_STAR C4). The row-space snap keeps 748 dimensions continuous.

**Follow vs correct.**
- Altabaa keeps true targets: the model learns to correct corruptions.
- In VB a reduced value's own operands are VOIDed, so correction needs the bank history or a deep recompute. The history evidence favours the Markov follow semantics.
- A "correct" variant (corrupt only the newest copy) is descriptive only.

### 2.4 Secondary mechanisms (only after the stated trigger)

- **OFFSET.**
  - With p 0.5, o ~ U{1..4} forced V_0 copies (HOLD targets) are appended before reduction 1. This decorrelates pass index, bank size and age from progress.
  - **Trigger:** COUNT-KEYED, PASS-KEYED, or DEPTH < 0.5 with STEP-COMPOSES.
  - STALL then becomes trained-for, and NO-CLOCK descriptive.
- **Bansal start** (2202.05826; LIT2 lead (ii)).
  - t₀ ~ U{0..n_i−1} passes run under no_grad (versions and bank K/V detached); the losses and gradient apply after t₀.
  - [DERIVED] Prefix writes get no gradient, so null-space cross-pass channels (S2 violations) are not reinforced from the prefix, and the step trains from its own state at a random depth.
  - **Trigger:** PASS-KEYED.
- **Capacity (b).**
  - **Trigger:** H-CAP.
  - First a fixed 2× core per pass, not learned halting (ACT / PonderNet collapse to one loop, RecurTrace 2609.03379).
  - If 2× helps, ELASTIC-RECODE: L = −log Σ_{monotone a, steps ∈ {0, 1}, a(K_i) = n_i} Π_t q_t(E_{a(t)}).
    - Holds come from the slack K_i − n_i ∈ {0..4}, so the loop cannot collapse below n_i reductions.
    - R1's window stays fixed, since posterior weighting would let the model dodge the loss [DERIVED].
- **History / gather.**
  - **Trigger:** H-GATHER or DILUTION.
  - Options: SLOT-LOCAL GATHER (U_t[j] = Σ_s α_{j,s} N(V_s[j])), or a descriptive bank = {V_0, V_{t−1}} arm. The second narrows the pass-global memory, so it is the user's call.
- **Rejected: operator routing (c).** The premise fails (E6), and learned ≈ frozen-random routing on modular addition (Smithline 2605.09403). Any routing cell needs a random-route control.

---

## 3. Mathematics

### 3.1 Targets

**C1 (COMMUTE is Markov)** [PROVED].
- On skeleton-matched states Ẽ_t = F(dec V_{t−1}): l, r, b and op follow from skel(dec V_{t−1}) through the leftmost-ready rule.
- So each term's minimiser is achievable from V_{t−1} alone, and V_0 carries no extra information (where dec V_{t−1}[l] ≠ x[l], V_0 contradicts the target).
- **Free-running counterpart** [PROVED]: when dec V_{t−1}[l] ≠ E_{t−1}[l], R2's target is a function of x and not of dec V_{t−1}. Its minimiser *requires* dependence on V_0 or on older versions.

**C2 (no first-order route to self-serving errors)** [DERIVED].
- Targets come from argmax under no_grad, so no gradient flows through them.
- "Skip to done" changes the skeleton, so the fallback fires and the skip is paid at its own pass.
- Consistently wrong chains need a wrong pass 1, which has the exact target.

**X1 (what L_X = 0 forces)** [PROVED].
- The value output at b is a non-constant function of the operand slots' R-row-space coordinates in versions ≥ 1.
- It is invariant to V_0's literal and to the pre-s_k subtree.
- Transfer to natural versions is not implied (EDIT-ONLY) [CONJECTURE].

**SN1 (canonical code)** [PROVED up to the LN tolerance, gated].
- At α = 1, every stored entry's R-logits equal L*(dec V_t). A reader of the bank through R's row space sees a pass-free, history-free code: S2 restricted to the row space.
- Snap-corrupted entries are identical to natural ones in that subspace.

**SN2 (edit size)** [PROVED]. ‖Δ_t[j]‖ ≤ σ_LN·‖L* − L‖ / σ_min(W_R).
- With R's small σ_min at init, early snapping would apply large off-manifold edits; hence the ramp and clip.
- On trained R, MCT-type edits measured ‖Δ‖/‖V‖ ≈ 1e-3 (211-3).
- The null space still drifts. How much of Altabaa's 100 vs 54.8 needs the full snap is unknown [CONJECTURE].

### 3.2 The race (why recompute is selected; how the tilt flips)

**Setup** [ASSUMED].
- C_rec reads V_0 and is right iff hh ≤ c₀ (probability π_s).
- C_cmp outputs op(dec operands), and π_c = P(the input operands decode correctly).
- Strengths grow at E[(1 − p)(1[right] − 1[wrong])].

| objective | drift(C_cmp) − drift(C_rec) [DERIVED in this model] | early sign |
|---|---|---|
| free-running R2 | ∝ 2(π_c − π_s) | **negative** (π_s ≈ 0.95 in stage 1, π_c low) |
| COMMUTE | ∝ 2(1 − π_s π_c) | **positive** (C_cmp's output is the target) |
| + OPX | adds ≈ 2ρ_X/K̄ | positive (C_rec is wrong on every edit) |

- The drift is the bias b of §199's selection SDE: P(computing basin) = Φ(μ_Z/σ_Z), with μ_Z linear in b. To move P from 0.6 to 0.95, a tilt needs a 4–7× change and noise levers 15–42× [DERIVED UNDER 199-A's one-dimensional assumption].
- **Prediction:** earlier, less seed-dependent entry [CONJECTURE].
- **Testable content:** the sign of S1's hh coefficient.

### 3.3 Composition, stationarity and coverage

**Product bound** [PROVED].
- P(final correct) ≥ Π_t (1 − h_t), where h_t is the prefix-clean hazard.
- Under COMMUTE, value errors are absorbing, so the bound is tight for them.

**Stationary transfer** [PROVED UNDER S1–S3].
- Assume:
  - (S1) the step depends on history only through V_{t−1};
  - (S2) given dec V_{t−1} = E, the law of the null-space part u = P_⊥ LN(V_{t−1}) is free of t and path (or the step ignores u);
  - (S3) the step reads E only through a bounded window π(E) around the handle.
- Then h_t = h(π(E_{t−1})) and acc(n) ≥ E Π_t (1 − h(π(E_{t−1}))) for **every** n. Training on n ≤ 8 visits every window type, so depth transfers.
- COMMUTE gives S1 at its optimum (C1). SNAP gives S2 in the row space (SN1); OPX pushes the value read there [CONJECTURE].
- Converged INJECT models violate S2 / S3 (E4: the t and length coefficients).

**Drift** [DERIVED, linearised].
- If u_t = A u_{t−1} + b_t, with ‖A‖ ≤ κ, ‖b‖ ≤ β, and the step is exact while ‖u‖ ≤ ρ:
  - κ < 1 and β/(1−κ) ≤ ρ ⇒ competence at every t;
  - κ > 1 ⇒ loss of competence at t* ≈ log(ρ/‖u_0‖)/log κ, i.e. a pass-keyed hazard at a fixed window (E4's hh = 1 rows).
- At init κ > 1 (variance ×14 by t = 8, REDTEAM F3c).
- κ is not targeted directly, since one-sided dynamics targets get gamed. It is read as STEP-STATIONARY.

**Coverage** [PROVED by counting; MEASURED shares].
- With idle offsets o ≤ O, n ≤ 8 training visits the (pass, operators-left) cell (t, r) iff r ≤ 8 and t ≤ 9 − r + O.
- O = 0 covers **0%** of n 9–12 test steps (all have t + r ≥ 10).
- O = 4 covers **76.2%**. The rest need r > 8, i.e. longer items.

---

## 4. Goodhart table

| route | status |
|---|---|
| G1 pass-1 precompute | Unchanged; bounded by L1 + CAP-aff on deep items; OPX acts only at t ≥ 2. PROBE(1)/(2) and ONESHOT. |
| G2 recompute from V_0 | Unrewarded (C1) and penalised at every height (X1). Natural-donor VALUE-SWAP own-share and RECOMPUTE-V0 measure it. |
| G3 coda solving | R1-C trains the coda to read dec V_t; solving earns nothing. PROBE uses true y. |
| G4 copying literals | Still required; OPX edits the copies, so the step must take literals from versions. |
| G6 clocks (bank size, age, VOID count) | Unaffected; both "until done" and count rules meet COMMUTE. STALL measures it. SNAP removes row-space clocks only; OFFSET if triggered. |
| G8 sabotage / timed release | No negative-likelihood term; targets are sg (C2). |
| **G-EDIT** (follow edits only) | Possible: the null space still says a. **EDIT-ONLY** = edit follow ≥ 0.9 with natural follow(ii) < 0.6. SNAP removes the row-space signature (SN1). Fallback: natural-donor OPX. |
| **G-SKIP / G-FALLBACK** | Closed by the skeleton fallback (C2). COMMUTE-INACTIVE if `commute_fb` > 0.2 over the last 8k. |
| **G-NULL** (information moves to the null space) | Possible. Descriptive null-space probes of pass index and values. |
| **G-ABSORB** (no repair) | A cost, not a route: DEEP needs per-step error ≤ 0.013 at n 8. |
| **G-SHADOW** (R reads a code the step ignores) | What OPX penalises (X1). |
| Trained-for readings | Under OPX, SERIAL-STEP tests edit → natural transfer, not emergence. PROBE-GAIN, SERIAL-ONSET, DEEP and NO-CLOCK stay untrained. |

---

## 5. Decisive readings, OC, censoring

**STEP-COMPOSES** (new, comparator-free).
- Criterion: value STEP-CAP ≥ 0.95 in every handle-height class {1, 2, 3, 4, ≥5}, on exact-input events (n 3..8, t ∈ [2, n]), with ≥ 50 events per class.
- Anchors (HH-FIT's marginal q, a conservative stand-in): 0 / 4 failed seeds pass, 11 / 12 converged do [MEASURED].

**SCR-C1** = VB-C1 (as registered) ∧ STEP-COMPOSES.

**Labels:**
- **RECOMPUTE-BOUNDED:** q(1–2) ≥ 0.97 and min(q(4), q(≥5)) < 0.8.
- **EDIT-ONLY.**
- **PASS-KEYED:** an hh = 1 hazard slope in t > 0.005 per pass.
- **ABSORBING-ERRORS:** STEP-COMPOSES ∧ ¬DEEP.
- **COMMUTE-INACTIVE.**
- **SNAP-CLIPPED:** clip rate > 0.1 over the last 8k.
- The §211 labels.

**Descriptive:** the E4 regression on the cell's own rows, OPERAND-FOLLOW, `commute_fb`, DEPTH, and the D2 arms.

**OC (rule a).**

| criterion | P(pass \| no effect) | P(pass \| effect) |
|---|---|---|
| 1 seed: STEP-COMPOSES | ≈ 0 (recompute regime, 0 / 4) | ≈ 0.9 (composing regime, 11 / 12) |
| 1 seed: SCR-C1 | ≈ 0.3 (§211's base) | ≈ 0.5 [CONJECTURE] |
| rate: STEP-COMPOSES ∧ DEEP, 3/3 (p₀ 0.45, p₁ 0.85) | **0.09** | **0.61** |
| rate: 2/2 | 0.20 | 0.72 |

At one seed the property reading is sharp and the effect reading weak. Realised $ replaces the projections in §6.

**Censoring (rule b).**
- A dead run or NO-ENTRY fails everything.
- An hh class with < 50 events counts as **failed** (never-reached = failure), so thinning cannot pass.
- VALUE-SWAP with < 100 clean pairs is UNREADABLE (not met).
- Event-only values are reported beside each reading.

---

## 6. Minimal first experiment (nested)

**Stage 0 ($0, local).**
1. D1, on one card between queue items.
2. D2 on each §211 final, as a vbank_eval part.
3. Gates:
   - the COMMUTE target unit test (injected value errors on 100k items must match a symbolic reduction; the fallback must fire on skeleton errors);
   - OPX W1 on l / r, other slots unchanged ≥ 0.999;
   - SNAP SN1 within 0.05 nats on a trained-R checkpoint, plus a straight-through check (∂L/∂V_t = ∂L/∂S_t);
   - per-term gradflow (L_X ≠ 0 into core / gather / R, = 0 into the prelude and coda);
   - the plain N2 path, constructed with 1 step and no new flags;
   - mem_check with the OPX branch;
   - G-TIME.
4. This §3 goes to Audit as the derivation note (math-backing rule).

**Stage 1** (RTX 3090; seed 1; 48k × bs 64; R4; §170 test set; one GPU class).

| cell | design | question | projected $ |
|---|---|---|---|
| **S1** | N2 + COMMUTE (L_rec^C, R1-C) + OPX (ρ_X 0.5) | Does the step compose at every height (SCR-C1)? | 48k × 0.28–0.48 s × 1.15–1.2 at $0.18–0.26/h: **$0.8–2.0** |
| **S2** | S1 + SELF-SNAP (ramp 4k–8k; snap corruption) | Does a canonical row-space code remove PASS-KEYED (DEPTH descriptive)? | S1 + about 10% (N3's MCT path +9% G-TIME): **$0.9–2.2** |
| S3 (optional) | N2 + COMMUTE only | Is removing the tilt enough? | $0.7–1.7 |

- **Family (S1 + S2):** $1.7–4.2. **Cap $5**; per-cell guard $2.5. S1 vs S2 at one seed is descriptive.
- **Base:** N2 (constraint-compliant).
- **Register only if** a §211 cell reads CARRY-NO-STEP, RECOMPUTE-BOUNDED or NO-ENTRY. If all §211 cells read STEP-COMPOSES, the bottleneck is absent in VB, and S1 / S2 would become a seed-robustness question needing its own OC.

**What follows.**
- **SCR-C1:** seeds 2–3 of the better cell ($2–4). Then + OFFSET ($1–2.5) for C2, with a pre-registered DEPTH-STATIONARY criterion (n 9–12 exact-input hazard at fixed hh, flat in t to ≤ 0.01 per pass). GSM8K is separate.
- **RECOMPUTE-BOUNDED:** the incentive account fails for VB. Next: the 2× core or SLOT-LOCAL GATHER, per D2.
- **EDIT-ONLY:** natural-donor OPX.
- **PASS-KEYED / COUNT-KEYED:** OFFSET, or the Bansal start.

---

## 7. Top risks and falsifiers

1. **The INJECT diagnosis may not carry to VB.** The limit may be capacity or the gather instead.
   - *Premise falsifier ($0):* D2 shows a flat-but-low profile in the failing §211 cells.
   - *Lever falsifier:* S1 reads RECOMPUTE-BOUNDED.
2. **G-EDIT.** *Falsifier of X1's transfer:* natural follow(ii) < 0.6 with edit follow ≥ 0.9 (S2's snap should shrink this).
3. **Absorbing errors cost DEEP** (per-step ≥ 0.987 is needed at n 8). If ABSORBING-ERRORS reads, the next step is a bank-repair variant (copy-slot targets true when an older version decodes them).
4. **Anneal and gather confound.** S1 / S2 inherit N2's anneal cost (211-11). Cross-position gather mixing may dilute edits, as in 211-12 (c); D2(iv) reads this. Moving the base to N1 is the user's call (the constraint).
5. **History hurts depth** (CoTFormer, Bohde). A cell can meet SCR-C1 and still fail DEPTH through its full-history bank; the {V_0, V_{t−1}} arm then needs the user.
6. **SNAP shifts the manifold** (SN2) or is not exercised (SNAP-CLIPPED). *Falsifier for SNAP:* S2 reads PASS-KEYED like S1, with null-space probes carrying the pass index.
7. **The race model is a caricature.** *Falsifier of the tilt account:* S1 enters, but its hh coefficient stays ≥ +0.6 (the failed-seed range).

**Overall falsifier** of "the step is the bottleneck and training signal fixes it": S1 reads RECOMPUTE-BOUNDED with OPERAND-FOLLOW ≥ 0.9. The step would then read its operands and still fail tall handles, so the limit is not where operands are read from.
