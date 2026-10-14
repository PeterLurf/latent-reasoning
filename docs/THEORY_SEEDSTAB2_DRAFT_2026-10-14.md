# SEEDSTAB2: a successor to CSR. SVR, a shallow pass-free value readout (draft rev 2, 2026-10-14)

> **Theory review (2026-10-14 18:30 EDT):** rev 1 reviewed. Per-head RMS normalisation let the readout compute mod-11 ops (perhead_norm.json: 1.000 vs 0.16-0.35), which broke Lemma L / CERT; rev 2 fixes the map and recalibrates the bars. Checked against raw files: D7 step_recover (14 of 16 runs, exact), pooled_t2p (conv 0.50-0.73 / fail 0.11-0.22), linmap_check numbers, Prop O's code premise (deltamem.py:278-280). **Not registered.** It is a new spend decision (about $8.5 central / $10.1 high for 3 seeds + C8k), so it goes to the user via Audit. G-pre (delta-only step_recover separates the 16 models at t = 2, 3) is a $0 gate to run first.

Opus math subagent for Theory. Read-only use of the repo. CPU checks (≤ 3 threads, no GPU, no .pt written) are in `theory/seedstab2/`:
`heights.py/.json`, `lp_sumdecomp.py/.json`, `direct_height.py/.json`, `h4_split.json`, `race_toy.py/.json`; rev 2: `linmap_check.py/.log/.json` (+ `linmap_merge.py`), `pooled_t2p.py/.json`.

## Changelog

**rev 2 (2026-10-14, after Theory's review).**
1. **MUST-FIX 1: the feature map is now linear in the state.**
   - Rev 1's per-head RMS normalisation multiplies superposed codes. Per head it gives (φ_h(a) + ψ_h(b))/‖φ_h(a) + ψ_h(b)‖, a cross-head a × b interaction. So the readout could perform a reduction, and the probe could defer two nested ops.
   - Theory's check (`theory/seedstab2/perhead_norm.json`) and mine (`linmap_check.json`) both read 1.000.
   - New map:
     - per-head **stop-gradient EMA constants** c_h;
     - **one global cosine**;
     - **no bias**.
   - Its argmax is a linear functional of S_t.
   - Re-done for the new map, with new CHECK numbers:
     - Lemma L is re-proved, and L′ is restated;
     - S1 is re-checked (unchanged);
     - the Goodhart rows, gradient table and contract are updated (expansion, norm growth, head domination, the additive magnitude clock, the EMA channel and the bias).
   - Delta heads only by default, set by a pre-registered $0 gate (G-pre).
2. **MUST-FIX 2: the bars are recalibrated for reachability.**
   - CERT h5p ≥ 0.40 at 16k is withdrawn.
   - CERT is now **h5p ≥ 0.30 at the final bin**. The bar is the C ceiling a_4 (high fit 0.21) plus a 0.09 misfit margin.
   - Reachability evidence and its gaps are stated.
   - A separate DELIVERED criterion (reachable) defines INSUFFICIENT-DOSE.
   - 16k values are descriptive.
   - **C8k is now part of the design.** P-V0 is the paired SVR − control step_recover at 8k, and the controls' expected 8k values are registered. Cost updated.
3. **New DATA:** K′=1 per-n accuracies (one think pass, then the answer decoder) put the trained think pass's one-shot reach at height ≤ 2. That supports A1′ in practice.
   - New finding: a single delta-rule erase inside the write is enough for a linear readout to recover a nested two-op value from operand copies (CHECK `erase3`, 1.000 for + and ×). So A1′ is empirical, not architectural.
   - Added: registered guards (g1) K′=1 and (g2) the pass-1 deep probe as a precondition on CERT (else ONE-PASS-COMPUTE), plus a descriptive input-swap diagnostic for the clocked variant.
4. **Kept unchanged:** G1, G2, the one-token Lemma 4 correction, S1, S2, Prop O (verified by Theory in deltamem.py 278–280), D7 (verified by Theory on 14 of 16 runs), the design structure and the cost basis.

**Tags.**
- **PROVED**: a complete argument is given here.
- **DERIVED(A…)**: follows from the named assumptions.
- **CONJECTURE**: argument or analogy only.
- **CHECK**: a CPU computation here.
- **DATA**: read from existing files. **POST-HOC** marks data seen before this design (all data here are post-hoc).

---

## 0. Summary

1. **Recommendation: SVR (shallow value readout).** It is a training-only probe term with these parts:
   - an 11-way cosine classifier (τ = 10, **no bias**);
   - inputs: the **linear** memory reads of S_t, scaled per head by a stop-gradient EMA constant and globally normalised, at the answer decoder's own " =" query (stop-gradient query). Delta heads only, unless G-pre says otherwise;
   - target: v_t, the value of reduction t, for t ≤ n;
   - weight w·γ(step), so it is off while the loop is open;
   - no pass token, no item text, dropped at test.
2. **Why this and not a dynamics target.**
   - **Prop G1 (PROVED):** any target measured through the index-aware full-trunk reader can be met by the contractive class C, because the reader does the reductions itself (M3/M3a DATA). This rules out (b).
   - **Prop G2 (PROVED):** any target on raw displacement (floor, band or ceiling) can be met without content. That is CSR's failure, and it rules out (c) and a content-free (d).
3. **SVR trains the property behind the strongest post-hoc separator we have.** The eval's linear probe recovers v_2 from the memory read at 0.73–0.89 in all 9 converged models and at 0.13–0.37 in all 7 failed ones (t = 3: 0.50–0.77 vs 0.10–0.27). There is no overlap, and the baseline is about 0.09 (DATA, POST-HOC). The failed models' full-trunk aux decodes e_3 at 0.97–1.00: their reader computes v_2 itself.
4. **What is proved about SVR (rev 2 map).**
   - **The readout performs no computation.** Its argmax is a linear functional of the state.
     - On superposed operand codes it cannot compute any mod-11 operation (Lemma L, PROVED). CHECK with learned codes: best accuracy 0.23–0.34, vs 1.000 for the rev-1 map.
     - Any reduction it reports must already be in the state's own interactions (L′). CHECK:
       - an outer-product store supports one op (1.000);
       - a pairwise store supports no nested pair of ops (+: 0.38–0.39; ×: 0.361 / 0.375);
       - **but a single delta-rule erase inside the write suffices for a nested two-op value from bare operand copies** (+: 1.000 / 1.000; ×: 1.000 / 1.000).
     - So the readout is clean, but the **write recurrence** can add levels within one pass. The one-pass ceiling (A1′) is therefore **empirical, not architectural**: supported by D12 and guarded by a registered K′=1 / pass-1 precondition on CERT.
   - **Stalled-state floor:** a stalled state pays ≥ 2 ln 2 − 2τδ per pair of consecutive distinct values (Thm S1, PROVED). This is undiluted by target length, unlike NOPASS (a correction to Part II Lemma 4).
   - **Value gap:** C's attainable SVR loss sits above an entropy floor that the computing class P can reach (Prop S2, PROVED).
   - **Certificate:** final-bin h ≥ 5 accuracy ≥ 0.30 certifies state-carried progress (Prop G3).
     - It is DERIVED under A1′. A1′ is empirical: K′=1 DATA support it, and the erase3 CHECK shows the architecture does not guarantee it.
     - CERT counts only if the K′=1 / pass-1 guard passes.
5. **Honest limit.** SVR is a **tilt, not a removal** (Prop R).
   - Toy (CHECK, illustrative): P moves from 0.62 to 0.90 at w = 1, and to 0.83 under full gaming. An equal-weight absorbable drive gives 0.81.
   - **No candidate examined has a guarantee of P → 1.**
6. **Design.**
   - 3 seeds on RTX 3090: RMS + `--svr 1.0,10`, paired with s185 RMS s1/s2/s3. Bitwise to step 2000.
   - **Plus C8k:** each control re-run to 8k with a checkpoint.
   - About $8.5 central, $10.1 high (at Audit's $10 line).

---

## 1. Facts from code and data

**Code (boxes/, read-only).**

| # | fact | where |
|---|---|---|
| F1 | The aux decoder is the SAME trunk and LM head, read-only. Prompt [EOT, " Step", t, " :"] (or no t under --aux_nopass), plus e_{t+1} **teacher-forced** (`seqs.append(pre + tgt[:-1])`). The read enters at block 6 at every prompt and prefix position; then blocks 6–11 run. No problem text. | nested170.py 228–233, 260–272; deltas5.py `fwd_read` 91–98 |
| F2 | Per-pass aux CE is a **token mean** over e_{t+1}+EOT. Then it is meaned over the item's passes t ≤ n, then over items. λ = 1.0 constant. | nested170.py 720–728; s185_rms_s1/args.json |
| F3 | No detach across passes: full BPTT through K = 8. | nested170.py 716–726 |
| F4 | Write input u = (1−γ)·LN_6(h6) + γ·LN_w(h_w). k, v and the gate come from u. h_w is pre-ln_f. γ = 0 for steps ≤ 2000, linear to 1 at 6000. Fade goes 0 → 1 over 500 steps. h6 is captured before read_fixed. | deltamem.py 187–188, 278–280, 285; nested170.py 1034–1035 |
| F5 | The read is r = fade·W_o[ e^{s_add}·RMS(φ(q)ᵀA)·add_gain ‖ e^{s_del}·Δq̂ ]. W_o is zero at init; α = σ(5), β = 1. The raw additive read φ(q)ᵀA is linear in A and grows ∝ t in the open loop (A_t = t·a). | deltamem.py 67–76, 153–154, 158–176 |
| F6 | The trunk is trainable. AdamW wd 0.01 over all m.parameters(); global clip at 1.0. | core.py 113; fsq120.py 731–737; nested170.py 959, 1067 |
| F7 | `step_recover`: a multinomial logistic, fit per t, on the pre-W_o reads with gains at the answer " =" position (768-d, standardised). Fit / test sizes by t: 4000/1000, 3929/873, 3398/752, 2857/628, 2287/509, 1739/387, 1157/257, 583/133. | nested170.py 568–597, 617–631; eval.json `step_recover_info` |
| F8 | CSR read R through W_o, with log-ratio λ and a floor-only pre term. Its gradient ∝ 1/‖W_o‖. | nested170.py 502–534, 709–734; THEORY.md 15113–15118 |

**Data (all POST-HOC).**

| # | fact | source |
|---|---|---|
| D1 | Order parameter: converged delta_cos 0.44–0.63 / agree 0.10–0.14; failed 0.69–0.98 / 0.29–0.82. | THEORY.md 14302–14306 |
| D2 | RMS s1 / s2 entered at 19k / 19k (Part II: 18.3k / 17.5k). s3 never entered (final 1.03, clip rate 1.0). | THEORY.md 13971–13972; THEORY_SEEDSTAB 929–942 |
| D3 | Failed pass-token models' late aux errors are 99–100% STRUCT-RIGHT value errors, with 0 lag and 0 echo. **The reader does the reductions.** | THEORY.md 14589–14595, 14631–14640 |
| D4 | E1: converged models compute in progress-gated delta heads (rms s1 / s2 "almost entirely"). Failed models route through the additive integrator. | THEORY.md 14751–14767 |
| D5 | T1: delta_cos co-moves with the loss and crosses 0.65 within ±4k of entry. Not an 8k detector in two entering runs. | THEORY.md 14925–14943 |
| D6 | CSR s1: P-A1 met (+0.325); 8k probe contractive (0.910 / 0.902). Delta-read λ_pre +0.12 / +0.30 / +0.44 at 2k / 4k / 8k. NOT-CROWDED. Never entered; its control entered. | THEORY.md 15224–15244 |
| D7 | **step_recover by t = 1..8** (majority baseline 0.08–0.11). Verified by Theory on 14 of 16 runs. | eval.json `s175` |

D7 in full:
- **Converged (9):**
  - rms s1 .97/.84/.68/.54/.52/.42/.43/.22; rms s2 .96/.84/.68/.58/.50/.40/.47/.22;
  - rmsnopass s1 .98/.89/.77/.73/.67/.61/.54/.44; rmsnopass s2 .97/.78/.60/.46/.38/.38/.27/.18;
  - nopass s1 .92/.73/.50/.43/.36/.42/.48/.38; nopass s2 .94/.79/.63/.55/.45/.38/.45/.29;
  - window s1 .98/.79/.58/.47/.34/.26/.26/.12; window s2 .98/.87/.76/.67/.62/.59/.54/.29;
  - reweight s2 .93/.78/.56/.52/.49/.51/.58/.60.
- **Failed (7):** t = 1..4 values: rms s3 .86/.23/.13/.14; csr s1 .29/.14/.12/.12; den s1 .95/.34/.24/.17; rwwin s1 .94/.37/.23/.15; rmsgf s3 .18/.13/.10/.12; hold s1 .90/.26/.16/.15; nofade s1 .94/.34/.27/.19. All ≤ 0.16 at t = 5–8.
- No overlap at t = 2 (min 0.73 vs max 0.37) or at t = 3 (0.50 vs 0.27). The failed models' aux accuracy at t = 2 is 0.97–1.00.

Further data and checks:

| # | fact | source |
|---|---|---|
| D8 | DIRECT acc_by_n for n 1–4: s1 1.00 / .98 / .64 / .43; s2 .99 / .99 / .75 / .53. | s170_direct_s{1,2}/eval.json |
| D9 | Fitted one-shot accuracy by height h 1..5: s1 1.00 / .97 / .51 / .15 / .14; s2 .99 / .98 / .67 / .21 / .12 (max residual 0.04). | CHECK `direct_height.json` |
| D10 | Target-height shares (h ≥ 3 / ≥ 4) by t: t2 0 / 0; t3 .34 / 0; t4 .53 / .15; t5 .61 / .33; t6 .71 / .51; t7 .85 / .73; t8 1 / 1. Pooled over t ≥ 2: h1 .25, h2 .32, h≥3 .43. Among h ≥ 4 targets: h = 4 63.5%, h ≥ 5 36.5%. | CHECK `heights.json`, `direct_height.json`, `h4_split.json` |
| D11 | Cost: s185 RMS 42,491 / 42,576 / 35,432 s per 48k. Hosts $0.179–0.228 /h. "About $2.3 per 48k 3090 cell". CSR s1 about $2.5. | eval.json; THEORY.md 14227, 14922, 14434, 15255 |
| D12 | **K′ = 1** (one think pass, then the answer decoder) at n = 1 / 2 / 3–6: rms s1 1.00 / .54 / ≤.10; rms s2 1.00 / .60 / ≤.12; rms s3 1.00 / .70 / ≤.13; rmsnopass s1 1.00 / .77 / ≤.11; nopass s1 .99 / .76 / ≤.11. **In trained models one think pass reaches height 2 at 0.54–0.77 and height ≥ 3 at chance**, below DIRECT's a_3. K′ = 0 is chance from n ≥ 2. | eval.json `acc_by_n_kprime` |
| D13 | Theory's review check (learned free codes, H 12, D 64, τ 10), best accuracy: + raw 0.157 / 0.165, global-no-bias 0.264 / 0.248, **per-head 1.000 / 1.000**; × raw 0.264, global 0.355 / 0.336, **per-head 1.000 / 1.000**. | `theory/seedstab2/perhead_norm.json` |

---

## 2. Mechanism recap and CSR post-mortem

**The bad class C (D3 / D7, consistent with Part I Cor. 1.2):**
- The memory carries an input-determined summary, plus at most about one reduction (v_1).
- The **pass-token full-trunk reader computes the remaining reductions itself**. It decodes e_3 at ≥ 0.97 while v_2 is linearly absent.
- Contraction (D1) is the symptom: a state with no pass-specific progress forgets its start.

**Correction to Part II Lemma 4 (PROVED).** Under teacher forcing (F1), consecutive passes' decoder inputs coincide only up to the first differing index j*.
- So the bound holds at **one token per pass pair**: CE_t[j*] + CE_{t+1}[j*] ≥ 2 ln 2 − 2ε (Lemma 4's 4ε is valid but loose).
- In token-mean aux units, the NOPASS penalty is ≈ 2 ln 2/(L_t + 1) per pair, with L_t = 5–25. Part I Thm 1(c) used j* correctly.
- So NOPASS is a weak tilt, and a pass-free penalty needs a **single-token target** to have full strength.

**Prop G1 (deep-reader targets are satisfiable by C; PROVED, constructive).**
- Let D compute e_{t+1} from (S*(x), t) for t ≤ d_R (D3: d_R ≈ 3–4).
- **(i)** Any target that is a function of D's outputs at pass t is optimal on C for t ≤ d_R. This covers the aux CE and candidate (b)'s margin log p(e_{t+1}|S_t) − log p(e_t|S_t).
- **(ii)** Without the token but with a state-carried clock c_t (the delta transient Δ*(I − P^t)), D reads t from c_t. The "state-swap" margin log p(e_{t+1}|S_t) − log p(e_{t+1}|S_{t−1}) is then also met: D refuses mismatched clocks.
- **Proof.** D(S, prompt) = render(σ^{t̂}(x̂(S))). ∎
- **Lesson.** A decoder-mediated target certifies progress only if the decoder can neither index passes nor reduce.

**Prop G2 (displacement targets are satisfiable without content; PROVED, constructive).** Any target on d_t = ‖R_{t+1} − R_t‖ alone is met without content:
- **floor:** expansion of a task-irrelevant read component (CSR s1: delta λ +0.44);
- **band ∋ 0:** a rotation, or a near-reflection with β → 2 on a content-free key (|1 − β| → 1, d_t constant);
- **ceiling:** a stall.

**CSR post-mortem.**
- CSR's statistic is content-free (G2), and its floor had no upper bound. It was met by runaway delta-read expansion (+0.44 at 8k).
- That lies outside the converged class's λ ≈ 0 region (Part I §1.4). The model stayed in C (0.910 at 8k), and the cause was not crowding (D6).
- Likeliest harm path (CONJECTURE): it rewarded content-free motion in the delta channel, the channel the good class computes in (D4).

**A safe target needs three properties.** CSR failed (S1) and (S3).
- **(S1) Value gap:** it cannot be optimised on C.
- **(S2) Alignment:** the good class nearly meets it.
- **(S3) Content:** motion, norm or scale cannot meet it.

---

## 3. Candidate comparison

| candidate | met by C or content-free motion? | value gap on C | paired with s185? | proof status | rank |
|---|---|---|---|---|---|
| **SVR** (M3 as an added linear probe) | only on low-height targets via one-shot precompute (bounded by A1′; detector registered) | yes (S2) | yes (bitwise to 2000) | PROVED readout limits; DERIVED tilt | **1** |
| M1 FAIR-START (γ from 0 + NOPASS + REWEIGHT) | tilt; NOPASS penalty is 1/(L+1) of what was claimed | partial | no (3 changes) | CONJ (post-hoc 6/7) | 2 |
| M3-REPLACE (shallow reader *instead of* the full aux) | no aux credit for C | yes | no | PROVED removal of aux credit | 3 |
| (d) M4 STEER, decoder-visible statistic | yes if content-free (G2); content-ful = SVR | — | yes | — | 4 (folds into SVR) |
| (b) decodable-progress floor through the full-trunk reader | **yes** (G1) | no | yes | gameable, PROVED | 5 |
| (c) two-sided band on the step ratio | **yes** (G2) | no | yes | gameable, PROVED | 6 |

M3-REPLACE's risk is the representational burden: every token of e_{t+1}, copies and shifts included, would have to be linearly readable at teacher-forced prefix queries. Use it only if SVR reads PRECOMPUTE-GAMED, or if SVR + NOPASS fails.

---

## 4. The method: SVR, with its derivation

### 4.1 Definition (rev 2)
- **Query (stop-gradient).** h6^= = blocks 0..5 on [EOT, x, " ="] at the " =" position, under no_grad. Then q = qkv_q(LN_6(h6^=)), also under no_grad. It is pass-invariant, with no pass token.
- **Raw reads at pass t, linear in S_t:**
  - delta heads: r_h = Δ_h q̂_h;
  - additive heads (only if G-pre selects all heads): r_h = φ(q_h)ᵀA_h, the raw einsum, with no RMS and no den division.
- **Per-head constants.** c_h is a **stop-gradient, state-independent** buffer: an EMA (decay 0.99) of the per-head RMS of r_h over items, passes t ≤ n and dimensions.
  - Updated under no_grad *after* each SVR step, from that step's reads. The value used at step s comes from steps < s.
  - Initialised at the first SVR step on a fixed calibration batch (own rng 19930, so no training-rng draw) under no_grad.
- **Features.** f_t = concat_h r_h/c_h (384-d with delta heads only, 768-d with all heads).
- **Readout.** z_c = τ·⟨w_c/‖w_c‖, f_t/‖f_t‖⟩, c ∈ Z_11, τ = 10, **no bias**.
  - Because τ/‖f_t‖ > 0 is common to all classes, **argmax_c z_c = argmax_c ⟨ŵ_c, f_t⟩, a linear functional of S_t**.
- **Loss.** ℓ_SVR = mean_items mean_{t ≤ n} CE(z(f_t), v_t), with v_t = `reductions(root)[t−1]`.
- **Total loss.** L = answer CE + 1.0·aux + w·γ(step)·ℓ_SVR, with w = 1.0.
- **Head choice (G-pre, $0, before launch; the rule is fixed now).**
  - Experiments recomputes step_recover from the **delta-head reads only** (384-d, same protocol and seeds) on the 16 models of D7.
  - If converged min > failed max at both t = 2 and t = 3, SVR reads **delta heads only**. Otherwise it reads all 12 heads, with the magnitude-clock caveat of §5.
  - Why the default: the raw additive read grows ∝ t (F5). Inside a pass-free linear readout, that is a pass-magnitude clock. E1 (D4) also places the good class's computation in the delta heads.

### 4.2 Lemmas and theorems

**Lemma L (rev 2: the readout cannot reduce superposed operand codes; PROVED, CHECK).**
- **Claim.** Let the state hold operand contributions superposed, S = S_a + S_b, in any heads. Then no SVR readout computes op(a, b) for op ∈ {+, −, ×, ÷} mod 11.
- **Proof.**
  - f is linear in S, so f(S) = f(S_a) + f(S_b).
  - The argmax is that of s_c = ⟨ŵ_c, f⟩ = f_c(a) + g_c(b). The common factor τ/‖f‖ > 0 drops out, and there is no bias.
  - Each class has a perfect matching M_c (+, −: {(a, c ∓ a)} on Z_11²; ×, ÷: {(a, ca⁻¹)} and {(cb, b)} on (Z_11^×)²).
  - Σ_{M_c} s_{c′} is the same for every matching, so correctness on M_c and on M_{c′} contradict each other. ∎
- **Why the two rev-2 changes matter.**
  - **Per-head normalisation** gives (φ_h(a) + ψ_h(b))/n_h(a, b) with different factors per head. Those factors are an interaction.
  - **A bias after global normalisation** gives argmax_c [s_c + b_c‖f(a, b)‖], a rank-1 interaction n(a, b)·b_c.
  - Both are removed. Class priors stay representable through a constant component of f, which sits inside the common factor and keeps s_c sum-decomposable.
- **CHECK** (`linmap_check.json`: learned free codes, readout learned end to end, H 12, D 64, τ 10, best training accuracy over 2 seeds; the EMA constants are stood in for by a detached batch RMS per head):

  | op | rev-2 linear map | rev-1 per-head map |
  |---|---|---|
  | + | 0.248 / 0.231 | 1.000 / 1.000 |
  | × | 0.250 / 0.340 | 1.000 / 1.000 |

  - These agree with Theory's D13 and with the rev-1 numpy fit (0.14–0.30).

**Lemma L′ (rev 2: all interactions must be in the state; PROVED for the readout, CHECK for stores).**
- Since the readout is linear in S, it adds no interaction. Anything it computes is a linear functional of interactions the **writes** put into S.
- **(i) Outer product, one op.** An outer-product association v(b)k(a)ᵀ, formed by the write from two operand copies, makes ⟨w_c, Δq̂⟩ = ⟨w_c, v(b)⟩(k(a)·q̂) bilinear in (a, b). So one op can be read out. CHECK `bilin`: + 1.000 / 1.000, × 1.000 / 1.000. A rank-10 Fourier form gives + mod 11 exactly.
- **(ii) Pairwise store, no nested pair.** A store of all pairwise outer products of three operand codes is degree 2. Its readout cannot compute (a ∘ b) + c.
  - **PROVED for (a ± b) + c** (Latin-square version of Lemma L).
    - The scores are pairwise-decomposable: s_y = f_y(a, b) + g_y(a, c) + h_y(b, c).
    - For each class y, T_y = {(a, b, y − a ∓ b)} contains every (a, b), every (a, c) and every (b, c) pair exactly once.
    - So Σ_{T_y} s_{y′} is the same for all y, and correctness on T_y and on T_{y′} contradict each other. ∎
  - For (a × b) + c the pair counts depend on y, so the proof does not apply; CHECK only.
  - CHECK `pair3`:
  - rev-2 map: + 0.381 / 0.389, × 0.361 / 0.375;
  - rev-1 per-head map: + 1.000 / 1.000, × 1.000 / 1.000, i.e. **two deferred ops, as Theory warned**.
- **(iii) Delta erase, degree 3.** A single delta-rule erase within the write, Δ ← Δ(I − βk_c k_cᵀ) + βv_c k_cᵀ, creates degree-3 terms (Δk_c)(k_c·q̂) (PROVED, algebra).
  - CHECK `erase3`, learned operand codes plus one erase with operand c's key: + **1.000 / 1.000**, × **1.000 / 1.000** under the rev-2 linear map (rev 1: 1.000 / 1.000 and 1.000 / 1.000).
  - **One delta step is enough for the memory itself to hold a nested two-op value, built from operand copies.**
  - This matches the state-tracking capacity of delta / negative-eigenvalue linear recurrences (Grazzi et al. 2411.12537).
- **Consequence.**
  - The readout adds no interaction. It can at most read out one op that an outer-product layout stores in factored form.
  - Further levels are computed by the **write recurrence**: within a pass (one-shot, the territory of A1′) or across passes (progress).
  - The within-pass recurrence is **not bounded by the transformer's depth**, so A1′ is an empirical assumption about this architecture as trained (D12), not a structural fact.

**Theorem S1 (pass-free single-token floor; PROVED; re-checked for rev 2).**
- **Claim.** With δ_t = ‖f̂_{t+1} − f̂_t‖ (f̂ = f/‖f‖) and v_t ≠ v_{t+1}: CE_t + CE_{t+1} ≥ 2 ln 2 − 2τδ_t.
- **Proof.** Unchanged:
  - with unit rows and no bias, |z_c(f̂) − z_c(f̂′)| ≤ τ‖f̂ − f̂′‖;
  - |Δ log softmax| ≤ 2‖Δz‖_∞;
  - p(y)p(y′) ≤ 1/4. ∎
- **Rev 2 details.**
  - δ_t ≤ 2‖f_{t+1} − f_t‖/‖f_t‖ is a *relative* change of the linear features. f̂(λS) = f̂(S) for λ > 0 (PROVED, linearity), so a common scale-up is invisible.
  - Under read-space contraction δ_t ≤ κ^{t−t₀}δ_{t₀}.
  - Per unit weight the floor is ≈ ln 2 per pass, ≈ (L+1)× NOPASS's.

**Prop S2 (value gap; PROVED for exact stalls, DERIVED for near-stalls; unchanged).**
- **Claim.** If f_t = f* over m late passes, then for every readout Σ CE(p*, v_t) = m[H(p̂_x) + KL(p̂_x‖p*)] ≥ m·H(p̂_x). P attains 0.
- **What C can still gain.** Only two things, both bounded:
  - (i) the per-item value set;
  - (ii) one-shot precompute with a clock (G3).

**Prop G3 (precompute ceiling and certificate; DERIVED(A1′) with L and L′).**
- **A1′:** one think pass, *including its within-pass delta recurrence* and the stop-gradient query path, computes a height-h value with accuracy ≤ a_h.
  - The proxy for a_h is DIRECT (D9): a_3 0.51–0.67, a_4 0.15–0.21, a_{≥5} ≈ chance.
  - **Support (D12, POST-HOC):** trained think passes reach height 2 at 0.54–0.77 and height ≥ 3 at chance (K′ = 1), even through the full-depth answer decoder. The DIRECT proxy is therefore conservative for models trained so far.
  - **Not guaranteed:** the `erase3` CHECK shows a single delta step can hold a nested two-op value. SVR pressure could teach a pass to exploit this.
  - **Registered guard (precondition for CERT), read on the final model:**
    - **(g1)** K′=1 pooled accuracy at n 3–8 ≤ 0.20. D12: ≤ 0.13 in every trained model, chance ≈ 0.09. This field already exists in eval.json `acc_by_n_kprime`.
    - **(g2)** A fresh step_recover-protocol probe on S_1 (after one pass), targeting the later values v_t with h ≥ 4, reads ≤ 0.21.
    - If either fails, the cell reads **ONE-PASS-COMPUTE**: CERT is void, because deep values are formed within a pass. (g1) also protects C1's THINK-NECESSARY.
    - The guard does not cover a *clocked* one-pass evaluator (v_t formed within pass t and selected by a clock). That route is only screened by the descriptive input-swap diagnostic (§7), which is the residual risk.
- **Ceiling.** A C-type (input-determined or clock-selected) solution has height-h accuracy ≤ max(a_h, a_{h−1}), where a_{h−1} is the ≤ 1 deferral of L′.
  - Pooled over the target mix (D10): h ≥ 4 ≤ 0.15–0.18 without deferral and ≤ 0.38–0.50 with it; **h ≥ 5 ≤ a_4 ≈ 0.15–0.21 even with deferral**.
- **CERT (rev 2): in-training svr_acc_h5p ≥ 0.30 over the final 1k bin (47001–48000).** The bar is a_4's high fit (0.21) plus a 0.09 margin:
  - more than 2× the fit residual (0.04) and the s1/s2 spread at h = 4 (0.06);
  - one chance unit (1/11).
  - Sampling noise is negligible: about 20,000 h ≥ 5 targets per 1k bin, SE ≈ 0.003.
  - Under D12's ceiling (chance at h ≥ 3), the effective margin is about 0.18.
- **Reachability by the good class (DATA + DERIVED, not established).**
  - **(a)** The final converged models' post-hoc probe at t = 5–7 reads ≥ 0.30 in 24 of 27 (model, t) cells (D7; exceptions window s1 t6 / t7 0.26 / 0.26 and rmsnopass s2 t7 0.27; CHECK `pooled_t2p.json`). The failed models read ≤ 0.16 there.
  - **(b)** That probe is sample-starved at late t: 768-d × 11 classes on 2,287 / 1,739 / 1,157 fits at t = 5 / 6 / 7 (F7). Its accuracy falls with fit size while the converged aux handle stays ≥ 0.99 at every t. So the D7 late-t values are lower bounds on linear presence (CONJECTURE: no learning curve exists). The trained head sees about 3 × 10⁶ (item, pass) targets.
  - **(c)** Under P, a value written at pass t does not depend on its height except through compounding of upstream errors, which is small at the converged per-step reliability (≥ 0.99; DERIVED). The h ≥ 5 subset is therefore expected near the t ≥ 5 level.
  - **(d)** Not shown: height-resolved presence in any existing model, and presence before entry. Hence CERT is read **at the final bin only**; 16k is descriptive.

**Prop O (the open loop trains only C; PROVED; verified in code by Theory).**
- For γ = 0, u = LN_6(h6) is input-only, with h6 captured before the read (deltamem 278–280). So any SVR gradient shapes only input-determined content.
- For γ ∈ (0, 1), the S-dependent part of the write carries γ. Hence the weight w·γ(step), and exact pairing to step 2000.

**Theorem T (tilt in the threshold kernel; kernel PROVED, mapping DERIVED(A_align) / CONJECTURE; unchanged).**
- ε_c = ε_needed·exp(−2∫η(D_aux + wD_SVR)).
- D_aux is absorbable (→ 0 on C as the reader learns). D_SVR ≥ D̲ > 0 on C (S2, G3). So z rises monotonically in w, at a slope C's reader cannot erode.
- **CHECK** (`race_toy.json`, illustrative):
  - P = 0.62 at w = 0;
  - no gaming: 0.78 / 0.84 / 0.90 / 0.95 at w = 0.25 / 0.5 / 1 / 2;
  - full shallow-target gaming: 0.71 / 0.77 / 0.83 / 0.89;
  - equal-weight absorbable drive: 0.81 (w_abs = 1) and 0.87 (w_abs = 2).

**Prop R (SVR is a tilt, not a removal; PROVED).** The pass-token aux keeps C's credit up to reader depth (D3). The answer CE keeps DIRECT-level credit.

**C1 preserved (PROVED).** The input is identical every pass. The memory remains the only cross-pass channel. SVR is read-only and dropped at test: the head and its buffer are excluded from model.pt. All C1 tests are unchanged.

---

## 5. Goodhart analysis (rev 2)

| route to meet SVR without task progress | status | why / detector |
|---|---|---|
| Expansion of read motion (CSR's route) | **blocked** | A common scale-up is invisible (f̂(λS) = f̂(S), PROVED). Growth of one content-free component tilts f̂ toward it and shrinks the value margin to ≤ 2τ‖f_rest‖/‖f‖ (PROVED), so it is self-defeating |
| Whole-state norm growth | **blocked** | same invariance. The EMA constants follow slow drifts |
| **One head dominating the global cosine** | **no game; self-defeating** | The readout stays linear (no interaction, Lemma L). A content-free dominant head caps the margin as above. A content-carrying dominant head is a legitimate register. The EMA (~100-step time constant) restores the balance |
| **Additive magnitude clock** (raw φ(q)ᵀA ∝ t) | **avoided by default** (delta heads only, G-pre); bounded if all heads | A linear readout with a magnitude clock can do clocked selection (upper envelope of t·α_c + β_c), but only among values already in the state. That is clocked precompute (next row) |
| Clocked one-shot precompute | **bounded + detected** | G3 ceiling. **PRECOMPUTE-GAMED** = DELIVERED and svr_acc_h12 ≥ 0.8 and svr_acc_h5p ≤ 0.20 at the final bin, plus a contractive final path_indep (delta_cos ≥ 0.69). Descriptive input-swap diagnostic (§7) |
| The readout does the reduction | **blocked** (Lemma L); **≤ 1 deferred** via outer-product layouts (L′(i)); nested deferral blocked (L′(ii) CHECK) | a deferred op still needs its operands (earlier values) in the state |
| **Within-pass delta recurrence computes extra levels** (L′(iii): one erase suffices, CHECK 1.000) | **not blocked by architecture**; bounded only empirically (D12) | **registered guard** (g1) K′=1 n 3–8 ≤ 0.20 and (g2) pass-1 deep probe ≤ 0.21, else ONE-PASS-COMPUTE (CERT void). The clocked variant is screened only by the descriptive input-swap diagnostic; risk 5 |
| A bias as an interaction channel | **removed** | no bias (§4.2) |
| EMA constants as a channel | **blocked** | stop-gradient; previous steps' batch statistics; no per-item or current-state dependence |
| Readout amplifies a vanishing clock | **bounded** | S1: margin ≤ 2τδ_t |
| Query-side precompute | **bounded** | q is stop-gradient (SVR cannot shape it) and is part of a one-shot pass, so A1′ applies |
| Encoding v_t without using it | **harmless** | for h ≥ 5 the value needs earlier values from the state (A1′), wherever the copy lives |
| Memorising items; pass-index leak | **blocked** | fresh draws; no token, prompt or item text in the probe |
| Learning only the per-item value set | **bounded** | the floor H(p̂_x) (S2) |

---

## 6. Gradient flow (from the code; ℓ_SVR alone; rev 2)

| parameter / buffer | receives ∇ℓ_SVR? | route |
|---|---|---|
| svr.W (11 × 384 or 11 × 768) | yes | direct. **There is no svr.b** |
| svr.c (EMA buffer, 6 or 12) | **no** (buffer, updated under no_grad) | — |
| mem.qkv rows [D:3D] (k, v), bias | yes | S_t ← write_seq k/v, every pass ≤ t (deltamem 187) |
| mem.gate (α, β) | yes | the delta update (deltamem 188, 210) |
| mem.ln | yes, via the write's LN_6(h6) (factor 1−γ), and via the think-pass reads for t′ ≥ 2 (γ) | **not** via the probe query (no_grad) |
| trunk blocks 0–5, wte, wpe | yes | h6 of the think input |
| mem.ln_w, trunk blocks 6–11 | yes iff γ > 0 | h_w |
| mem.wo, mem.qkv rows [0:D] (q), s_add, s_del, add_gain | yes iff γ > 0, through think reads of S_{t′−1}, t′ ≥ 2 (wo.bias from pass 1) | closed loop only; the probe's features exclude gains, RMS and W_o |
| ln_f | **exactly 0** | h_w is pre-ln_f (deltamem 285–288) |

**Smoke-test assertions.**
- **A-SVR-1:** at γ ∈ (0, 1), ℓ_SVR-only backward gives a finite, nonzero gradient on svr.W, mem.qkv.weight[D:3D], mem.gate.weight, mem.ln.weight, g h_0 c_attn, mem.ln_w.weight, g h_11 mlp, mem.wo.weight and mem.qkv.weight[0:D].
- **A-SVR-2:** with γ forced to 0, **exactly 0** on mem.ln_w, blocks 6–11, wo, qkv q-rows, s_add, s_del and add_gain. ln_f has 0 or None.
- **A-SVR-3 (rev 2):** svr.c is unchanged by backward and changes only in the post-step EMA update. svr has no bias parameter.
- **A-ANS / A-AUX:** each nonzero on its usual parameters.
- **A-PRE:** before γ > 0, svr.W.grad is None.

---

## 7. Predictions, design and cost (rev 2)

**Cells.**
- `s20x_svr_s1/s2/s3` = s185_rms_s{seed}/args.json + `--svr 1.0,10` + `--keep_ckpt_steps 4000,8000,12000,16000`. RTX 3090, 48k, uncached, fast_add, micro 1.
- **Pairing:** log records at steps 100–2000 bitwise equal to the control on loss, answer_ce and aux_ce; the control's driver (s1 / s2 570.172.08, s3 595.84); same micro. Otherwise UNPAIRED.
- **C8k (now part of the design):** each control's args run to 8k on its driver, keeping ckpt_model_8000.pt (simplest on the same host as its SVR cell, which needs the same driver).
  - PAIRED-C8k if records 100–8000 are bitwise equal to the s185 log. Otherwise UNPAIRED-C8k: an independent control-recipe draw, read descriptively.
  - This is the only source of control values at 8k.
- **Optional C16k** (+$1.2): the same re-runs continued to 16k.

**Log fields** (each 100-step interval; no rng): svr_w_eff, svr_ce, svr_acc_by_t, svr_acc_t2p (pooled t ≥ 2), svr_acc_h12, svr_acc_h3, svr_acc_h4p, svr_acc_h5p, the counts, and svr_c (the EMA constants).

**Checkpoint evals ($0, local, fixed seeds), at 8k / 16k / final for SVR cells and at 8k (16k) for C8k:**
- `step_recover` (unchanged protocol);
- s182.path_indep;
- gradshare (SVR cells, 4k / 8k);
- **input-swap diagnostic (descriptive, DERIVED(A_x)):**
  - S_t′ = F(S_{t−1}(x); x′), with x′ a derangement donor input. The probe query is q(x′), and accuracy is scored against v_t(x′) on h ≥ 4 targets, with the clean accuracy beside it.
  - C's one-shot mechanism transfers to x′ (acc_xswap ≈ acc_clean). P's does not (≈ chance), because v_t(x′) needs x′'s progress.
  - **A_x:** C overwrites the probed keys freshly each pass (failed models' β ≈ 0.85–0.9, D4).
  - Readout: the trained head for SVR cells; a fresh step_recover-protocol probe (fit on clean states) for both arms.

**Registered predictions.**

| id | prediction | bar | reading if missed |
|---|---|---|---|
| **P-V0** (8k early read, dose; per seed, paired) | SVR makes values present before the controls do | step_recover mean over t = 2..4 at 8k: SVR cell − its C8k control ≥ +0.15, in 3/3 | **EARLY-DOSE-MISSED** (descriptive; points to risk 2). Partly by construction, since SVR trains this very property: a dose check, not a class read |
| **C8k-expect** (registered, descriptive, controls) | the controls are all on the C side at 8k and not yet separated (T1; 8k losses 2.29–2.50, all pre-entry) | each control: t = 1 ≥ 0.8; mean over t = 2..4 ≤ 0.35; \|s1 − s3\| and \|s2 − s3\| ≤ 0.10. Control path_indep 8k delta_cos ≥ 0.85 | If s1 / s2 exceed s3 by ≥ 0.15, value presence is an 8k detector of the eventual basin (supports Part II P2). Reported, no verdict |
| C16k-expect (if run) | presence precedes or accompanies entry | s1 / s2 (17.5–19k entries) mean over t = 2..4 ≥ 0.45 at 16k; s3 ≤ 0.35 | descriptive |
| **DELIVERED** (final bin, per seed) | the probe term was met | svr_acc_t2p ≥ 0.35 over 47001–48000. Chance 0.09. The post-hoc probe pooled over t ≥ 2 (test-count weighted) reads 0.50–0.73 in all 9 converged and 0.11–0.22 in all 7 failed final models (CHECK `pooled_t2p.json`), so the bar is reachable by the good class. It is **not a certificate**: gamed h ≤ 2 targets alone could exceed it | with no entry → **INSUFFICIENT-DOSE** |
| **CERT** (final bin, per seed) | state-carried progress certified | svr_acc_h5p ≥ 0.30 (G3), **and** guard (g1) K′=1 n 3–8 ≤ 0.20 **and** (g2) pass-1 deep probe ≤ 0.21 on the final model | guard fails → **ONE-PASS-COMPUTE** (CERT void). Entered without CERT → "ENTERED-UNCERTIFIED" (method credit withheld) |
| 16k mechanism (descriptive) | deep values emerging | report svr_acc_h4p / h5p in the 15001–16000 bin. Expected direction: h5p > 0.21 if the seed is on the P path | — |
| P-V2 (entry) | enters | entry(0.5) ≤ 24k and final 1k bin < 0.1 | — |
| P-V3 (no harm; s1, s2) | still enters | entry(0.5) ≤ 24k | **HARMFUL** if never entered |
| P-V4 (class) | not B | final acc_from_zero ≥ 0.95 and (delta_cos ≤ 0.65 or acc_from_donor ≥ 0.9) | — |
| P-V5 (arm-level, descriptive) | tight, earlier entries | entry SD ≤ 2k; mean ≤ 19k | — |
| OP-8k (descriptive per T1) | below CSR s1's 8k reading | delta_cos_8k < 0.910 and agree_8k < 0.902 | not a falsifier |

**Labels.**
- **SUPPORTED:** DELIVERED, CERT, P-V2 and P-V4 in 3/3, with no HARMFUL.
- **FAILED:** any HARMFUL, or s3 not entered while DELIVERED.
- **PRECOMPUTE-GAMED:** §5.
- **ONE-PASS-COMPUTE:** guard g1 or g2 fails. CERT is void, and the cell is reported as a method failure on mechanism, whatever its entry.
- **INSUFFICIENT-DOSE:** not entered and not DELIVERED.
- **MIXED:** otherwise.

**Account vs method.**
- **Account falsifier:** final bin ≥ 0.5 (not entered) **and** CERT met **and** the input-swap diagnostic reads P-like (acc_xswap ≤ acc_clean − 0.15) **and** path-dependent (final delta_cos ≤ 0.65) **and** NOT-CROWDED.
  - Values carried in the state without entry means the reader shortcut was not the barrier.
- **Method failures** (the account is left untested): INSUFFICIENT-DOSE, PRECOMPUTE-GAMED, HARMFUL.
- **Crowding:** G_svr/G_task ≥ 1 on the memory branch at 4k or 8k reads CROWDED.

**Claim limits.**
- P(3/3 | p = 2/3) = 0.30.
- The s3 cell diverges at 2001, before the decision window, so P(s3 enters | no effect) ≈ 0.6.
- A P(enter) claim needs a second-stage panel; that is the user's call above $10.

**Cost.**
- SVR cells: about 43–46k s each at $0.179–0.228 /h, so $2.2–2.9 each; 3 cells $6.6–8.7.
- **C8k:** 3 × 8k × ~0.89 s ≈ 2.0 h each, $1.1–1.4 in total.
- **Total about $8.5 central, $10.1 high**, at Audit's $10 line. If it binds, Audit can drop C8k for one entering seed (s1 or s2), keeping s3's pair. P-V0 then reads on 2 seeds.
- C16k: +$1.2, optional.
- G-pre, checkpoint evals and gradshare: $0 local.

**Engineering contract (nested170.py; described, not edited).**
1. **Flag** `--svr w,tau`. Assert: inject arm, uncached loss_items path (no --k_window, --anneal, --cache_lower, --ckpt_passes or --csr), aux_step > 0.
2. **`class SVRHead`.**
   - Parameter W (11 × F) from `torch.Generator().manual_seed(19920)` (no global-rng draw). **No bias.**
   - Buffer c (one per probed head), initialised flag `c_init = False`.
   - Forward: the cosine readout on f/‖f‖.
   - Created as `m.svr` only with the flag.
3. **`svr_feats(mem, q, S, c, heads)`.**
   - The delta einsum with normalised q (and, if all heads, the raw additive einsum: no RMS, no den division, no gains).
   - Divide by c per head; concatenate. **No other normalisation before the global cosine.**
4. **In loss_items.** `svr_on = SVR and float(m.mem.gamma) > 0 and torch.is_grad_enabled()`.
   - h6^= and q are computed under no_grad once per batch.
   - **If not c_init:** compute c on the calibration batch (rng 19930) under no_grad, then set c_init.
   - After each `m.write`: CE over the items with t ≤ n. Add w·γ·ℓ_SVR. Collect stats and per-head mean squares under no_grad.
   - **After `opt.step()`:** c ← 0.99·c + 0.01·RMS(stats), under no_grad.
   - Targets come from `reductions`; heights from a new `reduction_heights(root)`, cached per item.
5. **mem_check:** set γ = 1 temporarily (SVR path at the largest shape), then restore it.
6. **Saving.**
   - **model.pt and ckpt_model_<step>.pt exclude `svr.*`.**
   - **svr_head.pt holds W and the EMA buffer c.**
   - ckpt_last.pt holds both, for resume.
   - Eval field `svr_eval` (descriptive).
7. **Eval scripts:** an input-swap mode in the checkpoint probe (or a scratch repro script).

**Gates.**
- **G-pre:** the delta-only step_recover rule (§4.1).
- **G0:** plain path bitwise.
- **G1:** flag-on, pre-γ steps bitwise, grads None.
- **G2:** A-SVR-1/2/3, A-ANS, A-AUX, A-PRE.
- **G3 fixtures (rev 2):**
  - linearity: f(S₁ + S₂) = f(S₁) + f(S₂) to 1e-6 at fixed c;
  - argmax(z(λS)) = argmax(z(S)) for λ > 0;
  - no bias parameter exists;
  - the EMA updates only after opt.step and under no_grad;
  - reduction_heights vs an independent recursion;
  - the S1 floor on a stalled state;
  - the stats bookkeeping.
- **G4:** mem_check.
- **G5:** strict model.pt load into a flags-off InjectN.
- **G6:** the pairing checks (cells and C8k).

---

## 8. Claim status (rev 2)

| claim | status |
|---|---|
| Part II Lemma 4 at one token per pass pair; NOPASS penalty ≈ 2 ln 2/(L+1) | PROVED |
| G1: index-aware deep-reader targets are met by C | PROVED (constructive; D3 DATA) |
| G2: displacement floors, bands and ceilings are met without content | PROVED (constructive) |
| Rev-1 Lemma L for the per-head-normalised map | **WITHDRAWN** (false: CHECK 1.000) |
| Lemma L (rev 2): the linear map cannot reduce superposed codes | PROVED; CHECK 0.23–0.34 |
| L′(i): an outer-product store allows one op | PROVED construction; CHECK 1.000 |
| L′(ii): a degree-2 store does not allow nested ops under the linear map | PROVED for (a ± b) + c (Latin-square sums); CHECK (+ 0.38–0.39, × 0.361 / 0.375) |
| L′(iii): one delta erase creates degree-3 terms, and suffices for a nested two-op value | PROVED (algebra); CHECK 1.000 / 1.000 (+ and ×) |
| Thm S1 (rev-2 map) | PROVED |
| S2 | PROVED (exact stall); DERIVED (near stall) |
| G3 ceiling; CERT h5p ≥ 0.30 at the final bin with guards g1 / g2 | DERIVED(A1′). A1′ is empirical: supported by D12 (POST-HOC), not guaranteed by the architecture (erase3 CHECK) |
| Reachability of CERT and DELIVERED by the good class | DATA (D7) + CONJECTURE (probe sample-starvation) + DERIVED (height-flatness under P) |
| Prop O | PROVED; verified in code |
| Thm T | kernel PROVED; mapping DERIVED / CONJECTURE; toy CHECK |
| Prop R (tilt, not removal) | PROVED |
| D7 separator; D9 ceiling; D12 K′1 depth | DATA, POST-HOC |
| "SVR raises P(enter) to ≥ 0.9" | CONJECTURE |

**Novelty: PARTLY KNOWN.** All verified via export.arxiv.org, 2026-10-14.

| relation | work |
|---|---|
| KNOWN: deep supervision / linear-probe heads | DSN 1409.5185; Alain & Bengio 1610.01644 |
| KNOWN: RNN auxiliary losses | Trinh 1803.00144 |
| KNOWN: weak-decoder remedies | Bowman 1511.06349; VLAE 1611.02731; Yang 1702.08139; He 1901.05534; Dieng 1807.04863 (opposite direction) |
| KNOWN: probe selectivity, the parent of G1 and of rev 2's linearity requirement | Hewitt & Liang 1909.03368; Voita & Titov 2003.12298; Belinkov 2102.12452 |
| latent-CoT neighbours | SIM-CoT 2509.20317; Markovian Transformers 2404.18988; Coconut 2412.06769; iCoT 2405.14838 |
| others | Gradient Flossing 2312.17306; Bansal 2202.05826; Anil 2211.09961; Chen 2306.04251; Feng 2305.15408 |
| **new in rev 2**, the reason A1′ must include the write recurrence | Grazzi et al. 2411.12537: linear RNNs with negative eigenvalues (β ∈ (0, 2)) can do state tracking |

- **NEW to our knowledge (Literature to check):**
  - G1 for looped models;
  - a pass-free, *linear*, scale-free, τ-capped value readout with a proved value gap;
  - the height-resolved certificate;
  - the γ-tied schedule.
- No ID failed verification.

---

## 9. Risks (rev 2)

1. **Tilt only (Prop R).**
   - The effect size is uncalibrated: the toy gives 0.62 → 0.83–0.90. Three seeds cannot show a change in P.
   - Escalations: SVR + NOPASS; SVR-WINDOW; M3-REPLACE.
2. **Presence may follow entry rather than cause it (T1 analogue).** Then P-V0 misses and DELIVERED may only come with entry. C8k and the 16k descriptives show this.
3. **Precompute gaming in 2–6k.** 57% of t ≥ 2 targets have h ≤ 2. Detector registered. Fallback: SVR on h ≥ 3 targets only.
4. **Reachability (rev 2).**
   - CERT at the final bin rests on the post-hoc late-t probe values, the sample-starvation argument and height-flatness under P. None of these is height-resolved or pre-entry.
   - If converged SVR cells read CERT < 0.30, the label is ENTERED-UNCERTIFIED, not a failure.
   - A $0 sharpening before launch: a height-resolved step_recover on the 9 converged models at t = 5–7 (Experiments). If h ≥ 5 presence is < 0.30 there, lower the bar to the midpoint between 0.21 and that value **before** launch.
5. **The within-pass delta recurrence** (L′(iii); Grazzi 2411.12537): one erase step already suffices for a nested two-op value (CHECK 1.000). SVR pressure could teach a pass to exploit this.
   - D12 says it does not happen under the current recipe.
   - The registered guards g1 / g2 catch the unclocked form (ONE-PASS-COMPUTE).
   - The clocked form is only screened descriptively by the input-swap diagnostic: if acc_xswap ≈ acc_clean on h ≥ 5, the cell is reported as PRECOMPUTE-GAMED-XSWAP and CERT is not credited.
   - **No assumption-free certificate exists against it.**
6. **Delta-only features (if G-pre selects them)** push values into the delta heads. That matches the good class (D4) but is a stronger requirement than the all-head evidence of D7. G-pre checks it first.
7. **Good-class slowdown or crowding.** Gradshare is pre-stated. w = 1 equals λ; there is no data-driven dose.
8. **Pairing logistics:** driver-matched hosts for cells and C8k.
9. **Budget:** about $10.1 at the high end, at Audit's line.
10. **C6 untouched:** no t > n targets.
