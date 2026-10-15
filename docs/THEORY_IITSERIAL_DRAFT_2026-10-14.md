# IIT-SERIAL: a coordinate-free whole-state interchange loss for seed-stable basin entry (draft rev 1, 2026-10-14)

> **Status: draft, not registered.** Opus math subagent for Theory; Theory reviews. It is the next seed-stability METHOD candidate after §203's SVR did not launch (G-use NOT-USED 9/9). All data are POST-HOC.
> CPU check (3 threads, Idle priority): `theory/dasused/iit_hybrid_fraction.py/.json`, on the generator.
> Companion diagnostic: docs/THEORY_DASUSED_DRAFT_2026-10-14.md (rev 2). Its pilot P0 supplies this method's alignment gate (G-ALIGN).
> Prior art: Literature, docs/LIT_IIT_TRAINING_2026-10-14.md (253ec5bb).

## 0. Summary

1. **Idea.** Train on the counterfactual that converged models already obey. Swap the whole memory after pass τ for that of a structure-matched donor whose first handle h and the root's bare leaf a differ. Continue on the recipient's text. Require the **hybrid** answer y_hyb: the recipient's tree with h set to the donor's value.
   - This is s170 SERIAL as a loss. It needs no subspace and no coordinates, which do not transfer across runs (DAS-USED T3).
2. **Plus an SIIT term** (InterpBench's rule). Hold the recipient's own pass-1 memory and resample **only h's two operands** in the text; the answer must stay y. These are exactly the input features the serial program has consumed by pass 1 (Prop R1, PROVED).
3. **Not met by accuracy (Prop A, PROVED).** y_hyb ≠ y on every used pair (and y′ ≠ y for SIIT), so a model that ignores the swap pays more as its clean accuracy rises.
4. **Penalises the contractive class (Prop C, PROVED).** If the answer is δ-insensitive to the pass-1 state, the per-pair loss is ≥ −log(1 − p_clean(y) + δ). Conversely, a pair loss ≤ ℓ forces sensitivity ≥ e^{−ℓ} − (1 − p_clean(y)).
   - It trains **use**, not presence (Prop U). The target depends on the donor's *value* (content), and the precompute and late-leaf-preload routes are excluded because y_hyb ≠ y_d (Prop P).
5. **Coverage (CHECK).** 31.5% of training items give a kept pair: 0 at n ≤ 2, 0.63 at n = 3 down to 0.23 at n = 8, about 20 per batch of 64.
   - Among eligible items: cf = own 5.0%, cf = donor 3.6%, both excluded; 92.1% kept. SIIT: 32.5% of training items.
6. **Goodhart.**
   - Root precompute and late-leaf preload are penalised (PROVED).
   - Intermediate one-pass precompute is not; the K′ = 1 / pass-1 guards cover it.
   - A swap detector that leaves the clean path contractive is a new route, labelled CLEAN-CONTRACTIVE.
   - The chained one-level-per-pass term is optional, because it fights converged models' natural 2-level pass 1 (SEEDSTAB2 D12).
7. **Design.** 3 seeds on RTX 3090, s185 RMS + `--iit 1.0,0.5,16`, weight w·γ(step) (Prop O′). Bitwise-paired with s185 RMS s1 / s2 / s3 through step 2000. Launch after G-ALIGN from DAS-USED P0.
8. **Cost.** IIT runs a second graph of 32 rows × 8 passes per step after turn-on (+45% central, +80% high). About $9.6 central ($8.6–10.9), $13.5 high. A half-rate variant (every 2nd step, ×2 weight) is about $8.1.
9. **Main risk.** The S2 alignment is partial: the unconditioned SERIAL hybrid at t = 1 is 0.38–0.90 in converged models. If G-ALIGN fails at τ = 1 and τ = 2, the target fights the good class too, and the method should not launch.

---

## 1. Facts

| # | fact | where |
|---|---|---|
| F1 | Identical think input every pass; S is the only carried state. `think(S, x, mask, K, skip, override)`: at pass t, the write is skipped if t ∈ skip, then S ← override[1] if override[0] == t. **So skip = (1..τ), override = (τ, S_τ^d) is a swap with no code change.** | nested170.py 235–252 |
| F2 | Training step: `its = train_items(rng, …)`; main loss and backward; write_grad_norm; clip at 1.0. | nested170.py 1039, 1061, 1064, 1067 |
| F3 | Main inject + aux path: passes 1..K with per-pass aux CE, decoder answer CE, la + λ·aux. | nested170.py 687–735 |
| F4 | γ(step) = 0 for step ≤ 2000, linear to 1 at 6000. Write input u = (1−γ)LN_6(h6) + γ LN_w(h_w); h6 is captured before the read. | nested170.py 1035; deltamem.py 187, 278–280 |
| F5 | `inject_answers` is `@torch.no_grad()` and must not be reused in a loss. Insert new closures after loss_items' return (735) and before `def mem_check` (737), not above a decorated def. | nested170.py 735–753 |
| F6 | mem_check includes the CSR path at its largest shape (`cur_step = 10**9 if CSR`); the IIT path must be added the same way. | nested170.py 737–750 |
| F7 | serial_items: the donor gets h's operands re-drawn (v′ ≠ v) and the root's bare operand changed; hybrid = the recipient with h → v′; kept iff hybrid ∉ {recipient, donor}. | nested170.py 1274– |
| D1 | **Unconditioned SERIAL, t = 1 / 2 (n = 8 pool).** Converged hybrid 0.38–0.90 / 0.53–0.96, recipient share 0.05–0.29 / 0.02–0.16. Failed hybrid 0.06–0.08 / 0.07–0.12. Causal within converged models; the class contrast is pending capability conditioning (DAS-USED rev 2 §2.5). | eval.json s170.serial |
| D2 | Failed clean accuracy at K′ = K is lower but not at chance (rms s3 n 3–8: .96 / .84 / .65 / .49 / .36 / .23). At K′ 0 / 1 it is near chance from n = 3. path_indep acc_from_donor: failed 0.22–0.64, converged 0.10–0.14. | eval.json; DAS-USED D3 / D4 |
| D3 | s185 RMS: s1 / s2 entered at 19k, s3 never (final 1.03, clip rate 1.0). Train time 42,491 / 42,576 / 35,432 s per 48k. 3090 hosts at $0.179–0.228/h. Peak memory 18.1 GiB at micro 1 (mem budget 21). Drivers: s1 / s2 570.172.08, s3 595.84. | THEORY.md §203; eval.json; mem_check.json |
| D4 | K′ = 1 (one pass then the decoder) reaches height 2 at 0.54–0.77 and height ≥ 3 at chance in trained models. | SEEDSTAB2 D12 |
| CHECK | Defined-pair fractions on the generator (§3.4). | theory/dasused/iit_hybrid_fraction.json |

---

## 2. The loss

**2.1 Pairing (per training step, from the step's own items; no training-rng draw).**
- **Eligible recipient x:**
  - (e1) the root has a bare-number child a (the late leaf);
  - (e2) the first leftmost-ready handle h is not the root;
  - (e3) height(root) ≥ 2τ + 1.
- For τ = 1, (e3) ⟺ n ≥ 3 given (e1) (CHECK: height ≥ 3 on 100% of eligible items).
- The bound means that even a 2-levels-per-pass schedule (D4) cannot have consumed a by pass τ.
- **IIT donor x_d** (own rng, seed + 20500):
  - h's operands re-drawn by the generator's conditional (uniform among the pairs with o(·, ·) = v′, v′ ≠ v_1 uniform);
  - a → a′ ≠ a;
  - everything else identical, so the rendering length matches and the masks align;
  - up to 10 redraws on a division by 0.
- **Targets:**
  - y = F(x);
  - **y_hyb** = F(x with h's operands := the donor's) = F_1(x, v_1(x_d));
  - y_d = F(x_d).
- **Exclusion rule:** keep the pair iff all three are defined and **y_hyb ∉ {y, y_d}**.
- **SIIT text x′** (same rng): x with h's operands re-drawn to v″ ≠ v_1, a kept. Keep iff y(x′) is defined and **y(x′) ≠ y**.
- **B_I = min(16, #kept)** recipients per step, taken from the step's batch in order. The same 16 serve IIT and SIIT.

**2.2 The two terms** (the swap site is the whole state S_τ, all heads, den's z included; τ = 1 by default).
- **IIT:**
  - S_τ^d = think(0, x_d, m_d, τ), with grad;
  - S_K = think(0, x, m_x, K, skip = (1..τ), override = (τ, S_τ^d));
  - **L_IIT = CE(dec_logits(S_K, x), y_hyb)**, full-vocabulary CE at " =" as the answer CE.
- **SIIT:**
  - S_τ = think(0, x, m_x, τ), with grad;
  - S_K′ = think(0, x′, m_x, K, skip = (1..τ), override = (τ, S_τ));
  - **L_SIIT = CE(dec_logits(S_K′, x′), y)**.
- **Total:** L = answer CE + λ_aux·aux + **γ(step)·(w_I·L_IIT + w_S·L_SIIT)**, with w_I = 1.0 and w_S = 0.5.
- **Choices (design, not derived):**
  - w_I = 1 matches SVR's w and the answer CE's scale (the same head, single-token CE).
  - w_S = 0.5 because SIIT overlaps IIT's penalty on re-reading h (§4 R1) and has an easier target (the own answer).
- **τ = 2 variant (only if G-ALIGN prefers it):** τ = 2 on kept items with height(root) ≥ 5 (10.9% of training items, about 7 per batch), τ = 1 otherwise. The two τ go in separate sub-forwards.

**2.3 Schedule: weight ∝ γ(step) (Prop O′).** Zero through step 2000, and the IIT code does not execute while the weight is 0. So cells are bitwise-paired with their controls through 2000.

**2.4 Placement in the step (F2, F5).**
- After the main `(lk / micro).backward()` (1061) and before `write_grad_norm` (1064): if γ(step) > 0, build pairs, compute both terms as **one separate graph** (32 rows: 16 IIT, 16 SIIT), and call `(γ·(w_I L_IIT + w_S L_SIIT)).backward()`. Gradients accumulate, then the existing clip at 1.0 applies to the sum.
- The peak memory is the max of the two graphs, not their sum (main 18.1 GiB; the IIT graph is about 5 GiB without checkpointing).
- No model change is needed. model.pt is unchanged in format.

---

## 3. Derivation

**Notation.**
- Q_x(S) is the answer distribution after passes τ+1..K on x and the decoder on x, starting from S after pass τ.
- Clean: p_c(·) = Q_x(S_τ(x)). Swapped: p_s(·) = Q_x(S_τ(x_d)).

**3.1 Not met by better task accuracy alone.**

**Prop A (PROVED).** On every kept IIT pair y_hyb ≠ y, and on every kept SIIT pair y(x′) ≠ y (the exclusion rule).
- **(i)** A model whose output ignores the swap (p_s = p_c) pays L_IIT = −log p_c(y_hyb) ≥ −log(1 − p_c(y)). This → ∞ as its clean accuracy p_c(y) → 1.
- **(ii)** A model whose output is fixed by the text (ignores the memory) answers x′ as y(x′) and pays L_SIIT ≥ −log(1 − p_clean,x′(y(x′))). This → ∞ likewise.
- So **raising clean accuracy without using the carried state strictly raises both terms.** ∎

**3.2 Precompute and preload are excluded by the pair choice (Literature's rule (b)).**

**Prop P (PROVED).**
- (i) If S_τ(x_d) carries the donor's **final answer** (root precomputed by pass τ) and the decoder outputs it, the swapped output is y_d ≠ y_hyb.
- (ii) If S_τ carries the donor's **late leaf a′** (preloaded) and the model uses it over the text, the root is evaluated with a′. The output is F_1(x_d) = y_d ≠ y_hyb.
- Both are excluded because the post-swap operation (the root, which consumes a) **differs** between donor and recipient. The pairs with y_hyb = y_d (3.6% of defined eligible pairs) are exactly those where it does not, and are dropped. ∎

**3.3 The contractive class pays; use, not presence, is trained.**

**Prop C (PROVED).**
- Let a model satisfy TV(Q_x(S), Q_x(S′)) ≤ δ for all reachable S, S′. This is contraction / path independence.
- Then on a kept pair p_s(y_hyb) ≤ p_c(y_hyb) + δ ≤ 1 − p_c(y) + δ, so the pair loss is ≥ −log(1 − p_c(y) + δ).
- **Converse (the sensitivity a low loss forces):** a pair loss ≤ ℓ implies TV(Q_x(S_τ(x_d)), Q_x(S_τ(x))) ≥ e^{−ℓ} − (1 − p_c(y)).
- So a model that is both accurate and meets the term cannot be δ-contractive with δ < e^{−ℓ} − ε. ∎

**Prop U (PROVED).** L_IIT depends on θ only through the *interventional* output Q_x(S_τ(x_d)).
- A model that keeps v_1 decodable in S_τ but whose Q_x is insensitive to it (presence without use, DAS-USED T1) satisfies Prop C's bound restricted to that content, and pays the same lower bound.
- Unlike a probe loss, the term is minimised only by downstream use (DAS-USED T2).

**Content (S3, PROVED).**
- Suppose the downstream depends on S_τ only through a statistic ψ independent of v_1 given x (norm, displacement, a clock). Then donors with equal ψ and different v_1^d get the same output, so at most one of their different targets is hit.
- So no content-free state change can meet the term. This is the property CSR lacked (SEEDSTAB2 G2).

**Value gap and alignment (S1 / S2; DATA + gate).**
- **Unconditioned (D1, argmax rates as a proxy for p_s):** converged models' current per-pair loss is about −log(0.38–0.90) = 0.1–1.0 nats at t = 1, against about −log(0.07) = 2.7 for failed models.
- **S2 is partial at τ = 1.** Converged models recompute h from the text on 5–29% of pairs.
- **G-ALIGN (pre-launch, $0, from DAS-USED P0 ii):**
  - capability-conditioned hybrid_ccc at the chosen τ (n 3–8 pooled, eligible strata) is ≥ 0.6 in both pilot converged models (rms s1, window s1), and ≤ 0.3 in both pilot failed models (rms s3, hold s1);
  - try τ = 1 first, then the τ = 2 variant;
  - if neither passes, **do not launch**.

**Prop O′ (PROVED; the γ weighting).**
- At γ = 0 the writes of passes τ+1..K are functions of x only (u = LN_6(h6), with h6 before the read; F4). So S_K = Φ_x(S_τ^d), with Φ_x input-only decay / erase / writes.
- The only way the answer can depend on the donor state is then the decoder's read of the transformed donor content. The IIT gradient would train "carry v_1, the decoder computes the rest", a C-type reader solution.
- For γ ∈ (0, 1) the S-dependent part of the writes carries γ. Hence the weight γ(step), as SVR's.

**Tilt, not removal (DERIVED, as SEEDSTAB2 Prop R).** The answer CE and the pass-token aux still credit C up to reader depth. IIT adds a term C cannot reduce (Prop C), so P(enter) rises with w, but there is no proof that it reaches 1.

**3.4 Coverage (CHECK: `iit_hybrid_fraction.json`; 100,000 `train_items` draws with the run's exclusion set, so n = 1 is nearly all excluded).**

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | all |
|---|---|---|---|---|---|---|---|---|---|
| eligible (e1–e3, τ = 1) | 0 | 0 | .675 | .496 | .388 | .331 | .284 | .251 | .342 |
| **IIT kept** | 0 | 0 | **.629** | .459 | .356 | .302 | .260 | .226 | **.315** |
| SIIT kept | 0 | 0 | .646 | .473 | .367 | .312 | .269 | .235 | .325 |
| IIT kept with height ≥ 5 (τ = 2 rule) | 0 | 0 | 0 | 0 | .116 | .198 | .231 | .223 | .109 |

- **Among eligible items:** defined 100% (≤ 10 redraws); y_hyb = y 5.0%; y_hyb = y_d 3.6%; **y_hyb ∉ {y, y_d} 92.1%**. About 20 kept pairs per batch of 64; P(fewer than 16) ≈ 0.10 under a binomial, and then B_I = the available count.
- **Bias:** (e1) makes kept items shallower-weighted (0.63 at n = 3, 0.23 at n = 8). A relaxed late leaf (the leaf child of the highest reduction outside h's subtree, with that node's height ≥ 3) would raise deep coverage. It is not in the default.

---

## 4. Goodhart

**R1 (the SIIT resample set; PROVED).**
- **Claim.** Under the serial program at τ = 1, the input features that may be resampled with the memory held, **without changing the counterfactual**, are **exactly h's two operand leaves.**
- **Why they can.** They are consumed only by h. h's reduction is complete by pass 1 under the CoT program (one leftmost reduction per pass), and under any schedule that does at least that. So the program reads h's value from memory and never re-reads them.
- **Why nothing else can.**
  - Every other leaf is an operand of a reduction r ≥ 2 that the CoT program performs after pass 1. Resampling it changes the program's output, so the target would not be y.
  - The structure tokens (operators, brackets) define the remaining operations.
  - Other height-1 handles may be done by pass 1 under a faster schedule, but the CoT program does not guarantee it, so they are excluded (conservative).
- **Where the memory alone decides:** h's value and everything computed from it. **Where the input legitimately decides:** every operation above h, and every other leaf.

**G1 One-pass precompute written into the state.**
- (a) **Root / answer precompute** is penalised (Prop P(i)).
- (b) **Intermediate precompute** (h's ancestors below the root computed in pass 1 from the text) is *consistent* with y_hyb, so it is not penalised.
  - It is bounded empirically by A1′ (D4: height ≥ 3 at chance in one pass).
  - **Guard (registered, as §203):** g1, K′ = 1 pooled accuracy at n 3–8 ≤ 0.20. A model holding all but the root after pass 1 lets the decoder finish one op, which g1 catches. g2: the pass-1 deep probe for v_t with h ≥ 4 ≤ 0.21. Either failing gives **ONE-PASS-COMPUTE**.
- (c) **Chained interchange (optional term, not default).**
  - The donor also differs in the leaf sibling s of h's parent p.
  - It needs a τ = 1 target F_1(x, v_1^d), with s read from the text, *and* a τ = 2 target F_p(x, op(v_1^d, s_d)).
  - This penalises 2-level pass-1 precompute (PROVED: a model that has op(v_1^d, s_d) in S_1 outputs the τ = 2 target at τ = 1, and the two differ).
  - **But it forces exactly one level per pass, which converged models violate (D4: 2-level pass 1 at 0.54–0.77).** It is not S2-aligned, so it is kept out of the default.

**G2 Swap detection (a new route).**
- The state may carry a fingerprint of its input, such as operand copies. A model could then learn: *on a state / text mismatch, use the memory (meets IIT and SIIT); on clean inputs, recompute from the text (contractive).*
- The task loss does not separate the two modes on items both solve. The contractive mode fails deep items (D2), so the incentive to keep it is weak, but not zero on shallow items.
- No undetectable causal test exists for a whole-state swap. So this is screened by the clean-path order parameters on the final model:
  - path_indep acc_from_donor ≤ 0.2 and delta_cos ≤ 0.65 (the converged range is 0.10–0.14 / 0.44–0.63);
  - DAS-USED's untrained families: node-t donors at t = 2 / 3, and τ = 3 / 4 on height ≥ 7 items;
  - a linear swapped-vs-clean probe on the pass-τ+1 reads (descriptive).
- **CLEAN-CONTRACTIVE** = MET while path_indep acc_from_donor > 0.2 or delta_cos > 0.65.

**G3 Schedule lock-in.** The target assumes h is carried and a is read after τ. (e3) keeps this compatible with ≤ 2 levels per pass. G-ALIGN checks that the good class meets it.

**G4 Carry h, deep reader.**
- A model that carries only v_1 and lets the decoder do the rest meets IIT on items within the reader's depth (SEEDSTAB2 G1: d_R ≈ 3–4).
- On deep items it cannot, because the remaining height exceeds the reader. So passes τ+1..K must iterate from the carried value.
- Descriptive check: acc_by_n_kprime at K′ = 2 / 4 by n.

**G5 Over-following.** G may learn to trust the state over the text everywhere. The late leaf must still come from the text (Prop P(ii)), and SIIT holds the memory but keeps a. Noise evals are descriptive.

**G6 Off-manifold training.** Swapped inputs pair a genuine state with a mismatched text. They add about 26% extra row-passes per step (32 × 9 against 64 × 17).

---

## 5. Gradient flow (from the code) and per-term assertions

**Paths.**
- **L_IIT:** gradients reach
  - (a) passes τ+1..K on x (q / k / v / gate / W_o and the trunk through the reads and writes);
  - (b) the decoder;
  - (c) the donor's passes 1..τ through S_τ^d (no detach, as the aux path's full BPTT; SEEDSTAB2 F3).
- **L_SIIT:** the same, with S_τ = the recipient's own first pass.
- **Not touched:** `inject_answers` (no_grad, F5), the aux decoders and EMA buffers.

**Smoke assertions** (scratch/gradflow_check.py style, per the gradient-flow rule; each term backward-ed alone on a fixed batch):
1. ‖∂L_IIT/∂θ‖ > 0 for mem.qkv.weight (q rows, and the k / v rows of the delta heads), mem.gate.weight, mem.wo.weight, mem.ln_w.weight, and block 6 and block 11 attn / mlp weights. ‖∂L_IIT/∂S_τ^d‖ > 0 (retain_grad).
2. The same for L_SIIT, with ‖∂L_SIIT/∂S_τ‖ > 0.
3. At γ = 0 the IIT closure is not called (call counter = 0, steps 1–2000). Steps 1999–2001 run with and without the flag are bitwise equal through 2000.
4. The main loss's gradient is unchanged by the IIT graph: compare p.grad after the main backward with the flag on and off at a step with γ > 0, before the IIT backward.
5. No @torch.no_grad on the closure; check its decorators via ast, per the decorated-def rule.
6. The plain path without `--iit` is bitwise identical to the current code (a 2-step smoke).
7. mem_check includes the IIT graph at the largest shape (longest n = 8 rows, B_I = 16).

---

## 6. C1 preserved (PROVED)

- The think input is identical every pass. The memory remains the only cross-pass channel. No pass token is added.
- The swap is a training-time operation on the state between passes. At test time the model is unchanged, and model.pt carries no new parameters.
- All C1 tests (K′ sweeps, skip_pass, THINK-NECESSARY) run as before.

---

## 7. Design, readings, predictions

**Cells (RTX 3090; 48k).** `s2xx_iit_s1 / s2 / s3`: the s185 RMS recipe exactly + `--iit 1.0,0.5,16` + `--keep_ckpt_steps 4000,8000,12000,16000`.
- Each is paired with s185_rms_s{seed} on that control's driver.
- **PAIRED** iff log records 100–2000 are bitwise equal on loss, answer_ce and aux_ce.

**Pre-launch gates ($0 unless stated).**
- **G-ALIGN** (§3.3).
- **G-HYB:** y_hyb from the tree equals eval_tokens on the rendered hybrid, on 10k pairs.
- **G-GRAD** (§5, 1–5); **G-PLAIN** (§5.6).
- **G-MEM:** ≤ 21 GiB with micro 1.
- **G-TIME:** `--time_steps 50` on a 3090, about $0.05; it replaces §8's estimate.
- Audit's diff review; a running cost guard.

**Eval fields (final model; also at the 4k / 8k / 12k / 16k checkpoints, descriptive).**
- **iit_eval:** on TE eligible items n 3–8, conditioned on recipient, donor and hybrid solved clean (DAS-USED §2.5): {hybrid_ccc, own, donor, other, n_ccc, by_n} at the trained τ and at untrained τ ∈ {2, 3} (height rule).
- **siit_eval:** {keep_ccc, n_ccc}.
- **untrained-family:** the DAS-USED whole-state matched swap at t = 2 / 3.
- **Clean order parameters:** path_indep, delta_cos.
- **Guards:** g1 and g2.
- **Training:** in-training iit_ce / siit_ce per 100 steps.

**Per-seed labels (final model).**
- **MET** := iit_eval hybrid_ccc ≥ 0.8 at the trained τ.
- **ENTERED** := entry(0.5) ≤ 24k and final 1k bin < 0.1 (§203 P-V2).
- **ENTERED-AND-MET / MET-NOT-ENTERED / ENTERED-NOT-MET / NEITHER.**
- **HARMFUL:** s1 or s2 never enters; their controls entered at 19k.
- **ONE-PASS-COMPUTE:** g1 > 0.20 or g2 > 0.21.
- **CLEAN-CONTRACTIVE:** MET, with path_indep acc_from_donor > 0.2 or delta_cos > 0.65.

**Arm.**
- **SUPPORTED** = 3/3 ENTERED-AND-MET, with no HARMFUL, ONE-PASS-COMPUTE or CLEAN-CONTRACTIVE.
- **FAILED** = any HARMFUL, any MET-NOT-ENTERED, or 0/3 MET.
- Otherwise **MIXED**.

**Predictions (CONJECTURE).**

| # | prediction | P |
|---|---|---|
| P-I1 | MET in 3/3 | 0.6 |
| P-I2 | s3 ENTERED (no-effect base about 0.6 once the trajectory is perturbed, §203; one entry is weak evidence, so the labels carry the reading) | 0.75 |
| P-I3 | no HARMFUL | 0.85 |
| P-I4 | ONE-PASS-COMPUTE in ≥ 1 seed | 0.15 |
| P-I5 | CLEAN-CONTRACTIVE in ≥ 1 seed | 0.15 |
| P-I6 | in MET seeds, in-training iit_ce < 0.5 nats before entry(0.5): use precedes entry | 0.5 |
| P-I7 | SUPPORTED | 0.35 |

**Falsifiers.**
- **MET-NOT-ENTERED** in any seed: using the carried value is not sufficient for entry, which falsifies the account.
- **0/3 MET at 48k:** the target is not attainable from these basins at w = 1.
- **HARMFUL.**
- **Claim limits** as §203: P(3/3 | p = 2/3) = 0.30. A P(enter) claim needs a second-stage panel (the user's call).

---

## 8. Cost

**Extra work per step after turn-on (steps 2001–48000, 95.8% of steps).** With τ = 1:
- IIT: 1 donor pass + 7 recipient passes + 1 decoder, on 16 rows;
- SIIT: 1 + 7 + 1, on 16 rows;
- batched as one 32-row graph of 8 think passes + 1 decoder, forward and backward.
- The main step is 64 rows × (8 think passes + 8 aux decoders + 1 decoder).
- The delta token loop is launch-bound, so extra sequential passes cost more than their row share. Estimate **+45% central, +80% high**. G-TIME replaces this.

| variant | hours (s1 / s2 / s3) | total h | $ central ($0.20/h; range $0.179–0.228) | $ high (+80%, $0.228/h) |
|---|---|---|---|---|
| default (every step) | 16.9 / 16.9 / 14.1 | 47.9 | **9.6** (8.6–10.9) | **13.5** |
| half-rate (every 2nd step, ×2 weight) | 14.4 / 14.4 / 12.0 | 40.7 | 8.1 | 10.5 |

- Eval-only fields are local ($0). G-TIME is about $0.05.

---

## 9. Claim status and prior art

| claim | status |
|---|---|
| A: y_hyb ≠ y (and y′ ≠ y) on every used pair; ignoring the swap or the memory costs more as accuracy rises | PROVED |
| P: root precompute and late-leaf preload are penalised (post-swap operations differ) | PROVED |
| C: per-pair loss ≥ −log(1 − p_c(y) + δ) for δ-contractive models; low loss forces sensitivity | PROVED |
| U: trains use, not presence | PROVED |
| S3 content: no content-free statistic meets it | PROVED |
| O′: at γ = 0 only a reader solution is trained, hence weight ∝ γ | PROVED (code F4) |
| R1: SIIT resample set = h's two operands, exactly | PROVED (under the CoT program) |
| Chained term forces one level per pass and penalises 2-level pass-1 precompute | PROVED; its S2 misalignment is DATA (D4) |
| Coverage 31.5% of training items; 92.1% of eligible; cf = own 5.0%, cf = donor 3.6% | CHECK (generator) |
| S2 alignment at τ = 1 or 2 | DATA (D1, unconditioned) + gate (G-ALIGN) |
| Tilt, not removal | DERIVED |
| G1(b) intermediate precompute bounded by A1′ | DATA (D4); guarded by g1 / g2 |
| G2 detector route | CONJECTURE (screened) |
| P-I1–P-I7 | CONJECTURE |

**Prior art (Literature 253ec5bb; IDs verified).**
- IIT, Geiger et al. 2112.00826 (ICML 2022): verified. The zero-loss causal-abstraction guarantee is a theorem.
- Known follow-ups: causal distillation 2112.02505, type-level IIT 2212.09897, Strict IIT 2407.14494 (InterpBench).
- **Claim line (Literature's):** "IIT is established (including a recurrent encoder); not, to our knowledge, applied to looped / weight-tied latent reasoners or with the whole recurrent state as the aligned variable, as a training objective for causal use; we adopt SIIT's constraint on the non-memory path."
- **IIT for seed reliability of basin entry would be a new aim.**
- Avoid: "first interchange training", "first causal-use objective".

---

## 10. Risks

1. **S2 only partial** (D1: 0.38–0.90 at t = 1 unconditioned). G-ALIGN is the gate. **This is the main risk.** If the good class does not meet the target, IIT fights it, as CSR's floor did.
2. **The detector route (G2):** the clean path stays contractive while swaps are handled. Screened by CLEAN-CONTRACTIVE, not prevented.
3. **Coverage skew** toward n = 3–4 (e1). Deep items, where use matters most, are 23% kept at n = 8.
4. **Cost** is set by the launch-bound loop. G-TIME may put it at the high end ($13.5); the half-rate variant is the fallback.
5. **Seed lottery:** one s3 entry is weak evidence (P(enter | no effect) ≈ 0.6 once perturbed). The mechanism labels carry the reading.
6. **One-pass precompute of intermediates** (G1b) is not penalised by the loss itself. It relies on g1 / g2.
