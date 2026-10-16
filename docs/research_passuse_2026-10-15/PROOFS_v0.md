# PROOFS v0: rigour pass on the VERSION-BANK + READY-RECODE claims

*Opus proof subagent for Theory, 2026-10-15 22:53 EDT. Repo read-only. CPU checks are in `scratchpad/proofs/`: `depth_dag.py` (longest paths in the unrolled DAG), `g1_check.py` (Gate G1 on a random 12-layer fp64 model) and `nce_prior.py` (Prop 4's prior artefact on the §170 generator). New references were checked on arXiv abs pages: 2002.10689, 1011.0686, 1511.05101, 1905.06922, 1807.03748, 2502.21074, 2412.06769.*

## Verdict table

| # | Claim (source) | Verdict | Fix |
|---|---|---|---|
| 1 | L1: depth(V_t) = P + t·d; output after K passes ≤ P + K·d + c (A §4(i)) | **[PROVED]** for CARRY and PURE, including the coda's bank reads, decoder tokens and K′ = 0 (= P + c) | Replace "masked weight-tied transformer" with "layered block network with frozen positions". Under CARRY, the carry alone gives the same depth, so "depth builds up through the attention memory itself" holds for PURE only |
| 2 | L1-b: same-layer (KV) bank reads add no depth (A) | **[PROVED]** when the pass input has no feedback (PURE-type) | Add "no input feedback". With a CARRY input, depth P + t·d comes back through the carry |
| 3 | CODI / Coconut: all depth growth runs through the fed-back vector (A §1.4) | **[PROVED]** (Thm 2.2): latent j's output has depth j(L + π) + L; a path with f feedback edges has length ≤ (f+1)L + fπ | Also say what KV access *does* add: width, fan-in, parallel aggregation, and access to every earlier fed-back vector |
| 4 | "CODI's serial channel is RNN-like" (A) | **Overclaim** | The channel is narrow (one vector per hop) but append-only and attended at every layer. Say "narrow", not "RNN-like" |
| 5 | L1-c: passes reading no V_{s≥1} compute a one-pass-class function (A) | **[PROVED after restating]** | Define reads interventionally, *including the carry*. An idle CARRY pass reads V_{t−1}. The "one-pass class" is depth P + d + c at width (K+1)T + L_dec, not DIRECT's class. General form: depth ≤ P + d·L_eff + c |
| 6 | §4(iii): zero L_rec + one-pass bound r₁ ⇒ a chain whose length grows with n (A) | **[FALSE as stated]**: r₁ only forces chain length ≥ 2 (counterexample M2) | **[PROVED UNDER CAP-aff]**: L_eff(t; x) ≥ 1 + ⌈(h_t(x) − c₀)/c⌉, where h is the reduced tree height. Asymptotically unbounded **[PROVED UNDER TC⁰ ≠ NC¹]**, with no finite n |
| 7 | R1 creates no G8 incentive | **[PROVED]** | — |
| 8 | R1 and count-keyed schedules (G11) | No separate optimum on-distribution. NOPASS hides no clock: the age bias on V_0 is an absolute pass index, the bank size reveals t, and so does the content | Add a **STALL** reading. SKIP-DELAY as specified cannot detect a bank-size clock |
| 9 | B Prop H (hitting set) | **[PROVED]** on the unconditioned row distribution, once R(F) is redefined as a factorisation | On *kept* rows it needs A-KEEP (the keep filter breaks A-IND) |
| 10 | B Prop D (memory mediation) | **[PROVED]** | In VB it implies re-coding only if the input entry V_0^(p) is *replaced* each pass, not appended |
| 11 | B Prop B (snapshot bypass = 0) | **[PROVED]** (by construction) | The coincidence rate is ≈ 1/10, not 1/11, under the exclusion |
| 12 | B Prop M(a): timed release gives R > 0 for all K, j | **[FALSE pointwise]**: R < 0 on items whose precomputed logits misrank y | **[PROVED]** in expectation (proper-scoring argument). Bounded logits cap the gain at 2B + log 11 but do not remove the incentive |
| 13 | B Prop M(b): detached = reweighted CE | **[PROVED]** with per-item weights w = φ′(log p_K − log p_{K−j}) (= 1 for φ = id) | State the weights. "Exact" holds per step, with the weights at the current θ |
| 14 | C Prop 4 (value-only InfoNCE bound) | **[PROVED UNDER]** single-variable, exchangeable candidates | C's candidate set mixes two variables, so a v_t-blind student's floor is log 3, not log 4. The prior artefact is 0.003 nats (negligible) |
| 15 | C Lemma 3 (TOM ⇒ Markov sufficiency, in loss units) | **[FALSE as written: vacuous]**: S_t is a function of x, so the conditional MI is 0 for every model | **[PROVED UNDER no-conflict + CAP]** in 𝒱-information: masked-pass excess ≤ ε/p_m |
| 16 | R3 teacher-forced versions are a valid substitute | **Not well-posed as written** | Needs index alignment (W1), teacher correctness (W2) and read invariance (W3). The candidate fixes only the clock mismatch. Proposal: the **minimal-correction teacher** (§8) |
| 17 | G1: CARRY, K = 1, β = −∞, g = 1 is exactly GPT-2 | **[PROVED]**: fp64 max abs Δ = 3.6e-15 to 1.2e-13 on random nets | Readout at the per-item last slot; coda causal; the bank masked in core *and* coda |
| 18 | G1 at β = −5: "≈ 1.3% bank mass, near-exact" (A §1.2) | **[FALSE as a bound]** | 1.3% is the equal-logit mid-query value: 0.67% at " =", 15% at slot 0. A 5-nat raw-logit advantage cancels β. Use the exact per-head identity, a ratio test, and a per-head calibrated β |

---

## 0. Model and definitions

**The unrolled DAG 𝒢.** Nodes are residual vectors. Sources (depth 0) are token / position embeddings and parameters, which includes PURE's q + wpe and GSM8K's scratch inits.
- A *block node* (layer λ, position j) has edges from its own position's previous residual and from every key/value vector in the support of its attention mask. A finite bias is an edge; −∞ is not.
- Normalisations have depth max(parents) and cost nothing: ln, CARRY's N(V)[j] = g·V[j]·rms(V_0[j])/rms(V[j]), and the head R.
- δ(v) is the longest source-to-v path, counted in blocks. Counting attention and MLP as separate sublayers doubles every number below.

**Notation.** P = 6, d = 4, c = 2 (VB-12); T slots; L_dec decoder tokens. D*_t := max_{s≤t} δ(V_s).

---

## 1. Lemma L1 (critical path) [PROVED]

**Statement.** In VB with all bank biases finite:
- δ(V_t[j]) = P + t·d for every j and t ≥ 0, under both CARRY and PURE;
- the coda output after pass t has depth P + t·d + c, which is P + c at K′ = 0.

**Proof.**
- *Prelude.* Each node has its own position's previous node as a parent, and attention is causal. So δ(h^{(ℓ)}[j]) = ℓ for every j, and δ(V_0) = P uniformly.
- *Pass t, PURE.* δ(U_t^{(0)}) = 0. Every core layer reads all of 𝔅_{t−1}, so δ(U_t^{(1)}) = 1 + max(0, D*_{t−1}). For ℓ ≥ 2, δ(U_t^{(ℓ)}) = 1 + max(δ(U_t^{(ℓ−1)}), D*_{t−1}) = ℓ + D*_{t−1}. Hence δ(V_t) = d + D*_{t−1}, and by induction (D* is increasing) δ(V_t) = P + t·d.
- *Pass t, CARRY.* δ(U_t^{(0)}) = max(δ(V_{t−1}), δ(V_0)) = P + (t−1)d = D*_{t−1}. The same recursion follows.
- *Coda after pass t.*
  - V_t's positions have depth P + t·d.
  - The bank V_0..V_{t−1} has depth ≤ P + (t−1)d.
  - Decoder tokens are prelude-encoded among themselves, so they are x-independent nodes of depth P.
  - Each coda layer adds 1 to the maximum, giving P + t·d + c. At t = 0 the bank is empty and the output is P + c = 8.
- The nested readout is at the " =" slot, which is the last causal position of V_t, so decoder tokens never reach it. ∎

`depth_dag.py` confirms every formula for (P, d, c) ∈ {(6,4,2), (4,2,2), (3,1,1)} and K = 0..6.

**Restatements.**
1. **The class is not a standard transformer.** "A masked, weight-tied transformer of depth P + Kd + c" is accurate only as a *layered block network*: unrolled layer λ applies one fixed block to a fixed, input-independent subset of positions (pass s's slots, frozen afterwards) and the identity elsewhere. That is not a standard transformer layer. However, the per-layer TC⁰ simulation (Merrill–Sabharwal 2207.00729) goes through layer by layer, because the subset and the mask are input-independent, and N's division and square root are TC⁰ at log precision. C1-asym needs exactly this.
2. **CARRY gets its depth from the carry.** Under CARRY, depth P + t·d is reached through the carry chain alone, even with the bank fully masked (`depth_dag.py`: CARRY with an empty read set gives 6, 10, 14, … exactly). So A's summary sentence ("depth builds up through the attention memory itself") is true **for PURE only**. Under CARRY the memory is a depth-redundant channel, which is the root of G10.

---

## 2. L1-b, and what it says about CODI

**Lemma 2.1 (graded DAG) [PROVED].** If every edge u→v satisfies γ(v) ≥ γ(u) + 1 for some grade function γ ≥ 0 on the nodes, then δ(v) ≤ γ(v).
- *Proof:* along any path the grade rises by at least 1 per edge. ∎

**L1-b [PROVED].**
- *Setting:* the pass input has depth 0 (PURE-type), and core layer ℓ of pass t reads, as keys/values, only earlier passes' layer-(ℓ−1) states and the problem's own layer-(P+ℓ−1) states (the KV-cache pattern).
- Grade U_s^{(ℓ)} by P + ℓ. Every edge then raises the grade by exactly 1, so δ(V_t) ≤ P + d for all t, with equality.
- A CARRY edge V_{t−1} → U_t^{(0)} *lowers* the grade from P + d to P. That breaks the grading, and depth P + t·d returns (`depth_dag.py`).
- So the correct statement is: *same-layer reads add no depth; only grade-lowering (feedback) edges do.*

**Theorem 2.2 (CODI / Coconut) [PROVED].**

*Model.*
- An L-layer model. The question tokens have depth ℓ at layer ℓ.
- Latent j's input is h_0^j = proj(h_L^{j−1}). Latent 1's input is the last question state. proj has depth π: π = 0 for Coconut. For CODI, a two-layer MLP followed by LN (checked in the 2502.21074 HTML), so π ≈ 1 sublayer.
- Latent j at layer ℓ attends to the layer-(ℓ−1) states of the question, of latents j′ < j, and of itself.

*Then:*
- (i) δ(h_ℓ^j) = j(L + π) + ℓ. The output depths (j+1)L for j = 1..6 match the DAG computation.
- (ii) A path with f feedback edges has length ≤ (f+1)L + fπ. *Proof:* grade = ℓ; KV and vertical edges raise it by 1, and each feedback edge resets it from L to 0.
- (iii) The longest path to h_L^j is unique. It is the vertical stack of every latent, joined by every feedback edge 1→2→…→j. *Proof:* to leave latent j′ through its feedback edge, a path must reach h_L^{j′}; KV edges only go to *later* latents and cannot return; so between feedback edges the path stays in one latent's column. ∎

So "all depth growth runs through the fed-back chain" is a theorem. Each hop carries one D-vector at one position.

**What same-layer KV access does add (so the draft does not overclaim):**
- (a) *Width.* At grade ℓ, the computation spans question + j latent positions.
- (b) *Unbounded fan-in in one layer.* Sums, majorities and lookups across all earlier steps.
- (c) *Parallel aggregation.* Many shallow results computed at different steps can be combined.
- (d) *Cumulative deep content.* Latent j can read every earlier fed-back vector's processed states. The deep information available is j vectors, not 1. Only the *latest increment* of depth must cross the newest feedback edge.

None of these changes the TC⁰ class at fixed depth, but at finite n width can stand in for depth (G9 applies to CODI too).

**"RNN-like" verdict.** Fair about per-hop width only (a chain of single vectors). Unfair about persistence: nothing is overwritten, and every past vector stays attended at every layer, so CODI's channel is an append-only memory of continuous tokens, pass-global in the user's sense.

The defensible contrasts with VB are:
- per-hop width: 1 vector vs T (+ M_s);
- what is stored: in VB, stored items are *outputs read raw at every core layer* (the Feedback Transformer pattern), whereas CODI's enter at layer 0 of the next latent only;
- per-hop depth: L = 12 vs d = 4.

**Suggested wording:** "CODI's depth-carrying channel is one fed-back vector per step (narrow but persistent); VB's is a T-slot version per step."

---

## 3. L1-c restated

**Definition (interventional reads).** A set R_t ⊆ {1..t−1} is a *read set* of pass t if the following holds: for all x, x′ in the support, V_t(x) is unchanged when every V_s(x) with s ∈ {1..t−1} ∖ R_t is replaced by V_s(x′) in **all** of pass t's inputs.
- Those inputs are the bank entries, and, under CARRY, the carry N(V_{t−1}).
- Joint replacement is required, because single-version invariance does not imply joint invariance.
- Chain length: L(0) = 0 and L(t) = 1 + max_{s ∈ R_t ∪ {0}} L(s). Define the same for the coda's read set.

**Lemma L1-c′ [PROVED].** V_t is computed by a layered block network of depth P + L(t)·d and width ≤ (t+1)T. The output after pass t is computed by one of depth P + L*·d + c, with L* = max(L(t), max_{s∈R^𝒟} L(s)).
- *Proof:* fix a reference input x*. By invariance, V_t(x) = Φ_t(V_{R_t}(x), V_{¬R_t}(x*)). The second argument is a constant, so it has depth 0. Induct on t. ∎

**Consequences.**
1. *A's L1-c is the case L ≡ 1.* A pass that interventionally ignores every V_{s≥1}, *carry included*, writes depth ≤ P + d.
2. **Idling under CARRY is a read.** The pretrained residual stream carries V_{t−1} forward, so an idle pass (V_t = V_{t−1}) has t−1 ∈ R_t. A recompute-only CARRY pass must actively overwrite its own residual stream. "(recomputes from V_0, or idles)" must therefore drop "or idles" under CARRY.
3. **Exact copies can be contracted.** A pass with V_t ≡ V_{t−1} is a depth-0 wire. Its duplicate bank entry equals a log 2 logit offset on V_{t−1}, which a fixed modified mask implements. So depth ≤ P + d·L_eff + c, where L_eff counts non-copy hops. This is what G5's "idle passes before completion leave the critical path too short" needs. It is proved for exact copies; approximate copies need a robustness assumption.
4. **Width.** Passes that read only V_0 can still compute *different* functions of it, because the bank size and V_0's bias β + γ(t−1) change with t. K such passes form a depth-(P + d + c) network of width (K+1)T + L_dec. The "one-pass class" must be defined at that width. DIRECT calibrates width T only, so G9 lives inside L1-c itself.

---

## 4. §4(iii): what zero L_rec forces

**Setting.**
- *"Zero L_rec"* means exact decoding: argmax R(V_t(x)[j]) = E_t(x)[j] for all x, j and t ≤ K. CE itself cannot reach 0.
- h_t(x) is the maximum tree height among the nodes reduced in steps 1..t, with h_t ≤ t.
- **Formal membership:** "x ↦ E_t ∈ 𝒞(D, W)" means that some weights of depth D and width W compute it. No finite-n lower bound is known for GPT-2-sized nets. Addendum C notes that Feng's Thm 3.1 is asymptotic under TC⁰ ≠ NC¹. At n ≤ 16, a 10-layer, 768-wide network computing every E_t is not excluded.
- **Empirical membership:** a model of that depth trained at this budget reaches accuracy ≥ 1 − ε on E_t.

**Counterexample to the r₁-only argument (M2).**
- Pass 1 writes E_1 (depth P + d).
- Every pass t ≥ 2 reads only V_0 and V_1, and writes E_t in d more layers (depth P + 2d = 14).
- M2 satisfies exact L_rec at every t, the readiness window, and "one pass computes ≤ r₁ ≈ 2 levels". Yet L(t) = 2 for every n.
- M2 is excluded only by a bound on what depth P + 2d can do. So r₁ alone yields "pass t reads some V_{s≥1}", that is L ≥ 2, and not a growing chain.

**Theorem 4a (finite n) [PROVED UNDER CAP-aff].**
- *CAP-aff:* in the trained model, a version computed with L_eff = m decodes E_t(x) only if h_t(x) ≤ c₀ + (m − 1)c. Calibration: c₀ ≈ 2 (DIRECT), c ≈ 1.
- *Then,* L_eff(t; x) ≥ 1 + ⌈(h_t(x) − c₀)/c⌉.
- *Proof:* L1-c′ plus CAP-aff. ∎

**Theorem 4b (asymptotic) [PROVED UNDER TC⁰ ≠ NC¹].**
- For each fixed m, exact decoding of E_n via chains of length ≤ m fails for some n. The reason: E_n contains the root value; the network has fixed depth P + m·d and polynomial width; and Feng's Thm 3.1 applies through §1, restatement 1.
- So L_eff is unbounded in n, but no n(m) is given.

**Minimal assumption.** A finite capability profile r(m) for every m ≤ K, of which CAP-aff is the natural affine form. It must hold *for the trained VB model's own sub-computations*. That is the whole empirical content. The lemma converts a capability measurement into a structural conclusion. It does not replace SKIP-DELAY or SERIAL-SLOT, which test the conclusion directly.

**Growth is in tree height, not n.** The required chain grows with height (means 2.67 / 5.04 / 7.03 at n = 3 / 8 / 16), not with n.

---

## 5. Readiness-window answer loss (R1)

**Prop 5.1 (no G8 incentive) [PROVED].**
- L = L_ans + λL_rec is a nonnegative combination of terms −log q_θ(target), and every target (y_i, E_t) is θ-independent.
- No likelihood enters with a negative sign. Answers at t < τ_i do not enter at all.
- Hence any θ, θ′ with equal in-window and recode likelihoods have equal L. Degrading an early answer is never rewarded, and degrading early *content* raises L_rec. ∎
- This is the structural opposite of Cho 2505.23060's r₂ − r₁.

**Count keying.**
- *n is cheap.* n is the number of operator tokens, which one attention layer over V_0 counts at log precision.
- *t is available three ways under NOPASS:*
  - (a) The bank's normaliser: at fixed β, the bank mass is monotone in t.
  - (b) **The age bias on V_0.** V_0 is always present, with relative age t − 1. Its logit β + γ(t−1) is therefore an *absolute* pass index whenever γ ≠ 0. A relative-age bias with a fixed-origin entry is an absolute clock.
  - (c) Content: the number of live op slots in V_{t−1} is n − (t−1).

  So NOPASS hides no clock.
- **Both rules are zero-loss on-distribution.** Under exact L_rec every V_t is pinned (E_t, and HOLD past n). Consider:
  - "reduce until done" (rule C): reduce the leftmost op if any op is alive, else hold;
  - "count-keyed" (rule K): reduce if t ≤ n, else hold.

  Both reach zero loss and agree on every training trajectory, so **neither is an optimum over the other**. They differ only when per-pass progress ≠ 1.
- **SKIP-DELAY cannot tell them apart.** As specified (no V_t written), the bank-size clock skips too, so rule K also recovers at n + 1.
- **Proposed STALL reading.** At t = ⌈n/2⌉, append a copy V_t := V_{t−1}.
  - Rule C recovers at K′ = n + 1.
  - Rule K (with a bank-size or γ clock) holds with one op alive and never recovers.

  STALL is an on-distribution test of G11 and complements A's off-distribution DEPTH/REACH.
- **Simplicity favours C [CONJECTURE].** "Any op slot alive" is an OR over linearly decodable classes, one layer, which is cheaper than a comparison of two counts.
- **Without L_rec** (N3 text aux, GSM8K), R1 is a deadline at τ: it rewards ≥ 1 step per pass. The U{0..4} tail trains holding only to τ + 4 ≤ 12.

---

## 6. B's propositions

**Prop H.**
- *The definition of R(F) is ill-posed.* "Passes whose text can change F's output" fails because texts are coupled across passes: an in-window literal appears on every pass of its window, so changing one x^(p) alone leaves the support, and the set can be empty for every F.
- **Replacement:** F *reads S* if F = G((x^(p))_{p∈S}) on the support.
- **Proof check:** if S misses W(ℓ), then conditional on u_{−ℓ}, the texts in S depend on ℓ only through the decoys b_ℓ, a_ℓ, which are independent of u (A-IND). So F ⊥ u_ℓ | u_{−ℓ}, while y is not (A-DEP). Hence P(F ≠ y) ≥ E[1 − max_v P(y = v | u_{−ℓ})] > 0. **[PROVED]**, and so |S| ≥ h(τ).
- **On kept rows** the keep event {y ∉ {P(v_p)}} depends jointly on (u, b, a), so A-IND fails. H then needs **A-KEEP**: with positive kept-probability, the out-of-window texts leave ≥ 2 possible values of y.
- **The exclusion leaks** ≤ log(11/(11 − m)) nats about y (m = distinct excluded values). Snapshot policies cannot exploit it (a snapshot's answer is as deep as the item), but the coincidence rate becomes ≈ 1/10.

**Prop D [PROVED].**
- *Data processing.* For p > hi(ℓ), the texts are independent of u_ℓ given u_{−ℓ} and the decoys. And M_{lo−1} ⊥ u_ℓ. So u_ℓ's information sits in Z_lo..Z_hi *given* M_{lo−1}; it may be synergistic with M_{lo−1}.
- **VB caveat.** In VB (SYNTH's STAGGER add-on), if the input snapshot V_0^(p) is *appended* to the bank, then V_0^(p) itself is the carrier, and D implies no re-coding. Re-coding needs the input entry to be *replaced* each pass. Otherwise timing is forced only by depth (§3).

**Prop B [PROVED].** Trivial by the exclusion rule. Mixed (multi-snapshot) policies are covered by H, not by B.

**Prop M(a): timed release.**
- *Pointwise R > 0 fails.* d/dβ log softmax(βℓ*)_y = ℓ*_y − E_β[ℓ*]. This is > 0 for all β iff y is the unique argmax. If ℓ* misranks y, the derivative turns negative once the softmax concentrates, so **R < 0 on those items**. "R > 0 for every K, j" is false pointwise.
- **Restated [PROVED].** Take calibrated ℓ* = log q. By the proper-scoring property, β ↦ E_{y∼q}[log softmax(βℓ*)_y] is concave with its maximum at β = 1. So any strictly increasing g ≤ 1 gives **E[R] > 0 for all K, j**.
- *This is release by under-confidence.* Early passes are deliberately worse than they could be, which is G8 by construction, as B says.
- **The clock is general.** VB's bank size or γ provides it, and any state-carrying system can build a counter in content. So M(a) holds even for size-invariant reads (C's top-k).
- **Bounded logits** (|z_v| ≤ B) give log p_y ∈ [−2B − log 11, 0]. The increments telescope, so the total release is ≤ 2B + log 11 per item. Release with a bounded, strictly increasing g still gives E[R] > 0: bounded logits cap the reward, they do not remove the incentive.
- **Unbounded logits** with both terms differentiated: the objective is unbounded above whenever y is known at K − j, by driving p_{K−j}(y) → 0. That is strict sabotage, the Cho route.

**Prop M(b) [PROVED].**
- ∇_θ φ(log p_K − sg log p_{K−j}) = φ′(Δ)·∇ log p_K, with Δ = log p_K − log p_{K−j}.
- So the step is CE at the sampled K with per-item weight w = φ′(Δ). For φ = id the weight is 1: the plain K-mixture CE.
- **The weights also depend on θ through p_K.** "Reweighted" is therefore exact *per step*, with weights evaluated at the current θ. The descended objective is Σ_i φ(log p_K^i − c_i) with c_i frozen. Its stationary points are those of a CE reweighted by φ′ at that point.
- The update lies in span{∇ log p_K^i}. No term separates iterative from one-pass solutions of equal p_K.

---

## 7. C's Prop 4 and Lemma 3

**Prop 4, restated [PROVED UNDER single-variable + exchangeability].**
- *Setting:* candidates c_0..c_N vary one value v*. The values (v_0, w_1..w_N) are exchangeable given the context, for example v* | ctx uniform and negatives drawn uniformly without replacement. G ⊥ v* | ctx.
- *Then* the score vector is exchangeable in the index, so E[p_true] = 1/(N+1) and E[ℓ^NCE] ≥ log(N+1) by Jensen.
- *MI form:* with negatives i.i.d. from p(v* | ctx), I(G; v* | ctx) ≥ log(N+1) − E[ℓ^NCE] (CPC 1807.03748; Poole et al. 1905.06922).
- "E[ℓ] ≥ H(v* | G_t)" should read **H(index | G_t, C_t)**.

**Two gaps for reuse:**
1. **C's C_t mixes variables.** ẽ³ varies an *older* value. A student that is blind to v_t but carries the older value can push ẽ³'s mass to 0, so its floor is **log 3**, not log 4. Only a fully value-blind student (a literal copier) has log 4. That is still enough for C's G4 row.
2. **Prior artefact.** With the context-only critic (parent op and side), the measured value is E[ℓ] = 1.0954 vs log 3 = 1.0986, which is **0.003 nats**. Divisor contexts exclude 0 on both sides, which is consistent. Negligible.

**Lemma 3 [FALSE as written: vacuous].**
- The task state S_t (the re-posed e_{t+1}) is a deterministic function of x, and masked passes see x. So I(S_t; anything | G_{t−1}, x) = 0 *for every model*. The ε/p_m bound says nothing.
- The constraint that bites is computational, and Shannon information cannot express it.

**Restated [PROVED UNDER no-conflict + CAP], in 𝒱-information (Xu et al. 2002.10689).**
- *Setup:* L = (1 − p_m)L_free + p_m·L_mask, with L* = inf L. **No-conflict:** some θ minimises both terms.
- **Step 1.** L(θ) ≤ L* + ε ⇒ L_mask(θ) − inf L_mask ≤ ε/p_m.
  - *Proof:* no-conflict gives L* = (1 − p_m)·inf L_free + p_m·inf L_mask, so L(θ) − L* = (1 − p_m)(L_free − inf L_free) + p_m(L_mask − inf L_mask). Both brackets are ≥ 0.
- **Step 2.** Suppose perfect re-coding is realisable (inf L_mask = 0), and suppose CAP holds as H_𝒱(S_t | x) ≥ κ > 0 for t > c₀, where 𝒱 is the one-pass class. Then the newest group supplies 𝒱-usable information I_𝒱(G_{t−1} → S_t | x) ≥ κ − ε/p_m. ∎
- **Scope.** Only masked passes are constrained (G13 stays measured); RE-POSE's cosine and NCE terms are surrogates for S_t-CE units.

---

## 8. R3 (teacher-forced versions): well-posedness

The student's V_s (depth P + s·d) and the teacher's V^T_s = core(prelude(render_VOID(e_s))) (depth P + d) come from different circuits. Depth is not itself the problem. Substitution needs three conditions.

- **(W1) Index alignment.** L_rec trains pass 1 to perform one reduction, R(V_1(x)) = E_1(x). So one pass on render(E_k) decodes E_{k+1}.
  - For V^T_s to stand in for V_s (which decodes E_s), the teacher must run on **E_{s−1}**.
  - "e_s" is correct only in C's indexing (e_1 = x). In A's E-indexing (E_0 = x), the teacher is one step *ahead*, and TF trains copying.
  - Gate: R(V^T_s) must decode E_s.
- **(W2) Teacher correctness.** Filler-rendered inputs are never pass-1 training inputs (L_rec sees clean x), so R(V^T_s) = E_s is not guaranteed at the start.
  - Either gate TF on the teacher's decode accuracy, or add L_rec on the teacher pass (target E_s, with no sg on that term).
- **(W3) Read invariance.**
  - Under PURE, a pass reads a bank entry v only through π(v) = (W_K^ℓ·LN_ℓ(v), W_V^ℓ·LN_ℓ(v)) for ℓ ∈ core ∪ coda. Exact substitution ⇔ π(V^T_s) = π(V_s).
  - Under CARRY, N(V_{t−1}) enters the residual whole, so substitution needs V^T_{t−1} ∝ V_{t−1} slot-wise.
  - Agreement through R constrains only a 20-logit image of a 768-vector. Everything else is free: the teacher's shallow token-identity features, and the student's history and depth features.
  - R3 must also say whether the carry is replaced (it should be). Otherwise passes see a mixed history.

**Exposure bias.**
- C paired TF with ℓ^cos, which makes student codes substitutable. **SYNTH's R3 has no matching term.**
- TF-trained step maps compound error roughly O(t²ε) against O(tε) on-policy (Ross et al. 1011.0686).
- The anneal p_TF → 0 by 24k leaves 24k on-policy steps, so the final objective is free-running. The scheduled-sampling inconsistency (Huszár 1511.05101) is therefore moot.
- The real risk is a teacher-specific read learned early that the on-policy phase must unlearn: TF-ONLY / CODE-DRIFT, read by ROUND-TRIP.

**Fixed-point criterion.** A teacher is clean iff V^T_s = V_s whenever V_s is already correct. R3's teacher fails this: it is built from a different input at a different depth.

**The candidate** ("V_1 on render_VOID(e_s) after s passes of identity"):
- It matches the bank size and γ ages, the implicit clock of §5, so it removes one mismatch.
- But the identity padding is imposed, not the student's own machinery. Content and history features still differ, W1 and W2 remain, and the fixed point fails.
- *Verdict:* a partial improvement.

**Proposal: the minimal-correction teacher (MCT).**
- *Definition:* V^T_s = V_s + Δ_s, where Δ_s is the minimum-norm edit that makes R decode E_s with margin μ. It is taken in LN coordinates, in the row space of R's 20×768 weight, using the pseudo-inverse on the logit error, then mapped back with V_s's own mean and scale. Δ_s = 0 when V_s already decodes E_s.
- *Properties:*
  - the same machinery, the same context, and the same depth and history features;
  - the fixed point by construction, so there is no exposure bias at convergence;
  - the carry and the bank use the same V^T_s.
- *Precondition:* later passes read E-content through R's row space. TF trains exactly this. Test it by patching along R's row space vs its null space.
- *Cost:* weaker forcing at init, since the correct content occupies only a 20-dimensional subspace.
- *Recommendation:* MCT for the R3 cell, or R3 with gates W1–W3 plus the TF-ONLY reading.

---

## 9. Gate G1 (function preservation)

**Prop 9.1 [PROVED].** CARRY at K = 1, with β = β^𝒟 = −∞ and g = 1, gives exactly GPT-2's logits at " =" on [EOT, problem, " ="].
- *Prelude.* V_0 = h_6.
- *Carry.* U_1^0 = V_0·rms(V_0)/rms(V_0) = V_0. In floating point r/r = 1.0 exactly, provided both factors come from the same tensor and kernel; with an eps inside rms the ratio is still 1.
- *Core.* Bank logits are −∞, so the softmax runs over the current causal keys only. The row max is unchanged, every row keeps its diagonal (no NaN), and γ·age stays −∞. Blocks 6–9 are therefore GPT-2's, and V_1 = h_10.
- *Coda.* It is causal over V_1, with the bank at −∞. Decoder tokens follow the " =" slot and are invisible to it. This gives h_12, then ln_f and the tied head. ∎

**Conditions the code must meet.**
- The readout is at the per-item position len_i − 1, as in `InjectN.dec_logits`, not at the padded T − 1.
- The coda attention over V_t is *causal*. A non-causal coda would change earlier slots, which feed the " =" slot through block 11.
- β^𝒟 is masked as well as β.
- Eval mode (no dropout).
- The mask is −∞ or ≤ −1e4: exp underflows to exactly 0 beyond about 745 nats in fp64.

**Numeric check (`g1_check.py`).** On random 12-layer fp64 nets, max |Δlogit| = 3.6e-15, 1.3e-14 and 1.2e-13. This is rounding from the changed reduction shapes. Plain GPT-2 against VB is therefore not bitwise, but it is far inside the 1e-9 gate.

**At β = −5.**

*The exact per-head identity.* For fixed inputs, o′_j = (1 − m_j)·o_j + m_j·b̄_j, with:
- m_j = e^β Z_b/(Z_c + e^β Z_b) ≤ e^β Z_b/Z_c, where Z sums exp(raw logits) over the bank or the allowed current keys;
- b̄_j = the bank-softmax average of the values.

Hence:
- ‖Δo_j‖ = m_j·‖b̄_j − o_j‖ exactly;
- Δz = Σ_{ℓ,h,j} J_{ℓ,h,j}·m_{ℓ,h,j}·(b̄ − o) + O(m²), where J is the Jacobian of the logit with respect to the head output;
- for β ≪ 0, Δz = e^β·C + O(e^{2β}).

*Layer 6 at t = 1 is special.* The bank keys and values are the current ones (the same vector through the same block). The duplicates cancel, and only *future* keys i > j matter: m^f_j = e^β Z_f/((1 + e^β)Z_c + e^β Z_f). This is **zero at " ="**: the measured change there is 1.8e-15, against 8.4 at slot 0.

*Why 1.3% is not a bound.*
- It is the equal-raw-logit value at a mid query. At t = 1 with T = 26, that value is 0.67% at " =" and 15% at slot 0.
- With a raw-logit advantage Δ for bank keys, the mass grows like e^{β+Δ}, so a 5-nat advantage cancels β = −5.
- Random nets with mild attention: per-layer maximum mass 0.14–0.45, and Δz(−5)/Δz(−10) = 87 (first order predicts e^5 = 148).
- Random nets with peaked attention: maximum mass 1.0 at every layer, and a ratio of about 1. That regime is non-perturbative.
- So "near-exact at −5" does not follow. GPT-2's sinks and peaked heads must be measured.

**G1 additions.**
- Report max_j and the 99th percentile of m per (layer, head).
- The ratio test: Δz(−5)/Δz(−10) ∈ [0.5, 2]·e^5 confirms the first-order regime.
- If the test fails, calibrate at step 0: β_{ℓ,h} := log μ − q_{0.99,j}[log Z_b − log Z_c] on a calibration batch. This guarantees m ≤ μ on that batch (for example μ = 0.02). It also sets A's "bank gradient not e⁻⁸-small" scale explicitly (§5, mitigation 4).
