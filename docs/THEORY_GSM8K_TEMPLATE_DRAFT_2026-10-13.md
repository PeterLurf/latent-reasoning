# GSM8K-TEMPLATE-RESAMPLE pilot: §191's recipe on GSM8K-Aug, with per-draw name relabelling and exact number resampling (DRAFT, 2026-10-13 23:10 EDT; Opus subagent at the user's direction; reviewed by Theory 23:2x; not registered; needs Audit's review and the USER's approval for GSM8K spend)

**Status.**
* Drafted by a Theory subagent; D:\Latent was read only; no GPU, no launch.
* Measurements: CPU, idle priority, ≤ 3 threads. Scripts and JSON outputs are in this folder (§7).
* **Launch gate.** GSM8K-scale spend needs **the user's explicit approval**, whatever the projected cost. A peer's OK is not that approval.
  - The user's ladder rule: "beat SOTA on smaller tasks, then run GSM8K". §191 A reads 0.950 on ProsQA-deep against text-CoT's 0.938. That *matches* text on a custom set; it is not a published-SOTA win. **The user decides** whether §191 satisfies the ladder.
* **Inputs folded in:**
  - the user's "consider gsm8k aug" (relayed by Audit about 22:4x): GSM8K-Aug is the **primary** training set;
  - Audit's (a)–(d);
  - Audit's K-schedule / WINDOW request;
  - Literature's verified bars (docs/LIT_GSM8K_AUG_BARS_2026-10-13.md, fbc78a25).

**Tags.**
* PROVED: a full proof under the stated assumptions.
* DERIVED (A…): follows from the named, unverified assumptions.
* CONJECTURE: an argument or a transfer from another task.
* DATA: read from existing files. **(post hoc)** marks numbers looked at while drafting.
* calc: arithmetic done here.

---

## 0. Summary

1. **Measured lengths: GSM8K inputs are 7.6× shorter than ProsQA, not longer.** Think input = [EOT] + question, in GPT-2 tokens (mean / p95 / max):
   - ProsQA p8_na_train: 440 / 560 / 722;
   - GSM8K test: 57.8 / 99 / 183;
   - GSM8K-Aug train: **43.3** / 68 / 214.
   - **The Aug train set is shorter and shallower than the GSM8K test.** It averages 2.60 equations against the test's 3.25. 25% of test items are longer than Aug's p95.
2. **Cost: §191 A's 6.99 s/step anchor already includes caching.** It ran with `cache_lower: true`, `ckpt_passes: true`, `chunk_delta: 64` and micro 4 (args.json), so the brief's "uncached" premise is wrong.
   - Scaled by measured tokens and K, and calibrated on the realised §191 A and nested RMS runs, a GSM8K-Aug latent cell projects to **≈ 0.9 s/step (high 1.34)**. That is **$1.5–2.3 per 24k-step cell**.
   - The TEXT-COT cell projects to $0.6–0.9.
   - **Stage 1 (4 Aug cells + 2 raw-GSM8K cells): ≈ $6–9. Maximum with all conditional cells: ≈ $12–18.**
3. **GSM8K-Aug audit (Audit's (d)).** Every `<<expr=val>>` was re-evaluated by a whitelisted-AST evaluator (no eval):
   - 97.8% of 999,252 equations are exact, 1.7% are rounding, **0.31% are wrong** and 0.13% are unparsable;
   - 0.77% of items contain a wrong equation. Filter: drop any item with a wrong or unparsed equation.
   - **77.2% of items can be recomputed exactly** under number resampling.
4. **Exposure is what separates memorisation regimes (Audit's (a)):**
   - 24k steps × bs 64 = **4.0 epochs of Aug**;
   - the same budget is 205 epochs of raw GSM8K;
   - §168 s2 memorised at 64 epochs of p8.
   - So on Aug the name / number lever is predicted **small** (P1, P2). Its clean test is a raw-GSM8K tier (fixed vs resampled; P3).
5. **Structure memorisation (the part resampling cannot remove), (post hoc).**
   - 55.6% of GSM8K test items have a computation fingerprint that occurs in Aug train.
   - **Every public GPT-2 checkpoint is 0.20–0.29 more accurate on seen than on unseen structures** (m-stratified). Text CoT is 0.25. Coconut + SIM-CoT is 0.58 vs 0.27.
   - Resampling is derived to leave this gap unchanged (Prop S, P4).
6. **Length.** p8_ctl is **not** a length out-of-distribution test:
   - 0% of its items exceed the training maximum, and its mean is 475 tokens against the test's 459;
   - it is a **length–depth mismatch**: its length overstates depth by +3.6 hops.
   - On GSM8K the set-level size-keyed depth error is −0.06 equations (none), but its spread is ±0.7 per tercile.
   - **Falsifiable prediction (CONJECTURE transfer):** a neutral 32-token filler costs the latent arm **≥ 0.04 more** accuracy than text-CoT (central 0.075) if the latent arm is size-keyed, and ≤ 0.015 if it is length-robust.
7. **K schedule:** fixed **K = 10**, as in §191.
   - That is the test maximum of 8 equations + 2. It covers 99.997% of Aug items with at least 2 trained post-completion passes on every test item.
   - **WINDOW** gets its own derivation (Lemma W). It is a *conditional* cell, opened only if A fails STABLE.
   - CSR (§199) is named but not stacked.

---

## 1. Measurements (CPU, done now)

### 1.1 Think-input token lengths (GPT-2 / tiktoken, the encoder prosqa168.ENC uses; [EOT] + question; `lengths.json`)

| set | n | mean | p5 | p50 | p95 | p99 | max | mean α^L at init (α = σ(5) = 0.99331) |
|---|---|---|---|---|---|---|---|---|
| ProsQA p8_na_train | 11,985 | 440.0 | 323 | 439 | 560 | 611 | 722 | 0.059 |
| ProsQA p8_na_test | 4,053 | 459.4 | 332 | 462 | 584 | 631 | 714 | 0.053 |
| ProsQA p8_ctl | 795 | 475.4 | 367 | 478 | 587 | 621 | 701 | 0.046 |
| GSM8K train (openai) | 7,473 | 56.2 | – | 52 | 96 | 122 | 214 | 0.69 |
| **GSM8K test** | 1,319 | **57.8** | 31 | 54 | 99 | 120 | 183 | **0.685** |
| **GSM8K-Aug train** | 384,620 | **43.3** | 26 | 41 | 68 | 85 | 214 | **0.750** |
| GSM8K-Aug valid | 500 | 56.3 | 31 | 52 | 97 | 117 | 166 | 0.69 |

* **ProsQA / GSM8K test mean = 7.61×.** The ProsQA latent cost does not transfer to GSM8K; the GSM8K cost is far lower (§3.7).
* **Aug train is shorter than the test:**
  - 25.3% of test items exceed Aug's p95 (68 tokens), 11.2% exceed its p99, and 0% exceed its maximum.
  - ProsQA test items exceed the train p95 at only the nominal rate (9.8%).
* **Text-CoT sequence length** (question + equation-format CoT + answer): Aug mean 66.9, p99 131, max 250. ProsQA: 519.
* **Aux targets** (" <<expr=val>>"): mean 7.8 tokens, p95 11, max 63.
* GSM8K-Aug test = the 1,319 GSM8K test items (identical lengths and equation counts).

### 1.2 Equation counts m (number of `<<…>>`) and K coverage (`lengths.json`)

| m | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11–13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aug train (n) | 88 | 62,651 | 143,204 | 103,993 | 48,072 | 17,853 | 5,651 | 2,354 | 575 | 126 | 43 | 10 |
| Aug cumulative share | 0.0002 | 0.163 | 0.535 | 0.806 | 0.931 | 0.977 | 0.992 | 0.998 | 0.9995 | 0.9999 | 0.99997 | 1 |
| GSM8K test (n) | 18 | 65 | 357 | 364 | 290 | 138 | 57 | 21 | 9 | – | – | – |

* **The test is deeper than Aug train:**
  - mean m is 3.25 vs 2.60;
  - m ≥ 5 is 17.1% of test vs 6.9% of Aug;
  - the test maximum is 8.
* **K = 10:**
  - covers every test item with at least 2 slack passes;
  - covers 99.997% of Aug items with an aux target per step.
  - The 10 Aug items with m > 10 get aux for t ≤ 10 only (logged). The 88 items with m = 0 get no aux.
* Our bench's step buckets (GSM8K_REPRO) match this table.

### 1.3 GSM8K-Aug equation audit (Audit's (d); `audit_aug.py` → `aug_audit.json`)

**Method.** Each `expr=val` is evaluated by a whitelisted AST evaluator: numbers, the operators + − * / ** % //, unary signs and parentheses. There are no names or calls, power is capped, and nothing is eval'd.

| status (per equation) | Aug train | GSM8K train / test / Aug valid |
|---|---|---|
| exact | 977,400 (97.81%) | 100% |
| round (matches when rounded to the RHS's decimals; e.g. 20/60 = .333) | 17,468 (1.75%) | 0 |
| **wrong** (e.g. `1500/20=-75`, `25-15/2=10`) | **3,105 (0.31%)** | 0 |
| unparsed (non-arithmetic text) | 1,279 (0.13%) | 0 |

* **Items:**
  - all exact 94.9%; exact or round 98.95%;
  - **any wrong 0.77%; any unparsed 0.29%**;
  - no equation 0.02%.
* **Answer = last equation's value:** Aug 95.6%. GSM8K train itself is 93.7%, so most mismatches are native (the final answer is not the last annotation).
* **Operand grounding** (Aug; 2.15M operands):

  | category | share |
  |---|---|
  | question literal (Q) | 55.3% |
  | previous result (R) | 30.7% |
  | number word in the question (W) | 8.5% |
  | common constant (K) | 3.4% |
  | **ungrounded (U)** | **2.1%** |

  - 9.5% of items have a U operand, against GSM8K train's 14.8%, so the GPT-4 items are *not* noisier on this measure.
* **Recomputable for number resampling:** **77.2% of Aug train** (GSM8K train 65.1%, test 62.4%).
  - Definition: all equations exact, no U operand, answer = last value, and every question literal value occurs once.
  - The recomputable subset is representative: mean L 41.9 vs 43.3, mean m 2.48 vs 2.60.
* **Target numerals in recomputable items:** 91.5% vary under resampling (Q copies, R carries, results); 8.5% are fixed constants.
* **Filter (registered):** drop any item with a wrong or unparsed equation. That is at most 1.06% of Aug (union bound); the harness logs the exact count. "Round" equations are kept: they are text targets.
* **Overlap:**
  - 6,955 of GSM8K's 7,473 train questions appear verbatim in Aug train (Aug ⊇ 93% of GSM8K train);
  - **0 GSM8K test questions** appear verbatim in Aug train;
  - gsmlen/dedupe.json flags one near-duplicate test item (602) against the short split. A full 8-gram near-duplicate scan is a gate (§3.6).
* **Names (heuristic: capitalised, non-sentence-initial, stop-listed):**
  - 42% of Aug items (GSM8K train 52%) have a candidate;
  - 8,403 distinct candidates in Aug (GSM8K train 2,741);
  - the most frequent are John 10,721, Sarah 5,460, Tom 4,548, and so on.
  - **Aug's names are already diverse and non-identifying** (§2 D1).

### 1.4 Draws per distinct item (Audit's (a); bs 64)

| steps | draws | epochs of Aug (384,620) | epochs of raw GSM8K (7,473) | epochs of p8 (11,985) |
|---|---|---|---|---|
| 12k | 768k | 2.0 | 102.8 | 64.1 (§168 s2 / §191) |
| **24k** | 1.536M | **4.0** | 205.5 | 128 |
| 48k | 3.07M | 8.0 | 411 | 256 |

### 1.5 Structure overlap (`audit_aug.py`, `extra_stats.py`, `seen_split.py`)

**Definitions.**
* **Skeleton:** each equation's left side, with numbers mapped to q_i (the i-th distinct question number), r_j (the j-th previous result), k_v (a word or constant), or u.
* **Fingerprint:** the skeleton evaluated at two fixed pseudo-random literal assignments, plus m. Commutations and reordered steps of one function therefore collide (intended). Items with a u or unparsed operand get no fingerprint.

**Counts.**
* Aug train has 107,754 distinct skeletons and 50,426 distinct fingerprints. GSM8K train has 6,126 skeletons and 4,226 fingerprints.
* 984 skeletons cover 50% of Aug items.

**GSM8K test: share whose fingerprint occurs in Aug train.** It is 0.556 (GSM8K train: 0.343). 1,120 of 1,319 test items have a fingerprint.

| m | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| seen share | 0.62 | 0.85 | 0.70 | 0.36 | 0.21 | 0.04 | 0.05 | 0.00 |

**Public GPT-2 checkpoints on seen / unseen structures (POST HOC, descriptive).** Per-item `correct` from bench/results/gsm8k__*.json; 733 seen, 387 unseen and 199 without a fingerprint. The last column is m-stratified, with weights n_s·n_u / (n_s + n_u).

| checkpoint | all | seen | unseen | seen − unseen (m-stratified) |
|---|---|---|---|---|
| CoT (Dilgren ck25) | 0.418 | 0.561 | 0.222 | **0.248** |
| No-CoT (ck25) | 0.165 | 0.246 | 0.052 | 0.150 |
| Coconut (ck33, K 6) | 0.331 | 0.486 | 0.116 | 0.207 |
| CODI L6 | 0.425 | 0.583 | 0.222 | 0.212 |
| SIM-CoT + CODI L6 | 0.394 | 0.559 | 0.153 | 0.291 |
| SIM-CoT + Coconut K10 | 0.448 | 0.583 | **0.266** | 0.200 |
| SIM-CoT + Coconut K0 | 0.051 | 0.067 | 0.034 | −0.010 |

* **Reading (descriptive).**
  - Every trained reasoner, text CoT included, is about 0.2–0.3 more accurate on seen structures at the same m.
  - Unseen-structure accuracy is 0.22–0.27, so none of them is a pure retriever.
  - The seen / unseen contrast is confounded by within-m difficulty: unseen structures are rarer and wordier. It is a comparator, not a causal estimate.

### 1.6 Length ↔ depth (`extra_stats.py`)

**ProsQA (p8_na_train).**
* corr(L, depth) = 0.54 (R² 0.29); by depth, mean L rises from 386 (depth 3) to 499 (depth 10).
* E_train[depth | L] − depth:

  | set | mean (hops) |
  |---|---|
  | test, depths 3–10 | −0.19 |
  | **p8_ctl** | **+3.58** |
  | ctl, m = 4 | +3.06 |
  | ctl, m = 8 | +4.15 |

* **So p8_ctl's items are within the training length range** (0% above the train maximum; the 10% above train p95 matches the test's rate). What ctl changes is **the length–depth relation**: long inputs with answer depth 3–4.
* The §191 line "input-length control: A 0.640 vs B 0.816" should read as **"not robust to a length–depth mismatch (distractor chains)"**, not to length.

**GSM8K-Aug.**
* corr(L, m) = 0.51 (R² 0.26), similar to ProsQA. The slope is 0.047 equations per token, i.e. 5.59 tokens per equation for L on m.
* **Test vs Aug train:**
  - the length residual at fixed m is +10.6 tokens (median +7.1; 66% of items positive);
  - Aug valid: +9.6;
  - but the **size-keyed depth error E_aug[m | L] − m is −0.06 equations** on average. The test is longer *and* deeper.
  - By length-residual tercile, the error is **−0.81 / −0.05 / +0.68** equations.

### 1.7 TemplateGSM 2411.18104: availability and format (`dl_tgsm/`, `tgsm_sample.py`)

**Availability.**
* **Not in the repo.** It is on HF as `math-ai/TemplateGSM`: license tag cc-by-4.0; config `templategsm-7473-1k`, i.e. 7,473 templates × 1k instances, one jsonl per template (7,480 files).
* Fields: problem, solution_code (Python), result, solution_wocode (NL), source ("gsm8k-train-round2-seed42"), template_id, problem_id.
* **template_id = the GSM8K train index.** Checked on templates 0, 463, 1813 and 3615.

**Fidelity sample.** 81 rows from 27 templates, via the datasets-server rows API. Code is parsed only by a whitelisted evaluator and never executed.
* 69 / 81 rows have straight-line code that parses. Of those, 82.6% have a computed-assignment count equal to the prototype's equation count.
* result = the code's value in 100%. 98.6% of results are integers.
* Problems are 27% longer than their prototypes (67 vs 54 tokens).
* **Surface noise:**
  - 15% of rows contain underscore placeholders ("a_tea_house_boutique in Pickaway_County,_Ohio");
  - broken plurals ("skirtss");
  - misplaced item slots ("averaged 16 a campground trading post per game");
  - template 0 states the "half as many" intermediate (119) explicitly in the problem, turning a 2-step prototype into a 1-step problem.
* **No `<<expr=val>>` CoT:** it would have to be derived from solution_code.

**Consequence.** Our own templating of Aug / GSM8K items is the primary resampler. It is exact, keeps the structure, and gives free `<<…>>` targets. TemplateGSM is optional, for Tier 2 robustness only, behind a fidelity filter.
* **Literature:** confirm CC-BY-4.0 covers training use and record the dataset sha (0c8ed6b6…, lastModified 2025-08-02).
* **GSM-Symbolic** (apple/GSM-Symbolic, CC-BY-NC-ND-4.0, test-derived) is **eval-only**. It is fetched at eval time, never trained on, and never committed.

### 1.8 Harness facts (read in code; not assumed)

* **boxes/prosqa168s2.py has `--cache_lower`** (`ProsMem.lower`: blocks 0..5 on [EOT, question] once per step, shared by the K passes; it asserts the identical input). **§191 A used it.**
  - §191 A's args.json: `cache_lower: true, ckpt_passes: true, chunk_delta: 64, micro 4, fast_add`.
* **Timing anchors (DATA):**
  - ProsQA step-aux, bs 32 / 64: 3.50 / 6.96 s/step. This is linear in batch, so it is compute-bound at L ≈ 440.
  - chunk_delta 0 (the token loop): 28.3 s/step at bs 32.
  - Realised: §191 A 83,823 s / 12k = 6.985 s/step; §191 B (TEXT-COT) 16,180 s / 12k = 1.348 s/step; nested RMS s1 42,491 s / 48k = 0.885 s/step (uncached, L ≈ 26.4); §198 (a) cached nested 0.861 s/step.
* **A GSM8K family can keep `--cache_lower`:** the think input [EOT, question] is identical on every pass, with no pass index on the think path. nested170's hold rule applies: never compare a cached latent cell with an uncached latent comparator. Here every latent cell is cached.
* **`--ckpt_passes` is not needed at GSM8K length.** The longest Aug question is 214 tokens, against ProsQA's 754-token worst case, which peaked at 18.4 GiB at micro 4. A mem_check decides micro.

---

## 2. Derivation

### D0. Setup and assumptions

**The item and its targets.**
* Item i = (s_i, ν_i, x_i):
  - s is the template: the story text with slots, plus the computation graph σ;
  - ν is the person-name slot fillers (the set N_i);
  - x is the literal vector.
* The question q = render(s, ν, x). The targets are T = (e_1, …, e_m, y): equations and answer.
* **A0 (name-free targets):** T = g_σ(x). Names never occur in `<<…>>` or the answer.
  - CHECK: the evaluator accepts digits and operators only, so any equation with letters would be counted as unparsed (0.13%).

**The two transforms**, drawn fresh at every training draw, independently of everything else; test items are never transformed:
* **R_ν (relabel):** π, uniform over injections N_i → V.
  - V is a gender-matched name pool: the top 250 names per gender in Aug train, with gender from pronoun co-occurrence.
  - It is applied consistently to every occurrence, including possessives.
* **R_x (resample):** for recomputable items, x′ ~ D_i, and T′ = g_σ(x′) recomputed.
  - D_i keeps each literal's digit count, decimal places and integrality, and rejects draws that make any intermediate negative, non-integer where the original was integer, or a division by zero.
  - There are up to 50 tries, else the original x. The acceptance rate is logged.

### D1. Names carry zero information about the targets (Theorem N, PROVED)

**Claim.** Under R_ν, let Z = π(ν) be the rendered name tuple and W the rest of the rendering. Then:
* **(i)** Z ⊥ (s, x, T, W) given |N_i|.
* **(ii)** For any loss ℓ and any predictor h(Z, W):

  min_h E ℓ(h(Z, W), T) = min_{h′} E ℓ(h′(W), T).

  A predictor using Z alone attains at best min_c E[ℓ(c, T) | |N|]: the |N|-conditional constant, i.e. chance-level, risk.
* **(iii)** Two independent draws of one item share their name tuple with probability 1/(|V|)_{|N|}, where (|V|)_{|N|} is the falling factorial. With |V| = 250 that is 4·10⁻³ at |N| = 1 and 1.6·10⁻⁵ at |N| = 2. A lookup keyed on names hits a fresh draw at most that often.

**Proof.**
* π is uniform over injections and independent of (s, x), so P(Z = z | s, x, T, W) = 1/(|V|)_{|N|} for every injective z. That depends on |N| only, which gives (i).
* For (ii), by the tower rule and (i):

  E ℓ(h(Z, W), T) = E_W E_Z E_{T|W} ℓ(h(Z, W), T) ≥ E_W min_z E_{T|W} ℓ(h(z, W), T) ≥ min_{h′} E ℓ(h′(W), T).

  Equality is attained by h(z, w) = h′(w).
* (iii) is direct. ∎

**What it does and does not do on GSM8K-Aug (DATA plus CONJECTURE).**
* On ProsQA, the name configuration over a fixed graph *was* the item's identity, and the model saw it 64 times. Relabelling removed the only cheap identity key, and test accuracy went from 0.49 to 0.95 (§191, DATA).
* On Aug:
  - only 42% of items have candidate names;
  - there are 8,403 of them, and the frequent ones are shared by thousands of items ("John" by 10,721);
  - the story text W identifies the item anyway.
* So **Theorem N removes a channel that carries little item identity on Aug** (CONJECTURE: small effect; tested by P2).

### D2. Number resampling: recall is worth at most the answer's collision probability (Theorem X, PROVED), and how much it removes (calc)

**Claim.** Fix a recomputable item i with x′ ~ D_i, and any target coordinate τ (an equation's value or the answer) that depends on x′. For any predictor h that does not read x′, for example recall of i's stored value keyed on the story:

P(h = τ(x′)) ≤ max_v P(τ(x′) = v) = 2^{−H∞(τ(x′))}.

**Proof.** h is independent of x′ given i, so P(h = τ) = Σ_v P(h = v)·P(τ = v) ≤ max_v P(τ = v). ∎

* For 2–3 operations on literals of 1–3 digits, H∞ is several bits. The harness logs the empirical max-probability of the answer over 100 resamples per item (median).

**The fraction removed (calc, from §1.3).**
* Within recomputable items, 91.5% of the target numerals depend on x′. Every answer does, by definition.
* Set-wide, 77.2% × 91.5% ≈ **71% of Aug's target numerals become non-recallable**.
* The other 29% are:
  - the fixed constants;
  - the numerals of non-recomputable items, which get names only.
* **What is not removed is σ:** which slots combine, by which operators, in which order. That is D3.

**Recall-share decomposition (DERIVED, A-fit).**
* Let fit_fix be accuracy on training items as drawn, and fit_rs accuracy on fresh resampled instances of the same items.
* **A-fit:** the resampled instances are equally hard for the item's computation. D_i preserves digit counts and integrality; a check is that TEXT-COT's two fits agree within 0.03.
* Then by Theorem X, fit_fix − fit_rs = (the recall share of train fit) ± 2^{−H∞}.
* This is measurable on every arm. It is the GSM8K analogue of §191's train-fit-at-K′0 split.

### D3. Structure memorisation: what resampling cannot remove (Proposition S, PROVED; detection DERIVED)

**Claim (PROVED).** A **structure retriever** maps a question to some σ̂ in the training fingerprint set Σ_tr and executes it. On an item whose fingerprint F ∉ Σ_tr, it is correct only by coincidence, at a rate c_i = P(some σ̂ ∈ Σ_tr gives y on x_i). Hence

acc ≤ π_seen + (1 − π_seen)·c̄,

while its train fit can be 1.

* With π_seen = 0.556 on GSM8K test (§1.5), its ceiling by m is:

  | m | 2 | 3 | 4 | 5 | ≥ 6 |
  |---|---|---|---|---|---|
  | ceiling | 0.85 | 0.70 | 0.36 | 0.21 | ≤ 0.05 |

**Why R_ν and R_x leave it intact (PROVED).** Both transforms keep σ fixed by construction (Theorem N's injection and D_i's literal map leave g_σ unchanged), so Σ_tr is invariant. Neither can add a structure.

**When structure memorisation still yields high train fit but poor transfer (DERIVED).**
* Train fit stays high as long as the model's surface → σ classifier fits the training templates, which holds whenever every training structure has many instances. Under R_x it has many.
* Test transfer is then bounded by π_seen(test), plus the model's genuine composition on unseen σ.
* The tell-tale pattern is fit_rs ≈ fit_fix (no number recall) with acc(unseen σ) ≪ acc(seen σ) at matched m.

**Two measurable splits.**
1. **Natural, on the GSM8K test (descriptive, all arms):** the m-stratified seen − unseen gap. The public comparators are 0.20–0.29 (§1.5).
2. **Constructed and causal, Tier 2 on raw GSM8K:**
   - hold out 473 of the 7,473 templates (seed 19301) from training;
   - **HO-TEMPLATE** = fresh resampled instances of the held-out recomputable templates (about 300 templates × 3 draws);
   - **HO-INST** = fresh resampled instances of 473 training templates, drawn at random;
   - STRUCT-MEM index = HO-INST − HO-TEMPLATE, m-stratified.
   - In GSM8K-Aug the template of an Aug item is not recoverable, so Tier 1 uses split 1 only. This keeps Aug unmodified for the like-for-like SOTA comparison.

### D4. Exposures set the memorisation regime (CONJECTURE, with DATA)

* GPT-2 small's capacity is not the binding limit. Aug's targets carry about 385k × 2.6 equations × ~3 numerals × ~10 bits ≈ 30 Mbit (calc), far below 124M parameters × any per-parameter capacity.
* The binding variable is **exposures per item**:
  - §168 s2 memorised p8 completely at 64 exposures (train 1.000, test 0.490; DATA);
  - nested arithmetic, with fresh items per step, never memorised.
* **CONJECTURE:** recall grows with exposures, and is negligible at ≤ 4.
* **Prediction:** Aug at 4 epochs is in the low-recall regime, so R_ν and R_x matter little there (P1, P2). Raw GSM8K at 103 epochs is in §168's regime, so they matter (P3).
* Literature to verify: the exposure–memorisation scaling (e.g. Allen-Zhu & Li's knowledge-capacity work) before this is cited as more than a conjecture.

### D5. Length: what the α^L account and the p8_ctl data predict for GSM8K

**α^L (DERIVED, A_len).**
* A_len: the trained per-token delta decay stays near its init value, and the closed-loop gain is calibrated on the training-length distribution.
* By SEEDSTAB Part I Thm 4, the open-loop delta map has ‖P‖ ≤ Π_τ α_τ ≈ α^L, and the closed loop must supply ‖Eu‖ ≥ 1 − α^L (§4.3).
* At GSM8K length, 1 − α^L is 0.25 (Aug) to 0.32 (test), against 0.95 on ProsQA. GSM8K sits in the nested-arithmetic regime (n 8–12: 0.16–0.23).
* So **input length is a smaller contraction risk on GSM8K than on ProsQA, where §191 A converged anyway**. P4.2's length risk is not expected to bind.
* The test-vs-train excess of +14.5 tokens multiplies the open-loop factor by α^{14.5} = 0.91. A residual in the top tercile (+33) multiplies it by 0.80.

**Size-keying (CONJECTURE, a transfer from ProsQA DATA).**
* In nested, the per-step accuracy at fixed t falls with n (SEEDSTAB §2.1, DATA). On ProsQA ctl, the latent arm lost **0.188 more** than text when length overstated depth by 3.58 + 0.19 = 3.77 hops (DATA: A 0.950 → 0.640; B 0.938 → 0.816). That is 0.050 per hop of overstatement.
* GSM8K's length is as predictive of depth as ProsQA's (R² 0.26 vs 0.29), so the incentive to key on size is similar.
* **Transfer, assuming one equation ≈ one hop and linearity:**
  - (a) **Set level:** the mean size-keyed error on the test is −0.06, so **no set-level length penalty** is predicted, despite the +10.6-token shift.
  - (b) **Neutral filler (FILLER-N), k tokens of number-free, entity-free prose prepended to each test question:** overstatement = 0.047·k equations, so the excess drop is ≈ 0.050 × 0.047·k: **0.038 at k = 16 and 0.075 at k = 32**.
  - (c) **Natural terciles:** a top − bottom contrast of 1.49 equations predicts a difference-in-differences of ≈ −0.075. Its SE is about 0.037, so this is descriptive.
* **Caveat:** ctl adds *structured* distractor premises. Neutral prose may engage less of a size-keyed path. A numeric distractor (FILLER-D, one irrelevant number sentence; the GSM-NoOp-like analogue of ctl) is reported beside it, descriptively.

### D6. K schedule: fixed K = 10 (primary), and WINDOW's derivation (Lemma W)

**Coverage (DATA).**
* K = 10 = test max m (8) + 2. Every test item has a trained post-completion horizon j₀ = K − m ≥ 2.
* 83% of test items (m ≤ 4) have j₀ ≥ 6. For Aug train it is 93%.
* §191 is the same recipe: K = 10 with depths up to 10, and K′12 helped depths 11–14.
* **C2 note:** the GSM8K test has no stratum deeper than Aug training (Aug reaches m = 13), so this pilot cannot test depth extrapolation.

**Lemma W (PROVED, under the integrator assumption of SEEDSTAB Thm 2(d) / A2).**
* Assumptions:
  - after completion at pass m, the read moves by ‖ΔR_j‖ ≤ d per pass (no fixed point);
  - the answer decoder's margin M(R) is L_D-Lipschitz;
  - training enforces M(R_{m+j}) ≥ μ for each j in a training set J.
* Then M(R_{m+j}) ≥ μ − L_D·d·dist(j, J), so reads are correct for dist(j, J) < μ/(L_D d).
* **Fixed K:** J = {K − m}, a single read time per depth. Deep items get the fewest post-completion passes (j₀ → 0 as m → K). Nothing in the loss constrains d.
* **WINDOW:** K = m + U{1..4}, so J = {1, 2, 3, 4} for *every* m.
  - The loss is minimised only if four consecutive post-completion reads all clear the margin, which bounds the drift: L_D·d ≤ (μ(1) − μ)/3.
  - So the extrapolation horizon scales with the window and is the same for every depth.
* **Under the good class (Thm 3, CASM):** the terminal code is a fixed point and the drift is bounded by η/(1 − κ) for all j. Any CASM with margin minimises the WINDOW objective. An integrator whose drift is harmless only at j₀ does not.
  - So WINDOW **shrinks the set of minimisers toward post-completion-stable maps** (DERIVED).
  - Fixed K constrains post-completion behaviour only for shallow items.
* **Reach is *not* derived.** The step aux already pins step t to pass t. DATA: same-seed DEPTH for WINDOW vs fixed-K §184 is 0.465 / 0.659 vs 0.486 / 0.665, so there is no reach gain. WINDOW is for C6 (§182: STABLE 2 / 2; K′24 0.9985 on s2), not C2.
* **Prediction for fixed K = 10 on GSM8K (CONJECTURE, transferring SEEDSTAB §2.1's horizon map).**
  - The DATA map: error at K′16 is 0.25 / 0.08 / 0.047 / 0.019 / 0.004 at trained horizons j₀ = 0 / 1 / 2 / 3 / 4.
  - For GSM8K: m = 8 items (j₀ = 2) lose about 0.05 at K′16; m ≤ 6 items (j₀ ≥ 4) lose ≤ 0.004.
  - Pooled: **STABLE-12** (K′12 ≥ K′10 − 0.02) and **STABLE-16** (K′16 ≥ K′10 − 0.05).
  - §191 on ProsQA failed this at depths ≥ 8 (K′16 worst 0.471). There j₀ ≤ 2 for 36% of training items (depth ≥ 8); on the GSM8K test, j₀ ≤ 2 only for m ≥ 8 (0.7%), and j₀ ≤ 3 for m ≥ 7 (2.3%).
  - If A fails STABLE-16, the conditional cell E (WINDOW) opens.
* **CSR (§199, a completion-switched step-ratio target)** is the registered, untested alternative for post-completion contraction. It is named, not stacked.

---

## 3. Design

### 3.1 Cells (RTX 3090 for every cell; GPU recorded per cell; seed 1 at stage 1)

| tier | cell | model | data | transform | role |
|---|---|---|---|---|---|
| 1 | **A** `gsm_lat_aug_s1` | latent step-aux (§191 A recipe) | Aug train, filtered | none | the latent method, like-for-like vs published bars |
| 1 | **B** `gsm_lat_augrs_s1` | same | same | R_ν (all) + R_x (the 77% recomputable) per draw | the resampling lever |
| 1 | **C** `gsm_txt_aug_s1` | TEXT-COT (GPT-2, Coconut's GSM format) | same | none | **in-house C3 control** |
| 1 | **D** `gsm_txt_augrs_s1` | TEXT-COT | same | R_ν + R_x | matched resampling control (§191 B's analogue) |
| 2 | **G-FIX** `gsm_lat_g8k_s1` | latent step-aux | raw GSM8K train minus 473 held-out templates | none | the §168 regime (≈ 103 epochs) |
| 2 | **G-RS** `gsm_lat_g8krs_s1` | same | same | R_ν + R_x | the §191 transfer test; HO-TEMPLATE / HO-INST |
| cond. | **E** `gsm_lat_augwin_s1` | A + WINDOW (K = m + U{1..4}, cap 12) | as A | as A | opens only if A fails STABLE-16 |
| cond. | s2, s3 of the carried latent arm, plus C (or D) s2, s3 | | | | open only by the stage-2 rule (R2) |

### 3.2 Recipe (as §191 A; changes listed)

**Unchanged from §191 A:**
* GPT-2 small with DeltaMem-R (6 + 6 heads after block 6, top writes);
* the γ ramp from 2k to 6k, the W_o fade over 500, fast_add, chunk_delta 64, and g_add calibrated at step 0;
* INJECT think passes over [EOT, question] with the memory carried, **K = 10**, `--cache_lower` (no ckpt_passes unless the mem_check requires it);
* bs 64, lr 1e-4, seed 1.

**Aux (step aux):** at pass t ≤ min(m, 10), the read-only shared-trunk decoder emits [EOT, " Step", " t", " :"] + " <<e_t>>" + EOT, teacher-forced, reading M_t.
* λ = 1 throughout (`--aux_fade 24000,24000`).
* The pass token " t" is kept, as in §191. NOPASS is a known alternative (§190) and is not stacked.

**Answer:** [EOT, question, "\n", "### " + answer, EOS], reading M_10.
* Training is CE on the answer + EOS.
* Test is greedy after "### " (≤ 12 tokens, stop at EOS or newline), scored by exact numeric match after bench's `norm_number`.

**Steps:** 24k (4.0 Aug epochs) for Tier 1 and E; 12k for Tier 2 (≈ 103 epochs, §168's regime). Aug-valid accuracy is logged every 2k.
* **STILL-RISING:** valid accuracy rose ≥ 1.0 point per 2k over the last 6k. Any extension goes to the user.

**TEXT-COT (C, D):** question + "\n" + each `<<e_t>>` on its own line + "### " + answer + EOS. Same steps, lr, bs and seed; greedy CoT, then the answer parse.

**Harness:** a new family (e.g. `boxes/gsm8k_stepaux.py`), copying prosqa168s2's ProsMem / aux / textcot paths with a GSM8K data layer and these flags:
* `--relabel_names`, `--resample_numbers` (train only; asserted never on test);
* `--heldout_templates` (Tier 2);
* `--filter_bad_eq`.
* Lazy-import care: keep parameters optional, with an `--eval_only` path.

### 3.3 Evaluation (all cells)

**Headline and checkpoint rule.**
* **Headline:** GSM8K test (1,319), greedy, at K′ = 10.
* **Checkpoint rule (registered): the final checkpoint for every claim.**
  - The checkpoint chosen on Aug valid (500; every 2k) is reported beside it, labelled "valid-selected".
  - **The test is never used for selection.**
  - Published bars may be optimistic from test selection: RiM's 35.5 test-selected vs 33.6 held-out, +1.9 (Literature).
* Only the final checkpoint and the valid-best checkpoint are kept (data retention).

**Latent-only readouts.**
* K′ ∈ {0, 1, 2, 4, 6, 8, 10, 12, 16}, overall and by m.
* aux exact by pass on test, teacher-forced (descriptive).

**Train-fit decomposition (D2), all cells:**
* fit_fix on 1,000 training items (fixed rng);
* fit_rs on fresh resampled instances of the recomputable ones (fixed rng);
* recall share = fit_fix − fit_rs.

**Structure, length and robustness evals.**
* **Structure:**
  - seen / unseen fingerprint accuracy, m-stratified (§1.5 code), against the public comparators;
  - Tier 2 adds HO-TEMPLATE / HO-INST.
* **Length:**
  - FILLER-N at k ∈ {16, 32}: fixed neutral sentences (no digits, no capitalised names) prepended; the same strings for every arm;
  - FILLER-D (one fixed irrelevant numeric sentence), descriptive;
  - length-residual terciles (descriptive).
* **GSM-Symbolic** main / p1 / p2: eval-only, fetched at eval time, never trained on or committed. Accuracy plus the per-template SD over instances. Descriptive.

**Cost (191-1's rule).** Meter FLOPs per test item and timed s/item, on one 3090 for all arms.
* calc: latent K′10 is about 4.6k block-token forwards per item, against text-CoT's ~1.0k (prompt plus ~24 generated tokens, KV-cached), i.e. about 4.6×.
* If accuracy is equal, the wording is "matches, costs more".

### 3.4 Readings (fixed at registration; `theory/score_gsmtpl.py`)

**Thresholds.**
* Unpaired 2 SE at n = 1,319 is 2.7–2.8 points.
* Paired 2 SE is 2.5–3.0 at item discordance 0.20–0.30 (calc), so **±2.8 points** is used throughout.

**R1, C3 in-house (the user's C3 standard).** For A vs C and B vs D, same seed, paired:
* **above** if latent − text ≥ +2.8;
* **matches** if within ±2.8;
* **below** if ≤ −2.8.
* Quoted beside, never in place: our bench CoT 41.8 (the Dilgren checkpoint), CoT-SFT as reported by CODI 44.1, and SIM-CoT's re-run 42.7.

**R2, SOTA (PCCoT 49.48 ± 0.31, 3 runs, its own).**
* Per seed, "above PCCoT" iff accuracy ≥ **52.28** (49.48 + 2.8).
* **Stage 2 opens** iff the carried latent arm's s1 is ≥ 49.48 *and* R1 reads matches or above.
  - The carried arm is A or B, whichever is higher on **Aug valid**, never on test.
  - Stage 2 runs s2, then s3 only if s2 ≥ 46.68, for that arm and its text control, on fresh seeds. That keeps the arm selection independent of the confirmation.
* **SOTA-ABOVE:** 3 / 3 seeds ≥ 52.28.
* **SOTA-MATCH:** 3-seed mean in [46.68, 52.28), with every seed ≥ 46.68.
* **BELOW-SOTA:** otherwise, including when stage 2 does not open.
* Context quoted, not compared: Coconut + SIM-CoT 44.8; LOTUS 44.1 ± 0.7; CODI 43.7; RiM 33.6; AGCLR 34.0.

**R3, RESAMPLE.** B − A and D − C, paired:
* **RS-HELPS** if ≥ +2.8;
* **RS-HURTS** if ≤ −2.8;
* **RS-NEUTRAL** otherwise.
* **LATENT-SPECIFIC** if RS-HELPS for latent and RS-NEUTRAL for text.

**R4, THINK-NECESSARY (C1; §191's form).** A and B at K′ 0 and at K′ 1 are each ≤ 0.5 × their K′ 10 accuracy. Reported beside every "latent ≈ text" statement.

**R5, STRUCTURE.**
* Tier 1: the m-stratified seen − unseen gap, per arm.
* Tier 2:
  - **STRUCT-MEM** if HO-INST − HO-TEMPLATE ≥ 0.15;
  - **COMPOSITIONAL** if ≤ 0.05.

**R6, LENGTH (FILLER-N).** Excess = latent drop − text drop, A vs C.
* **SIZE-KEYED:** excess ≥ 0.04 at k = 32 and ≥ 0.015 at k = 16.
* **LENGTH-ROBUST:** excess ≤ 0.015 at k = 32.
* Otherwise **mixed**.

**R7, STABLE (A).**
* **STABLE-12:** K′12 ≥ K′10 − 0.02.
* **STABLE-16:** K′16 ≥ K′10 − 0.05.
* If STABLE-16 fails, E opens.

**R8, NUM / NAME memorisation (Tier 2, the §191 analogue).**
* **MEM-WAS-BARRIER:** G-RS test ≥ G-FIX test + 2.8, *and* G-FIX recall share ≥ 0.15.
* **NO-MEM:** G-FIX recall share < 0.05.

**Basin entry.** The pinned rule (`theory/basin_entry.py`) on answer_ce, plus final loss, is reported per cell.
* **UNCLEAR-ENTRY (Theory's review, as in §198):** a latent cell whose training loss never enters (entry(0.5) none by its last step) reads UNCLEAR-ENTRY for R1 / R2 / R3, **not** below or BELOW-SOTA.
  - The seed lottery fails about 1 in 3 on nested (§199), so a single-seed non-entry is not evidence against the method.
  - Its follow-up is a fresh seed, which is a new decision (the user if it is above the approved budget).
* Every cross-cell comparison carries the CONFOUNDED-BY-ENTRY flag (|Δentry(0.5)| > 5k steps).
* Single-seed comparisons are same-seed only (note SEED-1).

### 3.5 Predictions (registered with their derivations; falsifiers stated)

| id | prediction | from | falsified if |
|---|---|---|---|
| P1 | A's recall share ≤ 0.05 at 24k | D4 (CONJECTURE) + D2 (DERIVED, A-fit) | recall share > 0.10 |
| P2 | RS-NEUTRAL for both the latent pair (B − A) and the text pair (D − C) on Aug | D1 (PROVED) + D2 + D4 | B − A ≥ +2.8 |
| P3 | G-FIX recall share ≥ 0.30, and G-RS − G-FIX ≥ +2.8 on the GSM8K test | Thm X + D4 (§168: train 1.0 / test 0.49 at 64 exposures) | G-FIX recall share < 0.15, or G-RS ≤ G-FIX |
| P4 | Resampling leaves structure memorisation intact: \|gap(B) − gap(A)\| ≤ 0.05 (natural split); G-RS's HO-INST − HO-TEMPLATE ≥ 0.15 | Prop S (PROVED invariance of Σ_tr) | G-RS's HO gap ≤ 0.05; or gap(B) < gap(A) − 0.05 (resampling then also teaches structure: an augmentation effect) |
| P5 | THINK-NECESSARY holds for A | §191 DATA (K′0 / 1 at chance); GSM8K_REPRO (CODI −20, SIM-CoT + Coconut −40 points without latents) | K′0 ≥ 0.5 × K′10 (the trunk answers directly, as no-CoT's 16–19) |
| P6 | Set level: no length penalty from the +10.6-token shift (the size-keyed error is −0.06); FILLER-N excess at k = 32: 0.075 if size-keyed (bar 0.04) | D5 (α^L DERIVED, A_len; size-keying CONJECTURE transfer) | an excess in the mixed band, 0.015–0.04, falsifies the linear transfer; ≤ 0.015 means LENGTH-ROBUST (the ctl gap was ProsQA-specific distractor structure) |
| P7 | STABLE-12 and STABLE-16 hold for A | D6 (Lemma W + the SEEDSTAB §2.1 horizon map, CONJECTURE transfer) | either fails |
| P8 | GSM-Symbolic main within 3 points of GSM8K test for every arm; B's drop ≤ A's | no test contamination (0 overlap); R_x widens numeric coverage (CONJECTURE) | a drop > 3 points (number-range sensitivity) |

**No derivation predicts SOTA.** On priors (published CoT 41.8–44.1; best latent 49.48), BELOW-SOTA is the more likely outcome. The stage-2 rule prices that in.

### 3.6 Gates (before launch; Audit's diff review on all)

**Data layer.**
* Parse round-trip on 10k Aug items: the questions, equations and answers re-render identically.
* Filter counts are logged; the 0.31% wrong / 0.13% unparsed reproduce.

**Transforms.**
* R_ν:
  - injective and consistent;
  - the answer and equations are unchanged on 10k items;
  - names only from the whitelist pool (no places or brands);
  - the gender match is logged.
* R_x:
  - the identity (x′ = x) reproduces every equation exactly;
  - resampled equations evaluate exactly;
  - the acceptance rate and the median answer max-probability (H∞) are logged.
* Test items are asserted never transformed.

**Model and memory.**
* An fp64 cached-vs-uncached identity on a GSM8K batch (the `--cache_lower` exactness rule).
* Gradflow: nonzero answer-CE and aux-CE gradients per term.
* Plain-path smoke: construct + 1 step with no new flags.
* mem_check at the longest Aug item (214 tokens, m 13) with K = 10 and the aux.

**Contamination and timing.**
* **Near-duplicate scan:** 8-gram shingles of GSM8K test vs all of Aug train (gsmlen/dedupe.py's method). Flagged items are listed, and the headline is reported with and without them.
* `--time_steps 200` on a 3090 for one latent and one text cell. The projection is restated to Audit and **to the user before launch**.

### 3.7 Cost, budget and the OC block (`oc_cost.py` → `oc_cost.json`)

**Cost model (calc, calibrated on realised runs).**
* Training cost = 3 × forward block-token units: cached lower blocks once, upper blocks × K passes, aux decoders for t ≤ m, the answer decoder, and checkpointing's recompute where used.
* **Calibration:** §191 A, realised: 133k units per item → **8.19·10⁻⁷ s per unit** (compute-bound: bs 32 / 64 = 3.50 / 6.96 s).
* **Overhead:** nested RMS realised 0.885 s/step against 0.598 predicted from units → **+0.287 s/step** of launch overhead at short L.

| cell | units per item | s/step (compute only / central / high ×1.5) |
|---|---|---|
| latent, Aug, K 10, cached (A, B, E) | 11.5k | 0.60 / **0.89** / 1.34 |
| latent with ckpt_passes, if the mem_check needs it | 14.1k | 0.74 / 1.03 / 1.54 |
| TEXT-COT, Aug (C, D) | 2.4k | 0.13 / 0.22 / 0.33 |

* The model under-predicts realised ProsQA TEXT-COT by 20% (1.08 vs 1.35), so text cells are costed at 0.28 / 0.42.

**Per-cell $** (a projection; replaced by realised $ once each cell lands, 198-2a). $0.227/h (the §191 host class) + $0.10 per host + eval time.

| cell | steps | hours (central) | $ central / high |
|---|---|---|---|
| latent (A, B, E) | 24k | 5.9 | **1.54 / 2.26** |
| text (C, D) | 24k | 1.9 | **0.60 / 0.90** |
| latent, Tier 2 (G-FIX, G-RS) | 12k | 3.0 | **0.86 / 1.25** |

* For scale: §191 A's realised training was $5.29 for 12k ProsQA steps.

**Budget.**

| block | central | high |
|---|---|---|
| stage 1, Tier 1 (A + B + C + D) | $4.28 | $6.32 |
| stage 1, Tier 2 (G-FIX + G-RS) | $1.72 | $2.50 |
| --time_steps gates | $0.15 | $0.20 |
| **Stage 1 total** | **$6.2** | **$9.0** |
| stage 2 (carried latent s2, s3 + text s2, s3) | +$4.28 | +$6.32 |
| E (WINDOW), conditional | +$1.54 | +$2.26 |
| **Maximum** | **$12.0** | **$17.6** |

* Tier 2 can be dropped without changing any Tier 1 reading (−$1.7 to −$2.5).

**OC block (R2's SOTA claim).**
* p₀ / p₁ are framed on the true seed-mean accuracy μ, since the endpoint is an accuracy bar.
* Per seed, p(μ) = Φ((μ − 52.28)/√(σ_s² + 1.38²)), with σ_s ∈ [0.5, 1.5]:
  - PCCoT's ± is 0.31;
  - LOTUS's is 0.7;
  - Coconut's is 1.5.
* **Design.**
  - Stage 2 opens when s1 ≥ 49.48 and R1 reads matches or above.
  - s2 then runs (with its text control).
  - s3 runs only if s2 ≥ 46.68, since below that both ABOVE and MATCH are already impossible.
  - **SOTA-ABOVE = all 3 seeds ≥ 52.28.** Seeds 2 and 3 are fresh. One GPU class (3090).
  - A stage-2 seed pair costs $2.14 (high $3.16).

| true μ | P(open) | p(seed ≥ 52.28) | **P(SOTA-ABOVE)** | E[stage-2 $], central (high) |
|---|---|---|---|---|
| 45.0 (about the prior) | 0.001–0.014 | ≤ 0.0002 | ≈ 0 | $0.00–0.04 (0.05) |
| 49.48 (PCCoT level: the null) | 0.50 | 0.028–0.085 | **≤ 0.0006** | $2.1 (3.1) |
| 52.28 (at the bar) | 0.92–0.97 | 0.50 | 0.125 | $3.9–4.2 (6.1) |
| 53.5 | 0.98–1.0 | 0.73–0.80 | 0.38–0.51 | $4.2–4.3 (6.3) |
| 55.0 | ≈ 1.0 | 0.91–0.97 | **0.75–0.91** | $4.3 (6.3) |

* Ranges are over σ_s = 0.5–1.5. The opening probability ignores the R1 condition, so its costs are upper bounds.

* **seedstats.py (exact):**
  - all-of-2: size 0.25 at p 0.5, power 0.81 at p 0.9;
  - all-of-3: 0.125 / 0.729;
  - exact 95% CIs: 2 / 2 → 0.158–1.0; 3 / 3 → 0.292–1.0.
* So the 3 / 3 rule is used, and the claim is worded as "3 / 3 seeds above PCCoT + 2.8", not as a rate.
* **Under the prior** (μ ≈ 42–45): stage 2 opens with P ≤ 0.001–0.02, so **E[$] ≈ stage 1 ≈ $6.2–9.0.**

---

## 4. Risks, and what would falsify the approach

**Risks.**
1. **Undertraining.** 4 Aug epochs is fewer than the published runs; Literature should check PCCoT's and CODI's epochs. A BELOW-SOTA reading with STILL-RISING is "undertrained", not "method fails". Extending to 48k is a user decision (+$1.4–2.0 per cell).
2. **Aux noise.** Wrong equations are filtered (0.77% of items). Rounded equations are kept (1.7%). The answer ≠ last equation in 4.4% of items, versus 6.3% in GSM8K itself.
3. **The resampling shift.** Resampled magnitudes can change difficulty, which A-fit covers. Gate: digit-count and integrality preservation, the acceptance rate, and TEXT-COT's fit_fix ≈ fit_rs within 0.03.
4. **Name heuristic noise:**
   - the candidates include places and brands, so the whitelist pool matters;
   - a gender mismatch is a surface oddity only (the answer is unchanged);
   - mitigation: a gender-matched pool.
5. **The structure confound in R5's natural split:** unseen structures are rarer and wordier. Tier 2's constructed split is the causal one.
6. **Seed lottery and GPU class.** One seed at stage 1, on one GPU class. Basin entry is reported, and single-seed contrasts are same-seed only.
7. **Pass-token aux.** Tokens " 7".." 10" are rarely trained (m ≥ 7 is 0.8% of Aug). The decoder is training-only, but the token pinned the done signal in nested (§187 / §190). NOPASS is not stacked.
8. **Contamination.** There is no verbatim test overlap. The near-duplicate scan is a gate. GSM-Symbolic is test-derived, so eval-only.
9. **The cost projection.** Its band is ×1.5, and --time_steps comes first. GSM8K is short-L, so it may be launch-bound (the nested regime). The central figure already includes the measured overhead.
10. **The ladder.** Whether §191 (matches text on ProsQA-deep) clears "beat SOTA on smaller tasks first" is the user's call.

**What would falsify the approach.**
* **Resampling as a GSM8K lever:**
  - P1 holds (recall share ≤ 0.05) together with RS-NEUTRAL (P2). Aug's scale already removes recall, and the lever is irrelevant at like-for-like scale.
  - P3 fails (G-RS ≤ G-FIX with G-FIX recall share ≥ 0.15). §191's mechanism does not transfer to arithmetic word problems.
* **The latent method on GSM8K:**
  - R4 fails: the passes don't carry the computation (cf. ProsQA's 0.5 at K′0).
  - R1 reads below: no C3.
  - R5 shows a seen − unseen gap above text-CoT's 0.25 by ≥ 0.05: a stronger reliance on structure retrieval than text.
* **The theory:**
  - P4 fails: structure is not invariant in effect, i.e. resampling teaches structure. That refutes Prop S's practical reading, not its proof.
  - P6 lands in the mixed band: the linear size-keying transfer is wrong.
  - P7 fails at m ≤ 6 (j₀ ≥ 4): the horizon map does not transfer.

---

## 5. Prior-art distinctions (Literature to confirm wording)

* **RiM 2605.30343** (Aichberger & Hochreiter): fixed `<m>` memory blocks in one forward pass, with no recurrence.
  - Its Stage 1 (per-block step targets) and Stage 2 (per-block answer targets) are **close to our STEP-AUX → answer schedule**.
  - Ours differs: a persistent fast-weight memory across K re-reading passes, and a removal test (K′ 0 / 1). RiM has none: no C1 evidence.
  - GPT-2: 33.6 (LoRA, held-out selection) / 35.5 (test-selected). **A GSM8K competitor.**
* **AGCLR 2606.07720:** a persistent gated memory over Coconut passes, no step aux; 34.0.
* **SIM-CoT 2509.20317:** step-level aux on latent tokens; Coconut + SIM-CoT 44.8.
* **PCCoT 2506.18582:** 49.48 ± 0.31 (the bar).
* **TemplateGSM 2411.18104:** static template instances from GSM8K train, not used for latent models.
* **GSM-Symbolic 2410.05229:** test-derived and eval-only.
* **Literature found no per-draw template resampling for latent or implicit-CoT models** (LIT_GSM_TEMPLATES).

---

## 6. Corrections to premises in the brief (for Theory)

1. **"§191 A trained uncached":** wrong. args.json has `cache_lower: true` (and `ckpt_passes`, `chunk_delta 64`, micro 4). The 6.99 s/step anchor is a cached anchor.
2. **"GSM8K is longer":** wrong. Measured, it is 7.6× shorter than ProsQA (57.8 vs 440 tokens), so a GSM8K latent cell costs about 8× less per step than §191 A.
3. **"p8_ctl is an input-length control":** misleading. Its lengths are in-support (0% above the train max). It is a length–depth mismatch (+3.6 hops). The §191 result line should say so.
4. **GSM8K-Aug train ≠ the GSM8K test distribution:** it is shorter (43 vs 58) and shallower (m 2.60 vs 3.25). Aug valid matches the test (56 tokens), so valid-based selection is reasonable.

## 7. Files (this folder)

**Scripts:**
* `measure_lengths.py` → `lengths.json`;
* `audit_aug.py` → `aug_audit.json`;
* `extra_stats.py` → `extra_stats.json`;
* `seen_split.py` → `seen_split.json` (post hoc);
* `tgsm_sample.py` → `tgsm_sample.json` (raw API JSON in `dl_tgsm/`; untrusted, never executed);
* `oc_cost.py` → `oc_cost.json` (imports `genlat/tools/seedstats.py` read-only).

**Rerun:** `python -I <script>` from this folder. Each takes ≤ 4 min on CPU at idle priority.
