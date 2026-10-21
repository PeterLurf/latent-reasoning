# PORT-STATE: latents that carry the remaining program, on a small word-problem task (DRAFT v0, 2026-10-21, not registered)

**Origin.** Audit's item 2, relaying the user's 10-21 go-ahead (priority C1 on language, small tasks first). Built on Note 221-2 (descriptive, floors cleared): CODI-GPT2's GSM8K latents are **carriers of intermediate values**. The **program and remaining operands are re-read from the question** (QMASK → 0; INT-FOLLOW 0.33–0.54 vs floor ≤ 0.04), so a donor's latents yield the counterfactual, not the donor's answer (§221 CONTEXT-BOUND).

**Goal.** A latent state that carries the **remaining program and its operands**, so that a mid-chain swap makes the model finish the **donor's** problem (portable, question-independent state), on a word task whose depth exceeds one forward pass by construction. Read with §221's battery so the result is comparable.

## 0. What failed before, and the constraint each failure sets
- **§208 (question once + answer bottleneck, GSM8K):** PASS1-PRECOMPUTE. GPT-2-solvable GSM8K items are shallow, so pass 1 does everything. → **C-a:** the task must need more than one pass (depth n with one-pass accuracy at chance for n ≥ 3–4, measured).
- **§204 / §206 (per-pass targets / CoT coupling, GSM8K):** per-pass targets were met by pass-1 precompute or by the decoder. → **C-b:** per-step targets must be read by a decoder that **cannot see the question**, and each target must be a **state** (the re-rooted problem), not a value.
- **§221 / 221-2:** latents carry values while the program comes from the question. → **C-c:** the target is the **remaining program with substituted values** (the re-rooted problem text R_t), and the registered reading is a **mid-chain** swap (the continuation must follow the donor's program).
- **§199 CSR (Goodhart):** one-sided dynamics targets get gamed. → **C-d:** no target on internal dynamics; only text targets with exact answers.

## 1. Task: NESTED-WORDS (small, templated language, exact answers, depth-controlled)
- A **straight-line word program** of n steps over named quantities, values mod 11 (§170's arithmetic), for example: *"Ann has 3 apples. Bob has 4 more than Ann. Cara has twice as many as Bob. Dan has 5 fewer than Cara. How many does Dan have?"*
  - Each step reads one or two earlier quantities or constants, so the dependency DAG has depth up to n.
  - Names, verbs and sentence order are sampled. Distractor sentences are optional, descriptive only.
- **R_t (the re-rooted problem):** the problem text after the first t dependency-ordered steps are computed. Computed quantities become constants: *"Bob has 7 apples. Cara has twice as many as Bob. …"*. R_t is unique given a fixed reduction order (topological, innermost-first as §170). R_n = the answer.
- Depth n ∈ 1..8 for training (curriculum as CurGate), n 9–12 for extrapolation (descriptive).
- **Generator gates (G-W0):** one-pass DIRECT accuracy at n ≥ 3 below a pre-stated level (to measure); no surface shortcut (heuristic panel as §218 G-L0); a fixed held-out test set with a sha.

## 2. Model and objective (the method change)
- **Base:** GPT-2 small, CODI-style continuous latent loop (boxes/codi_port.py: z_{j+1} = prj(h_last), K latent steps). The question is **visible to every latent step**, as CODI. K ≥ n (K = n + U{0..2}, capped).
- **(M1) STATE-RECODE:** a **question-blind** side decoder D (a small GPT-2 head that sees only z_t, or the latent rows' KV) is trained to emit R_t from latent t, with teacher-forced CE.
  - Per-step targets come from the generator's own reduction. On real language they would come from CoT-derived R_t, which is C4-legal under the user's "distillation from CoT" standard.
- **(M2) ANSWER BOTTLENECK:** the answer phase cannot attend to question **content** tokens (CODI's QMASK, trained). It reads the latents (and BOS / markers) only.
- **(M3) none of:** question-once, interchange training, dynamics targets. These are not in v0; IIT-style swap training failed in §205 and is held for a later design.
- **Loss:** CE(answer) + λ · Σ_t CE(D(z_t) → R_t). λ is fixed by the derivation (§3), not tuned on readings.

## 3. Derivation sketch (to be completed with proofs by an Opus subagent)
- **D1 (what the swap tests).** With M2, the final answer is a function of the latents alone. A **full** latent swap (CACHE / STATE-SWAP at all positions) then follows the donor **trivially**, so full-swap follow is not evidence. The registered reading must be the **mid-chain** swap PREFIX-SWAP(t): donor latents 1..t, then steps t+1..K run with the **recipient's question visible**.
  - Donor-final follow requires steps t+1..K to compute from the state's program, not re-read the recipient's question.
  - Counterfactual follow (the recipient's program on the donor's values) means "carriers" as in 221-2.
  - Own means the state is ignored.
- **D2 (why M1 makes donor-follow possible but not trivial).** M1 makes the program **decodable** from z_t (the state contains R_t). It does not force the continuation to **use** it. The question is still visible and in training always agrees with the state, so nothing in the loss prefers state over question.
  - Decodable-but-unused is the §203 lesson. That is exactly what the registered reading measures, and why a null is informative.
  - **Prediction [ASSUMED]:** donor-final follow at mid-chain t ≥ 0.30 with P ≈ 0.3. Counterfactual is the likely alternative.
- **D3 (precompute defence).** C-a: the depth makes a one-pass answer at n ≥ 3 near chance (to be measured, G-W0).
  - M1's per-step R_t targets are read by a question-blind decoder, so z_1 must encode R_1 (one step done), not R_n. Encoding the answer as well is possible but cannot satisfy R_1's CE alone.
  - The serial readings (STALL, ONSET of R_t decodability with t, PROBE-GAIN) detect a pass-1 precompute.
- **D4 (Goodhart analysis).**
  - (i) D could memorise R_t from item statistics without z: guarded by D seeing only z, and by a **z-shuffled control** (D on a random item's z → chance).
  - (ii) R_t could be encoded but stored in a way the continuation ignores: measured by D2's reading, not hidden.
  - (iii) The answer bottleneck could leak the question through BOS / markers: a QMASK leak test (question-content KV removed → identical outputs).
  - (iv) The state could carry R_t only at the final latent: STATE-ONSET (decodability of R_t at t).
- **D5 (cost).** About one or two 3090 cells at 24k (CODI port ≈ 2 s/step at bs 64?), ≈ $3–6 per cell. A DIRECT baseline cell plus the PORT-STATE cell, and an M1-only ablation (no M2) as the attribution control. Cap ≤ $30.

## 4. Readings (§221 battery as registered, plus serial)
- **Primary, PORT-FOLLOW:** PREFIX-SWAP(t) donor-final follow at t = ⌈n/2⌉ on exact-length pairs at matched n (n 4–8), ≥ floor + 0.30, with the leak control (LEAK-SEG ≤ 0.25) and SELF-SWAP identity (§221 machinery).
- **Secondary / descriptive:**
  - full CACHE / STATE-SWAP follow (expected trivial under M2: a sanity check, not evidence);
  - INT-FOLLOW (counterfactual);
  - QMASK / QDROP;
  - STATE-ONSET; STALL; LOGIT-LENS;
  - the K-sweep (C2 analogue);
  - n 9–12 extrapolation.
- Per-item outputs stored.
- **Wording:** "portable latent state on a templated word task (one seed)". Never GSM8K, never C4.

## 5. Feasibility ($0, before v1)
- (F1) The generator and G-W0 shortcut panel.
- (F2) A short local DIRECT run (8 GB, short steps) giving one-pass accuracy by n, to set C-a's depth band.
- (F3) An untrained-model known-answer check of the swap machinery on the new task, in the §221 harness: SELF-SWAP bitwise; a stub model whose answer phase reads only latents gives full-swap follow = 1.

## 6. Open (for the proof check / red team)
1. Should the primary be PREFIX-SWAP at t = ⌈n/2⌉, or pooled over t ∈ [2, n−1] with a slope?
2. Should M2 also mask the question from latent steps after t₀ (partial question-once), or keep it visible (the cleaner test of "uses state despite access")?
3. λ: fix it from the CE scale ratio, or from a known-answer stub?
4. Is CODI's 6-latent architecture the right base, or §218's LangSLVA? Keep CODI for comparability with §221.
5. Wouldn't the M1-only ablation (no M2) be the cleaner attribution? M2 alone is §208 on a deep task.
