# REVIEW_LB: red team of LB_draft.md (9c38f364) (2026-10-24 00:48 EDT)

Reviewer: Theory's red-team subagent. Read-only on the repo, apart from this file and `review_scripts/` (CPU, ≤ 2 threads, $0). No blinded file was read. Qwen2.5-0.5B weights (988 MB, pinned revision 060db649) were downloaded to the scratchpad for check Q and then deleted.
- **Inputs:** LB_draft.md plus its five scripts and JSONs; THEORY Notes 226-2, 227-2, 228-1, 228-2 and 229-1 (grep only); LAT_v1.md, LAT_build_s228.md and REVIEW_LAT.md; boxes/deltamem.py (init, write_seq, read), boxes/inject_ow.py (loss_items, the aux value index), boxes/nested170.py InjectN; NORTH_STAR C1 / C4; Literature's LIT_LARGEBASE_CHECK_2026-10-24.md.
- **Checks (review_scripts/):**
  - `rv_lb_trained_probe.py` → `.json` (T): the trained §228 model, `s228_plat2_s1/model.pt`. Is v_1 present in the write source u and in S_1?
  - `rv_lb_deeplinear.py` → `.json` (D): D2's escape law under DeltaMem's init.
  - `rv_lb_qwen_checks.py` → `.json` (Q): weight RMS; memory-layer geometry at init, Qwen vs GPT-2; GQA expansion; tokenizer; answer distribution.
  - `rv_lb_oc_spend.py` → `.json` (O): LP3 OC where the applicability rule bites; expected spend under the fixes.
  - The draft's arithmetic reproduces: lb_oc spend tree, OC values, RMS ratio 0.1597 → 1.6e-5, µP 8.6e-5, and the cost-model calibration.

## Verdict: REVISE. Not STOP; no new red team needed if fixes 1–9 are applied as written.
- **The premise survives in a corrected form.** Check T, on the trained §228 model, locates the GPT-2-small failure. On d = 2 items the think pass never computes v_1: v_1 is at chance in the write source u and therefore in S_1. On d = 1 items, where v_1 is the asked-for answer, it is present.
  - So the bottleneck is the eager computation of an unasked intermediate, not storage and not name→value lookup.
  - Scale could plausibly fix that. The draft's predictor (OPD lookup), its gate (LP2 pooled over d) and its derivation (a sigmoidal escape law) all point at the wrong quantity.
- **A1 is cheap (≈ $1.6–1.8)** and still worth running as the single-factor test the user asked for. It should run only after it is re-aimed: the predictor on EAGER at d ≥ 2 at the write source, LP2 read on d = 2 items, and an lr bracket or an external anchor.

## Issues, ranked by severity

**1. [HIGH] LP2 pooled over d ∈ {1, 2} can pass without step-1 storage; the GPT-2 failure is d-specific.**
- **Check T** (§228 trained model; 3,000 fresh TOPO items, d 1 / 2 alternating; the model's own teacher-forced pass-1 read): aux pass-1 value acc is **0.51 pooled (reproduces 228-2), 0.77 on d = 1, 0.25 on d = 2**.
- Linear probes for v_1, by depth (test n 375 per d; majority 0.12):

| features | d = 1 | d = 2 | fit on d = 2 only |
|---|---|---|---|
| u at the end of S_1's sentence (U_EAGER) | 0.53 | 0.16 | 0.14 |
| u at the last token (U_LAST) | 0.60 | 0.11 | 0.12 |
| S_1, all heads (random projection, 1,536 features) | 0.55 | 0.16 | 0.13 |

- On d = 2 items, v_1 is not in the trained write source, so S_1 cannot hold it (S_1 ≈ u in decodability at both depths). Where v_1 is present (d = 1, the question's target), S_1 keeps it.
- Under the draft's LP2, a Qwen run with d = 1 at 1.0 and d = 2 at 0.60 passes (pooled 0.80) while carrying no intermediate.
- **Fix:**
  - LP2 = pass-1 aux value acc **on d = 2 items** ≥ 0.80; d = 1 descriptive.
  - In A2 / Stage B, the pass-t value acc is read on items with d > t.
  - This needs a by-(d, pass) log key: `aux_vacc_by_d_pass`, a logging-only addition, frozen before launch and in the G-B7 diff.
  - The A2 futility rule uses the same d > 1 subset.

**2. [HIGH] D1–D3: the escape-time law does not hold at DeltaMem's init. The surviving claim is presence, and the probe aims at the wrong target.**
- D2 cites Saxe's τ ≈ ln(s/a₀)/(ηs) for small balanced init. DeltaMem has W1 = O(1) (the copied k / v) and W2 = wo = 0 exactly: a₀ = 0, so the formula is undefined, and the dynamics are a near-convex regression for wo.
- **Check D** (2-layer linear, s = 3 / 1 / 0.3 / 0.1, steps to reach half strength):

| init and optimiser | half-strength steps (s = 3 / 1 / 0.3 / 0.1) | τ·s |
|---|---|---|
| Saxe init, GD | 60 / 160 / 500 / 1,370 | 137–180, constant (the law holds) |
| WO0 (DeltaMem), GD | 10 / 20 / 20 / 20 | — |
| WO0 (DeltaMem), Adam | 50 / 60 / 60 / 70 | — |

  - With WO0 the ordering by s disappears. The draft's own 8k-step failure cannot be a slow escape.
- **What survives:** if v_1 is not linearly present in the read features (s ≈ 0), the linear path cannot fit it at any speed. That is a presence statement, and check T confirms it on GPT-2 small, but for EAGER at d = 2, not for OPD.
- The draft's primary predictor P1 is OPD (S_0's value at its mention, a copy/lookup). EAGER is only descriptive, and the probe picks any layer of the HF hidden states (the last of which is post-final-norm).
- **Fix:**
  - (i) Rewrite D2 as "[DERIVED] presence is necessary for the linear path; [ASSUMED] presence in the frozen base predicts presence after FT".
  - (ii) Drop the τ law and D5's "rate unchanged" (issue 5).
  - (iii) **P1′ = EAGER on d = 2 items**, using the actual write sources: RMSNorm₁₂(h₁₂) and h₂₄ before the final norm. Add d = 3 items for v_1 and v_2 with no layer search.
  - (iv) **Make P1′ a launch condition for A1:** if Qwen's frozen P1′ ≤ 0.30 at both write sources, D3 predicts an LP2 fail. Do not spend; report to the user with issue 7's alternative.
  - (v) State the coupling: high EAGER-d2 also predicts that DIRECT-Q partly solves d = 2 (band shift; issue 6).
  - Tag the linearisation [ASSUMED] (Literature N2).

**3. [HIGH] The trunk lr is one Adam-sign derivation with no anchor, and an LP2 fail closes the line.**
- RMS ratio: 0.1597 (Q-A reproduces). At 1e-4, Qwen's relative update is 6.26× GPT-2's. That risks drift of the very features D3 relies on.
- At 1.6e-5 the trunk may not learn in 8k steps to compute intermediates. GPT-2's trunk at 1e-4 did not either, so slower is not obviously safer for the eager feature the trunk must build.
- The relative-update rule also ignores depth: 24 vs 12 layers roughly doubles the accumulated perturbation per step.
- Literature: Shapiro 2608.11233 retrofits a loop onto **this exact base**. That is the best external anchor available.
- **Fix:**
  - (a) Literature extracts Shapiro's trunk lr, optimiser, steps and full FT vs adapters. If it is within 2× of 1.6e-5, keep 1.6e-5 and cite it.
  - (b) Otherwise, A1 adds a second LAT arm at 8.6e-5 (µP width) on the same host: +$1.22 point / $1.58 guard. **LB closes on LP2 only if both arms fail.** It is pre-registered, so it is not a retune.
  - (c) Save the final DIRECT-Q and LAT-Q model.pt. Re-run P1′ on them post hoc ($0) as a feature-retention check: drift vs failure to build. Keep both checkpoints until read.

**4. [HIGH] GQA expansion collapses the memory to 2 distinct heads at init. Qwen's copied keys are also less selective than GPT-2's.**
- Q-C: with 7 + 7 heads and repeat-interleave 7×, additive heads 0–6 all copy kv head 0 and delta heads 7–13 all copy kv head 1.
  - **Distinct k/v parameter sets: 1 per type.** The gate weight is 0, so α and β are identical too.
  - At step 0 the state is two distinct 64×64 matrices, each replicated 7×, vs GPT-2's 12 distinct heads. Symmetry breaks only through q and wo once wo ≠ 0.
- Q-B geometry, Qwen L12 vs GPT-2 L6:

| quantity | Qwen L12 | GPT-2 L6 |
|---|---|---|
| key-bias share ‖b_k‖ / ‖k‖ | **0.51** | 0.05 |
| mean pairwise cosine of normalised delta keys | 0.81 | 0.70 |
| mean pairwise cosine of φ(k), additive heads | 0.92 | 0.87 |

  - Qwen's copied q/k were also trained under RoPE and are used here without rotation.
  - "Init from the layer's attention" is functionally faithful in GPT-2 and much less so in Qwen.
- **Fix:**
  - Per-head random orthogonal rotations R_h ∈ O(64) of the copied k and v, with R_0 = I. This preserves norms and in-head geometry and breaks the replication.
  - Disclose it as deviation L6b.
  - New gate G-B9: at step 0, pairwise state correlation between heads < 0.9; and after 200 steps, no head pair > 0.99.
  - Keep the 7 / 7 split and L/2.

**5. [MED] D5 ("memory-side rate unchanged") is false at the read.**
- Q-B: residual norm entering the memory layer is 15.0 (Qwen) vs 81.6 (GPT-2).
- The g_add-calibrated self-read ‖o‖/‖h‖ is **0.37 vs 0.14**. At the same 1e-4 on wo, each memory step perturbs Qwen's residual stream ≈ 2.6× more, relative to its size, while the trunk moves 6.3× less.
- **Fix:**
  - Delete D5's claim.
  - Either scale wo's lr by 0.14 / 0.37 (→ ≈ 3.8e-5; [DERIVED] by the same relative-update logic as L7) or keep 1e-4 and disclose it.
  - Log ‖fade·wo(o)‖/‖h‖ per 100 steps (logging only), so that a read swamping the d = 1 path is visible.

**6. [MED] The band can vanish at ≤ $30, and the LP3 applicability cut wastes a readable range.**
- Check O: LP3 (LAT ≥ 0.70, LAT ≥ DIR + 0.30) stays powered well above DIR 0.40:

| LAT | DIR 0.55 | DIR 0.60 | DIR 0.65 |
|---|---|---|---|
| 0.90 | P(pass) 0.997 | 0.51 | — |
| 0.95 | — | 0.999 | 0.51 |

- The draft's N/A threshold of 0.40 discards a binding test exactly where Qwen is likely to land.
- In Stage B, B ⊆ {2..6} needs ≥ 3 depths with ≥ 2 of them ≥ 4. If DIRECT-Q learns d ≤ 3, B ⊆ {4, 5, 6}, with no slack: one TEXT-Q miss at d 6 kills it.
  - §227 needed ≈ 6k post-introduction steps for d = 2, and TEXT-Q stops at 22k (d 6 gets ≥ 4k).
- **Fix:**
  - LP3 is N/A only if DIRECT-Q d = 2 > 0.55.
  - Pre-state: **if A2's DIRECT-Q d = 3 > 0.50, Stage B is not launched.** The d ≤ 8 design then goes to the user, since U{1..6} cannot form a band with margin.
  - TEXT-Q to 30k: +$0.40 point / $0.52 guard (O: 30k = $1.50 / $1.95).
  - TEXT-Q's lr: keep 1.6e-5, but disclose that a TEXT fail could be an lr artefact. If TEXT fails while LAT passes A1/A2, one TEXT re-run at 8.6e-5 is allowed before "band undefined".

**7. [MED] Arm choice: keep INJECT for A1, but name the better-justified alternative now.**
- Single-factor logic favours INJECT, and A1 is cheap. But check T shows that what INJECT needs at pass 1 is the trunk computing a step that the think input never asks for. The think input is identical in every pass, so nothing tells pass t which step to compute; the aux names S_t only at read time.
- That is an architecture property, and scale fixes it only if the base computes intermediates eagerly (P1′).
- Shapiro 2608.11233 (a loop retrofit on Qwen2.5-0.5B) and CODI-Llama-1B show that other arms are feasible at this scale. They also make a positive LB result narrow (Literature N1 NARROW).
- **Fix:**
  - Keep INJECT.
  - Pre-state in §6: "LB closes on A1 → the next registration is an arm whose pass t is told its step (a pass-cued think input, which drops cache_lower), or a loop retrofit following Shapiro, first on GPT-2 small at ≈ $0.6, where the mechanism question is cheapest". Not a fallback inside LB.
  - Close wording: "the §184 INJECT recipe at Qwen2.5-0.5B". A null does not close "too small a base" beyond 0.5B. Drop "closes the too-small-a-base explanation".

**8. [MED] The $30 line: the staged plan must be the whole design, not the first half of a split.**
- The house rule (10-17) says a design is never split to stay under the limit.
- The draft keeps "Stage B (Audit) or the full Stage 1 (user)" plus an over-cap second seed as continuations. That is a split.
- **Fix:** register LB with **Stage B as its terminal cell.** Family cap = staged guard with fixes **$24.42** (O; point $18.79), rounded to a $25 cap, which is Audit-approvable.
  - The U{1..8} / 48k design and any second seed are outside LB. They are a fresh user decision made with A2 data, disclosed now.
  - Say in one line to the user now that the full design is $30.53 at guard (over the line) and that LB runs the staged design unless they prefer the full one.
  - State whether "$30 total" counts §228 / §229's $1.19 (it should not; different family).

**9. [MED] Build gates miss four traps.**
- **(a) The aux value index.** inject_ow's `vidx = q.index(" has") + 1` would point to Qwen's standalone "Ġ" token, since "Kim has 10" → [Ġhas, Ġ, 1, 0] (Q-D). LP2 would then read a near-always-correct space token and pass spuriously.
  - **G-B8b:** on hand-built aux targets, vidx covers exactly the digit tokens; on a random-output model, LP2 ≈ chance.
- **(b) Editing boxes/deltamem.py:** the draft adds a dispatch there. It is lazily imported by live §230 jobs (vbank.py / vbank_diag.py import it). That is the lazy-import skew and dirty-tree-deploy lessons.
  - Put the Qwen read / write / lower paths in `trunk_qwen.py` (or `deltamem_q.py`) and leave deltamem.py byte-identical. G-B0 becomes "deltamem sha unchanged".
- **(c) The two-group optimiser** is a new make_opt path. It needs flags-off bitwise (one group when `--lr_mem` is unset).
- **(d) The answer CE changes.** §228's CE is full-vocab at the number token; the draft's is softmax₁₁. That is an unlisted loss change, not "reduces to the existing read" (true only for accuracy).
  - Use the sequence NLL: −log q₁(first digit), with −log q₂("0"|"1") for class 10 and −log(1 − q₂("0"|"1")) for class 1. Keep softmax₁₁ for acc only. G-B8 checks both.

**10. [LOW] The prefix-factorised read itself is sound.**
- Causal, read-only decoder, no write: no leak into q₁. Q-D confirms ": 10" → [:, Ġ, 1, 0] and the aux prompt [ĠStep, Ġ, 1, Ġ:].
- "Chance = 1/11" is not exact: answer 0 never occurs at d = 1 (0 / 4,000) and is 7 % at d = 2, so majority is 0.107 / 0.100. Applies equally to §228; reword as "majority ≈ 0.10".

**11. [LOW] Cost.**
- The model reproduces. Compute ratio ≈ 4.4× GPT-2 (358M vs 85M non-embedding params) matches A1 LAT compute 2.02 s/step.
- **Not modelled:**
  - (i) Stage-B eval parts (TEST2 K′ 0..14, mean-state, swaps with floors, fp32, at 4× the model). The +15 % is a pilot-eval allowance. F2 must time one Stage-B eval batch.
  - (ii) Qwen full-state checkpoints are ≈ 7.4 GB each (16 B/param), and off-host mirrors carry 7.4 GB per copy. Set ckpt / mirror cadence in F2 and count the transfer time.
  - (iii) Wall clock: Stage B LAT ≈ 42 h on one 3090 (lb_cost); state it for the guard marks.
- **Expected spend:** $3.93 is the right arithmetic for the draft's priors, but it ignores the applicability branch.
  - Counting it (P(N/A) 0.3): $4.38 / $5.69. With fixes 3b + 6: **$5.71 point / $7.42 guard.** With LP2 on d = 2 (prior ≈ 0.25): $3.33 / $4.32.
  - It is honest to within ±$2. The approval number is the cap (fix 8), not the expectation.

**12. [LOW] Censoring and single-change.**
- **Censoring** is right for LAT and TEXT, with these additions:
  - a censored DIRECT-Q makes LP3 / LP4 unreadable. One re-run of DIRECT on the same GPU class, not a LAT fail.
  - LP2's denominator is d = 2 items (fix 1).
- **Single change:** list as choices, not "forced": L7 lr, L8 CHUNK_DELTA, the answer CE (9d) and the GQA init (fix 4). Name them in the null statement.
- **The self-calibrated LP0** replays clean on §228 / §229: 0 windows > 2× max(windows 3–10) in s228_plat2_s1, s228_pdir2c_s1 and s229_plat2c_s1. Keep it.
- Keep the tied embedding trainable (as §228). Keep L/2. One 3090 class. A second seed at A1 is not needed.

## Answers to the brief, one line each
1. **Escape derivation:** it does not follow from wo = 0 (a₀ = 0; regression regime; check D). Only presence survives. The frozen OPD probe is the wrong proxy; use EAGER at d ≥ 2 at the write sources (check T).
2. **Base and memory model:** Qwen is a sound choice and the memory fits. The GQA init collapses heads (fix 4). RoPE: the q/k copy is less faithful, while cache_lower and position-free memory stay valid. The digit split is handled; chance ≈ 0.10, no leak; add the vidx and CE fixes (9a, 9d).
3. **lr:** 1.6e-5 reproduces but is unanchored. Bracket with 8.6e-5 or anchor on Shapiro's lr; add the retention probe. wo is 2.6× hotter relative to the residual (fix 5).
4. **Arm:** INJECT for A1 (single factor). Pre-name the pass-cued / loop-retrofit arm as the next registration (fix 7).
5. **Cost / OC:** the numbers reproduce. Staged with fixes is $24.42 guard. Full is $30.53 (user). Expected ≈ $4–6, honest within ±$2. The staged plan must be terminal (fix 8).
6. **Censoring / diffs / gates:** sound, plus fixes 1, 9 and 12.
