# REDTEAM_D3: red team of D3 (SLVA + FROZEN-WPE + COPY-ANCHOR), 2026-10-16 14:11 EDT

**Scope.** This is an Opus red-team subagent's report for Theory. The repo was read only. Everything ran on CPU (3 threads, idle priority), eval-only, with my own code in `scratchpad/research2/rt_d3/`. Each script has its JSON beside it.

**How I measured.**
- I re-implemented pass 1 from the module's primitives. It matches `m.rollout(x, pad, 1)` with max |Δ| = 0.0.
- My items come from my own rngs, in batches grouped by exact token length, so no batch is padded.
- My re-implemented PA gather matches `VBankN.gather` with max |Δ V_K| = 0.0.

**Not repeated here.** PROOFS_D3 already covers the NO-KNEE defect, the OC arithmetic, Thm 1(d), the V_0 age clock, the seam that FT-SLVA keeps, and COPY-ANCHOR's R / coda / duplicate issues. I add only to those points.

**Not run, on purpose.** I did not run COPY-REFRESH. It is a pre-stated selector and should stay blind.

## 1. Verification table

| # | D3 claim | Verdict | My numbers |
|---|---|---|---|
| a1 | N2's reduced (changed) slots are correct at every T | **VERIFIED (≈)** | 1.000 / 0.998 / 1.000 / 0.999 at T 28–29 / 30–31 / 32–37 / ≥ 38 (n 382 / 413 / 1,120 / 1,420; 3 errors). "1.000 everywhere" is slightly overstated. `p1_n2.json` |
| a2 | N2's failures are late copy errors; median first wrong slot 25–27 for T ≥ 29 | **VERIFIED** | Copy-slot accuracy 0.993 / 0.955 / 0.792 / 0.652. Median first wrong slot 25 / 27 / 26 / 26. The locus is an **absolute** position: the first error's distance from the end grows with T (median 3 / 3 / 8 / 15). At T ≥ 32, 46–61% of wrong copies decode a **neighbour's class** (j−1, j+1 or j+2). That is the signature of a mis-mixed pass input |
| a3 | N1's errors also sit late | **VERIFIED, with qualifications** | 37 of 43 failing items first err at j ≥ 29, but 6 err at j 1–4. The late errors sit **near the end** (median distance from the end 2 / 4.5, 90th percentile ≤ 12). At T ≥ 38, 30 of 40 wrong decodes are **VOID** (spurious consumption), not copies of other symbols. So Thm 4(c)'s "per-slot copy error ε_ℓ" mislabels them. `p1_n1.json` |
| a4 | (§1.2 reading) "In the PA cells the step's logic is length-free" | **REFUTED for N3** | N3's changed-slot accuracy is 0.952 / 0.900 at T 32–37 / ≥ 38, and its early slots j 0–9 are 0.957 / 0.927. The claim holds for N2 only. `p1_n3.json` |
| b1 | Adjacent-row 1 − cos: 0.014–0.058 (N2), 0.005–0.008 (N1), 0.001 (pretrained) | **VERIFIED** | Rows 20–31: 0.0141–0.0580, 0.0049–0.0078 and 0.0010–0.0012. `rt_wpe.json` |
| b2 | "PA moved the wpe rows about 7× more" (212-10's wording) | **REFUTED as a movement claim** | Mean drift norm over rows 1–28: N2 / N1 = **2.71**, N3 / N1 = 2.29. Relative drift is 0.12–0.43 against 0.07–0.16. Only the *excess decorrelation* (1 − cos above pretrained) is 5.3–7.2× for N2 and 7–10× for N3. Say "decorrelated about 6×", not "moved 7×" |
| c | Rows ≥ 33 changed only by weight decay; a seam at the trained edge | **VERIFIED, and stronger; the "seam" needs rewording** | Rows 33–1023 are **bitwise identical across N1, N2 and N3** (max \|Δ\| = 0). A scalar fit gives a = 0.97595–0.97631 with residual ≤ 3.7e-4; ∏(1 − η_tλ) = 0.97633. **But the jump ‖w₃₃ − w₃₂‖ (N1 0.41, N2 0.83) equals the in-range adjacent step (0.40 / 0.94).** What changes at the edge is the spacing: 0.15 beyond it, i.e. a regime change, not a discontinuity. Rows 29–32 appear in ≤ 0.26% of items, yet they moved as far as well-trained rows (N2 0.25–0.38). That looks like Adam amplifying rare gradients (noise), not sharpening |
| d | N2's gather at j ≥ 33: 0.94 of the mass far, ≈ 0 on self; queries cos 0.999 | **VERIFIED** | Self 0.0006–0.0007, far 0.944–0.947, sink 0.045–0.048 (3,412 query slots). Adjacent-query cos is 0.9991–0.9994, but cos(q₃₃, q₆₀) is 0.79, so "the same vector" holds only locally. **New:** the far mass lands on keys < 25 (0.944), not on out-of-range keys. In T ≥ 38 items, in-range queries at j 25–28 / 29–32 put **0.34 / 0.26** of their mass on keys ≥ 33 (key capture) |
| e | A class + relative-offset fit leaves KL 0.31 nats | **VERIFIED for D3's family; the conclusion is overstated** | D3's family: 0.316 (held-out 0.318). For scale: entropy 0.90, KL to uniform 2.15, offset-only 0.43. Adding class × offset and bracket-depth terms (still translation-invariant) gives **0.143** (held-out 0.143). "Not a class + offset rule" mostly reflects the parametrisation. `symfit_n2.json` |

## 2. New $0 measurements that bear on D3

**M1. Per-slot GATHER-SPLICE on N2** (`gsj_n2.json`, T ≥ 36 items).

| | intact | after the splice |
|---|---|---|
| slots j ≥ 33 | 0.18–0.34 | **0.83–0.92** (gs33) |
| slots j 29–32 | 0.17–0.33 | 0.74–0.84 (gs29) |

- The in-range carry-input baseline is gs10 on T ≤ 27 items, at j 10–24: **0.66–0.84**. With carry input everywhere from J = 24, accuracy *rises* with j: 0.64–0.69 at j 24–28, 0.74–0.81 at 29–32, 0.84–0.91 at ≥ 33.
- **So, once the gather is bypassed, the PA-trained blocks copy the never-trained rows ≥ 33 at least as well as in-range rows.**
- At j ≥ 29 the gather is the dominant per-slot cause. The step-level null of 212-8 is a product effect: the splice costs 10–30% per spliced slot, because the core is co-adapted to mixed inputs.
- **Within-slot association at j 25–28:** accuracy is 0.81 / 0.48 / 0.23 when the gather's local mass on j−1..j+2 is ≥ 0.6 / 0.3–0.6 / < 0.3.
- **kmask33** (keys ≥ 33 masked for queries < 33) helps only a little: 0.38 → 0.49 at j 25–28, T ≥ 38.

**M2. Where N2 uses its history** (`hist_n2.json`, `hist2_n2.json`; §170 test items, 150 per n, K′ n + 2).

| arm | n 6 | n 8 |
|---|---|---|
| intact | 1.000 | 0.993 |
| registered BANK-ABLATE (mid versions masked everywhere) | 0.727 | 0.327 |
| mid versions masked in the **gather only** | 0.720 | 0.333 |
| mid versions masked in the **core + coda only** | 1.000 | 0.993 |
| gather: mid versions masked at **other slots** only | 1.000 | 0.993 |
| gather: mid versions masked at the **own slot** only | **0.900** | **0.487** |

- **All of N2's HISTORY-USED runs through the gather, and most of it reads the slot's own older versions.**
- This is exactly the channel SLVA keeps. It is the strongest evidence in D3's favour: SLVA is the right shape for the history use that the attention-memory claim rests on.

**M3. 212-13's WPE-RESTORE, split by path** (N2; `wsplit2_n2.json`; exposed control m ≤ 8, T 25–29, n 1,092). In a PA cell the table feeds both the prelude and the gather's content-free queries; wpe_b1 changes both.

| arm (rows ≥ 24 pretrained) | registered exact (VOIDs dropped) | strict per-slot exact | slot acc at j ≥ 24 |
|---|---|---|---|
| intact | 0.997 | 0.997 | 1.000 |
| both paths (= 212-13 b1) | 0.517 (212-13: 0.508) | **0.000** | 0.459 |
| prelude only | 0.514 | 0.514 | 0.141 |
| gather queries only | 0.650 | **0.000** | 0.454 |

- **Both paths are bound to the table.** The prelude swap makes genuine slot errors. The gather swap makes **slot-misaligned but re-renderable** states: VOIDs fall in the wrong place, and the symbol order survives.
- 212-13's reading ("PA bound the core to its table") survives the confound. About half of b1's damage is the gather's queries, though.
- The registered VOID-dropped exactness hides misalignment: strict exactness is 0.000 against 0.517 for the same states. For a slot-aligned memory, misalignment matters at the next pass.
- N1, by my split (`wsplit_n1.json`): b1 / b2 keep 1.000 / 1.000 strict on both sets.

**M4. COPY-ANCHOR census under exact decodes** (`anchor.json`, §170 test set).
- 34–44% of stored slot-versions (n ≥ 5) would be V_0 copies.
- Each slot holds **2.3 distinct classes** (max 5) over its whole trajectory.
- A computed value is anchored back to its input token's V_0 (value = right operand) at **2.8–10.4% of reductions**, which is 10–41% of items at n 1–16. Sanity check: P(value = right operand) at n 1 is 0.103. **PROOFS_D3's "0.14–0.57%" looks about 20× too low; please reconcile.**

## 3. Top findings

### F1. BLOCKER: as written, D3-C2 can be met by an RNN-in-function model, and the readings would not show it

**What happens.**
- D3 deliberately trains in the N1 regime, and N1 is RNN-in-function (§211: bank-swap own 1.000, carry-swap follow 1.000, BANK-ABLATE ≈ intact).
- History use is "reported, not imposed" (§4). "LENGTH-KEPT ∧ HISTORY-IGNORED → Stage 1b still proposed". §7 risk 2 is "not a falsifier of D3's C2 claim".
- So a positive is likely to be "C2 in an attention memory that is unused". That is exactly the contrast §211 used to discount N1: "VB-C1 alone does not establish attention-memory use."

**Why the readings miss it.**
- The registered BANK-ABLATE masks the core, the coda and `VBankN.gather`'s keep. SLVA replaces the gather, so the ablation masks SLVA only if the override implements `keep`.
- Version-mass profiles (CARRY-LIKENESS, "0.44 on the same slot") are not evidence. Copy-slot versions are near-duplicates, so mass on them is functionally inert.

**Fix (pre-register):** D3-C2-ATTN = D3-C2 ∧ HISTORY-USED, with three readings, all causal:
- **(i) BANK-ABLATE-SLVA.** Versions 1..t−2 masked in SLVA, core and coda. HISTORY-USED needs a drop ≥ 0.10 at n 6–8.
- **(ii) SELF-ABLATE.** Own-slot mid versions masked in SLVA only. The N2 reference is 0.900 / 0.487 at n 6 / 8 (M2).
- **(iii) The N1 swap triple, adapted:**
  - NEWEST-SWAP: the donor's V_t is fed only as SLVA's newest version;
  - BANK-SWAP: the donor's V_t sits only in the core / coda bank;
  - OLD-SWAP: the donor's V_s, 1 ≤ s ≤ t−2, at every reader.
  - **RNN-in-function** = NEWEST-SWAP follow ≥ 0.95 ∧ OLD-SWAP follow ≤ 0.02 ∧ (i) not met.

Report a positive as "C2 in a pass-global attention memory" only under D3-C2-ATTN. Otherwise report "C2 with the memory in form only".

M2 shows this is reachable: N2's real history use is slot-local.

### F2. MAJOR: collapse to CARRY is the expected outcome, especially for FT-SLVA, and COPY-ANCHOR makes it structural

**Why SLVA will drift to CARRY.**
- The task is Markov.
- For copy slots, ∂U/∂α ∝ N(V_s) − U ≈ 0, so α gets no gradient.
- For changed slots, mixing the old symbol with the new value is harmful, so the gradient points to the newest version.
- FT-SLVA starts at N1 (HISTORY-IGNORED) with LR 3e-5.
- The only useful history function is anti-drift: reading V_0 or older versions of unchanged slots. That is plausibly what N2's own-slot history does (N2 is STABLE at K′ 24; N1 is not).
- COPY-ANCHOR performs anti-drift **by rule**. It leaves SLVA with about 2.3 distinct contents per slot and nothing to learn (M4).

**Prediction.** HISTORY-USED P 0.4 for FT-SLVA is optimistic. I put it at ≤ 0.2, and ≈ 0 with ANCHOR.

**Fix.**
- Never combine COPY-ANCHOR with SLVA in one cell. If drift needs a fix, let SLVA earn it through its β⁰ / content read, and measure the V_0 mass on unchanged slots *causally* (SELF-ABLATE).
- Give FT-SLVA a precondition, **SLVA-MOVED**: OLD-SWAP follow ≥ 0.05, or BANK-ABLATE-SLVA ≥ 0.05. Without it the FT is **UNINFORMATIVE**, not LENGTH-KEPT. A carry-like FT tests nothing about "version attention re-creates length keying".

### F3. MAJOR: FROZEN-WPE is justified by a mechanism the data do not support; its real support is the graceful decay

**What the data show.**
- §1.3's account ("a seam in every block's input" causes the cliff) is contradicted per slot: bypass the gather and the PA blocks copy rows ≥ 33 at 0.83–0.92, at least as well as in range (M1).
- At j 25–32, the failures track the gather's local mass and key capture (M1, row d).
- 212-13's evidence for freezing is real but different:
  - the PA core is bound to its fine-tuned rows **≤ 32** (M3, prelude-only 0.514);
  - N1 *gains* from the pretrained table (m 16: 0.616 → 0.860).

**So FROZEN-WPE is a STEP-GENERAL lever (m 12–16), not the LENGTH-FREE lever.**
- It also undercuts Thm 4(c)'s inference that "N1 has an additional count-type error at m ≥ 15". Much of that error is the table.
- CORE-BAND / EXIST-SCALE's trigger should be re-derived after a pretrained-table baseline.

**Fix.**
- Re-state the rationale: freeze to prevent row-specific binding (M3) and to keep N1's b2 gain.
- Drop "the seam causes the PA cliff" from §1.3, or tag it [REFUTED per slot].
- Change P(STEP-GENERAL) accordingly. N1-b2 already reaches 0.977 / 0.931 / 0.860 at m 14–16, which is close to the bar.

### F4. MAJOR: the FT path (the coordinator's question)

**Is FT from N1 the only coherent FT? Yes, among existing checkpoints.**
- N2 and N3 cannot start under either D3 ingredient:
  - an all-pretrained table kills the step (control 0.000);
  - SLVA at pass 1 is the plain carry, which the PA core rejects (gs10 in range 0.72 per slot; gs24 control 0.79).
- Any FT from them would be a recovery run from a broken start.

**But the FT as written does not test Stage 1b's recipe.** Fix it as follows:
- **(a) Swap N1's table to pretrained at FT step 0, then freeze it.**
  - N1 is unharmed: control 1.000, target 0.997 → 0.999, strict 1.000 (M3).
  - The FT then tests the table Stage 1b uses.
  - G-D1 compares against the **CARRY rollout under the pretrained table**, not N1 as saved.
- **(b) The trainer's fresh-start path calls `VBankN.calibrate`.** It writes `self.beta.data` / `beta0.data` in place, which overwrites N1's trained biases, so "step 0 = N1 bitwise" fails. `--resume` needs a full checkpoint dict, not model.pt.
  - *Fix:* add an `--init_from` path that skips calibration.
- **(c) The defaults need pins:**
  - `--cur 3,5,8` would restart the curriculum at n ≤ 3, so pin `--cur 8`;
  - `lr_at` is cosine only, so "3e-5 constant" needs a flag;
  - pass `--c_sched 500,6000`.
- **(d) Add a matched CARRY-only FT** (same steps, LR and table), $0.26–0.55, or at least report the b2-N1 step-0 readings. Without a control, SELF-SIMILAR or LENGTH-KEPT movement cannot be credited to SLVA.
- **(e) The FT cannot answer Stage 1b's real question:** do blocks trained from pretrained under SLVA come out N1-like, and do they learn N2-like own-slot history (M2)? State that.

### F5. MAJOR: harness traps that would silently produce a wrong cell or wrong readings

1. **Silent CARRY training.** `vbank.main` fills c via `c_at(...) if a.mode == "pa" else 1e4`. Any new mode string trains SLVA at c = 1e4 for 48k steps. That is exact CARRY (Thm 1b), an N1 replicate that nothing flags. The `gather_log`, carry_mass and gather_gn logging are also gated on `"pa"`.
   - *Fix:* gate on a capability flag, log SLVA's off-newest mass on changed slots, and assert c(step) at 2k / 8k / 14k in the smoke.
2. **Wrong class at load.** `VBankN.__init__` asserts mode ∈ {pa, carry}. `vbank_eval.load` and `vbank_diag` (every §211 / §212 reading) build `VB.VBankN(g, args["mode"])`.
   - If SLVA reuses PA parameter names under mode "pa", a strict load can succeed and **evaluate the model with the PA gather**.
   - *Fix:* use mode "slva", distinct parameter names (LN_q, LN_k, β⁰_slva), a loader factory, and `assert type(m).__name__ == args["arch"]`.
3. **Eval hooks live inside `VBankN.gather`:** `keep` (BANK-ABLATE), `off_v0` (VALUE-SWAP iv, RECOMPUTE-V0), `age_floor`, `gather_splice`, `gather_qmod`, `gather_log` (CARRY-LIKENESS), and FOREIGN's `view()` substitution.
   - SLVA's override must honour every one, plus PROOFS_D3's pad rule.
   - Otherwise BANK-ABLATE masks the core but not SLVA (HISTORY-USE is under-read), and arm (iv) leaves V_0 inside SLVA.
   - *Fix:* add a hook-parity gate. Each hook on SLVA at c = 1e4 must equal the same hook on CARRY, bitwise.
4. **COPY-ANCHOR needs R inside `push()`,** but `vb_loss` evaluates R after the rollout. Its order relative to `edit` (VALUE-SWAP, STALL, MCT) is unspecified. A swapped donor value whose class equals the recipient's input class gets re-anchored to the recipient's V_0.
   - *Fix:* pin "edit, then anchor", or exempt edited slots.
5. **Cost.** SLVA saves about nothing. G-TIME is 0.462 (CARRY) against 0.476 (PA) s/step, and the realised 48k runs were 0.363 / 0.373 / 0.392 s/step. The $ estimates stand; drop "SLVA is cheaper".

### F6. MAJOR: the length reading looks only where SLVA cannot differ

- LENGTH-FREE uses the fresh pass 1, where SLVA equals CARRY by Thm 1(a).
- SLVA's one new degree of freedom, a position-keyed α (PROOFS_D3 item 5), acts only at t ≥ 2. So a D3 cell can read LENGTH-FREE while its version reads are length-keyed. DEPTH-GENERAL would then fail with no diagnosis.
- **Fix:** add **LENGTH-FREE-K**: the chain LONG-STEP by (m, T) at t ≥ 2 on exact inputs (`eval_steppos` already has `long_step_m_T`), plus the strict per-slot first-wrong-slot at t ≥ 2 against the edge.
- Apply PROOFS_D3's NO-KNEE fix (m 9–10 only, or ≥ 6k draws at m 11) to both readings.
- Report strict (slot-aligned) exactness beside the VOID-dropped one (M3).

### F7. MINOR: COPY-ANCHOR contradicts D3's own constraint check, and COPY-REFRESH's labels conflate causes

**The contradiction.** §2.5 says "nothing is overwritten … by rule". The anchor replaces the pass's computed output by rule, at 34–44% of slot-versions, before storage. It is append-only in form only.

**The selector's labels.** COPY-REFRESH's g < 0.2 → LAYOUT conflates three causes:
- layout;
- stale V_0 context;
- drift in VOID and value slots, which are never refreshed.

**Fix.**
- Add **RESTART-GAPPED**: restart from the decoded state at its original slots, with VOID keys masked. It separates layout from the other two.
- Keep the selector blind until that arm is pinned.

### F8. MINOR: diagnosis wording to correct

- **"Position randomisation: Never" is a non sequitur.** Eval-time offset failure in models *trained without* randomisation says nothing about training *with* it (sink kept). Keep it as a triggered lever.
- **"Corruption spreads about 3 slots left"** is better explained by key capture (0.26–0.34 of the mass on keys ≥ 33) and by the noisy rare rows 29–32 (row c).
- **Fix the overstated claims:**
  - (b2): "moved 7×" should read "decorrelated about 6×";
  - (a4): the length-free step logic holds for N2 only, not "the PA cells";
  - (e): "not a class + offset rule" is overstated.

## 4. Bottom line

**Keep.**
- **SLVA is the right shape:** it removes the per-slot failure locus (M1) and keeps the slot-local history channel that N2 actually used (M2).
- Freezing the table at pretrained values from step 0 is supported (M3; N1-b2), but as a binding / graceful-decay lever.

**Before any registration:**
1. Make the claim's attention-memory use causal and required (F1).
2. Separate COPY-ANCHOR from SLVA (F2, F7).
3. Rebuild the FT as N1 + pretrained table + CARRY control, with the init / calibration / curriculum pins (F4).
4. Close the silent-CARRY and loader traps (F5).
5. Add LENGTH-FREE-K (F6).

**Scripts** (all in `scratchpad/research2/rt_d3/`): `rt_wpe.py`, `rt_pass1.py`, `rt_gsj.py`, `rt_symfit.py`, `rt_hist.py`, `rt_hist2.py`, `rt_wpesplit.py`, `rt_anchor.py`, each with its JSON / log.
