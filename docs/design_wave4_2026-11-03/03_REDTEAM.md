# 03 C3 KV-SHARED / CUE-ROW-CACHED PASSES: independent red team (2026-11-04)

**Verdict: GO-WITH-FIXES for E0 and the $0.1 S0′ read only. NO-GO for CR-FT as a C3 experiment.**
- E1 is exact, and the design is right about where exactness stops (F1). It is accounting and eval-speed engineering, not a C3 result.
- The binding C3 criterion of CR (CR-C3F) is decided by the design's own arithmetic before any run: it FAILS at every d in stage 1 and at d ≥ 2 in stage 2 (F2). So the FT spend cannot change the C3 verdict.
- The C3-relevant unknowns are latency (E0, $0.10) and text-CoT accuracy at matched compute (§266, already registered). Run those first.

**Method.** Read-only. Every code cite in §2-§3 was checked against the file. All cites hold.
- I wrote two CPU scripts in the scratchpad; no stored run outputs were read:
  - `C:\Users\peter\AppData\Local\Temp\claude\D--\ffe8dd4f-366e-5d47-92b5-bf5f7ac71716\scratchpad\flops.py` re-derives the F_* formulas and sweeps d 1-8.
  - `C:\Users\peter\AppData\Local\Temp\claude\D--\ffe8dd4f-366e-5d47-92b5-bf5f7ac71716\scratchpad\e1chk.py` is an fp64 exactness check on a random 12-block GPT-2-like model with a block-6 read.
- I read only args.json for bun_s1 and of_s1.

---

## Findings

### F1 (verified, no fix needed; scope wording LOW). E1 is exact. The memory read does make prompt-row KV of blocks ≥ 6 pass-dependent, and the design does not share those blocks.
- **Evidence (code):**
  - `InjectAW.pass_aw` (inject_aw.py:47-56) applies `DM.read_fixed(mem, h, S)` to **all T rows** on entering block `mem.layer`.
  - read_fixed adds `fade·W_o[...]` at every row (deltamem.py:163, 176).
  - So the K/V of blocks 6-11 at prompt rows are functions of S_{t-1}, and caching them across passes is not exact.
  - Blocks 0-5 see only `embed(x)+wpe` (deltamem.py:254-259, `gpt_lower`). Every pass input has the identical token prefix (asserted, inject_st.py:60), the decoder input is `think + SFX` (inject_ow.py:73), and padding is on the right (nested170.py:166-170).
- **Evidence (computation, fp64):** prefix-cached lower plus suffix-over-cache equals the full forward to a max relative error of **1.2e-15**.
  - Prompt-row K at block 5 is invariant to S (Δ = 0.0). At block 6 it changes with S (Δ = 0.87).
  - read_fixed on S_0 adds exactly b_o (Δ = 0.0), so CR pass 1 ≡ bun pass 1 holds.
- **Wording (LOW):** the title says "KV-shared passes".
  - Only blocks 0-5 (plus the q-slice) are shared exactly. Blocks 6-11, which are 97 of 128 G at d 8, are not.
  - The title and any C3 sentence should say "lower-half-shared". CR is a different, trained model, not a cached bun.

### F2 (HIGH). CR-C3F, the only binding C3 row for CR, is a pre-run arithmetic FAIL. This is the Note 246-0 trap again.
- **Evidence:** the design's own formulas, P = 35 + 13(d−1), sentence CoT 5d + 6 generated tokens (Note 246-0). GFLOPs at K′ = d:

| d | text val | text sent | CR-P (stage 1) | CR-long (stage 2) | CR-S2 | CR-P / sent | CR-long / sent | CR-S2 / val |
|---|---|---|---|---|---|---|---|---|
| 1 | 6.5 | 8.7 | 11.7 | 8.6 | 7.1 | 1.35 | 0.99 | 1.09 |
| 2 | 8.9 | 12.2 | 17.0 | 12.8 | 9.7 | 1.40 | 1.05 | 1.08 |
| 4 | 13.9 | 19.2 | 27.7 | 21.2 | 14.8 | 1.44 | 1.10 | 1.07 |
| 8 | 24.0 | 33.3 | 49.2 | 38.1 | 25.3 | 1.48 | 1.14 | 1.06 |

- **What the table shows:**
  - CR-C3F requires F ≤ F_TEXT-SENT "at every d 1-8" for "the passing stage". Stage 1 (CR-P) fails at every d by 35-48%. Stage 2 fails at d ≥ 2.
  - No CR variant reaches value-only parity even with an oracle halt (CR-S2 is 1.06-1.09x).
  - At the deployed K′ 8, every ratio is worse.
- **Prior:** the OC prior of 0.15 for CR-C3F is inconsistent with the table. It is ≈ 0, up to the real-P correction from E0.
- **Tag:** PROVED (arithmetic), given the F_* model. It is DERIVED-UNDER-ASSUMPTIONS only for the real P distribution.
- **Fix:**
  - Delete CR-C3F as a binding row. Record it now as an analytic result: "cached-prompt latent loops are 1.06-1.48x text CoT FLOPs at oracle halt; C3-on-FLOPs closed for this family."
  - Any CR-FT must be justified on non-C3 grounds (F6), or not run.

### F3 (HIGH). "Matched compute" is defined only as a FLOP or latency ratio. C3 is accuracy at matched compute, and the design never pairs the two.
- NORTH_STAR C3 reads: "At matched compute, equal or better accuracy than written CoT".
- E0 measures cost with random weights. CR-ACC measures latent accuracy. Nothing combines text accuracy with the text budget.
- **Where the latent can win C3 at all:** in the depths where text CoT trained on the same data fails, i.e. extrapolation at d 6-8.
  - §266 X3 measures exactly that, with text trained on bun's REG d ≤ 4 data. It is registered and unscored.
- **Fairness gap:** a fair "matched" comparison must let text spend the latent's extra FLOPs. At d 4, K′ 8, bun uses 104.5 G against 13.9 G for value-only text, so text could run **self-consistency / majority vote over n ≈ F_lat / F_text samples** (≈ 7 at d 4 K′ 8, ≈ 9 at d 8).
  - Without that, "latent more accurate at d 7" is an accuracy result at *unequal* compute that favours the latent.
- **Fix:** pre-state a C3 scorer row that reads §266's per-d text accuracy (greedy, and majority@n with n = ceil(F_lat / F_text) at sampling T = 0.7; a $0 extension if §266's eval can sample, otherwise a small eval-only part) against of / bun accuracy at K′ 8 and at the C5 halt.
  - C3-ACC-AT-MATCHED-FLOPs holds at depth d iff latent acc ≥ text majority@n acc − 0.02 (Wilson, NEAR).
  - C3-LATENCY holds at depth d iff E0's measured latency ratio is ≤ 0.83 **and** latent acc ≥ text acc − 0.02 at that d.
  - Both are labels for sota_arch.json, pre-stated before E0 or §266 output.

### F4 (MED). Numeric error in the E1 headline, and the headline uses the oracle K′.
- **The d 1 value is wrong:** F_E1 at d 1, K′ 1 evaluates to **11.8 G (−21%)**, not 10.8 G (−28%).
  - Breakdown: 6blk(35) 2.97 + cue lower 0.94 + SFX lower 0.34 + upper 4.02 + decoder upper 3.40 + head 0.08.
  - d 4 (47.4) and d 8 (127.5) reproduce.
- **The headline uses K′ = d:** at the deployed think_k 8, E1 saves only **9% / 26% / 41%** at d 1 / 4 / 8 (44.0 → 40.1, 104.5 → 77.3, 217 → 128), leaving 6.2x / 5.6x / 5.3x value-only text.
- **Fix:** correct to "21-41% at K′ = d, 9-41% at K′ 8", and lead the table with K′ 8 and the C5 halt (R6 already says so; the table and E0-LAT1 still use K′ = d, see F5).

### F5 (MED). E0-LAT1 is defined at K′ = d (an oracle halt), which contradicts R6. E0 also has no pre-stated consequence.
- **Contradiction:** §5 E0-LAT1 says "K′ = d". R6 says E0 must lead with K′ 8 and the stability-rule row.
- **Predictability:** in serial layers (design formula 18 + 6K′, vs value-only 12(d + 1)):
  - at K′ 8, bun is 66 against 24 / 60 / 108 at d 1 / 4 / 8;
  - at bs 1, GPT-2-small is launch-bound, so latency tracks the kernel count per serial layer.
  - The outcome is therefore largely predictable (LOSE at d ≤ 3, NEAR at d 4-5, WIN only at d ≥ 6), but it has not been measured, and it is cheap.
- **E1 may be slower at bs 1:** E1 splits prefix and suffix into separate SDPA calls with explicit masks, which adds launches. E0-LAT1 should report BUN and BUN-E1 separately and quote the faster as "bun latency".
- **Graph fairness:** both arms should get a CUDA-graph (or torch.compile mode="reduce-overhead") row beside eager. Deployed text decoding uses graphs, and eager launch overhead hits the latent's ~45 extra DeltaMem kernels per pass and the text's per-token loop differently.
- **Fix:**
  - E0-LAT1 is read at **K′ 8 and at the C5 halt rule** (halt at d−1 with a decoder each pass, Note 263-3). K′ = d is a reported bound.
  - Pre-state what each label does to sota_arch.json's C3 row via F3's rule.

### F6 (MED). CR repeats CACHED-CUE's open mechanism question. Its real value is C1-mechanistic, not C3, and S0′ should run on **of**, not only bun.
- **What differs from Note 251-13:**
  - (a) b_o is kept (S0′). R3 is correct: the read_mask hook multiplies `mem.wo`'s output, bias included (inject_aw_eval.py:1017-1019, 1036).
  - (b) Fine-tuning instead of masking.
  - (c) A bun / of init instead of rcb.
- **What is the same:** the single read point at block-6 entry on cue rows.
  - For hop t, the cue row must query S with S_{t−1}'s identity. But the cue names only S_t (inject_st.py:43-45). S_{t−1}'s name exists only inside S_t's op sentence in the prompt.
  - So blocks 0-5 of the cue row must resolve "S_t → its operand's name" before the read. In bun the operand-name prompt rows do this lookup by a direct name match.
  - This is a CONJECTURE, but it matches 251-13: hop 2 survives masking (WS pass 2 0.84 / 1.00) and hop 3 collapses (0.49 / 0.17).
  - If true, FT must teach a lower-half pointer chase, and S0′ (bias only) will not rescue it. The 0.8 prior on S0′ ≤ 0.3 is reasonable.
- **of is the better test bed:**
  - of_s1 / of_s2 are the SOTA head (Note 263-3).
  - Their recipe trains ':'-row reads under a name cue: ask_p 0.25 and NAME-SWAP j = 1 follow 1.000 (Note 263-3).
  - If cue-row reads suffice anywhere in this lineage, it is in of.
- **What a positive result would show:** "the state is the only cross-pass carrier, read at 11 rows" is a stronger C1 statement than any C3 statement CR can make (F2).
- **Fix:**
  - S0′ on **of_s1, of_s2, bun_s1, bun_s2** (eval-only, ~$0.15), with WS per pass.
  - Add the descriptive "operand-name decodability" probe at the cue row's block-6 input (is S_{t−1}'s name linearly decodable there?) to test the CONJECTURE.
  - Any CR-FT goes in a separate registration, framed as C1 (cue-row-only carriage). It warms from **of** and supports `perm_p` / `ask_p`, since the current asserts (§3, inject_aw.py:361-419 row) exclude exactly of's recipe.

### F7 (MED). Required registration blocks are missing.
- **Math tags:** the design has **0** PROVED / DERIVED-UNDER-ASSUMPTIONS / CONJECTURE / DATA tags (user rule 10-13). Each claim needs one:
  - E1 exactness: PROVED (causality plus S-independence of blocks 0-5; fp64-checked).
  - F_* formulas: DERIVED, under no-overhead / causal-half attention.
  - "CR ≈ sentence parity": PROVED-by-arithmetic as a FAIL.
  - CR trainability: CONJECTURE.
  - 251-13 numbers: DATA.
- **OC block:** "OC block (outcome priors)" lists priors only. The standing rule needs P(pass | no effect) and P(pass | effect) at the seed count.
  - Example: CR-ACC "each of 3 seeds, LB ≥ 0.93". If the per-seed pass rate is 0.35, then P(3/3) ≈ 0.04. The rule is nearly unpassable under the design's own prior and should be stated as such.
- **Controls:** CR-CTRL compares each FT seed with **one** CTRL-STD run. CTRL needs the same 3 seeds and data order, or CR-CTRL is per-seed only for s1.

### F8 (LOW). The FLOP gate may sit at its tolerance by convention, not by bug.
- `FlopCounterMode` counts SDPA as full T² (the causal mask is ignored) and does not count layernorm / gelu / einsum-free ops.
- The analytic blk() uses the causal half, 2DT(T+1). At T ≈ 137 the gap is ≈ 1.5% for full-row configs, against a 2% tolerance.
- **Fix:** compare against an analytic "counter convention" variant (4DT² attention), and keep the causal-half formula for reporting.

### F9 (LOW). Cost is underestimated, but still far below $30.
- **Step time:** the CR step is assumed to be 0.5x bun (0.55 s). The aux path (full 12 blocks, d per step), the hold / far passes and the stage-1 full-row decoder are unchanged. Expect 0.7-1.0 s on a 3090 (bun_s1_gtime: 1.306 s/step).
- **Missing items:** per-cell setup (~0.35 h), 3 CTRL seeds (F7), and of-init support.
- **Realistic total:** ≈ $3-4, guard $5. This matters only if CR-FT is ever registered (F6).

### F10 (LOW). Naming and duplication.
- "G-TIME" already means the host training s/step admission probe (Notes 222-1, 268-1; `*_gtime/timing.json` dirs). Rename E0's harness **G-LAT** to avoid scorer and doc collisions.
- E1b, the decoder's lower half from the shared prefix, overlaps 11_efficiency_implementation.md **EF6 DEC-LOWER-CACHE**. Land one implementation with one gate set.
- E1's serial-layer count: with the decoder lower shared, E1 is 12 + 6K′ (bun's formula is 18 + 6K′, or 12 + 6K′ if the decoder lower overlaps the passes).
- E1-FLIP = 0 matters only for eval speed. C3 latency claims use timing plus the old-path accuracy, so E1 flips cannot bias them.

---

## Minimal staged plan I would approve

1. **Stage A ($0, Theory):**
   - Correct F4.
   - Add math tags and a real OC block.
   - Record F2's table as a PROVED analytic note: C3-on-FLOPs closed for lower-shared and cue-row-cached loops; best case CR-S2 1.06x value-only at oracle halt.
   - Pre-state F3's two C3 labels and the sota_arch.json consequence before any E0 or §266 output.
2. **Stage B (≈ $0.15, one cloud 3090, eval-only, no FT; one cell):**
   - **G-LAT** (E0): random-weight trunk; real P from of's items; BUN and BUN-E1 (E1 gated in fp64 only, timing use); TEXT-VAL / TEXT-SENT with a preallocated cache plus the HF row; eager and CUDA-graph; K′ 8, the C5 halt rule, and K′ = d as a bound; bs 1 / 64; 2 processes.
   - **S0′** on of_s1 / of_s2 / bun_s1 / bun_s2, with the operand-name probe.
3. **Stage C ($0):** score C3 once §266 lands (text greedy plus majority@n at matched FLOPs, per F3), combined with Stage B latency. Update the C3 row: "latency WIN at d ≥ X, accuracy at matched compute Y; FLOPs not met (PROVED)".
4. **Stage D (conditional, separate registration, C1 framing):**
   - Only if S0′ on of gives d 3-4 ≥ 0.70 at K′ 8 in both seeds, or the probe shows the operand name is decodable at the cue row: draft CR-FT from of with the fixes in F6, F7 and F9 (3 seeds plus 3 CTRL, guard ≈ $5).
   - Otherwise CR stays parked, and C3 work ends at Stage C.

E1 itself (the training / eval speedup) is left to the efficiency track (11, EF6). It needs no C3 registration.
