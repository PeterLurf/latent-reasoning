# 01 MC3 ENTITY-KEYED MULTI-WRITE: independent red team (2026-11-04)

**Verdict: GO-WITH-FIXES.** Do not register as written. Five HIGH findings need fixing first:
- the statement rows share bun's ':' write projections and their step-0 geometry is unmeasured (F1);
- the CTRL-SHUF control cannot reach the reused eval and hold paths (F2);
- ROW-SWAP has no positive control and will likely give false negatives (F3);
- the OC block gives priors only (F4);
- math tags are missing (F5).

The core idea is sound and is better motivated than NK. The fixes are cheap: about $0.1 of stage-0 work plus build edits.

Method: read-only. Every code claim in the design was checked against the cited lines (all cites verified, exceptions noted below). The only computation was a synthetic numpy delta-rule simulation in the scratchpad (`C:\Users\peter\AppData\Local\Temp\claude\D--\ffe8dd4f-366e-5d47-92b5-bf5f7ac71716\scratchpad\dr.py`). No stored outputs were read.

## Why it could beat §262 NK, and why PARK is still likely (context for F1 / F4)

**What forces use of the table: the text-free recall loss (b), more than MASK-GIVEN.**
- MASK-GIVEN makes memory necessary. It does not make the statement rows necessary.
- The recall loss reads S with no text. It supervises the table directly, and the read is the existing `read_fixed` at the name token (deltamem.py 158-176), so no new read projection has to learn.
- That is the real difference from NK, where the new read was the unused part.
- But (b) is identical in CTRL-COLON. It trains bun's ':' write to become a name table too.

**CTRL-COLON can in principle hold the whole table** (DERIVED: linear algebra, exact).
- One ':' write per head is rank 1: Sd = β v kᵀ, so the read is v (k·q(X)).
- The read coefficient k·q(X) is an arbitrary linear function of the name's query, because k is a free 64-dim vector. So one head can encode a name -> scalar map over every name whose q is linearly independent (≤ 64).
- With 12 heads, capacity is not the obstacle. Learnability is: the trunk would have to build Σ_i f(v_i) q*(X_i) by attention at ':'.
- The statement rows get the outer product name ⊗ value from the write itself. That is MC3's genuine advantage.
- My PARK-UNUSED prior is ~0.35-0.45 (author: 0.30).

**The MASK manipulation is clean, with two consistency gaps (F7).**
- " some" is one GPT-2 token, and values are one token each (mod-11 values 0..10; ow_words.py 1-14), so lengths and E_b are unchanged.
- Pass 1 is unmasked. Causality makes the pass-1 rows cue-independent. The read of S = 0 is the constant `fade·wo.bias`, the same at every position (deltamem.py 174-176), so the design's claim holds (PROVED).
- No leak through aux (aux_target value: "S_t has v_t u.", inject_ow.py 63-71) or through the decoder (masked).
- " some" is natural English, so the distribution shift is small.

## Findings

### F1 HIGH: the statement rows reuse bun's ':' write projections, and nothing measures their step-0 geometry or perturbation

**Evidence:**
- `write_seq` takes k, v, α, β from the shared `mem.qkv` and `mem.gate` (deltamem.py 186-188).
- bun (write_src answer) has only ever written from the ':' row (inject_aw.py 47-56). So qkv's k / v rows and the gate were trained on ':' rows only. At "." rows, α, β (β ∈ (0, 2); near 2 reflects instead of writing) and the key spread are off-distribution and unknown.

**My simulation** (D = 64, sequential delta writes, β = 1, then decoding each entry's value among 11 codes):

| keys / gates | n = 6 | n = 10 | n = 14 |
|---|---|---|---|
| mean key cos ρ = 0 | 1.00 | 1.00 | 1.00 |
| ρ = 0.3 | 0.99 | 0.96 | 0.93 |
| ρ = 0.6 | 0.44 | 0.35 | 0.30 |
| ρ ≥ 0.8 (≈ last-row only) | 0.24 | 0.18 | 0.16 |
| ρ 0.3, n 10, α_row 0.95 / 0.8 / 0.5 | — | 0.88 / 0.52 / 0.28 | — |

GPT-2 late-layer states at the same token type (".") are usually strongly anisotropic, so ρ ≥ 0.6 at init is plausible. If so:
- at s = 1 the step-0 "table" is a recency register with no content;
- its crosstalk T·q enters every later read, because `read_fixed` reads at every token of passes 2..K and the decoder;
- bun's function is perturbed at step 0 (the design's G-EQ1 checks only s = 0);
- the cheapest repair for gradient descent is to null the perturbation (shrink v_se, or rotate k_se ⟂ q), which is the NK equilibrium.

And because the projections are shared, learning to separate "." keys moves bun's ':' writes on every pass, which is a B6 risk.

**Fixes:**
- (a) **$0 stage-0 gate SE-GEOM on bun_s1** (eval-only, 256 TEST2 d 3-4 items). Report:
  - per-delta-head pairwise key cosine at E_b;
  - α and β at E_b;
  - step-0 MC3 (s = 1) accuracy on clean TEST2 d 1-4 vs bun;
  - step-0 text-free recall of givens.
  - Pre-state the branches: if the median pairwise cos is > 0.6 in ≥ 4 / 6 heads, or the step-0 clean d 1-4 drop is > 0.03, use the init in (b) below.
- (b) Give the statement rows their **own write projections**: `qkv_se` (k / v rows only) and `gate_se`, copied from bun's at init.
  - With the copy, s = 0 is still bitwise bun, and the ':' path's parameters are no longer pulled by the table.
  - Optionally scale v_se by 0.1 at init. Gradient into v_se is (dL/dread)(k_i·q), which is non-zero, so the path is not dead. Rule 8's usage monitor still applies.
- (c) Note that §269 "keys" measures **':' keys only**. Under every §269 outcome (NAME, VALUE / CONSTANT, or not landed), the "." key geometry that MC3 depends on stays unmeasured. SE-GEOM is MC3's own gate; §269 only re-prices B2's prior, as §8 says.

### F2 HIGH: CTRL-SHUF's shuffled forward cannot reach the reused eval and hold paths

**Evidence:**
- §3 builds `x_shuf` in `tok_mc3` (at item level).
- Every caller passes only token tensors to `m.write` / `m.pass_aw`:
  - eval `passes()` inject_aw_eval.py 71, plus 324, 681, 775, 848 and 1166;
  - the hold far pass, inject_aw.py 186-196;
  - `ws_readout` 144.
- The design says these are "reused unchanged (zero-detect carries the multi-write)". For SHUF the zero-detect would fire, but `x_shuf` is not there. The SHUF arm would then either crash, or silently write **true** rows at eval and in the far pass. That turns the SHUF control into MC3 at test time and corrupts B3.

**Fix:**
- Derive `x_shuf` inside `pass_aw` from `x` alone. Names are single tokens from `NAMES`, so: find the name-token ids present before the cue, and apply a derangement seeded by a hash of the item's tokens (deterministic, the same in training and eval).
- Extend G-PATHS: for every part, `shuf_n` equals the number of zero-state starts in the SHUF arm.
- Extend G-SHUF: rows from the in-module shuffle equal the tok-level shuffle on 1k items.

### F3 HIGH: B5 ROW-SWAP has no positive control and is biased toward false negatives

**Evidence:** G_t's value a can be carried by several rows besides G_t's own "." row:
- every later "." row whose statement mentions G_t's name (the op statement S_t, which attends back by causality);
- the pass-1 ':' row, which attends to the whole text.

A single-row swap can therefore give follow < 0.5 in a model that does use the statement rows. B5 binding at follow ≥ 0.5 then makes "GO" depend on redundancy, not use. There is also no positive control showing that the swap machinery can produce "follow" at all.

**Fixes:**
- (i) **B5-POS** (positive control, both arms): replace the whole pass-1 state S_1 with the donor's.
  - Expect follow ≥ 0.8 in MC3 and CTRL-COLON.
  - If it is lower, the passes are not reading the given from memory, and B5 is VOID-CTRL.
  - In CTRL-COLON this also shows the PARK mechanism directly.
- (ii) **B5 binding** = swap **all** statement-end rows from the donor, keeping the recipient's ':' row: follow ≥ 0.5.
- (iii) Descriptive: the single G_t row, and the ':' row alone.
- Keep the distractor-row negative control (keep ≥ 0.95).
- Report unconditional follow beside the WS-conditioned figure (the conditioning in §5's censoring inflates follow).
- State that VOID-LOWN blocks GO.

### F4 HIGH: the OC block has priors, not operating characteristics

**Evidence:** §5's OC table lists outcome priors only. The registration rule asks for P(pass | no effect) and P(pass | effect) at the seed count. σ_seed for MG-NEED accuracy under this new recipe is unknown (σ_seed 0.090 is a d 6 clean figure).

**My calculation** (DERIVED: normal approximation, arm-level σ, independent seeds; a true effect of 0.30 is assumed):

| σ_seed per arm | P(B2 ≥ 0.20 \| no effect), 1 seed | P(B2 ≥ 0.20 \| effect 0.30), 1 seed | the same, 3 seeds all passing | P(PARK clause B2 < 0.05 \| effect 0.30) |
|---|---|---|---|---|
| 0.03 | < 1e-4 | 0.99 | 0.97 | < 1e-4 |
| 0.05 | 0.002 | 0.92 | 0.78 | 2e-4 |
| 0.09 | 0.058 | 0.78 | 0.48 | 0.025 |

GO is the conjunction of B1-B7, so P(GO | effect) is below each single-bar figure.

**Fix:**
- Add this table, and state that the full stage needs **all 3 seeds** per bar. At σ 0.09 that conjunction caps power at ~0.48 for B2 alone.
- Either relax the full-stage rule to "mean over seeds ≥ bar and every seed ≥ bar − 0.10", or accept the stated power.
- Add a pilot label PILOT-NOTE for "B2 between 0.05 and 0.20", which goes to NOTE rather than GO.

### F5 HIGH: there are no math-backing tags (user rule 10-13)

Each claim needs a tag. Suggested tags:

| Claim | Tag |
|---|---|
| s = 0 gives an exact identity (α_se = 1, β_se = 0; `Sd + 0.0`: only the sign of zero can differ, so G-EQ1 must compare with torch.equal, not a byte hash) | PROVED |
| Pass-1 rows are cue-independent | PROVED (causal attention; the S = 0 read is constant) |
| Delta-rule retrieval is exact for orthogonal keys | PROVED |
| Retrieval under correlated keys / α < 1 | DERIVED-UNDER-ASSUMPTIONS (the F1 table: random keys, β = 1) |
| The hold-penalty normaliser weakens by (‖S_bun‖ / ‖S_mc3‖)² | DERIVED (exact, given the same numerator) |
| The (1 − α)·S term of `S_pen − S` grows with the table, so the penalty pushes hold α toward 1 | DERIVED (this supports §2.5's prediction) |
| CTRL-COLON's rank-1 ':' capacity for name maps | DERIVED (see the section above) |
| Table survival over K' 128 | CONJECTURE (the simulation retains decode 0.76 at α 0.993 with a repeated hold key, but only 0.15 when 128 hold writes use **different** keys) |
| By-name update via KEY-ALIGN | CONJECTURE |
| All priors | CONJECTURE |

### F6 MED: s is "trainable" in name only

**Evidence:** AdamW at lr 1e-4 with warm-up and cosine over 8k (fsq120.py 731-743), and w sits in the default group. The summed lr over steps 1-4000 is 0.32, so |Δw| ≤ ~0.32 over the pilot.
- s cannot turn the rows off, or learn per-head roles, in 4k steps.
- The "s per head" usage monitor is uninformative.
- The projection argument (§2.2) is moot.

**Fix:** either put w in its own group with an lr multiplier (e.g. ×30) and log it, or declare s fixed at its init and drop "trainable". Also note that `--init_model`'s strict load will fail on the new parameter unless it is handled; G-EQ0 will catch this.

### F7 MED: two MASK consistency gaps in the hold penalty

**Evidence:**
- `hold_step`'s penalty re-forwards `xs[-1]` from `keep`, which is consistent only if `loss_reg` stores the **masked** xs.
- The far pass rebuilds its inputs with `cue_inputs(m, far_items, dev)` (inject_aw.py 186), which produces **unmasked** text. So the far penalty regularises a pass the masked items never run.

**Fix:** carry the per-item mask coin into the far pass, and assert that the penalty inputs equal the trained pass inputs for masked items (gate G-MASK).

### F8 MED: `--hd_drop` is under-specified, and it interferes with the hold penalty

**Evidence:**
- read_mask is a forward **post**-hook on `wo`'s output (inject_aw_eval.py 1017-1036). The design calls it a pre-hook at "the same hook site". A pre-hook on the input is the right choice for head dropout, but it is a new pattern.
- `wo` runs inside every `read_fixed`: passes, decoder, aux reads, recall reads, and the hold penalty's re-forward (inject_aw.py 179-182, 193).
- Fresh dropout draws in the penalty forward make `S_pen − S` include dropout noise, so the penalty fights hd_drop.

**Fix:**
- Arm and disarm the hook explicitly per call, with an assert in the style of read_mask's "un-armed read".
- Specify the drop sites: passes t ≥ 2 and the decoder only.
- Turn the drop off in the hold penalty / far pass, aux and recall (or reuse the pass's own mask).
- Note that the drop is family-level (all 6 additive heads at once), not per-head DropHead.

### F9 MED: B4 SE-ABLATE has no exposed negative control, and KILL-RETAIN cannot be attributed

**B4.** Removing writes that a model was trained with is out-of-distribution, so B4 can pass with content-free rows. B3 and B5 cover the binding, but B4 is also half of the PARK rule.
- **Fix:** report `se_off` on CTRL-SHUF too. Its drop is the "presence, not content" baseline. Pre-state B4 as MC3 drop − SHUF drop ≥ 0.10, descriptive at pilot.

**KILL-RETAIN.** Recipe pieces (b)-(d) are shared by all arms. If CTRL-COLON also has B6 < 0.95, the recipe is at fault, not the rows.
- **Fix:** add the label KILL-RETAIN-RECIPE (CTRL-COLON B6 < 0.95). Then try hd_drop 0.05, or mask_p 0.25, before blaming MC3.

### F10 LOW

1. **MG-NEED size.** bun runs with `mul_s_p 0` (bun_s1 args.json), so there is no forced mul_s, and P(≥ 1 `*_v` op at t ≥ 2) = 1 − (4/7)^(d−1): 0.67 at d 3 and 0.81 at d 4. That is fine if TEST2 has 1000 per d. But if TEST2 was built with forced mul_s, d 3 falls to ~0.43 (n ~430 < 500). Compute and record n per d at build.
2. **Scope for C4.** The recall targets include simulator S_j values. Like Note 263-2, these are hand-built intermediate states, so MC3 is not a C4 route. Say so in §1.
3. **Clean-text use.** Add `se_off` on clean d 1-8 as a descriptive. It shows whether the table is used when the text still carries the value, which is the C1 scope claim.
4. **VOID-RERUN.** A second crash should become FAIL for its binding bars (conservative censoring).
5. **Recall load.** Recall over all of known(t) includes distractor values that the task never needs. This raises the table load (n up to ~14 with inserts; still ≥ 0.93 decode at ρ 0.3). Keep it, but report recall split by class (givens / distractors / chain).
6. **Cites.** All checked line cites match. The one exception is that the read_mask hook is a post-hook (F8).

## Cost

- Pilot: 3 arms × 4k × ~1.9 s ≈ 6.3 h ≈ $1.3 (guard $2.5).
- Full stage: ~$5 (guard $8).
- Both are far under the $30 line, and the fixes add < $0.2.

## Minimal staged plan I would approve

| Stage | Contents | Cost | Decision |
|---|---|---|---|
| **0** | Gates: G-EQ0 / G-EQ1, G-SEG, G-MASK / G-SHUF (in-module shuffle, F2), G-PATHS with `shuf_n`, gradflow, mem_check. **SE-GEOM + step-0 perturbation on bun_s1** (F1a). **MC2 stage 0** text-free KEEP-k on bun. | ≤ $0.1 | Pre-stated: choose the init / own-projection branch (F1b); re-price B2 from MC2 stage 0 |
| **1** | **1k CTRL-COLON with the full recipe (b)-(d)** (not MASK only), read MG-NEED d 3-4 | ~$0.1 | If CTRL ≥ 0.78, then B2 (MC3 ≥ CTRL + 0.20) is unreachable: STOP (necessity fails). If CTRL B6 < 0.95: KILL-RETAIN-RECIPE, so fix the recipe before any MC3 spend. Otherwise CTRL resumes to 4k as the pilot arm (nothing wasted) |
| **2** | MC3 + CTRL-SHUF to 4k (CTRL-COLON resumes), with the fixed B5 (B5-POS positive control, all-rows binding swap), the SHUF `se_off` control, and the F4 OC table registered | ~$1.0 | Labels as in §5, plus PILOT-NOTE and KILL-RETAIN-RECIPE |
| **3** | Full stage on GO only: 3 seeds, with the per-seed rule stated with its power | ~$5 | — |

Stage 1 is the cheapest decisive step: a $0.1 read can kill the whole design (CTRL-COLON alone solves MASK-GIVEN), and the run is not wasted if it survives.
