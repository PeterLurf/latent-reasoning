# RED TEAM: SELF-DEPTH-CHAIN draft (docs/research_selfdepth_2026-10-17/SELFDEPTH_draft.md)

Started 2026-10-17 01:50 EDT (clock). Opus red-team subagent for Theory. $0: no training, no GPU, no repo edits. CPU only (3 threads, Idle / nice 19; the three local GPUs were idle at launch).
Scripts and outputs: `scratchpad/research4/rt_sd/` (real_compose.py, compose_ref.py = byte copy of the draft's compose.py, real_compose_*.json, compose_gate_repro20k.json).
Verdicts: **VERIFIED / REFUTED / UNVERIFIED / NEEDS-FIX**, with numbers. Tags as the draft: [MEASURED] (raw file / new CPU run), [CODE] (read in boxes/*.py, line numbers), [COMPUTED].

Housekeeping disclosure: my first CPU run imported the draft's compose.py from docs/.../scripts/, which wrote `scripts/__pycache__/compose.cpython-314.pyc` (gitignored, .gitignore:11). I deleted that directory at 02:02 and later runs import a byte copy from rt_sd/ with bytecode writing off. No other file outside the scratchpad was written.

---

## 0. Verdict summary

| # | claim / design element | verdict | numbers |
|---|---|---|---|
| 1 | COMPOSE-VALID costs â‰ˆ 0 extra drop over KEEP on real decodes | **VERIFIED by a new CPU measurement** (n 9â€“16), but the draft's stated reason is a non-sequitur | CV = KEEP on 3,960 / 3,960 kept chains at n 9â€“12 and 1,892 / 1,892 at n 13â€“16; value on the span's last slot and frame intact on every kept step; Ï€^E 0.99975 pooled (Â§1.1); reason: chain_exact is string-level (vbank_diag.py 1213), blind to SPAN / FRAME |
| 2 | Composition of fresh decodes onto the original frame == vbank.recode_targets | **VERIFIED** (reproduced, 20k items, 170,011 steps, 0 failures; code read) | Â§1.3 |
| 3 | Post-filter precision (label level) â‰¥ 0.997 bounds the R2 target error | **REFUTED as stated / NEEDS-FIX**: label-right, state-wrong chains exist | fresh n 9: 1 / 1,000 COMPOSE-VALID chains has E_self â‰  gold with the right label (0 label errors; pooled n 9â€“12 Ï€^E 3,959 / 3,960); compose fixture: 274 / 2,875 state-corrupted chains (9.5 %, â‰ˆ 1/11 in Z11) keep the right label |
| 4 | "Every value-correct chain is chain-exact" | **REFUTED on fresh items** (holds on the Â§170 test items only) | fresh n 9: 1 counterexample |
| 5 | R2 targets for pass t from the chain = the right in-trajectory targets; no off-by-one; R1 window | **VERIFIED** [CODE vbank.py 454â€“502] with one spec gap (HOLD padding to the batch Kb) | Â§2 |
| 6 | Prop S0 (identity, strong form, (a), (b)) | **VERIFIED** (counting on targets) | Â§3 |
| 7 | DEPTH-BEYOND-SELF â‰¥ 0.80 reachable; P(DBS) â‰ˆ 0.11, P(SD-C2) â‰ˆ 0.075 | arithmetic **VERIFIED**; priors **UNVERIFIED and generous** (no VB cell has shown > +2 margin) | shade to P(DBS) â‰ˆ 0.03â€“0.06, SD-C2 â‰ˆ 0.02â€“0.04 |
| 8 | "Primary C2" label for DBS; "beyond gold-supervised depth" for ABSORB | **NEEDS-FIX (wording)** | NORTH_STAR C2 = gain at Kâ€² > K in every competent seed; DBS at Kâ€² n + 2 is Kâ€² â‰¤ 16 = K at n 13â€“14; one seed; labels are gold-equivalent (Â§2.2) |
| 9 | ABSORB is a meaningful success by itself | **No for C2 / C4** (in support by S0c; labeller learned its step from simulator R2 states); enabler only | Â§3.3 |
| 10 | Fresh-once pool, 120k kept per n | **NEEDS-FIX**: the self curriculum overdraws n 9â€“10 | worst case 128k (stage A cap in global steps) or â‰ˆ 160k (cap in self steps) per n at n 9â€“10 vs 120k |
| 11 | FT-CTRL sees identical plain batches | **NEEDS-FIX (spec)**: holds only if self K offsets come from rng_s, not rng_k; hash must include U draws | Â§4 |
| 12 | KEEPS-SELF-STAB "(hist24, n 3..8)" | **NEEDS-FIX**: the scorer and the 0.782 baseline are n 3..6 | theory/score_s213.py 125 |
| 13 | calibrate() bypass under --init_from | trap **VERIFIED**; plus a new one: `cal` undefined â†’ NameError at the first checkpoint, not reached by G-S4 (--stop_at 1) | vbank.py 602â€“610, 710 |
| 14 | Gold-leak guards (no root / t in pool, monkeypatch) | **NEEDS-FIX**: slva.mass_stats calls rec_targets â†’ recode_targets(it["root"]) every log step (not in the edit list); vb_loss needs it["t"] (contradicts "reject records carrying t" unless renamed); smokes at --stop_at 1 never reach a log step | slva.py 108â€“111, vbank.py 458, 485, 664â€“666 |
| 15 | mem_check_self / G-S9 | **NEEDS-FIX**: 1-step mem_check omits AdamW moments (~1 GiB); D4 v1 fixed this with --mem_check_steps 2, this draft did not; budget 21.5 vs the base's 21.0 | vbank.py 636â€“651, 711 |
| 16 | Memory / cost projections | arithmetic **VERIFIED**; model **ASSUMED** (one measured point); PROOFS_SELFDEPTH Â§11 corrects its inputs (padded T 31, AdamW) to 26.0 GiB / micro 2, and the kextra-2 fallback to micro 2 | 23.2 GiB at K 16 / T 47; FT pair $0.82â€“1.39 (PROOFS: $0.82â€“1.43) |
| 17 | GATE-q OC | **VERIFIED** (label level); Ï€^E and the COMPOSE-VALID keep term were not in the OC; both now measured â‰ˆ 1 at n 9â€“12 | gate_oc2.json |
| 18 | OC arithmetic (SD-SELF 0.45, MG 0.25, DBS 0.11, SD-C2 0.075, SD-C2-MEM 0.05) | **VERIFIED** (re-computed: 0.448 / 0.256 / 0.111 / 0.0763 / 0.053) | Â§6 |
| 19 | Readings that pass under no extrapolation effect | **NEEDS-FIX (labels)**: SD-SELF, STOP-MOVED, SELF-ATTRIBUTED, THINK-GAIN-BEYOND, C2-TAIL, DEPTH-GENERAL all pass under H_shift; only DBS, MARGIN-GROWS and Kâ€²-BEYOND (unthresholded today) discriminate; mechanism labels are not exhaustive; STOP-LOC reads unreadable at the base and under exact H_shift, so SHIFT is unassignable (concur with PROOFS_SELFDEPTH Â§5) | Â§6 |
| 20 | Lee 2502.01612 / 2511.07378 details | **UNVERIFIED** here (not re-fetched) | Literature to check before THEORY |

**Blockers before registration** (none needs spend): #10 pool sizing, #13 `cal` NameError + a checkpoint/resume smoke, #14 mass_stats + the in-memory label key + a log-step smoke, #3 / #4 state-level precision in GATE-q (already there as Ï€^E), in LABEL-AUDIT and in Prop S2. Everything else is wording or spec.


---

## 1. Label correctness and R2 targets (attack 1)

### 1.1 New measurement: COMPOSE-VALID, value-slot placement and Ï€^E on D3-T's real fresh decodes [MEASURED, CPU fp32]

`rt_sd/real_compose.py`: the exact 213-37 / 213-38 fresh items (rng 21337, Â§170 test + SERIAL exclusion, all 1,000 drawn per n so the stream matches), D3-T model.pt loaded with vbank_eval.load on CPU (fp32, 3 threads, bs 32, length-sorted batches, cap n + 4: part_chainlabel_fresh's loop), but **every step's full decoded class vector is kept** and scored with the draft's own compose.py checks (FRAME, SPAN, ORDER, CANON, ONE_OP) and its E_self composition, then compared with vbank.recode_targets(gold root) at every t (evaluation only). The KEEP decision is asserted equal between my loop and compose.chain_compose on every item.

| n | items | KEEP (213-38 GPU) | KEEP (CPU here) | COMPOSE-VALID | extra drop | kept-wrong (label) | value slot = span's last (all steps) | FRAME bad steps | **Ï€^E** (E_self == gold E_t âˆ€t, among COMPOSE-VALID) |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 1,000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 9,000 / 9,000 | 0 | **0.999** (1 label-right, state-wrong) |
| 10 | 1,000 | 0.996 | 0.996 | 0.996 | 0 | 0 | 9,966 / 9,968 (2 "other", both on FAIL chains) | 0 | 1.000 |
| 11 | 1,000 | 0.988 | 0.988 | 0.988 | 0 | 0 | 10,880 / 10,882 (2 on FAIL chains) | 0 | 1.000 |
| 12 | 1,000 | 0.976 | 0.976 | 0.976 | 0 | 0 | 11,763 / 11,766 (3 on FAIL chains) | 0 | 1.000 |
| **pooled 9â€“12** | 4,000 | | 3,960 kept | **3,960** | **0** | **0** | 41,609 / 41,616 (7 "other", all on FAIL chains) | **0** | **3,959 / 3,960 = 0.99975** |

(real_compose_n9_12.json; 1,008 s on 3 CPU threads.)

**Round-2 / BASE-LABELLED fallback rows** (real_compose_n13_16.json; the first 500 of 213-38's 1,000 items per n, 874 s):

| n | items | KEEP (213-38, 1,000) | KEEP (CPU, first 500) | COMPOSE-VALID | extra drop | kept-wrong | value slot = span's last | FRAME bad | Ï€^E |
|---|---|---|---|---|---|---|---|---|---|
| 13 | 500 | 0.972 | 0.972 | 0.972 | 0 | 0 | 6,345 / 6,346 | 0 | 1.000 |
| 14 | 500 | 0.952 | 0.946 | 0.946 | 0 | 0 | 6,701 / 6,705 | 0 | 1.000 |
| 15 | 500 | 0.943 | 0.954 | 0.954 | 0 | 0 | 7,228 / 7,232 | 0 | 1.000 |
| 16 | 500 | 0.917 | 0.912 | 0.912 | 0 | 0 | 7,481 / 7,485 | 0 | 1.000 |
| **pooled 13â€“16** | 2,000 | | 1,892 kept | **1,892** | **0** | 0 | (all non-last on FAIL chains) | **0** | **1,892 / 1,892** |

(213-38's 3 kept-wrong at n 15 fall in items 501â€“1,000, not run here.)

- The CPU run reproduces 213-38's GPU keep rate exactly at every n 9â€“12 (a check that the harness is the same chain; fp32 CPU vs fp32 3070 flipped no keep decision at the rate level).
- **Every SPAN / ORDER / CANON / ONE_OP failure sits on a chain that already fails KEEP** (unparseable or wrong step count). Among KEEP chains, COMPOSE-VALID never fails: the real decodes put the value on the span's last slot, VOID exactly the handle (operands, operator, own brackets), keep the frame (EOT, " =") and re-render canonically.
- So the draft's prediction "extra drop â‰ˆ 0" is **VERIFIED by measurement** on the base labeller at n 9â€“16 (5,852 kept chains, 0 extra drops), and S0-1 (a)â€“(c) are answered for round 1 and for a base-labelled round 2 at $0. A round 2 labelled by the FT'd model needs the same check on that model (GATE-2).

### 1.2 Why the draft's stated reason did not support that prediction

- The draft (Â§2.1) infers "predicted extra drop â‰ˆ 0, because every value-correct chain is chain-exact". chain_exact (vbank_diag.py 1213) compares `VB.rerender(dec)` with `render(reduce_k(true, 1))`. `rerender` drops VOIDs and the frame (vbank.py 50â€“51). It is **string-level** and blind to exactly the two things COMPOSE-VALID adds: value placement inside the span and the frame. The draft's own fixtures show this (KEEP passes 3,000 / 3,000 value_first and frame corruptions).
- The conclusion was right; the argument was not. The fix is to cite Â§1.1, not chain_exact.

### 1.3 Composition == the trainer's targets [VERIFIED]

- Reproduced the draft's gate: `compose.py ... 20000` â†’ **20,000 items, 170,011 steps, 0 failures** (compose_gate_repro20k.json), fixtures identical to the draft's.
- Code read: recode_targets (vbank.py 32â€“47) VOIDs the handle's span over the **live** slots and writes the value on the span's last live slot; earlier VOID gaps stay VOID. compose maps dec_k's interior slots onto E_{kâˆ’1}'s live slots in order. With CANON at k âˆ’ 1, render(cur_{kâˆ’1}) is exactly that live sequence, so a canonical fresh decode composes to recode_targets' row k, gaps included (e.g. "( VOID VOID 6 + 4 )" â†’ the bracket slots and the operands VOID, the value on ")").

### 1.4 Label precision is not R2 target precision [REFUTED as stated; NEEDS-FIX]

- **Fresh n 9: 1 of 1,000 COMPOSE-VALID chains has the right final label but a wrong intermediate state**, so its E_self differs from gold at some t while label-level precision reads 1.000 (rt_sd/real_compose_n9_ex.json, a re-run on the same 1,000 items with example capture, reproduces the count: keep 1,000, CV 1,000, 1 state-wrong). The item is `3 - 6 * ( 5 / ( 4 + 0 / ( 8 - ( 0 - 7 ) / 8 - 9 ) ) )`: at step 3 the fresh step reduces `8 - 6` to **6** (gold 2), step 4 then correctly computes `6 - 9` = 8 (gold `2 - 9` = 4), and step 5's `0 / 8` = 0 = `0 / 4` absorbs the error. So R2 targets at passes 3 and 4 teach a wrong state (`8 - 6 = 6`) under a right R1 label.
- This is expected arithmetic, not a fluke: in Z11 a wrong intermediate value usually maps to a uniformly wrong final value, but lands on the right one with p â‰ˆ 1/11 (and with p = 1 under "Â· 0"). The draft's own fixture shows it: wrong_value corruptions give **2,875 E-wrong COMPOSE-VALID chains but only 2,601 wrong labels; 274 (9.5 %) are label-right, state-wrong**. The draft reports the 2,601 and not the 274.
- Consequences:
  - 213-35's "every value-correct chain is chain-exact" is a test-set fact (n 9â€“16, 8,000 chains); it **does not hold on fresh items** (the population the pool is drawn from).
  - Prop S2's Ï†, Ï€ and "post-filter precision â‰¥ 0.997" are label-level. They bound R1 noise, not R2 noise. The R2-relevant rate is w_E â‰ˆ w_label Ã— (1 + ~0.1) plus compensating errors; at the measured rates (â‰¤ 1e-3) this changes no conclusion, but the proposition must be stated on states.
  - **LABEL-AUDIT** ("gold precision of 5,000 consumed pool items â‰¥ 0.99") must be state-level (E_self == gold E_t at every t), or it misses this class by construction.
  - GATE-q's Ï€^E term is the right one; its OC was not computed (gate_oc2 is label-level). At Ï€^E â‰ˆ 0.999 measured here it passes with P â‰ˆ 1.

### 1.5 Is KEEP âˆ§ COMPOSE-VALID sufficient for correct R2 targets?

- For layout: yes (Â§1.1, Â§1.3).
- For values: no label-free check can see a value error (the draft says so). The residual is ~1e-3 per item at n 9â€“12 (Â§1.1) and is what Ï€^E / LABEL-AUDIT must report.
- The pool record must store the per-step decoded value for every step (the draft's (j_lo, j_hi, j_val, value) does), so a state-level audit is possible after training.

---

## 2. Distribution mismatch, indexing, budgets (attack 2)

### 2.1 Are the chain's targets the right in-trajectory targets? [VERIFIED, CODE]

- vb_loss (vbank.py 462â€“502): R2 at pass t is CE(R(V_t), E[:, t]) over the item's real slots, with the changed-slot mask E[:, t] â‰  E[:, t âˆ’ 1]; R1 is the answer CE for t âˆˆ [n_i, K_i] (line 470); the answer target is it["t"] (line 485).
- On a COMPOSE-VALID, state-correct chain, E_self,t is **bitwise** recode_targets(root)[t] (Â§1.3). So the target at in-trajectory pass t is exactly the gold target D3-T was trained on at n â‰¤ 8, over the same original frame. The fresh-vs-trajectory mismatch does not enter the targets at all.
- What it does enter is **selection**: an item is kept iff the *fresh* chain succeeds. At n 12, 2.4 % of items are dropped (KEEP 0.976). If the latent loop fails on the same items, ABSORB's ceiling is â‰ˆ 0.976 at n 12 (still â‰¥ 0.95). Descriptive only.
- Pass 1 (Prop S3a): rollout's V_1 depends only on V_0 and pass 1 (vbank.py 315â€“339), so pass 1 of a self item is the chain's step 1 at round start, up to dtype and padding (training bf16 in padded batches, the labeller fp32 length-grouped; PROOFS Â§4). VERIFIED as the same computation, not bitwise.

### 2.2 Consequence the draft under-states: round-1 self-labels are gold-equivalent

- On â‰¥ 97.6 % of items (KEEP at n 12) the R1 label and every R2 row equal the simulator's, except ~1e-3 state errors (Â§1.4).
- So FT-SELF round 1 is, to within 0.1 %, **"gold per-step fine-tuning at n 9â€“12 on the subset the base's fresh chain solves"**. The self-labelling contributes the procedural property "no gold field reaches training beyond n 8". It contributes no new information about depth, and it does not move C4 (the labeller's step was learned from simulator R2 states at n â‰¤ 8; the draft says strict and revised C4 stay unmet).
- **All of the experiment's information is in ABSORB (capacity) and in DBS / MARGIN-GROWS (does the extrapolation margin grow with B)**. Neither depends on whether the labels are self-made. Â§7 draws the implication.

### 2.3 Indexing, budgets, R1 window [VERIFIED with one spec gap]

- compose's E list: E[0] = x classes, E[k] after k chain steps; vb_loss needs row 0 for the changed-slot mask and rows 1..Kb. No off-by-one.
- **Spec gap (NEEDS-FIX):** rec_targets builds (B, Kb + 1, T) with Kb = the batch max K (vbank.py 454â€“459); recode_targets HOLDs to Kb. E_self has n + 1 rows; the new branch must HOLD-pad to the **batch** Kb (â‰¤ 16), not the item's K, or the tensor fill fails / mis-aligns. Add to G-S2: "trainer E_self == recode_targets(root, Kb) bitwise on 1,000 kept pool items at Kb 16".
- K = min(n + U{0..4}, 16): n 12 gets K 12â€“16, n 9 gets 9â€“13. R1 window [n, K] uses n = the operator count (asserted in G-S1). VERIFIED.
- The S1 table assumes the stage-B self law (uniform n 9â€“12) for all 12k steps. With the stage-A curriculum (n 9â€“10 for up to 4k steps) the early passes t â‰¥ 11 are rarer than tabled. Descriptive; G-S3's 20k logged draws must be taken in stage B or the table recomputed with the curriculum.
- Cosmetic: support_oc.py's D3-T base row reports max_rho_plus_r 9 (a zero-weight self n 9 enters the law); it is 8.

---

## 3. The support argument S0, C2 framing, ABSORB (attack 3)

### 3.1 Prop S0 [VERIFIED as statements about targets]

- Identity: COMPOSE-VALID âˆ§ STEPS gives exactly one VOIDed handle per step for n steps; HOLD after; so Ï_{tâˆ’1} + r_t = n â‰¤ B on every training target. Strong form and (a), (b) follow by counting. Correct, and correctly scoped ("about targets").

### 3.2 Is DEPTH-BEYOND-SELF (n 13â€“16 mean â‰¥ 0.80) achievable?

- **Base rate.** Every VB cell (N1, N2, N3, D3-T, D3-F) STOPS-AT-10 with support 8 (213-26). D3-T's acc at Kâ€² n + 2 past its support: **+1 0.923, +2 0.603, +3 0.266, +4 0.127** (eval_vb acc_k: n 9 is 0.928 at Kâ€² 9 and flat after; n 11 is 0.273 at Kâ€² 10 and flat; n 12 â‰ˆ 0.12 at every Kâ€² â‰¥ 9, i.e. chance + bias). DBS â‰¥ 0.80 needs about 0.95 / 0.90 / 0.75 / 0.60 at B + 1..B + 4, i.e. a +3 / +4 margin no cell in the family has shown.
- Under S0b, no loss term asks for that. DBS can only come from a changed inductive bias (H_full, or the top of H_grow). The draft's P(DBS) â‰ˆ 0.11 is carried by P(H_full | ABSORB) = 0.10 Ã— 0.9 and P(H_grow) 0.30 Ã— 0.25, both [ASSUMED] without in-family evidence. **I would state P(DBS) â‰ˆ 0.03â€“0.06 and P(SD-C2) â‰ˆ 0.02â€“0.04.** The OC's qualitative message (a DBS pass is strong; a DBS null is weak against H_grow) stands.
- **T-support confound (descriptive).** T â‰ˆ 3n + 2 almost deterministically (item_space.json: T mean 38.0 at n 12, 41.0 / 44.0 / 47.0 / 50.0 at n 13â€“16). Round 1 trains positions up to 47, so the new T boundary falls at n 14â€“15, exactly where H_shift puts the stop. The base's decline at n 9â€“12 was not T-driven (81 % of n 10 items have T â‰¤ 33, yet 0.60), and the within-n size slope is âˆ’0.035; still, keep the T-split at n 13â€“16 in the report.

### 3.3 Framing

- **"Primary C2" (NEEDS-FIX, wording).** NORTH_STAR C2 = "accuracy rises with latent steps past the trained budget (Kâ€² > K), in every competent seed". DBS reads acc at Kâ€² n + 2 = 15â€“18 against trained K â‰¤ 16, so only n 15â€“16 are Kâ€² > K; and one seed cannot speak for "every competent seed". It is the project's DEPTH-GENERAL convention (Â§213 used it as D3-C2), which is fine, but the label should be "depth extrapolation past the self-labelled range (C2 proxy, one seed)", with Kâ€²-BEYOND (n 15â€“16) as the C2-literal row.
- **"Beyond gold-supervised depth" (NEEDS-FIX, wording).** Literally true (no gold field beyond n 8). Substantively the n 9â€“12 targets are gold-equivalent (Â§2.2), generated by a step the simulator taught. Honest phrase: *"latent depth extended from 8 to B by distilling the model's own re-rooted single-step chain into the loop; no gold labels beyond n 8, labels gold-equivalent at â‰¥ 0.999; in support; not C2, not C4"*.
- **Is ABSORB a meaningful success by itself for the north star? No, as a criterion.**
  - C2: in support by construction (S0c).
  - C4: unmet (the labeller's competence came from simulator per-step states).
  - C3: the comparison is against the model's own decode / re-encode chain, not a text-CoT model at matched compute.
  - Its value is as an **enabler** (a model trained to depth 12 for the next round or for D4) and as a capacity check (can the loop run 12 exact serial passes at T â‰¤ 47?). It is the likeliest positive (0.45) and passes under no extrapolation effect, so it must never be the headline.

---

## 4. Goodhart routes (attack 4)

| route | finding | verdict |
|---|---|---|
| Copying the chain's decode format | There is no separate format: COMPOSE-VALID targets are canonical E_t (Â§1.1â€“1.3). Pass-1 targets equal the model's own argmax (S3a) and only sharpen it | closed **VERIFIED** |
| New fixed deadline / stop at B (or at pass â‰¥ 13) | Zero loss on the support (S0b; S1: P(R pass \| t â‰¥ 13) = 0). This *is* H_shift and cannot be closed by loss | open, correctly stated |
| Memorising fresh items | Fresh-once is **violated by the pool sizing** under the self curriculum (rt_sd/pool_sizing.py, 200 sims): stage A advancing at 2k â†’ n 9 / 10 / 11 / 12 consumption 112k / 112k / 80k / 80k (max 115k); **stage A at its 4k cap (global steps) â†’ 128k / 128k / 64k / 64k (P(> 120k) = 1.0)**; cap counted in self steps â†’ 160k / 160k / 32k / 32k. The pool holds 120k per n | **NEEDS-FIX**: size n 9â€“10 at â‰¥ 165k and n 11â€“12 at â‰¥ 85k (or 170k uniform, +40 % local labelling, $0), state whether the cap counts global or self steps, and make exhaustion a hard stop (void), not "log reuse" |
| Starving the deepest window | Same sims: if stage A runs to its cap, n 12 gets only 64k items (â‰ˆ 1,000 batch-equivalents) of the 384k self items. ABSORB's binding n (12) is the least trained | NEEDS-FIX (disclose the per-n self-item counts; or cap stage A at â‰¤ 3k steps, or run uniform n 9â€“12 from the start: the base already solves n 9 at 0.92 and n 10 at 0.60, so stage A buys little) |
| FT-CTRL sees identical plain batches | True only if every self-step draw (n, the K offset U, the pool cursor) comes from rng_s, not rng / rng_k. The draft names rng_s for the coin only. If the self K offsets use rng_k (the existing draw() path, vbank.py 646), CTRL's plain U stream desynchronises after the first self step | **NEEDS-FIX (spec)**; G-S3's hash must include the U draws (as D4's G-P12) |
| CTRL's attribution value | STOP-MOVED (ACC-GAIN-AFTER-10 on n 11â€“14) follows from ABSORB alone: with n 11â€“12 in support at â‰ˆ 0.95 and Kâ€² 10 near chance there, the gain is â‰ˆ 0.6â€“0.7 under every hypothesis. So SELF-ATTRIBUTED attributes an in-support effect, not extrapolation. CTRL's real uses are FT-EFFECT on in-range erosion and MEM-TRADED attribution (and sharing with D4, whose FT-CTRL flags match: --lr 3e-5 --steps 12000 --cur 8 --c_sched 0,1, plain K â‰¤ 12 under either kmax) | say so in Â§5 |
| In-range drift (FT-DIVERGED) | The base ends at loss 2.4e-5, gn 7e-4, lr 0 (log.jsonl 48000). A fresh AdamW (the base's ckpt_last.pt is deleted by design, vbank.py 731â€“733) takes noise-normalised steps on the near-zero plain gradients. The censor (acc_n2 < 0.99 / acc_k24 < 0.97 at any n 1â€“8; base 1.000) reads it | adequate |
| KEEPS-SELF-STAB / MEM-TRADED | Self items give SLVA's history reads gradient on long, aged, unfinished trajectories (K up to 16), the D4 Lemma-D3 route; P(MEM-TRADED) â‰ˆ 0.3 is plausible. **Wording bug:** the draft (and D4 v1 line 326) says "hist24, n 3..8"; the scorer averages **n 3..6** (score_s213.py 125) and 0.782 is the n 3..6 mean | NEEDS-FIX (n 3..6) |
| Label-right, state-wrong targets | Â§1.4: ~1e-3 at n 9; invisible to label-level audits | NEEDS-FIX (state-level LABEL-AUDIT) |
| Mechanism labels not exhaustive | FULL / GROW / SHIFT / SHRINK leave a gap: mean in [0.40, 0.63) with STOP < 13 (or STOP-LOC unreadable) gets no label | NEEDS-FIX (add MIXED / OTHER) |

---

## 5. Implementation traps (attack 5)

| trap | finding | verdict |
|---|---|---|
| calibrate() under --init_from | The non-resume branch always runs m.calibrate + ratio_test, which overwrite beta / beta0 (vbank.py 602â€“610) | trap **VERIFIED**; G-S4 asserts it |
| **`cal` undefined** | The checkpoint dict stores `"calibration": cal` (vbank.py 710). If the init_from branch skips the calibration block, `cal` is never bound â†’ **NameError at the first checkpoint** (step ckpt_every: 1,000 in D3-T's args, 2,000 in the draft), after ~10â€“20 min of paid host time. G-S4 (--stop_at 1) never reaches a checkpoint | **NEEDS-FIX**: bind cal = the base calibration.json; smoke `--stop_at 2 --ckpt_every 1` and resume from it (G-S11's tested resume, run *before* launch) |
| **slva.mass_stats** | Called on every log step (step % 100) for each micro-batch (vbank.py 664â€“666); it calls VB.rec_targets(items) â†’ recode_targets(it["root"]) (slva.py 108â€“111). On a self batch: KeyError crash at the first self log step, or, if someone adds a parsed root "to make it run", gold E_t is computed (diagnostic only, but it breaks NO-GOLD's assert). Not in the draft's edit list (D4's red team caught the same family: "mass_stats map") | **NEEDS-FIX**: mass_stats uses E_self; G-S7 / G-S8 smokes with `--log_every 1` on a self batch |
| The in-memory answer key | vb_loss reads it["t"] for R1 (vbank.py 485) and it["n"] for the window (469). The draft's loader "rejects any record carrying root / t / gold keys". Both can hold only if the pool's label field has another name (e.g. `chain_label`) and the loader maps it to the in-memory "t" | **NEEDS-FIX (spec)**: name the mapping; the G-S0 grep must allow exactly that one line |
| NO-GOLD monkeypatch timing | N.generate uses the precomputed PAIRS table, not apply (nested170.py 26â€“45), so N.apply / N.value / recode_targets can be patched to raise **before** generation, not only after; the parsed input tree alone would let any apply-call recompute gold, so the patch is the real guard | strengthen (cheap) |
| strict VE.load | vbank_eval.load: make_model factory + load_state_dict (strict default), arch asserted; SLVA's parameters unchanged | **VERIFIED** |
| mem_check forcing K | forced = mem_check or time_steps â†’ Ks = kmax on the longest plain n â‰¤ 8 items (vbank.py 636â€“646); auto_micro measures the plain shape only (617â€“631). A self shape needs the new --mem_check_self | correctly flagged by the draft |
| **AdamW moments in mem_check** | mem_check runs one step (last = 1, line 648); exp_avg / exp_avg_sq are allocated by that step's opt.step, after the activation peak, so the peak omits them (~1 GiB for the ~125M fp32 parameters). D4 v1 fixed this with --mem_check_steps 2; this draft's G-S9 does not | **NEEDS-FIX** (2-step check); also align the budget (21.5 in G-S9 vs the base's mem_budget 21.0) |
| ~23 GiB projection at micro 1 | 2 + 0.58Â·16Â·47/33 + 0.0258Â·(47/33)Â²Â·Î£_{t=1..16}(t + 1) = 2 + 13.2 + 8.0 = 23.2 GiB | arithmetic **VERIFIED**; the scaling law is one measured point (11.28 GiB at K 12, T 33) plus [ASSUMED] exponents; G-S9 decides |
| Self CurGate bins | CurGate.update closes a bin when (step âˆ’ start + 1) % 1000 == 0 (vbank.py 522). Called only on self steps with the global step, a bin closes only when a self step lands on the boundary (~half of them), so bins are irregular and the 2-bin rule is mis-timed | **NEEDS-FIX**: drive it with a self-step counter, state the cap's unit |
| Labelling cost | 120k Ã— (0.0059 + 0.0081 + 0.0073 + 0.0085) s = 3,576 s â‰ˆ 1.0 h on the 3070 | **VERIFIED**; add +~40 % for the resized pool and compose's per-step Python checks (my CPU loop: compose scoring is small next to the forward) |
| Pool file on the host | ~85 B Ã— ~500k â‰ˆ 40 MB; deploys ship the working tree, so the pool must be committed / placed before the deploy and its sha logged | spec |
| Gate / pool card | GATE-1 and the pool must be labelled on the same card class and bs (fp32; ties at bs changes are rare but the project has seen them) | spec |

---

## 6. OC arithmetic and the predictions (attack 6)

**Arithmetic [VERIFIED, re-computed].** kept 0.9 Â· 0.9 Â· 0.85 = 0.6885; P(SD-SELF) = 0.65 Â· 0.6885 = **0.448**; P(MARGIN-GROWS) = 0.65 Â· (0.55 Â· 0.07 + 0.30 Â· 0.85 + 0.10 Â· 1.0) = **0.256**; P(DBS) = 0.65 Â· (0.55 Â· 0.01 + 0.30 Â· 0.25 + 0.10 Â· 0.9) = **0.111**; P(SD-C2) = **0.0763**; P(SD-C2-MEM) = **0.053**; P(STOP-MOVED) = **0.726**. (PROOFS_SELFDEPTH Â§12 gets the same.)

**Dependence.** "kept independent of H" and "DBS = MG = 0 without ABSORB" are implicit; both are reasonable and should be written.

**The priors that carry SD-C2** are P(H_full | ABSORB) = 0.10 and P(DBS | H_grow) = 0.25 (the H_grow curve's own mean is 0.69 < 0.80, so 0.25 needs a fat upper tail). No VB cell has shown a margin beyond +2 (Â§3.2). A shaded set (H_shift 0.65 / H_grow 0.25 / H_full 0.03 / SHRINK 0.07; P(DBS | H_grow) 0.15) gives **P(DBS) 0.046, P(MG) 0.19, P(SD-C2) 0.032, P(SD-C2-MEM) 0.022**. Either set leaves the design's logic intact; the shaded one is the honest headline for the user.

**Readings that pass under no extrapolation effect (H_shift):**

| reading | P(pass \| H_shift, ABSORB) | role |
|---|---|---|
| SD-SELF | 0.69 (kept factor) | in support by design: never evidence for C2 |
| STOP-MOVED, SELF-ATTRIBUTED | â‰ˆ 1 (implied by ABSORB, Â§4) | attributes the in-support absorption only |
| THINK-GAIN-BEYOND | â‰ˆ 1 (â‰ˆ 0.30 â‰¥ 0.10) | labelled "C2-literal" in Â§5: **misleading**, demote to descriptive |
| C2-TAIL | â‰ˆ 1 | trivial |
| DEPTH-GENERAL | â‰ˆ 1 | in support |
| **DBS** | â‰¤ 0.01 | discriminating |
| **MARGIN-GROWS** | 0.07 (curve prior; 24 SE by sampling, PROOFS Â§5) | discriminating |
| **Kâ€²-BEYOND** (n 15â€“16, Kâ€² n + 2 vs Kâ€² 16) | â‰ˆ 0 under H_shift (the answer is flat past the stop) | discriminating, and the only Kâ€² > K row: give it a pre-stated threshold (e.g. â‰¥ 0.10) and report it next to DBS as the C2-literal reading |

- **STOP-LOC [concur with PROOFS_SELFDEPTH Â§5, re-checked]:** with "readable = acc(Kâ€² n + 2) â‰¥ 0.18 at â‰¥ 2 n", the base reads unreadable at every k (k 10 has only n 11 at 0.266; n 12â€“14 are 0.127 / 0.141 / 0.127), not "10" as Â§5's table says. So SHIFT (which requires STOP-TO-SUPPORT) is unassignable under H_shift. Fix the readability rule and add an UNLABELLED / MIXED class.
- **GATE-q OC [VERIFIED]** for the two value terms (exact binomial reproduces 0.9986 / 0.504 / 0.0125 for R1). The Ï€^E and KEEP âˆ§ CV keep terms were outside the OC; Â§1.1 now measures both at â‰ˆ KEEP and â‰ˆ 0.999â€“1.000 at n 9â€“11, so P(GATE-1 pass) â‰ˆ 1.

---

## 7. Cheaper or simpler first (attack 7)

1. **Done here, $0, CPU:** S0-1 (a)â€“(c) (COMPOSE-VALID per n, value-slot agreement, Ï€^E) on real fresh decodes at n 9â€“12 (and 13â€“16 for the base-labelled fallback). It needed no GPU: ~4 min per 1,000 items per n on 3 CPU threads. The draft's "~5 min, 3070" step can be replaced by this file (or re-run on the 3070 for card-class consistency with the pool).
2. **$0 fixes before any spend:** pool sizing, `cal`, mass_stats, the label-key mapping, rng_s for self K, CurGate units, 2-step mem_check, STOP-LOC readability, state-level LABEL-AUDIT. All are spec / code-review items.
3. **A design question for the user, not a blocker: is self-labelling worth its code at round 1?** The labels are gold-equivalent (Â§2.2), so a **GOLD-DEPTH FT** (the same FT with gold per-step targets at n 9â€“12) is predicted to read the same as FT-SELF on every reading, within the â‰¤ 2.4 % selection and ~1e-3 state noise. It needs only init_from + per-type micro + the memory / time gates; it drops the pool builder, compose path, E_self branch, mass_stats map and NO-GOLD gates (roughly half the new code and most of the blockers above). It gives up only the procedural "no gold beyond n 8" property, which does not move C4 here. If the question is the mechanism (SHIFT vs GROW), GOLD-DEPTH is the cheaper-to-build answer; if the user values the self-labelling demonstration as a template for natural tasks (where no simulator exists), keep SELF and accept the extra code.
4. **CTRL.** Because STOP-MOVED and SELF-ATTRIBUTED follow from ABSORB, FT-CTRL's value here is FT-EFFECT on in-range erosion and MEM-TRADED attribution. If D4's FT-CTRL runs anyway (matching flags), share it (S1-A's note) rather than buying a second one; otherwise S1-B ($1.0) loses little for the extrapolation question.
5. **Free learning curve:** the off-host checkpoints every 2k (G-S11) can be evaluated descriptively at 4k / 8k (acc_n2 n 9â€“16 only, ~10 min each locally), which shows whether ABSORB is still climbing at 12k (the Â¬ABSORB follow-up's question) without a new run. Pre-state them as descriptive.

---

## 8. Relation to PROOFS_SELFDEPTH.md (written in parallel, 02:04â€“02:08 EDT)

- Concur (found independently): pool undersized at n 9â€“10 (my sims: 128k at the 4k cap, 160k if counted in self steps); rng_s for self K; it["t"] routing; the 1-step mem_check omits AdamW state and 21.5 â‰  21.0; COMPOSE-VALID's stated premise is a non sequitur (string-level chain_exact); Ï€^E â‰¤ Ï€ with compensated chains; OC arithmetic; STOP-LOC unreadable at base and under H_shift; support_oc's cosmetic max_rho_plus_r 9.
- **New here:** (i) the real-decode measurement that settles COMPOSE-VALID â‰ˆ KEEP and value placement at n 9â€“12 (PROOFS leaves it [CONJECTURE]); (ii) a measured compensated chain on fresh n 9 (Ï€^E 0.999) and the 274 / 2,875 in the draft's own fixture; (iii) the `cal` NameError at the first checkpoint; (iv) slva.mass_stats crashing / leaking on self batches at the first log step; (v) self-CurGate bins mis-timed when driven by the global step on self steps only; (vi) KEEPS-SELF-STAB's n range (3..6, not 3..8); (vii) gold-equivalence â†’ STOP-MOVED / SELF-ATTRIBUTED / THINK-GAIN-BEYOND carry no extrapolation information, and Kâ€²-BEYOND is the discriminating C2-literal row; (viii) shaded priors for DBS / SD-C2; (ix) the GOLD-DEPTH alternative.

---

## 9. Required fixes, in order (all $0)

1. **Pool sizing** (Â§4): â‰¥ 135k kept at n 9â€“10 and â‰¥ 85k at n 11â€“12 (or cap stage A at â‰¤ 3k steps, or drop stage A); state the cap's unit; exhaustion = hard stop.
2. **`cal` at init_from** (Â§5): bind it from the base calibration.json; smoke `--stop_at 2 --ckpt_every 1` plus a resume, before launch.
3. **mass_stats and the label key** (Â§5): mass_stats reads E_self on self batches; the pool's label field is mapped to the in-memory "t" in one named line (G-S0 grep allows exactly that); smokes with `--log_every 1` on a self batch.
4. **STOP-LOC readability + an exhaustive mechanism partition** (Â§6; PROOFS Â§5): SHIFT is unassignable under H_shift as written.
5. **State-level precision** (Â§1.4): LABEL-AUDIT and Prop S2 on E_self == gold E_t (Ï€^E), not on labels; report compensated chains (1 / 5,852 measured here; 9.5 % of state-corrupted chains in the draft's own fixture).
6. **Streams and gates** (Â§4, Â§5): self n / U / cursor from rng_s; G-S3 hashes include U; self CurGate on a self-step counter; mem_check_self runs 2 steps (AdamW state), budget 21.0; HOLD-pad E_self to the batch Kb (G-S2 at Kb 16).
7. **Wording** (Â§3.3, Â§6): DBS = "depth extrapolation past the self range (C2 proxy, one seed)"; Kâ€²-BEYOND gets a threshold and is the C2-literal row; THINK-GAIN-BEYOND demoted to descriptive; ABSORB phrased as distillation of the model's own (simulator-taught) chain, gold-equivalent labels, in support; KEEPS-SELF-STAB on n 3..6; replace the chain_exact premise with Â§1.1's measurement.
8. **OC headline** (Â§6): state the shaded priors beside the draft's (P(DBS) â‰ˆ 0.05, P(SD-C2) â‰ˆ 0.03), and that every positive except DBS / MARGIN-GROWS / Kâ€²-BEYOND passes under H_shift.
9. **Design choice for the user** (Â§7): SELF vs GOLD-DEPTH (same predicted readings, about half the new code); CTRL shared with D4 or dropped.

*Red team completed 2026-10-17 02:30 EDT (clock). CPU only; no training, no GPU, no repo edits (the one stray __pycache__ in docs/.../scripts was removed at 02:02).*
