# FP_v1 — §270 FULL-PERM: does extending the §263 recipe's permutation to all sentences make C1 hold under any sentence order?

Drafted 2026-11-04 00:37 EST for Theory (Opus draft agent); REGISTERED by Theory as §270 on 2026-11-04 00:40 EST with the init choice (bun_sX_init, single-flag diff vs of_sX) adopted. Read-only: no runs and no reads of stored outputs. The only numbers quoted are from Note 263-3 and Note 262-3.
This draft closes **§263's scope gap**. It is incremental (C1 scope, one lineage, ~$3).
- §270 is a placeholder number. `scratch/s270_*` is already Experiments' build serial for §263, so the build needs a fresh serial.
- Run names fp_s1 / fp_s2 are unused (checked with `ls boxes/runs_fsq`).

## 0. Motivation (DATA, POST-HOC w.r.t. this design)

of_s1 / of_s2 (§263) are ORDER-FREE-C1 2/2: TEST-PERM d 4 = 0.982 / 0.961 and LATE-PREMISE >= 0.979. The graded shuffle in Note 263-3 is descriptive and **post-hoc** for §270:

| of seed | independent-only shuffle d 4 | full shuffle d 2 / 3 / 4 |
|---|---|---|
| s1 | 0.967 | 0.806 / 0.596 / 0.462 |
| s2 | 0.878 | 0.695 / 0.468 / 0.376 |

The scope of the §263 claim is "op sentences in any order". This draft asks whether "any sentence order" can be had with the same recipe.

## 1. Mechanism and math backing

**Setup (from the code).**
- An item has sentences Σ = {S_0, G_t (givens of `_v` ops), S_1..S_d (ops), D_k (distractors)} plus the question q. The text is the sentences in `it["order"]`, then q.
- `NW._finish` always appends q last.
- The training stream uses `order_mode "forward"` (`ow_words.forward_order`): S_0, then [G_t] S_t for t = 1..d, with each D_k at a uniform slot.
- `perm_render.permuted` (TEST-PERM and §263 training) reassigns S_1..S_d over the op slots only. S_0, G_t and D_k keep their slots.
- Ask items (`inject_aw.ask_items`) use the same op-only `PR.permuted`.

**Code finding: ask-item coverage** (`boxes/ask_render.ask_of`; class = sha1(sig) mod 3, or mod 4 at d 2):
- (i) S_1 → v_1;
- (ii) **S_0 → v_0**;
- (iii) the first distractor; else `givens[0]`; else S_0;
- (iv) d 2 only: S_2, with S_1's sentence replaced by a given sentence.

So S_0 is queried in ~1/3 of ask items, and more through the fallback. Givens are queried only through the class-(iii) fallback: P(no distractor) = 1/3 (n_dis ~ U{0,1,2}), times P(a `_v` op exists), and only the first given. That is ≲ 1/9 of ask items. Ops S_2..S_d are never asked, except class (iv) at d 2.

The decisive point is placement, not coverage. Ask items are permuted op-only, so **every S_0 lookup ever trained has S_0 before all op and given sentences**. A position shortcut for S_0 is therefore not excluded by the data.

**Claims.**
- **A1 (PROVED; `OW.solve` regex solver, asserted by `PR.check` in every render).** The answer depends only on the multiset Σ. Any order of Σ with q last is a valid item. All-sentence permutation needs no new target code: `R_step` and the aux values are order-free, and the cue names S_t.
- **A2 (DERIVED from the code; the strong form of hypothesis H-C is false).** "A given placed after the op that uses it" is not out-of-distribution for §263.
  - G_t keeps the slot just before op slot t, and S_t moves to op slot σ(t).
  - Under a uniform non-identity σ, P(σ(t) < t) ≈ (t-1)/d, which is 0.375 averaged over t at d 4.
  - So ~19% of the d 4 training items that have a given contain that case (perm_p 0.5).
- **A3 (DERIVED from the code).** Four layout invariants hold in every §263 training item (chain items and ask items alike) and are broken by the full shuffle:
  - N1: S_0 precedes every op and every given.
  - N2: every base given precedes at least one op.
  - N3: no op sentence precedes S_0.
  - N4: the last sentence before q is an op or a distractor.
  - The inserts (perm-then-insert) keep N1-N4. They are placed relative to S labels, after the S_0 slot.
- **A4 (DERIVED, counting).** The share of full-shuffle orders that §263's training support could produce:
  - d 4 with 1 given and 1 distractor: 4!·7 / 7! = 168 / 5040 = 3.3%;
  - with 2 givens and 2 distractors: 4!·72 / 9! = 0.48%;
  - with no givens and no distractors: 24 / 120 = 20%.
  - So the full-shuffle eval is ≥ 80% off-support at d 4, and usually more than 95%.
- **A5 (PROVED).** A uniform permutation of all n sentences induces a uniform relative order on the op subset. So `perm_scope all` trains every op order, TEST-PERM's and LATE-PREMISE's included, at the same per-order rate as op-scope (1/d! vs 1/(d! - 1)), while also breaking N1-N4. Bound: P(S_0 first | an all-scope item) = 1/n. Clean TEST2 layouts stay in training through the unpermuted 50%, as in §263.
- **H-B (CONJECTURE; the main hypothesis).**
  - The §263 loop selects the pass-t sentence by name among positions with the trained op layout (N1-N4). It is not fully content-addressed: GPT-2's learned absolute positions plus a layout prior.
  - Supporting post-hoc DATA:
    - the independent shuffle keeps op slots and breaks only N1 / N2, and costs 0.015 / 0.089 at d 4;
    - the full shuffle also breaks N3 / N4 and op contiguity, and costs 0.52 / 0.59.
  - So op-layout breaks dominate S_0 / given placement.
- **H-A (CONJECTURE, minor).** S_0 is read partly by position: the trained S_0 lookups all have S_0 first. The s2 independent-shuffle cost (0.089) is the evidence.
- **Why perm_scope all should work (DERIVED-UNDER-ASSUMPTIONS).** Assume:
  - (a) §263's mechanism generalises across orders within its training support (DATA: TEST-PERM and LATE-PREMISE ≥ 0.96 on held-out items and orders);
  - (b) the failure is support coverage (A3 / A4), not capacity.
  - Then uniform all-sentence perm on the same 50% of items, including the ask items (whose S_0 lookups then see S_0 at every position), makes the full-shuffle distribution in-support. The §263 generalisation then applies.
  - (b) is the bet. Literature (4a8ce800, LIT_S263_ORDERFREE_RESULT): "perm_scope all goes beyond DAG-valid permutations (He et al. 2502.19907 permute independent premises only); open bet". Perm alone failed in this lineage (§252 NP, §254 PERM-BLOCKER). Here perm is extended inside a recipe that already works, which de-risks it but does not prove it.
- **Falsifiable predictions** (Theory's pre-stated probabilities; per seed unless marked):
  - P1: FULL-ORDER passes (fresh full-perm d 2-4 ≥ 0.85) in both seeds: **0.50**.
  - P2: full_perm d 4 gain vs the same-seed of on the same fresh items (paired by sig) ≥ +0.30: **0.75**.
  - P3: TEST-PERM d 2-4 ≥ 0.90 in both seeds: **0.85**.
  - P4: TEST2 d 1-4 ≥ 0.98 in both seeds: **0.85**.
  - P5: C2 raw retention (d 6 and d 7 ≥ of - 0.05) in both seeds: **0.55**.
  - P6 (descriptive mechanism read): on of, full_perm d 4 errors are ≥ 2x as frequent in items that break N3 or N4 as in items that keep them: **0.65**. If fp also passes, this supports H-B over H-A.
  - Theory's prior that the effect is real: ~0.6.

## 2. Arms

**Recommendation: train from the same init as of (bun_sX_init), not warm from of.** One flag differs from of_s1's args.json, and of_s1 / of_s2 are the exact same-seed controls, already run.

Warm-from-of (of + 8k) adds the confound of 8k more steps. That would need a CTRL arm (of + 8k op-scope, ~$1.5), and it gives a 2-flag args diff. It is rejected here; Theory may flip it.

| Arm | Args | Steps / card |
|---|---|---|
| fp_s1 / fp_s2 | of_s1 / of_s2 args.json + `--perm_scope all` | 8k (the §263 schedule; the recipe formed by 2k in the §262 pilot), RTX 3090, micro 2 pinned, no auto_micro |

- **Args diff vs of_sX/args.json** (single-change; Audit G-D checks it with `--args_only --args_ref`):
  - {perm_scope: op → all}, plus bookkeeping {out, args_ref};
  - init_model / init_model_sha (bun_sX_init, 1ad3517b… for s1), seed, perm_p 0.5, ask_p 0.25, ask_perm_p 0.5, distract_mix, hold_inc 1.6 and every other arg unchanged.
- **No CTRL arm.** of_sX has every component except the scope change, and it is exposed to the same eval. A further control would receive no extra intervention.
- **No ask-class extension.** It would be a second change. The ask items' *placement* is changed by the same flag:
  - the scope applies at both perm call sites, so it is one semantic "what gets permuted";
  - the ask class rule is unchanged.
  - Conditional follow-up (not registered here): if FULL-ORDER fails and the fp errors concentrate in items with a displaced given (descriptive split, §3), §270b adds given ask classes.
- **`--perm_scope all` render semantics** (new `perm_render.permuted_all(it, rng)`):
  - d ≥ 2 only, with **the same coin** as §263 (`default_rng((seed, SEED_PERM_TRAIN + 1, step, k)) < perm_p`), so the same items are permuted;
  - new order = `rng.permutation` of all of `it["order"]` (S_0, G_t, S_t, D_k), redrawn while it equals the identity;
  - rendered by `PR.render`, then `PR.check` (fields equal, `OW.solve == answer`), and `perm = 1`;
  - training rng `default_rng((seed, 27000, step, k))`; ask items `default_rng((seed, 27001, step, k))` (replaces SEED_ASK_PERM's draw under scope all; the ask coin is unchanged);
  - then the §257 inserts as now (perm-then-insert, placed by S labels). Inserts are not shuffled; disclosed;
  - **q stays last.** `OW.solve` parses the question as the last sentence, the per-pass cue is appended after the text in both formats, and moving q would change the readout format, not the order. The claim is scoped "any order of the statements, question last".
- **Note 270-1 (MED-2):** the multiset check below is reject-and-resample (next stream item), not an assert. A cumulative `multiset_rejects` counter goes to log.jsonl and train_summary.json, and G3 reports the reject rate. This is a disclosed small difference vs of.
- **Disjointness (existing excl protocol plus two asserts, scope all only):**
  - training texts ∉ TP_TEXTS ∪ INS_TEXTS (as now) ∪ FP_TEXTS (the new eval part's texts, d 2-6) ∪ SG_TEXTS (shuffle_graded's two families);
  - no training base item shares its sentence multiset (`tuple(sorted(statements))`) with any TEST2 d 1-8 item. The stream's sig EXCL keys on the forward text, so it does not cover reorders by itself. This assert closes that gap.
- **Seeds 27000 / 27001 / 27010:** grep 2026-11-04 over boxes/*.py, theory/*.py, scratch/*.py and scratch/*/*.py found no hits. The highest boxes family in use is 26430. Re-grep at build.
- **Seeds:** 2 (s1, s2) for the binding C1 label. C2 claims need ≥ 3 seeds (Audit rule), so **C2 is retention-only** here.

## 3. Eval parts

All parts reuse the §263 list:
- in args: acc, test_perm, insert_ho, swap_kd, lp5h, stab;
- run after, as in §263's chain: late_premise, long_k128, name_swap, shuffle_graded.

Plus one new part, **`full_perm`**, in a new `boxes/fp_parts.py`. `of_parts.py` stays byte-identical.
- **Items and orders:** REG TEST2 d 2-6, all items. Item i's order is `default_rng((27010, d, i)).permutation(len(order))` over all statements, redrawn from the same rng while it equals the identity. q is last; `PR.render` + `PR.check`.
- **Readouts:** fp32, no_grad (ast-check the decorator); acc at K' = think_k (8) and K' 2.
- **Per-item record:** sig, pred, ans, order, and layout flags:
  - s0_pos;
  - n3 (an op before S_0);
  - n4 (last pre-q sentence is S_0 or a base given);
  - given_after_consumer;
  - n_viol (N1-N4).
- **Output:** by_d {d: {acc, acc_k2, n, changed, by_flag}}.
- **Status:** d 2-4 binding; d 5-6 descriptive (C2 × order).
- **The fresh family 27010 is distinct from:**
  - TEST-PERM (25150 + d);
  - shuffle_graded (26310 / 26311), whose numbers are now post-hoc;
  - training (27000 / 27001).
- **Also run eval-only on of_s1 / of_s2** (~10 min each): the paired same-item baseline for P2 / P6, read as a paired difference by sig (ST4, if statlib has landed).

## 4. Binding per seed (theory/score_fp.py; label FULL-ORDER-C1)

| Crit | Field (exact paths; check each with a field-path grep against the module) | Bar |
|---|---|---|
| FULL-ORDER | eval_full_perm by_d[d].acc, d 2-4 | ≥ 0.85 each |
| RETAIN-ORDER | eval_test_perm by_d[d].acc_kprime['8'], d 2-4; eval_late_premise by_d[d].acc, d 2-4 | ≥ 0.90; ≥ 0.85 |
| RETAIN | eval_acc d 1-4 acc ≥ 0.98; NOT-DECODER by_d['4'].acc_kprime['2'] ≤ 0.20; SWAP4 follow_U ≥ 0.50 and ≥ floor + 0.30; LP5-D4 intact - mean_at_t ≥ 0.30 | as §263 |
| C2-RETAIN | eval_acc d 6, d 7 ≥ **of_sX** (same seed) - 0.05; c2 rows with NEAR-SEED (sigma_seed 0.090, Note 269-2) | raw letter binding; claim NEAR-SEED / PASS / FAIL |
| C6 | eval_long_k128 d 1-6: acc['64'], acc['128'] ≥ acc['8'] - 0.02 | as §263 |
| CONTENT | eval_insert_ho fam['H-CHAINSRC-OP'] acc8 ≥ 0.85, min by_pos ≥ 0.80 | as §263 |
| STABLE | score_ao.stable | — |

- **Labels:** FULL-ORDER-C1 per seed, otherwise `FAIL: <crits>`. Per arm: **2/2 = claim** (SOTA scope → "any statement order, question last"); 1/2 = SPLIT, no claim; 0/2 = FAIL.
- **NEAR flag:** every binding cell carries a Wilson 95% CI (n from items) and a NEAR flag (CI contains the bar). The letter is scored on the raw value.
- **C1 causal gates:** NOT-DECODER (K' 2 on TEST2 d 4) is binding; the full_perm K' 2 is descriptive. C1 causal gates at d 6-7 (Audit standing rule 3) are needed only when C1 and C2 are claimed together. C2 is retention-only here, so none are run, and this is stated in the registration.
- **Descriptive:**
  - full_perm d 5-6;
  - by_flag splits (P6, H-A vs H-B);
  - the of-vs-fp paired diff;
  - shuffle_graded (comparable to Note 263-3);
  - name_swap;
  - C5 r = 3;
  - an ask acc by class, log-only.

## 5. OC block (2 seeds; scope one lineage)

- **"Effect"** = all-scope training transfers: true full-order d 4 ≳ 0.92.
  - Per seed: P(FULL-ORDER | effect) ≈ 0.97 (Wilson half-width ~0.017 at n 1000);
  - × RETAIN-ORDER 0.9 × RETAIN 0.95 × C2-RETAIN 0.7 × the rest 0.95 → **≈ 0.55** per seed;
  - **P(2/2 | effect) ≈ 0.40** (seeds correlated, so this is above 0.55² = 0.30).
- **"No effect"** = full order stays near of (d 4 0.38-0.46; true < 0.80). Observing ≥ 0.85 at n 1000 is > 3 SE away: **P(pass per seed | no effect) ≤ 0.01; P(2/2) ≤ 0.001.**
- **Overall:** P(2/2) ≈ 0.6 × 0.40 ≈ 0.24; P(≥ 1/2) ≈ 0.4. The main failure mode under "effect" is the C2 retention cost (the §262 pilot showed a d 6 dip at 2k).
- **Censoring (conservative):** every binding readout is fixed-K' (no halting), so there are no censored items. A run or eval that never lands gets one VOID-RERUN, then counts as **FAIL**; never pending at final scoring. A solve-assert failure is a build bug, not an excluded item. C5 descriptive: never-halt = wrong at its budget.
- **Literature (4a8ce800):** "perm_scope all goes beyond DAG-valid permutations; open bet". No "first" claim on order invariance.
- **Scope:**
  - plan given (the cue names S_t); the aux names steps in training;
  - REG frames 0-5; inserts in their trained local placement; question last;
  - d ≤ 4 binding;
  - **one lineage** (bun → of → fp).

## 6. Cost and guard

- **Per cell:** 8000 × 1.65 s/step (§263 measured) = 3.67 h; × $0.21/h = **$0.77 training**.
- **Evals and setup:** ~$0.6-0.7 per cell, from §263's realised $3.99 for 2 cells less ~$1.0 of idle time.
- **full_perm on of_s1 / s2:** ~$0.1.
- **Total ≈ $3.0.** Cell cap $1.6; **guard $4** (< $30, Audit auto-approve).
- **Wall cap 6.5 h per cell.** ckpt_every 1000 with off-host mirror per run name. keep_final_ckpt off; delete model.pt after verify unless FULL-ORDER-C1, in which case keep it as the lineage head.
- **Host:** Vast RTX 3090 ≥ 16 GB, one filler at a time, drain waiters armed at launch.

## 7. Build spec (Experiments)

**Flag and files.**
- `inject_aw.py`: `--perm_scope {op, all}` (default op), asserted to need `--perm_p` or `--ask_p`.
  - In the perm-then-insert branch, the plain `--perm_p` branch and `ask_items`: scope all calls `PR.train_perm_all(r_, seed, step, k)` / `PR.permuted_all(...)` with seeds 27000 / 27001.
  - Scope op runs the existing lines untouched.
  - The FP_TEXTS / SG_TEXTS / multiset asserts run under scope all only.
- `perm_render.py`: add `SEED_PERM_ALL_TRAIN, SEED_ASK_PERM_ALL = 27000, 27001`, `permuted_all`, `train_perm_all`. Existing functions unchanged.
- New `boxes/fp_parts.py` (`SEED_FP = 27010`; part_full_perm). It is registered in inject_aw_eval's run(). Ship it as a blob and list it in host_shas.
- Log-only: an `ask_by_class` field under scope all.
- Python 3.11 only on hosts; no 3.12+ APIs.

**Gates (all before launch; Audit G-D on the diff, reading full functions):**
- **G1 OFFBIT:**
  - `--perm_scope op` (and omitted) gives bitwise-identical token batches (chain + ask items) to the §263 module (sha 4d185e6d…) for steps 1-300 at seeds 1 / 2. This is CPU only; the flag touches data only.
  - Plus `--args_only` diff vs of_s1 = {perm_scope, out, args_ref}.
  - Optional: a 20-step host loss trace equal under the deterministic wrap.
- **G2 RENDER** (20k all-scope items, d 2-4, scope all):
  - solve and fields asserted; sentence multiset equal; q last; never the identity;
  - op relative order uniform (χ² over the 6 orders at d 3, p > 0.01);
  - s0_pos roughly uniform;
  - N1-N4 break rates reported;
  - targets / cues / aux equal to the unpermuted item (the DISTRACT assert pattern);
  - inserts after perm solve.
- **G3 DISJOINT:** the four text sets plus the multiset assert hold on 2k training steps' items; seeds re-grepped unused.
- **G4 GRADFLOW:** no new loss term, so gradflow_check is not required. The smoke still confirms nonzero grads on the ':' ask CE and the aux, as in §263.
- **G5 SMOKE AT FULL DATA SCALE:** run the host lines locally on the real sets (EXCL n 206,599), CPU:
  - `--stop_at 3` with scope all;
  - plus `part_full_perm` on of_s1/model.pt at d 2, full n.
  - Scratch files under D:/Latent/scratch/_tmp; delete the .pt after.

**Scorer `theory/score_fp.py`** (from score_of.py):
- RUNS = {fp_s1: (of_s1, bun_s1), fp_s2: (of_s2, bun_s2)}. The C2 bar uses of, with bun descriptive.
- Use statlib `SL.cell` / `seed_cell` (template v7) if theory/statlib.py has landed by the scorer commit (it is absent on 2026-11-04). Otherwise use score_of's `c2_row` and a local Wilson, and record `near_def`.
- `--final` turns any missing binding file into FAIL (censoring). The verdict string is "FULL-ORDER-C1 k/2 (...); C2 claim ...".
- **Fixtures (19):**
  1. all pass → 2/2.
  2. full_perm d 4 0.84 → FAIL: FULL-ORDER.
  3. only d 2 below → FAIL: FULL-ORDER.
  4. TEST-PERM d 4 0.89 → FAIL: RETAIN-ORDER.
  5. LATE-PREMISE 0.84 → FAIL: RETAIN-ORDER.
  6. TEST2 d 3 0.97 → FAIL: RETAIN.
  7. K' 2 0.6 → FAIL: RETAIN.
  8. SWAP4 → FAIL: RETAIN.
  9. LP5 → FAIL: RETAIN.
  10. C2 d 7 far below of - 0.05 → FAIL: C2-RETAIN, claim FAIL.
  11. C2 within sigma_seed → claim NEAR-SEED.
  12. the C2 bar reads of, not bun (bun and of differ so that the wrong ref flips the label).
  13. C6 → FAIL: C6.
  14. CONTENT acc8 → FAIL: CONTENT.
  15. CONTENT by_pos → FAIL: CONTENT.
  16. blank log → pending (MED-1).
  17. a missing eval_full_perm → pending; with `--final` → FAIL: VOID.
  18. full_perm d 4 0.86, n 1000 → PASS letter + NEAR flag.
  19. one seed fails → "FULL-ORDER-C1 1/2".
