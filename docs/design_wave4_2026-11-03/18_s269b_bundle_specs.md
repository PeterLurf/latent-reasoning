# §269b DIAGNOSTIC BUNDLE B: scorer-ready specs for the five deferred eval-only parts on bun (wave-4 lead design, 2026-11-03 17:25 EST)

Status: design only. No code, no outputs, no stored-output numbers were computed. Every threshold below is pre-stated; Audit's ack precedes any build.

## 1. Goal and backlog IDs

§269 (approved 11-03, guard $1.5) deferred six reads to a "§269b". This document specifies five of them as independent eval-only parts on `bun_s1` (confirmation on `bun_s2`), in the §269 form: one queue line and one JSON per part, descriptive labels that route backlog lanes, one stdlib scorer (`theory/score_dgb.py`, listed in §6, fixtures 23/23 run from the listing).

| Part | Backlog ID (§13.1 / §13.3) | Question it decides | Routes |
|---|---|---|---|
| noise | RB2 STATE-NOISE-RADIUS | is the hold-pass basin narrower than the compute-pass basin (sigma_50 per pass)? | D10 / D11 memory design, TS2 NOISE-TRAIN-SC, the sigma of TS1 |
| subs | CG1 SUB_S ZERO-SHOT | is a step operator x source (factorised) or a nearest-template lookup? | CG2 / CG3 / CG4, T-lane op coverage, N1 NUM-100 |
| tta | TS3 TTA-VOTE | are d 7-8 errors render-dependent (recoverable by voting) or systematic? | TS4 LRM-SELECT, TH3 selection margin vs memory redesign |
| sclat | TS1 SC-LATENT | same question under state noise (oracle@N over noise rollouts) | TS2, TS4; a NOISE-DESTRUCTIVE read kills TS1 |
| noop | RB1 NOOP-CLAUSE | does an in-sentence irrelevant number / same-name other-unit fact break the step? | the $0.6 noop FT arm before Lane D (GSM-NoOp risk), §265 |

Not covered here (left for a §269c): I1 op-select heads, TH2 mirror pairs, RB3 occlusion footprint.

## 2. Exact design

### 2.0 Shared setting

- Model `bun_s1` (args read: `think_k 8`, `d_max 4`, `cue step`, `write_src answer`, `mul_s_p 0` = REG law, `distract_mix s256,chain,givop`, `fast_add true`); `bun_s2` confirms. Cloud host only (local RAM bit-flips), fp32, `@torch.no_grad`, `eval_bs 100`, K' = `think_k` = 8 unless stated.
- Items: REG TEST2 (`inject_ow.build_sets(mul_s_p=0)`: `OW.Stream(23800, frames 0-5, order topo, dag)`, 1,000 per d), or new streams under the same law with the eval exclusion `sets["excl"] | TEST2 sigs` (the `part_stab_fresh` pattern). New seeds, unused in `boxes/*.py`, `theory/*.py`, `scratch/*.py` (grep 11-03 17:23): `SEED_NOISE 26950, SEED_SUBS 26951, SEED_TTA 26952, SEED_SCL 26953, SEED_NOOP 26954`.
- State: `m.mem.zero_state` gives `S (B, 12, 64, 64)` for bun (`add_norm None`, so no z block): blocks 0-5 additive, 6-11 delta (`deltamem.pack_state`). All state edits go through `passes(..., edit=)`; readouts through `passes(..., on_state=)` and `AW.ws_readout`.
- "Relative per-head noise" (noise, sclat): for item i and head block h, `S[i,h] += sigma * ||S[i,h]||_F * eps_h / ||eps_h||_F`, `eps_h ~ N(0, I)` of shape (64, 64) from `np.random.default_rng(key)`, cast to the state dtype. sigma = 0 is the identity path (`S + 0 * x == S` bitwise in fp32).
- Chance = 1/11 for every accuracy. Wilson 95% CIs on every share; a NEAR flag when the CI straddles the label bar.
- Every rewritten or perturbed text is checked by the independent solver (`OW.solve(text) == answer`; for noop on the stripped text, §2.5). Set shas are committed from a twice-identical local build (`scratch/s269b_shas.json`) and asserted on the host (the `part_insert_ho_s256` pattern).

### 2.1 noise (RB2 STATE-NOISE-RADIUS): `eval_noise.json`

- Items: REG TEST2 d in {2, 3, 4, 6}, the first 500 per d (Note 269-2: 500 suffice at d >= 6; d 6 is descriptive extrapolation).
- Intervention: for each injection pass t in 1..8 and each sigma on the grid `G = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]`, one `passes` call with `edit = (S + noise after pass t, else None)`; the noise direction `eps` is drawn once per (d, t, item) from `default_rng((26950, d, t, i))` and scaled by sigma, so each item's accuracy curve over sigma is along one ray. Answer read at K' 8. Intact and sigma-0 (identity) arms run through the same `edit` path.
- Family arms (descriptive, d 4 only, t in {1, 2, 3, 6, 8}): noise on blocks 0-5 only (`add`) or 6-11 only (`del`), same grid.
- JSON: `{"identity_ok": bool (identity preds == intact preds, every d), "by_d": {d: {"n": 500, "intact": acc, "grid": G, "acc": {t: [acc per sigma]}}}, "fam_d4": {"add"|"del": {t: [acc per sigma]}}, "items": {d: {"sig": [...], "pred_intact": [...], "fail_sigma_idx": {t: [first grid index wrong, or -1]}}}}`.
- Scorer: `sigma_50(d, t)` = the sigma at which acc crosses `intact - 0.5 * (intact - 1/11)`, log-linear between grid points; never crossing = censored (reported; treated as `2 x grid max` = a lower bound). Compute passes = t <= d - 1, hold passes = t >= d + 2, medians pooled over d 3-4 (d 2 has one compute pass; d 6 reported only).
- Labels: INVALID (identity fails) / INSENSITIVE (both medians censored) / FRAGILE (compute median <= 0.05) / NARROWING (hold <= 0.5 compute) / WIDENING (hold >= 2 compute) / FLAT.
- Routing: NARROWING -> the late-pass basin is a mechanism of the d 7-8 decay: D10 DG-ADD / D11 RMS-MIX and TS2 NOISE-TRAIN-SC move up in Lane E, and TS1's sigma is read from the compute median; FLAT / WIDENING -> late errors are not a radius problem, §269 triage's label routes alone; FRAGILE -> the decoder reads a sharp code, SC-LATENT (TS1) is dropped; INSENSITIVE -> the K' 8 answer does not depend on the state at that pass: a C1 alarm, re-gate with the §259 swap gates before anything else.

### 2.2 subs (CG1 SUB_S ZERO-SHOT): `eval_subs.json`

- Fact from the code: `NW.OPS = {add,sub,mul} x {c,v} + add_s + mul_s`; there is no `sub_s`, and `FRAMES["add_s"] = FRAMES["add_v"]`, `FRAMES["mul_s"] = FRAMES["mul_v"]` (the partner `{C}` is an earlier spine name). bun's law (`mul_s_p 0`) draws regular ops from `OPS_REG` (incl. `add_s`, no `mul_s`); its ancestors `aw_plaw2_s1` / `aw_plaw4_s1` had no `mul_s_p` key (law 1, forced `mul_s` at d >= 3), `rcb_plaw4_s1`, `dis_s1`, `dch_s1`, `bun` are law 0. So `sub_s` is the never-trained cell and `mul_s` is a stale cell (trained two inits back).
- Base pool: `OW.Stream(26951, frames FRAME_TRAIN, order topo, dag, mul_s_p 0, excl = sets["excl"] | TEST2 sigs)`, d = 4, step t* = 3, kept iff (i) `ops[2]` is a `*_c` op (no given sentence to remove), (ii) source j drawn from `default_rng((26951, 1, i))` in {0, 1} (stratified: the first 500 kept with j 0, the first 500 with j 1), (iii) `v_2, v_j in 2..10` and the five readouts `{v_2 - v_j, v_2 + v_j, v_j - v_2, v_2, v_j} mod 11` pairwise distinct (so `v_j != v_2`, `v_2 != 2 v_j`, `v_j != 2 v_2`), (iv) the recomputed chain for every one of the nine cells has no `mul` on a zero state. A second pool at d 3 (t* 2, j 0, n 500) is descriptive.
- Nine cells by rewrite of the base item's step t* (`operator in {add, sub, mul}` x `source in {c, v, s}`), value-matched: the operand value is `v_j` in every cell, so the three sources of one operator row give identical `v_3`, downstream values and answer. c: constant `c = v_j`, frame `FRAMES[op_c][f]` with the base step's frame index f; v: a new given "Y has v_j u." (Y from the unused name pool) inserted in the order directly before S_3's sentence, frame `FRAMES[op_v][f]` with `C = Y`; s: frame `FRAMES[op_v][f]` with `C = spine[j]` (no new sentence). `sub_s` is rendered by the part with `FRAMES["sub_v"][f]`; `NW.FRAMES` is not edited. `NW.solve` already resolves `{C}` as a name, so the assert holds for every cell. The cue at pass 3 is unchanged (`cue_str` depends only on names and unit).
- Readouts: acc at K' 8; WS at pass t* (`AW.ws_readout`'s ':' argmax over the 11 number tokens vs the cell's `v_3`); for the s-cells the share of pass-t* argmax equal to each alternative: `add_s` (v_2 + v_j), `reversed` (v_j - v_2), `copy_prev` (v_2), `copy_j` (v_j).
- JSON: `{"by_cell": {"add_c".."mul_s","sub_s": {"by_d": {"4": {"n", "acc", "ws_t", "alt_add_s", "alt_reversed", "alt_copy_prev", "alt_copy_j", "by_j": {"0": {...}, "1": {...}}}, "3": {...}}}}, "set_shas": {...}, "items": {...}}`.
- Labels (d 4, pooled over j; `ref = min(ws_t(add_s), ws_t(sub_v))`): FACTORISED if `ws_t(sub_s) >= 0.8 ref` and `>= 0.5`; LOOKUP if `ws_t(sub_s) <= 0.25` and some alternative share `>= 0.4`; else PARTIAL. `mul_s` is reported as `stale_mul_s` with no routing.
- Routing: FACTORISED -> CG3 NONCE-OP and CG2 COGS split are live; the T-lane can rely on operator x source composition for NUM-100 / BRANCH-DAG. LOOKUP -> CG4 CONSTANT-PRIMITIVE HOLDOUT is demoted (it would fail for the same reason), the T-lane needs full operator coverage in data (a data rule, not a mechanism fix), and the alternative with the largest share names the lookup (`add_s` = source-matched template; `reversed` = operator-matched but argument-swapped). PARTIAL -> CG2 at $0.4 decides.

### 2.3 tta (TS3 TTA-VOTE): `eval_tta.json`

- Items: REG TEST2 d in {4, 6, 7, 8}, the first 500 per d.
- Variants per item, v_0 = the item, v_1..v_7:
  - family `dis` (the TEST2 distractor law re-drawn): `rerender_dis(rng_k, rerender_dis(None, it, drop=True), n_extra = n_k)`, `n_k ~ U{0,1,2}` and `rng_k = default_rng((26952, 0, d, i, k))`; same program, order and names bar the plain distractor givens;
  - family `ins` (the training inserts): `insert_render.distract_mix(it, 26952, k, i, n = 1 if k odd else 2)`; its internal key `(26952, 25205, k, i)` is disjoint from every training draw (those use the run seed 1 / 2).
  - family `perm` (statement order re-shuffled, `NW.topo_shuffle`) is NOT run on bun (order-bound, ~0.1); it is the arm to add on `of_s1 / of_s2` if §263 passes, same layout.
- Readouts: pred at K' 8 per variant; `vote_N` = plurality over v_0..v_{N-1} (ties: v_0's prediction if tied, else the smallest value), `oracle_N` = any correct among the first N, N in {1, 2, 4, 8}; `var_acc` = mean acc of v_1..v_7.
- JSON: `{"fam": {"dis"|"ins": {"by_d": {d: {"n", "intact", "var_acc", "vote_1", "vote_2", "vote_4", "vote_8", "oracle_1", ..., "oracle_8"}}}}, "items": {fam: {d: {"sig", "pred": [[8 preds]], "ans"}}}}`.
- Labels per family at d 7-8 pooled (n-weighted): VALID iff `var_acc(d 4) >= intact(d 4) - 0.03` (re-rendering is not destructive in range); RECOVERABLE if `oracle_8 - intact >= 0.15` and `vote_8 - intact >= 0.05`; SYSTEMATIC if `oracle_8 - intact <= 0.05`; DIVERSE-UNVOTABLE if the oracle gain `>= 0.15` but the vote gain `< 0.05`; else MIXED. Headline = the family with the larger oracle gain (both reported).
- Routing: RECOVERABLE -> TS4 LRM-SELECT and TH3 selection-margin work move up; C2 at d 7-8 is partly render noise, so §264's K'-curve reading carries a vote arm. SYSTEMATIC -> the errors are structural; §269 triage's label routes (D1b / memory / I3). DIVERSE-UNVOTABLE -> a selector (TS4), not voting.

### 2.4 sclat (TS1 SC-LATENT): `eval_sclat.json`

- Items as tta. Rollouts r = 0..7 per item and sigma in {0.03, 0.1, 0.3}: r = 0 is sigma 0 (identity, preds must equal intact); r >= 1 adds relative per-head noise after every pass t = 1..8, `eps` from `default_rng((26953, si, d, i, r, t))`. MC-dropout is not run (it changes the eval law: GPT-2's dropout on the trunk and the lower cache, not a state perturbation).
- Readouts: `vote_N`, `oracle_N` over r = 0..N-1 as in tta; `roll_acc` = mean acc of r 1..7.
- JSON: `{"identity_ok": bool, "by_sigma": {"0.03"|"0.1"|"0.3": {"by_d": {d: {"n", "intact", "roll_acc", "vote_N", "oracle_N"}}}}, "items": {...}}`.
- Labels: INVALID if identity fails; NOISE-DESTRUCTIVE if no sigma has `vote_8(d 4) >= intact(d 4) - 0.03`; else the largest valid sigma gets tta's labels at d 7-8.
- Routing: RECOVERABLE -> TS2 NOISE-TRAIN-SC ($1.2) is the cheapest C2 lever at d 7-8; NOISE-DESTRUCTIVE -> TS1 / TS2 dropped and the noise read (§2.1) is expected FRAGILE (cross-check); SYSTEMATIC with tta SYSTEMATIC -> the late-pass loss is deterministic selection or memory, and the triage label routes.

### 2.5 noop (RB1 NOOP-CLAUSE): `eval_noop.json`

- Items: REG TEST2 d 1-4, 1,000 each; d 6, 500 (descriptive). One perturbed step t* ~ U{1..d} per item from `default_rng((26954, arm, d, i))`; B = `spine[t*]` (the step's target entity; for t* = d, the asked entity); `a, w ~ U{1..10}` excluding `{c_t*, v_t*, v_{t*-1}}`; `u2 ~ UNITS minus {u}`.
- Arms (one per JSON key; the same items in each):
  - `lead` (in-sentence irrelevant number, adjacent to the target name): `"{B} is {a} years old, and " + s`, where s is S_t*'s rendered sentence with its first character lower-cased unless its first word is a name (frames 2 / 5 / 6 / 7 of some ops start with "Compared", "If", "Multiply", "Add", "Take", "Together", "The", "For");
  - `trail` (same-name other-unit, in-sentence): `s[:-1] + ", and {B} also has {w} {u2}."`;
  - `sent` (same-name other-unit, a separate sentence, GSM-NoOp style): `"{B} has {w} {u2}."` inserted directly before S_t*'s sentence.
  - `clean`: the same items unperturbed, run in-part (same batches).
- Solver check: the inserted span is removed by exact string replacement, the result is asserted equal to the clean text, and `OW.solve(clean) == answer`. (The GIVEN regex is unit-bound, so `sent` / `trail` texts are not solvable as-is; the check is on the stripped text by construction.)
- Readouts: acc at K' 8; WS at pass t* (`ws_t`) and on the clean items (`ws_t_clean`); `capture` = share of pass-t* argmax equal to the inserted number (a or w).
- JSON: `{"clean": {"by_d": {d: {"n", "acc"}}}, "arms": {"lead"|"trail"|"sent": {"by_d": {d: {"n", "acc", "ws_t", "ws_t_clean", "capture", "by_t": {t*: acc}}}}}, "items": {...}}`.
- Labels (pooled d 1-4, n-weighted; `worst` = the lowest arm acc): CLAUSE-ROBUST if `worst >= clean - 0.05`; CLAUSE-READ if `worst <= clean - 0.20`; else PARTIAL; suffix `+NUMBER-CAPTURE` if any arm's `capture >= 0.25`.
- Routing: CLAUSE-READ -> register the RB1 FT arm ($0.6: `distract_mix` type `noop` at p 0.5, a p = 0 equivalence gate) before Lane D's public-benchmark ports, and add a noop row to RB6's robust table; NUMBER-CAPTURE says the ':' write takes the nearest number to the cued name, which also bears on CG5 / open-vocab; PARTIAL -> a noop row folded into §265's dense-distract design; ROBUST -> no action (the GSM-NoOp risk is lower than the literature predicts for this pipeline).

## 3. Code touch points (read)

- `boxes/inject_aw_eval.py`: `PARTS` / `AW_ONLY` tuples (36-37, add the five names); `seq_std` (44-47); `passes` (55-81: `edit(t, S, sl)` replaces S after pass t, `on_state(t, S, sl)` sees every S_t, `kset` answers at K'); `tokd` (95-96); `part_acc` (104-114, the intact preds the identity arms must equal); `part_cue_wrong` (174-194, a `seqs_of_t` text-swap example); `grab_states` / `dec_with` (235-251); `rerender_dis` (577-597, used by tta `dis`); `part_insert_ho_s256` (1077-1107, set-sha assertion pattern); `part_stab_fresh` (1256-1268, a fresh exclusive stream); `load` / `run` (1291-1349; the dispatch dict 1320-1340 gets five `__import__("boxes.dgb_parts", ...)` lines as §264's `sc_parts` did).
- `boxes/sc_parts.py` (1-90): the sibling-module pattern (module-level seeds, `seqs_of_t` with asserted cue token, `distract_mix` with forced n, per-item records).
- `boxes/insert_render.py`: `statements` (18-28), `_names` / `_frame` / `_seq` / `_put` (84-98), `distract_mix` (194-230; keys `(seed, 25205, step, k)`), `set_shas` (165-167).
- `boxes/ow_words.py`: `_draw_program` (45-85), `_decorate` (99-105), `make` / `Stream` (108-160), `solve` (178-184; swaps `NW.apply` for mod 11), `set_sha` (187-188).
- `boxes/nested_words.py`: `NAMES` / `UNITS` / `OPS` / `FRAMES` (33-60), `GIVEN` / `QUESTION` (82-83), `_statements` (128-143), `_finish` (146-167), `topo_shuffle` (247-255), `solve` (258-285).
- `boxes/inject_aw.py`: `InjectAW.write` / `colon_logits` (58-63), `tok_aw` (76-93; `think_cues` = text + cue + " The answer is:"), `ws_readout` (135-151), `state_n` (154-156).
- `boxes/inject_st.py`: `cue_str` (44-46), `tok_st` (49-63). `boxes/inject_ow.py`: `TEST_DS` / `TEST_N` / `OWKW` (51-52), `tok_ow` (63-71), `build_sets` (127-169). `boxes/deltamem.py`: `H_ADD, H_DEL, DH` (18), `zero_state` (86-87), `read_fixed` (158-176), `write_seq` (179-223), `pack_state` (134-139).
- `theory/score_dg.py` (1-147): the scorer pattern copied in §6. `boxes/runs_fsq/bun_s1/args.json`: the run's law and flags.
- New files: `boxes/dgb_parts.py` (five `part_*` functions + item builders + a `--smoke` cap), `theory/score_dgb.py` (§6), `scratch/s269b_shas.json`.

## 4. Gates

1. **Set build (G-D):** every item set built twice locally, byte-identical shas; `scratch/s269b_shas.json` committed before launch; asserted on the host. Solver asserts on every rewritten text (stripped text for noop).
2. **p = 0 / identity equivalence:** the sigma-0 arm (noise, sclat) and the clean arm (noop) run through the same `edit` / text path and their K' 8 preds must equal `eval_acc.json`'s intact preds on the same items (`identity_ok`; an INVALID label otherwise). On the local CPU smoke the sigma-0 state is checked bitwise (`torch.equal`) over 20 items. (GPU bitwise checks need the deterministic wrap; preds equality is the host gate.)
3. **No change to existing parts:** the three registration edits in `inject_aw_eval.py` are gated by running `acc` on the local smoke model before and after: `eval_acc.json` byte-identical except `module_sha256`.
4. **Smoke at full item scale:** each builder runs at its full n locally (generator only, CPU); each part runs on a 20-item slice per cell (`--smoke 20`) through a CPU-loaded model, and the scorer is run on the smoke JSONs (keys and fixtures only; no label is read from a smoke). Smoke dirs go under `scratch/_tmp` and are deleted after (no model.pt left in the scratchpad).
5. **gradflow / mem_check:** not applicable (no training, every part `@torch.no_grad`). Memory: `eval_bs 100` at K' 8 is the `acc` part's footprint; noise adds one (12, 64, 64) fp32 tensor per item (0.2 MB), sclat the same per pass.
6. **Mechanism at the claimed depth:** subs labels on `ws_t` (the pass-3 write), not acc, so a decoder last-hop shortcut cannot pass it; noop reports `ws_t` and `capture` beside acc; tta / sclat gains are read at d 7-8 with the d 4 non-destruction control.
7. **Shortcut-exclusion arm:** subs's value-matched c / v cells are the exposed controls (same values, trained sentence forms); noise's family arms expose whether one head family carries the code; the sclat r = 0 rollout and tta's v_0 pin the intact prediction inside every vote.

## 5. Pre-registration table, OC block, censoring

All labels are descriptive routing reads (no C-claim), on `bun_s1`; `bun_s2` confirms, a disagreement is reported as MIXED with no routing change (§269's rule). One lineage; plan given.

| Part | Field | Bar | Label | n | NEAR rule |
|---|---|---|---|---|---|
| noise | median sigma_50 hold / compute, d 3-4 | <= 0.5 / >= 2 | NARROWING / WIDENING, else FLAT | 500 per d | medians are point reads; the per-pass sigma_50 spread (IQR) is reported; CENSORED counts shown |
| noise | compute median | <= 0.05 | FRAGILE | | grid floor 0.02 |
| noise | both medians censored | | INSENSITIVE | | |
| subs | ws_t(sub_s) vs 0.8 min(ws_t(add_s), ws_t(sub_v)) and 0.5 | >= both | FACTORISED | 1,000 (d 4) | Wilson CI straddles 0.5 / 0.25 / 0.8 ref |
| subs | ws_t(sub_s) <= 0.25 and max alt share >= 0.4 | | LOOKUP | | |
| tta | oracle_8 - intact (d 7-8) ; vote_8 - intact | >= 0.15 ; >= 0.05 | RECOVERABLE | 1,000 | CI of oracle gain straddles 0.15 or 0.05 |
| tta | oracle gain <= 0.05 | | SYSTEMATIC | | |
| tta | var_acc(d 4) >= intact - 0.03 | | VALID prefix | 500 | |
| sclat | largest sigma with vote_8(d 4) >= intact - 0.03 | none | NOISE-DESTRUCTIVE | 500 | |
| sclat | as tta at that sigma | | RECOVERABLE / SYSTEMATIC / DIVERSE-UNVOTABLE | 1,000 | |
| noop | worst arm acc vs clean, d 1-4 | >= -0.05 / <= -0.20 | CLAUSE-ROBUST / CLAUSE-READ, else PARTIAL | 4,000 | CI straddles clean - 0.05 or - 0.20 |
| noop | capture | >= 0.25 | +NUMBER-CAPTURE | | chance 0.09 |

**OC block (outcome priors, Theory 11-03):**
- noise: NARROWING 0.45, FLAT 0.35, WIDENING 0.10, FRAGILE 0.05, INSENSITIVE 0.05. (Wave-3 prediction: "late passes narrower basin".)
- subs: LOOKUP 0.50, PARTIAL 0.30, FACTORISED 0.20 (the phrasing result "unseen phrasings fail all-or-nothing" predicts a template lookup).
- tta: SYSTEMATIC 0.50, RECOVERABLE 0.30, DIVERSE-UNVOTABLE 0.15, INVALID 0.05.
- sclat: SYSTEMATIC 0.40, NOISE-DESTRUCTIVE 0.25, RECOVERABLE 0.20, DIVERSE-UNVOTABLE 0.15.
- noop: CLAUSE-READ 0.55 (wave-3 prediction), PARTIAL 0.30, ROBUST 0.15; NUMBER-CAPTURE given CLAUSE-READ 0.6.

**Censoring rules:**
- noise: sigma_50 beyond the grid (never crossing at 2.0) is censored and enters the median as 2 x grid max (a lower bound); if more than half of one side is censored the label is still computed from the bounded medians, and the censored counts are printed beside it; a majority-censored compute side with an uncensored hold side cannot read NARROWING (the bound makes it WIDENING or FLAT, which is the conservative direction).
- subs: base items rejected by the filter are counted (`n_rejected`); if a j stratum ends under 400 kept items the part keeps drawing (cap 200 x n draws) and reports the shortfall; under 400 the label gets an UNDERPOWERED prefix.
- tta: a variant whose builder fails (name budget: `_budget` caps inserts) falls back to the previous variant and is counted (`n_fallback`); items with any fallback are excluded from the vote and reported.
- sclat: non-finite states after noise (none expected at sigma <= 0.3) set `identity_ok`-style flag `finite_ok`; any non-finite item is excluded and counted.
- noop: an item whose first word is unknown to the lower-casing rule (not a name, not in the frame-initial word list) is skipped for `lead` and counted; d 6 is descriptive and never enters the label.
- Seeds: `bun_s2` disagreeing on a label -> MIXED; a NEAR flag on `bun_s1` plus agreement on `bun_s2` keeps the label with the flag.

## 6. Scorer: `theory/score_dgb.py` (stdlib only; fixtures 23/23 run from this listing on 2026-11-03)

```python
"""THEORY §269b DIAGNOSTIC BUNDLE B scorer (DGB_v1; Theory is the only scorer; committed before output; descriptive labels that route lanes C / D / E).
Five eval parts on bun_s1 (+ bun_s2 confirmation), each its own queue line / JSON (layouts: design_wave4 doc §3):
 eval_noise.json RB2  {"identity_ok": bool, "by_d": {d: {"n": n, "intact": acc, "grid": [sigma...], "acc": {t: [acc per sigma]}}}, "fam_d4": {...descriptive}}
 eval_subs.json  CG1  {"by_cell": {cell: {"by_d": {d: {"n": n, "acc": x, "ws_t": y, "alt_add_s": s, "alt_reversed": s, "alt_copy_prev": s, "alt_copy_j": s}}}}}
 eval_tta.json   TS3  {"fam": {"dis"|"ins": {"by_d": {d: {"n": n, "intact": x, "var_acc": y, "vote_1".."vote_8": x, "oracle_1".."oracle_8": x}}}}}
 eval_sclat.json TS1  {"identity_ok": bool, "by_sigma": {"0.03"|"0.1"|"0.3": {"by_d": {d: {"n", "intact", "roll_acc", "vote_N", "oracle_N"}}}}}
 eval_noop.json  RB1  {"clean": {"by_d": {d: {"n", "acc"}}}, "arms": {"lead"|"trail"|"sent": {"by_d": {d: {"n", "acc", "ws_t", "ws_t_clean", "capture"}}}}}
    python -B theory/score_dgb.py [--fixtures]"""
import argparse, json, math, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq"); OUT = os.path.join(_HR, "theory", "results", "score_dgb.json"); RUNS = ("bun_s1", "bun_s2")
CH = 1 / 11; IN = ("1", "2", "3", "4"); DEEP = ("7", "8")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return (c - h, c + h)


def near(ci, thr):
    return ci[0] < thr < ci[1]                                              # NEAR = the Wilson CI straddles the bar


def pool(cells, key, ds=None):
    """n-weighted mean of cells[d][key] over the str keys ds (None = all) -> (mean, n)."""
    ds = [d for d in cells if ds is None or d in ds]; n = sum(cells[d]["n"] for d in ds)
    return (sum(cells[d][key] * cells[d]["n"] for d in ds) / n if n else None, n)


def med(xs):
    xs = sorted(xs); n = len(xs); return None if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def sigma50(grid, acc, intact, chance=CH):
    """the relative noise at which acc crosses the midpoint between intact and chance (log-linear on the grid); None = censored (never crosses)."""
    half = intact - 0.5 * (intact - chance)
    for i, a in enumerate(acc):
        if a <= half:
            if i == 0:
                return grid[0]
            a0, s0, s1 = acc[i - 1], math.log(grid[i - 1]), math.log(grid[i]); f = (a0 - half) / (a0 - a) if a0 > a else 1.0
            return math.exp(s0 + f * (s1 - s0))
    return None


def noise(e):
    by = {}; comp = []; hold = []; cens = 0; tot = 0; gmax = 0
    for d, r in e["by_d"].items():
        di = int(d); s = {}; gmax = max(gmax, max(r["grid"]))
        for t, acc in r["acc"].items():
            v = sigma50(r["grid"], acc, r["intact"]); s[t] = v; tot += 1; cens += v is None
            if di in (3, 4):
                (comp if int(t) <= di - 1 else hold if int(t) >= di + 2 else []).append(v)
        by[d] = {"sigma50": s, "intact": r["intact"]}
    cap = 2 * gmax; c = med([cap if v is None else v for v in comp]); h = med([cap if v is None else v for v in hold])   # censored = a lower bound 2 x grid max
    if not e.get("identity_ok", False):
        lab = "INVALID"
    elif c is None or h is None:
        lab = "MISSING"
    elif c >= cap and h >= cap:
        lab = "INSENSITIVE"
    elif c <= 0.05:
        lab = "FRAGILE"
    else:
        lab = "NARROWING" if h <= 0.5 * c else "WIDENING" if h >= 2 * c else "FLAT"
    return {"label": lab, "sigma50_comp": c, "sigma50_hold": h, "censored": [cens, tot], "by_d": by}


def subs(e):
    p = {c: {"acc": pool(v["by_d"], "acc")[0], "ws": pool(v["by_d"], "ws_t")[0], "n": pool(v["by_d"], "acc")[1]} for c, v in e["by_cell"].items()}
    alt = {k: pool(e["by_cell"]["sub_s"]["by_d"], "alt_" + k)[0] for k in ("add_s", "reversed", "copy_prev", "copy_j")}
    ref = min(p["add_s"]["ws"], p["sub_v"]["ws"]); w = p["sub_s"]["ws"]; n = p["sub_s"]["n"]; ci = wilson(round(w * n), n)
    lab = "FACTORISED" if (w >= 0.8 * ref and w >= 0.5) else "LOOKUP" if (w <= 0.25 and max(alt.values()) >= 0.4) else "PARTIAL"
    return {"label": lab, "ws_sub_s": w, "ci": ci, "ref": ref, "alt": alt, "near": near(ci, 0.5) or near(ci, 0.25) or near(ci, 0.8 * ref), "cells": p,
            "stale_mul_s": p.get("mul_s")}


def gains(cells, vkey, okey):
    it, n = pool(cells, "intact", DEEP); v, _ = pool(cells, vkey, DEEP); o, _ = pool(cells, okey, DEEP)
    return it, v, o, n


def vlabel(it, v, o, n, valid):
    go, gv = o - it, v - it
    lab = ("RECOVERABLE" if go >= 0.15 and gv >= 0.05 else "SYSTEMATIC" if go <= 0.05 else "DIVERSE-UNVOTABLE" if go >= 0.15 else "MIXED")
    ci = wilson(round(o * n), n); return {"label": lab if valid else "INVALID (" + lab + ")", "intact": it, "vote8": v, "oracle8": o, "gain_oracle": go, "gain_vote": gv,
                                          "ci_oracle": ci, "near": near((ci[0] - it, ci[1] - it), 0.15) or near((ci[0] - it, ci[1] - it), 0.05), "n": n}


def tta(e):
    out = {}
    for fam, v in e["fam"].items():
        c = v["by_d"]; valid = c["4"]["var_acc"] >= c["4"]["intact"] - 0.03; out[fam] = vlabel(*gains(c, "vote_8", "oracle_8"), valid)
    head = max(out, key=lambda f: out[f]["gain_oracle"]); return {"label": out[head]["label"], "headline_fam": head, "fam": out}


def sclat(e):
    if not e.get("identity_ok", False):
        return {"label": "INVALID"}
    ok = [s for s, v in e["by_sigma"].items() if v["by_d"]["4"]["vote_8"] >= v["by_d"]["4"]["intact"] - 0.03]
    if not ok:
        return {"label": "NOISE-DESTRUCTIVE", "sigmas": list(e["by_sigma"])}
    s = max(ok, key=float); r = vlabel(*gains(e["by_sigma"][s]["by_d"], "vote_8", "oracle_8"), True); r["sigma"] = s; return r


def noop(e):
    cl, _ = pool(e["clean"]["by_d"], "acc", IN); arms = {}; worst = 1.0; cap = False
    for arm, v in e["arms"].items():
        a, na = pool(v["by_d"], "acc", IN); ca = pool(v["by_d"], "capture", IN)[0]; ci = wilson(round(a * na), na)
        arms[arm] = {"acc": a, "drop": cl - a, "ci": ci, "capture": ca, "near": near(ci, cl - 0.05) or near(ci, cl - 0.20)}; worst = min(worst, a); cap = cap or ca >= 0.25
    lab = "CLAUSE-ROBUST" if worst >= cl - 0.05 else "CLAUSE-READ" if worst <= cl - 0.20 else "PARTIAL"
    return {"label": lab + ("+NUMBER-CAPTURE" if cap else ""), "clean": cl, "arms": arms}


PARTS = {"noise": ("eval_noise.json", noise), "subs": ("eval_subs.json", subs), "tta": ("eval_tta.json", tta), "sclat": ("eval_sclat.json", sclat), "noop": ("eval_noop.json", noop)}


def score(runs=R):
    o = {"section": "269b", "seeds": {}}
    for run in RUNS:
        o["seeds"][run] = {}
        for part, (fn, f) in PARTS.items():
            e = jl(os.path.join(runs, run, fn)); o["seeds"][run][part] = "pending" if e is None else f(e)
    s1 = o["seeds"]["bun_s1"]
    o["verdict"] = "; ".join("%s %s" % (p, s1[p] if isinstance(s1[p], str) else s1[p].get("label")) for p in PARTS); return o


def fixtures():
    res = []

    def chk(name, got, want):
        res.append((name, got == want, got))
    G = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    def curve(s50):                                                         # acc 1.0 below the radius, chance from it on; None = never falls
        return [1.0 if (s50 is None or g < s50) else CH for g in G]

    def nz(comp, hold, ident=True):
        by = {}
        for d in (3, 4):
            by[str(d)] = {"n": 500, "intact": 1.0, "grid": G, "acc": {str(t): curve(comp if t <= d - 1 else hold if t >= d + 2 else comp) for t in range(1, 9)}}
        return {"identity_ok": ident, "by_d": by}
    chk("noise narrowing", noise(nz(0.5, 0.1))["label"], "NARROWING")
    chk("noise flat", noise(nz(0.3, 0.3))["label"], "FLAT")
    chk("noise widening", noise(nz(0.1, 0.5))["label"], "WIDENING")
    chk("noise hold censored = widening", noise(nz(0.3, None))["label"], "WIDENING")
    chk("noise insensitive", noise(nz(None, None))["label"], "INSENSITIVE")
    chk("noise fragile", noise(nz(0.02, 0.02))["label"], "FRAGILE")
    chk("noise invalid", noise(nz(0.5, 0.1, False))["label"], "INVALID")
    chk("sigma50 interp", round(sigma50(G, [1.0, 1.0, 0.9, 0.3, CH, CH, CH, CH, CH, CH], 1.0), 3), round(math.exp(math.log(0.1) + (0.9 - (1 + CH) / 2) / 0.6 * math.log(2)), 3))

    def cell(acc, ws, alt=(0.0, 0.0, 0.0, 0.0)):
        return {"by_d": {str(d): {"n": 1000, "acc": acc, "ws_t": ws, "alt_add_s": alt[0], "alt_reversed": alt[1], "alt_copy_prev": alt[2], "alt_copy_j": alt[3]} for d in (2, 3, 4)}}
    base = {c: cell(0.95, 0.96) for c in ("add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v", "add_s", "mul_s")}
    chk("subs factorised", subs({"by_cell": {**base, "sub_s": cell(0.9, 0.9)}})["label"], "FACTORISED")
    chk("subs lookup", subs({"by_cell": {**base, "sub_s": cell(0.1, 0.1, (0.6, 0.1, 0.1, 0.05))}})["label"], "LOOKUP")
    chk("subs partial", subs({"by_cell": {**base, "sub_s": cell(0.4, 0.4, (0.3, 0.1, 0.1, 0.05))}})["label"], "PARTIAL")
    chk("subs low-but-diffuse = partial", subs({"by_cell": {**base, "sub_s": cell(0.1, 0.1, (0.2, 0.2, 0.2, 0.2))}})["label"], "PARTIAL")

    def tc(it, var, v8, o8):
        return {"by_d": {str(d): {"n": 500, "intact": it if d >= 7 else 0.99, "var_acc": var if d == 4 else it, "vote_1": it, "vote_8": v8 if d >= 7 else 0.99,
                                  "oracle_1": it, "oracle_8": o8 if d >= 7 else 0.99} for d in (4, 6, 7, 8)}}
    chk("tta recoverable", tta({"fam": {"dis": tc(0.6, 0.98, 0.7, 0.85), "ins": tc(0.6, 0.98, 0.6, 0.62)}})["label"], "RECOVERABLE")
    chk("tta systematic", tta({"fam": {"dis": tc(0.6, 0.98, 0.6, 0.63), "ins": tc(0.6, 0.98, 0.6, 0.62)}})["label"], "SYSTEMATIC")
    chk("tta diverse", tta({"fam": {"dis": tc(0.6, 0.98, 0.62, 0.85)}})["label"], "DIVERSE-UNVOTABLE")
    chk("tta invalid", tta({"fam": {"dis": tc(0.6, 0.9, 0.7, 0.85)}})["label"], "INVALID (RECOVERABLE)")
    sc = {"identity_ok": True, "by_sigma": {"0.1": tc(0.6, 0.98, 0.7, 0.85), "0.3": tc(0.6, 0.98, 0.5, 0.9)}}; sc["by_sigma"]["0.3"]["by_d"]["4"]["vote_8"] = 0.9
    r_ = sclat(sc); chk("sclat picks the largest valid sigma", (r_["label"], r_["sigma"]), ("RECOVERABLE", "0.1"))
    sc2 = {"identity_ok": True, "by_sigma": {"0.1": tc(0.6, 0.98, 0.7, 0.85)}}; sc2["by_sigma"]["0.1"]["by_d"]["4"]["vote_8"] = 0.8
    chk("sclat destructive", sclat(sc2)["label"], "NOISE-DESTRUCTIVE")
    chk("sclat invalid", sclat({**sc, "identity_ok": False})["label"], "INVALID")

    def na(acc, capt=0.0):
        return {"by_d": {str(d): {"n": 1000, "acc": acc, "ws_t": acc, "ws_t_clean": 0.99, "capture": capt} for d in (1, 2, 3, 4)}}
    clean = {"by_d": {str(d): {"n": 1000, "acc": 0.98} for d in (1, 2, 3, 4)}}
    chk("noop robust", noop({"clean": clean, "arms": {"lead": na(0.96), "trail": na(0.95), "sent": na(0.97)}})["label"], "CLAUSE-ROBUST")
    chk("noop read+capture", noop({"clean": clean, "arms": {"lead": na(0.5, 0.4), "trail": na(0.95), "sent": na(0.97)}})["label"], "CLAUSE-READ+NUMBER-CAPTURE")
    chk("noop partial", noop({"clean": clean, "arms": {"lead": na(0.88), "trail": na(0.95), "sent": na(0.97)}})["label"], "PARTIAL")
    d = tempfile.mkdtemp()
    for run in RUNS:
        os.makedirs(os.path.join(d, run)); json.dump({"clean": clean, "arms": {"lead": na(0.5, 0.4)}}, open(os.path.join(d, run, "eval_noop.json"), "w"))
    o = score(d); chk("score pending parts", ("noop CLAUSE-READ+NUMBER-CAPTURE" in o["verdict"]) and ("noise pending" in o["verdict"]), True)
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "->", r[2])
    k_ = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k_, len(res))); return k_ == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
```

Fields read by the scorer, per part: noise `identity_ok, by_d[d].{n, intact, grid, acc[t]}`; subs `by_cell[cell].by_d[d].{n, acc, ws_t, alt_*}`; tta `fam[f].by_d[d].{n, intact, var_acc, vote_8, oracle_8}`; sclat `identity_ok, by_sigma[s].by_d[d].{n, intact, vote_8, oracle_8}`; noop `clean.by_d[d].{n, acc}, arms[a].by_d[d].{n, acc, capture}`. The builder greps `dgb_parts.py` for each of these keys before launch (scorer field-path check).

## 7. Cost (RTX 3090, ~$0.2/h, fp32, eval_bs 100)

One run = 8 cued passes + decoder on 500 items, about 3-4 s on a 3090 (the lower six blocks are cached per distinct input). noise: 4 d x (80 grid runs + 2) + 100 family runs = 428 runs, ~25-30 min. subs: 9 cells x 1,500 items + WS readouts, ~5 min. tta: 2 families x 4 d x 8 variants x 500 = ~64 runs + builds, ~6 min. sclat: 3 sigmas x 4 d x 8 rollouts = 96 runs, ~7 min. noop: 4 arms x 4,500 items x (passes + WS) ~ 8 min. Total ~55 min per seed plus model ship and host setup (~10 min): **~$0.25 per seed, $0.5 for s1 + s2; guard $0.8.** Eval-only queue lines ship `log.jsonl` (the drain rule). Host logs land in `logs/vast_<host>_<port>/`.

## 8. RED-TEAM (separate sub-agent; verbatim findings and my responses)

(Pending: the sub-agent slot was unavailable at 17:25 EST; this section is filled below once it returns.)

## 9. Verified arXiv ids (each `https://arxiv.org/abs/<id>` fetched 11-03, title matched)

- 2410.05229 Mirzadeh et al., GSM-Symbolic (GSM-NoOp: one irrelevant clause drops accuracy) -> noop.
- 2302.00093 Shi et al., Large Language Models Can Be Easily Distracted by Irrelevant Context -> noop `sent`.
- 2402.19255 Li et al., GSM-Plus (perturbation robustness benchmark; distractor insertion and rephrasing rows) -> noop / tta framing. (ID corrected 11-04 per Literature e573e123.)
- 2109.05925 Kumar et al., Adversarial Examples for Evaluating Math Word Problem Solvers (question reordering, paraphrase) -> tta `perm` arm rationale.
- 2402.08939 Chen et al., Premise Order Matters in Reasoning with LLMs -> why `perm` waits for §263. (ID corrected 11-04 per Literature e573e123.)
- 2203.11171 Wang et al., Self-Consistency Improves Chain of Thought Reasoning -> vote@N / oracle@N definitions.
- 2510.07745 You et al., Parallel Test-Time Scaling for Latent Reasoning Models (uncertainty-based latent sampling + a latent reward model) -> sclat; TS4 routing.
- 2505.11484 Xu et al., SoftCoT++ (test-time scaling with diversified continuous thoughts) -> sclat.
- 2106.14342 Bai et al., Stabilizing Equilibrium Models by Jacobian Regularization (contraction / basin of a fixed-point update) -> noise's NARROWING reading and the contraction arm of RB2.
- 1805.01500 Dieng et al., Noisin: Unbiased Regularization for RNNs (hidden-state noise) -> TS2 NOISE-TRAIN-SC routing.
- 2305.18654 Dziri et al., Faith and Fate: Limits of Transformers on Compositionality (pattern matching vs composition) -> subs's LOOKUP hypothesis.
- 2301.10884 Lepori et al., Break It Down: Evidence for Structural Compositionality in Neural Networks -> subs's FACTORISED hypothesis.
- Not fetched, so not cited here: 2311.12997, 2108.04378, 1908.08351 (listed under CG1 in the backlog).
