# Scorer bug-hunt, wave 2 — theory/score_*.py for §251–§269

Date: 2026-11-03 17:40 EST (Audit; read-only review, fixtures only, no commits, no scorer runs on real data).
Scope: 23 scorers — ae, lb, lb2, pc, cur_pilot, ds, dch, hx, bu, c5b, c5c, gb, gbf, tc, nk_pilot, of, sc, dd, tx, em, ls, dg, insert.
Method: five parallel reviewers, each checked docstring/THEORY rule vs code (direction, >= vs >, depth/pass sets, AND/OR, label order), degenerate thresholds, field paths vs the eval writers (boxes/inject_aw_eval.py, inject_aw.py, of_parts.py, sc_parts.py, nk.py, scratch/s255_insert.py, scratch/s258_askname.py, scratch/s266_planfree.py, scratch/s268_target.py), fixture coverage of every FAIL branch, silent None swallowing, and re-derived every recorded verdict in theory/results from the stored eval fields. The lead re-verified every HIGH/MED claim in source before listing it.

Fixtures: all 23 pass (ae 8/8, lb 9/9, lb2 15/15, pc 6/6, cur_pilot 6/6, ds 9/9, dch 9/9, hx 8/8, bu 9/9, insert 5/5, c5b 4/4, c5c 4/4, gb 5/5, gbf 8/8, tc 6/6, nk_pilot 6/6, of 10/10, sc 10/10, dd 12/12, tx 14/14, em 11/11, ls 10/10, dg 15/15).

Severity: HIGH = flips or destroys a recorded verdict; MED = can change a future verdict or block one; LOW = descriptive / fixture gap / defensive.

**Bottom line: no bug flips any recorded verdict label.** One HIGH is a record-destruction hazard (score_ls name collision). The dominant MED is a project-wide pattern: a missing training log scores as STABLE in 9 binding scorers.

---

## 1. Cross-cutting

### X-1 [MED, 9 scorers] Missing log.jsonl scores as STABLE
- `theory/score_ao.py:40-41` `if not L: return None, "no loss rows"` — AO.stable returns `(None, …)` when log.jsonl is absent/empty/has no {step, loss} rows; `ts = jl(train_summary.json) or {}` makes a missing train_summary `n_nonfinite 0`.
- Every binding caller treats only `sok is False` as a failure: score_ae.py:48, score_lb.py:70, score_lb2.py:94, score_ds.py:50, score_dch.py:47, score_bu.py:49 (`st = sok is not False`), score_gbf.py:53, score_of.py:46, score_dd.py:86 (`"STABLE": sok is not False`).
- Rule text: every one of these registrations lists STABLE (score_ao) as a binding criterion.
- Why wrong: a run whose log never shipped (memory: cloud eval-only drains blank the local log; a drain failure leaves no log) passes STABLE with no stability check; only the descriptive `"stable": [null, "no loss rows"]` field reveals it.
- Already happened (descriptive only): §251 CTRL re-eval dirs rcb_plaw4_s1_re / s2_re have no log.jsonl; stored score_ae.json records `ctrl.*.stable = [null, "no loss rows"]`, label EXTEND. CTRL is descriptive, no flip.
- Fixture: any scorer's "all-pass" world minus log.jsonl -> still the PASS label (expected pending). None of the 9 scorers has a fixture for it.
- Fix: treat `sok is None` as `pending` (or `FAIL: STABLE`) everywhere; require train_summary.json.

### X-2 [LOW] Fixture gaps are systematic
No scorer in scope has a fixture for STABLE failing, a missing log, or (in most) the `pending` branch; several have no fixture for the label they actually recorded (score_lb PARTIAL, score_ds plain FAIL, score_c5b NO-SIGNAL). Listed per scorer below.

---

## 2. Per-scorer findings

### score_ls.py (§268 LINEAGE-SEED)
- **LS-1 [HIGH, record overwrite]** `score_ls.py:22 OUT = theory/results/score_ls.json`. The §268 scorer was written over the §241 LEN-STEP scorer in place (git 4b25ce59 modified theory/score_ls.py; 927e7e45 / 150a093f were the §241 scorer). `theory/results/score_ls.json` currently holds the §241 record (`"section": 241`, verdict "LEN-STEP STEPS | s1: STEPS / s2: STEPS"). The first real `python theory/score_ls.py` run overwrites it, and the §241 scorer source now exists only in git history. No label flips; a recorded §241 verdict file is destroyed and §241 is no longer re-scorable from the tree. Fix: rename to score_lin.py / score_lin.json (and restore the §241 scorer under its old name).
- **LS-2 [MED, verdict can never land]** `score_ls.py:66,71` reads bun_s3/eval_insert_ho.json for CONTENT. bun_s1 and bun_s2 never ran insert_ho (bun_s1 has acc/cue_only/halt_inc/halthead/halthead2/long_k128/lp5h/stab_fresh/swap_kd/ws_deep only; insert_ho exists in dis_s1 / dch_s1). §268: "Stage evals use the parts each s1 stage ran" -> bun_s3 will have no eval_insert_ho.json and score() returns `pending (bun_s3)` forever (line 68-69 lumps all five files into one pending). Either queue an insert_ho line for bun_s3 or read CONTENT from dch_s3. Also note the 0.85 CONTENT bar is the §262/§264 bar; §259/score_bu has no CONTENT criterion.
- LS-3 [LOW-MED] `score_ls.py:81` dead ternary: both branches yield "LINEAGE-PARTIAL (FAIL: …)", so a bun_s3 RETAIN failure (the C1 claim itself not replicating) gets the same label as a C6 miss. Docstring defines PARTIAL as "RETAIN passes, others fail". Fixture "decoder" (k2 0.5 -> "LINEAGE-PARTIAL (FAIL: RETAIN)") pins the ambiguous behaviour. Suggest BREAKS-AT-bun.
- LS-4 [LOW] NEAR uses the item SE only (2 SE ~0.01-0.02 at d 6/7); Note 269-2 requires every C2 seed reading to carry the seed-spread caveat (sigma_seed 0.090 at d 6). No seed-level NEAR field. Bars unchanged per 269-2(1).
- Stage-gate divergences from the original registered gates are registered design choices (listed for the record): askname re-applies TEST2 d 1 >= 0.98 only (§233 was P1-ASK+NAME per class); aw_plaw2 d 2 >= 0.95 + WS d2 pass 2 >= 0.90 (stricter acc, adds a mechanism gate, drops §234's causal gates); rcb_plaw4 d 1-4 >= 0.95 (stricter than §237, drops chain/causal gates); dis 0.95 is LOOSER than §256's own RETAIN 0.98 (the only leniency); dch 0.98 matches; bun C2 = mean(bun_s1, s2) - 0.05 instead of §259's dch - 0.05 and omits ws_deep passes 5-6. Stop-on-first-failure and pending ordering correct; all field paths verified against the writer and real bun files.

### score_bu.py (§259 BUNDLE) — three MED, recorded "FAIL: C5 / FAIL: C5" unaffected
- **BU-1 [MED]** `score_bu.py:38` only checks `os.path.isdir(ctrl)`; `:46-47` `LH.norm_guard` returns `(None, "no paired norm_guard bins")` (score_lh.py:71-72) and `FD.lp0` returns `(None, "missing windows")` (score_fd.py:40-41) when the bunc log is absent; `c6 = … and ng_ok is True and lp_ok is True` -> a missing/empty BUN-CTRL log yields "FAIL: C6" instead of pending. Rule (BU_v1): "missing parts are pending; FAIL at final". Probe: ctrl dir present, log.jsonl removed -> "BUNDLE 0/2 (FAIL: C6 / FAIL: C6)".
- **BU-2 [MED]** `score_lh.py:70 common = sorted(set(f) & set(c))`: a truncated CTRL log silently narrows the 1.25x norm guard to the common bins. Rule: "GAMED if ||S_8|| or ||S_{d+2}|| exceeds 1.25 x CTRL in any bin". Probe: ctrl log cut at step 4000 (bins 1-4), bun gamed at 1.4x in bins 5-8 -> "BUNDLE 2/2 (BUNDLE-PASS)", max_ratio 1.0. Fix: require len(common) == number of bun bins or report dropped bins. (Recorded: bun and bunc both have 8 bins.)
- **BU-3 [MED/LOW]** `score_bu.py:48 c5 = c5r["label"] == "C5-EXTRAP"`: a VOID halt head (val_acc < 0.80) reads "FAIL: C5", not VOID, while §259 censoring is "VOID-RERUN once". Fixture: head.val_acc 0.7. (Recorded val_acc 0.986.)
- LOW: no STABLE-fail or pending fixture; the want-substring check (`"C1" in verdict`) would accept "FAIL: C1,C2". Provenance note: bun_s1's halt head (val 0.9856) is a fresh fit on frozen BUN, not dch_s1's head (0.9828) — consistent with BU_v1 §21; flag only if Note 259-1's "the dch L1 head" is read as weights.

### score_ds.py (§256 DISTRACT) — one MED, recorded "SELECTOR 0/2, C2 2/2 (C2-ONLY / C2-ONLY)" unaffected
- **DS-1 [MED] asymmetric C2.** `score_ds.py:51` `… "SELECTOR+C2" if (held and d5ok and amb and credit) else "SELECTOR" if held else "C2-ONLY" if amb else …` and `:62 c2 = sum(l_ in ("SELECTOR+C2", "C2-ONLY") …)`. Rule (DS_v1 §3 / §256): C2 = d 5 >= 0.80; AMBITION d 6 >= 0.65 and >= rcb + 0.10; SELECTION-CREDIT WS d 6 passes 1-4 >= 0.90. With HELD-OUT passing, C2 needs d5 + AMBITION + credit; with HELD-OUT failing, C2-ONLY needs AMBITION alone (d 5 and WS credit unchecked). Probe (scorer's own world()): {ws:0.5} -> "SELECTOR 2/2, C2 0/2"; {ws:0.5, fam:0.5} -> "SELECTOR 0/2, C2 2/2 (C2-ONLY)". A seed is credited C2 without the credit gate iff it also failed HELD-OUT. Recorded dis_s1/s2: d5 0.993/0.986, ws6 p1-4 >= 0.993 -> no flip. Any scorer that copied this block should be checked.
- LOW: fixtures never produce the plain "FAIL" label; indist "every cond" includes 'clean' (harmless, 1.0).

### score_gbf.py (§260 full) — X-1 only (`:52-53`); recorded "PLAN-FREE-C1 1/2 (PLAN-FREE-C1 / FAIL: CONTENT,C2)" re-derived: gb_s2 CONTENT min by_pos '2' = 0.7465 < 0.80. C6 tracks Notes 260-1/260-3 (K'64 and K'128 at d 1-6). gates_pass.json plan_free writer is scratch/s266_planfree.py:31. LOW: no fixture for STABLE fail / missing log / pending.

### score_of.py (§263) — X-1 (`:45-46`). No inputs or result yet. LOW: dumps entire eval_name_swap / eval_shuffle_graded items into score_of.json; swap_kd with no kept pair -> KeyError (loud); fixture gaps RETAIN via d 1-4 / LP5-D4, C6 via K'64, STABLE, pending. All thresholds and field paths (of_parts.part_late_premise, part_test_perm, part_long) verified.

### score_dd.py (§265) — X-1 (`:85-86`). No inputs yet. LOW: `:108-109` D1a AUC over ALL d 8 items while §265 says "odd items" (drops do use odd; descriptive); wilson n = 1000 hard-coded (eval_acc has no "n"); fixture gaps RETAIN via d6 / SWAP4 / LP5-D4, STABLE, D1a MIXED, pending, single-seed PARK. Pilot tracks Note 265-2 (unconditional ok7, shortcut guard >= 0.80, mean-over-seeds gains); ok7/rel7 on stored bun files reproduce Note 264-1's 0.59 / 0.80.

### score_sc.py (§264 SCORECARD-CHECK) — the known tau collapse, re-derived independently
- **SC-1 [MED as rule hole, LOW in practice]** `score_sc.py:65 tau = math.sqrt(c * max(h, c / 100)) if c > 0 else 0.0`. Note 264-3(a) floored h -> 0, but the `c > 0` guard re-introduces the collapse when c == 0: tau = 0 and `contrast = c >= 3*h` is True (0 >= 0), so the depth is NOT routed to NO-CONTRAST; `median(…) < 0` is False everywhere -> hold_er False (ER "COMP"), hold_lf False (LF "CUE"), hold_lfi False; nothing flags it (h_floor is reported only). Fixture: STD items rel_inc all 0.0 (c = h = 0), ER items rel_inc 0.0 with pred constant from K' 1 (a total hold) -> expected PLAN / NO-CONTRAST, scorer gives ER COMP; at d 4-6 in both seeds -> "C5 unchanged". c == 0 needs a bitwise-unchanged state over passes 2..d (unreachable for bun in fp32). Fix: c < 1e-6 -> NO-CONTRAST / VOID at that d.
- LOW: EARLY = P(khalt <= d-1) is near-automatic since the STD halt mode is d-1 (Note 264-4); matches the registered text, so a loose threshold, not a mismatch. Fixture gaps: ER MIXED, C6 MIXED, seed disagreement, pending. Layout vs sc_parts.py confirmed (rel_inc [N][23] for t = 2..24, denominator ||S_{t-1}|| at sc_parts.py:75, Frobenius over the full state).

### score_tx.py (§266 SAME-DATA BASELINES) — writer unbuilt
- **TX-1 [MED] registered censoring not implemented.** §266 OC: "a text cell with parsed < 0.9 at d <= 4 is a build bug (VOID-RERUN once)". `score_tx.py:55` only reports parsed_min; `:54, :57-59` label such a cell UNDERTRAINED -> "VOID-EXTEND once (+12k)" (Note 266-1's remedy for a weak cell), i.e. the wrong censoring action, and a cell with parsed 0.5 but acc-over-parsed 1.0 counts as VALID. Fixture: by_d d 1-4 {acc 1.0, parsed 0.5, n 1000} -> VALID / LATENT-SPECIFIC; should be VOID-RERUN. Writer contract risk: the scorer assumes "acc" is over all n items; assert acc <= parsed.
- Writer-name collision: boxes/text_eval.py ALREADY writes `<run>/eval_text.json` with the §219 E1 layout ("ns"/"items", no by_d). If the §266 part is built on it, score_tx KeyErrors (loud). Pick a distinct file name.
- TX-2 [LOW] c7() pools whatever cells exist; no check for exactly the four registered cells. TX-3 [LOW] an UNDERTRAINED cell's +12k extension lands in a new run dir but TEXT is fixed to tx_s1/tx_s2/txv_s1 (`:22`).
- Clean: validity gate tracks Note 266-1; comparator = mean bun eval_acc d 7/8 at k 8 (0.8545 / 0.530 match §266's 0.86 / 0.53); fixtures cover every binding label.

### score_em.py (§267 EXACT-EM-11) — writer unbuilt
- **EM-1 [MED, crash blocks the binding verdict]** `score_em.py:88` cell() indexes `e["swap_v"]["3"]` and `e["mem_ablate"]["3"]["acc"]` for CTRL cells too (`:107` runs cell() on EM + CTRL). emt_s1 (TOKEN-INJECT-11, no DeltaMem) has no swap_v / mem_ablate; a CTRL eval_em.json without them raises KeyError and blocks em_s1/s2. CTRL is correctly non-binding otherwise (`:110`). Fix: .get with None for CTRL cells.
- EM-2 [LOW-MED, descriptive] `:66-68` best_bijection `C[a][b] += 1` with a = pred; a pred of -1 (sentinel) indexes C[10] silently, inflating bijection acc. Fixture: pred = [-1]*500, v = [10]*500 -> 1.0 -> PERMUTED-LATENT if raw ws < 0.5. Assert 0 <= pred < 11.
- EM-3 [LOW] onset() needs rows at step exactly 1000 / 2000 with ws_readout['3']['1']; a logger binning at 999/1999 -> permanent pending. EM-4 [LOW] NEAR/CI only on acc d3 and swap follow, not on the ws 0.90 / ONSET 0.80 bars; no fixture for FORM via acc d2 / ws3['1'] or PERMUTED-LATENT through score().
- Clean: ONSET / FORM / USE / PARK thresholds match §267 + Note 267-1; DP is a correct exact assignment max.

### score_dg.py (§269 DIAGNOSTIC BUNDLE) — writer unbuilt
- **DG-1 [MED, writer contract]** `score_dg.py:53 valid = e["ctrl"]["text_keep"] >= 0.95 and e["ctrl"]["state_keep"] >= 0.95` treats keeps as rates and never uses n_ok_ctrl, while the docstring lists `"n_ok_ctrl": M` beside them (counts). If the writer emits counts, valid is always True and INVALID can never fire; conversely `fail` entries are divided by n_fail (`:52`) — if rates are written all shares shrink. No assertion that text_fix + state_fix + both + neither == n_fail or keeps in [0,1]. Fixture: {"n_fail": 100, "n_ok_ctrl": 500, "fail": {…}, "ctrl": {"text_keep": 400, "state_keep": 490}} -> VALID (should be INVALID at 0.80). Pin the contract (rates) in the docstring and assert.
- **DG-2 [MED, registered reading rule not implemented]** §269: "labels on bun_s1 route the lanes. bun_s2 confirms, and a disagreement is reported as MIXED with no routing change." `score_dg.py:104-105` builds the verdict from bun_s1 only; bun_s2 labels are stored but never compared. Fixture: s1 triage SELECTION, s2 MEMORY -> verdict "triage SELECTION" (expected MIXED).
- DG-3 [LOW] openvocab() tests OPEN before CLOSED (`:71-72`); harmless for bun (base d 1-4 = 1.0). DG-4 [LOW] readuse() `drop.get("STEP", 0)` defaults silently; alphahold() zip() truncates; no fixtures for triage / readuse / alphahold MIXED.
- Clean: triage shares use n_fail; JOINT -> SELECTION -> MEMORY -> COMPUTATION -> MIXED is equivalent to the conjunctive definitions; missing part -> per-part pending.

### score_lb2.py (§258) — recorded BRIDGE-OK / S1 / NO-SEED re-derived
- **LB2-1 [MED, future]** `score_lb2.py:74-87`: if p2m_*_ws6k/eval_ws.json is missing, ws6 is None, `rising` is False, and a run with ws12 < 0.50 is labelled NO-SEED (STOP) rather than pending, although the Note 258-4 extension rule depends on the 6k read. The fixture "S2 write missing" uses ws2 = 0.02 and passes for the wrong reason; world(ws2 = 0.3, no ws6 dir) exposes it. Both recorded S2 reads had the 6k file.
- LOW: S1 winner wording (LB2_v1 "TEST2 d 1 and P1-ASK+NAME" vs code mean class acc, tie -> 3e-5); both arms 1.000. X-1 at `:94` (S3; dism_s1 never ran). eval_st_classes.json writer is scratch/s258_askname.py:97-104.

### score_ae.py (§251/§255) — recorded labels re-derived (aex_s2 SWAP4 0.461 -> EXTEND-D3-ONLY; s3/s4 -> NOT-EXTEND)
- X-1 at `:47-48` (already hit the descriptive CTRL re-eval dirs, see X-1).
- LOW `:43` swap_mid_t2 is only written under the REG law (inject_aw_eval.py:418); a non-REG eval makes EXTEND-D3-ONLY impossible. All §251/§255 runs are REG.
- Fixture gaps: UNSTABLE, LP5-D4-only failure, ACC4 failing only on the CD+0.25 margin, missing log.

### score_hx.py (Note 257-2 / 259-1; shared halt()/hold() engine for bu, c5b, c5c)
- LOW `:32 (e.get("head") or {}).get("val_acc", 1.0) < 0.80`: a missing head block silently passes the VOID gate (probe: delete "head" -> C5-EXTRAP). The writer always emits head (inject_aw_eval.py:899); `:41 r.get("never_halt_n", 0)` same pattern. No NO-SIGNAL fixture.
- LOW `:44` event_only_mean hard-codes the §248 cap 8 for never-halt K'; halt_inc / halthead2 cap at 14, so score_c5b's recorded event_only_mean values are slightly off (bun_s1 d 2: 1.0641 vs true 1.0581). Descriptive only.
- Rules verified: IN-RANGE = §247 C5 gates exactly; EXTRAP d 5-7 acc + mean_khalt <= d + strictly increasing over 1..7; censoring per Audit 257-2(1) (never-halt in hist['8']). Recorded C5-IN-RANGE / DRIFT correct.

### score_c5b.py (Note 259-5) — CLEAN on binding logic
- tau selection (inject_aw_eval.py:1190-1193) is the largest feasible value over grid [0] + logspace(-6, 0.5, 400); tau 0 is always feasible (never halt -> NO-SIGNAL, not a pass). Recorded H-INC NO-SIGNAL / IN-RANGE and H-HEAD2 IN-RANGE re-derived.
- LOW (design, Note 259-6 already states it): H-INC d 1 vs d 2 is a structural tie (khalt 1 at both), so strict increase fails by construction; bun_s2 got IN-RANGE only via one item at d 2 halting at K' 2. No fixture for NO-SIGNAL / VOID at the c5b level.

### score_c5c.py (Note 259-7) — CLEAN. Recorded r1/r2/r3 labels re-derived; float boundary at the 0.02 margin checked for all acc_k8 in [0.80, 1.00]. LOW: d 8 (descriptive per Note 259-11) is not reported by the scorer.

### score_gb.py (§260 pilot) — CLEAN. GO/PARK mutually exclusive; recorded GO re-derived.

### score_tc.py (§261 pilot) — CLEAN on the rule (pending; pilot_tc_s1 absent). LOW `:26-31` an absent gates_pass.json (scratch/s268_target.py:37 writes it only if the assert holds) -> target_only None -> silently NOTE with no "assert missing" flag; no fixture for the absent file.

### score_dch.py (§257) — CLEAN. Tracks Audit 257-1 (TRAINED-TYPE descriptive). LOW `:35 rf = {}` vestigial. X-1 at `:47`.

### score_lb.py (§253) — CLEAN. Recorded PARTIAL re-derived. LOW: no fixture produces PARTIAL (the recorded label) or UNSTABLE. X-1 at `:70`.

### score_pc.py (§254) — CLEAN (KeyError on a missing field crashes, not silent). score_cur_pilot.py (Note 254-1/2) — CLEAN (all three labels + boundary + pending covered). score_nk_pilot.py (§262) — CLEAN (recorded PARK recomputed 0.934 / 0.929 / 0.957; a None cell -> TypeError crash). score_insert.py (Note 255-3) — CLEAN (recorded labels re-derived; cosmetic: a model key named "verdict" would collide at `:50`).

---

## 3. Writers not in boxes/*.py (for the field-path check)
- gates_pass.json: plan_free <- scratch/s266_planfree.py:31; target_only <- scratch/s268_target.py:37 (file only written when the assert holds).
- eval_st_classes.json <- scratch/s258_askname.py:97-104. insert.json <- scratch/s255_insert.py:110-111.
- Unbuilt (pre-registered scorers): eval_text.json (§266 by_d layout — name already taken by boxes/text_eval.py's §219 layout), eval_pipeline.json, eval_text_sub.json, eval_em.json, eval_triage / eval_keys / eval_openvocab / eval_readuse / eval_alphahold. Their key paths are unverifiable until built; DG-1 and TX-1 are the contract risks.
- Every other key read by the 23 scorers has a confirmed writer in boxes/inject_aw_eval.py ("eval_%s.json" at :1342), inject_aw.py (log bins), of_parts.py, sc_parts.py or nk.py, and was checked against the real files where they exist.

## 4. Suggested fix order
1. LS-1: rename score_ls.py / score_ls.json for §268 before any real run (destroys the §241 record otherwise).
2. X-1: `sok is None -> pending` in score_ao callers (one-line change in 9 files) + a missing-log fixture each.
3. BU-1/BU-2: pending on a missing/short CTRL log; require full bin overlap in score_lh.norm_guard.
4. DS-1: define C2-ONLY with the same d 5 + credit gates (or register the asymmetry explicitly).
5. LS-2, EM-1, DG-1, DG-2, TX-1, LB2-1, SC-1 before their inputs land.
