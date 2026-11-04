# 07 PLAN-FREE LANE (P9 / H8 / TH5, stage 1 PLAN-PROBE-F): independent red team, 2026-11-04

Target: `docs/design_wave4_2026-11-03/07_plan_lane_cuefade_answerable_dataflow.md` (the "doc"). Read-only review: no runs, no computation on stored run outputs. Every code citation below was checked against the cited lines. Disclosure: besides `boxes/runs_fsq/of_s1/args.json`, I also read `boxes/runs_fsq/gb_s1/args.json`, a config file and not an output, to check the TH5 assert in finding 6. No other run files were opened.

## Verdict: GO-WITH-FIXES for the lane. NO-GO on S1-S3 as written.

- The direction is right. In §263 the plan still sits in every pass's cue. Note 260-5 shows that without it the loop follows forward reachability. So plan-free selection is the largest remaining C1 caveat, and it is also the main C4 caveat at test time. This should be the next C1 lane after §263 / §270.
- The doc fixes red team 06's harness-stop problem properly:
  - inputs come from the text only;
  - stops are model-side (hold, C5 r = 3, H8);
  - twins are added.
  M1 is a correct pigeonhole.
- **The blocking problem is finding 1.** The training law the lane inherits from of_s1 (`distract_mix s256,chain,givop`, question always on the spine end) contains no item where a question-blind policy is wrong. So none of the arms gets the "training pressure" that the decision rule (§1) relies on. As written:
  - P9 is predicted to reproduce gb's null on every twin row;
  - PLAN-PROBE-F mostly measures which shortcut SGD picks;
  - TH5's NK path is predicted to go unused, as §262's did.
  The fix is cheap: retarget the question in training. It must come before any spend.

## Findings

### 1. HIGH: the of training law makes a question-blind policy optimal, so no arm gets question-conditioning "by training pressure".

**Evidence (code)**
- `boxes/insert_render.py:194-230` (`distract_mix`). Every training insert creates a **sink**: X is never read.
  - `chain` puts X = S_j + c **directly after S_j's sentence**, with j ~ U{1..d-1} (l. 210-212). So S_d is never read in training.
  - `s256` / `givop` put X = Y + c on a new given.
- The question is always QUESTION(u, spine[-1]) (`insert_render.py:28`, `nested_words.py:147`). Inserts are never asked.
- The ask items are one-hop only (Note 262-2: X = S_1 / S_0 / distractor given / S_2).
- The aux names the plan at every pass: R_step = " <S_t> has <x> <unit>." (`_inject_aw_eval_s251.py:744` docstring; `inject_ow.py:70`).
- The hold penalty indexes the true d (`inject_aw.py:163`, H = d+3..K).

**Consequence**
- On every training item, one fixed blind policy reproduces the plan exactly:
  - follow an op that reads the current value;
  - prefer a non-sink;
  - among sinks, take the asked name (one-hop equality);
  - stop when the current entity has no reader.
  Two other blind rules also fit the training data: "skip the reader that sits right after c's sentence" and "skip add_c-on-FRAME_TRAIN sentences".
- The KD teacher (of_s1 on unfaded cues) produces that same trajectory, so KD adds no pressure either (M7).
- **DATA that fits this:** gb (Note 260-5) keeps the *trained* types at H-CHAINSRC 1.000 and H-GIVPOS 0.99. It collapses exactly where a blind rule fails:
  - H-EXTEND 0.08 / 0.02, where S_4 has a reader, so "stop when no reader" continues;
  - H-LIVE 0.40 / 0.41 and CHAINSRC-LIVE 0.06 / 0.41, where the distractor is not a sink.

**Effect on each arm**
- (a) P9-T / P9-G / TC-ABRUPT are trained on that law, so they are predicted to land at the M2 null on every TWIN row. The doc's PR4 0.40 / PR5 0.50 assume pressure that the data does not supply.
- (b) PLAN-PROBE-F trains on the same law (§4: "of's stream law"). A FAIL then parks P9 because of the shared law, not because one pass lacks capacity. M4's "FAIL ⇒ P9 cannot" is not licensed as a capacity statement.
- (c) TH5: the ':' path already solves every training item blind. So nothing forces the decoder to read the NK dataflow block. This is the §262 outcome again (NK ablation drop 0.004). PR9 0.20 is plausible, but for this reason, not for capacity.
- (d) H8: "answer correct now" is fitted equally well by "current entity has no reader" (blind) and by "current entity == asked name". H8-gb tells you which one SGD picked, not whether a halt can be question-conditioned.

**Fix: one lane-level law change, shared by every training arm and by probe arm B.**
- Add `--retarget_p 0.25`. On a seeded share of the d >= 2 items, re-point the final question after perm + inserts (new seed family, e.g. 27076). Either:
  - (i) ask an intermediate spine entity S_j (j < d). The asked entity then has a reader, which is H-EXTEND-type stop pressure; or
  - (ii) ask an insert's X. Retargeting a `chain` X turns S_{j+1..d} into a **live** distractor chain of length d - j. Retargeting an `s256` / `givop` X turns the whole spine into a live distractor.
- Answer, aux, plan cue and d are recomputed from the new target (OW.solve asserted).
- Insert chains stay length 1, so the eval families with 2-op insert chains (H-EXTEND, H-LIVE, H-CHAINSRC-LIVE) stay structurally held out.
- Tag it PROVED: on a retargeted item and its un-retargeted twin, a pass policy invariant to the question gets pair accuracy 0 (M1). So the law itself makes the blind policy lose loss mass (DATA-SIM below).
- **$0 CPU check before registration:** simulate the four blind rules above on 10k items from (A) of's law and (B) the retarget law. Pre-state: A ≈ 1.00 on all; B <= 0.85 on the retargeted share. This is the "null policy OC row" red team 06 finding 12 asked for, applied to the training law and not only the eval.

### 2. HIGH: PLAN-PROBE-F is a learnability probe with blind shortcuts available. It is neither a use test nor, as specified, a capacity test.

**Evidence**
- All parameters are fine-tuned for 1.5k steps on a new query format (" Which name comes after <c>?") with a new output vocabulary: names through the tied head, which also moves wte (red team 06 finding 9).
  - It is not "exactly of's pass-1 computation" (§4): the input text and the readout both differ.
  - S = 0, and c is *given by name*. P9's pass t has to infer c from S_{t-1}'s values (A1).
- So PASS licenses only this: "one forward of this trunk can be trained to output the question-conditioned successor of a named c". That is a necessary condition for P9 (M4), not a sufficient one. It says nothing about what of_s1 or a P9 loop *uses*, and the label must be worded that way.
- FORK_1-3 and START are single items, not twins. The HO families place X *directly after S_j*: `insert_render.py:123` and `held_out2` l. 248-249 both use `_put(..., "after", "S%d" % j)`. Training `chain` inserts sit in the same place. So "skip the reader adjacent to c" solves FORK_1-3 on both H-CHAINSRC and H-CHAINSRC-LIVE without the question.
- STOP is fitted by "none iff c has no reader" (finding 1).
- QMASK_3 is the only question-use clause, and it **deletes** the question. That is a format never seen in training. It also shifts the absolute positions of Q_c and SFX, so the head's output can change for reasons other than selection. The contrast is exposed to the intervention, but it is not exact. M1's exactness needs inputs that differ *only* in what the question names.
- PR1a (0.85 for STOP and FORK_3 >= 0.90) ignores the no-reader and adjacency shortcuts.

**Fix**
- **Binding cells become TWIN pairs at the probe level.** For each FORK_j / STOP / START query, add the twin: the same text and c, with the question re-pointed at X_end. Its label is the insert branch (X, X1, or for STOP the next insert X1 versus " none").
  - Binding: pair accuracy >= 0.85 per cell.
  - A question-invariant head scores exactly 0 (M1).
  - Drop QMASK to descriptive.
- **Two probe arms ($0.12 each):**
  - A, of's law: does the of_s1 training distribution induce the computation?
  - B, the finding 1 retarget law: capacity.
- **Routing:**
  - B FAIL → P9 parked (capacity, under M4).
  - A FAIL with B PASS → P9 runs only on the retarget law. This is the expected outcome.
  - Both PASS → the law change is not needed.
- **Optimization sanity, binding for validity:** non-fork queries (c = S_0..S_{d-1} on clean TEST-PERM) >= 0.95. If this fails, the run is VOID, not FAIL, so a format-learning failure in 1.5k steps can't park P9.
- Fix the readout contradiction (§4 says both "restricted to the 41 tokens" and "argmax outside the set fails"). Pick the full-vocabulary argmax.

### 3. HIGH: S2 GO (and the OC's "P(GO | null) = 0") can be passed by decoder selection from a multi-branch state. The decoder sees the question.

**Evidence**
- `dec = think + SFX` (`inject_ow.py:69-70`). The decoder input is the full text including the question, and `fwd_read` reads S at every token.
- Suppose a pass policy writes S_4 at pass 4 and X1 at pass 5 on H-EXTEND (blind "continue"). If the DeltaMem write keys separate the entities, the decoder at K' 8 can return v_4 to the S_4 twin and v_X1 to the X1 twin.
- S2 GO's twin cells are decoder `pair_acc`, and its WS cell is plain TEST-PERM, not twin. Nothing in S2 binds ws_pair.
- The OC's "no effect" world is a single-branch, gb-like trajectory, so this "superposition null" is not simulated.
- S3 is protected by MECH (ws_pair >= 0.70 at t*) and by Q-USE. S2 is not, and S2 decides the S3 arm. M1's "= 0 by construction" holds exactly only for the ':' readout under identical inputs. G-TWIN is that tautology: a build check, not a control.

**Fix**
- Add to S2 GO: ws_pair >= 0.30 on H-EXTEND and on H-CHAINSRC j 3, plus decoder pair_acc - DECODER-RESCUE pair_acc >= 0.25.
- Add the superposition null to the OC (blind passes, perfect decoder selection) and show P(GO) = 0 under it.
- In S3, make the Q-USE ablation format-matched: re-point the question at an off-path given (or "someone") instead of deleting the sentence. Keep MECH as the airtight row.

### 4. MED: TWIN rows confound selection with extrapolation and with an untrained target class.

**Evidence**
- The twin targets are X1 at depth 5 (H-EXTEND) and X2 at depth 5 (H-CHAINSRC-LIVE j 3). d_max is 4.
- Under of's law the model has never been asked about an insert entity (finding 1).
- So a twin-b miss can be C2 or target-class novelty, not question-blind selection. That costs power, not false positives: M1 still holds.
- The "gold-plan twin reference on of_s1" (§3.2), which would separate these causes, is only descriptive.

**Fix**
- Build the binding twins on TEST2 **d 3** bases, so both members have depth <= 4. Keep the d 4 versions descriptive.
- Run the gold-plan twin reference (of_s1, step cue naming each twin's own chain, eval-only, ~$0.05) **before S2** as a validity gate. A family whose twin-b reference is below 0.85 has its TWIN bar re-scoped before output.

### 5. MED: the plan-free guard covers `passes()` only. Several readers bypass it, and D-BLIND as worded can't run.

**Evidence**
- The PLAN-FREE assert fires only when `seqs_of_t is seq_std` (`inject_aw_eval.py:63-66`).
- `rh_run` (l. 765), `hh_dump` (l. 856-858, its own `pad(seq_std(...))` with no assert) and `ws_readout` (`inject_aw.py:142`, via `cue_inputs`) never reach it.
- Parts with a custom `seqs_of_t` (cue_only, cue_wrong) or spine-built cues (name_swap) skip it by design.
- D-BLIND says "an item copy with names.spine, values and d deleted must give byte-identical tensors". But `tok_st` needs `it["d"]` and the spine to build `think_cues` (`inject_st.py:52-60`), so the copy cannot be tokenised.
- LP5-D4 and SWAP-MID intervene at pass ceil(d/2) / swap_t(d) and read at K' = d (`inject_aw_eval.py:510-520`, 391-405). Those are d-indexed readouts inside a "plan-free" RETAIN row.

**Fix**
- Enforce at tokenisation: for cue in {generic, target, fade_t, fade_g} at p = 1, `tok_aw` asserts that all think_cues are equal and that the only name in the cue slot is the one parsed from the text's final question.
- Add a text-only eval builder `pf_inputs(text)`. D-BLIND compares `pf_inputs(text)` with the trainer's tokenisation.
- Assert in `cue_inputs` as well.
- Score plan-free LP5-D4 / SWAP4 at K' = think_k. The K' = d readings become descriptive.

### 6. MED: TH5 is under-specified, and its oracle is a C4 scope item.

**Evidence**
- of_s1 has `ask_p 0.25`. The assert at `inject_aw.py:267-268` requires `cue == "step"` when ask_p > 0, so `--cue generic` on of's args trips it. (gb_s1 ran with no ask_p.) TH5's diff therefore needs `--ask_p 0`, a second change, or a widened assert. The doc lists neither.
- The write key is the generator's "entity this sentence defines" (nested_words.py:128-144): a hand-built address per sentence end. NORTH_STAR C4 excludes hand-built intermediate structure. The step aux is already a simulator target (Note 263-2), but the oracle adds hand-built *wiring*.
- M11a gives only existence (rank 40 <= 64). W_n learns the keys, so orthonormality is not given at init (`nk.py:43-44`, N(0, 0.02)).
- Formation is predicted to fail under finding 1 (the §262 NK ablation drop was 0.004).

**Fix**
- Label TH5 "DATAFLOW-C1 (parse-given keys; C4 scope)".
- Run the learned-key variant (k = normalize(W_n h_11[p_s]), the doc's §14.2) as a co-arm in the S5 pilot (+$0.35). Do not leave it as a follow-up.
- Put TH5 on the retarget law.
- State the ask_p handling in the diff.

### 7. MED: there is no continuation control, so RETAIN / C2-RETAIN shifts can't be attributed.

**Evidence**
- Every arm runs 8k more steps from of_s1's final weights (`--init_model of_s1/model.pt`) on a fresh lr schedule. Retention changes (d 6 / d 7) can come from the extra 8k, or from the retarget law if it is adopted, not from fading.
- NOKD contrasts only KD.

**Fix**
- Add CTRL-CONT: of_s1 init, the S3 law, step cue, flags off, 1 seed, ~$1. It is descriptive for C2-RETAIN attribution.
- Also explain why RETAIN-PF uses d 1-4 >= 0.95 when score_of.py:6 uses 0.98, or adopt 0.98.

### 8. MED: editing `inject_aw.py` collides with §270 / §271, which are registered first.

**Evidence**
- §270 runs `inject_aw` main.
- §271's `inject_so` imports `InjectAW`, `hold_step`, `cue_inputs` and `build_aw` from it (N1_v1.md:256).
- The doc's §6 edits `inject_aw.py`, `inject_st.py` and `inject_aw_eval.py` in place. Deploys ship the working tree, and long jobs import lazily (memories: deploy ships working tree, lazy-import skew).

**Fix:** build `boxes/inject_pf.py` and `boxes/pf_parts.py` from a mk-script copy with each edit asserted to apply once, as §233 / §271 did. This leaves `inject_aw*.py` byte-identical, and G1 then holds by construction.

### 9. MED: the fade schedule leaves P9-T barely trained at full fade, and P9-G's hold passes use a mismatched input.

**Evidence**
- By M5 + M6, P9-T at d 4 fades only passes 1-3:
  - p_1 > 0 only once f > 0.5;
  - p_3 > 0 only once f > 5/14.
- With pilot `fade_win 250,1500`, only 500 of 2,000 steps run fully faded, while GO is read at p = 1.
- In P9-G, `hold_step` applies `xs[-1]` (pass K's coin) to every penalised pass d+3..K (`inject_aw.py:177-181`). The far pass re-tokenises with the stored cues.

**Fix**
- Use pilot `fade_win 250,1000`, so at least 50% of steps run fully faded. Report the last-1k fade share.
- For P9-G, force p = 1 on passes t >= d + 3, or give hold_step the per-pass inputs. Add this to the SMOKE.

### 10. LOW-MED: scope wording.

- P9 is plan-free **at test only**. In training, the aux names S_t, KD carries the teacher's plan-cued distribution, and the hold penalty uses d. Label: "PLAN-FREE-AT-TEST (plan-supervised in training: aux, KD, hold)". Do not call it plan-free unqualified.
- TC-ABRUPT uses QUESTION(u, S_d), not §261's TGT_FMT (`inject_aw.py:66`). It is §261's *condition*, not §261's arm. "§261 superseded" stays an Audit ruling.
- The §270 overlap is fine: op-only perm here, `full_perm` descriptive. Since the full shuffle is 0.46 / 0.38, every lane-07 ORDER claim reads "op order".
- No overlap with §271's GENERIC arm (N1_v1.md:117): it is a different task (shuffled objects). The only shared surface is the module, finding 8.

### 11. LOW: math and wording.

- TH2's "branches exchangeable under renaming" is false. The branches differ in length (H-LIVE: 4-op spine vs a 2-op insert chain), in op law (REG vs add_c on FRAME_TRAIN) and in position (adjacent to S_j). M1 does not need exchangeability, so drop the word and keep the pigeonhole.
- The TH2 table's H-LIVE branch point should read "S_1 vs X1 at pass 1", not "S_0 vs Y".
- M8: the KD term is zero at s < w0 because the faded set is empty. Teacher = student is irrelevant to that.
- M6 values check: f = 0.25 gives p_8 = 0.5, p_5 = 1/14, p_<=4 = 0; f = 0.5 gives (t-1)/7.
- M11b check: two beta = 1 delta writes on orthonormal keys give S(I - k1k1ᵀ - k2k2ᵀ) + v1k1ᵀ + v2k2ᵀ in either order. PROVED.
- H8 labels at t < d are correct by chance about 1/11 of the time. Report the label noise.
- PR1a is overconfident (finding 2).

### 12. LOW: verified items.

- Code citations hold, within ±2 lines:
  - pass_aw 47-56; colon_logits 61-63; GEN_CUE 73; tok_aw 76-93 (name assert 91-92); ws_readout 134-152; hold_step xs[-1] 179-181;
  - DISTRACT assert 329-330; disjointness 337-338;
  - cue_str 44-46; padding assert 77;
  - NK key 43-44 / read 46-54 / write 56-61 / CUE_NAME_OFF 28;
  - insert families 121-123, 135-138, 139-143, 233, 248-250; _put 97-98; _finish_ho 101-104;
  - test_perm_sets 44-55; SEED_NS 26300 at of_parts.py:15; score_of l. 6 and 45-50; wilson score_fp.py:36.
- Tokens: " someone" = [2130] and " none" = [4844], single tokens (tiktoken, scratchpad).
- Seeds 27070-27079: grep over boxes/*.py, theory/*.py, scratch/*.py, THEORY.md and the backlog finds no use. 70700 is unused.
- of_s1 args match §5's base.
- Guards: S1 0.5; P9 9.0 = 2.0 + 7.0; H8 1.5 = 0.5 + 1.0; TH5 5.8 = 0.8 + 5.0. Each is <= $30, and the total of $16.8 is correct. Cost per 1k steps = 1.65 s × 1000 / 3600 × $0.21 = $0.096.
- The C2-RETAIN power of 0.25-0.77 is disclosed with NEAR-SEED handling and no rescue. That is acceptable for a retention-only label. Prefer the reference min(of_s1, of_s2) - 0.05 over of_s1 alone, so the reference's own seed noise doesn't decide the label.
- Three seeds per C2 claim from one init is "one lineage", consistent with the CI + lineage rules.

## Minimal staged plan I would approve

| Stage | Contents | $ | Guard | Gate / exit |
|---|---|---|---|---|
| S0 ($0) | Apply fixes 1-11. Retarget law spec with PROVED tag; CPU blind-policy sim on law A / law B and on HO-PERM twins (OC null rows, including the superposition null); d 3-base binding twins; `inject_pf.py` copied module; tokenisation-level plan-free assert + `pf_inputs`; scorers + fixtures | 0 | — | Audit G-D; sim pre-state: law A blind ≈ 1.0, law B blind <= 0.85 on the retargeted share |
| S0b | Gold-plan twin reference on of_s1 (eval-only, step cue naming each twin's chain) | 0.05 | 0.2 | twin-b >= 0.85 per binding family, else re-scope that family's TWIN bar before output |
| S1 | PLAN-PROBE-F': arm A (of law) + arm B (retarget law). Binding: probe-level TWIN pair-acc FORK_1-3 / STOP / START >= 0.85; sanity (non-fork >= 0.95, else VOID); QMASK descriptive. Scope "learnable in one pass, c named" | 0.25 | 0.6 | B FAIL → P9 parked (capacity); A FAIL / B PASS → P9 on law B only |
| S1b | H8-gb, frozen head, as written; routing label only (which halt rule SGD picks under law A) | 0.15 | 0.5 | descriptive routing |
| S2 | Pilot from of_s1 on law B, seed 1, 2k: GEN-B (generic cue, no KD: the law-only plan-free control) / P9-G (fade_win 250,1000, KD) / TH5-oracle / TH5-learned (ask_p handling stated). GO adds ws_pair >= 0.30 and the decoder-rescue gap | ~1.4 | 2.5 | GO per arm; if GEN-B is GO, KD / fade is unnecessary and P9 drops to descriptive |
| S3 | Only for GO arm(s): 3 seeds × 8k on law B + CTRL-CONT × 1 (of_s1, law B, step cue). Rows as the doc's §8 S3 with fixes 3-5 (format-matched Q-USE, K' = think_k mechanism readouts, d 3 binding twins) | ~4.7 per arm + 1 | 7.0 per design | PLAN-FREE-AT-TEST-CONTENT-C1 k/3; C2-RETAIN 3/3 vs min(of_s1, of_s2) - 0.05 |
| S4 | H8 on the S3 models (as the doc), if an S3 arm passes MECH | 0.45 | 1.0 | HALT-OWN-PF |

Guards per design stay <= $30. The lane total is about $12 including one TH5 full stage. The cheapest decisive spend is S0b + S1 ($0.3): it tells whether the problem is the law (expected) or capacity, before any $1+ cell.
