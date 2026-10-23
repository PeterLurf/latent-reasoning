# REVIEW_OW: red team of OW_draft.md (ef5740c7) (2026-10-23)

Reviewer: Theory's red-team subagent. Read-only on the repo, apart from this file and `review_scripts/` (CPU, 1 thread, $0).
- **Inputs:** OW_draft.md, ow_words.py, ow_gen_check.json, ow_oc.py / .json; Notes 223-7 / 223-9 / 223-10 / 223-11 (with 223-11's correction); boxes/codi_ps.py, ps_eval.py, nested_words.py; LIT_ONEPASS_DEPTH_THRESHOLD (with Literature's 10-23 addendum); NORTH_STAR C1 / C4; scratch/gate_s223/pilot3a.json.
- **Checks:** `review_scripts/rv_task.py` → `rv_task.json` (wrap share, held-out-frame exposure, cf(a) vs cf_b on kept pairs); `review_scripts/rv_oc.py` → `rv_oc.json` (an independent OC: a normal approximation plus a separate simulation, with extra scenarios).

## Verdict: REVISE, but the P0 panel can start now
- **REVISE before registration:** issues 1–3 change the R3 target, the test set and the cell sequence, and issue 6 is an unresolved recipe choice.
- **Not STOP.** The generator is clean (I re-ran the F1 numbers I used), the OC arithmetic reproduces, and the serial guard is the right kind of reading.
- **What can go ahead:** the P0 panel (issue 4) needs only the `--task` / `--order_mode` flags and the generator, so it can start once those pass G-D, the flags-off equivalence gate and the plain-path smoke. Nothing in issues 1–3 touches P0's train-bin gates.

## Issues, ranked by severity

**1. [HIGH] R3 scores cf(a) only, so a genuinely serial model can fail it.**
- **Evidence (rv_task.json):** on kept pairs at d 4–8, **cf_b ≠ cf(a) on 54–69 %** (a post-t step back-references some j < t on 64–78 %).
- With the question visible to every latent, a carrier that reads donor S_t from z_t but re-derives the back-referenced S_j from the recipient's text outputs **cf_b**. That is still serial latent use, since donor S_t only exists in the swapped latents.
- **OC:** such a carrier (true follow 0.55) gives P(R3) = **0.000** when cf(a) alone is scored, and 1.000 when cf(a) ∪ cf_b is scored (rv_oc.json). The label would come out DEEP-NOT-SERIAL: a false negative that the draft's OC never models.
- **Fix:**
  - keep a pair iff own, donor-final and cf(a) are distinct **and** cf_b ∉ {own, donor-final};
  - R3 = follow(cf(a) ∪ cf_b) − floor, with the floor scored against the same union;
  - cf(a) vs cf_b share stays descriptive (it says where back-references are read from);
  - with this rule, kept per d ≈ 0.60–0.74 × 0.84–0.87 of the 3,000 candidates ≈ **1,500–1,900**, so the cap of 1,000 per d still holds.

**2. [HIGH] Held-out frames confound the band and R1.**
- The test and pair sets draw from all 8 frames. **66–88 % of test items at d 4–8 contain at least one held-out frame** (rv_task: 0.66 / 0.77 / 0.81 / 0.87 / 0.88; analytic 1 − 0.75^d).
- So DIRECT's band mixes one-pass depth with paraphrase generalisation compounding per step. R1 / R3 inherit the mix, and LAT may generalise frames differently from DIRECT.
- **Fix:**
  - primary test set and pairs on FRAME_TRAIN (frames 0–5); B, R1, R2 and R3 read there;
  - a second set on frames 6–7 is descriptive (paraphrase transfer);
  - a new test-set sha; exclusion unchanged.

**3. [HIGH] Sequencing: build the expensive flags only after the cheap band check.**
- The draft runs P0 → P1 (needs `--recode_target`, `--text_cot`, `k_abl`, `text`, with G-D, equivalence and gradflow gates) → cloud DIRECT → G-W0.
- The draft itself gives ONE-PASS-SUFFICES-OW ≈ 0.25 given a P0 pass. In that case all the LAT / TEXT engineering is wasted.
- **Fix:**
  - order the cells P0 panel → cloud DIRECT (≈ $0.55–0.85; needs only the task flags plus the existing `acc` part) → G-W0 Note → then build and gate the LAT / TEXT flags → P1 → cloud LAT + TEXT;
  - **concern (5) pre-check, $0:** if P0 passes G3 before 12k, continue the same run to stop_at 24k (≈ 2.4 h total on the 4060, short-local);
  - if DIRECT then advances by threshold through d_max 5 with bin acc ≥ 0.85, record **ONE-PASS-LIKELY** and move straight to the stage-2 / DAG-depth fallback named in issue 4's mapping, before any cloud spend.

**4. [HIGH] Make P0 an attribution panel (concern 1 and Literature's question), pre-stated, $0, about 3.6 h on the 4060.**
- **Concern (1) confirmed:** v1.2 had 0 / 20,000 order violations, so OW's levers are the value law (mod 11, no range rejection, operands and distractors 1..10) plus given placement and layout (gap 4.43 → 0.10; canonical share 0.22 → 1.00).
- "VAL-only" is mislabelled. `--order_mode topo` on mod-11 values is **mod 11 + v1.2 layout** (call it TOPO).
- **Key fact:** §223 P3a ran with the *same* recipe (lr 3e-4, cg_thr 0.45, cg_cap 6000, steps 48k, stop 12k, 4060, registered kernel). So P3a is the third corner, and the panel is attributable with one card class (the GPU-class rule).

| cell | generator | isolates (Literature's list) | status |
|---|---|---|---|
| P0-OW | forward, mod 11 | — | binding (G1–G3) |
| P0b-TOPO | topo, mod 11 | P0 − P0b = (1) given distance / layout; P0b − P3a = (2) values (with English semantics, issue 7) | **mandatory**, descriptive |
| P0c-CHAIN | forward, mod 11, **no add_s and no forced mul_s** (strict previous-state chain; _c inline, _v adjacent) | P0 − P0c = (4) DAG back-reads | recommended, descriptive |
| P0d-FULLFT | P0-OW with full fine-tuning (lr 1e-4, fp32 master weights) | (5) recipe | **conditional:** only if P0 fails G2, and before any close |
| (3) frames | 1 frame per op | (3) layout diversity (Arun) | only if P0 and P0d both fail; lowest priority, since language is the point |

- **Notes on the panel:**
  - P0c needs a generator-only flag (`dag=False`) plus its own $0 F1 rerun. Its affine-chain answer is the task, not a shortcut.
  - Its value for interpretation: if the chain is easy and the DAG is hard, the band's source is the back-reads, which gives the cleanest "exceeds one pass" story.
  - Each extra cell is ≈ 72 min at 0.36 s/step.
- **Pre-stated mapping (no-loop rule; one fallback only):**
  - P0 passes → proceed (issue 3 order); P0b / P0c are descriptive.
  - P0 fails and P0c passes → the single allowed fallback is OW-CHAIN as the stage-1 task, under a dated Note. G-W0 is unchanged, and the band risk rises.
  - P0 and P0c both fail → P0d. If P0d also fails → close descriptively.
- **Args diff:** state P0's diff against **P3a's args.json**: it must be {task, order_mode, stream_seed, hashes}. P0b vs P0 = {order_mode}; P0c vs P0 = {dag}.

**5. [MED-HIGH] LoRA vs full FT (concern 2).**
- §170's one-pass success (n ≤ 2 at ≥ 0.90 on mod-11 expressions) used full FT at lr 1e-4. §223's failures were all LoRA.
- A P0 close declared on LoRA alone would be recipe-confounded.
- **Fix:** P0d is conditional (issue 4). It is not needed if P0 passes, because LoRA is then shown to suffice for d ≤ 2.
- Building a full-FT path into codi_ps (bf16 base → fp32 master) is a new code path with its own G-D, equivalence and gradflow gates. Budget it now, rather than discovering it at a close. Issue 6 gets it for free.

**6. [MED] The latent recipe has no C1 record. Justify codi_ps, or port §184.**
- CODI-style codi_ps: M1 ≈ DIRECT (223-7 / 223-9); §209 CODI was MIXED.
- §184 (INJECT + step aux + AUX-REWEIGHT, full FT, nested170) **met the C1 conjunction on mod-11 nested at 2 / 2 seeds**.
- Porting §184 to OW items would make the claim single-factor ("does the C1 recipe survive an English surface?"). nested170 also already has full-FT `direct` and `textcot` arms (closes issue 5).
- **Cost:** an OW item adapter in nested170 (question tokens, step-sentence aux per pass, single-token answer) with G-D and an equivalence gate on the §184 path. That is not trivial, since nested170 is heavily flagged.
- **Fix (minimum):** a paragraph in v1 choosing between them, with the reason and the prior. The draft's P(R1 | B) = 0.30 should say which recipe record it comes from.

**7. [MED] Mod-11 makes most statements false in English (concern 1, last part).**
- **Measured (rv_task):** wrap shares are **add 0.48, sub 0.52, mul 0.85** of statements.
- **82 %** of d = 1 items and **≥ 98 %** of items at d ≥ 3 contain at least one false statement.
- So the task is English-templated modular arithmetic. This confounds the P0b − P3a "values" attribution (the value change also flips the semantics), and it weakens "language" in C1-LANG.
- The draft is right that dropping mul is not a fix: add/sub only is a signed-sum shortcut (Lin's commutativity), and with it there is no band.
- **Fix:**
  - word every label "English-templated modular arithmetic (mod 11)";
  - add a P0 readout at stop: test acc on d = 1 / 2 (train frames) split by whether a step wraps (descriptive; it measures the prior conflict);
  - disclose the confound in the P0b − P3a reading.

**8. [MED] CurGate 0.45 and the THRESHOLD-MISSET rule (concern 3).**
- **P3a anchors (100 classes):** CE 0.443 ↔ acc 0.895; 0.273 ↔ 0.921; 1.41 ↔ 0.574.
- With 11 classes, error mass spreads over fewer classes, so the same CE means *lower* accuracy. A scaled uniform-confidence model puts 0.45 at acc ≈ 0.83–0.87 [ASSUMED]: if anything looser than on 0–99. The likely mis-set case is therefore G2 passing at 0.80–0.84 while G3 misses.
- **Flaws in the rule:**
  - (a) **Asymmetric:** if G1 fails on the cap with d = 1 bin acc ≥ 0.85, the draft closes OW. This should map to THRESHOLD-MISSET too.
  - (b) "CE of the first bin with acc ≥ 0.80, + 0.05" sets the cloud threshold *below* the G2 bar (advances at ≈ 0.78). Instead, interpolate P0's own (acc, CE) 1k-bin pairs at a fixed target acc of 0.85.
  - (c) The DIRECT-derived threshold lets TEXT (answer CE after the CoT ≈ 0) advance early. Disclose this as a per-cell exposure difference.
- **Fix:** (a)–(c), plus log per-d num_acc (logging only, outside the bitwise training path), or read it at stop from the checkpoint.

**9. [MED] The R3 floor arm is not matched.**
- The "next kept pair's donor" has other names and another skeleton, so its z carries foreign names. The recipient's continuation may collapse to own (which is excluded), pushing the floor below the ideal-carrier 0.09–0.14 and inflating follow − floor by up to about 0.14.
- **Fix:**
  - matched floor = a **second redraw donor on the recipient's skeleton** (same names / order / token length), scored against donor 1's target set;
  - average over ≥ 3 such donors to tighten it;
  - ps_eval's docstring scores the floor against *donor-final*. The OW scorer must score against cf(a) ∪ cf_b. State this in the scorer spec and test it on synthetic rows (the scorer field-path rule).

**10. [MED] C4 legality.**
- NORTH_STAR C4 excludes "a task simulator's per-step states". Generator-written step sentences are exactly those, rendered in English. Distilling from TEXT does not launder them, because TEXT is trained on the same generator CoT.
- By the §175 / §184 precedent the most this supports is "C4-qualified (generator CoT)".
- **Fix:** the headline claims C1 only, with C4 "not assessed on a templated task".

**11. [MED-LOW] The OC block: reproduced, but its null is vacuous.**
- **Independent recompute (rv_oc.json, normal approx + own simulation):**
  - no-effect: R1 gap z = **−22**, so 0.000 is correct but uninformative. Even a gap-0.20 null sits at −7 SD → 0.000;
  - at the bars, the joint pass is **0.05–0.12**: 0.05 with R1 and R2 independent, 0.12 (the draft) with LAT shared, which is the more realistic case;
  - effect_marginal: 0.85 (the draft, independent K′ arms) to 0.96 (paired arms, mine).
- The threat is systematic, not sampling.
- **Fix:** add a mechanism-null table to §8 (pause, relay, cf_b-type carrier, floor-collapse weak carrier) under the issue-1 / issue-9 rules.
- Two minor points:
  - set the OC's N_PAIRS to the realised minimum of 2,000. B needs ≥ 2 depths ≥ 4 and each gives 1,000, so the "< 1,000 → DEEP-UNREAD" clause can only trigger on an engineering fault. Say so;
  - the band-selection winner's-curse bias is −0.0015 (negligible).

**12. [LOW] R3 counts and the permutation floor (concern 4).**
- 3,000 candidates per d give **≈ 1,810–2,210 kept** (rv_task, kept share 0.60–0.74); the cap is 1,000 per d, so N ≥ 2,000.
- SE: follow ≈ 0.011, floor ≈ 0.008. The 0.30 bar is ≈ 25 SE.
- The ideal-carrier floor is 0.09–0.14; the union floor (issue 1) is ≈ 0.2.
- Adequate. Only the matching (issue 9) matters, not N.

**13. [LOW] R2 wording vs NORTH_STAR C1.**
- C1 asks for K′ 0 / 1 to drop "to chance". With the question visible, K′ = 0 will sit near the one-pass level, not 0.091.
- **Fix:** word R2 as "to the one-pass level" and report acc(K′ 0 / 1) against both DIRECT and chance.

**14. [LOW] G2 is pooled.**
- Pooled d ≤ 2 ≥ 0.80 passes with d = 2 at ≈ 0.65 (P3a: d = 1 ≈ 0.9).
- **Fix:** add a d = 2 test readout from the stop checkpoint (train frames, 1,000 items, ≥ 0.70), binding or descriptive, pre-stated.

**15. [LOW] Realised $ for LAT is optimistic.**
- The 4060's 3.16 s/step (M1) is dominated by the K-step latent loop with backward through K. The step target only shrinks D.
- At 2.5 s/step on a 3090, 48k steps ≈ 33 h ≈ $7.3, which exceeds the LAT share of $6.
- **Fix:** F2 also times `--lam 0` (to apportion the cost), and an over-cap rule is pre-stated (re-scope via a Note, not a silent step cut).

**Goodhart check (no issue).**
- D's input is z_j only, with no question KV (verified in codi_ps DHead). So the step target is a binding decodability target, not a dynamics target.
- Redraw keeps names, so swaps do not move name tokens.
- D6 holds: decodability alone lands in NO-GAIN or DEEP-UNREMOVED, never C1-LANG.

**Controls exposed.**
- K′, MEAN-Z, SELF and the floor all receive the intervention. Fine, given issue 9's matching.
- d = 3 is correctly excluded (cf(a) ≡ donor-final).

**Censoring.**
- Rule (b) is applied. Unreached depths are scored as wrong, and the caps guarantee reach (7 × 6k = 42k < 48k).
- **Add:** report per-d training exposure per cell, and label R1 CONFOUNDED if LAT's exposure at B depths is > 2× DIRECT's (curriculum reach, alongside the basin-entry rule).
