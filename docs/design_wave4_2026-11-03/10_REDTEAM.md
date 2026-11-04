Status: independent red team of `10_multitask_joint4.md` (v1, commit 534c46a4), 2026-11-04. Read-only: no training, no reads of run outputs. Files read under `boxes/runs_fsq/`: `of_s1/args.json` and directory listings only. Two CPU checks ran in the session scratchpad (`python -I`, numpy only). (a) `10_oc/mt_oc.py` and `mt_lr.py` were re-run: `mt_oc.py`'s output is byte-identical to `mt_oc_out.txt`, and `mt_lr.py` reproduces J-3's numbers. (b) `rt10/mde.py` (seed 27499) gives the MDE, masking and momentum numbers below, tagged DATA-SIM.

# Red team: MT JOINT (WP + SO, one weight set) — design 10

## Verdict: GO-WITH-FIXES

The core question is worth $7-10, and the design is careful in most places:
- the core question: can one weight set, with no task token, carry both C1 batteries (WP: NOT-DECODER / SWAP4 / LP5; SO: CWS / SWAP-MID / K'2);
- staging is behind §271, and no reading pre-empts it;
- the SO half reproduces §271's batches exactly, and its nominal LR exposure is matched;
- censoring is conservative, and REFERENCE-SHORT blocks MT-PASS.

Four problems must be fixed before registration:
1. The WP half has no matched single-task comparator, and the only control is conditional on failure (F1).
2. The labels NO-INTERFERENCE / TRANSFER claim more than the test can detect (F2).
3. "One design / task-agnostic in form" over-reads two hand-chosen cue handles plus family-restricted readouts (F3).
4. The build edits §271's own trainer while §271 S2 cells may still deploy or rerun (F4).

None of these needs more than about $3 extra. The total stays well under $30.

## Findings

### F1 (HIGH). The WP comparator is not matched, and the WP-only control fires only on failure, so the test is biased toward NO-INTERFERENCE.

**Evidence.**
- The SO half is compared with LOOP-SI: of_s1 plus 8k SO steps, the same batches (G1b), the same nominal LR.
- The WP half is compared with **of_s1 itself**, which has zero further steps. R-WP-DEPTH uses the bars of_s1 − 0.05 (0.921 / 0.859, §6.2).
- The joint WP half is of_s1 plus 8k more of-law steps. Those steps re-warm the LR from ≈0 (of_s1 ended its cosine) back to 1e-4 and start a fresh AdamW.
- ctl_wp_s1 (of_s1 plus 8k WP-only steps) is stage 3a. It runs only "on R-WP-DEPTH FAIL or NEAR-SEED", with one seed.
- Lineage data shows that continued training moves exactly this row by about the margin. Note 263-3: of_s1 d 7 0.909 vs bun_s1 0.857 (+0.052), and of_s2 +0.043, after 8k steps of a changed law from bun.

**Why it matters.** Suppose continued WP training lifts d 6 / d 7 by about +0.05, as bun→of did, and the joint run only holds of_s1's level. Then the true interference relative to a matched WP-only model is −0.05.
- R-WP-DEPTH passes with P = 0.72 (DATA-SIM, `mde.py`), and the label reads NO-INTERFERENCE.
- The control never runs in that branch.
- The asymmetric trigger therefore hides interference whenever drift is positive. It is only checked when drift is negative.
- This also breaks the "control must be exposed" rule: the WP control does not receive the treatment's other ingredients (re-warm, fresh Adam, 8k WP steps) on every branch.

**Fix.**
- Make **ctl_wp_s1..s3 unconditional** in stage 2: of_s1 plus 8k WP-only of-law steps, WP data seeds 101-103, bitwise the jt_sk WP-half batches (the G1b analogue for WP), `--steps 8000`. Cost ≈ 3 × $1.0, so stage 2's guard goes from 7.5 to about 10.5. Still under $30.
- Rebind the WP rows on **jt − ctl**, not jt − of_s1:
  - R-WP-DEPTH = seed-mean(jt) − seed-mean(ctl) ≥ −0.05 at d 6 and d 7, with NEAR-SEED at σ√(2/3);
  - per-seed differences paired by WP batch listed;
  - jt − of_s1 kept as a descriptive "absolute retention" row.
- Drop stage 3a. ctl becomes the matched WP reference, just as LOOP-SI is for SO.
- The labels DRIFT-WP and INTERFERENCE-WP then come from the same 3 + 3 cells instead of one conditional seed.

### F2 (HIGH). Power: small or moderate interference cannot be detected, but the labels name it as if it could.

**Evidence (DATA-SIM, `mde.py`, σ_seed 0.09, the design's own bars).**

R-WP-DEPTH, P(pass) by true shift:

| shift | 0 | −0.04 | −0.06 | −0.08 | −0.10 | −0.12 |
|---|---|---|---|---|---|---|
| P(pass) | 0.72 | 0.44 | 0.30 | 0.18 | 0.10 | 0.05 |

So 80% power to flag interference is reached only at about −0.08, and the false-INTERFERENCE rate at zero shift is 28%.

R-SO-DEPTH, P(pass) by true Δ:

| Δ | 0 | −0.04 | −0.08 | −0.12 | −0.16 | −0.20 |
|---|---|---|---|---|---|---|
| P(pass) | 0.92 | 0.80 | 0.61 | 0.40 | 0.21 | 0.09 |

So 80% power is reached only at about −0.16.

With F1's matched control the WP contrast becomes two-sample, and power falls further. For jt vs ctl on d 7 alone, P(pass) is 0.79 / 0.54 / 0.27 / 0.10 at 0 / −0.05 / −0.10 / −0.15. The 80%-power MDE is then about −0.15 for both families.

The in-range and mechanism rows (W1-W4, S1-S2) sit near ceiling against bars of 0.97-0.98 and 0.50 follow. They cannot see a moderate loss at all. Yet J-6 (ii) predicts that any interference would appear in exactly those causal rows.

The doc's own OC gives P(MT-PASS | truly no interference) ≈ 0.40 at σ 0.09 (§7). So a genuine success fails 60% of the time, while an interference of −0.10 on WP passes 10% of the time.

**Fix.**
- (a) Rename the labels.
  - NO-INTERFERENCE becomes **RETAIN-WITHIN-MARGIN**. Its text states the MDE at 80% power, ≈ 0.15 per family at σ 0.09, and that smaller effects are not resolved.
  - TRANSFER+ stays descriptive, as written.
  - Add **UNRESOLVED** for |Δ| below the MDE with the CI covering 0, so that a pass is never written up as "no interference".
- (b) σ_seed 0.09 (Note 269-2) came from lineage seeds. Here all six cells continue from one init. The of_s1 / of_s2 spread at d 6 / d 7 is 0.012 / 0.014 (Note 263-3), but those two came from different inits.
  - Pre-register a descriptive σ̂ from the 3 jt + 3 ctl + 3 LOOP-SI seeds, and report the MDE at σ̂ beside the binding row.
  - Leave the bars and the NEAR-SEED σ unchanged (no post-hoc tightening).
- (c) Make the mechanism-row comparisons graded as well as thresholded. Report the paired follow / CWS differences jt − reference with Wilson CIs, so that J-6 (ii) can be tested at all. Descriptive.
- (d) The existence claim is the high-power part: JOINT-C1 per seed, P 0.83 arm when retained. State in §1 that this, not interference size, is what the design buys.

### F3 (MED). "One cue rule" amounts to two hand-designed, task-specific handles under a shared name. The readout is also family-specific.

**Evidence.**
- The WP cue is "How many {u} does {S_t} have?" (`nested_words.py:83`, `inject_st.py:44-46`). It names the next dataflow variable, and that variable *is* WP's plan. §260 / Note 260-5 shows the loop fails without it (d 7 0.25 / 0.07).
- The SO cue is " After swap {L(t)}, who has {o*}?" (worktree `so_words.py:91-92`). The label L(t) is printed in the text as "Swap L:" (`so_words.py:114`). So the cue is a pointer to sentence t: a schedule / location handle. In chain-ordered SO that is also the whole plan.
- Neither cue leaks the answer. Both supply the plan, in different forms: which variable (WP) and which sentence (SO).
- The rule "re-ask the question about step t" leaves the real design decision to a human for each family: what the step handle is. A third family needs a new handle designed by hand. KIN would need "slot names", per Appendix A.
- The training laws differ by family: perm / ask / distract run on WP steps only.
- The eval readout is argmax over each family's own candidate set (NUM_IDS vs people_ids). J-5 concedes that this makes cross-family type errors invisible by construction. A family-restricted readout is a task flag at decode time.
- The family is also identified by the first cue token (" How" vs " After", §2.2), so "no task flag" is true only in a literal sense.

**What joint training licenses beyond §271 alone:** "one weight set, trained on two plan-given families, keeps each family's C1 battery". This is multi-task capacity with no catastrophic interference beyond the MDE (F2). It does **not** license a task-agnostic method, cue or readout. Task-agnosticism in the user's sense ("one shared design and code across families") is tested more directly by a third family run on the same code unchanged (§128-style FAMILY-PASS) than by sharing weights.

**Fix.**
- §0 / §2.2 wording: "one weight set, family-rendered plan-given cues; the step handle is hand-chosen per family". Remove "task-agnostic in form".
- Add a **UNION readout** row: argmax over NUM_IDS ∪ VAL_IDS (49 tokens) on W1 / S1 and on WP d 4 / SO n 5 m 4. Bind it to MT-PASS at ≥ (restricted reading − 0.01) per seed. It costs nothing (same forward pass) and is the honest "no task flag at decode" check.
- Keep `leak_f` as a training log.

### F4 (MED-HIGH). The build edits §271's trainer while §271 S2 may still be deploying, resuming or rerunning.

**Evidence.**
- §8 puts `--joint`, VOCS, gating, the logs and the eval views inside `boxes/inject_so.py`. That is §271's LOOP-SI trainer.
- Stage 1 triggers on §271 S1 GO, which is exactly when §271 S2 launches: s1 resumes 4k→8k, s2 / s3 start, plus a pre-stated VOID-RERUN reserve.
- Deploys ship the working tree (standing memory). A §271 resume or rerun tarred during the MT build would carry a modified trainer.
- G1a only proves p = 0 equivalence on the first 50 CPU steps. It does not prove that a half-built tree never ships.
- §271 records `so_eval_module_sha` but not, from what I read, the trainer sha as a hard assert.

**Fix.**
- Put the joint logic in a **new module** `boxes/inject_mt.py`. It imports `inject_so`'s draw / loss / hold functions (refactoring them out only if they are byte-equivalent, under the refactor-equivalence gate).
- Keep `inject_so.py` byte-identical and sha-asserted in jt args.json.
- If a shared-function refactor of `inject_so.py` turns out to be unavoidable, build only after every §271 S2 cell, including any VOID-RERUN, has landed and been drained.

### F5 (MED). The LR matching (J-3) is nominal only. Adam's shared state changes the effective per-family step, and momentum mixes families inside every update.

**Evidence.**
- `make_opt` is AdamW (β defaults 0.9 / 0.999, wd 0.01, `fsq120.py:731-734`). `lr_at` is at 740-743 (verified, and identical in the worktree).
- With strict alternation and β1 0.9, the own-family share of the momentum buffer at an update is 0.526 (`mde.py`). Every update is therefore about half the other family's gradient. "Single-family batches" separate the activations (J-1 holds), but not the updates. The joint run behaves much like mixed-family batches.
- v̂ averages g² over about 1k steps across both families. If ‖g_WP‖ ≪ ‖g_SO‖, which is likely because WP starts converged, then:
  - SO's effective step is up to √2 = 1.41× LOOP-SI's;
  - WP's effective step is much smaller than in a WP-only run.
- So "same examples at the same LR exposure to < 1%" does not hold for the step that matters. Any SO TRANSFER+ or deficit at the 0.05-0.10 scale is confounded with effective LR.
- J-2's first-order formula omits momentum entirely.
- Weight decay applies on all 16k steps, twice LOOP-SI's: about 0.4% extra shrink (negligible; LOW).

**Fix.**
- J-3 re-tagged "PROVED for the nominal LR". Add a sentence: the effective Adam step is not matched.
- Log per-family update RMS (‖Δθ‖ / ‖θ‖ on each family's steps) and the per-group v̂ in each 1k bin.
- Report SO's update RMS against a 300-step LOOP-SI-law replay at G6 (same host, no reading of §271 outputs).
- J-2 adds the momentum term: an update at an A step carries about 0.47 of B's recent gradient.
- An optimiser-state-split arm stays out of scope, but the doc should say that "interference" here means joint training *including* optimiser coupling, not weight sharing alone.

### F6 (MED). A schedule mismatch is not in the args diff: memory fade and hold onset follow the global step.

**Evidence.**
- Worktree `inject_so.py:500`: `m.mem.fade = min(1, step / (500·sc))`.
- `:509`: `inc_on = step >= hold_start·sc`.
- of_s1 has `sched_scale 1.0` and `hold_start 500` (args.json read).
- In the joint run the SO half reaches full fade and hold onset at SO step ≈ 250, against 500 in LOOP-SI. So the "SO half differs only by interleaving" claim (§3.2) is not exact in the first 500 steps.

**Fix.** Either (a) drive fade, hold onset and lam_at from the family-local counter s_f (preferred, since it keeps the single-change rule), or (b) list the change in the args diff and in G1b as a disclosed second difference.

### F7 (MED). No positive control, and the cheapest decisive read is missing: does LOOP-SI already forget WP?

**Evidence.**
- Nothing in the design shows that the WP battery would flag a model that lost WP.
- No arm tells whether joint training is needed at all. If LOOP-SI (SO-only from of_s1) keeps W1-W6, then one weight set already serves both families after sequential training, and the joint run's question shrinks to transfer.
- §271 runs LOOP-SI with `keep_final_ckpt false` (N1_v1 §2). Its final weights will be gone after the drain.

**Fix (eval-only, ≈ $0.3, no pre-emption).**
- Before §271 S2 launches, add by a Note to §271 a descriptive, additive eval part on loop_si_s1..s3: the WP battery (TEST2 d 1-4, TEST-PERM, NOT-DECODER K'2, SWAP4, d 6 / d 7) through the fam_wp view. The Note 269-6 precedent covers this kind of additive descriptive cell.
- Alternatively, set `keep_final_ckpt true` on those cells and run the WP view after §271 is scored.
- This gives the forgetting baseline. It also serves as the positive control for the WP retention rows: a model expected to lose WP shows whether the rows can detect the loss.
- Mirror it on the SO side with of_s1 zero-shot on the SO gate rows (the "SO memory unused" reference).

### F8 (MED). Stage-1 timing and gating mismatches.

**Evidence and fixes.**
- **(a) Exposure after §271 EXTEND.** If §271 S1 GOes only after its EXTEND (8k SO steps), the joint pilot still gates at 4k SO steps (§5). It will most likely EXTEND too, and burn the pilot.
  - Fix: the stage-1 stop is at 2× §271's realised GO exposure.
- **(b) WP rows read mid-schedule.** Stage-1 WP rows are read at jt step 8000, where the LR is 0.5 × 1e-4 mid-cosine. They are judged against absolute bars calibrated on the annealed of_s1.
  - Fix: either read the WP rows as descriptive at stage 1, or accept that EXTEND will fire more often and say so in P1.
- **(c) J-7's stage-1 WP pass ≈ 0.95 at bar 0.97 is not in any committed script.** `mt_oc.py`'s stage-1 cell uses the 0.98 bar plus LATE-PREMISE and gives 0.833. My independent check reproduces 0.956.
  - Fix: add the 0.97-bar cell to `mt_oc.py` before registration.

### F9 (MED). The wall cap and the G6 re-cost threshold are inconsistent.

**Evidence.**
- A 16k cell gets an 11 h wall cap (§10).
- Subtract the battery (1.5 h) and boot (0.25 h), and the cap allows at most 9.25 h × 3600 / 16000 = **2.08 s/step**.
- G6 only re-costs above 2.3 s/step.
- So at 2.1-2.3 s/step the guard kills cells before the final step and battery. This is the guard-marks-must-reach-the-final-step trap.
- Note 271-1 already raised §271's realistic cost about 30% above N1_v1's 1.9 s/step basis, which is exactly the basis used here.

**Fix.**
- Set the G6 threshold to ≤ 2.0 s/step, or set the cap from the measured s/step (cap = 16000 · s/step + 2 h).
- Re-cost with Note 271-1's realism factor:
  - expected about $9.5-10.5 with F1's ctl ×3;
  - guards $2.0 / $10.5 (stage 2 including ctl and the rerun reserve) / – (3a dropped) = **$12.5**, under $30.

### F10 (LOW-MED). The eval views synthesise args.json files.

**Evidence.** `inject_aw_eval.run` reads the whole `<out>/args.json` (lines 1297-1298, verified). Several WP parts depend on more than src / cue / shas, for example the insert / distract and perm settings.

**Fix.** `fam_wp/args.json` = of_s1's args.json with only `out`, `init_model*` and `eval_parts` overridden; same for `fam_so` from loop_si_s1's args. Add a G3 check that the key-by-key diff against the reference args is exactly that set.

### F11 (LOW-MED). D1 logs a different quantity from the one J-2 uses.

**Evidence.**
- J-2 defines c̃ = cos(P^½ g_WP, P^½ g_SO).
- §3.4 logs cos(g_WP ⊘ (√v̂ + ε), g_SO) = g_WPᵀ P g_SO / (‖P g_WP‖ ‖g_SO‖).
- The sign is the same, but the magnitude differs, and D1's ±0.10 bands are magnitude thresholds.

**Fix.** Log c̃ exactly: multiply both gradients by (√v̂ + ε)^−½.

### F12 (LOW). Code-citation drift.

Verified correct:
- `nested_words.py:83`, `:33-34`;
- `inject_st.py:44-46`, `:70-79`;
- `inject_aw.py:159-161`, `:268`;
- `inject_aw_eval.py:1297`;
- `fsq120.py:740-743`;
- `ow_words.py:147-159`;
- `score_n1.py:298`;
- `score_ao.py:28`;
- worktree `so_words.py:25` / `:91` / `:104`, `so_tok.py:19-21`, `inject_so.py:148-149` / `159-164` / `386` / `388-391` / `503`, `inject_so_eval.py:100-104` / `633`.

Wrong in the committed `inject_aw.py`:
- `:363` is `pall_check`; the answer CE on NUM_IDS is at **396**;
- `:552` is the resume RNG restore; the one-d-per-step draw is at **588**;
- `:316-346` is not `draw_items`; it starts at **339**.

The worktree line numbers will move again when §271 commits. The of_s1_init file the doc names does not exist yet (§271 creates it), so its sha is open.

**Fix:** re-cite every line against the committed §271 tree at registration, and pin the init sha in G0.

### F13 (LOW). Censoring gaps.

- If ctl_wp, or under F1 any ctl seed, never lands, the outcome is not stated.
  - Fix: ctl never-reached → the WP contrast is VOID-REFERENCE, and MT-PASS is not awarded (mirroring REFERENCE-SHORT).
- An EXTEND at stage 1 reads the gate and the binding rows on the same jt_s1 16k checkpoint.
  - Fix: state that seed 1's binding rows are then read with this disclosed and counted as usual. Seeds 2-3 are clean.

### F14 (LOW). Is this the right next task-agnostic step?

- For the user's "one shared design and code" direction, the higher-information next step after §271 is a **third family on unchanged code**. That family would ideally have no hand-designed step handle (a plan-free or generic cue), since that is what makes a method task-agnostic.
- KIN is rightly deferred: it needs its own mechanism first.
- A natural-language family, such as ProsQA via doc 05, is closer to the GSM8K path but costs 3-4× per step.
- The joint run is still a cheap, well-staged complement. It answers "one model", not "one method".
- Recommendation: register it as written with the fixes, at a priority **below** §270 / §271 S2 and below a §271 S3 GENERIC read. The JOINT-GENERIC branch (3b) is where the joint design would bear on task-agnosticism, so keep it pre-stated.

## Math tags: checked

- **J-1** holds for activations. F5 qualifies its "weight-only" consequence.
- **J-3** is numerically reproduced, for the nominal LR (F5).
- **J-4** is a correct proof: a question-blind answer equals h_m(o*) for exactly one o*.
- **J-5** is correct, since the token sets are disjoint (`so_tok.py:19-21`).
- **J-6** is correctly tagged CONJECTURE.
- **J-7**: the 0.95 is reproduced ad hoc and is missing from the committed script (F8c).
- **J-2** needs the momentum term (F5).

## Minimal staged plan I would approve

| stage | trigger | cells | binding | guard |
|---|---|---|---|---|
| 0 | now (Note to §271, before S2 launch) | add descriptive WP-battery eval (fam_wp view), or keep_final_ckpt, on loop_si_s1..s3; of_s1 zero-shot on SO gate rows | none (forgetting baseline / positive control, F7) | ≈ $0.3, inside §271's S2 guard or its own $0.5 |
| 0b | §271 code committed + G-D'd | build `inject_mt.py` (F4) with the F5 / F6 / F10 / F11 changes; gates G0-G6 plus the F10 args-diff check; G6 threshold ≤ 2.0 s/step (F9) | build gates | $0 (G4-G6 billed to stage 1) |
| 1 | §271 S1 GO | jt_s1 to 2× §271's realised GO exposure; SO gate rows binding, WP rows descriptive or with the EXTEND risk stated (F8) | GO / PARK / EXTEND as in §5 | $2.0 |
| 2 | stage-1 GO and §271 S2 scored, and every §271 S2 cell drained | jt_s1..s3 (16k) **plus ctl_wp_s1..s3** (8k, bitwise WP batches) | JOINT-C1 per seed (+ UNION readout, F3); RETAIN-WITHIN-MARGIN on jt − ctl (WP) and jt − LOOP-SI (SO), with MDE and UNRESOLVED stated (F1, F2) | ≈ $10.5 |
| 3b | §271 S3 GENERIC passes and a plan-free WP line passes | JOINT-GENERIC | own registration | – |

Expected ≈ $9.5-10.5. Guards total ≈ $13. Claim wording: "one lineage; one weight set; two plan-given families with hand-chosen step handles; retention within ±0.15 resolved, smaller effects unresolved".
