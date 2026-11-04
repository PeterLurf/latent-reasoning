# 14 RED TEAM: independent review of the paper outline (14_paper_outline.md)

Independent red team, written 2026-11-04. It replaces the lead's self-review in §8 of the outline. Read-only: no runs and no new computation on stored outputs. Every number below was read from docs/THEORY.md (status rows 239-271 and the Notes cited), docs/sota/sota_arch.json (as_of 2026-11-04), theory/results/score_{of,bu,c5c_fresh}.json, docs/EXPERIMENT_BACKLOG.md §12.8 / §13.0, docs/LITERATURE.md and docs/LIT_*.md.

## Verdict: REVISE (NOT-YET for submission)

The outline is a sound plan. Its scope words are mostly right and its decision rule is honest. But four things are wrong as written:

1. The headline mechanism, "content selection, not depth budget", is contradicted by the project's own Note 256-1.
2. The "three nulls" citation points at sections that are not depth-budget tests.
3. The outline predates §263 (SOTA is now of_s1 / of_s2), but every pending read (§264, §266, §268, §269) is on bun. The paper has to choose which model it is about.
4. The key S1 baseline (§266 X3) is not plan-matched.

None of these needs new compute to fix in the text. Items 1-2 are rewording, 3 is a framing choice plus one ~$1 stage, and 4 is an added arm or a reworded S1.

## Findings

### F1 (HIGH). The headline mechanism is contradicted by Note 256-1

- **The claim.** Outline §1 H1, title option 1 and the abstract say "the depth horizon is set by *content selection* … not by the pass budget" and "the extrapolation came … from training the step to *select* its sentence by content instead of position".
- **The evidence against it.**
  - Note 256-1 (§256 DISTRACT): C2 was unlocked at §256 while **SELECTOR failed 0/2**: d 6 0.981 / 0.961 vs rcb 0.346 / 0.368. The inserts taught "a surface skip-rule … not lookup by the cued name". H-CHAINSRC 0.69 / 0.806, H-LIVE 0.13 / 0.27, and TEST-PERM stayed 0.10-0.18.
  - The Note's own reading is: "Breaking absolute slot counting during training, **even via a surface filter**, is what unlocked depth extrapolation."
  - Content selection came one stage later (§257, Note 257-3: CONTENT-SELECTOR 2/2). It left d 6 unchanged (0.971 / 0.963 vs dis 0.981 / 0.961) and raised d 7 only slightly (0.900 / 0.889 vs 0.874 / 0.809).
- **Fix.**
  - Reword H1 and title 1 to "breaking positional slot counting (distractor insertion as a slot-decorrelation device), not pass budget". That is the wording Literature adopted in Note 256-1, citing Ruoss 2305.16843.
  - Present content selection (§257) as a separate C1-scope result that keeps C2, not as its cause.
  - Drop "content selection" from title option 1, or use title 2 or 3.
  - The outline's red-team item 4 flagged the confound but not this contradiction.

### F2 (HIGH). The "three pass-budget nulls (§239 / §240 / §243)" are mis-cited

- **What the cited sections are.**
  - §239 LONG-RUN is a C6 hold test at d 2-4 (K' 8-64 drift).
  - §240 LANG-HOLD is the hold penalty (C6).
  - Neither tests whether more passes extend depth.
  - §243 DEPTH-CURRIC trained d <= 6 with think_k 10. It is a curriculum test, and s2 passed EXTRAP (f_DC(+2) 0.87 vs CTRL 0.26). Its only budget-like clause is C2-RISE (d 12 flat).
- **Evidence that does exist.**
  - §241 LEN-STEP (STEPS: length effect 0.002 / 0.005; d 6 fails at any length).
  - Note 256-1: accuracy at d 6 rises at K' 5 and is "flat to K' 14", so passes past d - 1 add nothing.
  - Note 269-4(c): the K'-probe at d 9-10 is "MIXED, uninformative" (floor effect).
  - Backlog §12.8 taxonomy: "extra passes as depth lever (4)" failures. These are not traced to sections.
- **The scope word "K' > K does not help"** (outline §2.2) is stated as fact. Its test is §264 STD K' 8-24 at d 6-7, which is registered and has not been read. Note 269-4(c) explicitly moved the question there.
- **Fix.**
  - Theory traces the actual "extra passes" nulls by section before the paper cites any count.
  - Until §264 reads, write "K' > K is not yet shown to help" and drop "three nulls".

### F3 (HIGH). The outline has not been updated for §263, and the SOTA model differs from the model the pending reads test

sota_arch.json (as_of 11-04) names **of_s1 / of_s2** (Note 263-3). The outline is written about bun. The following are stale:

| Outline text | Current fact (source) | Fix |
|---|---|---|
| Limitations: "chain order until §263 reads"; T7 "§263 running" | §263 SCORED: ORDER-FREE-C1 2/2. TEST-PERM d 4 0.982 / 0.961 (Wilson [0.972, 0.989] / [0.947, 0.971]). LATE-PREMISE >= 0.979. **Full all-sentence shuffle d 2/3/4 = 0.806/0.596/0.462 and 0.695/0.468/0.376.** Independent shuffle d 4 0.967 / 0.878 (Note 263-3; score_of.json graded_desc) | Limitation becomes "op order free; givens / S_0 must keep their trained placement relative to the ops (full shuffle d 4 0.46 / 0.38); §270 FULL-PERM pending" |
| Abstract: d 7 "0.85-0.86" | bun 0.857 / 0.852 (Note 259-4) is correct for bun. of is 0.909 / 0.895 (score_of.json c2_rows), with seed-level C2 claim **NEAR-SEED / NEAR-SEED**: no C2 improvement may be claimed (Note 263-3, Note 269-2 (3)) | If the paper is about of, quote 0.91 / 0.90 with the NEAR-SEED label. If about bun, say so and give of as the order-free extension |
| §2.3 Setup: "6-stage lineage, 68k steps" | of adds a 7th stage (8k from bun with perm-then-insert + 25% one-hop ':'-CE co-training; sota_arch training_lineage), so ~76k steps | Update. Also disclose that the co-training items are simulator-built intermediate states (Note 263-2: not a C4 route) |
| F4 "done on dch; bun re-score $0.1" | of passed CONTENT as a **binding** criterion on insert_ho (Note 263-1, 263-3). Only bun lacks it (Note 269-3 (a)) | If the model is of, F4 is done ($0). The bun re-score matters only if bun stays the paper's model |
| T2: causal battery from §259 | of passes NOT-DECODER / SWAP4 / LP5 under RETAIN (sota_arch C1 evidence; Note 263-3 "RETAIN" true in score_of.json) | Source T2 from §263 if of is the model |
| F8: C5 "done (r = 3 post hoc; gb out-of-sample)" | On bun, r = 3 was confirmed on a pre-stated FRESH draw (Note 259-11 / 259-12, "r3 C5-EXTRAP / C5-EXTRAP -> PASS"). It is registered, not post hoc. On **of**, C5 is the same-draw TEST2 reading, computed in memory (sota_arch C5 evidence) | Fix both the F8 row and red-team item 6. If the paper uses of, C5 on of is descriptive (same draw) |
| Limitations: "GPT-2 small only (medium shortcuts the last hop, Note 264-1)" | Correct (Note 264-1 M1: V1+TEXT). But **small also computes the last hop in the decoder**: Note 264-4 DECODER-LAST-HOP (d 4 acc at K' 3 = 1.000 / 1.000); sota_arch "decoder … adds one hop itself" | See F7 |

**The structural problem.** §264, §266 (X3 / X5 / C7), §268 (lineage seed 3 of *bun's* lineage: askname -> … -> bun, Note 268-1) and §269 / §269b are all **on bun**. §270 and §271 start from of / bun_sX_init.

- A paper "about of" therefore has no plan-clock read, no same-data text baseline, no third seed and no triage on its own model.
- A paper "about bun" is no longer about the SOTA model.

**Fix.** Make bun the paper's core model (every pending read lands on it). Report of as the order-free extension, with its own 2-seed / one-lineage / NEAR-SEED labels. Then either:

- (a) add a §268 follow-on stage, "of from bun_s3" (8k; at Note 268-1's ~1.7 s/step that is ~4 h, ~$0.8 at $0.2/h); or
- (b) state that of's rows are 2 seeds of one lineage and are never pooled with bun's 3.

### F4 (HIGH). S1 (§266 X3) is not plan-matched, so a pass is open to the obvious objection

- **The asymmetry.** X3's text CoT is trained to write "<S_t> has <v_t> <u>." (tx) or bare values (txv) with **no step cue** (§266 registration). The text model must generate its own plan. The loop receives the plan every pass.
- §260 (Note 260-5) shows that without the plan the loop's own d 7 falls to 0.246 / 0.066. If text loses at d 7-8, a reviewer will say the plan explains the gap, not the latent loop.
- The outline's red-team item 1 notes the training-budget asymmetry (24k from pretrained vs ~68k lineage + askname init). Note 266-1 asks for steps / tokens to be reported. The plan asymmetry is not mentioned anywhere.
- **Fix (pick one).**
  - Add a plan-given text arm: the cue sequence S_1..S_d is printed in the prompt, or each text step is prefixed by the cued name. This is ~$1.2, one cell, and needs its own registration as a §266 amendment *before output*.
  - Or word S1's pass as "latent-specific relative to plan-free text CoT" and compare against gb (plan-free loop) too.
  - Also add the outline's own falsifier (a text cell from s233_askname_s1, ~$1.2) to equalise the init.
- **The pipeline result is also likely to cut against the headline.** §266's own OC puts **P(PIPELINE-SOLVES) ~0.7**. A PASS rewords C1 to "latent value transport under a given plan; an external one-hop program of the same model solves the depths the loop loses". The outline treats S9 as "wording" only.
  - The most likely §266 outcome is that a trivial external loop beats the latent loop at d 7-8. The intro must be drafted for that case now (Note 266-2 (b) PIPELINE-MATCHES), not after.

### F5 (HIGH). The C2 headline violates Audit's ">= 3 seeds" rule as written

- **The rule.** Note 269-2 (2): "Any C2 claim needs >= 3 seeds, and >= 2 lineages once §268 lands. Each seed is reported as its margin vs the bar, plus the seed-level mean ± SD."
- **The abstract** prints d 6 / d 7 ranges from 2 seeds of one lineage.
- **The decision rule** lets SUBMIT-H1 go ahead on "LINEAGE-PARTIAL with C1 + C6", i.e. with C2 *not* replicated.
- **Fix.**
  - Under PARTIAL, C2 drops from the abstract to a "2-seed, one-lineage, not claimed" line.
  - The abstract's C2 sentence is conditional on §268 LINEAGE-REPLICATES with per-seed margins.
  - For of, C2 is NEAR-SEED and not claimable at all (F3).

### F6 (MED). "Seed spread (sigma 0.09 at d 6)" is mis-scoped

- sigma_seed 0.090 is the between-seed SD **pooled over 21 _s1 / _s2 pairs across many recipes** (Note 269-1 ST2), not bun's spread.
- bun's own seeds differ by 0.016 at d 6 (0.972 / 0.956) and 0.005 at d 7.
- Draw-to-draw variation is also visible: bun d 7 is 0.857 / 0.852 on TEST2 but 0.883 / 0.881 on the fresh draw (score_c5c_fresh.json).
- **Fix.** Write "project-wide between-seed SD 0.09 at d 6 (21 seed pairs); bun's two seeds differ by 0.016". Report the TEST2 vs fresh-draw range for d 7.

### F7 (MED). "The latent state carries each step's value" overstates the mechanism

- **What the evidence shows.**
  - The decoder computes the last hop itself: DECODER-LAST-HOP (Note 264-4) and sota_arch "decoder … adds one hop itself".
  - The r = 3 halt lands at d - 1 for that reason.
  - Causal gates exist only at d 3-4: swap_kd / lp5h loop over d 3..d_max (Note 269-3 (c)).
  - T2b (d 6-7) is not registered and is not in the §269b spec (18_s269b_bundle_specs.md covers noise / subs / tta / sclat / noop only).
- **Fix.**
  - Abstract: "carries steps 1..d-1; the decoder completes the last hop".
  - Cite Note 269-4 (a) ERROR-PROPAGATION as the existing behavioural evidence at d 5-8: SERIAL / MIXED, recovery 0.077 / 0.111 vs chance 0.091. Present it as descriptive support, not a substitute for S2.
  - Register T2b, with the outline's red-team item 2 (NOT-DECODER at d 6 and swap at t = d - 1) built in.

### F8 (MED). The C5 framing is weaker than F8 implies, and the evidence for that exists already

- Note 264-4: on in-range items the halt is "essentially a function of d, i.e. of the given plan's length" (P(khalt = mode d - 1) >= 0.976 at d 3-4).
- Note 269-4 (b): |acc_halt - acc(pred[d-1])| <= 0.007 at every d 1-7. "In accuracy, the r = 3 halt adds nothing to 'read the answer at d-1 given d'."
- **Fix.** Move C5 out of the headline now. Present F8 as "a training-free stability stop that coincides with the plan's clock", with §264 ER / LFI / CY deciding whether it is anything more.
- The outline's prior P(PLAN-CLOCKED) ~0.7 is consistent with this. The headline should not wait for it.

### F9 (MED). The expressivity claim has no written proof in the repo

- "A single 12-layer pass could express these programs" (d ~ 64) is a wave-3 agent finding stated in EXPERIMENT_BACKLOG §13.0.
- The proof memo is still a backlog item (TH1 EXPRESSIVITY-MEMO, $0).
- "N1 (S5) is where a loop is provably needed" holds only under the standard TC0 ≠ NC1 assumption at bounded precision.
- **Fix.** Write TH1, with a construction or a citation (2305.15408 / 2310.07923 / 2502.17416), before the abstract states it. Phrase N1 as "conditional on TC0 ≠ NC1".
- Also cite the existing **trained one-pass baseline** on REG: CD-AN at d 3 / 4 0.085 / 0.090 (§237). It is the "learnability, not expressivity" evidence and the outline does not use it.

### F10 (MED). Related work and novelty: no paper-level Literature check; key priors are missing

- **What has been checked.** No LIT file checks the paper's conjunction claim ("causal latent steps + extrapolation + fixed point in one pretrained LM"). The checks that exist are narrow:
  - §256 distractor insertion: NARROW, "not found … as a training device for depth extrapolation in looped / latent models" (Note 256-1).
  - §263 order-free: "no 'first' claim on order invariance; the training remedy itself is prior art (2502.19907)" (docs/LIT_S263_ORDERFREE_RESULT_2026-11-04.md).
  - The hold penalty is cited as prior (outline §2.5, correct; backlog §12.8).
- The outline makes no explicit "first", which is good. But its intro contrast ("Coconut / CODI no depth extrapolation; Huginn / Ouro no causal battery") is an unchecked statement about other papers' contents.
- **Missing from §2.5, and most likely to be raised by reviewers:**
  - **PAR 2610.27628.** A retrofitted looped Qwen2.5-0.5B with per-iteration progress supervision and norm anchoring; swap-S5 trained on <= 32 moves, 95.5% at 512 (LITERATURE.md l. 203). This is a far larger extrapolation ratio than +2-3 depths. It is the closest "looped pretrained LM + per-step supervision + stability" prior.
  - **2609.35643** (Cheng & Zhang). Already in §2.5 but under-weighted: trace-free latents beat CoT on held-out 7-12 hops, **with state transplants that redirect the answer** (LIT_DEEP_PROSQA_2026-10-08.md correction). That is causal latent steps plus depth extrapolation, from scratch. The paper's difference must be "pretrained LM + given plan + hold", stated explicitly.
  - §263 priors: Premise Order 2402.08939, order-centric augmentation 2502.19907, Wu / Geiger / Millière 2505.20896, Dai 2409.05448.
  - Distractor priors: GSM-IC 2302.00093, iGSM 2407.20311, Ruoss 2305.16843.
  - Copy-shortcut gate 2605.22870 (Note 266-2).
  - Causal-carrier critiques of CODI / Coconut: 2608.27265 SCIT, 2512.21711.
- **Fix.** Ask Literature for a LIT_PAPER_NOVELTY check of each contribution sentence in the abstract and of the intro's claims about Coconut / CODI / Huginn / Ouro, *before* the intro is drafted. Add the papers above.

### F11 (MED). H2 is a different model from H1, and the comparisons need care

- **Different model.** D1 (05_prosqa_ood_port.md) ports the **§191 loop** (write from every token, `[EOT] + question` every pass, plan-free, text-step supervision), not bun / of. H2 is therefore "H1 + a different architecture on a public split". The outline does not say so.
- **Plan-free.** §260 shows plan-free selection collapses depth on REG (d 7 0.07-0.25). That is a strong prior against S8, so P(S8 | D1) ~0.3 may be optimistic.
- **External numbers.** SubLAT 2610.40283 reports GPT-2 SubLAT 69.34 vs COCONUT 67.52 OOD avg (LITERATURE.md l. 197). The splits are regenerated, so these are context only. The outline's red-team item 8 is right on this.
- **N1 is stale in the outline.**
  - §271 is registered (not "not yet").
  - Its arms are LOOP-SI / **DIRECT-SI** (binding one-pass control) / DIRECT-OF. There are no text / pause / Coconut arms.
  - Guards are $2.5 / $6.5 / $2.0; Note 271-1 gives realistic ~$9.3. The outline says $3-4, so H2 via N1 is ~$23-25, not ~$18.
  - A paper N1 table needs a text CoT arm. Otherwise the only baseline is one pass.
- **Fix.**
  - Name the H2 model explicitly.
  - Update T9 / §7 to §271's arms and costs.
  - Add a same-supervision text CoT arm to N1, or drop "text" from T9.
  - Keep external numbers in a "context, not comparable" column.

### F12 (MED). The decision rule depends on unregistered items

- S2 (T2b) and S3 (B1 PAUSE-PASS) are binding for SUBMIT-H1, but neither has a § number or a registration. B1 needs build gates (outline §4).
- **Fix.** Register T2b and B1 before any of §264-§269 lands, so the submit rule cannot be tuned to the outcomes.
- Under the Audit pre-state rule, score_paper.py's thresholds also need to be committed and acked before the first S-row input lands. §264 / §266 can land any day.

### F13 (LOW). §261 is cited as evidence but has no data

- Outline §2.3 (6) and T3 use "§261 (target-only cue)". §261 has no result and is parked by Audit (THEORY l. 24254; CURRENT_TASKS 11-03 14:48).
- **Fix.** Drop §261 from T3, or mark it "not run".

### F14 (LOW). Small wording fixes

- "Hold … keeps accuracy and state norm fixed to K' = 128": ||S|| ratio is 1.02-1.14, hold verified at d 1-6 (Note 259-4; score_bu.json s_norm_ratio_128_8). Write "within 1.02-1.14x, d 1-6".
- "~3.5 bits" = log2 11 = 3.46. Fine, but say it is a property of the task (mod 11), not a measured capacity.
- §5 priors: P(SUBMIT-H1) ~0.3 exceeds the product of the stated component priors (0.6 x 0.6 x 0.8 x 0.6 ≈ 0.17). Either state the assumed positive correlation or lower it.
- "2.4-9.1x the FLOPs" matches Note 266-1. Add "analytic; the C5 halt adds ~2x" as Note 266-1 does.

### F15 (LOW). Numbers verified as correct

Each of these matches its source:

- d <= 4 1.000 (Note 259-4)
- bun d 6 0.972 / 0.956 and d 7 0.857 / 0.852 (Note 259-4; score_bu.json)
- chance 1/11
- K' 128 hold (score_bu.json acc128 = acc8 at d 1-6)
- plan-free d 7 0.246 / 0.066 (Note 260-5)
- 2.4-9.1x FLOPs (Note 266-1)
- sigma_seed 0.090 (Note 269-1; mis-scoped, see F6)
- Note 264-1 per-pass reliability (pass 7 at d 8 0.59 / 0.80)
- X3 rule t7 <= bun7 - 0.10 and t8 <= bun8 - 0.10 (§266)
- H1 cost sum ~$14.2
- C5 fresh-draw values (Note 259-12; score_c5c_fresh.json)

## Fair-comparison checklist (per claim)

| Claim | Baseline it needs | Exists? | Fair as planned? |
|---|---|---|---|
| C1 causal latent steps | NOT-DECODER, SWAP-mid, LP5, CUE-ONLY at the claimed depth; CD-AN one-pass | d 3-4 only; CD-AN at d 3-4 (§237) | Not at d 6-7 (T2b unregistered) |
| C2 latent-specific extrapolation | same-data text CoT (§266 X3), plan-matched; pipeline (X5); one-pass (CD-AN / §271 DIRECT-SI on N1) | registered, not read | No: the text arm lacks the plan (F4) |
| C2 replication | 3 seeds, 2 lineages (Note 269-2) | §268 registered, bun only | of not covered (F3) |
| C6 hold | BUN-CTRL (same 8k without penalty) | yes, DRIFT in both (Note 259-4) | yes |
| C5 halt | plan-clock control (§264 ER / LFI / CY) | registered; Note 269-4 (b) already shows clock-equivalence | the halt must be framed as clock-equivalent until §264 |
| C3 compute | value-only text CoT at matched FLOPs (§266 descriptive) | analytic only (Note 266-1) | no efficiency claim, as the outline says |
| H2 public split | own-run CoT-SFT / No-CoT / Coconut / CODI, 3 seeds, zero-latent / noise arms | not run | fair if own-run on the same regenerated splits; SubLAT / PAR / Coconut published numbers are context only |

## Reviewer's most likely rejection reasons (in order)

1. "The plan is given at every pass. This is value transport, and an external one-hop pipeline does as well" (§266 X5 at P ~0.7; §260).
2. "Per-step supervision with simulator values. The model is told the intermediate answers" (C4 not met; Note 263-2 extends this to the co-training items).
3. "A toy templated mod-11 task with 6 frames on GPT-2 small. Frames 6-7 don't transfer (§249: 0.54-0.27); medium doesn't form the write (§258); no public task" (H2 not run).
4. "Two seeds, one lineage, with a between-seed SD of 0.09 at the depths claimed" (Note 269-1 / 269-2).
5. "The extrapolation is +2-3 hops. PAR (2610.27628) gets 16x on S5 from a retrofitted loop, and 2609.35643 shows causal depth-generalising latents."
6. "The text CoT baseline is under-trained and plan-free" (F4).
7. "The mechanism story changed between stages (count -> skip-rule -> content -> name)." This is fixable by F1's wording.

## Claims we can make today (scoped: plan given; per-step supervision; templated REG frames 0-5, mod 11; GPT-2 small; one lineage, 2 seeds; Wilson CIs printed)

1. **C1 causal battery at d 3-4** on bun and of: NOT-DECODER 0.020 / 0.019 (bun), SWAP4, LP5-D4, CUE-ONLY (Notes 259-4, 263-3). The decoder completes the last hop (Note 264-4).
2. **Distractor insertion breaks positional slot counting and is followed by +2 depths at >= 0.96**: dis d 6 0.981 / 0.961 vs rcb 0.346 / 0.368, under a surface skip-rule (Note 256-1). Literature NARROW.
3. **Content selection on held-out distractor families** (dch / of; Notes 257-3, 263-3), stopping at the asked entity (H-EXTEND 0.92 / 0.89 on dch).
4. **Op-order-free selection** on of: TEST-PERM d 4 0.982 / 0.961, LATE-PREMISE >= 0.979. Not full-shuffle free (d 4 0.46 / 0.38). No "first" (LIT_S263). Selection by name among computed entities, behavioural only (Note 263-3).
5. **Fixed-point hold to K' 128** at d 1-6, ||S|| 1.02-1.14x, attributed by BUN-CTRL drift (Note 259-4). Cited as the same idea as 2608.18222 / 2606.18206.
6. **Training-free r = 3 stability halt**, pre-registered fresh-draw PASS on bun (Note 259-12). It must be framed as equivalent in accuracy to the plan's clock (Notes 264-4, 269-4 (b)).
7. **What the plan buys**: plan-free C1 holds on clean d 1-4 (1/2 full pass), but selection follows reachability, and d 7 falls to 0.07-0.25 (Note 260-5).
8. **Late-pass errors are carried forward, not recomputed** (ERROR-PROPAGATION SERIAL / MIXED; Note 269-4 (a)), with first errors at untrained passes (Note 264-1). Descriptive.
9. **C3 not met, analytic**: 2.4-9.1x FLOPs of value-only text CoT; 1.2-1.6x fewer serial layers at K' = d (Note 266-1).
10. **C2 partial, descriptive**: d 6 0.96-0.97, d 7 0.85-0.86 (bun) or 0.90-0.91 (of, NEAR-SEED). Reported per seed and labelled "2 seeds, not claimed under the >= 3-seed rule".

## Claims blocked on (and what unblocks each)

| Blocked claim | Unblocked by |
|---|---|
| C2 as a claim (not descriptive); "replicates beyond one lineage" | **§268** LINEAGE-SEED (bun_s3) -> REPLICATES; for of, an added of-from-bun_s3 stage (F3, unregistered) |
| Extrapolation is latent-specific | **§266** X3 LATENT-SPECIFIC, plus a plan-given text arm (F4; needs a pre-output amendment) |
| C1 worded as more than "value transport under a given plan" | **§266** X5 not PIPELINE-SOLVES |
| C1 causal at the claimed depth (d 6-7) | **T2b** (unregistered; swap_kd / lp5h at d 6-7 + NOT-DECODER at d 6) |
| "Not pass budget" / "K' > K does not help" | **§264** STD K'-curve (S6) |
| C5 as the model's own halt (not plan-clocked) | **§264** ER / LFI / CY (S5) |
| Memory is necessary (not a pause effect) | **B1** PAUSE-PASS (unregistered) |
| Content selection on bun itself | **§269** insert_ho part (or use of, already binding in §263) |
| Late-pass failure cause (selection / memory / computation) | **§269** triage part (F9) |
| "Denser training lifts the horizon" | **§265** DENSE-DISTRACT |
| C1 under any statement order | **§270** FULL-PERM |
| A loop is necessary (non-solvable group) + one-pass control | **§271** N1 (DIRECT-SI binding) + the TH1 expressivity memo |
| Held-out structure (frames 6-7, unseen names) | **B2** (unregistered); note §249 already failed frame transfer |
| C4 (no hand-built per-pass targets) | **§267** EXACT-EM-11 (only C4-eligible route) or PAR-style self-distillation under Note 269-4 (d) |
| H2 public-split claim | **D1** ProsQA-OOD (unregistered; §191 model, plan-free) or §271 + a text arm |
| Any novelty sentence in the abstract / intro | a Literature paper-level novelty check (F10) |
