# 09 REDTEAM: independent review of 09_lineage_compression.md (2026-11-04)

Reviewer: independent red-team sub-agent, working read-only. No training, no computation on outputs. Reads were limited to args.json, train_summary.json and the first 1k / 2k bin records of log.jsonl in s233_askname_s1, aw_plaw2_s1, rcb_plaw4_s1, dis_s1, dch_s1, bun_s1, bridge_s1 and of_s1. Code was read in boxes/inject_aw.py, boxes/fsq120.py and boxes/deltamem.py. THEORY.md sections §268-§271 and Notes 258-9, 266-1 and 269-1/2 were also read.

## Verdict: GO-WITH-FIXES (BRIDGE2 + a ONE-RUN formation gate); NO-GO for STATE-DISTILL as written

The schedule-flag build is sound, and the code citations check out (see the end of this file). Three problems block registration as drafted:

- **Scoping.** Compressed recipes cannot stand in for §268 or count toward bun's or of's lineage rule.
- **Scoring.** The per-seed NEAR-SEED rule makes C2-6 unpassable by arithmetic.
- **DISTILL.** It trains the very quantity its FORM gates read, and it has no replication value.

## Findings

### F1 (HIGH): a compressed recipe is a different recipe, so its seeds are not lineage replicates of bun / of

Evidence:
- §268 defines the lineage as askname -> aw_plaw2 -> rcb -> dis -> dch -> bun, and says every 2/2 so far descends from aw_plaw2_s1 + askname_s1.
- BRIDGE2 stage 2 puts seeds 2-3 on the same aw_plaw2_s1 init (design §2.2). By §268's own definition, BRIDGE2's three seeds are therefore **one lineage**.
- DISTILL students inherit bun_s1's behaviour through the teacher, so they belong to bun_s1's lineage by construction.
- The design's "Why" paragraph frames compression as a way to make §268-style seeds cheaper. That is true only for future work built on the compressed recipe. It is never true for the bun / of scorecard.

Fix:
- Add a Scope line: "BRIDGE2 (or ONE-RUN) recipe; one lineage (askname_s1 / aw_plaw2_s1) unless a seed starts from §268's aw_plaw2_s3."
- BRIDGE2 seeds can never count toward Note 269-2's ">= 2 lineages" for bun or of.
- Re-assign seed 3 of the chosen arm to init from §268's aw_plaw2_s3, gated on §268's aw stage passing. That seed is the only route to a second lineage for the BRIDGE2 recipe.
- Any "BRIDGE2 ≈ bun" wording is an equivalence claim. It needs its own power statement (F2).

### F2 (HIGH): the C2 rows cannot pass as written, and the OC block is missing

Evidence:
- The design defines NEAR-SEED per seed as |acc - bar| < 0.09, with the C2-6 bar at 0.914.
- A seed clear of NEAR-SEED would need d6 >= 1.004, so **C2-6 can never pass**.
- At d7, a clear pass needs >= 0.895, above bun's own mean of 0.855.
- Suppose the compressed recipe is exactly as good as bun (sigma_seed 0.09). Then:
  - P(one seed >= bar) = Φ(0.05 / 0.09) = 0.71;
  - P(all 3 seeds >= bar) = 0.36;
  - P(all 3 clear the NEAR band) is about 0.04 at d7 and 0 at d6.
- Proper non-inferiority with margin δ 0.05 at n = 3 needs mean - 1.645 · 0.09 / √3 >= 0.914, so mean >= 0.999. That is unattainable. Margin δ 0.10 would need mean >= 0.949, which is feasible.
- The block labelled "OC (priors)" gives P(OK) guesses only. It does not give P(pass | no effect) and P(pass | effect) at the panel's seed count, which the 10-14 standing rule requires.
- "Restarts matter": the SE of a difference of two 3-seed means is 0.09 · √(2/3) = 0.073. The 0.15 bar is 2.0 SE, so P(false positive) ≈ 0.02 one-sided and the power at a true 0.15 difference is ≈ 0.5. State both.

Fix:
- Bind C2 on the 3-seed mean margin, with the NEAR band set to sigma_seed / √n = 0.052. Report per-seed margins descriptively.
- Either pre-state δ = 0.10, or label equivalence-to-bun as descriptive (NOT TESTABLE at 3 seeds).
- Add a full OC table: P(OK | equal to bun), P(OK | 0.10 worse), and the A-vs-B false-positive rate and power above.
- Also report the recipe-local sigma (bun_s1 / s2 spread 0.016 at d6). The 0.090 pools mixed recipes.

### F3 (HIGH): DISTILL's KL term trains the readout its FORM gates measure, and it adds no lineage value

Evidence:
- **The KL target is the WS readout itself.** L_kl uses z = colon_logits(h, pos)[:, NUM_IDS] at each pass's ':'. That is the WS-READOUT: the same function the training log and eval_ws use (inject_aw.py 61-63 and 383-391, compared to values[min(t, d)]). FORM-2, FORM-4 and WS-DEEP on DISTILL arms are therefore trained readings (memory: trained readings are artefacts; dynamics targets get gamed).
- **The teacher's targets are hand-built.** Teacher accuracy is 1.000 at d 1-4, so at τ 2 its per-pass soft targets are close to one-hot step values. The teacher learned those values from simulator-built aux targets (aux_target value, aux_step 1.0 in every stage's args). DISTILL-S is therefore "per-pass write-token value targets by proxy", close to a §251-style hand-target arm. The design correctly says it is not C4 (NORTH_STAR C4: no simulator per-step states; Note 266-1).
- **The §236 0/2 comparison carries no evidence.** It sets a run with direct per-pass write supervision against one without it.
- **No replication value.** The design says the "Compresses replication (seeds, ports)". F1 removes the seeds half.
- **The ports half is already decided.** DISTILL-M's condition ("only if §258 S2b reads NO-SEED") was met on 11-02: Note 258-9 scored S2b NO-SEED and the medium line PARKED. The text is stale. A medium port is a separate C2/port registration, not lineage compression.
- **Teacher choice.** bun_s1 matches the DISTILL-S student's data (no perm / ask). of_s1 is the SOTA, but it would need the student to train on perm_p 0.5 + ask_p 0.25 too. Do not mix them.
- **Build traps** (if DISTILL is ever built):
  - (a) The teacher must compute its own h_low. Passing the student's `hl` cache gives the teacher the student's lower blocks.
  - (b) Assert the teacher's mem.fade == 1 and gamma == 0 after load. fade is a persistent buffer (deltamem.py:73), and the loop's `fade.fill_` touches only the student.
  - (c) The new `elif DIST:` sits in a chain with `elif a.perm_p:` (line 383). A teacher-of student with perm on would silently skip one branch.
  - (d) L_state assumes the student's fresh DeltaMem (qkv = askname block-6 c_attn) shares a basis with the teacher's after 68k steps. That is a CONJECTURE and should be tagged as one.

Fix: remove DISTILL from this design. If it is revived:
- gate it only on causal / behavioural rows (SWAP4, LP5, d 6-7 accuracy, C1 at d 6-7), with WS rows labelled TRAINED-READING;
- add a teacher-free control: CE of colon_logits at every pass to values[min(t, d)]. If DISTILL ≈ that control, the teacher adds nothing;
- re-register DISTILL-M as a port design on its own.

### F4 (MED): the lineage replays the same item stream, and compression changes that too

Evidence:
- The ihash at the 1k and 2k bins is identical (c83bb39bb21be701 / cb6f31f27876909e) in rcb_plaw4_s1, dis_s1, dch_s1, bun_s1 and bridge_s1. of_s1 differs because of perm / ask.
- Cause: every stage rebuilds `OW.Stream(OW.SEED_TRAIN, ...)` (line 280, a seed that does not depend on a.seed) and `rng_d = default_rng(seed + SEED_D)` (line 271), all with seed 1.
- The insert coins are keyed on (seed, step, k) (line 325). So bun's 8k batches are dch's 8k batches with the same §257 inserts, replayed with the hold penalty.
- The first 8k d<=4 base items are seen four times along the lineage, and steps 8-16k twice.
- BRIDGE2 and ONE-RUN draw a continuous stream instead, with no replay. This is a further recipe difference beyond LR, AdamW and fade.

Fix:
- Add "data replay (rng / stream reseed per stage)" to the design's §8 item 6 list of things a restart does not reproduce.
- Read ONE-RUN arm B as "LR + moments", never as "the lineage's restarts".
- Optional, not needed now: a `--reseed_at_restart` flag would let arm B reproduce the replay.

### F5 (MED): formation risk under one cosine, and the cheapest decisive gate

Evidence:
- **LR at onset.** aw_plaw2_s1 formed at t* 7k (Note 269-1). Its LR factor there was 0.5(1 + cos(π · 7/12)) = 0.37. ONE-RUN arm A at 7k runs at 0.5(1 + cos(π · 7/36)) = 0.91. Formation in the only formed d<=2 recipe happened during decay, so arm A tests a high-LR P-2. A plain NO-FORM is plausible (FD1: 3/15 formed overall).
- **Arm B's P-2 is nearly aw_plaw2_s1 again.** Its first segment, lr_at(step, 12000), is exactly aw_plaw2's schedule. With seed 1, the d-sequence (rng_d), the item stream (SEED_TRAIN; EXCL comes from build_sets and does not depend on d_max, line 277) and the LR all match aw_plaw2_s1. Only three things differ:
  - g_add calibration: d = 1 + i % a.d_max over d 1-4 rather than 1-2 (line 533);
  - micro 2 vs 1;
  - hold_log, which takes no gradient.
- So arm B's phase 1 adds nothing new. Arm A seed 1 is a **paired** test against aw_plaw2_s1 on identical data, where only the LR (plus micro and g_add) differs.
- **Arm B is not the lineage in one process.** Its phase lengths (12 / 8 / 8 / 8) drop rcb's no-insert d<=4 stage and halve dis.

Fix: run the cheapest decisive gate first.
- **The gate:** ONE-RUN arm A seed 1, `--steps 36000 --stop_at 12000` (~1.7-2 h, ~$0.45 with overhead).
- **Read** log ws_readout['2']['2'] per 1k bin at 4k-12k against aw_plaw2_s1's bins (>= 4k marker per FD1). Bind FORM-2 at 12k.
- **If A is NO-FORM:** run one conditional arm-B-to-12k (~$0.45) to separate LR from micro / g_add. Its expected outcome is FORM, as aw_plaw2_s1.
- **If A forms:** resume to 20k under the same `--steps` and read FORM-4 ws_readout['4'] passes 3-4 at >= 4k after the switch (16k), binding at 20k.
- Run ONE-RUN arm B past 12k only if BRIDGE2 and arm A leave a question it alone answers.

### F6 (MED): resume-exactness and the restart gate are mis-specified

Evidence:
- **Gate 3 does not compute what it says.** The design §4 gate 3 says that under `--lr_restarts "201" --stop_at 400` the logged lr equals lr_at(step - 200, 200). The last segment's length is `steps - r_k + 1`, not 200, unless `--steps 400`. With stop_at alone, steps stays at its default.
- **The alignment rule is not what makes resume correct.** The `k * ckpt_every + 1` rule is harmless but not the reason resume is right. Resume is schedule-exact because the opt state is checkpointed (lines 515, 659) and every schedule is a pure function of `step`.
- **The bug a resume can hit is a double reset.** A resume from a ckpt inside a segment must not reset again.
- **"Exact" overstates it.** Resumes in this codebase are not bitwise (Note 190-1). The open 1k bins (wsbin / acbin / askbin) are not checkpointed. Claim "schedule-exact", not "replays exactly".
- **Implementation order.** The reset has to happen before the `for gr in opt.param_groups` lr loop (line 545). The re-bound `opt` has to be the object saved at line 659.
- **Gate 1 must run on a GPU host.** The CPU offbit is not bitwise on the local box (CURRENT_TASKS, 11-03).

Fix:
- Gate 3: `--steps 400 --lr_restarts 201`. Assert fp64 lr, `opt.state == {}` right after the step-201 reset, and that a resume from ckpt 300 does not reset (opt.state non-empty at 301).
- Gate 1: run on the host with the deterministic wrap.

### F7 (MED): math backing and the OC block are missing

Evidence: the design changes the schedule, and DISTILL changes the loss. Under the user's 10-13 rule, that requires a derivation section with assumptions, mechanism, tagged claims and predictions. The design has none.

Fix: add the section. Examples:
- DERIVED-UNDER-ASSUMPTIONS: step-pure schedules + a checkpointed opt imply schedule-exact resume.
- DATA: ihash replay (F4); LR factor at onset 0.37 vs 0.91 (F5).
- CONJECTURE: re-warm or moment reset aids pass-3/4 formation at the d 2 -> 4 switch (the swamp hypothesis, Note 236-5); replay aids RETAIN.
- Predictions: if arm A forms, the onset t* shifts later than 7k; gn_preclip_by_d spikes at 12k in arm A and not in arm B.

Pair this with the F2 OC table and the conservative censoring line. The existing NO-FORM = result rule is fine.

### F8 (MED): BRIDGE2 compresses toward a superseded model

Evidence:
- The SOTA is of_s1 / of_s2 (bun + 8k perm 0.5 + ask 0.25; Note 263-3).
- §270 FULL-PERM builds on the §263 recipe.
- BRIDGE2 ends at a bun-equivalent and is scored against bun's bars.

Fix:
- Append the of stage (8k, of_s1's args as args_ref, init = the BRIDGE2 final; ~$0.8). The output can then be scored against of_s1 / of_s2 (Note 269-6: d6 0.971 / 0.959) and serve as the cheap base for later C1 work.
- Recipe = aw 12k + 16k + 8k = 36k after askname, vs 76k.

### F9 (MED): the savings are overstated

Evidence (s/step from train_summary.json):

| run | s/step |
|---|---|
| aw | 0.50 |
| rcb | 0.87 |
| dis | 1.08 |
| dch | 1.20 |
| bun | 1.65 |
| bridge | 1.18 |
| of | 1.69 |

- Full lineage after askname: about 16.7 h, about $3.5.
- BRIDGE2 after askname: about 1.7 + 6.5 h = 8.2 h, about $1.7.
- That is ~2x, not 2.5x.
- Within-lineage seeds (the bun_s2 pattern) cost only the last stage (~$0.8). Compression saves money only on independent lineages, and Note 269-2 needs just one more of those.

Fix: state the saving as "~$1.8 and ~8 h per independent lineage". Rank the work by what it unblocks, not by $.

### F10 (LOW): missing standing-rule items

- C1 causal gates at d 6-7 (SWAP / LP5) are required when C2 and CONTENT (C1 scope) are claimed together (11-03 rule 3). The design has SWAP4 / LP5-D4 only.
- The "one lineage" scope line is missing.
- The K' curve is listed as descriptive; that is fine.

### F11 (LOW): schedule and cost details

- **hold_log cost.** hold_log from step 1 adds ~0.3-0.45 s/step (bun 1.65 vs dch 1.20) in phases with no penalty. That is justified in BRIDGE2 1-8k as the no-penalty window. It is wasted in ONE-RUN P-2: disclose it, or start hold_log only for ONE-RUN runs past 12k (by resume with the flag added, which is allowed since it takes no gradient; the args diff must say so).
- **BRIDGE2 1-8k is mostly a replay.** It nearly replays bridge_s1's first 8k (same init, seed, LR schedule and ihash). That is fine as the in-run control, but it adds little else.
- **Hold under low LR.** BRIDGE2-B's hold runs at LR 0.5 -> 0, while bun's ran at full LR from step 500. ONE-RUN arm A's hold runs at 0.12 -> 0. C6 rows can fail from LR alone; disclose this as a confound of arm A vs B, alongside "restart vs LR level".

### F12 (LOW): stale text

- DISTILL-M's "only if §258 S2b reads NO-SEED" has been met since 11-02 (F3).
- The design cites §258 S2 / medium as a live route, but the medium line is parked (Note 258-9).

## Code claims verified

All verified:

| claim | location | status |
|---|---|---|
| distract coin | inject_aw.py:325 | as stated |
| distract_mix / distract call | 326 | as stated |
| resume block | 514-528 (opt / rng / stream states loaded) | as stated |
| d draw `rng_d.integers(1, a.d_max + 1)` | 552 | as stated |
| lr loop | 545-546 | as stated |
| fade `min(1, step / int(500 sc))` | 549 | as stated |
| `inc_on = hold_inc > 0 and step >= hold_start * sc` | 558 | as stated |
| make_opt | 508 | as stated |
| init_model sha + strict load | 292-295 | as stated |
| lam_at | 299-302 | as stated |
| MIXT assert | 312 | as stated |
| ckpt | 655-663 | as stated |
| lr_at | fsq120.py:740-743 | = base · mult · min(1, step/100) · 0.5(1 + cos(π step / steps)); warm_from unused here |

Verified from the args.json files: 8k + 12k + 16k + 16k + 8k + 8k = 68k steps; all stages lr 1e-4, seed 1, aux_fade 48000,48000.

Unverified (out of scope): "+35% teacher cost".

## Priority

- **Where LC sits.** LC is C2 infrastructure. The user's order is C1 > C6 > C4 = C2. §270 (C1) and §271 (C1 / C2, new family) are registered, and §268 is queued.
- **§268 goes first.** Its aw_plaw2_s3 is the input that gives any compressed recipe a second lineage, and a BREAKS-AT-aw result would show that every compressed recipe inherits the same fragile P-2.
- **LC stage 1 is a filler.** It is cheap (< $3) and can run as a one-at-a-time filler once §270 / §271's first cells are admitted. It should not jump them.

## Minimal staged plan I would approve

**Stage 0 ($0).**
- Build the four schedule flags only (no distill code).
- Gates 1-2, gate 3 as fixed in F6, gates 6-7, and the plain-path smoke, all on host.
- Add the math-backing section (F7), the OC table (F2), the scope line (F1) and the d 6-7 C1 parts (F10).

**Stage 1 (guard $4).** Two independent cells, one filler at a time:
- **(a)** BRIDGE2 arm B seed 1 from aw_plaw2_s1_init, 16k (~$1.6), then the of stage, 8k (~$0.8). Score against of_s1 / of_s2, descriptive at n = 1.
- **(b)** ONE-RUN arm A seed 1, `--steps 36000 --stop_at 12000` (~$0.45). FORM-2 at 12k is binding and paired with aw_plaw2_s1's bins.
- **(b')** Only if (b) is NO-FORM: arm B to 12k (~$0.45).

**Stage 2 (after Audit scores stage 1 and §268's aw stage lands; guard $6).**
- BRIDGE2-B+of seed 2 on aw_plaw2_s1 and seed 3 on aw_plaw2_s3, binding on the 3-seed mean with the 0.052 NEAR band and a pre-stated δ.
- If (b) formed: arm A to 20k (FORM-4), then to 36k.
- Otherwise ONE-RUN closes with "single cosine blocks P-2 formation (1 paired seed)".

**Not registered:** STATE-DISTILL (F3), and ONE-RUN arm B past 12k (F4 / F5).
