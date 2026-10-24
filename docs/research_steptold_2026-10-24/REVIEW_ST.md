# REVIEW_ST: red team of ST_draft.md v0 (ec88980f) (2026-10-24 12:30 EDT)

Reviewer: Theory's red-team subagent. Read-only on the repo, apart from this file and `review_scripts/` (CPU, 1 thread, idle priority, $0). No blinded file read; no model loaded.
- **Inputs:** ST_draft.md and its four scripts / three JSONs; THEORY Notes 226-2, 227-2, 228-2, 229-1, 231-2, 231-3 (grep only); LAT_v1.md; REVIEW_LB #1; boxes/inject_ow.py (tok_ow, aux_logits, loss_items, main), boxes/ow_words.py, boxes/inject_ow_eval.py (tokd); NORTH_STAR C1 / C4; Literature's LIT_STEPTOLD_2026-10-24.md (0033ec25).
- **Check:** `review_scripts/rv_st_checks.py` → `.json`: (A) cue length over all 40 names × 10 units; (B) the P1-ASK target vs structure; (C) the Stage-1 cost and wall clock, one host vs two; (D) the CUE-ONLY guard as written.
- **Reproduced:** st_oc's P-2 / P-4 / Stage-1 host-hours and $, the guards (1.16 / 2.82 / 14.78), the family total of 18.76, the expected spend of $1.99 and P(C1-TOLD) 0.042. The OC values follow from the code (LP2 with N_eff 495 is conservative; LP5's pairing is conservative). st_cue_check C1–C3 / C6 and st_told_zs's pipeline read correctly.

## Verdict: GO to v1 with fixes 1–7. No new red team is needed if they are applied as written.
- The leak proofs hold, CUED-DIRECT is the right binding comparator, and R3 / R2′ still mean serial use with the cue kept.
- But the $0 gate does not test A1 as stated (issue 1), D2 misdescribes §228 (issue 2), and the CUE-ONLY guard as written VOIDs every pilot (issue 3).
- **STOP condition (new):** if the combined P1-ASK + P1-NAME run fails its pre-stated rule, ST stops at $0.

## Issues, ranked by severity

**1. [HIGH] P1-ASK cannot test A1 (name-conditioned computation). Its target is always v_1, which structure alone identifies.**
- In every P1-ASK item (d ∈ {1, 2}) the target S_1 is the unique computed entity whose inputs are all given. Check B: 4,000 / 4,000 items; by generator construction, add_s needs t ≥ 2 and mul_s needs t ≥ 3. So a model can pass P1-ASK by ignoring the cue: "compute the first node". This is the same name-structure redundancy as F5, only flipped.
- That is a valid test of LAT-ST's **pass-1** job (the pass-1 cue is always S_1). It says nothing about pass t ≥ 2, where the named S_t is not the first node, and A1 is exactly the claim that the name is read.
- **The bars:** d = 2 ≥ 0.70 sits below LP2's 0.80. Yet P1-ASK is the easier setting: 12 blocks, direct answer CE at the read position, no memory bottleneck. Reaching 0.70–0.79 there predicts an LP2 fail.
- **Fix: one combined local run, P1-ASK+NAME (same ≈ 30–60 min on the 4060, $0).** Per item, the asked entity X (cue appended, APP form) is drawn uniformly from four classes:
  - (i) S_1 (one hop; the drafted P1-ASK class);
  - (ii) S_0 (given);
  - (iii) a distractor, or G_t, when present (given; otherwise S_0);
  - (iv) d = 2 only, **S_2 with S_1's defining sentence replaced by GIVEN "S_1 has v_1 u."** (one hop, but not the first node: pass t ≥ 2's job in text form).
  - Target: v_X.
  - **PASS iff (i) ≥ 0.80 and (iv) ≥ 0.80 and (ii) ∪ (iii) ≥ 0.90, read only if d = 1 (i) ≥ 0.80.** UNDERTRAINED → one 16k run, same rule. FAIL → ST stops at $0 and goes to the user.
  - Report (i) alone under the drafted 0.70 / 0.80 rule as descriptive.
  - Build: locate S_1's sentence as st_cue_check C6 does (contains S_1 and S_0, not S_2, not "How many"); assert exactly one match; G-D on 20 decoded fixtures.

**2. [HIGH] D2 (and N2) misdescribe §228: pass 1 *was* asked for v_1, through the aux. ST changes only the input side.**
- In §228 the pass-1 aux target is always R_step[1] = "<S_1> has <v_1> <u>.". It is teacher-forced, so the reader is given S_1's name before it predicts v_1 (inject_ow tok_ow / aux_logits). At pass 1 the target never depends on d.
- §228's pass 1 therefore had a fixed, supervised target that structure identifies. The aux gradient and the d = 2 answer gradient both pushed for v_1 in S_1, and the result was 0.25 on d = 2.
- What §228 lacked was an **input-side pointer** to S_1 (and to S_t at t ≥ 2). ST adds exactly that and nothing else: the supervision path (write → S_1 → aux read) is unchanged.
- **Consequences:**
  - The prediction's argument "under ST the aux and answer gradients agree, unlike §228" is wrong for d = 2: they agreed in §228 too.
  - The mechanism ST tests is "a name pointer makes the write learnable", not "being asked".
  - F4 (the trunks ignore names) makes the pointer an untrained feature. So P(LP2 | P1 pass) = 0.35 needs a justification beyond §0's rule, or lowering (I would put it at 0.2–0.3).
- **Fix:**
  - Rewrite D2 as "the pass-1 target was supervised but unlocated; the cue supplies the location".
  - Drop the "gradients now agree" sentence.
  - Reword N2 to "no pass is **cued** for the intermediate".
  - State in A1 that the pointer must be learned through the aux-through-memory path. P1-NAME tests it only through direct CE, so LP2 remains the binding A1 test in the real path.

**3. [HIGH] The CUE-ONLY guard as written VOIDs every pilot.**
- The guard reads answer accuracy "on TEST2 at the pilot's depths" with the decoder unchanged. The decoder sees the full item text, and LAT answers d = 1 from text alone (228-2: d = 1 1.000 without the memory).
- With d = 1 pooled in, the answer reading is ≈ 0.55 > 0.16, so LEAK fires with P = 1.0 (check D). The OC (false alarm 0.000) assumed a majority-level answer.
- **Fix:**
  - The CUE-ONLY answer is read on d ≥ 2 only (P-2: d = 2; P-4: d 3–4; Stage 1: band depths).
  - Its bar is ≤ max(majority, the same-items plain-DIRECT acc) + 0.05.
  - The aux readings stay as written (the aux prompt carries no text).
  - Also say what the guard is for: the cue carries no value by proof (C1–C3), so CUE-ONLY catches build faults (a cue function reading item values; a cache keyed wrongly), not cue information. Add G2b: the cached lower rows of item i equal the uncached rows of item i alone (batch-of-one, fp64). That catches cross-item cache reuse, which CUE-ONLY cannot see.

**4. [MEDIUM] LP5 / R2′ can miss serial use through self-repair in the hold passes, and P-2 has no all-pass MEAN reading.**
- At d = 2 with K = 8, MEAN-AT-1 leaves six later passes. A serial model that learned to recover (recompute v_1, then v_2, over passes 2..8) shows no drop. It would be labelled TOLD-RECOMPUTE although it uses the state serially.
- Conversely, aux multi-task training can make the shared trunk a better one-pass solver (the §207 route). MEAN-AT-1 does not separate that from serial use; the pooled MEAN-STATE does, and it appears only at Stage 1.
- **Fix:**
  - Read LP5 at K′ = d (= 2: no room to repair) as binding, with K′ 8 descriptive. Pre-state "drop at K′ = d but not at K′ 8 → SERIAL-REPAIR (counts as serial; flagged)".
  - Add the pooled MEAN-STATE (S := the TEST2 mean of S_8 at every pass) to P-2 as binding: acc_MEAN ≤ CD + 0.10. The eval cost is ≈ 0.
  - Add both to st_oc as named truths (self-repair; trunk-improved one-pass).

**5. [MEDIUM] No reading tells whether the cue is used. A pass with the cue ignored would be mislabelled C1-TOLD.**
- If LAT-ST learns structurally (the cue ignored), the result is plain C1 (stronger), but it would be filed as C1-TOLD. If the cue is read, the label is right. Neither CUE-SKIP nor CUE-FROZEN is binding, and neither is read at P-2.
- **Fix:** add CUE-WRONG at P-2 and Stage 1, both descriptive but deciding the label. At pass t the cue names a different spine entity of the same item (S_{t+1} where t < d, else S_0).
  - acc drop ≥ 0.30 → **C1-TOLD (cue-used)**.
  - drop < 0.10 → **C1 (cue-unused)**, reported as plain C1 on this band.
  - In between → TOLD-PARTIAL.

**6. [MEDIUM] Stage 1 should run on two hosts, with a host-speed gate. The 35 h estimate is the one-host serial sum; LAT-ST alone is 26–34 h.**
- Check C: LAT-ST trains for 26.4–34.0 h. CD + DIRECT + their eval take 5.6 h. On two 3090s (same class), the wall clock is 28.9–36.5 h vs 34.5–42.1 h; host-hours and $ are unchanged ($11.37 high).
- 3090 hosts differ ≈ 2× on this launch-bound loop (memory: host speed variance). A slow host makes LAT-ST 58–73 h, past the ×1.3 guard.
- The LAT run cannot be split, so parallelism helps only by moving the comparators off its host. The one-GPU-class rule allows that: §228 / LAT_v1 bind the class, not the host.
- **Fix:**
  - Stage 1 = host A (LAT-ST) + host B (CD, then plain DIRECT), both RTX 3090.
  - HOSTPROBE gate: time 300 LAT-ST steps at d = 8 before the run; if the projected train time is > 40 h, destroy and redeploy (≤ 2 tries).
  - Off-host full-state checkpoints every 2k with a tested resume, and guard marks to step 48,000 (the 10-15 / 10-21 rules).
  - Apply the same split to P-4 (6.5–8 h → ≈ 5.5–7 h wall).
  - The cap stays at $19.

**7. [MEDIUM] The ZS slip: the reading needs re-labelling, and the ASK0 interpretation is wrong.**
- **Pipeline validity:** ASK0 is not a pipeline positive control. The pipeline is shown valid by D1 and ORIG, which reproduce the closed runs: DIRECT d = 1 0.648 vs 0.623; ORIG v_2 0.094 vs 0.093; LAT aux d = 1 0.753 vs 0.77. ASK0 at chance is a *finding*: on d = 2 items the trunk cannot even copy a written value when asked. That strengthens F4; the readout is not unreadable.
- **What else is needed, beyond the disclosure:**
  - (a) Label ZS **DESCRIPTIVE (rule mis-set; no zero-shot cue effect)**, not WEAK. It gates nothing, because P1-ASK+NAME is binding. Keep APP, and set the P(LP2) prior from issue 2's argument, not from the WEAK branch.
  - (b) A THEORY Note line recording the slip and that ec88980f now commits the P1 rule before its run. Amend that rule per fix 1 and commit it, with the wrapper's sha, **before** Experiments launches.
  - (c) Re-set st_told_zs's NULL reference to ORIG for the record (descriptive only).
  - (d) Literature's N2 wording must not cite ZS as "trunks ignore the question" without "trained §228 trunks, d = 2 items".
  - No re-run is needed.

**8. [LOW] Exactness of the per-distinct-cue cache (open Q6): exact, with one build assert.**
- Every cue is exactly 7 tokens, and every name is 1 token (check A: one length over 400 name × unit pairs).
- Per item, the pass-t inputs differ only at the name token (position L_i + 5), so right-padding is identical across passes.
- **Fix:**
  - Assert the cue length is 7 in G-CUE.
  - Optional saving: cache the text rows once and recompute only the 7 cue rows per distinct cue. Exact under causal attention with padding after the cue. Lower work drops from d× to ≈ 1×, and the d = 8 factor from 2.76 to ≈ 2.4. Gate it with G2's fp64 test.

**9. [LOW] Cheaper ordering: run CD d ≤ 2 locally before the cloud pilot.**
- st_pcd2_s1 costs ≈ 13 min on a 3090 (≈ 30 min on the 4060). If the plan alone lets one pass solve d = 2 (CUE-ONE-PASS: CD d = 2 ≥ 0.50 with d = 1 ≥ 0.80), ST ends at d = 2 at $0.
- Prior evidence says CD will fail. In the forward layout the text order *is* the plan, and one pass still failed d = 2 (226-1: pooled ≤ 0.57). Say that in §5.
- **Fix:** local CD as a $0 pre-gate (descriptive band only). The cloud CD stays the binding LP3 comparator (one GPU class with LAT-ST).

**10. [LOW] Pilot length (open Q8).**
- §227 needed ≈ 6k post-introduction steps for d = 2 even in text, and ST's pointer is an untrained feature (F4).
- **Fix:** either run P-2 at 12k (+≈ $0.35; plain DIRECT then fresh, +$0.07), or keep 8k with a pre-stated LP2-STILL-RISING rule. Under that rule, a last bin in [0.55, 0.80) with an OLS slope over the last 4 d = 2 bins > +0.02 / bin allows one fresh 16k re-run (no resume: the lr schedule changes). Decide before the commit.

**11. [LOW] Build-gate additions.**
- G9: decode 3 fixtures per d (1, 2, 4, 8) and diff the decoded think_t / CD dec strings against the spec, including the hold at t > d.
- G10: the new `aux_vacc_by_d_pass`, pooled over d, equals `aux_vacc_by_pass` on a 50-step run.
- G6 for CD: an args diff vs s228_pdir2c_s1 of exactly {cue = all}.
- A wrapper note: st_ask_local's args.json "format" string still reads plain DIRECT. Record the true input in ask_meta.json (already done) and have the scorer read ask_meta.

## Answers to the brief, one line each
1. **D1–D4 / A1:** D1 follows from A1–A4. D2 is wrong as stated (issue 2). A3, a 6-block op after the block-6 read, is untested on language and read only at LP3. P1-ASK is the right *kind* of first test but tests pass 1 only. Use P1-ASK+NAME with 0.80 bars (issue 1).
2. **Leak / lookup:** the proofs hold (C1–C3 reproduce; the cue is skeleton-only and has constant length). CUE-ONLY is mis-specified (issue 3). Can CD plus the cue solve d ≥ 2 one-pass? Unlikely (226-1's forward layout already gives the plan by order), and if it does, LP3 / CUE-ONE-PASS catch it. The loop is still needed wherever CD is at chance.
3. **C1 meaning:**
   - R3 with the cue kept still shows serial use: donors have identical cues by C2, so following cf(a) ∪ cf_b needs the state.
   - LP5 / R2′ catch recompute but can miss self-repair, and P-2 needs the pooled MEAN-STATE (issue 4); add CUE-WRONG (issue 5).
   - C1-TOLD is honest if worded "C1, scaffolded (plan given by the generator)". It meets C1's test form but adds nothing to C4. Against the user's operative standard (latent vs text), ST gets the plan free, so any C3 comparison must give TEXT the same plan.
4. **OC / cost:** all reproduce. Use two hosts and the HOSTPROBE gate for Stage 1 (issue 6). Cap $19 unchanged; expected spend ≈ $2.
5. **Slip:** re-label ZS descriptive. Commit the amended P1 rule before the run (issue 7).
6. **Diffs / gates / alternatives:** issues 8, 9 and 11. The simplest informative sequence is P1-ASK+NAME and local CD ($0), then P-2.

## Novelty (Literature, LIT_STEPTOLD_2026-10-24.md, 0033ec25; folded in)
- **N1: OK, narrow.** Word it as "a latent-loop analogue of decomposition prompting", scoped to C1-TOLD.
  - Text-side prior: least-to-most 2205.10625, self-ask 2210.03350, successive prompting 2212.04092, planning tokens 2310.05707.
  - Literature found no latent loop driven by step cues or a given plan. CUED-DIRECT is the key design point.
- **N2: NARROW.** Kin: Biran 2406.12775 (the bridge entity is resolved too late), Yang 2402.16837, and Press's compositionality gap (2210.03350). Reword per issue 2 ("not cued", not "not asked").
- **N3: NARROW.** Mean ablation is standard, and the swaps follow SCIT 2608.27265 / §221. The per-pass placement with the cue held fixed is a design refinement.
- No "new" / "first" anywhere.
