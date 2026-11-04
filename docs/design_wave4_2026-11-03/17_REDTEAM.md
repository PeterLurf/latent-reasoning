Status: red team of 17_hrm_trm_transplant.md (DSX), 2026-11-04, independent sub-agent; read-only (code + of_s1/args.json only; one CPU toy in the scratchpad)

# 17 RED TEAM: DSX detached deep supervision from of_s1

## Verdict: GO-WITH-FIXES (Stage 1 only, restructured; no C2 claim wording until F1-F4 are fixed)

The experiment is cheap ($2-3 for Stage 1) and asks a real, untested question: can answer-only training at d 5-8 on an in-basin of model move the horizon past K = 8? But it is mis-labelled. Under the gradient algebra, DSX is "answer-only depth extension (§251 AEX) applied to of_s1, trained one hop at a time on the model's own states". The HRM / TRM transplant adds only two things: the answer-hold terms at t > d, and truncation to TBPTT-1. The design has no arm that isolates them from plain answer-only extension (F3). Its binding C2 row can also be met without any pass past K computing (F2). Fix those, plus the micro pin (F5), and Stage 1 is worth its cost.

## Code citations checked (2026-11-04 tree)

| Claim | Verdict |
|---|---|
| inject_aw.py 739 lines; pass_aw 47-56, read_fixed at mem.layer (52) | correct |
| hold_step 159-199; line 170 `den = state_n(Ssv[d + 2])` raises IndexError for d >= 7 | **correct.** Ssv = [S_0] + K appends = 9 entries (403, 420-421), so d = 7 gives index 9. H (168) = range(d+3, K+1) is empty for d >= 6. |
| The far pass is unaffected | correct (185-198 compute their own den at t = d + 2; T_far - 1 >= 15) |
| loss_reg 394-452; aux gated by aux_d_max at 427 | correct. `(not a.aux_d_max or d <= aux_d_max)`: of_s1's aux_d_max 0 means aux on all items, so 4 is identical on d <= 4. |
| dec_logits inject_ow.py 86-90 via deltas5.fwd_read 91-98 (same read_fixed at block 6) | correct. **The decoder is the full shared GPT-2 trunk** (fwd_read runs all g.L blocks), so "decoder-side" and "loop-side" share every trunk parameter (see F1). |
| aux_logits inject_ow.py 92-103 (text-free prompt, t <= d) | correct |
| d draw at 588 `rng_d.integers(1, a.d_max + 1)` | correct; the §4.1 law with ext_p 0 replays it (G1b claim holds) |
| ws_readout (135-151): the ':' LM argmax at pass t vs v_min(t, d) | correct. It is a readout of h11[':'], which DSX never trains directly (only the ask items train the ':' head). |
| mem_train 488-494 measures only MC_TR = d_max items | correct. This is the root of F5. |
| d_max 8 changes EXCL (reg_excl, 281, since mul_s_p != 1), TP_TEXTS (317), WS_ITEMS (486) and MC_TR (480-484) | correct. G1b's "disclose excl_n" covers it. Also: **mem_train's hold_step call on d_max 8 items hits the IndexError for CONT too**, so the line-170 fix is needed by every arm, including the "flags-off" control. |
| of_s1 args: d_max 4, aux_d_max 0, distract_nmax default 2, hold_far_n 16 / 16-40, ask_p 0.25, init bun_s1_init | correct |
| insert name budget at d 8 + 2 inserts | handled (insert_render._budget caps n and logs it, 170-229) |

Toy check: C:\Users\peter\AppData\Local\Temp\claude\D--\ffe8dd4f-366e-5d47-92b5-bf5f7ac71716\scratchpad\dsx_toy.py. It uses a shared-weight additive recurrence and a decoder with a text path.
- (i) The DSX gradient of a depth-d item equals, to 1e-7, the sum of independent one-pass gradients taken at the frozen on-policy states S_{t-1}, t ∈ T_d.
- (ii) Under DSF, ‖∂L/∂S_s‖ > 0 for every s < d; it decays from 6.9 to 1.2 at d 8.
- (iii) The decoder's text path gets gradient from every ℓ_t.

## Findings

**F1 (HIGH): gradient flow. On a deep item, DSX trains only "one hop from the given state" plus decoder reading. The intermediate-pass computation on deep items, which is exactly the X4a failure cell, is never trained by that item.**
- Evidence: §5 algebra and the toy (i). For a d 8 item, the only grad-carrying pass is pass 8. Passes 1..7 run under no_grad.
- X4a's failure is pass 7 *at d 8* (0.59 / 0.80 vs 0.91 at d 7). Under DSX, pass 7 is trained only as the *last* hop of d 7 items, i.e. with at most 7 + 2 op sentences, and never as a non-final pass whose write must be read by pass 8.
- What does train "pass j at depth j": the shared weights, through the mixture d ~ U{5..8}. So DSX trains each of passes 5-8 in its last-hop role, on its own state distribution (on-policy, after j - 1 writes). That last part is the genuinely new lever versus §265, which never shows a pass a state that has had 5-7 writes.
- Two consequences:
  - (a) Pass j's write is trained to be decoder-readable, not next-pass-readable. They share read_fixed but use different queries: the decoder input is text + ' The answer is:' with no cue, while pass j + 1 has the cue S_{j+1}. §2 M2's "holds" is a CONJECTURE, not DERIVED.
  - (b) At t = d the loss is satisfied equally by pass d writing v_d or by the decoder hopping v_{d-1} → v_d from text (DECODER-LAST-HOP = 1.000, Note 264-4). Nothing in L_ext prefers the loop route, and the decoder runs on the same trunk.
- So yes: DSX can raise deep answer accuracy partly through the decoder, without the loop computing more.
- The Stage-1a WS gate (pass-8 ':' readout at d 8) is the right mechanism check. Keep it, and do not let accuracy fields substitute for it anywhere (see F2).
- **Fix:**
  - Relabel §2 M2 as CONJECTURE.
  - Add `ds_ws_by_d_pass`, the training-batch ':' readout at pass d on extension items. It is log-only.
  - Make the Stage 1b IN-DEPTH WS clause (eval_ws_deep d 8, passes 7 and 8 >= 0.85) a co-binding condition of every C2 label, not only of NOT-EXTENDED.

**F2 (HIGH): the C2-EXT row is half in-pass-budget, and the decoder hop can satisfy it.**
- d 9 at K' 11 needs passes 1..8 correct. Pass 9 can hold, because the decoder hops v_8 → v_9 = a (the same 264-4 mechanism).
- Every pass d 9 needs is therefore inside K = 8 and trained at d 8. Pooling d 9 with d 10 lets "IN-DEPTH d 8 + decoder hop at 9 op sentences" carry about half of the pooled bar.
- Only d 10 (pass 9 > K) actually needs K' > K computation. K'-GAIN at d 10 (K' 12 vs K' 8) is close to automatic: at K' 8, d 10 is structurally near floor (§8 P2b says so). So K'-GAIN checks "d 10 above floor at all", not "more passes help".
- **Fix:**
  - Binding C2 rows are d 10 and d 11 at K' = d + 2. d 9 is descriptive (IN-DEPTH+1).
  - Add a mechanism clause: extend ws_deep (eval-only flag) to TEST2 d 9-10 at K' 12 and require the pass-9 ':' WS at d 10 >= 0.5 and >= init + 0.3.
  - Replace K'-GAIN with acc_d10(K' 10) − acc_d10(K' 9) >= 0.10. Pass 10 must add over the decoder hop from v_9.
  - Pre-state the decoder-only expectation: acc_d10(K' 9) ≈ P(passes 1-9 right).

**F3 (HIGH): there is no arm for plain answer-only extension, so nothing can be attributed to HRM / TRM.**
- The repo already has this recipe: §251 / §255 AEX, `--d_max N --aux_d_max k`, answer CE at S_K, full graph. On of_s1 it needs only the new d law: `--ext_p 0.25 --ds_ext 0`, with no DS branch.
- At d 8, DSF is *identical* to PLAIN (T_d = {8}). DSF differs from PLAIN only by the intermediate answer-hold terms on d 5-7. DSX differs from PLAIN by those terms plus TBPTT-1.
- CONT is a pure null. It never receives deep items, and P3 = 0.90 says it will fail, so it confirms the warm-restart confound and nothing else. There is no positive control.
- **Fix:**
  - Stage 1 arms become DSX s1, PLAIN s1 (AEX-OF) and CONT s1. Drop DSF from Stage 1: it is ≈ PLAIN, its attribution reads one seed, and it needs extra code.
  - Pre-state the attribution as transplant = DSX − PLAIN on the F2 rows.
  - If PLAIN >= DSX − 0.05, Stage 2 runs PLAIN, and the wording is "answer-only depth extension of of", not "deep supervision".
  - Gate PLAIN at 2k with the same WS rule.
  - Claim wording must not say "HRM / TRM" unless DSX beats PLAIN by > 1 σ_seed in every seed.

**F4 (MED → HIGH for wording): C2 scope and fairness.**
- After DSX the trained depth is 8 (answers), so d 10 is +2 beyond training. of_s1 already extrapolates +3 at fixed K (trained d <= 4 → d 7 0.909). A DSX "C2" at d 10 >= 0.40 is a *smaller* relative extrapolation than of already has.
- It is new only in the K' > K sense, which is the registered NORTH_STAR C2 construct ("past the trained budget"). So the claim is legitimate only as "K' > K helps at d 10 after training to d 8". It is not "moves the depth horizon" relative to of, and INIT + 0.20 compares models with different trained depths.
- Lineage: Note 269-2 (2) says >= 2 lineages once §268 lands, and §268 is launching now (b422). Stage 2 will very likely be scored after §268 lands, so the 3/3 one-lineage claim of §9.3 would already be inadmissible.
- **Fix:**
  - Word the result as "K' > K extension at trained depth + 2", and report extrapolation as (d − trained depth).
  - Pre-state: if §268 has landed when Stage 2 is scored, the arm verdict is capped at "3/3, one lineage, no C2 claim". Alternatively, add a DSX arm on the §268 lineage head when it exists.
  - Treat seeds 1-3 from one init weight file as data-order seeds; say so.

**F5 (MED; §251 precedent): mem_check measures the wrong shape, and micro differs across arms.**
- mem_train measures only MC_TR = d_max items (488-494). Under DSX, MC_TR goes through the DS branch: 1-4 grad passes, no aux.
- DSX's real peak can instead be the d 4 + 2-insert full-graph step with aux (8 grad passes) or the d 8 hold far pass.
- §251's AEX OOM was this exact failure: aux_d_max lowered the d_max mem_check peak, auto_micro picked micro 1, and the run OOMed for a $1.21 void.
- §4.3 also lets the arms run different micro (CONT 2, DSX "expect 4", DSF 4-8). That is a non-semantic but numerics-changing difference in a single-change comparison under bf16 (repro-batch-size lesson).
- **Fix:**
  - mem_check = max over {d_max DS shape, d ext_lo − 1 full-graph + aux shape, far-pass shape}.
  - Pin one micro, the max over arms, for DSX / PLAIN / CONT, and put it in the args diff.
  - P5 (DSX peak <= 0.75 DSF) is then moot. Drop it, or read it descriptively.

**F6 (MED): the wrong-state credit pushes a text shortcut, and the guard is descriptive.**
- §5 (b): about 30-50% of d 8 items have an earlier pass wrong (X4a). For them, ℓ_8's gradient trains the shared trunk (pass 8 and the decoder) to produce a from text.
- That is a direct multi-hop-from-text shortcut. It would raise deep accuracy and damage C1, and NOT-DECODER at d 4 K' 2 will not see it.
- **Fix:** make NOT-DECODER-DEEP binding inside RETAIN:
  - acc d 8 at K' 5 <= init(K' 5) + 0.10;
  - acc d 10 at K' 8 <= floor + 0.10.
  A DSX that passes C2-EXT while failing these is SHORTCUT, not C2.

**F7 (MED): the C1 gates sit at trained depth.**
- Standing rule 3's d 6-7 gates were set for of, where d 6-7 were extrapolated. Under DSX, d 6-7 are trained depths (answers), while the C2 claim is at d 10.
- swap_kd / lp5h loop d 3..d_max = 8 (inject_aw_eval part_swap_kd / part_lp5h). They cannot reach 9-10 without an eval change.
- **Fix:** an eval-only flag extending swap_kd / lp5h to d 9-10, with C1-DEEP read at d 10 for the bundle (d 6-7 stay RETAIN-type). If that is not built, drop the C1 + C2 bundle claim from this design.

**F8 (MED): the 2k gate.**
- (a) The GO margin of +0.10 on 200 items is ~2 SE, unpaired. Both arms come from the same init on the same 200 items, so use a paired McNemar-type difference with a Wilson / NEAR flag. Also require DSX ws['8']['8'] >= 0.5 absolute; of's level is unknown because of_s1 never logged d 8 WS (eval_parts had no ws_deep, and d_max was 4).
- (b) The DRIFT clause needs every one of 10 cells (d 1-4, t <= d) >= 0.95 at n = 200 with an absolute bar. If of_s1 has any cell at 0.95-0.97, a false PARK is likely. Bar: each cell >= init − 0.03 (Stage 0 (b)), with NEAR flagged.

**F9 (LOW): T_d = {d..K}.**
- This is plan-given scope, not a step label: d equals the number of distinct cue inputs (len(xs)), which the input already reveals. ℓ_d is the final answer, not an intermediate value. So C4-partial wording is unaffected.
- But it trains "answer decodable from pass d onward", a halting clock. Any C5 / halthead reading on DSX models is then a trained artefact (trained-readings lesson). State that C5 is not readable on DSX.

**F10 (LOW): the line-170 fix.** `Ssv[min(d + 2, K)]` changes no gradient: H is empty for d >= 6, and the far pass has its own den. But for d 7-8 the logged norm guard s8 / sd2 becomes 1 identically. Report the norm guard by d for d <= 6 only. It is needed for CONT too (mem_train); disclose it in G1b.

**F11 (LOW): the warm restart turns the hold penalty off for 500 steps** (hold_start 500) on a model trained with it. This is the same in all arms, but C6 at 4k depends on it. Prefer hold_start 0 for all arms, or disclose it as a C6 risk.

**F12 (LOW): overlap and order with §265.**
- Both target the X4a horizon. F1 shows DSX's late-pass training mostly covers "last hop at more op sentences" (§265's axis, at a larger op count) plus the new deep-state distribution.
- §265's pilot (b418-b421, on bun) is next in the queue and reads rel7 directly.
- Order: run DSX Stage 1 after the §265 pilot reads (not the §265 full), and re-state priors from it. If §265 reads FADING, DSX's P1 / P2 should drop.
- Novelty: Bansal 2202.05826's progressive loss (a no-grad prefix plus a supervised suffix) is DSX's direct ancestor. Literature must check it before any "transplant" wording.

**F13 (LOW): OC and censoring.** These are fine and conservative (FAIL after one rerun, no extension past 4k). Update the OC for F2's d 10 / 11 rows, which will lower P(C2 | effect), and for the lineage cap (F4).

## Is it worth running?

Stage 1, yes. It is the cheapest direct test of whether *answer-only* d 5-8 training on an in-basin of model gives K' > K computation at d 10. Nothing in the repo answers that, and a negative is informative too: it would point to §265 / SSMax or to the state distribution. The HRM / TRM framing and Stage 2 are not worth committing to until PLAIN vs DSX is read.

## Minimal staged plan I would approve

- **Stage 0 ($0):** as written.
  - Add of_s1 eval_acc d 8 at K' 5 and d 10 at K' 8 / 9 / 10 (F6 / F2 refs).
  - Add of_s1's final d 1-4 WS cells (F8b).
- **Build:**
  - the line-170 fix;
  - the ext_p d law;
  - the DS branch;
  - mem_check over all three shapes and one pinned micro for all arms (F5);
  - eval-only flags: ws_deep d 9-10 at K' 12, and swap_kd / lp5h d 9-10 (F2 / F7);
  - G1-G8, with G3 written as the toy's identity (the DSX grad equals the sum of one-pass grads at the frozen states).
- **Stage 1a (2k):** DSX s1, PLAIN s1 (`--ext_p 0.25 --ds_ext 0`), CONT s1.
  - Gate DSX and PLAIN separately with the paired WS rule (F8a) and the relative DRIFT bar (F8b).
  - PARK the family if neither passes.
- **Stage 1b (4k):** the arms that passed, plus CONT.
  - Binding per seed: IN-DEPTH (acc + WS d 8); C2 at d 10 / 11 at K' = d + 2 with the pass-9 WS clause and the K' 10 − K' 9 increment (F2); RETAIN plus NOT-DECODER-DEEP (F6); C6.
  - d 9 descriptive. Attribution = DSX − PLAIN.
  - Guard ≈ $3.5 (3 cells × ~$0.75 + gates).
- **Stage 2:** only for the winning arm (DSX if it beats PLAIN by > 0.05, else PLAIN), at 3 seeds.
  - Wording capped per F4: one lineage. No C2 claim if §268 has landed, unless a second-lineage arm is added.
  - C1 + C2 bundle only with the d 10 C1 gates (F7).
