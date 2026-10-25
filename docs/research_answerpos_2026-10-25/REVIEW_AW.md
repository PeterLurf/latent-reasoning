# REVIEW_AW: red team of AW_draft.md v0 (0982fcd9) (2026-10-25)

Reviewer: Theory's red-team subagent. Read-only on the repo apart from this file and `review_scripts/` (CPU, 2 threads, idle priority, $0). No blinded file read. aw_gate.py was **not** run, and nothing was loaded from s233_askname_s1.
- **Inputs:** AW_draft.md, aw_checks / aw_gate / aw_oc (.py + .json); THEORY Notes 233-1, 233-2, 233-4 (grep); §160's γ-ramp record (grep); locus.json, probe.json; ST_v1.md, REVIEW_ST.md; boxes/deltamem.py (read_fixed, write_seq, gpt_forward_mem_r), deltas5.fwd_read, inject_ow (InjectOW, loss_items, calibration, γ schedule), inject_st (tok_st, the loss), ow_words; Literature's addendum in LIT_STEPTOLD_2026-10-24.md (9ce7aeb5).
- **Check:** `review_scripts/rv_aw_checks.py` → `rv_aw_checks.json` (+ `rv_aw_oc_reseed.json`):
  - (A) exact-binomial OC, plus aw_oc re-simulated with another seed;
  - (B) the cost and family guard, with launch overheads, and ST-AN inside vs after P-2;
  - (C) the γ schedule arithmetic;
  - (D) D2's read structure on base GPT-2 + DeltaMem as InjectOW builds it.
- **Reproduced:**
  - G-AW0's OC (exact: 0.524 at a true 0.80, 0.9999 at 0.855, 0.0023 at 0.75);
  - LP2's OC (exact at N_eff 495: 0.887 at 0.82);
  - every P-2 truth's PASS within ±0.005 under another seed;
  - P-2 $0.78–1.78 (guard $2.31), ST-AN guard $2.05, P-4 guard $4.17, Stage 1 guard $21.28, family **$29.81**;
  - P(P-2) 0.189 and expected spend $4.18;
  - aw_checks TOK / W2 / PAIRS read correctly.

## Verdict: GO to v1 with fixes 1–7 (8–10 are wording). No new red team is needed if they are applied as written. Fix 1 adds one $0 gate whose rule Audit G-Ds.
- **What holds:**
  - D1 holds at init.
  - The leak proof holds on the code path (issue 4).
  - CD-AN is the right binding comparator for the C1 claim.
  - The OC and cost reproduce.
- **What needs fixing:**
  - The main risk, A1 (read-into-compute), is first tested at $1.8. A $0 frozen-trunk test exists (issue 1).
  - The γ ramp contradicts the treatment (issue 3).
  - FX-LEAK (a) is nearly tautological as prototyped (issue 4).
  - The family guard leaves $0.19 of headroom (issue 6).

## Issues, ranked by severity

**1. [HIGH] A1 (read-into-compute) is the main risk, yet nothing tests it before spend. A $0 frozen-trunk "use" gate tells "cannot consume" from "did not learn".**
- **Readout direction vs composable value is not an expressivity question.**
  - At init, u1 is the final-layer state that unembeds to v_1: a readout code.
  - There are 11 classes, and their 11 class means in 768 d are affinely independent. So a linear W_o can map that code onto any 11-point operand code the upper blocks expect.
- **The real unknowns are two:**
  - (a) Can blocks 6–11 consume an operand supplied as a **broadcast** block-6 residual vector? It has no positional binding. In text form (E3), v_1 is a token that the circuits reach by name → value attention, possibly before layer 6. And only 6 blocks follow the read (REVIEW_ST already flagged this as A3, untested on language).
  - (b) Does the gradient find W_o, starting from W_o = 0?
- **LP2 does not answer this.** LP2 shows a trained read can bring v_1 *out* as a token (readout), not *into* computation.
- **Fix: G-AW1 FROZEN-USE ($0, binding as below; run only after its rule is committed, like G-AW0).**
  - **Model:** the askname trunk + DeltaMem built as in §2.1, everything frozen except W_o (+ bias, s_add, s_del); γ = 1, fade = 1. S_1 = the single-token write of u1 at ':' of x_1 (the AW pass-1 write, no gradient into it).
  - **Fit:** decoder read at K′ = 1 (dec = [EOT] + text + SFX, read_fixed(S_1) at block 6); target v_2 on d = 2 items.
    - Fresh unused stream seed; ≥ 6,000 train / 1,000 val / 1,000 test, split by sig.
    - Adam lr 1e-3, ≤ 2,000 steps, early stop on val.
    - A local 8 GB card (frozen trunk, T ≈ 50) or CPU, ≤ 20 min.
    - Also fit a rank-32 W_o, descriptive (overfitting check).
  - **Controls:**
    - (c1) S := 0 → acc ≤ majority + 0.05;
    - (c2) the harness works: the same fit through LP2's path (aux prompt, teacher-forced R_step[1], target v_1 at the value token) ≥ 0.80.
  - **Rule:**
    - PASS iff test acc(v_2) ≥ 0.50 with (c1) and (c2) valid.
    - FAIL iff ≤ majority + 0.10 with valid controls → **to the user before P-2**, with P(P-2) revised to ≈ 0.05: "the frozen trunk cannot compose from a block-6 read; P-2 rests on fine-tuning building a new circuit". Not an automatic stop, because full FT can adapt the trunk.
    - In between → GO, priors unchanged. A control fails → UNREADABLE: fix once, re-run once.
  - **Descriptive:**
    - SWAP: S_1 from a same-skeleton donor (the aw_checks PAIRS law) → share following cf;
    - T1 (closed-form, CPU): on class-(iv) inputs, patch the value token's block-6 residual from a donor with a different v_1 and report the share that follows op2(donor v_1). This shows whether the trunk still consumes the value after layer 6.
  - **Why it is worth it:** it makes every P-2 null interpretable.
    - NOT-READ with a G-AW1 PASS → the objective / optimisation did not find a read that exists (next design: warm-start W_o, issue 2).
    - NOT-READ with a G-AW1 FAIL → the architecture (read layer / format) is the bottleneck.

**2. [HIGH] Is a null informative as is? Partly. NOT-READ is well detected, but NOT-STORED is confounded with source drift, and one outcome cell has no label.**
- **Detection works.** In `pass1_only` (aw_oc), LP3, LP5 and SWAP-K all fail (PASS 0.000), and the pattern LP2 pass ∧ LP3 fail ∧ LP5 < 0.10 is specific. LP2 passing itself shows S_1 holds v_1, so drift cannot fake NOT-READ.
- **NOT-STORED vs drift.** LP2 can fail because the ':' code eroded (A3). The decoder CE trains the same trunk's ':' after a text whose only question is the final one, and no loss keeps the readout of the appended sub-question. The other possibility is that the write / storage failed.
  - The draft separates the two with WS-READOUT (the unembedding argmax). But the aux gradient through S_1 can keep v_1 in the W_v subspace while the unembedding readout decays.
  - WS-READOUT < 0.80 is therefore ambiguous. A SOURCE-DRIFT truth gives the same binding pattern as no_effect (rv A: LP2 0.000, PASS 0.000).
- **Fix:**
  - (i) An end-of-run `u1|v1` probe on the trained model (G-AW0's protocol and items, eval-only from model.pt, $0 locally).
    - LP2 fail ∧ probe < 0.80 → **AW-SOURCE-DRIFT**;
    - LP2 fail ∧ probe ≥ 0.80 → **AW-NOT-STORED**.
  - (ii) WS-READOUT every 1k steps on a fixed held-out set (256 SEED_MEMCHECK d = 2 items). Run it in a separate no_grad function, not inside the training think path, so FX-LEAK's channel (T) stays literally true. It times any drift.
  - (iii) Add **AW-WEAK-READ** (LP2 pass, LP3 fail, 0.10 ≤ LP5 < 0.30) so every cell has a label.
- **No mechanism that forces reading should go into P-2.** It would be a second change on the user's chosen arm and a problem → fix step.
  - Pre-name, as *new designs* (not fallbacks): (a) W_o warm-started from G-AW1's fit; (b) ELIDE batches (S_1's sentence removed from the pass ≥ 2 and decoder inputs, so v_1 exists only in the state).
  - With issue 1 and fix (i), a null tells which of these is motivated.

**3. [MEDIUM-HIGH] The γ ramp contradicts the treatment. Set γ = 1 from step 1 (write_mix 0, 0), as part of the treatment, not as a separate change.**
- Under the ramp, steps ≤ 2,000 write from LN_6(h_6) at ':' (v_1 content unknown on askname; on the base trunk `u0|v1` 0.083, and §233's `i_h6_cue|v1` was 0.092), and steps up to 6,000 write a mix. So for half the run the write source is not D1's.
- W_k / W_v and W_o co-adapt to the u0 distribution and must re-adapt during the ramp. §160 CTRL-R records the loss rising through exactly this ramp (0.013 at 2k → 0.31 at 6k, never settling).
- **Fix:**
  - `--write_mix_start 0 --write_mix_end 0`. inject_ow line 352 then gives γ = 1.0 from step 1 (rv C).
  - Define the treatment as "the write source = the top-layer ':' state for the whole run" (w1 + w2 + schedule). CD-AN has no write, so it is unaffected.
  - G-AW0's binding feature is then the source at every step. G6's args diff lists write_mix.
  - ST-AN (if run) uses the same schedule, so attribution isolates the write *position*.
- **No SOURCE-CE in P-2.**
  - At d = 2, pass 2's question is the decoder's question, so SOURCE-CE at pass 2 duplicates the decoder CE. Pass 1 is already at 1.000.
  - It is a second change. Its only distinct role is anti-drift and per-pass supervision at d ≥ 3.
  - Pre-name it for P-4 only, if issue 2's probe shows drift.
  - It would not make the loop text-CoT (no token is fed), and it adds no supervision class beyond the aux's teacher-forced v_t.

**4. [MEDIUM] The leak proof holds on the code, but FX-LEAK (a) is nearly vacuous as prototyped, and its positive control plants the leak through an explicit argument.**
- **Verified:**
  - `InjectOW.dec_logits(S, items)` → `deltas5.fwd_read(g, embed(dec), mem, S)`. That is read_fixed at block 6, no write, no `h_low`, then ln_f.
  - tok_ow's dec = [EOT] + enc(" " + text) + SFX.
  - So the decoder logits = F(θ, dec(text), S_K), plus module globals READ_CLAMP (None), POS_OFFSET (0) and mem.ablate (None).
  - The aux returns logits only, and S is not modified (check (d)).
- **The gap in the prototype:**
  - `chk_leak` calls `m.dec_logits(S_fix, items)`. Think inputs can reach it only through state stashed on the module, so (a) is true by signature.
  - The planted leak (`leaky_dec`) receives `lastx` as an argument. Of course it is detected; that proves nothing about hidden channels.
- **Fix (G7 as built):**
  - (a) Drive the real entry points (inject_aw's `loss_items` and the eval `acc` part), with a hook that replaces S_K by S_fix just before `dec_logits`, across real / wrong-cue / other-text think inputs.
  - (b) The positive control is a test-only subclass whose think pass stashes the ':' argmax (or h_low) on `self`, and whose dec_logits reads that stash. The fixture from (a) must flag it.
  - (c) Assert `InjectAW.dec_logits is InjectOW.dec_logits`, and READ_CLAMP None / POS_OFFSET 0 / ablate None at every eval part.
  - (d) Batch-composition invariance: item i's decoder logits are bitwise unchanged when the other rows of the batch change.

**5. [MEDIUM] D2 and D3 are stated too strongly. The read is 12 per-head scalars × v_h, and S_K at d = 2 is not one term.**
- **The read (rv D, base GPT-2, exact to 4e-7).** After one write, the read at a position is c_h(pos, item) · v_h per head.
  - The additive c_h are > 0. They vary 11 % over positions and 2.7 % over items, and the max/min across heads is ≈ 1.6.
  - The delta c_h = β k̂·q̂ is **negative at 22 % of positions**.
  - So v_1 reaches W_o as head-blocks with position-varying weights and signs. That is learnable, but it is not "one scalar times v".
- **Fix:**
  - Reword D2.
  - Replace the descriptive `s1|v1` probe (49,152-d outer products on 1,800 train items; §233's `iii_S1` precedent had train acc 1.0) with `read1|v1`: the pre-W_o read [oa ‖ od] (768-d) at the decoder ':' of the d = 2 item, from S_1. That is the object D2 is about.
- **D3 holds for S_1 only.**
  - At d = 2, S_K = the pass-1 write + 7 writes from the S_2 question. At init these carry the one-pass guess (chance: CUED-DIRECT 0.089).
  - The additive heads hold v_1 at 1:7 against structured noise. The delta heads (α ≈ 0.993, β ≈ 1, similar ':' keys) mostly overwrite it.
  - So D6's "pass 1 + decoder" route starts swamped. The clean route is chaining: pass 2 reads S_1 alone.
- **Keep K = 8** (single change), but:
  - make the decoder at K′ 1 and K′ 2 descriptive readings beside LP5;
  - log the cosine of k̂ between the pass-1 and pass-2 writes.
  - At P-4 this becomes the back-reference risk (Q8: near-identical ':' keys blend v_1..v_t in the additive heads, and the delta heads keep only the latest).

**6. [MEDIUM] Cost: the family guard is $29.81, with $0.19 of headroom under the $30 cap. Realistic overheads already exceed it.**
- **Overheads:**
  - §233's per-launch overheads (refused-admission BAD $0.24, BAD $0.28, eval re-run overrun) at ≈ $0.26 × 6 launches → **$31.37** (rv B).
  - The censoring rule's single UNREADABLE re-run of P-2 alone takes its guard to $4.62.
- **The high anchor is pessimistic.** It is ST's micro-2 time, which includes the T-token delta-head write loop. AW writes one token, and ST's own micro-1 peak was 21.2 GiB, so AW at micro 1 is likely. The high anchor also has AW's extra 4 tokens.
- **Fix:**
  - Approve the family at $30 with a running rule. Before each stage, realised spend (BAD, re-runs and overruns included) + the remaining stages' guards must be ≤ $30, else the user decides.
  - Stage 1 is re-costed from P-2's *measured* AW s/step, not P-4's alone.
  - P-2 admission is unchanged ($1.85 projected, G-TIME, 4-BAD).
  - Expected spend is $4.18 as drafted.

**7. [MEDIUM] Attribution: ST-AN stays after a pass, but its rule should be pinned now.**
- **Cost / benefit:**
  - Inside P-2, ST-AN raises expected spend from $4.18 to $5.41 (+$1.23, rv B), since P(P-2) = 0.19.
  - If AW fails, ST-AN mostly confirms "init alone is not enough". The surprising case, ST-AN passing while AW fails, is possible through the 0.855 name-token source inside ST's all-token write, but has a low prior.
  - The C1 claim (loop vs one-pass on the same init) rests on CD-AN, not ST-AN. ST-AN attributes the *mechanism* only.
- **Fix:**
  - Commit ST-AN's args and rule now: §233 LAT-ST from askname, with issue 3's γ schedule, the same GPU class, LP3 read.
  - The labels WRITE-SOURCE-CREDITED / INIT-SUFFICES stay as drafted.
  - Exposure (Q2) is matched between AW and CD-AN, and TEST2 is excluded. No change.

**8. [LOW] G-AW0: pre-state the "fix once", and the bar change is acceptable.**
- **The bar change:**
  - 0.90 → 0.80 was made on the descriptive base reference, before any askname reading, to 231-2's precedent, with `.void_bar090` kept. Acceptable.
  - Audit should confirm from git that no `aw_gate_askname.json` predates the rule's commit.
- **Fix:** name the UNREADABLE fix now. If `v0ctl|v0` < 0.80 on askname (its top layer at '.' predicts the next sentence), the binding control becomes `v0ctl_u0|v0` ≥ 0.80 (mid-layer; 1.000 on base). Same items, one re-run.
- D1 is algebraically sound: both LNs use eps 1e-5 over the same n(h_w), and the probe standardises. G-AW0 is a build check, as the draft says.

**9. [LOW] D6 / Q5: WS-READOUT pass 2 as a P-2 requirement is nearly moot at d = 2.**
- Pass 2's question is the decoder's question, under shared weights. Any decoder use of v_1 implies pass 2 can do the same.
- Keep DEC-STEP2 descriptive, and require chaining at P-4 (LP4 + WS-READOUT pass t ≥ 0.50 at d 3–4, pre-stated now).

**10. [LOW] Novelty: fold in Literature's addendum (9ce7aeb5) and drop any "only" / "first" framing.**
- **N1 NARROW:** "a query-conditioned variant of Coconut-style last-hidden-state feedback, written into a recurrent memory (plan given)".
  - Priors: Coconut 2412.06769 (the last hidden state fed back); CODI 2502.21074 (distils the ':' answer-position state, as a target, not written to memory); RMT 2207.06881 (verified; writes from dedicated write tokens).
  - Not found: a per-pass sub-question whose answer-position state alone is written to a recurrent memory.
- **N2 NARROW:** kin are Biran 2406.12775, Yang 2402.16837 and CODI §5.1. The measurement is our own.
- **N3 NARROW:** mean ablation and swaps are standard (SCIT 2608.27265 / §221).

## Answers to the brief, one line each
1. **Derivation:**
   - D1 is sound at init (exact up to per-dimension affine maps).
   - D2 holds per head, not as one scalar (issue 5).
   - The ':' state is a readout code. Whether the upper blocks can *compose* with it is the open question, and G-AW1 FROZEN-USE tests it at $0 (issue 1).
2. **NOT READ:**
   - LP5 / MEAN-STATE / SWAP-K detect it (PASS 0.000 under `pass1_only`).
   - Add AW-WEAK-READ and the drift / storage split (issue 2).
   - No forcing mechanism in P-2. With G-AW1, a null is informative.
3. **Attribution:** keep ST-AN after a pass (+$1.23 expected if inside), but pin its rule and γ now (issue 7).
4. **Leak:** F(θ, text, S_K) verified on dec_logits → fwd_read. FX-LEAK must drive the real entry points and catch a stashed-on-module planted leak (issue 4).
5. **γ:** start at 1 (write_mix 0, 0) as part of the treatment. No think-pass answer CE in P-2 (issue 3).
6. **OC / cost:**
   - The OC and cost reproduce; censoring is sound.
   - Add the drift label.
   - The family is effectively > $30 with overheads, so add the running-cap rule (issue 6).
   - The bar change is acceptable (issue 8).
   - Novelty: NARROW × 3 (issue 10).
