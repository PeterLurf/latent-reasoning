Status: independent red team of 19_scale_port_qwen3_spi.md, 2026-11-04. Read-only: no training, no download, no computation on stored run outputs. Read: the design, THEORY Notes 234-2, 251-7, 258-7..9, 264-1, 269-1, NORTH_STAR C1, the code cited below, and args.json / train_summary.json of p2m_s1b, askm_lr1e4, aw_plaw2_s1.

# 19 RED TEAM: SCALE PORT (S-only decoder on medium, then Qwen3-0.6B)

## Verdict: GO-WITH-FIXES, for Stages 0-1 only. Stages 2-4 (Qwen) are not approved as written.

The plumbing half of the design is careful. The RoPE / GQA / QK-norm port, the M5 restricted-trie read, the default-preserving edits, the bitwise gates, censoring and cost guards are mostly sound. The science half has two HIGH problems:

1. The diagnosis's key contrast is contradicted by recorded DATA. Small has the same decoder shortcut (d 2 solved at K' 1 = 1.00), and its pass-2 WS still formed.
2. The Stage 1 GO identification (SWAP2 under an S-only decoder) has a hole: operand-carry. The hop can move from the decoder's text read into the state itself, so P(false GO) is not ≈ 0.

Both can be fixed cheaply. There is an eval-only mechanism battery on the stored p2m_s1b model.pt (cents of host time), plus one extra causal part. The battery should run before any training spend, and it should decide whether an S-only decoder is even the right remedy.

## Findings

### F1 (HIGH): the shortcut diagnosis does not separate small from medium

**Claim.** §1.2(b) and §3 M2 say that medium's decoder does the last hop from v_1 plus the text, so the answer gradient into the pass-2 write vanishes. Small, it says, cannot do that hop ("small's lucky constraint", G-AW1 0.118). M2's prediction is tagged "already met".

**Evidence against.**
- G-AW1 (Note 234-2) is a **frozen** askname trunk with only W_o fitted. It says nothing about the trained P-2 model.
- Note 251-7 (THEORY ~23467): "Every model solves d 2 at K' 1: the decoder takes the last step (**aw_plaw2 init 1.00**, CTRL 1.00, AEX 1.00 ...). So d <= 2 answers never require pass 2 to write v_2."
- So the trained small P-2 seed has assumption A1 (the decoder route) exactly as medium does: K' 1 = 1.00 vs medium's 0.974 / 1.000. Yet its pass-2 WS = 1.00.

**Consequences.**
- Both bases have an aux-formed write, and both have the decoder shortcut. They differ **only** in what the pass-2 ':' LM readout shows: v_2 on small, an echo of v_1 on medium (0.84 / 1.00, Note 264-1).
- M2 cannot explain that difference. The only surviving version of M2 is an **order-of-acquisition** story: on medium the shortcut fit before the write, and on small after. Nothing cited tests that, and the per-bin K' 1 trajectory is, as far as I can see, not logged.
- The S-only decoder is therefore a remedy for a cause that is not shown to be the cause. A Stage 1 PARK or GO would not be interpretable against the §258 diagnosis.

**Fix.**
- Rewrite §1.2(b) and M2: drop "small's lucky constraint" and the "prediction already met" tag, and cite Note 251-7.
- Make the stage that actually discriminates the readings the routing stage (F3).
- State M2 as order of acquisition (CONJECTURE). If that version is kept, give it a prediction someone can check.

### F2 (HIGH): operand-carry breaks M1, M3 Corollary 1 and M4, so SWAP2 under S-only is not "pass 2 computed the hop"

**M1 is overclaimed.** "Aux 1.00 at pass 2 ⇒ S_2 determines v_2" is right. "⇒ some pass ≤ 2 applied op_2" does not follow.
- The aux reader is a trunk forward: `fwd_read` over `[EOT, " Step", " t", " :"]` + target. See inject_ow.py 92-103 and nested170 aux_logits 260-273.
- On medium, 12 trained blocks sit above the read at L/2. Note 264-1 shows medium's upper blocks apply one op to a value read from S.
- So if S_2 stores (v_1, op_2's operand / type), a reader computes v_2. This is the "trained readouts are artefacts" lesson in a new place: the aux is a trained reader that can do the hop itself.

**M3 Corollary 1 is not PROVED.** With `--dec_src cue` the decoder cannot read the op-2 sentence from the text, but pass 2 can **write it into S**.
- Under swap_1 at K' 2, S_2 = W(S_1^don, x_rec) then holds v_1^don (carried from S_1) plus op_2^rec (written by pass 2). A decoder that applies the op it reads from S returns op_2^rec(v_1^don) = ca. follow_ca ≥ 0.90 then passes SWAP2 with **no hop in any pass**.
- This is exactly the medium shortcut moved one step inward. Medium has already shown it can do "one hop from a read value + op" in a single forward.
- **M4 has the same gap.** "Pressure persists until S_K determines v_d" is met by (v_1, op_2) in S. So S-only does not force the hop into a pass at all, not only "not into pass 2".

**Labels are misnamed.** Under operand-carry in pass 1, NOT-DEC fails and the run is labelled EAGER-P1 ("pass 1 computed both hops"). It could equally be "pass 1 stored op_2".

**Fix: add a write-transplant part (WT2) to swap_1f / so_parts and make it binding for SEED-CAUSAL.**
- For pairs (A, B) with v_1^A ≠ v_1^B and op_2^A(v_1^B) ∉ {v_2^A, v_2^B}:
  1. Run pass 2 of A on (S_1^A, x_A) and capture its write inputs (h6, h_w) at ':'. With γ = 1, k / v / gates come only from `mem.ln_w(h_w)`; see write_seq, deltamem.py 179-223.
  2. Apply `write_seq(mem, h6^A, hw^A, S_1^B)`.
  3. Read with the cue decoder (and the aux t = 2) on B's question.
- **follow_valA** = v_2^A means a computed value was written. **follow_opA** = op_2^A(v_1^B) means an operand was written.
- Register: **COMPUTE** if follow_valA ≥ 0.5 and follow_opA ≤ 0.2; **OPERAND-CARRY** if the reverse.
- GO via SEED-CAUSAL needs SWAP2 ∧ AUX2 ∧ COMPUTE. Add OPERAND-CARRY as a label before S11-PASS.
- Floor: MEAN-1 S_1 in place of S_1^B.
- About 40 lines in so_parts, no_grad. Add a G-DS1-style fixture: on a hand-built state where the write is known, the part returns the known class.
- Also retag M1 "PROVED: S_2 determines v_2; which pass / reader applied op_2: open", and M3 Cor. 1 "DERIVED-UNDER no-operand-in-S".

### F3 (HIGH, the cheap discriminator the design says does not exist): Stage 0b, an eval-only mechanism battery on p2m_s1b model.pt

§1.2(d) says "stored data cannot separate" ECHO-READOUT from NO-PASS-2-COMPUTE. Stored *outputs* cannot. The stored *model* can, with eval-only parts on one A6000 host (≈ 15 min, ≈ $0.15). All of these run under the **text** decoder or the text-free aux, so they need no training:

| read | what | separates |
|---|---|---|
| R0c AUX-SWAP2 | swap_1 (S_1 := donor's), then the **aux t = 2** on S_2 (text-free reader). follow_ca / stay_own / follow_v1don | whether pass 2's write depends on S_1 and on the recipient's op 2 (rules out "pass 1 eager", "accumulation" and "echo-only write") |
| R0d WT2 | F2's write transplant, read by the aux t = 2 and by the text decoder (descriptive) | COMPUTE vs OPERAND-CARRY in medium's pass-2 write |
| R0e AUX2-ON-S1 | the aux t = 2 prompt on S_1 (no pass 2) | an eager v_2 / op_2 already in S_1 |
| R0f WS-LAG | already in the design as R0a (WS passes 3-8) | ECHO-READOUT's lag signature |

**Routing (pre-state before compute; Audit ack per the pre-state rule):**
- **AUX-SWAP2 ≥ 0.90 ∧ COMPUTE ∧ AUX2-ON-S1 ≤ 0.2 → ECHO-READOUT.** Medium's pass 2 *does* compute v_2; WS was the wrong gate.
  - The S-only decoder is then unnecessary for formation. §258 stays parked as registered (no post-hoc rescoring), but the next step is a new registration that ports with **aux + AUX-SWAP2 + WT2 as the formation gate**, and the Qwen stage can drop the decoder change. Fewer factors, and the lineage stays C1-comparable (F4).
- **OPERAND-CARRY or AUX-SWAP2 at floor → the write is not a computed v_2.** Run Stage 1 as designed, plus WT2 (F2) and the small fresh control (F5).
- Anything else: run Stage 1 as designed, with WT2.
- This costs less than 10% of SO-W and can make all of Stage 1 unnecessary.

### F4 (MED): an S-only decoder changes what the model is; the C1 scope must say so

**What C1 asks** (NORTH_STAR): "removing or blocking the latent steps drops accuracy to chance ... (K' 0 and K' 1 far below K)". Under `--dec_src cue`, K' 0 is at chance **by construction**, since the decoder has no access to the problem. So the K' 0 half of C1 becomes vacuous, and only K' 1..d−1 and the serial swaps carry evidence.

**Comparability.** The small SOTA lineage (of_s1, bun) uses the text decoder, and its C1 rows include decoder-route checks (NOT-DECODER). A Qwen S-only seed is a different model class. Its C1 / C2 readings are not comparable to the small lineage without a same-class reference. It also gives up the "decoder may read the text" capability that GSM8K-style use needs (the answer must be produced from a question the decoder can see).

**Fix.**
- Add to the scope line: "S-only decoder: K' 0 is uninformative; C1 evidence = NOT-DEC at K' < d + serial swaps + WT2."
- Pre-state that any later REG d ≤ 4 on Qwen either keeps S-only, with this caveat, or tests a text-decoder re-attach as its own factor.
- Test the decoder change on the small lineage first (F5). The design's SO-WS covers only warm d ≤ 2 behaviour.

### F5 (MED): Stage 1 is a warm repair; nothing tests fresh formation under S-only before the $8.5 Qwen fresh run

**The gap.** Stage 1's GO is read on a model whose S_2 already decodes v_2 (M10). Stage 4 is a **fresh** formation under a decoder change that has never been tried fresh on any base. If S-only breaks or slows fresh formation (for instance by inviting operand-carry from step 0), a Stage 4 NO-SEED cannot be attributed to scale versus decoder. This is the "base size vs decoder shortcut" separation the user asked about.

**Fix.** Add **SO-F-S**: a small fresh P-2 = aw_plaw2_s1's args exactly + `--dec_src cue`.
- Init s233_askname_s1, 12k, seed 1, on **RTX 3090** (aw_plaw2_s1's class; the reference is the stored aw_plaw2_s1, same class, single change).
- Read with §7.1's fields + WT2.
- Cost ≈ 12k × ~0.75 s/step × ~$0.20-0.25/h (3090) ≈ $0.6-0.8; guard $1.2.
- Stage 4 is admitted only if SO-F-S reaches S11-PASS or SEED-CAUSAL with COMPUTE.
- SO-F-S also replaces most of SO-WS's interpretive role. SO-WS can stay as a cheap descriptive arm, but on 3090 (F6).

### F6 (MED): GPU class

- **SO-WS** warm-starts a 3090-trained init (aw_plaw2_s1 train_gpu RTX 3090, args.json) on an A6000. Yet §7.1 uses it to **attribute** an SO-W failure ("attributed to the S-only switch itself, not to scale"). That attribution is confounded by the class flip the memory lessons warn about.
  - Fix: run the small arms (SO-WS, SO-F-S) on 3090, as a separately-labelled sub-registration, so that every comparison is within one class (small cue vs small text on 3090; medium cue vs medium text on A6000). Or drop the attribution sentence.
- **Stage 4's class.** A6000 is forced for the medium warm starts, but Stage 4 is a fresh run with no A6000 reference. Every SOTA comparator is 3090.
  - Qwen3-0.6B at bs 64 with micro pinned probably fits 24 GB: fp32 master + AdamW ≈ 9.6 GB, and medium ran at mem_budget 21.
  - Choosing 3090 for Stage 4 (its own registration, as Stage 5 already is) would make Qwen-vs-small descriptive comparisons same-class. If A6000 is kept, disclose that every Qwen-vs-small row is cross-class.
- G-DS3 ("step-0 eval_ws equals p2m_s1b's per item") is a cross-host comparison. Default SDPA is not run-to-run bitwise. Fix: use the det wrap, or state a tolerance (argmax agreement ≥ 0.995) instead of equality.

### F7 (MED): the Qwen tokenization surface is larger than §5.2 lists, and the aux gate has the 1 / 10 collision

**Hard-coded GPT-2 assumptions that are not listed for porting:**
- inject_st G-CUE: `len(tk) == CUE_LEN` (7), `tk ∩ NUM_IDS`;
- tok_aw's §262 assert `q[-7] == ENC.encode(" " + name)[0]` (inject_aw.py 92-93);
- tok_ow's `vidx = q.index(HAS_TOK) + 1` (inject_ow ~70);
- `pred_of` = argmax over NUM_IDS (inject_aw_eval 39-40), used by every eval part;
- swap_pairs' equal-token-length filter, which uses `IO.ENC` (inject_aw_eval 217-232).

Qwen splits " 10" into 3 tokens and " 7" into 2, so the length filter, the cue offsets and pred_of all change. §5.2 mentions SFX_Q and the aux prompt only.

**The aux 1 / 10 collision.** §5.2 puts "the aux target index on digit tokens only". Today aux_vacc reads **one** index (`pr[ar, vi] == y_a[ar, vi]`, passes() in inject_aw_eval 75-76). On Qwen the first digit of 10 is "1", the same token as the class 1, so an aux that confuses 1 and 10 scores as correct. AUX2 is binding.

**Fix.**
- List tok_ow / tok_st / tok_aw / pred_of / swap_pairs in §5.2 as ported (Qwen variants in inject_q, never edited in place).
- Score aux exactness over **all** digit tokens plus the terminator (sequence-exact), or through numread's s(·).
- Add to G-Q5: 20 items per d tokenized and decoded round trip; aux sequence-exact on a hand-built 1 vs 10 case; swap_pairs' length filter under the Qwen tokenizer with n_pairs reported (it may fall under 1,000; pre-state the floor).

### F8 (MED): the warm 6k horizon plus "censored = PARK" could false-PARK

M10 (adapts within 6k) is a CONJECTURE. FD1 (Note 269-1) puts fresh onset at 4-7k, and medium's aux at 9-10k. SO-W's AdamW state is fresh, with a 300-step warmup, and the decoder has to learn a new input map. A 6k run that is still rising at 6k is PARKed.

**Fix.** Pre-state one +4k extension of SO-W, using Stage 4's trigger rule (SWAP2 or WS rising ≥ 0.10 over the last 2k, or AUX2 ≥ 0.5 with SWAP2 < 0.9). Guard +$1.2, which fits the family cap. Keep the "PARK is a cost decision" wording (§9), and use it in the Note.

### F9 (LOW-MED): SEED-CAUSAL as a GO (Unresolved #1)

My answer: SEED-CAUSAL should count only together with WT2 = COMPUTE (F2). On its own, a causal swap through a trained reader does not locate the hop.
- Do not move the formation gate entirely off WS. WS is the only in-pass readout that no reader computes on top of; that is its value.
- Keep FORM-WS as the strong label, and SEED-CAUSAL+COMPUTE as the weak GO.

LATE-HOP (Unresolved #2): agree that it is "not a seed".

### F10 (LOW): math-tag hygiene

- **M1:** see F2.
- **M3 Cor. 1:** see F2.
- **M4:** the CE bound ≥ H(v_d | q) assumes the decoder has no other channel. Fine as DERIVED once A2 is checked, but its conclusion should read "until S_K determines v_d, *by any encoding*".
- **M5:** PROVED is fine. Note that s(1) = log p("1") + log(1 − p("0" | "1")) treats "11"-style continuations as mass for class 1. That is acceptable under the 11-class restriction, but state it.
- **M7:** fine.
- **M8:** a CONJECTURE whose stated P 0.65 is not used by any gate. Fine.
- **§1.3:** "Qwen3 tokenizer = Qwen2.5's BPE". This matches my knowledge: Qwen3 reuses the Qwen2.5 byte-level BPE, with digits as single \p{N} tokens, so " 7" → [" ", "7"]. G-Q5 asserts it anyway. Good.

### F11 (LOW): cost basis

- The rates check out against the stored files: p2m_s1b A6000 train_sec 15,127 at 12k; askm_lr1e4 2,110 s for 8k = 0.26 s/step; the stated 0.252 is consistent.
- But p2m_s1b's $3.554 includes about $1.5-1.8 of non-training cost (setup + a 9-part eval). SO-W's eval allowance of "$0.12" for a step-0 eval plus acc / ws / swap_1 / swap_1f at KMAX 14 looks low. Expect ~$0.3-0.5.
- The guards absorb this, but the realised cost will likely be about $2.2, not $1.8.
- The Qwen factor 1.6 ignores the 3x larger vocabulary projection at the aux / decoder positions and the 2048-wide q / o. It is probably within the ×1.4 guard, and G-TIME plus the $7.5 projection stop catch it. OK.
- Family with my additions: ≈ +$0.15 (Stage 0b) + $1.2 (SO-F-S guard) + $1.2 (SO-W extension guard) → guard sum ≈ $22, under the $30 line. Raise the family guard to $23.

### F12 (LOW): `DEC_SRC` as a module global

The global covers every tok_aw call site I checked (inject_aw 103, 351, 364, 373-374, 482-486, 569; inject_aw_eval 96 ... 1273, as listed). But scratch scripts and the `_inject_aw_eval_p*.py` copies import AW without setting it, so they would silently evaluate a cue model with the text decoder.

**Fix.** In `dec_logits`, assert that the item's dec form matches `AW.DEC_SRC`, and that `AW.DEC_SRC` was set explicitly from args.json, using a sentinel default of None in eval paths. COV already checks the meta.

## Answers to the questions asked

1. **Diagnosis.** "The write formed, but the decoder shortcut kills the gradient" is not supported as the *difference* between small and medium: small has the same shortcut (Note 251-7). Two readings fit the data equally well:
   - (a) the aux is reader-computed from an operand or echo state;
   - (b) medium's pass 2 computes v_2 but its ':' head reports the read value.

   Cheap discriminator: Stage 0b on p2m_s1b model.pt (AUX-SWAP2, WT2, AUX2-ON-S1), at about $0.15.
2. **--dec_src cue.**
   - It changes the model class. The K' 0 half of C1 becomes vacuous (F4).
   - It should be tested fresh on small first (SO-F-S, 3090, same-class single change vs aw_plaw2_s1) to separate decoder from base size.
   - Stage 1 is a warm repair and cannot show fresh formation. It is decisive for PARK only as a cost decision, and only with WT2 and an extension rule.
3. **GPU class.** Medium warm arms on A6000 are right. The small arms must run on 3090, or lose the attribution sentence. Stage 4 should be a separate registration whose class is chosen with the comparators in view (F6).
4. **Qwen3.**
   - Config and tokenizer assumptions look right and are gated (G-Q1/Q5).
   - The 1.6x cost factor and $0.42/h are plausible and guarded by G-TIME and the projection stop.
   - The gates miss the tokenization surface (cue-length asserts, pred_of, the swap length filter) and the aux 1 / 10 collision (F7).
5. **Math / OC / censoring / seeds / guards.**
   - The tags on M1 and M3 are overclaimed (F2), and P(false GO) ≈ 0 is wrong without WT2.
   - Censoring is good.
   - One seed for a warm PARK is acceptable as a cost decision.
   - Guards stay under $30 with the additions.
6. **Right next scale step?** Not yet. The right next step is the $0.15 mechanism battery (Stage 0b), because it can show that medium already computes pass 2 (then: re-gate, not re-architect) or that the write carries operands (then the S-only decoder needs WT2 to mean anything). Building Qwen comes after that.

## Minimal staged plan I would approve

- **Stage 0 ($0 reads + ≈ $0.15 host eval).** Pre-state, push, get the Audit ack.
  - R0a / R0b as designed.
  - **R0c AUX-SWAP2, R0d WT2, R0e AUX2-ON-S1** on p2m_s1b model.pt (A6000, eval-only, new parts in so_parts with fixtures + G-DS1-style equivalence).
  - Routing as in F3.
- **Stage 1 (only if Stage 0 ≠ ECHO-READOUT; guard ≈ $5).**
  - SO-W (A6000, 6k, + one pre-stated 4k extension) with swap_1f + WT2. GO = S11-PASS, or SEED-CAUSAL ∧ COMPUTE.
  - In parallel on **3090**: SO-F-S (small fresh, aw_plaw2_s1 args + dec_src cue, 12k), plus optional SO-WS.
  - CTRL-W conditional as designed.
- **If Stage 0 = ECHO-READOUT:** skip Stage 1. Draft a new registration that re-gates the port on aux + AUX-SWAP2 + WT2, keeping the text decoder (C1-comparable).
- **Stages 2-3 (build + ASK-Q):** only after a Stage 1 GO with SO-F-S ≠ fail, or an ECHO-READOUT route.
  - Add F7's tokenization ports, the aux sequence-exact scoring and the G-Q5 fixtures.
- **Stage 4 (P-2-Q):** a separate registration with its GPU class chosen there (F6). It carries the WT2 binding, and FORM-WS stays the strong label.
- **Family guard** for Stages 0-1: ≈ $6.5. The whole design stays ≤ $23 if every branch runs.
