# REVIEW_CC: red team of CC_draft.md (Opus subagent, 2026-10-30). Verdict: NO-GO as drafted; GO-with-fixes if staged. Theory: $0 stage S0 first (Note 251-12)

1. **BLOCKER (recipe):** a cold askname + fresh DeltaMem at REG d ≤ 4 for 16k is §236, which failed in both seeds (d 2 0.295 / 0.166). Only the warm §237 passed. rcb is ~28k steps in total (12k P-2 + 16k), and CC stacks five changes on top of the failed recipe. Fix: stage it.
   - S0: a $0 read-mask eval.
   - S1: cue-position reads only, long cue, warm from rcb.
   - S2: the short cue, warm from S1.
2. **BLOCKER (mechanism untested):** read_fixed adds S to every prompt row (deltamem.py:158-176), and no eval restricts reads. In CC every latent operand must enter through block-6 reads at the cue positions. Fix (S0): an eval-only read mask on rcb_plaw4_s1 / s2 (prompt rows read nothing), reporting accuracy by d and K′ and WS.
3. **BLOCKER (short cue):**
   - 33% of cued names already appear in the prompt as "X has <relation>", so " X has" is a literal prompt prefix (copy risk).
   - The askname skill sits at ':' and is frame-narrow.
   - Fix: measure the step-0 readout at " has" vs ':'. Consider " So <S> has" (3 tokens).
4. **FLOPs:** the arithmetic is broadly right. Per item at d 4:

   | Path | GFLOPs |
   |---|---|
   | CC | 13.98 |
   | TEXT, sentences | 18.73 |
   | TEXT with CC's scaffold | 14.85 |
   | **TEXT value-only (1 token / step)** | **13.47** |

   CC loses to value-only text CoT on 100% of items at every d. So the dominance argument depends on the text format. Scope the claim as "vs sentence and same-scaffold text CoT", or add a TEXT-VAL arm.
5. Several changes vs rcb, with no args diff; staging fixes this.
6. Gates must be rebuilt and redefined: CUE-ONLY on an EOT cache; NOT-DECODER (decoder = pass-d cue); SWAP-MID keeping the recipient's prompt KV; a new gate that prompt KV is independent of S; hold_inc re-implemented.
7. Build: prefix-cache masks, wpe offsets, padding invariance. Launch-bound, so ~1.5-2× wall-clock, not 10×; ~$0.7-1.0 per cell; G-TIME first.
8. Novelty: close to cached-prompt recurrent memory, pause tokens, Coconut / CODI. Project priors on cheap reads are poor (§171 / §198 / §42). C3 is fifth priority; justified only if S0 passes.
