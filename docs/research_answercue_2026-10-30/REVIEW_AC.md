# REVIEW_AC: red team of AC_draft.md (Opus subagent, 2026-10-30). Verdict: GO-with-fixes (3 blockers) plus a $0 pilot first. Theory: pilot first (Note 251-10)

1. **Mechanism: mostly right, overstated.**
   - The pass-d input is the decoder prompt with the final question repeated. It reads S the same way, so ACUE ≈ "decoder answer at K′ = d−1" (1.00 in every working loop).
   - One pass cannot do two steps (K′ 0 ≤ 0.25 everywhere).
   - It differs from the decoder CE: it trains the write source h at the ':'. In aex0_s2 the decoder at K′ 1 scores 0.99 while the pass-2 ':' scores ~0.3.
   - The loss is not already satisfied in the failing AO runs (WS d 2 pass 2 0.04–0.26).
2. **BLOCKER (claim):** the cell that drifts, pass t on d > t items, is never labelled. AO already keeps WS d 1 pass 1 at 1.00 while d 2 pass 1 falls. The only pressure on it is the one-hop path via pass d, which is "indirect pressure", not "requires". In aex0_s2 a one-hop decoder read did not form the pass-2 write for d 3 items. Fix: make the untrained readings binding (WS d 2 pass 1, d 3 pass 2 ≥ 0.8; probe u1|v1 ≥ 0.8).
3. **BLOCKER (gate):** "d 2 at K′ 1 ≤ 0.2" fails every success (working loops score 1.0 there). Use K′ 0 ≤ 0.2 plus a causal pass-1 gate (SWAP-1 by_k / LP5 at pass 1); keep NOT-DECODER at d 4.
4. **BLOCKER (comparator):** AO4 does not exist, and the draft changes law, depth and loss at once. Stage it instead:
   - (A) ACUE in the exact §244 recipe vs ao_s1–s4 (existing);
   - (B) ACUE in AEX0's recipe vs aex0_s1 / s2.
5. **C4 (disclose):** ACUE is the diagonal (t = d) subset of the aux targets, with the final answer as value. That makes it a qualified C4: final answers at the depth-matched pass, plan and d given. It is the same step-count leak as K = T(n).
6. **Novelty:** Fan et al. 2409.15647 (loss only at T(n), T(n) given) is the closest precedent; also deep supervision (TRM / HRM), ATC, §61 F1-S. The new part is narrow: the readout point is the recurrent write source under per-pass cues.
7. **Build:**
   - The loop calls m.write and discards h. Call pass_aw, then colon_logits for t = d, over the full vocabulary.
   - Default-0 bitwise gate under a det wrap; log ACUE acc by d; check STABLE caps.
   - λ = 1 adds gradient in the onset window.
   - Cost ~$1.15 per cell.
8. **$0 pilot first:**
   - 2k local steps on the §244 recipe with cue_ans 1. Read WS d 2 pass 1 at 1k / 2k vs the AO seeds' 0.22 / 0.09 / 0.16 / 0.27.
   - Rule: ≥ 0.8 → GO to stage (A); ≤ 0.3 → park.
   - Also 2–3k from aex0_s2, reading WS d 3 pass 2.
   - The local GPU class means the pilot predicts only.
