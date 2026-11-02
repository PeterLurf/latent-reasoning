# REVIEW_LB2: red team of LB2_draft.md (Opus subagent, 2026-11-01; no model.pt loaded). Verdict: GO-with-fixes. All adopted in LB2_v1.md

1. **BLOCKER (cost):**
   - Compute scales with non-embedding params (small 85M → large 708M = 8.3×, not 6.2×).
   - S3 is the worst activation shape: rcb d 4 22.5 GiB at micro 1; dis ~28 GiB at micro 1, so ~140 GiB at large, micro 8.
   - The loop is launch-bound. lb_cost.json puts large at ~10 s/step on a 3090. So large S0-S3 ≈ $18-27, and R does not fit.
   - Fix: G-TIME at the S3 shape; medium as primary (≈ 2.3× cheaper than large).
2. **BLOCKER:** S1's perm gate tests a property S2 erases (Note 254-3: 0.996 → 0.097), and dis's lineage used askname without perm. Fix: S1 = the s233 recipe exactly.
3. **BLOCKER:** merging rcb with distract in S3 is untested (perm blocked write formation, Note 254-1) and confounds scale. Fix: a small-base BRIDGE (aw_plaw2 → REG d ≤ 4 + distract, 16k, ~$1.1). If it fails, use two stages.
4. **BLOCKER (code):**
   - s5.load_gpt2 hard-codes the small checkpoint and H 12 (s5.py:120; core.py:110);
   - build_aw hard-codes block 6, heads 6+6 and an h.6 assert (inject_aw.py:93);
   - DeltaMem asserts (h_add + h_del)·64 = D;
   - hh_* hard-code 768.
   - Fix: a --base arg, layer L/2, heads H/2 + H/2, a bitwise equivalence gate on small, a plain-path smoke, host-side loads, off-host ckpts.
5. **BLOCKER:** "improves with scale" had no test. Add IMPROVES (d 8 ≥ 0.70 or d 7 ≥ 0.95) vs the bridge and dis; name the eval fields; S2 pass 2 uses the eval WS field.
6. **Minor:** one seed. Label it PILOT, with a pre-stated seed-2 rule; disclose the GPU class.
7. **Minor:** S2 formation rule (read at 6k / 12k; one +6k extension).
8. **Minor:** lr 1e-4 is high at scale (GPT-3 ratio → ~4e-5). Bracket {3e-5, 1e-4} in S1. DeltaMem init needs no rescale.
9. **Minor:** text-CoT R is not essential for C2 (internal). Demote it to a separate design / R-small first.
10. **Base:** GPT-2 medium / large is right for a single-factor scale test; Qwen is a follow-up. Lean to medium as primary.
