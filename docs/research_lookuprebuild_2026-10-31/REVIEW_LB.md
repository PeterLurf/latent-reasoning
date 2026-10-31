# REVIEW_LB: red team of LB_draft.md (Opus subagent, 2026-10-31). Verdict: GO-with-fixes (2 blockers). All adopted in LB_v1.md

1. **No build is needed for perm on TOPO.** The assert at inject_aw.py:224 has no law condition, and perm_render asserts the spine order. CPU check: TOPO TEST2 sha matches; 600 train_perm draws solve and are disjoint from TP_TEXTS. Forced mul_s only acts at d ≥ 3, so it is outside Stage 1. Replaced the builds with a G-D smoke.
2. **BLOCKER:** part_test_perm uses ks = {d−1, d, K}, so d 4 never gets K′ 2 (eval:1050). Add K′ 2 at d 4.
3. **BLOCKER:** add test_perm to eval_parts in both stages (args diff), name the fields, and write score_lb.py before any output. Done: fixtures 9/9.
4. **Step-0 gate:** askperm's 0.988 was measured on REG no-wrap items only. Require step-0 ws['1'] ≥ 0.95 on all 1,000 TEST-PERM d 2 items.
5. **Stage-1:**
   - split STOP into NOT-FORMED and POSITION-REEMERGED;
   - aw_plaw2 formed at 6–7k, so 12k leaves ~5k of margin;
   - add p2p_s2 in parallel.
6. **Keep perm_p 0.5.** 1.0 would make d 2 a new positional rule. The cue names S_t at every pass, so the lookup key is always present.
7. **Attribution:** the NP pilot is not a control for "the init matters" (warm from a formed positional loop, 2k). Add P2P-OLD (askname + perm, Stage 1 only) or drop the causal init wording. Added.
8. **Labels:** add a first-match order, plus LOOKUP-NOCAUSAL and DECODER; copy STABLE.
9. **Stage-2 args:** init_trunk is overridden by init_model's strict load. Set it to askperm for provenance and assert init_model_sha.
10. **Cost and box:**
    - ~$0.68 per P2P and ~$0.8 per LBR; with the additions ~$4–4.5, guard $5;
    - binding evals from the cloud only (local box faults); --keep_final_ckpt.
11. **Novelty:** a real C1 strengthening. Caveat: every test order is a trained order. Related, to be checked before citing: Chen et al. 2024 "Premise Order Matters"; Allen-Zhu & Ye Physics of LMs 2.1 (iGSM random sentence order); Berglund et al. 2023 (reversal curse).
