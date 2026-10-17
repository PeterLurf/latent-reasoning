# D4 research (2026-10-16)

- D4_draft.md: Opus subagent draft ($0) for the remaining depth limit after §213: computed PAUSE passes (branch A/B by D3-T EDGE-COLLAPSE / STEP-BY-T); derivation (Props D0-D2, Lemmas D3-D5), Goodhart, readings, OC, Stage 0/1.
- scripts/: s0_collapse.py, s0_finishpass.py, s0_clockkey.py, d4_edge_collapse.py (fixture-checked EC / DEADLINE-GONE / ORS reader), d4_schedule_stats.py, with outputs.
- PROOFS_D4.md: rigour pass on D4 §3 / §5 (D0 / D1 hold; D1 B-row, §2.1-vs-§2.2 schedule, Lemma D3 direction, D4 delta_0, D5 TBPTT fixed; DEADLINE-GONE and the slope falsifier do not separate a shifted deadline -> slope n 12-16 + START-D on n + D); scripts/proofs_d4/
- REDTEAM_D4.md: red team of D4 (feasible; 6 blockers: DG never-finish / shifted fixtures, decode-free readings, size-keyed stop clause, AGE-MCT arm, trailing-pause spec, n 13-16 C2 reading; N2 size-keyed; decode unfaithful beyond range; code traps); scripts/rt_d4/
- D4_v1.md: revision after PROOFS_D4 + REDTEAM_D4 + Audit (decode-free STOPS-AT-10 diagnosis; base D3-T; branch A by COMPRESSION; lastR schedule; pi-free claim readings; DEADLINE-BOUND ladder L1-L3; KEEPS-SELF-STAB beside D4-C2; PROGRESS-ONLY / SIZE-KEYED hard stops; Stage 0 = 213-27 AGE-PROJECT + S0-3 AGE-REVERT/AGE-MCT + S0-2 CARRIER; Stage 1 S1-A/B/C; P(D4-C2) ~0.06 per cell)
