# SELF-DEPTH-CHAIN research (2026-10-17)

- SELFDEPTH_draft.md: Opus subagent draft ($0): self-training by depth with D3-T's re-rooted single-step chain as the labeller (KEEP = parses + exactly n steps; COMPOSE-VALID for R2), round 1 = FT at n 9-12 vs FT-CTRL; S0 support boundary moves (default prediction SHIFT); readings DEPTH-BEYOND-SELF / ABSORB / STOP-MOVED + kept LENGTH-FREE / VB-C1 / K24 / KEEPS-SELF-STAB; OC P(SD-SELF) ~0.45, P(SD-C2) ~0.075; cap $1.5.
- scripts/: CPU checks and fixtures (composition gate, schedule / noise numerics).
- Precondition data: THEORY notes 213-33..213-39 (CHAIN-LABEL, CHAIN-LABEL-FRESH).
- PROOFS_SELFDEPTH.md: rigour pass (S0 / GATE-q / OC hold; fixes: STOP-LOC unreadable under H_shift, fresh pool undersized, cost/mem inputs (padded T, AdamW), COMPOSE-VALID premise, S2 numbers, trainer spec gaps); scripts/proofs_sd/
- REDTEAM_SELFDEPTH.md: red team (design holds; COMPOSE-VALID == KEEP measured on real decodes, state precision 0.99975; FT-SELF ~ gold per-step FT; odds generous (shaded P(SD-C2) ~0.03); 6 blockers: pool size, cal under init_from, mass_stats gold read, STOP-LOC, state-level LABEL-AUDIT, streams/gates); scripts/rt_sd/
- SELFDEPTH_v1.md: registration-ready v1 (24-row change table: every PROOFS / REDTEAM / Audit / Literature fix; contamination rules; readings + field contract; OC headline P(SD-C2) ~0.032; gates G-S0..S15; cap $1.5 with admission bounds); scripts/v1_checks/
