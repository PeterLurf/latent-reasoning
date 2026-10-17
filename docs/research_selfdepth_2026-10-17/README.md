# SELF-DEPTH-CHAIN research (2026-10-17)

- SELFDEPTH_draft.md: Opus subagent draft ($0): self-training by depth with D3-T's re-rooted single-step chain as the labeller (KEEP = parses + exactly n steps; COMPOSE-VALID for R2), round 1 = FT at n 9-12 vs FT-CTRL; S0 support boundary moves (default prediction SHIFT); readings DEPTH-BEYOND-SELF / ABSORB / STOP-MOVED + kept LENGTH-FREE / VB-C1 / K24 / KEEPS-SELF-STAB; OC P(SD-SELF) ~0.45, P(SD-C2) ~0.075; cap $1.5.
- scripts/: CPU checks and fixtures (composition gate, schedule / noise numerics).
- Precondition data: THEORY notes 213-33..213-39 (CHAIN-LABEL, CHAIN-LABEL-FRESH).
- PROOFS_SELFDEPTH.md: rigour pass (S0 / GATE-q / OC hold; fixes: STOP-LOC unreadable under H_shift, fresh pool undersized, cost/mem inputs (padded T, AdamW), COMPOSE-VALID premise, S2 numbers, trainer spec gaps); scripts/proofs_sd/
