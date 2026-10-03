# docs/ — index

One line per file. Rewritten 2026-10-03 by Audit at the user's request ("too much old detail; summarise").
Superseded one-off notes moved to `archive/` (unchanged, still in git); nothing that code loads by path was moved.
Pre-registrations (`PROTOCOL*.md`) are append-only and are never rewritten.

## Start here

| File | What |
|---|---|
| `NORTH_STAR.md` | The goal (C1–C7), one status line per condition, current design, open questions |
| `CURRENT_TASKS.md` | Live handoff: what is running, where, how to check/fetch/stop it |
| `THEORY.md` | Every registration (§1–§98+), its predictions, amendments and results; summary at the top |
| `PROTOCOL12.md` | Scorecards for the recent registrations run by Experiments (§61b onward) |
| `../report/findings.md` | Plain-language write-up of what stands |
| `../bench/LEADERBOARD.md` | Our models vs published numbers on the shared benchmark items |

## Pre-registrations and scorecards (append-only)

| Part | Topic | Files |
|---|---|---|
| I–III (09-23/24) | Toy graph search: branch edits, width, the clocked sweep | `PROTOCOL.md`, `PROTOCOL2.md`, `PROTOCOL3.md`, `protocol_frozen.json`, `protocol2_frozen.json` |
| IV–V (09-24/25) | Part III claims, reliability, extrapolation predictors | `PROTOCOL4.md` (+`_RESULTS`), `PROTOCOL5.md` (+`_RESULTS`) |
| VI–VII (09-25/30) | Why sweeps keep running; horizon confirmation suite | `PROTOCOL6.md` (+`_RESULTS`), `PROTOCOL7.md` (+`_RESULTS`) |
| VIII (09-29/10-01) | Pretrained Coconut GPT-2 on deep ProsQA; checkpoints | `PROTOCOL8*.md`, `PROTOCOL9*.md`, `PROTOCOL10.md`, `PROTOCOL11.md` |
| IX (10-01 →) | GPT-2 latent automata, S3–S5 tracking, GSM8K, latent restarts | `PROTOCOL12.md` (+`_TUNING`), `GSM8K_REPRO.md` |

## Logs and planning

| File | What |
|---|---|
| `RESEARCH_LOG.md` | Chronology to 2026-09-29 (decisions, deviations, disclosures); later events live in THEORY / PROTOCOL12 / git |
| `NEXT_STEPS.md` | Historical plan (09-30); superseded by NORTH_STAR's open questions |
| `EXTERNAL_IDEAS_RL_2026-10-01.md` | External agent's RL/acquisition proposals (basis of §69) |

## Literature

**`LIT_INDEX.md`** lists every literature note in one line each: synthesis notes first, then each per-registration
prior check by § with its verdict and key prior IDs. The per-note files stay in place because THEORY cites them by path.
`LITERATURE.md` and `LITERATURE_NOTES.md` are historical (to 09-27), and `LIT_NORTHSTAR_2026-09-29.md` holds the C1–C7
test criteria.

## Archive (`archive/`)

Superseded one-off notes, unchanged: advisor report and first audit (09-29), external-agent prompts and replies (09-30,
10-01 C1), adjacent-field ideas, literature audit and requests (09-28/29), new-directions seeds, proposals and the
boundary-memory proposal, research timeline (Parts I–VI), theory framework, training to-do, DOI list.
