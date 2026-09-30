# docs/ — index

What each file is, and whether it may be edited. Files stay at these paths: code, scripts, other docs and Claude
memory refer to them by path (see `../DO_NOT_MOVE_WHILE_RUNNING.md`). Index written 2026-09-28, updated 2026-09-29 ~13:05 (PROTOCOL7, theory stream, audits).

## Start here

| File | What | Editing |
|---|---|---|
| `CURRENT_TASKS.md` | Live handoff: what is running, where, how to check/fetch/stop it, what awaits scoring | **Live**: the session running the experiments keeps it current |
| `RESEARCH_LOG.md` | Chronology of every decision, run, deviation and disclosure (all parts) | **Live**, append only |
| `RESEARCH_TIMELINE.md` | Why each part asked what it asked; claim ledger; next steps (Parts I–VI) | Narrative; no new results |
| `../report/findings.md` | Consolidated write-up: what stands, what was revised, what is open (Parts I–VI) | Narrative; no new results |

## Pre-registrations and scorecards, by part

Pre-registration files (`PROTOCOL*.md`) are **append-only**: predictions are never rewritten, only followed by dated
amendments.

| Part | Dates | Pre-registration | Frozen values | Scorecard |
|---|---|---|---|---|
| I | 09-23 | `PROTOCOL.md` | `protocol_frozen.json` (read by `latent/study.py`) | `../results/summary.json`, `../report/latent_branch_report.html` |
| II | 09-24 | `PROTOCOL2.md` | `protocol2_frozen.json` (read by `latent/followup_confirm.py`) | `../results/followup_confirm.json`, `../report/latent_branch_report.html` |
| III | 09-24 | `PROTOCOL3.md` (addenda B–K) | — | `../results/protocol3_confirm.json`, `../report/latent_branch_report.html` |
| IV | 09-24 → 09-25 | `PROTOCOL4.md` (addenda A–E) | — | `PROTOCOL4_RESULTS.md`, `../report/protocol4_report.html` |
| V | 09-25 | `PROTOCOL5.md` (addenda A–C) | — | `PROTOCOL5_RESULTS.md` |
| VI | 09-25 → running (R) | `PROTOCOL6.md` (addenda A–R) — **live** | — | `PROTOCOL6_RESULTS.md` — **live** |
| VII | 09-28 → running | `PROTOCOL7.md` (confirmation suite T1–T13, amendments) | — | `PROTOCOL7_RESULTS.md` |

Notes: `PROTOCOL6.md` has an empty "Results — Pending." heading between addenda H and I; the Part VI results are in
`PROTOCOL6_RESULTS.md`. ProsQA (Part VI addenda C–N) code and per-seed results live in `../prosqa/` (`res_*/`).

## Literature

| File | What |
|---|---|
| `LITERATURE_NOTES.md` | Short positioning note (2026-09-26): closest prior work, the sceptical line, what is new here |
| `LITERATURE.md` | Full annotated bibliography grouped by question, with a novelty-status section (updated 2026-09-27) |
| `LITERATURE_AUDIT.md` | Targeted novelty check against prior work (2026-09-28; abstract-level, not exhaustive) |
| `LIT_REQUESTS_2026-09-29.md` | Literature session's answers to agent requests: anytime-supervision novelty, random-budget precedents, checkpoints and deep datasets, PROTOCOL9 novelty (Experiments); citation check, prior work for §26/F5a/F1a, pre-registration reporting standards (Audit) |
| `IDEAS_ADJACENT_2026-09-29.md` | Literature session: 12 ideas to borrow from adjacent fields (NCA pool training, periodic clocks, normalisation and working-memory models, Neural Data Router, Hopfield clean-up, theta cycling, energy-based reasoning …), each mapped to C1–C4 with a cheapest test |
| `ADVISOR_2026-09-29.md` | Literature session's advisory report: papers supporting or scrutinising each claim of the experiments and theory streams, with suggested actions |
| `literature_dois.txt` | Bare DOI list (94; 85 arXiv DOIs), one per line (last changed 2026-09-27); not cross-referenced in `LITERATURE.md` |

## Theory stream (separate Claude session, from 2026-09-29)

| File | What |
|---|---|
| `THEORY.md` | Why some models fail to extrapolate: mechanism analyses (exploratory) and its own pre-registered pilots A–E; code and results in `../theory/` |
| `PROPOSALS.md` | Proposals for improving latent reasoning, drawn from the mechanism results and the literature |

## Audits

| File | What |
|---|---|
| `AUDIT_2026-09-29.md` | Independent re-check of the scored verdicts (all reproduce) and a list of record-keeping fixes |

## Historical planning (kept because other files cite them)

| File | What |
|---|---|
| `NEXT_STEPS.md` | The user's 24 Sep handoff that framed Parts II–III |
| `TRAINING_TODO.md` | The 24 Sep training plan, run as Parts IV–VI |
