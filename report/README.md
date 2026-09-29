# report/ — index

Written 2026-09-28. Files stay at these paths (the build scripts and several docs refer to them).

| File | What | Covers |
|---|---|---|
| `findings.md` | Consolidated write-up: findings 1–10, revisions, open questions, caveats | Parts I–VI |
| `latent_branch_report.html` / `_print.html` / `.pdf` | Full report page (screen and print versions) | Parts I–III |
| `protocol4_report.html` / `.md` / `.pdf` | Part IV report | Part IV |

No report page exists yet for Parts V–VI; their scorecards are `docs/PROTOCOL5_RESULTS.md` and
`docs/PROTOCOL6_RESULTS.md`, summarised in `findings.md`.

## Building

| Script | Reads | Writes |
|---|---|---|
| `build_report.py` | `results/summary.json` (+ other `results/*.json`), `template.html`, `render.js` + `render2.js` + `render3.js` (concatenated in that order) | `latent_branch_report.html` |
| `build_protocol4_report.py` | `results/protocol4_confirm.json`, `protocol4_variants.json`, `confirm_sweep*.json` and others | `protocol4_report.html`, `.md`, and `.pdf` (headless Edge/Chrome, if found) |

`render*.js` are the page's chart/prose code for Parts I, II and III respectively; `scratch/patch_*.py` are the one-off
edits that built them up (historical).
