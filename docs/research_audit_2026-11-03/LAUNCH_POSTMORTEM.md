# Launch post-mortem, 2026-11-01 .. 11-03 (Audit research subagent, 16:24 11-03 EST, tree e0cdf0e7)

Sources: `git log --since=2026-11-01`; `logs/s258..s270_admitted.log`, `*_bad_cost.log`, `fill_s267.out`, `fill_s269.out`, `gen_deploy.log`; `scratch/s270_fill.sh` (template v5), `check_queue_ships.py`, `check_py311.py`; fixtures in `scratch/s267_build/`, `scratch/fill_v5_build/`.

## 1. Failures since 11-01

Classes: SHIP (host file never shipped), HOSTENV (host Python / tree differs), GATEMEM (gate line OOM), CAPS (hand-set admit / guard / PMAX / weights), DRAIN (spend after QUEUE_DONE outside the cap), BUILD (edit bug seen only on the host), RESIDUE (voided attempt's watchers alive), HOST (BAD / vanished host). "Prevented" = a v5 refusal or an mkq predeploy assert catches it before create.

| cell | symptom | class | $ wasted | fix | prevented? |
|---|---|---|---|---|---|
| b394 dch_s2 | launched at $0.2174/h, no PMAX (default 0.25) | CAPS | 0 | 0016b698 | yes |
| b395 dis_s2_hx259 | rc 1: `--model path=sha` init unshipped (auto_ship takes plain paths) | SHIP | ~0.03 + redo | 1c5a57d1, 77fb37d9 | yes (check_queue_ships) |
| b396 bridge_s1 | drain fetch past the $1.55 guard; raised by hand to $1.75 | DRAIN | ~0.01 | ledger note | **no** |
| s260 fam guard | every ledger cell projected at this filler's STEPS (100) | CAPS | 0 | 201d630a / f1c7af76 ledger_steps | yes (v3) |
| b397 anchor a1 | gates rc 1: ref tar `boxes/*.py` pathspec carried run dirs, FileExistsError | HOSTENV | ~0.03 | 201d630a | partly (gates first + gates_pass.json; no fake-host test) |
| b397 hostprobe | 3090 limit 0.854 refused an A6000 at 0.987 | CAPS | 0.100 | 2e17bb53 | partly (per-mkq header) |
| b398 a1 / b399 | step0 gate crashed: inline `# note` after `;` commented out `P = np.zeros` (hh_dump); TRAIN refused; b399 destroyed mid-deploy | BUILD | 0.250 + 0.074 | 2c1aeb6c | **no** (CPU smoke skipped that part) |
| b401 p2m_s1 | medium model.pt + ckpt_final (~4 GB) fetch past guard; raised to $3.80 | DRAIN | 0.25 over admit | ledger note | **no** |
| b402 a1 | TRAIN refused: `--blob` sha jsons unshipped | SHIP | 0.089 | f9060c41 | yes |
| b402 a1 residue | its post_admit, not killed at the void, HOLD'd 150 min later: touched the stop file, vdestroyed a gone instance | RESIDUE | 0 (luck) | 0ab09806 note | **no** (refuse() kills only the drain) |
| b402 log | long line overwrote the train line's traceback | tooling | 0 | e13fc559 | yes |
| b406 p2m_s1b | medium drain past guard; raised to $4.20 | DRAIN | 0.51 over admit | note | **no** |
| b409 pilot_gb_s1 | drain looped FETCH_INCOMPLETE: stop_at dir has no model.pt but its step-1200 record matched the s224 RUN marker; guard to $0.60 | DRAIN | ~0.13 | 9418f9ac | marker yes; drain budget no |
| b412 a1 / a2 | BAD host 51.89.50.15; then $0.2177/h projected $0.45 > cap $0.40 | HOST / CAPS | 0.200 | 09c1f3b9 | yes (PMAX <= admit / hours) |
| b412 a3 | gate_s268 `os.path.isjunction` (py 3.12) on 3.11 hosts; gates crashed, TRAIN refused | HOSTENV | 0.176 | 6099e83a, 550ddedc check_py311 | yes |
| b412 a4 | 3 BAD hosts + a $0.2567/h offer that could never pass admit: BADMAX hold | HOST / CAPS | 0.215 | 09c1f3b9 | partly |
| b412 a5 | gates line died: `scratch/s259_guarded.py` --blob'd but unshipped once the step-0 line was dropped | SHIP | 0.102 | 3b44605a | yes |
| b413 a1 | gates OOM: `--micro 1` = whole 64-item batch in fp32; gradflow beside 2.6 GB held by the parent | GATEMEM | 0.215 | 4f1b76a9, bdbd049c | **no generic check** |
| b414 a1 | wall guard STOP at $0.503 >= $0.50 with 4/5 lines: cap lacked gate hours; 2k ckpt lost, no evals | CAPS | 0.503 | 4842bf1c GATES_H (v5) | partly (operator env, default 0) |
| b414 rerun | TRAIN --blob listed gate-only files that would not ship | SHIP | 0 | 5aa285be | yes |
| s267 / s269 fam() | a VOIDed attempt's realised line replaced the relaunch projection ($0.336 vs ~$1.06) | CAPS | 0 | 3e57f66a v5, fixture 3/3 | yes from s270 |
| b413 / b414 / b416 | vanished / BAD hosts; one create without INSTANCE line (58949488) | HOST | 0.198 | - | n/a |

## 2. Spend

Landed cells b393..b414: **$19.39 useful** (b415 in flight). Voided attempts **$1.34** (b398a1, b402a1, b412a3, b412a5, b413a1, b414a1). BAD / refused ledgers **$1.14** (projection / hostprobe refusals $0.41, BAD or vanished hosts $0.74). Drain overruns inside landed cells **$0.90** above admit (b396 / b401 / b406 / b409): useful fetches, but outside any approved cap until a manual raise. Total ~$21.9; hard waste ~$2.5 (11 %): ~$1.75 preflight-preventable, ~$0.74 host luck. Three families (s260, s267, s269) needed 2-5 relaunch approvals each.

## 3. Five classes still unprevented

1. **DRAIN budget** ($0.90, four manual raises). `s270_fill.sh:117` projects setup + steps + gates + eval; the wall guard STOPs on spent >= CAP even after QUEUE_DONE; the fetch (4 GB medium at $0.45/h) is in neither. Change: mkq writes `# FETCH_GB <n>` (from `--keep_final_ckpt`, `--base`, NOFETCH lines); the filler adds `FETCH_GB / rate` hours to `proj` and to the last WALLW weight; `s232_wall_guard.py` switches to a *draining* mode after QUEUE_DONE (ALERT, not STOP, until `logs/drain_gen_b<n>.out` says DRAINED / DRAIN FAILED). Test: the guard already takes `GUARD_VASTCLI` / `GUARD_VDESTROY` stubs; a fixture with a fake queue log reaching QUEUE_DONE at spent > CAP asserts no vdestroy until a fake drain log says DRAINED.
2. **Queue dry-run on a fake host** (b397a1, b398a1 / b399, b412a3, b412a5: $0.63 and two parked sections). Every voided gate run failed at import / argparse / first file open. Change: `scratch/s270_build/fixture_queue_dryrun.sh` builds `scratch/_tmp/qdry/workspace/Latent` from exactly the shipped set (plain tokens + `check_py311.files_of`, tarred as `gen_deploy_120_s267.sh` does) and runs each line's module there with `CUDA_VISIBLE_DEVICES=` and `--steps 2` / `--dry_run`; any wrapper that exits before its first part fails. Wire it beside `check_queue_ships` in `sNNN_mkq.py` (`s270_mkq.py:69`). Pattern: `fixture_gen10.sh` (sed ssh / scp into `bash -c` / `cp`).
3. **Gate-line memory** (b413a1; §265 asks for mem_check at d4+8 by hand). The `_mem` line measures only the training shape. Change: the gate wrapper (`s269_gates.py` pattern) runs each part in its own subprocess with `--mem_budget` from the `_mem` result; `check_queue_ships` refuses a gates line whose bitwise parts lack `--micro`. Test: on the 4060 with a fake part that allocates N GB; assert isolation and a clean message, no parent OOM.
4. **Void residue** (b402a1; memory "Voiding an admitted cell"). `refuse()` (`s270_fill.sh:48`) kills only the drain. Change: `scratch/void_cell.sh <fam> <n> <k>` kills post_admit, `cost_guard.py sNNN_b<n>`, the wall guard and the mirror by argv, rotates their logs, retags the realised line `:a<k>`, moves the queue to `_attempt<k>.txt`; both the filler's void path and the operator call it. Test: spawn `sleep` with those argv, void, assert none left and the ledger tagged.
5. **Hand-typed caps** (b414a1; b412's five CAPF / admit edits; PMAX 0.21 / 0.25 drift; the A6000 limit). WALLW, EVAL_H, GATES_H, SPS_EST, HOSTPROBE live on the chain command line; only the WALLW count is checked (`s270_fill.sh:81`). Change: mkq writes one `# CAPS wallw=.. eval_h=.. gates_h=.. sps_est=.. hostprobe=.. class=..` header derived from the lines (gates line -> gates_h > 0; medium -> class limit); the filler REFUSES (rc 2) when env and header disagree. Test: mismatch fixture expects rc 2; a match reaches offer_pick (stub returns nothing).

## 4. Launch checklist (Experiments, every new family)

1. Tree clean (`git status --porcelain boxes/ scratch/`, run dirs aside); queue, frozen args, mkq committed.
2. `python scratch/sNNN_mkq.py` -> `predeploy_ok true`; it calls `check_queue_ships` (ships + py3.11) on every `.hold`; compare the "ships OK (n paths)" count with the models / inits / data blobs expected.
3. After removing any line (NO_STEP0, NO_GATES): rerun `check_queue_ships`; diff TRAIN's `--blob` list against the previous queue.
4. Gates line: bitwise parts `--fp32 --micro 8`; in-process parts last, cache freed; wrapper run once locally with the real part list; `gates_pass.json` only when all pass, `--require`d by TRAIN.
5. Any edited part or trainer path: a CPU smoke that executes *that* part on the shipped tree, not only the plain path.
6. Caps: admit = (STEPS x SPS_EST / 3600 + 0.4 + GATES_H + EVAL_H + fetch h) x PMAX; guard >= admit + 0.05; CAPF >= cells + $0.20 BAD per attempt; PMAX <= admit / hours; medium / A6000 cells carry their class HOSTPROBE limit.
7. WALLW: one weight per non-comment line; gates weight = GATES_H; last weight includes the fetch; `GATES_H`, `EVAL_H` exported to the chain.
8. Drain: every unneeded `.pt` is `# NOFETCH`; `--keep_final_ckpt` only for a planned resume; stop_at dirs not `*_eqA` / `*_eqB`.
9. Ledger: fresh `logs/sNNN_admitted.log` header; family guard log rotated; no ALERT / STOP left.
10. ADMIT + 120 s: cell, wall, family (every cell with steps) guards and post_admit alive with `start:` and no Traceback; `logs/gen_<run>.log` head shows free / df / GPU memory.
11. On a void: kill post_admit, mirror, cell + wall guards by argv; tag the realised line `:a<k>`; keep `gen_q_b<n>_attempt<k>.txt`; note CAPF and spent so far.
12. Relaunch only after the root cause is in a commit message with the failing log path (as 3b44605a).
