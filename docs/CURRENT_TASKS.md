# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-05 09:05 EDT** (local = UTC − 4; Vast hosts log UTC).

**Read first in a new session:** this file → `docs/NORTH_STAR.md` → `docs/THEORY.md` (§97 on; §0–§96 verbatim in `docs/archive/THEORY_s0-s96.md`, search "## N.") (registrations and
results, § numbers) → `docs/PROTOCOL12.md` (experiments-side results) → `docs/README.md` (index).

## Sessions

| Session | Owns |
|---|---|
| **Experiments** | implements and runs registered THEORY § designs; local GPUs; Vast fleets; fetch, commit, report |
| **Theory** | registrations and scoring (`theory/score_s*.py`) |
| **Audit** | reviews, spend approvals, 20-min ticks |
| **Literature** | prior-work checks (`docs/LIT_*.md`) |

Messages go through SendMessage. The names "Theory" and "Audit" currently need an identity confirm, pending the user,
so send to the pipe addresses their messages come from.

## Running now (09:05 2026-10-05, night mode; credit ~$95, floor $30; <= 16 hosts; gradflow_check PASS required for new code paths)

**Order (user, via Audit):** C1 > C6 > C4 = C2 > C3 > C5 > C7.

### Vast (all `scratch/gen_deploy_*.sh`, auto_drain_gen5 + `watch_any2.sh` waiters; ids/ssh in `scratch/live_hosts.txt`, `scratch/pods.txt`)

| Tag | Work |
|---|---|
| s3, s4 | §101 b0.1 / b0.01 seeds |
| a2 | §118 LONG s82 alone at k=32 (s81 OOM'd here at P=2) |
| a3 | §118 LONG s81 RESUMED from ckpt 18000 (shipped; `gen_deploy_118r.sh` RESUME_FILES), P=1 |
| a5 | §118 ANCHOR extension s74-75 (note 118-4) |
| b3 | §120 FF FINAL s1 + S5 TRACE s2 (**SIGSTOPped at 12.7k**; on-host watcher SIGCONTs after FF FINAL s1's eval) |
| b4, b5 | §120 FF FINAL s2 + FF TRACE s1 / FF TRACE s2 |
| b12 | §120 S5 TRACE s1 rerun from scratch (OOM at P=2 -> *.void_oom), P=1 `--micro 2` |
| i4, i5, i6 | §122 TEXTMIX+ANCHOR+READ s1 (proof, t1 rising) / s2-3 / control TEXTMIX+READ s1-2 |
| j1, j3 | §117 MK s51 + ST s61 / ST s62-63 (MK s52-53 done: 0/0) |
| o4, o5 | §116 NONE s42-45 (NONE+COPY VOID: no_grad bug, note 116-5; o7 stopped) |

**Queued (slot order, Theory + Audit 09:15; each step only while burn < ~$3.5/h, 3090s only):** (1) finish running work
incl. a4 = §118 LONG s83 (P=1, 24 GB); (2) §123 once §120's pilot reads, on Theory's call (S5 dose b6-b9 first, then
FSQ+DITHER b10-b11 if credit binds; FF σ* later); (3) §116 OPDEF o1 / o3; (4) §119 deferred.

**Data retention (user, 2026-10-05 via Audit):** keep eval / log / args / preds and all logs; keep a final model.pt only
where a later analysis can use it (passing / acquired seeds, models named by a registered analysis); delete transfer
tarballs, staging, ckpt.pt after verification and per-round snapshots unless a registration needs them; record every
deletion (logs/prune_rounds.log); ask Audit / Theory before any new bulky artefact. One-off sweep of closed sections pending
(model.pt candidate list to Audit / Theory first). **Sweep DONE (user approved in Audit's session, ~10:30):** 15 void .pt + 266 D: originals deleted (E:\LatentArchive keeps
the sha-verified copies, MANIFEST.txt); logs/retention_delete_20261005.log. Restore = copy back from E:\LatentArchive\<same path>.
* Watchers don't detect host reboots: check uptime on the C1 hosts each pass.

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070)

* 4060 / 3070: §116 OPDEF s33 / s34 (`scratch/c1_q_116_local.txt`, py_local_worker2).
* 2070: §115 on READ s10 (Theory wants the no-snap result), then `loop114 --rel_only` on a114_loopr_4x4_s1.
* Post-hoc for every drained §120 / §123 FSQ cell: `fsq120 --growth_only` and `--cert_only` (45 min) locally.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **§97 C6:** complete (GEN-4 and HYB-4 s4-5 fetched).
* **§98:** ICoT-SI done (0.998 / 0.996); TC/NC done; MASK 4x4 + 5x5 done; COCO 4x4 s1-2 + 5x5 s1 in, rest local; GEN-M rerunning (r1/r2).
* **§99** kcurr/markov/ctrl done; **§100** MASK2 4x4 done; **§105** done (models local for Theory); **§106** pilot gate failed (106-2); others running (above).
* **§83 s2/s3** extras (local, fp32); **§89** soft s2 / raw s3 landed (descriptive; F1 final from the arms in hand).
* **§94 Step 0** on the 13 §94 models: done (5d07f7fa).

## Recently done (details in THEORY / PROTOCOL12 / git log)

* **§94 Step 0 on the new models** (note 94-3): the decoder arms are linearly available 10/10; no decoder 1/3.
* **§94** W1, W2 met (budget fixes "where").
* **§95** foothold dose (scored).
* **§96** (Theory).
* **§97** G4 met: GEN-4 and HYB-4 3/3; GEN-1 certified.
* **§93** not met.
* **§94 Step 0:** the state is linearly available ≈ 1.0, controls at chance.

## Standing rules

* **Pre-register before running.** THEORY § and notes are committed before launch. PROTOCOL files are append-only:
  dated notes, never rewrites. Keep aborted-run logs (`*.void`, commit, "aborted unread").
* **Spend.**
  - Projection to Audit for > $5.
  - Keep ≥ $10 credit.
  - Prove one host first, then ≤ 5 per experiment and ≤ about 10 live in total.
  - ≤ about $0.34/h per host (log exceptions in `pods.txt`).
  - Replace hosts stuck loading > 15 min.
  - Log every create and destroy in `scratch/pods.txt`.
* **Vast API key:** stored only by the Vast CLI (`C:\Users\peter\.config\vastai\vast_api_key`). Never copy it into
  the repo, logs or memory.
* **Local GPUs:**
  - One GPT-2 job per 8 GB card; keep the cards busy.
  - The 2070 is Turing (no bf16 SDPA): fp32 only, and §89-sized fade jobs OOM there.
  - Memory-check the largest shape before raising concurrency.
  - CPU analyses: ≤ 3 threads, Idle priority.
* **The user permits** Experiments to stop and restart its own local jobs. Never stop a peer session's process on its
  behalf.
* **Git:** the repo is LF (Python on Windows writes CRLF unless `newline="\n"`). The worktree is shared: pull before
  editing docs and stage only your own files. Vast tarballs in `/tmp` (C: drive) are cleaned by
  drain_gen / vdeploy; check C: free space before big fleets.
* **No GSM8K training** beyond what is registered (user: small tasks first).
