# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-05 04:50 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (04:50 2026-10-05, night mode: user asleep, "make progress throughout the night"; credit $104 at 03:50, floor $30; <= 16 hosts)

**Order (user, via Audit):** C1 > C6 > C4 = C2 > C3 > C5 > C7. Night launch order: §117 -> §113/§118 -> §120 -> §116 -> §119 (credit check).

### Vast (all `scratch/gen_deploy_*.sh`, auto_drain_gen5 + `watch_any2.sh` waiters; ids/ssh in `scratch/live_hosts.txt`, `scratch/pods.txt`)

| Tag | Work |
|---|---|
| l1 | §114 LOOP-R s1 (then `loop114 --rel_only` locally) |
| s3, s4 | §101 b0.1 / b0.01 seeds |
| a1 | §118 ANCHOR s71-73 |
| a2 | §118 LONG s81-82 (proof host for LONG; a3 = s83 after it) |
| i2, i3 | §121 ANCHOR+READ s2-3 / READ s1-2 (i1 = s1 done: op3 0.231, fail) |
| o6 | §116 NONE+COPY s41-42 (proof host passed: 13.4 GB at P=2) |
| o4, o5, o7, o8 | §116 NONE 42-45 / NONE+COPY 43-45 (deploying 04:48) |
| j1 | §117 MK s51 + ST s61 (proof host, f = 0.1, note 117-3; deploying) |

**Queued:** j2 (MK s52-53), j3 (ST s62-63) after j1's 1k log + memory; §120 remaining 7 runs (P=2, 4 hosts, ~$3) after b1's
local eval_only is clean (fix 577ca96b: bf16 -> numpy); §116 o1 (OPDEF 31-32), o3 (OPDEF 35 + NONE 41) when slots free;
o2 (OPDEF 33-34) runs LOCALLY. §122 (accel99 `--textmix 0.25`) implemented, **conditional: launch only on Theory's call**.
§119 not implemented (spend projection to Audit first).

**Data retention (user, 2026-10-05 via Audit):** keep eval / log / args / preds and all logs; keep a final model.pt only
where a later analysis can use it (passing / acquired seeds, models named by a registered analysis); delete transfer
tarballs, staging, ckpt.pt after verification and per-round snapshots unless a registration needs them; record every
deletion (logs/prune_rounds.log); ask Audit / Theory before any new bulky artefact. One-off sweep of closed sections pending
(model.pt candidate list to Audit / Theory first). **02:14 sweep:** Audit+Theory OK'd scratch/retention_candidates.txt, but the
permission classifier blocked the deletes -> **USER DECISION PENDING** (delete 19 void .pt 7 GB; delete D: side of 266 B .pt
91 GB once archived). Copy-only archive running: scratch/retention_archive_copy.sh -> E:\LatentArchive + MANIFEST.txt (sha256).
* Watchers don't detect host reboots: check uptime on the C1 hosts each pass.

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070)

* 4060: §116 OPDEF s33 (`scratch/c1_q_116_local.txt`, py_local_worker2).
* 3070: §120 b1 `fsq120 --eval_only` (boxes/runs_fsq/s120_s5_final_s1), then §116 OPDEF s34 from the same queue.
* 2070: §115 on READ s9-10. g105 s1 §115 OOMs at the same 5.56 GiB at bs 125 and 60 (trigger_g97 ignores --bs?); pending.
* §115 done: READ s8 no-snap 2048 = 1.0; MK-4 s1/s3/s5, READ s6/s7 no-snap 0; trigger 1.0 everywhere.

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
