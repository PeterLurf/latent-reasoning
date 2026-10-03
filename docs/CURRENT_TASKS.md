# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-03 19:55 EDT** (local = UTC − 4; Vast hosts log UTC).

**Read first in a new session:** this file → `docs/NORTH_STAR.md` → the tail of `docs/THEORY.md` (registrations and
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

## Running now

### Vast (credit $72.52 at 19:55; 7 hosts, ≈ $1.3/h)

Every host is deployed by `scratch/gen_deploy_9{7,8}.sh`, has a watcher `scratch/auto_drain_gen3.sh`
(log `logs/auto_drain_gen_<tag>.out`), and drains automatically (fetch, verify, destroy, `scratch/pods.txt`).

| Tag | Instance | ssh (`-i ~/.ssh/runpod_ed25519 root@…`) | Work |
|---|---|---|---|
| x4 | 54067029 | `-p 40062 96.20.36.104` | §97 C6: GEN-4 s4–5 → `boxes/runs_s5/g97_gen4_s{4,5}` |
| x5 | 54067034 | `-p 41236 83.27.30.232` | §97 C6: HYB-4 s4–5 → `g97_hyb4_s{4,5}` |
| y1 | 54069464 | `-p 6026 142.127.68.223` | §98 GEN-M 4x4 s1–3 → `boxes/runs_mult/m98_genm_4x4_s*` |
| y2 | 54070485 | `-p 22496 142.113.28.123` | §98 GEN-M 5x5 s1–3 |
| y3 | 54070493 | `-p 42229 167.179.138.57` | §98 MASK 4x4 + 5x5 s1–3 (A5000) |
| y4 | 54070499 | `-p 40069 96.20.36.104` | §98 COCO 4x4 s1–2 only (see below) |
| y5 | 54073316 | `-p 22815 142.113.28.123` | §98 TC-M + NC-M 4x4/5x5 s1–3 (12 runs, about 10 h) |

* **Check a host:** `ssh … "cd /workspace/Latent; tail logs/queue_gen_<tag>.out; nvidia-smi"`.
* **Manual drain:** `bash scratch/drain_gen.sh <ip> <port> gen_<tag> <instance>`.
* **Destroy:** `bash scratch/vdestroy.sh <instance>`.
* **Watchers exit silently** on DRAIN FAILED, so keep a waiter on their logs (`scratch/watch_any.sh` in the session
  scratchpad does that).
* **y4:** its 4 other COCO runs OOM'd at start. Its drain will report FETCH_INCOMPLETE. Fetch ONLY `m98_coco_4x4_s1/s2`
  and `logs/` (the OOM logs as `*.void`), then destroy. The failed runs' directory names collide with the local reruns.

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070; nvidia-smi numbers differ: 1, 2, 0)

| Card | Now | Next |
|---|---|---|
| 4060 | §89 fade_raw s3 (last fade run) | §98 COCO reruns (`scratch/c1_q_98coco.txt`, `--micro 4`) via `scratch/s98_local_chain.sh cuda:0` |
| 3070 | §89 fade_soft s2 (last fade run) | the same COCO queue via `s98_local_chain.sh cuda:1` |
| 2070 | §83 s2/s3 primaries + extras, fp32 (`scratch/s83_local.sh` v2) | free |
| CPU | §94 Step 0 probes on the 13 §94 models (`scratch/s94_probes_new.sh`, Idle priority) → `theory/results/s94_step0/` | — |

* **Workers:** `scratch/py_local_worker2.sh <dev> <queue>` (lock dirs `c1/runs/locks/<run>`; log `logs/c1_local.log`).
* **To stop a job:** stop its python and its worker together. Add `-notlike '*shell-snapshots*'` to every
  Get-CimInstance kill filter.
* **The fade line is closed** (user, 19:15). The remaining §89 seeds are dropped and locked; no more fade runs.

## Waiting to be scored (Theory scores; Experiments reports when all runs are in)

* **§97 C6:** GEN-4 and HYB-4 s4–5 (x4, x5). The C6 rule: ≥ 4/5 seeds at ≥ 0.99 at 512.
* **§98:** 30 runs (`theory/score_s98.py`, keys fixed). The SOTA bar is ICoT-SI, our greedy eval: 4x4 0.998, 5x5
  0.996 (`boxes/runs_mult/icotsi_*`).
  - Automatic 4× follow-up (96k steps) only where M1 holds and M-SOTA misses (note 98-1); Theory says if it triggers.
* **§89:** final F1 from the arms in hand once fade_soft s2 lands (`python c1/score_89.py`).
* **§83 s2/s3** (descriptive; s2/s3 fp32, s1 bf16).
* **§94 Step 0 probes** on the new models (descriptive).

## Recently done (details in THEORY / PROTOCOL12 / git log)

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
