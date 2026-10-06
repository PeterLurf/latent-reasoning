# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-05 21:00 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (21:00 2026-10-05; burn $2.23/h, credit ~$154, floor $30; <= 16 hosts; gradflow_check PASS required for new code paths)

**Order (user, via Audit):** C1 > C6 > C4 = C2 > C3 > C5 > C7.

### Vast (all `scratch/gen_deploy_*.sh`, auto_drain_gen5 + `watch_any2.sh` waiters; ids/ssh in `scratch/live_hosts.txt`, `scratch/pods.txt`)

| Tag | GPU | Work |
|---|---|---|
| s4 | 3090 | §101 b0.01 s3 (decisive seed; s1 = 0, s2 = 0.9995), ~10 h, at 4.5k |
| a2 | 3090 Ti | §118 LONG s82 (k = 32), at 23k |
| a4 | 3090 | §118 LONG s83, at 19k |
| j1 | A5000 | §117 MK s51 + ST s61 (both at loss ~9.3; MK s52-53 and ST s62-63 were 0) |
| b23 | 3090 | §123 FF σ* = 0.25 s1, at 24k (then local fp32 contraction, no `--std`) |
| b24, b25 | 3090 | §126 RAW-H s1 / s4 reruns, P = 1 (both pass the 2k flag; loss rising with L_max like s3) |
| c2 | A10 | §124 seed-28 trio (EXPOSE / COPY / BASE) |
| c3 | 3090 | §124 COPY s26-27 (+BASE; same machine type as c1's EXPOSE s26-27) |
| c4 | 4090 | §124 seed-29 trio (BASE, COPY evaluated; EXPOSE at L22) |

* Drains now keep failed runs' args.json + log.jsonl as `<dir>.void_partial` (99192404, Theory).
* §126: no more ANNEAL (126-8). §125 closed. §119 deferred. No unrun registration pending.

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070; py_local_worker2, log `logs/c1_local.log`)

* 4060 / 3070: note 112-5 resume runs (`scratch/c1_q_112r_local.txt`): f050 s24 done (0.968 -> 0.996); s110 s1 (4060) and f010 s22 (3070)
  running. f010 s22 stalled ~5 h at final-round step 4100 (15:38-20:45, 3070 contention; timing only, trajectory unaffected).
* 2070: §116 NONE s41 -> OPDEF s31 -> s32 (`scratch/c1_q_116c_local.txt`).
* Post-hoc for every drained §120 / §123 / §126 FSQ cell: `fsq120 --growth_only` and `--cert_only` (FF adds cert_task, cert_task_reads)
  and `--contraction_only --fp32` locally.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (21:29):** §123 closed: causal question not reached. **§126** waits on RAW-H s1 / s4 and the ANNEAL s2 fixed-cast run (b26). **§127** (entry panel: VAR / SKIP / STD2 / LR10, seed-2 proofs, delivery checks 127-1) is registered and awaits implementation. Scorer score_s127.py.
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
