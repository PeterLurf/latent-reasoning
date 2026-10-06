# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-06 10:45 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (10:45 2026-10-06; credit ~$120, cap $4/h, floor $30; <= 16 hosts; gradflow_check PASS required for new code paths)

**Order (user, via Audit):** C1 > C6 > C4 = C2 > C3 > C5 > C7.

### Vast (4 hosts; auto_drain_gen6 + drain_gen_partial (2-slot fetch semaphore, small files first, model-skip rule
`scratch/drain_posthoc.py` + `scratch/model_keep_patterns.txt`); waiter `watch_any3.sh a4 b43 d8 e31`; exclusions `scratch/vast_exclude_ips.txt`)

| Tag | Work | Next |
|---|---|---|
| a4 | §118 LONG s83 | drain, commit, report |
| b43 | §132 rung-1 slot 1 (`scratch/restart132.py`) | if accepted run exact -> slots 2-4; 4 failed attempts -> stop + report; rung 2 only after rung-1 PASS; alert if projection > $6 |
| d8 | §119 CODI s2/s3 lr 1e-3 (3090, bf16) | drain, report to Theory |
| e31 | §128 FF s3/s4 | drain, commit, report (TASK-AGNOSTIC gate) |

* b30 destroyed 10:41: §127 ctrl s3 model.pt lost (scp stalled at 340/462 MB every try); fp32 contraction done on host (259a5cd9).

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070; log `logs/c1_local.log`)

* 4060: `scratch/run130.py --card 4060` (pid 18516): TEST-SAME done (s74), now s130chk_s5_s{5,6,7}_4060. Also 128-4
  `cert_cover128` mod_s1 (pid 20316, fp32 pass). Then `scratch/c1_q_certcover128.sh` (waits on both pids via PowerShell)
  runs the other 10 certified MOD/CYC/RDFA runs; stop file `scratch/certcover128.stop`. rdfa_s4 done: bf16 3 -> fp32 0.
  Note: MSYS `kill -0` cannot see Windows pids (it started rdfa_s4 early at 10:03, in parallel).
* 3070: run130 3070 s130chk s17/s30/s34/s12; `scratch/posthoc128_local.sh cuda:1` (contraction + CERT as runs land).
* 2070: §116 NONE s41 rerun.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (12:24):** **§128 TASK-AGNOSTIC = true** (all five families; held-out CYC / MOD / RDFA 4/4 closed, 0 restarts; S5 3/4). Closed today: §101, §124, §127, §129, §130, §128. Open: **§131** (4k flag; interim PASS; endpoint = 7 check runs + 4 2070 runs), **§132** (in-context tables; rung-1 pilot 0.9865, slots 2–4 running, per-table breakdown on b44), **§119** (CODI s2 / s3; Coconut and SIM-CoT 0/3). Scorers: score_s119 / s131 / s132.
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
