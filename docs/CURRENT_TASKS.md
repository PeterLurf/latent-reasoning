# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-04 19:45 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (07:00 2026-10-04; credit about $155; Audit: ~14 live, temporarily 16 for C1 items; one proof host per new code path)

**Note 98-3 (Theory, 2026-10-03 ~23:40):** mult98's GEN-M/MASK final slot had two forks sharing a first token. Fixed
(one final fork, step text ++ " #### product"); guard test in `theory/results/s98_smoke/guard98_3_output.txt`; old runs
kept as `*.void_forkconflict`.
**Order (user, via Audit):** C1 > C6 > C4 = C2 > C3 > C5 > C7.

### Vast (all `scratch/gen_deploy_*.sh`, watchers gen4/gen5 + partial drains; `scratch/watch_any2.sh` waiters)

| Tag | Instance | ssh (`-i ~/.ssh/runpod_ed25519 root@…`) | Work |
|---|---|---|---|
| s2 | 54122048 | `-p 42390 167.179.138.57` | §101 micro proof |
| m11 | 54153333 | `-p 23005 5.2.174.20` | §107 lat4 s1-3 |
| n1 | 54179841 | `-p 33603 136.59.129.136` | §111 MK-4-2x s6-8 (48k) |
| n5 | 54215540 | `-p 35368 142.172.103.36` | §111 MK-4-DAG s9-10 |
| n6 | 54215806 | `-p 40070 96.20.36.104` | §111 MK-4-READ s9-10 (decides REL -> §113) |
| l1 | 54199003 | `-p 45266 223.167.245.91` | §114 LOOP-R s1 (proof passed; ~10 h; needs --rel_only after) |
| l2 | 54217478 | `-p 12446 81.166.162.13` | §114 LOOP-R s2 |
| l3 | 54217482 | `-p 48053 86.123.234.69` | §114 LOOP-R s3 |
| l4 | 54217490 | `-p 49343 86.123.234.69` | §114 LOOP-1 s1-3 |
| d1 | 54201369 | `-p 40042 96.20.36.104` | §112 f050 s21-22 (proof) |
| d2 | 54209664 | `-p 53089 108.55.118.247` | §112 f050 s23-24 (redeploy) |
| d3 | 54202521 | `-p 2660 142.180.224.148` | §112 f050 s25 + f025 s21 |
| d4 | 54212376 | `-p 46960 79.116.20.245` | §112 f025 s22-23 |
| d5 | (deploying) | 167.179.138.57 | §112 f025 s24-25 |
| e5 | 54216219 | `-p 28525 46.107.144.5` | §110 STUDENT s1/s3/s5 (gen_deploy_110s.sh ships the pools) |

**Queued, not yet deployed** (Audit: fleet <= 14-16; stagger §112 into freed hosts):
1. §113 (`gen_q_e7..e9`, `gen_deploy_110.sh`): implemented; launch only on Theory's REL call after READ s9-10.
2. §110 STUDENT s2/s4 (`gen_q_e6`, `gen_deploy_110s.sh`) after e5.
3. §112 d6, d7 as hosts free up.
4. §101 `gen_q_s3/s4` after s2.
* Watchers don't detect host reboots: check uptime on the C1 hosts each pass.

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070)

* 4060: §115 chain (MK-4 s5 rel backfill, then snap115 on MK-4 s1/s3/s5, then g105 s1-3 at n=500).
* 3070: §112 f010 s21.
* 2070: §115 on READ s6-8 (descriptive), then rel backfills for §111 2x s9-10.

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
