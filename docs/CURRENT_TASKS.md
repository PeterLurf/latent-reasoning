# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-03 19:55 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (04:00 2026-10-04; credit about $163; Audit: up to ~14 live overnight, one proof host per new code path)

**Note 98-3 (Theory, 2026-10-03 ~23:40):** mult98's GEN-M/MASK had two forks at the final slot sharing a first token,
so the last step and the answer failed by construction. Fixed (one final fork, step text ++ " #### product"). The guard
test passed (FIXED fits and decodes, OLD fails; `theory/results/s98_smoke/guard98_3_output.txt`). The old §98 GEN-M/MASK
and §99 runs are voided (`*.void_forkconflict`, kept); their hosts were aborted with `scratch/abort_host.sh`.
**Order (user, via Audit):** C1 > C6 > C4 = C2 > C3 > C5 > C7.

### Vast (all `scratch/gen_deploy_*.sh`, watchers gen4/gen5 + partial drains; `scratch/watch_any2.sh` waiters)

| Tag | Instance | ssh (`-i ~/.ssh/runpod_ed25519 root@…`) | Work |
|---|---|---|---|
| y5 | 54073316 | `-p 22815 142.113.28.123` | §98 TC-M + NC-M x12 (unaffected by 98-3); tc_5x5_s1 eval OOM'd, `scratch/finish_evals.sh` re-runs it at QUEUE_DONE then drains |
| r3 | 54110367 | `-p 42018 109.173.238.203` | §98 rerun MASK 4x4 s1-3 (gen4; eval without OOM fallback) |
| r4 | 54110967 | `-p 40974 77.48.24.153` | §98 rerun MASK 5x5 s1-3 |
| u3 | 54110971 | `-p 40042 96.20.36.104` | §100 MASK2 4x4 s1-3 (k = 2) |
| u4 | 54110975 | `-p 41232 83.27.30.232` | §100 MASK2 5x5 s1-3 |
| m1 | 54111465 | `-p 22599 142.113.28.123` | §99 markov s1-3 |
| m2 | 54111472 | `-p 41152 77.48.24.153` | §99 ctrl s1-3 |
| m3 | 54111478 | `-p 33615 136.59.129.136` | §99 kcurr s1-3 |
| f1 | 54111560 | `-p 9195 136.36.139.116` | §105 gen97 GEN-4 --fork mask s1-3 (keep model.pt; Theory runs s104_serial) |
| s1 | 54111926 | `-p 6412 142.127.68.223` | §101 b0 s2 only (s1/s3 OOM'd at P2 before --micro; logs kept) |
| s2 | 54122048 | `-p 42390 167.179.138.57` | §101 micro proof: b0 s1/s3, then b1 s1/s2 (P2) |
| t1 | 54122055 | `-p 40071 96.20.36.104` | §103 self103, 3 seeds in lockstep -> boxes/runs_s5/s103/s103_s* (move to boxes/runs_s5/ on fetch) |
| p1 | 54122094 | `-p 11776 142.127.68.223` | §106 pilot GEN-S-MASK s1-3 |
| m10 | 54122096 | `-p 42075 167.179.138.57` | §107 lat2 s1-3 |

**Queued, not yet deployed** (deploy as hosts drain; keep <= ~10 live). In order:
1. §101 (C6): `gen_q_s2..s4` via `scratch/gen_deploy_101.sh`, P2 with --micro 2.
2. §103 (C4): `gen_q_t1` via `gen_deploy_103.sh`; one process, 3 seeds in lockstep. Move `boxes/runs_s5/s103/s103_s*` to `boxes/runs_s5/` on fetch.
3. §102 (statepath): `gen_q_m4` via `gen_deploy_99b.sh`.
4. The C3 parts: §98 GEN-M (`gen_q_r1/r2`, `gen_deploy_98r.sh`); §100 GEN2 (`gen_q_u1/u2`, `gen_deploy_100.sh`); the rest of §99 (std, textfirst, textfirst_mask, tfslot, combo: queues to write).
5. §106: not implemented; waits for §98 GEN-M to show multiplication is learned.

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070)

* 4060: COCO 4x4 s3 finishing; then free for Theory's §104 (C1). COCO 5x5 s2/s3 are HELD (lock dirs in `c1/runs/locks/`); release (rmdir) and start `py_local_worker2.sh` after §104.
* 3070: COCO 5x5 s1.
* 2070: §83 extras (fp32).

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **§97 C6:** complete (GEN-4 and HYB-4 s4-5 fetched).
* **§98:** ICoT-SI done (0.998 / 0.996); COCO 4x4 s1-2 in, s3 + 5x5 local; TC/NC on y5; GEN-M/MASK rerunning (98-3).
* **§99, §100, §101, §105:** running (above).
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
