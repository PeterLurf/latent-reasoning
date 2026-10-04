# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-04 07:00 EDT** (local = UTC − 4; Vast hosts log UTC).

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
| u4 | 54110975 | `-p 41232 83.27.30.232` | §100 MASK2 5x5 s1-3 |
| s1 | 54111926 | `-p 6412 142.127.68.223` | §101 b0 s2 only (s1/s3 OOM'd at P2 before --micro; logs kept) |
| s2 | 54122048 | `-p 42390 167.179.138.57` | §101 micro proof: b0 s1/s3, then b1 s1/s2 |
| t1 | 54122055 | `-p 40071 96.20.36.104` | §103 self103 -> boxes/runs_s5/s103/s103_s* (move to boxes/runs_s5/ on fetch) |
| m10 | 54122096 | `-p 42075 167.179.138.57` | §107 lat2 s1-3 (proof for m11/m12) |
| m4 | 54125241 | `-p 40067 96.20.36.104` | §102 statepath s1-3 |
| r1 | 54130307 | `-p 50171 137.175.22.196` | §98 GEN-M 4x4 s1-3 (s1 restarted by hand; its queue "rc 1" line is the old tiktoken timeout) |
| r2 | 54131932 | `-p 28503 149.200.35.231` | §98 GEN-M 5x5 s1-3 |
| u1 | 54131907 | `-p 50139 137.175.22.196` | §100 GEN2 4x4 s1-3 |
| m8 | 54134511 | `-p 27762 149.200.35.231` | §99 tfslot s1-3 (C1) |
| k1 | 54134514 | `-p 30047 154.61.62.158` | §108 MK-4 s1-3 (proof for k2-k4) |
| u2 | 54138454 | `-p 40605 77.48.24.153` | §100 GEN2 5x5 s1-3 |
| m5 | 54138455 | `-p 20154 151.237.25.16` | §99 std s1-3 |
| m6 | 54138463 | `-p 47253 86.123.234.69` | §99 textfirst s1-3 |
| p2 | 54139052 | `-p 12419 81.166.162.13` | §106 note 106-3 GEN-S-MASK-k2 s1-3 (C1 group) |
| p3 | 54139062 | `-p 49822 86.123.234.69` | §106 note 106-3 WC-S-k2 s1-3 (new code, own proof) |

**Queued, not yet deployed** (backfill as hosts drain; a second host of an experiment only after its first host's first eval.json and drain):
1. §108 k2-k4 (`gen_q_k2..k4`, `gen_deploy_108.sh`) after k1.
2. §101 `gen_q_s3/s4` after s2.
3. §107 lat4/pause4 (`gen_q_m11/m12`) after m10.
4. The rest of §99: `gen_q_m7` textfirst_mask, `m9` combo.
5. §106: the k = 4 pilot failed its gate (note 106-2); latent k = 4 arms not run; TC-S deferred (Theory).

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070)

* 4060: COCO 4x4 s3 (restart) then 5x5 s3 (`scratch/c1_q_98coco_c.txt`).
* 3070: COCO 5x5 s2 (`scratch/c1_q_98coco_b.txt`).
* 2070: §83 extras (fp32).

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
