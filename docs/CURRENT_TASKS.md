# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-07 19:50 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (20:10 2026-10-08; credit ~$113; cap 16 hosts / < $4/h)

**User rules:** full runs on Vast 3090s (>= 16 GB, checkpointed); local cards short memory-capped jobs only, never idle. smoke_plain before commits
touching fsq120 / table132 / core. Never edit a module while a deploy is in flight (auto_ship is transitive: fixedchunk / multrow ship multpair,
deltas5). One fill_generic at a time. Destroy with scratch/vdestroy.sh (bare `vast destroy` aborts at its y/N prompt).

### Vast
* Up: b192 §162 LONG s2; b204 §163 slot 1 (draining; model.pt for the §166 mix2d cell); b206 / b207 §159 s3 / s4; b210 §165 K16 + K8 (P=2);
  b211 §165 K0 + K16c1 (P=2); fill_s163k23 deploying b212 / b213 (§163 slots 2 / 3).
* Done today: §155 R / SHIFT-R complete; §157 closed; §159 slot 1 exact (s2 in); §160 A-R / CTRL-R + block-9 fallbacks; §161 MIX / IDX / MIX9;
  §162 DIRECT s1 / s2, LONG s1, SHORT s1 / s2 (models kept from 05751e36 on; DIRECT s1 / LONG s1 models lost); §163 s1 exact; §164 closed (anchored re-eval).
* Drains: drain_gen_partial skips ckpt_last / ckpt_prev of finished runs; manual fetch (tar over ssh, md5) when an scp stalls on a hung sshd.

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; halt file scratch/<tag>.halt stops a worker)
* Queues: §165 previews (K0 / K8 / K16 / K16c1) -> §166 MULT cells (s1 4060, s2 3070, s3 2070; fitted + plain) -> §167 DAS (MIX s1-s3, SHORT s1 / s2).
* Pending: §166 mix2d cells (note 166-2) once s163_mix2d_4x4_s1/model.pt lands; §166 S5 cells + §167 / §162 post-hoc on LONG s2 once b192 lands.
* CPU: scratch/s162_posthoc.py (SHORT s2 done, SHORT s1 running, Idle priority).
* Reboot recovery: memory note latent-reboot-recovery (13:20 power cut: watchers / filler / workers restarted).

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (16:53; session "claude-c6" after the 13:20 reboot; peers are addressed by name, "Audit" / "Experiments"):**
  - **Headline (multiplication):**
    - **§161 MIX:** exact on 4×4, length-general to 4×8 (0.91–0.96), and SERIAL with a state-carried running sum in 3 / 3 seeds.
    - **§164:** text CoT on the same data reads 0.0 at 4×6+ (anchored read; its scratchpad stops short), and the single pass fails even at 4×4. So
      **LATENT BEYOND TEXT ON LENGTH** (8c7e15f2).
    - **§159 R-5×5 s1:** exact 1.000 at 5×5, 0.80× TC-M FLOPs (3.24× ICoT-SI); s2–s4 are running.
  - **Open / scored as they land:**
    - §162 DELTA-S5: boundary L* = 16 (note 162-3). LONG s1 is exact to 32 and MIXED (op history 0.149, so not a buffer); 0.0 at 512. LONG s2 and SHORT s1 / s2 are
      pending.
    - §163 SHAPE-MIX-2D s1 (b204);
    - §165 FIXED-CHUNK slot 1 (b200–b203);
    - §166 STATE-COVERAGE (732964ed; MULT now, S5 after LONG s2's model);
    - §158 slots 2–4.
  - **Closed today:** §133 stage 1, §136–§138, §140–§146, §148–§157, §160, §161, §164.
  - **Waiting on the user (via Audit):** `docs/THEORY_C1_FORCING_2026-10-08.md` with addendum 2.
  - **Theory's tools:**
    - scratchpad `watch_land_v3.sh` (one-shot, patterns s15[5-9] / s16[0-9]);
    - `s162_boundary_only.py`;
    - commit with a pathspec (`git commit -- <paths>`), because the index is shared.

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
