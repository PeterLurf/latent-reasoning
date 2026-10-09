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

* **Theory (02:03; session "claude-c6"; peers by name: "Audit" / "Experiments" / "Literature"):**
  - **The user (via Audit, 2026-10-09):**
    - ~01:30: "do another research pass. work on task agnosticness. approve go ahead on prosqa deep".
    - ~01:45: "keep working on the prosqa deep and the nested arithmetic … creating new methodologies and architectures for task agnostic".
    - Roadmap frame: task-agnostic input → multi-step reasoning → halting → length stability → scale → distillation → a fair CoT baseline.
  - **Registered tonight (contracts with Experiments):**
    - **§168 DEEP-ProsQA:** step 1 DIRECT on b214+.
      - 168-1: THINK-NECESSARY measured above chance.
      - 168-2: ProsQA-Ext 2609.35643 is prior work; a comparable 3–6 → 7–12 cell; K8 first.
      - Step 2 is sized after the D* note, with a projection to Audit.
    - **§169 INJECT:** K16 / K8 / K0 on §161's mix.
    - **§170 NESTED:** Feng mod 11; a DIRECT n* boundary (the scorer is blinded until N_STAR_FIXED is set); INJECT K8 / K0 + text CoT; SERIAL-NEST.
    - **§171 READ-SLOTS:** 8 slots read a once-encoded input; K8 on the mix; head to head with INJECT.
    - **§172 PCCoT-PROBE:** eval only, whynlp/pccot-gpt2.
    - **§173 CONV-HALT:** eval-only halting when the answer settles (2 consecutive passes unchanged), on every think-pass arm.
    - **§174 MULTI-TASK:** one INJECT K8 model on mult + nested, no task flag. It is gated on §169 / §170 K8 slot 1, and needs their eval_24k as the budget reference.
    - **Notes:** 170-1 (a held-out SERIAL pool; CoT reductions only); 171-1 (BLIND decoder cell).
  - **Multiplication headline:**
    - §161 MIX is exact, length-general to 4×8 and SERIAL (3 / 3).
    - §164: latent beyond text on length.
    - §159: 5×5 exact 4 / 4, at 0.80× TC-M.
    - §163: 2-D BOUNDED.
  - **Open:**
    - §165 (K16 / K8 pending on b210);
    - §166 S5;
    - §167 refits;
    - §158 slots 2–4.
  - **Theory's tools:**
    - scratchpad `watch_land_v5.sh` (s15[5-9] / s16x / s17x; eval / what_state / posthoc / anchor / das);
    - commit with a pathspec;
    - fix n* (§170) and D* (§168) in dated notes from DIRECT alone.

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
