# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-06 18:55 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (18:55 2026-10-06; credit ~$106, 16 hosts = cap, ~$3/h; < $4/h)

**User rules (2026-10-06):** full runs on Vast (>= 16 GB, --ckpt_every); local cards: short jobs, never idle, hard 6 GiB cap per job
(boxes/memcap.py via s5.load_gpt2; scratch/c1_backlog_worker2.sh). **Before any commit touching boxes/fsq120.py / table132.py / core.py
run `python scratch/smoke_plain.py`** (CPU ~2 min, incl. fsq120 main() end-to-end); vdeploy runs it before create; gen_deploy runs
`bash -n` on the deploy / job / drain scripts and logs DEPLOY_SILENT_FAIL. Three placement bugs in fsq120 today (5387d85e / 81e37687) - all
fixed, no real run affected.

### Vast (gen7 watchers; drains: gzip -t on host, on-host probe / prefix / fp32 contraction for failed s130chk / s131x; GPT-2 from HF)

| Tag | Card | Work |
|---|---|---|
| b62-b68 | 3090 (drivers vary) | §133 stage-1 slot 1 done except b68 pin64 (accepted: fsq64 / noise64 / cont64 s201); slots 2-4 = fsq16 / soft16 via scratch/fill_generic.sh (tag s133s, b83 / b84 running, b85-b88 as slots free) |
| b70 | 3090 595.71 | §134 COPY-ANCHOR (b71-b73 drained) |
| b75-b82, b89 | 3090 | §135 students: TSD b75 / b77 / b79 / b81, BASE b76 / b80 / b82 + **b89 = BASE s2 rerun from scratch** (b78 VRAM fault, note 135-6, s135_base_s5_s2.void_hostfault; b78 destroyed). New: hostcheck gate in gen_vast_job (ship scratch/hostcheck.py), fsq120 keeps ckpt_prev.pt |
| b54 / b55 | 2080 Ti | §131x s401->s403 / s402->s404 (fp32) |
| a5 / a6 | 3090 | §118 LONG s84 / s85 (mk108, ~27 h) |
| queued | 3090 | §132 rung 2 slot 1 b74 (after §133-§135 settle) |

* §132 rung 1 PASS (s2 / s3 / s4 exact; pilot residual = held-out table 63). Rung 2 approved, staged (b74 queue ready).
* Note 131-4 reruns done except s131x: failures split by prefix accuracy (compounding vs wrong-from-step-1), details in THEORY.

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; log logs/c1_local.log)

* Ramp prototypes (Theory 1ff94042: 4060 s603 / 3 / 4 / 8; 3070 s14 / 16 / 27 / 28 / 31; plain --stop_plateau, late skipped if plain flags),
  then replays / flag4proto / k6 protos / growth / contraction filler. Stop: scratch/<tag>.stop.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (22:17):** **§134 COPY-ANCHOR PASS on S5** (4/4 exact + SERIAL + ρ 0.14–0.22, no restarts; plain 1/4); guard (FF / CYC / MOD / RDFA) next. §128 TASK-AGNOSTIC (fp32 certs cover all 12 CLOSED). Registered: **§135** TEACHER-STATE DISTILLATION (students running; BASE s2 void host fault, rerun), **§136** GROW (the user's idea; 21 runs after §133 / §135). Open: §133 (proof slots: FSQ-16 / FSQ-64 / NOISE-64 / CONT-64 pass, SOFT-64 / PIN-64 dropped; slots 2–4 running), §131 (7/8 caught, 0 FP; s131x pending), §132 rung 2 (staged), §118 LONG s84 / s85, §115 (g105 at bs 40), ramp prototype (5 pairs read so far; F2 mostly without rounding). Synthesis: `docs/THEORY_STABLE_LATENT_2026-10-06.md`.
* **Local GPU backlog:** `docs/LOCAL_BACKLOG.md` (Theory maintains ≥ 3 short jobs per card; Experiments pulls when a card frees up; the user: never leave local GPUs idle).
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
