# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-07 11:05 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (17:05 2026-10-07; credit ~$153 after the user's top-up; cap 16 hosts / < $4/h; Audit: proofs + their completions approved)

**User rules:** full runs on Vast 3090s (>= 16 GB, --ckpt_every); local cards short memory-capped jobs only, never idle. Before any commit touching
boxes/fsq120.py / table132.py / core.py run `python scratch/smoke_plain.py`. Never edit a module while a deploy is in flight (the tar happens minutes
after create): record the host's sha256 and commit exactly that file.

### Vast (fill_generic + vdeploy cap gate; auto_ship sends the queue's modules / file args / bench files, imports to a fixed point; host import gate)

| Tag | Work |
|---|---|
| b137 | **§142 MULT-READ A s1** (48k; teacher cache 0 invalid / 0 mismatch; fsq120 7150979 = 366652ac) - training since 16:47 |
| b138 | **§143 FULL-GROW s1** via scratch.memgate (deploying 16:57; 595.71, unpaired vs cnoise s1's 595.80) |
| b139 | **§145 MULTI-SLOT SLOT4 slot 1** (restart145; multpair c4457fc) - setup since 17:00 |
| b131 | §141 r4s16 slot 1 (ends ~18:15) |

### Built, gating (CPU smokes running; Audit review next)

* **§144 RUNNING-SUM** (fsq120 sum4, committed 366652ac; restart144): ground truth + gradflow (CPU / CUDA) PASS; e2e smokes running.
* **§147 DELTA-STATE** (boxes/deltamem.py + multpair --carrier deltamem, uncommitted): step-0 |d logits| 0; gradflow PASS; e2e smoke running.
* **§148 LOOP-RETRO** (boxes/loop148.py, uncommitted): r = 1 |d logits| 0; gradflow PASS; e2e smoke running.

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; log logs/c1_local.log; stop: scratch/<tag>.stop)

* 4060 / 3070: §133 / §141 filler protos (+ queued gate jobs first); 2070: replay_x 543-560 (fp32).

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (16:52):** the research program (the user's wide sweep, 16:40) is in `docs/THEORY_PROGRAM_2026-10-07.md`.
  - **Ranking:** B1 DELTA-STATE (§147) > B3 LOOP-RETRO (§148); B2 MID-STACK parked by §146's NO SIGNAL.
  - **Registered and Audit-approved for proofs:** §142 MULT-READ (b137), §143 FULL-GROW, §144 RUNNING-SUM, §145 MULTI-SLOT, §147 (in parallel), §148 (after).
  - **Disclosures:** 142-2 / 145-2 (internal priors §121 / §107); pre-run notes 145-1 / 147-1.
  - **Pending:** §141 r4s16 (b131, about 19:00). §133 stage-2 proofs (133-11 / 136-2) come after these.
* **§136** GROW: 16 local runs with eval.json; grow4 s1-s4 running; then score_s136 (fp32 contractions in hand).
* **§139** A' / B proof (descriptive). **§141** r4f8 / r4s16 (r1s16 DROP, note 141-4). **§134** MOD s2 / s3.
* **§138** after its proofs. Closed today: §137 DROP, §140 DROP, §133 stage 1 PASS (FSQ-64 / NOISE-64 / CONT-64 / SOFT-16).

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
