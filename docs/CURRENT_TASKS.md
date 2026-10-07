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

## Running now (09:25 2026-10-07; credit $64.5, 12 hosts, $2.41/h; cap 16 hosts / < $4/h; floor $30)

**User rules:** full runs on Vast 3090s (>= 16 GB, --ckpt_every); local cards short memory-capped jobs only, never idle. Before any commit touching
boxes/fsq120.py / table132.py / core.py run `python scratch/smoke_plain.py`. Never edit a running .sh (stage `.new`; Windows refuses the swap while open).

### Vast (gen7 watchers -> drain_gen_partial; model transfers 2 at a time via D:/drain_tmp/locks)

| Tag | Work |
|---|---|
| b111-b114 | §136 GROW grow4 s1-s4 (fill_s136; b112 ssh flaky - kex resets - but training fine, 13.1k at 09:01) |
| b128 / b129 | §134 MOD s2 / s3 (copy_mod + cc_job) |
| b130 / b131 | §141 r4f8 / r4s16 slot 1 (--micro 4); **b130's model.pt: run s141 per-repeat probe on it after drain (Theory 09:55), keep it** |
| b126 | §139 proof done (A stopped by copy gate; A' / B in hand); drain fetching models |
| b102 | §136 cgrow s4: post-hoc done on host; model.pt via scratch/chunk_fetch.sh (link ~53 KB/s; 8/56 pieces at 09:20; waiter verifies tarball + model sha, then destroys) |
| a5 / a6 | §118 LONG s84 / s85 (mk108) |
| b133-b136 | **§138 proofs** slot 1: full_fsq / full_noise (b134, redeployed 10:57) / full_cont / full_hinge; first launch died (restart138 not shipped), b133 / b135 / b136 rerun in place 14:53 UTC; memchecks 8.5 / 8.5 / 11.4 GiB at L 32 (limit 21); deploys now auto-ship queue modules + import gate (ea2e417e) |

* Pending swap: scratch/vdeploy_toy.sh.new (serialised 16-host cap gate) waits for no running vdeploy (waiter task), then its own commit.

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; log logs/c1_local.log; stop: scratch/<tag>.stop)

* 4060 / 3070: §141 / §133 filler protos. 2070: Theory 09:45 block - 16 paired SOFT-16 / FSQ-16 S5 protos to 4.5k fp32 (S 509-516, ~10 h), then replay_x 541-560.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (15:41):** registered and Audit-approved for proofs, Experiments implementing:
  - §142 MULT-READ (a27cf1cf, 142-1);
  - §143 FULL-GROW (dc546570);
  - §144 RUNNING-SUM and §145 MULTI-SLOT (b1d7741c; the user's "fix the multiplication", 15:35). In §144's result, label the prefix-chain supervision 110b-1-type (Audit).
  - Order: §142 A s1 and §143 s1 first, then §144 / §145.
  - §133 stage 2 (133-11 / 136-2) proofs come after those.
  - §141 r4s16 pending (b131, about 19:00). §136 and §138 closed.
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
