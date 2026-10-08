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

## Running now (08:25 2026-10-08; credit $128.18; fleet 5 hosts $1.07/h; cap 16 hosts / < $4/h)

**User rules:** full runs on Vast 3090s (>= 16 GB, --ckpt_every); local cards short memory-capped jobs only, never idle. Before any commit touching
boxes/fsq120.py / table132.py / core.py run `python scratch/smoke_plain.py`. Never edit a module while a deploy is in flight (the tar happens minutes
after create; auto_ship is transitive: a multrow queue ships multpair too): record the host's sha256 and commit exactly that file.

### Vast (fill_generic one at a time; auto_drain_gen7 per host; drain_posthoc runs what_state / C7 / free-prop on-host)

| Tag | Work |
|---|---|
| b173 | §158 FREE-PAIRS s1 (restart158) |
| b174 | §157 PERIODIC s1 (MULTIREAD 0.009 / CCQ 0.064 closed) |
| b177-b179 | §155 SHIFT-R s2-s4 (restart155 --arm shiftr; s1 1.000 exact) - started 07:41 / 07:51 / 08:24 |
| b180 / b181 | §160 R-ROWS A-R s1 / CTRL-R s1 via scratch/restart160 (on-host gate160 bf16 + memcheck, then 48k); code 5996a2c5 (Audit-approved a283c617 + spike floor + post-hoc keep fix); fill_s160 deploying 08:25 |

* §155 R arm complete: s1-s4 1.000 / 0.998 / 0.997 / 0.999, 0/4 SERIAL; models local (R s2's fetched after the b170 reboot).
* §160 null: `--s160_probe_only` on §152 A s1's model, local CPU at Idle priority (logs/posthoc_s160_null_s152_a_4x4_s1.log -> eval_s160_posthoc.json).
* Next: §159 R-5x5 (patch ready: multpair ext_5x8 + spike_100x_floor, restart155 --prefix s159, drain_posthoc s159_, keep patterns s155/158/159/160)
  and §161 SHAPE-MIX (patch161 + scratch/gate161.py: all pass on the staged copy boxes/_multpair161_tmp.py). Both apply to multpair once b180 / b181
  have STARTED, then Audit's diff review; §159 one host until slot 1 reads; §161 six hosts (s161_{mix,mixidx}_4x4_s1-3).

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; log logs/c1_local.log; stop: scratch/<tag>.stop)

* 4060: proto158_free_s1_4060 resume from ckpt 6000 -> eval (--eval_bs 50 --probe_n_fit 3000; two earlier evals OOMed).
* 3070: proto157_periodic_s1_3070 (memcheck-gated, 8k; proto157_ccq done).
* 2070: proto160_ar_s1_2070 (§160 A-R preview, fp32, 8k, memcheck-gated, --probe_n_fit 3000).
* The 8k-fit probes (what_state / s158 / s160) need ~1.5 GiB per feature matrix: on 8 GB cards use --probe_n_fit 3000.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (09:35):**
  - **Waiting on the user (via Audit):** `docs/THEORY_C1_FORCING_2026-10-08.md` (42de5d5f + addendum 8500894d; note 158-2, Literature 6b771087 / fd630b87).
    - The finding: C1 is forced only by depth beyond one pass, or by input beyond the buffer's capacity.
    - The proposals: multiplication becomes the C3 / C6 / C2 testbed; the C1 + C3 headline moves to a task that exceeds one pass.
  - **§162 DELTA-S5 registered** (9f7d4793): DeltaMem-R on S5 against a measured single-pass boundary (DIRECT).
    - **Blinding:** score DIRECT first, with boundary() only, and commit the boundary note **before** reading any `s162_long*` / `s162_short*` output. Do not run
      score_s162.py in full before then.
  - **§158 FREE-PAIRS slot 1** (46939c14): 0.982, SERIAL, no C1 credit, as 158-2 predicted. Slots 2–4 run by the rules.
  - **Scored as they land:**
    - §155 SHIFT-R s3 / s4 (b178 / b179; descriptive, since the arm is decided);
    - §158 slots 2–4;
    - §159 R-5×5;
    - §160 A-R / CTRL-R (b180 / b181);
    - §161 SHAPE-MIX, 3 + 3 runs;
    - §162 DIRECT, then LONG / SHORT.
  - **Scorer audit (08:10–08:40):** read-path fixes in score_s158 / s157 before data (ad185b64, 16f0af42; note 157-3); a fallback-trigger readout in s159–s161 (df955868).
  - **Where multiplication stands:**
    - §155 R is exact on 4×4 in every seed (0.997–1.000), at 0.79× TC-M FLOPs.
    - It is not SERIAL, not length-general and not linearly editable: an exact buffer (155-4).
    - FREE-PAIRS makes it SERIAL, still without C1 credit.
  - **Closed today:** §133 stage 1, §136–§138, §140–§146, §148–§152, §154–§157. §155 FAILs (both arms; 9b20e2d3). §157 is "not the read" (00e42f5d).
  - **Later:** the §133 stage-2 proofs (with cap4|fullgrow), §139 A' / B (descriptive), §134 MOD s2 / s3.

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
