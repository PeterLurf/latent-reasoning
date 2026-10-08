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

## Running now (21:58 2026-10-07; credit ~$153 after the user's top-up; cap 16 hosts / < $4/h; Audit: proofs + their completions approved)

**User rules:** full runs on Vast 3090s (>= 16 GB, --ckpt_every); local cards short memory-capped jobs only, never idle. Before any commit touching
boxes/fsq120.py / table132.py / core.py run `python scratch/smoke_plain.py`. Never edit a module while a deploy is in flight (the tar happens minutes
after create): record the host's sha256 and commit exactly that file.

### Vast (fill_generic + vdeploy cap gate; auto_ship sends queue modules / file args / bench files, imports to a fixed point; host import gate)

| Tag | Work |
|---|---|
| b137 | §142 MULT-READ A s1 (48k) - fsq120 7150979 / multrow 435638c |
| b138 | §143 FULL-GROW s1 (memgate PASS 7.9 GiB at L 32 width 768; 595.71, unpaired vs cnoise s1's 595.80) |
| b139 | §145 MULTI-SLOT SLOT4 slot 1 (multpair c4457fc) |
| b140 | §148 LOOP-RETRO loop4 + plain s1 (P=2) |
| b141 | §147 DELTA-STATE slot 1 (exp(s) read gains, alpha_by_head; multpair c1ae4ec, deltamem e5a8d4e) |
| b142 / b143 | §144 RUNNING-SUM F8 / N64 slot 1 (restart144 --memcheck); b142 retrying after a no-start host |
| b145 | §150 VOCAB-SLOT DIGIT slot 1 (multpair 8efc841: slot head under bf16 autocast) |
| b144 | §150 VOCAB-SLOT FULL slot 1 (restart150 --memcheck picks --micro; fp32 slot head) - deploying |
| b131 | §141 r4s16 slot 1 (finishing) |
| b146-b148 | §144 F8 follow-ups (m2 / m3 / noflag) |
| b149 / b150 / b151 | §143 FULL-GROW completions s2 / s3 / s4 (note 143-1 s1 PASS); b149 on 550.144.03 (no 595.71 offer); fill_f143c filling b150 / b151 |

* Drains: drain_posthoc runs the on-host fp32 contraction for s144_ / s145_ / s150_ too; keep patterns cover s142-s150.
* Disk cleanup 19:10-19:40 (user): 277 cold models -> E:/latent_archive (manifest docs/archive/MODEL_ARCHIVE_MANIFEST_2026-10-07.tsv).
* §143 note 143-2: drain_posthoc runs `fsq120 --ablate704_only` on-host for s143_ (eval_ablate704.json + eval.json s143.ablate704); s1's model.pt was not kept (no ablation for s1).
* §149 SOTA-PROBE: all four evals done and committed (sanity 42.46 / 33.13 / 39.35 / 44.81 vs gate 42.5 / 33.1 / 39.4 / 44.8); probe now memcapped (d4bc6e19).

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; log logs/c1_local.log; stop: scratch/<tag>.stop)

* 4060: Theory 862fe2f4 §150 soft-only DIGIT S501 / S502 (micro 2), then §144 sum4 8k no-stop S513-516.
* 2070: §150 soft-only DIGIT S503 / S504 (fp32, micro 4).
* 3070: §144 harness guard first (01929b61: sum4 mod 100 --lmax_fixed 1, S601 / 602 x F8 / N64), then sum4 8k no-stop S511-512.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (22:13):** today's multiplication reads:
  - **§147 DELTA-STATE broke the middle-column wall** (0.743; columns 3–7 0.92–0.99; DROP by rule, IMPROVES), leading to **§151 DELTA-LONG** (48k × 4 seeds, approved).
  - §142 / §145 / §148 DROP; §150 DIGIT collapsed (150-3 correction); §149 SOTA-PROBE scored (no SERIAL model).
  - **§143 FULL-GROW s1 passes** (143-1 / 143-2: about 4% of e from the new dims; the ablation decides in s2–s4).
  - **Pending:** §143 s2–s4, §144 (the L=1 harness check first; F8 entry is a seed lottery), §150 FULL, §151.
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
