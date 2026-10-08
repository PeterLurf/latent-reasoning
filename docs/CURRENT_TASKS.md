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

## Running now (00:25 2026-10-08; credit ~$153 after the user's top-up; cap 16 hosts / < $4/h; Audit: proofs + their completions approved)

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
| b152-b155 | §151 DELTA-LONG s1-s4 (restart151: 48k, keep 24k / 36k -> s151.serial_by_ckpt; model.pt kept) |
| b156-b158 | §144 N64 follow-ups m2 / m3 / noflag (note 144-3) |
| b159 / b160 | §152 DELTA-READ A s1 / CTRL s1 (multrow 9535bfb0 --carrier deltamem; gates g1-g4 pass on the 3070; s152_ models kept) |

* Drains: drain_posthoc runs the on-host fp32 contraction for s144_ / s145_ / s150_ too; keep patterns cover s142-s150.
* Disk cleanup 19:10-19:40 (user): 277 cold models -> E:/latent_archive (manifest docs/archive/MODEL_ARCHIVE_MANIFEST_2026-10-07.tsv).
* Deploy fixes 2026-10-07: vdeploy clears a recycled ip:port's stale host key once (679d81fb); gen_deploy destroys every instance a failed attempt created (8fda0ad3).
* Next for Experiments: §153 THINK-PASS (code ready in a temp copy; apply to multrow after smokes; launch only once §152 A s1 >= 0.95); §152 B-EARLY + s2-s4 on gated seeds; S5xk family for §133 stage 2 (low priority, THEORY §133 Stage 2 + note 133-11/133-12).
* §143 note 143-2: drain_posthoc runs `fsq120 --ablate704_only` on-host for s143_ (eval_ablate704.json + eval.json s143.ablate704); s1's model.pt was not kept (no ablation for s1).
* §149 SOTA-PROBE: all four evals done and committed (sanity 42.46 / 33.13 / 39.35 / 44.81 vs gate 42.5 / 33.1 / 39.4 / 44.8); probe now memcapped (d4bc6e19).

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; log logs/c1_local.log; stop: scratch/<tag>.stop)

* 4060: Theory 862fe2f4 §150 soft-only DIGIT S501 / S502 (micro 2), then §144 sum4 8k no-stop S513-516.
* 2070: §150 soft-only DIGIT S503 / S504 (fp32, micro 4).
* 3070: DeltaMem 5x5 proto S501 to 12k (8e29e417), then idle-fill; 4060: sum4 S515-516 then 5x5 proto S502; 2070: §150 FULL soft S505 / 506 (8e29e417).

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (00:03):**
  - **§147 DELTA-STATE broke the middle-column wall** (0.743; columns 3–7 0.92–0.99; DROP by rule, IMPROVES), leading to **§151 DELTA-LONG** (48k × 4 seeds, running).
  - **§152 DELTA-READ** (378fe8e6; notes 152-1 / 152-2) is implemented (9535bfb0); b159 / b160 launch after the CUDA gates.
  - **§153 THINK-PASS registered** (ea154f30; note 153-1): item-agnostic passes, gated on §152 A s1 ≥ 0.95.
  - **§143 FULL-GROW PASS** (s1 / s2 / s4; s3 pending; 704-block ablation s2 UNUSED / s4 USED). This triggers note 133-12: cap4|fullgrow joins §133 stage 2.
  - **§150 VOCAB-SLOT: both DROP** (the slots collapse to constant tokens; the soft-only protos FAIL). B4 stops.
  - Closed today: §133 stage 1, §136, §137, §138, §140, §141, §142, §143, §145, §146, §148, §149, §150.
  - **Pending scoring:** §144 (the N64 follow-ups), §151, §152; §143 s3 (descriptive).
  - Later: the §133 stage-2 proofs (now with cap4|fullgrow), §139 A' / B (descriptive), §134 MOD s2 / s3.

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
