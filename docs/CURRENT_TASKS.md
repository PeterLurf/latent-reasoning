# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-05 22:15 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (22:15 2026-10-05; credit ~$150, burn ~$3.3/h incl. new hosts, cap $4/h, floor $30; <= 16 hosts; gradflow_check PASS required for new code paths)

**Order (user, via Audit):** C1 > C6 > C4 = C2 > C3 > C5 > C7.

### PAUSE (user, ~23:40 2026-10-05, via Audit): "pause experiments on GPUs for now". Do NOT restart the local queues or workers on the 2070 / 4060 / 3070 (§116 OPDEF s31 / s32 resume, post-hoc jobs) until the user says so. Vast hosts are unaffected unless the user says otherwise.

### REBOOT (user, ~23:00 2026-10-05, to free the 2070 wedged by PID 6904) - post-reboot checklist (Experiments)

Safe at 23:00: no drain / deploy in flight; local §116 OPDEF s31 (3070) / s32 (4060) checkpoint at every round boundary
(ckpt.pt, auto-resumed by rerunning the same command); Vast hosts keep running (watchers live here: nothing is lost, an
unwatched host whose queue ends just bills idle until its watcher restarts).
1. `bash scratch/rebuild_live_hosts.sh`; check `vast show instances` (8 hosts: s4 a2 a4 j1 c2 c3 d1 b27).
2. Restart every drain watcher: `bash scratch/post_reboot_redrain.sh` (auto_drain_gen5 per host, logs appended). A host
   that finished during the reboot is drained by its restarted watcher.
3. Local §116 seeds resume from ckpt.pt: the lock dirs block a plain restart, so
   `rmdir c1/runs/locks/s116_opdef_s31 c1/runs/locks/s116_opdef_s32`, then
   `nohup bash scratch/py_local_worker2.sh cuda:1 scratch/c1_q_116c_local.txt &` and the same for cuda:0
   (each claims one seed; check log.jsonl continues from the last round). The 2070 (cuda:2) is free.
4. Post-hoc on the 2070 (fp32, one at a time): fp32 `--contraction_only` for s126_s5_rawh_s1 (CPU run killed by the
   reboot), s126_s5_rawh_s4, s126_s5_anneal_s2_fp32cast; then `--growth_only` / `--cert_only` on the ANNEAL rerun.
5. Deferred until after the reboot (no deploy may straddle it): §127 b28 replacement (STD2 + LR10 + replay s2,
   `EXTRA_120="scratch/replay_basin.py scratch/replay_wrap.py" bash scratch/gen_deploy_120.sh "<3090 offer> 28"`; the AT
   host was BAD) and §119 CODI rerun `bash scratch/gen_deploy_119.sh "<offer> 3"` (scratch/gen_q_d3.txt, fix c1caef75).
6. Re-arm the waiters (watch_any2.sh on all tags); Audit restarts its gpu_watch / cron.

### Vast (all `scratch/gen_deploy_*.sh`, auto_drain_gen5 + `watch_any2.sh` waiters; ids/ssh in `scratch/live_hosts.txt`, `scratch/pods.txt`)

| Tag | GPU | Work |
|---|---|---|
| s4 | 3090 | §101 b0.01 s3 (decisive seed), ~10 h |
| a2 / a4 | 3090 Ti / 3090 | §118 LONG s82 / s83 |
| j1 | A5000 | §117 MK s51 + ST s61 (loss ~9.3, failing like the others) |
| b27 | 3090 | §127 Stage 1 seed 2: VAR + SKIP (`--stop_plateau`), then replay_basin s1 (bf16); b28 (STD2 + LR10 + replay s2) was a BAD HOST -> redeploy after reboot |
| c2 | A10 | §124 seed-28 trio |
| c3 | 3090 | §124 COPY s26-27 (+BASE) |
| d1 | A5000 | §119 Coconut lr sweep s1 (3e-5 / 1e-4 / 3e-4, P=3) |
| (d3) | - | §119 CODI sweep RERUN after the autocast-cache fix (d2's three runs void: LoRA got no gradient) - deploy after reboot |

* §119 next: pick lr by train_split_final_acc -> rename the chosen s1 to s119_<method>_s1, seeds 2-3; SIM-CoT from Coconut s<seed> (--init model.pt).
* §127: stopped runs (plateau at 2k) have no eval.json; the drain stores them as <dir>.void_partial -> rename back to <dir> (scorer reads log.jsonl). Escaped arms -> Stage 2 (S5 s1/s3/s4, FF s1-2, + ctrl s3-4).
* Drains keep failed runs' args.json + log.jsonl as `<dir>.void_partial` (99192404).
* Local RAM: GPU jobs hold 7-10 GB private each; commit nearly full -> run CPU checks one at a time (two smokes died silently).

### Local PC (torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070; py_local_worker2, log `logs/c1_local.log`)

* 112-5 resumes done: f050 s24 0.968 -> 0.996, f010 s22 0.9645 -> 0.995 (confirms); §110 s1 resume finished on the 4060 (to report).
* §116 (`scratch/c1_q_116c_local.txt`, lock-dir claims): 2070 NONE s41; 3070 OPDEF s31; 4060 OPDEF s32.
* Post-hoc for every drained §120 / §123 / §126 FSQ cell: `fsq120 --growth_only` and `--cert_only` (FF adds cert_task, cert_task_reads)
  and `--contraction_only --fp32` locally.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (23:39):** §123 / §126 / 112-5 scored and closed. **§127:** VAR rerun with the forward's own group means (127-3); SKIP s2 rerun with the guard only, skip-off decode descriptive (127-4); SKIP Stage 2; b28 STD2 + LR10 redeploy. **§128 registered (e7078432):** five families (S5, FF + held-out CYC / MOD / RDFA), restart-on-plateau, Vast 3090s, awaiting Experiments. Scorers: score_s127.py, score_s128.py. **§124** waits on COPY / BASE s26–28 and EXPOSE s28. Local GPUs are off (user ~23:45).
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
