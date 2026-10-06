# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-06 13:55 EDT** (local = UTC − 4; Vast hosts log UTC).

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

## Running now (13:55 2026-10-06; credit ~$117, ~15 hosts, ~$2.3/h; cap $4/h, <= 16 hosts, floor $30)

**User rules (2026-10-06):** full training runs on Vast (>= 16 GB, --ckpt_every 2000); local cards: short jobs only, memory-checked
(>= 1.5 GiB free), one per card, never idle (standing backlog docs/LOCAL_BACKLOG.md). Order C1 > C6 > C4 = C2 > C3 > C5 > C7.

### Vast (auto_drain_gen6; waiters watch_any3.sh; GPT-2 now fetched on-host from HF, SHIP_GPT2=1 to ship)

| Tag | Card | Work |
|---|---|---|
| b47 / b48 / b49 | 4060 Ti 16 GB | note 131-4 reruns s130chk_s5_s{5,6,7}_4060ti (b47: Ada parity_ckpt first) |
| b50 / b51 / b52 / b53 | A4000 | s130chk_s5_s{17,30,34,12}_a4000 (b53: A4000 parity_ckpt first) |
| b54 / b55 | 2080 Ti 22 GB | note 131-6 s131x_s5_s{401,403} / s{402,404}_2080ti (fp32) |
| b44 / b45 / b46 | 3090 595.71 | §132 rung-1 slots 2 / 3 / 4 (b44: s132bd_k4_s1 per-table breakdown first) |
| b61 | 3090 | §128 S5 fp32 CERTs (s128cert_s5_*) |
| a4 / d8 | 3090 | §118 LONG s83 / §119 CODI s2-3 |

* All reruns go through scratch/micro_gate.py (smallest micro of 1/2/4 fitting L 31/32 + L 16 at micro 1, 1.5 GiB margin; Audit:
  report any micro > 1 before 14k). A RESUME (fsq120 --resume ckpt_last.pt) waits for its card class's parity PASS (Ada: PASS, local).
* fsq120: --micro applies at L > 16 (root cause of the 14k OOMs); --ckpt_every / --resume / --stop_after / --probe_only.

### Local (scratch/c1_backlog_worker.sh per card; log logs/c1_local.log)

* 4060 bl4060 <- scratch/c1_backlog_4060.txt; 3070 bl3070b <- c1_backlog_3070.txt; 2070 bl2070b <- c1_backlog_2070.txt (after the
  2070 CERT queue, which stops at its memory gate after FF). Stop: touch scratch/<tag>.stop. New entries: append to the card's file.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (17:08):** §128 TASK-AGNOSTIC = true (fp32 certificates cover the eval in all 12 CLOSED runs, 128-6 / 128-7). Stable-latent synthesis: `docs/THEORY_STABLE_LATENT_2026-10-06.md` (F2 failure classes by prefix positions 1–2 + end probe; candidates 1 = §134 COPY, 2 / 2b round-trip / SELF-ANCHOR, 3 ramp, **4 = §135 TEACHER-STATE DISTILLATION** (the user's idea; stage A approved ~$7.5, after §133 / §134 launch)). Open: **§131** (5/6 caught, 0 false positives; endpoint pending), **§132** (slot 4 PASS; slots 2–3 evaluating), **§133** (7 proofs), **§134** (note 134-4), **§118 LONG** s84 / s85, **§115** (g105_mask4_s1), ramp prototype (`theory/proto_ramp.py`). Scorers: score_s131 / s132 / s133 / s134 / s135 / s118 / s115.
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
