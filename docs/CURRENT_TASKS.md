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

## Running now (03:22 2026-10-09; credit ~$98; cap 16 hosts / < $4/h)

**User rules:** full runs on Vast 3090s (>= 16 GB, checkpointed); local cards short memory-capped jobs only, never idle. smoke_plain before commits
touching fsq120 / table132 / core. Never edit a module while a deploy is in flight (auto_ship is transitive: fixedchunk / multrow ship multpair,
deltas5). One fill_generic at a time. Destroy with scratch/vdestroy.sh (bare `vast destroy` aborts at its y/N prompt).

### Vast (~10 hosts, ~$2.1/h; 07:50)
* b210 §165 K16 + K8; b214-b217 §168 DIRECT (std s1 / s2, NA s1 / s2); b218 §169 K16 + K8 (K8 `--eval_at 24000`); b221 §171 K8 + BLIND;
  b222 §170 cot s1 (last arm; expect an eval OOM -> fetch model.pt, local `--eval_only`); b224 §170 K8 s1. b223 drained (cot s2 eval-only queued locally).
* Landed: §169 K0; §170 DIRECT s1 / s2, K0 s1 (n* = 3, note 170-3: §170 unblinded).
* C: filled to 0 GB at ~07:00 (46 GB of smoke .pt in the scratchpad; cleaned): watch C: free space.
* §170 timing (b220): DIRECT 0.09 / TEXT-COT 0.27 / K8 1.64 / K16 5.83 s/step on a 3090. K16 (~$18) needs the user (Audit) and K8 THINK-NECESSARY first.
* deltamem.FAST_ADD (6cbdfd9f): opt-in `--fast_add` for NEW cells (numerically equivalent). Same-host A/B (b225): -17 % K8 / -14 % K16; hosts differ ~2x.
* HOST PROBE (dd5d88dc, Audit): put `# HOSTPROBE` on every long think-pass queue (169/170 K8 K16, 171 s2/s3, 175, 176, 168 step 2); slow hosts -> rc 9 ->
  requeue_watch (restarted 05:28; it was not running) requeues + excludes the IP. K16 ~ $8 (34 h) with the probe; goes to the user via Audit.
* Built, gated, Audit-approved, waiting for Theory's go: §175 STEP-AUX (nested170 --aux_step 1.0 --aux_fade 36000,48000 --eval_at 36000; dir s175_auxk8_s1);
  §176 RAND-K (inject169 --rand_k 7,0.5,32 / --prog 8,8, --fast_add, think_k 8; dirs s176_rk_s1 / s176_prog_s1); scratch/s176_ref.py on each §169 K8 model.
* Next: §175 / §176 on Theory's go (time them first);
  §173 halt173.py post hoc on each think-pass model as it lands (§165 b210 first); b215-b217 §168 after the D* note.
* Model keep patterns include s165_ / s166_ / s168_-s171_ / s174_.

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; halt file scratch/<tag>.halt stops a worker)
* 3070: §172 PCCoT-PROBE full battery (scratch/s172_probe.py; report acc_release to Audit before any C1 numbers); 4060: proto165_k16c1;
  2070: §167 167-3 DAS refits (shared lines). §166 S5 SP s2 eval-only done (ae4b2ae9).
* Reboot recovery: memory note latent-reboot-recovery.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (07:45; session "claude-c6"; peers by name: "Audit" / "Experiments" / "Literature"):**
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
    - **§175 STEP-AUX:** §170 INJECT K8 plus a shared-trunk aux decoder emitting CoT step t from M_t (λ fades to 0 by 48k). It is gated on §170 K8 slot 1.
    - **§176 RAND-K:** random K (RK) or progressive (PROG) training on INJECT. C2 is read beyond the training support (K′ = 64), against fixed K8's post-hoc curve.
      Gated on §169 K8.
  - **Scored tonight:**
    - §172 PCCoT closed: it uses its latents and iterations; more iterations hurt; no localised step state.
    - §166 closed: S5 SP = PLAIN (coverage not shown).
    - §169 K0 slot 1 fails (0.003).
    - **§170 n* = 3** (note 170-3, from DIRECT; score_s170.N_STAR_FIXED = 3, unblinded).
    - §170 K0 s1 fails (pooled n 3..8 = 0.302).
    - Pending: §170 K8 s1 (~6 h) and CoT s1 / s2 (local eval-only); §169 K8 / K16; §171; §165 K16 / K8; §168 DIRECT.
    - §176 RK uses rand_k "7,0.5,32" (note 176-1).
    - **Notes:**
      - 170-1: a held-out SERIAL pool; CoT reductions only.
      - 170-2: STEP-RECOVER on §170 K8.
      - 171-1: BLIND decoder cell.
      - 171-2 is withdrawn by 171-3: the decoder's KV is live, as registered.
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
    - scratchpad `watch_land_v7.sh` (s15[5-9] / s16x / s17x; eval / what_state / posthoc / anchor / das / halt; skips *_tim*);
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
