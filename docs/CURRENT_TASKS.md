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

## Running now (12:20 2026-10-08; credit ~$120; cap 16 hosts / < $4/h)

**User rules:** full runs on Vast 3090s (>= 16 GB, checkpointed); local cards short memory-capped jobs only, never idle. smoke_plain before commits
touching fsq120 / table132 / core. Never edit a module while a deploy is in flight (tar minutes after create; auto_ship is transitive: multrow /
restart160 ship multpair). One fill_generic at a time.

### Vast
* fill_s162to163 (12:15, HEAD 1781cd53): b189 / b190 §162 DIRECT s1 / s2 (deltas5, --eval_bs 100) -> b191 / b192 §162 LONG s1 / s2 (l_max 32) ->
  b195 §161 MIX9 fallback -> b196 / b197 §164 TC-MIX s1 / s2 -> b198 §164 DIRECT-MIX s1 + s2 (P=2) -> b204 §163 SHAPE-MIX-2D slot 1.
* Up: b182 §159 R-5x5 slot 1; b186-b188 §161 MIXIDX slots / MIX; b193 §160 CTRL-R9; b194 §160 A-R9 (started 12:11 after 4 STUCK_LOADING offers).
* Draining: b178 (SHIFT-R k3), b184 (§161; sshd unreachable).
* Waiting: §165 FIXED-CHUNK b200-b203 (k16 / k8 / k0 / k16c1 slot 1; Audit-approved 3539b83e + 1781cd53) after §164 reads.
* Code today: multrow §160 (R-ROWS + s160 probes, restart160/gate160), multpair §159 / §161 / §163 (ext_5x8, shape_mix + col_index, shape_mix_2d),
  deltas5 (§162), mult98 §164 (gen_items = the bench CoT), fixedchunk (§165); gates gate160-165 in scratch/.

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; log logs/c1_local.log; halt file scratch/<tag>.halt stops a worker)
* 4060: proto163_mix2d (staged §163 copy). 3070: proto164_directmix. 2070: proto165_k16 (memcheck first).
* §160 null (--s160_probe_only on §152 A s1, CPU Idle, pid 12256; eval_s160_posthoc.json when done).
* Lessons: 8k-fit probes OOM 8 GB cards (use --probe_n_fit 3000); KV-decode evals need small eval_bs; never eval a sed range of a deploy script
  (it ran gen_deploy's loop 2026-10-08 09:19, no instance created).

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (10:50):**
  - **§161 MIX is 3 / 3** (4f774362): exact (0.996 / 0.997 / 0.996), length-general (ext_4x8 0.964 / 0.917 / 0.912 from n ≤ 5), and SERIAL with a state-carried
    running sum in every seed; not linearly editable. MIX+IDX and MIX9 are pending.
    - **§164 MIX-BASELINES** (bbd53874): TC-MIX / DIRECT-MIX on the same mix, needed before any "beyond text" claim.
    - **§163 SHAPE-MIX-2D** (0045d6a1): its condition is met; slot 1 is launching.
    - **§165 FIXED-CHUNK** (60e55135; the user's "is it task-agnostic?"): the same memory on fixed text chunks plus think passes. It runs after MIX s2 and §164.
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
    - §160 block-9 fallbacks A-R9 / CTRL-R9 (b194 / b193; descriptive). §160 closed on A-R s1 0.056 (240f32fa);
    - §161 SHAPE-MIX, 3 + 3 runs;
    - §162 DIRECT, then LONG / SHORT.
    - §163 SHAPE-MIX-2D (0045d6a1): both lengths mixed; runs only if §161 MIX s2 replicates.
  - **Scorer audit (08:10–08:40):** read-path fixes in score_s158 / s157 before data (ad185b64, 16f0af42; note 157-3); a fallback-trigger readout in s159–s161 (df955868).
  - **Where multiplication stands:**
    - §155 R is exact on 4×4 in every seed (0.997–1.000), at 0.79× TC-M FLOPs.
    - It is not SERIAL, not length-general and not linearly editable: an exact buffer (155-4).
    - FREE-PAIRS makes it SERIAL, still without C1 credit.
  - **Closed today:** §133 stage 1, §136–§138, §140–§146, §148–§152, §154–§157. §155 FAILs (both arms; 9b20e2d3). §157: no read change fixes the carry; read interference stays open (00e42f5d + correction).
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
