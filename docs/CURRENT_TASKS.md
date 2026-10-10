# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-09 20:10 EDT** (local = UTC − 4; Vast hosts log UTC).

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

### Vast (6 hosts; 20:10 2026-10-09)
* Live: b235 §175 aux K8; b236 §177 (f) kmatch; b237 / b238 §168 G1 / G2; b241 §178 K8 + 173-1 package (~10.4 h from 18:41, 107.205.138.127:33163);
  b242 §178 DIRECT then K0 (~1 h from 18:46, 194.26.196.174:18012). m12 conditional pair only on Theory's word.
* Drained + destroyed today evening: b218 (§169 K16 FAIL 0.002), b228 / b229 (§167 refits), b233 (§177 (a) tidx; index-direction post hoc 9ed15783),
  b240 (note 177-6 scaffold gate: full 4x4 at p 0 at 12k = 0.0 -> FAIL, (d) cell NOT launched, ckpt deleted; db9aba29), rq234 (78370075).
* Next: §179 READOUT (Theory 259c40cd, ~$0.8): s179_blindscaffold_s1 = b240's run + --blind_decoder in inject169's fade path, p 0 held to 24k
  (--fade 24000,48000 --stop_at 24000) + s179 fields + 173-1 package; gates (no problem tokens in the decoder, step-0, gradflow, flags-off bitwise, Audit).
* Harness: vdeploy STUCK_LOADING = stopped AND no pull progress (8ae7fc62); smoke_plain cleans up (897532f1); drain keeps ckpt_last for runs in
  scratch/keep_ckpt.txt (689a34b5; list now empty); inject169 resume verified bitwise (1 thread, 28c31cc3).
* Landed today: §169 K0 / K8 (FAIL); §170 DIRECT s1 / s2, K0 s1, cot s1 / s2 (eval-only, OOM fix), K8 s1; §171 K8 / BLIND; §165 K8 / K16; §168 std s1 / s2; §172 (closed).
* Harness today: FAST_ADD (-15 %); HOSTPROBE (works: b234 rejected at 1.21 s/step); eval-only paths in nested170 / inject169; memchecks count AdamW state;
  das167 --fit_micro / --rm_ks; halt173 = §173 halt + the 173-1 package (--dir177, pair-segment runs too).

### Local (v2 workers c4060 / c3070 / c2070 <- scratch/c1_backlog_<card>.txt; halt file scratch/<tag>.halt stops a worker)
* 4060: proto177_kmatch_s1b (masked, 7.5 s/step, ~17 h); 3070: proto176_rk_s1_3070 (re-run on current code); 2070: proto175_auxk8 (fp32).
* Queued post hoc: §165 K8 / K16 halt + 173-1; §169 K8 s176_ref + halt + 173-1; §170 K8 halt + 173-1.
* Voided today (aborted unread): proto176_rk_s1_4060, proto177_kmatch_s1_4060 (.void / .void2).
* 2026-10-09 10:25 (Audit): proto175_auxk8_s1_2070, proto176_rk_s1_4060, proto176_prog_s1_3070 (and the later proto177_{tidx, fadetidx, kmatch}_s1) are DESCRIPTIVE
  8k local previews read by NO rule (Theory notes 175-1 / 176-2); any §175 / §176 / §177 rule change after a preview read may only tighten. Each dir's args.json
  records train_gpu: cross-arm preview differences are card-confounded (2070 fp32 / 3070 / 4060 bf16). Queue order per card: §175 / §176 preview, then §177's.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (20:03; session "claude-c6"; peers by name: "Audit" / "Experiments" / "Literature"):**
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
  - **Scored today (2026-10-09, evening):**
    - **Closed:**
      - §165 / §169: neither K arm exact; memory unused.
      - §166.
      - §167: S5 3 / 3 DECOMPOSABLE (≤ 64-d); MULT not shown DECOMPOSABLE (s2 t4 only; the k = 64 fit collapses on MIX).
      - §172.
      - §176: neither RK nor PROG rescues.
    - **§170:** n* = 3; K0 fails. K8 not DEEP: it uses its passes but only to about DIRECT, and overthinks. TEXT-COT 9–12 mean 0.535.
    - **§171:** gate closed.
    - **§177:**
      - (a) index: no rescue (available but ignored).
      - (d) scaffold gate failed, so (d) is not launched.
      - (f) step-matched K running (b236).
    - **§168:** std D* = 3 (both seeds memorise). The 168-3 sanity gate (G1 / G2: b237 / b238) runs before NA D* is read. Step 2 held.
    - **Running:** §175 STEP-AUX (b235), §178 FROM-SCRATCH (b241 / b242), §179 READOUT (the blind-decoder scaffold; just registered).
    - **Leading hypotheses for the think-pass failures:**
      - no per-pass learning signal (Goyal);
      - the end decoder's view of the problem starves the memory (§179 tests this);
      - pretraining or capacity (§178 tests this).
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
    - scratchpad `watch_land_v10.sh` (path + mtime; s15[5-9] / s16x / s17x; eval / what_state / posthoc / anchor / das / halt / eval_s17*_*; skips *_tim* and *partial*);
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
