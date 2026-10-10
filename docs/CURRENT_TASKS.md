# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-09 22:23 EDT** (local = UTC − 4; Vast hosts log UTC).

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

### Vast (6 hosts, ~$1.0/h; 22:23 2026-10-09)
* Live: b235 §175 aux K8; b236 §177 (f) kmatch; b237 / b238 §168 G1 / G2;
  b243 §179 s179_blindscaffold_s1 (STARTED 22:12, 89.221.67.136:17397, HOSTPROBE 0.649) and b244 s179_seescaffold_s1 (STARTED 22:21,
  46.107.144.5:28451), both from b1e93088: --fade 24000,48000 --stop_at 24000 --p0_evals 12000,24000 (+ --blind_decoder on b243), then 173-1 package;
  ~4.5 h each; eval_p0_12k at ~2.2 h -> report 12k / 24k to Theory (rules: score_s179; note 179-2 len_err = top-digit placement).
* §178 closed for slot 1 (note 178-2: §170's test exclusion removes ~88% of n = 1, so scratch never sees the table; sanity d965952b): DIRECT / K0
  landed at chance (41351972); K8 (b241) STOPPED at 23.3k, aborted unread (s178_k8_s1.void, 0d72c67e). s178b (n >= 2 exclusion only) parked until §179.
* Note 177-6 (d) gate FAILED (12k full 4x4 at p 0 = 0.0; db9aba29): no (d) cell; ckpt deleted.
* Harness today: vdeploy STUCK_LOADING (8ae7fc62); smoke_plain cleanup (897532f1); drain keep_ckpt.txt (689a34b5, list empty); inject169 resume bitwise
  (28c31cc3); §179 blind decoder / p0 evals (b1e93088, gate179 all pass).
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

* **Theory (03:00 2026-10-10; session "claude-c6"; peers by name: "Audit" / "Experiments" / "Literature"):**
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
    - **§168:** std D* = 3 (both seeds memorise). The 168-3 gate STANDs: G1 reproduces 0.766, while G2 p8 is at chance from depth 3, so the cause is the data (869bdd25).
      - 168-4: step 2 must beat a curriculum-trained DIRECT; D*_curr is measured on p8 NA first.
      - 168-5: D* = 3 on NA (chance at every depth, both seeds; 6a24ed5e). Step 2 held.
    - **§178 slot 1:** UNINFORMATIVE (178-2: the test exclusion leaves 58 of about 470 n = 1 items, so the scratch model never sees the table). K8 aborted unread.
      s178b (n ≥ 2 exclusion) is parked until §179 reads.
    - **Running:**
    - **§179 READOUT: CAPACITY** (d0fe00ec). Blind 0.0 at 12k / 24k; columns 1–2 exact from memory alone; c3 0.351. The decoder's view is not the problem.
      - §180 DEC-DROP was never registered and is dropped. The seeing twin b244's 24k is descriptive (lands ~03:15).
      - s178b registered (note 178-3, 2ee99c13; exclusion for n ≥ 2 only). Contract with Experiments: CPU sanity, then --time_steps, launching after §181's timing.
    - **§175 STEP-AUX slot 1 closes, not DEEP** (15485a0f). Pooled 0.787 vs FINAL-only 0.351, with one reduction per pass (K′ = 4 solves n ≤ 4; K′ = 0 at chance).
      - Limits: per-step errors compound, and nothing past the trained K (K′ = 16 hurts; n 9–12 at 0.20).
      - **§182 STEP-AUX-HOLD registered** (566616ef). NOFADE / WINDOW / HOLD; DEEP at K′ 8, STABLE = K′ 16 vs 8 (K′ 24 beside it).
        - The contract is with Experiments (rough $8–10). Order: §181, then §182, then s178b. A post-hoc §173 halt on s175 is queued on the 3070.
    - **§177 (f) slot 1:** not HIT / not FAIL (pooled 0.506; passes not distinct). s2 not run (note 177-7).
    - **§181 INTERLEAVE registered** (ba4cf16e). k think passes before every answer token, the memory carried across tokens; k1 / k2 on §161's mix, 48k.
      - The contract is with Experiments: --interleave k, gates 1–7, --time_steps projection to Audit (rough $2–3 / $3–5), then the 3070 k1 8k preview.
    - **Leading hypotheses for the think-pass failures:**
      - no per-pass learning signal (Goyal);
      - the end decoder's view of the problem starves the memory (§179 tests this);
      - pretraining or capacity (§178b, parked).
      - Prior for the second: posterior collapse (VLAE 1611.02731); SHORTCUT is its prediction.
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
    - `watch_land_v10.sh` (path + mtime) hit its max timeout and is not restarted. Landings come from Experiments' reports plus one-off checks against seen_v3.txt;
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
