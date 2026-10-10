# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-10 09:12 EDT** (local = UTC − 4; Vast hosts log UTC).

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

### Vast (4 hosts; 09:12 2026-10-10)
* Live: rq249 §181 s181_interleave_k2_s1 (77.29.35.113, from 05:08; k1 landed d72491f5: 4x4 0.0); b251 §182 s182_nofade_s1 (185.62.108.226, from 09:04;
  ~11.8 h; gates the GSM8K draft; Audit OK; nested170 87f4021a); b252 §183 s183_ilblind_k1_s1 (89.221.67.137, from 09:11; <= 3.4 h; inject169 07839b2c).
* §182 WINDOW / HOLD (~$12-14) HELD for the user's approval (Audit carries it; note 182-4). Queues: write from b251's line + --think_k 12 --k_window 1,4
  --kmatch_impl masked (+ --aux_hold); timing d8e0d45d (1.96 / 2.11 s/step, micro 2).
* Local: 3070 proto183_ilblind -> ilseeing; 2070 proto183_ilseeing -> ilblind (compressed 8k: --steps 8000 --sched_scale 0.1667); 4060 idle (nothing fits).
* Landed overnight / morning: §179 (CAPACITY); §175 closes + AUX-DONE (exact pass-n halting at n <= 8) / T-SHIFT (completion at n 4-6, counter at 8) /
  PATH-INDEP (agree 0.29); §177 (f) slot 1; §168 G1 / G2 / NA (D* inputs); §178 closed (178-2 / 178-4; s178b not launched); §181 k1 FAIL-profile.
* Harness: fill_generic prefers scratch/vast_prefer_ips.txt (bfc82c7b); exclude list grew (slow / stuck / download-failed hosts); requeue_watch (one
  instance, Oct 9) re-checked; nested170 eval_only wg_hist fix (8a7c191a); halt173 --auxdone / --tshift; inject169 --interleave / --blind_decoder.
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

* **Theory (08:16 2026-10-10; session "claude-c6"; peers by name: "Audit" / "Experiments" / "Literature"):**
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
      - s178b: CPU gate failed (n = 1 flat at 0.13–0.16), so not launched. **§178 closed** (178-4, 9145c8ea): §175 answers the pretrained side.
    - **§175 STEP-AUX slot 1 closes, not DEEP** (15485a0f). Pooled 0.787 vs FINAL-only 0.351, with one reduction per pass (K′ = 4 solves n ≤ 4; K′ = 0 at chance).
      - Limits: per-step errors compound, and nothing past the trained K (K′ = 16 hurts; n 9–12 at 0.20).
      - **§182 STEP-AUX-HOLD registered** (566616ef). NOFADE / WINDOW / HOLD; DEEP at K′ 8, STABLE = K′ 16 vs 8 (K′ 24 beside it).
        - The contract is with Experiments (rough $8–10). Order: §181 (running), then §182. A post-hoc §173 halt on s175 is queued on the 3070.
    - **§177 (f) slot 1:** not HIT / not FAIL (pooled 0.506; passes not distinct). s2 not run (note 177-7).
    - **§181 k1: FAIL** (652176bc). The single-pass column profile; skip drops about 0. k2 (b249) is running.
    - **§183 INTERLEAVE-BLIND registered** (d5c160a7). §181 k1 with a blind per-token decoder; it tests 'per-pass signal through a view-free reader'.
      - Local 8k compressed previews: blind and seeing on each of the 3070 and 2070. The cloud cell follows --time_steps.
    - **§182:** NOFADE (b251) is deploying. WINDOW and HOLD wait for the user's approval (note 182-4).
      - §175 descriptives done: AUX-DONE (exact t = n, but counter-like at n = 8 per T-SHIFT) and PATH-INDEP (answer_agree 0.292).
    - **Drafted, not registered: STEP-AUX-GSM8K** (Audit's 04:05 lead; scratchpad sec18x_gsm8k_draft.md; Literature b908b475).
      - §175's recipe on GSM8K-Aug (K = 6, per-pass equation aux), plus an in-pipeline TEXT-COT baseline cell.
      - C3 against PCCoT 49.48 (SOTA claim: above by 2.8 in 2 / 2 seeds) and against our TEXT-COT.
      - Registers after §182 NOFADE. **The launch needs the user's explicit approval** (the small-tasks-first directive); Audit carries the projection.
    - **§181 INTERLEAVE registered** (ba4cf16e). k think passes before every answer token, the memory carried across tokens; k1 / k2 on §161's mix, 48k.
      - Cells running: k1 b248 (04:34) and k2 b249 (04:46). The 3070 k1 preview is descriptive.
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
