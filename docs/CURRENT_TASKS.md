# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-10 14:27 EDT** (local = UTC − 4; Vast hosts log UTC).

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

### Vast (7 hosts, ~$1.4/h; 14:27 2026-10-10)
* Live cells (check git status on each dir every turn; drain waiters cap at 2 h):
  - b251 §182 s182_nofade_s1 LANDED 20:35 (3a0253c7; Theory 384a00c2: not DEEP, not STABLE, pooled 0.786); post-hocs normcap / auxdone / tshift on the 4060 backlog
  - rq253 §182 s182_window_s1 (89.221.67.136, from 12:29, ~26 h) / b254 §182 s182_hold_s1 (77.29.35.113, from 12:48, ~28 h) - user-approved
  - rq255 §184 s184_reweight_s1 (from 13:32, ~11.8 h) - Audit OK
  - b257 §186 s186_ilblind_k4_s1 (58.136.231.3, from 14:16, ~9 h est.)
  - §168s2 slot 1 USER-APPROVED (168-11, bs 64, ~$7.5-11). CHUNK_DELTA g5 PASS (d7e9abea). b263 stepaux (chunk/ckpt/cache) deploying 23:2x; b264 currdirect+textcot at the next free slot, then §185 den (b261), then rms (b262). Fleet <= 5 (Theory).
* Fleet (5): WINDOW b253, §168s2 stepaux b263, ProsQA baselines b264, §185 den b261, s184 s2 b265 (deploying). Next free slot -> §185 rms (b262), then §187 (b266). b267 (168-12 STEPAUXRW) USER-APPROVED 2026-10-11 16:05 EDT ("approve", typed by the user in Audit's session, relayed by Audit) - launches into the next free slot (WINDOW b253) with a drain waiter. Local: 2070 deepaux s184 -> s184 cap sweep; 3070 / 4060 §185 den / rms 8k previews then the shared reads. BLIND (note 184-1) f* = 2.0 FIXED (Theory, before any s2 read): on s184_reweight_s2 landing queue its train-C normcap then halt173 --normcap --normcap_c <s2>/eval_s185_normcap.json --normcap_factor 2.0 --out <s2>/eval_normcap_fstar.json; s2 numbers may be read now.
  args + --add_norm den|rms; package lines: --dir177, --auxdone, --tshift, --normcap).
* Landed today: §181 k1 / k2 (fail profile); §183 cell (skip collapses read columns; inner columns chance); §179 BLIND / SEEING; §175 post hocs
  (AUX-DONE, T-SHIFT, PATH-INDEP, memnorm, NORM-CLAMP, NORM-CAP-GLOBAL: cap 1x keeps 0.696 at K' 16 vs 0.414); §168 G1 / G2 / NA; §177 (f) slot 1.
* Local: §183 previews (3070 blind -> seeing; 2070 seeing -> blind; 4060 seed-2 blind -> seeing), compressed 8k schedule.
* Harness today: requeue_watch misses EXTRA_120 (rq256 crashed without the p8 data; fix = per-queue .extra sidecar, after tonight); fill_generic prefers
  scratch/vast_prefer_ips.txt; deploy shas in logs/deploy_shas.log.
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

* **Theory (20:38 2026-10-10; session "claude-c6"; peers by name: "Audit" / "Experiments" / "Literature"):**
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
    - **§181 closed: neither HIT** (90395c97). Both arms keep the single-pass profile, and skip drops are about 0.
    - **§183 closed: FAIL, NOT SUPPORTED** (201d4cea). The blind reader makes passes causal, but inner columns stay at chance: per-pass compute binds.
    - **§186: FAIL / NOT** (179f526a). **The mult interleave line is closed** (pre-registered stop); within-token identical-input passes sit idle (cos 0.993).
    - **NORM-CAP-GLOBAL:** a zero-training C6 fix (t16 0.696 vs 0.414; 76% of the fall, 97% of the oracle). C came from test items; 185-1 takes C from train items.
      - Local previews (3 cards, blind / seeing) are descriptive and still landing.
    - **§182 NOFADE: not DEEP, not STABLE** (384a00c2). The fade changes nothing; it shares one trajectory with §175 to step 36k. WINDOW (b253) / HOLD (b254) are running.
      - 168-10: §168 s2 and GSM8K use no fade.
      - Running: §184 (b255) and §186 (b257).
      - §185: den launches when b259 ends, rms at the next free slot (fleet ≤ 5). NOFADE post hocs are on the 4060.
    - **§168 s2 projection:** with CHUNK_DELTA, stepaux runs at 6.96 s/step at bs 64. Slot 1 is about $7.5–11 at bs 64, about $4.5–6.5 at bs 32.
      - **The user approved slot 1 at bs 64** (168-11). g5 PASSED (d_chunk 0.0049 vs d_reorder 0.0029; 8.1×). **stepaux launched on b263** (~23 h); the baselines go on b264 next.
      - Fleet priority: stepaux, then the two baselines in sequence, then §185 den / rms. GSM8K stays unapproved.
      - §175 descriptives done: AUX-DONE (exact t = n, but counter-like at n = 8 per T-SHIFT) and PATH-INDEP (answer_agree 0.292).
    - **Training-schedule / architecture proposals (the user asked, 2026-10-10).** Ranked by the user's priorities with Audit, prior-checked by Literature (09f395a8):
      - #1 AUX-REWEIGHT = §184. **Slot 1: HELPS + DEEP** (598b02b7). Pooled 0.974 vs 0.787; the first DEEP think-pass model on INJECT, one seed.
        - s2 queued (~$3). NORM-CAP / AUX-DONE / T-SHIFT on it run locally.
        - **The reweighted ProsQA cell (168-12) is user-approved** (16:05 via Audit; 168-14). b267 launches at the next free slot, and 168-13 (a) will then be read.
        - **§187 AUX-NOPASS registered** (87b7e23d; ~$3, after §184 s2): does pass-token keying limit depth beyond n = 8?
        - **The depth limit is narrowing** (496bdb8d). §184's deep-item handle decays from pass 3, inside the seen tokens, and the NORM-CAP HURTS deep items (it's a within-depth C6 aid only).
          - Interference hypothesis: 185-3 pre-states that DEN / RMS give no depth change.
          - **187-2 read STRUCTURAL (both models), so ADD-DECAY is dropped.**
          - The count effect differs by model: §184 +0.158 for n 9 → 10, NOFADE +0.031. 187-3's count test (n 11 / 12 rows) is running on CPU.
      - #4 becomes **§185 ADD-NORM** (2a1aac15; ~$6). The additive read had no normaliser.
        - NORM-CLAMP: about 80% of §175's overthinking is that read's growth.
        - Arms DEN and RMS. Launch after NORM-CAP-GLOBAL (6k items, 3070) and NOFADE.
      - #5 aux-first: an arm inside the ProsQA cell.
      - #2 path-independence (Anil's recipe): if §182 HOLD's answer_agree < 0.5.
      - #3 halt head: when a C5 claim is wanted.
    - **§168 step 2 registered: STEP-AUX-ProsQA** (note 168-7, 410d7cbc; the user's 'smaller language task first').
      - §175's per-hop aux on p8 NA, with K = 10; CURR-DIRECT and TEXT-COT bars.
      - p8 depth = unbranched serial lookups (168-6).
      - Experiments: implement, gate, --time_steps only. **No launch until the user approves** slot 1 (~$13–17, via Audit).
      - The fade is fixed by §182 NOFADE in a dated note. GSM8K-short follows this one.
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
