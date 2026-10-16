# Current tasks — live handoff

**Purpose:** what is running, where, how to check or stop it, what waits to be scored, and the standing rules. Update
it whenever a job starts or finishes. History before this rewrite (dated state blocks, 09-28 reorganisation, theory-pa
hosts, Pilot sections): `docs/archive/CURRENT_TASKS_history_to_2026-10-03.md`.
**Last updated: 2026-10-14 16:3x EDT** (local = UTC − 4; Vast hosts log UTC).

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
* 02:3x 10-15 (Experiments): §205 IIT-SERIAL: s205_fill (one filler, driver-pinned) - b298 s3 RUNNING on 209.226.246.174:53250 (56051379, 595.84,
  STARTED 01:53, HOSTPROBE OK); b296 s1 / b297 s2 wait for a 570.172.08 offer <= $0.28 (retry every 10 min); guard s205 cap $18. gpu_watch is now driven
  by scratch/gpuwatch_autoadd.sh (291 295 296 297 298; restarts gpu_watch by its own pid when a cell STARTs or the watcher exits). §206 JOINT-COT: harness
  in boxes/gsm8k_jcot.py + flag-gated stepaux hook (uncommitted until gates finish) with the §9 diagnostics (KD-ROUTE per log step, TEACHER-CONTEXT,
  term norms / kappa_hat every 500, T2-ACC, KD-GEN, KD-ANS, eval.json kd_route / kd_ans / kd_gen / t2_acc / cot_valid_final); queues b299 (J0) / b300 (J1)
  written as .txt.hold - launch only after Audit's diff review; guard $6; must not run a second filler beside s205_fill.
* 02:3x 10-15 (Experiments): §206 LAUNCHED (Audit APPROVED; a1634095): s206_fill -> b300 J1 (line 1 = W eval-only ref -> rename
  gsm_lat_warm_s1_ref/eval.json to gsm_lat_warm_s1/eval_s206ref.json after the drain; delete the _ref model.pt) then b299 J0; 570.172.08 reserved for
  s205_fill (disjoint offers). Guard s206 $6 (G8 = J1 step 200; J0 mirrored on J1; GUARD_STOP destroys both). Scorer theory/score_s206.py (Theory).
  03:0x: J1 b300 on 46.63.50.129:51033 (56055329, STARTED 02:36; G8 PASS 1.608 s/step, $1.57 / cell; W ref acc = landed), J0 b299 on 171.5.176.46:45477
  (56058113, STARTED 03:01); §205 s2 b297 on 218.150.198.117:51947 (STARTED 02:45); s1 b296 still waiting for 570.172.08. All in gpu_watch + gen8 drains.
  04:3x: §205 spend cut (Audit): post-gamma rates -> family ~$17.8 vs $18. s205_fill replaced by s205_fill_s1.sh (b296 only, <= $0.24/h);
  cost_guard s205 replaced by s205_rate_guard.py (CHECK every 4k; family > $18 -> stop s1 before its 4k, else ALERT). If no s1 offer by ~10:30, tell Audit
  (s1 is a converged-control seed; Audit takes 2-seed vs more $ to the user).
  08:24: §206 J1 LANDED (05b0c763; acc 0.138, cot_valid_final 0.344, kd_route share_mem 0.073; ~$1.63); W ref -> gsm_lat_warm_s1/eval_s206ref.json (= W). J0 ~10:15; Theory scores.
  09:05: §204 F-P LANDED (49c54350; acc 0.1387, validsel 0.1471; ~$2.56); A ref -> gsm_lat_aug_s1/eval_s204ref.json. CLOCK / PROBE-LEAK eval-only via boxes/stepfactor_probe.py
  (e02a54cb; clock reproduces clock_A.json): clock F-P (CPU) + probe A then F-P (3070 fp32) running -> <run>/eval_s204_clock.json / eval_s204_probe.json; then commit + tell Theory.
  10:40: §206 COMPLETE - J0 LANDED (f4b6b432; acc 0.1440, cot_valid_final 0.358; ~$1.61; pair ~$3.24 / $6); Theory scores. §204 add-ons committed (1c4b8215).
  §207 GSM-SWAP done on A / B / W / F-P / J1 / J0 (scratch/gsm_swap.py; 7fd4fa69 + f4b6b432): all RECOMPUTES-FROM-INPUT except B MIXED (0.799; ratio-form question with Theory).
  §205 s1: no 570.172.08 offer in 6 h (none on the market at any price); options (wait / unpaired on 595.84 / 2 seeds) + the $0.06 margin sent to Audit for the user;
  s205_fill_s1b keeps retrying meanwhile. Live cloud: b291 (§193 s2), b297 / b298 (§205 s2 / s3).
* 12:4x 10-15 (Experiments):
  - §205 s1 UNPAIRED (user via Audit ~11:0x-11:15, cap $18.50): b296 on 85.90.17.171:18155 (595.71.05, $0.2228/h, STARTED 11:19, admitted at $3.73);
    s205_rate_guard (cap 18.5, CHECK every 4k) + s205_fill_s1c (done). Family ~$15.4.
  - §208 QUESTION-ONCE (759b3396 harness, Audit APPROVED): b301 gsm_lat_ansbn_qonce_nopass_s1 on 194.14.47.19:22364 (STARTED 12:00; 0.42 s/step,
    ~$0.89; lands ~14:55). Guard s208 $3.20. G6 PASS on the smoke QO model (scratch/qonce_eval.py); post-landing: python scratch/qonce_read.py
    boxes/runs_fsq/gsm_lat_ansbn_qonce_nopass_s1 all --device cuda:1 (fp32; G6 re-run first) -> eval_qonce.json; Theory writes score_s208.
    Until §208 lands: NO edits to pass_inputs / ans_prefix / modules §208 imports (note 208-1).
  - §209 CODI-AT-OUR-BUDGET (boxes/codi_port.py, Audit APPROVED; gates scratch/gate_s209/gate.json): b302 gsm_codi_4ep_s1 on 190.226.72.134:40613
    (STARTED 12:29; micro 32 x 4; 2.14 s/step -> ~7.3 h, lands ~20:00); guard s209 $4 (GUARD_STOP). Train counts: scratch/gate_s209/train_counts.json.
  - gpu_watch via gpuwatch_autoadd 291 296 297 298 301 302.
* 15:3x 10-15: §208 LANDED (~$0.88) + eval_qonce.json complete (G6 PASS on the final model; Theory scores). Live: b291, §205 b296/b297/b298, §209 b302 (~19:45).
* 19:5x 10-15: §209 LANDED (b3a762ac; acc 0.1888, 4.0 epochs, ~$1.51); k_latent_ablation 0-latent 0.325 = an explicit-CoT arm (87% '<<..>>' decodes; eval_latent_texts.json). Theory (offline, message queued) scores. Live: b291 + §205 b296/b297/b298 (~$15.4 / $18.50).
* 20:2x 10-15: §210 CODI-BUDGET-20 LAUNCHED (de919de5; 06da84e9): s209b_fill (A100 SXM4 <= $0.60, admission $11.5, G1' line 1 via boxes/codi_g1.py) + guards s210gt ($11.5 @200) / s210 ($12); no A100 offer yet (tell Audit by ~08:00 if none). §193 s2 LANDED (e41fbf5a). §205: s205_post_landing.sh runs scratch/s205_eval_iit.py per seed after its drain -> eval_iit.json; ping Theory (pipe 85e9119e) per seed. Theory / Audit now on local pipes 85e9119e / 4dc2a843.
* 22:4x 10-15: §205 s3 LANDED + eval_iit (ac986c1b; hybrid 0.996) -> Theory: MET-NOT-ENTERED, arm FAILED (bc4f236b). Retention per Theory: per seed keep ckpt_model_8000 + model.pt, delete ckpt_model_4000 / 12000 / 16000 AFTER its eval_iit.json is in (s3 done; s1 ~01:40 / s2 ~02:50 by hand - s205_post_landing.sh is running, do not edit it). §210 on RTX 3090 (user 'go back to 3090s'): s209b_fill (GPUS RTX_3090, PMAX 0.25, BAD0=1), vdeploy STUCK rule + 'Retrying in' pulls. ckpt_mirror.py running for s205 s1 / s2 (add 303:gsm_codi_20ep_s1 when §210 trains).
* 03:4x 10-16 (Experiments): §205 COMPLETE - s1 b296 LANDED 02:12 (2f6920f3; ~$3.44) + s2 b297 LANDED 03:40 (2b66a327; ~$6.02); both converged (acc n1-9 ~1.0, IIT hybrid ~1.0, SIIT keep ~1.0, g2 0.098 / 0.081); family ~$15.41 / $18.50; instances destroyed; per seed 8k ckpt + model.pt kept for the 8k descriptive tests (203-4 / 203-6, on Theory's word). §211: N2 recovered from the anneal-end bump (<0.3 by ~10k; 211-9); N3 stuck after the anneal end (answer_ce ~1.2-1.4 plateau, advanced to n<=8 by CAP at 13k; 1.04 at 24k when p_TF hit 0; 211-12 rows 18k/24k sent, 30k pending via scratch/s211_rows.py). 211-14: vbank ver_keep hook + registered bank_ablate (29e6d43d). ckpt mirror fixed (local == pre-copy sha; 3/3 failures before) and §211 mirrors restarted 03:3x - confirm first MIRROR s211_* lines. Egress $/GB b304 0.0052 / b305 0.0039 / b306 0.039: add to realised $ at landing (~<= $0.25). Cells land ~05:00.
* 01:1x 10-16 (Experiments): §211 ALL THREE RUNNING. b304 N2 85.30.169.224:40210 ($1.38); b305 N1 95.86.5.33:40496 (56201193, $0.1619/h, $1.08); b306 N3 89.221.67.136:16521 (56202519, $0.2294/h, G-TIME 0.518, $1.70; released per Theory 211-8). Family proj ~$4.18 / $9 (incl. BAD $0.02). gpu_watch autoadd now 296 297 303 304 305 306. b304 end-of-anneal bump (answer_ce 0.32 -> 1.17 at 8k -> 0.71 at 8.6k) sent to Theory: continue unchanged (211-9); watch scratch/s211_watch.py sends b304 10k / 12k rows + each stage advance (N1 = stage-only comparator).
* 00:5x 10-16 (Experiments): §211 b304 N2 ADMITTED 00:30 on 85.30.169.224:40210 (56198062, 595.71.05, $0.2014/h; G-TIME 0.476 s/step micro 1, proj $1.38); mirror (logs/ckpt_mirror_b304.out), guards s211_b304 ($3 alert) + s211fam ($9 GUARD_STOP). b305 N1: 1st host BAD ($0.023), filler retrying. b306 N3 HELD FOR THEORY (Audit's rule, note 211-6): G1b split pa n4-8 CE 9.07 -> 2.42 -> 2.37 vs pamct 9.07 -> 7.68 -> 10.47 (scratch/gate_s211/g1b_split.json). eval_bs 32 (note 211-7); eval memcheck passes. Keep the s211_g1b_pa_r2 / pamct ckpts until Theory decides.
* 00:1x 10-16 (Experiments): §211 LAUNCHED (Audit GO 00:1x): G1b PASS pa/carry (acc43412), compressed anneal smooth (log in boxes/runs_fsq/s211_anneal_smoke). s211_fill (QS b304 N2, b305 N1) running: deploy -> HOSTPROBE -> G-TIME admit <= $3/cell -> ckpt mirror + per-cell $3 alert guard + family s211fam $9 GUARD_STOP=1. b306 N3 HELD until the pamct G1b (2070, boxes/runs_fsq/s211_g1b_pamct) passes the same drift / CE criteria - then mv .hold -> .txt and run QS="306:s211_vb_pamct_s1" nohup bash scratch/s211_fill.sh >> logs/fill_s211.out 2>&1 & (no Audit wait; send Audit the line). WATCH (Audit): rec_ce / answer_ce over steps 7.5k-8.5k (end of the anneal); a bump > 0.3 nats lasting > 1k steps -> Theory. Post-landing: drain, then python -m boxes.vbank_eval --run boxes/runs_fsq/<cell> --device cuda:0 (eval_bs 64, fp32) -> eval_vb.json; Theory scores (score_s211).
* 23:3x 10-15 (Experiments): §211 VERSION-BANK Stage 0 built (boxes/vbank.py, new module; commits 1a3bcd47..86c5391d). Gates green (scratch/gate_s211/*.json): E_t 0 diffs / 100k vs red-team ref; FP 1.6e-15; PA c=1e4 == CARRY bitwise; masked-K 5.7e-15; G1-cal m99 0.0198, ratio 0.68-0.85 e^5; path gradflow; MCT W1/W3. G1b 2k smokes running locally (s211_g1b_pa 4060 / _carry 3070 / _pamct 2070; analyse: python scratch/gate_s211.py g1b). Cloud HELD for Audit's diff review: queues scratch/gen_q_b304/305/306.txt.hold (N2/N1/N3), filler scratch/s211_fill.sh (G-TIME admission <= $3/cell, <= $9 family incl. BAD, ckpt mirror + per-cell guard on admit). To launch: mv the .hold files to .txt, then nohup bash scratch/s211_fill.sh >> logs/fill_s211.out 2>&1 &. eval_vb.json (local fp32 post-landing) still to write (boxes/vbank_eval.py). §205 s1 b296 / s2 b297 land ~01:40 / ~02:50 (eval_iit auto); §210 b303 ~08:30.
* 19:48 10-15 (Theory): §209 SCORED MIXED (77b17549): CODI4 0.189, r 0.24, +0.042 vs A; ARCH-BOUND excluded, BUDGET-BOUND within 1 SD; same weights do text CoT at 0.325. Step 2 (budget scaling) = separate user decision, not proposed. Theory waits on §205 (score at landing after eval_iit.json).
* 20:17 10-15 (Theory): §210 CODI-BUDGET-20 REGISTERED (de919de5; user approve 20:15 via Audit; cap $12; A100): gsm_codi_20ep_s1 = 4ep args with --epochs 20; G1prime (released ckpt within 0.010 of 0.425 on the A100) + G-TIME step 200 <= $11.5; scorer score_s210.py (10144879). Contract sent to Experiments.
* 20:25 10-15 (Theory): §193 s2 SCORED (64770c16): JOINT + LONG-STABLE (K′24 0.954, DEPTH 0.647) -> C6 CLAIM HOLDS 2/2 for RMS x WINDOW. §210 Audit-OK, filler waiting for an A100. Theory waits on §205 s3 (~22:00) eval_iit.json.
* 23:0x 10-15 (Theory): §205 s3 MET-NOT-ENTERED -> arm FAILED by rule (bc4f236b; s1/s2 pending for HARMFUL). §210 running (G1prime PASS 0.4246). DEEP RESEARCH (user 21:58 / constraint 22:07) delivered: docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md (fc852c5e) + docs/research_passuse_2026-10-15/ + LIT_PASSUSE_PRIORART; DRAFT only, awaiting user OK via Audit (Stage 1 cap $9).
* 23:22 10-15 (Theory): §211 VERSION-BANK REGISTERED (5e554c39; user approve 23:06; Stage 0+1, cap $9): cells s211_vb_pa_s1 (N2 main) / s211_vb_carry_s1 (N1) / s211_vb_pamct_s1 (N3); scorer score_s211 (3d0d72af); eval_vb.json contract = note 211-1. Experiments builds Stage 0 gates (boxes/vbank.py); Audit diff review before cloud; off-host ckpt mirror required.
* 01:3x 10-15 (Experiments): §204 b295 RUNNING on 66.203.214.60:50792 (56040673, 595.84): A ref re-run done (bitwise = A; renamed to
  gsm_lat_aug_s1/eval_s204ref.json after the drain); F-P gsm_lat_factor_s1 started 00:25, lands ~12:30; guard s204 projects $2.55 / $3.90; gen8 drain +
  gpu_watch (b291 + b295). IIT-SERIAL G-TIME DONE (20efb91d: +83% s/step, 18.0 GiB micro 1, ~$4.7 / 48k seed). DAS closed: PCA swap COMPACT-FIT-LIMITED
  (8d766901), refit shows Adam scale destroys the 0.71 PCA init (65b97269). JOINT-COT-L1 harness drafted in scratch/_tmp/gsm8k_jcot_draft.py (not
  committed / not importable) pending the user's approval + Theory's registration. Local cards idle by design.
* 00:1x 10-15 (Experiments): §202 W LANDED 0.146 (e3c4ff16; W-LIFT NEUTRAL) -> §204 A-base b295 (A eval-only ref + F-P gsm_lat_factor_s1; Audit GO;
  guard scratch/cost_guard.py s204 3.90) filling: 4 STUCK_LOADING hosts so far (~$0.16; reason now logged, d71f41e5). IIT-SERIAL --iit done + Audit
  OK (27d05cef / 62343096); G-TIME b294 queued behind b295 (one filler). DAS-USED: P0 all 16, P1 pilots, k-ladder on rms s1 -> no compact used
  subspace <= 1024-d (35bf38de) -> DAS fits STOP (no P2 for the other pilots, no Stage B); 4060 rms s1 P2 finishing (archive). Notes 203-5 / 203-6
  CONSUMED-MASK + INPUT-SWAP on all 16 (bb3799b7; note 203-8). Local 3070 / 2070 free.
* 21:4x 10-14 (Experiments): §202 C+ LANDED 0.336 (997ff5bd, ~$0.37); W b292 ~8k/12k, lands ~22:55. §204 harness final (7f707887 +
  ae289bcb: plan_ce, lotus_sep, --eval_out). DAS-USED pilot (scratch/dasused.py): P0a conditioned SERIAL on all 16 (12dd5cd4; note 203-4
  CONTRAST-SURVIVES), P0b on the 4 pilots (5d3e7396). 3070 batch (logs/local_batch_2130.out): P1 x4 pilots, then scratch/maskswap.py (notes 203-5 /
  203-6 CONSUMED-MASK + INPUT-SWAP) x15 (window s1 done) until ~23:30. Pending: P2 DAS fits (scope / timing question to Theory), --iit (IIT-SERIAL,
  flag-gated nested170 edit after local jobs stop importing it; Audit gradflow review before the <= $0.25 G-TIME host). §200 D SKIPPED (user).
* 20:3x 10-14 (Experiments): §202 W b292 (222.230.13.166:55189, 595.84, $0.243) ~4.4k/12k, lands ~22:40; C+ b293 (171.5.176.46:45132, 595.71.05,
  $0.208) ~10k/12k; guard done (pair $1.61). Each has a gen8 drain + gpu_watch (b291 / b292 / b293). The 18:00 W launch shipped no .extra
  (MODULE_MISSING bench.tasks) -> relaunched with EXTRA_120 (memory note). §203 SVR: G-pre PASS, G-use NOT-USED 9/9 (+ ALL-heads add-on 9/9) ->
  DO NOT LAUNCH (Theory 2b5b09e1 / a9182a47). §204 STEPFACTOR harness final 7f707887, gates PASS (scratch/gate_s204/gate.json), Audit-approved;
  launches after §202 lands on the branch its rule picks; its .extra must add bench/data/gsm8k_aug/s204_op_vocab.json. Note 204-1 SIM-CoT D1
  done (00d08bc4). Local cards idle (no registered local work).
* 17:5x 10-14 (Experiments): §200 A / B drained + committed (0.147 / 0.160; GSM-Symbolic local 01fcde1d); family ≈ $7.81, D not run. §199 s3 destroyed
  16:45 (step 35.9k; family $5.93; ckpts deleted per Theory). §193 s2 b291 training on 89.221.67.136:16081 (55989781, STARTED 16:50; gen8 drain + gpu_watch).
  §202 WARM-START: harness + gate_s202 PASS (091f4e12), Audit OK -> fill_generic_px s202 292 (W) then 293 (C+), DPH_MAX 0.28, 3090 only; guard
  scratch/s202_guard.py (pair cap $4, logs/s202_guard.log). Local: T4 rms s1 (2070) / s184 s2 (4060) running; s193 T4 done (9aad31c2); 3070 free.
* 16:3x 10-14 (Experiments, after the 13:45 LOCAL REBOOT killed every watcher / filler / local job): §200 A b287 + B b288 FINISHED on host
  (QUEUE_DONE 16:13 / 14:52 local); gen8 drains re-armed 16:27 (scratch/post_reboot_redrain_1014.sh). D b290's host 55966557 failed HOSTPROBE_SLOW
  (1.03 s/step) at 13:46 and idled -> DESTROYED 16:28 ($0.64); §200 family ≈ $7.88 > $7.50 cap, so D is NOT refilled (asked Audit). §199 s3 b285
  stopping: temp fetch scratch/_tmp/s3_fetch.sh -> md5 -> boxes/runs_fsq/s199_csr_s3, then vdestroy 55929684 + clear reserved_drivers.txt.
  §193 s2 b291 refilling (fill_generic_px DPH_MAX 0.30; the 13:43 attempt made no instance). Restarted: requeue_watch3, autoexclude_watch.
  Local T4 (201-7 / 201-8) restarted from scratch: 2070 s185_rms_s1, 4060 s184_reweight_s2, 3070 s193_rmswin_s1 (logs/s201_t4_*_r.out; partials *.void_reboot).
* 10:5x 10-14 (Experiments): §200: A b287 (218.150.198.117 / 570.172.08, $0.248) and B b288 (209.226.246.174 / 595.84, $0.258) training, land
  ~16:20 / ~15:00; C LANDED 0.321 (+ local GSM-Symbolic main 0.156 / p1 0.059 / p2 0.022); D HELD (family $6.58 w/o D vs the $7 cap; Audit asked the user
  for $7.50). Fillers: fill_generic_px (DPH_MAX, reserved drivers file, heartbeat), requeue_watch3 (per-queue .dph caps). §199: s1 b283 + s3 b285 (595.84,
  PAIRED 199-A3) running; s2 waits PAIRED for 570.172.08 via scratch/s199_dual_fill.sh (Audit cap: 4 more BAD HOSTs / family $9.50; logs/s199_bad_cost.log).
  Landed today: §198 a / b / c (family ~$6.29), §193 (~$7.81, 187-3 rows done), stage-2 pilot 24k (196-2 read UNCLEAR; ckpts deleted).
  §201: T1 / T2 on all 7 nested + control scored (fb1d39ed); T2-e scored (FD-on-signal = the soft halt); 201-6 queue: 4060 eval_timing_4060 (§198 / §191),
  2070 FD-on-signal on all models (+ s193), 3070 ProsQA dump -> T2. Owed: §198 (a) + §191 A / B eval_timing on a 3090.
  USER: please delete D:/Latent_s200sym and D:/Latent_smk199 (scratch copies, ~1 GB; a safety check blocks my rm of drive-root dirs).
* 01:2x 10-14 (Experiments): §200 GSM8K-AUG PILOT (user-approved Tier 1, $7 cap): harness 2d0964ab + name fix 64c98681, gate_s200 run 4 all 8 PASS,
  Audit OK. Timing host b286 (scratch/gen_q_b286.txt + .extra data sidecar) takes the FIRST free slot (chain_s200t_s199 in scratchpad: b286, then §199 s2
  570.172.08, then s3 595.84). b286's timing.json -> projection to Audit AND the user before any of the four §200 cells (A/B/C/D queues not built yet).
  §201 OVERTHINK-EXIT T1 running on the 2070 (scratchpad run_s201_t1.sh, 8 models ~6 h, dumps in scratchpad/s201_dump); T2 (scratch/s201_t2.py) runs per
  model after its T1 (CPU); T2-e (FA-on-signal) and the ProsQA s191 descriptive T1 still to build.
* 23:4x 10-13 (Experiments): fleet 5 = b277 §193, b280 §198a, b281 §198b, b282 §198c (STARTED 22:41, family proj ~$6.5), b283 §199 s1 (STARTED 23:13,
  570.172.08; A4 PAIRED under 199-A2, e4fb6c7b). Pairing rule = score_s199_a.paired() (step-500 tolerance 1e-6 x |loss_ctrl|, Audit
  settled); a4_check.json's pred-relative residual is descriptive only. s2 / s3: chain_s199c (DRIVER_EQ 570.172.08 then 595.84) waits for slots (~06:00); scratchpad a4_arm_s23.sh
  starts their A4 checks at STARTED (logs/a4_s199_s{2,3}.out). §199 ckpt_model 4/8/12/16k come with the drain -> probe_pathindep -> delete.
  Landed tonight: §191 A (a7aced23; NAME-MEM), §196 (208a3a6c; NO-EFFECT). Local $0 notes done: 199-M2 LMC (cc3061a7), 199-M3/M3a/M3b lag split
  (5225b140, ad1645bd). eval_timing191 (~$0.15) when a slot frees. path_indep cadence (probe_cadence.sh, 3070). Pilot leg 1 done 12k (1e3661cb); leg 2 -> 24k ~05:00-06:00:
  then the standard probe (n 3..8, K' 16) + --stage 6:6 (n <= 6, K' 12, 6-pass donor) on the final, loss_seen yes. KEEP ckpt_12k_leg1_end.pt until
  Theory posts the 196-2 reading, then delete it.
  §197 NOT registering (Theory) - no §197 cell. GSM8K pilot is a DRAFT (7ce8c78a) - nothing to build until the user approves via Audit.
* Live cells (check git status on each dir every turn; drain waiters cap at 2 h):
  - b251 §182 s182_nofade_s1 LANDED 20:35 (3a0253c7; Theory 384a00c2: not DEEP, not STABLE, pooled 0.786); post-hocs normcap / auxdone / tshift on the 4060 backlog
  - rq253 §182 s182_window_s1 (89.221.67.136, from 12:29, ~26 h) / b254 §182 s182_hold_s1 (77.29.35.113, from 12:48, ~28 h) - user-approved
  - rq255 §184 s184_reweight_s1 (from 13:32, ~11.8 h) - Audit OK
  - b257 §186 s186_ilblind_k4_s1 (58.136.231.3, from 14:16, ~9 h est.)
  - §168s2 slot 1: baselines CURR-DIRECT 0.508, TEXT-COT 0.735; STEP-AUX (b263) trained (train CE 2.5e-6) but test at CHANCE 0.490 (eval-only on the 4060 after the in-run eval crash; logreg n_cls fix 83f83e1a, --eval_only c6dc68fa). 168-12 STEPAUXRW (b267, user-approved) lands ~17:00 10-12 and will crash its eval the same way -> eval-only after the drain (keep_ckpt set).
* Fleet (4, 12:10 10-13): §191 A b271 (~21:30), §193 b277 (WINDOW+RMS, ~06:00 10-14), §194 b278 (~19:30), §196 GROKFAST-S3 b279 (11e1e88a, gate_grokfast all PASS, STARTED 12:05, same driver 595.84 as the s185_rms_s3 control, ~01:00 10-14); one slot held free (Theory). LANDED today: §188 (weak), §190 s1, §191 B (0.938), §192 s3 (never entered basin, clip 1.0), WINDOW s2, §187 s2.
* §198 CHEAP-THINK (e69bc32a; user request, ONE item, $10 hard cap): (a) b280 s198_injc_s1 running since 17:34 ($0.179/h, 0.861 s/step, ~$2.2 proj); (b) b281 s198_memhi_s1 (--mem_layer 9, gate_memlayer) and (c) b282 s198_rsaux_s1 (--arm readslots, gate_readslots) on ONE launcher (launch_when_slot s198bc 281 282) - recheck the family projection with b281's real $/h before b282 fires (kill the launcher if > $10); b281 / b282 end with the 198-2a eval_timing line; (a) needs a separate eval-only eval_timing198 run on a 3090 after it lands (~06:00 10-14). Report realised $ per cell at landing.
* Local 4060: pilot196_s3_stage2_4060 (anneal 4:4:12000,6:6:48000, two legs, ~00:30-01:00; KEEP every ckpt until Theory reads 196-2). New nested170 code today (all gated): grokfast, --anneal (+ entry-gated --anneal_gate), --probe_loss_every, --cache_lower, --mem_layer, --arm readslots.
* 168-12 b267 (s168s2_stepauxrw_s1) CANCELLED 2026-10-13 08:4x by user instruction via Audit (answer_ce 3.5e-6 at 11k: memorised like stepaux s1); aborted unread, partial log 04d5b020; its eval-only plan is dropped. Fleet 4 (b271 §191 A, b276 §192, b277 §193, b278 §194); slot free for Theory.
* WINDOW path GATED 2026-10-13 11:4x: gate_gnlog all PASS (g1b WINDOW flags-off incl. eval, g4 WINDOW resume bitwise incl. kmatch_counts); WINDOW cells may launch / resume on the current nested170.
* Fleet (5, 23:25 10-12): §191 A b271 (s191_stepaux_relabel_s1, STARTED 22:20, 6.9 s/step, ~21:30 10-13) + B b272 (s191_textcot_relabel_s1, STARTED 23:17, ~04:00-05:00) both verified on host (relabel_n = steps x 64, mem on the worst relabel); §190 s1 resumed b275 (~00:00-01:00), §188 b269 (~01:00), 168-12 b267 (~13:00 10-13). LANDED: WINDOW s2 b268 22:11 (ff2a3aab, pooled K'32 0.996; local package 53168b08), §187 s2 b273 23:08 (06a6f7df; rows 537ccce2).
* Fleet (5, 20:05 10-12): b267 stepauxrw, b268 WINDOW s2 (~22:00), b269 §188, b273 §187 s2, b275 §190 s1 RESUMED (b274's host rebooted 19:25 at ~27.2k; ckpt 27000 fetched read-only, b274 destroyed, b275 = same line + --resume, STARTED 19:53 on a 3090 / 570.172; post-resume 27.1k loss matches; non-bitwise after 27k: Theory note 190-1). §191 launcher (scratch/launch_when_slot.sh s191 271 272) takes the next free slots: A b271 then B b272.
* Fleet (5, 11:00 10-12): §187 s2 b273 (s187_nopass_s2, STARTED 10:59 after 4 bare CREATE FAILEDs -> vdeploy logs the create reply, d8647a9d; ~01:00-02:00 10-13). §187 s1 b266 LANDED 09:26 (c0aea65f; Theory: TOKEN-WAS-THE-LIMIT, DEEP; STABLE to K'32 0.942). ETAs 07:25: §189 rms s2 b270 (31.3k/48k, ~11:30), §182 WINDOW s2 b268 (21.1k/48k, ~22:00), §188 b269 (13.1k/48k, ~01:00 10-13), 168-12 stepauxrw b267 (3.9k/12k at 12.96 s/step vs b263's 7.07: weak-CPU host, $0.195/h -> ~$8.4 total, ABOVE the ~$5-7 Audit quoted the user for 168-12 (Audit telling the user; < $10, no stop); left in place: a move loses < 1k steps (ckpt every 1k) and would land ~13 h sooner / ~$1 cheaper, but 168-12 trains on the fixed set so the earlier landing buys little; lands ~13:00 10-13, then eval-only on the 4060). gen8 drain watchers on all 5.
  - §191 PROSQA-RELABEL: prosqa168s2 --relabel / --test500_every committed 097d69ee, gate191 all 8 PASS (g8: 3090 proj 18.2 GiB), Audit-approved; premise order in p8 already random (no shuffled-T500 arm; Theory / Audit agree).
    USER APPROVED 2026-10-12 16:40 EDT ("approve", typed in Audit's session; relayed by Audit): pair b271 (A stepaux+relabel) / b272 (B textcot+relabel), ~$8-12. Launch after Theory's §191 registration lands; next free slots (fleet <= 5; b268 frees ~22:00); B first (short). Queues b271 (A stepaux) / b272 (B textcot) drafted, NOT LAUNCHED: user approval pending.
* Fleet (5, 20:08 10-11): §168s2 stepaux b263, §185 rms b262, 168-12 stepauxrw b267 (user-approved 16:05), §187 b266, §182 WINDOW s2 b268. Landed 10-11: §184 s1/s2, §182 HOLD/WINDOW, §185 den, §186, ProsQA baselines. CPU chain: 187-4 selfdepth -> 187-5 posoffset (s184 s1, NOFADE) -> s184 s2 rows n9-12 + posoffset. 4060 after preview: s184 s2 normcap + f* 2.0. BLIND (note 184-1) f* = 2.0 FIXED (Theory, before any s2 read): on s184_reweight_s2 landing queue its train-C normcap then halt173 --normcap --normcap_c <s2>/eval_s185_normcap.json --normcap_factor 2.0 --out <s2>/eval_normcap_fstar.json; s2 numbers may be read now.
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
* 04:16 10-12: 2070 = proto187 preview (nested170 rw+nopass); 3070 = 187 package gap-fills (keep0 / readclamp_padfree on WINDOW s1, DEN s1; RMS s1 readclamp as a clamp-code sanity check); 4060 = IDLE (gap-fills done 04:14; 168-17 baseline train-item evals done 04:39, e9400362: TEXT-COT train 1.000 / test 0.735, CURR-DIRECT 1.000 / 0.508; next: b267's eval-only ~13:00 10-13). 3070 idle since 04:32 (gap-fills done). WINDOW s1 halting package (auxdone / tshift / normcap / selfdepth / sizeprobe) done on the 3070 (§188's comparator).
* (10-09) 4060: proto177_kmatch_s1b (masked, 7.5 s/step, ~17 h); 3070: proto176_rk_s1_3070 (re-run on current code); 2070: proto175_auxk8 (fp32).
* Queued post hoc: §165 K8 / K16 halt + 173-1; §169 K8 s176_ref + halt + 173-1; §170 K8 halt + 173-1.
* Voided today (aborted unread): proto176_rk_s1_4060, proto177_kmatch_s1_4060 (.void / .void2).
* 2026-10-09 10:25 (Audit): proto175_auxk8_s1_2070, proto176_rk_s1_4060, proto176_prog_s1_3070 (and the later proto177_{tidx, fadetidx, kmatch}_s1) are DESCRIPTIVE
  8k local previews read by NO rule (Theory notes 175-1 / 176-2); any §175 / §176 / §177 rule change after a preview read may only tighten. Each dir's args.json
  records train_gpu: cross-arm preview differences are card-confounded (2070 fp32 / 3070 / 4060 bf16). Queue order per card: §175 / §176 preview, then §177's.

## Waiting to be scored (Theory scores; Experiments reports when a section's runs are all in)

* **Theory (18:31 2026-10-11; session "claude-c6"; peers by name: "Audit" / "Experiments" / "Literature"):**
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
      - #1 AUX-REWEIGHT = §184, **closed: HELPS; DEEP + THINK-NECESSARY + SERIAL-NEST at 2 / 2 seeds** (feabf664). §170's C1 conjunction is met on nested arithmetic.
        - s2 pooled 0.9997. SERIAL-NEST 0.85 / 0.94.
        - Big seed spread beyond training (n 9–12: 0.486 / 0.665). C2 / C6 are not claimed. 184-1's capped s2 read is on the 4060 about 23:00.
        - Depth limit: 187-3 COUNT-OOD (size-keyed; s1 / NOFADE). 187-2 STRUCTURAL, so ADD-DECAY is dropped.
          - 187-4 BLOCKED-BY-TOKEN-EDGE (373c6d69). Pending: 187-5 and the s2 re-runs.
          - **187-6 (e2aa8722): COUNT-OOD replicates on §184 s2; RMS s1 is NOT count-OOD** (t 3–6 handle ≈ 0.95–0.99 at n 9–12; fixed onset t ≈ 7). The normalised read removes size-keying. §189 s2 expectation pre-stated. 187-6a (Audit): the t ≈ 7 onset is pass token × remaining work, not undertraining, so the lead order is RMS × NOPASS first (unregistered); discriminating prediction pre-stated. Literature asked for priors.
          - **§187 s1: TOKEN-WAS-THE-LIMIT** (depth 0.623 vs 0.486, DEEP, no aux cost); descriptively STABLE to 4× K (0.998 → 0.942 at K′32). s2 opens (about $3).
          - **§189 closed: SUBSTITUTE.** **§190 s1: mixed** (LATE 0.878, DEPTH 0.659 vs 0.587). Descriptively size-free early, no token edge, AUX-DONE tracks n to ~10 (190-2); the remaining limit is content-level size OOD.
          - **Queued:** §192 RMS seed 3 (b276, running); §193 RMS × WINDOW (gating, §188's slot ~03:30); §194 §190-s2 descriptive replication (next slot). **§187 failed at 2 / 2** (basin-entry confound; flag pinned in theory/basin_entry.py).
          - **§168 s2 STEP-AUX: not DEEP-LATENT, GENUINE memorisation** (train 1.0, test 0.49 at every K′). Fixed ~12k-item ProsQA vs nested's fresh items. Lead (unregistered): ProsQA-FRESH (procedural graphs). b267 is read as registered.
          - **Conditional lead, not registered (Audit 04:5x): GSM8K template resampling.** Per-draw renaming and number resampling of GSM8K's own 7.5k items as a training augmentation. **The source must be TemplateGSM 2411.18104** (from the 7,473 TRAIN problems, CC-BY-4.0) or our own templating of GSM8K-Aug train items, with the equation CoT recomputed. **Never GSM-Symbolic 2410.05229**: its templates come from the GSM8K TEST set and are CC-BY-NC-ND, so they are eval-only (Literature 2d3828d8). The trigger is §191 A reading NAME-MEM (≥ 0.65). The shape is a GSM8K-short pilot of fixed vs resampled templates on TEXT-COT and the current latent base (RMS, or RMS + NOPASS, not §184's reweight), with GSM8K-Aug (385k) as the reference. GSM8K-scale spend needs the user's approval.
          - **§198 CHEAP-THINK registered (e69bc32a; the user's request; Audit: one item, $10 cap):** an all-cached nested family. (a) cached INJECT (−44%), (b) MEM-HIGH l_m 9 (−67%), (c) READ-SLOTS + RMS + step aux (−87% at L 300). Experiments is building (b)'s --mem_layer and porting (c). The cost clause is at L = 300. Ruled out: MELT / LLA / RLT / T²MLR (cross-loop KV).
          - **Theory 16:2x (after the 13:45 MACHINE REBOOT; this session is now named claude-5a):**
            - Local §201 T4 runs died: rms s1 (2070), §184 s2 (4060) and s193 (3070, note 201-8). s190 s1 and rms s2 are done (6a4c5e39). Re-queue, $0.
            - Cloud §200 A / B / D and §193 s2 need their drain watchers checked.
            - §199-A is CLOSED (1 of 3 seeds; CSR FLOOR-GAMED, harmful).
            - Ready: the WARM-START-FROM-COT conditional draft (scratchpad warmcot/WARMCOT_DRAFT.md; registers only if A / B read LOOP-NOT-FORMED or below; user-pre-approved ≤ $4 for the pair; latent cost 1.48 s/step at micro 4, not 0.89).
          - **Theory 01:51 (10-15):** USER APPROVED (~01:4x via Audit) -> REGISTERED §205 IIT-SERIAL (FULL, 3 seeds, cap $18; launch now; scorer score_s205 + eval_iit.json contract, note 205-1) and §206 JOINT-COT-L1 pair (cap $6; harness + gates + Audit diff review first; score_s206 after the field names). §202 scored NEUTRAL (faster, same ceiling; CoT trunk forgotten) -> §204 F-P on the A base running (b295, lands ~12:30; score_s204 ready). Seed-stab findings: 203-4 CONTRAST-SURVIVES, 203-8 failed seeds follow the input (Huginn-like), 203-7 SEEDSELECT INCONCLUSIVE / explained by curve mechanics, 203-9b DAS null = fitter (COMPACT-FIT-LIMITED), 167-x MULT read fit-limited.
          - **Theory 21:01:** Seed stability: whole-state swap is the USED separator - note 203-4 CONTRAST-SURVIVES (clean-ccc n 3-8: converged hybrid >= 0.80 vs failed <= 0.18; failed seeds ignore their state). IIT-SERIAL (whole-state interchange loss) G-ALIGN passed; --iit code next (Experiments), Audit diff/gradflow review, G-TIME <= $0.25, then to the user (~$9.6 / $13.5; headline power; deep-n push on the good class; n<=7 variant cutoff chosen post-pilot). DAS-USED pilot P0b/P1 running (3070). SEEDSELECT blind draft (e2f01350) awaits Audit before --freeze. §204 scorer + notes 204-5..7 ready; waits for W (~22:40) -> score_s202.
          - **Theory 20:16:** §203 SVR + §204 STEPFACTOR REGISTERED (f6f8d51d, user-approved). §203 result: G-use DO NOT LAUNCH, D7 separator NOT-USED 9/9 incl. ALL heads (2b5b09e1, a9182a47); $0 spent. §204 harness building (Experiments; notes 204-1..4: Literature D1, SIM-CoT check MIXED (MEAN supports), ptr collisions / string equality, gate 9 fp32). Opus drafts in progress, not registered: DAS-USED (find the USED value representation; $0), SEEDSELECT (blind early-selection pre-registration; $0 retrospective), LOTUS contrast (capacity vs teacherless supervision; feeds a §204 pre-output note).
          - **Theory 18:31:** Scored: §200 GSM-Symbolic P8 FALSIFIED (b42efea1), §201 T4 SNAP-HURTS 4/4 (be650099; the text write fails beyond range). Note 202-1 THINK-NECESSARY-DEEP (b0d6e0d9; A/B reference 0038e04f: deep K′1/K′10 0.88 / 0.70). Drafts NOT registered, both to the user via Audit: STEP-TARGET-FACTOR (6e28fc01, after §202, ~$3-3.9; D5b RMS question in progress) and SEEDSTAB2 / SVR (0c005392, ~$8.5 / $10.1; G-pre $0 gate contracted to Experiments). §202 W / C+ filling; score_s202 at landing.
          - **Theory 16:53:** §200 read BELOW → §202 WARM-START registered (30401c78), waiting on Experiments' fp32 gate + Audit's diff review; reader theory/score_s202.py. W's g_add_init 6.2e-4 vs A's 2.29e-4 (same calibration; no confound). Seed-stability successor to CSR: Opus-subagent derivation in progress (scratchpad seedstab2/; candidates SHALLOW STEP READER / decodable-progress floor / two-sided band, with a Goodhart analysis); not registered, needs the user's OK via Audit.
          - **Theory status 2026-10-14 08:1x (supersedes the §199 / general lines below where they differ):**
            - **§199 (seed stability):**
              - M1 PREDICTED (α^L tracks the size-keyed depth limit, not the lottery).
              - M2 LMC: no pair is connected.
              - M3 / M3a PREDICTED: the failed models make right-structure, wrong-value errors.
              - E1: converged models gate β with progress; failed models offload to the integrator.
              - T1: path_indep co-moves with the loss, so it is not an 8k detector.
              - **CSR (199-A, ~$8.4 cap $10):** s1 b283 is running, PAIRED (A4 pass). s3 is waiting for a 595.84 host (Audit: one try on the bad-listed host, else wait). s2 is waiting for s1's 570.172.08 host (~16:00). Reader: score_s199_a.
            - **§200 GSM8K-Aug pilot** (user-approved, cap $7):
              - C TEXT-COT landed 0.321 (budget-limited; note 200-4; published CoT is 15–40 epochs vs our 4);
              - A / B / D running;
              - LOOP-NOT-FORMED rule (200-1); clarifications 200-2 / 200-3.
            - **§201 overthinking (the user's request) RESULT:** the additive account is FALSIFIED. Overthinking = post-completion DELTA writes (freezing them recovers ~all). The learned halt gives NO-OVERTHINK on 7/7. Next ($0): the FD-on-signal soft halt (descriptive), the ProsQA T1, the window control.
            - **§193 RMS × WINDOW s1:** JOINT + LONG-STABLE (K′24 0.989, DEPTH 0.688). s2 (~$7–8) is with the USER via Audit.
            - **§198:** (b) CHEAPER-WORSE (confounded); (c) FAILS. Family ~$6.29.
            - **§191:** NAME-MEM.
            - **§196:** NO-EFFECT. §197 is not registered as drafted (needs fresh seeds plus a derivation).
            - **Rules added:** math backing for every change (user); a seed-rate OC block and conservative censoring (to the user via Audit); match the original --bs in bf16 reproductions (201-5).
          - **§199 CONTRACTION-TRAP (the user's seed-stability request, 2026-10-13 20:5x via Audit; Theory, with Opus subagents per the user):**
            - Opened with M1 only (ac074515 / 0531d0b9; scorer theory/score_s199_m1.py). The order parameter is s182.path_indep: converged models have delta_cos 0.44–0.63, failed models 0.69–0.98, across 19 models (Audit-reproduced). The failed models compute contractively.
            - Derivations are in the scratchpad (seedstab/MATH_contraction.md, MATH_selection.md, LIT_contraction.md).
              - A κ-contraction resolution bound.
              - Init is deep in the contractive regime (delta α^L: 0.91 at n 4, 0.77 at n 12).
              - The open-loop phase (γ = 0 to 2k) is the bad set.
              - Selection P = Φ(μ_Z/σ_Z): only tilt levers work, √-levers need 15–42×.
              - The pass token removes the aux penalty on contraction.
            - **Amendment 199-A registered (0420e8b6; reader theory/score_s199_a.py; derivations docs/THEORY_SEEDSTAB_2026-10-13.md):** method **CSR** (completion-switched step-ratio target; renamed from PCLT per Literature 67329f72, whose verdict is that the sign switch is NEW). Cells s199_csr_s1/s2/s3 = s185 RMS args + --csr, 3090, uncached, paired with the s185 controls by a bitwise check on steps 100–500 (s1 / s2 driver 570.172.08, s3 595.84). **Audit APPROVED as ONE item, ~$7, cap $10** (no second stage, NOPASS-s3 or extra seed under it). The contract went to Experiments 22:1x (code + gates g1–g6). Pending: the CPU gate-spectrum split (M1 inputs), Experiments' checkpoint path_indep trajectory.
            - §198 note 198-3 / 198-3a: READ-SLOTS predicted less contractive and less size-keyed (descriptive).
          - **General latent-training improvements (the user's request, 2026-10-13 ~22:0x):** three Opus subagents (G1 methods, G2 optimisation / stability, G3 methodology), outputs in scratchpad/genlat/. Literature verifies them before any docs page.
          - **Basin-entry levers (the seed lottery; §192: RMS s3 never entered):** §196 GROKFAST-S3 is registered (78487d7a, running). If it reads NO-EFFECT, next in line: (i) **BUDGET-ANNEAL** on s3 (Zhang 2607.20594's fix for its A5 seed lottery, 4 / 4: train small K / n_max, then warm-start into larger. Its precondition holds, since s3 fails at composition from t 3); (ii) weight decay (wd 0.01 vs about 1.0 in grokking work; riskier for a pretrained trunk). §197 EASY-FIRST was not registered (its precondition failed: s3 learns the operator, t 1–2 ≥ 0.998).
          - **§191 PROSQA-RELABEL registered (ab7d85cb; user-approved 2026-10-12 16:40 via Audit; about $8–12):** A (stepaux + relabel) launches at the next free slot, then B (TEXT-COT). Reader `theory/score_s191.py`.
          - **187-7 SIZE-PROBE: PREMISE True, account WEAKENED (09b7021d).** RMS's read decodes n at 0.895 by direction, as §184's does, yet RMS is size-free. **187-8 closed: PARTIAL** (pad-free re-run: n 12 t 3–6 handle +0.08 / +0.13, control intact). Rescaling to in-distribution norms makes s1's early steps nearly size-free but costs answers (DEPTH −0.07 / −0.14). The lever is training with a normalised read (§189).
          - **187-5 closed: CONTROL-BROKEN** (KEEP-SINK offsets collapse n 3–5 in both models; the memory is position-entangled). With 187-6, the position remedy is deprioritised.
          - **184-1 HOLDS (d6d589ee):** STABLE-with-cap at 2 / 2 (s1 fitting, s2 held-out); near-neutral on s2 within n ≤ 8, costs n 9–12 at K′16 (0.459 vs 0.619).
          - **§185 closed (cbdba1ef): NO-FIX both arms.** Descriptive: RMS alone DEEP (min 0.994) + depth n 9–12 0.587 (§184 s1 0.486), one seed. **§189 RMS-REP (852206c8)**: seed 2, s185_rms_s2, free slot (~$3); SUBSTITUTE / DEEP-ONLY / NOT REPLICATED vs §184 s2; reader `theory/score_s189.py`.
          - **§188 REWEIGHT × WINDOW registered (5b2ebd2e; Audit cost-approved, ~$4–7):** WINDOW + --aux_reweight; JOINT (K′16), a C6 claim needs LONG-STABLE (K′24) at 2 / 2. Needs Experiments' kmatch_aux_loss change (the window path ignores --aux_reweight today). Launching now (188-3, Audit OK); WINDOW s2 read beside, not a gate.
          - **173-2 / 173-3 closed: PREMATURE** (answer2 and MOVED2, held-out s184 s2 + WINDOW): stable wrong pre-completion answers; halting (C5) parked behind COUNT-OOD.
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
* **Mathematical backing (user, 2026-10-13: "any revised architectural/methodological changes should come with
  mathematical backing").** Every new or revised architecture, training-objective, schedule or methodology change
  carries a derivation in its registration:
  - the assumptions;
  - the mechanism linking the change to the targeted failure;
  - each claim tagged PROVED / DERIVED-UNDER-ASSUMPTIONS / CONJECTURE / DATA (post-hoc data marked);
  - falsifiable predictions derived from the math.
  Long derivations go in `docs/THEORY_*.md` with CPU check scripts in `theory/`. The user wants proofs and research
  delegated to Opus subagents, with their key proofs reviewed before use. Audit checks for the section before
  approving spend. Bug fixes and logging-only changes are exempt.
* **Seed-rate OC block and conservative censoring (user, 2026-10-14 ~21:1x, typed in Audit's session and relayed verbatim; recorded 21:12 EDT: "approve the 2 that u reccomend").** Every new registration, from IIT-SERIAL onward, carries:
  - (a) a seed-rate OC block: P(pass rule | no effect) and P(pass | effect) at the panel's seed count, plus realised $;
  - (b) conservative censoring: a run that never reaches the event by its end counts as a FAILURE, not missing, with the event-only mean reported beside it.
  The same approval SKIPS §200 run D (text CoT with resampled numbers): §200's R1 B-D and R3 text pair stay UNAVAILABLE.
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
