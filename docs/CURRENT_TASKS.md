# Current tasks — live handoff (keep this file current)

**Purpose.** If a Claude session crashes or a new chat takes over, this file says what is running, where, how to check
or stop it, what is waiting to be scored, and the context needed to decide what comes next. Update it whenever a job
starts or finishes. Last updated: **2026-09-30 17:45 EDT** (local time = UTC − 4).

Read first in a new session: this file → `docs/PROTOCOL6.md` (pre-registrations A–H, amendments) →
`docs/PROTOCOL6_RESULTS.md` (scored so far) → the tail of `docs/RESEARCH_LOG.md` (chronology) →
`docs/LITERATURE_NOTES.md` (+ the older, broader `docs/LITERATURE.md`).

> **Folder reorganisation (2026-09-28 18:45):** the user is about to have another agent tidy this folder. Live jobs use
> `latent/`, `scratch/`, `runs/`, `results/`, `logs/`, `prosqa/` and the four live docs by path — see
> `DO_NOT_MOVE_WHILE_RUNNING.md` at the repo root. Backup (code, docs, results, logs, run configs/logs; no weights):
> `D:\Latent_backup_2026-09-28_1840.tgz`. If any path here stops resolving, check whether it was moved before
> assuming a job failed.

> **Theory stream (separate Claude session, from 2026-09-29 ~01:10, at the user's request).** Researches *why* some
> models fail to extrapolate and designs new architectures. It works only in `theory/` (code + `theory/results/`) and
> `docs/THEORY.md`, runs on the **CPU** (never the GPUs), reads existing `runs/` and `results/` read-only, and does not
> touch queues, `latent/`, `scratch/`, pre-registrations or cloud instances. Findings there are exploratory until tested.
> **Theory-stream Vast instances (user-approved 2026-09-29 ~01:50: "3–4 GPUs"; labels `theory-pa-h*`; do not destroy —
> the theory session fetches and destroys them):** 53312599 A4000 `ssh -p 40473 root@38.29.145.230` ($0.121/h),
> 53312701 RTX 4070 `ssh -p 1712 root@174.0.242.177` ($0.149), 53312759 RTX 3080 `ssh -p 40283 root@181.116.79.48`
> ($0.132), 53312847 RTX 3090 `ssh -p 42213 root@109.173.238.203` ($0.178) — ≈ $0.58/h. Pilot A (architecture
> variants, pre-registered in `docs/THEORY.md` §6): `/workspace/Latent/theory/runs/pa_*`, log
> `/workspace/Latent/logs/queue_theory-pa-h<N>.out`; fetch with `bash theory/vfetch_arch.sh <host> <port>`. Queues hold Pilots A–E (`theory/queues/`; A scored, B nearly done, C running, then D/E = 28 same-seed anytime/control pairs, the depth-wall vs schedule-end test pre-registered in `docs/THEORY.md` §20; D/E now expected to end ~01:00–05:00 EDT 2026-09-30), then Pilot H (THEORY §27: fine-tune repair of 20 P30 failures, 48 short runs) on all hosts; hosts 2–4 continue with Pilot F (§23: jitter, jitter + anytime; 16 runs after amendment F-1), G (§24: set memory, 16), I (§28: periodic clock PER2, 8) and J (§29: chunk-reset attention, 16). Host 1 is fetched and destroyed after its D/E + H runs (~06:00–08:00 Wed); hosts 2–4 run dry ~16:00–19:00 EDT Wed. **Update 15:44: credit topped up ($145.83); six more theory hosts theory-x1..x6 (instances 53408503/08/12/18/22/26, list in `theory/hosts_theory.txt`, create/destroy log `theory/vast_log.txt`) run F, G, I, J (withdrawn arms restored, amendments F-2/I-2) and 3 D/E pairs moved from host 1; hosts 2–4 keep their D/E + H. All theory hosts expected to drain by ~03:00 Wed and are destroyed as each finishes; the experiments session takes cards from ~07:00 Wed. Theory spend from 15:30 ≈ $19.** **Update 19:29:** theory-pa-h1 fetched and destroyed automatically at 18:43; theory-x2 died (SSH refused) and was replaced by theory-x2b at 16:35; runs take ~55 min, so four more hosts theory-y1..y4 (19:18) share the unstarted runs — 13 theory hosts, $1.98/h, drain ~03:00–03:30, theory spend from 15:30 ≈ $20–22. Current host list: `theory/hosts_theory.txt`; watcher `theory/watch_hosts2.sh` (auto verified fetch + confirmed destroy); hourly fetch + measurements `theory/incremental.sh`. Host 4 also runs the GPT-2 latent-reasoner probe (`/workspace/codi_batch.sh`, log `/workspace/codi_batch.log`). Theory CPU jobs locally: keep ≤ 3–4 threads at Idle priority (they starved the local GPUs once, 2026-09-29 ~07:00).

> **Theory ↔ experiments coordination (committed 2026-09-29 11:47 EDT; the "~12:10" first written here was an estimate).** Messages exchanged between the experiments session and the
> theory session (claude-54). Shared understanding: the horizon is a propagation limit, the readout generalises. Theory
> additions (docs/THEORY.md, executive summary at top): the latent thought is a pointer that breaks at step K + 1;
> failing seeds retrieve it from earlier latent positions on a step-indexed schedule (masking that from step K on rescues
> 17/23, zero in-distribution effect); the circuit is chosen in curriculum stage 3; the cause is slack (budget > hops
> needed). **Proposed joint test:** anytime (minimum-progress) supervision at K = 8 and on cap-3 graphs — a depth wall
> predicts the front still stops at the training depth, a schedule end predicts it continues (K = 6 anytime already
> gives fronts to depth 11 in 2/4 runs; Pilot D 12 + 12 running on Vast). Ownership (theory Vast hosts vs a PROTOCOL8)
> to be agreed. The theory session analyses O/R/T9/T10 runs only after the experiments session has scored them.

## 0. STATE AT 2026-09-30 00:36 — authoritative; sections 1–3 below are partly history

**Budget (12:45): SPEND FREEZE until the user tops up.** Vast $76.17. The rest of PROTOCOL8 is ≈ $52 and PROTOCOL8b
step 1 ≈ $4, leaving ≈ $20 (below Audit's $30 alarm). No Vast creates beyond the queued PROTOCOL8 and step-1 seeds
(ctl_s3 is the only one left to place). Re-check the projection once all hosts are placed. Every create/destroy goes
in `scratch/pods.txt`.

**Theory (details in docs/THEORY.md, one-liners at the top of its executive summary):** all overnight pilots are
scored (D–M, §§34–38). No theory hosts are live. T12 (§39, 8 runs) runs on the experiments session's Vast hosts.

**This session's live Vast hosts** (full lifecycle in `scratch/pods.txt`; destroy only via the drain scripts):

| Group | Hosts (instance: host:port) | Done → drained |
|---|---|---|
| done 09:34 | Phase 0 Coconut | **F1 and F1b not met** (aᴄ(H, 10) = 0.656; K′ = 0 at chance): outcome (iii), Coconut does not learn deep problems; `docs/PROTOCOL8_PHASE0_RESULTS.md` |
| **PROTOCOL8** (registered 482a8e9; `prosqa/pod_p8c.sh`) | coco_s1 53533281 151.237.25.16:20413 · coco_s2 53533300 213.163.75.246:50708 · at_s1 53533289 142.112.39.215:63460 · rt_s1 53533295 86.123.234.69:46460 · rt_s2 53533311 185.101.222.13:17298 · cot_s1–s4 53533314/19/22/26 (38.29.145.230:40495, 142.127.68.223:13228, 159.48.242.15:42038, 38.247.78.6:44010) · at_s2 + s3/s4 of every latent arm placed by `scratch/p8c_fill.sh` (log `logs/p8c_fill.log`) | latent ~Thu 04:00–10:00; cot ~16:00 → `drain_prosqa.sh` |
| **PROTOCOL8b step 1** (registered 7d61c0a; `prosqa/pod_p8b1.sh`) | ckpt_s1–s3, ctl_s1, ctl_s3 drained (15:11–16:44). Live: ctl_s2 53555209 5.2.174.20:23725 (last epoch at 16:48, ~18:00 incl. eval) | then rename `prosqa/res_p8b1/p8b1_p8b1_<arm>_s<seed>.json` → `p8b1_<arm>_s<seed>.json`, `python prosqa/score_p8b1.py` |

**Watchers (re-armed 16:46 over the 16 live hosts: 15 p8c + p8b1 ctl_s2):** `scratch/auto_drain_prosqa.sh` (verified
drain on ANALYSIS_DONE; log `logs/auto_drain.log`) and `scratch/gpu_watch.sh` (5-min idle check, user request). Both
exit on their first event (error, failed drain, idle host), so re-arm each with the current host list straight after
any event, create or destroy. The toy-side `auto_drain.sh`, `p8_watch.sh` and `p7s_monitor.sh` have nothing left to watch.

- **Phase 0:** pre-registered in `docs/PROTOCOL8_PHASE0.md` (3a9c05e, 15:09:38). Pod log `/workspace/logs/prosqa.out`,
  results `/workspace/prosqa/res/`. Watcher `bash scratch/p8_watch.sh "142.112.39.215 63335 coconut" "38.29.145.27
  40817 nocot"`. On ANALYSIS_DONE, fetch `res/` and the final checkpoint (`prosqa/vfetch.sh`, checksums), destroy,
  then score by the Phase 0 rules.
- **Phase 0 note (19:20):** Coconut's in-run validation (exact match, generation capped at 128 tokens by `run.py`) reads
  0/100 through the early stages. The deep problems' language CoT runs past 128 tokens before the answer, so it
  climbs only as latents replace steps (7/100 at epoch 15). It isn't a Phase 0 measure: the pre-registered measures
  are candidate log-prob comparisons (`long_eval2.py`, `eval_p8.py`, max 220 new tokens for CoT). The Coconut pod runs
  ~14 min per epoch, so training ends ~02:40 and evaluation ~05:30–06:00.
- **T9 on Vast:** PROTOCOL7 amendment 8 (9bb1944, 15:39). Toy code lives in `/workspace/Latent`, log
  `logs/queue_t9_v?.out` (host clock is UTC). Deploy script `scratch/vdeploy_toy.sh`. When a queue shows QUEUE_DONE,
  run `bash scratch/vfetch_toy.sh <host> <port> t9` and destroy.
- **Vast CLI:** the stable wrapper is `C:/Users/peter/.local/vastai/vast.sh` (package in `.../vastai/lib`; key in
  `~/.config/vastai`, never copied). Shared with the theory session. The old copy in the session scratchpad is
  deprecated.
- **Auto-drain:** `scratch/auto_drain.sh "<host> <port> <tag> <id> <t9|p9>" …` runs `drain_toy.sh` on each QUEUE_DONE
  (log `logs/auto_drain.log`; drained hosts listed in `logs/drained_hosts.txt`, which `gpu_watch.sh` skips). It wakes the
  session on a failed drain, a new failure line, unreachable, or stalled.
- **Drain finished hosts with the drain scripts, never a bare fetch plus destroy:**
  - `bash scratch/drain_toy.sh <host> <port> <t9|p9> <id> <queue tag>` for toy queues;
  - `bash scratch/drain_prosqa.sh <host> <port> <id> <local res dir> <remote ckpt> <local ckpt> <expected res files…>` for
    ProsQA.
  They fetch with checksums (3 tries), check that every evaluated run's files are present locally, and only then call
  vdestroy.sh and append to pods.txt. An incomplete fetch prints FETCH_INCOMPLETE and destroys nothing (Audit lesson:
  theory-L01).
- **Destroy with `bash scratch/vdestroy.sh <id>`.** It passes -y (a bare `vastai destroy` stops at a [y/N] prompt
  and destroys nothing) and checks `show instances` until the id is gone. Log it in pods.txt only after
  DESTROYED_VERIFIED. An unreadable instance list gives UNKNOWN (exit 2), never "verified".
- The theory session owns `theory-pa-h1..h4` and `theory-x1..x6` (its log `theory/vast_log.txt`); never touch them.

**Local GPUs (detached queues; torch `cuda:0` = 4060, `cuda:1` = 3070, `cuda:2` = 2070):**

| GPU | Now | Next (chained, automatic) | Runs dry |
|---|---|---|---|
| RTX 4060 | lent to Theory 15:26: §45 CLR-1b (log `logs/queue_local4060-CLRb.out`) | — | ~18:55 (Theory update 17:45) |
| RTX 3070 | lent to Theory 15:26: §45 CLR-1b (log `logs/queue_local3070-CLRb.out`) | — | ~18:55 (Theory update 17:45) |
| RTX 2070 | base ref done 17:36 (`prosqa/res_p8b/p8b1_base_ref.json`). 17:44: exploratory PROTOCOL8b hop sweep (`scratch/p8b_hopsweep.sh`, `prosqa/hopsweep_p8b.py --skip_probe`: probe-disjoint problems, no accuracies printed, not read until P8b1 is scored; log `logs/p8b_hopsweep.log`) | ctl_s2 waits for its verified drain | ~19:00 |

- **Overnight follow-up: PROTOCOL9** (`docs/PROTOCOL9.md`, a28ff0c 16:09:10): anytime supervision on the T9/T10
  architectures, paired by seed with the T9/T10 controls. 32 runs (`scratch/queue_worker3.sh`, frozen theory code in
  `scratch/p9_frozen/`), chained after T10 on every card by `scratch/chain_after_w.sh`. Score with
  `PYTHONPATH=. python scratch/score_p9.py` once all pairs are in.
- **GPU idle watcher:** `scratch/gpu_watch.sh`, polling every 5 min (user request); re-arm it with the current host list
  after any create or destroy.
- **Monitor:** `bash scratch/p7s_monitor.sh` (a queue failure, QUEUE_DONE, or a log.jsonl idle > 75 min). The 3070
  runs thermally throttled at 80–86 °C. Keep jobs under ~6.5 GB per 8 GB card.
- **Pilot G feasibility:** done 15:54 (priority slot via `scratch/priority_run.sh`); both runs failed the gate, and
  Theory has amended Pilot G (G-1).

**Waiting to be scored / timeline (Wed 30 Sep; updated 07:45):**

| When | Item | Action |
|---|---|---|
| done 07:43 | T9 + T10 | **T9 pooled supported (AUC 0.89); T10 supported (AUC 1.00)**; per family in `docs/PROTOCOL7_RESULTS.md`; secondary causal tests pending |
| done 07:45 | PROTOCOL9 primary | **AP1 not supported (d8 5/10); AP2 supported (10 vs 3, p 0.046); AP3 anytime more learnable (21 vs 12)**; `docs/PROTOCOL9_RESULTS.md` |
| ~09:00 | Phase 0 Coconut: training ends ~08:10 (epochs ran ~28 min, not 14); `coco_test_e48.json` first, full evaluation ~11:00 | read F1/F1b from e48 as soon as it exists; drain at ANALYSIS_DONE |
| ~09:30–10:00 | PROTOCOL8 registration, after F1/F1b | launch ~10:30: 12 latent-arm hosts (~16–20 h each) + 4 CoT; ≈ $50 of $88 |
| done 13:31 | PROTOCOL9 pooled 64 | **AP1/AP2 inconclusive (AP2 18 vs 10, p 0.09); the primary AP2 does not replicate**; `docs/PROTOCOL9_RESULTS.md` |
| — | T12 (THEORY §39) | fetched 06:11; scored by Theory |

**Scored 2026-09-29 (details in PROTOCOL6_RESULTS.md and PROTOCOL7_RESULTS.md; all re-verified by the audit):**
- O1 not supported.
- P1 supported, P2 not.
- Q1 and Q2 not supported.
- PROTOCOL7:
  - T1b, T1c, T2b, T13 and T4 supported.
  - T3, T5 and T7 pass.
  - T1, T2, T6 and T8b not supported.
  - T11 pilot not learned.
  - T12 not triggered.

**Theory results to cite:**
- §21: the out-of-sample marker failed; T3 early P7 AUC 0.92.
- §26 plus controls (71b9830): re-anchoring at the frontier's depth takes probe 7-hop success (distance-7 bal-acc ≥ 0.60, no
  CI clause, so looser than "extrapolates") from 24 to 55/59, i.e. 31 of 34 failures rescued. Random same-depth anchor
  56/59; random-node anchor 0/59; oracle 59/59.

## 1. What is running now (HISTORY before 2026-09-29 — see section 0)

### Local PC (3 GPUs; torch `cuda:0` = RTX 4060, `cuda:1` = RTX 3070, `cuda:2` = RTX 2070 — nvidia-smi numbers differ)

| GPU | Job | Queue file (`scratch/`) | Log (`logs/`) | Expected end |
|---|---|---|---|---|
| 4060 | Addendum K (P30) | P7 as a ranking | done | **scored 03:40: supported** — AUC 0.993 (EA 0.673), 42 competent runs |
| 3070 | Addendum K (P30): `p30_s112..123` (no step emb, bf16) | `p6_p30_3070.txt` | `queue_p30_3070.out` | ~Mon 00:30 (≈55 min/run) |
| 2070 | Addendum K (P30): `dev6_s51..56` (dev6 recipe, fp32) | `p6_p30_2070.txt` | `queue_p30_2070.out` | ~Mon 01:30 (≈2 h/run) |

* **Chained next (addendum L, K = 8):** `scratch/chain_after.sh` starts `p6_k8_<gpu>.txt` on each card when its P30 log
  shows QUEUE_DONE (4060: k8_s200..205, 3070: k8_s206..211, 2070 fp32: k8_s212..215); logs `queue_k8_<gpu>.out`;
  EA loop `p6_after2.sh cuda:2 scratch/p6_k8_*.txt`. Score: P7 with the frozen far_sweep (seed 2033) on competent runs;
  rules in PROTOCOL6.md addendum L — **scored 13:20 with `scratch/score_l.py`: L1 supported (AUC 0.90), L2 supported (12/13)**. Amendment 01:45 appended k8_s216..222 (3070 ×4, 4060 ×2, 2070 ×1); their EA is NOT auto-measured (the after2 loop read the queues at start) → measure when scoring. Queues run dry ~10:00–12:00.
* **Chained after L (addendum M, depth cap):** `chain_after.sh logs/queue_k8_<gpu>.out m_<gpu> ...` starts `p6_m_<gpu>.txt` (3070: m_s300..307, 4060: m_s308..313, 2070 fp32: m_s314..316; `--depth_cap 4` = train only to distance 5). EA not auto-measured. **DONE, scored 2026-09-28 ~20:10** (`scratch/score_m.py` → `results/p_m.json`): M1 supported (7/10), M2 unresolved.
* **Idea parked (not built):** deeper training graphs (graft branch onto an attached leaf, BFS labels, parent-aux fix) — the other direction of M.
* **Addendum P (ProsQA deeper than training, evaluation only; pre-registered 2026-09-28 ~21:18):** `prosqa/run_p.sh cuda:2 g_s0 g_s1 g_s2 j_s0 j_s1` running detached on the RTX 3070 alongside O (log `logs/p_eval.out`) and `run_p.sh cuda:1 j_s1 j_s0 j_s2` (log `logs/p_eval_4060.out` — misnamed: torch cuda:2 = RTX 2070 and cuda:1 = RTX 3070, so the P runners are on the 2070 and 3070; existing outputs are skipped, so the overlap is harmless; ~2.7 h per model, done ~Tue 06:00; j_s2 needs `prosqa/ckpt/j_s2_checkpoint_50` fetched first, else re-run it, output `prosqa/res_p/<tag>_{long,ctl}.json`; data from `prosqa/make_long.py`). After J s2 is fetched: `bash prosqa/run_p.sh cuda:2 j_s2`. Scorer: `python prosqa/score_p.py` → `prosqa/res_p/p_score.json` (written before any output).
* **Addendum R (toy dose-response, pre-registered 2026-09-28 ~22:02):** `chain_after.sh logs/queue_o_<gpu>.out r_<gpu> ...` starts `scratch/p6_r_{3070,4060}.txt` (r3_s400..411 `--depth_cap 3`, m_s317..328 `--depth_cap 4`, interleaved) after each card's O queue. Cut-off: runs not started by Wed 2026-09-30 06:00 dropped. Scorer `scratch/score_r.py` (written before data; reproduces M 7/10) (cap-3 gate matched4 ≥ 0.60; P5 = e5/in4; R3 pools cap-4 runs with M — rules in PROTOCOL6.md addendum R).
* **Addendum Q (GPT-2 trained on 3–4 hop ProsQA, pre-registered 2026-09-28 ~22:02) — Vast:**

| Instance | Seed / config | SSH (direct) | $/h |
|---|---|---|---|
| 53289329 | seed 0, `prosqa/prosqa_q.yaml` (name `prosqa-coconut-q`), `pod_prosqa_q.sh` | `ssh -p 40044 root@96.20.36.104` | 0.176 |
| 53289703 | seed 1, `prosqa/prosqa_q_s1.yaml` (name `prosqa-coconut-q-s1`) | `ssh -p 40409 root@142.198.191.106` | 0.192 |

  Launched ~22:10 with `prosqa/vdeploy.sh` (RunPod had no card ≤ $0.34/h: community 4090/A6000/4000 Ada gone, secure 4090 billed $0.74 and A6000 $0.53, both deleted within a minute). Both cuInit 0, drivers 555/565, md5 of regenerated long/ctl/test sets match. Expect ~16 h training + ~4 h analyses (done ~Tue 18:00–20:00). Status `bash prosqa/status.sh vast:<host>:<port>`; fetch `bash prosqa/vfetch.sh <host> <port> prosqa/res_q_s<seed> /workspace/ckpt/<name>/checkpoint_50 prosqa/ckpt/q_s<seed>_checkpoint_50`, then destroy. Progress 2026-09-29 00:55: both at epoch 12/50 (~14 min/epoch so far). Scorer `prosqa/score_q.py` (written before data) (Q1 gap 3–4 vs 5–6 hop at K=6 from `kprime_test_e50.json`; Q2 K'=12 − K=6 on 5–6 hop).
* **PROTOCOL7 confirmation suite (pre-registered 2026-09-28 ~23:30, amendments 1–5; results `docs/PROTOCOL7_RESULTS.md`).**
  Scored: T1, T2 not supported (distance 7 is a query-time hop; T2 dose ≈ uniform); T3 supported (AUC 0.93); T5 supported;
  T6 not supported by 0.4 % (layer 1 robust, layer 2 not); T7 pass; T8a as expected; T8b fail (tiny structural component).
  **Pending, run when addendum O is complete (local + fetched Vast O runs):** `PYTHONPATH=. python scratch/score_p7s_amend3.py
  --device cuda:0` (T1b, T1c, T2b, T13; measures with `latent/frontier_causal2.py` and frozen far_sweep), then
  `PYTHONPATH=. python scratch/p7s_t4.py --device cuda:0` (T4; needs the amend3 output first).
  **Chained, start automatically:** T9 architectures `scratch/p7s_t9_{3070,4060}.txt` after each card's R queue
  (logs `logs/queue_t9_*.out`); T10 fork graphs `scratch/p7s_t10_2070.txt` after the 2070's O queue (log
  `logs/queue_t10_2070.out`). Score T9/T10 with `scratch/far_sweep_layers.py` (layer by largest in_peak) — scorer
  `scratch/score_p7s_t9.py` (written before data; rules in PROTOCOL7 T9/T10). T11 (from-scratch ProsQA) and T12 need cloud credit.
* **GitHub: private repo https://github.com/PeterLurf/latent-reasoning (branch main), first pushed 2026-09-29 ~01:22.**
  Push from D:\Latent with `git add -A && git commit -m ... && git push` (set GIT_TERMINAL_PROMPT=0). The repo's local
  git config uses only the gh credential helper (`credential.helper` reset, then `!C:/Users/peter/.local/gh/bin/gh.exe
  auth git-credential`; gh logged in as PeterLurf) — the system Git Credential Manager hangs on a hidden login window.
  Keep `core.autocrlf false` (bash scripts must stay LF). Weights and raw ProsQA logs are git-ignored.
* **DONE 2026-09-29 ~07:00 — not learned (test 0.51–0.55 at every K′; fetched to prosqa/res_s11p, Vast 53306310 destroyed; T11 stops).** Was: PROTOCOL7 T11 pilot (from-scratch 2-layer GPT-2 on ProsQA; amendment 6) — Vast 53306310, RTX 3090, $0.173/h,
  `ssh -p 6399 root@142.127.68.223`**, launched 2026-09-29 ~00:51 with `prosqa/vdeploy.sh` (RunPod A4500 at $0.19 vanished
  on create). cuInit 0, driver 560, md5 of regenerated sets match. ~3.7 min/epoch → training ~3 h, then analyses. Fetch:
  `bash prosqa/vfetch.sh 142.127.68.223 6399 prosqa/res_s11p /workspace/ckpt/prosqa-coconut-s11p/checkpoint_50 prosqa/ckpt/s11p_checkpoint_50`,
  then destroy. Continue to confirmatory seeds only if test ≥ 0.85 at K = 6 and the hop clock increases (T11 rule);
  fix the P7-analogue floor in an amendment first (needs credit: ~10 seeds × ~$0.7).
* **vast3 (174.0.242.177:1743, RTX 4070) continues after its O queue with 6 T9 seeds** (amendment 7,
  `scratch/p7s_t9_vast3.txt`, log `logs/queue_t9_vast3.out` on the instance). When its O queue is DONE: fetch the O runs
  (`bash scratch/vfetch_toy.sh 174.0.242.177 1743 o`) but **do not destroy it**; destroy only after the t9 queue shows
  QUEUE_DONE and `bash scratch/vfetch_toy.sh 174.0.242.177 1743 t9` is checksum-verified. **Status 2026-09-29 ~07:05:**
  vast1 and vast2 O runs fetched (16 + 12 models; 4 stage-1 failures) and destroyed; vast3 O runs fetched (12 models,
  4 stage-1 failures), T9 continuing there. The 4 `theory-pa-h*` Vast instances belong to another agent — leave them.
* **Theory session (peer "Latent", docs/THEORY.md) owns the depth-wall vs schedule-end test:** THEORY §20 Pilot E +
  amendment D-1 to §17 (commit 3ded4f5, frozen before any run). Anytime supervision vs control, same-seed pairs: K 6
  (seeds 900-911), K 8 (1000-1007), cap-3 (1010-1017); scored on d8 (d6 for cap 3). Runs on its Vast hosts (theory-pa-h*),
  done ~2026-09-30 01:00; it will message the scored result. Don't touch its instances or theory/ files.
* **Chained after M (addendum O, long scaffold, paired with P30 seeds):** `chain_after.sh logs/queue_m_<gpu>.out o_<gpu> ...`
  starts `p6_o_<gpu>.txt` (4060: o_s100..111, 3070: o_s112..123, 2070: o_dev6_s51..56; `--aux_steps 20000`). Score:
  paired success (competent + extrapolates) vs `p30_s*` / `dev6_s51..56`; rules in PROTOCOL6.md addendum O (scorer `scratch/score_o.py`, written before data; was: scorer not
  yet written). Runs until ~Tue morning.
* Workers: `scratch/queue_worker2.sh <tag> <device> <queue>` (claims lines via `<queue>.claimed`, trains under
  `scratch/stage1_guard.py`, then runs `scratch/eval_battery.sh`). Chains: `scratch/p6_stall_chain.sh` (starts the 3070's
  H queue when `logs/queue_prec_3070b.out` shows QUEUE_DONE). Measurement loops: `scratch/p6_after2.sh <dev> <queues...>`
  (end activity + strict clock + checkpoint trajectory → `results/p6_endact.json`, `p6_strict.json`, `p6_traj.json`).
* All were launched **detached** from PowerShell (`Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList ...
  -WindowStyle Hidden`), so they survive a Claude session ending. Check with
  `Get-CimInstance Win32_Process | ? { $_.CommandLine -match "queue_worker2|p6_" }` and `nvidia-smi`.
* A queue whose lines are all claimed prints `QUEUE_DONE` in its log. To add work, append lines to an unclaimed
  queue file (workers re-read it at each claim) — **never edit a running `.sh` file** (bash reads scripts
  incrementally).

### Cloud: Vast.ai (3 × RTX 3090) — PROTOCOL6 addendum G (bypass-blocked Coconut on ProsQA)

| Instance | Seed / config | SSH (direct) | $/h | Started (EDT) |
|---|---|---|---|---|
| ~~52844008~~ | seed 0 — **done, fetched (`prosqa/res_g_s0`, `prosqa/ckpt/g_s0_checkpoint_50`), destroyed 14:30** | | | |
| ~~52846875~~ | seed 1 — **done, fetched (`prosqa/res_g_s1`, `prosqa/ckpt/g_s1_checkpoint_50`), destroyed 11:40** | | | |
| ~~52846882~~ | seed 2 — **done, fetched (`prosqa/res_g_s2`, `prosqa/ckpt/g_s2_checkpoint_50`), destroyed 13:10** | | | |

* Key: `~/.ssh/runpod_ed25519` (registered on Vast). On each host `/workspace/prosqa/pod_prosqa_g.sh <cfg>` does:
  clone official Coconut → speed patch (`patch_coconut.py`) → answer mask (`patch_mask.py`) → mask test
  (`test_mask.py`, passed on seed 0: 0 mass crosses, eager = SDPA) → train 50 epochs → analyses with
  `analyze.py --mask_answer` into `/workspace/prosqa/res` → prints `ANALYSIS_DONE` in `/workspace/logs/prosqa.out`.
* ~9.5 min/epoch in stage 0; later stages slower → expect **~20–25 h** (done ~Sun 16:00–21:00 EDT), ~$15–19 total.
* Status: `bash prosqa/status.sh vast:<host>:<port>` (GPU, errors, epoch, checkpoints, result files).
* **When a host shows ANALYSIS_DONE:** fetch with
  `bash prosqa/vfetch.sh <host> <port> prosqa/res_g_s<seed> /workspace/ckpt/<name>/checkpoint_50 prosqa/ckpt/g_s<seed>_checkpoint_50`
  (checksum-verified; `<name>` = `prosqa-coconut-g`, `prosqa-coconut-g-s1`, `prosqa-coconut-g-s2`), then **destroy the
  instance** (`vastai destroy instance <id>`) and mark it in `scratch/pods.txt`.
* Vast CLI: installed with `pip install --target <dir> vastai`; run as
  `PYTHONPATH=<dir> python -c "import sys; from vastai.cli.main import main; sys.exit(main())" <args>`. The API key is
  stored by the CLI in `C:\Users\peter\.config\vastai\vast_api_key` (never copy it into the repo). The 2026-09-26
  install lives in the session scratchpad and may be gone — reinstall if so.
* Validation accuracy printed during training is meaningless before the final stage (the official eval generates the
  written steps after `<end-latent>`, which the mask hides from the question). Only epochs ≥ 30 are meaningful.

### Cloud: Vast.ai — PROTOCOL6 addendum J (delayed-query ProsQA), started 2026-09-27 ~12:00

| Instance | Seed / config | SSH (direct) | $/h |
|---|---|---|---|
| ~~52981944~~ | seed 0 — **done, fetched (`prosqa/res_j_s0`), destroyed ~03:50; scored (PROTOCOL6_RESULTS.md)** | | |
| ~~52981949~~ | seed 1 — **done, fetched (`prosqa/res_j_s1`), destroyed ~04:45; scored** | | |
| ~~53145506~~ | seed 2 (tiebreak) — **done, fetched (`prosqa/res_j_s2`, ckpt `prosqa/ckpt/j_s2_checkpoint_50`), destroyed ~21:50; scored ~21:55: J1 needed (2 of 3), addendum J closed** | | |

* Script `prosqa/pod_prosqa_j.sh <cfg>`: as G plus `patch_delayed.py`, `make_delayed.py` (data/prosqa_d_*.json) and the
  pre-training check `test_delayed.py` (must print DELAYED_TEST PASS); analyses on the delayed test set incl.
  `kprime_small_test_e50.json` (K′ = 0–3). Expect ~17–20 h (done ~Mon 05:00–08:00 EDT).
* Fetch: `bash prosqa/vfetch.sh <host> <port> prosqa/res_j_s<seed> /workspace/ckpt/<name>/checkpoint_50 prosqa/ckpt/j_s<seed>_checkpoint_50`,
  then destroy. Score: `python prosqa/score.py prosqa/res_j_s<seed>` (J2 = T1, J3 = T2) + J0/J1/J1b/J1c from
  `kprime_test_e50.json` and `kprime_small_test_e50.json` (rules in PROTOCOL6.md addendum J). Seed 2 only if seeds
  0 and 1 disagree on J0, J1, J1b or J2.
* RunPod had no card ≤ $0.34/h on CUDA 12.4–12.8 at 11:50 (only A100/H100/H200 ≥ $1.39), hence Vast. Cheap Vast hosts
  on driver 580/590 were avoided (the driver family that broke on RunPod); both hosts passed cuInit.

### Cloud: Vast.ai — addendum K toy hosts: **all done and destroyed** (host A 01:55, host B 03:25; 34 runs fetched, sha-verified)

### Cloud: Vast.ai — PROTOCOL6 addendum N (delayed-query ProsQA on 4-layer GPT-2), started 2026-09-28 ~04:10

| Instance | Seed / config | SSH (direct) | $/h |
|---|---|---|---|
| ~~53139194~~ | seed 0 — **done, fetched (`prosqa/res_n4_s0`), destroyed ~15:40; N0 not competent (0.544)** | | |
| ~~53139198~~ | seed 1 — **done, fetched (`prosqa/res_n4_s1`), destroyed ~12:05; N0 not competent (0.552)** | | |

* `prosqa/pod_prosqa_n.sh <cfg>` = J script + `patch_layers.py` (run.py prints `N_LAYER 4`); K′ small budgets 0–5.
  Fetch/score as J (`res_n4_s<seed>`, `ckpt/n4_s<seed>_checkpoint_50`); `gpt2c.py` infers the layer count.
  Expected ~8–12 h (4 of 12 layers).

### Cloud: Vast.ai — addendum O extension (paired base/long scaffold on the same instance), started 2026-09-28 ~16:50

| Instance | Queue (`scratch/`) | SSH (direct) | $/h |
|---|---|---|---|
| 53251755 (RTX A4000) | `p6_o_vast1.txt`: ob/o_s124..131 | `ssh -p 40758 root@38.29.145.27` | 0.115 |
| 53251761 (RTX A4000) | `p6_o_vast2.txt`: ob/o_s132..139 | `ssh -p 40771 root@38.29.145.230` | 0.128 |
| 53251763 (RTX 4070) | `p6_o_vast3.txt`: ob/o_s140..147 | `ssh -p 1743 root@174.0.242.177` | 0.144 |

* 16 runs each (~50–60 min per run) → done ~Tue 08:00–10:00. Worker log `/workspace/Latent/logs/queue_o_vastN.out`.
* Fetch: `bash scratch/vfetch_toy.sh <host> <port> o` (checksum-verified; skips runs without model.pt), then destroy.

### RunPod — nothing running (`list-pods` empty). Registry: `scratch/pods.txt`.

**Provider order (user's instruction):** for any new cloud job, try RunPod first (prepaid credit left) at ≤ $0.34/h
with `allowedCudaVersions` 12.4/12.6/12.8; use Vast.ai only when RunPod has no stock. Jobs already on Vast stay there.

## 2. Budget (the user wants a warning when credit runs low)

* RunPod: ~$20 credit on 2026-09-25; **$11.42 spent** (2026-09-25 $4.51, 2026-09-26 $6.91) → ~$8.5 left.
  Rate cap $0.34/h per card; secure-cloud prices are often billed above the listing — check `cost`, delete if > $0.34.
* Vast.ai: **$12.97 at 2026-09-29 00:58**, 6 instances at $1.00/h (3 O-ext $0.39, 2 Q $0.43, T11 pilot $0.18).
  Estimated to finish: O-ext ~$2.5 (vast1/vast3 10/16 runs, vast2 7/16 at 00:50), Q ~$7 (epoch 12/50), T11 pilot ~$0.8
  → **~$10.5 of $12.97, ~$2.5 margin. User warned 01:05.** T11 confirmatory seeds (~$7) need a top-up (Vast or RunPod).
  If credit hits 0 Vast stops the instances — fetch O-ext hosts as they finish to cut spend.
* Local GPUs are free.

## 3. Waiting to be scored (and how)

| Item | What | How to score | Status |
|---|---|---|---|
| Addendum F (P27, precision) | same-card bf16/fp32 pairs, seeds 50–57 (+ 2070 arm 50–53) | done | **scored 22:05: not supported** (b = 0, f = 2); see PROTOCOL6_RESULTS.md |
| Addendum H (P28, fp32 stalls) | same-card pairs, seeds 70–77 | done | **scored 10:30: not supported** (b = 1, f = 1) |
| Addendum G (ProsQA, 3 seeds) | G0–G3 | done | **all 3 seeds: competent, needed, sweep absent, not step-locked; K′ = 1 ≈ 0.93** |
| Addendum I (P29) | P7 ≥ 0.2 on new runs | done | **scored 13:25: not supported** (acc 0.50, AUC 0.73) |
| Addendum K (P30) | P7 as a ranking, floor in_peak ≥ 0.03 | **scorer not yet written**: adapt `scratch/score_p29.py` (TEST = p30_s100..123 + dev6_s51..56; exclude in_peak < 0.03; AUC ≥ 0.80 and > AUC(EA) → supported; < 0.65 → not). EA auto-measured by `p6_after2.sh cuda:2` loop | ~Mon 02:00 |
| Attention sharpness (exploratory) | is closure visible at full-depth entry? | done 23:45 (18 runs): **no** — see PROTOCOL6_RESULTS.md exploratory section; rerun `scratch/sharpness.py` on the `stall_*` runs when H finishes if wanted | done |
| Extension B″ (P26 power, exploratory extension) | dev6 recipe seeds 45–50 on the 2070 | not yet in `latent/protocol6.py`: pool all competent dev6 s30–s50 with the frozen P26 rule (`p26()`), and fp32-only; B1-style class at full+1000 vs final | after the runs (Sun morning) |

After scoring: add to `docs/PROTOCOL6_RESULTS.md`, `docs/RESEARCH_LOG.md`, and this file.

## 4. Context: where the research stands (details in PROTOCOL6_RESULTS.md)

* Toy setting: 2-layer Coconut-style model on graph reachability (delayed format: the query comes after the thoughts,
  so the thoughts are causally necessary). Established earlier: breadth-first frontier sweep; readable ≠ usable;
  per-frontier-edge attention budget; **end activity (EA)** at the last trained step predicts extrapolation to longer
  chains (prospective, PROTOCOL5 P26).
* Part VI so far: EA is **stable** under 15 000 extra steps (A1, 11/12); P26 holds on a second recipe (thin: one low-EA
  run); the open/closed outcome is **fixed by full-depth entry** and not before stage 4 (branching E1: 2/4 → 5/6 → 6/6);
  closure can happen during normal training (dev6_s35) and is decided in the weights before EA shows it.
* ProsQA transfer (addendum D): the official Coconut GPT-2 (reproduced, 0.974) **does not use its thoughts** — 0 thoughts
  0.976, no hop-by-hop sweep; answer hidden from the question → chance. Independently replicated on the public
  checkpoint `bmarti44/coconut-curriculum-checkpoints` (0.977 vs 0.973; 0.54/0.59 masked). Matches Aswal et al.
  2606.12689 and Rizvi-Martel et al. 2604.06374 (fine-tuned models take shortcuts). Addendum G asks the real transfer
  question: when the answer can only see the thoughts, do they sweep hop by hop?
* Precision: fp32-trained runs stall in the curriculum far more than bf16 (F data) → addendum H tests it on new seeds.
* Literature position (LITERATURE_NOTES.md): the BFS/superposition mechanism is prior work (Zhu et al. NeurIPS 2025,
  ICLR 2026); our distinct contributions are the per-edge budget law, readable ≠ usable (quantitative), EA as a
  prospective extrapolation predictor, and the commitment window (branching).

## 5. Candidate next steps (choose from evidence; pre-register first)

1. F is scored (P27 not supported). Score H; if H is supported, the "fp32 open sweeps don't extrapolate" pattern is a training-reliability
   effect — say so in the findings.
2. Score G per seed. If the thoughts become necessary and sweep hop by hop → the mechanism transfers to GPT-2 when
   the bypass is removed (headline result). If they are necessary but don't sweep → a different (goal-directed?)
   latent algorithm; analyse which statements each thought reads.
3. Sharpness at full-depth entry with the new checkpointed runs (mechanistic account of the commitment window,
   bounded index-matching logit, Zhu et al. 2026).
4. Update `report/findings.md` and `docs/RESEARCH_TIMELINE.md` with Part VI once F/G/H are scored.

## 6. Conventions and gotchas

* Every confirmatory prediction goes into a dated PROTOCOL addendum **before** its data exist; deviations are dated
  amendments; exploratory analyses are labelled as such.
* Windows: Python writes CRLF — run `sed -i 's/\r$//'` on any `.sh` before sending it to Linux. Use `;` in
  PYTHONPATH on Windows. Local jobs must be detached (PowerShell `Start-Process`) or they die with the session.
* Cloud: RunPod driver-580 / CUDA 13.0 hosts are broken (`cuInit` 999) — filter with `allowedCudaVersions`; always
  run the cuInit check. Vast hosts on driver 550/565 work. Coconut's `eval()` returns None (don't chain it).
* Official Coconut: `datasets.map(num_proc=32)` forks — build models before datasets (as run.py does), or it hangs.
* Memory notes for Claude live in `C:\Users\peter\.claude\projects\D--\memory\` (RunPod setup, Vast setup, autonomy).
* The user has given full control over direction and spending within the caps; report results and costs, warn when
  credit runs low.
