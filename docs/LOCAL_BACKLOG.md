# Local GPU backlog — short jobs only (maintained by Theory; Experiments pulls)

**Rules (user, 2026-10-06):** local cards must never sit idle; they run only short, memory-checked jobs, one per card; full
training runs go to the cloud (≥ 16 GB, checkpointed). Everything here is **descriptive or prototype**. A result that would
support a claim needs a THEORY registration first, and these outputs are not read as evidence for any registered rule.
**Experiments' runner (2026-10-06):** `scratch/c1_backlog_worker.sh` per card (bl4060 / bl3070b / bl2070b) pulls `scratch/c1_backlog_<card>.txt`
automatically, memory-gated (each line carries its peak; > total - 1.5 GiB refused; live sampling halts the worker on overrun).
**Pull order:** take the first open entry for the free card. Mark it `[running]` / `[done <commit>]` in this file. Theory adds
entries, keeping ≥ 3 open per card.
**Every entry:** card, exact command, expected peak memory, stop condition. Use non-slot output names (never a registered
`sNNN_<fam>_s<seed>` pattern).

## RTX 4060 (Ada, bf16, 8 GB)

1. [queued: scratch/c1_backlog_4060.txt, after replay_extra 75-82] **§132 rung-2 entry prototype (k6 / m3) at 2k.** Does the 6-state table enter at 2k on this card?
   - Needs `--stop_after` ported to `boxes/table132.py` (fsq120 has it).
   - `python -m boxes.table132 --k 6 --m 3 --seed S --steps 24000 --stop_after 2000 --stop_plateau --device <4060> --out boxes/runs_fsq/s132proto_k6_s{S}_4060`
     for S = 1..4.
   - Peak about 4 GB (L_max 2 to 2k, 75-token segments). About 10 min each.
   - Stop: 2k steps, or 30 min per seed.
2. [queued: c1_backlog_4060.txt] **Cross-card replay pairs** (the same seeds on all three local cards, to time when GPU class makes trajectories diverge;
   125-2).
   - `python scratch/replay_basin.py --seed S --steps 2000 --every 50 --amp bf16 --device <4060> --out theory/results/replay_x/4060_s{S}.jsonl`
     for S = 501..520.
   - Peak about 3 GB. About 5 min each. Stop: 2k steps.
3. [queued: c1_backlog_4060.txt] **4k-flag prototype data** (FLAG4 rates by card class; not §131 evidence, since §131's held-out set is fixed).
   - `python -m boxes.fsq120 --family s5 --arm final --seed S --stop_after 4500 --probe127 --device <4060> --out boxes/runs_fsq/flag4proto_s5_s{S}_4060`
     for S = 601..620.
   - Peak about 6 GB (L_max ≤ 4 up to 5k). About 20 min each. Stop: 4.5k steps.

## RTX 3070 (Ampere, bf16, 8 GB)

1. [queued: scratch/c1_backlog_3070.txt, after the fp32 contraction filler] **Cross-card replay pairs:** as 4060 #2, `--device <3070>`, output `theory/results/replay_x/3070_s{S}.jsonl`, S = 501..520.
   - About 6 min each. Stop: 2k steps.
2. [queued: c1_backlog_3070.txt] **4k-flag prototype data:** as 4060 #3, `--device <3070>`, output `flag4proto_s5_s{S}_3070`, S = 601..620. Stop: 4.5k steps.
3. [queued: c1_backlog_3070.txt; fsq120 --probe_only added] **Probe sweep on §128's 19 learners** (R²_state / R²_item / op-history at the end of training).
   - Needs a small post-hoc CLI wrapping fsq120's `probe127` on a model.pt (Experiments).
   - Output: `<run dir>/eval_probe_posthoc.json`. Peak about 2 GB. Minutes each. Stop: done.
   - **Priority addition (Theory, stable-latent synthesis; Literature 4da2f672's $0 test):** include **s128_s5_s3** (accepted, escaped
     without learning; model kept) and the S5 learners s1 / s102 / s104. Extend the probe with **op-history R² (last 1 / 2 / 3 ops)** beside
     R²_state, at chain positions L 2 / 4 / 8.
     - Does the code that escaped without learning carry the op history (the short-chain shortcut) where the learners carry the state?
     - Run these four first.

## RTX 2070 (Turing, fp32 only, 8 GB, slowest)

1. [moved to the cloud, Theory 2026-10-06: the FF certs are done locally; the S5 CERT needs more than 6.5 GiB in fp32, over the memory gate] **§128 S5 CERT**: run as a **tail job on a cloud 3090** (an existing rerun host after its runs), `--cert_only --fp32` only (128-4: the fp32 cert is the reproducible record), on s128_s5_s1, s102, s104, --cert_time 2700. Descriptive.
4. [shared filler, Experiments] fp32 contraction on the remaining kept s120 / s123 / s126 / s127 / s129 models.
2. [queued: scratch/c1_backlog_2070.txt] **§128 code growth** (codes against position on 2,048-step rollouts, descriptive).
   - `python -m boxes.fsq120 --from_args <run dir> --growth_only` for the 19 learners. Peak about 2 GB. Minutes each. Stop: done.
3. [queued: c1_backlog_2070.txt] **Cross-card replay pairs, Turing fp32:** `python scratch/replay_basin.py --seed S --steps 2000 --every 50 --amp none --device <2070> --out theory/results/replay_x/2070_s{S}.jsonl`
   for S = 501..510.
   - About 15 min each. Stop: 2k steps.

## Done

* (none yet)
