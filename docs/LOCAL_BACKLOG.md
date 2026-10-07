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

4. **Ramp-disentangling prototype** (Theory, stable-latent candidate 3; descriptive). Does escape without learning move with the rounding
   ramp?
   - Paired short runs on seeds 601–610, fsq120 S5 FINAL, `--stop_after 8000`:
     - **plain;**
     - **late-ramp:** p_hard 0 → 1 over 6k–8k, L_max unchanged. Needs a `--p_hard_start` / `--p_hard_end` option (Experiments).
   - Outputs `boxes/runs_fsq/rampproto_{plain,late}_s{S}_4060`.
   - Read loss(4k) / loss(2.6k) and loss(7k) / loss(5.6k) per run.
   - **Reading fixed before any late run** (`theory/proto_ramp.py`, revised after the helper's review; late arm stops at 8.9k). Outcomes: STAYS (late A-flags ≥ ¾ of plain's), MOVES (late R′ = 8.9k / 6.6k flags cover ≥ ½ of the plain A-flagged seeds), VANISHES (late A-flags ≤ ½ of plain's, ≤ 1 late R′-flag, and lower late loss on most of those seeds; candidate 3's claim). Needs ≥ 2 plain A-flags. Pooled verdict, with per-card readings printed.
   - **Seeds revised (Theory, after 601 / 602 flagged at 2k, before any informative pair):** 4060 runs 603 (running), then **3, 4, 8**; 3070 runs **14, 16, 27, 28, 31** (in place of 606–610). They are seeds whose same-card 2k replay was not flagged; seeds with a known 24k outcome (s130chk) are excluded. A plain run that flags at 2k stops there, and its late twin is skipped. Dirs `rampproto_{plain,late}_s{S}_{card}`.
   - Peak about 6 GB (L_max 8 from 5k; micro as needed; memcheck first). About 30 min each. Stop: 8k steps.

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
     - Then add **s127_s5_ctrl_s4_full** (a plateau that ran to 24k) and, when fetched, the §127 lever models (std2 s1–s3, lr10 s2 / s3, skip s2 /
       rerun / s3). The helper asked for these; the line format is as above.

4. **§115 completion (short; Theory).** Missing eval_s115.json. Correction: only **g105_mask4_s1** is in §115's registered set (§108 MK-4 s1 / s3 / s5 plus §105 s1–3). s108_mk4_s2 / s4 are basin seeds outside it, so they are **descriptive only**:
   - `python -m boxes.snap115 --run boxes/runs_s5/s108_mk4_s2 --device <3070>`
   - `python -m boxes.snap115 --run boxes/runs_s5/s108_mk4_s4 --device <3070>`
   - `python -m boxes.snap115 --run boxes/runs_s5/g105_mask4_s1 --model g97 --device <3070>` (part B only, as for g105 s2 / s3)
   - Memcheck first; eval only, bs 250. Stop: done. g105_mask4_s1 finishes §115's per-model rules.

## RTX 2070 (Turing, fp32 only, 8 GB, slowest)

1. [moved to the cloud, Theory 2026-10-06: the FF certs are done locally; the S5 CERT needs more than 6.5 GiB in fp32, over the memory gate] **§128 S5 CERT**: run as a **tail job on a cloud 3090** (an existing rerun host after its runs), `--cert_only --fp32` only (128-4: the fp32 cert is the reproducible record), on s128_s5_s1, s102, s104, --cert_time 2700. Descriptive.
4. [shared filler, Experiments] fp32 contraction on the remaining kept s120 / s123 / s126 / s127 / s129 models.
2. [queued: scratch/c1_backlog_2070.txt] **§128 code growth** (codes against position on 2,048-step rollouts, descriptive).
   - `python -m boxes.fsq120 --from_args <run dir> --growth_only` for the 19 learners. Peak about 2 GB. Minutes each. Stop: done.
3. [queued: c1_backlog_2070.txt] **Cross-card replay pairs, Turing fp32:** `python scratch/replay_basin.py --seed S --steps 2000 --every 50 --amp none --device <2070> --out theory/results/replay_x/2070_s{S}.jsonl`
   for S = 501..510.
   - About 15 min each. Stop: 2k steps.

5. **[Theory 2026-10-07, priority] §133 SOFT-16 s1 fp32 contraction** (its slot verdict needs it; the model is local):
   - `python -m boxes.fsq120 --from_args boxes/runs_fsq/s133_soft16_s1/args.json --contraction_only --fp32 --device <2070> --out boxes/runs_fsq/s133_soft16_s1`
   - fp32 is device-independent here (128-4), and the device is recorded. Minutes.
6. **[Theory 2026-10-07] Post-hoc prefix + end probe on the local §133 / §134 models** (descriptive mechanism, F2 taxonomy): for each of
   s133_{cont64_s201, fsq16_s1, fsq64_s201, noise64_s201, soft16_s1, soft16_s4} and s134_copy_s5_s{1..4}:
   - `python -m boxes.fsq120 --from_args <run>/args.json --prefix_only --device <2070> --out <run>`
   - `python -m boxes.fsq120 --from_args <run>/args.json --probe_only --device <2070> --out <run>`
   - About 2–3 GB, minutes each. Delete nothing.
7. **[Theory 2026-10-07] §136 entry prototype, d 4 against d 8 at 2k, paired on the 2070** (descriptive; previews GROW-4's entry. Different card class
   and numerics from §136's 3090 runs, so not part of §136's rules):
   - `python -m boxes.fsq120 --family s5 --arm final --seed S --d_q W --stop_after 2000 --no_cert --device <2070> --out boxes/runs_fsq/proto136_d{W}_s{S}_2070`
     for W in (4, 8), S = 501..512, interleaved (d4 s501, d8 s501, …).
   - Read: the 123-3 flag at 2k per pair. The 2070's fp32 d-8 2k replays (replay_x) flag 13/40, as a cross-check of the d-8 leg.
   - About 2–3 GB, about 15 min each. Stop: 2k.

**[Theory 2026-10-07, any free card, $0] C7 readability check (Audit's steward idea; descriptive):** on the kept exact models (§134 COPY s1–s4, §128 families, §133
SOFT-16), project each eval code's embedding W_up(code_k) through GPT-2's final LN and tied LM head. Report the top-1 / top-5 agreement with the state's
text token (S5: the first state token; FF / CYC / MOD / RDFA: the answer token) per family.
* A short script (the experiments session), eval only, about 1–2 GB, minutes per model. Output `<run>/eval_c7_lmhead.json`.
* It is a cheap number for C7 (a human-readable latent) and does not enter any rule.

## Done

* (none yet)
