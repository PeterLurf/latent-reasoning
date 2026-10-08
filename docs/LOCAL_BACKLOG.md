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

4. [done 2026-10-07 11:34 EDT: MOVES by the registered rule, mostly seeds stuck in both arms; details in `docs/THEORY_STABLE_LATENT_2026-10-06.md`, candidate 3] **Ramp-disentangling prototype** (Theory, stable-latent candidate 3; descriptive). Does escape without learning move with the rounding
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

**[Theory 2026-10-07 04:10, night filler; descriptive prototypes only, read for no rule; each a short run under the 6 GiB cap]**
* **4060 + 3070 (bf16): §140 digit-pair 4×4 entry prototypes to 5k.** `python -m boxes.multpair --task 4x4 --seed S --stop_after 5000 --no_cert --device <card> --out
  boxes/runs_fsq/proto140_4x4_s{S}_{card}` for S = 501..508 (alternating cards).
  - Read the per-step CE at c_max 2 (2k–5k) and the 2k flag. It previews whether the pair op enters before §140 could launch.
* **4060 + 3070: §137 column 4×4 prototypes to 5k, the same seeds 501..504** (`--family mult_col --task 4x4 --stop_after 5000`, out `proto137_4x4_s{S}_{card}`).
  - This is the paired comparison with the pair op at c_max 2 and 4, on the same cards. The §137 proof on b116 shows the per-column CE rising with L_max.
* **2070 (fp32): §136 d4-vs-d8 2k entry pairs, more seeds**: as item 7 for S = 513..524, interleaved. That takes the paired comparison to n = 24.
* Order per card: the §140 prototypes, then §137, then the §136 pairs. About 25–40 min per 5k run; together with the C7 checks this fills the night.

**[Theory 2026-10-07 06:35, 3070 + 4060 day filler; descriptive prototypes, read for no rule; under the 6 GiB cap]** (in order; at least 3 jobs and at least 2 h per card)
1. **§141 r4 entry prototypes, 4×4 to 5k:** `multcol --task 4x4 --repeats 4 --repeat_index --micro 4 --seed S --stop_after 5000 --no_cert` for S = 501..504 (FSQ-8),
   alternating cards; out `proto141_r4f8_s{S}_{card}`. Read whether the middle columns move at c_max 2 → 4 against §137's r1 trajectory.
2. **§140 digit-pair 4×4 prototypes to 5k:** S = 501..508, alternating; out `proto140_4x4_s{S}_{card}` (from the night list, if not run yet).
3. **S5 soft against hard entry, to 4.5k** (the 133-10 hint that SOFT-16 learns seed 3 where FSQ-8 and FSQ-16 escape and fail):
   - `fsq120 --family s5 --arm final --seed S --d_q 16 --soft_beta 0.1 --stop_after 4500 --no_cert`, and the same with `--d_q 16` only (FSQ-16);
   - S = 501..508, paired, alternating cards; out `proto133_{soft16,fsq16}_s{S}_{card}`;
   - read the 2k flag and FLAG4 per pair.
4. **C7 LM-head check on every remaining exact model** (minutes each).

**[Theory 2026-10-07 09:45, RTX 2070 (fp32, at most 6 GiB), at least 2 h]**
1. **probe140_acc** (if not yet run): the true accumulator's R² from code and from h on s140_pair_4x4_s1, by step within the column (1st–4th pair) and by magnitude bin
   (< 100, 100–200, > 200), into `eval_probe_acc.json`. Minutes.
2. **S5 SOFT-16 against FSQ-16 entry pairs to 4.5k** (133-10's hint), the 2070 share: S = 509..516, paired (`--d_q 16 --soft_beta 0.1` against `--d_q 16`), fp32,
   `--stop_after 4500 --no_cert`; out `proto133_{soft16,fsq16}_s{S}_2070`. Read the 2k flag and FLAG4 per pair (a third card class beside the 3070 / 4060 pairs at
   S 501–508). About 40 min each in fp32, so about 10 h in all.
3. **The per-repeat probe on any finished §141 r4 prototype** (proto141_r4f8_*): the partial column sum's and the carry's R² from code per repeat, plus ‖Δcode‖ (§141's
   s141_probe fields), into `<run>/eval_probe_repeat.json`. Minutes each.
4. Filler if all else is done: fp32 replay_x seeds 541..560 (as earlier).

**[Theory 2026-10-07 21:13, 4060 + 2070 (the cards idle since about 20:55); descriptive prototypes, $0; the readings fixed now, before any run; under the 6 GiB cap, memory-checked first]**
1. **§150 soft-only DIGIT working-memory prototype** (Audit's suggestion; the user's token-like working memory).
   - **Why:** DIGIT s1's loss was 0.016 while soft (2k), then collapsed as τ annealed and slots hardened (1.02 at 6k, 1.72 at 24k).
   - **Command:** `python -m boxes.multpair --family mult_pair --task 4x4 --slots 4 --slot_space digits --slot_gumbel --slot_tau 1:1:2000:6000 --slot_hard 999999:1000000
     --seed S --stop_after 8000 --no_cert --micro M --device {DEV} --out boxes/runs_fsq/proto150_digitsoft_s{S}_{card}`
     - τ is held at 1 and the slots never harden.
     - **Micro:** M = 2 on the 4060 and 4 on the 2070 (fp32), effective batch 64. Memcheck first.
   - **Seeds:** S = 501 / 502 on the 4060 and 503 / 504 on the 2070.
   - **Reading (fixed now, from the logged loss; c_max is 4 from 5k):**
     - **CARRIES** if the mean loss over 6k–8k is ≤ 0.1, the near-0 level DIGIT reached while soft at c_max 2. The soft vocab memory then keeps carrying past two columns, and
       the s1 collapse came from annealing / hardening.
     - **FAILS** if it is ≥ 0.3 (§140's level at c_max 4). The soft vocab memory hits the same wall.
     - PARTIAL in between.
   - **Time:** about 1 h per run on the 4060 and about 2 h on the 2070.
2. **§144 sum4 no-stop entry prototypes, more seeds** (4060, after item 1), beside the 3070's S 509–512.
   - **Command:** `python -m boxes.fsq120 --family sum4 --sum_mod 10000 --arm final --seed S --stop_after 8000 --no_cert --device {DEV}
     --out boxes/runs_fsq/proto144_{f8,n64}_s{S}_4060`, for S = 513..516 alternating F8 (`--d_q 8`) and N64 (`--d_q 64 --no_round --dither 0.25`), with no
     `--stop_plateau`.
   - **Reading (descriptive):** whether loss(8k) falls below 5.0 (entry after the 2k flag), per arm.
   - About 40–60 min each.

**[Theory 2026-10-07 21:50, PRIORITY (next free local card); a descriptive harness-validity check for §144 (Audit), $0; the reading fixed now, before any run]**
1. **sum4 L = 1 check.**
   - **Why:** F8's mod-100 rung sat at exact chance for 24k (ln 100, codes_per_state 2.99, ρ = 0).
   - **Needs** a one-line `--lmax_fixed L` option in fsq120 (every chain has length L; the experiments session's code).
   - **Command:** `python -m boxes.fsq120 --family sum4 --sum_mod 100 --arm final --lmax_fixed 1 --seed S --stop_after 5000 --no_cert --device {DEV}
     --out boxes/runs_fsq/proto144_L1_{f8,n64}_s{S}_{card}`, for S = 601 / 602, with F8 (`--d_q 8`) and N64 (`--d_q 64 --no_round --dither 0.25`). No `--stop_plateau`.
   - **Reading (fixed now, from the logged loss):**
     - **HARNESS OK** if the loss falls below 0.1 by 5k in at least one seed per arm. The harness can write one sum into the code, so §144's chance results are an
       entry / curriculum lottery.
     - **STRUCTURAL BLOCK** if all four stay at or above 4.4 (≈ ln 100). Something blocks writing even one sum (FSQ at init, the 4-token decoder, the start path), and §144's
       FINAL results are not read as entry or carriage until it is found.
     - PARTIAL otherwise.
   - About 10–15 min each.

**[Theory 2026-10-07 23:26, overnight; descriptive prototypes, $0; the readings fixed now, before any run; memory-checked first, under the 6 GiB cap]**
0. **First:** Experiments' §152 gates and smokes (step-0, gradflow, write-nothing-back, memory, smoke_plain) take any free card before the items below.
1. **§150 soft-only FULL protos** (2070, after proto150_digitsoft_s504; fp32, micro 4): item 1 of 21:13 with `--slot_space vocab`, S = 505 / 506, out
   `proto150_fullsoft_s{S}_2070`. Same reading: **CARRIES** if the 6k–8k mean loss is ≤ 0.1, **FAILS** if ≥ 0.3, else PARTIAL.
   - The DIGIT twins (s501–s503) all FAIL: 6k–8k means 1.09 / 1.57 / 1.11.
   - If FULL fails too, the single-vector token slot (B4) fails at the soft level for both vocabularies, independent of hardening.
2. **DeltaMem 5×5 digit-pair protos to 12k** (3070 after proto144_n64_8k_s512, then 4060 after s516):
   `python -m boxes.multpair --family mult_pair --task 5x5 --carrier deltamem --steps 12000 --seed S --no_cert --micro M --device {DEV}
   --out boxes/runs_fsq/proto147_5x5_s{S}_{card}`, with S = 501 on the 3070 and 502 on the 4060.
   - Run a `--steps 50 --train_n 2000` smoke first: the end-of-run eval (donor / ext) must pass on 5×5 before the 12k run.
   - **Reading (descriptive):** acc_by_column.per_column at 12k (c_max 8 from 9k). Column 5 has 5 products, which 4×4 never has. Column 5 ≥ 0.9 means DeltaMem handles
     5-product columns, the precondition for §152 at 5×5. Below 0.5 means it does not at this budget.
   - About 4 h each.

**[Theory 2026-10-08 03:11, 3070 (idle since 02:53); descriptive previews, read for no rule; memory-checked first, under the 6 GiB cap]**
1. **§156 (a) local previews**, after Audit re-checks §156's code: loop148 `--loops 1`, 8k steps, seed 1, with `--answer_sep interleave`, `--answer_sep pause`, and no flag (PLAIN),
   out `proto156_{sep,pause,plain}_s1_3070`.
   - **Reading (descriptive):** the test exact and per-column curves on the 3070 class, as an early look before the registered 3090 runs. They do not score §156.
   - About 15 min each.
2. Then §154 SHIFT's local 4×4 proto at 8k (multpair `--carrier deltamem --add_reset column`, seed 501, out `proto154_shift_s501_3070`): the early column profile and α by flag
   beside b161 (descriptive).

**[Theory 2026-10-08 06:30, 3070 + 4060 (dry since 06:13 / 06:21); descriptive previews, read for no rule, kept apart from the registered cells; memory-checked first, under the 6 GiB cap]**
1. **4060: §158 FREE-PAIRS preview.** §158's command (§155 R's args + `--free_pairs 0.5`) with `--steps 8000`, seed 1, out `proto158_free_s1_4060`.
   - **Read (descriptive):** consistent-bench test_acc, free-chain accuracy and the row-level donor at 8k. It is an early look beside b173 (24k, 3090) and does not score §158.
2. **3070: §157 CCQ preview.** §157 CCQ's command (§152 A's args + `--read_mode ccq`, note 157-1) with `--steps 8000`, seed 1, out `proto157_ccq_s1_3070`.
   - **Read (descriptive):** non-finite steps, λ by head, loss at 8k. It is an early instability check of the numerically riskiest arm.
   - Then the same for `--read_mode multiread` (`proto157_multiread_s1_3070`) if time allows.

## Done

* **§160 A-R preview (2070, 8k, fp32, descriptive):** test 0.0 (by rows 1.0 / 0.998 / 0.12 / 0.0); columns 0.99 / 0.69 / 0.17 / …; finite, loss 0.17; 0 floor spikes;
  7.10e9 FLOPs, 12 passes. At 8k it is still on row 3, so it says nothing about b180 (48k, 3090).
* **§157 previews (8k, descriptive; 06:30 item 2):** CCQ s1 (3070) test 0.0 (by length 1.0 / 0.26 / 0.03 / 0.0), no non-finite steps, final loss 0.34; λ ≤ 0.008 on 8 of
  12 heads. MULTIREAD s1 (2070, fp32) 0.0 (1.0 / 0.13 / 0.01 / 0.0), loss 0.49. Both are still on row 2 at 8k, so they say nothing about the read (the registered 48k cells: 0.064
  / 0.009).
  - **PERIODIC s1 (3070, 8k):** 0.0 (1.0 / 0.22 / 0.03 / 0.0), finite, loss 0.30. The read scalar is not integer-valued: |s − round(s)| has median 0.25 and share < 0.25
    of 0.50 in every column, which is the uniform value. The off-column weight is 0.89–0.97, so the read is a blend at 8k.
* **§158 FREE-PAIRS preview (4060, 8k, descriptive; 06:30 item 1):** consistent test 0.0, with columns 1.0 / 1.0 / 0.68 / 0.09 …; free chains 0.0; no non-finite steps;
  loss 0.26.
  - The state buffers raw digits rather than sums: raw 0.88 against accumulator 0.12 pooled (after pairs 4 / 8 / 12: raw 1.0 / 0.92 / 0.82 against 0.28 / 0.07 / 0.05;
    n_fit 3,000).
  - Propagation 0.09 against null 0.08. It is still learning column 3 at 8k, so it says nothing about b173 (24k, 3090).
* **§150 soft-only DIGIT (21:13 item 1): FAILS**, 4 of 4 (6k–8k mean loss 1.09 / 1.57 / 1.11 / 1.10, ≥ 0.3). **Soft-only FULL (overnight item 1): FAILS**,
  S505 / S506 at 1.114 / 1.119. The loss rises when c_max reaches 2 at 2k, without
  annealing or hardening. DIGIT s1's collapse was the curriculum, not the hardening (as 150-3).
* **§144 L = 1 check (21:50): PARTIAL.** Loss at 5k: s601 F8 2.44 / N64 2.48 (about ln 10: one of two digits); s602 F8 4.89 / N64 4.98 (chance).
* **DeltaMem 5×5 proto s501 (3070, 12k; 8e29e417 item 2):** per column 1.0 / 1.0 / 0.915 / 0.725 / **0.494** / 0.178 / 0.109 / 0.089 / 0.07 / 0.178. Column 5 is below 0.5, so it
  does not handle 5-product columns at this budget (fixed reading). **s502 (4060):** 1.0 / 1.0 / 0.837 / 0.36 / **0.151** / 0.10 / 0.095 / 0.076 / 0.101 / 0.187; column 5 is
  again below 0.5, so both seeds read "no 5-product columns at 12k".
* **§144 sum4 8k no-stop (3070):** no entry. Loss at 8k: s511 F8 9.79 / N64 10.16; s512 F8 9.9 at 4.5k (all ≥ 5.0).
