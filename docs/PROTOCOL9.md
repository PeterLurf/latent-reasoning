# PROTOCOL9 — does anytime supervision extend the horizon across architectures?

Written 2026-09-29 16:08 EDT, committed before any PROTOCOL9 run started. At that time:
- none of the paired T9/T10 control runs (seeds below) had finished;
- THEORY Pilots D/E (anytime vs control on the base architecture) had not reported.

Append-only; changes go in dated amendments at the end.

## Why

The theory session's account is that the horizon is a schedule end, caused by slack. Its proposed fix is anytime
(minimum-progress) supervision: after every latent step k, supervise the queries whose answer depth is ≤ k + C. So far
the fix has been tested only on the base 2-layer, 256-wide architecture: Pilot B (exploratory, 3/4 extrapolate), and
Pilots D/E (running, same-seed pairs, K = 6 / K = 8 / cap 3).

PROTOCOL7 T9/T10 already train controls on four other families: 3 layers, 4 layers, 512 wide, and fork graphs. Adding
`--anytime 1` to the same seeds gives a paired test of whether the fix generalises beyond the architecture it was found
on. This is the experiments session's role: an independent, pre-registered check of a theory fix. It is registered
before D/E report, so its design does not depend on their outcome.

## Design

- **Anytime arm.** 32 runs. Each queue line is the paired control's line plus `--anytime 1` (C = 1: the delayed query
  adds one hop).

  | Family | Anytime runs | Paired control (T9/T10) | Precision |
  |---|---|---|---|
  | 3 layers | `p9at_l3_s500..507` | `t9l3_s500..507` | bf16 |
  | 4 layers | `p9at_l4_s520..527` | `t9l4_s520..527` | bf16 |
  | 512 wide | `p9at_w512_s540..547` | `t9w512_s540..547` | bf16 |
  | fork graphs | `p9at_fork_s560..567` | `t10fork_s560..567` | fp32 |

  Queues: `scratch/p9_bf16.txt` (sha256 b5d670750efd0ff5…) and `scratch/p9_fork.txt` (1ad2aedbb0e22490…).
- **Controls.** The same-seed T9/T10 runs, trained and evaluated under PROTOCOL7 (amendment 8 placement). The controls
  ran on Vast cards, the anytime arm runs on local cards, with the same precision per family. The hardware difference
  is a known, accepted source of run-to-run noise; it is not systematic in the arm.
- **Code.** A frozen copy of the theory session's training script, taken at commit 6e30507: `scratch/p9_frozen/`, with
  `train_arch.py` (sha256 cb520b7228b667cc…) and `arch.py` (f439c3eab053d43d…). It's run by `scratch/queue_worker3.sh`
  (ebbd21e0fe591d2e…), which is `queue_worker2.sh` with the training entry point swapped. Same base recipe, stage-1
  guard and evaluation battery. With `--arch coconut` (the default), the saved model is a plain `LatentReasoner`, so
  the standard evaluators apply. That was checked on `theory/runs/pb_at_s753` with `latent.competence`. CPU smoke tests
  passed for anytime with 4 layers, 512 wide and fork graphs.
- **Cards.**
  - The RTX 3070 and RTX 4060 run `p9_bf16.txt` after their T10 workers finish, then join `p9_fork.txt`.
  - The RTX 2070 runs `p9_fork.txt` after T10.
  - Chained with `scratch/chain_after_w.sh`.
- **Cut-off.** Runs not started by Sun 2026-10-04 00:00 are dropped (and their pairs).

## Outcomes (scorer `scratch/score_p9.py`, sha256 8a252e45422b1dfa…, written before any data)

- **Measures.**
  - Competence and extrapolation use the PROTOCOL5 rules (`latent.protocol5_confirm.pilot_run`). A run with no model
    (a stage-1 failure after its relaunch budget) is not competent.
  - d8 = latent accuracy at distance 8 with K′ = 12 (`length_gen`, trained distance + 2: the first distance past the
    query-time hop, as THEORY §20 E-1).
- **AP1 (primary, pooled over the four families).**
  - Evaluable pairs: both runs competent and d8 not exactly tied; at least 10 are needed, else "unresolved".
  - **Supported** if the anytime run has the higher d8 in ≥ 70 % of evaluable pairs and the one-sided sign test gives
    p < 0.05.
  - **Not supported** if ≤ 50 %; otherwise inconclusive.
- **AP2 (extrapolation success).** l = pairs where only the anytime run extrapolates; b = pairs where only the control
  does.
  - **Supported** if l − b ≥ 4 and the one-sided sign test gives p ≤ 0.05.
  - **Not supported** if l ≤ b; otherwise inconclusive.
- **AP3 (learnability cost).** Pooled competent counts, anytime vs control. It's called a "cost" if the anytime arm has
  ≥ 6 fewer competent runs of 32. Per-family counts are reported.
- **Descriptive:**
  - the AP1 share per family;
  - mean d7/d8/d9 by arm (competent runs);
  - P7 by arm (frozen `scratch/far_sweep_layers.py`, seed 2033, layer by in_peak, as T9);
  - whether P7 still separates extrapolators within the anytime arm.
- **Reading.**
  - AP1 supported with no AP3 cost: the anytime fix generalises across depth, width and graph shape.
  - AP1 not supported while THEORY D/E support anytime on the base architecture: the fix is architecture-specific, a
    scope limit on the slack account.

## Amendment 1 (2026-09-29 16:11 EDT, before any PROTOCOL9 run started and before any paired control finished)

* **Ceiling rule** (suggested by the theory session). With 3–4 layers or more width, the delayed query may resolve more
  than one hop, or one latent step may advance more than one hop. Distance 8 could then be in range for both arms, and
  the pairs would tie at the ceiling.
  - Per family: if the competent controls' median d8 (K′ = 12) is ≥ 0.85, that family's beyond-depth measure becomes
    the mean accuracy at distances 10–12 with K′ = 16 (`length_gen` rows).
  - If that median is also ≥ 0.85, the family leaves AP1 and is reported as "controls never fail beyond depth".
  - The rule is decided from the controls alone. AP1's thresholds and minimum are unchanged. Implemented in
    `scratch/score_p9.py` (unit-checked on synthetic pairs).
* **C = 1 unchanged.** If a deeper readout adds more than one hop, C = 1 only masks nodes the model could already reach,
  so the pressure is weaker, never impossible. Noted as a caveat on a null in the deeper families.
* **Disclosure.** While checking the `length_gen` file format for this amendment, I saw the K′ = 6 and K′ = 12 accuracy
  rows of one T9 run, `t9l3_s508` (fetched from vast3). That run is not in PROTOCOL9's pairs. Its values (near 5 ≈ 0.7,
  far 7–12 ≈ 0.5) did not inform the rule. The rule is Theory's proposal, adopted as stated.

## Amendment 2 (2026-09-29 22:22 EDT, before any PROTOCOL9 run started; placement only)

* At the user's request ("spin up more GPUs … 4 GPUs for 1 h cost the same as 1 for 4 h"), all 32 runs move from
  the local cards to Vast hosts, one queue per host:
  - the 24 bf16 runs on 8 hosts, `scratch/p9_vb1..8.txt` (3 runs each, families interleaved);
  - the 8 fork runs (fp32) on 4 hosts, `scratch/p9_vf1..4.txt` (2 each).
* Same frozen code (`scratch/p9_frozen/`), the same worker (`scratch/queue_worker3.sh`) and the same evaluation
  battery. Deployed with `scratch/vdeploy_toy.sh <offer> <label> <queue> <tag> scratch/queue_worker3.sh`, and fetched
  with `scratch/vfetch_toy.sh <host> <port> p9`.
* All 32 are marked claimed in the local `scratch/p9_bf16.txt.claimed` and `p9_fork.txt.claimed`, so the chained
  local PROTOCOL9 workers end at once. The local cards keep PROTOCOL7 T10.
* Controls (T9 on Vast, T10 local) and anytime runs now run on mixed hardware in both arms. Rules, seeds, measures and
  the cut-off are unchanged.

## Amendment 3 (2026-09-29 22:35 EDT, instrument fix; no PROTOCOL9 run has trained)

* **What happened.** All 32 first attempts (12 Vast hosts, started ~22:23–22:35) failed at import before any training
  step. The frozen `scratch/p9_frozen/train_arch.py` puts `scratch/` on `sys.path` (in `theory/` it put the repo
  root), so `import latent` failed. The pre-launch CPU smoke test had `PYTHONPATH=.` set by hand, which hid this.
  The workers logged "no model" for every run and ended.
* **Fix.** `scratch/queue_worker3.sh` now exports `PYTHONPATH` as the repo root. The frozen training code is unchanged
  (same sha256). The import was verified in a clean copy without `PYTHONPATH`: it fails without the export and works
  with it.
* **Handling.** These attempts are void: an instrument failure, not stage-1 failures. They produced no model and no
  outcome. On each host the failed attempt's files (`runs/p9at_*`, their logs and the queue claim file) are deleted
  and the same queue is restarted with the fixed worker. Nothing else changes.

## Note (2026-09-29 23:31 EDT, not a rule change): north-star conditions

PROTOCOL9 bears on **C2** (more thinking helps beyond training, in every competent seed, across architectures) and
**C6** (seed-to-seed variance of the outcome). It uses the toy model only, so C3 does not apply. See
`docs/NORTH_STAR.md`.

## Amendment 4 (2026-09-29 23:49 EDT, instrument fix for the fork family)

* The same `length_gen` crash (PROTOCOL7 amendment 9) hit PROTOCOL9's fork runs on hosts vf1–4. The fix is in
  `latent/length_gen.py`: the probe is built under the default shape and is unchanged for default-shape runs. It is
  copied to vf1–4. `length_gen` is re-run for fork runs already evaluated, and later runs use the fixed code.
* **Disclosure.** Diagnosing the crash on vf4 showed `p9at_fork_s563`'s in-distribution evaluation lines
  (competence and budget sweep by depth, K′ ≤ 10). No d7/d8 value (the AP1/AP2 measures) was seen; none existed.

## Amendment 5 (2026-09-30 02:42 EDT, host outage; decided before any of the affected runs' outcomes exist)

* Vast host vb7 (53453343) went offline between 02:31 and 02:41 with its three runs unfetched (`p9at_l3_s502`,
  `p9at_w512_s544`, `p9at_l4_s527`).
* **Rule, fixed now:**
  - If vb7 is back by 03:45, it is drained normally.
  - If not, the three runs are re-queued unchanged on a new host (`scratch/p9_vb7b.txt`), and the re-runs are the ones
    scored.
  - If vb7 returns later, its results are fetched for the record but not used, so there is no post-hoc choice between
    the two copies.

## Amendment 6 (2026-09-30 03:06 EDT, extension; before any PROTOCOL9 outcome was scored or read)

* **Why.** The local cards run dry at ~03:30, and PROTOCOL7 trained 16 control seeds per family while PROTOCOL9 paired
  only 8. So the other 8 seeds per family get the anytime arm too, at no cost, doubling the pairs.
* **Runs.** 32 more:
  - `p9at_l3_s508..515`, `p9at_l4_s528..535`, `p9at_w512_s548..555`, each paired with the same-seed T9 control;
  - `p9at_fork_s568..575`, paired with T10.
  - Queues `scratch/p9x_bf16.txt` (RTX 3070 / 4060, bf16) and `scratch/p9x_fork.txt` (RTX 2070 first, fp32). The
    code and worker are the same as for the primary runs.
* **Analysis, fixed now:**
  - AP1–AP3 on the original 32 pairs stay the primary, registered result.
  - The same rules applied to all 64 pairs (`scratch/score_p9.py --ext`, results `results/p9_pooled64.json`) are
    reported beside them as a pre-specified pooled replication.
  - The extension's 32 pairs alone are descriptive.
  - Cut-off unchanged (Sun 2026-10-04 00:00).
* **Blinding.** While adding the `--ext` flag, the scorer was run once to check it; only the count of scorable pairs
  was printed (10 of 64), no outcome, and the output file was deleted.

## Amendment 7 (2026-09-30 03:50 EDT, placement; no extension outcome read)

* On the local 8 GB RTX 3070, the extension's anytime runs with 4 layers and with width 512 ran out of CUDA memory:
  `p9at_l4_s528` at step 0, `p9at_w512_s548` at step 400 (`torch.OutOfMemoryError` in logs/*.err). Their guard
  logged "exited without stage-1 advance", not a stage-1 failure. These attempts are **void** (instrument, not
  outcome); the partial run dirs were renamed `runs/_void_oom_*`.
* **Placement.** The extension's 4-layer (8), width-512 (8) and fork (8, fp32; the 2070 is also 8 GB) runs move to 8
  Vast hosts, 3 runs each (`scratch/p9x_v1..8.txt`), where the primary runs of these families already ran fine. The
  local RTX 3070 and 4060 keep the 8 three-layer runs. Everything else is unchanged.
