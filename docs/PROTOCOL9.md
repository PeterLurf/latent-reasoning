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
