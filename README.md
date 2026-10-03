# Latent reasoning — can a language model think usefully without writing its steps?

The project's goal (set 2026-09-29) is one pretrained language model whose latent "thoughts" meet seven conditions:
they do the work, help beyond the trained length, compete with text chain-of-thought, need no step labels, decide when
to think and write, stay stable, and are faithfully readable. Definitions and current status: `docs/NORTH_STAR.md`.

**Where it stands (2026-10-03).** On S5 permutation tracking a pretrained GPT-2 carries the state in per-step latents,
snapped to a re-embedded text checkpoint every k steps: exact to 2,048 steps from training at 128 (certified at every
length for k = 1), the latents do the work (removal → floor), at 0.3–0.6× the compute of writing the state, in every
seed, with no hand-built state (§92, §97). Its step labels can come from a model trained on final answers only (§96).
Next: the same recipe on published benchmarks (multiplication, §98), then training from final answers inside the LM.

## Where to look

* `docs/NORTH_STAR.md` — goal, one status line per condition, design, open questions.
* `docs/CURRENT_TASKS.md` — live handoff: what is running, where, how to check or stop it.
* `DO_NOT_MOVE_WHILE_RUNNING.md` — paths that running jobs depend on; read before moving anything.
* `docs/THEORY.md` — every registration (§1 onward) with predictions, amendments and results.
* `docs/PROTOCOL*.md` — pre-registrations and scorecards (append-only); index in `docs/README.md`.
* `report/findings.md` — plain-language write-up; `bench/LEADERBOARD.md` — our models vs published numbers.

## Code map

| Directory | What |
|---|---|
| `latent/` | Parts I–VI: 2-layer toy decoder with Coconut-style thoughts on graph reachability (train, probes, interventions, scorers) |
| `prosqa/` | Official Coconut GPT-2 on ProsQA: patches, analyses, the shortcut-free set, F1-S label channel |
| `theory/` | Theory session's analyses, toy cells, automaton tables, scorers (`score_s*.py`) and `results/` |
| `c1/` | GPT-2 latent automata (S3–S5, A5): exact-marginal training, certificates, paraphrase ops (§67–§96) |
| `boxes/` | Box/permutation tracking: written-state restarts (§64–§71), latent restarts (§86/§92), fade (§89/§94), general recipe (§97), multiplication (§98) |
| `gsmlen/`, `s68/` | GSM8K step-count split (§78/§84); §68 throughput measurements |
| `bench/` | Shared benchmark harness: tasks, adapters, results, published numbers, `LEADERBOARD.md` |
| `bm/` | Boundary-memory proposal, Phase A (§60, failed its gate) |
| `ext/` | Third-party released checkpoints (ICoT-SI) used as baselines |
| `scratch/` | Queues, deploy/fetch/drain scripts for Vast and local cards, one-off analyses; `pods.txt` is the host ledger |
| `results/`, `runs/`, `logs/` | Outputs, run directories and logs (large; not edited by hand) |

## History in one paragraph per phase

* **Parts I–III (toy graph search).** Branch edits on a shared latent weaken with width; probe directions are causally
  inert (readable ≠ usable); recurrent models run a clocked breadth-first sweep that sometimes continues past the
  trained depth.
* **Parts IV–VI.** Neither simple account of extrapolation survived; whether a sweep stays open is fixed late in
  training and is predicted inside the budget (P7, AUC 0.993); restarts from a written state fix the horizon.
* **Pretrained GPT-2 (Parts VII–VIII).** Coconut's thoughts on ProsQA carry a one-pass answer; a proximity shortcut
  explains much of ProsQA accuracy; GSM8K latent methods reproduced, with latents carrying some work.
* **Latent automata and restarts (10-01 →).** Discrete latents learn from final answers once GPT-2 features are
  standardised; length is solved by restarts with a cut history; per-step targets are necessary; latent restarts and
  the general text-snap recipe give the S5 result above.

Full detail per phase: `report/findings.md`, the PROTOCOL scorecards, `docs/archive/RESEARCH_TIMELINE.md`, and the
older per-module tables in this file's git history.

## Hardware

Local RTX 4060, RTX 3070 (bf16) and RTX 2070 (fp32 only), 8 GB each; cloud Vast.ai (mostly RTX 3090) and earlier RunPod
hosts, logged in `scratch/pods.txt`.
