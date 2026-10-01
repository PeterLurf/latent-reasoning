# bench: reusable reasoning tests

One command runs any of our models on any of the standard latent-reasoning test sets, and writes a result file that
records what was run. A second command turns all result files plus the published numbers into `LEADERBOARD.md`.

```bash
python -m bench.run --list                                   # tasks
python -m bench.run --task prosqa_na --model "coconut:prosqa/ckpt/checkpoint_50?K=6" --device cuda:0
python -m bench.run --task prosqa_deep_na --model "f1s:prosqa/runs_ail/f1s_s3?P=16" --per_depth 100 --device cuda:0
python -m bench.run --task prosqa --model shortcut            # the no-reasoning rule (0.911)
python -m bench.report                                        # writes bench/LEADERBOARD.md
```

## Tasks

| Task | Data | Kind |
|---|---|---|
| `prosqa` | Coconut's ProsQA test (500), 3–6 hops | two options |
| `prosqa_na` | the same test with the proximity shortcut neutralised (THEORY §59) | two options |
| `prosqa_deep`, `prosqa_deep_na` | ProsQA generator extended deeper (PROTOCOL8), with and without the shortcut | two options |
| `gsm8k` | GSM8K test (1,319), downloaded on first use | generate a number |
| `mult4x4`, `mult5x5` | implicit-CoT multiplication tests (1,000 each), downloaded on first use | generate digits |

Add a task by writing a loader in `tasks.py` that returns items in the shared schema and registering it in `TASKS`.
Give it its no-reasoning baselines: every report prints them next to the model scores.

## Models

| Spec | What runs |
|---|---|
| `coconut:<ckpt>?K=6&mask=0` | Coconut GPT-2 with K continuous thoughts (`mask=1` for answer-masked models, e.g. PROTOCOL8 arms) |
| `nocot:<ckpt>` | answer scored straight after the question |
| `cot:<ckpt>?max_new=220` | greedy text chain-of-thought, then the answers scored |
| `f1s:<run_dir>?P=16` | per-entity label-channel model (THEORY §61), P passes |
| `gen:<ckpt>?max_new=256` | greedy generation for GSM8K / multiplication; answer after the last `####` |
| `shortcut`, `chance` | baselines |

Add a model by writing an adapter class in `models.py` with `predict(item)` and a branch in `load()`.

## Rules

- Results go only to `bench/results/`; nothing under `prosqa/`, `latent/` or `theory/` is modified.
- Each result file carries per-item outcomes, accuracy by depth, a 95% Wilson interval, mean compute (latent passes,
  generated tokens), and provenance: git commit, data hashes, checkpoint fingerprint and command line.
- A published number counts as a head-to-head comparison only when `same_items` is true in `literature.json`.
  Numbers start unverified (marked †) until Literature checks them against the paper.
- CPU runs default to 3 threads so local GPU queues stay fed; real runs go on GPUs via the experiments session.
