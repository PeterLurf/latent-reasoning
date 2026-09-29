# Read before reorganising this folder

Experiments are running **right now** (until about Tue 2026-09-29 12:00 EDT). Detached background processes on this PC and
on Vast.ai cloud machines read and write files in this folder **by path**. Moving, renaming or editing the items below
while they run will silently break experiments (bash reads running scripts line by line; queue workers re-read their queue
files; new training runs import `latent/` fresh).

## Do not move, rename, delete or edit (until the runs finish)

| Path | Why |
|---|---|
| `latent/` | the training/eval package; every new run imports it (`python -m latent.train`, `latent.protocol6`, …) |
| `scratch/` | running scripts (`queue_worker2.sh`, `stage1_guard.py`, `eval_battery.sh`, `chain_after.sh`, `p6_after2.sh`, `p6_wait3.sh`, `vfetch_toy.sh`, `score_*.py`, `far_sweep_frozen_I.py`) and live queue files (`p6_m_*.txt`, `p6_o_*.txt` + their `.claimed` / `.lock`) |
| `runs/` | models being written right now (`m_s*`, `o_s*`, `o_dev6_s*`) and every earlier model the scorers read |
| `results/` | evaluation outputs written by the running jobs and read by the scorers |
| `logs/` | queue logs the chains watch for `QUEUE_DONE` (`logs/queue_*.out`) |
| `prosqa/` | ProsQA scripts used to check/fetch/score the cloud runs (`status.sh`, `vfetch.sh`, `score.py`, `analyze.py`, `gpt2c.py`), results `res_*`, checkpoints `ckpt/` (6.5 GB) |
| `docs/CURRENT_TASKS.md`, `docs/PROTOCOL6.md`, `docs/PROTOCOL6_RESULTS.md`, `docs/RESEARCH_LOG.md` | live handoff and pre-registration files; Claude sessions and memory refer to these exact paths |

## Safe to reorganise now

* `report/` and the other files in `docs/` (keep the four named above in place), `README.md` (keep its link to
  `docs/CURRENT_TASKS.md`).
* Anything else: please wait until `docs/CURRENT_TASKS.md` says nothing is running.

## Also

* Pre-registration records (`docs/PROTOCOL*.md`) must not be rewritten — only appended to with dated amendments.
* A backup of code, docs, results, logs and run configs/logs (not model weights) was made before any reorganisation:
  `D:\Latent_backup_2026-09-28_1840.tgz` (outside this folder).
* If a move is unavoidable after the runs, update every path in `docs/CURRENT_TASKS.md` and the scripts that reference it.
