# Read before reorganising this folder

**Refreshed 2026-10-03 19:55 EDT.** Jobs are running right now: the Vast fleets for §97 C6 and §98 (until about
2026-10-04 06:00), plus local GPU and CPU jobs (§89's last fade runs, §98 COCO reruns, §83, §94 probes). The live
list is in `docs/CURRENT_TASKS.md`. Detached processes on this PC and on Vast read and write files here **by path**.
Bash reads running scripts line by line, workers re-read queue files, and new runs import the modules fresh. So
moving, renaming or editing the items below silently breaks them.

## Do not move, rename, delete or edit (while `docs/CURRENT_TASKS.md` lists running jobs)

| Path | Why |
|---|---|
| `boxes/` (`core.py`, `s5.py`, `latent86.py`, `gen97.py`, `mult98.py`, `fade89.py`, `probe94.py`, `train.py`) | imported by every running and queued job; `boxes/runs*/` are their outputs |
| `theory/s83_restart.py`, `theory/results/` | §83 job on the 2070; probe outputs |
| `scratch/` | running scripts and queues. Workers: `py_local_worker2.sh`, `s83_local.sh`, `s98_local_chain.sh`, `s94_probes_new.sh`. Vast: `gen_vast_job.sh`, `vdeploy_toy.sh`, `drain_gen.sh`, `auto_drain_gen{2,3}.sh`, `vdestroy.sh`. Queues: `c1_q_next.txt`, `c1_q_98coco.txt`, `gen_q_*.txt`. Registry: `pods.txt`, `live_hosts.txt` |
| `c1/runs/locks/` | local queue claims (a lock dir also holds the dropped §89 runs) |
| `bench/data/` | §98 test files and `mult_train/` (gitignored), S5 data |
| `prosqa/ckpt/gpt2_pretrained.bin` | the GPT-2 weights every job loads and every deploy uploads |
| `ext/icotsi/` | ICoT-SI released checkpoints (gitignored) |
| `logs/` | queue and watcher logs the chains and drains watch |
| `docs/CURRENT_TASKS.md`, `docs/THEORY.md`, `docs/PROTOCOL*.md`, `docs/RESEARCH_LOG.md` | live handoff and registration files that sessions and memory refer to by path |

## Safe to reorganise

* Other files in `docs/` (Audit moved superseded notes to `docs/archive/` on 2026-10-03), `report/`, and anything not
  listed above. Check that no script or doc still points at a path before moving it.

## Also

* Pre-registration records (`docs/PROTOCOL*.md`, THEORY §§) are never rewritten, only appended to with dated notes.
* Add new modules or new flags whose default keeps today's behaviour. Don't change existing functions that running
  jobs or registered scorers use.
* The repo is LF (`core.autocrlf false`); the running bash scripts must stay LF.
