#!/usr/bin/env bash
# bench batch worker (local): bash scratch/bench_worker.sh <device> <queue file>. Lines "<name> <task> <spec> [extra args]";
# claims by lock dirs bench/results/locks/<name>; log logs/bench.log, per job logs/bench_<name>.log.
cd "$(dirname "$0")/.."
mkdir -p bench/results/locks
grep -v '^#' "$2" | while read -r n t sp rest; do
  mkdir bench/results/locks/$n 2>/dev/null || continue
  python -u -m bench.run --task $t --model "$sp" --device $1 $rest > logs/bench_$n.log 2>&1
  echo "$(date +%T) $n rc $? ($1)" >> logs/bench.log
done
echo "$(date +%T) WORKER_DONE $1 $2" >> logs/bench.log
