#!/usr/bin/env bash
# §60b job on a Vast host: bash scratch/bm_vast_job.sh <tag> <device> <queue file>. Queue lines: "<run> <mode> <lam> <seed> <steps> <bs>"
# mode pilot -> eval on dev only; mode panel -> full eval. Writes logs/queue_<tag>.out ending in QUEUE_DONE.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs bm/runs
grep -v '^#' "$q" | grep -v '^$' | while read -r r mode lam seed steps bs; do
  echo "$(date +%T) start $r" >> $log
  python -u bm/run_b.py train --pool bm/data_pool_b --out bm/runs/$r --lam $lam --seed $seed --steps $steps --bs $bs --device $dev > logs/$r.out 2>&1
  echo "$(date +%T) trained $r rc $?" >> $log
  if [ "$mode" = pilot ]; then python -u bm/run_b.py eval --pool bm/data_pool_b --run bm/runs/$r --sets dev --device $dev >> logs/$r.out 2>&1
  else python -u bm/run_b.py eval --pool bm/data_pool_b --run bm/runs/$r --device $dev >> logs/$r.out 2>&1; fi
  echo "$(date +%T) evaluated $r rc $?" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
