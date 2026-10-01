#!/usr/bin/env bash
# §64b seeds 9 and 10 (st2 queue 4) on a local card, in parallel: bash scratch/st2_local.sh <device>
cd "$(dirname "$0")/.."
for s in 9 10; do
  ( python -u theory/st_model.py --group A5 --seed $s --curriculum --device $1 --out theory/runs/st/a5c_s$s > logs/tj_a5c_s$s.log 2>&1
    echo "$(date +%T) st2 a5c_s$s rc $? ($1)" >> logs/st_local.log ) &
done
wait; echo "$(date +%T) WORKER_DONE st2 $1" >> logs/st_local.log
