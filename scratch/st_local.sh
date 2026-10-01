#!/usr/bin/env bash
# §64 runs of st queue 9 (seed 10) on a local card, both in parallel: bash scratch/st_local.sh <device>
cd "$(dirname "$0")/.."
for g in A5 S3; do lg=$(echo $g | tr 'A-Z' 'a-z')
  ( python -u theory/st_model.py --group $g --seed 10 --device $1 --out theory/runs/st/${lg}_s10 > logs/tj_${lg}_s10.log 2>&1
    echo "$(date +%T) st ${lg}_s10 rc $? ($1)" >> logs/st_local.log ) &
done
wait; echo "$(date +%T) WORKER_DONE $1" >> logs/st_local.log
