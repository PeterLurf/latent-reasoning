#!/usr/bin/env bash
# THEORY §67 amendment 67-1: bash scratch/c1_671.sh <arm> <device> <extra args...>; seeds 99 1 2 sequentially.
cd "$(dirname "$0")/.."
arm=$1; dev=$2; shift 2
for s in 99 1 2; do
  [ -f c1/runs/a671_${arm}_s$s/eval.json ] && continue
  python -u -m c1.train --arm $arm --seed $s --updates 2000 --bs 512 "$@" --device $dev --out c1/runs/a671_${arm}_s$s > logs/c1_a671_${arm}_s$s.log 2>&1
  echo "$(date +%T) a671_${arm}_s$s $(grep -o 'EVAL.*' logs/c1_a671_${arm}_s$s.log | cut -c1-140)" >> logs/c1_a671.log
done
echo "$(date +%T) DONE $arm" >> logs/c1_a671.log
