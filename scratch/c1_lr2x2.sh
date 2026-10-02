#!/usr/bin/env bash
cd "$(dirname "$0")/.."
dev=$1
one() { arm=$1; lr=$2; s=$3; o=c1/runs/dx_${arm}_lr${lr}_s$s; [ -f $o/eval.json ] && return
  python -u -m c1.train --arm $arm --seed $s --updates 2000 --bs 512 --lr $lr --device $dev --out $o > logs/c1_dx_${arm}_lr${lr}_s$s.log 2>&1
  echo "$(date +%T) dx_${arm}_lr${lr}_s$s $(grep -o 'EVAL.*' logs/c1_dx_${arm}_lr${lr}_s$s.log | cut -c1-140)" >> logs/c1_dx.log; }
for s in 99 1 2; do one tab 1e-3 $s; one mlp 1e-2 $s; done; for s in 1 2; do one tab 1e-2 $s; done
echo "$(date +%T) DX_DONE" >> logs/c1_dx.log
