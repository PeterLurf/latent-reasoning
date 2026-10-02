#!/usr/bin/env bash
# THEORY §78 one-host throughput pilot (local, one card; fixes epochs <= 6 and batch only; no test predictions).
# Each arm 400 optimiser steps at bs 16 x grad_acc 2 (bs 32 OOMs an 8 GB card: fp32 logits) bf16 (+ TC fp32 for the bf16 parity check), short-valid eval at the end.
cd "$(dirname "$0")/.."
d=${1:-cuda:0}; touch scratch/.hold76_$d
for x in "tc " "tc --fp32" "ws " "wspad "; do set -- $x; tag=$1${2:+_fp32}
  python gsmlen/train78.py train --arm $1 $2 --max_steps 400 --bs 16 --grad_acc 2 --seed 1 --device $d --out gsmlen/pilot/$tag > logs/p78_pilot_$tag.log 2>&1
  echo "$(date +%T) p78 pilot $tag rc $? ($d)" >> logs/c1_local.log
done
rm -f scratch/.hold76_$d
