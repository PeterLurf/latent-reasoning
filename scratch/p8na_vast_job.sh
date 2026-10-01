#!/usr/bin/env bash
# PROTOCOL8 amendment 3 on a Vast host: bash scratch/p8na_vast_job.sh <tag> <device> <queue file>; queue lines "<job>".
# Checkpoints were uploaded with the code (prosqa/ckpt/p8c_<job>_checkpoint_{48,30}). Outputs prosqa/res_p8c_na/<job>.json.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs prosqa/res_p8c_na
grep -v '^#' "$q" | grep -v '^$' | while read -r j; do
  echo "$(date +%T) start $j" >> $log
  case $j in
    cot_*) python -u prosqa/eval_p8.py --ckpt prosqa/ckpt/p8c_${j}_checkpoint_30 --data prosqa/p8data/p8_na_test.json --mode cot --n_per_m 150 --out prosqa/res_p8c_na/$j.json --device $dev > logs/p8na_$j.log 2>&1 ;;
    *) python -u prosqa/eval_na_p8.py --ckpt prosqa/ckpt/p8c_${j}_checkpoint_48 --out prosqa/res_p8c_na/$j.json --device $dev > logs/p8na_$j.log 2>&1 ;;
  esac
  echo "$(date +%T) evaluated $j rc $?" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
