#!/usr/bin/env bash
# PROTOCOL8-NA on a Vast host, jobs IN PARALLEL on one GPU (each eval is batch-1, ~3.6 GB; user flag: A4000 under-used).
#   bash scratch/p8na_vast_par.sh <tag> <device> <queue file>; queue lines "<job>"; logs/queue_<tag>.out ends in QUEUE_DONE.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs prosqa/res_p8c_na
pip install -q tiktoken > logs/pip_na.log 2>&1 || echo "$(date +%T) PIP_FAILED rc 9" >> $log
run() {
  j=$1; echo "$(date +%T) start $j" >> $log
  case $j in
    cot_*) python -u prosqa/eval_p8.py --ckpt prosqa/ckpt/p8c_${j}_checkpoint_30 --data prosqa/p8data/p8_na_test.json --mode cot --n_per_m 150 --out prosqa/res_p8c_na/$j.json --device $dev > logs/p8na_$j.log 2>&1 ;;
    *) python -u prosqa/eval_na_p8.py --ckpt prosqa/ckpt/p8c_${j}_checkpoint_48 --out prosqa/res_p8c_na/$j.json --device $dev > logs/p8na_$j.log 2>&1 ;;
  esac
  rc=$?; [ -s prosqa/res_p8c_na/$j.json ] || rc=7
  echo "$(date +%T) evaluated $j rc $rc" >> $log
}
for j in $(grep -v '^#' "$q" | grep -v '^$'); do run $j & done
wait
echo "$(date +%T) QUEUE_DONE" >> $log
