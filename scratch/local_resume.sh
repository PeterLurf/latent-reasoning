#!/usr/bin/env bash
# Local job queue for one GPU (resume of the PROTOCOL4 local runs halted at 00:42 on 2026-09-25):
#   bash scratch/local_resume.sh <gpu-tag> <device> <item> ...   item = train:<run>:<extra args, comma-separated> | eval:<run>[:clock]
# Training uses the base recipe under the stage-1 guard (as scratch/chain.py did); eval uses scratch/eval_battery.sh.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; shift 2
log=logs/local_$tag.out
BASE="--mode coconut --bs 512 --steps 30000 --warmup 300 --eval_every 1000 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2"
for it in "$@"; do
  IFS=: read -r kind r extra <<<"$it"
  echo "$(date +%T) start $it" >> $log
  case $kind in
    train) python scratch/stage1_guard.py runs/$r 0 3 --wait -- -m latent.train $BASE --device $dev --resume --out runs/$r ${extra//,/ } >> logs/$r.out 2>> logs/$r.err ;;
    eval)  bash scratch/eval_battery.sh $r $dev $extra >> $log 2>&1 ;;
  esac
  echo "$(date +%T) exit $? $it" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
