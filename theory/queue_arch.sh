#!/usr/bin/env bash
# Theory-stream job queue on a cloud host:  bash theory/queue_arch.sh <tag> <device> <queue-file>
# Each queue line: "<run> <extra train args...>". Trains with the P30 base recipe under the stage-1 guard via
# theory/train_arch.py, then evaluates with theory/eval_arch.py. Claimed lines go to <queue>.claimed.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3
log=logs/queue_$tag.out
BASE="--mode coconut --bs 512 --steps 30000 --warmup 300 --eval_every 1000 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2 --no_step_emb --K 6 --lr 5e-4 --amp bf16 --save_every 1000"
mkdir -p logs theory/runs
claim() {
  until mkdir "$q.lock" 2>/dev/null; do sleep 1; done
  touch "$q.claimed"
  line=$(grep -v '^#' "$q" | grep -v '^$' | while read -r r rest; do grep -qx "$r" "$q.claimed" || { echo "$r $rest"; break; }; done)
  [ -n "$line" ] && echo "${line%% *}" >> "$q.claimed"
  rmdir "$q.lock"
  echo "$line"
}
while :; do
  line=$(claim)
  [ -z "$line" ] && break
  read -r r extra <<<"$line"
  echo "$(date +%T) start $r ($extra)" >> $log
  python scratch/stage1_guard.py theory/runs/$r 0 3 --wait -- theory/train_arch.py $BASE --device $dev --resume --out theory/runs/$r $extra >> logs/$r.out 2>> logs/$r.err
  echo "$(date +%T) trained $r rc $?" >> $log
  [ -f theory/runs/$r/model.pt ] && python theory/eval_arch.py theory/runs/$r --device $dev >> $log 2>> logs/eval_$r.err
  echo "$(date +%T) evaluated $r" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
