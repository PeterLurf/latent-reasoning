#!/usr/bin/env bash
# Shared job queue for several local GPUs (PROTOCOL5 addendum A pilot):
#   bash scratch/queue_worker2.sh <gpu-tag> <device> <queue-file> [override args...]
# Override args are appended after the queue line's own args (argparse: the last value wins), e.g. --amp fp32.
# Each line of <queue-file> is "<run> <extra train args...>". A worker claims the next unclaimed line (mkdir lock),
# trains it with the base recipe under the stage-1 guard, then runs the addendum-D battery (scratch/eval_battery.sh).
# Claimed lines are recorded in <queue-file>.claimed; the worker exits when every line is claimed.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; shift 3; over="$*"
log=logs/queue_$tag.out
BASE="--mode coconut --bs 512 --steps 30000 --warmup 300 --eval_every 1000 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2"
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
  echo "$(date +%T) start $r ($extra $over)" >> $log
  python scratch/stage1_guard.py runs/$r 0 3 --wait -- -m latent.train $BASE --device $dev --resume --out runs/$r $extra $over >> logs/$r.out 2>> logs/$r.err
  echo "$(date +%T) trained $r rc $?" >> $log
  bash scratch/eval_battery.sh $r $dev >> $log 2>&1
  echo "$(date +%T) evaluated $r" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
