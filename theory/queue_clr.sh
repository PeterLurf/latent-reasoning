#!/usr/bin/env bash
# CLR pilot worker (THEORY §45): bash theory/queue_clr.sh <tag> <device> <queue-file>; lines "<run> <train_clr.py args>".
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3
log=logs/queue_$tag.out
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
  python theory/train_clr.py $extra --device $dev --out theory/runs/$r >> logs/$r.out 2>> logs/$r.err
  echo "$(date +%T) trained $r rc $?" >> $log
  [ -f theory/runs/$r/model.pt ] && python theory/eval_arch.py theory/runs/$r --device $dev >> $log 2>> logs/eval_$r.err
  echo "$(date +%T) evaluated $r" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
