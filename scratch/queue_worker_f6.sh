#!/usr/bin/env bash
# THEORY §58 F6 runs (transplant), executed by the experiments session with a frozen copy of the theory
# code at commit 5d3d2b6 (scratch/f6_frozen/): trains into theory/runs/<run> under the stage-1 guard, then runs the
# frozen eval_arch.py (eval.json in the run dir). Same claim protocol as queue_worker2.sh.
#   bash scratch/queue_worker_t12.sh <tag> <device> <queue file>
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
tag=$1; dev=$2; q=$3
log=logs/queue_$tag.out
BASE="--Q 16 --gate 0.9 --mode coconut --bs 512 --steps 30000 --warmup 300 --eval_every 1000 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2 --no_step_emb --K 6 --lr 5e-4 --amp bf16 --save_every 1000"
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
  python scratch/stage1_guard.py theory/runs/$r 0 3 --wait -- scratch/f6_frozen/train_arch.py $BASE --device $dev --resume --out theory/runs/$r $extra >> logs/$r.out 2>> logs/$r.err
  echo "$(date +%T) trained $r rc $?" >> $log
  if [ -f theory/runs/$r/model.pt ]; then python scratch/f6_frozen/eval_arch.py theory/runs/$r --device $dev >> $log 2>> logs/eval_$r.err; else echo "no model for $r" >> $log; fi
  echo "$(date +%T) evaluated $r" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
