#!/usr/bin/env bash
# Block until something worth acting on happens, print it, exit (used as a background waiter).
#   bash scratch/p6_wait.sh <pod-ssh-user|none>
cd "$(dirname "$0")/.."
U=$1
stagecks() { grep -oE "checkpoint_[0-9]*[05]( |$)" | tr -d ' ' | tr '\n' ' '; }
n0=$(cat logs/queue_p6_*.out 2>/dev/null | grep -cE "QUEUE_DONE|rc [1-9]")
ck0=""; [ "$U" != none ] && ck0=$(bash prosqa/status.sh $U 2>/dev/null | stagecks)
for i in $(seq 1 144); do
  sleep 300
  n=$(cat logs/queue_p6_*.out 2>/dev/null | grep -cE "QUEUE_DONE|rc [1-9]")
  if [ "$n" != "$n0" ]; then cat logs/queue_p6_*.out | grep -E "QUEUE_DONE|rc [1-9]" | tail -n $((n - n0)); exit 0; fi
  if [ "$U" != none ]; then
    s=$(bash prosqa/status.sh $U 2>/dev/null)
    [ -z "$s" ] && { sleep 30; s=$(bash prosqa/status.sh $U 2>/dev/null); [ -z "$s" ] && { echo "POD UNREACHABLE"; exit 0; }; }
    echo "$s" | grep -qE "procs 0|Error|TRAIN_EXIT|ANALYSIS_DONE|NO_FINAL" && { echo "POD ATTENTION"; echo "$s"; exit 0; }
    ck=$(echo "$s" | stagecks)
    [ "$ck" != "$ck0" ] && { echo "POD NEW STAGE CKPT"; echo "$s"; exit 0; }
  fi
done
echo "WAIT_TIMEOUT_12H"
