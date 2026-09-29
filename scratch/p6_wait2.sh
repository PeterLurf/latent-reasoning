#!/usr/bin/env bash
# Block until something worth acting on happens (any local queue done or failed, the overnight chain done or failed,
# ProsQA pod trouble / stage checkpoint / analysis done), print it, exit.   bash scratch/p6_wait2.sh <pod-ssh-user|none>
cd "$(dirname "$0")/.."
U=$1
ev() { cat logs/queue_*.out logs/p6_night.log 2>/dev/null | grep -aE "QUEUE_DONE|rc [1-9]|P6_NIGHT_DONE|BRANCH_BUILD_FAILED|FAIL "; }
stagecks() { grep -oE "checkpoint_[0-9]*[05]( |$)" | tr -d ' ' | tr '\n' ' '; }
n0=$(ev | wc -l)
ck0=""; [ "$U" != none ] && ck0=$(bash prosqa/status.sh $U 2>/dev/null | stagecks)
for i in $(seq 1 ${WAIT_ROUNDS:-144}); do
  sleep 300
  n=$(ev | wc -l)
  if [ "$n" -gt "$n0" ]; then ev | tail -n $((n - n0)); exit 0; fi
  if [ "$U" != none ]; then
    s=$(bash prosqa/status.sh $U 2>/dev/null)
    [ -z "$s" ] && { sleep 30; s=$(bash prosqa/status.sh $U 2>/dev/null); [ -z "$s" ] && { echo "POD UNREACHABLE"; exit 0; }; }
    echo "$s" | grep -qE "procs 0|Error|TRAIN_EXIT|ANALYSIS_DONE|NO_FINAL" && { echo "POD ATTENTION"; echo "$s"; exit 0; }
    ck=$(echo "$s" | stagecks)
    [ "$ck" != "$ck0" ] && { echo "POD NEW STAGE CKPT"; echo "$s"; exit 0; }
  fi
done
echo "WAIT_TIMEOUT_12H"
