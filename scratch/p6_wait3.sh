#!/usr/bin/env bash
# Block until something worth acting on happens on the local queues or any of several cloud machines; print, exit.
#   bash scratch/p6_wait3.sh <target> [<target> ...]     (targets as prosqa/status.sh takes them; "none" = local only)
cd "$(dirname "$0")/.."
ev() { cat logs/queue_*.out logs/p6_night.log 2>/dev/null | grep -aE "QUEUE_DONE|rc [1-9]|P6_NIGHT_DONE|BRANCH_BUILD_FAILED|FAIL "; }
stagecks() { grep -oE "checkpoint_[0-9]*[05]( |$)" | tr -d ' ' | tr '\n' ' '; }
n0=$(ev | wc -l); ev | sort > /tmp/p6w_old.txt
declare -A ck0
for U in "$@"; do [ "$U" != none ] && ck0[$U]=$(bash prosqa/status.sh $U 2>/dev/null | stagecks); done
for i in $(seq 1 ${WAIT_ROUNDS:-144}); do
  sleep 300
  n=$(ev | wc -l)
  if [ "$n" -gt "$n0" ]; then ev | sort > /tmp/p6w_new.txt; comm -13 /tmp/p6w_old.txt /tmp/p6w_new.txt; exit 0; fi
  for U in "$@"; do
    [ "$U" = none ] && continue
    s=$(bash prosqa/status.sh $U 2>/dev/null)
    [ -z "$s" ] && { sleep 30; s=$(bash prosqa/status.sh $U 2>/dev/null); [ -z "$s" ] && { echo "UNREACHABLE $U"; exit 0; }; }
    echo "$s" | grep -qE "procs 0|Error|TRAIN_EXIT|ANALYSIS_DONE|NO_FINAL|MASK_TEST_FAILED" && { echo "ATTENTION $U"; echo "$s"; exit 0; }
    [ -n "$NOCKPT" ] && continue
    ck=$(echo "$s" | stagecks)
    [ "$ck" != "${ck0[$U]}" ] && { echo "NEW STAGE CKPT $U"; echo "$s"; exit 0; }
  done
done
echo "WAIT_TIMEOUT_12H"
