#!/usr/bin/env bash
# Local monitor (2026-09-29): wake on a new training failure, a new QUEUE_DONE, or a running run whose log.jsonl has
# not advanced for 75 min (stall / memory thrash). Prints the event and exits.
cd "$(dirname "$0")/.."
ev() { cat logs/queue_*.out 2>/dev/null | grep -aE "QUEUE_DONE|rc [1-9]" | sort; }
ev > /tmp/mon_old.txt
for i in $(seq 1 300); do
  sleep 600
  ev > /tmp/mon_new.txt
  new=$(comm -13 /tmp/mon_old.txt /tmp/mon_new.txt)
  [ -n "$new" ] && { echo "EVENT $new"; exit 0; }
  for f in logs/queue_r_*.out logs/queue_t9_*.out logs/queue_t10_*.out logs/queue_p9*.out; do
    [ -f "$f" ] || continue
    last=$(tail -1 "$f")
    case "$last" in *" start "*) r=$(echo "$last" | awk '{print $3}');;
                    *) continue;; esac
    lj="runs/$r/log.jsonl"
    [ -f "$lj" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$lj") ))
    [ $age -gt 4500 ] && { echo "EVENT STALL $r log.jsonl idle ${age}s ($f)"; exit 0; }
  done
done
echo MONITOR_TIMEOUT
