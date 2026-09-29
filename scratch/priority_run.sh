#!/usr/bin/env bash
# Run a high-priority job on a card between two runs of a live queue_worker2.sh queue, without editing the queue.
#   bash scratch/priority_run.sh <queue-file> <queue-log> <tag> -- <command...>
# Holds <queue-file>.lock (the worker's claim lock), so the worker finishes its current run + eval and then waits in
# claim(). Once the queue log's last line is "evaluated ...", this runs the command (logs/prio_<tag>.out), then
# releases the lock and the worker resumes with its next line. The trap releases the lock on any exit.
cd "$(dirname "$0")/.."
q=$1; qlog=$2; tag=$3; shift 3; [ "$1" = "--" ] && shift
log=logs/prio_$tag.out
until mkdir "$q.lock" 2>/dev/null; do sleep 1; done
trap 'rmdir "$q.lock" 2>/dev/null' EXIT
echo "$(date +%T) holding $q.lock; waiting for the current run of $qlog to finish" >> $log
until tail -n 1 "$qlog" | grep -q " evaluated "; do sleep 30; done
echo "$(date +%T) start: $*" >> $log
bash -c "$*" >> $log 2>&1
echo "$(date +%T) done rc $?" >> $log
echo "$(date +%T) PRIO_DONE $tag" >> $log
