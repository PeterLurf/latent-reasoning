#!/usr/bin/env bash
# Keeps gpu_watch2 running overnight (Audit): every trip line is appended to logs/gpu_watch_trips.log, then the watcher
# restarts after 5 min (debounce, so a host in setup does not loop). Stop: touch scratch/.stop_gpu_watch_loop.
cd "$(dirname "$0")/.."
while [ ! -f scratch/.stop_gpu_watch_loop ]; do
  out=$(bash scratch/gpu_watch2.sh 2>&1 | tail -n 3)
  echo "$(date '+%F %T') TRIP: $out" >> logs/gpu_watch_trips.log
  sleep 300
done
