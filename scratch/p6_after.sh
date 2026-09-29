#!/usr/bin/env bash
# PROTOCOL6: measure each finished run (end activity + strict clock; checkpoint trajectory) once its battery is done.
#   bash scratch/p6_after.sh <device>      (detached; exits when every queued run is measured)
cd "$(dirname "$0")/.."
dev=$1
runs=$(grep -hv '^#' scratch/p6_queue.txt scratch/p6_queue_2070.txt | awk '{print $1}')
while :; do
  left=0
  for r in $runs; do
    if [ -s results/budget_sweep_$r.json ]; then
      python -m latent.protocol6 measure runs/$r --device $dev >> logs/p6_after.log 2>&1
      python -m latent.protocol6 traj runs/$r --device $dev >> logs/p6_after.log 2>&1
    else
      left=$((left + 1))
    fi
  done
  [ $left -eq 0 ] && break
  sleep 300
done
python -m latent.protocol6 measure $(for r in $runs; do echo runs/$r; done) --device $dev >> logs/p6_after.log 2>&1
echo "$(date +%T) P6_AFTER_DONE" >> logs/p6_after.log
