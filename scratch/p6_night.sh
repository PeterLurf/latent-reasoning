#!/usr/bin/env bash
# PROTOCOL6 overnight chain: wait for the addendum A/B queues and their measurements, build the addendum E branches,
# run them on all three GPUs (4060/3070: bf16 branches then fp32 ones; 2070: fp32 only) and measure them.
#   bash scratch/p6_night.sh        (detached; logs/p6_night.log)
cd "$(dirname "$0")/.."
L=logs/p6_night.log
echo "$(date +%T) waiting for PROTOCOL6 queues" >> $L
until [ "$(cat logs/queue_p6_*.out 2>/dev/null | grep -c QUEUE_DONE)" -ge 3 ] && grep -q P6_AFTER_DONE logs/p6_after.log 2>/dev/null; do sleep 120; done
echo "$(date +%T) queues and measurements done; building branches" >> $L
PYTHONPATH=. python scratch/p6_branch.py >> $L 2>&1 || { echo "$(date +%T) BRANCH_BUILD_FAILED" >> $L; exit 1; }
B=scratch/p6_branch_queue.txt; F=scratch/p6_branch_queue_2070.txt
( bash scratch/queue_worker2.sh br_4060 cuda:0 $B; bash scratch/queue_worker2.sh br_4060b cuda:0 $F ) &
sleep 20
( bash scratch/queue_worker2.sh br_3070 cuda:1 $B; bash scratch/queue_worker2.sh br_3070b cuda:1 $F ) &
bash scratch/queue_worker2.sh br_2070 cuda:2 $F &
sleep 60
bash scratch/p6_after2.sh cuda:0 $B $F &
wait
echo "$(date +%T) P6_NIGHT_DONE" >> $L
