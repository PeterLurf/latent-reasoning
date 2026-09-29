#!/usr/bin/env bash
# PROTOCOL6 addendum H: 4060 now; 3070 when its addendum-F queue is done; measure every run.
cd "$(dirname "$0")/.."
bash scratch/queue_worker2.sh stall_4060 cuda:0 scratch/p6_stall_4060.txt &
( until grep -q QUEUE_DONE logs/queue_prec_3070b.out 2>/dev/null; do sleep 60; done
  bash scratch/queue_worker2.sh stall_3070 cuda:1 scratch/p6_stall_3070.txt ) &
sleep 120
bash scratch/p6_after2.sh cuda:0 scratch/p6_stall_4060.txt scratch/p6_stall_3070.txt &
wait
echo "$(date +%T) P6_STALL_DONE" >> logs/p6_stall_chain.log
