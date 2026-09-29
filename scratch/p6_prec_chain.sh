#!/usr/bin/env bash
# PROTOCOL6 addendum F: start each card's precision queue when that card's branch work is done; measure every run.
#   bash scratch/p6_prec_chain.sh      (detached; logs/p6_prec_chain.log)
cd "$(dirname "$0")/.."
L=logs/p6_prec_chain.log
bash scratch/queue_worker2.sh prec_2070 cuda:2 scratch/p6_prec_2070.txt &
( until grep -q QUEUE_DONE logs/queue_br_4060b.out 2>/dev/null; do sleep 60; done
  echo "$(date +%T) 4060 free" >> $L; bash scratch/queue_worker2.sh prec_4060 cuda:0 scratch/p6_prec_4060.txt ) &
( until grep -q QUEUE_DONE logs/queue_br_3070b.out 2>/dev/null; do sleep 60; done
  echo "$(date +%T) 3070 free" >> $L; bash scratch/queue_worker2.sh prec_3070 cuda:1 scratch/p6_prec_3070.txt ) &
sleep 120
bash scratch/p6_after2.sh cuda:0 scratch/p6_prec_4060.txt scratch/p6_prec_3070.txt scratch/p6_prec_2070.txt &
wait
echo "$(date +%T) P6_PREC_DONE" >> $L
