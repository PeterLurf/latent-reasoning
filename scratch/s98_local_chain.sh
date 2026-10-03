#!/usr/bin/env bash
# §98 on the local bf16 cards (user 19:15): after this card's §89 worker finishes (WORKER_DONE <dev> scratch/c1_q_next.txt),
# run scratch/c1_q_98coco.txt with py_local_worker2 (lock dirs shared, so the two cards split the queue).
#   bash scratch/s98_local_chain.sh cuda:0
cd "$(dirname "$0")/.."
d=$1; n0=$(grep -c "WORKER_DONE $d scratch/c1_q_next.txt" logs/c1_local.log)
until [ "$(grep -c "WORKER_DONE $d scratch/c1_q_next.txt" logs/c1_local.log)" -gt "$n0" ]; do sleep 60; done
echo "$(date +%T) $d: §89 worker done; starting §98 COCO reruns (scratch/c1_q_98coco.txt)" >> logs/c1_local.log
bash scratch/py_local_worker2.sh $d scratch/c1_q_98coco.txt
