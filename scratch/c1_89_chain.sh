#!/usr/bin/env bash
# §89 placement: on each card, start a §89 worker once that card's §88 worker has finished.
cd "$(dirname "$0")/.."
for d in cuda:0 cuda:1 cuda:2; do
  ( until grep -q "WORKER_DONE $d scratch/c1_q_88.txt" logs/c1_local.log; do sleep 300; done
    bash scratch/py_local_worker.sh $d scratch/c1_q_89.txt ) &
done
wait
