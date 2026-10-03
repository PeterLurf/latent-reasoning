#!/usr/bin/env bash
# After §88 on each card: one worker on scratch/c1_q_after88.txt (§86 re-pilot first, claimed by the first free card;
# then §89). The §86 panel is added by hand only if the re-pilot's cumulative at 512 >= 0.95 (Theory 86-4).
cd "$(dirname "$0")/.."
for d in cuda:0 cuda:1 cuda:2; do
  ( until grep -q "WORKER_DONE $d scratch/c1_q_88.txt" logs/c1_local.log; do sleep 120; done
    bash scratch/py_local_worker.sh $d scratch/c1_q_after88.txt ) &
done
wait
