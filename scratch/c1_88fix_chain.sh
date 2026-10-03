#!/usr/bin/env bash
# §88 fix: rerun align k2 s3 on cuda:0 once its post-§88 worker has finished.
cd "$(dirname "$0")/.."
until grep -q "WORKER_DONE cuda:0 scratch/c1_q_after88.txt" logs/c1_local.log; do sleep 300; done
bash scratch/py_local_worker.sh cuda:0 scratch/c1_q_88fix.txt
