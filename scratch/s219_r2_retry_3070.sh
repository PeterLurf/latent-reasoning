#!/bin/bash
# §219 round 2 retry on the 3070 (cuda:1) after queue r2d (Note 219-9): GATE-T2 / GATE-T1' label with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (a);
# if that fails, (b) the same with --full_bs 64 (descriptive LABELLER-FULL in micro-chunks of 64 inside each 128 batch; primary TF-GREEDY at bs 128);
# if (b) fails too -> stop and report (c). Then score -> fallback on failed windows -> T2 pool. Gate items: the 06:38 rerun (identical items to 04:44).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs/s219_3070_queue.out
until grep -q "3070 queue (r2d) done" $L 2>/dev/null; do sleep 60; done
run() { local tag=$1; shift; "$@" > logs/s219_q_$tag.log 2>&1; local rc=$?; echo "$(date '+%F %T') $tag rc $rc" >> $L; return $rc; }
noslots() { python -c "
import json; e = json.load(open('boxes/runs_fsq/s219_gate_t2_s1/eval_gate.json')); print(','.join(w for w, v in e['window_labeller'].items() if v is None))"; }
FB=""
run r2_label_a python -B scratch/s219_gate.py --round 2 --stage label --labeller primary --device cuda:1 || {
  FB="--full_bs 64"; run r2_label_b python -B scratch/s219_gate.py --round 2 --stage label --labeller primary --device cuda:1 $FB || {
    echo "$(date '+%F %T') r2 label failed under (a) and (b) -> (c): report LABELLER-FULL W2 'not run (memory)' to Theory; round 2 held" >> $L; exit 3; }; }
run r2_score python -B scratch/s219_gate.py --round 2 --stage score || exit 3
fw=$(noslots); echo "$(date '+%F %T') r2 windows without a passing primary labeller: '${fw}'" >> $L
if [ -n "$fw" ]; then
  run r2_label_fallback python -B scratch/s219_gate.py --round 2 --stage label --labeller fallback --windows "$fw" --device cuda:1 $FB && run r2_score2 python -B scratch/s219_gate.py --round 2 --stage score
  echo "$(date '+%F %T') r2 windows still without a labeller after the fallback: '$(noslots)'" >> $L
fi
run r2_pool python -B scratch/s219_build_pool.py --round 2 --stage pool --device cuda:1
echo "$(date '+%F %T') §219 round-2 retry ended (see the rc lines; STAGE_BOUNDARY: commit eval_gate.json / pool_meta.json)" >> $L
