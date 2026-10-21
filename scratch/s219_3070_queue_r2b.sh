#!/bin/bash
# The 3070 (cuda:1) queue, Audit's order (10-21 04:5x; by priority / deadline): 1 §220 zen official_self (Note 220-3) -> 2 Note 218-23 sweep R2 / R3 x1.5
# lin + xor (3070 files) -> 3 §219 round 2 (Note 219-8): B1 gates (already done 04:44, rerun is identical) -> GATE-T2 / GATE-T1' label (primary T1
# labeller; base fallback on failed windows only) -> score -> T2 pool -> 4 T1's eval_textself GPU parts (descriptive).
# Replaces scratch/s219_3070_queue_r2.sh, stopped by pid ~30 s into r2_label (nothing written; log .void_reorder).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s219_3070_queue.out
run() { local tag=$1; shift; "$@" > logs/s219_q_$tag.log 2>&1; local rc=$?; echo "$(date '+%F %T') $tag rc $rc" >> $L; return $rc; }
noslots() { python -c "
import json; e = json.load(open('boxes/runs_fsq/s219_gate_t2_s1/eval_gate.json')); print(','.join(w for w, v in e['window_labeller'].items() if v is None))"; }
for sd in 22030 22031; do run zen_shuffle_$sd python -B scratch/s220_gcs.py --model zen --device cuda:1 --official_shuffle $sd; done
run zen_official_self python -B scratch/s220_official_self.py --model zen
echo "$(date '+%F %T') s220 zen official_self rc (3070 queue): $(tail -n 1 logs/s219_q_zen_official_self.log)" >> logs/s220_stage0.out
rs() { local name=$1 arm=$2 lr=$3 steps=$4 m=$5; run $name python -B scratch/s218_gl5_known.py --device cuda:1 --arms $arm --k3_mults $m --probe_lr $lr --k3_steps $steps --out_name $name; }
rs sweep_R2_lin_x1.5_3070 k3anti 3e-4 2000 1.5
rs sweep_R2_xor_x1.5_3070 k3antixor 3e-4 2000 1.5
rs sweep_R3_lin_x1.5_3070 k3anti 1e-3 2000 1.5
rs sweep_R3_xor_x1.5_3070 k3antixor 1e-3 2000 1.5
if run r2_gates python -B scratch/s219_build_pool.py --round 2 --stage gates && run r2_label python -B scratch/s219_gate.py --round 2 --stage label --labeller primary --device cuda:1 \
   && run r2_score python -B scratch/s219_gate.py --round 2 --stage score; then
  fw=$(noslots); echo "$(date '+%F %T') r2 windows without a passing primary labeller: '${fw}'" >> $L
  if [ -n "$fw" ]; then
    run r2_label_fallback python -B scratch/s219_gate.py --round 2 --stage label --labeller fallback --windows "$fw" --device cuda:1 && run r2_score2 python -B scratch/s219_gate.py --round 2 --stage score
    echo "$(date '+%F %T') r2 windows still without a labeller after the fallback: '$(noslots)'" >> $L
  fi
  run r2_pool python -B scratch/s219_build_pool.py --round 2 --stage pool --device cuda:1
  echo "$(date '+%F %T') §219 round-2 gates + pool step ended (see the rc lines; STAGE_BOUNDARY: commit eval_gate.json / pool_meta.json)" >> $L
fi
LATENT_MEM_CAP_GB=6.0 run t1_textself_gpu python -B -m boxes.text_eval --run boxes/runs_fsq/s219_t1_s1 --part textself --round 1 --parts gpu --device cuda:1
echo "$(date '+%F %T') 3070 queue (r2b) done" >> $L
