#!/bin/bash
# Note 218-23 sweep, 3070 part (Theory 10-21): R2 (lr 3e-4) and R3 (lr 1e-3), 2,000 steps, eps x1.5, anti-lin + anti-xor each ON THIS CARD (a candidate's lin
# and xor on the same card), untrained LangSLVA, fp32, after the 3070 is released by the §219 T1 post-landing chain AND zen's Note 220-3 shuffles.
# -> g_l5_known/sweep_<R>_<kind>_x1.5_3070.json (card in meta).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out
until grep -q "s220 zen official_self rc" logs/s220_stage0.out 2>/dev/null; do sleep 60; done
run() { local name=$1 arm=$2 lr=$3 steps=$4 m=$5
  python -B scratch/s218_gl5_known.py --device cuda:1 --arms $arm --k3_mults $m --probe_lr $lr --k3_steps $steps --out_name $name > logs/s218_$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') 218-23 $name (untrained, 3070) rc $rc" >> $L; }
run sweep_R2_lin_x1.5_3070 k3anti 3e-4 2000 1.5
run sweep_R2_xor_x1.5_3070 k3antixor 3e-4 2000 1.5
run sweep_R3_lin_x1.5_3070 k3anti 1e-3 2000 1.5
run sweep_R3_xor_x1.5_3070 k3antixor 1e-3 2000 1.5
echo "$(date '+%F %T') 218-23 sweep (3070 part) done" >> $L
