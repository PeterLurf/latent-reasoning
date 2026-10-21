#!/bin/bash
# Note 218-23 lr sweep (Audit-approved; Theory 10-21), DESCRIPTIVE, untrained LangSLVA, 2070 (cuda:2, fp32), antisymmetric plant, registered batches / z
# streams, held-out every 250. Required at eps x1.5, anti-lin AND anti-xor each: R2 lr 3e-4 2,000 steps; R3 lr 1e-3 2,000 steps; R1 lr 1e-4 10,000 steps
# (LATE = 9,500 / 9,750 / 10,000). Optional: R2 / R3 anti-xor at x10. Order (Audit 10-21, the rule's order): R1 (lin + xor), R2, R3, then the x10 extras.
# -> scratch/gate_s218/g_l5_known/sweep_<R>_<kind>_x1.5_2070.json (optional: sweep_<R>_xor_x10_2070.json). Replaces s218_antixor_lr_2070.sh (never started).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out
run() { local name=$1 arm=$2 lr=$3 steps=$4 m=$5
  python -B scratch/s218_gl5_known.py --device cuda:2 --arms $arm --k3_mults $m --probe_lr $lr --k3_steps $steps --out_name $name > logs/s218_$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') 218-23 $name (untrained, 2070) rc $rc" >> $L; }
run sweep_R1_lin_x1.5_2070 k3anti 1e-4 10000 1.5
run sweep_R1_xor_x1.5_2070 k3antixor 1e-4 10000 1.5
run sweep_R2_lin_x1.5_2070 k3anti 3e-4 2000 1.5
run sweep_R2_xor_x1.5_2070 k3antixor 3e-4 2000 1.5
run sweep_R3_lin_x1.5_2070 k3anti 1e-3 2000 1.5
run sweep_R3_xor_x1.5_2070 k3antixor 1e-3 2000 1.5
run sweep_R2_xor_x10_2070 k3antixor 3e-4 2000 10
run sweep_R3_xor_x10_2070 k3antixor 1e-3 2000 10
echo "$(date '+%F %T') 218-23 sweep (2070) done" >> $L
