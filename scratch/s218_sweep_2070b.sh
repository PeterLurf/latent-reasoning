#!/bin/bash
# Note 218-23 sweep, 2070 part (Theory 10-21: R2 / R3 move to the 3070 in parallel; R1 stays here). s218_sweep_2070.sh's bash wrapper was stopped by pid while
# its first python (sweep_R1_lin, Windows pid 11020) kept running; this waits for that JSON (or a crash), then R1-xor, then the descriptive x10 extras.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out
until grep -qE "^WROTE sweep_R1_lin_x1.5_2070|Traceback" logs/s218_sweep_R1_lin_x1.5_2070.log 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') 218-23 sweep_R1_lin_x1.5_2070 ended ($(grep -cE '^WROTE' logs/s218_sweep_R1_lin_x1.5_2070.log) WROTE line)" >> $L
run() { local name=$1 arm=$2 lr=$3 steps=$4 m=$5
  python -B scratch/s218_gl5_known.py --device cuda:2 --arms $arm --k3_mults $m --probe_lr $lr --k3_steps $steps --out_name $name > logs/s218_$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') 218-23 $name (untrained, 2070) rc $rc" >> $L; }
run sweep_R1_xor_x1.5_2070 k3antixor 1e-4 10000 1.5
run sweep_R2_xor_x10_2070 k3antixor 3e-4 2000 10
run sweep_R3_xor_x10_2070 k3antixor 1e-3 2000 10
echo "$(date '+%F %T') 218-23 sweep (2070 part) done" >> $L
