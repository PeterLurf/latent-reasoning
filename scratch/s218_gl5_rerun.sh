#!/bin/bash
# §218 G-L5 re-run on the 2070 (cuda:2, fp32) after the 10-18 22:10 outage killed the first run at probe step 7500 (kept as g_l5.void_outage). Same command.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
G=scratch/gate_s218/g_l5; mkdir -p $G
python -B -m boxes.langvb_eval --untrained --cal_n 16 --out_dir $G --device cuda:2 --parts probe,fd --probe_ts 0 --plant_ts 0,1,2 --eval_bs 16 > $G/run.log 2>&1; rc=$?
echo "$(date '+%F %T') l5 rerun rc $rc" >> logs/s218_gates.out
